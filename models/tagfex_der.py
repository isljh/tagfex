import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet_TagFex
from utils.toolkit import count_parameters, tensor2numpy
import swanlab

EPSILON = 1e-8

init_epoch = 200
init_lr = 5e-4
init_weight_decay = 5e-4

epochs = 170
update_lr = 5e-4
batch_size = 128
weight_decay = 5e-4
num_workers = 16


class TagFex(BaseLearner):
    def __init__(self, args):
        print("TagFex_DER 初始化开始")
        super().__init__(args)
        self._network = DERNet_TagFex(args, False)

        # --- SwanLab 初始化 ---
        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            swanlab.init(
                project="PyCIL_TagFex",
                experiment_name=f"TagFex_DER",
                config=self.args,
                suffix="timestamp"
            )

    def after_task(self):
        self._known_classes = self._total_classes
        if self.args.get("local_rank", 0) <= 0:
            logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        print("incremental_train 开始")
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        ptr = self._network.module if hasattr(self._network, 'module') else self._network
        ptr.update_fc(self._total_classes)

        local_rank = self.args.get("local_rank", 0)
        is_distributed = self.args.get("is_distributed", False)

        if is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                current_batch_size = batch_size // world_size
                if local_rank <= 0:
                    logging.info(
                        f"DDP Mode: Total batch size {batch_size} split into {world_size} GPUs. Local batch size: {current_batch_size}")
            else:
                current_batch_size = batch_size
        else:
            current_batch_size = batch_size

        if local_rank <= 0:
            logging.info(
                "Learning on {}-{}".format(self._known_classes, self._total_classes)
            )

        # 冻结旧参数
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in ptr.convnets[i].parameters():
                    p.requires_grad = False

        if local_rank <= 0:
            logging.info("All params: {}".format(count_parameters(self._network)))
            logging.info("Trainable params: {}".format(count_parameters(self._network, True)))

        # 数据准备
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        print("开始创建 DataLoader")

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_distributed else None
        self.train_loader = DataLoader(
            train_dataset, batch_size=current_batch_size,
            shuffle=(train_sampler is None), num_workers=num_workers,
            pin_memory=True, sampler=train_sampler, drop_last=True
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
        )

        # 包装 DDP
        if is_distributed:
            self._network.to(self._device)
            if not hasattr(self._network, 'module'):
                self._network = torch.nn.parallel.DistributedDataParallel(
                    self._network, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True
                )
        elif len(self._multiple_gpus) > 1:
            if not hasattr(self._network, 'module'):
                self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if hasattr(self._network, 'module'):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            self._network = self._network.module
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def train(self):
        self._network.train()
        ptr = self._network.module if hasattr(self._network, 'module') else self._network
        ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        torch.backends.cudnn.benchmark = True

        trainable_params = filter(lambda p: p.requires_grad, self._network.parameters())

        if self._cur_task == 0:
            optimizer = torch.optim.AdamW(trainable_params, lr=init_lr, weight_decay=init_weight_decay)
            warmup_steps = len(train_loader)
            total_steps = len(train_loader) * init_epoch
            s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                                             eta_min=init_lr / 1000)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            torch.cuda.empty_cache()
            trainable_params = filter(lambda p: p.requires_grad, self._network.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=update_lr, weight_decay=weight_decay)
            warmup_steps = len(train_loader)
            total_steps = len(train_loader) * epochs
            s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

            ptr = self._network.module if hasattr(self._network, 'module') else self._network
            ptr.weight_align(self._total_classes - self._known_classes)
            torch.cuda.empty_cache()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(init_epoch), disable=disable_tqdm)

        V_dim = self.args.get('num_views', 8)
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, correct, total = 0.0, 0, 0
            for i, data in enumerate(train_loader):
                vs = data[1].to(self._device)   # [N, V, C, H, W]
                targets = data[-1].to(self._device)

                inputs = vs.flatten(0, 1)        # [N*V, C, H, W]
                out = self._network(inputs)
                logits = out["logits"]

                y_rep = targets.repeat_interleave(V_dim)
                loss = F.cross_entropy(logits, y_rep)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(y_rep).sum().item()
                batch_total = y_rep.size(0)
                batch_acc = batch_correct * 100 / batch_total

                if local_rank <= 0:
                    swanlab.log({
                        "init/total_loss": loss.item(),
                        "init/batch_acc": batch_acc,
                        "init/lr": optimizer.param_groups[0]['lr'],
                    }, step=batch_step)
                    batch_step += 1

                correct += batch_correct
                total += batch_total
                losses += loss.item()

            if not disable_tqdm:
                train_acc = np.around(correct * 100 / total, decimals=2)
                prog_bar.set_description(
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{init_epoch} Loss {losses / len(train_loader):.3f}, Acc {train_acc:.2f}")

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(epochs), disable=disable_tqdm)

        V_dim = self.args.get('num_views', 8)
        task_prefix = f"Task_{self._cur_task}"
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, losses_clf, losses_aux, correct, total = 0.0, 0.0, 0.0, 0, 0
            for i, data in enumerate(train_loader):
                vs = data[1].to(self._device)   # [N, V, C, H, W]
                targets = data[-1].to(self._device)

                inputs = vs.flatten(0, 1)        # [N*V, C, H, W]
                outputs = self._network(inputs)
                logits = outputs["logits"]
                aux_logits = outputs["aux_logits"]

                y_rep = targets.repeat_interleave(V_dim)
                loss_clf = F.cross_entropy(logits, y_rep)

                aux_targets = y_rep.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    torch.zeros_like(aux_targets)
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)

                loss = loss_clf + self.args['aux_factor'] * loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_aux += loss_aux.item()
                _, preds = torch.max(logits, dim=1)
                batch_acc = preds.eq(y_rep).sum().item() * 100.0 / y_rep.size(0)
                correct += preds.eq(y_rep).cpu().sum()
                total += y_rep.size(0)

                if local_rank <= 0:
                    swanlab.log({
                        f"{task_prefix}/total_loss": loss.item(),
                        f"{task_prefix}/train_acc": batch_acc,
                        f"{task_prefix}/clf_loss": loss_clf.item(),
                        f"{task_prefix}/aux_loss": loss_aux.item(),
                        f"{task_prefix}/lr": optimizer.param_groups[0]['lr'],
                        f"{task_prefix}/epoch": epoch,
                    }, step=batch_step)
                    batch_step += 1

            if not disable_tqdm:
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                prog_bar.set_description(
                    f"Task {self._cur_task} Epoch {epoch + 1}/{epochs} Loss {losses / len(train_loader):.3f} "
                    f"Loss_clf {losses_clf / len(train_loader):.3f} "
                    f"Loss_aux {losses_aux / len(train_loader):.3f} Acc {train_acc:.2f}")

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)