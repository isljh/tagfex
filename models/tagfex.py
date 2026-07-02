# Please note that only "cifar100_aa" and "cifar10_aa" are supported for TagFex in PyCIL.
# For large datasets like ImageNet, please refer to the offical code repo https://github.com/bwnzheng/TagFex_CVPR2025.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.utils.data import DataLoader
import swanlab
from models.base import BaseLearner
from utils.inc_net import TagFexNet
from utils.toolkit import count_parameters, tensor2numpy

#ε
EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005
momentum = 0.9

epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 64
weight_decay = 2e-4
num_workers = 8
T = 2


class TagFex(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TagFexNet(args, False)
        self._global_step = 0
        self.linear_probe = None
        self.linear_probe_optimizer = None

        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            swanlab.init(
                project="PyCIL_TagFex_Ablation",
                experiment_name="{}_{}_{}".format(
                    self.args.get("prefix", "run"),
                    self.args.get("dataset", "dataset"),
                    self.args.get("model_name", "tagfex"),
                ),
                config=self.args,
                suffix="timestamp",
            )

    def _init_task0_online_probe(self, feature_dim, num_classes):
        if not self.args.get("online_linear_probe", False):
            return
        if self._cur_task != 0 or self.linear_probe is not None:
            return
        if num_classes <= 0:
            raise ValueError("online_linear_probe requires a positive Task0 class count.")

        self.linear_probe = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        ).to(self._device)
        self.linear_probe_optimizer = torch.optim.AdamW(
            self.linear_probe.parameters(),
            lr=self.args.get("probe_lr", 1e-3),
            weight_decay=self.args.get("probe_weight_decay", 1e-7),
        )

    def _update_task0_online_probe(self, features, targets):
        if not self.args.get("online_linear_probe", False):
            return {}
        if self._cur_task != 0:
            return {}
        if features.size(0) != targets.size(0):
            raise ValueError(
                "online_linear_probe feature/target mismatch: {} vs {}".format(
                    features.size(0), targets.size(0)
                )
            )

        if self.linear_probe is None:
            self._init_task0_online_probe(features.size(1), int(self._total_classes))

        self.linear_probe.train()
        probe_features = features.detach().float()
        probe_targets = targets.detach().long()
        logits = self.linear_probe(probe_features)
        probe_loss = F.cross_entropy(logits, probe_targets)

        self.linear_probe_optimizer.zero_grad()
        probe_loss.backward()
        self.linear_probe_optimizer.step()

        probe_top1 = (logits.argmax(dim=1) == probe_targets).float().mean().item() * 100
        return {
            "probe_loss": probe_loss.item(),
            "probe_top1": probe_top1,
        }

    def get_online_probe_state(self):
        if self.linear_probe is None:
            return None
        linear = self.linear_probe[-1]
        return {
            "enabled": bool(self.args.get("online_linear_probe", False)),
            "probe_feature": self.args.get("probe_feature", "embedding"),
            "task_id": int(self._cur_task),
            "feature_dim": int(linear.in_features),
            "num_classes": int(linear.out_features),
            "probe_state_dict": self.linear_probe.state_dict(),
            "probe_optimizer_state_dict": (
                self.linear_probe_optimizer.state_dict()
                if self.linear_probe_optimizer is not None
                else None
            ),
        }

    def load_online_probe_state(self, state):
        if not state:
            return False
        self._init_task0_online_probe(
            int(state["feature_dim"]),
            int(state["num_classes"]),
        )
        self.linear_probe.load_state_dict(state["probe_state_dict"])
        optimizer_state = state.get("probe_optimizer_state_dict")
        if optimizer_state is not None and self.linear_probe_optimizer is not None:
            self.linear_probe_optimizer.load_state_dict(optimizer_state)
        return True

    def after_task(self):
        self._known_classes = self._total_classes
        ptr = self._network.module if hasattr(self._network, "module") else self._network
        self.last_ta_net = ptr.get_freezed_copy_ta()
        self.last_projector = ptr.get_freezed_copy_projector()
        if self.args.get("local_rank", 0) <= 0:
            logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        ptr = self._network.module if hasattr(self._network, "module") else self._network
        ptr.update_fc(self._total_classes)

        local_rank = self.args.get("local_rank", 0)
        is_distributed = self.args.get("is_distributed", False)

        if is_distributed and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            current_batch_size = batch_size // world_size
            if local_rank <= 0:
                logging.info(
                    "DDP Mode: Total batch size {} split into {} GPUs. Local batch size: {}".format(
                        batch_size, world_size, current_batch_size
                    )
                )
        else:
            current_batch_size = batch_size

        if local_rank <= 0:
            logging.info(
                "Learning on {}-{}".format(self._known_classes, self._total_classes)
            )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in ptr.convnets[i].parameters():
                    p.requires_grad = False

        if local_rank <= 0:
            logging.info("All params: {}".format(count_parameters(self._network)))
            logging.info(
                "Trainable params: {}".format(count_parameters(self._network, True))
            )

        #最终数据 = 新类数据 + 旧类memory 
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if is_distributed
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        if is_distributed:
            self._network.to(self._device)
            if not hasattr(self._network, "module"):
                self._network = torch.nn.parallel.DistributedDataParallel(
                    self._network,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True,
                )
        elif len(self._multiple_gpus) > 1:
            if not hasattr(self._network, "module"):
                self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        #这里的样本是怎么选的还没看------------------------------------------
        if hasattr(self._network, "module"):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            self._network = self._network.module
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def train(self):
        self._network.train()
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network

        if self.args.get("freeze_ta_after_task0", False) and self._cur_task > 0:
            self._network_module_ptr.ta_net.eval()
            for p in self._network_module_ptr.ta_net.parameters():
                p.requires_grad = False

        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        #让 cuDNN 在首次运行时自动搜索最优卷积算法
        torch.backends.cudnn.benchmark = True
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            #lr衰减策略
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            new_task_size = self._total_classes - self._known_classes
            if new_task_size > 0:
                if hasattr(self._network, "module"):
                    self._network.module.weight_align(new_task_size)
                else:
                    self._network.weight_align(new_task_size)

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

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = local_rank > 0
        prog_bar = tqdm(range(init_epoch), disable=disable_tqdm, dynamic_ncols=True)

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                out = self._network(inputs)
                logits = out["logits"]
                embedding = out["embedding"]

                ce_loss = F.cross_entropy(logits, targets)
                infonce_loss = infoNCE_loss(embedding, self.args['infonce_temp'])
                loss = ce_loss + infonce_loss * self.args['contrast_factor']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                probe_log = self._update_task0_online_probe(embedding, targets)
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
                correct += batch_correct
                total += len(targets)

                if local_rank <= 0:
                    batch_acc = np.around(
                        tensor2numpy(batch_correct) * 100 / len(targets), decimals=2
                    )
                    log_payload = {
                        "init/total_loss": loss.item(),
                        "init/ce_loss": ce_loss.item(),
                        "init/infonce_loss": infonce_loss.item(),
                        "init/batch_acc": batch_acc,
                        "init/epoch": epoch,
                    }
                    if probe_log:
                        log_payload.update({
                            "init/probe_loss": probe_log["probe_loss"],
                            "init/probe_top1": probe_log["probe_top1"],
                        })
                    swanlab.log(log_payload, step=self._global_step)
                    self._global_step += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                if local_rank <= 0:
                    swanlab.log(
                        {
                            "init/epoch_train_acc": train_acc,
                            "init/epoch_test_acc": test_acc,                  
                        },
                        step=self._global_step,
                    )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                if local_rank <= 0:
                    swanlab.log(
                        {
                            "init/epoch_train_acc": train_acc,
                            "init/epoch": epoch,
                        },
                        step=self._global_step,
                    )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        if local_rank <= 0:
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = local_rank > 0
        prog_bar = tqdm(range(epochs), disable=disable_tqdm, dynamic_ncols=True)
        task_prefix = f"task_{self._cur_task}"
        freeze_ta_losses = self.args.get("freeze_ta_after_task0", False) and self._cur_task > 0
        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            losses_infonce = 0.0
            losses_kd = 0.0
            losses_trans_cls = 0.0
            losses_transfer = 0.0
            correct, total = 0, 0
            for i, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)
                
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                embedding = outputs['embedding']

                if freeze_ta_losses:
                    infonce_loss = torch.tensor(0.0, device=self._device)
                else:
                    infonce_loss = infoNCE_loss(embedding, self.args['infonce_temp'])
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                if freeze_ta_losses:
                    kd_loss = torch.tensor(0.0, device=self._device)
                else:
                    predicted_feature = outputs['predicted_feature']
                    with torch.no_grad():
                        old_ta_feature = self.last_ta_net(inputs.contiguous())['features']
                    kd_loss = infoNCE_distill_loss(self.last_projector(predicted_feature), self.last_projector(old_ta_feature), self.args['infonce_kd_temp'])
                trans_logits = outputs["trans_logits"]
                #把batch中的新类样本挑出来
                cur_task_mask = (targets >= self._known_classes)
                if cur_task_mask.any():
                    trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask], targets[cur_task_mask] - self._known_classes)
                    #判断要不要进行迁移，左侧分类损失和右侧分类损失对比·
                    if trans_cls_loss < loss_clf:
                        T = self.args['kd_temp']
                        transfer_loss = F.kl_div((logits[cur_task_mask][:, self._known_classes:] / T).log_softmax(dim=1), (trans_logits.detach()[cur_task_mask] / T).softmax(dim=1), reduction='batchmean')
                    else:
                        transfer_loss = torch.tensor(0., device=self._device)
                else:
                    trans_cls_loss = torch.tensor(0., device=self._device)
                    transfer_loss = torch.tensor(0., device=self._device)

                auto_kd_factor = self._known_classes / self._total_classes
                loss = loss_clf + \
                self.args['aux_factor'] * loss_aux + \
                self.args['contrast_factor'] * (infonce_loss * (1 - auto_kd_factor) + self.args['contrast_kd_factor'] * kd_loss * auto_kd_factor) + \
                self.args['trans_cls_factor'] * trans_cls_loss + \
                self.args['transfer_factor'] * transfer_loss         

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_infonce += infonce_loss.item()
                losses_kd += kd_loss.item()
                losses_trans_cls += trans_cls_loss.item()
                losses_transfer += transfer_loss.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
                correct += batch_correct
                total += len(targets)

                if local_rank <= 0:
                    batch_acc = np.around(
                        tensor2numpy(batch_correct) * 100 / len(targets), decimals=2
                    )
                    swanlab.log(
                        {
                            f"{task_prefix}/total_loss": loss.item(),
                            f"{task_prefix}/clf_loss": loss_clf.item(),
                            f"{task_prefix}/aux_loss": loss_aux.item(),
                            f"{task_prefix}/infonce_loss": infonce_loss.item(),
                            f"{task_prefix}/kd_loss": kd_loss.item(),
                            f"{task_prefix}/trans_cls_loss": trans_cls_loss.item(),
                            f"{task_prefix}/transfer_loss": transfer_loss.item(),
                            f"{task_prefix}/auto_kd_factor": auto_kd_factor,
                            f"{task_prefix}/batch_acc": batch_acc,
                            f"{task_prefix}/epoch": epoch,
                        },
                        step=self._global_step,
                    )
                    self._global_step += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                if local_rank <= 0:
                    swanlab.log(
                        {
                            f"{task_prefix}/epoch_train_acc": train_acc,
                            f"{task_prefix}/epoch_test_acc": test_acc,
                        },
                        step=self._global_step,
                    )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                if local_rank <= 0:
                    swanlab.log(
                        {
                            f"{task_prefix}/epoch_train_acc": train_acc,
                            f"{task_prefix}/epoch": epoch,
                        },
                        step=self._global_step,
                    )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        if local_rank <= 0:
            logging.info(info)

def infoNCE_loss(feats, t):
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll

def infoNCE_distill_loss(p_feats, z_feats, t):
    # print(p_feats.shape, z_feats.shape)
    cos_sim = F.cosine_similarity(p_feats[:,None,:], z_feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll
