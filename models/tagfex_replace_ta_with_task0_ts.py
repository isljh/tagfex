import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import swanlab

from models.base import BaseLearner
from utils.inc_net import TagFexTask0TSReplaceNet
from utils.toolkit import count_parameters, tensor2numpy


init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8


class TagFexReplaceTAWithTask0TS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TagFexTask0TSReplaceNet(args, False, use_transfer_branch=True)
        self._global_step = 0

        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            swanlab.init(
                project="PyCIL_TagFex_Ablation",
                experiment_name="{}_{}_{}".format(
                    self.args.get("prefix", "run"),
                    self.args.get("dataset", "dataset"),
                    self.args.get("model_name", "tagfex_replace_ta_with_task0_ts"),
                ),
                config=self.args,
                suffix="timestamp",
            )

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            network_ptr = self._network.module
        else:
            network_ptr = self._network

        if network_ptr.task0_ta_substitute is not None:
            network_ptr.task0_ta_substitute.eval()

        network_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                network_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        torch.backends.cudnn.benchmark = True
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=init_lr,
                momentum=0.9,
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
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        prog_bar = tqdm(range(init_epoch), disable=local_rank > 0, dynamic_ncols=True)

        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1 = inputs1.to(self._device)
                inputs2 = inputs2.to(self._device)
                targets = targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                outputs = self._network(inputs)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
                correct += batch_correct
                total += len(targets)

                if local_rank <= 0:
                    batch_acc = np.around(tensor2numpy(batch_correct) * 100 / len(targets), decimals=2)
                    swanlab.log(
                        {
                            "init/total_loss": loss.item(),
                            "init/batch_acc": batch_acc,
                            "init/epoch": epoch,
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
                            "init/epoch_train_acc": train_acc,
                            "init/epoch_test_acc": test_acc,
                        },
                        step=self._global_step,
                    )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc
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
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc
                )
            prog_bar.set_description(info)

        if local_rank <= 0:
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        prog_bar = tqdm(range(epochs), disable=local_rank > 0, dynamic_ncols=True)
        task_prefix = f"task_{self._cur_task}"

        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            losses_trans_cls = 0.0
            losses_transfer = 0.0
            correct, total = 0, 0

            for _, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1 = inputs1.to(self._device)
                inputs2 = inputs2.to(self._device)
                targets = targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                trans_logits = outputs["trans_logits"]

                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)

                cur_task_mask = targets >= self._known_classes
                trans_cls_loss = F.cross_entropy(
                    trans_logits[cur_task_mask],
                    targets[cur_task_mask] - self._known_classes,
                )
                if trans_cls_loss < loss_clf:
                    T = self.args["kd_temp"]
                    transfer_loss = F.kl_div(
                        (logits[cur_task_mask][:, self._known_classes:] / T).log_softmax(dim=1),
                        (trans_logits.detach()[cur_task_mask] / T).softmax(dim=1),
                        reduction="batchmean",
                    )
                else:
                    transfer_loss = torch.tensor(0.0, device=self._device)

                loss = (
                    loss_clf
                    + self.args["aux_factor"] * loss_aux
                    + self.args["trans_cls_factor"] * trans_cls_loss
                    + self.args["transfer_factor"] * transfer_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_aux += loss_aux.item()
                losses_trans_cls += trans_cls_loss.item()
                losses_transfer += transfer_loss.item()

                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
                correct += batch_correct
                total += len(targets)

                if local_rank <= 0:
                    batch_acc = np.around(tensor2numpy(batch_correct) * 100 / len(targets), decimals=2)
                    swanlab.log(
                        {
                            f"{task_prefix}/total_loss": loss.item(),
                            f"{task_prefix}/clf_loss": loss_clf.item(),
                            f"{task_prefix}/aux_loss": loss_aux.item(),
                            f"{task_prefix}/trans_cls_loss": trans_cls_loss.item(),
                            f"{task_prefix}/transfer_loss": transfer_loss.item(),
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
