# TA-only incremental learner where classification reads detached TA features.
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
from utils.inc_net import TACLSDetachNet
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8

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


class TagFexTADetach(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TACLSDetachNet(args, False)

        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            swanlab.init(
                project="PyCIL_TagFex",
                experiment_name="TagFex_TA_Detach",
                config=self.args,
                suffix="timestamp"
            )

    def after_task(self):
        self._known_classes = self._total_classes
        ptr = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        self.last_ta_net = ptr.get_freezed_copy_ta()
        self.last_projector = ptr.get_freezed_copy_projector()
        if self.args.get("local_rank", 0) <= 0:
            logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
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
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(init_epoch), disable=disable_tqdm)
        batch_step = 0
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

                out = self._network(inputs)
                logits = out["logits"]
                embedding = out["embedding"]

                ce_loss = F.cross_entropy(logits, targets)
                infonce_loss = infoNCE_loss(embedding, self.args["infonce_temp"])
                loss = ce_loss + infonce_loss * self.args["contrast_factor"]
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
                    swanlab.log({
                        "init/total_loss": loss.item(),
                        "init/ce_loss": ce_loss.item(),
                        "init/infonce_loss": infonce_loss.item(),
                        "init/batch_acc": batch_acc,
                        "init/lr": optimizer.param_groups[0]["lr"],
                    }, step=batch_step)
                    batch_step += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc
                )
            prog_bar.set_description(info)

        if local_rank <= 0:
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(epochs), disable=disable_tqdm)
        task_prefix = f"Task_{self._cur_task}"
        batch_step = 0
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            correct, total = 0, 0
            for _, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1 = inputs1.to(self._device)
                inputs2 = inputs2.to(self._device)
                targets = targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                outputs = self._network(inputs)
                logits = outputs["logits"]
                embedding = outputs["embedding"]
                predicted_feature = outputs["predicted_feature"]

                infonce_loss = infoNCE_loss(embedding, self.args["infonce_temp"])
                loss_clf = F.cross_entropy(logits, targets)
                old_ta_feature = self.last_ta_net(inputs.contiguous())["features"]
                kd_loss = infoNCE_distill_loss(
                    self.last_projector(predicted_feature),
                    self.last_projector(old_ta_feature),
                    self.args["infonce_kd_temp"],
                )

                auto_kd_factor = self._known_classes / self._total_classes
                loss = loss_clf + self.args["contrast_factor"] * (
                    infonce_loss * (1 - auto_kd_factor)
                    + self.args["contrast_kd_factor"] * kd_loss * auto_kd_factor
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
                correct += batch_correct
                total += len(targets)

                if local_rank <= 0:
                    batch_acc = np.around(tensor2numpy(batch_correct) * 100 / len(targets), decimals=2)
                    swanlab.log({
                        f"{task_prefix}/total_loss": loss.item(),
                        f"{task_prefix}/clf_loss": loss_clf.item(),
                        f"{task_prefix}/infonce_loss": infonce_loss.item(),
                        f"{task_prefix}/kd_loss": kd_loss.item(),
                        f"{task_prefix}/train_acc": batch_acc,
                        f"{task_prefix}/lr": optimizer.param_groups[0]["lr"],
                        f"{task_prefix}/epoch": epoch,
                    }, step=batch_step)
                    batch_step += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        if local_rank <= 0:
            logging.info(info)


def infoNCE_loss(feats, t):
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    return nll.mean()


def infoNCE_distill_loss(p_feats, z_feats, t):
    cos_sim = F.cosine_similarity(p_feats[:, None, :], z_feats[None, :, :], dim=-1)
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    return nll.mean()
