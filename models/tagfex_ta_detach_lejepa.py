# TA-only incremental learner where classification reads detached TA features
# and the self-supervised objective is LeJEPA.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import swanlab
from models.base import BaseLearner
from utils.inc_net import TACLSDetachNet
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 5e-4
init_weight_decay = 5e-4

epochs = 170
update_lr = 5e-4
batch_size = 128
weight_decay = 5e-4
num_workers = 16


class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        device = proj.device
        A = torch.randn(proj.size(-1), 256, device=device)
        A = A.div_(A.norm(p=2, dim=0))

        t = self.t.to(device)
        phi = self.phi.to(device)
        weights = self.weights.to(device)

        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class TagFexTADetachLeJEPA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TACLSDetachNet(args, False)
        self.sig_reg = SIGReg(knots=17)

        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            swanlab.init(
                project="PyCIL_TagFex",
                experiment_name="TagFex_TA_Detach_LeJEPA",
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
        self.sig_reg.to(self._device)
        torch.backends.cudnn.benchmark = True

        if self._cur_task == 0:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            warmup_steps = len(train_loader)
            total_steps = len(train_loader) * init_epoch
            s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=init_lr / 1000
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[s1, s2], milestones=[warmup_steps]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=update_lr,
                weight_decay=weight_decay,
            )
            warmup_steps = len(train_loader)
            total_steps = len(train_loader) * epochs
            s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[s1, s2], milestones=[warmup_steps]
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

        V_dim = self.args.get("num_views", 8)
        lamb = self.args.get("lejepa_lambda", 0.05)
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            self.train()
            losses, correct, total = 0.0, 0, 0
            for _, data in enumerate(train_loader):
                vs = data[1].to(self._device)
                targets = data[-1].to(self._device)
                N = vs.shape[0]

                out = self._network(vs.flatten(0, 1))
                logits = out["logits"]
                embedding = out["embedding"]

                proj = embedding.reshape(N, V_dim, -1)
                proj_mean = proj.mean(1, keepdim=True)
                inv_loss = (proj_mean - proj).square().mean()
                sigreg_loss = self.sig_reg(embedding)
                lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

                y_rep = targets.repeat_interleave(V_dim)
                ce_loss = F.cross_entropy(logits, y_rep)
                loss = ce_loss + lejepa_loss * self.args.get("contrast_factor", 1.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(y_rep).sum().item()
                correct += batch_correct
                total += y_rep.size(0)
                losses += loss.item()

                if local_rank <= 0:
                    batch_acc = np.around(batch_correct * 100 / y_rep.size(0), decimals=2)
                    swanlab.log({
                        "init/total_loss": loss.item(),
                        "init/ce_loss": ce_loss.item(),
                        "init/lejepa_loss": lejepa_loss.item(),
                        "init/Prediction_Invariance_loss": inv_loss.item(),
                        "init/SIGReg_loss": sigreg_loss.item(),
                        "init/batch_acc": batch_acc,
                        "init/lr": optimizer.param_groups[0]["lr"],
                    }, step=batch_step)
                    batch_step += 1

            train_acc = np.around(correct * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc
            )
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc
                )
            prog_bar.set_description(info)

        if local_rank <= 0:
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(epochs), disable=disable_tqdm)

        V_dim = self.args.get("num_views", 8)
        lamb = self.args.get("lejepa_lambda", 0.05)
        task_prefix = f"Task_{self._cur_task}"
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            self.train()
            losses, losses_clf, correct, total = 0.0, 0.0, 0, 0
            for _, data in enumerate(train_loader):
                vs = data[1].to(self._device)
                targets = data[-1].to(self._device)
                N = vs.shape[0]

                outputs = self._network(vs.flatten(0, 1))
                logits = outputs["logits"]
                embedding = outputs["embedding"]
                predicted_feature = outputs["predicted_feature"]

                proj = embedding.reshape(N, V_dim, -1)
                proj_mean = proj.mean(1, keepdim=True)
                inv_loss = (proj_mean - proj).square().mean()
                sigreg_loss = self.sig_reg(embedding)
                lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

                y_rep = targets.repeat_interleave(V_dim)
                loss_clf = F.cross_entropy(logits, y_rep)
                old_ta_feature = self.last_ta_net(vs.flatten(0, 1).contiguous())["features"]
                kd_loss = infoNCE_distill_loss(
                    self.last_projector(predicted_feature),
                    self.last_projector(old_ta_feature),
                    self.args["infonce_kd_temp"],
                )

                auto_kd_factor = self._known_classes / self._total_classes
                loss = loss_clf + self.args["contrast_factor"] * (
                    lejepa_loss * (1 - auto_kd_factor)
                    + self.args["contrast_kd_factor"] * kd_loss * auto_kd_factor
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(y_rep).sum().item()
                correct += batch_correct
                total += y_rep.size(0)

                if local_rank <= 0:
                    batch_acc = np.around(batch_correct * 100 / y_rep.size(0), decimals=2)
                    swanlab.log({
                        f"{task_prefix}/total_loss": loss.item(),
                        f"{task_prefix}/clf_loss": loss_clf.item(),
                        f"{task_prefix}/kd_loss": kd_loss.item(),
                        f"{task_prefix}/lejepa_loss": lejepa_loss.item(),
                        f"{task_prefix}/Prediction_Invariance_loss": inv_loss.item(),
                        f"{task_prefix}/SIGReg_loss": sigreg_loss.item(),
                        f"{task_prefix}/train_acc": batch_acc,
                        f"{task_prefix}/lr": optimizer.param_groups[0]["lr"],
                        f"{task_prefix}/epoch": epoch,
                    }, step=batch_step)
                    batch_step += 1

            train_acc = np.around(correct * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), losses_clf / len(train_loader), train_acc
            )
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
            prog_bar.set_description(info)

        if local_rank <= 0:
            logging.info(info)


def infoNCE_distill_loss(p_feats, z_feats, t):
    p_feats = F.normalize(p_feats, dim=-1)
    z_feats = F.normalize(z_feats, dim=-1)
    cos_sim = torch.matmul(p_feats, z_feats.T) / t
    labels = torch.arange(p_feats.size(0), device=p_feats.device)
    return F.cross_entropy(cos_sim, labels)
