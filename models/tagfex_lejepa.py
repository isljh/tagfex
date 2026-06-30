# Please note that only "cifar100_aa" and "cifar10_aa" are supported for TagFex in PyCIL_DDP.
import csv
import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import TagFexNet
from utils.toolkit import count_parameters, tensor2numpy
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import swanlab
import torch.distributed as dist

EPSILON = 1e-8

init_epoch = 200
init_lr = 5e-4
#init_milestones = [60, 120, 170]
#init_lr_decay = 0.1
init_weight_decay = 5e-4
#momentum = 0.9

epochs = 170
update_lr = 5e-4
#milestones = [80, 120, 150]
#lrate_decay = 0.1
batch_size = 64
weight_decay = 5e-4
num_workers = 16
T = 2


class SIGReg(nn.Module):


    def __init__(self, knots=17, matrix_mode="per_batch_random"):
        super().__init__()
        # 初始化积分节点和权重，用于改进的积分近似
        # 1. 在 [0, 3] 之间切 17 个等距离的点,t就是采样点的位置
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        # 2. 设置积分权重（梯形法则）
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # 目标高斯分布的特征函数
        self.register_buffer("weights", weights * window)
        self.matrix_mode = matrix_mode
        self.fixed_A = None
        self.running_A = None
        self.running_count = 0
        self.matrix_seed = None
        self.matrix_generator = None

    def set_matrix_seed(self, seed):
        self.matrix_seed = int(seed)
        self.matrix_generator = torch.Generator()
        self.matrix_generator.manual_seed(self.matrix_seed)

    def reset_running_state(self):
        self.running_A = None
        self.running_count = 0

    def _sample_matrix(self, feat_dim, device):
        if self.matrix_generator is None:
            A = torch.randn(feat_dim, 256, device=device)
        else:
            A = torch.randn(feat_dim, 256, generator=self.matrix_generator, dtype=torch.float32).to(device)
        return A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(EPSILON))

    def _normalize_columns(self, A):
        return A / A.norm(p=2, dim=0, keepdim=True).clamp_min(EPSILON)

    def get_matrix_state(self):
        return {
            "matrix_mode": self.matrix_mode,
            "fixed_A": self.fixed_A.detach().cpu() if self.fixed_A is not None else None,
            "running_A": self.running_A.detach().cpu() if self.running_A is not None else None,
            "running_count": self.running_count,
            "matrix_seed": self.matrix_seed,
        }

    def load_matrix_state(self, state, device=None):
        if not state:
            return False

        saved_mode = state.get("matrix_mode")
        if saved_mode is not None and saved_mode != self.matrix_mode:
            logging.warning(
                "Skip loading SIGReg matrix state because checkpoint mode {} != current mode {}.".format(
                    saved_mode, self.matrix_mode
                )
            )
            return False

        device = device or self.t.device
        fixed_A = state.get("fixed_A")
        running_A = state.get("running_A")
        self.fixed_A = fixed_A.to(device) if isinstance(fixed_A, torch.Tensor) else None
        self.running_A = running_A.to(device) if isinstance(running_A, torch.Tensor) else None
        self.running_count = int(state.get("running_count", 0))
        return True

    def forward(self, proj):
        device = proj.device
        feat_dim = proj.size(-1)
        if self.matrix_mode == "per_batch_random":
            A = self._sample_matrix(feat_dim, device)
        elif self.matrix_mode == "fixed":
            if self.fixed_A is None or self.fixed_A.shape[0] != feat_dim or self.fixed_A.device != device:
                self.fixed_A = self._sample_matrix(feat_dim, device)
            A = self.fixed_A
        elif self.matrix_mode == "running_avg":
            A_new = self._sample_matrix(feat_dim, device)
            if self.running_A is None or self.running_A.shape[0] != feat_dim or self.running_A.device != device:
                self.running_A = A_new
                self.running_count = 1
            else:
                self.running_A = (self.running_count * self.running_A + A_new) / (self.running_count + 1)
                self.running_A = self._normalize_columns(self.running_A)
                self.running_count += 1
            A = self.running_A
        else:
            raise ValueError("Unknown sigreg_matrix_mode: {}".format(self.matrix_mode))

        t = self.t.to(device)
        phi = self.phi.to(device)
        weights = self.weights.to(device)

        # 2. 计算特征函数并与标准高斯分布对比
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()

        # 3. 计算统计量
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()

class TagFex(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TagFexNet(args, False)
        """
        {
            'ta_feature': ta_feature,                      ta分支得到的特征
            'embedding': embedding,                        自监督嵌入,ta分支得到的特征经过projector得到的嵌入
            'trans_logits': trans_logits,                  融合后的特征进行分类
            'predicted_feature': predicted_feature,        服务于 知识蒸馏,根据当前的 ta_feature 去“预测”旧模型提取出来的特征
            'features': features                           所有任务特定专家提取特征的总和
            'logits': logits                               合并特征分类
            'aux_logits':aux_logits                        辅助分支分类（新旧类别）
        }
        """
        # --- 实例化 SIGReg ---
        self.sig_reg = SIGReg(
            knots=17,
            matrix_mode=self.args.get("sigreg_matrix_mode", "per_batch_random"),
        )
        self._sigreg_task_matrix_states = {}
        self._pending_rehearsal_memory_build = False
        self._final_sigreg_last_state = None
        self.linear_probe = None
        self.linear_probe_optimizer = None

        # --- SwanLab 初始化 ---
        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            # 每个任务开始时，初始化或更新实验记录
            swanlab.init(
                project=self.args.get("swanlab_project", "PyCIL_TagFex"),
                experiment_name=self.args.get(
                    "swanlab_experiment_name", self.args.get("prefix", "TagFex")
                ),
                config=self.args,  # 自动记录所有传入的 args
                suffix="timestamp"  # 防止重名
            )
        # ---------------------

    def _sigreg_matrix_seed_for_task(self, task_id):
        base_seed = int(self.args.get("sigreg_matrix_seed", self.args.get("seed", 0)))
        if self.sig_reg.matrix_mode == "fixed":
            return base_seed
        return base_seed + int(task_id) * 1000003

    def _prepare_sigreg_for_task(self, local_rank=0):
        matrix_seed = self._sigreg_matrix_seed_for_task(self._cur_task)
        self.sig_reg.set_matrix_seed(matrix_seed)
        if self.sig_reg.matrix_mode == "running_avg":
            self.sig_reg.reset_running_state()
            if local_rank <= 0:
                logging.info(
                    "SIGReg running_avg matrix reset for task {} with seed {}.".format(
                        self._cur_task, matrix_seed
                    )
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
            weight_decay=self.args.get("probe_weight_decay", 1e-6),
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
            "probe_feature": self.args.get("probe_feature", "ta_feature"),
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

    def _record_sigreg_task_state(self, state):
        if state.get("matrix_mode") != "running_avg" or state.get("running_A") is None:
            return

        task_id = int(state.get("task_id", self._cur_task))
        self._sigreg_task_matrix_states[str(task_id)] = {
            "matrix_mode": state["matrix_mode"],
            "task_id": task_id,
            "running_A": state["running_A"],
            "running_count": state["running_count"],
            "matrix_seed": state.get("matrix_seed"),
        }

    def get_sigreg_state(self):
        current_state = self.sig_reg.get_matrix_state()
        current_state["task_id"] = self._cur_task
        self._record_sigreg_task_state(current_state)

        return {
            "current": current_state,
            "task_matrix_states": self._sigreg_task_matrix_states,
        }

    def load_sigreg_state(self, state):
        if not state:
            return False

        if "current" in state:
            self._sigreg_task_matrix_states = state.get("task_matrix_states", {})
            state = state["current"]

        state.setdefault("task_id", self._cur_task)
        self._record_sigreg_task_state(state)
        return self.sig_reg.load_matrix_state(state, self._device)

    def after_task(self):
        self._known_classes = self._total_classes
        # 优化点：使用 hasattr 自动探测 DDP 状态
        ptr = self._network.module if hasattr(self._network, 'module') else self._network
        self.last_ta_net = ptr.get_freezed_copy_ta()
        self.last_projector = ptr.get_freezed_copy_projector()

        if self.args.get("local_rank", 0) <= 0:
            logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # 优化点：更新 FC 层
        ptr = self._network.module if hasattr(self._network, 'module') else self._network
        ptr.update_fc(self._total_classes)

        local_rank = self.args.get("local_rank", 0)
        is_distributed = self.args.get("is_distributed", False)
        self._prepare_sigreg_for_task(local_rank)

        # --- 新增：自动计算每个 GPU 的 batch_size ---
        if is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                # 自动除以显卡数量，保持总 batch_size = 128
                current_batch_size = batch_size // world_size
                if local_rank <= 0:
                    logging.info(
                        f"DDP Mode: Total batch size {batch_size} split into {world_size} GPUs. Local batch size: {current_batch_size}")
            else:
                current_batch_size = batch_size
        else:
            current_batch_size = batch_size
        # ------------------------------------------

        if local_rank <= 0:
            logging.info(
                "Learning on {}-{}".format(self._known_classes, self._total_classes)
            )

        # 冻结旧参数
        if self._cur_task > 0:
            # 这里的 ptr 已经在上面获取过了，直接使用即可
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

        if self.args.get("class_mean_fc_init", False):
            init_loader = DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
            self._init_fc_with_class_means(init_loader)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_distributed else None
        self.train_loader = DataLoader(
            train_dataset, batch_size=current_batch_size,
            shuffle=(train_sampler is None), num_workers=num_workers,
            pin_memory=True, sampler=train_sampler,drop_last=True
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,drop_last=False
        )

        # 包装 DDP (仅当尚未包装时)
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

        if self.args.get("final_sigreg", False):
            if hasattr(self._network, "module"):
                self._network = self._network.module
            self._pending_rehearsal_memory_build = True
            return

        self.complete_incremental_train(data_manager)

    def has_pending_final_sigreg(self):
        return bool(self.args.get("final_sigreg", False) and self._pending_rehearsal_memory_build)

    def complete_incremental_train(self, data_manager):
        if hasattr(self._network, 'module'):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            self._network = self._network.module  # 解包，回到 TagFexNet 原始类
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._pending_rehearsal_memory_build = False

    def train(self):
        self._network.train()
        # 统一处理 DDP 或单卡指针
        ptr = self._network.module if hasattr(self._network, 'module') else self._network
        ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                ptr.convnets[i].eval()


    def _record_accuracy_curves_enabled(self):
        return bool(self.args.get("record_accuracy_curves", False)) and self.args.get("local_rank", 0) <= 0

    def _compute_sigreg_loss(self, embedding, num_items, num_views):
        view_mode = self.args.get("sigreg_view_mode", "mixed")
        if view_mode == "mixed":
            return self.sig_reg(embedding)

        proj = embedding.reshape(num_items, num_views, -1)
        if view_mode == "view_wise":
            losses = [self.sig_reg(proj[:, view_idx, :]) for view_idx in range(num_views)]
            return torch.stack(losses).mean()

        raise ValueError("Unknown sigreg_view_mode: {}".format(view_mode))

    def _compute_selfsup_losses(self, embedding, num_items, num_views, lamb):
        selfsup_mode = self.args.get("selfsup_mode", "lejepa")
        zero = embedding.new_zeros(())
        if selfsup_mode == "none":
            return zero, zero, zero

        proj = embedding.reshape(num_items, num_views, -1)
        proj_mean = proj.mean(1, keepdim=True)

        if selfsup_mode == "lejepa":
            inv_loss = (proj_mean - proj).square().mean()
            sigreg_loss = self._compute_sigreg_loss(embedding, num_items, num_views)
            selfsup_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
            return inv_loss, sigreg_loss, selfsup_loss

        if selfsup_mode == "inv_only":
            inv_loss = (proj_mean - proj).square().mean()
            return inv_loss, zero, inv_loss

        if selfsup_mode == "sigreg_only":
            sigreg_loss = self._compute_sigreg_loss(embedding, num_items, num_views)
            return zero, sigreg_loss, sigreg_loss

        raise ValueError("Unknown selfsup_mode: {}".format(selfsup_mode))

    def _num_ts_views(self, num_views):
        return min(int(self.args.get("num_ts_views", 2)), num_views)

    def _select_ts_views(self, tensor, num_items, num_views, selected_views):
        view_shape = (num_items, num_views) + tuple(tensor.shape[1:])
        out_shape = (num_items * selected_views,) + tuple(tensor.shape[1:])
        return tensor.reshape(view_shape)[:, :selected_views].reshape(out_shape)

    def _si_blurry_curve_class_sets(self):
        si_blurry_groups = self.args.get("si_blurry_eval_groups")
        if not si_blurry_groups:
            return set(), set()

        session_by_task = si_blurry_groups.get("session", [])
        disjoint_by_task = si_blurry_groups.get("disjoint", [])
        if len(session_by_task) == 0 or self._cur_task < 0:
            return set(), set()

        upto_task = min(self._cur_task, len(session_by_task) - 1)
        exposed_classes = set()
        for task_classes in session_by_task[: upto_task + 1]:
            exposed_classes.update(int(class_idx) for class_idx in task_classes)
        exposed_classes = {
            class_idx for class_idx in exposed_classes if class_idx < self._total_classes
        }

        disjoint_classes = set()
        for task_classes in disjoint_by_task[: upto_task + 1]:
            disjoint_classes.update(int(class_idx) for class_idx in task_classes)
        disjoint_classes = {
            class_idx
            for class_idx in disjoint_classes
            if class_idx in exposed_classes and class_idx < self._total_classes
        }
        blurry_classes = exposed_classes - disjoint_classes
        return disjoint_classes, blurry_classes

    def _new_accuracy_counts(self):
        return {
            "disjoint_only_correct": 0,
            "disjoint_only_count": 0,
            "blurry_exposed_correct": 0,
            "blurry_exposed_count": 0,
        }

    def _update_accuracy_counts(self, counts, preds, targets):
        disjoint_classes, blurry_classes = self._si_blurry_curve_class_sets()
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        correct = preds == targets

        def update_group(prefix, class_set):
            if len(class_set) == 0:
                return
            mask = np.isin(targets, list(class_set))
            if not mask.any():
                return
            counts[f"{prefix}_correct"] += int(correct[mask].sum())
            counts[f"{prefix}_count"] += int(mask.sum())

        update_group("disjoint_only", disjoint_classes)
        update_group("blurry_exposed", blurry_classes)

    def _accuracy_counts_to_metrics(self, counts):
        def acc(prefix):
            total = counts[f"{prefix}_count"]
            if total == 0:
                return None
            correct = counts[f"{prefix}_correct"]
            return float(np.around(correct * 100.0 / total, decimals=2))

        return {
            "disjoint_only_acc": acc("disjoint_only"),
            "blurry_exposed_acc": acc("blurry_exposed"),
        }

    def _append_accuracy_curve_row(self, row):
        diagnostics_dir = self.args.get("diagnostics_dir")
        if not diagnostics_dir:
            return
        os.makedirs(diagnostics_dir, exist_ok=True)
        csv_path = os.path.join(
            diagnostics_dir,
            "accuracy_curves_task_{}.csv".format(self._cur_task),
        )
        fieldnames = [
            "task_id",
            "phase",
            "epoch",
            "split",
            "disjoint_only_acc",
            "blurry_exposed_acc",
            "loss",
            "lr",
            "known_classes",
            "total_classes",
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _record_epoch_accuracy_diagnostics(
        self,
        phase,
        epoch,
        total_epochs,
        train_counts,
        avg_loss,
        lr,
        test_loader,
    ):
        if not self._record_accuracy_curves_enabled() or train_counts is None:
            return

        train_metrics = self._accuracy_counts_to_metrics(train_counts)
        row = {
            "task_id": self._cur_task,
            "phase": phase,
            "epoch": epoch + 1,
            "split": "train",
            "loss": float(avg_loss),
            "lr": float(lr),
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
        }
        row.update(train_metrics)
        self._append_accuracy_curve_row(row)

        metric_prefix = f"Task_{self._cur_task}/Diagnostics/{phase}"
        log_payload = {
            f"{metric_prefix}/train_disjoint_only_acc": train_metrics["disjoint_only_acc"],
            f"{metric_prefix}/train_blurry_exposed_acc": train_metrics["blurry_exposed_acc"],
        }
        log_payload = {key: value for key, value in log_payload.items() if value is not None}
        if log_payload:
            swanlab.log(log_payload, step=epoch + 1)

    def _flatten_augmented_inputs(self, inputs, targets):
        if inputs.ndim == 5:
            num_views = inputs.shape[1]
            return inputs.flatten(0, 1), targets.repeat_interleave(num_views)
        return inputs, targets

    def _compute_class_mean_fc_weights(self, features, targets, reference_weight, scale_mode):
        norm_features = features.float() / features.float().norm(p=2, dim=1, keepdim=True).clamp_min(EPSILON)
        num_classes = int(self._total_classes)
        class_means = torch.zeros(num_classes, features.shape[1], dtype=torch.float32)
        counts = torch.zeros(num_classes, dtype=torch.long)
        missing = []

        for class_idx in range(num_classes):
            mask = targets == class_idx
            counts[class_idx] = int(mask.sum())
            if counts[class_idx] == 0:
                missing.append(class_idx)
                continue
            mean = norm_features[mask].mean(dim=0)
            class_means[class_idx] = mean / mean.norm(p=2).clamp_min(EPSILON)

        if missing:
            raise ValueError("No samples for class_mean_fc_init classes: {}".format(missing))

        if scale_mode == "none":
            return class_means, counts
        if scale_mode == "global_weight_norm":
            scale = reference_weight.detach().float().norm(p=2, dim=1).mean()
            return class_means * scale, counts
        if scale_mode == "per_class_weight_norm":
            scales = reference_weight.detach().float().norm(p=2, dim=1).clamp_min(EPSILON)
            return class_means * scales[:, None], counts
        raise ValueError("Unknown class_mean_fc_init_scale: {}".format(scale_mode))

    def _apply_class_mean_fc_init(self, ptr, class_mean_weights):
        mode = self.args.get("class_mean_fc_init_mode", "partial_new_regions")
        bias_mode = self.args.get("class_mean_fc_init_bias", "zero")
        old_classes = int(self._known_classes)
        total_classes = int(self._total_classes)
        old_feature_dim = ptr.feature_dim - ptr.out_dim

        with torch.no_grad():
            if mode == "partial_new_regions":
                if old_classes > 0 and old_feature_dim > 0:
                    ptr.fc.weight[:old_classes, old_feature_dim:] = class_mean_weights[:old_classes, old_feature_dim:]
                ptr.fc.weight[old_classes:total_classes, :] = class_mean_weights[old_classes:total_classes, :]
                bias_slice = slice(0, total_classes)
            elif mode == "all_seen":
                ptr.fc.weight[:total_classes, :] = class_mean_weights[:total_classes, :]
                bias_slice = slice(0, total_classes)
            elif mode == "new_classes":
                ptr.fc.weight[old_classes:total_classes, :] = class_mean_weights[old_classes:total_classes, :]
                bias_slice = slice(old_classes, total_classes)
            else:
                raise ValueError("Unknown class_mean_fc_init_mode: {}".format(mode))

            if ptr.fc.bias is not None:
                if bias_mode == "zero":
                    ptr.fc.bias[bias_slice].zero_()
                elif bias_mode == "keep":
                    pass
                else:
                    raise ValueError("Unknown class_mean_fc_init_bias: {}".format(bias_mode))

    def _init_fc_with_class_means(self, init_loader):
        if self._cur_task == 0:
            return

        ptr = self._network.module if hasattr(self._network, "module") else self._network
        local_rank = self.args.get("local_rank", 0)
        distributed_ready = dist.is_available() and dist.is_initialized()
        should_compute = (not distributed_ready) or dist.get_rank() == 0
        if local_rank <= 0:
            logging.info(
                "Running class_mean_fc_init before Task {} training: mode={}, scale={}, bias={}".format(
                    self._cur_task,
                    self.args.get("class_mean_fc_init_mode", "partial_new_regions"),
                    self.args.get("class_mean_fc_init_scale", "global_weight_norm"),
                    self.args.get("class_mean_fc_init_bias", "zero"),
                )
            )

        was_training = ptr.training
        ptr.to(self._device)
        ptr.eval()

        if should_compute:
            features, targets = [], []
            with torch.no_grad():
                for batch in tqdm(init_loader, desc="Class mean fc init", dynamic_ncols=True, disable=local_rank > 0):
                    inputs = batch[1].to(self._device, non_blocking=True)
                    batch_targets = batch[-1].to(self._device, non_blocking=True)
                    inputs, batch_targets = self._flatten_augmented_inputs(inputs, batch_targets)
                    outputs = ptr(inputs)
                    features.append(outputs["features"].detach().cpu())
                    targets.append(batch_targets.detach().cpu())

            features = torch.cat(features, dim=0)
            targets = torch.cat(targets, dim=0)
            reference_weight = ptr.fc.weight.detach().cpu().clone()
            class_mean_weights, counts = self._compute_class_mean_fc_weights(
                features,
                targets,
                reference_weight,
                self.args.get("class_mean_fc_init_scale", "global_weight_norm"),
            )
            self._apply_class_mean_fc_init(ptr, class_mean_weights.to(self._device))

            if local_rank <= 0:
                old_counts = counts[: self._known_classes].tolist()
                new_counts = counts[self._known_classes : self._total_classes].tolist()
                logging.info(
                    "class_mean_fc_init sample counts: old_classes={}, new_classes={}".format(
                        old_counts, new_counts
                    )
                )

        if distributed_ready:
            dist.broadcast(ptr.fc.weight.data, src=0)
            if ptr.fc.bias is not None:
                dist.broadcast(ptr.fc.bias.data, src=0)

        if was_training:
            ptr.train()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self.sig_reg.to(self._device)

        torch.backends.cudnn.benchmark = True

        # 过滤需要梯度的参数
        trainable_params = filter(lambda p: p.requires_grad, self._network.parameters())

        if self._cur_task == 0:
            optimizer = torch.optim.AdamW(trainable_params,  lr=init_lr, weight_decay=init_weight_decay)
            # 使用包含 Warmup 的调度器（LeJEPA 官方推荐）
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

            # 权重对齐
            ptr = self._network.module if hasattr(self._network, 'module') else self._network
            ptr.weight_align(self._total_classes - self._known_classes)
            torch.cuda.empty_cache()

    def denormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(3, 1, 1)
        return tensor * std + mean

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(init_epoch), disable=disable_tqdm)

        V_dim = self.args.get('num_views', 8)
        lamb = self.args.get('lejepa_lambda', 0.05)
        ts_global_only = self.args.get("ts_global_only", False)
        ts_views = self._num_ts_views(V_dim) if ts_global_only else V_dim

        # --- [关键：重置本阶段步数] ---
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, correct, total = 0.0, 0, 0
            diagnostic_counts = self._new_accuracy_counts() if self._record_accuracy_curves_enabled() else None
            for i, data in enumerate(train_loader):
                vs = data[1].to(self._device)
                targets = data[-1].to(self._device)
                N = vs.shape[0]

                # --- [可视化] ---
                if i == 0 and epoch % 10 == 0 and local_rank <= 0:
                    # 取第 0 个样本的所有视图：vs[0] -> [8, 3, 224, 224]
                    sample_8_views = vs[0].cpu()
                    swan_images = []
                    for idx in range(V_dim):
                        # 加入 denormalize 还原颜色
                        raw_img = torch.clamp(self.denormalize(sample_8_views[idx]), 0, 1)
                        label = "Global" if idx < 2 else "Local"
                        swan_images.append(swanlab.Image(to_pil_image(raw_img), caption=f"{label}_{idx}"))
                    swanlab.log({"Visual/8_Views_Check": swan_images})


                # --- [前向传播] ---
                out = self._network(vs.flatten(0, 1))
                logits, embedding = out["logits"], out["embedding"]
                probe_feature_name = self.args.get("probe_feature", "ta_feature")
                if probe_feature_name not in out:
                    raise ValueError("Unknown probe_feature: {}".format(probe_feature_name))
                probe_features = out[probe_feature_name]

                # --- [LeJEPA / self-supervised loss] ---
                inv_loss, sigreg_loss, lejepa_loss = self._compute_selfsup_losses(
                    embedding, N, V_dim, lamb
                )

                # 4. 分类损失
                y_rep = targets.repeat_interleave(ts_views)
                logits_ts = logits
                if ts_global_only:
                    logits_ts = self._select_ts_views(logits, N, V_dim, ts_views)
                ce_loss = F.cross_entropy(logits_ts, y_rep)
                loss = ce_loss + lejepa_loss * self.args.get('contrast_factor', 1.0)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                probe_targets = targets.repeat_interleave(V_dim)
                probe_log = self._update_task0_online_probe(probe_features, probe_targets)

                _, preds = torch.max(logits_ts, dim=1)
                batch_correct = preds.eq(y_rep).sum().item()
                batch_total = y_rep.size(0)
                batch_acc = (batch_correct * 100 / batch_total)

                # --- [核心修改：每个 Batch 记录一次数据] ---
                if local_rank <= 0:
                    # SwanLab 默认会自动累计 step，直接 log 即可。
                    # 如果你想跨 Task 保持步数连续，可以传入 step=self.global_step
                    log_payload = {
                        "init/total_loss": loss.item(),
                        "init/batch_acc": batch_acc,
                        "init/Prediction_Invariance_loss": inv_loss.item(),
                        "init/SIGReg_loss": sigreg_loss.item(),
                        "init/LeJEPA_total_loss": lejepa_loss.item(),
                        "init/ce_loss": ce_loss.item(),
                        "init/lr": optimizer.param_groups[0]['lr'],
                    }
                    if probe_log:
                        log_payload.update({
                            "init/probe_loss": probe_log["probe_loss"],
                            "init/probe_top1": probe_log["probe_top1"],
                        })
                    swanlab.log(log_payload, step=batch_step)
                    batch_step += 1

                losses += loss.item()
                correct += batch_correct
                total += batch_total  # 统计总预测数 (N*V)
                if diagnostic_counts is not None:
                    self._update_accuracy_counts(diagnostic_counts, preds, y_rep)



            avg_loss = losses / len(train_loader)
            if not disable_tqdm:
                train_acc = np.around(correct * 100 / total, decimals=2)

                prog_bar.set_description(
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{init_epoch} Loss {avg_loss:.3f}, Acc {train_acc:.2f}")

            self._record_epoch_accuracy_diagnostics(
                "init",
                epoch,
                init_epoch,
                diagnostic_counts,
                avg_loss,
                optimizer.param_groups[0]['lr'],
                test_loader,
            )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(epochs), disable=disable_tqdm)

        V_dim = self.args.get('num_views', 8)
        lamb = self.args.get('lejepa_lambda', 0.05)
        ts_global_only = self.args.get("ts_global_only", False)
        global_views = self._num_ts_views(V_dim)
        ts_views = global_views if ts_global_only else V_dim
        kd_global_only = self.args.get("kd_global_only", False) or ts_global_only
        kd_views = global_views if kd_global_only else V_dim

        task_prefix = f"Task_{self._cur_task}"
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, losses_clf, losses_aux, correct, total = 0.0, 0.0, 0.0, 0, 0
            diagnostic_counts = self._new_accuracy_counts() if self._record_accuracy_curves_enabled() else None
            for i, data in enumerate(train_loader):
                vs = data[1].to(self._device)  # [Batch, 8, 3, 224, 224]
                targets = data[-1].to(self._device)  # [Batch]
                N = vs.shape[0]

                outputs = self._network(vs.flatten(0, 1))
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                embedding = outputs['embedding']

                # --- [LeJEPA / self-supervised loss] ---
                inv_loss, sigreg_loss, lejepa_loss = self._compute_selfsup_losses(
                    embedding, N, V_dim, lamb
                )

                # --- [分类与增量损失] ---
                y_rep = targets.repeat_interleave(ts_views)
                logits_ts = logits
                aux_logits_ts = aux_logits
                if ts_global_only:
                    logits_ts = self._select_ts_views(logits, N, V_dim, ts_views)
                    aux_logits_ts = self._select_ts_views(aux_logits, N, V_dim, ts_views)
                loss_clf = F.cross_entropy(logits_ts, y_rep)

                # Aux Loss
                aux_targets = y_rep.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                                          aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(aux_logits_ts, aux_targets)

                # Distill Loss
                predicted_feature = outputs['predicted_feature']  # [N*V, Dim]
                with torch.no_grad():
                    old_out = self.last_ta_net(vs.flatten(0, 1))
                    old_ta_feature = old_out['features']

                if kd_global_only:
                    # LeJEPA 仍使用全部 8 个视图进行自监督；仅将 KD 约束限制在前两个 global views。
                    predicted_feature = predicted_feature.reshape(N, V_dim, -1)[:, :kd_views, :].reshape(N * kd_views, -1)
                    old_ta_feature = old_ta_feature.reshape(N, V_dim, -1)[:, :kd_views, :].reshape(N * kd_views, -1)

                z_target = self.last_projector(old_ta_feature)
                p_pred = self.last_projector(predicted_feature)
                kd_loss = infoNCE_distill_loss(p_pred, z_target, self.args['infonce_kd_temp'])

                # Transfer Loss
                trans_logits = outputs["trans_logits"]
                if ts_global_only:
                    trans_logits = self._select_ts_views(trans_logits, N, V_dim, ts_views)
                cur_task_mask = (y_rep >= self._known_classes)
                # trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask],targets[cur_task_mask] - self._known_classes)
                y_rep_new = y_rep - self._known_classes  # 偏移标签
                trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask], y_rep_new[cur_task_mask])

                if trans_cls_loss < loss_clf:
                    temp_T = self.args['kd_temp']
                    transfer_loss = F.kl_div(
                        (logits_ts[cur_task_mask][:, self._known_classes:] / temp_T).log_softmax(dim=1),
                        (trans_logits.detach()[cur_task_mask] / temp_T).softmax(dim=1), reduction='batchmean')
                else:
                    transfer_loss = torch.tensor(0., device=self._device)

                auto_kd_factor = self._known_classes / self._total_classes
                loss = loss_clf + \
                       self.args['aux_factor'] * loss_aux + \
                       self.args['contrast_factor'] * (lejepa_loss * (1 - auto_kd_factor) + self.args[
                    'contrast_kd_factor'] * kd_loss * auto_kd_factor) + \
                       self.args['trans_cls_factor'] * trans_cls_loss + \
                       self.args['transfer_factor'] * transfer_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits_ts, dim=1)
                batch_acc = preds.eq(y_rep).sum().item() * 100.0 / y_rep.size(0)
                correct += preds.eq(y_rep).cpu().sum()
                total += len(y_rep)
                if diagnostic_counts is not None:
                    self._update_accuracy_counts(diagnostic_counts, preds, y_rep)
                # ==========================================================================================
                # [Loss Functions Summary - TagFex with LeJEPA]
                # 1. 核心分类 (Task-specific Classification):
                #    - loss_clf: 主分类损失。约束 Backbone 提取具有判别性的特征以区分当前任务类别。
                #    - loss_aux: 辅助分类损失 (DER)。将旧类视为整体，专注于提升新类别的特征提取质量。
                #
                # 2. LeJEPA 自监督 (Task-agnostic Representation):
                #    - inv_loss (Invariance): 视图不变性损失。拉近同一样本不同增强视图间的距离，学习物体本质特征。
                #    - sigreg_loss (Variance/Covariance): 正则项。防止特征空间坍缩，确保特征维度分布的独立性与多样性。
                #    - lejepa_loss: 上述两者的加权组合，代表无监督表征学习的整体质量。
                #
                # 3. 知识保持与防遗忘 (Knowledge Preservation):
                #    - kd_loss (InfoNCE Distillation): 特征级蒸馏。利用对比学习强制当前模型复现旧模型的特征布局。
                #    - transfer_loss (KL Divergence): 逻辑对齐。当迁移分类器表现更好时，引导主分类器模仿其输出概率。
                #
                # 4. 特征迁移 (Feature Transfer):
                #    - trans_cls_loss: 迁移分类损失。优化 Merge Attention 模块，使其能有效聚合任务无关与任务相关的特征。
                #
                # 5. 权重平衡 (Dynamic Balancing):
                #    - auto_kd_factor: 动态因子 (已知类/总类)。任务前期侧重 LeJEPA 探索，任务后期侧重 KD 蒸馏以抑制遗忘。
                # ==========================================================================================
                if local_rank <= 0:
                    swanlab.log({
                        f"{task_prefix}/total_loss": loss.item(),
                        f"{task_prefix}/train_acc": batch_acc,
                        f"{task_prefix}/clf_loss": loss_clf.item(),
                        f"{task_prefix}/kd_loss": kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
                        f"{task_prefix}/Prediction_Invariance_loss": inv_loss.item(),
                        f"{task_prefix}/SIGReg_loss": sigreg_loss.item(),
                        f"{task_prefix}/lejepa_loss": lejepa_loss.item(),
                        f"{task_prefix}/epoch": epoch
                    }, step=batch_step)
                    batch_step += 1
                # -------------------------------


            avg_loss = losses / len(train_loader)
            if not disable_tqdm:
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                prog_bar.set_description(
                    f"Task {self._cur_task} Epoch {epoch + 1}/{epochs} Loss {avg_loss:.3f} Acc {train_acc:.2f}")

            self._record_epoch_accuracy_diagnostics(
                "update",
                epoch,
                epochs,
                diagnostic_counts,
                avg_loss,
                optimizer.param_groups[0]['lr'],
                test_loader,
            )

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

    def _final_sigreg_seed_for_task(self):
        base_seed = int(self.args.get("final_sigreg_matrix_seed", self.args.get("seed", 0)))
        return base_seed + int(self._cur_task) * 1000003

    def _make_final_sigreg_matrix(self, feat_dim, num_directions):
        generator = torch.Generator(device="cpu")
        seed = self._final_sigreg_seed_for_task()
        generator.manual_seed(seed)
        matrix = torch.randn(feat_dim, num_directions, generator=generator, dtype=torch.float32)
        matrix = matrix / matrix.norm(p=2, dim=0, keepdim=True).clamp_min(EPSILON)
        return matrix, seed

    def _sigreg_loss_with_matrix(self, proj, matrix):
        matrix = matrix.to(proj.device)
        t = self.sig_reg.t.to(proj.device)
        phi = self.sig_reg.phi.to(proj.device)
        weights = self.sig_reg.weights.to(proj.device)
        x_t = (proj @ matrix).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()

    def _final_sigreg_embedding(self, ptr, inputs):
        ta_out = ptr._get_active_ta_model()(inputs)
        ta_fmap = ta_out["fmaps"][-1]
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
        return ptr.projector(ta_feature)

    def _set_final_sigreg_update_scope(self):
        ptr = self._network.module if hasattr(self._network, "module") else self._network
        requires_grad_state = {
            name: param.requires_grad for name, param in ptr.named_parameters()
        }
        for param in ptr.parameters():
            param.requires_grad_(False)

        ta_model = ptr._get_active_ta_model()
        for param in ta_model.parameters():
            param.requires_grad_(True)
        for param in ptr.projector.parameters():
            param.requires_grad_(True)

        trainable_params = [
            param for param in list(ta_model.parameters()) + list(ptr.projector.parameters())
            if param.requires_grad
        ]
        return ptr, requires_grad_state, trainable_params

    def _restore_requires_grad(self, ptr, requires_grad_state):
        for name, param in ptr.named_parameters():
            if name in requires_grad_state:
                param.requires_grad_(requires_grad_state[name])

    def run_final_sigreg_calibration(self, data_manager, artifact_dir, ckpt_dir=None):
        if not self.args.get("final_sigreg", False):
            return None

        local_rank = self.args.get("local_rank", 0)
        artifact_dir = artifact_dir or "."
        os.makedirs(artifact_dir, exist_ok=True)
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)

        ptr, requires_grad_state, trainable_params = self._set_final_sigreg_update_scope()
        if not trainable_params:
            raise RuntimeError("Final SIGReg has no trainable TA/projector parameters.")

        ptr.to(self._device)
        self.sig_reg.to(self._device)
        ptr.train()
        for convnet in ptr.convnets:
            convnet.eval()
        if ptr.fc is not None:
            ptr.fc.eval()
        if ptr.aux_fc is not None:
            ptr.aux_fc.eval()
        if ptr.trans_classifier is not None:
            ptr.trans_classifier.eval()
        if ptr.predictor is not None:
            ptr.predictor.eval()
        ptr._get_active_ta_model().train()
        ptr.projector.train()

        batch_size_final = int(self.args.get("final_sigreg_batch_size", batch_size))
        num_workers_final = int(self.args.get("final_sigreg_num_workers", num_workers))
        epochs_final = int(self.args.get("final_sigreg_epochs", 1))
        lr_final = float(self.args.get("final_sigreg_lr", 5e-5))
        weight_decay_final = float(self.args.get("final_sigreg_weight_decay", weight_decay))
        loss_factor = float(self.args.get("final_sigreg_factor", 1.0))
        num_directions = int(self.args.get("final_sigreg_num_directions", 256))
        V_dim = int(self.args.get("num_views", 8))

        final_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=None,
        )
        final_loader = DataLoader(
            final_dataset,
            batch_size=batch_size_final,
            shuffle=True,
            num_workers=num_workers_final,
            pin_memory=self._device.type == "cuda",
            drop_last=False,
        )

        feat_dim = int(self.args.get("proj_output_dim", 1024))
        A_final, matrix_seed = self._make_final_sigreg_matrix(feat_dim, num_directions)
        a_final_path = None
        if ckpt_dir is not None and local_rank <= 0:
            a_final_path = os.path.join(
                ckpt_dir,
                "{}_{}_task_{}_final_sigreg_A_final.pth".format(
                    self.args["prefix"], self.args["seed"], self._cur_task
                ),
            )
            torch.save(
                {
                    "task_id": self._cur_task,
                    "matrix_mode": "final_fixed",
                    "A_final": A_final,
                    "projection_matrix": A_final,
                    "matrix_seed": matrix_seed,
                    "num_directions": num_directions,
                    "feat_dim": feat_dim,
                },
                a_final_path,
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr_final,
            weight_decay=weight_decay_final,
        )

        loss_rows = []
        global_step = 0
        for epoch in range(epochs_final):
            epoch_loss, epoch_samples = 0.0, 0
            for batch_idx, data in enumerate(final_loader):
                vs = data[1].to(self._device, non_blocking=True)
                if vs.ndim == 5:
                    n = vs.shape[0]
                    inputs = vs.flatten(0, 1)
                    num_samples = n * vs.shape[1]
                elif vs.ndim == 4:
                    inputs = vs
                    n = max(1, vs.shape[0] // V_dim)
                    num_samples = vs.shape[0]
                else:
                    raise ValueError("Unexpected final SIGReg input shape: {}".format(list(vs.shape)))

                embedding = self._final_sigreg_embedding(ptr, inputs)
                sigreg_loss = self._sigreg_loss_with_matrix(embedding, A_final)
                loss = sigreg_loss * loss_factor

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                sigreg_value = float(sigreg_loss.detach().cpu())
                epoch_loss += loss_value
                epoch_samples += num_samples
                row = {
                    "task_id": self._cur_task,
                    "epoch": epoch,
                    "iter": batch_idx,
                    "global_step": global_step,
                    "loss": loss_value,
                    "sigreg_loss": sigreg_value,
                    "lr": optimizer.param_groups[0]["lr"],
                    "batch_samples": int(num_samples),
                    "batch_items": int(n),
                }
                loss_rows.append(row)

                if local_rank <= 0:
                    swanlab.log(
                        {
                            f"Task_{self._cur_task}/final_sigreg_loss": loss_value,
                            f"Task_{self._cur_task}/final_sigreg_raw": sigreg_value,
                        }
                    )
                global_step += 1

            if local_rank <= 0:
                logging.info(
                    "Final SIGReg task {} epoch {}/{} loss {:.6f}.".format(
                        self._cur_task,
                        epoch + 1,
                        epochs_final,
                        epoch_loss / max(1, len(final_loader)),
                    )
                )

        loss_csv_path = os.path.join(
            artifact_dir,
            "final_sigreg_task_{}_loss.csv".format(self._cur_task),
        )
        if local_rank <= 0:
            with open(loss_csv_path, "w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "task_id",
                    "epoch",
                    "iter",
                    "global_step",
                    "loss",
                    "sigreg_loss",
                    "lr",
                    "batch_samples",
                    "batch_items",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(loss_rows)

        self._restore_requires_grad(ptr, requires_grad_state)
        ptr.eval()

        final_state = {
            "enabled": True,
            "task_id": self._cur_task,
            "epochs": epochs_final,
            "lr": lr_final,
            "weight_decay": weight_decay_final,
            "loss_factor": loss_factor,
            "batch_size": batch_size_final,
            "num_workers": num_workers_final,
            "matrix_mode": "final_fixed",
            "update_scope": "ta_projector",
            "matrix_seed": matrix_seed,
            "num_directions": num_directions,
            "feat_dim": feat_dim,
            "A_final_path": a_final_path,
            "loss_csv_path": loss_csv_path if local_rank <= 0 else None,
            "num_loss_rows": len(loss_rows),
            "final_loss": loss_rows[-1]["loss"] if loss_rows else None,
            "avg_loss": float(np.mean([row["loss"] for row in loss_rows])) if loss_rows else None,
        }
        self._final_sigreg_last_state = final_state
        return final_state

    def get_final_sigreg_state(self):
        return self._final_sigreg_last_state


def infoNCE_loss(feats, t):
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    return nll.mean()

"""
def infoNCE_distill_loss(p_feats, z_feats, t):
    cos_sim = F.cosine_similarity(p_feats[:, None, :], z_feats[None, :, :], dim=-1)
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    return nll.mean()
"""


def infoNCE_distill_loss(p_feats, z_feats, t):
    # p_feats: [N*V, Dim], z_feats: [N*V, Dim]
    # 归一化特征
    p_feats = F.normalize(p_feats, dim=-1)
    z_feats = F.normalize(z_feats, dim=-1)

    # 1. 计算所有样本对之间的余弦相似度矩阵 [N*V, N*V]
    cos_sim = torch.matmul(p_feats, z_feats.T) / t

    # 2. 确定正确的正样本掩码：对角线上的才是同一个样本的视图对齐
    # 因为 p_feats 和 z_feats 的顺序是一一对应的 (N*V)
    labels = torch.arange(p_feats.size(0)).to(p_feats.device)

    # 3. 使用交叉熵计算 InfoNCE (这会自动把对角线当做正样本，其他当做负样本)
    loss = F.cross_entropy(cos_sim, labels)
    return loss
