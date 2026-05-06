# Please note that only "cifar100_aa" and "cifar10_aa" are supported for TagFex in PyCIL_DDP.
import logging
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

        # --- SwanLab 初始化 ---
        local_rank = self.args.get("local_rank", 0)
        if local_rank <= 0:
            # 每个任务开始时，初始化或更新实验记录
            swanlab.init(
                project="PyCIL_TagFex",
                experiment_name=f"TagFex",
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

        # --- 关键修改：Task 结束后的处理 ---
        # 无论 DDP 还是 DP，在构建 memory 前保持包装，build 完后统一解包
        if hasattr(self._network, 'module'):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            self._network = self._network.module  # 解包，回到 TagFexNet 原始类
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def train(self):
        self._network.train()
        # 统一处理 DDP 或单卡指针
        ptr = self._network.module if hasattr(self._network, 'module') else self._network
        ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                ptr.convnets[i].eval()

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

        # --- [关键：重置本阶段步数] ---
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, correct, total = 0.0, 0, 0
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

                # --- [LeJEPA 损失计算] ---
                # 1. 不变性损失 (Invariance)
                proj = embedding.reshape(N, V_dim, -1)
                proj_mean = proj.mean(1, keepdim=True)
                inv_loss = (proj_mean - proj).square().mean()

                # 2. 高斯正则化 (SIGReg)
                sigreg_loss = self.sig_reg(embedding)

                # 3. 汇总
                lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

                # 4. 分类损失
                y_rep = targets.repeat_interleave(V_dim)
                ce_loss = F.cross_entropy(logits, y_rep)
                loss = ce_loss + lejepa_loss * self.args.get('contrast_factor', 1.0)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


                _, preds = torch.max(logits, dim=1)
                batch_correct = preds.eq(y_rep).sum().item()
                batch_total = y_rep.size(0)
                batch_acc = (batch_correct * 100 / batch_total)

                # --- [核心修改：每个 Batch 记录一次数据] ---
                if local_rank <= 0:
                    # SwanLab 默认会自动累计 step，直接 log 即可。
                    # 如果你想跨 Task 保持步数连续，可以传入 step=self.global_step
                    swanlab.log({
                        "init/total_loss": loss.item(),
                        "init/batch_acc": batch_acc,
                        "init/Prediction_Invariance_loss": inv_loss.item(),
                        "init/SIGReg_loss": sigreg_loss.item(),
                        "init/LeJEPA_total_loss": lejepa_loss.item(),
                        "init/ce_loss": ce_loss.item(),
                        "init/lr": optimizer.param_groups[0]['lr'],
                    }, step=batch_step)
                    batch_step += 1

                losses += loss.item()
                correct += batch_correct
                total += batch_total  # 统计总预测数 (N*V)



            if not disable_tqdm:
                train_acc = np.around(correct * 100 / total, decimals=2)

                prog_bar.set_description(
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{init_epoch} Loss {losses / len(train_loader):.3f}, Acc {train_acc:.2f}")

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(epochs), disable=disable_tqdm)

        V_dim = self.args.get('num_views', 8)
        lamb = self.args.get('lejepa_lambda', 0.05)
        kd_global_only = self.args.get("kd_global_only", False)

        task_prefix = f"Task_{self._cur_task}"
        batch_step = 0

        for _, epoch in enumerate(prog_bar):
            if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, losses_clf, losses_aux, correct, total = 0.0, 0.0, 0.0, 0, 0
            for i, data in enumerate(train_loader):
                vs = data[1].to(self._device)  # [Batch, 8, 3, 224, 224]
                targets = data[-1].to(self._device)  # [Batch]
                N = vs.shape[0]

                outputs = self._network(vs.flatten(0, 1))
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                embedding = outputs['embedding']

                # --- [LeJEPA 损失] ---
                proj = embedding.reshape(N, V_dim, -1)
                proj_mean = proj.mean(1, keepdim=True)
                inv_loss = (proj_mean - proj).square().mean()
                sigreg_loss = self.sig_reg(embedding)
                lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

                # --- [分类与增量损失] ---
                y_rep = targets.repeat_interleave(V_dim)
                loss_clf = F.cross_entropy(logits, y_rep)

                # Aux Loss
                aux_targets = y_rep.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                                          aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)

                # Distill Loss
                predicted_feature = outputs['predicted_feature']  # [N*V, Dim]
                with torch.no_grad():
                    old_out = self.last_ta_net(vs.flatten(0, 1))
                    old_ta_feature = old_out['features']

                if kd_global_only:
                    # LeJEPA 仍使用全部 8 个视图进行自监督；仅将 KD 约束限制在前两个 global views。
                    predicted_feature = predicted_feature.reshape(N, V_dim, -1)[:, :2, :].reshape(N * 2, -1)
                    old_ta_feature = old_ta_feature.reshape(N, V_dim, -1)[:, :2, :].reshape(N * 2, -1)

                z_target = self.last_projector(old_ta_feature)
                p_pred = self.last_projector(predicted_feature)
                kd_loss = infoNCE_distill_loss(p_pred, z_target, self.args['infonce_kd_temp'])

                # Transfer Loss
                trans_logits = outputs["trans_logits"]
                cur_task_mask = (y_rep >= self._known_classes)
                # trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask],targets[cur_task_mask] - self._known_classes)
                y_rep_new = y_rep - self._known_classes  # 偏移标签
                trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask], y_rep_new[cur_task_mask])

                if trans_cls_loss < loss_clf:
                    temp_T = self.args['kd_temp']
                    transfer_loss = F.kl_div(
                        (logits[cur_task_mask][:, self._known_classes:] / temp_T).log_softmax(dim=1),
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
                _, preds = torch.max(logits, dim=1)
                batch_acc = preds.eq(y_rep).sum().item() * 100.0 / y_rep.size(0)
                correct += preds.eq(y_rep).cpu().sum()
                total += len(y_rep)
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


            if not disable_tqdm:
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                avg_loss = losses / len(train_loader)

                prog_bar.set_description(
                    f"Task {self._cur_task} Epoch {epoch + 1}/{epochs} Loss {losses / len(train_loader):.3f} Acc {train_acc:.2f}")

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
