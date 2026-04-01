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
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class SIGReg(nn.Module):
    """
    LeJEPA 的核心组件：Sketched Isotropic Gaussian Regularization (SIGReg)
    该损失函数确保 learned embeddings 符合标准正态分布，从而最小化下游风险。
    """

    def __init__(self, knots=17):
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

    def forward(self, proj):
        # proj(Projected Embeddings投影后的特征向量): [Views, Batch, Dim] 或者是拼接后的特征 [N, Dim][128,1024]
        # 1. 随机投影到一个子空间（Sketched）
        A = torch.randn(proj.size(-1), 256, device=proj.device)  # [1024, 256]
        A = A.div_(A.norm(p=2, dim=0))

        # 2. 计算特征函数并与标准高斯分布对比
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # 3. 计算统计量
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class TagFex(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TagFexNet(args, False)
        # --- 新增：实例化 SIGReg ---
        self.sig_reg = SIGReg(knots=17)

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
            pin_memory=True, sampler=train_sampler
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
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
        torch.backends.cudnn.benchmark = True

        # 过滤需要梯度的参数
        trainable_params = filter(lambda p: p.requires_grad, self._network.parameters())

        if self._cur_task == 0:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=init_milestones, gamma=init_lr_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(trainable_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

            # 权重对齐
            ptr = self._network.module if hasattr(self._network, 'module') else self._network
            ptr.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(init_epoch), disable=disable_tqdm)

        V = self.args.get('num_views', 8)
        lamb = self.args.get('lejepa_lambda', 0.5)

        # 1. 组合随机增强逻辑 (不含 ToTensor)
        train_trsf = transforms.Compose(train_loader.dataset.trsf)
        # 2. 组合标准化逻辑 (ToTensor + Normalize)
        common_trsf = transforms.Compose(train_loader.dataset.common_trsf)

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, correct, total = 0.0, 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # --- [核心：生成 V 个视图] ---
                multi_views = []
                for _ in range(V):
                    # 对 Batch 里的每张图独立应用随机增强
                    view = torch.stack([common_trsf(train_trsf(img)) for img in inputs])
                    multi_views.append(view)

                vs = torch.stack(multi_views, dim=1).to(self._device)  # [N, V, 3, 224, 224]
                N, V_dim = vs.shape[0], vs.shape[1]

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

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            if not disable_tqdm:
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                prog_bar.set_description(
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{init_epoch} Loss {losses / len(train_loader):.3f}, Acc {train_acc:.2f}")

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        local_rank = self.args.get("local_rank", 0)
        disable_tqdm = (local_rank > 0)
        prog_bar = tqdm(range(epochs), disable=disable_tqdm)

        for _, epoch in enumerate(prog_bar):
            if train_loader.sampler is not None:
                train_loader.sampler.set_epoch(epoch)

            self.train()
            losses, losses_clf, losses_aux, correct, total = 0.0, 0.0, 0.0, 0, 0
            for i, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)
                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                embedding = outputs['embedding']

                infonce_loss = infoNCE_loss(embedding, self.args['infonce_temp'])
                loss_clf = F.cross_entropy(logits, targets)

                # Aux Loss
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                                          aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)

                # Distill Loss
                predicted_feature = outputs['predicted_feature']
                old_ta_feature = self.last_ta_net(inputs.contiguous())['features']
                kd_loss = infoNCE_distill_loss(self.last_projector(predicted_feature),
                                               self.last_projector(old_ta_feature), self.args['infonce_kd_temp'])

                # Transfer Loss
                trans_logits = outputs["trans_logits"]
                cur_task_mask = (targets >= self._known_classes)
                trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask],
                                                 targets[cur_task_mask] - self._known_classes)

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
                       self.args['contrast_factor'] * (infonce_loss * (1 - auto_kd_factor) + self.args[
                    'contrast_kd_factor'] * kd_loss * auto_kd_factor) + \
                       self.args['trans_cls_factor'] * trans_cls_loss + \
                       self.args['transfer_factor'] * transfer_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            if not disable_tqdm:
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
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


def infoNCE_distill_loss(p_feats, z_feats, t):
    cos_sim = F.cosine_similarity(p_feats[:, None, :], z_feats[None, :, :], dim=-1)
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    return nll.mean()