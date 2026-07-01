# TA-LeJEPA Probe

这个目录用于做一个独立的 TA-LeJEPA 诊断实验：只训练 Task-Agnostic branch，也就是 TA encoder + projector，并用 online linear probe 观察 TA 表示是否逐渐变得线性可分。

它刻意不走完整的 TagFex / CIL 训练流程，目的是把问题拆干净：

```text
TA 分支本身按 LeJEPA 目标训练，到底能不能学到可分类表示？
```

## 实验范围

这个脚本会使用：

- 现有 `DataManager` 中的 `imagenet100_lejepa` Task0 数据
- 当前 8-view 训练输入和增强流程
- 一个独立的 ResNet18 TA encoder
- 当前 TagFex-LeJEPA 的 projector 结构：`512 -> 2048 -> 2048 -> 1024`
- LeJEPA invariance loss + SIGReg loss
- 基于 `ta_feature.detach()` 的 online linear probe

这个脚本不会使用：

```text
TS experts
mean fusion
主分类器 CE loss
transfer loss
KD loss
memory replay
CIL task update
```

因此它不是完整 TagFex 实验，而是一个 standalone TA 自监督分支诊断实验。

## 运行命令

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0.json
```

默认配置文件是：

```text
exps/lejepa_sigreg/ta_lejepa_probe_task0.json
```

第一版默认配置：

```text
epochs: 200
batch_size: 64
eval_batch_size: 256
eval_every: 10
eval_train: false
lr: 5e-4
weight_decay: 5e-4
lejepa_lambda: 0.05
probe_lr: 1e-3
probe_weight_decay: 1e-7
probe_norm: layernorm
num_views: 8
```


## 继续训练

如果 200 epoch 结束后发现还没有收敛，可以从 `last` checkpoint 继续训练。推荐使用 `--extra-epochs`，语义是“在 checkpoint 后额外训练多少个 epoch”：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0.json \
  --resume-checkpoint standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200/checkpoints/standalone_ta_lejepa_seed1993_task0_last_ep200.pth \
  --extra-epochs 100
```

继续训练会加载：

```text
encoder 权重
projector 权重
linear probe 权重
optimizer 状态
probe optimizer 状态
随机数状态
历史 best 信息
```

默认情况下，继续训练会按当前配置重新启动这一段训练的 warmup + cosine 学习率调度，并把 optimizer 里的学习率重置为配置中的 `lr` / `probe_lr`。如果想保留 checkpoint 里的学习率，可以加：

```bash
--no-resume-reset-lr
```

如果训练中途被手动中断，但已经保存了 `best_test_probe` checkpoint，并且只是想接着原来的训练进度继续跑，建议保留 checkpoint 里的 optimizer / scheduler 学习率状态：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0.json \
  --resume-checkpoint standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200/checkpoints/standalone_ta_lejepa_seed1993_task0_best_test_probe.pth \
  --no-resume-reset-lr
```

继续训练时仍然会保存新的 `last` checkpoint，并按 `standalone_ta_eval/test_probe_top1` 更新 `best_test_probe` checkpoint。

## 输出文件

结果会保存在配置文件中的 `output_dir`，默认是：

```text
standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200
```

目录中包含：

```text
standalone_ta_metrics.csv          # step 级 loss 和 probe top1
standalone_ta_epoch_metrics.csv    # epoch 平均指标
summary.json                       # 运行摘要
checkpoints/*_last_*.pth            # 训练结束时的最后权重
checkpoints/*_best_test_probe.pth   # 按 epoch-end test_probe_top1 保存的最佳权重
```

`last` 和 `best` 的区别是：

```text
last：最后一个 epoch 结束后的模型状态
best：训练过程中 standalone_ta_eval/test_probe_top1 最高时的模型状态
```


## SwanLab 项目

这个 standalone 实验建议单独放在一个 SwanLab project 下，避免和完整 TagFex / PyCIL 主实验混在一起。

默认配置为：

```text
project: TagFex_TA_LeJEPA_Probe
run: task0_lr5e-4_lambda0.05_ep200
```

## SwanLab 指标

逐 step 记录：

```text
standalone_ta/Prediction_Invariance_loss
standalone_ta/SIGReg_loss
standalone_ta/LeJEPA_total_loss
standalone_ta/probe_loss
standalone_ta/probe_top1
standalone_ta/lr
```

每个 epoch 记录一次训练过程中的 batch 平均值：

```text
standalone_ta_epoch/Prediction_Invariance_loss
standalone_ta_epoch/SIGReg_loss
standalone_ta_epoch/LeJEPA_total_loss
standalone_ta_epoch/probe_loss
standalone_ta_epoch/probe_top1
standalone_ta_epoch/lr
```

为了避免每个 epoch 都完整跑测试集导致训练过慢，默认每 10 个 epoch 做一次 epoch-end eval，并且最后一个 epoch 必做一次。默认只评估 Task0 test set，不额外评估 train set：

```text
standalone_ta_eval/test_probe_loss
standalone_ta_eval/test_probe_top1
```

如果需要同时评估 train set，可以在命令中加入：

```bash
--eval-train
```

对应会额外记录：

```text
standalone_ta_eval/train_probe_loss
standalone_ta_eval/train_probe_top1
```

其中 `standalone_ta_epoch/probe_top1` 是 epoch 内 batch online probe 的平均值；`standalone_ta_eval/*_probe_top1` 是 epoch 结束后完整跑数据集得到的 linear probe 准确率，更接近 LeJEPA minimal 代码里的 epoch-end `test/acc`。

## 结果怎么看

主要看两类指标：

```text
standalone_ta_epoch/probe_top1          # 每个 epoch 都记录，表示 batch online probe 平均值
standalone_ta_eval/test_probe_top1      # 默认每 10 epoch 记录一次，表示完整 test set linear probe 准确率
```

如果这些指标明显超过当前 TagFex-LeJEPA Task0 约 74% 到 75% 的 online probe 水平，说明 TA-LeJEPA 分支本身是可以训得更好的，问题更可能出在接回完整 TagFex 后的联合训练方式。

如果 standalone 版本也长期卡在 70% 到 80%，说明瓶颈更可能在 TA-LeJEPA 自身配置，例如 backbone、augmentation、projector、`lejepa_lambda`、学习率或训练 epoch 数。

注意：这里的 probe 是训练过程中同步更新的 online linear probe。epoch-end eval 比 batch 平均更标准，但仍然不是 frozen linear eval，也不等价于官方 benchmark 中的 kNN top1。当前脚本会同时保存最后权重和按 `standalone_ta_eval/test_probe_top1` 选择的最佳权重；由于默认 `eval_every=10`，best checkpoint 只会在执行 epoch-end eval 的 epoch 上更新。
