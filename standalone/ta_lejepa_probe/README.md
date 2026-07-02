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
- 可选的 split-view 输入：2 个 224 global views + 6 个 98 local views
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

原始 stacked-view 版本会把 6 个 local views resize 回 224，最终以 `[B, 8, 3, 224, 224]` 进入网络：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0.json
```

split-view 版本更接近 LeJEPA 论文/README 的 multi-crop 输入：2 个 global views 保持 224，6 个 local views 保持 98，训练时分组 forward，再在 embedding 空间拼回 8 个 view 计算 LeJEPA loss：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0_split_views.json
```

默认 stacked-view 配置文件是：

```text
exps/lejepa_sigreg/ta_lejepa_probe_task0.json
```

split-view 配置文件是：

```text
exps/lejepa_sigreg/ta_lejepa_probe_task0_split_views.json
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
sigreg_view_mode: view_wise
probe_lr: 1e-3
probe_weight_decay: 1e-7
probe_norm: layernorm
probe_view_mode: global
num_views: 8
```

split-view 版本额外使用：

```text
dataset: imagenet100_lejepa_split
num_global_views: 2
num_local_views: 6
sigreg_view_mode: view_wise
probe_view_mode: global
```

`probe_view_mode` 用来显式控制 online linear probe 的统计口径：

```text
global：只使用前 2 个 global views 的 ta_feature.detach()
all：使用全部 8 个 views 的 ta_feature.detach()
```

LeJEPA self-supervised loss 不受 `probe_view_mode` 影响，始终使用 8 个 view。为了让 stacked-view 和 split-view 的 probe_top1 可以公平比较，两个配置默认都使用 `probe_view_mode: global`。这样 stacked-view 会从 `[B, 8, 3, 224, 224]` 中只取前 2 个 global views 做 probe，split-view 会直接使用 `global` 分组做 probe。

## 已跑实验版本

SwanLab 中目前主要有下面几组 run。它们不是完全同一口径，解读时要分清楚每一组想回答的问题。

简要版本：

```text
task0_lr5e-4_lambda0.05_ep200
  stacked 输入 + mixed SIGReg + 早期 probe 口径

task0_lr5e-4_lambda0.05_ep200_view_wise
  stacked 输入 + view-wise SIGReg + 早期 probe 口径

task0_lr5e-4_lambda0.05_ep200_stacked_view_wise
  stacked 输入 + view-wise SIGReg + global-only probe

task0_lr5e-4_lambda0.05_ep200_split_views
  split 输入 + view-wise SIGReg + global-only probe

task0_lr5e-4_lambda0.05_ep400_split_viewwise_probe_global
  split 输入 + view-wise SIGReg + global-only probe，主 split-view 配置直接从头训练 400 epoch

task0_lr5e-4_lambda0.05_ep250_split_views_continue
  从第四组 ep200 last checkpoint 按原学习率 restart 继续训练 50 epoch，已跑完，用于观察高 lr restart 的影响

task0_lr5e-5_lambda0.05_ep300_split_views_continue
  从第四组 ep200 last checkpoint 用 1/10 学习率 restart 继续训练 100 epoch，用于更温和地确认 split-view 是否继续收敛
```

```text
task0_lr5e-4_lambda0.05_ep200
```

最早的 standalone TA-LeJEPA baseline。它用于确认“只拆出 TA 分支、用当前 TagFex-LeJEPA projector 和 8-view 输入时，LeJEPA 自监督分支能不能学到可分类表示”。这组实验使用 stacked 输入，local views 会被 resize 回 224；SIGReg 是 mixed 口径，也就是把 `B × 8` 个 view embedding 混在一起，当成一个整体分布做约束。早期版本的 probe 口径也可能是 all views。因此它主要作为历史参考，不建议和后续 split-view 直接比较绝对 probe_top1。

```text
task0_lr5e-4_lambda0.05_ep200_view_wise
```

在 stacked 输入不变的基础上，把 SIGReg 改成更贴近论文 Algorithm 2 的 view-wise 方式：每个 view 单独计算 SIGReg 后取平均。它主要回答“SIGReg 从 mixed 改成 view-wise 后，loss 曲线和 probe 表现是否更稳定”。这组仍然不是最终公平对比版本，因为当时 probe 口径可能还没有显式固定为 global。

```text
task0_lr5e-4_lambda0.05_ep200_stacked_view_wise
```

当前 stacked-view 对照组。它保留旧输入组织方式：2 个 global views 和 6 个 local views 都 resize/stack 成 `[B, 8, 3, 224, 224]`，但使用 `sigreg_view_mode: view_wise` 和 `probe_view_mode: global`。它用于和 split-view 版本公平比较，因为两者的 LeJEPA loss 都用 8 个 view，online probe 都只看前 2 个 global views。

```text
task0_lr5e-4_lambda0.05_ep200_split_views
```

第四组 split-view 实验。它更接近 LeJEPA 论文/README 的 multi-crop 输入：2 个 global views 保持 224，6 个 local views 保持 98，分组 forward 后在 embedding 空间拼回 8 个 view 计算 LeJEPA loss。这个 run 验证了“global/local 分组 forward”能显著缩短训练时间，并且 local views 保持 98 分辨率不会破坏 ResNet18 forward。它和 `task0_lr5e-4_lambda0.05_ep200_stacked_view_wise` 是当前最适合直接对比的一组；解读时以当前配置中的 `sigreg_view_mode` 和 `probe_view_mode` 为准。

```text
task0_lr5e-4_lambda0.05_ep400_split_viewwise_probe_global
```

第四组 split-view 主配置现在直接改为从头训练 400 epoch。它不从 ep200 checkpoint resume，而是从随机初始化开始训练，因此学习率调度是一条完整的 400 epoch warmup + cosine 曲线。这个设置用于回答“如果一开始就给 split-view 足够预算，它最终是否能追上 stacked-view”，避免 ep250/ep300 续训中的 restart 学习率策略影响判断。

```text
task0_lr5e-4_lambda0.05_ep250_split_views_continue
```

第四组的第一版续训。观察 SwanLab 曲线后发现，第四组在 200 epoch 结束时 `Prediction_Invariance_loss`、`SIGReg_loss`、`LeJEPA_total_loss` 和 `probe_loss` 仍然保持下降趋势，`probe_top1` 也还没有形成稳定平台。因此先从第四组 `last_ep200` checkpoint 继续训练 50 epoch，总训练进度到 epoch 250。这组沿用原配置中的 `lr: 5e-4` / `probe_lr: 1e-3` 重新启动 warmup + cosine 调度；实际观察到 restart 后 online probe 会有明显扰动，因此它作为高学习率 restart 的已跑结果保留。

```text
task0_lr5e-5_lambda0.05_ep300_split_views_continue
```

第四组的第二版续训。它同样从第四组 `last_ep200` checkpoint 出发，但把学习率降到原来的 1/10，即 `lr: 5e-5` / `probe_lr: 1e-4`，继续训练 100 epoch，总训练进度到 epoch 300。它用于更温和地检查 split-view 在 ep200 后是否还能继续收敛。

## 当前观察结论

1. `task0_lr5e-4_lambda0.05_ep200_view_wise` 相比 `task0_lr5e-4_lambda0.05_ep200` 准确率更高。

这说明在当前 TA-LeJEPA standalone 设置下，`view_wise` SIGReg 比 `mixed` SIGReg 更合适。也就是说，与其把 `B × 8` 个 view embedding 混在一起当成一个整体分布约束，不如对每个 view 单独计算 SIGReg 后再平均；这也更贴近 LeJEPA 论文 Algorithm 2 的写法。

2. `task0_lr5e-4_lambda0.05_ep200_split_views` 一开始相比 `task0_lr5e-4_lambda0.05_ep200_view_wise` 准确率更高，但这个对比后来发现不够干净。

第四组是在第二组的基础上，把 6 个 local views 从 resize 到 224 改成保持 98，并和 224 global views 分开 forward。这个改动显著降低了训练时间，也一开始带来了更高的 probe_top1。但后来发现第二组和第四组的 probe 输入口径不同：第二组属于早期 probe 口径，可能使用 all views；第四组使用的是 global-only probe。因此这两个 run 的准确率不能直接作为“split 输入一定更好”的证据。

3. 为了排除 probe 输入口径影响，补做了 `task0_lr5e-4_lambda0.05_ep200_stacked_view_wise`。

第三组保留 stacked 输入，但把 probe 也统一成 global-only。因此第三组和第四组都使用 `view_wise` SIGReg 和 global-only probe，区别主要是 local views 的输入方式：

```text
第三组：local views resize 到 224，和 global views 一起 stacked forward
第四组：local views 保持 98，和 global views 分组 forward
```

目前 200 epoch 观察点上，第三组结果高于第四组。但第四组的 loss 曲线在 epoch 200 仍然没有明显收敛，因此这个对比还不能作为“split-view 准确率一定低于 stacked-view”的最终结论。后续分三条线看：ep250 保留原学习率 restart 的已跑结果，ep300 使用 1/10 学习率做更温和的 restart，ep400 从头训练完整长 schedule，再判断 split-view 是否能追平或超过 stacked-view。

这两组当前主实验的对比可以回答：

```text
在相同 ResNet18 TA、projector、lr、lambda、view-wise SIGReg、global-only probe 口径下，
把 local views 从 resize 到 224 的 stacked 输入，改成保持 98 的 split 输入，
是否能降低训练时间，并改善或保持 TA 表示的 global-view 线性可分性。
```

当前 200 epoch 观察是 split-view 明显更快，但 global probe_top1 暂时低于 stacked-view。由于第四组在 200 epoch 后尚未收敛，后续分三条线看：ep250 高学习率 restart 作为已跑结果保留，ep300 低学习率 restart 用于观察温和续训，ep400 从头训练用于判断完整长 schedule 下的最终收敛。之后再决定是否需要尝试 local size 128 或 160，寻找速度和准确率之间的折中点。

## 继续训练

如果 200 epoch 结束后发现还没有收敛，可以从 `last` checkpoint 继续训练。推荐使用 `--extra-epochs`，语义是“在 checkpoint 后额外训练多少个 epoch”：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0.json \
  --resume-checkpoint standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200_stacked_view_wise/checkpoints/standalone_ta_lejepa_seed1993_task0_last_ep200.pth \
  --extra-epochs 100
```

针对第四组 split-view 实验，主 split-view 配置已直接改成从头训练 400 epoch；同时保留两组基于旧 ep200 checkpoint 的续训配置作为历史对照。

第一组是已跑完的 ep250 高学习率 restart：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0_split_views_continue_ep250.json
```

```text
resume_checkpoint: standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200_split_views/checkpoints/standalone_ta_lejepa_seed1993_task0_last_ep200.pth
extra_epochs: 50
lr: 5e-4
probe_lr: 1e-3
output_dir: standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep250_split_views_continue
run: task0_lr5e-4_lambda0.05_ep250_split_views_continue
```

第二组是新的 ep300 低学习率 restart：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0_split_views_continue_ep300.json
```

```text
resume_checkpoint: standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200_split_views/checkpoints/standalone_ta_lejepa_seed1993_task0_last_ep200.pth
extra_epochs: 100
lr: 5e-5
probe_lr: 1e-4
output_dir: standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-5_ep300_split_views_continue
run: task0_lr5e-5_lambda0.05_ep300_split_views_continue
```

第三组是主 split-view 配置从头训练到 ep400 的完整 schedule：

```bash
python standalone/ta_lejepa_probe/train.py \
  --run-config exps/lejepa_sigreg/ta_lejepa_probe_task0_split_views.json
```

```text
resume_checkpoint: none
epochs: 400
lr: 5e-4
probe_lr: 1e-3
output_dir: standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep400_split_viewwise_probe_global
run: task0_lr5e-4_lambda0.05_ep400_split_viewwise_probe_global
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
  --resume-checkpoint standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200_stacked_view_wise/checkpoints/standalone_ta_lejepa_seed1993_task0_best_test_probe.pth \
  --no-resume-reset-lr
```

继续训练时仍然会保存新的 `last` checkpoint，并按 `standalone_ta_eval/test_probe_top1` 更新 `best_test_probe` checkpoint。

## 输出文件

结果会保存在配置文件中的 `output_dir`，默认是：

```text
standalone/ta_lejepa_probe/results/ta_lejepa_probe_task0_lr5e-4_ep200_stacked_view_wise
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
run: task0_lr5e-4_lambda0.05_ep200_stacked_view_wise
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

step 级 CSV 会额外记录：

```text
input_mode          # stacked 或 split
num_views           # LeJEPA loss 使用的 view 数
num_probe_views     # online probe 使用的 view 数
probe_view_mode     # global 或 all
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

## 两阶段配置对齐说明

这里需要严格区分两个阶段，避免把 LeJEPA 自监督预训练的超参数和 linear probe / frozen linear eval 的超参数混在一起。

### 阶段 1：LeJEPA 自监督预训练

这个阶段训练的是：

```text
TA encoder / backbone + projector
```

训练目标是：

```text
LeJEPA loss = invariance loss + lambda * SIGReg loss
```

`reference_repos/lejepa/README.md` 中给出的 ResNet 预训练推荐配置是：

```text
optimizer: AdamW
lr: 5e-4
weight_decay: 5e-4 for ResNets
schedule: linear warmup + cosine annealing
final lr: initial_lr / 1000
views: 2 global views + 6 local views
全球视图: 224x224, RandomResizedCrop scale=(0.3, 1.0)
局部视图: 98x98, RandomResizedCrop scale=(0.05, 0.3)
```

当前 standalone TA-LeJEPA ResNet18 配置基本对齐这一阶段：

```text
lr: 5e-4
weight_decay: 5e-4
lejepa_lambda: 0.05
num_views: 8
sigreg_view_mode: view_wise
schedule: linear warmup + cosine annealing, eta_min = lr / 1000
```

因此，当前主训练里的 `lr` 和 `weight_decay` 指的是 LeJEPA 预训练阶段的 encoder/projector 优化参数，不是最终 linear eval 的分类头参数。

### 阶段 2：Linear Probe / Frozen Linear Eval

这个阶段用于正式评估表示质量。流程应该是：

```text
加载阶段 1 的 checkpoint
冻结 encoder / backbone
丢弃 projector 或至少不使用 projector 做分类
重新训练一个 linear classifier
报告 frozen backbone linear eval top1
```

`reference_repos/lejepa/README.md` 中 linear probe 部分的设置是：

```text
normalization: LayerNorm 或 BatchNorm，默认使用 LayerNorm
optimizer: AdamW
weight_decay: 1e-6
schedule: same as pretraining，也就是 linear warmup + cosine annealing lr: 5e-4
```

论文 Figure 9 的 INet10 结果还会对 linear eval 的 learning rate 和 weight decay 做 cross-validation，也就是冻结同一个 backbone 后，训练多组 linear classifier，选择表现最好的 lr / wd 组合报告结果。

当前 standalone 脚本里的 probe 是 online probe：

```text
probe_lr: 1e-3
probe_weight_decay: 1e-7
probe_norm: layernorm
probe_view_mode: global
```

它使用 `ta_feature.detach()`，所以不会反向影响 TA encoder / projector 的 LeJEPA 训练。它的作用是训练过程监控：观察 TA 表示是否越来越线性可分、判断大概什么时候值得保存 checkpoint。它不是论文式 frozen linear eval，也不能直接等同于 Figure 9 的最终结果。

### 当前实验应如何解读

当前 89% 到 90% 左右的 online probe top1 应理解为：

```text
TA-LeJEPA 分支已经学到了明显可分类的表示，但这只是训练过程诊断指标。
```

如果要和论文 INet10 的 91.5% 到 95% frozen linear eval 结果对标，需要额外做阶段 2：冻结已经收敛的 TA encoder，重新训练 linear classifier，并对 linear eval 的 lr / wd 做小网格搜索。只有这个 frozen linear eval 结果，才适合作为最终对标论文 Figure 9 的指标。

