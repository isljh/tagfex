# 2026-06-25 Task0 Online Linear Probe 诊断

## 背景

TagFex 的 LeJEPA 版本把原来的 SimCLR / InfoNCE 自监督分支替换为 LeJEPA 风格的 invariance + SIGReg 目标后，整体效果不理想。和师兄讨论后，一个可能原因是：自监督分支本身没有训练好，训练出足够可用的 representation。

为了只观察自监督分支训练状态，本次参考 LeJEPA 的 online linear probing 做法，在 Task0 训练过程中额外挂一个线性探测器。这个 probe 只作为诊断仪表，不作为正式方法改进。

## 参考实现

参考仓库：

```text
reference_repos/ijepa
reference_repos/lejepa
```

LeJEPA 的 `MINIMAL.md` 中使用：

```python
probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10))
y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
probe_loss = F.cross_entropy(yhat, y_rep)
```

其中关键点是 `emb.detach()`：probe 使用标签训练，但标签梯度不会回传到自监督 encoder / projector。

I-JEPA 官方仓库主要提供预训练代码和 linear eval 结果说明，没有在当前仓库中提供 online probe 训练脚本。因此本次主要借鉴 LeJEPA 的 online probe 形式，同时保留 I-JEPA / JEPA 评估思路：预训练表示不直接吃标签梯度，标签只用于 probe 诊断。

## 实现边界

本次只实现 Task0 online diagnostic probe：

```text
LeJEPA embedding -> detach -> LayerNorm -> Linear -> CE / top1
```

约束如下：

```text
1. 只在 self._cur_task == 0 时启用。
2. probe 不放进 self._network，避免进入主 optimizer。
3. probe 使用单独 AdamW optimizer。
4. probe 输入使用 embedding.detach().float()。
5. probe_loss 不加入主训练 loss。
6. probe 只记录训练 batch 上的诊断曲线。
7. 强制检查 features.size(0) == targets.size(0)，避免 multi-view target 对齐错误。
```

当前使用的 feature 是 `outputs["embedding"]`，即 TA feature 经过 projector 后的自监督嵌入。没有使用最终分类 logits，也没有使用主分类器 `fc` 前的 TS concat feature。

## 修改内容

代码修改：

```text
models/tagfex_lejepa.py
```

新增成员：

```text
self.linear_probe
self.linear_probe_optimizer
```

新增函数：

```text
_init_task0_online_probe(feature_dim, num_classes)
_update_task0_online_probe(features, targets)
```

Task0 `_init_train()` 中，在主模型完成正常 backward / optimizer.step / scheduler.step 之后，单独更新 probe：

```text
probe_targets = targets.repeat_interleave(V_dim)
probe_log = self._update_task0_online_probe(embedding, probe_targets)
```

这里使用 `V_dim`，因为 probe 输入是完整的 `embedding: [N * V, D]`。即使分类分支开启 `ts_global_only` 只使用 global views，probe 仍然诊断完整 LeJEPA self-supervised embedding。

## Multi-view 对齐修复

本次顺手修复了 `_flatten_augmented_inputs()` 中一个潜在错误。

原逻辑风险：

```python
if inputs.ndim == 5:
    return inputs.flatten(0, 1), targets.repeat_interleave(inputs.shape[1])
```

如果先 flatten，再读 `inputs.shape[1]`，可能把通道数 `C` 当成 view 数 `V`。当前实现改为先保存 view 数：

```python
if inputs.ndim == 5:
    num_views = inputs.shape[1]
    return inputs.flatten(0, 1), targets.repeat_interleave(num_views)
```

同时 probe 更新函数中保留强制检查：

```text
features.size(0) == targets.size(0)
```

## 配置文件

新增配置：

```text
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json
```

该配置基于：

```text
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100.json
```

只修改实验前缀并打开 probe：

```json
"prefix": "ablation_lejepa_mean_fusion_online_probe",
"online_linear_probe": true,
"probe_lr": 0.001,
"probe_weight_decay": 0.0000001
```

`probe_weight_decay=1e-7` 对齐 LeJEPA minimal 示例。

`probe_norm` 和 `probe_feature` 当前没有暴露成配置项，而是固定为：

```text
probe_norm: LayerNorm
probe_feature: embedding
```

## 运行命令

```bash
python main.py \
  --config exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json
```

## 记录指标

开启 `online_linear_probe=true` 后，Task0 每个 batch / step 额外记录：

```text
init/probe_loss
init/probe_top1
```

原有 Task0 日志仍保留：

```text
init/total_loss
init/batch_acc
init/Prediction_Invariance_loss
init/SIGReg_loss
init/LeJEPA_total_loss
init/ce_loss
init/lr
```

## 结果解释

`init/probe_top1` 是在线训练 batch 准确率，不是正式 frozen linear evaluation，也不是 test accuracy。

解释时应保持保守：

```text
probe_top1 明显上升：
  当前 LeJEPA embedding 至少包含逐渐增强的线性可分信号。

probe_top1 长期不上升：
  可以怀疑自监督 embedding 没有学到有用类别语义，但仍需要后续 offline / frozen linear probe 更严谨确认。

probe_top1 上升但主结果差：
  问题可能不在 Task0 自监督 embedding 本身，而在融合、主分类器、KD、replay 或 Task1+ 增量流程。
```

## 验证

已运行：

```bash
python -m py_compile models/tagfex_lejepa.py
python -m json.tool exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json
```


## 2026-06-25 追加：原 SimCLR / TagFex probe reference

为了给 LeJEPA online probe 曲线提供参考，本次也在普通类增量设定下的原 `models/tagfex.py` SimCLR / InfoNCE Task0 训练流程中加入同款 online linear probe。

这组实验使用普通 class-incremental ImageNet100 配置，不使用 Si-Blurry 设置。为了和 LeJEPA online probe 配置减少差异，SimCLR reference 也显式设置 `fusion_type=mean`。不过它仍然是 recipe-level reference，而不是严格公平的 SimCLR vs LeJEPA 方法对比，因为原 TagFex 和 LeJEPA 版本的主训练 recipe 仍不完全一致，包括：

```text
主模型 optimizer / learning rate / scheduler
batch size
view 数量与增强方式
自监督 loss 形式
部分训练流程实现
```

因此这组结果只能回答：

```text
在原 TagFex / SimCLR recipe 下，InfoNCE projector embedding 的在线线性可分性如何；
在当前 TagFex / LeJEPA recipe 下，LeJEPA projector embedding 的在线线性可分性如何。
```

不能直接得出“LeJEPA 方法本身一定优于或弱于 SimCLR”的结论。

### 实现保持一致的部分

为了让 probe 这个测量工具一致，SimCLR reference 使用和 LeJEPA probe 相同的设置：

```text
Task0 only
embedding.detach()
LayerNorm + Linear
单独 AdamW optimizer
probe_lr = 1e-3
probe_weight_decay = 1e-7
逐 batch / step 记录 init/probe_loss 与 init/probe_top1
```

在 SimCLR Task0 中，`inputs1` 和 `inputs2` 会拼成 `[2B]`，`targets` 也会拼成 `[2B]`。probe 直接使用 InfoNCE 同一个 `outputs["embedding"]`，因此特征数和标签数天然对齐；同时 `_update_task0_online_probe()` 仍保留 `features.size(0) == targets.size(0)` 检查。

### 新增配置

```text
exps/standard_cil/tagfex_imagenet100_online_probe.json
```

该配置基于：

```text
exps/standard_cil/tagfex.json
```

只修改实验前缀并打开 probe：

```json
"prefix": "tagfex_imagenet100_online_probe",
"online_linear_probe": true,
"probe_lr": 0.001,
"probe_weight_decay": 0.0000001,
"fusion_type": "mean"
```

### 运行命令

```bash
python main.py \
  --config exps/standard_cil/tagfex_imagenet100_online_probe.json
```

### 建议对比方式

优先比较两组训练过程中的同名曲线：

```text
SimCLR reference: init/probe_loss, init/probe_top1
LeJEPA current:   init/probe_loss, init/probe_top1
```

解释时使用保守表述：

```text
如果 SimCLR probe_top1 明显上升而 LeJEPA probe_top1 不上升：
  当前 LeJEPA 接入/训练 recipe 下的自监督 embedding 可能没有学出足够线性可分的语义。

如果两者 probe_top1 都上升，但 LeJEPA 最终结果差：
  问题可能更多来自融合、分类器、KD、replay 或 Task1+ 增量流程。
```

### 追加验证

已运行：

```bash
python -m py_compile models/tagfex.py models/tagfex_lejepa.py
python -m json.tool exps/standard_cil/tagfex_imagenet100_online_probe.json
python -m json.tool exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json
```


## 2026-06-26 追加：Task0 self-supervised continuation

### 目的

当前 LeJEPA online probe run 已完成 Task0 约 200 epoch。SwanLab 曲线显示：

```text
init/probe_loss 仍在下降
init/probe_top1 仍在上升，但结束时约 70%+
init/lr 已衰减到接近 0
```

这说明当前 LeJEPA self-supervised embedding 不是完全没有学到线性可分信息，但很可能还没有充分拟合。为了只验证 Task0 自监督分支是否需要更多训练，本次新增一个 Task0-only continuation 诊断脚本。

### 已有 Task0 checkpoint

本次 continuation 默认从下面这个已经保存的 Task0 checkpoint 继续：

```text
logs/ablation_lejepa_mean_fusion_online_probe/imagenet100_lejepa/0/10/20260625_181747/checkpoints/ablation_lejepa_mean_fusion_online_probe_1993_task_0.pth
```

注意：这个历史 checkpoint 生成时还没有保存 `linear_probe` 权重，因此 continuation 中 TA / projector 会从 checkpoint 继续，但 probe head 会重新初始化。解释 extra 阶段曲线时不要要求 `task0_extra/probe_top1` 和原 `init/probe_top1` 起点无缝衔接；更应该看 extra 后期是否能达到更高 plateau。

### 代码修改

新增脚本：

```text
analysis/task0_selfsup_continue/task0_selfsup_continue.py
```

新增配置：

```text
exps/lejepa_sigreg/task0_selfsup_continue_online_probe.json
```

同时给 `models/tagfex.py` 和 `models/tagfex_lejepa.py` 增加：

```text
get_online_probe_state()
load_online_probe_state(state)
```

并在 `trainer.py` checkpoint 保存 / 恢复中加入：

```text
online_probe_state
```

这样之后新保存的 checkpoint 会携带 linear probe 权重和 probe optimizer 状态。旧 checkpoint 没有这个字段时，会提示 probe 将重新初始化。

### Continuation 训练范围

脚本只用于 Task0 诊断，不进入 Task1，不改增量训练流程。

训练参数范围：

```text
训练：TA net + projector
冻结：TS0 / fc / aux_fc / trans_classifier / predictor 等其他模块
loss：只使用 LeJEPA self-supervised loss
probe：继续使用 embedding.detach() -> LayerNorm + Linear
```

脚本不会走 `_network.forward()` 计算完整 TS/fc，而是直接：

```text
input views -> active TA -> ta_feature -> projector -> embedding
```

这样 extra 阶段只花时间在自监督分支上。

### 默认 extra 配置

```json
"epochs": 100,
"lr": 0.0002,
"weight_decay": 0.0005,
"batch_size": 64,
"num_workers": 16,
"probe_lr": 0.001,
"probe_weight_decay": 0.0000001
```

lr 使用新的 continuation schedule，而不是沿用原 Task0 已经衰减到接近 0 的 `init/lr`。

### 运行命令

推荐直接使用 run config：

```bash
python analysis/task0_selfsup_continue/task0_selfsup_continue.py \
  --run-config exps/lejepa_sigreg/task0_selfsup_continue_online_probe.json
```

如果只想先检查脚本参数：

```bash
python analysis/task0_selfsup_continue/task0_selfsup_continue.py --help
python analysis/task0_selfsup_continue/task0_selfsup_continue.py \
  --run-config exps/lejepa_sigreg/task0_selfsup_continue_online_probe.json \
  --epochs 0 \
  --num-workers 0 \
  --no-swanlab \
  --output-dir /tmp/tagfex_task0_selfsup_continue_smoke
```

如果不想记录 SwanLab：

```bash
python analysis/task0_selfsup_continue/task0_selfsup_continue.py \
  --run-config exps/lejepa_sigreg/task0_selfsup_continue_online_probe.json \
  --no-swanlab
```

### 记录指标

extra 阶段使用新前缀，避免和原 Task0 `init/*` 曲线混在一起：

```text
task0_extra/Prediction_Invariance_loss
task0_extra/SIGReg_loss
task0_extra/LeJEPA_total_loss
task0_extra/probe_loss
task0_extra/probe_top1
task0_extra/lr
```

同时保存本地结果：

```text
analysis/task0_selfsup_continue/results/.../task0_selfsup_continue_metrics.csv
analysis/task0_selfsup_continue/results/.../task0_selfsup_continue_summary.json
analysis/task0_selfsup_continue/results/.../checkpoints/*_task_0_selfsup_continue.pth
```

结果目录已加入 `.gitignore`。

### 解释方式

如果 extra 阶段 `task0_extra/probe_top1` 继续明显上升并接近 90%+：

```text
说明 LeJEPA 分支不是学不动，而是原 200 epoch / 原 lr schedule 对自监督分支不够。
```

如果 extra 阶段 `task0_extra/LeJEPA_total_loss` 下降，但 `task0_extra/probe_top1` 仍卡住：

```text
说明 self-supervised objective 仍在优化，但 embedding 没有继续变得更线性可分，需要检查 loss 权重、views、batch size、lr 或接入方式。
```

如果 extra 阶段 loss 和 probe 都不动：

```text
说明单纯增加 epoch 可能不是主要瓶颈。
```

### 追加验证

已运行：

```bash
python -m py_compile analysis/task0_selfsup_continue/task0_selfsup_continue.py models/tagfex_lejepa.py models/tagfex.py trainer.py
python -m json.tool exps/lejepa_sigreg/task0_selfsup_continue_online_probe.json
python analysis/task0_selfsup_continue/task0_selfsup_continue.py --help
```


## 2026-06-30 修正：LeJEPA probe feature 对齐官方做法

复查 LeJEPA `MINIMAL.md` 后发现，官方 online linear probe 使用的是 encoder representation：

```python
emb, proj = net(vs)
lejepa_loss = sigreg_loss(proj) * lamb + inv_loss(proj) * (1 - lamb)
y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
```

也就是说：

```text
self-supervised loss 用 projector output / proj
linear probe 用 projector 前的 encoder feature / emb
```

此前 TagFex-LeJEPA online probe 使用的是 `outputs["embedding"] = projector(ta_feature)`，也就是 projector output。这和 LeJEPA 官方 probe 口径不一致，可能低估 encoder representation 本身的线性可分性。

本次修正为：

```text
LeJEPA self-supervised loss: 仍使用 embedding = projector(ta_feature)
LeJEPA online probe: 默认使用 ta_feature.detach()
```

新增配置项：

```json
"probe_feature": "ta_feature"
```

支持两个取值：

```text
ta_feature: projector 前的 TA encoder representation，和 LeJEPA 官方 online probe 更一致
embedding: projector output，保留作对照
```

已更新配置：

```text
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_online_probe.json
exps/lejepa_sigreg/task0_selfsup_continue_online_probe.json
```

Task0 continuation 脚本也同步改为：

```text
input views -> TA -> ta_feature -> projector -> embedding
LeJEPA loss 使用 embedding
probe 使用 probe_feature 指定的特征，默认 ta_feature
```

这次修正后，后续再看 `init/probe_top1` 或 `task0_extra/probe_top1`，口径才更接近 LeJEPA 官方文档中的 online probe。
