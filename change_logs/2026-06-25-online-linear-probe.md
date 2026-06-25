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
