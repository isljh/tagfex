# Classifier Calibration Analysis

这个目录用于分析 TagFex / LeJEPA 版本中主分类器 `fc` 是否是性能瓶颈。当前优先面向普通类增量设定，例如 `imagenet100_lejepa/0/10`。

当前脚本聚焦一个问题：

```text
在已经训练好的 Task1+ feature extractor 上，class mean 初始化主分类器后，只微调 fc，是否比随机初始化 fc 更好？
```

## 脚本

```text
class_mean_fc_finetune.py
```

它会读取已有 checkpoint，冻结除主分类器 `fc` 外的所有参数，并比较三种结果：

```text
original: checkpoint 原始 fc，直接评估
random:   随机重置 fc 后，只 finetune fc
class_mean: 用 concat TS feature 的 class mean 初始化 fc 后，只 finetune fc
```

## 数据边界

脚本不会使用旧类全量训练数据。默认用于 finetune / 计算 class mean 的数据是：

```text
旧类：checkpoint 中保存的 rehearsal memory
当前 task 新类：当前 task 全量 train 数据
```

这对应 Task1+ 训练时合法可见的数据。

## 使用示例

建议至少跑两组超参，并使用不同 `--output-dir`，避免结果互相覆盖。

### 设置 A：fc-only 快速诊断

```bash
python analysis/classifier_calibration/class_mean_fc_finetune.py \
  --config exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --checkpoint /root/autodl-tmp/tagfex_logs/ablation_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/0/10/20260603_201153/checkpoints/ablation_lejepa_mean_fusion_final_sigreg_1993_task_1.pth \
  --device 0 \
  --epochs 30 \
  --lr 1e-3 \
  --weight-decay 0 \
  --output-dir analysis/classifier_calibration/results/ablation_lejepa_mean_fusion_final_sigreg_task1_lr1e-3_wd0
```

### 设置 B：LeJEPA-aligned

LeJEPA 版本原训练使用 AdamW，`lr=5e-4`，`weight_decay=5e-4`。因此这组更贴近原训练设定。

```bash
python analysis/classifier_calibration/class_mean_fc_finetune.py \
  --config exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --checkpoint /root/autodl-tmp/tagfex_logs/ablation_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/0/10/20260603_201153/checkpoints/ablation_lejepa_mean_fusion_final_sigreg_1993_task_1.pth \
  --device 0 \
  --epochs 30 \
  --lr 5e-4 \
  --weight-decay 5e-4 \
  --output-dir analysis/classifier_calibration/results/ablation_lejepa_mean_fusion_final_sigreg_task1_lr5e-4_wd5e-4
```

### 当前已跑的混合设置

如果要复现当前 `ablation_lejepa_mean_fusion_final_sigreg_task1` 结果目录中的配置，使用：

```bash
python analysis/classifier_calibration/class_mean_fc_finetune.py \
  --config exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --checkpoint /root/autodl-tmp/tagfex_logs/ablation_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/0/10/20260603_201153/checkpoints/ablation_lejepa_mean_fusion_final_sigreg_1993_task_1.pth \
  --device 0 \
  --epochs 30 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --output-dir analysis/classifier_calibration/results/ablation_lejepa_mean_fusion_final_sigreg_task1
```

## 运行反馈

脚本会在加载 checkpoint、构建数据、抽取 concat feature、每个 finetune epoch 和评估阶段显示进度条或阶段日志，避免长时间无输出。结束时终端也会打印一张 Markdown 格式的结果表，便于直接对比 original / random / class_mean。

## 输出

如果指定 `--output-dir`，会保存：

```text
class_mean_fc_finetune_summary.json
class_mean_fc_finetune_accuracy.csv
class_mean_fc_finetune_accuracy.md
```

## 已跑结果

当前已在普通类增量 Task1 checkpoint 上完成三组 classifier-only finetune 诊断：

```text
checkpoint:
/root/autodl-tmp/tagfex_logs/ablation_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/0/10/20260603_201153/checkpoints/ablation_lejepa_mean_fusion_final_sigreg_1993_task_1.pth
```

整体 top1 对比如下：

| setting | original | random | class_mean | class_mean - random | class_mean - original |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr=1e-3, wd=0` | 81.20 | 82.10 | 82.10 | +0.00 | +0.90 |
| `lr=1e-3, wd=5e-4` | 81.20 | 82.10 | 82.10 | +0.00 | +0.90 |
| `lr=5e-4, wd=5e-4` | 81.20 | 82.00 | 82.30 | +0.30 | +1.10 |

旧类与新类分组精度如下：

| setting | branch | group_00_09 | group_10_19 |
| --- | --- | ---: | ---: |
| original | checkpoint | 73.40 | 89.00 |
| `lr=1e-3, wd=0` | random | 73.20 | 91.00 |
| `lr=1e-3, wd=0` | class_mean | 73.80 | 90.40 |
| `lr=1e-3, wd=5e-4` | random | 73.20 | 91.00 |
| `lr=1e-3, wd=5e-4` | class_mean | 73.80 | 90.40 |
| `lr=5e-4, wd=5e-4` | random | 73.00 | 91.00 |
| `lr=5e-4, wd=5e-4` | class_mean | 74.20 | 90.40 |

### 结果分析

三组结果都显示 `random` / `class_mean` finetune 后的 top1 高于原始 checkpoint fc，说明当前 Task1 的主分类器确实可能存在一定训练不足；冻结 feature extractor 后，只重新训练 `fc` 也能带来约 `+0.8` 到 `+1.1` 的 top1 提升。

但 class mean 初始化相对 random reinit 的优势不稳定。`lr=1e-3` 的两组中，`class_mean` 和 `random` 的整体 top1 持平；只有更贴近 LeJEPA 原训练设置的 `lr=5e-4, wd=5e-4` 中，`class_mean` 比 `random` 高 `+0.30`。

分组结果更有解释价值：`random` 往往更偏向新类，`group_10_19` 到 `91.00`；`class_mean` 的新类精度略低一些，但旧类 `group_00_09` 更高，最高从原始 `73.40` 提到 `74.20`。这说明 class mean 初始化可能主要是在帮助旧类权重区域校准，而不是带来整体性的大幅提升。

当前结论是：

```text
1. classifier-only finetune 有稳定收益，支持“原始 fc 可能没训够”的判断。
2. class mean 初始化本身只有弱正向证据，不足以说明它稳定优于 random reinit。
3. class mean 的潜在价值更像是改善旧类稳定性，而不是显著提升所有类别的分类能力。
```

因此，后续可以继续做 Task1 训练前 class mean 初始化实验，但预期应设为“小幅改善旧类稳定性 / 缓解新增随机权重区域”，而不是期待大幅提高整体 top1。

## 解释方式

```text
random > original:
  原始 fc 可能没训够，额外 classifier-only finetune 有帮助。

class_mean > random:
  class mean 初始化本身有帮助。

class_mean ~= random > original:
  主要收益来自只微调 fc，不一定来自 class mean 初始化。

random / class_mean 都不如 original:
  原始联合训练得到的 fc 更好，冻结特征后重训 fc 不容易。
```
