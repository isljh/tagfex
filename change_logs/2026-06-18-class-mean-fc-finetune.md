# 2026-06-18 Class Mean FC Finetune 诊断

## 问题

当前分析对象是普通类增量设定下的 LeJEPA mean-fusion final-SIGReg run，例如：

```text
/root/autodl-tmp/tagfex_logs/ablation_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/0/10/20260603_201153/
```


老师指出 TagFex 的主分类器会随着 task 增加不断扩展输入维度。以 Task1 为例，`fc.weight` 从 `[C0, D]` 扩展到 `[C0 + C1, 2D]` 后，只有旧类对应旧 TS0 特征的左上区域继承自 Task0，其余区域都是随机初始化。

这可能导致分类器训练不足，尤其是在旧类只有少量 replay memory 的情况下。

## 实验思路

本次讨论后，class mean 初始化分类器被拆成两个实验版本。它们的核心区别是：计算 class mean 时，新增 TS1 是否已经经过 Task1 训练。

### 实验一：Task1 训练前 class mean 初始化

流程是：

```text
1. 从 Task0 checkpoint resume
2. 进入 Task1
3. update_fc，生成 warm-start TS1 和扩展后的 fc
4. 用 Task0 memory + Task1 全量训练数据提取 concat feature
5. 计算 class mean
6. 用 class mean 初始化 fc.weight 中原本随机的区域
7. 正常训练 Task1
8. 和原始 Task1 结果对比
```

这个实验最贴合老师提出的问题：Task1 训练开始时，`fc` 新增随机区域是否应该用 class mean 初始化。

但它有一个关键风险：Task1 刚 `update_fc` 后，TS1 还没有经过 Task1 训练。虽然当前代码中的 TS1 不是完全随机，而是由 TS0 和 TA 插值得到：

```text
TS1 ~= init_interpolation_factor * TS0 + (1 - init_interpolation_factor) * TA
```

默认 `init_interpolation_factor = 0.95`，所以 TS1 是 warm-start expert。但它还没学过 Task1 新类，因此此时的 1024 维 concat feature：

```text
concat(TS0(x), TS1(x))
```

里面 TS1 对应的 512 维还不一定有很强的类别判别性。用这个未成熟特征计算 class mean，质量可能有限。

### 实验二：已有 Task1 checkpoint 上做 classifier-only finetune

当前优先实现实验二。流程是：

```text
1. 加载已经训练好的 Task1 checkpoint
2. 冻结 TS / TA / projector / predictor 等 feature extractor
3. 用训练后的 TS0 + TS1 提取 1024 维 concat feature
4. 计算每个 seen class 的 class mean
5. 用 class mean 初始化 fc
6. 只重新训练主分类器 fc
7. 对比 finetune 前后，以及不同初始化方式
```

这个实验的动机是先绕开实验一中“TS1 还没训练，class mean 不可靠”的问题。Task1 checkpoint 中的 TS1 已经经过 Task1 训练，因此它提取的 1024 维 concat feature 更成熟，用它计算 class mean 意义更强。

它回答的问题是：

```text
在训练后的 concat feature 空间里，class mean 对分类器是否有潜在帮助？
```

如果实验二都没有效果，说明即使 TS1 已经训练好，1024 维特征空间中的 class mean 也不能帮助分类器，那么实验一大概率更难有效。

如果实验二有效，再做实验一就更有支撑：既然训练后的特征空间 class mean 有用，下一步再验证是否能把它前移到 Task1 训练前作为初始化策略。

### 两个实验的关系

| 实验 | TS1 状态 | class mean 的含义 | 主要问题 |
| --- | --- | --- | --- |
| 实验一：Task1 训练前初始化 | TS1 未训练，只是 warm-start | 未成熟特征上的 class mean | 能不能作为训练前初始化 |
| 实验二：Task1 checkpoint 后 classifier-only finetune | TS1 已训练 | 成熟 concat feature 上的 class mean | class mean 对分类器是否有潜在价值 |

本次新增脚本实现的是实验二。

## 当前实现

本次先不改主训练流程，而是在 `analysis/` 下加入一个事后诊断脚本：

```text
analysis/classifier_calibration/class_mean_fc_finetune.py
```

它读取已有 Task1+ checkpoint，冻结 TS / TA / projector 等 feature extractor，只重新初始化并微调主分类器 `fc`。

脚本比较三组结果：

```text
original   checkpoint 原始 fc，直接评估
random     随机重置 fc 后，只 finetune fc
class_mean 用 concat TS feature 的 class mean 初始化 fc 后，只 finetune fc
```

之所以必须有 `random` 对照，是因为 classifier-only finetune 如果提升了，可能只是额外训练 `fc` 带来的，不一定来自 class mean 初始化本身。

## 数据边界

用于计算 class mean 和 finetune fc 的数据为：

```text
旧类：checkpoint 中保存的 rehearsal memory
当前 task 新类：当前 task 的全量 train 数据
```

不使用旧类全量训练数据，不修改 checkpoint，也不修改 replay memory。

## 修改内容

新增文件：

```text
analysis/classifier_calibration/class_mean_fc_finetune.py
analysis/classifier_calibration/README.md
```

更新文件：

```text
analysis/README.md
```

其中 `analysis/README.md` 增加了 classifier calibration 入口、实验目的、使用示例和结果解释。

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

## 结果解释

```text
random > original
```

说明原始 fc 可能没训够，额外 classifier-only finetune 有帮助。

```text
class_mean > random
```

说明 class mean 初始化本身有帮助。

```text
class_mean ~= random > original
```

说明主要收益可能来自只微调 fc，不一定来自 class mean 初始化。

```text
random / class_mean 都不如 original
```

说明冻结 feature extractor 后重训 fc 不容易，原始联合训练得到的 fc 更好。

## 2026-06-21 追加：Task1 训练前 class mean 初始化实验

基于 classifier-only finetune 的结果，新增了一个真正接入训练流程的实验开关，用来验证：

```text
在 Task1 update_fc 扩展完分类器之后、正式训练 Task1 之前，
能否用当前可见数据的 concat TS feature class mean 初始化 fc 中原本随机的区域。
```

### 修改位置

代码修改在：

```text
models/tagfex_lejepa.py
```

新增实验配置为：

```text
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg_class_mean_fc_init.json
```

该配置从 LeJEPA mean-fusion final-SIGReg baseline 复制而来，只修改实验 `prefix`、写入 Task0 `resume` 路径，并打开 class mean fc init 相关参数。原始 baseline 配置不变。


本次实验直接从已经训练好的 Task0 checkpoint resume，不重新训练 Task0：

```text
/root/autodl-tmp/tagfex_logs/ablation_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/0/10/20260603_201153/checkpoints/ablation_lejepa_mean_fusion_final_sigreg_1993_task_0.pth
```

### 插入时机

新增初始化发生在 `incremental_train()` 中：

```text
1. 进入新 task
2. ptr.update_fc(self._total_classes)
3. 构建当前 task train_dataset
   - 旧类：rehearsal memory
   - 新类：当前 task train data
4. 如果 class_mean_fc_init=true：
   - 用 train_dataset 建一个 drop_last=False 的 init_loader
   - eval 模式提取当前 concat TS feature
   - 计算 seen classes 的 class mean
   - 写回 fc.weight 指定区域
5. 再构建正常 train_loader 并开始 Task1 训练
```

注意这里使用的是训练开始前的 TS 状态。Task1 的新 TS1 还没有经过 Task1 训练，只是由 TS0 和 TA 按 `init_interpolation_factor` warm-start 得到。因此这个实验回答的是“未成熟但 warm-start 的 concat feature class mean 是否能改善训练起点”，不同于前面的 Task1 checkpoint 后 classifier-only finetune。

### 新增参数定义

本次新增四个配置参数：

```json
"class_mean_fc_init": true,
"class_mean_fc_init_mode": "partial_new_regions",
"class_mean_fc_init_scale": "global_weight_norm",
"class_mean_fc_init_bias": "zero"
```

#### `class_mean_fc_init`

总开关。

```text
true:  在 update_fc 之后、正式训练当前 task 之前执行 class mean fc 初始化。
false: 保持原始逻辑，fc 新增区域仍然使用随机初始化。
```

当前实现中 Task0 会自动跳过该初始化，因为 Task0 没有旧分类器扩展问题。

#### `class_mean_fc_init_mode`

控制写入 `fc.weight` 的哪些区域。当前实现支持：

```text
partial_new_regions
all_seen
new_classes
```

本次实验默认使用：

```text
partial_new_regions
```

以 Task1 为例，`fc.weight` 可按类别和特征来源拆成四块：

```text
                      TS0 特征列       TS1 特征列
旧类 0-9              A               B
新类 10-19            C               D
```

原始 `update_fc` 逻辑是：

```text
A = 继承 Task0 fc 权重
B = 随机初始化
C = 随机初始化
D = 随机初始化
```

`partial_new_regions` 的逻辑是：

```text
A = 保留 Task0 继承权重，不覆盖
B = 用旧类 class mean 在 TS1 特征维度上的部分初始化
C = 用新类 class mean 在 TS0 特征维度上的部分初始化
D = 用新类 class mean 在 TS1 特征维度上的部分初始化
```

也就是说，它只替换原本随机初始化的区域，避免破坏已经学过的旧类-旧特征权重。

另外两个模式用于后续 ablation：

```text
all_seen:    所有 seen classes 的整行 fc.weight 都替换为 class mean。
new_classes: 只替换当前 task 新类行，旧类行完全不动。
```

#### `class_mean_fc_init_scale`

控制 class mean 写入 `fc.weight` 前的尺度。当前实现支持：

```text
none
global_weight_norm
per_class_weight_norm
```

本次实验默认使用：

```text
global_weight_norm
```

计算方式与 classifier calibration 脚本对齐：

```text
1. 对每个样本 feature 做 L2 normalize。
2. 对每个类别求 mean。
3. 再对类别 mean 做 L2 normalize，得到类别方向。
4. 乘上当前 fc.weight 的平均 L2 norm。
```

直观含义是：

```text
方向来自 class mean，整体尺度接近当前分类器权重。
```

这样可以避免直接把 feature mean 写入分类器导致 logit 尺度过大或过小。

其他模式含义：

```text
none:                 只使用 normalize 后的 class mean 方向，不额外匹配 fc 权重尺度。
per_class_weight_norm: 每个类别使用当前对应 fc.weight 行的范数进行缩放。
```

#### `class_mean_fc_init_bias`

控制被初始化区域对应的 `fc.bias` 如何处理。当前实现支持：

```text
zero
keep
```

本次实验默认使用：

```text
zero
```

含义是：对被 class mean 初始化覆盖到的类别 bias 置 0。因为当前采用的是“类别方向初始化”，不是严格 NME 线性化，所以 bias 置 0 更保守。

`keep` 则保留 `update_fc` 后的 bias 状态。

### 当前默认实验命令

```bash
python main.py \
  --config exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg_class_mean_fc_init.json

python main.py \
  --config exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg_class_mean_fc_init.json
```

该实验应与原 baseline 对比：

```text
baseline config:
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json

class mean fc init config:
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg_class_mean_fc_init.json
```

重点观察：

```text
1. Task1 top1 是否提升。
2. 旧类 group_00_09 是否比 baseline 更稳定。
3. 新类 group_10_19 是否出现明显下降。
4. Task2+ 是否有累积副作用。
```

当前预期不是大幅提高整体 top1，而是验证它能否小幅改善旧类稳定性、缓解新增随机权重区域带来的训练不足。

实现中额外处理了 DDP 边界：如果 `torch.distributed` 已初始化，只由 rank0 计算 class mean 并写入 `fc`，随后 broadcast `fc.weight` / `fc.bias` 到其他 rank，避免不同进程因随机增强导致初始化不一致。

## 验证

已运行：

```bash
python -m py_compile analysis/classifier_calibration/class_mean_fc_finetune.py
python analysis/classifier_calibration/class_mean_fc_finetune.py --help
python -m py_compile models/tagfex_lejepa.py
python -m json.tool exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg_class_mean_fc_init.json
```
