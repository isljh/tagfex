# 分析工具

这个文件夹专门存放和训练流程分离的分析脚本与说明文档。这里的脚本用于读取已经训练好的 checkpoint 做事后诊断，不应该改变正常训练代码。

## 目录结构

```text
oracle_nme/            TA 特征或 projector embedding 的 Oracle NME 上界分析
sigreg_projection/     SIGReg 投影矩阵与 embedding 高斯性诊断
feature_distribution/  早期 embedding / Pre-ReLU 特征分布可视化
ideas/                 暂未验证的未来研究想法
```

每个子目录都有自己的 `README.md`，说明该分析任务的目标、脚本和典型用法。

## 2026-06-15：TA 特征 Oracle NME 上界分析

### 修改目的

当前想判断的问题是：

> LeJEPA 训练出来的 TA 分支特征，本身到底有没有较好的类别区分能力？

完整的 SiBlurry 结果会混入很多因素，例如 task-specific experts、fusion、KD、transfer、memory、分类头和类别不均衡。因此先加入一个更直接的诊断：冻结已训练模型，只看 TA 特征空间本身。

第一版先实现 **测试集原型 Oracle NME**。

### 分析逻辑

脚本执行流程如下：

1. 读取实验配置文件。
2. 读取保存好的 checkpoint。
3. 根据 checkpoint 所在 task 重建模型结构。
4. 加载 checkpoint 中的模型权重。
5. 对测试集图片提取 TA 特征。
6. 使用测试集真实标签，为每个类别计算一个测试集类别原型。
7. 对每个测试样本，计算它到所有类别原型的距离。
8. 选择最近的类别原型作为预测结果。
9. 输出 Oracle NME accuracy。

### 指标含义

这个指标可以叫：

- Test-prototype Oracle NME
- Oracle class-mean NME
- 测试集原型 NME
- 标签泄露的 NME 上界

它回答的是：

> 如果给模型最理想、最贴近测试分布的类别原型，这个 TA 特征空间最多能把类别分到什么程度？

### 重要提醒

这个结果 **不能作为正式测试准确率汇报**。

原因是类别原型由测试集真实标签计算得到，测试标签信息已经泄露进分类器。因此它只能作为诊断上界，用来判断特征空间本身是否有区分能力。

推荐汇报表述：

> Test-prototype Oracle NME 使用测试集标签构建类别原型，用于估计 TA 特征空间在理想原型条件下的分类上界。该指标存在标签泄露，不能作为标准测试准确率。

### 当前文件

- `oracle_nme/ta_oracle_nme.py`：计算 TA 特征或 projector embedding 的测试集原型 Oracle NME。

### 使用示例

评估 TA backbone 输出的原始特征：

```bash
python analysis/oracle_nme/ta_oracle_nme.py \
  --config exps/si_blurry/si_blurry_tagfex_lejepa_imagenet100_final_sigreg.json \
  --checkpoint /path/to/checkpoint_task_0.pkl \
  --feature ta_feature \
  --device 0
```

评估 projector 后的 embedding：

```bash
python analysis/oracle_nme/ta_oracle_nme.py \
  --config exps/si_blurry/si_blurry_tagfex_lejepa_imagenet100_final_sigreg.json \
  --checkpoint /path/to/checkpoint_task_0.pkl \
  --feature embedding \
  --device 0
```

保存预测结果、标签和类别原型：

```bash
python analysis/oracle_nme/ta_oracle_nme.py \
  --config exps/si_blurry/si_blurry_tagfex_lejepa_imagenet100_final_sigreg.json \
  --checkpoint /path/to/checkpoint_task_0.pkl \
  --feature ta_feature \
  --device 0 \
  --output-dir /path/to/output_dir
```


## 2026-06-15：SiBlurry LeJEPA Task0 Oracle NME 运行记录

### 分析对象

Checkpoint：

```text
/root/autodl-tmp/TagFex_SiBlurry/si_blurry_lejepa_mean_fusion_final_sigreg/imagenet100_lejepa/si_blurry/tasks5_n50_m10/20260603_215147/checkpoints/si_blurry_lejepa_mean_fusion_final_sigreg_1993_task_0.pth
```

配置文件：

```text
exps/si_blurry/si_blurry_tagfex_lejepa_imagenet100_final_sigreg.json
```

### 重要说明

该 checkpoint 的 SiBlurry Task0 不是固定 10 类。日志和 checkpoint 元数据均显示：

```text
task = 0
total_classes = 38
task_increments = [38, 13, 13, 26, 10]
```

因此本次 Task0 Oracle NME 评估的是 mapped class `0-37`，共 38 类，测试样本数为 1900。

### 结果

TA backbone 原始特征 `ta_feature`：

```text
Oracle NME top1 total = 33.16
top5 = 64.16
disjoint_only = 37.62
blurry_exposed = 29.91
```

Projector 后的 `embedding`：

```text
Oracle NME top1 total = 28.21
top5 = 60.74
disjoint_only = 29.50
blurry_exposed = 27.27
```

### 初步观察

`embedding` 的 Oracle NME 低于 `ta_feature`，说明经过 projector/LeJEPA 自监督空间后，类别原型结构没有变好，反而更不利于最近类中心分类。

同时，即使使用测试集标签构建 oracle prototype，`ta_feature` 的 top1 也只有 33.16。这说明在该 SiBlurry Task0 的 38 类测试空间里，LeJEPA TA 特征本身的类别可分性较弱。

注意：该结果是标签泄露的上界诊断，不能作为正式测试准确率。


## 2026-06-15：SiBlurry 原版 TagFex Task0 Oracle NME 对比

### 分析对象

原版 TagFex checkpoint：

```text
/root/autodl-tmp/TagFex_SiBlurry/si_blurry_tagfex/imagenet100_aa/si_blurry/tasks5_n50_m10/20260602_170711/checkpoints/si_blurry_tagfex_imagenet100_1993_task_0.pth
```

配置文件：

```text
exps/si_blurry/si_blurry_tagfex_imagenet100_aa.json
```

### Split 确认

原版 TagFex 日志显示：

```text
Learning on 0-38
new(mapped) = [0, 1, ..., 37]
session(mapped) = [0, 1, ..., 37]
```

因此原版 TagFex 与 LeJEPA 的 Task0 Oracle NME 都是在 SiBlurry Task0 的 38 个 mapped 类上评估，测试样本数均为 1900。

### 原版 TagFex 结果

TA backbone 原始特征 `ta_feature`：

```text
Oracle NME top1 total = 70.05
top5 = 91.74
disjoint_only = 79.12
blurry_exposed = 63.45
```

Projector 后的 `embedding`：

```text
Oracle NME top1 total = 69.00
top5 = 93.37
disjoint_only = 80.12
blurry_exposed = 60.91
```

### 与 LeJEPA 对比

| 方法 | 特征空间 | Oracle NME top1 | top5 | disjoint_only | blurry_exposed |
| --- | --- | ---: | ---: | ---: | ---: |
| 原版 TagFex | ta_feature | 70.05 | 91.74 | 79.12 | 63.45 |
| 原版 TagFex | embedding | 69.00 | 93.37 | 80.12 | 60.91 |
| LeJEPA | ta_feature | 33.16 | 64.16 | 37.62 | 29.91 |
| LeJEPA | embedding | 28.21 | 60.74 | 29.50 | 27.27 |

### 初步结论

在同一个 SiBlurry Task0 的 38 类测试集上，即使使用测试集标签构建 oracle prototype，LeJEPA 的 TA 特征上界也明显低于原版 TagFex。

这说明 LeJEPA 在当前设置下的问题不只是分类头、memory、KD 或增量过程导致的；至少在 Task0 阶段，它的 TA 表征本身已经缺少足够的类别可分性。

另外，LeJEPA 的 `embedding` 低于自身 `ta_feature`，而原版 TagFex 的 `embedding` 与 `ta_feature` 接近。这进一步说明 LeJEPA 的 projector/自监督空间没有形成更适合类别原型分类的结构。

注意：所有 Oracle NME 都是测试集标签泄露的上界诊断，不能作为正式测试准确率。


## 2026-06-15：LeJEPA 接入问题记录：SIGReg 的 view 维度被混合

### 问题描述

原版 LeJEPA 的 SIGReg 会保留 view 维度。其 projector 输出通常组织为：

```text
[V, N, D]
```

其中：

```text
V = view 数量，例如 8
N = batch 内原始图片数量
D = embedding 维度
```

在这种组织方式下，SIGReg 是对每个 view 下的 batch 特征分布分别做高斯正则，然后再对不同 view 的正则结果求平均。

也就是说，原版更接近：

```text
view 1 的 N 个样本 -> 统计一个特征分布
view 2 的 N 个样本 -> 统计一个特征分布
...
view 8 的 N 个样本 -> 统计一个特征分布
```

### 当前实现

当前 `models/tagfex_lejepa.py` 中先把 8 个 view 展平成一个大 batch：

```python
out = self._network(vs.flatten(0, 1))
embedding = out["embedding"]
sigreg_loss = self.sig_reg(embedding)
```

此时 `embedding` 的形状是：

```text
[N * V, D]
```

因此当前实现等价于：

```text
把所有图片的所有 view 混在一起，只统计一个整体特征分布
```

而不是：

```text
每个 view 分别统计一个特征分布
```

### 准确表述

可以这样描述这个问题：

> 原版 LeJEPA 在 SIGReg 中保留 view 维度，输入形状为 `[V, N, D]`，因此它对每个 view 下的 batch 特征分布分别做高斯正则，再对不同 view 的正则结果求平均。而当前实现把 8 个 view 展平成 `[N*V, D]` 后送入 SIGReg，相当于只约束所有 view 混合后的整体分布，丢失了 view-wise distribution regularization。

一句话版本：

> 当前实现不是看 8 个 view 的 8 个分布，而是把 8 个 view 混在一起，只看 1 个混合分布。

### 可能影响

global view 和 local view 的裁剪尺度不同，特征分布可能并不一致。如果把它们混在一起做 SIGReg，正则目标可能会变得不清晰，从而干扰 TA 表征学习。

这可能是 LeJEPA 在 Task0 oracle NME 中明显低于原版 TagFex 的原因之一。

### 后续 ablation 建议

优先尝试把 SIGReg 改成 view-wise 形式：

```python
proj = embedding.reshape(N, V_dim, -1).transpose(0, 1)  # [V, N, D]
sigreg_loss = self.sig_reg(proj)
```

或者更保守地先只在 global views 上做 LeJEPA/SIGReg，避免 6 个 local views 和 global views 的分布混合。


## 2026-06-15：LeJEPA 低 Oracle NME 的可能主因：空间张开但类别边界不清晰

### 核心想法

当前更合理的主因仍然是：

> LeJEPA 的 SIGReg 可以防止特征整体坍缩，让特征空间更分散；但它不显式要求不同类别之间拉开边界。因此特征空间可能是“张开的”，但不是“按类别张开的”。

原版 TagFex 使用 InfoNCE：

```text
同一张图的两个 view 被拉近
batch 里其他图片作为负样本被拉远
```

因此原版 InfoNCE 不只是学习 view invariance，也会隐式把不同样本、不同类别的表示拉开。

当前 LeJEPA 使用：

```text
同一张图的多个 view 被拉近
SIGReg 防止整体坍缩、约束分布形状
```

但它没有明确告诉模型：

```text
不同类别应该分开
```

所以在 SiBlurry 这种类别暴露复杂、Task0 类别数较多的设定里，LeJEPA 可能得到一个整体分散但类别边界不清楚的表示空间，导致 NME 特别低。

### 关于 blurry / 混入类样本数的修正

之前把 Task0 中的 blurry 类统一说成“不是少样本类”，这个表述不够准确。SiBlurry Task0 里需要区分两种情况：

1. **本任务分配到的 blurry 类**：样本数并不少。
2. **从其他 task 的 blurry pool 提前混入当前 task 的外来类**：样本数非常少。

Task0 总训练样本数：

```text
42747
```

Task0 session 中实际出现 38 个 raw 类，其中：

```text
home disjoint 类：16 个
home blurry 类：17 个
foreign mixed-in 类：5 个
```

本任务 disjoint / blurry 类样本数：

```text
home disjoint 平均样本数 = 1291.56
home disjoint min/max = 1165 / 1300

home blurry 平均样本数 = 1296.82
home blurry min/max = 1264 / 1300
```

提前混入的外来类样本数：

```text
class 6  = 2 张
class 10 = 12 张
class 15 = 8 张
class 40 = 8 张
class 42 = 6 张

foreign mixed-in 总样本数 = 36
foreign mixed-in 平均样本数 = 7.2
foreign mixed-in min/max = 2 / 12
```

因此更准确的结论是：

> Task0 中本任务自己的 blurry 类不是少样本类；但从混合池提前暴露进来的外来 blurry 类确实是极少样本类。

### 对当前 Oracle NME 的影响

这件事会增加 SiBlurry Task0 的难度，因为模型训练时几乎没见过那些外来混入类，却要在测试时区分它们。

但需要注意：当前做的是 **test-prototype Oracle NME**，类别原型由测试集标签直接计算，因此 prototype 本身不受训练样本少的影响。

也就是说，外来混入类样本少会影响模型学到的特征，但不会导致 oracle prototype 估计不准。

另外，外来混入类只有 5 个，共 36 张训练样本。它们是一个额外困难因素，但不足以单独解释 LeJEPA 与原版 TagFex 的巨大差距。

### 与结果的对应

原版 TagFex：

```text
ta_feature Oracle NME top1 = 70.05
disjoint_only = 79.12
blurry_exposed = 63.45
```

LeJEPA：

```text
ta_feature Oracle NME top1 = 33.16
disjoint_only = 37.62
blurry_exposed = 29.91
```

在同样的 38 类 Task0、同样的测试集 oracle prototype 条件下，LeJEPA 明显低于原版 TagFex。这说明问题更可能来自 LeJEPA 表征本身的类别可分性不足，而不是单纯由 Task0 样本数量不足造成。

### 推荐表述

可以这样总结：

> SiBlurry Task0 中确实存在少量从其他 task 提前混入的外来 blurry 类，这些类在当前 task 中样本极少，会增加训练难度。但本任务自身的 blurry 类并不是少样本类，而且当前使用的是 test-prototype Oracle NME，原型由测试集标签计算，不受训练样本数影响。因此 LeJEPA 显著低于原版 TagFex，更主要的原因仍可能是 LeJEPA 缺少 InfoNCE 的负样本机制：它能防止整体坍缩，却不显式拉开不同类别边界，导致特征空间张开但类别结构不清晰。


## 2026-06-15：后续问题：NME 低是否代表线性分类也低？

### 问题背景

当前 Oracle NME 结果显示 LeJEPA 明显低于原版 TagFex。但 NME 和 linear probe 衡量的特征性质并不完全一样。

NME 依赖类别原型：

```text
每一类计算一个 class mean
测试样本找最近 class mean
```

因此 NME 更要求特征空间满足：

```text
类内紧凑
类间中心分离
```

如果某个类别的特征分布很散、不是单中心结构，或者一个类被分成多个子簇，即使整体仍有一定可分性，NME 也可能较低。

Linear probe 则是训练一个线性分类器：

```text
feature -> linear classifier -> class
```

它不要求每个类别都围绕一个中心聚好，只要求不同类别可以被线性边界分开。

### 可能情况

因此存在一种可能：

```text
LeJEPA 的 NME 低
但 linear probe 准确率相对不低
```

这说明 LeJEPA 特征可能不是没有分类信息，而是其类别结构不适合最近类中心分类。

也可能出现：

```text
LeJEPA 的 NME 低
linear probe 准确率也低
```

这说明 LeJEPA 的 TA 表征本身分类判别性确实弱。

### 当前结果的含义

需要注意的是，目前使用的是 test-prototype Oracle NME，也就是用测试集标签构建了相对理想的类别原型。即使在这个上界条件下，LeJEPA 的 `ta_feature` 也只有：

```text
Oracle NME top1 = 33.16
```

这说明 LeJEPA 当前特征的类中心结构确实较差。

但它仍然不能完全等价于 linear probe 结果。因此下一步需要继续验证线性可分性。

### 下一步实验建议

冻结训练好的 TA backbone，只训练一个线性分类器：

```text
image -> TA backbone -> ta_feature -> linear classifier -> class
```

分别比较：

```text
原版 TagFex TA feature + linear probe
LeJEPA TA feature + linear probe
```

可选地也比较 projector 后的 embedding：

```text
原版 TagFex embedding + linear probe
LeJEPA embedding + linear probe
```

### 推荐表述

> 当前 Oracle NME 说明 LeJEPA 特征不具备良好的类别原型结构，但 NME 低不一定等价于线性分类能力低。下一步需要通过冻结 TA backbone 并训练 linear probe，判断 LeJEPA 表征是仅仅不适合 NME，还是分类判别性本身也不足。


# 讨论后的结果
## 老师想法
### 分类器问题

老师指出，当前 TagFex 的主分类器组织形式比较笨重。它不是在一个固定特征空间上维护分类头，而是每来一个 task 就新增一个 task-specific expert，然后把所有 TS expert 的输出特征拼接起来做分类。

对应到代码中，`TagFexNet.feature_dim` 是：

```python
return self.out_dim * len(self.convnets)
```

因此随着 task 增加，主分类器 `fc` 的输入维度会不断变大：

```text
task 0: [TS0 feature] -> C0 类
task 1: [TS0 feature, TS1 feature] -> C0 + C1 类
task 2: [TS0 feature, TS1 feature, TS2 feature] -> C0 + C1 + C2 类
...
```

同时输出维度也会随着累计类别数不断变大。`update_fc` 中每次都会重新生成一个更大的 `fc`：

```python
fc = self.generate_fc(self.feature_dim, nb_classes)
```

如果从 task t-1 到 task t 来看，新的分类权重矩阵可以理解成四个区域：

```text
                  旧 TS 特征列        新 TS 特征列
旧类别行        继承旧分类器权重      随机初始化
新类别行        随机初始化            随机初始化
```

代码里只有左上角被显式继承：

```python
fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
fc.bias.data[:nb_output] = bias
```

也就是说：

1. 旧类别对旧 TS 特征的分类权重被保留。
2. 旧类别对新 TS 特征的权重是新初始化的。
3. 新类别对旧 TS 特征的权重是新初始化的。
4. 新类别对新 TS 特征的权重也是新初始化的。

这样会带来几个问题：

1. **分类器参数越来越大**：task 越多，concat 后的特征越长，`fc` 权重矩阵也越大。
2. **新增参数主要依赖当前 task 训练**：除左上角外，其余区域都需要通过当前 task 的新类样本和少量旧类 memory 学出来。
3. **旧类回放样本越来越少**：总 memory 固定时，类别数越多，每类 exemplar 越少，导致旧类别相关的新权重区域很难充分训练。
4. **没有充分利用每个 TS expert 的输出结构**：所有 TS 特征只是简单 concat 后交给一个越来越大的线性分类器，分类器需要自己学会“哪个 expert 对哪个类别/task 有用”。

因此老师认为，这种分类方式比较重，而且不够自然。尤其是在 SiBlurry 里类别暴露本身已经复杂，分类头再不断扩张，会进一步增加训练难度。

#### 和当前 Oracle NME 结果的关系

这条反馈和目前的 Oracle NME 诊断可以形成两个层次的解释：

1. **分类器层面**：当前主分类头确实可能有结构性问题。它随 task 扩展输入和输出维度，大量新权重依赖有限 replay 学习，容易造成分类器训练不充分或偏向新类。
2. **表征层面**：但 Task0 的 Oracle NME 已经显示，LeJEPA 的 TA 特征本身类别结构较弱。Task0 时还没有多 task 分类头膨胀问题，LeJEPA `ta_feature` 的 test-prototype Oracle NME top1 只有 33.16，而原版 TagFex 是 70.05。

所以更准确的判断是：

> 分类器结构确实是一个后续 task 中会放大的问题；但 LeJEPA 当前的主要短板不只在分类器。至少在 Task0 阶段，TA 表征本身的类别可分性已经明显弱于原版 TagFex。

#### 后续实验方向

为了把老师提出的分类器问题单独验证出来，可以设计几组更干净的实验：

1. **TA-only linear classifier**

冻结 TA backbone，只在 `ta_feature` 上训练一个固定输入维度的线性分类器：

```text
image -> TA backbone -> ta_feature -> linear classifier
```

这样可以绕开 TS concat 分类器，直接判断 TA 表征是否线性可分。

2. **Prototype / NME classifier**

不用不断扩大的 `fc`，改用每类原型分类：

```text
image -> feature -> nearest class prototype
```

可以测试“分类头参数膨胀”去掉后，性能是否改善。

3. **Per-expert classifier 或 expert-wise logits fusion**

不要把所有 TS 特征直接 concat 到一个大分类器，而是让每个 TS expert 先得到自己的 logits，再做融合：

```text
TS0 feature -> classifier0 -> logits0
TS1 feature -> classifier1 -> logits1
...
fuse(logits0, logits1, ...)
```

这样可以更显式地利用每个 TS expert 的输出，而不是让一个大矩阵隐式学习 expert 分工。

4. **固定维度融合后分类**

先把多个 TS expert 的特征融合回固定维度，再接分类器：

```text
[TS0, TS1, ..., TSt] -> fusion -> D-dim feature -> classifier
```

这样 `fc` 的输入维度不再随 task 增长，只让输出类别数增长，能减少新增随机权重区域。

### 推荐汇报表述

> 老师指出当前 TagFex 的主分类器随着 task 增加会同时扩展输入维度和输出维度。新分类器中只有旧类别-旧特征对应的左上角权重继承自上一阶段，其余涉及新 TS 特征或新类别的权重都需要重新学习。但随着类别数增加，每类 replay 样本越来越少，这会让新增权重区域训练不足，也没有很好利用每个 TS expert 的输出结构。因此分类头本身可能是一个重要瓶颈。与此同时，我们的 Task0 Oracle NME 结果说明 LeJEPA 在还没有分类头膨胀之前，TA 表征类别可分性已经明显弱于原版 TagFex，所以后续需要同时区分“表征弱”和“分类器组织不合理”这两个问题。

#### 下一步优先实验1：class mean 初始化分类器权重

老师提出的第一个可做方向是：

> 当前主分类器是线性分类器，可以用每个类别的 class mean 来初始化分类器权重，而不是让新增权重保持随机初始化。

以 task1 为例，当前有两个 TS expert：

```text
TS0 feature = 512 维
TS1 feature = 512 维
concat feature = 1024 维
```

如果累计类别数是 20 类，那么线性分类器可以写成：

```text
x: 1024 x 1
W: 20 x 1024
b: 20 x 1
logits = W x + b
```

在 PyTorch 中，`fc.weight` 的实际形状是：

```text
[num_classes, feature_dim] = [20, 1024]
```

也就是每一行对应一个类别的分类方向：

```text
logit_c = W[c] · x + b[c]
```

因此，用 class mean 初始化分类器，就是对每个类别 c 计算当前 concat 特征空间里的类别均值：

```text
mu_c = mean(concat(TS0(x_i), TS1(x_i))), label_i = c
```

然后令：

```text
fc.weight[c] = mu_c
fc.bias[c] = 0
```

更稳妥的版本可以先对 `mu_c` 做 L2 normalize，再写入 `fc.weight[c]`，避免不同类别特征范数差异直接影响 logit 尺度。


#### 为什么 class mean 可以用来初始化线性分类器

这个想法的理论依据来自 NME 和线性分类器之间的关系。NME 的预测规则是：

```text
pred = argmin_c ||x - mu_c||^2
```

其中 `x` 是样本特征，`mu_c` 是第 c 类的 class mean。把距离展开：

```text
||x - mu_c||^2 = ||x||^2 - 2 x · mu_c + ||mu_c||^2
```

对同一个样本 `x` 来说，`||x||^2` 对所有类别都一样，因此比较类别时可以忽略。于是 NME 等价于：

```text
pred = argmax_c 2 x · mu_c - ||mu_c||^2
```

这其实就是一个特殊的线性分类器：

```text
logit_c = W[c] · x + b[c]
```

如果令：

```text
W[c] = 2 mu_c
b[c] = - ||mu_c||^2
```

那么这个线性分类器的预测结果就和欧氏距离 NME 一致。

因此，用 class mean 初始化线性分类器不是完全没有依据的。它可以理解成：先把 NME/prototype classifier 的类别中心结构写进线性分类器，再让模型继续用交叉熵训练调整分类边界。

实际实现时可以有两种初始化形式：

1. **严格 NME 线性化初始化**

```text
W[c] = 2 mu_c
b[c] = - ||mu_c||^2
```

这个版本和欧氏距离 NME 有最直接的数学对应关系。

2. **类别方向初始化**

```text
W[c] = normalize(mu_c)
b[c] = 0
```

这个版本不严格等价于 NME，更接近 cosine classifier 的初始化方式。它的含义是把第 c 类的分类方向初始化为该类特征中心方向。后续训练仍然可以通过交叉熵继续调整权重。

老师的想法更像第二种：不是要求初始化后完全复现 NME，而是避免新增权重从随机方向开始，让分类器一开始就有和类别中心相关的分类方向。

需要注意的是，这个方法是否有效取决于当前特征空间的类别中心结构。如果特征本身不按类别聚集，或者 class mean 估计很差，那么 class mean 初始化不一定带来提升。换句话说，它不是保证提升的方法，而是一个用来验证“分类器随机初始化是否是瓶颈”的 ablation。

#### 三种 ablation 版本

task1 时，新分类器权重矩阵可以按类别和特征来源切成四块：

```text
                      TS0 特征列       TS1 特征列
旧类别 0-9             A               B
新类别 10-19           C               D
```

其中：

```text
A: 旧类使用旧 TS0 特征的权重
B: 旧类使用新 TS1 特征的权重
C: 新类使用旧 TS0 特征的权重
D: 新类使用新 TS1 特征的权重
```

当前代码的逻辑是：

```text
A = 继承上一阶段旧权重
B = 随机初始化
C = 随机初始化
D = 随机初始化
```

基于老师的 class mean 初始化思路，可以做三种版本。

##### 版本 1：只初始化新类权重

只替换新类别对应的整行权重：

```text
A = 继承旧权重
B = 随机初始化
C = 新类 class mean 初始化
D = 新类 class mean 初始化
```

也就是：

```text
W1[10:20, :] = new_class_means
```

旧类别仍按原始代码处理：

```text
W1[:10, :512] = W0
W1[:10, 512:] = 随机初始化
```

这个版本最保守，对旧类已有分类器影响最小；但它没有解决旧类在新 TS 特征列上的随机初始化问题。

##### 版本 2：初始化所有已见类权重

对当前所有 seen classes 都重新计算当前 concat 特征空间里的 class mean，然后重置整个分类器权重：

```text
A = 旧类 class mean 初始化
B = 旧类 class mean 初始化
C = 新类 class mean 初始化
D = 新类 class mean 初始化
```

也就是：

```text
W1[0:20, :] = all_seen_class_means
```

旧类 class mean 用 memory 样本计算，新类 class mean 用当前 task 训练样本计算。

这个版本最接近老师的原始想法，逻辑也最干净：

```text
每个类别的 classifier weight = 当前特征空间里的 class mean
```

缺点是它会覆盖 A 中上一阶段已经学到的旧类分类权重；而旧类 class mean 只由 memory 估计，可能没有完整训练集均值稳定。

##### 版本 3：保留旧左上角，只补原本随机的区域

这个版本保留旧类别在旧 TS 特征上的已学习权重，只用 class mean 替换原本随机的 B/C/D：

```text
A = 继承旧权重
B = 旧类 memory class mean 初始化
C = 新类 class mean 初始化
D = 新类 class mean 初始化
```

也就是：

```text
W1[:10, :512] = W0
W1[:10, 512:] = old_class_means[:, 512:]
W1[10:20, :] = new_class_means
```

这个版本最贴合“四块矩阵”的问题分析：保留已有旧知识，只把原本随机的区域换成类别原型。缺点是 A 来自训练得到的权重，B/C/D 来自特征均值，二者尺度可能不一致，因此需要考虑 normalize 或后续 weight align。

#### 推荐实验顺序

建议优先做版本 2：

```text
每个 task 用当前 train + memory 重新计算所有 seen classes 的 concat feature mean，
然后用这些 class mean 初始化整个 fc.weight。
```

原因是版本 2 最容易实现和解释，也最接近老师的想法。如果版本 2 有明显提升，再继续做版本 3，观察“保留旧权重 + 只补随机区域”是否更稳定。如果版本 2 没有提升，说明分类器随机初始化可能不是当前最主要瓶颈，需要回到 TA/TS 表征质量或训练损失本身继续排查。

### 下一步优先实验2：LeJEPA 自监督目标消融

第二个需要做的实验是回到原本类增量设定下，判断 LeJEPA 这个自监督方法对当前框架到底有没有帮助。

实验1关注的是：

```text
分类器随机初始化是否是瓶颈？
```

实验2关注的是：

```text
把原版 TagFex 的自监督/对比学习方式换成 LeJEPA，是否真的提升了类增量性能？
```

这个实验应该先保持原始分类器设定，不加入 class mean 初始化。否则如果性能发生变化，就无法区分提升来自 LeJEPA，还是来自分类器初始化改动。

### 实验2的核心变量

核心变量是：

```text
是否使用 LeJEPA 自监督目标，以及使用哪一种自监督目标
```

建议至少比较三组：

1. **原版 TagFex**

使用原本的 InfoNCE / SimCLR 式自监督目标。

2. **TagFex + LeJEPA**

使用当前 LeJEPA 版本，也就是 view invariance + SIGReg。

3. **TagFex 去掉自监督**

不使用原版 InfoNCE，也不使用 LeJEPA，只保留分类、KD、transfer、memory 等原本类增量组件。

如果资源允许，可以继续拆 LeJEPA 内部组件：

4. **LeJEPA 去掉 SIGReg**

只保留 view invariance，观察是否出现坍缩或性能下降。

5. **LeJEPA 去掉 invariance**

只保留 SIGReg，观察分布正则本身是否有帮助。

### 关键对比关系

最重要的是前三组，因为它们能回答最核心的问题：

```text
原版 TagFex vs TagFex + LeJEPA
```

回答：LeJEPA 是否比原版 InfoNCE 更好？

```text
TagFex + LeJEPA vs TagFex 去掉自监督
```

回答：LeJEPA 相比完全不用自监督是否有帮助？

```text
原版 TagFex vs TagFex 去掉自监督
```

回答：原版自监督目标本身是否有帮助？

### 可能结果与解释

如果结果是：

```text
LeJEPA > 原版 TagFex > 无自监督
```

说明 LeJEPA 在当前类增量设定下确实带来了正向帮助。

如果结果是：

```text
原版 TagFex > LeJEPA > 无自监督
```

说明自监督是有用的，但当前 LeJEPA 不如原版 InfoNCE。

如果结果是：

```text
原版 TagFex > 无自监督 > LeJEPA
```

说明当前 LeJEPA 不仅没有帮助，还可能干扰了表征学习或分类训练。

如果结果是：

```text
原版 TagFex ≈ 无自监督 > LeJEPA
```

说明在这个类增量设定下，自监督可能不是主要贡献点，而当前 LeJEPA 更不是正向贡献。

### 评价指标

实验2不能只看最终 accuracy，还应该看完整类增量表现：

```text
CNN top1 curve
Final average accuracy
Forgetting
NME curve
每个 task 的 old/new class accuracy
disjoint_only / blurry_exposed
```

原因是 LeJEPA 可能只对某些 task、旧类保持、或者 blurry exposed 类有帮助。如果只看最终平均值，可能会漏掉局部变化。

### 实验顺序建议

为了避免变量混在一起，建议按下面顺序做：

```text
实验1：固定当前方法，只改分类器初始化，验证分类头问题。
实验2：保持原始分类器设定，只消融 LeJEPA 自监督目标，验证 LeJEPA 是否有用。
实验3：如果实验1和实验2各自有效，再组合 class mean init + best self-supervised setting。
```

这样可以分别回答两个问题：

```text
分类器初始化是否有问题？
LeJEPA 自监督目标是否有帮助？
```

最后再判断二者是否可以叠加提升。

## 自己的想法：LeJEPA 接入方式的修正性消融

这部分是我自己在分析当前实现时发现的问题，不是老师直接提出的新方法。它更适合定位为：对当前 LeJEPA 实验设置的修正性消融，或者说是 LeJEPA 接入方式校正实验。

它的目的不是提出一个新的模型模块，而是先排查当前 LeJEPA 结果是否被实现细节影响。现在的 LeJEPA 版本相对原版 TagFex 不只是替换了自监督损失，还同时改变了 view 组织方式和 TS 分类分支的输入分布。因此需要先判断当前结果差是否来自 LeJEPA 方法本身，还是来自接入方式不合理。

这个修正性消融主要包含两个问题。

### 问题 1：SIGReg 是否应该保持 view-wise 分布

当前 LeJEPA 训练中，数据形状是：

```text
vs: [N, V, C, H, W]
V = 8
```

前向时会先展平 view 维度：

```python
out = self._network(vs.flatten(0, 1))
embedding = out["embedding"]
```

此时：

```text
embedding: [N * V, D]
```

当前 invariance loss 会重新恢复 view 维度：

```python
proj = embedding.reshape(N, V_dim, -1)
proj_mean = proj.mean(1, keepdim=True)
inv_loss = (proj_mean - proj).square().mean()
```

但 SIGReg 使用的是展平后的 embedding：

```python
sigreg_loss = self.sig_reg(embedding)
```

这意味着当前 SIGReg 看到的是：

```text
所有样本的所有 view 混在一起形成的一个大分布
```

而不是：

```text
view 0 下的 N 个样本是一个分布
view 1 下的 N 个样本是一个分布
...
view 7 下的 N 个样本是一个分布
```

如果原版 LeJEPA 的 SIGReg 是 view-wise 的，那么当前实现会丢掉 view 维度，把 global view 和 local view 混在一个统计分布里。由于 global crop 和 local crop 的尺度不同，它们的特征分布可能并不一致，混在一起会让 SIGReg 的目标变得不清晰。

因此需要做一个修正实验：

```text
mixed-view SIGReg  ->  view-wise SIGReg
```

可选实现方式：

```python
proj = embedding.reshape(N, V_dim, -1)  # [N, V, D]
sigreg_loss = mean(self.sig_reg(proj[:, v, :]) for v in range(V_dim))
```

如果 `SIGReg` 支持 3D 输入，也可以组织成：

```python
proj_vnd = proj.transpose(0, 1)  # [V, N, D]
sigreg_loss = self.sig_reg(proj_vnd)
```

这个实验回答的问题是：

```text
LeJEPA 表现差，是否部分来自 SIGReg 的 view 分布被错误混合？
```

### 问题 2：8 个 views 是否都应该进入 TS 分类分支

原版 TagFex 的训练输入是两个增强视图：

```python
inputs = torch.cat([inputs1, inputs2], dim=0)
outputs = self._network(inputs)
```

也就是说，每个样本随机增强成两张图，这两张图同时进入 TA 和 TS 分支，并参与分类损失。

当前 LeJEPA 版本的数据增强是：

```text
2 个 global views + 6 个 local views = 8 views
```

训练时直接把 8 个 views 全部送入整个网络：

```python
outputs = self._network(vs.flatten(0, 1))
y_rep = targets.repeat_interleave(V_dim)
loss_clf = F.cross_entropy(logits, y_rep)
```

这意味着 8 个 views 不只用于 TA 自监督，也同时用于：

```text
TS 主分类 logits
aux_logits
trans_logits
KD
transfer loss
```

这里可能有问题：6 个 local views 是局部裁剪，可能只包含物体的一小部分。它们适合用于自监督学习 view invariance，但不一定适合强制 TS 分类分支用完整标签做交叉熵。换句话说，当前 LeJEPA 版本可能让 TS 分支在大量 local crop 上学习分类，改变了原版 TagFex 的分类输入分布。

因此需要做一个输入消融实验：

```text
LeJEPA 自监督仍使用 8 views，
但 TS 分类相关损失只使用前两个 global views。
```

一个较省显存的实现方式是：仍然 forward 8 个 views，但计算分类损失时只取前两个 global views：

```python
logits = logits.reshape(N, V_dim, -1)[:, :2, :].reshape(N * 2, -1)
aux_logits = aux_logits.reshape(N, V_dim, -1)[:, :2, :].reshape(N * 2, -1)
y_cls = targets.repeat_interleave(2)
loss_clf = F.cross_entropy(logits, y_cls)
```

对应地，`aux_loss`、`trans_cls_loss`、`transfer_loss`、以及需要分类标签的统计，也应该只对 global views 计算。

更干净但成本更高的版本是分离 forward：

```text
TA LeJEPA loss: 使用 8 views
TS 分类/aux/transfer: 使用 2 个 global views
```

这样可以更彻底地避免 local views 影响 TS 分类分支，但会增加计算和显存成本。

### 这个修正性消融的推荐顺序

建议按下面顺序做：

1. **先修 view-wise SIGReg**

这是最直接的实现修正。如果修正后 Oracle NME 或正常类增量性能提升，说明之前 mixed-view SIGReg 确实干扰了 LeJEPA 表征学习。

2. **再做 classification-global-only**

也就是 LeJEPA loss 仍使用 8 views，但分类、aux、transfer 等监督损失只使用前两个 global views。这个实验判断当前性能差是否来自 local views 对 TS 分类分支造成干扰。

3. **最后再考虑 TA/TS 分离 forward**

这是最干净的接入方式，但代价更高。可以在前两个实验有改善后再做。

### 和实验2的关系

实验2是宏观消融：

```text
原版 TagFex vs TagFex + LeJEPA vs 去掉自监督
```

它回答 LeJEPA 自监督目标整体是否有用。

这部分自己的想法是接入修正：

```text
当前 LeJEPA 表现不好，是否因为实现时混合了 view-wise SIGReg，或者让 8 个 views 全部进入 TS 分类分支？
```

因此这个修正性消融应该优先于对 LeJEPA 下结论。只有在修正这些接入问题后，才能更公平地判断 LeJEPA 本身是否适合当前类增量框架。

### 推荐表述

> 我自己发现，当前 LeJEPA 版本相对原版 TagFex 同时改变了两个因素：一是 SIGReg 从 view-wise distribution 变成了所有 view 混合后的整体分布；二是训练输入从两个增强视图变成 8 个 views，并且这 8 个 views 都进入 TS 分类分支参与分类、aux 和 transfer 等监督损失。因此需要先做接入修正实验：将 SIGReg 改回 view-wise 形式，并测试 LeJEPA 自监督使用 8 views、但 TS 分类损失只使用前两个 global views 的版本。这样才能判断当前性能问题到底来自 LeJEPA 方法本身，还是来自接入方式改变了原版 TagFex 的训练分布。

