# 未来想法记录

这个文件用于记录暂时不作为近期实验、但后续可能继续展开的研究想法。这里的内容偏灵感和方向，不代表已经验证有效。

## 方向 1：基于类别特征统计量的回放与约束

### 背景

当前类增量学习里，旧类主要依赖 exemplar memory 回放。随着 task 增加，总 memory 固定时，每个类能保存的图片样本会越来越少。

因此可以考虑：

> 不只保存旧类的少量图片样本，还保存每个类在特征空间中的统计量，例如 class mean、variance 或 covariance。

也就是把 replay 从单纯的样本级回放，扩展到分布级回放。

### 可保存的统计量

对每个类别 c，可以保存：

```text
mu_c: class mean均值
var_c: per-dimension variance方差
Sigma_c: covariance matrix协方差
n_c: 计算统计量时使用的样本数
```

其中最简单的是：

```text
mean + diagonal variance
```

也就是只保存每个特征维度上的均值和方差。完整 covariance 信息更多，但存储和计算成本也更高。

### 可能用途 1：feature-level replay（特征级回放）

用均值和方差生成旧类伪特征

如果保存了旧类的特征分布统计，可以假设旧类特征近似服从高斯分布：

```text
z_c ~ N(mu_c, Sigma_c)
```

训练新 task 的分类器时，可以从旧类分布中采样 pseudo features：

```text
pseudo_feature = mu_c + eps * std_c
```

然后把这些伪特征作为旧类 replay，用于训练分类器或辅助保持旧类决策边界。

这种方式的目标是：

```text
用类别分布统计补充有限 exemplar memory。
```

它的优点是不用保存大量旧类图片；缺点是它依赖“类别特征分布可以被均值和方差较好描述”这个假设。如果真实特征分布是多峰的，简单高斯可能不够准确。

### 可能用途 2：Mahalanobis / distribution-aware classifier（马氏距离 / 分布感知分类器）

用方差做更好的分类器 / NME

普通 NME 只使用 class mean：

```text
pred = argmin_c ||x - mu_c||^2
```

如果保存了方差或协方差，可以进一步使用 Mahalanobis distance：

```text
pred = argmin_c (x - mu_c)^T Sigma_c^{-1} (x - mu_c)
```

直觉是：

```text
类内波动大的方向，偏离 class mean 一些也可以接受，就不要太惩罚；
类内波动小的方向，偏离一点就更说明不像这个类。
```

这样分类时不再把每个类别看成一个点，而是看成一个带形状的分布。

### 可能用途 3：distribution-aware exemplar selection（分布感知样本选择）

用均值和方差辅助 exemplar 选择

当前 exemplar 选择通常强调：

```text
选出的 exemplar 均值尽量接近全类 class mean。
```

未来可以进一步要求：

```text
选出的 exemplar 不仅 mean 接近全类 mean，
它们的 variance / diversity 也尽量接近全类分布。
```

这样 memory 样本不只是代表类别中心，也能覆盖类内多样性。

可以理解为：

```text
从 mean-preserving exemplar selection
扩展到 mean-and-variance-preserving exemplar selection。
```

### 可能用途 4：约束旧类特征分布漂移

用旧类统计量做正则

训练新 task 时，旧类特征可能随着模型更新发生漂移。可以用旧类保存的统计量作为约束：

```text
current_mu_c ≈ stored_mu_c
current_var_c ≈ stored_var_c
```

也就是在训练过程中，让当前模型提取到的旧类 memory 特征分布，不要偏离上一阶段保存的旧类分布太多。

这种方法比只约束单个样本特征更偏向分布层面，可以作为 feature distribution regularization（特征分布正则）。

### 和当前 class mean 初始化实验的关系

近期要做的实验 1 是用 class mean 初始化线性分类器权重：

```text
fc.weight[c] = mu_c
```

这个未来方向可以看作它的进一步扩展：

```text
实验 1：只使用 class mean。
未来方向：进一步使用 variance / covariance。
```

如果 class mean 初始化有效，说明类别统计量确实能帮助分类器或旧类保持。后续就可以继续考虑引入方差、协方差或分布级 replay。

### 需要注意的问题

这个方向暂时不作为近期实验，原因是它涉及的问题更多：

1. 特征统计量应该在 TA feature、TS concat feature，还是 projector embedding 上计算？
2. 旧类统计量随 task 更新时，要不要重算？如果重算，只能用 memory，估计可能有偏。
3. 方差/协方差如何稳定估计？旧类 memory 很少时，统计量可能不可靠。
4. pseudo feature replay 会不会生成不真实的旧类特征？
5. 如果类别分布是多峰的，单个 mean + variance 是否足够？

因此它更适合作为后续方向，而不是当前第一批实验。

### 一句话表述

> 未来可以考虑引入 class-wise feature distribution statistics。除了保存 exemplar，还保存每个类在特征空间中的均值和方差/协方差，用于 feature-level replay（特征级回放）、Mahalanobis prototype classification（马氏距离原型分类）、distribution-aware exemplar selection（分布感知样本选择），或者约束旧类特征分布漂移。这样回放不再只依赖少量图片样本，而是利用旧类的统计分布信息补充 memory。
