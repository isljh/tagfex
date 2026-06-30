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

## Standalone TA-LeJEPA 诊断实验

### 背景

当前在线线性探测器结果显示，TagFex-LeJEPA 的 Task0 训练 200 epoch 后，`ta_feature` 的 epoch 内 batch 平均 probe top1 约为 74.16%。在冻结 TagFex 其他模块、只继续训练 TA encoder 和 projector 后，probe top1 还能继续上升到 75% 以上。

这说明两个问题：

1. LeJEPA 自监督分支在原始 Task0 训练结束时还没有完全收敛。
2. 延长 TA-LeJEPA 训练对表示线性可分性有正向作用，但目前提升速度较慢。

不过，继续训练实验是从已经经过 TagFex 联合训练的 Task0 checkpoint 出发，仍然会受到前期训练状态影响。它可以说明“继续训练有用”，但不能干净回答：

```text
TA 分支本身在当前数据输入、增强、ResNet18 backbone 和 projector 配置下，
按照 LeJEPA 目标从头训练，最多能学到多强的可分类表示？
```

因此需要一个更独立的诊断实验：把 TA 分支单独拆出来，从随机初始化开始训练，只判断 TA-LeJEPA 本身是否能训好。

### 实验定义

实验名称可以定义为：

```text
Standalone TA-LeJEPA Task0 pretrain
```

核心设定如下：

```text
数据：imagenet100_lejepa 的 Task0 数据
输入：沿用当前 8-view 数据输入和增强
模型：只使用 TA encoder，也就是 ResNet18
projector：沿用当前 TagFex-LeJEPA 的 projector 配置，即 512 -> 2048 -> 2048 -> 1024 的三层 MLP
loss：只使用 LeJEPA 自监督目标
probe：使用 ta_feature.detach() 后接 LayerNorm + Linear
```

当前 TagFex-LeJEPA projector 的具体结构是：

```text
ta_feature_dim = 512
proj_hidden_dim = 2048
proj_output_dim = 1024

TagFex_SimpleLinear(512, 2048)
BatchNorm1d(2048)
ReLU
TagFex_SimpleLinear(2048, 2048)
BatchNorm1d(2048)
ReLU
TagFex_SimpleLinear(2048, 1024)
BatchNorm1d(1024)
```

其中 `TagFex_SimpleLinear` 本质上是一个 `nn.Linear`，并使用 Kaiming uniform 初始化权重、常数初始化 bias。

对应的 LeJEPA 官方参考设定可以写成：

```text
数据：官方 minimal 示例使用 Imagenette；benchmark 中通常按对应数据集重新构建自监督训练集
输入：多视图增强；minimal 示例使用 8 个 views，包含 RandomResizedCrop、ColorJitter、RandomGrayscale、GaussianBlur 等增强
模型：minimal 示例使用 ViT-Small；Lightly benchmark 表中也给出了 ResNet18 对比结果
projector：MLP projector，minimal 示例为 512 -> 2048 -> 2048 -> proj_dim，并使用 BatchNorm1d
loss：同样只使用 LeJEPA 自监督目标，即 invariance loss + SIGReg loss
probe：使用 encoder embedding.detach() 后接 LayerNorm + Linear
```

两者主要差异是：

```text
数据集：我们的实验是 imagenet100_lejepa Task0，官方 minimal 是 Imagenette 示例
backbone：我们的 TA 是 ResNet18，官方 minimal 是 ViT-Small；官方 benchmark 里也有 ResNet18 结果
输入增强：我们沿用当前 8-view 数据管线，官方使用自己的多视图增强配置
projector：我们沿用 TagFex-LeJEPA projector，官方 minimal 是带 BatchNorm1d 的 MLP projector
评估指标：我们先看 online linear probe，官方 benchmark 图里常见的是 kNN top1 或正式 linear eval
```

LeJEPA loss 保持当前形式：

```text
LeJEPA_total_loss = SIGReg_loss * lambda + invariance_loss * (1 - lambda)
```

这个实验不使用：

```text
TS 分支
mean fusion
主分类器 CE loss
transfer loss
KD loss
memory replay
CIL task update
```

也就是说，它不再判断完整 TagFex 的类增量性能，而是只回答 TA 自监督分支本身能不能训练出足够线性可分的表示。

### 第一版配置

为了尽量和 LeJEPA 官方的 ResNet 场景对齐，第一版可以使用：

```text
epochs: 200
lr: 5e-4
weight_decay: 5e-4
lejepa_lambda: 0.05
probe_lr: 1e-3
probe_weight_decay: 1e-7
probe_feature: ta_feature
num_views: 8
batch_size: 64 或显存允许的更大 batch size
```

其中：

1. `lr=5e-4` 与 LeJEPA 官方推荐 starting point 一致。
2. `weight_decay=5e-4` 对 ResNet 合理，官方说明 ViT 常用 `5e-2`，ResNet 常用 `5e-4`。
3. `probe_lr=1e-3`、`probe_weight_decay=1e-7` 与官方 minimal 示例中的 online probe 设置一致。
4. `lejepa_lambda=0.05` 是官方常见 sweep 范围 `0.01/0.02/0.05/0.1` 中的合理默认值。

### 需要记录的指标

逐 step 记录：

```text
standalone_ta/Prediction_Invariance_loss
standalone_ta/SIGReg_loss
standalone_ta/LeJEPA_total_loss
standalone_ta/probe_loss
standalone_ta/probe_top1
standalone_ta/lr
```

每个 epoch 记录一次 batch 平均：

```text
standalone_ta_epoch/probe_top1
standalone_ta_epoch/probe_loss
standalone_ta_epoch/LeJEPA_total_loss
```

其中 `standalone_ta_epoch/probe_top1` 仍然是 epoch 内 batch probe top1 的平均值，不是正式 frozen linear eval，也不是 kNN top1。它主要用于观察训练趋势。

如果后续要和 LeJEPA 官方 benchmark 中的 90% 左右结果比较，还需要额外增加 epoch-end kNN 或 frozen linear evaluation。否则 online probe 只能作为诊断指标，不能直接等价于官方 kNN top1。

### 结果解释

如果 standalone TA-LeJEPA 的 probe top1 明显超过当前 74% 到 75% 区间，甚至能接近 85% 到 90%，说明：

```text
LeJEPA 自监督分支本身在当前数据和 TA 架构下是可以训好的。
之前性能不理想更可能来自接回 TagFex 后的联合训练方式、
分类损失干扰、fusion/transfer 结构或训练调度。
```

如果 standalone TA-LeJEPA 也长期卡在 70% 到 80%，说明：

```text
主要瓶颈更可能在 TA-LeJEPA 自身设置，
例如 ResNet18 backbone、augmentation、lambda、projector、学习率或训练 epoch 数。
```

因此，这个实验可以把问题拆清楚：

```text
是 LeJEPA + 当前 TA 设置本身学不好，
还是 LeJEPA 接入完整 TagFex 后被其他模块和损失项影响了。
```

### 推荐结论表述

> 为了判断 LeJEPA 自监督分支本身是否能训练好，需要把 TA 分支从完整 TagFex 中拆出来，做一个 Standalone TA-LeJEPA Task0 pretrain。该实验只保留 TA encoder、projector、LeJEPA invariance loss、SIGReg loss 和 online linear probe，不使用 TS、fusion、分类 CE、transfer、KD、memory 或 CIL task update。这样可以直接验证在当前 imagenet100_lejepa 数据输入、8-view 增强、ResNet18 TA 和 projector 配置下，LeJEPA 自监督目标本身能否学到足够线性可分的表示。如果 standalone 结果明显高于当前 74% 到 75% 的在线 probe 水平，则说明 LeJEPA 分支本身可训练，问题更可能在 TagFex 联合训练接入方式；如果 standalone 仍然卡在 70% 到 80%，则说明需要优先调整 TA-LeJEPA 自身配置。

