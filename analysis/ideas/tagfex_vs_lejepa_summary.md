# 原版 TagFex 与 LeJEPA 修改版差异总结

本文总结当前 `tagfex` 与 `tagfex_lejepa` 两条实现线的主要差异。结论是：LeJEPA 版不是简单把原版 TagFex 的 InfoNCE loss 换成 LeJEPA loss，而是同时改变了训练输入形式、自监督目标、优化器与调度器、融合方式、KD 对齐方式、分类分支使用的 views，以及训练后的 final SIGReg 校准与诊断流程。

## 对比对象

原版 TagFex：

```text
models/tagfex.py
exps/si_blurry/si_blurry_tagfex_imagenet100_aa.json
```

LeJEPA 修改版：

```text
models/tagfex_lejepa.py
exps/si_blurry/si_blurry_tagfex_lejepa_imagenet100_final_sigreg.json
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100.json
exps/lejepa_fixes/
```

两者底层网络主体仍然使用同一个 `TagFexNet`：

```text
utils/inc_net.py
```

也就是说，TS experts、TA net、projector、predictor、aux classifier、trans classifier 这些基本模块并没有完全换掉。主要差异集中在训练逻辑和配置。

## 1. 数据增强与 view 数量不同

原版 TagFex 使用两个增强视图：

```json
"aug": 2,
"num_views": 2
```

训练时 dataloader 返回：

```text
inputs1, inputs2, targets
```

然后拼成一个 batch：

```python
inputs = torch.cat([inputs1, inputs2], dim=0)
targets = torch.cat([targets, targets], dim=0)
```

LeJEPA 版使用 8 个 views：

```json
"aug": 1,
"num_views": 8
```

训练时输入通常是：

```text
vs: [N, V, C, H, W]
```

forward 前展平成：

```python
out = self._network(vs.flatten(0, 1))
```

因此 LeJEPA 版默认会让 TA、TS、projector、分类头、KD 等模块都看到 `N * 8` 个 view，而原版只有 `N * 2`。

这个差异很重要，因为 LeJEPA 的 8 个 views 通常包含：

```text
2 个 global views + 6 个 local views
```

如果所有 local crops 都带完整图像标签进入分类损失，可能会改变原版 TagFex 的分类输入分布。

## 2. 自监督目标从 InfoNCE 改为 LeJEPA loss

原版 Task0 的训练目标是：

```text
loss = CE + contrast_factor * InfoNCE
```

其中 InfoNCE 作用在 projector 输出的 embedding 上，用两个增强视图构造正样本关系。

LeJEPA 版 Task0 的训练目标是：

```text
loss = CE + contrast_factor * LeJEPA_loss
```

其中：

```text
LeJEPA_loss = lambda * SIGReg_loss + (1 - lambda) * invariance_loss
```

默认：

```json
"lejepa_lambda": 0.05
```

LeJEPA loss 的两部分含义是：

1. `invariance_loss`：把同一张图的不同 views 的 projector embedding 拉近。
2. `SIGReg_loss`：约束 embedding 分布，防止坍缩，并鼓励特征空间有足够的分布结构。

因此，原版是对比学习式目标；LeJEPA 版是 view invariance + 分布正则。

## 3. SIGReg 是 LeJEPA 版新增的关键模块

LeJEPA 版新增了 `SIGReg` 模块，用随机投影矩阵把 embedding 投影到多个方向，再通过特征函数统计量约束投影分布接近高斯。

当前支持三种随机投影矩阵策略：

```text
per_batch_random
fixed
running_avg
```

SiBlurry final SIGReg 配置中使用：

```json
"sigreg_matrix_mode": "per_batch_random"
```

之前发现的一个接入问题是：当前 LeJEPA 接入 TagFex 时，默认 SIGReg 使用展平后的 embedding：

```text
embedding: [N * V, D]
```

这会把所有 views 混成一个整体分布。原版 LeJEPA 更接近保留 view 维度：

```text
[N, V, D]
```

然后对每个 view 下的 batch 分布分别做 SIGReg，再对 views 求平均。

因此后来新增了：

```json
"sigreg_view_mode": "view_wise"
```

用于测试：

```text
mixed-view SIGReg vs view-wise SIGReg
```

对应实验配置：

```text
exps/lejepa_fixes/tagfex_lejepa_mean_fusion_view_wise_sigreg.json
```

## 4. TS 分类分支是否应该使用全部 8 个 views

LeJEPA mean-fusion 基线默认会让 8 个 views 都参与 TS 分类相关损失，包括：

```text
Task0 CE
Task1+ main CE
Task1+ aux loss
Task1+ trans_cls_loss
Task1+ transfer_loss
Task1+ KD
train accuracy / diagnostics
```

这带来一个问题：

```text
6 个 local crops 是否应该带完整图像标签进入分类分支？
```

为此新增了独立消融开关：

```json
"ts_global_only": true,
"num_ts_views": 2
```

开启后：

1. LeJEPA 自监督仍使用全部 8 个 views。
2. SIGReg / invariance / LeJEPA loss 不受影响。
3. 分类相关损失只使用前 2 个 global views。
4. KD 在该开关开启时也限制到 global views。

对应实验配置：

```text
exps/lejepa_fixes/tagfex_lejepa_mean_fusion_ts_global_only.json
```

这个实验和 view-wise SIGReg 是并列的独立消融，不默认叠加。

## 5. 融合方式从默认 attention 改为 mean fusion

`TagFexNet` 中默认融合方式是：

```json
"fusion_type": "ts_attention"
```

也就是用 `TSAttention` 融合 TA feature 和 TS feature。

LeJEPA mean-fusion 相关配置改为：

```json
"fusion_type": "mean"
```

对应融合逻辑从：

```python
self.ts_attn(ta_features, ts_feature).mean(1)
```

变成：

```python
(ta_features + ts_feature).mean(1)
```

因此 LeJEPA mean-fusion 版本不仅换了自监督目标，也改变了 trans classifier 前的 TA/TS 融合方式。

## 6. 优化器与学习率调度不同

原版 TagFex 使用：

```text
SGD
init_lr = 0.1
lrate = 0.1
MultiStepLR
batch_size = 128
num_workers = 8
```

LeJEPA 版使用：

```text
AdamW
init_lr = 5e-4
update_lr = 5e-4
Linear warmup + CosineAnnealingLR
batch_size = 64
num_workers = 16
```

这也是一个很大的训练条件变化。Task0 表征质量下降时，不能只归因于 LeJEPA loss 本身，优化器和学习率策略也需要单独考虑。

## 7. KD 的 InfoNCE 正样本定义被修改

原版 `infoNCE_distill_loss` 使用 batch 内 roll 的方式找正样本：

```text
positive = index + batch_size // 2
```

这个写法适合两个增强视图的组织方式。

LeJEPA 版中，特征顺序是 `N * V`，多 view 情况下继续使用 roll 不一定合理。因此 KD loss 被改成 p/z 一一对应的交叉熵形式：

```text
p_feats[i] 对齐 z_feats[i]
```

这更适合当前的多 view 展平顺序。

另外，LeJEPA 版还支持：

```json
"kd_global_only": true
```

用于把 KD 限制到 global views。

## 8. LeJEPA 版增加 final SIGReg calibration

SiBlurry LeJEPA final 配置中打开了：

```json
"final_sigreg": true
```

训练完一个 task 后，不是立刻 build rehearsal memory，而是先挂起 memory 构建，额外跑一个 final SIGReg calibration。

该阶段只更新：

```text
TA backbone + projector
```

主要配置包括：

```json
"final_sigreg_epochs": 1,
"final_sigreg_lr": 5e-05,
"final_sigreg_update_scope": "ta_projector",
"final_sigreg_num_directions": 256
```

这个流程是 LeJEPA 版新增的，原版 TagFex 没有。

## 9. LeJEPA 版增加了更多诊断记录

LeJEPA final 配置中额外打开：

```json
"record_accuracy_curves": true,
"record_confusion_matrix": true,
"record_final_sigreg_nme": true
```

这些诊断不会直接定义主要训练目标，但会改变训练后的分析流程，并生成更多用于排查的 artifact，例如：

```text
accuracy curves
confusion matrix
final SIGReg before/after NME
Oracle NME analysis
```

## 10. 当前 Task0 Oracle NME 诊断结论

在同一个 SiBlurry Task0 的 38 类测试集上，用测试集标签构建 oracle prototype 后，已有结果是：

| 方法 | 特征空间 | Oracle NME top1 | top5 | disjoint_only | blurry_exposed |
| --- | --- | ---: | ---: | ---: | ---: |
| 原版 TagFex | `ta_feature` | 70.05 | 91.74 | 79.12 | 63.45 |
| 原版 TagFex | `embedding` | 69.00 | 93.37 | 80.12 | 60.91 |
| LeJEPA | `ta_feature` | 33.16 | 64.16 | 37.62 | 29.91 |
| LeJEPA | `embedding` | 28.21 | 60.74 | 29.50 | 27.27 |

这个结果说明：

1. LeJEPA 的问题不只是后续 task 的 memory、KD、分类头或增量过程造成的。
2. 在 Task0 阶段，LeJEPA 的 TA 表征类别可分性已经明显弱于原版 TagFex。
3. 原版 TagFex 的 projector embedding 与 ta_feature 接近，而 LeJEPA 的 embedding 还低于自身 ta_feature，说明当前 projector/LeJEPA 自监督空间没有形成更适合类别原型分类的结构。

注意：Oracle NME 使用测试集标签构建 prototype，属于标签泄露的上界诊断，不能作为正式测试准确率。

## 后续优先消融建议

为了定位 LeJEPA 版性能下降的来源，建议按下面顺序做更干净的消融：

1. **8 views 分类输入问题**

   对比全部 8 views 进入分类分支与只用 2 个 global views 进入分类分支。

   ```text
   exps/lejepa_fixes/tagfex_lejepa_mean_fusion_ts_global_only.json
   ```

2. **SIGReg mixed vs view-wise**

   对比把所有 views 混成一个分布，和每个 view 单独计算 SIGReg 后求平均。

   ```text
   exps/lejepa_fixes/tagfex_lejepa_mean_fusion_view_wise_sigreg.json
   ```

3. **InfoNCE vs LeJEPA loss**

   保持其他设置尽量一致，只比较原版 InfoNCE 目标和 LeJEPA invariance + SIGReg 目标。

4. **SGD vs AdamW / MultiStepLR vs Cosine**

   当前优化器和调度器变化很大，需要单独确认是否影响 Task0 表征。

5. **ts_attention vs mean fusion**

   LeJEPA mean-fusion 不只是换 loss，也把 trans 分支融合方式改成 mean。需要确认 mean fusion 是否削弱了原版 TagFex 的迁移分类分支。

6. **final SIGReg calibration**

   对比打开和关闭 final SIGReg，观察它对 TA/projector 表征、NME、正式 accuracy 的影响。

## 一句话总结

当前 LeJEPA 修改版与原版 TagFex 的差异是系统性的：输入从 2 views 变成 8 views，自监督目标从 InfoNCE 变成 invariance + SIGReg，优化器从 SGD 变成 AdamW，融合从 attention 变成 mean，并额外引入了 global-only、view-wise SIGReg、KD view selection 和 final SIGReg calibration 等机制。已有 Task0 Oracle NME 显示，LeJEPA 版在还没有进入复杂增量遗忘之前，TA 表征本身已经明显弱于原版 TagFex，因此后续需要优先把 view 使用方式、自监督目标和优化策略拆开做消融。
