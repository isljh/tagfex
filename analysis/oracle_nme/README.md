# Oracle NME 分析

这个目录用于分析已经训练好的 checkpoint 中，TA 特征或 projector embedding 本身的类别可分性。

核心脚本：

```text
ta_oracle_nme.py
```

它会冻结模型，提取指定特征，然后用测试集真实标签构建每个类别的 prototype，再用最近类中心分类得到 Oracle NME accuracy。

这个分析回答的问题是：

```text
如果给特征空间最理想的类别原型，TA 特征或 embedding 最多能把类别分到什么程度？
```

注意：这个指标使用测试集标签构建类别原型，存在标签泄露，只能作为特征空间诊断上界，不能作为正式测试准确率汇报。

典型用法：

```bash
python analysis/oracle_nme/ta_oracle_nme.py \
  --config exps/si_blurry/si_blurry_tagfex_lejepa_imagenet100_final_sigreg.json \
  --checkpoint /path/to/checkpoint_task_0.pth \
  --feature ta_feature \
  --device 0
```

结果目录：

```text
results/
```

其中保存 summary、预测标签、真实标签和类别原型，方便后续复查。
