# 特征分布可视化

这个目录保存较早期的特征分布可视化脚本，用于快速观察模型内部特征是否接近高斯形状。

这些脚本更偏直观检查，不像 `sigreg_projection/` 那样系统比较投影矩阵和多方向统计。

## verify_sigreg.py

读取 checkpoint，提取 projector 后的 embedding，并画出整体数值直方图，与拟合高斯和标准高斯进行对比。

它主要回答：

```text
projector embedding 的整体数值分布看起来是否接近标准高斯？
```

## draw_simclr.py

读取 checkpoint，在 TA backbone 的最后一个 ReLU 前注册 hook，提取 Pre-ReLU 激活分布并画图。

它主要回答：

```text
TA backbone 的 Pre-ReLU 激活分布是否接近高斯？
不同训练设置下这个分布有没有明显变化？
```

这两个脚本依赖手动填写 checkpoint 路径，更适合作为临时可视化工具。正式分析 SIGReg 高斯性时，优先使用：

```text
analysis/sigreg_projection/
```
