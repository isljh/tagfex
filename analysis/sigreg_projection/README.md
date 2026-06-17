# SIGReg 投影诊断

这个目录用于诊断 LeJEPA / SIGReg 训练中，projector embedding 经过随机投影矩阵后是否接近标准高斯分布。

这里的分析不直接回答分类准确率，而是回答：

```text
SIGReg 的正则目标有没有真的把 embedding 的投影分布推向高斯？
投影矩阵本身是否健康，是否出现异常几何结构？
```

## fixed projection

脚本：

```text
analyze_fixed_projection.py
```

用于分析 fixed random projection matrix 版本。它会比较：

```text
固定矩阵 A_fixed 下的投影分布
随机矩阵下的投影分布
batch 级别和 task 全量级别的统计
```

主要输出包括 mean、std、skew、kurtosis、SIGReg loss、投影方向直方图，以及可选 PCA 图。

## running average projection

脚本：

```text
analyze_running_avg_projection.py
```

用于分析 checkpoint 中保存的 running_A。它在 fixed projection 分析基础上，额外检查矩阵几何性质，例如：

```text
列范数
列之间 cosine 相似度
condition number
effective rank
stable rank
running_A 与随机矩阵的对比
```

典型用法：

```bash
python analysis/sigreg_projection/analyze_fixed_projection.py --task-id 0 --device 0
python analysis/sigreg_projection/analyze_running_avg_projection.py --task-id 0 --device 0
```

默认输出目录：

```text
outputs/fixed_projection_analysis
outputs/running_avg_projection_analysis
```
