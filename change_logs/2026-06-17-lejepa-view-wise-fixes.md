# 2026-06-17 LeJEPA 接入修正实验：View-wise SIGReg

## 问题

当前 LeJEPA 接入 TagFex 时，SIGReg 使用的是展平后的 projector embedding：

```text
embedding: [N * V, D]
```

这会把同一个 batch 中所有 view 混在一起，只对一个整体分布做 SIGReg。原版 LeJEPA 更符合 view-wise 的组织方式，即保留 view 维度，对每个 view 下的 batch 分布分别做正则，再对 view 求平均。

本次实验目标是做一个更干净的接入修正：以现有 mean-fusion LeJEPA 配置为基线，只验证 mixed-view SIGReg 改成 view-wise SIGReg 的影响，不修改随机投影矩阵策略。

## 修改内容

### 1. 新增 SIGReg view mode 开关

修改文件：`models/tagfex_lejepa.py`

新增 `_compute_sigreg_loss()`：

- 默认 `sigreg_view_mode = "mixed"`，保持原有行为不变。
- 当 `sigreg_view_mode = "view_wise"` 时，将 `embedding` reshape 为：

```text
[N, V, D]
```

然后逐个 view 计算 SIGReg：

```text
view 0: [N, D]
view 1: [N, D]
...
view V-1: [N, D]
```

最后对各 view 的 SIGReg loss 取平均。

训练阶段的两处 LeJEPA loss 计算都改为调用该 helper：

- Task0 init training
- 后续 task update representation

### 2. 新增 LeJEPA 修正类实验目录

新增目录：`exps/lejepa_fixes/`

该目录用于保存 LeJEPA 接入修正类实验配置，例如 view 组织方式、local/global view 使用方式等。它不用于保存随机投影矩阵策略实验。

随机投影矩阵相关实验仍保留在：

```text
exps/lejepa_sigreg/
```

### 3. 新增 view-wise SIGReg 配置

新增文件：`exps/lejepa_fixes/tagfex_lejepa_mean_fusion_view_wise_sigreg.json`

该配置基于普通类增量的 mean-fusion LeJEPA 配置：

```text
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100.json
```

保留基线中的 mean fusion：

```json
"fusion_type": "mean",
"kd_global_only": true
```

并新增：

```json
"sigreg_matrix_mode": "per_batch_random",
"sigreg_view_mode": "view_wise"
```

并将 prefix 改为：

```json
"prefix": "ablation_lejepa_mean_fusion_view_wise_sigreg"
```

因此本实验使用 mean-fusion LeJEPA 作为基线，只测试 SIGReg view 组织方式这一处变化。

## 验证

已运行：

```bash
python -m py_compile models/tagfex_lejepa.py
python -m json.tool exps/lejepa_fixes/tagfex_lejepa_mean_fusion_view_wise_sigreg.json
```

两项检查均通过。
CUDA_VISIBLE_DEVICES=0 python main.py --config exps/lejepa_fixes/tagfex_lejepa_mean_fusion_view_wise_sigreg.json
## 备注

- `sigreg_view_mode` 默认值是 `mixed`，所以旧配置不设置该字段时仍保持原行为。
- `sigreg_matrix_mode` 仍为 `per_batch_random`，本次不修改随机投影矩阵采样策略。
- 本次修正实验不放在 `exps/lejepa_sigreg/` 下，避免和 fixed matrix / running average matrix 等随机投影矩阵实验混在一起。
- 后续“问题二”相关配置也可以继续放在 `exps/lejepa_fixes/` 下。
