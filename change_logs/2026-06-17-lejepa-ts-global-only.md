# 2026-06-17 LeJEPA 接入修正实验：TS Global-only

## 问题

当前 LeJEPA mean-fusion 版本使用 8 个 views：

```text
2 个 global views + 6 个 local views
```

训练时 8 个 views 会一起 forward，并且 TS 分类相关损失也使用全部 8 个 views。这样 6 个 local crops 也会带完整图片标签进入主分类、aux、trans/transfer 等监督损失，可能改变原版 TagFex 的分类输入分布。

本次实验回答的问题是：

```text
8 个 views 是否都应该进入 TS 分类分支？
```

这是和 view-wise SIGReg 并列的独立消融实验，不叠加问题一的 `sigreg_view_mode = "view_wise"`。

## 修改内容

### 1. 新增 TS view 选择开关

修改文件：`models/tagfex_lejepa.py`

新增配置开关：

```json
"ts_global_only": true,
"num_ts_views": 2
```

行为：

- LeJEPA 自监督仍使用全部 8 个 views。
- SIGReg 仍保持默认 mixed-view 行为，即所有 views 混成一个分布。
- TS 分类相关损失只使用前 `num_ts_views` 个 views，默认前 2 个 global views。

受影响的训练项：

- Task0 `ce_loss`
- Task1+ `loss_clf`
- Task1+ `loss_aux`
- Task1+ `trans_cls_loss`
- Task1+ `transfer_loss`
- Task1+ KD 在该开关开启时也限制在 global views
- train accuracy / diagnostics 统计

不受影响的训练项：

- `inv_loss`
- `sigreg_loss`
- `lejepa_loss`

### 2. 新增问题二实验配置

新增文件：`exps/lejepa_fixes/tagfex_lejepa_mean_fusion_ts_global_only.json`

该配置基于：

```text
exps/lejepa_sigreg/tagfex_lejepa_mean_fusion_imagenet100.json
```

只新增：

```json
"ts_global_only": true,
"num_ts_views": 2
```

并将 prefix 改为：

```json
"prefix": "ablation_lejepa_mean_fusion_ts_global_only"
```

该配置不设置 `sigreg_view_mode`，因此 SIGReg 保持原来的 mixed-view 行为。

## 验证

已运行：

```bash
python -m py_compile models/tagfex_lejepa.py
python -m json.tool exps/lejepa_fixes/tagfex_lejepa_mean_fusion_ts_global_only.json
```

两项检查均通过。

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config exps/lejepa_fixes/tagfex_lejepa_mean_fusion_ts_global_only.json
```
