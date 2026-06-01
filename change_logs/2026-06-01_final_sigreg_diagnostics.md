# 2026-06-01 Final SIGReg 诊断功能修改日志

## 问题背景

当前实验中，最后追加 fixed-A SIGReg 的策略让 NME 准确率明显变好，但 CNN 分类准确率偏低，尤其是旧类准确率下降明显。需要补充诊断信息，用来判断问题来自新类偏置、分类头和特征空间不匹配，还是表征本身退化。

## 修改目标

- 记录 CNN 在训练集和测试集上的准确率曲线，并拆成 total、old classes、new classes。
- 保存 raw confusion matrix 和 row-normalized confusion matrix。
- 在原本只记录 final SIGReg 前后 CNN 准确率的基础上，补充记录 final SIGReg 前后的 NME 准确率。

## 重要说明

NME 依赖 class means。final SIGReg 前后的 NME 诊断使用 freshly computed analysis class means，也就是用当前模型重新从所有 seen classes 的训练数据中计算临时 class means。这个过程只用于分析，不会修改 rehearsal memory，也不会修改 official class means。

## 已完成修改

### 1. CNN 准确率曲线诊断

修改文件：`models/tagfex_lejepa.py`

新增功能：

- 每个 epoch 记录训练集 CNN accuracy。
- 按配置间隔记录测试集 CNN accuracy。
- 每条记录都包含：
  - `total_acc`
  - `old_acc`
  - `new_acc`
  - `total_count`
  - `old_count`
  - `new_count`
- 测试集评估由 `diagnostic_eval_interval` 控制，现在设置为每个 epoch 都记录一次测试集准确率。
- 结果写入当前实验日志目录下：

```text
diagnostics/accuracy_curves_task_{task}.csv
```

- 同时向 SwanLab 写入 `Diagnostics/*` 指标，方便在线查看曲线。

### 2. 混淆矩阵保存

修改文件：`trainer.py`

新增功能：

- 在 official task 结束评估点保存 CNN 混淆矩阵。
- 如果已有 class means，也保存 official NME 混淆矩阵。
- 在 final SIGReg 前后保存 CNN 混淆矩阵。
- 如果启用了 final SIGReg NME 诊断，也保存 before/after 的 `NME_analysis` 混淆矩阵。
- 每个混淆矩阵保存三种格式：
  - `.csv`
  - `.npy`
  - `.png`，当 matplotlib 可用时生成

输出路径：

```text
diagnostics/confusion_matrices/
final_sigreg/confusion_matrices/
```

每个阶段都会保存两类矩阵：

```text
*_confusion_raw.csv
*_confusion_row_normalized.csv
```

### 3. final SIGReg 前后 NME 准确率记录

修改文件：`trainer.py`

新增功能：

- 在 final SIGReg 之前计算 `before_nme_analysis`。
- 在 final SIGReg 之后计算 `after_nme_analysis`。
- 自动计算 `nme_analysis_delta_after_minus_before`。
- 结果写入原来的 final SIGReg accuracy 文件：

```text
final_sigreg/final_sigreg_task_{task}_accuracy.csv
final_sigreg/final_sigreg_task_{task}_accuracy.json
```

现在这个文件会同时包含：

- `CNN` before / after / delta
- `NME_analysis` before / after / delta

### 4. final_sigreg 实验配置更新

修改文件：`exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json`

新增配置：

```json
"record_accuracy_curves": true,
"diagnostic_eval_interval": 1,
"record_confusion_matrix": true,
"record_final_sigreg_nme": true,
"analysis_nme_batch_size": 64,
"analysis_nme_num_workers": 4
```

这些配置只打开诊断记录，不改变 final SIGReg 的核心训练逻辑。

## 验证结果

已通过：

```bash
python3 -m py_compile trainer.py models/tagfex_lejepa.py
python3 -m json.tool exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json
git diff --check
```

## 实现备注

本环境中 `apply_patch` 可以创建新文件，但在更新已有文件时会因为 sandbox namespace 限制失败。因此已有文件修改是通过一次性 Python 脚本完成的。

另外，`models/tagfex_lejepa.py` 原文件存在混合行尾格式。修改时尽量只改动必要区域，并通过 `git diff --check` 确认没有引入 trailing whitespace 或冲突标记。




## 可复现性补充检查与修改

### 当前版本已经保存的内容

task checkpoint 文件位于：

```text
checkpoints/{prefix}_{seed}_task_{task}.pth
```

现在 checkpoint 中会保存：

- `model_state_dict`：完整模型权重。
- `data_memory` / `targets_memory`：rehearsal memory 样本和标签。
- `class_means`：NME 使用的类均值。
- `sigreg_state`：SIGReg 随机投影矩阵状态，包括 fixed/running matrix、running count、matrix seed 等。
- `final_sigreg_state`：final SIGReg 的运行状态。
- `cnn_curve` / `nme_curve` / `cnn_matrix` / `nme_matrix`：阶段性评估结果。
- `args`：本次运行的完整参数字典。
- `args_json_ready`：可读的 JSON 友好版参数。
- `runtime_reproducibility`：Python、PyTorch、CUDA、cuDNN、git commit、run command 等运行环境信息。
- `rng_state`：Python random、NumPy、PyTorch CPU、PyTorch CUDA RNG 状态。

final SIGReg 的固定矩阵也会单独保存：

```text
checkpoints/{prefix}_{seed}_task_{task}_final_sigreg_A_final.pth
```

其中包含：

- `A_final`
- `projection_matrix`
- `matrix_seed`
- `num_directions`
- `feat_dim`

### 本次为可复现性新增的内容

修改文件：`trainer.py`

新增：

- 保存完整 resolved config：

```text
resolved_config.json
```

- 保存运行元信息：

```text
run_reproducibility.json
```

这两个文件会写在每次实验的 `{timestamp}` 日志目录下。

`resolved_config.json` 中会包含：

- 原始配置合并命令行参数后的最终参数。
- `class_order`：实际类别顺序。
- `task_increments`：每个 task 的类别数量。
- `config_path`：使用的配置文件路径。
- `run_command`：实际启动命令。
- `git_branch` / `git_commit` / `git_dirty`。

`run_reproducibility.json` 中会额外记录：

- Python 版本。
- PyTorch 版本。
- CUDA/cuDNN 信息。
- `CUDA_VISIBLE_DEVICES`。
- `TAGFEX_DATA_ROOT`，保存为 `tagfex_data_root`。
- `cudnn_deterministic` / `cudnn_benchmark`。
- 是否 DDP。

同时，checkpoint 中新增保存：

- `args`
- `args_json_ready`
- `runtime_reproducibility`
- `rng_state`

resume 时会尝试恢复 `rng_state`。

### 随机性控制补充

修改文件：`trainer.py`

`_set_random()` 现在会同时设置：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

修改文件：`models/tagfex_lejepa.py`

之前 `_train()` 中会强制设置：

```python
torch.backends.cudnn.benchmark = True
```

这会影响严格复现。现在改为读取配置：

```python
torch.backends.cudnn.deterministic = bool(self.args.get("cudnn_deterministic", True))
torch.backends.cudnn.benchmark = bool(self.args.get("cudnn_benchmark", False))
```

修改文件：`exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json`

新增：

```json
"cudnn_deterministic": true,
"cudnn_benchmark": false
```

### 仍需注意

- task checkpoint 是 task 结束后的恢复点，不是任意 iteration 的中断恢复点，所以没有保存 optimizer/scheduler state。
- 如果以后需要“训练到一半断点续跑并逐步完全一致”，还需要保存 optimizer state、scheduler state、当前 epoch、当前 dataloader 进度。
- 目前的保存方式足够支持：复现实验配置、保存最终/阶段模型权重、恢复到下一个 task 继续训练、分析 final SIGReg 前后结果。



## 数据集路径说明

本机 ImageNet100 实际路径是：

```text
/root/autodl-tmp/ImageNet100/train/
/root/autodl-tmp/ImageNet100/val/
```

代码中的 `utils/data.py` 已更新 ImageNet100 路径候选顺序：

1. 优先使用环境变量 `TAGFEX_DATA_ROOT`。
2. 然后自动尝试 `/root/autodl-tmp/ImageNet100`。
3. 再尝试旧路径 `/root/autodl-tmp/datasets/ImageNet100`。
4. 最后尝试实验室路径 `/media/DATASET/person_data/ImageNet100`。

为了可复现，建议运行命令里显式写上：

```bash
TAGFEX_DATA_ROOT=/root/autodl-tmp/ImageNet100
```

## 修改后如何运行实验

### 1. 直接运行当前 final_sigreg 诊断实验

在项目根目录 `/root/tagfex` 下运行：

```bash
cd /root/tagfex
TAGFEX_DATA_ROOT=/root/autodl-tmp/ImageNet100 CUDA_VISIBLE_DEVICES=0 python main.py \
  --config exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --run_mode debug
```


说明：

- 现在工作区有未提交修改，所以建议先用 `--run_mode debug` 跑通。
- `main.py` 会读取 `--config` 指定的 JSON 配置。
- 当前配置中已经打开诊断开关：accuracy curves、confusion matrix、final_sigreg NME before/after 都会自动记录。

### 2. 后台运行并保存终端日志

如果想让实验在后台跑，并把终端输出保存下来，可以运行：

```bash
cd /root/tagfex
mkdir -p run_logs
TAGFEX_DATA_ROOT=/root/autodl-tmp/ImageNet100 CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --config exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --run_mode debug \
  > run_logs/final_sigreg_diagnostics_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

查看实时输出：

```bash
tail -f run_logs/final_sigreg_diagnostics_*.log
```

### 3. 正式实验模式

如果要用正式实验模式，需要先保证 git 工作区干净，也就是把本次代码修改提交或 stash 掉。否则 `--run_mode exp` 会报错。

正式模式命令：

```bash
cd /root/tagfex
TAGFEX_DATA_ROOT=/root/autodl-tmp/ImageNet100 CUDA_VISIBLE_DEVICES=0 python main.py \
  --config exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --run_mode exp
```

### 4. 多卡 DDP 运行方式

如果要用 2 张 GPU，可以用 `torchrun`：

```bash
cd /root/tagfex
TAGFEX_DATA_ROOT=/root/autodl-tmp/ImageNet100 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py \
  --config exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json \
  --run_mode debug
```

正式 DDP 实验同样需要先提交或 stash 修改，然后把 `--run_mode debug` 改成 `--run_mode exp`。

### 5. 本次新增诊断结果在哪里看

配置里的 `log_root` 是：

```text
/root/autodl-tmp/tagfex_logs
```

一次实验的输出目录大致是：

```text
/root/autodl-tmp/tagfex_logs/
  ablation_lejepa_mean_fusion_final_sigreg/
  imagenet100_lejepa/
  10/
  10/
  {timestamp}/
```

重点查看这些文件：

```text
diagnostics/accuracy_curves_task_{task}.csv
```

记录训练集和测试集上的 CNN accuracy curve，包括：

- `total_acc`
- `old_acc`
- `new_acc`

```text
diagnostics/confusion_matrices/
```

保存 official task 结束后的 CNN/NME 混淆矩阵。

```text
final_sigreg/final_sigreg_task_{task}_accuracy.csv
final_sigreg/final_sigreg_task_{task}_accuracy.json
```

保存 final SIGReg 前后的 CNN 和 `NME_analysis` 准确率，以及 after-before 的变化量。

```text
final_sigreg/confusion_matrices/
```

保存 final SIGReg before/after 的 CNN 和 `NME_analysis` 混淆矩阵。

## 2026-06-02 SwanLab accuracy curve 命名修正

### 问题

原来的 SwanLab 诊断曲线使用了类似下面的全局 metric 名：

```text
Diagnostics/update_train_old_acc
Diagnostics/update_train_new_acc
Diagnostics/update_test_old_acc
```

这个命名有歧义。因为不同 task 中 old/new 的含义不同：

- task1: old = 0-9, new = 10-19
- task2: old = 0-19, new = 20-29
- task3: old = 0-29, new = 30-39

如果所有 task 都写到同一个 `Diagnostics/update_train_old_acc`，SwanLab 里就会把不同含义的 old accuracy 混在一张图上，看起来像 init/update 阶段混乱。

### 修改

修改文件：`models/tagfex_lejepa.py`

SwanLab epoch-level 诊断曲线现在放到对应 task 分类下：

```text
Task_0/Diagnostics/init/train_total_acc
Task_0/Diagnostics/init/train_new_acc
Task_0/Diagnostics/init/test_total_acc
Task_0/Diagnostics/init/test_new_acc

Task_1/Diagnostics/update/train_old_acc
Task_1/Diagnostics/update/train_new_acc
Task_1/Diagnostics/update/test_old_acc
Task_1/Diagnostics/update/test_new_acc
```

同时把 SwanLab step 从 `epoch` 改成 `epoch + 1`，避免图上从 0 开始导致误读。

修改文件：`trainer.py`

在每个 task 的 official evaluation 结束后，新增 task-level summary 曲线：

```text
Summary/CNN_total_acc
Summary/CNN_old_acc
Summary/CNN_new_acc
Summary/CNN_top5
Summary/NME_total_acc
Summary/NME_old_acc
Summary/NME_new_acc
Summary/NME_top5
```

这些 summary 的横轴是 task id，适合看整体遗忘趋势。task0 没有 old classes，所以 task0 不记录 old summary。

### 新的看图方式

- 分析某一个 task 内部训练过程：看 `Task_{task_id}/Diagnostics/...`。
- 看所有 task 结束后的总体趋势：看 `Summary/...`。
- 顶层 `Diagnostics` 不再堆每个 task 的曲线，避免 SwanLab 页面过乱。
- 不再使用旧的全局 `Diagnostics/update_train_old_acc` 来解释结果。

## 2026-06-02 SwanLab 分组层级调整

### 问题

虽然上一版已经把 metric 名改成了 `Diagnostics/Task_{id}/...`，但是 SwanLab 顶层仍然会把所有 task 的诊断曲线都归到 `Diagnostics` 分组里，页面会堆很多卡片，不利于查看。

### 修改

修改文件：`models/tagfex_lejepa.py`

将 task 内部诊断曲线从：

```text
Diagnostics/Task_1/update/train_old_acc
```

调整为：

```text
Task_1/Diagnostics/update/train_old_acc
```

这样每个 task 的 accuracy curve 会出现在对应 `Task_{id}` 分组下面；顶层只保留 `Summary` 用来看跨 task 总体趋势。

## 2026-06-02 测试集准确率记录频率调整

### 修改

修改文件：`exps/tagfex_lejepa_mean_fusion_imagenet100_final_sigreg.json`

将：

```json
"diagnostic_eval_interval": 10
```

改为：

```json
"diagnostic_eval_interval": 1
```

### 效果

- train accuracy：仍然每个 epoch 记录一次。
- test accuracy：现在也每个 epoch 记录一次。

这样每个 task 内部的 train/test old/new accuracy curve 会更密集，方便观察 CNN 准确率变化过程。代价是每个 epoch 都要额外跑一次 test loader，训练会更慢。
