# 2026-06-02 Si-Blurry ImageNet100 适配

## 问题

当前 Si-Blurry 设定只有一个明确的 CIFAR100 配置。用户希望适配 ImageNet100，并且后续要和参考的 FlyGCL 风格论文/代码结果对比，所以需要确认参考代码里 ImageNet100 的数据组织方式。

## 修改内容

- `main.py`
  - 在配置文件和命令行参数合并逻辑中加入 `data_root`。
  - 新增 `--data_root` 命令行参数。

- `trainer.py`
  - 新增 `_apply_data_root(args)`。
  - 在创建 `DataManager` 前，将 `data_root` 写入环境变量 `TAGFEX_DATA_ROOT`。
  - 当数据集名称包含 ImageNet100 时，同时写入 `TAGFEX_IMAGENET100_ROOT`。

- `utils/data.py`
  - 新增 `_get_data_root()`、`_get_split_dirs()` 和 `_split_imagefolder_subset()`。
  - 更新 `iImageNet100.download_data()`，支持从可配置的 `data_root/train` 与 `data_root/val` 或 `data_root/test` 加载数据。
  - 如果加载到的 `ImageFolder` 超过 100 类，则保留 ImageFolder 类别索引 `0..99`，对应 `reference_flygcl/ImageNet100.py` 中可见的 `keep = range(100)` 逻辑。

- `exps/si_blurry_tagfex_imagenet100_aa.json`
  - 新增 ImageNet100 AA 的 Si-Blurry 配置，关键字段如下：
    - `dataset: imagenet100_aa`
    - `setting: si_blurry`
    - `n_tasks: 5`
    - `n: 50`
    - `m: 10`
    - `rnd_NM: true`

## 验证

- 运行 Python 语法检查：
  - `python -m py_compile main.py trainer.py utils\data.py utils\data_manager.py utils\si_blurry_sampler.py`
- 运行 JSON 格式检查：
  - `python -m json.tool exps\si_blurry_tagfex_imagenet100_aa.json`

两项检查均通过。

## 运行命令

如果数据集就在配置文件中的默认路径 `/root/autodl-tmp/datasets/ImageNet100`：

```bash
python main.py --config exps/si_blurry_tagfex_imagenet100_aa.json
```

如果 ImageNet100 在其他路径，用 `--data_root` 覆盖：

```bash
python main.py --config exps/si_blurry_tagfex_imagenet100_aa.json --data_root /path/to/ImageNet100
```

期望的数据目录结构：

```text
ImageNet100/
  train/
    class_folder_1/
    class_folder_2/
    ...
  val/
    class_folder_1/
    class_folder_2/
    ...
```

## 备注

- 当前 `reference_flygcl` 中可见的 ImageNet100 组织方式是 `root/train` 和 `root/val`，通过 `ImageFolder` 加载。
- 参考代码中可见的类别选择逻辑是在 ImageFolder 排序后保留 `range(100)`。
- 当前参考文件夹里没有显式的 ImageNet100 类别名单，也没有 `collections/imagenet100/*.json`，所以这次适配依据的是当前能看到的参考代码，而不是独立确认过的论文类别列表。
