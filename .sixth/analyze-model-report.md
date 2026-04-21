# TagFex_DER 静态分析报告

目标：解释为什么 `tagfex_der` 在 **task0** 理论上只训练一个 `resnet18`，却可能耗时 **11–13 小时**。以下结论仅基于静态代码分析，不修改代码。

## 结论摘要

`task0` 慢的最核心原因，不是 task0 里偷偷训练了多个 backbone，也不是每个 epoch 都在做完整测试评估，而是：

1. **task0 训练轮数本身极高：固定 200 epoch**
   - `models/tagfex_der.py` 中 `init_epoch = 200`
   - 对 ImageNet 风格数据来说，200 epoch 的 resnet18 本身就不是“短训练”。

2. **每个 batch 实际前向/反向的样本数被放大了 `num_views` 倍**
   - 训练时输入不是 `[N, C, H, W]`，而是 `[N, V, C, H, W]`
   - 然后在 `task0` 里被直接 `flatten(0, 1)` 成 `[N*V, C, H, W]`
   - 即：名义 batch size = 128，但真实送入 resnet18 的是 `128 * V`
   - 若配置 `num_views = 8`，则每 step 实际处理 **1024 张图**

3. **DataLoader 的 batch_size 没有按 views 缩小**
   - `current_batch_size` 仍然是配置 batch size（分布式下只按 GPU 数量均分）
   - 并没有考虑 `num_views`
   - 所以训练吞吐开销近似变成普通训练的 `V` 倍

4. **task0 结束后还会额外执行一次非常重的 exemplar memory 构建**
   - `incremental_train()` 在 `_train()` 后无条件调用 `build_rehearsal_memory(...)`
   - `BaseLearner._construct_exemplar()` 会对每个新类完整提特征、再做贪心 exemplar 选择、再对选中的 exemplar 再提一次特征
   - 这部分虽然不属于 epoch 内训练，但会显著拉长“task0 总耗时”

5. **DDP 开启了 `find_unused_parameters=True`，会引入额外开销**
   - 对这种图结构并不复杂的 DER task0，通常会比默认模式更慢
   - 但这更像次要放大项，不是 11–13 小时的第一原因

6. **每个 batch 都做 SwanLab 日志上报**
   - `swanlab.log(...)` 在训练内层循环中每 step 调用
   - 如果 step 数很大，会增加一定 CPU/IO/网络开销
   - 单独不足以解释 10+ 小时，但会雪上加霜

---

## 代码级证据

## 1. task0 的训练阶段就是 200 epoch

文件：`models/tagfex_der.py`

```python
init_epoch = 200
```

在 `_train()` 中：

```python
if self._cur_task == 0:
    optimizer = torch.optim.AdamW(trainable_params, lr=init_lr, weight_decay=init_weight_decay)
    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * init_epoch
    ...
    self._init_train(train_loader, test_loader, optimizer, scheduler)
```

在 `_init_train()` 中：

```python
prog_bar = tqdm(range(init_epoch), disable=disable_tqdm)
```

这说明 task0 没有任何早停、阶段切换或减轮数逻辑，就是完整跑 **200 个 epoch**。

---

## 2. 每个 batch 被扩展为 `N * V` 张图送进网络

文件：`models/tagfex_der.py`

```python
V_dim = self.args.get('num_views', 8)
```

在 `_init_train()` 里，每个 batch：

```python
vs = data[1].to(self._device)   # [N, V, C, H, W]
targets = data[-1].to(self._device)

inputs = vs.flatten(0, 1)        # [N*V, C, H, W]
out = self._network(inputs)
logits = out["logits"]

y_rep = targets.repeat_interleave(V_dim)
loss = F.cross_entropy(logits, y_rep)
```

这段代码非常关键：

- 数据集返回的是 **多视图输入**
- 一个样本不是一张图，而是 `V` 张增强视图
- 训练时没有做 view-level 的轻量聚合，而是**把所有视图全部展开后逐张过 backbone**

所以 task0 虽然只训练一个 `resnet18`，但这个 resnet18 每步实际上处理的是：

- 普通训练：`N`
- 这里：`N * V`

如果 `N=128, V=8`，那就是 **1024 张图/step**

这会让训练时间接近普通单视图训练的数倍。

---

## 3. batch size 只按 GPU 数量切分，没有按 views 缩小

文件：`models/tagfex_der.py`

```python
batch_size = 128
```

分布式时：

```python
if is_distributed:
    ...
    current_batch_size = batch_size // world_size
else:
    current_batch_size = batch_size
```

后面 DataLoader：

```python
self.train_loader = DataLoader(
    train_dataset, batch_size=current_batch_size,
    shuffle=(train_sampler is None), num_workers=num_workers,
    pin_memory=True, sampler=train_sampler, drop_last=True
)
```

注意这里的 `current_batch_size` 是“样本数”，不是“图像数”。  
但每个样本包含 `V` 个 views，并且在训练时全部 `flatten` 到 batch 维。

因此真实 backbone 输入规模是：

- 单卡：`current_batch_size * V`
- DDP 每卡：`(batch_size // world_size) * V`

例如 4 卡 + `batch_size=128` + `V=8`：
- 每卡 DataLoader batch = 32 个样本
- 每卡网络实际输入 = `32 * 8 = 256` 张图
- 全局每 step 实际处理 = `256 * 4 = 1024` 张图

所以从算力角度看，它绝不是“普通 batch=128 的单个 resnet18”。

---

## 4. DERNet_TagFex 的 task0 前向本身不复杂：不是多阶段，不是重复前向多个分支

文件：`utils/inc_net.py`

`DERNet_TagFex.forward()`：

```python
ts_outs = [convnet(x) for convnet in self.convnets]
features = [ts_out["features"] for ts_out in ts_outs]
features = torch.cat(features, 1)

out = {}
out["logits"] = self.fc(features)
aux_logits = self.aux_fc(features[:, -self.out_dim:])
out.update({"aux_logits": aux_logits, "features": features})
```

对于 task0：

- `self.convnets` 里只有一个 convnet
- 所以这里只会执行 **一次** `convnet(x)`
- 不存在 TagFex 正式版那种 `ta_net` / `projector` / `attention` / `predictor` 的额外前向

对照 `TagFexNet.forward()` 可以看出，`tagfex_der` 已经删掉了这些额外模块：

- 无 `ta_net(x)`
- 无 `projector(ta_feature)`
- 无 `ts_attn(...)`
- 无 `trans_classifier(...)`
- 无 `predictor(...)`

因此：
- **慢的主要原因不是模型结构过于复杂**
- **而是“单 backbone + 多视图展开 + 高 epoch + 后处理构建 memory”**

---

## 5. task0 没有每个 epoch 内做测试评估

`models/tagfex_der.py` 的 `_init_train()` 中，没有调用：

- `_compute_accuracy`
- `eval_task`
- `self.test_loader` 遍历
- `save_checkpoint`

每个 epoch 里只是训练循环和 tqdm 文本更新：

```python
for _, epoch in enumerate(prog_bar):
    ...
    for i, data in enumerate(train_loader):
        ...
    prog_bar.set_description(...)
```

所以：
- **task0 慢不是因为每个 epoch 末尾都做完整测试集验证**
- `test_loader` 虽然创建了，但在 task0 训练函数里并未使用

---

## 6. task0 没有样本级相似度矩阵/大规模 pairwise 运算

对比 `models/tagfex.py` 正式版，后者有：
- `SIGReg`
- `infoNCE_distill_loss` 的 `N*V × N*V` 相似度矩阵
- 迁移分类、蒸馏、attention 等

而 `models/tagfex_der.py` task0 只有：

```python
loss = F.cross_entropy(logits, y_rep)
```

没有：
- pairwise similarity matrix
- InfoNCE
- KD
- projector / predictor
- 注意力模块

因此：
- **task0 的 11–13 小时，不是被 NxN 相似度矩阵拖慢**
- 这类高阶运算在 `tagfex_der` task0 中并不存在

---

## 7. DDP 使用 `find_unused_parameters=True` 会增加额外同步/图遍历开销

文件：`models/tagfex_der.py`

```python
self._network = torch.nn.parallel.DistributedDataParallel(
    self._network, device_ids=[local_rank], output_device=local_rank,
    find_unused_parameters=True
)
```

对于 task0 的 `DERNet_TagFex`：

- 图比较简单
- 主要参数都会被使用
- `find_unused_parameters=True` 会让 DDP 做额外 autograd 图分析

这通常会带来可见的性能损失，尤其在 step 数很多时。

它不是第一主因，但会进一步放大总耗时。

---

## 8. 每个 batch 都调用 `swanlab.log`，step 数巨大时会累积明显开销

文件：`models/tagfex_der.py`

```python
if local_rank <= 0:
    swanlab.log({
        "init/total_loss": loss.item(),
        "init/batch_acc": batch_acc,
        "init/lr": optimizer.param_groups[0]['lr'],
    }, step=batch_step)
    batch_step += 1
```

这是在 **内层 batch 循环** 里调用的，而不是每个 epoch 一次。  
当总 step 数为：

```python
200 * len(train_loader)
```

时，日志调用次数会非常多。

如果数据集较大、step 数很多，日志系统会带来：
- Python 层开销
- 序列化开销
- IO/网络开销
- 训练线程被阻塞的风险

单独看不是 10+ 小时根因，但在“本来就很重”的训练上会继续增时。

---

## 9. task0 结束后会执行一次很重的 rehearsal memory 构建

这部分非常容易被忽略，因为它不在 `_init_train()` 里，而是在 `incremental_train()` 收尾阶段。

文件：`models/tagfex_der.py`

```python
self._train(self.train_loader, self.test_loader)

if hasattr(self._network, 'module'):
    self.build_rehearsal_memory(data_manager, self.samples_per_class)
    self._network = self._network.module
else:
    self.build_rehearsal_memory(data_manager, self.samples_per_class)
```

也就是说：
- task0 训练一结束
- 就会立刻构建 exemplar memory
- 用户感知到的“task0 完成时间”会包含这部分额外成本

### 9.1 task0 会走 `_construct_exemplar`

文件：`models/base.py`

```python
def build_rehearsal_memory(self, data_manager, per_class):
    if self._fixed_memory:
        self._construct_exemplar_unified(data_manager, per_class)
    else:
        self._reduce_exemplar(data_manager, per_class)
        self._construct_exemplar(data_manager, per_class)
```

task0 时 `self._known_classes == 0`，所以：
- `_reduce_exemplar(...)` 基本没什么事可做
- 真正重的是 `_construct_exemplar(...)`

### 9.2 `_construct_exemplar()` 对每个类别都要完整提特征

```python
for class_idx in range(self._known_classes, self._total_classes):
    data, targets, idx_dataset = data_manager.get_dataset(
        np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
    )
    idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    vectors, _ = self._extract_vectors(idx_loader)
```

这意味着：
- 对 task0 的每个新类，都会单独构建一个数据集/loader
- 然后把该类所有样本重新过一遍网络提取特征

若初始类数较大（如 50、100），这就是非常可观的额外推理成本。

### 9.3 还包含贪心 exemplar 选择，复杂度不低

```python
selected_exemplars, exemplar_vectors = [], []
for k in range(1, m + 1):
    S = np.sum(exemplar_vectors, axis=0) if len(exemplar_vectors) > 0 else 0
    mu_p = (vectors + S) / k
    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
    selected_exemplars.append(np.array(data[i]))
    exemplar_vectors.append(np.array(vectors[i]))
    vectors = np.delete(vectors, i, axis=0)
    data = np.delete(data, i, axis=0)
```

这里有典型的 CPU/Numpy 贪心选择过程：
- 多次 `np.sum`
- `np.sqrt`
- `np.argmin`
- `np.delete`

如果每类样本多、`m` 也不小，这段 CPU 计算会进一步拉长时间。

### 9.4 选完 exemplar 后，又重新提一次特征算 class mean

```python
idx_dataset = data_manager.get_dataset([], source="train", mode="test", appendent=(selected_exemplars, exemplar_targets))
idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
vectors, _ = self._extract_vectors(idx_loader)
```

即每个类至少两轮额外特征提取：
1. 对该类全量样本提特征
2. 对选中 exemplar 再提一次特征

因此，task0 总耗时 = 训练耗时 + exemplar memory 构建耗时。  
如果用户是看“整个 task0 结束”时间，这部分必须算进去。

---

## 10. `_extract_vectors()` 还可能有额外 Python/PIL 转换负担

文件：`models/base.py`

```python
for _, _inputs, _targets in loader:
    _targets = _targets.numpy()

    if not isinstance(_inputs, torch.Tensor):
        if isinstance(_inputs, (list, tuple)):
            _inputs = torch.stack(
                [pilot_trsf(img) if not isinstance(img, torch.Tensor) else img for img in _inputs])
        else:
            _inputs = pilot_trsf(_inputs).unsqueeze(0)
```

这里说明 memory 构建时的特征提取流程可能出现：
- PIL -> Tensor 的 Python 循环转换
- `torch.stack`
- 额外 Normalize

如果 `get_dataset(..., mode="test")` 返回的不是已经 tensor 化的数据，这会引入明显 CPU 开销。  
这对 task0 训练本体无影响，但对 task0 收尾阶段的 exemplar 构建很可能有影响。

---

## 11. `resnet18` 本身不是被重复前向两次，但输入分辨率若是 ImageNet 风格会进一步放大代价

文件：`convs/resnet.py`

对于 imagenet 类数据：

```python
elif 'imagenet' in args["dataset"]:
    if args["init_cls"] == args["increment"]:
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    else:
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
```

说明模型会适配 ImageNet 风格输入。  
如果数据管线使用 224×224 且又有 `num_views=8`，那么：
- 单步输入量大
- 单张图分辨率也大
- 200 epoch 代价自然非常高

这部分需要结合配置和数据集分析，但从模型侧可确认其不是 CIFAR 那种 32×32 的轻量训练路径。

---

## 12. 与正式版 `TagFex` 对比：`tagfex_der` 已经删掉了很多重模块，因此“慢”更能指向数据规模与训练设置

对照 `models/tagfex.py`，正式版 task0 还包含：

- `ta_net` 前向
- `projector`
- `SIGReg`
- view-level LeJEPA loss
- 可视化日志

而 `models/tagfex_der.py` 没有这些。  
因此如果 `tagfex_der` 的 task0 仍然达到 11–13 小时，更说明瓶颈不在 fancy loss，而在：

- 200 epoch
- 多视图展开训练
- 大输入分辨率/大数据集
- task 末 memory 构建
- DDP/日志等次级额外开销

---

## 对“是否存在这些问题”的逐项回答

### 是否有多阶段训练？
**task0 没有多阶段训练。**
- 只有 `_init_train(...)` 一个阶段
- 没有额外 finetune / eval-train / classifier calibration 阶段

### 是否有特征提取重复前向？
**task0 训练阶段没有重复前向多个分支；但 task0 结束后 memory 构建会做大量额外特征提取。**
- 训练时只有一个 convnet
- 但 exemplar 构建时会多次 `_extract_vectors()`

### 是否有样本级相似度/矩阵运算？
**task0 没有。**
- 不存在 InfoNCE/SIGReg/NxN similarity matrix
- 只有标准交叉熵

### 内存回放如何参与？
**task0 训练时通常没有旧 memory 参与，但 task0 结束后会立即构建 memory，且这一步很重。**
- `appendent=self._get_memory()` 在 task0 大概率为 `None`
- 但 task0 结束后 `build_rehearsal_memory(...)` 一定执行

### DataLoader 是否重复遍历？
**训练阶段按正常 epoch 遍历；但 task0 结束后的 exemplar 构建会按类重复遍历数据集。**
- 每个类单独 `get_dataset(...)`
- 每类至少一次全量特征提取，另加一次 exemplar 特征提取

### 每个 epoch 内是否有额外评估/拷贝网络/保存 checkpoint？
**没有明显的 epoch 内评估或 checkpoint 保存。**
- 无 `_compute_accuracy()` 调用
- 无 `eval_task()` 调用
- 无 `save_checkpoint()` 调用

---

## 最可能导致 11–13 小时的主因排序

### 主因 1：`num_views` 导致真实输入规模膨胀 `V` 倍
证据：
```python
inputs = vs.flatten(0, 1)
```
这是最直接的算力放大器。

### 主因 2：task0 固定 200 epoch
证据：
```python
init_epoch = 200
```
对大数据集/高分辨率场景极重。

### 主因 3：task0 结束后还要做 exemplar memory 构建
证据：
```python
self.build_rehearsal_memory(data_manager, self.samples_per_class)
```
并且 `BaseLearner._construct_exemplar()` 是按类多次提特征的重过程。

### 主因 4：若数据集是 ImageNet 风格 224×224，单图成本本来就高
证据：
`convs/resnet.py` 明确区分 cifar / imagenet 路径。

### 次因 5：DDP `find_unused_parameters=True`
证据：
```python
find_unused_parameters=True
```

### 次因 6：每 batch 都进行 `swanlab.log`
证据：
内层训练循环直接 log。

---

## 一句结论

`tagfex_der` 的 task0 虽然**模型结构上只训练了一个 resnet18 分支**，但代码实际上是在做 **“200 epoch × 每样本 8 视图展开后的大 batch 训练 + task 末按类多轮特征提取构建 exemplar memory”**，因此其总耗时远不能按“普通单视图 resnet18 训练”估计；若数据还是 ImageNet 风格分辨率，这种 11–13 小时的现象在代码层面是完全有依据的。