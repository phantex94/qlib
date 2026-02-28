# Qlib 中“磁盘数据 → 内存 → MLP 训练/预测”的完整 Data Flow 说明

本文以 Qlib 的 `DatasetH/DataHandlerLP + DNNModelPytorch(Net)` 这条常见路径为例，解释训练时数据如何流动：

1. 从磁盘（二进制特征文件）读取到内存。
2. 经过数据集接口切分为 train/valid/test。
3. 送入 MLP 做前向、损失计算、反向传播。
4. 输出预测值并组装为带索引的 `pd.Series`。

---

## 1) 从磁盘加载到内存：Provider + DataLoader

Qlib 的底层数据读取由 `D`（Data API）触发，常见文件格式是本地 provider 维护的二进制存储。数据请求会经过：

- `DataHandlerLP.fetch()`
- `DataLoader.load()`（如 `QlibDataLoader`）
- 再到 `D.features(...)` 读取指定 instruments、时间范围、表达式字段

在 `QlibDataLoader.load_group_df` 中，最终调用的是：

```python
df = D.features(instruments, exprs, start_time, end_time, freq=freq, inst_processors=inst_processors)
```

这一步完成“从磁盘到 DataFrame（内存）”的关键转换。读取结果是 MultiIndex（`datetime`, `instrument`）的表格，并带列分组（feature/label）。

> 直观理解：磁盘上的按标的/日期组织的数据块，被按训练配置切片后拼成一个 Pandas DataFrame 放进内存。

---

## 2) 内存中样本组织：DataHandlerLP 与 DatasetH.prepare

训练脚本通常先构建 `DatasetH`，内部持有 `DataHandlerLP`。流程上，`DatasetH.prepare()` 会：

- 根据 segment（train/valid/test）映射时间区间。
- 调用 handler 的 `fetch` 拿到 `feature/label`。
- 按 `data_key` 决定使用学习态数据（`DK_L`）还是推理态数据（`DK_I`）。

`DNNModelPytorch.fit()` 中可以看到核心调用：

```python
df = dataset.prepare(seg, col_set=["feature", "label"], data_key=...)
all_df["x"][seg] = df["feature"]
all_df["y"][seg] = df["label"]
all_t[v][seg] = torch.from_numpy(all_df[v][seg].values).float()
```

也就是说，样本在这一层从 **Pandas DataFrame** 进一步变成 **Torch Tensor**，正式进入可训练的内存张量形态。

---

## 3) 送入 MLP：batch 采样与设备传输

在 `DNNModelPytorch.fit()` 的训练循环中：

1. 先从训练集 Tensor 中随机采样一个 batch：
   - `choice = np.random.choice(train_num, self.batch_size)`
2. 取出 batch 特征、标签、权重：
   - `x_batch_auto = all_t["x"]["train"][choice]`
3. 将 batch 传到计算设备（CPU/GPU）：
   - `.to(self.device)`

所以这一步是 **主机内存（CPU Tensor） → 设备内存（GPU Tensor，可选）**。

---

## 4) MLP 内部如何运算与变形（以 `Net` 为例）

Qlib 默认 MLP `Net` 结构在 `qlib.contrib.model.pytorch_nn.Net`：

- 输入：`[batch_size, input_dim]`
- 网络层序列：
  - Dropout(0.05)
  - 若干个 `Linear + BatchNorm1d + 激活(LeakyReLU/SiLU)`
  - Dropout(0.05)
  - 输出层 `Linear(hidden_units, output_dim)`
- 输出：`[batch_size, output_dim]`（通常 `output_dim=1`）

`forward` 的逻辑是把输入依次通过 `self.dnn_layers`：

```python
cur_output = x
for now_layer in self.dnn_layers:
    cur_output = now_layer(cur_output)
return cur_output
```

### 形状示例

若配置：

- `batch_size=2000`
- `input_dim=360`
- `layers=(256,)`
- `output_dim=1`

则一次前向大致形状变化：

- `x`: `[2000, 360]`
- 经过隐藏层后：`[2000, 256]`
- 输出层后：`[2000, 1]`

---

## 5) 损失、反向传播、参数更新

同一训练 step 中：

1. 前向：`preds = self.dnn_model(x_batch_auto)`
2. 损失（MSE）会先 reshape：
   - `pred.reshape(-1), target.reshape(-1), w.reshape(-1)`
3. 计算加权均方误差：
   - `((pred-target)^2 * w).mean()`
4. 反向与更新：
   - `cur_loss.backward()`
   - `self.train_optimizer.step()`

这就完成了从输入样本到梯度更新的一次闭环。

---

## 6) 验证与早停

每隔 `eval_steps`，会在 valid 集上整批推理并计算 loss/metric：

- `preds = self._nn_predict(all_t["x"]["valid"], return_cpu=False)`
- 更新最优模型并 `torch.save(...)`
- 若连续若干轮无提升，触发 `early_stop_rounds`

因此模型参数并非使用最后一步，而是回滚到验证集最优权重。

---

## 7) 预测输出值如何产生

`predict()` 时的流程：

1. `x_test_pd = dataset.prepare(segment, col_set="feature", data_key=DK_I)`
2. `_nn_predict` 按小批量前向，得到 numpy 预测数组。
3. 返回：

```python
pd.Series(preds.reshape(-1), index=x_test_pd.index)
```

所以最终输出不仅有预测值，也保留了 `(datetime, instrument)` 索引，方便后续回测/分析模块直接消费。

---

## 8) 一张总流程图（文字版）

```text
磁盘二进制数据
   ↓ (D.features)
Pandas DataFrame(feature/label, MultiIndex)
   ↓ (DatasetH.prepare + DataHandlerLP)
train/valid/test 切分后的 DataFrame
   ↓ (torch.from_numpy)
CPU Tensor(x, y, w)
   ↓ (.to(device))
GPU/CPU batch Tensor
   ↓ (MLP forward: Linear/BN/Activation/...)
pred Tensor
   ↓ (MSE + backward + optimizer.step)
更新后的模型参数
   ↓ (predict)
pd.Series(pred, index=[datetime, instrument])
```

---

## 9) 你可以重点记住的 4 个“边界”

1. **磁盘→内存边界**：`D.features(...)` 把底层存储读成 DataFrame。
2. **DataFrame→Tensor 边界**：`torch.from_numpy(...).float()`。
3. **CPU→GPU 边界**：`.to(self.device)`。
4. **Tensor→业务输出边界**：`pd.Series(preds, index=...)`。

把这 4 个边界理解清楚，Qlib 中绝大多数深度学习模型（不仅 MLP）的 data flow 都能快速看懂。
