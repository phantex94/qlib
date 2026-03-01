# Qlib 中 Transformer 与 Transformer_TS 模型说明（`qlib/contrib/model`）

本文专门说明 Qlib 中两个 Transformer 实现：

- `qlib.contrib.model.pytorch_transformer.TransformerModel`
- `qlib.contrib.model.pytorch_transformer_ts.TransformerModel`

二者模型骨架都基于 `nn.TransformerEncoder`，但**数据组织方式与训练入口不同**：

- `pytorch_transformer.py`：偏截面/表格路线（`DatasetH`）。
- `pytorch_transformer_ts.py`：偏时间序列路线（`TSDatasetH` + `TSDataSampler`）。

---

## 1. 文件定位与配置映射

- 非 `_ts` 版本：`qlib/contrib/model/pytorch_transformer.py`
  - benchmark 示例配置用 `DatasetH`（Alpha360）。
- `_ts` 版本：`qlib/contrib/model/pytorch_transformer_ts.py`
  - benchmark 示例配置用 `TSDatasetH` 且 `step_len: 20`（Alpha158）。

这说明 Qlib 官方示例中这两者已按“2D/3D 数据路线”区分使用。

---

## 2. 共同点：网络主体几乎一致

两份代码中的 `Transformer` 子模块都采用以下结构：

1. `feature_layer: Linear(d_feat -> d_model)`
2. `PositionalEncoding`
3. `TransformerEncoderLayer + TransformerEncoder`
4. `decoder_layer: Linear(d_model -> 1)`
5. 取最后时刻 token 输出做回归

并且二者 `forward` 的处理逻辑是“同骨架、不同入口形状”：

- `transformer`：先把输入从 `[N, F*T]` reshape 为 `[N, T, F]`，再转置为 `[T, N, F]` 喂给 encoder；
- `transformer_ts`：输入本来就是 `[N, T, F]`，直接转置为 `[T, N, F]` 喂给 encoder；
- 最后都取最后时刻输出映射到标量预测。

所以从“网络计算图”看，二者核心思想一致：都把样本视为一段长度为 `T` 的序列。

---

## 3. 关键差异一：数据入口不同（`DatasetH` vs `TSDatasetH`）

### 3.1 `pytorch_transformer.py`（非 `_ts`）

`fit()` 中直接：

```python
df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], ...)
```

随后把 `x_train.values` / `y_train.values` 直接转 tensor，按索引切 batch。

这条路依赖 `DatasetH` 给出的 tabular DataFrame，再由模型代码内部 reshape 成序列输入。

### 3.2 `pytorch_transformer_ts.py`（`_ts`）

`fit()` 中：

```python
dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DK_L)
```

这里的 `dl_train` 是 `TSDatasetH` 返回的 `TSDataSampler`，再交给 `DataLoader` 组 batch；
样本天然是时间窗口，不需要在训练循环里自己重排索引关系。

---

## 4. 关键差异二：训练循环的 batch 组织不同

### 4.1 非 `_ts`：DataFrame/numpy 驱动

- `x_train_values = x_train.values`
- shuffle 索引后按 `batch_size` 切片
- 每个 batch 直接 `feature = torch.from_numpy(...)`

特点：

- “样本是行”这一抽象更明显；
- 批处理逻辑由模型训练代码自己控制。

### 4.2 `_ts`：Sampler/DataLoader 驱动

- `DataLoader(ConcatDataset(dl_train, wl_train), ...)`
- 训练中从 `data` 切出 `feature = data[:, :, 0:-1]`

特点：

- 序列窗口由 dataset 层保证；
- 训练代码更像标准时序模型（LSTM/GRU 的写法）。

---

## 5. 关键差异三：典型使用场景

### `pytorch_transformer.py` 更适合

- 已有 tabular 流程（`DatasetH`）并希望快速替换模型为 Transformer；
- 更偏截面样本组织，时间窗口构造不完全交给 dataset 层。

### `pytorch_transformer_ts.py` 更适合

- 明确希望走“单股票时间窗口”范式；
- 与 LSTM/GRU/TCN 的 `TSDatasetH` 训练范式保持一致；
- 希望通过 `step_len` 统一管理历史长度。

---

## 6. 形状对照（直观版）

设：

- `batch_size = B`
- `step_len = T`
- `d_feat = F`

两者在前向阶段的关键形状是：

- 非 `_ts`：输入先是 `x_batch: [B, F*T]`，在 `forward` 内 reshape 为 `[B, T, F]`；
- `_ts`：输入直接是 `x_batch: [B, T, F]`；
- 二者随后都转置为 `[T, B, d_model]`，最后输出 `[B]`（或 `[B, 1]` 再 squeeze）。

差别在于：

- 非 `_ts` 版本：序列维度是由特征平铺后在 `forward` 内恢复；
- `_ts` 版本：序列维度在数据集阶段就显式保留。

---

## 7. 如何选型（实践建议）

如果你已经在项目里用：

- `DatasetH`（以及大量 tabular 模型）➡ 优先尝试 `pytorch_transformer.py`。
- `TSDatasetH`（以及 LSTM/GRU 一套范式）➡ 优先使用 `pytorch_transformer_ts.py`。

简单记忆：

- `transformer`：偏“表格管线接入 Transformer”。
- `transformer_ts`：偏“标准时序管线中的 Transformer”。

