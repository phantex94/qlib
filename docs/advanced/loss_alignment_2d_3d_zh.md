# 从 Loss Function 视角理解 DatasetH / TSDatasetH 的样本-Label 对齐

本文只聚焦一个问题：

> **在计算 loss 时，模型输出 `pred` 与标签 `label` 是如何一一对齐的？**

我们分别在两条数据路线下说明：

- `DatasetH`（2D，截面/表格）
- `TSDatasetH`（3D，时序窗口）

并按四个模型层级展开：

1. 朴素版本（线性输出）
2. 全连接神经网络版本（带 `hidden_dim`）
3. 残差连接版本（Residual）
4. Wide-and-Deep 版本

---

## 1. 统一记号（先把“对齐”说清楚）

设底层逻辑数据：

- 特征：`X[d, t, f]`
- 标签：`Y[d, t]`
- `d` 是日期，`t` 是股票，`f` 是特征维

### 1.1 DatasetH（2D）

展平后：

- `X_flat[i] = X[d, t, :]`，形状 `[F]`
- `Y_flat[i] = Y[d, t]`，标量

一个 batch：

- `x_batch`：`[B, F]`
- `y_batch`：`[B]`（或 `[B, 1]`）

**对齐原则**：第 `k` 个预测对应第 `k` 个标签。

### 1.2 TSDatasetH（3D）

每个样本是窗口：

- `x_seq(d,t) = [X[d-L+1,t,:], ..., X[d,t,:]]`，形状 `[L, F]`
- `y_seq(d,t) = Y[d,t]`（锚点/窗口末端标签）

一个 batch：

- `x_batch_seq`：`[B, L, F]`
- `y_batch_seq`：`[B]`

**对齐原则**：第 `k` 条序列样本对应其末端时点标签 `y_batch_seq[k]`。

---

## 2. 朴素版本（线性输出）

## 2.1 DatasetH 视角

模型：

- `pred = x_batch @ w + b`
- `pred` 形状 `[B]`（或 `[B,1]`）

MSE：

```text
loss = mean((pred[k] - y_batch[k])^2), k=1..B
```

这里没有任何时间维，`loss` 完全由“样本轴 k”对齐。

## 2.2 TSDatasetH 视角

先把序列编码为单向量（例如取最后一步线性投影）：

- `h_k = Proj(x_batch_seq[k, -1, :])`
- `pred[k] = Linear(h_k)`

MSE 仍是：

```text
loss = mean((pred[k] - y_batch_seq[k])^2)
```

区别在于：`pred[k]` 来自一个窗口；`y_batch_seq[k]` 是窗口末端标签。

---

## 3. 全连接神经网络版本（带 hidden_dim）

设 `hidden_dim = H`。

## 3.1 DatasetH 视角（标准 MLP）

```text
x: [B,F]
h1 = Act(BN(Linear(F->H)(x)))      -> [B,H]
h2 = Act(BN(Linear(H->H)(h1)))     -> [B,H]   # 可选
pred = Linear(H->1)(h2 or h1)      -> [B,1]
```

loss：

```text
loss = mean((pred.squeeze(-1)[k] - y_batch[k])^2)
```

**对齐不变**：始终是第 `k` 行样本对第 `k` 个标签。

## 3.2 TSDatasetH 视角（先池化/压缩再 MLP）

一种常见方式：

1. 序列先压缩成固定向量（如最后时刻、mean pooling、attention pooling）
2. 再送入 MLP(`hidden_dim=H`)

```text
x_seq: [B,L,F]
z = Pool(x_seq)                    -> [B,F]
h = Act(BN(Linear(F->H)(z)))       -> [B,H]
pred = Linear(H->1)(h)             -> [B,1]
```

loss 形式和 DatasetH 一样，但语义不同：

- `pred[k]` 是由整段窗口导出的预测
- 对齐到 `y_batch_seq[k]`（末端时点标签）

---

## 4. 残差连接版本（Residual）

## 4.1 DatasetH 视角

令输入投影到 `H` 维：

```text
u0 = Linear(F->H)(x)                         -> [B,H]
r1 = Block(u0) = Act(Linear(H->H)(u0))       -> [B,H]
u1 = u0 + r1                                  -> [B,H]
r2 = Block(u1)                                -> [B,H]
u2 = u1 + r2                                  -> [B,H]
pred = Linear(H->1)(u2)                       -> [B,1]
```

loss 仍按样本轴对齐：`pred[k]` 对 `y_batch[k]`。

## 4.2 TSDatasetH 视角

两种主流做法：

- **做法A：先序列池化再残差 MLP**（最直接）
  - `z = Pool(x_seq) -> [B,F]`，后续同上
- **做法B：时序编码器 + 残差头**
  - `z = Encoder(x_seq) -> [B,H]`，再 residual head 输出 `[B,1]`

不论哪种，loss 对齐规则都不变：

```text
loss = mean((pred[k] - y_batch_seq[k])^2)
```

关键是 `pred[k]` 是否使用了窗口历史信息，而不是对齐索引本身。

---

## 5. Wide-and-Deep 版本（示例）

Wide-and-Deep 一般把预测拆成：

- `wide`：线性部分，擅长记忆显式交叉/稀疏模式
- `deep`：非线性部分，擅长泛化
- 最终 `pred = pred_wide + pred_deep`

## 5.1 DatasetH 视角

```text
x: [B,F]
pred_wide = Linear(F->1)(x)                  -> [B,1]
h = MLP(F->H->...->1)(x)                      -> [B,1]
pred = pred_wide + h                          -> [B,1]
loss = mean((pred.squeeze(-1)[k]-y_batch[k])^2)
```

## 5.2 TSDatasetH 视角

把序列分成两路后融合：

- wide 路：对窗口做简单聚合后线性（例如最后时刻或均值）
- deep 路：时序编码 + MLP

```text
x_seq: [B,L,F]
z_wide = WidePool(x_seq)                      -> [B,F]
pred_wide = Linear(F->1)(z_wide)              -> [B,1]

z_deep = DeepEncoder(x_seq)                    -> [B,H]
pred_deep = MLP(H->...->1)(z_deep)             -> [B,1]

pred = pred_wide + pred_deep                   -> [B,1]
loss = mean((pred.squeeze(-1)[k]-y_batch_seq[k])^2)
```

这里最容易犯错的是：

- `pred_wide` 与 `pred_deep` 必须对应同一条序列样本 `k`；
- 融合后的 `pred[k]` 只能与同一 `k` 的末端标签 `y_batch_seq[k]` 计算 loss。

---

## 6. 一个“对齐检查清单”（实操）

无论哪种模型，训练前可快速检查：

1. **shape 对齐**：
   - `pred.shape[0] == label.shape[0] == B`
2. **语义对齐**：
   - DatasetH：同一行 `(d,t)`
   - TSDatasetH：同一窗口锚点 `(d,t)`
3. **索引对齐**：
   - 采样索引（或 DataLoader 顺序）必须同时作用于 feature 与 label
4. **loss 输入对齐**：
   - `pred[k]` 与 `label[k]` 一一对应

只要这 4 条成立，模型结构（朴素/MLP/残差/Wide&Deep）怎么变，loss 对齐都不会错。

---

## 7. 小结

- 从 loss 的角度看，`DatasetH` 与 `TSDatasetH` 的**数学形式可以一致**（如 MSE/BCE）。
- 真正的差异在 `pred[k]` 的信息来源：
  - `DatasetH`：来自单点特征 `X[d,t,:]`
  - `TSDatasetH`：来自窗口特征 `X[d-L+1:d,t,:]`
- 因此“样本-label 对齐”本质是：
  - **2D 对齐单点**，**3D 对齐窗口锚点**。
