# Qlib 中截面模型与时间序列模型两条数据路线：设计思想、数据流与训练差异

本文从**自上而下**的角度，系统梳理 Qlib 在深度学习场景中两条常见路线：

- **截面（Cross-Sectional）路线**：同一时点的不同股票样本（或 tabular 样本）作为训练对象；
- **时间序列（Time-Series）路线**：单只股票在一段历史窗口上的序列样本作为训练对象。

并回答三个核心问题：

1. 两种路线在 Qlib 中的**设计思想**是什么？
2. 两种路线的数据在系统里分别如何**流动与变形**？
3. 两种路线在**训练方法**上有哪些关键差异？

---

## 1. 顶层设计：Qlib 为什么同时支持两种模式？

Qlib 的数据底座是统一的二维表结构（`datetime`, `instrument` 索引 + 特征列/标签列），但不同模型对“一个样本”的定义不同：

- 截面模型认为：
  - 一个样本 ≈ 某个 `(datetime, instrument)` 下的一行特征向量；
  - 重点建模“同一时点各股票之间”的相对关系与横截面对比。
- 时间序列模型认为：
  - 一个样本 ≈ 同一股票在过去 `step_len` 个时点组成的特征序列；
  - 重点建模“单股票沿时间轴的动态演化”。

因此 Qlib 采用了“**同一底层数据 + 两种数据组织器**”的方案：

- `DatasetH`：输出 tabular 视图，适合截面与传统模型；
- `TSDatasetH`：在 `DatasetH` 之上额外做序列化，输出 `TSDataSampler`，适合时序模型。

这套设计允许模型层选择不同输入形状，但共享同一个数据处理生态（handler、processor、segment、DK_I/DK_L 等）。

---

## 2. 公共基础层：两条路线共享的数据入口

无论截面还是时序，都先走统一入口：

1. `Dataset.prepare(...)` / `DatasetH.prepare(...)`
2. 传入 `segment`（train/valid/test）、`col_set`（feature/label）、`data_key`（`DK_L`/`DK_I`）
3. 下沉到 `DataHandlerLP.fetch(...)` 从 `_learn` 或 `_infer` 视图取数据

其中：

- `DK_L`：训练态（learn）数据；
- `DK_I`：推理态（infer）数据。

这意味着两条路线的分歧并不发生在“读取数据”这一步，而是发生在“**如何把同一份二维数据组织成模型样本**”这一步。

---

## 3. 截面路线（Cross-Sectional）

### 3.1 设计思想

截面路线假设：每个样本可被独立表示为一个固定长度向量，模型主要学习“特征到标签”的映射，不强依赖同一股票的历史轨迹。

这条路线适合：

- MLP/Linear/GBDT/TabNet 等 tabular 建模；
- 或者虽为神经网络，但输入仍是二维特征（`[batch, feat]`）。

### 3.2 数据流动状态（以 `DatasetH + DNNModelPytorch` 为代表）

1. `dataset.prepare("train", col_set=["feature", "label"], data_key=DK_L)`
   - 返回 DataFrame（MultiIndex：datetime/instrument）。
2. 训练器将 DataFrame 拆出 `feature` 与 `label`；
3. `torch.from_numpy(...).float()` 转成张量；
4. 训练时随机采样 batch（`np.random.choice`）；
5. 喂给 MLP `Net`：输入形状 `[batch, input_dim]`，输出 `[batch, 1]`。

这是一条典型 **2D Tensor 路线**。

### 3.3 截面路线的两种 batch 组织方式

虽然都属于截面思想，但实现上有两种常见 batch 策略：

- **随机样本 batch（最常见）**：
  - 如 `DNNModelPytorch`，从全体训练样本随机抽取行，batch 可能混合多个日期。
- **按日截面 batch（更“纯截面”）**：
  - 如 HIST/GATs 一类实现，会先按 `datetime` 分组，然后每个 batch 是“某一天的全部股票样本”。

两者输入仍是二维（每行一只股票），区别在 batch 的统计结构：

- 随机行 batch 更像 i.i.d. 近似；
- 按日 batch 更强调“同一天横截面”的联合关系。

---

## 4. 时间序列路线（Time-Series）

### 4.1 设计思想

时间序列路线假设：标签依赖于**同一股票过去一段时间**的状态，样本的最小单元是序列窗口。

因此，Qlib 不直接返回 DataFrame 给模型，而是通过 `TSDatasetH` 将二维表重排成可索引的时序样本集合。

### 4.2 `TSDatasetH` 的关键机制

`TSDatasetH` 在 `DatasetH` 基础上增加了三件事：

1. `step_len`：定义历史窗口长度；
2. `_extend_slice(...)`：对 train/valid/test 的时间片向前补足窗口所需历史；
3. `_prepare_seg(...) -> TSDataSampler`：返回一个类 Dataset 的对象，`__getitem__` 能按样本索引产出 `[step_len, feat(+label)]`。

直观上，`TSDatasetH` 把“二维大表”变成了“可逐样本抽取的三维时序张量源”。

### 4.3 数据流动状态（以 `TSDatasetH + LSTM` 为代表）

1. `dataset.prepare("train", ...)` 返回 `TSDataSampler`；
2. `DataLoader` 基于 sampler 取 batch；
3. 一个 batch 的 `data` 形状通常是 `[batch, step_len, feat+label]`；
4. 训练器中切分：
   - `feature = data[:, :, 0:-1]` -> `[batch, step_len, feat]`
   - `label = data[:, -1, -1]` -> `[batch]`
5. LSTM/GRU 等模型前向，通常取序列最后时刻隐状态映射到预测值。

这是一条典型 **3D Tensor 路线**。

---

## 5. 两条路线训练方法的核心差异

下面给出最关键的训练差异对照。

| 维度 | 截面路线（DatasetH） | 时间序列路线（TSDatasetH） |
|---|---|---|
| 样本定义 | 单时点单股票的一行特征 | 单股票过去 `step_len` 的窗口序列 |
| 主要张量形状 | `[batch, feat]` | `[batch, step_len, feat]` |
| 数据组织器 | DataFrame / numpy / Tensor | TSDataSampler + DataLoader |
| batch 采样 | 随机行或按日截面 | 以“序列样本”为粒度 |
| 模型层类型 | MLP/Linear/树模型等 | RNN/TCN/Transformer-TS 等 |
| 标签对齐 | 与当前行对齐 | 常见做法是与窗口最后时点对齐 |
| 对时间依赖建模 | 弱/间接（靠特征工程） | 强/直接（网络结构建模时序） |

### 5.1 优化与训练循环上的差异

- 截面 MLP 训练通常是：
  - 随机抽样行 -> 前向 -> MSE/BCE -> 反向更新。
- 时序 LSTM 训练通常是：
  - DataLoader 取序列 -> 切 feature/label -> RNN 前向 -> 损失 -> 反向更新。

二者优化器、早停、学习率调度等框架机制可以一样，但“batch 语义”和“forward 输入结构”本质不同。

### 5.2 `seq_len=1` 的边界情况怎么理解？

如果在 `TSDatasetH` 下设置 `step_len=1`，形状会退化为 `[batch, 1, feat]`：

- 形式上仍是时序路线（仍走 TSDataSampler/3D 输入）；
- 信息上接近截面（没有历史展开）。

所以它是“时序机制下的最短窗口特例”，不是完全等价于 `DatasetH` 截面路线。

---

## 6. 从工程角度如何选路线？

### 6.1 优先选截面路线的场景

- 你主要依赖强特征工程（Alpha 因子已编码历史信息）；
- 模型需要高吞吐、快速迭代；
- 更关注同日横截面排序质量（IC/RankIC）。

### 6.2 优先选时间序列路线的场景

- 你希望模型自动学习时序模式（动量、反转、状态切换）；
- 特征中“路径形态”很关键，不只是某日静态快照；
- 你愿意承担更高计算与调参复杂度。

---

## 7. 一个统一视角：Qlib 的“同源分流”架构

可以把 Qlib 理解为：

1. **同源**：底层仍是同一份处理后的 `(_learn/_infer)` 二维数据；
2. **分流**：
   - `DatasetH` 直接输出 tabular 样本；
   - `TSDatasetH` 输出序列化 sampler；
3. **再汇合**：都进入 PyTorch 训练循环，只是输入张量阶数和 batch 语义不同。

这个架构带来的好处是：

- 你可以在统一的数据处理框架下切换模型范式；
- 能清晰定位问题发生在“数据处理层、样本组织层，还是模型训练层”。

---

## 8. 快速总结

- Qlib 里**确实同时存在**截面与时间序列两种建模路线；
- 区别不在底层数据来源，而在“样本定义与组织方式”：
  - 截面：`DatasetH` -> 2D batch；
  - 时序：`TSDatasetH/TSDataSampler` -> 3D batch；
- 训练差异的根本是：
  - 截面重“横截面映射”；
  - 时序重“历史轨迹建模”；
- `step_len=1` 是时序路线的极短窗口特例，而非完全退化成截面实现。

---

## 9. 形状对照图：同一份数据在 `DatasetH(2D)` 与 `TSDatasetH(3D)` 下的具体例子

下面用一个具体数字例子说明两条路线的数据形状关系。

设定：

- 交易日数 `D = 3`（`d1, d2, d3`）
- 股票数 `T = 4`（`A, B, C, D`）
- 特征数 `F = 5`
- 时序窗口 `step_len = 2`

也就是同一份底层二维数据可理解为逻辑上的 `dates × tickers × features = 3 × 4 × 5`。

### 9.1 DatasetH 路线（2D）：先展平样本轴，再按行采样

```text
逻辑三维视角（便于理解）
X[d, t, f] -> [D=3, T=4, F=5]

DatasetH 实际输出（DataFrame / Tensor）
X_flat -> [N, F], 其中 N = D*T = 12

行索引（MultiIndex）示意：
0:(d1,A) 1:(d1,B) 2:(d1,C) 3:(d1,D)
4:(d2,A) 5:(d2,B) 6:(d2,C) 7:(d2,D)
8:(d3,A) 9:(d3,B) 10:(d3,C) 11:(d3,D)

若 batch_size=4，随机 choice=[10, 2, 7, 0]
则 x_batch_auto 形状 = [4, 5]
对应样本 = [(d3,C), (d1,C), (d2,D), (d1,A)]
```

要点：

- 对 MLP 而言，输入就是 `x_batch_auto: [batch_size, input_dim] = [4, 5]`；
- `date*ticker` 被折叠进“样本轴”，时间维不显式保留在张量维度中。

### 9.2 TSDatasetH 路线（3D）：每个样本是“单股票时间窗口”

```text
同一份底层数据先经 TSDatasetH + TSDataSampler 组织成序列样本：

样本1: (d2, A) -> 窗口 [d1,d2] 的 A -> [step_len=2, F=5]
样本2: (d2, B) -> 窗口 [d1,d2] 的 B -> [2,5]
...
样本k: (d3, D) -> 窗口 [d2,d3] 的 D -> [2,5]

DataLoader 组 batch 后：
x_batch_ts -> [batch_size, step_len, F]

若 batch_size=4，则 x_batch_ts 形状 = [4, 2, 5]
```

要点：

- 这里“样本轴”不再是单点 `(d,t)` 的快照，而是 `(d,t)` 对应的一段历史窗口；
- 时间维 `step_len` 在张量中是显式存在的，因此是 3D 输入。

### 9.3 一眼看懂版：2D 与 3D 的关系

```text
同源数据（逻辑）: [D, T, F] = [3,4,5]

DatasetH:
  reshape(展平 D*T) -> [N, F] = [12,5]
  sample rows -> [B, F]

TSDatasetH:
  以每个 (d,t) 取历史窗口 -> [step_len, F]
  stack batch -> [B, step_len, F]
```

你可以把它理解为：

- `DatasetH`：把 `D*T` 先压成一个样本轴；
- `TSDatasetH`：在每个样本内部保留时间轴，再组成 batch。
