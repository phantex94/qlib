.. _pytorch_lstm_gru_zh:

====================================================
Qlib 时序深度学习实现解读：pytorch_lstm.py 与 pytorch_gru.py
====================================================

概览
====

``qlib/contrib/model/pytorch_lstm.py`` 与 ``qlib/contrib/model/pytorch_gru.py``
是 Qlib 中两套结构几乎平行的 PyTorch 时序模型实现：

- ``LSTM`` / ``GRU`` 类负责训练流程管理（数据准备、epoch 迭代、early stop、保存参数、推理）。
- ``LSTMModel`` / ``GRUModel`` 类负责神经网络结构定义（RNN 主体 + 线性输出层）。

二者都继承 ``qlib.model.base.Model``，因此可以直接接入 Qlib 的 Dataset/Workflow 体系。


模型输入与张量形状
==================

这两个实现都假设输入样本是二维平铺特征：``[N, F*T]``，其中：

- ``N``: batch size
- ``F``: 每个时间步的特征数（``d_feat``）
- ``T``: 序列长度（由输入总维度与 ``d_feat`` 自动推断）

在 ``forward`` 中，都会进行同样的变换：

1. ``reshape(len(x), d_feat, -1)`` 把平铺向量还原为 ``[N, F, T]``。
2. ``permute(0, 2, 1)`` 交换维度得到 ``[N, T, F]``（PyTorch RNN 的 ``batch_first=True`` 格式）。
3. 输入 RNN（LSTM 或 GRU），取最后一个时间步的隐藏状态 ``out[:, -1, :]``。
4. 经过 ``Linear(hidden_size, 1)`` 得到单值预测并 ``squeeze()``。

这意味着：模型学习的是“过去 T 个时间步 -> 当前单值目标”的映射。


训练主流程（fit）
=================

共同机制
--------

两者的训练流程骨架一致：

- 从 ``DatasetH`` 读取 ``feature`` 和 ``label``。
- 按 batch 训练：前向、loss、反向传播、梯度裁剪、优化器更新。
- 每个 epoch 后评估得分（默认 ``-loss``）并执行 early stopping。
- 记录最佳参数 ``best_param``，训练结束后回滚到最佳 epoch 并 ``torch.save``。

损失与评估函数也一致：

- ``loss_fn`` 当前只支持 ``mse``。
- 对 label 的 NaN/非有限值做掩码过滤，避免脏标签污染训练与打分。
- ``metric==""`` 或 ``"loss"`` 时，返回 ``-loss``，使“越大越好”与 early stop 逻辑统一。

LSTM 版本的关键点
-----------------

``pytorch_lstm.py`` 的 ``fit`` 固定请求 ``train/valid/test`` 三段数据，
并要求 train 与 valid 都非空，否则报错。
它会始终在验证集上做 early stop。

GRU 版本的关键点
----------------

``pytorch_gru.py`` 的 ``fit`` 在数据段处理上更灵活：

- 仅强制要求 ``train`` 存在。
- ``valid`` 可选；若缺失则不会进行验证集 early stop 更新。
- 训练/验证数据都会 ``dropna()``，对缺失值更稳健。

此外，GRU 在训练后会把 ``evals_result`` 逐步写入 ``R.get_recorder()``，
方便在 Qlib Workflow 中查看可视化曲线。


LSTM 与 GRU 实现差异（工程视角）
===============================

1. **网络单元不同**

   - LSTM 使用 ``nn.LSTM``（带 cell state，参数更多）。
   - GRU 使用 ``nn.GRU``（门控更简化，通常更轻量）。

2. **评估阶段是否显式 no_grad**

   - GRU 的 ``test_epoch`` 使用了 ``with torch.no_grad()``。
   - LSTM 的 ``test_epoch`` 没有显式 ``no_grad``，虽不影响数值正确性，但会增加一些显存/图构建开销。

3. **数据段依赖性**

   - LSTM 对 ``valid`` 是强依赖。
   - GRU 支持无验证集训练（但没有验证监督时，最佳参数可能长期保持初始化或较早状态）。

4. **实验追踪**

   - GRU 默认向 Recorder 写指标。
   - LSTM 当前仅在 logger 中输出，不主动写 Workflow recorder。


超参数理解与调参建议
===================

- ``d_feat``：每个时间步特征维度。必须与输入编码方式一致。
- ``hidden_size``：时序隐层宽度。增大可提升表达能力，也会增加过拟合风险。
- ``num_layers``：RNN 堆叠层数。层数越深，训练稳定性越依赖学习率和梯度裁剪。
- ``dropout``：只在 ``num_layers>1`` 时由 PyTorch RNN 生效。
- ``batch_size``：实现里会直接丢弃最后一个不足 batch 的尾部样本；
  若数据量较小，建议适当减小 batch。
- ``early_stop``：验证集波动较大时可适当增大，避免过早停止。


常见坑位与排查清单
==================

1. **输入维度不能被 ``d_feat`` 整除**

   ``reshape`` 会失败。需要确认特征工程输出的 ``F*T`` 与 ``d_feat`` 一致。

2. **标签含大量 NaN**

   虽然有 mask 保护，但有效样本过少会导致 loss/metric 不稳定。

3. **valid 段为空（LSTM）**

   LSTM 会直接报错；若场景无法提供验证集，可考虑改用 GRU 版本或自定义 fit 逻辑。

4. **GPU 显存压力**

   两者都支持 GPU，并在训练结束后尝试 ``torch.cuda.empty_cache()``；
   仍 OOM 时优先降低 ``batch_size`` 与 ``hidden_size``。


如何在 Qlib 配置中使用
====================

示例（LSTM）：

.. code-block:: python

   task = {
       "model": {
           "class": "LSTM",
           "module_path": "qlib.contrib.model.pytorch_lstm",
           "kwargs": {
               "d_feat": 6,
               "hidden_size": 64,
               "num_layers": 2,
               "dropout": 0.0,
               "n_epochs": 200,
               "lr": 1e-3,
               "batch_size": 2000,
               "early_stop": 20,
               "loss": "mse",
               "optimizer": "adam",
               "GPU": 0,
           },
       },
   }

GRU 仅需将 ``class`` 与 ``module_path`` 替换为对应项：

- ``class": "GRU"``
- ``module_path": "qlib.contrib.model.pytorch_gru"``


总结
====

这两份代码是 Qlib 里最直接、可读性高的时序深度学习基线实现：

- 结构简洁，适合作为二次开发模板。
- 与 Qlib Dataset/Workflow 紧密集成，便于纳入完整投研流程。
- GRU 版本在工程鲁棒性（可选 valid、记录指标）上略优；
  LSTM 版本在经典序列建模上更“标准”，但可进一步优化评估阶段内存行为。
