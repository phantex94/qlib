# Chapter 6 — Building your own model

*Prerequisites: Ch. 4–5, working PyTorch. Builds: the model contract, the one Trainer, and the capability ladder from ridge to cross-sectional Transformers. This is your question 3, part 1; losses and optimizers get their own chapter next.*

---

## 6.1 The contract

Everything the rest of Loom knows about models:

```python
class Model(Protocol):
    def fit(self, train: TensorSet, valid: TensorSet, spec: TrainSpec) -> "Artifact": ...

class Artifact(Protocol):
    def predict(self, X) -> np.ndarray          # scores, one per (date, entity) row
    def save(self, path) / load(path)
```

Scores are *ranks-to-be* — arbitrary scale, only order matters cross-sectionally (the Evaluator rank-normalizes anyway). For gradient models, `fit` is not implemented per model (qlib's P4); it's the shared Trainer:

```python
def train(spec: TrainSpec, tr: TensorSet, va: TensorSet) -> Artifact:
    torch.manual_seed(spec.seed)
    net   = spec.model.build(tr.n_features)          # nn.Module — the ONLY per-model part
    lossf = LOSSES[spec.loss.name](**spec.loss.kwargs)         # Ch. 7
    opt   = spec.optim.build(net.parameters())                 # Ch. 7
    sampler = DateBatchSampler(tr) if lossf.cross_sectional else RowSampler(tr)
    best, patience = None, spec.patience
    for epoch in range(spec.epochs):
        net.train()
        for batch in sampler:
            opt.zero_grad()
            loss = lossf(net(batch.X), batch.y, batch.w)       # weights from §5.5
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), spec.clip)
            opt.step()
        score = rank_ic(predict(net, va), va)                  # early stop on the JUDGE’s
        best, patience = keep_best(best, net, score, patience) #   metric, not the loss
        if patience == 0: break
    return TorchArtifact(best)
```

~40 real lines, written once. **Early stopping on validation RankIC, not validation loss**, deserves emphasis: loss and IC frequently disagree (a model can trade MSE for better ordering), and the judge is IC. Non-gradient models (§6.2–6.3) implement `fit` directly — the contract, not the Trainer, is the law.

## 6.2 Rung 0: Ridge regression — the mandatory baseline

Always first, no exceptions:

```python
class RidgeModel:
    def fit(self, tr, va, spec):
        from sklearn.linear_model import Ridge
        return SkArtifact(Ridge(alpha=spec.model.alpha).fit(tr.X, tr.y, sample_weight=tr.w))
```

Why mandatory: (a) it establishes the **floor** — any deep model that can't beat ridge OOS is negative progress dressed in CUDA; (b) it's nearly overfit-proof, so it *validates your pipeline* — if ridge shows IC 0.15, you have a leak, not a discovery (real features give ridge ~0.01–0.03); (c) its coefficients tell you which features carry signal linearly, which is most of the signal there is in this domain. Sobering, published fact (Gu–Kelly–Xiu 2020): linear-ish methods capture the large majority of extractable predictability; deep nets add real but *modest* gains. Calibrate expectations accordingly.

## 6.3 Rung 1: Gradient-boosted trees — the honest benchmark

LightGBM is the strongest tabular baseline in finance, usually by a margin: native missing-value handling, automatic interactions, monotonic-constraint support, trains in minutes. Wrap it in the same contract (`LgbModel.fit` → `LgbArtifact`). **Your deep models are judged against LightGBM, not against ridge.** In qlib's own benchmark tables, LightGBM sits stubbornly near the top of the model zoo — a fact worth internalizing before falling in love with attention mechanisms. Also steal its `feature_importance()`: it's your cheapest feature-selection signal for Ch. 9's loop.

## 6.4 Rung 2: MLP — deep learning begins

```python
class MLP(nn.Module):
    def __init__(self, F, hidden=(256,128,64), p=0.4):
        layers = []
        for h in hidden:
            layers += [nn.Linear(F, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            F = h
        layers += [nn.Linear(F, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)
```

The low-SNR playbook (applies to every rung above this one):
- **Small.** 2–4 layers. Parameters are noise-memorization capacity here (Ch. 1 §1.3a).
- **Heavy dropout** (0.3–0.5) and weight decay; early stopping *always* triggers before max epochs — if it doesn't, the model is undertrained or the validation metric is broken.
- **Seed ensembles:** train 5 seeds, average scores. In this noise regime, seed variance is embarrassing (±30% of your IC); the ensemble is both variance reduction and an honesty device — *report the ensemble and the per-seed spread* in the ledger. A method whose seeds disagree wildly is fragile regardless of its mean.

## 6.5 Rung 3: Sequence models — let the network see time

Input becomes `(N, T, F)`: trailing `T ≈ 20–60` days of features per sample (PanelBuilder's sequence packing, §5.4). Architecture is the *only* thing that changes — Trainer, losses, evaluation are untouched. That orthogonality is Loom paying rent.

```python
class GRUNet(nn.Module):                      # qlib's 339-line pytorch_gru.py, the honest size
    def __init__(self, F, hidden=64, layers=2, p=0.2):
        self.rnn  = nn.GRU(F, hidden, layers, batch_first=True, dropout=p)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):                     # x: (N, T, F)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)
```

- **GRU/LSTM:** the workhorse; GRU ≈ LSTM here, fewer parameters.
- **TCN (dilated causal convolutions):** parallelizable, stable, explicit receptive field; causal padding *enforces* no intra-window lookahead architecturally — a firewall in the weights. Often matches RNNs at lower cost.
- **Attention over time (a small Transformer encoder):** add positional encodings, mask the future (causal mask — never optional). Wins when long-range temporal structure exists; in daily equities, the honest finding is "sometimes, modestly."

**What sequence models actually buy you:** the ability to learn *feature dynamics* (momentum of volatility, acceleration) that you'd otherwise hand-engineer. If your feature set already includes rich trailing statistics, rung 3 often adds little over rung 2 — test, don't assume.

## 6.6 Rung 4: Cross-sectional models — let stocks see each other

Everything so far scores each entity independently; the market is relative (Ch. 1). Two upgrades:

- **Cross-sectional attention:** for each date, run self-attention *across the N entities* — each stock's representation attends to every other's. The model can learn "rank me relative to my industry," lead–lag effects, crowding. Batch = one full date (the `DateBatchSampler` again — infrastructure decisions echo).

```python
class XSecTransformer(nn.Module):
    def __init__(self, F, d=64, heads=4, blocks=2):
        self.embed  = nn.Linear(F, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, blocks)
        self.head   = nn.Linear(d, 1)
    def forward(self, x):            # x: (1, N_stocks, F) — one DATE per forward pass
        return self.head(self.encoder(self.embed(x))).squeeze(-1)
```

- **Graph variants:** restrict attention to sector/supply-chain neighbors (a mask), injecting structure as inductive bias — helpful exactly when data is short relative to capacity, i.e., always (Ch. 1 §1.3c).

This rung is where deep learning earns its keep in equities — it models the *interaction structure* that tabular and per-entity sequence models cannot represent at all.

## 6.7 Choosing your rung (decision discipline)

```
Have a trusted pipeline?  ──no──► Ch. 5. Nothing here matters yet.
   └─yes─► Ridge floor + LightGBM benchmark (half a day, both).
              └─► MLP + seed ensemble beats LGBM OOS?
                     ├─ no ──► your edge is in FEATURES, not architecture.
                     │         Ch. 9's loop, feature direction.   (most common outcome)
                     └─ yes ─► sequence model if features are dynamic-poor;
                               cross-sectional attention if you believe in interactions.
```

The uncomfortable truth this ladder encodes: **architecture is rarely the binding constraint in daily-frequency finance.** Features, labels, loss (Ch. 7), and the improvement loop (Ch. 9) usually dominate. Master the ladder so you know *when* to climb — mastery includes declining to.

## 6.8 Build exercise (4 hours)

1. Implement the Trainer exactly once, with the contract above. Then implement `RidgeModel`, `LgbModel`, `MLP`, `GRUNet` — the last two must share the Trainer *unmodified*.
2. Run all four through your Ch. 5 walk-forward on real data. Ledger the results. Expected shape: LightGBM ≥ MLP ≥ GRU ≥ ridge > 0, with overlapping seed spreads — if GRU dominates everything by 3×, hunt the leak before celebrating.
3. Seed-ensemble the MLP (5 seeds). Record mean and spread of RankIC. Write one honest ledger note: "is my deep model's edge real, given the spread?"

**Next:** the loss function — where we stop optimizing what's easy and start optimizing what the judge scores.
