# Chapter 7 — Losses and optimizers from first principles

*Prerequisites: Ch. 6. Builds: the LossSpec/OptimSpec plug-ins — your question 3, part 2. This chapter is where "master" starts to diverge from "user of libraries."*

---

## 7.1 First principles: what are we actually optimizing?

The judge (Ch. 8) scores a *portfolio built from your score ordering*, net of costs. The chain is:

$$\text{scores} \xrightarrow{\text{rank}} \text{positions} \xrightarrow{\text{hold, pay costs}} \text{PnL} \xrightarrow{\text{aggregate}} \text{Sharpe/IR}$$

MSE on returns is far to the left of that chain. Each loss below moves the training objective rightward — closer to the judge — at the price of noisier gradients and more machinery. The art is choosing your point on that frontier *deliberately*.

**Losses are 20-line functions in Loom** (`LOSSES` registry, consumed by the Trainer). Each declares `cross_sectional: bool` so the Trainer picks the right sampler (Ch. 5 §5.4). Compare: changing the loss in qlib means editing 27 training loops (Ch. 3, P4).

## 7.2 MSE — know exactly what you're assuming

$$\mathcal{L} = \tfrac{1}{n}\sum (\hat y_i - y_i)^2$$

MSE is maximum likelihood under Gaussian noise with *uniform variance*. Both assumptions are false here (fat tails; volatility varies 10× across stocks), with real consequences: the loss is dominated by high-volatility names and outlier days, so the model spends capacity predicting *magnitude* where magnitude is least predictable — while the judge only scores *order*. Mitigations if you stay with MSE: rank-normalize labels (Ch. 5 §5.2 — this quietly fixes most of it, and is why qlib's MSE-everywhere is less broken than it looks), winsorize, or switch to Huber. MSE with ranked labels is a perfectly respectable rung-0 loss. But we can do better:

## 7.3 IC loss — train on the judge's metric

Daily IC is Pearson correlation between scores and returns within a date. It's differentiable — so use it *as the loss*, batching one date at a time:

```python
def ic_loss(pred, y, w=None):                    # pred, y: one full date's cross-section
    pred = pred - pred.mean();  y = y - y.mean()
    ic = (pred * y).sum() / (pred.norm() * y.norm() + 1e-8)
    return -ic                                    # maximize IC
LOSSES["ic"] = Loss(ic_loss, cross_sectional=True)
```

Properties worth understanding, not just using: it's **scale- and shift-invariant** (only the direction of the score vector matters — magnitude is unconstrained, so add small weight decay to keep weights bounded); it **normalizes each date to equal importance** (per-era balancing from Ch. 5 §5.5, for free — a crisis day can't dominate the gradient); it aligns training with the reported metric almost exactly. Weaknesses: with small cross-sections (N < ~50) the per-batch IC is very noisy — average over several dates per batch; and it's still *linear* correlation — outlier returns retain leverage. Hence:

## 7.4 Ranking losses — order is all you need

**Soft RankIC.** RankIC = Pearson on *ranks*; ranking is a step function (no gradient), so substitute a differentiable **soft rank** (each element's rank ≈ sum of sigmoids of pairwise differences, temperature τ controlling sharpness) and apply `ic_loss` on soft-ranked predictions vs. ranked labels. Robust to outliers on both sides; τ is a real hyperparameter (too sharp → vanishing gradients; too soft → it's just IC again).

**Pairwise (RankNet-style).** Sample pairs $(i, j)$ within a date with $y_i > y_j$; penalize inversions:
$$\mathcal{L} = \log\left(1 + e^{-(\hat y_i - \hat y_j)}\right)$$
Sample pairs with *large* label gaps (top-vs-bottom decile) — those are the pairs the portfolio monetizes; adjacent-rank pairs are noise. O(pairs) cost, embarrassingly effective.

**Listwise (ListMLE).** Model the probability of the observed full ordering (Plackett–Luce) and maximize its likelihood. Theoretically cleanest; in practice sensitive to exactly the noisy mid-ranks pairwise sampling let you ignore. Try pairwise first.

**⚖ Design call.** Loom's default loss ladder: `mse(ranked labels)` → `ic` → `pairwise(top/bottom decile)`. Each is one `LossSpec` string away; the ledger will tell *you* which earns its keep on *your* features — the literature's answer ("ranking helps, modestly, usually") is a prior, not a verdict.

## 7.5 Sharpe surrogates — differentiate through the portfolio

The frontier's right edge: build the portfolio *inside* the loss and optimize its risk-adjusted PnL directly.

```python
def sharpe_loss(pred_seq, y_seq, temp=20.0, cost=5e-4):
    # pred_seq, y_seq: (D, N) — a WINDOW of dates (needs a window sampler)
    w = torch.softmax(pred_seq * temp, dim=1) - torch.softmax(-pred_seq * temp, dim=1)
    w = w / w.abs().sum(dim=1, keepdim=True)               # soft long-short, unit gross
    pnl = (w * y_seq).sum(1) - cost * (w[1:] - w[:-1]).abs().sum(1).pad_front(1)
    return -(pnl.mean() / (pnl.std() + 1e-8))
```

Now turnover costs and concentration are *inside the gradient* — the model learns to prefer stable rankings because unstable ones pay costs. This is the only loss family that can teach a model about **transaction costs**, which no IC-family loss sees at all. Costs (pun available): the loss is a ratio of window statistics → high variance and non-convex; needs date-window batching; softmax temperature is now part of your strategy definition. Use it as a *fine-tuning* stage on a model pre-trained with IC/pairwise loss, not from scratch — a curriculum across the frontier.

**Multi-task, briefly:** predict return *and* volatility with two heads, `L = L_ret + λ L_vol`. The vol head is easy signal (vol is very predictable), acts as a regularizing auxiliary task, and gives you a sizing input for Ch. 8's portfolio. Cheap, often worth it.

## 7.6 Optimizers: what actually matters here

The ranked-by-impact truth for this domain:

1. **AdamW, lr 1e-3 → 1e-4, weight decay 1e-3 → 1e-2.** Decoupled weight decay (the W) matters because capacity control is the main event (Ch. 1). This is the default; you need a reason to leave it.
2. **Schedule: warmup (3–5 epochs) + cosine decay.** Warmup matters more than usual — early gradients from IC/ranking losses are high-variance (small effective batch = one date), and a hot start walks into a bad basin.
3. **Gradient clipping (norm 1–5): non-negotiable.** Crisis-day cross-sections produce gradient spikes that can destroy a converged model in one step. The Trainer clips unconditionally (Ch. 6's loop already does).
4. **Batch composition is secretly an optimizer choice.** One date per step = high variance; 8 dates per step = smoother but slower adaptation. Tune it like a learning rate — it *is* one, in disguise.

```python
@dataclass(frozen=True)
class OptimSpec:
    name: str = "adamw"; lr: float = 1e-3; wd: float = 1e-3
    warmup: int = 5; clip: float = 3.0
    def build(self, params): ...
```

## 7.7 Custom optimizers — the pedagogy of writing one

Write one optimizer from scratch in your life; this domain offers a genuinely apt candidate. **SAM (Sharpness-Aware Minimization)** seeks flat minima — parameters whose neighborhood is uniformly good — and flat minima are precisely the ones most robust to distribution shift, which is our disease (Ch. 1 §1.3b; the full loss-landscape theory arrives in Ch. 17):

```python
class SAM(torch.optim.Optimizer):
    """Two-step: ascend to the worst nearby point, then descend from THERE."""
    def __init__(self, params, base=torch.optim.AdamW, rho=0.05, **kw):
        super().__init__(params, dict(rho=rho)); self.base = base(self.param_groups, **kw)
    @torch.no_grad()
    def first_step(self):                         # perturb: w ← w + rho * g/|g|
        gn = torch.norm(torch.stack([p.grad.norm() for g in self.param_groups
                                     for p in g["params"] if p.grad is not None]))
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                p.state = p.clone();  p.add_(p.grad, alpha=g["rho"] / (gn + 1e-12))
    @torch.no_grad()
    def second_step(self):                        # restore w, step base optimizer
        for g in self.param_groups:
            for p in g["params"]:
                if hasattr(p, "state"): p.copy_(p.state)
        self.base.step()
```

(The Trainer needs a two-closure variant of its step for SAM — a 5-line change *in one place*; savor that.) Other levers worth knowing: **parameter groups** (no weight decay on biases/norms; lower lr on pretrained embeddings), **Lookahead** as a wrapper (slow weights average fast weights — a 15-line stabilizer), and **EMA of weights** for free ensembling. Whether SAM/Lookahead beat tuned AdamW *on your problem* is a ledger question — the meta-lesson of this whole brochure.

## 7.8 Build exercise (3–4 hours)

1. Implement `LOSSES`: `mse`, `ic`, `pairwise` (top/bottom-quintile pair sampling). Unit-test `ic_loss` against `scipy.stats.pearsonr` on random vectors.
2. Same GRU, same splits, three losses. Three ledger entries. Compare RankIC *and* the top-minus-bottom decile spread (Ch. 8) — ranking losses often improve the spread more than the IC; understand why (where in the ordering does each loss spend its gradients?).
3. Implement warmup+cosine and confirm the early-epoch validation-IC variance drops vs. constant lr.
4. Stretch: SAM. Compare against AdamW across *5 seeds each* (Ch. 6's spread discipline — SAM's claim is precisely lower spread). Write the verdict in the ledger, whatever it is.

**Next:** the judge itself — evaluation and backtesting, and the statistics that keep you from fooling yourself.
