# Chapter 12 — Alignment: surviving the optimizer

*Prerequisites: Ch. 7–9. Builds: the bridge between smooth model optimization and real-world trading terrain — liquidity, factor risk, position sizing, costs, short limits — with the constraints placed where they retain signal without inviting overfitting. Origin: Core Research Areas discussion 1.*

---

## 12.1 The objective, corrected: TC · IC, not IC

Chapter 1 gave you $\text{IR} \approx \text{IC} \cdot \sqrt{\text{breadth}}$. That law is for an *unconstrained* trader, and you are not one. Grinold's refinement introduces the missing term:

$$\text{IR} \approx \text{TC} \cdot \text{IC} \cdot \sqrt{\text{breadth}}$$

**TC, the transfer coefficient**, is the cross-sectional correlation between your *ideal* positions (what raw scores imply) and your *actual* positions (what survives the liquidity filter, the beta/factor constraints, position bounds, borrow availability, and cost-aware sizing). Realistic constrained portfolios run TC ≈ 0.3–0.7. Read that again as an engineer: **the execution layer routinely discards 30–70% of the signal after the model is finished.** No architecture change in Ch. 6 moves the needle that much.

Consequences, in the Ch. 1 style:

1. **The research objective is the product TC·IC.** A change that costs 0.005 of IC but lifts TC from 0.45 to 0.65 is a large win — and a pure-IC research loop (which is what Ch. 9 built, so far) will *reject it every time*. That is the misalignment this chapter repairs.
2. **TC is partly a modeling choice**, not just an execution fact. What you train the model to predict determines how much of its output the optimizer can keep (§12.3).
3. **TC belongs in the Evaluator and the ledger**, or the loop stays blind to half the objective (§12.6).

## 12.2 Measure before you architect: the signal attrition waterfall

The three attrition sources — liquidity, risk constraints, costs — have completely different fixes. So before placing constraints anywhere, make the attrition *observable per run*:

```
W0  IC(raw scores, full universe)                       the model's fantasy
W1  IC(scores | tradeable universe as-of t)             liquidity attrition
W2  IC(factor-neutralized scores | tradeable)           risk-model attrition
W3  TC(optimizer weights vs ideal weights)              constraint/turnover attrition
W4  net-of-cost Sharpe                                  execution attrition
```

```python
def waterfall(scores, rets, universe_mask, F, weights_ideal, weights_actual, pnl_net):
    return {
        "W0_ic":  rank_ic(scores, rets),
        "W1_ic":  rank_ic(scores[universe_mask], rets[universe_mask]),
        "W2_ic":  rank_ic(neutralize(scores[universe_mask], F), rets[universe_mask]),
        "W3_tc":  xsec_corr(weights_actual, weights_ideal),
        "W4_sharpe_net": sharpe(pnl_net),
    }
```

Every run logs its waterfall. The Ch. 9 proposer then targets the *largest drop* instead of grinding W0: a big W0→W1 drop means you're training on names you can't trade (§12.3, universe alignment); W1→W2 means your signal is mostly factor bets the risk optimizer strips (§12.3, residualization); W2→W3 means the constraint set or turnover is binding (§12.4–12.5); W3→W4 means costs — slow the labels down (Ch. 8's decay curve tells you how far you can). **This decomposition turns "maximize signal retention" from a slogan into a search direction.**

## 12.3 Strategy 1 — Decoupled, with alignment hygiene (do all of this, always)

Keep prediction and optimization separate; stop wasting model capacity on signal the optimizer will discard. Three interventions, all near-zero overfitting risk:

**Universe alignment.** Train *and* evaluate only on the as-of tradeable universe: ADV floor for everything, borrow-available for the short side. A model spending gradient ranking untradeable micro-caps is pure waste — and worse, micro-cap "alpha" is the easiest false signal to find (wide spreads masquerade as predictability). This is a PanelBuilder mask (Ch. 5 §5.5), point-in-time like everything else: today's borrow list must not filter 2019's training rows.

**Residualized labels — the highest-yield trick in this chapter.** If the optimizer will enforce beta- and sector-neutrality, any signal component aligned with beta or sectors is dead on arrival: predicted, then stripped. So strip it from the *label* first and let the model spend its entire capacity on what survives:

```python
def residualize(rets_t, F_t):
    """rets_t: (N,) forward returns; F_t: (N, K) PIT factor loadings
       (beta, log-size, industry dummies). Returns factor-orthogonal component."""
    beta_hat = np.linalg.lstsq(F_t, rets_t, rcond=None)[0]
    return rets_t - F_t @ beta_hat        # per date, in the label pipeline (Ch. 5)
```

The factor loadings come from a **risk model** you build from data already in the Store: rolling 252-day regression betas against the universe return, log market cap, industry membership events. Crude is fine — the point is alignment with *your* optimizer's constraint directions, not commercial-grade risk decomposition. PIT discipline applies: loadings as-of $t$, from the same event log, firewall-tested.

**Tradability weighting.** Weight the loss by $\sqrt{\text{dollar volume}}$ (the Ch. 5 §5.5 `weight` column) so capacity concentrates where capital can actually go. Square root, not linear: linear weighting collapses the problem to mega-caps, where competition has already eaten the signal.

**⚖ Design call.** Residualized labels change *what question the model answers* — you're forecasting the constrained-tradeable component of returns, not returns. That is the right question for production, but raw-IC comparisons against unneutralized baselines become apples-to-oranges. The ledger therefore tags every run with its **label regime** (`label_regime: "raw" | "residual_v1"`), and champion–challenger duels (Ch. 9) refuse to pair across regimes. Skip this tag and the loop silently corrupts.

## 12.4 Strategy 2 — Constraint-aware losses (soft alignment)

Shape the objective toward the constraint set's *shape* without hard-coding its parameters. Three entries for the `LOSSES` registry:

**Neutralized IC loss.** Compute the Ch. 7 IC loss on factor-projected scores — the differentiable projection is three lines:

```python
def neutral_ic_loss(pred, y, F_pinv, F, w=None):
    """IC loss on the component of pred orthogonal to factor loadings F.
       F_pinv = pinv(F), precomputed per date in the panel (constant in training)."""
    proj = F @ (F_pinv @ pred)
    return ic_loss(pred - proj, y, w)
```

Only the orthogonal component earns reward, so the model *learns* not to spend capacity on factor bets — the loss-side twin of §12.3's label-side residualization. (Use one or the other as your primary; using both is fine and mildly redundant.)

**Turnover-penalized objectives.** The Ch. 7 Sharpe surrogate already carries a linear cost term on weight changes; that is the soft version of a turnover constraint, and it is the *only* mechanism in this book by which a model can learn to prefer stable rankings. Fine-tuning the champion with it is the standard move when the waterfall shows W3→W4 pain.

**Extremes-focused ranking.** A top-k long-short book monetizes the tails of the ordering, nothing else. The Ch. 7 pairwise loss sampled from top/bottom deciles already concentrates gradient exactly there — under a top-k execution reality, it is not just a robustness trick but an *alignment* trick. (Under a full-cross-section optimizer, mid-ranks matter again; match the loss to the book you'll run.)

## 12.5 Strategy 3 — End-to-end differentiable optimization, and its specific disease

The frontier: put portfolio construction *inside* the network — a differentiable QP layer (cvxpylayers/OptNet-style) or the soft long-short of Ch. 7 §7.5 — and backpropagate through it, so gradients reflect exactly what survives the constraints. Perfect alignment in principle. In practice it has a disease with a name:

**Constraint-set overfitting.** The model doesn't overfit the returns; it overfits *the optimizer's parameters*. Your cost model, borrow list, ADV estimates, and turnover penalty are themselves noisy estimates — and a gradient will happily exploit their sample-specific artifacts ("this name is cheap to short in the training window") that do not exist out of sample, let alone in production. You asked for constraints in the architecture *without* overfitting; this is precisely where that trade lives. Defenses, in order:

1. **Constraint randomization** (domain randomization, imported from robotics): per batch, jitter cost estimates (±50%), drop random names from the borrow list, scale liquidity caps. Only signal robust *across* constraint realizations earns reward. This is the single most effective defense and costs five lines.
2. **Curriculum, never cold-start:** pretrain with §12.4 losses, fine-tune end-to-end at low learning rate — the Ch. 7 §7.5 pattern. Training through a QP from scratch is noisy, slow, and finds constraint exploits before it finds signal.
3. **Sim-to-real gap test** (add to `tests/`): evaluate with an optimizer that differs in details from the training-time one — different penalty form, different cost curve, perturbed constraint bounds. **If the end-to-end gain survives only under the exact training configuration, you fit the optimizer, not the market.** This is the Ch. 9 negative-control mentality applied to execution.

**⚖ Design call.** Strategy 3 is *gated, not default*: Loom reaches for it only when the waterfall shows W2→W3 attrition still binding *after* §12.3–12.4 are fully applied. Differentiating through a QP to fix a problem you haven't measured is the second-system effect wearing a research costume. In my experience of the literature and practice, hygiene + soft alignment capture most of the recoverable TC; the end-to-end residual is real but small and expensive. Your waterfall may disagree — that's what it's for.

## 12.6 What this adds to Loom (and what it deliberately doesn't)

One new component, two extensions — all at existing seams (Ch. 4 §4.3 paying rent again):

**⑤½ Portfolio** — between Trainer and Evaluator:

```python
@dataclass(frozen=True)
class ExecSpec:                      # hashed, ledgered, searchable — like everything
    max_adv_frac: float = 0.02       # position ≤ 2% of daily volume
    max_weight: float = 0.01
    beta_bounds: tuple = (-0.05, 0.05)
    sector_bounds: tuple = (-0.03, 0.03)
    gross: float = 2.0; net: float = 0.0
    borrow: str = "universe"         # or a borrow-list view name
    cost_bps: float = 15.0; turnover_penalty: float = 1.0

def optimize(scores, risk: RiskView, spec: ExecSpec) -> Weights:
    """Pure function. Maximize scores·w − turnover_penalty·|Δw|·cost
       s.t. factor bounds, position bounds, gross/net, borrow. One small QP per date."""
```

Pedagogical implementation: `cvxpy`, ~30 lines, one solve per date (daily research scale laughs at the compute). Its differentiable twin for Strategy 3 is the same program through `cvxpylayers` — same `ExecSpec`, two backends, so training-time and judge-time constraints can never drift apart *unless the sim-to-real test wants them to*.

**Evaluator additions:** TC and the five-line waterfall join the standard report and the fingerprint. **Ledger additions:** `label_regime` tag (§12.3) and `exec_hash`. Note what falls out for free: since `ExecSpec` is hashed config like any other, **the Ch. 9 loop can search execution policy too** — and one of the most common findings in practice is that a *slower rebalance* dominates a better model once costs are real. Your improvement loop should be allowed to discover that; now it is.

What we still refuse (Ch. 4 §4.4 discipline): no market-impact models beyond linear cost (fidelity beyond daily data's resolution is decoration), no borrow-fee term structures, no multi-period optimization. Each has a seam waiting if a real problem ever demands it.

## 12.7 Build exercise (4–5 hours)

1. **Risk model + residualization:** rolling betas and sector dummies from your Store; add `residual_v1` as a label regime. Re-run your champion under it. Compare waterfalls — W1→W2 should nearly close; where did W0 go?
2. **The Portfolio component:** implement `optimize()` with cvxpy (beta-neutral, sector-bounded, position-capped, long-short). Unit-test on a 5-stock fixture where you can verify the active constraints by hand.
3. **The waterfall in the Evaluator:** run raw-label vs. residual-label champions through the full report. Ledger both; write the honest note on which stage each one bleeds at.
4. **The sim-to-real test:** train one small model with the Ch. 7 Sharpe surrogate at `cost_bps=15`, evaluate at 10 and 30 with a perturbed optimizer. Does the gain survive? Add `test_sim_to_real.py` to the suite — the sixth way this field fools people, now encoded.
5. **Stretch (Strategy 3):** cvxpylayers fine-tune of the champion, *with* constraint randomization, 5 seeds. Verdict in the ledger — including the likely one: "not worth it at my scale yet." That verdict, reached honestly, is mastery of this chapter.

**Meta-note for the ledger, per your template's Step 5:** this chapter migrated a Q4 item ("signal dies in execution — nobody had accounted for where") into the known: it now has a metric (TC), a decomposition (the waterfall), and a defense ladder (hygiene → soft → end-to-end, gated). That is what "incorporating practical constraints without overfitting" cashes out to.
