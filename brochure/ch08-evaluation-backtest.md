# Chapter 8 — Evaluation and backtesting

*Prerequisites: Ch. 5–7. Builds: the Evaluator (component ⑤) — the judge, and the statistics that make its verdicts mean something. In this field, evaluation IS the hard part; models are cheap, honest verdicts are expensive.*

---

## 8.1 The metric hierarchy

Evaluate in *layers*, cheapest first; each layer can kill a model before the next spends time on it:

```
L1  Signal quality      IC / RankIC / ICIR / decay        seconds     kills most ideas
L2  Monetization shape  quantile spreads, long-short      seconds     kills "clever but flat" signals
L3  Portfolio reality   backtest w/ costs, turnover       minutes     kills fragile & expensive signals
L4  Statistical honesty t-stats, multiplicity, holdout    permanent   kills self-deception
```

### L1 — Signal quality
For each date $t$: $\text{IC}_t = \text{corr}(\hat y_{\cdot t},\, r_{\cdot t})$ (Pearson) or on ranks (Spearman = **RankIC**, the workhorse — robust to outliers). Report the *series*, then:

- **mean IC** — the raw strength (0.02 real, 0.05 excellent for daily equities);
- **ICIR** = mean/std of daily IC — consistency; this predicts live viability better than mean IC;
- **hit rate** — % days IC > 0 (a 54% hit rate compounds beautifully);
- **IC decay** — recompute IC against returns at horizons 1..20; the decay curve tells you your signal's natural holding period and thus its cost budget: fast decay = must trade fast = high turnover = costs eat you unless the signal is strong. *Choose your label horizon to match the decay you observe* — a loop between Ch. 5 and here.

### L2 — Monetization shape
Bucket entities into score deciles each day; track each decile's mean forward return. You want **monotonic** decile returns with the money in the *extremes* (that's where portfolios live). A signal with great IC but flat top decile is a mid-rank signal — statistically real, unmonetizable. The **top-minus-bottom decile spread** is your first PnL-flavored number and the metric ranking losses (Ch. 7) target most directly.

### L3 — Portfolio reality
Scores → positions → costs → PnL:

```python
def backtest(scores, rets, k=50, cost_bps=15, hold=5):
    """Long top-k, short bottom-k, equal weight, `hold`-day tranches, linear costs."""
    ...  # ~40 lines. Yes, really. Write them.
```

**⚖ Design call.** Loom's backtester is deliberately simple: daily bars, linear costs, no market impact, tranche-based holding (1/`hold` of the book rebalances each day — smooths turnover exactly like qlib's `TopkDropoutStrategy`, in a fraction of the machinery; the full event-driven exchange simulator in `qlib/backtest/` is nine files of fidelity that daily research cannot use — fidelity beyond your data's resolution is decoration). Report, always net *and* gross: **Sharpe, max drawdown, annual turnover, cost drag** (gross−net). A signal whose gross Sharpe is 2 and net is 0.3 is a *cost problem* — the fix is slower labels or turnover penalties (Ch. 7's Sharpe loss), not better prediction.

Costs are a *model parameter you must justify*: 5–15 bps per side for liquid US large-caps, multiples of that for small caps and emerging markets. Run L3 at 1× and 2× your cost estimate; a strategy that dies at 2× is a strategy that dies.

### L4 — Statistical honesty
The layer that separates professionals from backtest artists:

- **The IC t-stat, done right.** Daily ICs are autocorrelated → naive $\bar{IC}/\sigma \cdot \sqrt{T}$ overstates. Use **Newey–West** (HAC) standard errors, lag ≈ label horizon. Rule of thumb: you need |t| > 3, not 2 — see next bullet.
- **Multiplicity is your life now.** You will run hundreds of experiments (Ch. 9 makes it easy — that's the point *and* the danger). The best of 200 random signals looks great. Defenses: t>3 threshold; track *number of trials* in the ledger (the ledger makes your trial count auditable — most researchers can't even count theirs); the **deflated Sharpe** idea (Bailey & López de Prado): given N trials, how good must the best look before it's evidence? Even applying it roughly beats not knowing it exists.
- **Era-based reporting.** Metrics per walk-forward fold, per year, per regime — *never only pooled*. A model that made all its money in 2020–2021 is a bet on 2020 happening again (Ch. 1 §1.3c: eras, not rows, are your sample size).
- **The locked holdout** (established in Ch. 5, spent in Ch. 11): touched once, at the very end, by the single pre-registered final model. It's the only number in the whole project that escapes multiplicity — which is why it's the only number an outsider should believe.

## 8.2 Reading a result (worked example)

Ledger entry says: `rank_ic 0.031, icir 0.29, t_NW 4.1, decile_spread 8.9bps/d, turnover 11x/yr… wait, 1100%/yr, gross_sharpe 1.9, net_sharpe(15bps) 1.1, worst_year 2022: ic 0.006`.

The trained read: signal is real (t=4.1 with honest SEs), decently consistent (ICIR .29), monetizable shape (spread ≈ 2.2%/yr per side before costs at this turnover), turnover is high for daily equities but survivable at 15 bps (cost drag 0.8 Sharpe — check the 2× cost run), and **the 2022 number is the finding**: near-zero IC in the one rising-rate year in sample. Next experiment isn't a bigger model — it's a rate-regime feature, or a recency-weighting test, or acceptance that this signal sleeps in some regimes (then size it accordingly). *That* chain of reasoning — from table to next experiment — is the skill this chapter exists to build; Ch. 9 automates its bookkeeping, never its judgment.

## 8.3 The evaluator's contract (recap + one addition)

Dataframes in — `(scores, realized, meta)` — verdict out; importable alone; every metric era-resolved. The addition: the Evaluator also emits a **fingerprint** — the daily IC series itself, stored with the run. Fingerprints let Ch. 9 do the two things aggregate metrics can't: test whether two models are *different* (paired stats on their IC series) and whether an ensemble would help (correlation of their fingerprints — two 0.03-IC models correlated 0.3 are worth far more together than a single 0.04).

## 8.4 Build exercise (3 hours)

1. Implement the Evaluator: IC/RankIC series, ICIR, NW t-stat, decay curve, decile table, and the 40-line backtester. Unit-test the backtester on a hand-computable 3-stock, 4-day fixture — off-by-one turnover bugs are the classic.
2. Run your Ch. 6 model zoo through it. Make the four plots that become your standard report: cumulative IC, decay curve, decile bars, net equity curve. (Plots catch what tables hide: a cumulative-IC line that's flat-then-cliff *is* a regime story.)
3. Multiplicity, felt: generate 200 pure-noise signals, evaluate all, plot the best one's equity curve. It will look fundable. Now compute what t-stat the best-of-200 needs before you should care. Keep that plot where you can see it for the rest of the capstone.

**Next:** the chapter the whole book has been building toward — wiring evaluation, ledger, and search into a loop that improves the system faster than you can by hand, without eating itself.
