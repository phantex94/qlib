# Chapter 1 — What return prediction actually is

*Prerequisites: intro probability. Builds: your intuition for why everything downstream is designed the way it is.*

---

## 1.1 The object of study

Fix an asset universe $\{1,\dots,N\}$ and a horizon $h$. The **forward return** of asset $i$ at time $t$ is

$$r_{i,t\to t+h} = \frac{P_{i,t+h}}{P_{i,t}} - 1$$

Return prediction is the estimation of

$$\hat{y}_{i,t} \approx \mathbb{E}\left[\, r_{i,t\to t+h} \mid \mathcal{F}_t \,\right]$$

where $\mathcal{F}_t$ is *everything knowable at time $t$*. That conditioning bar is the single most important symbol in this brochure. Almost every catastrophic research failure is a violation of it — using information from the future of $t$ and calling it $\mathcal{F}_t$. Chapter 2 builds machinery to enforce it; for now, tattoo it somewhere.

Two framings of the problem exist, and they are more different than they look:

| | **Time-series** framing | **Cross-sectional** framing |
|---|---|---|
| Question | "Will *this* asset go up?" | "Which assets will beat which *today*?" |
| Prediction target | $r_{i,t\to t+h}$, absolute | $r_{i,t\to t+h}$ *relative to peers at $t$* |
| Enemy | Non-stationarity in time | Non-stationarity in time (still), but market-wide shocks cancel |
| Typical user | CTA / macro / market-maker | Equity stat-arb (qlib's home turf) |

The cross-sectional framing is where most ML-for-equities lives, for one deep reason: **when you rank assets against each other at the same instant, the market's common component drops out.** You don't need to know if tomorrow is a crash; you need to know that A falls less than B. That's a much smaller ask of a model, and small asks are all this domain can afford — see §1.3.

**⚖ Design call.** Loom treats the *cross-sectional panel* as the default learning problem and absolute time-series forecasting as a special case (a universe of size 1). qlib does roughly the same, implicitly. We'll make it explicit.

## 1.2 The fundamental law: why being barely right is enough

Grinold's **fundamental law of active management** is the economic first principle of this whole field:

$$\text{IR} \approx \text{IC} \cdot \sqrt{\text{breadth}}$$

- **IC (information coefficient):** the correlation between your predictions $\hat{y}_{i,t}$ and realized returns $r_{i,t\to t+h}$, usually computed cross-sectionally each day and then averaged. It measures *how right you are*.
- **Breadth:** the number of *independent* bets per year. A daily-rebalanced strategy over 3,000 stocks has enormous breadth (though far less than $3000 \times 252$ — bets are correlated).
- **IR (information ratio):** risk-adjusted excess return. It measures *how good the business is*.

Plug in numbers. An IC of **0.03** — a correlation so weak you could never see it on a scatter plot — across a few hundred effectively-independent daily bets gives an IR above 2. That is a phenomenal business. Meanwhile an IC of 0.15 on twelve bets a year is a hobby.

Three consequences that shape everything we build:

1. **The signal is tiny.** Daily cross-sectional $R^2$ of 0.5–1% is *excellent*. Your loss curves will look like failure by ImageNet standards. Calibrate your emotions now.
2. **Breadth is an engineering problem.** More assets, more (independent) rebalances, more horizons — the system that makes it *cheap to add breadth* wins. This is a systems argument, not a modeling argument, and it's why Chapter 4 exists.
3. **Evaluation must respect the law.** We will judge models by IC-family metrics and portfolio outcomes (Ch. 8), not MSE — and eventually *train* on IC-family objectives (Ch. 7).

## 1.3 Why finance breaks your ML intuitions

If you learned ML on vision or NLP, four assumptions you didn't know you were making are false here:

**(a) Low signal-to-noise, irreducibly.** In vision, the label is a deterministic function of the input; error is model inadequacy. Here, $r = \text{signal} + \text{noise}$ with noise variance ~100× signal variance, *and it must stay that way*: if returns were predictable at high $R^2$, someone would trade the predictability away. Markets are adversarial noise generators pointed at your loss function. Implication: **capacity is dangerous.** A model that can memorize will memorize noise with enthusiasm. Regularization is not a tuning detail; it's the main event.

**(b) Non-stationarity is the rule.** $P(y|x)$ drifts: factors crowd, regimes change, the 2007 quant quake happened because everyone's models agreed. There is no fixed distribution to converge to. Implication: walk-forward everything (Ch. 5), monitor drift (Ch. 9), prefer models that retrain cheaply over models that are 2% better but take a week to fit.

**(c) The effective sample size is small.** You may have 10M rows (3,000 stocks × 15 years × 252 days), but rows within a day are correlated (they share the market), and days within a regime are correlated. The number of *independent* observations is closer to "number of distinct market regimes you've seen" — dozens, not millions. Implication: cross-validation row counts lie to you; block/era-based statistics (Ch. 8) don't.

**(d) The metric is not the loss.** You'll minimize a differentiable surrogate, but the judge is a portfolio's net-of-cost performance, which involves ranking, position sizing, turnover penalties, and risk constraints. The gap between surrogate and judge is where Chapter 7 lives.

## 1.4 Labels: more design freedom than you think

The naive label is raw forward return. You will almost never use it raw:

- **Horizon $h$:** 1–20 days is typical for daily equity ML. Shorter = more breadth, more turnover cost sensitivity. Longer = fatter signal per bet, fewer bets. (Ch. 8 shows how to measure your signal's natural decay and choose $h$ to match.)
- **Excess vs. raw:** subtract the universe mean (or beta-adjust) so the label is *relative* performance — this is what makes the problem cross-sectional.
- **Normalization:** per-day z-scoring or rank-transforming the label tames fat tails and makes days comparable. qlib's default alpha label is essentially `Ref($close,-2)/Ref($close,-1)-1` — note the `-2`: it skips one day to avoid impossible same-close execution. Label timing *is* execution modeling. We make this explicit in Loom: **a label is a declared contract: (formula, horizon, execution lag, normalization)** — a first-class object, not a string.
- **Beyond returns:** you can predict volatility (useful for sizing), or multi-task both (Ch. 7).

## 1.5 The map of the territory

Everything in this brochure is one box of this diagram, and the diagram *is* the answer to "how does data flow":

```
 raw world ──► [ Store: point-in-time event log ]        Ch. 2
                       │  as-of queries
                       ▼
               [ Features + Labels → Panel ]             Ch. 5
                       │  purged walk-forward splits
                       ▼
               [ Model: fit / predict ]                  Ch. 6
                       │  scores        ▲ gradients from
                       ▼                │ losses that match the judge   Ch. 7
               [ Evaluator + Backtester ]                Ch. 8
                       │  metrics, PnL
                       ▼
               [ Ledger ] ──► [ Search / Retrain / Monitor ]  Ch. 9
                       ▲                │
                       └── the self-improvement loop ────┘
```

qlib implements this diagram too — tangled (Ch. 3). Loom implements it with seven small components and visible seams (Ch. 4).

## 1.6 Build exercise (do it — 1 hour)

Simulate the fundamental law to *feel* the low-SNR regime:

```python
import numpy as np
rng = np.random.default_rng(0)
N, T, ic_true = 1000, 2520, 0.03          # 1000 stocks, 10 years
signal = rng.standard_normal((T, N))
noise  = rng.standard_normal((T, N))
rets   = ic_true * signal + np.sqrt(1 - ic_true**2) * noise   # daily xsec returns (unit var)

daily_ic = np.array([np.corrcoef(signal[t], rets[t])[0, 1] for t in range(T)])
pnl = (signal * rets).mean(axis=1)         # naive "hold the signal" portfolio
print("mean IC", daily_ic.mean(), "IC t-stat", daily_ic.mean()/daily_ic.std()*np.sqrt(T))
print("Sharpe", pnl.mean()/pnl.std()*np.sqrt(252))
```

1. Confirm the Sharpe is embarrassingly good for a 0.03 correlation. 2. Now set `N=20` and watch the business evaporate: that's breadth. 3. Plot a scatter of `signal[0]` vs `rets[0]` and appreciate that *this invisible relationship is the entire industry*. 4. Estimate: how many days of live trading would you need before the daily-IC t-stat distinguishes IC=0.03 from IC=0? (This number is why Chapter 9's monitoring design is hard.)

**Next:** none of this matters if $\mathcal{F}_t$ is contaminated. Chapter 2 is about the data model that makes contamination structurally difficult.
