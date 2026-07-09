# Chapter 14 — Market time: alternative bars and the combination layer

*Prerequisites: Ch. 2, 8, 10, 13. Builds: the sampling clock as a config parameter (volume/dollar bars, event anchors — at ~50 lines of new code), and the signal-combination layer that integrates predictions across clocks, horizons, and objectives. Origin: Core Research Areas discussion 3.*

---

## 14.1 The clock is a modeling choice you've been making implicitly

Fixed-interval time bars encode an assumption so old it's invisible: that information arrives uniformly in wall-clock time. It doesn't. Markets run on *activity* — trading volume, order flow, news — which is fiercely clustered: a quiet Tuesday afternoon bar and the bar spanning a Fed announcement are "equal" samples only to the calendar. Time bars therefore **oversample the quiet and undersample the busy**, which is exactly backwards for a learner.

The classical fix (Clark's 1973 mixture-of-distributions insight, revived by López de Prado) is to sample in **market time**: emit a bar whenever a fixed amount of *activity* has occurred.

- **Tick bars** — every N trades.
- **Volume bars** — every V shares. Better, but a share of a $5 stock ≠ a share of a $500 stock, and splits break V.
- **Dollar bars** — every $D of value traded. The robust default: invariant to price level, splits, and (mostly) secular volume growth.
- **Information-driven bars** (imbalance/run bars) — emit when order-flow imbalance exceeds its trailing expectation; the most adaptive, and the most machinery.

The payoff is statistical, and it's real: returns sampled in market time are much closer to IID-Gaussian — volatility clustering gets absorbed *into the clock* (busy periods simply produce more bars), so everything downstream that quietly prefers homoskedastic inputs (your losses, your normalizations, your risk estimates) works better. And **event-driven forecasting** (Ch. 10 §10.2) is the limiting case of the same idea: sample exactly when something happened, with the "something" defined by an event kind rather than a cumulative threshold.

Unifying claim: *time bars, volume bars, dollar bars, and event anchors are not different data types. They are different **clocks** — policies for choosing the sample times $\{t_k\}$ per entity.* Once you say it that way, the adaptation cost question answers itself.

## 14.2 Cost-effective adaptation: bars are folds over the event log

Here is where the Ch. 2 bet pays its third dividend (after PIT and event streams). In a grid-native system, the calendar is the *foundation* — qlib's `CalendarProvider` is the first thing its data layer defines — so "support dollar bars" means re-plumbing the entire data path. In Loom, a bar is a **materialized view**, and a bar builder is a ~15-line fold over the trade event stream:

```python
def dollar_bars(trades, threshold):          # trades: iterator of trade events
    cum, o, h, l, v, t0 = 0.0, None, -inf, inf, 0.0, None
    for tr in trades:
        px, sz = tr.payload["px"], tr.payload["sz"]
        o = px if o is None else o; t0 = tr.t if t0 is None else t0
        h, l, v, cum = max(h, px), min(l, px), v + sz, cum + px * sz
        if cum >= threshold:
            yield Event(tr.entity, "bar.dollar", t_effective=t0, t_observed=tr.t,
                        payload={"open": o, "high": h, "low": l, "close": px,
                                 "volume": v, "dollar": cum})
            cum, o, h, l, v, t0 = 0.0, None, -inf, inf, 0.0, None
```

Emitted bars are appended to the log as a new event kind — `t_observed` is the closing trade's time, honestly. Then one abstraction makes the whole system clock-generic:

```python
class Clock(Protocol):
    def times(self, store, entity, span) -> list[Timestamp]: ...

TimeClock("1d")                      # the old default, demoted to one option
BarClock("bar.dollar")               # sample at each dollar-bar close
EventClock(kind="earnings", lag="1d")  # Ch. 10's sample_at, now first-class
```

`PanelBuilder` takes `clock=` as a config field; features already work (they're as-of functions — they never cared where $t$ came from); labels become "forward return over the next $K$ *clock ticks*." Total new code: bar builders + the `Clock` protocol + threshold calibration — ~50–80 lines. **That is the cost-effective answer: don't adapt the models to the bars; make the clock a parameter and let the existing machinery not care.** The Ch. 9 loop can then *search over clocks* like anything else — which is the correct epistemic status for "which bars are best?": an empirical, per-signal question.

Three sharp edges, named:

- **Endogenous-sampling leak.** Bar boundaries are computed *from the data*. Calibrating the threshold (e.g., "target ~50 bars/day") on the full sample leaks future activity levels into past bar shapes. Thresholds must be calibrated on trailing windows only — and this goes double for imbalance bars, whose "expectation" term is a fitted quantity. Add it to the firewall suite: rebuild bars from a truncated log; boundaries before the truncation point must be bit-identical (the Ch. 2 test, aimed at a new target).
- **Labels change meaning.** A "5-bar forward return" in dollar time is a *variable wall-clock horizon* — short in crises, long in calm. That's usually what you want (constant-information horizons) but it changes what the model predicts and how you evaluate: convert to wall-clock for portfolio simulation, or your backtest quietly assumes you can always trade at bar closes that arrive minutes apart in a crash.
- **The cross-section breaks.** Different entities' dollar bars close at different instants — there is no shared "date" axis, so the cross-sectional machinery (IC-per-day, `DateBatchSampler`, Ch. 13's mixer) has no natural grid. Three honest responses: **(a)** work per-asset in pure bar time (the Ch. 13 degenerate TS mode — fine for futures/crypto books); **(b)** asynchronous cross-sections: at each decision time, take every entity's *latest closed* bars via as-of, carrying staleness Δt as a feature — which is exactly Ch. 10 Strategy B's machinery reused; **(c)** whatever the model's clock, **evaluate on the decision clock** (see below) so metrics stay comparable across all signals.

## 14.3 Your deeper question: predictions are events too

Now the integration problem. After this chapter you'll have models on different clocks with different horizons and objectives: a daily cross-sectional ranker, a dollar-bar momentum model on liquid names, an earnings-drift event model, maybe a volatility forecaster (Ch. 7's multi-task head). How do these become *one* decision?

The Loom answer is almost embarrassing once seen: **a prediction is an event.** Emit every model's scores back into the event log:

```
Event(entity, kind="signal.<model_id>", t_effective=t_emit, t_observed=t_emit,
      payload={"score": s, "horizon": "5bar~2d", "objective": "xsec_return"})
```

Consequences, each load-bearing:

- **Integration becomes feature engineering on signal events** — a problem you already have all the tools for. The combiner is just another model whose features are as-of samples of signal streams (with staleness Δt), built by the same `PanelBuilder` on the **decision clock** (e.g., daily at rebalance time), trained with the same walk-forward, judged by the same Evaluator, ledgered like everything else. No new subsystem.
- **Bitemporality protects you again:** the combiner can only see scores *as they were emitted* — no accidental use of revised or recomputed signals. Backtesting the combiner over history requires replaying signal emission honestly (walk-forward generation — see the trap in §14.4).
- **Signals become auditable streams** with their own fingerprints (Ch. 8), decay curves, and live monitors (Ch. 9 §9.5) — per stream, so a dying signal is visible *before* it drags the blend down.

## 14.4 The combination ladder

Climb only as high as your data supports — the meta-learner's capacity budget is the *date axis* budget from Ch. 13 §13.2 (a combiner learns across decision times: a few thousand samples, heavily regime-correlated). In ascending order:

**Rung 0 — Align and decay.** Bring every stream to the decision clock by as-of sampling, then discount stale scores by their *own measured decay curve* (Ch. 8's IC-decay, finally load-bearing): a signal with 2-day IC half-life, emitted 6 hours ago, enters at ~0.9 weight; yesterday's, at ~0.7. Z-score each stream cross-sectionally. This rung is mandatory hygiene for everything above it.

**Rung 1 — Equal weight.** Mean of aligned z-scores. Do not laugh: equal-weighting is brutally hard to beat out-of-sample (the same shrinkage logic as 1/N portfolios — combination weights are estimated with ~date-axis samples, and estimation error eats the optimality gap). This is the **champion the combiner must duel**.

**Rung 2 — IC-covariance weighting.** Treat streams as assets: each stream's "return" per period is its realized IC (from fingerprints). Mean-variance weight them — $w \propto \Sigma^{-1}\,\overline{IC}$ — with heavy shrinkage of $\Sigma$ toward diagonal and trailing-window estimation. This is Grinold's "forecast of forecasts" in modern clothes, and the fingerprint correlation matrix gives you the answer to "is a new stream worth building?" *before* you build its combiner: a candidate correlated 0.9 with an existing stream adds nothing at any weight (Ch. 8 §8.3 promised this payoff).

**Rung 3 — Stacked meta-model.** A small learner (ridge, shallow GBDT) on features = {aligned scores, staleness, per-stream trailing IC, regime context like realized vol}. This can learn *conditional* blending — "trust the event model only in earnings weeks; fade the fast signal when spreads blow out" — which is where objectives integrate too: a volatility forecast enters not as an additive term but as an **interaction/sizing feature** (score ÷ predicted vol is Sharpe-flavored sizing; an event-drift probability *gates* rather than adds). **The trap that kills most stackers:** meta-training data must be the base models' *walk-forward out-of-sample* emissions, never in-sample refits — otherwise the meta-learner learns each base model's overfitting signature and inverts it, beautifully, until live. Predictions-as-events enforces this mechanically *if* you only ever emit from honest walk-forward runs; the ledger's `run_id` on each signal event makes it auditable.

**Rung 4 — Reconcile at the portfolio, not the forecast.** Different horizons are not competing answers to one question — they are forecasts of *different integrals of the same return path*, and can all be right simultaneously. So the deepest integration is positional: slow signals set core positions; fast signals trade *around* them; the Ch. 12 optimizer nets the desired trades before execution. Netting is free money — the fast model's sell crossing the slow model's hold saves two spreads — and it's a Ch. 12 `ExecSpec` concern, not a forecasting one. The cheap, robust version: per-family sleeves sized by trailing ICIR and capacity, netted at execution. Full multi-period optimization is another Ch. 4 §4.4 refusal — a seam left waiting.

**⚖ Design call — merge vs. federate, the principle.** Ch. 13 *merged* TS and XS into one architecture; this chapter *federates* clocks into separate models plus a combiner. The difference isn't taste: **merge when the factorizations share a sample** (one tensor, one date, one loss — composition inside the model is cheap and jointly trainable); **federate when the clocks don't align** (samples don't even co-occur; a monster multi-clock model re-couples everything, so one data bug or one regime break degrades the whole, and the Ch. 9 loop loses its per-stream iteration speed). Signal space — cheap, typed, bitemporal — is the federation protocol, and each stream keeps its own champion, fingerprint, and monitor.

## 14.5 Build exercise (4–5 hours)

1. Implement `dollar_bars()` and `BarClock`; materialize dollar bars for your crypto/equity trades (crypto is ideal here — you have real trade events from Ch. 2). Verify the Clark effect: kurtosis of dollar-bar returns vs. time-bar returns of matched average frequency. Seeing fat tails shrink because you *changed the clock* is one of the field's genuinely beautiful moments.
2. Add the endogenous-sampling firewall test (truncated-log bar-boundary invariance).
3. Train your Ch. 6 GRU on the dollar-bar clock (per-asset, bar-time labels) and emit its scores as `signal.gru_dollar` events from a walk-forward run. Also emit your daily champion's scores.
4. Build the Rung 0–1 combiner on the daily decision clock: as-of align, decay by measured half-life, z-score, equal weight. Duel it against each stream alone. Then Rung 2 with shrunk IC-covariance. Ledger the ladder — and expect Rung 1 to be uncomfortably hard to beat; write down *why* in your own words (it's the date-axis sample budget again).
5. Stretch: one conditional feature at Rung 3 (trailing 20-day realized vol as an interaction), with meta-training strictly on walk-forward emissions. Verify the trap: refit the base models in-sample, restack, and watch the fake improvement appear — your negative control for stacking, forever.

**Ledger meta-note (template Step 5):** discussion 3 turned "which bars?" into a searchable config axis (the Clock) and "how to integrate?" into a data-engineering pattern (predictions-as-events) plus a capacity-budgeted ladder. Both questions stopped being architecture and became experiments — which is, by now, recognizably the Loom answer to everything.
