# Chapter 16 — Frequency: horizon structure in features, models, and Multi-Period Optimization

*Prerequisites: Ch. 7–8, 12, 14. Builds: the frequency dimension made explicit — features as band-limited signals, the alignment principle, an honest verdict on spectral decomposition, and the opening of the seam Ch. 14 left waiting: Multi-Period Optimization, with its ladder and its gate. Origin: Core Research Areas discussion 5.*

---

## 16.1 Features have frequencies: the horizon response curve

Your observation is exactly right, and it's measurable. Every feature has a **horizon response curve**: its IC computed against forward returns at horizons $h = 1, 2, \dots, H$. This generalizes Ch. 8's decay curve (which asked "how fast does an emitted *signal* rot?") to a design-time question: *at which horizon does this feature's information live?* Empirically the curves are stable in shape and famously spread out: order-flow imbalance peaks at minutes-to-hours; short-term reversal at 1–5 days; momentum at weeks-to-months; value at quarters. A feature is thus characterized by three numbers — peak strength, **peak horizon**, and **bandwidth** (how wide the effective range is).

Make it a standard Evaluator diagnostic: the **feature × horizon IC matrix**, one heatmap per feature set, ledgered like everything else. It immediately exposes the sin most pipelines commit silently: mixing features whose peaks sit at 1 day and 60 days into one model with one 5-day label. The mismatched features aren't useless there — they're *attenuated*, entering as noisy proxies of themselves, and the model spends capacity re-deriving the attenuation. The matrix tells you, before any training, which features belong to which prediction task.

## 16.2 The alignment principle (the chapter's spine)

A prediction pipeline is **frequency-aligned** when five quantities agree:

$$\underbrace{\text{feature peak horizon}}_{\S16.1} \;\approx\; \underbrace{\text{model memory}}_{\text{receptive field } T} \;\approx\; \underbrace{\text{label horizon}}_{h} \;\approx\; \underbrace{\text{holding period}}_{\text{turnover}} \;\approx\; \underbrace{\text{cost-viable frequency}}_{\text{Ch. 8/12}}$$

Each pairwise mismatch is a familiar, named waste: momentum features on a 1-day label (attenuation, §16.1); a 60-day GRU receptive field on a 1-day label from fast features (wasted memory — the net must learn to ignore 55 days); a 1-day signal traded weekly (decayed before monetized, Ch. 14 Rung 0); a daily-turnover strategy on a signal whose cost-viable frequency is monthly (the Ch. 12 waterfall's W3→W4 bleeding). Alignment isn't an optimization — it's the *removal of self-inflicted attenuation*, and it's free.

The constructive version: **multi-horizon multi-task heads**. One shared encoder (which may legitimately be long-memory now), K small heads predicting returns at $h \in \{1, 5, 20\}$ days — Ch. 7's multi-task pattern, one `ModelSpec` change. Each head naturally learns to weight features by their horizon response (the matrix from §16.1 gets learned implicitly), the auxiliary horizons regularize each other, and — the payoff that makes this chapter cohere — the K heads emit exactly the object §16.4 needs: **a term structure of expected returns** per stock. Sampled per-head into the signal log (Ch. 14), each horizon is also independently fingerprinted, monitored, and duel-able.

## 16.3 The spectral temptation: what's real, what's a trap

"Price as a superposition of overlapping cycles" — handle this framing with gloves, because it contains one deep truth wrapped in the field's oldest trap.

**The trap first, named:** the **Slutsky–Yule effect** (1927): applying moving-average filters to *pure white noise* manufactures smooth, compelling, apparent cycles. Every smoothing window is a band-pass filter; band-passed noise oscillates at the band's frequency *by construction*. So a wavelet/Fourier decomposition of returns will always hand you beautiful cycle components — whether or not any exist — and extrapolating a fitted sinusoid is the canonical way to trade pure noise with total confidence. The spectra of actual asset returns are, after a century of looking, almost flat: weak power at reversal and momentum bands, nothing that survives out-of-sample as a literal oscillator. Any spectral method you adopt must therefore pass the Ch. 2 shuffled-future canary *and* a **Slutsky control**: run it on matched white noise; if it "finds cycles" there too, it found your filter, not the market.

**What's real, in descending order of evidence:**

1. **Calendar and event periodicities.** Intraday U-shaped volume/volatility, day-of-week and turn-of-month effects, options-expiry cycles, the earnings cycle. These are genuine periodicities — but they're anchored to *institutional clocks*, not sinusoids, so the right representation is **features and events** (time-of-day, days-to-expiry, days-since-earnings — Ch. 10's recency verbs), not Fourier modes. Events beat Fourier for market cycles because market cycles have calendars, not frequencies.
2. **Multi-resolution feature engineering.** Returns, volatility, and flow aggregated over multiple trailing windows (1, 5, 20, 60 days) form a crude but honest dyadic wavelet basis — *causal by construction*, which is the non-negotiable constraint: symmetric/centered wavelet transforms peek at the future, and the firewall test will catch the standard `pywt` misuse. Your feature set probably already is a multi-scale decomposition; §16.1's matrix tells you which scales carry signal.
3. **Multi-scale inductive bias in the model.** TCN dilations (Ch. 6) *are* dyadic frequency decomposition in the weights; multi-branch encoders reading the same series at several downsamplings are the explicit version. Frequency-domain deep architectures (FEDformer-style) earn their keep on data with true strong seasonality — load, traffic, weather; on the near-flat spectrum of financial returns, expect them to underperform a well-sized TCN. Ledger question, low prior.

**Decomposition done right** is then modest and useful: split returns into slow and fast components with *causal* filters (e.g., trailing EWMA trend + residual), predict each component with its horizon-matched features and labels, recombine. Notice what this is: §16.2's multi-task alignment, expressed in the frequency domain. The two views are the same design — which is the reassurance that you're aligning with structure, not chasing sinusoids.

## 16.4 Multi-Period Optimization: why tomorrow's alpha changes today's trade

Ch. 12 optimized each rebalance in isolation. The moment transaction costs exist, that's wrong in a specific, quantifiable way: **the optimal trade today depends on expected alphas tomorrow**, because positions are durable assets with an entry fee. Two consequences you can feel intuitively: don't build a full position in a fast-decaying signal (you'll pay to unwind it before it pays you), and *do* pre-position toward where slow signals will want you next week (amortize the entry cost over a long holding). Single-period optimization can see neither.

**The closed-form intuition — Gârleanu–Pedersen (2013).** With quadratic costs and exponentially-decaying alphas, the optimum has a two-line structure worth memorizing:

$$x_t = x_{t-1} + \tau\,(\text{aim}_t - x_{t-1}), \qquad \text{aim}_t = \sum_h w_h \,\text{Markowitz}_h, \quad w_h \uparrow \text{ for slow-decaying alphas}$$

Trade *partially* toward an **aim portfolio** (trading rate $\tau$ set by cost vs. urgency), where the aim is a weighted blend of current and future Markowitz portfolios that **overweights slow alphas relative to their instantaneous strength** — "aim in front of the target," like a hunter leading a moving duck. Fast signals get systematically down-weighted in *positions* even when strong in *forecasts*: the frequency structure of your alphas, priced into the trade.

**The computational version — Boyd et al. (2017), convex MPO.** Plan the whole position trajectory: choose $x_{t+1}, \dots, x_{t+H}$ maximizing $\sum_h \big[\alpha_h^\top x_h - \text{costs}(x_h - x_{h-1}) - \text{risk}(x_h)\big]$ subject to Ch. 12's constraints at every step. A block-QP, H× the variables of single-period, cvxpy-native — and here §12.8's discipline pays a second time: the DPP-parametrized `optimize()` extends to `MPOSpec(H, ...)` as the same program stacked H deep, judge-time and any future differentiable use provably consistent. Crucially, MPO is executed as **model predictive control**: solve for the trajectory, execute *only the first step*, re-solve next period with fresh forecasts. The plan is disposable; the planning is what prices the future.

**What MPO demands from the prediction stack — and we already built it.** MPO's input is the alpha term structure $\alpha_{t+1}, \dots, \alpha_{t+H}$ per asset. Two suppliers, both in hand: cheaply, *each signal's current score × its measured decay curve* (Ch. 8 fingerprints — the decay curve stops being a diagnostic and becomes an input); richly, the multi-horizon heads of §16.2. This is the moment three chapters click together: Ch. 8 measured decay, Ch. 14 aligned clocks and blended signals, and MPO consumes both to time the trades. Ch. 14's Rung 4 ("slow core, fast satellites, net at execution") now reveals itself as the *heuristic limit* of MPO — the closed-form GP solution *is* that heuristic, made exact.

## 16.5 The MPO ladder, and its gate

Ascending machinery, ledger-gated as always:

- **Rung 0 — horizon-matched single-period.** Pick the label horizon and rebalance frequency jointly from decay + cost curves (Ch. 8). This *is* degenerate MPO and captures the first-order effect. Most books stop here profitably.
- **Rung 1 — the GP aim portfolio.** Needs only per-signal decay estimates and a quadratic cost guess; ~30 lines, no solver in the loop. Delivers the two qualitative behaviors that matter (partial trading, decay-weighted aims). **Best value-per-line on the ladder.**
- **Rung 2 — convex MPO (Boyd).** Full constraint set at each of H steps, real term structure from multi-horizon heads. Reach for it when Rung 1's *assumptions* visibly bind: strongly non-quadratic costs, hard constraints (borrow, ADV) that the aim-portfolio blend can't see, or predictable *flows* (index rebalances, expiry) worth planning around.
- **Refused (Ch. 4 §4.4 discipline):** stochastic dynamic programming over forecast uncertainty. MPC re-solving plus the robustness devices below capture most of its value at a fraction of the machinery.

**The gate and its honesty tests.** Enter the ladder above Rung 0 only when the Ch. 12 waterfall shows W3→W4 (cost) attrition binding — MPO is a *cost amortization* technology; without material costs it optimizes nothing. And two frequency-specific negative controls join the suite: **(a) perturb-the-decay test** — MPO's edge over single-period must survive ±30% error in decay-curve estimates, since term structures are estimated on the date-axis budget (Ch. 13 §13.2 — far-horizon alphas are the noisiest; shrink them toward zero as $h$ grows, and note the GP weights do this implicitly); **(b) the Slutsky control** from §16.3 for any spectral component anywhere upstream. Horizon planning compounds forecast errors; the controls make sure it compounds signal instead.

**⚖ Design call.** Loom integrates frequency as *measurements and one optimizer upgrade*, not as a new subsystem: the feature×horizon matrix and decay curves in the Evaluator (measurement), multi-horizon heads as a `ModelSpec` option (supply), `MPOSpec` extending the Ch. 12 Portfolio component (demand), and the ladder gated by the waterfall (discipline). The frequency "dimension" thus lives in the ledger — where, as with every Part V question, it becomes something you *read* rather than believe.

## 16.6 Build exercise (5 hours)

1. **The matrix.** Compute the feature × horizon IC matrix (h ∈ {1,2,5,10,20,60}) for your Ch. 5 feature set. Find one badly misaligned feature in your current champion (there will be one) and re-route it. Cheapest IC improvement in this book.
2. **Multi-horizon heads.** Add 3 heads (1/5/20d) to your champion encoder. Verify the implicit routing: correlate each head's learned feature attributions (or a simple ablation) with the matrix from (1).
3. **Slutsky, felt.** Band-pass filter pure white noise with a 20-day moving average; plot the "cycles"; fit and extrapolate a sinusoid; backtest it on the noise. Frame the equity curve next to Ch. 8's best-of-200 plot. Two exhibits in your museum of self-deception.
4. **The GP aim portfolio.** Implement Rung 1 with two synthetic signals (half-lives 2d and 40d, equal instantaneous IC). Verify both signature behaviors: positions load ~2–3× heavier on the slow signal, and net-of-cost Sharpe beats the single-period optimizer that treats them equally. Then the perturb-the-decay test.
5. **Stretch — Rung 2.** Stack your Ch. 12 cvxpy program H=5 deep (the DPP discipline makes this mechanical), run as MPC on the same synthetic pair, and ledger the Rung 2 − Rung 1 margin. Expect it to be small — and expect to be glad you measured rather than assumed.

**Ledger meta-note (template Step 5):** discussion 5 converted "frequency" from a mystique (cycles, spectra) into three artifacts — a matrix, a term structure, and a trading-rate — connected by one principle (alignment) and guarded by one new negative control (Slutsky). The seam Ch. 14 refused is now open, with its gate installed before its machinery. That ordering — gate before machinery — is the discipline this brochure has been teaching all along.
