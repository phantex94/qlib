# The Loom Blueprint, as Open Questions

*A working document, not a chapter. It summarizes the seven Core Research Areas discussions (Ch. 12–18) plus the core design (Ch. 1–11) into the form that actually drives work: **open questions**, each with an ID, the principle that frames it, and the criterion that closes it. Per the project template's Step 5, this is the epistemic matrix in its end state: what we settled, what we deliberately left to experiments, and what remains genuinely undecided.*

**How to read the tiers.** Every discussion ended the same way — a "which architecture?" question dissolved into (a) a principle, (b) a small mechanism at a seam, and (c) an experiment. So the open questions come in three kinds:

- **Tier A — open by design.** Empirical questions the system was *built* to answer cheaply. The ledger closes them; each lists its closing experiment. These are not failures of the blueprint — they are its products.
- **Tier B — open at the design level.** Genuine architecture/methodology uncertainties, each parked behind a named gate or revisit-trigger. Closing one may change the blueprint.
- **Tier C — open to the human.** Only you can close these.

---

## Tier A — Open by design (the ledger's job list)

### From Discussion 1 — Execution alignment (Ch. 12)
*Settled: the objective is TC·IC; attrition is measured by the waterfall; constraints enter at three depths (hygiene → soft losses → end-to-end), gated.*

- **A1.1 — Where does our signal die?** Per strategy: which waterfall stage (W0→W4) bleeds most? *Closes when:* first full waterfall report on the champion. Everything in Ch. 12 branches on this number.
- **A1.2 — Does residualized labeling beat raw + neutralized scores?** Same information, label-side vs. score-side. *Closes:* paired duel, same `label_regime` tagging discipline.
- **A1.3 — Does end-to-end (cvxpylayers) ever pay after hygiene + soft alignment?** Includes the cvxpylayers verdict of §12.8 (the "minor topic" addendum). *Closes:* Strategy-3 fine-tune vs. Strategy-1+2 champion, with constraint randomization, only if A1.1 shows W2→W3 binding.

### From Discussion 2 — Time-series × cross-section (Ch. 13)
*Settled: two factorizations of one tensor; the asymmetry principle (big shared temporal encoder, small constrained mixer); gated composition; view-level separation.*

- **A2.1 — How much cross-sectional structure does each universe pay for?** The learned gate magnitude *is* the measurement. *Closes:* three-way duel (TS-only / XS-only / joint) per universe; re-run when the universe changes.
- **A2.2 — Graph prior vs. free attention: where's the crossover?** Prediction: priors win at short history, converge as data grows. *Closes:* masked vs. unmasked mixer duel at two history lengths.
- **A2.3 — Is captured "lead–lag" real?** *Closes:* planted-effect test (synthetic lead–lag found by attention) plus entity-permutation equivariance test, before any live claim.

### From Discussion 3 — Clocks and combination (Ch. 14)
*Settled: the sampling clock is config (bars = folds over the event log); predictions are events; combination is a capacity-budgeted ladder; merge when samples align, federate when clocks don't.*

- **A3.1 — Which clock per signal family?** Dollar bars must beat time bars *net of the broken cross-section*. *Closes:* same model, two clocks, paired duel; kurtosis reduction (the Clark effect) logged as supporting evidence, not proof.
- **A3.2 — How high up the combination ladder does our date-axis budget reach?** Equal weight is the null. *Closes:* Rungs 1→2→3 duels; the ladder stops at the first rung that fails.
- **A3.3 — What is the right decision-clock frequency?** Joint function of decay curves and cost (links to A6.1). *Closes:* backtest grid over rebalance frequencies at measured decay, 1× and 2× costs.

### From Discussion 4 — Inductive bias and MoE (Ch. 15)
*Settled: diversity is money (ambiguity decomposition), structural > seed diversity; the MoE transplant breaks in three places; federated default, entity-axis routing if neural, gates must duel equal weight.*

- **A4.1 — Does any learned gate beat the equal-weight null out-of-sample?** The central mixture question. *Closes:* Design-A combiner and Design-B neural MoE, each vs. Rung-1, paired.
- **A4.2 — How much does structural diversity add over a seed ensemble?** Funds (or defunds) multi-family maintenance. *Closes:* 3-family blend vs. 3-seed GRU ensemble, fingerprint correlations reported.
- **A4.3 — Does the noise expert starve?** Standing health check, not one-shot. *Closes:* never — it's control panel item 5, re-run per gate change.

### From Discussion 5 — Frequency and MPO (Ch. 16)
*Settled: features have horizon response curves; the five-way alignment principle; spectra guarded by the Slutsky control; MPO ladder (horizon-matching → GP aim portfolio → convex MPC) gated on cost attrition.*

- **A6.1 — What does the feature × horizon IC matrix say?** The cheapest unclaimed IC in the system (re-routing misaligned features). *Closes:* the matrix as standing Evaluator artifact; one re-routing experiment.
- **A6.2 — Term structure supplier: multi-horizon heads or score × decay-curve?** Rich vs. cheap. *Closes:* MPO Rung 1 fed by each, paired duel.
- **A6.3 — What is the Rung-2 − Rung-1 margin?** Expected small; must survive the perturb-the-decay test (±30%). *Closes:* synthetic two-signal benchmark, then live panel.
- **A6.4 — Does any spectral feature survive the Slutsky control?** Prior: calendar/event features yes, extrapolated cycles no. *Closes:* per candidate, automatically.

### From Discussion 6 — Overfitting (Ch. 17)
*Settled: three levels (weight/selection/regime) with danger ordering S > R > W; the shifted-surface case for flatness; SWA adopt, ASAM measure, natural gradient deprioritize; the eight-item control panel.*

- **A7.1 — Does flatness predict OOS RankIC on *our* ledger?** The meta-experiment; licenses (or blocks) ASAM in the champion pipeline. *Closes:* regression of OOS RankIC on perturbation-radius score across ledgered runs, per family.
- **A7.2 — What do SWA/EMA and ASAM actually buy?** Mean gain and, especially, seed-spread reduction. *Closes:* 5-seed duels each.
- **A7.3 — What recency half-life / window policy wins, per regime?** The cheapest Level-R lever. *Closes:* window-policy sweep as a `cfg` axis (Ch. 9 §9.3), charged to the trial counter.

### From Discussion 7 — Training design (Ch. 18)
*Settled: the injection test ("what does this stage know that end-to-end doesn't?"); cascade with OOF + shrinkage + the SNR decay law; skips yes / layer-staging no; ex-ante-not-ex-post weighting; the leverage table.*

- **A8.1 — Cascade or blend, and how many stages survive?** Prior: linear-first, two stages. *Closes:* duel-gated cascade vs. flat blend (Ch. 18 exercise 1).
- **A8.2 — How much of the committee does distillation retain?** Also: are students flatter (links A7.1)? *Closes:* student vs. seed vs. ensemble on RankIC + perturbation radius.
- **A8.3 — Which ex-ante weighting (vol-scaled, liquidity, era) helps, and how much?** *Closes:* one-factor-at-a-time weight duels; the ex-post reweighting demo stays in the ledger as the adjacent cautionary entry.

### From Discussion 8 — Loop engineering (Ch. 19)
*Settled: four nested internal loops (gradient / search / structure / adaptation) guarding the three overfitting levels; the external loop federates projects through signal and feedback events; two laws — all loops close through the ledger, and timescales must separate.*

- **A9.1 — Are the loop cadences matched to information half-lives?** Nightly L1 / monthly L3 / quarterly L2 are priors, not measurements. *Closes:* regret analysis on the ledger's history — would a faster/slower cadence have dominated?
- **A9.2 — Which proposal engine earns its budget?** Random, ASHA, LLM-reader, human — ranked by information gained per experiment. *Closes:* per-engine ledger attribution over a quarter.

---

## Tier B — Open at the design level (gates and revisit-triggers)

- **B1 — Incremental materialization.** No DAG/cache framework (Ch. 4 §4.4). *Trigger:* panel builds exceed ~15 minutes. *If tripped:* content-hash memoization at finer granularity first; a real DAG tool only after that fails.
- **B2 — Neural point processes (Strategy C, Ch. 10).** Parked as a documented extension. *Trigger:* a problem where event *timing itself* is the target (liquidity forecasting, cascade risk).
- **B3 — Sparse end-to-end MoE (Design C, Ch. 15).** *Trigger:* all-neural experts + millions of genuinely distinct samples (intraday clocks) + compute actually binding. Expect to need TRA-style collapse machinery; prefer never.
- **B4 — Stochastic dynamic programming over forecast uncertainty (Ch. 16).** Refused in favor of MPC re-solving + shrunk far-horizon alphas. *Trigger:* evidence that forecast-uncertainty *structure* (not just level) is predictable — a high bar.
- **B5 — Market-impact and borrow-fee fidelity (Ch. 12).** Linear costs only, by design. *Trigger:* strategy capacity analysis begins to matter (real capital), or A1.1 shows W3→W4 attrition that cost-model refinement would explain.
- **B6 — The t-bar function `t_required = f(n_trials)` (Ch. 9).** A function exists; its *shape* is a methodological choice (deflated-Sharpe-derived vs. simple monotone). *Open:* calibrate against the noise-rejection test — the bar is right when 50/50 noise features are rejected with minimal collateral damage to real ones.
- **B7 — Live-gap measurement.** The whole discipline aims at shrinking backtest-to-live gap, which no backtest can measure. *Trigger for design work:* paper-trading phase (Ch. 11 §11.6) — the monitor (Ch. 9 §9.5) is built; the *protocol* for attributing live shortfall (decay mis-estimate vs. regime vs. cost model) is unwritten.
- **B8 — Co-adaptation monitoring (Ch. 19).** Signal↔execution feedback can oscillate (predator–prey); only timescale separation prevents it today, nothing *detects* it. *Trigger:* first sustained live feedback; then build a cross-correlation watch between signal turnover and realized-cost series.

---

## Tier C — Open to you (unchanged from the charter, plus two new)

The five Step-2 charter questions in [index.md](index.md) remain unanswered — rejection probe, example probe, priority collision (simplicity vs. event-native), audience, and live-trading scope. Two more accumulated during Part V:

- **C6 — Data commitment.** Which real dataset anchors the capstone (crypto ticks were recommended for the event path)? Every Tier-A experiment needs it; this is now the binding constraint on starting the ledger.
- **C7 — The holdout pact.** Ch. 5 requires locking the final 1–2 years *before* iteration begins. That lock needs your explicit sign-off (it's a Ulysses pact — it only works if made once, in advance, by you).

---

## The state of the matrix (closing accounting)

| Quadrant | Where it stands |
|---|---|
| Q1 · Known knowns | The 18 chapters: principles, mechanisms, seams — the system design is settled. |
| Q2 · Known unknowns | Tier A: 20 named questions, each with a closing experiment the system makes cheap. |
| Q4 · Former unknown unknowns | Named and fenced: constraint-set overfitting, date-axis memorization, the Slutsky effect, noise-mining, expert collapse, the stacking trap — each now has a control on the panel (Ch. 17 §17.6). |
| Q3 + residue | Tier B/C: design gates awaiting triggers, and seven questions only you can answer. |

The pattern this table encodes — and the last thing the brochure has to teach: **a mature blueprint is not a list of answers. It is a list of questions, each priced, each with its closing experiment attached, and a system that makes asking them cheap.** Loom's build (Ch. 11) plus this page *is* the project plan: close C6 and C7, start the ledger, and work down Tier A.
