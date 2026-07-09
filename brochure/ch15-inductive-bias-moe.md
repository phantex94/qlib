# Chapter 15 — Inductive bias, diverse ensembles, and the Mixture-of-Experts question

*Prerequisites: Ch. 6–9, 13–14. Builds: a working taxonomy of what each model family refuses to learn, the theory of why their disagreement is money, mechanical ways to "train for targeted directions" — and a first-principles answer to the MoE question: what transplants from LLMs, what breaks, and three designs in ascending coupling. Origin: Core Research Areas discussion 4 (the final one).*

---

## 15.1 Inductive bias is what a model refuses to consider

Expressive power isn't the interesting axis — with enough parameters everything is a universal approximator. The interesting axis is what each family finds *easy*, *hard*, and *impossible-by-construction*, because in a low-SNR world (Ch. 1 §1.3a) a model mostly learns what its bias makes cheap, and noise fills everything else. The taxonomy, in finance terms:

| | **Linear (ridge)** | **Trees / GBDT** | **Neural networks** |
|---|---|---|---|
| Hypothesis space | global, additive, monotone | piecewise-constant, axis-aligned splits | smooth, compositional |
| Finds easily | factor-style monotone signal | thresholds, regimes, interactions ("if vol>x *and* mom<y"), missing-data patterns | temporal dynamics (Ch. 6 r3), relational structure (Ch. 13), learned representations |
| Cannot express | any interaction or threshold | smooth global trends; anything temporal without hand-made features | — (that's the problem: it can express everything, including the noise) |
| Behavior outside the training hull | extrapolates linearly — predictable, sometimes wrong | **constant** — never extrapolates | interpolates confidently in learned space — unpredictable |
| Statistical appetite | tiny (why it survives IC-scale SNR) | moderate | large; seed-variance heavy (Ch. 6 §6.4) |

Two rows deserve a beat. The **extrapolation row is a volatility statement**: when a crisis pushes features outside the training hull, your three families fail in three *different directions* (linear keeps sloping, trees freeze at the boundary value, nets hallucinate smoothly) — which is precisely why holding all three is a hedge against the market's habit of leaving the hull at the worst moment. And the **appetite row is Ch. 13's asymmetry principle wearing model-family clothes**: match each family's capacity to the sample budget of the structure you're asking it to learn.

## 15.2 Why disagreement is money: the ambiguity decomposition

The first principle of ensembling (Krogh–Vedelsby) is an identity, not a heuristic. For an averaged ensemble:

$$E_{\text{ensemble}} \;=\; \bar{E} \;-\; \bar{A}$$

ensemble error = average individual error **minus** average *ambiguity* (how much members disagree with the ensemble). Diversity isn't a bonus — it's a subtraction, on the same axis as skill. Same-architecture seed ensembles (Ch. 6) harvest cheap ambiguity from optimization noise; **cross-family ensembles harvest structural ambiguity** — approximation errors living in different subspaces of function space — which is larger and doesn't shrink as training converges.

And you can *measure it before you build anything*: fingerprint correlations (Ch. 8 §8.3). Two streams with IC 0.03 each, fingerprint-correlated 0.4, blend to a materially higher ICIR than a single 0.04 stream. Architecture diversity is the cheapest diversity in the shop — same features, same data, one `ModelSpec` change — and the ledger prices it for you.

## 15.3 "Targeted directions," made mechanical

Your instinct — train each family *toward* what its bias favors — can be done by assignment or by construction:

**By assignment (information diets).** Give each family the sub-problem its bias fits: ridge on slow, residualized factor-style features (Ch. 12); GBDT on interaction-rich, missing-data-rich tabular/fundamental features; sequence nets on raw dynamics (Ch. 6 r3); attention/graph on the relational axis (Ch. 13). Different diets *also* decorrelate errors — you're diversifying on two axes (bias × information) at once, and Ch. 14's federation machinery treats each as just another stream.

**By construction (the residual cascade).** Make diversity structural: fit ridge; fit GBDT *on ridge's out-of-fold residuals*; fit the net on what's left. Each learner sees only what the previous bias could not express — decorrelation by construction, and the clean formalization of "targeted directions." Two disciplines keep it honest in low SNR: residuals must be **walk-forward out-of-fold** (in-sample residuals teach the next stage the previous stage's overfitting — the stacking trap of Ch. 14 §14.4 in another costume), and each later stage gets *less* capacity, because it's fitting a signal that is smaller and noisier by construction. Expect the cascade to beat naive blending modestly or not at all — run the duel; the ledger decides, as always.

## 15.4 The MoE transplant: what maps, what breaks

Now your final question. LLM-style MoE: N expert subnetworks, a learned router picks top-k per token, everything trains end-to-end with load-balancing auxiliary losses. The mapping to our problem is seductive — experts ↔ architectures or regime-specialists, router ↔ a gate on market state, token ↔ a (date, entity) sample — and the *motivation* transfers perfectly: different market regimes plausibly reward different hypotheses, exactly your volatility argument. Three things break in transit, each traceable to earlier chapters:

1. **The router's sample budget.** If experts specialize by *regime*, the router learns a function of dates — and its effective sample count is the number of regime transitions in your history: **dozens** (Ch. 13 §13.2, worst case). LLM routers see trillions of tokens. A confident learned router here is a date-memorizer with a steering wheel. Escape hatch below (§15.5-B): route on the *entity* axis, which is data-rich.
2. **Sparse routing + low SNR = collapse.** Top-k routing's known failure — the router sends everything to one expert and the rest atrophy — is driven by rich-get-richer gradient dynamics, and weak gradient signal makes it *worse*. Evidence in this very repo: qlib's TRA (`qlib/contrib/model/pytorch_tra.py`, 930 lines) is precisely a finance MoE — multiple "trading pattern" states with a router — and it needs **optimal-transport regularization** (`transport_method="router"`, the OT assignment at `pytorch_tra.py:230`) essentially to force the router to keep experts alive. When a method needs OT machinery to not degenerate, that's the domain telling you the failure mode is the default.
3. **Heterogeneous experts don't share a gradient.** GBDT and ridge can't sit inside an end-to-end differentiable MoE at all. LLM-style joint training is only available for all-neural expert sets — which surrenders §15.1's structural diversity, the thing that motivated the ensemble in the first place.

Plus one trading-specific cost nobody in the LLM literature has to think about: **routing discontinuity is turnover.** A top-1 router flipping experts at a regime boundary swaps your entire score vector in one step — a portfolio churn spike, paid in Ch. 12 costs, precisely at the volatile moment when trading is most expensive.

## 15.5 Three designs, ascending coupling

**Design A — the federated MoE (default).** Recognize that Ch. 14 already built an MoE: diverse-architecture streams as experts (predictions-as-events), and the Rung-3 conditional combiner as a **soft router on observable regime features**. Trained stagewise — experts first (each honestly walk-forward evaluated, fingerprinted, monitored), gate second, on the experts' out-of-sample emissions. Heterogeneous experts: no problem, nothing needs a shared gradient. Router capacity: explicitly budgeted small, shrunk toward equal weight — so it degrades to Rung 1, exactly the gated-residual philosophy of Ch. 13. This *is* a mixture of experts; it's just trained the way our sample budget can afford.

**Design B — the compact neural MoE (worth building once).** Inside one model: shared temporal encoder trunk (Ch. 13), K ≈ 3–5 expert heads, **soft dense gating** — no top-k; sparsity is a *compute* optimization for thousand-expert LLMs, and at K=4 it buys nothing while costing stability and turnover (§15.4). The one genuinely new idea to exploit: **route on the entity axis, not the date axis.** A gate reading *stock characteristics* (size, liquidity, vol, sector) learns across entities × dates — millions of samples, not dozens — and "small-caps obey different dynamics than mega-caps" is a far better-evidenced specialization than regime clairvoyance. Regularize with a load-balance term, an entropy floor on gate weights, and (novel but obvious in our framework) a **gate-smoothness penalty across consecutive dates** — the differentiable anti-turnover device, Ch. 12 thinking applied inside the architecture. Diagnostics are non-negotiable: log per-expert utilization and per-expert IC-by-regime into the ledger; the gate is a *measurement* of specialization, not just a mechanism.

**Design C — LLM-style sparse end-to-end MoE.** Justified when: all-neural experts, millions of genuinely distinct samples (large intraday/multi-asset universes on Ch. 14 clocks), compute actually binding. Even then: prefer soft gating or top-2 with high temperature, keep the smoothness penalty, and expect to reinvent TRA's collapse machinery. This is a Ch. 4 §4.4-style *seam, not a default* — the waterfall of evidence has to lead you here.

**⚖ Design call.** Loom's answer to "how do we achieve the optimal output": **A as the system, B as one expert within A, C only under evidence.** The optimum is not one perfectly routed monolith; it's a *shallow hierarchy* — structurally diverse, separately honest experts, combined by a small, humble, observable-feature gate that must duel equal-weighting to justify its existence. In this domain, given the date-axis budget, a near-uniform gate posterior is not a disappointing result — it's the *correct* one most of the time, and the occasions where the gate confidently deviates (and survives the duel) are findings worth a ledger note each.

## 15.6 Guardrails specific to mixtures

1. **The equal-weight duel** (Ch. 14 Rung 1) is the null hypothesis for every gating mechanism. Most published regime-switching gains die in this duel out-of-sample. Yours must survive it, paired, on fingerprints.
2. **Regime lag is real.** Any regime detector — HMM, vol threshold, the gate itself — identifies transitions *after* they happen, and transitions are where the money moves. Backtest the gate with its detection lag included, and remember the Ch. 9 monitor arithmetic: distinguishing "expert died" from noise takes months.
3. **Multiplicity, squared.** An MoE lets you search over experts × routers × regularizers — a combinatorial trial space. The Ch. 9 trial counter must charge the *whole tree*, not each leaf.
4. **Noise-expert canary.** Add one expert trained on shuffled labels. A healthy gate starves it to ~zero utilization; a gate that feeds it is memorizing dates. This is the Ch. 9 noise-rejection test, promoted into the architecture — and your sixth negative control.

## 15.7 Build exercise (5 hours) — the capstone's capstone

1. Train ridge, LightGBM, and your GRU on identical features/splits. Compute the 3×3 fingerprint correlation matrix and the ambiguity decomposition: verify numerically that $E_{ens} = \bar{E} - \bar{A}$ on your data, then compare the cross-family blend to a 3-seed GRU ensemble. Structural vs. optimization diversity, priced in your own ledger.
2. Residual cascade (ridge → GBDT on OOF residuals) vs. flat equal-weight blend of the same two. Duel, paired. Write the honest note.
3. Design B, minimal: your Ch. 13 encoder + 3 expert heads + soft gate on (size, vol, liquidity) characteristics, entropy floor, smoothness penalty. Log per-expert utilization by regime. Duel against the seed-ensembled single-head model.
4. The noise-expert canary (§15.6.4). Watch utilization; keep the plot.
5. Read `qlib/contrib/model/pytorch_tra.py` end to end — you now have every concept needed to understand all 930 lines, including *why* the OT trick exists. Measure how far you've come since Ch. 3 asked you to trace a YAML file.

---

## 15.8 Coda: the four discussions, one lesson

Look back at what Part V did. Execution constraints (Ch. 12) became a measurable waterfall and a defense ladder. TS-vs-XS (Ch. 13) became a factorization choice with a capacity principle and a gate. Bars and multi-scale integration (Ch. 14) became a clock parameter and a combination ladder. And architecture diversity with MoE (Ch. 15) became a shallow hierarchy with a humble router and four guardrails. Every time, the same move: **a question posed as "which architecture?" resolved into (a) a first principle about information or sample budgets, (b) a small mechanism at an existing seam, and (c) an experiment the ledger can adjudicate.** That move — not any particular answer — is what mastery in this field looks like. The system is done learning from me; from here, it learns from the ledger. So do you.
