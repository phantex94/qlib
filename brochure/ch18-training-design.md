# Chapter 18 — Training design: cascades, curricula, and the reweighting trap

*Prerequisites: Ch. 6–7, 15, 17. Builds: training as a designed path rather than a single fit — the cross-family residual cascade made rigorous, residual learning inside the network sorted into what's real and what's redundant, and a first-principles inversion of boosting-style hard-example weighting. Origin: Core Research Areas discussion 7.*

---

## 18.1 Training design is a path, not a point

Everything before this chapter chose a *point*: one model, one loss, one weighting, fit once. Your proposals are all versions of a deeper idea: **training as a trajectory** through three axes, each stage conditioning the next —

- the **model-family axis**: linear → tree → net, each fitting the previous stage's residual (§18.2);
- the **architecture axis**: residual structure *inside* the network (§18.3);
- the **data axis**: reweighting samples across stages, boosting-style (§18.4).

Plus one axis you didn't name but already use: the **objective axis** — MSE pretrain → IC → ranking → Sharpe/end-to-end fine-tune (Ch. 7 §7.5, Ch. 12 §12.5), which is a curriculum over losses. The unifying question for any staged design is always the same: *does the staging inject information (bias diversity, noise structure, constraint knowledge) — or does it just re-order what joint training would find anyway?* Stages that inject, pay. Stages that re-order, cost complexity for nothing. Hold that test against each proposal.

## 18.2 The cross-family cascade, made rigorous

Ch. 15 §15.3 seeded this; now the mechanics. Fit ridge; fit GBDT on ridge's residuals; fit the net on what remains — recognize it formally: **stagewise additive modeling (gradient boosting) with heterogeneous learners**, and each stage sees only what the previous *bias could not express* (§15.1's taxonomy is doing the work). Passing the §18.1 test: the staging injects bias diversity *and* rations capacity — the flexible net never gets to re-learn (and overfit) the easy linear share, which in equities is the *majority* share (Gu–Kelly–Xiu). This is also Ch. 12's residualization generalized: there you orthogonalized labels against *risk factors*; here against *the span of simpler models*. Same operation, different basis — "predict what the simpler explanation can't."

Four disciplines, of which two are new:

1. **OOF only** (from Ch. 15): stage $k{+}1$ trains on stage $k$'s walk-forward out-of-fold residuals — in-sample residuals teach the next stage to invert the previous stage's overfitting.
2. **The SNR decay law (new, and load-bearing).** The residual after a competent stage is the original noise plus a *smaller* signal: signal variance shrinks stage by stage while noise variance doesn't. **SNR strictly decreases down the cascade.** Ch. 1 said capacity must match SNR; therefore capacity must *decrease* and regularization *increase* down the cascade — which inverts the naive instinct of "save the biggest model for last." The DL stage in your layer-three slot should be the *smallest, most regularized* net in the book, hunting a signal fainter than anything in Ch. 6. Most cascade failures are exactly this: a large stage-3 net enthusiastically fitting stage-2's noise.
3. **Shrinkage between stages:** add only $\nu \cdot f_k$ with $\nu \approx 0.3\text{–}0.7$ — boosting's oldest trick, and in low SNR not optional: full-strength stage additions hand the next stage a residual dominated by the previous stage's estimation error.
4. **Stop by duel:** each stage must beat the truncated cascade in a paired fingerprint duel (Ch. 9) or the cascade ends. Expect two stages to survive often, three rarely — the SNR decay law says so.

**Cascade vs. flat blend** (Ch. 15's ensemble): the cascade wins when the biases are genuinely *nested* — the first stage captures a large variance share the later ones would otherwise waste capacity re-deriving — and when you want interpretability by construction (each stage's contribution is a separable, attributable signal). The blend wins when errors are roughly symmetric across families and you'd rather not inherit the cascade's sequential fragility (stage-1 drift silently poisons stages 2–3; the blend degrades gracefully). Ledger question with a prior: in equities, linear-first cascades are well-motivated because the linear share is large; past two stages, expect the duel to say stop.

## 18.3 Residuals inside the network: mostly already yours

Your "equivalently, residual learning per layer within the DL model" bundles two different things:

**(a) Skip connections (ResNet form: $x_{l+1} = x_l + f_l(x_l)$).** Adopt unconditionally — but be clear about *why*: identity initialization means each block starts as a no-op and learns a perturbation, and — the Ch. 17 connection — skip connections dramatically *smooth the loss landscape* (Li et al.'s visualizations show chaotic surfaces turning convex-ish when skips are added). Flatter, better-conditioned surfaces are exactly what §17.3 argued our shifted-surface problem needs. This is architecture hygiene, ~2 lines per block, already implicit in the Transformer blocks of Ch. 6/13.

**(b) Explicit within-net staging** — train layer 1's head to predict $y$, freeze, train layer 2 on the residual, and so on (GrowNet-style boosted shallow networks). Apply the §18.1 test: within one family, staging injects *no new bias* — every stage is the same function class — so its only effect is to constrain the optimization path that joint training with skips explores freely. The evidence agrees: stagewise-trained nets rarely beat jointly-trained residual nets of matched capacity. Verdict: **skips give you within-net residual learning implicitly and jointly; explicit within-net boosting is mostly a slower way to the same place.** One worthwhile exception: *boosted shallow MLPs* (2-layer weak learners, GBDT-style, with shrinkage) as a middle rung between LightGBM and deep nets on tabular features — one ledger duel's worth of curiosity. And note you already own the pattern's best form: Ch. 13's gated mixer ($s = s_{ts} + g \cdot s_{xs}$, gate initialized ≈ 0) *is* within-model residual learning across a real bias boundary — temporal → relational — with the gate playing the role of learned shrinkage $\nu$. Residual staging pays exactly when it crosses a bias boundary; Ch. 13 crossed one, layer-by-layer staging doesn't.

## 18.4 Hard-example weighting: invert it

Here I'll push back on the proposal directly, because it contains the chapter's most instructive trap.

AdaBoost-style logic — upweight what the current model gets wrong — is sound when labels are *clean*: a misfit sample marks model inadequacy (a boundary region needing attention). Now run the logic in our regime. Labels are $y = s + \varepsilon$ with $\text{Var}(\varepsilon) \approx 100 \times \text{Var}(s)$ (Ch. 1). A sample with a large residual is, with overwhelming probability, a sample with a large *noise draw* — its misfit carries almost no information about model inadequacy. **Hard-example mining in low SNR is a noise-mining machine**: it systematically concentrates training weight on exactly the observations whose labels are least informative. This isn't hypothetical — AdaBoost's collapse under label noise is one of the best-documented failures in classical ML (the exponential loss piles weight onto corrupted points until they dominate), and financial labels are, functionally, ~99% "corrupted."

So the first-principles move is the **inversion**: in low SNR, *down-weight* the hard examples. That's what several things you already do secretly are:

- **Winsorized / rank-normalized labels** (Ch. 5 §5.2): clipping extreme labels *is* down-weighting the samples boosting would have upweighted hardest.
- **Huber-type robust losses**: gradient saturates on large residuals — soft down-weighting of the noisiest points.
- **Volatility-scaled weighting**: weight samples by $1/\hat\sigma_i^2$ (inverse *ex-ante* label variance). High-vol names have noisier labels; this is principled heteroskedasticity handling (GLS in ML clothing), and it down-weights precisely the structurally-hard samples.
- **Easy-first curricula**: train early epochs on the cleaner subset (liquid names, calm eras), widen later — curriculum learning, pointed the correct direction for our noise regime.

The rule that separates legitimate reweighting from the trap, in one line: **weight on *ex-ante* noise estimates (volatility, liquidity, data quality, era representation), never on *ex-post* fitting errors.** Ex-ante weights encode structure you know; ex-post error weights, at our SNR, encode the noise realization. Every scheme in the legitimate list above is ex-ante; AdaBoost is ex-post. (The one honest ex-post-flavored exception: *era* reweighting where whole underrepresented regimes get upweighted — Ch. 5 §5.5 — because eras are the unit at which "hard" can mean "structurally different" rather than "noisy draw." Even then, weight by regime identity, not by regime loss.)

And note the pleasing symmetry: gradient boosting with *squared* loss (your tree stage in §18.2) survives low SNR because residual-fitting spreads attention smoothly, while AdaBoost's exponential reweighting does not. Your first proposal and your third proposal are, under the hood, the benign and malignant versions of the same idea.

## 18.5 My opinion, assembled: the training-design leverage table

You asked directly, so — ranked by expected metric improvement per unit of added machinery, on this domain:

1. **Label & loss design** (rank-normalized labels, IC/pairwise losses — Ch. 5, 7). Still the biggest lever; nothing in this chapter outranks it.
2. **Early stop on the judge's metric + seed ensembles + SWA/EMA** (Ch. 6, 17). Boring, near-free, compounding.
3. **The objective curriculum** (MSE→IC→ranking→cost-aware fine-tune; Ch. 7, 12). Staging across losses injects real information (the judge's shape) at every step — the best-performing "residual" idea in the book is on the loss axis, not the model axis.
4. **Ex-ante sample weighting** (vol-scaled, liquidity, recency, era-balanced — §18.4's legitimate list). Small, principled, additive.
5. **Cross-family cascade with shrinkage + SNR-decayed capacity** (§18.2). Modest but real; two stages usually.
6. **Multi-task auxiliaries** (the vol head, multi-horizon heads — Ch. 7, 16). Regularization through related tasks; cheap.
7. **Distill the committee**: train your 5-seed ensemble, then distill it into one net matching the ensemble's *scores*. Distillation targets have far higher SNR than raw labels (the ensemble averaged the noise away), so the student trains on a cleaner problem — and you deploy one model with most of the committee's metric. Underused in finance; recommended.
8. **Within-net explicit staging / GrowNet** (§18.3). One curiosity duel.
9. **Ex-post hard-example weighting** (§18.4). Negative expected value at our SNR. Run the exercise below once, frame the result, never again.

The meta-opinion, echoing §18.1: training designs pay when a stage *injects information the joint fit can't see* — a different inductive bias, the judge's loss shape, ex-ante noise structure, a cleaner distillation target. Designs that merely re-sequence the optimization (within-family staging, error-chasing weights) add fragility and multiply the Ch. 9 trial count without adding information. Ask "what does this stage know that end-to-end doesn't?" — if the answer is nothing, delete the stage.

## 18.6 Build exercise (4–5 hours)

1. **The disciplined cascade.** Ridge → LightGBM(OOF residuals, $\nu{=}0.5$) → *small* MLP (obey the SNR decay law: half the Ch. 6 width, double the dropout). Paired duels at each stage vs. the truncated cascade, and the full cascade vs. the flat 3-family blend of Ch. 15. Two ledger notes: where did the duel say stop, and cascade-vs-blend.
2. **The trap, demonstrated.** Implement AdaBoost-style reweighting (weight ∝ |residual|) on your MLP for 3 rounds. Watch validation RankIC degrade while train loss on the upweighted set improves — noise-mining, observed live. Then run the same loop with weights ∝ $1/\hat\sigma^2$ (ex-ante) and watch the sign flip. This pair of runs is the whole §18.4 argument in two ledger entries; keep them adjacent.
3. **Distill the committee.** 5-seed GRU ensemble → single student trained on ensemble scores (plus a small true-label term). Compare student vs. single seed vs. full ensemble on OOS RankIC and on Ch. 17's perturbation-radius score — distilled students are usually *flatter*, a satisfying convergence of two chapters.
4. **Skip-connection landscape check.** Train your MLP with and without skips; compare Ch. 17 diagnostics (perturbation radius, seed mode-connectivity barrier). Seeing the barrier drop when skips go in is Li et al.'s famous figure, reproduced on your own problem.

**Ledger meta-note (template Step 5):** discussion 7's three proposals sorted cleanly under one test — *does the stage inject information?* The cascade injects bias diversity (keep, with the SNR decay law); within-net staging injects nothing skips don't (skip the ceremony, keep the skips); hard-example weighting injects the noise itself (invert to ex-ante). Training design, it turns out, is governed by the same law as everything else in this book: you can't create information at training time — only route it, ration it, or chase its absence.
