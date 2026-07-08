# Chapter 11 — Capstone: build Loom

*Prerequisites: all chapters, all build exercises. This chapter turns the pieces you've already written into a system, gates your weeks, and defines what "master" means measurably.*

---

## 11.1 The repository you're building

```
loom/
  loom/
    store.py        # ~250 loc  event log, asof, truncated, views, ingestion adapters
    features.py     # ~150 loc  @feature/@label registry, lookback enforcement
    panel.py        # ~250 loc  build_panel, walk_forward, normalizers, 3 packers, samplers
    train.py        # ~200 loc  Trainer, TrainSpec/LossSpec/OptimSpec, LOSSES, artifacts
    models.py       # ~200 loc  ridge, lgbm adapters; MLP, GRUNet, XSecTransformer
    eval.py         # ~200 loc  IC family, NW t-stats, decay, deciles, backtester, fingerprints
    ledger.py       # ~100 loc  append, lookup, duel, trial counters, t-bar
    conduct.py      # ~80  loc  run(), the loop from Ch. 9
  tests/
    test_firewall.py        # truncation invariance + shuffled-future canary  (Ch. 2)
    test_purge.py           # brute-force split overlap check                 (Ch. 5)
    test_backtest.py        # hand-computed fixture                           (Ch. 8)
    test_noise_rejection.py # the loop rejects 50 noise features              (Ch. 9)
    test_padding.py         # event-mask invariance                           (Ch. 10)
  experiments/      # your ledger + configs; the actual research lives here
```

~1,430 core lines against the 1,500 budget (Ch. 4). If a file wants to double, something is trying to become a framework — reread Ch. 4 §4.4 before permitting it. Notice what the test directory is: **every test encodes one way this field fools people.** That test suite is arguably worth more than the system it tests, and it is portable to every research stack you'll ever touch.

## 11.2 The six-week schedule, with gates

Each gate is an *acceptance test* — objectively passed or not. Don't advance on a red gate; the chapters compound.

| Week | Build | Gate (must pass) |
|---|---|---|
| 1 | Ch. 1–2 exercises: fundamental-law sim; event-log Store | Firewall test 1 catches your deliberately-planted lookahead |
| 2 | Ch. 3–4: qlib trace; Loom interfaces; conductor skeleton | The 7 protocols fit in ~150 lines; conductor dry-runs on fixtures |
| 3 | Ch. 5: panel, walk-forward, samplers | Purge test green; **shuffled-future canary green end-to-end** |
| 4 | Ch. 6–7: Trainer + 5 models + 3 losses | Ridge floor & LGBM benchmark ledgered; GRU and MLP share the Trainer *unmodified*; IC loss beats/ties MSE on RankIC |
| 5 | Ch. 8–9: Evaluator, backtester, the loop | 200-noise-signals plot made; **noise-rejection test: 50/50 rejected**; champion–challenger duel runs unattended overnight |
| 6 | Ch. 10 + endgame: event features, drift study | Event features duel the champion; pre-registration written; **holdout spent — once** |

**The endgame ritual (Week 6, day 5).** Write the pre-registration ledger entry (Ch. 9, guardrail 5): chosen model, expected RankIC range, failure criterion. Run the holdout. Whatever happens, write the honest postmortem entry. A null result documented with this much rigor is a *strong* outcome — it's the difference between "my backtest was good" (worthless sentence) and "my research process is trustworthy" (hireable sentence).

## 11.3 The mastery checklist

You are a *master* of this material — not of markets; nobody masters markets — when you can do these without notes:

**Explain (whiteboard, 5 minutes each):**
- [ ] Why IC 0.03 can be a business, and what breadth really counts (Ch. 1)
- [ ] The five leakage disguises and which layer of Loom kills each (Ch. 2, 5)
- [ ] Why bitemporal events unify time series and event streams (Ch. 2, 10)
- [ ] qlib's six pathologies *and the temptation behind each* (Ch. 3 — the steelman is the mastery test)
- [ ] The loss frontier from MSE to Sharpe surrogates, and its price axis (Ch. 7)
- [ ] Why a trial counter is the load-bearing wall of a self-improving loop (Ch. 9)

**Do (the ledger proves it):**
- [ ] All five tests in `tests/` green, including both negative controls
- [ ] ≥ 30 ledger entries, at least 10 of them negative results with causes
- [ ] One paired duel where you *rejected* a model you liked
- [ ] One event study, honestly purged, whatever its result
- [ ] The holdout: spent once, pre-registered, postmortem written

**Judge (the real test):** hand this brochure to someone and defend, against pushback, one place where you'd design Loom *differently*. If six weeks of building haven't produced a genuine disagreement with me, you followed instructions — you didn't master the material. The ⚖ blocks were the invitation.

## 11.4 Where this leaves you vs. qlib (your original three criteria)

- **Simpler & more elegant:** seven components, data-only interfaces, one-page conductor, ~1,430 lines vs. a codebase you needed 7 files to trace one label through. You didn't get this by writing less — you got it by *deciding more* (Ch. 4 §4.4's refusals did the work).
- **Faster self-improvement:** config-as-value + memoizing conductor + ledger-with-duels means an experiment is `replace(cfg, loss="pairwise")` and minutes later an honest verdict. The loop runs overnight and cannot lie to you within its stated guardrails — velocity *times* trustworthiness is the actual metric, and guardrails are what let you spend the velocity.
- **Time series + event streams:** one substrate, two sampling policies, five of seven components shared verbatim (Ch. 10 §10.1). Flexibility came from the data model, not from abstraction layers — the deepest design lesson in the brochure.

## 11.5 The reading map (after, not instead of, building)

- **Grinold & Kahn, *Active Portfolio Management*** — the fundamental law and everything around it; Ch. 1 was a compression of its spirit.
- **López de Prado, *Advances in Financial Machine Learning*** — purging, embargo, deflated Sharpe, meta-labeling; the paranoia canon. You've already implemented its best chapters.
- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** — the honest empirical baseline for what ML buys in equities.
- **Bailey & López de Prado, "The Deflated Sharpe Ratio"** — the multiplicity math behind Ch. 8 L4 and Ch. 9's rising bar.
- **qlib's papers** (Qlib: An AI-oriented Quantitative Investment Platform; DDG-DA) — read them *now*, after the autopsy; you'll see both what they were reaching for and what the implementation cost them.
- **Du et al., neural temporal point processes** (survey) — when Strategy C's day comes.

## 11.6 After the capstone

Three worthy directions, in rising ambition: (1) **a real dataset upgrade** — crypto tick data through the event path, where Strategy B has genuine bite; (2) **the LLM proposal engine** (Ch. 9 §9.4c) made real, with the screen doing its job — you have the safe harness few people have; (3) **live paper trading** — the monitor (Ch. 9 §9.5) against reality, where you finally measure the backtest-to-live gap that all of this discipline was designed to shrink.

And one standing instruction, from your own template (Part B, Step 5): at each milestone, ask *"what did we learn that we didn't know we didn't know?"* — and write it in the ledger. The filled ledger is the reusable residue of the project. It always was the product.

---

*End of brochure. Argue with it — that was the point.*
