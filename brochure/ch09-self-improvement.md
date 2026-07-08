# Chapter 9 — The self-improvement loop

*Prerequisites: everything so far. Builds: your question 4 — the loop that makes the system improve itself — and, equally, the guardrails that stop a self-improving system in an adversarial-noise domain from optimizing itself into fiction. "Faster self-improvement" was one of your three design goals; this chapter is why the other chapters were shaped the way they were.*

---

## 9.1 What "self-improving" actually means here

Not AGI mysticism. A research system self-improves when this cycle runs with minimal human time per revolution:

```
        ┌────────────────────────────────────────────────┐
        ▼                                                │
   PROPOSE cfg' ──► RUN (conductor, Ch.4) ──► LEDGER ──► SELECT
   (search, human,        │                     │        (what survived?)
    or LLM: §9.4)         ▼                     ▼            │
                      MONITOR (live IC,     ANALYZE          │
                      drift: §9.5)          (fingerprints)   │
        ▲                 │                                  │
        └── RETRAIN ◄─────┴── walk-forward schedule (§9.3) ◄─┘
```

The velocity of that cycle — *experiments per week, times bits of honest information per experiment* — is the system's true fitness. Every Loom decision was secretly about this: pure functions → results memoize and reproduce; config-as-dataclass → `cfg'` is `dataclasses.replace(cfg, …)`, so the *loop can write configs programmatically* (try generating qlib YAML-with-class-paths from a search loop, and you'll see why P2 kills loop velocity); one Trainer → a new loss is one experiment, not 27 edits; the ledger → SELECT is a query, not archaeology.

## 9.2 The ledger is the loop's memory

Upgrade the Ch. 4 ledger from diary to *substrate*:

- **`parent` links** → the experiment tree. "Show me every descendant of the GRU baseline that changed only the loss" is a filter, and the tree *is* your search history, auditable.
- **Fingerprints (Ch. 8)** → SELECT can require `new model beats parent` *paired* on the same days (a paired t-test on IC differences needs ~5× less data than comparing two noisy means — in a field where data is eras, this is the difference between an answerable and unanswerable question).
- **`n_trials` counters per family** → multiplicity control (Ch. 8 L4) becomes automatic: the loop knows it has tried 47 momentum variants, so the 48th needs a higher bar. **A self-improving loop without a trial counter is a self-deceiving loop.** This is the single most important sentence in this chapter.
- **Negative results are entries too.** The loop's efficiency next month depends on remembering what already failed and why.

## 9.3 Retraining: improvement without new ideas

The cheapest self-improvement is *keeping up with drift* (Ch. 1 §1.3b). The schedule is just the conductor run on a cron:

- **Rolling retrain:** every `step` (monthly/quarterly), refit on the trailing window, redeploy if the new artifact beats the incumbent on the most recent validation era (*champion–challenger* — never auto-deploy without the duel).
- **Window choice is a bias-variance dial:** expanding window = more data, staler; rolling = fresher, noisier. Recency weighting (Ch. 5 §5.5) is the continuous version and often dominates both. Make window policy a `cfg` field — the loop can search it like anything else.
- qlib ships this idea as its `Rolling` workflow and the DDG-DA meta-learning paper (predicting how the data distribution shifts, then pre-adapting). Read DDG-DA's *idea*; note that its qlib implementation is a multi-file trek precisely because of P4/P5 — in Loom it's a wrapper that reweights training samples before `train()`, ~40 lines, because the seams are data (Ch. 4 §4.3).

## 9.4 Search: improvement with new ideas

Three proposal engines, in ascending ambition — all emit `cfg'` into the same conductor, all land in the same ledger, all face the same SELECT bar:

**(a) Hyperparameter search.** Random search first (it's embarrassingly strong), then **ASHA/successive halving**: run many configs on *one early fold*, promote the top third to more folds. Walk-forward gives a natural fidelity axis — folds — that generic HPO frameworks don't know to exploit; your 30-line ASHA-over-folds beats a bolted-on Optuna in both honesty and simplicity here. Budget rule: HPO is the *lowest*-yield search direction in this domain — cap it (say 20% of compute) or it will happily consume everything while features starve.

**(b) Feature search.** The highest-yield direction (Ch. 6 §6.7 told you why). The loop: generate candidates (transformations of existing features: lags, ratios, cross-sectional ranks, interactions; LightGBM importances suggest *where* to look) → **screen cheaply** (univariate RankIC + correlation-vs-existing-features; a candidate correlated 0.9 with what you have is not information) → survivors get a real conductor run *added to the champion's feature set* → paired-fingerprint duel → adopt or ledger-and-discard. Note the multiplicity engine revving: thousands of candidates is exactly the §9.2 trial-counter scenario. Screen thresholds must scale with candidates generated.

**(c) LLM-in-the-loop.** The 2025-era upgrade: an LLM reads the ledger (it's JSONL; it's *made* to be read) — champion config, recent failures, fingerprint weaknesses ("IC dies in high-vol regimes") — and proposes hypotheses as candidate features or config edits. Treat it as a *fecund but unaccountable* proposal engine: everything it suggests enters at the same screen as (b), *no exceptions and no prompt-injected shortcuts around the SELECT bar*. The loop's integrity lives entirely in evaluation, so proposal can be as wild as you like. This division — creative proposer, ruthless mechanical judge — is the sane architecture for AI-accelerated research generally, and your ledger+conductor is precisely what makes it safe to try.

## 9.5 Monitoring: the loop's sensory organ

Deployed models rot. The monitor is the Evaluator run daily on live (or paper) scores, plus alarms:

- **IC control chart:** trailing 60-day mean IC with bands at ±2·SE from the *backtest era distribution* (not the live one). Breach → challenger duel triggered early. Remember Ch. 1's exercise: detecting that IC 0.03 became 0.00 takes *months* of daily data — set expectations and bands accordingly; a monitor that alarms weekly is a random-number generator with a pager.
- **CUSUM on daily IC** for slow bleeds the control chart misses.
- **Feature drift:** population-stability index per feature vs. training distribution. Drifted inputs precede degraded outputs — this alarm fires *earlier* than the IC ones and is nearly free.
- Every alarm and its resolution is a ledger event: the loop learns *its own* false-positive rate over time. That is meta-self-improvement, and it costs one JSON line per alarm.

## 9.6 The guardrails (this section is the difference between a research system and a fiction generator)

A loop that runs hundreds of experiments against finite noisy history will, *by construction*, find things that look great and are false (Ch. 8's 200-noise-signals exercise — now automated and tireless). Speed makes this worse, and you asked for speed. The guardrails, in order of importance:

1. **The locked holdout** (Ch. 5): the final 1–2 years exist outside the loop. The loop cannot read them; CI enforces it (the holdout files are literally absent from the loop's store view — a *capability* boundary, not a policy request). Spent once, Ch. 11.
2. **The trial counter + deflated bar** (§9.2): SELECT's threshold rises with the family's trial count. Codified, not vibes: `t_required = f(n_trials)` lives in one function everyone can read.
3. **Paired duels on fingerprints** (§9.2): challengers beat champions on *the same days*, or they don't pass. No cherry-picking eras.
4. **Peek budget:** the loop may evaluate on validation folds freely, but *test-fold* metrics are computed only for duel finalists — each test-fold read is a ledger event. You cannot overfit what you rarely observe, and you can *count* how often you observed it.
5. **Pre-registration for the endgame:** before touching the holdout, write in the ledger *which* model, *why*, and *what result would count as failure*. Then run it once. (You will feel the temptation to run it "just once more." That feeling is the entire epistemology of this field, distilled. The ledger entry is your Ulysses pact.)

**⚖ Design call.** Guardrails as *code in the loop* vs. researcher discipline: Loom chooses code, everywhere possible. Not because you're weak-willed — because the loop must remain trustworthy even when it (not you) is generating and judging 100 experiments a night, and because an auditable loop is one you can *show* someone (your Step-2 audience question, whoever the answer is).

## 9.7 Build exercise (a real week — this is the capstone's heart)

1. Wire the loop skeleton: `propose → conduct → select` over a grid of (loss × window policy × 2 features), champion tracked in the ledger. Fully automatic, end to end, including the paired duel.
2. Implement the trial counter and a rising t-bar. Feed the loop 50 *noise features* (you know they're noise). **The loop must reject all 50.** This is the shuffled-future canary's big sibling — the negative control for your entire research program. Do not skip it; a loop that passes noise is worse than no loop.
3. Implement the IC control chart on a simulated "live" feed (stream your test folds day by day). Inject a signal death (zero out the signal mid-stream); measure detection lag. Compare with your Ch. 1 exercise-4 prediction.
4. Optional stretch: one LLM-proposal round through the (b) screen. Ledger the outcome — including, especially, if it's a rejection.

**Next:** event streams — the second "case" you demanded, and the payoff of the Ch. 2 foundation: it's an extension, not a rebuild.
