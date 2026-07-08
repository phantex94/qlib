# Chapter 4 — Designing Loom

*Prerequisites: Ch. 1–3. This chapter answers your questions 1 and 2 directly: what the components are, what each is responsible for, and exactly how data flows between them.*

---

## 4.1 Design principles (the whole philosophy in five lines)

1. **Components are values.** Constructed explicitly, passed explicitly. No singletons, no ambient state, no `init()`.
2. **Truth is an event log; everything else is a pure function of it.** Same input hashes → same outputs. Caching, reproducibility, and debugging all fall out of this one property.
3. **Configuration is code.** A frozen dataclass per experiment; its content-hash is the experiment's identity.
4. **The conductor fits on one page.** If you can't read the whole orchestration in one screen, the architecture has failed — no matter how good the parts are.
5. **The loop is the product.** Loom is optimized for *research iterations per week*, not for any single run. Every design choice is scored on: does it make the next experiment cheaper?

Budget: the core is ≤ ~1,500 lines. That's not asceticism — it's the second-system-effect defense from the risk register, and it's what makes principle 4 achievable.

## 4.2 The seven components and their responsibilities

```
loom/
  store.py       # ① Store        — bitemporal event log + as-of queries + views
  features.py    # ② FeatureLib   — feature/label definitions, pure, firewall-tested
  panel.py       # ③ PanelBuilder — panels, splits, normalization, tensor packing
  train.py       # ④ Trainer      — the ONE training loop; models/losses/optims plug in
  eval.py        # ⑤ Evaluator    — IC-family metrics + backtest
  ledger.py      # ⑥ Ledger       — append-only run records, hash-keyed
  conduct.py     # ⑦ Conductor    — the one-page orchestration
```

### ① Store — *"what was knowable, when"*
**Owns:** the append-only event log (Ch. 2); `asof()`, `truncated()`; materialized views (the daily bar cube; latest-fundamentals-as-of).
**Guarantees:** no API returns events with `t_observed > t`. Append-only: ingestion adds, never edits.
**Must not:** compute features, know about models, know about "train/test" — time discipline only.

### ② FeatureLib — *"named, tested, versioned transformations"*
**Owns:** a plain-dict registry of features and labels. Each is a pure function with a declared lookback and version:

```python
@feature(lookback="60d", version=2)
def mom_20(view: BarView, t: Timestamp) -> pd.Series:      # one value per entity
    px = view.close.last("21d")
    return px.iloc[-1] / px.iloc[0] - 1

@label(horizon="5d", exec_lag="1d", normalize="xsec_rank")
def fwd_ret_5d(view, t): ...
```

**Guarantees:** every registered feature passes the truncation-invariance firewall in CI. `lookback` bounds how much history the engine loads (performance) *and* is itself checked (a feature reading beyond its declared lookback fails).
**Must not:** touch files, know about splits or models. — This is qlib's expression engine reborn as ordinary Python: composable with the full language, visible to the type checker, and 10× less machinery.

### ③ PanelBuilder — *"from log to tensors, without lies"*
**Owns:** assembling `Panel = (X: [dates × entities × features], y, meta)`; **splits** (purged walk-forward with embargo, Ch. 5); **normalization** with statistics fit on train slices only; packing tensors for tabular `(N,F)`, sequence `(N,T,F)`, and event `(N,[events])` model families.
**Guarantees:** a split is a *value* (`Split(train=…, valid=…, test=…, purge=…, embargo=…)`) recorded in the ledger; normalizers are returned as fitted objects so inference reuses train statistics — qlib's DK_I/DK_L distinction, expressed as two function calls instead of stateful copies.
**Must not:** fetch anything not via Store; ever fit a statistic on non-train rows.

### ④ Trainer — *"one loop to rule the 27 files"*
**Owns:** the single fit loop: batching, epochs, early stopping (on validation *RankIC*, not loss), gradient clipping, checkpointing, seeds.
**Plugs in (this is Ch. 6–7's playground):**

```python
@dataclass(frozen=True)
class TrainSpec:
    model: ModelSpec          # architecture + hyperparams (builds an nn.Module)
    loss: LossSpec            # "mse" | "ic" | "listmle" | custom callable
    optim: OptimSpec          # optimizer factory + schedule
    epochs: int = 100
    patience: int = 10
    seed: int = 0
```

**Guarantees:** any (model × loss × optimizer) combination runs — the three axes vary independently. Non-gradient models (ridge, LightGBM) satisfy the same `fit(panel) → Artifact` / `predict(X) → scores` contract via thin adapters, so the Evaluator can't tell the difference.
**Must not:** know where data came from. It receives tensors.

### ⑤ Evaluator — *"the judge, and the honest one"*
**Owns:** daily IC/RankIC series + Newey-West t-stats, IC decay curves, quantile spreads, turnover; the backtester (scores → portfolio weights → net-of-cost PnL, Ch. 8).
**Guarantees:** every metric computed *per era* (per day/week), never pooled — pooled metrics hide regime failure (Ch. 1 §1.3c).
**Must not:** be import-coupled to any model. It sees `(scores, realized_returns, meta)` — dataframes in, verdict out.

### ⑥ Ledger — *"the lab notebook the loop reads"*
**Owns:** one append-only JSONL (or SQLite) of runs:

```json
{"run": "a3f9…", "config_hash": "…", "data_hash": "…", "git": "…",
 "split": {...}, "metrics": {"rank_ic": 0.041, "icir": 0.38, "sharpe_net": 1.7},
 "artifacts": "runs/a3f9/", "parent": "88c1…", "note": "ic loss vs mse, same GRU"}
```

**Guarantees:** identical `(config_hash, data_hash)` → the cached result is returned, not recomputed (memoization for free, from principle 2). `parent` links make the search tree explicit — Ch. 9 turns this from a diary into the substrate of self-improvement.
**Must not:** require a server. It's a file. `grep` is a query language.

### ⑦ Conductor — *"the whole system, legible"*

```python
def run(cfg: ExperimentCfg, store: Store, ledger: Ledger) -> Result:
    if (cached := ledger.lookup(cfg.hash, store.data_hash)):        # principle 2
        return cached
    panel   = build_panel(store, cfg.features, cfg.label, cfg.universe)   # ③←②←①
    result  = []
    for split in walk_forward(panel.dates, cfg.split):                    # ③
        tr, va, te   = panel.tensors(split)                               # ③
        artifact     = train(cfg.train_spec, tr, va)                      # ④
        scores       = artifact.predict(te.X)                             # ④
        result.append(evaluate(scores, te, cfg.costs))                    # ⑤
    report = aggregate(result)
    ledger.append(cfg, store.data_hash, report)                           # ⑥
    return report
```

That's it. That is the answer to *"how is everything orchestrated."* Not a framework, a **function** — every arrow of the Ch. 1 diagram is a visible line; adding a stage (a feature-selection pass, an ensemble) is editing a page of code you fully understand. The self-improvement loop (Ch. 9) is just a second function that calls this one in a loop with varying `cfg`.

## 4.3 The dataflow, end to end

```
            ingestion adapters (yfinance, exchange, files…)
                          │ append events
                          ▼
 ①  STORE   [ bitemporal event log ]──materialize──►[ bar cube / views ]
                          │ asof(entities, kinds, t)
                          ▼
 ②  FEATURELIB   f(view, t) → Series      (firewall-tested, lookback-bounded)
                          │ feature & label columns
                          ▼
 ③  PANELBUILDER  Panel(X, y, meta) ──► Split ──► normalized tensors
                          │ (train, valid) tensors          │ test tensors
                          ▼                                 ▼
 ④  TRAINER   TrainSpec(model, loss, optim) → Artifact ──► scores
                                                            │
                          ┌─────────────────────────────────┘
                          ▼
 ⑤  EVALUATOR  IC/RankIC per era ─► backtest w/ costs ─► Report
                          │
                          ▼
 ⑥  LEDGER    append {config_hash, data_hash, metrics, artifacts, parent}
                          │
                          ▼
 ⑨  LOOP (Ch. 9)  propose next cfg ──► CONDUCTOR ⑦ ──► (back to ③)
```

Interfaces between boxes are **data, not objects**: Series, Panels, tensors, dataframes, JSON. That's the deepest difference from qlib, where boxes hold references to each other (model→dataset→handler→D→C) and the flow is a call graph you excavate. Data interfaces mean each box is testable with fixtures, replaceable in isolation, and — for the event-stream case — reusable: only ② and ③ grow event-aware variants (Ch. 10); ①, ④–⑦ don't change.

## 4.4 What we deliberately did NOT include

Restraint is a feature. No plugin system (you own the code — edit it). No DAG scheduler (the conductor is the DAG; Python is the scheduler). No config language (Python is the config language). No distributed anything (one GPU box outruns your idea rate for years; principle 5 says optimize idea rate). No live-trading module (out of scope by charter). Each of these is a place qlib or its peers spent complexity that a research system doesn't need — and each can be *added later at the seams*, because the seams are data.

**⚖ Design call.** The contestable one: no DAG/cache framework (Airflow, Dagster, even `make`). Counterargument: at 10× more features and multi-hour panel builds, you'll want incremental materialization. Loom's bet: content-hash memoization at the conductor level (①'s `data_hash` + cfg hash) captures 90% of that value at 1% of the machinery. Revisit if panel builds exceed ~15 minutes.

## 4.5 Build exercise (2 hours)

1. Write all seven interfaces as Python `Protocol`s / frozen dataclasses — signatures only, `...` bodies. Fitting them in ~150 lines *is* the exercise; wherever you can't, your understanding of a responsibility is fuzzy — reread that section.
2. Hand-simulate the conductor on paper for `cfg = (features=[mom_20], label=fwd_ret_5d, model=ridge)`: write every intermediate value's *type and shape* at each of the 8 lines.
3. Adversarial: where would *you* add a plugin system anyway? Write the argument, then the rebuttal from principle 5.

**Next:** the panel layer in earnest — splits that don't lie, and tensors for three model families.
