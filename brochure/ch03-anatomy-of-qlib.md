# Chapter 3 — Anatomy of qlib: an autopsy

*Prerequisites: Ch. 1–2. Builds: a concrete map of how a real system orchestrates this problem — and an evidence-based list of what to do differently. Every claim below cites a file in this repo; go read the bodies.*

You called qlib "not well written and malfunctioning in design patterns." Vibes are a fine starting hypothesis, but we're going to convict on evidence — because *the specific way* each pathology hurts tells us the design principle that prevents it. That's the method of this chapter: symptom → root cause → principle for Loom.

---

## 3.1 How qlib actually orchestrates (the honest tour)

A standard qlib workflow (`qrun config.yaml`, or the notebook equivalent) flows like this:

```
qlib.init(provider_uri=...)                                  # ①
   └─ mutates global config C            (qlib/config.py:526)
   └─ register_all_wrappers(C)           (qlib/data/data.py:1292)
        └─ fills global singletons: Cal, Inst, FeatureD, PITD,
           ExpressionD, DatasetD, D      (qlib/data/data.py:1284-1289)

YAML config ──► init_instance_by_config  (qlib/utils/mod.py:122)   # ②
   └─ instantiates classes from {"class": "...", "module_path": "...", "kwargs": ...}

DataHandlerLP                             (qlib/data/dataset/handler.py:383)  # ③
   └─ loader pulls expressions like "Ref($close,-2)/Ref($close,-1)-1"
      through the D singleton's expression engine
   └─ holds THREE dataframes: DK_R raw, DK_I infer, DK_L learn
      (handler.py:53-55) transformed by two processor chains

DatasetH(handler, segments={train/valid/test})               # ④
   └─ slices the handler's dataframes by date ranges

model.fit(dataset)                        (qlib/model/base.py)   # ⑤
   └─ the model PULLS data out of the dataset object,
      runs its own private training loop

Recorder / R (mlflow wrapper)             (qlib/workflow/__init__.py)  # ⑥
   └─ logs params, metrics, pickled artifacts

SignalRecord → backtest → PortAnaRecord                       # ⑦
   └─ predictions → TopkDropoutStrategy → exchange simulation → report
```

Hold that picture. It *is* the diagram from §1.5 — every research system converges to that diagram — but look at how the boxes are wired.

## 3.2 Six pathologies, with evidence

### P1 · The global singleton universe
`C = QlibConfig(_default_config)` at `qlib/config.py:526` is a **mutable module-level global**, and `register_all_wrappers` (`qlib/data/data.py:1292`) populates seven more module-level singletons (`data.py:1284-1289`). Every data access anywhere in qlib routes through `D`, which routes through `C`.

*How it hurts:* you cannot hold two data configurations in one process (compare US and China data side by side — you can't, cleanly); tests must mutate and restore global state; any function's behavior depends on who called `qlib.init` with what, invisibly. This is the textbook Service-Locator anti-pattern.
**Principle for Loom → components are values.** A `Store` is an object you construct and *pass*. Two stores? Two variables. Nothing reads ambient state.

### P2 · Stringly-typed configuration as the primary API
`init_instance_by_config` (`qlib/utils/mod.py:122`) turns `{"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt", "kwargs": {...}}` dicts — usually written in YAML — into live objects, recursively. Features are strings in a custom expression language, parsed at runtime.

*How it hurts:* typos surface as deep runtime stack traces, not at definition time; no IDE completion, no type checker, no refactoring support; the *real* program lives in YAML where no tool can see it; kwargs pass through untyped so version skew fails silently. Config-driven instantiation looks like flexibility but is actually an untyped, unversioned plugin protocol.
**Principle for Loom → configuration is code.** Experiments are Python dataclasses: typed, diffable, IDE-navigable, and *hashable* — the hash becomes the experiment's identity in the ledger (Ch. 9). YAML, if you want it, is a thin serialization of the dataclass, never the source of truth.

### P3 · `DataHandlerLP`: one object, four jobs
The handler (`qlib/data/dataset/handler.py:383`) loads raw data, holds it, transforms it via *two* processor chains into *three* internal dataframes — `DK_R`, `DK_I`, `DK_L` (`handler.py:53-55`) — and serves slices of them. Callers must pass the right `data_key` to get the right variant; processors are classified by `PROC_TYPE` flags; `fit_start_time/fit_end_time` thread through so normalizers don't leak.

*How it hurts:* the learn/infer distinction is real (you z-score labels for training but not for inference), but encoding it as *stateful copies inside a God object* means the leakage boundary is maintained by convention across ~800 lines, memory triples, and understanding any single output requires tracing both processor chains. Storage, transformation, and split policy are three responsibilities fused into one class.
**Principle for Loom → pipelines are pure functions over immutable panels.** `panel → transform(panel, stats_fit_on=train_slice) → panel`. The learn/infer difference is two small function compositions, not two mutable class attributes.

### P4 · Every model brings its own training loop
Look at `qlib/contrib/model/pytorch_gru.py`: 339 lines, of which the actual architecture is ~20 (a `nn.GRU` plus a linear head). The rest is a private `fit` / `train_epoch` / `test_epoch` / early-stopping implementation with loss hardcoded to MSE behind a string switch (`pytorch_gru.py:143-144`). Now `ls qlib/contrib/model/pytorch_*.py` — **27 files**, each repeating this loop with drift-y variations.

*How it hurts:* want IC loss instead of MSE across the zoo? Edit 27 files. Want gradient clipping changes, a scheduler, mixed precision? 27 files. The copies have already diverged (some reindex differently, some handle NaN masks differently) — so *models aren't comparable*, which defeats the purpose of a model zoo.
**Principle for Loom → one Trainer, models are `nn.Module`s, losses are parameters.** The architecture/optimization/objective axes must vary *independently* (Ch. 6–7). This single factoring is the biggest "faster self-improvement" win available: a new experiment touches one axis, not a copied loop.

### P5 · The model pulls; nothing owns the flow
qlib's `Model.fit(dataset)` hands the model a `Dataset` object to interrogate (`prepare("train", data_key=...)`). Data preparation policy — which columns, which segments, which data_key, how to batch — is decided *inside each model*.

*How it hurts:* the orchestration is nowhere. To know what data a model trained on, read that model's source. Samplers, augmentation, or a new split policy can't be added system-wide because each model privately negotiates with the dataset.
**Principle for Loom → the conductor pushes.** One visible function builds panels, makes tensors, and hands `(train, valid)` to a trainer. Models receive tensors; they never see the data layer. Inversion of control, restored.

### P6 · The calendar-grid foundation
Everything bottoms out in `CalendarProvider` (`qlib/data/data.py:65`) — aligned arrays indexed by trading-day slots. Elegant for daily bars; but anything irregular (news, PIT fundamentals, high-frequency) needs parallel machinery bolted alongside (a separate `PITProvider` with its own file format at `data.py:338`, a separate high-frequency example stack).

*How it hurts:* N data models = N query APIs = N chances to leak. You've already seen Loom's answer (Ch. 2): **events are the foundation; grids are views.**

## 3.3 What qlib gets *right* (steal these)

An autopsy that finds only disease is propaganda. Four ideas in qlib are genuinely good, and Loom keeps all four — as ideas, shorn of their mechanics:

1. **Point-in-time is a first-class concern** (`PITProvider`). Most academic code doesn't even know this problem exists. We generalized it (bitemporal events) rather than dropped it.
2. **The expression idea** — `Ref($close, -2)/Ref($close, -1) - 1` is a genuinely compact way to state features, and the engine caches/parallelizes evaluation. Loom keeps the *compactness* but hosts it in Python (plain functions + a tiny combinator library) so the type checker and the firewall test can see it.
3. **Rolling/walk-forward retraining is built in, with research attached** (qlib's `Rolling` workflows and the DDG-DA meta-learning work). qlib takes non-stationarity seriously. Ch. 9 builds the same capability with a simpler mechanism.
4. **Every run is recorded** (the mlflow `Recorder`). The reflex is correct and non-negotiable; Loom's ledger (Ch. 9) is a leaner take: append-only, hash-keyed, greppable.

## 3.4 The scorecard

| Axis | qlib | Loom's principle |
|---|---|---|
| State | global singletons (`C`, `D`, …) | components are values, passed explicitly |
| Config | YAML + class-path strings | typed dataclasses; config *is* code, hash = identity |
| Data model | calendar grid + bolted-on PIT | bitemporal event log; grids are views |
| Transform | stateful 3-frame handler | pure functions over immutable panels |
| Training | 27 private loops, loss welded in | one Trainer; model/loss/optimizer orthogonal |
| Orchestration | diffused into models & YAML | one conductor function, readable in one page |
| Leak defense | per-author convention | firewall tests in CI |

## 3.5 Build exercise (1–2 hours, in this repo)

1. Trace one line end to end: start at `examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml`, and follow `Alpha158`'s label string through `init_instance_by_config` → handler → dataset → `pytorch_gru.py`'s `fit`. Write down every file you had to open (I count ≥ 7). That number is the cost of diffused orchestration.
2. `diff qlib/contrib/model/pytorch_gru.py qlib/contrib/model/pytorch_lstm.py`. Estimate the true architectural difference (hint: one class name and constructor). Everything else in the diff is P4's tax.
3. Adversarial exercise: try to defend qlib. For P1 and P2, write three sentences on why its authors chose them (real answers exist: `D` gives notebook ergonomics; YAML gives non-programmers reproducible pipelines). Knowing the *temptation* is how you resist it in your own designs.

**Next:** we design the replacement. Seven components, one page of orchestration.
