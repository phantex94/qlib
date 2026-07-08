# Chapter 2 — Data: the event log and the temporal firewall

*Prerequisites: Ch. 1. Builds: Loom's single most important design decision, and the test that protects your entire research program.*

---

## 2.1 The one deadly sin, in five disguises

Every impressive backtest you will ever produce is guilty until proven innocent of **leakage** — information from after $t$ contaminating a prediction "made at" $t$. It comes in five practical disguises:

1. **Lookahead in features.** Using today's close in a signal that trades at today's open. Using a moving average aligned wrong by one bar. Off-by-one errors here are *silent* and *profitable-looking* — the worst combination.
2. **Restatement bias (the point-in-time problem).** Q3 earnings are "for" September 30 but *announced* November 2 — and possibly *restated* in January. If your dataset stores one number per quarter, you are trading on November's (or January's!) knowledge in October. Fundamental datasets that keep only final values are poisoned at the source. qlib, to its credit, has real PIT support (`qlib/data/data.py` has a dedicated `PITProvider`); most homemade systems don't.
3. **Survivorship bias.** Building your universe from *today's* index members means every backtested day secretly knows who survived. Enron never bankrupts a survivorship-biased backtest. The universe itself must be point-in-time.
4. **Normalization leakage.** Fitting a z-score, PCA, or any statistic on the full dataset, then splitting. The train rows have tasted test-set moments. Subtle version: cross-sectional ranks are safe (same-day only), but *time-series* normalizations fit over the whole history are not.
5. **Split contamination.** Random K-fold on time series puts tomorrow in train and today in test. With overlapping labels (a 5-day forward return spans 5 rows), even *contiguous* splits leak at the boundary — hence purging and embargo (Ch. 5).

The disguises share one root cause: **the dataset forgot when each fact became known.** So the fix is not vigilance. The fix is a data model in which every fact carries its knowledge time — making most leaks *unrepresentable* rather than merely discouraged.

## 2.2 The bitemporal event log: Loom's foundation

**⚖ Design call — the big one.** Loom's ground truth is a single, append-only **event log**. Every fact, of every kind, is one event:

```
Event:
  entity      : str          # "AAPL", "BTC-USD", "US-CPI", "*" (market-wide)
  kind        : str          # "bar.1d", "earnings", "news", "trade", "membership"
  t_effective : timestamp    # when the fact is *about*   (Q3 quarter-end)
  t_observed  : timestamp    # when the fact became *knowable* (Nov 2 announcement)
  payload     : dict         # {"open":..., "close":...} or {"eps": 1.31, "restated": false}
```

Two timestamps — **bitemporality** — is the entire trick. Daily bars are events where `t_observed = bar_close (+ publication delay)`. Earnings are events where the two timestamps differ by weeks. Index membership changes are events (killing survivorship bias: the universe at $t$ is "membership events observed ≤ $t$"). News, ticks, and order flow are just… events. Restatements are *new events* about the same `t_effective` — never overwrites, because the log is append-only.

The only query the Store ever answers is the **as-of query**:

```python
store.asof(entities, kinds, t)   # → all events with t_observed <= t
```

Sit with the consequences:

- **Disguises 1–3 become structurally impossible** at the storage layer. There is no API that returns future knowledge, so features can't ask for it. (Disguises 4–5 live in the dataset layer; Ch. 5 handles them.)
- **Time series and event streams are unified.** A "time series" is just what you get when you *sample* the as-of view on a regular grid; an "event stream" is what you get when you don't. Regular and irregular data differ in a *view*, not in the foundation. This is why Loom handles both "cases" you asked about with one data model — and why qlib, whose foundation is the regular calendar grid (`CalendarProvider` is the first thing `qlib/data/data.py` defines), needs a separate universe of machinery for anything irregular.
- **Reproducibility is a query parameter.** "Rerun the backtest exactly as the data looked last March" is `asof(..., t_observed <= last_march)` — no snapshots, no data-versioning bolt-on.

The cost, honestly stated: as-of joins are more expensive than aligned-array lookups, and bar-shaped workloads pay overhead for generality they don't use. Loom's answer is the standard one from database engineering: **keep the event log as truth, and materialize derived views** (e.g., a nice aligned `(date × ticker × field)` Parquet cube for daily bars) that are cache, not truth. Views can be regenerated and audited against the log; truth is never edited.

```
event log (truth, append-only)
   ├── materialize → bar cube (dates × tickers × fields)   fast path, Ch. 5
   ├── materialize → PIT fundamentals (as-of latest per entity)
   └── raw as-of stream                                    event path, Ch. 10
```

## 2.3 Adjustments, calendars, and other sharp edges

- **Corporate actions.** A 7-for-1 split makes the price drop 86% overnight — not a return. Store *raw* prices plus split/dividend **events**; compute adjusted series in the view layer (adjustment factors are cumulative products of action events). Never store only adjusted prices: adjustment is a function of *when you ask* (a future split changes the whole adjusted history — a subtle bitemporal fact most vendors get wrong for you).
- **Calendars.** Trading days differ by venue; half-days exist; crypto never sleeps. A calendar in Loom is just the sorted set of `bar.1d` event times for a venue entity — derived, not axiomatic.
- **Missing ≠ zero ≠ halted.** A missing bar (no event) is different from a zero-volume bar (event, volume=0) is different from a halt (a halt event). The event model represents all three without sentinel values.
- **Currency & units.** Store native currency in payloads; convert in views with FX *events*. Unit bugs are leakage's boring cousin — wrong, but at least visibly wrong.

## 2.4 The temporal firewall: leakage as a failing test

A data model prevents leaks by construction only below its own API. Above it — feature code, dataset assembly — you defend with an *automated adversarial test*, which Loom treats as core infrastructure, not documentation:

**Firewall test 1 — truncation invariance.** For a random sample of times $t$: compute features as-of $t$ with the full log, and again with a log physically truncated at $t$ (all events with `t_observed > t` deleted). The vectors must be **bit-identical**. Any dependence on the future, however subtle — a pandas `resample` that peeked, a normalization fit too broadly — shows up as a diff. This one test catches disguises 1, 2, and 4-within-features mechanically.

**Firewall test 2 — the shuffled-future canary.** Replace all returns after each training cutoff with white noise; rerun your full pipeline. Every reported out-of-sample metric must collapse to zero. If *anything* still looks good, your pipeline is grading itself with the answer key. Run it once per pipeline change; it's a 20-line loop that has embarrassed every research stack I know of at least once.

```python
def test_truncation_invariance(store, features, sample_times, entities):
    for t in sample_times:
        full = features.compute(store, entities, t)
        trunc = features.compute(store.truncated(t), entities, t)
        assert full.equals(trunc), f"leak at {t}: {diff_report(full, trunc)}"
```

**⚖ Design call.** Firewall tests run in CI on every feature merged into Loom's registry. A feature isn't *in* the system until it survives truncation. Compare: in qlib, leakage safety rests on each expression author aligning `Ref()` offsets correctly, checked by eyeball.

## 2.5 Where the data comes from (pragmatics)

For the capstone you need *some* daily equity or crypto data. Free-tier reality: `yfinance` (survivorship-biased universe — acceptable for learning if you *say so* in the ledger), Stooq, or crypto exchange APIs (clean, complete, no survivorship problem, and a genuine event stream — a good reason Ch. 10–11 use crypto examples). The design point: **Loom's Store doesn't care.** Ingestion adapters turn any source into events; everything downstream is source-blind. qlib instead ships its own binary format and a data-conversion toolchain (`scripts/dump_bin.py`) that welds the system to one layout.

## 2.6 Build exercise (2–3 hours)

1. Implement a minimal event log on Parquet: one file per `kind`, columns `entity, t_effective, t_observed, payload_json`. Implement `asof(entities, kinds, t)` and `truncated(t)`.
2. Ingest ~50 tickers of daily OHLCV from any free source (set `t_observed = t_effective + 1 minute` — be honest about publication lag even when faking it).
3. Fabricate an "earnings" event stream with 30-day observation lags. Write the truncation-invariance test; then deliberately break a feature with lookahead (use `close[t+1]`) and watch the firewall catch it. *The failure is the lesson.*

**Next:** with truth on disk and a firewall overhead, we can look at how qlib arranges this same territory — and where its architecture fights itself.
