# Chapter 5 — Panels, splits, and samplers

*Prerequisites: Ch. 2, 4. Builds: PanelBuilder (component ③) — where most silent research errors are either committed or prevented.*

---

## 5.1 The panel

Loom's canonical learning object is the **panel**: a 3-tensor `X[date, entity, feature]` with labels `y[date, entity]` and `meta` (universe membership masks, sector tags, listing status). Concretely: a long dataframe indexed by `(date, entity)`, pivoted on demand.

Assembly, per date $t$ in the requested range:
1. **Universe as-of $t$** from membership events (Ch. 2 — survivorship dies here).
2. **Features as-of $t$**: each FeatureLib function gets a view window `[t − lookback, t]`.
3. **Label**: forward return over `(t + exec_lag, t + exec_lag + horizon]` — note the label reads the *future* deliberately and *declaredly*; it exists only for rows whose future has fully arrived (the last `horizon` days of any panel have features but `y = NaN` — those are your live-scoring rows, not training rows).

**Missing data policy** is declared per feature, not improvised: `ffill(limit=5)` for slow fundamentals, `NaN → cross-sectional median` for fast prices, plus an explicit `is_missing` indicator column when missingness is informative (it often is — a stock with no volume *is* information).

## 5.2 Cross-sectional normalization (and its leakage traps)

Per Ch. 1, the game is relative. So both features and labels are usually normalized *within each date*:

- **Rank-normalize** to $[-0.5, 0.5]$ (robust, kills outliers, loses magnitude), or **z-score with winsorization** at ±3 MAD (keeps magnitude, needs care with fat tails).
- Cross-sectional (within-date) statistics are **leak-safe by construction** — they only use same-instant data. This is the normalization you should reach for first.
- **Time-series normalizations** (rolling z-scores of a feature against its own history) are fine *if* the window is trailing — the firewall test catches the classic error (centering with `rolling(center=True)` or fitting a scaler over the full history).
- **Label normalization is training-only.** Train on ranked labels; evaluate on *raw* returns. Two different function compositions of the same panel — this is the honest core of qlib's DK_L/DK_I distinction, in two lines instead of a stateful handler:

```python
train_view = panel.pipe(xsec_rank_features).pipe(xsec_rank_label)
infer_view = panel.pipe(xsec_rank_features)          # label untouched
```

## 5.3 Splits that don't lie: purged walk-forward with embargo

Random K-fold is fraud here (Ch. 2, disguise 5). The honest scheme:

```
time ──────────────────────────────────────────────────────────►
[══ train ══════════][purge][─ valid ─][embargo][── test ──]
        then roll the whole window forward and repeat:
          [══ train ══════════][purge][─ valid ─][embargo][── test ──]
```

- **Walk-forward:** every test range is strictly after its train range; you always simulate "trained on the past, deployed on the future." Multiple folds = multiple eras of evidence, which Ch. 8's statistics need.
- **Purge:** drop `horizon + exec_lag` days between train and anything after it. A 5-day label at the train boundary *overlaps* the valid period — those train rows contain valid-period returns. Purging removes exactly the overlap.
- **Embargo:** drop an extra buffer (~1–2% of history) after each test block when folds are reused across many experiments — serial correlation bleeds information backward; the embargo is cheap insurance (López de Prado's argument).

```python
@dataclass(frozen=True)
class WalkForward:
    train_len: str = "3y"; valid_len: str = "6m"; test_len: str = "6m"
    step: str = "6m"; purge: str = "7d"; embargo: str = "10d"

def walk_forward(dates, cfg) -> list[Split]:   # ~20 lines, write it yourself
    ...
```

A `Split` is a frozen value logged in the ledger with the run: which days trained which model is forever answerable. **Retrain-per-fold is non-negotiable** — one model evaluated across all folds is a subtle but real cheat (later folds' *hyperparameters* were chosen while looking at earlier test eras… see Ch. 9's peek budget for the full paranoia).

Reserve a **final holdout** — the last 1–2 years — which no experiment touches until the capstone's end. That's Ch. 9's "locked era" guardrail, established *now*, because it only works if you set it before you start iterating.

## 5.4 Tensors for three model families

PanelBuilder's last job: packing. One panel, three shapes:

| Family | Shape per sample | Built by | Used in |
|---|---|---|---|
| Tabular (ridge, GBDT, MLP) | `X: (F,)` | flatten `(date, entity)` rows | Ch. 6 §2–3 |
| Sequence (GRU, TCN, Transformer) | `X: (T, F)` — trailing T days of features for one entity | sliding window per entity; sample id is still `(date, entity)` | Ch. 6 §4–5 |
| Event (Ch. 10) | ragged `[(Δt, kind, payload), …]` | as-of event pulls, padded + masked | Ch. 10 |

The critical subtlety is **batching for cross-sectional losses**. IC loss and ranking losses (Ch. 7) compare entities *within the same date* — a batch must therefore be **one or more complete dates**, never a random shuffle of rows:

```python
class DateBatchSampler:
    """Yields all (date, entity) rows of each date together; shuffles dates, not rows."""
    def __iter__(self):
        for d in self.rng.permutation(self.dates):
            yield self.rows_of[d]
```

This one sampler class is why Loom can offer ranking losses system-wide while qlib's per-model loops (P4) mostly can't: the *sampler* is a Trainer concern, chosen by the LossSpec, not re-implemented per model.

## 5.5 Weighting and curriculum (quiet levers, big effects)

- **Recency weighting:** exponential decay of sample weight with age (half-life 1–2 years) is the simplest non-stationarity hedge and routinely beats fancy adaptation methods. It's a `weight` column in the panel — the Trainer multiplies it into any loss.
- **Liquidity weighting / filtering:** train on tradeable names (top-N by dollar volume as-of each date) or weight by √(dollar volume); a model brilliant on micro-caps is a backtest ornament.
- **Era balancing:** if one crisis month contributes 10× the loss variance, per-date loss normalization (which IC loss does implicitly — Ch. 7) stops 2020-03 from being 40% of your gradient.

## 5.6 Build exercise (3 hours)

1. Implement `build_panel(store, features, label, universe, dates)` against your Ch. 2 store: long dataframe, `(date, entity)` index.
2. Implement `walk_forward()` and *unit-test the purge*: construct a synthetic panel with a 5-day label, assert no train row's label window intersects any valid/test date. Do it by brute force over rows — the test should be dumb and airtight.
3. Implement `DateBatchSampler` and verify each batch is date-complete.
4. Run firewall test 2 (shuffled-future canary, Ch. 2) through the whole panel pipeline: white-noise labels → every split's IC statistically zero. Frame the passing output. You now have a pipeline you can *trust*, which puts you ahead of most practitioners.

**Next:** models — finally. And because the pipeline is trustworthy, every number they produce will mean something.
