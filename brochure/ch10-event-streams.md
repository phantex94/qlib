# Chapter 10 — Event streams as first-class citizens

*Prerequisites: Ch. 2, 5, 6. Builds: the second half of your flexibility requirement — irregular data (news, earnings, trades, order flow) handled by the same system, not a bolted-on annex. This is where Loom's Ch. 2 bet pays out.*

---

## 10.1 Why irregular data breaks grid-native systems

A calendar-grid system (qlib's foundation — Ch. 3, P6) answers one question fast: *"value of field f for entity i at bar t."* Events ask different questions: *"what happened to i in the last 6 hours?"*, *"how long since the last earnings surprise?"*, *"what does this burst of trades imply?"*. Forcing events onto a grid means either aggregating early (destroying timing information — often the *only* alpha in an event is its timing) or exploding the grid to the finest event resolution (a sparse-matrix catastrophe).

Loom doesn't have this dilemma, by construction: the Store *is* an event log; the daily bar cube was always a materialized view (Ch. 2 §2.2). Time series and event streams are two *sampling policies* over one substrate:

```
                    ┌── sample on calendar grid ──► panel (Ch. 5)   "time-series case"
event log ── asof ──┤
                    └── sample at event times / windows ──► event set   "event-stream case"
```

Which means the event-stream case needs exactly two new pieces — event-aware *features* (②) and event-aware *packing* (③) — while Store, Trainer, Evaluator, Ledger, Conductor are untouched. Compare the surface area of adding, say, a news-flow model to a grid-native stack. This asymmetry is what "flexibility from the data model" means concretely.

## 10.2 Strategy A: featurize events onto the grid (do this first)

Keep the daily prediction problem; summarize each entity's recent events into grid-aligned features:

```python
@feature(lookback="90d", kinds=["earnings"])
def days_since_earnings(view: EventView, t):   # recency
    return (t - view.last_event_time()).days

@feature(lookback="30d", kinds=["news"])
def news_burst(view, t):                       # intensity vs. own baseline
    return view.count("3d") / (view.count("30d") / 10 + 1)

@feature(lookback="1d", kinds=["trade"])
def signed_flow(view, t):                      # content aggregation
    tr = view.events("1d")
    return (tr.sign * tr.size).sum() / tr.size.sum()
```

The three universal event-feature verbs: **recency** (time since), **intensity** (counts vs. baseline — bursts predict volatility), **content** (aggregated payloads — sentiment sums, signed volume). This strategy captures a large share of event alpha at zero architectural cost, and it's firewall-tested like any feature. *Exhaust it before Strategy C.* Its ceiling: within-window ordering and inter-event timing structure are averaged away.

**Post-event drift, a label subtlety:** for event studies ("do earnings surprises drift?"), the natural row is not (date, entity) but **(event, entity)** — sample the panel *at event times + lag*. PanelBuilder gains an alternative sampler (`sample_at=events(kind="earnings", lag="1d")`), ~15 lines; everything downstream (Trainer, Evaluator) is indifferent to where rows came from. Note what just happened: an *event-driven research design* — a different scientific question — reused the whole stack.

## 10.3 Strategy B: sequences with time as an input

Give a sequence model (Ch. 6, rung 3) the events themselves, with **Δt as a feature** — the model learns what timing means instead of you hand-coding it:

```
sample for (entity, t):  last K events before t (per as-of, of course)
  each event → [ embed(kind) ‖ payload features ‖ φ(Δt_since_prev) ‖ φ(t − t_event) ]
  → GRU / causal Transformer → score
```

- **Time encodings φ:** log-bucketed Δt one-hots (robust default), or sinusoidal time2vec embeddings. Log matters: the difference between 1s and 10s is information; between 30d and 30d+9s, none.
- **Time-decay recurrence:** decay the hidden state by learned $e^{-\lambda \Delta t}$ between events — the model forgets at a learned rate; the cleanest inductive bias for "old events matter less."
- **Packing (the real work):** ragged sequences → pad to K, carry a mask; the collate function and mask handling are 90% of the implementation effort and 100% of the subtle bugs. Test the mask by asserting that padding events can be permuted/zeroed with *bit-identical* model output — the firewall mentality, applied to architecture.

Strategy B is the right tool for order-flow, tick, and news-sequence problems where *pattern across events* — not just their summary — carries the signal.

## 10.4 Strategy C: point processes — when the timing IS the target

Strategies A–B use events to predict returns. Sometimes you must model the event process itself — arrival intensity of trades (liquidity/volatility forecasting), cascade risk, "will a follow-on event occur?".

The Hawkes intuition in one line: events *excite* future events — intensity jumps at each arrival and decays — $\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}$ — which captures the empirical signature of markets: clustering (of trades, of volatility, of news). Neural temporal point processes replace the kernel with a learned network. The log-likelihood ($\sum_i \log \lambda(t_i) - \int \lambda$) is just another 20-line entry in the `LOSSES` registry; the Trainer doesn't care that the "label" is now the event stream itself.

**⚖ Design call — honoring your priority-collision question (index, Step 2 #3).** Full TPP machinery is genuinely complex, and "simpler" vs. "event-native" collide right here. Loom's resolution: Strategies A and B are *core* (they reuse everything and cover most practical needs); Strategy C is a *documented extension* with the loss-registry entry point sketched, built only when a problem demands it. Simplicity wins ties; the data model keeps the door open. If your Step-2 answer flips this priority, this section is the one that grows.

## 10.5 Evaluation quirks for event-driven predictions

Ch. 8's machinery mostly transfers — with three traps flagged:

- **No natural cross-section?** Event-time predictions may have few simultaneous peers; IC-per-day breaks. Fall back to *pooled-within-era* rank correlation, era = week/month (statistical honesty via eras survives; the daily cross-section was a convenience, not a principle).
- **Clustered outcomes:** events cluster in time (that was §10.4's whole point), so outcome overlap between nearby events is severe — purging (Ch. 5) by *label-window overlap between event rows*, not merely by dates.
- **Selection into events:** entities *choose* some events (announcements) — conditioning on the event is conditioning on a choice; comparing event-rows to non-event baselines needs care (matched controls beat naive pooled comparisons). One honest paragraph in the ledger note beats a spurious 0.08 IC.

## 10.6 Build exercise (4 hours)

1. Strategy A end-to-end: add your Ch. 2 fake-earnings stream (or real crypto trades) as three grid features (recency / intensity / content) to your champion model. Ledger the duel vs. the champion (Ch. 9 discipline — event features face the same SELECT bar as everything else).
2. Implement the `sample_at=events(...)` panel sampler; run the post-event-drift study on your data. Even a null result is a completed, honest event study — few undergrads have ever produced one.
3. Strategy B skeleton: ragged collate + mask + Δt log-buckets + GRU, on synthetic Hawkes data (simulate with the thinning algorithm — 25 lines, and simulating it teaches the intensity concept better than reading about it). Verify the model learns to predict higher intensity right after event bursts.
4. Write the padding-invariance test from §10.3. Watch it catch your first mask bug (it will).

**Next and last:** the capstone — assembling everything you've built into Loom v1, the weekly gates, and the mastery checklist.
