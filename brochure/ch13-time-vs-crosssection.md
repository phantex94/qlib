# Chapter 13 — One tensor, two axes: merging time-series and cross-sectional prediction

*Prerequisites: Ch. 5–6, 9. Builds: the resolution of the TS-vs-XS dichotomy — an architecture that composes both, a statistical principle that sizes each part, and the system design that makes "separate or merged?" an empirical question instead of a religious one. Origin: Core Research Areas discussion 2.*

---

## 13.1 Dissolving the dichotomy: two factorizations of one tensor

Put the two "modes" side by side and look at what they actually are. The panel (Ch. 5) is one object: a tensor $X[d, i, t, f]$ — date, entity, trailing time-step, feature — with labels $y[d, i]$. Then:

- A **time-series model** (per-stock sequence model, weights shared across stocks — your "same model trained on every stock") processes $X[d, i, :, :]$ for each $(d, i)$ *independently*. It factorizes the problem **along the entity axis**: no term in it ever couples stock $i$ to stock $j$.
- A **cross-sectional model** (including graph models) processes $X[d, :, t{=}\text{now}, :]$ for each $d$ *jointly across entities*. It factorizes **along the time axis**: rich coupling across stocks, but each stock's history enters only as whatever summary features you precomputed.

So these are not two model families in need of a merger negotiation. They are **two factorization choices applied to the same tensor**, each discarding one axis's structure to afford the other. Your two observed weaknesses are exactly the two discarded terms: TS models can't see lead–lag (an entity-axis coupling); XS models can't see dynamics (a time-axis coupling). And the "merged architecture" is not a new invention — it's simply *declining to factorize*: a spatiotemporal model over $(N, T, F)$ per date. Other fields crossed this bridge years ago (video = space × time; traffic forecasting = graph × time, e.g. Graph WaveNet; weather); finance's versions (HIST, MASTER-style market transformers) are the same move.

**The question is therefore not "can we merge?" — composition is easy — but "how much of each axis's coupling can your data afford?"** That is a statistical question, and it has a sharp answer.

## 13.2 The statistical asymmetry (the load-bearing insight)

Count effective samples per axis, Ch. 1 §1.3(c) style:

- Parameters in a **shared temporal encoder** learn from every (date, entity) pair: ~2,500 days × ~3,000 stocks ≈ **7.5M samples** (correlated, but still an enormous pool). This is *why* your "one model for every stock" instinct works so well: weight sharing across entities is an inductive bias that multiplies effective data three-thousand-fold. It's the single best statistical deal in this domain.
- Parameters in a **cross-sectional mixer** learn a function of *whole dates*. Each date is **one sample** of the joint cross-sectional distribution. Effective pool: ~2,500 — and after regime correlation, far less. A big mixer doesn't learn "how stocks interact"; it memorizes *which dates looked how*.

So the merged model must be **asymmetric by design**:

> **The asymmetry principle.** Capacity that learns across entities × dates may be large (temporal encoder). Capacity that learns across dates only must be small, heavily regularized, and structurally constrained (cross-sectional mixer). Symmetric spatiotemporal architectures imported from video/weather — where both axes are data-rich — will overfit the date axis here, every time.

Corollaries: graph masks (sector, supply-chain) aren't just domain flavor — they are *parameter-count reduction* for the data-poor axis, exactly the regime where priors beat learning (Ch. 6 §6.6). And attention's permutation-equivariance over entities is mandatory, not stylistic: any architecture that assigns stocks fixed slots (concatenating N stocks into an MLP) learns tickers, not structure, and breaks the moment the universe changes.

## 13.3 The architecture: temporal encoder → gated cross-sectional mixer

```
one date d:   X[d]  ∈  (N, T, F)
                │  per-entity, weights shared across entities
                ▼
   TemporalEncoder (GRU / TCN, Ch. 6 rung 3)  →  h_i ∈ (N, d_model)   "each stock's story"
                │
                ├────────────────────────────► s_ts = head_ts(h)      TS score (N,)
                ▼
   XSecMixer: 1–2 attention blocks over the N entities,
              optionally masked by a relation graph      →  m ∈ (N, d_model)
                │
                ▼
   s = s_ts + g ⊙ head_xs(m)          g: learned gate, init ≈ 0
```

Design notes, each earning its place:

- **Time-then-space captures lead–lag correctly.** Lead–lag is "stock $i$'s *past* predicts stock $j$'s *future*." The mixer attends over **temporal embeddings** $h_i$ — summaries of each stock's history — so attention weight $a_{j \leftarrow i}$ composed with $h_i$ is precisely a learned lead–lag channel. A mixer over current-snapshot features (most naive XS models) structurally cannot represent this; a mixer over histories can. This one routing decision is the whole merge.
- **The gated residual makes the mixer *earn its existence*.** With $g$ initialized near zero, the model *is* the pure TS model at epoch 0 and adds cross-sectional corrections only where gradients justify them. Graceful degradation is built in: if your data has no exploitable inter-stock structure, training returns you the TS model, not a corrupted hybrid. (This is the architectural twin of Ch. 9's champion–challenger duel.)
- **Interleaving (time–space–time–space…) is the deluxe version** — MASTER-style alternating blocks. Per §13.2, resist it until a single mixer layer has proven marginal value on *your* ledger; each extra mixer layer spends date-axis capacity you probably don't have.
- **Feasibility, so you don't fear the joint batch:** one date at N=3,000, T=60, F=64 is ~46M floats ≈ 180 MB fp32 activations through the encoder, and N² attention at 3,000 entities is 9M pairs — one consumer GPU handles it. The `DateBatchSampler` (Ch. 5) already delivers date-complete batches because the IC loss needed them. Infrastructure decisions echoing again.
- **Degenerate configs come free:** N=1 (a futures or single-crypto book) → the mixer has nothing to attend over, the gate stays closed, and you have the Ch. 4 claim ("time-series forecasting is cross-sectional with universe of 1") realized in weights. Mixer-only (freeze a trivial encoder) recovers the pure XS model. One architecture, three modes, all `ModelSpec` fields.

## 13.4 The systems answer: reject the two-stack separation

Now the actual question you asked — qlib keeps the approaches separate; keep or redesign?

qlib's separation is not one decision but a **cascade through every layer**: tabular models use `DatasetH`, sequence models need `TSDatasetH`/`TSDataSampler` (a parallel dataset implementation), and graph models in `contrib` hand-roll their own data plumbing on top — each with its own private training loop (Ch. 3, P4). The damage is not duplication per se; it's that **TS-vs-XS becomes an unanswerable question**: comparing a GRU to a graph model means comparing different data paths, different preprocessing, different loop details — the confounds are baked into the architecture of the *system*, so no experiment can isolate the modeling choice.

**⚖ Design call — Loom's resolution:** *separate as views, merged as architecture, decided by the ledger.*

- **Views:** the packing table (Ch. 5 §5.4) gains its fourth row — joint: `(N, T, F)` per date — next to tabular, sequence, and event. Four `pack=` options on one `PanelBuilder`, one panel, one split policy, one normalization path. The separation survives exactly one layer deep, where it's a shape, not a stack.
- **Architecture:** the §13.3 composition lives in `models.py` as one module whose degenerate configs *are* the pure TS and pure XS models.
- **Decision:** because data path, loss, splits, and evaluation are identical across the three configs, "should we use TS, XS, or joint *for this universe and feature set*?" is three ledger entries and two paired fingerprint duels (Ch. 8–9) — an afternoon, run honestly. The answer will differ across datasets (dense liquid equity universes reward the mixer; sparse or tiny universes don't), which is precisely why it must be *cheap to re-ask*, not settled by architecture dogma.

That last point is the meta-answer to your question: **don't design the system to encode a verdict; design it so the verdict costs one config diff.** qlib's real failure here isn't choosing separation — it's making the comparison structurally impossible. A system where the TS/XS/joint choice is a `ModelSpec` field has already merged them in the only sense that matters.

## 13.5 Expectations, calibrated

What the mixer can plausibly buy, in decreasing order of documented reality: industry/peer relative-value effects (real, modest), lead–lag from large to small caps and along supply chains (real, decays as it gets arbitraged, needs the §13.3 routing to capture), crowding/common-ownership effects (real, hard), and "the market's full joint state" (mostly a date-memorization trap — §13.2). Published gains of joint models over strong per-stock baselines are consistently *modest* — worth having, not transformative. If your ledger shows the gate wide open and a huge joint-model win, your first hypothesis should be leakage through the entity axis (e.g., cross-sectional normalization computed with the label period included), not genius. The paranoia transfers to the new axis.

## 13.6 Build exercise (4 hours)

1. Add the `joint` packer: `(N, T, F)` date-batches with a universe mask. Test: permute the entity order within a batch and assert scores permute identically (equivariance — the entity-axis twin of Ch. 10's padding-invariance test).
2. Implement §13.3 exactly: your Ch. 6 GRU as encoder, one masked-attention mixer block, gated residual head. Three configs: gate frozen at 0 (pure TS), encoder replaced by a per-date linear (pure-ish XS), full joint.
3. Run the three-way duel through the standard walk-forward. Ledger the paired verdicts *and the learned gate magnitude* — the gate is a measurement of how much cross-sectional structure your data actually pays for.
4. Inject a synthetic lead–lag (make stock 0's return today equal stock 1's return yesterday plus noise) into simulated data. Verify: pure TS model can't capture it, joint model's attention finds it (inspect the attention row for stock 0). Seeing the mechanism work on a planted effect is worth ten papers.
5. Stretch: sector-mask the mixer and re-duel. Does the prior beat free attention at your data size? (§13.2 predicts: yes for short histories, converging as history grows.)

**Ledger meta-note (template Step 5):** discussion 2 migrated "TS vs XS" from a framework fork (qlib's answer) to a measured quantity — a gate value and a duel verdict per dataset. Dichotomies that survive in mature systems are the ones that became *parameters*.
