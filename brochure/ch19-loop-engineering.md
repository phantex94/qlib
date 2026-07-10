# Chapter 19 — Loop engineering: the internal and external project cycles

*Prerequisites: Ch. 4, 9, 12, 14, 17. Builds: the formalization of Loom's time dimension — four nested internal loops that turn measurements into parameter and structure changes, and an external loop that federates Loom with the projects around it (optimizer, risk, data, sibling alphas, and the human). With diagrams, the build process, and the operating runbook. Origin: Core Research Areas discussion 8.*

---

## 19.1 What loop engineering means here

Chapters 1–18 designed *components* — the space dimension. But a research system's value comes from its **cycles**: the paths by which a measurement becomes a change, and a change becomes a new measurement. Loop engineering is designing those cycles deliberately, with the same rigor we gave components. Every loop in Loom is an instance of one contract:

```python
@dataclass(frozen=True)
class LoopSpec:
    state:    ...   # what persists between iterations (weights, champion, blueprint)
    propose:  ...   # generates a candidate change
    evaluate: ...   # runs the experiment — ALWAYS via the Conductor (Ch. 4)
    gate:     ...   # accept/reject rule — ALWAYS a duel + significance bar
    commit:   ...   # ledger append + state update; nothing changes off-ledger
    cadence:  ...   # what wakes it: budget, schedule, event, or human
```

Two laws govern every instance, and they are the chapter's core:

- **Law 1 — All loops close through the ledger.** No feedback path may change state invisibly. If a loop learned something, there is a ledger entry saying what, from which experiment, at which gate. (This is what makes the loops *auditable* — and what makes human–agent collaboration possible at all: the ledger is the shared memory both parties read.)
- **Law 2 — Timescale separation.** Nested loops need frequency separation, exactly like nested control loops in control theory: each loop's cadence must be ≥ 5–10× slower than the loop inside it, or the system resonates — the outer loop reacts to the inner loop's transients instead of its equilibria, and you get oscillation (champions swapping weekly on noise, contracts churning under adapting models). Cadences below are chosen by this law, not convenience.

## 19.2 The Internal Project Loop: four nested cycles

"Build and optimize model parameters and organizational structures, given input conditions, for best out-of-sample generalization" decomposes into four loops at four timescales — and, satisfyingly, **each one guards one level of the Ch. 17 overfitting taxonomy**:

```
┌─ L3 · ADAPTATION LOOP ──────────────────────── cadence: calendar/monitor ─┐
│   guards Level R (regime overfitting)                                     │
│  ┌─ L2 · STRUCTURE LOOP ───────────────────── cadence: weeks ──────────┐  │
│  │   guards Level S (selection), structural half                       │  │
│  │  ┌─ L1 · SEARCH LOOP ─────────────────── cadence: hours–days ────┐  │  │
│  │  │   guards Level S (selection), parametric half                 │  │  │
│  │  │  ┌─ L0 · GRADIENT LOOP ───────────── cadence: ms–minutes ─┐   │  │  │
│  │  │  │   guards Level W (weight overfitting)                  │   │  │  │
│  │  │  │   state: weights · propose: gradient step              │   │  │  │
│  │  │  │   gate: early-stop on validation RankIC                │   │  │  │
│  │  │  │   commit: Artifact                                     │   │  │  │
│  │  │  └──────────────────────────────────────────────────────-┘   │  │  │
│  │  │   state: champion cfg + trial counters                        │  │  │
│  │  │   propose: hparam/feature/LLM engines (Ch. 9 §9.4)            │  │  │
│  │  │   gate: paired fingerprint duel + rising t-bar                │  │  │
│  │  │   commit: champion lineage update in ledger                   │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │   state: the blueprint's Tier-A list + system structure             │  │
│  │   propose: open questions A*.* (which mixer? which clock? cascade?) │  │
│  │   gate: duel + the injection test (Ch. 18 §18.1)                    │  │
│  │   commit: structure change + blueprint tier-status update           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│   state: deployed artifact + monitor baselines                            │
│   propose: scheduled retrain, or monitor alarm (IC chart, CUSUM, PSI)     │
│   gate: challenger beats incumbent on recent eras, paired                 │
│   commit: deployment swap + monitor re-baseline                           │
└───────────────────────────────────────────────────────────────────────────┘
```

**L0 — the gradient loop** (inside the Trainer, Ch. 6). Optimizes *parameters given a structure*. Its `propose` is the optimizer (AdamW/ASAM), its landscape behavior is Ch. 17's subject (SWA averages its tail; date-batches shape its noise), and its gate — early stopping on validation RankIC — is the first place generalization enters. Note it already obeys the LoopSpec contract; we just never drew it as a loop.

**L1 — the search loop** (Ch. 9 §9.4, now formalized). Optimizes *configurations given a structure*: hyperparameters, feature subsets, losses, weightings. Its proposal engines can be dumb (random/ASHA), smart (LLM reading the ledger), or human; all emit `cfg'` into the same Conductor and face the same gate. Its guardrails are Ch. 9 §9.6 — trial counters, peek budget — because L1 is where selection-level overfitting is manufactured at scale.

**L2 — the structure loop** (new name for what the blueprint's Tier A already is). Optimizes the **organizational structure** — your phrase, and the right one: not values *in* the config schema but the schema itself. Should the model have a cross-sectional mixer (A2.1)? Which clock (A3.1)? Cascade or blend (A8.1)? Add a combination rung (A3.2)? Each iteration picks the open question with the highest expected information value, runs its closing experiment, and commits a *structural* change plus a blueprint status update. Its extra gate is the injection test — a structure must bring information end-to-end training can't see, or it's deleted. L2 is deliberately slow (weeks): by Law 2 it must integrate over many L1 runs before judging a structure, and structures churned faster than evidence accumulates are fashion, not engineering.

**L3 — the adaptation loop** (Ch. 9 §9.3 + §9.5). Unlike L0–L2 it is *not* nested in a search — it's driven by the calendar and the monitors, because its enemy is time itself (Level R): scheduled walk-forward retrains, champion–challenger duels on the freshest eras, and alarm-triggered early retrains. Its design constraint is the detection-lag arithmetic from Ch. 1's exercise: at IC≈0.03, distinguishing "signal died" from noise takes months, so its bands are wide, its alarms rare, and its default action is the scheduled retrain, not the panic.

**Why the nesting is load-bearing:** each loop's *evaluate* is the loop inside it run to convergence. L1 evaluates a config by running L0. L2 evaluates a structure by running a *batch* of L1. L3 evaluates deployment-worthiness by consulting L1/L2's champion lineage on new data. One Conductor serves all four — the loops differ only in what they hold fixed and what they let vary. That is the whole internal mechanism, and it is small.

## 19.3 The External Project Loop: federation with the world

Loom does not live alone. Around it: an **optimizer/execution project** (Ch. 12's Portfolio grown into its own codebase), a **risk-model project**, a **data/ingestion project**, possibly **sibling alpha projects** (other Looms on other universes/clocks), and — always — the **human owner**. The external loop is how Loom's outputs flow out and how the world's feedback flows back in:

```
                              ┌────────────────────────────────────────────┐
                              │              THE MARKET                    │
                              └────────▲──────────────────────┬────────────┘
                                trades │                      │ fills, PnL,
                                       │                      │ realized costs
┌──────────────┐  factor loadings   ┌──┴──────────────────────▼───────────┐
│ RISK PROJECT │ ─── (events) ────► │      OPTIMIZER / EXECUTION PROJECT  │
└──────────────┘                    │  consumes signal events as-of;      │
                                    │  emits FEEDBACK EVENTS:             │
┌──────────────┐   raw events       │  realized weights (⇒ TC), costs,    │
│ DATA PROJECT │ ─── (events) ──┐   │  fills, waterfall attribution,      │
└──────────────┘                │   │  ExecSpec-change events             │
                                ▼   └──────▲──────────────────┬───────────┘
                        ┌───────────────┐  │ signal events    │ feedback events
   sibling alphas ────► │  EVENT LOG +  │──┘ (kind=signal.*,  │ (kind=exec.*,
   (signal events)      │    LEDGER     │◄── horizon, decay,  │  tagged with
                        │ the protocol  │    fingerprint,     │  signal run_ids)
                        └──▲─────────┬──┘    config hash)     │
                           │         │                        │
                 ┌─────────┴─────────▼────────────────────────▼──────────┐
                 │            LOOM  (L0–L3 internal loops)               │
                 └─────────▲─────────────────────────────────┬───────────┘
                    charter│revisions, Tier-C answers,       │ weekly report
                    holdout│pact, priorities                 │ (from ledger)
                        ┌──┴─────────────────────────────────▼──┐
                        │              HUMAN OWNER               │
                        └────────────────────────────────────────┘
```

**The protocol is the event log + ledger — not APIs into each other's internals.** This is Ch. 14's merge-vs-federate principle promoted to project scale: projects whose iteration clocks don't align must federate through typed, bitemporal, append-only data. Four contract rules:

1. **Everything crosses as events.** Signals out (`signal.*` with horizon, objective, decay metadata, fingerprint reference, config hash); feedback in (`exec.realized_weights`, `exec.costs`, `exec.fills`, `risk.loadings`, `execspec.changed`). Bitemporality means every consumer sees exactly what was knowable when — the backtest of a *collaboration* stays honest for free.
2. **Feedback must be attributable.** Every feedback event carries the run/signal IDs it responds to, so the internal loops can *learn from it*: realized weights close the TC measurement (A1.1 goes from simulated to real), realized costs recalibrate `ExecSpec`, fill-quality times the decay curves (Ch. 16), live IC feeds L3's monitors.
3. **Contract changes are events too.** When the optimizer project changes its constraint set, it emits `execspec.changed` — which matters enormously, because Ch. 12 taught Loom to *predict what the optimizer keeps*: a changed constraint basis silently invalidates residualized labels and neutralized losses. The event wakes L2 to re-run the affected Tier-A questions. Without this rule, the collaboration rots invisibly.
4. **The human is a loop participant with a defined interface**, per your original template's Step 5: inbound — Tier-C answers, charter revisions, the holdout pact, priority overrides; outbound — the weekly report (§19.5), generated *from the ledger* so it cannot drift from reality. Human–agent collaboration here isn't a chat; it's a slow, high-authority loop with the same contract shape as the others: the human's `gate` is sign-off, the human's `commit` is a charter revision.

**The stability problem external loops add** — worth naming because it bites real multi-project systems: Loom adapts to execution feedback while the optimizer adapts to Loom's signals — a co-adaptation dynamic that can oscillate (predator–prey style: signals chase costs, costs chase signals). Law 2 is the defense, applied across projects: **contract-level change must be the slowest cadence in the whole system** (quarterly-ish), model adaptation faster, gradient steps fastest — plus hysteresis at every gate (a challenger must win by the bar, not by ε, before a swap). Frequency separation is what makes a loop *of* loops converge instead of ring.

## 19.4 The build process (what you actually write, in order)

The loop machinery is deliberately small — it reuses the Conductor for all evaluation — with a budget of **~300 lines total**:

| Step | Build | ~Lines | Depends on |
|---|---|---|---|
| 1 | **L0**: already exists — the Trainer *is* it. Formalize: emit its gate decision (early-stop epoch, best val RankIC) into the run record | 10 | Ch. 6 |
| 2 | **Ledger upgrades**: champion-lineage table, trial counters per family, peek counter — the shared `state` of L1–L3 | 60 | Ch. 9 |
| 3 | **L1 driver**: `while budget: cfg' = propose(queue); r = conduct(cfg'); if gate(r, champion): commit(...)` — proposal queue seeded by hand, then by engines | 80 | 1–2 |
| 4 | **L3**: a cron entry + the three monitors (IC control chart with backtest-derived bands, CUSUM, feature PSI) + the challenger-duel retrain rule | 80 | 2 |
| 5 | **External interfaces**: signal-emission adapter (champion scores → `signal.*` events) and feedback-ingestion adapters (`exec.*`, `risk.*` → Store); the `execspec.changed` hook that enqueues affected Tier-A questions for L2 | 60 | Ch. 14 |
| 6 | **Human loop artifacts**: weekly-report generator reading the ledger (control-panel status, questions closed, lineage changes, next-three queue) | 40 | 2 |
| — | **L2** is *not code* — it is the blueprint document + you (or an agent) executing its Tier-A queue through the L1 driver. Resist automating it until the queue's priorities are stable; premature L2 automation is how second systems are born (Ch. 4 §4.4) | 0 | blueprint.md |

Build order matters: 2 before 3 (a search loop without trial counters is a self-deception engine — Ch. 9's most important sentence); 4 can parallel 3; 5 only when a real external consumer exists (don't build federation for an empty room); 6 from day one, because the human loop starts at iteration zero.

**Acceptance tests for the loops themselves** (they join the control panel): the L1 driver must pass the noise-feature rejection test *while running autonomously* (50 planted noise features, 50 rejections, unattended); L3's monitor must detect a synthetically-killed signal within its stated lag bound and *not* alarm on 5 years of healthy simulation; the `execspec.changed` hook must demonstrably enqueue A1.2's re-run when the constraint basis is perturbed.

## 19.5 How it operates: the runbook

A composite week once everything runs:

- **Nightly (L1):** the driver spends its experiment budget on the proposal queue — tens of runs, each memoized, gated, ledgered. Morning artifact: champion lineage diff (usually empty — that's health, not stagnation).
- **Continuous (external in):** feedback events land as they occur; TC and realized-cost series update; nothing *acts* on them directly — they accumulate as evidence for the slower loops. (Fast data, slow decisions: Law 2 again.)
- **Weekly (human loop):** the report generator emits: control-panel status, Tier-A questions closed with verdicts, lineage changes, the next three questions with expected information value, and anything migrating between blueprint tiers. The human answers what's theirs (Tier C, priorities, charter drift) — asynchronously, against the ledger, not against memory.
- **Monthly (L3):** scheduled walk-forward retrain; challenger duels incumbent on the freshest eras; swap only past the hysteresis bar; monitors re-baseline. Alarm-triggered versions of the same path can fire earlier, rarely, by design.
- **Quarterly (L2 + external contracts):** structure review — read the accumulated evidence, pick the next Tier-A structural question, run its closing experiment batch; process any `execspec.changed` re-runs; this is also the only cadence at which cross-project contracts may change.
- **Once (per charter):** the holdout is spent — outside every loop, pre-registered, exactly as Ch. 9's guardrail 5 demands. No loop touches it; that exclusion is enforced in the Store view, not in policy.

The signature property of a well-engineered loop system, visible in this runbook: **almost nothing happens on most days.** Loops that fire constantly are reacting to noise; the design pushes every action's cadence out to the timescale at which its evidence actually accumulates. Fast loops iterate; slow loops decide; the ledger remembers; the human steers.

## 19.6 New open questions (appended to the blueprint)

- **A9.1 — Loop cadence calibration.** Are the chosen cadences (nightly L1, monthly L3, quarterly L2) matched to information half-lives on our data? *Closes:* measure regret — how often did a slower/faster cadence, simulated on the ledger's history, dominate?
- **A9.2 — Proposal-engine value.** Rank L1's engines (random, ASHA, LLM-reader, human) by information gained per experiment spent. *Closes:* per-engine ledger attribution over a quarter.
- **B8 — Co-adaptation monitoring.** No mechanism yet *detects* signal↔execution oscillation (only Law 2 prevents it). *Trigger:* first sustained live feedback; then design a cross-correlation watch between signal turnover and realized-cost series.

## 19.7 Build exercise (3 hours, mostly assembly)

1. Write `LoopSpec` and re-express L0 and L1 as instances — the exercise is realizing you already built both; only the framing is new.
2. Implement the L1 driver (step 3 above) and run it unattended overnight on a 20-config queue including 5 noise features. The morning ledger must show the noise rejected and every trial counted.
3. Simulate the external loop locally: pipe your champion's scores through your own Ch. 12 optimizer, emit `exec.realized_weights` back into the Store, and compute *realized* TC from feedback events alone. You have now closed A1.1's loop end-to-end — on one machine, with one protocol, which is the whole point of §19.3.
4. Draw your own version of the §19.2 diagram from memory. If a loop's gate or state is fuzzy when you draw it, that loop will be fuzzy when it runs.

**Ledger meta-note (template Step 5):** discussion 8 didn't add capability — it added *time structure*: four internal cadences guarding the three overfitting levels, one external protocol carrying signals out and truth back in, and two laws (close through the ledger; separate the timescales) that keep a loop of loops from ringing. The system now has a shape in space (Ch. 4) and a shape in time (this chapter). Everything else is operation.
