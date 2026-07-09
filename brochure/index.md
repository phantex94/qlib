# Loom — A Pedagogical Brochure
### From newbie to master: deep learning for financial return prediction, by building a system better than qlib

> **The premise.** You do not master a field by reading about it. You master it by building the machine that does it, breaking the machine, and understanding why it broke. This brochure tutors you by having us design and build **Loom** — a return-prediction research system that is *simpler* than qlib, *more elegant* in its contracts, *faster* at self-improvement, and *natively* handles both regular time series and irregular event streams.

---

## Part B — The charter (per your initialization template)

You gave me the four-quadrant template and "the vibe" instead of a filled Part A. Per your own protocol, gaps become typed claims. Here is Step 1–4, executed.

### Step 1 — Restatement with typed claims

- **[FACT]** Goal: a multi-chapter markdown brochure (`index.md` + chapters) that tutors an undergrad from newbie to master over several weeks, by designing/building a qlib-like-but-better system.
- **[FACT]** The four questions to answer: (1) orchestration & data flow, (2) component responsibilities, (3) building your own model — architecture, loss design, optimizer customization, (4) building a self-improvement loop.
- **[FACT]** Judged criteria: simpler & more elegant design than qlib; faster self-improvement; flexibility across time-series *and* event-stream cases.
- **[ASSUMPTION 1]** "Undergrad" means: comfortable with Python, NumPy, basic PyTorch, and intro statistics; *not* assumed: measure theory, market microstructure, production ML. Every chapter states its prerequisites.
- **[ASSUMPTION 2]** Target asset class for concreteness: cross-sectional equities (daily bars), because that is qlib's home turf and the richest pedagogy. Event-stream methods (Ch. 10) generalize to crypto ticks, news, and order flow.
- **[ASSUMPTION 3]** "Master in several weeks" means mastery of *system design and research methodology*, not profitable live trading. The brochure is a curriculum + reference design, with code you write yourself in the capstone (Ch. 11).
- **[ASSUMPTION 4]** Deliverable is the brochure itself; Loom's full implementation is *your* capstone work, with skeletons and acceptance tests provided. (Tutoring means you type the code.)
- **[INFERENCE]** Since you called qlib "malfunctioning in design patterns" and want elegance, the critique chapter (Ch. 3) must be *evidence-based* — real file/line citations — or it's just vibes replacing vibes. Done.
- **[OPEN]** See Step 2. Answer these whenever; the chapters are written so any answer refines rather than invalidates them.

### Step 2 — Q3 extraction: five questions (answer async, they steer revisions)

1. **Rejection probe:** A brochure that explains everything beautifully but whose Loom design you'd never actually build — what would make it feel that way? (Too abstract? Too much code? Not enough math?)
2. **Example probe:** Besides qlib, what's the closest thing to what you want — a course (fast.ai?), a book (López de Prado?), a codebase? What would you change about *it*?
3. **Priority collision:** If "simpler" and "handles event streams natively" conflict (they will — Ch. 10 explains why), which wins?
4. **Audience probe:** Who else sees this — a professor, a club, an employer? First thing they'd poke at?
5. **Silence probe:** You said nothing about *live data / paper trading*. Deliberate scope-out, or overlooked? (I assumed scope-out: [ASSUMPTION 3].)

### Step 3 — Q4 risk register

| Risk | Early symptom | Defense |
|---|---|---|
| **You overfit the research loop itself** — the #1 killer in this field | Backtest Sharpe climbs every week while live/holdout IC doesn't | Ch. 9's guardrails: locked holdout eras, peek budget, deflated metrics. *Mitigate now — it's designed in.* |
| **Data leakage** silently inflates every result | "Too good" metrics; model loves features with subtle lookahead | Ch. 2's temporal firewall test — an automated leakage detector, not a checklist. *Mitigate now.* |
| **Second-system effect** — Loom becomes a grander qlib | Weeks of framework code, zero experiments run | Hard budget: Loom core ≤ ~1,500 lines; Ch. 4's "conductor fits on one page" rule. *Accept & monitor.* |
| **Curriculum stall** — chapters read, nothing built | Week 3 with no ledger entries | Every chapter ends with a *build exercise*; Ch. 11 has weekly gates. *Needs your decision to honor them.* |
| **Compute/data access** — no good free daily data | Can't reproduce numbers | Chapters use synthetic + free data (yfinance/crypto) patterns; Loom's Store is source-agnostic. *Accept & monitor.* |

### Step 4 — Charter (proposed frozen; flag revisions explicitly)

```yaml
charter:
  goal: Tutor from newbie to master via designing Loom — a return-prediction
        research system simpler, more elegant, and faster-improving than qlib,
        unified over time series and event streams.
  acceptance_criteria:
    must:  [answers all four questions, evidence-based qlib critique,
            runnable-quality code sketches, leakage & loop-overfit defenses built in,
            event streams and time series share one data model]
    should: [6-week schedule with gates, mastery checklist, further-reading map]
    out_of_scope: [live trading, broker integration, profitable-alpha guarantees]
  roadmap: chapters 1-11 below; capstone = you build Loom with provided skeletons
  assumptions_in_force: [1, 2, 3, 4]
  change_rule: your answers to Step-2 questions trigger explicit chapter revisions
```

---

## The chapters

**Part I — First principles** *(Week 1)*
- [Chapter 1 — What return prediction actually is](ch01-first-principles.md) · The fundamental law, why IC=0.03 is a business, cross-section vs. time series, why finance breaks standard ML intuition.
- [Chapter 2 — Data: the event log and the temporal firewall](ch02-data.md) · Point-in-time truth, the five leaks, and the one data model that unifies bars and events.

**Part II — Systems** *(Week 2)*
- [Chapter 3 — Anatomy of qlib: an autopsy](ch03-anatomy-of-qlib.md) · How it orchestrates, how data flows through it, and six design pathologies with file:line evidence — plus the four ideas worth stealing.
- [Chapter 4 — Designing Loom](ch04-loom-design.md) · Seven components, their exact responsibilities and contracts, the one-page conductor, and the dataflow diagram. **This answers your questions 1 & 2.**

**Part III — The modeling core** *(Weeks 3–4)*
- [Chapter 5 — Panels, splits, and samplers](ch05-datasets.md) · Purged walk-forward, embargo, cross-sectional normalization, tensor shapes for tabular/sequence/event batches.
- [Chapter 6 — Building your own model](ch06-models.md) · The model contract, the shared Trainer, and a capability ladder: ridge → LightGBM → MLP → GRU/TCN → cross-sectional Transformer. **Question 3, part 1.**
- [Chapter 7 — Losses and optimizers from first principles](ch07-losses-optimizers.md) · Why MSE is the wrong prior, differentiable IC, ranking losses, Sharpe surrogates; LR schedules, custom optimizers (SAM), parameter groups. **Question 3, part 2.**
- [Chapter 8 — Evaluation and backtesting](ch08-evaluation-backtest.md) · The metric hierarchy, IC decay, turnover & costs, and the statistics that keep you honest.

**Part IV — Mastery** *(Weeks 5–6)*
- [Chapter 9 — The self-improvement loop](ch09-self-improvement.md) · The ledger, walk-forward retraining, automated search, drift monitors — and the guardrails that stop the loop from eating itself. **Question 4.**
- [Chapter 10 — Event streams as first-class citizens](ch10-event-streams.md) · Irregular time, three modeling strategies, time-aware architectures, and why Loom gets this almost for free.
- [Chapter 11 — Capstone: build Loom](ch11-capstone.md) · File-by-file skeleton, weekly gates with acceptance tests, the mastery checklist, and the reading map.

**Part V — Research frontiers** *(chapters grown from our Core Research Areas discussions; read after Part III)*
- [Chapter 12 — Alignment: surviving the optimizer](ch12-alignment.md) · The transfer coefficient, the signal-attrition waterfall, liquidity/risk/cost constraints placed at three architectural depths — and the defenses against constraint-set overfitting. *(Discussion 1.)*
- [Chapter 13 — One tensor, two axes: merging time-series and cross-sectional prediction](ch13-time-vs-crosssection.md) · The TS/XS dichotomy dissolved into factorization choices, the asymmetry principle for sizing each axis's capacity, the gated encoder–mixer architecture, and why the choice should cost one config diff. *(Discussion 2.)*
- [Chapter 14 — Market time: alternative bars and the combination layer](ch14-clocks-and-combination.md) · Volume/dollar bars and event anchors unified as clocks (folds over the event log, ~50 lines), the broken cross-section, predictions-as-events, and the five-rung signal-combination ladder with its capacity budget. *(Discussion 3.)*
- [Chapter 15 — Inductive bias, diverse ensembles, and the Mixture-of-Experts question](ch15-inductive-bias-moe.md) · What each model family refuses to learn, the ambiguity decomposition, targeted-direction training via information diets and residual cascades — and the MoE transplant: three breaks, three designs, a humble router, and the noise-expert canary. *(Discussion 4.)*
- [Chapter 16 — Frequency: horizon structure in features, models, and Multi-Period Optimization](ch16-frequency-mpo.md) · The feature×horizon IC matrix, the five-way alignment principle, an honest verdict on spectral decomposition (with the Slutsky–Yule control), and the MPO ladder from horizon matching through Gârleanu–Pedersen aim portfolios to convex MPC. *(Discussion 5.)*
- [Chapter 17 — The unified theory of not fooling yourself: overfitting at three levels](ch17-anti-overfitting.md) · Weight-, selection-, and regime-level overfitting with the finance-reversed danger ordering, the shifted-surface case for flat minima, SWA/ASAM verdicts vs. natural gradient, the assembled negative-control panel, and the ledger-as-dataset meta-experiment. *(Discussion 6 — the synthesis.)*

---

## How to use this brochure

1. **Read a chapter, then do its build exercise before the next.** The exercises compound; skipping one makes the next chapter abstract.
2. **Keep a ledger from day one** (Ch. 9 shows the schema; start with a plain CSV). Mastery is measurable as ledger entries: experiments run, hypotheses killed.
3. **Argue with the design.** Loom is *a* good answer, not *the* answer. Every chapter marks its contestable decisions with **⚖ Design call** blocks — disagreeing with them, with reasons, is the curriculum working.

*Estimated effort: 6 weeks at ~10 focused hours/week. Prerequisites: Python, NumPy, basic PyTorch, one statistics course.*
