# VERDICT — global rr=10/atr=3.0 + trail + 21d shadow halt

Adjudicated against `PROTOCOL_symbols.md`, `variations.csv`/`variations_output.txt`,
`compare_startdates.csv`/`compare_output.txt`, `criteria_output.txt`,
`agent_out/red2/RED_TEAM2.md`, `NULL_TEST.md`, `MARKET_CONTROLS.md`,
`agent_out/REPLICATION2.md`, `agent_out/cmp/RESULTS.md`, and direct re-computation from
`universe_trail.parquet`, `uni_glob_rr10_am3_same.parquet`, `grid_inuni.parquet`.
Every disputed number below was re-derived, not taken on either debater's word.

---

## Dispute 1 — what is the right comparator?

**Ruling: PRO is right about which baseline to score the motion against; CON is right
that this doesn't make OLD-BASE irrelevant.**

The four-way cell that neither brief lays out side by side (`compare_output.txt`, FULL RUN,
$1,500 start):

| | fixed TP | trail |
|---|---|---|
| **fitted (728 configs)** | OLD-BASE $132,605, DD 70.2%, holdout −8.4% | CURRENT $8,587, DD 64.9%, holdout −43.4% |
| **global rr10/atr3** | GLOBAL-FIX $1,334, DD 75.7%, holdout −4.5% | GLOBAL $20,775, DD 47.5%, holdout +8.3% |

The trail is not independently good or bad — it **interacts with the sign of the parameter
change**: it collapses fitted-params equity by 15x (132,605→8,587) and *lifts* global-params
equity by 15.6x (1,334→20,775). Because the motion bundles "swap params" with "keep trail,"
and trail is already deployed in production (CLAUDE.md §6.1, live since 2026-08-02), the
correct "what happens if we do nothing" baseline for an up/down vote on this exact motion is
**CURRENT** — that is what config.yaml runs today. Scored against CURRENT, GLOBAL wins on
every axis in every one of `variations_output.txt`'s 12 toggle scenarios except raw dollars
in a couple of stress cells.

But CON is correct that "off-motion" is not the same as "irrelevant." Reverting the trail
(OLD-BASE) is a separate, reversible, one-line action that is *not mutually exclusive* with
adopting HALT 21d, and it beats GLOBAL 8/8 on restart-date equity by a wide margin
(`compare_startdates.csv`). The honest framing is: there are three live options on the
table, not two — stay on CURRENT, revert to OLD-BASE, or adopt GLOBAL — and the debate's
binary "GLOBAL vs. its baseline" framing understates that OLD-BASE is a genuine competitor,
not a straw man. Settled below (Dispute 2) on which of the two non-CURRENT options is more
trustworthy going forward.

**Score: PRO wins the procedural question (CURRENT is the right baseline to grade the
motion). CON wins the substantive point (OLD-BASE deserves a seat at the table as an
alternative, not a dismissal).**

---

## Dispute 2 — is OLD-BASE actually best? Outlier-robustness, computed directly

Per the assignment: `net_R = r − 0.00242/stop_frac`, computed flat (no compounding, no
position sizing) on `universe_trail.parquet.r_fixed` (OLD-BASE) and
`uni_glob_rr10_am3_same.parquet.r_result` (GLOBAL).

| removed | OLD-BASE remaining mean R | OLD-BASE % of total profit removed | GLOBAL remaining mean R | GLOBAL % of total profit removed |
|---|---|---|---|---|
| top 0.5% | +0.197 | 20.3% | +0.144 | 25.2% |
| top 1% | +0.148 | 40.4% | +0.112 | 42.1% |
| top 2% | +0.049 | 80.7% | +0.062 | 68.3% |
| top 5% | **−0.260** | 200.6% | **−0.061** | 130.2% |

Both arms stay solidly positive through 2% removed — **neither is a fragile arm on this
specific R-based test.** They diverge at the 5% tail: OLD-BASE's remaining edge collapses
much further negative (−0.260R/trade, and 200.6% of its lifetime profit sits in its own top
5%) than GLOBAL's (−0.061R/trade, 130.2%). **GLOBAL degrades more gently and is the more
outlier-robust of the two by this measure.**

This is a different (and looser) test than the one already run by the study itself on the
actual A0/A1 pair (`criteria_output.txt` Criterion 5, `NULL_TEST.md`): there, **A0 = today's
live config.yaml, i.e. fitted-*with-trail*** (not OLD-BASE's fixed-TP variant), and it is far
more fragile — its per-trade net R goes **negative** after just 1% removed (+0.062 → −0.016),
and 125.9% of its total net R sits in its top 1% by one denominator (NULL_TEST.md's
different denominator — % of *gross positive* R — gives 8.8%; both are internally consistent
computations, just answering different questions, see Dispute 4-adjacent note below). GLOBAL
(A1) stays positive through 1% removed (+0.191 → +0.111) and only turns negative under the
"hostile" A1-only-trimmed test at 2%.

**Conclusion: the fragile arm in this study is not "fitted params" per se — it's fitted
params *combined with the trail*, i.e. exactly CURRENT, the arm nobody is defending. Between
OLD-BASE and GLOBAL specifically, GLOBAL is modestly more robust to losing its best trades,
not less, contradicting the intuitive reading of the top1%R figures in `variations_output.txt`
(155.8% vs 150.6%, dollar/compounded, which read as roughly a wash and should not be over-read
as an OLD-BASE advantage).**

**Score: neither debater ran this specific test; PRO's rebuttal-1 framing ("OLD-BASE's dollars
depend on not missing the jackpot trade") is directionally supported once you compare it to
GLOBAL specifically, not just cited from the CLAUDE.md concentration figure.**

---

## Dispute 3 — the selection-contamination charge

**Ruling: verified true for the specific statistic named, so CON's blanket charge is
overstated there — but CON's separate A6 finding survives untouched.**

Checked directly: `grid_inuni.parquet` has 420,953 rows, 1,100 unique `(symbol, div_type)`
pairs across **275 symbols × all 4 divergence types** — i.e. every live symbol, every
div type, not just the winners. Cross-checked against `config.yaml`: it selected 728
`(symbol, div_type)` pairs; **724 of those 728 are present inside grid_inuni's 1,100**
(4 missing are dated futures contracts absent from the cache — an edge case, not a coverage
gap), and the other **376 pairs in grid_inuni are exactly the ones the walk-forward
rejected**. `REPORT_global.html`'s Spearman section explicitly states its rho figures
(fitted method ρ=0.064 vs global-param ρ=0.619) were computed "across four anchored
walk-forward folds on the full 420,953-row parameter grid" — that row count matches
grid_inuni.parquet exactly. **So the headline transfer statistic was honestly computed on
the unselected population, not laundered through the 728 winners.** CON's framing ("everything
runs on the walk-forward's in-sample-selected pairs") is false as stated for this statistic.

That said, CON's *other* number (RED_TEAM2 A6), also drawn from this same honest grid, stands
unrebutted: GLOBAL's net R/trade is **+0.1235 on the selected 728 pairs vs +0.0311 on the full
186,834-trade unselected population — a real 4x shrinkage**, and **−0.1776 on just the 376
rejected pairs**. So: selection inflates the magnitude of GLOBAL's edge substantially, but
does not zero it out or flip its sign on the full, unselected universe. That is a genuinely
reassuring floor, not a debunking, but it is also not nothing — a quarter of the selected-pair
edge remaining is still a real caveat on how much of GLOBAL's headline number is "the global
parameter is good" versus "the same in-sample symbol-picking that broke the fitted arm is also
propping up the global arm's headline figure."

**Score: PRO wins on the letter of the charge (the transfer statistic is clean); CON wins on
the substance (A6's magnitude-shrinkage finding is real, unrebutted, and the A2/A4/A5 controls
that would fully resolve it were genuinely never run — the protocol's own DEVIATIONS section
is empty, which is a real process failure regardless of how the numbers land).**

---

## Dispute 4 — the 21-day halt: unambiguous win?

**Ruling: real in this study's holdout, but weaker than either side implies once checked
against the study's own halt-specific caveats and prior repo history — not the clean
unambiguous win it's being treated as.**

Confirmed from `variations.csv`/`variations_output.txt`: HALT 21d improves holdout% for
all four configs (OLD-BASE −12.4→−5.4, CURRENT −48.4→−12.6, GLOBAL +7.9→+15.8,
GLB-10/2 −15.5→+23.4) with DD essentially flat (median −0.0pt) — that part of both debaters'
claims is accurate. But two things materially discount it:

1. **`variations.py`'s own docstring says the backtest models an optimistic best case**:
   "Live this is ADVISORY... Modelling it as a hard filter is therefore the OPTIMISTIC
   reading: it assumes the operator acts instantly every time. The repo's own note says
   acting a week late costs most of the benefit." Neither PRO nor CON's brief flags this —
   PRO's headline numbers (holdout +15.8%, DD 47.0%) are the instant-hard-filter case, not
   what an advisory Telegram alert actually delivers.
2. **This is not the first time the 21d shadow-R halt was tested.** A prior 10-window
   rolling backtest (`backtest_halt_multiwindow.py`, referenced in repo memory
   `shadow-gate-validated-kill-switch.md`) found the 21d gate helped in only 2 of 10 windows,
   hurt in 2, and was inert in the rest — with its benefit "concentrated in 2026-01 onward,
   precisely the period the thresholds were calibrated on," which overlaps this exact study's
   holdout window. The prior conclusion was explicit: **"Keep it ADVISORY. Do NOT
   automate. An automated halt would have cost money in every period we can measure."**
   `CLAUDE.md` §8 independently corroborates this pattern for the related weekly-R
   kill-switch: also tested, also rejected as an automated trigger because it "clipped the
   jackpot months that pay for the system."

**Conclusion: HALT 21d is a legitimate, low-regret piece of advice to give the *operator*
(check the dashboard, use `/stop` when the shadow-R trend is bad) — but it should not be
counted as a load-bearing, mechanically-guaranteed contributor to the motion's headline
holdout numbers. The single 6-week holdout used everywhere in this study sits inside the
same calibration-adjacent window the prior 10-window test flagged as the source of its
apparent edge.** Treat the +15.8% GLOBAL+HALT21d figure as an upper bound, not an
expectation.

**Score: both debaters accepted this as settled and neither surfaced the advisory-modeling
gap or the prior 10-window result — this is the one dispute where the judge found something
both sides missed.**

---

## Dispute 5 — the engine disagreement

**Ruling: reduces confidence in every dollar figure quoted anywhere in this debate by a
lot; does not change the ranking on the two metrics that actually matter for a forward
decision (holdout sign, drawdown reduction).**

Two independent engines on the same trades:

| arm | reference engine (`compare_output.txt`) | independent replication (`cmp/RESULTS.md`) |
|---|---|---|
| OLD-BASE | $132,605 (also $91,223 in `variations_output.txt`'s separate run) | $13,796 |
| CURRENT | $8,587 | $1,605 |
| GLOBAL | $20,775 | $15,150 |

Absolute dollars disagree by 2.5–10x depending on which two numbers you pick, and even the
"reference" engine disagrees with itself run-to-run ($132,605 vs $91,223 for the identical
OLD-BASE arm across two of its own output files) — that alone should discount every specific
dollar figure quoted in either brief. But on the decision-relevant axes both engines agree:
**CURRENT is the worst or second-worst arm in both**; **GLOBAL has the smallest max drawdown
in both** (47.5%/44.7%); **GLOBAL is the only arm with a positive holdout in both**
(+8.3%/+1.0%) while **OLD-BASE is negative on holdout in both** (−8.4%/−40.2%, sign agrees,
magnitude does not); and `REPLICATION2.md`'s separate from-scratch replication of the A0/A1
(CURRENT-vs-GLOBAL) pair independently confirms the same three headline directional claims,
with GLOBAL's drawdown landing within 1.1pp of the reference — the tightest agreement in
either replication. The one ranking that genuinely flips between engines is **OLD-BASE vs.
GLOBAL on full-history/restart-date dollars** — the reference engine has OLD-BASE winning
8/8 restart dates; the independent engine has GLOBAL ahead ($15,150 vs $13,796). Both
replication authors attribute the magnitude gap to portfolio-engine path-dependency in the
regime-multiplier/anti-pyramid interaction, not to a disagreement about which trades were
taken (the underlying trade sets matched within a few percent).

**Score: this should reduce confidence in "OLD-BASE beats GLOBAL by 6–10x" (CON's Attack 1
headline) more than it reduces confidence in "GLOBAL is the best-drawdown, only-positive-
holdout arm" (PRO's core claim), because the latter is the part that replicated tightly and
the former is the part that flipped sign.**

---

## Where each debater overstated

**PRO overstated:** calling the case "cleared all six pre-registered criteria" glosses over
Criterion 5's genuinely mixed verdict (`NULL_TEST.md`'s own bottom line calls it **FAIL**,
not pass — the hostile-trim test does break at 2%) and folds the advisory-vs-hard-filter gap
on HALT 21d into the headline numbers without flagging it. The "6/8 restart dates" framing
also undersells that OLD-BASE's median restart equity is ~10x GLOBAL's even where GLOBAL
"wins" the count.

**CON overstated:** Attack 1 ("OLD-BASE dominates... cannot be waved past") treats the
reference engine's dollar figures as settled when CON's own Attack 6 shows a second honest
engine flips this exact ranking — a concession CON does make, but the opening case is written
before that concession lands, so a reader who stops at Attack 1 gets a false sense of
certainty. Attack 3's framing ("everything runs on... selected pairs... controls never run")
is false for the specific transfer statistic (Dispute 3), even though it's substantially true
for A6's magnitude point.

---

## VERDICT ON THE MOTION: **CARRIED WITH AMENDMENT**

**Carried:** replace the 728 fitted `(rr, atr_mult)` entries with the single global
`rr=10/atr_mult=3.0`, keep the trailing stop on. This is the only arm across two independent
engines, the pre-registered protocol, and a from-scratch replication that is simultaneously
(a) not CURRENT — which nobody defends and which is the worst-or-near-worst arm everywhere,
(b) lower drawdown than every fitted alternative in every engine, and (c) positive on the
untouched holdout in every engine that measured it, at the trade level (`criteria_output.txt`
HOLDOUT row: A0 −0.183R/trade vs A1 +0.017R/trade) and the dollar level.

**Amendment:**
1. **Keep the 21-day shadow-R halt exactly as currently shipped — advisory, not
   automated.** Do not treat its backtested +7.9%→+15.8% holdout lift as a guaranteed
   component of the package; it is an optimistic, instant-execution reading of a signal
   that a prior 10-window test found inert-to-mixed outside its calibration window.
2. **Run the still-missing A2 (honest walk-forward per-symbol) and A5 (no-selection, global
   params, full universe) holdout-specific controls before scaling account size up on
   GLOBAL.** The magnitude-shrinkage finding in Dispute 3 (edge 4x smaller off the selected
   728 pairs) is real and unresolved; it doesn't currently threaten the sign of the
   recommendation, but nobody has checked whether it threatens the sign *in the holdout
   specifically* on the unselected population.
3. **Re-validate margin/tick-size/funding assumptions before the 728-entry rewrite goes
   live** — GLOBAL's 2.3x wider stops and ~5x longer holds are real operational deltas
   (RED_TEAM2 A7) that the R-level backtest does not price.

Rejected as a companion move: **reverting the trail to run OLD-BASE fitted+fixed instead.**
It is the highest-dollar arm in one engine and a real, low-risk alternative, but it is
negative on holdout in **all 12 of the 12 toggle combinations tested** and negative in
**both** independent engines — the one property that should matter most for money that has
to survive the *next* six months rather than replay the last three years again.

---

## The decision, if this were my money

**Switch off CURRENT (the live config today) onto GLOBAL rr=10/atr_mult=3.0 with the trail
kept on, immediately** — every table in this study, from both engines, agrees CURRENT is the
worst or second-worst arm on offer, so staying on it is the one option the evidence rules
out cleanly. Keep the shadow-R halt exactly as configured today (advisory), and do not
increase risk-per-trade or account size until the A5 no-selection holdout check comes back.

**The one piece of evidence that would change my mind:** the untouched 6-week holdout net R
of GLOBAL's parameters, measured *only* on the 376 walk-forward-**rejected** pairs (or
equivalently, the pre-registered A5 arm's own holdout, not its full-history aggregate). A6's
full-history number on rejected pairs is already negative (−0.1776R/trade) — if that negative
sign also holds in the untouched holdout window specifically, it means GLOBAL's positive
holdout is still being carried by the same in-sample symbol-selection that broke the fitted
arm, just relabeled as a "global parameter" result, and the motion should be withdrawn
pending a genuinely honest re-selection.
