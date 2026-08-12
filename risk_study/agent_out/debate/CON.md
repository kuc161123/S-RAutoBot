# CON — against replacing 728 fitted configs with global rr=10/atr=3.0 (+trail +21d halt)

## Opening case

The motion asks us to take a live account, rip out 728 individually walk-forward-fitted
`(rr, atr_mult)` entries, and replace all of them with a single point — `rr=10,
atr_mult=3.0` — that (a) the bot's own per-symbol optimizer has never once selected in
production, (b) was validated on a six-week holdout where 3 of 4 training-fold confidence
intervals include zero, and (c) sits on data that was never re-selected honestly, so we
cannot separate "the parameter is better" from "the symbol universe was already
cherry-picked by the same process." Meanwhile the same evidence base contains an
option — **OLD-BASE: today's fitted configs with the trailing stop turned back off** —
that beats the proposed replacement at every one of 8 independent restart dates by a
median of roughly 10x. My position: **do not adopt the motion. At most, apply the 21-day
halt to the bot as currently configured (or revert the trailing stop off) as a reversible
holding action, and run the two pre-registered control arms (A2 honest walk-forward, A5
no-selection) before touching a single config entry.** If forced to choose the single
best-supported action on this evidence, it is closer to "stop or shrink trading" than
"redeploy into an unvalidated global parameter."

## Attack 1 — OLD-BASE dominates the recommendation on the study's own primary engine

`compare_output.txt` (the leak-free reference engine, $1,500 start, 24.2bps): OLD-BASE
$132,605 final vs GLOBAL's $20,775 — 6.4x. Restarted at 8 independent dates, OLD-BASE beats
GLOBAL in **8/8**, median $95,931 vs $9,886 — a ~10x gap, the single largest effect size
anywhere in this study. If the question is "what should the bot run," an arm that wins
every robustness slice by an order of magnitude and requires zero new parameter risk
(it's just turning the Aug-2 trailing-stop change back off) cannot be waved past in favor
of a value the bot has never traded before. Any case for GLOBAL has to explain why it beats
a demonstrably stronger, already-live-tested alternative — the study never makes that
argument, because OLD-BASE isn't one of the pre-declared arms.

## Attack 2 — atr_mult=3.0's entire positive holdout evaporates inside the bot's real grid

`RED_TEAM2.md` attack A1: restricting the challenger set to `atr_mult ∈ {1.0,1.5,2.0}` —
the only values the live per-symbol optimizer has ever actually chosen — **every global
arm's holdout return is still negative** (best is rr10/atr2 at −11.7%). The one thing that
turns the holdout positive is specifically the 3.0 grid point, one step beyond anything
production has run. I concede the study's protocol pre-registered atr_mult=3.0
deliberately (`PROTOCOL_symbols.md` §3: "the grid extends one step... so the live choice
is interior"), so this is not post-hoc p-hacking. But pre-registered or not, the entire
absolute-dollar case for the motion rests on a single grid cell the bot's own historical
optimization has never validated on live capital, and RED_TEAM2's own gradient probe (A2)
shows net R is *not* monotone past that cell — it peaks near 2–3 and goes negative by 4.0.
That is a highly curvature-sensitive knob to bet the account on from a 4-point grid.

## Attack 3 — the improvement rides on symbol/pair selection the study never controlled for

`RED_TEAM2.md` A6: GLOBAL's edge on the walk-forward's own selected 728 pairs
(+0.1235 net R/trade) is roughly **4x** its edge on the full, unselected in-universe pair
set (+0.0311). The three arms built specifically to separate "does a global parameter
transfer" from "does the walk-forward's pair selection transfer" — A2, A4, A5 — were never
run, with no logged reason. `STRATEGY_VERDICT_2026-08-11.md` already proved this exact
selection artefact once (the bot's historical edge was "per-symbol RR/ATR selection with
zero persistence OOS" stacked on an underpriced cost model). Deploying GLOBAL onto the same
728-pair set that produced that first artefact, without running the honest-selection
control, risks repeating it in a new guise.

## Attack 4 — the significance claims are thinner than "cleared all six criteria" implies

`criteria_output.txt`'s own fold table plus `RED_TEAM2.md`'s bootstrap CIs: only **1 of 4**
training folds individually excludes zero (F1: [+0.0228,+0.4889]; F2/F3/F4 all straddle
zero). "Wins 4/4" is a sign count on point estimates on non-independent, overlapping,
nested training windows — not four independent confirmations. Separately, two agents
working the same underlying numbers (`criteria_output.txt` vs `NULL_TEST.md`) reached
opposite PASS/FAIL verdicts on Criteria 3 and 5, and report top-1%-concentration figures
that differ 5x from each other on the same statistic under different denominators. A
recommendation whose own supporting documents disagree on whether it passed its
pre-registered bar should not be treated as cleanly validated.

## Attack 5 — the standing verdict says this whole exercise may be choosing how to lose

`STRATEGY_VERDICT_2026-08-11.md`: honest, unselected gross alpha ≈ +0.12R/trade against a
measured execution cost of 0.232R/trade — alpha is roughly half of cost, and "no exit
rule, cost cut, filter, sizing scheme or portfolio construction closes a gap of that
shape." Its own decisive test (§1.3) found **zero of 120 out-of-sample cells positive**
for the honest, unselected version of this exact strategy family. If that verdict is
right, then OLD-BASE, CURRENT, and GLOBAL are all trading a negative-expectancy signal, and
the entire debate is about which configuration loses money more slowly and with a better
drawdown shape — a real question, but a much smaller one than "which config should we
deploy to grow the account." That argues for capital preservation (halt, or minimum viable
exposure) over any redeployment, global or otherwise.

## Attack 6 — the two independent engines disagree by 2.5–7x on absolute magnitude

Reference engine: OLD-BASE $132,605 full-history / $91,223 in the headline evidence block.
Independent from-scratch replication (`cmp/RESULTS.md`): OLD-BASE $13,796 — roughly a 7-10x
gap on the *same underlying trade set* (matched within a few percent per
`REPLICATION2.md`), traced to portfolio-engine path-dependency in the anti-pyramid/regime-
multiplier interaction, not to a disagreement about the signals. **I must concede this cuts
both ways**: the independent engine actually ranks GLOBAL *above* OLD-BASE (best on every
axis: highest equity, 44.7% vs 76.0% maxDD, only arm with a positive holdout and more up-
than-down months) — a real point for the motion I am arguing against, and I will not bury
it. But that disagreement is exactly the problem: two honestly-built simulators disagree by
an order of magnitude on which arm even wins, based on compounding/regime-loop mechanics
neither author could fully reconcile. That is not a foundation solid enough to justify an
irreversible rewrite of 728 live config entries. It is a foundation solid enough to justify
caution.

## Attack 7 — operational risk is real and unpriced

`RED_TEAM2.md` A7: GLOBAL's median stop distance is 2.3x wider (4.10% vs 1.79%) and median
hold time 5-5.5x longer (33h vs 6h) than the fitted incumbent. That means materially
different position notional, minimum-order-size exposure on cheap alts, funding accrual,
and anti-pyramid slot occupancy (24.6% of GLOBAL's own candidates blocked vs 8.8% for
fitted) — none of which the R-level backtest fully prices, and all of which need
re-validating against live margin/tick-size behavior before 728 entries change at once on
a real account.

## Concessions

- Per-symbol fitting, as currently practiced, is very likely capturing noise (A0 sits at
  the 0th percentile of 2,000 random-assignment draws) — I do not defend the specific 728
  fitted values as good.
- The genuine holdout diff for GLOBAL-vs-CURRENT does clear a real if thin bootstrap CI
  ([+0.0102,+0.3956]) and replicates directionally in an independent implementation.
- Cost sensitivity favors GLOBAL, not against it — this is not a cost-model artefact.
- The independent replication engine ranks GLOBAL above OLD-BASE on every axis, which is
  real evidence for the motion that a CON case must not suppress.
- CURRENT (today's actual live config, fitted+trail) is bad by every measure in every
  engine — the status quo is not a strong alternative either.

## Recommended alternative

Do not rewrite the 728 configs today. Apply the 21-day halt to the current live
configuration as a reversible, low-risk holding action (variations_output.txt: HALT 21d
lifts OLD-BASE's holdout from −12.4% to −5.4% and GLOBAL's from +7.9% to +15.8% — halting
helps every arm, so it is the one move with no serious downside). In parallel, run the
missing pre-registered A2/A4/A5 controls before any parameter rewrite, and treat
`STRATEGY_VERDICT_2026-08-11.md` as the operative constraint: if the honest, unselected
edge is below cost, the responsible default is minimum viable exposure, not redeployment
into an out-of-production parameter value on the strength of a six-week holdout and a
4-point grid.
