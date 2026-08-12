# RED TEAM — attacking "risk-per-trade must not be raised above 0.3%"

Agent: AGENT-RED. Read-only against `risk_study/*.parquet` and `risk_study/results/*.csv`,
plus the prior-agent artefacts already sitting in `risk_study/agent_out/` (SPEC.md,
WALKFORWARD.md, ENGINE_AUDIT.md, COST_ANALYSIS.md, LIQ_ANALYSIS.md, REPLICATION.md — all
verified before reuse, not taken on faith). New probes for this task live alongside this
file: `redteam_bootstrap.py`, `redteam_window_rank.py`, `redteam_kelly_sensitivity.py`, with
their CSV outputs. No repo file was edited.

**Bottom line up front: the draft conclusion survives every attack aimed at its central
claim (HOLDOUT is genuinely dead) and survives most of the Kelly attack. It is overstated
in exactly one place — the false-precision of the "0.17–0.20%" Kelly number and the implicit
suggestion that the in-sample data is unambiguous — and it silently under-sells itself in
another place, because a real engine bug (already found, not yet fixed) makes the drawdown
numbers used to build the strongest counter-case look better than they should. Net: raising
risk above 0.3% is not defensible on this evidence, but "not defensible" is a weaker, more
honest claim than "we proved the edge is dead," and the report should say the weaker thing.**

---

## (a) Attacks that LANDED

### A1. The cluster-Kelly "0.17–0.20%" figure is more sensitive to an arbitrary choice than the draft admits — and the original script has a real bug

`kelly.py`'s `cluster_returns` aggregates net-R by `.resample(f"{hours}h")` intersected with
`.dt.floor(f"{hours}h").unique()` via `.isin()`. Those two pandas operations anchor at
**different reference points** — `resample` anchors at the first timestamp in the series,
`floor` anchors at the Unix epoch. For 24h (and 6h, 12h) buckets this coincidentally lines
up because the data starts at midnight and the bucket width divides a day evenly. For **48h
and 168h it silently returns zero overlapping buckets**, so `kelly_f` is called with a
size-0 array — the original script would have thrown, or on a slight change of the calling
guard, returned NaN unnoticed. This is a real, previously undiscovered bug in the study's
own tooling. Rewrote it (`redteam_kelly_sensitivity.py::cluster_returns`) to group directly
by the floor bucket, no resample/isin round-trip, and it works correctly at every width.

With the bug fixed, cluster-Kelly *f\** across 6h/12h/24h/48h/168h buckets:

| period | f\*@6h | f\*@12h | f\*@24h | f\*@48h | f\*@168h | max/min ratio |
|---|---|---|---|---|---|---|
| DEV | 0.239% | 0.192% | 0.174% | 0.132% | 0.105% | **2.28x** |
| VAL | 0.299% | 0.242% | 0.198% | 0.198% | 0.179% | **1.67x** |
| FULL | 0.187% | 0.149% | 0.130% | 0.108% | 0.088% | **2.12x** |
| HOLDOUT | 0.000% | 0.000% | 0.000% | 0.000% | n/a (9 blocks) | 0x (mean net R < 0 at every width) |

So the specific number "0.17%" is not a fact about the strategy — it is what you get at the
24h bucket specifically. A defensible reader choosing 168h (weekly clustering — arguably the
*more* correct choice, since `CLAUDE.md`'s own worst-cluster analysis and `LIQ_ANALYSIS.md`
§4 show correlated stop-out events can span a full day and their knock-on effects on regime/
sizing persist for the following week) would report f\* ≈ 0.09–0.18%, roughly half the
24h number. **This is genuine, real sensitivity — up to 2.3x — and the draft states a single
point figure without a sensitivity band. That's an overstatement of precision.**

What does NOT change under attack: **every bucket width from 6h to 168h, in every period,
puts cluster-Kelly f\* below 0.30%** — the ratio moves the number around within a
sub-current-risk range, it never pushes it above the live setting. So the imprecision is
real, but it doesn't flip the qualitative conclusion (the strongest reading a hostile analyst
could construct from this table is "f\* is somewhere under 0.3%, we're not sure exactly
where" — not "f\* might be above 0.3%").

### A2. The simulator's own best-ROI/DD f *disagrees sharply* with cluster-Kelly, and the disagreement direction matters

| window | simulator argmax f (ROI/DD) | cluster-Kelly range (6–168h) | ratio |
|---|---|---|---|
| DEV | 0.75% (ROI/DD 10.96, maxDD 92.9%) | 0.105–0.239% | ~4–7x higher |
| VAL | 1.50% (ROI/DD 33.76, maxDD 61.1%) | 0.179–0.299% | ~5–8x higher |
| FULL | 0.75% (ROI/DD 24.71, maxDD 92.9%) | 0.088–0.187% | ~4–8x higher |

This is a real, large disagreement, and per the task brief I'll say plainly which I believe
and why: **I believe the cluster-Kelly number, not the simulator argmax**, for three
reasons, two of which are new findings from this task (not just asserted):

1. `ENGINE_AUDIT.md` (prior agent, independently verified by me below in A-item 6/§(d)) found
   that `backtest_production_correct.py`'s mark-to-market equity proxy — confirmed still
   live in the working tree, and confirmed to be exactly the configuration `risk_study/sweep.py`
   uses (`taper_basis="wallet", size_basis="equity"`, `backtest_shadow_gate.py:38`) —
   interpolates each open position's *already-known final PnL* linearly toward the present,
   which (i) lets a trade's future outcome inflate other trades' sizing before the market has
   earned it, and (ii) structurally understates intra-trade drawdown because real price paths
   round-trip and a linear interpolant cannot show that. Both effects scale with the dollar
   size of each open trade, i.e. **get worse exactly as risk-per-trade rises** — precisely
   the arms the argmax search is choosing between. A metric with a size-dependent downward
   bias on its own denominator (DD) is not a fair referee for "which size is best."
2. `WALKFORWARD.md` (prior agent) independently found the simulator's own ROI/DD-optimal f
   has **no cross-window stability** — mean pairwise Spearman rank correlation of f-vs-ROI/DD
   across 11 rolling 6-month windows is 0.228 (near-zero), 29% of window pairs are
   *negatively* correlated, and the single global-best f (1.50%) still gives up a mean 1.47
   ROI/DD units versus each window's own (unknowable in advance) best choice. A statistic
   this unstable is not a reliable point estimate of anything; it's closer to picking the
   best-performing lottery ticket after the draw.
3. Kelly, by construction, penalises variance directly (it maximises E[log(1+fR)], so a
   single catastrophic path costs it more than a linear-utility metric like ROI/DD credits an
   equally-sized win) — which is the correct property to want when ~45 positions can move
   together, per `CLAUDE.md`'s and `LIQ_ANALYSIS.md`'s correlated-cluster findings. ROI/DD on
   a *single realized path* has no such protection; it just reports what happened to
   rewarded the path that happened to occur.

**Verdict on this sub-attack: it does not overturn the draft's number, it explains why the
draft was right to trust Kelly over the simulator's own optimizer — but the draft doesn't
say this explicitly, and should, because "the simulator disagrees with us by 5-8x and here's
why we don't believe the simulator" is a stronger, more defensible sentence than silently
using the Kelly number.**

### A3. The strongest honest pro-increase case, built out — and where it actually breaks

Built from `risk_study/results/sweep_main.csv` (34.1bps cost, the study's own headline
figure; cross-checked at 11/18/25bps — see A5).

**(a) If you believe DEV/VAL and treat HOLDOUT as noise:** the in-sample data says the
argmax f (by ROI/DD) is 0.75% (DEV) to 1.50% (VAL) — both **above** the current 0.3%. This is
a real number in the file, not a straw man.

**(b) Does ROI/DD actually improve from 0.3% to that argmax, and by how much?**

| window | f=0.30% ROI/DD | f=0.75% ROI/DD | f=1.50% ROI/DD |
|---|---|---|---|
| DEV | 7.88 | **10.96** (peak) | 6.64 (already past peak) |
| VAL | 10.69 | 12.31 | **33.76** (peak) |

Yes — a genuine, non-trivial improvement in the risk-adjusted metric, in both windows,
between the live setting and somewhere in the 0.75–1.5% range. This is the best version of
the pro-increase argument and it should be stated, not hidden.

**(c) Does that hold while keeping simulated drawdown under 30%? No — and this is where the
case collapses.** Reading `sweep_main.csv` directly:

| window | f | maxDD% (mtm column) |
|---|---|---|
| DEV | 0.10% | 30.4% (29.1%) |
| DEV | 0.30% (current) | **69.7%** (67.7%) |
| DEV | 0.75% (in-sample argmax) | 92.9% (90.9%) |
| VAL | 0.10% | 26.6% (24.7%) |
| VAL | 0.30% (current) | **55.4%** (52.8%) |
| VAL | 1.50% (in-sample argmax) | 61.1% (45.6%) |

**There is no risk level in either in-sample window, at 34.1bps cost, that keeps max
drawdown under 30% except f ≤ 0.10–0.20% — i.e. AT or BELOW the current live setting, never
above it.** The current 0.3% setting is already producing 55–70% modelled in-sample
drawdown; asking "can we raise risk and stay under 30% DD" answers itself in the wrong
direction before HOLDOUT even enters the argument. This directly falsifies the most literal
reading of part 4(c) of the brief in the increase-friendly direction.

**Net verdict on A3**: the honest strongest pro-increase case is real (0.75–1.5% beats 0.3%
on ROI/DD, in-sample, at every cost level I checked) but it (i) requires discarding HOLDOUT
entirely, (ii) requires accepting 90%+ modelled drawdown, which independently (§A6/(d) below)
is itself understated by a live engine bug, and (iii) is exactly the pattern
`WALKFORWARD.md` shows is not stable across time. It is a real case; it is not a *good* case.

---

## (b) Attacks that FAILED

### B1. Block-bootstrap CI on HOLDOUT (weekly blocks) — does NOT rescue the edge

Ran a proper weekly block bootstrap (20,000 resamples, whole ISO weeks resampled with
replacement to preserve the burst-correlation the brief worried about) on
`universe_chopBOS.parquet`, `redteam_bootstrap.py`:

| period | n | weeks | metric | point | 95% CI | P(mean>0) |
|---|---|---|---|---|---|---|
| DEV | 17,163 | 110 | gross R | +0.4375 | [+0.219, +0.665] | 100.0% |
| DEV | | | net R @ 34.1bps | +0.2265 | [+0.007, +0.458] | 97.9% |
| VAL | 13,257 | 47 | gross R | +0.4975 | [+0.284, +0.731] | 100.0% |
| VAL | | | net R @ 34.1bps | +0.2487 | [+0.035, +0.489] | 98.9% |
| **HOLDOUT** | 2,509 | 9 | **gross R** | **−0.2650** | **[−0.581, +0.102]** | **7.5%** |
| **HOLDOUT** | | | **net R @ 34.1bps** | **−0.5213** | **[−0.856, −0.124]** | **0.6%** |

The gross-R CI barely brushes zero at its upper end (7.5% probability the true mean is
positive); the cost-adjusted CI is clearly negative end to end. **DEV-minus-HOLDOUT and
VAL-minus-HOLDOUT gaps**, bootstrapped independently on each side and differenced: point gap
≈ +0.70–0.77R, 95% CI **[+0.27, +1.19]**, P(gap ≤ 0) ≤ 0.1% in every cost/period combination.
HOLDOUT's own 97.5th-percentile bootstrap mean (gross: +0.10, net: −0.12) never reaches
DEV's or VAL's point estimate (+0.44 / +0.50 gross). I also ran a daily-block bootstrap as a
robustness check on the thin 9-weekly-block sample (62 daily blocks instead): net-R 95% CI
**[−0.755, −0.242]**, P(mean>0) = 0.0% — tighter, same conclusion.

**This attack fails. The holdout's badness is not a block-bootstrap artefact of naive
per-trade inference — it survives the correlation-aware test, both at the weekly grain the
brief specified and at a daily grain used as a cross-check.**

### B2. Is HOLDOUT just an unlucky 2-month draw from a strategy that periodically has bad patches? No — it's close to the worst window in 3 years

Scanned every 61-day window in the full 2023-06-01→2026-07-26 history two ways
(`redteam_window_rank.py`):

**156 overlapping windows, stepped weekly** — HOLDOUT ranks:
- gross R: **rank 3rd-worst of 156 (1.9th percentile)**
- net R @ 34.1bps: **rank 1st-worst (worst) of 156 (0.6th percentile)** — the current
  holdout is the single worst 2-month window, by cost-adjusted mean R, of the entire
  reconstructed 3-year history.
- 18.6% of all overlapping windows are gross-negative; 40.4% are net-negative — so "some
  2-month windows lose money" is true and unremarkable, but HOLDOUT isn't merely negative,
  it's at the extreme tail.

**19 non-overlapping (independent) 61-day windows** — only 1/19 is as bad or worse
(gross or net); 4/19 (gross) or 7/19 (net) of the 19 independent windows in three years are
negative *at all*, versus HOLDOUT sitting at or past the single worst of them.

**Recovery check**: the one historically comparable bad window (starting 2023-06-15, net
mean −0.555) was followed by a next-61-day period that was *also* negative (net mean
−0.077), not a bounce-back. n=1, so this doesn't prove HOLDOUT won't recover — it's simply
not evidence that it will, and it's the only precedent in the data.

**This attack fails, and fails harder than B1: HOLDOUT is not "a normal bad patch," it is at
or past the extreme tail of the entire 3-year reconstructed history by both the overlapping
and the independent-window tests.** Calling it "a bad patch we should expect to recover
from" is not supported — the honest label is "worse than anything else we've seen," which
doesn't prove permanence, but does mean the draft's framing survives this specific attack.

---

## Cost estimator (item 5) — attacked and independently re-measured

Pulled `risk_study/results/live_exec_log.parquet` directly (no DB access needed — it's
already saved locally from a prior run) and reproduced the "excess loss beyond −1.0R"
estimator myself, rather than trusting the printed log:

- **133 stop-outs, 2026-08-03 → 2026-08-11 (8 days), 108 distinct symbols.**
  Mean implied round trip = **24.24 bps**, median 23.12 bps.
- **This is genuinely a small, short-window sample** — the attack's premise is right to
  flag it. Per-trade SE gives a 95% CI of **[19.8, 28.7] bps**; a **day-block bootstrap**
  (9 days, correlation-aware, since a single volatile day could dominate) gives an almost
  identical **[19.5, 28.1] bps** — no single day carries the estimate (max single-day share
  is 25/133 trades on 08-06). So the CI is honestly narrow *given this window*; the deeper
  problem is whether this window generalises, not whether the arithmetic on it is noisy.
- **Does it generalise to the 3-year backtest universe? Not cleanly.** The live sample's
  `stop_frac` (median 1.20%, mean 1.40%) is **~34% tighter** than the backtest universe's
  (median 1.82%, mean 2.21%) — this 8-day live window is not drawn from the same stop-width
  mix as the historical universe the cost gets applied to. Since `cost_R = bps / stop_frac`,
  applying one flat bps figure across both is an extrapolation, not a matched measurement.
- **Does "excess loss beyond −1R" conflate cost with gap-through risk?** Yes, and this is a
  real cost either way (a stop that fills 30bps through its intended level because the
  market gapped is money actually lost), but it is not a *fee*, and it will not scale
  linearly with position size the way a fee does — it scales with *how much liquidity your
  order needs relative to what's on the book*, which is exactly the thing raising
  risk-per-trade would make worse (bigger clips on the same thin alts). The 24.2bps figure
  was measured at TODAY's tiny position sizes (0.3% risk); it says nothing about market
  impact at 3–10x that size. `LIQ_ANALYSIS.md`'s and `ENGINE_AUDIT.md`'s finding that the
  study's market-impact model (`liq_impact_k`) is off by default applies here too.
- **My own best estimate**: central **~24 bps**, plausible range **19–30 bps** for cost *at
  current position sizes*, trending higher (unmeasured, but directionally certain) at any
  meaningfully larger size. This is *lower* than the study's stated 34.1 bps trailstats
  figure, which if anything makes DEV/VAL/FULL's net alpha look slightly better than the
  draft credits.
- **But this entire sub-debate turns out to be moot for the actual recommendation**: HOLDOUT's
  breakeven cost is **−35.3 bps** (`breakeven_cost.csv`) — negative, meaning it loses money
  even at *zero* cost. No cost estimate in the 11–45 bps range I or any prior agent
  considered can turn HOLDOUT positive; I confirmed this directly by rerunning the sweep at
  11, 18, and 25 bps (`sweep_cost0.0011.csv`, `.0018.csv`, `.0025.csv`) — HOLDOUT's best-f
  ROI/DD is −0.82, −0.88, −0.92 respectively, negative at every cost level tested and every
  risk level swept. **The cost estimate matters for how profitable DEV/VAL look; it is
  irrelevant to whether HOLDOUT clears the bar, because HOLDOUT doesn't clear zero.**

---

## Engine-bias direction (item 6) — confirmed, and it cuts in ONE direction only

Verified directly against the working tree (not just trusting the prior audit):

- **Defect #1 (close-batch dict-insertion-order bug) is FIXED** in the current
  `backtest_production_correct.py` (`git diff` shows `closed_keys` now sorted by
  `pos['exit_time']`, both mid-run and in the end-of-data flush) — confirms the task
  description's claim.
- **Defect #2 (mark-to-market equity leaks each open trade's already-known final PnL via
  linear time-interpolation) is NOT fixed** — still present verbatim
  (`backtest_production_correct.py` STEP E0, `unrealized += pos['pnl'] * frac`). Confirmed
  this is the exact configuration `risk_study/sweep.py` runs with: `size_basis="equity"`
  (`backtest_shadow_gate.py:38`), i.e. `sweep_main.csv` — the file underpinning both the
  draft's Kelly cross-check and my own A3 pro-increase case above — **is subject to this
  bias on every row.**
- **Direction, unambiguous**: the prior audit's own numeric demonstration (real data,
  `production` scenario) showed switching to the equity/MTM basis pulls reported drawdown
  down 4.6 percentage points (87.7%→83.0%) *in the same run*, and reasoned that the error
  scales with each open trade's dollar PnL — i.e. **with risk-per-trade**. I did not need to
  re-derive this; it follows mechanically from the formula (`unrealized` is a linear function
  of `pos['pnl']`, which is `r_result * risk_usd`, which is linear in `f`) and the
  `size_basis="equity"` confirmation above pins it to the exact file the study relies on.

**This means every `max_dd_pct`/`roi_dd_ratio` number in `sweep_main.csv` at higher f is
flattered relative to lower f — the bias grows with the thing being tested.** It cuts in
exactly one direction: **it makes the case for raising risk look better than it should, never
worse.** There is no plausible mechanism by which this specific bug would make the
anti-increase case artificially strong — it is a one-way thumb on the scale, and it's on the
scale in favour of the reading this red-team was asked to attack *for*. That's an ironic
finding for an adversarial report: the one uncorrected engine bug found in this study makes
my own strongest pro-increase argument (A3) look better than it should, and correcting for
it would only widen the gap the draft is pointing at.

---

## (c) Revised statement of what the evidence actually supports

The draft conclusion is **directionally correct and the recommendation should stand**, but
two of its supporting claims are overstated as written:

1. "Risk-per-trade must NOT be raised above 0.3%" — **survives** every attack aimed at it.
   HOLDOUT's negative edge is not a statistical artefact (block-bootstrap CI excludes zero at
   net cost, and only marginally touches zero at gross), is not a normal bad patch (it ranks
   at or past the worst 2-month window of the entire 3-year reconstructed history, both by
   overlapping- and independent-window tests), and is not a cost-model artefact (breakeven
   cost in HOLDOUT is negative — it loses at zero cost, so no defensible cost assumption
   rescues it). This is the load-bearing claim and it holds up.
2. "The growth-optimal fraction is ~0.17–0.20%" — **directionally right, numerically
   overprecise.** The true cluster-Kelly answer, at defensible clustering choices from 6h to
   1 week, ranges 0.09–0.30% across DEV/VAL/FULL — a real ~2x band, not a point estimate —
   and disagrees by 4–8x with the simulator's own best-ROI/DD f (0.75–1.5%). I side with
   Kelly over the simulator (§A2), but the draft should present a range and state its reason
   for preferring Kelly explicitly, not a bare number.
3. **The single cleanest fact in the whole study, and the one I'd lead with if rewriting the
   recommendation**: at the *current* 0.3% setting, the study's own in-sample simulation
   already shows 55–70% max drawdown in DEV/VAL. The question "should risk go up" is
   answered before HOLDOUT is even invoked — the only f-values that keep modelled drawdown
   under a sane 30% ceiling are at or below where the bot already sits, never above. HOLDOUT
   then independently rules out even that in-sample tradeoff going forward. Two separate
   arguments, both landing on "don't raise," is a stronger form of the conclusion than either
   one alone — and it's not the form the draft chose to lead with.

## (d) The single strongest reason the anti-increase recommendation could still be wrong

Not a statistical objection — every statistical objection I tried failed. The strongest
remaining objection is **epistemic**: the entire study, on both sides of the debate, is a
backtest reconstruction of a strategy whose actual rules changed multiple times across the
window being replayed (walk-forward RR/ATR refits, the CHOP-lookahead fix, the BOS-timing
fix applied 2026-07-25, the trailing stop added 2026-08-02 — none of which existed
throughout the full replayed history). `universe_chopBOS.parquet` applies *today's* rule set
uniformly backward across three years that were not actually traded under today's rules. The
repo's own prior finding (`OVERFIT_VERDICT.md`, cited in `CLAUDE.md` §10) is exactly this:
only 13.7% of past live trades would be retaken by today's bot. If that reconstruction has
even one more undiscovered lookahead bias of the kind already found and fixed once (the CHOP
bug was worth +0.29R/trade and flipped the gate's sign OOS) — and this task found one more
real bug in adjacent tooling within an hour of looking (§A1) — then HOLDOUT's badness could
be partly an artefact of the *reconstruction*, not of the live strategy. This can't be ruled
out from inside this dataset; it can only be resolved by comparing the reconstruction's
`HOLDOUT` window trade-for-trade against the bot's real logged trades over the same dates
(`live_exec_log.parquet`/`live_trade_history.parquet` already contain live data from
2026-08-03 onward — a direct backtest-vs-live parity check over the days that do overlap
would be the single highest-value follow-up, and it's outside this task's scope to run).
