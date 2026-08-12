# PRO: Replace fitted (rr, atr_mult) with global rr=10/atr=3.0, keep trail on, adopt HALT 21d

## Opening case

The correct baseline for this motion is **CURRENT (fitted + trail)** — today's actual
live bot, trail deployed 2026-08-02 — not OLD-BASE (fitted + fixed TP), which is a
strategy the bot no longer runs. The motion stipulates "keep the trailing stop on," so
any comparator without a trail is off-motion by construction. Against the real baseline:

| | CURRENT (today's bot) | GLOBAL (the motion) |
|---|---|---|
| final $ (live defaults) | $5,744 | **$17,493** (3.0x) |
| maxDD | 65.3% | **47.0%** |
| holdout | **−48.4%** | **+7.9%** |
| top-1%-of-trades share of profit | 524.8% | 150.6% |
| beats CURRENT, 8 restart dates | — | **6/8** |

Add the pre-registered HALT 21d (also part of the motion): CURRENT holdout improves to
only −12.6%; GLOBAL's improves to **+15.8%**, DD unchanged at 47.0%. On every axis that
matters for whether to run this bot going forward — drawdown, holdout sign, and how much
of the return depends on a handful of lottery trades — GLOBAL dominates the thing it is
actually replacing.

It also clears all six pre-registered criteria in `PROTOCOL_symbols.md`: beats the
incumbent 4/4 walk-forward folds and in the untouched holdout; beats a 2,000-draw random
(rr,atr) assignment at the 100th percentile (p=0.0005) while the incumbent's *own* fitted
config sits at the 0th percentile of that same null — the walk-forward's per-symbol
fitting is statistically indistinguishable from noise, or worse; beats BTC buy-and-hold,
a long-only basket, and a 200-EMA trend follower on ROI/DD; survives having its best 1-5%
of trades stripped out (A0 goes negative when its top 1% is removed, A1 stays positive
until 2% is removed); and was independently reproduced from scratch by a second agent
with no shared code, matching drawdown within 1.1pp and confirming all three headline
directional claims.

## Rebuttals to the five strongest objections

**1. "OLD-BASE makes 5x more money and beats GLOBAL 8/8 restart dates."** True, and I
won't hide it — but OLD-BASE is not a legal move under this motion, which keeps the trail
on. OLD-BASE's dollars are also the reason the trail was deployed in the first place:
71.4% max drawdown, and **156-225% of its entire lifetime profit sits in the top 1% of
trades** — a strategy whose survival depends on not missing the one jackpot trade a
quarter, and whose OOS holdout return is **negative in all 12 of 12 toggle combinations
tested** (−5.0% to −24.8%). GLOBAL is positive in 6 of the same 12, including live
defaults and the actual proposed package (defaults + HALT 21d, +15.8%). If the real
choice is "trail off and hope the fitted config's fat tail keeps paying" vs. "trail on
with a config that is OOS-positive under the majority of stress toggles," the second is
the more defensible bet for money that has to survive the next six months, not just
replay the last three years once.

**2. "atr=3.0 is outside the bot's own searched range; at atr≤2.0 every global arm's
holdout is negative."** Conceded as stated in isolation — GLB-10/2 (atr=2.0) holdout is
−15.5% at live defaults. But two things cut against reading this as fatal. First, the
red team's own gradient extension (`extended_atr_grid_raw.parquet`, atr ∈
{2,3,4,5,7,10}) shows net R/trade **peaking at atr=2.0-3.0 and degrading monotonically
beyond that** (+0.068 → +0.053 → +0.011 → −0.079 → −0.180 → −0.180). atr=3.0 is one step
past a boundary the optimizer was never allowed to cross, sitting at a real local peak —
not a runaway extrapolation chasing an unbounded "wider is always better" gradient.
Second, and more directly: under the motion's own HALT 21d, atr=2.0's holdout flips to
**+23.4%** — actually higher than atr=3.0's +15.8% in the same scenario (atr=3.0 keeps
the drawdown edge, 47.0% vs 56.1%). The boundary objection holds against the naked
parameter swap; it does not hold against the actual three-part motion on the table,
which pairs the parameter change with the halt.

**3. "Three of four fold CIs include zero."** Correct — only F1's confidence interval
excludes zero individually; F2-F4 are directionally consistent (all positive point
estimates) but not each individually significant, and the folds are nested/overlapping
so they aren't independent draws. I won't dress this up as four independent
confirmations. What it does not undermine: the protocol's decision rule was pre-declared
as a sign count across folds specifically because folds this short and this correlated
were expected to be individually noisy — the aggregate evidence is the random-null test
(100th percentile, p=0.0005, a completely different statistical test from the fold CIs)
and the holdout itself, whose bootstrap CI **does** exclude zero ([+0.0104, +0.3890]),
independently reconfirmed by the red team. One weak leg doesn't collapse a stool with
three others still standing.

**4. "The edge is ~4x smaller off the walk-forward's selected pairs."** True in the
red-team's own numbers (+0.1235 R/trade on the selected 728 pairs vs +0.0311 on the full
unselected in-universe set), and I'll concede it's a real open question about *why*
GLOBAL works. But it doesn't touch *this* motion. The motion doesn't propose changing
the symbol universe — it keeps trading the same 728 pairs the bot already trades today,
and swaps only the (rr, atr_mult) assigned to them. A0 and A1 are measured on that exact
same identical pair set, so the comparison is apples-to-apples for the decision actually
being asked. The unresolved question — does this generalize to symbols nobody has
cherry-picked yet — is a real caveat for a *future* universe-selection motion (protocol
arms A2/A4/A5), not evidence against replacing fitted params on the universe already
being traded.

**5. "The holdout is six weeks."** Yes, and the protocol says so itself: it can refute
a strong claim but can't establish a weak one on its own. It isn't standing alone here.
The rolling-window check (159 overlapping 42-day windows across the full 2023-06→2026-07
history) has GLOBAL beating FITTED in **116/159 = 73.0%** of them, with the actual
pre-registered holdout landing at the 66th percentile of that distribution — an
above-average window for the claim, not a cherry-picked lucky tail. Six weeks is thin by
itself; six weeks that also sits inside a 73%-favorable three-year distribution is not
just a lucky draw.

## Concessions, stated plainly

- OLD-BASE earns more dollars than GLOBAL over the full backtest and at every restart
  date — but it is off-motion (no trail) and OOS-negative everywhere it was tested.
- atr=3.0 was never searched by the bot's own walk-forward; the positive absolute holdout
  at live-default settings depends on that one-step extension.
- Individual walk-forward fold significance is weak (1 of 4 clears its own CI); the case
  rests on the sign pattern plus the independently-significant random-null and holdout
  tests, not fold-by-fold significance.
- The mechanism is not fully disentangled: some of GLOBAL's edge may be attributable to
  the pre-selected pair set rather than the parameter choice in isolation, and the
  protocol's own arms built to separate the two (A2/A4/A5) were not run.
- Six weeks of untouched holdout is a real constraint on how much confidence any single
  number here should carry.

## The single strongest argument for the motion

**The incumbent's own selection mechanism is measurably worse than chance.** A0 — the
config the bot runs today — sits at the **0th percentile** of 2,000 random (rr, atr_mult)
assignments over the identical trade set, while a single global choice sits at the
100th percentile of that same null (p=0.0005). Whatever uncertainty remains about
*which* global parameter is optimal or *why* it works, the one thing this study
establishes cleanly is that per-symbol fitting, as currently practiced, is not just
unhelpful — it is actively worse than not fitting at all. Combined with a trail that
cuts drawdown and profit concentration in every regime tested, and a halt that turns
holdout positive in the scenario that matters most, replacing 728 numbers chosen by a
process indistinguishable from noise with one number that beats that noise at the 100th
percentile is the more defensible default, even before the remaining open questions
about the exact optimum are settled.
