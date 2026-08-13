# AUDIT-A — Feature ablation on the CURRENT configuration

Config audited: global rr=10.0/atr_mult=3.0, 728 pairs, s3_a1 trail, risk_per_trade 0.003,
net_directional_cap 0.10, gross_open_risk_cap 0.10, btc_short_gate **false**,
long_bull_boost 1.3, CHOP thresholds {favorable:52, cautious:45, adverse:52, critical:55},
committed 2026-08-12. Cost 24.2bps round trip. Split IS<2026-05-25 / HOLDOUT>=2026-05-25.
Last 21 days of entries trimmed. `size_basis='wallet'` throughout (no lookahead).

**Methodology note that is itself a finding:** the repo's own reference sweep scripts —
`risk_study/search_roidd.py` and `risk_study/variations.py` — inherit `short_gate=True`
from `backtest_shadow_gate.LIVE` and never override it. The live config has
`btc_short_gate: false` (removed 2026-08, cost -705.8R). So `risk_study/results/search_roidd.csv`
and `risk_study/results/variations.csv`, both regenerated today (2026-08-12), are **already
testing a dead parameter** — exactly the failure mode this audit exists to catch, one layer
up. This audit does not reuse those files for its dollar-level conclusions; all engine runs
below explicitly set `short_gate=False`. (Recommend someone fix those two scripts.)

Engine harness: `risk_study/agent_out/audit_a/harness.py`. Item scripts:
`run_item{2,2b,3,4,5,6}_*.py`. `run_item2*` use a rebuilt ungated global rr10/atr3
population (`ungated_glob_rr10_am3.parquet`, 71,423 rows, 728 live pairs, btc_bull/impulse
reattached from `build_risk_universe.annotate_btc`) since the pre-built `uni_glob_*` files
are already CHOP<52-filtered and cannot be used to test looser thresholds. Items 3-6 use
the pre-gated `uni_glob_rr10_am3_same.parquet` (correct, since those params don't touch
CHOP admission). Full numeric results: `results.csv`,
`item1_chop_sweep_results.csv` (pandas, R-level, weekly-block-bootstrap CIs),
`item{2,2b,3,4,5,6}_*_results.csv` (engine, $-level + per-window avg R).

---

## Headline: does the CHOP filter still earn its 62% block rate at rr=10/atr=3.0?

**Short answer: its per-trade SELECTIVITY does not transfer out-of-sample at all. Its
exposure-reduction (fewer trades = smaller loss in a bad stretch) still has some value,
but that value is redundant with `gross_open_risk_cap`, which is already doing the same
job more directly and more robustly (see below). Net verdict: UNPROVEN, not clearly
REMOVE-able either, given how thin the holdout sample is (931-3,561 trades, ~2 months).**

### 1. Flat-threshold sweep, R-level (no engine, no $ sizing) — `item1_chop_sweep.py`

Population: `grid_inuni.parquet` restricted to the 728 live (symbol,div_type) pairs,
atr_mult=3.0, `net_r = r_10 - 0.00242/stop_frac`. 70,080 rows after the 21-day trim.
Live gate ≈ flat 52 most of the time (bot spends most of its history in adverse/critical
regime, where the inverted schedule maps to 52/55).

| threshold | IS block% | IS avg net R | IS 95% CI | HOLD block% | HOLD avg net R | HOLD 95% CI |
|---|---|---|---|---|---|---|
| none (no gate) | 0% | +0.115 | [-0.05,+0.29] | 0% | **-0.464** | [-0.87,+0.15] |
| 38 (tightest) | 94.2% | **+0.217** | [-0.06,+0.55] | 95.8% | **-0.838** (worst) | [-1.06,-0.38] |
| 42 | 88.2% | **+0.269** (best IS) | [+0.03,+0.56] | 91.0% | -0.693 | [-0.92,-0.37] |
| 45 | 81.1% | +0.228 | [+0.03,+0.47] | 84.8% | -0.687 | [-0.97,-0.26] |
| 48 | 71.7% | +0.200 | [+0.02,+0.42] | 74.6% | -0.585 | [-0.93,-0.11] |
| **52 (live)** | 55.8% | +0.186 | [-0.00,+0.39] | 57.5% | -0.512 | [-0.88,-0.01] |
| 56 | 37.7% | +0.165 | [-0.01,+0.35] | 38.4% | -0.504 | [-0.90,-0.00] |
| 60 | 20.7% | +0.135 | [-0.04,+0.30] | 19.9% | -0.473 | [-0.87,-0.01] |
| 65 (loosest) | 6.8% | +0.112 | [-0.05,+0.31] | 5.6% | -0.467 (best) | [-0.88,+0.09] |

Two clean, robust facts here:

- **In-sample, tighter is monotonically-ish better** (peaks at 42, not 52 — the live
  threshold isn't even the in-sample optimum among the values tested).
- **Out-of-sample, the relationship INVERTS: tighter is worse.** Every threshold from
  38 through 52 makes the average trade WORSE than doing nothing (none: -0.464). Only
  loosening past 52 (56/60/65) starts to close the gap back toward "no gate," and "no
  gate" is statistically indistinguishable from 52 (CIs overlap almost completely:
  52 → [-0.88,-0.01], none → [-0.87,+0.15]).
- This is the textbook signature of the same overfitting problem already caught
  elsewhere in this bot ("Selection transfer measured": per-symbol rr/atr rho 0.03-0.06
  is noise). CHOP is fit to the same in-sample data the walk-forward pair selection was
  fit to, and it does not select better trades OOS at the current params — no lookahead
  needed to reach this; `chop_bos` (BOS-bar CHOP, not entry-bar) was used throughout.

### 2. Rolling 6-month windows (avg net R, same population)

| window | none | 38 | 42 | 45 | 48 | **52** | 56 | 60 | 65 |
|---|---|---|---|---|---|---|---|---|---|
| 23-06..23-12 | 0.014 | 0.023 | 0.012 | 0.089 | 0.010 | 0.018 | 0.084 | 0.046 | 0.026 |
| 23-12..24-06 | -0.106 | 0.391 | 0.230 | 0.106 | 0.111 | 0.020 | -0.020 | -0.063 | -0.099 |
| 24-06..24-12 | 0.098 | -0.232 | 0.130 | 0.094 | 0.038 | 0.051 | 0.078 | 0.087 | 0.091 |
| 24-12..25-06 | 0.281 | 0.341 | 0.281 | 0.337 | 0.395 | 0.448 | 0.369 | 0.316 | 0.288 |
| 25-06..25-12 | 0.160 | 0.639 | 0.764 | 0.566 | 0.429 | 0.341 | 0.271 | 0.209 | 0.161 |
| 25-12..26-06 | 0.105 | 0.063 | 0.028 | 0.041 | 0.067 | 0.079 | 0.087 | 0.086 | 0.085 |
| **26-06..(holdout)** | **-0.694** | -0.876 | -0.811 | -0.869 | -0.748 | -0.704 | -0.676 | -0.652 | -0.682 |

52 (live) beats "no gate" in only 4/7 windows and beats itself trivially; no threshold
dominates on both IS and every window. Win/loss record vs no-gate: 38→4/7, 42→3/7, 45→4/7,
48→3/7, **52→4/7**, 56→5/7, 60→5/7, 65→5/7. There is a weak drift toward looser thresholds
winning more windows, but none of these margins clear the bootstrap CIs above.

### 3. Regime-aware schedule shape, through the production engine — items 2 / 2b

Three shapes compared: **current** (inverted: {fav 52, caut 45, adv 52, crit 55}),
**flat-45**, **flat-52**, **corrected** (flipped: {fav 55, caut 48, adv 42, crit 35} —
tighter exactly when the regime is worse, the "obvious" design), and **no gate**.

**No shadow-halt overlay (cleanest read of CHOP alone), fresh $1,500 restart each window:**

| schedule | IS final | IS ROI | IS DD | HOLD final | HOLD ROI | HOLD DD | HOLD n |
|---|---|---|---|---|---|---|---|
| **current (live)** | $4,819 | **+221%** | 40.6% | $1,034 | **-31.0%** (worst) | **33.4%** (worst) | 931 |
| flat-45 | $2,592 | +73% | 58.4% | $1,105 | -26.3% | 26.6% | 459 |
| flat-52 | $3,804 | +154% | 42.4% | $1,116 | -25.6% | 29.0% | 837 |
| corrected | $2,516 | +68% (worst) | 55.6% | $1,392 | **-7.2%** (best) | **7.5%** (best) | 132 |
| no gate | $4,512 | +201% | 43.2% | $1,158 | -22.8% | 27.6% | 1,206 |

The **corrected (tighter-when-worse) schedule cuts holdout drawdown from 33.4% to 7.5%**
and the holdout loss from -31% to -7%. This looks like a big win, and if this were the
whole story it would be the top recommendation of this audit.

**It is not the whole story — per-window avg-R tells a different one.** Splitting the same
comparison into the 7 rolling windows (net of the halt), the corrected schedule's avg net R
per trade **loses to the current schedule in 4 of 7 windows**, including the clearly bullish
24-12..25-06 window (current 0.665 vs corrected 0.222) — it only wins the two worst windows
(25-06..25-12: 0.61 vs 0.404, and marginally elsewhere). **This is exactly the pattern
`GROWTH_VALIDATION_REPORT.md` §5 already found and rejected for the old fitted-per-symbol
config** ("fixing the inverted CHOP thresholds → not robust, helps down-markets, hurts
bull") — and it replicates cleanly under the brand-new global rr10/atr3 config. The dollar
improvement in holdout comes almost entirely from **taking far fewer trades** (132 vs 931,
an 86% reduction) during the one bad stretch in the sample, not from picking better trades
— i.e. it functions as a volume throttle, not a selectivity filter.

**Verdict: do not flip the CHOP inversion.** It is not proven robust, again, now on the
new config too. The 90-day-freeze recommendation in `OVERFIT_VERDICT.md` applies here
directly: this needs a genuinely independent bad-market window to test, which the sample
does not yet contain (the "bad" window IS the holdout — testing on it is not confirmation).

### CHOP overall verdict: **UNPROVEN, not REMOVE, not CHANGE**

- The R-level selectivity claim that historically justified CHOP (52, inverted) does not
  hold at rr=10/atr=3.0 — average trade quality gets *worse*, not better, as you tighten,
  out of sample.
- No flat threshold or corrected schedule beats live-52 robustly across rolling windows on
  a per-trade basis.
- The dollar-level drawdown benefit visible under aggressive tightening (corrected
  schedule) is an exposure-reduction effect, not a selection effect, and it is not shown to
  survive outside the one bad stretch that IS the holdout.
- `gross_open_risk_cap` (see below) achieves a similar exposure-management job more
  directly, with an actual holdout-validated optimum at its live value.
- **Action: none this week.** Re-test after `OVERFIT_VERDICT.md`'s 90-day freeze produces
  250+ fresh trades under fixed rules — CHOP is the single best candidate for that re-test
  given the size of the claim resting on it (62% block rate) and how thin the evidence for
  it now looks.

---

## Ranked findings (all other items)

### KEEP — `gross_open_risk_cap` 0.10 — high confidence

`item4_caps_results.csv`, sweep {0.05, 0.075, **0.10**, 0.15, 0.20, 0.30, None}, net_dir_cap
held at live 0.10, short_gate correctly False.

| gross_cap | IS ROI | HOLD ROI | HOLD DD | beats-live on avg R (7 windows) |
|---|---|---|---|---|
| 0.05 | +675% | -4.3% | 14.6% | 4/7 |
| 0.075 | +1,440% | +1.1% | 17.8% | 2/7 |
| **0.10 (live)** | +2,183% | **+11.0% (peak)** | 18.3% | — |
| 0.15 | +2,413% | +7.6% | 22.3% | 2/7 |
| 0.20 / 0.30 / None | +2,385% | +7.6% | 22.3% | 2/7 (identical to 0.15 — non-binding beyond it) |

**Holdout ROI peaks exactly at the live value (0.10)** and degrades in both directions —
tighter costs IS and HOLD return without buying back DD proportionally; looser gains IS
return but gives back HOLD ROI (7.6% vs 11.0%) *and* HOLD DD (22.3% vs 18.3%). This is a
genuinely well-calibrated parameter. Confidence: medium-high (single continuous holdout
window, only ~2 months of data, but the effect is monotonic and clean in both directions
around 0.10, which a coincidence would not usually produce).
**Config diff: none. Keep `gross_open_risk_cap: 0.10`.**

### REMOVE (redundant, not harmful) — `net_directional_cap` 0.10

Same sweep, net_dir_cap ∈ {0.05, **0.10**, 0.15, 0.20, None}, gross_cap held at live 0.10.
0.10, 0.15, 0.20, and None produce **byte-identical** trade sets and dollar outcomes
(IS $34,247 / +2,183% / HOLD +11.0%, all four rows). Only 0.05 (tighter than live) changes
anything, and it's worse on every axis (IS +1,113%, HOLD -3.8%). **At the live
gross_open_risk_cap of 0.10, net_directional_cap never binds** — the gross cap is reached
first in every scenario tested. This is not a bug and costs nothing to leave in
(zero measured cost — literally identical output), but it is currently doing no work: the
correlated-exposure protection this cap was built for is being provided entirely by the
gross cap.
**Config diff: none required. Optionally simplify by removing the net-directional-cap
code path, since it is provably inert at current settings** — but note it would become
load-bearing again if `gross_open_risk_cap` were ever loosened past ~0.10, so removing the
*code* (not just leaving the config key alone) trades a small maintenance win for a latent
risk if someone changes the gross cap later without re-adding it. **Recommendation: leave
both as configured; no action.**

### KEEP — taper schedule (current, descending) vs flat 0.3%

`item5_taper_results.csv` + a supplementary continuous (non-restart) run to see the taper's
actual purpose (protecting an *already-grown* account, which a $1,500 holdout restart can't
exercise — see caveat below).

Fresh-restart-per-window (search_roidd.py convention): IS $34,247 (current) vs $40,609
(flat) — flat wins IS by 19% simply by staying at the top risk rung longer. HOLD is
**bit-for-bit identical** between the two (both $1,664, +11.0%, 18.3% DD, n=661) — because
in this restart-at-$1,500 test the balance never climbs past the $1,500-3,000 rung where
the schedules actually diverge, so the fresh-restart test cannot see the taper's real
effect at all.

**Supplementary continuous run** (single compounding path start-to-end, no restart,
holdout measured as the tail segment of that one curve — this is what the taper is
actually *for*): current schedule reaches $39,925 with **+16.6% holdout-segment growth**;
flat 0.3% reaches a higher $43,125 total but only **+6.3% holdout-segment growth**. Same
max DD both ways (~44.5%). This matches the taper's documented purpose exactly (commit
`ea730b4`: "reduces the SIZE of losses while the edge is cold; does not create profit") —
confirmed again on the new global config: it trades ~7% of total compounded profit for a
2.6x better holdout-segment outcome once the account is actually large.
**Config diff: none. Keep the descending taper schedule as configured.**
Confidence: medium — one continuous path is one draw (methodology point 6 flags this),
and the restart-based test (the more rigorous convention) can't see the effect at all
because the account never reaches the higher rungs within the 2-month holdout alone.

### WEAK EVIDENCE, NOT ACTIONABLE — `long_bull_boost` 1.3

`item3_boost_results.csv`, sweep {1.0, 1.1, **1.3**, 1.5, 2.0}, short_gate correctly False.

| boost | IS ROI | HOLD ROI | HOLD avg R | beats-live (7 windows) |
|---|---|---|---|---|
| 1.0 (off) | +1,866% | +9.9% | +0.068 | 1/7 |
| 1.1 | +2,149% | +9.8% | +0.071 | 1/7 |
| **1.3 (live)** | +2,183% | **+11.0%** | +0.086 | — |
| 1.5 | +2,203% | +10.6% | +0.092 | 4/7 |
| 2.0 | +2,247% | +10.7% | +0.110 | 4/7 |

Monotonic IS improvement with higher boost, but HOLD ROI is basically flat 9.8-11.0%
across the whole range (noise-level differences on ~660 holdout trades) and live (1.3)
is the HOLD-ROI peak even though 1.5/2.0 win more rolling windows on avg R alone.
**Verdict: the differences are too small relative to the holdout sample to act on.**
Boost 1.3 is not clearly wrong, and neither is 1.5-2.0; this needs a larger holdout before
it's worth touching. Not actionable this week.

### WEAK EVIDENCE, NOT ACTIONABLE — regime window length (20 vs 10/40/60)

`item6_regime_results.csv`, tier thresholds held fixed, only the trailing-trade-count
window varied.

| window | IS ROI | HOLD ROI | HOLD DD | beats-live-20 (7 windows, avg R) |
|---|---|---|---|---|
| 10 | +2,456% | +8.5% | 17.6% | 3/7 |
| **20 (live)** | +2,183% | **+11.0%** | **18.3%** | — |
| 40 | +1,588% | +11.2% | 19.0% | **6/7** |
| 60 | +1,280% | +6.0% | 19.9% | 6/7 |
| regime OFF | +2,316% | +3.6% | 22.8% | 6/7 |

Windows of 40 and 60 beat the live 20-trade window on average-R in 6 of 7 rolling windows
— a real, if modest, signal that a longer, less noise-reactive regime window might size
slightly better on a per-trade basis. But on the dollar/holdout metric that matters most
(HOLD ROI, HOLD DD), window=20 is competitive-to-best: 40 is marginally better on ROI
(+11.2% vs +11.0%, noise) but worse on DD (19.0% vs 18.3%); 60 is worse on both. **Regime
OFF is clearly worst on HOLD ROI (+3.6%) despite also winning 6/7 windows on avg R alone**
— a reminder that per-trade avg-R wins and portfolio-level dollar/DD outcomes can point
opposite directions once compounding and correlated-exposure effects are in play, and the
dollar number is the one that matters. **Verdict: weak, mixed evidence; live window=20 is
not dominated by any alternative on the metric that counts. Not actionable — would need a
larger holdout to resolve the 40-vs-20 question either way.**

---

## Summary table (ranked by evidence strength / actionability)

| # | Item | Verdict | Confidence | Action this week? |
|---|---|---|---|---|
| 1 | `gross_open_risk_cap` 0.10 | **KEEP** | medium-high | No — already correct |
| 2 | `net_directional_cap` 0.10 | **REMOVE-eligible (inert)**, KEEP as configured | high (identical output, not a statistical claim) | No — costs nothing either way |
| 3 | taper schedule (descending) | **KEEP** | medium | No — already correct |
| 4 | CHOP filter / thresholds / inversion | **UNPROVEN** — selectivity doesn't transfer OOS; corrected schedule not robust across windows (replicates a known-rejected finding on the new config) | medium (R-level: strong; $-level exposure effect: real but redundant with gross cap) | **No config change** — flag for the 90-day-freeze re-test; this is the single most important thing to re-check once 250+ fresh trades accrue |
| 5 | `long_bull_boost` 1.3 | inconclusive, differences within noise | low | No |
| 6 | regime window (20 trades) | inconclusive, mixed signal | low | No |

**Bottom line for this workstream:** nothing here justifies an immediate config change.
The two caps and the taper schedule earn their place cleanly. The CHOP filter — the
single most load-bearing gate in the system, blocking 55-62% of signals — does **not**
demonstrate the selectivity benefit its historical justification rests on, once tested
honestly against the parameters the bot runs today; but the fix is not to remove or
re-tune it (that path has been tried on the old config and rejected, and this audit found
the same instability again on the new one). The fix is to stop trusting a filter that
hasn't been re-validated on its actual live population, and get the fresh, frozen sample
`OVERFIT_VERDICT.md` already called for — the CHOP threshold should be the first thing
re-scored when that data exists.

## What would change these conclusions

- **CHOP**: an independent bad-market OOS window (not the current holdout, which the
  corrected-schedule result is fit to by construction) that still shows the corrected
  schedule winning would flip this from "unproven" to a real CHANGE recommendation.
- **`long_bull_boost` / regime window**: a longer/cleaner holdout (250+ trades per
  `OVERFIT_VERDICT.md`'s own standard) — current holdout is 630-980 trades over ~2 months,
  too thin to separate live values from neighbors that differ by <1pp of HOLD ROI.
- **`net_directional_cap`**: becomes relevant again only if `gross_open_risk_cap` is ever
  loosened past ~0.10-0.15; worth re-checking jointly if that cap is ever revisited.
