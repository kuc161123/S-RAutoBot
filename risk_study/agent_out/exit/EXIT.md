# AGENT-EXIT — exit-rule search (LEAD 1: scale-out+breakeven, LEAD 2: trigger/trail tuning)

Benchmark to beat: global rr=10/atr_mult=3.0 with the s3_a1 trail — $17,205 from $1,500,
47.0% max drawdown, ROI/DD ~22, +15.8% on the untouched holdout (per task brief).

Split protocol used throughout: **fit/explore on entries before 2026-05-25, judge on
entries on/after 2026-05-25 (untouched holdout)**. Everything below is at the per-trade
R level (net of cost), not a portfolio/dollar/DD replay — no arm here changes the
mean-R verdict enough to be worth the extra machinery, so none was built.

**Bottom line up front: nothing found here beats the live s3_a1 trail in both periods.
LEAD 1 (scale-out+breakeven) is net-of-cost WORSE than the trail even in-sample. LEAD 2
(trigger/trail tuning) cannot turn the holdout positive at any of the 24 grid points —
every single cell, including the current live constants (3.0/1.0), is net-negative
after 2026-05-25. That is the strategy's edge going negative OOS (already flagged in
memory as "Edge negative OOS since May 2026"), not an exit-rule problem, and no
exit-rule retune fixes it.**

---

## LEAD 1 — scale-out with breakeven

### Schema of `scaleout_universe.parquet`

24,931 rows, 277 symbols, entries 2025-01-05 → 2026-07-12 (this file is NOT a full
3-year universe — `GEN_FROM = 2025-01-05` in the builder — so LEAD 1's "in-sample"
window is ~16.5 months and its holdout is only **7 weeks / 2,286 trades**).

Columns: `entry_time, symbol, side, entry_price, sl_price, rr_cfg, atr_mult, sl_dist,
n_tp_hit, n_tp_hit_nobe, so_stop_r, so_stop_t, nb_stop_t, nb_end_t, base_stop_t,
base_tp_t, tr_stop_t, tr_stop_r, tr_tp_t, tp1_t, tp2_t, tp3_t, tp4_t, btc_bull,
btc_impulse`.

It stores **one shared bar-walk** with four resolvable arms baked in:
- `base_stop_t/base_tp_t` — plain fixed-TP (no trail, no scale-out)
- `tr_stop_t/tr_stop_r/tr_tp_t` — the live s3_a1 trail (trigger=3.0R, trail=1.0 ATR)
- TP1..TP4 fill times + `so_stop_r`/`so_stop_t` — **scale-out with breakeven-after-TP1**
- `n_tp_hit_nobe`/`nb_stop_t`/`nb_end_t` — **scale-out control, same 4 TPs, no
  breakeven move** (isolates the BE effect from the scale-out effect)

`(rr_cfg, atr_mult)` pairs present are the **per-symbol live-FITTED** values from
`config.yaml` (3/1, 3/1.5, 3/2, 5/1, 5/1.5, 5/2, 8/1, 8/1.5, 8/2, 10/1, 10/1.5, 10/2) —
**not** the global rr=10/atr_mult=3.0 arm that is the task's stated current-best
benchmark. This file tests "does breakeven+scale-out help the live bot's actual
picks," not "does it help the global-rr10/atr3 arm." Flagging this clearly: the
`trail` column here is the live per-symbol-config trail, a different (weaker,
per CLAUDE.md's own numbers) benchmark than the $17,205 global arm.

### Contamination check (per the task's explicit warning)

`build_scaleout_universe.py:256-260`:
```python
# CHOP is read on the BOS bar, NOT the entry bar. ch[e] is computed from
# bar e's own high/low/close, which do not exist until an hour AFTER the
# fill at o[e] -- a lookahead worth ~+0.29 R/trade that flipped this
# gate's sign out-of-sample. See STRATEGY_VERDICT_2026-08-11.md 2.1.
if np.isfinite(ch[bos]) and ch[bos] >= CHOP_T:
    continue
```
**Uses `ch[bos]`, not `ch[e]`. NOT contaminated** — this file was already rebuilt after
the CHOP-lookahead fix (the comment cites the exact fix). No rebuild needed.

### R-level results (net of cost = `r - 0.00242/stop_frac`, applied flat per trade —
conservative for scale-out, see caveat below)

| arm | n_ins | mean R (in-sample) | 95% CI | n_oos | mean R (holdout) | 95% CI |
|---|---:|---:|---|---:|---:|---|
| base (fixed TP) | 22,616 | **+0.5385** | [0.327, 0.747] | 2,286 | **-0.1929** | [-0.547, 0.200] |
| trail (live s3_a1, per-symbol picks) | 22,616 | +0.2594 | [0.151, 0.366] | 2,286 | **-0.0904** | [-0.332, 0.147] |
| scale-out + breakeven | 22,616 | +0.2354 | [0.113, 0.353] | 2,286 | -0.1378 | [-0.379, 0.114] |
| scale-out, no BE (control) | 22,616 | +0.3028 | [0.162, 0.436] | 2,286 | -0.1432 | [-0.415, 0.148] |

Gross (no cost), full sample, for comparison to the `config.yaml` comment's claim:

```
base            +0.6371
trail           +0.3931
scaleout_be     +0.3670   <- matches the config.yaml "+0.367" figure exactly
scaleout_nobe   +0.4276
```

The **+0.367 figure reproduces exactly** — confirms this analysis replicates the same
computation the config.yaml comment cites. But the comment's comparator ("+0.208") does
not match any of the four arms measured here (closest control here is scaleout_nobe at
+0.4276 gross, not +0.208); it likely referenced a different comparison (a bare-trail
breakeven variant, not this scale-out-without-BE control) and should not be read as "BE
beats no-BE scale-out" — in this data it's the opposite: **no-BE scale-out beats
BE scale-out**, gross and net, in-sample and in holdout.

### Verdict — LEAD 1 is a negative result

- **Net of cost, scale-out+breakeven is not the best of even these four arms
  in-sample** (+0.2354, below base +0.5385, below scale-out-no-BE +0.3028, and below
  its own trail benchmark +0.2594).
- In the holdout, all four arms are net-negative. Scale-out+BE (-0.1378) is not
  better than the trail (-0.0904) — the trail is the **least bad** arm of the four in
  the untouched period.
- Adding breakeven to scale-out does **not** clearly help vs. not adding it: no-BE
  scale-out has a higher in-sample mean R (+0.3028 vs +0.2354) and a similar (slightly
  worse) holdout mean (-0.1432 vs -0.1378) — a wash, not a case for BE.
- The holdout here is only 7 weekly blocks (2,286 trades) — CIs are wide and every
  arm's CI straddles zero, so none of these differences should be read as
  statistically established; the point estimates simply give no reason to switch.

**LEAD 1 does not beat the current live trail. Do not deploy scale-out+breakeven.**

Caveat in scale-out's favor that the numbers above do NOT capture: the flat cost model
(`0.00242/stop_frac` charged on the whole trade) overstates scale-out's true cost —
three of its four fills are resting limit orders that in reality pay no slippage (see
`report_scaleout.py`'s own note). A more realistic (lower) cost for scale-out would
close some but not all of the ~0.024-0.07R net gap to the trail seen in-sample, and
would not flip the holdout sign. This does not change the verdict.

---

## LEAD 2 — trail trigger/ATR tuning for the global rr=10/atr_mult=3.0 arm

### Method

Rebuilt (did not edit `risk_study/build_universe_trail_helpers.py` or
`build_global_trail.py`) a standalone resolver in this directory
(`build_trigger_sweep.py`) that reproduces `build_global_trail.py --rr 10 --atr 3
--pairs same` signal-for-signal (same detection, BOS-within-12-bars, EMA gate at
`bos`, CHOP gate at `bos` < 52, entry at `open[bos+1]`, ATR from `atr[bos]`) and, for
each signal, walks the 1H bars **once**, evaluating all 24 `(trigger_r, trail_atr)`
combinations in parallel on that single walk (stop-wins-ties per combo; a combo that
does not resolve by the end of the cache is dropped from **every** cell for that
signal, so all 24 cells share an identical trade set).

Sanity check against the existing `risk_study/uni_glob_rr10_am3_same.parquet` (built
with the live constants trigger=3.0/trail=1.0): this rebuild's (3.0, 1.0) cell gives
net R +0.1876 over 32,054 trades vs. +0.1916 over 32,102 trades in the original file —
same signal set within the handful of trades dropped by the "all-24-must-resolve"
requirement. Confirms the resolver replicates the validated `resolve_trail` logic.

32,054 signals, 277 symbols, 2023-06-01 → 2026-07-25. Split: 29,721 in-sample (156
weekly blocks) / 2,333 holdout (**only 9 weekly blocks** — treat CIs as indicative,
not conclusive).

### Results — mean net R/trade, all 24 cells (ranked by in-sample mean)

| trigger_r | trail_atr | mean R (in-sample) | 95% CI | mean R (holdout) | 95% CI |
|---:|---:|---:|---|---:|---|
| 4.0 | 0.5 | +0.2748 | [0.124, 0.445] | -0.1441 | [-0.527, 0.340] |
| 3.0 | 0.5 | +0.2488 | [0.120, 0.393] | -0.1418 | [-0.483, 0.262] |
| 4.0 | 1.0 | +0.2446 | [0.097, 0.410] | -0.1756 | [-0.547, 0.287] |
| 2.5 | 0.5 | +0.2278 | [0.116, 0.349] | -0.1465 | [-0.461, 0.219] |
| 4.0 | 1.5 | +0.2258 | [0.079, 0.391] | -0.2027 | [-0.564, 0.248] |
| **3.0** | **1.0 (current live constants)** | **+0.2161** | [0.089, 0.356] | **-0.1753** | [-0.504, 0.211] |
| ... | ... | ... | ... | ... | ... |
| 1.0 | 1.0 | +0.0745 | [0.022, 0.124] | -0.1264 | [-0.290, 0.052] |
| 1.0 | 1.5 | +0.0700 | [0.009, 0.134] | -0.1580 | [-0.318, 0.012] |

Full 24-row table: `trigger_sweep_results.csv`.

**Every one of the 24 cells is net-negative in the holdout.** Point estimates range
from -0.083 (trigger=1.0, trail=0.5) to -0.237 (trigger=2.5, trail=2.0) — there is no
sign flip anywhere on the grid. The current live constants (3.0, 1.0) sit almost
exactly in the middle of that range at -0.175.

Top 5 in-sample cells vs. their holdout numbers (over-fitting check):

| trigger_r | trail_atr | mean R in-sample | mean R holdout |
|---:|---:|---:|---:|
| 4.0 | 0.5 | +0.2748 | -0.1441 |
| 3.0 | 0.5 | +0.2488 | -0.1418 |
| 4.0 | 1.0 | +0.2446 | -0.1756 |
| 2.5 | 0.5 | +0.2278 | -0.1465 |
| 4.0 | 1.5 | +0.2258 | -0.2027 |

Every top-5 in-sample cell is negative OOS — textbook over-fitting shape: the ranking
that looks best in-sample carries no information about the holdout.

"Beats the current benchmark cell (3.0/1.0) in both periods" (i.e. higher mean_ins
AND less-negative mean_oos): 3 of 24 cells technically qualify —
(4.0, 0.5): +0.2748 ins / -0.1441 oos;
(3.0, 0.5): +0.2488 ins / -0.1418 oos;
(2.5, 0.5): +0.2278 ins / -0.1465 oos.
**These are NOT wins.** All three are still net-negative in the holdout — "less bad
than -0.175" is not "profitable." Their holdout CIs ([-0.53, 0.34], [-0.48, 0.26],
[-0.46, 0.22]) comfortably contain the current benchmark's own holdout mean, so this
is statistical noise around "everything on this grid loses money right now," not
evidence that a tighter trail (0.5 ATR) is a real improvement.

Profit concentration (`top1pct_*` in `trigger_sweep_results.csv`): in-sample top-1%
share ranges ~33-88% of total profit depending on the cell (tighter trails / lower
triggers concentrate more, since the trade with the single biggest excursion dominates
a smaller total). Since holdout totals are net-negative, "share of profit from the top
1%" is not a coherent statistic there (dividing a positive tail sum by a negative
total) — reported in the CSV for completeness but not used as a decision criterion for
the holdout.

### Verdict — LEAD 2 is a negative result

**No (trigger_r, trail_atr) pair rescues the holdout.** The arming/trail geometry is
not the reason this arm is currently underperforming out-of-sample; the signal edge
itself is negative in this window (consistent with the existing repo finding "Edge
negative OOS since May 2026" — refitting/retuning does not fix it, matching that note's
warning that fit↔OOS correlation is ~0.036). Retuning `trigger_r`/`trail_atr` is not
worth doing right now — there is nothing on this 24-point grid, including the
untested corner nearest the "wide-stop" intuition in the task brief (trigger=1.0,
i.e. arming at ~3 ATR excursion to match a 3-ATR stop), that turns the holdout
positive.

---

## Overall verdict

**Nothing tested here beats the current s3_a1 trail (trigger=3.0R / trail=1.0 ATR) in
both periods.** LEAD 1 (scale-out+breakeven) underperforms the trail net-of-cost even
in-sample, on the live per-symbol-config universe available. LEAD 2 (trigger/trail
tuning for the global rr=10/atr=3 arm) cannot produce a single cell that is net-positive
in the untouched holdout — the whole 24-point grid, including the live constants
themselves, loses money after 2026-05-25. That is a signal-edge problem, not an
exit-rule problem, and no exit-rule change in this search closes it.

Recommendation: keep the live s3_a1 trail (trigger=3.0, trail=1.0 ATR) exactly as
configured. Do not add scale-out or breakeven. Do not retune the trail constants off
this grid — the in-sample ranking has no OOS predictive value here (top-5 in-sample
cells are 5/5 negative OOS).

## Caveats

- Both leads work at the per-trade R level with a flat cost model, not a portfolio/DD
  replay — "ROI/DD" as asked for in the task is not directly computed; the mean-net-R
  results make it clear no arm changes the return sign, so a DD-only case for either
  lead was not pursued.
- LEAD 1's holdout is 7 weekly blocks (2,286 trades); LEAD 2's is 9 weekly blocks
  (2,333 trades). Both are thin for block-bootstrap CIs — treat interval widths as a
  floor on real uncertainty, not a precise measurement.
- LEAD 1's benchmark ("trail") column is the live per-symbol-fitted-config trail, not
  the global rr=10/atr_mult=3.0 arm that is the task's stated current-best. A
  scale-out+BE universe built specifically against the global arm (would need a 5m-bar
  walk, `cache_5m`, for all 277 symbols) was not built — out of scope for the compute
  budget here, and LEAD 1's own in-sample numbers already rule it out without needing
  that stronger benchmark.
- Cost model for LEAD 1 (flat `0.00242/stop_frac` on the whole trade) is conservative
  against scale-out specifically (see note above) — real-world scale-out costs would be
  somewhat lower, but not by enough to flip the verdict (the in-sample gap to the
  no-BE control alone is ~0.07R, larger than any plausible cost adjustment favoring BE
  specifically over no-BE, since the cost model treats both scale-out arms identically).

## Files in this directory

- `build_trigger_sweep.py` — LEAD 2 signal/exit builder (standalone copy, does not
  import or edit `risk_study/build_universe_trail_helpers.py` or `build_global_trail.py`
  in place)
- `trigger_sweep_universe.parquet` — 32,054 signals x 24 trigger/trail combo R columns
- `analyze_trigger_sweep.py` / `trigger_sweep_results.csv` — LEAD 2 split-sample results
- `analyze_scaleout.py` / `scaleout_arm_results.csv` / `scaleout_arm_trades.parquet` —
  LEAD 1 split-sample results (reads the repo-root `scaleout_universe.parquet`, not
  copied — it's 24,931 rows and was not modified)
- `exit_results.csv` — combined table (both leads, one row per cell/arm) — the
  requested top-level deliverable
- `EXIT.md` — this file
