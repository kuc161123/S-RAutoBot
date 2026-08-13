# AUDIT-C — the signal layer (pivot width, staleness, MIN_PIVOT_DISTANCE, max_wait, RSI period, 2nd EMA gate)

Agent: AUDIT-C. Scope: `autobot/core/divergence_detector.py`'s free parameters, held against
the CURRENT config (global rr=10, atr_mult=3.0, s3_a1 trail, CHOP gate at `ch[bos]<52`,
EMA gate at BOS, entry at `open[bos+1]`, cost `net R = r - 0.00242/stop_frac`), on the exact
728 `(symbol, div_type)` pairs `config.yaml` trades. Engine: private copies of
`backtest_3yr_walkforward.py`'s signal logic and `build_universe_trail_helpers.resolve_trail`,
parameterised — originals untouched. Code: `build_param_universe.py`, `analyze.py`.

**Run was stopped early on explicit instruction** after 17 of the planned 19 cells finished
(all of pivot width, max_wait, pivot staleness, MIN_PIVOT_DISTANCE, and one RSI cell). Two
cells did **not** run and have no data below: **`rsi_21`** (RSI period 21) and **`dropema`**
(drop the second, BOS-time EMA-200 gate). Everything reported is from the 17 completed cells.

---

## 0. Sanity gate — PASS, plus one free finding

Target: reproducing CURRENT settings (pivot 3,3 / wait 12 / stale 10 / dist 3 / RSI 14) should
land near 32,102 trades, gross +0.2631, net +0.1916 on the full (untrimmed) history.

Reproduced **exactly**: 32,102 trades, gross **+0.2631**, net **+0.1916**, WR 29.45%. Two
independent cells hit this number bit-for-bit:

- `baseline_dist0` — `min_pivot_dist=0`, i.e. my re-implementation of
  `backtest_3yr_walkforward.py`'s `detect_signals` with **no** MIN_PIVOT_DISTANCE enforcement
  (which is what that file actually does — see below).
- `current` — `min_pivot_dist=3`, the value CLAUDE.md and the live
  `autobot/core/divergence_detector.py` both call "current."

**These two are byte-identical** (32,102 trades, same R to 4dp) — MIN_PIVOT_DISTANCE=3 is a
**complete no-op** at pivot width (3,3) on this data: no two pivots that a (3,3) fractal ever
produces as "the two most recent lows/highs found scanning backward" are ever closer than 3
bars apart, so the constraint never binds. `dist_2` (the CLI value below current) is *also*
byte-identical to `current`, confirming the same thing from the other side.

**Free finding, not asked for but worth one line:** `backtest_3yr_walkforward.py`'s
`detect_signals` — the engine behind every number in CLAUDE.md and every archived universe
in `risk_study/` — never enforces MIN_PIVOT_DISTANCE at all (it just takes the next pivot
found scanning backward, full stop), unlike the live `divergence_detector.py`, which does
enforce it (`prev_pli < curr_pli - MIN_PIVOT_DISTANCE`). At pivot width (3,3) this
discrepancy is provably harmless (see above). It would **not** be harmless if pivot width
were ever changed without also porting the constraint faithfully into the backtest — flag
this if anyone touches pivot width in the backtest engine later.

---

## 1. What ran, full-history (untrimmed) numbers

One factor moved at a time from the current baseline (pivot 3,3 / wait 12 / stale 10 / dist 3
/ RSI 14). Full history = 2023-06-01 through the cache's last candle (2026-07-25), **not**
trimmed, **not** split — this is the same accounting as the sanity-gate target above, for a
direct read of "does this knob move the number at all."

| cell | dimension | value | trades | gross R/trade | net R/trade | WR |
|---|---|---|---:|---:|---:|---:|
| **current** | baseline | pivot(3,3) dist=3 stale=10 rsi=14 wait=12 | 32,102 | +0.2631 | **+0.1916** | 29.45% |
| piv_2_2 | pivot_width | (2,2) | 43,495 | +0.2470 | +0.1765 | 29.17% |
| piv_4_4 | pivot_width | (4,4) | 26,322 | +0.2496 | +0.1781 | 28.89% |
| piv_5_5 | pivot_width | (5,5) | 22,808 | +0.2357 | +0.1647 | 28.56% |
| piv_3_2 | pivot_width | (3,2) asym | 34,360 | +0.2395 | +0.1682 | 28.92% |
| piv_2_3 | pivot_width | (2,3) asym | 40,408 | +0.2639 | +0.1932 | 29.59% |
| wait_4 | max_wait | 4 | 23,812 | +0.2434 | +0.1724 | 28.99% |
| wait_8 | max_wait | 8 | 29,035 | +0.2607 | +0.1893 | 29.36% |
| wait_18 | max_wait | 18 | 35,774 | +0.2592 | +0.1876 | 29.34% |
| wait_24 | max_wait | 24 | 38,896 | +0.2603 | +0.1886 | 29.23% |
| stale_5 | pivot_stale | 5 | 28,751 | +0.2660 | +0.1942 | 29.51% |
| stale_15 | pivot_stale | 15 | 33,039 | +0.2639 | +0.1925 | 29.48% |
| stale_20 | pivot_stale | 20 | 33,240 | +0.2644 | +0.1930 | 29.49% |
| dist_2 | min_pivot_dist | 2 | 32,102 | +0.2631 | +0.1916 | 29.45% (= current) |
| dist_5 | min_pivot_dist | 5 | 31,271 | +0.2657 | +0.1941 | 29.46% |
| dist_8 | min_pivot_dist | 8 | 31,967 | +0.2149 | +0.1437 | 28.45% |
| rsi_7 | rsi_period | 7 | 37,013 | +0.1951 | +0.1255 | 28.00% |
| rsi_21 | rsi_period | 21 | — | **NOT RUN** | — | — |
| dropema | bos_ema_gate | drop 2nd gate | — | **NOT RUN** | — | — |

Every completed cell lands within **±0.05 R/trade net** of current except `dist_8` (−0.05)
and `rsi_7` (−0.064) — the two furthest departures from current in their respective
dimensions. Nothing tested beats current by more than **+0.0026 R/trade** (`piv_2_3`,
+0.0932 vs +0.1916 → actually +0.0016; `stale_5` +0.0026) on this full-history, unsplit
accounting.

---

## 2. In-sample vs holdout, weekly block-bootstrap CIs

Trimmed to the last-21-days-of-entries convention (common cutoff across all cells:
**2026-07-04**, so unresolved trades near the data edge don't bias slow/losing trades out
disproportionately), then split at **2026-05-25** (IS < that date, holdout ≥). CIs are
weekly block bootstrap (resample ISO weeks with replacement, 1,500 draws).

Top 5 cells by in-sample net R, current's rank included for reference (current ranks 5th/17):

| cell | n (IS) | net R IS | 95% CI (IS) | n (holdout) | net R holdout | 95% CI (holdout) |
|---|---:|---:|---|---:|---:|---|
| stale_5 | 26,559 | **+0.2195** | [+0.079, +0.367] | 1,578 | +0.035 | [−0.388, +0.483] |
| dist_5 | 28,957 | +0.2174 | [+0.084, +0.356] | 1,669 | +0.046 | [−0.387, +0.491] |
| stale_20 | 30,794 | +0.2170 | [+0.086, +0.354] | 1,746 | +0.039 | [−0.410, +0.512] |
| stale_15 | 30,603 | +0.2165 | [+0.087, +0.353] | 1,738 | +0.042 | [−0.376, +0.504] |
| **current** | 29,722 | +0.2162 | [+0.086, +0.352] | 1,701 | +0.032 | [−0.376, +0.429] |
| dist_2 (=current) | 29,722 | +0.2162 | [+0.083, +0.354] | 1,701 | +0.032 | [−0.370, +0.487] |
| piv_2_3 | 37,267 | +0.2157 | [+0.089, +0.361] | 2,290 | +0.103 | [−0.373, +0.584] |
| wait_24 | 36,005 | +0.2142 | [+0.074, +0.362] | 2,074 | +0.013 | [−0.356, +0.446] |
| wait_8 | 26,930 | +0.2132 | [+0.082, +0.356] | 1,493 | +0.041 | [−0.383, +0.510] |
| wait_18 | 33,124 | +0.2120 | [+0.085, +0.351] | 1,890 | +0.030 | [−0.346, +0.476] |
| piv_2_2 | 40,073 | +0.2049 | [+0.079, +0.343] | 2,515 | −0.014 | [−0.477, +0.427] |
| piv_4_4 | 24,377 | +0.2006 | [+0.065, +0.364] | 1,342 | +0.083 | [−0.375, +0.610] |
| piv_3_2 | 31,738 | +0.1987 | [+0.070, +0.331] | 1,903 | −0.077 | [−0.434, +0.328] |
| wait_4 | 22,098 | +0.1979 | [+0.071, +0.337] | 1,212 | −0.014 | [−0.426, +0.444] |
| piv_5_5 | 21,117 | +0.1855 | [+0.039, +0.336] | 1,138 | +0.141 | [−0.384, +0.735] |
| dist_8 | 29,547 | +0.1664 | [+0.042, +0.287] | 1,736 | +0.027 | [−0.409, +0.484] |
| rsi_7 | 34,332 | +0.1526 | [+0.031, +0.292] | 1,855 | −0.077 | [−0.446, +0.285] |

Full table: `results.csv`.

**Read this carefully — the spread between rank 1 and rank 5 (current) in-sample is 0.0033
R/trade. The IS bootstrap CI half-width is ~0.13-0.15 R/trade. The holdout CI half-width is
~0.4-0.5 R/trade** (holdout has only ~6 weeks of post-2026-05-25 data and ~1,200-2,500 trades
per cell after CHOP). Every single cell's CI overlaps every other cell's CI, in both periods,
completely. **No cell is statistically distinguishable from current at either 95% CI, in
either period.** The apparent "top 5" ranking is real numbers, correctly computed, but it is
ranking inside noise.

**The one pattern worth flagging as a genuine overfitting warning sign, not a candidate
change:** the pivot-width dimension inverts sign between periods. `piv_2_3`, `piv_4_4`, and
especially `piv_5_5` all rank *worse than current in-sample* but *better than current in the
holdout point estimate* (`piv_5_5`: −0.031 IS delta, **+0.109** holdout delta — the largest
holdout gain in the whole sweep, attached to the cell that lost the most in-sample). That is
the textbook shape of noise, not signal — an in-sample loser becoming the largest holdout
winner is exactly what you'd expect from six weeks of holdout data and no real effect, and is
consistent with the pivot_width dimension's transfer rho being **negative** (§3).

---

## 3. Transfer statistic — the deliverable that matters more than any single winner

Four anchored walk-forward folds, train from 2023-06-01 to the boundary (expanding),
test from that boundary to the next (fold 4's test end = the 2026-07-04 trim cutoff, no 5th
boundary exists): F1 train→2024-06-30/test→2024-12-31, F2 →2024-12-31/→2025-06-30,
F3 →2025-06-30/→2025-12-31, F4 →2025-12-31/→cutoff. Within each fold, rank all 17 completed
cells by train-period net R and separately by test-period net R; Spearman correlate the two
rank vectors.

| fold | n cells | spearman rho | p |
|---|---:|---:|---|
| F1 (train→24-06, test 24-06→24-12) | 17 | **−0.045** | 0.86 |
| F2 (train→24-12, test 24-12→25-06) | 17 | **+0.426** | 0.089 |
| F3 (train→25-06, test 25-06→25-12) | 17 | **−0.013** | 0.96 |
| F4 (train→25-12, test 25-12→cutoff) | 17 | **+0.733** | 0.0008 |

Mean fold rho **+0.275**, median **+0.206**, pooled (all 4×17 rank pairs, one Spearman)
**+0.272** (p=0.025).

Per-dimension breakdown (pooled across folds within that dimension's own cells + current):

| dimension | n cells | pooled rho | p |
|---|---:|---:|---|
| min_pivot_dist | 4 | +0.473 | 0.064 |
| max_wait | 5 | +0.275 | 0.241 |
| pivot_stale | 4 | **0.000** | 1.00 |
| pivot_width | 6 | **−0.086** | 0.69 |

**Where this falls on the prior scale:** the `selection-transfer-measured` memory note found
per-symbol RR/ATR selection transfers at rho **0.03–0.06** (noise) while a single global
rr/atr choice transfers at rho **+0.619** (real, load-bearing). Signal-layer parameters land
at **~0.27 mean/pooled** — well below the global-parameter benchmark, and the honest
headline number is not the 0.27 average but the **fold-to-fold spread**: two of four folds
(F1, F3) are statistically indistinguishable from zero, one (F2) is a non-significant
+0.43, and only one fold (F4) shows a strong, significant correlation. A statistic that is
zero in half its folds and only "works" in the fold richest with recent-regime data is not a
stable transfer signal — it is exactly what you'd see if there were **no real train→test
relationship** and F4 happened to catch a lucky alignment (F4's test window is the
last ~7 months of data, which also has the fewest independent weeks and therefore the
highest bootstrap variance per cell — consistent with a spurious high correlation drawn from
a noisy tail). Two of the four per-dimension slices (`pivot_stale` exactly 0, `pivot_width`
negative) show **zero or negative** transfer on their own, which is the single strongest
piece of evidence against retuning: even holding the search space to just six pivot-width
candidates, whichever one wins on training data is not more likely to win on test data than
chance.

---

## 4. Verdict

**Do not touch the signal layer.** Three independent pieces of evidence point the same way:

1. **The full-history, unsplit numbers** (§1) show every completed cell within ±0.05 R/trade
   of current, and the two winners that separate meaningfully from the pack (`dist_8`,
   `rsi_7`) are both **losers**, not winners — the sweep found real degradations at the
   extremes but no real improvement near current.
2. **The IS/holdout comparison** (§2) shows a top-5 spread of 0.003 R/trade against a
   bootstrap CI half-width 40-150x that size. Nothing clears its own noise band, let alone
   current's.
3. **The transfer statistic** (§3) — the deliverable this workstream was built to produce —
   is weak (~0.27, well short of the +0.619 a real global parameter shows) and, more
   importantly, **unstable**: two of four folds and two of four dimensions show ~zero or
   negative transfer. A search this noisy, run against a search space this cheap to explore,
   is close to guaranteed to produce an attractive-looking in-sample "winner" by chance alone
   — which is exactly what §2's top-5 table is.

`min_pivot_dist` and `pivot_stale` are additionally near-total no-ops in the ranges tested
around current (dist 2/3/5 barely move the number; stale 10/15/20 barely move it) — moving
them is not even a live risk, just wasted attention. `dist=8` and `rsi=7` are real, negative,
and worth remembering as "don't" data points, not "consider" ones.

Two cells were not run (`rsi_21`, `dropema`) — the RSI-period sweep is therefore only
half-done (one departure tested, in the direction that hurt) and the 2nd-EMA-gate toggle is
**completely untested**. Both are cheap to finish (~40s each with the existing harness in
this directory) if this workstream is revisited, but nothing in the 17 completed cells
suggests either would change the verdict — the pattern across every other completed
dimension is "current is already near the flat part of a noisy surface."

**This is exactly the CLAUDE.md §12 standing recommendation restated from a different
angle: freeze these parameters and stop spending compute here.** The signal layer is not
"secretly great" or "secretly leaving money on the table" — it is a flat, high-variance
region where the honest answer is that 3 years of data and 728 pairs are not enough signal
to distinguish pivot width 3 from 4, or a 10-bar staleness limit from a 15-bar one. Retuning
it on this evidence would be noise-fitting.

---

## Files

- `build_param_universe.py` — parameterised signal-layer universe builder (private copy,
  originals untouched).
- `build_universe_trail_helpers.py` — verbatim copy of the shared s3_a1 trail resolver.
- `analyze.py` — fold construction, weekly block bootstrap, transfer-rho computation.
- `universes/*.parquet` — 17 built universes (one per completed cell) + `baseline_dist0`
  and `current` (both reproduce the sanity-gate target).
- `results.csv` — full per-cell table (all columns: totals, IS/holdout, all 4 fold
  train/test splits).
- `transfer_fold_rho.csv`, `transfer_dim_rho.csv`, `transfer_summary.json` — the transfer
  statistic broken out by fold and by dimension.
