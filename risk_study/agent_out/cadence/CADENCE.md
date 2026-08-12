# Re-optimisation cadence study — how often should (rr, atr_mult) be refit?

**Answer: never (or as close to never as operationally possible).** Every cadence tested
(12mo / 6mo / 3mo / 1mo) that touches the GLOBAL (rr, atr_mult) parameter underperforms
simply picking one pair after the 6-month burn-in and holding it for the next ~2.5 years.
Refitting PER-SYMBOL x DIV-TYPE (what the bot's `config.yaml` walk-forward process does
today) is worse than the global choice at every cadence, and gets *worse*, not better,
the more strictly you require sample size before trusting a per-symbol pick.

- **Recommended cadence: NEVER.**
- **What to refit: nothing, ideally.** If forced to pick one lever, it is the GLOBAL
  (rr, atr_mult) pair, picked once — not a per-symbol/per-div-type table, and not on a
  schedule.
- **The number that justifies it:** of the 47 (cadence × lookback × rule) cells tested
  against the S4 static benchmark (rr=10, atr_mult=3.0, fixed for all history), only
  **one** beats it with a 95% CI excluding zero: `never / S3_shrunk_thr20`, **+0.0062
  net-R/trade** (95% weekly-block-bootstrap CI **[+0.0028, +0.0106]**, n=40,990 trades).
  Every refit cadence of the global rule (`S1_global`) underperforms `never`, and two
  cells are **significantly worse** than never refitting at all: `3m/anchored/S1_global`
  (**−0.069 R/trade**, CI [−0.134, −0.0086]) and `3m/rolling12/S1_global` (**−0.111
  R/trade**, CI [−0.190, −0.038]).
- **Churn (the noise-fitting signature):** refitting the GLOBAL pair annually changes the
  chosen (rr, atr_mult) **100% of the time** (both anchored and rolling-12mo lookback).
  Quarterly refits still churn 45–70% of the time; even monthly refits change the pick
  19–42% of months. Per-symbol/div-type tables churn 45–98% of pairs at every refit. None
  of this churn buys any OOS improvement — most of it actively costs R.

This matches the prior finding in this repo (`selection-transfer-measured`: per-symbol
train→test Spearman rho ≈ 0.064 vs global rho ≈ 0.619) and extends it: it is not just that
per-symbol selection *transfers weakly* — refitting it on **any** cadence loses money
relative to not touching it, and the failure mode gets worse, not better, as the
minimum-trade bar is raised (more "rigorous" per-symbol selection is more destructive,
because it's fitting to less and less data per bucket).

---

## Method

- Data: `risk_study/grid_inuni.parquet`, 420,953 rows already restricted to
  `in_universe==True`. CHOP gate `chop_bos < 52` (drops NaN too) → 183,084 signal rows.
  Trimmed last 21 days of entries for right-censoring (cutoff `2026-07-04`). Period
  used: 2023-06-01 → 2026-07-04.
- Cost: `net_r = r_<rr> − 0.00242 / stop_frac`, applied per (signal, atr_mult, rr) row.
  Long format: one row per (signal, atr_mult, rr) with a resolved `net_r` (unresolved /
  right-censored rows dropped) → 914,395 rows across the 20 (atr_mult, rr) pairs.
- 6-month burn-in: first refit / OOS start `T0 = 2023-12-01`. All rules trade
  `T0 → 2026-07-04` OOS, ~40,900–41,700 trades depending on rule (small differences come
  from which (grp, pair) combinations clear their trade-count threshold in a given
  window).
- **Refit simulation** — at each refit date `t`, parameters are selected using only
  `entry_time < t`, then traded from `t` to the next refit date; OOS segments are
  concatenated into one track record per (cadence, lookback, rule). No lookahead.
- **Cadences:** `never` (1 refit, held forever), `12m`, `6m`, `3m`, `1m`. Monthly was kept
  in scope — the full grid (5 cadences × 2 lookbacks × 4 rule variants, ~914k-row long
  table) ran in **32 seconds** once the per-group selection loop was vectorised via a
  merge instead of a Python loop over ~1,100 (symbol, div_type) groups (the latter is
  almost certainly what made the earlier attempt die — a 3+ minute test run with the
  un-vectorised loop was killed and rewritten before the real run).
- **Lookback:** anchored (all history so far) vs rolling 12 months.
- **Selection rules:**
  - `S1_global` — single best (rr, atr_mult) by trailing net R, min 100 trades.
  - `S2_persym_thrN` — best (rr, atr_mult) per (symbol, div_type), min N trades
    (N = 20 and 50 both reported). No fallback: groups below threshold simply don't
    trade that period.
  - `S3_shrunk_thrN` — as S2, but groups below threshold (or never seen in training)
    fall back to the global S1 pick.
  - `S4_static` — rr=10, atr_mult=3.0, fixed for all history, never refit. This is the
    benchmark the owner is considering standardising on.
- **Churn:** fraction of (group's) selected pair that differs from the previous refit's
  selection, averaged over all refit transitions (S1: single pair; S2/S3: averaged over
  groups common to both consecutive selections). Undefined (NaN) for `never` (only one
  refit, no transition).
- **Paired comparison vs S4:** weekly block bootstrap (2,000 resamples, blocks = ISO
  weeks of `entry_time`) on the difference in trade-weighted mean net R between each
  (cadence, lookback, rule) cell and S4, over the shared OOS window. Reported as point
  diff + 95% CI.

## Headline table — mean net R/trade by rule × cadence

S4 static benchmark: **0.0735 R/trade**, n=40,983.

| rule | lookback | never | 12m | 6m | 3m | 1m |
|---|---|---|---|---|---|---|
| S1_global | anchored | **0.0735** | 0.0660 | 0.0630 | 0.0045 | 0.0510 |
| S1_global | rolling12 | **0.0735** | 0.0461 | 0.0536 | −0.0374 | 0.0532 |
| S2_persym_thr20 | anchored | −0.0046 | −0.0161 | 0.0414 | 0.0273 | 0.0452 |
| S2_persym_thr20 | rolling12 | −0.0046 | 0.0296 | 0.0508 | 0.0328 | 0.0479 |
| S2_persym_thr50 | anchored | n/a (0 trades) | −0.0418 | −0.0834 | −0.0660 | −0.0194 |
| S2_persym_thr50 | rolling12 | n/a (0 trades) | −0.4285 | −0.4417 | −0.6042 | −1.1649 |
| S3_shrunk_thr20 | anchored | **0.0797** | 0.0534 | 0.0755 | 0.0375 | 0.0671 |
| S3_shrunk_thr20 | rolling12 | **0.0797** | 0.0576 | 0.0673 | 0.0019 | 0.0522 |
| S3_shrunk_thr50 | anchored | 0.0735 | 0.0644 | 0.0659 | 0.0122 | 0.0568 |
| S3_shrunk_thr50 | rolling12 | 0.0735 | 0.0462 | 0.0538 | −0.0372 | 0.0532 |

Reading it: in every rule × lookback row, `never` is the best or tied-best cell. The one
exception (`S2_persym_thr20`, where `never` looks worst) is a sample-size artefact, not a
refit benefit — `never`'s one-shot training window only had 6 months of data, so only 554
trades cleared the 20-trade-per-(symbol,div_type) bar; later cadences accumulate more
training history before their first refit and so field more (still-mediocre) per-symbol
bets. Even at its best (0.0508, `6m/rolling12`), `S2_persym_thr20` never approaches the
static benchmark (0.0735), let alone `S3_shrunk`'s never-refit figure (0.0797).

## Churn table — fraction of pairs changed per refit

| rule | lookback | 12m | 6m | 3m | 1m |
|---|---|---|---|---|---|
| S1_global | anchored | **1.00** | 0.60 | 0.60 | 0.29 |
| S1_global | rolling12 | **1.00** | 0.80 | 0.70 | 0.42 |
| S2_persym_thr20 | anchored | 0.49 | 0.45 | 0.27 | 0.14 |
| S2_persym_thr20 | rolling12 | 0.83 | 0.70 | 0.48 | 0.25 |
| S3_shrunk_thr20 | anchored | 0.81 | 0.55 | 0.45 | 0.20 |
| S3_shrunk_thr20 | rolling12 | 0.93 | 0.79 | 0.66 | 0.39 |
| S3_shrunk_thr50 | anchored | 0.98 | 0.60 | 0.56 | 0.26 |
| S3_shrunk_thr50 | rolling12 | 1.00 | 0.80 | 0.70 | 0.42 |

An annual refit of the single global parameter picks a **different** (rr, atr_mult) pair
every single time it's run, over ~2.5 years of refits — with a net cost, not benefit,
relative to picking once. That is the noise-fitting signature the study was designed to
surface: high churn, no matching OOS gain, and in the 3-month global case, a
statistically significant loss.

## Paired vs-S4 significance (selected cells)

| cell | diff vs S4 (R/trade) | 95% CI | verdict |
|---|---|---|---|
| never / anchored / S3_shrunk_thr20 | **+0.0062** | [+0.0028, +0.0106] | significantly better, but small and fragile (see caveat) |
| never / rolling12 / S3_shrunk_thr20 | +0.0062 | [+0.0028, +0.0103] | same cell, lookback is a no-op with 1 refit |
| 3m / anchored / S1_global | −0.0689 | [−0.1340, −0.0086] | significantly **worse** |
| 3m / rolling12 / S1_global | −0.1109 | [−0.1895, −0.0380] | significantly **worse** |
| everything else | mixed sign, small | CI spans zero | no evidence of benefit either way |

## Does recency help? (anchored vs rolling-12mo lookback)

No consistent pattern. Rolling-12mo sometimes beats anchored at the same cadence (e.g.
`S2_persym_thr20` improves at every cadence under rolling-12mo vs anchored — but is still
below the static benchmark either way) and sometimes loses badly (`3m/rolling12/S1_global`
is the worst cell in the whole global-rule table, −0.037, while `3m/anchored` is 0.0045 —
both bad, rolling is worse here). At `S2_persym_thr50`, rolling-12mo is catastrophic
(−0.43 to −1.16 R/trade on 10–32 trades) because a 12-month rolling window rarely
accumulates 50 trades for a single (symbol, div_type, atr_mult, rr) cell, so the "best"
pick is a handful of trades with no real basis for selection — the extreme case of the
same overfitting mechanism. Recency is not a reliable substitute for more data; if
anything it makes the small-sample problem at the per-symbol level worse.

## Caveat on the one positive result

`never/S3_shrunk_thr20`'s +0.0062 R/trade edge is statistically significant in this
bootstrap, but it rests on parameter picks made from a single 6-month training window
(Dec 2023) for a small minority of (symbol, div_type) groups that happened to clear 20
trades that early — the vast majority of groups still fall back to the same global pick
as S4. Given the previously measured per-symbol train→test rho of 0.064, this is a
plausible-but-fragile improvement, not a strategy to lean on. It is not a case for
building or maintaining a per-symbol override table — it's a case for, at most, a single
one-time look that may or may not be worth the added operational complexity. The safe,
well-supported recommendation is simply: **keep the fixed global (rr, atr_mult) pair,
and don't refit it.**

## Bottom line for the owner

The walk-forward re-optimisation process currently used to populate `config.yaml`'s
per-symbol `configs` blocks (§13 of CLAUDE.md) has no empirical support at **any**
cadence tested — monthly, quarterly, semi-annual or annual all underperform a single
global (rr, atr_mult) choice held forever, and the per-symbol variant of that process is
worse still, with churn of 45–98% per refit and often deeply negative OOS returns at
higher trade-count thresholds. If the bot moves to the static rr=10/atr_mult=3.0 control
under consideration, this study finds no evidence that re-running the optimiser on a
schedule would improve on that — freeze it.

Full numeric detail (overall + per-year breakdown, churn, bootstrap CI, per cell) is in
`cadence_results.csv`.
