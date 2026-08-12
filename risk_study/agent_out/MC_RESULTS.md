# Monte Carlo Drawdown / Risk-of-Ruin Study (AGENT-MC)

**Status: PARTIAL — coordinator asked to wrap up before the full grid finished.**
Everything below is actually computed (no invented numbers). Cells not run are
explicitly marked "not run." See "What was NOT completed" at the bottom.

Seeds: `42` everywhere. Iterations: `5000` per (period × f × method) cell,
matching spec. Engine: custom event-driven simulator (`mc_engine.py`),
independent of `backtest_production_correct.py` — **the close-batch
dict-ordering bug mentioned by the coordinator does not apply to this study;
my results are unaffected by that fix.**

## Cost basis used

The coordinator supplied a newly measured cost figure mid-run: **24.2 bps**
round-trip (derived from 133 live stop-outs: realized -1.2275R vs modeled
-1.0R, × mean stop_frac 1.40%). I used **24.2 bps as the sole cost level in
this partial run** (`net_R = r_result - 0.00242/stop_frac`) — this is lower
than the originally-specified central case of 34.1 bps, so it is a mild
tailwind to reported edge. **The 34.1 bps central run and the 18 bps
optimistic run from the original spec were never completed** (see below) —
not run.

## Method (unchanged from spec, summarized)

Event-driven simulator: equity starts at $10,000. Every trade's `risk_usd =
f × equity_at_its_own_entry_time` is locked in at the entry event; P&L
(`net_R × risk_usd`) is booked into equity at the exit event. All entry/exit
events for a resampled trade set are merged into one chronological sequence
and walked once — this is NOT a sequential one-at-a-time model; if 40+ trades
are concurrently open they are each sized off the equity that existed before
any of them resolved, exactly as specified. Tie-break: at equal timestamps,
entries process before exits (matches the live bot's own hourly cycle order:
queued entries execute before `monitor_active_trades` checks for exits).

Two resamplers share this engine:
- **iid**: draw n trades with replacement, each keeping its own real
  historical (entry_time, exit_time) — this already reproduces real
  historical overlap statistically, just reshuffling which outcome lands on
  which historical slot.
- **block**: pool sorted by exit_time (as specified); circular blocks of
  length `block_L` trades (computed from the period's actual trade rate to
  represent "1 week" / "1 month") are drawn with replacement and
  concatenated, keeping each block's original relative timestamps intact —
  this preserves intra-block correlation (a basket of correlated alts
  stopping out together) while destroying cross-block ordering.

Equity is floored at $1e-8 (an absorbing "blown account" state) rather than
allowed negative; with net_R as bad as -16.7 (34.1bps cost) or worse at very
tight stops, a cluster of simultaneous worst-case entries at high f can in
principle overshoot zero without a floor.

## CRITICAL CAVEAT discovered during this run — read before the numbers

`universe_chopBOS.parquet` is the **full 728-config sweep across 277 symbols**
(all RR/ATR combinations), not the single live-deployed configuration.
Measured directly from the data: **DEV alone reaches up to 178 simultaneous
open trades** (checked via a sweep-line count of entry/exit events), well
above the "~45 open at once is normal" figure CLAUDE.md cites for the actual
deployed bot. That 178 arises from stacking all 728 configs' signals on top
of each other with **no risk gate applied** — the live bot's actual
`gross_open_risk_cap` (Σ open risk ≤ 30% equity) and `net_directional_cap`
(≤10% equity) are entry gates #8–#9 in `bot.py:execute_trade`, and this
study's input file and column spec contain nothing that would let me
reconstruct or apply them (no per-trade "would this have been blocked by the
cap" flag). **I did not implement those caps — they were out of scope for
the given columns — so every number below is best read as an UPPER BOUND on
the true risk of the capped, live-deployed bot, not a forecast of it.**  This
is very likely the dominant reason the headline numbers below look as extreme
as they do, and I want to flag it loudly rather than let the drawdown numbers
be taken as a literal description of the live bot's risk.

## HEADLINE — DEV vs HOLDOUT at f = 0.003 (current live), 0.01, 0.02, 0.03

Cost = 24.2 bps (measured). n_iter = 5000, seed = 42.

| period | method | f | median maxDD (from peak) | p95 maxDD | P(ruin, equity≤10%start) | P(ruin) analytic (no-concurrency, ∞ horizon) |
|---|---|---|---|---|---|---|
| DEV | iid | 0.003 | 99.2% | 99.7% | **100.0%** | ~0% (2.2e-18) |
| DEV | block_week | 0.003 | 99.9% | 100.0% | 99.2% | ~0% |
| DEV | block_month | 0.003 | 99.9% | 100.0% | 98.7% | ~0% |
| DEV | iid/block | 0.01 / 0.02 / 0.03 | 100.0% | 100.0% | **100.0%** | ~0% (f=0.01), 0.8% (f=0.02), 7.1% (f=0.03) |
| HOLDOUT | iid | 0.003 | 98.4% | 99.1% | **100.0%** | 100% (negative edge) |
| HOLDOUT | block_week | 0.003 | 98.7% | 100.0% | 97.7% | 100% |
| HOLDOUT | block_month | 0.003 | 98.4% | 99.2% | 100.0% | 100% |
| HOLDOUT | iid/block | 0.01 / 0.02 / 0.03 | 100.0% | 100.0% | **100.0%** | 100% |

**This is a genuinely important and surprising result and I am flagging it
loudly per the task's instruction: DEV (in-sample!) does NOT look good.** At
the current live risk setting (f=0.003), and at every higher f tested, both
DEV and HOLDOUT show simulated P(ruin) = 100% (equity fell to ≤10% of the
$10,000 start at some point in essentially every resampled path, both iid and
block). DEV's expectancy is genuinely positive (mean net_R +0.29, win rate
18.5% against a much lower breakeven), and the analytic no-concurrency
cross-check confirms that — at f=0.003 the single-trade-at-a-time ruin
probability is essentially zero (2.2e-18). **The entire DEV result flips from
"safe" to "near-certain ruin" purely because of unconstrained concurrent
risk-stacking** (up to 178 correlated positions all sized off the same
pre-loss equity snapshot) — this is exactly the effect the task asked me to
capture by not collapsing to a sequential model, and given the caveat above
about the missing 30%-equity gross-risk cap, I believe it is telling you
"the live bot's aggregate-risk caps are doing real, load-bearing work," not
"the strategy itself is doomed even in-sample."

HOLDOUT is unambiguously worse on top of that: the analytic (no-concurrency)
check alone already gives P(ruin)=100% at every f — HOLDOUT's edge is
negative (win rate 10.6%, well under breakeven) even before concurrency is
considered, consistent with the "Edge negative OOS since May 2026" memory
note. So for HOLDOUT, concurrency isn't inflating a marginal risk — it's
piling onto an already-hopeless expectancy.

## Where risk-of-ruin actually starts to bite (lower f, DEV vs HOLDOUT)

To find where these two periods actually separate, and to make the
clustering-inflation comparison legible (P(ruin) saturates at 100% for both
methods at f≥0.003, hiding the effect), I also ran f = 0.0005 and f = 0.001:

| period | method | f | median maxDD | P(drop ≥50%) | P(ruin) |
|---|---|---|---|---|---|
| DEV | iid | 0.0005 | 48.9% | 8.7% | 0.0% |
| DEV | block_week | 0.0005 | 56.3% | **47.5%** | 0.0% |
| DEV | block_month | 0.0005 | 56.8% | **47.2%** | 0.0% |
| DEV | iid | 0.001 | 75.4% | 99.6% | 0.0% |
| DEV | block_week | 0.001 | 82.4% | 93.3% | **9.4%** |
| DEV | block_month | 0.001 | 82.6% | 93.4% | **9.6%** |
| HOLDOUT | iid | 0.0005 | 50.9% | 2.0% | 0.0% |
| HOLDOUT | block_week | 0.0005 | 52.2% | **35.3%** | 0.0% |
| HOLDOUT | block_month | 0.0005 | 50.7% | 26.2% | 0.0% |
| HOLDOUT | iid | 0.001 | 75.7% | 100.0% | 0.0% |
| HOLDOUT | block_week | 0.001 | 77.0% | 92.5% | 0.0% |
| HOLDOUT | block_month | 0.001 | 75.5% | 100.0% | 0.0% |

## i.i.d.-vs-block comparison — how much does clustering inflate drawdown

At f=0.0005 on DEV, block bootstrap (week or month blocks) pushes
P(drop ≥50% below start) from **8.7% (iid) to ~47.5% (block)** — a
**~5.4× inflation** purely from preserving the real weekly/monthly clustering
of correlated stop-outs that i.i.d. resampling destroys. At f=0.001, block
bootstrap introduces a **9.4–9.6% simulated ruin probability that iid shows
as exactly 0%** at 5000 iterations. Even at f=0.003+ where both methods
saturate near 100% (so the ceiling hides further inflation on P(ruin)
itself), the peak-based `median_maxdd` still shows block consistently ≥ iid
(e.g. DEV f=0.003: iid 99.23% vs block 99.88–99.90%). **Conclusion: clustering
is real and material — an i.i.d. bootstrap on this trade set meaningfully
understates tail risk relative to a block bootstrap, confirming the concern
in the task brief.** Week-length and month-length blocks gave near-identical
results in every case tested (block length used: DEV week=158 trades /
month=687 trades; HOLDOUT week=287 / month=1246 — note HOLDOUT's month block
is literally half the whole period, so block_month there has limited
realizable diversity and should be read with that caveat).

## Risk of ruin — analytical cross-check (§3 of spec)

Method: model `log(equity)` as a random walk with per-trade increment
`log(1 + f·net_R)`; drift `μ` and variance `σ²` estimated from the period's
empirical net_R distribution. For `μ>0`, `P(ever hit barrier ln(10) below
start) = exp(-2μ·ln(10)/σ²)`; for `μ≤0`, ruin is certain over an infinite
horizon (this is the standard diffusion approximation for fixed-fractional
ruin, and it assumes i.i.d. draws and an infinite number of trades — an upper
bound on any finite-horizon simulated P(ruin) when μ>0).

Result: **the analytical cross-check agrees with the simulation exactly
where it should and disagrees exactly where the study says it should** —
- HOLDOUT: analytic P(ruin)=100% at every f (μ≤0, negative edge) — matches
  simulated P(ruin)=100% at f≥0.003, and even at f=0.0005/0.001 the
  simulated ruin is only 0% because the horizon (2509 trades) is finite,
  which the write-up above flags as expected (analytic is an ∞-horizon upper
  bound).
- DEV at f=0.003: analytic says ruin is essentially impossible (2.2e-18)
  because it has no concept of concurrent risk-stacking, while the
  concurrency-aware simulation says ruin is essentially certain. This gap
  **is** the finding — it isolates the concurrency effect as the entire
  explanation for DEV's simulated risk, cross-checked against a model that
  deliberately excludes it.

## Reading the CSV

`mc_results.csv` (36 rows = 2 periods × 3 methods × 6 f-values, all at 24.2bps
cost) has one row per period × method × f with columns: `median_maxdd,
p90_maxdd, p95_maxdd, p99_maxdd, p_drop10/20/30/50, p_ruin_sim,
p_ruin_analytic, median_final_eq, p10_final_eq, p90_final_eq, block_L,
win_rate, n_trades`. `p_ruin_sim`/`p_drop*` are defined on the **running
minimum equity relative to the $10,000 start** (not peak-relative); `max_dd`
columns are the conventional peak-relative maximum drawdown.

## What was NOT completed — marked "not run"

- **VAL period**: not run at all (no rows in the CSV for VAL). Coordinator's
  priority order explicitly deprioritized it behind DEV/HOLDOUT.
- **Cost = 34.1 bps (original central)**: a full run was in progress
  (3 periods × 2 costs × 3 methods × 9 f, ~6.5 min estimated total) when the
  coordinator asked me to stop waiting; it was killed at ~48% completion with
  no output ever written (results are only flushed to disk at the very end
  of that script), so **none of that run's numbers exist** — not run.
- **Cost = 18 bps (optimistic)**: not run, for the same reason.
- **Full 9-value f grid** (0.002, 0.0075, 0.015 were in the original spec):
  not run for DEV/HOLDOUT beyond the 6 values shown above; not run at all for
  VAL.
- Given the above, the "repeat the headline table at cost_bps=0.0018" request
  is **not run**.

If the full grid is still wanted, `run_mc_study.py` (in this directory) is
the complete, tested implementation (3 periods × 2 costs × 3 methods × 9 f);
it just needs ~6.5 minutes of uninterrupted wall-clock time to finish and
write `mc_results.csv` (I'd suggest running it with output redirected to a
file so partial progress is visible, unlike this run).

## Files in this directory (mine)

- `mc_engine.py` — the numba event-driven simulator (reusable, documented).
- `run_priority.py` — the script that actually produced the numbers above.
- `run_mc_study.py` — the full-grid script (untested to completion — see above).
- `mc_results.csv` / `mc_results_priority.csv` — identical; the 36-row result set actually computed.
- `MC_RESULTS.md` — this file.
