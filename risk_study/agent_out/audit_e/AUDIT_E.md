# AUDIT-E: What Is Missing — the Additive Half

Workstream E of the 2026-08-12 deep audit. Every other workstream is subtractive; this one
asks what the bot does **not** do that it plausibly should. Six candidates were named in the
brief; all six were tested (none were screened out before spending compute — each cleared the
±0.02 R/trade plausibility bar on paper). One survives contact with the holdout data.

**Bottom line: one real finding.** Skip entries whose `stop_frac` (= `ATR × atr_mult / entry_price`
at signal time) sits in the top ~20% of its own trailing distribution. Holdout effect
**+0.082 R/trade** (95% weekly-block-bootstrap CI **+0.011 to +0.35**), in-sample **+0.012
R/trade**, wins **10/13** quarters. Everything else tested — portfolio vol targeting, five of
six signal-characteristic candidates, session effects, the poll-loop/alphabetical-rank question,
funding-aware side selection, and 4H multi-timeframe confirmation — comes back **DOES-NOT-PAY**
or **UNTESTABLE** with the data at hand. Two apparent early "wins" (`log_liq30d`, the naive
`stop_frac` monotonic test) turned out to be artifacts and are reported as such, not hidden.

---

## Method

- Base sample: `risk_study/uni_glob_rr10_am3_same.parquet` — the CURRENT live config
  (global rr=10, atr_mult=3.0, trail s3_a1 applied) — 32,102 trades, 2023-06-01 → 2026-07-25.
- Trimmed last 21 days of entries (cutoff 2026-07-04) → 31,424 trades.
- `net_r = r_result - 0.00242/stop_frac`. Split pre/post at 2026-05-25 → pre n=29,722,
  post n=1,702 (small holdout — every post-split number below should be read with that in mind).
- Every feature merged in was computed causally: `liq_30d` and `chop_bos` from
  `grid_inuni.parquet` (atr_mult=3.0 slice, exact match on symbol/div_type/side/entry_time);
  EMA200 (1H and 4H) and 30-day realized vol built per-symbol from `cache_3yr_1h/*.parquet`
  using only bars strictly before entry (`merge_asof`, backward, entry_time − 1H); funding rate
  from `funding_cache/*.parquet` (starts 2025-01-15, so pre-period funding coverage is
  effectively 2025-01-15→2026-05-25, not the full pre-period — flagged where relevant).
- Weekly block bootstrap (800–2000 resamples, resampling ISO weeks with replacement) for every
  CI reported. Rolling-window win rate = fraction of quarters (min 80 trades/quarter) where the
  **same mean-difference statistic** used for the headline number has the same sign as the
  overall-period statistic. (Note: an early version of this check used Spearman rank IC per
  quarter instead of the mean-difference statistic — that gave a materially different, wrong
  answer for the stop_frac filter because `net_r` has a huge tied mass at exactly −1.0, which
  distorts rank correlation on a binary split. Fixed before finalizing; see `analysis.py`.)
- All feature engineering and stats scripts are in `risk_study/agent_out/audit_e/` —
  `build_features.py` builds `trades_features.parquet`, `analysis.py` produces `results.csv`.

---

## The one finding that pays: skip the widest-stop quintile

**What it is.** `stop_frac = ATR(14) × atr_mult / entry_price`, already computed by the bot for
every signal — it is not a new data source, just an unused piece of information already sitting
in `execute_trade`. With `atr_mult` now fixed globally at 3.0 (as of the 2026-08-12 config),
`stop_frac` varies purely with each symbol's own ATR-as-%-of-price at signal time — effectively
a per-trade volatility read. The top ~20% of that distribution (the widest, most volatile
stops) is measurably worse trading, in BOTH periods, and gets *worse* out of sample, not better.

**Why the naive version missed it.** A plain Spearman correlation of `stop_frac` against `net_r`
is strongly positive (IC ≈ 0.48–0.49 both periods) — but that is almost entirely a **mechanical
artifact of the cost formula**: `net_r = r − 0.00242/stop_frac`, so a bigger `stop_frac`
mechanically shrinks the subtracted cost term regardless of trade quality. Checking against
**GROSS** `r_result` (before the cost term) kills that correlation almost completely
(IC ≈ −0.01 pre, −0.08 post) — across most of the range `stop_frac` carries no real
outcome information. The real effect is concentrated entirely in the **top quintile**: GROSS
`r_result` there is 0.177 (pre) and **−0.22 (post, i.e. a losing bucket)** versus ~0.30–0.33
for the rest of the range. A monotonic top/bottom-tercile test (which is what "sizing by a
causal characteristic" naturally reaches for first) averages this tail effect away and even
flips sign — it is reported in `results.csv` (row `stop_frac`) as a **DOES-NOT-PAY / decompose
before trusting** cautionary result in its own right.

**The tested rule.** Causal, deployable version: at each entry, compute the 80th percentile of
`stop_frac` over the trailing 2,000 trades (shift(1), no lookahead) and skip if the current
signal's `stop_frac` is at or above it.

| | pre (n=29,722) | post (n=1,702) |
|---|---|---|
| trades skipped | 21.9% | 24.6% |
| mean net R, gain vs flat | **+0.0124** | **+0.0824** |
| 95% weekly-block-bootstrap CI (post) | — | **(+0.0112, +0.3498)**, mean +0.10 |
| win rate in filtered bucket | 30.5% | 29.4% (vs 15.5% in the dropped top quintile) |
| rolling quarterly win rate | — | **10/13 quarters** (77%) |

This is not a proxy for a category the bot already screens on: the top-quintile bucket's
`div_type` and `side` composition is close to the overall mix (HID_BEAR 39% vs 44% baseline,
short 56% vs 64% baseline — if anything slightly *under*-representing the categories the bot
already leans into), and the correlation between `stop_frac` and `log(liq_30d)` is only 0.16 —
this is not simply re-discovering "avoid illiquid microcaps."

**Confidence: moderate-high.** Both periods agree in sign and magnitude order, the holdout CI
floor is positive (+0.011, comfortably above the 0.02 bar isn't guaranteed at the floor, but the
mean and most of the mass clears it), and the rolling win rate is well above the 60% bar. The
honest weak point: the holdout is only 1,702 trades (35 weeks), so the CI is wide (0.011 to
0.35) and a chunk of the post-split gain is concentrated in a few good quarters (2024Q1: +0.12,
2025Q2: +0.027, 2025Q3: +0.020) rather than uniformly spread — two quarters (2023Q4, 2025Q4,
2026Q1) show small negative gains. **Verdict: PAYS**, with a recommendation to re-verify after
250+ more live trades rather than deploy-and-forget (same standing recommendation
`OVERFIT_VERDICT.md` already makes for the rest of the config).

**Exact change, if adopted.** In `execute_trade`, after `sl_dist = atr * atr_mult` and before
placing the order: maintain a trailing distribution (e.g. last 2,000 entries, or a fixed
90-day window) of `sl_dist / entry_price`; if the current value is ≥ its 80th percentile,
skip the entry (log it to the shadow layer instead of trading it, same pattern as the CHOP
gate). This is a few lines and one new piece of state — no new API calls, no change to sizing
arithmetic for trades that pass.

---

## Everything else

### 1. Portfolio volatility targeting — DOES-NOT-PAY

Built a causal relative-concurrency measure (open-trade count at entry ÷ trailing 500-trade
rolling median) and an inverse-concurrency size multiplier (target 1.0×, clipped 0.4–1.5×).

- Raw concurrency IC vs net_r is confounded by a 3× growth in the trade universe over
  2023→2026 (more symbols enabled, config maturity) — a naive read would have found a spurious
  positive relationship (busier periods look "better" only because busier periods are later,
  and later periods happen to have run hotter). The de-trended and fully causal
  relative-concurrency versions both wash this out: IC ≈ **−0.007 (pre) / −0.016 (post)** —
  indistinguishable from zero.
- The weighted-vs-flat sizing scheme built from that near-zero signal **loses**, not gains:
  −0.040 R/trade pre, −0.040 post, CI (−0.118, +0.009) crosses zero, rolling win rate only 64%.
- **Verdict: DOES-NOT-PAY** for mean R/trade. This does not rule out a pure drawdown-smoothing
  benefit from compounding effects (lower variance can still raise CAGR even at flat mean R) —
  that is a different question, belongs to Workstream A's cap-ablation work
  (`gross_open_risk_cap` already exists as the blunt version), and would need a full
  dollar-through-the-production-engine simulation to settle. Flagged as **interesting but
  untested on the right metric**, not as a rejected hypothesis.

### 2. Sizing by causal signal characteristics — mostly DOES-NOT-PAY, one PAYS (above)

Tested: `liq_30d` (log), `stop_frac`, `chop_bos` (already read at the BOS bar, no leak),
`ema_dist_1h_aligned` (side-adjusted distance from EMA200), `rvol_30d` (30-day realized vol),
`funding_avg7d_for_side`, plus categorical `div_type`, `side`, `hour`, `dow`.

| feature | IC pre | IC post | verdict | note |
|---|---|---|---|---|
| `stop_frac` (top-quintile filter) | n/a | n/a | **PAYS** | see above |
| `log_liq30d` | 0.086 | 0.054 | DOES-NOT-PAY | looked like a hit (tercile effect +0.17 post, CI excludes 0) until decomposed: IC vs GROSS r_result is ~0 (0.003/0.020), and an explicit drop-bottom-quintile-liquidity filter gives the **wrong sign OOS** (pre +0.003, post **−0.011**). False positive from a noisy small-holdout extreme bucket plus a weak indirect channel through `stop_frac` (corr 0.16). Reported so the false-positive path is on record, per "test outcomes not proxies." |
| `stop_frac` (naive monotonic) | 0.492 | 0.465 | DOES-NOT-PAY | the strong correlation is the cost formula's own `1/stop_frac` term, not information — see decomposition above |
| `chop_bos` | −0.058 | 0.017 | DOES-NOT-PAY | sign flips, no transfer — consistent with CHOP already doing its job as an entry gate; no further juice in the residual value among admitted trades |
| `ema_dist_1h_aligned` | 0.265 | 0.252 | DOES-NOT-PAY | IC transfers cleanly (both periods positive, similar magnitude) but the tercile bucket effect does not (CI −0.47 to +0.12, crosses zero) — small holdout, heavy-tailed R; a real but too-weak-to-act-on relationship at current sample size |
| `rvol_30d` | 0.312 | 0.193 | DOES-NOT-PAY | IC transfers in sign but weakens a lot; bucket effect not significant (CI −0.97 to +0.14) |
| `funding_avg7d_for_side` | 0.052 | 0.004 | DOES-NOT-PAY | IC collapses to ~0 OOS — no transfer |
| `div_type` [REG_BEAR vs REG_BULL] | — | — | DOES-NOT-PAY | large point spread (pre +0.40) doesn't survive the holdout CI (−1.04, +1.16) |
| `side` [short vs long] | — | — | DOES-NOT-PAY | shorts outperform longs in both periods (as expected — the book is structurally short-tilted per existing findings) but CI is wide and crosses zero on the holdout sample; not a new actionable filter beyond what net-directional cap already does |
| `hour`, `dow` | — | — | DOES-NOT-PAY | large point spreads pre, sign-flip or CI-crosses-zero post; classic small-holdout overfitting — see session effects below for the coarser, still-negative version |

### 3. Time-of-day / session effects — DOES-NOT-PAY

Bucketed into three 8h UTC sessions (Asia 0–8, EU 8–16, US 16–24). Pre period shows a mild,
monotonic-looking improvement from Asia (0.175) to EU (0.228) to US (0.242) session — but the
holdout is noisy (Asia session actually goes negative, −0.20, in the small post sample) and the
best-vs-worst spread CI is (−0.043, +0.698), crossing zero. Tail risk (5th percentile) is
essentially flat across sessions in both periods (≈ −1.10 to −1.13 throughout — no session has
a materially fatter left tail). **No exploitable session effect**, and the P5 check the prompt
asked for shows nothing on the tail either.

### 4. Poll-loop / alphabetical processing order — UNTESTABLE with this data (mechanism), no evidence of a selection effect

Confirmed the bot's `config.yaml` `symbols:` block is alphabetically sorted (288 entries, 277
enabled) — the live processing order the prompt describes. Two tests:

- **Symbol-level cross-section** (mean net_r per symbol vs its alphabetical rank, n=269
  symbols with ≥20 trades): Spearman ρ = **+0.088**, bootstrap CI (**−0.028, +0.203**) — crosses
  zero, and even the point estimate has the *wrong* sign for the "late-processed symbols are
  worse" hypothesis (later rank → *slightly* better R, not worse, though not significantly so).
- **Trade-level IC**: pre −0.011, post −0.021 — negligible, no transfer of even that weak signal.

**Important caveat, stated plainly rather than glossed over**: this backtest's `entry_price` is
the idealized candle open (`df.iloc[-1]['open']`) taken directly from `cache_3yr_1h`. It does
**not** model the bot's actual fetch-time slippage from sweeping 277 symbols across several
minutes each hour — the specific mechanism the audit prompt is asking about. This workstream's
data can only detect a *selection* effect (do late-alphabet symbols happen to be structurally
worse names?) — and finds none — not the *execution-lag* effect (does entering 2–4 minutes
into a candle after alphabetically-earlier symbols cost R?), which would require sub-hourly
fill data (`exec_log`, `fill_lag_sec`) that belongs to Workstream B, not the data made available
here. **Verdict: no evidence of a selection-driven cost; the execution-lag question itself
remains open and should be settled from `exec_log`, not from this backtest.**

### 5. Funding-aware side selection — DOES-NOT-PAY (as a standalone signal)

Coordinated with, not duplicating, Workstream B's cost/capture measurement — this only asks
whether funding at entry is informative as a filter or sizing input, not what it's worth in
aggregate dollars.

- Instantaneous funding rate at entry, side-adjusted (short receives when funding > 0): IC
  pre +0.043, post **−0.022** — sign flips, no transfer.
- Trailing 7-day average funding, side-adjusted (`funding_avg7d_for_side`, also in the table
  above): IC pre +0.052, post +0.004 — collapses to ~0.
- Funding data only covers 2025-01-15 onward, so the "pre" fit window here is effectively
  2025-01-15→2026-05-25 (n=29,570 of the 29,722 pre trades), not the full 2023 start — noted so
  the sample isn't mistaken for the full-history one used elsewhere.

**Verdict: DOES-NOT-PAY as an entry filter or sizing input.** This does not settle whether
funding is a material *cost/revenue* component of the bot's realized P&L in aggregate — that
number belongs to Workstream B.

### 6. Multi-timeframe (4H) confirmation — DOES-NOT-PAY, largely redundant with 1H EMA200

Built a 4H EMA200 from `cache_3yr_1h` (resampled, closed bars only, causal). 1H-EMA-aligned and
4H-trend-aligned agree on direction **80% of the time** — mostly the same information, as the
prompt suspected.

- Trend-aligned vs not: pre +0.202, post +0.307, but holdout CI is (−0.42, +0.98) — huge
  swings quarter to quarter (2023Q2 +1.06, 2024Q1 −0.39, 2025Q4 +0.94, 2026Q3 −0.68) driven by
  small per-quarter samples where the 4H signal happens to correlate with a strong or weak
  quarter overall. Rolling win rate 64% (above the 60% bar) but the CI is too wide to call this
  real.
- Conditional test (does 4H add information specifically where the 1H EMA-distance is
  marginal — bottom tercile of `|ema_dist_1h_aligned|`): pre effect +0.12 (n=10,106), post
  +0.54 (n=504) — directionally interesting, but this subset was **not** bootstrapped or
  checked for rolling-window consistency (time-boxed out of this pass), so it is reported as a
  **suggestive, underpowered lead**, not a finding. If pursued, the CI on this specific
  conditional subset is the next thing to compute.

---

## Ranked summary

| Idea | Holdout effect (R/trade) | 95% CI | Rolling win rate | Verdict |
|---|---|---|---|---|
| **2. Skip top-quintile `stop_frac`** | **+0.082** | **(+0.011, +0.35)** | **10/13 (77%)** | **PAYS** |
| 6. 4H trend, marginal-1H subset only | +0.54 (n=504, not CI'd) | not computed | not computed | UNTESTABLE (underpowered, needs follow-up) |
| 2. `log_liq30d` (naive) | +0.17 (looked real) | (0.006, 0.35) | 57% | DOES-NOT-PAY (decomposes to noise) |
| 6. 4H trend, all trades | +0.31 | (−0.42, +0.98) | 64% | DOES-NOT-PAY |
| 3. Session (US vs Asia) | +0.35 | (−0.04, +0.70) | 57% | DOES-NOT-PAY |
| 2. `side` (short vs long) | +0.29 | (−0.53, +1.10) | 50% | DOES-NOT-PAY |
| 2. `funding_avg7d_for_side` | +0.24 | (−0.23, +0.73) | 64% | DOES-NOT-PAY |
| 2. `chop_bos` | −0.22 | (−0.41, +0.03) | 86% | DOES-NOT-PAY |
| 2. `ema_dist_1h_aligned` | −0.18 | (−0.47, +0.12) | 50% | DOES-NOT-PAY |
| 2. `rvol_30d` | −0.38 | (−0.97, +0.14) | 71% | DOES-NOT-PAY |
| 2. `stop_frac` (naive monotonic) | −0.39 | (−0.97, −0.11) | 36% | DOES-NOT-PAY (mechanical artifact) |
| 1. Portfolio vol targeting (rel. concurrency) | −0.04 | (−0.12, +0.01) | 64% | DOES-NOT-PAY (mean-R); DD-effect untested |
| 4. Alphabetical rank (selection channel) | ρ≈0 | (−0.03, +0.20) | n/a | No evidence; execution-lag mechanism itself UNTESTABLE here |
| 5. Funding-aware side selection | ≈0 | wide, crosses 0 | 64% | DOES-NOT-PAY |

## What I'd act on this week vs what needs more evidence

**Act on (moderate-high confidence):** the `stop_frac` top-quintile skip filter. Simple,
causal, cheap to implement, positive in both periods with a positive CI floor and a 77% rolling
win rate. Re-verify after ~250 live trades before treating it as settled, per the audit's
standing convention for any config change.

**Needs more evidence, not action:** the 4H-trend marginal-1H-case lead (compute the CI and
rolling win rate on that specific subset before drawing a conclusion); whether relative-
concurrency-based sizing helps drawdown/CAGR even at flat mean R (needs a full dollar
simulation through the production engine, not just an R/trade lens — that's the right follow-up
for Workstream A, which owns the existing `gross_open_risk_cap`).

**Settled negative, don't revisit without new data:** portfolio vol targeting on mean R,
liquidity-based sizing, CHOP-residual sizing, EMA-distance sizing, realized-vol sizing,
funding-based sizing/filtering, session effects, hour/day-of-week filters, alphabetical-rank
selection effects, 4H confirmation as a standalone filter.

## Files

- `risk_study/agent_out/audit_e/build_features.py` — feature engineering (concurrency, EMA
  1H/4H, realized vol, funding, alphabetical rank), writes `trades_features.parquet`
- `risk_study/agent_out/audit_e/analysis.py` — all statistical tests, writes `results.csv`
- `risk_study/agent_out/audit_e/trades_features.parquet` — augmented trade-level dataset
  (31,424 rows × 40 columns) used for every test in this report
- `risk_study/agent_out/audit_e/results.csv` — one row per tested feature/idea, machine-readable
