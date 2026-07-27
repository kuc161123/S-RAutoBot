# Strategy Search — PRE-REGISTRATION

**Committed 2026-07-27, BEFORE any result exists.** This document fixes the candidate list,
the parameter grids, the data partition, the trial count and the pass criteria. It is
committed first so the exercise is falsifiable: anything added after results are seen must
be logged below as a separate, later trial and counted in `N`.

## Why this document exists

The current live config was produced by a search that selected on its own test windows. It
showed **+0.2197 R/trade in-sample (t = +15.66, 73.7% of pairs)** and transferred as
**−0.0437 R/trade out-of-sample (t = −1.09, 38.2% of pairs)** — correlation between
fitting-window and out-of-sample performance was **+0.036**. Testing ~18 strategies and
picking the winner is the same error at a different level of abstraction. Everything below
exists to prevent repeating it.

**The expected outcome is that nothing passes.** That is a valid result and will be reported
as plainly as a winner would be.

---

## 1. Data partition (the firewall)

| segment | range | permitted use |
|---|---|---|
| DESIGN | 2023-06-01 → 2026-02-01 | all fitting, all ranking, all out-of-fold folds, all iteration |
| **HOLDOUT** | 2026-02-01 → present | **untouched. One single scoring at the very end.** |
| regime-analogue | the 17 months listed in §5 | secondary check, drawn from DESIGN only |

Rules:
- A trade belongs to a segment by **entry time**, and must **resolve inside it** — unresolved
  trades are dropped (right-censoring, as in `backtest_selection_rigorous.py`).
- **14-day embargo** between a fold's training data and its test data.
- No strategy, parameter or threshold may be chosen with reference to HOLDOUT results.
- If the HOLDOUT is examined for any reason before the final scoring, it is burned and the
  window must be re-cut with a later start date, recorded here.

## 2. Cost model (fixed, not tuned)

- fee 0.06% per side, slippage 0.03% per side → **round-trip 0.18%**
- `fee_r = round_trip × entry / sl_distance` — so tighter stops are penalised harder, which
  is the mechanism that makes low-timeframe strategies fail
- funding applied by `backtest_production_correct.run_simulation` in the dollar phase
- **Sensitivity is mandatory**: every surviving strategy is re-scored at **1× / 2× / 3×** cost

Baseline for comparison: the incumbent's fees consume **66% of its gross edge**.

## 3. Execution semantics (identical for every candidate)

Post-BOS-fix live semantics, from `backtest_selection_rigorous.py:98`:
- signal confirmed on the **last closed candle**; entry at the **next candle's open**
- ATR taken from the confirmation candle
- **SL wins ties** when a bar touches both stop and target (conservative)
- per-side fees and dynamic slippage applied
- unresolved trades dropped, never counted as flat

## 4. Candidate list — 18 pre-registered trials

Grids are fixed here. Every cell counts toward `N`.

**Controls**
| # | strategy | grid |
|---|---|---|
| C1 | Random-entry placebo (per strategy, matched symbol/side/geometry) | — |
| C2 | Incumbent: RSI divergence + BOS | live config as deployed |
| C3 | Buy & hold BTC | — |
| C4 | Equal-weight long-only basket | — |

**Repo strategies**
| # | strategy | source | grid |
|---|---|---|---|
| S1 | Donchian breakout (DONCH) | `autobot/core/breakout_detector.py` | channel {48, 96, 168} × atr {1.0, 1.5, 2.0} × rr {5, 8, 10} |
| S2 | Double Divergence (RSI + second oscillator) | `backtest_divergence_comparison.py:598` | atr {1.0, 1.5, 2.0} × rr {3, 5, 8, 10} |
| S3 | Boom Hunter Pro (Ehlers EOT) | `backtest_boom_hunter.py` | trigger tier {lime, green, all} × atr {1.0, 1.5, 2.0} × rr {2, 3, 5} |
| S4 | EMA-stack pullback (20>50>200 + RSI) | `backtest_trend_follow_optimized.py` | rsi {30, 35, 40} × atr {1.5, 2.0} × rr {3, 4.5} |
| S5 | Wysetrade (div in OB/OS + key level + structure break) | `backtest_wysetrade.py` | atr {1.0, 1.5} × rr {2, 3} |
| S6 | VWAP bounce | `backtest_vwap_bounce.py` | atr {1.0, 1.5, 2.0} × rr {2, 3, 5} |
| S7 | Liquidity-zone divergence | `backtest_liquidity_zone_div.py` | atr {1.0, 1.5, 2.0} × rr {3, 5, 8} |
| S8 | MACD divergence | `backtest_divergence_comparison.py:352` | atr {1.0, 1.5, 2.0} × rr {3, 5, 8, 10} |
| S9 | OBV divergence | `backtest_rsi_obv.py` | atr {1.0, 1.5, 2.0} × rr {3, 5, 8, 10} |
| S10 | Stochastic divergence | `backtest_divergence_comparison.py:436` | atr {1.0, 1.5, 2.0} × rr {3, 5, 8, 10} |

**New classes (external evidence, not present in repo)**
| # | strategy | rationale | grid |
|---|---|---|---|
| N1 | Cross-sectional momentum | published work finds cross-sectional is the form that works in crypto specifically | lookback {7, 14, 30, 90}d × decile {10%, 20%} × rebalance {1, 7}d |
| N2 | Time-series momentum, vol-scaled | risk-managed momentum literature | lookback {30, 60, 90}d × vol target {20%, 40%} |
| N3 | Funding-rate carry — delta-neutral | market-neutral; funding history verified paginable | entry threshold {0.005%, 0.01%, 0.02%} per 8h |
| N4 | Funding-rate sign as directional filter | cheap variant, no spot leg | threshold {0, 0.01%} × applied to {C2, S1} |

**Exit-side (applied only to an entry signal that already passed Phase 5)**
| # | variant |
|---|---|
| E1 | partial 50% at 1R + breakeven, trail remainder |
| E2 | ATR trailing stop after 2R |
| E3 | fixed TP (control) |

**Trial count for the Deflated Sharpe correction: N = 147**
(sum of all grid cells across S1–S10, N1–N4; controls and exit variants excluded as they
are not independent searches). This number is fixed now and will be used verbatim.

## 5. Regime-analogue months

Current conditions (2026-07): BTC 30d return +8.1%, CHOP 50.2, annualised vol 40%, 77% of
the universe above EMA200. Against BTC's monthly history, the months matching within
|Δ30d-return| < 8pp and |ΔCHOP| < 2 are:

`2023-04, 2023-09, 2023-11, 2024-03, 2024-05, 2024-07, 2024-09, 2024-10, 2025-01, 2025-04,
2025-05, 2025-06, 2025-07, 2025-09, 2026-03, 2026-04` — 16 months inside DESIGN, plus the
current month itself (excluded, it is in HOLDOUT).

This list is fixed now and will not be re-derived after seeing results.

## 6. Explicitly NOT tested (already falsified — do not re-litigate)

Per-symbol RR/ATR fitting (fit↔OOS corr +0.036 and +0.043, failed under two independent
methodologies) · symbol re-selection (522-symbol firewall study: every rule negative on
holdout, "trade everything" least bad) · 3M / 5M / 15M / 30M timeframes (3M "consistently
failed walk-forward"; 5M −1,025R / 1 of 24 symbols profitable; 15M 3,843 configs → 0
candidates) · BB squeeze (0 setups of 400 symbols) · EMA scalp (0 winners of 400) ·
RSI-extreme reversal (0 profitable setups) · regime-based halting (self-referential
lock-in, 0 trades) · removing the CHOP filter (PF 0.53, −100%) · raising base risk (DD 90%+)
· the quantedge branch (3 rounds, all abandoned).

## 7. Pass criteria — all seven must hold

A strategy is reported as a candidate for deployment **only** if it satisfies every one:

| # | criterion |
|---|---|
| 1 | beats its own random-entry placebo, **t > 3** |
| 2 | positive avg R net of costs, out-of-fold on DESIGN |
| 3 | positive in **≥ 60%** of the 16 regime-analogue months |
| 4 | positive avg R on the **untouched HOLDOUT** |
| 5 | **Deflated Sharpe > 0** given N = 147 trials |
| 6 | still positive at **2× assumed costs** |
| 7 | positive in **dollars** end-to-end through `run_simulation`, and **ROI/DD better than the incumbent** |

Criterion 7 exists because per-trade average R reversed against compounded dollars three
separate times in the 2026-07-26 analysis. Per-trade metrics alone are not sufficient
evidence for any change.

## 8. Reporting commitment

`STRATEGY_SEARCH_RESULTS.md` will list **every** candidate including failures, score each of
the seven criteria pass/fail, and state N so the reader can discount for multiple testing.
No candidate will be added, no grid widened, and no criterion relaxed after results are seen.
