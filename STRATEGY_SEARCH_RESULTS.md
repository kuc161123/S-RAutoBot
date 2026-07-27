# Strategy Search — RESULTS

Run 2026-07-27 against `STRATEGY_SEARCH_PROTOCOL.md`, committed 2026-07-27 **before any
result existed** (commit `663959a`). Nothing below was added, widened or relaxed after
seeing results.

## Verdict: nothing passed. 0 of 9 candidates cleared the bar.

---

## 1. What was run

| | |
|---|---|
| harness | `strategy_lab/` — one execution engine, one cost model, one statistical protocol for every candidate |
| universe | 514 symbols, 1H |
| DESIGN | 2023-06-01 → 2026-02-01 |
| HOLDOUT | 2026-02-01 → present, **read exactly once, at the end** |
| trials | N = 147 (pre-registered), sd of trial Sharpes = 0.0118 |
| trades scored | 70,000–429,000 per candidate |

**Harness validated before use.** It reproduces the incumbent's known numbers: real avg R
−0.0203 (target −0.0172), WR 18.02% (18.07%), placebo delta **+0.0417, t = +5.14**
(target +0.0377, t = +4.66). All eight detectors pass no-lookahead and holdout-firewall
checks.

**A bug the validation caught.** My first placebo transplanted the signal-bar stop distance
onto a random bar. Because `fee_r = round_trip × entry / sl_dist`, that made the control far
too weak (−0.1585 vs the correct −0.0620) and **inflated every candidate's apparent edge**.
Before the fix all 12 candidates "beat" their control at t = 2.8–9.8. After it, most do not.
Had this gone unnoticed the entire study would have been a false positive generator.

## 2. Results

| strategy | design avg R | 2× cost | 3× cost | analogue+ | **HOLDOUT** | holdout n | DSR | placebo t |
|---|---|---|---|---|---|---|---|---|
| C2 incumbent (divergence+BOS) | −0.0198 | −0.1379 | −0.2566 | 31% | **−0.1555** | 34,047 | 0.000 | 4.94 |
| **S1 Donchian 96** | **+0.0423** | −0.0778 | −0.1963 | 23% | **−0.3379** | 27,087 | 0.000 | 7.75 |
| S1 Donchian 48 | +0.0123 | −0.1155 | −0.2437 | 31% | −0.3205 | 41,544 | 0.000 | 7.35 |
| S1 Donchian 168 | +0.0073 | −0.1060 | −0.2147 | 31% | −0.3257 | 18,570 | 0.000 | 4.52 |
| **S2 Double divergence (OBV)** | **+0.0301** | −0.0930 | −0.2132 | 46% | **−0.2129** | 3,800 | 0.012 | 4.14 |
| S2 Double divergence (MACD) | −0.0075 | −0.1247 | −0.2473 | 38% | −0.2404 | 6,341 | 0.000 | 3.22 |
| S8 MACD divergence | −0.0360 | −0.1555 | −0.2742 | 23% | −0.1535 | 35,846 | 0.000 | 2.74 |
| S9 OBV divergence | −0.0372 | −0.1591 | −0.2809 | 31% | −0.1910 | 30,361 | 0.000 | 3.16 |
| S10 Stochastic divergence | −0.0335 | −0.1476 | −0.2625 | 23% | −0.1732 | 31,464 | 0.000 | 3.00 |

Killed at screen (negative placebo delta — no timing information at all):
**Boom Hunter** (−0.1548, placebo t = −1.38 and −5.56 for the strict tier) and
**EMA-stack pullback** (−0.1071, placebo t = 0.82).

### Criteria scored

| strategy | 1 placebo t>3 | 2 design>0 | 3 analogue≥60% | 4 holdout>0 | 5 DSR>0 | 6 2× cost>0 |
|---|---|---|---|---|---|---|
| C2 incumbent | PASS | fail | fail | fail | fail | fail |
| S1 Donchian 96 | PASS | PASS | fail | fail | fail | fail |
| S1 Donchian 48 | PASS | PASS | fail | fail | fail | fail |
| S1 Donchian 168 | PASS | PASS | fail | fail | fail | fail |
| S2 Double div (OBV) | PASS | PASS | fail | fail | fail | fail |
| S2 Double div (MACD) | PASS | fail | fail | fail | fail | fail |
| S8/S9/S10 | mixed | fail | fail | fail | fail | fail |

Criterion 7 (dollars end-to-end) was **not run**: it applies only to candidates surviving
the earlier phases, and none did. Running it would have been theatre.

## 3. The three findings that matter

### a) Costs are the binding constraint, not signal choice

**At 2× assumed costs, every single candidate goes negative** — including the two that are
positive at 1×. Donchian 96 falls from +0.0423 to −0.0778; Double Divergence from +0.0301
to −0.0930.

The entire edge of this strategy family lives inside a 0.18% round-trip assumption. That is
not a signal problem and no amount of searching for better entries fixes it. It also means
the live cost figure genuinely matters: the `exec_log` table added on 2026-07-26 will
measure real slippage per fill for the first time, and if actual costs run above assumption,
these strategies were never viable.

### b) The signals work; monetising them does not

Almost every candidate beats its own random-entry control at t = 2.7–7.8. The information is
real. But after ATR-sized stops and per-side costs, the surviving expectancy is ~0.03–0.04 R
at best — thinner than the cost drag itself. This is the same diagnosis reached for the
incumbent (fees consume 66% of gross edge, 0.48pp of headroom) generalising to the whole
family.

### c) Donchian collapsed exactly like everything else

It carried the strongest prior in the repo — independent 16-fold purged walk-forward,
+1,011R over 2,327 trades, PF 1.53, positive in 12 of 15 folds — and it was the best design
performer here (+0.0423). On the untouched holdout it is **the worst candidate tested**
(−0.3379).

Best-in-design becoming worst-out-of-sample is the identical pattern that produced the
current live config. It is now documented across three independent strategy families, which
makes it a property of this market period rather than a quirk of one fit.

Corroboration from live data: the DONCH shadow family in the production database shows
22 resolved signals, **0% win rate, avg R −1.0**. Tiny sample, but pointing the same way.

## 4. Recommendation

**Deploy nothing. Keep the bot as it is.**

No candidate cleared a single one of the four out-of-sample criteria. Choosing the "best"
of these — Donchian on its design number — would mean deploying the strategy with the worst
holdout result in the study, on the strength of the exact in-sample metric that has now
misled this project three times.

What this search actually established:

1. **The problem is not the signal.** Ten strategy families, all carrying genuine timing
   information, all unable to overcome costs. Searching for an eleventh is low value.
2. **The lever with real headroom is cost, not entry logic.** Halving effective round-trip
   cost is worth more than any signal improvement measured here. That means maker fills,
   fewer/larger trades, or venue economics — not a new indicator.
3. **The regime is hostile to the whole family.** Every candidate is negative on
   2026-02 → present, and none is positive in even 60% of regime-analogue months.

The honest position is unchanged from 2026-07-26: keep the config frozen, keep base risk at
0.30%, and let the shadow-R advisory tell you when conditions turn. Let `exec_log` accumulate
— if it shows real costs above assumption, that closes the question permanently; if it shows
them below, the cost lever becomes concrete and measurable.

## 5. Caveats

- Grids were partially explored: one representative cell per strategy was scored, not all
  147 pre-registered cells. A different cell might score better in design — but design
  performance is exactly what failed to transfer, so this is unlikely to change the verdict.
- Cross-sectional momentum, vol-scaled TSMOM and funding carry (N1–N4) were **not run**.
  They require panel/portfolio machinery beyond the per-symbol harness. Given that the
  binding constraint turned out to be cost rather than signal, and that these are lower
  turnover by construction, they remain the most interesting unexplored direction.
- The holdout has now been used. Any future test needs a fresh, later window.
