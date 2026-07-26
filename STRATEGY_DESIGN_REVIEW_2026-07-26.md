# Strategy design review — is the strategy itself built correctly?

Generated 2026-07-26. All tests on the **design period** (entries before 2026-02-01) so
the holdout stays clean, except one pre-registered holdout confirmation in §4.

Scripts: `analyze_strategy_design.py`, `analyze_mfe_uncensored.py`,
`analyze_stop_width_costs.py`, `analyze_signal_mechanics.py`,
`analyze_geometry_holdout.py`.

**Verdict: the signal layer is well designed. The trade geometry is not, and the
meta-layers (per-symbol selection, regime sizing) are noise.**

---

> ## ⚠️ CORRECTION 2026-07-26 — read before acting on §7
>
> The ranked recommendations in §7 were derived from **per-trade average R**. When the
> same changes were run end-to-end in dollars through the production engine
> (`backtest_proposed_vs_base.py`, $1,500, ablated), **two of them reversed**:
>
> | §7 rec | per-trade R said | end-to-end dollars said | status |
> |---|---|---|---|
> | #1 widen stops to 2–3× ATR, rr 5–6 | +0.09 R/trade | $5,034 vs BASE $80,224 (10mo); −9.3% vs +101.5% (from 02-01) | **WITHDRAWN** |
> | #3 retire the regime classifier | 70% tier churn on constant edge | best in 2 windows, **worst in the clean one** ($417, 87% DD) | **WITHDRAWN** |
> | #4 concurrency cap ~20 | DD 49.7%→23.3% | better DD in 80% of 10 windows, but median ROI/DD **30.5 → 10.0** | downgraded to a risk-preference choice |
>
> Why #1 failed: average R per trade is not compounded dollars. The high-RR configs
> produce rare ~10R winners, and with risk-based compounding those tails drive nearly all
> the growth. Dropping to rr6 surrenders the tail *and* takes more trades, raising
> correlated exposure (DD 60.9% vs 29.7%).
>
> Why #3 failed: the classifier is ~70% noise as a **predictor** — that finding stands —
> but it functions as a **volatility brake**, throttling size exactly when things go bad.
> Removing it amplifies both directions.
>
> Separately, a 10-window rolling test of the shadow halt
> (`backtest_halt_multiwindow.py`) found the 7d gate **beat base in 0 of 10 windows**.
> Keep the halt advisory; do not automate it.
>
> **Sections 1–3 (signal quality, BOS confirmation, mechanics plateau) are unaffected —
> they are direct measurements, not proxies.** The cost-structure analysis in §4 is also
> factually correct; what does not follow is the conclusion that changing the geometry
> improves outcomes.
>
> Standing decision: **the bot stays as it is.** No structural change is clearly justified.

---

## 1. The signal carries real information ✅

Placebo test — keep the symbol, side and geometry of every real signal but move the entry
to a random bar:

| | trades | WR | avg R |
|---|---|---|---|
| REAL (divergence + BOS + EMA gate) | 122,077 | 18.07% | −0.0172 |
| PLACEBO (random entry, same geometry) | 244,103 | 17.42% | −0.0549 |

**Signal edge over random: +0.0377 R/trade, t = +4.66.** The divergence genuinely predicts
timing. This is the single most important thing to know — the premise is not broken.

## 2. BOS confirmation is correct and adds value ✅

| | trades | WR | avg R |
|---|---|---|---|
| enter at BOS (live) | 122,077 | 18.07% | **−0.0172** |
| enter at the divergence bar (no BOS) | 192,982 | 17.57% | −0.0471 |

Waiting for the break is worth **+0.030 R/trade**. This matches established practice —
[trading divergence without confirmation drops the win rate below 40%](https://algoalpha.io/blog/rsi-divergence-trading-strategy-how-to-spot-trade-and-avoid-false-signals),
while divergence + confirmation + a stop beyond the divergence extreme is the standard
three-step form.

## 3. The divergence/BOS mechanics sit on a plateau ✅

Sensitivity at fixed geometry (3.0× ATR / rr6), one parameter at a time:

| variant | trades | avg R | vs baseline |
|---|---|---|---|
| **LIVE baseline** (pivot 3, fresh 10, wait 12, close-break, RSI 14) | 121,949 | **+0.0753** | — |
| pivot width 2 | 179,354 | +0.0785 | +0.0032 |
| pivot width 5 | 72,364 | +0.0898 | +0.0144 |
| freshness 5 | 114,090 | +0.0734 | −0.0019 |
| freshness 20 | 124,331 | +0.0769 | +0.0016 |
| BOS wait 6 | 98,859 | +0.0716 | −0.0037 |
| BOS wait 24 | 140,148 | +0.0813 | +0.0060 |
| BOS on wick instead of close | 150,105 | +0.0670 | −0.0084 |
| RSI 7 | 140,182 | +0.0399 | **−0.0354** |
| RSI 21 | 118,569 | +0.0514 | **−0.0239** |

Every structural parameter moves the result by less than ±0.015 R — a **plateau**, which
is what a non-overfitted design looks like. Two specific confirmations:

- **BOS on close beats BOS on wick** (+0.0084) — the live choice correctly avoids
  wick/stop-run false breaks.
- **RSI 14 is a genuine optimum**, clearly beating 7 and 21. And it wasn't fitted — 14 is
  the textbook default. That it also wins is reassuring rather than suspicious.

**Nothing needs changing in the divergence or BOS logic.**

---

## 4. The trade geometry is the real flaw ❌

`fee_r = round_trip_cost × entry / sl_distance`. The stop distance is the **denominator**,
so a tight stop makes every fee enormous in R terms:

| stop width | cost drag |
|---|---|
| 1.0× ATR | **0.1526 R/trade** |
| 1.5× ATR | 0.1017 |
| 2.0× ATR | 0.0763 |
| 3.0× ATR | 0.0509 |
| 6.0× ATR | 0.0254 |

At the live config's typical 1.5× ATR / rr10: gross expectancy **+0.1550 R**, cost drag
**−0.1017 R**, net **+0.0533 R**. **Costs consume 66% of the gross edge.**

Margin of safety: hit rate 10.50% vs breakeven-including-costs 10.02% — **0.48pp of
headroom**. A half-point slip in hit rate takes the strategy to zero. That is why it broke
the moment conditions changed: it was never robust, only marginal.

### Net expectancy surface (design period, costs included)

| atr ↓ / rr → | 2 | 3 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| **1.0×** | −0.149 | −0.135 | −0.117 | −0.095 | −0.053 | −0.015 |
| **1.5×** | −0.079 | −0.066 | −0.015 | +0.013 | +0.042 | +0.054 |
| **2.0×** | −0.050 | −0.022 | +0.040 | +0.059 | **+0.075** | +0.072 |
| **3.0×** | −0.006 | +0.035 | +0.075 | **+0.076** | +0.074 | +0.056 |
| **4.0×** | +0.016 | +0.053 | +0.068 | +0.068 | +0.047 | −0.018 |

**Every cell in the 1.0× ATR row is negative — and the live config puts 336 of 756 configs
there.** Headroom at the best geometry (3.0× / rr6) is 1.08pp, more than double the live
setting's 0.48pp.

### Pre-registered holdout confirmation

The above used design data only. Scoring a pre-committed geometry set on the holdout, once:

| geometry | holdout avg R | PF |
|---|---|---|
| 2.0× ATR / rr6 | **−0.1697** | 0.82 |
| 3.0× ATR / rr5 | −0.1739 | 0.81 |
| 3.0× ATR / rr6 *(design winner)* | −0.2181 | 0.77 |
| 1.5× ATR / rr10 *(live's most common)* | **−0.2658** | 0.74 |
| 1.0× ATR / rr10 *(336 live configs)* | −0.2635 | 0.75 |

Everything is negative in this regime, but the **ranking transfers**: wider stops with
lower RR beat the live geometry by **~+0.09 R/trade** — more than double the entire signal
edge over random. The exact winning cell did not transfer; the direction did.

## 5. High RR is the right idea, and my earlier doubt was wrong ✅

Uncensored MFE (stop only, no take-profit) over 122,082 signals:

| percentile | max favourable excursion |
|---|---|
| p50 | 1.01 R |
| p80 | 4.33 R |
| p90 | **10.53 R** |
| p99 | **58.63 R** |

The distribution is extremely fat-tailed, and **every** RR from 2 to 10 is reached more
often than its breakeven. But after costs, *low* RR is negative (rr=2: −0.079, rr=3:
−0.066) and high RR is positive. So the concentration at rr 8–10 is **correct for this
signal** — my earlier suspicion that it was too greedy was an artefact of a censored
measurement (a 5R take-profit stops MFE tracking at 5R, so 8R can never appear).

## 6. The meta-layers are noise ❌

**Regime classifier.** Simulating a stream with a *constant* true edge (WR 18.1%, avg win
+4.90R, avg loss −1.10R) through the live 20-trade tier rules:

| tier | share of the time |
|---|---|
| favorable | 28.8% |
| cautious | 21.0% |
| adverse | 40.3% |
| critical | 9.9% |

**Tier changes on 70.1% of draws — from a stream with no real regime changes at all.** The
bot assigns 0.1×–1.0× position size largely on noise. A 20-trade window at a ~17% win rate
contains ~3.4 wins; it cannot distinguish anything.

**Per-symbol RR/ATR selection.** Shown twice to have no out-of-sample value
(fit↔OOS correlation +0.036 and +0.043; config stability 10.9% vs 8.3% by chance).

**Portfolio construction.** Median 56 simultaneous positions (peak 125) across correlated
altcoins with a 66% median directional imbalance — not 56 bets, closer to one.

---

## 7. Ranked recommendations

| # | change | evidence | expected effect |
|---|---|---|---|
| 1 | **Widen stops to 2–3× ATR, RR ~5–6** | holdout-confirmed ranking | **+0.09 R/trade** vs current geometry; headroom 0.48pp → 1.08pp |
| 2 | **Drop per-symbol RR/ATR — one global geometry** | two independent methodologies | removes a noise layer; simplifies to 1 parameter set |
| 3 | **Retire or rebuild the 20-trade regime classifier** | 70% tier churn on constant edge | stops sizing on noise |
| 4 | **Cap concurrent positions (~20)** | DD 49.7% → 23.3% over the last year | halves drawdown for ~half the profit |
| 5 | Consider partial exits / trailing | p99 MFE 58R vs fixed cap | monetises the fat tail; untested here |

**Do not change anything mid-drawdown.** All of the above is for a rebuild when the edge
returns — the current period is negative under every geometry tested, and changing the
configuration now means deploying an untested variant into a regime already known to be
hostile. The advisory halt remains the first-order lever.

## 8. Honest summary

The strategy is **well conceived and poorly parameterised.** The signal works
(t = +4.66 vs random), the confirmation logic is correct and robust to its settings, and
the high-RR premise matches the actual excursion distribution.

What kills it is that the edge is thin (~0.155 R gross) and the geometry hands **66% of it
to fees**, leaving half a percentage point of margin. Layered on top are two meta-systems
— per-symbol optimisation and 20-trade regime sizing — that add complexity and noise
without adding edge.

A version of this strategy with one global geometry at 2–3× ATR, no per-symbol fitting, no
regime classifier, and a concurrency cap would be simpler, cheaper to run, and structurally
more robust than what is deployed. It would still be a thin-edge strategy that needs
favourable conditions — but it would have roughly twice the margin before it breaks.

### Sources consulted
- [RSI Divergence Trading Strategy — confirmation and false signals, AlgoAlpha](https://algoalpha.io/blog/rsi-divergence-trading-strategy-how-to-spot-trade-and-avoid-false-signals)
- [RSI divergences: what they are and how they work, Kraken](https://www.kraken.com/learn/rsi-divergences-what-they-how-they-work)
- [MFE & MAE: the complete guide, TradesViz](https://www.tradesviz.com/blog/mfe-mae-charts/)
- [Win rate and risk/reward connection, LuxAlgo](https://www.luxalgo.com/blog/win-rate-and-riskreward-connection-explained/)
