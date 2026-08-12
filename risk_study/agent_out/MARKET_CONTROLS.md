# CRITERION 4 — Market benchmark controls

Protocol: `risk_study/PROTOCOL_symbols.md` §5 (C2/C3), decision rule §7.4 ("beats
C2/C3 (market controls) on ROI/DD"). This note adds **C4** per the task brief
(BTC 200-EMA trend follow) as the fair "simple systematic" control alongside the
two passive ones, and answers the question directly: do the two arms that matter
(A0 live-fitted, A1 global rr=10/atr=3.0) beat these controls on risk-adjusted
return, full period and holdout.

Window: **2023-06-01 → 2026-07-25**, holdout **2026-05-25 → 2026-07-04**. All
controls start at **$1,500**, unlevered. Source data: `cache_3yr_1h/*.parquet`
(1H bars, columns `start, open, high, low, close, volume, turnover`).

Script: `risk_study/agent_out/market_controls.py`. Outputs: this file,
`MARKET_CONTROLS_stats.csv`, `MARKET_CONTROLS_equity_hourly.csv` (C2, C4),
`MARKET_CONTROLS_equity_C3_daily.csv` (C3), `C3_eligibility_by_rebalance.csv`,
`MARKET_CONTROLS_strategy_arms_recomputed.csv` (A0/A1 recomputed metrics),
`market_controls_run_meta.txt`.

---

## 1. Benchmark construction

**C2 — BTC buy-and-hold.** Bought at the first 1H `open` in range, marked at
each subsequent `close`, held to the last `close` (2026-07-25 18:00). No costs,
no leverage.

**C3 — long-only equal-weight basket of the 277 live symbols.** "Live" =
`enabled: true` and non-empty `configs` in `config.yaml`; confirmed **277/277**
present in `cache_3yr_1h`. Closes resampled to daily (last 1H close per UTC
day). Rebalanced to equal weight on the first available trading day of each
calendar month (38 rebalances). At each rebalance, capital is split equally
across symbols with a valid price **that day**; a symbol not yet listed is
simply excluded until the first rebalance on/after its first cached bar (no
price is fabricated pre-listing). Between rebalances, unit holdings are fixed
(true buy-and-hold-then-rebalance, not daily-reweighted). No transaction costs
charged (the brief specifies costs only for C4; noted as an asymmetry in §4).

Eligibility grew from **89/277** symbols on 2023-06-01 (most of the universe
hadn't listed yet — median first-cached-bar across the 277 is 2024-06-05, and
16 symbols list as late as mid/late 2025) to **277/277** by 2025-10-01, holding
at 277 through 2026-06-01, then **272/277** at the final (2026-07-01) rebalance
(5 symbols missing a same-day print, not investigated further — immaterial:
0.9M of the ~1,150-day curve). Full eligibility history:
`C3_eligibility_by_rebalance.csv`. This is the correct read of "no survivorship
bias within the live set" from the brief: no live symbol drops out early, they
only phase in late, and the monthly rebalance handles that by construction
rather than needing an ad hoc fix.

**C4 — BTC 200-EMA trend follow.** `ema200 = close.ewm(span=200, adjust=False,
).mean()` on 1H bars (i.e. a 200-*hour*, ~8.3-day EMA — matches how the live
bot's own EMA200 gate is computed, on the same 1H series). Position at bar t =
1 (long) if `close[t-1] > ema200[t-1]` else 0 (flat) — lagged one bar, causal.
**24.2 bps charged at every switch** (flat→long or long→flat), per the brief's
literal instruction ("24.2 bps round trip per switch"); note this charges the
study's standard *round-trip* cost figure at each leg of a cycle rather than
once per completed cycle, i.e. a full flat→long→flat cycle pays ~48.4 bps, not
24.2 bps. Independently re-verified the crossing count outside the cost model:
**1,010 switches** over 1,150 days (~0.88/day, ~320/yr) — confirmed by counting
raw `close > ema200` sign changes with no lag applied, so this is a genuine
property of a fast EMA on noisy hourly crypto data, not a bug. Flagged as a
convention choice in §4 with a sensitivity note.

---

## 2. Full-period results (2023-06-01 → 2026-07-25, ~3.15 yr)

| | ROI % | CAGR % | maxDD % | ROI/maxDD | Calmar | Sharpe(d) | Sortino(d) | Longest UW (days) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **C2** BTC buy-hold | +137.71 | +31.61 | 53.76 | 2.56 | 0.59 | 0.83 | 1.25 | 292 |
| **C3** basket, monthly EW | −18.72 | −6.37 | 77.35 | −0.24 | −0.08 | 0.28 | 0.38 | 594 |
| **C4** BTC 200EMA trend | −83.60 | −43.65 | 85.81 | −0.97 | −0.51 | −1.43 | −2.04 | 863.5 |
| A0 LIVE fitted (recomputed) | +472.49 | +73.98 | 64.92 | 7.28 | 1.14 | — | — | — |
| A1 GLOBAL rr=10/atr=3 (recomputed) | +1285.03 | +130.30 | 47.53 | 27.04 | 2.74 | — | — | — |

(A0/A1 ROI/CAGR/Calmar recomputed from `global_arms.csv`'s `final`/`maxdd` using
the same $1,500 base and the same 1,150-day window; Sharpe/Sortino/underwater
aren't derivable from the two summary numbers alone, so they're left blank
rather than guessed — full A0/A1 equity paths weren't provided to this agent.)

BTC's own full-period return: **+137.71%** (this *is* C2). The window contains
a large crypto bull leg, which is exactly why C2 is here as a control.

## 3. Holdout results (2026-05-25 → 2026-07-04, 40 days)

| | ROI % | maxDD % | Sharpe(d) | Sortino(d) |
|---|---:|---:|---:|---:|
| **C2** BTC buy-hold | −18.66 | 25.02 | −3.39 | −4.40 |
| **C3** basket, monthly EW | −12.88 | 22.56 | −3.17 | −1.85 |
| **C4** BTC 200EMA trend | −16.13 | 19.09 | −4.19 | −6.00 |
| A0 LIVE fitted | **−44.80** | n/a (not supplied) | — | — |
| A1 GLOBAL rr=10/atr=3 | **+6.87** | n/a (not supplied) | — | — |

BTC's own holdout return: **−18.66%**. The holdout was simply a bad six weeks
for crypto broadly — all three controls lost money.

CAGR/Calmar are not reported for the 40-day holdout: annualizing a 40-day
window (×9.1) amplifies noise past the point of being informative (e.g. C2's
"CAGR" would print as −85%, which is an artefact of the exponent, not a real
annual rate). ROI is the right holdout metric and is what's used for PASS/FAIL
below.

---

## 4. PASS / FAIL grid

**Full period, on ROI/maxDD** (the protocol's stated metric):

| | vs C2 | vs C3 | vs C4 |
|---|---|---|---|
| **A0** (7.28) | PASS (2.56) | PASS (−0.24) | PASS (−0.97) |
| **A1** (27.04) | PASS (2.56) | PASS (−0.24) | PASS (−0.97) |

**Holdout, on ROI** (maxDD unavailable for the arms — see caveat below):

| | vs C2 | vs C3 | vs C4 |
|---|---|---|---|
| **A0** (−44.80%) | **FAIL** (−18.66%) | **FAIL** (−12.88%) | **FAIL** (−16.13%) |
| **A1** (+6.87%) | PASS (−18.66%) | PASS (−12.88%) | PASS (−16.13%) |

**A0 fails every market control in the holdout — it lost 2.4×–3.5× more than
simply holding BTC, holding a passive alt basket, or a naive trend filter would
have lost over the same six weeks.** A1 is the only arm here that made money in
the holdout and beat all three controls on it.

Full period is a clean sweep for both arms on ROI/maxDD, but see the leverage
caveat below before reading that as "the strategy has skill" — it is necessary
but not sufficient, and the holdout split is the more decision-relevant result
precisely because it's the one window this protocol pre-committed not to
overfit to.

---

## 5. Fairness notes (addressed as requested)

**Leverage/compounding asymmetry.** The strategy arms size positions at 0.3%
equity risk per trade with stops well inside the entry price, so realized
notional exposure and its variance are materially higher than an unlevered
$1,500 buy-and-hold or basket — and they compound daily off the taper/regime
multiplier machinery on top of that. ROI/maxDD **partially** normalizes for
this: it rescales return by the drawdown the *same* underlying path produced,
so a strategy that is simply "the same bet, bigger" would see both numerator
and denominator scale together and the ratio would be roughly unchanged. What
it does **not** correct for: (a) compounding and risk-management overlays
(taper, regime multiplier, net-directional/gross-risk caps) reshape the *path*
non-linearly, not just its scale, so ROI/maxDD on a managed, resized book isn't
apples-to-apples with ROI/maxDD on a fixed unlevered position; (b) it says
nothing about tail/liquidation risk, funding costs, or margin calls, which
scale worse than linearly with leverage and aren't visible in a peak-to-trough
equity number. Read the full-period ROI/maxDD PASS as "clears a necessary bar
for a levered strategy," not as full risk-adjusted parity with the unlevered
controls.

**Bull market context.** 2023-06→2026-07 contains a large crypto bull (BTC
+137.7% unlevered, C2 above). A0 and A1 still both beat C2 on full-period
ROI/maxDD despite that tailwind for a *long* benchmark, which is a genuinely
informative full-period result. But both arms beating a long-biased buy-hold
control during a bull is a low bar in isolation — see next point — and the
holdout (a chop/down window) is where the real discrimination shows up, and
that's exactly where A0 fails and A1 doesn't.

**Structural net-short.** The bot's config runs ~1.4:1 bear:bull configs, so it
is structurally net-short. Underperforming a long-only benchmark in a bull
window is the *expected* shape for this strategy and isn't by itself
disqualifying — full-period C2 comparison should be read with that prior. It
**is** decision-relevant, though: it's the mechanical reason A0's holdout loss
is so much worse than the passive controls' — a chop/down 40-day window is
close to the best case for a net-short book, and it still lost far more than
staying in cash-adjacent beta would have. That the structural tilt didn't pay
off in a window that should have favored it is itself informative, not just an
excusable mismatch.

---

## 6. Caveats

- A0/A1 holdout PASS/FAIL above is decided on ROI only — `global_arms.csv` does
  not carry holdout-window maxDD for the arms, and this agent did not have
  access to their full equity paths to derive it. The ROI gap (A0 −44.8% vs
  worst control −18.7%) is large enough that it's very unlikely a maxDD-based
  comparison would flip the verdict, but it is not directly verified here.
- C4's cost convention (24.2 bps at *every* switch, ~48.4 bps per completed
  flat→long→flat cycle) is the literal reading of the brief; if the intent was
  24.2 bps per completed round trip (12.1 bps/switch), C4's full-period ROI
  would still be sharply negative (1,010 switches at 12.1 bps ≈ 122% cumulative
  cost drag vs ≈244% at the literal reading) — it does not change C4's rank as
  the weakest control or any PASS/FAIL cell above.
- C3 carries no transaction costs (the brief only specifies a cost model for
  C4); the strategy arms and C4 both pay costs. This makes C3 a slightly easier
  bar than it would be with realistic monthly-rebalance frictions — immaterial
  here since both A0 and A1 beat it by wide margins in both windows except A0's
  holdout, which A0 still fails even against this cost-free version of C3.
- C3's basket composition is mechanical (equal-weight, no performance
  screening) — it is a market/beta control, not a claim about what an optimal
  passive alt allocation would have returned.
