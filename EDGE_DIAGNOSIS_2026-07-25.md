# Edge diagnosis — why the bot is losing, and what to do

Generated 2026-07-25. All figures from `cache_3yr_1h` (topped up to 2026-07-25 18:00 UTC
this session), the live `config.yaml` (277 symbols / 728 configs), and the validated
engine `backtest_3yr_walkforward` + `backtest_production_correct`.

Scripts: `gen_universe_8mo.py`, `backtest_protection_matrix.py`,
`backtest_selection_value.py`, `backtest_bos_timing_ab.py`, `extend_cache_to_today.py`.
Artefacts: `matrix_8mo.csv`, `matrix_oos.csv`, `selection_value_grid.csv`,
`bos_ab_summary_*.txt`.

---

## 0. The window that matters

The current config set went live **2026-05-25** (commit `e90114b`), and was walk-forward
fitted on data *through that same date*. So:

| window | status |
|---|---|
| 2025-11-01 → 2026-05-25 | the config's own **fitting/test** window — contaminated by selection |
| **2026-05-25 → 2026-07-25** | **genuinely out-of-sample**, and the only honest read |

Any 8-month number mixes the two and flatters the config. Everything below is split.

**Truncation control.** The universe only contains *resolved* trades, and wins take far
longer to resolve than losses (median 38h vs 6h). Raw recent numbers are therefore biased
negative. Every figure below uses a **cohort-complete estimator**: only entries with ≥H
hours of data remaining, counting only trades that resolved within H. At H=336h (14d),
98.4% of trades resolve. Conclusions are identical at H=168h (7d).

---

## 1. The edge is negative out-of-sample, and it is not noise

Cohort-complete, H=14d, net of fees:

| window | trades | WR | avg R | PF |
|---|---|---|---|---|
| fitting window (to 05-25) | 18,366 | 17.02% | **+0.1502** | 1.16 |
| **genuine OOS (05-25 → today)** | **4,523** | **13.53%** | **−0.1470** | **0.85** |

- bootstrap P(avg R ≥ 0) = **0.0004**, t = **−3.68**
- monthly avg R: Nov +0.04, Dec −0.11, Jan +0.99, Feb +0.20, Mar −0.02, Apr −0.35,
  May +0.31, Jun −0.16, **Jul −0.48**
- July WR is 10.4% against a ~13.9% breakeven

For contrast, the last *real* live ledger (`validation_6month_trades.csv`, 1,357 actual
trades to 2026-02-22, under the **previous** config) earned **+1,059R, PF 1.9, with no
decay trend** (trade-level slope p=0.028 but r²≈0.4%). The strategy worked. This
configuration of it does not.

---

## 2. No protection setting fixes it

28 combinations run through the production engine over the OOS window, $1,876 start.
**Every single one loses money.** Selected rows:

| scenario | trades | ROI | max DD |
|---|---|---|---|
| LIVE @ max 10 concurrent | 398 | **−9.4%** | 15.0% |
| LIVE @ max 20 concurrent | 740 | −14.4% | 23.0% |
| LIVE @ net-dir cap 5% | 2,162 | −29.3% | 47.1% |
| LIVE @ risk 0.15% | 2,484 | −29.9% | 51.5% |
| **LIVE (current settings)** | 2,484 | **−40.8%** | 59.0% |
| LIVE − net-dir cap | 2,596 | −56.0% | 68.8% |
| LIVE @ risk 1.2% (pre-Jul 21) | 2,143 | −61.3% | 81.1% |
| NAKED (no protections at all) | 3,087 | −93.1% | 98.5% |

Reading:
- The protections are **working** — naked is catastrophically worse. They are not the
  problem.
- The two levers that measurably reduce bleed are **concurrency** and **size**, and both
  work by *trading less*. Nothing creates profit.
- The bot has **no concurrency cap at all** today and runs 45+ positions. Capping at 10
  turns −40.8% into −9.4%. That is the single largest unforced gap.
- The **BTC short-gate was inert** in this window — BTC's 30d return never exceeded +10%,
  so impulse-bull fired on 0.0% of entries.
- The live sim (−40.8%) vs the account's actual (−10.9%) differ because the real bot is
  margin-constrained (effectively concurrency-limited, which the −9.4% row approximates)
  and ran 1.2% risk until 4 days ago. Direction agrees; magnitude does not.

---

## 3. The cause: the per-symbol RR/ATR fitting captured zero real signal

Replayed the full 4 RR × 3 ATR grid for all 728 (symbol, div_type) pairs — 292,475
trades — and asked whether the variant `config.yaml` actually chose beat the 11 it was
chosen over.

| | chosen avg R | alternates avg R | selection edge | paired t | pairs won |
|---|---|---|---|---|---|
| fitting window | +0.3268 | +0.1072 | **+0.2197** | **+15.66** | 73.7% |
| **true OOS** | −0.3366 | −0.2928 | **−0.0437** | −1.09 | **38.2%** |

A **+15.7 sigma** in-sample effect that transfers as **exactly nothing** out-of-sample —
worse than a coin flip. This is textbook overfitting. Each pair had only ~30 fitting
trades to choose the best of 12 variants; that is noise selection by construction.

**Correlation between a pair's fitting-window R and its OOS R: +0.036** (n=417).
Past per-symbol performance has essentially **no** predictive power for future
performance. Keeping only the top 50% / 25% by fitting-window R gives OOS −0.30 / −0.33 —
no improvement.

---

## 4. And it is not fixable by picking different symbols or settings

Every zero-fitting **global** config — one RR/ATR applied to all symbols and all divergence
types, no selection whatsoever — is also negative OOS:

| rr | atr 1.0 | atr 1.5 | atr 2.0 |
|---|---|---|---|
| 3 | −0.2262 | −0.1395 | **−0.1206** (best) |
| 5 | −0.2545 | −0.1952 | −0.1655 |
| 8 | −0.3450 | −0.2341 | −0.2991 |
| 10 | −0.3497 | −0.3472 | −0.4208 |

- **72.4%** of the 272 symbols with ≥30 OOS trades are negative; median symbol −0.36
- **all four** divergence types negative (HID_BEAR −0.15, HID_BULL −0.32, REG_BEAR −0.39,
  REG_BULL −0.48)
- only **30.6%** of the 421 chosen pairs with ≥5 OOS trades were positive

There is no configuration in the searched space that is currently profitable. The edge is
absent **market-wide for this strategy family**, not mis-selected.

---

## 5. The BOS timing bug is real, worth fixing, and is not the cure

`check_pending_bos` judges BOS on the still-forming candle, delaying entry a full candle
(see CLAUDE.md §11). Over 3 years / 277 symbols / 728 configs the fix is clearly better:
avg R +0.2645 → +0.4720, PF 1.297 → 1.548, drawdown −33%, better in **13/14 quarters**,
69% of configs, sign-test p = 5.8e-25. Decomposition shows the entry delay causes
essentially all of it.

**But in the OOS window the fix is slightly worse** (−0.2905 vs −0.1470). That is the
expected behaviour: better execution amplifies whatever edge exists, and the current edge
is negative. Fix it because it is correct — not as a remedy.

---

## 5b. Tested: "it's the choppy market, not the strategy"

A reasonable hypothesis, and decidable. `analyze_chop_hypothesis.py` annotates every trade
with the symbol's own CHOP at entry (what the live gate reads) plus BTC CHOP/ADX.

**The premise doesn't hold.** The market barely changed:

| | symbol CHOP | BTC CHOP | BTC ADX |
|---|---|---|---|
| fitting window | 49.62 | 49.58 | 27.42 |
| OOS | 51.01 | 51.15 | 28.67 |

Monthly BTC CHOP over all 9 months sits in a 48.2–50.9 band — no choppiness spike. BTC ADX
is slightly *higher* out-of-sample (more trending, not less).

**And conditioning on CHOP kills the hypothesis outright.** Every bucket went from positive
to negative:

| CHOP bucket | FIT avg R | OOS avg R | delta |
|---|---|---|---|
| 0–38 (most trending) | +0.2283 | **−0.3259** | **−0.5542** |
| 38–44 | +0.0258 | −0.1759 | −0.2017 |
| 44–48 | +0.1199 | −0.1713 | −0.2912 |
| 48–52 | +0.0997 | −0.2669 | −0.3666 |
| 52–56 | +0.2457 | −0.0327 | −0.2784 |
| 56–100 (choppiest) | +0.1721 | −0.0808 | −0.2529 |

The **most trending** bucket deteriorated the **most**. An Oaxaca decomposition of the
−0.2971 total drop attributes **0.0% to the CHOP mix shift and 100% to within-bucket
deterioration**. Same conditions, worse results.

Tightening the CHOP gate does not rescue it at any threshold (avg R stays −0.13 to −0.23
from CHOP<56 all the way down to CHOP<32, which keeps only 1.9% of trades).

Conclusion: choppiness is not the explanation. Waiting for a trending market will not, on
this evidence, bring the edge back — the strategy underperformed *most* in exactly the
trending conditions it is supposed to exploit.

## 5c. Audit note

An independent audit of this study's code found four defects; all are fixed:
`btc_impulse` had up to 23h of lookahead (resample label vs `.last()`), `btc_bull` used the
entry candle's own close, `chop_only` silently stripped the taper as well as the regime
multiplier, and drawdown peaks seeded from a module global. Re-running the OOS matrix after
the fixes changed nothing material — the short-gate was inert (impulse-bull fired on 0.0%
of OOS entries), and only the "− regime sizing" row moved (−72.2% → −75.4%).

Note the core findings in §1, §3, §4 and §5b use **no simulation engine at all** — they are
computed directly from the trade universe — so they were never exposed to these defects.

## 5d. Shadow-R kill-switch: swept, and it works

`backtest_shadow_gate.py` sweeps a halt/resume state machine on trailing shadow R
(3 windows × 5 stop levels × 4 restart levels = 60 combos). Causality is preserved two
ways: a signal only enters the ledger when it **resolves**, and because losses resolve far
faster than wins (6h vs 38h median) the real-time trailing sum is structurally biased
negative — the gate has to work through that handicap. Halting blocks new entries only;
open positions run to stop/target, matching live `/stop`.

**Genuine OOS (2026-05-25 → today), $1,876 start:**

| | ROI | max DD | PF |
|---|---|---|---|
| BASE — no gate (current bot) | **−40.8%** | 59.1% | 0.84 |
| FLAT — never trade | 0.0% | 0.0% | — |
| gated, median of all 60 combos | **+16.2%** | 22.9% | — |
| best (7d, stop −300R, start +400R) | **+38.3%** | 15.2% | — |

- **100% of the 60 threshold combos beat the ungated base**
- **93.3% are outright profitable**, worst combo −30.3%
- the 21-day window is the most stable: **20/20 positive**, range +5.9% … +16.2%
- stop −200R…−400R is the sweet spot; −100R halts too eagerly (51% of the time for only
  +5.2%), −600R halts too rarely

This is not a knife-edge optimum — the result is broad enough that a threshold picked
blind from this grid still turns −40.8% into a median +16.2%.

**Full 8 months (includes the config's own fitting window):** base +238.4%; only 23.3% of
combos beat it, though 91.7% remain profitable (median +149.0%). That is the honest
trade-off — **the gate is insurance. It costs upside when the edge is working and saves
you when it is not.**

**Calibration bug worth fixing:** the live advisory bands are `warn ≤ −25R`, `crit ≤ −35R`
on trailing 7d, but this population's trailing-7d R has median −19R and p10 −385R. The
bands are roughly **10× too tight** — as thresholds they would halt almost permanently.
Real values live in the hundreds.

Note the shadow layer has in fact been reading CRIT for weeks (live `/learn`: 7d −391.7R,
W27–W30 all −200R to −385R). On this evidence it was right, and acting on it would have
been worth roughly 55–80 points of ROI over the period.

## 5e. Dollars from a $1,500 start, month by month

`report_monthly_1500.py`. Genuine OOS (2026-05-25 → today):

| month | BASE P&L / bal | GATED 21d P&L / bal | GATED 7d P&L / bal |
|---|---|---|---|
| 2026-05 | −120 → 1,380 | −120 → 1,380 | −120 → 1,380 |
| 2026-06 | +178 → 1,558 | +385 → 1,765 | +710 → 2,089 |
| 2026-07 | **−671 → 887** | **−22 → 1,743** | **−14 → 2,075** |
| **final** | **$887 (−40.8%)** | **$1,743 (+16.2%)** | **$2,075 (+38.3%)** |
| max DD | 59.1% | 33.3% | 15.2% |

July is the whole story: ungated loses $671, gated loses $22. The gate sat out the month
the edge was worst. Never trading leaves $1,500.

Over the full 8 months (fitting window included, so optimistic): BASE $6,071, GATED 21d
**$6,569**, GATED 7d $3,784. The 21d gate wins in *both* windows; the 7d gate wins big OOS
but gives up a lot when the edge works.

## 5f. Base-risk sweep — and why it is the strongest evidence yet

`report_risk_sweep_1500.py`, $1,500 start, taper ladder scaled proportionally with the base
so its shape is preserved. Genuine OOS:

| base risk | UNGATED $ | UNGATED DD | GATED $ | GATED DD | gated ROI/DD |
|---|---|---|---|---|---|
| 0.10% | 1,154 | 32.1% | 1,449 | 19.1% | −0.18 |
| 0.15% | 1,059 | 41.2% | 1,491 | 23.6% | −0.03 |
| **0.30%** | 887 | 59.1% | **1,743** | **33.3%** | **0.486** ← best |
| 0.50% | 756 | 69.1% | 1,757 | 37.3% | 0.459 |
| 0.80% | 532 | 82.4% | 1,798 | 45.9% | 0.433 |
| 1.20% | 452 | 84.5% | 1,807 | 50.7% | 0.404 |
| 2.00% | 471 | 92.1% | 1,743 | 66.7% | 0.243 |

**The two columns have different shapes, and that is the point.** Ungated is *monotonic* —
every increase in risk loses more, with no interior optimum. That is the signature of a
negative edge: size has no optimum, only "smaller is less bad". Gated is *concave*, peaking
at 1.20% and falling away either side. That is the signature of a real positive edge. The
gate does not merely reduce losses; it changes the sign of the thing being compounded.

Risk-adjusted, **0.30% is the knee**: best ROI/DD (0.486), capturing 79% of the maximum
dollar gain at 66% of the drawdown of the 1.20% peak. Your current setting is already 0.30%.

Ignore the full-8-month risk table (`$171,563` at 2% ungated) — that is a compounding
artefact of a fitted window and is exactly the kind of number that should not drive
decisions.

## 6. Recommendation

### Final setup (supersedes the halt-only advice below)

From $1,500, in the only window the config hadn't seen:

| setup | result |
|---|---|
| what you run today | **$887** (−40.8%), DD 59% |
| never trading | $1,500 |
| **recommended** | **$1,743** (+16.2%), DD 33% |

**Configure it this way:**

1. **Keep base risk at 0.30%** and keep the taper ladder as-is. It is the risk-adjusted
   knee (ROI/DD 0.486) and captures 79% of the maximum dollar gain at two-thirds the
   drawdown. No change needed — this one is already right.
2. **Add the shadow-R gate: 21-day window, halt at −300R, resume at +200R.** This is the
   single change that matters. It is the only intervention tested that turns the period
   profitable.
3. **Recalibrate the `/learn` bands** from −25R/−35R to the hundreds. They are ~10× too
   tight for this signal population and currently read CRIT permanently.
4. **Add a max-concurrent-position cap (~10).** There is none today; the bot runs 45+.
   Independently worth −9.4% vs −40.8% in the ungated matrix.
5. **Do not re-fit symbols or RR/ATR.** See §3 — the selection has zero out-of-sample
   value (fit↔OOS correlation +0.036).
6. **Remove the 6 dead symbols.**

**Why the 21d window rather than the 7d** (which scored better OOS at +38.3%): 21d was
positive in **20/20** threshold combinations versus 17/20 for 7d, and it also beat the
ungated base over the full 8 months ($6,569 vs $6,071) where 7d gave up a lot ($3,784).
It is the choice that does not depend on the edge staying broken.

**Why trust this rather than the specific numbers:** every threshold combination tested
(60/60) beat the ungated base, and 93% were profitable. The result does not hinge on
picking the right threshold. And the risk sweep's *shape change* — monotonic ungated,
concave gated — is independent evidence that the gate restores a genuine edge rather than
just trading less.

**What this is not:** it is not a fix for the strategy. The underlying edge is still
negative (§1) and the gate works by sitting out roughly 40% of the time. If the edge does
not return, this setup grinds slowly rather than bleeding fast. The honest fallback remains
that not trading leaves $1,500.

### Original (pre-gate) advice, kept for context

**Do not re-fit, and do not keep trading as-is.** Re-fitting is what produced this: the
last re-fit went negative the day it was deployed, and §3 shows the fitting procedure
extracts zero transferable signal. A new fit would search a space where §4 shows every
point is currently unprofitable, and would return whichever points were luckiest lately.

1. **Halt new entries now** (`/stop`). Justification is not the 4-day streak — it is
   4,523 OOS signals at t = −3.68 with no profitable configuration among 28 tested.
2. **If you keep trading anyway**, add a **max-concurrent-position cap of ~10** (the bot
   has none) and drop risk to 0.15%. That is −9.4% instead of −40.8% in the OOS sim.
   Understand this slows the bleed; it does not create profit.
3. **Fix the BOS timing** and **remove the 6 dead symbols** (`BTCUSDT-26JUN26`,
   `ETHUSDT-26JUN26` — expired dated futures still enabled — plus `MBOXUSDT`, `MLNUSDT`,
   `SOLVUSDT`, `SWARMSUSDT`). Both are correctness fixes worth doing regardless.
4. **Let the shadow layer decide when to restart.** It already logs and grades every
   evaluated signal and never gates trading — this is exactly what it was built for.
   Define the re-entry rule *in advance*: resume only after the shadow ledger shows a
   positive cost-adjusted edge over a pre-committed number of signals (250+).
5. **If you ever re-fit, cut the parameter count hard.** One global RR/ATR, or a handful
   of buckets — not 728 independent selections on ~30 trades each. Fewer parameters will
   not manufacture edge, but they will stop manufacturing false confidence.

### What would change this conclusion

The strategy is regime-dependent — January 2026 returned +0.99 avg R. Two months is a
short window even at 4,523 signals, and this may be a hostile regime rather than a dead
edge. That is precisely why the recommendation is **halt and shadow-monitor**, not
"abandon". The machinery to detect the edge returning already exists and costs nothing to
run while flat.
