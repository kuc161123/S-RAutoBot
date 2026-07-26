# Can we find better symbols and configs? — rigorous re-selection study

Generated 2026-07-26. Script: `backtest_selection_rigorous.py`.
Artefacts: `rigorous_selection_trades.parquet`, `rigorous_selection_holdout.csv`,
`rigorous_full.log`.

---

## 1. What was wrong with the original selection

The live config came from `backtest_3yr_walkforward.py`. Its selection is not a
walk-forward at all:

| line | problem |
|---|---|
| 544 | `best_total_test_r` — the chosen config is the one that **maximises the "test" window R**. The test windows are a second training set. |
| 517 | configs are also **filtered** on test-window avg-R / PF / drawdown — survivorship on out-of-sample results |
| 563 | the Monte Carlo runs on **those same test trades**, so it validates nothing |
| 57 | `MIN_TEST_TRADES = 3` — best-of-30 variants chosen on three trades |
| 66-70 | 4 div × 3 atr × 10 rr × ~500 symbols ≈ **60,000 tests, uncorrected** |
| 49-54 | the four windows **overlap heavily**, so "passes all 4" is not 4 independent confirmations |
| — | no purge/embargo (trades span the boundary); **no untouched holdout anywhere** |

Measured consequence (`backtest_selection_value.py`): the selection was worth
**+0.2197 R/trade in-sample (t = +15.66)** and **−0.0437 out-of-sample (t = −1.09)**.

## 2. What the replacement does differently

- **Holdout firewall.** Everything from **2026-02-01** onward is untouched until one
  final evaluation. `design` requires trades to *resolve* before the firewall.
- **Selection uses train only**, with a **14-day embargo** before each fold.
- **Honest minimums** — ≥40 trades per config (not 8/3).
- **Shrinkage** (James–Stein style, pulled toward the global mean by sample size) so a
  config cannot win on a small lucky sample.
- **Smaller search space** (4 rr × 3 atr, not 10 × 3) — every extra option costs OOS power.
- **The null is tested**: every rule is compared against "one global config" and
  "trade everything".
- **Live fidelity**: BOS on the last CLOSED candle, entry at next candle open, ATR from
  the BOS candle, EMA-200 gate at confirmation, 12-candle wait, SL-wins-ties, fees +
  slippage per side, CHOP gate at the entry candle — i.e. the bot as it runs *after* the
  2026-07-25 BOS fix.

Universe widened to **522 cached symbols** (the live config trades 277), all four
divergence types, long and short. 978,512 candidate trades.

## 3. Result — inside the design period, everything looks fine

Out-of-fold, 6 rolling folds:

| rule | trades | WR | avg R | PF |
|---|---|---|---|---|
| trade everything (no selection) | 512,256 | 20.08% | +0.2187 | 1.248 |
| ONE global (rr, atr) | 42,747 | 13.41% | +0.2792 | 1.284 |
| per-symbol best (old way) | 38,206 | 18.63% | +0.2310 | 1.256 |
| per-symbol + min-N + shrinkage | 11,750 | 16.40% | +0.2187 | 1.234 |

## 4. Result — on the untouched holdout, everything loses

| rule | trades | WR | avg R | PF |
|---|---|---|---|---|
| trade everything (no selection) | 214,616 | 16.07% | **−0.1128** | 0.880 |
| ONE global rr=10/atr=1.0 | 17,963 | 9.10% | −0.1663 | 0.843 |
| per-symbol best (old way) | 17,632 | 13.74% | −0.1346 | 0.861 |
| per-symbol + min-N + shrinkage | 7,581 | 12.21% | −0.1614 | 0.838 |

**No rule is profitable.** And "trade everything" — *no selection at all* — is the least
bad. Selection subtracts value.

## 5. Why: what selection captures is regime, not skill

- config stability across design halves: **10.9%** of 617 pairs keep the same (rr, atr),
  versus **8.3% by chance**
- corr(design first half R, design second half R) = **+0.262** — some within-era persistence
- corr(design R, **holdout** R) = **+0.043** — none across the regime boundary

So a symbol/config's past performance predicts its near-term future *within the same
market regime* and predicts nothing once the regime turns. Selection is fitting the era.

### The clinching evidence: RR preference is pure regime

avg R by RR, every quarter:

| quarter | rr=3 | rr=5 | rr=8 | rr=10 | best |
|---|---|---|---|---|---|
| 2024Q2 | +0.137 | +0.198 | +0.283 | **+0.336** | rr=10 |
| 2025Q1 | +0.188 | +0.305 | +0.463 | **+0.542** | rr=10 |
| 2025Q4 | +0.029 | +0.130 | +0.253 | **+0.286** | rr=10 |
| 2026Q2 | **−0.033** | −0.070 | −0.104 | −0.116 | rr=3 |
| 2026Q3 | **−0.155** | −0.336 | −0.555 | −0.662 | rr=3 |

Winners across 14 quarters: rr=10 six times, rr=8 four, rr=3 four.
**Mean correlation of the RR profile between consecutive quarters: −0.00.**

High RR wins in trending markets, low RR in chopping/declining ones — and you cannot
predict which is next. Design period says rr=10 is best (+0.2482); the holdout says rr=3.
A complete flip.

**This is why the holdout's apparent preference for rr=3 must NOT be acted on.** Doing so
would fit the holdout and reproduce the exact error this study exists to avoid.

## 6. Holdout detail (diagnostic only — not a basis for selection)

By month: Feb −0.043, Mar −0.122, Apr −0.224, **May +0.121**, Jun −0.139, **Jul −0.423**.
By side: long −0.032 (PF 0.97), short −0.169 (PF 0.82).
By type: HID_BULL −0.019, REG_BULL −0.055, HID_BEAR −0.126, REG_BEAR −0.271.

Longs and hidden-bullish are close to breakeven; shorts and regular-bearish carry the
damage. Consistent with a market that stopped falling.

## 7. Conclusion

**There is no better set of symbols and configs to be found in this search space.** Not
because the search was weak — it was widened to 522 symbols and run with a proper
firewall — but because:

1. selection does not transfer across a regime boundary (corr +0.043), and
2. the entire strategy family is negative on untouched 2026 data, at every RR, on both
   sides, for 72% of symbols.

Re-selecting now would produce a config set that back-tests beautifully and fails on
contact, exactly as the current one did.

**Recommendation: do not re-fit. Keep the current config frozen** (it is no worse than any
alternative found here), keep base risk at 0.30%, and let the shadow-R advisory tell you
when the regime turns. When it does, the *existing* configs will work again — because what
changed was never the configs.

The one genuinely repeatable finding across this whole study is negative and worth
stating plainly: **per-symbol RR/ATR optimisation on this strategy is noise.** If the
strategy is ever rebuilt, it should use one global RR/ATR — the search for per-symbol
parameters has now failed twice under two different methodologies.
