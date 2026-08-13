# AUDIT B — Execution and Cost

Read-only audit. No files under `autobot/` or `config.yaml` were touched. No network calls
were made — `funding_cache/*.parquet` (277 symbols, already fetched to disk 2026-08-11) was
used as-is; `cache_5m/*.parquet` was checked but stops 2026-08-02, one day before the
`exec_log` window begins, so it could not be used to reconstruct real fill prices.

All dollar/R figures below are on the **current config** (`risk_study/uni_glob_rr10_am3_same.parquet`
— global rr=10, atr_mult=3.0, 2023-06 → 2026-07, entries trimmed 21 days, split at
2026-05-25) unless marked OLD. Source data: `autobot/brokers/bybit.py`, `autobot/core/bot.py`,
`autobot/core/telemetry.py`, `risk_study/results/live_exec_log.parquet` (196 rows, 176
resolved, 2026-08-03→2026-08-11), `funding_cache/`.

---

## Headline

**Cost is close to its structural floor. The one specific fix the codebase already half-built
— a maker take-profit — is worth ~0.00005 R/trade (0.07% of measured cost), not the
meaningful fraction its framing suggests, because the trailing stop (`s3_a1`, armed 2026-08-02)
now intercepts 99.6% of winning trades before they ever reach the fixed TP.** There is no
maker-order lever available on entry or stop-loss without weakening the "position is never
unprotected" guarantee those legs exist for. Funding is real but small (+0.0006 R/trade,
CI crosses zero) and is not worth engineering effort. The measurement infrastructure built to
answer these exact questions (`exec_log`) is broken in two places — both simple, both fixable,
neither yet fixed — so several of the sub-questions below have to be answered by proxy instead
of direct measurement.

Also worth surfacing up front: the **AUDIT_PROMPT's own "~67" figure for mean 1/stop_frac is
stale.** It comes from `universe_chopBOS.parquet`, the *old* per-symbol config (atr_mult
1.0–2.0, rr 3–10 mixed). On the current global config (atr_mult=3.0 flat) mean 1/stop_frac is
**29.5**, roughly half — because atr_mult roughly tripled from the old per-symbol average.
That means "1 bp round trip ≈ 0.0067 R/trade" is also stale; on the current config it's
**≈0.0030 R/trade**. This doesn't overturn the "cost is roughly a third of gross alpha" framing
(0.0715R cost / 0.19–0.29R gross ≈ 25–38%, in the same ballpark either way) but every number
downstream of stop_frac in this report uses the corrected 29.5, not 67, and future work citing
"1 bp = X R" should say which config it's from.

---

## 1. Maker vs taker — establish the order types, then quantify

**All three legs are taker Market orders in the live path today.**

- **Entry**: `place_market()` (`autobot/brokers/bybit.py:757-795`) — `orderType: "Market"`,
  `timeInForce: "IOC"`. No maker option; a BOS-confirmation entry by construction fires when
  price has already broken structure, so this cannot become a resting limit without changing
  what the strategy trades (see below).
- **TP**: `place_market()` attaches `takeProfit`/`tpTriggerBy: "LastPrice"` at
  `bybit.py:782-784` but never sets `tpOrderType`. Bybit's default for an unset `tpOrderType`
  on `/v5/order/create` is **Market**. The fix exists in the codebase — the *dead* function
  `set_tpsl` (`bybit.py:~1080-1110`, calls `/v5/position/trading-stop`, never imported by
  `execute_trade`) explicitly sets `"tpOrderType": "Limit"` with the comment "Limit order for
  Take Profit" — but it was never wired into the live entry path (`bot.py:2145-2151` calls
  `place_market` directly). This is exactly the gap `CLAUDE.md` §9 already flags as dead code.
- **SL**: same pattern — `slOrderType` unset → Market default, both on the original bracket
  and on every trail ratchet via `amend_stop_loss`.

**Quantifying the TP fix — this is the part that changes the recommendation.** Assumed fee
tier (repo cannot verify online, stated per the task): taker 5.5 bps/side, maker 2.0 bps/side
→ 3.5 bps saved on a leg that becomes maker. If *every* winning trade exited exactly at the
original fixed TP, this would be worth something. It doesn't:

| | value |
|---|---|
| trades with `r_result` within 3% of `rr` (i.e., the fixed TP was truly what closed the trade) | **128 / 32,102 = 0.40%** |
| median `r_result` among *all* winners | **2.96 R** (rr is 10) |
| 90th pctile `r_result` among winners | 4.14 R |

The trailing stop arms at MFE ≥ 3R and ratchets a stop, not a limit order — so the overwhelming
majority of wins close through the **SL leg** (the trail-ratcheted stop, still Market), not the
TP leg. Averaged over *all* trades, converting only the TP leg to maker recovers:

```
0.40% of trades × 3.5 bps × mean(1/stop_frac | true-TP trades)
  → avg savings across all trades ≈ 0.00005 R/trade  (≈0.07% of the 0.0715 R measured cost)
```

This is noise-level. **The maker-TP fix from `set_tpsl` is a real, correct fix and there is no
reason not to carry it forward if it's cheap to do — but it should not be sold as a cost
recovery, because under the current trailing-stop config it almost never fires.** It would have
mattered much more before `s3_a1` went live (when the fixed TP was the primary win exit); today
it's close to irrelevant. This is worth re-checking if the trailing stop is ever turned off or
its trigger raised well above typical winner MFE.

**Could the SL leg (or the trail's ratchet amendments) become maker?** No — not without giving
up the guarantee the bracket order exists for. A resting limit stop can fail to fill on a fast
move, which is precisely the scenario the atomic-bracket design (`bot.py:2142-2151`, "position
NEVER unprotected") exists to prevent. This is ~99.6% of all exits (every loss, every trail-cut
win). Recommend leaving it Market.

**Could the entry become a limit order?** Not without changing the strategy. BOS confirmation
already means price has moved through the break level by the time the signal fires; a
"pullback" limit entry at a better price is a different edge that has never been tested here
and would need its own fill-rate/opportunity-cost study — it is not a mechanical, no-new-alpha
change and is out of scope for this workstream. `CLAUDE.md` §11 documents that even a
*known-correct* one-candle timing fix here required a dedicated 3-year A/B
(`bos_ab_summary_*.txt`) before being trusted; a limit-entry redesign would need the same rigor,
not a quick patch.

**Net:** cost is already close to the floor achievable without changing the strategy. The one
concrete "free" fix available (`set_tpsl`'s Limit TP, ported into `place_market`) is real but
worth ~0.00005 R/trade at today's trailing-stop trigger, not a meaningful fraction of the 24.2
bps.

---

## 2. Funding — real, measured from `funding_cache/`, not material

`funding_cache/*.parquet` (277 files, real Bybit `/v5/market/funding/history`, back to
2022-12/2023-02, fetched 2026-08-11, no new network call made) was joined to every trade in
`uni_glob_rr10_am3_same.parquet` by summing `funding_rate` over each 8h settlement between
`entry_time` and `exit_time`, signed so shorts gain when `funding_rate>0` (standard: positive
rate = longs pay shorts) and expressed in R via `/stop_frac` (same unit as `r_result` and
`cost_R`, so it's directly additive/comparable).

| | long | short | overall (book-weighted) |
|---|---|---|---|
| n trades | 11,355 | 20,587 | 31,942 |
| mean funding R/trade | **−0.00526** | **+0.00385** | **+0.00061** |
| 95% CI (weekly block bootstrap, 165 weeks, overall) | | | **[−0.00093, +0.00221]** |

The book is net short ~1.8:1 by realized trade count (not quite the 1.4:1 config ratio cited
in the audit prompt, but in the same direction), so the short-side funding tailwind slightly
outweighs the long-side drag. **The point estimate is positive but the confidence interval
straddles zero — funding is not statistically distinguishable from a wash, and either way it is
an order of magnitude smaller than the ~0.07–0.16 R/trade cost floor.** It is real money (+19.6
R summed over the whole 3-year sample) but not a lever worth engineering time on its own.

**Holding time did roughly double under the current config**, which is exactly why funding
went from irrelevant to marginally positive:

| | mean hold, long | mean hold, short | funding R/trade |
|---|---|---|---|
| OLD config (`uni_glob_rr8_am2_same`, atr_mult~2, rr~8) | 25.2h | 33.7h | **−0.0001** (wash) |
| CURRENT config (atr_mult=3, rr=10) | 54.9h | 71.8h | **+0.0006** |

**Is the backtest's flat funding model right?** Directionally yes (shorts net-earn), but the
constants are off in ways that mostly cancel:

- `FUNDING_LONG = 0.0001`/8h (backtest assumes longs *pay* 1bp/8h). Measured: longs' own
  holding windows actually averaged a slightly *negative* funding rate (≈ −0.0000293/8h,
  i.e., longs on average *received* a trickle) — opposite sign to the assumption, but tiny
  in magnitude either way.
- `FUNDING_SHORT = -0.00003`/8h (backtest assumes shorts *earn* 0.3bps/8h). Measured: shorts'
  windows averaged ≈ +0.0000166/8h funding rate (they earn, correct sign) — about half the
  assumed magnitude.

Net: the backtest's funding model is directionally defensible and small enough in either
direction that getting the exact constant right is not worth chasing. **Funding is real,
measured, mostly explained by the wider stops making holds longer, and immaterial next to cost
and alpha.** No action recommended beyond noting it for completeness.

**Per-trade funding attribution in production does not exist.** `bot.py:2685` hard-codes
`funding_usd=None` on every `exec_log` close, with the comment that Bybit reports funding on
the transaction log rather than the closed-pnl record and "is joined later from
`get_wallet_movement_summary()`" — that join was never built (`get_wallet_movement_summary` is
only called from `telegram_handler.py` for account-wide dashboard aggregates, never keyed back
to a `trade_key`). This is a real gap in the telemetry the repo built specifically to answer
this question, but low priority given how small the effect is.

---

## 3. The fill-price gap — the measurement is broken, not just imprecise

**`exec_log` cannot currently answer the question it was built for.** Direct check on all 196
rows (2026-08-03 → 2026-08-11):

| field | finding |
|---|---|
| `actual_entry == intended_entry` | **196/196 rows, exact** |
| `slippage_frac` | **0.0 in all 196 rows** |
| `fee_usd` | **null in all 196 rows** |
| `funding_usd` | **null in all 196 rows** (expected — see §2, never joined) |

This matches what an earlier run of `risk_study/analyze_live_cost.py` already found (its output
`live_measured_cost.csv` is all blank) — this is not new breakage, just newly root-caused here.

**Root cause, `actual_entry`:** `bot.py:2164`:
```python
actual_entry = float(result.get('avgPrice') or entry_price)
```
`result` is the response body of `/v5/order/create` (`bybit.py:795`, `place_market`'s direct
return). Bybit's v5 `order/create` response does not carry a fill price — only `orderId` /
`orderLinkId` — because market orders fill asynchronously after the create call returns; the
fill price has to be fetched separately (the codebase already knows this: `get_order_status`,
`bybit.py:869-897`, reads `avgPrice` from `/v5/order/realtime` or `/v5/order/history`, and
`place_limit`, `bybit.py:851-866`, does exactly this follow-up poll after its own order calls —
`place_market` just never does it). So `result.get('avgPrice')` is always `None`/missing and
the code falls back to `entry_price` (== `intended_entry`) every single time. **This is not
"slippage happened to be zero" — it is the fallback branch firing on every trade.**

**Root cause, `fee_usd`:** `bot.py:2676`: `_fee = matched.get('closedFee')`, where `matched`
comes from `get_closed_pnl` → `/v5/position/closed-pnl`. That endpoint's documented response
fields are `closedPnl`, `avgEntryPrice`, `avgExitPrice`, `cumEntryValue`, `cumExitValue`,
`leverage`, etc. — **there is no `closedFee` field in this endpoint at all.** `matched.get(...)`
always returns `None`, and `float(_fee) if _fee not in (None, '') else None` writes `None`.
Real per-trade fees exist on `/v5/execution/list` (`execFee` field) or can be backed out of
`cumEntryValue`/`cumExitValue` vs `closedPnl`, but neither is queried here.

**Can this be recovered from data on disk?** No, for the same window. `cache_5m/*.parquet`
(the only sub-hourly cache in the repo) tops out at **2026-08-02 08:05:00**, one day before the
first `exec_log` row (2026-08-03 13:26). No network calls were made per instructions, so real
sub-minute prices for this exact window are unavailable. **This means the actual sign and
magnitude of the fill-price gap cannot be measured from anything currently in this repo — the
telemetry needs the two code fixes above, then ~2-4 weeks of fresh live data (196 trades took
8 days; 60+ resolved trades is the bar the trail-shadow module already uses elsewhere for
"enough data").**

**A back-of-envelope order-of-magnitude estimate**, using only what's already in `exec_log`
(no external data): treat the trade's own 1H ATR as a proxy for hourly volatility and scale by
`sqrt(fill_lag_sec / 3600)` to estimate the 1-sigma price drift over the observed
processing lag (median 75.9s, matching the audit prompt's cited 74s; p95 122.5s, matching the
cited 122s; one extreme outlier at 1567s / 26 minutes, late in the 277-symbol sweep):

```
expected_drift_frac ≈ (atr/entry) * sqrt(fill_lag_sec/3600)
expected_drift_R    ≈ expected_drift_frac / stop_frac
```

Median expected 1-sigma drift: **~12.0 bps of notional / ~0.107 R** — versus the backtest's
flat assumption of 3 bps entry slippage (`SLIPPAGE_PER_SIDE`). This is a rough proxy (ATR is a
range statistic, not a per-second return stdev, so this likely somewhat overstates true
diffusion — but the direction of the finding, that the assumed 3bps is probably too small
relative to what a 75+ second delay across a volatile-alt book should produce, looks right).
**Whether this is a real cost or noise depends entirely on sign, which cannot be recovered**:
if fills are unbiased around the intended price, this washes out over many trades and is
"noise" in the sizing sense (it does add variance but not expected cost); if BOS-confirmation
entries systematically chase already-moving price (plausible — the signal *is* a momentum
continuation trigger), the drift is adverse on average and this becomes a real, non-trivial
cost comparable to or larger than the entire measured 24.2bps round trip. **This is the single
most consequential open question in this workstream and it is currently unanswerable — not
because it's hard, but because the two-line telemetry bug above has been silently discarding
the answer since the feature was built.**

**Separately — does the SL/TP/qty-not-recomputed-from-actual-fill issue matter?** `bot.py`
computes `sl_price`/`tp_price`/`qty` from `entry_price = df.iloc[-1]['open']`
(`bot.py:2126-2131`, well before this) and never recomputes them from `actual_entry` once the
market order reports back (`bot.py:2164` computes `actual_entry` but nothing downstream reads
it except for logging/notification/`ActiveTrade.entry_price`). Given the drift estimate above
(median ~0.1R equivalent price movement possible over the fill lag), realized risk-per-trade
can plausibly drift several percent from planned on a given trade — but this is symmetric
(over- and under-sized in roughly equal measure) unless entries are directionally biased the
same way as the price-gap question above, so its effect on the *mean* R is likely small even if
its effect on any single trade's dollar risk is not. Cannot be sized more precisely without the
same fix (real `actual_entry`) this section already asks for.

---

## 4. Cost sensitivity of the whole strategy

Recomputed on `uni_glob_rr10_am3_same.parquet`, entries trimmed 21 days, split at
2026-05-25, weekly block bootstrap (2000-3000 draws) for CIs. `net_R = r_result −
(bps/10000)/stop_frac`.

| round trip | PRE (n=29,722) mean net R | PRE 95% CI | POST/holdout (n=1,702) mean net R | POST 95% CI |
|---|---|---|---|---|
| 11.0 bps | 0.2549 | [0.1244, 0.3933] | 0.0723 | [−0.3335, 0.5098] |
| 15.0 bps | 0.2432 | [0.1124, 0.3812] | 0.0608 | [−0.3490, 0.5286] |
| 18.0 bps | 0.2344 | [0.1028, 0.3733] | 0.0521 | [−0.3538, 0.5131] |
| **24.2 bps** (measured) | **0.2162** | [0.0904, 0.3533] | **0.0341** | [−0.3712, 0.4940] |
| 30.0 bps | 0.1992 | [0.0694, 0.3392] | 0.0173 | [−0.3913, 0.5005] |
| 40.0 bps | 0.1699 | [0.0385, 0.3060] | −0.0117 | [−0.4194, 0.4383] |

**Breakeven cost:** ≈**98 bps in-sample**, ≈**36 bps on the holdout** (linear in bps since
`net_R` is linear in `bps` for fixed trades — no need to search). The holdout's point-estimate
breakeven (36 bps) is closer to the measured 24.2 bps than the in-sample figure is, but the
holdout is thin (1,702 trades from a ~6-week window after the 21-day trim, since this universe
file itself only runs to 2026-07-25) and its CI at every cost level tested **already includes
zero at the measured 24.2 bps** ([−0.3712, +0.4940]). This is the same "alpha below cost,
can't be confirmed or ruled out at current cost" picture as
`STRATEGY_VERDICT_2026-08-11.md` — cost sensitivity itself is not the story here; the strategy
is robust to cost swings up to 40bps *in-sample*, but the holdout sample is too small and too
recent to say anything with confidence about whether today's true edge clears today's true
cost. That is a signal-layer / sample-size question, not an execution-cost one.

**How much of the edge would maker-TP recover?** ~0.00005 R/trade (§1) — invisible against a
gross alpha of 0.10–0.29 R/trade and a cost floor of 0.07–0.16 R/trade. It would not move any
number in the table above by a visible amount.

---

## Ranked findings

| # | Finding | Effect | In-sample / holdout | Confidence | Action |
|---|---|---|---|---|---|
| 1 | `exec_log` telemetry is broken: `actual_entry` always falls back to `intended_entry` (Bybit `order/create` has no `avgPrice`, `place_market` never polls for it); `fee_usd` always null (`closedFee` isn't a real field on `/v5/position/closed-pnl`) | Unknown — this is what's blocking measurement of the single largest open cost question (§3) | n/a (measurement gap) | **High confidence the bug is real** (verified against both the API response shape and the dead code's own workaround pattern in `place_limit`/`get_order_status`); low confidence on what the true fill-gap number is until fixed | **Fix this week.** Two small, self-contained code changes (poll `/v5/order/realtime` for `avgPrice` after `place_market`; pull fee from `/v5/execution/list` or back it out of `cumEntryValue`/`cumExitValue`). Does not touch trading logic, only telemetry — low risk to ship, high value to have running before the next audit. |
| 2 | Fixed-TP → maker Limit order (the `set_tpsl` fix already written, never wired into `place_market`) | ≈+0.00005 R/trade average (0.4% of trades even reach the fixed TP once `s3_a1` trailing is armed) | Same order under both windows — this is a structural, not a sample, result | High confidence in the smallness of the effect (direct count of true-TP exits); the maker/taker bps assumption is stated, not verified | **Not worth doing for cost recovery.** Fine to carry forward for hygiene/consistency if touching this code anyway, but do not report it as a cost win — it would have mattered before `s3_a1` shipped, not after. |
| 3 | Funding | +0.0006 R/trade point estimate, CI [−0.0009, +0.0022] crosses zero | Full 3yr sample only (too few holdout trades to split meaningfully) | Medium — real data, real join, but the effect itself is statistically indistinguishable from zero | **No action.** Immaterial next to both alpha and cost. Not worth building the missing `funding_usd` join for its size alone (though it's cheap enough that it could ride along with fix #1's telemetry work). |
| 4 | SL/TP/qty never recomputed from `actual_entry` after the market fill | Directionally plausible (drift proxy ~0.1R at 1-sigma over the fill lag) but likely near-symmetric on mean R; real magnitude blocked by finding #1 | Cannot be measured yet | Low — this is a hypothesis with a plausible mechanism, not a measured number | **Needs evidence, not action yet.** Once #1 is fixed, this becomes directly measurable (compare intended vs. actual `risk_usd_at_entry`) within weeks. |
| 5 | Cost floor vs. strategy edge | Strategy tolerates cost up to ~98bps in-sample / ~36bps holdout point estimate before breakeven; current 24.2bps has comfortable in-sample margin but a holdout CI that already straddles zero | Both shown | High confidence in the arithmetic; the underlying question ("is there a real edge left") is outside this workstream's scope | **Not an execution problem.** Cost is not the thing standing between this strategy and profitability on the holdout window — sample size and edge decay (workstream A/C territory) are. |

**"Nothing found" is part of this result, stated plainly:** the maker-order question — flagged
in the prompt as "the biggest single question" — resolves to **no material recoverable cost**
once you account for what the trailing stop already does to the TP exit path, and the entry/SL
legs cannot safely be made maker without giving up guarantees the bot depends on. That is a real
answer, not a non-finding — it means engineering effort here is better spent fixing the broken
*measurement* of cost (finding #1) than chasing an order-type change that the data says is
worth 0.07% of the round trip.

---

## Files

- `risk_study/agent_out/audit_b/AUDIT_B.md` — this report
- `risk_study/agent_out/audit_b/results.csv` — every quantified number above, one row per metric
