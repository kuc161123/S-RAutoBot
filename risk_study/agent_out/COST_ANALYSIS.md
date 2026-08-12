# Honest round-trip execution cost for the divergence bot

Agent: AGENT-COST. All work in `risk_study/agent_out/`, no repo files touched, no network
calls, no API credentials used. Everything below is either read directly from source
(cited `file:line`) or computed from local parquet/CSV files already in the repo.

**Bottom line up front:** central estimate **28 bps round trip**, defensible range
**18–45 bps**. The 18 bps figure used by most backtest engines is a bare fee floor with
zero spread/slippage; the 34.1 bps figure in `STRATEGY_VERDICT_2026-08-11.md` is closer to
a genuine all-in estimate than its "slippage" label suggests, but it is a small,
unreproducible-from-here, no-CI number. `fee_drag_r` in the 1,357-trade recon file is a
**modelled constant**, and — this was not previously documented — it isn't even the 18 bps
constant the rest of the repo claims; it's 12 bps, fee-only, with the slippage term never
actually added to the formula despite being defined.

---

## 1. The fee floor — what order type is each leg, and what does it cost

**Entry.** `execute_trade` (`autobot/core/bot.py:2145`) calls `self.broker.place_market(...)`,
which builds `orderType: "Market"` (`autobot/brokers/bybit.py:771`). Entry is unconditionally
a taker market order.

**Exit — TP and SL when triggered.** The bracket is attached to that same market order via
`data["takeProfit"]`/`data["stopLoss"]` with `tpTriggerBy`/`slTriggerBy: "LastPrice"`
(`bybit.py:783-788`). Critically, `place_market` never sets `tpOrderType` or `slOrderType`.
Compare with the *dead* `set_tpsl` (`bybit.py:1055-1113`, unreferenced by any live path —
confirmed by CLAUDE.md §9 and by grep), which explicitly requests
`"tpOrderType": "Limit"` for the take-profit leg "for better fills" — i.e. someone on this
team already knew a Limit TP would be cheaper. That knowledge was never carried into the
live path. `amend_stop_loss`, the function that actually moves the trailing stop
(`bybit.py:1282`), explicitly sets `"slOrderType": "Market"` (`bybit.py:1331`).

On Bybit's v5 API, when `tpOrderType`/`slOrderType` are omitted the default is `Market` for
both legs. I could not re-verify this against live Bybit docs (no network access permitted
in this task), so I'm flagging it explicitly: **this is my recollection of Bybit's
documented default, not something I re-confirmed today.** But it's also the conservative
assumption to make, and it's corroborated by the code itself — the team clearly knows how
to request a Limit TP (`set_tpsl` does it) and chose not to on the only path that actually
runs. Given that, and given `amend_stop_loss` explicitly forcing `Market` for the trailing
leg, the working conclusion is:

**Both entry and exit — TP and SL alike — fill as taker/market orders. No leg of this
strategy earns a maker rebate.**

**Fee rate.** Bybit's published standard (non-VIP) USDT-perpetual taker fee has been in the
0.055%–0.06% range in recent fee schedules (I'm using this from training-time knowledge; I
was not able to check Bybit's current fee page from this sandbox, so treat the exact figure
as ±a few bps uncertain, not authoritative). `config.yaml`'s own assumption
(`fee_pct: 0.0006`, `config.yaml:15`) — used by `backtest_production_correct.py:52-60` and
`validate_6month.py` — is 0.06%/side, at the high end of that range, so it's a reasonable
and slightly conservative planning number.

**Fee-only round trip = 2 × taker fee ≈ 0.11–0.12% = 11–12 bps.** This is the hard floor —
no config, no limit order, no better routing beats this without Bybit changing the account's
VIP tier. Everything past this floor is spread/slippage/funding.

---

## 2. Auditing the 34.1 bps claim

`grep -rn "34.1\|trailstats"` across the repo turns up only `STRATEGY_VERDICT_2026-08-11.md`
(§2.3, line 188): *"Calibrated to live `/trailstats` (n=195), the true round trip is 34.1
bps."* That number is `mean_slip` out of `autobot/core/trail_shadow.py`
(`_stats`, lines 445-452):

```python
slip = []
for r in rows:
    if not r['closed'] or r['actual_r'] is None:
        continue
    modelled = float(r['r_trail']) if r['live_trailing'] else float(r['r_fixed'])
    slip.append(float(r['actual_r']) - modelled)
mean_slip = (sum(slip) / len(slip)) if slip else None
```

`actual_r` is `r_value = pnl_usd / trade.risk_usd_at_entry` from real Bybit closed-PnL
(`bot.py:2791` feeds it in), i.e. genuine realized dollars, net of real fees/funding, over a
fixed dollar risk target. `modelled` (`r_fixed`/`r_trail`) comes from `walk_fixed`/
`walk_trail` (`trail_shadow.py:364-368`), a **pure price-level replay** off freshly re-fetched
klines (`resolve_pending`, `trail_shadow.py:340-379`) — it contains **zero fees, zero
funding, and zero spread**. So on its face, `mean_slip` = (real, all-in dollar R) −
(frictionless price-level R) should equal *all* real friction: fees + funding + exit-side
slippage, plus whatever the replay gets wrong.

I traced one thing that could have contaminated this in a specific way and it turns out
**not to**: `log_open` (`trail_shadow.py:266-283`) stores `trade.entry_price`, and in
`bot.py:2164/2201`, `trade.entry_price = actual_entry` — the real market-order fill price,
not the planned candle-open price the SL/TP levels were originally sized from
(`bot.py:2167-2169` even has a comment calling that intent-vs-reality gap "unmeasured
today"). Since the modelled replay starts from the *same* `actual_entry` that the real
dollar P&L is computed against (both use the same qty, sized off the same original
`risk_dist`), the entry-fill drift cancels out of the subtraction. **`mean_slip` does not
capture entry-side slippage — that's a separate, still genuinely unmeasured cost (see below)
— but it isn't contaminating this number either way.**

What *could* still be contaminating it, both ways:

- **Pessimistic bias (understates true cost):** the resolver scores an ambiguous bar
  (stop and TP both touched in the same 1H candle) as the stop — "SL-wins-ties"
  (`trail_shadow.py:27-31`). That makes `modelled` *lower* than a fair coin-flip would, which
  pushes `mean_slip = actual − modelled` *up* (toward looking like less cost). This bias is
  explicitly documented as "conservative" for the `r_trail − r_fixed` comparison the module
  was built for, but nobody has reasoned through its effect on `mean_slip` specifically — and
  the effect there runs the other way, toward *understating* cost.
- **Optimistic bias (overstates true cost):** the replay assumes exact fills at the stop/TP
  *price level* the instant it's touched; real market orders triggered off `LastPrice` can
  slip past that level, especially on the SL side during the fast moves that produce most of
  this strategy's exits (win rate ~14%, so ~86% of trades are stop-outs). If the real fill is
  consistently worse than the modelled touch-price, `mean_slip` overstates true friction.
- **No funding model at all**, so any funding actually paid flows entirely into `mean_slip`
  as if it were slippage — which is fine for our purposes (we want an all-in number) but
  means the 34.1 bps figure is not "spread cost," despite how §2.3 of the verdict doc labels
  it.
- **Re-fetch discrepancy:** `resolve_pending` re-fetches klines fresh from the broker
  (`trail_shadow.py:344-345`) rather than replaying the exact candles the live bot saw in
  real time. Historical kline revisions or timestamp/bar-boundary mismatches are a plausible,
  un-quantified source of noise (this is exactly the kind of bug class CLAUDE.md §6.1 lists
  four previously-found instances of).
- **n=195, no confidence interval reported anywhere in the code** — `_stats` computes a
  t-stat for `mean_delta` (the fixed-vs-trail comparison) but never for `mean_slip`. There is
  no way to know from the code whether 34.1 bps is precise to ±5 bps or ±20 bps.
- **This number lives only in Postgres** (`trail_shadow` table) — it is explicitly
  unavailable to this agent, so it cannot be independently reproduced here at all, only
  audited for mechanism.

**Verdict: 34.1 bps is better-founded than its "slippage" label implies — mechanically it's
close to an all-in round-trip cost estimate (fees + funding + exit slippage), not a spread
measurement — but it is a small-sample (n=195), no-CI, Postgres-only number with at least one
identified bias pulling it low (SL-wins-ties) partially offset by at least one pulling it
high (touch-price vs real fill), net direction unresolved. Treat it as directionally
credible and roughly the right order of magnitude, not as a precise measurement.**

---

## 3. `fee_drag_r` in the 1,357-trade recon file — is it modelled or measured?

Tested the hypothesis numerically against `verification_results/07_live_vs_backtest_recon.csv`
(1,357 rows). For rows with `r_raw == -1.0` (n=1,040 — clean stop-outs, so
`stop_frac = |exit_price − entry_price| / entry_price` is unambiguous):

```
implied_const = fee_drag_r * stop_frac
mean   = 0.0012000056
std    = 0.00000092      (relative std ≈ 0.08%)
min/max = 0.0011954 / 0.0012044   (rounding noise only)
```

**This is a constant to 5+ significant figures — it is not measured, it is computed.**
`fee_drag_r` is `0.0012 / stop_frac`, not `0.0018 / stop_frac` as I expected from
`backtest_production_correct.py`'s `ROUND_TRIP_COST = 0.0018` and as
`STRATEGY_VERDICT_2026-08-11.md` §2.3 claims ("Engines charge `ROUND_TRIP_COST = 0.0018`").

Tracing where this CSV's live ledger (`validation_6month_trades.csv`, loaded via
`LIVE_LEDGER` in `verify_overfit.py:51`) actually comes from: `validate_6month.py:432/450`:

```python
fee_drag = (FEE_PCT * 2 * entry_price) / sl_dist if sl_dist > 0 else 0
```

`FEE_PCT = config['execution']['fee_pct']` = 0.0006 (`config.yaml:15`). So
`fee_drag = 2 * 0.0006 / stop_frac = 0.0012 / stop_frac` — **fee only, doubled for round
trip, with the round-trip cost coming out to exactly 12 bps.** `SLIPPAGE_PCT` is loaded at
`validate_6month.py:59` (0.0003) and even printed in the script's own banner ("Slippage:
0.03% | Fee: 0.06% per side", `validate_6month.py:532`) — **but it is never added into the
`fee_drag` formula.** The 12 bps constant this dataset actually charges is *half* of the
18 bps the script's own header claims and half of what `backtest_production_correct.py`
uses elsewhere in the same repo.

For confirmation, I checked the *other* dataset `STRATEGY_VERDICT_2026-08-11.md` cites for
the 0.0018 constant — `halt_universe_live.parquet` (38,753 rows, `fee_r` column). There,
`implied_const = fee_r * stop_frac` is **0.0018000000 exactly, std ≈ 5e-19 (floating-point
noise only)** across all 38,753 rows — so that specific claim in the verdict doc checks out.

**Net finding, not previously documented anywhere I found in the repo: the codebase has at
least two different hardcoded cost constants in play across its own "validated" datasets —
18 bps (`halt_universe_live`, `backtest_production_correct.py`) and 12 bps, fee-only, no
slippage (`validation_6month_trades.csv` → `verify_overfit.py`'s recon →
`OVERFIT_VERDICT.md`'s `current_avg_fee_drag_r`). Both are assumptions dressed up as inputs
to a "live vs backtest reconciliation." Neither has ever been measured against a real fill.
`measure_real_costs.py` (repo root) is the one script that would measure it directly from
Bybit's `/v5/execution/list` (actual fee, actual maker/taker flag, actual rate) — it requires
`BYBIT_API_KEY`/`BYBIT_API_SECRET`, which are not present locally, so per this task's
constraints it was read but not run.**

---

## 4. First-principles slippage estimate

**Position notional is tiny.** Using the taper schedule and regime multiplier from
CLAUDE.md §5 (`risk_per_trade` base 0.3% at low balances, tapering down as balance grows;
regime multiplier 0.1–1.0) and this universe's mean `stop_frac` ≈ 2.2% (see §5 below),
implied notional (`risk_usd / stop_frac`) works out to roughly:

| balance | regime mult 1.0 (favorable) | regime mult 0.1 (critical) |
|---|---|---|
| $1,500 | ≈ $205 | ≈ $20 |
| $10,000 | ≈ $950 | ≈ $95 |

These are small clips by any perpetual-futures standard — market impact from walking the
book is not the binding constraint here. **The binding constraint is the quoted spread
itself** (crossing it as a taker) plus adverse selection at the moment of the fill, which
matters a lot for this strategy because ~86% of trades exit via stop-loss (win rate ~14%),
i.e. the majority of exits happen during the kind of fast, one-directional move that widens
spreads at exactly the wrong moment.

**Universe liquidity, measured from `cache_3yr_1h/`** (turnover column = USDT notional
traded per 1H bar; computed per enabled symbol from `config.yaml`'s 277-symbol universe,
last ~90 days of each symbol's cached history, median hourly turnover):

```
n symbols with cache data: 277 / 277
median hourly turnover           :  $20,686
25th percentile                  :   $8,477
10th percentile                  :   $5,106
5th percentile                   :   $3,312
90th percentile                  :  $440,572
symbols below $10k/hr turnover   :   85 / 277  (31%)
symbols below $50k/hr turnover   :  189 / 277  (68%)
symbols above $5M/hr turnover    :    6 / 277  (BTC, ETH, SOL, XRP, HYPE, ZEC-adjacent)
```

Full table: `risk_study/agent_out/liquidity_by_symbol.csv`. The universe is extremely
bimodal — a handful of majors carry $1–130M/hr, while the median symbol trades ~$20k/hr and
a third of the book is under $10k/hr. I don't have order-book/spread data locally (no
network access), so I can't measure quoted spread directly, but turnover this thin is a
reliable proxy for wide quoted spreads on a venue like Bybit — liquid majors (BTC, ETH,
SOL, XRP) typically run ~0.5–3 bps quoted spread; well-covered mid-caps ~3–8 bps; the
bottom quartile of a 277-symbol alt-perp universe is plausibly 10–30+ bps, wider still
during the volatile bar that actually triggers a stop.

**Low / central / high slippage-only estimate (spread + adverse-selection, excluding fees
and funding):**

- **Low (~6 bps):** if fills were dominated by the liquid tail (BTC/ETH/majors-weighted),
  consistent with tight spreads and small clip sizes not moving the book.
- **Central (~15 bps):** turnover-weighted toward the median/thin end of the universe,
  reflecting that the strategy runs across all 277 symbols roughly proportionally to how
  often each fires signals (not weighted toward the majors), plus the adverse-selection tilt
  from 86% of exits being stop-outs during fast moves.
- **High (~30 bps):** if a meaningful share of fills land in the bottom liquidity quartile
  (85 symbols under $10k/hr median turnover) during volatile stop-triggering bars, where
  effective spread can be several multiples of the quoted resting spread.

Add the 11–12 bps fee floor (§1) and a small funding allowance (positions are typically held
hours to a few days; net book is short-heavy per CLAUDE.md §12, and funding on USDT perps is
usually single-digit bps per day — call it 1–3 bps average, sign ambiguous given the mixed
long/short book) and the **all-in range is ~18–45 bps**, bracketing both the repo's 18 bps
floor and the 34.1 bps live-calibrated figure comfortably, with 34.1 bps sitting roughly at
the center-to-high end of what first principles would predict for a 277-symbol,
alt-heavy, stop-out-dominated book.

---

## 5. Bottom line — cost_R at each candidate bps level

`stop_frac` distribution from `risk_study/universe_chopBOS.parquet` (n=32,929, mean
2.21%, median 1.82%, IQR 1.26%–2.69%):

| round-trip bps | source | mean cost_R (bps×1e-4 / stop_frac, per-trade average) | median cost_R |
|---|---|---|---|
| 11 | fee-floor low (taker ≈0.055%×2) | 0.0741 | 0.0605 |
| 12 | `validation_6month_trades.csv`'s actual (undocumented) constant / fee-floor high | 0.0808 | 0.0660 |
| 18 | repo's most common assumption (`backtest_production_correct.py`, `halt_universe_live`) | 0.1212 | 0.0991 |
| 25 | mid-range first-principles estimate | 0.1684 | 0.1376 |
| **34.1** | live-calibrated (`trailstats`, n=195) | **0.2297** | 0.1877 |
| 45 | high end of first-principles range | 0.3031 | 0.2477 |

Cross-check: `STRATEGY_VERDICT_2026-08-11.md` reports "measured execution cost is 0.232
R/trade" at 34.1 bps — this independently-recomputed 0.2297 matches to within noise,
confirming the same `stop_frac` distribution (or one statistically identical to it) underlies
both numbers.

**Against the honest gross alpha of ~+0.12 R/trade (per `STRATEGY_VERDICT_2026-08-11.md`
§0):**

- At the repo's bare fee floor (11–12 bps): cost ≈0.07–0.08R — alpha *barely* clears cost.
- At the commonly-assumed 18 bps: cost ≈0.12R — **alpha ≈ cost, breakeven**, before any real
  spread is counted.
- At 25 bps: cost ≈0.17R — net negative by ≈0.05R/trade.
- At the live-calibrated 34.1 bps: cost ≈0.23R — net negative by ≈0.11R/trade, i.e. **the
  alpha is about half the cost of harvesting it**, matching the verdict doc's framing exactly.
- At 45 bps: cost ≈0.30R — net negative by ≈0.18R/trade.

**Under no defensible assumption — not even the bare 11–12 bps fee floor with zero spread,
zero slippage, zero funding — does this strategy clear its cost by a comfortable margin.**
The 18 bps figure used throughout most of the repo's backtests is provably the least
realistic of the numbers on this table: it is neither the exchange's actual fee-only floor
(11–12 bps) nor a measurement of real friction (34.1 bps, itself imprecise); it is simply
`config.yaml`'s assumed `fee_pct`/`entry_slippage_pct` doubled, and even that assumption is
inconsistently applied — the repo's own 1,357-trade "live" reconciliation silently drops the
slippage term and only charges 12 bps.

---

## Files produced

- `risk_study/agent_out/COST_ANALYSIS.md` — this report.
- `risk_study/agent_out/liquidity_by_symbol.csv` — median/mean hourly turnover per enabled
  symbol (277 rows), sourced from `cache_3yr_1h/*.parquet`.
