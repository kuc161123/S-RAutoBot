# Verified Risk / Position-Sizing Specification

Traced against source at commit `263a4ff` (working tree, `config.yaml` has local
uncommitted edits — verified against the tree as it sits on disk, not the last commit).
All line numbers are `file.py:line`. CLAUDE.md was used for orientation only; every
claim below was re-verified by reading the cited lines directly.

---

## 1. Exact position-sizing chain: config value → order `qty`

Entry point: `execute_trade()`, `autobot/core/bot.py:1809`. All steps below occur inside
that function, in this order.

```
1. base_risk = risk_config['risk_per_trade']                         config.yaml:43 = 0.003
                                                                       read at bot.py:718

2. taper: for (threshold, risk) in taper_schedule, ascending order,
   base_risk = risk if wallet_balance >= threshold (last match wins)  bot.py:721-725
   taper_schedule                                                    config.yaml:146-160
   e.g. wallet >= 1500 -> 0.003, >=3000 -> 0.0028, ... >=40000 -> 0.0014

3. (label, regime_mult, diag) = get_regime_status()                  bot.py:611, called at 727 (inside
                                                                       get_adaptive_risk) and again at 1843
   regime_mult in {1.0, 0.5, 0.25, 0.1}, driven ONLY by win-rate/avg-R
   of the last 20 closed trades (n_trades<10 -> forced 0.1 "critical") bot.py:642-690

4. final_risk_frac = base_risk * regime_mult                         bot.py:728  (get_adaptive_risk return)

5. margin_pct = get_adaptive_risk(balance=wallet_balance)             bot.py:2004
   risk_amount = account_balance * margin_pct                        bot.py:2005
   -- NOTE: step 2's taper lookup uses WALLET balance (get_wallet_balance(),
      excludes unrealized PnL), but step 5's dollar risk multiplies
      EQUITY (get_balance(), which prefers the "equity" field including
      unrealized PnL). This wallet/equity basis mismatch is deliberate
      (see CLAUDE.md §5) and confirmed live: bot.py:1998-1999,2004-2005.

6. [LONG BULL-BOOST] if side=='long' and long_bull_boost != 1.0
   and overlays ramped on and BTC 1H close > EMA200:
       risk_amount *= long_bull_boost                                bot.py:2007-2016
   long_bull_boost = 1.3                                             config.yaml:80

7. [GATES — can abort the trade, do not modify risk_amount]
   net-directional cap check                                        bot.py:2022-2042 (§3 below)
   gross open-risk cap check                                        bot.py:2047-2067 (§3 below)

8. sl_distance = atr * atr_mult   (atr_mult per-symbol/per-divergence-type
   from config.yaml `configs:` blocks; atr from df.iloc[-2] closed candle)  bot.py:1985-1990

9. raw_qty = risk_amount / sl_distance                                bot.py:2075

10. leverage = broker.get_max_leverage(symbol); set_leverage(symbol, leverage)
    (exchange MAXIMUM for that symbol, re-read after set in case the
    exchange's risk-limit tier capped it lower)                       bot.py:2078-2081

11. position_value = raw_qty * entry_price
    required_margin = position_value / leverage                      bot.py:2084-2085

12. available_balance = account_balance - sum(positionIM for all
    open exchange positions, this account, not just bot-tracked)      bot.py:2089-2095
    if required_margin > available_balance: ABORT (no trade)          bot.py:2098-2100

13. qty_step = broker._get_precisions(symbol)[1]  (exchange lot size,
    default "0.001" only on a fetch failure)                          bot.py:2104, bybit.py:1115-1155
    position_size_qty = floor(raw_qty, qty_step)   [Decimal ROUND_DOWN] bot.py:2105-2108
    if position_size_qty <= 0: ABORT                                  bot.py:2115-2117

14. Order placed: place_market(qty=position_size_qty,
    take_profit=tp_price, stop_loss=sl_price)                         bot.py:2145-2151, bybit.py:757
```

**Closed-form formula** (assuming no gate blocks the trade and qty_step rounding is
negligible):

```
qty ≈ [ equity × taper(wallet) × regime_mult × (long_boost if applicable) ] / (ATR × atr_mult)
```

where `equity = get_balance()` (bybit.py:346, prefers "equity" field, i.e. includes
unrealized PnL) and `wallet = get_wallet_balance()` (bybit.py:392, "walletBalance"
field, excludes unrealized PnL) — two different account snapshots feeding the same
formula.

---

## 2. Four distinct quantities, worked example

**Definitions, each traced to source:**

- **Risk per trade** ($) = `risk_amount` from step 5/6 above = fraction of *equity*
  lost if the stop fills exactly at `sl_price`. This is what `risk_per_trade` in
  config.yaml actually controls (bot.py:2005).
- **Notional exposure** ($) = `position_value = raw_qty * entry_price` (bot.py:2084) —
  the dollar size of the market order, i.e. what actually moves with 1:1 price
  sensitivity.
- **Margin consumed** ($) = `required_margin = position_value / leverage` (bot.py:2085)
  — what Bybit locks against the position (`positionIM`), read back at bot.py:2092.
- **Leverage set on exchange** = `broker.get_max_leverage(symbol)` (bybit.py:433) — the
  exchange's own maximum for that instrument (from `leverageFilter.maxLeverage` on
  `/v5/market/instruments-info`, bybit.py:445-446), applied via `set_leverage`
  (bybit.py:459). There is no config ceiling anywhere in the codebase — confirmed by
  grep, no `max_leverage` key is read in `bot.py` or `symbol_rr_mapping.py`. Leverage
  is a margin-headroom lever only; it does not change `qty` (that comes purely from
  `risk_amount / sl_distance`, step 9, computed *before* leverage is even looked up).

**Worked example** — $1,500 equity, current config, `atr_mult = 1.5`, symbol ATR = 2% of
entry price, regime assumed `favorable` (mult=1.0, i.e. best case / no derate — see
note below for other tiers):

```
base_risk (taper, wallet=$1,500, first rung >= 1500)     = 0.003          (config.yaml:43,147-148)
regime_mult (favorable, illustrative)                     = 1.0
risk_amount = $1,500 × 0.003 × 1.0                         = $4.50

Let entry_price = $100 (arbitrary; only the % relationships matter)
ATR = 2% × $100 = $2.00
sl_distance = ATR × atr_mult = $2.00 × 1.5                = $3.00

raw_qty = risk_amount / sl_distance = 4.50 / 3.00          = 1.5 units
notional = qty × entry_price = 1.5 × 100                   = $150.00

notional / equity ratio                                    = 150 / 1500 = 0.10  (10%)
```

Margin depends on exchange max leverage for the symbol (fetched live, not statically
known from source — see caveat below). At representative leverage tiers:

| leverage | required_margin | margin as % of equity |
|---|---|---|
| 10x | $15.00 | 1.00% |
| 25x | $6.00 | 0.40% |
| 50x | $3.00 | 0.20% |
| 75x | $2.00 | 0.13% |
| 100x | $1.50 | 0.10% |

**Caveat (UNVERIFIED, flagged honestly):** exact per-symbol max leverage is not stored
anywhere static in the repo — `leverage_cache` (bybit.py:25) is populated live from the
exchange at runtime and there is no committed snapshot of it. The margin-column numbers
above are illustrative arithmetic against the formula, not a claim about any specific
symbol's actual max leverage today.

**Regime sensitivity** — the same worked example at other regime tiers (risk_amount
scales linearly, everything else identical): critical (0.1x) → $0.45 risk, $15
notional; adverse (0.25x) → $1.125 risk, $37.50 notional; cautious (0.5x) → $2.25 risk,
$75 notional; favorable (1.0x) → $4.50 risk, $150 notional as above. With
`long_bull_boost=1.3` stacked on a favorable-regime long in a BTC uptrend: $5.85 risk,
$195 notional (13% of equity).

---

## 3. Hard caps/gates — do they bind as risk_per_trade rises?

All are evaluated in `execute_trade`, bot.py:1809-2123, in this order (after the
divergence/BOS/CHOP/short-gate filters, which are signal-quality gates, not sizing
gates, so out of scope here):

| Gate | Condition (source) | Function of equity or absolute? |
|---|---|---|
| Net-directional cap | `abs(long_risk_open − short_risk_open) ≤ net_directional_cap × equity` after adding the new trade. `_net_directional_risk_ok`, bot.py:989-1022. Called bot.py:2022. `net_directional_cap = 0.10` (config.yaml:48). Auto-ramped off below `overlay_ramp_min_balance` (currently 0 = always on, config.yaml:56). | **Equity-scaled cap, but trade count to saturation shrinks as risk_per_trade rises** — the dollar ceiling (`0.10 × equity`) does NOT grow with `risk_per_trade`; each trade just consumes more of it. |
| Gross open-risk cap | `Σ risk_usd_at_entry (all open) + new_risk ≤ gross_open_risk_cap × equity`. `_gross_risk_ok`, bot.py:1024-1049ish. Called bot.py:2047. `gross_open_risk_cap = 0.30` (config.yaml:75). | Same shape as above — equity-scaled ceiling, binds after fewer concurrent trades as risk_per_trade rises. |
| Margin sufficiency | `required_margin (= qty×entry/leverage) ≤ available_balance (= equity − Σ positionIM across ALL open exchange positions)`. bot.py:2089-2100. | **Absolute in the sense that it directly tracks real margin usage, which scales roughly linearly with risk_per_trade** (see quantitative reasoning below) — this is the gate most likely to start binding as risk rises, especially with many concurrent low-leverage-symbol positions. |
| Exchange min order size | **Not checked anywhere in the bot.** No `minOrderQty` reference exists in `bot.py` or `bybit.py` (confirmed by grep — zero matches). The only local check is `position_size_qty <= 0` after floor-rounding to `qty_step` (bot.py:2115-2117). A qty below the exchange's real minimum would simply be rejected by Bybit at `place_market` (bybit.py:757) and logged as a failed order (bot.py:2158-2162) — the bot does not pre-validate or retry with a larger size. | N/A — absolute exchange-side floor, invisible to the bot's own logic. Raising `risk_per_trade` makes this LESS likely to bind (bigger risk → bigger qty → further above the floor). |
| qtyStep rounding | `floor(raw_qty, qty_step)`, Decimal ROUND_DOWN (bot.py:2104-2108). Default `qty_step = "0.001"` only on an uncached fetch failure (bybit.py:1155). | Absolute, tiny (rounds down at most one step). Immaterial at any risk level tested; matters most at very LOW risk_per_trade where `raw_qty` is close to one `qty_step` (rounding can zero out a trade) — the opposite direction from the question asked. |

**Quantitative reasoning on the margin check specifically**, given the bot holds 45+
concurrent positions and always sets leverage to the exchange max (bot.py:2078-2081,
"3. Apply Max Leverage to minimize Margin Usage"):

For a single trade, `required_margin = risk_amount / (atr_pct × atr_mult × leverage)`,
since `notional = qty × entry = (risk_amount/sl_distance) × entry = risk_amount /
(atr_pct × atr_mult)` where `atr_pct = ATR/entry_price`. Margin is therefore
**directly proportional to `risk_amount`**, hence to `risk_per_trade`, for a fixed
symbol/config. Doubling `risk_per_trade` doubles the margin every open trade consumes.

Aggregate margin across N concurrent trades scales the same way. Using the gross
open-risk cap as the binding risk ceiling (`Σrisk ≤ 0.30 × equity`), and the worked
example's `atr_pct × atr_mult = 0.03`:

```
Σ margin ≤ (0.30 × equity) / (0.03 × leverage) = 10 × equity / leverage
```

- At leverage = 25x: Σmargin ≤ 0.40 × equity — comfortably inside available balance.
- At leverage = 10x: Σmargin ≤ 1.0 × equity — margin check starts to bind right at the
  gross-risk ceiling.
- At leverage = 5x (a plausible risk-limit-tier cap on some low-liquidity alts —
  UNVERIFIED per-symbol, see §2 caveat): Σmargin ≤ 2.0 × equity — the margin check
  would bind well BEFORE the gross-risk cap does, meaning on low-leverage symbols the
  margin gate — not the gross/net risk caps — becomes the effective ceiling on how many
  positions can be open simultaneously.

This means: raising `risk_per_trade` from 0.3% → 1% → 2% → 3% does not change the
*equity-fraction* ceiling the net/gross caps enforce (still 10%/30% of equity), but it
proportionally raises the margin each trade consumes, so the **margin check becomes the
practically-binding constraint sooner** — especially for symbols whose exchange max
leverage is low — well before the net/gross risk caps would otherwise stop opening
new positions. This is directionally verified from the formula; exact leverage-tier
values per symbol are UNVERIFIED (fetched live, not in repo).

---

## 4. Concurrent position limit

**No max-concurrent-positions limit exists anywhere in the live path.** Verified by
grep across `autobot/core/bot.py` for `max_concurrent`, `max_open_positions`,
`MAX_POSITIONS`, `position_limit`, and any comparison against `len(self.active_trades)`
— the only hit (bot.py:1215) is unrelated sync-reconciliation logic
(`len(actual_open_keys) == 0 and len(self.active_trades) > 0`), not a cap. CLAUDE.md's
claim that `max_daily_loss` and `max_position_size_pct` are dead config keys is also
confirmed: neither string is referenced anywhere outside `config.yaml` itself (grep,
zero hits in `autobot/`).

The only things that indirectly bound concurrency are the net-directional cap (§3, caps
net imbalance, not total count) and the gross open-risk cap (§3, caps total open
`risk_usd_at_entry`, which — for a fixed `risk_per_trade` — caps the *count* of
concurrently open trades at roughly `gross_open_risk_cap / (risk_per_trade × regime_mult)`
per side, not an absolute position count). With `gross_open_risk_cap=0.30` and
`risk_per_trade=0.003` at regime_mult=1.0, that ceiling is ≈ `0.30/0.003 = 100`
concurrently open trades before the gross cap itself blocks further entries (fewer if
regime_mult < 1 raises the effective risk fraction... actually regime_mult LOWERS
risk_amount, which raises the trade-count ceiling — so 100 is the floor/worst case at
full regime multiplier).

**Implied structural maximum**: 277 enabled symbols × up to 4 configs (divergence types)
each = 728 total (symbol, divergence-type) slots (verified: `python3` parse of
`config.yaml`'s `symbols:` block, 277 `enabled: true` symbols, 728 total `configs`
entries across them, matching CLAUDE.md's "277 symbols / 728 configs"). Per-symbol
guards apply: at most one open position per (symbol, side) via the internal duplicate
check (bot.py:1917-1919) and the exchange-side duplicate check (bot.py:1928-1935); AND
the opposite-side guard (bot.py:1936-1953) blocks opening the opposite side on a symbol
that already has a position, unless `risk.allow_opposite_side_entry: true` — which is
**not set** in `config.yaml` (grep confirms no such key present), so it defaults to
`False` (bot.py:1942). Net effect: **at most one open position per symbol** (either
long or short, not both) under current config, so the structural ceiling is **277**
simultaneously open positions (one per enabled symbol), not 554. In practice CLAUDE.md's
cited live state ("40 shorts / 5 longs open", "45+ open at once is normal") describes
the observed range, well under the 277 structural ceiling; there is no code-level
concurrent-count cap enforcing any lower number.

---

## 5. Third-party assertion table — CORRECT / WRONG

| # | Claim | Verdict | Actual (source) |
|---|---|---|---|
| 1 | ~1% risk per trade | **WRONG** | `risk_per_trade = 0.003` (0.3%) at config.yaml:43, further reduced by taper and regime multiplier (down to 0.1x in "critical" regime, i.e. as low as 0.03%). Never reaches 1% under current config. |
| 2 | Market BTCUSDT only | **WRONG** | 277 enabled symbols, 728 (symbol, divergence-type) configs (verified by parsing `config.yaml`'s `symbols:` block). BTCUSDT is one of 277; it is additionally used as a market-regime reference for the long-boost and short-gate overlays (bot.py:734-786), but trading is not restricted to it. |
| 3 | 15-minute entry timeframe | **WRONG** | `strategy.timeframe: '60'` (config.yaml:166) → 1-hour candles. Read at `self.timeframe = self.strategy_config.get('timeframe', '240')` (bot.py:1090). |
| 4 | 4-hour trend timeframe | **WRONG** | Same 1H series is used for everything, including the EMA-200 trend gate (`divergence_detector.py:143`, `ewm(span=200)` on the 1H closes; `bot.py:755` for the BTC-bullish overlay, also 1H). No separate 4H data source exists (the "4h" naming in `fetch_4h_data` is a documented legacy misnomer per CLAUDE.md, confirmed: the function fetches `interval='60'`). |
| 5 | Support/resistance + market-structure strategy | **WRONG** | The live strategy is RSI(14) divergence (regular/hidden, bull/bear) confirmed by a Break-of-Structure check against the `swing_level` (extreme high/low since the triggering pivot), gated by EMA-200 trend — `autobot/core/divergence_detector.py`. Grep for "engulf", "retest", "zone_width", "sl_buffer", "order block" across `autobot/core/*.py` returns zero matches: there is no S/R-zone or market-structure-zone model in the code. |
| 6 | 4H close vs 200 EMA trend filter | **PARTIALLY WRONG** | An EMA-200 gate exists and is applied twice (detection + BOS confirmation, per CLAUDE.md, confirmed via `divergence_detector.py`), but it runs on the 1H series, not 4H (see #4). |
| 7 | Retest-of-zone entry with engulfing confirmation | **WRONG** | No such code exists (see #5). Entry is a market order at the next candle's open after BOS confirmation (`execute_trade`, entry_price = `df.iloc[-1]['open']`, bot.py:1970), not a limit-order retest, and no candlestick-pattern (engulfing) check exists anywhere in `divergence_detector.py` or `bot.py`. |
| 8 | Swing length 3-5 | **WRONG** | Pivot detection is fractal with a FIXED `left=3, right=3` (`MIN_PIVOT_DISTANCE=3`, `PIVOT_RIGHT=3`, divergence_detector.py:30-31) — not a 3-5 range parameter, and it's a pivot-detection window, not a "swing length" in the S/R sense claimed. |
| 9 | Zone width 0.5 ATR | **WRONG** | No "zone" concept exists in the code at all (see #5, #7). |
| 10 | SL buffer 0.2 ATR | **WRONG** | Stop distance is `atr * atr_mult` where `atr_mult` is a per-symbol, per-divergence-type value from `config.yaml` (`configs:` blocks), taking values **1.0, 1.5, or 2.0** (verified distribution: 324 configs at 1.0x, 226 at 1.5x, 178 at 2.0x) — not a fixed 0.2 buffer, and not a "buffer added to a zone" (there is no zone). |
| 11 | Reward:risk 1:2 | **WRONG** | `rr` is per-symbol, per-divergence-type from config.yaml, taking values **3.0, 5.0, 8.0, or 10.0** (verified distribution: 124 configs at 3.0, 153 at 5.0, 179 at 8.0, 272 at 10.0 — the mass is at 8-10, not 2). CLAUDE.md's derived ~13.9% breakeven win rate follows from this RR range, not from 1:2. |
| 12 | ONE open position at a time | **WRONG** | No max-concurrent-position limit exists anywhere in the code (§4, confirmed by grep). The only per-symbol restriction is one position per (symbol, side) via the internal/exchange-side duplicate guards (bot.py:1917-1953); many symbols and both sides across different symbols can be open simultaneously. CLAUDE.md documents 45+ concurrent positions as the routine live state. |
| 13 | Commission 0.04% | **UNVERIFIED / NOT APPLICABLE** | No commission/fee rate is read anywhere in the live sizing or execution path (`autobot/core/bot.py`, `autobot/brokers/bybit.py` — grep for `fee_pct`, `entry_slippage_pct`, and any `config['execution']`/`config.get('execution'` access returns zero hits). `config.yaml`'s `execution.fee_pct: 0.0006` (0.06%) exists in the file but is in a section CLAUDE.md documents as dead/never-read, confirmed here — the live bot does not model or cap commission at all; real Bybit taker fees apply automatically on the exchange side, outside the bot's control or awareness. |
| 14 | Starting balance $10,000 | **UNVERIFIED (no config value exists to check)** | There is no `starting_balance` key anywhere in `config.yaml` (grep confirms only unrelated string matches on symbol names like `1000000BABYDOGEUSDT`, `10000SATSUSDT`). `lifetime_stats['starting_balance']` (bot.py:165,391) is populated from whatever the real exchange balance is at first run (or reset via `/setbalance`/`/resetlifetime` per CLAUDE.md) — it is not a fixed configured figure, so "$10,000" cannot be confirmed or denied from source; it depends on live account state at deploy time, which this read-only source review cannot observe. |

**Summary: 13 of 14 rows are WRONG as stated against current source; row 14 is
unverifiable from static source (depends on live account state, not config) and row 13
is unverifiable in the sense that the bot has no commission model to check against — the
figure in `config.yaml` exists but is dead code.** The third party's description matches
a different strategy archetype entirely (support/resistance zone + engulfing-candle
entry, single-position, 15m/4H multi-timeframe) — none of which exists in this
codebase. The actual live strategy is 1H RSI-divergence + Break-of-Structure, 277
symbols / 728 configs, ATR-based stops/targets varying by symbol and divergence type,
no concurrent-position cap.
