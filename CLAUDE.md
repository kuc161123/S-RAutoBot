# CLAUDE.md — AutoTrading Bot

Working notes for Claude Code. Written 2026-07-25 from a full read of the live code
tree. Everything below was verified against source, not inferred from docs or commit
messages. Line refs are `file.py:line` at commit `5cd884f`.

---

## 1. What this is

A live Bybit USDT-perpetual futures bot. One strategy: **1H RSI divergence + Break of
Structure confirmation**, fixed ATR-based stop, fixed R:R take-profit, no partials.
Since 2026-08-02 the stop **trails** once a trade reaches +3R (§6.1). It runs 277
symbols / 728 configs concurrently and has no cap on open positions (45+ open at once
is normal).

Deployed remotely (Postgres via `DATABASE_URL`, Telegram control surface). No local
state files — everything lives in Postgres or is fetched from Bybit.

### Entry point

`main.py` → kills stale instances (PID file `/tmp/divergence_bot.pid`) → starts
`dashboard.py` Flask app on :8888 in a daemon thread → `Bot4H().run()`
(`autobot/core/bot.py:2238`).

### The live module tree (everything else is dead — see §8)

```
main.py
└── autobot/core/bot.py            (2473)  orchestrator, risk gates, regime, execution
    ├── autobot/brokers/bybit.py   (1536)  Bybit v5 REST, async aiohttp, no websocket
    ├── autobot/core/divergence_detector.py (465)  RSI div + BOS + EMA gate
    ├── autobot/config/symbol_rr_mapping.py (245)  loads config.yaml symbols block
    ├── autobot/core/storage.py    (313)   Postgres (2 tables) w/ JSON fallback
    ├── autobot/utils/telegram_handler.py (2109) dashboard + all slash commands
    └── shadow layer (observation only, never gates trades)
        ├── shadow_logger.py   (464)
        ├── shadow_scanner.py  (304) ── breakout_detector.py (64)
        └── shadow_analyst.py  (423)
```

---

## 2. Main loop

`bot.py:2323` — a `while True` with `asyncio.sleep(60)`. Each tick iterates all 277
enabled symbols. Per symbol (`process_symbol`, `bot.py:1132`):

1. `check_new_candle_close` — returns True at most once per wall-clock hour per symbol.
   Everything below only runs on that first tick of the hour.
2. Staleness guard: skip if >55 min past the hour (`bot.py:1152`).
3. `fetch_4h_data` — 1000×1H klines. **Includes the still-forming candle as the last
   row** (verified against the live API). Name is a legacy misnomer; timeframe is `'60'`.
4. `prepare_dataframe` — adds RSI(14), ATR(14), EMA(200 span, ewm), CHOP(14). No dropna.
5. **Execute queued entries from the previous hour** at `df.iloc[-1]['open']`
   (`bot.py:1185-1193`).
6. Radar cache (dashboard `/radar` only, no trading effect).
7. `detect_divergences` → new signals → `pending_signals[symbol]`.
8. `check_pending_bos` → BOS + EMA re-check → queue into `confirmed_entries`.
9. `monitor_active_trades` → detect exits.

Then per cycle: position sync every 5 min, shadow resolve/scan/analyst, Telegram scan
summary, auto-dashboard every ~55 min.

**Consequence of the 60s/hourly design:** a full sweep of 277 symbols with several API
round-trips each takes many minutes. Symbols late in the list are processed well after
the hour opens.

---

## 3. Signal generation (`divergence_detector.py`)

**Pivots** (`find_pivots:146`): fractal, `left=3 right=3`. A pivot at `i` is only
knowable at bar `i+3`. The scan loop runs `range(scan_start, n - PIVOT_RIGHT)`, so
detection never touches the last 3 rows — **no lookahead in detection.**

**Four divergence types** (`detect_divergences:179`), each maps to a fixed side:

| Code | Pattern | Side |
|---|---|---|
| `REG_BULL` | price LL, RSI HL | long |
| `REG_BEAR` | price HH, RSI LH | short |
| `HID_BULL` | price HL, RSI LL | long |
| `HID_BEAR` | price LH, RSI HH | short |

Constraints: pivots ≥3 bars apart (`MIN_PIVOT_DISTANCE`), triggering pivot ≤10 bars old,
dedup on `(curr_pivot_idx, prev_pivot_idx, direction)`.

**`swing_level`** = extreme high (bull) / low (bear) between the triggering pivot and the
scan bar. This is the BOS trigger level.

**EMA-200 gate is applied twice**: at ingestion (`bot.py:1341`, using the flag recorded
at detection) and again at BOS confirmation (`bot.py:1454`). Deliberate — lets a
divergence queue up and become valid when price later crosses the EMA.

**Pending signals** live up to `max_wait_candles = 12` (12 hours), then expire.
Dedup by `symbol + side + div_code + pivot_timestamp` in the `seen_signals` table.
Signals older than 24h are dropped outright (restart protection).

**Dashboard "Signals 88D/31BOS · pending 44"**: 88 divergences detected today, 31 BOS
confirmations today (not necessarily from the same 88 — a BOS today can come from
yesterday's divergence), 44 signals currently waiting for BOS.

---

## 4. Trade construction (`execute_trade`, `bot.py:1508`)

```
rr        = per-symbol, per-divergence-type from config.yaml   (3.0 / 5.0 / 8.0 / 10.0)
atr_mult  = per-symbol, per-divergence-type from config.yaml   (1.0 / 1.5 / 2.0)
atr       = df.iloc[-2]['atr']          # last CLOSED candle — correct
entry     = df.iloc[-1]['open']         # forming candle's open
sl_dist   = atr * atr_mult
sl_price  = entry ∓ sl_dist
tp_price  = entry ± sl_dist * rr
qty       = risk_amount / sl_dist       # rounded DOWN to qtyStep
```

Order: single atomic **market bracket** — `place_market()` with `takeProfit`/`stopLoss`
attached at creation, so a position is never unprotected. Position mode is **one-way**
(`positionIdx: 0` everywhere). Leverage is set to the **exchange maximum** per symbol at
startup; there is no config ceiling. Leverage only affects margin headroom — sizing is
stop-distance based.

**Because RR is 3–10 with the mass at 8–10, this strategy is structurally low-win-rate.**
Weighted breakeven win rate across the 728 live configs is **~13.9%** (matches the 13.5%
independently computed in `verification_results/OVERFIT_VERDICT.md`). A 20% win rate is
a good month; a 9% win rate is a losing one. Judge the bot against ~14%, not 50%.

---

## 5. Risk stack (order of application)

Everything in `config.yaml`'s `risk:` block. Current values:

```
risk_per_trade        0.003   (0.3%)     lowered from 1.2% on 2026-07-21
taper_schedule        1500→0.3%  3000→0.28%  5000→0.25%  8000→0.22%
                      12000→0.2% 20000→0.17% 40000→0.14%
net_directional_cap   0.10    (10% of equity)
gross_open_risk_cap   0.30    (30% of equity)
btc_short_gate        true    short_gate_ret30 0.10
long_bull_boost       1.3
overlay_ramp_min_balance  0   → all overlays ALWAYS ON
```

### The sizing chain

```python
base   = taper_schedule lookup on WALLET balance      # 0.003 at $1.3k
mult   = regime multiplier (1.0 / 0.5 / 0.25 / 0.1)   # from last 20 trades
final  = base * mult                                   # 0.00075
risk_$ = EQUITY balance * final                        # $1,413 * 0.00075 = $1.06
if long and BTC>EMA200: risk_$ *= 1.3
```

Note the deliberate **wallet-taper / equity-sizing mismatch** (`bot.py:1675-1681`) —
previously tested; aligning them hurt returns. Leave it.

### Regime — trade-quality tiers, NOT market indicators

`get_regime_status()` (`bot.py:554`). Inputs are **only** win rate and average R over the
**last 20 closed trades**. BTC ADX and BTC CHOP appear on the dashboard but feed nothing.
Drawdown, daily R, and loss streak are computed into `diagnostics` but explicitly
excluded from the multiplier.

```python
if wr >= 0.18 and avg_r >= 0.15:  1.0   favorable
elif wr >= 0.18 or avg_r >= 0.1:  0.5   cautious
elif wr >= 0.10 or avg_r >= -0.5: 0.25  adverse
else:                             0.1   critical
n_trades < 10                  →  0.1   critical ("SAFE START")
```

Not configurable — hardcoded. Manual override via `/setregime`, auto-clears after 10
trades. A `'halted'` label exists in display dicts but is unreachable.

### Entry gates, in execution order (`execute_trade`)

| # | Gate | Where | Notes |
|---|---|---|---|
| 1 | shadow log | 1523 | fires *before* everything, so observation continues while halted |
| 2 | `/stop` flag | 1530 | |
| 3 | regime halt (mult==0) | 1535 | unreachable — min mult is 0.1 |
| 4 | CHOP filter | 1545 | see inversion note below |
| 5 | BTC impulse-bull short-gate | 1575 | shorts only; BTC 30d return > +10% |
| 6 | already-in-trade (internal) | 1592 | |
| 7 | already-on-exchange / opposite-side guard | 1598 | prevents one-way-mode force-close |
| 8 | net-directional cap | 1698 | \|long risk − short risk\| ≤ 10% equity |
| 9 | gross open-risk cap | 1716 | Σ open risk ≤ 30% equity |
| 10 | margin sufficiency | 1760 | |

**CHOP thresholds are inverted on purpose.**
`{'favorable': 52, 'cautious': 45, 'adverse': 52, 'critical': 55}` (`bot.py:1546`) —
the filter gets *looser* as the regime gets worse, which contradicts every optimisation
run in `chop_optimize_summary.txt` (those want adverse ~44, critical ~38). This was
tested and deliberately kept: `GROWTH_VALIDATION_REPORT.md` §5 — *"Fixing the inverted
CHOP thresholds → not robust (helps down-markets, hurts bull). Leave CHOP as-is."*
Do not "fix" this without re-running that study.

Also from that report: the CHOP filter is **load-bearing** — removing it takes PF to 0.53
and wipes the account.

### Risk keys that do nothing

`max_daily_loss: 0.1` and `max_position_size_pct: 0.15` are **never read anywhere in the
codebase**. There is no daily-loss circuit breaker and no per-position size cap. There is
also no max-concurrent-positions limit anywhere. Don't be reassured by those keys.

---

## 6. Exits and accounting

Polling only — no websocket, no local price monitoring. `monitor_active_trades`
(`bot.py:1890`) asks Bybit whether the position still exists; `size == 0` triggers
`handle_trade_exit` (`bot.py:1923`), which fetches `closed-pnl` (3 retries) and matches
by side + entry price within 1%.

`r_value = pnl_usd / trade.risk_usd_at_entry` — correct per-trade R.

Defensive behaviour worth preserving: if `get_positions()` returns empty (API failure
returns `[]`), sync **skips** stale detection rather than mass-closing tracked trades
(`bot.py:959`, `2001`). If no closed-PnL record is found, it re-verifies against the
exchange before assuming a close.

Restart adoption (`sync_with_exchange`, `bot.py:992`) rebuilds `ActiveTrade` objects from
exchange positions. `entry_time` becomes `now()` and `risk_usd_at_entry` is unknown for
adopted trades — those are skipped by the net-directional cap and fall back to
price-based R.

### 6.1 Trailing stop (added 2026-08-02) — variant `s3_a1`

**The rule.** Nothing happens until a bar's excursion reaches **+3R**. From then on the
stop ratchets to `high − 1×ATR` (long) / `low + 1×ATR` (short), computed on **closed**
candles only, never widening. **There is no breakeven move and one must never be added.**

`_update_trailing_stops` → `_trail_one` (`bot.py`), called once per symbol per hourly
close from `process_symbol`, using the dataframe already fetched — no extra API calls.
Amendments go through `bybit.amend_stop_loss` (a new method; `set_sl_only` could not be
used because it rejects any price > 100000, a false positive on BTCUSDT, and it raises).

**Why it is deployed.** `report_trail_live_realistic.py` replayed 22 exit rules against
identical signals over the live config (277 symbols, 38,753 signals, 2023-06 → 2026-07):

| | last 15mo | maxDD 15mo | full 3yr DD | top 1% of trades = % of profit |
|---|---|---|---|---|
| base (fixed TP) | $101,452 | 50.4% | 39.7% | **112%** |
| `s3_a1` | **$119,030** | **28.3%** | **29.3%** | **24%** |

`s3_a1` is ahead at 1.0×, 1.5× **and** 2.0× costs. Returns are period-dependent (base
wins the 10-month and full-history windows); the **drawdown** and **profit-concentration**
improvements are what held in every window. It went live as risk control, not as a
profit upgrade — and it does not make the bot profitable OOS.

**Things that were tested and are worse — do not "improve" these:**
- Any breakeven move: `s2_a1` $107,230 → `be1_s2_a1` $61,778 (−42%). Every `be*` variant
  lost to its non-`be` twin in every period.
- Tight trails: `s1_a1` $80,799.

**Four subtleties that took a bug each to find** (`verify_trailing_parity.py` drives the
live engine, the shadow resolver and the backtest over the same candles and asserts
agreement trade-by-trade — currently 100% over 3,200+ trades):

1. **Arming uses each bar's own excursion, not a running peak.** The validated backtest
   re-tests `mfe >= 3R` fresh every bar, so when price falls back the stop *holds*
   instead of continuing to ratchet. A running-peak version is a different rule.
2. **`risk_dist` is carried explicitly.** Reconstructing the stop distance as
   `|entry − original_stop_loss|` loses the last ulp; on a low-priced symbol that turned
   an excursion of exactly 3.0R into 2.999999999999996 and skipped a ratchet.
3. **`entry_bar_ts` anchors the replay, not `entry_time`.** `df.index` is UTC
   (epoch-derived) while `entry_time` is `datetime.now()` (local) — flooring the latter
   picks the wrong bar on a non-UTC host.
4. **The clamp must not un-ratchet.** A trail level can land through the last close on a
   reversal bar; the exchange rejects that, so it is pinned just inside — but bounded by
   `max(..., current_stop)`, or a clamp limit below the current stop would *widen* risk.

**State and restarts.** `original_stop_loss`, `risk_dist`, `entry_bar_ts` and trail
progress are persisted in `lifetime_stats['open_trade_trailing']` (same pattern as
`open_trade_regimes`). A restart-adopted position **with no record is never trailed** —
`original_stop_loss` stays 0.0 and it keeps the exact bracket it was opened under. On the
first deploy that is every open position, and the bot says so in a Telegram notice.

Also fixed here: `handle_trade_exit`'s price-based R fallback divided by `trade.stop_loss`,
which the trail mutates — it now uses `original_stop_loss`.

**Controls.** `/trail` (status), `/trail on|off` (runtime toggle, reverts to config.yaml
on restart; turning it off never widens stops already moved), `/trailstats` (verdict).
Config: `risk.trailing_stop`.

### 6.2 Trail shadow — is trailing actually worth it?

`autobot/core/trail_shadow.py`, table `trail_shadow`. Observation only, same contract as
the rest of the shadow layer. Every trade is replayed from klines under **both** exit
rules against the same bars, so `r_trail − r_fixed` is attributable to the exit and
nothing else. It also measures **slippage** — realized R vs the modelled R of whichever
arm was actually live, which is the number that decides whether the 1× or 3× cost column
is real. `/trailstats` prints the comparison and a recommendation; the bar is asymmetric
(stay on unless significantly worse on R *and* not better on drawdown) because that is
the trade the feature was deployed to make. Needs 60 resolved trades before it calls it.

Same SL-wins-ties bias as `ShadowLogger`, which penalises the *trailed* arm more (its stop
sits closer to price) — so a positive delta there reads conservative.

---

## 7. Telegram dashboard — read the R numbers carefully

`build_dashboard_message()` (`telegram_handler.py:220`). Commands: `/dashboard /pnl
/edge /regime /blocks /positions /radar /learn /trail /trailstats /stats
/performance /risk /stop /start /setregime /setbalance /resetstats /resetlifetime /debug /help`.

**R is now computed per-trade (fixed 2026-07-25).** It previously divided aggregate
dollars by a single scalar `base_risk_amount = equity × tapered base risk`, which ignored
the regime multiplier and therefore ran ~4× too large in an adverse regime — and was wrong
in principle regardless, because the trades in that sum were opened at different risk
levels. Now:

- **Realized R** (TODAY / 7-DAY) sums `lifetime_stats['daily_r']`, which accumulates
  `pnl_usd / risk_usd_at_entry` per closed trade. TODAY's realized R and the WHY block's
  "Daily R" now agree — they previously showed different numbers for the same day.
- **Open R** uses each position's own `risk_usd_at_entry` from the internal tracker.
  Restart-adopted positions have no recorded entry risk; those fall back to
  `base_risk_amount` and the total is suffixed `~` to mark it partial.

Still true and unfixable retroactively: base risk moved 1.2% → 0.3% on 2026-07-21, so
**lifetime R totals spanning that date sum non-comparable units.** "DD from peak: 103.0R"
against a real trading drawdown of 10.9% / $203 is that artefact, not a contradiction —
it is now labelled `(cum-R, mixed basis)` on the dashboard. Trust the dollar drawdown.

One residual seam: `realized_pnl` comes from Bybit's closed-PnL records for today while
`realized_r` comes from the bot's own `daily_r`. If the bot misses a close, the dollar and
R figures on the TODAY line can disagree in magnitude or even sign.

**The header badge understates the regime.** `adverse` (0.25×) renders as
**"🟠 CAUTIOUS"**, while the genuinely better `cautious` tier (0.5×) renders as
"🟡 SELECTIVE" (`telegram_handler.py:768`). Seeing "CAUTIOUS" in the header means the bot
is one tier *worse* than the name suggests. The only place the true tier appears is the
`Regime:` line in the WHY block.

Other quirks:

- **7-DAY and TODAY come from Bybit** `get_all_closed_pnl(limit=200)` (raw execution
  records, win = `pnl > 0`); ALL-TIME comes from local `lifetime_stats` (bot-detected
  position closes, win = `r_value > 0`). Different sources, different windows, different
  win definitions, plus a 200-record cap. They will not reconcile trade-for-trade.
- **`Signals D/BOS` can be a day stale.** The daily counters reset lazily *inside*
  `_track_divergence_detected`/`_track_bos_confirmed` (`bot.py:821`), but the dashboard
  reads them directly without checking `last_reset`. Just after local midnight it shows
  yesterday's totals until the first new signal fires.
- **`/setbalance` re-anchors the dollar counters but not the R counters.** It resets
  `starting_balance` and the trading-equity peak; it leaves `total_r`, `peak_equity_r`,
  `weighted_total_r`, `regime_stats` and `start_date` alone. Only `/resetlifetime` clears
  everything. Run `/setbalance` mid-stream and the dollar and R sections silently
  describe different windows.
- **"taper 0.30%" is not currently a taper step.** Wallet is $1,327, below the lowest
  rung ($1,500), so the taper lookup never fires and the displayed figure is just
  `risk_per_trade`. The label says "taper" regardless.
- **`ALL-TIME (4d)`** = days since `lifetime_stats.start_date`, which `/resetlifetime`
  and `/setbalance` re-anchor. It is not the bot's real age.
- **EDGE CHECK is a small-sample artefact.** "Favorable" is *defined* as "the last 20
  trades won", so wins auto-cluster into it. `GROWTH_VALIDATION_REPORT.md` §0: over
  23,000 trades the separation collapses to 1.48 vs 1.11. Do not build decisions on it.
- Drawdown is shown twice: "trading" DD (starting balance + cumulative trading P&L —
  immune to deposits/withdrawals) and raw "equity" DD.
- BTC ADX and BTC CHOP are fetched live per `/dashboard` call (60×1H BTCUSDT klines,
  `telegram_handler.py:600`). They are display-only — nothing in the trading path reads
  them. The CHOP filter uses each *symbol's own* CHOP, not BTC's.
- `Protection: 🟢 ON (...)` is a static echo of `config.yaml`, not a live state check.
- When copying the dashboard out of Telegram, the ALL-TIME header line and the EDGE CHECK
  rows tend to merge into one line (they use different formats: ALL-TIME `PF {:.2f}`,
  EDGE CHECK `PF {:.1f} | ${net} | {t}t ({blocked} blocked) | {n} open`). A pasted line
  reading `147t · 8.8% WR WR | PF 0.0 | $-12.15 | 4t (27 blocked)` is that artefact, not
  a code bug.

---

## 8. Shadow layer — advisory only, verified

Three active modules write to Postgres (`shadow_signals`, `shadow_candidates`,
`shadow_alerts`) and never touch trading state. I traced every caller: the only outputs
that leave the subsystem are Telegram strings. `log_signal` is deliberately placed
*before* the `/stop` gate so observation continues while halted.

- **`ShadowLogger`** — records every BOS-confirmed signal (executed *and* blocked) with
  intended entry/SL/TP, then grades it by walking forward through klines. Loss is exactly
  `-1.0`; ambiguous bars are scored **SL-wins-ties** (deliberately pessimistic).
- **`ShadowScanner`** — a second strategy family (`DONCH`, Donchian breakout, 45 configs
  in `shadow_families.yaml`) plus daily out-of-universe scouting (≥$5M turnover, ≤100
  symbols, tagged `in_universe=False`).
- **`ShadowAnalyst`** — read-only stats and advisory Telegram alerts.

### Why `/learn week` shows −315R

`weekly_r()` (`shadow_analyst.py:211`) sums cost-adjusted R over **every evaluated
signal**, executed or blocked, across the whole universe. `ShadowLogger.edge_rows()`
supports `executed_only=True` but `ShadowAnalyst._rows()` never passes it. Combined with
the SL-wins-ties bias (which hits the RR-8/10 configs hardest), a few-hundred-negative-R
week is the expected shape of this metric, not evidence the live book lost 300R.

The module header records that an automatic weekly-R kill-switch was tested and
**rejected** — it clipped the jackpot months that pay for the system. `suggested_mult` is
labelled "(paper)" and is never applied.

**Gap worth closing:** the flagship advisory number conflates "all signals we looked at"
with "how the bot did". An `executed=True` variant would make it readable.

---

## 9. Dead code — do not edit these thinking they are live

Roughly 7,000 of the 16,000 lines under `autobot/` are unreachable:

| File | Lines | Status |
|---|---|---|
| `core/bot_5m_old.py` | 4054 | zero importers anywhere |
| `core/unified_learner.py` | 1957 | only imported by `bot_5m_old.py` |
| `core/smart_learner.py` | 543 | imported nowhere |
| `core/combo_learner.py` | 500 | imported nowhere |
| `core/divergence_detector_5m_old.py` | 259 | imported nowhere; different algorithm |
| `core/shadow_auditor.py` | 98 | only `bot_5m_old.py`; also has a broken import (`detect_divergence` singular does not exist) |

The three learner modules implement symbol blacklisting, combo promotion, and
**automatic `git commit && git push` on promote** (`unified_learner.py:1082`). If anyone
ever rewires them into the live bot, that behaviour needs removing first.

Dead in `bybit.py` too (only `bot_5m_old.py` calls them): `set_tpsl`, `set_sl_only`,
`set_trailing_sl`, `place_limit`, `place_reduce_only_limit`. `set_tpsl` is missing an
`await` on its return (`bybit.py:1113`) — harmless while unused.

Dead `config.yaml` sections (never read by `load_config`): `execution`, `indicators`,
`legacy`, `monitoring`, `notifications`. Indicator periods are hardcoded module constants
in `divergence_detector.py`. Also `strategy.entry_params` and `strategy.signal_params`
**do not exist in the file**, so `max_wait_candles=12` and `lookback_bars=50` silently
run on code defaults.

Repo root is 369 loose `.py` scripts, 146 CSVs, 33 YAMLs, and ~3 GB of kline cache. The
only ones that matter for understanding decisions are listed in §10.

---

## 10. What the research actually concluded

Read these before proposing any strategy change — most obvious ideas have been tested
and rejected already.

**`verification_results/OVERFIT_VERDICT.md`** (2026-05-14) — 8-test overfitting suite on
1,357 live trades. Verdict 🟡 CAUTION. Params robust (−17.6% worst ±10%), universe not
cherry-picked (0.98 OOS retention), edge not regime-concentrated (max 53%), p=5e-23.
**The one RED: only 13.7% of past live trades would be re-taken by today's bot**
(config drift 52%, CHOP-blocked 58%). The historical +1059R was earned by a *different*
version. Treat the current bot as an unproven strategy.

**`GROWTH_VALIDATION_REPORT.md`** (2026-06-19) — genuine OOS is **PF ~1.16 with 55–80%
drawdown**, matching live PF 1.14. In-sample dollar figures are fiction. The bot is a
net-short machine that loses money in bull markets (PF 0.93, 88% DD in the 24-10→25-02
bull). Net-directional cap 10–15% is the closest thing to a free lunch. Tested and
rejected: halting in critical/adverse (self-referential lock-in, 0 trades), favorable-only
trading, fixing CHOP inversion, raising risk.

**Commit `ea730b4`** (2026-07-21) — base risk 1.2% → 0.3%. Cut cold-period drawdown
63% → 26% on OOS data. The commit message is explicit: *"Reduces the SIZE of losses while
the edge is cold; does not create profit."*

**Commit `1f2ce91`** (2026-07-15) — `/stop` was set but never checked; opposite-side
guard; gross-risk cap; short-gate v2 (daily 30d return, ~1.5 flips/mo, replacing the 1H
EMA200 that flipped ~26×/mo and blocked profitable ordinary-bull shorts).

---

## 11. Open issues

### ~~BOS is judged on the forming candle~~ — FIXED 2026-07-25

**Applied.** `check_pending_bos` now uses `current_idx = len(df) - 2` (last closed candle),
and the confirmed entry is drained in the **same cycle** by `_drain_confirmed_entries`
(`bot.py`), entering at `df.iloc[-1]['open']` — the candle immediately after the BOS
candle. `execute_trade` needed no change: its `entry = df.iloc[-1]['open']` /
`atr = df.iloc[-2]['atr']` were already correct for this ordering (they were only correct
*by accident* before, because the queue delayed them a cycle).

Verified with a synthetic replay: a break closing at 06:00 now enters at the 07:00 open
with ATR from the 06:00 candle; previously it entered at 08:00.

The original diagnosis is kept below for context.

### BOS forming-candle bug (historical description)

`check_pending_bos` uses `current_idx = len(df) - 1` (`bot.py:1439`), and `df.iloc[-1]` is
the **still-forming** candle — verified directly against the Bybit kline endpoint. So
`check_bos` and `is_trend_aligned` (`divergence_detector.py:392`, `415`) both test a
provisional close that is still moving.

The code already knows this. `execute_trade`'s own comment (`bot.py:1642`) says *"ATR
comes from the PREVIOUS candle (the BOS candle), which is `df.iloc[-2]` … since df now
includes the new candle"*, and `_is_btc_impulse_bull` (`bot.py:729`) explicitly skips row
0 for the same reason. Only the BOS/EMA re-check was missed.

Effect: the crossing that truly happened at candle H's close is not acted on until the
H+1 poll sees it in H+1's forming close, and entry then lands at H+2's open — roughly one
candle late versus backtest semantics. Recorded impact: ~9% of trades missed,
+0.037R/trade fill cost. Detection is unaffected (the scan loop stops at `n-3`).

Fix is two-part and must be done together:
1. `current_idx = len(df) - 2` in `check_pending_bos`;
2. execute the confirmed entry **in the same cycle** at `df.iloc[-1]['open']`, instead of
   queueing it for the next hour.

**Validated 2026-07-25 — not yet applied.** `backtest_bos_timing_ab.py` replays both arms
over `cache_3yr_1h` (277 symbols, 728 live configs, 2023-03 → 2026-05) sharing signal
detection, fees, dynamic slippage and exit resolution with the validated engine, so only
timing differs. Artefacts: `bos_ab_summary_chop.txt`, `_nochop.txt`, `_decompose.txt`.

| | trades | WR | avg R | cum R | PF | maxDD |
|---|---|---|---|---|---|---|
| live (current) | 39,163 | 17.67% | +0.2645 | +10,358 | 1.297 | 890R |
| fixed (proposed) | 36,891 | 20.35% | +0.4720 | +17,412 | 1.548 | 599R |

Fixed wins **13/14 quarters**, improves **69% of the 728 configs** (sign test p = 5.8e-25),
and takes ~6% *fewer* trades for ~68% more R. Drawdown falls 33%. On the 33,690 signals
both arms take, 2,279 live losses become wins against 892 the other way.

Two checks worth knowing:
- **No-CHOP sensitivity** (`bos_ab_summary_nochop.txt`): with the gate off both arms take
  near-identical signals (70,134 shared of ~70,160) — a clean paired test. The fix still
  gains **+0.088 R/trade**, PF 1.154 → 1.254, maxDD 258R → 150R. So the headline +0.21
  includes a CHOP-gate-position benefit; the pure timing effect is +0.09. Both favour the
  fix.
- **Decomposition** (`--decompose`): of the +0.2075 R/trade gain, **the one-candle entry
  delay accounts for effectively all of it** (+0.2077); the ATR-source shift contributes
  +0.0435 with −0.0437 interaction. The fix does not hinge on the ATR detail.

Caveat the harness cannot cover: the intra-hour drift component. In 1H data
`open[i+1] == close[i]` (99.996% exact), so the trigger price is identical in both arms and
the drift-induced miss rate needs sub-hourly klines to measure. The delay measured here is
the structural part; drift is second-order on top.

Note when reading these artefacts: `stats()` computes `max_dd_r` on the R series in the
order given, so it **must** be fed exit-time-sorted rows (`_r_chrono`). Earlier runs that
pooled rows in symbol order reported meaningless drawdowns.

### SL/TP are anchored to a price the order does not fill at

`sl_price`/`tp_price`/`qty` are all computed from `entry_price = df.iloc[-1]['open']`, but
the order is a **market** order placed minutes later. `actual_entry` is recorded for
reporting, but the bracket levels and `risk_usd_at_entry` are never recomputed from it. On
a symbol processed late in the sweep, real risk-per-trade drifts from planned.

### Other

- No rate limiter in `bybit.py` beyond a 0.1s sleep during startup leverage config.
  277 symbols × several calls per hourly sweep runs unthrottled.
- `_get_precisions` falls back to a hardcoded `("0.0001", "0.001")` on fetch failure and
  **caches it** for the session (`bybit.py:1148`) — wrong tick size persists.
- `get_max_leverage` is defined twice, identically (`bybit.py:72` and `:433`).
- `storage.py` JSON fallback writes without atomic rename or locking; a crash mid-write
  truncates the file. Postgres failures fall back to JSON silently, so state can diverge
  across restarts if connectivity flaps.
- 11 symbols disabled with no comment: `CAMPUSDT DEGENUSDT GNOUSDT GODSUSDT OLUSDT
  ORBSUSDT PYRUSDT RDNTUSDT SCUSDT TONUSDT TRUUSDT` (delisted — see commit `88a08ef`).
- `symbols_all.yaml` is orphaned (no importer) and contains dated futures contracts.
- `strategy.description` says 288 symbols / 756 configs; top-level `strategy_description`
  says 235. The latter is stale. Live enabled counts are **277 symbols / 728 configs**.

---

## 12. Reading the current live state (as of the 2026-07-25 dashboard)

- Regime `adverse` (0.25×) because the last 20 trades ran 10% WR / −0.91R.
- 147 trades since the last lifetime reset, 8.8% WR. Against a ~13.9% breakeven that is
  **P(≤13 wins | true WR = breakeven) ≈ 0.06** — a bad run, but *not* statistically
  sufficient to declare the edge dead. Do not re-tune on this sample.
- 40 shorts / 5 longs open. That is the expected shape (bear configs outnumber bull
  ~1.4:1) and exactly the net-short concentration the 10% net-directional cap exists to
  bound.
- `OVERFIT_VERDICT.md`'s standing recommendation is unmet: **freeze CHOP thresholds and
  per-symbol RR/ATR for 90 days**, collect 250+ fresh trades under today's exact rules,
  then re-run `verify_overfit.py`. Every re-tune resets that clock.

---

## 13. Conventions

- Don't touch `config.yaml`'s per-symbol `configs` blocks by hand — they are walk-forward
  output (train < 2025-11-01, test through 2026-05-25).
- Risk/protection changes belong in `config.yaml`'s `risk:` block, with an inline comment
  citing the validation that justifies them. Every existing key follows this; keep it.
- The shadow layer's design contract ("must NEVER affect trading") is load-bearing —
  every shadow call in `bot.py` is wrapped in a bare `try/except`. Keep it that way.
- Backtests live at repo root as `backtest_*.py` / `simulate_*.py` / `validate_*.py` /
  `verify_*.py`. `backtest_production_correct.py` is the one that mirrors the live
  USD/regime/taper engine.
