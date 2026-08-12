# Margin / Liquidation Risk at Raised Risk-Per-Trade

Read-only analysis. Data: `halt_universe_live.parquet` (38,753 signals, 277-symbol
live universe, 2023-06-01 → 2026-07-25). Script: `liq_analysis.py` in this dir.
All CSVs referenced below are in this same directory.

## Headline

**Liquidation-style stress becomes material starting around f ≈ 1%, and a
realistic worst-case correlated-loss day crosses full account ruin (100% equity
loss from stop-outs alone, before any liquidation slippage) at roughly f ≈ 1.9%
— i.e. raising risk-per-trade from today's 0.3% by roughly 3–6x is enough to move
the binding constraint from "stop-loss did its job" to "the account doesn't
survive the day."** And a second, independent failure mode gets there even
sooner: because the bot runs **no cap on concurrent positions** and **exchange-max
leverage** on every symbol, the account's own **aggregate initial-margin
requirement already exceeds 100% of equity in the worst historical hours at
today's 0.3% risk** (p99 ≈ 92% of equity, max ≈ 134%, with the gross-risk cap
applied) — before touching maintenance margin/liquidation math at all. That
means margin exhaustion (orders silently blocked, not liquidated) is already an
occasional live constraint today, and gets worse roughly linearly with f.

Both of these are real failure modes the R-multiple backtests structurally cannot
see, because they size and grade every trade purely by stop distance and never
touch notional, leverage, or the exchange's shared margin pool.

---

## 0. Margin mode — cross, not isolated (decisive, verified by absence)

- `positionIdx: 0` is hardcoded everywhere orders are placed — **one-way mode**,
  confirmed (`autobot/brokers/bybit.py:775,834,1094,1110,1248,1258,1332,1403,1464,1553`;
  `autobot/core/bot.py:1937` comment corroborates).
- Balance is fetched from the **`UNIFIED`** account type first
  (`autobot/brokers/bybit.py:349,395` — `account_types = ["UNIFIED", "CONTRACT", "SPOT"]`),
  i.e. this is a Bybit **Unified Trading Account**.
- **There is no call anywhere in the codebase to `/v5/position/switch-isolated`**,
  and no order payload anywhere sets a `tradeMode` field (`grep -rn -i
  "isolated|marginMode|margin_mode|switch-isolated" autobot/` returns nothing).
  Every `place_market`/`place_limit`/order-amend call only sets `category`,
  `symbol`, `side`, `qty`, `positionIdx`, and TP/SL fields — never a margin-mode
  field.

**Conclusion: the account runs on whatever the exchange's default margin mode is
for a UTA (cross), because the bot never switches it.** In cross margin, all
open positions draw from and are protected by the **same shared equity pool** —
a large adverse move on one symbol reduces the maintenance-margin buffer
available to every other open position simultaneously. This is the mechanism
that makes the correlated-loss scenario in §4 a genuine liquidation risk and not
just an accounting curiosity: it is not 45 independent bets, it is 45 bets
against one shared collateral balance.

---

## 1. Stop-distance / notional-multiplier distribution

`stop_frac = |entry − sl| / entry`. Since `qty = risk_usd / sl_distance`,
`notional = risk_usd × (1/stop_frac)` — `1/stop_frac` is the "leverage" the
position sizing implies **before any exchange leverage is even applied** (it's
purely how many dollars of notional one dollar of risk buys).

| percentile | stop_frac | notional multiplier (1/stop_frac) |
|---|---|---|
| 1 | 0.382% | 12.9x |
| 5 | 0.698% | 21.1x |
| 25 | 1.239% | 37.8x |
| 50 | 1.789% | 55.9x |
| 75 | 2.642% | 80.7x |
| 95 | 4.744% | 143.3x |
| 99 | 7.745% | 261.8x |

mean stop_frac = 2.16%, mean notional multiplier = **68.2x**.

So a single dollar of risk typically buys ~56x notional, and in the tail (tight
stops on low-volatility symbols/RR configs) up to **260x+ notional per dollar of
risk**. This is the multiplier that converts "risk %" into "how much of the
book is actually levered exposure" — full table in `1_stop_frac_distribution.csv`.

---

## 2. Concurrency and aggregate notional/equity

Full hourly event-driven timeline (every entry/exit from the parquet), computed
**both uncapped (every signal taken) and with `gross_open_risk_cap = 0.30`
applied greedily in chronological order** (config.yaml:75 — this cap is live
today). Equity held constant (this isolates the sizing/margin mechanics; §4
relaxes that for the loss-cluster question). Full table:
`2_concurrency_notional_summary.csv`.

| f | scenario | conc median/p90/p99/max | notional/equity median/p90/p99/max |
|---|---|---|---|
| 0.3% | uncapped | 49 / 99 / 146 / 201 | 6.9x / 15.3x / 25.7x / 35.0x |
| 0.3% | **capped** | 48 / 87 / 99 / 99 | 6.7x / 13.4x / 19.7x / 28.2x |
| 1.0% | uncapped | 49 / 99 / 146 / 201 | 23.1x / 51.1x / 85.6x / 116.6x |
| 1.0% | **capped** | 27 / 29 / 29 / 29 | 11.7x / 17.7x / 25.7x / 44.6x |
| 2.0% | uncapped | 49 / 99 / 146 / 201 | 46.1x / 102.1x / 171.1x / 233.1x |
| 2.0% | **capped** | 15 / 15 / 15 / 15 | 12.2x / 20.4x / 28.2x / 44.6x |
| 3.0% | uncapped | 49 / 99 / 146 / 201 | 69.2x / 153.2x / 256.7x / 349.7x |
| 3.0% | **capped** | 9 / 9 / 9 / 9 | 11.0x / 18.5x / 29.1x / 50.7x |

At today's live f=0.3%, the capped concurrency figures (p90=87, p99≈99, capping
out near 99) match CLAUDE.md's observed "45+ positions open at once is normal"
reasonably well as an upper envelope — the cap is rarely binding at 0.3%.

Two things worth noting:
- **`gross_open_risk_cap` is risk-dollar-denominated, not notional-denominated,
  so it does throttle aggregate notional indirectly** — raising f from 0.3% to
  3% only moves capped median notional/equity from ~6.7x to ~11.0x (not 10x, as
  it would uncapped), because the cap forces concurrency down roughly in
  proportion to f (99 → 29 → 15 → 9 concurrent positions as f rises). **This is
  the single most important thing keeping raised risk from immediately
  detonating aggregate notional exposure.**
- But it does **not** eliminate the tail: capped notional/equity still reaches
  **44–51x equity at the max** across all four f levels — a handful of large,
  wide-stop, high-notional positions can still stack up even under the cap.

## 2b. Initial margin / equity — margin can already bind *today*

Using exchange-max leverage per symbol (the one leverage table actually present
in this repo, copied verbatim from `backtest_production_correct.py:81-89`:
BTC/ETH=100x, `1000`-prefixed=25x, a defined set of ~30 top-cap alts=50x, all
other alts=20x — this is a backtest-side approximation of "exchange maximum,"
not a live-fetched Bybit table, and is flagged as such in §5's caveats).
`margin_used/equity` aggregated the same way as notional (`2b_initial_margin_over_equity.csv`):

| f | scenario | margin/equity median | p90 | p99 | max |
|---|---|---|---|---|---|
| 0.3% | uncapped | 0.31x | 0.71x | 1.19x | 1.63x |
| 0.3% | **capped** | 0.30x | 0.62x | **0.92x** | **1.34x** |
| 1.0% | **capped** | 0.53x | 0.80x | 1.13x | 2.12x |
| 2.0% | **capped** | 0.54x | 0.91x | 1.26x | 2.23x |
| 3.0% | **capped** | 0.49x | 0.84x | 1.31x | 1.88x |

**At today's live risk (0.3%), the aggregate initial margin required to hold all
open positions already reaches ~92% of equity at the 99th percentile hour, and
~134% at the historical max — with the gross-risk cap already applied.** Above
100%, new entries would be margin-blocked (this is exactly the `margin_blocked`
counter that exists in `backtest_production_correct.py:344,448,676` — the live
bot has no such counter/gate, it just lets Bybit's order-create call fail).
Because leverage is fixed at exchange-max and there's no concurrent-position cap,
this ratio doesn't move much with f under the cap (~0.5x median regardless of f)
— **the position-count throttling from `gross_open_risk_cap` is, again, what's
preventing this from scaling with f.** If that cap were ever loosened or
mis-configured, this is the number that would blow up fastest, faster than
liquidation risk in §3.

---

## 3. Maintenance-margin / liquidation stress

**Assumption, explicitly stated and NOT sourced from this repo** (no Bybit tier
table exists anywhere in the codebase — confirmed by grep): flat maintenance
margin rate (MMR) of **1.0% of notional**, applied uniformly across the book.
Bybit's real base-tier MMR is roughly 0.4–0.5% for BTC/ETH and typically
1–2%+ for smaller alts (this universe is ~90% alts), stepping up further at
higher tiers as single-symbol notional grows — which this flat estimate does
**not** capture, so this likely **understates** stress for any single large alt
position specifically, even though 1% is a reasonable book-wide average.

Stress defined on `total_maintenance_margin / equity = MMR × (notional/equity)`:
- **YELLOW** (>50%): maintenance margin alone consumes half of equity before any
  adverse price move — almost no buffer left.
- **RED** (>100%): maintenance margin alone exceeds equity with **zero** price
  movement — the account is already in a liquidation-adjacent state on margin
  math alone.

| f | scenario | max MM/equity | % hours YELLOW | % hours RED |
|---|---|---|---|---|
| 0.3% | uncapped | 0.35x | 0.0% | 0.0% |
| 0.3% | **capped** | 0.28x | 0.0% | 0.0% |
| 1.0% | uncapped | 1.17x | 10.6% | 0.19% |
| 1.0% | **capped** | 0.45x | 0.0% | 0.0% |
| 2.0% | uncapped | 2.33x | 45.8% | 10.6% |
| 2.0% | **capped** | 0.45x | 0.0% | 0.0% |
| 3.0% | uncapped | 3.50x | 66.0% | 29.9% |
| 3.0% | **capped** | 0.51x | 0.011% | 0.0% |

**Honest read: with `gross_open_risk_cap=0.30` doing its job (as it does live
today), pure maintenance-margin stress under this MMR assumption stays low
across all four risk levels — the cap keeps aggregate notional roughly bounded
regardless of f (§2).** The uncapped column exists to show what *would* happen
if that cap were ever removed or defeated (e.g. by a bug, or by
`net_directional_cap`/CHOP not applying to a correlated basket) — at f=2–3%
uncapped, the book would spend 30–66% of all hours already past the YELLOW
threshold. §2b's initial-margin finding is the more binding one at realistic
(capped) settings; §3's maintenance-margin finding becomes binding only if the
concurrency-limiting cap is bypassed.

---

## 4. Correlated-loss clusters — the scenario the caps exist for

Grouped every **losing** trade (`r_net < 0`, fees already included) by exit hour
and exit day, uncapped first (`4_worst_hours.csv`, `4_worst_days.csv`), then
re-run with `gross_open_risk_cap=0.30` actually gating which trades were taken,
per f (`4b_worst_cluster_gross_cap_applied.csv` — **this is the honest, live-
config-consistent figure**, since the cap is live in `config.yaml` today).

**Uncapped** worst single hour: 2025-08-22 14:00, 102 simultaneous stop-outs,
Σr_net = **−114.14R**. Worst single day: 2025-08-22, 153 stop-outs, Σr_net =
**−171.10R**.

**Capped (honest)** — the cap changes *which* trades get taken as f rises, so
the worst cluster moves to a different date (2025-03-02) and shrinks:

| f | worst hour: stop-outs / ΣR / equity loss | worst day: stop-outs / ΣR / equity loss |
|---|---|---|
| 0.3% | 64 / −72.3R / **21.7%** | 95 / −105.9R / **31.8%** |
| 1.0% | 45 / −51.1R / **51.1%** | 59 / −66.6R / **66.6%** |
| 2.0% | 41 / −46.6R / **93.3%** | 46 / −52.3R / **104.7%** |
| 3.0% | 39 / −44.5R / **133.4%** | 43 / −48.9R / **146.7%** |

Interpolating on the (honest, capped) **worst-day** figures:
- **f ≈ 0.7%** is where a repeat of the worst historical correlated-loss day
  would take out **50% of equity** (between 0.3%→31.8% and 1.0%→66.6%).
- **f ≈ 1.9%** is where the same day crosses **100% equity loss — ruin** —
  purely from stop-loss fills, no liquidation slippage included yet (between
  1.0%→66.6% and 2.0%→104.7%).

This is a **lower bound on the real ruin threshold**, not an upper bound, for
two reasons specific to cross-margin liquidation mechanics that the R-based
calculation cannot capture:
1. **Fills would be worse than modeled.** A 95–153-symbol simultaneous stop-out
   day is a correlated market event (crash/flash-crash), exactly when slippage
   through stop levels on thin alts is largest — the parquet's `sl_price` is
   the *intended* stop, not a guaranteed fill price.
2. **Cross margin liquidates on unrealized loss, not on stop-hit.** Before any
   individual stop-loss order even triggers, if the *combined mark-to-market*
   loss across all 45+ open positions pushes the shared equity pool's margin
   ratio below Bybit's threshold, the exchange force-closes positions at the
   **liquidation price** (worse than the modeled stop) to protect itself — this
   can happen intra-candle, faster than the bot's 60-second/hourly poll loop
   can react (`bot.py:2323`, `monitor_active_trades` at `bot.py:1890` is
   poll-only, no websocket).

So **f ≈ 1.9% is the point at which the worst *already-observed* historical
cluster, replayed under today's live gross-risk cap, produces total ruin from
stop-losses alone** — real liquidation mechanics would very plausibly move that
number lower, not higher.

---

## 5. Does `backtest_production_correct.py` model liquidation? **No.**

Read in full (1,060 lines). It models:
- `qty = risk_usd / sl_distance`, `position_value = qty × entry_price`
  (`backtest_production_correct.py:668-669`, matches live `bot.py:1436`).
- `required_margin = position_value / leverage` using a **hardcoded per-symbol
  leverage table** (`get_leverage()`, lines 81-89 — BTC/ETH=100x, `1000`-prefixed
  =25x, ~30 top alts=50x, else 20x) — this is the same table reused in §2b above.
- A shared `wallet_balance − margin_used` pool gate: if `required_margin >
  available`, the trade is **skipped and counted in `margin_blocked`**
  (lines 671-676). `margin_used` is incremented on open and decremented on
  close (lines 383, 728, 738) — i.e. it models **initial margin reservation
  against a shared balance**, which is the §2b failure mode.

It does **not** model, anywhere in the file (confirmed by grep — zero hits for
`liquidat|maintenance.*margin|margin_ratio|mmr`):
- Maintenance margin or a liquidation price for any position.
- Mark-to-market of open positions' floating P&L against the shared margin pool
  between entry and exit — `margin_used` only changes at trade open/close, never
  intra-trade, so a position moving deep against the stop before the stop
  triggers has **zero effect on the model's margin state**.
- Any chance that price gaps through the stop-loss and the exchange force-closes
  at a worse (liquidation) price instead of the modeled SL price.

**In short: the backtest checks whether you had enough margin to *open* a
trade, using a static per-symbol leverage table and a naive reserve/release
model — it has no concept of a trade being force-closed while open. That is
precisely the failure mode this study was commissioned to look for, and it
confirms the backtest cannot see it.**

---

## Assumptions used in this study (all stated explicitly)

1. **Maintenance margin rate = 1.0% of notional, flat, book-wide** (§3) — not
   sourced from the repo (no Bybit tier table exists here); a round,
   directionally-conservative single number given ~90% of the universe is
   altcoins. Real MMR is tiered and rises with position size per symbol, which
   this flat number does not capture — likely understates stress on any single
   large alt position.
2. **Leverage table = `backtest_production_correct.py`'s `get_leverage()`**
   (§2b, §5) — the only per-symbol leverage assumption that exists anywhere in
   this repo. It is a backtest-side approximation of "exchange maximum," not a
   live-fetched value; CLAUDE.md states the live bot fetches true exchange max
   per symbol at startup with no config ceiling, which could differ from this
   table in either direction per symbol.
3. **Equity held constant** through the concurrency/notional timeline (§2, §2b,
   §3) so that the sizing mechanics could be isolated from compounding; §4
   relaxes this only insofar as reporting loss as a *fraction* of a fixed
   starting equity, consistent with "one bad day/hour" rather than a multi-day
   compounding ruin path.
4. **`gross_open_risk_cap` is applied greedily in strict chronological order**,
   gating an entry outright (not scaling it down) when it would breach the
   30%-of-equity open-risk budget — this matches config.yaml's live setting
   (`open_risk_mode` scale-vs-block was not verified beyond this cap check;
   the live bot's exact gate logic at `bot.py:1698,1716` was not re-derived
   line-by-line for this study, only the cap *value* from `config.yaml:48,75`).
5. **`net_directional_cap` (10%) and the CHOP filter were NOT modeled** in this
   study's capped scenario — only `gross_open_risk_cap` was applied. Both would
   further reduce concurrency/notional in practice, so §2–§4's "capped" columns
   are, if anything, conservative (upper bounds on real exposure).

## Files in this directory

- `liq_analysis.py` — the full analysis script (reproducible, read-only against
  the parquet).
- `1_stop_frac_distribution.csv`, `2_concurrency_notional_summary.csv`,
  `2b_initial_margin_over_equity.csv`, `3_margin_stress.csv`,
  `4_worst_hours.csv`, `4_worst_days.csv`, `4_ruin_fraction_by_f.csv`,
  `4_breakeven_f.txt`, `4b_worst_cluster_gross_cap_applied.csv` — all numeric
  tables referenced above.
