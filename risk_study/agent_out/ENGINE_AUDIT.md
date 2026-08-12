# Engine audit — `backtest_production_correct.py::run_simulation()`

Scope: is the EQUITY / COMPOUNDING / COST arithmetic trustworthy enough to decide
whether to raise the bot's risk-per-trade, and specifically, does it get *more*
wrong as risk-per-trade scales up. Read-only audit; probes live in this directory
(`engine_probe.py` = instrumented copy of the real engine, `run_probes.py` /
`synthetic_probe.py` = drivers). All line numbers below refer to the real file
`/Users/lualakol/AutoTrading Bot/backtest_production_correct.py` unless noted.

---

## TOP DEFECTS, ranked by how much they'd distort a risk-per-trade comparison

### 1. BUG — close-batch events applied in entry-insertion order, not exit-time order
**backtest_production_correct.py:378-439 (STEP A)**

```python
closed_keys = [k for k, pos in open_positions.items()
               if pos['exit_time'] <= entry_time]
for pk in closed_keys:
    ...
    wallet_balance += pnl
    ... peak_balance / max_dd_pct updated HERE, incrementally, per pk ...
```

`open_positions` is a plain dict; `.items()` iterates in **insertion order** (the
order positions were *opened*), not by `exit_time`. Whenever two or more open
positions both have `exit_time <= entry_time` of the row that finally triggers
their closure (i.e. no intervening row's `entry_time` landed between their two
exits), they are realized to `wallet_balance` — and `peak_balance`/`max_dd_pct`
sampled — in the wrong order. This is a real bug, not a display artifact: because
`recent_closed` (the regime window) and `wallet_balance` are mutated in this same
wrong order, it also perturbs regime classification, taper rung, and downstream
sizing/margin decisions for *every trade after the first mis-ordered batch* —
i.e. it doesn't just mismeasure drawdown, it can change which trades get taken.

**Real-data demonstration** (full `regime_backtest_all_trades.csv`, scenario=`production`,
default settings, via `run_probes.py` Probe 1):
- 2,106 batches closed 2+ positions simultaneously; **1,421 of those (67%) were out of
  chronological exit-time order**; largest batch = 52 positions closed "at once."
- `max_dd_pct` **AS-IS: 87.64%** vs correctly-ordered **87.32%** (small net effect here,
  but that's a coincidence of this dataset — see synthetic below for the ceiling).
- Cascading effect on the simulation itself, not just the DD label: final balance
  **$30,902.78 (AS-IS) vs $31,094.15 (exit-order-corrected)** — a 0.6% difference —
  and **9,496 vs 9,477 trades entered**, purely from re-ordering *when the same
  closes are realized*, no other change.

**Isolated synthetic demonstration** (`synthetic_probe.py`, 3 trades, no CHOP/regime
confounds): one big loser opened first (closes last, T+5d) and one winner opened
second (closes first, T+1d), both realized in the same Step-A batch:
```
AS-IS (insertion order):      max_dd_pct = 50.33%
CORRECTED (exit-time order):  max_dd_pct = 34.22%
Final balance identical both ways: $985.35
```
+16.1 percentage points, **+47% relative overstatement of drawdown**, from pure
event-ordering, with the underlying P&L math untouched (proof it's a measurement
bug, not a PnL bug).

**Why this gets worse as risk-per-trade rises:** batch mis-ordering error is
proportional to the $ size of the swings inside a mis-ordered batch. Raise
risk-per-trade and every trade's PnL swing (and thus every batch's potential
distortion) scales up in lockstep, while the *frequency* of mis-ordered batches
(governed by trade density/timing, not risk%) stays the same. A head-to-head "risk
0.3% vs risk 1.2%" comparison will show the higher-risk arm's `max_dd_pct` polluted
by proportionally larger ordering artifacts than the lower-risk arm's.

**Fix:** sort `closed_keys` by `pos['exit_time']` before the loop (one line,
verified by `sort_close_batch=True` in the probe copy).

---

### 2. BUG — mark-to-market equity proxy leaks each open trade's *known final outcome*, and biases reported drawdown DOWN
**backtest_production_correct.py:516-543 (STEP E0)**

```python
for pos in open_positions.values():
    frac = (entry_time - pos['entry_time']).total_seconds() / span   # time-elapsed fraction
    unrealized += pos['pnl'] * frac       # <- pos['pnl'] is the trade's FINAL, already-known PnL
equity_now = wallet_balance + unrealized
```

`pos['pnl']` was computed once, at the position's *own* open (line 713,
`gross_pnl = r_result * risk_usd`), from `r_result` — a value the CSV already
knows because it's a completed historical replay. So "unrealized PnL" here isn't a
live/uncertain mark, it's *the trade's fully-known future outcome*, smoothly
apportioned backward in time by a linear time-fraction. Two consequences:

- **Lookahead (defect G):** whenever the caller uses `taper_basis='equity'` or
  `size_basis='equity'` — which the function's own docstring says is *exactly* the
  pair that "mirrors the live bot's actual mismatched config" — a currently-open
  trade's ultimate win/loss changes the position sizing of *other, unrelated*
  trades entered while it's still open. A trade that will eventually close +9R
  starts inflating other trades' risk budget from the moment it opens, not from
  when the market has actually earned that gain.
- **Drawdown bias (defect B):** real price paths are not linear — a trade that
  finishes +9R may well have sat at -1R mid-flight (per `CLAUDE.md` §6.1, this is
  the whole reason the trailing-stop feature exists: arming happens on fresh
  per-bar excursions because price genuinely round-trips). The linear-interpolation
  proxy can never show that trough. `max_dd_mtm_pct` is *structurally* a floor on
  true intra-trade drawdown, not an estimate of it.

**Numeric demonstration** (`run_probes.py` Probe 4, real data, `production` scenario):
```
size_basis=wallet:  max_dd_pct 87.6%   max_dd_mtm_pct 87.3%
size_basis=equity:  max_dd_pct 87.7%   max_dd_mtm_pct 83.0%
```
Switching to the equity/MTM basis — the config the docstring says matches the live
bot — pulls the *reported* drawdown down by **4.6 percentage points** relative to
the wallet-based measurement in the very same run, in the direction that makes
raising risk look safer than it is.

**Why this gets worse at higher risk-per-trade:** the interpolation error is
proportional to each open trade's PnL magnitude, which scales directly with
risk-per-trade. At higher risk, more $ of "already-known-but-not-yet-real" PnL
gets smoothed into `equity_now` at any moment, so both the lookahead-driven sizing
distortion and the drawdown-understatement widen together — right when the study
most needs an honest drawdown number.

**Fix:** for sizing/taper, `equity_now` should be built from a genuine live mark
(if tick data isn't available, at minimum stop interpolating toward the *known*
final PnL — interpolate toward 0 with a volatility-scaled band, or drop the equity
basis for the risk-per-trade study and report only the wallet-based `max_dd_pct`,
itself fixed per item 1).

---

### 3. BUG (real, but scenario-invariant — low priority for a *relative* risk comparison) — round-trip cost priced entirely off *entry* notional
**backtest_production_correct.py:664-685**

```python
qty = risk_usd / sl_distance
position_value = qty * entry_price          # ENTRY notional, fixed at open
...
trade_cost = position_value * ROUND_TRIP_COST   # both legs priced off entry notional
```

`ROUND_TRIP_COST = 0.0018` is meant to cover fee+slippage on *both* legs
(`2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)`), but it's applied once, entirely against
the entry-time notional. The exit leg's true fee/slippage should be
`qty * exit_price * (fee+slip)`, not `qty * entry_price * (fee+slip)`. For this
RR-3-to-10 strategy the two can differ a lot on winners.

**Demonstration** (`run_probes.py` Probe 3, real >5R winners):
```
1000TURBOUSDT  R=+5.96  entry->exit notional drift ~19.7%
SCUSDT         R=+9.92  entry->exit notional drift ~14.4%
TRUMPUSDT      R=+9.97  entry->exit notional drift ~37.5%
BLURUSDT       R=+9.96  entry->exit notional drift ~32.3%
```
The exit leg's true cost is understated by exactly that drift %, i.e. total
round-trip cost is understated by roughly *half* that (the entry leg is priced
correctly) — up to ~19% understatement of total cost on the biggest winners.
Per `CLAUDE.md`, "top 1% of trades = 24% of profit" — this cost error concentrates
precisely on the trades that drive the reported edge.

**Why it's ranked #3, not #1:** the error is a fixed *percentage* of notional and
scales identically with position size at every risk-per-trade level — it inflates
absolute PF/ROI for every scenario roughly equally, so it does **not** differentially
distort a risk-0.3%-vs-1.2% *comparison* the way #1 and #2 do. It should still be
fixed (it inflates the standalone case for "the strategy is profitable enough to
raise risk"), but it's a scenario-invariant bias, not a risk-scaling-sensitive one.

**Fix:** compute `trade_cost` as `qty * entry_price * (fee+slip) + qty * exit_price
* (fee+slip)`; `exit_price` is recoverable from `entry_price`, `sl_price`, and
`r_result` the same way the probe estimates it.

---

### 4. Known, already-documented mismatch (not a bug in this file) — opposite-side same-symbol concurrency is allowed here, disallowed live
**backtest_production_correct.py:441-444 (anti-pyramid)** vs **`autobot/core/bot.py` ~1935-1943 (live OPPOSITE-SIDE GUARD)**

```python
trade_key = f"{symbol}_{side}"
if trade_key in open_positions:
    pyramid_blocked += 1
    continue
```

This only blocks a *second* position on the same `symbol+side`; a long and a short
on the same symbol can be open at once in the sim. The live bot cannot do this — it
trades one-way mode (`positionIdx=0`), so an opposite-side entry force-closes the
existing position instead of hedging. The live code's own comment already flags
this exact gap:
> "Backtests model both sides as independent coexisting positions and never model
> that forced closure (~3.4% of signals collide, ~58/mo)."

**Confirmation** (`run_probes.py` Probe 5, full candidate set pre-filter): 123/330
symbols have both long and short signals; **506 raw overlapping long+short windows**
on the same symbol.

**Relevance to risk scaling:** low-to-moderate. It modestly overstates achievable
diversification/net-directional headroom (both "legs" get counted as independent
risk when live only one can exist), but the ~3.4% collision rate is a fixed
fraction of signal flow and scales proportionally with risk-per-trade like
everything else — it doesn't preferentially help or hurt one risk level over
another. Flagging for completeness since it was explicitly in scope (defect E),
not because it changes the risk-per-trade verdict.

---

### 5. SUSPICIOUS / modeling blind spot (not a bug — explicitly opt-in, but this is exactly the gap a risk-raise decision needs closed) — no liquidation model, no default market-impact model
**backtest_production_correct.py:664-728 (position construction), :693-711 (opt-in "REALISM LAYER")**

- There is no liquidation price, no check that a loss could exceed posted margin,
  and no code path where equity goes negative or a position is force-closed by the
  exchange. This is possible *by construction*: every loss is capped at exactly the
  CSV's `r_result` (comment at line 682: "SL = -1" raw), so the engine can never
  represent a stop that fails to fill at its intended level (a gap, a liquidity
  hole, a stale/duplicate order, exchange downtime). **This engine cannot represent
  an account blowup; it also cannot let equity "recover" from one because it never
  happens.** State this plainly to whoever reads the risk-per-trade study: a clean
  `max_dd_pct` here says nothing about tail/liquidation risk.
- Market-impact/liquidity modeling exists (`liq_turnover_col`, `liq_impact_k`,
  `liq_skip_frac`, `stop_gap_pct`, lines 687-711) but is **off by default**
  (`liq_impact_k=0.0`, `liq_skip_frac=0.0`, `stop_gap_pct=0.0`). Unless a caller
  explicitly turns these on, raising risk-per-trade in the sim scales position size
  on 277 symbols (many thin alts, per `CLAUDE.md`'s 20x-default leverage tier) with
  **zero** cost penalty for the larger clip size relative to each symbol's real
  liquidity.

**Why this matters most for exactly this decision:** the entire mechanism by which
"raise risk-per-trade" goes wrong in real trading — worse fills, slippage-through-
stop, occasional liquidation on a thin symbol — is present in the code as opt-in
flags and is silent by default. A comparison run without `liq_impact_k`/
`liq_skip_frac`/`stop_gap_pct` set will make higher risk-per-trade look uniformly
better with no penalty, because the mechanism that would push back is turned off.
**Recommend the risk-per-trade study explicitly enable and sweep these three
parameters, using each symbol's real turnover if available, rather than trusting
the zero-impact default.**

---

## Other findings (checked, mostly clean)

- **A. Compounding / balance-at-entry (backtest_production_correct.py:369, 545-546,
  625):** `risk_usd` sizing correctly uses `wallet_balance` (or `equity_now`, if
  configured) *as observed at the current entry_time*, i.e. after all prior closes
  up to that point have already been applied via STEP A, and before this trade's
  own outcome exists. **CLEAN in principle** — the only way it goes wrong is
  indirectly, through defect #1 (wrong close-batch order corrupts what
  "wallet_balance at entry_time" actually is for later trades).
- **PnL applied exactly once (backtest_production_correct.py:399-400, 713,
  747-748):** `pos['pnl']` is computed once at open (`gross_pnl - trade_cost`),
  `funding_cost` subtracted once at close, `wallet_balance += pnl` happens exactly
  once per position (STEP A pop, or the end-of-data force-close loop — mutually
  exclusive, a position can only be popped once). **CLEAN**, cost is applied
  before the balance update (correct order).
- **D. Margin accounting (backtest_production_correct.py:383/728/723/738):**
  `margin_used` is incremented by `required_margin` at open and decremented by the
  *same stored dict value* (`pos['margin']`) at close — never recomputed — so no
  drift is possible by construction. **Empirically confirmed**: instrumented run
  over the full real dataset (all 3 scenarios) ends with `final_margin_used` ≈
  `1e-11`–`2e-11` (floating-point noise, i.e. exactly 0). **CLEAN, no leak.**
  Minor simplification worth noting: the margin *check* (line 674,
  `available = wallet_balance - margin_used`) always uses realized `wallet_balance`,
  never `equity_now`, even when `size_basis='equity'`. This is a conservative bias
  (real cross-margin accounts count unrealized gains toward buying power) — it
  would make the engine *under*-state achievable position count at higher risk, the
  opposite direction from defects #1/#2. Worth knowing, not urgent.
- **E. Anti-pyramid bypass:** cannot be bypassed for same `symbol+side` — dict-key
  check is airtight (`trade_key in open_positions`). The only gap is opposite-side
  concurrency, covered as defect #4 above (documented elsewhere, not a "bypass").
- **F. Liquidation:** see defect #5 — no model, plainly stated.
- **G. Lookahead:** the CHOP-blocked and daily-halt-r shadow tracking correctly use
  only past/entry-time-knowable data (`daily_pnl` only accumulates from already-
  realized closes, `lookup_chop` explicitly steps back one bar to avoid the
  documented CHOP-lookahead bug from `chop-lookahead-contaminates-backtests`). The
  one real outcome-leaks-into-entry-decision path is defect #2 above (equity
  interpolation using each open trade's already-known final PnL).

---

## Bottom line for the risk-per-trade study

Two real bugs (#1 close-batch ordering, #2 equity-interpolation lookahead) both
bias in the *same, dangerous* direction for this specific use — they make
`max_dd_pct` / `max_dd_mtm_pct` **artificially better** (or at minimum
non-deterministically wrong) precisely as risk-per-trade scales up, and precisely
in the ROI/maxDD ratio the study is built around. Recommend, before trusting any
raise-risk verdict from this engine:

1. Fix #1 (`closed_keys` sort by `exit_time`) — trivial, ~1 line, verified in
   `engine_probe.py`.
2. Either fix #2's equity proxy or restrict the risk-per-trade comparison to
   `taper_basis='wallet', size_basis='wallet'` and report only `max_dd_pct` (now
   fixed by #1), not `max_dd_mtm_pct`.
3. Explicitly enable and sweep `liq_impact_k` / `liq_skip_frac` / `stop_gap_pct`
   (defect #5) rather than relying on the zero-impact default — this is the one
   piece of the engine actually designed to answer "what happens to costs when
   position size goes up," and it's off unless asked for.
4. #3 (entry-notional-only cost) and #4 (opposite-side concurrency) are real but
   scenario-invariant; fix opportunistically, don't block the study on them.

## Probe files (this directory)
- `engine_probe.py` — instrumented copy of `run_simulation` (adds
  `sort_close_batch` + `batch_stats` params, exposes `final_margin_used`). No
  behavior change when `sort_close_batch=False` and `batch_stats=None` (defaults).
- `run_probes.py` — drives 5 probes against real `regime_backtest_all_trades.csv` +
  `cache_3yr_1h` CHOP data.
- `synthetic_probe.py` — isolated 3-trade reproduction of defect #1.
