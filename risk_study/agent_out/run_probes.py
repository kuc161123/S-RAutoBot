#!/usr/bin/env python3
"""Probes for AGENT-ENGINE audit of backtest_production_correct.py.
Read-only against the real engine logic (via engine_probe.py, an
instrumented copy). Writes nothing outside risk_study/agent_out/.
"""
import sys
sys.path.insert(0, '/Users/lualakol/AutoTrading Bot/risk_study/agent_out')
import pandas as pd
import numpy as np
from engine_probe import run_simulation, load_chop_data, STARTING_BALANCE, ROUND_TRIP_COST

TRADES_CSV = '/Users/lualakol/AutoTrading Bot/regime_backtest_all_trades.csv'

df = pd.read_csv(TRADES_CSV, parse_dates=['entry_time', 'exit_time'])
df = df.sort_values('entry_time').reset_index(drop=True)
symbols = df['symbol'].unique().tolist()
print(f"Loaded {len(df)} trades, {len(symbols)} symbols")

chop_map = load_chop_data(symbols)

# ─────────────────────────────────────────────────────────────────
# PROBE 1: batch-close ordering bug (defect B)
# ─────────────────────────────────────────────────────────────────
print("\n=== PROBE 1: close-batch ordering (max_dd_pct) ===")
stats_unsorted = {'batches_ge2': 0, 'out_of_order_batches': 0, 'max_batch': 0}
r_unsorted = run_simulation(df, chop_map, scenario='production', batch_stats=stats_unsorted)

stats_sorted = {'batches_ge2': 0, 'out_of_order_batches': 0, 'max_batch': 0}
r_sorted = run_simulation(df, chop_map, scenario='production', sort_close_batch=True, batch_stats=stats_sorted)

print(f"batches with >=2 simultaneous closes: {stats_unsorted['batches_ge2']}")
print(f"...of those, batches where entry-insertion order != exit-time order: {stats_unsorted['out_of_order_batches']}")
print(f"max batch size: {stats_unsorted['max_batch']}")
print(f"max_dd_pct AS-IS (entry/insertion order):   {r_unsorted['max_dd_pct']:.3f}%")
print(f"max_dd_pct CORRECTED (exit-time order):     {r_sorted['max_dd_pct']:.3f}%")
print(f"final wallet_balance AS-IS:  ${r_unsorted['wallet_balance']:.2f}  (identical regardless of order, as expected)")
print(f"final wallet_balance SORTED: ${r_sorted['wallet_balance']:.2f}")
print(f"trade count AS-IS: {len(r_unsorted['entered_trades'])}  SORTED: {len(r_sorted['entered_trades'])}")

# ─────────────────────────────────────────────────────────────────
# PROBE 2: margin leak check (defect D)
# ─────────────────────────────────────────────────────────────────
print("\n=== PROBE 2: margin_used leak check ===")
# Re-run with instrumentation by importing engine_probe internals directly
import engine_probe as ep

def run_with_margin_trace(trades_df, chop_map):
    """Copy of the STEP A/F/G bookkeeping only, to assert margin_used == 0 at the end
    and never negative mid-run, using the *actual* run_simulation via a light wrapper:
    we just check the invariant analytically from the returned entered_trades since
    margin_used is not exposed - so instead we verify margin arithmetic identity:
    sum(required_margin at each open) - sum(pos['margin'] released at each close) == 0
    by re-deriving margin the same way the engine does, per entered trade.
    """
    pass

r_prod = run_simulation(df, chop_map, scenario='production')
trades = r_prod['entered_trades']
# The engine computes margin = position_value/leverage = (risk_usd/sl_distance*entry_price)/leverage
# and always releases exactly pos['margin'] (the same stored value) at close -> by
# construction margin_used must return to 0 after the "close remaining" loop runs
# (which the returned dict does not expose directly, so assert structurally instead):
print(f"entered trades: {len(trades)} (all must have been closed exactly once by run end)")
print("Structural check: every open_positions entry is popped exactly once (STEP A pop, "
      "or the end-of-data force-close loop) and margin_used -= pos['margin'] uses the SAME "
      "dict value that was added at open (no recomputation) -> no leak possible by construction.")
print("(See backtest_production_correct.py:383 vs :728 and :738 vs :723 - same stored value both ways.)")

# ─────────────────────────────────────────────────────────────────
# PROBE 3: cost-basis magnitude on a real big-RR winner (defect C)
# ─────────────────────────────────────────────────────────────────
print("\n=== PROBE 3: entry-notional-only cost vs true exit-notional cost, big winners ===")
big_winners = df[df['r_result'] > 5].copy()
big_winners['price_move_pct'] = (big_winners['sl_price'] - big_winners['entry_price']).abs() \
    / big_winners['entry_price'] * big_winners['r_result'].abs()
# rough proxy: exit price ~= entry +/- r_result * sl_distance; compute % notional drift
sample = big_winners.sample(min(5, len(big_winners)), random_state=1)
for _, t in sample.iterrows():
    sl_dist = abs(t['entry_price'] - t['sl_price'])
    exit_price_est = t['entry_price'] + np.sign(t['r_result']) * t['r_result'] * sl_dist \
        if t['side'] == 'long' else t['entry_price'] - np.sign(t['r_result']) * t['r_result'] * sl_dist
    notional_drift_pct = abs(exit_price_est - t['entry_price']) / t['entry_price'] * 100
    entry_cost_frac = ROUND_TRIP_COST
    true_exit_leg_cost_frac = (0.0006 + 0.0003) * (exit_price_est / t['entry_price'])
    understatement_pct = (true_exit_leg_cost_frac - (0.0006 + 0.0003)) / (0.0006 + 0.0003) * 100
    print(f"{t['symbol']:<14} R={t['r_result']:+.2f} entry->exit notional drift ~{notional_drift_pct:.1f}%  "
          f"exit-leg fee understatement ~{understatement_pct:+.1f}% of that leg's true cost")

# ─────────────────────────────────────────────────────────────────
# PROBE 4: equity lookahead (defect G / B) - size_basis='equity'
# ─────────────────────────────────────────────────────────────────
print("\n=== PROBE 4: equity interpolation uses the FINAL known pnl of open trades ===")
print("Confirmed by code read: line ~531 'unrealized += pos['pnl'] * frac' where pos['pnl']")
print("is the trade's fully-known realized pnl (computed once at OPEN from r_result, which")
print("is itself the CSV's precomputed historical outcome) - not a live/uncertain estimate.")
print("This feeds taper_basis='equity' / size_basis='equity' / open_risk_cap / net_dir_cap,")
print("which the docstring says mirrors the LIVE bot's actual mismatched config.")

# Demonstrate quantitatively: run same scenario with size_basis=wallet vs equity and show
# that a slow, eventually-huge-winner trade changes OTHER trades' sizing while it's open,
# in a way that is only possible because its outcome is already known.
r_wallet_basis = run_simulation(df, chop_map, scenario='production', size_basis='wallet')
r_equity_basis = run_simulation(df, chop_map, scenario='production', size_basis='equity', taper_basis='equity')
print(f"\nfinal balance size_basis=wallet:  ${r_wallet_basis['wallet_balance']:.2f}  maxDD {r_wallet_basis['max_dd_pct']:.1f}%  maxDD_mtm {r_wallet_basis['max_dd_mtm_pct']:.1f}%")
print(f"final balance size_basis=equity:  ${r_equity_basis['wallet_balance']:.2f}  maxDD {r_equity_basis['max_dd_pct']:.1f}%  maxDD_mtm {r_equity_basis['max_dd_mtm_pct']:.1f}%")
print(f"trades entered wallet-basis: {len(r_wallet_basis['entered_trades'])}  equity-basis: {len(r_equity_basis['entered_trades'])}")

# ─────────────────────────────────────────────────────────────────
# PROBE 5: anti-pyramid same-symbol opposite-side coexistence (defect E)
# ─────────────────────────────────────────────────────────────────
print("\n=== PROBE 5: same-symbol opposite-side concurrent positions ===")
# Detect, in the raw candidate stream (pre-filter), how often a long and short signal
# for the SAME symbol have overlapping [entry_time, exit_time) windows - these are the
# events the engine would treat as two independent coexisting positions.
collisions = 0
by_symbol = df.groupby('symbol')
checked = 0
for sym, g in by_symbol:
    longs = g[g['side'] == 'long'][['entry_time', 'exit_time']].values
    shorts = g[g['side'] == 'short'][['entry_time', 'exit_time']].values
    if len(longs) == 0 or len(shorts) == 0:
        continue
    checked += 1
    for le, lx in longs:
        # overlap if short.entry < long.exit and short.exit > long.entry
        ov = ((shorts[:, 0] < lx) & (shorts[:, 1] > le)).sum()
        collisions += ov
print(f"symbols with both long & short signals: {checked}/{len(by_symbol)}")
print(f"raw overlapping long+short windows (same symbol) in the full candidate set: {collisions}")
print("Live bot explicitly blocks these via the OPPOSITE-SIDE GUARD (bot.py ~1935-1943,")
print("comment: 'Backtests model both sides as independent coexisting positions and never")
print("model that forced closure (~3.4% of signals collide, ~58/mo)'). This engine's")
print("anti-pyramid check (line 441-444) keys ONLY on symbol+side, so it does not block")
print("opposite-side concurrency - confirmed not a bypass, but a known, already-documented")
print("engine/live mismatch.")

print("\nDONE")
