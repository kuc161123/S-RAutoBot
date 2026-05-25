#!/usr/bin/env python3
"""
PRODUCTION-CORRECT BACKTEST
============================
Event-driven simulation matching the exact bot formulas:
  - Position sizing: qty = risk_usd / sl_distance (bot.py:1436)
  - Margin: position_value / leverage (bot.py:1446)
  - Leverage: per-symbol from Bybit risk limits (bot.py:1439-1442)
  - Regime: 4-tier from 20-trade closed window (bot.py:549-601)
  - CHOP filter: regime-aware thresholds (bot.py:1335-1354)
  - Taper schedule: balance-based risk reduction (bot.py:621-642)
  - Anti-pyramid: 1 position per symbol+side (bot.py:1356-1360)
  - Funding fees: per 8-hour period held

Data source: regime_backtest_all_trades.csv (12,021 trades, May 2025 - Apr 2026)
  - Has exact entry_price, sl_price, r_result, side, symbol, entry/exit times
  - sl_distance = abs(entry_price - sl_price) — no estimation needed

3 Scenarios:
  1. PRODUCTION — regime + CHOP + taper + anti-pyramid + correct margin
  2. NO REGIME — flat 1.2% risk, CHOP + anti-pyramid + correct margin
  3. NO FILTERS — flat 1.2% risk, no CHOP, anti-pyramid + correct margin only
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from copy import deepcopy

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

TRADES_CSV = Path('/Users/lualakol/AutoTrading Bot/regime_backtest_all_trades.csv')
CACHE_DIR = Path('/Users/lualakol/AutoTrading Bot/cache_3yr_1h')
CHOP_PERIOD = 14

STARTING_BALANCE = 850.0
MAX_BALANCE = 50_000.0
BASE_RISK = 0.012

TAPER_SCHEDULE = [
    (1500, 0.007), (2000, 0.006), (3000, 0.0055), (5000, 0.005),
    (8000, 0.0045), (12000, 0.004), (20000, 0.0035), (40000, 0.003),
]

# Aligned with bot.py:1351 — chop blocks when value >= threshold per regime
CHOP_THRESHOLDS = {'favorable': 52, 'cautious': 45, 'adverse': 52, 'critical': 55}

FUNDING_LONG = 0.0001    # 0.01% per 8h (longs pay)
FUNDING_SHORT = -0.00003  # shorts earn slightly
# Per-side trading + slippage cost, applied on both entry and exit on notional.
# Matches config.yaml: execution.fee_pct=0.0006, execution.entry_slippage_pct=0.0003.
# Note: bot config only defines entry slippage; we apply the same to exit since
# real fills slip in both directions. Round-trip cost = ~0.18% of notional.
FEE_PER_SIDE = 0.0006
SLIPPAGE_PER_SIDE = 0.0003
ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)  # 0.0018

# Top liquid alts that get 50x leverage on Bybit
TOP_ALTS_50X = {
    'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT',
    'DOTUSDT', 'MATICUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'APTUSDT',
    'NEARUSDT', 'FILUSDT', 'ARBUSDT', 'OPUSDT', 'MKRUSDT', 'AAVEUSDT',
    'ATOMUSDT', 'XLMUSDT', 'TRXUSDT', 'ICPUSDT', 'SUIUSDT', 'SEIUSDT',
    'TIAUSDT', 'STXUSDT', 'INJUSDT', 'IMXUSDT', 'RUNEUSDT', 'FETUSDT',
    'WLDUSDT', 'PEPEUSDT', 'SHIBUSDT', 'BNBUSDT', 'TONUSDT', 'FTMUSDT',
    'RNDRUSDT', 'GRTUSDT', 'THETAUSDT', 'ALGOUSDT', 'VETUSDT', 'SANDUSDT',
    'MANAUSDT', 'AXSUSDT', 'EGLDUSDT', 'FLOWUSDT', 'GALAUSDT', 'APEUSDT',
    'LDOUSDT', 'CRVUSDT', 'SNXUSDT', 'COMPUSDT', 'GMXUSDT', 'PENDLEUSDT',
    'JUPUSDT', 'WUSDT', 'ENAUSDT', 'ONDOUSDT', 'POLUSDT',
}


# ═══════════════════════════════════════════════════════════════════════════════
# LEVERAGE LOOKUP (matches Bybit risk limits)
# ═══════════════════════════════════════════════════════════════════════════════

def get_leverage(symbol: str) -> int:
    """Per-symbol leverage matching Bybit risk limits."""
    if symbol in ('BTCUSDT', 'ETHUSDT'):
        return 100
    if symbol.startswith('1000') or symbol.startswith('10000'):
        return 25
    if symbol in TOP_ALTS_50X:
        return 50
    return 20  # default for smaller alts


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME (bot.py:521-609)
# ═══════════════════════════════════════════════════════════════════════════════

def get_regime(recent_trades):
    """4-Tier Graduated regime from 20-trade closed window."""
    n = len(recent_trades)
    if n < 10:
        return 'critical', 0.1

    window = recent_trades[-20:]
    wr = sum(1 for t in window if t['r'] > 0) / len(window)
    avg_r = sum(t['r'] for t in window) / len(window)

    if wr >= 0.18 and avg_r >= 0.15:
        return 'favorable', 1.0
    elif wr >= 0.18 or avg_r >= 0.10:
        return 'cautious', 0.5
    elif wr >= 0.10 or avg_r >= -0.5:
        return 'adverse', 0.25
    else:
        return 'critical', 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# TAPER (bot.py:621-642)
# ═══════════════════════════════════════════════════════════════════════════════

def get_tapered_risk(wallet_balance):
    """Balance-based taper schedule."""
    base = BASE_RISK
    for threshold, risk in TAPER_SCHEDULE:
        if wallet_balance >= threshold:
            base = risk
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# CHOP INDEX COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chop(df):
    """Compute Choppiness Index on 1H OHLCV data."""
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    atr_sum = tr.rolling(CHOP_PERIOD).sum()
    highest = df['high'].rolling(CHOP_PERIOD).max()
    lowest = df['low'].rolling(CHOP_PERIOD).min()
    hl_diff = highest - lowest

    chop = 100 * np.log10(atr_sum / (hl_diff + 1e-10)) / np.log10(CHOP_PERIOD)
    return chop


def load_chop_data(symbols):
    """Load and compute CHOP for all symbols. Returns {symbol: pd.Series indexed by datetime}."""
    print(f"\n  Loading CHOP data for {len(symbols)} symbols...")
    chop_map = {}
    loaded = 0
    for sym in symbols:
        fpath = CACHE_DIR / f'{sym}.parquet'
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        df = df.set_index('start').sort_index()
        chop_series = compute_chop(df)
        chop_series.index = df.index
        chop_map[sym] = chop_series
        loaded += 1
        if loaded % 50 == 0:
            print(f"    ... loaded {loaded}/{len(symbols)}")

    print(f"  Loaded CHOP for {loaded} symbols")
    return chop_map


def lookup_chop(chop_map, symbol, entry_time):
    """Get CHOP value at entry_time for a symbol (nearest 1H candle <= entry_time)."""
    if symbol not in chop_map:
        return None
    series = chop_map[symbol]
    # Floor to nearest hour
    ts = pd.Timestamp(entry_time).floor('h')
    if ts in series.index:
        val = series.loc[ts]
        return val if pd.notna(val) else None
    # Try getting the last available value before entry_time
    mask = series.index <= ts
    if mask.any():
        val = series.loc[mask].iloc[-1]
        return val if pd.notna(val) else None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT-DRIVEN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(trades_df, chop_map, scenario='production'):
    """
    Run event-driven backtest simulation.

    scenario:
      'production' — regime + CHOP + taper + anti-pyramid + correct margin
      'no_regime'  — flat 1.2% risk, CHOP + anti-pyramid + correct margin
      'no_filters' — flat 1.2% risk, no CHOP, anti-pyramid + correct margin only
    """
    wallet_balance = STARTING_BALANCE
    total_withdrawn = 0.0
    margin_used = 0.0
    open_positions = {}  # key: "SYMBOL_side" -> position dict
    recent_closed = []   # regime window: list of {'r': float}
    entered_trades = []
    regime_transitions = []

    # Blocking counters
    pyramid_blocked = 0
    chop_blocked_list = []  # shadow-track outcomes
    margin_blocked = 0
    regime_info_log = []

    # Track peak for drawdown
    peak_balance = STARTING_BALANCE
    max_dd_pct = 0.0

    # Monthly tracking
    monthly_pnl = defaultdict(float)
    monthly_balance = {}

    prev_regime = None

    for idx, trade in trades_df.iterrows():
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        sl_price = trade['sl_price']
        r_result = trade['r_result']
        side = trade['side']
        symbol = trade['symbol']

        # ─── STEP A: Close expired positions (exit_time <= current entry_time) ───
        closed_keys = [k for k, pos in open_positions.items()
                       if pos['exit_time'] <= entry_time]

        for pk in closed_keys:
            pos = open_positions.pop(pk)
            margin_used -= pos['margin']

            # Funding fees
            hold_hours = (pos['exit_time'] - pos['entry_time']).total_seconds() / 3600
            funding_periods = hold_hours / 8.0
            if pos['side'] == 'long':
                funding_cost = pos['position_value'] * FUNDING_LONG * funding_periods
            else:
                funding_cost = pos['position_value'] * FUNDING_SHORT * funding_periods

            pnl = pos['pnl'] - funding_cost
            wallet_balance += pnl

            # $50K cap withdrawal
            if wallet_balance > MAX_BALANCE:
                withdrawn = wallet_balance - MAX_BALANCE
                total_withdrawn += withdrawn
                wallet_balance = MAX_BALANCE

            # Track monthly P&L by exit month
            exit_month = pos['exit_time'].strftime('%Y-%m')
            monthly_pnl[exit_month] += pnl

            # Update regime window (closed trades only)
            recent_closed.append({'r': pos['r_result']})

            # Record completed trade
            entered_trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': pos['exit_time'],
                'symbol': pos['symbol'],
                'side': pos['side'],
                'r_result': pos['r_result'],
                'risk_usd': pos['risk_usd'],
                'pnl': pnl,
                'balance_after': wallet_balance,
                'regime': pos['regime'],
                'regime_mult': pos['regime_mult'],
            })

            # Peak / drawdown tracking
            effective_bal = wallet_balance + total_withdrawn
            if effective_bal > peak_balance:
                peak_balance = effective_bal
            dd = (peak_balance - effective_bal) / peak_balance * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

        # ─── STEP B: Anti-pyramid check ───
        trade_key = f"{symbol}_{side}"
        if trade_key in open_positions:
            pyramid_blocked += 1
            continue

        # ─── STEP C: Regime ───
        if scenario == 'production':
            regime_label, regime_mult = get_regime(recent_closed)
        else:
            regime_label, regime_mult = 'favorable', 1.0  # flat risk for no_regime and no_filters

        # Track regime transitions
        if regime_label != prev_regime:
            regime_transitions.append({
                'time': entry_time,
                'from': prev_regime,
                'to': regime_label,
                'mult': regime_mult,
                'n_closed': len(recent_closed),
                'balance': wallet_balance,
            })
            prev_regime = regime_label

        # ─── STEP D: CHOP filter ───
        if scenario == 'production':
            chop_thresh = CHOP_THRESHOLDS.get(regime_label)  # favorable=None (never blocked)
        elif scenario == 'no_regime':
            chop_thresh = 55  # fixed cautious-level threshold regardless of regime
        else:
            chop_thresh = None  # no_filters: no CHOP at all

        if chop_thresh is not None:
            chop_val = lookup_chop(chop_map, symbol, entry_time)
            if chop_val is not None and chop_val >= chop_thresh:
                chop_blocked_list.append({
                    'symbol': symbol, 'side': side,
                    'entry_time': entry_time, 'r_result': r_result,
                    'regime': regime_label, 'chop_val': chop_val,
                    'chop_thresh': chop_thresh,
                })
                continue

        # ─── STEP E: Position sizing (THE CRITICAL FIX) ───
        if scenario == 'production':
            base_risk = get_tapered_risk(wallet_balance)
            risk_pct = base_risk * regime_mult
        else:
            risk_pct = BASE_RISK  # flat 1.2% for no_regime and no_filters

        risk_usd = wallet_balance * risk_pct
        if risk_usd < 0.01:
            continue

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            continue

        qty = risk_usd / sl_distance
        position_value = qty * entry_price
        leverage = get_leverage(symbol)
        required_margin = position_value / leverage

        # ─── STEP F: Margin check ───
        available = wallet_balance - margin_used
        if required_margin > available:
            margin_blocked += 1
            continue

        # ─── STEP G: Open position ───
        # Gross P&L from R-result, then subtract round-trip trading + slippage
        # cost on notional. r_result in the CSV is now expected to be RAW
        # (TP = +rr, SL = -1); fees are applied here so the cost model is
        # transparent and consistent across CSVs.
        gross_pnl = r_result * risk_usd
        trade_cost = position_value * ROUND_TRIP_COST
        pnl = gross_pnl - trade_cost

        open_positions[trade_key] = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'side': side,
            'r_result': r_result,
            'risk_usd': risk_usd,
            'pnl': pnl,
            'margin': required_margin,
            'position_value': position_value,
            'regime': regime_label,
            'regime_mult': regime_mult,
        }
        margin_used += required_margin

    # ─── Close remaining open positions at end of data ───
    for pk, pos in list(open_positions.items()):
        margin_used -= pos['margin']

        hold_hours = (pos['exit_time'] - pos['entry_time']).total_seconds() / 3600
        funding_periods = hold_hours / 8.0
        if pos['side'] == 'long':
            funding_cost = pos['position_value'] * FUNDING_LONG * funding_periods
        else:
            funding_cost = pos['position_value'] * FUNDING_SHORT * funding_periods

        pnl = pos['pnl'] - funding_cost
        wallet_balance += pnl

        if wallet_balance > MAX_BALANCE:
            withdrawn = wallet_balance - MAX_BALANCE
            total_withdrawn += withdrawn
            wallet_balance = MAX_BALANCE

        exit_month = pos['exit_time'].strftime('%Y-%m')
        monthly_pnl[exit_month] += pnl

        recent_closed.append({'r': pos['r_result']})

        entered_trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': pos['exit_time'],
            'symbol': pos['symbol'],
            'side': pos['side'],
            'r_result': pos['r_result'],
            'risk_usd': pos['risk_usd'],
            'pnl': pnl,
            'balance_after': wallet_balance,
            'regime': pos['regime'],
            'regime_mult': pos['regime_mult'],
        })

        effective_bal = wallet_balance + total_withdrawn
        if effective_bal > peak_balance:
            peak_balance = effective_bal
        dd = (peak_balance - effective_bal) / peak_balance * 100
        if dd > max_dd_pct:
            max_dd_pct = dd

    open_positions.clear()

    return {
        'scenario': scenario,
        'wallet_balance': wallet_balance,
        'total_withdrawn': total_withdrawn,
        'final_effective': wallet_balance + total_withdrawn,
        'max_dd_pct': max_dd_pct,
        'entered_trades': entered_trades,
        'pyramid_blocked': pyramid_blocked,
        'chop_blocked': chop_blocked_list,
        'margin_blocked': margin_blocked,
        'regime_transitions': regime_transitions,
        'monthly_pnl': dict(monthly_pnl),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT / REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results, total_trades):
    """Print comprehensive results for a scenario."""
    sc = results['scenario'].upper().replace('_', ' ')
    trades = results['entered_trades']
    n = len(trades)

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO: {sc}")
    print(f"{'=' * 70}")

    if n == 0:
        print("  No trades entered!")
        return

    wins = sum(1 for t in trades if t['r_result'] > 0)
    losses = n - wins
    wr = wins / n * 100
    avg_r = np.mean([t['r_result'] for t in trades])
    total_pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    final = results['final_effective']
    ret_pct = (final - STARTING_BALANCE) / STARTING_BALANCE * 100

    print(f"\n  Starting:    ${STARTING_BALANCE:,.2f}")
    print(f"  Final Bal:   ${results['wallet_balance']:,.2f}")
    print(f"  Withdrawn:   ${results['total_withdrawn']:,.2f}")
    print(f"  Effective:   ${final:,.2f}  ({ret_pct:+,.1f}%)")
    print(f"  Max DD:      {results['max_dd_pct']:.1f}%")
    print(f"  Return/DD:   {ret_pct / results['max_dd_pct']:.1f}" if results['max_dd_pct'] > 0 else "  Return/DD:   inf")
    print(f"\n  Trades:      {n} entered")
    print(f"  Win Rate:    {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg R:       {avg_r:+.3f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Total P&L:   ${total_pnl:+,.2f}")

    # Blocking breakdown
    print(f"\n  --- Blocking Breakdown ---")
    print(f"  Pyramid blocked:  {results['pyramid_blocked']}")
    print(f"  CHOP blocked:     {len(results['chop_blocked'])}")
    print(f"  Margin blocked:   {results['margin_blocked']}")
    blocked_total = results['pyramid_blocked'] + len(results['chop_blocked']) + results['margin_blocked']
    print(f"  Total blocked:    {blocked_total}")
    print(f"  Entered + Blocked = {n + blocked_total} (total trades: {total_trades})")

    # Monthly breakdown
    print(f"\n  --- Monthly Breakdown ---")
    print(f"  {'Month':<10} {'Trades':>7} {'WR':>6} {'Avg R':>7} {'PF':>6} {'P&L':>11} {'Balance':>11}")
    print(f"  {'─' * 60}")

    months = sorted(set(t['exit_time'].strftime('%Y-%m') for t in trades))
    running_balance = STARTING_BALANCE
    for month in months:
        mt = [t for t in trades if t['exit_time'].strftime('%Y-%m') == month]
        if not mt:
            continue
        m_wins = sum(1 for t in mt if t['r_result'] > 0)
        m_wr = m_wins / len(mt) * 100
        m_avg_r = np.mean([t['r_result'] for t in mt])
        m_pnl = sum(t['pnl'] for t in mt)
        m_gp = sum(t['pnl'] for t in mt if t['pnl'] > 0)
        m_gl = abs(sum(t['pnl'] for t in mt if t['pnl'] < 0))
        m_pf = m_gp / m_gl if m_gl > 0 else float('inf')
        running_balance += m_pnl
        pf_str = f"{m_pf:.2f}" if m_pf < 100 else "inf"
        print(f"  {month:<10} {len(mt):>7} {m_wr:>5.1f}% {m_avg_r:>+7.3f} {pf_str:>6} ${m_pnl:>+9.2f} ${running_balance:>9,.2f}")

    # CHOP shadow tracking
    chop_b = results['chop_blocked']
    if chop_b:
        print(f"\n  --- CHOP Shadow Tracking (what blocked trades would have made) ---")
        cb_wins = sum(1 for t in chop_b if t['r_result'] > 0)
        cb_wr = cb_wins / len(chop_b) * 100
        cb_avg_r = np.mean([t['r_result'] for t in chop_b])
        print(f"  Blocked: {len(chop_b)} trades, WR: {cb_wr:.1f}%, Avg R: {cb_avg_r:+.3f}")
        print(f"  Filtering was {'BENEFICIAL' if cb_avg_r < avg_r else 'HARMFUL'} (blocked avg R {cb_avg_r:+.3f} vs entered avg R {avg_r:+.3f})")

    # Regime transitions (production only)
    if results['regime_transitions']:
        print(f"\n  --- Regime Transitions (first 30) ---")
        print(f"  {'Time':<20} {'From':<12} {'To':<12} {'Mult':>5} {'Closed':>7} {'Balance':>10}")
        print(f"  {'─' * 68}")
        for rt in results['regime_transitions'][:30]:
            fr = rt['from'] or 'START'
            print(f"  {str(rt['time']):<20} {fr:<12} {rt['to']:<12} {rt['mult']:>5.2f} {rt['n_closed']:>7} ${rt['balance']:>9,.2f}")
        if len(results['regime_transitions']) > 30:
            print(f"  ... {len(results['regime_transitions']) - 30} more transitions")


def print_head_to_head(all_results):
    """Side-by-side comparison of all scenarios."""
    print(f"\n{'=' * 70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 70}")

    header = f"  {'Metric':<20}"
    for r in all_results:
        sc = r['scenario'].upper().replace('_', ' ')
        header += f" {sc:>16}"
    print(header)
    print(f"  {'─' * (20 + 17 * len(all_results))}")

    def row(label, values, fmt='s'):
        line = f"  {label:<20}"
        for v in values:
            if fmt == '$':
                line += f" ${v:>14,.2f}"
            elif fmt == '%':
                line += f" {v:>15.1f}%"
            elif fmt == 'i':
                line += f" {v:>16,}"
            elif fmt == 'f':
                line += f" {v:>16.2f}"
            elif fmt == 'r':
                line += f" {v:>+15.3f}"
            else:
                line += f" {str(v):>16}"
        print(line)

    row('Final Balance', [r['wallet_balance'] for r in all_results], '$')
    row('Withdrawn', [r['total_withdrawn'] for r in all_results], '$')
    row('Effective', [r['final_effective'] for r in all_results], '$')
    row('Return %', [(r['final_effective'] - STARTING_BALANCE) / STARTING_BALANCE * 100 for r in all_results], '%')
    row('Max DD %', [r['max_dd_pct'] for r in all_results], '%')

    # Return/DD
    rdd = []
    for r in all_results:
        ret = (r['final_effective'] - STARTING_BALANCE) / STARTING_BALANCE * 100
        rdd.append(ret / r['max_dd_pct'] if r['max_dd_pct'] > 0 else 0)
    row('Return/DD', rdd, 'f')

    row('Trades Entered', [len(r['entered_trades']) for r in all_results], 'i')

    wrs = []
    for r in all_results:
        t = r['entered_trades']
        wrs.append(sum(1 for x in t if x['r_result'] > 0) / len(t) * 100 if t else 0)
    row('Win Rate %', wrs, '%')

    row('Avg R', [np.mean([t['r_result'] for t in r['entered_trades']]) if r['entered_trades'] else 0 for r in all_results], 'r')

    pfs = []
    for r in all_results:
        gp = sum(t['pnl'] for t in r['entered_trades'] if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in r['entered_trades'] if t['pnl'] < 0))
        pfs.append(gp / gl if gl > 0 else 0)
    row('Profit Factor', pfs, 'f')

    row('Pyramid Blocked', [r['pyramid_blocked'] for r in all_results], 'i')
    row('CHOP Blocked', [len(r['chop_blocked']) for r in all_results], 'i')
    row('Margin Blocked', [r['margin_blocked'] for r in all_results], 'i')


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def sanity_check(results, total_trades):
    """Verify accounting identity."""
    trades = results['entered_trades']
    sc = results['scenario']

    sum_pnl = sum(t['pnl'] for t in trades)
    expected = results['wallet_balance'] + results['total_withdrawn'] - STARTING_BALANCE
    diff = abs(sum_pnl - expected)

    entered = len(trades)
    blocked = results['pyramid_blocked'] + len(results['chop_blocked']) + results['margin_blocked']
    # Note: some trades may be skipped due to risk_usd < 0.01 or sl_distance <= 0,
    # so entered + blocked may be slightly less than total_trades

    print(f"\n  --- Sanity Check: {sc.upper()} ---")
    print(f"  sum(pnl) = ${sum_pnl:+,.2f}")
    print(f"  final - start + withdrawn = ${expected:+,.2f}")
    print(f"  Difference: ${diff:.4f} {'OK' if diff < 0.01 else 'MISMATCH!'}")
    print(f"  Entered ({entered}) + Blocked ({blocked}) = {entered + blocked} / {total_trades} total")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PRODUCTION-CORRECT BACKTEST")
    print(f"  Starting balance: ${STARTING_BALANCE:,.0f}")
    print("  Exact bot formulas: qty=risk/sl_dist, margin=pos_value/leverage")
    print("=" * 70)

    # Load trades
    df = pd.read_csv(TRADES_CSV, parse_dates=['entry_time', 'exit_time'])
    print(f"\n  Loaded {len(df)} trades from {df['entry_time'].min()} to {df['entry_time'].max()}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    total_trades = len(df)

    # Sort by entry time
    df = df.sort_values('entry_time').reset_index(drop=True)

    # Load CHOP data
    symbols = df['symbol'].unique().tolist()
    chop_map = load_chop_data(symbols)

    # Run 3 scenarios
    print(f"\n  Running PRODUCTION scenario...")
    r_prod = run_simulation(df, chop_map, scenario='production')

    print(f"  Running NO REGIME scenario...")
    r_noreg = run_simulation(df, chop_map, scenario='no_regime')

    print(f"  Running NO FILTERS scenario...")
    r_nofilt = run_simulation(df, chop_map, scenario='no_filters')

    all_results = [r_prod, r_noreg, r_nofilt]

    # Print head-to-head
    print_head_to_head(all_results)

    # Print detailed results for each
    for r in all_results:
        print_results(r, total_trades)

    # Sanity checks
    for r in all_results:
        sanity_check(r, total_trades)

    # Regime distribution for production
    print(f"\n{'=' * 70}")
    print(f"  REGIME DISTRIBUTION (PRODUCTION)")
    print(f"{'=' * 70}")
    regime_counts = defaultdict(int)
    for t in r_prod['entered_trades']:
        regime_counts[t['regime']] += 1
    for regime in ['favorable', 'cautious', 'adverse', 'critical']:
        cnt = regime_counts.get(regime, 0)
        pct = cnt / len(r_prod['entered_trades']) * 100 if r_prod['entered_trades'] else 0
        print(f"  {regime:<12} {cnt:>6} trades ({pct:>5.1f}%)")

    # Average risk USD per regime
    print(f"\n  Average risk USD by regime:")
    for regime in ['favorable', 'cautious', 'adverse', 'critical']:
        rt = [t for t in r_prod['entered_trades'] if t['regime'] == regime]
        if rt:
            avg_risk = np.mean([t['risk_usd'] for t in rt])
            print(f"  {regime:<12} avg risk: ${avg_risk:>8.2f}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
