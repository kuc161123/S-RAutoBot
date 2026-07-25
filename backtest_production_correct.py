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

def get_tapered_risk(wallet_balance, base_risk=None, schedule=None):
    """Balance-based taper schedule.

    Optional overrides allow risk experiments without mutating module-level
    constants. Defaults preserve original behaviour.
    """
    base = BASE_RISK if base_risk is None else base_risk
    sched = TAPER_SCHEDULE if schedule is None else schedule
    for threshold, risk in sched:
        if wallet_balance >= threshold:
            base = risk
    return base


def apply_dd_taper(risk_pct, current_dd_pct, dd_taper):
    """Apply optional drawdown-based taper that shrinks risk in deep DD.

    dd_taper: None or list of (dd_threshold_pct, multiplier) tuples, in
    increasing dd order. Multipliers compound through thresholds crossed —
    so [(30, 0.5), (50, 0.5)] means risk *= 0.5 once DD>=30%, and *0.25 once
    DD>=50%.
    """
    if not dd_taper:
        return risk_pct
    out = risk_pct
    for thresh, mult in dd_taper:
        if current_dd_pct >= thresh:
            out *= mult
    return out


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


def load_btc_trend():
    """Build an hourly BTC market-state frame for the portfolio risk-off filter.

    Returns a DataFrame indexed by hour with columns:
      'slope'   — 7-day slope of the daily SMA50 close (fractional), forward-
                  filled to every hour. <0 means BTC downtrend.
      'atr_pct' — 1H ATR(14) as a fraction of close (volatility), per hour.
    Mirrors the BTC-regime approach in verify_overfit.py:668-687.
    """
    fpath = CACHE_DIR / 'BTCUSDT.parquet'
    if not fpath.exists():
        return None
    btc = pd.read_parquet(fpath).copy()
    btc['start'] = pd.to_datetime(btc['start'])
    btc = btc.set_index('start').sort_index()

    # 1H ATR% (volatility)
    hl = btc['high'] - btc['low']
    hc = (btc['high'] - btc['close'].shift()).abs()
    lc = (btc['low'] - btc['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = atr / btc['close']

    # Daily SMA50 trend slope (7-day), forward-filled to hourly
    daily = btc['close'].resample('1D').last().to_frame('close')
    daily['sma50'] = daily['close'].rolling(50).mean()
    daily['slope'] = daily['sma50'].diff(7) / daily['sma50']
    slope_hourly = daily['slope'].reindex(btc.index, method='ffill')

    out = pd.DataFrame({'slope': slope_hourly, 'atr_pct': atr_pct}, index=btc.index)
    return out


def lookup_btc(btc_map, entry_time):
    """Return (slope, atr_pct) for BTC at entry_time (nearest 1H candle <= time)."""
    if btc_map is None or len(btc_map) == 0:
        return None, None
    ts = pd.Timestamp(entry_time).floor('h')
    if ts in btc_map.index:
        row = btc_map.loc[ts]
    else:
        mask = btc_map.index <= ts
        if not mask.any():
            return None, None
        row = btc_map.loc[mask].iloc[-1]
    slope = row['slope'] if pd.notna(row['slope']) else None
    atr_pct = row['atr_pct'] if pd.notna(row['atr_pct']) else None
    return slope, atr_pct


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT-DRIVEN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(trades_df, chop_map, scenario='production',
                  base_risk=None, dd_taper=None, starting_balance=None,
                  max_concurrent=None,
                  custom_taper=None, smooth_taper=None, concurrent_taper=None,
                  taper_basis='wallet', size_basis='wallet',
                  open_risk_cap=None, open_risk_mode='block',
                  daily_halt_r=None, net_dir_cap=None, net_dir_cap_min_balance=0.0,
                  btc_risk_off=None, btc_map=None, trade_risk_mult_col=None,
                  btc_bull_col=None, btc_short_col=None, overlay_min_balance=0.0,
                  short_gate=False, long_boost=1.0,
                  liq_turnover_col=None, liq_impact_k=0.0, liq_impact_cap=0.01,
                  liq_skip_frac=0.0, stop_gap_pct=0.0):
    """
    Run event-driven backtest simulation.

    base_risk: optional override for BASE_RISK (e.g. 0.008 for 0.8%/trade).
    dd_taper:  optional list of (dd_threshold_pct, multiplier) tuples; applied
               after the balance-based taper to shrink risk on drawdown.
    starting_balance: optional override for STARTING_BALANCE.

    taper_basis: which balance selects the taper rung — 'wallet' (closed-only,
                 the live + historical default) or 'equity' (closed + unrealized
                 PnL of open positions). Equity is mark-to-market-approximated
                 by linear time-interpolation of each open position's final PnL
                 (no tick data in the trade CSV — assumes ~linear PnL accrual).
    size_basis:  which balance the per-trade dollar risk is multiplied against —
                 'wallet' or 'equity' (same proxy). The live bot today is the
                 mismatched pair taper_basis='wallet', size_basis='equity'.
                 Defaults ('wallet','wallet') reproduce the original engine
                 exactly so existing callers are unaffected.

    DRAWDOWN-REDUCTION OVERLAYS (all default None -> no behaviour change):
    open_risk_cap: cap aggregate simultaneous open risk. A new entry is blocked
                 (open_risk_mode='block') or shrunk to fit ('scale') when
                 Sum(open risk_usd) + new_risk_usd > open_risk_cap * equity_now.
                 Directly bounds "if every open SL hits at once, max loss".
    daily_halt_r: halt NEW entries for the rest of a calendar day once realized
                 PnL that day <= daily_halt_r (in R, e.g. -5.0).
    net_dir_cap: cap |long open risk - short open risk| <= net_dir_cap*equity_now
                 (limits net directional/market-beta exposure; hedged books OK).
    btc_risk_off: dict gating new entries on BTC trend/vol. Keys:
                 'slope_block' (skip if BTC SMA50 7d slope <= this, e.g. -0.01),
                 'vol_block'   (skip if BTC ATR%% >= this), 'mult' (if set, scale
                 risk by mult instead of skipping). Requires btc_map.
    btc_map: output of load_btc_trend(); DataFrame indexed by hour with columns
                 'slope' and 'atr_pct'. Aligned to entry_time (floor-to-hour).

    scenario:
      'production' — regime + CHOP + taper + anti-pyramid + correct margin
      'no_regime'  — flat 1.2% risk, CHOP + anti-pyramid + correct margin
      'no_filters' — flat 1.2% risk, no CHOP, anti-pyramid + correct margin only
    """
    wallet_balance = STARTING_BALANCE if starting_balance is None else starting_balance
    total_withdrawn = 0.0
    margin_used = 0.0
    open_positions = {}  # key: "SYMBOL_side" -> position dict
    recent_closed = []   # regime window: list of {'r': float}
    entered_trades = []
    regime_transitions = []

    # Aggregate open-risk tracking (for open_risk_cap / net_dir_cap overlays)
    open_risk_sum = 0.0      # sum of risk_usd across currently-open positions
    open_long_risk = 0.0     # sum of risk_usd for open longs
    open_short_risk = 0.0    # sum of risk_usd for open shorts

    # Daily realized PnL (for daily_halt_r circuit breaker), keyed by calendar day
    daily_pnl = defaultdict(float)

    # Blocking counters
    pyramid_blocked = 0
    chop_blocked_list = []  # shadow-track outcomes
    margin_blocked = 0
    liquidity_skipped = 0
    risk_capped = 0          # blocked by open_risk_cap / net_dir_cap
    daily_halted = 0         # blocked by daily_halt_r
    btc_blocked = 0          # blocked/scaled by btc_risk_off
    regime_info_log = []

    # Track peak for drawdown. Seed from the RESOLVED starting balance, not the module
    # constant — a caller passing starting_balance= without also mutating the global
    # would otherwise get a wrong max_dd_pct.
    peak_balance = wallet_balance
    max_dd_pct = 0.0

    # Mark-to-market drawdown (on equity incl. unrealized) — the lived DD these
    # overlays actually target. Sampled at every event.
    peak_equity_mtm = wallet_balance
    max_dd_mtm_pct = 0.0

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
            # Release aggregate open-risk
            open_risk_sum -= pos['risk_usd']
            if pos['side'] == 'long':
                open_long_risk -= pos['risk_usd']
            else:
                open_short_risk -= pos['risk_usd']

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

            # Track daily realized PnL (for daily_halt_r), keyed by exit calendar day
            daily_pnl[pos['exit_time'].strftime('%Y-%m-%d')] += pnl

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

        # ─── STEP B2: Concurrent-position cap (optional, anti-DD lever) ───
        if max_concurrent is not None and len(open_positions) >= max_concurrent:
            margin_blocked += 1  # reuse counter so head-to-head shows in same column
            continue

        # ─── STEP C: Regime ───
        # regime_label is ALWAYS computed (the CHOP gate is regime-aware); only whether
        # it scales position size depends on the scenario. This lets CHOP and regime
        # sizing be ablated independently:
        #   production        regime sizing + regime-aware CHOP   (live bot)
        #   production_nochop regime sizing, no CHOP
        #   chop_only         flat risk,     regime-aware CHOP
        #   no_regime         flat risk,     fixed CHOP 55
        #   no_filters        flat risk,     no CHOP
        _regime_sizes = scenario in ('production', 'production_nochop')
        regime_label, regime_mult = get_regime(recent_closed)
        if not _regime_sizes:
            regime_mult = 1.0

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
        if scenario in ('production', 'chop_only'):
            chop_thresh = CHOP_THRESHOLDS.get(regime_label)  # favorable=None (never blocked)
        elif scenario == 'no_regime':
            chop_thresh = 55  # fixed cautious-level threshold regardless of regime
        else:
            chop_thresh = None  # no_filters / production_nochop: no CHOP at all

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

        # ─── STEP D2: BTC portfolio risk-off filter (optional) ───
        # Stand aside / shrink risk when the whole market (BTC) is dumping or
        # volatility is spiking — directly targets correlated-crash drawdown.
        btc_risk_mult = 1.0
        if btc_risk_off and btc_map is not None:
            slope, atr_pct = lookup_btc(btc_map, entry_time)
            trig = False
            if slope is not None and btc_risk_off.get('slope_block') is not None \
                    and slope <= btc_risk_off['slope_block']:
                trig = True
            if atr_pct is not None and btc_risk_off.get('vol_block') is not None \
                    and atr_pct >= btc_risk_off['vol_block']:
                trig = True
            if trig:
                if btc_risk_off.get('mult') is not None:
                    btc_risk_mult = btc_risk_off['mult']  # scale instead of skip
                else:
                    btc_blocked += 1
                    continue

        # ─── STEP E0: Mark-to-market equity (proxy) for taper/size/overlays ───
        # equity = closed wallet + unrealized PnL of currently-open positions,
        # where each open position's unrealized PnL is linearly interpolated by
        # time-elapsed fraction toward its final PnL (no tick data available).
        need_equity = (taper_basis == 'equity' or size_basis == 'equity'
                       or open_risk_cap is not None or net_dir_cap is not None)
        if need_equity:
            unrealized = 0.0
            for pos in open_positions.values():
                span = (pos['exit_time'] - pos['entry_time']).total_seconds()
                if span <= 0:
                    frac = 1.0
                else:
                    frac = (entry_time - pos['entry_time']).total_seconds() / span
                    frac = min(1.0, max(0.0, frac))
                unrealized += pos['pnl'] * frac
            equity_now = wallet_balance + unrealized
        else:
            equity_now = wallet_balance

        # Mark-to-market drawdown tracking (the lived DD the overlays target)
        eff_equity_mtm = equity_now + total_withdrawn
        if eff_equity_mtm > peak_equity_mtm:
            peak_equity_mtm = eff_equity_mtm
        if peak_equity_mtm > 0:
            dd_mtm = (peak_equity_mtm - eff_equity_mtm) / peak_equity_mtm * 100
            if dd_mtm > max_dd_mtm_pct:
                max_dd_mtm_pct = dd_mtm

        taper_input = equity_now if taper_basis == 'equity' else wallet_balance
        size_input = equity_now if size_basis == 'equity' else wallet_balance

        # ─── BTC SHORT-GATE (balance-conditional) ───
        # Mirror the live bot's btc_short_gate, but only ACTIVE once equity reaches
        # overlay_min_balance (the "ramp"). Below that, shorts are allowed even in a
        # BTC uptrend (max-growth phase). btc_bull_col holds per-trade BTC>200EMA.
        # Live uses DIFFERENT BTC signals for the two overlays: the short-gate is v2
        # (trailing 30d daily return > +10%, ~1.5 flips/mo) while the long-boost kept the
        # 1H EMA200 trigger. btc_short_col lets the sim mirror that; it falls back to
        # btc_bull_col so existing callers are unaffected.
        _short_col = btc_short_col if btc_short_col is not None else btc_bull_col
        if (short_gate and _short_col is not None and side == 'short'
                and equity_now >= overlay_min_balance and bool(trade.get(_short_col, False))):
            chop_blocked_list.append({'symbol': symbol, 'side': side,
                                      'entry_time': entry_time, 'r_result': r_result,
                                      'regime': 'shortgate', 'chop_val': 0.0, 'chop_thresh': 0.0})
            continue

        # ─── STEP E: Position sizing (THE CRITICAL FIX) ───
        # Taper and the regime MULTIPLIER are separate knobs and must be ablated
        # separately. 'chop_only' exists to remove ONLY the regime multiplier, so it
        # keeps the balance taper (otherwise that row silently ablates taper+regime
        # together and is not comparable with the other one-at-a-time rows).
        # 'no_regime'/'no_filters' keep their historical flat-risk behaviour so existing
        # callers are bit-for-bit unaffected.
        if _regime_sizes or scenario == 'chop_only':
            # Determine raw (pre-regime) risk via the chosen taper form.
            if smooth_taper:
                base = smooth_taper.get('base', 0.012)
                half_at = smooth_taper.get('half_at', 3000.0)
                floor = smooth_taper.get('min', 0.0025)
                # smooth log decay: risk approaches floor as balance grows
                tapered = max(floor, base / (1.0 + max(0.0, taper_input) / half_at))
            elif custom_taper:
                base = base_risk if base_risk is not None else BASE_RISK
                tapered = base
                for threshold, risk in custom_taper:
                    if taper_input >= threshold:
                        tapered = risk
            else:
                tapered = get_tapered_risk(taper_input, base_risk=base_risk)
            # regime_mult was already forced to 1.0 above for non-regime-sizing scenarios
            risk_pct = tapered * regime_mult
        else:
            risk_pct = BASE_RISK if base_risk is None else base_risk

        # Optional drawdown-based taper
        if dd_taper:
            effective_bal = wallet_balance + total_withdrawn
            if peak_balance > 0:
                cur_dd_pct = max(0.0, (peak_balance - effective_bal) / peak_balance * 100)
                risk_pct = apply_dd_taper(risk_pct, cur_dd_pct, dd_taper)

        # Optional concurrent-position-aware taper: shrink per-trade risk when
        # many positions are already open. This BOTH reduces concentration risk
        # and ensures we never margin-block (smaller risk -> smaller notional).
        if concurrent_taper:
            n_open = len(open_positions)
            mult = 1.0
            for thresh, m in concurrent_taper:
                if n_open >= thresh:
                    mult = m
            risk_pct *= mult

        # BTC risk-off scaling (if configured to shrink rather than skip)
        risk_pct *= btc_risk_mult

        # Optional per-trade risk multiplier from a universe column (e.g. boost long
        # risk in confirmed BTC-bull). Backward-compatible: default None -> no change.
        if trade_risk_mult_col is not None:
            risk_pct *= float(trade.get(trade_risk_mult_col, 1.0) or 1.0)

        # ─── BULL LONG-BOOST (balance-conditional) ───
        # Mirror the live bot's long_bull_boost, ACTIVE only once equity reaches
        # overlay_min_balance. Boost long risk in a BTC uptrend (the profitable side).
        if (long_boost != 1.0 and btc_bull_col is not None and side == 'long'
                and equity_now >= overlay_min_balance and bool(trade.get(btc_bull_col, False))):
            risk_pct *= long_boost

        risk_usd = size_input * risk_pct
        if risk_usd < 0.01:
            continue

        # ─── STEP E1: Daily-loss circuit breaker (optional) ───
        # Halt NEW entries for the rest of a calendar day once realized PnL that
        # day <= daily_halt_r (expressed in R, using current per-trade risk_usd).
        if daily_halt_r is not None and risk_usd > 0:
            day_key = entry_time.strftime('%Y-%m-%d')
            if daily_pnl[day_key] / risk_usd <= daily_halt_r:
                daily_halted += 1
                continue

        # ─── STEP E2: Aggregate open-risk budget cap (optional, NOVEL) ───
        # Bound simultaneous downside: if every open SL hit at once, the loss is
        # ~Sum(open risk_usd). Cap that sum to open_risk_cap * equity.
        if open_risk_cap is not None:
            budget = open_risk_cap * equity_now
            room = budget - open_risk_sum
            if room <= 0:
                risk_capped += 1
                continue
            if risk_usd > room:
                if open_risk_mode == 'scale':
                    risk_usd = room  # shrink to fit the remaining budget
                else:
                    risk_capped += 1
                    continue

        # ─── STEP E3: Net-directional exposure cap (optional) ───
        # Limit |long open risk - short open risk| (net market beta). A balanced
        # (hedged) book can stay large; a one-sided book is throttled.
        if net_dir_cap is not None and equity_now >= net_dir_cap_min_balance:
            proj_long = open_long_risk + (risk_usd if side == 'long' else 0.0)
            proj_short = open_short_risk + (risk_usd if side == 'short' else 0.0)
            if abs(proj_long - proj_short) > net_dir_cap * equity_now:
                risk_capped += 1
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

        # ─── REALISM LAYER (all default-off) ───
        # Size/liquidity market impact via the square-root law (Almgren et al.):
        #   impact_frac = k * sqrt(notional / daily_volume)
        # Grows with position size relative to the symbol's *daily* turnover; applied
        # both entry and exit. This is what makes large balances on thin alts cost
        # more — the dominant source of optimism at size.
        if liq_turnover_col is not None and (liq_impact_k > 0.0 or liq_skip_frac > 0.0):
            turn_h = float(trade.get(liq_turnover_col, 0.0) or 0.0)  # hourly turnover
            # Liquidity skip: a real fill can't take a large % of an hour's volume.
            if liq_skip_frac > 0.0:
                if turn_h <= 0 or position_value > liq_skip_frac * turn_h:
                    liquidity_skipped += 1
                    continue
            if liq_impact_k > 0.0:
                adv = turn_h * 24.0  # daily volume
                if adv > 0:
                    impact = min(liq_impact_cap, liq_impact_k * (position_value / adv) ** 0.5)
                else:
                    impact = liq_impact_cap
                trade_cost += position_value * impact * 2.0  # entry + exit

        # Stop gap-risk: losing trades fill WORSE than the -1R trigger in fast moves
        # (price gaps through the stop). Add extra adverse slippage on SL exits only.
        if stop_gap_pct > 0.0 and r_result < 0:
            trade_cost += position_value * stop_gap_pct

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
        # Track aggregate open-risk for the budget / net-directional caps
        open_risk_sum += risk_usd
        if side == 'long':
            open_long_risk += risk_usd
        else:
            open_short_risk += risk_usd

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
        'max_dd_mtm_pct': max_dd_mtm_pct,
        'entered_trades': entered_trades,
        'pyramid_blocked': pyramid_blocked,
        'chop_blocked': chop_blocked_list,
        'margin_blocked': margin_blocked,
        'liquidity_skipped': liquidity_skipped,
        'risk_capped': risk_capped,
        'daily_halted': daily_halted,
        'btc_blocked': btc_blocked,
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
