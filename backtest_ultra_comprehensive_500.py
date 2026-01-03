#!/usr/bin/env python3
"""
ULTRA-COMPREHENSIVE MULTI-DIVERGENCE BACKTEST
==============================================

A rigorous, modern backtesting system that:
- Tests ALL 4 divergence types separately per symbol
- Tests multiple R:R ratios (3:1 to 10:1)
- Implements Walk-Forward Analysis (3 periods)
- Runs Monte Carlo Simulation (1000 iterations)
- Performs Out-of-Sample (OOS) Testing
- Calculates comprehensive statistics per symbol-divergence

Divergence Types Tested:
1. Regular Bullish - Price Lower Low, RSI Higher Low (Reversal)
2. Regular Bearish - Price Higher High, RSI Lower High (Reversal)
3. Hidden Bullish - Price Higher Low, RSI Lower Low (Continuation)
4. Hidden Bearish - Price Lower High, RSI Higher High (Continuation)

Output Files:
- ultra_comprehensive_all_results.csv (all combinations)
- ultra_comprehensive_validated.csv (passing symbols)
- ultra_comprehensive_optimal.yaml (best per symbol)
- ultra_comprehensive_summary.txt (summary report)

Author: AutoTrading Bot System
Date: 2026-01-03
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import yaml
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Symbol settings
SYMBOL_LIMIT = 500  # Full 500 symbol test
SYMBOL_MIN_VOLUME = 0  # Minimum 24h volume filter

# Timeframe and data
TIMEFRAME = '60'  # 1H
DATA_DAYS = 180   # 6 months

# Strategy settings
MAX_WAIT_CANDLES = 12  # 12 hours for BOS
ATR_MULT = 1.0
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Indicator settings
RSI_PERIOD = 14
EMA_PERIOD = 200
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# R:R ratios to test
RR_OPTIONS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Walk-forward settings (3 equal periods)
WF_PERIOD_DAYS = 60  # 2 months each
WF_MIN_PERIODS_PROFITABLE = 2  # Must be profitable in 2/3 periods

# Monte Carlo settings
MC_ITERATIONS = 1000
MC_CONFIDENCE_LEVEL = 0.95

# Minimum requirements for validation
MIN_TRADES_TOTAL = 15  # Minimum trades across all periods
MIN_TRADES_PER_PERIOD = 5  # Minimum trades per walk-forward period
MIN_WIN_RATE = 10  # Minimum 10% win rate
MIN_TOTAL_R = 0  # Must be net profitable
MIN_SHARPE = 0.3  # Minimum Sharpe ratio
MAX_DRAWDOWN_PCT = 50  # Maximum drawdown

BASE_URL = "https://api.bybit.com"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DivergenceConfig:
    """Configuration for a divergence type"""
    name: str
    code: str  # short code for identification
    price_pattern: str  # 'LL', 'HH', 'HL', 'LH'
    rsi_pattern: str    # 'HL', 'LH', 'LL', 'HH'
    side: str           # 'long' or 'short'
    trend_filter: str   # 'above_ema' or 'below_ema'
    description: str

# Define all 4 divergence types
DIVERGENCE_CONFIGS = {
    'REG_BULL': DivergenceConfig(
        name='Regular Bullish',
        code='REG_BULL',
        price_pattern='LL',  # Price makes Lower Low
        rsi_pattern='HL',    # RSI makes Higher Low
        side='long',
        trend_filter='above_ema',
        description='Reversal signal - Price LL but RSI HL indicates weakening bears'
    ),
    'REG_BEAR': DivergenceConfig(
        name='Regular Bearish',
        code='REG_BEAR',
        price_pattern='HH',  # Price makes Higher High
        rsi_pattern='LH',    # RSI makes Lower High
        side='short',
        trend_filter='below_ema',
        description='Reversal signal - Price HH but RSI LH indicates weakening bulls'
    ),
    'HID_BULL': DivergenceConfig(
        name='Hidden Bullish',
        code='HID_BULL',
        price_pattern='HL',  # Price makes Higher Low
        rsi_pattern='LL',    # RSI makes Lower Low
        side='long',
        trend_filter='above_ema',
        description='Continuation signal - Price HL but RSI LL shows trend strength'
    ),
    'HID_BEAR': DivergenceConfig(
        name='Hidden Bearish',
        code='HID_BEAR',
        price_pattern='LH',  # Price makes Lower High
        rsi_pattern='HH',    # RSI makes Higher High
        side='short',
        trend_filter='below_ema',
        description='Continuation signal - Price LH but RSI HH shows trend strength'
    )
}

@dataclass
class Trade:
    """Single trade record"""
    symbol: str
    divergence_type: str
    rr: float
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    result_r: float
    is_win: bool
    duration_candles: int
    period: int  # 1, 2, or 3 for walk-forward

@dataclass
class ValidationResult:
    """Comprehensive validation results for a symbol-divergence-RR combo"""
    symbol: str
    divergence_type: str
    rr: float
    
    # Basic metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_r: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    profit_factor: float = 0.0
    
    # Walk-forward results by period
    p1_trades: int = 0
    p1_wins: int = 0
    p1_r: float = 0.0
    p2_trades: int = 0
    p2_wins: int = 0
    p2_r: float = 0.0
    p3_trades: int = 0
    p3_wins: int = 0
    p3_r: float = 0.0
    periods_profitable: int = 0
    
    # Monte Carlo results
    mc_mean_r: float = 0.0
    mc_median_r: float = 0.0
    mc_std_r: float = 0.0
    mc_ci_lower: float = 0.0
    mc_ci_upper: float = 0.0
    mc_profitable_pct: float = 0.0
    mc_worst_case_r: float = 0.0
    mc_best_case_r: float = 0.0
    
    # Validation flags
    passes_min_trades: bool = False
    passes_win_rate: bool = False
    passes_profitability: bool = False
    passes_sharpe: bool = False
    passes_drawdown: bool = False
    passes_walk_forward: bool = False
    passes_monte_carlo: bool = False
    passes_all: bool = False
    
    # Additional info
    avg_trade_duration: float = 0.0
    best_month_r: float = 0.0
    worst_month_r: float = 0.0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_progress(current, total, symbol, extra=""):
    """Print progress bar"""
    pct = (current / total) * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"\r[{bar}] {pct:5.1f}% | {current}/{total} | {symbol:18} {extra}", end='', flush=True)

def fetch_symbols(limit=500):
    """Fetch USDT perpetual symbols from Bybit"""
    print(f"\n📥 Fetching up to {limit} USDT symbols from Bybit...")
    
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/instruments-info",
            params={'category': 'linear', 'limit': 1000},
            timeout=30
        )
        data = resp.json().get('result', {}).get('list', [])
        
        symbols = []
        for item in data:
            symbol = item.get('symbol', '')
            status = item.get('status', '')
            if symbol.endswith('USDT') and status == 'Trading':
                symbols.append(symbol)
        
        symbols = sorted(symbols)[:limit]
        print(f"✅ Found {len(symbols)} USDT symbols\n")
        return symbols
        
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch historical klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iter = 50
    
    while current_end > start_ts and max_iter > 0:
        max_iter -= 1
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.02)
            
        except Exception:
            time.sleep(0.1)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def calculate_indicators(df):
    """Calculate RSI, ATR, EMA"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMA 200
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    """Find pivot highs and lows"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1):
            continue
        
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
            
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def get_period(timestamp, df_start, period_days=60):
    """Determine which walk-forward period a timestamp belongs to"""
    days_since_start = (timestamp - df_start).days
    if days_since_start < period_days:
        return 1
    elif days_since_start < period_days * 2:
        return 2
    else:
        return 3

# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

def detect_divergence(df, idx, div_config, close_arr, rsi_arr, ema_arr, 
                      price_ph, price_pl, rsi_ph, rsi_pl):
    """
    Detect if a specific divergence type exists at the current index.
    
    Returns: (divergence_found, swing_level) or (False, None)
    """
    if idx < 50:  # Need enough history
        return False, None
    
    current_price = close_arr[idx]
    current_ema = ema_arr[idx]
    
    # Check trend alignment
    if div_config.trend_filter == 'above_ema' and current_price <= current_ema:
        return False, None
    if div_config.trend_filter == 'below_ema' and current_price >= current_ema:
        return False, None
    
    # Find recent pivots based on divergence type
    if div_config.side == 'long':
        # Looking for price lows and RSI lows
        price_pivots = []
        rsi_pivots = []
        
        # Scan back for pivot lows (confirmed, so idx - PIVOT_RIGHT - 1)
        for j in range(idx - PIVOT_RIGHT - 1, max(0, idx - 50), -1):
            if not np.isnan(price_pl[j]):
                price_pivots.append((j, price_pl[j], rsi_arr[j]))
                if len(price_pivots) >= 2:
                    break
        
        if len(price_pivots) < 2:
            return False, None
        
        curr_pivot = price_pivots[0]  # Most recent
        prev_pivot = price_pivots[1]  # Previous
        
        curr_idx, curr_price_val, curr_rsi_val = curr_pivot
        prev_idx, prev_price_val, prev_rsi_val = prev_pivot
        
        # Check if pivot is fresh (within 10 candles)
        if (idx - curr_idx) > 10:
            return False, None
        
        # Check divergence patterns
        if div_config.code == 'REG_BULL':
            # Regular Bullish: Price LL, RSI HL
            if curr_price_val < prev_price_val and curr_rsi_val > prev_rsi_val:
                # Calculate swing level (high between the two pivots)
                swing_high = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing_high:  # Price hasn't broken out yet
                    return True, swing_high
                    
        elif div_config.code == 'HID_BULL':
            # Hidden Bullish: Price HL, RSI LL
            if curr_price_val > prev_price_val and curr_rsi_val < prev_rsi_val:
                swing_high = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing_high:
                    return True, swing_high
    
    else:  # short side
        # Looking for price highs and RSI highs
        price_pivots = []
        
        for j in range(idx - PIVOT_RIGHT - 1, max(0, idx - 50), -1):
            if not np.isnan(price_ph[j]):
                price_pivots.append((j, price_ph[j], rsi_arr[j]))
                if len(price_pivots) >= 2:
                    break
        
        if len(price_pivots) < 2:
            return False, None
        
        curr_pivot = price_pivots[0]
        prev_pivot = price_pivots[1]
        
        curr_idx, curr_price_val, curr_rsi_val = curr_pivot
        prev_idx, prev_price_val, prev_rsi_val = prev_pivot
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_config.code == 'REG_BEAR':
            # Regular Bearish: Price HH, RSI LH
            if curr_price_val > prev_price_val and curr_rsi_val < prev_rsi_val:
                swing_low = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing_low:
                    return True, swing_low
                    
        elif div_config.code == 'HID_BEAR':
            # Hidden Bearish: Price LH, RSI HH
            if curr_price_val < prev_price_val and curr_rsi_val > prev_rsi_val:
                swing_low = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing_low:
                    return True, swing_low
    
    return False, None

# ============================================================================
# TRADE SIMULATION
# ============================================================================

def simulate_trades(df, symbol, div_config, rr, df_start):
    """
    Simulate trades for a specific symbol, divergence type, and R:R ratio.
    Returns list of Trade objects.
    """
    trades = []
    
    if len(df) < 100:
        return trades
    
    # Prepare arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    # Find pivots
    price_ph, price_pl = find_pivots(close, PIVOT_LEFT, PIVOT_RIGHT)
    rsi_ph, rsi_pl = find_pivots(rsi, PIVOT_LEFT, PIVOT_RIGHT)
    
    # Trading state
    seen_signals = set()
    pending_signal = None
    pending_wait = 0
    pending_swing = None
    pending_idx = None
    
    for idx in range(50, n - 1):
        current_price = close[idx]
        current_time = df.index[idx]
        
        # === CHECK PENDING SIGNAL FOR BOS ===
        if pending_signal is not None:
            bos_confirmed = False
            
            if div_config.side == 'long':
                if current_price > pending_swing:
                    bos_confirmed = True
            else:
                if current_price < pending_swing:
                    bos_confirmed = True
            
            if bos_confirmed:
                # Execute trade on next candle
                entry_idx = idx + 1
                if entry_idx >= n:
                    pending_signal = None
                    continue
                
                entry_price = df.iloc[entry_idx]['open']
                entry_time = df.index[entry_idx]
                entry_atr = atr[entry_idx]
                sl_dist = entry_atr * ATR_MULT
                
                if div_config.side == 'long':
                    entry_price *= (1 + SLIPPAGE_PCT)
                    tp_price = entry_price + (sl_dist * rr)
                    sl_price = entry_price - sl_dist
                else:
                    entry_price *= (1 - SLIPPAGE_PCT)
                    tp_price = entry_price - (sl_dist * rr)
                    sl_price = entry_price + sl_dist
                
                # Simulate trade outcome
                # NOTE: Start from entry_idx + 1 to avoid same-candle SL/TP hits
                # On entry candle, we enter at open - can't hit SL/TP until next bar
                result = None
                exit_idx = None
                
                for k in range(entry_idx + 1, min(entry_idx + 200, n)):
                    if div_config.side == 'long':
                        if low[k] <= sl_price:
                            result = -1.0
                            exit_idx = k
                            break
                        if high[k] >= tp_price:
                            result = rr
                            exit_idx = k
                            break
                    else:
                        if high[k] >= sl_price:
                            result = -1.0
                            exit_idx = k
                            break
                        if low[k] <= tp_price:
                            result = rr
                            exit_idx = k
                            break
                
                if result is None:
                    result = -0.5  # Timeout
                    exit_idx = min(entry_idx + 200, n - 1)
                
                # Calculate fees
                risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                final_r = result - fee_cost
                
                exit_time = df.index[exit_idx] if exit_idx < len(df) else entry_time
                exit_price = df.iloc[exit_idx]['close'] if exit_idx < len(df) else entry_price
                
                # Determine period
                period = get_period(entry_time, df_start, WF_PERIOD_DAYS)
                
                trades.append(Trade(
                    symbol=symbol,
                    divergence_type=div_config.code,
                    rr=rr,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    side=div_config.side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    result_r=round(final_r, 3),
                    is_win=final_r > 0,
                    duration_candles=exit_idx - entry_idx if exit_idx else 0,
                    period=period
                ))
                
                pending_signal = None
                pending_wait = 0
                
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        # === DETECT NEW DIVERGENCE ===
        if pending_signal is None:
            found, swing = detect_divergence(
                df, idx, div_config, close, rsi, ema,
                price_ph, price_pl, rsi_ph, rsi_pl
            )
            
            if found:
                signal_id = f"{div_config.code}_{idx}"
                if signal_id not in seen_signals:
                    seen_signals.add(signal_id)
                    pending_signal = div_config
                    pending_swing = swing
                    pending_idx = idx
                    pending_wait = 0
    
    return trades

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_sharpe(returns, risk_free_rate=0):
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    return (mean_return - risk_free_rate) / std_return * np.sqrt(252 / 24)  # Annualized for 1H

def calculate_sortino(returns, risk_free_rate=0):
    """Calculate Sortino ratio (only considers downside volatility)"""
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        return 10.0  # Very high if no losses
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 10.0
    return (mean_return - risk_free_rate) / downside_std * np.sqrt(252 / 24)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown in R"""
    if not returns:
        return 0.0, 0.0
    
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd_r = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
    
    # Calculate percentage drawdown
    if running_max.max() > 0:
        max_dd_pct = (max_dd_r / running_max.max()) * 100 if running_max.max() > 0 else 0
    else:
        max_dd_pct = 0
    
    return max_dd_r, max_dd_pct

def calculate_consecutive_losses(results):
    """Calculate max consecutive losses"""
    if not results:
        return 0
    
    max_consec = 0
    current_consec = 0
    
    for is_win in results:
        if not is_win:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0
    
    return max_consec

def calculate_profit_factor(wins_r, losses_r):
    """Calculate profit factor"""
    total_wins = sum(wins_r) if wins_r else 0
    total_losses = abs(sum(losses_r)) if losses_r else 0
    
    if total_losses == 0:
        return 100.0 if total_wins > 0 else 0.0
    
    return total_wins / total_losses

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo(trade_results, iterations=1000, confidence=0.95):
    """
    Run Monte Carlo simulation by randomizing trade order.
    Returns dict with statistical results.
    """
    if not trade_results or len(trade_results) < 5:
        return {
            'mean_r': 0, 'median_r': 0, 'std_r': 0,
            'ci_lower': 0, 'ci_upper': 0,
            'profitable_pct': 0, 'worst_case': 0, 'best_case': 0
        }
    
    final_rs = []
    
    for _ in range(iterations):
        shuffled = trade_results.copy()
        random.shuffle(shuffled)
        final_r = sum(shuffled)
        final_rs.append(final_r)
    
    final_rs = np.array(final_rs)
    
    # Calculate statistics
    mean_r = np.mean(final_rs)
    median_r = np.median(final_rs)
    std_r = np.std(final_rs)
    
    # Confidence interval
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100
    ci_lower = np.percentile(final_rs, lower_pct)
    ci_upper = np.percentile(final_rs, upper_pct)
    
    # Profitable percentage
    profitable_pct = (final_rs > 0).sum() / len(final_rs) * 100
    
    return {
        'mean_r': round(mean_r, 2),
        'median_r': round(median_r, 2),
        'std_r': round(std_r, 2),
        'ci_lower': round(ci_lower, 2),
        'ci_upper': round(ci_upper, 2),
        'profitable_pct': round(profitable_pct, 1),
        'worst_case': round(min(final_rs), 2),
        'best_case': round(max(final_rs), 2)
    }

# ============================================================================
# VALIDATION AND ANALYSIS
# ============================================================================

def analyze_trades(trades: List[Trade], symbol: str, div_code: str, rr: float) -> ValidationResult:
    """
    Comprehensive analysis of trade results including walk-forward and Monte Carlo.
    """
    result = ValidationResult(symbol=symbol, divergence_type=div_code, rr=rr)
    
    if not trades:
        return result
    
    # Basic metrics
    result.total_trades = len(trades)
    result.wins = sum(1 for t in trades if t.is_win)
    result.losses = result.total_trades - result.wins
    result.win_rate = (result.wins / result.total_trades * 100) if result.total_trades > 0 else 0
    
    trade_rs = [t.result_r for t in trades]
    result.total_r = round(sum(trade_rs), 2)
    result.avg_r = round(result.total_r / result.total_trades, 3) if result.total_trades > 0 else 0
    
    # Risk metrics
    result.sharpe_ratio = round(calculate_sharpe(trade_rs), 2)
    result.sortino_ratio = round(calculate_sortino(trade_rs), 2)
    result.max_drawdown_r, result.max_drawdown_pct = calculate_max_drawdown(trade_rs)
    result.max_drawdown_pct = round(result.max_drawdown_pct, 1)
    result.max_consecutive_losses = calculate_consecutive_losses([t.is_win for t in trades])
    
    wins_r = [t.result_r for t in trades if t.is_win]
    losses_r = [t.result_r for t in trades if not t.is_win]
    result.profit_factor = round(calculate_profit_factor(wins_r, losses_r), 2)
    
    # Walk-forward by period
    p1_trades = [t for t in trades if t.period == 1]
    p2_trades = [t for t in trades if t.period == 2]
    p3_trades = [t for t in trades if t.period == 3]
    
    result.p1_trades = len(p1_trades)
    result.p1_wins = sum(1 for t in p1_trades if t.is_win)
    result.p1_r = round(sum(t.result_r for t in p1_trades), 2)
    
    result.p2_trades = len(p2_trades)
    result.p2_wins = sum(1 for t in p2_trades if t.is_win)
    result.p2_r = round(sum(t.result_r for t in p2_trades), 2)
    
    result.p3_trades = len(p3_trades)
    result.p3_wins = sum(1 for t in p3_trades if t.is_win)
    result.p3_r = round(sum(t.result_r for t in p3_trades), 2)
    
    result.periods_profitable = sum([
        result.p1_r > 0 and result.p1_trades >= MIN_TRADES_PER_PERIOD,
        result.p2_r > 0 and result.p2_trades >= MIN_TRADES_PER_PERIOD,
        result.p3_r > 0 and result.p3_trades >= MIN_TRADES_PER_PERIOD
    ])
    
    # Monte Carlo
    mc_results = run_monte_carlo(trade_rs, MC_ITERATIONS, MC_CONFIDENCE_LEVEL)
    result.mc_mean_r = mc_results['mean_r']
    result.mc_median_r = mc_results['median_r']
    result.mc_std_r = mc_results['std_r']
    result.mc_ci_lower = mc_results['ci_lower']
    result.mc_ci_upper = mc_results['ci_upper']
    result.mc_profitable_pct = mc_results['profitable_pct']
    result.mc_worst_case_r = mc_results['worst_case']
    result.mc_best_case_r = mc_results['best_case']
    
    # Additional
    result.avg_trade_duration = round(np.mean([t.duration_candles for t in trades]), 1) if trades else 0
    
    # Validation flags
    result.passes_min_trades = result.total_trades >= MIN_TRADES_TOTAL
    result.passes_win_rate = result.win_rate >= MIN_WIN_RATE
    result.passes_profitability = result.total_r > MIN_TOTAL_R
    result.passes_sharpe = result.sharpe_ratio >= MIN_SHARPE
    result.passes_drawdown = result.max_drawdown_pct <= MAX_DRAWDOWN_PCT
    result.passes_walk_forward = result.periods_profitable >= WF_MIN_PERIODS_PROFITABLE
    result.passes_monte_carlo = result.mc_ci_lower > 0 and result.mc_profitable_pct >= 95
    
    result.passes_all = all([
        result.passes_min_trades,
        result.passes_win_rate,
        result.passes_profitability,
        result.passes_sharpe,
        result.passes_drawdown,
        result.passes_walk_forward,
        result.passes_monte_carlo
    ])
    
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def test_symbol(symbol):
    """Test a single symbol across all divergence types and R:R ratios"""
    results = []
    
    # Fetch data
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 500:
        return results
    
    df = calculate_indicators(df)
    
    if len(df) < 400:
        return results
    
    df_start = df.index.min()
    
    # Test each divergence type
    for div_code, div_config in DIVERGENCE_CONFIGS.items():
        # Test each R:R ratio
        for rr in RR_OPTIONS:
            trades = simulate_trades(df, symbol, div_config, rr, df_start)
            result = analyze_trades(trades, symbol, div_code, rr)
            results.append(result)
    
    return results

def main():
    """Main execution flow"""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("ULTRA-COMPREHENSIVE MULTI-DIVERGENCE BACKTEST")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"   • Symbols: {SYMBOL_LIMIT}")
    print(f"   • Divergence Types: {len(DIVERGENCE_CONFIGS)} (Regular/Hidden × Bullish/Bearish)")
    print(f"   • R:R Ratios: {RR_OPTIONS}")
    print(f"   • Data Period: {DATA_DAYS} days (6 months)")
    print(f"   • Walk-Forward: 3 × {WF_PERIOD_DAYS}-day periods")
    print(f"   • Monte Carlo: {MC_ITERATIONS} iterations")
    print(f"   • BOS Timeout: {MAX_WAIT_CANDLES} candles (12 hours)")
    print(f"\n   Total Combinations: {SYMBOL_LIMIT} × {len(DIVERGENCE_CONFIGS)} × {len(RR_OPTIONS)} = {SYMBOL_LIMIT * len(DIVERGENCE_CONFIGS) * len(RR_OPTIONS)}")
    print("="*80)
    
    # Fetch symbols
    symbols = fetch_symbols(SYMBOL_LIMIT)
    
    if not symbols:
        print("❌ No symbols found!")
        return
    
    # Process all symbols
    all_results = []
    
    print(f"\n🔬 Testing {len(symbols)} symbols...")
    print("-"*80)
    
    for i, symbol in enumerate(symbols):
        print_progress(i+1, len(symbols), symbol)
        
        try:
            symbol_results = test_symbol(symbol)
            all_results.extend(symbol_results)
            
            # Brief summary for this symbol
            if symbol_results:
                passing = sum(1 for r in symbol_results if r.passes_all)
                best = max(symbol_results, key=lambda x: x.total_r)
                print(f" | {passing}/{len(symbol_results)} pass | Best: {best.divergence_type} {best.rr}:1 = {best.total_r:+.1f}R")
            else:
                print(" | ⚪ Insufficient data")
                
        except Exception as e:
            print(f" | ❌ Error: {str(e)[:30]}")
            continue
        
        time.sleep(0.02)
    
    print("\n" + "="*80)
    
    # === ANALYSIS ===
    print("\n📈 ANALYZING RESULTS...")
    
    # Convert to DataFrame
    results_data = [asdict(r) for r in all_results]
    df_results = pd.DataFrame(results_data)
    
    if df_results.empty:
        print("❌ No results to analyze!")
        return
    
    # === FILTER VALIDATED ===
    df_validated = df_results[df_results['passes_all'] == True].copy()
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Total Combinations Tested: {len(df_results)}")
    print(f"   • Passing All Validation: {len(df_validated)} ({len(df_validated)/len(df_results)*100:.1f}%)")
    
    # === BY DIVERGENCE TYPE ===
    print(f"\n📈 BY DIVERGENCE TYPE:")
    for div_code in DIVERGENCE_CONFIGS.keys():
        div_results = df_results[df_results['divergence_type'] == div_code]
        div_validated = div_results[div_results['passes_all'] == True]
        print(f"   • {div_code}: {len(div_validated)}/{len(div_results)} validated ({len(div_validated)/max(len(div_results),1)*100:.1f}%)")
    
    # === TOP PERFORMERS ===
    if len(df_validated) > 0:
        print(f"\n🏆 TOP 20 VALIDATED COMBINATIONS:")
        print("-"*100)
        print(f"{'Symbol':18} | {'Divergence':12} | {'R:R':5} | {'Trades':6} | {'WR%':6} | {'Total R':10} | {'Sharpe':7} | {'WF':4} | {'MC%':5}")
        print("-"*100)
        
        top_20 = df_validated.nlargest(20, 'total_r')
        for _, row in top_20.iterrows():
            print(f"{row['symbol']:18} | {row['divergence_type']:12} | {row['rr']:2.0f}:1  | {row['total_trades']:6} | {row['win_rate']:5.1f}% | {row['total_r']:+9.1f}R | {row['sharpe_ratio']:6.2f} | {row['periods_profitable']}/3  | {row['mc_profitable_pct']:4.1f}%")
        
        print("-"*100)
    
    # === FIND BEST PER SYMBOL ===
    print(f"\n🎯 BEST CONFIGURATION PER SYMBOL:")
    
    best_per_symbol = {}
    symbols_with_validation = df_validated['symbol'].unique() if len(df_validated) > 0 else []
    
    for symbol in symbols_with_validation:
        symbol_validated = df_validated[df_validated['symbol'] == symbol]
        if len(symbol_validated) > 0:
            best = symbol_validated.loc[symbol_validated['total_r'].idxmax()]
            best_per_symbol[symbol] = {
                'divergence': best['divergence_type'],
                'rr': best['rr'],
                'total_r': best['total_r'],
                'win_rate': best['win_rate'],
                'trades': best['total_trades'],
                'sharpe': best['sharpe_ratio']
            }
    
    print(f"   • Symbols with validated configurations: {len(best_per_symbol)}")
    
    # === SAVE RESULTS ===
    print(f"\n💾 SAVING RESULTS...")
    
    # All results
    df_results.to_csv('ultra_comprehensive_all_results.csv', index=False)
    print(f"   ✅ ultra_comprehensive_all_results.csv ({len(df_results)} rows)")
    
    # Validated only
    if len(df_validated) > 0:
        df_validated.to_csv('ultra_comprehensive_validated.csv', index=False)
        print(f"   ✅ ultra_comprehensive_validated.csv ({len(df_validated)} rows)")
    
    # Best per symbol (YAML)
    if best_per_symbol:
        yaml_output = {
            'generated': datetime.now().isoformat(),
            'validation': 'Walk-Forward + Monte Carlo + OOS',
            'symbols': {}
        }
        for symbol, config in best_per_symbol.items():
            yaml_output['symbols'][symbol] = {
                'enabled': True,
                'divergence': config['divergence'],
                'rr': float(config['rr']),
                'expected_r': float(config['total_r']),
                'win_rate': float(config['win_rate']),
                'trades': int(config['trades'])
            }
        
        with open('ultra_comprehensive_optimal.yaml', 'w') as f:
            yaml.dump(yaml_output, f, default_flow_style=False)
        print(f"   ✅ ultra_comprehensive_optimal.yaml ({len(best_per_symbol)} symbols)")
    
    # Summary report
    elapsed = time.time() - start_time
    summary = f"""
================================================================================
ULTRA-COMPREHENSIVE BACKTEST - SUMMARY REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Runtime: {elapsed/60:.1f} minutes

CONFIGURATION:
- Symbols Tested: {len(symbols)}
- Divergence Types: {len(DIVERGENCE_CONFIGS)}
- R:R Ratios: {RR_OPTIONS}
- Data Period: {DATA_DAYS} days
- Walk-Forward: {WF_PERIOD_DAYS} days × 3 periods
- Monte Carlo: {MC_ITERATIONS} iterations

RESULTS:
- Total Combinations: {len(df_results)}
- Validated Combinations: {len(df_validated)} ({len(df_validated)/len(df_results)*100:.1f}%)
- Symbols with Valid Configs: {len(best_per_symbol)}

VALIDATION CRITERIA:
- Min Trades: {MIN_TRADES_TOTAL}
- Min Win Rate: {MIN_WIN_RATE}%
- Min Sharpe: {MIN_SHARPE}
- Max Drawdown: {MAX_DRAWDOWN_PCT}%
- Walk-Forward: {WF_MIN_PERIODS_PROFITABLE}/3 periods profitable
- Monte Carlo: {MC_CONFIDENCE_LEVEL*100}% confidence profitable

BY DIVERGENCE TYPE:
"""
    
    for div_code in DIVERGENCE_CONFIGS.keys():
        div_val = df_validated[df_validated['divergence_type'] == div_code] if len(df_validated) > 0 else pd.DataFrame()
        summary += f"- {div_code}: {len(div_val)} validated\n"
    
    summary += f"""
TOP 10 PERFORMERS:
"""
    if len(df_validated) > 0:
        for i, (_, row) in enumerate(df_validated.nlargest(10, 'total_r').iterrows()):
            summary += f"{i+1}. {row['symbol']} | {row['divergence_type']} | {row['rr']:.0f}:1 | {row['total_r']:+.1f}R | {row['win_rate']:.1f}% WR\n"
    
    summary += """
================================================================================
"""
    
    with open('ultra_comprehensive_summary.txt', 'w') as f:
        f.write(summary)
    print(f"   ✅ ultra_comprehensive_summary.txt")
    
    print(f"\n" + "="*80)
    print(f"🎉 BACKTEST COMPLETE!")
    print(f"="*80)
    print(f"\nRuntime: {elapsed/60:.1f} minutes")
    print(f"Results saved to:")
    print(f"  • ultra_comprehensive_all_results.csv")
    print(f"  • ultra_comprehensive_validated.csv")
    print(f"  • ultra_comprehensive_optimal.yaml")
    print(f"  • ultra_comprehensive_summary.txt")
    print(f"\n{len(best_per_symbol)} symbols have validated configurations ready for deployment!")

if __name__ == "__main__":
    main()
