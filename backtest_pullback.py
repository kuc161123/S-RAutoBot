#!/usr/bin/env python3
"""
PULLBACK/RETEST Strategy Backtest

Strategy Logic:
1. Identify Support/Resistance zones (1H timeframe concepts)
2. Detect bullish/bearish structure (HH/HL or LH/LL)
3. Wait for breakout above resistance (or below support)
4. Wait for pullback to the broken level
5. Enter when structure resumes in breakout direction

Features:
- Multi-timeframe concept (using 15M as base)
- Structure detection (HH/HL/LH/LL)
- S/R zone detection using swing points
- 3:1 R:R target
- Realistic fees and slippage
- Walk-forward validation

RESULTS ONLY - Does not affect live bot.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
import yaml
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Tuple, List, Optional

BYBIT_BASE = "https://api.bybit.com"

# REALISTIC EXECUTION COSTS
TOTAL_FEE_PCT = 0.11      # 0.055% x 2 round-trip
SLIPPAGE_PCT = 0.02       # Slippage on entry/exit
TOTAL_COST_PCT = TOTAL_FEE_PCT + SLIPPAGE_PCT

# R:R RATIO - 3:1 for pullback entries
RR_RATIO = 3.0

# MINIMUM REQUIREMENTS
MIN_TRADES = 10
MIN_WR = 35              # At 3:1, breakeven is 25%, target 35%+
MIN_EV = 0.5             # Positive EV at 3:1

# STRUCTURE DETECTION PARAMS
SWING_LOOKBACK = 5       # Candles on each side for swing detection
SR_TOLERANCE = 0.003     # 0.3% tolerance for S/R clustering
PULLBACK_TOLERANCE = 0.005  # 0.5% tolerance for pullback detection


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Calculate Wilson score lower bound for win rate confidence."""
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)


def calculate_ev(wr_decimal: float, rr: float = 3.0) -> float:
    """Calculate expected value at given R:R."""
    return (wr_decimal * rr) - ((1 - wr_decimal) * 1.0)


def get_all_symbols(limit=100) -> list:
    """Fetch top symbols by volume."""
    url = f"{BYBIT_BASE}/v5/market/tickers"
    params = {'category': 'linear'}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get('retCode') != 0:
            return []
        
        tickers = data.get('result', {}).get('list', [])
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_tickers.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        
        return [t['symbol'] for t in usdt_tickers[:limit]]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


def fetch_klines(symbol: str, interval: str = '15', days: int = 60) -> pd.DataFrame:
    """Fetch historical klines with pagination."""
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    retries = 0
    max_retries = 3
    
    while current_end > start_time:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            
            if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
                retries += 1
                if retries >= max_retries:
                    break
                time.sleep(0.5)
                continue
                
            klines = data['result']['list']
            all_data.extend(klines)
            earliest = int(klines[-1][0])
            if earliest <= start_time:
                break
            current_end = earliest - 1
            retries = 0
            time.sleep(0.05)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                break
            time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    if len(df) < 50:
        return df
    
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # EMA 20 for trend
    df['ema20'] = df['close'].ewm(span=20).mean()
    
    # Mark swing highs and lows
    df['swing_high'] = False
    df['swing_low'] = False
    
    for i in range(SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
        # Swing high: higher than N candles on each side
        is_swing_high = True
        is_swing_low = True
        
        for j in range(1, SWING_LOOKBACK + 1):
            if df.iloc[i]['high'] <= df.iloc[i-j]['high'] or df.iloc[i]['high'] <= df.iloc[i+j]['high']:
                is_swing_high = False
            if df.iloc[i]['low'] >= df.iloc[i-j]['low'] or df.iloc[i]['low'] >= df.iloc[i+j]['low']:
                is_swing_low = False
        
        df.loc[df.index[i], 'swing_high'] = is_swing_high
        df.loc[df.index[i], 'swing_low'] = is_swing_low
    
    return df.dropna()


def get_recent_swing_points(df: pd.DataFrame, idx: int, lookback: int = 50) -> Tuple[List[float], List[float]]:
    """Get recent swing highs and lows before given index."""
    start_idx = max(0, idx - lookback)
    subset = df.iloc[start_idx:idx]
    
    swing_highs = subset[subset['swing_high']]['high'].tolist()
    swing_lows = subset[subset['swing_low']]['low'].tolist()
    
    return swing_highs, swing_lows


def detect_sr_levels(swing_highs: List[float], swing_lows: List[float], 
                     current_price: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect nearest support and resistance levels.
    Returns (support, resistance) or None if not found.
    """
    if not swing_highs or not swing_lows:
        return None, None
    
    # Find resistance (nearest swing high above current price)
    resistances = [h for h in swing_highs if h > current_price * 1.001]
    resistance = min(resistances) if resistances else None
    
    # Find support (nearest swing low below current price)
    supports = [l for l in swing_lows if l < current_price * 0.999]
    support = max(supports) if supports else None
    
    return support, resistance


def detect_structure(df: pd.DataFrame, idx: int, lookback: int = 30) -> str:
    """
    Detect market structure using recent swing points.
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    start_idx = max(0, idx - lookback)
    subset = df.iloc[start_idx:idx+1]
    
    # Get swing points with their indices
    swing_highs = []
    swing_lows = []
    
    for i, row in subset.iterrows():
        if row['swing_high']:
            swing_highs.append((i, row['high']))
        if row['swing_low']:
            swing_lows.append((i, row['low']))
    
    # Need at least 2 highs and 2 lows for structure
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'neutral'
    
    # Get last 2 swing highs and lows
    last_2_highs = swing_highs[-2:]
    last_2_lows = swing_lows[-2:]
    
    # Check for HH/HL (bullish)
    hh = last_2_highs[-1][1] > last_2_highs[-2][1]
    hl = last_2_lows[-1][1] > last_2_lows[-2][1]
    
    # Check for LH/LL (bearish)
    lh = last_2_highs[-1][1] < last_2_highs[-2][1]
    ll = last_2_lows[-1][1] < last_2_lows[-2][1]
    
    if hh and hl:
        return 'bullish'
    elif lh and ll:
        return 'bearish'
    else:
        return 'neutral'


def detect_breakout(df: pd.DataFrame, idx: int, level: float, side: str) -> bool:
    """
    Detect if there was a breakout in recent candles.
    side: 'long' (break above resistance) or 'short' (break below support)
    """
    if idx < 5:
        return False
    
    lookback = 10
    start_idx = max(0, idx - lookback)
    
    for i in range(start_idx, idx):
        candle = df.iloc[i]
        
        if side == 'long':
            # Breakout above resistance: close above level
            if candle['close'] > level * 1.001:
                return True
        else:
            # Breakout below support: close below level
            if candle['close'] < level * 0.999:
                return True
    
    return False


def detect_pullback(df: pd.DataFrame, idx: int, breakout_level: float, side: str) -> bool:
    """
    Detect if price has pulled back to the breakout level.
    For long: price touching back toward resistance (now support)
    For short: price touching back toward support (now resistance)
    """
    candle = df.iloc[idx]
    tolerance = breakout_level * PULLBACK_TOLERANCE
    
    if side == 'long':
        # Pullback to broken resistance (now support)
        # Low touches near the level but close is above
        if abs(candle['low'] - breakout_level) <= tolerance:
            if candle['close'] > breakout_level:
                return True
    else:
        # Pullback to broken support (now resistance)
        # High touches near the level but close is below
        if abs(candle['high'] - breakout_level) <= tolerance:
            if candle['close'] < breakout_level:
                return True
    
    return False


def confirm_entry(df: pd.DataFrame, idx: int, side: str) -> bool:
    """
    Confirm entry after pullback.
    Looking for structure resuming in breakout direction.
    """
    if idx < 3:
        return False
    
    c0 = df.iloc[idx]      # Current candle
    c1 = df.iloc[idx - 1]  # Previous candle
    c2 = df.iloc[idx - 2]  # 2 candles ago
    
    if side == 'long':
        # Bullish confirmation: 
        # 1. Current close > previous high (bullish momentum)
        # 2. Current low > 2 candles ago low (higher low)
        # 3. Bullish candle (close > open)
        bullish_candle = c0['close'] > c0['open']
        momentum = c0['close'] > c1['high']
        higher_low = c0['low'] > c2['low']
        
        return bullish_candle and (momentum or higher_low)
    else:
        # Bearish confirmation:
        # 1. Current close < previous low (bearish momentum)
        # 2. Current high < 2 candles ago high (lower high)
        # 3. Bearish candle (close < open)
        bearish_candle = c0['close'] < c0['open']
        momentum = c0['close'] < c1['low']
        lower_high = c0['high'] < c2['high']
        
        return bearish_candle and (momentum or lower_high)


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, 
                   entry: float, atr: float, max_candles: int = 100) -> Tuple[str, int, float]:
    """
    Simulate trade with 3:1 R:R.
    Returns: (outcome, candles_held, r_achieved)
    """
    # SL = 1 ATR, TP = 3 ATR
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr * 1.2, min_sl_dist)  # Slightly wider SL for pullback
    tp_dist = sl_dist * RR_RATIO
    
    # Apply execution costs
    entry_cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + entry_cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - entry_cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    # Simulate candle-by-candle
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        
        if side == 'long':
            # Check SL first (conservative)
            if candle['low'] <= sl:
                return 'loss', i - entry_idx, -1.0
            elif candle['high'] >= tp:
                return 'win', i - entry_idx, RR_RATIO
        else:
            if candle['high'] >= sl:
                return 'loss', i - entry_idx, -1.0
            elif candle['low'] <= tp:
                return 'win', i - entry_idx, RR_RATIO
    
    return 'timeout', max_candles, -0.5  # Timeout = partial loss


def find_pullback_setups(df: pd.DataFrame, start_idx: int = 100) -> List[dict]:
    """
    Scan through data to find all pullback setups.
    Returns list of setup dictionaries.
    """
    setups = []
    
    # Track breakout levels
    active_breakout = None
    breakout_side = None
    
    for idx in range(start_idx, len(df) - 100):
        row = df.iloc[idx]
        current_price = row['close']
        
        # Get swing points and S/R levels
        swing_highs, swing_lows = get_recent_swing_points(df, idx)
        support, resistance = detect_sr_levels(swing_highs, swing_lows, current_price)
        
        # Detect current structure
        structure = detect_structure(df, idx)
        
        # Look for new breakouts
        if resistance and structure == 'bullish' and not active_breakout:
            if detect_breakout(df, idx, resistance, 'long'):
                active_breakout = resistance
                breakout_side = 'long'
        
        if support and structure == 'bearish' and not active_breakout:
            if detect_breakout(df, idx, support, 'short'):
                active_breakout = support
                breakout_side = 'short'
        
        # Check for pullback and entry confirmation
        if active_breakout and breakout_side:
            if detect_pullback(df, idx, active_breakout, breakout_side):
                if confirm_entry(df, idx, breakout_side):
                    setups.append({
                        'idx': idx,
                        'side': breakout_side,
                        'entry': current_price,
                        'breakout_level': active_breakout,
                        'atr': row['atr']
                    })
                    # Reset after finding setup
                    active_breakout = None
                    breakout_side = None
        
        # Reset if price moves too far from breakout level
        if active_breakout:
            distance = abs(current_price - active_breakout) / active_breakout
            if distance > 0.03:  # 3% away = invalidate
                active_breakout = None
                breakout_side = None
    
    return setups


def process_single_symbol(symbol: str, days: int = 60, train_pct: float = 0.6) -> dict:
    """Process a single symbol for pullback setups."""
    df = fetch_klines(symbol, '15', days)  # 15-minute candles
    
    if len(df) < 500:
        return None, 0, 0
    
    df = calculate_indicators(df)
    
    if len(df) < 200:
        return None, 0, 0
    
    total_candles = len(df)
    train_end = int(total_candles * train_pct)
    
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_test = df.iloc[train_end:].reset_index(drop=True)
    
    # Find setups in train period
    train_setups = find_pullback_setups(df_train, start_idx=100)
    test_setups = find_pullback_setups(df_test, start_idx=100)
    
    # Track results by side
    results = {
        'long': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0, 'r_values': []},
        'short': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0, 'r_values': []}
    }
    
    # Process train setups
    for setup in train_setups:
        outcome, candles, r_achieved = simulate_trade(
            df_train, setup['idx'], setup['side'], setup['entry'], setup['atr']
        )
        side = setup['side']
        if outcome == 'win':
            results[side]['train_wins'] += 1
        else:
            results[side]['train_losses'] += 1
        results[side]['r_values'].append(r_achieved)
    
    # Process test setups
    for setup in test_setups:
        outcome, candles, r_achieved = simulate_trade(
            df_test, setup['idx'], setup['side'], setup['entry'], setup['atr']
        )
        side = setup['side']
        if outcome == 'win':
            results[side]['test_wins'] += 1
        else:
            results[side]['test_losses'] += 1
        results[side]['r_values'].append(r_achieved)
    
    total_setups = len(train_setups) + len(test_setups)
    total_wins = sum(r['train_wins'] + r['test_wins'] for r in results.values())
    overall_wr = (total_wins / total_setups * 100) if total_setups > 0 else 0
    
    return results, total_setups, overall_wr


def run_backtest(num_symbols: int = 100, days: int = 60, train_pct: float = 0.6):
    """Run pullback strategy backtest."""
    print("="*70)
    print("📊 PULLBACK/RETEST STRATEGY BACKTEST")
    print("="*70)
    print(f"   Strategy: Breakout-Pullback-Entry")
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Timeframe: 15M (with structure analysis)")
    print(f"   Days: {days}")
    print(f"   Walk-forward: {train_pct*100:.0f}% train / {(1-train_pct)*100:.0f}% test")
    print(f"   Total Cost: {TOTAL_COST_PCT}%")
    print(f"   Min WR: {MIN_WR}% | Min EV: {MIN_EV}")
    print("="*70)
    
    # Get symbols
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    # Process each symbol
    all_results = []
    symbols_with_setups = 0
    
    print(f"\n📊 Processing symbols...")
    print("-" * 70)
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        try:
            results, total_setups, overall_wr = process_single_symbol(symbol, days, train_pct)
            
            if results and total_setups > 0:
                symbols_with_setups += 1
                
                for side in ['long', 'short']:
                    r = results[side]
                    train_total = r['train_wins'] + r['train_losses']
                    test_total = r['test_wins'] + r['test_losses']
                    total = train_total + test_total
                    
                    if total >= MIN_TRADES:
                        wins = r['train_wins'] + r['test_wins']
                        wr = wins / total * 100
                        lb_wr = wilson_lower_bound(wins, total)
                        avg_r = np.mean(r['r_values']) if r['r_values'] else 0
                        ev = calculate_ev(wins / total, RR_RATIO)
                        
                        if wr >= MIN_WR and ev >= MIN_EV:
                            all_results.append({
                                'symbol': symbol,
                                'side': side,
                                'trades': total,
                                'wins': wins,
                                'wr': round(wr, 1),
                                'lb_wr': round(lb_wr, 1),
                                'avg_r': round(avg_r, 2),
                                'ev': round(ev, 2),
                                'train_n': train_total,
                                'train_wr': round(r['train_wins'] / train_total * 100, 1) if train_total > 0 else 0,
                                'test_n': test_total,
                                'test_wr': round(r['test_wins'] / test_total * 100, 1) if test_total > 0 else 0
                            })
                            print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | ✅ {side:5} | N={total:3} | WR={wr:.1f}% | EV={ev:+.2f}R")
                
                if not any(r['symbol'] == symbol for r in all_results):
                    print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | {total_setups:3} setups | WR={overall_wr:.1f}% | ❌ Below threshold")
            else:
                print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | ⚠️  No pullback setups found")
                
        except Exception as e:
            print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | ❌ Error: {str(e)[:30]}")
        
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(symbols) - i - 1)
            print(f"\n⏱️  Progress: {i+1}/{len(symbols)} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m\n")
        
        time.sleep(0.02)
    
    elapsed = time.time() - start_time
    
    # Sort by EV
    all_results.sort(key=lambda x: x['ev'], reverse=True)
    
    print(f"\n✅ Processing complete!")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Symbols processed: {len(symbols)}")
    print(f"   Symbols with setups: {symbols_with_setups}")
    print(f"   Profitable setups: {len(all_results)}")
    
    # Count by side
    long_count = len([r for r in all_results if r['side'] == 'long'])
    short_count = len([r for r in all_results if r['side'] == 'short'])
    print(f"   Long: {long_count} | Short: {short_count}")
    
    # Print top results
    print("\n" + "="*70)
    print(f"🏆 TOP PULLBACK SETUPS (WR ≥ {MIN_WR}%, EV ≥ {MIN_EV})")
    print("="*70)
    
    print(f"\n📊 TOP 20 (by EV):")
    print("-"*70)
    for i, r in enumerate(all_results[:20], 1):
        print(f"{i:2}. {r['symbol']:15} {r['side']:5} | N={r['trades']:3} | WR={r['wr']:.1f}% (LB:{r['lb_wr']:.1f}%) | EV={r['ev']:+.2f}R")
        print(f"    Train: {r['train_wr']:.0f}% (N={r['train_n']}) | Test: {r['test_wr']:.0f}% (N={r['test_n']})")
    
    # Export to YAML
    output_file = 'backtest_pullback_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'strategy': 'Pullback/Retest',
            'rr_ratio': f'{RR_RATIO}:1',
            'timeframe': '15M',
            'generated': datetime.utcnow().isoformat(),
            'days_backtested': days,
            'symbols_tested': len(symbols),
            'symbols_with_setups': symbols_with_setups,
            'profitable_setups': len(all_results),
            'long_setups': long_count,
            'short_setups': short_count,
            'criteria': {
                'min_trades': MIN_TRADES,
                'min_wr': MIN_WR,
                'min_ev': MIN_EV
            },
            'execution_costs': TOTAL_COST_PCT,
            'runtime_minutes': round(elapsed/60, 1),
            'note': 'RESULTS ONLY - Review before deploying'
        }
    }
    
    for r in all_results:
        sym = r['symbol']
        side = r['side']
        
        if sym not in yaml_output:
            yaml_output[sym] = {'long': None, 'short': None}
        
        yaml_output[sym][side] = {
            'trades': r['trades'],
            'wins': r['wins'],
            'wr': r['wr'],
            'lb_wr': r['lb_wr'],
            'ev': r['ev'],
            'train_n': r['train_n'],
            'train_wr': r['train_wr'],
            'test_n': r['test_n'],
            'test_wr': r['test_wr']
        }
    
    with open(output_file, 'w') as f:
        f.write("# PULLBACK/RETEST Strategy Backtest Results\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# R:R Ratio: {RR_RATIO}:1 | Timeframe: 15M\n")
        f.write(f"# Symbols: {len(symbols)} | Days: {days}\n")
        f.write(f"# Criteria: WR>={MIN_WR}%, EV>={MIN_EV}\n")
        f.write("#\n")
        f.write("# NOTE: RESULTS ONLY - Review before deploying.\n\n")
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Profitable setups: {len(all_results)}")
    print(f"\n⚠️ NOTE: Review results before deploying to live bot!")
    
    return all_results


if __name__ == "__main__":
    run_backtest(num_symbols=100, days=60, train_pct=0.6)
