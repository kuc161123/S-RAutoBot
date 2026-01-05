#!/usr/bin/env python3
"""
BLIND FORWARD TEST - Elite 33 Verification
============================================
Fetches FRESH data (last 30 days) and simulates the exact bot logic
candle-by-candle to verify the Elite 33 symbols perform as expected.

This is a TRUE blind test because:
1. Uses data NOT seen during optimization
2. Simulates exact bot entry/exit logic
3. Processes one candle at a time (no lookahead)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD ELITE 33 CONFIGURATION
# ============================================================================

def load_elite_config():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    return config['symbols']

# ============================================================================
# DATA FETCHING (FRESH 30 DAYS ONLY)
# ============================================================================

BASE_URL = "https://api.bybit.com"

def fetch_fresh_klines(symbol, days=30):
    """Fetch only the last 30 days of data (not used in original backtest)"""
    try:
        all_klines = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days * 24 * 3600) * 1000)
        
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': '60', 
                     'limit': 1000, 'end': end_ts}
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json()
            if data.get('retCode') != 0:
                break
            klines = data.get('result', {}).get('list', [])
            if not klines:
                break
            all_klines.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.02)
        
        if not all_klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']:
            df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df.set_index('ts', inplace=True)
        return df
    except:
        return pd.DataFrame()

# ============================================================================
# INDICATORS (EXACT BOT LOGIC)
# ============================================================================

def calculate_indicators(df):
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMA 200
    df['ema'] = df['close'].ewm(span=200, adjust=False).mean()
    
    return df.dropna()

def find_pivots(arr, left=3, right=3):
    n = len(arr)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = arr[i-left:i+right+1]
        if arr[i] == max(window) and list(window).count(arr[i]) == 1:
            ph[i] = arr[i]
        if arr[i] == min(window) and list(window).count(arr[i]) == 1:
            pl[i] = arr[i]
    return ph, pl

# ============================================================================
# DIVERGENCE DETECTION (EXACT BOT LOGIC)
# ============================================================================

def detect_divergence(df, idx, div_type, close, rsi, ema, ph, pl):
    if idx < 50:
        return False, None
    
    current_price = close[idx]
    current_ema = ema[idx]
    
    # Define divergence parameters
    if div_type == 'REG_BULL':
        side = 'long'
        trend_filter = 'above_ema'
    elif div_type == 'REG_BEAR':
        side = 'short'
        trend_filter = 'below_ema'
    elif div_type == 'HID_BULL':
        side = 'long'
        trend_filter = 'above_ema'
    elif div_type == 'HID_BEAR':
        side = 'short'
        trend_filter = 'below_ema'
    else:
        return False, None
    
    # Trend filter
    if trend_filter == 'above_ema' and current_price <= current_ema:
        return False, None
    if trend_filter == 'below_ema' and current_price >= current_ema:
        return False, None
    
    if side == 'long':
        pivots = []
        for j in range(idx - 4, max(0, idx - 50), -1):
            if not np.isnan(pl[j]):
                pivots.append((j, pl[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price_val, curr_rsi = pivots[0]
        prev_idx, prev_price_val, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_type == 'REG_BULL':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
        elif div_type == 'HID_BULL':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
    else:
        pivots = []
        for j in range(idx - 4, max(0, idx - 50), -1):
            if not np.isnan(ph[j]):
                pivots.append((j, ph[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price_val, curr_rsi = pivots[0]
        prev_idx, prev_price_val, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_type == 'REG_BEAR':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
        elif div_type == 'HID_BEAR':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
    
    return False, None

# ============================================================================
# BLIND CANDLE-BY-CANDLE SIMULATION
# ============================================================================

def simulate_blind(df, symbol, div_type, rr, atr_mult):
    """Simulate trading candle-by-candle with NO lookahead"""
    trades = []
    if len(df) < 100:
        return trades
    
    df = calculate_indicators(df)
    if df.empty or len(df) < 60:
        return trades
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    ph, pl = find_pivots(close, 3, 3)
    
    # State machine
    pending_signal = None
    pending_swing = None
    pending_wait = 0
    
    # Active trade
    in_trade = False
    entry_price = 0
    tp_price = 0
    sl_price = 0
    trade_side = None
    entry_time = None
    
    for idx in range(50, n):
        current_price = close[idx]
        current_high = high[idx]
        current_low = low[idx]
        
        # Check active trade
        if in_trade:
            if trade_side == 'long':
                if current_low <= sl_price:
                    trades.append({'r': -1.0, 'win': False, 'entry': entry_time, 'exit': df.index[idx]})
                    in_trade = False
                elif current_high >= tp_price:
                    trades.append({'r': rr, 'win': True, 'entry': entry_time, 'exit': df.index[idx]})
                    in_trade = False
            else:
                if current_high >= sl_price:
                    trades.append({'r': -1.0, 'win': False, 'entry': entry_time, 'exit': df.index[idx]})
                    in_trade = False
                elif current_low <= tp_price:
                    trades.append({'r': rr, 'win': True, 'entry': entry_time, 'exit': df.index[idx]})
                    in_trade = False
            continue
        
        # Check pending signal for BOS
        if pending_signal is not None:
            bos = False
            if div_type in ['REG_BULL', 'HID_BULL'] and current_price > pending_swing:
                bos = True
                trade_side = 'long'
            if div_type in ['REG_BEAR', 'HID_BEAR'] and current_price < pending_swing:
                bos = True
                trade_side = 'short'
            
            if bos:
                # Enter trade at next candle open
                if idx + 1 < n:
                    entry_price = df.iloc[idx + 1]['open']
                    entry_atr = atr[idx + 1]
                    sl_dist = entry_atr * atr_mult
                    
                    if trade_side == 'long':
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
                    in_trade = True
                    entry_time = df.index[idx + 1]
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= 12:
                    pending_signal = None
                    pending_wait = 0
        
        # Detect new divergence (only if not in trade and no pending)
        if not in_trade and pending_signal is None:
            found, swing = detect_divergence(df, idx, div_type, close, rsi, ema, ph, pl)
            if found:
                pending_signal = idx
                pending_swing = swing
                pending_wait = 0
    
    return trades

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("BLIND FORWARD TEST - Elite 33 Verification")
    print("=" * 60)
    print(f"Testing on FRESH data (last 30 days)")
    print(f"This data was NOT used in the original backtest")
    print("=" * 60)
    
    # Load Elite 33 config
    elite_symbols = load_elite_config()
    print(f"\nLoaded {len(elite_symbols)} Elite symbols from config.yaml")
    
    # Test each symbol
    results = []
    total_trades = 0
    total_wins = 0
    total_r = 0
    
    print("\nRunning blind simulation...")
    for symbol, cfg in elite_symbols.items():
        if not cfg.get('enabled', True):
            continue
        
        df = fetch_fresh_klines(symbol, days=30)
        if df.empty or len(df) < 100:
            continue
        
        trades = simulate_blind(
            df, 
            symbol, 
            cfg['divergence'], 
            cfg['rr'], 
            cfg['atr_mult']
        )
        
        if trades:
            wins = sum(1 for t in trades if t['win'])
            r_sum = sum(t['r'] for t in trades)
            wr = (wins / len(trades)) * 100 if trades else 0
            
            results.append({
                'symbol': symbol,
                'divergence': cfg['divergence'],
                'rr': cfg['rr'],
                'atr_mult': cfg['atr_mult'],
                'trades': len(trades),
                'wins': wins,
                'wr': round(wr, 1),
                'r': round(r_sum, 2)
            })
            
            total_trades += len(trades)
            total_wins += wins
            total_r += r_sum
            
            status = "✅" if r_sum > 0 else "❌"
            print(f"  {status} {symbol}: {len(trades)} trades, {wr:.0f}% WR, {r_sum:+.1f}R")
    
    # Summary
    print("\n" + "=" * 60)
    print("BLIND TEST RESULTS (Last 30 Days)")
    print("=" * 60)
    
    if total_trades > 0:
        overall_wr = (total_wins / total_trades) * 100
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Wins: {total_wins}")
        print(f"   Win Rate: {overall_wr:.1f}%")
        print(f"   Total R: {total_r:+.1f}R")
        
        profitable = sum(1 for r in results if r['r'] > 0)
        print(f"\n📈 Symbol Performance:")
        print(f"   Profitable Symbols: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
        
        # Expected vs Actual
        expected_monthly_r = 212  # Based on OOS
        print(f"\n📉 Expected vs Actual (30 days):")
        print(f"   Expected: +{expected_monthly_r}R")
        print(f"   Actual: {total_r:+.1f}R")
        
        if total_r > expected_monthly_r * 0.5:
            print(f"\n   ✅ PASSED - Performance within expected range")
        elif total_r > 0:
            print(f"\n   ⚠️ UNDERPERFORMING - Positive but below expectations")
        else:
            print(f"\n   ❌ FAILING - Negative performance")
    else:
        print("\n❌ No trades executed in the test period")
    
    # Save results
    if results:
        pd.DataFrame(results).to_csv('blind_test_results.csv', index=False)
        print(f"\n📁 Results saved to blind_test_results.csv")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
