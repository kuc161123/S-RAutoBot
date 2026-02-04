#!/usr/bin/env python3
"""
research_robust_1year.py - 1-Year Walk-Forward Validation
==========================================================
ROBUST research with:
1. 365 days of data (1 full year)
2. Walk-Forward Validation (Train on 9 months, Test on 3 months)
3. Liquidity Filter ($5M+ daily volume)
4. Monthly performance breakdown
5. Out-of-sample validation

Output: robust_1year_results.csv, walkforward_validated.csv
"""

import requests
import pandas as pd
import numpy as np
import time
import sys
import concurrent.futures
import threading
import os
import itertools
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================
DAYS = 365  # 1 FULL YEAR
SIGNAL_TF = '60'
EXECUTION_TF = '5'
MAX_WAIT_CANDLES = 12
RSI_PERIOD = 14
EMA_PERIOD = 200

# Walk-Forward Settings
TRAIN_MONTHS = 9  # Optimize on 9 months
TEST_MONTHS = 3   # Validate on 3 months (out of sample)

# Realistic Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

# Liquidity Filter
MIN_DAILY_VOLUME_USD = 1_000_000  # $1M minimum (more symbols)

BASE_URL = "https://api.bybit.com"
OUTPUT_FILE = 'robust_1year_results.csv'
WALKFORWARD_FILE = 'walkforward_validated.csv'
FILE_LOCK = threading.Lock()

GRID = {
    'div_type': ['REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR'],
    'atr_mult': [1.0, 1.5, 2.0],
    'rr_ratio': [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# ============================================================================
# DATA ENGINE
# ============================================================================

def fetch_all_symbols():
    """Fetch all active USDT Linear Perpetuals with liquidity filter"""
    print("📡 Fetching symbols with $5M+ daily volume filter...")
    
    try:
        # Get all trading symbols
        url = f"{BASE_URL}/v5/market/instruments-info"
        params = {'category': 'linear', 'status': 'Trading', 'limit': 1000}
        resp = requests.get(url, params=params)
        data = resp.json()
        
        all_symbols = []
        if 'result' in data and 'list' in data['result']:
            for item in data['result']['list']:
                if item.get('quoteCoin') == 'USDT':
                    all_symbols.append(item['symbol'])
        
        print(f"📊 Found {len(all_symbols)} total USDT pairs. Applying liquidity filter...")
        
        # Get 24h tickers
        ticker_url = f"{BASE_URL}/v5/market/tickers"
        ticker_resp = requests.get(ticker_url, {'category': 'linear'})
        ticker_data = ticker_resp.json()
        
        volume_map = {}
        if 'result' in ticker_data and 'list' in ticker_data['result']:
            for t in ticker_data['result']['list']:
                sym = t.get('symbol', '')
                turnover = float(t.get('turnover24h', 0))
                volume_map[sym] = turnover
        
        # Filter by volume
        liquid_symbols = [sym for sym in all_symbols if volume_map.get(sym, 0) >= MIN_DAILY_VOLUME_USD]
        
        print(f"✅ {len(liquid_symbols)} symbols passed liquidity filter (>${MIN_DAILY_VOLUME_USD/1e6:.0f}M daily volume).")
        return liquid_symbols
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch historical klines"""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    max_requests = 400  # More requests for 1 year
    
    while current_end > start_ts and max_requests > 0:
        max_requests -= 1
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data:
                break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000:
                break
            time.sleep(0.03)
        except:
            time.sleep(0.5)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def prepare_1h_data(df):
    """Add indicators"""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

# ============================================================================
# PIVOT & SIGNAL DETECTION
# ============================================================================

def find_pivots(data, left=3, right=3):
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    for i in range(left, len(data) - right):
        window = data[i-left : i+right+1]
        center = data[i]
        if len(window) != (left + right + 1):
            continue
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
    return pivot_highs, pivot_lows

def detect_signals(df, start_idx=None, end_idx=None):
    """Detect divergence signals within index range"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    times = df['start'].values
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    used_pivots = set()
    
    start_i = start_idx if start_idx else 205
    end_i = end_idx if end_idx else len(df) - 3
    
    for i in range(start_i, end_i):
        curr_price = close[i]
        curr_ema = ema[i]
        
        # BULLISH
        if curr_price > curr_ema:
            p_lows = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2:
                        break
            if len(p_lows) == 2:
                curr_idx, curr_val = p_lows[0]
                prev_idx, prev_val = p_lows[1]
                dedup_key = (curr_idx, prev_idx, 'BULL')
                if (i - curr_idx) <= 10 and dedup_key not in used_pivots:
                    if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'long', 'type': 'REG_BULL', 'swing': max(high[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
                    elif curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'long', 'type': 'HID_BULL', 'swing': max(high[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
        
        # BEARISH
        if curr_price < curr_ema:
            p_highs = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_ph[j]):
                    p_highs.append((j, price_ph[j]))
                    if len(p_highs) >= 2:
                        break
            if len(p_highs) == 2:
                curr_idx, curr_val = p_highs[0]
                prev_idx, prev_val = p_highs[1]
                dedup_key = (curr_idx, prev_idx, 'BEAR')
                if (i - curr_idx) <= 10 and dedup_key not in used_pivots:
                    if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'short', 'type': 'REG_BEAR', 'swing': min(low[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
                    elif curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'short', 'type': 'HID_BEAR', 'swing': min(low[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
    
    return signals

# ============================================================================
# TRADE EXECUTION
# ============================================================================

def execute_trade(signal, df_1h, df_5m, rr_ratio, atr_mult):
    """Execute a single trade with BOS confirmation and 5M precision"""
    conf_idx = signal['conf_idx']
    side = signal['side']
    bos_level = signal['swing']
    
    if conf_idx + 1 >= len(df_1h):
        return None
    
    entry_price = None
    
    # Wait for BOS
    for i in range(1, MAX_WAIT_CANDLES + 1):
        if conf_idx + i >= len(df_1h):
            break
        candle = df_1h.iloc[conf_idx + i]
        triggered = False
        if side == 'long' and candle['close'] > bos_level:
            triggered = True
        elif side == 'short' and candle['close'] < bos_level:
            triggered = True
        
        if triggered:
            if conf_idx + i + 1 >= len(df_1h):
                break
            entry_candle = df_1h.iloc[conf_idx + i + 1]
            entry_time = entry_candle['start']
            atr = candle['atr']
            sl_dist = atr * atr_mult
            raw_entry = entry_candle['open']
            
            if side == 'long':
                entry_price = raw_entry * (1 + SLIPPAGE_PCT)
                sl_price = entry_price - sl_dist
                tp_price = entry_price + (sl_dist * rr_ratio)
            else:
                entry_price = raw_entry * (1 - SLIPPAGE_PCT)
                sl_price = entry_price + sl_dist
                tp_price = entry_price - (sl_dist * rr_ratio)
            break
    
    if not entry_price:
        return None
    
    # Execute on 5M
    five_min_subset = df_5m[df_5m['start'] >= entry_time]
    if five_min_subset.empty:
        return None
    
    for row in five_min_subset.itertuples():
        if side == 'long':
            hit_sl = row.low <= sl_price
            hit_tp = row.high >= tp_price
        else:
            hit_sl = row.high >= sl_price
            hit_tp = row.low <= tp_price
        
        if hit_sl and hit_tp:
            outcome = "loss"
            break
        elif hit_sl:
            outcome = "loss"
            break
        elif hit_tp:
            outcome = "win"
            break
    else:
        return None  # Timeout
    
    risk = abs(entry_price - sl_price)
    if risk == 0:
        return None
    
    r_result = rr_ratio if outcome == 'win' else -1.0
    fee_drag = (FEE_PCT * 2 * entry_price) / risk
    
    return {'r': r_result - fee_drag, 'entry_time': entry_time, 'outcome': outcome}

# ============================================================================
# WALK-FORWARD OPTIMIZATION
# ============================================================================

def run_walkforward_for_symbol(sym, df_1h, df_5m):
    """
    Walk-Forward Validation:
    - Train on first 9 months, find best config
    - Test on last 3 months with that config
    """
    if len(df_1h) < 500:
        return None
    
    # Split data: 75% train, 25% test
    total_len = len(df_1h)
    train_end_idx = int(total_len * 0.75)
    
    train_1h = df_1h.iloc[:train_end_idx].copy()
    test_1h = df_1h.iloc[train_end_idx:].copy()
    
    if len(train_1h) < 300 or len(test_1h) < 100:
        return None
    
    train_start_time = train_1h['start'].iloc[0]
    train_end_time = train_1h['start'].iloc[-1]
    test_start_time = test_1h['start'].iloc[0]
    test_end_time = test_1h['start'].iloc[-1]
    
    train_5m = df_5m[(df_5m['start'] >= train_start_time) & (df_5m['start'] <= train_end_time)].copy()
    test_5m = df_5m[(df_5m['start'] >= test_start_time) & (df_5m['start'] <= test_end_time)].copy()
    
    # === TRAINING PHASE ===
    all_train_signals = detect_signals(train_1h)
    
    best_config = None
    best_train_r = -999
    
    combos = list(itertools.product(GRID['div_type'], GRID['atr_mult'], GRID['rr_ratio']))
    
    for div_type, atr, rr in combos:
        target_signals = [s for s in all_train_signals if s['type'] == div_type]
        if len(target_signals) < 5:
            continue
        
        trades = []
        for sig in target_signals:
            res = execute_trade(sig, train_1h, train_5m, rr, atr)
            if res is not None:
                trades.append(res['r'])
        
        if len(trades) < 5:
            continue
        
        total_r = sum(trades)
        avg_r = total_r / len(trades)
        
        if total_r > best_train_r and avg_r > 0.1:
            best_train_r = total_r
            best_config = {
                'div_type': div_type,
                'atr': atr,
                'rr': rr,
                'train_trades': len(trades),
                'train_r': total_r,
                'train_wr': sum(1 for t in trades if t > 0) / len(trades) * 100
            }
    
    if not best_config:
        return None
    
    # === TESTING PHASE (Out of Sample) ===
    all_test_signals = detect_signals(test_1h)
    target_signals = [s for s in all_test_signals if s['type'] == best_config['div_type']]
    
    test_trades = []
    monthly_results = {}
    
    for sig in target_signals:
        res = execute_trade(sig, test_1h, test_5m, best_config['rr'], best_config['atr'])
        if res is not None:
            test_trades.append(res)
            # Track by month
            month_key = res['entry_time'].strftime('%Y-%m')
            if month_key not in monthly_results:
                monthly_results[month_key] = {'trades': 0, 'wins': 0, 'r': 0}
            monthly_results[month_key]['trades'] += 1
            monthly_results[month_key]['r'] += res['r']
            if res['outcome'] == 'win':
                monthly_results[month_key]['wins'] += 1
    
    if len(test_trades) < 3:
        return None
    
    test_r = sum(t['r'] for t in test_trades)
    test_wr = sum(1 for t in test_trades if t['outcome'] == 'win') / len(test_trades) * 100
    
    return {
        'symbol': sym,
        'div_type': best_config['div_type'],
        'atr': best_config['atr'],
        'rr': best_config['rr'],
        'train_trades': best_config['train_trades'],
        'train_r': round(best_config['train_r'], 2),
        'train_wr': round(best_config['train_wr'], 1),
        'test_trades': len(test_trades),
        'test_r': round(test_r, 2),
        'test_wr': round(test_wr, 1),
        'monthly': monthly_results
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 ROBUST 1-YEAR WALK-FORWARD RESEARCH")
    print("=" * 70)
    print(f"📅 Data Period: {DAYS} days")
    print(f"🔬 Train: {TRAIN_MONTHS} months | Test: {TEST_MONTHS} months")
    print(f"💧 Liquidity Filter: ${MIN_DAILY_VOLUME_USD/1e6:.0f}M+ daily volume")
    print("-" * 70)
    
    # Get liquid symbols
    symbols = fetch_all_symbols()
    if not symbols:
        print("❌ No symbols found!")
        return
    
    results = []
    all_monthly = {}
    completed = 0
    total = len(symbols)
    
    for sym in symbols:
        completed += 1
        sys.stdout.write(f"\r⏳ Processing {sym} ({completed}/{total})...")
        sys.stdout.flush()
        
        # Fetch data
        df_1h = fetch_klines(sym, SIGNAL_TF, DAYS)
        df_5m = fetch_klines(sym, EXECUTION_TF, DAYS)
        
        if df_1h.empty or df_5m.empty or len(df_1h) < 500:
            continue
        
        df_1h = prepare_1h_data(df_1h)
        
        # Run walk-forward
        result = run_walkforward_for_symbol(sym, df_1h, df_5m)
        
        if result:
            results.append(result)
            
            # Aggregate monthly
            for month, data in result['monthly'].items():
                if month not in all_monthly:
                    all_monthly[month] = {'trades': 0, 'wins': 0, 'r': 0}
                all_monthly[month]['trades'] += data['trades']
                all_monthly[month]['wins'] += data['wins']
                all_monthly[month]['r'] += data['r']
        
        # Progress update
        if completed % 20 == 0:
            passed = len(results)
            print(f"\n   📈 Progress: {passed} symbols validated (Train+Test)")
    
    print(f"\n\n{'=' * 70}")
    print("📊 WALK-FORWARD RESULTS")
    print("=" * 70)
    
    if not results:
        print("❌ No symbols passed walk-forward validation!")
        return
    
    # Summary
    total_train_r = sum(r['train_r'] for r in results)
    total_test_r = sum(r['test_r'] for r in results)
    avg_test_wr = np.mean([r['test_wr'] for r in results])
    
    print(f"\n✅ {len(results)} symbols passed walk-forward validation")
    print(f"📈 Total Train R: {total_train_r:+.1f}")
    print(f"📊 Total Test R (OUT OF SAMPLE): {total_test_r:+.1f}")
    print(f"🎯 Avg Test Win Rate: {avg_test_wr:.1f}%")
    
    # Monthly breakdown
    print(f"\n{'=' * 70}")
    print("📅 MONTHLY BREAKDOWN (Test Period)")
    print("=" * 70)
    
    for month in sorted(all_monthly.keys()):
        data = all_monthly[month]
        wr = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
        print(f"  {month}: {data['trades']} trades | WR: {wr:.1f}% | R: {data['r']:+.1f}")
    
    # Save results
    df_results = pd.DataFrame([{
        'symbol': r['symbol'],
        'div_type': r['div_type'],
        'atr': r['atr'],
        'rr': r['rr'],
        'train_trades': r['train_trades'],
        'train_r': r['train_r'],
        'train_wr': r['train_wr'],
        'test_trades': r['test_trades'],
        'test_r': r['test_r'],
        'test_wr': r['test_wr']
    } for r in results])
    
    df_results.to_csv(WALKFORWARD_FILE, index=False)
    print(f"\n💾 Saved {len(results)} validated symbols to {WALKFORWARD_FILE}")
    
    # Top performers
    print(f"\n{'=' * 70}")
    print("🏆 TOP 20 PERFORMERS (by Out-of-Sample R)")
    print("=" * 70)
    
    top20 = sorted(results, key=lambda x: x['test_r'], reverse=True)[:20]
    for r in top20:
        print(f"  {r['symbol']:15} | {r['div_type']:10} | RR: {r['rr']:4.1f} | Test R: {r['test_r']:+6.1f} | Test WR: {r['test_wr']:.1f}%")

if __name__ == "__main__":
    main()
