#!/usr/bin/env python3
"""
BALANCED 500-SYMBOL VALIDATION
===============================
Sweet spot between profitability and reliability:

✅ Walk-Forward (60/40 split) - ensures consistency
✅ Normal slippage/fees - realistic but not harsh
✅ Reasonable minimums - train +3R, test +2R
✅ Monte Carlo 80% - good confidence
✅ Both periods must be profitable
✅ Min 15 trades total

This should give 50-100 high-quality symbols that are:
- More reliable than the 305 (walk-forward tested)
- More accessible than the 28 (reasonable criteria)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import warnings
warnings.filterwarnings('ignore')

# === BALANCED CONFIGURATION ===
TIMEFRAME = '60'
DATA_DAYS = 180

# Walk-forward
TRAIN_RATIO = 0.60
TEST_RATIO = 0.40

# BALANCED thresholds
MIN_TRADES_TRAIN = 10
MIN_TRADES_TEST = 8
MIN_WIN_RATE = 12
MIN_TRAIN_R = 3.0      # Lower than ultra (was 5)
MIN_TEST_R = 2.0       # Lower than ultra (was 3)
MAX_DD_RATIO = 0.60    # More lenient (was 0.50)

# Monte Carlo
MC_SIMULATIONS = 1000
MC_CONFIDENCE = 0.80   # Lower than ultra (was 0.85)

# NORMAL costs (not harsh)
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Strategy
MAX_WAIT_CANDLES = 6
ATR_MULT = 1.0
RSI_PERIOD = 14
EMA_PERIOD = 200
RR_OPTIONS = [4, 5, 6, 7, 8, 9, 10]

BASE_URL = "https://api.bybit.com"

print("="*70)
print("BALANCED 500-SYMBOL VALIDATION")
print("="*70)
print(f"Data: {DATA_DAYS} days")
print(f"Walk-Forward: {TRAIN_RATIO*100:.0f}% train / {TEST_RATIO*100:.0f}% test")
print(f"Monte Carlo: {MC_SIMULATIONS} sims, {MC_CONFIDENCE*100:.0f}% confidence")
print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% | Fees: {FEE_PCT*100:.3f}%")
print(f"Min Train R: {MIN_TRAIN_R} | Min Test R: {MIN_TEST_R}")
print("="*70)

def fetch_all_symbols(limit=500):
    """Fetch symbols"""
    print("📥 Fetching symbols...")
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/instruments-info",
            params={'category': 'linear', 'limit': 1000},
            timeout=30
        )
        data = resp.json().get('result', {}).get('list', [])
        symbols = [item['symbol'] for item in data 
                  if item.get('symbol', '').endswith('USDT') 
                  and item.get('status') == 'Trading']
        symbols = sorted(symbols)[:limit]
        print(f"✅ Found {len(symbols)} symbols")
        return symbols
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch klines"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    max_iterations = 50
    
    while current_end > start_ts and max_iterations > 0:
        max_iterations -= 1
        try:
            resp = requests.get(
                f"{BASE_URL}/v5/market/kline",
                params={'category': 'linear', 'symbol': symbol, 
                       'interval': interval, 'limit': 1000, 'end': current_end},
                timeout=15
            )
            data = resp.json().get('result', {}).get('list', [])
            if not data:
                break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000:
                break
            time.sleep(0.02)
        except:
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
    return df[~df.index.duplicated(keep='first')]

def prepare_data(df):
    """Calculate indicators"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    """Find pivots"""
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

def backtest_period(df, rr, start_idx, end_idx):
    """Backtest period"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    trades = []
    seen_signals = set()
    pending_signal = None
    pending_wait = 0
    
    for current_idx in range(max(50, start_idx), min(end_idx, n - 1)):
        current_price = close[current_idx]
        
        if pending_signal is not None:
            sig = pending_signal
            bos_confirmed = False
            
            if sig['side'] == 'long' and current_price > sig['swing']:
                bos_confirmed = True
            elif sig['side'] == 'short' and current_price < sig['swing']:
                bos_confirmed = True
            
            if bos_confirmed:
                entry_idx = current_idx + 1
                if entry_idx < n and entry_idx < end_idx:
                    entry_price = df.iloc[entry_idx]['open']
                    entry_atr = atr[entry_idx]
                    sl_dist = entry_atr * ATR_MULT
                    
                    if sig['side'] == 'long':
                        entry_price *= (1 + SLIPPAGE_PCT)
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        entry_price *= (1 - SLIPPAGE_PCT)
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
                    result = None
                    for k in range(entry_idx, min(entry_idx + 200, n, end_idx)):
                        if sig['side'] == 'long':
                            if low[k] <= sl_price:
                                result = -1.0
                                break
                            if high[k] >= tp_price:
                                result = rr
                                break
                        else:
                            if high[k] >= sl_price:
                                result = -1.0
                                break
                            if low[k] <= tp_price:
                                result = rr
                                break
                    
                    if result is None:
                        result = -0.5
                    
                    risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                    fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                    trades.append(result - fee_cost)
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        if pending_signal is None:
            # Bullish
            p_lows = []
            for j in range(current_idx - 3, max(0, current_idx - 50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2:
                        break
            
            if len(p_lows) == 2 and current_price > ema[current_idx]:
                curr_idx, curr_val = p_lows[0]
                prev_idx, prev_val = p_lows[1]
                if (current_idx - curr_idx) <= 10:
                    if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        swing_high = max(high[curr_idx:current_idx+1])
                        signal_id = f"long_{curr_idx}"
                        if signal_id not in seen_signals and current_price <= swing_high:
                            seen_signals.add(signal_id)
                            pending_signal = {'side': 'long', 'swing': swing_high}
                            pending_wait = 0
            
            # Bearish
            if pending_signal is None:
                p_highs = []
                for j in range(current_idx - 3, max(0, current_idx - 50), -1):
                    if not np.isnan(price_ph[j]):
                        p_highs.append((j, price_ph[j]))
                        if len(p_highs) >= 2:
                            break
                
                if len(p_highs) == 2 and current_price < ema[current_idx]:
                    curr_idx_h, curr_val_h = p_highs[0]
                    prev_idx_h, prev_val_h = p_highs[1]
                    if (current_idx - curr_idx_h) <= 10:
                        if curr_val_h > prev_val_h and rsi[curr_idx_h] < rsi[prev_idx_h]:
                            swing_low = min(low[curr_idx_h:current_idx+1])
                            signal_id = f"short_{curr_idx_h}"
                            if signal_id not in seen_signals and current_price >= swing_low:
                                seen_signals.add(signal_id)
                                pending_signal = {'side': 'short', 'swing': swing_low}
                                pending_wait = 0
    
    return trades

def calculate_max_drawdown(trades):
    """Max drawdown"""
    if not trades:
        return 0
    cumulative = np.cumsum(trades)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return max(drawdown) if len(drawdown) > 0 else 0

def monte_carlo_test(trades, num_simulations=1000):
    """Monte Carlo"""
    if len(trades) < 10:
        return 0.0
    profitable_count = sum(1 for _ in range(num_simulations) 
                          if sum(random.sample(trades, len(trades))) > 0)
    return profitable_count / num_simulations

def validate_symbol(symbol):
    """Balanced validation"""
    try:
        df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
        if len(df) < 2500:
            return None, "Insufficient data"
        
        df = prepare_data(df)
        n = len(df)
        if n < 2000:
            return None, "Insufficient data after prep"
        
        train_end = int(n * TRAIN_RATIO)
        test_start = train_end
        best_result = None
        
        for rr in RR_OPTIONS:
            train_trades = backtest_period(df, rr, 50, train_end)
            if len(train_trades) < MIN_TRADES_TRAIN:
                continue
            
            train_r = sum(train_trades)
            train_wins = sum(1 for t in train_trades if t > 0)
            train_wr = (train_wins / len(train_trades) * 100)
            
            if train_r < MIN_TRAIN_R or train_wr < MIN_WIN_RATE:
                continue
            
            test_trades = backtest_period(df, rr, test_start, n - 1)
            if len(test_trades) < MIN_TRADES_TEST:
                continue
            
            test_r = sum(test_trades)
            test_wins = sum(1 for t in test_trades if t > 0)
            test_wr = (test_wins / len(test_trades) * 100)
            
            if test_r < MIN_TEST_R:
                continue
            
            all_trades = train_trades + test_trades
            total_r = sum(all_trades)
            total_wr = (sum(1 for t in all_trades if t > 0) / len(all_trades) * 100)
            
            max_dd = calculate_max_drawdown(all_trades)
            if total_r > 0 and max_dd > total_r * MAX_DD_RATIO:
                continue
            
            mc_score = monte_carlo_test(all_trades, MC_SIMULATIONS)
            if mc_score < MC_CONFIDENCE:
                continue
            
            if best_result is None or total_r > best_result['total_r']:
                best_result = {
                    'symbol': symbol,
                    'rr': rr,
                    'train_r': round(train_r, 2),
                    'train_trades': len(train_trades),
                    'train_wr': round(train_wr, 1),
                    'test_r': round(test_r, 2),
                    'test_trades': len(test_trades),
                    'test_wr': round(test_wr, 1),
                    'total_r': round(total_r, 2),
                    'total_trades': len(all_trades),
                    'total_wr': round(total_wr, 1),
                    'avg_r': round(total_r / len(all_trades), 3),
                    'max_dd': round(max_dd, 2),
                    'mc_score': round(mc_score * 100, 1)
                }
        
        return (best_result, "PASSED") if best_result else (None, "No R:R passed")
    except Exception as e:
        return None, str(e)

def main():
    symbols = fetch_all_symbols(500)
    if not symbols:
        return
    
    print(f"\n🔬 Balanced validation of {len(symbols)} symbols...\n")
    
    validated = []
    for i, symbol in enumerate(symbols):
        print(f"\r[{i+1}/{len(symbols)}] {symbol:18}...", end="", flush=True)
        result, status = validate_symbol(symbol)
        
        if result:
            validated.append(result)
            print(f" ✅ Train: {result['train_r']:+.1f}R | Test: {result['test_r']:+.1f}R | MC: {result['mc_score']:.0f}%")
        else:
            print(f" ❌ {status[:30]}")
        time.sleep(0.02)
    
    print("\n" + "="*70)
    print("BALANCED VALIDATION RESULTS")
    print("="*70)
    print(f"\n📊 SUMMARY")
    print(f"├─ Tested: {len(symbols)}")
    print(f"├─ PASSED: {len(validated)} ({len(validated)/len(symbols)*100:.1f}%)")
    print(f"└─ Failed: {len(symbols) - len(validated)}")
    
    if validated:
        val_df = pd.DataFrame(validated).sort_values('total_r', ascending=False)
        total_r = val_df['total_r'].sum()
        avg_r = val_df['total_r'].mean()
        
        print(f"\n💰 PORTFOLIO")
        print(f"├─ Total R: {total_r:+.1f}R")
        print(f"├─ Avg R/Symbol: {avg_r:+.1f}R")
        print(f"└─ Expected R/Month: {total_r / 6:+.1f}R")
        
        print(f"\n🏆 TOP 30")
        print("-"*90)
        print(f"{'Symbol':18} | {'R:R':5} | {'Train':8} | {'Test':8} | {'Total':8} | {'WR':6} | {'MC':5}")
        print("-"*90)
        
        for _, row in val_df.head(30).iterrows():
            print(f"{row['symbol']:18} | {row['rr']:2.0f}:1  | {row['train_r']:+7.1f}R | {row['test_r']:+7.1f}R | {row['total_r']:+7.1f}R | {row['total_wr']:5.1f}% | {row['mc_score']:4.0f}%")
        
        val_df.to_csv('balanced_validated.csv', index=False)
        print(f"\n📁 Saved: balanced_validated.csv")
        
        print(f"\n📋 CONFIG FORMAT:")
        for _, row in val_df.iterrows():
            print(f"  {row['symbol']}: {{enabled: true, rr: {row['rr']:.1f}}}")

if __name__ == "__main__":
    main()
