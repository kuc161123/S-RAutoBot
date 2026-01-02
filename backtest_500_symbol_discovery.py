#!/usr/bin/env python3
"""
500-SYMBOL DISCOVERY BACKTEST
==============================
Tests 500 USDT perpetual symbols from Bybit to find profitable ones.
Uses the exact same 1H divergence logic as the live bot.

Features:
- Fetches all available USDT perpetuals from Bybit
- Tests each with optimized R:R ratios (4:1 to 10:1)
- Identifies profitable symbols
- Provides detailed statistics

Output:
- How many symbols are profitable
- Best R:R for each symbol
- Sorted by profitability
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1H
DATA_DAYS = 180   # 6 months
MAX_WAIT_CANDLES = 6
ATR_MULT = 1.0
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006
RSI_PERIOD = 14
EMA_PERIOD = 200

# R:R ratios to test
RR_OPTIONS = [4, 5, 6, 7, 8, 9, 10]

# Minimum thresholds for profitability
MIN_TRADES = 15
MIN_TOTAL_R = 0      # Must be net profitable
MIN_WIN_RATE = 10    # At least 10% WR

BASE_URL = "https://api.bybit.com"

def fetch_all_symbols(limit=500):
    """Fetch all USDT perpetual symbols from Bybit"""
    print("📥 Fetching symbol list from Bybit...")
    
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/instruments-info",
            params={'category': 'linear', 'limit': 1000},
            timeout=30
        )
        data = resp.json().get('result', {}).get('list', [])
        
        # Filter for USDT pairs only, active status
        symbols = []
        for item in data:
            symbol = item.get('symbol', '')
            status = item.get('status', '')
            if symbol.endswith('USDT') and status == 'Trading':
                symbols.append(symbol)
        
        # Sort by popularity (optional - could sort by volume)
        symbols = sorted(symbols)[:limit]
        
        print(f"✅ Found {len(symbols)} USDT perpetual symbols")
        return symbols
        
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 50
    
    while current_end > start_ts and max_iterations > 0:
        max_iterations -= 1
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
    """Find pivot highs/lows"""
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

def backtest_symbol_rr(df, rr, symbol):
    """Backtest a symbol with specific R:R"""
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
    
    for current_idx in range(50, n - 1):
        current_price = close[current_idx]
        
        # Check pending signal for BOS
        if pending_signal is not None:
            sig = pending_signal
            
            bos_confirmed = False
            if sig['side'] == 'long':
                if current_price > sig['swing']:
                    bos_confirmed = True
            else:
                if current_price < sig['swing']:
                    bos_confirmed = True
            
            if bos_confirmed:
                entry_idx = current_idx + 1
                if entry_idx < n:
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
                    
                    # Simulate trade
                    result = None
                    for k in range(entry_idx, min(entry_idx + 200, n)):
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
                    
                    # Fees
                    risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                    fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                    final_r = result - fee_cost
                    trades.append(final_r)
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        # Detect divergences
        if pending_signal is None:
            # BULLISH
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
                        
                        if signal_id not in seen_signals:
                            if current_price <= swing_high:
                                seen_signals.add(signal_id)
                                pending_signal = {'side': 'long', 'swing': swing_high}
                                pending_wait = 0
            
            # BEARISH
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
                            
                            if signal_id not in seen_signals:
                                if current_price >= swing_low:
                                    seen_signals.add(signal_id)
                                    pending_signal = {'side': 'short', 'swing': swing_low}
                                    pending_wait = 0
    
    return trades

def test_symbol(symbol):
    """Test a symbol with all R:R options and return best result"""
    try:
        df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
        
        if len(df) < 500:
            return None
        
        df = prepare_data(df)
        
        if len(df) < 400:
            return None
        
        best_result = None
        best_rr = None
        best_r = -999
        
        for rr in RR_OPTIONS:
            trades = backtest_symbol_rr(df, rr, symbol)
            
            if len(trades) >= MIN_TRADES:
                total_r = sum(trades)
                wins = sum(1 for t in trades if t > 0)
                wr = (wins / len(trades) * 100)
                
                if total_r > best_r:
                    best_r = total_r
                    best_rr = rr
                    best_result = {
                        'symbol': symbol,
                        'rr': rr,
                        'trades': len(trades),
                        'wins': wins,
                        'wr': round(wr, 1),
                        'total_r': round(total_r, 2),
                        'avg_r': round(total_r / len(trades), 3) if trades else 0
                    }
        
        return best_result
        
    except Exception as e:
        return None

def main():
    # Fetch all symbols
    symbols = fetch_all_symbols(500)
    
    if not symbols:
        print("❌ No symbols found!")
        return
    
    print(f"\n📊 Testing {len(symbols)} symbols with R:R options {RR_OPTIONS}...")
    print("="*70)
    
    results = []
    profitable = []
    unprofitable = []
    insufficient_data = []
    
    for i, symbol in enumerate(symbols):
        print(f"\r[{i+1}/{len(symbols)}] Testing {symbol:15}...", end="", flush=True)
        
        result = test_symbol(symbol)
        
        if result:
            results.append(result)
            
            if result['total_r'] > MIN_TOTAL_R and result['wr'] >= MIN_WIN_RATE:
                profitable.append(result)
                print(f" ✅ +{result['total_r']:.1f}R ({result['rr']}:1)")
            else:
                unprofitable.append(result)
                print(f" ❌ {result['total_r']:+.1f}R")
        else:
            insufficient_data.append(symbol)
            print(f" ⚪ Insufficient data")
        
        time.sleep(0.02)
    
    # === RESULTS ===
    print("\n" + "="*70)
    print("500-SYMBOL DISCOVERY RESULTS")
    print("="*70)
    
    print(f"\n📊 SUMMARY")
    print(f"├─ Symbols Tested: {len(symbols)}")
    print(f"├─ With Sufficient Data: {len(results)}")
    print(f"├─ Profitable (R > 0): {len(profitable)} ({len(profitable)/len(results)*100:.1f}%)" if results else "├─ Profitable: 0")
    print(f"├─ Unprofitable: {len(unprofitable)}")
    print(f"└─ Insufficient Data: {len(insufficient_data)}")
    
    if profitable:
        # Sort by total R
        profitable_df = pd.DataFrame(profitable).sort_values('total_r', ascending=False)
        
        total_portfolio_r = profitable_df['total_r'].sum()
        avg_r_per_symbol = profitable_df['total_r'].mean()
        
        print(f"\n💰 PROFITABLE PORTFOLIO POTENTIAL")
        print(f"├─ Total R (all profitable): {total_portfolio_r:+.1f}R")
        print(f"├─ Avg R per Symbol: {avg_r_per_symbol:+.1f}R")
        print(f"└─ Expected R/Month: {total_portfolio_r / 6:+.1f}R")
        
        print(f"\n🏆 TOP 30 PROFITABLE SYMBOLS")
        print("-"*70)
        print(f"{'Symbol':18} | {'R:R':5} | {'Trades':7} | {'WR':6} | {'Total R':10} | {'Avg R':8}")
        print("-"*70)
        
        for _, row in profitable_df.head(30).iterrows():
            print(f"{row['symbol']:18} | {row['rr']:2.0f}:1  | {row['trades']:7} | {row['wr']:5.1f}% | {row['total_r']:+9.1f}R | {row['avg_r']:+7.3f}R")
        
        print("-"*70)
        
        # R:R distribution
        print(f"\n📈 R:R DISTRIBUTION (Profitable Symbols)")
        rr_dist = profitable_df.groupby('rr').size()
        for rr, count in rr_dist.items():
            print(f"  {rr}:1 → {count} symbols")
        
        # Save results
        profitable_df.to_csv('500_symbol_profitable.csv', index=False)
        
        # Also save full results
        full_df = pd.DataFrame(results).sort_values('total_r', ascending=False)
        full_df.to_csv('500_symbol_all_results.csv', index=False)
        
        print(f"\n📁 Results saved to:")
        print(f"   - 500_symbol_profitable.csv ({len(profitable)} symbols)")
        print(f"   - 500_symbol_all_results.csv ({len(results)} symbols)")
        
        # Show symbols ready for config
        print(f"\n📋 SYMBOLS READY FOR config.yaml (Top 50):")
        print("-"*50)
        for _, row in profitable_df.head(50).iterrows():
            print(f"  {row['symbol']}: {{enabled: true, rr: {row['rr']:.1f}}}")
    
    else:
        print("\n⚠️ No profitable symbols found with current criteria")

if __name__ == "__main__":
    main()
