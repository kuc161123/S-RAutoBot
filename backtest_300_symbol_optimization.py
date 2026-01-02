#!/usr/bin/env python3
"""
300-SYMBOL OPTIMIZATION BACKTEST
================================
- 1 Year of data
- 300 USDT perpetual symbols
- R:R sweep: 4:1 to 10:1
- ATR sweep: 0.5x to 2.0x
- Find best config per symbol
- Filter: R > 10
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
TIMEFRAME = '60'  # 1 hour
DATA_DAYS = 365  # 1 year
MAX_WAIT_CANDLES = 6
TARGET_SYMBOLS = 300

# Parameter grid
RR_RATIOS = [4, 5, 6, 7, 8, 9, 10]
ATR_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
RSI_PERIOD = 14
EMA_PERIOD = 200

print(f"="*70)
print(f"300 SYMBOL OPTIMIZATION - 1 YEAR BACKTEST")
print(f"="*70)
print(f"Data: {DATA_DAYS} days (1 year)")
print(f"R:R Ratios: {RR_RATIOS}")
print(f"ATR Multipliers: {ATR_MULTS}")
print(f"Configs per symbol: {len(RR_RATIOS) * len(ATR_MULTS)}")
print(f"="*70)

def get_tradable_symbols(limit=300):
    """Fetch top tradable USDT perpetual symbols"""
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers", params={'category': 'linear'}, timeout=15)
        data = resp.json().get('result', {}).get('list', [])
        
        # Filter USDT perpetuals and sort by volume
        usdt_symbols = []
        for item in data:
            symbol = item.get('symbol', '')
            if symbol.endswith('USDT') and not symbol.endswith('USDTUSDT'):
                try:
                    volume = float(item.get('turnover24h', 0))
                    usdt_symbols.append((symbol, volume))
                except:
                    pass
        
        # Sort by volume and take top N
        usdt_symbols.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in usdt_symbols[:limit]]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 40  # More for 1 year
    
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
                
            time.sleep(0.1)
            
        except Exception:
            time.sleep(0.3)
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

def backtest_config(df, rr, atr_mult, symbol):
    """Backtest a specific R:R and ATR configuration"""
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
        
        # Check pending signal
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
                    sl_dist = entry_atr * atr_mult
                    
                    if sig['side'] == 'long':
                        entry_price *= (1 + SLIPPAGE_PCT)
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        entry_price *= (1 - SLIPPAGE_PCT)
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
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
                    
                    risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                    if risk_pct > 0:
                        fee_cost = (FEE_PCT * 2) / risk_pct
                        final_r = result - fee_cost
                        trades.append(final_r)
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        # Look for new signals
        if pending_signal is None:
            # BULLISH
            p_lows = []
            for j in range(current_idx-3, max(0, current_idx-50), -1):
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
                        signal_id = f"{symbol}_long_{curr_idx}"
                        
                        if signal_id not in seen_signals:
                            if current_price <= swing_high:
                                seen_signals.add(signal_id)
                                pending_signal = {'side': 'long', 'swing': swing_high}
                                pending_wait = 0
            
            # BEARISH 
            if pending_signal is None:
                p_highs = [] 
                for j in range(current_idx-3, max(0, current_idx-50), -1):
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
                            signal_id = f"{symbol}_short_{curr_idx_h}"
                            
                            if signal_id not in seen_signals:
                                if current_price >= swing_low:
                                    seen_signals.add(signal_id)
                                    pending_signal = {'side': 'short', 'swing': swing_low}
                                    pending_wait = 0
    
    return trades

def test_symbol(symbol):
    """Test all configurations for a symbol"""
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 1000:
        return None
    
    df = prepare_data(df)
    
    best_result = None
    best_config = None
    
    for rr in RR_RATIOS:
        for atr_mult in ATR_MULTS:
            trades = backtest_config(df, rr, atr_mult, symbol)
            
            if len(trades) < 10:  # Minimum trades
                continue
            
            total_r = sum(trades)
            wins = sum(1 for t in trades if t > 0)
            wr = (wins / len(trades) * 100) if trades else 0
            
            if best_result is None or total_r > best_result['total_r']:
                best_result = {
                    'symbol': symbol,
                    'rr': rr,
                    'atr_mult': atr_mult,
                    'total_r': round(total_r, 2),
                    'wr': round(wr, 1),
                    'trades': len(trades),
                    'wins': wins
                }
                best_config = (rr, atr_mult)
    
    return best_result

def main():
    # Get symbols
    print("\n📡 Fetching top 300 USDT perpetual symbols...")
    symbols = get_tradable_symbols(TARGET_SYMBOLS)
    print(f"✓ Found {len(symbols)} symbols")
    
    all_results = []
    
    for i, sym in enumerate(symbols):
        print(f"\r[{i+1}/{len(symbols)}] Testing {sym:15}...", end="", flush=True)
        
        try:
            result = test_symbol(sym)
            if result:
                all_results.append(result)
                if result['total_r'] > 10:
                    print(f" ✅ {result['rr']}:1 R:R, {result['atr_mult']}x ATR → {result['total_r']:+.1f}R ({result['wr']:.0f}% WR)")
                else:
                    print(f" → {result['total_r']:+.1f}R")
            else:
                print(f" ⚠️ No valid config")
        except Exception as e:
            print(f" ❌ Error: {e}")
        
        time.sleep(0.1)
    
    # Save all results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv('300_symbol_all_results.csv', index=False)
    
    # Filter for R > 10
    df_good = df_all[df_all['total_r'] > 10].sort_values('total_r', ascending=False)
    df_good.to_csv('300_symbol_r10_plus.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION RESULTS - 1 YEAR, 300 SYMBOLS")
    print(f"{'='*70}")
    print(f"Total Symbols Tested: {len(all_results)}")
    print(f"Symbols with R > 10: {len(df_good)} ({len(df_good)/len(all_results)*100:.1f}%)" if len(all_results) > 0 else "N/A")
    
    if len(df_all) > 0:
        print(f"\nOverall Stats:")
        print(f"  Total R (all): {df_all['total_r'].sum():+.1f}R")
        print(f"  Total Trades: {df_all['trades'].sum()}")
        print(f"  Avg R per Symbol: {df_all['total_r'].mean():+.1f}R")
    
    if len(df_good) > 0:
        print(f"\n{'='*70}")
        print(f"TOP 20 SYMBOLS (R > 10)")
        print(f"{'='*70}")
        for idx, row in df_good.head(20).iterrows():
            print(f"{row['symbol']:15} | {row['rr']:.0f}:1 R:R | {row['atr_mult']:.2f}x ATR | {row['total_r']:+7.1f}R | WR: {row['wr']:5.1f}% | N: {int(row['trades']):3}")
        
        print(f"\n{'='*70}")
        print(f"R:R DISTRIBUTION (for R > 10 symbols)")
        print(f"{'='*70}")
        rr_dist = df_good.groupby('rr').size()
        for rr, count in rr_dist.items():
            print(f"  {rr}:1 R:R → {count} symbols")
        
        print(f"\n{'='*70}")
        print(f"ATR DISTRIBUTION (for R > 10 symbols)")
        print(f"{'='*70}")
        atr_dist = df_good.groupby('atr_mult').size()
        for atr, count in atr_dist.items():
            print(f"  {atr:.2f}x ATR → {count} symbols")

if __name__ == "__main__":
    main()
