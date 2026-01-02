#!/usr/bin/env python3
"""
6-MONTH ROBUST BACKTEST - CURRENT 87 SYMBOLS
=============================================
Tests the EXACT configuration in config.yaml
Uses realistic bot behavior (one-per-symbol, dedup, etc)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1 hour
DATA_DAYS = 180  # 6 months
MAX_WAIT_CANDLES = 6

SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
RSI_PERIOD = 14
EMA_PERIOD = 200

# Load current config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get enabled symbols and their R:R ratios
SYMBOLS_CONFIG = {}
for sym, props in config.get('symbols', {}).items():
    if props.get('enabled', False):
        SYMBOLS_CONFIG[sym] = props.get('rr', 5.0)

SYMBOLS = list(SYMBOLS_CONFIG.keys())

print("="*70)
print("6-MONTH ROBUST BACKTEST - CURRENT BOT CONFIG")
print("="*70)
print(f"Symbols: {len(SYMBOLS)}")
print(f"Data: {DATA_DAYS} days (6 months)")
print(f"R:R Range: {min(SYMBOLS_CONFIG.values()):.0f}:1 to {max(SYMBOLS_CONFIG.values()):.0f}:1")
print("="*70)

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 20
    
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
                
            time.sleep(0.12)
            
        except Exception:
            time.sleep(0.5)
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

def realistic_backtest(df, rr, symbol):
    """
    Simulate EXACTLY how the live bot operates
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    
    trades = []
    trade_details = []  # For detailed analysis
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
                    sl_dist = entry_atr
                    
                    if sig['side'] == 'long':
                        entry_price *= (1 + SLIPPAGE_PCT)
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        entry_price *= (1 - SLIPPAGE_PCT)
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
                    result = None
                    exit_idx = None
                    for k in range(entry_idx, min(entry_idx + 200, n)):
                        if sig['side'] == 'long':
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
                        result = -0.5
                        exit_idx = min(entry_idx + 200, n - 1)
                    
                    # Apply fees
                    risk_pct = sl_dist / entry_price
                    if risk_pct > 0:
                        fee_cost = (FEE_PCT * 2) / risk_pct
                        final_r = result - fee_cost
                        trades.append(final_r)
                        trade_details.append({
                            'symbol': symbol,
                            'side': sig['side'],
                            'entry_idx': entry_idx,
                            'exit_idx': exit_idx,
                            'result': 'WIN' if final_r > 0 else 'LOSS',
                            'r_value': final_r
                        })
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        # Look for new signals (one at a time)
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
    
    return trades, trade_details

def test_symbol(symbol, rr):
    """Test a single symbol with its configured R:R"""
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 500:
        return None
    
    df = prepare_data(df)
    
    trades, details = realistic_backtest(df, rr, symbol)
    
    if len(trades) == 0:
        return None
        
    total_r = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    wr = (wins / len(trades) * 100) if trades else 0
    avg_r = total_r / len(trades) if trades else 0
    
    result = {
        'symbol': symbol,
        'rr': rr,
        'total_r': round(total_r, 2),
        'wr': round(wr, 1),
        'avg_r': round(avg_r, 3),
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins
    }
    
    return result

def main():
    all_results = []
    
    for i, sym in enumerate(SYMBOLS):
        rr = SYMBOLS_CONFIG[sym]
        print(f"\r[{i+1}/{len(SYMBOLS)}] Testing {sym:15} (R:R {rr}:1)...", end="", flush=True)
        
        result = test_symbol(sym, rr)
        if result:
            all_results.append(result)
            status = "✅" if result['total_r'] > 0 else "❌"
            print(f" {status} {result['total_r']:+.1f}R ({result['wr']:.0f}% WR, N={result['trades']})")
        else:
            print(f" ⚠️ No data/trades")
        
        time.sleep(0.15)
    
    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('6month_87_symbols_backtest.csv', index=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"6-MONTH BACKTEST RESULTS - 87 SYMBOLS")
    print(f"{'='*70}")
    
    total_r = df_results['total_r'].sum()
    total_trades = df_results['trades'].sum()
    total_wins = df_results['wins'].sum()
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    profitable_symbols = len(df_results[df_results['total_r'] > 0])
    
    print(f"Symbols Tested: {len(df_results)}")
    print(f"Profitable Symbols: {profitable_symbols} ({profitable_symbols/len(df_results)*100:.1f}%)")
    print(f"Total Trades: {total_trades}")
    print(f"Total R: {total_r:+.1f}R")
    print(f"Overall Win Rate: {overall_wr:.1f}%")
    print(f"Avg R per Trade: {total_r/total_trades:+.3f}R" if total_trades > 0 else "N/A")
    print(f"\nAnnualized: {total_r * 2:+.0f}R/year")
    
    # Top 10
    print(f"\n{'='*70}")
    print(f"TOP 10 PERFORMERS")
    print(f"{'='*70}")
    top = df_results.sort_values('total_r', ascending=False).head(10)
    for idx, row in top.iterrows():
        print(f"{row['symbol']:15} | {row['rr']:.0f}:1 R:R | {row['total_r']:+7.1f}R | WR: {row['wr']:5.1f}% | N: {int(row['trades']):3}")
    
    # Bottom 10
    print(f"\n{'='*70}")
    print(f"BOTTOM 10 PERFORMERS (Consider Blacklisting)")
    print(f"{'='*70}")
    bottom = df_results.sort_values('total_r', ascending=True).head(10)
    for idx, row in bottom.iterrows():
        print(f"{row['symbol']:15} | {row['rr']:.0f}:1 R:R | {row['total_r']:+7.1f}R | WR: {row['wr']:5.1f}% | N: {int(row['trades']):3}")
    
    # R:R distribution
    print(f"\n{'='*70}")
    print(f"PERFORMANCE BY R:R RATIO")
    print(f"{'='*70}")
    for rr in sorted(df_results['rr'].unique()):
        subset = df_results[df_results['rr'] == rr]
        print(f"{rr:.0f}:1 R:R → {len(subset)} symbols | {subset['total_r'].sum():+.1f}R total | Avg: {subset['total_r'].mean():+.1f}R/symbol")

if __name__ == "__main__":
    main()
