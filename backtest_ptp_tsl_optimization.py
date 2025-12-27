import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures
from datetime import datetime, timedelta
import itertools

# --- CONFIGURATION ---
DAYS = 200
SYMBOL_COUNT = 20
TIMEFRAME = '5'
RSI_PERIOD = 14
DIVERGENCE_LOOKBACK = 10
MAX_WAIT_CANDLES = 10
SL_ATR_MULT = 0.8
FEE = 0.0006
SLIPPAGE = 0.0003

# Optimization Grid
PTP_LEVELS = [0.5, 0.8, 1.0, 1.5]  # Take 50% profit at XR
TSL_BE_TRIGGERS = [0.5, 0.75, 1.0, 1.5]  # Move to Break Even at XR
FINAL_RR_LEVELS = [2.0, 3.0, 4.0]

def get_top_symbols(limit=20):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp['result']['list']
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_tickers.sort(key=lambda x: float(x['turnover24h']), reverse=True)
        return [t['symbol'] for t in usdt_tickers[:limit]]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    url = "https://api.bybit.com/v5/market/kline"
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    all_klines = []
    current_end = end_time
    
    while current_end > start_time:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "end": current_end
        }
        try:
            resp = requests.get(url, params=params, timeout=10).json()
            if resp['retCode'] != 0 or not resp['result']['list']:
                break
            klines = resp['result']['list']
            all_klines.extend(klines)
            current_end = int(klines[-1][0]) - 1
            if len(klines) < 1000:
                break
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            break
            
    if not all_klines:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = df[col].astype(float)
    return df

def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Swings for Divergence
    df['swing_low'] = df['low'].rolling(window=DIVERGENCE_LOOKBACK).min()
    df['swing_high'] = df['high'].rolling(window=DIVERGENCE_LOOKBACK).max()
    df['rsi_low'] = df['rsi'].rolling(window=DIVERGENCE_LOOKBACK).min()
    df['rsi_high'] = df['rsi'].rolling(window=DIVERGENCE_LOOKBACK).max()
    
    return df

def detect_signals(df):
    signals = []
    for i in range(DIVERGENCE_LOOKBACK, len(df)):
        # Bullish Divergence
        if df['low'].iloc[i] <= df['swing_low'].iloc[i-1] and df['rsi'].iloc[i] > df['rsi_low'].iloc[i-1] and df['rsi'].iloc[i] < 30:
            signals.append({'index': i, 'type': 'bullish', 'price': df['close'].iloc[i], 'swing': df['swing_high'].iloc[i-1]})
            
        # Bearish Divergence
        elif df['high'].iloc[i] >= df['swing_high'].iloc[i-1] and df['rsi'].iloc[i] < df['rsi_high'].iloc[i-1] and df['rsi'].iloc[i] > 70:
            signals.append({'index': i, 'type': 'bearish', 'price': df['close'].iloc[i], 'swing': df['swing_low'].iloc[i-1]})
            
    return signals

def backtest_strategy(df, signals, ptp_r, tsl_be_r, final_rr):
    trades = []
    for sig in signals:
        start_idx = sig['index']
        sig_type = sig['type']
        trigger_price = sig['swing']
        
        # Wait for Structure Break
        entry_idx = -1
        for j in range(start_idx + 1, min(start_idx + MAX_WAIT_CANDLES + 1, len(df))):
            if sig_type == 'bullish' and df['close'].iloc[j] > trigger_price:
                entry_idx = j + 1 # Entry at next candle open
                break
            elif sig_type == 'bearish' and df['close'].iloc[j] < trigger_price:
                entry_idx = j + 1 # Entry at next candle open
                break
        
        if entry_idx == -1 or entry_idx >= len(df):
            continue
            
        entry_price = df['open'].iloc[entry_idx]
        if sig_type == 'bullish':
            entry_price *= (1 + SLIPPAGE)
        else:
            entry_price *= (1 - SLIPPAGE)
            
        atr = df['atr'].iloc[entry_idx - 1]
        sl_dist = atr * SL_ATR_MULT
        
        if sig_type == 'bullish':
            initial_sl = entry_price - sl_dist
            p1_price = entry_price + (sl_dist * ptp_r)
            be_trigger = entry_price + (sl_dist * tsl_be_r)
            final_tp = entry_price + (sl_dist * final_rr)
        else:
            initial_sl = entry_price + sl_dist
            p1_price = entry_price - (sl_dist * ptp_r)
            be_trigger = entry_price - (sl_dist * tsl_be_r)
            final_tp = entry_price - (sl_dist * final_rr)
            
        current_sl = initial_sl
        portion1_closed = False
        portion1_r = 0
        portion2_r = 0
        
        for k in range(entry_idx, len(df)):
            low = df['low'].iloc[k]
            high = df['high'].iloc[k]
            
            # Check for PTP
            if not portion1_closed:
                if (sig_type == 'bullish' and high >= p1_price) or (sig_type == 'bearish' and low <= p1_price):
                    portion1_closed = True
                    portion1_r = ptp_r - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    # Note: Simplified fee calc in R terms
            
            # Check for TSL / BE
            if (sig_type == 'bullish' and high >= be_trigger) or (sig_type == 'bearish' and low <= be_trigger):
                current_sl = entry_price
                
            # Check for Exit Part 1 (if not hit PTP yet) or Part 2
            if sig_type == 'bullish':
                # SL hit
                if low <= current_sl:
                    if not portion1_closed:
                        portion1_r = -1.0 - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                        portion2_r = -1.0 - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    else:
                        # Portion 2 stopped at BE
                        portion2_r = 0.0 - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    break
                # TP hit
                if high >= final_tp:
                    if not portion1_closed:
                        portion1_r = final_rr - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    portion2_r = final_rr - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    break
            else: # Bearish
                # SL hit
                if high >= current_sl:
                    if not portion1_closed:
                        portion1_r = -1.0 - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                        portion2_r = -1.0 - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    else:
                        # Portion 2 stopped at BE
                        portion2_r = 0.0 - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    break
                # TP hit
                if low <= final_tp:
                    if not portion1_closed:
                        portion1_r = final_rr - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    portion2_r = final_rr - (FEE + SLIPPAGE) * (entry_price / sl_dist)
                    break
                    
        total_r = (portion1_r + portion2_r) / 2.0
        trades.append(total_r)
        
    return trades

def run_for_symbol(symbol, combo):
    ptp_r, tsl_be_r, final_rr = combo
    df = fetch_klines(symbol, TIMEFRAME, DAYS)
    if df.empty: return []
    df = calculate_indicators(df)
    signals = detect_signals(df)
    results = backtest_strategy(df, signals, ptp_r, tsl_be_r, final_rr)
    return results

def main():
    symbols = get_top_symbols(SYMBOL_COUNT)
    print(f"Testing {len(symbols)} symbols...")
    
    # Pre-fetch data to avoid redundant API calls in loop
    symbol_data = {}
    for sym in symbols:
        print(f"Fetching {sym}...")
        df = fetch_klines(sym, TIMEFRAME, DAYS)
        if not df.empty:
            df = calculate_indicators(df)
            signals = detect_signals(df)
            symbol_data[sym] = (df, signals)
            
    combos = list(itertools.product(PTP_LEVELS, TSL_BE_TRIGGERS, FINAL_RR_LEVELS))
    
    print(f"Running {len(combos)} combinations...")
    
    all_results = []
    
    for combo in combos:
        ptp_r, tsl_be_r, final_rr = combo
        combo_trades = []
        for sym, (df, signals) in symbol_data.items():
            trades = backtest_strategy(df, signals, ptp_r, tsl_be_r, final_rr)
            combo_trades.extend(trades)
            
        if combo_trades:
            total_r = sum(combo_trades)
            avg_r = total_r / len(combo_trades)
            win_rate = len([t for t in combo_trades if t > 0]) / len(combo_trades)
            all_results.append({
                'combo': combo,
                'total_r': total_r,
                'avg_r': avg_r,
                'win_rate': win_rate,
                'trades': len(combo_trades)
            })
            
    # Sort by total_r
    all_results.sort(key=lambda x: x['total_r'], reverse=True)
    
    print("\n--- RESULTS (Top 10) ---")
    header = f"{'PTP':<5} {'BE':<5} {'FINAL':<6} | {'WR':<8} {'Avg R':<10} {'Total R':<10} {'N':<6}"
    print(header)
    print("-" * len(header))
    for res in all_results[:10]:
        c = res['combo']
        print(f"{c[0]:<5} {c[1]:<5} {c[2]:<6} | {res['win_rate']:>7.2%} {res['avg_r']:>9.4f} {res['total_r']:>9.2f} {res['trades']:>6}")

if __name__ == "__main__":
    main()
