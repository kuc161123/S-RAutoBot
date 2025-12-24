#!/usr/bin/env python3
"""
3M ROUND 4 - Tighter SL Exploration
Testing smaller SL sizes (0.25-0.6%) with high R targets (5-8)
to find if smaller SL + bigger moves compensates for noise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
TIMEFRAME = '3'
DAYS = 30
SYMBOLS_COUNT = 50

# Parameters to test - TIGHTER SLs
SL_PCTS = [0.0025, 0.003, 0.004, 0.005, 0.006]  # 0.25%, 0.3%, 0.4%, 0.5%, 0.6%
MAX_RS = [5, 6, 7, 8]
COOLDOWNS = [3, 5, 10]  # Shorter cooldowns

# Fixed trailing logic
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1

# Fee model
TAKER_FEE = 0.00055

def load_symbols(n=50):
    from pybit.unified_trading import HTTP
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    session = HTTP(testnet=False, api_key=config.get('api_key', ''), api_secret=config.get('api_secret', ''))
    
    result = session.get_tickers(category="linear")
    tickers = result.get('result', {}).get('list', [])
    
    usdt_perps = [t for t in tickers if t['symbol'].endswith('USDT') and 'USDC' not in t['symbol']]
    sorted_tickers = sorted(usdt_perps, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    
    return [t['symbol'] for t in sorted_tickers[:n]]

def fetch_data(symbol, timeframe, days):
    from pybit.unified_trading import HTTP
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    session = HTTP(testnet=False, api_key=config.get('api_key', ''), api_secret=config.get('api_secret', ''))
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_end = end_time
    
    while current_end > start_time:
        result = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            start=start_time,
            end=current_end,
            limit=1000
        )
        
        klines = result.get('result', {}).get('list', [])
        if not klines:
            break
            
        all_data.extend(klines)
        current_end = int(klines[-1][0]) - 1
        
        if len(klines) < 1000:
            break
    
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def detect_divergences(df):
    signals = []
    lookback = 20
    
    for i in range(lookback, len(df) - 1):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        
        if pd.isna(rsi):
            continue
        
        low_window = df['low'].iloc[i-lookback:i]
        high_window = df['high'].iloc[i-lookback:i]
        
        signal_base = {'idx': i, 'entry': price}
        
        # Regular Bullish
        prev_low_idx = low_window.idxmin()
        if prev_low_idx is not None and low < low_window[prev_low_idx]:
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'type': 'regular_bullish', 'side': 'LONG'})
        
        # Regular Bearish
        prev_high_idx = high_window.idxmax()
        if prev_high_idx is not None and high > high_window[prev_high_idx]:
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'type': 'regular_bearish', 'side': 'SHORT'})
        
        # Hidden Bullish
        if low > low_window.min():
            prev_low_idx = low_window.idxmin()
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'type': 'hidden_bullish', 'side': 'LONG'})
        
        # Hidden Bearish
        if high < high_window.max():
            prev_high_idx = high_window.idxmax()
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'type': 'hidden_bearish', 'side': 'SHORT'})
    
    return signals

def simulate_trade(df, idx, side, entry, sl_pct, max_r):
    sl_distance = entry * sl_pct
    
    if side == 'LONG':
        sl = entry - sl_distance
    else:
        sl = entry + sl_distance
    
    current_sl = sl
    best_r = 0
    be_moved = False
    bars_in_trade = 0
    
    for j in range(idx + 1, min(idx + 480, len(df))):
        bars_in_trade += 1
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            if low <= current_sl:
                r = (current_sl - entry) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, bars_in_trade
            
            current_r = (high - entry) / sl_distance
            if current_r > best_r:
                best_r = current_r
                
                if best_r >= max_r:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return max_r - fee_r, bars_in_trade
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry + (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry + (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                r = (entry - current_sl) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, bars_in_trade
            
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                
                if best_r >= max_r:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return max_r - fee_r, bars_in_trade
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry - (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry - (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    final_price = df['close'].iloc[min(idx + 479, len(df) - 1)]
    if side == 'LONG':
        r = (final_price - entry) / sl_distance
    else:
        r = (entry - final_price) / sl_distance
    
    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
    return r - fee_r, bars_in_trade

def run_backtest(symbol_data, sl_pct, max_r, cooldown):
    total_r = 0
    wins = 0
    trades = 0
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        last_trade_bar = -cooldown - 1
        
        for sig in signals:
            idx = sig['idx']
            
            if idx - last_trade_bar < cooldown:
                continue
            
            r, bars = simulate_trade(df, idx, sig['side'], sig['entry'], sl_pct, max_r)
            
            total_r += r
            if r > 0:
                wins += 1
            trades += 1
            
            last_trade_bar = idx + bars
    
    wr = (wins / trades * 100) if trades > 0 else 0
    avg_r = total_r / trades if trades > 0 else 0
    fee_impact = (TAKER_FEE * 2) / sl_pct  # Fee as R-multiple
    
    return {
        'sl_pct': sl_pct,
        'max_r': max_r,
        'cooldown': cooldown,
        'trades': trades,
        'wins': wins,
        'wr': wr,
        'total_r': total_r,
        'avg_r': avg_r,
        'fee_impact': fee_impact
    }

def main():
    print("=" * 120)
    print("🔬 3M ROUND 4 - TIGHTER SL EXPLORATION")
    print("=" * 120)
    print(f"Timeframe: 3min | Symbols: {SYMBOLS_COUNT} | Days: {DAYS}")
    print(f"SL sizes: {[f'{s*100:.2f}%' for s in SL_PCTS]}")
    print(f"Max R targets: {MAX_RS}")
    print(f"Cooldowns: {COOLDOWNS}")
    
    total_configs = len(SL_PCTS) * len(MAX_RS) * len(COOLDOWNS)
    print(f"Total configurations: {total_configs}")
    print()
    
    # Load symbols
    print("📋 Loading symbols...")
    symbols = load_symbols(SYMBOLS_COUNT)
    print(f"  Loaded {len(symbols)} symbols")
    
    # Load all data and signals once
    print("\n📥 Loading data and detecting signals...")
    symbol_data = {}
    
    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(symbols)}] {symbol}")
        
        try:
            df = fetch_data(symbol, TIMEFRAME, DAYS)
            if df is None or len(df) < 100:
                continue
            
            df = calculate_indicators(df)
            signals = detect_divergences(df)
            
            if signals:
                symbol_data[symbol] = {'df': df, 'signals': signals}
        except Exception as e:
            continue
    
    print(f"\n✅ {len(symbol_data)} symbols with signals")
    
    # Run all configurations
    print("\n🔄 Testing configurations...")
    results = []
    
    configs = list(product(SL_PCTS, MAX_RS, COOLDOWNS))
    
    for i, (sl_pct, max_r, cooldown) in enumerate(configs):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(configs)}] SL={sl_pct*100:.2f}% R={max_r} CD={cooldown}")
        
        result = run_backtest(symbol_data, sl_pct, max_r, cooldown)
        results.append(result)
    
    # Sort by total R
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_r', ascending=False)
    
    print("\n" + "=" * 120)
    print("📊 TOP 20 CONFIGURATIONS (Sorted by Total R)")
    print("=" * 120)
    print(f"{'SL':<8} {'MaxR':<6} {'CD':<5} {'N':>8} {'WR':>7} {'Total_R':>10} {'Avg_R':>10} {'Fee/R':>8} {'Status':<6}")
    
    for i, row in results_df.head(20).iterrows():
        status = '✅' if row['total_r'] > 0 else '❌'
        print(f"{row['sl_pct']*100:.2f}%  {row['max_r']:<6} {row['cooldown']:<5} {row['trades']:>8} {row['wr']:>6.1f}% {row['total_r']:>+10.0f} {row['avg_r']:>+10.4f} {row['fee_impact']:>7.3f}R {status:<6}")
    
    # Summary
    profitable = results_df[results_df['total_r'] > 0]
    print(f"\n📈 Profitable configs: {len(profitable)}/{len(results_df)}")
    
    # Save results
    results_df.to_csv('3m_round4_results.csv', index=False)
    print("\n✅ Saved to 3m_round4_results.csv")
    
    # Best configuration
    if len(profitable) > 0:
        best = results_df.iloc[0]
        print("\n" + "=" * 80)
        print("🏆 BEST 3M CONFIGURATION (Round 4)")
        print("=" * 80)
        print(f"SL: {best['sl_pct']*100:.2f}%")
        print(f"Max R Target: {best['max_r']}")
        print(f"Cooldown: {best['cooldown']} bars")
        print(f"Trades: {best['trades']}")
        print(f"Win Rate: {best['wr']:.1f}%")
        print(f"Total R: {best['total_r']:+.0f}")
        print(f"Avg R/Trade: {best['avg_r']:+.4f}")
        print(f"Fee Impact: {best['fee_impact']:.3f}R per trade")
        print("\n✅ PROFITABLE CONFIGURATION FOUND!")

if __name__ == "__main__":
    main()
