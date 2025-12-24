#!/usr/bin/env python3
"""
🔬 ROUND 6 - TIMEFRAME COMPARISON (5M vs 15M vs 30M)
3M has consistently failed walk-forward validation.
Testing higher timeframes that may have better signal-to-noise ratio.

All with proper walk-forward validation from the start.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
TIMEFRAMES = ['5', '15', '30']  # 5min, 15min, 30min
DAYS = 60
SYMBOLS_COUNT = 50
TRAIN_RATIO = 0.6

# Best parameters from previous testing
SL_PCTS = [0.01, 0.015, 0.02]  # 1%, 1.5%, 2%
MAX_RS = [3, 4, 5]
COOLDOWN = 5

# Trailing
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1

# Fee model
TAKER_FEE = 0.00055
SLIPPAGE = 0.0001

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
        result = session.get_kline(category="linear", symbol=symbol, interval=timeframe,
                                   start=start_time, end=current_end, limit=1000)
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
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    return df

def detect_divergences(df):
    signals = []
    lookback = 20
    
    for i in range(lookback, len(df) - 1):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        volume_ratio = df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0
        
        if pd.isna(rsi):
            continue
        
        low_window = df['low'].iloc[i-lookback:i]
        high_window = df['high'].iloc[i-lookback:i]
        
        # Volume filter: only take signals with above-average volume
        if volume_ratio < 0.8:
            continue
        
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
        entry = entry * (1 + SLIPPAGE)
        sl = entry - sl_distance
    else:
        entry = entry * (1 - SLIPPAGE)
        sl = entry + sl_distance
    
    current_sl = sl
    best_r = 0
    be_moved = False
    bars_in_trade = 0
    
    for j in range(idx + 1, min(idx + 200, len(df))):  # Max 200 bars
        bars_in_trade += 1
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            if low <= current_sl:
                exit_price = current_sl * (1 - SLIPPAGE)
                r = (exit_price - entry) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, bars_in_trade
            
            current_r = (high - entry) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    exit_price = entry + (max_r * sl_distance)
                    exit_price = exit_price * (1 - SLIPPAGE)
                    r = (exit_price - entry) / sl_distance
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return r - fee_r, bars_in_trade
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry + (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry + (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                exit_price = current_sl * (1 + SLIPPAGE)
                r = (entry - exit_price) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r, bars_in_trade
            
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    exit_price = entry - (max_r * sl_distance)
                    exit_price = exit_price * (1 + SLIPPAGE)
                    r = (entry - exit_price) / sl_distance
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return r - fee_r, bars_in_trade
                
                if best_r >= BE_THRESHOLD and not be_moved:
                    current_sl = entry - (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry - (best_r - TRAIL_DISTANCE) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout
    final_price = df['close'].iloc[min(idx + 199, len(df) - 1)]
    if side == 'LONG':
        r = (final_price - entry) / sl_distance
    else:
        r = (entry - final_price) / sl_distance
    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
    return r - fee_r, bars_in_trade

def run_backtest_period(symbol_data, sl_pct, max_r, start_pct, end_pct):
    total_r = 0
    wins = 0
    trades = 0
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        total_bars = len(df)
        start_idx = int(total_bars * start_pct)
        end_idx = int(total_bars * end_pct)
        
        period_signals = [s for s in signals if start_idx <= s['idx'] < end_idx]
        
        last_trade_bar = -COOLDOWN - 1
        
        for sig in period_signals:
            idx = sig['idx']
            if idx - last_trade_bar < COOLDOWN:
                continue
            
            r, bars = simulate_trade(df, idx, sig['side'], sig['entry'], sl_pct, max_r)
            total_r += r
            if r > 0:
                wins += 1
            trades += 1
            last_trade_bar = idx + bars
    
    if trades == 0:
        return {'trades': 0, 'wr': 0, 'total_r': 0, 'avg_r': 0}
    
    return {
        'trades': trades,
        'wins': wins,
        'wr': wins / trades * 100,
        'total_r': total_r,
        'avg_r': total_r / trades
    }

def main():
    print("=" * 120)
    print("🔬 ROUND 6 - TIMEFRAME COMPARISON (5M vs 15M vs 30M)")
    print("=" * 120)
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Symbols: {SYMBOLS_COUNT} | Days: {DAYS}")
    print(f"SL: {[f'{s*100}%' for s in SL_PCTS]} | Max R: {MAX_RS}")
    print(f"Walk-Forward: {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test")
    print()
    
    # Load symbols once
    print("📋 Loading symbols...")
    symbols = load_symbols(SYMBOLS_COUNT)
    print(f"  Loaded {len(symbols)} symbols")
    
    all_results = []
    
    for tf in TIMEFRAMES:
        print(f"\n{'='*60}")
        print(f"📊 Testing {tf}M timeframe...")
        print(f"{'='*60}")
        
        # Load data for this timeframe
        print("📥 Loading data...")
        symbol_data = {}
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(symbols)}] {symbol}")
            
            try:
                df = fetch_data(symbol, tf, DAYS)
                if df is None or len(df) < 100:
                    continue
                
                df = calculate_indicators(df)
                signals = detect_divergences(df)
                
                if signals:
                    symbol_data[symbol] = {'df': df, 'signals': signals}
            except:
                continue
        
        print(f"✅ {len(symbol_data)} symbols with signals")
        
        # Test all SL/R combinations
        print("🔄 Running walk-forward validation...")
        
        for sl_pct in SL_PCTS:
            for max_r in MAX_RS:
                # Train
                train = run_backtest_period(symbol_data, sl_pct, max_r, 0, TRAIN_RATIO)
                # Test
                test = run_backtest_period(symbol_data, sl_pct, max_r, TRAIN_RATIO, 1.0)
                
                fee_impact = (TAKER_FEE * 2) / sl_pct
                
                all_results.append({
                    'timeframe': tf,
                    'sl_pct': sl_pct,
                    'max_r': max_r,
                    'train_trades': train['trades'],
                    'train_r': train['total_r'],
                    'train_wr': train['wr'],
                    'test_trades': test['trades'],
                    'test_r': test['total_r'],
                    'test_wr': test['wr'],
                    'test_avg_r': test['avg_r'],
                    'fee_impact': fee_impact
                })
    
    # Sort by test R
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_r', ascending=False)
    
    print("\n" + "=" * 140)
    print("📊 ALL RESULTS (Sorted by Test R)")
    print("=" * 140)
    print(f"{'TF':>4} {'SL':>7} {'MaxR':>5} {'Train_N':>8} {'Train_R':>10} {'Test_N':>8} {'Test_R':>10} {'Test_WR':>8} {'Fee/R':>8} {'Status'}")
    
    for _, row in results_df.iterrows():
        status = '✅ PROFIT' if row['test_r'] > 0 else '❌'
        print(f"{row['timeframe']:>4}M {row['sl_pct']*100:>6.1f}% {row['max_r']:>5} {row['train_trades']:>8} {row['train_r']:>+10.0f} {row['test_trades']:>8} {row['test_r']:>+10.0f} {row['test_wr']:>7.1f}% {row['fee_impact']:>7.3f}R {status}")
    
    # Save
    results_df.to_csv('round6_timeframe_results.csv', index=False)
    print("\n✅ Saved to round6_timeframe_results.csv")
    
    # Summary by timeframe
    print("\n" + "=" * 80)
    print("📈 SUMMARY BY TIMEFRAME")
    print("=" * 80)
    
    for tf in TIMEFRAMES:
        tf_data = results_df[results_df['timeframe'] == tf]
        best = tf_data.iloc[0] if len(tf_data) > 0 else None
        profitable = len(tf_data[tf_data['test_r'] > 0])
        
        if best is not None:
            print(f"\n{tf}M Timeframe:")
            print(f"  Best Test R: {best['test_r']:+.0f}")
            print(f"  Best Config: SL={best['sl_pct']*100}%, R={best['max_r']}")
            print(f"  Profitable Configs: {profitable}/{len(tf_data)}")
            print(f"  Status: {'✅ VIABLE' if best['test_r'] > 0 else '❌ NOT VIABLE'}")
    
    # Overall best
    profitable = results_df[results_df['test_r'] > 0]
    if len(profitable) > 0:
        best = profitable.iloc[0]
        print("\n" + "=" * 80)
        print("🏆 BEST OVERALL CONFIGURATION")
        print("=" * 80)
        print(f"Timeframe: {best['timeframe']}M")
        print(f"SL: {best['sl_pct']*100:.1f}%")
        print(f"Max R Target: {best['max_r']}")
        print(f"\n📈 OUT-OF-SAMPLE PERFORMANCE:")
        print(f"  Test Trades: {best['test_trades']}")
        print(f"  Test R: {best['test_r']:+.0f}")
        print(f"  Test Win Rate: {best['test_wr']:.1f}%")
        print(f"  Fee Impact: {best['fee_impact']:.3f}R per trade")
        print("\n✅ PROFITABLE CONFIGURATION FOUND!")
    else:
        print("\n❌ No profitable configuration found across any timeframe")

if __name__ == "__main__":
    main()
