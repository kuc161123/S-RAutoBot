#!/usr/bin/env python3
"""
Multi-Timeframe Backtest Analysis
==================================

Tests Fixed 4R TP strategy across multiple timeframes:
- 3M, 5M, 15M, 30M

Uses walk-forward validation to find the most robust timeframe.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import yaml
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

TIMEFRAMES = [3, 5, 15, 30]  # Minutes
SL_PCT = 1.2  # Fixed 1.2% stop loss  
TP_R = 4.0  # Fixed 4R take profit (winner from previous test)
COOLDOWN_BARS = 3
DATA_DAYS = 90
WF_SPLITS = 5

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch klines from Bybit"""
    from pybit.unified_trading import HTTP
    
    session = HTTP(testnet=False)
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_end = end_time
    
    while current_end > start_time:
        try:
            resp = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=start_time,
                end=current_end,
                limit=1000
            )
            klines = resp.get("result", {}).get("list", [])
            if not klines:
                break
            all_klines.extend(klines)
            current_end = int(klines[-1][0]) - 1
        except:
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_divergences(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish and bearish divergences"""
    df = df.copy()
    df['rsi'] = calculate_rsi(df)
    
    lookback = 5
    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_high'] = df['rsi'].rolling(lookback).max()
    
    df['bullish_div'] = (
        (df['low'] <= df['price_low']) & 
        (df['rsi'] > df['rsi_low'].shift(lookback)) &
        (df['rsi'] < 40)
    )
    
    df['bearish_div'] = (
        (df['high'] >= df['price_high']) &
        (df['rsi'] < df['rsi_high'].shift(lookback)) &
        (df['rsi'] > 60)
    )
    
    return df

def simulate_trade(df: pd.DataFrame, idx: int, side: str, entry: float,
                   sl_distance: float) -> tuple:
    """Simulate trade with fixed 4R TP"""
    tp_price = entry + (TP_R * sl_distance) if side == 'long' else entry - (TP_R * sl_distance)
    sl_price = entry - sl_distance if side == 'long' else entry + sl_distance
    
    for i in range(idx + 1, min(idx + 1000, len(df))):
        candle = df.iloc[i]
        bars_held = i - idx
        
        if side == 'long':
            if candle['low'] <= sl_price:
                return (-1.0, "SL", bars_held)
            if candle['high'] >= tp_price:
                return (TP_R, "TP", bars_held)
        else:
            if candle['high'] >= sl_price:
                return (-1.0, "SL", bars_held)
            if candle['low'] <= tp_price:
                return (TP_R, "TP", bars_held)
    
    final_candle = df.iloc[min(idx + 999, len(df) - 1)]
    if side == 'long':
        exit_r = (final_candle['close'] - entry) / sl_distance
    else:
        exit_r = (entry - final_candle['close']) / sl_distance
    
    return (exit_r, "TIMEOUT", 1000)

def backtest_symbol(symbol: str, df: pd.DataFrame) -> list:
    """Run backtest for a single symbol"""
    trades = []
    
    try:
        if len(df) < 100:
            return trades
        
        df = detect_divergences(df)
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ok'] = df['volume'] > df['volume_sma']
        
        last_trade_idx = -COOLDOWN_BARS
        
        for idx in range(50, len(df) - 10):
            if idx - last_trade_idx < COOLDOWN_BARS:
                continue
            
            row = df.iloc[idx]
            
            side = None
            if row['bullish_div'] and row['volume_ok']:
                side = 'long'
            elif row['bearish_div'] and row['volume_ok']:
                side = 'short'
            
            if side is None:
                continue
            
            entry = row['close']
            sl_distance = entry * (SL_PCT / 100)
            
            exit_r, exit_reason, bars_held = simulate_trade(df, idx, side, entry, sl_distance)
            
            trades.append({
                'symbol': symbol,
                'side': side,
                'exit_r': exit_r,
                'exit_reason': exit_reason,
                'bars_held': bars_held,
                'timestamp': row['timestamp']
            })
            
            last_trade_idx = idx
            
    except:
        pass
    
    return trades

def run_timeframe_test(symbols: list, timeframe: int) -> dict:
    """Run backtest for a single timeframe"""
    print(f"\n{'='*60}")
    print(f"TESTING {timeframe}M TIMEFRAME")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"Fetching {timeframe}M data for {len(symbols)} symbols...")
    symbol_data = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_klines, sym, str(timeframe), DATA_DAYS): sym 
                   for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                df = future.result()
                if len(df) >= 100:
                    symbol_data[sym] = df
            except:
                pass
    
    print(f"Loaded data for {len(symbol_data)} symbols")
    
    # Walk-forward validation
    all_trades = []
    wf_results = []
    
    for fold in range(WF_SPLITS):
        fold_trades = []
        
        for sym, df in symbol_data.items():
            fold_size = len(df) // WF_SPLITS
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < WF_SPLITS - 1 else len(df)
            
            fold_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            trades = backtest_symbol(sym, fold_df)
            fold_trades.extend(trades)
        
        all_trades.extend(fold_trades)
        
        if fold_trades:
            trades_df = pd.DataFrame(fold_trades)
            total_r = trades_df['exit_r'].sum()
            wins = (trades_df['exit_r'] > 0).sum()
            total = len(trades_df)
            wr = wins / total * 100 if total > 0 else 0
            wf_results.append({'fold': fold + 1, 'trades': total, 'total_r': total_r, 'wr': wr})
            print(f"  Fold {fold+1}: {total} trades, {total_r:+.1f}R, {wr:.1f}% WR")
    
    # Overall stats
    if all_trades:
        all_df = pd.DataFrame(all_trades)
        total_r = all_df['exit_r'].sum()
        wins = (all_df['exit_r'] > 0).sum()
        total = len(all_df)
        wr = wins / total * 100 if total > 0 else 0
        avg_r = all_df['exit_r'].mean()
        profitable_folds = sum(1 for r in wf_results if r['total_r'] > 0)
        
        # Calculate avg trade duration in real time
        avg_bars = all_df['bars_held'].mean()
        avg_mins = avg_bars * timeframe
        
        return {
            'timeframe': f"{timeframe}M",
            'total_r': round(total_r, 1),
            'trades': total,
            'wr': round(wr, 1),
            'avg_r': round(avg_r, 3),
            'profitable_folds': profitable_folds,
            'wf_score': f"{profitable_folds}/{WF_SPLITS}",
            'avg_bars': round(avg_bars, 1),
            'avg_duration_mins': round(avg_mins, 1),
            'symbols': len(symbol_data),
            'trades_per_day': round(total / DATA_DAYS, 1)
        }
    
    return {
        'timeframe': f"{timeframe}M",
        'total_r': 0,
        'trades': 0,
        'wr': 0,
        'avg_r': 0,
        'profitable_folds': 0,
        'wf_score': '0/5',
        'avg_bars': 0,
        'avg_duration_mins': 0,
        'symbols': 0,
        'trades_per_day': 0
    }

def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME BACKTEST ANALYSIS")
    print("=" * 70)
    print(f"Strategy: Fixed 4R TP | SL: {SL_PCT}% | Cooldown: {COOLDOWN_BARS} bars")
    print(f"Data: {DATA_DAYS} days | Walk-Forward: {WF_SPLITS} splits")
    print(f"Timeframes to test: {TIMEFRAMES}")
    print("=" * 70)
    
    # Load symbols
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config.get('trade', {}).get('divergence_symbols', [])
    except:
        symbols = []
    
    if not symbols:
        print("ERROR: No symbols found!")
        return
    
    print(f"\nLoaded {len(symbols)} symbols")
    
    # Test each timeframe
    results = []
    for tf in TIMEFRAMES:
        result = run_timeframe_test(symbols, tf)
        results.append(result)
        
        print(f"\n📊 {tf}M Summary:")
        print(f"   Total R: {result['total_r']:+.1f} | Trades: {result['trades']} | WR: {result['wr']}%")
        print(f"   Walk-Forward: {result['wf_score']} | Trades/Day: {result['trades_per_day']}")
    
    # Final comparison
    results_df = pd.DataFrame(results).sort_values('total_r', ascending=False)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS - RANKED BY TOTAL R")
    print("=" * 70)
    
    for i, row in results_df.iterrows():
        emoji = "🥇" if row['total_r'] == results_df['total_r'].max() else "  "
        print(f"\n{emoji} {row['timeframe']} TIMEFRAME")
        print(f"   Total R: {row['total_r']:+.1f}")
        print(f"   Trades: {row['trades']} ({row['trades_per_day']}/day)")
        print(f"   WR: {row['wr']}% | Avg R: {row['avg_r']}")
        print(f"   Walk-Forward: {row['wf_score']} profitable")
        print(f"   Avg Duration: {row['avg_duration_mins']} mins ({row['avg_bars']} bars)")
    
    results_df.to_csv('multi_timeframe_results.csv', index=False)
    print(f"\n📁 Results saved to multi_timeframe_results.csv")
    
    # Recommendation
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATION")
    print("=" * 70)
    print(f"Best Timeframe: {best['timeframe']}")
    print(f"Expected: {best['total_r']:+.1f}R over {best['trades']} trades")
    print(f"Trade Frequency: {best['trades_per_day']} trades/day")
    print(f"Walk-Forward: {best['wf_score']} profitable")

if __name__ == "__main__":
    main()
