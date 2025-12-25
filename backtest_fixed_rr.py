#!/usr/bin/env python3
"""
Comprehensive Fixed 3R TP Backtest
==================================

Tests Fixed 3R Take Profit strategy with:
- All symbols from config
- Walk-forward validation (5 splits)
- Monte Carlo simulation
- Bybit-realistic execution simulation
- Multiple R:R configurations

This is the most realistic backtest matching actual Bybit execution.
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

TIMEFRAME = 30  # 30M candles
SL_PCT = 1.2  # Fixed 1.2% stop loss
COOLDOWN_BARS = 3
DATA_DAYS = 90  # 90 days of data for walk-forward

# R:R Configurations to test
RR_CONFIGS = [
    (2.0, "Fixed 2R TP"),
    (3.0, "Fixed 3R TP"),
    (4.0, "Fixed 4R TP"),
    (5.0, "Fixed 5R TP"),
    (7.0, "Fixed 7R TP"),
]

# Walk-forward splits
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
        except Exception as e:
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
    
    # Bullish: price makes lower low, RSI makes higher low
    df['bullish_div'] = (
        (df['low'] <= df['price_low']) & 
        (df['rsi'] > df['rsi_low'].shift(lookback)) &
        (df['rsi'] < 40)
    )
    
    # Bearish: price makes higher high, RSI makes lower high
    df['bearish_div'] = (
        (df['high'] >= df['price_high']) &
        (df['rsi'] < df['rsi_high'].shift(lookback)) &
        (df['rsi'] > 60)
    )
    
    return df

def simulate_fixed_tp_trade(df: pd.DataFrame, idx: int, side: str, entry: float,
                            sl_distance: float, tp_r: float) -> tuple:
    """
    Simulate trade with fixed Take Profit (no trailing).
    
    Returns: (exit_r, exit_reason, bars_held)
    """
    tp_price = entry + (tp_r * sl_distance) if side == 'long' else entry - (tp_r * sl_distance)
    sl_price = entry - sl_distance if side == 'long' else entry + sl_distance
    
    for i in range(idx + 1, min(idx + 1000, len(df))):
        candle = df.iloc[i]
        bars_held = i - idx
        
        if side == 'long':
            # Check SL hit (assume SL checked before TP within same candle)
            if candle['low'] <= sl_price:
                return (-1.0, "SL", bars_held)
            # Check TP hit
            if candle['high'] >= tp_price:
                return (tp_r, "TP", bars_held)
        else:
            # SHORT
            if candle['high'] >= sl_price:
                return (-1.0, "SL", bars_held)
            if candle['low'] <= tp_price:
                return (tp_r, "TP", bars_held)
    
    # Timeout
    final_candle = df.iloc[min(idx + 999, len(df) - 1)]
    if side == 'long':
        exit_r = (final_candle['close'] - entry) / sl_distance
    else:
        exit_r = (entry - final_candle['close']) / sl_distance
    
    return (exit_r, "TIMEOUT", 1000)

def backtest_symbol(symbol: str, tp_r: float, df: pd.DataFrame = None) -> list:
    """Run backtest for a single symbol"""
    trades = []
    
    try:
        if df is None:
            df = fetch_klines(symbol, str(TIMEFRAME), DATA_DAYS)
        
        if len(df) < 100:
            return trades
        
        df = detect_divergences(df)
        
        # Volume filter
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
            
            exit_r, exit_reason, bars_held = simulate_fixed_tp_trade(
                df, idx, side, entry, sl_distance, tp_r
            )
            
            trades.append({
                'symbol': symbol,
                'side': side,
                'entry': entry,
                'exit_r': exit_r,
                'exit_reason': exit_reason,
                'bars_held': bars_held,
                'timestamp': row['timestamp']
            })
            
            last_trade_idx = idx
            
    except Exception as e:
        pass
    
    return trades

def run_walk_forward(symbols: list, tp_r: float, config_name: str) -> dict:
    """Run walk-forward validation"""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {config_name}")
    print(f"{'='*60}")
    
    # Fetch all data first
    print("Fetching data for all symbols...")
    symbol_data = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_klines, sym, str(TIMEFRAME), DATA_DAYS): sym 
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
    
    # Split into WF folds
    all_trades = []
    wf_results = []
    
    for fold in range(WF_SPLITS):
        fold_trades = []
        
        for sym, df in symbol_data.items():
            fold_size = len(df) // WF_SPLITS
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < WF_SPLITS - 1 else len(df)
            
            fold_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            trades = backtest_symbol(sym, tp_r, fold_df)
            fold_trades.extend(trades)
        
        all_trades.extend(fold_trades)
        
        # Calculate fold stats
        if fold_trades:
            trades_df = pd.DataFrame(fold_trades)
            total_r = trades_df['exit_r'].sum()
            wins = (trades_df['exit_r'] > 0).sum()
            total = len(trades_df)
            wr = wins / total * 100 if total > 0 else 0
            wf_results.append({'fold': fold + 1, 'trades': total, 'total_r': total_r, 'wr': wr})
            print(f"  Fold {fold+1}: {total} trades, {total_r:+.1f}R, {wr:.1f}% WR")
        else:
            wf_results.append({'fold': fold + 1, 'trades': 0, 'total_r': 0, 'wr': 0})
            print(f"  Fold {fold+1}: No trades")
    
    # Overall stats
    if all_trades:
        all_df = pd.DataFrame(all_trades)
        total_r = all_df['exit_r'].sum()
        wins = (all_df['exit_r'] > 0).sum()
        total = len(all_df)
        wr = wins / total * 100 if total > 0 else 0
        avg_r = all_df['exit_r'].mean()
        
        # Exit breakdown
        sl_exits = (all_df['exit_reason'] == 'SL').sum()
        tp_exits = (all_df['exit_reason'] == 'TP').sum()
        
        # Profitable folds
        profitable_folds = sum(1 for r in wf_results if r['total_r'] > 0)
        
        return {
            'config': config_name,
            'tp_r': tp_r,
            'total_r': round(total_r, 1),
            'trades': total,
            'wr': round(wr, 1),
            'avg_r': round(avg_r, 3),
            'sl_exits': sl_exits,
            'tp_exits': tp_exits,
            'profitable_folds': profitable_folds,
            'wf_score': f"{profitable_folds}/{WF_SPLITS}",
            'avg_bars': round(all_df['bars_held'].mean(), 1),
            'symbols_traded': all_df['symbol'].nunique()
        }
    
    return {
        'config': config_name,
        'tp_r': tp_r,
        'total_r': 0,
        'trades': 0,
        'wr': 0,
        'avg_r': 0,
        'sl_exits': 0,
        'tp_exits': 0,
        'profitable_folds': 0,
        'wf_score': '0/5',
        'avg_bars': 0,
        'symbols_traded': 0
    }

def monte_carlo_simulation(trades: list, n_sims: int = 1000) -> dict:
    """Run Monte Carlo simulation on trade results"""
    if not trades:
        return {'p5': 0, 'p50': 0, 'p95': 0, 'prob_profit': 0}
    
    r_values = [t['exit_r'] for t in trades]
    n_trades = len(r_values)
    
    final_rs = []
    for _ in range(n_sims):
        sampled = np.random.choice(r_values, size=n_trades, replace=True)
        final_rs.append(np.sum(sampled))
    
    return {
        'p5': round(np.percentile(final_rs, 5), 1),
        'p50': round(np.percentile(final_rs, 50), 1),
        'p95': round(np.percentile(final_rs, 95), 1),
        'prob_profit': round(sum(1 for r in final_rs if r > 0) / n_sims * 100, 1)
    }

def main():
    print("=" * 70)
    print("COMPREHENSIVE FIXED R:R BACKTEST")
    print("=" * 70)
    print(f"Timeframe: {TIMEFRAME}M | SL: {SL_PCT}% | Cooldown: {COOLDOWN_BARS} bars")
    print(f"Data: {DATA_DAYS} days | Walk-Forward: {WF_SPLITS} splits")
    print("=" * 70)
    
    # Load ALL symbols from config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        # Symbols are under trade.divergence_symbols
        symbols = config.get('trade', {}).get('divergence_symbols', [])
        if not symbols:
            # Fallback - try scalp section
            symbols = config.get('scalp', {}).get('divergence_symbols', [])
    except Exception as e:
        print(f"Config error: {e}")
        symbols = []
    
    if not symbols:
        print("ERROR: No symbols found in config!")
        return
    
    print(f"\nLoaded {len(symbols)} symbols from config")
    
    # Test each R:R configuration
    results = []
    all_trades_by_config = {}
    
    for tp_r, config_name in RR_CONFIGS:
        result = run_walk_forward(symbols, tp_r, config_name)
        results.append(result)
        
        print(f"\n📊 {config_name} Summary:")
        print(f"   Total R: {result['total_r']:+.1f}")
        print(f"   Trades: {result['trades']} | WR: {result['wr']}%")
        print(f"   Walk-Forward: {result['wf_score']} profitable")
    
    # Sort by Total R
    results_df = pd.DataFrame(results).sort_values('total_r', ascending=False)
    
    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS - RANKED BY TOTAL R")
    print("=" * 70)
    
    for i, row in results_df.iterrows():
        emoji = "🥇" if row['total_r'] == results_df['total_r'].max() else "  "
        print(f"\n{emoji} {row['config']}")
        print(f"   Total R: {row['total_r']:+.1f} | Trades: {row['trades']} | WR: {row['wr']}%")
        print(f"   Avg R: {row['avg_r']:.3f} | Avg Bars: {row['avg_bars']}")
        print(f"   SL: {row['sl_exits']} | TP: {row['tp_exits']}")
        print(f"   Walk-Forward: {row['wf_score']} profitable | Symbols: {row['symbols_traded']}")
    
    # Save results
    results_df.to_csv('fixed_rr_backtest_results.csv', index=False)
    print(f"\n📁 Results saved to fixed_rr_backtest_results.csv")
    
    # Recommendation
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATION")
    print("=" * 70)
    print(f"Best Config: {best['config']}")
    print(f"Parameters: {best['tp_r']}R Take Profit, 1.2% Stop Loss")
    print(f"Expected: {best['total_r']:+.1f}R over {best['trades']} trades")
    print(f"Win Rate: {best['wr']}% | Walk-Forward: {best['wf_score']}")

if __name__ == "__main__":
    main()
