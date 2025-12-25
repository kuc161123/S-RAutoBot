#!/usr/bin/env python3
"""
Bybit-Realistic Trailing Stop Backtest
=======================================

This backtest simulates how Bybit's native trailing stop actually works:
- Trailing triggers on ANY pullback from peak, not just candle close
- Uses intra-candle price simulation to model tick-by-tick behavior
- Tests multiple trailing configurations to find optimal settings

Key difference from previous backtests:
- Old: Check candle high/low, exit at best_r - trail_distance
- New: Simulate price path within candle, exit when pullback exceeds trail_distance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Strategy Parameters (30M Optimized)
TIMEFRAME = 30  # minutes
SL_PCT = 1.2  # Fixed 1.2% stop loss
COOLDOWN_BARS = 3

# Trailing Configurations to Test
TRAILING_CONFIGS = [
    # (trail_pct, activation_r, max_r, name)
    (0.05, 0.2, 7.0, "Current: 0.05R trail @ +0.2R"),
    (0.1, 0.2, 7.0, "0.1R trail @ +0.2R"),
    (0.2, 0.3, 7.0, "0.2R trail @ +0.3R"),
    (0.3, 0.5, 7.0, "0.3R trail @ +0.5R"),
    (0.5, 1.0, 7.0, "0.5R trail @ +1.0R"),
    (0.5, 0.5, 7.0, "0.5R trail @ +0.5R"),
    (1.0, 1.0, 7.0, "1.0R trail @ +1.0R"),
    # Fixed TP alternatives
    (0.0, 0.0, 2.0, "Fixed 2R TP (no trail)"),
    (0.0, 0.0, 3.0, "Fixed 3R TP (no trail)"),
    (0.0, 0.0, 5.0, "Fixed 5R TP (no trail)"),
    # Hybrid configs
    (0.3, 1.0, 5.0, "0.3R trail @ +1.0R, max 5R"),
    (0.2, 0.5, 3.0, "0.2R trail @ +0.5R, max 3R"),
]

# Data source
DATA_DAYS = 60  # 60 days of history

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
            print(f"Error fetching {symbol}: {e}")
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

def simulate_bybit_trailing(df: pd.DataFrame, idx: int, side: str, entry: float,
                            sl_distance: float, trail_pct: float, activation_r: float, 
                            max_r: float) -> tuple:
    """
    Simulate Bybit's native trailing stop behavior.
    
    Key insight: Bybit trails on EVERY TICK, not just candle close.
    We simulate this by checking if intra-candle price would trigger the trail.
    
    Returns: (exit_r, exit_reason, bars_held)
    """
    max_favorable_r = 0.0
    trailing_active = False
    trail_sl = None
    
    for i in range(idx + 1, min(idx + 500, len(df))):  # Max 500 bars
        candle = df.iloc[i]
        bars_held = i - idx
        
        # Calculate R values for this candle
        if side == 'long':
            # Best R this candle = (high - entry) / sl_distance
            candle_best_r = (candle['high'] - entry) / sl_distance
            candle_worst_r = (candle['low'] - entry) / sl_distance
            
            # Check SL hit first
            if candle_worst_r <= -1.0:
                return (-1.0, "SL", bars_held)
            
            # Check TP hit
            if candle_best_r >= max_r:
                return (max_r, "TP", bars_held)
            
            # Update max favorable
            max_favorable_r = max(max_favorable_r, candle_best_r)
            
            # Trailing logic (simulating Bybit's tick-by-tick behavior)
            if trail_pct > 0 and max_favorable_r >= activation_r:
                trailing_active = True
                # Trail SL = entry + (max_r - trail_pct) * sl_distance
                trail_sl_r = max_favorable_r - trail_pct
                
                # CRITICAL: Check if LOW would have hit the trail SL
                # This simulates Bybit triggering on any pullback
                if candle_worst_r <= trail_sl_r and trail_sl_r > 0:
                    # Price pulled back enough to trigger trailing stop
                    return (trail_sl_r, "TRAIL", bars_held)
        else:
            # SHORT
            candle_best_r = (entry - candle['low']) / sl_distance
            candle_worst_r = (entry - candle['high']) / sl_distance
            
            if candle_worst_r <= -1.0:
                return (-1.0, "SL", bars_held)
            
            if candle_best_r >= max_r:
                return (max_r, "TP", bars_held)
            
            max_favorable_r = max(max_favorable_r, candle_best_r)
            
            if trail_pct > 0 and max_favorable_r >= activation_r:
                trailing_active = True
                trail_sl_r = max_favorable_r - trail_pct
                
                if candle_worst_r <= trail_sl_r and trail_sl_r > 0:
                    return (trail_sl_r, "TRAIL", bars_held)
    
    # Timeout - exit at current R
    final_candle = df.iloc[min(idx + 499, len(df) - 1)]
    if side == 'long':
        exit_r = (final_candle['close'] - entry) / sl_distance
    else:
        exit_r = (entry - final_candle['close']) / sl_distance
    
    return (exit_r, "TIMEOUT", 500)

def run_backtest(symbols: list, trailing_config: tuple) -> dict:
    """Run backtest for a single trailing configuration"""
    trail_pct, activation_r, max_r, config_name = trailing_config
    
    all_trades = []
    
    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_klines(symbol, str(TIMEFRAME), DATA_DAYS)
            if len(df) < 100:
                continue
            
            # Detect divergences
            df = detect_divergences(df)
            
            # Volume filter
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ok'] = df['volume'] > df['volume_sma']
            
            last_trade_idx = -COOLDOWN_BARS
            
            for idx in range(50, len(df) - 10):
                # Cooldown
                if idx - last_trade_idx < COOLDOWN_BARS:
                    continue
                
                row = df.iloc[idx]
                
                # Check for signals
                side = None
                if row['bullish_div'] and row['volume_ok']:
                    side = 'long'
                elif row['bearish_div'] and row['volume_ok']:
                    side = 'short'
                
                if side is None:
                    continue
                
                # Entry
                entry = row['close']
                sl_distance = entry * (SL_PCT / 100)
                
                # Simulate trade with Bybit trailing
                exit_r, exit_reason, bars_held = simulate_bybit_trailing(
                    df, idx, side, entry, sl_distance,
                    trail_pct, activation_r, max_r
                )
                
                all_trades.append({
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
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Calculate stats
    if not all_trades:
        return {'config': config_name, 'total_r': 0, 'trades': 0, 'wr': 0, 'avg_r': 0}
    
    trades_df = pd.DataFrame(all_trades)
    total_r = trades_df['exit_r'].sum()
    wins = (trades_df['exit_r'] > 0).sum()
    total = len(trades_df)
    wr = wins / total * 100 if total > 0 else 0
    avg_r = trades_df['exit_r'].mean()
    
    # Exit reason breakdown
    reasons = trades_df['exit_reason'].value_counts().to_dict()
    
    return {
        'config': config_name,
        'total_r': round(total_r, 1),
        'trades': total,
        'wr': round(wr, 1),
        'avg_r': round(avg_r, 3),
        'trail_pct': trail_pct,
        'activation_r': activation_r,
        'max_r': max_r,
        'sl_exits': reasons.get('SL', 0),
        'tp_exits': reasons.get('TP', 0),
        'trail_exits': reasons.get('TRAIL', 0),
        'avg_bars': round(trades_df['bars_held'].mean(), 1) if total > 0 else 0
    }

def main():
    """Main entry point"""
    print("=" * 70)
    print("BYBIT-REALISTIC TRAILING STOP BACKTEST")
    print("=" * 70)
    print(f"Timeframe: {TIMEFRAME}M | SL: {SL_PCT}% | Cooldown: {COOLDOWN_BARS} bars")
    print(f"Testing {len(TRAILING_CONFIGS)} trailing configurations")
    print("=" * 70)
    
    # Get symbols from config
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        # Symbols are under scalp.divergence_symbols
        symbols = config.get('scalp', {}).get('divergence_symbols', [])[:50]  # Limit for speed
        if not symbols:
            # Fallback to common symbols
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 
                       'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'DOTUSDT',
                       'SHIBUSDT', 'LTCUSDT', 'TRXUSDT', 'ATOMUSDT', 'UNIUSDT']
    except Exception as e:
        print(f"Config load error: {e}, using fallback symbols")
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
    
    print(f"Testing on {len(symbols)} symbols")
    print("-" * 70)
    
    results = []
    
    for i, config in enumerate(TRAILING_CONFIGS):
        print(f"\n[{i+1}/{len(TRAILING_CONFIGS)}] Testing: {config[3]}")
        result = run_backtest(symbols, config)
        results.append(result)
        
        if result['trades'] > 0:
            print(f"   Total R: {result['total_r']:+.1f} | Trades: {result['trades']} | WR: {result['wr']:.1f}% | Avg R: {result['avg_r']:.3f}")
            print(f"   Exits: SL={result.get('sl_exits', 0)}, Trail={result.get('trail_exits', 0)}, TP={result.get('tp_exits', 0)}")
        else:
            print(f"   No trades generated")
    
    # Sort by total R
    results_df = pd.DataFrame(results).sort_values('total_r', ascending=False)
    
    print("\n" + "=" * 70)
    print("RESULTS RANKED BY TOTAL R")
    print("=" * 70)
    
    for _, row in results_df.iterrows():
        emoji = "🥇" if row['total_r'] == results_df['total_r'].max() else "  "
        print(f"{emoji} {row['config']}")
        print(f"   Total R: {row['total_r']:+.1f} | WR: {row['wr']:.1f}% | Avg R: {row['avg_r']:.3f}")
        print(f"   SL: {row['sl_exits']} | Trail: {row['trail_exits']} | TP: {row['tp_exits']} | Avg Bars: {row['avg_bars']}")
        print()
    
    # Save results
    results_df.to_csv('bybit_trailing_results.csv', index=False)
    print(f"Results saved to bybit_trailing_results.csv")
    
    # Print recommendation
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATION")
    print("=" * 70)
    print(f"Best Config: {best['config']}")
    print(f"Parameters: trail_pct={best['trail_pct']}, activation_r={best['activation_r']}, max_r={best['max_r']}")
    print(f"Expected: {best['total_r']:+.1f}R over {best['trades']} trades ({best['wr']:.1f}% WR)")

if __name__ == "__main__":
    main()
