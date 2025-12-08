#!/usr/bin/env python3
"""
Realistic Feature-Enhanced Backtest

Improvements over previous backtest:
1. Simulates spread/slippage (-0.03% per trade)
2. Uses candle-close only signals (like live bot)
3. Adds session, hour, and volatility regime features
4. Tests if filtering by features improves WR
"""

import requests
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from collections import defaultdict

BYBIT_BASE = "https://api.bybit.com"

TOP_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'NEARUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SEIUSDT',
]

# Realistic execution costs
SPREAD_PCT = 0.01  # 0.01% spread
SLIPPAGE_PCT = 0.02  # 0.02% slippage
TOTAL_COST_PCT = SPREAD_PCT + SLIPPAGE_PCT  # 0.03% total

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def fetch_klines(symbol: str, interval: str = '3', days: int = 60) -> pd.DataFrame:
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    
    while current_end > start_time:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
                break
                
            klines = data['result']['list']
            all_data.extend(klines)
            earliest = int(klines[-1][0])
            if earliest <= start_time:
                break
            current_end = earliest - 1
            time.sleep(0.05)
        except:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # ATR percentile for volatility regime
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Fib
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # Volume relative
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    return df.dropna()

def get_combo(row) -> str:
    rsi = row['rsi']
    if rsi < 30: r_bin = '<30'
    elif rsi < 40: r_bin = '30-40'
    elif rsi < 60: r_bin = '40-60'
    elif rsi < 70: r_bin = '60-70'
    else: r_bin = '70+'
    
    m_bin = 'bull' if row['macd'] > row['macd_signal'] else 'bear'
    
    high, low, close = row['roll_high'], row['roll_low'], row['close']
    if high == low:
        f_bin = '0-23'
    else:
        fib = (high - close) / (high - low) * 100
        if fib < 23.6: f_bin = '0-23'
        elif fib < 38.2: f_bin = '23-38'
        elif fib < 50.0: f_bin = '38-50'
        elif fib < 61.8: f_bin = '50-61'
        elif fib < 78.6: f_bin = '61-78'
        elif fib < 100: f_bin = '78-100'
        else: f_bin = '100+'
    
    return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

def get_session(hour: int) -> str:
    if 0 <= hour < 8:
        return 'asia'
    elif 8 <= hour < 16:
        return 'london'
    else:
        return 'newyork'

def get_volatility_regime(atr_pct: float, atr_percentile: float) -> str:
    if atr_percentile < 0.33:
        return 'low'
    elif atr_percentile < 0.66:
        return 'medium'
    else:
        return 'high'

def simulate_trade_realistic(df, entry_idx, side, entry, atr, rr_ratio=2.0, max_candles=80):
    """Simulate with force minimum SL and execution costs."""
    MIN_SL_PCT = 0.5
    
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * rr_ratio
    
    # Add execution costs (spread + slippage)
    cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        # Long pays more on entry
        adjusted_entry = entry + cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        # Short gets less on entry
        adjusted_entry = entry - cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        if side == 'long':
            if candle['low'] <= sl:
                return 'loss'
            elif candle['high'] >= tp:
                return 'win'
        else:
            if candle['high'] >= sl:
                return 'loss'
            elif candle['low'] <= tp:
                return 'win'
    
    return 'timeout'

def run_backtest():
    print("🔬 Realistic Feature-Enhanced Backtest")
    print("="*70)
    print(f"Spread simulation: {SPREAD_PCT}%")
    print(f"Slippage simulation: {SLIPPAGE_PCT}%")
    print(f"Total execution cost: {TOTAL_COST_PCT}%")
    
    # Fetch data
    symbol_data = {}
    print(f"\n📥 Fetching 60-day data for {len(TOP_SYMBOLS)} symbols...")
    
    for i, symbol in enumerate(TOP_SYMBOLS):
        print(f"   [{i+1}/{len(TOP_SYMBOLS)}] {symbol}...", end=' ', flush=True)
        df = fetch_klines(symbol, '3', 60)
        if len(df) > 1000:
            df = calculate_indicators(df)
            # Calculate ATR percentile
            df['atr_percentile'] = df['atr_pct'].rank(pct=True)
            symbol_data[symbol] = df
            print(f"✅ {len(df)} candles")
        else:
            print(f"❌")
        time.sleep(0.3)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Split: 40 days train, 20 days test
    train_days = 40
    
    # Feature analysis stats
    session_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    hour_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    vol_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    volume_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    combo_stats = defaultdict(lambda: {
        'base_wins': 0, 'base_total': 0,
        'filtered_wins': 0, 'filtered_total': 0
    })
    
    # Process signals
    total_signals = 0
    
    for symbol, df in symbol_data.items():
        total_candles = len(df)
        test_start = int(total_candles * (train_days / 60))
        df_test = df.iloc[test_start:]
        
        # Only check every 20 candles (like live bot checking every ~1 hour)
        for idx in range(0, len(df_test) - 80, 20):
            row = df_test.iloc[idx]
            combo = get_combo(row)
            entry = row['close']
            atr = row['atr']
            
            # Features
            hour = row['start'].hour
            session = get_session(hour)
            vol_regime = get_volatility_regime(row['atr_pct'], row['atr_percentile'])
            vol_ratio = row['vol_ratio']
            high_volume = vol_ratio > 1.2
            
            for side in ['long', 'short']:
                total_signals += 1
                
                outcome = simulate_trade_realistic(df_test, idx, side, entry, atr)
                if outcome == 'timeout':
                    continue
                
                is_win = outcome == 'win'
                key = f"{symbol}|{side}|{combo}"
                
                # Base stats (no filtering)
                combo_stats[key]['base_total'] += 1
                if is_win:
                    combo_stats[key]['base_wins'] += 1
                
                # Feature stats
                session_stats[session]['total'] += 1
                hour_stats[hour]['total'] += 1
                vol_stats[vol_regime]['total'] += 1
                volume_stats['high' if high_volume else 'low']['total'] += 1
                
                if is_win:
                    session_stats[session]['wins'] += 1
                    hour_stats[hour]['wins'] += 1
                    vol_stats[vol_regime]['wins'] += 1
                    volume_stats['high' if high_volume else 'low']['wins'] += 1
                
                # Filtered stats (good conditions only)
                good_session = session in ['london', 'newyork']
                good_volatility = vol_regime in ['medium', 'high']
                good_volume = high_volume
                
                if good_session and good_volatility:
                    combo_stats[key]['filtered_total'] += 1
                    if is_win:
                        combo_stats[key]['filtered_wins'] += 1
    
    # Print Feature Analysis
    print("\n" + "="*70)
    print("📊 FEATURE ANALYSIS")
    print("="*70)
    
    print("\n🕐 BY SESSION:")
    for session, stats in sorted(session_stats.items(), key=lambda x: x[1]['wins']/max(x[1]['total'],1), reverse=True):
        if stats['total'] > 0:
            wr = stats['wins'] / stats['total'] * 100
            print(f"   {session:10}: {wr:.1f}% WR ({stats['total']} trades)")
    
    print("\n⏰ BY HOUR (best hours):")
    hour_sorted = sorted(hour_stats.items(), key=lambda x: x[1]['wins']/max(x[1]['total'],1), reverse=True)
    for hour, stats in hour_sorted[:5]:
        if stats['total'] > 50:
            wr = stats['wins'] / stats['total'] * 100
            print(f"   Hour {hour:02d}:00: {wr:.1f}% WR ({stats['total']} trades)")
    
    print("\n📈 BY VOLATILITY REGIME:")
    for regime, stats in sorted(vol_stats.items(), key=lambda x: x[1]['wins']/max(x[1]['total'],1), reverse=True):
        if stats['total'] > 0:
            wr = stats['wins'] / stats['total'] * 100
            print(f"   {regime:10}: {wr:.1f}% WR ({stats['total']} trades)")
    
    print("\n📊 BY VOLUME:")
    for vol_type, stats in volume_stats.items():
        if stats['total'] > 0:
            wr = stats['wins'] / stats['total'] * 100
            print(f"   {vol_type:10}: {wr:.1f}% WR ({stats['total']} trades)")
    
    # Compare base vs filtered
    print("\n" + "="*70)
    print("🔍 BASE vs FILTERED COMPARISON")
    print("="*70)
    
    base_total_wins = sum(c['base_wins'] for c in combo_stats.values())
    base_total_trades = sum(c['base_total'] for c in combo_stats.values())
    filtered_total_wins = sum(c['filtered_wins'] for c in combo_stats.values())
    filtered_total_trades = sum(c['filtered_total'] for c in combo_stats.values())
    
    print(f"\nBASE (no filtering):")
    print(f"   Trades: {base_total_trades}")
    print(f"   Wins: {base_total_wins}")
    print(f"   Win Rate: {base_total_wins/base_total_trades*100:.1f}%" if base_total_trades > 0 else "   N/A")
    
    print(f"\nFILTERED (London/NY + Medium/High Vol):")
    print(f"   Trades: {filtered_total_trades}")
    print(f"   Wins: {filtered_total_wins}")
    print(f"   Win Rate: {filtered_total_wins/filtered_total_trades*100:.1f}%" if filtered_total_trades > 0 else "   N/A")
    
    if base_total_trades > 0 and filtered_total_trades > 0:
        base_wr = base_total_wins / base_total_trades * 100
        filtered_wr = filtered_total_wins / filtered_total_trades * 100
        improvement = filtered_wr - base_wr
        print(f"\n📈 Improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print("✅ Feature filtering IMPROVES win rate!")
        else:
            print("⚠️ Feature filtering does NOT improve win rate")
    
    # Find best combos
    print("\n" + "="*70)
    print("🏆 TOP PERFORMING COMBOS (Filtered)")
    print("="*70)
    
    good_combos = []
    for key, stats in combo_stats.items():
        if stats['filtered_total'] >= 10:
            wr = stats['filtered_wins'] / stats['filtered_total'] * 100
            if wr >= 45:
                symbol, side, combo = key.split('|')
                good_combos.append({
                    'symbol': symbol,
                    'side': side,
                    'combo': combo,
                    'trades': stats['filtered_total'],
                    'wins': stats['filtered_wins'],
                    'wr': wr
                })
    
    good_combos.sort(key=lambda x: x['wr'], reverse=True)
    
    print(f"\nFound {len(good_combos)} combos with ≥45% WR (filtered):\n")
    
    for i, c in enumerate(good_combos[:15], 1):
        print(f"{i:2}. {c['symbol']} {c['side']} {c['combo']}")
        print(f"    WR: {c['wr']:.1f}% (N={c['trades']})")
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    run_backtest()
