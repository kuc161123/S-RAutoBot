#!/usr/bin/env python3
"""
Test Force Minimum SL on Analytics Data

Uses actual signals from trade_history database to compare:
1. Original outcomes (as recorded)
2. Simulated outcomes with force minimum SL
"""

import psycopg2
import pandas as pd
import requests
import math
import time
from datetime import datetime, timedelta

DB_URL = "postgresql://postgres:JVjCwwHvcmUmZCJsLhHwqutctwyfVwxC@yamanote.proxy.rlwy.net:19297/railway"
BYBIT_BASE = "https://api.bybit.com"

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def load_signals():
    """Load signals from trade_history."""
    print("📊 Loading signals from PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    
    query = """
        SELECT id, symbol, side, combo, outcome, 
               atr_percent, created_at, is_executed
        FROM trade_history
        WHERE outcome IN ('win', 'loss')
        ORDER BY created_at DESC
        LIMIT 1000
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Loaded {len(df)} resolved signals")
    print(f"   Win rate: {(df['outcome'] == 'win').mean():.1%}")
    print(f"   Executed: {df['is_executed'].sum()}")
    
    return df

def fetch_klines(symbol: str, start_time: datetime, limit: int = 200) -> pd.DataFrame:
    """Fetch klines from Bybit for a specific time window."""
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': '3',
        'limit': limit,
        'start': int(start_time.timestamp() * 1000)
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
            return pd.DataFrame()
        
        klines = data['result']['list']
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float})
        df['start'] = pd.to_datetime(df['start'], unit='ms')
        df = df.sort_values('start').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def simulate_trade(klines: pd.DataFrame, entry: float, side: str, sl_dist: float, tp_dist: float, max_candles: int = 80) -> str:
    """Simulate trade outcome."""
    if side == 'long':
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    
    for i in range(min(max_candles, len(klines))):
        candle = klines.iloc[i]
        high = candle['high']
        low = candle['low']
        
        if side == 'long':
            if low <= sl:
                return 'loss'
            elif high >= tp:
                return 'win'
        else:
            if high >= sl:
                return 'loss'
            elif low <= tp:
                return 'win'
    
    return 'timeout'

def run_comparison():
    print("🔬 Force Minimum SL on Analytics Data")
    print("="*60)
    
    # Load signals
    df = load_signals()
    
    # Get unique symbols with low ATR
    MIN_SL_PCT = 0.5
    
    # Filter signals where ATR < minimum (these are the ones that would have been adjusted)
    low_atr_signals = df[df['atr_percent'] < MIN_SL_PCT].copy()
    normal_signals = df[df['atr_percent'] >= MIN_SL_PCT].copy()
    
    print(f"\n📈 Signal Breakdown:")
    print(f"   Low ATR (<0.5%): {len(low_atr_signals)} signals")
    print(f"   Normal ATR (>=0.5%): {len(normal_signals)} signals")
    
    # Compare outcomes for low ATR signals
    if len(low_atr_signals) > 0:
        low_atr_wr = (low_atr_signals['outcome'] == 'win').mean() * 100
        print(f"\n📉 Low ATR Signals (original outcomes):")
        print(f"   Win Rate: {low_atr_wr:.1f}%")
        print(f"   Outcome distribution:")
        print(low_atr_signals['outcome'].value_counts().to_string())
    
    if len(normal_signals) > 0:
        normal_wr = (normal_signals['outcome'] == 'win').mean() * 100
        print(f"\n📊 Normal ATR Signals (original outcomes):")
        print(f"   Win Rate: {normal_wr:.1f}%")
    
    # Now simulate with force minimum SL on a sample of low ATR signals
    print(f"\n🧪 Simulating Force Minimum SL on {min(50, len(low_atr_signals))} low ATR signals...")
    
    sample = low_atr_signals.head(50)
    results = {'original_win': 0, 'original_loss': 0, 'simulated_win': 0, 'simulated_loss': 0, 'timeout': 0}
    
    for idx, row in sample.iterrows():
        symbol = row['symbol']
        side = row['side']
        atr_pct = row['atr_percent'] / 100  # Convert to decimal
        created_at = row['created_at']
        original_outcome = row['outcome']
        
        # Track original
        if original_outcome == 'win':
            results['original_win'] += 1
        else:
            results['original_loss'] += 1
        
        # Fetch klines from signal time
        klines = fetch_klines(symbol, created_at, 100)
        if len(klines) < 10:
            continue
        
        entry = klines.iloc[0]['close']
        
        # Force minimum SL
        forced_sl_dist = entry * (MIN_SL_PCT / 100)
        forced_tp_dist = forced_sl_dist * 2  # 2:1 R:R
        
        # Simulate
        simulated = simulate_trade(klines.iloc[1:], entry, side, forced_sl_dist, forced_tp_dist)
        
        if simulated == 'win':
            results['simulated_win'] += 1
        elif simulated == 'loss':
            results['simulated_loss'] += 1
        else:
            results['timeout'] += 1
        
        time.sleep(0.1)
    
    # Print results
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    total_original = results['original_win'] + results['original_loss']
    total_simulated = results['simulated_win'] + results['simulated_loss']
    
    if total_original > 0:
        original_wr = results['original_win'] / total_original * 100
        print(f"\nORIGINAL (recorded in DB):")
        print(f"  Wins: {results['original_win']}, Losses: {results['original_loss']}")
        print(f"  Win Rate: {original_wr:.1f}%")
    
    if total_simulated > 0:
        simulated_wr = results['simulated_win'] / total_simulated * 100
        print(f"\nSIMULATED (Force Minimum SL):")
        print(f"  Wins: {results['simulated_win']}, Losses: {results['simulated_loss']}")
        print(f"  Timeouts: {results['timeout']}")
        print(f"  Win Rate: {simulated_wr:.1f}%")
        
        diff = simulated_wr - original_wr
        print(f"\n📈 Difference: {diff:+.1f}%")
        
        if diff > 0:
            print("✅ Force Minimum SL shows IMPROVEMENT!")
        elif diff < -5:
            print("⚠️ Force Minimum SL shows degradation")
        else:
            print("➡️ Similar performance")

if __name__ == "__main__":
    run_comparison()
