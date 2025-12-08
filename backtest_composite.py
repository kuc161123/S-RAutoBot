#!/usr/bin/env python3
"""
Walk-Forward Backtest for Composite Scoring Execution Path

Compares:
- Path 1: Combo-only (current: LB WR >= 42%)
- Path 2: Composite Score (new: score >= 50)
- Combined: Either path triggers

Uses walk-forward validation to avoid overfitting.
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Database connection
DB_URL = "postgresql://postgres:JVjCwwHvcmUmZCJsLhHwqutctwyfVwxC@yamanote.proxy.rlwy.net:19297/railway"

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Calculate Wilson score lower bound (95% confidence)."""
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def load_data():
    """Load trade_history from PostgreSQL."""
    print("📊 Loading data from PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    
    query = """
        SELECT id, symbol, side, combo, outcome, 
               session, hour_utc, day_of_week, 
               btc_trend, volatility_regime, atr_percent,
               time_to_result, max_r_reached, is_executed, created_at
        FROM trade_history
        WHERE created_at > NOW() - INTERVAL '60 days'
        ORDER BY created_at
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Loaded {len(df)} trades from last 60 days")
    print(f"   Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"   Unique symbols: {df['symbol'].nunique()}")
    print(f"   Unique combos: {df['combo'].nunique()}")
    print(f"   Win rate: {(df['outcome'] == 'win').mean():.1%}")
    
    return df

def calculate_combo_wr(df_train, symbol, side, combo):
    """Calculate combo-level Wilson LB WR from training data."""
    subset = df_train[(df_train['symbol'] == symbol) & 
                      (df_train['side'] == side) & 
                      (df_train['combo'] == combo)]
    if len(subset) < 5:
        return None, 0
    wins = (subset['outcome'] == 'win').sum()
    total = len(subset)
    return wilson_lower_bound(wins, total), total

def calculate_symbol_wr(df_train, symbol):
    """Calculate symbol-level Wilson LB WR from training data."""
    subset = df_train[df_train['symbol'] == symbol]
    if len(subset) < 5:
        return None, 0
    wins = (subset['outcome'] == 'win').sum()
    total = len(subset)
    return wilson_lower_bound(wins, total), total

def calculate_session_wr(df_train, session):
    """Calculate session-level Wilson LB WR from training data."""
    subset = df_train[df_train['session'] == session]
    if len(subset) < 5:
        return 50, 0  # Default neutral
    wins = (subset['outcome'] == 'win').sum()
    total = len(subset)
    return wilson_lower_bound(wins, total), total

def calculate_btc_alignment(side, btc_trend):
    """Score BTC alignment (0-100)."""
    if btc_trend == 'bullish' and side == 'long':
        return 100
    elif btc_trend == 'bearish' and side == 'short':
        return 100
    elif btc_trend == 'neutral':
        return 50
    else:
        return 0  # Counter-trend

def calculate_volatility_fit(df_train, volatility_regime):
    """Calculate volatility regime performance."""
    subset = df_train[df_train['volatility_regime'] == volatility_regime]
    if len(subset) < 5:
        return 50, 0
    wins = (subset['outcome'] == 'win').sum()
    total = len(subset)
    return wilson_lower_bound(wins, total), total

def calculate_composite_score(df_train, symbol, side, combo, session, btc_trend, volatility_regime):
    """Calculate composite score (0-100)."""
    # Weights
    W_COMBO = 0.40
    W_SYMBOL = 0.25
    W_SESSION = 0.15
    W_BTC = 0.10
    W_VOL = 0.10
    
    # Get component scores
    combo_wr, combo_n = calculate_combo_wr(df_train, symbol, side, combo)
    symbol_wr, symbol_n = calculate_symbol_wr(df_train, symbol)
    session_wr, session_n = calculate_session_wr(df_train, session)
    btc_score = calculate_btc_alignment(side, btc_trend)
    vol_wr, vol_n = calculate_volatility_fit(df_train, volatility_regime)
    
    # Use defaults if insufficient data
    combo_wr = combo_wr if combo_wr is not None else 40
    symbol_wr = symbol_wr if symbol_wr is not None else 40
    
    # Calculate composite
    score = (
        W_COMBO * combo_wr +
        W_SYMBOL * symbol_wr +
        W_SESSION * session_wr +
        W_BTC * btc_score +
        W_VOL * vol_wr
    )
    
    return score, {
        'combo_wr': combo_wr,
        'symbol_wr': symbol_wr,
        'session_wr': session_wr,
        'btc_score': btc_score,
        'vol_wr': vol_wr
    }

def walk_forward_backtest(df):
    """
    Walk-forward backtest using index-based splits.
    
    Since data only spans ~2 days, we split by index position.
    Train on first 60%, test on last 40%.
    """
    print(f"\n🚀 Running Walk-Forward Backtest")
    
    df = df.sort_values('created_at').reset_index(drop=True)
    n = len(df)
    
    # Simple train/test split (60/40)
    train_end = int(n * 0.6)
    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]
    
    print(f"   Train: {len(df_train)} trades ({df_train['created_at'].min()} to {df_train['created_at'].max()})")
    print(f"   Test: {len(df_test)} trades ({df_test['created_at'].min()} to {df_test['created_at'].max()})")
    
    results = {
        'combo_only': {'trades': 0, 'wins': 0, 'signals': []},
        'composite': {'trades': 0, 'wins': 0, 'signals': []},
        'combined': {'trades': 0, 'wins': 0, 'signals': []},
    }
    
    # Test each signal in test period
    for _, row in df_test.iterrows():
        symbol = row['symbol']
        side = row['side']
        combo = row['combo']
        session = row['session']
        btc_trend = row['btc_trend']
        vol_regime = row['volatility_regime']
        actual_outcome = row['outcome']
        
        # Path 1: Combo-only (current logic)
        combo_wr, combo_n = calculate_combo_wr(df_train, symbol, side, combo)
        combo_pass = combo_wr is not None and combo_wr >= 42 and combo_n >= 5
        
        # Path 2: Composite Score
        comp_score, breakdown = calculate_composite_score(
            df_train, symbol, side, combo, session, btc_trend, vol_regime
        )
        composite_pass = comp_score >= 50
        
        # Record results
        if combo_pass:
            results['combo_only']['trades'] += 1
            if actual_outcome == 'win':
                results['combo_only']['wins'] += 1
            results['combo_only']['signals'].append({
                'symbol': symbol, 'outcome': actual_outcome, 'combo_wr': combo_wr
            })
        
        if composite_pass:
            results['composite']['trades'] += 1
            if actual_outcome == 'win':
                results['composite']['wins'] += 1
            results['composite']['signals'].append({
                'symbol': symbol, 'outcome': actual_outcome, 'score': comp_score, **breakdown
            })
        
        if combo_pass or composite_pass:
            results['combined']['trades'] += 1
            if actual_outcome == 'win':
                results['combined']['wins'] += 1
    
    return results

def print_results(results):
    """Print backtest results."""
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    
    for path_name, data in results.items():
        trades = data['trades']
        wins = data['wins']
        wr = (wins / trades * 100) if trades > 0 else 0
        lb_wr = wilson_lower_bound(wins, trades) if trades > 0 else 0
        
        print(f"\n{path_name.upper()}:")
        print(f"  Trades: {trades}")
        print(f"  Wins: {wins}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  LB Win Rate (95% CI): {lb_wr:.1f}%")
    
    # Comparison
    print("\n" + "="*60)
    print("📈 COMPARISON")
    print("="*60)
    
    combo_trades = results['combo_only']['trades']
    composite_trades = results['composite']['trades']
    combined_trades = results['combined']['trades']
    
    # Additional trades from composite
    if combo_trades > 0:
        additional = combined_trades - combo_trades
        print(f"\n✅ Composite would add {additional} additional trades (+{additional/combo_trades*100:.0f}%)")
    
    combo_wr = results['combo_only']['wins'] / results['combo_only']['trades'] * 100 if results['combo_only']['trades'] > 0 else 0
    composite_wr = results['composite']['wins'] / results['composite']['trades'] * 100 if results['composite']['trades'] > 0 else 0
    combined_wr = results['combined']['wins'] / results['combined']['trades'] * 100 if results['combined']['trades'] > 0 else 0
    
    print(f"\nWin Rate Comparison:")
    print(f"  Combo-only:  {combo_wr:.1f}%")
    print(f"  Composite:   {composite_wr:.1f}%")
    print(f"  Combined:    {combined_wr:.1f}%")

def main():
    print("🔬 Composite Scoring Walk-Forward Backtest")
    print("="*60)
    
    # Load data
    df = load_data()
    
    if len(df) < 100:
        print("❌ Insufficient data for backtest (need at least 100 trades)")
        return
    
    # Run backtest
    results = walk_forward_backtest(df)
    
    # Print results
    print_results(results)
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main()

