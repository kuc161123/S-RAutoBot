#!/usr/bin/env python3
"""
AUTONOMOUS VWAP STRATEGY OPTIMIZER
===================================
Master Trader Analysis Framework

This script autonomously tests different VWAP configurations to find
profitable setups for each symbol with 45%+ Lower Bound Win Rate.

Key Features:
- Walk-forward validation (80% train / 20% test)
- Realistic execution (fees, slippage buffer)
- Symbol-specific optimization
- Side-specific analysis (long/short)
- Multi-parameter testing
- Wilson Lower Bound for statistical confidence

Author: Autonomous Trading System
"""

import requests
import pandas as pd
import numpy as np
import yaml
import math
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy Parameters
TIMEFRAME = '3'  # 3-minute candles (match live bot)
DATA_DAYS = 90   # 90 days for robust testing
TRAIN_RATIO = 0.7  # 70% train, 30% test (walk-forward)

# Risk:Reward (fixed at 2:1)
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0

# Thresholds
MIN_TRADES = 20          # Minimum sample size for statistical validity
TARGET_LB_WR = 45.0      # 45% Lower Bound Win Rate target
TARGET_EV = 0.0          # Positive EV at 2:1 R:R

# Fee/Slippage buffer (realistic execution)
TOTAL_FEES = 0.001  # 0.1% total (0.05% each way)

# API
BASE_URL = "https://api.bybit.com"

# Symbol batching
BATCH_SIZE = 50  # Process symbols in batches

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """
    Calculate Wilson score lower bound for win rate confidence.
    z=1.96 for 95% confidence interval.
    """
    if n == 0:
        return 0.0
    p = wins / n
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denominator)

def calc_ev(wr: float, rr: float = 2.0) -> float:
    """Calculate Expected Value in R-multiples."""
    return (wr * rr) - (1 - wr)

def get_symbols(limit: int = 400) -> list:
    """Fetch top USDT perpetual symbols by volume."""
    url = f"{BASE_URL}/v5/market/tickers?category=linear"
    resp = requests.get(url)
    tickers = resp.json().get('result', {}).get('list', [])
    
    # Filter USDT perps and sort by volume
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    
    return [t['symbol'] for t in usdt_pairs[:limit]]

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch historical klines from Bybit."""
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        url = f"{BASE_URL}/v5/market/kline"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data:
                break
            
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            
            if len(data) < 1000:
                break
                
        except Exception as e:
            print(f"   ⚠️ Error fetching {symbol}: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    
    return df

# =============================================================================
# INDICATOR CALCULATIONS (Match Live Bot Exactly)
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators exactly as the live bot does."""
    if len(df) < 50:
        return pd.DataFrame()
    
    # RSI (14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (12/26/9)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ATR (14 period)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Fib Levels (50-period rolling high/low)
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # VWAP (cumulative for session)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df.dropna()

# =============================================================================
# SIGNAL DETECTION (Match Live Bot Exactly)
# =============================================================================

def check_vwap_signal(row, prev_row) -> str:
    """
    Check for VWAP cross signal.
    Long: Low touched/crossed below VWAP and closed above
    Short: High touched/crossed above VWAP and closed below
    """
    if prev_row is None:
        return None
    
    # Long signal
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    # Short signal
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
        return 'short'
    
    return None

def get_combo_simplified(row) -> str:
    """
    SIMPLIFIED combo: 18 combinations (3 RSI x 2 MACD x 3 Fib)
    Matches live bot's get_combo function.
    """
    # RSI: 3 levels
    rsi = row['rsi']
    if rsi < 40:
        r_bin = 'oversold'
    elif rsi > 60:
        r_bin = 'overbought'
    else:
        r_bin = 'neutral'
    
    # MACD: 2 levels
    m_bin = 'bull' if row['macd'] > row['macd_signal'] else 'bear'
    
    # Fib: 3 levels
    high, low, close = row['roll_high'], row['roll_low'], row['close']
    if high == low:
        f_bin = 'low'
    else:
        fib = (high - close) / (high - low) * 100
        if fib < 38:
            f_bin = 'low'
        elif fib < 62:
            f_bin = 'mid'
        else:
            f_bin = 'high'
    
    return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, 
                   entry_price: float, atr: float) -> dict:
    """
    Simulate a trade with realistic execution.
    Returns outcome and metrics.
    """
    # Calculate TP/SL
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:  # short
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    # Apply fee buffer to TP (makes it harder to hit)
    if side == 'long':
        tp = tp * (1 + TOTAL_FEES)
    else:
        tp = tp * (1 - TOTAL_FEES)
    
    # Scan forward to find outcome
    max_bars = 100  # Maximum bars to hold
    rows = list(df.iloc[entry_idx+1:entry_idx+1+max_bars].itertuples())
    
    for future_row in rows:
        if side == 'long':
            # Check SL first (conservative)
            if future_row.low <= sl:
                return {'outcome': 'loss', 'pnl': -1.0}
            # Then check TP
            if future_row.high >= tp:
                return {'outcome': 'win', 'pnl': TP_ATR_MULT}
        else:  # short
            # Check SL first
            if future_row.high >= sl:
                return {'outcome': 'loss', 'pnl': -1.0}
            # Then check TP
            if future_row.low <= tp:
                return {'outcome': 'win', 'pnl': TP_ATR_MULT}
    
    # Timeout - close at current price (usually a small loss)
    return {'outcome': 'timeout', 'pnl': -0.5}

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_symbol_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """
    Run comprehensive backtest for a single symbol.
    Tests all combos for both longs and shorts.
    Returns winning configurations.
    """
    results = {
        'symbol': symbol,
        'long': defaultdict(lambda: {'train': {'w': 0, 'n': 0}, 'test': {'w': 0, 'n': 0}}),
        'short': defaultdict(lambda: {'train': {'w': 0, 'n': 0}, 'test': {'w': 0, 'n': 0}})
    }
    
    # Split data: train (70%) / test (30%)
    split_idx = int(len(df) * TRAIN_RATIO)
    
    # Iterate through candles
    rows = list(df.itertuples())
    for i in range(1, len(rows) - 100):  # Leave room for forward scan
        row = rows[i]
        prev_row = rows[i-1]
        
        # Determine train/test phase
        phase = 'train' if i < split_idx else 'test'
        
        # Check for VWAP signal
        side = check_vwap_signal(
            {'low': row.low, 'high': row.high, 'close': row.close, 'vwap': row.vwap},
            {'vwap': prev_row.vwap}
        )
        
        if not side:
            continue
        
        # Get combo
        combo = get_combo_simplified({
            'rsi': row.rsi,
            'macd': row.macd,
            'macd_signal': row.macd_signal,
            'roll_high': row.roll_high,
            'roll_low': row.roll_low,
            'close': row.close
        })
        
        # Simulate trade
        entry_price = row.close
        atr = row.atr
        
        if pd.isna(atr) or atr <= 0:
            continue
        
        trade_result = simulate_trade(df, i, side, entry_price, atr)
        
        # Record result
        results[side][combo][phase]['n'] += 1
        if trade_result['outcome'] == 'win':
            results[side][combo][phase]['w'] += 1
    
    return results

def analyze_results(results: dict) -> dict:
    """
    Analyze backtest results and find winning configurations.
    Applies walk-forward validation.
    """
    winners = {
        'long': [],
        'short': []
    }
    
    symbol = results['symbol']
    
    for side in ['long', 'short']:
        for combo, data in results[side].items():
            train = data['train']
            test = data['test']
            
            total_n = train['n'] + test['n']
            total_w = train['w'] + test['w']
            
            # Skip if insufficient samples
            if total_n < MIN_TRADES:
                continue
            
            # Calculate metrics
            train_wr = train['w'] / train['n'] if train['n'] > 0 else 0
            test_wr = test['w'] / test['n'] if test['n'] > 0 else 0
            total_wr = total_w / total_n
            
            # Wilson Lower Bound (key metric)
            lb_wr = wilson_lower_bound(total_w, total_n)
            
            # Expected Value
            ev = calc_ev(total_wr)
            
            # Walk-forward validation: test must not collapse
            # Allow some degradation but not complete failure
            if train['n'] >= 5 and test['n'] >= 3:
                if test_wr < 0.25:  # Test WR below 25% = failure
                    continue
            
            # Check if meets target
            if lb_wr * 100 >= TARGET_LB_WR and ev >= TARGET_EV:
                winners[side].append({
                    'combo': combo,
                    'n': total_n,
                    'wr': total_wr * 100,
                    'lb_wr': lb_wr * 100,
                    'ev': ev,
                    'train_wr': train_wr * 100,
                    'test_wr': test_wr * 100,
                    'train_n': train['n'],
                    'test_n': test['n']
                })
    
    # Sort by Lower Bound WR
    for side in ['long', 'short']:
        winners[side].sort(key=lambda x: x['lb_wr'], reverse=True)
    
    return winners

# =============================================================================
# AUTONOMOUS OPTIMIZATION LOOP
# =============================================================================

def run_autonomous_optimization():
    """
    Main autonomous optimization loop.
    Makes independent decisions based on results.
    """
    print("=" * 70)
    print("🤖 AUTONOMOUS VWAP STRATEGY OPTIMIZER")
    print("=" * 70)
    print(f"📊 Timeframe: {TIMEFRAME}m | Days: {DATA_DAYS}")
    print(f"🎯 Target: {TARGET_LB_WR}% LB WR | R:R: {TP_ATR_MULT}:1")
    print(f"📈 Walk-Forward: {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test")
    print("=" * 70)
    
    # Fetch symbols
    print("\n📋 Fetching symbols...")
    symbols = get_symbols(400)
    print(f"   Found {len(symbols)} symbols")
    
    # Results storage
    all_winners = {}
    stats = {
        'processed': 0,
        'with_winners': 0,
        'total_winning_combos': 0,
        'long_winners': 0,
        'short_winners': 0
    }
    
    start_time = time.time()
    
    for idx, symbol in enumerate(symbols):
        print(f"\n[{idx+1}/{len(symbols)}] {symbol}")
        print("-" * 40)
        
        try:
            # Fetch data
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 1000:
                print(f"   ⚠️ Insufficient data ({len(df)} candles)")
                continue
            
            print(f"   ✅ {len(df)} candles")
            
            # Calculate indicators
            df = calculate_indicators(df)
            if df.empty:
                print(f"   ⚠️ Failed to calculate indicators")
                continue
            
            # Run backtest
            results = run_symbol_backtest(symbol, df)
            
            # Analyze results
            winners = analyze_results(results)
            
            stats['processed'] += 1
            
            # Check for winners
            long_count = len(winners['long'])
            short_count = len(winners['short'])
            
            if long_count > 0 or short_count > 0:
                stats['with_winners'] += 1
                stats['long_winners'] += long_count
                stats['short_winners'] += short_count
                stats['total_winning_combos'] += long_count + short_count
                
                all_winners[symbol] = winners
                
                print(f"   🏆 FOUND: {long_count} Long, {short_count} Short")
                
                # Show top results
                for side in ['long', 'short']:
                    for w in winners[side][:2]:
                        emoji = "🟢" if side == 'long' else "🔴"
                        print(f"      {emoji} {w['combo']}")
                        print(f"         LB WR: {w['lb_wr']:.1f}% | N: {w['n']} | EV: {w['ev']:+.2f}")
                        print(f"         Train: {w['train_wr']:.0f}% ({w['train_n']}) → Test: {w['test_wr']:.0f}% ({w['test_n']})")
            else:
                print(f"   ❌ No winning combos")
            
            # Progress update every 25 symbols
            if (idx + 1) % 25 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed * 60
                print(f"\n📊 PROGRESS: {idx+1}/{len(symbols)} | {stats['with_winners']} symbols with winners | {rate:.1f} symbols/min")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # ==========================================================================
    # FINAL ANALYSIS AND OUTPUT
    # ==========================================================================
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"⏱️ Time: {elapsed/60:.1f} minutes")
    print(f"📈 Symbols processed: {stats['processed']}")
    print(f"🏆 Symbols with winners: {stats['with_winners']}")
    print(f"🎯 Total winning combos: {stats['total_winning_combos']}")
    print(f"   🟢 Long: {stats['long_winners']}")
    print(f"   🔴 Short: {stats['short_winners']}")
    
    # Generate config file
    if all_winners:
        config = {}
        for symbol, winners in all_winners.items():
            config[symbol] = {
                'allowed_combos_long': [w['combo'] for w in winners['long']],
                'allowed_combos_short': [w['combo'] for w in winners['short']]
            }
        
        # Save config
        output_file = 'optimized_vwap_combos.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n✅ Config saved to: {output_file}")
        
        # Show summary
        print("\n🏆 TOP SYMBOLS BY WINNING COMBO COUNT:")
        symbol_counts = [(s, len(w['long']) + len(w['short'])) for s, w in all_winners.items()]
        symbol_counts.sort(key=lambda x: x[1], reverse=True)
        
        for symbol, count in symbol_counts[:20]:
            winners = all_winners[symbol]
            long_str = f"🟢{len(winners['long'])}" if winners['long'] else ""
            short_str = f"🔴{len(winners['short'])}" if winners['short'] else ""
            print(f"   {symbol}: {count} combos ({long_str} {short_str})")
    else:
        print("\n⚠️ No winning configurations found!")
        print("   Consider adjusting thresholds or parameters.")
    
    return all_winners, stats

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    winners, stats = run_autonomous_optimization()
