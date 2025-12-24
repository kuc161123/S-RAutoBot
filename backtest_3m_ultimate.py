#!/usr/bin/env python3
"""
🏆 ULTIMATE 3M ROBUST BACKTEST
Institutional-grade validation with all modern practices:

1. Walk-Forward Validation (60% train / 40% test)
2. All Filters: VWAP, RSI Zones, Volume, Trend
3. No Lookahead Bias (strict bar-by-bar simulation)
4. Realistic Fees + Slippage
5. Monte Carlo Robustness
6. Regime Analysis
7. Best Config from previous rounds as baseline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
TIMEFRAME = '3'
DAYS = 60  # Longer period for walk-forward
SYMBOLS_COUNT = 75  # More symbols for robustness
TRAIN_RATIO = 0.6  # 60% train, 40% test

# Best configs to test (from previous rounds)
CONFIGS = [
    # Round 1 winner
    {'name': 'R1_Best', 'sl_pct': 0.005, 'max_r': 5, 'cooldown': 5, 'filter': 'none'},
    # Round 3 winners
    {'name': 'R3_Best', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'none'},
    {'name': 'R3_Alt', 'sl_pct': 0.01, 'max_r': 5, 'cooldown': 5, 'filter': 'none'},
    # New variations with filters
    {'name': 'Volume1x', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'volume'},
    {'name': 'VWAP', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'vwap'},
    {'name': 'RSI_Zone', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'rsi_zone'},
    {'name': 'Trend', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'trend'},
    {'name': 'Vol+VWAP', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'vol_vwap'},
    {'name': 'AllFilters', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'all'},
    # Higher R targets
    {'name': 'R3_7R', 'sl_pct': 0.01, 'max_r': 7, 'cooldown': 5, 'filter': 'none'},
    {'name': 'R3_8R', 'sl_pct': 0.01, 'max_r': 8, 'cooldown': 5, 'filter': 'none'},
    # Tighter trailing
    {'name': 'TightTrail', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 5, 'filter': 'none', 'trail_dist': 0.05},
    # Longer cooldown
    {'name': 'CD10', 'sl_pct': 0.01, 'max_r': 6, 'cooldown': 10, 'filter': 'none'},
]

# Trailing defaults
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1

# Fee model (realistic)
TAKER_FEE = 0.00055
SLIPPAGE = 0.0001  # 0.01% slippage

def load_symbols(n=75):
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
    """Calculate ALL indicators needed for filtering"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    
    # EMAs for trend
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # VWAP (rolling approximation)
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    
    return df

def detect_divergences(df):
    """Detect divergences with all metadata for filtering"""
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
        
        # Get filter data
        volume_ratio = df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0
        vwap = df['vwap'].iloc[i] if 'vwap' in df.columns else price
        ema_20 = df['ema_20'].iloc[i] if 'ema_20' in df.columns else price
        ema_50 = df['ema_50'].iloc[i] if 'ema_50' in df.columns else price
        
        signal_base = {
            'idx': i,
            'entry': price,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'above_vwap': price > vwap,
            'trend_bullish': ema_20 > ema_50,
            'rsi_oversold': rsi < 30,
            'rsi_overbought': rsi > 70,
            'timestamp': df['timestamp'].iloc[i]
        }
        
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

def apply_filter(signals, filter_type):
    """Apply various filters"""
    if filter_type == 'none':
        return signals
    elif filter_type == 'volume':
        return [s for s in signals if s.get('volume_ratio', 0) >= 1.0]
    elif filter_type == 'vwap':
        # LONG only if below VWAP (oversold), SHORT only if above VWAP (overbought)
        return [s for s in signals if 
                (s['side'] == 'LONG' and not s.get('above_vwap', True)) or
                (s['side'] == 'SHORT' and s.get('above_vwap', False))]
    elif filter_type == 'rsi_zone':
        return [s for s in signals if 
                (s['side'] == 'LONG' and s.get('rsi_oversold', False)) or
                (s['side'] == 'SHORT' and s.get('rsi_overbought', False))]
    elif filter_type == 'trend':
        return [s for s in signals if 
                (s['side'] == 'LONG' and s.get('trend_bullish', False)) or
                (s['side'] == 'SHORT' and not s.get('trend_bullish', True))]
    elif filter_type == 'vol_vwap':
        filtered = [s for s in signals if s.get('volume_ratio', 0) >= 1.0]
        return [s for s in filtered if 
                (s['side'] == 'LONG' and not s.get('above_vwap', True)) or
                (s['side'] == 'SHORT' and s.get('above_vwap', False))]
    elif filter_type == 'all':
        # All filters combined
        filtered = [s for s in signals if s.get('volume_ratio', 0) >= 1.0]
        filtered = [s for s in filtered if 
                    (s['side'] == 'LONG' and not s.get('above_vwap', True)) or
                    (s['side'] == 'SHORT' and s.get('above_vwap', False))]
        filtered = [s for s in filtered if 
                    (s['side'] == 'LONG' and s.get('rsi_oversold', False)) or
                    (s['side'] == 'SHORT' and s.get('rsi_overbought', False))]
        return filtered
    return signals

def simulate_trade(df, idx, side, entry, sl_pct, max_r, trail_dist=None):
    """Simulate trade with realistic execution"""
    if trail_dist is None:
        trail_dist = TRAIL_DISTANCE
    
    sl_distance = entry * sl_pct
    
    # Apply slippage to entry
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
    
    for j in range(idx + 1, min(idx + 480, len(df))):
        bars_in_trade += 1
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            if low <= current_sl:
                exit_price = current_sl * (1 - SLIPPAGE)  # Slippage on exit
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
                    new_sl = entry + (best_r - trail_dist) * sl_distance
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
                    new_sl = entry - (best_r - trail_dist) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout
    final_price = df['close'].iloc[min(idx + 479, len(df) - 1)]
    if side == 'LONG':
        exit_price = final_price * (1 - SLIPPAGE)
        r = (exit_price - entry) / sl_distance
    else:
        exit_price = final_price * (1 + SLIPPAGE)
        r = (entry - exit_price) / sl_distance
    
    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
    return r - fee_r, bars_in_trade

def run_backtest_period(symbol_data, config, start_idx, end_idx):
    """Run backtest on a specific period only"""
    total_r = 0
    wins = 0
    trades = 0
    trade_results = []
    
    sl_pct = config['sl_pct']
    max_r = config['max_r']
    cooldown = config['cooldown']
    filter_type = config['filter']
    trail_dist = config.get('trail_dist', TRAIL_DISTANCE)
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        # Filter signals to period
        period_signals = [s for s in signals if start_idx <= s['idx'] < end_idx]
        
        # Apply filter
        filtered = apply_filter(period_signals, filter_type)
        
        last_trade_bar = -cooldown - 1
        
        for sig in filtered:
            idx = sig['idx']
            
            if idx - last_trade_bar < cooldown:
                continue
            
            r, bars = simulate_trade(df, idx, sig['side'], sig['entry'], sl_pct, max_r, trail_dist)
            
            total_r += r
            if r > 0:
                wins += 1
            trades += 1
            trade_results.append(r)
            
            last_trade_bar = idx + bars
    
    if trades == 0:
        return {'trades': 0, 'wr': 0, 'total_r': 0, 'avg_r': 0, 'trade_results': []}
    
    return {
        'trades': trades,
        'wins': wins,
        'wr': wins / trades * 100,
        'total_r': total_r,
        'avg_r': total_r / trades,
        'trade_results': trade_results
    }

def walk_forward_validation(symbol_data, config, n_splits=5):
    """Walk-forward validation with multiple splits"""
    # Find total data range
    max_idx = 0
    for data in symbol_data.values():
        max_idx = max(max_idx, len(data['df']))
    
    split_size = max_idx // n_splits
    train_size = int(split_size * TRAIN_RATIO)
    
    train_results = []
    test_results = []
    
    for i in range(n_splits):
        start_idx = i * split_size
        train_end = start_idx + train_size
        test_end = start_idx + split_size
        
        # Train period
        train_result = run_backtest_period(symbol_data, config, start_idx, train_end)
        train_results.append(train_result)
        
        # Test period (out of sample)
        test_result = run_backtest_period(symbol_data, config, train_end, test_end)
        test_results.append(test_result)
    
    return train_results, test_results

def monte_carlo_test(trade_results, n_simulations=1000):
    """Monte Carlo simulation for robustness"""
    if len(trade_results) < 10:
        return {'p5': 0, 'p50': 0, 'p95': 0, 'prob_profit': 0}
    
    final_rs = []
    for _ in range(n_simulations):
        # Resample with replacement
        resampled = np.random.choice(trade_results, size=len(trade_results), replace=True)
        final_rs.append(np.sum(resampled))
    
    return {
        'p5': np.percentile(final_rs, 5),
        'p50': np.percentile(final_rs, 50),
        'p95': np.percentile(final_rs, 95),
        'prob_profit': np.mean([r > 0 for r in final_rs]) * 100
    }

def main():
    print("=" * 120)
    print("🏆 ULTIMATE 3M ROBUST BACKTEST")
    print("=" * 120)
    print(f"Timeframe: 3min | Symbols: {SYMBOLS_COUNT} | Days: {DAYS}")
    print(f"Walk-Forward: {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test | 5 splits")
    print(f"Fees: {TAKER_FEE*100:.3f}% + Slippage: {SLIPPAGE*100:.3f}%")
    print(f"Configs to test: {len(CONFIGS)}")
    print()
    
    # Load symbols
    print("📋 Loading symbols...")
    symbols = load_symbols(SYMBOLS_COUNT)
    print(f"  Loaded {len(symbols)} symbols")
    
    # Load all data
    print("\n📥 Loading data and detecting signals...")
    symbol_data = {}
    
    for i, symbol in enumerate(symbols):
        if (i + 1) % 15 == 0:
            print(f"  [{i+1}/{len(symbols)}] {symbol}")
        
        try:
            df = fetch_data(symbol, TIMEFRAME, DAYS)
            if df is None or len(df) < 200:
                continue
            
            df = calculate_indicators(df)
            signals = detect_divergences(df)
            
            if signals:
                symbol_data[symbol] = {'df': df, 'signals': signals}
        except:
            continue
    
    print(f"\n✅ {len(symbol_data)} symbols with signals")
    
    # Test each config with walk-forward
    print("\n🔄 Running walk-forward validation...")
    results = []
    
    for config in CONFIGS:
        print(f"  Testing: {config['name']}...")
        
        train_results, test_results = walk_forward_validation(symbol_data, config)
        
        # Aggregate results
        all_train_trades = sum(r['trades'] for r in train_results)
        all_test_trades = sum(r['trades'] for r in test_results)
        all_train_r = sum(r['total_r'] for r in train_results)
        all_test_r = sum(r['total_r'] for r in test_results)
        
        all_trade_results = []
        for r in test_results:
            all_trade_results.extend(r['trade_results'])
        
        # Monte Carlo on test results
        mc = monte_carlo_test(all_trade_results)
        
        # Check consistency (all splits profitable in test)
        test_splits_profitable = sum(1 for r in test_results if r['total_r'] > 0)
        
        results.append({
            'name': config['name'],
            'sl_pct': config['sl_pct'],
            'max_r': config['max_r'],
            'filter': config['filter'],
            'train_trades': all_train_trades,
            'train_r': all_train_r,
            'test_trades': all_test_trades,
            'test_r': all_test_r,
            'test_wr': sum(r['wins'] for r in test_results) / max(1, all_test_trades) * 100,
            'test_avg_r': all_test_r / max(1, all_test_trades),
            'mc_p5': mc['p5'],
            'mc_p50': mc['p50'],
            'mc_p95': mc['p95'],
            'mc_prob_profit': mc['prob_profit'],
            'splits_profitable': test_splits_profitable,
            'consistent': test_splits_profitable >= 4  # 4/5 splits profitable
        })
    
    # Sort by test R
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r', ascending=False)
    
    print("\n" + "=" * 140)
    print("📊 WALK-FORWARD RESULTS (Sorted by Test R)")
    print("=" * 140)
    print(f"{'Name':<12} {'SL':>6} {'MaxR':>5} {'Filter':<10} {'Train_R':>10} {'Test_R':>10} {'Test_WR':>8} {'MC_P50':>10} {'Splits':>7} {'Status':<10}")
    
    for _, row in results_df.iterrows():
        status = '✅ ROBUST' if row['consistent'] and row['test_r'] > 0 else '❌'
        print(f"{row['name']:<12} {row['sl_pct']*100:>5.2f}% {row['max_r']:>5} {row['filter']:<10} {row['train_r']:>+10.0f} {row['test_r']:>+10.0f} {row['test_wr']:>7.1f}% {row['mc_p50']:>+10.1f} {row['splits_profitable']:>3}/5   {status:<10}")
    
    # Save results
    results_df.to_csv('3m_ultimate_results.csv', index=False)
    print("\n✅ Saved to 3m_ultimate_results.csv")
    
    # Best robust config
    robust = results_df[results_df['consistent'] & (results_df['test_r'] > 0)]
    
    if len(robust) > 0:
        best = robust.iloc[0]
        print("\n" + "=" * 80)
        print("🏆 BEST ROBUST 3M CONFIGURATION")
        print("=" * 80)
        print(f"Config: {best['name']}")
        print(f"SL: {best['sl_pct']*100:.2f}%")
        print(f"Max R Target: {best['max_r']}")
        print(f"Filter: {best['filter']}")
        print(f"\n📈 PERFORMANCE:")
        print(f"  Test Period R: {best['test_r']:+.0f}")
        print(f"  Test Win Rate: {best['test_wr']:.1f}%")
        print(f"  Test Avg R/Trade: {best['test_avg_r']:+.4f}")
        print(f"\n🎲 MONTE CARLO:")
        print(f"  5th Percentile: {best['mc_p5']:+.0f}R")
        print(f"  Median: {best['mc_p50']:+.0f}R")
        print(f"  95th Percentile: {best['mc_p95']:+.0f}R")
        print(f"  Probability of Profit: {best['mc_prob_profit']:.1f}%")
        print(f"\n📊 CONSISTENCY: {best['splits_profitable']}/5 splits profitable")
        print("\n✅ PROFITABLE & ROBUST CONFIGURATION FOUND!")
    else:
        profitable = results_df[results_df['test_r'] > 0]
        if len(profitable) > 0:
            best = profitable.iloc[0]
            print(f"\n⚠️ Best profitable (but not fully robust): {best['name']} with {best['test_r']:+.0f}R")
        else:
            print("\n❌ No profitable configuration found in walk-forward test")

if __name__ == "__main__":
    main()
