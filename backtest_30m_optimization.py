#!/usr/bin/env python3
"""
🏆 30M COMPREHENSIVE OPTIMIZATION
Find the absolute best 30M configuration with extreme robustness validation.

Tests ALL parameter combinations:
- SL sizes: 0.8%, 1.0%, 1.2%, 1.5%, 2.0% (5)
- R:R targets: 3, 4, 5, 6, 7 (5)
- BE thresholds: 0.2, 0.3, 0.5 (3)
- Trail distances: 0.05, 0.1, 0.2 (3)
- Cooldowns: 3, 5, 10 (3)

Total: 5 * 5 * 3 * 3 * 3 = 675 configurations
Top 20 tested with extreme validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
TIMEFRAME = '30'
DAYS = 90
SYMBOLS_COUNT = 100  # 75 in-sample + 25 OOS

# Parameters to test
SL_PCTS = [0.008, 0.01, 0.012, 0.015, 0.02]  # 0.8% - 2%
MAX_RS = [3, 4, 5, 6, 7]
BE_THRESHOLDS = [0.2, 0.3, 0.5]
TRAIL_DISTANCES = [0.05, 0.1, 0.2]
COOLDOWNS = [3, 5, 10]

# Validation settings
N_WALK_FORWARD = 5
N_MONTE_CARLO = 500

# Fees
TAKER_FEE = 0.00055
SLIPPAGE = 0.0001

def load_symbols(n=100):
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
        
        if pd.isna(rsi) or volume_ratio < 0.8:
            continue
        
        low_window = df['low'].iloc[i-lookback:i]
        high_window = df['high'].iloc[i-lookback:i]
        
        signal_base = {'idx': i, 'entry': price, 'timestamp': df['timestamp'].iloc[i]}
        
        # All 4 divergence types
        prev_low_idx = low_window.idxmin()
        if prev_low_idx is not None and low < low_window[prev_low_idx]:
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'side': 'LONG'})
        
        prev_high_idx = high_window.idxmax()
        if prev_high_idx is not None and high > high_window[prev_high_idx]:
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'side': 'SHORT'})
        
        if low > low_window.min():
            prev_low_idx = low_window.idxmin()
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'side': 'LONG'})
        
        if high < high_window.max():
            prev_high_idx = high_window.idxmax()
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'side': 'SHORT'})
    
    return signals

def simulate_trade(df, idx, side, entry, sl_pct, max_r, be_threshold, trail_distance):
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
    
    for j in range(idx + 1, min(idx + 200, len(df))):
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        
        if side == 'LONG':
            if low <= current_sl:
                exit_price = current_sl * (1 - SLIPPAGE)
                r = (exit_price - entry) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r
            
            current_r = (high - entry) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return max_r - fee_r
                
                if best_r >= be_threshold and not be_moved:
                    current_sl = entry + (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry + (best_r - trail_distance) * sl_distance
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                exit_price = current_sl * (1 + SLIPPAGE)
                r = (entry - exit_price) / sl_distance
                fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                return r - fee_r
            
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= max_r:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return max_r - fee_r
                
                if best_r >= be_threshold and not be_moved:
                    current_sl = entry - (sl_distance * 0.01)
                    be_moved = True
                
                if be_moved:
                    new_sl = entry - (best_r - trail_distance) * sl_distance
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    final_price = df['close'].iloc[min(idx + 199, len(df) - 1)]
    if side == 'LONG':
        r = (final_price - entry) / sl_distance
    else:
        r = (entry - final_price) / sl_distance
    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
    return r - fee_r

def get_all_trades(symbol_data, sl_pct, max_r, be_threshold, trail_distance, cooldown):
    all_trades = []
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        last_trade_idx = -cooldown - 1
        
        for sig in signals:
            idx = sig['idx']
            if idx - last_trade_idx < cooldown:
                continue
            
            r = simulate_trade(df, idx, sig['side'], sig['entry'], sl_pct, max_r, be_threshold, trail_distance)
            all_trades.append({
                'symbol': symbol,
                'idx': idx,
                'timestamp': sig['timestamp'],
                'r': r
            })
            last_trade_idx = idx
    
    return sorted(all_trades, key=lambda x: x['timestamp'])

def walk_forward_test(trades, n_splits=5):
    if len(trades) < 100:
        return 0, 0
    
    split_size = len(trades) // n_splits
    test_results = []
    
    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else len(trades)
        split_trades = trades[start:end]
        
        train_end = int(len(split_trades) * 0.6)
        test_trades = split_trades[train_end + 5:]  # Purge 5 trades
        
        if len(test_trades) > 10:
            test_r = sum(t['r'] for t in test_trades)
            test_results.append(test_r)
    
    profitable_splits = sum(1 for r in test_results if r > 0)
    total_test_r = sum(test_results)
    return profitable_splits, total_test_r

def monte_carlo(trades, n_simulations=500):
    if len(trades) < 50:
        return 0, 0
    
    trade_rs = [t['r'] for t in trades]
    final_rs = []
    
    for _ in range(n_simulations):
        resampled = np.random.choice(trade_rs, size=len(trade_rs), replace=True)
        final_rs.append(np.sum(resampled))
    
    return np.percentile(final_rs, 5), np.mean([r > 0 for r in final_rs]) * 100

def quick_eval(symbol_data, sl_pct, max_r, be_threshold, trail_distance, cooldown):
    """Quick evaluation for initial screening"""
    trades = get_all_trades(symbol_data, sl_pct, max_r, be_threshold, trail_distance, cooldown)
    
    if len(trades) < 100:
        return {'total_r': -9999, 'wr': 0, 'trades': 0}
    
    total_r = sum(t['r'] for t in trades)
    wr = sum(1 for t in trades if t['r'] > 0) / len(trades) * 100
    
    return {
        'total_r': total_r,
        'wr': wr,
        'trades': len(trades),
        'avg_r': total_r / len(trades)
    }

def full_validation(symbol_data, symbol_data_oos, config):
    """Full validation for top configs"""
    sl_pct, max_r, be_threshold, trail_distance, cooldown = config
    
    # Get trades
    trades = get_all_trades(symbol_data, sl_pct, max_r, be_threshold, trail_distance, cooldown)
    trades_oos = get_all_trades(symbol_data_oos, sl_pct, max_r, be_threshold, trail_distance, cooldown)
    
    if len(trades) < 100:
        return None
    
    # Basic stats
    total_r = sum(t['r'] for t in trades)
    wr = sum(1 for t in trades if t['r'] > 0) / len(trades) * 100
    
    # Walk-forward
    wf_splits, wf_r = walk_forward_test(trades, N_WALK_FORWARD)
    
    # Monte Carlo
    mc_p5, mc_prob = monte_carlo(trades, N_MONTE_CARLO)
    
    # OOS
    oos_r = sum(t['r'] for t in trades_oos) if trades_oos else 0
    oos_n = len(trades_oos)
    
    # Max consecutive losses
    max_losses = 0
    current_losses = 0
    for t in trades:
        if t['r'] < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    
    return {
        'sl_pct': sl_pct,
        'max_r': max_r,
        'be_threshold': be_threshold,
        'trail_distance': trail_distance,
        'cooldown': cooldown,
        'trades': len(trades),
        'total_r': total_r,
        'wr': wr,
        'avg_r': total_r / len(trades),
        'wf_splits': wf_splits,
        'wf_r': wf_r,
        'mc_p5': mc_p5,
        'mc_prob': mc_prob,
        'oos_n': oos_n,
        'oos_r': oos_r,
        'max_losses': max_losses
    }

def main():
    print("=" * 120)
    print("🏆 30M COMPREHENSIVE OPTIMIZATION")
    print("=" * 120)
    
    total_configs = len(SL_PCTS) * len(MAX_RS) * len(BE_THRESHOLDS) * len(TRAIL_DISTANCES) * len(COOLDOWNS)
    print(f"Total configurations: {total_configs}")
    print(f"SL: {[f'{s*100:.1f}%' for s in SL_PCTS]}")
    print(f"Max R: {MAX_RS}")
    print(f"BE Threshold: {BE_THRESHOLDS}")
    print(f"Trail Distance: {TRAIL_DISTANCES}")
    print(f"Cooldown: {COOLDOWNS}")
    print()
    
    # Load symbols
    print("📋 Loading symbols...")
    all_symbols = load_symbols(SYMBOLS_COUNT)
    in_sample = all_symbols[:75]
    oos = all_symbols[75:]
    print(f"  In-sample: {len(in_sample)} | OOS: {len(oos)}")
    
    # Load in-sample data
    print("\n📥 Loading in-sample data...")
    symbol_data = {}
    for i, symbol in enumerate(in_sample):
        if (i + 1) % 15 == 0:
            print(f"  [{i+1}/{len(in_sample)}]")
        try:
            df = fetch_data(symbol, TIMEFRAME, DAYS)
            if df is None or len(df) < 100:
                continue
            df = calculate_indicators(df)
            signals = detect_divergences(df)
            if signals:
                symbol_data[symbol] = {'df': df, 'signals': signals}
        except:
            continue
    
    # Load OOS data
    print("📥 Loading OOS data...")
    symbol_data_oos = {}
    for symbol in oos:
        try:
            df = fetch_data(symbol, TIMEFRAME, DAYS)
            if df is None or len(df) < 100:
                continue
            df = calculate_indicators(df)
            signals = detect_divergences(df)
            if signals:
                symbol_data_oos[symbol] = {'df': df, 'signals': signals}
        except:
            continue
    
    print(f"✅ In-sample: {len(symbol_data)} | OOS: {len(symbol_data_oos)}")
    
    # Phase 1: Quick screening of all configs
    print(f"\n🔄 Phase 1: Quick screening {total_configs} configs...")
    all_results = []
    configs = list(product(SL_PCTS, MAX_RS, BE_THRESHOLDS, TRAIL_DISTANCES, COOLDOWNS))
    
    for i, (sl, r, be, trail, cd) in enumerate(configs):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(configs)}]")
        
        result = quick_eval(symbol_data, sl, r, be, trail, cd)
        result['config'] = (sl, r, be, trail, cd)
        all_results.append(result)
    
    # Sort by total R and get top 30
    all_results.sort(key=lambda x: x['total_r'], reverse=True)
    top_configs = all_results[:30]
    
    print(f"\n✅ Top 30 configs identified")
    
    # Phase 2: Full validation of top configs
    print(f"\n🔬 Phase 2: Full validation of top 30...")
    validated = []
    
    for i, result in enumerate(top_configs):
        print(f"  [{i+1}/30] Testing SL={result['config'][0]*100:.1f}% R={result['config'][1]} BE={result['config'][2]} Trail={result['config'][3]} CD={result['config'][4]}")
        
        val = full_validation(symbol_data, symbol_data_oos, result['config'])
        if val:
            validated.append(val)
    
    # Sort validated by walk-forward R
    validated.sort(key=lambda x: x['wf_r'], reverse=True)
    
    print("\n" + "=" * 150)
    print("📊 TOP 20 VALIDATED CONFIGURATIONS (Sorted by Walk-Forward R)")
    print("=" * 150)
    print(f"{'SL':>6} {'MaxR':>5} {'BE':>5} {'Trail':>6} {'CD':>4} {'N':>7} {'Total_R':>10} {'WR':>7} {'WF_R':>10} {'WF':>5} {'MC_P5':>8} {'OOS_R':>8} {'MaxLoss':>8}")
    
    for v in validated[:20]:
        wf_status = '✅' if v['wf_splits'] >= 4 else '⚠️' if v['wf_splits'] >= 3 else '❌'
        print(f"{v['sl_pct']*100:>5.1f}% {v['max_r']:>5} {v['be_threshold']:>5.1f} {v['trail_distance']:>6.2f} {v['cooldown']:>4} {v['trades']:>7} {v['total_r']:>+10.0f} {v['wr']:>6.1f}% {v['wf_r']:>+10.0f} {v['wf_splits']:>2}/5{wf_status} {v['mc_p5']:>+8.0f} {v['oos_r']:>+8.0f} {v['max_losses']:>8}")
    
    # Best config
    if validated:
        # Filter for robust (WF >= 4, OOS > 0)
        robust = [v for v in validated if v['wf_splits'] >= 4 and v['oos_r'] > 0]
        
        if robust:
            best = robust[0]
        else:
            best = validated[0]
        
        print("\n" + "=" * 100)
        print("🏆 BEST CONFIGURATION")
        print("=" * 100)
        print(f"\nStop Loss: {best['sl_pct']*100:.1f}%")
        print(f"Max R Target: {best['max_r']}")
        print(f"BE Threshold: {best['be_threshold']}R")
        print(f"Trail Distance: {best['trail_distance']}R")
        print(f"Cooldown: {best['cooldown']} bars")
        print(f"\n📈 PERFORMANCE:")
        print(f"  Total R: {best['total_r']:+.0f}")
        print(f"  Win Rate: {best['wr']:.1f}%")
        print(f"  Walk-Forward R: {best['wf_r']:+.0f}")
        print(f"  Walk-Forward Splits: {best['wf_splits']}/5")
        print(f"  Monte Carlo P5: {best['mc_p5']:+.0f}")
        print(f"  Monte Carlo Prob: {best['mc_prob']:.1f}%")
        print(f"  OOS R: {best['oos_r']:+.0f} ({best['oos_n']} trades)")
        print(f"  Max Consecutive Losses: {best['max_losses']}")
        
        # Config YAML
        print("\n📋 RECOMMENDED BOT CONFIGURATION:")
        print(f"""
timeframe: 30
stop_loss_percent: {best['sl_pct']*100:.1f}
max_r_target: {best['max_r']}
be_threshold: {best['be_threshold']}
trail_distance: {best['trail_distance']}
cooldown_bars: {best['cooldown']}
volume_filter: true
risk_percent: 0.5
""")
        
    # Save all results
    pd.DataFrame(validated).to_csv('30m_optimization_results.csv', index=False)
    print("\n✅ Saved to 30m_optimization_results.csv")

if __name__ == "__main__":
    main()
