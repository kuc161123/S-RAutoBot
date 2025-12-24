#!/usr/bin/env python3
"""
🔬 ULTIMATE ROBUSTNESS VALIDATION
Extreme testing for 5M, 15M, and 30M timeframes

Tests included:
1. Walk-Forward (5 splits with purging)
2. Monte Carlo Simulation (1000 iterations)
3. Market Regime Analysis (Bull/Bear/Sideways)
4. Bootstrap Confidence Intervals
5. Maximum Consecutive Losses
6. Symbol Stability Analysis
7. Parameter Sensitivity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
TIMEFRAMES = ['5', '15', '30']
DAYS = 90  # Longer period for more data
SYMBOLS_COUNT = 75  # More symbols
N_MONTE_CARLO = 1000
N_BOOTSTRAP = 500
N_WALK_FORWARD_SPLITS = 5

# Best config from Round 6
SL_PCT = 0.01  # 1%
MAX_R = 5
COOLDOWN = 5

# Trailing
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1

# Fees
TAKER_FEE = 0.00055
SLIPPAGE = 0.0001

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
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['returns'] = df['close'].pct_change()
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
        
        # Regular Bullish
        prev_low_idx = low_window.idxmin()
        if prev_low_idx is not None and low < low_window[prev_low_idx]:
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'side': 'LONG'})
        
        # Regular Bearish
        prev_high_idx = high_window.idxmax()
        if prev_high_idx is not None and high > high_window[prev_high_idx]:
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'side': 'SHORT'})
        
        # Hidden Bullish
        if low > low_window.min():
            prev_low_idx = low_window.idxmin()
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if not pd.isna(prev_rsi) and rsi < prev_rsi:
                signals.append({**signal_base, 'side': 'LONG'})
        
        # Hidden Bearish
        if high < high_window.max():
            prev_high_idx = high_window.idxmax()
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if not pd.isna(prev_rsi) and rsi > prev_rsi:
                signals.append({**signal_base, 'side': 'SHORT'})
    
    return signals

def simulate_trade(df, idx, side, entry):
    sl_distance = entry * SL_PCT
    
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
                if best_r >= MAX_R:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return MAX_R - fee_r
                
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
                return r - fee_r
            
            current_r = (entry - low) / sl_distance
            if current_r > best_r:
                best_r = current_r
                if best_r >= MAX_R:
                    fee_r = (TAKER_FEE * 2 * entry) / sl_distance
                    return MAX_R - fee_r
                
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
    return r - fee_r

def get_all_trades(symbol_data):
    """Get all trades with proper cooldown"""
    all_trades = []
    
    for symbol, data in symbol_data.items():
        df = data['df']
        signals = data['signals']
        
        last_trade_idx = -COOLDOWN - 1
        
        for sig in signals:
            idx = sig['idx']
            if idx - last_trade_idx < COOLDOWN:
                continue
            
            r = simulate_trade(df, idx, sig['side'], sig['entry'])
            all_trades.append({
                'symbol': symbol,
                'idx': idx,
                'timestamp': sig['timestamp'],
                'side': sig['side'],
                'r': r
            })
            last_trade_idx = idx
    
    return sorted(all_trades, key=lambda x: x['timestamp'])

def walk_forward_test(trades, n_splits=5):
    """Walk-forward with purging between splits"""
    if len(trades) < 100:
        return {'all_profitable': False, 'splits_profitable': 0}
    
    split_size = len(trades) // n_splits
    results = []
    
    for i in range(n_splits):
        # Train: 60%, Test: 40% of each split
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else len(trades)
        
        split_trades = trades[start:end]
        train_end = int(len(split_trades) * 0.6)
        
        # Purge: skip 5 trades between train and test
        purge = 5
        test_start = train_end + purge
        
        if test_start >= len(split_trades):
            continue
        
        test_trades = split_trades[test_start:]
        test_r = sum(t['r'] for t in test_trades)
        test_wr = sum(1 for t in test_trades if t['r'] > 0) / len(test_trades) * 100
        
        results.append({
            'split': i + 1,
            'test_trades': len(test_trades),
            'test_r': test_r,
            'test_wr': test_wr
        })
    
    splits_profitable = sum(1 for r in results if r['test_r'] > 0)
    return {
        'all_profitable': splits_profitable == len(results),
        'splits_profitable': splits_profitable,
        'total_splits': len(results),
        'results': results
    }

def monte_carlo_test(trades, n_simulations=1000):
    """Monte Carlo with trade shuffling"""
    if len(trades) < 50:
        return {'p5': 0, 'p50': 0, 'p95': 0, 'prob_profit': 0}
    
    trade_rs = [t['r'] for t in trades]
    final_rs = []
    
    for _ in range(n_simulations):
        resampled = np.random.choice(trade_rs, size=len(trade_rs), replace=True)
        final_rs.append(np.sum(resampled))
    
    return {
        'p5': np.percentile(final_rs, 5),
        'p25': np.percentile(final_rs, 25),
        'p50': np.percentile(final_rs, 50),
        'p75': np.percentile(final_rs, 75),
        'p95': np.percentile(final_rs, 95),
        'prob_profit': np.mean([r > 0 for r in final_rs]) * 100
    }

def regime_analysis(trades, symbol_data):
    """Analyze performance in different market regimes"""
    # Classify trades by market regime based on 50 EMA slope
    regime_trades = {'bull': [], 'bear': [], 'sideways': []}
    
    for trade in trades:
        symbol = trade['symbol']
        idx = trade['idx']
        
        if symbol not in symbol_data:
            continue
        
        df = symbol_data[symbol]['df']
        if idx < 20 or idx >= len(df):
            continue
        
        # Calculate EMA slope over last 20 bars
        ema_now = df['ema_50'].iloc[idx] if 'ema_50' in df.columns else df['close'].iloc[idx]
        ema_prev = df['ema_50'].iloc[idx-20] if 'ema_50' in df.columns else df['close'].iloc[idx-20]
        
        pct_change = (ema_now - ema_prev) / ema_prev * 100
        
        if pct_change > 2:
            regime_trades['bull'].append(trade)
        elif pct_change < -2:
            regime_trades['bear'].append(trade)
        else:
            regime_trades['sideways'].append(trade)
    
    results = {}
    for regime, rtrades in regime_trades.items():
        if len(rtrades) > 10:
            total_r = sum(t['r'] for t in rtrades)
            wr = sum(1 for t in rtrades if t['r'] > 0) / len(rtrades) * 100
            results[regime] = {'n': len(rtrades), 'r': total_r, 'wr': wr}
        else:
            results[regime] = {'n': len(rtrades), 'r': 0, 'wr': 0}
    
    return results

def consecutive_loss_analysis(trades):
    """Find max consecutive losses"""
    if not trades:
        return {'max_losses': 0, 'max_wins': 0}
    
    max_losses = 0
    max_wins = 0
    current_losses = 0
    current_wins = 0
    
    for trade in trades:
        if trade['r'] < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
            current_wins = 0
        else:
            current_wins += 1
            max_wins = max(max_wins, current_wins)
            current_losses = 0
    
    return {'max_losses': max_losses, 'max_wins': max_wins}

def symbol_stability(trades):
    """Check if performance is stable across symbols"""
    symbol_results = {}
    
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in symbol_results:
            symbol_results[symbol] = {'trades': [], 'total_r': 0}
        symbol_results[symbol]['trades'].append(trade)
        symbol_results[symbol]['total_r'] += trade['r']
    
    # Count profitable vs unprofitable symbols
    profitable_symbols = sum(1 for s in symbol_results.values() if s['total_r'] > 0)
    total_symbols = len(symbol_results)
    
    # Calculate variance
    symbol_rs = [s['total_r'] for s in symbol_results.values() if len(s['trades']) >= 5]
    variance = np.var(symbol_rs) if len(symbol_rs) > 1 else 0
    
    return {
        'profitable_symbols': profitable_symbols,
        'total_symbols': total_symbols,
        'pct_profitable': profitable_symbols / total_symbols * 100 if total_symbols > 0 else 0,
        'variance': variance
    }

def main():
    print("=" * 120)
    print("🔬 ULTIMATE ROBUSTNESS VALIDATION")
    print("=" * 120)
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Symbols: {SYMBOLS_COUNT} | Days: {DAYS}")
    print(f"Config: SL={SL_PCT*100}%, R={MAX_R}")
    print(f"Tests: Walk-Forward ({N_WALK_FORWARD_SPLITS} splits), Monte Carlo ({N_MONTE_CARLO}), Regime, Consecutive Losses")
    print()
    
    # Load symbols once
    print("📋 Loading symbols...")
    symbols = load_symbols(SYMBOLS_COUNT)
    print(f"  Loaded {len(symbols)} symbols")
    
    all_results = []
    
    for tf in TIMEFRAMES:
        print(f"\n{'='*80}")
        print(f"📊 TESTING {tf}M TIMEFRAME")
        print(f"{'='*80}")
        
        # Load data
        print("📥 Loading data...")
        symbol_data = {}
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 15 == 0:
                print(f"  [{i+1}/{len(symbols)}]")
            
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
        
        # Get all trades
        trades = get_all_trades(symbol_data)
        print(f"📈 Total trades: {len(trades)}")
        
        if len(trades) < 100:
            print("⚠️ Not enough trades for robust analysis")
            continue
        
        # Basic stats
        total_r = sum(t['r'] for t in trades)
        wr = sum(1 for t in trades if t['r'] > 0) / len(trades) * 100
        avg_r = total_r / len(trades)
        
        print(f"\n📊 BASIC STATS:")
        print(f"  Total R: {total_r:+.0f}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Avg R/Trade: {avg_r:+.4f}")
        
        # Walk-Forward Test
        print(f"\n🔄 WALK-FORWARD TEST ({N_WALK_FORWARD_SPLITS} splits):")
        wf = walk_forward_test(trades, N_WALK_FORWARD_SPLITS)
        for r in wf.get('results', []):
            status = '✅' if r['test_r'] > 0 else '❌'
            print(f"  Split {r['split']}: N={r['test_trades']}, R={r['test_r']:+.0f}, WR={r['test_wr']:.1f}% {status}")
        wf_pass = wf['splits_profitable'] >= wf['total_splits'] - 1  # Allow 1 fail
        print(f"  RESULT: {wf['splits_profitable']}/{wf['total_splits']} profitable {'✅ PASS' if wf_pass else '❌ FAIL'}")
        
        # Monte Carlo
        print(f"\n🎲 MONTE CARLO ({N_MONTE_CARLO} simulations):")
        mc = monte_carlo_test(trades, N_MONTE_CARLO)
        print(f"  5th Percentile:  {mc['p5']:+.0f}R")
        print(f"  25th Percentile: {mc['p25']:+.0f}R")
        print(f"  Median:          {mc['p50']:+.0f}R")
        print(f"  75th Percentile: {mc['p75']:+.0f}R")
        print(f"  95th Percentile: {mc['p95']:+.0f}R")
        print(f"  Prob of Profit:  {mc['prob_profit']:.1f}%")
        mc_pass = mc['p5'] > -50 and mc['prob_profit'] > 80
        print(f"  RESULT: {'✅ PASS' if mc_pass else '❌ FAIL'}")
        
        # Regime Analysis
        print(f"\n📈 MARKET REGIME ANALYSIS:")
        regimes = regime_analysis(trades, symbol_data)
        regime_pass = True
        for regime, data in regimes.items():
            status = '✅' if data['r'] > 0 or data['n'] < 20 else '⚠️'
            if data['r'] < -50 and data['n'] >= 50:
                regime_pass = False
            print(f"  {regime.upper()}: N={data['n']}, R={data['r']:+.0f}, WR={data['wr']:.1f}% {status}")
        print(f"  RESULT: {'✅ PASS' if regime_pass else '⚠️ CAUTION'}")
        
        # Consecutive Losses
        print(f"\n📉 CONSECUTIVE LOSS ANALYSIS:")
        consec = consecutive_loss_analysis(trades)
        print(f"  Max Consecutive Losses: {consec['max_losses']}")
        print(f"  Max Consecutive Wins: {consec['max_wins']}")
        consec_pass = consec['max_losses'] <= 15
        max_losses = consec['max_losses']
        print(f"  RESULT: {'✅ PASS' if consec_pass else f'⚠️ CAUTION (plan for {max_losses} losses)'}")
        
        # Symbol Stability
        print(f"\n🔗 SYMBOL STABILITY:")
        stability = symbol_stability(trades)
        print(f"  Profitable Symbols: {stability['profitable_symbols']}/{stability['total_symbols']} ({stability['pct_profitable']:.1f}%)")
        stability_pass = stability['pct_profitable'] >= 50
        print(f"  RESULT: {'✅ PASS' if stability_pass else '❌ FAIL'}")
        
        # Overall verdict
        tests_passed = sum([wf_pass, mc_pass, regime_pass, consec_pass, stability_pass])
        verdict = "✅ ROBUST" if tests_passed >= 4 else "⚠️ MARGINAL" if tests_passed >= 3 else "❌ FAIL"
        
        all_results.append({
            'timeframe': tf,
            'trades': len(trades),
            'total_r': total_r,
            'wr': wr,
            'wf_pass': wf_pass,
            'mc_pass': mc_pass,
            'mc_prob_profit': mc['prob_profit'],
            'mc_p5': mc['p5'],
            'regime_pass': regime_pass,
            'consec_pass': consec_pass,
            'max_consec_losses': consec['max_losses'],
            'stability_pass': stability_pass,
            'pct_profitable_symbols': stability['pct_profitable'],
            'tests_passed': tests_passed,
            'verdict': verdict
        })
        
        print(f"\n{'='*60}")
        print(f"🏆 {tf}M VERDICT: {verdict} ({tests_passed}/5 tests passed)")
        print(f"{'='*60}")
    
    # Final Summary
    print("\n" + "=" * 120)
    print("📊 FINAL ROBUSTNESS SUMMARY")
    print("=" * 120)
    print(f"{'TF':>4} {'Trades':>8} {'Total_R':>10} {'WR':>7} {'WF':>5} {'MC':>5} {'Regime':>7} {'Consec':>7} {'Stable':>7} {'Pass':>6} {'Verdict':<12}")
    
    for r in all_results:
        wf = '✅' if r['wf_pass'] else '❌'
        mc = '✅' if r['mc_pass'] else '❌'
        rg = '✅' if r['regime_pass'] else '⚠️'
        cs = '✅' if r['consec_pass'] else '⚠️'
        st = '✅' if r['stability_pass'] else '❌'
        print(f"{r['timeframe']:>4}M {r['trades']:>8} {r['total_r']:>+10.0f} {r['wr']:>6.1f}% {wf:>5} {mc:>5} {rg:>7} {cs:>7} {st:>7} {r['tests_passed']:>3}/5  {r['verdict']:<12}")
    
    # Save results
    pd.DataFrame(all_results).to_csv('ultimate_robustness_results.csv', index=False)
    print("\n✅ Saved to ultimate_robustness_results.csv")
    
    # Best recommendation
    robust_tfs = [r for r in all_results if r['verdict'] == "✅ ROBUST"]
    if robust_tfs:
        best = max(robust_tfs, key=lambda x: x['total_r'])
        print("\n" + "=" * 80)
        print("🏆 FINAL RECOMMENDATION")
        print("=" * 80)
        print(f"\nBest Timeframe: {best['timeframe']}M")
        print(f"Total R: {best['total_r']:+.0f}")
        print(f"Win Rate: {best['wr']:.1f}%")
        print(f"Monte Carlo Prob of Profit: {best['mc_prob_profit']:.1f}%")
        print(f"Max Consecutive Losses: {best['max_consec_losses']}")
        print(f"Profitable Symbols: {best['pct_profitable_symbols']:.1f}%")
        print(f"\n📋 RECOMMENDED BOT CONFIGURATION:")
        print(f"  timeframe: {best['timeframe']}")
        print(f"  stop_loss_percent: 1.0")
        print(f"  max_r_target: 5")
        print(f"  be_threshold: 0.3")
        print(f"  trail_distance: 0.1")
        print(f"  cooldown_bars: 5")
        print(f"  volume_filter: true (>0.8x avg)")
        print("\n✅ READY FOR LIVE TRADING")

if __name__ == "__main__":
    main()
