#!/usr/bin/env python3
"""
Deep Verification: 5M vs 30M Timeframes
========================================

Extended verification testing:
- Monte Carlo simulation (1000 runs)
- Multiple R:R ratios (3R, 4R, 5R)
- Symbol-by-symbol performance
- Drawdown analysis
- Win/Loss streaks
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

TIMEFRAMES = [5, 30]  # Focus on these two
SL_PCT = 1.2
RR_RATIOS = [3.0, 4.0, 5.0]  # Test multiple
COOLDOWN_BARS = 3
DATA_DAYS = 90
WF_SPLITS = 5
MONTE_CARLO_RUNS = 1000

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
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_divergences(df: pd.DataFrame) -> pd.DataFrame:
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
                   sl_distance: float, tp_r: float) -> tuple:
    tp_price = entry + (tp_r * sl_distance) if side == 'long' else entry - (tp_r * sl_distance)
    sl_price = entry - sl_distance if side == 'long' else entry + sl_distance
    
    for i in range(idx + 1, min(idx + 1000, len(df))):
        candle = df.iloc[i]
        bars_held = i - idx
        
        if side == 'long':
            if candle['low'] <= sl_price:
                return (-1.0, "SL", bars_held)
            if candle['high'] >= tp_price:
                return (tp_r, "TP", bars_held)
        else:
            if candle['high'] >= sl_price:
                return (-1.0, "SL", bars_held)
            if candle['low'] <= tp_price:
                return (tp_r, "TP", bars_held)
    
    final_candle = df.iloc[min(idx + 999, len(df) - 1)]
    if side == 'long':
        exit_r = (final_candle['close'] - entry) / sl_distance
    else:
        exit_r = (entry - final_candle['close']) / sl_distance
    
    return (exit_r, "TIMEOUT", 1000)

def backtest_symbol(symbol: str, df: pd.DataFrame, tp_r: float) -> list:
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
            
            exit_r, exit_reason, bars_held = simulate_trade(df, idx, side, entry, sl_distance, tp_r)
            
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

def monte_carlo(r_values: list, n_sims: int = 1000) -> dict:
    """Run Monte Carlo simulation"""
    if not r_values:
        return {'p5': 0, 'p50': 0, 'p95': 0, 'prob_profit': 0}
    
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

def calculate_drawdown(r_values: list) -> dict:
    """Calculate max drawdown"""
    if not r_values:
        return {'max_dd': 0, 'avg_dd': 0}
    
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    
    return {
        'max_dd': round(min(drawdowns), 1),
        'avg_dd': round(np.mean(drawdowns[drawdowns < 0]), 1) if len(drawdowns[drawdowns < 0]) > 0 else 0
    }

def analyze_streaks(r_values: list) -> dict:
    """Analyze win/loss streaks"""
    if not r_values:
        return {'max_win_streak': 0, 'max_loss_streak': 0}
    
    wins = [1 if r > 0 else 0 for r in r_values]
    
    max_win = max_loss = current_win = current_loss = 0
    
    for w in wins:
        if w == 1:
            current_win += 1
            current_loss = 0
            max_win = max(max_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss = max(max_loss, current_loss)
    
    return {'max_win_streak': max_win, 'max_loss_streak': max_loss}

def run_deep_test(symbols: list, timeframe: int) -> dict:
    """Run comprehensive test for a timeframe"""
    print(f"\n{'='*70}")
    print(f"DEEP VERIFICATION: {timeframe}M TIMEFRAME")
    print(f"{'='*70}")
    
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
    
    results = {}
    
    # Test each R:R ratio
    for tp_r in RR_RATIOS:
        print(f"\n--- Testing {tp_r}R Take Profit ---")
        
        all_trades = []
        wf_results = []
        symbol_stats = {}
        
        # Walk-forward
        for fold in range(WF_SPLITS):
            fold_trades = []
            
            for sym, df in symbol_data.items():
                fold_size = len(df) // WF_SPLITS
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < WF_SPLITS - 1 else len(df)
                
                fold_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
                trades = backtest_symbol(sym, fold_df, tp_r)
                fold_trades.extend(trades)
                
                # Track per-symbol stats
                if sym not in symbol_stats:
                    symbol_stats[sym] = []
                symbol_stats[sym].extend(trades)
            
            all_trades.extend(fold_trades)
            
            if fold_trades:
                trades_df = pd.DataFrame(fold_trades)
                total_r = trades_df['exit_r'].sum()
                wins = (trades_df['exit_r'] > 0).sum()
                wf_results.append({'fold': fold + 1, 'total_r': total_r, 'wr': wins / len(trades_df) * 100})
                print(f"  Fold {fold+1}: {len(fold_trades)} trades, {total_r:+.1f}R")
        
        if all_trades:
            all_df = pd.DataFrame(all_trades)
            r_values = all_df['exit_r'].tolist()
            
            # Basic stats
            total_r = sum(r_values)
            wins = sum(1 for r in r_values if r > 0)
            wr = wins / len(r_values) * 100
            avg_r = np.mean(r_values)
            
            # Monte Carlo
            mc = monte_carlo(r_values, MONTE_CARLO_RUNS)
            
            # Drawdown
            dd = calculate_drawdown(r_values)
            
            # Streaks
            streaks = analyze_streaks(r_values)
            
            # Walk-forward score
            profitable_folds = sum(1 for r in wf_results if r['total_r'] > 0)
            
            # Top/bottom symbols
            sym_performance = []
            for sym, trades in symbol_stats.items():
                if trades:
                    sym_r = sum(t['exit_r'] for t in trades)
                    sym_performance.append((sym, sym_r, len(trades)))
            sym_performance.sort(key=lambda x: x[1], reverse=True)
            
            results[f"{tp_r}R"] = {
                'total_r': round(total_r, 1),
                'trades': len(r_values),
                'wr': round(wr, 1),
                'avg_r': round(avg_r, 3),
                'wf_score': f"{profitable_folds}/{WF_SPLITS}",
                'mc_p5': mc['p5'],
                'mc_p50': mc['p50'],
                'mc_p95': mc['p95'],
                'mc_prob_profit': mc['prob_profit'],
                'max_dd': dd['max_dd'],
                'max_win_streak': streaks['max_win_streak'],
                'max_loss_streak': streaks['max_loss_streak'],
                'top_5': sym_performance[:5],
                'bottom_5': sym_performance[-5:]
            }
            
            print(f"  Total R: {total_r:+.1f} | WR: {wr:.1f}% | WF: {profitable_folds}/{WF_SPLITS}")
            print(f"  Monte Carlo P5/P50/P95: {mc['p5']}/{mc['p50']}/{mc['p95']} ({mc['prob_profit']}% profit)")
            print(f"  Max DD: {dd['max_dd']}R | Streaks: {streaks['max_win_streak']}W/{streaks['max_loss_streak']}L")
    
    return results

def main():
    print("=" * 70)
    print("DEEP VERIFICATION: 5M vs 30M TIMEFRAMES")
    print("=" * 70)
    print(f"Testing R:R ratios: {RR_RATIOS}")
    print(f"Data: {DATA_DAYS} days | Walk-Forward: {WF_SPLITS} splits")
    print(f"Monte Carlo: {MONTE_CARLO_RUNS} simulations")
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
    
    all_results = {}
    
    for tf in TIMEFRAMES:
        all_results[f"{tf}M"] = run_deep_test(symbols, tf)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: 5M vs 30M")
    print("=" * 70)
    
    for tp_r in RR_RATIOS:
        print(f"\n{'='*50}")
        print(f"{tp_r}R TAKE PROFIT COMPARISON")
        print(f"{'='*50}")
        
        for tf in TIMEFRAMES:
            r = all_results[f"{tf}M"].get(f"{tp_r}R", {})
            if r:
                print(f"\n📊 {tf}M TIMEFRAME:")
                print(f"   Total R: {r['total_r']:+.1f} | WR: {r['wr']}% | Trades: {r['trades']}")
                print(f"   Walk-Forward: {r['wf_score']} | Max DD: {r['max_dd']}R")
                print(f"   Monte Carlo: P5={r['mc_p5']} P50={r['mc_p50']} P95={r['mc_p95']}")
                print(f"   Prob of Profit: {r['mc_prob_profit']}%")
                print(f"   Streaks: {r['max_win_streak']}W / {r['max_loss_streak']}L")
                print(f"   Top 3 Symbols: {[s[0] for s in r['top_5'][:3]]}")
    
    # Save detailed results
    summary_data = []
    for tf in TIMEFRAMES:
        for tp_r in RR_RATIOS:
            r = all_results[f"{tf}M"].get(f"{tp_r}R", {})
            if r:
                summary_data.append({
                    'timeframe': f"{tf}M",
                    'tp_r': tp_r,
                    'total_r': r['total_r'],
                    'trades': r['trades'],
                    'wr': r['wr'],
                    'avg_r': r['avg_r'],
                    'wf_score': r['wf_score'],
                    'mc_p5': r['mc_p5'],
                    'mc_p50': r['mc_p50'],
                    'mc_prob_profit': r['mc_prob_profit'],
                    'max_dd': r['max_dd']
                })
    
    pd.DataFrame(summary_data).to_csv('deep_verification_results.csv', index=False)
    print(f"\n📁 Results saved to deep_verification_results.csv")
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("🎯 FINAL RECOMMENDATION")
    print("=" * 70)
    
    # Find best overall
    best = max(summary_data, key=lambda x: (x['total_r'], x['mc_prob_profit']))
    print(f"Best Config: {best['timeframe']} + {best['tp_r']}R TP")
    print(f"Expected: {best['total_r']:+.1f}R over {best['trades']} trades")
    print(f"Monte Carlo P50: {best['mc_p50']}R ({best['mc_prob_profit']}% profit probability)")
    print(f"Walk-Forward: {best['wf_score']} profitable")

if __name__ == "__main__":
    main()
