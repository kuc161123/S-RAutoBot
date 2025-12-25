#!/usr/bin/env python3
"""
Extended 5M Validation Backtest
================================

Further validation of 5M Fixed 5R TP strategy:
- 120 days of data (vs 90 days before)
- Monthly performance breakdown
- Drawdown analysis
- Symbol-by-symbol performance
- Win/loss streak analysis
- Best/worst days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import yaml
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION - 5M Fixed 5R TP
# ============================================

TIMEFRAME = 5
SL_PCT = 1.2
RR_RATIO = 5.0
COOLDOWN_BARS = 3
DATA_DAYS = 120  # Extended from 90
WF_SPLITS = 6    # More splits for robustness

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

def simulate_trade(df: pd.DataFrame, idx: int, side: str, entry: float, sl_distance: float) -> dict:
    tp_price = entry + (RR_RATIO * sl_distance) if side == 'long' else entry - (RR_RATIO * sl_distance)
    sl_price = entry - sl_distance if side == 'long' else entry + sl_distance
    
    for i in range(idx + 1, min(idx + 1000, len(df))):
        candle = df.iloc[i]
        bars_held = i - idx
        
        if side == 'long':
            if candle['low'] <= sl_price:
                return {'exit_r': -1.0, 'exit_reason': 'SL', 'bars': bars_held, 'exit_time': candle['timestamp']}
            if candle['high'] >= tp_price:
                return {'exit_r': RR_RATIO, 'exit_reason': 'TP', 'bars': bars_held, 'exit_time': candle['timestamp']}
        else:
            if candle['high'] >= sl_price:
                return {'exit_r': -1.0, 'exit_reason': 'SL', 'bars': bars_held, 'exit_time': candle['timestamp']}
            if candle['low'] <= tp_price:
                return {'exit_r': RR_RATIO, 'exit_reason': 'TP', 'bars': bars_held, 'exit_time': candle['timestamp']}
    
    return {'exit_r': 0, 'exit_reason': 'TIMEOUT', 'bars': 1000, 'exit_time': df.iloc[-1]['timestamp']}

def backtest_symbol(symbol: str, df: pd.DataFrame) -> list:
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
            
            result = simulate_trade(df, idx, side, entry, sl_distance)
            
            trades.append({
                'symbol': symbol,
                'side': side,
                'entry_time': row['timestamp'],
                'exit_time': result['exit_time'],
                'exit_r': result['exit_r'],
                'exit_reason': result['exit_reason'],
                'bars_held': result['bars']
            })
            
            last_trade_idx = idx
            
    except Exception as e:
        pass
    
    return trades

def analyze_monthly(trades_df: pd.DataFrame) -> dict:
    """Analyze performance by month"""
    if trades_df.empty:
        return {}
    
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly = {}
    
    for month, group in trades_df.groupby('month'):
        total_r = group['exit_r'].sum()
        wins = (group['exit_r'] > 0).sum()
        wr = wins / len(group) * 100 if len(group) > 0 else 0
        monthly[str(month)] = {'total_r': round(total_r, 1), 'trades': len(group), 'wr': round(wr, 1)}
    
    return monthly

def analyze_drawdown(r_values: list) -> dict:
    """Calculate drawdown metrics"""
    if not r_values:
        return {}
    
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    
    max_dd = min(drawdowns)
    
    # Find max drawdown duration
    in_dd = drawdowns < 0
    dd_periods = []
    current_dd_len = 0
    
    for is_dd in in_dd:
        if is_dd:
            current_dd_len += 1
        else:
            if current_dd_len > 0:
                dd_periods.append(current_dd_len)
            current_dd_len = 0
    
    if current_dd_len > 0:
        dd_periods.append(current_dd_len)
    
    max_dd_duration = max(dd_periods) if dd_periods else 0
    
    return {
        'max_dd': round(max_dd, 1),
        'max_dd_duration': max_dd_duration,
        'recovery_factor': round(sum(r_values) / abs(max_dd), 2) if max_dd != 0 else float('inf')
    }

def analyze_streaks(r_values: list) -> dict:
    """Analyze win/loss streaks"""
    if not r_values:
        return {}
    
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

def analyze_symbols(trades_df: pd.DataFrame) -> tuple:
    """Get top and bottom performing symbols"""
    if trades_df.empty:
        return [], []
    
    sym_perf = trades_df.groupby('symbol').agg({
        'exit_r': ['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(1)
    sym_perf.columns = ['total_r', 'trades', 'wr']
    sym_perf = sym_perf.sort_values('total_r', ascending=False)
    
    top5 = sym_perf.head(5).reset_index().to_dict('records')
    bottom5 = sym_perf.tail(5).reset_index().to_dict('records')
    
    return top5, bottom5

def monte_carlo(r_values: list, n_sims: int = 1000) -> dict:
    """Run Monte Carlo simulation"""
    if not r_values:
        return {}
    
    n_trades = len(r_values)
    final_rs = []
    
    for _ in range(n_sims):
        sampled = np.random.choice(r_values, size=n_trades, replace=True)
        final_rs.append(np.sum(sampled))
    
    return {
        'p5': round(np.percentile(final_rs, 5), 1),
        'p25': round(np.percentile(final_rs, 25), 1),
        'p50': round(np.percentile(final_rs, 50), 1),
        'p75': round(np.percentile(final_rs, 75), 1),
        'p95': round(np.percentile(final_rs, 95), 1),
        'prob_profit': round(sum(1 for r in final_rs if r > 0) / n_sims * 100, 1)
    }

def main():
    print("=" * 70)
    print("EXTENDED 5M VALIDATION BACKTEST")
    print("=" * 70)
    print(f"Strategy: Fixed {RR_RATIO}R TP | SL: {SL_PCT}% | Cooldown: {COOLDOWN_BARS} bars")
    print(f"Data: {DATA_DAYS} days | Walk-Forward: {WF_SPLITS} splits")
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
    
    # Fetch data
    print(f"\nFetching {TIMEFRAME}M data for {len(symbols)} symbols ({DATA_DAYS} days)...")
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
    
    # Run backtest with walk-forward
    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    
    all_trades = []
    wf_results = []
    
    for fold in range(WF_SPLITS):
        fold_trades = []
        
        for sym, df in symbol_data.items():
            fold_size = len(df) // WF_SPLITS
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < WF_SPLITS - 1 else len(df)
            
            fold_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            trades = backtest_symbol(sym, fold_df)
            fold_trades.extend(trades)
        
        all_trades.extend(fold_trades)
        
        if fold_trades:
            trades_df = pd.DataFrame(fold_trades)
            total_r = trades_df['exit_r'].sum()
            wins = (trades_df['exit_r'] > 0).sum()
            wr = wins / len(trades_df) * 100
            wf_results.append({'fold': fold + 1, 'total_r': total_r, 'trades': len(trades_df), 'wr': wr})
            print(f"  Fold {fold+1}: {len(fold_trades)} trades, {total_r:+.1f}R, {wr:.1f}% WR")
    
    # Overall results
    if not all_trades:
        print("ERROR: No trades generated!")
        return
    
    all_df = pd.DataFrame(all_trades)
    r_values = all_df['exit_r'].tolist()
    
    total_r = sum(r_values)
    wins = sum(1 for r in r_values if r > 0)
    wr = wins / len(r_values) * 100
    avg_r = np.mean(r_values)
    profitable_folds = sum(1 for r in wf_results if r['total_r'] > 0)
    
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Total R: {total_r:+.1f}")
    print(f"Trades: {len(r_values)} ({len(r_values)/DATA_DAYS:.1f}/day)")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Avg R: {avg_r:.4f}")
    print(f"Walk-Forward: {profitable_folds}/{WF_SPLITS} profitable")
    
    # Monthly breakdown
    print(f"\n{'='*70}")
    print("MONTHLY BREAKDOWN")
    print(f"{'='*70}")
    monthly = analyze_monthly(all_df)
    for month, stats in monthly.items():
        print(f"  {month}: {stats['total_r']:+.1f}R | {stats['trades']} trades | {stats['wr']:.1f}% WR")
    
    # Drawdown analysis
    print(f"\n{'='*70}")
    print("DRAWDOWN ANALYSIS")
    print(f"{'='*70}")
    dd = analyze_drawdown(r_values)
    print(f"Max Drawdown: {dd['max_dd']}R")
    print(f"Max DD Duration: {dd['max_dd_duration']} trades")
    print(f"Recovery Factor: {dd['recovery_factor']}x")
    
    # Streak analysis
    print(f"\n{'='*70}")
    print("STREAK ANALYSIS")
    print(f"{'='*70}")
    streaks = analyze_streaks(r_values)
    print(f"Max Win Streak: {streaks['max_win_streak']}")
    print(f"Max Loss Streak: {streaks['max_loss_streak']}")
    
    # Symbol performance
    print(f"\n{'='*70}")
    print("SYMBOL PERFORMANCE")
    print(f"{'='*70}")
    top5, bottom5 = analyze_symbols(all_df)
    print("Top 5 Symbols:")
    for s in top5:
        print(f"  {s['symbol']}: {s['total_r']:+.1f}R | {s['trades']} trades | {s['wr']:.1f}% WR")
    print("Bottom 5 Symbols:")
    for s in bottom5:
        print(f"  {s['symbol']}: {s['total_r']:+.1f}R | {s['trades']} trades | {s['wr']:.1f}% WR")
    
    # Monte Carlo
    print(f"\n{'='*70}")
    print("MONTE CARLO SIMULATION (1000 runs)")
    print(f"{'='*70}")
    mc = monte_carlo(r_values)
    print(f"P5: {mc['p5']}R | P25: {mc['p25']}R | P50: {mc['p50']}R | P75: {mc['p75']}R | P95: {mc['p95']}R")
    print(f"Probability of Profit: {mc['prob_profit']}%")
    
    # Save results
    results_summary = {
        'total_r': total_r,
        'trades': len(r_values),
        'wr': wr,
        'avg_r': avg_r,
        'wf_score': f"{profitable_folds}/{WF_SPLITS}",
        'max_dd': dd['max_dd'],
        'recovery_factor': dd['recovery_factor'],
        'mc_p50': mc['p50'],
        'mc_prob_profit': mc['prob_profit']
    }
    
    pd.DataFrame([results_summary]).to_csv('extended_5m_validation.csv', index=False)
    all_df.to_csv('extended_5m_trades.csv', index=False)
    print(f"\n📁 Results saved to extended_5m_validation.csv and extended_5m_trades.csv")
    
    # Final verdict
    print(f"\n{'='*70}")
    print("🎯 VALIDATION VERDICT")
    print(f"{'='*70}")
    
    if mc['prob_profit'] >= 95 and profitable_folds >= WF_SPLITS - 1:
        print("✅ HIGHLY VALIDATED - Strategy is robust!")
    elif mc['prob_profit'] >= 80 and profitable_folds >= WF_SPLITS // 2 + 1:
        print("✅ VALIDATED - Strategy shows consistent profitability")
    elif mc['prob_profit'] >= 60:
        print("⚠️ PARTIALLY VALIDATED - Some risk, monitor closely")
    else:
        print("❌ NOT VALIDATED - Strategy may need adjustment")
    
    print(f"\nRecommendation: {'PROCEED with 5M Fixed 5R TP' if mc['prob_profit'] >= 80 else 'REVIEW before deployment'}")

if __name__ == "__main__":
    main()
