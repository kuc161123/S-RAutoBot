#!/usr/bin/env python3
"""
5M Strategy Grid Search
========================

Comprehensive search for profitable 5M configuration:
- R:R Ratios: 2, 3, 4
- SL Percentages: 0.8%, 1.0%, 1.5%
- Symbol filtering based on performance
- 120-day data with 6-way walk-forward
- Monte Carlo validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import yaml
import itertools
warnings.filterwarnings('ignore')

# ============================================
# GRID SEARCH PARAMETERS
# ============================================

TIMEFRAME = 5
RR_RATIOS = [2.0, 3.0, 4.0]
SL_PCTS = [0.8, 1.0, 1.5]
COOLDOWN_BARS = 3
DATA_DAYS = 120
WF_SPLITS = 6

# Bad symbols from previous analysis (WR < 10%)
BAD_SYMBOLS = ['XAUTUSDT', 'PAXGUSDT', 'TRXUSDT', 'ATOMUSDT', 'USTCUSDT']

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
                   sl_distance: float, rr_ratio: float) -> dict:
    tp_price = entry + (rr_ratio * sl_distance) if side == 'long' else entry - (rr_ratio * sl_distance)
    sl_price = entry - sl_distance if side == 'long' else entry + sl_distance
    
    for i in range(idx + 1, min(idx + 500, len(df))):
        candle = df.iloc[i]
        bars_held = i - idx
        
        if side == 'long':
            if candle['low'] <= sl_price:
                return {'exit_r': -1.0, 'bars': bars_held}
            if candle['high'] >= tp_price:
                return {'exit_r': rr_ratio, 'bars': bars_held}
        else:
            if candle['high'] >= sl_price:
                return {'exit_r': -1.0, 'bars': bars_held}
            if candle['low'] <= tp_price:
                return {'exit_r': rr_ratio, 'bars': bars_held}
    
    return {'exit_r': 0, 'bars': 500}

def backtest_config(symbol_data: dict, rr_ratio: float, sl_pct: float, 
                    filter_bad: bool = False) -> dict:
    """Backtest a specific configuration"""
    all_trades = []
    wf_results = []
    
    symbols_to_test = {k: v for k, v in symbol_data.items() 
                       if not filter_bad or k not in BAD_SYMBOLS}
    
    for fold in range(WF_SPLITS):
        fold_trades = []
        
        for sym, df in symbols_to_test.items():
            if len(df) < 100:
                continue
                
            fold_size = len(df) // WF_SPLITS
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < WF_SPLITS - 1 else len(df)
            
            fold_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            if len(fold_df) < 50:
                continue
            
            # Detect divergences
            fold_df = detect_divergences(fold_df)
            fold_df['volume_sma'] = fold_df['volume'].rolling(20).mean()
            fold_df['volume_ok'] = fold_df['volume'] > fold_df['volume_sma']
            
            last_trade_idx = -COOLDOWN_BARS
            
            for idx in range(50, len(fold_df) - 10):
                if idx - last_trade_idx < COOLDOWN_BARS:
                    continue
                
                row = fold_df.iloc[idx]
                
                side = None
                if row['bullish_div'] and row['volume_ok']:
                    side = 'long'
                elif row['bearish_div'] and row['volume_ok']:
                    side = 'short'
                
                if side is None:
                    continue
                
                entry = row['close']
                sl_distance = entry * (sl_pct / 100)
                
                result = simulate_trade(fold_df, idx, side, entry, sl_distance, rr_ratio)
                fold_trades.append(result['exit_r'])
                last_trade_idx = idx
        
        all_trades.extend(fold_trades)
        
        if fold_trades:
            total_r = sum(fold_trades)
            wf_results.append(total_r > 0)
    
    if not all_trades:
        return None
    
    total_r = sum(all_trades)
    wins = sum(1 for r in all_trades if r > 0)
    wr = wins / len(all_trades) * 100
    wf_score = sum(wf_results)
    
    # Monte Carlo
    n_sims = 500
    final_rs = [sum(np.random.choice(all_trades, size=len(all_trades), replace=True)) 
                for _ in range(n_sims)]
    mc_p50 = np.percentile(final_rs, 50)
    mc_prob = sum(1 for r in final_rs if r > 0) / n_sims * 100
    
    # Drawdown
    equity = np.cumsum(all_trades)
    max_dd = min(equity - np.maximum.accumulate(equity))
    
    return {
        'rr': rr_ratio,
        'sl': sl_pct,
        'filter_bad': filter_bad,
        'total_r': round(total_r, 1),
        'trades': len(all_trades),
        'wr': round(wr, 1),
        'wf_score': f"{wf_score}/{WF_SPLITS}",
        'wf_num': wf_score,
        'mc_p50': round(mc_p50, 1),
        'mc_prob': round(mc_prob, 1),
        'max_dd': round(max_dd, 1)
    }

def main():
    print("=" * 70)
    print("5M STRATEGY GRID SEARCH")
    print("=" * 70)
    print(f"Testing: R:R {RR_RATIOS} × SL {SL_PCTS} × Filter [Yes/No]")
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
    print(f"Bad symbols to filter: {BAD_SYMBOLS}")
    
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
    
    # Grid search
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}")
    
    all_results = []
    
    for rr, sl, filter_bad in itertools.product(RR_RATIOS, SL_PCTS, [False, True]):
        filter_str = "Filtered" if filter_bad else "All Syms"
        print(f"Testing R:R={rr}, SL={sl}%, {filter_str}...", end=" ")
        
        result = backtest_config(symbol_data, rr, sl, filter_bad)
        
        if result:
            all_results.append(result)
            status = "✅" if result['mc_prob'] >= 80 and result['wf_num'] >= 4 else "❌"
            print(f"{status} {result['total_r']:+.0f}R, WR={result['wr']:.1f}%, WF={result['wf_score']}, MC={result['mc_prob']:.0f}%")
        else:
            print("No trades")
    
    # Rank results
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS (Ranked by MC Profit Prob)")
    print(f"{'='*70}")
    
    sorted_results = sorted(all_results, key=lambda x: (x['mc_prob'], x['total_r']), reverse=True)
    
    for i, r in enumerate(sorted_results[:10], 1):
        filter_str = "✓" if r['filter_bad'] else "✗"
        status = "🏆" if r['mc_prob'] >= 80 and r['wf_num'] >= 4 else ""
        print(f"{i}. R:R={r['rr']}:1 | SL={r['sl']}% | Filter={filter_str} | "
              f"Total={r['total_r']:+.0f}R | WR={r['wr']}% | WF={r['wf_score']} | "
              f"MC={r['mc_prob']}% | DD={r['max_dd']}R {status}")
    
    # Save results
    pd.DataFrame(all_results).to_csv('grid_search_results.csv', index=False)
    print(f"\n📁 Results saved to grid_search_results.csv")
    
    # Best config
    best = sorted_results[0] if sorted_results else None
    
    print(f"\n{'='*70}")
    print("🎯 BEST CONFIGURATION")
    print(f"{'='*70}")
    
    if best and best['mc_prob'] >= 80 and best['wf_num'] >= 4:
        print(f"✅ VALIDATED PROFITABLE CONFIG FOUND!")
        print(f"   R:R: {best['rr']}:1")
        print(f"   SL: {best['sl']}%")
        print(f"   Filter Bad Symbols: {best['filter_bad']}")
        print(f"   Expected: {best['total_r']:+.0f}R | {best['wr']}% WR")
        print(f"   Walk-Forward: {best['wf_score']} profitable")
        print(f"   Monte Carlo: {best['mc_prob']}% profit prob")
        print(f"   Max Drawdown: {best['max_dd']}R")
        return best
    elif best:
        print(f"⚠️ No config meets validation criteria")
        print(f"   Best attempt: R:R={best['rr']}, SL={best['sl']}%")
        print(f"   MC Prob: {best['mc_prob']}% (need ≥80%)")
        print(f"   WF Score: {best['wf_score']} (need ≥4/{WF_SPLITS})")
        return None
    else:
        print("❌ No profitable configs found")
        return None

if __name__ == "__main__":
    main()
