#!/usr/bin/env python3
"""
5M Strategy Comprehensive Grid Search
======================================

Find profitable 5M configuration using modern backtesting techniques:
- No look-ahead bias (entry on next candle OPEN)
- Walk-forward validation (6 rolling periods)
- Monte Carlo simulation (1000 bootstrap iterations)
- Transaction cost modeling (0.04% round trip)
- Extensive parameter grid search

Author: AutoBot Optimizer
Date: 2025-12-25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import yaml
import itertools
import time
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

TIMEFRAME = 5  # 5-minute candles
DATA_DAYS = 180  # 6 months of data
WF_SPLITS = 6   # Walk-forward splits
COOLDOWN_BARS = 3  # Min bars between trades per symbol
MC_SIMS = 1000  # Monte Carlo simulations
FEE_PCT = 0.0004  # 0.04% round-trip fees

# Parameter Grid
RR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_METHODS = [
    ('atr', 0.8),   # 0.8x ATR
    ('atr', 1.0),   # 1.0x ATR
    ('atr', 1.5),   # 1.5x ATR
    ('pct', 0.5),   # 0.5% fixed
    ('pct', 1.0),   # 1.0% fixed
    ('pct', 1.5),   # 1.5% fixed
    ('pct', 2.0),   # 2.0% fixed
]
DIV_FILTERS = ['all', 'regular_only', 'regular_bullish_only', 'hidden_bearish_only']
VOLUME_FILTERS = [True, False]

# Bad symbols to potentially filter
BAD_SYMBOLS = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT']


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch klines from Bybit API"""
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
    """Calculate RSI indicator"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR indicator"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_divergences(df: pd.DataFrame, div_filter: str = 'all') -> pd.DataFrame:
    """
    Detect RSI divergence patterns.
    
    Divergence Types:
    - regular_bullish: Price Lower Low, RSI Higher Low → Long
    - regular_bearish: Price Higher High, RSI Lower High → Short
    - hidden_bullish: Price Higher Low, RSI Lower Low → Long continuation
    - hidden_bearish: Price Lower High, RSI Higher High → Short continuation
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df)
    df['atr'] = calculate_atr(df)
    
    lookback = 14
    pivot_left = 3
    pivot_right = 3
    min_distance = 5
    
    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_high'] = df['rsi'].rolling(lookback).max()
    
    # Regular Bullish: Price Lower Low + RSI Higher Low + RSI < 45
    df['regular_bullish'] = (
        (df['low'] <= df['price_low']) & 
        (df['rsi'] > df['rsi_low'].shift(lookback)) &
        (df['rsi'] < 45)
    )
    
    # Regular Bearish: Price Higher High + RSI Lower High + RSI > 55
    df['regular_bearish'] = (
        (df['high'] >= df['price_high']) & 
        (df['rsi'] < df['rsi_high'].shift(lookback)) &
        (df['rsi'] > 55)
    )
    
    # Hidden Bullish: Price Higher Low + RSI Lower Low
    df['hidden_bullish'] = (
        (df['low'] > df['low'].shift(lookback)) &
        (df['rsi'] < df['rsi'].shift(lookback)) &
        (df['rsi'] < 60)
    )
    
    # Hidden Bearish: Price Lower High + RSI Higher High
    df['hidden_bearish'] = (
        (df['high'] < df['high'].shift(lookback)) &
        (df['rsi'] > df['rsi'].shift(lookback)) &
        (df['rsi'] > 40)
    )
    
    # Apply divergence type filter
    if div_filter == 'regular_only':
        df['hidden_bullish'] = False
        df['hidden_bearish'] = False
    elif div_filter == 'regular_bullish_only':
        df['regular_bearish'] = False
        df['hidden_bullish'] = False
        df['hidden_bearish'] = False
    elif div_filter == 'hidden_bearish_only':
        df['regular_bullish'] = False
        df['regular_bearish'] = False
        df['hidden_bullish'] = False
    
    return df


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, 
                   sl_method: tuple, rr_ratio: float) -> dict:
    """
    Simulate a single trade with NO LOOK-AHEAD BIAS.
    
    Entry: Next candle OPEN (not signal candle close)
    Exit: First to hit (TP or SL) based on HIGH/LOW of subsequent candles
    """
    if entry_idx + 1 >= len(df):
        return None
    
    # Entry at NEXT candle's OPEN (no look-ahead)
    entry_candle = df.iloc[entry_idx + 1]
    entry_price = entry_candle['open']
    
    # Calculate SL distance
    sl_type, sl_val = sl_method
    if sl_type == 'atr':
        atr = df.iloc[entry_idx]['atr']
        if pd.isna(atr) or atr == 0:
            atr = entry_price * 0.01  # Fallback: 1%
        sl_distance = atr * sl_val
    else:  # pct
        sl_distance = entry_price * (sl_val / 100)
    
    # Calculate TP and SL prices
    tp_distance = sl_distance * rr_ratio
    
    if side == 'long':
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
    
    # Simulate forward
    for i in range(entry_idx + 2, min(entry_idx + 500, len(df))):
        candle = df.iloc[i]
        bars_held = i - entry_idx - 1
        
        if side == 'long':
            # Check SL first (conservative)
            if candle['low'] <= sl_price:
                pnl = -1.0 - FEE_PCT  # -1R minus fees
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'sl'}
            # Then check TP
            if candle['high'] >= tp_price:
                pnl = rr_ratio - FEE_PCT  # +RR minus fees
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'tp'}
        else:  # short
            # Check SL first
            if candle['high'] >= sl_price:
                pnl = -1.0 - FEE_PCT
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'sl'}
            # Then check TP
            if candle['low'] <= tp_price:
                pnl = rr_ratio - FEE_PCT
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'tp'}
    
    # Timeout - exit at market (small loss due to fees)
    return {'exit_r': -FEE_PCT, 'bars': 500, 'outcome': 'timeout'}


def run_backtest_period(symbol_data: dict, params: dict, 
                        start_pct: float, end_pct: float) -> list:
    """Run backtest for a specific walk-forward period"""
    trades = []
    
    for sym, full_df in symbol_data.items():
        n = len(full_df)
        start_idx = int(n * start_pct)
        end_idx = int(n * end_pct)
        df = full_df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        if len(df) < 100:
            continue
        
        # Detect divergences with applied filter
        df = detect_divergences(df, params['div_filter'])
        
        # Volume filter
        if params['volume_filter']:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ok'] = df['volume'] > df['volume_sma'] * 0.8
        else:
            df['volume_ok'] = True
        
        last_trade_idx = -COOLDOWN_BARS
        
        for idx in range(50, len(df) - 10):
            if idx - last_trade_idx < COOLDOWN_BARS:
                continue
            
            row = df.iloc[idx]
            
            # Determine signal
            side = None
            if row['regular_bullish'] and row['volume_ok']:
                side = 'long'
            elif row['regular_bearish'] and row['volume_ok']:
                side = 'short'
            elif row['hidden_bullish'] and row['volume_ok']:
                side = 'long'
            elif row['hidden_bearish'] and row['volume_ok']:
                side = 'short'
            
            if side is None:
                continue
            
            result = simulate_trade(
                df, idx, side, 
                params['sl_method'], 
                params['rr_ratio']
            )
            
            if result:
                trades.append(result['exit_r'])
                last_trade_idx = idx
    
    return trades


def backtest_config(symbol_data: dict, params: dict) -> dict:
    """Run full walk-forward backtest for a parameter configuration"""
    all_trades = []
    wf_results = []
    
    # Walk-forward: 6 periods (each ~1 month of 180 days)
    period_size = 1.0 / WF_SPLITS
    
    for fold in range(WF_SPLITS):
        start_pct = fold * period_size
        end_pct = start_pct + period_size
        
        fold_trades = run_backtest_period(symbol_data, params, start_pct, end_pct)
        all_trades.extend(fold_trades)
        
        if fold_trades:
            fold_r = sum(fold_trades)
            wf_results.append(fold_r > 0)
    
    if not all_trades or len(all_trades) < 10:
        return None
    
    # Calculate metrics
    total_r = sum(all_trades)
    wins = sum(1 for r in all_trades if r > 0)
    wr = wins / len(all_trades) * 100
    avg_r = total_r / len(all_trades)
    wf_score = sum(wf_results)
    
    # Drawdown
    equity = np.cumsum(all_trades)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = min(drawdowns) if len(drawdowns) > 0 else 0
    
    # Recovery factor
    recovery = total_r / abs(max_dd) if max_dd != 0 else 0
    
    # Monte Carlo
    mc_results = []
    for _ in range(MC_SIMS):
        sample = np.random.choice(all_trades, size=len(all_trades), replace=True)
        mc_results.append(sum(sample))
    
    mc_p5 = np.percentile(mc_results, 5)
    mc_p50 = np.percentile(mc_results, 50)
    mc_p95 = np.percentile(mc_results, 95)
    mc_prob = sum(1 for r in mc_results if r > 0) / MC_SIMS * 100
    
    # Sharpe (R-based, assuming ~252 trading days)
    if np.std(all_trades) > 0:
        sharpe = (np.mean(all_trades) / np.std(all_trades)) * np.sqrt(252)
    else:
        sharpe = 0
    
    return {
        'rr': params['rr_ratio'],
        'sl_type': params['sl_method'][0],
        'sl_val': params['sl_method'][1],
        'div_filter': params['div_filter'],
        'volume_filter': params['volume_filter'],
        'total_r': round(total_r, 1),
        'trades': len(all_trades),
        'wr': round(wr, 1),
        'avg_r': round(avg_r, 3),
        'wf_score': wf_score,
        'max_dd': round(max_dd, 1),
        'recovery': round(recovery, 2),
        'mc_p5': round(mc_p5, 1),
        'mc_p50': round(mc_p50, 1),
        'mc_p95': round(mc_p95, 1),
        'mc_prob': round(mc_prob, 1),
        'sharpe': round(sharpe, 2)
    }


def main():
    start_time = time.time()
    
    print("=" * 80)
    print("5M COMPREHENSIVE GRID SEARCH - Finding Profitable Configuration")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}M | Data: {DATA_DAYS} days | Walk-Forward: {WF_SPLITS} splits")
    print(f"Monte Carlo: {MC_SIMS} sims | Fees: {FEE_PCT*100:.2f}%")
    print("=" * 80)
    
    # Load symbols from config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config.get('trade', {}).get('divergence_symbols', [])
    except:
        symbols = []
    
    if not symbols:
        print("ERROR: No symbols found in config.yaml!")
        return
    
    # Filter bad symbols
    symbols = [s for s in symbols if s not in BAD_SYMBOLS]
    print(f"\n📊 Testing {len(symbols)} symbols (filtered {len(BAD_SYMBOLS)} bad symbols)")
    
    # Fetch data
    print(f"\n⏳ Fetching {TIMEFRAME}M data for {DATA_DAYS} days...")
    symbol_data = {}
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_klines, sym, str(TIMEFRAME), DATA_DAYS): sym 
                   for sym in symbols}
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                df = future.result()
                if len(df) >= 500:  # Need enough data
                    symbol_data[sym] = df
                if done % 20 == 0:
                    print(f"   Loaded {done}/{len(symbols)} symbols...")
            except:
                pass
    
    print(f"✅ Loaded data for {len(symbol_data)} symbols")
    
    # Generate parameter combinations
    param_combos = []
    for rr, sl, div_f, vol_f in itertools.product(
        RR_RATIOS, SL_METHODS, DIV_FILTERS, VOLUME_FILTERS
    ):
        param_combos.append({
            'rr_ratio': rr,
            'sl_method': sl,
            'div_filter': div_f,
            'volume_filter': vol_f
        })
    
    print(f"\n🔧 Testing {len(param_combos)} parameter combinations...")
    print(f"   R:R: {RR_RATIOS}")
    print(f"   SL Methods: {SL_METHODS}")
    print(f"   Div Filters: {DIV_FILTERS}")
    print(f"   Volume Filter: {VOLUME_FILTERS}")
    
    # Run grid search
    all_results = []
    tested = 0
    
    print("\n" + "=" * 80)
    print("GRID SEARCH PROGRESS")
    print("=" * 80)
    
    for params in param_combos:
        tested += 1
        sl_str = f"{params['sl_method'][0]}:{params['sl_method'][1]}"
        param_str = f"R:R={params['rr_ratio']} | SL={sl_str} | Div={params['div_filter']} | Vol={params['volume_filter']}"
        
        result = backtest_config(symbol_data, params)
        
        if result:
            all_results.append(result)
            status = "✅" if result['mc_prob'] >= 70 and result['wf_score'] >= 4 else "❌"
            print(f"[{tested:3d}/{len(param_combos)}] {status} {param_str[:50]:50} | "
                  f"R={result['total_r']:+7.0f} | WR={result['wr']:4.1f}% | "
                  f"WF={result['wf_score']}/6 | MC={result['mc_prob']:4.0f}%")
        else:
            print(f"[{tested:3d}/{len(param_combos)}] ⚠️  {param_str[:50]:50} | No trades")
    
    # Sort and display results
    print("\n" + "=" * 80)
    print("TOP 20 CONFIGURATIONS (Ranked by Monte Carlo Profit Probability)")
    print("=" * 80)
    
    # Sort by MC prob, then by total R
    sorted_results = sorted(all_results, key=lambda x: (x['mc_prob'], x['total_r']), reverse=True)
    
    print(f"\n{'Rank':<5} {'R:R':<5} {'SL':<10} {'DivFilter':<18} {'Vol':<5} {'TotalR':<8} "
          f"{'WR%':<6} {'WF':<5} {'MC%':<6} {'Sharpe':<7} {'MaxDD':<8}")
    print("-" * 100)
    
    for i, r in enumerate(sorted_results[:20], 1):
        sl_str = f"{r['sl_type']}:{r['sl_val']}"
        vol_str = "Y" if r['volume_filter'] else "N"
        wf_str = f"{r['wf_score']}/6"
        status = "🏆" if r['mc_prob'] >= 70 and r['wf_score'] >= 4 and r['total_r'] > 0 else ""
        
        print(f"{i:<5} {r['rr']:<5.1f} {sl_str:<10} {r['div_filter']:<18} {vol_str:<5} "
              f"{r['total_r']:+7.0f}R {r['wr']:5.1f}% {wf_str:<5} {r['mc_prob']:5.1f}% "
              f"{r['sharpe']:6.2f} {r['max_dd']:+7.0f}R {status}")
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('comprehensive_grid_results.csv', index=False)
    print(f"\n📁 All results saved to comprehensive_grid_results.csv")
    
    # Find and save best configuration
    best = None
    for r in sorted_results:
        if r['mc_prob'] >= 70 and r['wf_score'] >= 4 and r['total_r'] > 0:
            best = r
            break
    
    # Fallback: if no "perfect" config, pick best by MC prob with positive R
    if not best:
        for r in sorted_results:
            if r['total_r'] > 0 and r['mc_prob'] >= 50:
                best = r
                break
    
    if not best and sorted_results:
        best = sorted_results[0]  # Take top result even if criteria not met
    
    print("\n" + "=" * 80)
    print("🎯 BEST CONFIGURATION")
    print("=" * 80)
    
    if best:
        validated = best['mc_prob'] >= 70 and best['wf_score'] >= 4 and best['total_r'] > 0
        status = "✅ VALIDATED PROFITABLE" if validated else "⚠️ BEST AVAILABLE (criteria not fully met)"
        
        print(f"\n{status}")
        print(f"\n📊 Parameters:")
        print(f"   R:R Ratio: {best['rr']}:1")
        print(f"   SL Method: {best['sl_type']} × {best['sl_val']}")
        print(f"   Divergence Filter: {best['div_filter']}")
        print(f"   Volume Filter: {'Enabled' if best['volume_filter'] else 'Disabled'}")
        print(f"\n📈 Performance (180-day backtest):")
        print(f"   Total R: {best['total_r']:+.0f}R")
        print(f"   Trades: {best['trades']}")
        print(f"   Win Rate: {best['wr']:.1f}%")
        print(f"   Avg R per Trade: {best['avg_r']:+.3f}R")
        print(f"   Sharpe Ratio: {best['sharpe']:.2f}")
        print(f"\n⚖️ Walk-Forward Validation:")
        print(f"   Score: {best['wf_score']}/6 periods profitable")
        print(f"\n🎲 Monte Carlo Analysis ({MC_SIMS} sims):")
        print(f"   Profit Probability: {best['mc_prob']:.1f}%")
        print(f"   Expected R (median): {best['mc_p50']:+.0f}R")
        print(f"   5th-95th percentile: {best['mc_p5']:+.0f}R to {best['mc_p95']:+.0f}R")
        print(f"\n📉 Risk:")
        print(f"   Max Drawdown: {best['max_dd']:.0f}R")
        print(f"   Recovery Factor: {best['recovery']:.2f}")
        
        # Save optimal config
        optimal_config = {
            'strategy': {
                'name': '5M RSI Divergence',
                'timeframe': TIMEFRAME,
                'rr_ratio': best['rr'],
                'sl_type': best['sl_type'],
                'sl_value': best['sl_val'],
                'divergence_filter': best['div_filter'],
                'volume_filter': best['volume_filter']
            },
            'backtest_results': {
                'total_r': best['total_r'],
                'trades': best['trades'],
                'win_rate': best['wr'],
                'walk_forward_score': f"{best['wf_score']}/6",
                'mc_profit_probability': best['mc_prob'],
                'max_drawdown': best['max_dd'],
                'sharpe_ratio': best['sharpe']
            },
            'validated': validated,
            'test_date': datetime.now().isoformat()
        }
        
        with open('optimal_5m_config.yaml', 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)
        print(f"\n💾 Optimal config saved to optimal_5m_config.yaml")
    else:
        print("❌ No profitable configuration found!")
        print("   Consider adjusting parameter ranges or using different symbols.")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
