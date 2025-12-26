#!/usr/bin/env python3
"""
FOCUSED BACKTEST: Top 20 High-Liquidity Coins
=============================================

Modern backtesting with:
- No look-ahead bias (entry on next candle OPEN)
- Walk-forward validation (6 periods)
- Monte Carlo simulation (500 iterations)
- Real-time progress output
- Fee modeling (0.04% round-trip)

Author: AutoBot Optimizer
Date: 2025-12-26
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import yaml
import time
import sys
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

TIMEFRAME = 5  # 5-minute candles
DATA_DAYS = 90  # 3 months of data (faster than 6 months)
WF_SPLITS = 6   # Walk-forward splits
COOLDOWN_BARS = 3  # Min bars between trades per symbol
MC_SIMS = 500  # Monte Carlo simulations (reduced for speed)
FEE_PCT = 0.0004  # 0.04% round-trip fees

# HIGH-LIQUIDITY SYMBOLS ONLY (from config.yaml)
HIGH_LIQUIDITY_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ARBUSDT',
    'OPUSDT', 'ATOMUSDT', 'UNIUSDT', 'INJUSDT', 'TONUSDT'
]

# Parameter Grid (focused - fewer combinations for speed)
RR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]  # 5 options
SL_METHODS = [
    ('atr', 0.8),   # Current optimal
    ('atr', 1.0),   # Standard
    ('atr', 1.5),   # Wider
]  # 3 options
DIV_FILTERS = ['all', 'regular_only']  # 2 options
VOLUME_FILTERS = [False]  # Disabled (validated)

# Total combinations: 5 × 3 × 2 × 1 = 30 (much faster!)

print("=" * 80)
print("FOCUSED BACKTEST: HIGH-LIQUIDITY COINS ONLY")
print("=" * 80)
print(f"Symbols: {len(HIGH_LIQUIDITY_SYMBOLS)} high-liquidity coins")
print(f"Timeframe: {TIMEFRAME}M | Data: {DATA_DAYS} days | Walk-Forward: {WF_SPLITS} splits")
print(f"Combinations: {len(RR_RATIOS) * len(SL_METHODS) * len(DIV_FILTERS) * len(VOLUME_FILTERS)}")
print("=" * 80)

# ============================================
# DATA FETCHING
# ============================================

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch historical klines from Bybit API."""
    from pybit.unified_trading import HTTP
    
    try:
        session = HTTP(testnet=False)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_klines = []
        current_end = end_time
        
        while current_end > start_time:
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=start_time,
                end=current_end,
                limit=1000
            )
            
            if response['retCode'] != 0:
                break
                
            klines = response['result']['list']
            if not klines:
                break
                
            all_klines.extend(klines)
            current_end = int(klines[-1][0]) - 1
            
        if not all_klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"  ⚠️ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI using SMA (matches bot logic)."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using SMA (matches bot logic)."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_divergences(df: pd.DataFrame, div_filter: str = 'all') -> pd.DataFrame:
    """Detect RSI divergence patterns (matches bot logic exactly)."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df)
    df['atr'] = calculate_atr(df)
    
    lookback = 14
    
    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_high'] = df['rsi'].rolling(lookback).max()
    
    # Regular Bullish: Price Lower Low + RSI Higher Low (uses shift for distant window)
    df['regular_bullish'] = (
        (df['low'] <= df['price_low']) & 
        (df['rsi'] > df['rsi_low'].shift(lookback)) &
        (df['rsi'] < 45)
    )
    
    # Regular Bearish: Price Higher High + RSI Lower High
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
    
    # Apply filter
    if div_filter == 'regular_only':
        df['hidden_bullish'] = False
        df['hidden_bearish'] = False
    
    return df


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, 
                   sl_method: tuple, rr_ratio: float) -> dict:
    """Simulate a single trade with NO LOOK-AHEAD BIAS."""
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
    
    # Simulate forward (max 500 bars)
    for i in range(entry_idx + 2, min(entry_idx + 500, len(df))):
        candle = df.iloc[i]
        bars_held = i - entry_idx - 1
        
        if side == 'long':
            # Check SL first (conservative)
            if candle['low'] <= sl_price:
                pnl = -1.0 - FEE_PCT
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'sl'}
            if candle['high'] >= tp_price:
                pnl = rr_ratio - FEE_PCT
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'tp'}
        else:
            if candle['high'] >= sl_price:
                pnl = -1.0 - FEE_PCT
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'sl'}
            if candle['low'] <= tp_price:
                pnl = rr_ratio - FEE_PCT
                return {'exit_r': pnl, 'bars': bars_held, 'outcome': 'tp'}
    
    # Timeout
    return {'exit_r': -FEE_PCT, 'bars': 500, 'outcome': 'timeout'}


def run_backtest_period(symbol_data: dict, params: dict, 
                        start_pct: float, end_pct: float) -> list:
    """Run backtest for a specific walk-forward period."""
    trades = []
    
    for sym, full_df in symbol_data.items():
        n = len(full_df)
        start_idx = int(n * start_pct)
        end_idx = int(n * end_pct)
        df = full_df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        if len(df) < 100:
            continue
        
        df = detect_divergences(df, params['div_filter'])
        df['volume_ok'] = True  # Volume filter disabled
        
        last_trade_idx = -COOLDOWN_BARS
        
        for idx in range(50, len(df) - 10):
            if idx - last_trade_idx < COOLDOWN_BARS:
                continue
            
            row = df.iloc[idx]
            
            side = None
            if row['regular_bullish'] and row['volume_ok']:
                side = 'long'
            elif row['regular_bearish'] and row['volume_ok']:
                side = 'short'
            elif row['hidden_bullish'] and row['volume_ok']:
                side = 'long'
            elif row['hidden_bearish'] and row['volume_ok']:
                side = 'short'
            
            if side:
                result = simulate_trade(df, idx, side, params['sl_method'], params['rr_ratio'])
                if result:
                    trades.append({
                        'symbol': sym,
                        'side': side,
                        **result
                    })
                    last_trade_idx = idx
    
    return trades


def monte_carlo_simulation(trades: list, n_sims: int = 500) -> dict:
    """Run Monte Carlo simulation on trade sequence."""
    if len(trades) < 10:
        return {'p50': 0, 'prob_profit': 0}
    
    returns = [t['exit_r'] for t in trades]
    
    final_equities = []
    for _ in range(n_sims):
        sampled = np.random.choice(returns, size=len(returns), replace=True)
        final_equities.append(np.sum(sampled))
    
    return {
        'p50': np.percentile(final_equities, 50),
        'prob_profit': np.mean([1 for e in final_equities if e > 0]) * 100
    }


def backtest_config(symbol_data: dict, params: dict) -> dict:
    """Run full backtest with walk-forward validation."""
    all_trades = []
    wf_results = []
    
    # Walk-forward splits
    for i in range(WF_SPLITS):
        start_pct = i / WF_SPLITS
        end_pct = (i + 1) / WF_SPLITS
        
        trades = run_backtest_period(symbol_data, params, start_pct, end_pct)
        all_trades.extend(trades)
        
        if trades:
            period_r = sum(t['exit_r'] for t in trades)
            wf_results.append(period_r > 0)
    
    if len(all_trades) < 20:
        return None
    
    # Calculate metrics
    total_r = sum(t['exit_r'] for t in all_trades)
    wins = sum(1 for t in all_trades if t['exit_r'] > 0)
    wr = wins / len(all_trades) * 100
    
    # Walk-forward score
    wf_score = sum(wf_results)
    
    # Monte Carlo
    mc = monte_carlo_simulation(all_trades, MC_SIMS)
    
    # Max drawdown
    equity = [0]
    for t in all_trades:
        equity.append(equity[-1] + t['exit_r'])
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    
    # Sharpe (simplified)
    returns = [t['exit_r'] for t in all_trades]
    sharpe = np.mean(returns) / (np.std(returns) + 0.001) * np.sqrt(252)
    
    return {
        'rr': params['rr_ratio'],
        'sl_type': params['sl_method'][0],
        'sl_val': params['sl_method'][1],
        'div_filter': params['div_filter'],
        'volume_filter': params['volume_filter'],
        'total_r': total_r,
        'trades': len(all_trades),
        'wr': wr,
        'wf_score': wf_score,
        'mc_prob': mc['prob_profit'],
        'mc_p50': mc['p50'],
        'max_dd': max_dd,
        'sharpe': sharpe,
        'avg_r': total_r / len(all_trades) if all_trades else 0
    }


def main():
    start_time = time.time()
    
    # Fetch data for high-liquidity symbols
    print(f"\n⏳ Fetching {TIMEFRAME}M data for {len(HIGH_LIQUIDITY_SYMBOLS)} symbols...")
    symbol_data = {}
    
    for i, sym in enumerate(HIGH_LIQUIDITY_SYMBOLS):
        sys.stdout.write(f"\r   Loading {i+1}/{len(HIGH_LIQUIDITY_SYMBOLS)}: {sym}...          ")
        sys.stdout.flush()
        df = fetch_klines(sym, str(TIMEFRAME), DATA_DAYS)
        if len(df) >= 500:
            symbol_data[sym] = df
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Generate parameter combinations
    import itertools
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
    
    # Run grid search with progress
    all_results = []
    
    print("\n" + "=" * 100)
    print(f"{'#':>3} {'R:R':>4} {'SL':>8} {'Filter':>12} {'Trades':>7} {'TotalR':>8} {'WR%':>6} {'WF':>4} {'MC%':>5}")
    print("=" * 100)
    
    for i, params in enumerate(param_combos):
        sl_str = f"{params['sl_method'][0]}:{params['sl_method'][1]}"
        
        result = backtest_config(symbol_data, params)
        
        if result:
            all_results.append(result)
            status = "✅" if result['mc_prob'] >= 60 and result['wf_score'] >= 4 else "  "
            print(f"{i+1:3} {result['rr']:4.1f} {sl_str:>8} {result['div_filter']:>12} "
                  f"{result['trades']:7} {result['total_r']:+8.0f} {result['wr']:5.1f}% "
                  f"{result['wf_score']}/6 {result['mc_prob']:4.0f}% {status}")
        else:
            print(f"{i+1:3} {params['rr_ratio']:4.1f} {sl_str:>8} {params['div_filter']:>12} "
                  f"{'---':>7} {'---':>8} {'---':>6} {'---':>4} {'---':>5} ⚠️ No trades")
    
    # Sort and display top results
    print("\n" + "=" * 100)
    print("🏆 TOP 10 CONFIGURATIONS (Ranked by Total R)")
    print("=" * 100)
    
    sorted_results = sorted(all_results, key=lambda x: x['total_r'], reverse=True)
    
    for i, r in enumerate(sorted_results[:10], 1):
        sl_str = f"{r['sl_type']}:{r['sl_val']}"
        status = "🏆" if r['mc_prob'] >= 60 and r['wf_score'] >= 4 else ""
        print(f"{i}. R:R={r['rr']:.1f} SL={sl_str} Div={r['div_filter']:<12} | "
              f"R={r['total_r']:+.0f} WR={r['wr']:.1f}% WF={r['wf_score']}/6 MC={r['mc_prob']:.0f}% {status}")
    
    # Save results
    if all_results:
        pd.DataFrame(all_results).to_csv('focused_backtest_results.csv', index=False)
        print(f"\n📁 Results saved to focused_backtest_results.csv")
        
        # Save best config
        if sorted_results:
            best = sorted_results[0]
            best_config = {
                'high_probability_trio': {
                    'enabled': True,
                    'divergence_filter': best['div_filter'],
                    'require_volume': best['volume_filter'],
                    'rr_ratio': best['rr'],
                    'sl_atr_multiplier': best['sl_val'] if best['sl_type'] == 'atr' else 0.8,
                    'backtest_stats': {
                        'total_r': int(best['total_r']),
                        'trades': best['trades'],
                        'win_rate': round(best['wr'], 1),
                        'walk_forward_score': f"{best['wf_score']}/6",
                        'monte_carlo_prob': round(best['mc_prob'], 1)
                    }
                }
            }
            with open('focused_optimal_config.yaml', 'w') as f:
                yaml.dump(best_config, f, default_flow_style=False)
            print(f"📁 Best config saved to focused_optimal_config.yaml")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
    print("=" * 100)


if __name__ == "__main__":
    main()
