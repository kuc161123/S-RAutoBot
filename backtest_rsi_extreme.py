#!/usr/bin/env python3
"""
RSI EXTREME Strategy Backtest

Strategy Logic (Proven 70% WR in research):
- LONG: RSI crosses below 30, then back above 30 = entry
- SHORT: RSI crosses above 70, then back below 70 = entry
- R:R = 2:1 (TP = 2 ATR, SL = 1 ATR)
- Timeframe: 15M

Based on documented backtests showing 70% win rate over 2 years.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
import yaml
from datetime import datetime, timedelta
from collections import defaultdict

BYBIT_BASE = "https://api.bybit.com"

# EXECUTION COSTS
TOTAL_FEE_PCT = 0.11
SLIPPAGE_PCT = 0.02
TOTAL_COST_PCT = TOTAL_FEE_PCT + SLIPPAGE_PCT

# R:R RATIO - 2:1 as per research
RR_RATIO = 2.0

# RSI THRESHOLDS
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MINIMUM REQUIREMENTS
MIN_TRADES = 10
MIN_WR = 45  # At 2:1, breakeven is 33%


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)


def calculate_ev(wr_decimal: float, rr: float = 2.0) -> float:
    return (wr_decimal * rr) - ((1 - wr_decimal) * 1.0)


def get_all_symbols(limit=100) -> list:
    url = f"{BYBIT_BASE}/v5/market/tickers"
    params = {'category': 'linear'}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get('retCode') != 0:
            return []
        tickers = data.get('result', {}).get('list', [])
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_tickers.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt_tickers[:limit]]
    except Exception as e:
        print(f"Error: {e}")
        return []


def fetch_klines(symbol: str, interval: str = '15', days: int = 60) -> pd.DataFrame:
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    
    while current_end > start_time:
        params = {
            'category': 'linear', 'symbol': symbol, 'interval': interval,
            'limit': 1000, 'end': current_end
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
                break
            klines = data['result']['list']
            all_data.extend(klines)
            earliest = int(klines[-1][0])
            if earliest <= start_time:
                break
            current_end = earliest - 1
            time.sleep(0.05)
        except:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Previous RSI for crossover detection
    df['prev_rsi'] = df['rsi'].shift(1)
    
    return df.dropna()


def detect_rsi_signal(row, prev_row) -> str:
    """
    Detect RSI extreme signals:
    - LONG: RSI was below 30, now crosses back above 30
    - SHORT: RSI was above 70, now crosses back below 70
    """
    if prev_row is None:
        return None
    
    prev_rsi = prev_row['rsi']
    curr_rsi = row['rsi']
    
    # Long: RSI crosses UP through 30 (was oversold, recovering)
    if prev_rsi < RSI_OVERSOLD and curr_rsi >= RSI_OVERSOLD:
        return 'long'
    
    # Short: RSI crosses DOWN through 70 (was overbought, reversing)
    if prev_rsi > RSI_OVERBOUGHT and curr_rsi <= RSI_OVERBOUGHT:
        return 'short'
    
    return None


def simulate_trade(df, entry_idx, side, entry, atr, max_candles=100):
    """Simulate trade with 2:1 R:R."""
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * RR_RATIO
    
    entry_cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + entry_cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - entry_cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        
        if side == 'long':
            if candle['low'] <= sl:
                return 'loss', i - entry_idx
            elif candle['high'] >= tp:
                return 'win', i - entry_idx
        else:
            if candle['high'] >= sl:
                return 'loss', i - entry_idx
            elif candle['low'] <= tp:
                return 'win', i - entry_idx
    
    return 'timeout', max_candles


def process_symbol(symbol: str, days: int = 60, train_pct: float = 0.6):
    df = fetch_klines(symbol, '15', days)
    
    if len(df) < 500:
        return None, 0, 0
    
    df = calculate_indicators(df)
    if len(df) < 200:
        return None, 0, 0
    
    total_candles = len(df)
    train_end = int(total_candles * train_pct)
    
    results = {
        'long': {'train_w': 0, 'train_l': 0, 'test_w': 0, 'test_l': 0},
        'short': {'train_w': 0, 'train_l': 0, 'test_w': 0, 'test_l': 0}
    }
    
    total_signals = 0
    total_wins = 0
    
    # Process all candles
    for idx in range(20, len(df) - 100):
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        
        side = detect_rsi_signal(row, prev_row)
        if not side:
            continue
        
        outcome, candles = simulate_trade(df, idx, side, row['close'], row['atr'])
        total_signals += 1
        
        is_train = idx < train_end
        
        if outcome == 'win':
            total_wins += 1
            if is_train:
                results[side]['train_w'] += 1
            else:
                results[side]['test_w'] += 1
        else:
            if is_train:
                results[side]['train_l'] += 1
            else:
                results[side]['test_l'] += 1
    
    overall_wr = (total_wins / total_signals * 100) if total_signals > 0 else 0
    return results, total_signals, overall_wr


def run_backtest(num_symbols=100, days=60):
    print("="*70)
    print("📊 RSI EXTREME STRATEGY BACKTEST")
    print("="*70)
    print(f"   Strategy: RSI Oversold/Overbought Reversal")
    print(f"   RSI Thresholds: Buy < {RSI_OVERSOLD}, Sell > {RSI_OVERBOUGHT}")
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Timeframe: 15M")
    print(f"   Days: {days}")
    print(f"   Min WR: {MIN_WR}%")
    print("="*70)
    
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    all_results = []
    symbols_profitable = 0
    
    print(f"\n📊 Processing...")
    print("-" * 70)
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        try:
            results, total_signals, overall_wr = process_symbol(symbol, days)
            
            if results and total_signals > 0:
                for side in ['long', 'short']:
                    r = results[side]
                    total = r['train_w'] + r['train_l'] + r['test_w'] + r['test_l']
                    wins = r['train_w'] + r['test_w']
                    
                    if total >= MIN_TRADES:
                        wr = wins / total * 100
                        lb_wr = wilson_lower_bound(wins, total)
                        ev = calculate_ev(wins / total)
                        
                        train_total = r['train_w'] + r['train_l']
                        test_total = r['test_w'] + r['test_l']
                        train_wr = r['train_w'] / train_total * 100 if train_total > 0 else 0
                        test_wr = r['test_w'] / test_total * 100 if test_total > 0 else 0
                        
                        if wr >= MIN_WR:
                            all_results.append({
                                'symbol': symbol, 'side': side, 'trades': total,
                                'wins': wins, 'wr': round(wr, 1), 'lb_wr': round(lb_wr, 1),
                                'ev': round(ev, 2), 'train_wr': round(train_wr, 1),
                                'test_wr': round(test_wr, 1)
                            })
                            symbols_profitable += 1
                            print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | ✅ {side:5} | N={total:3} | WR={wr:.1f}% | EV={ev:+.2f}R")
                
                if not any(r['symbol'] == symbol for r in all_results):
                    if overall_wr >= 30:
                        print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | N={total_signals:3} | WR={overall_wr:.1f}%")
            else:
                print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | ⚠️  Insufficient signals")
                
        except Exception as e:
            print(f"[{i+1:3}/{len(symbols)}] {symbol:15} | ❌ Error: {str(e)[:25]}")
        
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(symbols) - i - 1)
            print(f"\n⏱️  {i+1}/{len(symbols)} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m\n")
        
        time.sleep(0.02)
    
    elapsed = time.time() - start_time
    
    all_results.sort(key=lambda x: x['ev'], reverse=True)
    
    long_count = len([r for r in all_results if r['side'] == 'long'])
    short_count = len([r for r in all_results if r['side'] == 'short'])
    
    print(f"\n✅ Complete! Time: {elapsed/60:.1f}m")
    print(f"   Profitable setups: {len(all_results)}")
    print(f"   Long: {long_count} | Short: {short_count}")
    
    if all_results:
        avg_wr = np.mean([r['wr'] for r in all_results])
        avg_ev = np.mean([r['ev'] for r in all_results])
        print(f"   Avg WR: {avg_wr:.1f}% | Avg EV: {avg_ev:+.2f}R")
    
    print("\n" + "="*70)
    print(f"🏆 TOP 20 RSI EXTREME SETUPS")
    print("="*70)
    
    for i, r in enumerate(all_results[:20], 1):
        print(f"{i:2}. {r['symbol']:15} {r['side']:5} | N={r['trades']:3} | WR={r['wr']:.1f}% (LB:{r['lb_wr']:.1f}%) | EV={r['ev']:+.2f}R")
        print(f"    Train: {r['train_wr']:.0f}% | Test: {r['test_wr']:.0f}%")
    
    # Save to YAML
    output_file = 'backtest_rsi_extreme_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'strategy': 'RSI Extreme Reversal',
            'rr_ratio': f'{RR_RATIO}:1',
            'rsi_oversold': RSI_OVERSOLD,
            'rsi_overbought': RSI_OVERBOUGHT,
            'timeframe': '15M',
            'generated': datetime.utcnow().isoformat(),
            'profitable_setups': len(all_results),
            'avg_wr': round(avg_wr, 1) if all_results else 0,
            'avg_ev': round(avg_ev, 2) if all_results else 0
        }
    }
    
    for r in all_results:
        sym = r['symbol']
        if sym not in yaml_output:
            yaml_output[sym] = {}
        yaml_output[sym][r['side']] = {
            'trades': r['trades'], 'wr': r['wr'], 'lb_wr': r['lb_wr'],
            'ev': r['ev'], 'train_wr': r['train_wr'], 'test_wr': r['test_wr']
        }
    
    with open(output_file, 'w') as f:
        yaml.dump(yaml_output, f, default_flow_style=False)
    
    print(f"\n✅ Saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    run_backtest(num_symbols=100, days=60)
