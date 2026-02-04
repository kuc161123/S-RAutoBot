#!/usr/bin/env python3
"""
research_multi_divergence.py - Multi-Divergence Walk-Forward Validation
========================================================================
ROBUST research with ALL profitable divergence types per symbol.

Key Features:
1. 365 days of data (1 full year)
2. Walk-Forward Validation (Train on 9 months, Test on 3 months)
3. Liquidity Filter ($1M+ daily volume)
4. MULTIPLE divergence types per symbol if validated
5. Monthly performance breakdown

Output: multi_div_validated.csv
"""

import requests
import pandas as pd
import numpy as np
import time
import sys
import concurrent.futures
import threading
import os
import itertools
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================
DAYS = 365  # 1 FULL YEAR
SIGNAL_TF = '60'
EXECUTION_TF = '5'
MAX_WAIT_CANDLES = 12
RSI_PERIOD = 14
EMA_PERIOD = 200

# Walk-Forward Settings
TRAIN_RATIO = 0.75  # 75% train, 25% test

# Realistic Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

# Liquidity Filter
MIN_DAILY_VOLUME_USD = 1_000_000  # $1M minimum

# Validation Thresholds
MIN_TRAIN_TRADES = 5
MIN_TEST_TRADES = 3
MIN_AVG_R = 0.1  # Minimum expectancy to qualify

BASE_URL = "https://api.bybit.com"
OUTPUT_FILE = 'multi_div_validated.csv'

GRID = {
    'div_type': ['REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR'],
    'atr_mult': [1.0, 1.5, 2.0],
    'rr_ratio': [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# ============================================================================
# DATA ENGINE
# ============================================================================

def fetch_all_symbols():
    """Fetch all active USDT Linear Perpetuals with liquidity filter"""
    print("📡 Fetching symbols with $1M+ daily volume filter...")
    
    try:
        url = f"{BASE_URL}/v5/market/instruments-info"
        params = {'category': 'linear', 'status': 'Trading', 'limit': 1000}
        resp = requests.get(url, params=params)
        data = resp.json()
        
        all_symbols = []
        if 'result' in data and 'list' in data['result']:
            for item in data['result']['list']:
                if item.get('quoteCoin') == 'USDT':
                    all_symbols.append(item['symbol'])
        
        print(f"📊 Found {len(all_symbols)} total USDT pairs. Applying liquidity filter...")
        
        ticker_url = f"{BASE_URL}/v5/market/tickers"
        ticker_resp = requests.get(ticker_url, {'category': 'linear'})
        ticker_data = ticker_resp.json()
        
        volume_map = {}
        if 'result' in ticker_data and 'list' in ticker_data['result']:
            for t in ticker_data['result']['list']:
                sym = t.get('symbol', '')
                turnover = float(t.get('turnover24h', 0))
                volume_map[sym] = turnover
        
        liquid_symbols = [sym for sym in all_symbols if volume_map.get(sym, 0) >= MIN_DAILY_VOLUME_USD]
        
        print(f"✅ {len(liquid_symbols)} symbols passed liquidity filter (>${MIN_DAILY_VOLUME_USD/1e6:.0f}M daily volume).")
        return liquid_symbols
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch historical klines"""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    max_requests = 400
    
    while current_end > start_ts and max_requests > 0:
        max_requests -= 1
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data:
                break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000:
                break
            time.sleep(0.03)
        except:
            time.sleep(0.5)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def prepare_1h_data(df):
    """Add indicators"""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

# ============================================================================
# PIVOT & SIGNAL DETECTION
# ============================================================================

def find_pivots(data, left=3, right=3):
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    for i in range(left, len(data) - right):
        window = data[i-left : i+right+1]
        center = data[i]
        if len(window) != (left + right + 1):
            continue
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
    return pivot_highs, pivot_lows

def detect_signals(df):
    """Detect ALL divergence signals"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    used_pivots = set()
    
    for i in range(205, len(df) - 3):
        curr_price = close[i]
        curr_ema = ema[i]
        
        # BULLISH
        if curr_price > curr_ema:
            p_lows = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2:
                        break
            if len(p_lows) == 2:
                curr_idx, curr_val = p_lows[0]
                prev_idx, prev_val = p_lows[1]
                dedup_key = (curr_idx, prev_idx, 'BULL')
                if (i - curr_idx) <= 10 and dedup_key not in used_pivots:
                    if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'long', 'type': 'REG_BULL', 'swing': max(high[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
                    elif curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'long', 'type': 'HID_BULL', 'swing': max(high[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
        
        # BEARISH
        if curr_price < curr_ema:
            p_highs = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_ph[j]):
                    p_highs.append((j, price_ph[j]))
                    if len(p_highs) >= 2:
                        break
            if len(p_highs) == 2:
                curr_idx, curr_val = p_highs[0]
                prev_idx, prev_val = p_highs[1]
                dedup_key = (curr_idx, prev_idx, 'BEAR')
                if (i - curr_idx) <= 10 and dedup_key not in used_pivots:
                    if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'short', 'type': 'REG_BEAR', 'swing': min(low[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
                    elif curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        signals.append({'conf_idx': i, 'side': 'short', 'type': 'HID_BEAR', 'swing': min(low[curr_idx:i+1])})
                        used_pivots.add(dedup_key)
    
    return signals

# ============================================================================
# TRADE EXECUTION
# ============================================================================

def execute_trade(signal, df_1h, df_5m, rr_ratio, atr_mult):
    """Execute a single trade with BOS confirmation and 5M precision"""
    conf_idx = signal['conf_idx']
    side = signal['side']
    bos_level = signal['swing']
    
    if conf_idx + 1 >= len(df_1h):
        return None
    
    entry_price = None
    entry_time = None
    
    for i in range(1, MAX_WAIT_CANDLES + 1):
        if conf_idx + i >= len(df_1h):
            break
        candle = df_1h.iloc[conf_idx + i]
        triggered = False
        if side == 'long' and candle['close'] > bos_level:
            triggered = True
        elif side == 'short' and candle['close'] < bos_level:
            triggered = True
        
        if triggered:
            if conf_idx + i + 1 >= len(df_1h):
                break
            entry_candle = df_1h.iloc[conf_idx + i + 1]
            entry_time = entry_candle['start']
            atr = candle['atr']
            if pd.isna(atr) or atr <= 0:
                break
            sl_dist = atr * atr_mult
            raw_entry = entry_candle['open']
            
            if side == 'long':
                entry_price = raw_entry * (1 + SLIPPAGE_PCT)
                sl_price = entry_price - sl_dist
                tp_price = entry_price + (sl_dist * rr_ratio)
            else:
                entry_price = raw_entry * (1 - SLIPPAGE_PCT)
                sl_price = entry_price + sl_dist
                tp_price = entry_price - (sl_dist * rr_ratio)
            break
    
    if not entry_price or not entry_time:
        return None
    
    five_min_subset = df_5m[df_5m['start'] >= entry_time]
    if five_min_subset.empty:
        return None
    
    outcome = None
    for row in five_min_subset.itertuples():
        if side == 'long':
            hit_sl = row.low <= sl_price
            hit_tp = row.high >= tp_price
        else:
            hit_sl = row.high >= sl_price
            hit_tp = row.low <= tp_price
        
        if hit_sl and hit_tp:
            outcome = "loss"
            break
        elif hit_sl:
            outcome = "loss"
            break
        elif hit_tp:
            outcome = "win"
            break
    
    if outcome is None:
        return None
    
    risk = abs(entry_price - sl_price)
    if risk == 0:
        return None
    
    r_result = rr_ratio if outcome == 'win' else -1.0
    fee_drag = (FEE_PCT * 2 * entry_price) / risk
    
    return {'r': r_result - fee_drag, 'entry_time': entry_time, 'outcome': outcome}

# ============================================================================
# MULTI-DIVERGENCE WALK-FORWARD
# ============================================================================

def run_multi_walkforward(sym, df_1h, df_5m):
    """
    Walk-Forward for ALL divergence types.
    Returns a list of validated configs (not just one).
    """
    if len(df_1h) < 500:
        return []
    
    total_len = len(df_1h)
    train_end_idx = int(total_len * TRAIN_RATIO)
    
    train_1h = df_1h.iloc[:train_end_idx].copy()
    test_1h = df_1h.iloc[train_end_idx:].copy()
    
    if len(train_1h) < 300 or len(test_1h) < 100:
        return []
    
    train_start = train_1h['start'].iloc[0]
    train_end = train_1h['start'].iloc[-1]
    test_start = test_1h['start'].iloc[0]
    test_end = test_1h['start'].iloc[-1]
    
    train_5m = df_5m[(df_5m['start'] >= train_start) & (df_5m['start'] <= train_end)].copy()
    test_5m = df_5m[(df_5m['start'] >= test_start) & (df_5m['start'] <= test_end)].copy()
    
    # Get signals for both periods
    train_signals = detect_signals(train_1h)
    test_signals = detect_signals(test_1h)
    
    validated_configs = []
    
    # Test EACH divergence type independently
    for div_type in GRID['div_type']:
        train_div_signals = [s for s in train_signals if s['type'] == div_type]
        test_div_signals = [s for s in test_signals if s['type'] == div_type]
        
        if len(train_div_signals) < MIN_TRAIN_TRADES:
            continue
        
        # === FIND BEST ATR/RR FOR THIS DIV TYPE ===
        best_config = None
        best_train_r = -999
        
        for atr, rr in itertools.product(GRID['atr_mult'], GRID['rr_ratio']):
            trades = []
            for sig in train_div_signals:
                res = execute_trade(sig, train_1h, train_5m, rr, atr)
                if res:
                    trades.append(res['r'])
            
            if len(trades) < MIN_TRAIN_TRADES:
                continue
            
            total_r = sum(trades)
            avg_r = total_r / len(trades)
            wr = sum(1 for t in trades if t > 0) / len(trades) * 100
            
            if total_r > best_train_r and avg_r >= MIN_AVG_R:
                best_train_r = total_r
                best_config = {
                    'atr': atr,
                    'rr': rr,
                    'train_trades': len(trades),
                    'train_r': total_r,
                    'train_wr': wr
                }
        
        if not best_config:
            continue
        
        # === VALIDATE ON TEST DATA ===
        test_trades = []
        monthly_r = {}
        
        for sig in test_div_signals:
            res = execute_trade(sig, test_1h, test_5m, best_config['rr'], best_config['atr'])
            if res:
                test_trades.append(res)
                month = res['entry_time'].strftime('%Y-%m')
                if month not in monthly_r:
                    monthly_r[month] = {'trades': 0, 'wins': 0, 'r': 0}
                monthly_r[month]['trades'] += 1
                monthly_r[month]['r'] += res['r']
                if res['outcome'] == 'win':
                    monthly_r[month]['wins'] += 1
        
        if len(test_trades) < MIN_TEST_TRADES:
            continue
        
        test_r = sum(t['r'] for t in test_trades)
        test_wr = sum(1 for t in test_trades if t['outcome'] == 'win') / len(test_trades) * 100
        test_avg_r = test_r / len(test_trades)
        
        # Only include if test performance is positive
        if test_avg_r >= 0:
            validated_configs.append({
                'symbol': sym,
                'div_type': div_type,
                'atr': best_config['atr'],
                'rr': best_config['rr'],
                'train_trades': best_config['train_trades'],
                'train_r': round(best_config['train_r'], 2),
                'train_wr': round(best_config['train_wr'], 1),
                'test_trades': len(test_trades),
                'test_r': round(test_r, 2),
                'test_wr': round(test_wr, 1),
                'monthly': monthly_r
            })
    
    return validated_configs

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 MULTI-DIVERGENCE WALK-FORWARD RESEARCH")
    print("=" * 70)
    print(f"📅 Data Period: {DAYS} days (1 year)")
    print(f"🔬 Train: {int(TRAIN_RATIO*100)}% | Test: {int((1-TRAIN_RATIO)*100)}%")
    print(f"💧 Liquidity Filter: ${MIN_DAILY_VOLUME_USD/1e6:.0f}M+ daily volume")
    print(f"🎯 Multi-Div: Testing ALL 4 divergence types per symbol")
    print("-" * 70)
    
    symbols = fetch_all_symbols()
    if not symbols:
        print("❌ No symbols found!")
        return
    
    all_results = []
    all_monthly = {}
    completed = 0
    total = len(symbols)
    
    for sym in symbols:
        completed += 1
        sys.stdout.write(f"\r⏳ Processing {sym} ({completed}/{total})...")
        sys.stdout.flush()
        
        df_1h = fetch_klines(sym, SIGNAL_TF, DAYS)
        df_5m = fetch_klines(sym, EXECUTION_TF, DAYS)
        
        if df_1h.empty or df_5m.empty or len(df_1h) < 500:
            continue
        
        df_1h = prepare_1h_data(df_1h)
        
        results = run_multi_walkforward(sym, df_1h, df_5m)
        
        for r in results:
            all_results.append(r)
            for month, data in r['monthly'].items():
                if month not in all_monthly:
                    all_monthly[month] = {'trades': 0, 'wins': 0, 'r': 0}
                all_monthly[month]['trades'] += data['trades']
                all_monthly[month]['wins'] += data['wins']
                all_monthly[month]['r'] += data['r']
        
        if completed % 10 == 0:
            passed = len(all_results)
            symbols_with_configs = len(set(r['symbol'] for r in all_results))
            total_test_r_so_far = sum(r['test_r'] for r in all_results)
            avg_wr = np.mean([r['test_wr'] for r in all_results]) if all_results else 0
            multi_count = sum(1 for s in set(r['symbol'] for r in all_results) 
                             if sum(1 for r in all_results if r['symbol'] == s) > 1)
            print(f"\n   📈 [{completed}/{total}] {passed} configs | {symbols_with_configs} symbols | Test R: {total_test_r_so_far:+.1f} | Avg WR: {avg_wr:.1f}% | Multi-div: {multi_count}")
    
    print(f"\n\n{'=' * 70}")
    print("📊 MULTI-DIVERGENCE RESULTS")
    print("=" * 70)
    
    if not all_results:
        print("❌ No configs passed validation!")
        return
    
    unique_symbols = set(r['symbol'] for r in all_results)
    total_train_r = sum(r['train_r'] for r in all_results)
    total_test_r = sum(r['test_r'] for r in all_results)
    avg_test_wr = np.mean([r['test_wr'] for r in all_results])
    
    print(f"\n✅ {len(all_results)} CONFIGS validated across {len(unique_symbols)} symbols")
    print(f"📈 Total Train R: {total_train_r:+.1f}")
    print(f"📊 Total Test R (OUT OF SAMPLE): {total_test_r:+.1f}")
    print(f"🎯 Avg Test Win Rate: {avg_test_wr:.1f}%")
    
    # Symbols with multiple divergences
    symbol_counts = {}
    for r in all_results:
        symbol_counts[r['symbol']] = symbol_counts.get(r['symbol'], 0) + 1
    
    multi_div_symbols = [(s, c) for s, c in symbol_counts.items() if c > 1]
    print(f"\n🔥 {len(multi_div_symbols)} symbols have MULTIPLE validated divergence types!")
    
    # Monthly breakdown with DRAWDOWN
    print(f"\n{'=' * 70}")
    print("📅 MONTHLY BREAKDOWN (Test Period - Out of Sample)")
    print("=" * 70)
    print(f"{'Month':<10} {'Trades':>8} {'WR':>8} {'Total R':>10} {'Avg R':>8} {'Max DD':>8}")
    print("-" * 60)
    
    INITIAL_CAP = 700.0
    RISK_PCT = 0.001
    
    for month in sorted(all_monthly.keys()):
        data = all_monthly[month]
        wr = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
        avg_r = data['r'] / data['trades'] if data['trades'] > 0 else 0
        
        # Calculate max DD for this month (simplified - based on cumulative R)
        # Real DD would need trade-by-trade data, this is approximate
        equity = INITIAL_CAP
        peak = equity
        max_dd = 0
        # Simulate losses followed by wins as worst case
        losses = data['trades'] - data['wins']
        for _ in range(losses):
            equity -= INITIAL_CAP * RISK_PCT
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        for _ in range(data['wins']):
            avg_rr = (data['r'] + losses) / data['wins'] if data['wins'] > 0 else 0
            equity += avg_rr * INITIAL_CAP * RISK_PCT
            peak = max(peak, equity)
        
        print(f"  {month:<10} {data['trades']:>8} {wr:>7.1f}% {data['r']:>+10.1f} {avg_r:>+8.2f} {max_dd:>7.1f}%")
    
    # Save results
    df_results = pd.DataFrame([{
        'symbol': r['symbol'],
        'div_type': r['div_type'],
        'atr': r['atr'],
        'rr': r['rr'],
        'train_trades': r['train_trades'],
        'train_r': r['train_r'],
        'train_wr': r['train_wr'],
        'test_trades': r['test_trades'],
        'test_r': r['test_r'],
        'test_wr': r['test_wr']
    } for r in all_results])
    
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved {len(all_results)} validated configs to {OUTPUT_FILE}")
    
    # Top performers
    print(f"\n{'=' * 70}")
    print("🏆 TOP 30 PERFORMERS (by Out-of-Sample R)")
    print("=" * 70)
    
    top30 = sorted(all_results, key=lambda x: x['test_r'], reverse=True)[:30]
    for r in top30:
        print(f"  {r['symbol']:15} | {r['div_type']:10} | RR: {r['rr']:4.1f} | ATR: {r['atr']:.1f}x | Test R: {r['test_r']:+6.1f} | WR: {r['test_wr']:.1f}%")
    
    # Multi-div symbols
    if multi_div_symbols:
        print(f"\n{'=' * 70}")
        print("🔥 SYMBOLS WITH MULTIPLE DIVERGENCE TYPES")
        print("=" * 70)
        for sym, count in sorted(multi_div_symbols, key=lambda x: -x[1])[:20]:
            sym_results = [r for r in all_results if r['symbol'] == sym]
            sym_test_r = sum(r['test_r'] for r in sym_results)
            divs = ', '.join(r['div_type'] for r in sym_results)
            print(f"  {sym:15} | {count} divs: {divs} | Combined Test R: {sym_test_r:+.1f}")

if __name__ == "__main__":
    main()
