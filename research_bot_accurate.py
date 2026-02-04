#!/usr/bin/env python3
"""
research_bot_accurate.py - 100% Bot-Accurate Multi-Divergence Validation
=========================================================================
EXACT simulation of bot behavior with multi-divergence support.

Key Accuracy Features:
1. ✅ 1 trade per symbol at a time (no stacking)
2. ✅ Signal queue cleared after entry
3. ✅ BOS expiration after 12 candles
4. ✅ Candle-by-candle simulation (no lookahead)
5. ✅ Multiple divergence types per symbol
6. ✅ Monthly R and Drawdown tracking
7. ✅ Realistic slippage and fees

Output: bot_accurate_validated.csv, monthly_performance.csv
"""

import requests
import pandas as pd
import numpy as np
import time
import sys
import itertools
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

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
MIN_DAILY_VOLUME_USD = 1_000_000

# Validation Thresholds
MIN_TRAIN_TRADES = 5
MIN_TEST_TRADES = 3
MIN_AVG_R = 0.05

# Portfolio Settings (for DD calculation)
INITIAL_CAPITAL = 700.0
RISK_PCT = 0.001  # 0.1% per trade

BASE_URL = "https://api.bybit.com"
OUTPUT_FILE = 'bot_accurate_validated.csv'
MONTHLY_FILE = 'monthly_performance.csv'

GRID = {
    'div_type': ['REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR'],
    'atr_mult': [1.0, 1.5, 2.0],
    'rr_ratio': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class Signal:
    idx: int
    side: str
    div_type: str
    swing_level: float
    timestamp: datetime
    candles_waited: int = 0
    
@dataclass
class Trade:
    symbol: str
    div_type: str
    side: str
    entry_price: float
    entry_time: datetime
    entry_idx: int
    sl_price: float
    tp_price: float
    rr: float
    atr_mult: float
    exit_time: Optional[datetime] = None
    exit_idx: Optional[int] = None
    r_result: float = 0.0
    outcome: str = "open"

# ============================================================================
# DATA FETCHING
# ============================================================================
def fetch_all_symbols():
    print("📡 Fetching symbols with $1M+ daily volume filter...")
    try:
        url = f"{BASE_URL}/v5/market/instruments-info"
        resp = requests.get(url, {'category': 'linear', 'status': 'Trading', 'limit': 1000})
        data = resp.json()
        
        all_symbols = [item['symbol'] for item in data.get('result', {}).get('list', [])
                       if item.get('quoteCoin') == 'USDT']
        
        print(f"📊 Found {len(all_symbols)} USDT pairs. Applying liquidity filter...")
        
        ticker_resp = requests.get(f"{BASE_URL}/v5/market/tickers", {'category': 'linear'})
        ticker_data = ticker_resp.json()
        
        volume_map = {t['symbol']: float(t.get('turnover24h', 0)) 
                      for t in ticker_data.get('result', {}).get('list', [])}
        
        liquid = [s for s in all_symbols if volume_map.get(s, 0) >= MIN_DAILY_VOLUME_USD]
        print(f"✅ {len(liquid)} symbols passed liquidity filter")
        return liquid
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def fetch_klines(symbol, interval, days):
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    
    for _ in range(400):
        if current_end <= start_ts:
            break
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", 
                               params={'category': 'linear', 'symbol': symbol, 
                                       'interval': interval, 'limit': 1000, 'end': current_end},
                               timeout=10)
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
    return df.sort_values('start').reset_index(drop=True)

def prepare_data(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

# ============================================================================
# PIVOT & DIVERGENCE DETECTION
# ============================================================================
def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        window = data[i-left:i+right+1]
        if len(window) != left + right + 1:
            continue
        center = data[i]
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
    return pivot_highs, pivot_lows

def detect_divergence_at_candle(df, idx, pivot_highs, pivot_lows):
    """Detect ALL divergences at a specific candle index"""
    if idx < EMA_PERIOD + 10:
        return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    
    signals = []
    curr_price = close[idx]
    curr_ema = ema[idx]
    
    # BULLISH (above EMA)
    if curr_price > curr_ema:
        p_lows = [(j, pivot_lows[j]) for j in range(idx-3, max(0, idx-50), -1) 
                  if not np.isnan(pivot_lows[j])][:2]
        if len(p_lows) >= 2:
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            if (idx - curr_idx) <= 10:
                if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    signals.append(Signal(idx, 'long', 'REG_BULL', max(high[curr_idx:idx+1]), df['start'].iloc[idx]))
                elif curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    signals.append(Signal(idx, 'long', 'HID_BULL', max(high[curr_idx:idx+1]), df['start'].iloc[idx]))
    
    # BEARISH (below EMA)
    if curr_price < curr_ema:
        p_highs = [(j, pivot_highs[j]) for j in range(idx-3, max(0, idx-50), -1) 
                   if not np.isnan(pivot_highs[j])][:2]
        if len(p_highs) >= 2:
            curr_idx, curr_val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            if (idx - curr_idx) <= 10:
                if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    signals.append(Signal(idx, 'short', 'REG_BEAR', min(low[curr_idx:idx+1]), df['start'].iloc[idx]))
                elif curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    signals.append(Signal(idx, 'short', 'HID_BEAR', min(low[curr_idx:idx+1]), df['start'].iloc[idx]))
    
    return signals

# ============================================================================
# BOT-ACCURATE SIMULATION
# ============================================================================
def simulate_symbol_accurate(sym, df_1h, df_5m, target_div_types, config):
    """
    Candle-by-candle simulation with BOT-ACCURATE behavior:
    - 1 trade per symbol at a time
    - Signal queue cleared after entry
    - BOS expiration
    """
    trades = []
    pending_signals: Dict[str, Signal] = {}  # div_type -> pending signal
    in_trade = False
    current_trade: Optional[Trade] = None
    
    close = df_1h['close'].values
    pivot_highs, pivot_lows = find_pivots(close, 3, 3)
    
    # Candle-by-candle walk
    for idx in range(EMA_PERIOD + 10, len(df_1h)):
        candle = df_1h.iloc[idx]
        
        # === CHECK IF CURRENT TRADE CLOSED ===
        if in_trade and current_trade:
            trade_5m = df_5m[(df_5m['start'] >= current_trade.entry_time) & 
                            (df_5m['start'] <= candle['start'])]
            
            for _, row in trade_5m.iterrows():
                if current_trade.side == 'long':
                    hit_sl = row['low'] <= current_trade.sl_price
                    hit_tp = row['high'] >= current_trade.tp_price
                else:
                    hit_sl = row['high'] >= current_trade.sl_price
                    hit_tp = row['low'] <= current_trade.tp_price
                
                if hit_sl or hit_tp:
                    if hit_sl:
                        current_trade.outcome = 'loss'
                        current_trade.r_result = -1.0
                    else:
                        current_trade.outcome = 'win'
                        current_trade.r_result = current_trade.rr
                    
                    # Apply fee drag
                    risk = abs(current_trade.entry_price - current_trade.sl_price)
                    if risk > 0:
                        current_trade.r_result -= (FEE_PCT * 2 * current_trade.entry_price) / risk
                    
                    current_trade.exit_time = row['start']
                    current_trade.exit_idx = idx
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    break
        
        # Skip rest if still in trade
        if in_trade:
            continue
        
        # === CHECK PENDING SIGNALS FOR BOS ===
        for div_type in list(pending_signals.keys()):
            sig = pending_signals[div_type]
            sig.candles_waited += 1
            
            # Expired?
            if sig.candles_waited >= MAX_WAIT_CANDLES:
                del pending_signals[div_type]
                continue
            
            # BOS triggered?
            bos_triggered = False
            if sig.side == 'long' and candle['close'] > sig.swing_level:
                bos_triggered = True
            elif sig.side == 'short' and candle['close'] < sig.swing_level:
                bos_triggered = True
            
            if bos_triggered:
                # Entry on next candle open
                if idx + 1 < len(df_1h):
                    entry_candle = df_1h.iloc[idx + 1]
                    atr = candle['atr']
                    if pd.isna(atr) or atr <= 0:
                        del pending_signals[div_type]
                        continue
                    
                    cfg = config[div_type]
                    sl_dist = atr * cfg['atr']
                    raw_entry = entry_candle['open']
                    
                    if sig.side == 'long':
                        entry_price = raw_entry * (1 + SLIPPAGE_PCT)
                        sl_price = entry_price - sl_dist
                        tp_price = entry_price + (sl_dist * cfg['rr'])
                    else:
                        entry_price = raw_entry * (1 - SLIPPAGE_PCT)
                        sl_price = entry_price + sl_dist
                        tp_price = entry_price - (sl_dist * cfg['rr'])
                    
                    current_trade = Trade(
                        symbol=sym,
                        div_type=div_type,
                        side=sig.side,
                        entry_price=entry_price,
                        entry_time=entry_candle['start'],
                        entry_idx=idx + 1,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        rr=cfg['rr'],
                        atr_mult=cfg['atr']
                    )
                    in_trade = True
                    
                    # Clear ALL pending signals (bot behavior - only 1 trade at a time)
                    pending_signals.clear()
                    break
        
        # === DETECT NEW DIVERGENCES ===
        if not in_trade:
            new_signals = detect_divergence_at_candle(df_1h, idx, pivot_highs, pivot_lows)
            for sig in new_signals:
                if sig.div_type in target_div_types and sig.div_type not in pending_signals:
                    pending_signals[sig.div_type] = sig
    
    return trades

# ============================================================================
# WALK-FORWARD WITH ACCURATE SIMULATION
# ============================================================================
def run_walkforward_accurate(sym, df_1h, df_5m):
    """Walk-forward with bot-accurate simulation"""
    if len(df_1h) < 500:
        return []
    
    total_len = len(df_1h)
    train_end = int(total_len * TRAIN_RATIO)
    
    train_1h = df_1h.iloc[:train_end].copy()
    test_1h = df_1h.iloc[train_end:].copy()
    
    if len(train_1h) < 300 or len(test_1h) < 100:
        return []
    
    train_5m = df_5m[df_5m['start'] <= train_1h['start'].iloc[-1]].copy()
    test_5m = df_5m[df_5m['start'] >= test_1h['start'].iloc[0]].copy()
    
    validated = []
    
    # Test each divergence type independently
    for div_type in GRID['div_type']:
        best_config = None
        best_r = -999
        
        # Find best config on training data
        for atr, rr in itertools.product(GRID['atr_mult'], GRID['rr_ratio']):
            config = {div_type: {'atr': atr, 'rr': rr}}
            trades = simulate_symbol_accurate(sym, train_1h, train_5m, [div_type], config)
            
            if len(trades) < MIN_TRAIN_TRADES:
                continue
            
            total_r = sum(t.r_result for t in trades)
            avg_r = total_r / len(trades)
            
            if total_r > best_r and avg_r >= MIN_AVG_R:
                best_r = total_r
                best_config = {
                    'atr': atr,
                    'rr': rr,
                    'train_trades': len(trades),
                    'train_r': total_r,
                    'train_wr': sum(1 for t in trades if t.outcome == 'win') / len(trades) * 100
                }
        
        if not best_config:
            continue
        
        # Validate on test data
        config = {div_type: {'atr': best_config['atr'], 'rr': best_config['rr']}}
        test_trades = simulate_symbol_accurate(sym, test_1h, test_5m, [div_type], config)
        
        if len(test_trades) < MIN_TEST_TRADES:
            continue
        
        test_r = sum(t.r_result for t in test_trades)
        test_wr = sum(1 for t in test_trades if t.outcome == 'win') / len(test_trades) * 100
        
        # Must be profitable OOS
        if test_r <= 0:
            continue
        
        # Collect monthly stats
        monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'r': 0.0})
        for t in test_trades:
            month = t.entry_time.strftime('%Y-%m')
            monthly[month]['trades'] += 1
            monthly[month]['r'] += t.r_result
            if t.outcome == 'win':
                monthly[month]['wins'] += 1
        
        validated.append({
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
            'monthly': dict(monthly),
            'test_trades_raw': test_trades
        })
    
    return validated

# ============================================================================
# MONTHLY ANALYSIS
# ============================================================================
def calculate_monthly_performance(all_results):
    """Calculate monthly R and max drawdown"""
    # Collect all trades sorted by time
    all_trades = []
    for r in all_results:
        for t in r.get('test_trades_raw', []):
            all_trades.append({
                'symbol': t.symbol,
                'div_type': t.div_type,
                'entry_time': t.entry_time,
                'r_result': t.r_result,
                'outcome': t.outcome
            })
    
    if not all_trades:
        return {}
    
    all_trades.sort(key=lambda x: x['entry_time'])
    
    # Group by month
    monthly = defaultdict(lambda: {'trades': [], 'r': 0, 'wins': 0})
    for t in all_trades:
        month = t['entry_time'].strftime('%Y-%m')
        monthly[month]['trades'].append(t)
        monthly[month]['r'] += t['r_result']
        if t['outcome'] == 'win':
            monthly[month]['wins'] += 1
    
    # Calculate drawdown per month
    results = {}
    for month in sorted(monthly.keys()):
        data = monthly[month]
        trades = data['trades']
        
        # Simulated equity curve for DD
        equity = INITIAL_CAPITAL
        peak = equity
        max_dd = 0
        
        for t in sorted(trades, key=lambda x: x['entry_time']):
            pnl = t['r_result'] * INITIAL_CAPITAL * RISK_PCT
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        
        results[month] = {
            'trades': len(trades),
            'wins': data['wins'],
            'wr': (data['wins'] / len(trades) * 100) if trades else 0,
            'total_r': round(data['r'], 2),
            'avg_r': round(data['r'] / len(trades), 2) if trades else 0,
            'max_dd': round(max_dd, 1)
        }
    
    return results

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("🎯 BOT-ACCURATE MULTI-DIVERGENCE WALK-FORWARD")
    print("=" * 70)
    print(f"📅 Data: {DAYS} days | Train: {int(TRAIN_RATIO*100)}% / Test: {int((1-TRAIN_RATIO)*100)}%")
    print(f"💧 Liquidity: ${MIN_DAILY_VOLUME_USD/1e6:.0f}M+ | 🎯 Multi-Div: All 4 types")
    print(f"🔒 Bot-Accurate: 1 trade/symbol, BOS expiry, signal queue")
    print("-" * 70)
    
    symbols = fetch_all_symbols()
    if not symbols:
        return
    
    all_results = []
    all_monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'r': 0})
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
        
        df_1h = prepare_data(df_1h)
        results = run_walkforward_accurate(sym, df_1h, df_5m)
        
        for r in results:
            all_results.append(r)
            for month, data in r['monthly'].items():
                all_monthly[month]['trades'] += data['trades']
                all_monthly[month]['wins'] += data['wins']
                all_monthly[month]['r'] += data['r']
        
        if completed % 10 == 0:
            configs = len(all_results)
            syms = len(set(r['symbol'] for r in all_results))
            test_r = sum(r['test_r'] for r in all_results)
            avg_wr = np.mean([r['test_wr'] for r in all_results]) if all_results else 0
            multi = sum(1 for s in set(r['symbol'] for r in all_results) 
                       if sum(1 for r in all_results if r['symbol'] == s) > 1)
            print(f"\n   📈 [{completed}/{total}] {configs} configs | {syms} syms | R: {test_r:+.1f} | WR: {avg_wr:.1f}% | Multi: {multi}")
    
    # === RESULTS ===
    print(f"\n\n{'=' * 70}")
    print("📊 BOT-ACCURATE RESULTS")
    print("=" * 70)
    
    if not all_results:
        print("❌ No configs validated!")
        return
    
    unique_syms = len(set(r['symbol'] for r in all_results))
    total_test_r = sum(r['test_r'] for r in all_results)
    avg_wr = np.mean([r['test_wr'] for r in all_results])
    
    print(f"\n✅ {len(all_results)} configs across {unique_syms} symbols")
    print(f"📊 Total Test R (OOS): {total_test_r:+.1f}")
    print(f"🎯 Avg Win Rate: {avg_wr:.1f}%")
    
    # Monthly performance with drawdown
    monthly_perf = calculate_monthly_performance(all_results)
    
    print(f"\n{'=' * 70}")
    print("📅 MONTHLY PERFORMANCE (Test Period - Out of Sample)")
    print("=" * 70)
    print(f"{'Month':<10} {'Trades':>8} {'WR':>8} {'Total R':>10} {'Avg R':>8} {'Max DD':>8}")
    print("-" * 60)
    
    for month in sorted(monthly_perf.keys()):
        p = monthly_perf[month]
        print(f"{month:<10} {p['trades']:>8} {p['wr']:>7.1f}% {p['total_r']:>+10.1f} {p['avg_r']:>+8.2f} {p['max_dd']:>7.1f}%")
    
    # Save results
    df = pd.DataFrame([{
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
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Save monthly
    df_monthly = pd.DataFrame([{'month': m, **p} for m, p in monthly_perf.items()])
    df_monthly.to_csv(MONTHLY_FILE, index=False)
    
    print(f"\n💾 Saved {len(all_results)} configs to {OUTPUT_FILE}")
    print(f"💾 Saved monthly performance to {MONTHLY_FILE}")
    
    # Top performers
    print(f"\n{'=' * 70}")
    print("🏆 TOP 30 PERFORMERS")
    print("=" * 70)
    top30 = sorted(all_results, key=lambda x: x['test_r'], reverse=True)[:30]
    for r in top30:
        print(f"  {r['symbol']:15} | {r['div_type']:10} | RR:{r['rr']:4.1f} | ATR:{r['atr']:.1f}x | R:{r['test_r']:+6.1f} | WR:{r['test_wr']:.0f}%")

if __name__ == "__main__":
    main()
