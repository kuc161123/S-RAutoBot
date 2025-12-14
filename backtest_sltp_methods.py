#!/usr/bin/env python3
"""
TP/SL METHOD COMPARISON BACKTEST
=================================
Tests different methods for setting Stop Loss and Take Profit:

1. ATR-Based (Current): SL = 1×ATR, TP = 2.05×ATR
2. Swing-Based: SL at recent swing low/high, TP at 2×distance
3. Percentage-Based: Fixed 1% SL, 2% TP
4. Tight ATR: SL = 0.5×ATR, TP = 1×ATR (tighter stops)
5. Wide ATR: SL = 1.5×ATR, TP = 3×ATR (wider stops)
6. Dynamic: SL based on recent volatility percentile

All methods use the same RSI divergence signals for fair comparison.
"""

import requests
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEFRAME = '15'
DATA_DAYS = 60
NUM_SYMBOLS = 150

SLIPPAGE_PCT = 0.0005
FEE_PCT = 0.0004
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

BASE_URL = "https://api.bybit.com"

# =============================================================================
# SL/TP METHODS
# =============================================================================

class SLTPMethod:
    """Base class for SL/TP calculation methods"""
    
    @staticmethod
    def atr_current(row, side, entry):
        """Current method: 1×ATR SL, 2.05×ATR TP"""
        atr = row.atr
        if side == 'long':
            sl = entry - atr
            tp = entry + (2.05 * atr)
        else:
            sl = entry + atr
            tp = entry - (2.05 * atr)
        return sl, tp, 2.05
    
    @staticmethod
    def atr_tight(row, side, entry):
        """Tight stops: 0.5×ATR SL, 1×ATR TP"""
        atr = row.atr
        if side == 'long':
            sl = entry - (0.5 * atr)
            tp = entry + atr
        else:
            sl = entry + (0.5 * atr)
            tp = entry - atr
        return sl, tp, 2.0
    
    @staticmethod
    def atr_wide(row, side, entry):
        """Wide stops: 1.5×ATR SL, 3×ATR TP"""
        atr = row.atr
        if side == 'long':
            sl = entry - (1.5 * atr)
            tp = entry + (3 * atr)
        else:
            sl = entry + (1.5 * atr)
            tp = entry - (3 * atr)
        return sl, tp, 2.0
    
    @staticmethod
    def percentage_fixed(row, side, entry):
        """Fixed percentage: 1% SL, 2% TP"""
        if side == 'long':
            sl = entry * 0.99  # -1%
            tp = entry * 1.02  # +2%
        else:
            sl = entry * 1.01  # +1%
            tp = entry * 0.98  # -2%
        return sl, tp, 2.0
    
    @staticmethod
    def swing_based(row, side, entry, rows, idx):
        """SL at recent swing, TP at 2× distance"""
        lookback = 10
        
        if side == 'long':
            # Find recent low for SL
            lows = [r.low for r in rows[max(0, idx-lookback):idx+1]]
            sl = min(lows) if lows else entry - row.atr
            sl_dist = entry - sl
            tp = entry + (2 * sl_dist) if sl_dist > 0 else entry + row.atr
        else:
            # Find recent high for SL
            highs = [r.high for r in rows[max(0, idx-lookback):idx+1]]
            sl = max(highs) if highs else entry + row.atr
            sl_dist = sl - entry
            tp = entry - (2 * sl_dist) if sl_dist > 0 else entry - row.atr
        
        rr = 2.0
        return sl, tp, rr
    
    @staticmethod
    def volatility_adaptive(row, side, entry):
        """ATR + volatility multiplier based on percentile"""
        atr = row.atr
        atr_pct = getattr(row, 'atr_pct', 50) / 100  # 0-1 scale
        
        # Low volatility = tighter stops, high volatility = wider stops
        sl_mult = 0.5 + (atr_pct * 1.0)  # 0.5 to 1.5
        tp_mult = sl_mult * 2.05
        
        if side == 'long':
            sl = entry - (sl_mult * atr)
            tp = entry + (tp_mult * atr)
        else:
            sl = entry + (sl_mult * atr)
            tp = entry - (tp_mult * atr)
        
        return sl, tp, 2.05

# =============================================================================
# HELPERS
# =============================================================================

def calc_ev(wr, rr):
    return (wr * rr) - (1 - wr)

def wilson_lb(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def get_symbols(limit=150):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear")
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    all_candles = []
    current_end = end_ts
    candles_needed = days * 24 * 60 // int(interval)
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows

def detect_signals(df):
    if len(df) < 100: return []
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE: prev_pl, prev_pli = price_pl[j], j; break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: prev_ph, prev_phi = price_ph[j], j; break
        
        if curr_pl and prev_pl and curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli] and rsi[i] < RSI_OVERSOLD + 15:
            signals.append({'idx': i, 'side': 'long'}); continue
        if curr_ph and prev_ph and curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi] and rsi[i] > RSI_OVERBOUGHT - 15:
            signals.append({'idx': i, 'side': 'short'}); continue
        if curr_pl and prev_pl and curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli] and rsi[i] < RSI_OVERBOUGHT - 10:
            signals.append({'idx': i, 'side': 'long'}); continue
        if curr_ph and prev_ph and curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi] and rsi[i] > RSI_OVERSOLD + 10:
            signals.append({'idx': i, 'side': 'short'})
    
    return signals

def simulate_trade(rows, signal_idx, side, sl, tp, entry):
    """Simulate trade with given SL and TP"""
    start_idx = signal_idx + 1
    if start_idx >= len(rows) - 50:
        return 'timeout', 0
    
    # Apply slippage to entry
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp = tp * (1 - TOTAL_COST)  # Reduce TP for costs
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp = tp * (1 + TOTAL_COST)
    
    for bar_idx, future_row in enumerate(rows[start_idx:start_idx+100]):
        if side == 'long':
            if future_row.low <= sl:
                return 'loss', bar_idx + 1
            if future_row.high >= tp:
                return 'win', bar_idx + 1
        else:
            if future_row.high >= sl:
                return 'loss', bar_idx + 1
            if future_row.low <= tp:
                return 'win', bar_idx + 1
    
    return 'timeout', 100

# =============================================================================
# MAIN
# =============================================================================

def run_comparison():
    print("=" * 80)
    print("🔬 TP/SL METHOD COMPARISON BACKTEST")
    print("=" * 80)
    print("\nComparing SL/TP methods:")
    print("  1. ATR Current: 1×ATR SL, 2.05×ATR TP (current strategy)")
    print("  2. ATR Tight:   0.5×ATR SL, 1×ATR TP (tighter stops)")
    print("  3. ATR Wide:    1.5×ATR SL, 3×ATR TP (wider stops)")
    print("  4. Percentage:  1% SL, 2% TP (fixed)")
    print("  5. Swing-Based: Recent swing for SL, 2× for TP")
    print("  6. Adaptive:    ATR adjusted by volatility percentile")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📦 Fetching data for {len(symbols)} symbols...\n")
    
    all_symbols_data = []
    start = time.time()
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 400: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            
            # ATR percentile for adaptive method
            df['atr_pct'] = df['atr'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
            
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 150: continue
            
            signals = detect_signals(df)
            rows = list(df.itertuples())
            
            # Filter valid signals
            valid_signals = []
            last_idx = -20
            for sig in signals:
                i = sig['idx']
                if i - last_idx < 10 or i >= len(rows) - 100: continue
                row = rows[i]
                if pd.isna(row.atr) or row.atr <= 0 or not row.vol_ok: continue
                valid_signals.append({'idx': i, 'side': sig['side'], 'row': row, 'rows': rows})
                last_idx = i
            
            all_symbols_data.extend(valid_signals)
            
            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{NUM_SYMBOLS}] Loaded: {len(all_symbols_data)} signals")
            
            time.sleep(0.02)
        except Exception as e:
            continue
    
    print(f"\n⏱️ Data loaded in {(time.time()-start)/60:.1f}m | {len(all_symbols_data)} signals")
    
    # Define methods to test
    methods = {
        'ATR Current (1:2.05)': lambda r, s, e, rows, idx: SLTPMethod.atr_current(r, s, e),
        'ATR Tight (0.5:1)': lambda r, s, e, rows, idx: SLTPMethod.atr_tight(r, s, e),
        'ATR Wide (1.5:3)': lambda r, s, e, rows, idx: SLTPMethod.atr_wide(r, s, e),
        'Percentage (1%:2%)': lambda r, s, e, rows, idx: SLTPMethod.percentage_fixed(r, s, e),
        'Swing-Based': lambda r, s, e, rows, idx: SLTPMethod.swing_based(r, s, e, rows, idx),
        'Volatility Adaptive': lambda r, s, e, rows, idx: SLTPMethod.volatility_adaptive(r, s, e),
    }
    
    results = {}
    
    for method_name, get_sltp in methods.items():
        print(f"\n  Testing: {method_name}...")
        wins = losses = timeouts = 0
        bars_win = []
        bars_loss = []
        rr_sum = 0
        
        for sig in all_symbols_data:
            row = sig['row']
            side = sig['side']
            idx = sig['idx']
            rows = sig['rows']
            entry = rows[idx + 1].open if idx + 1 < len(rows) else row.close
            
            sl, tp, rr = get_sltp(row, side, entry, rows, idx)
            rr_sum += rr
            
            result, bars = simulate_trade(rows, idx, side, sl, tp, entry)
            
            if result == 'win':
                wins += 1
                bars_win.append(bars)
            elif result == 'loss':
                losses += 1
                bars_loss.append(bars)
            else:
                timeouts += 1
        
        total = wins + losses
        wr = wins / total if total > 0 else 0
        avg_rr = rr_sum / len(all_symbols_data) if all_symbols_data else 2.0
        ev = calc_ev(wr, avg_rr)
        lb = wilson_lb(wins, total)
        total_r = (wins * avg_rr) - losses
        
        results[method_name] = {
            'wins': wins, 
            'losses': losses, 
            'timeouts': timeouts,
            'total': total, 
            'wr': wr, 
            'ev': ev, 
            'lb': lb, 
            'total_r': total_r,
            'avg_rr': avg_rr,
            'avg_bars_win': np.mean(bars_win) if bars_win else 0,
            'avg_bars_loss': np.mean(bars_loss) if bars_loss else 0
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📊 RESULTS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<25} {'N':<8} {'WR%':<8} {'R:R':<6} {'EV':<8} {'Total R':<12} {'Win Bars':<10}")
    print("-" * 85)
    
    best_ev = max(r['ev'] for r in results.values())
    best_total = max(r['total_r'] for r in results.values())
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['ev'], reverse=True):
        ev_flag = "🏆" if r['ev'] == best_ev else "  "
        total_flag = "💰" if r['total_r'] == best_total else "  "
        print(f"{name:<25} {r['total']:<8} {r['wr']*100:.1f}%{'':<2} {r['avg_rr']:.1f}  {r['ev']:+.2f}{'':<2} {r['total_r']:+,.0f}R{'':<4} {r['avg_bars_win']:.1f} {ev_flag}{total_flag}")
    
    # Winner analysis
    print("\n" + "=" * 80)
    print("🏆 ANALYSIS")
    print("=" * 80)
    
    best_ev_method = max(results.items(), key=lambda x: x[1]['ev'])
    best_total_method = max(results.items(), key=lambda x: x[1]['total_r'])
    current = results.get('ATR Current (1:2.05)', {})
    
    print(f"\n📈 Best EV per trade: {best_ev_method[0]}")
    print(f"   EV: {best_ev_method[1]['ev']:+.2f} | WR: {best_ev_method[1]['wr']*100:.1f}%")
    
    print(f"\n💰 Best Total Profit: {best_total_method[0]}")
    print(f"   Total R: {best_total_method[1]['total_r']:+,.0f}")
    
    if current:
        print(f"\n📊 Current Method (ATR 1:2.05):")
        print(f"   EV: {current['ev']:+.2f} | WR: {current['wr']*100:.1f}% | Total: {current['total_r']:+,.0f}R")
        
        if best_ev_method[1]['ev'] > current['ev']:
            diff = best_ev_method[1]['ev'] - current['ev']
            print(f"\n✅ {best_ev_method[0]} beats current by +{diff:.2f} EV per trade")
        else:
            print(f"\n✅ Current method is optimal or near-optimal!")

if __name__ == "__main__":
    run_comparison()
