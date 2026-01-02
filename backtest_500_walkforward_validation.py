#!/usr/bin/env python3
"""
500-SYMBOL WALK-FORWARD VALIDATION
===================================
Tests the top 100 symbols from the initial discovery across MULTIPLE
non-overlapping 6-month periods to validate robustness and consistency.

This prevents overfitting by ensuring symbols perform well across
different market conditions and time periods.

Validation Periods:
- Period 1: Last 6 months (0-180 days ago) [ORIGINAL]
- Period 2: 6-12 months ago (180-360 days ago) [OUT-OF-SAMPLE]
- Period 3: 12-18 months ago (360-540 days ago) [EXTENDED OOS]

A symbol MUST be profitable in at least 2 out of 3 periods to pass.

Output:
- Symbols that pass walk-forward validation
- Consistency metrics
- Period-by-period performance
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1H
PERIOD_DAYS = 180   # 6 months per period
MAX_WAIT_CANDLES = 6
ATR_MULT = 1.0
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006
RSI_PERIOD = 14
EMA_PERIOD = 200

# Validation criteria
MIN_TRADES = 10  # Lower for shorter periods
MIN_PROFITABLE_PERIODS = 2  # Must be profitable in 2/3 periods

BASE_URL = "https://api.bybit.com"

print("="*80)
print("500-SYMBOL WALK-FORWARD VALIDATION")
print("="*80)
print(f"Strategy: 1H RSI Divergence + EMA 200 + BOS")
print(f"Periods: 3 x 6-month periods (18 months total)")
print(f"Validation: Must be profitable in 2+ periods")
print("="*80)

def load_top_symbols():
    """Load top 100 symbols from previous backtest"""
    try:
        df = pd.read_csv('500_symbol_validated.csv')
        # Get top 100 by total R
        top_100 = df.nlargest(100, 'total_r')
        symbols_dict = dict(zip(top_100['symbol'], top_100['rr']))
        print(f"\n✅ Loaded {len(symbols_dict)} symbols from 500_symbol_validated.csv\n")
        return symbols_dict
    except FileNotFoundError:
        print("❌ 500_symbol_validated.csv not found. Run backtest_500_comprehensive.py first!")
        return {}

def fetch_klines_period(symbol, interval, start_days_ago, end_days_ago):
    """Fetch klines for a specific time period"""
    now = int(datetime.now().timestamp() * 1000)
    end_ts = now - (end_days_ago * 24 * 60 * 60 * 1000)
    start_ts = now - (start_days_ago * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 50
    
    while current_end > start_ts and max_iterations > 0:
        max_iterations -= 1
        params = {
            'category': 'linear', 
            'symbol': symbol, 
            'interval': interval, 
            'limit': 1000, 
            'end': current_end
        }
        
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.02)
            
        except Exception:
            time.sleep(0.1)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Filter to exact period
    period_start = datetime.now() - timedelta(days=start_days_ago)
    period_end = datetime.now() - timedelta(days=end_days_ago)
    df = df[(df.index >= period_start) & (df.index <= period_end)]
    
    return df

def prepare_data(df):
    """Calculate indicators"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    """Find pivot highs/lows"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1): 
            continue
        
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
            
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def backtest_symbol_rr(df, rr):
    """Backtest a symbol with specific R:R"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    
    trades = []
    seen_signals = set()
    pending_signal = None
    pending_wait = 0
    
    for current_idx in range(50, n - 1):
        current_price = close[current_idx]
        
        if pending_signal is not None:
            sig = pending_signal
            
            bos_confirmed = False
            if sig['side'] == 'long':
                if current_price > sig['swing']:
                    bos_confirmed = True
            else:
                if current_price < sig['swing']:
                    bos_confirmed = True
            
            if bos_confirmed:
                entry_idx = current_idx + 1
                if entry_idx < n:
                    entry_price = df.iloc[entry_idx]['open']
                    entry_atr = atr[entry_idx]
                    sl_dist = entry_atr * ATR_MULT
                    
                    if sig['side'] == 'long':
                        entry_price *= (1 + SLIPPAGE_PCT)
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        entry_price *= (1 - SLIPPAGE_PCT)
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
                    result = None
                    for k in range(entry_idx, min(entry_idx + 200, n)):
                        if sig['side'] == 'long':
                            if low[k] <= sl_price:
                                result = -1.0
                                break
                            if high[k] >= tp_price:
                                result = rr
                                break
                        else:
                            if high[k] >= sl_price:
                                result = -1.0
                                break
                            if low[k] <= tp_price:
                                result = rr
                                break
                    
                    if result is None:
                        result = -0.5
                    
                    risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                    fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                    final_r = result - fee_cost
                    trades.append(final_r)
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        if pending_signal is None:
            # BULLISH
            p_lows = []
            for j in range(current_idx - 3, max(0, current_idx - 50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2:
                        break
            
            if len(p_lows) == 2 and current_price > ema[current_idx]:
                curr_idx, curr_val = p_lows[0]
                prev_idx, prev_val = p_lows[1]
                
                if (current_idx - curr_idx) <= 10:
                    if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        swing_high = max(high[curr_idx:current_idx+1])
                        signal_id = f"long_{curr_idx}"
                        
                        if signal_id not in seen_signals:
                            if current_price <= swing_high:
                                seen_signals.add(signal_id)
                                pending_signal = {'side': 'long', 'swing': swing_high}
                                pending_wait = 0
            
            # BEARISH
            if pending_signal is None:
                p_highs = []
                for j in range(current_idx - 3, max(0, current_idx - 50), -1):
                    if not np.isnan(price_ph[j]):
                        p_highs.append((j, price_ph[j]))
                        if len(p_highs) >= 2:
                            break
                
                if len(p_highs) == 2 and current_price < ema[current_idx]:
                    curr_idx_h, curr_val_h = p_highs[0]
                    prev_idx_h, prev_val_h = p_highs[1]
                    
                    if (current_idx - curr_idx_h) <= 10:
                        if curr_val_h > prev_val_h and rsi[curr_idx_h] < rsi[prev_idx_h]:
                            swing_low = min(low[curr_idx_h:current_idx+1])
                            signal_id = f"short_{curr_idx_h}"
                            
                            if signal_id not in seen_signals:
                                if current_price >= swing_low:
                                    seen_signals.add(signal_id)
                                    pending_signal = {'side': 'short', 'swing': swing_low}
                                    pending_wait = 0
    
    return trades

def test_symbol_periods(symbol, rr):
    """Test symbol across all 3 periods"""
    periods = [
        {'name': 'P1 (Recent)', 'start': 180, 'end': 0},
        {'name': 'P2 (6-12mo)', 'start': 360, 'end': 180},
        {'name': 'P3 (12-18mo)', 'start': 540, 'end': 360}
    ]
    
    results = []
    
    for period in periods:
        try:
            df = fetch_klines_period(symbol, TIMEFRAME, period['start'], period['end'])
            
            if len(df) < 300:
                results.append(None)
                continue
            
            df = prepare_data(df)
            
            if len(df) < 200:
                results.append(None)
                continue
            
            trades = backtest_symbol_rr(df, rr)
            
            if len(trades) >= MIN_TRADES:
                total_r = sum(trades)
                wins = sum(1 for t in trades if t > 0)
                wr = (wins / len(trades) * 100)
                
                results.append({
                    'period': period['name'],
                    'trades': len(trades),
                    'wins': wins,
                    'wr': round(wr, 1),
                    'total_r': round(total_r, 2),
                    'is_profitable': total_r > 0
                })
            else:
                results.append(None)
                
        except Exception:
            results.append(None)
    
    return results

def main():
    symbols_dict = load_top_symbols()
    
    if not symbols_dict:
        return
    
    print(f"📊 Testing {len(symbols_dict)} symbols across 3 periods...\n")
    
    validated_symbols = []
    failed_symbols = []
    
    start_time = time.time()
    
    for i, (symbol, rr) in enumerate(symbols_dict.items()):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 1
        remaining = avg_time * (len(symbols_dict) - i - 1)
        eta_mins = remaining / 60
        
        print(f"\r[{i+1}/{len(symbols_dict)}] {symbol:18} (R:R {rr}:1, ETA: {eta_mins:.1f}m)...", end="", flush=True)
        
        period_results = test_symbol_periods(symbol, rr)
        
        # Count profitable periods
        profitable_periods = sum(1 for r in period_results if r and r['is_profitable'])
        total_periods = sum(1 for r in period_results if r)
        
        if total_periods >= 2 and profitable_periods >= MIN_PROFITABLE_PERIODS:
            # Calculate consistency
            total_r_all = sum(r['total_r'] for r in period_results if r)
            avg_r = total_r_all / total_periods if total_periods > 0 else 0
            
            validated_symbols.append({
                'symbol': symbol,
                'rr': rr,
                'profitable_periods': profitable_periods,
                'total_periods': total_periods,
                'consistency': f"{profitable_periods}/{total_periods}",
                'avg_r_per_period': round(avg_r, 1),
                'total_r_all_periods': round(total_r_all, 1),
                'p1': period_results[0],
                'p2': period_results[1],
                'p3': period_results[2]
            })
            print(f" ✅ {profitable_periods}/{total_periods} profitable | Avg: {avg_r:+.1f}R")
        else:
            failed_symbols.append({
                'symbol': symbol,
                'profitable_periods': profitable_periods,
                'total_periods': total_periods
            })
            print(f" ❌ {profitable_periods}/{total_periods} profitable")
        
        time.sleep(0.02)
    
    print("\n\n" + "="*80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*80)
    
    print(f"\n📊 SUMMARY")
    print(f"├─ Symbols Tested: {len(symbols_dict)}")
    print(f"├─ ✅ Passed Validation: {len(validated_symbols)} ({len(validated_symbols)/len(symbols_dict)*100:.1f}%)")
    print(f"└─ ❌ Failed: {len(failed_symbols)}")
    
    if not validated_symbols:
        print("\n⚠️ No symbols passed walk-forward validation")
        return
    
    # Sort by total R across all periods
    validated_df = pd.DataFrame(validated_symbols).sort_values('total_r_all_periods', ascending=False)
    
    print(f"\n🏆 TOP 30 WALK-FORWARD VALIDATED SYMBOLS")
    print("-"*80)
    print(f"{'Symbol':20} | {'R:R':5} | {'Consistency':11} | {'Total R':10} | {'Avg R/Period':12}")
    print("-"*80)
    
    for _, row in validated_df.head(30).iterrows():
        print(f"{row['symbol']:20} | {row['rr']:2.0f}:1  | {row['consistency']:11} | {row['total_r_all_periods']:+9.1f}R | {row['avg_r_per_period']:+11.1f}R")
    
    print("-"*80)
    
    # Save results
    save_df = validated_df.drop(['p1', 'p2', 'p3'], axis=1)
    save_df.to_csv('500_walkforward_validated.csv', index=False)
    
    print(f"\n📁 Results saved to:")
    print(f"   - 500_walkforward_validated.csv ({len(validated_symbols)} symbols)")
    
    print("\n" + "="*80)
    print("🎉 WALK-FORWARD VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nRobustness Verified:")
    print(f"  • {len(validated_symbols)} symbols profitable across multiple periods")
    print(f"  • These symbols show CONSISTENT performance")
    print(f"  • Much more reliable than single-period backtest")
    
    # Show consistency breakdown
    consistency_breakdown = validated_df.groupby('consistency').size()
    print(f"\n📈 CONSISTENCY BREAKDOWN:")
    for consistency, count in consistency_breakdown.items():
        print(f"  {consistency} periods: {count} symbols")

if __name__ == "__main__":
    main()
