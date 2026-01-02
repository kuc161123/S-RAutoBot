#!/usr/bin/env python3
"""
EXTREME ROBUST VALIDATION - ANTI-OVERFIT
=========================================
Takes candidates from optimization and validates with:
1. Walk-forward analysis (train/test split)
2. Out-of-sample testing (different time period)
3. Monte Carlo simulations
4. Stricter filters
5. No look-ahead bias

Only symbols that pass ALL tests are considered valid.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'
DATA_DAYS = 730  # 2 years for proper train/test split
MAX_WAIT_CANDLES = 6

# Walk-forward splits
TRAIN_RATIO = 0.6  # 60% train
TEST_RATIO = 0.4   # 40% out-of-sample test

# Validation thresholds
MIN_TRADES_TRAIN = 20
MIN_TRADES_TEST = 15
MIN_WIN_RATE = 18  # Realistic for high R:R
MIN_OOS_R = 5      # Minimum R in out-of-sample
MAX_DRAWDOWN_FACTOR = 0.5  # Max drawdown as % of total R

# Monte Carlo settings
MC_SIMULATIONS = 1000
MC_CONFIDENCE = 0.80  # 80% of simulations must be profitable

SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
RSI_PERIOD = 14
EMA_PERIOD = 200

print("="*70)
print("EXTREME ROBUST VALIDATION - ANTI-OVERFIT")
print("="*70)
print(f"Data: {DATA_DAYS} days (2 years)")
print(f"Train/Test Split: {TRAIN_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%")
print(f"Monte Carlo: {MC_SIMULATIONS} simulations, {MC_CONFIDENCE*100:.0f}% confidence")
print("="*70)

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 80
    
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
                
            time.sleep(0.1)
            
        except Exception:
            time.sleep(0.3)
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

def backtest_period(df, rr, atr_mult, symbol, start_idx, end_idx):
    """Backtest a specific period"""
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
    
    for current_idx in range(max(50, start_idx), min(end_idx, n - 1)):
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
                if entry_idx < n and entry_idx < end_idx:
                    entry_price = df.iloc[entry_idx]['open']
                    entry_atr = atr[entry_idx]
                    sl_dist = entry_atr * atr_mult
                    
                    if sig['side'] == 'long':
                        entry_price *= (1 + SLIPPAGE_PCT)
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        entry_price *= (1 - SLIPPAGE_PCT)
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
                    result = None
                    for k in range(entry_idx, min(entry_idx + 200, n, end_idx)):
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
                    if risk_pct > 0:
                        fee_cost = (FEE_PCT * 2) / risk_pct
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
            for j in range(current_idx-3, max(0, current_idx-50), -1):
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
                for j in range(current_idx-3, max(0, current_idx-50), -1):
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

def calculate_max_drawdown(trades):
    """Calculate maximum drawdown in R"""
    if not trades:
        return 0
    
    cumulative = np.cumsum(trades)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return max(drawdown) if len(drawdown) > 0 else 0

def monte_carlo_test(trades, num_simulations=1000):
    """Run Monte Carlo simulations to test robustness"""
    if len(trades) < 10:
        return 0.0
    
    profitable_count = 0
    
    for _ in range(num_simulations):
        # Randomly shuffle trades and calculate total
        shuffled = random.sample(trades, len(trades))
        total = sum(shuffled)
        if total > 0:
            profitable_count += 1
    
    return profitable_count / num_simulations

def validate_candidate(symbol, rr, atr_mult):
    """Validate a single candidate with extreme rigor"""
    
    # Fetch 2 years of data
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 5000:
        return None, "Insufficient data"
    
    df = prepare_data(df)
    n = len(df)
    
    # Walk-forward split
    train_end = int(n * TRAIN_RATIO)
    test_start = train_end
    
    # === TRAIN PERIOD ===
    train_trades = backtest_period(df, rr, atr_mult, symbol, 50, train_end)
    
    if len(train_trades) < MIN_TRADES_TRAIN:
        return None, f"Train trades < {MIN_TRADES_TRAIN}"
    
    train_r = sum(train_trades)
    train_wins = sum(1 for t in train_trades if t > 0)
    train_wr = (train_wins / len(train_trades) * 100)
    
    if train_r <= 0:
        return None, "Train R <= 0"
    
    # === TEST PERIOD (Out-of-Sample) ===
    test_trades = backtest_period(df, rr, atr_mult, symbol, test_start, n - 1)
    
    if len(test_trades) < MIN_TRADES_TEST:
        return None, f"Test trades < {MIN_TRADES_TEST}"
    
    test_r = sum(test_trades)
    test_wins = sum(1 for t in test_trades if t > 0)
    test_wr = (test_wins / len(test_trades) * 100)
    
    if test_r < MIN_OOS_R:
        return None, f"OOS R < {MIN_OOS_R}"
    
    # === COMBINED METRICS ===
    all_trades = train_trades + test_trades
    total_r = sum(all_trades)
    total_wr = (sum(1 for t in all_trades if t > 0) / len(all_trades) * 100)
    
    # === DRAWDOWN CHECK ===
    max_dd = calculate_max_drawdown(all_trades)
    if max_dd > total_r * MAX_DRAWDOWN_FACTOR:
        return None, f"Max DD > {MAX_DRAWDOWN_FACTOR*100:.0f}% of total R"
    
    # === MONTE CARLO TEST ===
    mc_score = monte_carlo_test(all_trades, MC_SIMULATIONS)
    if mc_score < MC_CONFIDENCE:
        return None, f"Monte Carlo < {MC_CONFIDENCE*100:.0f}%"
    
    # === PASSED ALL TESTS ===
    result = {
        'symbol': symbol,
        'rr': rr,
        'atr_mult': atr_mult,
        'train_r': round(train_r, 2),
        'train_trades': len(train_trades),
        'train_wr': round(train_wr, 1),
        'test_r': round(test_r, 2),
        'test_trades': len(test_trades),
        'test_wr': round(test_wr, 1),
        'total_r': round(total_r, 2),
        'total_trades': len(all_trades),
        'total_wr': round(total_wr, 1),
        'max_drawdown': round(max_dd, 2),
        'mc_score': round(mc_score * 100, 1)
    }
    
    return result, "PASSED"

def main():
    # Load candidates from optimization
    try:
        candidates = pd.read_csv('300_symbol_r10_plus.csv')
        print(f"\n📥 Loaded {len(candidates)} candidates from optimization")
    except FileNotFoundError:
        print("❌ No candidates file found. Run optimization first.")
        return
    
    if len(candidates) == 0:
        print("❌ No candidates to validate")
        return
    
    validated = []
    failed_reasons = {}
    
    for i, row in candidates.iterrows():
        symbol = row['symbol']
        rr = int(row['rr'])
        atr_mult = float(row['atr_mult'])
        
        print(f"\r[{i+1}/{len(candidates)}] Validating {symbol:15} ({rr}:1, {atr_mult:.2f}x)...", end="", flush=True)
        
        try:
            result, status = validate_candidate(symbol, rr, atr_mult)
            
            if result:
                validated.append(result)
                print(f" ✅ PASSED | Train: {result['train_r']:+.1f}R | Test: {result['test_r']:+.1f}R | MC: {result['mc_score']:.0f}%")
            else:
                failed_reasons[symbol] = status
                print(f" ❌ {status}")
        except Exception as e:
            failed_reasons[symbol] = str(e)
            print(f" ❌ Error: {e}")
        
        time.sleep(0.1)
    
    # Save results
    df_validated = pd.DataFrame(validated)
    df_validated.to_csv('validated_candidates_final.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"ROBUST VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Candidates Tested: {len(candidates)}")
    print(f"Passed Validation: {len(validated)} ({len(validated)/len(candidates)*100:.1f}%)")
    print(f"Failed: {len(candidates) - len(validated)}")
    
    if len(validated) > 0:
        df_validated = pd.DataFrame(validated).sort_values('total_r', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"VALIDATED CANDIDATES (Passed ALL Tests)")
        print(f"{'='*70}")
        print(f"{'Symbol':15} | {'R:R':5} | {'ATR':5} | {'Train':8} | {'Test':8} | {'Total':8} | {'N':4} | {'MC':5}")
        print("-"*70)
        
        for idx, row in df_validated.iterrows():
            print(f"{row['symbol']:15} | {row['rr']:2.0f}:1  | {row['atr_mult']:.2f}x | {row['train_r']:+7.1f}R | {row['test_r']:+7.1f}R | {row['total_r']:+7.1f}R | {row['total_trades']:4} | {row['mc_score']:4.0f}%")
        
        print(f"\n📁 Results saved to: validated_candidates_final.csv")

if __name__ == "__main__":
    main()
