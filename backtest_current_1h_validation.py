#!/usr/bin/env python3
"""
CURRENT 1H STRATEGY VALIDATION
==============================
Tests the EXACT configuration currently running live:
- 1H timeframe
- 63 validated symbols with individual R:R ratios
- Recent data (last 6 months)
- Same divergence detection logic
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1 hour
DATA_DAYS = 180  # 6 months recent data
MAX_WAIT_CANDLES = 6

# Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
RSI_PERIOD = 14
EMA_PERIOD = 200

# Load symbols and R:R config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('symbol_rr_config.yaml', 'r') as f:
    rr_config = yaml.safe_load(f)

SYMBOLS = [s for s, props in config.get('symbols', {}).items() if props.get('enabled', False)]

print(f"="*70)
print(f"VALIDATING CURRENT 1H STRATEGY")
print(f"="*70)
print(f"Symbols: {len(SYMBOLS)}")
print(f"Data Period: {DATA_DAYS} days (last 6 months)")
print(f"Using individual R:R ratios from symbol_rr_config.yaml")
print(f"="*70)

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 20
    
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
                
            time.sleep(0.12)
            
        except Exception:
            time.sleep(0.5)
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
    """Calculate indicators - NO LOOK-AHEAD"""
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
    """Find pivot highs/lows - NO LOOK-AHEAD"""
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

def detect_divergences(df):
    """Detect divergences - EXACT SAME LOGIC AS LIVE BOT"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 15):
        curr_price = close[i]
        curr_ema = ema[i]
        
        # BULLISH
        p_lows = []
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: 
                    break
        
        if len(p_lows) == 2 and curr_price > curr_ema:
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            
            if (i - curr_idx) <= 10:
                if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    swing_high = max(high[curr_idx:i+1])
                    signals.append({
                        'idx': curr_idx,
                        'conf_idx': i,
                        'side': 'long',
                        'swing': swing_high,
                        'price': curr_val
                    })

        # BEARISH
        p_highs = [] 
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_ph[j]):
                p_highs.append((j, price_ph[j]))
                if len(p_highs) >= 2: 
                    break
        
        if len(p_highs) == 2 and curr_price < curr_ema:
            curr_idx, curr_val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            
            if (i - curr_idx) <= 10:
                if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    swing_low = min(low[curr_idx:i+1])
                    signals.append({
                        'idx': curr_idx,
                        'conf_idx': i,
                        'side': 'short',
                        'swing': swing_low,
                        'price': curr_val
                    })
                            
    return signals

def backtest_symbol(df, signals, rr, atr_mult=1.0):
    """Backtest with symbol's R:R ratio"""
    rows = list(df.itertuples())
    trades = []
    
    for sig in signals:
        start_idx = sig['conf_idx']
        side = sig['side']
        
        if start_idx >= len(rows):
            continue
            
        # Check for BOS within max wait
        entry_idx = None
        for j in range(start_idx + 1, min(start_idx + 1 + MAX_WAIT_CANDLES, len(rows))):
            row = rows[j]
            if side == 'long':
                if row.close > sig['swing']:
                    entry_idx = j + 1
                    break
            else:
                if row.close < sig['swing']:
                    entry_idx = j + 1
                    break
        
        if not entry_idx or entry_idx >= len(rows): 
            continue
        
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        atr = entry_row.atr
        sl_dist = atr * atr_mult
        
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
            tp_price = entry_price + (sl_dist * rr)
            sl_price = entry_price - sl_dist
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
            tp_price = entry_price - (sl_dist * rr)
            sl_price = entry_price + sl_dist
            
        result = None
        
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl_price:
                    result = -1.0
                    break
                if row.high >= tp_price:
                    result = rr
                    break
            else:
                if row.high >= sl_price:
                    result = -1.0
                    break
                if row.low <= tp_price:
                    result = rr
                    break
                    
        if result is None:
            result = -0.5
            
        risk_pct = abs(entry_price - sl_price) / entry_price
        if risk_pct == 0: 
            risk_pct = 0.01
        total_fee_cost = (FEE_PCT * 2) / risk_pct
        final_r = result - total_fee_cost
        trades.append(final_r)
        
    return trades

def test_symbol(symbol):
    """Test a single symbol with its configured R:R"""
    print(f"\n{'='*60}")
    print(f"Testing {symbol}")
    
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 500:
        print(f"  ❌ Insufficient data ({len(df)} candles)")
        return None
    
    print(f"  ✓ Fetched {len(df)} candles")
    
    df = prepare_data(df)
    signals = detect_divergences(df)
    
    print(f"  ✓ Detected {len(signals)} divergences")
    
    if len(signals) < 3:
        print(f"  ⚠️  Too few signals")
        return None
    
    # Get R:R for this symbol
    symbol_rr = rr_config.get(symbol, {}).get('rr', 5.0)
    
    trades = backtest_symbol(df, signals, symbol_rr, atr_mult=1.0)
    
    if len(trades) == 0:
        print(f"  ⚠️  No completed trades")
        return None
        
    total_r = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    wr = (wins / len(trades) * 100) if trades else 0
    avg_r = total_r / len(trades) if trades else 0
    
    result = {
        'symbol': symbol,
        'rr': symbol_rr,
        'total_r': round(total_r, 2),
        'wr': round(wr, 1),
        'avg_r': round(avg_r, 3),
        'trades': len(trades),
        'wins': wins
    }
    
    print(f"  ✅ {symbol_rr}:1 R:R → {total_r:+.1f}R ({wr:.0f}% WR, N={len(trades)})")
    
    return result

def main():
    all_results = []
    
    for i, sym in enumerate(SYMBOLS):
        print(f"\n[{i+1}/{len(SYMBOLS)}] {sym}")
        result = test_symbol(sym)
        if result:
            all_results.append(result)
        time.sleep(0.15)
    
    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('current_1h_validation.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS - LAST 6 MONTHS")
    print(f"{'='*70}")
    
    total_r = df_results['total_r'].sum()
    total_trades = df_results['trades'].sum()
    total_wins = df_results['wins'].sum()
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Total Symbols Tested: {len(df_results)}")
    print(f"Total Trades: {total_trades}")
    print(f"Total R: {total_r:+.1f}R")
    print(f"Overall Win Rate: {overall_wr:.1f}%")
    print(f"Avg R per Trade: {total_r/total_trades:+.3f}R")
    print(f"\nAnnualized Projection: {total_r * 2:+.0f}R/year")
    
    print(f"\n{'='*70}")
    print(f"TOP 10 PERFORMERS")
    print(f"{'='*70}")
    top = df_results.sort_values('total_r', ascending=False).head(10)
    for idx, row in top.iterrows():
        print(f"{row['symbol']:15} | {row['rr']:.0f}:1 R:R | {row['total_r']:+7.1f}R | WR: {row['wr']:5.1f}% | N: {int(row['trades']):3}")
    
    print(f"\n{'='*70}")
    print(f"BOTTOM 10 PERFORMERS")
    print(f"{'='*70}")
    bottom = df_results.sort_values('total_r', ascending=True).head(10)
    for idx, row in bottom.iterrows():
        print(f"{row['symbol']:15} | {row['rr']:.0f}:1 R:R | {row['total_r']:+7.1f}R | WR: {row['wr']:5.1f}% | N: {int(row['trades']):3}")

if __name__ == "__main__":
    main()
