#!/usr/bin/env python3
"""
6-MONTH VALIDATION BACKTEST (12-HOUR BOS TIMEOUT)
===================================================
Tests if increasing BOS timeout from 6 to 12 candles improves confirmation rate.

Compares:
- Original: 6-candle (6-hour) timeout
- This test: 12-candle (12-hour) timeout

All other settings identical to original validation.
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
TIMEFRAME = '60'  # 1H
DATA_DAYS = 180   # 6 months
MAX_WAIT_CANDLES = 12  # ⚡ TESTING: 12 hours instead of 6
ATR_MULT = 1.0
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006
RSI_PERIOD = 14
EMA_PERIOD = 200

BASE_URL = "https://api.bybit.com"

# Load symbols from config.yaml
def load_symbols_from_config():
    """Load enabled symbols and their R:R from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = {}
    for sym, data in config.get('symbols', {}).items():
        if data.get('enabled', False):
            symbols[sym] = data.get('rr', 5.0)
    
    return symbols

SYMBOLS = load_symbols_from_config()

print("="*70)
print("6-MONTH VALIDATION - 12-HOUR BOS TIMEOUT TEST")
print("="*70)
print(f"Symbols: {len(SYMBOLS)} enabled")
print(f"Timeframe: {TIMEFRAME} (1H)")
print(f"Data Period: {DATA_DAYS} days (6 months)")
print(f"BOS Timeout: {MAX_WAIT_CANDLES} candles (12 hours) ⚡ INCREASED")
print(f"Strategy: RSI Divergence + EMA200 + BOS")
print("="*70)

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
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
                
            time.sleep(0.05)
            
        except Exception as e:
            time.sleep(0.2)
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
    """Calculate indicators - exact same as live bot"""
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

def backtest_symbol(df, rr, symbol):
    """
    Backtest a symbol with EXACT bot logic.
    Returns list of trades with full details.
    """
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
        current_time = df.index[current_idx]
        
        # === CHECK PENDING SIGNAL FOR BOS ===
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
                    entry_time = df.index[entry_idx]
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
                    
                    # Simulate trade
                    result = None
                    exit_idx = None
                    for k in range(entry_idx, min(entry_idx + 200, n)):
                        if sig['side'] == 'long':
                            if low[k] <= sl_price:
                                result = -1.0
                                exit_idx = k
                                break
                            if high[k] >= tp_price:
                                result = rr
                                exit_idx = k
                                break
                        else:
                            if high[k] >= sl_price:
                                result = -1.0
                                exit_idx = k
                                break
                            if low[k] <= tp_price:
                                result = rr
                                exit_idx = k
                                break
                    
                    if result is None:
                        result = -0.5  # Timeout penalty
                        exit_idx = min(entry_idx + 200, n - 1)
                    
                    # Calculate fees
                    risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                    fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                    final_r = result - fee_cost
                    
                    exit_time = df.index[exit_idx] if exit_idx and exit_idx < len(df) else entry_time
                    
                    trades.append({
                        'symbol': symbol,
                        'side': sig['side'],
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'rr': rr,
                        'result_r': round(final_r, 3),
                        'is_win': final_r > 0
                    })
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        # === DETECT NEW DIVERGENCES ===
        if pending_signal is None:
            # BULLISH divergence
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
            
            # BEARISH divergence
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

def main():
    all_trades = []
    symbol_results = []
    
    print(f"\n📊 Testing {len(SYMBOLS)} symbols...\n")
    
    for i, (symbol, rr) in enumerate(SYMBOLS.items()):
        print(f"\r[{i+1}/{len(SYMBOLS)}] {symbol:15} (R:R {rr}:1)...", end="", flush=True)
        
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            
            if len(df) < 500:
                print(f" ⚠️ Insufficient data ({len(df)} candles)")
                continue
            
            df = prepare_data(df)
            trades = backtest_symbol(df, rr, symbol)
            
            if trades:
                all_trades.extend(trades)
                
                total_r = sum(t['result_r'] for t in trades)
                wins = sum(1 for t in trades if t['is_win'])
                wr = (wins / len(trades) * 100) if trades else 0
                
                symbol_results.append({
                    'symbol': symbol,
                    'rr': rr,
                    'trades': len(trades),
                    'wins': wins,
                    'wr': round(wr, 1),
                    'total_r': round(total_r, 2),
                    'avg_r': round(total_r / len(trades), 3) if trades else 0
                })
                
                status = "✅" if total_r > 0 else "❌"
                print(f" {status} {len(trades)} trades, {total_r:+.1f}R, {wr:.0f}% WR")
            else:
                print(f" ⚪ No trades")
                
        except Exception as e:
            print(f" ❌ Error: {e}")
        
        time.sleep(0.05)
    
    if not all_trades:
        print("\n❌ No trades found!")
        return
    
    # Convert to DataFrame for analysis
    trades_df = pd.DataFrame(all_trades)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    trades_df['entry_month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    trades_df['entry_week'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('W')
    
    # === OVERALL METRICS ===
    total_trades = len(trades_df)
    total_r = trades_df['result_r'].sum()
    total_wins = trades_df['is_win'].sum()
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_r_per_trade = total_r / total_trades if total_trades > 0 else 0
    
    # === TIME METRICS ===
    date_range = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = total_trades / date_range if date_range > 0 else 0
    trades_per_week = trades_per_day * 7
    trades_per_month = trades_per_day * 30
    r_per_day = total_r / date_range if date_range > 0 else 0
    r_per_week = r_per_day * 7
    r_per_month = r_per_day * 30
    
    # === MONTHLY BREAKDOWN ===
    monthly = trades_df.groupby('entry_month').agg({
        'result_r': ['sum', 'count', 'mean'],
        'is_win': 'sum'
    }).round(2)
    monthly.columns = ['total_r', 'trades', 'avg_r', 'wins']
    monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(1)
    
    # === DAILY BREAKDOWN ===
    daily = trades_df.groupby('entry_date').agg({
        'result_r': ['sum', 'count'],
        'is_win': 'sum'
    })
    daily.columns = ['total_r', 'trades', 'wins']
    
    # === PRINT RESULTS ===
    print("\n" + "="*70)
    print("6-MONTH VALIDATION RESULTS")
    print("="*70)
    
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"├─ Total Trades: {total_trades}")
    print(f"├─ Total R: {total_r:+.1f}R")
    print(f"├─ Wins: {total_wins} | Losses: {total_trades - total_wins}")
    print(f"├─ Win Rate: {overall_wr:.1f}%")
    print(f"└─ Avg R/Trade: {avg_r_per_trade:+.3f}R")
    
    print(f"\n⏱️ TRADE FREQUENCY (Expected)")
    print(f"├─ Trades per Day: {trades_per_day:.1f}")
    print(f"├─ Trades per Week: {trades_per_week:.1f}")
    print(f"├─ Trades per Month: {trades_per_month:.0f}")
    print(f"├─ R per Day: {r_per_day:+.2f}R")
    print(f"├─ R per Week: {r_per_week:+.1f}R")
    print(f"└─ R per Month: {r_per_month:+.1f}R")
    
    print(f"\n📅 MONTHLY BREAKDOWN")
    print("-"*60)
    print(f"{'Month':12} | {'Trades':7} | {'Wins':6} | {'WR%':6} | {'Total R':10} | {'Avg R':8}")
    print("-"*60)
    for month, row in monthly.iterrows():
        print(f"{str(month):12} | {row['trades']:7.0f} | {row['wins']:6.0f} | {row['wr']:5.1f}% | {row['total_r']:+9.1f}R | {row['avg_r']:+7.3f}R")
    print("-"*60)
    
    # Top/Bottom symbols
    sym_df = pd.DataFrame(symbol_results).sort_values('total_r', ascending=False)
    
    print(f"\n🏆 TOP 10 PERFORMERS")
    print("-"*60)
    for _, row in sym_df.head(10).iterrows():
        print(f"  {row['symbol']:15} | {row['rr']:2.0f}:1 | {row['trades']:3} trades | {row['total_r']:+7.1f}R | {row['wr']:.0f}% WR")
    
    print(f"\n⚠️ BOTTOM 10 PERFORMERS")
    print("-"*60)
    for _, row in sym_df.tail(10).iterrows():
        print(f"  {row['symbol']:15} | {row['rr']:2.0f}:1 | {row['trades']:3} trades | {row['total_r']:+7.1f}R | {row['wr']:.0f}% WR")
    
    # === PROFITABILITY CHECK ===
    print("\n" + "="*70)
    print("VALIDATION STATUS")
    print("="*70)
    
    profitable_months = (monthly['total_r'] > 0).sum()
    total_months = len(monthly)
    
    checks = []
    checks.append(("Total R > 0", total_r > 0, f"{total_r:+.1f}R"))
    checks.append(("Win Rate > 15%", overall_wr > 15, f"{overall_wr:.1f}%"))
    checks.append(("Avg R/Trade > 0", avg_r_per_trade > 0, f"{avg_r_per_trade:+.3f}R"))
    checks.append((f"Profitable Months > 50%", profitable_months / total_months > 0.5, f"{profitable_months}/{total_months}"))
    checks.append(("Trades/Day >= 1", trades_per_day >= 1, f"{trades_per_day:.1f}"))
    
    all_passed = True
    for name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {value}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 VALIDATION PASSED - Strategy is robust and profitable!")
    else:
        print("⚠️ VALIDATION FAILED - Some checks did not pass")
    print("="*70)
    
    # === SAVE RESULTS ===
    trades_df.to_csv('6month_validation_trades.csv', index=False)
    sym_df.to_csv('6month_validation_symbols.csv', index=False)
    monthly.to_csv('6month_validation_monthly.csv')
    
    print(f"\n📁 Results saved to:")
    print(f"   - 6month_validation_trades.csv (all trades)")
    print(f"   - 6month_validation_symbols.csv (per-symbol)")
    print(f"   - 6month_validation_monthly.csv (monthly breakdown)")

if __name__ == "__main__":
    main()
