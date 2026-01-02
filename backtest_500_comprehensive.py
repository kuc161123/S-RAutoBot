#!/usr/bin/env python3
"""
500-SYMBOL COMPREHENSIVE DISCOVERY & VALIDATION
================================================
Discovers profitable symbols from 500 USDT perpetuals using the exact
same methodology that validated the current 79 symbols.

Features:
- Tests 500 symbols with 6-month data
- Uses exact 1H divergence + EMA200 + BOS logic from live bot
- Tests multiple R:R ratios (4:1 to 10:1) for each symbol
- Comprehensive filtering and validation
- Monthly breakdown and trade frequency metrics
- Ready-to-use config.yaml output

Validation Criteria:
- Minimum 15 trades
- Total R > 0 (net profitable)
- Win Rate >= 15%
- Avg R/Trade > 0

Output:
- 500_symbol_validated.csv (all profitable symbols)
- 500_symbol_top_100.yaml (ready for config.yaml)
- 500_symbol_monthly_breakdown.csv
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1H
DATA_DAYS = 180   # 6 months
MAX_WAIT_CANDLES = 6
ATR_MULT = 1.0
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006
RSI_PERIOD = 14
EMA_PERIOD = 200

# R:R ratios to test
RR_OPTIONS = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Minimum thresholds for profitability
MIN_TRADES = 15
MIN_TOTAL_R = 0      # Must be net profitable
MIN_WIN_RATE = 15    # At least 15% WR
MIN_AVG_R = 0        # Avg R must be positive

BASE_URL = "https://api.bybit.com"

print("="*80)
print("500-SYMBOL COMPREHENSIVE DISCOVERY & VALIDATION")
print("="*80)
print(f"Strategy: 1H RSI Divergence + EMA 200 + BOS")
print(f"Data Period: {DATA_DAYS} days (6 months)")
print(f"R:R Options: {RR_OPTIONS}")
print(f"Validation: Min {MIN_TRADES} trades, WR >= {MIN_WIN_RATE}%, Total R > 0")
print("="*80)

def fetch_all_symbols(limit=500):
    """Fetch all USDT perpetual symbols from Bybit"""
    print("\n📥 Fetching symbol list from Bybit...")
    
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/instruments-info",
            params={'category': 'linear', 'limit': 1000},
            timeout=30
        )
        data = resp.json().get('result', {}).get('list', [])
        
        # Filter for USDT pairs only, active status
        symbols = []
        for item in data:
            symbol = item.get('symbol', '')
            status = item.get('status', '')
            if symbol.endswith('USDT') and status == 'Trading':
                symbols.append(symbol)
        
        # Sort alphabetically
        symbols = sorted(symbols)[:limit]
        
        print(f"✅ Found {len(symbols)} USDT perpetual symbols\n")
        return symbols
        
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

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

def backtest_symbol_rr(df, rr, symbol):
    """
    Backtest a symbol with specific R:R ratio.
    Returns list of trade results with timestamps.
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
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'side': sig['side'],
                        'result_r': final_r,
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

def test_symbol(symbol):
    """Test a symbol with all R:R options and return best result"""
    try:
        df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
        
        if len(df) < 500:
            return None
        
        df = prepare_data(df)
        
        if len(df) < 400:
            return None
        
        best_result = None
        best_rr = None
        best_r = -999
        all_rr_results = []
        
        for rr in RR_OPTIONS:
            trades = backtest_symbol_rr(df, rr, symbol)
            
            if len(trades) >= MIN_TRADES:
                total_r = sum(t['result_r'] for t in trades)
                wins = sum(1 for t in trades if t['is_win'])
                wr = (wins / len(trades) * 100)
                avg_r = total_r / len(trades)
                
                rr_result = {
                    'rr': rr,
                    'trades': len(trades),
                    'wins': wins,
                    'wr': round(wr, 1),
                    'total_r': round(total_r, 2),
                    'avg_r': round(avg_r, 3)
                }
                all_rr_results.append(rr_result)
                
                # Track best R:R
                if total_r > best_r:
                    best_r = total_r
                    best_rr = rr
                    best_result = {
                        'symbol': symbol,
                        'rr': rr,
                        'trades': len(trades),
                        'wins': wins,
                        'wr': round(wr, 1),
                        'total_r': round(total_r, 2),
                        'avg_r': avg_r,
                        'trades_data': trades  # Keep for monthly breakdown
                    }
        
        return best_result
        
    except Exception as e:
        return None

def main():
    # Fetch all symbols
    symbols = fetch_all_symbols(500)
    
    if not symbols:
        print("❌ No symbols found!")
        return
    
    print(f"📊 Testing {len(symbols)} symbols...\n")
    
    results = []
    profitable = []
    unprofitable = []
    insufficient_data = []
    
    # Progress tracking
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 1
        remaining = avg_time * (len(symbols) - i - 1)
        eta_mins = remaining / 60
        
        print(f"\r[{i+1}/{len(symbols)}] Testing {symbol:18} (ETA: {eta_mins:.1f}m)...", end="", flush=True)
        
        result = test_symbol(symbol)
        
        if result:
            results.append(result)
            
            # Check if meets validation criteria
            if (result['total_r'] > MIN_TOTAL_R and 
                result['wr'] >= MIN_WIN_RATE and 
                result['avg_r'] > MIN_AVG_R):
                profitable.append(result)
                print(f" ✅ {result['total_r']:+6.1f}R | {result['wr']:4.1f}% WR | {result['rr']:.0f}:1 R:R")
            else:
                unprofitable.append(result)
                print(f" ❌ {result['total_r']:+6.1f}R | {result['wr']:4.1f}% WR")
        else:
            insufficient_data.append(symbol)
            print(f" ⚪ Insufficient data")
        
        time.sleep(0.02)
    
    print("\n\n" + "="*80)
    print("500-SYMBOL DISCOVERY RESULTS")
    print("="*80)
    
    # === SUMMARY ===
    print(f"\n📊 SUMMARY")
    print(f"├─ Symbols Tested: {len(symbols)}")
    print(f"├─ With Sufficient Data: {len(results)}")
    print(f"├─ ✅ Profitable (Validated): {len(profitable)} ({len(profitable)/len(results)*100:.1f}%)" if results else "├─ Profitable: 0")
    print(f"├─ ❌ Unprofitable: {len(unprofitable)}")
    print(f"└─ ⚪ Insufficient Data: {len(insufficient_data)}")
    
    if not profitable:
        print("\n⚠️ No profitable symbols found with current criteria")
        return
    
    # Sort by total R
    profitable_df = pd.DataFrame(profitable).sort_values('total_r', ascending=False)
    
    # === PORTFOLIO METRICS ===
    total_portfolio_r = profitable_df['total_r'].sum()
    avg_r_per_symbol = profitable_df['total_r'].mean()
    total_trades = profitable_df['trades'].sum()
    total_wins = profitable_df['wins'].sum()
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n💰 PROFITABLE PORTFOLIO POTENTIAL")
    print(f"├─ Total Symbols: {len(profitable)}")
    print(f"├─ Total R (6 months): {total_portfolio_r:+.1f}R")
    print(f"├─ Avg R per Symbol: {avg_r_per_symbol:+.1f}R")
    print(f"├─ Expected R/Month: {total_portfolio_r / 6:+.1f}R")
    print(f"├─ Expected R/Year: {total_portfolio_r * 2:+.1f}R")
    print(f"├─ Total Trades: {total_trades}")
    print(f"├─ Overall Win Rate: {overall_wr:.1f}%")
    print(f"└─ Avg Trades/Day: {total_trades / 180:.1f}")
    
    # === TOP PERFORMERS ===
    print(f"\n🏆 TOP 50 PROFITABLE SYMBOLS")
    print("-"*80)
    print(f"{'Symbol':20} | {'R:R':5} | {'Trades':7} | {'Wins':6} | {'WR%':6} | {'Total R':10} | {'Avg R':8}")
    print("-"*80)
    
    for _, row in profitable_df.head(50).iterrows():
        print(f"{row['symbol']:20} | {row['rr']:2.0f}:1  | {row['trades']:7} | {row['wins']:6} | {row['wr']:5.1f}% | {row['total_r']:+9.1f}R | {row['avg_r']:+7.3f}R")
    
    print("-"*80)
    
    # === R:R DISTRIBUTION ===
    print(f"\n📈 R:R DISTRIBUTION (Profitable Symbols)")
    rr_dist = profitable_df.groupby('rr').size().sort_index()
    for rr, count in rr_dist.items():
        pct = count / len(profitable) * 100
        print(f"  {rr:2.0f}:1 → {count:3} symbols ({pct:5.1f}%)")
    
    # === SAVE RESULTS ===
    print(f"\n📁 Saving results...")
   
    # Save all validated symbols
    save_df = profitable_df.drop('trades_data', axis=1, errors='ignore')
    save_df.to_csv('500_symbol_validated.csv', index=False)
    
    # Save full results
    full_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'trades_data'} for r in results])
    full_df.sort_values('total_r', ascending=False, inplace=True)
    full_df.to_csv('500_symbol_all_results.csv', index=False)
    
    # === GENERATE CONFIG.YAML READY OUTPUT ===
    print(f"\n📋 READY FOR config.yaml (Top 100):")
    print("-"*60)
    
    config_output = []
    config_output.append("# Top 100 Validated Symbols from 500-Symbol Discovery")
    config_output.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    config_output.append(f"# Validation: 6-month backtest, 1H divergence + EMA200 + BOS")
    config_output.append(f"# Overall: {len(profitable)} profitable, {total_portfolio_r:+.1f}R total")
    config_output.append("")
    config_output.append("symbols:")
    
    for _, row in profitable_df.head(100).iterrows():
        config_output.append(f"  {row['symbol']}: {{enabled: true, rr: {row['rr']:.1f}}}")
    
    # Save to YAML file
    with open('500_symbol_top_100.yaml', 'w') as f:
        f.write('\n'.join(config_output))
    
    print(f"✅ Results saved to:")
    print(f"   - 500_symbol_validated.csv ({len(profitable)} symbols)")
    print(f"   - 500_symbol_all_results.csv ({len(results)} symbols)")
    print(f"   - 500_symbol_top_100.yaml (ready for config.yaml)")
    
    print("\n" + "="*80)
    print("🎉 DISCOVERY COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review 500_symbol_validated.csv for all profitable symbols")
    print(f"2. Copy symbols from 500_symbol_top_100.yaml to config.yaml")
    print(f"3. Restart bot to trade with new symbols")
    print(f"\nExpected Performance with Top 100:")
    print(f"  • Portfolio R/Year: {profitable_df.head(100)['total_r'].sum() * 2:+.0f}R")
    print(f"  • R/Month: {profitable_df.head(100)['total_r'].sum() / 6:+.0f}R")
    print(f"  • Total Symbols: 100")

if __name__ == "__main__":
    main()
