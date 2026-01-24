#!/usr/bin/env python3
"""
1-YEAR BLIND BACKTEST - Current 231 Symbol Configuration
==========================================================
This is a TRUE blind test:
- Uses EXACT config.yaml settings (per-symbol divergence, ATR, RR)
- Fetches 1 year of 1H data (365 days)
- Processes candle-by-candle with NO lookahead
- Simulates exact bot logic (BOS confirmation, 12 candle max wait)

Created: 2026-01-06
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD CURRENT CONFIG
# ============================================================================

def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

# ============================================================================
# DATA FETCHING (1 YEAR)
# ============================================================================

BASE_URL = "https://api.bybit.com"

def fetch_1yr_klines(symbol, days=365):
    """Fetch 1 year of 1H klines for a symbol"""
    all_klines = []
    end_ts = int(time.time() * 1000)
    start_ts = int((time.time() - days * 24 * 3600) * 1000)
    
    while end_ts > start_ts:
        params = {
            'category': 'linear', 
            'symbol': symbol, 
            'interval': '60', 
            'limit': 1000, 
            'end': end_ts
        }
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json()
            if data.get('retCode') != 0:
                break
            klines = data.get('result', {}).get('list', [])
            if not klines:
                break
            all_klines.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.02)  # Rate limit
        except Exception as e:
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
    df = df.iloc[::-1].reset_index(drop=True)  # Oldest first
    for c in ['open', 'high', 'low', 'close', 'vol']:
        df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df.set_index('ts', inplace=True)
    return df

# ============================================================================
# INDICATORS (EXACT BOT LOGIC)
# ============================================================================

def calculate_indicators(df):
    df = df.copy()
    
    # RSI (14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # ATR (14 period)
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMA 200 (trend filter)
    df['ema'] = df['close'].ewm(span=200, adjust=False).mean()
    
    return df.dropna()

def find_pivots(arr, left=3, right=3):
    """Find pivot highs and lows"""
    n = len(arr)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = arr[i-left:i+right+1]
        if arr[i] == max(window) and list(window).count(arr[i]) == 1:
            ph[i] = arr[i]
        if arr[i] == min(window) and list(window).count(arr[i]) == 1:
            pl[i] = arr[i]
    return ph, pl

# ============================================================================
# DIVERGENCE DETECTION (EXACT BOT LOGIC)
# ============================================================================

def detect_divergence(df, idx, div_type, close, rsi, ema, ph, pl):
    """Detect divergence at a specific candle index - EXACT bot logic"""
    if idx < 50:
        return False, None
    
    current_price = close[idx]
    current_ema = ema[idx]
    
    # Determine side and trend filter
    if div_type == 'REG_BULL':
        side = 'long'
        trend_filter = 'above_ema'
    elif div_type == 'REG_BEAR':
        side = 'short'
        trend_filter = 'below_ema'
    elif div_type == 'HID_BULL':
        side = 'long'
        trend_filter = 'above_ema'
    elif div_type == 'HID_BEAR':
        side = 'short'
        trend_filter = 'below_ema'
    else:
        return False, None
    
    # Apply trend filter
    if trend_filter == 'above_ema' and current_price <= current_ema:
        return False, None
    if trend_filter == 'below_ema' and current_price >= current_ema:
        return False, None
    
    # Find pivot points for divergence
    if side == 'long':
        pivots = []
        for j in range(idx - 4, max(0, idx - 50), -1):
            if not np.isnan(pl[j]):
                pivots.append((j, pl[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price_val, curr_rsi = pivots[0]
        prev_idx, prev_price_val, prev_rsi = pivots[1]
        
        # Signal must be fresh (within 10 candles)
        if (idx - curr_idx) > 10:
            return False, None
        
        # Check divergence conditions
        if div_type == 'REG_BULL':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
        elif div_type == 'HID_BULL':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
    else:  # short
        pivots = []
        for j in range(idx - 4, max(0, idx - 50), -1):
            if not np.isnan(ph[j]):
                pivots.append((j, ph[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price_val, curr_rsi = pivots[0]
        prev_idx, prev_price_val, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_type == 'REG_BEAR':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
        elif div_type == 'HID_BEAR':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
    
    return False, None

# ============================================================================
# BLIND CANDLE-BY-CANDLE SIMULATION
# ============================================================================

def simulate_blind(df, symbol, div_type, rr, atr_mult):
    """
    Simulate trading candle-by-candle with NO lookahead
    Returns list of trade results
    """
    trades = []
    if len(df) < 250:  # Need at least ~10 days of data after indicators
        return trades
    
    df = calculate_indicators(df)
    if df.empty or len(df) < 220:
        return trades
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    ph, pl = find_pivots(close, 3, 3)
    
    # State machine
    pending_signal = None
    pending_swing = None
    pending_wait = 0
    
    # Active trade
    in_trade = False
    entry_price = 0
    tp_price = 0
    sl_price = 0
    trade_side = None
    entry_time = None
    
    for idx in range(220, n):  # Start after EMA 200 has enough data
        current_price = close[idx]
        current_high = high[idx]
        current_low = low[idx]
        
        # Check active trade first
        if in_trade:
            if trade_side == 'long':
                if current_low <= sl_price:
                    trades.append({
                        'r': -1.0, 
                        'win': False, 
                        'entry': entry_time, 
                        'exit': df.index[idx],
                        'symbol': symbol
                    })
                    in_trade = False
                elif current_high >= tp_price:
                    trades.append({
                        'r': rr, 
                        'win': True, 
                        'entry': entry_time, 
                        'exit': df.index[idx],
                        'symbol': symbol
                    })
                    in_trade = False
            else:  # short
                if current_high >= sl_price:
                    trades.append({
                        'r': -1.0, 
                        'win': False, 
                        'entry': entry_time, 
                        'exit': df.index[idx],
                        'symbol': symbol
                    })
                    in_trade = False
                elif current_low <= tp_price:
                    trades.append({
                        'r': rr, 
                        'win': True, 
                        'entry': entry_time, 
                        'exit': df.index[idx],
                        'symbol': symbol
                    })
                    in_trade = False
            continue  # Don't look for new signals while in trade
        
        # Check pending signal for BOS confirmation
        if pending_signal is not None:
            bos = False
            if div_type in ['REG_BULL', 'HID_BULL'] and current_price > pending_swing:
                bos = True
                trade_side = 'long'
            if div_type in ['REG_BEAR', 'HID_BEAR'] and current_price < pending_swing:
                bos = True
                trade_side = 'short'
            
            if bos:
                # Enter trade at NEXT candle open (realistic execution)
                if idx + 1 < n:
                    entry_price = df.iloc[idx + 1]['open']
                    entry_atr = atr[idx + 1]
                    sl_dist = entry_atr * atr_mult
                    
                    if trade_side == 'long':
                        tp_price = entry_price + (sl_dist * rr)
                        sl_price = entry_price - sl_dist
                    else:
                        tp_price = entry_price - (sl_dist * rr)
                        sl_price = entry_price + sl_dist
                    
                    in_trade = True
                    entry_time = df.index[idx + 1]
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= 12:  # Max 12 candle wait
                    pending_signal = None
                    pending_wait = 0
        
        # Detect new divergence (only if not in trade and no pending)
        if not in_trade and pending_signal is None:
            found, swing = detect_divergence(df, idx, div_type, close, rsi, ema, ph, pl)
            if found:
                pending_signal = idx
                pending_swing = swing
                pending_wait = 0
    
    return trades

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("1-YEAR BLIND BACKTEST - Current 231 Symbol Configuration")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Fetching 365 days of 1H data for each symbol...")
    print("This will take 15-30 minutes. Please wait.")
    print("=" * 70)
    
    # Load config
    config = load_config()
    symbols_cfg = config['symbols']
    
    enabled_symbols = {k: v for k, v in symbols_cfg.items() if v.get('enabled', True)}
    print(f"\nEnabled symbols: {len(enabled_symbols)}")
    
    # Track all results
    all_trades = []
    symbol_results = []
    symbols_processed = 0
    symbols_with_data = 0
    
    start_time = time.time()
    
    for symbol, cfg in enabled_symbols.items():
        symbols_processed += 1
        
        # Progress update every 25 symbols
        if symbols_processed % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {symbols_processed}/{len(enabled_symbols)} symbols ({elapsed/60:.1f}m elapsed)...")
        
        # Fetch 1 year of data
        df = fetch_1yr_klines(symbol, days=365)
        if df.empty or len(df) < 500:  # Need substantial data
            continue
        
        symbols_with_data += 1
        
        # Simulate with this symbol's specific settings
        trades = simulate_blind(
            df, 
            symbol, 
            cfg['divergence'], 
            cfg['rr'], 
            cfg.get('atr_mult', 1.0)
        )
        
        if trades:
            all_trades.extend(trades)
            
            # Calculate symbol stats
            wins = sum(1 for t in trades if t['win'])
            r_sum = sum(t['r'] for t in trades)
            wr = (wins / len(trades)) * 100 if trades else 0
            
            symbol_results.append({
                'symbol': symbol,
                'divergence': cfg['divergence'],
                'rr': cfg['rr'],
                'atr_mult': cfg.get('atr_mult', 1.0),
                'trades': len(trades),
                'wins': wins,
                'losses': len(trades) - wins,
                'wr': round(wr, 1),
                'total_r': round(r_sum, 2)
            })
    
    elapsed_total = time.time() - start_time
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("1-YEAR BLIND BACKTEST RESULTS")
    print("=" * 70)
    
    if not all_trades:
        print("\n❌ No trades generated. Check symbol configs.")
        return
    
    # Overall stats
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t['win'])
    total_losses = total_trades - total_wins
    total_r = sum(t['r'] for t in all_trades)
    overall_wr = (total_wins / total_trades) * 100
    avg_r_per_trade = total_r / total_trades
    
    # Monthly breakdown
    monthly_r = defaultdict(float)
    for trade in all_trades:
        month_key = trade['exit'].strftime('%Y-%m')
        monthly_r[month_key] += trade['r']
    
    # Calculate drawdown
    equity_curve = []
    running_r = 0
    for trade in sorted(all_trades, key=lambda x: x['exit']):
        running_r += trade['r']
        equity_curve.append(running_r)
    
    peak = 0
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    
    # Profit factor
    gross_profit = sum(t['r'] for t in all_trades if t['win'])
    gross_loss = abs(sum(t['r'] for t in all_trades if not t['win']))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Best/Worst
    best_trade = max(all_trades, key=lambda x: x['r'])
    worst_trade = min(all_trades, key=lambda x: x['r'])
    
    # Profitable symbols
    profitable_symbols = [s for s in symbol_results if s['total_r'] > 0]
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Symbols with data: {symbols_with_data}/{len(enabled_symbols)}")
    print(f"   Symbols producing trades: {len(symbol_results)}")
    print(f"   Profitable symbols: {len(profitable_symbols)} ({len(profitable_symbols)/len(symbol_results)*100:.0f}%)")
    print()
    print(f"📈 TRADE STATISTICS")
    print(f"   Total Trades: {total_trades}")
    print(f"   Wins: {total_wins} | Losses: {total_losses}")
    print(f"   Win Rate: {overall_wr:.1f}%")
    print(f"   Avg R per Trade: {avg_r_per_trade:+.3f}R")
    print()
    print(f"💰 PROFIT METRICS")
    print(f"   Total R (1 Year): {total_r:+,.1f}R")
    print(f"   Monthly Avg R: {total_r/12:+,.1f}R")
    print(f"   Profit Factor: {profit_factor:.2f}x")
    print(f"   Max Drawdown: {max_dd:.1f}R")
    print()
    print(f"🏆 BEST/WORST TRADES")
    print(f"   Best: +{best_trade['r']:.1f}R ({best_trade['symbol']})")
    print(f"   Worst: {worst_trade['r']:.1f}R ({worst_trade['symbol']})")
    
    # Monthly breakdown
    print(f"\n📅 MONTHLY BREAKDOWN")
    for month in sorted(monthly_r.keys()):
        r = monthly_r[month]
        bar = "█" * int(abs(r) / 20) if r != 0 else ""
        sign = "+" if r >= 0 else ""
        print(f"   {month}: {sign}{r:,.1f}R {bar}")
    
    # Top 10 symbols
    print(f"\n🏆 TOP 10 SYMBOLS (by Total R)")
    top_10 = sorted(symbol_results, key=lambda x: x['total_r'], reverse=True)[:10]
    for i, s in enumerate(top_10, 1):
        print(f"   {i}. {s['symbol']}: {s['total_r']:+.1f}R ({s['trades']} trades, {s['wr']:.0f}% WR)")
    
    # Bottom 5 symbols
    print(f"\n❌ BOTTOM 5 SYMBOLS")
    bottom_5 = sorted(symbol_results, key=lambda x: x['total_r'])[:5]
    for s in bottom_5:
        print(f"   {s['symbol']}: {s['total_r']:+.1f}R ({s['trades']} trades, {s['wr']:.0f}% WR)")
    
    # Save results
    results_df = pd.DataFrame(symbol_results)
    results_df.to_csv('blind_test_1yr_results.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Time Period: 1 Year (365 days)")
    print(f"   Symbols Tested: {len(enabled_symbols)}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {overall_wr:.1f}%")
    print(f"   Total R: {total_r:+,.1f}R")
    print(f"   Profit Factor: {profit_factor:.2f}x")
    print(f"   Max Drawdown: {max_dd:.1f}R")
    print(f"\n   Processing Time: {elapsed_total/60:.1f} minutes")
    print(f"   Results saved to: blind_test_1yr_results.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()
