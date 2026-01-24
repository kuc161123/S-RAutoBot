#!/usr/bin/env python3
"""
BOT-EXACT BLIND VALIDATION TEST
================================
This test mirrors the EXACT behavior of the live bot:
1. Loads symbols from config.yaml with their specific settings (divergence, ATR, RR)
2. Processes ALL symbols simultaneously, candle-by-candle (time-synchronized)
3. ONE trade per symbol at a time (no overlapping trades)
4. Uses exact bot logic: divergence detection, BOS confirmation, 12-candle max wait
5. Tracks per-symbol and portfolio-level performance

Goal: Validate why live results differ from expectations.
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

# CONFIG
DAYS = 180  # 6 months
BASE_URL = "https://api.bybit.com"

# ============================================================================
# LOAD CURRENT BOT CONFIG
# ============================================================================

def load_config():
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    symbols_cfg = cfg.get('symbols', {})
    enabled = {k: v for k, v in symbols_cfg.items() if v.get('enabled', True)}
    
    print(f"Loaded {len(enabled)} enabled symbols from config.yaml")
    return enabled

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_klines(symbol, days=180):
    """Fetch historical 1H klines"""
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
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json()
            if data.get('retCode') != 0:
                break
            klines = data.get('result', {}).get('list', [])
            if not klines:
                break
            all_klines.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.015)
        except:
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
    df = df.iloc[::-1].reset_index(drop=True)
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
    
    return df

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
    """Detect divergence at a specific candle index"""
    if idx < 50:
        return False, None
    
    current_price = close[idx]
    current_ema = ema[idx]
    
    # Determine side and trend filter
    if div_type == 'REG_BULL':
        side = 'long'
        trend_check = current_price > current_ema
    elif div_type == 'REG_BEAR':
        side = 'short'
        trend_check = current_price < current_ema
    elif div_type == 'HID_BULL':
        side = 'long'
        trend_check = current_price > current_ema
    elif div_type == 'HID_BEAR':
        side = 'short'
        trend_check = current_price < current_ema
    else:
        return False, None
    
    # Apply trend filter
    if not trend_check:
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
                return True, swing
        elif div_type == 'HID_BULL':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
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
                return True, swing
        elif div_type == 'HID_BEAR':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                return True, swing
    
    return False, None

# ============================================================================
# SYMBOL STATE MACHINE (EXACT BOT BEHAVIOR)
# ============================================================================

class SymbolState:
    """Tracks the state of each symbol - EXACTLY like the bot"""
    def __init__(self, symbol, div_type, rr, atr_mult):
        self.symbol = symbol
        self.div_type = div_type
        self.rr = rr
        self.atr_mult = atr_mult
        
        # State
        self.in_trade = False
        self.pending_signal = False
        self.pending_swing = None
        self.pending_wait = 0
        self.side = None
        
        # Trade details
        self.entry_price = 0
        self.tp_price = 0
        self.sl_price = 0
        self.entry_time = None
        
        # Stats
        self.trades = []
        self.wins = 0
        self.losses = 0

# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================

def run_bot_exact_simulation(symbols_cfg):
    """
    Run a time-synchronized simulation across ALL symbols.
    Processes each hourly candle for ALL symbols before moving to the next hour.
    """
    print("\n" + "=" * 70)
    print("BOT-EXACT BLIND VALIDATION TEST")
    print("=" * 70)
    print(f"Period: Last {DAYS} days (6 months)")
    print(f"Symbols: {len(symbols_cfg)}")
    print("Behavior: One trade per symbol at a time, exact bot logic")
    print("=" * 70)
    
    # Step 1: Fetch data for all symbols
    print("\nStep 1: Fetching historical data...")
    symbol_data = {}
    symbol_states = {}
    
    for i, (symbol, cfg) in enumerate(symbols_cfg.items()):
        if (i + 1) % 25 == 0:
            print(f"  Fetched {i + 1}/{len(symbols_cfg)} symbols...")
        
        df = fetch_klines(symbol, DAYS)
        if len(df) < 250:
            continue
        
        df = calculate_indicators(df)
        if df.empty or len(df) < 220:
            continue
        
        # Pre-calculate pivots
        close = df['close'].values
        ph, pl = find_pivots(close, 3, 3)
        
        symbol_data[symbol] = {
            'df': df,
            'close': close,
            'high': df['high'].values,
            'low': df['low'].values,
            'open': df['open'].values,
            'rsi': df['rsi'].values,
            'ema': df['ema'].values,
            'atr': df['atr'].values,
            'ph': ph,
            'pl': pl,
            'index': df.index
        }
        
        symbol_states[symbol] = SymbolState(
            symbol,
            cfg['divergence'],
            cfg['rr'],
            cfg.get('atr_mult', 1.0)
        )
    
    print(f"\nStep 2: Running simulation on {len(symbol_data)} symbols...")
    
    # Step 2: Get common time range
    all_times = set()
    for data in symbol_data.values():
        all_times.update(data['index'].tolist())
    common_times = sorted(all_times)
    
    # Start from index 220 to ensure enough data for indicators
    start_idx = 220
    
    # Track portfolio
    all_trades = []
    
    # Step 3: Process candle by candle (time-synchronized)
    for t_idx, timestamp in enumerate(common_times[start_idx:], start=start_idx):
        
        # Process each symbol at this timestamp
        for symbol, data in symbol_data.items():
            state = symbol_states[symbol]
            
            # Find index for this timestamp in this symbol's data
            try:
                idx = data['index'].get_loc(timestamp)
            except KeyError:
                continue  # This symbol doesn't have data for this candle
            
            if idx < 50:
                continue
            
            current_close = data['close'][idx]
            current_high = data['high'][idx]
            current_low = data['low'][idx]
            current_atr = data['atr'][idx]
            
            # ==========================================
            # CHECK ACTIVE TRADE FIRST
            # ==========================================
            if state.in_trade:
                if state.side == 'long':
                    if current_low <= state.sl_price:
                        # Stop loss hit
                        state.trades.append({
                            'r': -1.0,
                            'win': False,
                            'entry': state.entry_time,
                            'exit': timestamp,
                            'symbol': symbol
                        })
                        state.losses += 1
                        state.in_trade = False
                    elif current_high >= state.tp_price:
                        # Take profit hit
                        state.trades.append({
                            'r': state.rr,
                            'win': True,
                            'entry': state.entry_time,
                            'exit': timestamp,
                            'symbol': symbol
                        })
                        state.wins += 1
                        state.in_trade = False
                else:  # short
                    if current_high >= state.sl_price:
                        # Stop loss hit
                        state.trades.append({
                            'r': -1.0,
                            'win': False,
                            'entry': state.entry_time,
                            'exit': timestamp,
                            'symbol': symbol
                        })
                        state.losses += 1
                        state.in_trade = False
                    elif current_low <= state.tp_price:
                        # Take profit hit
                        state.trades.append({
                            'r': state.rr,
                            'win': True,
                            'entry': state.entry_time,
                            'exit': timestamp,
                            'symbol': symbol
                        })
                        state.wins += 1
                        state.in_trade = False
                
                # If still in trade, skip signal detection
                if state.in_trade:
                    continue
            
            # ==========================================
            # CHECK PENDING SIGNAL FOR BOS
            # ==========================================
            if state.pending_signal:
                bos = False
                if state.side == 'long' and current_close > state.pending_swing:
                    bos = True
                elif state.side == 'short' and current_close < state.pending_swing:
                    bos = True
                
                if bos:
                    # Enter trade at NEXT candle open
                    if idx + 1 < len(data['close']):
                        entry_price = data['open'][idx + 1]
                        entry_atr = data['atr'][idx + 1]
                        sl_dist = entry_atr * state.atr_mult
                        
                        if state.side == 'long':
                            state.tp_price = entry_price + (sl_dist * state.rr)
                            state.sl_price = entry_price - sl_dist
                        else:
                            state.tp_price = entry_price - (sl_dist * state.rr)
                            state.sl_price = entry_price + sl_dist
                        
                        state.in_trade = True
                        state.entry_price = entry_price
                        state.entry_time = data['index'][idx + 1]
                    
                    state.pending_signal = False
                    state.pending_wait = 0
                else:
                    state.pending_wait += 1
                    if state.pending_wait >= 12:  # Max 12 candle wait
                        state.pending_signal = False
                        state.pending_wait = 0
            
            # ==========================================
            # DETECT NEW DIVERGENCE (only if not in trade and no pending)
            # ==========================================
            if not state.in_trade and not state.pending_signal:
                found, swing = detect_divergence(
                    data['df'], idx, state.div_type,
                    data['close'], data['rsi'], data['ema'],
                    data['ph'], data['pl']
                )
                if found:
                    state.pending_signal = True
                    state.pending_swing = swing
                    state.pending_wait = 0
                    state.side = 'long' if state.div_type in ['REG_BULL', 'HID_BULL'] else 'short'
    
    # ==========================================
    # COLLECT RESULTS
    # ==========================================
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Collect all trades
    for symbol, state in symbol_states.items():
        all_trades.extend(state.trades)
    
    if not all_trades:
        print("No trades generated!")
        return
    
    # Overall stats
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t['win'])
    total_losses = total_trades - total_wins
    total_r = sum(t['r'] for t in all_trades)
    win_rate = (total_wins / total_trades) * 100
    avg_r = total_r / total_trades
    
    # Profit factor
    gross_profit = sum(t['r'] for t in all_trades if t['win'])
    gross_loss = abs(sum(t['r'] for t in all_trades if not t['win']))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Monthly breakdown
    monthly_r = defaultdict(float)
    monthly_trades = defaultdict(int)
    for trade in all_trades:
        month_key = trade['exit'].strftime('%Y-%m')
        monthly_r[month_key] += trade['r']
        monthly_trades[month_key] += 1
    
    # Max drawdown
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
    
    # Per-symbol performance
    symbol_results = []
    for symbol, state in symbol_states.items():
        if state.trades:
            s_wins = sum(1 for t in state.trades if t['win'])
            s_r = sum(t['r'] for t in state.trades)
            s_wr = (s_wins / len(state.trades)) * 100
            symbol_results.append({
                'symbol': symbol,
                'div': state.div_type,
                'rr': state.rr,
                'atr': state.atr_mult,
                'trades': len(state.trades),
                'wins': s_wins,
                'wr': round(s_wr, 1),
                'total_r': round(s_r, 1)
            })
    
    profitable_symbols = [s for s in symbol_results if s['total_r'] > 0]
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Symbols with trades: {len(symbol_results)}/{len(symbol_states)}")
    print(f"   Profitable symbols: {len(profitable_symbols)} ({len(profitable_symbols)/len(symbol_results)*100:.0f}%)")
    print()
    print(f"📈 TRADE STATISTICS")
    print(f"   Total Trades: {total_trades}")
    print(f"   Wins: {total_wins} | Losses: {total_losses}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Avg R per Trade: {avg_r:+.3f}R")
    print()
    print(f"💰 PROFIT METRICS")
    print(f"   Total R (6 Months): {total_r:+,.1f}R")
    print(f"   Monthly Avg R: {total_r/6:+,.1f}R")
    print(f"   Profit Factor: {pf:.2f}x")
    print(f"   Max Drawdown: {max_dd:.1f}R")
    print()
    
    # Monthly breakdown
    print(f"📅 MONTHLY BREAKDOWN")
    for month in sorted(monthly_r.keys()):
        r = monthly_r[month]
        trades = monthly_trades[month]
        bar = "█" * int(abs(r) / 20) if r != 0 else ""
        sign = "+" if r >= 0 else ""
        print(f"   {month}: {sign}{r:,.1f}R ({trades} trades) {bar}")
    
    # Top symbols
    print(f"\n🏆 TOP 10 SYMBOLS")
    top_10 = sorted(symbol_results, key=lambda x: x['total_r'], reverse=True)[:10]
    for i, s in enumerate(top_10, 1):
        print(f"   {i}. {s['symbol']}: {s['total_r']:+.1f}R ({s['trades']} trades, {s['wr']:.0f}% WR)")
    
    # Bottom symbols
    print(f"\n❌ BOTTOM 5 SYMBOLS")
    bottom_5 = sorted(symbol_results, key=lambda x: x['total_r'])[:5]
    for s in bottom_5:
        print(f"   {s['symbol']}: {s['total_r']:+.1f}R ({s['trades']} trades, {s['wr']:.0f}% WR)")
    
    # Save results
    results_df = pd.DataFrame(symbol_results)
    results_df.to_csv('bot_exact_validation_results.csv', index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   This test EXACTLY mirrors your live bot behavior.")
    print(f"   One trade per symbol at a time. Exact config settings.")
    print(f"   Period: 6 Months ({DAYS} days)")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Total R: {total_r:+,.1f}R")
    print(f"   Results saved to: bot_exact_validation_results.csv")
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    symbols_cfg = load_config()
    run_bot_exact_simulation(symbols_cfg)
