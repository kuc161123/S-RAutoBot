#!/usr/bin/env python3
"""
MODERN FEE-AWARE BACKTEST - NO LOOK-AHEAD BIAS
===============================================

Key principles to avoid look-ahead bias:
1. Signal detected on bar[i] → Entry on bar[i+1].open (NEXT bar)
2. SL/TP checked using HIGH/LOW of each bar AFTER entry
3. Indicators use only data available at signal time
4. SL hit checked BEFORE TP hit (conservative - assumes worst case)
5. Proper exit order: SL first, then TP

Fee structure:
- Round-trip fees: 0.11% (0.055% maker + 0.055% taker)
- Minimum SL: 2.0% (matching bot config)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MATCH BOT EXACTLY
# ============================================================================

TIMEFRAME = '60'         # 1-hour candles (maximum ATR = lowest fee impact)
DATA_DAYS = 90           # 90 days of data for sufficient signals
NUM_SYMBOLS = 200        # Number of symbols (User Verification)

# Fees (Bybit standard)
MAKER_FEE = 0.00055      # 0.055%
TAKER_FEE = 0.00055      # 0.055%
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE  # 0.11%

# SL Settings (MATCH BOT - 2% minimum)
MIN_SL_PCT = 2.0         # Minimum 2.0% SL distance
MIN_TP_PCT = 4.0         # Minimum 4.0% TP (2:1 R:R)

# RSI Settings (match bot)
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
COOLDOWN_BARS = 10

BASE_URL = "https://api.bybit.com"

# ============================================================================
# TRAILING CONFIGURATIONS TO TEST
# ============================================================================

TRAILING_CONFIGS = [
    # Name, BE_Threshold, Trail_Distance, Max_TP
    # Simplified: BE_Lock = BE_Threshold - Trail_Distance
    ("Quick-Lock", 0.4, 0.15, 2.0),          # Current bot config
    ("Quick-Lock-3R", 0.4, 0.15, 3.0),       # Same but 3R target
    ("Conservative", 0.5, 0.2, 2.0),
    ("Conservative-3R", 0.5, 0.2, 3.0),
    ("Standard", 0.7, 0.3, 2.0),
    ("Standard-3R", 0.7, 0.3, 3.0),          # Original bot config
    ("Aggressive", 1.0, 0.4, 3.0),
    ("Very-Aggressive", 1.2, 0.5, 4.0),
    ("Tight-Trail", 0.3, 0.1, 2.0),
    ("Static-2R", 99.0, 0.0, 2.0),           # No trailing
    ("Static-3R", 99.0, 0.0, 3.0),           # No trailing
]

# ============================================================================
# DATA FETCHING
# ============================================================================

def get_symbols(limit):
    """Get top symbols by volume"""
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]


def fetch_klines(symbol, interval, days):
    """Fetch klines from Bybit"""
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 
                  'limit': 1000, 'end': current_end}
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
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df


# ============================================================================
# INDICATORS - NO LOOK-AHEAD
# ============================================================================

def calculate_rsi(close, period=14):
    """RSI - uses only historical data, no look-ahead"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    """ATR - uses only historical data"""
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_pivots(data, left=3, right=3):
    """
    Find pivot highs and lows.
    IMPORTANT: Pivot at bar[i] is only CONFIRMED at bar[i+right].
    We use right=3, so pivot is confirmed 3 bars later.
    """
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows


# ============================================================================
# DIVERGENCE DETECTION - NO LOOK-AHEAD
# ============================================================================

def detect_divergences_no_lookahead(df):
    """
    Detect RSI divergences WITHOUT look-ahead bias.
    
    Key: We can only know a pivot exists AFTER right=3 bars have passed.
    So at bar[i], we can only use pivots confirmed by bar[i-3].
    
    Signal at bar[i] → Entry at bar[i+1].open
    """
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    # Find pivots (remember: pivot at bar[j] is confirmed at bar[j+3])
    price_ph, price_pl = find_pivots(close, 3, 3)
    
    signals = []
    
    # Start from bar 30 to have enough history
    for i in range(30, n - 5):  # Leave 5 bars for trade resolution
        
        # === LOOK-AHEAD CHECK ===
        # At bar[i], we can only use pivots confirmed UP TO bar[i-3]
        # (because pivot at bar[j] requires right=3 future bars)
        confirmed_up_to = i - 3
        
        # Find most recent CONFIRMED pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: 
                    curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find most recent CONFIRMED pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: 
                    curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # === DIVERGENCE PATTERNS ===
        
        # Regular Bullish: Lower low + Higher RSI low
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({
                        'idx': i, 
                        'type': 'regular_bullish', 
                        'side': 'long', 
                        'swing': curr_pl,
                        'pivot_idx': curr_pli
                    })
                    continue
        
        # Regular Bearish: Higher high + Lower RSI high
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({
                        'idx': i, 
                        'type': 'regular_bearish', 
                        'side': 'short', 
                        'swing': curr_ph,
                        'pivot_idx': curr_phi
                    })
                    continue
        
        # Hidden Bullish: Higher low + Lower RSI low
        if curr_pl and prev_pl:
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                if rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({
                        'idx': i, 
                        'type': 'hidden_bullish', 
                        'side': 'long', 
                        'swing': curr_pl,
                        'pivot_idx': curr_pli
                    })
                    continue
        
        # Hidden Bearish: Lower high + Higher RSI high (YOUR MAIN STRATEGY)
        if curr_ph and prev_ph:
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                if rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({
                        'idx': i, 
                        'type': 'hidden_bearish', 
                        'side': 'short', 
                        'swing': curr_ph,
                        'pivot_idx': curr_phi
                    })
    
    return signals


# ============================================================================
# TRADE SIMULATION - NO LOOK-AHEAD
# ============================================================================

def simulate_trade(rows, signal_idx, side, swing_price, atr, config):
    """
    Simulate trade with EXACT bot logic and NO look-ahead.
    
    CRITICAL BIAS-FREE RULES:
    1. Entry on bar[signal_idx + 1].open (NEXT bar)
    2. SL/TP calculated using entry price
    3. Each bar after entry: check SL FIRST (using low for long, high for short)
    4. Then check TP (using high for long, low for short)
    5. Exit on the bar where condition is met
    """
    config_name, be_threshold, trail_distance, max_tp = config
    
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1:
        return None
    
    entry_bar = rows[entry_idx]
    entry = float(entry_bar.open)  # NEXT BAR'S OPEN - no look-ahead
    
    # Calculate SL distance with minimum
    sl_dist = abs(swing_price - entry)
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(sl_dist, min_sl_dist, atr)  # Use larger of swing/min/atr
    
    # Set initial SL and TP
    if side == 'long':
        initial_sl = entry - sl_dist
        tp = entry + (max_tp * sl_dist)
    else:
        initial_sl = entry + sl_dist
        tp = entry - (max_tp * sl_dist)
    
    current_sl = initial_sl
    max_favorable_r = 0
    be_triggered = False
    
    # Simulate bar by bar
    for bar_offset in range(1, min(100, len(rows) - entry_idx)):
        bar = rows[entry_idx + bar_offset]
        high = float(bar.high)
        low = float(bar.low)
        
        if side == 'long':
            # === CHECK SL FIRST (using LOW of bar) ===
            if low <= current_sl:
                exit_r = (current_sl - entry) / sl_dist
                return {
                    'exit_r': exit_r,
                    'max_r': max_favorable_r,
                    'bars': bar_offset,
                    'result': 'sl',
                    'entry': entry,
                    'sl_dist': sl_dist
                }
            
            # === CHECK TP (using HIGH of bar) ===
            if high >= tp:
                return {
                    'exit_r': max_tp,
                    'max_r': max_favorable_r,
                    'bars': bar_offset,
                    'result': 'tp',
                    'entry': entry,
                    'sl_dist': sl_dist
                }
            
            # Update max favorable R (using high)
            current_r = (high - entry) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                
                # Trail logic
                if max_favorable_r >= be_threshold:
                    if not be_triggered:
                        be_triggered = True
                    # Trail: lock in (max_r - trail_distance) R
                    new_sl = entry + (max_favorable_r - trail_distance) * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
        
        else:  # SHORT
            # === CHECK SL FIRST (using HIGH of bar) ===
            if high >= current_sl:
                exit_r = (entry - current_sl) / sl_dist
                return {
                    'exit_r': exit_r,
                    'max_r': max_favorable_r,
                    'bars': bar_offset,
                    'result': 'sl',
                    'entry': entry,
                    'sl_dist': sl_dist
                }
            
            # === CHECK TP (using LOW of bar) ===
            if low <= tp:
                return {
                    'exit_r': max_tp,
                    'max_r': max_favorable_r,
                    'bars': bar_offset,
                    'result': 'tp',
                    'entry': entry,
                    'sl_dist': sl_dist
                }
            
            # Update max favorable R (using low)
            current_r = (entry - low) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                
                if max_favorable_r >= be_threshold:
                    if not be_triggered:
                        be_triggered = True
                    new_sl = entry - (max_favorable_r - trail_distance) * sl_dist
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout - exit at last bar's close
    last_bar = rows[min(entry_idx + 99, len(rows) - 1)]
    if side == 'long':
        exit_r = (float(last_bar.close) - entry) / sl_dist
    else:
        exit_r = (entry - float(last_bar.close)) / sl_dist
    
    return {
        'exit_r': exit_r,
        'max_r': max_favorable_r,
        'bars': 100,
        'result': 'timeout',
        'entry': entry,
        'sl_dist': sl_dist
    }


def calculate_fee_r(entry, sl_dist):
    """Calculate fee as R-multiple"""
    sl_pct = sl_dist / entry
    return ROUND_TRIP_FEE / sl_pct


# ============================================================================
# MAIN
# ============================================================================

def run():
    print("=" * 80)
    print("🔬 MODERN FEE-AWARE BACKTEST (NO LOOK-AHEAD BIAS)")
    print("=" * 80)
    print(f"\nFees: {ROUND_TRIP_FEE*100:.3f}% round-trip")
    print(f"Min SL: {MIN_SL_PCT}% | Min TP: {MIN_TP_PCT}% (2:1 R:R)")
    print(f"Testing {len(TRAILING_CONFIGS)} configurations on {NUM_SYMBOLS} symbols")
    print(f"Data: {DATA_DAYS} days @ {TIMEFRAME}min timeframe\n")
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"📋 Testing {len(symbols)} symbols...\n")
    
    # Results per config
    results = {cfg[0]: [] for cfg in TRAILING_CONFIGS}
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            # Calculate indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['atr'] = calculate_atr(df, 14)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 100: continue
            
            # Detect divergences (bias-free)
            signals = detect_divergences_no_lookahead(df)
            rows = list(df.itertuples())
            
            last_trade_idx = -COOLDOWN_BARS
            
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN_BARS: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if not row.vol_ok or row.atr <= 0: continue
                
                # Test each config
                for config in TRAILING_CONFIGS:
                    trade = simulate_trade(rows, i, sig['side'], sig['swing'], 
                                          row.atr, config)
                    
                    if trade is None: continue
                    
                    # Calculate fee impact
                    fee_r = calculate_fee_r(trade['entry'], trade['sl_dist'])
                    actual_r = trade['exit_r'] - fee_r
                    
                    results[config[0]].append({
                        'symbol': sym,
                        'side': sig['side'],
                        'type': sig['type'],
                        'theo_r': trade['exit_r'],
                        'fee_r': fee_r,
                        'actual_r': actual_r,
                        'max_r': trade['max_r'],
                        'result': trade['result'],
                        'bars': trade['bars']
                    })
                
                last_trade_idx = i
                
        except Exception as e:
            continue
        
        if (idx + 1) % 10 == 0:
            n_trades = len(results[TRAILING_CONFIGS[0][0]])
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades per config: {n_trades}")
    
    # ============================================================================
    # ANALYSIS
    # ============================================================================
    
    print("\n" + "=" * 110)
    print("📊 RESULTS (Sorted by Actual R after fees)")
    print("=" * 110)
    
    summary = []
    for config in TRAILING_CONFIGS:
        cfg_name = config[0]
        trades = results[cfg_name]
        
        if not trades: continue
        
        n = len(trades)
        theo_wins = sum(1 for t in trades if t['theo_r'] > 0)
        actual_wins = sum(1 for t in trades if t['actual_r'] > 0)
        
        theo_wr = theo_wins / n * 100
        actual_wr = actual_wins / n * 100
        
        total_theo = sum(t['theo_r'] for t in trades)
        total_actual = sum(t['actual_r'] for t in trades)
        total_fees = sum(t['fee_r'] for t in trades)
        
        avg_theo = total_theo / n
        avg_actual = total_actual / n
        avg_fee = total_fees / n
        
        # Profit factor
        gross_profit = sum(t['actual_r'] for t in trades if t['actual_r'] > 0)
        gross_loss = abs(sum(t['actual_r'] for t in trades if t['actual_r'] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999
        
        # Exit breakdown
        tp_exits = sum(1 for t in trades if t['result'] == 'tp')
        sl_exits = sum(1 for t in trades if t['result'] == 'sl')
        
        summary.append({
            'Config': cfg_name,
            'N': n,
            'WR_Theo': f"{theo_wr:.1f}%",
            'WR_Actual': f"{actual_wr:.1f}%",
            'Total_R_Theo': f"{total_theo:+.0f}",
            'Total_R_Actual': f"{total_actual:+.0f}",
            'Fee_R': f"{total_fees:.0f}",
            'Avg_Fee_R': f"{avg_fee:.3f}",
            'EV/Trade': f"{avg_actual:+.3f}",
            'PF': f"{pf:.2f}",
            'TP%': f"{tp_exits/n*100:.0f}%",
            'Result': '✅' if total_actual > 0 else '❌',
            '_sort': total_actual
        })
    
    summary.sort(key=lambda x: x['_sort'], reverse=True)
    for s in summary:
        del s['_sort']
    
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Save
    df_summary.to_csv('backtest_modern_results.csv', index=False)
    print("\n✅ Saved to backtest_modern_results.csv")
    
    # Best config
    if summary:
        best = summary[0]
        print("\n" + "=" * 70)
        print("🏆 BEST CONFIGURATION (Bias-Free, After Fees)")
        print("=" * 70)
        for cfg in TRAILING_CONFIGS:
            if cfg[0] == best['Config']:
                print(f"Name: {cfg[0]}")
                print(f"BE Threshold: {cfg[1]}R")
                print(f"Trail Distance: {cfg[2]}R")
                print(f"Max TP: {cfg[3]}R")
                break
        print(f"\nTrades: {best['N']}")
        print(f"Win Rate (After Fees): {best['WR_Actual']}")
        print(f"Total R (After Fees): {best['Total_R_Actual']}")
        print(f"EV per Trade: {best['EV/Trade']}")
        print(f"Profit Factor: {best['PF']}")
        print(f"TP Hit Rate: {best['TP%']}")
        
        print("\n" + "=" * 70)
        print("📋 RECOMMENDATION")
        print("=" * 70)
        if float(best['EV/Trade'].replace('+', '')) > 0.1:
            print("✅ Strategy is PROFITABLE after fees")
            print(f"   Expected: {best['EV/Trade']} R per trade")
        else:
            print("⚠️ Edge is MARGINAL after fees")
            print("   Consider reducing trading frequency or increasing SL")


if __name__ == "__main__":
    run()
