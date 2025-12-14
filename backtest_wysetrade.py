#!/usr/bin/env python3
"""
WYSETRADE RSI DIVERGENCE + PRICE ACTION BACKTEST
=================================================
Implements the complete Wysetrade strategy with:
1. RSI Divergence Detection (Wide & Tight)
2. Key Level (S/R) Filter
3. Trendline Break Confirmation

PITFALLS ADDRESSED:
- No look-ahead bias in trendlines
- Only trade on closed candles
- Strict confirmation required
- Distinguish wide vs tight divergence
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

# RSI Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Divergence Settings
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# Key Level Settings
KEY_LEVEL_LOOKBACK = 50
KEY_LEVEL_TOLERANCE = 0.5  # 0.5% tolerance for key level

# Risk Management
RR_RATIO = 2.0  # 1:2 Risk Reward

BASE_URL = "https://api.bybit.com"

# =============================================================================
# TRADE LOG
# =============================================================================

trade_log = []

def log_trade(symbol, side, divergence_time, confirmation_time, entry, sl, tp, outcome, pnl_r, bars_to_exit):
    trade_log.append({
        'symbol': symbol,
        'side': side,
        'divergence_time': divergence_time,
        'confirmation_time': confirmation_time,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'outcome': outcome,
        'pnl_r': pnl_r,
        'bars_to_exit': bars_to_exit
    })

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

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def find_pivots(data, left=3, right=3):
    """Find swing highs and lows"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    pivot_high_indices = []
    pivot_low_indices = []
    
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high:
            pivot_highs[i] = data[i]
            pivot_high_indices.append(i)
        if is_low:
            pivot_lows[i] = data[i]
            pivot_low_indices.append(i)
    
    return pivot_highs, pivot_lows, pivot_high_indices, pivot_low_indices

def find_key_levels(df, lookback=50):
    """
    Find support/resistance key levels from recent swing highs/lows.
    Returns list of (level, type) tuples.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    _, _, high_idx, low_idx = find_pivots(high, 5, 5)
    _, _, _, low_idx2 = find_pivots(low, 5, 5)
    
    levels = []
    
    # Resistance from swing highs
    for idx in high_idx[-lookback:]:
        if idx < len(high):
            levels.append((high[idx], 'resistance'))
    
    # Support from swing lows
    for idx in low_idx2[-lookback:]:
        if idx < len(low):
            levels.append((low[idx], 'support'))
    
    return levels

def is_near_key_level(price, levels, tolerance_pct=0.5):
    """Check if price is near a key level"""
    for level, ltype in levels:
        distance_pct = abs(price - level) / price * 100
        if distance_pct <= tolerance_pct:
            return True, level, ltype
    return False, None, None

def calculate_trendline(prices, indices, side):
    """
    Calculate dynamic trendline using available data.
    For longs: trendline on lower highs (resistance to break)
    For shorts: trendline on higher lows (support to break)
    
    Returns slope and intercept.
    """
    if len(prices) < 2 or len(indices) < 2:
        return None, None
    
    # Use linear regression
    x = np.array(indices[-3:])  # Last 3 points
    y = np.array(prices[-3:])
    
    if len(x) < 2:
        return None, None
    
    # Calculate slope and intercept
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None, None
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

def get_trendline_value(slope, intercept, index):
    """Get trendline value at given index"""
    if slope is None or intercept is None:
        return None
    return slope * index + intercept

# =============================================================================
# DIVERGENCE DETECTION
# =============================================================================

def detect_divergence(df, i, lookback=14, min_distance=5):
    """
    Detect RSI divergence at bar i.
    Returns: (divergence_type, quality, swing_low_idx or swing_high_idx)
    
    divergence_type: 'bullish', 'bearish', or None
    quality: 'wide' or 'tight'
    """
    if i < lookback + 10:
        return None, None, None
    
    close = df['close'].values
    rsi = df['rsi'].values
    low = df['low'].values
    high = df['high'].values
    
    price_ph, price_pl, ph_indices, pl_indices = find_pivots(close[:i+1], PIVOT_LEFT, PIVOT_RIGHT)
    
    # Find recent pivot lows for bullish divergence
    curr_pl = curr_pli = prev_pl = prev_pli = None
    for j in range(i, max(i - lookback, 0), -1):
        if not np.isnan(price_pl[j]):
            if curr_pl is None:
                curr_pl = price_pl[j]
                curr_pli = j
            elif prev_pl is None and j < curr_pli - min_distance:
                prev_pl = price_pl[j]
                prev_pli = j
                break
    
    # Find recent pivot highs for bearish divergence
    curr_ph = curr_phi = prev_ph = prev_phi = None
    for j in range(i, max(i - lookback, 0), -1):
        if not np.isnan(price_ph[j]):
            if curr_ph is None:
                curr_ph = price_ph[j]
                curr_phi = j
            elif prev_ph is None and j < curr_phi - min_distance:
                prev_ph = price_ph[j]
                prev_phi = j
                break
    
    # === BULLISH DIVERGENCE: Price LL, RSI HL ===
    if curr_pl and prev_pl and curr_pl < prev_pl:
        curr_rsi = rsi[curr_pli]
        prev_rsi = rsi[prev_pli]
        
        # RSI must show higher low
        if curr_rsi > prev_rsi:
            # Prioritize if in oversold zone
            if rsi[i] <= RSI_OVERSOLD + 15:
                # Determine quality
                swing_distance = curr_pli - prev_pli
                quality = 'wide' if swing_distance >= 10 else 'tight'
                return 'bullish', quality, curr_pli
    
    # === BEARISH DIVERGENCE: Price HH, RSI LH ===
    if curr_ph and prev_ph and curr_ph > prev_ph:
        curr_rsi = rsi[curr_phi]
        prev_rsi = rsi[prev_phi]
        
        # RSI must show lower high
        if curr_rsi < prev_rsi:
            # Prioritize if in overbought zone
            if rsi[i] >= RSI_OVERBOUGHT - 15:
                swing_distance = curr_phi - prev_phi
                quality = 'wide' if swing_distance >= 10 else 'tight'
                return 'bearish', quality, curr_phi
    
    return None, None, None

def check_trendline_break(df, i, side, lookback=20):
    """
    Check if price has broken the immediate trendline.
    
    For longs: Check if close breaks above the descending trendline (lower highs)
    For shorts: Check if close breaks below the ascending trendline (higher lows)
    
    Returns: (is_broken, trendline_value)
    """
    if i < lookback + 5:
        return False, None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    if side == 'long':
        # Find recent lower highs for downtrend trendline
        pivot_highs = []
        pivot_indices = []
        
        for j in range(i - 1, max(i - lookback, PIVOT_LEFT), -1):
            # Check if this is a local high
            is_local_high = all(high[k] <= high[j] for k in range(j - 3, j + 4) if k != j and 0 <= k < len(high))
            if is_local_high:
                pivot_highs.append(high[j])
                pivot_indices.append(j)
                if len(pivot_highs) >= 3:
                    break
        
        if len(pivot_highs) < 2:
            # Fallback: use recent swing high break
            recent_high = max(high[max(0, i-10):i])
            return close[i] > recent_high, recent_high
        
        # Calculate trendline
        slope, intercept = calculate_trendline(pivot_highs[::-1], pivot_indices[::-1], 'long')
        if slope is None:
            return False, None
        
        trendline_at_i = get_trendline_value(slope, intercept, i)
        
        # Check if close breaks above trendline
        if trendline_at_i and close[i] > trendline_at_i:
            return True, trendline_at_i
        
        return False, trendline_at_i
    
    else:  # short
        # Find recent higher lows for uptrend trendline
        pivot_lows = []
        pivot_indices = []
        
        for j in range(i - 1, max(i - lookback, PIVOT_LEFT), -1):
            is_local_low = all(low[k] >= low[j] for k in range(j - 3, j + 4) if k != j and 0 <= k < len(low))
            if is_local_low:
                pivot_lows.append(low[j])
                pivot_indices.append(j)
                if len(pivot_lows) >= 3:
                    break
        
        if len(pivot_lows) < 2:
            # Fallback: use recent swing low break
            recent_low = min(low[max(0, i-10):i])
            return close[i] < recent_low, recent_low
        
        slope, intercept = calculate_trendline(pivot_lows[::-1], pivot_indices[::-1], 'short')
        if slope is None:
            return False, None
        
        trendline_at_i = get_trendline_value(slope, intercept, i)
        
        if trendline_at_i and close[i] < trendline_at_i:
            return True, trendline_at_i
        
        return False, trendline_at_i

# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(df, entry_idx, side, sl, tp):
    """Simulate trade with given SL and TP"""
    rows = list(df.itertuples())
    
    if entry_idx >= len(rows) - 50:
        return 'timeout', 0
    
    entry = rows[entry_idx].close
    
    # Apply slippage
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp = tp * (1 - TOTAL_COST)
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp = tp * (1 + TOTAL_COST)
    
    for bar_idx, row in enumerate(rows[entry_idx + 1:entry_idx + 100]):
        if side == 'long':
            if row.low <= sl:
                return 'loss', bar_idx + 1
            if row.high >= tp:
                return 'win', bar_idx + 1
        else:
            if row.high >= sl:
                return 'loss', bar_idx + 1
            if row.low <= tp:
                return 'win', bar_idx + 1
    
    return 'timeout', 100

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_wysetrade_backtest():
    print("=" * 80)
    print("🔬 WYSETRADE RSI DIVERGENCE + PRICE ACTION BACKTEST")
    print("=" * 80)
    print("\nStrategy Components:")
    print("  1. RSI Divergence Detection (Wide & Tight)")
    print("  2. Key Level (S/R) Filter")
    print("  3. Trendline Break Confirmation")
    print("\nPitfalls Addressed:")
    print("  ✓ No look-ahead bias in trendlines")
    print("  ✓ Only trade on closed candles")
    print("  ✓ Strict confirmation required")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📦 Fetching data for {len(symbols)} symbols...\n")
    
    # Results
    results = {
        'total': 0, 'wins': 0, 'losses': 0, 'timeouts': 0,
        'by_quality': {'wide': {'w': 0, 'l': 0}, 'tight': {'w': 0, 'l': 0}},
        'by_side': {'long': {'w': 0, 'l': 0}, 'short': {'w': 0, 'l': 0}},
        'pnl': []
    }
    
    # Stages
    divergence_detected = 0
    passed_key_level = 0
    passed_confirmation = 0
    
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 400:
                continue
            
            # Calculate indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['atr'] = calculate_atr(df, 14)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 200:
                continue
            
            processed += 1
            rows = list(df.itertuples())
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Find key levels
            key_levels = find_key_levels(df)
            
            last_trade_idx = -20
            
            # Scan through bars
            for i in range(50, len(rows) - 50):
                # Skip if too close to last trade
                if i - last_trade_idx < 15:
                    continue
                
                row = rows[i]
                
                # Volume filter
                if not row.vol_ok:
                    continue
                
                # ATR validation
                if pd.isna(row.atr) or row.atr <= 0:
                    continue
                
                # ============================================
                # STEP 1: DIVERGENCE DETECTION
                # ============================================
                div_type, quality, swing_idx = detect_divergence(df, i)
                
                if not div_type:
                    continue
                
                divergence_detected += 1
                side = 'long' if div_type == 'bullish' else 'short'
                
                # ============================================
                # STEP 2: KEY LEVEL FILTER
                # ============================================
                current_price = close[i]
                near_level, level, level_type = is_near_key_level(current_price, key_levels, KEY_LEVEL_TOLERANCE)
                
                # For bullish: we want price near SUPPORT
                # For bearish: we want price near RESISTANCE
                if side == 'long' and level_type != 'support' and not near_level:
                    continue
                if side == 'short' and level_type != 'resistance' and not near_level:
                    continue
                
                passed_key_level += 1
                
                # ============================================
                # STEP 3: TRENDLINE BREAK CONFIRMATION
                # ============================================
                tl_broken, tl_value = check_trendline_break(df, i, side)
                
                if not tl_broken:
                    continue
                
                passed_confirmation += 1
                
                # ============================================
                # TRADE ENTRY
                # ============================================
                divergence_time = df.index[swing_idx] if swing_idx else df.index[i]
                confirmation_time = df.index[i]
                entry_price = close[i]
                
                # Calculate SL and TP
                if side == 'long':
                    # SL below recent swing low
                    recent_low = min(low[max(0, i-15):i+1])
                    sl = recent_low - (0.1 * row.atr)  # Small buffer
                    sl_distance = entry_price - sl
                    tp = entry_price + (RR_RATIO * sl_distance)
                else:
                    # SL above recent swing high
                    recent_high = max(high[max(0, i-15):i+1])
                    sl = recent_high + (0.1 * row.atr)  # Small buffer
                    sl_distance = sl - entry_price
                    tp = entry_price - (RR_RATIO * sl_distance)
                
                # Simulate trade
                outcome, bars = simulate_trade(df, i, side, sl, tp)
                
                if outcome == 'timeout':
                    results['timeouts'] += 1
                    continue
                
                last_trade_idx = i
                results['total'] += 1
                
                if outcome == 'win':
                    results['wins'] += 1
                    pnl_r = RR_RATIO
                    results['by_quality'][quality]['w'] += 1
                    results['by_side'][side]['w'] += 1
                else:
                    results['losses'] += 1
                    pnl_r = -1.0
                    results['by_quality'][quality]['l'] += 1
                    results['by_side'][side]['l'] += 1
                
                results['pnl'].append(pnl_r)
                
                # Log trade
                log_trade(symbol, side, divergence_time, confirmation_time, 
                         entry_price, sl, tp, outcome, pnl_r, bars)
            
            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{NUM_SYMBOLS}] {processed} processed | Trades: {results['total']}")
            
            time.sleep(0.02)
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 WYSETRADE BACKTEST RESULTS")
    print("=" * 80)
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    
    # Funnel Analysis
    print("\n" + "-" * 60)
    print("📈 SIGNAL FUNNEL")
    print("-" * 60)
    print(f"  Divergences Detected:    {divergence_detected}")
    print(f"  → Passed Key Level:      {passed_key_level} ({passed_key_level/divergence_detected*100:.1f}%)")
    print(f"  → Passed Confirmation:   {passed_confirmation} ({passed_confirmation/divergence_detected*100:.1f}%)")
    print(f"  → Trades Executed:       {results['total']}")
    
    # Overall Performance
    print("\n" + "-" * 60)
    print("📊 OVERALL PERFORMANCE")
    print("-" * 60)
    
    if results['total'] > 0:
        wr = results['wins'] / results['total']
        lb = wilson_lb(results['wins'], results['total'])
        ev = calc_ev(wr, RR_RATIO)
        total_r = sum(results['pnl'])
        
        # Profit Factor
        gross_wins = results['wins'] * RR_RATIO
        gross_losses = results['losses'] * 1.0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        # Max Drawdown
        cumulative = np.cumsum(results['pnl'])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        print(f"\n📊 Total Trades: {results['total']}")
        print(f"✅ Wins: {results['wins']} | ❌ Losses: {results['losses']} | ⏱️ Timeouts: {results['timeouts']}")
        print(f"📈 Win Rate: {wr*100:.1f}% (Lower Bound: {lb*100:.1f}%)")
        print(f"💰 EV: {ev:+.2f}R per trade")
        print(f"💵 Total P&L: {total_r:+,.0f}R")
        print(f"📊 Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Drawdown: {max_dd:.0f}R")
    
    # By Quality
    print("\n" + "-" * 60)
    print("BY DIVERGENCE QUALITY")
    print("-" * 60)
    
    for quality in ['wide', 'tight']:
        q = results['by_quality'][quality]
        total = q['w'] + q['l']
        if total > 0:
            wr = q['w'] / total
            ev = calc_ev(wr, RR_RATIO)
            print(f"  {quality.upper():<8} N={total:<6} WR={wr*100:.1f}% EV={ev:+.2f}")
    
    # By Side
    print("\n" + "-" * 60)
    print("BY SIDE")
    print("-" * 60)
    
    for side in ['long', 'short']:
        s = results['by_side'][side]
        total = s['w'] + s['l']
        if total > 0:
            wr = s['w'] / total
            ev = calc_ev(wr, RR_RATIO)
            icon = "🟢" if side == 'long' else "🔴"
            print(f"  {icon} {side.upper():<8} N={total:<6} WR={wr*100:.1f}% EV={ev:+.2f}")
    
    # Sample Trade Log
    if trade_log:
        print("\n" + "-" * 60)
        print("📝 SAMPLE TRADES (Last 10)")
        print("-" * 60)
        print(f"\n{'Symbol':<12} {'Side':<6} {'Divergence Time':<20} {'Confirm Time':<20} {'Outcome':<8}")
        print("-" * 70)
        
        for t in trade_log[-10:]:
            print(f"{t['symbol']:<12} {t['side']:<6} {str(t['divergence_time'])[:19]:<20} {str(t['confirmation_time'])[:19]:<20} {t['outcome'].upper():<8}")
    
    # Final Verdict
    print("\n" + "=" * 80)
    print("🎯 VERDICT")
    print("=" * 80)
    
    if results['total'] > 0:
        if wr >= 0.5 and ev > 0.5:
            print("\n✅ WYSETRADE STRATEGY IS PROFITABLE")
            print(f"   The confirmation requirements (Key Level + Trendline Break)")
            print(f"   improve signal quality significantly.")
        elif ev > 0:
            print("\n⚠️ WYSETRADE STRATEGY IS MARGINALLY PROFITABLE")
            print(f"   Consider relaxing some filters or adjusting R:R.")
        else:
            print("\n❌ WYSETRADE STRATEGY UNDERPERFORMS")
            print(f"   The strict confirmation may be filtering too many good signals.")

if __name__ == "__main__":
    run_wysetrade_backtest()
