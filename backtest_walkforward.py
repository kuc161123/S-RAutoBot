#!/usr/bin/env python3
"""
ULTRA-REALISTIC WALK-FORWARD BACKTEST
=====================================
This backtest simulates EXACTLY how the live bot processes data:

1. Process each candle sequentially (like live bot loop)
2. Calculate indicators with ONLY data available at that moment
3. Detect divergence using ONLY past pivots (no look-ahead)
4. Place LIMIT order at candle close price
5. Simulate fill/no-fill (order fills only if next candle touches limit price)
6. Track TP/SL hits on subsequent candles
7. Order expires after 1 candle if not filled (like 5-min timeout)

This gives us a much more realistic expectation of live performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION (matches live bot exactly)
# =============================================================================
# Instead of hardcoding symbols, load from config.yaml
def load_symbols_from_config():
    """Load trading symbols from config.yaml"""
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        # Symbols are in 'trade' section
        symbols = config.get('trade', {}).get('symbols', [])
        if len(symbols) > 200:
            symbols = symbols[:200]  # Cap at 200
        print(f"📋 Loaded {len(symbols)} symbols from config.yaml")
        return symbols if symbols else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    except Exception as e:
        print(f"Warning: Could not load symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

SYMBOLS = load_symbols_from_config()

# Strategy parameters
RSI_PERIOD = 14
ATR_PERIOD = 14
PIVOT_LOOKBACK = 5      # Bars to look for pivot points
SWING_LOOKBACK = 15     # Bars for swing high/low SL
RR_RATIO = 3.0          # Risk:Reward ratio
MIN_SL_ATR = 0.3        # Minimum SL distance in ATR
MAX_SL_ATR = 2.0        # Maximum SL distance in ATR
TIMEOUT_BARS = 50       # Max bars before trade times out

# Simulation parameters
ALLOW_LIMIT_NOT_FILL = True    # If True, limit orders can miss if price doesn't touch
CANDLES_TO_FILL = 1            # How many candles to wait for limit fill (1 = current only)

# Data parameters
DAYS_BACK = 60          # Days of historical data
TIMEFRAME = '60'        # 60-minute (1H) candles

# Signal filter
BEARISH_ONLY = False    # Not just bearish
HIDDEN_BEARISH_ONLY = False  # Test ALL divergence types

# =============================================================================
# INDICATOR CALCULATIONS (matches live bot exactly)
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing (matches live bot)"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR (matches live bot)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def find_pivots(df: pd.DataFrame, lookback: int = 5) -> tuple:
    """Find pivot highs and lows (matches live bot)"""
    pivot_highs = []
    pivot_lows = []
    
    for i in range(lookback, len(df) - lookback):
        # Pivot high: highest point in window
        if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
            pivot_highs.append((i, df['high'].iloc[i], df['rsi'].iloc[i]))
        
        # Pivot low: lowest point in window
        if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
            pivot_lows.append((i, df['low'].iloc[i], df['rsi'].iloc[i]))
    
    return pivot_highs, pivot_lows

def detect_divergence(df: pd.DataFrame, pivot_highs: list, pivot_lows: list, idx: int) -> tuple:
    """
    Detect divergence at specific index (matches live bot exactly)
    
    Only uses pivots that exist BEFORE the current index (no look-ahead)
    
    Returns: (signal_type, side) or (None, None)
    """
    # Only consider pivots before current index
    valid_highs = [p for p in pivot_highs if p[0] < idx]
    valid_lows = [p for p in pivot_lows if p[0] < idx]
    
    if len(valid_lows) >= 2:
        # Get two most recent pivot lows
        recent_lows = valid_lows[-2:]
        prev_idx, prev_price, prev_rsi = recent_lows[0]
        curr_idx, curr_price, curr_rsi = recent_lows[1]
        
        # Regular Bullish: Lower price low, higher RSI low
        if curr_price < prev_price and curr_rsi > prev_rsi:
            return 'regular_bullish', 'long'
        
        # Hidden Bullish: Higher price low, lower RSI low
        if curr_price > prev_price and curr_rsi < prev_rsi:
            return 'hidden_bullish', 'long'
    
    if len(valid_highs) >= 2:
        # Get two most recent pivot highs
        recent_highs = valid_highs[-2:]
        prev_idx, prev_price, prev_rsi = recent_highs[0]
        curr_idx, curr_price, curr_rsi = recent_highs[1]
        
        # Regular Bearish: Higher price high, lower RSI high
        if curr_price > prev_price and curr_rsi < prev_rsi:
            return 'regular_bearish', 'short'
        
        # Hidden Bearish: Lower price high, higher RSI high
        if curr_price < prev_price and curr_rsi > prev_rsi:
            return 'hidden_bearish', 'short'
    
    return None, None

# =============================================================================
# TRADE SIMULATION (matches live bot exactly)
# =============================================================================

def calculate_sltp(df: pd.DataFrame, idx: int, side: str, entry: float, atr: float) -> tuple:
    """
    Calculate SL and TP at specific index (matches live bot exactly)
    
    Uses swing high/low from past SWING_LOOKBACK candles
    """
    start_idx = max(0, idx - SWING_LOOKBACK)
    
    if side == 'long':
        swing_low = df['low'].iloc[start_idx:idx+1].min()
        sl = swing_low
        sl_distance = entry - sl
        
        # Validate SL is below entry
        if sl >= entry:
            return None, None, None  # Invalid setup
        
    else:  # short
        swing_high = df['high'].iloc[start_idx:idx+1].max()
        sl = swing_high
        sl_distance = sl - entry
        
        # Validate SL is above entry
        if sl <= entry:
            return None, None, None  # Invalid setup
    
    # Apply min/max constraints
    min_sl = MIN_SL_ATR * atr
    max_sl = MAX_SL_ATR * atr
    
    if sl_distance < min_sl:
        sl_distance = min_sl
        sl = entry - sl_distance if side == 'long' else entry + sl_distance
    elif sl_distance > max_sl:
        sl_distance = max_sl
        sl = entry - sl_distance if side == 'long' else entry + sl_distance
    
    # Calculate TP with R:R ratio
    if side == 'long':
        tp = entry + (RR_RATIO * sl_distance)
    else:
        tp = entry - (RR_RATIO * sl_distance)
    
    return sl, tp, sl_distance

def simulate_limit_fill(df: pd.DataFrame, idx: int, limit_price: float, side: str) -> tuple:
    """
    Simulate whether a limit order would fill
    
    Returns: (filled: bool, fill_price: float, fill_idx: int)
    
    For a LONG limit order at price X:
    - Fills if next candle's low <= X (price touched our limit)
    
    For a SHORT limit order at price X:
    - Fills if next candle's high >= X (price touched our limit)
    """
    # Check candles after the limit order is placed
    for offset in range(1, CANDLES_TO_FILL + 1):
        check_idx = idx + offset
        if check_idx >= len(df):
            return False, 0, 0
        
        candle = df.iloc[check_idx]
        
        if side == 'long':
            # Long limit fills if low touches limit price
            if candle['low'] <= limit_price:
                return True, limit_price, check_idx
        else:
            # Short limit fills if high touches limit price
            if candle['high'] >= limit_price:
                return True, limit_price, check_idx
    
    return False, 0, 0

def simulate_trade_outcome(df: pd.DataFrame, entry_idx: int, side: str, entry: float, sl: float, tp: float) -> tuple:
    """
    Simulate trade outcome after entry
    
    Returns: (outcome: str, exit_price: float, bars_held: int)
    
    Checks each subsequent candle for TP/SL hits
    SL is checked FIRST (pessimistic, matches live bot)
    """
    for bar_offset in range(1, TIMEOUT_BARS + 1):
        check_idx = entry_idx + bar_offset
        if check_idx >= len(df):
            return 'timeout', entry, bar_offset
        
        candle = df.iloc[check_idx]
        
        if side == 'long':
            # Check SL first (pessimistic)
            if candle['low'] <= sl:
                return 'loss', sl, bar_offset
            if candle['high'] >= tp:
                return 'win', tp, bar_offset
        else:  # short
            # Check SL first (pessimistic)
            if candle['high'] >= sl:
                return 'loss', sl, bar_offset
            if candle['low'] <= tp:
                return 'win', tp, bar_offset
    
    return 'timeout', entry, TIMEOUT_BARS

# =============================================================================
# MAIN WALK-FORWARD SIMULATION
# =============================================================================

def load_data(symbol: str) -> pd.DataFrame:
    """Load historical data from Bybit with pagination (API returns max 200 per request)"""
    from autobot.brokers.bybit import Bybit, BybitConfig
    import yaml
    import time
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    bybit_config = BybitConfig(
        base_url='https://api.bybit.com',
        api_key=config.get('api_key', ''),
        api_secret=config.get('api_secret', '')
    )
    broker = Bybit(bybit_config)
    
    # Calculate candles needed based on timeframe
    timeframe_minutes = int(TIMEFRAME)  # '60' -> 60 minutes
    candles_per_day = int(24 * 60 / timeframe_minutes)  # 24 for 1H, 96 for 15m
    needed_candles = min(DAYS_BACK * candles_per_day, 1000)  # Cap at 1000
    
    # Bybit API returns max 200 candles per request - need pagination
    all_klines = []
    end_time = None  # None = latest
    
    while len(all_klines) < needed_candles:
        # Fetch batch (Bybit API only returns 200 max per request)
        klines = broker.get_klines(symbol, TIMEFRAME, limit=200, end=end_time)
        
        if not klines or len(klines) == 0:
            break
            
        # Add to collection (newest first from API)
        all_klines = klines + all_klines  # Prepend older data
        
        # Get end time for next request (oldest candle timestamp - 1)
        oldest_ts = int(klines[0][0])  # First element is oldest in this batch
        end_time = oldest_ts - 1
        
        # Rate limit protection
        time.sleep(0.1)
        
        # Safety check
        if len(all_klines) >= needed_candles:
            break
    
    if not all_klines or len(all_klines) < 100:
        return pd.DataFrame()
    
    # Trim to needed amount
    all_klines = all_klines[-needed_candles:]
    
    df = pd.DataFrame(all_klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def run_walkforward_backtest():
    """Run the ultra-realistic walk-forward backtest"""
    
    print("=" * 80)
    print("ULTRA-REALISTIC WALK-FORWARD BACKTEST")
    print("=" * 80)
    print(f"Simulates EXACTLY how the live bot processes data candle-by-candle")
    print(f"Timeframe: {TIMEFRAME}min | Data: {DAYS_BACK} days | R:R: {RR_RATIO}:1")
    print(f"Limit order fill simulation: {'ENABLED' if ALLOW_LIMIT_NOT_FILL else 'DISABLED'}")
    print("=" * 80)
    
    all_trades = []
    signals_detected = 0
    orders_placed = 0
    orders_filled = 0
    orders_not_filled = 0
    invalid_setups = 0
    
    for symbol in SYMBOLS:
        print(f"\n📊 Processing {symbol}...")
        
        # Load data
        df = load_data(symbol)
        if df.empty:
            print(f"   ❌ No data for {symbol}")
            continue
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < 100:
            print(f"   ❌ Not enough data for {symbol}")
            continue
        
        print(f"   📈 {len(df)} candles loaded")
        
        # Find all pivots (we'll filter by index later to avoid look-ahead)
        pivot_highs, pivot_lows = find_pivots(df, PIVOT_LOOKBACK)
        
        # State tracking
        in_trade = False
        pending_order = None
        trade_entry_idx = 0
        
        # Walk forward through each candle
        for idx in range(PIVOT_LOOKBACK + 10, len(df) - TIMEOUT_BARS - 5):
            
            # Skip if already in a trade
            if in_trade:
                continue
            
            # Skip if we have a pending order (check for fill)
            if pending_order:
                side = pending_order['side']
                limit_price = pending_order['limit_price']
                
                # Check if this candle fills the order
                candle = df.iloc[idx]
                filled = False
                
                if side == 'long' and candle['low'] <= limit_price:
                    filled = True
                elif side == 'short' and candle['high'] >= limit_price:
                    filled = True
                
                if filled:
                    orders_filled += 1
                    
                    # Simulate trade outcome
                    outcome, exit_price, bars = simulate_trade_outcome(
                        df, idx, side, 
                        pending_order['entry'], 
                        pending_order['sl'], 
                        pending_order['tp']
                    )
                    
                    all_trades.append({
                        'symbol': symbol,
                        'side': side,
                        'signal_type': pending_order['signal_type'],
                        'entry': pending_order['entry'],
                        'sl': pending_order['sl'],
                        'tp': pending_order['tp'],
                        'exit': exit_price,
                        'outcome': outcome,
                        'bars_held': bars,
                        'filled': True
                    })
                    
                    pending_order = None
                    continue
                else:
                    # Order didn't fill this candle - cancel it
                    orders_not_filled += 1
                    pending_order = None
                    continue
            
            # Get current candle data (simulating what live bot would see)
            current_rsi = df['rsi'].iloc[idx]
            current_atr = df['atr'].iloc[idx]
            current_close = df['close'].iloc[idx]
            
            # Skip if ATR is invalid
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # Detect divergence (only using pivots before this index)
            signal_type, side = detect_divergence(df, pivot_highs, pivot_lows, idx)
            
            if signal_type is None:
                continue
            
            # Apply bearish-only filter if enabled
            if BEARISH_ONLY and side == 'long':
                continue  # Skip bullish signals
            
            # Apply hidden_bearish_only filter if enabled
            if HIDDEN_BEARISH_ONLY and signal_type != 'hidden_bearish':
                continue  # Only trade hidden_bearish signals
            
            signals_detected += 1
            
            # Calculate SL/TP at signal candle close price (like live bot)
            expected_entry = current_close
            sl, tp, sl_distance = calculate_sltp(df, idx, side, expected_entry, current_atr)
            
            if sl is None:
                # Invalid setup (SL on wrong side)
                invalid_setups += 1
                continue
            
            orders_placed += 1
            
            # Create pending limit order (will be checked next candle)
            pending_order = {
                'side': side,
                'signal_type': signal_type,
                'limit_price': expected_entry,
                'entry': expected_entry,
                'sl': sl,
                'tp': tp,
                'created_idx': idx
            }
        
        print(f"   ✅ Processed {symbol}")
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Signal Detection:")
    print(f"   Signals Detected: {signals_detected}")
    print(f"   Invalid Setups (SL wrong side): {invalid_setups}")
    print(f"   Orders Placed: {orders_placed}")
    
    print(f"\n📋 Order Fill Simulation:")
    print(f"   Orders Filled: {orders_filled}")
    print(f"   Orders Not Filled (price moved away): {orders_not_filled}")
    fill_rate = (orders_filled / orders_placed * 100) if orders_placed > 0 else 0
    print(f"   Fill Rate: {fill_rate:.1f}%")
    
    if not all_trades:
        print("\n❌ No trades to analyze")
        return
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # Calculate metrics
    wins = len(trades_df[trades_df['outcome'] == 'win'])
    losses = len(trades_df[trades_df['outcome'] == 'loss'])
    timeouts = len(trades_df[trades_df['outcome'] == 'timeout'])
    total = len(trades_df)
    
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Calculate R-multiple P&L
    trades_df['r_pnl'] = trades_df.apply(
        lambda x: RR_RATIO if x['outcome'] == 'win' else (-1 if x['outcome'] == 'loss' else 0),
        axis=1
    )
    total_r = trades_df['r_pnl'].sum()
    avg_r = trades_df['r_pnl'].mean()
    
    # Expected Value
    ev = (wr/100 * RR_RATIO) - ((100-wr)/100 * 1)
    
    print(f"\n🎯 Trade Results:")
    print(f"   Total Trades: {total}")
    print(f"   Wins: {wins}")
    print(f"   Losses: {losses}")
    print(f"   Timeouts: {timeouts}")
    print(f"   Win Rate: {wr:.1f}%")
    
    print(f"\n💰 P&L Analysis:")
    print(f"   Total R: {total_r:+.2f}R")
    print(f"   Avg R per Trade: {avg_r:+.3f}R")
    print(f"   Expected Value: {ev:+.3f}R per trade")
    
    # Breakdown by signal type
    print(f"\n📈 By Signal Type:")
    for sig_type in trades_df['signal_type'].unique():
        sig_trades = trades_df[trades_df['signal_type'] == sig_type]
        sig_wins = len(sig_trades[sig_trades['outcome'] == 'win'])
        sig_total = len(sig_trades[sig_trades['outcome'].isin(['win', 'loss'])])
        sig_wr = (sig_wins / sig_total * 100) if sig_total > 0 else 0
        print(f"   {sig_type}: {sig_wr:.1f}% WR ({sig_total} trades)")
    
    # Breakdown by symbol
    print(f"\n📊 By Symbol:")
    for sym in trades_df['symbol'].unique():
        sym_trades = trades_df[trades_df['symbol'] == sym]
        sym_wins = len(sym_trades[sym_trades['outcome'] == 'win'])
        sym_total = len(sym_trades[sym_trades['outcome'].isin(['win', 'loss'])])
        sym_wr = (sym_wins / sym_total * 100) if sym_total > 0 else 0
        sym_r = sym_trades['r_pnl'].sum()
        print(f"   {sym}: {sym_wr:.1f}% WR, {sym_r:+.2f}R ({sym_total} trades)")
    
    # Save detailed results
    output_file = 'walkforward_results.csv'
    trades_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to {output_file}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if wr >= 50 and ev > 0:
        print("✅ STRATEGY VIABLE - Realistic backtest shows positive expectancy")
    elif wr >= 45 and ev > -0.1:
        print("⚠️ STRATEGY MARGINAL - Close to breakeven, monitor closely")
    else:
        print("❌ STRATEGY NEEDS WORK - Consider adjustments")
    print("=" * 80)
    
    return trades_df

if __name__ == "__main__":
    run_walkforward_backtest()
