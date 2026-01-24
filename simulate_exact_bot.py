#!/usr/bin/env python3
"""
simulate_exact_bot.py - 100% Accurate Bot Simulation
=====================================================
Replicates the EXACT logic of the live bot for historical simulation.

Key Features Replicated:
1. Divergence Detection (from divergence_detector.py)
   - find_pivots(close, left=3, right=3)
   - detect_divergences with per-symbol type filter
   
2. BOS Confirmation (from bot.py)
   - check_bos: Price breaks swing level
   - is_trend_aligned: EMA 200 check
   - Max wait: 12 candles
   - Signal expiration after 12 candles if no BOS
   
3. Entry Logic
   - Entry on NEXT CANDLE OPEN after BOS confirmation
   - SL = Entry ± (ATR × atr_mult)
   - TP = Entry ± (SL_Distance × rr)
   
4. Trade Execution (5M Precision)
   - Use 5-minute data to check SL/TP hits
   - SL checked BEFORE TP (worst-case assumption)
"""

import requests
import pandas as pd
import numpy as np
import yaml
import time
import concurrent.futures
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ============ CONFIGURATION ============
INITIAL_CAPITAL = 700.0
RISK_PCT = 0.001  # 0.1% risk per trade

# Simulation Period - JANUARY 2026 ONLY
START_DATE = "2026-01-01 00:00:00"
END_DATE = "2026-01-17 23:59:59"

# Divergence Detection Constants (from divergence_detector.py)
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
EMA_PERIOD = 200

# BOS Constants (from bot.py)
MAX_WAIT_CANDLES = 12

# Month definitions for reporting - JANUARY 2026 ONLY
MONTHS = [
    ("JAN 2026", "2026-01-01 00:00:00", "2026-01-17 23:59:59"),
]

# ============ DATA CLASSES ============
@dataclass
class DivergenceSignal:
    """Represents a divergence signal"""
    symbol: str
    side: str  # 'long' or 'short'
    divergence_code: str  # 'REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR'
    divergence_idx: int
    swing_level: float
    rsi_value: float
    price: float
    timestamp: pd.Timestamp
    pivot_timestamp: pd.Timestamp
    daily_trend_aligned: bool

@dataclass
class PendingSignal:
    """Tracks a divergence waiting for BOS confirmation"""
    signal: DivergenceSignal
    detected_at_idx: int
    candles_waited: int = 0
    max_wait_candles: int = MAX_WAIT_CANDLES
    
    def is_expired(self) -> bool:
        return self.candles_waited >= self.max_wait_candles
    
    def increment_wait(self):
        self.candles_waited += 1

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str
    entry_price: float
    entry_time: int  # timestamp ms
    exit_time: int
    r_result: float  # -1.0 for loss, rr for win, 0 for running

# ============ DATA FETCHING ============
def fetch_klines(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch historical klines from Bybit"""
    end_req = int(datetime(2026, 1, 18, tzinfo=timezone.utc).timestamp() * 1000)
    
    # OPTIMIZED: Tighter date ranges for faster fetch - JANUARY 2026 ONLY
    if interval == '60':
        # 1H: Need EMA 200 buffer (200 hours = ~8 days before Jan 1)
        start_req = int(datetime(2025, 12, 20, tzinfo=timezone.utc).timestamp() * 1000)
    else:
        # 5M: Only need from Jan 1 for execution
        start_req = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    
    current_end = end_req
    all_candles = []
    
    while current_end > start_req:
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&end={current_end}&limit=1000"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data.get('retCode') != 0:
                return None
            candles = data.get('result', {}).get('list', [])
            if not candles:
                break
            all_candles.extend(candles)
            oldest = int(candles[-1][0])
            if oldest >= current_end:
                break
            current_end = oldest
            time.sleep(0.05)
        except Exception as e:
            print(f"[{symbol}] Fetch error: {e}")
            return None
    
    if not all_candles:
        return None
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
    df['start'] = pd.to_datetime(df['start'], unit='ms', utc=True)
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    df.set_index('start', inplace=True)
    
    return df

# ============ INDICATOR CALCULATIONS ============
def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR"""
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame with all required indicators"""
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    df['daily_ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

# ============ PIVOT DETECTION (EXACT COPY) ============
def find_pivots(data: np.ndarray, left: int = PIVOT_LEFT, right: int = PIVOT_RIGHT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find pivot highs and lows in price data.
    A pivot at index i is confirmed at i+right.
    This matches the validated backtest logic - NO LOOK-AHEAD BIAS.
    """
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        # Check if local high
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high:
            pivot_highs[i] = data[i]
        
        # Check if local low
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_low:
            pivot_lows[i] = data[i]
    
    return pivot_highs, pivot_lows

# ============ DIVERGENCE DETECTION (EXACT COPY) ============
def detect_divergences(df: pd.DataFrame, symbol: str, allowed_type: str = None) -> List[DivergenceSignal]:
    """
    Detect all 4 RSI divergence types with Daily Trend filter.
    EXACT copy of divergence_detector.py logic.
    """
    if len(df) < 100:
        return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    daily_ema = df['daily_ema'].values
    
    n = len(df)
    price_high_pivots, price_low_pivots = find_pivots(close, PIVOT_LEFT, PIVOT_RIGHT)
    
    signals = []
    
    # Scan for divergences
    for i in range(30, n - PIVOT_RIGHT):
        current_price = close[i]
        current_ema = daily_ema[i]
        
        # ========== BULLISH DIVERGENCES (LONG) ==========
        curr_pl = curr_pli = prev_pl = prev_pli = None
        
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_low_pivots[j]):
                if curr_pl is None:
                    curr_pl, curr_pli = price_low_pivots[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_low_pivots[j], j
                    break
        
        if curr_pl is not None and prev_pl is not None:
            swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
            trend_aligned = current_price > current_ema
            
            # REG_BULL: Price LL (curr < prev), RSI HL (curr > prev)
            if (allowed_type is None or allowed_type == 'REG_BULL'):
                if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                    if allowed_type == 'REG_BULL' or allowed_type is None:
                        signals.append(DivergenceSignal(
                            symbol=symbol, side='long', divergence_code='REG_BULL',
                            divergence_idx=i, swing_level=swing_high, rsi_value=rsi[i],
                            price=current_price, timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_pli], daily_trend_aligned=trend_aligned
                        ))
            
            # HID_BULL: Price HL (curr > prev), RSI LL (curr < prev)
            if (allowed_type is None or allowed_type == 'HID_BULL'):
                if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                    if allowed_type == 'HID_BULL' or allowed_type is None:
                        signals.append(DivergenceSignal(
                            symbol=symbol, side='long', divergence_code='HID_BULL',
                            divergence_idx=i, swing_level=swing_high, rsi_value=rsi[i],
                            price=current_price, timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_pli], daily_trend_aligned=trend_aligned
                        ))
        
        # ========== BEARISH DIVERGENCES (SHORT) ==========
        curr_ph = curr_phi = prev_ph = prev_phi = None
        
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_high_pivots[j]):
                if curr_ph is None:
                    curr_ph, curr_phi = price_high_pivots[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_high_pivots[j], j
                    break
        
        if curr_ph is not None and prev_ph is not None:
            swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
            trend_aligned = current_price < current_ema
            
            # REG_BEAR: Price HH (curr > prev), RSI LH (curr < prev)
            if (allowed_type is None or allowed_type == 'REG_BEAR'):
                if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                    if allowed_type == 'REG_BEAR' or allowed_type is None:
                        signals.append(DivergenceSignal(
                            symbol=symbol, side='short', divergence_code='REG_BEAR',
                            divergence_idx=i, swing_level=swing_low, rsi_value=rsi[i],
                            price=current_price, timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_phi], daily_trend_aligned=trend_aligned
                        ))
            
            # HID_BEAR: Price LH (curr < prev), RSI HH (curr > prev)
            if (allowed_type is None or allowed_type == 'HID_BEAR'):
                if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                    if allowed_type == 'HID_BEAR' or allowed_type is None:
                        signals.append(DivergenceSignal(
                            symbol=symbol, side='short', divergence_code='HID_BEAR',
                            divergence_idx=i, swing_level=swing_low, rsi_value=rsi[i],
                            price=current_price, timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_phi], daily_trend_aligned=trend_aligned
                        ))
    
    return signals

# ============ BOS CHECKS (EXACT COPY) ============
def check_bos(df: pd.DataFrame, signal: DivergenceSignal, current_idx: int) -> bool:
    """Check if Break of Structure (BOS) has occurred."""
    if current_idx >= len(df):
        return False
    current_close = df.iloc[current_idx]['close']
    if signal.side == 'long':
        return current_close > signal.swing_level
    else:
        return current_close < signal.swing_level

def is_trend_aligned(df: pd.DataFrame, signal: DivergenceSignal, current_idx: int) -> bool:
    """Check if current price is still aligned with daily trend."""
    if current_idx >= len(df):
        return False
    current_price = df.iloc[current_idx]['close']
    current_ema = df.iloc[current_idx]['daily_ema']
    if signal.side == 'long':
        return current_price > current_ema
    else:
        return current_price < current_ema

# ============ MAIN SIMULATION ============
def process_symbol(args) -> List[Trade]:
    """Process a single symbol through the exact bot logic"""
    sym, cfg = args
    
    if not cfg.get('enabled', True):
        return []
    
    allowed_div = cfg.get('divergence_type')
    rr = cfg.get('rr', 5.0)
    atr_mult = cfg.get('atr_mult', 1.0)
    
    # 1. Fetch 1H Data
    df_1h = fetch_klines(sym, '60')
    if df_1h is None or len(df_1h) < 250:
        return []
    df_1h = prepare_dataframe(df_1h)
    
    # 2. Fetch 5M Data
    df_5m = fetch_klines(sym, '5')
    if df_5m is None:
        return []
    
    # 3. Simulation State
    pending_signals: List[PendingSignal] = []
    completed_trades: List[Trade] = []
    last_exit_time = 0
    seen_pivots = set()  # Deduplication by pivot timestamp
    
    # Parse date range
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    # 4. Candle-by-Candle Simulation (like live bot loop)
    for current_idx in range(100, len(df_1h)):
        current_candle = df_1h.iloc[current_idx]
        current_ts = int(current_candle.name.timestamp() * 1000)
        
        # Skip if outside simulation range
        if current_ts < start_ts or current_ts > end_ts:
            continue
        
        # Skip if currently in a trade
        if current_ts < last_exit_time:
            continue
        
        # ====== STEP 1: Detect new divergences ======
        # Create a slice of df up to current_idx (no look-ahead)
        df_slice = df_1h.iloc[:current_idx + 1]
        new_signals = detect_divergences(df_slice, sym, allowed_div)
        
        # Filter: Only signals on the LAST candle of the slice (the current candle)
        # The divergence_idx is relative to the slice, so we check if it's at len(df_slice) - PIVOT_RIGHT - 1
        # Actually, the divergence is detected at index i, which is the confirmation point.
        # After slicing, the "last valid signal index" is len(df_slice) - 1 - PIVOT_RIGHT
        last_valid_idx = len(df_slice) - 1 - PIVOT_RIGHT
        
        for sig in new_signals:
            # Accept signals detected on recent candles (within PIVOT_RIGHT of current)
            if sig.divergence_idx >= last_valid_idx:
                # Deduplication: Check pivot timestamp
                pivot_key = f"{sym}_{sig.side}_{sig.divergence_code}_{sig.pivot_timestamp}"
                if pivot_key in seen_pivots:
                    continue
                seen_pivots.add(pivot_key)
                
                # Only one pending signal at a time
                if len(pending_signals) == 0:
                    pending_signals.append(PendingSignal(signal=sig, detected_at_idx=current_idx))
        
        # ====== STEP 2: Check pending signals for BOS ======
        signals_to_remove = []
        for pending in pending_signals:
            # Check if expired
            if pending.is_expired():
                signals_to_remove.append(pending)
                continue
            
            # Check if still trend-aligned
            if not is_trend_aligned(df_slice, pending.signal, current_idx):
                signals_to_remove.append(pending)
                continue
            
            # Check for BOS
            if check_bos(df_slice, pending.signal, current_idx):
                # BOS Confirmed! Execute trade on NEXT candle open
                if current_idx + 1 < len(df_1h):
                    entry_candle = df_1h.iloc[current_idx + 1]
                    entry_price = entry_candle['open']
                    entry_ts = int(entry_candle.name.timestamp() * 1000)
                    atr = entry_candle['atr']
                    
                    # Calculate SL/TP
                    sl_dist = atr * atr_mult
                    if pending.signal.side == 'long':
                        tp = entry_price + (sl_dist * rr)
                        sl = entry_price - sl_dist
                    else:
                        tp = entry_price - (sl_dist * rr)
                        sl = entry_price + sl_dist
                    
                    # Execute with 5M precision
                    outcome = -1.0
                    exit_ts = entry_ts + (24 * 3600000)  # Default timeout
                    
                    sub_5m = df_5m[df_5m.index >= entry_candle.name]
                    
                    for row in sub_5m.itertuples():
                        curr_ts = int(row.Index.timestamp() * 1000)
                        if pending.signal.side == 'long':
                            if row.low <= sl:
                                outcome = -1.0
                                exit_ts = curr_ts
                                break
                            if row.high >= tp:
                                outcome = rr
                                exit_ts = curr_ts
                                break
                        else:
                            if row.high >= sl:
                                outcome = -1.0
                                exit_ts = curr_ts
                                break
                            if row.low <= tp:
                                outcome = rr
                                exit_ts = curr_ts
                                break
                    
                    completed_trades.append(Trade(
                        symbol=sym,
                        side=pending.signal.side,
                        entry_price=entry_price,
                        entry_time=entry_ts,
                        exit_time=exit_ts,
                        r_result=outcome
                    ))
                    
                    last_exit_time = exit_ts
                
                signals_to_remove.append(pending)
            else:
                # Increment wait counter
                pending.increment_wait()
        
        # Remove processed signals
        for sig in signals_to_remove:
            if sig in pending_signals:
                pending_signals.remove(sig)
    
    return completed_trades

def main():
    print(f"🔬 EXACT BOT SIMULATION - JANUARY 2026 ONLY")
    print(f"   Replicating EXACT bot logic: BOS wait, signal expiration, per-symbol filtering")
    start_time = time.time()
    
    with open('config.yaml', 'r') as f:
        conf = yaml.safe_load(f)
    
    tasks = list(conf.get('symbols', {}).items())
    print(f"   Processing {len(tasks)} symbols...")
    
    all_trades = []
    
    # OPTIMIZED: More workers and better progress logging
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as exc:
        futures = {exc.submit(process_symbol, task): task[0] for task in tasks}
        processed = 0
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                all_trades.extend(result)
            except Exception as e:
                print(f"   [{sym}] Error: {e}")
            processed += 1
            if processed % 10 == 0:
                print(f"   Processed {processed}/{len(tasks)} symbols...", end='\r')
    
    print(f"\n\n   Total Trades Found: {len(all_trades)}")
    
    # ====== SIMULATE PORTFOLIO ======
    all_trades.sort(key=lambda x: x.entry_time)
    
    balance = INITIAL_CAPITAL
    peak_balance = INITIAL_CAPITAL
    max_drawdown = 0
    
    monthly_stats = {m[0]: {'wins': 0, 'losses': 0, 'net_r': 0, 'start_bal': 0, 'end_bal': 0} for m in MONTHS}
    
    def get_month_key(ts):
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        for m_name, m_start, m_end in MONTHS:
            m_start_ts = int(datetime.strptime(m_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            m_end_ts = int(datetime.strptime(m_end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            if m_start_ts <= ts <= m_end_ts:
                return m_name
        return None
    
    # Set initial start balance
    monthly_stats[MONTHS[0][0]]['start_bal'] = INITIAL_CAPITAL
    current_month = None
    
    for t in all_trades:
        m_key = get_month_key(t.entry_time)
        if m_key is None:
            continue
        
        if current_month != m_key:
            if current_month:
                monthly_stats[current_month]['end_bal'] = balance
            monthly_stats[m_key]['start_bal'] = balance
            current_month = m_key
        
        risk_amt = balance * RISK_PCT
        realized_r = t.r_result - 0.1  # Fee adjustment
        pnl = realized_r * risk_amt
        balance += pnl
        
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd
        
        monthly_stats[m_key]['net_r'] += realized_r
        if t.r_result > 0:
            monthly_stats[m_key]['wins'] += 1
        else:
            monthly_stats[m_key]['losses'] += 1
    
    if current_month:
        monthly_stats[current_month]['end_bal'] = balance
    
    # ====== PRINT RESULTS ======
    print("\n" + "=" * 100)
    print(f"{'MONTH':<10} | {'TRADES (W/L)':<15} | {'WR%':<6} | {'NET R':<10} | {'START BAL':<12} | {'END BAL':<12} | {'CHANGE'}")
    print("=" * 100)
    
    for m_name, _, _ in MONTHS:
        s = monthly_stats[m_name]
        total = s['wins'] + s['losses']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        change = s['end_bal'] - s['start_bal']
        pct = (change / s['start_bal'] * 100) if s['start_bal'] > 0 else 0
        
        print(f"{m_name:<10} | {total:<4} ({s['wins']}/{s['losses']}){'':<5} | {wr:<6.1f} | {s['net_r']:<10.1f} | ${s['start_bal']:<11.2f} | ${s['end_bal']:<11.2f} | {pct:+.1f}%")
    
    print("=" * 100)
    print(f"FINAL BALANCE: ${balance:.2f}")
    print(f"TOTAL RETURN: {(balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:.1f}%")
    print(f"MAX DRAWDOWN: {max_drawdown:.1f}%")
    print(f"TIME TAKEN: {time.time() - start_time:.1f}s")
    print("=" * 100)

if __name__ == "__main__":
    main()
