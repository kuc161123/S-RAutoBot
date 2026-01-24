#!/usr/bin/env python3
"""
debug_january_simulation.py - Debug Script for January 2026
============================================================
Runs a detailed simulation with extensive logging to verify correctness.
"""

import requests
import pandas as pd
import numpy as np
import yaml
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ============ CONFIGURATION ============
# Simulation Period - JANUARY 2026 ONLY
START_DATE = "2026-01-01 00:00:00"
END_DATE = "2026-01-17 23:59:59"

# Divergence Detection Constants
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
EMA_PERIOD = 200
MAX_WAIT_CANDLES = 12

# ============ DATA CLASSES ============
@dataclass
class DivergenceSignal:
    symbol: str
    side: str
    divergence_code: str
    divergence_idx: int
    swing_level: float
    rsi_value: float
    price: float
    timestamp: pd.Timestamp
    pivot_timestamp: pd.Timestamp
    daily_trend_aligned: bool

@dataclass
class Trade:
    symbol: str
    side: str
    divergence_code: str
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp_price: float
    exit_time: datetime
    exit_price: float
    r_result: float
    exit_reason: str  # 'SL', 'TP', or 'TIMEOUT'

# ============ DATA FETCHING ============
def fetch_klines(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch historical klines from Bybit"""
    end_req = int(datetime(2026, 1, 18, tzinfo=timezone.utc).timestamp() * 1000)
    
    if interval == '60':
        start_req = int(datetime(2025, 12, 20, tzinfo=timezone.utc).timestamp() * 1000)
    else:
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
            return None
    
    if not all_candles:
        return None
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
    df['start'] = pd.to_datetime(df['start'], unit='ms', utc=True)
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    df.set_index('start', inplace=True)
    
    return df

def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    df['daily_ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

def find_pivots(data: np.ndarray, left: int = PIVOT_LEFT, right: int = PIVOT_RIGHT) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high:
            pivot_highs[i] = data[i]
        
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_low:
            pivot_lows[i] = data[i]
    
    return pivot_highs, pivot_lows

def detect_all_divergences(df: pd.DataFrame, symbol: str) -> List[DivergenceSignal]:
    """Detect ALL divergence types (not filtered)"""
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
    
    for i in range(30, n - PIVOT_RIGHT):
        current_price = close[i]
        current_ema = daily_ema[i]
        
        # BULLISH
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
            
            # REG_BULL
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                signals.append(DivergenceSignal(
                    symbol=symbol, side='long', divergence_code='REG_BULL',
                    divergence_idx=i, swing_level=swing_high, rsi_value=rsi[i],
                    price=current_price, timestamp=df.index[i],
                    pivot_timestamp=df.index[curr_pli], daily_trend_aligned=trend_aligned
                ))
            
            # HID_BULL
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                signals.append(DivergenceSignal(
                    symbol=symbol, side='long', divergence_code='HID_BULL',
                    divergence_idx=i, swing_level=swing_high, rsi_value=rsi[i],
                    price=current_price, timestamp=df.index[i],
                    pivot_timestamp=df.index[curr_pli], daily_trend_aligned=trend_aligned
                ))
        
        # BEARISH
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
            
            # REG_BEAR
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                signals.append(DivergenceSignal(
                    symbol=symbol, side='short', divergence_code='REG_BEAR',
                    divergence_idx=i, swing_level=swing_low, rsi_value=rsi[i],
                    price=current_price, timestamp=df.index[i],
                    pivot_timestamp=df.index[curr_phi], daily_trend_aligned=trend_aligned
                ))
            
            # HID_BEAR
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                signals.append(DivergenceSignal(
                    symbol=symbol, side='short', divergence_code='HID_BEAR',
                    divergence_idx=i, swing_level=swing_low, rsi_value=rsi[i],
                    price=current_price, timestamp=df.index[i],
                    pivot_timestamp=df.index[curr_phi], daily_trend_aligned=trend_aligned
                ))
    
    return signals

def main():
    print("=" * 80)
    print("🔬 DEBUG JANUARY 2026 SIMULATION")
    print("=" * 80)
    
    with open('config.yaml', 'r') as f:
        conf = yaml.safe_load(f)
    
    symbols = conf.get('symbols', {})
    print(f"Total symbols in config: {len(symbols)}")
    
    # Sample 5 symbols for detailed debug
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
    
    # Also find some that ARE in config for comparison
    config_symbols = [s for s in symbols.keys() if symbols[s].get('enabled', True)][:5]
    
    all_debug_symbols = list(set(test_symbols + config_symbols))
    
    print(f"\nDebug Symbols: {all_debug_symbols}")
    print(f"Config Symbols (first 5 enabled): {config_symbols}")
    
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    total_signals = 0
    filtered_signals = 0
    traded_signals = 0
    wins = 0
    losses = 0
    
    all_trades = []
    
    for sym in all_debug_symbols:
        print(f"\n{'='*60}")
        print(f"[{sym}] Processing...")
        
        cfg = symbols.get(sym, {})
        is_enabled = cfg.get('enabled', False)
        allowed_div = cfg.get('divergence_type')
        rr = cfg.get('rr', 5.0)
        atr_mult = cfg.get('atr_mult', 1.0)
        
        print(f"   Config: enabled={is_enabled}, divergence_type={allowed_div}, rr={rr}, atr_mult={atr_mult}")
        
        # Fetch data
        df_1h = fetch_klines(sym, '60')
        if df_1h is None:
            print(f"   ❌ Failed to fetch 1H data")
            continue
        
        df_1h = prepare_dataframe(df_1h)
        print(f"   📊 Fetched {len(df_1h)} 1H candles")
        
        df_5m = fetch_klines(sym, '5')
        if df_5m is None:
            print(f"   ❌ Failed to fetch 5M data")
            continue
        print(f"   📊 Fetched {len(df_5m)} 5M candles")
        
        # Find all divergences in Jan
        jan_divergences = []
        for current_idx in range(100, len(df_1h)):
            candle = df_1h.iloc[current_idx]
            ts = int(candle.name.timestamp() * 1000)
            
            if ts < start_ts or ts > end_ts:
                continue
            
            df_slice = df_1h.iloc[:current_idx + 1]
            signals = detect_all_divergences(df_slice, sym)
            
            last_valid_idx = len(df_slice) - 1 - PIVOT_RIGHT
            for sig in signals:
                if sig.divergence_idx >= last_valid_idx:
                    jan_divergences.append((current_idx, sig))
        
        total_signals += len(jan_divergences)
        print(f"   📈 Total divergences detected in Jan: {len(jan_divergences)}")
        
        # Count by type
        type_counts = {}
        for _, sig in jan_divergences:
            key = sig.divergence_code
            type_counts[key] = type_counts.get(key, 0) + 1
        print(f"   📊 By type: {type_counts}")
        
        # Filter by allowed type
        if allowed_div:
            matching = [(idx, sig) for idx, sig in jan_divergences if sig.divergence_code == allowed_div]
            filtered_out = len(jan_divergences) - len(matching)
            print(f"   🔍 After filtering for '{allowed_div}': {len(matching)} (filtered out: {filtered_out})")
        else:
            matching = jan_divergences
        
        filtered_signals += len(matching)
        
        # Check trend alignment
        trend_aligned = [(idx, sig) for idx, sig in matching if sig.daily_trend_aligned]
        print(f"   📈 Trend aligned: {len(trend_aligned)} out of {len(matching)}")
        
        # Simulate BOS and trades
        for idx, sig in trend_aligned[:3]:  # Limit to first 3 for debug
            print(f"\n   🎯 Signal: {sig.divergence_code} at {sig.timestamp}")
            print(f"      Side: {sig.side}, Swing: {sig.swing_level:.4f}, Price: {sig.price:.4f}")
            
            # Check BOS in next 12 candles
            bos_confirmed = False
            entry_idx = None
            for wait in range(1, MAX_WAIT_CANDLES + 1):
                check_idx = idx + wait
                if check_idx >= len(df_1h):
                    break
                
                check_price = df_1h.iloc[check_idx]['close']
                if sig.side == 'long':
                    if check_price > sig.swing_level:
                        bos_confirmed = True
                        entry_idx = check_idx + 1
                        print(f"      ✅ BOS confirmed candle {wait}: price {check_price:.4f} > swing {sig.swing_level:.4f}")
                        break
                else:
                    if check_price < sig.swing_level:
                        bos_confirmed = True
                        entry_idx = check_idx + 1
                        print(f"      ✅ BOS confirmed candle {wait}: price {check_price:.4f} < swing {sig.swing_level:.4f}")
                        break
            
            if not bos_confirmed:
                print(f"      ❌ BOS NOT confirmed in 12 candles - EXPIRED")
                continue
            
            if entry_idx >= len(df_1h):
                print(f"      ❌ Entry candle beyond data range")
                continue
            
            # Entry
            entry_candle = df_1h.iloc[entry_idx]
            entry_price = entry_candle['open']
            entry_time = entry_candle.name
            atr = entry_candle['atr']
            
            sl_dist = atr * atr_mult
            if sig.side == 'long':
                sl = entry_price - sl_dist
                tp = entry_price + (sl_dist * rr)
            else:
                sl = entry_price + sl_dist
                tp = entry_price - (sl_dist * rr)
            
            print(f"      📍 Entry: {entry_price:.4f} @ {entry_time}")
            print(f"      📍 SL: {sl:.4f}, TP: {tp:.4f}, ATR: {atr:.4f}")
            print(f"      📍 Distance to SL: {abs(entry_price - sl):.4f} ({abs(entry_price - sl)/entry_price*100:.2f}%)")
            print(f"      📍 Distance to TP: {abs(entry_price - tp):.4f} ({abs(entry_price - tp)/entry_price*100:.2f}%)")
            
            # Check outcome with 5M data
            sub_5m = df_5m[df_5m.index >= entry_time]
            
            outcome = -1.0
            exit_reason = 'TIMEOUT'
            exit_time = entry_time
            exit_price = entry_price
            
            for row in sub_5m.itertuples():
                if sig.side == 'long':
                    if row.low <= sl:
                        outcome = -1.0
                        exit_reason = 'SL'
                        exit_time = row.Index
                        exit_price = sl
                        break
                    if row.high >= tp:
                        outcome = rr
                        exit_reason = 'TP'
                        exit_time = row.Index
                        exit_price = tp
                        break
                else:
                    if row.high >= sl:
                        outcome = -1.0
                        exit_reason = 'SL'
                        exit_time = row.Index
                        exit_price = sl
                        break
                    if row.low <= tp:
                        outcome = rr
                        exit_reason = 'TP'
                        exit_time = row.Index
                        exit_price = tp
                        break
            
            print(f"      💰 Outcome: {outcome:+.1f}R ({exit_reason}) @ {exit_time}")
            
            if outcome > 0:
                wins += 1
            else:
                losses += 1
            
            traded_signals += 1
            
            all_trades.append(Trade(
                symbol=sym,
                side=sig.side,
                divergence_code=sig.divergence_code,
                entry_price=entry_price,
                entry_time=entry_time,
                sl_price=sl,
                tp_price=tp,
                exit_time=exit_time,
                exit_price=exit_price,
                r_result=outcome,
                exit_reason=exit_reason
            ))
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total raw divergences: {total_signals}")
    print(f"After type filter: {filtered_signals}")
    print(f"Trades executed: {traded_signals}")
    print(f"Wins: {wins}, Losses: {losses}")
    if traded_signals > 0:
        print(f"Win Rate: {wins/traded_signals*100:.1f}%")
    
    print("\n🔍 TRADES DETAIL:")
    for t in all_trades:
        print(f"   {t.symbol} {t.side} {t.divergence_code}: {t.entry_price:.4f} -> {t.exit_price:.4f} = {t.r_result:+.1f}R ({t.exit_reason})")

if __name__ == "__main__":
    main()
