#!/usr/bin/env python3
"""
validate_current_config.py - 6-Month Super Realistic Validation
================================================================
Simulates the EXACT bot logic from August 2025 to January 2026.

Features:
1. Candle-by-candle simulation (no lookahead)
2. Monthly performance breakdown
3. Live progress reporting
4. Uses current config.yaml symbols and parameters
5. Tracks Max Drawdown, Win Rate, and R per month

Run: python validate_current_config.py
"""

import requests
import pandas as pd
import numpy as np
import yaml
import time
import sys
import concurrent.futures
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ============ CONFIGURATION ============
INITIAL_CAPITAL = 700.0
RISK_PCT = 0.001  # 0.1% risk per trade

# Simulation Period: August 2025 to January 2026 (6 months)
START_DATE = "2025-08-01"
END_DATE = "2026-01-28"

# Indicator Constants
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
EMA_PERIOD = 200

# BOS Constants
MAX_WAIT_CANDLES = 12

# Costs (realistic)
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

# Month definitions for reporting
MONTHS = [
    ("2025-08", "2025-08-01", "2025-08-31"),
    ("2025-09", "2025-09-01", "2025-09-30"),
    ("2025-10", "2025-10-01", "2025-10-31"),
    ("2025-11", "2025-11-01", "2025-11-30"),
    ("2025-12", "2025-12-01", "2025-12-31"),
    ("2026-01", "2026-01-01", "2026-01-28"),
]

BASE_URL = "https://api.bybit.com"

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
class PendingSignal:
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
    symbol: str
    side: str
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float
    tp_price: float
    rr: float
    atr_mult: float
    exit_time: Optional[pd.Timestamp] = None
    r_result: float = 0.0
    outcome: str = "open"

@dataclass
class MonthlyStats:
    month: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    max_dd: float = 0.0
    peak_equity: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0
    
    @property
    def avg_r(self) -> float:
        return self.total_r / self.trades if self.trades > 0 else 0.0

# ============ DATA FETCHING ============
def fetch_klines(symbol: str, interval: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch historical klines with buffer for indicators"""
    end_ts = int(pd.to_datetime(end_date, utc=True).timestamp() * 1000)
    
    # Add buffer for EMA calculation
    if interval == '60':
        buffer_days = 15  # Extra days for EMA 200
        start_ts = int((pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=buffer_days)).timestamp() * 1000)
    else:
        start_ts = int(pd.to_datetime(start_date, utc=True).timestamp() * 1000)
    
    all_candles = []
    current_end = end_ts
    max_requests = 200
    
    while current_end > start_ts and max_requests > 0:
        max_requests -= 1
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
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
            current_end = oldest - 1
            time.sleep(0.03)
        except Exception as e:
            time.sleep(0.5)
            continue
    
    if not all_candles:
        return None
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
    df['start'] = pd.to_datetime(df['start'], unit='ms', utc=True)
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df

# ============ INDICATOR CALCULATIONS ============
def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

# ============ PIVOT DETECTION ============
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

# ============ DIVERGENCE DETECTION ============
def detect_divergence_at_candle(df: pd.DataFrame, idx: int, symbol_cfg: dict) -> Optional[DivergenceSignal]:
    """Detect divergence at a specific candle index - NO LOOKAHEAD"""
    if idx < EMA_PERIOD + 10:
        return None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    timestamps = df['start'].values
    
    target_type = symbol_cfg.get('divergence_type', 'REG_BEAR')
    
    # Find pivots up to current candle only
    pivot_highs, pivot_lows = find_pivots(close[:idx+1])
    
    curr_price = close[idx]
    curr_ema = ema[idx]
    curr_rsi = rsi[idx]
    
    # === BULLISH DIVERGENCES (long) ===
    if target_type in ['REG_BULL', 'HID_BULL'] and curr_price > curr_ema:
        # Find two most recent pivot lows
        p_lows = []
        for j in range(idx - 3, max(0, idx - 50), -1):
            if not np.isnan(pivot_lows[j]):
                p_lows.append((j, pivot_lows[j]))
                if len(p_lows) >= 2:
                    break
        
        if len(p_lows) >= 2:
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            
            if (idx - curr_idx) <= LOOKBACK_BARS:
                # Regular Bullish: Lower lows in price, higher lows in RSI
                if target_type == 'REG_BULL' and curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    return DivergenceSignal(
                        symbol=symbol_cfg['symbol'],
                        side='long',
                        divergence_code='REG_BULL',
                        divergence_idx=idx,
                        swing_level=max(high[curr_idx:idx+1]),
                        rsi_value=curr_rsi,
                        price=curr_price,
                        timestamp=pd.Timestamp(timestamps[idx]),
                        pivot_timestamp=pd.Timestamp(timestamps[curr_idx]),
                        daily_trend_aligned=True
                    )
                # Hidden Bullish: Higher lows in price, lower lows in RSI
                if target_type == 'HID_BULL' and curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    return DivergenceSignal(
                        symbol=symbol_cfg['symbol'],
                        side='long',
                        divergence_code='HID_BULL',
                        divergence_idx=idx,
                        swing_level=max(high[curr_idx:idx+1]),
                        rsi_value=curr_rsi,
                        price=curr_price,
                        timestamp=pd.Timestamp(timestamps[idx]),
                        pivot_timestamp=pd.Timestamp(timestamps[curr_idx]),
                        daily_trend_aligned=True
                    )
    
    # === BEARISH DIVERGENCES (short) ===
    if target_type in ['REG_BEAR', 'HID_BEAR'] and curr_price < curr_ema:
        p_highs = []
        for j in range(idx - 3, max(0, idx - 50), -1):
            if not np.isnan(pivot_highs[j]):
                p_highs.append((j, pivot_highs[j]))
                if len(p_highs) >= 2:
                    break
        
        if len(p_highs) >= 2:
            curr_idx, curr_val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            
            if (idx - curr_idx) <= LOOKBACK_BARS:
                # Regular Bearish: Higher highs in price, lower highs in RSI
                if target_type == 'REG_BEAR' and curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    return DivergenceSignal(
                        symbol=symbol_cfg['symbol'],
                        side='short',
                        divergence_code='REG_BEAR',
                        divergence_idx=idx,
                        swing_level=min(low[curr_idx:idx+1]),
                        rsi_value=curr_rsi,
                        price=curr_price,
                        timestamp=pd.Timestamp(timestamps[idx]),
                        pivot_timestamp=pd.Timestamp(timestamps[curr_idx]),
                        daily_trend_aligned=True
                    )
                # Hidden Bearish: Lower highs in price, higher highs in RSI
                if target_type == 'HID_BEAR' and curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    return DivergenceSignal(
                        symbol=symbol_cfg['symbol'],
                        side='short',
                        divergence_code='HID_BEAR',
                        divergence_idx=idx,
                        swing_level=min(low[curr_idx:idx+1]),
                        rsi_value=curr_rsi,
                        price=curr_price,
                        timestamp=pd.Timestamp(timestamps[idx]),
                        pivot_timestamp=pd.Timestamp(timestamps[curr_idx]),
                        daily_trend_aligned=True
                    )
    
    return None

# ============ BOS CHECK ============
def check_bos(signal: DivergenceSignal, candle: pd.Series) -> bool:
    """Check if Break of Structure occurred on candle close"""
    if signal.side == 'long':
        return candle['close'] > signal.swing_level
    else:
        return candle['close'] < signal.swing_level

# ============ TRADE EXECUTION ON 5M ============
def execute_trade_on_5m(trade: Trade, df_5m: pd.DataFrame) -> Trade:
    """Execute trade on 5-minute data - candle by candle"""
    # Find 5m candles after entry
    mask = df_5m['start'] >= trade.entry_time
    subset = df_5m[mask]
    
    if subset.empty:
        trade.outcome = "timeout"
        trade.r_result = 0.0
        return trade
    
    for _, row in subset.iterrows():
        if trade.side == 'long':
            hit_sl = row['low'] <= trade.sl_price
            hit_tp = row['high'] >= trade.tp_price
        else:
            hit_sl = row['high'] >= trade.sl_price
            hit_tp = row['low'] <= trade.tp_price
        
        # SL checked first (conservative)
        if hit_sl and hit_tp:
            trade.outcome = "loss"
            trade.r_result = -1.0
            trade.exit_time = row['start']
            return trade
        elif hit_sl:
            trade.outcome = "loss"
            trade.r_result = -1.0
            trade.exit_time = row['start']
            return trade
        elif hit_tp:
            trade.outcome = "win"
            trade.r_result = trade.rr
            trade.exit_time = row['start']
            return trade
    
    # Timeout - no SL/TP hit
    trade.outcome = "timeout"
    trade.r_result = 0.0
    return trade

# ============ SYMBOL SIMULATION ============
def simulate_symbol(symbol: str, cfg: dict, df_1h: pd.DataFrame, df_5m: pd.DataFrame) -> List[Trade]:
    """Candle-by-candle simulation for a single symbol"""
    trades = []
    pending: Optional[PendingSignal] = None
    in_trade = False
    
    symbol_cfg = {
        'symbol': symbol,
        'divergence_type': cfg.get('divergence_type', 'REG_BEAR'),
        'rr': cfg.get('rr', 5.0),
        'atr_mult': cfg.get('atr_mult', 1.0)
    }
    
    start_ts = pd.to_datetime(START_DATE, utc=True)
    
    for idx in range(EMA_PERIOD + 10, len(df_1h)):
        row = df_1h.iloc[idx]
        
        # Skip if before simulation start
        if row['start'] < start_ts:
            continue
        
        # === CHECK PENDING SIGNAL FOR BOS ===
        if pending and not in_trade:
            pending.increment_wait()
            
            if pending.is_expired():
                pending = None
            elif check_bos(pending.signal, row):
                # BOS confirmed! Entry on NEXT candle
                if idx + 1 < len(df_1h):
                    next_candle = df_1h.iloc[idx + 1]
                    entry_price = next_candle['open']
                    atr = row['atr']
                    sl_dist = atr * symbol_cfg['atr_mult']
                    
                    # Apply slippage
                    if pending.signal.side == 'long':
                        entry_price *= (1 + SLIPPAGE_PCT)
                        sl_price = entry_price - sl_dist
                        tp_price = entry_price + (sl_dist * symbol_cfg['rr'])
                    else:
                        entry_price *= (1 - SLIPPAGE_PCT)
                        sl_price = entry_price + sl_dist
                        tp_price = entry_price - (sl_dist * symbol_cfg['rr'])
                    
                    trade = Trade(
                        symbol=symbol,
                        side=pending.signal.side,
                        entry_price=entry_price,
                        entry_time=next_candle['start'],
                        sl_price=sl_price,
                        tp_price=tp_price,
                        rr=symbol_cfg['rr'],
                        atr_mult=symbol_cfg['atr_mult']
                    )
                    
                    # Execute on 5M
                    trade = execute_trade_on_5m(trade, df_5m)
                    
                    # Apply fee drag
                    if trade.r_result != 0:
                        fee_drag = (FEE_PCT * 2 * entry_price) / sl_dist
                        trade.r_result -= fee_drag
                    
                    trades.append(trade)
                
                pending = None
                continue
        
        # === DETECT NEW DIVERGENCE ===
        if not pending and not in_trade:
            signal = detect_divergence_at_candle(df_1h, idx, symbol_cfg)
            if signal:
                pending = PendingSignal(signal=signal, detected_at_idx=idx)
    
    return trades

# ============ MAIN SIMULATION ============
def main():
    print("=" * 70)
    print("🚀 SUPER REALISTIC VALIDATION: August 2025 - January 2026")
    print("=" * 70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols_config = config.get('symbols', {})
    enabled_symbols = {k: v for k, v in symbols_config.items() if v.get('enabled', False)}
    
    print(f"📊 Loaded {len(enabled_symbols)} symbols from config.yaml")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print("-" * 70)
    
    all_trades: List[Trade] = []
    completed = 0
    total = len(enabled_symbols)
    
    # Process symbols
    for symbol, cfg in enabled_symbols.items():
        completed += 1
        sys.stdout.write(f"\r⏳ Processing {symbol} ({completed}/{total})...")
        sys.stdout.flush()
        
        # Fetch data
        df_1h = fetch_klines(symbol, '60', START_DATE, END_DATE)
        df_5m = fetch_klines(symbol, '5', START_DATE, END_DATE)
        
        if df_1h is None or df_5m is None or len(df_1h) < 300:
            continue
        
        df_1h = prepare_dataframe(df_1h)
        
        # Simulate
        trades = simulate_symbol(symbol, cfg, df_1h, df_5m)
        all_trades.extend(trades)
        
        # Live progress every 10 symbols
        if completed % 10 == 0:
            wins = sum(1 for t in all_trades if t.outcome == 'win')
            losses = sum(1 for t in all_trades if t.outcome == 'loss')
            total_r = sum(t.r_result for t in all_trades)
            wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            print(f"\n   📈 Progress: {len(all_trades)} trades | WR: {wr:.1f}% | Total R: {total_r:+.1f}")
    
    print(f"\n\n{'=' * 70}")
    print("📊 MONTHLY BREAKDOWN")
    print("=" * 70)
    
    # Monthly stats
    for month_name, m_start, m_end in MONTHS:
        m_start_ts = pd.to_datetime(m_start, utc=True)
        m_end_ts = pd.to_datetime(m_end, utc=True) + pd.Timedelta(days=1)
        
        month_trades = [t for t in all_trades if m_start_ts <= t.entry_time < m_end_ts]
        
        if not month_trades:
            print(f"\n{month_name}: No trades")
            continue
        
        wins = sum(1 for t in month_trades if t.outcome == 'win')
        losses = sum(1 for t in month_trades if t.outcome == 'loss')
        total_r = sum(t.r_result for t in month_trades)
        wr = (wins / len(month_trades) * 100) if month_trades else 0
        avg_r = total_r / len(month_trades) if month_trades else 0
        
        # Max DD calculation
        equity = INITIAL_CAPITAL
        peak = equity
        max_dd = 0
        for t in sorted(month_trades, key=lambda x: x.entry_time):
            pnl = t.r_result * INITIAL_CAPITAL * RISK_PCT
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        
        print(f"\n📅 {month_name}:")
        print(f"   Trades: {len(month_trades)} | Wins: {wins} | Losses: {losses}")
        print(f"   Win Rate: {wr:.1f}%")
        print(f"   Total R: {total_r:+.1f}")
        print(f"   Avg R/Trade: {avg_r:+.2f}")
        print(f"   Max DD: {max_dd:.1f}%")
    
    # Overall summary
    print(f"\n{'=' * 70}")
    print("📊 OVERALL SUMMARY (6 Months)")
    print("=" * 70)
    
    if all_trades:
        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t.outcome == 'win')
        total_losses = sum(1 for t in all_trades if t.outcome == 'loss')
        total_r = sum(t.r_result for t in all_trades)
        overall_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        overall_avg_r = total_r / total_trades if total_trades else 0
        
        # Overall max DD
        equity = INITIAL_CAPITAL
        peak = equity
        overall_max_dd = 0
        for t in sorted(all_trades, key=lambda x: x.entry_time):
            pnl = t.r_result * INITIAL_CAPITAL * RISK_PCT
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            overall_max_dd = max(overall_max_dd, dd)
        
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {overall_wr:.1f}%")
        print(f"   Total R: {total_r:+.1f}")
        print(f"   Avg R/Trade: {overall_avg_r:+.2f}")
        print(f"   Max Drawdown: {overall_max_dd:.1f}%")
        print(f"   Final Equity: ${equity:.2f}")
    
    # Save trades
    if all_trades:
        df_trades = pd.DataFrame([
            {
                'symbol': t.symbol,
                'side': t.side,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'outcome': t.outcome,
                'r_result': t.r_result
            }
            for t in all_trades
        ])
        df_trades.to_csv('validation_6month_trades.csv', index=False)
        print(f"\n💾 Saved {len(all_trades)} trades to validation_6month_trades.csv")

if __name__ == "__main__":
    main()
