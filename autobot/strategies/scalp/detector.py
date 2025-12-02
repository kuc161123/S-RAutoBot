from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

@dataclass
class ScalpSettings:
    # Risk
    rr: float = 1.0  # 1:1 R:R
    atr_mult: float = 2.0  # 2 ATR Stop/Target
    
    # Breakout Thresholds
    bbw_pct_min: float = 0.45
    vol_ratio_min: float = 0.8
    
    # Allowed Combos (loaded from overrides)
    allowed_combos_long: List[str] = field(default_factory=list)
    allowed_combos_short: List[str] = field(default_factory=list)

@dataclass
class ScalpSignal:
    side: str
    entry: float
    sl: float
    tp: float
    reason: str
    meta: dict

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators needed for Combo generation"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI 14
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    
    # VWAP (Rolling 20)
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()
    atr = (high - low).rolling(14).mean()
    vwap_dist_atr = (close - vwap).abs() / atr
    
    # MTF (EMA 20 vs 50)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    mtf_agree = ema20 > ema50
    
    # Fib Zone (50 bar lookback)
    roll_max = high.rolling(50).max()
    roll_min = low.rolling(50).min()
    fib_ret = (roll_max - close) / (roll_max - roll_min) * 100
    
    # BB Width Pct
    std = close.rolling(20).std()
    ma = close.rolling(20).mean()
    upper = ma + 2*std
    lower = ma - 2*std
    bbw = (upper - lower) / close
    bbw_pct = bbw.rolling(100).rank(pct=True)
    
    # Volume Ratio
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma
    
    # Assign to DF (copy to avoid warning)
    d = df.copy()
    d['rsi'] = rsi
    d['macd_hist'] = macd_hist
    d['vwap_dist_atr'] = vwap_dist_atr
    d['mtf_agree'] = mtf_agree
    d['fib_ret'] = fib_ret
    d['bbw_pct'] = bbw_pct
    d['vol_ratio'] = vol_ratio
    d['atr'] = atr
    
    return d

def get_combo(row) -> str:
    """Build combo string from indicator values (matches backtest logic)"""
    # RSI Bin
    r = row['rsi']
    if r < 30: rb = '<30'
    elif r < 40: rb = '30-40'
    elif r < 60: rb = '40-60'
    elif r < 70: rb = '60-70'
    else: rb = '70+'
    
    # MACD Bin
    mb = 'bull' if row['macd_hist'] > 0 else 'bear'
    
    # VWAP Bin
    v = row['vwap_dist_atr']
    if v < 0.6: vb = '<0.6'
    elif v < 1.2: vb = '0.6-1.2'
    else: vb = '1.2+'
    
    # Fib Bin
    f = row['fib_ret']
    if f < 23.6: fb = '0-23'
    elif f < 38.2: fb = '23-38'
    elif f < 50.0: fb = '38-50'
    elif f < 61.8: fb = '50-61'
    elif f < 78.6: fb = '61-78'
    else: fb = '78-100'
    
    # MTF
    ma = 'MTF' if row['mtf_agree'] else 'noMTF'
    
    return f"RSI:{rb} MACD:{mb} VWAP:{vb} Fib:{fb} {ma}"

def detect_scalp_signal(df: pd.DataFrame, s: ScalpSettings, symbol: str = "") -> Optional[ScalpSignal]:
    """
    Detects Adaptive Combo signals.
    1. Checks Breakout Condition (BBW, Vol).
    2. Generates Combo String.
    3. Checks if Combo is in allowed list for the symbol.
    """
    if df is None or len(df) < 100:
        return None
        
    # Calculate indicators
    d = calculate_indicators(df)
    row = d.iloc[-1]
    
    # 1. Breakout Check
    if row['bbw_pct'] <= s.bbw_pct_min:
        return None
    if row['vol_ratio'] <= s.vol_ratio_min:
        return None
        
    # 2. Generate Combo
    combo = get_combo(row)
    
    # 3. Check Allowed Combos
    entry = row['close']
    atr = row['atr']
    
    # Long
    if row['close'] > row['open']:
        if combo in s.allowed_combos_long:
            sl = entry - s.atr_mult * atr
            tp = entry + s.atr_mult * atr
            return ScalpSignal(
                side='long',
                entry=entry,
                sl=sl,
                tp=tp,
                reason=f"COMBO_LONG: {combo}",
                meta={'combo': combo, 'atr': atr}
            )
            
    # Short
    elif row['close'] < row['open']:
        if combo in s.allowed_combos_short:
            sl = entry + s.atr_mult * atr
            tp = entry - s.atr_mult * atr
            return ScalpSignal(
                side='short',
                entry=entry,
                sl=sl,
                tp=tp,
                reason=f"COMBO_SHORT: {combo}",
                meta={'combo': combo, 'atr': atr}
            )
            
    return None
