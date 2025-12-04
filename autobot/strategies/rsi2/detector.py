from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

@dataclass
class Rsi2Settings:
    # Strategy Params
    rsi_period: int = 2
    ema_period: int = 200
    rsi_oversold: int = 10
    rsi_overbought: int = 90
    
    # Risk (Fixed TP/SL for now)
    tp_pct: float = 0.02  # 2% Target
    sl_pct: float = 0.05  # 5% Stop (Wide safety net)

def calculate_rsi2_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI 2 and EMA 200"""
    close = df['close']
    
    # RSI 2
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=1, adjust=False).mean() # RSI 2 -> com=1
    ma_down = down.ewm(com=1, adjust=False).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    
    # EMA 200
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    d = df.copy()
    d['rsi2'] = rsi
    d['ema200'] = ema200
    return d

def detect_rsi2_signal(df: pd.DataFrame, s: Rsi2Settings = None, symbol: str = "") -> Optional[object]:
    """
    Detects RSI 2 Mean Reversion signals.
    Returns a ScalpSignal-like object (duck typing) or None.
    """
    if s is None:
        s = Rsi2Settings()
        
    if df is None or len(df) < 205: # Need 200 EMA
        return None
        
    d = calculate_rsi2_indicators(df)
    row = d.iloc[-1]
    
    # Logic
    close = row['close']
    ema = row['ema200']
    rsi = row['rsi2']
    
    # Long: Trend UP + Oversold
    if close > ema and rsi < s.rsi_oversold:
        # Import ScalpSignal here to avoid circular imports if possible, 
        # or just return a compatible object/dict. 
        # For safety, let's return a simple object that looks like ScalpSignal
        from autobot.strategies.scalp.detector import ScalpSignal
        
        entry = close
        tp = entry * (1 + s.tp_pct)
        sl = entry * (1 - s.sl_pct)
        
        return ScalpSignal(
            side='long',
            entry=entry,
            sl=sl,
            tp=tp,
            reason=f"RSI2_LONG (RSI={rsi:.1f})",
            meta={'strategy': 'rsi2', 'rsi': rsi, 'ema': ema}
        )
        
    # Short: Trend DOWN + Overbought
    elif close < ema and rsi > s.rsi_overbought:
        from autobot.strategies.scalp.detector import ScalpSignal
        
        entry = close
        tp = entry * (1 - s.tp_pct)
        sl = entry * (1 + s.sl_pct)
        
        return ScalpSignal(
            side='short',
            entry=entry,
            sl=sl,
            tp=tp,
            reason=f"RSI2_SHORT (RSI={rsi:.1f})",
            meta={'strategy': 'rsi2', 'rsi': rsi, 'ema': ema}
        )
        
    return None
