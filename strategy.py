from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Settings:
    left:int=2
    right:int=2
    atr_len:int=14
    sl_buf_atr:float=0.5
    rr:float=2.0
    use_ema:bool=False
    ema_len:int=200
    use_vol:bool=False
    vol_len:int=20
    vol_mult:float=1.2
    both_hit_rule:str="SL_FIRST"  # or "TP_FIRST"

@dataclass
class Signal:
    side:str             # "long" or "short"
    entry:float
    sl:float
    tp:float
    reason:str
    meta:dict

def _pivot_high(h:pd.Series, L:int, R:int):
    # confirmed pivot high at index i if h[i] is the max across i-L..i+R
    win = h.rolling(L+R+1, center=True).max()
    mask = (h == win) & h.notna()
    return np.where(mask, h, np.nan)

def _pivot_low(l:pd.Series, L:int, R:int):
    win = l.rolling(L+R+1, center=True).min()
    mask = (l == win) & l.notna()
    return np.where(mask, l, np.nan)

def _atr(df:pd.DataFrame, n:int):
    prev_close = df["close"].shift()
    tr = np.maximum(df["high"]-df["low"],
         np.maximum(abs(df["high"]-prev_close), abs(df["low"]-prev_close)))
    return pd.Series(tr, index=df.index).rolling(n).mean().values

def _ema(s:pd.Series, n:int):
    return s.ewm(span=n, adjust=False).mean().values

def detect_signal(df:pd.DataFrame, s:Settings) -> Signal|None:
    """
    df: DataFrame with columns ['open','high','low','close','volume'] indexed by datetime
    Returns a Signal at bar close (last row), or None.
    """
    need = max(200, s.atr_len) + s.left + s.right + 3
    if len(df) < need:
        return None

    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    ph = pd.Series(_pivot_high(high, s.left, s.right), index=df.index)
    pl = pd.Series(_pivot_low(low,  s.left, s.right),  index=df.index)

    dh = ph.dropna()
    dl = pl.dropna()
    if len(dh) < 2 or len(dl) < 2:
        return None

    lastHigh, prevHigh = float(dh.iloc[-1]), float(dh.iloc[-2])
    lastLow,  prevLow  = float(dl.iloc[-1]), float(dl.iloc[-2])

    trendUp = (lastHigh > prevHigh) and (lastLow > prevLow)
    trendDn = (lastHigh < prevHigh) and (lastLow < prevLow)

    nearestRes, nearestSup = lastHigh, lastLow

    atr = float(_atr(df, s.atr_len)[-1])
    ema_ok_long = True
    ema_ok_short = True
    if s.use_ema:
        ema_val = float(_ema(close, s.ema_len)[-1])
        ema_ok_long = close.iloc[-1] > ema_val
        ema_ok_short= close.iloc[-1] < ema_val

    vol_ok = True
    if s.use_vol:
        vol_ok = vol.iloc[-1] > vol.rolling(s.vol_len).mean().iloc[-1] * s.vol_mult

    c = float(close.iloc[-1])
    crossRes = (c > nearestRes)
    crossSup = (c < nearestSup)

    if trendUp and crossRes and vol_ok and ema_ok_long:
        entry = c
        sl = nearestSup - s.sl_buf_atr * atr
        if entry <= sl:
            return None
        R = entry - sl
        tp = entry + s.rr * R
        return Signal("long", entry, sl, tp, "Up-structure breakout over resistance",
                      {"atr":atr, "res":nearestRes, "sup":nearestSup})

    if trendDn and crossSup and vol_ok and ema_ok_short:
        entry = c
        sl = nearestRes + s.sl_buf_atr * atr
        if sl <= entry:
            return None
        R = sl - entry
        tp = entry - s.rr * R
        return Signal("short", entry, sl, tp, "Down-structure breakdown under support",
                      {"atr":atr, "res":nearestRes, "sup":nearestSup})

    return None