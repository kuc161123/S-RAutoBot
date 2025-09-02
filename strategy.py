from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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

def detect_signal(df:pd.DataFrame, s:Settings, symbol:str="") -> Signal|None:
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
    
    # Log S/R levels detected
    logger.info(f"[{symbol}] S/R Levels - Resistance: {lastHigh:.4f} (prev: {prevHigh:.4f}), Support: {lastLow:.4f} (prev: {prevLow:.4f})")

    trendUp = (lastHigh > prevHigh) and (lastLow > prevLow)
    trendDn = (lastHigh < prevHigh) and (lastLow < prevLow)
    
    # Log market structure
    if trendUp:
        logger.info(f"[{symbol}] Market Structure: UPTREND (HH: {lastHigh:.4f} > {prevHigh:.4f}, HL: {lastLow:.4f} > {prevLow:.4f})")
    elif trendDn:
        logger.info(f"[{symbol}] Market Structure: DOWNTREND (LH: {lastHigh:.4f} < {prevHigh:.4f}, LL: {lastLow:.4f} < {prevLow:.4f})")
    else:
        logger.info(f"[{symbol}] Market Structure: RANGING (no clear HH/HL or LH/LL pattern)")

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
    
    # Log price position relative to S/R
    logger.info(f"[{symbol}] Current Price: {c:.4f} | Above Resistance: {crossRes} | Below Support: {crossSup}")

    if trendUp and crossRes and vol_ok and ema_ok_long:
        entry = c
        sl = nearestSup - s.sl_buf_atr * atr
        if entry <= sl:
            logger.info(f"[{symbol}] Long signal rejected - invalid SL placement")
            return None
        R = entry - sl
        tp = entry + s.rr * R
        logger.info(f"[{symbol}] 🟢 LONG SIGNAL - Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, R:R = 1:{s.rr}")
        return Signal("long", entry, sl, tp, "Up-structure breakout over resistance",
                      {"atr":atr, "res":nearestRes, "sup":nearestSup})

    if trendDn and crossSup and vol_ok and ema_ok_short:
        entry = c
        sl = nearestRes + s.sl_buf_atr * atr
        if sl <= entry:
            logger.info(f"[{symbol}] Short signal rejected - invalid SL placement")
            return None
        R = sl - entry
        tp = entry - s.rr * R
        logger.info(f"[{symbol}] 🔴 SHORT SIGNAL - Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, R:R = 1:{s.rr}")
        return Signal("short", entry, sl, tp, "Down-structure breakdown under support",
                      {"atr":atr, "res":nearestRes, "sup":nearestSup})
    
    # Log why no signal was generated
    if not trendUp and not trendDn:
        logger.debug(f"[{symbol}] No signal - ranging market")
    elif trendUp and not crossRes:
        logger.debug(f"[{symbol}] No signal - uptrend but price {c:.4f} below resistance {nearestRes:.4f}")
    elif trendDn and not crossSup:
        logger.debug(f"[{symbol}] No signal - downtrend but price {c:.4f} above support {nearestSup:.4f}")
    elif not vol_ok:
        logger.debug(f"[{symbol}] No signal - volume filter not met")
    elif trendUp and not ema_ok_long:
        logger.debug(f"[{symbol}] No signal - price below EMA filter")
    elif trendDn and not ema_ok_short:
        logger.debug(f"[{symbol}] No signal - price above EMA filter")

    return None