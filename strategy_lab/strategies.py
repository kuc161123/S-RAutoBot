"""The pre-registered candidate strategies.

Each is a thin adapter over logic that already exists in the repo (or, for the new classes,
a direct implementation) reshaped to the common Signal contract so the shared execution
engine can score them identically.

Causality rule for every detect(): a Signal with conf_idx = i must be derivable from bars
0..i only. contract.assert_causal() verifies this and is run before any strategy is scored.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contract import Signal


# ---------------------------------------------------------------------------
# shared indicators (causal by construction)
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.where(d > 0, 0.0).rolling(period).mean()
    loss = -d.where(d < 0, 0.0).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()


def _macd_hist(close: pd.Series) -> pd.Series:
    macd = _ema(close, 12) - _ema(close, 26)
    return macd - _ema(macd, 9)


def _stoch_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    lo = df["low"].rolling(period).min()
    hi = df["high"].rolling(period).max()
    return 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)


def _find_pivots(x: np.ndarray, left: int = 3, right: int = 3):
    """Fractal pivots. A pivot at i needs bars up to i+right, so any consumer must not
    treat i as known before i+right — the detectors below all respect that."""
    ph = np.full(len(x), np.nan)
    pl = np.full(len(x), np.nan)
    for i in range(left, len(x) - right):
        w = x[i - left:i + right + 1]
        c = x[i]
        if c == w.max() and (w == c).sum() == 1:
            ph[i] = c
        if c == w.min() and (w == c).sum() == 1:
            pl[i] = c
    return ph, pl


def _prep_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = _rsi(df["close"])
    df["atr"] = _atr(df)
    df["ema200"] = _ema(df["close"], 200)
    return df


# ===========================================================================
# C2 — INCUMBENT: RSI divergence + Break of Structure (the control baseline)
# ===========================================================================
class DivergenceBOS:
    """Reproduces the live strategy exactly, as the reference every candidate is judged
    against. If the harness cannot reproduce this strategy's known numbers
    (placebo delta +0.0377 R/trade, t=+4.66) then the harness is wrong."""
    confirm_lag = 3   # find_pivots(right=3): a scan bar is only reachable 3 bars later
    name = "C2_divergence_bos"
    MAX_WAIT = 12
    LOOKBACK = 50
    MIN_PIV_DIST = 3
    FRESHNESS = 10

    def prepare(self, df):
        return _prep_common(df)

    def detect(self, df, atr_mult=1.5, rr=5.0, types=("REG_BULL", "REG_BEAR",
                                                      "HID_BULL", "HID_BEAR")):
        c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
        rsi = df["rsi"].to_numpy(); ema = df["ema200"].to_numpy(); atr = df["atr"].to_numpy()
        ph, pl = _find_pivots(c, 3, 3)
        n = len(c)
        out = []
        used = set()
        start = max(210, self.LOOKBACK + 4)
        for i in range(start, n - 3):
            if not (np.isfinite(rsi[i]) and np.isfinite(ema[i])):
                continue
            bull = c[i] > ema[i]
            piv = pl if bull else ph
            found = []
            for j in range(i - 3, max(0, i - self.LOOKBACK), -1):
                if np.isfinite(piv[j]):
                    found.append((j, piv[j]))
                    if len(found) >= 2:
                        break
            if len(found) != 2:
                continue
            (ci, cv), (pi, pv) = found
            if (i - ci) > self.FRESHNESS or (ci - pi) < self.MIN_PIV_DIST:
                continue
            k = (ci, pi, bull)
            if k in used:
                continue
            if bull:
                if cv < pv and rsi[ci] > rsi[pi]:
                    t = "REG_BULL"
                elif cv > pv and rsi[ci] < rsi[pi]:
                    t = "HID_BULL"
                else:
                    continue
                side, lvl = "long", float(np.max(h[ci:i + 1]))
            else:
                if cv > pv and rsi[ci] < rsi[pi]:
                    t = "REG_BEAR"
                elif cv < pv and rsi[ci] > rsi[pi]:
                    t = "HID_BEAR"
                else:
                    continue
                side, lvl = "short", float(np.min(l[ci:i + 1]))
            if t not in types:
                continue
            used.add(k)
            # break of structure on a CLOSED bar, then entry next open
            for w in range(1, self.MAX_WAIT + 1):
                b = i + w
                if b >= n:
                    break
                if (side == "long" and c[b] > lvl) or (side == "short" and c[b] < lvl):
                    if not (np.isfinite(ema[b]) and np.isfinite(atr[b]) and atr[b] > 0):
                        break
                    if side == "long" and not c[b] > ema[b]:
                        break
                    if side == "short" and not c[b] < ema[b]:
                        break
                    out.append(Signal(b, side, float(atr[b] * atr_mult), float(rr),
                                      {"type": t}))
                    break
        return out


# ===========================================================================
# S1 — Donchian breakout (already shadow-running; best unexploited evidence)
# ===========================================================================
class Donchian:
    """+1011R / 2,327 trades / PF 1.53, positive in 12/15 purged walk-forward folds —
    the repo's second-best-validated idea, never deployed."""
    name = "S1_donchian"
    COOLDOWN = 12

    def prepare(self, df):
        return _prep_common(df)

    def detect(self, df, channel=96, atr_mult=1.5, rr=8.0):
        c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
        atr = df["atr"].to_numpy()
        n = len(c)
        # prior-window extremes, strictly excluding the current bar
        hh = pd.Series(h).shift(1).rolling(channel).max().to_numpy()
        ll = pd.Series(l).shift(1).rolling(channel).min().to_numpy()
        out = []
        last = {"long": -10**9, "short": -10**9}
        for i in range(max(210, channel + 2), n):
            if not (np.isfinite(hh[i]) and np.isfinite(ll[i]) and
                    np.isfinite(atr[i]) and atr[i] > 0):
                continue
            up = c[i] > hh[i]
            dn = c[i] < ll[i]
            prev_up = np.isfinite(hh[i - 1]) and c[i - 1] > hh[i - 1]
            prev_dn = np.isfinite(ll[i - 1]) and c[i - 1] < ll[i - 1]
            if up and not prev_up and i - last["long"] >= self.COOLDOWN:
                out.append(Signal(i, "long", float(atr[i] * atr_mult), float(rr),
                                  {"ch": channel}))
                last["long"] = i
            elif dn and not prev_dn and i - last["short"] >= self.COOLDOWN:
                out.append(Signal(i, "short", float(atr[i] * atr_mult), float(rr),
                                  {"ch": channel}))
                last["short"] = i
        return out


# ===========================================================================
# S2 — Double Divergence (best recorded R/DD in the repo, never re-tested cleanly)
# ===========================================================================
class DoubleDivergence:
    """RSI divergence that a second oscillator (OBV or MACD histogram) confirms on the
    same pivot pair. Recorded PF 1.93 / R-DD 14.2 on 2,075 trades — but 5m and in-sample."""
    confirm_lag = 3   # find_pivots(right=3): a scan bar is only reachable 3 bars later
    name = "S2_double_divergence"
    MAX_WAIT = 12
    LOOKBACK = 50
    FRESHNESS = 10

    def prepare(self, df):
        df = _prep_common(df)
        df["obv"] = _obv(df)
        df["macdh"] = _macd_hist(df["close"])
        return df

    def detect(self, df, atr_mult=1.5, rr=5.0, second="obv"):
        c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
        rsi = df["rsi"].to_numpy(); ema = df["ema200"].to_numpy(); atr = df["atr"].to_numpy()
        sec = df[second].to_numpy()
        ph, pl = _find_pivots(c, 3, 3)
        n = len(c)
        out = []
        used = set()
        for i in range(max(210, self.LOOKBACK + 4), n - 3):
            if not (np.isfinite(rsi[i]) and np.isfinite(ema[i]) and np.isfinite(sec[i])):
                continue
            bull = c[i] > ema[i]
            piv = pl if bull else ph
            found = []
            for j in range(i - 3, max(0, i - self.LOOKBACK), -1):
                if np.isfinite(piv[j]):
                    found.append((j, piv[j]))
                    if len(found) >= 2:
                        break
            if len(found) != 2:
                continue
            (ci, cv), (pi, pv) = found
            if (i - ci) > self.FRESHNESS or (ci - pi) < 3:
                continue
            k = (ci, pi, bull)
            if k in used:
                continue
            if bull:
                rsi_div = cv < pv and rsi[ci] > rsi[pi]
                sec_div = cv < pv and sec[ci] > sec[pi]
                side, lvl = "long", float(np.max(h[ci:i + 1]))
            else:
                rsi_div = cv > pv and rsi[ci] < rsi[pi]
                sec_div = cv > pv and sec[ci] < sec[pi]
                side, lvl = "short", float(np.min(l[ci:i + 1]))
            if not (rsi_div and sec_div):          # BOTH must diverge
                continue
            used.add(k)
            for w in range(1, self.MAX_WAIT + 1):
                b = i + w
                if b >= n:
                    break
                if (side == "long" and c[b] > lvl) or (side == "short" and c[b] < lvl):
                    if not (np.isfinite(atr[b]) and atr[b] > 0 and np.isfinite(ema[b])):
                        break
                    if (side == "long") != (c[b] > ema[b]):
                        break
                    out.append(Signal(b, side, float(atr[b] * atr_mult), float(rr),
                                      {"second": second}))
                    break
        return out


# ===========================================================================
# S3 — Boom Hunter Pro (Ehlers EOT) — coded in repo, NEVER scored
# ===========================================================================
class BoomHunter:
    """Ehlers Early Onset Trend. Completely orthogonal to everything else here — no RSI,
    no EMA, no VWAP. Implemented in the repo but no results file has ever existed."""
    name = "S3_boom_hunter"

    @staticmethod
    def _eot(close: np.ndarray, lp_period: int, k1: float):
        """Highpass -> SuperSmoother -> fast-attack/slow-decay normalisation."""
        n = len(close)
        a1 = math.exp(-1.414 * math.pi / lp_period)
        b1 = 2 * a1 * math.cos(1.414 * math.pi / lp_period)
        c2, c3 = b1, -a1 * a1
        c1 = 1 - c2 - c3
        hp = np.zeros(n)
        alpha = (math.cos(0.707 * 2 * math.pi / 100) + math.sin(0.707 * 2 * math.pi / 100) - 1) / \
                math.cos(0.707 * 2 * math.pi / 100)
        for i in range(2, n):
            hp[i] = ((1 - alpha / 2) ** 2 * (close[i] - 2 * close[i - 1] + close[i - 2])
                     + 2 * (1 - alpha) * hp[i - 1] - (1 - alpha) ** 2 * hp[i - 2])
        filt = np.zeros(n)
        for i in range(2, n):
            filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
        peak = np.zeros(n)
        for i in range(1, n):
            peak[i] = max(0.991 * peak[i - 1], abs(filt[i]))
        x = np.divide(filt, peak, out=np.zeros(n), where=peak > 0)
        q = np.zeros(n)
        for i in range(1, n):
            q[i] = (x[i] + k1) / (k1 * x[i] + 1)
        return q

    def prepare(self, df):
        df = _prep_common(df)
        c = df["close"].to_numpy(dtype=float)
        df["q1"] = self._eot(c, 6, 0.0)
        df["q3"] = self._eot(c, 27, 0.8)
        df["trig"] = pd.Series(df["q1"]).rolling(2).mean().to_numpy()
        return df

    def detect(self, df, atr_mult=1.5, rr=3.0, tier="all"):
        q1 = df["q1"].to_numpy(); q3 = df["q3"].to_numpy(); tg = df["trig"].to_numpy()
        atr = df["atr"].to_numpy()
        n = len(q1)
        out = []
        last = -10**9
        for i in range(210, n):
            if not (np.isfinite(tg[i]) and np.isfinite(tg[i - 1]) and
                    np.isfinite(atr[i]) and atr[i] > 0):
                continue
            if i - last < 12:
                continue
            cross_up = q1[i] > tg[i] and q1[i - 1] <= tg[i - 1]
            cross_dn = q1[i] < tg[i] and q1[i - 1] >= tg[i - 1]
            # 'lime' = the strictest published tier: deeply oversold red wave
            strict_long = cross_up and q3[i] <= -0.9
            strict_short = cross_dn and q3[i] >= 0.9
            if tier == "lime":
                go_long, go_short = strict_long, strict_short
            elif tier == "green":
                go_long, go_short = cross_up and q3[i] <= -0.5, cross_dn and q3[i] >= 0.5
            else:
                go_long, go_short = cross_up, cross_dn
            if go_long:
                out.append(Signal(i, "long", float(atr[i] * atr_mult), float(rr),
                                  {"tier": tier}))
                last = i
            elif go_short:
                out.append(Signal(i, "short", float(atr[i] * atr_mult), float(rr),
                                  {"tier": tier}))
                last = i
        return out


# ===========================================================================
# S4 — EMA-stack pullback (best non-divergence trend idea in the repo)
# ===========================================================================
class EMAStackPullback:
    name = "S4_ema_stack_pullback"

    def prepare(self, df):
        df = _prep_common(df)
        df["ema20"] = _ema(df["close"], 20)
        df["ema50"] = _ema(df["close"], 50)
        return df

    def detect(self, df, rsi_trigger=35, atr_mult=1.5, rr=4.5):
        c = df["close"].to_numpy(); rsi = df["rsi"].to_numpy(); atr = df["atr"].to_numpy()
        e20 = df["ema20"].to_numpy(); e50 = df["ema50"].to_numpy()
        e200 = df["ema200"].to_numpy()
        n = len(c)
        out = []
        last = -10**9
        for i in range(210, n):
            if not all(np.isfinite(v[i]) for v in (rsi, atr, e20, e50, e200)) or atr[i] <= 0:
                continue
            if i - last < 12:
                continue
            up = e20[i] > e50[i] > e200[i]
            dn = e20[i] < e50[i] < e200[i]
            if up and rsi[i] <= rsi_trigger and rsi[i - 1] > rsi_trigger:
                out.append(Signal(i, "long", float(atr[i] * atr_mult), float(rr), {}))
                last = i
            elif dn and rsi[i] >= (100 - rsi_trigger) and rsi[i - 1] < (100 - rsi_trigger):
                out.append(Signal(i, "short", float(atr[i] * atr_mult), float(rr), {}))
                last = i
        return out


# ===========================================================================
# S8/S9/S10 — single-oscillator divergence variants (swap RSI for something else)
# ===========================================================================
class OscillatorDivergence:
    """Same pivot/BOS skeleton, different oscillator. Isolates whether RSI specifically
    carries the information, or whether any momentum oscillator would do."""
    confirm_lag = 3   # find_pivots(right=3): a scan bar is only reachable 3 bars later
    MAX_WAIT = 12
    LOOKBACK = 50
    FRESHNESS = 10

    def __init__(self, osc: str):
        self.osc = osc
        self.name = {"macdh": "S8_macd_div", "obv": "S9_obv_div",
                     "stoch": "S10_stoch_div"}[osc]

    def prepare(self, df):
        df = _prep_common(df)
        df["macdh"] = _macd_hist(df["close"])
        df["obv"] = _obv(df)
        df["stoch"] = _stoch_k(df)
        return df

    def detect(self, df, atr_mult=1.5, rr=5.0):
        c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
        o = df[self.osc].to_numpy(); ema = df["ema200"].to_numpy(); atr = df["atr"].to_numpy()
        ph, pl = _find_pivots(c, 3, 3)
        n = len(c)
        out = []
        used = set()
        for i in range(max(210, self.LOOKBACK + 4), n - 3):
            if not (np.isfinite(o[i]) and np.isfinite(ema[i])):
                continue
            bull = c[i] > ema[i]
            piv = pl if bull else ph
            found = []
            for j in range(i - 3, max(0, i - self.LOOKBACK), -1):
                if np.isfinite(piv[j]):
                    found.append((j, piv[j]))
                    if len(found) >= 2:
                        break
            if len(found) != 2:
                continue
            (ci, cv), (pi, pv) = found
            if (i - ci) > self.FRESHNESS or (ci - pi) < 3:
                continue
            k = (ci, pi, bull)
            if k in used or not (np.isfinite(o[ci]) and np.isfinite(o[pi])):
                continue
            if bull:
                ok = (cv < pv and o[ci] > o[pi]) or (cv > pv and o[ci] < o[pi])
                side, lvl = "long", float(np.max(h[ci:i + 1]))
            else:
                ok = (cv > pv and o[ci] < o[pi]) or (cv < pv and o[ci] > o[pi])
                side, lvl = "short", float(np.min(l[ci:i + 1]))
            if not ok:
                continue
            used.add(k)
            for w in range(1, self.MAX_WAIT + 1):
                b = i + w
                if b >= n:
                    break
                if (side == "long" and c[b] > lvl) or (side == "short" and c[b] < lvl):
                    if not (np.isfinite(atr[b]) and atr[b] > 0 and np.isfinite(ema[b])):
                        break
                    if (side == "long") != (c[b] > ema[b]):
                        break
                    out.append(Signal(b, side, float(atr[b] * atr_mult), float(rr),
                                      {"osc": self.osc}))
                    break
        return out


REGISTRY = {
    "C2_divergence_bos": DivergenceBOS,
    "S1_donchian": Donchian,
    "S2_double_divergence": DoubleDivergence,
    "S3_boom_hunter": BoomHunter,
    "S4_ema_stack_pullback": EMAStackPullback,
}
