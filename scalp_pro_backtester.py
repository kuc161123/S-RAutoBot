"""
Scalp Pro-Rules Backtester

This script replays the 3m Scalp strategy over historical candles and then
applies the Pro rules layer (RSI/MACD/VWAP/Fib/MTF/volume) to each signal.

It reports:
  - Raw Scalp signal stats (all detections)
  - Pro-accepted signal stats (what would execute live under Pro rules)
  - Pro-rejected signal stats (what Pro rules filtered out)

Usage (from repo root):
  python scalp_pro_backtester.py --symbol BTCUSDT --limit 20000
"""

import argparse
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from candle_storage_postgres import CandleStorage
from strategy_scalp import detect_scalp_signal, ScalpSettings, ScalpSignal


logger = logging.getLogger(__name__)


def _simulate_trade(df: pd.DataFrame, entry_index: int, signal: ScalpSignal, max_lookahead: int = 500) -> str:
    """Simulate trade outcome on 3m data (win/loss/no_outcome)."""
    for i in range(entry_index + 1, min(entry_index + max_lookahead, len(df))):
        future_high = float(df["high"].iloc[i])
        future_low = float(df["low"].iloc[i])
        if signal.side == "long":
            if future_high >= signal.tp:
                return "win"
            if future_low <= signal.sl:
                return "loss"
        else:
            if future_low <= signal.tp:
                return "win"
            if future_high >= signal.sl:
                return "loss"
    return "no_outcome"


def _compute_rsi_14(close: pd.Series) -> float:
    try:
        if len(close) < 15:
            return 50.0
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(max(0.0, min(100.0, rsi.iloc[-1])))
    except Exception:
        return 50.0


def _compute_macd_hist(close: pd.Series) -> float:
    try:
        if len(close) < 35:
            return 0.0
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(hist.iloc[-1])
    except Exception:
        return 0.0


def _compute_fib_zone(high: pd.Series, low: pd.Series, price: float) -> Tuple[float, str]:
    """Return (fib_ret, fib_zone) for last swing over ~50 bars, like live."""
    try:
        if len(high) == 0 or len(low) == 0:
            return 0.5, "38-50"
        hh = float(high.tail(50).max())
        ll = float(low.tail(50).min())
        span = max(1e-9, hh - ll)
        fib_ret = float((hh - price) / span)  # 0 at HH, 1 at LL
        fr = fib_ret * 100.0
        zone = (
            "0-23"
            if fr < 23.6
            else "23-38"
            if fr < 38.2
            else "38-50"
            if fr < 50.0
            else "50-61"
            if fr < 61.8
            else "61-78"
            if fr < 78.6
            else "78-100"
        )
        return max(0.0, min(1.0, fib_ret)), zone
    except Exception:
        return 0.5, "38-50"


def _compute_vwap_dist_atr(tail: pd.DataFrame, atr: float) -> float:
    """Compute EVWAP-based distance to price in ATR units, similar to live."""
    try:
        close = tail["close"]
        tp = (tail["high"] + tail["low"] + tail["close"]) / 3.0
        vol = tail["volume"].clip(lower=0.0)
        vwap = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
        if len(vwap.dropna()) == 0 or atr <= 0:
            return 0.0
        return float(abs(close.iloc[-1] - vwap.iloc[-1]) / max(1e-9, atr))
    except Exception:
        return 0.0


def _compute_atr_14(df: pd.DataFrame) -> float:
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev).abs(),
                (low - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        if len(tr) < 14:
            return float(tr.iloc[-1])
        return float(tr.rolling(14).mean().iloc[-1])
    except Exception:
        return 0.0


def _compute_volume_ratio(tail: pd.DataFrame) -> float:
    try:
        vol = tail["volume"]
        if len(vol) >= 20 and vol.rolling(20).mean().iloc[-1] > 0:
            return float(vol.iloc[-1] / max(1e-9, vol.rolling(20).mean().iloc[-1]))
        return 1.0
    except Exception:
        return 1.0


def _compute_ema_dir_15m(df3: pd.DataFrame) -> str:
    """Approximate 15m EMA direction using resampled 3m data (20/50 EMAs)."""
    try:
        if df3 is None or df3.empty:
            return "none"
        # Resample to 15m OHLC for multi-timeframe context
        df15 = df3.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        # Require enough history for longer EMAs
        if len(df15) < 250:
            return "none"
        ema_fast = df15["close"].ewm(span=100, adjust=False).mean()
        ema_slow = df15["close"].ewm(span=250, adjust=False).mean()
        last_fast = float(ema_fast.iloc[-1])
        last_slow = float(ema_slow.iloc[-1])
        if last_fast > last_slow:
            return "up"
        if last_fast < last_slow:
            return "down"
        return "none"
    except Exception:
        return "none"


def build_pro_features(df3: pd.DataFrame, idx: int, signal: ScalpSignal) -> Dict:
    """
    Build the subset of features used by Pro rules for a given signal.

    Uses a 50-bar 3m window ending at index idx, plus 15m EMA direction from
    resampled 3m data.
    """
    # Use history up to the signal bar
    df_hist = df3.iloc[: idx + 1]
    tail = df_hist.tail(50).copy()
    if len(tail) < 20:
        return {}

    close = tail["close"]
    high = tail["high"]
    low = tail["low"]

    price = float(close.iloc[-1])

    atr = _compute_atr_14(df_hist)
    rsi = _compute_rsi_14(close)
    macd_hist = _compute_macd_hist(close)
    fib_ret, fib_zone = _compute_fib_zone(high, low, price)
    vwap_dist_atr = _compute_vwap_dist_atr(tail, atr)
    vol_ratio = _compute_volume_ratio(tail)

    ema_dir_15m = _compute_ema_dir_15m(df_hist)
    side = str(signal.side).lower()
    mtf_agree_15 = bool(
        (side == "long" and ema_dir_15m == "up")
        or (side == "short" and ema_dir_15m == "down")
    )

    return {
        "rsi_14": rsi,
        "macd_hist": macd_hist,
        "vwap_dist_atr": vwap_dist_atr,
        "fib_ret": fib_ret,
        "fib_zone": fib_zone,
        "mtf_agree_15": mtf_agree_15,
        "volume_ratio": vol_ratio,
    }


def pro_rules_allow(side: str, feats: Dict) -> Tuple[bool, str]:
    """
    Apply Pro rules similar to live_bot._scalp_combo_allowed (fallback_mode='pro').

    Uses RSI/MACD/VWAP/Fib/MTF bins, with mtf_agree_15 required and
    MTF direction provided by _compute_ema_dir_15m (100/250 EMAs on 15m).
    Returns (allowed, reason).
    """
    s = str(side).lower()
    try:
        rsi = float(feats.get("rsi_14", 0.0) or 0.0)
    except Exception:
        rsi = 0.0
    try:
        mh = float(feats.get("macd_hist", 0.0) or 0.0)
    except Exception:
        mh = 0.0
    try:
        vwap = float(feats.get("vwap_dist_atr", 0.0) or 0.0)
    except Exception:
        vwap = 0.0
    fibz = feats.get("fib_zone")
    mtf = bool(feats.get("mtf_agree_15", False))
    try:
        vol = float(feats.get("volume_ratio", 0.0) or 0.0)
    except Exception:
        vol = 0.0

    # Derive fib zone if missing from fib_ret
    if not isinstance(fibz, str) or not fibz:
        try:
            frel = float(feats.get("fib_ret"))
            fr = frel * 100.0 if frel <= 1.0 else frel
            fibz = (
                "0-23"
                if fr < 23.6
                else "23-38"
                if fr < 38.2
                else "38-50"
                if fr < 50.0
                else "50-61"
                if fr < 61.8
                else "61-78"
                if fr < 78.6
                else "78-100"
            )
        except Exception:
            fibz = None

    # Derive RSI / VWAP bins for readability
    rsi_bin = (
        "<30"
        if rsi < 30
        else "30-40"
        if rsi < 40
        else "40-50"
        if rsi < 50
        else "50-60"
        if rsi < 60
        else "60-70"
        if rsi < 70
        else "70+"
    )
    macd_state = "bull" if mh > 0 else "bear"
    mh_abs = abs(mh)
    mh_floor = 0.0005
    vwap_bin = (
        "<0.6"
        if vwap < 0.6
        else "0.6-1.0"
        if vwap < 1.0
        else "1.0-1.2"
        if vwap < 1.2
        else "1.2+"
    )
    vol_strong = vol >= 1.50

    if s == "long":
        # RSI: 40–70 (trend strength) OR 30–40 (pullback).
        rsi_ok = (40.0 <= rsi < 70.0) or (30.0 <= rsi < 40.0)
        # VWAP: <0.6, 0.6–1.0, or 1.0–1.2 with strong volume.
        v_ok = (vwap_bin == "<0.6") or (vwap_bin == "0.6-1.0") or (
            vwap_bin == "1.0-1.2" and vol_strong
        )
        # MACD: bullish with minimum hist strength.
        macd_ok = (macd_state == "bull" and mh_abs >= mh_floor)
        # Fib: shallow + golden zones (23–61).
        fib_ok = fibz in ("23-38", "38-50", "50-61")
        ok = bool(rsi_ok and v_ok and macd_ok and fib_ok and mtf)
        reason = (
            "ok"
            if ok
            else f"block_long rsi={rsi_bin} vwap={vwap_bin} fib={fibz or 'n/a'} mtf={mtf}"
        )
        return ok, reason

    # SHORT rules
    # RSI: 30–60, or 60–70 with strong volume.
    rsi_ok = (30.0 <= rsi < 60.0) or (60.0 <= rsi < 70.0 and vol_strong)
    # MACD: bearish with minimum hist strength.
    macd_ok = (macd_state == "bear" and mh_abs >= mh_floor)
    # VWAP: allow <0.6, 0.6–1.0, 1.0–1.2; block 1.2+.
    v_ok = vwap_bin in ("<0.6", "0.6-1.0", "1.0-1.2")
    # Fib: 23–78 for shorts.
    fib_ok = fibz in ("23-38", "38-50", "50-61", "61-78")
    ok = bool(rsi_ok and macd_ok and v_ok and fib_ok and mtf)
    reason = (
        "ok"
        if ok
        else f"block_short rsi={rsi_bin} vwap={vwap_bin} fib={fibz or 'n/a'} mtf={mtf}"
    )
    return ok, reason


def backtest_pro_rules(symbol: str, limit: Optional[int] = None, settings: Optional[ScalpSettings] = None) -> Dict:
    storage = CandleStorage()
    df3 = storage.load_candles_3m(symbol, limit=limit)
    if df3 is None or df3.empty:
        logger.error(f"[{symbol}] No 3m candles available for backtest")
        return {}

    df3 = df3.copy()
    df3 = df3[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"[{symbol}] Loaded {len(df3)} 3m candles for Pro-rules backtest")

    if settings is None:
        settings = ScalpSettings()

    start = max(60, 200)  # warm-up for indicators

    stats = {
        "raw_signals": 0,
        "raw_wins": 0,
        "raw_losses": 0,
        "pro_accept": 0,
        "pro_accept_wins": 0,
        "pro_accept_losses": 0,
        "pro_reject": 0,
        "pro_reject_wins": 0,
        "pro_reject_losses": 0,
    }

    for i in range(start, len(df3) - 1):
        df_slice = df3.iloc[: i + 1]
        sig = detect_scalp_signal(df_slice.copy(), settings, symbol)
        if not isinstance(sig, ScalpSignal):
            continue

        side = str(sig.side).lower()
        if side not in ("long", "short"):
            continue

        feats = build_pro_features(df_slice, i, sig)
        if not feats:
            continue

        outcome = _simulate_trade(df3, i, sig)

        stats["raw_signals"] += 1
        if outcome == "win":
            stats["raw_wins"] += 1
        elif outcome == "loss":
            stats["raw_losses"] += 1

        allowed, _ = pro_rules_allow(side, feats)
        if allowed:
            stats["pro_accept"] += 1
            if outcome == "win":
                stats["pro_accept_wins"] += 1
            elif outcome == "loss":
                stats["pro_accept_losses"] += 1
        else:
            stats["pro_reject"] += 1
            if outcome == "win":
                stats["pro_reject_wins"] += 1
            elif outcome == "loss":
                stats["pro_reject_losses"] += 1

    def _wr(w: int, l: int) -> float:
        total = w + l
        return (w / total * 100.0) if total > 0 else 0.0

    stats["raw_wr"] = _wr(stats["raw_wins"], stats["raw_losses"])
    stats["pro_accept_wr"] = _wr(
        stats["pro_accept_wins"], stats["pro_accept_losses"]
    )
    stats["pro_reject_wr"] = _wr(
        stats["pro_reject_wins"], stats["pro_reject_losses"]
    )

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Backtest Scalp strategy Pro rules on historical 3m data."
    )
    parser.add_argument(
        "--symbol",
        required=True,
        nargs="+",
        help="Symbol(s) to backtest (e.g. BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Max number of 3m candles to load (newest first).",
    )
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbol]
    for symbol in symbols:
        stats = backtest_pro_rules(symbol, limit=args.limit)
        if not stats:
            continue

        print(f"=== Scalp Pro-Rules Backtest: {symbol} ===")
        print(f"3m candles used: ~{args.limit} (or available)")
        print("")
        print(f"Raw signals: {stats['raw_signals']}")
        print(
            f"  Raw WR: {stats['raw_wr']:.1f}% "
            f"(wins={stats['raw_wins']}, losses={stats['raw_losses']})"
        )
        print("")
        print(f"Pro-accepted signals: {stats['pro_accept']}")
        print(
            f"  Pro-accept WR: {stats['pro_accept_wr']:.1f}% "
            f"(wins={stats['pro_accept_wins']}, losses={stats['pro_accept_losses']})"
        )
        print("")
        print(f"Pro-rejected signals: {stats['pro_reject']}")
        print(
            f"  Pro-reject WR (what was filtered): {stats['pro_reject_wr']:.1f}% "
            f"(wins={stats['pro_reject_wins']}, losses={stats['pro_reject_losses']})"
        )
        print("")


if __name__ == "__main__":
    main()
