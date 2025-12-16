#!/usr/bin/env python3
"""
Hidden Bearish Divergence - LIVE-MATCHING Backtest (candle-by-candle)
====================================================================
This backtest is designed to match the CURRENT live bot behavior in:
- `autobot/core/bot.py` (cooldown, volume gate, hidden_bearish_only, pivot SL w/ ATR clamps, 3R)
- `autobot/core/divergence_detector.py` (signal detection + indicators)

Key "live matching" behaviors:
- Detect signals on CLOSED candles only (no lookahead beyond the candle you're on).
- Cooldown is time-based in live code: COOLDOWN_BARS=10 with CANDLE_MINUTES hardcoded to 15
  => 2.5 hours between signals per symbol.
- Order placement: LIMIT at signal close (expected_entry = signal close).
- Pending order: 5 minute timeout (live monitor_pending_limit_orders); during pending:
  - If price breaches SL before fill -> cancel and record theoretical LOSS
  - If price breaches TP before fill -> cancel and record theoretical WIN (missed entry)
  - Else, if not filled before timeout -> NO_FILL
- If filled, simulate TP/SL candle-by-candle with SL checked first.

Important: With only 1H candles, a 5-minute pending window cannot be represented faithfully.
To address this, the backtest can run in "high fidelity" mode using 3-minute candles for:
- Pending fill window (5 minutes)
- TP/SL resolution after fill (optional; default uses 3m as well to match live candle_data usage)

Usage examples:
  python backtest_hidden_bearish_live_match.py --days 60 --end-ts 1734316800000
  python backtest_hidden_bearish_live_match.py --high-fidelity --days 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

from autobot.core.divergence_detector import detect_divergence, prepare_dataframe

BASE_URL = "https://api.bybit.com"


@dataclass
class TradeRecord:
    symbol: str
    side: str
    signal_type: str
    signal_time: str
    entry_time: Optional[str]
    exit_time: Optional[str]
    outcome: str  # win|loss|timeout|no_fill
    entry: float
    sl: float
    tp: float
    exit_price: float
    bars_held: int
    r_pnl: float
    notes: str = ""


def _utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def _cache_key(symbol: str, interval: str, start_ms: int, end_ms: int) -> str:
    return f"{symbol}__{interval}__{start_ms}__{end_ms}.json"


def fetch_klines_public(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Optional[Path] = None,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """
    Fetch Bybit v5 klines for [start_ms, end_ms] inclusive.
    Uses public endpoint (no auth). Bybit returns newest-first lists; we normalize to ascending.
    """
    if cache_dir:
        _ensure_cache_dir(cache_dir)
        key = _cache_key(symbol, interval, start_ms, end_ms)
        cached_path = cache_dir / key
        if cached_path.exists():
            try:
                with open(cached_path, "r") as f:
                    payload = json.load(f)
                return _klines_payload_to_df(payload)
            except Exception:
                pass

    all_rows: List[list] = []
    cursor_end = end_ms

    # Bybit caps at 200 per request for kline
    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": 200,
            "end": cursor_end,
        }
        resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
        j = resp.json()
        data = (j.get("result") or {}).get("list") or []
        if not data:
            break

        all_rows.extend(data)

        # Oldest timestamp in this returned page (because newest-first)
        oldest_ts = int(data[-1][0])
        # Stop if we've crossed start_ms
        if oldest_ts <= start_ms:
            break
        cursor_end = oldest_ts - 1

        if sleep_s:
            import time

            time.sleep(sleep_s)

        # Safety: if API returns < limit, no more pages
        if len(data) < 200:
            break

    payload = {"symbol": symbol, "interval": interval, "start_ms": start_ms, "end_ms": end_ms, "rows": all_rows}

    df = _klines_payload_to_df(payload)
    # Filter to requested window, then ensure enough ordering
    df = df[(df.index_ms >= start_ms) & (df.index_ms <= end_ms)].copy()
    df.sort_values("index_ms", inplace=True)
    df.set_index("start", inplace=True)

    if cache_dir:
        try:
            with open(cached_path, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass

    return df


def _klines_payload_to_df(payload: dict) -> pd.DataFrame:
    rows = payload.get("rows") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["start", "open", "high", "low", "close", "volume", "turnover"])
    df["start"] = pd.to_datetime(df["start"].astype(int), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = df[c].astype(float)
    df["index_ms"] = df["start"].astype("int64") // 1_000_000
    return df


def _calc_pivot_sltp_like_live(
    df_closed: pd.DataFrame,
    side: str,
    signal_price: float,
    atr: float,
    signal_swing_low: Optional[float] = None,
    signal_swing_high: Optional[float] = None,
    rr_ratio: float = 3.0,
    swing_lookback: int = 15,
    min_sl_atr: float = 0.3,
    max_sl_atr: float = 2.0,
) -> Optional[Tuple[float, float]]:
    """
    Mirrors the pivot SL logic in `DivergenceBot.execute_divergence_trade()` but without
    exchange tick rounding (backtest operates on raw prices).
    Returns (sl, tp) or None if invalid (SL wrong side).
    """
    expected_entry = float(signal_price)

    if signal_swing_low is not None and signal_swing_high is not None:
        swing_low_val = float(signal_swing_low)
        swing_high_val = float(signal_swing_high)
        constraint_atr = float(atr)
    else:
        swing_low_val = float(df_closed["low"].tail(swing_lookback).min())
        swing_high_val = float(df_closed["high"].tail(swing_lookback).max())
        constraint_atr = float(atr)

    if side == "long":
        sl = swing_low_val
        if sl >= expected_entry:
            return None
        sl_distance = expected_entry - sl
    else:
        sl = swing_high_val
        if sl <= expected_entry:
            return None
        sl_distance = sl - expected_entry

    min_sl_dist = min_sl_atr * constraint_atr
    max_sl_dist = max_sl_atr * constraint_atr
    if sl_distance < min_sl_dist:
        sl_distance = min_sl_dist
        sl = expected_entry - sl_distance if side == "long" else expected_entry + sl_distance
    elif sl_distance > max_sl_dist:
        sl_distance = max_sl_dist
        sl = expected_entry - sl_distance if side == "long" else expected_entry + sl_distance

    tp = expected_entry + (rr_ratio * sl_distance) if side == "long" else expected_entry - (rr_ratio * sl_distance)
    return float(sl), float(tp)


def _simulate_pending_window_and_fill(
    micro: pd.DataFrame,
    start_ms: int,
    end_ms: int,
    side: str,
    entry: float,
    sl: float,
    tp: float,
) -> Tuple[str, Optional[int]]:
    """
    Simulate pending window [start_ms, end_ms] inclusive on micro candles.
    Returns:
      - status: filled|no_fill|cancel_loss|cancel_win
      - fill_ms: timestamp ms when filled (if filled)

    Mirrors live pending behavior:
    - FILL if price touches entry (long: low<=entry, short: high>=entry)
    - CANCEL LOSS if SL breached before fill (long: price<=sl, short: price>=sl) using close
    - CANCEL WIN if TP breached before fill (long: price>=tp, short: price<=tp) using close
    - If window ends w/o fill => no_fill
    """
    window = micro[(micro.index_ms >= start_ms) & (micro.index_ms <= end_ms)].copy()
    if window.empty:
        return "no_fill", None

    for _, row in window.iterrows():
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        ts = int(row["index_ms"])

        # Invalidation checks use "current_price" (close) in live code
        if side == "long":
            if close <= sl:
                return "cancel_loss", None
            if close >= tp:
                return "cancel_win", None
            # Fill check uses touched range
            if low <= entry:
                return "filled", ts
        else:
            if close >= sl:
                return "cancel_loss", None
            if close <= tp:
                return "cancel_win", None
            if high >= entry:
                return "filled", ts

    return "no_fill", None


def _simulate_position_outcome(
    micro: pd.DataFrame,
    entry_ms: int,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    max_hold_ms: int,
) -> Tuple[str, float, int, int]:
    """
    After fill, simulate TP/SL candle-by-candle on micro candles.
    SL checked FIRST (conservative) to match your live/learner behavior.
    Returns: (outcome, exit_price, bars_held)
    """
    end_ms = entry_ms + max_hold_ms
    window = micro[(micro.index_ms > entry_ms) & (micro.index_ms <= end_ms)].copy()
    if window.empty:
        return "timeout", entry, 0, end_ms

    bars = 0
    for _, row in window.iterrows():
        bars += 1
        high = float(row["high"])
        low = float(row["low"])
        ts = int(row["index_ms"])
        if side == "long":
            if low <= sl:
                return "loss", sl, bars, ts
            if high >= tp:
                return "win", tp, bars, ts
        else:
            if high >= sl:
                return "loss", sl, bars, ts
            if low <= tp:
                return "win", tp, bars, ts

    return "timeout", entry, bars, end_ms


def _r_pnl(outcome: str, rr_ratio: float = 3.0) -> float:
    if outcome == "win":
        return rr_ratio
    if outcome == "loss":
        return -1.0
    return 0.0


def run_backtest(
    days: int,
    end_ts_ms: int,
    timeframe_minutes: int,
    high_fidelity: bool,
    micro_interval: str,
    pending_minutes: int,
    max_hold_hours: int,
    cache_dir: Optional[Path],
    only_hidden_bearish: bool = True,
    progress: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    cfg = _load_config("config.yaml")
    symbols = cfg.get("trade", {}).get("divergence_symbols") or []
    if not symbols:
        raise RuntimeError("No trade.divergence_symbols found in config.yaml")

    # Live: cooldown time-based due to hardcoded 15m constant
    COOLDOWN_SECONDS = 10 * 15 * 60  # 2.5 hours

    end_dt = datetime.fromtimestamp(end_ts_ms / 1000, tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ts_ms = _utc_ms(start_dt)

    tf_ms = timeframe_minutes * 60 * 1000
    pending_window_ms = pending_minutes * 60 * 1000
    max_hold_ms = max_hold_hours * 60 * 60 * 1000

    records: List[TradeRecord] = []

    # Per symbol state (matches live: single pending/active per symbol)
    last_signal_time_ms: Dict[str, int] = {}

    for sym in symbols:
        if progress:
            print(f"[{sym}] loading candles...", flush=True)
        # Fetch main timeframe candles
        df_tf = fetch_klines_public(sym, str(timeframe_minutes), start_ts_ms, end_ts_ms, cache_dir=cache_dir)
        if df_tf.empty or len(df_tf) < 80:
            continue

        # Prepare dataframe on full history for convenience; we still slice candle-by-candle to avoid lookahead
        # NOTE: prepare_dataframe drops NaNs; that shortens the dataframe. We'll handle by iterating on the prepared df.
        df_prep = prepare_dataframe(df_tf[["open", "high", "low", "close", "volume", "turnover"]].copy())
        if df_prep.empty or len(df_prep) < 60:
            continue

        # Micro data for fill/exit simulation
        micro_df: Optional[pd.DataFrame] = None
        if high_fidelity:
            if progress:
                print(f"[{sym}] loading micro candles ({micro_interval}m)...", flush=True)
            micro_df = fetch_klines_public(sym, micro_interval, start_ts_ms, end_ts_ms, cache_dir=cache_dir)
            if micro_df.empty:
                # Without micro candles, we can't model 5-min pending; skip symbol
                continue
            micro_df.sort_values("index_ms", inplace=True)

        # Block signals while a trade is pending or active (matches live bot)
        blocked_until_ms: int = 0

        # Candle-by-candle scan: only evaluate the "current" candle as the last row of the slice
        for i in range(55, len(df_prep)):
            slice_df = df_prep.iloc[: i + 1].copy()
            last_row = slice_df.iloc[-1]
            t = slice_df.index[-1]
            t_ms = int(pd.Timestamp(t).value // 1_000_000)

            if blocked_until_ms and t_ms < blocked_until_ms:
                continue

            # Cooldown (time-based)
            last_ms = last_signal_time_ms.get(sym)
            if last_ms is not None and (t_ms - last_ms) < COOLDOWN_SECONDS * 1000:
                continue

            # Detect divergence on the slice (detect_divergence checks only latest bar)
            signals = detect_divergence(slice_df, sym)
            if not signals:
                continue

            # Only consider hidden_bearish signals (your mode)
            for sig in signals:
                if only_hidden_bearish and sig.signal_type != "hidden_bearish":
                    continue
                if sig.side != "short":
                    continue

                # Volume gate (same columns as live)
                if "vol_ok" in last_row and not bool(last_row["vol_ok"]):
                    continue

                atr = float(last_row["atr"])
                if math.isnan(atr) or atr <= 0:
                    continue

                signal_price = float(last_row["close"])

                # Use signal-time swing values (live does this at signal time)
                swing_lookback = 15
                swing_low = float(slice_df["low"].tail(swing_lookback).min())
                swing_high = float(slice_df["high"].tail(swing_lookback).max())

                sltp = _calc_pivot_sltp_like_live(
                    df_closed=slice_df,
                    side="short",
                    signal_price=signal_price,
                    atr=atr,
                    signal_swing_low=swing_low,
                    signal_swing_high=swing_high,
                )
                if not sltp:
                    # Invalid SL side
                    last_signal_time_ms[sym] = t_ms  # live still starts cooldown once queued
                    continue

                sl, tp = sltp
                entry = signal_price

                # Start cooldown at signal time (matches live: set when queued)
                last_signal_time_ms[sym] = t_ms

                # Pending fill + exit simulation
                outcome = "no_fill"
                entry_time = None
                exit_time = None
                exit_price = 0.0
                bars_held = 0
                notes = ""
                end_state_ms = t_ms  # when we're free to take next signal

                if not high_fidelity or micro_df is None:
                    # Coarse fallback: assume fill if next TF candle touches entry (overestimates fills vs 5m window).
                    # Then resolve outcome using TF candles with SL-first up to max_hold_hours.
                    if i + 1 >= len(df_prep):
                        continue
                    next_row = df_prep.iloc[i + 1]
                    filled = float(next_row["high"]) >= entry
                    if not filled:
                        outcome = "no_fill"
                        exit_price = float(next_row["close"])
                        exit_time = str(df_prep.index[i + 1])
                        notes = "coarse_no_fill_next_candle"
                        end_state_ms = int(pd.Timestamp(df_prep.index[i + 1]).value // 1_000_000)
                    else:
                        entry_time = str(df_prep.index[i + 1])
                        # Resolve on subsequent TF candles
                        max_bars = int(max_hold_hours * 60 / timeframe_minutes)
                        for k in range(1, max_bars + 1):
                            j = i + 1 + k
                            if j >= len(df_prep):
                                outcome = "timeout"
                                exit_price = entry
                                bars_held = k
                                exit_time = str(df_prep.index[-1])
                                end_state_ms = int(pd.Timestamp(df_prep.index[-1]).value // 1_000_000)
                                break
                            row = df_prep.iloc[j]
                            bars_held = k
                            high = float(row["high"])
                            low = float(row["low"])
                            if high >= sl:
                                outcome = "loss"
                                exit_price = sl
                                exit_time = str(df_prep.index[j])
                                end_state_ms = int(pd.Timestamp(df_prep.index[j]).value // 1_000_000)
                                break
                            if low <= tp:
                                outcome = "win"
                                exit_price = tp
                                exit_time = str(df_prep.index[j])
                                end_state_ms = int(pd.Timestamp(df_prep.index[j]).value // 1_000_000)
                                break
                        if outcome == "no_fill":
                            outcome = "timeout"
                            exit_price = entry
                            notes = "coarse_timeout"
                else:
                    # High fidelity: use micro candles to simulate the 5-minute pending order window and exits.
                    pending_start = t_ms
                    pending_end = t_ms + pending_window_ms
                    pending_status, fill_ms = _simulate_pending_window_and_fill(
                        micro=micro_df,
                        start_ms=pending_start,
                        end_ms=pending_end,
                        side="short",
                        entry=entry,
                        sl=sl,
                        tp=tp,
                    )
                    if pending_status == "filled" and fill_ms is not None:
                        entry_time = str(datetime.fromtimestamp(fill_ms / 1000, tz=timezone.utc))
                        outcome, exit_price, bars_held, exit_ms = _simulate_position_outcome(
                            micro=micro_df,
                            entry_ms=fill_ms,
                            side="short",
                            entry=entry,
                            sl=sl,
                            tp=tp,
                            max_hold_ms=max_hold_ms,
                        )
                        exit_time = str(datetime.fromtimestamp(exit_ms / 1000, tz=timezone.utc))
                        end_state_ms = exit_ms
                    elif pending_status == "cancel_loss":
                        outcome = "loss"
                        exit_price = sl
                        exit_time = str(datetime.fromtimestamp(pending_end / 1000, tz=timezone.utc))
                        notes = "pending_invalidation_sl"
                        end_state_ms = pending_end
                    elif pending_status == "cancel_win":
                        outcome = "win"
                        exit_price = tp
                        exit_time = str(datetime.fromtimestamp(pending_end / 1000, tz=timezone.utc))
                        notes = "pending_invalidation_tp"
                        end_state_ms = pending_end
                    else:
                        outcome = "no_fill"
                        # Use last close in window as reference
                        window = micro_df[(micro_df.index_ms >= pending_start) & (micro_df.index_ms <= pending_end)]
                        exit_price = float(window["close"].iloc[-1]) if not window.empty else entry
                        exit_time = str(datetime.fromtimestamp(pending_end / 1000, tz=timezone.utc))
                        notes = "pending_timeout_no_fill"
                        end_state_ms = pending_end

                rec = TradeRecord(
                    symbol=sym,
                    side="short",
                    signal_type=sig.signal_type,
                    signal_time=str(t),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    outcome=outcome,
                    entry=float(entry),
                    sl=float(sl),
                    tp=float(tp),
                    exit_price=float(exit_price),
                    bars_held=int(bars_held),
                    r_pnl=float(_r_pnl(outcome, rr_ratio=3.0)),
                    notes=notes,
                )
                records.append(rec)

                # Block further signals on this symbol until we're out of pending/position
                blocked_until_ms = max(blocked_until_ms, int(end_state_ms))

        if progress:
            sym_trades = sum(1 for r in records if r.symbol == sym)
            print(f"[{sym}] done. trades_recorded={sym_trades}", flush=True)

    df_out = pd.DataFrame([asdict(r) for r in records])
    summary = summarize(df_out)
    return df_out, summary


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "no_fill": 0,
            "timeouts": 0,
            "wr": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
        }
    wins = int((df["outcome"] == "win").sum())
    losses = int((df["outcome"] == "loss").sum())
    no_fill = int((df["outcome"] == "no_fill").sum())
    timeouts = int((df["outcome"] == "timeout").sum())
    decided = wins + losses
    wr = (wins / decided * 100) if decided > 0 else 0.0
    total_r = float(df["r_pnl"].sum()) if "r_pnl" in df.columns else 0.0
    avg_r = float(df["r_pnl"].mean()) if "r_pnl" in df.columns and len(df) else 0.0
    return {
        "total_trades": int(len(df)),
        "wins": wins,
        "losses": losses,
        "no_fill": no_fill,
        "timeouts": timeouts,
        "decided": decided,
        "wr": wr,
        "total_r": total_r,
        "avg_r": avg_r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--timeframe-minutes", type=int, default=60)
    ap.add_argument("--end-ts", type=int, default=0, help="End timestamp (ms). 0 = now (UTC).")
    ap.add_argument("--high-fidelity", action="store_true", help="Use micro candles for 5m fill + exits.")
    ap.add_argument("--micro-interval", type=str, default="3", help="Micro timeframe interval (minutes) for high fidelity (default 3).")
    ap.add_argument("--pending-minutes", type=int, default=5, help="Pending limit timeout in minutes (live is 5).")
    ap.add_argument("--max-hold-hours", type=int, default=25, help="Max hold time for TP/SL resolution (25h matches learner timeout).")
    ap.add_argument("--cache-dir", type=str, default=".cache/backtest", help="Cache directory for klines.")
    ap.add_argument("--out", type=str, default="hidden_bearish_live_match_results.csv")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress printing.")
    args = ap.parse_args()

    end_ts_ms = args.end_ts if args.end_ts and args.end_ts > 0 else _utc_ms(datetime.now(tz=timezone.utc))
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    df, summary = run_backtest(
        days=args.days,
        end_ts_ms=end_ts_ms,
        timeframe_minutes=args.timeframe_minutes,
        high_fidelity=args.high_fidelity,
        micro_interval=args.micro_interval,
        pending_minutes=args.pending_minutes,
        max_hold_hours=args.max_hold_hours,
        cache_dir=cache_dir,
        only_hidden_bearish=True,
        progress=not args.no_progress,
    )

    # Write results
    df.to_csv(args.out, index=False)

    print("=" * 80)
    print("HIDDEN BEARISH - LIVE MATCH BACKTEST")
    print("=" * 80)
    print(f"End (UTC): {datetime.fromtimestamp(end_ts_ms/1000, tz=timezone.utc)} | End_ts(ms): {end_ts_ms} | Days: {args.days}")
    print(f"Mode: {'HIGH_FIDELITY(3m)' if args.high_fidelity else 'COARSE(1h candles)'}")
    print(f"Symbols: {len((_load_config('config.yaml').get('trade', {}) or {}).get('divergence_symbols') or [])}")
    print("-" * 80)
    for k, v in summary.items():
        print(f"{k:>12}: {v}")
    print("-" * 80)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


