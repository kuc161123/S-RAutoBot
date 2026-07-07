"""
Donchian Channel Breakout Detector — SHADOW/OBSERVATION ONLY (Phase 1b).

1:1 port of the detection logic from the validated family research study
(gen_families.py, rolling-fold walk-forward validated 2026-07: Tier B pooled
OOS +1011R / 2327 trades / PF 1.53, positive in 12 of 15 folds). Do NOT
"improve" the rules here — any change invalidates that validation.

Rules (per channel length N in {48, 96, 168}):
  raw_long[i]  = close[i] > max(high[i-N .. i-1])
  raw_short[i] = close[i] < min(low[i-N .. i-1])
  signal fires on the CROSS: raw[i] and not raw[i-1],
  with a 12-bar cooldown per (subtype, side).
Entry convention (matches live bot + study): signal at bar-i close, entry at
bar i+1 OPEN. SL = ATR(14, SMA-of-TR)[i] * atr_mult; TP = SL distance * RR.

Fill modeling note: shadow rows log the RAW next-bar open (same convention as
the live DIV shadow rows); costs/slippage are applied at analysis time by the
shadow analyst, keeping families comparable.
"""
from __future__ import annotations

COOLDOWN_BARS = 12
CHANNELS = (48, 96, 168)


def detect_donchian_last_bar(df, channels=CHANNELS):
    """Check whether the LAST CLOSED bar of `df` fires a Donchian cross signal.

    Args:
        df: prepared OHLCV DataFrame (needs high/low/close and enough history);
            the last row must be the just-closed candle. Read-only.
        channels: channel lengths to evaluate.

    Returns:
        list of dicts: {'code': 'DONCH48_BULL', 'side': 'long', 'channel': 48}
        for each channel/side whose cross condition fires on the last bar.
        Cooldown is NOT applied here (stateless) — the caller enforces it.
    """
    out = []
    n = len(df)
    if n < min(channels) + 3:
        return out
    close = df['close']
    high = df['high']
    low = df['low']
    i = n - 1
    for N in channels:
        if n < N + 3:
            continue
        # prior-N-bars extremes, excluding the current bar (shift 1)
        hh_i = high.iloc[i - N:i].max()
        hh_prev = high.iloc[i - 1 - N:i - 1].max()
        ll_i = low.iloc[i - N:i].min()
        ll_prev = low.iloc[i - 1 - N:i - 1].min()
        raw_l_i = close.iloc[i] > hh_i
        raw_l_prev = close.iloc[i - 1] > hh_prev
        raw_s_i = close.iloc[i] < ll_i
        raw_s_prev = close.iloc[i - 1] < ll_prev
        if raw_l_i and not raw_l_prev:
            out.append({'code': f'DONCH{N}_BULL', 'side': 'long', 'channel': N})
        if raw_s_i and not raw_s_prev:
            out.append({'code': f'DONCH{N}_BEAR', 'side': 'short', 'channel': N})
    return out
