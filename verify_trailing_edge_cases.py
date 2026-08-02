#!/usr/bin/env python3
"""Trailing stop — the paths verify_trailing_parity.py cannot reach.

Parity proves the MATH matches the validated backtest. This proves the STATEFUL
behaviour is safe: restarts, exchange rejections, the runtime toggle, positions the
bot cannot reconstruct, and the short side. Every case here asserts the one invariant
that matters most — **the engine may tighten a stop, never widen one.**

Run:  python3 verify_trailing_edge_cases.py
"""
from __future__ import annotations

import asyncio
import sys
import types
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

from autobot.core.bot import ActiveTrade, Bot4H

ENTRY = 100.0
RISK = 5.0


class FakeBroker:
    def __init__(self):
        self.calls = []

    async def amend_stop_loss(self, symbol, stop_loss, side):
        self.calls.append(float(stop_loss))
        return {"retCode": 0}


class RejectingBroker:
    async def amend_stop_loss(self, *a, **k):
        return None


class Stub:
    def __init__(self):
        self.trailing_enabled = True
        self.trailing_config = {"trigger_r": 3.0, "atr_mult": 1.0,
                                "min_step_pct": 0.0, "notify": False}
        self.active_trades = {}
        self.lifetime_stats = {}
        self.telegram = None
        self.trail_shadow = None
        self.broker = FakeBroker()
        self.trail_stats = {"armed": 0, "moves": 0, "failures": 0}

    def save_lifetime_stats(self):
        pass


def bind(b):
    for n in ("_trail_params", "_trail_one", "_persist_trail_state",
              "_restore_trail_state", "_notify_trail", "_update_trailing_stops",
              "trail_eligible_counts"):
        setattr(b, n, types.MethodType(getattr(Bot4H, n), b))
    return b


def frame(highs, lows, atr=2.0):
    idx = pd.date_range("2026-01-01", periods=len(highs), freq="h")
    return pd.DataFrame({"open": [ENTRY] * len(highs), "high": highs, "low": lows,
                         "close": [(a + b) / 2 for a, b in zip(highs, lows)],
                         "atr": [atr] * len(highs)}, index=idx)


def mk(df, **kw):
    d = dict(symbol="XUSDT", side="long", entry_price=ENTRY, stop_loss=ENTRY - RISK,
             take_profit=ENTRY + RISK * 10, rr_ratio=10.0, position_size=1.0,
             entry_time=df.index[0].to_pydatetime(), original_stop_loss=ENTRY - RISK,
             risk_dist=RISK, entry_bar_ts=df.index[0])
    d.update(kw)
    return ActiveTrade(**d)


def main():
    df = frame([101, 102, 103, 104, 106, 121, 122, 121, 120, 119, 118, 117],
               [99, 100, 101, 102, 104, 118, 119, 118, 117, 116, 115, 114])
    run = lambda b, d: asyncio.run(b._update_trailing_stops("XUSDT", d))  # noqa: E731

    # 1 — arms at +3R and ratchets to high - 1*ATR
    bot = bind(Stub())
    tr = mk(df)
    bot.active_trades["XUSDT_long"] = tr
    run(bot, df.iloc[:7])
    assert tr.trail_moves == 1 and abs(tr.stop_loss - 119.0) < 1e-9, tr.stop_loss
    print(f"1. armed          stop={tr.stop_loss} peak={tr.trail_peak_r:.2f}R "
          f"locked={tr.trail_locked_r:.2f}R")
    persisted = dict(bot.lifetime_stats)

    # 2 — restart. Rebuilt from EXCHANGE data only (which reports the already-trailed
    #     stop) and with entry_time=now(), i.e. the wrong clock. Must still continue.
    bot2 = bind(Stub())
    bot2.lifetime_stats = persisted
    ad = ActiveTrade(symbol="XUSDT", side="long", entry_price=ENTRY, stop_loss=119.0,
                     take_profit=ENTRY + RISK * 10, rr_ratio=10.0, position_size=1.0,
                     entry_time=datetime.now())
    assert bot2._restore_trail_state("XUSDT_long", ad)
    assert ad.risk_dist == RISK and ad.original_stop_loss == ENTRY - RISK
    assert ad.entry_bar_ts == df.index[0]
    bot2.active_trades["XUSDT_long"] = ad
    run(bot2, df.iloc[:8])
    assert abs(ad.stop_loss - 120.0) < 1e-9, ad.stop_loss
    print(f"2. after restart  stop={ad.stop_loss} moves={ad.trail_moves}")

    # 3 — price collapses through the stop. The clamp limit now sits BELOW the current
    #     stop; it must NOT pull the stop down.
    before = ad.stop_loss
    run(bot2, df)
    assert ad.stop_loss == before, f"stop widened {before} -> {ad.stop_loss}"
    print(f"3. clamp          stop held at {ad.stop_loss}, never widened "
          f"(clamped={bot2.trail_stats.get('clamped', 0)})")

    # 4 — no persisted record: the bot cannot know the original risk, so it must
    #     leave the position entirely alone.
    bot3 = bind(Stub())
    orph = ActiveTrade(symbol="XUSDT", side="long", entry_price=ENTRY, stop_loss=95.0,
                       take_profit=150.0, rr_ratio=10.0, position_size=1.0,
                       entry_time=datetime.now())
    assert not bot3._restore_trail_state("XUSDT_long", orph)
    bot3.active_trades["XUSDT_long"] = orph
    run(bot3, df)
    assert orph.stop_loss == 95.0 and bot3.broker.calls == []
    print("4. orphan         untouched, 0 API calls")

    # 5 — /trail off stops new ratchets and never widens what is already there
    bot2.trailing_enabled = False
    b4, n4 = ad.stop_loss, len(bot2.broker.calls)
    run(bot2, df)
    assert ad.stop_loss == b4 and len(bot2.broker.calls) == n4
    print(f"5. /trail off     stop unchanged at {ad.stop_loss}, no API calls")

    # 6 — exchange rejects the amendment: local state must not advance, or every
    #     downstream R would be computed against a stop that isn't really there.
    bot4 = bind(Stub())
    bot4.broker = RejectingBroker()
    t6 = mk(df)
    bot4.active_trades["XUSDT_long"] = t6
    run(bot4, df.iloc[:7])
    assert t6.stop_loss == ENTRY - RISK and t6.trail_moves == 0
    assert bot4.trail_stats["failures"] == 1
    print(f"6. amend rejected stop stays {t6.stop_loss}, moves=0, failures=1")

    # 7 — short side mirrors correctly
    dfs = frame([101, 100, 99, 98, 96, 82, 81], [99, 98, 97, 96, 94, 79, 78])
    bot5 = bind(Stub())
    ts = mk(dfs, side="short", stop_loss=ENTRY + RISK, original_stop_loss=ENTRY + RISK,
            take_profit=ENTRY - RISK * 10, entry_bar_ts=dfs.index[0])
    bot5.active_trades["XUSDT_short"] = ts
    run(bot5, dfs)
    assert abs(ts.stop_loss - 81.0) < 1e-9, ts.stop_loss   # low 79 + 1*ATR
    print(f"7. short          stop={ts.stop_loss} locked={ts.trail_locked_r:.2f}R")

    print("\n✅ restart · clamp · orphan · toggle · amend-failure · short — all correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
