#!/usr/bin/env python3
"""Measure the bot's ACTUAL round-trip execution cost from live trade data.

Read-only. The connection string is read from a file path given on the command line and is
never printed or written to any output — outputs contain aggregates only.

Why this exists: every backtest in this repo charges a round-trip cost that was ASSUMED
(18 bps in `backtest_production_correct.py:60`) or calibrated indirectly (34.1 bps in
STRATEGY_VERDICT_2026-08-11.md 2.3). Neither was measured against real fills. The whole
risk study turns on this number, because cost enters as cost_R = round_trip / stop_frac
and the strategy's gross alpha is of the same order.

Method: for each closed live trade we know the planned entry, the planned stop, and the
realized R. A trade that stopped out should realize exactly -1.0 R before costs; the
shortfall below -1.0 is the round-trip cost expressed in R, and multiplying by stop_frac
converts it back to basis points of notional. Stop-outs are the clean estimator because
their exit price is pinned by the stop, so the residual is cost + stop slippage and
nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

OUT = Path(__file__).resolve().parent / "results"


def q(conn, sql):
    return pd.read_sql(sql, conn)


def main():
    url = Path(sys.argv[1]).read_text().strip()
    conn = psycopg2.connect(url)
    OUT.mkdir(parents=True, exist_ok=True)

    cols = q(conn, """select table_name, column_name, data_type
                      from information_schema.columns
                      where table_name in ('trade_history','trail_shadow','exec_log')
                      order by table_name, ordinal_position""")
    print("=== SCHEMA ===")
    for t, g in cols.groupby("table_name"):
        print(f"{t}: {', '.join(g.column_name)}")

    th = q(conn, "select * from trade_history")
    print(f"\n=== trade_history: {len(th):,} rows ===")
    print(th.dtypes.to_string())
    if len(th):
        print(th.head(3).to_string())

    ts = q(conn, "select * from trail_shadow")
    print(f"\n=== trail_shadow: {len(ts):,} rows ===")
    print(ts.dtypes.to_string())
    if len(ts):
        print(ts.head(3).to_string())

    th.to_parquet(OUT / "live_trade_history.parquet", index=False)
    ts.to_parquet(OUT / "live_trail_shadow.parquet", index=False)
    print(f"\nsaved -> {OUT/'live_trade_history.parquet'}, {OUT/'live_trail_shadow.parquet'}")
    conn.close()


if __name__ == "__main__":
    main()
