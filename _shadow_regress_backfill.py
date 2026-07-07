"""Shadow resolver regression check + mfe_to_sl backfill.

Run where DATABASE_URL points at the bot's Postgres (e.g. Railway shell):
    python3 _shadow_regress_backfill.py            # regression check only (read-only)
    python3 _shadow_regress_backfill.py --backfill # also write mfe_to_sl/bars_to_outcome
                                                   # for resolved rows missing them

Uses Bybit PUBLIC klines (no API keys). Read-only w.r.t. trading; touches only
shadow_signals counterfactual columns when --backfill is passed.
"""
import os
import sys
import time

import psycopg2
import psycopg2.extras
import requests

from autobot.core.shadow_logger import walk_outcome

DB = os.getenv('DATABASE_URL')
if not DB:
    sys.exit("DATABASE_URL not set")
BACKFILL = '--backfill' in sys.argv
LIMIT = 200


def get_klines(symbol, start_ms):
    r = requests.get("https://api.bybit.com/v5/market/kline", params={
        'category': 'linear', 'symbol': symbol, 'interval': '60',
        'limit': 200, 'start': int(start_ms)}, timeout=10)
    rows = r.json().get('result', {}).get('list', []) or []
    out = []
    for x in rows:
        try:
            out.append((int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])))
        except (ValueError, IndexError):
            continue
    out.sort(key=lambda c: c[0])
    return out


with psycopg2.connect(DB) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute("""
        SELECT id, symbol, side, entry, sl, tp, rr, sig_time_ms, status, r_result, mfe, mae, mfe_to_sl
        FROM shadow_signals
        WHERE status IN ('win','loss','expired')
        ORDER BY resolved_ts DESC NULLS LAST LIMIT %s;
    """, (LIMIT,))
    rows = cur.fetchall()
print(f"checking {len(rows)} resolved rows (backfill={'ON' if BACKFILL else 'off'})")

mismatch = skipped = checked = backfilled = 0
for r in rows:
    candles = get_klines(r['symbol'], r['sig_time_ms'])
    candles = [c for c in candles if c[0] >= r['sig_time_ms']]
    time.sleep(0.05)
    if len(candles) < 2:
        skipped += 1
        continue
    st, rr_, mfe, mae, m2s, bars = walk_outcome(
        candles[1:], r['entry'], r['sl'], r['tp'], r['side'], r['rr'])
    checked += 1
    if st != r['status'] or abs((rr_ or 0) - (r['r_result'] or 0)) > 1e-9:
        mismatch += 1
        print(f"  MISMATCH {r['id']}: stored {r['status']}/{r['r_result']} vs walk {st}/{rr_}")
    if BACKFILL and r['mfe_to_sl'] is None:
        with psycopg2.connect(DB) as conn, conn.cursor() as cur:
            cur.execute("UPDATE shadow_signals SET mfe_to_sl=%s, bars_to_outcome=%s WHERE id=%s;",
                        (m2s, bars, r['id']))
            conn.commit()
        backfilled += 1

print(f"\nchecked={checked} skipped={skipped} mismatches={mismatch} backfilled={backfilled}")
print("NOTE: a few mismatches can occur legitimately if the exchange revised candles; "
      "investigate only if the rate exceeds ~2%.")
sys.exit(1 if checked and mismatch / max(checked, 1) > 0.02 else 0)
