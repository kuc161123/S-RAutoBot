"""
Trail Shadow — counterfactual learner for the trailing stop. OBSERVATIONAL ONLY.

Answers one question with live data instead of backtests: **is the trailing stop
actually earning its keep, right now, on this account?**

For every trade the bot opens it records the intended geometry, then replays the
same candles TWICE from klines:

    r_fixed  — the original fixed TP / fixed SL bracket (trailing switched off)
    r_trail  — arm at trigger_r, then trail atr_mult x ATR behind each closed
               candle's extreme, ratchet-only, no breakeven move

Exactly one of those is what really happened; the other is the counterfactual. Both
are computed from the same bars, so the difference is attributable to the exit rule
and nothing else. On top of that it measures SLIPPAGE — the realized R the exchange
actually paid versus the R this model says the live arm should have produced. That
is the number backtests cannot see, and the one that decides whether the 1x or the
2x cost column reflects reality.

DESIGN CONTRACT — identical to the rest of the shadow layer:
  - This module must NEVER affect trading. It reads state and writes to Postgres.
  - Every public method is wrapped and returns a safe default on ANY error.
  - No DATABASE_URL or no psycopg2 -> silently no-ops.
  - Callers additionally wrap invocations (belt and suspenders).

Known biases, stated so the output is read correctly:
  - 1H bars: when a bar touches both the stop and the TP the order is unknown, and
    this resolver scores it as the STOP (SL-wins-ties). That is pessimistic, and it
    is pessimistic *more often* for the trailed arm, whose stop sits closer to price.
    So a positive r_trail - r_fixed here is a conservative reading.
  - Unresolved trades are excluded entirely rather than counted as scratches, which
    would otherwise bias toward whichever arm exits faster (the trailed one).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    _PG_OK = True
except Exception:
    _PG_OK = False

WARMUP_BARS = 30          # bars fetched before entry so ATR(14) is warm at the entry bar
HORIZON_BARS = 400        # give the fixed-TP arm room; RR 8-10 trades run long
MIN_AGE_H = 6             # don't try to grade a trade younger than this


def _atr_series(candles, period: int = 14):
    """Simple 14-period rolling mean of true range.

    Must match the live bot: divergence_detector.calculate_atr uses
    tr.rolling(period).mean(), and backtest_3yr_walkforward.prepare_data uses the
    same. A Wilder EMA here would quietly shift every trail level.
    """
    n = len(candles)
    out = [float('nan')] * n
    trs = []
    prev_close = None
    for i, (_, o, h, l, c) in enumerate(candles):
        tr = (h - l) if prev_close is None else max(h - l, abs(h - prev_close),
                                                    abs(l - prev_close))
        trs.append(tr)
        prev_close = c
        if i >= period:
            out[i] = sum(trs[i - period + 1:i + 1]) / period
        elif i == period - 1:
            out[i] = sum(trs) / period
    return out


def walk_fixed(candles, entry, sl, tp, side, rr, risk=None):
    """Original bracket. Returns (r, bars) or (None, n) if unresolved in the window.

    `risk` MUST be passed the same value walk_trail receives. This used to hardcode
    -1.0 on a stop hit while walk_trail divided by sl_dist, so the two arms measured R
    on different denominators. Because `entry` is the ACTUAL market fill while `sl` is
    derived from the PLANNED entry (see CLAUDE.md 11), |entry - sl| != sl_dist on every
    trade, and the arms disagreed purely from entry-fill drift. That produced the
    impossible reading "trailing better 8 / worse 9" on a sample where the fixed arm
    lost -1.0R on all 18 trades — trailing is ratchet-only and can never be worse.
    """
    if risk is None:
        risk = (entry - sl) if side == 'long' else (sl - entry)
    if risk <= 0:
        return None, 0
    for i, (_, o, h, l, c) in enumerate(candles):
        if side == 'long':
            hit_sl, hit_tp = l <= sl, h >= tp
        else:
            hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl:                      # SL-wins-ties (conservative)
            return ((sl - entry) / risk if side == 'long'
                    else (entry - sl) / risk), i + 1
        if hit_tp:
            return float(rr), i + 1
    return None, len(candles)


def walk_trail(candles, atrs, entry, sl, tp, side, rr, trigger_r, atr_mult, risk=None):
    """Trailed bracket — the live rule, replayed.

    Mirrors autobot.core.bot._trail_one exactly:
      * MFE is measured against the ORIGINAL stop distance, never the ratcheted one.
      * The stop only arms once MFE >= trigger_r. No breakeven move, ever.
      * A level derived from bar i takes effect from bar i+1 (the ratchet happens at
        the bottom of the loop, after that bar's hit checks) — which is what a bot
        updating once per closed candle can actually do.
    """
    # `risk` is passed explicitly when the exact ATR*mult distance is on record.
    # Reconstructing it as |entry - sl| loses the last ulp, which on a low-priced symbol
    # is enough to turn an excursion of exactly 3.0R into 2.999999999999996 and skip a
    # ratchet the backtest takes. See verify_trailing_parity.py.
    if risk is None:
        risk = (entry - sl) if side == 'long' else (sl - entry)
    if risk <= 0:
        return None, 0, 0.0, 0
    long = side == 'long'
    stop = sl
    peak_r = 0.0
    moves = 0
    for i, (_, o, h, l, c) in enumerate(candles):
        if long:
            hit_sl, hit_tp = l <= stop, h >= tp
        else:
            hit_sl, hit_tp = h >= stop, l <= tp
        if hit_sl:
            r = (stop - entry) / risk if long else (entry - stop) / risk
            return r, i + 1, peak_r, moves
        if hit_tp:
            return float(rr), i + 1, peak_r, moves
        # THIS bar's excursion, re-tested each bar — not a running peak. See the matching
        # comment in bot._trail_one: the validated backtest holds the stop when price
        # falls back below the trigger rather than continuing to ratchet.
        bar_r = (h - entry) / risk if long else (entry - l) / risk
        peak_r = max(peak_r, bar_r)
        if bar_r < trigger_r:
            continue
        a = atrs[i] if i < len(atrs) else float('nan')
        if not (a == a) or a <= 0:
            continue
        cand = (h - a * atr_mult) if long else (l + a * atr_mult)
        new = max(stop, cand) if long else min(stop, cand)
        if new != stop:
            stop = new
            moves += 1
    return None, len(candles), peak_r, moves


def _max_dd(series):
    """Max drawdown of a cumulative-R path. Series must be in chronological order."""
    peak = 0.0
    cum = 0.0
    dd = 0.0
    for r in series:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


class TrailShadow:
    def __init__(self, db_url: str | None, enabled: bool = True):
        self.db_url = db_url or None
        self.enabled = bool(self.db_url) and _PG_OK and bool(enabled)
        if self.enabled:
            self._safe(self._ensure_schema)
            logger.info("[TRAILSHADOW] Trail shadow learner enabled (Postgres).")
        else:
            logger.info("[TRAILSHADOW] Trail shadow learner disabled.")

    # ---- infra ----
    def _conn(self):
        return psycopg2.connect(self.db_url)

    def _safe(self, fn, *a, default=None, **kw):
        if not self.enabled:
            return default
        try:
            return fn(*a, **kw)
        except Exception as e:
            logger.debug(f"[TRAILSHADOW] {getattr(fn, '__name__', fn)} failed: {e}")
            return default

    def _ensure_schema(self):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trail_shadow (
                    id            TEXT PRIMARY KEY,
                    ts            TIMESTAMPTZ DEFAULT now(),
                    entry_ms      BIGINT,
                    symbol        TEXT, side TEXT,
                    entry         DOUBLE PRECISION,
                    orig_sl       DOUBLE PRECISION,
                    tp            DOUBLE PRECISION,
                    rr            DOUBLE PRECISION,
                    sl_dist       DOUBLE PRECISION,   -- exact ATR*mult distance at entry
                    trigger_r     DOUBLE PRECISION,
                    atr_mult      DOUBLE PRECISION,
                    live_trailing BOOLEAN,          -- was the trail ON for this trade
                    risk_usd      DOUBLE PRECISION,
                    -- live outcome, filled at close
                    closed        BOOLEAN DEFAULT FALSE,
                    actual_r      DOUBLE PRECISION,
                    actual_exit   DOUBLE PRECISION,
                    live_moves    INT,
                    live_peak_r   DOUBLE PRECISION,
                    final_sl      DOUBLE PRECISION,
                    -- modelled outcomes, filled by the resolver
                    status        TEXT DEFAULT 'pending',
                    r_fixed       DOUBLE PRECISION,
                    r_trail       DOUBLE PRECISION,
                    bars_fixed    INT,
                    bars_trail    INT,
                    model_peak_r  DOUBLE PRECISION,
                    model_moves   INT,
                    resolved_ts   TIMESTAMPTZ,
                    checked_ts    TIMESTAMPTZ
                );
            """)
            cur.execute("ALTER TABLE trail_shadow ADD COLUMN IF NOT EXISTS sl_dist DOUBLE PRECISION;")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_trailshadow_status "
                        "ON trail_shadow(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_trailshadow_ts ON trail_shadow(ts);")
            conn.commit()

    @staticmethod
    def _sid(trade_key, trade):
        ts = int(trade.entry_time.timestamp())
        return f"{trade_key}:{ts}"

    # ---- write path ----
    def log_open(self, trade_key, trade, live_trailing, trigger_r, atr_mult):
        return self._safe(self._log_open, trade_key, trade, live_trailing,
                          trigger_r, atr_mult)

    def _log_open(self, trade_key, trade, live_trailing, trigger_r, atr_mult):
        sid = self._sid(trade_key, trade)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trail_shadow
                  (id, entry_ms, symbol, side, entry, orig_sl, tp, rr, sl_dist,
                   trigger_r, atr_mult, live_trailing, risk_usd)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO NOTHING;
            """, (sid, int(trade.entry_time.timestamp() * 1000), trade.symbol,
                  trade.side, float(trade.entry_price),
                  float(trade.original_stop_loss or trade.stop_loss),
                  float(trade.take_profit), float(trade.rr_ratio),
                  float(getattr(trade, 'risk_dist', 0.0) or 0.0),
                  float(trigger_r), float(atr_mult), bool(live_trailing),
                  float(trade.risk_usd_at_entry or 0.0)))
            conn.commit()
        return sid

    def log_move(self, trade_key, trade, prev, new):
        """Ratchets are already counted on the ActiveTrade; this only keeps the row's
        running view fresh so /trailstats shows live activity before the trade closes."""
        return self._safe(self._log_move, trade_key, trade)

    def _log_move(self, trade_key, trade):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE trail_shadow SET live_moves=%s, live_peak_r=%s, final_sl=%s
                WHERE id=%s;
            """, (int(trade.trail_moves), float(trade.trail_peak_r),
                  float(trade.stop_loss), self._sid(trade_key, trade)))
            conn.commit()

    def log_close(self, trade_key, trade, actual_r=None, actual_exit=None):
        return self._safe(self._log_close, trade_key, trade, actual_r, actual_exit)

    def _log_close(self, trade_key, trade, actual_r, actual_exit):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE trail_shadow
                SET closed=TRUE, actual_r=%s, actual_exit=%s,
                    live_moves=%s, live_peak_r=%s, final_sl=%s
                WHERE id=%s;
            """, (actual_r, actual_exit, int(getattr(trade, 'trail_moves', 0)),
                  float(getattr(trade, 'trail_peak_r', 0.0)),
                  float(trade.stop_loss), self._sid(trade_key, trade)))
            conn.commit()

    # ---- resolve path ----
    async def resolve_pending(self, broker, interval='60', limit_rows=25):
        if not self.enabled:
            return
        try:
            with self._conn() as conn, \
                    conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, symbol, side, entry, orig_sl, tp, rr, trigger_r,
                           atr_mult, entry_ms, sl_dist
                    FROM trail_shadow
                    WHERE status='pending'
                      AND ts < now() - interval '%s hours'
                      AND (checked_ts IS NULL OR checked_ts < now() - interval '2 hours')
                    ORDER BY ts ASC LIMIT %s;
                """ % (MIN_AGE_H, int(limit_rows)))
                rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"[TRAILSHADOW] resolve fetch failed: {e}")
            return
        for r in rows:
            try:
                await self._resolve_one(broker, r, interval)
            except Exception as e:
                logger.debug(f"[TRAILSHADOW] resolve {r.get('id')} failed: {e}")

    async def _resolve_one(self, broker, r, interval):
        # Fetch a warmup run-in so ATR(14) is already valid on the entry bar; without it
        # the first 14 bars of every trade would have no trail level at all.
        start = int(r['entry_ms']) - WARMUP_BARS * 3_600_000
        kl = await broker.get_klines(r['symbol'], interval, limit=HORIZON_BARS + WARMUP_BARS,
                                     start=start)
        candles = self._parse_klines(kl)
        if len(candles) < WARMUP_BARS + 5:
            self._safe(self._touch_checked, r['id'])
            return
        atrs_all = _atr_series(candles)
        # Trade bars = the entry candle onward. Entries are at the open of the entry
        # candle, so that candle's own range is live for both arms.
        idx = next((i for i, c in enumerate(candles) if c[0] >= int(r['entry_ms'])), None)
        if idx is None:
            self._safe(self._touch_checked, r['id'])
            return
        bars = candles[idx:]
        atrs = atrs_all[idx:]
        if len(bars) < 3:
            self._safe(self._touch_checked, r['id'])
            return

        _risk = float(r['sl_dist']) if r.get('sl_dist') else None
        rf, bf = walk_fixed(bars, r['entry'], r['orig_sl'], r['tp'], r['side'], r['rr'],
                            risk=_risk)
        rt, bt, peak, moves = walk_trail(bars, atrs, r['entry'], r['orig_sl'], r['tp'],
                                         r['side'], r['rr'], r['trigger_r'], r['atr_mult'],
                                         risk=_risk)
        if rf is None or rt is None:
            # One arm is still open. Grading now would systematically favour the arm
            # that exits sooner, so wait — unless the horizon is genuinely exhausted.
            if len(bars) < HORIZON_BARS:
                self._safe(self._touch_checked, r['id'])
                return
            self._safe(self._write_resolution, r['id'], 'expired', rf, rt, bf, bt,
                       peak, moves)
            return
        self._safe(self._write_resolution, r['id'], 'resolved', rf, rt, bf, bt,
                   peak, moves)

    def _touch_checked(self, sid):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE trail_shadow SET checked_ts=now() WHERE id=%s;", (sid,))
            conn.commit()

    def _write_resolution(self, sid, status, rf, rt, bf, bt, peak, moves):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE trail_shadow
                SET status=%s, r_fixed=%s, r_trail=%s, bars_fixed=%s, bars_trail=%s,
                    model_peak_r=%s, model_moves=%s, resolved_ts=now()
                WHERE id=%s;
            """, (status, rf, rt, bf, bt, peak, moves, sid))
            conn.commit()

    @staticmethod
    def _parse_klines(kl):
        out = []
        for row in (kl or []):
            try:
                if isinstance(row, dict):
                    t = int(row.get('start') or row.get('startTime') or row.get('t'))
                    o, h, l, c = (float(row['open']), float(row['high']),
                                  float(row['low']), float(row['close']))
                else:
                    t = int(row[0])
                    o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                out.append((t, o, h, l, c))
            except Exception:
                continue
        out.sort(key=lambda x: x[0])
        return out

    # ---- read path ----
    def stats(self, days: int = 90) -> dict:
        return self._safe(self._stats, days, default={}) or {}

    def _stats(self, days):
        with self._conn() as conn, \
                conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT r_fixed, r_trail, live_trailing, actual_r, closed,
                       model_moves, model_peak_r, bars_fixed, bars_trail
                FROM trail_shadow
                WHERE status='resolved' AND ts > now() - interval '%s days'
                ORDER BY ts ASC;
            """ % int(days))
            rows = cur.fetchall()
            cur.execute("SELECT count(*) AS n FROM trail_shadow WHERE status='pending';")
            pending = (cur.fetchone() or {}).get('n', 0)

        n = len(rows)
        if n == 0:
            return {'n': 0, 'pending': pending}

        fx = [float(r['r_fixed']) for r in rows]
        tr = [float(r['r_trail']) for r in rows]
        d = [t - f for t, f in zip(tr, fx)]
        mean_d = sum(d) / n
        var = sum((x - mean_d) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 and var > 0 else 0.0
        t_stat = (mean_d / se) if se > 0 else 0.0

        # Slippage: modelled R for the arm that was actually live, vs realized R.
        slip = []
        for r in rows:
            if not r['closed'] or r['actual_r'] is None:
                continue
            modelled = float(r['r_trail']) if r['live_trailing'] else float(r['r_fixed'])
            slip.append(float(r['actual_r']) - modelled)
        mean_slip = (sum(slip) / len(slip)) if slip else None

        out = {
            'n': n, 'pending': pending,
            'sum_fixed': sum(fx), 'sum_trail': sum(tr),
            'mean_fixed': sum(fx) / n, 'mean_trail': sum(tr) / n,
            'wr_fixed': sum(1 for x in fx if x > 0) / n,
            'wr_trail': sum(1 for x in tr if x > 0) / n,
            'mean_delta': mean_d, 't_stat': t_stat,
            'dd_fixed': _max_dd(fx), 'dd_trail': _max_dd(tr),
            'better': sum(1 for x in d if x > 0), 'worse': sum(1 for x in d if x < 0),
            'same': sum(1 for x in d if x == 0),
            'mean_slip': mean_slip, 'n_slip': len(slip),
            'avg_moves': sum(int(r['model_moves'] or 0) for r in rows) / n,
            'armed_pct': sum(1 for r in rows
                             if float(r['model_peak_r'] or 0) >= 1e-9) / n,
        }
        out['verdict'] = self._verdict(out)
        return out

    @staticmethod
    def _verdict(s) -> dict:
        """Turn the numbers into a recommendation, stated with its own uncertainty.

        The bar is deliberately asymmetric and matches why the trail was deployed: it
        went live as DRAWDOWN control, not as a profit upgrade. So it stays on while it
        is not materially worse on R and is better on drawdown; it only gets switched
        off on evidence it is genuinely losing money.
        """
        n = s['n']
        if n < 60:
            return {'code': 'COLLECTING',
                    'text': f"Collecting — {n}/60 resolved trades. No call yet.",
                    'icon': '⏳'}
        dd_better = s['dd_trail'] < s['dd_fixed']
        t = s['t_stat']
        md = s['mean_delta']
        if t <= -2.0 and not dd_better:
            return {'code': 'OFF',
                    'text': (f"Switch OFF. Trailing is worse on R ({md:+.3f}/trade, "
                             f"t={t:+.1f}) and does not reduce drawdown "
                             f"({s['dd_trail']:.1f}R vs {s['dd_fixed']:.1f}R)."),
                    'icon': '🔴'}
        if t <= -2.0 and dd_better:
            return {'code': 'JUDGEMENT',
                    'text': (f"Your call. Trailing costs {md:+.3f}R/trade (t={t:+.1f}) "
                             f"but cuts drawdown {s['dd_fixed']:.1f}R → "
                             f"{s['dd_trail']:.1f}R. That is the trade it was deployed "
                             f"to make — keep ON if the smoother curve is worth it."),
                    'icon': '🟡'}
        if t >= 2.0:
            return {'code': 'KEEP',
                    'text': (f"Keep ON. Trailing is ahead by {md:+.3f}R/trade "
                             f"(t={t:+.1f}) over {n} trades."),
                    'icon': '🟢'}
        return {'code': 'KEEP',
                'text': (f"Keep ON. No significant R difference ({md:+.3f}/trade, "
                         f"t={t:+.1f}), and drawdown is "
                         f"{'lower' if dd_better else 'not lower'} "
                         f"({s['dd_trail']:.1f}R vs {s['dd_fixed']:.1f}R)."),
                'icon': '🟢' if dd_better else '🟡'}
