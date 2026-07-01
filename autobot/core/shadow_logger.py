"""
Shadow Learner (Phase 1a) — OBSERVATIONAL ONLY.

Logs every BOS-confirmed signal the bot evaluates (executed AND blocked) with its
intended entry/SL/TP and market context to Postgres, then resolves each outcome from
klines. Powers the /learn dashboard command.

DESIGN CONTRACT: this module must NEVER affect trading.
  - Every public method is wrapped and returns a safe default on ANY error.
  - If DATABASE_URL is missing or psycopg2 is unavailable, it silently no-ops.
  - Callers additionally wrap invocations in try/except (belt and suspenders).

Not yet implemented here (Phase 1b): running detection on candidate symbols outside
the configured universe. The `in_universe` column is written now so 1b needs no schema
change.
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


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """95% Wilson lower bound on win-rate (fraction). Conservative edge estimate."""
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


class ShadowLogger:
    MIN_N = 50            # minimum sample before a combo can be called "significant"
    RESOLVE_HORIZON_H = 1  # don't try to resolve a signal until it's at least this old

    def __init__(self, db_url: str | None):
        self.db_url = db_url or None
        self.enabled = bool(self.db_url) and _PG_OK
        if self.enabled:
            self._safe(self._ensure_schema)
            logger.info("[SHADOW] Shadow learner enabled (Postgres).")
        else:
            logger.info("[SHADOW] Shadow learner disabled (no DATABASE_URL or psycopg2).")

    # ---- infra ----
    def _conn(self):
        return psycopg2.connect(self.db_url)

    def _safe(self, fn, *a, default=None, **kw):
        if not self.enabled:
            return default
        try:
            return fn(*a, **kw)
        except Exception as e:
            logger.debug(f"[SHADOW] {getattr(fn,'__name__',fn)} failed: {e}")
            return default

    def _ensure_schema(self):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shadow_signals (
                    id           TEXT PRIMARY KEY,
                    ts           TIMESTAMPTZ DEFAULT now(),
                    sig_time_ms  BIGINT,
                    symbol       TEXT, side TEXT, div_type TEXT,
                    in_universe  BOOLEAN DEFAULT TRUE,
                    entry DOUBLE PRECISION, sl DOUBLE PRECISION, tp DOUBLE PRECISION,
                    rr DOUBLE PRECISION, atr_mult DOUBLE PRECISION,
                    regime TEXT, regime_mult DOUBLE PRECISION,
                    btc_bull BOOLEAN, sym_chop DOUBLE PRECISION, turnover DOUBLE PRECISION,
                    executed BOOLEAN DEFAULT FALSE,
                    status TEXT DEFAULT 'pending',
                    r_result DOUBLE PRECISION, mfe DOUBLE PRECISION, mae DOUBLE PRECISION,
                    resolved_ts TIMESTAMPTZ
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS ix_shadow_status ON shadow_signals(status);")
            conn.commit()

    # ---- write path (called from execute_trade) ----
    def log_signal(self, bot, symbol, signal, df, use_candle_open) -> str | None:
        return self._safe(self._log_signal, bot, symbol, signal, df, use_candle_open)

    def _log_signal(self, bot, symbol, signal, df, use_candle_open):
        side = signal.side
        code = signal.divergence_code
        # intended entry/SL/TP — mirrors execute_trade's own math so blocked signals
        # get the same intended levels as executed ones (fair comparison).
        rr = None
        try:
            rr = bot.symbol_config.get_rr_for_symbol(symbol, code)
        except Exception:
            rr = None
        if rr is None:
            return None
        atr_mult = 1.0
        try:
            cfg = bot.symbol_config.get_config_for_divergence(symbol, code)
            if cfg:
                atr_mult = float(cfg.get('atr_mult', 1.0))
        except Exception:
            pass
        if use_candle_open:
            entry = float(df.iloc[-1]['open'])
            atr = float(df.iloc[-2]['atr']) if len(df) >= 2 else float(df.iloc[-1]['atr'])
        else:
            entry = float(df.iloc[-1]['close'])
            atr = float(df.iloc[-1]['atr'])
        sld = atr * atr_mult
        if side == 'long':
            sl, tp = entry - sld, entry + sld * rr
        else:
            sl, tp = entry + sld, entry - sld * rr

        # context (all best-effort, sync only)
        regime, rmult = 'unknown', 1.0
        try:
            regime, rmult, _ = bot.get_regime_status()
        except Exception:
            pass
        btc_bull = None
        try:
            btc_bull = bot._btc_trend_cache.get('bullish')
        except Exception:
            pass
        sym_chop = None
        try:
            if 'chop' in df.columns:
                sym_chop = float(df['chop'].iloc[-1])
        except Exception:
            pass
        turnover = None
        try:
            if 'turnover' in df.columns:
                turnover = float(df['turnover'].iloc[-1])
            elif 'volume' in df.columns:
                turnover = float(df['volume'].iloc[-1]) * entry
        except Exception:
            pass
        in_universe = True
        try:
            in_universe = symbol in set(bot.symbol_config.get_enabled_symbols())
        except Exception:
            pass
        # signal candle time -> ms; id dedups per symbol/side/div per hour
        try:
            st = signal.timestamp
            sig_ms = int(st.timestamp() * 1000)
        except Exception:
            sig_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        sid = f"{symbol}:{side}:{code}:{sig_ms // 3_600_000}"

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO shadow_signals
                  (id, sig_time_ms, symbol, side, div_type, in_universe, entry, sl, tp, rr,
                   atr_mult, regime, regime_mult, btc_bull, sym_chop, turnover)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO NOTHING;
            """, (sid, sig_ms, symbol, side, code, in_universe, entry, sl, tp, rr,
                  atr_mult, regime, rmult, btc_bull, sym_chop, turnover))
            conn.commit()
        return sid

    def mark_executed(self, sid):
        if not sid:
            return
        self._safe(self._mark_executed, sid)

    def _mark_executed(self, sid):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE shadow_signals SET executed=TRUE WHERE id=%s;", (sid,))
            conn.commit()

    # ---- resolve outcomes from klines (best-effort) ----
    async def resolve_pending(self, broker, interval='60', limit_rows=40):
        if not self.enabled:
            return
        try:
            with self._conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, symbol, side, entry, sl, tp, rr, sig_time_ms
                    FROM shadow_signals
                    WHERE status='pending'
                      AND ts < now() - interval '%s hours'
                    ORDER BY ts ASC LIMIT %s;
                """ % (self.RESOLVE_HORIZON_H, int(limit_rows)))
                rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"[SHADOW] resolve fetch failed: {e}")
            return
        for r in rows:
            try:
                await self._resolve_one(broker, r, interval)
            except Exception as e:
                logger.debug(f"[SHADOW] resolve_one {r.get('id')} failed: {e}")

    async def _resolve_one(self, broker, r, interval):
        kl = await broker.get_klines(r['symbol'], interval, limit=200, start=int(r['sig_time_ms']))
        candles = self._parse_klines(kl)
        # keep only candles at/after the signal candle
        candles = [c for c in candles if c[0] >= r['sig_time_ms']]
        if len(candles) < 2:
            return  # too early — leave pending
        entry, sl, tp, side, rr = r['entry'], r['sl'], r['tp'], r['side'], r['rr']
        status, r_result = 'expired', 0.0
        mfe, mae = 0.0, 0.0
        for _, o, h, l, c in candles[1:]:  # start on the candle after the signal candle
            if side == 'long':
                mfe = max(mfe, (h - entry) / (entry - sl) if entry > sl else 0)
                mae = min(mae, (l - entry) / (entry - sl) if entry > sl else 0)
                hit_sl, hit_tp = l <= sl, h >= tp
            else:
                mfe = max(mfe, (entry - l) / (sl - entry) if sl > entry else 0)
                mae = min(mae, (entry - h) / (sl - entry) if sl > entry else 0)
                hit_sl, hit_tp = h >= sl, l <= tp
            if hit_sl and hit_tp:
                status, r_result = 'loss', -1.0  # ambiguous bar -> assume SL first (conservative)
                break
            if hit_sl:
                status, r_result = 'loss', -1.0
                break
            if hit_tp:
                status, r_result = 'win', float(rr)
                break
        # if still pending status and the window is exhausted (200 candles), mark expired
        if status == 'expired' and len(candles) < 200:
            return  # not enough data yet, keep pending
        self._safe(self._write_resolution, r['id'], status, r_result, mfe, mae)

    def _write_resolution(self, sid, status, r_result, mfe, mae):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE shadow_signals
                SET status=%s, r_result=%s, mfe=%s, mae=%s, resolved_ts=now()
                WHERE id=%s;
            """, (status, r_result, mfe, mae, sid))
            conn.commit()

    @staticmethod
    def _parse_klines(kl):
        out = []
        if not kl:
            return out
        for row in kl:
            try:
                if isinstance(row, dict):
                    t = int(row.get('start') or row.get('startTime') or row.get('t'))
                    o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
                else:
                    t = int(row[0]); o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                out.append((t, o, h, l, c))
            except Exception:
                continue
        out.sort(key=lambda x: x[0])
        return out

    # ---- read path (called from /learn) ----
    def summary(self):
        return self._safe(self._summary, default={}) or {}

    def _summary(self):
        with self._conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT count(*) total,
                       count(*) FILTER (WHERE executed) executed,
                       count(*) FILTER (WHERE status<>'pending') resolved,
                       count(DISTINCT symbol) symbols,
                       count(DISTINCT symbol||side||div_type) combos,
                       min(ts) first_ts
                FROM shadow_signals;
            """)
            return dict(cur.fetchone())

    def edge_rows(self, min_n=None, executed_only=False):
        return self._safe(self._edge_rows, min_n, executed_only, default=[]) or []

    def _edge_rows(self, min_n, executed_only):
        min_n = self.MIN_N if min_n is None else min_n
        where = "status IN ('win','loss')"
        if executed_only:
            where += " AND executed"
        with self._conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT symbol, side, div_type,
                       count(*) n,
                       count(*) FILTER (WHERE status='win') wins,
                       avg(rr) rr,
                       avg(r_result) avg_r,
                       avg(turnover) turnover
                FROM shadow_signals
                WHERE {where}
                GROUP BY symbol, side, div_type;
            """)
            rows = [dict(x) for x in cur.fetchall()]
        for x in rows:
            n = x['n']; wins = x['wins']
            x['wr'] = wins / n if n else 0.0
            x['lb'] = wilson_lower_bound(wins, n)
            x['breakeven'] = 1.0 / (1.0 + (x['rr'] or 5.0))
            if n >= min_n and x['lb'] > x['breakeven']:
                x['cat'] = 'edge'
            elif x['wr'] > x['breakeven']:
                x['cat'] = 'promising'
            else:
                x['cat'] = 'negative'
        return rows
