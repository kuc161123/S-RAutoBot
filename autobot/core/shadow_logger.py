"""
Shadow Learner (Phase 1a) — OBSERVATIONAL ONLY.

Logs every BOS-confirmed signal the bot evaluates (executed AND blocked) with its
intended entry/SL/TP and market context to Postgres, then resolves each outcome from
klines. Powers the /learn dashboard command.

DESIGN CONTRACT: this module must NEVER affect trading.
  - Every public method is wrapped and returns a safe default on ANY error.
  - If DATABASE_URL is missing or psycopg2 is unavailable, it silently no-ops.
  - Callers additionally wrap invocations in try/except (belt and suspenders).

Phase 1b additions:
  - `mfe_to_sl`: max favorable excursion (R) over bars strictly BEFORE the SL-hit
    bar (or the full walk if SL never hits). Enables counterfactual RR analysis:
    at RR' the signal is a win iff mfe_to_sl >= RR' (same atr_mult only). The SL
    bar is excluded so counterfactuals inherit the SL-wins-ties convention.
  - `family` ('DIV', 'DONCH', ...) and `source` ('live', 'scan') columns +
    `log_raw()` so shadow scanners (second families / out-of-universe symbols)
    can log signals through the same pipeline.

Data hygiene: 1H bars — intrabar order unknown, ambiguous bars resolve as SL
(reported WR is biased DOWN, i.e. conservative). Shadow entries are modeled
fills, not real executions.
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


def wilson_upper_bound(wins: int, n: int, z: float = 1.96) -> float:
    """95% Wilson upper bound on win-rate. Used to *condemn* a combo with
    confidence (UB below breakeven), the mirror of the lower-bound edge test."""
    if n <= 0:
        return 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + margin) / denom)


def walk_outcome(candles, entry, sl, tp, side, rr, horizon=200):
    """Pure candle walk shared by the resolver (and unit-testable in isolation).

    Args:
        candles: [(ts, open, high, low, close), ...] — post-signal bars only,
                 i.e. the first element is the bar AFTER the signal candle.
        entry/sl/tp: intended levels; side: 'long'|'short'; rr: configured R:R.
        horizon: max bars considered (default 200, matching the fetch limit).

    Returns (status, r_result, mfe, mae, mfe_to_sl, bars_to_outcome):
        status: 'win' | 'loss' | 'expired' (expired = no SL/TP hit in the walk;
                the CALLER decides whether expired-with-partial-data stays pending).
        mfe/mae: accumulated up to and including the outcome bar (legacy semantics,
                 unchanged for regression compatibility).
        mfe_to_sl: max favorable excursion in R over bars STRICTLY BEFORE the
                 SL-hit bar; keeps accumulating past a TP hit until SL/horizon.
                 On ambiguous bars (SL+TP same bar) the bar is excluded — the
                 conservative SL-wins-ties convention extends to counterfactuals.
        bars_to_outcome: 1-based index of the outcome bar (bars walked if expired).
    """
    risk = (entry - sl) if side == 'long' else (sl - entry)
    if risk <= 0:
        return 'expired', 0.0, 0.0, 0.0, 0.0, 0
    status, r_result = 'expired', 0.0
    mfe = mae = 0.0
    mfe_to_sl = 0.0
    bars_to_outcome = 0
    outcome_seen = False
    n_walked = 0
    for _, o, h, l, c in candles[:horizon]:
        n_walked += 1
        if side == 'long':
            bar_fav = (h - entry) / risk
            bar_adv = (l - entry) / risk
            hit_sl, hit_tp = l <= sl, h >= tp
        else:
            bar_fav = (entry - l) / risk
            bar_adv = (entry - h) / risk
            hit_sl, hit_tp = h >= sl, l <= tp
        if not outcome_seen:
            mfe = max(mfe, bar_fav)
            mae = min(mae, bar_adv)
        if hit_sl:
            # SL bar excluded from mfe_to_sl (SL-wins-ties, conservative)
            if not outcome_seen:
                status, r_result = 'loss', -1.0
                bars_to_outcome = n_walked
            break
        mfe_to_sl = max(mfe_to_sl, bar_fav)
        if hit_tp and not outcome_seen:
            status, r_result = 'win', float(rr)
            bars_to_outcome = n_walked
            outcome_seen = True
            # keep walking to accumulate mfe_to_sl until SL/horizon
    if bars_to_outcome == 0:
        bars_to_outcome = n_walked
    return status, r_result, mfe, mae, mfe_to_sl, bars_to_outcome


class ShadowLogger:
    MIN_N = 50            # minimum sample before a combo can be called "significant"
    RESOLVE_HORIZON_H = 1  # don't try to resolve a signal until it's at least this old
    RECHECK_COOLDOWN_H = 2  # after a "too early to grade" check, skip the signal this long

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
            # Phase 1b migration — additive, idempotent; failure degrades to Phase 1a behavior
            cur.execute("ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS mfe_to_sl DOUBLE PRECISION;")
            cur.execute("ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS bars_to_outcome INT;")
            cur.execute("ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS family TEXT DEFAULT 'DIV';")
            cur.execute("ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'live';")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_shadow_family_status ON shadow_signals(family, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_shadow_resolved_ts ON shadow_signals(resolved_ts);")
            # Anti-starvation: when the resolver examines a pending signal that
            # can't be graded yet (TP/SL untouched, <200 candles), it stamps
            # checked_ts and the fetch skips it for RECHECK_COOLDOWN_H, so slow
            # signals can't permanently occupy the oldest-first LIMIT window.
            cur.execute("ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS checked_ts TIMESTAMPTZ;")
            # Phase 1b-3: out-of-universe candidate watchlist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shadow_candidates (
                    symbol   TEXT PRIMARY KEY,
                    added_ts TIMESTAMPTZ DEFAULT now(),
                    turnover_24h DOUBLE PRECISION
                );
            """)
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

    # ---- write path for shadow scanners (second families / out-of-universe) ----
    def log_raw(self, spec: dict) -> str | None:
        """Insert a scanner-generated shadow signal. `spec` must contain:
        symbol, side, code, sig_time_ms, entry, sl, tp, rr, atr_mult, family;
        optional: source ('scan'), in_universe (False), regime, regime_mult,
        btc_bull, sym_chop, turnover. Dedup id includes atr_mult so the same
        setup can be tracked at several SL widths. OBSERVATION ONLY."""
        return self._safe(self._log_raw, spec)

    def _log_raw(self, spec):
        sig_ms = int(spec['sig_time_ms'])
        sid = (f"{spec['symbol']}:{spec['side']}:{spec['code']}:"
               f"{spec['atr_mult']}:{sig_ms // 3_600_000}")
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO shadow_signals
                  (id, sig_time_ms, symbol, side, div_type, in_universe, entry, sl, tp, rr,
                   atr_mult, regime, regime_mult, btc_bull, sym_chop, turnover, family, source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO NOTHING;
            """, (sid, sig_ms, spec['symbol'], spec['side'], spec['code'],
                  bool(spec.get('in_universe', False)),
                  float(spec['entry']), float(spec['sl']), float(spec['tp']),
                  float(spec['rr']), float(spec['atr_mult']),
                  spec.get('regime'), spec.get('regime_mult'),
                  spec.get('btc_bull'), spec.get('sym_chop'), spec.get('turnover'),
                  spec.get('family', 'DIV'), spec.get('source', 'scan')))
            conn.commit()
        return sid

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
                      AND (checked_ts IS NULL
                           OR checked_ts < now() - interval '%s hours')
                    ORDER BY ts ASC LIMIT %s;
                """ % (self.RESOLVE_HORIZON_H, self.RECHECK_COOLDOWN_H, int(limit_rows)))
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
            self._safe(self._touch_checked, r['id'])
            return  # too early — leave pending
        status, r_result, mfe, mae, mfe_to_sl, bars = walk_outcome(
            candles[1:],  # start on the candle after the signal candle
            r['entry'], r['sl'], r['tp'], r['side'], r['rr'])
        # if still pending status and the window is exhausted (200 candles), mark expired
        if status == 'expired' and len(candles) < 200:
            self._safe(self._touch_checked, r['id'])
            return  # not enough data yet, keep pending
        self._safe(self._write_resolution, r['id'], status, r_result, mfe, mae, mfe_to_sl, bars)

    def _touch_checked(self, sid):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE shadow_signals SET checked_ts=now() WHERE id=%s;", (sid,))
            conn.commit()

    def _write_resolution(self, sid, status, r_result, mfe, mae, mfe_to_sl=None, bars_to_outcome=None):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE shadow_signals
                SET status=%s, r_result=%s, mfe=%s, mae=%s,
                    mfe_to_sl=%s, bars_to_outcome=%s, resolved_ts=now()
                WHERE id=%s;
            """, (status, r_result, mfe, mae, mfe_to_sl, bars_to_outcome, sid))
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

    # ---- candidate watchlist (Phase 1b-3) ----
    def replace_candidates(self, rows):
        """rows: [(symbol, turnover_24h), ...] — full refresh of the watchlist."""
        return self._safe(self._replace_candidates, rows)

    def _replace_candidates(self, rows):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM shadow_candidates;")
            for sym, t in rows:
                cur.execute("""
                    INSERT INTO shadow_candidates (symbol, turnover_24h) VALUES (%s,%s)
                    ON CONFLICT (symbol) DO UPDATE SET turnover_24h=EXCLUDED.turnover_24h;
                """, (sym, t))
            conn.commit()

    def get_candidates(self):
        return self._safe(self._get_candidates, default=[]) or []

    def _get_candidates(self):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT symbol, turnover_24h FROM shadow_candidates ORDER BY turnover_24h DESC;")
            return [(r[0], r[1]) for r in cur.fetchall()]

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
