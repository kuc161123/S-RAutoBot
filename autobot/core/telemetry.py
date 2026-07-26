"""Telemetry — the measurements that analysis kept needing and not having.

DESIGN CONTRACT (same as the shadow layer): this module must NEVER affect trading.
Every public method is wrapped in _safe(); a dead database, a bad row or a schema
mismatch degrades to a debug log and returns None. Nothing here is ever awaited on the
critical path, and no caller checks its return value for a trading decision.

Five tables, each answering a question that was unanswerable in the 2026-07 analysis:

  exec_log         intended vs ACTUAL fill, real fee, real funding, slippage.
                   Cost drag is ~66% of this strategy's gross edge and was, until now,
                   an assumed constant (0.0018 round-trip) rather than a measurement.

  market_state     one row per scan: breadth, dispersion, BTC regime, funding.
                   The live regime classifier reads the bot's own last 20 trades and
                   flips tiers on ~70% of draws given a CONSTANT true edge — it is
                   mostly noise. Market state is measured across the whole universe and
                   is the only realistic path to replacing it with something real.

  config_history   what was live, when, and what data it was fitted on.
                   Nine of ten backtest windows in the 2026-07 study were contaminated
                   because this was reconstructed from commit archaeology after the fact.

  signal_blocks    which gate rejected each signal, with the values at the time.
                   Turns every future protection ablation into a query.

  equity_snapshots periodic equity/risk/regime state, so the LIVE equity curve can be
                   reconstructed. lifetime_stats is a single overwritten blob today.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    _PG_OK = True
except Exception:  # pragma: no cover
    psycopg2 = None
    _PG_OK = False


class Telemetry:
    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv("DATABASE_URL") or None
        self.enabled = bool(self.db_url) and _PG_OK
        self._last_market = 0.0
        self._last_equity = 0.0
        if self.enabled:
            self._safe(self._ensure_schema)
            logger.info("[TELEM] Telemetry enabled (Postgres).")
        else:
            logger.info("[TELEM] Telemetry disabled (no DATABASE_URL or psycopg2).")

    # ---- infra --------------------------------------------------------------
    def _conn(self):
        return psycopg2.connect(self.db_url)

    def _safe(self, fn, *a, default=None, **kw):
        if not self.enabled:
            return default
        try:
            return fn(*a, **kw)
        except Exception as e:
            logger.debug(f"[TELEM] {getattr(fn, '__name__', fn)} failed: {e}")
            return default

    def _ensure_schema(self):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exec_log (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT now(),
                    trade_key TEXT, order_id TEXT,
                    symbol TEXT, side TEXT, div_type TEXT,
                    intended_entry DOUBLE PRECISION,
                    actual_entry   DOUBLE PRECISION,
                    slippage_frac  DOUBLE PRECISION,
                    intended_sl DOUBLE PRECISION, intended_tp DOUBLE PRECISION,
                    qty DOUBLE PRECISION, notional DOUBLE PRECISION,
                    risk_usd DOUBLE PRECISION,
                    atr DOUBLE PRECISION, atr_mult DOUBLE PRECISION, rr DOUBLE PRECISION,
                    leverage INT,
                    signal_bar_ms BIGINT, fill_lag_sec DOUBLE PRECISION,
                    regime TEXT, regime_mult DOUBLE PRECISION,
                    equity DOUBLE PRECISION,
                    exit_ts TIMESTAMPTZ, actual_exit DOUBLE PRECISION,
                    realized_pnl DOUBLE PRECISION,
                    fee_usd DOUBLE PRECISION, funding_usd DOUBLE PRECISION,
                    r_result DOUBLE PRECISION, outcome TEXT,
                    hold_hours DOUBLE PRECISION
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS ix_exec_symbol ON exec_log(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_exec_open ON exec_log(trade_key) "
                        "WHERE exit_ts IS NULL;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_state (
                    ts TIMESTAMPTZ PRIMARY KEY,
                    n_symbols INT,
                    pct_above_ema200 DOUBLE PRECISION,
                    pct_new_high_20 DOUBLE PRECISION,
                    pct_new_low_20  DOUBLE PRECISION,
                    median_chop DOUBLE PRECISION,
                    median_adx  DOUBLE PRECISION,
                    median_atr_pct DOUBLE PRECISION,
                    median_rsi DOUBLE PRECISION,
                    btc_close DOUBLE PRECISION,
                    btc_ret_30d DOUBLE PRECISION,
                    btc_chop DOUBLE PRECISION,
                    btc_adx DOUBLE PRECISION,
                    btc_above_ema200 BOOLEAN,
                    median_corr_btc DOUBLE PRECISION,
                    median_funding DOUBLE PRECISION
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS config_history (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT now(),
                    git_sha TEXT, config_hash TEXT,
                    risk_per_trade DOUBLE PRECISION,
                    n_symbols INT, n_configs INT,
                    risk_json JSONB, note TEXT
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_blocks (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT now(),
                    symbol TEXT, side TEXT, div_type TEXT,
                    reason TEXT, detail JSONB,
                    regime TEXT, equity DOUBLE PRECISION
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS ix_blocks_reason ON signal_blocks(reason);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_blocks_ts ON signal_blocks(ts);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    ts TIMESTAMPTZ PRIMARY KEY,
                    equity DOUBLE PRECISION, wallet DOUBLE PRECISION,
                    open_positions INT,
                    open_risk_usd DOUBLE PRECISION,
                    net_dir_risk_usd DOUBLE PRECISION,
                    long_positions INT, short_positions INT,
                    regime TEXT, regime_mult DOUBLE PRECISION,
                    risk_pct DOUBLE PRECISION,
                    shadow_gate_open BOOLEAN, trading_enabled BOOLEAN,
                    daily_r DOUBLE PRECISION, total_r DOUBLE PRECISION,
                    unrealized_pnl DOUBLE PRECISION
                );
            """)
            conn.commit()

    # ---- 1. execution truth -------------------------------------------------
    def log_entry(self, **kw):
        """Called right after a bracket order fills. Records intent vs reality."""
        return self._safe(self._log_entry, **kw)

    def _log_entry(self, trade_key, order_id, symbol, side, div_type,
                   intended_entry, actual_entry, intended_sl, intended_tp,
                   qty, risk_usd, atr, atr_mult, rr, leverage,
                   signal_bar_ms, regime, regime_mult, equity):
        slip = None
        if intended_entry:
            raw = (actual_entry - intended_entry) / intended_entry
            # sign so that POSITIVE always means "filled worse than intended"
            slip = raw if side == "long" else -raw
        lag = None
        if signal_bar_ms:
            lag = time.time() - (signal_bar_ms / 1000.0)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO exec_log (trade_key, order_id, symbol, side, div_type,
                    intended_entry, actual_entry, slippage_frac, intended_sl, intended_tp,
                    qty, notional, risk_usd, atr, atr_mult, rr, leverage,
                    signal_bar_ms, fill_lag_sec, regime, regime_mult, equity)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """, (trade_key, order_id, symbol, side, div_type,
                  intended_entry, actual_entry, slip, intended_sl, intended_tp,
                  qty, (qty or 0) * (actual_entry or 0), risk_usd, atr, atr_mult, rr,
                  leverage, signal_bar_ms, lag, regime, regime_mult, equity))
            conn.commit()

    def log_exit(self, **kw):
        """Called from handle_trade_exit once the exchange reports the close."""
        return self._safe(self._log_exit, **kw)

    def _log_exit(self, trade_key, actual_exit, realized_pnl, fee_usd, funding_usd,
                  r_result, outcome, hold_hours):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE exec_log SET exit_ts = now(), actual_exit = %s, realized_pnl = %s,
                    fee_usd = %s, funding_usd = %s, r_result = %s, outcome = %s,
                    hold_hours = %s
                WHERE id = (SELECT id FROM exec_log WHERE trade_key = %s
                            AND exit_ts IS NULL ORDER BY ts DESC LIMIT 1);
            """, (actual_exit, realized_pnl, fee_usd, funding_usd, r_result, outcome,
                  hold_hours, trade_key))
            conn.commit()

    # ---- 2. market state ----------------------------------------------------
    def log_market_state(self, snap: dict, min_interval_s: float = 1800):
        now = time.time()
        if now - self._last_market < min_interval_s:
            return None
        self._last_market = now
        return self._safe(self._log_market_state, snap)

    def _log_market_state(self, snap):
        cols = ["n_symbols", "pct_above_ema200", "pct_new_high_20", "pct_new_low_20",
                "median_chop", "median_adx", "median_atr_pct", "median_rsi",
                "btc_close", "btc_ret_30d", "btc_chop", "btc_adx", "btc_above_ema200",
                "median_corr_btc", "median_funding"]
        vals = [snap.get(c) for c in cols]
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO market_state (ts, {','.join(cols)})
                VALUES (date_trunc('minute', now()), {','.join(['%s'] * len(cols))})
                ON CONFLICT (ts) DO NOTHING;
            """, vals)
            conn.commit()

    # ---- 3. config provenance ----------------------------------------------
    def log_config(self, risk_cfg: dict, n_symbols: int, n_configs: int,
                   note: str = "startup"):
        return self._safe(self._log_config, risk_cfg, n_symbols, n_configs, note)

    def _log_config(self, risk_cfg, n_symbols, n_configs, note):
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            sha = None
        blob = json.dumps(risk_cfg, sort_keys=True, default=str)
        import hashlib
        chash = hashlib.sha256(blob.encode()).hexdigest()[:16]
        with self._conn() as conn, conn.cursor() as cur:
            # only write when something actually changed
            cur.execute("SELECT config_hash FROM config_history ORDER BY ts DESC LIMIT 1;")
            row = cur.fetchone()
            if row and row[0] == chash:
                return
            cur.execute("""
                INSERT INTO config_history (git_sha, config_hash, risk_per_trade,
                    n_symbols, n_configs, risk_json, note)
                VALUES (%s,%s,%s,%s,%s,%s,%s);
            """, (sha, chash, risk_cfg.get("risk_per_trade"), n_symbols, n_configs,
                  json.dumps(risk_cfg, default=str), note))
            conn.commit()
            logger.info(f"[TELEM] config change recorded ({chash}, git {sha})")

    # ---- 4. block reasons ---------------------------------------------------
    def log_block(self, symbol, side, div_type, reason, detail=None,
                  regime=None, equity=None):
        return self._safe(self._log_block, symbol, side, div_type, reason,
                          detail or {}, regime, equity)

    def _log_block(self, symbol, side, div_type, reason, detail, regime, equity):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signal_blocks (symbol, side, div_type, reason, detail,
                    regime, equity) VALUES (%s,%s,%s,%s,%s,%s,%s);
            """, (symbol, side, div_type, reason, json.dumps(detail, default=str),
                  regime, equity))
            conn.commit()

    # ---- 5. equity snapshots ------------------------------------------------
    def log_equity(self, snap: dict, min_interval_s: float = 900):
        now = time.time()
        if now - self._last_equity < min_interval_s:
            return None
        self._last_equity = now
        return self._safe(self._log_equity, snap)

    def _log_equity(self, snap):
        cols = ["equity", "wallet", "open_positions", "open_risk_usd",
                "net_dir_risk_usd", "long_positions", "short_positions",
                "regime", "regime_mult", "risk_pct", "shadow_gate_open",
                "trading_enabled", "daily_r", "total_r", "unrealized_pnl"]
        vals = [snap.get(c) for c in cols]
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO equity_snapshots (ts, {','.join(cols)})
                VALUES (date_trunc('minute', now()), {','.join(['%s'] * len(cols))})
                ON CONFLICT (ts) DO NOTHING;
            """, vals)
            conn.commit()
