"""
Shadow Scanner (Phase 1b) — OBSERVATIONAL ONLY.

Runs additional signal detection for the shadow learner:
  - Phase 1b-2: second-family (Donchian breakout) detection on the SAME prepared
    dataframes the live loop already fetched (observe_df — zero extra API calls).
  - Phase 1b-3: out-of-universe candidate symbol scanning (tick — throttled,
    public-data reads only).

DESIGN CONTRACT (same as ShadowLogger): this module must NEVER affect trading.
  - observe_df/tick swallow ALL exceptions internally and never mutate the bot,
    the broker, or the dataframes they are given (reads only).
  - Output goes exclusively to shadow_signals via ShadowLogger.log_raw, which is
    itself _safe-wrapped and no-ops without a database.

Entry convention mirrors the live bot: a signal fires on a CLOSED bar; the entry
is the NEXT bar's open. Because the next open is unknown when the signal bar
closes, specs wait in a one-bar in-memory queue and are completed on the next
observe_df call for that symbol. A restart loses at most one bar of queued
shadow signals (accepted, documented).
"""
from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import yaml

from autobot.core.breakout_detector import detect_donchian_last_bar, COOLDOWN_BARS

logger = logging.getLogger(__name__)

# Phase 1b-3 knobs
CANDIDATE_MIN_TURNOVER_24H = 5_000_000.0   # $5M/day liquidity floor
CANDIDATE_CAP = 100
DISCOVERY_INTERVAL_S = 24 * 3600           # refresh watchlist daily
SCAN_INTERVAL_S = 3300                     # scan candidates ~hourly
SCAN_STAGGER_S = 0.3                       # between kline fetches
CANDIDATE_ATR_MULTS = (1.0, 1.5, 2.0)      # no per-symbol config exists yet
CANDIDATE_NOMINAL_RR = 5.0                 # RR dimension recovered from mfe_to_sl
RECENT_ENTRY_WINDOW_S = 24 * 3600          # only log entries from the last day


class ShadowScanner:
    def __init__(self, shadow_logger, families_path: str = 'shadow_families.yaml',
                 bot=None, dry_run: bool = False):
        self.shadow_logger = shadow_logger
        self.bot = bot                      # optional, best-effort context only
        self.dry_run = dry_run
        self.enabled = bool(getattr(shadow_logger, 'enabled', False)) or dry_run
        # {symbol: [family config dicts]} from shadow_families.yaml
        self.family_configs: dict[str, list[dict]] = {}
        # one-bar queue: {symbol: [partial spec dicts awaiting next-bar open]}
        self._queued: dict[str, list[dict]] = {}
        # cooldown: {(symbol, code): last signal ts (epoch ms)}
        self._last_sig: dict[tuple, int] = {}
        # Phase 1b-3 state
        self._candidates: list[str] = []
        self._last_discovery = 0.0
        self._last_scan = 0.0
        try:
            with open(families_path) as f:
                data = yaml.safe_load(f) or {}
            for cfg in data.get('configs', []):
                self.family_configs.setdefault(cfg['symbol'], []).append(cfg)
            n_cfg = sum(len(v) for v in self.family_configs.values())
            logger.info(f"[SHADOW-SCAN] Loaded {n_cfg} family configs "
                        f"across {len(self.family_configs)} symbols "
                        f"({'DRY RUN' if dry_run else 'live logging'})")
        except Exception as e:
            logger.warning(f"[SHADOW-SCAN] families file not loaded ({e}) — scanner idle")
            self.enabled = False

    # ---- context helpers (all best-effort) ----
    def _context(self, df):
        ctx = {}
        try:
            if self.bot is not None:
                regime, rmult, _ = self.bot.get_regime_status()
                ctx['regime'] = regime
                ctx['regime_mult'] = rmult
        except Exception:
            pass
        try:
            if self.bot is not None:
                ctx['btc_bull'] = self.bot._btc_trend_cache.get('bullish')
        except Exception:
            pass
        try:
            if 'chop' in df.columns:
                ctx['sym_chop'] = float(df['chop'].iloc[-1])
        except Exception:
            pass
        try:
            if 'turnover' in df.columns:
                ctx['turnover'] = float(df['turnover'].iloc[-1])
        except Exception:
            pass
        return ctx

    # ---- Phase 1b-2: in-universe second family, rides the live data fetch ----
    def observe_df(self, symbol: str, df) -> int:
        """Called from process_symbol on every new closed candle, AFTER
        prepare_dataframe. Never raises; returns number of shadow rows logged."""
        if not self.enabled:
            return 0
        try:
            return self._observe_df(symbol, df)
        except Exception as e:
            logger.debug(f"[SHADOW-SCAN] observe_df({symbol}) failed: {e}")
            return 0

    def _observe_df(self, symbol, df):
        cfgs = self.family_configs.get(symbol)
        if not cfgs or df is None or len(df) < 60:
            return 0
        logged = 0
        bar_ts = df.index[-1]
        bar_ms = int(bar_ts.timestamp() * 1000)

        # 1) complete queued specs: entry = this (new) bar's open, but only if
        #    this bar is exactly the bar after the signal bar (no gaps/restarts)
        pending = self._queued.pop(symbol, [])
        for spec in pending:
            if bar_ms - spec['sig_time_ms'] != 3_600_000:
                continue  # gap (restart / missed hour) — drop, stale entry price
            entry = float(df['open'].iloc[-1])
            sld = spec.pop('_atr') * spec['atr_mult']
            if sld <= 0:
                continue
            if spec['side'] == 'long':
                spec['sl'] = entry - sld
                spec['tp'] = entry + sld * spec['rr']
            else:
                spec['sl'] = entry + sld
                spec['tp'] = entry - sld * spec['rr']
            spec['entry'] = entry
            if self.dry_run:
                logger.info(f"[SHADOW-SCAN] DRY {symbol} {spec['code']} "
                            f"entry={entry:.6g} sl={spec['sl']:.6g} tp={spec['tp']:.6g}")
                logged += 1
            else:
                if self.shadow_logger.log_raw(spec):
                    logged += 1

        # 2) detect on the just-closed bar and queue for next-bar entry
        hits = detect_donchian_last_bar(df)
        if not hits:
            return logged
        atr = None
        try:
            atr = float(df['atr'].iloc[-1])
        except Exception:
            pass
        if not atr or atr <= 0:
            return logged
        ctx = self._context(df)
        by_code = {}
        for c in cfgs:
            by_code.setdefault(c['code'], []).append(c)
        for hit in hits:
            for cfg in by_code.get(hit['code'], []):
                key = (symbol, cfg['code'])
                last = self._last_sig.get(key, 0)
                if bar_ms - last < COOLDOWN_BARS * 3_600_000:
                    continue
                self._last_sig[key] = bar_ms
                self._queued.setdefault(symbol, []).append({
                    'symbol': symbol,
                    'side': hit['side'],
                    'code': cfg['code'],
                    'sig_time_ms': bar_ms,
                    'rr': float(cfg['rr']),
                    'atr_mult': float(cfg['atr_mult']),
                    'family': 'DONCH',
                    'source': 'scan',
                    'in_universe': True,
                    '_atr': atr,
                    **ctx,
                })
        return logged

    # ---- Phase 1b-3: out-of-universe candidate scouting ----
    async def tick(self, broker) -> int:
        """Called from the main loop. Internally throttled (discovery daily,
        scan ~hourly). Never raises; returns number of shadow rows logged."""
        if not self.enabled or broker is None:
            return 0
        try:
            return await self._tick(broker)
        except Exception as e:
            logger.debug(f"[SHADOW-SCAN] tick failed: {e}")
            return 0

    async def _tick(self, broker):
        now = time.time()
        if not self._candidates and self._last_discovery == 0.0:
            # restore persisted watchlist on first tick after restart
            self._candidates = [s for s, _ in self.shadow_logger.get_candidates()]
        if now - self._last_discovery >= DISCOVERY_INTERVAL_S:
            await self._discover_candidates(broker)
            self._last_discovery = now
        if now - self._last_scan < SCAN_INTERVAL_S or not self._candidates:
            return 0
        self._last_scan = now
        logged = 0
        for sym in self._candidates:
            try:
                logged += await self._scan_candidate(broker, sym)
            except Exception as e:
                logger.debug(f"[SHADOW-SCAN] candidate {sym} failed: {e}")
            await asyncio.sleep(SCAN_STAGGER_S)
        if logged:
            logger.info(f"[SHADOW-SCAN] candidate sweep logged {logged} shadow signals "
                        f"({len(self._candidates)} symbols)")
        return logged

    async def _discover_candidates(self, broker):
        tickers = await broker.get_all_tickers()
        if not tickers:
            return
        enabled = set()
        try:
            enabled = set(self.bot.symbol_config.get_enabled_symbols())
        except Exception:
            pass
        if not enabled:
            return  # can't tell what's already traded — don't guess
        rows = []
        for t in tickers:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT') or '-' in sym or sym in enabled:
                continue
            try:
                turn = float(t.get('turnover24h') or 0)
            except (TypeError, ValueError):
                continue
            if turn >= CANDIDATE_MIN_TURNOVER_24H:
                rows.append((sym, turn))
        rows.sort(key=lambda x: -x[1])
        rows = rows[:CANDIDATE_CAP]
        self._candidates = [s for s, _ in rows]
        self.shadow_logger.replace_candidates(rows)
        logger.info(f"[SHADOW-SCAN] candidate watchlist refreshed: {len(rows)} symbols")

    async def _scan_candidate(self, broker, symbol):
        """Replicate the live divergence pipeline on one out-of-universe symbol:
        detect -> BOS within 12 bars -> EMA alignment at BOS -> entry next open.
        Logs one row per atr_mult at a nominal RR (counterfactual RRs come from
        mfe_to_sl at analysis time). sig_time_ms = BOS bar, so resolution starts
        exactly at the entry bar."""
        from autobot.core.divergence_detector import (
            prepare_dataframe, detect_divergences, check_bos, is_trend_aligned)
        kl = await broker.get_klines(symbol, '60', limit=300)
        if not kl or len(kl) < 260:
            return 0
        df = pd.DataFrame(kl, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype('int64'), unit='ms')
        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = df[col].astype(float)
        df = df.set_index('start').sort_index()
        df = prepare_dataframe(df)
        signals = detect_divergences(df, symbol)
        if not signals:
            return 0
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        n = len(df)
        logged = 0
        for sig in signals:
            for j in range(sig.divergence_idx + 1, min(sig.divergence_idx + 13, n - 1)):
                if not check_bos(df, sig, j):
                    continue
                if not is_trend_aligned(df, sig, j):
                    break  # BOS happened but EMA gate failed — signal dies (live semantics)
                entry_idx = j + 1
                bos_ms = int(df.index[j].timestamp() * 1000)
                if now_ms - bos_ms > RECENT_ENTRY_WINDOW_S * 1000:
                    break  # historical — dedup would likely skip anyway; don't backfill
                entry = float(df['open'].iloc[entry_idx])
                atr = float(df['atr'].iloc[j])
                if not atr or atr <= 0 or entry <= 0:
                    break
                for am in CANDIDATE_ATR_MULTS:
                    sld = atr * am
                    if sig.side == 'long':
                        sl, tp = entry - sld, entry + sld * CANDIDATE_NOMINAL_RR
                    else:
                        sl, tp = entry + sld, entry - sld * CANDIDATE_NOMINAL_RR
                    spec = {
                        'symbol': symbol, 'side': sig.side, 'code': sig.divergence_code,
                        'sig_time_ms': bos_ms, 'entry': entry, 'sl': sl, 'tp': tp,
                        'rr': CANDIDATE_NOMINAL_RR, 'atr_mult': am,
                        'family': 'DIV', 'source': 'scan', 'in_universe': False,
                    }
                    if self.dry_run:
                        logger.info(f"[SHADOW-SCAN] DRY candidate {symbol} {sig.divergence_code} "
                                    f"am={am} entry={entry:.6g}")
                        logged += 1
                    elif self.shadow_logger.log_raw(spec):
                        logged += 1
                break  # first BOS resolution only (live takes one entry per signal)
        return logged
