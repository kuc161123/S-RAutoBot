#!/usr/bin/env python3
"""
Pretraining runner for Trend Pullback strategy (server mode)

Runs an offline backtest that reuses the live detector (strategy_pullback.detect_signal_pullback)
with 15m candles and 3m microframe data, simulates outcomes conservatively, logs metrics, and
feeds results into the Trend ML scorer so the bot starts with a trained model.

Environment variables (optional):
- PRETRAIN_SYMBOLS: comma-separated list of symbols; defaults to top 30 from DB
- PRETRAIN_START_DAYS: lookback days (default 21)
- PRETRAIN_MAX_TRADES: cap per-symbol signals to process (default 200)
- FEES_BPS: total round-trip fees in basis points (default 11.5)
- SLIPPAGE_BPS: slippage per fill in basis points (default 5)
- DIVERGENCE_MODE: off|optional|strict (default uses Settings.div_mode)
- BOS_HOLD_MINUTES: int, if provided overrides Settings.bos_armed_hold_minutes
- BREAKOUT_BUFFER_ATR: float, overrides Settings.breakout_buffer_atr

This script is intentionally lightweight: it focuses on fidelity to live logic, clean logs,
and feeding ML. It does not persist backtest trades to Postgres (can be added later).
"""
from __future__ import annotations

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from candle_storage_postgres import CandleStorage
from backtest_persistence import BacktestPersistence
from strategy_pullback import (
    Settings as TrendSettings,
    Signal as TrendSignal,
    detect_signal_pullback,
    set_trend_microframe_provider,
    set_trend_event_notifier,
    reset_symbol_state,
    breakout_states,
)
from ml_scorer_trend import get_trend_scorer

logger = logging.getLogger("pretrain")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Pretrain] %(levelname)s - %(message)s')


def _fees_and_slippage_multiplier() -> float:
    try:
        fees_bps = float(os.getenv('FEES_BPS', '11.5'))  # e.g., ~0.115% round-trip
    except Exception:
        fees_bps = 11.5
    try:
        slip_bps = float(os.getenv('SLIPPAGE_BPS', '5.0'))  # additional slippage
    except Exception:
        slip_bps = 5.0
    total = (fees_bps + slip_bps) / 10000.0
    return 1.0 + total


def _conservative_outcome_3m(df3: pd.DataFrame, entry_idx: int, side: str, entry: float, sl: float, tp: float) -> Tuple[str, float]:
    """
    Conservative outcome labeling on 3m bars with SL-first tie-break.
    Returns: (outcome: 'win'|'loss', realized_r: float)
    """
    mult = _fees_and_slippage_multiplier()
    # Nudge target to account for costs so R is post-cost realistic
    if side == 'long':
        adj_tp = float(tp) * mult
        adj_sl = float(sl) / mult if sl > 0 else float(sl)
        R = abs(entry - adj_sl)
        for i in range(entry_idx + 1, len(df3)):
            hi = float(df3['high'].iloc[i]); lo = float(df3['low'].iloc[i])
            hit_tp = hi >= adj_tp
            hit_sl = lo <= adj_sl
            if hit_tp and hit_sl:
                return 'loss', -1.0  # SL-first
            if hit_tp:
                return 'win', (adj_tp - entry) / max(1e-9, R)
            if hit_sl:
                return 'loss', (adj_sl - entry) / max(1e-9, R)
    else:
        adj_tp = float(tp) / mult
        adj_sl = float(sl) * mult
        R = abs(entry - adj_sl)
        for i in range(entry_idx + 1, len(df3)):
            hi = float(df3['high'].iloc[i]); lo = float(df3['low'].iloc[i])
            hit_tp = lo <= adj_tp
            hit_sl = hi >= adj_sl
            if hit_tp and hit_sl:
                return 'loss', -1.0
            if hit_tp:
                return 'win', (entry - adj_tp) / max(1e-9, R)
            if hit_sl:
                return 'loss', (entry - adj_sl) / max(1e-9, R)
    # No outcome → conservative loss
    return 'loss', -0.5


def _build_features(df15: pd.DataFrame, df3: Optional[pd.DataFrame], sig: TrendSignal, st) -> Dict:
    """Build feature dict aligned with ml_scorer_trend expectations."""
    try:
        close = df15['close']; high = df15['high']; low = df15['low']
        price = float(close.iloc[-1])
        prev = close.shift()
        tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.iloc[-1])
        atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
        # Break distance and retrace depth in ATRs
        break_dist_atr = 0.0
        retrace_depth_atr = 0.0
        try:
            if st and float(st.breakout_level or 0.0) > 0 and price > 0 and atr > 0:
                if sig.side == 'long':
                    break_dist_atr = (price - float(st.breakout_level)) / atr
                    if float(st.pullback_extreme or 0.0) > 0:
                        retrace_depth_atr = (float(st.breakout_level) - float(st.pullback_extreme)) / atr
                else:
                    break_dist_atr = (float(st.breakout_level) - price) / atr
                    if float(st.pullback_extreme or 0.0) > 0:
                        retrace_depth_atr = (float(st.pullback_extreme) - float(st.breakout_level)) / atr
        except Exception:
            pass
        # EMA and range expansion
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(close) >= 20 else price
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
        ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else (50.0 if ema20 != ema50 else 0.0)
        rng_today = float(high.iloc[-1] - low.iloc[-1])
        med_range = float((high - low).rolling(20).median().iloc[-1]) if len(df15) >= 20 else max(1e-9, rng_today)
        range_expansion = float(rng_today / max(1e-9, med_range))
        # Divergence fields from state
        div_ok = bool(getattr(st, 'divergence_ok', False))
        div_score = float(getattr(st, 'divergence_score', 0.0))
        div_rsi_delta = float(getattr(st, 'div_rsi_delta', 0.0))
        div_tsi_delta = float(getattr(st, 'div_tsi_delta', 0.0))
        # Session/volatility placeholders (could be improved)
        session = 'us'
        vol_regime = 'normal'
        # Symbol cluster if available
        sym_cluster = 3
        try:
            from symbol_clustering import load_symbol_clusters
            # We'll set symbol later when available to lookup cluster
        except Exception:
            pass
        features = {
            'atr_pct': atr_pct,
            'break_dist_atr': float(break_dist_atr),
            'retrace_depth_atr': float(max(0.0, retrace_depth_atr)),
            'confirm_candles': int(getattr(st, 'confirmation_count', 0) or 0),
            'ema_stack_score': ema_stack_score,
            'range_expansion': range_expansion,
            'div_ok': 1 if div_ok else 0,
            'div_score': float(div_score),
            'div_rsi_delta': float(div_rsi_delta),
            'div_tsi_delta': float(div_tsi_delta),
            'session': session,
            'symbol_cluster': sym_cluster,
            'volatility_regime': vol_regime,
        }
        return features
    except Exception as e:
        logger.warning(f"Feature build error: {e}")
        return {}


def _pick_symbols(storage: CandleStorage) -> List[str]:
    env_syms = os.getenv('PRETRAIN_SYMBOLS')
    if env_syms:
        return [s.strip().upper() for s in env_syms.split(',') if s.strip()]
    # Fallback: attempt to load a broad set from config.yaml if present
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        syms = cfg.get('symbols') or cfg.get('symbol_universe') or []
        if isinstance(syms, list) and syms:
            return [str(s).upper() for s in syms][:50]
    except Exception:
        pass
    # Last resort: pull recent symbols by scanning DB via stats/load_all_frames (not cheap)
    logger.info("PRETRAIN_SYMBOLS not set and config has no universe; defaulting to a small common set")
    return [
        'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','AVAXUSDT','LINKUSDT','DOTUSDT'
    ]


def run_pretraining(
    do_sweep: bool = False,
    wf_folds: int = 1,
    persist_db: bool = False,
    sweep_spec: Optional[str] = None,
    max_minutes: Optional[int] = None,
) -> None:
    storage = CandleStorage()

    # Settings baseline
    base = TrendSettings()
    # Overrides via env
    div_mode = os.getenv('DIVERGENCE_MODE')
    if div_mode:
        if div_mode in ('off','optional','strict'):
            base = replace(base, div_enabled=(div_mode!='off'), div_mode=div_mode)
    try:
        hold = os.getenv('BOS_HOLD_MINUTES')
        if hold:
            base = replace(base, bos_armed_hold_minutes=int(hold))
    except Exception:
        pass
    try:
        bbuf = os.getenv('BREAKOUT_BUFFER_ATR')
        if bbuf:
            base = replace(base, breakout_buffer_atr=float(bbuf))
    except Exception:
        pass

    # Lookback window
    try:
        start_days = int(os.getenv('PRETRAIN_START_DAYS', '21'))
    except Exception:
        start_days = 21
    # Use UTC-aware timestamp directly; avoid tz_localize on tz-aware values
    start_ts = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=start_days)

    # Cap signals per symbol for speed
    try:
        max_trades = int(os.getenv('PRETRAIN_MAX_TRADES', '200'))
    except Exception:
        max_trades = 200

    # Register a microframe provider that respects a per-symbol cutoff timestamp
    df3m_cache: Dict[str, pd.DataFrame] = {}
    micro_cutoff: Dict[str, pd.Timestamp] = {}

    def _micro_provider(sym: str) -> Optional[pd.DataFrame]:
        d3 = df3m_cache.get(sym)
        if d3 is None:
            d3 = storage.load_candles_3m(sym, limit=None)
            if d3 is None or len(d3) < 50:
                return None
            df3m_cache[sym] = d3
        cut = micro_cutoff.get(sym)
        if cut is not None:
            return d3.loc[:cut]
        return d3

    set_trend_microframe_provider(_micro_provider)

    # Simple notifier to stream important events to logs only (no Telegram)
    def _notify(sym: str, text: str):
        logger.info(f"{text}")
    set_trend_event_notifier(_notify)

    # ML scorer
    tml = get_trend_scorer(enabled=True)

    universe = _pick_symbols(storage)
    logger.info(f"Universe: {len(universe)} symbols: {', '.join(universe)}")

    # DB persistence
    db = BacktestPersistence() if persist_db else None

    def _run_single_variant(label: str, settings: TrendSettings) -> Dict:
        total_signals = 0
        total_wins = 0
        total_losses = 0
        r_values: List[float] = []
        trades_for_db: List[Dict] = []
        run_id = 0
        if db:
            try:
                run_id = db.start_run(universe, settings.__dict__, label)
            except Exception as e:
                logger.warning(f"DB run start failed: {e}")

        for sym in universe:
            try:
                df15 = storage.load_candles(sym, limit=None)
                if df15 is None or len(df15) < 300:
                    logger.info(f"[{sym}] Skipping (insufficient 15m candles)")
                    continue
                # Restrict to lookback window
                df15 = df15[df15.index >= start_ts]
                if len(df15) < 200:
                    logger.info(f"[{sym}] Lookback too short after filter; skipping")
                    continue
                # Ensure 3m exists
                d3 = storage.load_candles_3m(sym, limit=None)
                if d3 is None or len(d3) < 200:
                    logger.info(f"[{sym}] Skipping (no 3m history)")
                    continue

                # Reset strategy state for symbol
                reset_symbol_state(sym)

                sym_signals = 0
                sym_wins = 0
                sym_losses = 0
                logger.info(f"[{sym}] Pretraining over {len(df15)} x 15m candles, start {df15.index[0]} → {df15.index[-1]}")

                # Optional walk-forward folds
                folds = [(0, len(df15))]
                if wf_folds and wf_folds > 1:
                    fold_size = len(df15) // wf_folds
                    folds = [(i*fold_size, (i+1)*fold_size if i < wf_folds-1 else len(df15)) for i in range(wf_folds)]

                for (lo, hi) in folds:
                    for idx in range(max(200, lo), hi):
                        micro_cutoff[sym] = df15.index[idx]
                        df_slice = df15.iloc[:idx+1]
                        s = settings
                        try:
                            sig = detect_signal_pullback(df_slice, s, symbol=sym)
                        except Exception as e:
                            logger.debug(f"[{sym}] detect error at {df_slice.index[-1]}: {e}")
                            continue
                        if not sig:
                            continue
                        st = breakout_states.get(sym)
                        feats = _build_features(df_slice, _micro_provider(sym), sig, st)

                        d3_now = _micro_provider(sym)
                        if d3_now is None or len(d3_now) < 5:
                            continue
                        entry_time = d3_now.index[-1]
                        d3_full = df3m_cache.get(sym)
                        if d3_full is None:
                            d3_full = storage.load_candles_3m(sym, limit=None)
                            if d3_full is None:
                                continue
                            df3m_cache[sym] = d3_full
                        entry_idx = d3_full.index.get_indexer([entry_time], method='nearest')[0]
                        outcome, r_realized = _conservative_outcome_3m(d3_full, entry_idx, sig.side, float(sig.entry), float(sig.sl), float(sig.tp))

                        total_signals += 1; sym_signals += 1
                        r_values.append(float(r_realized))
                        if outcome == 'win':
                            total_wins += 1; sym_wins += 1
                        else:
                            total_losses += 1; sym_losses += 1
                        logger.info(f"[{sym}] {label} {outcome.upper()} {sig.side} @ {sig.entry:.4f} SL {sig.sl:.4f} TP {sig.tp:.4f} | R={r_realized:+.2f}")

                        try:
                            rec = {'features': feats, 'was_executed': 1}
                            tml.record_outcome(rec, 'win' if outcome=='win' else 'loss', pnl_percent=0.0)
                        except Exception:
                            pass

                        if db:
                            try:
                                trades_for_db.append({
                                    'symbol': sym,
                                    'side': sig.side,
                                    'entry': float(sig.entry),
                                    'sl': float(sig.sl),
                                    'tp': float(sig.tp),
                                    'entry_time': entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time,
                                    'outcome': outcome,
                                    'realized_r': float(r_realized),
                                    'meta': {
                                        'div_ok': bool(getattr(st, 'divergence_ok', False)),
                                        'bos_hold_minutes': int(getattr(settings, 'bos_armed_hold_minutes', 0)),
                                        'breakout_level': float(getattr(st, 'breakout_level', 0.0) or 0.0)
                                    }
                                })
                            except Exception:
                                pass

                        if sym_signals >= max_trades:
                            break

            except Exception as e:
                logger.warning(f"[{sym}] Pretraining error: {e}")

        avg_r = float(np.mean(r_values)) if r_values else 0.0
        ev_r = avg_r
        wr = (total_wins / max(1, total_signals)) * 100.0

        if db and run_id:
            try:
                db.add_trades(run_id, trades_for_db)
                db.finish_run(run_id, total_signals, total_wins, total_losses, avg_r, ev_r)
            except Exception as e:
                logger.warning(f"DB finalize failed: {e}")

        return {
            'label': label,
            'signals': total_signals,
            'wins': total_wins,
            'losses': total_losses,
            'wr': wr,
            'avg_r': avg_r,
            'ev_r': ev_r,
        }

    # Build variants for sweep or run baseline
    variants: List[Tuple[str, TrendSettings]] = []
    if do_sweep:
        spec = sweep_spec or os.getenv('PRETRAIN_SWEEP', '')
        grid = {
            'div_mode': ['optional', 'strict'],
            'breakout_buffer_atr': [base.breakout_buffer_atr, 0.08],
            'bos_armed_hold_minutes': [base.bos_armed_hold_minutes, 180, 300],
            'sl_mode': ['breakout', 'hybrid'],
        }
        if spec:
            try:
                for part in spec.split(';'):
                    if not part.strip():
                        continue
                    k, vals = part.split('=')
                    grid[k.strip()] = [v.strip() for v in vals.split(',') if v.strip()]
            except Exception:
                logger.warning("PRETRAIN_SWEEP spec parse failed; using defaults")
        from itertools import product
        keys = list(grid.keys())
        for combo in product(*[grid[k] for k in keys]):
            s = base
            parts = []
            for k, v in zip(keys, combo):
                parts.append(f"{k}={v}")
                try:
                    if k in ('breakout_buffer_atr',):
                        s = replace(s, **{k: float(v)})
                    elif k in ('bos_armed_hold_minutes',):
                        s = replace(s, **{k: int(v)})
                    elif k in ('sl_mode', 'div_mode'):
                        s = replace(s, **{k: str(v)})
                except Exception:
                    pass
            variants.append((";".join(parts), s))
    else:
        variants.append(("baseline", base))

    results: List[Dict] = []
    best: Optional[Dict] = None
    for (label, settings) in variants:
        res = _run_single_variant(label, settings)
        results.append(res)
        if (best is None) or (res['ev_r'] > best['ev_r']):
            best = res

    # Train ML at the end and save to Redis
    try:
        ok = bool(tml.startup_retrain())
        logger.info(f"ML training completed: {'SUCCESS' if ok else 'SKIPPED/FAILED'}")
    except Exception as e:
        logger.warning(f"ML retrain error: {e}")

    # Summary of results
    for r in sorted(results, key=lambda x: x['ev_r'], reverse=True)[:5]:
        logger.info(f"RESULT {r['label']}: signals={r['signals']} WR={r['wr']:.1f}% avgR={r['avg_r']:.2f} EV_R={r['ev_r']:.2f}")
    if best:
        logger.info(f"BEST: {best['label']} | WR={best['wr']:.1f}% EV_R={best['ev_r']:.2f}")
    else:
        logger.info("No best variant (no signals)")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Trend Pullback Pretraining Runner')
    ap.add_argument('--sweep', action='store_true', help='Enable parameter sweep over variants')
    ap.add_argument('--wf-folds', type=int, default=int(os.getenv('PRETRAIN_WF_FOLDS', '1')), help='Walk-forward folds')
    ap.add_argument('--persist-db', action='store_true', help='Persist runs and trades to DB')
    ap.add_argument('--sweep-spec', type=str, default=os.getenv('PRETRAIN_SWEEP', ''), help='Grid spec key=v1,v2;key2=w1,w2')
    args = ap.parse_args()
    try:
        run_pretraining(
            do_sweep=bool(args.sweep),
            wf_folds=int(args.wf_folds),
            persist_db=bool(args.persist_db),
            sweep_spec=args.sweep_spec or None,
            max_minutes=None,
        )
    except KeyboardInterrupt:
        logger.info("Pretraining cancelled by user")
