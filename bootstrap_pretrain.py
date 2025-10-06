#!/usr/bin/env python3
"""
Bootstrap pretraining from candle data (OHLCV only).

Generates training samples for:
- Trend Breakout (Donchian-based)
- Mean Reversion (using live MR detector with df-index time)

Labels trades via conservative TP/SL-first rule (SL-first on same-bar hit).
Fits scalers/models and persists to Redis if available.
"""
import logging
from typing import List, Dict

import yaml
import pandas as pd
import os
import json
from datetime import datetime

from candle_storage_postgres import CandleStorage
from utils_data_quality import prepare_df_for_features

from strategy_trend_breakout import detect_signal as detect_trend_signal, TrendSettings
from strategy_mean_reversion import detect_signal as detect_mr_signal, reset_symbol_state as reset_mr_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BOOTSTRAP] %(levelname)s - %(message)s')


def _label_outcome_conservative(df: pd.DataFrame, start_idx: int, side: str, tp: float, sl: float, max_lookahead: int = 500) -> str:
    end = min(start_idx + max_lookahead, len(df))
    for i in range(start_idx, end):
        hi = float(df['high'].iloc[i])
        lo = float(df['low'].iloc[i])
        if side == 'long':
            hit_tp = hi >= tp
            hit_sl = lo <= sl
            if hit_tp and hit_sl:
                return 'loss'  # SL-first conservative tie-break
            if hit_tp:
                return 'win'
            if hit_sl:
                return 'loss'
        else:
            hit_tp = lo <= tp
            hit_sl = hi >= sl
            if hit_tp and hit_sl:
                return 'loss'
            if hit_tp:
                return 'win'
            if hit_sl:
                return 'loss'
    return 'loss'  # No outcome within window → treat as loss conservatively


def _trend_samples(df: pd.DataFrame, symbol: str, settings: TrendSettings) -> List[Dict]:
    samples: List[Dict] = []
    if df is None or len(df) < 220:
        return samples

    for i in range(200, len(df) - 1):
        df_slice = df.iloc[:i]
        sig = detect_trend_signal(df_slice, settings, symbol)
        if not sig:
            continue
        # Trend features
        close = df_slice['close']; high = df_slice['high']; low = df_slice['low']
        price = float(close.iloc[-1])
        ys = close.tail(20).values if len(close) >= 20 else close.values
        try:
            slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
        except Exception:
            slope = 0.0
        trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
        ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
        rng_today = float(high.iloc[-1] - low.iloc[-1])
        med_range = float((high - low).rolling(20).median().iloc[-1]) if len(df_slice) >= 20 else rng_today
        range_expansion = float(rng_today / max(1e-9, med_range))
        prev = close.shift(); trarr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
        atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
        atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
        close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
        feats = {
            'trend_slope_pct': trend_slope_pct,
            'ema_stack_score': ema_stack_score,
            'atr_pct': atr_pct,
            'range_expansion': range_expansion,
            'breakout_dist_atr': float(sig.meta.get('breakout_dist_atr', 0.0) if getattr(sig, 'meta', None) else 0.0),
            'close_vs_ema20_pct': close_vs_ema20_pct,
            'bb_width_pct': 0.0,
            'session': 'us',
            'symbol_cluster': 3,
            'volatility_regime': 'normal'
        }

        outcome = _label_outcome_conservative(df, i, sig.side, sig.tp, sig.sl)
        samples.append({'features': feats, 'outcome': 1 if outcome == 'win' else 0, 'was_executed': True})

    return samples


def _mean_reversion_samples(df: pd.DataFrame, symbol: str, mr_settings: PullbackSettings) -> List[Dict]:
    samples: List[Dict] = []
    if df is None or len(df) < 220:
        return samples

    # Ensure state is clean for this symbol
    reset_mr_state(symbol)

    for i in range(200, len(df) - 1):
        df_slice = df.iloc[:i]
        sig = detect_mr_signal(df_slice, mr_settings, symbol)
        if not sig:
            continue

        feats = sig.meta.get('mr_features', {})
        if not feats:
            continue

        outcome = _label_outcome_conservative(df, i, sig.side, sig.tp, sig.sl)
        samples.append({'features': feats, 'outcome': 1 if outcome == 'win' else 0})

    return samples


def main():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    symbols = cfg['trade']['symbols']
    timeframe = int(str(cfg['trade'].get('timeframe', '15')).replace('m', ''))

    storage = CandleStorage()
    pb_settings = TrendSettings()

    # MR uses Settings from strategy_pullback for SL/TP sizing
    try:
        from strategy_pullback import Settings as MRSettings
        mr_settings = MRSettings(
            atr_len=cfg['trade'].get('atr_len', 14),
            sl_buf_atr=cfg['trade'].get('sl_buf_atr', 0.5),
            rr=cfg['trade'].get('rr', 2.5),
            use_ema=cfg['trade'].get('use_ema', False),
            ema_len=cfg['trade'].get('ema_len', 200),
            use_vol=cfg['trade'].get('use_vol', False),
            vol_len=cfg['trade'].get('vol_len', 20),
            vol_mult=cfg['trade'].get('vol_mult', 1.2),
            both_hit_rule=cfg['trade'].get('both_hit_rule', 'SL_FIRST')
        )
    except Exception:
        mr_settings = None

    pullback_training: List[Dict] = []
    mr_training: List[Dict] = []
    total_pb_wins = total_pb_losses = 0
    total_mr_wins = total_mr_losses = 0

    for sym in symbols:
        logger.info(f"Loading candles for {sym}...")
        df = storage.load_candles(sym, limit=100000)
        if df is None or df.empty:
            logger.warning(f"No candles for {sym}")
            continue

        df = prepare_df_for_features(df, sym)
        logger.info(f"{sym}: {len(df)} candles after QA prep")

        # Trend samples
        pb = _trend_samples(df, sym, pb_settings)
        pullback_training.extend(pb)
        pb_wins = sum(1 for s in pb if s.get('outcome') == 1)
        pb_losses = len(pb) - pb_wins
        total_pb_wins += pb_wins
        total_pb_losses += pb_losses
        if len(pb) > 0:
            wr = (pb_wins / len(pb)) * 100
            logger.info(f"{sym}: trend samples {len(pb)} | wins {pb_wins} / losses {pb_losses} (WR {wr:.1f}%)")
        else:
            logger.info(f"{sym}: trend samples 0")

        # MR samples
        mr = _mean_reversion_samples(df, sym, mr_settings)
        mr_training.extend(mr)
        mr_wins = sum(1 for s in mr if s.get('outcome') == 1)
        mr_losses = len(mr) - mr_wins
        total_mr_wins += mr_wins
        total_mr_losses += mr_losses
        if len(mr) > 0:
            mr_wr = (mr_wins / len(mr)) * 100
            logger.info(f"{sym}: MR samples {len(mr)} | wins {mr_wins} / losses {mr_losses} (WR {mr_wr:.1f}%)")
        else:
            logger.info(f"{sym}: MR samples 0")

    logger.info(f"Total trend samples: {len(pullback_training)} (wins {total_pb_wins} / losses {total_pb_losses}, WR {(total_pb_wins / max(1, (total_pb_wins+total_pb_losses)))*100:.1f}%)")
    logger.info(f"Total MR samples: {len(mr_training)} (wins {total_mr_wins} / losses {total_mr_losses}, WR {(total_mr_wins / max(1, (total_mr_wins+total_mr_losses)))*100:.1f}%)")

    # Push samples to Redis if available so scorers can train from persisted data
    redis_client = None
    try:
        import redis  # type: ignore
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            logger.info("Connected to Redis for pretraining persistence")
    except Exception as e:
        logger.warning(f"Redis not available for pretraining: {e}")

    # If Redis exists, write datasets to the expected keys
    if redis_client:
        try:
            # Persist Trend executed trades to 'tml:trades'
            redis_client.delete('tml:trades')
            for s in pullback_training:
                record = {
                    'features': s['features'],
                    'outcome': 'win' if s['outcome'] == 1 else 'loss',
                    'pnl_percent': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
                redis_client.rpush('tml:trades', json.dumps(record))
            redis_client.set('tml:completed_trades', str(len(pullback_training)))
            redis_client.set('tml:last_train_count', '0')
            logger.info(f"Persisted {len(pullback_training)} trend samples to Redis")
        except Exception as e:
            logger.error(f"Failed to persist pullback samples to Redis: {e}")

        try:
            # Persist MR trades to 'ml:trades:mean_reversion'
            redis_client.delete('ml:trades:mean_reversion')
            for s in mr_training:
                record = {
                    'features': s['features'],
                    'outcome': int(s['outcome']),
                    'pnl_percent': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
                redis_client.rpush('ml:trades:mean_reversion', json.dumps(record))
            redis_client.set('ml:completed_trades:mean_reversion', str(len(mr_training)))
            redis_client.set('ml:last_train_count:mean_reversion', '0')
            logger.info(f"Persisted {len(mr_training)} MR samples to Redis")
        except Exception as e:
            logger.error(f"Failed to persist MR samples to Redis: {e}")

    # Train Trend ML
    try:
        from ml_scorer_trend import get_trend_scorer
        tml = get_trend_scorer()
        if not redis_client:
            for s in pullback_training:
                tml.record_outcome({'features': s['features'], 'was_executed': True}, 'win' if s['outcome']==1 else 'loss', 0.0)
        ok = False
        try:
            ok = bool(tml._retrain())
        except Exception:
            ok = False
        logger.info(f"Trend ML startup retrain: {'SUCCESS' if ok else 'SKIPPED/FAILED'}")
    except Exception as e:
        logger.error(f"Trend ML training error: {e}")

    # Train MR ML (original scorer)
    try:
        from ml_scorer_mean_reversion import get_mean_reversion_scorer
        mr_scorer = get_mean_reversion_scorer()
        if not redis_client:
            if not hasattr(mr_scorer, 'memory_storage'):
                mr_scorer.memory_storage = {'trades': []}
            mr_scorer.memory_storage['trades'] = mr_training
            mr_scorer.completed_trades = len(mr_training)
        # Use scorer's training entrypoint (handles scaler save)
        mr_scorer._retrain_models()
        logger.info(f"MR ML trained: {mr_scorer.is_ml_ready}")
    except Exception as e:
        logger.error(f"MR ML training error: {e}")

    logger.info("Bootstrap pretraining completed.")


if __name__ == '__main__':
    main()
