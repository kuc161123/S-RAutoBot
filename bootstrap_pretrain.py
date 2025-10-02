#!/usr/bin/env python3
"""
Bootstrap pretraining from candle data (OHLCV only).

Generates training samples for:
- Pullback (using original structure strategy for robust, stateless detection)
- Mean Reversion (using live MR detector with df-index time)

Labels trades via conservative TP/SL-first rule (SL-first on same-bar hit).
Fits scalers/models and persists to Redis if available.
"""
import logging
from typing import List, Dict

import yaml
import pandas as pd

from candle_storage_postgres import CandleStorage
from utils_data_quality import prepare_df_for_features

from strategy import detect_signal as detect_pullback_signal, Settings as PullbackSettings
from strategy_pullback_ml_learning import calculate_ml_features, BreakoutState
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


def _pullback_samples(df: pd.DataFrame, symbol: str, settings: PullbackSettings) -> List[Dict]:
    samples: List[Dict] = []
    if df is None or len(df) < 220:
        return samples

    for i in range(200, len(df) - 1):
        df_slice = df.iloc[:i]
        sig = detect_pullback_signal(df_slice, settings, symbol)
        if not sig:
            continue

        # Build a BreakoutState for S/R touches using strategy-provided meta
        state = BreakoutState()
        try:
            if sig.side == 'long':
                state.breakout_level = float(sig.meta.get('res', 0.0))
            else:
                state.breakout_level = float(sig.meta.get('sup', 0.0))
        except Exception:
            state.breakout_level = 0.0

        # Approximate retracement depth (percent) with a simple proxy
        # Use last 20 candles swing
        try:
            window = df_slice.iloc[-20:]
            hi = float(window['high'].max())
            lo = float(window['low'].min())
            rng = max(1e-9, hi - lo)
            if sig.side == 'long':
                retracement = max(0.0, min(100.0, (hi - sig.entry) / rng * 100))
            else:
                retracement = max(0.0, min(100.0, (sig.entry - lo) / rng * 100))
        except Exception:
            retracement = 50.0

        feats = calculate_ml_features(df_slice, state, sig.side, retracement)
        # Minimal cluster id if not present
        feats.setdefault('symbol_cluster', 3)

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
    pb_settings = PullbackSettings(
        left=cfg['trade'].get('left', 2),
        right=cfg['trade'].get('right', 2),
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

    mr_settings = pb_settings  # MR uses same Settings container for SL/TP sizing

    pullback_training: List[Dict] = []
    mr_training: List[Dict] = []

    for sym in symbols:
        logger.info(f"Loading candles for {sym}...")
        df = storage.load_candles(sym, limit=100000)
        if df is None or df.empty:
            logger.warning(f"No candles for {sym}")
            continue

        df = prepare_df_for_features(df, sym)
        logger.info(f"{sym}: {len(df)} candles after QA prep")

        # Pullback samples
        pb = _pullback_samples(df, sym, pb_settings)
        pullback_training.extend(pb)
        logger.info(f"{sym}: pullback samples {len(pb)}")

        # MR samples
        mr = _mean_reversion_samples(df, sym, mr_settings)
        mr_training.extend(mr)
        logger.info(f"{sym}: MR samples {len(mr)}")

    logger.info(f"Total pullback samples: {len(pullback_training)}")
    logger.info(f"Total MR samples: {len(mr_training)}")

    # Train Pullback ML
    try:
        from ml_signal_scorer_immediate import get_immediate_scorer
        pb_scorer = get_immediate_scorer()
        pb_scorer.memory_storage = {'trades': pullback_training, 'phantoms': []}
        pb_scorer.completed_trades = len(pullback_training)
        pb_scorer.force_retrain = True
        trained = pb_scorer.startup_retrain()
        logger.info(f"Pullback ML startup retrain: {'SUCCESS' if trained else 'SKIPPED/FAILED'}")
    except Exception as e:
        logger.error(f"Pullback ML training error: {e}")

    # Train MR ML (original scorer)
    try:
        from ml_scorer_mean_reversion import get_mean_reversion_scorer
        mr_scorer = get_mean_reversion_scorer()
        if not hasattr(mr_scorer, 'memory_storage'):
            mr_scorer.memory_storage = {'trades': []}
        mr_scorer.memory_storage['trades'] = mr_training
        mr_scorer.completed_trades = len(mr_training)
        mr_scorer._retrain_models()
        logger.info(f"MR ML trained: {mr_scorer.is_ml_ready}")
    except Exception as e:
        logger.error(f"MR ML training error: {e}")

    logger.info("Bootstrap pretraining completed.")


if __name__ == '__main__':
    main()

