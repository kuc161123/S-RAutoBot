import os
import json
import base64
import logging
from typing import Dict, List, Tuple

import numpy as np

try:
    import redis
except Exception:
    redis = None

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None  # type: ignore

logger = logging.getLogger(__name__)


class QScoreAdapterBase:
    """
    Base adapter that learns a mapping from qscore→P(win) per context bucket
    and derives an execution threshold per bucket based on EV > 0.

    Subclasses implement `_load_training_records()` to return a list of
    (qscore: float, outcome: int(1 win / 0 loss), pnl_percent: float, ctx: Dict).
    """

    MIN_RECORDS = 50
    RETRAIN_INTERVAL = 50

    def __init__(self, ns: str):
        self.ns = ns  # redis namespace prefix, e.g., qcal:trend
        self.redis = None
        if redis:
            try:
                url = os.getenv('REDIS_URL')
                if url:
                    self.redis = redis.from_url(url, decode_responses=True)
                    self.redis.ping()
            except Exception as e:
                logger.debug(f"QCal[{self.ns}] Redis unavailable: {e}")
                self.redis = None
        self.thresholds: Dict[str, float] = {}  # bucket_key -> qscore threshold
        self.last_train_count: int = 0
        self._load_state()

    # --------- Persistence ---------
    def _load_state(self):
        r = self.redis
        if not r:
            return
        try:
            thr_raw = r.get(f"{self.ns}:thr")
            if thr_raw:
                self.thresholds = json.loads(thr_raw)
            self.last_train_count = int(r.get(f"{self.ns}:last_train_count") or 0)
        except Exception as e:
            logger.debug(f"QCal[{self.ns}] load_state error: {e}")

    def _save_state(self):
        r = self.redis
        if not r:
            return
        try:
            r.set(f"{self.ns}:thr", json.dumps(self.thresholds))
            r.set(f"{self.ns}:last_train_count", str(self.last_train_count))
        except Exception as e:
            logger.debug(f"QCal[{self.ns}] save_state error: {e}")

    # --------- Public API ---------
    def get_threshold(self, ctx: Dict, floor: float = 60.0, ceiling: float = 95.0, default: float = None) -> float:
        key = self._bucket_key(ctx)
        thr = self.thresholds.get(key)
        if thr is None:
            thr = self.thresholds.get('global')
        if thr is None:
            return float(default if default is not None else floor)
        return float(max(floor, min(ceiling, thr)))

    def retrain_if_ready(self) -> bool:
        import os as _os
        # Global off switch for ML/QCal retrains
        try:
            if str(_os.getenv('DISABLE_ML', '')).strip().lower() in ('1','true','yes','on') or \
               str(_os.getenv('DISABLE_ML_RETRAIN', '')).strip().lower() in ('1','true','yes','on'):
                return False
        except Exception:
            pass
        try:
            records = self._load_training_records()
            n = len(records)
            if n < self.MIN_RECORDS:
                return False
            if (n - self.last_train_count) < self.RETRAIN_INTERVAL and self.thresholds:
                return False
            self._retrain(records)
            self.last_train_count = n
            self._save_state()
            return True
        except Exception as e:
            logger.debug(f"QCal[{self.ns}] retrain_if_ready skipped: {e}")
            return False

    # --------- Core ---------
    def _retrain(self, records: List[Tuple[float, int, float, Dict]]):
        by_bucket: Dict[str, List[Tuple[float, int, float]]] = {}
        for q, y, pnl, ctx in records:
            if q is None:
                continue
            key = self._bucket_key(ctx)
            by_bucket.setdefault(key, []).append((float(q), int(y), float(pnl)))
        new_thr: Dict[str, float] = {}
        for key, arr in by_bucket.items():
            try:
                if len(arr) < max(20, self.MIN_RECORDS // 2):
                    continue
                xs = np.array([a[0] for a in arr])
                ys = np.array([a[1] for a in arr])
                # Calibrate qscore→P(win)
                if IsotonicRegression is None:
                    # Fallback: monotonic binning via percentiles
                    p = np.mean(ys)
                    pred = lambda q: p
                else:
                    iso = IsotonicRegression(out_of_bounds='clip')
                    # Fit on sorted pairs to enforce monotonicity
                    order = np.argsort(xs)
                    iso.fit(xs[order], ys[order])
                    pred = lambda q: float(iso.predict([float(q)])[0])
                # Compute EV threshold p* from empirical avg win/loss magnitudes
                wins = [a[2] for a in arr if a[1] == 1]
                losses = [a[2] for a in arr if a[1] == 0]
                if not wins or not losses:
                    continue
                avg_win = np.mean([abs(w) for w in wins])
                avg_loss = np.mean([abs(l) for l in losses])
                if avg_win <= 0 or avg_loss <= 0:
                    continue
                p_min = avg_loss / (avg_loss + avg_win)
                # Invert calibrated function: find smallest q where P(win) ≥ p_min
                grid = np.linspace(40.0, 95.0, 111)  # 0.5 steps
                thr_q = None
                for qv in grid:
                    if pred(qv) >= p_min:
                        thr_q = qv
                        break
                if thr_q is None:
                    thr_q = 90.0
                new_thr[key] = float(thr_q)
            except Exception as _e:
                logger.debug(f"QCal[{self.ns}] bucket {key} retrain skipped: {_e}")
        if new_thr:
            self.thresholds = new_thr
            logger.info(f"QCal[{self.ns}] thresholds updated: {new_thr}")

    # --------- Helpers ---------
    def _bucket_key(self, ctx: Dict) -> str:
        try:
            sess = str(ctx.get('session', 'global'))
            vol = str(ctx.get('volatility_regime', 'global'))
            return f"{sess}|{vol}"
        except Exception:
            return 'global'

    # To be implemented by subclasses
    def _load_training_records(self) -> List[Tuple[float, int, float, Dict]]:
        raise NotImplementedError
