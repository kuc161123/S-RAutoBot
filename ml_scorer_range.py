"""
Range FBO ML Scorer (scaffold)

Mirrors Trend ML scorer API with a lighter feature set.
By default, returns a heuristic score until enough data accumulates and models are trained.
"""
import os, json, base64, pickle, logging
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    import redis
except Exception:
    redis = None


class RangeMLScorer:
    MIN_TRADES_FOR_ML = 300
    RETRAIN_INTERVAL = 100
    INITIAL_THRESHOLD = 75

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = float(self.INITIAL_THRESHOLD)
        self.completed_trades = 0
        self.last_train_count = 0
        self.models = {}
        self.redis_client = None
        self.KEY_NS = 'ml:range'
        if enabled and redis:
            try:
                url = os.getenv('REDIS_URL')
                if url:
                    self.redis_client = redis.from_url(url, decode_responses=True)
                    self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Range ML Redis unavailable: {e}")
        self._load_state()

    def _load_state(self):
        r = self.redis_client
        if not r:
            return
        try:
            self.completed_trades = int(r.get(f'{self.KEY_NS}:completed_trades') or 0)
            self.last_train_count = int(r.get(f'{self.KEY_NS}:last_train_count') or 0)
            thr = r.get(f'{self.KEY_NS}:threshold')
            if thr:
                self.min_score = float(thr)
        except Exception as e:
            logger.error(f"Range load state error: {e}")

    def _save_state(self):
        r = self.redis_client
        if not r:
            return
        try:
            r.set(f'{self.KEY_NS}:completed_trades', str(self.completed_trades))
            r.set(f'{self.KEY_NS}:last_train_count', str(self.last_train_count))
            r.set(f'{self.KEY_NS}:threshold', str(self.min_score))
        except Exception as e:
            logger.error(f"Range save state error: {e}")

    def _prepare_features(self, f: Dict) -> List[float]:
        order = [
            'range_width_pct','wick_ratio','retest_ok',
            'ts15','ts60','rc15','rc60','qscore'
        ]
        vec = []
        for k in order:
            v = f.get(k, 0)
            if k == 'retest_ok':
                v = 1.0 if bool(v) else 0.0
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
        return vec

    def score_signal(self, signal: Dict, features: Dict) -> Tuple[float, str]:
        # Heuristic until models are trained
        try:
            w = float(features.get('range_width_pct', 0.0))
            wick = float(features.get('wick_ratio', 0.0))
            ts15 = float(features.get('ts15', 0.0)); ts60 = float(features.get('ts60', 0.0))
            s = 50.0
            if 0.01 <= w <= 0.08:
                s += 10
            if wick >= 0.25:
                s += 10
            if max(ts15, ts60) < 60:
                s += 10
            return max(0.0, min(100.0, s)), 'Range heuristic'
        except Exception:
            return 50.0, 'Range heuristic'

    def record_outcome(self, signal: Dict, outcome: str, pnl_percent: float = 0.0):
        # Skip timeouts; count only executed trades toward completed
        try:
            if str(signal.get('exit_reason','')).lower() == 'timeout':
                return
        except Exception:
            pass
        try:
            if bool(signal.get('was_executed')):
                self.completed_trades += 1
        except Exception:
            pass
        r = self.redis_client
        try:
            rec = {
                'features': signal.get('features', {}),
                'outcome': 1 if outcome == 'win' else 0,
                'pnl_percent': float(pnl_percent),
                'timestamp': datetime.utcnow().isoformat(),
                'was_executed': 1 if signal.get('was_executed') else 0
            }
            if r:
                r.rpush(f'{self.KEY_NS}:trades', json.dumps(rec))
            self._save_state()
        except Exception as e:
            logger.error(f"Range record outcome error: {e}")

    def get_retrain_info(self) -> Dict:
        total = 0
        exec_count = 0
        if self.redis_client:
            try:
                arr = self.redis_client.lrange(f'{self.KEY_NS}:trades', 0, -1) or []
                total = len(arr)
                exec_count = sum(1 for t in arr if json.loads(t).get('was_executed'))
            except Exception:
                pass
        left = max(0, self.MIN_TRADES_FOR_ML - total)
        return {'is_ml_ready': total >= self.MIN_TRADES_FOR_ML, 'total_records': total, 'executed_count': exec_count, 'trades_until_next_retrain': left}

    def get_stats(self) -> Dict:
        total = 0; wins = 0
        if self.redis_client:
            try:
                arr = self.redis_client.lrange(f'{self.KEY_NS}:trades', -200, -1) or []
                for t in arr:
                    try:
                        rec = json.loads(t)
                        total += 1
                        wins += int(rec.get('outcome', 0))
                    except Exception:
                        continue
            except Exception:
                pass
        recent_wr = (wins/total*100.0) if total else 0.0
        return {'status': '⏳ Collecting', 'current_threshold': float(self.min_score), 'recent_win_rate': float(recent_wr), 'recent_trades': int(total)}


_range_scorer = None

def get_range_scorer(enabled: bool = True) -> RangeMLScorer:
    global _range_scorer
    if _range_scorer is None:
        _range_scorer = RangeMLScorer(enabled=enabled)
    return _range_scorer

