"""
ML Scorer for Scalping Strategy (Phase 0)

Independent scorer with its own Redis keys and simplified features.
Starts phantom-only; can be promoted to execution later.
"""
import logging, os, json, pickle, base64
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import redis
except Exception:
    redis = None


class ScalpMLScorer:
    MIN_TRADES_FOR_ML = 50
    RETRAIN_INTERVAL = 50
    INITIAL_THRESHOLD = 75

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = self.INITIAL_THRESHOLD
        self.models = {}
        self.scaler = StandardScaler()
        self.is_ml_ready = False
        self.completed_trades = 0
        self.last_train_count = 0
        self.redis_client = None
        if enabled and redis:
            try:
                url = os.getenv('REDIS_URL')
                if url:
                    self.redis_client = redis.from_url(url, decode_responses=True)
                    self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Scalp ML Redis unavailable: {e}")
        self._load_state()

    def _load_state(self):
        if not self.redis_client:
            return
        try:
            t = self.redis_client.get('ml:completed_trades:scalp')
            self.completed_trades = int(t) if t else 0
            last = self.redis_client.get('ml:last_train_count:scalp')
            self.last_train_count = int(last) if last else 0
            thr = self.redis_client.get('ml:threshold:scalp')
            if thr:
                self.min_score = float(thr)
            m = self.redis_client.get('ml:model:scalp')
            s = self.redis_client.get('ml:scaler:scalp')
            if m and s:
                self.models = pickle.loads(base64.b64decode(m))
                self.scaler = pickle.loads(base64.b64decode(s))
                self.is_ml_ready = True
                logger.info("Loaded Scalp ML models")
        except Exception as e:
            logger.error(f"Scalp load state error: {e}")

    def _save_state(self):
        if not self.redis_client:
            return
        try:
            self.redis_client.set('ml:completed_trades:scalp', str(self.completed_trades))
            self.redis_client.set('ml:last_train_count:scalp', str(self.last_train_count))
            self.redis_client.set('ml:threshold:scalp', str(self.min_score))
            if self.is_ml_ready:
                self.redis_client.set('ml:model:scalp', base64.b64encode(pickle.dumps(self.models)).decode('ascii'))
                self.redis_client.set('ml:scaler:scalp', base64.b64encode(pickle.dumps(self.scaler)).decode('ascii'))
        except Exception as e:
            logger.error(f"Scalp save state error: {e}")

    def _prepare_features(self, f: Dict) -> List[float]:
        order = [
            'atr_pct', 'bb_width_pct', 'impulse_ratio', 'ema_slope_fast', 'ema_slope_slow',
            'volume_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'vwap_dist_atr',
            'session', 'symbol_cluster', 'volatility_regime'
        ]
        vec = []
        for k in order:
            v = f.get(k, 0)
            if k == 'session':
                v = {'asian':0,'european':1,'us':2,'off_hours':3}.get(v,3)
            if k == 'volatility_regime':
                v = {'low':0,'normal':1,'high':2,'extreme':3}.get(v,1)
            vec.append(float(v) if v is not None else 0.0)
        return vec

    def score_signal(self, signal: Dict, features: Dict) -> Tuple[float, str]:
        if not self.is_ml_ready or not self.models:
            # Simple heuristic pre-ML
            score = 50.0
            try:
                if features.get('impulse_ratio',0) > 1.0:
                    score += 10
                if features.get('vwap_dist_atr',1) < 0.6:
                    score += 10
                if features.get('bb_width_pct',0) > 0.7:
                    score += 5
                if features.get('volume_ratio',1) > 1.3:
                    score += 10
            except Exception:
                pass
            return max(0,min(100,score)), 'Heuristic'
        try:
            x = np.array([self._prepare_features(features)])
            xs = self.scaler.transform(x)
            preds = []
            if self.models.get('rf'):
                preds.append(self.models['rf'].predict_proba(xs)[:,1])
            if self.models.get('gb'):
                preds.append(self.models['gb'].predict_proba(xs)[:,1])
            if preds:
                p = float(np.mean(preds))
                return p*100.0, 'Scalp ML Ensemble'
        except Exception as e:
            logger.error(f"Scalp scoring error: {e}")
        return 60.0, 'Fallback'

    def record_outcome(self, signal: Dict, outcome: str, pnl_percent: float = 0.0):
        self.completed_trades += 1
        if self.redis_client:
            try:
                record = {
                    'features': signal.get('features', {}),
                    'outcome': 1 if outcome=='win' else 0,
                    'pnl_percent': pnl_percent,
                    'timestamp': datetime.utcnow().isoformat(),
                    'was_executed': 1 if signal.get('was_executed') else 0
                }
                self.redis_client.rpush('ml:trades:scalp', json.dumps(record))
                self.redis_client.ltrim('ml:trades:scalp', -5000, -1)
                self._save_state()
            except Exception as e:
                logger.error(f"Scalp record outcome error: {e}")
        # Auto-retrain when enough new trades have accumulated
        try:
            data_len = len(self._load_training_data())
            if data_len >= self.MIN_TRADES_FOR_ML and (data_len - self.last_train_count) >= self.RETRAIN_INTERVAL:
                logger.info(f"Scalp ML retrain trigger: {data_len - self.last_train_count} new trades")
                self._retrain()
        except Exception as e:
            logger.debug(f"Scalp auto-retrain check failed: {e}")

    def _load_training_data(self) -> List[Dict]:
        data = []
        if self.redis_client:
            try:
                arr = self.redis_client.lrange('ml:trades:scalp', 0, -1)
                for t in arr:
                    try:
                        data.append(json.loads(t))
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Scalp load data error: {e}")
        return data

    def _retrain(self):
        data = self._load_training_data()
        if len(data) < self.MIN_TRADES_FOR_ML:
            logger.info(f"Scalp ML not enough data: {len(data)}/{self.MIN_TRADES_FOR_ML}")
            return False
        X, y, w = [], [], []
        exec_count = sum(1 for d in data if d.get('was_executed'))
        ph = [d for d in data if not d.get('was_executed')]
        ex = [d for d in data if d.get('was_executed')]
        max_ph = int(max(1, exec_count) * 1.5)
        if len(ph) > max_ph:
            ph = ph[-max_ph:]
        mix = ex + ph
        for d in mix:
            f = d.get('features', {})
            X.append(self._prepare_features(f))
            y.append(int(d.get('outcome', 0)))
            wt = 1.0 if d.get('was_executed') else 0.8
            if f.get('routing') == 'none':
                wt *= 0.5
            try:
                if int(f.get('symbol_cluster', 0)) == 3:
                    wt *= 0.7
            except Exception:
                pass
            w.append(wt)
        X = np.array(X); y = np.array(y); w = np.array(w)
        self.scaler.fit(X)
        XS = self.scaler.transform(X)
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=5, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
        }
        try:
            self.models['rf'].fit(XS, y, sample_weight=w)
            self.models['gb'].fit(XS, y, sample_weight=w)
        except Exception:
            self.models['rf'].fit(XS, y)
            self.models['gb'].fit(XS, y)
        self.is_ml_ready = True
        self.last_train_count = len(mix)
        self._save_state()
        logger.info("Scalp ML retrained")
        return True

    def startup_retrain(self) -> bool:
        try:
            return self._retrain()
        except Exception as e:
            logger.error(f"Scalp startup retrain error: {e}")
            return False

    def get_stats(self) -> Dict:
        """Return compact stats for dashboard."""
        try:
            total = 0
            if self.redis_client:
                total = len(self._load_training_data())
            return {
                'completed_trades': int(self.completed_trades),
                'total_records': total,
                'is_ml_ready': bool(self.is_ml_ready),
                'current_threshold': float(self.min_score)
            }
        except Exception:
            return {
                'completed_trades': int(self.completed_trades),
                'total_records': 0,
                'is_ml_ready': bool(self.is_ml_ready),
                'current_threshold': float(self.min_score)
            }


_scalp_scorer = None


def get_scalp_scorer(enabled: bool = True) -> ScalpMLScorer:
    global _scalp_scorer
    if _scalp_scorer is None:
        _scalp_scorer = ScalpMLScorer(enabled=enabled)
    return _scalp_scorer
