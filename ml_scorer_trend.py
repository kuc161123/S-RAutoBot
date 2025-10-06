"""
Trend (Donchian Breakout) ML Scorer

Immediate-learning style scorer with simple ensemble and Redis-backed state.
"""
import os, json, base64, pickle, logging
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


class TrendMLScorer:
    MIN_TRADES_FOR_ML = 30
    RETRAIN_INTERVAL = 50
    INITIAL_THRESHOLD = 70

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = float(self.INITIAL_THRESHOLD)
        self.completed_trades = 0
        self.last_train_count = 0
        self.models = {}
        self.scaler = StandardScaler()
        self.is_ml_ready = False
        self.redis_client = None
        if enabled and redis:
            try:
                url = os.getenv('REDIS_URL')
                if url:
                    self.redis_client = redis.from_url(url, decode_responses=True)
                    self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Trend ML Redis unavailable: {e}")
        self._load_state()

    def _load_state(self):
        r = self.redis_client
        if not r:
            return
        try:
            self.completed_trades = int(r.get('tml:completed_trades') or 0)
            self.last_train_count = int(r.get('tml:last_train_count') or 0)
            thr = r.get('tml:threshold')
            if thr:
                self.min_score = float(thr)
            m = r.get('tml:model')
            s = r.get('tml:scaler')
            if m and s:
                self.models = pickle.loads(base64.b64decode(m))
                self.scaler = pickle.loads(base64.b64decode(s))
                self.is_ml_ready = True
                logger.info("Loaded Trend ML models")
        except Exception as e:
            logger.error(f"Trend load state error: {e}")

    def _save_state(self):
        r = self.redis_client
        if not r:
            return
        try:
            r.set('tml:completed_trades', str(self.completed_trades))
            r.set('tml:last_train_count', str(self.last_train_count))
            r.set('tml:threshold', str(self.min_score))
            if self.is_ml_ready:
                r.set('tml:model', base64.b64encode(pickle.dumps(self.models)).decode('ascii'))
                r.set('tml:scaler', base64.b64encode(pickle.dumps(self.scaler)).decode('ascii'))
        except Exception as e:
            logger.error(f"Trend save state error: {e}")

    def _prepare_features(self, f: Dict) -> List[float]:
        # 10-feat simplified vector, encode session to int
        order = [
            'trend_slope_pct', 'ema_stack_score', 'atr_pct', 'range_expansion',
            'breakout_dist_atr', 'close_vs_ema20_pct', 'bb_width_pct', 'session',
            'symbol_cluster', 'volatility_regime'
        ]
        vec = []
        for k in order:
            v = f.get(k, 0)
            if k == 'session':
                v = {'asian':0,'european':1,'us':2,'off_hours':3}.get(str(v),3)
            if k == 'volatility_regime':
                v = {'low':0,'normal':1,'high':2,'extreme':3}.get(str(v),1)
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
        return vec

    def score_signal(self, signal: Dict, features: Dict) -> Tuple[float, str]:
        if self.is_ml_ready and self.models:
            try:
                x = np.array([self._prepare_features(features)])
                xs = self.scaler.transform(x) if hasattr(self.scaler, 'mean_') else x
                preds = []
                if self.models.get('rf'):
                    preds.append(self.models['rf'].predict_proba(xs)[:,1])
                if self.models.get('gb'):
                    preds.append(self.models['gb'].predict_proba(xs)[:,1])
                if preds:
                    p = float(np.mean(preds))
                    return p * 100.0, 'Trend ML Ensemble'
            except Exception as e:
                logger.warning(f"Trend ML scoring failed: {e}")
        # Fallback rule-based
        score = 50.0
        try:
            slope = float(features.get('trend_slope_pct', 0.0))
            ema_stack = float(features.get('ema_stack_score', 0.0))
            breakout = float(features.get('breakout_dist_atr', 0.0))
            rng_exp = float(features.get('range_expansion', 0.0))
            atr_pct = float(features.get('atr_pct', 0.0))
            if abs(slope) >= 5:
                score += 10
            if ema_stack >= 50:
                score += 10
            if breakout >= 0.2:
                score += 10
            if rng_exp >= 1.2:
                score += 10
            if atr_pct < 0.5:
                score -= 5
        except Exception:
            pass
        return max(0.0, min(100.0, score)), 'Trend heuristic'

    def record_outcome(self, signal: Dict, outcome: str, pnl_percent: float = 0.0):
        self.completed_trades += 1
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
                r.rpush('tml:trades', json.dumps(rec))
                r.ltrim('tml:trades', -5000, -1)
            self._save_state()
        except Exception as e:
            logger.error(f"Trend record outcome error: {e}")

        # Retrain trigger
        try:
            total = len(self._load_training_data())
            if total >= self.MIN_TRADES_FOR_ML and (total - self.last_train_count) >= self.RETRAIN_INTERVAL:
                self._retrain()
        except Exception as e:
            logger.debug(f"Trend retrain check failed: {e}")

    def _load_training_data(self) -> List[Dict]:
        data = []
        if self.redis_client:
            try:
                arr = self.redis_client.lrange('tml:trades', 0, -1)
                for t in arr:
                    try:
                        data.append(json.loads(t))
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Trend load data error: {e}")
        return data

    def _retrain(self):
        data = self._load_training_data()
        if len(data) < self.MIN_TRADES_FOR_ML:
            logger.info(f"Trend ML not enough data: {len(data)}/{self.MIN_TRADES_FOR_ML}")
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
        logger.info("Trend ML retrained")
        return True

    def get_retrain_info(self) -> Dict:
        data_len = 0
        if self.redis_client:
            try:
                data_len = len(self._load_training_data())
            except Exception:
                pass
        next_at = self.last_train_count + self.RETRAIN_INTERVAL
        left = max(0, next_at - data_len) if self.is_ml_ready else max(0, self.MIN_TRADES_FOR_ML - data_len)
        return {
            'is_ml_ready': bool(self.is_ml_ready),
            'completed_trades': int(self.completed_trades),
            'total_records': int(data_len),
            'trades_until_next_retrain': int(left),
            'next_retrain_at': int(next_at),
            'can_train': data_len >= self.MIN_TRADES_FOR_ML
        }

    def get_stats(self) -> Dict:
        """Return status summary compatible with UI expectations.

        Keys: status, completed_trades, current_threshold, recent_win_rate, models_active
        """
        recent_wr = 0.0
        total = wins = 0
        try:
            if self.redis_client:
                # Last 200 outcomes as proxy for recent WR
                arr = self.redis_client.lrange('tml:trades', -200, -1) or []
                for t in arr:
                    try:
                        rec = json.loads(t)
                        total += 1
                        wins += int(rec.get('outcome', 0))
                    except Exception:
                        pass
                if total > 0:
                    recent_wr = (wins / total) * 100.0
        except Exception:
            pass
        models_active = []
        try:
            if self.models.get('rf'):
                models_active.append('rf')
            if self.models.get('gb'):
                models_active.append('gb')
        except Exception:
            pass
        return {
            'status': '✅ Ready' if self.is_ml_ready else '⏳ Training',
            'completed_trades': int(self.completed_trades),
            'current_threshold': float(self.min_score),
            'recent_win_rate': float(recent_wr),
            'models_active': models_active,
        }

    def get_patterns(self) -> Dict:
        # Light patterns: feature importance from RF and time/session buckets
        out = {'feature_importance': {}, 'time_patterns': {}, 'market_conditions': {}, 'winning_patterns': [], 'losing_patterns': []}
        if not self.is_ml_ready or 'rf' not in self.models:
            return out
        try:
            names = ['trend_slope_pct','ema_stack_score','atr_pct','range_expansion','breakout_dist_atr','close_vs_ema20_pct','bb_width_pct','session','symbol_cluster','volatility_regime']
            imps = getattr(self.models['rf'], 'feature_importances_', [])
            if len(imps) > 0:
                pairs = list(zip(names[:len(imps)], imps))
                pairs.sort(key=lambda x: x[1], reverse=True)
                for feat, imp in pairs[:10]:
                    out['feature_importance'][feat] = round(float(imp) * 100.0, 1)
        except Exception as e:
            logger.debug(f"Trend patterns FI error: {e}")
        # Basic time/session buckets
        try:
            r = self.redis_client
            arr = r.lrange('tml:trades', -500, -1) if r else []
            hours = {}
            sess = {}
            for t in arr:
                try:
                    rec = json.loads(t)
                    ts = rec.get('timestamp')
                    dt = datetime.fromisoformat(ts)
                    hr = dt.hour
                    sess_key = 'us' if 16 <= hr < 24 else 'european' if 8 <= hr < 16 else 'asian' if 0 <= hr < 8 else 'off_hours'
                    outc = int(rec.get('outcome', 0))
                    def _add(dct, key):
                        a = dct.get(key, {'w':0,'n':0}); a['n']+=1; a['w']+=outc; dct[key]=a
                    _add(hours, hr); _add(sess, sess_key)
                except Exception:
                    pass
            def _fmt(dct):
                m = {}
                for k, a in dct.items():
                    n = a['n']; w = a['w']; wr = (w/n*100.0) if n else 0.0
                    m[str(k)] = f"WR {wr:.0f}% (N={n})"
                return m
            out['time_patterns'] = {'best_hours': dict(sorted(_fmt(hours).items(), key=lambda x: int(x[0]))), 'session_performance': _fmt(sess)}
        except Exception:
            pass
        return out


_trend_scorer = None


def get_trend_scorer(enabled: bool = True) -> TrendMLScorer:
    global _trend_scorer
    if _trend_scorer is None:
        _trend_scorer = TrendMLScorer(enabled=enabled)
    return _trend_scorer
