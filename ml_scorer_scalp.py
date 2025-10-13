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
from sklearn.neural_network import MLPClassifier
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
    # Allow phantom-only bootstrap when there are no executed scalp trades yet
    # Unlimited phantom usage policy for training (no cap)
    PHANTOM_BOOTSTRAP_MIN = 0
    PHANTOM_BOOTSTRAP_MAX = 10_000_000

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
            if self.models.get('nn'):
                try:
                    preds.append(self.models['nn'].predict_proba(xs)[:,1])
                except Exception:
                    pass
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
            data = self._load_training_data()
            # Unlimited phantoms: trainable set is the full dataset
            trainable = len(data)

            if trainable >= self.MIN_TRADES_FOR_ML and (trainable - self.last_train_count) >= self.RETRAIN_INTERVAL:
                logger.info(f"Scalp ML retrain trigger: {trainable - self.last_train_count} new trades (trainable={trainable})")
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
        # Use the entire dataset (executed + phantoms) without caps
        mix = data

        if len(mix) < self.MIN_TRADES_FOR_ML:
            logger.info(f"Scalp ML trainable set below minimum after policy: {len(mix)}/{self.MIN_TRADES_FOR_ML} (exec={exec_count})")
            return False
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
            'rf': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42),
            # New neural network head; uses the scaled features
            'nn': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=5e-4,
                                batch_size=64, learning_rate='adaptive', max_iter=200, random_state=42)
        }
        try:
            self.models['rf'].fit(XS, y, sample_weight=w)
            self.models['gb'].fit(XS, y, sample_weight=w)
            try:
                self.models['nn'].fit(XS, y, sample_weight=w)
            except TypeError:
                # Older sklearn without sample_weight support on MLP
                self.models['nn'].fit(XS, y)
        except Exception:
            self.models['rf'].fit(XS, y)
            self.models['gb'].fit(XS, y)
            try:
                self.models['nn'].fit(XS, y)
            except Exception:
                pass
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

    def get_retrain_info(self) -> Dict:
        """Report readiness and trades until next retrain under current policy."""
        try:
            data = self._load_training_data()
            exec_count = sum(1 for d in data if d.get('was_executed'))
            phantom_count = len(data) - exec_count
            if exec_count == 0 and len(data) >= self.PHANTOM_BOOTSTRAP_MIN:
                trainable = min(len(data), self.PHANTOM_BOOTSTRAP_MAX)
            else:
                trainable = exec_count + min(int(exec_count * 1.5), phantom_count)

            info = {
                'is_ml_ready': bool(self.is_ml_ready),
                'completed_trades': int(self.completed_trades),
                'total_records': int(len(data)),
                'trainable_size': int(trainable),
                'last_train_count': int(self.last_train_count),
                'trades_until_next_retrain': 0,
                'next_retrain_at': 0,
                'can_train': False
            }

            if not self.is_ml_ready:
                info['can_train'] = trainable >= self.MIN_TRADES_FOR_ML
                info['trades_until_next_retrain'] = max(0, self.MIN_TRADES_FOR_ML - trainable)
                info['next_retrain_at'] = self.MIN_TRADES_FOR_ML
            else:
                since_last = trainable - self.last_train_count
                info['can_train'] = True
                info['trades_until_next_retrain'] = max(0, self.RETRAIN_INTERVAL - max(0, since_last))
                info['next_retrain_at'] = self.last_train_count + self.RETRAIN_INTERVAL
            return info
        except Exception:
            return {
                'is_ml_ready': bool(self.is_ml_ready),
                'completed_trades': int(self.completed_trades),
                'total_records': 0,
                'trainable_size': 0,
                'last_train_count': int(self.last_train_count),
                'trades_until_next_retrain': 0,
                'next_retrain_at': 0,
                'can_train': False
            }

    def get_patterns(self) -> Dict:
        """Basic scalp ML pattern mining: feature importances + simple time/condition patterns.
        Returns a dict similar to other scorers.
        """
        out = {
            'feature_importance': {},
            'time_patterns': {},
            'market_conditions': {},
            'winning_patterns': [],
            'losing_patterns': []
        }
        try:
            # Feature importance from RF if available
            rf = self.models.get('rf') if self.models else None
            if rf and hasattr(rf, 'feature_importances_'):
                names = [
                    'atr_pct','bb_width_pct','impulse_ratio','ema_slope_fast','ema_slope_slow',
                    'volume_ratio','upper_wick_ratio','lower_wick_ratio','vwap_dist_atr',
                    'session','symbol_cluster','volatility_regime'
                ]
                imps = getattr(rf, 'feature_importances_', [])
                if len(imps) > 0:
                    if len(imps) != len(names):
                        names = names[:len(imps)]
                    pairs = list(zip(names, imps))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    for feat, imp in pairs[:10]:
                        out['feature_importance'][feat] = round(float(imp) * 100.0, 1)

            # Pull recent training data for simple patterns
            data = self._load_training_data()[-500:]
            if not data:
                return out

            from collections import defaultdict
            by_hour = defaultdict(lambda: {'w':0,'n':0})
            by_sess = defaultdict(lambda: {'w':0,'n':0})
            vol_bk = defaultdict(lambda: {'w':0,'n':0})
            vwap_bk = defaultdict(lambda: {'w':0,'n':0})
            bbw_bk = defaultdict(lambda: {'w':0,'n':0})

            def _out(rec):
                o = rec.get('outcome', 0)
                try:
                    return int(o)
                except Exception:
                    return 1 if str(o).lower()=='win' else 0

            for rec in data:
                try:
                    ts = rec.get('timestamp')
                    from datetime import datetime as _dt
                    hr = _dt.fromisoformat(str(ts).replace('Z','')).hour if ts else 0
                except Exception:
                    hr = 0
                by_hour[hr]['n'] += 1
                by_hour[hr]['w'] += _out(rec)
                f = rec.get('features', {}) or {}
                sess = f.get('session','off_hours')
                by_sess[sess]['n'] += 1
                by_sess[sess]['w'] += _out(rec)
                # Buckets
                vr = str(f.get('volatility_regime','normal'))
                vol_bk[vr]['n'] += 1; vol_bk[vr]['w'] += _out(rec)
                try:
                    vda = float(f.get('vwap_dist_atr', 0))
                except Exception:
                    vda = 0.0
                vwap_key = 'near(<=0.4)' if vda <= 0.4 else 'mid(0.4-0.8)' if vda <= 0.8 else 'far(>0.8)'
                vwap_bk[vwap_key]['n'] += 1; vwap_bk[vwap_key]['w'] += _out(rec)
                try:
                    bbw = float(f.get('bb_width_pct', 0))
                except Exception:
                    bbw = 0.0
                bbw_key = 'narrow(<=0.5)' if bbw <= 0.5 else 'normal(0.5-0.8)' if bbw <= 0.8 else 'wide(>0.8)'
                bbw_bk[bbw_key]['n'] += 1; bbw_bk[bbw_key]['w'] += _out(rec)

            def _wr_map(d):
                m = {}
                for k,v in d.items():
                    n = v['n']; w = v['w']
                    wr = (w/n*100.0) if n else 0.0
                    m[str(k)] = f"WR {wr:.0f}% (N={n})"
                return m

            out['time_patterns'] = {
                'best_hours': {str(h): f"WR {v['w']/v['n']*100:.0f}% (N={v['n']})" for h,v in sorted(by_hour.items(), key=lambda x: (x[1]['w']/x[1]['n']) if x[1]['n'] else 0, reverse=True)[:5] if v['n']>=3},
                'worst_hours': {str(h): f"WR {v['w']/v['n']*100:.0f}% (N={v['n']})" for h,v in sorted(by_hour.items(), key=lambda x: (x[1]['w']/x[1]['n']) if x[1]['n'] else 0)[:5] if v['n']>=3},
                'session_performance': _wr_map(by_sess)
            }
            out['market_conditions'] = {
                'volatility_regime': _wr_map(vol_bk),
                'vwap_dist_atr': _wr_map(vwap_bk),
                'bb_width_pct': _wr_map(bbw_bk)
            }

            # Simple narrative
            try:
                top_sess = max(out['time_patterns']['session_performance'].items(), key=lambda x: float(x[1].split('%')[0].split()[-1])) if out['time_patterns'].get('session_performance') else None
                if top_sess:
                    out['winning_patterns'].append(f"Best session: {top_sess[0]} {top_sess[1]}")
            except Exception:
                pass
            return out
        except Exception:
            return out


_scalp_scorer = None


def get_scalp_scorer(enabled: bool = True) -> ScalpMLScorer:
    global _scalp_scorer
    if _scalp_scorer is None:
        _scalp_scorer = ScalpMLScorer(enabled=enabled)
    return _scalp_scorer
