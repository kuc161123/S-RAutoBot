"""
Trend Pullback ML Scorer

Overhauled for the new pullback logic:
- Break S/R → HL/LH → 2-candle confirmation
- Features reflect breakout distance, retrace depth, and confirmations
Immediate-learning style scorer with ensemble and Redis-backed state.
"""
import os, json, base64, pickle, logging
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from sklearn.neural_network import MLPClassifier  # optional NN head
except Exception:
    MLPClassifier = None
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

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
        self.calibrator: IsotonicRegression | None = None
        self.ev_buckets: dict = {}
        self.is_ml_ready = False
        self.redis_client = None
        self.phantom_weight = 0.8  # default weight for phantom samples
        self.KEY_NS = 'ml:trend'
        # Optional NN head toggle (off by default)
        try:
            self.nn_enabled = bool(int(os.getenv('TREND_NN_ENABLED', '0')))
        except Exception:
            self.nn_enabled = False
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
            self.completed_trades = int(r.get(f'{self.KEY_NS}:completed_trades') or r.get('tml:completed_trades') or 0)
            self.last_train_count = int(r.get(f'{self.KEY_NS}:last_train_count') or r.get('tml:last_train_count') or 0)
            thr = r.get(f'{self.KEY_NS}:threshold') or r.get('tml:threshold')
            if thr:
                self.min_score = float(thr)
            m = r.get(f'{self.KEY_NS}:model') or r.get('tml:model')
            s = r.get(f'{self.KEY_NS}:scaler') or r.get('tml:scaler')
            c = r.get(f'{self.KEY_NS}:calibrator') or r.get('tml:calibrator')
            b = r.get(f'{self.KEY_NS}:ev_buckets') or r.get('tml:ev_buckets')
            if m and s:
                self.models = pickle.loads(base64.b64decode(m))
                self.scaler = pickle.loads(base64.b64decode(s))
                self.is_ml_ready = True
                logger.info("Loaded Trend ML models")
            if c:
                try:
                    self.calibrator = pickle.loads(base64.b64decode(c))
                except Exception:
                    self.calibrator = None
            if b:
                try:
                    self.ev_buckets = pickle.loads(base64.b64decode(b))
                except Exception:
                    self.ev_buckets = {}
            # Load phantom weight if provided
            try:
                w = r.get(f'{self.KEY_NS}:phantom_weight') or r.get('phantom:weight')
                if w:
                    self.phantom_weight = float(w)
            except Exception:
                pass
            # Load NN flag if present
            try:
                nn_flag = r.get(f'{self.KEY_NS}:nn_enabled')
                if nn_flag is not None:
                    self.nn_enabled = bool(int(nn_flag))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Trend load state error: {e}")

    def _save_state(self):
        r = self.redis_client
        if not r:
            return
        try:
            # New namespaced keys
            r.set(f'{self.KEY_NS}:completed_trades', str(self.completed_trades))
            r.set(f'{self.KEY_NS}:last_train_count', str(self.last_train_count))
            r.set(f'{self.KEY_NS}:threshold', str(self.min_score))
            # Write legacy keys for compatibility during transition
            r.set('tml:completed_trades', str(self.completed_trades))
            r.set('tml:last_train_count', str(self.last_train_count))
            r.set('tml:threshold', str(self.min_score))
            if self.is_ml_ready:
                enc_model = base64.b64encode(pickle.dumps(self.models)).decode('ascii')
                enc_scaler = base64.b64encode(pickle.dumps(self.scaler)).decode('ascii')
                r.set(f'{self.KEY_NS}:model', enc_model)
                r.set(f'{self.KEY_NS}:scaler', enc_scaler)
                try:
                    enc_cal = base64.b64encode(pickle.dumps(self.calibrator)).decode('ascii')
                    r.set(f'{self.KEY_NS}:calibrator', enc_cal)
                except Exception:
                    pass
                try:
                    enc_ev = base64.b64encode(pickle.dumps(self.ev_buckets)).decode('ascii')
                    r.set(f'{self.KEY_NS}:ev_buckets', enc_ev)
                except Exception:
                    pass
                # Legacy mirrors
                r.set('tml:model', enc_model)
                r.set('tml:scaler', enc_scaler)
                try:
                    r.set('tml:calibrator', enc_cal)
                except Exception:
                    pass
                try:
                    r.set('tml:ev_buckets', enc_ev)
                except Exception:
                    pass
            # Persist NN flag for ops visibility
            try:
                r.set(f'{self.KEY_NS}:nn_enabled', '1' if self.nn_enabled else '0')
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Trend save state error: {e}")

    def _prepare_features(self, f: Dict) -> List[float]:
        # New feature vector for Trend Pullback
        order = [
            'atr_pct',            # ATR as % of price
            'break_dist_atr',     # distance beyond broken level
            'retrace_depth_atr',  # pullback depth before HL/LH
            'confirm_candles',    # 2 if satisfied else <2
            'ema_stack_score',    # simple trend filter, optional
            'range_expansion',    # day range vs median
            # Divergence features (if available; defaults applied otherwise)
            'div_ok',
            'div_score',
            'div_rsi_delta',
            'div_tsi_delta',
            # Protective pivot presence (boolean)
            'protective_pivot_present',
            # Lifecycle flags (executed/phantom)
            'tp1_hit',
            'be_moved',
            'runner_hit',
            'time_to_tp1_sec',
            'time_to_exit_sec',
            'session', 'symbol_cluster', 'volatility_regime',
            # Composite HTF metrics (flattened)
            'ts15','ts60','rc15','rc60'
        ]
        vec = []
        for k in order:
            v = f.get(k, 0)
            if k == 'session':
                v = {'asian':0,'european':1,'us':2,'off_hours':3}.get(str(v),3)
            if k == 'volatility_regime':
                v = {'low':0,'normal':1,'high':2,'extreme':3}.get(str(v),1)
            if k in ('div_ok','protective_pivot_present','tp1_hit','be_moved','runner_hit'):
                try:
                    v = 1.0 if bool(v) else 0.0
                except Exception:
                    v = 0.0
            try:
                vec.append(float(v))
            except Exception:
                # Defaults for divergence and numerics when missing
                if k in ('div_ok','div_score','div_rsi_delta','div_tsi_delta'):
                    vec.append(0.0)
                else:
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
                if self.models.get('nn'):
                    try:
                        preds.append(self.models['nn'].predict_proba(xs)[:,1])
                    except Exception:
                        pass
                if preds:
                    p = float(np.mean(preds))
                    try:
                        if self.calibrator is not None:
                            p = float(self.calibrator.predict([p])[0])
                            p = max(0.0, min(1.0, p))
                    except Exception:
                        pass
                    return p * 100.0, 'Trend ML Ensemble (cal)'
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
        # Skip timeout/cancel outcomes to keep EV/training clean
        try:
            if str(signal.get('exit_reason','')).lower() == 'timeout':
                return
        except Exception:
            pass
        # Count only executed trades toward completed_trades; phantoms are stored but not counted
        try:
            if bool(signal.get('was_executed')):
                self.completed_trades += 1
        except Exception:
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
                # Append without trimming to keep full history for retraining (no limits)
                r.rpush(f'{self.KEY_NS}:trades', json.dumps(rec))
                # Legacy mirror
                try:
                    r.rpush('tml:trades', json.dumps(rec))
                except Exception:
                    pass
                # Update EV buckets
                try:
                    f = rec.get('features', {}) or {}
                    sess = str(f.get('session','unknown'))
                    vol = str(f.get('volatility_regime','unknown'))
                    key = f"{sess}|{vol}"
                    b = self.ev_buckets.get(key, {'w_sum':0.0,'w_n':0,'l_sum':0.0,'l_n':0})
                    if rec['outcome'] == 1:
                        b['w_sum'] += float(rec.get('pnl_percent', 0.0))
                        b['w_n'] += 1
                    else:
                        b['l_sum'] += float(rec.get('pnl_percent', 0.0))
                        b['l_n'] += 1
                    self.ev_buckets[key] = b
                except Exception:
                    pass
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
                arr = self.redis_client.lrange(f'{self.KEY_NS}:trades', 0, -1)
                if not arr:
                    arr = self.redis_client.lrange('tml:trades', 0, -1)
                # Prefer empty list if new key exists; fallback only if it truly doesn't exist
                try:
                    if self.redis_client.exists(f'{self.KEY_NS}:trades'):
                        arr = self.redis_client.lrange(f'{self.KEY_NS}:trades', 0, -1)
                except Exception:
                    pass
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
        # Use entire dataset (executed + phantom) without caps
        mix = data
        for d in mix:
            f = d.get('features', {})
            X.append(self._prepare_features(f))
            y.append(int(d.get('outcome', 0)))
            try:
                wt = 1.0 if d.get('was_executed') else float(self.phantom_weight)
            except Exception:
                wt = 0.8 if not d.get('was_executed') else 1.0
            w.append(wt)
        X = np.array(X); y = np.array(y); w = np.array(w)
        self.scaler.fit(X)
        XS = self.scaler.transform(X)
        self.models = {
            'rf': RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42)
        }
        # Optional simple NN head
        if self.nn_enabled and MLPClassifier is not None:
            try:
                self.models['nn'] = MLPClassifier(hidden_layer_sizes=(32,16), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=400, random_state=42)
            except Exception:
                self.models.pop('nn', None)
        try:
            self.models['rf'].fit(XS, y, sample_weight=w)
            self.models['gb'].fit(XS, y, sample_weight=w)
            if 'nn' in self.models:
                try:
                    self.models['nn'].fit(XS, y)
                except Exception:
                    self.models.pop('nn', None)
        except Exception:
            self.models['rf'].fit(XS, y)
            self.models['gb'].fit(XS, y)
            if 'nn' in self.models:
                try:
                    self.models['nn'].fit(XS, y)
                except Exception:
                    self.models.pop('nn', None)
        self.is_ml_ready = True
        self.last_train_count = len(mix)
        # Calibrate (isotonic) on mean-of-heads raw scores
        try:
            raw_preds = []
            for i in range(len(XS)):
                p_heads = []
                try:
                    p_heads.append(self.models['rf'].predict_proba(XS[i:i+1])[:,1])
                except Exception:
                    pass
                try:
                    p_heads.append(self.models['gb'].predict_proba(XS[i:i+1])[:,1])
                except Exception:
                    pass
                if p_heads:
                    raw_preds.append(float(np.mean(p_heads)))
            if len(raw_preds) >= 50:
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(np.array(raw_preds), y)
        except Exception as _e:
            logger.debug(f"Trend calibrator skipped: {_e}")
        # Stamp last retrain time
        try:
            if self.redis_client:
                self.redis_client.set('tml:last_train_ts', datetime.utcnow().isoformat())
                try:
                    self.redis_client.set(f'{self.KEY_NS}:last_train_ts', datetime.utcnow().isoformat())
                except Exception:
                    pass
        except Exception:
            pass
        self._save_state()
        logger.info("Trend ML retrained")
        return True

    def startup_retrain(self) -> bool:
        """Compatibility retrain entrypoint used by the bot.

        Attempts to retrain if enough data is present; returns True on success.
        """
        try:
            return bool(self._retrain())
        except Exception as e:
            logger.debug(f"Trend startup retrain skipped: {e}")
            return False

    def get_retrain_info(self) -> Dict:
        data_len = 0
        exec_count = 0
        ph_count = 0
        if self.redis_client:
            try:
                data = self._load_training_data()
                data_len = len(data)
                exec_count = sum(1 for d in data if d.get('was_executed'))
                ph_count = max(0, data_len - exec_count)
            except Exception:
                pass
        next_at = self.last_train_count + self.RETRAIN_INTERVAL
        left = max(0, next_at - data_len) if self.is_ml_ready else max(0, self.MIN_TRADES_FOR_ML - data_len)
        return {
            'is_ml_ready': bool(self.is_ml_ready),
            'completed_trades': int(self.completed_trades),
            'total_records': int(data_len),
            'total_combined': int(data_len),
            'executed_count': int(exec_count),
            'phantom_count': int(ph_count),
            'last_retrain_ts': (self.redis_client.get(f'{self.KEY_NS}:last_train_ts') if self.redis_client else None),
            'trades_until_next_retrain': int(left),
            'next_retrain_at': int(next_at),
            'can_train': data_len >= self.MIN_TRADES_FOR_ML
        }

    def get_ev_threshold(self, features: Dict, floor: float = 70.0, ceiling: float = 90.0, min_samples: int = 30) -> float:
        """EV-based threshold for Trend Pullback by session/volatility (percent)."""
        try:
            sess = str(features.get('session','unknown'))
            vol = str(features.get('volatility_regime','unknown'))
            key = f"{sess}|{vol}"
            b = self.ev_buckets.get(key) or {}
            w_n = int(b.get('w_n', 0)); l_n = int(b.get('l_n', 0))
            if (w_n + l_n) < min_samples or w_n == 0 or l_n == 0:
                return float(floor)
            avg_win = (b.get('w_sum', 0.0) / max(1, w_n))
            avg_loss = (b.get('l_sum', 0.0) / max(1, l_n))
            R_net = abs(avg_win) / max(1e-6, abs(avg_loss))
            p_min = 1.0 / (1.0 + max(1e-6, R_net))
            thr = max(floor, min(ceiling, p_min * 100.0))
            return float(thr)
        except Exception:
            return float(floor)

    def get_stats(self) -> Dict:
        """Return status summary compatible with UI expectations.

        Keys: status, completed_trades, current_threshold, recent_win_rate, models_active
        """
        recent_wr = 0.0
        total = wins = 0
        recent_trades = 0
        exec_count = 0
        ph_count = 0
        try:
            if self.redis_client:
                # Last 200 outcomes as proxy for recent WR (prefer new namespace)
                key = f"{self.KEY_NS}:trades"
                arr = self.redis_client.lrange(key, -200, -1) or []
                if (not arr) and (not bool(self.redis_client.exists(key))):
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
                recent_trades = total
                # Overall executed vs phantom counts for clarity
                all_arr = self.redis_client.lrange(key, 0, -1) or []
                if (not all_arr) and (not bool(self.redis_client.exists(key))):
                    all_arr = self.redis_client.lrange('tml:trades', 0, -1) or []
                for t in all_arr:
                    try:
                        rec = json.loads(t)
                        if int(rec.get('was_executed', 0)):
                            exec_count += 1
                        else:
                            ph_count += 1
                    except Exception:
                        pass
        except Exception:
            pass
        models_active = []
        try:
            if self.models.get('rf'):
                models_active.append('rf')
            if self.models.get('gb'):
                models_active.append('gb')
            if self.models.get('nn'):
                models_active.append('nn')
        except Exception:
            pass
        return {
            'status': '✅ Ready' if self.is_ml_ready else '⏳ Training',
            'completed_trades': int(self.completed_trades),
            'current_threshold': float(self.min_score),
            'recent_win_rate': float(recent_wr),
            'recent_trades': int(recent_trades),
            'models_active': models_active,
            'executed_count': int(exec_count),
            'phantom_count': int(ph_count),
            'total_records': int(exec_count + ph_count),
        }

    def get_patterns(self) -> Dict:
        # Light patterns: feature importance from RF and time/session buckets
        out = {'feature_importance': {}, 'time_patterns': {}, 'market_conditions': {}, 'winning_patterns': [], 'losing_patterns': []}
        if not self.is_ml_ready or 'rf' not in self.models:
            return out
        try:
            names = ['atr_pct','break_dist_atr','retrace_depth_atr','confirm_candles','ema_stack_score','range_expansion','session','symbol_cluster','volatility_regime']
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
            key = f"{self.KEY_NS}:trades"
            arr = r.lrange(key, -500, -1) if r else []
            if r and (not arr) and (not bool(r.exists(key))):
                arr = r.lrange('tml:trades', -500, -1)
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

    # Backward/forward-compatible API used by Telegram dashboard
    def get_learned_patterns(self) -> Dict:
        """Alias for get_patterns() to match UI expectation.

        Telegram's /mlpatterns expects `get_learned_patterns`; this method
        returns the same structure produced by get_patterns(). When models are
        not ready, it returns an empty structure—UI will display a collecting
        data message instead of a hard error.
        """
        try:
            return self.get_patterns()
        except Exception:
            return {'feature_importance': {}, 'time_patterns': {}, 'market_conditions': {}, 'winning_patterns': [], 'losing_patterns': []}


_trend_scorer = None


def get_trend_scorer(enabled: bool = True) -> TrendMLScorer:
    global _trend_scorer
    if _trend_scorer is None:
        _trend_scorer = TrendMLScorer(enabled=enabled)
    return _trend_scorer
