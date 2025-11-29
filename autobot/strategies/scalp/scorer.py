from __future__ import annotations
"""
ML Scorer for Scalping Strategy (Phase 0)

Independent scorer with its own Redis keys and simplified features.
Starts phantom-only; can be promoted to execution later.
"""
import logging, os, json, pickle, base64
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

try:
    import redis
except Exception:
    redis = None


class ScalpMLScorer:
    MIN_TRADES_FOR_ML = 50
    RETRAIN_INTERVAL = 50
    INITIAL_THRESHOLD = 75
    FEAT_VERSION = 2
    # Allow phantom-only bootstrap when there are no executed scalp trades yet
    # Unlimited phantom usage policy for training (no cap)
    PHANTOM_BOOTSTRAP_MIN = 0
    PHANTOM_BOOTSTRAP_MAX = 10_000_000

    def __init__(self, enabled: bool = True):
        try:
            # Global disable via env takes precedence
            env_off = str(os.getenv('DISABLE_ML', '')).strip().lower() in ('1','true','yes','on')
        except Exception:
            env_off = False
        self.enabled = (False if env_off else enabled)
        self.min_score = self.INITIAL_THRESHOLD
        self.models = {}
        self.scaler = StandardScaler()
        self.calibrator: Optional[IsotonicRegression] = None
        self.ev_buckets: dict = {}
        self.is_ml_ready = False
        self.completed_trades = 0
        self.last_train_count = 0
        self.redis_client = None
        if self.enabled and redis:
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
            t = self.redis_client.get('ml:scalp:completed_trades') or self.redis_client.get('ml:completed_trades:scalp')
            self.completed_trades = int(t) if t else 0
            last = self.redis_client.get('ml:scalp:last_train_count') or self.redis_client.get('ml:last_train_count:scalp')
            self.last_train_count = int(last) if last else 0
            thr = self.redis_client.get('ml:scalp:threshold') or self.redis_client.get('ml:threshold:scalp')
            if thr:
                self.min_score = float(thr)
            m = self.redis_client.get('ml:scalp:model') or self.redis_client.get('ml:model:scalp')
            s = self.redis_client.get('ml:scalp:scaler') or self.redis_client.get('ml:scaler:scalp')
            c = self.redis_client.get('ml:scalp:calibrator') or self.redis_client.get('ml:calibrator:scalp')
            b = self.redis_client.get('ml:scalp:ev_buckets') or self.redis_client.get('ml:ev_buckets:scalp')
            fv = self.redis_client.get('ml:scalp:featver') or self.redis_client.get('ml:featver:scalp')
            if m and s:
                self.models = pickle.loads(base64.b64decode(m))
                self.scaler = pickle.loads(base64.b64decode(s))
                self.is_ml_ready = True
                logger.info("Loaded Scalp ML models")
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
            try:
                if fv:
                    self.featver = int(fv)
            except Exception:
                self.featver = 1
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
        except Exception as e:
            logger.error(f"Scalp load state error: {e}")

    def _save_state(self):
        if not self.redis_client:
            return
        try:
            self.redis_client.set('ml:scalp:completed_trades', str(self.completed_trades))
            self.redis_client.set('ml:scalp:last_train_count', str(self.last_train_count))
            self.redis_client.set('ml:scalp:threshold', str(self.min_score))
            # Legacy mirrors
            self.redis_client.set('ml:completed_trades:scalp', str(self.completed_trades))
            self.redis_client.set('ml:last_train_count:scalp', str(self.last_train_count))
            self.redis_client.set('ml:threshold:scalp', str(self.min_score))
            if self.is_ml_ready:
                enc_model = base64.b64encode(pickle.dumps(self.models)).decode('ascii')
                enc_scaler = base64.b64encode(pickle.dumps(self.scaler)).decode('ascii')
                self.redis_client.set('ml:scalp:model', enc_model)
                self.redis_client.set('ml:scalp:scaler', enc_scaler)
                try:
                    enc_cal = base64.b64encode(pickle.dumps(self.calibrator)).decode('ascii')
                    self.redis_client.set('ml:scalp:calibrator', enc_cal)
                except Exception:
                    pass
                try:
                    enc_ev = base64.b64encode(pickle.dumps(self.ev_buckets)).decode('ascii')
                    self.redis_client.set('ml:scalp:ev_buckets', enc_ev)
                except Exception:
                    pass
                try:
                    self.redis_client.set('ml:scalp:featver', str(self.featver))
                except Exception:
                    pass
                # Legacy mirrors
                self.redis_client.set('ml:model:scalp', enc_model)
                self.redis_client.set('ml:scaler:scalp', enc_scaler)
                try:
                    self.redis_client.set('ml:calibrator:scalp', enc_cal)
                except Exception:
                    pass
                try:
                    self.redis_client.set('ml:ev_buckets:scalp', enc_ev)
                except Exception:
                    pass
                try:
                    self.redis_client.set('ml:featver:scalp', str(self.featver))
                except Exception:
                    pass
                try:
                    self.redis_client.set('ml:calibrator:scalp', base64.b64encode(pickle.dumps(self.calibrator)).decode('ascii'))
                except Exception:
                    pass
                try:
                    self.redis_client.set('ml:ev_buckets:scalp', base64.b64encode(pickle.dumps(self.ev_buckets)).decode('ascii'))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Scalp save state error: {e}")

    def _prepare_features(self, f: Dict, version: int = 2) -> List[float]:
        # v2: richer features and encodings; v1 fallback kept for compatibility
        if version <= 1:
            order = [
                'atr_pct', 'bb_width_pct', 'impulse_ratio', 'ema_slope_fast', 'ema_slope_slow',
                'volume_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'vwap_dist_atr',
                'session', 'symbol_cluster', 'volatility_regime'
            ]
        else:
            order = [
                'atr_pct','bb_width_pct','bb_width_pctile','bb_pos','impulse_ratio',
                'ema_slope_fast','ema_slope_slow','vwap_dist_atr','volume_ratio',
                'rsi_14','macd_hist','ma20_dist_pct','ma50_dist_pct','fib_ret',
                'ret_1','ret_3','ret_5','atr_slope','vol_of_vol','obv_slope',
                'rr_setup','mtf_agree_15','session','hour_utc','symbol_cluster','volatility_regime','ema_dir_15m','struct_state'
            ]
        vec = []
        for k in order:
            v = f.get(k, 0)
            if k == 'session':
                v = {'asian':0,'european':1,'us':2,'off_hours':3}.get(v,3)
            elif k == 'volatility_regime':
                v = {'low':0,'normal':1,'high':2,'extreme':3}.get(v,1)
            elif k == 'ema_dir_15m':
                v = {'down':0,'none':1,'up':2}.get(str(v),1)
            elif k == 'struct_state':
                v = {'LL':0,'LH':1,'HL':2,'HH':3,'none':1}.get(str(v),1)
            elif isinstance(v, bool):
                v = 1.0 if v else 0.0
            vec.append(float(v) if v is not None else 0.0)
        return vec

    def score_signal(self, signal: Dict, features: Dict) -> Tuple[float, str]:
        # When disabled, do not touch models/Redis; return lightweight heuristic
        if not self.enabled:
            return 50.0, 'ML disabled'
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
                # Meta-ensemble via logistic regression if available
                try:
                    meta = self.models.get('meta')
                    if meta is not None and len(preds) >= 2:
                        m_in = np.array([[float(p_) for p_ in preds]]).reshape(1, -1)
                        p = float(meta.predict_proba(m_in)[:,1][0])
                    else:
                        p = float(np.mean(preds))
                except Exception:
                    p = float(np.mean(preds))
                # Apply probability calibration if available
                try:
                    if self.calibrator is not None:
                        p = float(self.calibrator.predict([p])[0])
                        p = max(0.0, min(1.0, p))
                except Exception:
                    pass
                return p*100.0, 'Scalp ML Stacked (cal)'
        except Exception as e:
            logger.error(f"Scalp scoring error: {e}")
        return 60.0, 'Fallback'

    def record_outcome(self, signal: Dict, outcome: str, pnl_percent: float = 0.0):
        # Short-circuit when ML disabled
        if not self.enabled:
            return
        try:
            if bool(signal.get('was_executed')):
                self.completed_trades += 1
        except Exception:
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
                # Append without trimming to keep full history for retraining (no limits)
                self.redis_client.rpush('ml:scalp:trades', json.dumps(record))
                try:
                    self.redis_client.rpush('ml:trades:scalp', json.dumps(record))
                except Exception:
                    pass
                # Update EV buckets
                try:
                    f = record.get('features', {}) or {}
                    sess = str(f.get('session','unknown'))
                    vol = str(f.get('volatility_regime','unknown'))
                    key = f"{sess}|{vol}"
                    b = self.ev_buckets.get(key, {'w_sum':0.0,'w_n':0,'l_sum':0.0,'l_n':0})
                    if record['outcome'] == 1:
                        b['w_sum'] += float(record.get('pnl_percent', 0.0))
                        b['w_n'] += 1
                    else:
                        b['l_sum'] += float(record.get('pnl_percent', 0.0))
                        b['l_n'] += 1
                    self.ev_buckets[key] = b
                except Exception:
                    pass
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
                # Respect global disable at retrain time as well
                if self.enabled:
                    self._retrain()
        except Exception as e:
            logger.debug(f"Scalp auto-retrain check failed: {e}")

    def _load_training_data(self) -> List[Dict]:
        data = []
        if self.redis_client:
            try:
                arr = self.redis_client.lrange('ml:scalp:trades', 0, -1)
                if not arr:
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
            logger.info(f"Scalp ML trainable set below minimum after policy: {len(mix)}/{self.MIN_TRADES_FOR_ML}")
            return False
        for d in mix:
            f = d.get('features', {})
            X.append(self._prepare_features(f, self.FEAT_VERSION))
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
        # Simple stacking meta-learner (logistic regression) on holdout split
        try:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(XS, y, test_size=0.2, random_state=42)
            preds_tr = []
            for key in ('rf','gb','nn'):
                m = self.models.get(key)
                if m is not None:
                    preds_tr.append(m.predict_proba(X_tr)[:,1])
            if len(preds_tr) >= 2:
                M = np.vstack(preds_tr).T
                meta = LogisticRegression(max_iter=200)
                meta.fit(M, y_tr)
                self.models['meta'] = meta
        except Exception:
            pass
        # Calibrate probabilities (isotonic) on mean-of-heads raw scores
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
                try:
                    p_heads.append(self.models['nn'].predict_proba(XS[i:i+1])[:,1])
                except Exception:
                    pass
                if p_heads:
                    raw_preds.append(float(np.mean(p_heads)))
            if len(raw_preds) >= 50:
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(np.array(raw_preds), y)
        except Exception as _e:
            logger.debug(f"Scalp calibrator skipped: {_e}")
        self.is_ml_ready = True
        self.last_train_count = len(mix)
        self.featver = self.FEAT_VERSION
        # Stamp last retrain time
        try:
                if self.redis_client:
                    self.redis_client.set('ml:scalp:last_train_ts', datetime.utcnow().isoformat())
                    try:
                        self.redis_client.set('ml:last_train_ts:scalp', datetime.utcnow().isoformat())
                    except Exception:
                        pass
        except Exception:
            pass
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

    def get_ev_threshold(self, features: Dict, floor: float = 70.0, ceiling: float = 90.0, min_samples: int = 30) -> float:
        """Return EV-based threshold (percent) for given feature bucket (session|volatility).
        p_min ≈ 1/(1+R_net), where R_net≈|avg_win|/|avg_loss| from recent outcomes.
        """
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

    def get_retrain_info(self) -> Dict:
        """Report readiness and trades until next retrain under current policy."""
        try:
            data = self._load_training_data()
            # Trainable set equals total records (executed + phantom)
            trainable = len(data)

            info = {
                'is_ml_ready': bool(self.is_ml_ready),
                'completed_trades': int(self.completed_trades),
                'total_records': int(len(data)),
                'trainable_size': int(trainable),
                'last_train_count': int(self.last_train_count),
                'last_retrain_ts': None,
                'trades_until_next_retrain': 0,
                'next_retrain_at': 0,
                'can_train': False
            }
            # Read last retrain timestamp if available
            try:
                if self.redis_client:
                    info['last_retrain_ts'] = self.redis_client.get('ml:last_train_ts:scalp')
            except Exception:
                pass

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

    def recommend_strategies(self, days: int = 30, min_n: int = 50, top_n: int = 12) -> Dict:
        """Recommend high‑WR combos from ML training data (last N days).

        Uses RSI/MACD/VWAP/Fib/MTF/Volume/BBW to propose combos with WR and N.
        Returns dict with 'ranked' and 'yaml' fields.
        """
        try:
            data = self._load_training_data()
            if not data:
                return {'error': 'no_data'}
            from datetime import datetime, timedelta
            cutoff = datetime.utcnow() - timedelta(days=days)
            items = []
            for d in data:
                try:
                    ts = d.get('timestamp'); ts = datetime.fromisoformat(str(ts).replace('Z','')) if ts else None
                    if ts and ts < cutoff:
                        continue
                    f = d.get('features', {}) or {}
                    rsi = f.get('rsi_14'); mh = f.get('macd_hist'); vwap = f.get('vwap_dist_atr'); fibz = f.get('fib_zone'); mtf = f.get('mtf_agree_15'); vol = f.get('volume_ratio'); bbw = f.get('bb_width_pct')
                    if not (isinstance(rsi,(int,float)) and isinstance(mh,(int,float)) and isinstance(vwap,(int,float)) and isinstance(fibz,str) and isinstance(mtf,(bool,int)) and isinstance(vol,(int,float)) and isinstance(bbw,(int,float))):
                        continue
                    win = int(d.get('outcome', 0))
                    items.append((float(rsi), float(mh), float(vwap), fibz, bool(mtf), float(vol), float(bbw), win))
                except Exception:
                    continue
            if not items:
                return {'error': 'no_items'}
            # Bins
            rsi_bins = [("<30", lambda x: x < 30, 0, 30), ("30-40", lambda x: 30 <= x < 40, 30, 40), ("40-60", lambda x: 40 <= x < 60, 40, 60), ("60-70", lambda x: 60 <= x < 70, 60, 70), ("70+", lambda x: x >= 70, 70, 101)]
            macd_bins = [("bull", lambda h: h > 0, "bull"), ("bear", lambda h: h <= 0, "bear")]
            vwap_bins = [("<0.6", lambda x: x < 0.6, -999.0, 0.6), ("0.6-1.2", lambda x: 0.6 <= x < 1.2, 0.6, 1.2), ("1.2+", lambda x: x >= 1.2, 1.2, 999.0)]
            vol_bins = [("≥1.2x", lambda x: x >= 1.2, 1.2), ("1.0-1.2x", lambda x: 1.0 <= x < 1.2, 1.0)]
            bbw_bins = [("<1.2%", lambda x: x < 0.012, 0.012), ("1.2-2.0%", lambda x: 0.012 <= x < 0.02, 0.02)]
            fib_bins = ["0-23","23-38","38-50","50-61","61-78","78-100"]
            def lab(val, bins):
                for b in bins:
                    if b[1](val):
                        return b
                return None
            combos = {}
            for rsi, mh, vwap, fibz, mtf, vol, bbw, win in items:
                br = lab(rsi, rsi_bins); bm = lab(mh, macd_bins); bv = lab(vwap, vwap_bins); bf = fibz if fibz in fib_bins else None; bvol = lab(vol, vol_bins); bbwk = lab(bbw, bbw_bins)
                if not all([br, bm, bv, bf, bvol, bbwk]):
                    continue
                key = (br[0], bm[0], bv[0], bf, bool(mtf), bvol[0], bbwk[0])
                agg = combos.setdefault(key, {'n':0,'w':0})
                agg['n'] += 1; agg['w'] += int(win)
            ranked = sorted([(k,v) for k,v in combos.items() if v['n'] >= min_n], key=lambda kv: (kv[1]['w']/kv[1]['n']), reverse=True)[:top_n]
            lines = ["scalp:", "  exec:", "    pro_exec:", "      combos:"]
            cid = 300
            for (rsi_name, macd_name, vwap_name, fib_name, mtf_ok, vol_name, bbw_name), agg in ranked:
                cid += 1
                rsi_bin = next(b for b in rsi_bins if b[0]==rsi_name)
                macd_val = next(b for b in macd_bins if b[0]==macd_name)[2]
                vwap_bin = next(b for b in vwap_bins if b[0]==vwap_name)
                vol_min = next(b for b in vol_bins if b[0]==vol_name)[2]
                bbw_max = next(b for b in bbw_bins if b[0]==bbw_name)[2]
                wr = (agg['w']/agg['n']*100.0)
                lines.extend([
                    f"        - id: {cid}",
                    f"          rsi_min: {rsi_bin[2]}",
                    f"          rsi_max: {rsi_bin[3]}",
                    f"          macd_hist: {macd_val}",
                    f"          vwap_min: {vwap_bin[2]}",
                    f"          vwap_max: {vwap_bin[3]}",
                    f"          fib_zone: \"{fib_name}\"",
                    f"          mtf_agree_15: {str(bool(mtf_ok)).lower()}",
                    f"          volume_ratio_min: {vol_min}",
                    f"          bbw_max: {bbw_max}",
                    f"          risk_percent: 1.0",
                    f"          enabled: true  # WR {wr:.1f}% (N={agg['n']})"
                ])
            return {'ranked': ranked, 'yaml': lines}
        except Exception as e:
            logger.error(f"ML recommend_strategies error: {e}")
            return {'error': 'internal'}

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
        try:
            import os as _os
            env_off = str(_os.getenv('DISABLE_ML', '')).strip().lower() in ('1','true','yes','on')
        except Exception:
            env_off = False
        _scalp_scorer = ScalpMLScorer(enabled=(False if env_off else enabled))
    return _scalp_scorer
