from __future__ import annotations
"""
Dedicated Phantom Tracker for Scalping Strategy
Stores and labels scalp phantom trades separately from other strategies.
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
import uuid
import yaml
from position_mgr import round_step
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

try:
    import redis
except Exception:
    redis = None


@dataclass
class ScalpPhantomTrade:
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_time: datetime
    ml_score: float
    was_executed: bool
    features: Dict
    outcome: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_percent: Optional[float] = None
    max_favorable: Optional[float] = None
    max_adverse: Optional[float] = None
    one_r_hit: Optional[bool] = None
    two_r_hit: Optional[bool] = None
    realized_rr: Optional[float] = None
    exit_reason: Optional[str] = None
    phantom_id: str = ""

    def to_dict(self):
        d = asdict(self)
        d['signal_time'] = self.signal_time.isoformat()
        if self.exit_time:
            d['exit_time'] = self.exit_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict):
        dd = d.copy()
        dd['signal_time'] = datetime.fromisoformat(dd['signal_time'])
        if dd.get('exit_time'):
            dd['exit_time'] = datetime.fromisoformat(dd['exit_time'])
        return cls(**dd)


class ScalpPhantomTracker:
    def __init__(self):
        self.redis_client = None
        if redis:
            try:
                url = os.getenv('REDIS_URL')
                if url:
                    self.redis_client = redis.from_url(url, decode_responses=True)
                    self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Scalp Phantom Redis unavailable: {e}")
        # Allow multiple active scalp phantoms per symbol
        self.active: Dict[str, List[ScalpPhantomTrade]] = {}
        self.completed: Dict[str, List[ScalpPhantomTrade]] = {}
        # Local blocked counters fallback: day (YYYYMMDD) -> counts
        self._blocked_counts: Dict[str, Dict[str, int]] = {}
        # Optional notifier (e.g., Telegram) for open/close events
        self.notifier: Optional[Callable] = None
        self._load()
        # Default timeout hours for scalp phantom (config override available)
        self.timeout_hours: int = 24
        # Symbol meta for tick size
        self._symbol_meta: Dict[str, Dict] = {}
        self._load_symbol_meta()
        # Configurable timeout override from config.yaml (scalp.explore.timeout_hours)
        try:
            with open('config.yaml','r') as f:
                cfg = yaml.safe_load(f) or {}
            sc_cfg = cfg.get('scalp', {}) or {}
            exp = sc_cfg.get('explore', {}) or {}
            to = int(exp.get('timeout_hours', self.timeout_hours))
            self.timeout_hours = max(1, to)
        except Exception as _e:
            logger.debug(f"ScalpPhantom: using default timeout {self.timeout_hours}h ({_e})")

    def cancel_active(self, symbol: str):
        """Cancel and remove any active scalp phantoms for a symbol.

        Used when a high-ML execution occurs to avoid duplicate tracking.
        """
        try:
            if symbol in self.active:
                del self.active[symbol]
                self._save()
                logger.info(f"[{symbol}] Scalp phantom canceled due to executed trade")
        except Exception:
            pass

    def set_notifier(self, notifier: Optional[Callable]):
        """Attach a notifier callable(trade) for open/close events."""
        self.notifier = notifier

    def force_close_executed(self, symbol: str, exit_price: float, exit_reason: str = 'manual'):
        """Force-close any executed scalp phantom mirrors to align with exchange closure.

        Determines win/loss relative to original TP/SL/entry and closes the mirror so
        the active phantom list reflects reality after downtime or reconnection.
        """
        try:
            if symbol not in self.active:
                return
            keep: List[ScalpPhantomTrade] = []
            for ph in list(self.active.get(symbol, [])):
                if getattr(ph, 'was_executed', False):
                    # Decide outcome based on TP/SL proximity if exit_reason known, otherwise by P&L
                    if ph.side == 'long':
                        outcome = 'win' if exit_price >= ph.take_profit else ('loss' if exit_price <= ph.stop_loss else ('win' if exit_price > ph.entry_price else 'loss'))
                    else:
                        outcome = 'win' if exit_price <= ph.take_profit else ('loss' if exit_price >= ph.stop_loss else ('win' if exit_price < ph.entry_price else 'loss'))
                    self._close(symbol, ph, exit_price, outcome)
                else:
                    keep.append(ph)
            if keep:
                self.active[symbol] = keep
            else:
                try:
                    del self.active[symbol]
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[{symbol}] scalp force_close_executed error: {e}")

    def sweep_timeouts(self) -> int:
        """Cancel any active scalp phantoms that exceeded timeout without requiring price updates.

        Returns number of phantoms cancelled.
        """
        try:
            if not self.active or not self.timeout_hours:
                return 0
            from datetime import datetime as _dt, timedelta as _td
            cutoff = _dt.utcnow() - _td(hours=int(self.timeout_hours))
            cancelled = 0
            for symbol in list(self.active.keys()):
                items = list(self.active.get(symbol, []))
                remaining: List[ScalpPhantomTrade] = []
                for ph in items:
                    try:
                        if ph.signal_time and ph.signal_time < cutoff:
                            ph.exit_reason = 'timeout'
                            # Close at entry price to avoid biasing P&L
                            self._close(symbol, ph, ph.entry_price, 'timeout')
                            cancelled += 1
                        else:
                            remaining.append(ph)
                    except Exception:
                        remaining.append(ph)
                if remaining:
                    self.active[symbol] = remaining
                else:
                    try:
                        del self.active[symbol]
                    except Exception:
                        pass
            if cancelled:
                self._save()
            return cancelled
        except Exception:
            return 0

    def backfill_active(self, fetch_klines: Optional[Callable[[str, int, Optional[int]], list]] = None) -> int:
        """Retro-close active scalp phantoms by scanning historical candles.

        Args:
            fetch_klines: function(symbol, start_ms, end_ms) -> list of bars with
                          dict-like items containing 'high' and 'low' floats.
                          If None, no backfill is performed.

        Returns:
            Number of phantoms closed by backfill.
        """
        if not fetch_klines or not self.active:
            return 0
        closed = 0
        from datetime import datetime as _dt
        now_ms = int(_dt.utcnow().timestamp() * 1000)
        for symbol in list(self.active.keys()):
            items = list(self.active.get(symbol, []))
            remaining: List[ScalpPhantomTrade] = []
            for ph in items:
                try:
                    start_ms = int(ph.signal_time.timestamp() * 1000)
                    bars = fetch_klines(symbol, start_ms, now_ms) or []
                    outcome = None
                    for b in bars:
                        try:
                            hi = float(b.get('high'))
                            lo = float(b.get('low'))
                        except Exception:
                            try:
                                hi = float(b[2])
                                lo = float(b[3])
                            except Exception:
                                continue
                        if ph.side == 'long':
                            hit_sl = lo <= ph.stop_loss
                            hit_tp = hi >= ph.take_profit
                            if hit_sl or hit_tp:
                                outcome = 'loss' if hit_sl else 'win'
                                break
                        else:
                            hit_sl = hi >= ph.stop_loss
                            hit_tp = lo <= ph.take_profit
                            if hit_sl or hit_tp:
                                outcome = 'loss' if hit_sl else 'win'
                                break
                    if outcome:
                        exit_reason = 'tp' if outcome == 'win' else 'sl'
                        self._close(symbol, ph, ph.take_profit if outcome=='win' else ph.stop_loss, outcome)
                        closed += 1
                    else:
                        remaining.append(ph)
                except Exception:
                    remaining.append(ph)
            if remaining:
                self.active[symbol] = remaining
            else:
                try:
                    del self.active[symbol]
                except Exception:
                    pass
        if closed:
            self._save()
        return closed

    def update_with_bar(self, symbol: str, high: float, low: float, bar_time: Optional[datetime] = None) -> int:
        """Update active phantoms for a symbol with a single bar's high/low.

        Closes phantoms that hit TP/SL and updates max_favorable/adverse. Returns
        the number of phantoms closed by this update.
        """
        try:
            if symbol not in self.active:
                return 0
            closed = 0
            items = list(self.active.get(symbol, []))
            remaining: List[ScalpPhantomTrade] = []
            for ph in items:
                try:
                    # Track excursion metrics
                    if ph.side == 'long':
                        try:
                            ph.max_favorable = max(ph.max_favorable or ph.entry_price, high)
                            ph.max_adverse = min(ph.max_adverse or ph.entry_price, low)
                        except Exception:
                            pass
                        hit_sl = low <= ph.stop_loss
                        hit_tp = high >= ph.take_profit
                        if hit_sl or hit_tp:
                            ph.exit_reason = 'sl' if hit_sl else 'tp'
                            exit_price = ph.stop_loss if hit_sl else ph.take_profit
                            self._close(symbol, ph, exit_price, 'loss' if hit_sl else 'win')
                            closed += 1
                            continue
                    else:
                        try:
                            ph.max_favorable = min(ph.max_favorable or ph.entry_price, low)
                            ph.max_adverse = max(ph.max_adverse or ph.entry_price, high)
                        except Exception:
                            pass
                        hit_sl = high >= ph.stop_loss
                        hit_tp = low <= ph.take_profit
                        if hit_sl or hit_tp:
                            ph.exit_reason = 'sl' if hit_sl else 'tp'
                            exit_price = ph.stop_loss if hit_sl else ph.take_profit
                            self._close(symbol, ph, exit_price, 'loss' if hit_sl else 'win')
                            closed += 1
                            continue
                    remaining.append(ph)
                except Exception:
                    remaining.append(ph)
            if remaining:
                self.active[symbol] = remaining
            else:
                try:
                    del self.active[symbol]
                except Exception:
                    pass
            if closed:
                self._save()
            return closed
        except Exception:
            return 0

    def _load_symbol_meta(self):
        try:
            with open('config.yaml','r') as f:
                cfg = yaml.safe_load(f)
            sm = (cfg or {}).get('symbol_meta', {}) or {}
            if isinstance(sm, dict):
                self._symbol_meta = sm
        except Exception as e:
            logger.debug(f"ScalpPhantom: failed to load symbol meta: {e}")

    def has_active(self, symbol: str) -> bool:
        """Return True if there is an active scalp phantom for the symbol."""
        try:
            return bool(self.active.get(symbol))
        except Exception:
            return False

    def _tick_size_for(self, symbol: str) -> float:
        try:
            if symbol in self._symbol_meta and isinstance(self._symbol_meta[symbol], dict):
                return float(self._symbol_meta[symbol].get('tick_size', 0.000001) or 0.000001)
            default = self._symbol_meta.get('default', {}) if isinstance(self._symbol_meta, dict) else {}
            return float(default.get('tick_size', 0.000001) or 0.000001)
        except Exception:
            return 0.000001

    def _load(self):
        if not self.redis_client:
            return
        try:
            act = self.redis_client.get('scalp_phantom:active')
            if act:
                data = json.loads(act)
                for sym, rec in data.items():
                    items: List[ScalpPhantomTrade] = []
                    if isinstance(rec, list):
                        for t in rec:
                            try:
                                items.append(ScalpPhantomTrade.from_dict(t))
                            except Exception:
                                pass
                    elif isinstance(rec, dict):
                        try:
                            items.append(ScalpPhantomTrade.from_dict(rec))
                        except Exception:
                            pass
                    if items:
                        self.active[sym] = items
            comp = self.redis_client.get('scalp_phantom:completed')
            if comp:
                arr = json.loads(comp)
                for rec in arr:
                    pt = ScalpPhantomTrade.from_dict(rec)
                    self.completed.setdefault(pt.symbol, []).append(pt)
        except Exception as e:
            logger.error(f"Scalp Phantom load error: {e}")

    def _save(self):
        if not self.redis_client:
            return
        try:
            act = {sym: [t.to_dict() for t in lst] for sym, lst in self.active.items()}
            self.redis_client.set('scalp_phantom:active', json.dumps(act))
            comp_all = []
            cutoff = datetime.utcnow() - timedelta(days=30)
            for trades in self.completed.values():
                for t in trades:
                    if t.exit_time and t.exit_time >= cutoff:
                        comp_all.append(t.to_dict())
            comp_all = comp_all[-1500:]
            self.redis_client.set('scalp_phantom:completed', json.dumps(comp_all))
        except Exception as e:
            logger.error(f"Scalp Phantom save error: {e}")

    def record_scalp_signal(self, symbol: str, signal: Dict, ml_score: float, was_executed: bool, features: Dict) -> ScalpPhantomTrade | None:
        # Regime context: do not drop by volatility or micro-trend; only annotate for learning
        try:
            if not was_executed:
                vol = str((features or {}).get('volatility_regime', 'normal'))
                # Previously: micro-trend gating by EMA slopes. Now: do not drop — just annotate for analysis.
                try:
                    sl_fast = float((features or {}).get('ema_slope_fast', 0.0))
                except Exception:
                    sl_fast = 0.0
                try:
                    sl_slow = float((features or {}).get('ema_slope_slow', 0.0))
                except Exception:
                    sl_slow = 0.0
                side = str(signal.get('side', ''))
                try:
                    if isinstance(features, dict):
                        features = features.copy()
                        features['volatility_regime'] = vol
                        features['micro_slope_fast'] = sl_fast
                        features['micro_slope_slow'] = sl_slow
                        features['micro_match'] = bool((sl_fast >= 0.0 and sl_slow >= 0.0) if side=='long' else (sl_fast <= 0.0 and sl_slow <= 0.0))
                except Exception:
                    pass
        except Exception:
            # On any error in gating, default to recording to avoid losing data silently
            pass

        # Round TP/SL to tick size for better alignment
        try:
            ts = self._tick_size_for(symbol)
            raw_tp = float(signal.get('tp'))
            raw_sl = float(signal.get('sl'))
            r_tp = round_step(raw_tp, ts)
            r_sl = round_step(raw_sl, ts)
            if r_tp != raw_tp or r_sl != raw_sl:
                signal = signal.copy()
                signal['tp'] = r_tp
                signal['sl'] = r_sl
                logger.debug(f"[{symbol}] Scalp phantom TP/SL rounded to tick {ts}: TP {raw_tp}→{r_tp}, SL {raw_sl}→{r_sl}")
        except Exception:
            pass

        # Allow multiple active phantoms per symbol

        # Annotate features with versioning for data lineage
        try:
            if isinstance(features, dict):
                features = features.copy()
                features.setdefault('feature_version', 'scalp_v1')
        except Exception:
            pass

        ph = ScalpPhantomTrade(
            symbol=symbol,
            side=signal['side'],
            entry_price=signal['entry'],
            stop_loss=signal['sl'],
            take_profit=signal['tp'],
            signal_time=datetime.utcnow(),
            ml_score=ml_score,
            was_executed=was_executed,
            features=features or {},
            phantom_id=(uuid.uuid4().hex[:8])
        )
        self.active.setdefault(symbol, []).append(ph)
        self._save()
        try:
            tag = 'mirror' if bool(was_executed) else 'phantom'
            logger.info(f"[{symbol}] Scalp {tag} recorded: {signal['side'].upper()} {signal['entry']:.4f}")
        except Exception:
            logger.info(f"[{symbol}] Scalp phantom recorded: {signal['side'].upper()} {signal['entry']:.4f}")
        # Notify on open immediately (phantom-only)
        try:
            if self.notifier and not was_executed:
                res = self.notifier(ph)
                import asyncio
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
        except Exception:
            pass
        return ph

    def get_blocked_counts(self, day: Optional[str] = None) -> Dict[str, int]:
        # Return local blocked counters for a given day (YYYYMMDD)
        from datetime import datetime as _dt
        if day is None:
            day = _dt.utcnow().strftime('%Y%m%d')
        return self._blocked_counts.get(day, {'total': 0, 'trend': 0, 'mr': 0, 'scalp': 0})

    def update_scalp_phantom_prices(self, symbol: str, current_price: float, df: Optional[pd.DataFrame] = None):  # type: ignore
        if symbol not in self.active:
            return
        act_list = list(self.active.get(symbol, []))
        remaining: List[ScalpPhantomTrade] = []
        from datetime import datetime as _dt, timedelta as _td
        # Intrabar extremes (if df provided)
        try:
            cur_high = float(df['high'].iloc[-1]) if df is not None else current_price
            cur_low = float(df['low'].iloc[-1]) if df is not None else current_price
        except Exception:
            cur_high = current_price
            cur_low = current_price
        for ph in act_list:
            try:
                # Track extremes
                if ph.side == 'long':
                    ph.max_favorable = max(ph.max_favorable or current_price, current_price)
                    ph.max_adverse = min(ph.max_adverse or current_price, current_price)
                    # TP/SL checks
                    if cur_high >= ph.take_profit:
                        ph.exit_reason = 'tp'
                        self._close(symbol, ph, current_price, 'win')
                        continue
                    if cur_low <= ph.stop_loss:
                        ph.exit_reason = 'sl'
                        self._close(symbol, ph, current_price, 'loss')
                        continue
                else:
                    ph.max_favorable = min(ph.max_favorable or current_price, current_price)
                    ph.max_adverse = max(ph.max_adverse or current_price, current_price)
                    if cur_low <= ph.take_profit:
                        ph.exit_reason = 'tp'
                        self._close(symbol, ph, current_price, 'win')
                        continue
                    if cur_high >= ph.stop_loss:
                        ph.exit_reason = 'sl'
                        self._close(symbol, ph, current_price, 'loss')
                        continue

                # Timeout handling (per-phantom)
                try:
                    if self.timeout_hours and ph.signal_time:
                        if _dt.utcnow() - ph.signal_time > _td(hours=int(self.timeout_hours)):
                            # Cancel phantoms exceeding timeout; do not feed ML
                            ph.exit_reason = 'timeout'
                            self._close(symbol, ph, current_price, 'timeout')
                            continue
                except Exception:
                    pass

                # Keep active if not closed by TP/SL/timeout
                remaining.append(ph)
            except Exception:
                # On error, keep the phantom to try again next tick
                remaining.append(ph)

        if remaining:
            self.active[symbol] = remaining
        else:
            try:
                del self.active[symbol]
            except Exception:
                pass

    def _close(self, symbol: str, ph: ScalpPhantomTrade, exit_price: float, outcome: str):
        ph.outcome = outcome
        # Snap to exact TP/SL for clearer R:R on labeled tp/sl
        try:
            if str(getattr(ph, 'exit_reason', '')).lower() == 'tp':
                exit_price = float(ph.take_profit)
            elif str(getattr(ph, 'exit_reason', '')).lower() == 'sl':
                exit_price = float(ph.stop_loss)
        except Exception:
            pass
        ph.exit_price = exit_price
        ph.exit_time = datetime.utcnow()
        # PnL
        if ph.side == 'long':
            ph.pnl_percent = (exit_price - ph.entry_price) / ph.entry_price * 100
            R = ph.entry_price - ph.stop_loss
            ph.realized_rr = (exit_price - ph.entry_price) / R if R > 0 else 0.0
            one_r = ph.entry_price + R
            two_r = ph.entry_price + 2*R
            ph.one_r_hit = bool((ph.max_favorable or ph.entry_price) >= one_r)
            ph.two_r_hit = bool((ph.max_favorable or ph.entry_price) >= two_r)
        else:
            ph.pnl_percent = (ph.entry_price - exit_price) / ph.entry_price * 100
            R = ph.stop_loss - ph.entry_price
            ph.realized_rr = (ph.entry_price - exit_price) / R if R > 0 else 0.0
            one_r = ph.entry_price - R
            two_r = ph.entry_price - 2*R
            ph.one_r_hit = bool((ph.max_favorable or ph.entry_price) <= one_r)
            ph.two_r_hit = bool((ph.max_favorable or ph.entry_price) <= two_r)

        self.completed.setdefault(symbol, []).append(ph)
        try:
            self.active[symbol] = [t for t in self.active.get(symbol, []) if getattr(t,'phantom_id','') != getattr(ph,'phantom_id','')]
            if not self.active[symbol]:
                del self.active[symbol]
        except Exception:
            pass
        self._save()
        logger.info(f"[{symbol}] Scalp PHANTOM closed: {'✅ WIN' if outcome=='win' else '❌ LOSS'} ({ph.pnl_percent:+.2f}%)")

        # Notify on close
        try:
            if self.notifier:
                res = self.notifier(ph)
                import asyncio
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
        except Exception:
            pass

        # Update rolling WR list for WR guard (Scalp) — skip timeouts
        try:
            if self.redis_client and getattr(ph, 'exit_reason', None) != 'timeout':
                key = 'phantom:wr:scalp'
                val = '1' if outcome == 'win' else '0'
                self.redis_client.lpush(key, val)
                self.redis_client.ltrim(key, 0, 199)
        except Exception:
            pass

        # Feed outcome to Scalp ML scorer for learning — skip timeouts/cancels
        try:
            if str(getattr(ph, 'exit_reason', '')) != 'timeout' and outcome in ('win','loss'):
                from ml_scorer_scalp import get_scalp_scorer
                scorer = get_scalp_scorer()
                signal = {
                    'features': ph.features or {},
                    # Preserve whether this phantom mirrors an executed trade (for weighting/analysis)
                    'was_executed': bool(getattr(ph, 'was_executed', False)),
                    'exit_reason': getattr(ph, 'exit_reason', None)
                }
                scorer.record_outcome(signal, outcome, float(ph.pnl_percent or 0.0))
        except Exception as e:
            logger.debug(f"Scalp ML feed error: {e}")

    def get_learning_data(self) -> List[Dict]:
        out = []
        for trades in self.completed.values():
            for t in trades:
                if t.outcome in ['win','loss']:
                    out.append({
                        'features': t.features,
                        'outcome': 1 if t.outcome=='win' else 0,
                        'pnl_percent': t.pnl_percent,
                        'symbol': t.symbol,
                        'side': t.side,
                        'was_executed': t.was_executed,
                        'time_to_outcome_sec': int((t.exit_time - t.signal_time).total_seconds()) if t.exit_time else None,
                        'one_r_hit': t.one_r_hit,
                        'two_r_hit': t.two_r_hit,
                        'realized_rr': t.realized_rr
                    })
        return out

    def get_scalp_phantom_stats(self) -> Dict:
        total = sum(len(v) for v in self.completed.values())
        wins = sum(1 for arr in self.completed.values() for t in arr if t.outcome=='win')
        losses = total - wins
        wr = (wins/total*100) if total else 0.0
        return {'total': total, 'wins': wins, 'losses': losses, 'wr': wr}

    def compute_gate_status(self, phantom: ScalpPhantomTrade, config: Dict = None) -> Dict:
        """
        Compute which hard gates would pass/fail for this phantom based on its features.
        Uses current config gate thresholds or provided overrides.
        Returns: {htf: bool, vol: bool, body: bool, align_15m: bool, all: bool}
        """
        feats = phantom.features or {}
        side = phantom.side

        # Load gate thresholds from config if not provided
        if config is None:
            try:
                with open('config.yaml', 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                gate_cfg = ((cfg.get('reporting', {}) or {}).get('hard_gates', {}) or {})
            except Exception:
                gate_cfg = {}
        else:
            gate_cfg = config

        htf_min = float(gate_cfg.get('htf_min_ts15', 60.0))
        vol_min = float(gate_cfg.get('vol_ratio_min_3m', 1.30))
        body_min = float(gate_cfg.get('body_ratio_min_3m', 0.35))

        # HTF gate
        ts15 = float(feats.get('ts15', 0.0) or 0.0)
        htf_pass = ts15 >= htf_min

        # Volume gate
        vol_ratio = float(feats.get('volume_ratio', 0.0) or 0.0)
        vol_pass = vol_ratio >= vol_min

        # Body gate (size + direction)
        body_ratio = float(feats.get('body_ratio', 0.0) or 0.0)
        body_sign = str(feats.get('body_sign', ''))
        body_size_ok = body_ratio >= body_min
        body_dir_ok = (
            (side == 'long' and body_sign == 'up') or
            (side == 'short' and body_sign == 'down')
        )
        body_pass = body_size_ok and body_dir_ok

        # 15m alignment gate
        ema_dir_15m = str(feats.get('ema_dir_15m', 'none'))
        align_15m_pass = (
            (side == 'long' and ema_dir_15m == 'up') or
            (side == 'short' and ema_dir_15m == 'down')
        )

        # All gates pass
        all_pass = htf_pass and vol_pass and body_pass and align_15m_pass

        return {
            'htf': htf_pass,
            'vol': vol_pass,
            'body': body_pass,
            'align_15m': align_15m_pass,
            'all': all_pass,
            # Include values for debugging
            'htf_value': ts15,
            'vol_value': vol_ratio,
            'body_value': body_ratio,
            'body_sign': body_sign,
            'ema_dir_15m': ema_dir_15m
        }

    def get_gate_analysis(self, days: int = 30, min_samples: int = 20) -> Dict:
        """
        Analyze phantom win rates by gate pass/fail combinations.

        Args:
            days: Look back period in days
            min_samples: Minimum sample size to show a stat (default 20)

        Returns:
            Dict with overall stats, per-gate breakdowns, and combinations
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Filter completed phantoms within time window with defined outcomes
        phantoms = [
            p for arr in self.completed.values() for p in arr
            if p.exit_time and p.exit_time >= cutoff and p.outcome in ('win', 'loss')
        ]

        if not phantoms:
            return {'error': 'No phantom data available', 'total': 0}

        total = len(phantoms)
        wins = sum(1 for p in phantoms if p.outcome == 'win')
        baseline_wr = (wins / total * 100) if total > 0 else 0.0

        # Compute gate status for each phantom
        gate_statuses = [self.compute_gate_status(p) for p in phantoms]

        # Overall: all gates pass
        all_pass_phantoms = [p for i, p in enumerate(phantoms) if gate_statuses[i]['all']]
        all_pass_wins = sum(1 for p in all_pass_phantoms if p.outcome == 'win')
        all_pass_total = len(all_pass_phantoms)
        all_pass_wr = (all_pass_wins / all_pass_total * 100) if all_pass_total > 0 else 0.0

        # Per-gate analysis
        gates = ['htf', 'vol', 'body', 'align_15m']
        gate_stats = {}

        for gate in gates:
            gate_pass = [p for i, p in enumerate(phantoms) if gate_statuses[i][gate]]
            gate_fail = [p for i, p in enumerate(phantoms) if not gate_statuses[i][gate]]

            pass_wins = sum(1 for p in gate_pass if p.outcome == 'win')
            pass_total = len(gate_pass)
            pass_wr = (pass_wins / pass_total * 100) if pass_total > 0 else 0.0

            fail_wins = sum(1 for p in gate_fail if p.outcome == 'win')
            fail_total = len(gate_fail)
            fail_wr = (fail_wins / fail_total * 100) if fail_total > 0 else 0.0

            delta = pass_wr - fail_wr if (pass_total >= min_samples and fail_total >= min_samples) else None

            gate_stats[gate] = {
                'pass_wins': pass_wins,
                'pass_total': pass_total,
                'pass_wr': pass_wr,
                'fail_wins': fail_wins,
                'fail_total': fail_total,
                'fail_wr': fail_wr,
                'delta': delta,
                'sufficient_samples': pass_total >= min_samples and fail_total >= min_samples
            }

        # Gate combinations (bitmap: HVBA = HTF, Vol, Body, Align)
        from collections import defaultdict
        combo_stats = defaultdict(lambda: {'wins': 0, 'total': 0})

        for i, p in enumerate(phantoms):
            gs = gate_statuses[i]
            bitmap = ''.join(['1' if gs[g] else '0' for g in gates])
            combo_stats[bitmap]['total'] += 1
            if p.outcome == 'win':
                combo_stats[bitmap]['wins'] += 1

        # Sort combinations by total count
        sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        top_combos = []
        for bitmap, stats in sorted_combos[:10]:  # Top 10 combinations
            if stats['total'] >= min_samples:
                wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
                top_combos.append({
                    'bitmap': bitmap,
                    'wins': stats['wins'],
                    'total': stats['total'],
                    'wr': wr
                })

        return {
            'days': days,
            'total_phantoms': total,
            'total_wins': wins,
            'baseline_wr': baseline_wr,
            'all_gates_pass': {
                'wins': all_pass_wins,
                'total': all_pass_total,
                'wr': all_pass_wr,
                'delta': all_pass_wr - baseline_wr
            },
            'gate_stats': gate_stats,
            'top_combinations': top_combos,
            'min_samples': min_samples
        }


_scalp_phantom_tracker = None

def get_scalp_phantom_tracker() -> ScalpPhantomTracker:
    global _scalp_phantom_tracker
    if _scalp_phantom_tracker is None:
        _scalp_phantom_tracker = ScalpPhantomTracker()
    return _scalp_phantom_tracker
