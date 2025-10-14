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

    def record_scalp_signal(self, symbol: str, signal: Dict, ml_score: float, was_executed: bool, features: Dict) -> ScalpPhantomTrade:
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


_scalp_phantom_tracker = None

def get_scalp_phantom_tracker() -> ScalpPhantomTracker:
    global _scalp_phantom_tracker
    if _scalp_phantom_tracker is None:
        _scalp_phantom_tracker = ScalpPhantomTracker()
    return _scalp_phantom_tracker
