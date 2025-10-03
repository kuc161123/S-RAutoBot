from __future__ import annotations
"""
Dedicated Phantom Tracker for Scalping Strategy
Stores and labels scalp phantom trades separately from other strategies.
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
        self.active: Dict[str, ScalpPhantomTrade] = {}
        self.completed: Dict[str, List[ScalpPhantomTrade]] = {}
        self._load()

    def _load(self):
        if not self.redis_client:
            return
        try:
            act = self.redis_client.get('scalp_phantom:active')
            if act:
                data = json.loads(act)
                for sym, rec in data.items():
                    self.active[sym] = ScalpPhantomTrade.from_dict(rec)
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
            act = {sym: t.to_dict() for sym, t in self.active.items()}
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
        # Enforce single-active-per-symbol for scalp phantom signals
        if symbol in self.active:
            try:
                import os, redis
                url = os.getenv('REDIS_URL')
                r = redis.from_url(url, decode_responses=True) if url else None
                if r:
                    from datetime import datetime as _dt
                    day = _dt.utcnow().strftime('%Y%m%d')
                    r.incr(f'phantom:blocked:{day}')
                    r.incr(f'phantom:blocked:{day}:scalp')
            except Exception:
                pass
            logger.info(f"[{symbol}] Scalp phantom blocked: active trade in progress")
            return self.active[symbol]

        ph = ScalpPhantomTrade(
            symbol=symbol,
            side=signal['side'],
            entry_price=signal['entry'],
            stop_loss=signal['sl'],
            take_profit=signal['tp'],
            signal_time=datetime.utcnow(),
            ml_score=ml_score,
            was_executed=was_executed,
            features=features or {}
        )
        self.active[symbol] = ph
        self._save()
        logger.info(f"[{symbol}] Scalp phantom recorded: {signal['side'].upper()} {signal['entry']:.4f}")
        return ph

    def update_scalp_phantom_prices(self, symbol: str, current_price: float, df: Optional[pd.DataFrame] = None):  # type: ignore
        if symbol not in self.active:
            return
        ph = self.active[symbol]
        # Track extremes
        if ph.side == 'long':
            ph.max_favorable = max(ph.max_favorable or current_price, current_price)
            ph.max_adverse = min(ph.max_adverse or current_price, current_price)
            if current_price >= ph.take_profit:
                self._close(symbol, current_price, 'win')
            elif current_price <= ph.stop_loss:
                self._close(symbol, current_price, 'loss')
        else:
            ph.max_favorable = min(ph.max_favorable or current_price, current_price)
            ph.max_adverse = max(ph.max_adverse or current_price, current_price)
            if current_price <= ph.take_profit:
                self._close(symbol, current_price, 'win')
            elif current_price >= ph.stop_loss:
                self._close(symbol, current_price, 'loss')

    def _close(self, symbol: str, exit_price: float, outcome: str):
        if symbol not in self.active:
            return
        ph = self.active[symbol]
        ph.outcome = outcome
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
        del self.active[symbol]
        self._save()
        logger.info(f"[{symbol}] Scalp PHANTOM closed: {'✅ WIN' if outcome=='win' else '❌ LOSS'} ({ph.pnl_percent:+.2f}%)")

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
