from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ShadowTrade:
    symbol: str
    strategy: str  # 'pullback' | 'enhanced_mr'
    side: str      # 'long' | 'short'
    entry: float
    sl: float
    tp: float
    start_time: datetime
    ml_score: float
    features: Dict
    outcome: Optional[str] = None  # 'win' | 'loss'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None


class ShadowSimTracker:
    def __init__(self) -> None:
        self.active: Dict[str, List[ShadowTrade]] = {}
        self.closed: Dict[str, List[ShadowTrade]] = {
            'pullback': [],
            'enhanced_mr': []
        }

    def _adjust_with_ml(self, trade: ShadowTrade) -> ShadowTrade:
        # Bounded nudges based on ml_score
        score = float(trade.ml_score or 0.0)
        # SL buffer factor in [0.9 .. 1.1]
        buf = 1.0 + max(-0.1, min(0.1, (score - 75.0) / 100.0 * 0.2))
        # RR in [2.3 .. 2.7] for both PB/MR (narrow band)
        rr = 2.5 + max(-0.2, min(0.2, (score - 75.0) / 100.0 * 0.4))

        entry = trade.entry
        if trade.side == 'long':
            R = abs(entry - trade.sl)
            new_sl = entry - R * buf
            new_tp = entry + rr * (entry - new_sl)
        else:
            R = abs(trade.sl - entry)
            new_sl = entry + R * buf
            new_tp = entry - rr * (new_sl - entry)

        # Enforce a minimal stop distance similar to core logic (1% for 15m strategies)
        min_dist = max(entry * 0.01, 1e-8)
        if trade.side == 'long' and (entry - new_sl) < min_dist:
            new_sl = entry - min_dist
            new_tp = entry + rr * (entry - new_sl)
        if trade.side == 'short' and (new_sl - entry) < min_dist:
            new_sl = entry + min_dist
            new_tp = entry - rr * (new_sl - entry)

        trade.sl = float(new_sl)
        trade.tp = float(new_tp)
        return trade

    def record_shadow_trade(self, strategy: str, symbol: str, side: str,
                            entry: float, sl: float, tp: float,
                            ml_score: float, features: Dict) -> ShadowTrade:
        t = ShadowTrade(
            symbol=symbol,
            strategy=strategy,
            side=side,
            entry=float(entry),
            sl=float(sl),
            tp=float(tp),
            start_time=datetime.utcnow(),
            ml_score=float(ml_score or 0.0),
            features=features or {}
        )
        t = self._adjust_with_ml(t)
        self.active.setdefault(symbol, []).append(t)
        return t

    def update_prices(self, symbol: str, current_price: float):
        if symbol not in self.active:
            return
        remaining: List[ShadowTrade] = []
        for tr in self.active[symbol]:
            if tr.outcome is not None:
                continue
            if tr.side == 'long':
                if current_price >= tr.tp:
                    tr.outcome = 'win'
                    tr.exit_price = float(current_price)
                    tr.exit_time = datetime.utcnow()
                    self.closed.setdefault(tr.strategy, []).append(tr)
                    continue
                elif current_price <= tr.sl:
                    tr.outcome = 'loss'
                    tr.exit_price = float(current_price)
                    tr.exit_time = datetime.utcnow()
                    self.closed.setdefault(tr.strategy, []).append(tr)
                    continue
            else:
                if current_price <= tr.tp:
                    tr.outcome = 'win'
                    tr.exit_price = float(current_price)
                    tr.exit_time = datetime.utcnow()
                    self.closed.setdefault(tr.strategy, []).append(tr)
                    continue
                elif current_price >= tr.sl:
                    tr.outcome = 'loss'
                    tr.exit_price = float(current_price)
                    tr.exit_time = datetime.utcnow()
                    self.closed.setdefault(tr.strategy, []).append(tr)
                    continue
            remaining.append(tr)
        if remaining:
            self.active[symbol] = remaining
        else:
            del self.active[symbol]

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for strat in ('pullback', 'enhanced_mr'):
            arr = self.closed.get(strat, [])
            wins = sum(1 for t in arr if t.outcome == 'win')
            losses = sum(1 for t in arr if t.outcome == 'loss')
            total = wins + losses
            wr = (wins / total * 100.0) if total else 0.0
            out[strat] = {
                'wins': wins,
                'losses': losses,
                'total': total,
                'wr': wr
            }
        return out


_shadow_tracker: Optional[ShadowSimTracker] = None


def get_shadow_tracker() -> ShadowSimTracker:
    global _shadow_tracker
    if _shadow_tracker is None:
        _shadow_tracker = ShadowSimTracker()
    return _shadow_tracker

