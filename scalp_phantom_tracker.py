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

        # Startup health check
        try:
            total_active = sum(len(v) for v in self.active.values())
            total_completed = sum(len(v) for v in self.completed.values())
            logger.info(f"🩳 Scalp Phantom Tracker initialized: {total_active} active, {total_completed} completed")
            if total_active > 100:
                logger.warning(f"⚠️ PHANTOM HEALTH: {total_active} active phantoms (threshold: 100) - check outcome detection!")
            elif total_active > 50:
                logger.warning(f"⚠️ Elevated phantom count: {total_active} active (threshold: 50) - monitor closely")
        except Exception as e:
            logger.debug(f"Scalp phantom startup health check error: {e}")

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
            # Load active phantoms
            cnt_active = 0
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
                        cnt_active += len(items)

            # Log active phantoms loaded
            num_symbols = len(self.active)
            logger.info(f"Loaded {cnt_active} active scalp phantom trades across {num_symbols} symbols")

            # Load completed phantoms
            comp = self.redis_client.get('scalp_phantom:completed')
            if comp:
                arr = json.loads(comp)
                for rec in arr:
                    pt = ScalpPhantomTrade.from_dict(rec)
                    self.completed.setdefault(pt.symbol, []).append(pt)

            # Log completed phantoms loaded
            total_completed = sum(len(trades) for trades in self.completed.values())
            logger.info(f"Loaded {total_completed} completed scalp phantom trades")

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
            logger.info(f"[{symbol}] 🩳 Scalp {tag} recorded: {signal['side'].upper()} entry={signal['entry']:.4f} TP={signal['tp']:.4f} SL={signal['sl']:.4f} (id={ph.phantom_id})")
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
        # Log phantom update activity (DEBUG - too noisy for INFO)
        try:
            logger.debug(f"[{symbol}] 🩳 Phantom update: {len(act_list)} active phantom(s), current_price={current_price:.4f}")
        except Exception:
            pass
        remaining: List[ScalpPhantomTrade] = []
        from datetime import datetime as _dt, timedelta as _td
        # Intrabar extremes (if df provided)
        try:
            if df is not None and not df.empty:
                cur_high = float(df['high'].iloc[-1])
                cur_low = float(df['low'].iloc[-1])
                logger.debug(f"[{symbol}] 🩳 Phantom update: Using df high/low: {cur_high:.4f}/{cur_low:.4f}")
            else:
                cur_high = current_price
                cur_low = current_price
                # Warn if df is None - intrabar TP/SL hits may be missed (rate limited per symbol)
                if not hasattr(self, '_df_warning_logged'):
                    self._df_warning_logged = {}
                from time import time as _time
                now = _time()
                last_warn = self._df_warning_logged.get(symbol, 0)
                if (now - last_warn) > 300:  # Warn once per 5 minutes per symbol
                    logger.warning(f"[{symbol}] ⚠️ Phantom update: No dataframe provided, using current_price for high/low (may miss intrabar TP/SL hits!)")
                    self._df_warning_logged[symbol] = now
        except Exception as e:
            logger.warning(f"[{symbol}] Phantom update: Error getting high/low from df: {e}, using current_price")
            cur_high = current_price
            cur_low = current_price
        closed_count = 0
        for ph in act_list:
            try:
                # Track extremes
                if ph.side == 'long':
                    ph.max_favorable = max(ph.max_favorable or current_price, current_price)
                    ph.max_adverse = min(ph.max_adverse or current_price, current_price)
                    # TP/SL checks (log all checks to verify they're happening)
                    if cur_high >= ph.take_profit:
                        ph.exit_reason = 'tp'
                        logger.info(f"[{symbol}] 🩳 Phantom LONG TP HIT: cur_high={cur_high:.4f} >= TP={ph.take_profit:.4f} (id={ph.phantom_id})")
                        self._close(symbol, ph, current_price, 'win')
                        closed_count += 1
                        continue
                    if cur_low <= ph.stop_loss:
                        ph.exit_reason = 'sl'
                        logger.info(f"[{symbol}] 🩳 Phantom LONG SL HIT: cur_low={cur_low:.4f} <= SL={ph.stop_loss:.4f} (id={ph.phantom_id})")
                        self._close(symbol, ph, current_price, 'loss')
                        closed_count += 1
                        continue
                else:
                    ph.max_favorable = min(ph.max_favorable or current_price, current_price)
                    ph.max_adverse = max(ph.max_adverse or current_price, current_price)
                    # TP/SL checks (log all checks to verify they're happening)
                    if cur_low <= ph.take_profit:
                        ph.exit_reason = 'tp'
                        logger.info(f"[{symbol}] 🩳 Phantom SHORT TP HIT: cur_low={cur_low:.4f} <= TP={ph.take_profit:.4f} (id={ph.phantom_id})")
                        self._close(symbol, ph, current_price, 'win')
                        closed_count += 1
                        continue
                    if cur_high >= ph.stop_loss:
                        ph.exit_reason = 'sl'
                        logger.info(f"[{symbol}] 🩳 Phantom SHORT SL HIT: cur_high={cur_high:.4f} >= SL={ph.stop_loss:.4f} (id={ph.phantom_id})")
                        self._close(symbol, ph, current_price, 'loss')
                        closed_count += 1
                        continue

                # Timeout handling (per-phantom)
                try:
                    if self.timeout_hours and ph.signal_time:
                        age_hours = (_dt.utcnow() - ph.signal_time).total_seconds() / 3600
                        if _dt.utcnow() - ph.signal_time > _td(hours=int(self.timeout_hours)):
                            # Cancel phantoms exceeding timeout; do not feed ML
                            ph.exit_reason = 'timeout'
                            logger.info(f"[{symbol}] 🩳 Phantom TIMEOUT: age={age_hours:.1f}h > limit={int(self.timeout_hours)}h (id={ph.phantom_id})")
                            self._close(symbol, ph, current_price, 'timeout')
                            closed_count += 1
                            continue
                except Exception as e:
                    logger.warning(f"[{symbol}] Phantom timeout check error for {getattr(ph, 'phantom_id', 'unknown')}: {e}")

                # Keep active if not closed by TP/SL/timeout
                remaining.append(ph)
            except Exception as e:
                # On error, keep the phantom to try again next tick
                logger.error(f"[{symbol}] Phantom update error for {getattr(ph, 'phantom_id', 'unknown')}: {e} "
                           f"(side={ph.side}, entry={ph.entry_price}, tp={ph.take_profit}, sl={ph.stop_loss}, "
                           f"cur_price={current_price}, cur_high={cur_high}, cur_low={cur_low})")
                remaining.append(ph)

        if remaining:
            self.active[symbol] = remaining
        else:
            try:
                del self.active[symbol]
            except Exception:
                pass

        # Log summary only when phantoms close (avoid flooding)
        try:
            if closed_count > 0:
                logger.info(f"[{symbol}] 🩳 Phantom update: {closed_count} closed, {len(remaining)} still active")
        except Exception:
            pass

        # Health monitoring: Alert if too many active phantoms (rate limited to avoid spam)
        try:
            total_active = sum(len(v) for v in self.active.values())
            # Only log if count changed significantly or first time
            if not hasattr(self, '_last_health_check'):
                self._last_health_check = {'count': 0, 'time': 0}

            from time import time as _time
            now = _time()
            count_changed = abs(total_active - self._last_health_check['count']) > 50
            time_elapsed = (now - self._last_health_check['time']) > 300  # 5 minutes

            if count_changed or time_elapsed:
                if total_active > 100:
                    logger.warning(f"⚠️ PHANTOM HEALTH: {total_active} active phantoms (threshold: 100) - check timeout/outcome detection")
                elif total_active > 50:
                    logger.info(f"Phantom count elevated: {total_active} active (threshold: 50)")
                self._last_health_check = {'count': total_active, 'time': now}
        except Exception as e:
            logger.debug(f"Phantom health check error: {e}")

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
        """Return Scalp phantom stats excluding timeouts from WR.

        - total: decisive count (wins + losses)
        - wins: outcome == 'win'
        - losses: outcome == 'loss'
        - timeouts: outcome == 'timeout' (not included in WR)
        - wr: wins / (wins + losses)
        """
        decisive = []
        timeouts = 0
        for arr in self.completed.values():
            for t in arr:
                try:
                    if t.outcome in ('win', 'loss') and not getattr(t, 'was_executed', False):
                        decisive.append(t)
                    elif t.outcome == 'timeout' and not getattr(t, 'was_executed', False):
                        timeouts += 1
                except Exception:
                    continue
        total = len(decisive)
        wins = sum(1 for t in decisive if getattr(t, 'outcome', None) == 'win')
        losses = sum(1 for t in decisive if getattr(t, 'outcome', None) == 'loss')
        wr = (wins/total*100.0) if total else 0.0
        return {'total': total, 'wins': wins, 'losses': losses, 'wr': wr, 'timeouts': timeouts}

    def compute_gate_status(self, phantom: ScalpPhantomTrade, config: Dict = None) -> Dict:
        """
        Compute which hard gates and variables would pass/fail for this phantom.
        Now tracks 26+ variables across multiple categories for optimization.
        Returns: Dict with boolean pass/fail for each variable + debug values
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

        # === ORIGINAL 4 GATES (backward compatible) ===
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

        # === NEW BODY VARIATIONS ===
        body_040 = body_ratio >= 0.40 and body_dir_ok
        body_045 = body_ratio >= 0.45 and body_dir_ok
        body_050 = body_ratio >= 0.50 and body_dir_ok
        body_060 = body_ratio >= 0.60 and body_dir_ok

        # Wick alignment (rejection wick in trade direction)
        upper_wick = float(feats.get('upper_wick_ratio', 0.0) or 0.0)
        lower_wick = float(feats.get('lower_wick_ratio', 0.0) or 0.0)
        wick_align = (lower_wick > upper_wick) if side == 'long' else (upper_wick > lower_wick)

        # === NEW VWAP DISTANCE VARIATIONS ===
        vwap_dist = float(feats.get('vwap_dist_atr', 999.0) or 999.0)
        vwap_045 = vwap_dist < 0.45
        vwap_060 = vwap_dist < 0.60
        vwap_080 = vwap_dist < 0.80
        vwap_100 = vwap_dist < 1.00

        # === NEW VOLUME VARIATIONS ===
        vol_110 = vol_ratio >= 1.10
        vol_120 = vol_ratio >= 1.20
        vol_150 = vol_ratio >= 1.50

        # === BB WIDTH PERCENTILES ===
        # Using absolute thresholds as proxy (can be refined later with percentile calc)
        bb_width = float(feats.get('bb_width_pct', 0.0) or 0.0)
        bb_width_60p = bb_width >= 0.015  # Rough proxy for 60th percentile
        bb_width_70p = bb_width >= 0.018  # 70th percentile proxy
        bb_width_80p = bb_width >= 0.022  # 80th percentile proxy

        # === Q-SCORE VARIATIONS ===
        # Note: q_score may not be stored in features yet - will be 0 if missing
        q_score = float(feats.get('q_score', 0.0) or 0.0)
        q_040 = q_score >= 40.0
        q_050 = q_score >= 50.0
        q_060 = q_score >= 60.0
        q_070 = q_score >= 70.0

        # === IMPULSE VARIATIONS ===
        impulse = float(feats.get('impulse_ratio', 0.0) or 0.0)
        impulse_040 = impulse >= 0.40
        impulse_060 = impulse >= 0.60

        # === MICRO SEQUENCE ===
        # Will be False if not yet implemented in feature storage
        micro_seq = bool(feats.get('micro_seq_aligned', False))

        # === HTF VARIATIONS ===
        htf_070 = ts15 >= 70.0
        htf_080 = ts15 >= 80.0

        return {
            # ===== ORIGINAL 4 (backward compatible) =====
            'htf': htf_pass,
            'vol': vol_pass,
            'body': body_pass,
            'align_15m': align_15m_pass,
            'all': all_pass,

            # ===== BODY VARIATIONS (5) =====
            'body_040': body_040,
            'body_045': body_045,
            'body_050': body_050,
            'body_060': body_060,
            'wick_align': wick_align,

            # ===== VWAP VARIATIONS (4) =====
            'vwap_045': vwap_045,
            'vwap_060': vwap_060,
            'vwap_080': vwap_080,
            'vwap_100': vwap_100,

            # ===== VOLUME VARIATIONS (3) =====
            'vol_110': vol_110,
            'vol_120': vol_120,
            'vol_150': vol_150,

            # ===== BB WIDTH (3) =====
            'bb_width_60p': bb_width_60p,
            'bb_width_70p': bb_width_70p,
            'bb_width_80p': bb_width_80p,

            # ===== Q-SCORE (4) =====
            'q_040': q_040,
            'q_050': q_050,
            'q_060': q_060,
            'q_070': q_070,

            # ===== IMPULSE (2) =====
            'impulse_040': impulse_040,
            'impulse_060': impulse_060,

            # ===== MICRO (1) =====
            'micro_seq': micro_seq,

            # ===== HTF VARIATIONS (2) =====
            'htf_070': htf_070,
            'htf_080': htf_080,

            # ===== DEBUG VALUES =====
            'htf_value': ts15,
            'vol_value': vol_ratio,
            'body_value': body_ratio,
            'body_sign': body_sign,
            'ema_dir_15m': ema_dir_15m,
            'vwap_dist_value': vwap_dist,
            'upper_wick_value': upper_wick,
            'lower_wick_value': lower_wick,
            'bb_width_value': bb_width,
            'q_score_value': q_score,
            'impulse_value': impulse
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
            and not getattr(p, 'was_executed', False)
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

        # Per-variable analysis (all 26+ variables)
        # Define all trackable variables
        all_variables = [
            # Original 4
            'htf', 'vol', 'body', 'align_15m',
            # Body variations
            'body_040', 'body_045', 'body_050', 'body_060', 'wick_align',
            # VWAP variations
            'vwap_045', 'vwap_060', 'vwap_080', 'vwap_100',
            # Volume variations
            'vol_110', 'vol_120', 'vol_150',
            # BB width
            'bb_width_60p', 'bb_width_70p', 'bb_width_80p',
            # Q-score
            'q_040', 'q_050', 'q_060', 'q_070',
            # Impulse
            'impulse_040', 'impulse_060',
            # Micro
            'micro_seq',
            # HTF variations
            'htf_070', 'htf_080',
        ]

        variable_stats = {}

        for var in all_variables:
            var_pass = [p for i, p in enumerate(phantoms) if gate_statuses[i].get(var, False)]
            var_fail = [p for i, p in enumerate(phantoms) if not gate_statuses[i].get(var, False)]

            pass_wins = sum(1 for p in var_pass if p.outcome == 'win')
            pass_total = len(var_pass)
            pass_wr = (pass_wins / pass_total * 100) if pass_total > 0 else 0.0

            fail_wins = sum(1 for p in var_fail if p.outcome == 'win')
            fail_total = len(var_fail)
            fail_wr = (fail_wins / fail_total * 100) if fail_total > 0 else 0.0

            delta = pass_wr - fail_wr if (pass_total >= min_samples and fail_total >= min_samples) else None

            variable_stats[var] = {
                'pass_wins': pass_wins,
                'pass_total': pass_total,
                'pass_wr': pass_wr,
                'fail_wins': fail_wins,
                'fail_total': fail_total,
                'fail_wr': fail_wr,
                'delta': delta,
                'sufficient_samples': pass_total >= min_samples and fail_total >= min_samples
            }

        # Sort variables by delta (descending) to show best filters first
        sorted_variables = sorted(
            [(k, v) for k, v in variable_stats.items() if v['sufficient_samples']],
            key=lambda x: x[1]['delta'] if x[1]['delta'] is not None else -999,
            reverse=True
        )

        # Backward compatibility: keep gate_stats for original 4
        gates = ['htf', 'vol', 'body', 'align_15m']
        gate_stats = {g: variable_stats[g] for g in gates if g in variable_stats}

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
            'gate_stats': gate_stats,  # Original 4 gates (backward compat)
            'variable_stats': variable_stats,  # All 26+ variables
            'sorted_variables': sorted_variables,  # Ranked by delta impact
            'top_combinations': top_combos,
            'min_samples': min_samples
        }

    def _filter_phantoms_by_month(self, month_str: str) -> List:
        """
        Filter completed phantoms by calendar month.

        Args:
            month_str: 'YYYY-MM' format (e.g., '2025-10')

        Returns:
            List of phantoms from that month with defined outcomes
        """
        try:
            from datetime import datetime
            year, month = month_str.split('-')
            year, month = int(year), int(month)

            filtered = []
            for arr in self.completed.values():
                for p in arr:
                    if p.exit_time and p.outcome in ('win', 'loss') and not getattr(p, 'was_executed', False):
                        if p.exit_time.year == year and p.exit_time.month == month:
                            filtered.append(p)

            return filtered
        except Exception:
            return []

    def _compute_combination_wr(self, phantoms: List, variables: List[str]) -> Dict:
        """
        Compute win rate for phantoms that pass ALL specified variables.

        Args:
            phantoms: List of ScalpPhantomTrade objects
            variables: List of variable names (e.g., ['body_050', 'vwap_045'])

        Returns:
            {'wins': int, 'total': int, 'wr': float, 'delta': float}
        """
        if not phantoms or not variables:
            return {'wins': 0, 'total': 0, 'wr': 0.0, 'delta': 0.0}

        # Compute gate status for each phantom
        gate_statuses = [self.compute_gate_status(p) for p in phantoms]

        # Filter phantoms where ALL variables pass
        passing_phantoms = []
        for i, p in enumerate(phantoms):
            gs = gate_statuses[i]
            if all(gs.get(var, False) for var in variables):
                passing_phantoms.append(p)

        # Calculate WR
        total = len(passing_phantoms)
        wins = sum(1 for p in passing_phantoms if p.outcome == 'win')
        wr = (wins / total * 100) if total > 0 else 0.0

        # Calculate delta vs baseline
        baseline_wins = sum(1 for p in phantoms if p.outcome == 'win')
        baseline_wr = (baseline_wins / len(phantoms) * 100) if len(phantoms) > 0 else 0.0
        delta = wr - baseline_wr

        return {'wins': wins, 'total': total, 'wr': wr, 'delta': delta}

    def get_comprehensive_analysis(self, month: str = None, top_n: int = 10, min_samples: int = 20) -> Dict:
        """
        Comprehensive variable analysis: all variables, combinations, recommendations.

        Args:
            month: 'YYYY-MM' for specific month, None for last 30 days
            top_n: Limit combination search to top N solo performers (default 10)
            min_samples: Minimum sample size for valid statistics (default 20)

        Returns:
            Dict with solo_analysis, pair_analysis, triplet_analysis, recommendations
        """
        from itertools import combinations
        from datetime import datetime, timedelta

        # Get phantoms for analysis period
        if month:
            phantoms = self._filter_phantoms_by_month(month)
            period_desc = month
        else:
            cutoff = datetime.utcnow() - timedelta(days=30)
            phantoms = [
                p for arr in self.completed.values() for p in arr
                if p.exit_time and p.exit_time >= cutoff and p.outcome in ('win', 'loss')
                and not getattr(p, 'was_executed', False)
            ]
            period_desc = "last 30 days"

        if not phantoms:
            return {'error': 'No phantom data available', 'period': period_desc, 'total': 0}

        total = len(phantoms)
        wins = sum(1 for p in phantoms if p.outcome == 'win')
        baseline_wr = (wins / total * 100) if total > 0 else 0.0

        # Define all trackable variables (50+)
        all_variables = [
            # Original 4 gates
            'htf', 'vol', 'body', 'align_15m',
            # Body variations
            'body_040', 'body_045', 'body_050', 'body_060', 'wick_align',
            # VWAP variations
            'vwap_045', 'vwap_060', 'vwap_080', 'vwap_100',
            # Volume variations
            'vol_110', 'vol_120', 'vol_150',
            # BB width
            'bb_width_60p', 'bb_width_70p', 'bb_width_80p',
            # Q-score
            'q_040', 'q_050', 'q_060', 'q_070',
            # Impulse
            'impulse_040', 'impulse_060',
            # Micro
            'micro_seq',
            # HTF variations
            'htf_070', 'htf_080',
        ]

        # Step 1: Analyze all variables individually
        gate_statuses = [self.compute_gate_status(p) for p in phantoms]
        variable_stats = {}

        for var in all_variables:
            var_pass = [p for i, p in enumerate(phantoms) if gate_statuses[i].get(var, False)]
            var_fail = [p for i, p in enumerate(phantoms) if not gate_statuses[i].get(var, False)]

            pass_wins = sum(1 for p in var_pass if p.outcome == 'win')
            pass_total = len(var_pass)
            pass_wr = (pass_wins / pass_total * 100) if pass_total > 0 else 0.0

            fail_wins = sum(1 for p in var_fail if p.outcome == 'win')
            fail_total = len(var_fail)
            fail_wr = (fail_wins / fail_total * 100) if fail_total > 0 else 0.0

            delta = pass_wr - baseline_wr
            sufficient = pass_total >= min_samples and fail_total >= min_samples

            # Impact score: delta weighted by sample size (for ranking)
            import math
            impact_score = delta * math.sqrt(pass_total) if pass_total > 0 else -999

            variable_stats[var] = {
                'pass_wins': pass_wins,
                'pass_total': pass_total,
                'pass_wr': pass_wr,
                'fail_wins': fail_wins,
                'fail_total': fail_total,
                'fail_wr': fail_wr,
                'delta': delta,
                'sufficient_samples': sufficient,
                'impact_score': impact_score
            }

        # Step 2: Rank variables and select top N for combination analysis
        ranked_vars = sorted(
            [(k, v) for k, v in variable_stats.items() if v['sufficient_samples']],
            key=lambda x: x[1]['impact_score'],
            reverse=True
        )
        top_variables = [var for var, stats in ranked_vars[:top_n]]

        # Step 3: Analyze pairs (top N → N choose 2 combinations)
        pair_stats = {}
        if len(top_variables) >= 2:
            for v1, v2 in combinations(top_variables, 2):
                result = self._compute_combination_wr(phantoms, [v1, v2])
                if result['total'] >= min_samples:
                    # Calculate synergy (combo WR vs average of individual WRs)
                    expected_wr = (variable_stats[v1]['pass_wr'] + variable_stats[v2]['pass_wr']) / 2
                    synergy = result['wr'] - expected_wr

                    pair_stats[(v1, v2)] = {
                        **result,
                        'expected_wr': expected_wr,
                        'synergy': synergy
                    }

        # Sort pairs by WR
        sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['wr'], reverse=True)

        # Step 4: Analyze triplets (top N → N choose 3 combinations)
        triplet_stats = {}
        if len(top_variables) >= 3:
            for v1, v2, v3 in combinations(top_variables, 3):
                result = self._compute_combination_wr(phantoms, [v1, v2, v3])
                if result['total'] >= max(min_samples // 2, 10):  # Lower threshold for triplets
                    triplet_stats[(v1, v2, v3)] = result

        # Sort triplets by WR
        sorted_triplets = sorted(triplet_stats.items(), key=lambda x: x[1]['wr'], reverse=True)

        return {
            'period': period_desc,
            'total_phantoms': total,
            'total_wins': wins,
            'baseline_wr': baseline_wr,
            'solo_analysis': variable_stats,
            'ranked_variables': ranked_vars,
            'top_variables': top_variables,
            'pair_analysis': dict(sorted_pairs[:20]),  # Top 20 pairs
            'triplet_analysis': dict(sorted_triplets[:10]),  # Top 10 triplets
            'min_samples': min_samples
        }

    def generate_recommendations(self, analysis: Dict, min_wr: float = 60.0, min_samples: int = 30) -> Dict:
        """
        Generate actionable recommendations from comprehensive analysis.

        Args:
            analysis: Output from get_comprehensive_analysis()
            min_wr: Minimum win rate to recommend enabling (default 60%)
            min_samples: Minimum sample size for recommendations (default 30)

        Returns:
            Dict with enable/disable lists, best combos, and config snippet
        """
        if analysis.get('error'):
            return {'error': analysis['error']}

        baseline_wr = analysis['baseline_wr']
        solo_stats = analysis['solo_analysis']
        pair_stats = analysis.get('pair_analysis', {})
        triplet_stats = analysis.get('triplet_analysis', {})

        recommendations = {
            'enable': [],
            'disable': [],
            'best_pairs': [],
            'best_triplets': [],
            'config_snippet': ""
        }

        # Enable: High WR, sufficient samples, strong positive delta
        for var, stats in solo_stats.items():
            if (stats['pass_wr'] >= min_wr and
                stats['pass_total'] >= min_samples and
                stats['delta'] > 5.0 and
                stats['sufficient_samples']):
                recommendations['enable'].append({
                    'variable': var,
                    'wr': stats['pass_wr'],
                    'delta': stats['delta'],
                    'count': stats['pass_total'],
                    'reason': f"{stats['pass_wr']:.1f}% WR ({stats['delta']:+.1f}%)"
                })

        # Disable: Low WR or strong negative delta
        for var, stats in solo_stats.items():
            if ((stats['pass_wr'] < baseline_wr - 3.0 or stats['delta'] < -5.0) and
                stats['pass_total'] >= 20 and
                stats['sufficient_samples']):
                recommendations['disable'].append({
                    'variable': var,
                    'wr': stats['pass_wr'],
                    'delta': stats['delta'],
                    'count': stats['pass_total'],
                    'reason': f"{stats['pass_wr']:.1f}% WR ({stats['delta']:+.1f}%)"
                })

        # Sort by delta impact
        recommendations['enable'] = sorted(recommendations['enable'], key=lambda x: x['delta'], reverse=True)
        recommendations['disable'] = sorted(recommendations['disable'], key=lambda x: x['delta'])

        # Best pairs
        for (v1, v2), stats in sorted(pair_stats.items(), key=lambda x: x[1]['wr'], reverse=True)[:5]:
            if stats['total'] >= min_samples and stats['wr'] >= min_wr:
                recommendations['best_pairs'].append({
                    'variables': [v1, v2],
                    'wr': stats['wr'],
                    'delta': stats['delta'],
                    'count': stats['total'],
                    'synergy': stats.get('synergy', 0)
                })

        # Best triplets
        for (v1, v2, v3), stats in sorted(triplet_stats.items(), key=lambda x: x[1]['wr'], reverse=True)[:3]:
            if stats['total'] >= max(min_samples // 2, 10) and stats['wr'] >= min_wr:
                recommendations['best_triplets'].append({
                    'variables': [v1, v2, v3],
                    'wr': stats['wr'],
                    'delta': stats['delta'],
                    'count': stats['total']
                })

        # Generate YAML config snippet
        config_lines = ["# Scalp Strategy - Recommended Configuration"]
        config_lines.append("# Based on phantom data analysis\n")
        config_lines.append("scalp:")
        config_lines.append("  hard_gates:")
        config_lines.append("    apply_to_exec: true")
        config_lines.append("    apply_to_phantoms: false\n")

        # Add top 5 enable recommendations as comments
        config_lines.append("    # ━━━ ENABLE (High Win Rate) ━━━")
        for rec in recommendations['enable'][:5]:
            config_lines.append(f"    # {rec['variable']}: {rec['reason']}")

        # Add top 3 disable recommendations as comments
        if recommendations['disable']:
            config_lines.append("\n    # ━━━ AVOID (Low Win Rate) ━━━")
            for rec in recommendations['disable'][:3]:
                config_lines.append(f"    # {rec['variable']}: {rec['reason']}")

        # Add best combination as comment
        if recommendations['best_pairs']:
            best = recommendations['best_pairs'][0]
            config_lines.append(f"\n    # 🎯 Best Combo: {' + '.join(best['variables'])}: {best['wr']:.1f}% WR")

        recommendations['config_snippet'] = '\n'.join(config_lines)

        return recommendations

    def get_monthly_trends(self, months: List[str] = None) -> Dict:
        """
        Analyze variable performance trends across multiple months.

        Args:
            months: List of 'YYYY-MM' strings (e.g., ['2025-10', '2025-09'])
                   If None, auto-generates last 3 months

        Returns:
            Dict with variable_trends and overall_trend summary
        """
        from datetime import datetime, timedelta

        # Auto-generate last 3 months if not specified
        if months is None:
            now = datetime.utcnow()
            months = []
            for i in range(3):
                dt = now - timedelta(days=30 * i)
                months.append(f"{dt.year}-{dt.month:02d}")

        if not months:
            return {'error': 'No months specified', 'months': []}

        # Get comprehensive analysis for each month
        monthly_analyses = {}
        for month in months:
            analysis = self.get_comprehensive_analysis(month=month, top_n=10, min_samples=15)
            if not analysis.get('error'):
                monthly_analyses[month] = analysis

        if not monthly_analyses:
            return {'error': 'No data available for specified months', 'months': months}

        # Track each variable across months
        variable_trends = {}

        # Get all variables from first month
        first_month = list(monthly_analyses.keys())[0]
        all_vars = monthly_analyses[first_month]['solo_analysis'].keys()

        for var in all_vars:
            var_monthly = {}
            wr_values = []
            delta_values = []
            count_values = []

            for month in sorted(monthly_analyses.keys()):
                stats = monthly_analyses[month]['solo_analysis'].get(var, {})
                if stats.get('sufficient_samples'):
                    var_monthly[month] = {
                        'wr': stats['pass_wr'],
                        'delta': stats['delta'],
                        'count': stats['pass_total']
                    }
                    wr_values.append(stats['pass_wr'])
                    delta_values.append(stats['delta'])
                    count_values.append(stats['pass_total'])

            # Calculate trend
            if len(wr_values) >= 2:
                wr_change = wr_values[-1] - wr_values[0]
                trend = 'improving' if wr_change > 2 else 'degrading' if wr_change < -2 else 'stable'
                avg_wr = sum(wr_values) / len(wr_values)
                avg_delta = sum(delta_values) / len(delta_values)

                variable_trends[var] = {
                    'monthly_data': var_monthly,
                    'trend': trend,
                    'wr_change': wr_change,
                    'avg_wr': avg_wr,
                    'avg_delta': avg_delta,
                    'months_tracked': len(wr_values)
                }

        # Overall trend summary
        improving = [v for v, t in variable_trends.items() if t['trend'] == 'improving']
        degrading = [v for v, t in variable_trends.items() if t['trend'] == 'degrading']
        stable = [v for v, t in variable_trends.items() if t['trend'] == 'stable']

        return {
            'months': sorted(monthly_analyses.keys()),
            'variable_trends': variable_trends,
            'summary': {
                'improving_count': len(improving),
                'degrading_count': len(degrading),
                'stable_count': len(stable),
                'improving_vars': improving[:10],
                'degrading_vars': degrading[:10]
            }
        }


_scalp_phantom_tracker = None

def get_scalp_phantom_tracker() -> ScalpPhantomTracker:
    global _scalp_phantom_tracker
    if _scalp_phantom_tracker is None:
        _scalp_phantom_tracker = ScalpPhantomTracker()
    return _scalp_phantom_tracker
