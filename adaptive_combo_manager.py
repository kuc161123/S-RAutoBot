"""
Adaptive Combo Manager

Dynamically enables/disables combos based on rolling performance metrics.
Analyzes phantom trade data to identify high-performing combos (WR ≥ threshold)
and automatically filters out underperforming patterns.

Usage:
    manager = AdaptiveComboManager(config, redis_client, phantom_tracker)
    manager.update_combo_filters()  # Recalculate active combos
    active_combos = manager.get_active_combos()  # Get filtered combo list
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ComboPerformance:
    """Performance metrics for a combo pattern"""
    combo_id: str
    side: str  # 'long' or 'short'
    wr: float  # Win rate percentage
    n: int  # Sample size (all trades)
    wins: int  # Number of wins (all trades)
    ev_r: float  # Expected value in R multiples
    last_updated: str  # ISO timestamp
    enabled: bool  # Whether combo is active
    # Extended breakdown counts for analytics (exec vs phantom, 30d + 24h)
    n_exec: int = 0
    n_phantom: int = 0
    n_24h: int = 0
    n_exec_24h: int = 0
    n_phantom_24h: int = 0


class AdaptiveComboManager:
    """Manages dynamic combo filtering based on performance"""

    def __init__(self, config: dict, redis_client=None, phantom_tracker=None, telegram_bot=None):
        """
        Initialize adaptive combo manager

        Args:
            config: Bot configuration dictionary
            redis_client: Redis client for persistence (optional)
            phantom_tracker: Phantom trade tracker instance
            telegram_bot: Telegram bot for sending notifications (optional)
        """
        self.config = config
        self.redis_client = redis_client
        self.phantom_tracker = phantom_tracker
        self.telegram_bot = telegram_bot

        # Load adaptive config with defaults
        adaptive_cfg = ((config.get('scalp', {}) or {}).get('exec', {}) or {}).get('adaptive_combos', {}) or {}
        self.enabled = bool(adaptive_cfg.get('enabled', True))
        self.min_wr_threshold = float(adaptive_cfg.get('min_wr_threshold', 45.0))
        self.min_sample_size = int(adaptive_cfg.get('min_sample_size', 20))
        self.lookback_days = int(adaptive_cfg.get('lookback_days', 30))
        self.long_short_separate = bool(adaptive_cfg.get('long_short_separate', True))
        self.hysteresis_pct = float(adaptive_cfg.get('hysteresis_pct', 2.0))  # Avoid flip-flopping
        self.notify_changes = bool(adaptive_cfg.get('notify_changes', True))
        # Robust gating options
        self.use_wilson_lb = bool(adaptive_cfg.get('use_wilson_lb', True))
        try:
            self.ev_floor_r = float(adaptive_cfg.get('ev_floor_r', 0.0))
        except Exception:
            self.ev_floor_r = 0.0
        # Strict side loads: when True, load per-side keys directly to avoid long/short overwrites
        self.strict_side_keys = bool(adaptive_cfg.get('strict_side_keys', True))

        # Track update stats
        self.last_update = None
        self.update_count = 0
        self.combo_changes = []  # List of recent enable/disable events

        logger.info(f"Adaptive Combo Manager initialized: enabled={self.enabled}, min_WR={self.min_wr_threshold}%, min_N={self.min_sample_size}")

    def _wilson_lb(self, wins: int, n: int, z: float = 1.96) -> float:
        """Wilson score lower bound (%). Returns 0..100."""
        try:
            if n <= 0:
                return 0.0
            p = wins / n
            denom = 1.0 + (z*z)/n
            center = (p + (z*z)/(2*n)) / denom
            import math as _m
            margin = z * _m.sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
            return max(0.0, min(1.0, center - margin)) * 100.0
        except Exception:
            return 0.0

    def _get_redis_key(self, side: Optional[str] = None) -> str:
        """Get Redis key for storing combo performance data"""
        if side and self.long_short_separate:
            return f"adaptive_combos:{side}"
        return "adaptive_combos:all"

    def _analyze_combo_performance(self, side: Optional[str] = None) -> Dict[str, ComboPerformance]:
        """
        Analyze phantom data to calculate combo performance metrics

        Args:
            side: 'long', 'short', or None for all

        Returns:
            Dictionary mapping combo_key → ComboPerformance
        """
        # Get phantom tracker - try stored instance first, then try importing
        scpt = self.phantom_tracker
        if not scpt:
            try:
                from scalp_phantom_tracker import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
            except Exception as e:
                logger.debug(f"Could not get scalp phantom tracker: {e}")

        if not scpt:
            logger.warning("No phantom tracker available for combo analysis")
            return {}

        now = datetime.utcnow()
        cutoff = now - timedelta(days=self.lookback_days)
        cutoff_24h = now - timedelta(days=1)

        # Collect decisive phantoms with required features (robust to string types)
        items = []  # (rsi, macd_hist, vwap, fib_zone, mtf, win, rr, side)
        def _to_float(x):
            try:
                if x is None:
                    return None
                # Allow numeric-like strings
                v = float(x)
                # Exclude NaN/inf
                import math as _m
                if _m.isnan(v) or _m.isinf(v):
                    return None
                return v
            except Exception:
                return None
        def _to_bool(x):
            try:
                if isinstance(x, bool):
                    return x
                if isinstance(x, (int,)):
                    return bool(x)
                s = str(x).strip().lower()
                if s in ('1','true','yes','y','on'):
                    return True
                if s in ('0','false','no','n','off'):
                    return False
                return None
            except Exception:
                return None
        _skipped_missing = 0
        _skipped_type = 0
        for arr in (getattr(scpt, 'completed', {}) or {}).values():
            for p in arr:
                try:
                    et = getattr(p, 'exit_time', None)
                    if not et or et < cutoff:
                        continue
                    if getattr(p, 'outcome', None) not in ('win', 'loss'):
                        continue

                    p_side = str(getattr(p, 'side', '')).lower()
                    if side and p_side != side:
                        continue
                    if not p_side:
                        continue

                    f = getattr(p, 'features', {}) or {}
                    rsi = _to_float(f.get('rsi_14'))
                    mh = _to_float(f.get('macd_hist'))
                    vwap = _to_float(f.get('vwap_dist_atr'))
                    fibz_raw = f.get('fib_zone')
                    mtf = _to_bool(f.get('mtf_agree_15'))

                    # Mandatory numerics
                    if rsi is None or mh is None or vwap is None:
                        _skipped_missing += 1
                        continue

                    # Derive mtf flag if missing (default to False/noMTF)
                    if mtf is None:
                        mtf = False

                    # Resolve Fib zone: use provided, else derive from fib_ret, else neutral fallback
                    fibz: Optional[str]
                    try:
                        if fibz_raw is not None:
                            fibz = str(fibz_raw)
                        else:
                            # Try derive from fib_ret (0..1 or 0..100)
                            frel = _to_float(f.get('fib_ret'))
                            if frel is not None:
                                fr = float(frel * 100.0) if frel <= 1.0 else float(frel)
                                if fr < 23.6:
                                    fibz = '0-23'
                                elif fr < 38.2:
                                    fibz = '23-38'
                                elif fr < 50.0:
                                    fibz = '38-50'
                                elif fr < 61.8:
                                    fibz = '50-61'
                                elif fr < 78.6:
                                    fibz = '61-78'
                                else:
                                    fibz = '78-100'
                            else:
                                # Neutral fallback to avoid dropping older records entirely
                                fibz = '38-50'
                    except Exception:
                        fibz = None

                    if fibz is None:
                        _skipped_type += 1
                        continue

                    rr = getattr(p, 'realized_rr', None)
                    rr = _to_float(rr)
                    rr = rr if rr is not None else 0.0
                    win = 1 if getattr(p, 'outcome', None) == 'win' else 0
                    is_exec = bool(getattr(p, 'was_executed', False))
                    is_recent = bool(et >= cutoff_24h) if isinstance(et, datetime) else False

                    items.append(
                        (
                            float(rsi),
                            float(mh),
                            float(vwap),
                            str(fibz),
                            bool(mtf),
                            win,
                            rr,
                            p_side,
                            is_exec,
                            is_recent,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error processing phantom for combo analysis: {e}")
                    continue

        if not items:
            logger.info(f"No phantom data for combo analysis (side={side}, {self.lookback_days}d)")
            return {}

        # Define binning logic (same as Pro Analytics)
        rsi_bins = [
            ("<30", lambda x: x < 30),
            ("30-40", lambda x: 30 <= x < 40),
            ("40-60", lambda x: 40 <= x < 60),
            ("60-70", lambda x: 60 <= x < 70),
            ("70+", lambda x: x >= 70)
        ]
        macd_bins = [("bull", lambda h: h > 0), ("bear", lambda h: h <= 0)]
        vwap_bins = [
            ("<0.6", lambda x: x < 0.6),
            ("0.6-1.2", lambda x: 0.6 <= x < 1.2),
            ("1.2+", lambda x: x >= 1.2)
        ]
        fib_bins = ["0-23", "23-38", "38-50", "50-61", "61-78", "78-100"]

        def lab(val, bins):
            for lb, fn in bins:
                if fn(val):
                    return lb
            return None

        # Aggregate by combo pattern
        combos = {}  # combo_key → aggregates
        for rsi, mh, vwap, fibz, mtf, win, rr, p_side, is_exec, is_recent in items:
            r = lab(rsi, rsi_bins)
            m = lab(mh, macd_bins)
            v = lab(vwap, vwap_bins)
            fz = fibz if fibz in fib_bins else None
            ma = 'MTF' if bool(mtf) else 'noMTF'

            if not all([r, m, v, fz, ma]):
                continue

            # Combo key format: "RSI:40-60 MACD:bull VWAP:1.2+ Fib:0-23 noMTF"
            key = f"RSI:{r} MACD:{m} VWAP:{v} Fib:{fz} {ma}"
            agg = combos.setdefault(
                key,
                {
                    'n': 0,
                    'w': 0,
                    'rr': 0.0,
                    'side': p_side,
                    'n_exec': 0,
                    'n_phantom': 0,
                    'n_24h': 0,
                    'n_exec_24h': 0,
                    'n_phantom_24h': 0,
                },
            )
            agg['n'] += 1
            agg['w'] += int(win)
            agg['rr'] += rr
            if is_exec:
                agg['n_exec'] += 1
                if is_recent:
                    agg['n_exec_24h'] += 1
            else:
                agg['n_phantom'] += 1
                if is_recent:
                    agg['n_phantom_24h'] += 1
            if is_recent:
                agg['n_24h'] += 1

        # Convert to ComboPerformance objects
        results = {}
        now_iso = now.isoformat()

        for key, agg in combos.items():
            n = agg['n']
            w = agg['w']
            wr = (w / n * 100.0) if n else 0.0
            ev_r = (agg['rr'] / n) if n else 0.0
            combo_side = agg['side']

            # Determine if combo should be enabled
            enabled = (wr >= self.min_wr_threshold and n >= self.min_sample_size)

            results[key] = ComboPerformance(
                combo_id=key,
                side=combo_side,
                wr=wr,
                n=n,
                wins=w,
                ev_r=ev_r,
                last_updated=now_iso,
                enabled=enabled,  # preliminary; final gating applied in update_combo_filters()
                n_exec=int(agg.get('n_exec', 0)),
                n_phantom=int(agg.get('n_phantom', 0)),
                n_24h=int(agg.get('n_24h', 0)),
                n_exec_24h=int(agg.get('n_exec_24h', 0)),
                n_phantom_24h=int(agg.get('n_phantom_24h', 0)),
            )

        logger.info(f"Analyzed {len(items)} phantoms → {len(results)} combos (side={side}, {self.lookback_days}d)")
        try:
            if (_skipped_missing + _skipped_type) > 0:
                logger.info(f"Combo analysis skipped: missing={_skipped_missing} type_fail={_skipped_type} (side={side})")
        except Exception:
            pass
        return results

    def update_combo_filters(self, force: bool = False) -> Tuple[int, int, List[str]]:
        """
        Update combo filters based on current performance

        Args:
            force: Force update even if recently updated

        Returns:
            Tuple of (enabled_count, disabled_count, change_messages)
        """
        if not self.enabled:
            logger.debug("Adaptive combo manager disabled, skipping update")
            return (0, 0, [])

        try:
            # Analyze performance for longs and shorts
            all_combos = {}

            if self.long_short_separate:
                long_combos = self._analyze_combo_performance(side='long')
                short_combos = self._analyze_combo_performance(side='short')
                all_combos.update(long_combos)
                all_combos.update(short_combos)
            else:
                all_combos = self._analyze_combo_performance(side=None)

            if not all_combos:
                logger.warning("No combo performance data available for filtering")
                return (0, 0, [])

            # Load previous state from Redis
            prev_state = self._load_combo_state()

            # Detect changes and apply hysteresis
            changes = []
            enabled_count = 0
            disabled_count = 0

            for key, perf in all_combos.items():
                prev_enabled = prev_state.get(key, {}).get('enabled', False)
                # Compute WR metric (Wilson LB when enabled)
                wr_metric = self._wilson_lb(perf.wins, perf.n) if self.use_wilson_lb else float(perf.wr)
                n_ok = perf.n >= self.min_sample_size
                ev_ok = float(perf.ev_r) >= float(self.ev_floor_r)
                thr = float(self.min_wr_threshold)

                # Base gating (no hysteresis)
                base_enabled = bool(n_ok and ev_ok and (wr_metric >= thr))
                curr_enabled = base_enabled

                # Apply hysteresis around WR metric only (EV and N are hard gates)
                if prev_enabled and not curr_enabled:
                    # Only WR margin may keep it enabled; EV/N failures should disable
                    if ev_ok and n_ok and wr_metric >= (thr - self.hysteresis_pct):
                        curr_enabled = True
                elif (not prev_enabled) and curr_enabled:
                    # Require WR exceed threshold + hysteresis to enable
                    if wr_metric < (thr + self.hysteresis_pct):
                        curr_enabled = False

                # Track state changes
                if curr_enabled != prev_enabled:
                    status = "ENABLED" if curr_enabled else "DISABLED"
                    try:
                        msg = (
                            f"{status}: {key} ({perf.side.upper()}) - "
                            f"WR {perf.wr:.1f}% (LB {wr_metric:.1f}%) EV_R {perf.ev_r:+.2f} N={perf.n}"
                        )
                    except Exception:
                        msg = f"{status}: {key} ({perf.side.upper()}) - WR {perf.wr:.1f}% (N={perf.n})"
                    changes.append(msg)
                    logger.info(f"Combo state change: {msg}")

                if curr_enabled:
                    perf.enabled = True
                    enabled_count += 1
                else:
                    perf.enabled = False
                    disabled_count += 1

            # Save updated state to Redis
            self._save_combo_state(all_combos)

            # Update tracking
            self.last_update = datetime.utcnow()
            self.update_count += 1
            self.combo_changes.extend(changes)
            self.combo_changes = self.combo_changes[-50:]  # Keep last 50 changes

            logger.info(f"Combo filter update #{self.update_count}: {enabled_count} enabled, {disabled_count} disabled, {len(changes)} changes")

            # Send Telegram notifications for combo changes
            if changes and self.notify_changes and self.telegram_bot:
                try:
                    import asyncio
                    for change_msg in changes:
                        # Determine emoji and reason
                        if "ENABLED" in change_msg:
                            emoji = "🟢"
                            reason = f"WR above threshold ({self.min_wr_threshold:.1f}%) with sufficient data"
                        else:
                            emoji = "🔴"
                            reason = f"WR below threshold ({self.min_wr_threshold:.1f}%)"

                        notification = f"{emoji} **Combo Filter Update**\n{change_msg}\nReason: {reason}\nThreshold: WR ≥{self.min_wr_threshold:.1f}%, N ≥{self.min_sample_size}"
                        asyncio.create_task(self.telegram_bot.send_message(notification))
                except Exception as notif_err:
                    logger.debug(f"Failed to send combo change notification: {notif_err}")

            return (enabled_count, disabled_count, changes)

        except Exception as e:
            logger.error(f"Error updating combo filters: {e}", exc_info=True)
            return (0, 0, [])

    def _load_combo_state(self, side: Optional[str] = None) -> Dict[str, dict]:
        """Load previous combo state from Redis.

        When side is provided and strict_side_keys is True, loads only that side's
        key. When side is None, returns a combined view (legacy behavior) where
        identical combo_ids across sides may overwrite each other.
        """
        if not self.redis_client:
            return {}

        try:
            # Strict per-side load
            if self.long_short_separate and self.strict_side_keys and side in ('long', 'short'):
                key = self._get_redis_key(side)
                data = self.redis_client.get(key)
                return json.loads(data) if data else {}

            # Legacy combined behavior
            result: Dict[str, dict] = {}
            if self.long_short_separate:
                for s in ['long', 'short']:
                    key = self._get_redis_key(s)
                    data = self.redis_client.get(key)
                    if data:
                        side_combos = json.loads(data)
                        result.update(side_combos)
            else:
                key = self._get_redis_key()
                data = self.redis_client.get(key)
                if data:
                    result = json.loads(data)
            return result
        except Exception as e:
            logger.warning(f"Failed to load combo state from Redis: {e}")
            return {}

    def _save_combo_state(self, combos: Dict[str, ComboPerformance]):
        """Save combo state to Redis"""
        if not self.redis_client:
            return

        try:
            # Convert to dict for JSON serialization
            state = {key: asdict(perf) for key, perf in combos.items()}

            if self.long_short_separate:
                # Separate by side
                long_state = {k: v for k, v in state.items() if v['side'] == 'long'}
                short_state = {k: v for k, v in state.items() if v['side'] == 'short'}

                if long_state:
                    self.redis_client.setex(self._get_redis_key('long'), 86400 * 7, json.dumps(long_state))
                if short_state:
                    self.redis_client.setex(self._get_redis_key('short'), 86400 * 7, json.dumps(short_state))
            else:
                self.redis_client.setex(self._get_redis_key(), 86400 * 7, json.dumps(state))

            logger.debug(f"Saved {len(combos)} combo states to Redis")
        except Exception as e:
            logger.warning(f"Failed to save combo state to Redis: {e}")

    def get_active_combos(self, side: Optional[str] = None) -> List[Dict]:
        """
        Get list of currently active (enabled) combos

        Args:
            side: Filter by 'long' or 'short', or None for all

        Returns:
            List of active combo dictionaries with performance metrics
        """
        # Load side-specific state when requested to avoid any cross-side overwrites
        state = self._load_combo_state(side)

        active = []
        for key, data in state.items():
            if not data.get('enabled', False):
                continue
            if side and data.get('side') != side:
                continue

            # Convert combo key back to constraints for matching
            # This would need to parse "RSI:40-60 MACD:bull VWAP:1.2+ Fib:0-23 noMTF"
            # and map to numeric ranges. For now, return metadata only.
            active.append({
                'combo_id': key,
                'side': data.get('side'),
                'wr': data.get('wr', 0.0),
                'n': data.get('n', 0),
                'ev_r': data.get('ev_r', 0.0),
                'last_updated': data.get('last_updated', ''),
                'n_exec': data.get('n_exec', 0),
                'n_phantom': data.get('n_phantom', 0),
                'n_24h': data.get('n_24h', 0),
                'n_exec_24h': data.get('n_exec_24h', 0),
                'n_phantom_24h': data.get('n_phantom_24h', 0),
            })

        return active

    def is_combo_enabled(self, combo_pattern: dict) -> bool:
        """
        Check if a specific combo pattern is currently enabled

        Args:
            combo_pattern: Dictionary with combo features (RSI, MACD, VWAP, Fib, MTF)

        Returns:
            True if combo is enabled, False otherwise
        """
        # Build combo key from pattern
        try:
            # This would need proper binning logic
            # For now, check if any active combo matches
            state = self._load_combo_state()

            # Simplified: check if pattern's combo_id exists and is enabled
            combo_id = combo_pattern.get('combo_id')
            if combo_id and combo_id in state:
                return state[combo_id].get('enabled', False)

            return True  # Default to enabled if not found (fail open)
        except Exception as e:
            logger.debug(f"Error checking combo enabled status: {e}")
            return True  # Fail open

    def get_stats_summary(self) -> dict:
        """Get summary statistics for Telegram display"""
        if self.long_short_separate and self.strict_side_keys:
            long_state = self._load_combo_state('long')
            short_state = self._load_combo_state('short')
            long_enabled = sum(1 for v in long_state.values() if v.get('enabled'))
            long_disabled = sum(1 for v in long_state.values() if not v.get('enabled'))
            short_enabled = sum(1 for v in short_state.values() if v.get('enabled'))
            short_disabled = sum(1 for v in short_state.values() if not v.get('enabled'))
            total = len(long_state) + len(short_state)
            state_all = {}
            state_all.update(long_state or {})
            state_all.update(short_state or {})
        else:
            state = self._load_combo_state()
            long_enabled = sum(1 for v in state.values() if v.get('enabled') and v.get('side') == 'long')
            long_disabled = sum(1 for v in state.values() if not v.get('enabled') and v.get('side') == 'long')
            short_enabled = sum(1 for v in state.values() if v.get('enabled') and v.get('side') == 'short')
            short_disabled = sum(1 for v in state.values() if not v.get('enabled') and v.get('side') == 'short')
            total = len(state)
            state_all = state

        def _agg_side(state_dict: Dict[str, dict], side_label: str) -> dict:
            totals = {'n': 0, 'n_exec': 0, 'n_phantom': 0, 'n_24h': 0, 'n_exec_24h': 0, 'n_phantom_24h': 0}
            for v in (state_dict or {}).values():
                try:
                    if side_label and v.get('side') != side_label:
                        continue
                    totals['n'] += int(v.get('n', 0) or 0)
                    totals['n_exec'] += int(v.get('n_exec', 0) or 0)
                    totals['n_phantom'] += int(v.get('n_phantom', 0) or 0)
                    totals['n_24h'] += int(v.get('n_24h', 0) or 0)
                    totals['n_exec_24h'] += int(v.get('n_exec_24h', 0) or 0)
                    totals['n_phantom_24h'] += int(v.get('n_phantom_24h', 0) or 0)
                except Exception:
                    continue
            return totals

        long_totals = _agg_side(state_all, 'long')
        short_totals = _agg_side(state_all, 'short')

        return {
            'enabled': self.enabled,
            'min_wr_threshold': self.min_wr_threshold,
            'min_sample_size': self.min_sample_size,
            'lookback_days': self.lookback_days,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_count': self.update_count,
            'long_enabled': long_enabled,
            'long_disabled': long_disabled,
            'short_enabled': short_enabled,
            'short_disabled': short_disabled,
            'total_combos': total,
            'recent_changes': self.combo_changes[-10:],  # Last 10 changes
            'long_totals': long_totals,
            'short_totals': short_totals,
        }
