from __future__ import annotations
from datetime import datetime, timezone

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import UpdateType, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
try:
    # Use HTTPXRequest to tune timeouts and reduce httpx.ReadError during polling
    from telegram.request import HTTPXRequest
except Exception:
    HTTPXRequest = None
import telegram.error
import asyncio
import logging

# Disabled ML scorer imports removed (Trend, MR strategies disabled)
# Provide stubs to avoid NameError in legacy handlers; they will simply return None.
def get_trend_scorer():
    return None

def get_mean_reversion_scorer():
    return None

from autobot.utils.regime import get_enhanced_market_regime, get_regime_summary

logger = logging.getLogger(__name__)

class TGBot:
    def __init__(self, token:str, chat_id:int, shared:dict):
        # shared contains {"risk": RiskConfig, "book": Book, "panic": list, "meta": dict}
        # Build Application with a resilient HTTPX client when available
        if HTTPXRequest is not None:
            try:
                request = HTTPXRequest(
                    connect_timeout=10.0,
                    read_timeout=120.0,
                    write_timeout=45.0,
                    pool_timeout=60.0,  # Increased from 30s to 60s
                    connection_pool_size=50  # Increased from default 10 to 50 for high-throughput bot
                )
                self.app = Application.builder().token(token).request(request).build()
            except Exception:
                self.app = Application.builder().token(token).build()
        else:
            self.app = Application.builder().token(token).build()
        self.chat_id = chat_id
        self.shared = shared
        # Running flag early to avoid attribute errors if start_polling is called immediately
        self.running = False
        # Concurrent message tracking for flood detection
        self._concurrent_sends = 0
        self._max_concurrent_seen = 0
        # Downgrade noisy PTB warning about CancelledError during graceful shutdown
        try:
            # Reduce chatty PTB logs; Updater network loop errors will auto-retry
            logging.getLogger("telegram").setLevel(logging.WARNING)
            logging.getLogger("telegram.ext").setLevel(logging.WARNING)
            logging.getLogger("telegram.ext.Application").setLevel(logging.WARNING)
            logging.getLogger("telegram.ext.Updater").setLevel(logging.WARNING)
        except Exception:
            pass
        # Simple per-command cooldown
        self._cooldowns = {}
        self._cooldown_seconds = 2.0
        # Simple UI conversation state per chat for numeric settings
        self._ui_state: dict[int, dict] = {}
        
        # Add command handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("risk", self.show_risk))
        self.app.add_handler(CommandHandler("set_risk", self.set_risk))
        self.app.add_handler(CommandHandler("setrisk", self.set_risk))  # Alternative command name
        self.app.add_handler(CommandHandler("risk_percent", self.set_risk_percent))
        self.app.add_handler(CommandHandler("riskpercent", self.set_risk_percent))  # Alternative command name
        self.app.add_handler(CommandHandler("risk_usd", self.set_risk_usd))
        self.app.add_handler(CommandHandler("riskusd", self.set_risk_usd))  # Alternative command name
        self.app.add_handler(CommandHandler("ml_risk", self.ml_risk))
        self.app.add_handler(CommandHandler("ml_risk_range", self.ml_risk_range))
        self.app.add_handler(CommandHandler("mlriskrange", self.ml_risk_range))  # Alternative command name
        self.app.add_handler(CommandHandler("mlriskrank", self.ml_risk_range))  # Alternative command name
        self.app.add_handler(CommandHandler("ml90_risk", self.ml90_risk))
        self.app.add_handler(CommandHandler("scalp_set_risk_percent", self.scalp_set_risk_percent))
        self.app.add_handler(CommandHandler("scalp_get_risk", self.scalp_get_risk))
        # Manual A-tier combo risk controls
        self.app.add_handler(CommandHandler("manualcombo_risk", self.manual_combo_risk))
        self.app.add_handler(CommandHandler("manualcombo_riskusd", self.manual_combo_risk_usd))
        self.app.add_handler(CommandHandler("scalp_ml_recommend", self.scalp_ml_recommend))
        # Combos-only controls
        self.app.add_handler(CommandHandler("scalpcombos", self.scalp_combos_toggle))
        self.app.add_handler(CommandHandler("scalpcombosmute", self.scalp_combos_mute))
        # Adaptive combo management commands
        self.app.add_handler(CommandHandler("combo_status", self.combo_status_cmd))
        self.app.add_handler(CommandHandler("combostatus", self.combo_status_cmd))
        self.app.add_handler(CommandHandler("combos", self.combo_status_cmd))
        self.app.add_handler(CommandHandler("combo_refresh", self.combo_refresh_cmd))
        self.app.add_handler(CommandHandler("comborefresh", self.combo_refresh_cmd))
        self.app.add_handler(CommandHandler("combo_threshold", self.combo_threshold_cmd))
        self.app.add_handler(CommandHandler("combothreshold", self.combo_threshold_cmd))
        self.app.add_handler(CommandHandler("force_adaptive", self.force_adaptive_combos))
        self.app.add_handler(CommandHandler("forceadaptive", self.force_adaptive_combos))
        self.app.add_handler(CommandHandler("status", self.status))
        # Simple responsiveness probe
        self.app.add_handler(CommandHandler("ping", self.ping))
        self.app.add_handler(CommandHandler("panic_close", self.panic_close))
        self.app.add_handler(CommandHandler("balance", self.balance))
        self.app.add_handler(CommandHandler("health", self.health))
        self.app.add_handler(CommandHandler("symbols", self.symbols))
        self.app.add_handler(CommandHandler("dashboard", self.dashboard))
        self.app.add_handler(CommandHandler("watchlist", self.watchlist))
        self.app.add_handler(CommandHandler("analysis", self.analysis))
        self.app.add_handler(CommandHandler("stats", self.stats))
        self.app.add_handler(CommandHandler("recent", self.recent_trades))
        self.app.add_handler(CommandHandler("ml", self.ml_stats))
        self.app.add_handler(CommandHandler("ml_stats", self.ml_stats))
        self.app.add_handler(CommandHandler("mlrankings", self.ml_rankings))
        self.app.add_handler(CommandHandler("mlpatterns", self.ml_patterns))
        self.app.add_handler(CommandHandler("mlretrain", self.ml_retrain_info))
        self.app.add_handler(CommandHandler("reset_stats", self.reset_stats))
        self.app.add_handler(CommandHandler("phantom", self.phantom_stats))
        self.app.add_handler(CommandHandler("phantom_detail", self.phantom_detail))
        self.app.add_handler(CommandHandler("evolution", self.evolution_performance))
        self.app.add_handler(CommandHandler("force_retrain", self.force_retrain_ml))
        self.app.add_handler(CommandHandler("clusters", self.cluster_status))
        self.app.add_handler(CommandHandler("update_clusters", self.update_clusters))
        self.app.add_handler(CommandHandler("set_ml_threshold", self.set_ml_threshold))
        # MR commands removed (strategy disabled)
        self.app.add_handler(CommandHandler("parallel_performance", self.parallel_performance))
        self.app.add_handler(CommandHandler("parallelperformance", self.parallel_performance))  # Alternative command name
        # Trend pullback state snapshot
        # Trend commands removed (strategy disabled)
        self.app.add_handler(CommandHandler("regime_analysis", self.regime_analysis))
        self.app.add_handler(CommandHandler("regimeanalysis", self.regime_analysis))  # Alternative command name
        self.app.add_handler(CommandHandler("regime", self.regime_single))
        self.app.add_handler(CommandHandler("strategy_comparison", self.strategy_comparison))
        self.app.add_handler(CommandHandler("strategycomparison", self.strategy_comparison))  # Alternative command name
        self.app.add_handler(CommandHandler("system", self.system_status))
        self.app.add_handler(CommandHandler("telemetry", self.telemetry))
        self.app.add_handler(CommandHandler("training_status", self.training_status))
        self.app.add_handler(CommandHandler("trainingstatus", self.training_status))  # Alternative command name
        self.app.add_handler(CommandHandler("phantomqa", self.phantom_qa))
        self.app.add_handler(CommandHandler("scalpqa", self.scalp_qa))
        self.app.add_handler(CommandHandler("scalpgates", self.scalp_gate_analysis))
        self.app.add_handler(CommandHandler("scalpcomprehensive", self.scalp_comprehensive_analysis))
        self.app.add_handler(CommandHandler("scalpultimate", self.scalp_ultimate))
        self.app.add_handler(CommandHandler("scalprecommend", self.scalp_recommendations))
        # Adaptive risk controls
        self.app.add_handler(CommandHandler("scalpriskstate", self.scalp_risk_state))
        self.app.add_handler(CommandHandler("scalprisklimits", self.scalp_risk_limits))
        self.app.add_handler(CommandHandler("scalpriskladder", self.scalp_risk_ladder))
        self.app.add_handler(CommandHandler("scalptrends", self.scalp_monthly_trends))
        self.app.add_handler(CommandHandler("scalppromote", self.scalp_promotion_status))
        # Utility to verify chat/user IDs
        self.app.add_handler(CommandHandler("whoami", self.whoami))
        # Scalp gate risk adjustments
        self.app.add_handler(CommandHandler("scalpgaterisk", self.scalp_gate_risk))
        self.app.add_handler(CommandHandler("scalprisk", self.scalp_gate_risk))  # alias
        # Stream/WS status debug helpers
        self.app.add_handler(CommandHandler("stream", self.stream_status))
        self.app.add_handler(CommandHandler("wsstatus", self.stream_status))  # alias
        self.app.add_handler(CommandHandler("scalpreset", self.scalp_reset))
        self.app.add_handler(CommandHandler("scalppatterns", self.scalp_patterns))
        self.app.add_handler(CommandHandler("scalpqwr", self.scalp_qscore_wr))
        self.app.add_handler(CommandHandler("scalpmlwr", self.scalp_mlscore_wr))
        self.app.add_handler(CommandHandler("scalptimewr", self.scalp_time_wr))
        self.app.add_handler(CommandHandler("scalpoffhours", self.scalp_offhours_toggle))
        self.app.add_handler(CommandHandler("scalpoffhourswindow", self.scalp_offhours_window))
        self.app.add_handler(CommandHandler("scalpoffhoursexception", self.scalp_offhours_exception))
        # Vars-by-time command (implemented below); keep registration here
        self.app.add_handler(CommandHandler("scalptimewrvars", self.scalp_time_vars_cmd))
        # Trend promotion and high ML commands removed
        self.app.add_handler(CallbackQueryHandler(self.ui_callback, pattern=r"^ui:"))
        # Capture numeric input replies when a settings prompt is active
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_text))
        # Last-resort debug handler to observe any incoming updates (kept lightweight)
        self.app.add_handler(MessageHandler(filters.ALL, self._debug_update), group=99)
        self.app.add_handler(CommandHandler("mlstatus", self.ml_stats))
        self.app.add_handler(CommandHandler("panicclose", self.panic_close))
        self.app.add_handler(CommandHandler("forceretrain", self.force_retrain_ml))
        self.app.add_handler(CommandHandler("shadowstats", self.shadow_stats))
        self.app.add_handler(CommandHandler("flowdebug", self.flow_debug))
        self.app.add_handler(CommandHandler("flowstatus", self.flow_debug))
        # High-ML threshold controls per strategy (with and without underscores)
        self.app.add_handler(CommandHandler("scalp_highml", self.set_scalp_highml))
        # MR high ML setter removed
        # Trend high ML setter removed
        # Global error handler
        self.app.add_error_handler(self._on_error)

    def _session_label(self) -> str:
        try:
            hr = datetime.utcnow().hour
            return 'asian' if 0 <= hr < 8 else ('european' if hr < 16 else 'us')
        except Exception:
            return 'us'
        # Aliases without underscore as requested
        self.app.add_handler(CommandHandler("scalphighml", self.set_scalp_highml))
        # MR alias removed
        # Alias removed
        # Qscore threshold adjustments
        self.app.add_handler(CommandHandler("set_qscore", self.set_qscore))

    def _cooldown_ok(self, name: str) -> bool:
        """Return True if command is not rate-limited; otherwise False."""
        try:
            import time
            now = time.time()
            last = self._cooldowns.get(name, 0.0)
            if (now - last) < self._cooldown_seconds:
                return False
            self._cooldowns[name] = now
            return True
        except Exception:
            return True

    def _compute_risk_snapshot(self):
        """Return (per_trade_risk_usd, label) based on current configuration."""
        risk = self.shared.get("risk")
        last_balance = self.shared.get("last_balance")

        per_trade = float(getattr(risk, 'risk_usd', 0.0))
        label: str

        if getattr(risk, 'use_percent_risk', False) and last_balance:
            per_trade = last_balance * (getattr(risk, 'risk_percent', 0.0) / 100.0)
            label = f"{risk.risk_percent}% (~${per_trade:.2f})"
        elif getattr(risk, 'use_percent_risk', False):
            per_trade = float(getattr(risk, 'risk_usd', 0.0))
            label = f"{risk.risk_percent}% (fallback ${per_trade:.2f})"
        else:
            label = f"${per_trade:.2f}"

        if getattr(risk, 'use_ml_dynamic_risk', False):
            label += " (ML dynamic)"

        return per_trade, label

    # ===== Helpers: Adaptive Risk / Recency / Sparklines =====
    def _wilson_lb(self, wins:int, n:int, z:float=1.96) -> float:
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

    def _sparkline(self, vals:list, *, zero_to_hundred:bool=False) -> str:
        try:
            chars = '▁▂▃▄▅▆▇█'
            if not vals:
                return ''
            # Normalize range
            if zero_to_hundred:
                mn, mx = 0.0, 100.0
            else:
                mn, mx = float(min(vals)), float(max(vals))
                if mx <= mn:
                    mx = mn + 1.0
            rng = mx - mn
            out = []
            for v in vals:
                try:
                    x = (float(v) - mn) / rng
                    idx = max(0, min(7, int(round(x * 7))))
                except Exception:
                    idx = 0
                out.append('▁▂▃▄▅▆▇█'[idx])
            return ''.join(out)
        except Exception:
            return ''

    def _scalp_combo_recency(self, combo_key:str, side:str, days:int=7):
        """Return 7d recency stats for a combo: {n,w,rr_sum, days:[{n,w,wr_lb}]}
        Uses Redis aggregates from scalp_phantom_tracker: combos:scalp:{n,w,rr}:{YYYYMMDD}:{side}
        """
        out = { 'n': 0, 'w': 0, 'rr_sum': 0.0, 'days': [] }
        try:
            # Prefer Redis from phantom tracker if available
            r = None
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                r = getattr(scpt, 'redis_client', None)
            except Exception:
                r = None
            if r is None:
                try:
                    import os, redis
                    url = os.getenv('REDIS_URL')
                    if url:
                        r = redis.from_url(url, decode_responses=True)
                except Exception:
                    r = None
            if r is None:
                return out
            from datetime import datetime as _dt, timedelta as _td
            today = _dt.utcnow().date()
            series = []
            total_n = total_w = 0
            rr_sum = 0.0
            sside = str(side).lower()
            # Oldest→Newest for nice sparkline ordering
            for i in range(max(0, days)):
                dstr = (today - _td(days=(days-1-i))).strftime('%Y%m%d')
                n_key = f"combos:scalp:n:{dstr}:{sside}"
                w_key = f"combos:scalp:w:{dstr}:{sside}"
                rr_key = f"combos:scalp:rr:{dstr}:{sside}"
                try:
                    n_i = int(r.hget(n_key, combo_key) or 0)
                    w_i = int(r.hget(w_key, combo_key) or 0)
                    rr_i = float(r.hget(rr_key, combo_key) or 0.0)
                except Exception:
                    n_i = 0; w_i = 0; rr_i = 0.0
                total_n += n_i
                total_w += w_i
                rr_sum += rr_i
                wr_lb_i = self._wilson_lb(w_i, n_i) if n_i > 0 else 0.0
                series.append({'n': n_i, 'w': w_i, 'wr_lb': wr_lb_i})
            out['n'] = total_n; out['w'] = total_w; out['rr_sum'] = rr_sum; out['days'] = series
            return out
        except Exception:
            return out

    def _build_trend_dashboard(self):
        """Build Trend‑only dashboard text and keyboard."""
        frames = self.shared.get("frames", {})
        book = self.shared.get("book")
        last_analysis = self.shared.get("last_analysis", {})
        per_trade_risk, _risk_label = self._compute_risk_snapshot()

        lines = ["🎯 *Trend Pullback Dashboard*", "━━━━━━━━━━━━━━━━━━━━━", ""]

        # System
        lines.append("⚡ *System*")
        lines.append(f"• Status: {'✅ Online' if frames else '⏳ Starting up'}")
        timeframe = self.shared.get("timeframe")
        if timeframe:
            lines.append(f"• Timeframe: {timeframe}m + 3m")
        symbols_cfg = self.shared.get("symbols_config")
        if symbols_cfg:
            lines.append(f"• Universe: {len(symbols_cfg)} symbols")
        if last_analysis:
            try:
                latest_symbol, latest_time = max(last_analysis.items(), key=lambda kv: kv[1])
                if isinstance(latest_time, datetime):
                    ref_now = datetime.now(latest_time.tzinfo) if latest_time.tzinfo else datetime.now()
                    age_minutes = max(0, int((ref_now - latest_time).total_seconds() // 60))
                    lines.append(f"• Last scan: {latest_symbol} ({age_minutes}m ago)")
            except Exception:
                pass
        # Balance
        broker = self.shared.get("broker")
        balance = self.shared.get("last_balance")
        try:
            if broker and (balance is None or float(balance) <= 0.0):
                bal = broker.get_balance()
                if bal is not None:
                    balance = bal
                    self.shared["last_balance"] = bal
        except Exception:
            pass
        if balance is not None:
            lines.append(f"• Balance: ${float(balance):.2f} USDT")

        # Trend States summary
        lines.append("")
        lines.append("📐 *Trend States*")
        try:
            from strategy_pullback import get_trend_states_snapshot
            snap = get_trend_states_snapshot() or {}
            counts = {"NEUTRAL":0, "RESISTANCE_BROKEN":0, "SUPPORT_BROKEN":0, "HL_FORMED":0, "LH_FORMED":0}
            exec_count = 0
            bos_armed = 0
            for st in snap.values():
                s = (st.get('state') if isinstance(st, dict) else None) or 'NEUTRAL'
                if s in counts:
                    counts[s] += 1
                try:
                    if st.get('bos_crossed'):
                        bos_armed += 1
                except Exception:
                    pass
                try:
                    if bool(st.get('executed', False)):
                        exec_count += 1
                except Exception:
                    pass
            # Include BOS armed and protective pivot counts in summary
            lines.append(
                f"NEUTRAL {counts['NEUTRAL']} | RES {counts['RESISTANCE_BROKEN']} | SUP {counts['SUPPORT_BROKEN']} | BOS {bos_armed} | HL {counts['HL_FORMED']} | LH {counts['LH_FORMED']} | EXEC {exec_count}"
            )
        except Exception:
            lines.append("(unavailable)")

        # Range States summary
        try:
            rs = self.shared.get('range_states') or {}
            if rs:
                lines.append("")
                lines.append("📦 *Range States*")
                lines.append(
                    f"IN_RANGE {int(rs.get('in_range',0))} | NEAR_EDGE {int(rs.get('near_edge',0))} | EXEC {int(rs.get('exec_today',0))} | PHANTOM {int(rs.get('phantom_open',0))} | TP1(mid) {int(rs.get('tp1_mid_hits_today',0))}"
                )
            else:
                lines.append("")
                lines.append("📦 *Range States*")
                lines.append("(unavailable)")
        except Exception:
            pass

        # Scalp States summary
        try:
            ss = self.shared.get('scalp_states') or {}
            if ss:
                lines.append("")
                lines.append("🩳 *Scalp States*")
                lines.append(
                    f"MOM {int(ss.get('mom',0))} | PULL {int(ss.get('pull',0))} | VWAP {int(ss.get('vwap',0))} | Q≥thr {int(ss.get('q_ge_thr',0))} | EXEC {int(ss.get('exec_today',0))} | PHANTOM {int(ss.get('phantom_open',0))}"
                )
            else:
                lines.append("")
                lines.append("🩳 *Scalp States*")
                lines.append("(unavailable)")
        except Exception:
            pass

        # Rule‑Mode summary
        try:
            cfg = self.shared.get('config', {}) or {}
            rm = (((cfg.get('trend', {}) or {}).get('rule_mode', {}) or {}))
            enabled = bool(rm.get('enabled', False))
            exec_min = float(rm.get('execute_q_min', 78))
            ph_min = float(rm.get('phantom_q_min', 65))
            extreme = bool(((rm.get('safety', {}) or {}).get('extreme_vol_block', True)))
            # ML maturity
            ml = self.shared.get('trend_scorer')
            info = ml.get_retrain_info() if ml else {}
            t = int(info.get('total_records', 0)); e = int(info.get('executed_count', 0))
            rec_need = int(((rm.get('ml_influence', {}) or {}).get('min_records', 2000)))
            exe_need = int(((rm.get('ml_influence', {}) or {}).get('min_executed', 400)))
            matured = (t >= rec_need) and (e >= exe_need)
            lines.append("")
            lines.append("🧭 *Rule‑Mode*")
            lines.append(f"• Enabled: {'On' if enabled else 'Off'} | Exec≥{exec_min:.0f} | Phantom≥{ph_min:.0f} | Extreme‑vol block: {'On' if extreme else 'Off'}")
            lines.append(f"• ML tie‑break: {'Active' if matured else 'Not ready'} (records {t}/{rec_need}, executed {e}/{exe_need})")
            try:
                lines.append(
                    f"• Qscore-only: Trend {bool((cfg.get('trend',{}).get('exec',{}) or {}).get('qscore_only', True))} | "
                    f"Range {bool((cfg.get('range',{}).get('exec',{}) or {}).get('qscore_only', True))} | "
                    f"Scalp {bool((cfg.get('scalp',{}).get('exec',{}) or {}).get('qscore_only', True))}"
                )
            except Exception:
                pass
        except Exception:
            lines.append("(unavailable)")

        # Qscore WR at current thresholds (last 30d, phantoms)
        try:
            lines.append("")
            lines.append("📈 *Qscore WR @ thr (30d)*")
            # Thresholds
            t_thr = int(float(((cfg.get('trend',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 78)))
            r_thr = int(float(((cfg.get('range',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 78)))
            s_thr = int(float(((cfg.get('scalp',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 60)))
            # Helpers
            import datetime as _dt
            cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=30)
            # Trend/Range via PhantomTradeTracker
            tr_w = tr_l = rg_w = rg_l = 0
            try:
                pt = self.shared.get('phantom_tracker')
                for arr in (getattr(pt, 'phantom_trades', {}) or {}).values():
                    for p in arr:
                        try:
                            et = getattr(p, 'exit_time', None)
                            if not et or et < cutoff: continue
                            oc = getattr(p, 'outcome', None)
                            if oc not in ('win','loss'): continue
                            q = (getattr(p, 'features', {}) or {}).get('qscore', None)
                            if not isinstance(q, (int,float)): continue
                            strat = str(getattr(p, 'strategy_name', '') or '').lower()
                            if strat.startswith('range') and q >= r_thr:
                                if oc=='win': rg_w += 1
                                else: rg_l += 1
                            if ('trend' in strat or 'pullback' in strat) and q >= t_thr:
                                if oc=='win': tr_w += 1
                                else: tr_l += 1
                        except Exception:
                            continue
            except Exception:
                pass
            # Scalp via ScalpPhantomTracker
            sc_w = sc_l = 0
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                for arr in (getattr(scpt, 'completed', {}) or {}).values():
                    for p in arr:
                        try:
                            et = getattr(p, 'exit_time', None)
                            if not et or et < cutoff: continue
                            oc = getattr(p, 'outcome', None)
                            if oc not in ('win','loss'): continue
                            q = (getattr(p, 'features', {}) or {}).get('qscore', None)
                            if not isinstance(q, (int,float)) or q < s_thr: continue
                            if oc=='win': sc_w += 1
                            else: sc_l += 1
                        except Exception:
                            continue
            except Exception:
                pass
            def _wr(w,l):
                n=w+l
                return (w/n*100.0) if n else 0.0, n
            twr, tn = _wr(tr_w, tr_l); rwr, rn = _wr(rg_w, rg_l); swr, sn = _wr(sc_w, sc_l)
            lines.append(f"• Trend ≥{t_thr}: N={tn} WR={twr:.1f}% (W/L {tr_w}/{tr_l})")
            lines.append(f"• Range ≥{r_thr}: N={rn} WR={rwr:.1f}% (W/L {rg_w}/{rg_l})")
            lines.append(f"• Scalp ≥{s_thr}: N={sn} WR={swr:.1f}% (W/L {sc_w}/{sc_l})")
        except Exception:
            pass

        # Positions
        positions = book.positions if book else {}
        lines.append("")
        lines.append("📊 *Positions*")
        if positions:
            estimated_risk = per_trade_risk * len(positions)
            lines.append(f"• Open: {len(positions)} | Est risk: ${estimated_risk:.2f}")
        else:
            lines.append("• None")

        # Recent executed (last 5, trend only)
        try:
            tt = self.shared.get('trade_tracker')
            rec = getattr(tt, 'trades', []) or []
            rec = [t for t in rec if 'trend' in (getattr(t, 'strategy_name', '') or '').lower()]
            rec.sort(key=lambda t: getattr(t, 'exit_time', None) or getattr(t, 'entry_time', None), reverse=True)
            last5 = rec[:5]
            if last5:
                lines.append("")
                lines.append("📜 *Recent Executed*")
                for t in last5:
                    try:
                        sym = t.symbol; side = t.side
                        et = t.entry_time.strftime('%m-%d %H:%M') if t.entry_time else ''
                        pnl = f" | PnL ${float(t.pnl_usd):.2f}" if getattr(t, 'exit_price', None) else ''
                        lines.append(f"• {sym} {side.upper()} {et}{pnl}")
                    except Exception:
                        continue
        except Exception:
            pass

        # Trend Phantom aggregate
        try:
            pt = self.shared.get("phantom_tracker")
            total = wins = losses = timeouts = 0
            open_cnt = 0
            if pt:
                for trades in getattr(pt, 'phantom_trades', {}).values():
                    for p in trades:
                        if (getattr(p, 'strategy_name', '') or '').startswith('trend'):
                            oc = getattr(p, 'outcome', None)
                            if oc in ('win','loss'):
                                total += 1
                                wins += (1 if oc == 'win' else 0)
                                losses += (1 if oc == 'loss' else 0)
                            if (not getattr(p, 'was_executed', False)) and getattr(p, 'exit_reason', None) == 'timeout':
                                timeouts += 1
                # Count open (in-flight) trend phantoms
                try:
                    for lst in getattr(pt, 'active_phantoms', {}).values():
                        for p in (lst or []):
                            if (getattr(p, 'strategy_name', '') or '').startswith('trend'):
                                # Only count those without an exit
                                if not getattr(p, 'exit_time', None):
                                    open_cnt += 1
                except Exception:
                    pass
            wr = (wins/total*100.0) if total else 0.0
            lines.append("")
            lines.append("👻 *Trend Phantom*")
            lines.append(f"• Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}")
            # Learned threshold snapshot (Trend)
            try:
                from ml_qscore_trend_adapter import get_trend_qadapter
                thr = get_trend_qadapter().get_threshold({'session': self._session_label(), 'volatility_regime': 'global'}, default=78.0)
                lines.append(f"• Qthr (learned): {thr:.1f}")
            except Exception:
                pass
            try:
                from ml_qscore_trend_adapter import get_trend_qadapter
                qa = get_trend_qadapter()
                bcnt = len(getattr(qa, 'thresholds', {}) or {})
                recs = int(getattr(qa, 'last_train_count', 0))
                lines.append(f"  Buckets: {bcnt} | Records: {recs}")
            except Exception:
                pass
        except Exception:
            pass

        # Range Phantom aggregate
        try:
            pt = self.shared.get("phantom_tracker")
            total = wins = losses = timeouts = 0
            open_cnt = 0
            if pt:
                for trades in getattr(pt, 'phantom_trades', {}).values():
                    for p in trades:
                        if (getattr(p, 'strategy_name', '') or '').startswith('range'):
                            oc = getattr(p, 'outcome', None)
                            if oc in ('win','loss'):
                                total += 1
                                wins += (1 if oc == 'win' else 0)
                                losses += (1 if oc == 'loss' else 0)
                            if (not getattr(p, 'was_executed', False)) and getattr(p, 'exit_reason', None) == 'timeout':
                                timeouts += 1
                # Open phantoms
                try:
                    for lst in getattr(pt, 'active_phantoms', {}).values():
                        for p in (lst or []):
                            if (getattr(p, 'strategy_name', '') or '').startswith('range') and not getattr(p, 'exit_time', None):
                                open_cnt += 1
                except Exception:
                    pass
            wr = (wins/total*100.0) if total else 0.0
            lines.append("")
            lines.append("📦 *Range Phantom*")
            lines.append(f"• Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}")
            # Learned threshold snapshot (Range)
            try:
                from ml_qscore_range_adapter import get_range_qadapter
                thr = get_range_qadapter().get_threshold({'session': self._session_label(), 'volatility_regime': 'global'}, default=78.0)
                lines.append(f"• Qthr (learned): {thr:.1f}")
            except Exception:
                pass
            try:
                from ml_qscore_range_adapter import get_range_qadapter
                qa = get_range_qadapter()
                bcnt = len(getattr(qa, 'thresholds', {}) or {})
                recs = int(getattr(qa, 'last_train_count', 0))
                lines.append(f"  Buckets: {bcnt} | Records: {recs}")
            except Exception:
                pass
        except Exception:
            pass

        # Scalp Phantom aggregate
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            total = wins = losses = timeouts = 0
            open_cnt = 0
            # Completed
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        oc = getattr(p, 'outcome', None)
                        if oc in ('win','loss') and not getattr(p, 'was_executed', False):
                            total += 1
                            if oc == 'win':
                                wins += 1
                            else:
                                losses += 1
                        if str(getattr(p, 'exit_reason', '')).lower() == 'timeout':
                            timeouts += 1
                    except Exception:
                        continue
            # Open
            try:
                act = getattr(scpt, 'active', {}) or {}
                open_cnt = sum(len(lst) for lst in act.values())
            except Exception:
                pass
            wr = (wins/total*100.0) if total else 0.0
            lines.append("")
            lines.append("🩳 *Scalp Phantom*")
            lines.append(f"• Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}")
            # Learned threshold snapshot (Scalp)
            try:
                from autobot.strategies.scalp.qscore_adapter import get_scalp_qadapter
                thr = get_scalp_qadapter().get_threshold({'session': self._session_label(), 'volatility_regime': 'global'}, default=60.0)
                lines.append(f"• Qthr (learned): {thr:.1f}")
            except Exception:
                pass
            try:
                from ml_qscore_scalp_adapter import get_scalp_qadapter
                qa = get_scalp_qadapter()
                bcnt = len(getattr(qa, 'thresholds', {}) or {})
                recs = int(getattr(qa, 'last_train_count', 0))
                lines.append(f"  Buckets: {bcnt} | Records: {recs}")
            except Exception:
                pass
        except Exception:
            pass

        # Executed trade aggregates by strategy (Trend / Range / Scalp)
        try:
            tt = self.shared.get('trade_tracker')
            recs = []
            # Prefer DB (last 365d) for up-to-date executed stats
            if tt and getattr(tt, 'use_db', False) and getattr(tt, 'conn', None):
                try:
                    from datetime import datetime, timedelta
                    cutoff = datetime.utcnow() - timedelta(days=365)
                    rows = []
                    with tt.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT symbol, exit_time, pnl_usd, strategy_name
                            FROM trades
                            WHERE exit_time >= %s
                            ORDER BY exit_time DESC
                            LIMIT 50000
                            """,
                            (cutoff,)
                        )
                        rows = cur.fetchall()
                    class _Row:
                        def __init__(self, sym, et, pnl, strat):
                            self.symbol = sym
                            self.strategy_name = strat
                            if et and hasattr(et, 'tzinfo') and et.tzinfo is not None:
                                self.exit_time = et.replace(tzinfo=None)
                            else:
                                self.exit_time = et
                            self.pnl_usd = float(pnl)
                    recs = [_Row(r[0], r[1], r[2], r[3]) for r in rows]
                except Exception as _eagg:
                    logger.debug(f"Exec aggregates DB query failed: {_eagg}")
                    recs = getattr(tt, 'trades', []) if tt else []
            else:
                recs = getattr(tt, 'trades', []) if tt else []

            def _agg(group):
                arr = [t for t in recs if isinstance(getattr(t, 'strategy_name', None), str) and group(getattr(t,'strategy_name').lower()) and getattr(t, 'exit_time', None)]
                total = len(arr)
                wins = sum(1 for t in arr if float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0)
                losses = total - wins
                wr = (wins/total*100.0) if total else 0.0
                pnl = sum(float(getattr(t, 'pnl_usd', 0.0) or 0.0) for t in arr)
                return total, wins, losses, wr, pnl
            # Trend executed (always show, even if zero)
            t_total, t_w, t_l, t_wr, t_pnl = _agg(lambda s: ('trend' in s))
            lines.append("")
            lines.append("✅ *Trend Executed*")
            lines.append(f"• Closed: {t_total} | WR: {t_wr:.1f}% (W/L {t_w}/{t_l}) | PnL: ${t_pnl:.2f}")
            # Range executed
            r_total, r_w, r_l, r_wr, r_pnl = _agg(lambda s: s.startswith('range'))
            lines.append("")
            lines.append("✅ *Range Executed*")
            lines.append(f"• Closed: {r_total} | WR: {r_wr:.1f}% (W/L {r_w}/{r_l}) | PnL: ${r_pnl:.2f}")
            # Scalp executed
            s_total, s_w, s_l, s_wr, s_pnl = _agg(lambda s: s.startswith('scalp'))
            lines.append("")
            lines.append("✅ *Scalp Executed*")
            lines.append(f"• Closed: {s_total} | WR: {s_wr:.1f}% (W/L {s_w}/{s_l}) | PnL: ${s_pnl:.2f}")
            # 30d executed view for Scalp
            try:
                from datetime import datetime, timedelta
                cutoff = datetime.utcnow() - timedelta(days=30)
                arr30 = [t for t in recs if isinstance(getattr(t, 'strategy_name', None), str)
                         and getattr(t, 'strategy_name').lower().startswith('scalp')
                         and getattr(t, 'exit_time', None) and getattr(t, 'exit_time') >= cutoff]
                tot30 = len(arr30)
                w30 = sum(1 for t in arr30 if float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0)
                l30 = tot30 - w30
                wr30 = (w30/tot30*100.0) if tot30 else 0.0
                pnl30 = sum(float(getattr(t, 'pnl_usd', 0.0) or 0.0) for t in arr30)
                lines.append(f"• 30d: Closed {tot30} | WR: {wr30:.1f}% (W/L {w30}/{l30}) | PnL: ${pnl30:.2f}")
            except Exception:
                pass
        except Exception as exc:
            try:
                logger.debug(f"Executed aggregates error: {exc}")
            except Exception:
                pass

        # Trend ML snapshot
        try:
            ml_scorer = self.shared.get('ml_scorer')
            if ml_scorer:
                ms = ml_scorer.get_stats()
                lines.append("")
                lines.append("🤖 *Trend ML*")
                lines.append(f"• {ms.get('status','⏳')} | Thresh: {ms.get('current_threshold','?'):.0f}")
                lines.append(f"• Trades: {ms.get('completed_trades',0)} | Recent WR: {ms.get('recent_win_rate',0.0):.1f}%")
        except Exception:
            pass

        # State activity (last 10)
        try:
            evts = self.shared.get('trend_events') or []
            if evts:
                lines.append("")
                lines.append("🧭 *State Activity (last 10)*")
                for e in evts[-10:]:
                    sym = e.get('symbol','?'); txt = e.get('text','')
                    lines.append(f"• {sym}: {txt}")
        except Exception:
            pass

        # Config snapshot
        try:
            cfg = self.shared.get('config', {}) or {}
            tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
            sc = (tr_exec.get('scaleout',{}) or {})
            rr = (self.shared.get('trend_settings').rr if self.shared.get('trend_settings') else self.shared.get('risk_reward')) or 2.5
            lines.append("")
            lines.append("⚙️ *Config*")
            lines.append(f"• R:R: 1:{float(rr)} | Stream entry: {'On' if tr_exec.get('allow_stream_entry', True) else 'Off'}")
            lines.append(f"• Scale‑out: {'On' if sc.get('enabled', False) else 'Off'} (TP1 {sc.get('tp1_r',1.6)}R @ {sc.get('fraction',0.5):.2f}, TP2 {sc.get('tp2_r',3.0)}R, BE {'On' if sc.get('move_sl_to_be', True) else 'Off'})")
            # Timeouts summary
            try:
                conf_bars = int(self.shared.get('trend_settings').confirmation_timeout_bars)
                lines.append(f"• Timeouts: confirm {conf_bars} bars | Phantom {(getattr(self.shared.get('phantom_tracker'), 'timeout_hours', 0) or 0)}h")
            except Exception:
                pass
            # Range Execute snapshot
            try:
                rg = (cfg.get('range', {}) or {})
                rx = (rg.get('exec', {}) or {})
                status = 'On' if rx.get('enabled', False) and not rg.get('phantom_only', True) else 'Off'
                lines.append(f"• Range Exec: {status} | Risk {float(rx.get('risk_percent',0.0)):.2f}% | Daily cap {int(rx.get('daily_cap',0))}")
            except Exception:
                pass
            # Scalp Exec snapshot (high-ML stream override)
            try:
                scp = (cfg.get('scalp', {}) or {})
                ex = (scp.get('exec', {}) or {})
                status = 'On' if (bool(ex.get('enabled', False)) or bool(ex.get('allow_stream_high_ml', False))) else 'Off'
                sess = ",".join(scp.get('session_only', []) or [])
                rp = ex.get('risk_percent', None)
                cap = ex.get('daily_cap', None)
                extra = []
                if isinstance(rp, (int,float)):
                    extra.append(f"Risk {float(rp):.2f}%")
                if isinstance(cap, int) and cap > 0:
                    extra.append(f"Daily cap {cap}")
                extra_s = f" | {' | '.join(extra)}" if extra else ""
                lines.append(f"• Scalp Exec: {status} | TF {scp.get('timeframe','3')}m | Sessions {sess if sess else 'all'}{extra_s}")
            except Exception:
                pass
        except Exception:
            pass

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Refresh", callback_data="ui:dash:refresh:trend")],
            [InlineKeyboardButton("📐 Trend States", callback_data="ui:trend:states"), InlineKeyboardButton("📊 Positions", callback_data="ui:positions")],
            [InlineKeyboardButton("📜 Recent", callback_data="ui:recent"), InlineKeyboardButton("👻 Phantom", callback_data="ui:phantom:trend"), InlineKeyboardButton("📦 Range", callback_data="ui:phantom:range")],
            [InlineKeyboardButton("🤖 ML", callback_data="ui:ml:trend"), InlineKeyboardButton("🧠 Patterns", callback_data="ui:ml:patterns")],
            [InlineKeyboardButton("🩳 Scalp", callback_data="ui:scalp:qa"), InlineKeyboardButton("📈 Scalp Qscore", callback_data="ui:scalp:qscore")],
            [InlineKeyboardButton("📊 Qscores (All)", callback_data="ui:qscore:all"), InlineKeyboardButton("🧠 ML Stats", callback_data="ui:ml:stats")],
            [InlineKeyboardButton("📈 Exec WR", callback_data="ui:exec:wr")],
            [InlineKeyboardButton("🧭 Events", callback_data="ui:events"), InlineKeyboardButton("⚙️ Settings", callback_data="ui:settings")]
        ])

        return "\n".join(lines), kb

    def _build_scalp_dashboard(self, more: bool = False):
        """Build Scalp‑only dashboard text and keyboard.
        When more=True, show the expanded secondary actions keyboard.
        """
        frames = self.shared.get("frames", {})
        per_trade_risk, _risk_label = self._compute_risk_snapshot()
        cfg = self.shared.get('config', {}) or {}
        scalp_cfg = (cfg.get('scalp', {}) or {})

        lines = ["🩳 *Scalp Dashboard*", "━━━━━━━━━━━━━━━━━━━━━", ""]
        # System
        lines.append("⚡ *System*")
        lines.append(f"• Status: {'✅ Online' if frames else '⏳ Starting up'}")
        try:
            tf3 = str(scalp_cfg.get('timeframe', '3'))
            lines.append(f"• Timeframe: {tf3}m")
        except Exception:
            pass
        symbols_cfg = self.shared.get("symbols_config")
        if symbols_cfg:
            lines.append(f"• Universe: {len(symbols_cfg)} symbols")
        bal = self.shared.get('last_balance')
        if not isinstance(bal, (int,float)):
            try:
                broker = self.shared.get('broker')
                if broker:
                    nb = broker.get_balance()
                    if nb is not None:
                        self.shared['last_balance'] = nb
                        bal = nb
            except Exception:
                bal = None
        if isinstance(bal, (int,float)):
            lines.append(f"• Balance: ${float(bal):.2f} USDT")

        # Risk (compact)
        try:
            risk = self.shared.get("risk")
            per_trade_dollars = 0.0
            if getattr(risk, 'use_percent_risk', False) and isinstance(bal, (int,float)) and float(bal) > 0:
                per_trade_dollars = float(bal) * (float(getattr(risk, 'risk_percent', 0.0)) / 100.0)
                lines.append("")
                lines.append("💼 *Risk*")
                lines.append(f"• Per trade: {float(getattr(risk,'risk_percent',0.0)):.2f}% (~${per_trade_dollars:.2f}) | Max lev: {getattr(risk,'max_leverage', 20)}x")
        except Exception:
            pass

        # Executed trades (Scalp)
        try:
            tt = self.shared.get('trade_tracker')
            recs = getattr(tt, 'trades', []) if tt else []
            from datetime import datetime, timedelta
            cutoff24 = datetime.utcnow() - timedelta(hours=24)
            cutoff7 = datetime.utcnow() - timedelta(days=7)
            arr = [t for t in recs if isinstance(getattr(t, 'strategy_name', None), str)
                   and getattr(t, 'strategy_name').lower().startswith('scalp')
                   and getattr(t, 'exit_time', None)]
            total = len(arr)
            wins = sum(1 for t in arr if float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0)
            losses = total - wins
            wr = (wins/total*100.0) if total else 0.0
            pnl = sum(float(getattr(t, 'pnl_usd', 0.0) or 0.0) for t in arr)
            lines.append("")
            lines.append("✅ *Executed (Scalp)*")
            lines.append(f"• Closed: {total} | WR {wr:.1f}% ({wins}/{losses}) | PnL ${pnl:.2f}")
            # 24h/7d deltas
            try:
                n24 = sum(1 for t in arr if getattr(t,'exit_time', None) and t.exit_time >= cutoff24)
                n7 = sum(1 for t in arr if getattr(t,'exit_time', None) and t.exit_time >= cutoff7)
                lines.append(f"• 24h: +{n24} | 7d: +{n7}")
            except Exception:
                pass
        except Exception:
            pass

        # Positions (global)
        try:
            lines.append("")
            lines.append("📊 *Positions*")
            book = self.shared.get('book')
            pos_map = getattr(book, 'positions', {}) if book else {}
            open_items = []
            for sym, p in (pos_map.items() if isinstance(pos_map, dict) else []):
                try:
                    if not str(getattr(p, 'strategy_name', '')).lower().startswith('scalp'):
                        continue
                    open_items.append((sym, p))
                except Exception:
                    continue
            count = len(open_items)
            if count > 0:
                est = per_trade_risk * count
                lines.append(f"• {count} open | Est. risk: ${est:.2f}")
                # Show first few lines with risk% and combo tag when available
                try:
                    bot = self.shared.get('bot_instance')
                    pmeta = getattr(bot, '_position_meta', {}) if bot else {}
                    lsf = getattr(bot, '_last_signal_features', {}) if bot else {}
                except Exception:
                    pmeta, lsf = {}, {}
                shown = 0
                for sym, p in open_items[:5]:
                    try:
                        side = str(getattr(p, 'side', '?')).upper()
                        rpct = None
                        if isinstance(pmeta, dict):
                            rpct = (pmeta.get(sym, {}) or {}).get('risk_pct', None)
                        combo = None
                        try:
                            combo = (lsf.get(sym, {}) or {}).get('combo_id')
                        except Exception:
                            combo = None
                        rpct_str = f"{float(rpct):.2f}%" if isinstance(rpct,(int,float)) else "1.00%"
                        combo_str = f" | Combo: {combo}" if combo else ""
                        lines.append(f"• {sym} {side} — Risk {rpct_str}{combo_str}")
                        shown += 1
                    except Exception:
                        continue
                if count > shown:
                    lines.append(f"• … ({count - shown} more)")
            else:
                lines.append("• No open positions")
        except Exception:
            pass

        # Adaptive Risk block (status + gating mode + combo analytics)
        try:
            cfg = self.shared.get('config', {}) or {}
            ar = (((cfg.get('scalp', {}) or {}).get('exec', {}) or {}).get('adaptive_risk', {}) or {})
            lines.append("")
            lines.append("🧮 *Adaptive Risk*")
            if not bool(ar.get('enabled', False)):
                lines.append("• Status: Off")
            else:
                use_hw = bool(ar.get('use_for_high_wr', False))
                base = float(ar.get('base_percent', 1.0))
                rmin = float(ar.get('min_percent', 0.5))
                rmax = float(ar.get('max_percent', 3.0))
                rec = (ar.get('recency', {}) or {})
                r_days = int(rec.get('days', 7)); r_min = int(rec.get('min_samples', 15)); r_only = bool(rec.get('prefer_only', True))
                lines.append(f"• Status: On | Combos: {'On' if use_hw else 'Off'} | Base {base:.2f}% | Range {rmin:.2f}–{rmax:.2f}%")
                lines.append(f"• Recency: {r_days}d N≥{r_min} | Prefer‑only: {'yes' if r_only else 'no'}")
                # Gating mode indicator + recommendation banner
                try:
                    mgr = self.shared.get('adaptive_combo_mgr')
                    cfg = self.shared.get('config', {}) or {}
                    hours = int((((cfg.get('scalp', {}) or {}).get('exec', {}) or {}).get('manager_fresh_hours', 6)))
                    from datetime import datetime, timedelta
                    def _side_ready(side:str) -> bool:
                        try:
                            if not (mgr and getattr(mgr, 'enabled', False)):
                                return False
                            fresh_ok = True
                            if getattr(mgr, 'last_update', None) is not None:
                                fresh_ok = (datetime.utcnow() - mgr.last_update) <= timedelta(hours=max(1, hours))
                            enabled = bool(mgr.get_active_combos(side))
                            return bool(fresh_ok and enabled)
                        except Exception:
                            return False
                    r_long = _side_ready('long')
                    r_short = _side_ready('short')
                    lines.append(f"• Gating mode (Longs): {'Adaptive Combos' if r_long else 'Indicator Rules (Pro fallback)'}")
                    lines.append(f"• Gating mode (Shorts): {'Adaptive Combos' if r_short else 'Indicator Rules (Pro fallback)'}")
                    # Recommendation: suggest enabling combos when WR_lb ≥ 55% and N≥30 per side
                    rec_lines = []
                    def _recommend(side:str):
                        try:
                            if not (mgr and getattr(mgr, 'enabled', False)):
                                return None
                            active = mgr.get_active_combos(side)
                            best = None
                            for c in active:
                                n = int(c.get('n', 0) or 0)
                                wr = float(c.get('wr', 0.0) or 0.0)
                                if n >= 30 and wr >= 55.0:
                                    if best is None or wr > best['wr']:
                                        best = {'key': c.get('combo_id'), 'wr': wr, 'n': n}
                            return best
                        except Exception:
                            return None
                    rec_long = _recommend('long')
                    rec_short = _recommend('short')
                    if rec_long:
                        rec_lines.append(f"• Recommend: Enable COMBO for LONGS (best {rec_long['key']} WR {rec_long['wr']:.1f}% N={rec_long['n']})")
                    if rec_short:
                        rec_lines.append(f"• Recommend: Enable COMBO for SHORTS (best {rec_short['key']} WR {rec_short['wr']:.1f}% N={rec_short['n']})")
                    if not rec_lines:
                        rec_lines.append("• Recommend: Stay on Rules (combos still learning)")
                    lines.extend(rec_lines)
                except Exception:
                    pass

                # Active combos summary (exec vs phantom, 30d + 24h)
                mgr = self.shared.get('adaptive_combo_mgr')
                if mgr and getattr(mgr, 'enabled', False):
                    try:
                        stats = mgr.get_stats_summary()
                        lt = stats.get('long_totals', {}) or {}
                        st = stats.get('short_totals', {}) or {}
                        if (lt.get('n', 0) or 0) > 0 or (st.get('n', 0) or 0) > 0:
                            lines.append("")
                            lines.append("🎯 *Active Combos*")
                            # Longs summary
                            if lt.get('n', 0):
                                lines.append(
                                    f"🟢 Longs: N={lt.get('n',0)} "
                                    f"(Exec {lt.get('n_exec',0)}, Phantom {lt.get('n_phantom',0)}) "
                                    f"| 24h: +{lt.get('n_24h',0)} "
                                    f"(Exec {lt.get('n_exec_24h',0)}, Phantom {lt.get('n_phantom_24h',0)})"
                                )
                            # Shorts summary
                            if st.get('n', 0):
                                lines.append(
                                    f"🔴 Shorts: N={st.get('n',0)} "
                                    f"(Exec {st.get('n_exec',0)}, Phantom {st.get('n_phantom',0)}) "
                                    f"| 24h: +{st.get('n_24h',0)} "
                                    f"(Exec {st.get('n_exec_24h',0)}, Phantom {st.get('n_phantom_24h',0)})"
                                )

                        # Per-side active combos with WR breakdown
                        active_longs = mgr.get_active_combos('long')
                        active_shorts = mgr.get_active_combos('short')

                        def _render_active_side(combos, header, emoji):
                            if not combos:
                                return
                            lines.append("")
                            lines.append(f"{emoji} {header} ({len(combos)} enabled)")
                            combos_sorted = sorted(
                                combos,
                                key=lambda c: (float(c.get('wr', 0.0) or 0.0), int(c.get('n', 0) or 0)),
                                reverse=True
                            )[:3]
                            for idx, c in enumerate(combos_sorted, start=1):
                                wr = float(c.get('wr', 0.0) or 0.0)
                                n = int(c.get('n', 0) or 0)
                                ev = c.get('ev_r', None)
                                ev_str = ""
                                if ev is not None:
                                    try:
                                        ev_str = f", EV_R {float(ev):+.2f}R"
                                    except Exception:
                                        ev_str = ""
                                n_exec = int(c.get('n_exec', 0) or 0)
                                n_ph = int(c.get('n_phantom', 0) or 0)
                                w_exec = int(c.get('wins_exec', 0) or 0)
                                w_ph = int(c.get('wins_phantom', 0) or 0)
                                wr_exec = (w_exec / n_exec * 100.0) if n_exec > 0 else 0.0
                                wr_ph = (w_ph / n_ph * 100.0) if n_ph > 0 else 0.0
                                n_24 = int(c.get('n_24h', 0) or 0)
                                n_exec_24 = int(c.get('n_exec_24h', 0) or 0)
                                n_ph_24 = int(c.get('n_phantom_24h', 0) or 0)
                                lines.append(
                                    f"{idx}) {emoji} WR {wr:.1f}% (N={n}{ev_str})"
                                )
                                lines.append(
                                    f"   Exec {n_exec} (WR {wr_exec:.1f}%), Phantom {n_ph} (WR {wr_ph:.1f}%)"
                                )
                                lines.append(
                                    f"   24h: +{n_24} "
                                    f"(Exec {n_exec_24}, Phantom {n_ph_24})"
                                )
                                lines.append(f"   {c.get('combo_id','')}")

                        if active_longs:
                            _render_active_side(active_longs, "*Longs*", "🟢")
                        if active_shorts:
                            _render_active_side(active_shorts, "*Shorts*", "🔴")
                    except Exception as _ac:
                        logger.debug(f"Active combos panel error: {_ac}")
                # Optional: Blocked counters (last 24h)
                try:
                    import os, redis
                    from datetime import datetime, timedelta
                    url = os.getenv('REDIS_URL')
                    r = None
                    # Prefer phantom tracker redis if available
                    if r is None:
                        try:
                            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                            r = getattr(get_scalp_phantom_tracker(), 'redis_client', None)
                        except Exception:
                            r = None
                    if r is None and url:
                        r = redis.from_url(url, decode_responses=True)
                    if r is not None:
                        now = datetime.utcnow()
                        total_ad = 0; total_rules = 0
                        # Sum totals and per-reason buckets for the last 24 hours
                        reasons = {'mtf': 0, 'vwap': 0, 'rsi': 0, 'macd': 0, 'fib': 0, 'disabled': 0}
                        for i in range(24):
                            ts = (now - timedelta(hours=i)).strftime('%Y%m%d%H')
                            try:
                                total_ad += int(r.get(f'scalp:block:adaptive:{ts}') or 0)
                                total_rules += int(r.get(f'scalp:block:rules:{ts}') or 0)
                                # Reason buckets
                                reasons['disabled'] += int(r.get(f'scalp:block:adaptive:disabled:{ts}') or 0)
                                for k in ('mtf','vwap','rsi','macd','fib'):
                                    reasons[k] += int(r.get(f'scalp:block:rules:{k}:{ts}') or 0)
                            except Exception:
                                continue
                        lines.append(f"• Blocked (24h): {total_ad} by Adaptive | {total_rules} by Rules")
                        # Add breakdown lines when present
                        try:
                            rb = reasons
                            parts = []
                            if (rb['mtf'] + rb['vwap'] + rb['rsi'] + rb['macd'] + rb['fib']) > 0:
                                parts.append(f"MTF {rb['mtf']}")
                                parts.append(f"VWAP {rb['vwap']}")
                                parts.append(f"RSI {rb['rsi']}")
                                parts.append(f"MACD {rb['macd']}")
                                parts.append(f"Fib {rb['fib']}")
                                lines.append("  ↳ Rules breakdown: " + " | ".join(parts))
                            if rb['disabled'] > 0:
                                lines.append(f"  ↳ Adaptive breakdown: disabled {rb['disabled']}")
                        except Exception:
                            pass
                except Exception as _bc:
                    logger.debug(f"Blocked counters unavailable: {_bc}")
        except Exception as _e_ar:
            logger.debug(f"Adaptive risk block skipped: {_e_ar}")
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            st = scpt.get_scalp_phantom_stats()
            lines.append("")
            lines.append("👻 *Scalp Phantom*")
            # Decisive totals (wins/losses only)
            decisive = int(st.get('total', 0)); wins = int(st.get('wins', 0)); losses = int(st.get('losses', 0))
            wrp = float(st.get('wr', 0.0) or 0.0)
            # Open phantoms (active, no exit)
            try:
                active = sum(1 for lst in (getattr(scpt, 'active', {}) or {}).values() for p in (lst or []) if not getattr(p, 'exit_time', None))
            except Exception:
                active = 0
            lines.append(f"• Decisive: {decisive} | WR: {wrp:.1f}% (W/L {wins}/{losses}) | Open: {active}")
            # 30d recorded (wins+losses+timeouts) and new counts (1h/24h)
            try:
                from datetime import datetime, timedelta
                now = datetime.utcnow()
                cutoff30 = now - timedelta(days=30)
                cutoff1h = now - timedelta(hours=1)
                cutoff24 = now - timedelta(hours=24)
                dec30 = 0; w30 = 0; l30 = 0; t30 = 0
                new1h = 0; new24 = 0
                # Completed
                for arr in (getattr(scpt, 'completed', {}) or {}).values():
                    for p in arr:
                        stime = getattr(p, 'signal_time', None)
                        if stime and stime >= cutoff1h:
                            new1h += 1
                        if stime and stime >= cutoff24:
                            new24 += 1
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff30:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc in ('win','loss', 'timeout'):
                            if oc == 'timeout':
                                t30 += 1
                            else:
                                dec30 += 1
                                if oc == 'win':
                                    w30 += 1
                                else:
                                    l30 += 1
                # Active (no exit): count towards new1h/new24 and recorded totals window by signal_time
                for arr in (getattr(scpt, 'active', {}) or {}).values():
                    for p in arr:
                        stime = getattr(p, 'signal_time', None)
                        if stime and stime >= cutoff1h:
                            new1h += 1
                        if stime and stime >= cutoff24:
                            new24 += 1
                lines.append(f"• 30d Recorded: {dec30 + t30} (W/L/TO {w30}/{l30}/{t30})")
                lines.append(f"• New: 1h {new1h} | 24h {new24}")
            except Exception:
                pass
        except Exception:
            pass

        # Scalp executed stats
        try:
            tt = self.shared.get('trade_tracker')
            recs = []
            # Prefer DB for freshness when available (last 365d)
            if tt and getattr(tt, 'use_db', False) and getattr(tt, 'conn', None):
                try:
                    from datetime import datetime, timedelta
                    cutoff = datetime.utcnow() - timedelta(days=365)
                    rows = []
                    with tt.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT symbol, exit_time, pnl_usd, strategy_name
                            FROM trades
                            WHERE exit_time >= %s
                            ORDER BY exit_time DESC
                            LIMIT 50000
                            """,
                            (cutoff,)
                        )
                        rows = cur.fetchall()
                    class _Row:
                        def __init__(self, sym, et, pnl, strat):
                            self.symbol = sym
                            self.strategy_name = strat
                            if et and hasattr(et, 'tzinfo') and et.tzinfo is not None:
                                self.exit_time = et.replace(tzinfo=None)
                            else:
                                self.exit_time = et
                            self.pnl_usd = float(pnl)
                    recs = [_Row(r[0], r[1], r[2], r[3]) for r in rows]
                except Exception as _se_db:
                    logger.debug(f"Scalp exec DB query failed: {_se_db}")
                    recs = getattr(tt, 'trades', []) if tt else []
            else:
                recs = getattr(tt, 'trades', []) if tt else []

            arr = [t for t in recs if isinstance(getattr(t, 'strategy_name', None), str)
                   and getattr(t, 'strategy_name').lower().startswith('scalp')
                   and getattr(t, 'exit_time', None)]
            total = len(arr)
            wins = sum(1 for t in arr if float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0)
            losses = total - wins
            wr = (wins/total*100.0) if total else 0.0
            pnl = sum(float(getattr(t, 'pnl_usd', 0.0) or 0.0) for t in arr)
            lines.append("")
            lines.append("✅ *Scalp Executed*")
            lines.append(f"• Closed: {total} | WR: {wr:.1f}% (W/L {wins}/{losses}) | PnL: ${pnl:.2f}")
            # 30d executed view
            try:
                from datetime import datetime, timedelta
                cutoff = datetime.utcnow() - timedelta(days=30)

                # For 30d view, if DB is available re-query for exactly 30d to avoid filtering large arrays
                if tt and getattr(tt, 'use_db', False) and getattr(tt, 'conn', None):
                    arr30 = []
                    try:
                        rows30 = []
                        with tt.conn.cursor() as cur:
                            cur.execute(
                                """
                                SELECT symbol, exit_time, pnl_usd, strategy_name
                                FROM trades
                                WHERE exit_time >= %s
                                ORDER BY exit_time DESC
                                LIMIT 20000
                                """,
                                (cutoff,)
                            )
                            rows30 = cur.fetchall()
                        class _Row30:
                            def __init__(self, sym, et, pnl, strat):
                                self.symbol = sym
                                self.strategy_name = strat
                                if et and hasattr(et, 'tzinfo') and et.tzinfo is not None:
                                    self.exit_time = et.replace(tzinfo=None)
                                else:
                                    self.exit_time = et
                                self.pnl_usd = float(pnl)
                        recs30 = [_Row30(r[0], r[1], r[2], r[3]) for r in rows30]
                        arr30 = [t for t in recs30 if isinstance(getattr(t, 'strategy_name', None), str)
                                 and str(getattr(t, 'strategy_name')).lower().startswith('scalp')
                                 and getattr(t, 'exit_time', None)]
                    except Exception as _se30:
                        logger.debug(f"Scalp exec 30d DB query failed: {_se30}")
                        # Fallback: filter arr in memory
                        arr30 = [t for t in arr if getattr(t, 'exit_time', None) and getattr(t, 'exit_time') >= cutoff]
                else:
                    # Memory fallback
                    arr30 = [t for t in arr if getattr(t, 'exit_time', None) and getattr(t, 'exit_time') >= cutoff]

                tot30 = len(arr30)
                w30 = sum(1 for t in arr30 if float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0)
                l30 = tot30 - w30
                wr30 = (w30/tot30*100.0) if tot30 else 0.0
                pnl30 = sum(float(getattr(t, 'pnl_usd', 0.0) or 0.0) for t in arr30)
                lines.append(f"• 30d: Closed {tot30} | WR: {wr30:.1f}% (W/L {w30}/{l30}) | PnL: ${pnl30:.2f}")
                # Current open Scalp positions summary (inline here for quick context)
                try:
                    book = self.shared.get('book')
                    pos_map = getattr(book, 'positions', {}) if book else {}
                    open_scalp = []
                    for sym, p in (pos_map.items() if isinstance(pos_map, dict) else []):
                        try:
                            sname = str(getattr(p, 'strategy_name', '') or '')
                            if sname.lower().startswith('scalp'):
                                side = str(getattr(p, 'side', '?')).upper()
                                open_scalp.append((sym, side))
                        except Exception:
                            continue
                    if open_scalp:
                        # Show up to 8 tickers to keep line short
                        preview = ", ".join([f"{s} {sd}" for s, sd in open_scalp[:8]])
                        more = len(open_scalp) - min(8, len(open_scalp))
                        tail = f" (+{more} more)" if more > 0 else ""
                        lines.append(f"• Open: {len(open_scalp)} | {preview}{tail}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as _se:
            try:
                logger.debug(f"Scalp executed stats unavailable: {_se}")
            except Exception:
                pass

        # Adaptive Risk block (status + top combos with 7d sparklines)
        try:
            cfg = self.shared.get('config', {}) or {}
            ar = (((cfg.get('scalp', {}) or {}).get('exec', {}) or {}).get('adaptive_risk', {}) or {})
            lines.append("")
            lines.append("🧮 *Adaptive Risk*")
            if not bool(ar.get('enabled', False)):
                lines.append("• Status: Off")
            else:
                use_hw = bool(ar.get('use_for_high_wr', False))
                base = float(ar.get('base_percent', 1.0))
                rmin = float(ar.get('min_percent', 0.5))
                rmax = float(ar.get('max_percent', 3.0))
                rec = (ar.get('recency', {}) or {})
                r_days = int(rec.get('days', 7)); r_min = int(rec.get('min_samples', 15)); r_only = bool(rec.get('prefer_only', True))
                lines.append(f"• Status: On | Combos: {'On' if use_hw else 'Off'} | Base {base:.2f}% | Range {rmin:.2f}–{rmax:.2f}%")
                lines.append(f"• Recency: {r_days}d N≥{r_min} | Prefer‑only: {'yes' if r_only else 'no'}")

                # Show top few active combos with 7d recency and risk estimate
                mgr = self.shared.get('adaptive_combo_mgr')
                shown = 0
                if mgr and getattr(mgr, 'enabled', False):
                    try:
                        active = mgr.get_active_combos(None)  # both sides
                        # Compute recency stats and select top by 7d N
                        per = []
                        for c in active:
                            key = c.get('combo_id'); side = c.get('side') or 'short'
                            recs = self._scalp_combo_recency(key, side, days=r_days)
                            n7 = int(recs.get('n', 0)); w7 = int(recs.get('w', 0)); rr_sum = float(recs.get('rr_sum', 0.0))
                            series = recs.get('days', []) or []
                            wr_lb7 = self._wilson_lb(w7, n7) if n7 > 0 else 0.0
                            evr7 = (rr_sum / n7) if n7 > 0 else 0.0
                            per.append((n7, key, side, wr_lb7, evr7, series))
                        per.sort(key=lambda t: t[0], reverse=True)
                        # Ladder from config
                        ladder = ar.get('wr_lb_ladder') or {45:1.0,55:1.5,65:2.0,70:2.5}
                        # Normalize ladder to list of dicts sorted ascending by wr_lb
                        if isinstance(ladder, dict):
                            ladder_list = [{'wr_lb': float(k), 'mult': float(v)} for k, v in ladder.items()]
                        else:
                            ladder_list = [{'wr_lb': float(d.get('wr_lb')), 'mult': float(d.get('mult'))} for d in ladder]
                        ladder_list = sorted([d for d in ladder_list if d.get('wr_lb') is not None], key=lambda d: d['wr_lb'])
                        # Render top 4
                        for n7, key, side, wr_lb7, evr7, series in per[:4]:
                            # Determine risk estimate: recency‑only policy
                            risk_est = base
                            if n7 >= r_min:
                                mult = 1.0
                                for step in ladder_list:
                                    if wr_lb7 >= float(step['wr_lb']):
                                        mult = float(step['mult'])
                                # Optional EV floor (use recency field or top‑level)
                                ev_floor = float(rec.get('ev_floor_r', ar.get('ev_floor_r', 0.0)))
                                if evr7 >= ev_floor:
                                    risk_est = max(rmin, min(rmax, base * mult))
                                else:
                                    risk_est = base
                            else:
                                # prefer_only means fallback to base when sparse
                                risk_est = base

                            # Build sparklines for N and WR_LB
                            n_series = [d.get('n', 0) for d in series]
                            wr_series = [d.get('wr_lb', 0.0) for d in series]
                            n_spark = self._sparkline(n_series)
                            wr_spark = self._sparkline(wr_series, zero_to_hundred=True)
                            lines.append(f"• {side.upper()} {key} — 7d N={n7} WR_LB={wr_lb7:.1f}% EV_R={evr7:+.2f} | Risk≈{risk_est:.2f}%")
                            lines.append(f"  N:  {n_spark}")
                            lines.append(f"  WR: {wr_spark}")
                            shown += 1
                    except Exception as _ar:
                        logger.debug(f"Adaptive risk panel error: {_ar}")
                if shown == 0:
                    lines.append("• No recent combo recency data (yet)")
        except Exception as _e_ar:
            logger.debug(f"Adaptive risk block skipped: {_e_ar}")

        # Config snapshot for Scalp
        try:
            ex = (scalp_cfg.get('exec', {}) or {})
            status = 'On' if bool(ex.get('enabled', False)) else 'Off'
            thr = int(float(((scalp_cfg.get('rule_mode', {}) or {}).get('execute_q_min', 60))))
            lines.append("")
            lines.append("⚙️ *Config*")
            lines.append(f"• Exec: {status} | Qthr: {thr}")
            # Risk and caps
            rp = ex.get('risk_percent', None)
            cap = ex.get('daily_cap', None)
            extras = []
            if isinstance(rp, (int,float)):
                extras.append(f"Risk {float(rp):.2f}%")
            if isinstance(cap, int) and cap > 0:
                extras.append(f"Daily cap {cap}")
            if extras:
                lines.append("• " + " | ".join(extras))
            # Off-hours status (auto/fixed)
            try:
                oh = self.shared.get('scalp_offhours') or {}
                if oh:
                    if bool(oh.get('enabled', False)):
                        mode = str(oh.get('mode','auto')).lower()
                        if mode == 'auto':
                            lines.append("• Off-hours: Auto ON")
                        elif mode == 'fixed':
                            wins = oh.get('windows', []) or []
                            lines.append(f"• Off-hours: Fixed ON ({len(wins)} window{'s' if len(wins)!=1 else ''})")
                        else:
                            wins = oh.get('windows', []) or []
                            lines.append(f"• Off-hours: Hybrid ON ({len(wins)} fixed windows)")
                    else:
                        lines.append("• Off-hours: OFF")
            except Exception:
                pass
        except Exception:
            pass

        # Active Combos (from adaptive filter) – always show when manager enabled
        try:
            mgr = self.shared.get('adaptive_combo_mgr')
            if mgr and mgr.enabled:
                try:
                    stats_cm = mgr.get_stats_summary()
                except Exception:
                    stats_cm = {}

                lt = (stats_cm.get('long_totals') or {}) if isinstance(stats_cm, dict) else {}
                st = (stats_cm.get('short_totals') or {}) if isinstance(stats_cm, dict) else {}

                lines.append("")
                # Manual A-tier combo summary (exec path)
                try:
                    bot_mc = self.shared.get('bot_instance')
                    cfg_mc = self.shared.get('config', {}) or {}
                    mce_cfg = (((cfg_mc.get('scalp', {}) or {}).get('exec', {}) or {}).get('manual_combo_exec', {}) or {})
                    m_enabled = bool(mce_cfg.get('enabled', False))
                    m_longs = bool(mce_cfg.get('longs_only', True))
                    mode = str(mce_cfg.get('risk_mode', 'percent')).lower()
                    rp = float(mce_cfg.get('risk_percent', 0.0) or 0.0)
                    ru = float(mce_cfg.get('risk_usd', 0.0) or 0.0)
                    mc_stats = getattr(bot_mc, '_manual_combo_stats', {}) if bot_mc else {}
                    mc_exec = int(mc_stats.get('exec', 0) or 0)
                    mc_block = int(mc_stats.get('blocked_pos', 0) or 0)
                    mc_err = int(mc_stats.get('errors', 0) or 0)
                    mc_checked = int(mc_stats.get('checked', 0) or 0)
                    mc_nonmatch = int(mc_stats.get('nonmatch', 0) or 0)
                    lines.append("🧩 *Manual A-tier Combo*")
                    status_line = "On" if m_enabled else "Off"
                    status_line += " | Longs only" if m_longs else " | Long/Short"
                    lines.append(f"• Status: {status_line}")
                    if mode == 'percent':
                        lines.append(f"• Risk: {rp:.2f}% (mode: percent)")
                    else:
                        lines.append(f"• Risk: ≈${ru:.2f} (mode: usd)")
                    lines.append("• Pattern: RSI:40-60 MACD:bull VWAP:1.2+ Fib:50-61 noMTF")
                    lines.append(f"• Counters: Exec {mc_exec}, Blocked(pos) {mc_block}, Errors {mc_err}")
                    lines.append(f"• Checked: {mc_checked} (non-match {mc_nonmatch})")
                    lines.append("")
                except Exception:
                    pass

                lines.append("🎯 *Active Combos*")

                # Per-side active combos with WR and exec/phantom breakdown (top 3)
                long_combos = mgr.get_active_combos(side='long')
                short_combos = mgr.get_active_combos(side='short')

                # Snapshot of current positions for Exec mapping
                try:
                    book = self.shared.get('book')
                    pos_map = getattr(book, 'positions', {}) if book else {}
                except Exception:
                    pos_map = {}
                # Last signal features (per symbol) for combo_id mapping (fallback only)
                try:
                    bot = self.shared.get('bot_instance')
                    lsf = getattr(bot, '_last_signal_features', {}) if bot else {}
                except Exception:
                    lsf = {}
                # Per-position metadata (primary source for combo_id on executed trades)
                try:
                    bot = self.shared.get('bot_instance')
                    pos_meta = getattr(bot, '_position_meta', {}) if bot else {}
                except Exception:
                    pos_meta = {}
                # Snapshot of active scalp phantoms
                try:
                    from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                    scpt = get_scalp_phantom_tracker()
                    active_ph = getattr(scpt, 'active', {}) or {}
                except Exception:
                    active_ph = {}
                # Combo diagnostics (per-run, from bot)
                try:
                    bot_for_miss = self.shared.get('bot_instance')
                    combo_miss_stats = getattr(bot_for_miss, '_scalp_combo_miss_stats', {}) if bot_for_miss else {}
                    combo_block_stats = getattr(bot_for_miss, '_scalp_combo_block_stats', {}) if bot_for_miss else {}
                except Exception:
                    combo_miss_stats = {}
                    combo_block_stats = {}

                def _combo_key_from_feats(f: dict) -> str | None:
                    try:
                        rsi = f.get('rsi_14'); mh = f.get('macd_hist'); vwap = f.get('vwap_dist_atr')
                        fibz = f.get('fib_zone'); mtf = f.get('mtf_agree_15')
                        if not (isinstance(rsi,(int,float)) and isinstance(mh,(int,float)) and isinstance(vwap,(int,float)) and isinstance(fibz,str) and isinstance(mtf,(bool,int))):
                            return None
                        def _rsi_bin(x: float) -> str:
                            return '<30' if x < 30 else '30-40' if x < 40 else '40-60' if x < 60 else '60-70' if x < 70 else '70+'
                        def _macd_bin(h: float) -> str:
                            return 'bull' if h > 0 else 'bear'
                        def _vwap_bin(x: float) -> str:
                            return '<0.6' if x < 0.6 else '0.6-1.2' if x < 1.2 else '1.2+'
                        rb = _rsi_bin(float(rsi)); mb = _macd_bin(float(mh)); vb = _vwap_bin(float(vwap))
                        ma = 'MTF' if bool(mtf) else 'noMTF'
                        return f"RSI:{rb} MACD:{mb} VWAP:{vb} Fib:{fibz} {ma}"
                    except Exception:
                        return None

                def _render_side(combos, header, emoji, side_label: str):
                    if not combos:
                        lines.append(f"{emoji} {header}: None enabled")
                        return
                    combos_sorted = sorted(
                        combos,
                        key=lambda c: (float(c.get('wr', 0.0) or 0.0), int(c.get('n', 0) or 0)),
                        reverse=True
                    )[:3]
                    lines.append(f"{emoji} *{header} ({len(combos)} enabled)*")
                    for combo in combos_sorted:
                        try:
                            cid = combo.get('combo_id', '')
                            wr = float(combo.get('wr', 0.0) or 0.0)
                            n = int(combo.get('n', 0) or 0)
                            evr = float(combo.get('ev_r', 0.0) or 0.0)
                            wins_exec = int(combo.get('wins_exec', 0) or 0)
                            wins_ph = int(combo.get('wins_phantom', 0) or 0)
                            n_exec = int(combo.get('n_exec', 0) or 0)
                            n_ph = int(combo.get('n_phantom', 0) or 0)
                            wr_exec = (wins_exec / n_exec * 100.0) if n_exec > 0 else 0.0
                            wr_ph = (wins_ph / n_ph * 100.0) if n_ph > 0 else 0.0
                            # Wilson lower bound from wr and n
                            try:
                                wins_total = int(round((wr / 100.0) * n))
                                wr_lb = self._wilson_lb(wins_total, n)
                            except Exception:
                                wr_lb = wr
                            lines.append(
                                f"• WR {wr:.1f}% (LB {wr_lb:.1f}%) | EV_R {evr:+.2f} | N={n} | {cid}"
                            )
                            lines.append(
                                f"   Exec {n_exec} (WR {wr_exec:.1f}%), Phantom {n_ph} (WR {wr_ph:.1f}%)"
                            )
                            # Map open executed trades to this combo
                            open_exec = []
                            for sym, p in (pos_map.items() if isinstance(pos_map, dict) else []):
                                try:
                                    strat_name = str(getattr(p, 'strategy_name', '') or '').lower()
                                    if not strat_name.startswith('scalp'):
                                        continue
                                    pos_side = str(getattr(p, 'side', '') or '').lower()
                                    if pos_side != side_label:
                                        continue
                                    # Prefer combo_id from position metadata; fall back to last_signal_features for legacy entries
                                    meta = pos_meta.get(sym, {}) if isinstance(pos_meta, dict) else {}
                                    sym_combo = meta.get('combo_id')
                                    if not sym_combo:
                                        feats_sym = (lsf.get(sym, {}) or {})
                                        sym_combo = feats_sym.get('combo_id')
                                    if sym_combo == cid:
                                        open_exec.append((sym, p))
                                except Exception:
                                    continue
                            # Map open phantoms to this combo (exclude strict non-combo routes)
                            open_ph = []
                            for sym, lst in (active_ph.items() if isinstance(active_ph, dict) else []):
                                for ph in (lst or []):
                                    try:
                                        ph_side = str(getattr(ph, 'side', '') or '').lower()
                                        if ph_side != side_label:
                                            continue
                                        f = getattr(ph, 'features', {}) or {}
                                        # Skip phantoms that were created via strict non-combo blocks
                                        routing = str(f.get('routing', '') or '')
                                        if routing.startswith('noncombo_'):
                                            continue
                                        ck = _combo_key_from_feats(f)
                                        if ck == cid:
                                            open_ph.append((sym, ph))
                                    except Exception:
                                        continue
                            total_open = len(open_exec) + len(open_ph)
                            lines.append(
                                f"   Open: {total_open} (Exec {len(open_exec)}, Phantom {len(open_ph)})"
                            )
                            # 7d recency stats for this combo
                            try:
                                rec = self._scalp_combo_recency(cid, side_label, days=7)
                                n7 = int(rec.get('n', 0) or 0)
                                w7 = int(rec.get('w', 0) or 0)
                                if n7 > 0:
                                    wr7 = (w7 / n7) * 100.0
                                    lines.append(f"   7d: N={n7} WR {wr7:.1f}%")
                            except Exception:
                                pass
                            # Show up to 3 open entries
                            shown = 0
                            for sym, p in open_exec:
                                try:
                                    ep = float(getattr(p, 'entry', 0.0) or 0.0)
                                    side_up = str(getattr(p, 'side', '') or '').upper()
                                    lines.append(
                                        f"   • Exec — {sym} {side_up} @ {ep:.4f}"
                                    )
                                    shown += 1
                                    if shown >= 3:
                                        break
                                except Exception:
                                    continue
                            if shown < 3:
                                for sym, ph in open_ph:
                                    try:
                                        ep = float(getattr(ph, 'entry_price', 0.0) or 0.0)
                                        side_up = str(getattr(ph, 'side', '') or '').upper()
                                        lines.append(
                                            f"   • Phantom — {sym} {side_up} @ {ep:.4f}"
                                        )
                                        shown += 1
                                        if shown >= 3:
                                            break
                                    except Exception:
                                        continue
                            # Combo shadow-miss summary (diagnostics only)
                            try:
                                miss = combo_miss_stats.get(cid, {})
                                mt = int(miss.get('total', 0) or 0)
                                if mt > 0:
                                    mp = int(miss.get('position_exists', 0) or 0)
                                    me = int(miss.get('exec_error', 0) or 0)
                                    mo = int(miss.get('other', 0) or 0)
                                    lines.append(f"   Would-have-exec (missed): total {mt} (pos {mp}, errors {me}, other {mo})")
                            except Exception:
                                pass
                            # Combo block summary (diagnostics only)
                            try:
                                blk = combo_block_stats.get(cid, {})
                                bt = int(blk.get('total', 0) or 0)
                                if bt > 0:
                                    ba = int(blk.get('adaptive', 0) or 0)
                                    br = int(blk.get('rules', 0) or 0)
                                    lines.append(f"   Blocked signals: total {bt} (adaptive {ba}, rules {br})")
                            except Exception:
                                pass
                        except Exception:
                            lines.append(f"• WR {combo.get('wr',0.0):.1f}% (N={combo.get('n',0)})")

                if long_combos:
                    _render_side(long_combos, "Longs", "🟢", "long")
                else:
                    lines.append("🟢 Longs: None enabled")

                if short_combos:
                    _render_side(short_combos, "Shorts", "🔴", "short")
                else:
                    lines.append("🔴 Shorts: None enabled")

                # Show when last updated
                if mgr.last_update:
                    from datetime import datetime
                    mins_ago = int((datetime.utcnow() - mgr.last_update).total_seconds() / 60)
                    if mins_ago < 60:
                        lines.append(f"_Updated {mins_ago}m ago_")
                    else:
                        hours_ago = mins_ago // 60
                        lines.append(f"_Updated {hours_ago}h ago_")
        except Exception as combo_err:
            logger.debug(f"Active combos display error: {combo_err}")
            pass

        # Empirical break-even WR (30d window)
        try:
            tt = self.shared.get('trade_tracker')
            recs = []
            from datetime import datetime, timedelta
            cutoff = datetime.utcnow() - timedelta(days=30)
            # Prefer DB for freshness
            if tt and getattr(tt, 'use_db', False) and getattr(tt, 'conn', None):
                try:
                    rows = []
                    with tt.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT exit_time, pnl_usd, pnl_percent, strategy_name, exit_reason
                            FROM trades
                            WHERE exit_time >= %s
                            ORDER BY exit_time DESC
                            LIMIT 50000
                            """,
                            (cutoff,)
                        )
                        rows = cur.fetchall()
                    class _T:
                        def __init__(self, et, pnl_usd, pnl_pct, strat, reason):
                            # Normalize tz to naive UTC
                            self.exit_time = et.replace(tzinfo=None) if (et and getattr(et, 'tzinfo', None)) else et
                            self.pnl_usd = float(pnl_usd)
                            self.pnl_percent = float(pnl_pct) if pnl_pct is not None else None
                            self.strategy_name = strat
                            self.exit_reason = reason
                    recs = [_T(*r) for r in rows]
                except Exception:
                    recs = getattr(tt, 'trades', []) if tt else []
            else:
                recs = getattr(tt, 'trades', []) if tt else []

            # Filter Scalp executed trades within window
            ex = []
            for t in recs:
                try:
                    if not getattr(t, 'exit_time', None) or t.exit_time < cutoff:
                        continue
                    strat = str(getattr(t, 'strategy_name', '') or '').lower()
                    if not strat.startswith('scalp'):
                        continue
                    ex.append(t)
                except Exception:
                    continue

            if ex:
                try:
                    import statistics as _stats
                    wins = [abs(float(getattr(t, 'pnl_percent', 0.0) or 0.0)) for t in ex if float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0]
                    losses = [abs(float(getattr(t, 'pnl_percent', 0.0) or 0.0)) for t in ex if float(getattr(t, 'pnl_usd', 0.0) or 0.0) <= 0.0]
                    med_win = _stats.median(wins) if wins else 0.0
                    med_loss = _stats.median(losses) if losses else 0.0
                    emp_be = (med_loss / (med_loss + med_win) * 100.0) if (med_loss > 0.0 and med_win > 0.0) else None
                    # Planned BE from RR and fees
                    cfg = self.shared.get('config', {}) or {}
                    rr = float(((cfg.get('scalp', {}) or {}).get('rr', 2.1)))
                    planned_be = 100.0 / (1.0 + rr)
                    fee_total = float(((cfg.get('trade', {}) or {}).get('fee_total_pct', 0.00110)))*100.0  # to percent
                    be_with_fees = None
                    if med_loss > 0.0:
                        c_R = fee_total / med_loss
                        be_with_fees = (1.0 + c_R) / (1.0 + rr) * 100.0
                    lines.append("")
                    lines.append("⚖️ Break-even (Scalp)")
                    if emp_be is not None:
                        lines.append(f"• Empirical 30d: {emp_be:.1f}%  (med win {med_win:.2f}%, med loss {med_loss:.2f}%)")
                    else:
                        lines.append("• Empirical 30d: n/a (insufficient executed data)")
                    if be_with_fees is not None:
                        lines.append(f"• Planned (RR={rr:.1f}): no-fee {planned_be:.1f}% | with fees ~ {be_with_fees:.1f}%")
                    else:
                        lines.append(f"• Planned (RR={rr:.1f}): no-fee {planned_be:.1f}%")
                except Exception:
                    pass
        except Exception:
            pass

        # Buttons
        # Dynamic counts
        pos_count = 0
        try:
            book = self.shared.get('book')
            pos_map = getattr(book, 'positions', {}) if book else {}
            pos_count = sum(1 for p in (pos_map.values() if isinstance(pos_map, dict) else []) if str(getattr(p,'strategy_name','')).lower().startswith('scalp'))
        except Exception:
            pos_count = 0
        phantom_open = 0
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            phantom_open = sum(1 for lst in (getattr(scpt, 'active', {}) or {}).values() for p in (lst or []) if not getattr(p, 'exit_time', None))
        except Exception:
            phantom_open = 0

        if not more:
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data="ui:dash:refresh:scalp"), InlineKeyboardButton(f"📊 Positions ({pos_count})", callback_data="ui:positions")],
                [InlineKeyboardButton("📈 Exec WR", callback_data="ui:exec:wr"), InlineKeyboardButton("🧮 Risk", callback_data="ui:scalp:risk"), InlineKeyboardButton("🎯 Combos", callback_data="ui:scalp:combos")],
                [InlineKeyboardButton("⋯ More ▾", callback_data="ui:dash:more")]
            ])
        else:
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🧪 QA", callback_data="ui:scalp:qa"), InlineKeyboardButton("🧰 Gates", callback_data="ui:scalp:gates")],
                [InlineKeyboardButton("📊 Pro Analytics", callback_data="ui:scalp:proanalytics"), InlineKeyboardButton("⚖️ Break-even", callback_data="ui:scalp:risk")],
                [InlineKeyboardButton(f"👻 Phantom ({phantom_open})", callback_data="ui:scalp:open_phantoms"), InlineKeyboardButton("🗓 Sessions/Days", callback_data="ui:scalp:timewr")],
                [InlineKeyboardButton("📊 Advanced Combos", callback_data="ui:scalp:advancedcombos"), InlineKeyboardButton("🧠 Ultimate", callback_data="ui:scalp:ultimate")],
                [InlineKeyboardButton("🧩 Build Strategy", callback_data="ui:scalp:buildstrategy"), InlineKeyboardButton("⚙️ Settings", callback_data="ui:settings")],
                [InlineKeyboardButton("▲ Less", callback_data="ui:dash:less")]
            ])

        return "\n".join(lines), kb

    def _build_dashboard(self):
        """Build dashboard text and inline keyboard without sending it.
        Returns (text:str, keyboard:InlineKeyboardMarkup|None).
        """
        frames = self.shared.get("frames", {})
        book = self.shared.get("book")
        last_analysis = self.shared.get("last_analysis", {})
        per_trade_risk, risk_label = self._compute_risk_snapshot()

        lines = ["🎯 *Trading Bot Dashboard*", "━━━━━━━━━━━━━━━━━━━━━", ""]

        # System status
        lines.append("⚡ *System Status*")
        if frames:
            lines.append("• Status: ✅ Online")
            lines.append(f"• Symbols streaming: {len(frames)}")
        else:
            lines.append("• Status: ⏳ Starting up")

        timeframe = self.shared.get("timeframe")
        if timeframe:
            lines.append(f"• Timeframe: {timeframe}m")

        symbols_cfg = self.shared.get("symbols_config")
        if symbols_cfg:
            lines.append(f"• Universe: {len(symbols_cfg)} symbols")

        if last_analysis:
            try:
                latest_symbol, latest_time = max(last_analysis.items(), key=lambda kv: kv[1])
                if isinstance(latest_time, datetime):
                    ref_now = datetime.now(latest_time.tzinfo) if latest_time.tzinfo else datetime.now()
                    age_minutes = max(0, int((ref_now - latest_time).total_seconds() // 60))
                    lines.append(f"• Last scan: {latest_symbol} ({age_minutes}m ago)")
            except Exception as exc:
                logger.debug(f"Unable to compute last analysis recency: {exc}")

        broker = self.shared.get("broker")
        balance = self.shared.get("last_balance")
        if broker:
            if balance is None or float(balance) <= 0.0:
                try:
                    balance = broker.get_balance()
                    if balance is not None:
                        self.shared["last_balance"] = balance
                except Exception as exc:
                    logger.warning(f"Error refreshing balance: {exc}")
            if balance is not None:
                lines.append(f"• Balance: ${balance:.2f} USDT")
            # API key expiry (robust to different field names and formats)
            try:
                api_info = broker.get_api_key_info()
                expiry_val = None
                if api_info:
                    for k in ("expiredAt", "expireAt", "expireTime", "expirationTime"):
                        if k in api_info:
                            expiry_val = api_info.get(k)
                            break
                days_remaining = None
                if expiry_val is None:
                    # Unknown or non-expiring key
                    pass
                else:
                    try:
                        # Numeric: seconds or ms
                        ts = int(str(expiry_val))
                        if ts == 0:
                            days_remaining = None  # non-expiring
                        else:
                            if ts > 10_000_000_000:  # ms
                                ts = ts / 1000.0
                            expiry_date = datetime.fromtimestamp(ts)
                            days_remaining = (expiry_date - datetime.now()).days
                    except Exception:
                        # ISO string fallback
                        try:
                            iso = str(expiry_val).replace('Z', '+00:00')
                            expiry_date = datetime.fromisoformat(iso)
                            days_remaining = (expiry_date - datetime.now(expiry_date.tzinfo) if expiry_date.tzinfo else expiry_date - datetime.now()).days
                        except Exception:
                            days_remaining = None

                if days_remaining is None:
                    lines.append("🔑 API Key: no expiry")
                else:
                    if days_remaining < 14:
                        lines.append(f"⚠️ *API Key:* expires in {days_remaining} days")
                    else:
                        lines.append(f"🔑 API Key: {days_remaining} days remaining")
            except Exception as exc:
                logger.warning(f"Could not fetch API key expiry: {exc}")

        lines.append("")

        # Trading settings
        risk = self.shared.get("risk")
        lines.append("⚙️ *Trading Settings*")
        lines.append(f"• Risk per trade: {risk_label}")
        lines.append(f"• Max leverage: {risk.max_leverage}x")
        if getattr(risk, 'use_ml_dynamic_risk', False):
            lines.append("• ML Dynamic Risk: Enabled")

        # Trend ML
        ml_scorer = self.shared.get("ml_scorer")
        if ml_scorer:
            try:
                ml_stats = ml_scorer.get_stats()
                lines.append("")
                lines.append("🤖 *Trend ML*")
                lines.append(f"• Status: {ml_stats['status']}")
                lines.append(f"• Trades used: {ml_stats['completed_trades']}")
                if ml_stats.get('recent_win_rate'):
                    lines.append(f"• Recent win rate: {ml_stats['recent_win_rate']:.1f}%")
            except Exception as exc:
                logger.debug(f"Unable to fetch trend ML stats: {exc}")

        # Enhanced MR ML
        enhanced_mr = self.shared.get("enhanced_mr_scorer")
        if enhanced_mr:
            try:
                mr_info = enhanced_mr.get_retrain_info()
                lines.append("")
                lines.append("🧠 *Mean Reversion ML*")
                status = "✅ Ready" if mr_info.get('is_ml_ready') else "⏳ Training"
                lines.append(f"• Status: {status}")
                lines.append(f"• Trades (exec + phantom): {mr_info.get('total_combined', 0)}")
                lines.append(f"• Next retrain in: {mr_info.get('trades_until_next_retrain', 0)} trades")
                # Clarify executed vs phantom counts for transparency
                lines.append(f"• Executed: {mr_info.get('completed_trades', 0)} | Phantom: {mr_info.get('phantom_count', 0)}")
                # Last retrain timestamp (if available)
                try:
                    last_ts_mr = mr_info.get('last_retrain_ts')
                    if last_ts_mr:
                        try:
                            from datetime import datetime as _dt
                            t0 = _dt.fromisoformat(str(last_ts_mr).replace('Z',''))
                            delta = _dt.utcnow() - t0
                            mins = int(delta.total_seconds()//60)
                            lines.append(f"• Last retrain: {mins//60}h {mins%60}m ago")
                        except Exception:
                            lines.append(f"• Last retrain: {last_ts_mr}")
                except Exception:
                    pass
                # MR Promotion status
                try:
                    mr_stats = enhanced_mr.get_enhanced_stats()
                    mp = self.shared.get('mr_promotion', {}) or {}
                    cfg = self.shared.get('config', {}) or {}
                    prom_cfg = (cfg.get('mr', {}) or {}).get('promotion', {})
                    cap = int(prom_cfg.get('daily_exec_cap', 0))
                    promote_wr = float(prom_cfg.get('promote_wr', 0.0))
                    demote_wr = float(prom_cfg.get('demote_wr', 0.0))
                    lines.append("")
                    lines.append("🌀 *MR Promotion*")
                    lines.append(f"• Status: {'✅ Active' if mp.get('active') else 'Off'} | Used: {mp.get('count',0)}/{cap}")
                    lines.append(f"• Promote/Demote: {promote_wr:.0f}%/{demote_wr:.0f}% | Recent WR: {mr_stats.get('recent_win_rate',0.0):.1f}%")
                except Exception:
                    pass
            except Exception as exc:
                logger.debug(f"Unable to fetch MR ML stats: {exc}")

        # Trend ML
        try:
            from ml_scorer_trend import get_trend_scorer
            tr_scorer = get_trend_scorer()
            lines.append("")
            lines.append("📈 *Trend ML*")
            ready = '✅ Ready' if getattr(tr_scorer, 'is_ml_ready', False) else '⏳ Training'
            lines.append(f"• Status: {ready}")
            try:
                tstats = tr_scorer.get_retrain_info()
                total = tstats.get('total_records', 0)
                exec_n = tstats.get('executed_count', 0)
                ph_n = tstats.get('phantom_count', 0)
                lines.append(f"• Trades (exec + phantom): {total}")
                lines.append(f"• Next retrain in: {tstats.get('trades_until_next_retrain',0)} trades")
                lines.append(f"• Executed: {exec_n} | Phantom: {ph_n}")
                # Last retrain timestamp (if available)
                try:
                    last_ts_tr = tstats.get('last_retrain_ts')
                    if last_ts_tr:
                        try:
                            from datetime import datetime as _dt
                            t0 = _dt.fromisoformat(str(last_ts_tr).replace('Z',''))
                            delta = _dt.utcnow() - t0
                            mins = int(delta.total_seconds()//60)
                            lines.append(f"• Last retrain: {mins//60}h {mins%60}m ago")
                        except Exception:
                            lines.append(f"• Last retrain: {last_ts_tr}")
                except Exception:
                    pass
                lines.append(f"• Threshold: {getattr(tr_scorer, 'min_score', 70):.0f}")
            except Exception:
                pass
            # Trend Promotion status
            try:
                cfg = self.shared.get('config', {}) or {}
                tr_cfg = (cfg.get('trend', {}) or {}).get('promotion', {})
                tp = self.shared.get('trend_promotion', {}) or {}
                cap = int(tr_cfg.get('daily_exec_cap', 20))
                stats = tr_scorer.get_stats() if tr_scorer else {}
                lines.append("")
                lines.append("🚀 *Trend Promotion*")
                lines.append(f"• Status: {'✅ Active' if tp.get('active') else 'Off'} | Used: {tp.get('count',0)}/{cap}")
                lines.append(f"• Promote/Demote: {float(tr_cfg.get('promote_wr',55.0)):.0f}%/{float(tr_cfg.get('demote_wr',35.0)):.0f}% | Recent WR: {float(stats.get('recent_win_rate',0.0)):.1f}%")
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"Trend ML not available: {exc}")

        # Scalp ML
        try:
            from autobot.strategies.scalp.scorer import get_scalp_scorer
            sc_scorer = get_scalp_scorer()
            lines.append("")
            lines.append("🩳 *Scalp ML*")
            ready = '✅ Ready' if getattr(sc_scorer, 'is_ml_ready', False) else '⏳ Training'
            lines.append(f"• Status: {ready}")
            # Stats and retrain info
            try:
                s_stats = sc_scorer.get_stats()
                s_ret = sc_scorer.get_retrain_info()
                lines.append(f"• Records: {s_stats.get('total_records',0)} | Trainable: {s_ret.get('trainable_size',0)}")
                lines.append(f"• Next retrain in: {s_ret.get('trades_until_next_retrain',0)} trades")
                # Last retrain timestamp (if available)
                try:
                    last_ts_sc = s_ret.get('last_retrain_ts')
                    if last_ts_sc:
                        try:
                            from datetime import datetime as _dt
                            t0 = _dt.fromisoformat(str(last_ts_sc).replace('Z',''))
                            delta = _dt.utcnow() - t0
                            mins = int(delta.total_seconds()//60)
                            lines.append(f"• Last retrain: {mins//60}h {mins%60}m ago")
                        except Exception:
                            lines.append(f"• Last retrain: {last_ts_sc}")
                except Exception:
                    pass
                lines.append(f"• Threshold: {getattr(sc_scorer, 'min_score', 75):.0f}")
            except Exception:
                lines.append(f"• Samples: {getattr(sc_scorer, 'completed_trades', 0)}")
                lines.append(f"• Threshold: {getattr(sc_scorer, 'min_score', 75):.0f}")
        except Exception as exc:
            logger.debug(f"Scalp ML not available: {exc}")

        # Scalp Phantom
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            st = scpt.get_scalp_phantom_stats()
            lines.append("🩳 *Scalp Phantom*")
            lines.append(f"• Recorded: {st.get('total', 0)} | WR: {st.get('wr', 0.0):.1f}%")
        except Exception as exc:
            logger.debug(f"Scalp phantom not available: {exc}")
        # Scalp Promotion status (include current and recent WR)
        try:
            cfg2 = self.shared.get('config', {}) or {}
            sc_cfg = (cfg2.get('scalp', {}) or {})
            sp = self.shared.get('scalp_promotion', {}) or {}
            cap = int(sc_cfg.get('daily_exec_cap', 20))
            # Current overall WR from phantom stats
            cur_wr = 0.0
            try:
                cur_wr = float(st.get('wr', 0.0))
            except Exception:
                cur_wr = 0.0
            # Recent WR over last 50 scalp phantoms
            recent_wr = cur_wr
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt2 = get_scalp_phantom_tracker()
                recents = []
                for trades in getattr(scpt2, 'completed', {}).values():
                    for p in trades:
                        if getattr(p, 'outcome', None) in ('win','loss'):
                            recents.append(p)
                recents.sort(key=lambda x: getattr(x, 'exit_time', None) or getattr(x, 'signal_time', None))
                recents = recents[-50:]
                if recents:
                    rw = sum(1 for p in recents if getattr(p, 'outcome', None) == 'win')
                    recent_wr = (rw / len(recents)) * 100.0
            except Exception:
                recent_wr = cur_wr
            lines.append("")
            lines.append("🩳 *Scalp Promotion*")
            lines.append(f"• Status: {'✅ Active' if sp.get('active') else 'Off'} | Used: {sp.get('count',0)}/{cap}")
            # Gate description depending on metric
            metric = str(sc_cfg.get('promote_metric', 'recent')).lower()
            window = int(sc_cfg.get('promote_window', 50))
            lines.append(f"• Current WR: {cur_wr:.1f}% | Recent WR({window}): {recent_wr:.1f}%")
            if metric == 'recent':
                lines.append(f"• Promote WR: {float(sc_cfg.get('promote_min_wr',50.0)):.0f}% | Gate: Recent({window})")
            else:
                lines.append(f"• Promote WR: {float(sc_cfg.get('promote_min_wr',50.0)):.0f}% | Gate: Overall")
        except Exception:
            pass

        # Scalp Shadow (ML-based)
        try:
            from shadow_trade_simulator import get_shadow_tracker
            sstats = get_shadow_tracker().get_stats().get('scalp', {})
            if sstats:
                lines.append("🩳 *Scalp Shadow*")
                lines.append(f"• Trades: {sstats.get('total',0)} | WR: {sstats.get('wr',0.0):.1f}%")
        except Exception as exc:
            logger.debug(f"Scalp shadow stats unavailable: {exc}")

        # Positions
        positions = book.positions if book else {}
        lines.append("")
        lines.append("📊 *Positions*")
        if positions:
            estimated_risk = per_trade_risk * len(positions)
            lines.append(f"• Open positions: {len(positions)}")
            lines.append(f"• Estimated risk: ${estimated_risk:.2f}")
            if self.shared.get('use_enhanced_parallel', False):
                lines.append("• Routing: Enhanced parallel (Trend Pullback + MR)")
        else:
            lines.append("• No open positions")

        # Trend Phantom (aggregate from generic phantom tracker)
        phantom_tracker = self.shared.get("phantom_tracker")
        if phantom_tracker:
            try:
                total = 0; wins = 0; losses = 0; timeouts = 0; open_cnt = 0
                for trades in getattr(phantom_tracker, 'phantom_trades', {}).values():
                    for p in trades:
                        try:
                            if getattr(p, 'strategy_name', '') not in ('trend_breakout','trend_pullback'):
                                continue
                            if getattr(p, 'outcome', None) in ('win','loss'):
                                total += 1
                                if p.outcome == 'win': wins += 1
                                else: losses += 1
                            if not getattr(p, 'was_executed', False) and getattr(p, 'exit_reason', None) == 'timeout':
                                timeouts += 1
                        except Exception:
                            pass
                # Count open (in-flight) trend phantoms
                try:
                    for lst in getattr(phantom_tracker, 'active_phantoms', {}).values():
                        for p in (lst or []):
                            if getattr(p, 'strategy_name', '') in ('trend_breakout','trend_pullback'):
                                if not getattr(p, 'exit_time', None):
                                    open_cnt += 1
                except Exception:
                    pass
                wr = (wins/total*100.0) if total else 0.0
                lines.append("")
                lines.append("👻 *Trend Phantom*")
                lines.append(f"• Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}")
            except Exception as exc:
                logger.debug(f"Unable to fetch trend phantom stats: {exc}")

        # Range Phantom (aggregate)
        phantom_tracker = self.shared.get("phantom_tracker")
        if phantom_tracker:
            try:
                total = wins = losses = timeouts = open_cnt = 0
                for trades in getattr(phantom_tracker, 'phantom_trades', {}).values():
                    for p in trades:
                        try:
                            if not (getattr(p, 'strategy_name', '') or '').startswith('range'):
                                continue
                            if getattr(p, 'outcome', None) in ('win','loss'):
                                total += 1
                                if p.outcome == 'win': wins += 1
                                else: losses += 1
                            if not getattr(p, 'was_executed', False) and getattr(p, 'exit_reason', None) == 'timeout':
                                timeouts += 1
                        except Exception:
                            pass
                # Open
                try:
                    for lst in getattr(phantom_tracker, 'active_phantoms', {}).values():
                        for p in (lst or []):
                            if (getattr(p, 'strategy_name', '') or '').startswith('range') and not getattr(p, 'exit_time', None):
                                open_cnt += 1
                except Exception:
                    pass
                lines.append("")
                lines.append("📦 *Range Phantom*")
                wr = (wins/total*100.0) if total else 0.0
                lines.append(f"• Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}")
            except Exception as exc:
                logger.debug(f"Unable to fetch range phantom stats: {exc}")

        # MR Phantom section hidden (MR disabled)

        # Phantom + Executed summaries with WR and open/closed
        try:
            # Trend phantom summary (prefer in-memory; fallback to Redis snapshot if empty)
            pb_wins = pb_losses = pb_closed = 0
            pb_open_trades = pb_open_syms = 0
            if phantom_tracker:
                try:
                    for trades in phantom_tracker.phantom_trades.values():
                        for p in trades:
                            if not getattr(p, 'was_executed', False) and getattr(p, 'outcome', None) in ('win','loss'):
                                pb_closed += 1
                                if p.outcome == 'win':
                                    pb_wins += 1
                                else:
                                    pb_losses += 1
                    act = getattr(phantom_tracker, 'active_phantoms', {}) or {}
                    pb_open_syms = len(act)
                    try:
                        pb_open_trades = sum(len(lst) for lst in act.values())
                    except Exception:
                        pb_open_trades = pb_open_syms
                except Exception:
                    pass
                # Fallback to Redis snapshot when no in-memory counts are present
                try:
                    if (pb_wins + pb_losses + pb_open_trades) == 0 and getattr(phantom_tracker, 'redis_client', None):
                        r = phantom_tracker.redis_client
                        try:
                            comp_raw = r.get('phantom:completed')
                            if comp_raw:
                                import json as _json
                                items = _json.loads(comp_raw)
                                for rec in items:
                                    if not bool(rec.get('was_executed')) and rec.get('outcome') in ('win','loss'):
                                        pb_closed += 1
                                        if rec.get('outcome') == 'win':
                                            pb_wins += 1
                                        else:
                                            pb_losses += 1
                        except Exception:
                            pass
                        try:
                            act_raw = r.get('phantom:active')
                            if act_raw:
                                import json as _json
                                act = _json.loads(act_raw) or {}
                                pb_open_syms = len(act)
                                try:
                                    pb_open_trades = sum(len(v) if isinstance(v, list) else 1 for v in act.values())
                                except Exception:
                                    pb_open_trades = pb_open_syms
                        except Exception:
                            pass
                except Exception:
                    pass
            pb_wr = (pb_wins / (pb_wins + pb_losses) * 100.0) if (pb_wins + pb_losses) else 0.0

            # MR phantom summary
            mr_wins = mr_losses = mr_closed = 0
            mr_open_trades = mr_open_syms = 0
            if mr_phantom:
                try:
                    for trades in mr_phantom.mr_phantom_trades.values():
                        for p in trades:
                            if not getattr(p, 'was_executed', False) and getattr(p, 'outcome', None) in ('win','loss'):
                                mr_closed += 1
                                if p.outcome == 'win':
                                    mr_wins += 1
                                else:
                                    mr_losses += 1
                    act_mr = getattr(mr_phantom, 'active_mr_phantoms', {}) or {}
                    mr_open_syms = len(act_mr)
                    try:
                        mr_open_trades = sum(len(lst) for lst in act_mr.values())
                    except Exception:
                        mr_open_trades = mr_open_syms
                except Exception:
                    pass
            mr_wr = (mr_wins / (mr_wins + mr_losses) * 100.0) if (mr_wins + mr_losses) else 0.0

            # Scalp phantom summary
            sc_wins = sc_losses = sc_closed = 0
            sc_open_trades = sc_open_syms = 0
            sc_timeouts = 0
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                try:
                    for trades in scpt.completed.values():
                        for p in trades:
                            # Count timeouts regardless of outcome bucket
                            if str(getattr(p, 'exit_reason', '')).lower() == 'timeout':
                                sc_timeouts += 1
                            # Closed wins/losses (phantom-only)
                            if getattr(p, 'outcome', None) in ('win','loss') and not getattr(p, 'was_executed', False):
                                sc_closed += 1
                                if p.outcome == 'win':
                                    sc_wins += 1
                                else:
                                    sc_losses += 1
                    act_sc = getattr(scpt, 'active', {}) or {}
                    sc_open_syms = len(act_sc)
                    try:
                        sc_open_trades = sum(len(lst) for lst in act_sc.values())
                    except Exception:
                        sc_open_trades = sc_open_syms
                except Exception:
                    pass
            except Exception:
                pass
            sc_wr = (sc_wins / (sc_wins + sc_losses) * 100.0) if (sc_wins + sc_losses) else 0.0

            # Executed summaries by strategy (closed from trade tracker, open from book)
            exec_stats = {
                'trend': {'wins': 0, 'losses': 0, 'closed': 0, 'open': 0},
                'mr': {'wins': 0, 'losses': 0, 'closed': 0, 'open': 0},
                'scalp': {'wins': 0, 'losses': 0, 'closed': 0, 'open': 0},
            }
            # Closed executed
            try:
                tt = self.shared.get('trade_tracker')
                trades = getattr(tt, 'trades', []) or []
                for t in trades:
                    strat = (getattr(t, 'strategy_name', '') or '').lower()
                    grp = 'trend' if ('trend' in strat or 'pullback' in strat) else 'mr' if ('mr' in strat or 'reversion' in strat) else 'scalp' if 'scalp' in strat else None
                    if grp:
                        exec_stats[grp]['closed'] += 1
                        try:
                            pnl = float(getattr(t, 'pnl_usd', 0))
                        except Exception:
                            pnl = 0.0
                        if pnl > 0:
                            exec_stats[grp]['wins'] += 1
                        else:
                            exec_stats[grp]['losses'] += 1
            except Exception:
                pass
            # Open executed
            try:
                for pos in (self.shared.get('book').positions.values() if self.shared.get('book') else []):
                    strat = (getattr(pos, 'strategy_name', '') or '').lower()
                    grp = 'trend' if ('trend' in strat or 'pullback' in strat) else 'mr' if ('mr' in strat or 'reversion' in strat) else 'scalp' if 'scalp' in strat else None
                    if grp:
                        exec_stats[grp]['open'] += 1
            except Exception:
                pass

            lines.append("")
            lines.append("👻 *Phantom Summary*")
            # Trend (phantom)
            lines.append(f"• 🔵 Trend")
            lines.append(f"  ✅ W: {pb_wins}   ❌ L: {pb_losses}   🎯 WR: {pb_wr:.1f}%")
            lines.append(f"  🟢 Open: {pb_open_trades} ({pb_open_syms} syms)   🔒 Closed: {pb_closed}")
            # Mean Reversion (phantom) — hidden when MR disabled
            try:
                cfg = self.shared.get('config') or {}
                if not bool(((cfg.get('modes', {}) or {}).get('disable_mr', True))):
                    lines.append(f"• 🌀 Mean Reversion")
                    lines.append(f"  ✅ W: {mr_wins}   ❌ L: {mr_losses}   🎯 WR: {mr_wr:.1f}%")
                    lines.append(f"  🟢 Open: {mr_open_trades} ({mr_open_syms} syms)   🔒 Closed: {mr_closed}")
            except Exception:
                pass
            # Scalp (phantom)
            lines.append(f"• 🩳 Scalp")
            lines.append(f"  ✅ W: {sc_wins}   ❌ L: {sc_losses}   🎯 WR: {sc_wr:.1f}%")
            lines.append(f"  🟢 Open: {sc_open_trades} ({sc_open_syms} syms)   🔒 Closed: {sc_closed}")
            lines.append(f"  ⏱️ Timeouts: {sc_timeouts}")

            lines.append("")
            lines.append("✅ *Executed Summary*")
            # Executed Trend
            pbx = exec_stats['trend']
            pbx_wr = (pbx['wins'] / (pbx['wins'] + pbx['losses']) * 100.0) if (pbx['wins'] + pbx['losses']) else 0.0
            lines.append("• 🔵 Trend")
            lines.append(f"  ✅ W: {pbx['wins']}   ❌ L: {pbx['losses']}   🎯 WR: {pbx_wr:.1f}%")
            lines.append(f"  🔓 Open: {pbx['open']}   🔒 Closed: {pbx['closed']}")
            # Executed MR — hidden when MR disabled
            try:
                cfg = self.shared.get('config') or {}
                if not bool(((cfg.get('modes', {}) or {}).get('disable_mr', True))):
                    mrx = exec_stats['mr']
                    mrx_wr = (mrx['wins'] / (mrx['wins'] + mrx['losses']) * 100.0) if (mrx['wins'] + mrx['losses']) else 0.0
                    lines.append("• 🌀 Mean Reversion")
                    lines.append(f"  ✅ W: {mrx['wins']}   ❌ L: {mrx['losses']}   🎯 WR: {mrx_wr:.1f}%")
                    lines.append(f"  🔓 Open: {mrx['open']}   🔒 Closed: {mrx['closed']}")
            except Exception:
                pass
            # Executed Scalp
            scx = exec_stats['scalp']
            scx_wr = (scx['wins'] / (scx['wins'] + scx['losses']) * 100.0) if (scx['wins'] + scx['losses']) else 0.0
            lines.append("• 🩳 Scalp")
            lines.append(f"  ✅ W: {scx['wins']}   ❌ L: {scx['losses']}   🎯 WR: {scx_wr:.1f}%")
            lines.append(f"  🔓 Open: {scx['open']}   🔒 Closed: {scx['closed']}")
        except Exception as exc:
            logger.debug(f"Dashboard summary calc error: {exc}")

        lines.append("")
        lines.append("_Use /status for position details and /ml for full analytics._")

        # Inline UI
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Refresh", callback_data="ui:dash:refresh"),
             InlineKeyboardButton("📊 Symbols", callback_data="ui:symbols:0")],
            [InlineKeyboardButton("🤖 ML", callback_data="ui:ml:main"),
             InlineKeyboardButton("👻 Phantom", callback_data="ui:phantom:main")],
            [InlineKeyboardButton("🧪 Shadow", callback_data="ui:shadow:stats")],
            [InlineKeyboardButton("🩳 Scalp QA", callback_data="ui:scalp:qa"),
            InlineKeyboardButton("📈 Scalp Qscore", callback_data="ui:scalp:qscore")],
            [InlineKeyboardButton("📊 Qscores (All)", callback_data="ui:qscore:all"),
             InlineKeyboardButton("🧠 ML Stats", callback_data="ui:ml:stats")],
            [InlineKeyboardButton("🧱 HTF S/R", callback_data="ui:htf:status"),
             InlineKeyboardButton("🔄 Update S/R", callback_data="ui:htf:update")],
            [InlineKeyboardButton("⚙️ Risk", callback_data="ui:risk:main"),
             InlineKeyboardButton("🧭 Regime", callback_data="ui:regime:main")]
        ])

        return "\n".join(lines), kb

    async def start_polling(self):
        """Start the bot polling without blocking the caller.

        We deliberately run the legacy polling loop in a background task so that
        the trading bot startup (data backfill, WebSocket streams, etc.) can
        continue. Previously this awaited the polling loop directly, which
        meant the main bot `run()` coroutine never progressed beyond Telegram
        startup on some deployments.
        """
        # Idempotent guard
        if not hasattr(self, 'running'):
            self.running = False
        if self.running:
            return

        import asyncio

        async def _runner():
            # Extra diagnostics around the polling lifecycle to debug CancelledError
            try:
                logger.info("Telegram polling runner starting (_legacy_start_polling)...")
            except Exception:
                pass
            try:
                await self._legacy_start_polling()
            except asyncio.CancelledError:
                # Normal during shutdown; Application/Updater will log details
                try:
                    logger.info("Telegram polling runner cancelled (asyncio.CancelledError) — likely during shutdown")
                except Exception:
                    pass
            except Exception as e:
                try:
                    logger.error(f"Telegram polling loop crashed: {e}")
                except Exception:
                    pass
            else:
                try:
                    logger.info("Telegram polling runner completed without error")
                except Exception:
                    pass

        # Mark as running before scheduling to avoid races with concurrent callers
        self.running = True
        try:
            asyncio.create_task(_runner())
        except Exception:
            # If task scheduling fails, reset running flag
            self.running = False
            return

        # Start outbox drain after polling starts
        try:
            asyncio.create_task(self._outbox_drain())
        except Exception:
            pass

    async def _legacy_start_polling(self):
        """Initialize/start Application and start the Updater polling loop."""
        try:
            logger.info("Telegram legacy polling: initialize/start sequence beginning")
        except Exception:
            pass
        try:
            await self.app.initialize()
        except Exception:
            pass
        try:
            await self.app.start()
        except Exception:
            pass
        self.running = True
        try:
            await self.app.bot.delete_webhook(drop_pending_updates=True)
            logger.info("Telegram webhook cleared (drop_pending_updates=true)")
            try:
                import time as _t
                self._last_ok_ts = _t.time()
            except Exception:
                pass
        except Exception as _wh:
            logger.debug(f"Webhook delete skipped: {_wh}")
        # Ensure core commands show up in Telegram clients (helps discoverability)
        try:
            await self._ensure_bot_commands()
            try:
                import time as _t
                self._last_ok_ts = _t.time()
            except Exception:
                pass
        except Exception as _cmd:
            logger.debug(f"set_my_commands skipped: {_cmd}")
        # One-time getUpdates probe (optional) before starting legacy polling
        try:
            cfg = (self.shared or {}).get('config', {}) if hasattr(self, 'shared') else {}
            probe_cfg = ((cfg.get('telegram', {}) or {}).get('probe', {}) or {})
            if bool(probe_cfg.get('get_updates_once', False)):
                limit = int(probe_cfg.get('limit', 3))
                ups = await self.app.bot.get_updates(limit=limit, timeout=10)
                logger.info(f"TG get_updates probe: ok count={len(ups)} (limit={limit})")
                try:
                    import time as _t
                    self._last_ok_ts = _t.time()
                except Exception:
                    pass
                for u in ups[:5]:
                    try:
                        t = type(u).__name__
                        chat = getattr(getattr(u, 'message', None) or getattr(u, 'channel_post', None) or getattr(u, 'edited_message', None), 'chat', None)
                        chat_id = getattr(chat, 'id', None)
                        txt = getattr(getattr(u, 'message', None), 'text', None)
                        logger.info(f"[TG PROBE UPDATE] type={t} chat={chat_id} text={txt}")
                    except Exception:
                        pass
        except Exception as _pe:
            logger.error(f"TG get_updates probe failed: {_pe}")
        logger.info(f"Telegram bot starting updater.polling (chat_id={self.chat_id})")
        await self.app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message","callback_query","channel_post"],
            timeout=30
        )

    async def _outbox_drain(self):
        """Drain outbox messages when Telegram is healthy."""
        import asyncio, time
        while True:
            try:
                if not self._outbox:
                    await asyncio.sleep(2.0)
                    continue
                # Health: shared.network_status == ok OR last_ok within 180s
                healthy = False
                try:
                    st = (self.shared or {}).get('network_status') if hasattr(self, 'shared') else None
                    if st == 'ok':
                        healthy = True
                except Exception:
                    healthy = False
                if not healthy:
                    try:
                        last_ok = float(getattr(self, '_last_ok_ts', 0.0) or 0.0)
                        healthy = (last_ok > 0.0) and ((time.time() - last_ok) < 180.0)
                    except Exception:
                        healthy = False
                if not healthy:
                    await asyncio.sleep(3.0)
                    continue
                item = self._outbox.pop(0)
                text = item.get('text', '')
                reply_markup = item.get('reply_markup')
                try:
                    await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown', reply_markup=reply_markup)
                    try:
                        self._last_ok_ts = time.time()
                    except Exception:
                        pass
                except Exception:
                    # Push back and retry later
                    self._outbox.append(item)
                    await asyncio.sleep(5.0)
                    continue
            except Exception:
                try:
                    await asyncio.sleep(5.0)
                except Exception:
                    pass

    async def _ensure_bot_commands(self):
        """Ensure core commands exist without clobbering existing ones."""
        try:
            from telegram import BotCommand
            # Fetch current commands (best-effort)
            existing = []
            try:
                existing = await self.app.bot.get_my_commands()
            except Exception:
                existing = []
            by_name = {c.command: c for c in existing or []}
            # Ensure minimal set
            need = {
                "help": "Show help",
                "dashboard": "Dashboard",
                "status": "Bot status",
                "scalpcomprehensive": "Scalp comprehensive analysis",
                "scalpultimate": "Ultimate Scalp analysis",
            }
            for k, v in need.items():
                if k not in by_name:
                    by_name[k] = BotCommand(k, v)
            await self.app.bot.set_my_commands(list(by_name.values()))
        except Exception as e:
            logger.debug(f"set_my_commands error: {e}")

    async def _on_error(self, update: object, context):
        try:
            import traceback
            tb = "".join(traceback.format_exception(None, context.error, context.error.__traceback__))
            logger.error(f"[TG ERROR] {type(context.error).__name__}: {context.error}\n{tb}")
        except Exception:
            pass

    async def ping(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            uid = getattr(update.effective_user, 'id', 'n/a')
            cid = getattr(update.effective_chat, 'id', 'n/a')
            logger.info(f"/ping from user={uid} chat={cid}")
            await self.safe_reply(update, "🏓 pong — bot alive")
        except Exception as e:
            logger.error(f"Error in ping: {e}")
            try:
                await update.message.reply_text("❌ ping error")
            except Exception:
                pass

    async def whoami(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Echo chat and user identifiers to help verify configuration."""
        try:
            uid = getattr(update.effective_user, 'id', None)
            cid = getattr(update.effective_chat, 'id', None)
            uname = getattr(getattr(update, 'effective_user', None), 'username', None)
            await self.safe_reply(update, f"👤 user_id: `{uid}`\n💬 chat_id: `{cid}`\n🔤 username: `{uname}`", parse_mode='Markdown')
            logger.info(f"/whoami -> user_id={uid} chat_id={cid} username={uname}")
        except Exception as e:
            logger.error(f"Error in whoami: {e}")

    async def _debug_update(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            t = type(update).__name__
            chat = getattr(update.effective_chat, 'id', None)
            user = getattr(update.effective_user, 'id', None)
            txt = getattr(getattr(update, 'message', None), 'text', None)
            logger.info(f"[TG UPDATE] type={t} chat={chat} user={user} text={txt}")
        except Exception:
            pass

    async def stop(self):
        """Stop the bot"""
        if getattr(self, 'running', False):
            # Application.stop() stops the updater internally; avoid double-stop which can log CancelledError
            try:
                logger.info("Telegram stop() called — stopping Application and shutting down")
            except Exception:
                pass
            await self.app.stop()
            await self.app.shutdown()
            self.running = False
            logger.info("Telegram bot stopped")

    async def send_message(self, text:str, reply_markup=None):
        """Send message to configured chat with retry on network errors"""
        # Notification policy: mute all auto-push notifications except Scalp executions
        try:
            policy = (self.shared or {}).get('notify_policy', 'scalp_execute_only')
        except Exception:
            policy = 'scalp_execute_only'

        if str(policy).lower() == 'scalp_execute_only':
            try:
                t = str(text or '')
                # Allow only Scalp execution/flip notifications; mute all others (Trend/MR/ML/phantom/warnings/startup)
                allow = (
                    ('Scalp EXECUTE' in t) or
                    ('Scalp FLIP' in t)
                )
            except Exception:
                allow = False
            if not allow:
                # Drop silently to keep chat noise-free
                try:
                    logger.debug("TGBot.send_message muted (policy=scalp_execute_only)")
                except Exception:
                    pass
                return

        # Track concurrent sends for flood detection
        self._concurrent_sends += 1
        if self._concurrent_sends > self._max_concurrent_seen:
            self._max_concurrent_seen = self._concurrent_sends
            if self._concurrent_sends >= 20:
                logger.warning(f"⚠️ High Telegram concurrency: {self._concurrent_sends} simultaneous sends (peak: {self._max_concurrent_seen})")

        try:
            max_retries = 5
            base_delay = 1.5

            for attempt in range(max_retries):
                try:
                    # Try with Markdown first
                    await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown', reply_markup=reply_markup)
                    try:
                        import time as _t
                        self._last_ok_ts = _t.time()
                    except Exception:
                        pass
                    return  # Success, exit
                except telegram.error.BadRequest as e:
                    if "can't parse entities" in str(e).lower():
                        # Markdown parsing failed, try with better escaping
                        logger.warning("Markdown parsing failed, trying with escaped text")
                        try:
                            # Escape common problematic characters
                            escaped_text = text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')
                            await self.app.bot.send_message(chat_id=self.chat_id, text=escaped_text, parse_mode='Markdown', reply_markup=reply_markup)
                            try:
                                import time as _t
                                self._last_ok_ts = _t.time()
                            except Exception:
                                pass
                            return  # Success, exit
                        except:
                            # If still fails, send as plain text
                            logger.warning("Escaped markdown also failed, sending as plain text")
                            try:
                                await self.app.bot.send_message(chat_id=self.chat_id, text=text, reply_markup=reply_markup)
                                try:
                                    import time as _t
                                    self._last_ok_ts = _t.time()
                                except Exception:
                                    pass
                                return  # Success, exit
                            except Exception as plain_e:
                                if attempt < max_retries - 1:
                                    import random
                                    delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                                    logger.warning(f"Plain text send failed (attempt {attempt + 1}/{max_retries}): {plain_e} — retrying in {delay:.1f}s")
                                    await asyncio.sleep(delay)
                                    continue
                                else:
                                    logger.error(f"Failed to send message after {max_retries} attempts: {plain_e}")
                    else:
                        logger.error(f"Failed to send message: {e}")
                        return  # Don't retry on non-network errors
                except (telegram.error.NetworkError, telegram.error.TimedOut) as e:
                    # Network-related errors, retry
                    if attempt < max_retries - 1:
                        import random
                        delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                        logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e} — retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Failed to send message after {max_retries} attempts: {e}")
                except Exception as e:
                    # Check if it's a network-related error
                    error_str = str(e).lower()
                    if any(x in error_str for x in ['httpx.readerror', 'network', 'timeout', 'connection']):
                        if attempt < max_retries - 1:
                            import random
                            delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                            logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e} — retrying in {delay:.1f}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"Failed to send message after {max_retries} attempts: {e}")
                    else:
                        logger.error(f"Failed to send message: {e}")
                        return  # Don't retry on non-network errors
        finally:
            # Decrement concurrent send counter
            self._concurrent_sends = max(0, self._concurrent_sends - 1)

    async def send_system_message(self, text: str, reply_markup=None):
        """Bypass notification policy and send a system message with retries.

        Used for startup/ready banners and essential operator prompts.
        """
        # Reuse the same transport but skip the scalp_execute_only filter
        self._concurrent_sends += 1
        if self._concurrent_sends > self._max_concurrent_seen:
            self._max_concurrent_seen = self._concurrent_sends
        try:
            max_retries = 5
            base_delay = 1.5
            for attempt in range(max_retries):
                try:
                    await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown', reply_markup=reply_markup)
                    try:
                        import time as _t
                        self._last_ok_ts = _t.time()
                    except Exception:
                        pass
                    return
                except telegram.error.BadRequest as e:
                    if "can't parse entities" in str(e).lower():
                        try:
                            escaped_text = text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')
                            await self.app.bot.send_message(chat_id=self.chat_id, text=escaped_text, parse_mode='Markdown', reply_markup=reply_markup)
                            return
                        except Exception as plain_e:
                            if attempt < max_retries - 1:
                                import random
                                delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error(f"System send failed: {plain_e}")
                    else:
                        logger.error(f"System send failed: {e}")
                        return
                except (telegram.error.NetworkError, telegram.error.TimedOut) as e:
                    if attempt < max_retries - 1:
                        import random
                        delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"System send failed after retries: {e}")
                except Exception as e:
                    logger.error(f"System send error: {e}")
                    return
        finally:
            self._concurrent_sends = max(0, self._concurrent_sends - 1)

    async def safe_reply(self, update: Update, text: str, parse_mode: str = 'Markdown', reply_markup=None):
        """Safely reply to a message with automatic fallback and retry.

        If update.message is unavailable (e.g., channel posts, callbacks),
        fall back to sending directly to the configured chat.
        """
        # Fast path: when Update has no message (e.g., some callback/channel contexts)
        if getattr(update, 'message', None) is None:
            # Fallback to direct send
            await self.send_message(text, reply_markup=reply_markup)
            return

        max_retries = 5
        base_delay = 1.5
        
        for attempt in range(max_retries):
            try:
                await update.message.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
                try:
                    import time as _t
                    self._last_ok_ts = _t.time()
                except Exception:
                    pass
                return  # Success, exit
            except telegram.error.BadRequest as e:
                if "can't parse entities" in str(e).lower():
                    logger.warning(f"Parse entities failed in reply, trying escaped (mode={parse_mode})")
                    try:
                        # Escape based on parse mode
                        escaped_text = text
                        if parse_mode == 'HTML':
                            # HTML entity escaping
                            import html
                            escaped_text = html.escape(text)
                        else:
                            # Markdown backslash escaping
                            for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                                escaped_text = escaped_text.replace(char, f'\\{char}')
                        await update.message.reply_text(escaped_text, parse_mode=parse_mode, reply_markup=reply_markup)
                        return  # Success, exit
                    except Exception as e2:
                        # Final fallback to plain text
                        logger.warning(f"Escaped markdown also failed ({e2}), replying as plain text")
                        # Remove all markdown formatting
                        plain_text = text
                        for char in ['*', '_', '`', '~']:
                            plain_text = plain_text.replace(char, '')
                        try:
                            await update.message.reply_text(plain_text, parse_mode=None, reply_markup=reply_markup)
                            return  # Success, exit
                        except Exception as plain_e:
                            if attempt < max_retries - 1:
                                import random
                                delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                                logger.warning(f"Plain text reply failed (attempt {attempt + 1}/{max_retries}): {plain_e} — retrying in {delay:.1f}s")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error(f"Failed to reply after {max_retries} attempts: {plain_e}")
                else:
                    # Re-raise other BadRequest errors
                    logger.error(f"BadRequest error: {e}")
                    return
            except (telegram.error.NetworkError, telegram.error.TimedOut) as e:
                # Network-related errors, retry
                if attempt < max_retries - 1:
                    import random
                    delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                    logger.warning(f"Network error in reply (attempt {attempt + 1}/{max_retries}): {e} — retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to reply after {max_retries} attempts: {e}")
            except Exception as e:
                # Check if it's a network-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['httpx.readerror', 'network', 'timeout', 'connection']):
                    if attempt < max_retries - 1:
                        import random
                        delay = min(12.0, base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25)))
                        logger.warning(f"Network error in reply (attempt {attempt + 1}/{max_retries}): {e} — retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Failed to reply after {max_retries} attempts: {e}")
                else:
                    logger.error(f"Failed to reply: {e}")
                    return  # Don't retry on non-network errors

    async def start(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        msg = [
            "🤖 *Trend Pullback Bot*",
            "",
            "🧭 Trend Pullback:\n15m break → 3m HL/LH → 3m 2/2 confirms → entry",
            "Scale‑out: 50% @ ~1.6R, SL→BE, runner to ~3.0R",
            "",
            "📦 Range FBO (phantom‑only now):\nDetect failed breakouts back into range; record phantoms for learning",
            "",
            "Quick actions:",
            "• /dashboard — Dashboard",
            "• /trend_states — Current trend states",
            "• /recent — Recent executed (trend)",
            "• /ml — Trend ML status",
            "• /mlpatterns — Learned patterns",
            "",
            "Tips: Use dashboard buttons for Positions, Phantom (Trend/Range), ML, Events, Settings."
        ]
        await self.safe_reply(update, "\n".join(msg))

    async def watchlist(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current Scalp pre‑eligible and near‑eligible symbols."""
        try:
            wl = (self.shared or {}).get('scalp_watchlist') if hasattr(self, 'shared') else None
            if not wl:
                await self.safe_reply(update, "📋 Scalp watchlist\n\n_No current pre‑eligible symbols_")
                return
            from datetime import datetime
            ts = wl.get('ts', 0)
            when = datetime.utcfromtimestamp(ts).strftime('%H:%M:%S UTC') if ts else 'now'
            pro = wl.get('pro') or []
            combo = wl.get('combo') or []
            near = wl.get('near') or []
            lines = [
                "📋 *Scalp Watchlist*",
                f"_Updated {when}_",
                "",
                f"Pro‑rules pre‑eligible ({len(pro)}):",
                (", ".join(pro[:10]) or "—"),
                "",
                f"Combo‑enabled ({len(combo)}):",
                (", ".join(combo[:10]) or "—"),
                "",
                f"Near‑eligible ({len(near)}):",
                (", ".join(near[:10]) or "—"),
            ]
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in watchlist: {e}")
            await self.safe_reply(update, "❌ Error displaying watchlist")

    async def help(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        per_trade, risk_label = self._compute_risk_snapshot()
        timeframe = self.shared.get("timeframe", "15")
        lines = [
            "📚 *Bot Help*",
            "",
            "Monitoring",
            "• /dashboard — Dashboard",
            "• /execwr — Execution win rates",
            "",
            "Scalp",
            "• /scalpqa — Scalp quality report",
            "• /scalpgates — Gate analysis",
            "• /combo_status — Adaptive combo status",
            "• /scalppro — Pro analytics (RSI/MACD/VWAP/Fib/MTF)",
            "• /combo_threshold <L> [S] — Set LB WR thresholds (long/short)",
            "• /force_adaptive [N] — Force adaptive mode",
            "",
            "Risk",
            "• /risk — Show current risk",
            "• /risk_percent <V> — Set percent risk (e.g., 2.0)",
            "• /risk_usd <V> — Set fixed USD risk (e.g., 50)",
            "• /set_risk <amount> — Flexible (e.g., 3% or 50)",
            "• /scalpriskstate — Scalp adaptive risk state",
            "• /scalprisklimits <base> <min> <max>",
            "• /scalpriskladder <LB:mult,...> (e.g., 40:1.0,50:4.0)",
            "• /manualcombo_risk <pct> — Manual A-tier combo risk %",
            "• /manualcombo_riskusd <usd> — Manual A-tier combo risk in USDT",
            "",
            "Settings (via dashboard → Settings)",
            "• Rule thresholds, stream entry, scale‑out, timeouts",
            "",
            "Other",
            "• /status, /balance, /symbols",
            "• /panicclose SYMBOL",
            "",
            "Current",
            f"• Risk per trade: {risk_label}",
            f"• Timeframe: {timeframe}m + 3m"
        ]
        await self.safe_reply(update, "\n".join(lines))

    async def telemetry(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show lightweight ML/phantom telemetry counters"""
        try:
            tel = self.shared.get('telemetry', {})
            ml_rejects = tel.get('ml_rejects', 0)
            phantom_wins = tel.get('phantom_wins', 0)
            phantom_losses = tel.get('phantom_losses', 0)
            # Strategy-wise phantom stats
            tr_total = tr_wins = tr_active = 0
            mr_total = mr_wins = mr_active = 0
            sc_stats = {'total': 0, 'wins': 0, 'wr': 0.0}
            try:
                from phantom_trade_tracker import get_phantom_tracker
                pt = get_phantom_tracker()
                # Completed rejected
                for trades in pt.phantom_trades.values():
                    for p in trades:
                        if not getattr(p, 'was_executed', False) and getattr(p, 'outcome', None) in ('win','loss'):
                            tr_total += 1
                            if p.outcome == 'win':
                                tr_wins += 1
                tr_active = len(pt.active_phantoms or {})
            except Exception:
                pass
            try:
                from mr_phantom_tracker import get_mr_phantom_tracker
                mrpt = get_mr_phantom_tracker()
                for trades in mrpt.mr_phantom_trades.values():
                    for p in trades:
                        if not getattr(p, 'was_executed', False) and getattr(p, 'outcome', None) in ('win','loss'):
                            mr_total += 1
                            if p.outcome == 'win':
                                mr_wins += 1
                mr_active = len(mrpt.active_mr_phantoms or {})
            except Exception:
                pass
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                sc_stats = scpt.get_scalp_phantom_stats()
            except Exception:
                pass

            def _wr(wins:int, total:int) -> float:
                return (wins/total*100.0) if total else 0.0

            # Flow controller overview
            cfg = self.shared.get('config') or {}
            pf = cfg.get('phantom_flow', {})
            targets = pf.get('daily_target', {'trend':40,'mr':40,'scalp':40})
            # Attempt to read accepted/relax from Redis; fallback to derived
            tr_acc = mr_acc = sc_acc = 0
            tr_relax = mr_relax = sc_relax = 0.0
            try:
                import os, redis
                day = __import__('datetime').datetime.utcnow().strftime('%Y%m%d')
                r = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
                tr_acc = int(r.get(f'phantom:flow:{day}:trend:accepted') or 0)
                mr_acc = int(r.get(f'phantom:flow:{day}:mr:accepted') or 0)
                sc_acc = int(r.get(f'phantom:flow:{day}:scalp:accepted') or 0)
                tr_relax = float(r.get(f'phantom:flow:{day}:trend:relax') or 0.0)
                mr_relax = float(r.get(f'phantom:flow:{day}:mr:relax') or 0.0)
                sc_relax = float(r.get(f'phantom:flow:{day}:scalp:relax') or 0.0)
            except Exception:
                # Fallback to derived acceptances (completed + active phantom-only records today)
                try:
                    day = __import__('datetime').datetime.utcnow().strftime('%Y%m%d')
                    # For PB/MR/Scalp derive accepted today similarly to phantom_qa
                    # Trend accepted today
                    from phantom_trade_tracker import get_phantom_tracker
                    pt = get_phantom_tracker()
                    tr_acc = sum(1 for trades in pt.phantom_trades.values() for p in trades if getattr(p,'signal_time',None) and p.signal_time.strftime('%Y%m%d') == day and not getattr(p,'was_executed', False))
                    tr_acc += sum(1 for p in pt.active_phantoms.values() if p.signal_time.strftime('%Y%m%d') == day)
                except Exception:
                    pass
                try:
                    from mr_phantom_tracker import get_mr_phantom_tracker
                    mrpt = get_mr_phantom_tracker()
                    mr_acc = sum(1 for trades in mrpt.mr_phantom_trades.values() for p in trades if getattr(p,'signal_time',None) and p.signal_time.strftime('%Y%m%d') == day and not getattr(p,'was_executed', False))
                    mr_acc += sum(1 for p in mrpt.active_mr_phantoms.values() if p.signal_time.strftime('%Y%m%d') == day)
                except Exception:
                    pass
                try:
                    from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                    scpt = get_scalp_phantom_tracker()
                    sc_acc = sum(1 for trades in scpt.completed.values() for p in trades if p.signal_time.strftime('%Y%m%d') == day and not getattr(p,'was_executed', False))
                    sc_acc += sum(1 for p in scpt.active.values() if p.signal_time.strftime('%Y%m%d') == day)
                except Exception:
                    pass
                def _relax(accepted:int, tgt:int) -> float:
                    try:
                        h = __import__('datetime').datetime.utcnow().hour
                        pace = tgt * min(1.0, max(1, h)/24.0)
                        deficit = max(0.0, pace - float(accepted))
                        return min(1.0, deficit / max(1.0, tgt*0.5))
                    except Exception:
                        return 0.0
                tr_relax = _relax(tr_acc, int(targets.get('trend',40)))
                mr_relax = _relax(mr_acc, int(targets.get('mr',40)))
                sc_relax = _relax(sc_acc, int(targets.get('scalp',40)))

            # Build output
            trend_ml = self.shared.get('ml_scorer')
            enhanced_mr = self.shared.get('enhanced_mr_scorer')
            lines = [
                "🧪 *Telemetry*",
                f"• ML rejects → phantom: {ml_rejects}",
                f"• Phantom outcomes (rejected): ✅ {phantom_wins} / ❌ {phantom_losses}",
            ]
            if trend_ml:
                try:
                    lines.append(f"• Trend ML threshold: {trend_ml.min_score:.0f}")
                except Exception:
                    pass
            if enhanced_mr:
                try:
                    lines.append(f"• Enhanced MR threshold: {enhanced_mr.min_score:.0f}")
                except Exception:
                    pass

            # Per-strategy phantom WR
            lines.extend([
                "",
                "👻 *Phantom by Strategy*",
                f"• Trend: tracked {tr_total} (+{tr_active} active), WR { (tr_wins/tr_total*100.0) if tr_total else 0.0:.1f}%",
                f"• Mean Reversion: tracked {mr_total} (+{mr_active} active), WR { _wr(mr_wins, mr_total):.1f}%",
                f"• Scalp: tracked {sc_stats.get('total',0)}, WR {sc_stats.get('wr',0.0):.1f}%",
            ])

            # Add executed counts for clarity
            try:
                pb_exec = int(getattr(self.shared.get('ml_scorer'), 'completed_trades', 0)) if self.shared.get('ml_scorer') else 0
            except Exception:
                pb_exec = 0
            try:
                mr_exec = int(getattr(self.shared.get('enhanced_mr_scorer'), 'completed_trades', 0)) if self.shared.get('enhanced_mr_scorer') else 0
            except Exception:
                mr_exec = 0
            lines.extend([
                f"• Executed (Trend/MR): {pb_exec}/{mr_exec}",
            ])

            # Flow controller status
            lines.extend([
                "",
                "🎛️ *Flow Controller*",
                f"• Trend: {tr_acc}/{targets.get('trend',0)} (relax {tr_relax*100:.0f}%)",
                f"• Mean Reversion: {mr_acc}/{targets.get('mr',0)} (relax {mr_relax*100:.0f}%)",
                f"• Scalp: {sc_acc}/{targets.get('scalp',0)} (relax {sc_relax*100:.0f}%)",
            ])

            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in telemetry: {e}")
            await update.message.reply_text("Telemetry unavailable")

    async def show_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show current risk settings"""
        try:
            risk = self.shared["risk"]
            
            # Get current balance if available
            balance_text = ""
            balance = None
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    balance_text = f"💰 *Account Balance:* ${balance:.2f}\n"
            
            if risk.use_percent_risk:
                risk_amount = f"{risk.risk_percent}%"
                if balance_text and balance:
                    usd_amount = balance * (risk.risk_percent / 100)
                    risk_amount += f" (≈${usd_amount:.2f})"
                mode = "Percentage"
            else:
                risk_amount = f"${risk.risk_usd}"
                if balance_text and balance:
                    percent = (risk.risk_usd / balance) * 100
                    risk_amount += f" (≈{percent:.2f}%)"
                mode = "Fixed USD"
            
            # Get scalp-specific risk
            scalp_risk_text = ""
            try:
                bot = self.shared.get('bot_instance')
                if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                    scalp_risk = bot.config.get('scalp', {}).get('exec', {}).get('risk_percent')
                    if scalp_risk:
                        scalp_usd = ""
                        if balance:
                            scalp_usd = f" (≈${balance * (scalp_risk / 100):.2f})"
                        scalp_risk_text = f"\n🩳 *Scalp Risk:* {scalp_risk}%{scalp_usd}\n"
            except Exception:
                pass

            msg = f"""📊 *Risk Management Settings*

{balance_text}⚙️ *Mode:* {mode}
💸 *Risk per trade:* {risk_amount}
📈 *Risk/Reward Ratio:* 1:{risk.rr if hasattr(risk, 'rr') else 2.5}{scalp_risk_text}
*Commands:*
`/risk_percent 2.5` - Set to 2.5%
`/risk_usd 100` - Set to $100
`/set_risk 3%` or `/set_risk 50` - Flexible
`/scalp_set_risk_percent 1.5` - Set scalp risk"""
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in show_risk: {e}")
            await update.message.reply_text("Error fetching risk settings")
    
    async def set_risk_percent(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk as percentage of account"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /risk_percent 2.5")
                return
            
            value = float(ctx.args[0])
            
            # Validate
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than 0%")
                return
            elif value > 10:
                await update.message.reply_text("⚠️ Risk cannot exceed 10% per trade")
                return
            elif value > 5:
                # Warning for high risk
                await update.message.reply_text(
                    f"⚠️ *High Risk Warning*\n\n"
                    f"Setting risk to {value}% is aggressive.\n"
                    f"Confirm with: `/set_risk {value}%`",
                    parse_mode='Markdown'
                )
                return
            
            # Apply the change (global risk used by all strategies)
            self.shared["risk"].risk_percent = value
            self.shared["risk"].use_percent_risk = True
            # Also mirror to per-strategy exec configs so Range/Scalp overrides stay in sync
            try:
                cfg = self.shared.get('config', {})
                cfg.setdefault('range', {}).setdefault('exec', {})['risk_percent'] = value
                cfg.setdefault('scalp', {}).setdefault('exec', {})['risk_percent'] = value
                bot = self.shared.get('bot_instance')
                if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                    bot.config.setdefault('range', {}).setdefault('exec', {})['risk_percent'] = value
                    bot.config.setdefault('scalp', {}).setdefault('exec', {})['risk_percent'] = value
            except Exception:
                pass
            
            # Calculate USD amount if balance available
            usd_info = ""
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    usd_amount = balance * (value / 100)
                    usd_info = f" (≈${usd_amount:.2f} per trade)"
            
            await update.message.reply_text(
                f"✅ Risk updated to {value}% of account{usd_info}\n"
                f"Use `/risk` to view full settings"
            )
            logger.info(f"Risk updated to {value}% via Telegram")
            
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /risk_percent 2.5")
        except Exception as e:
            logger.error(f"Error in set_risk_percent: {e}")
            await update.message.reply_text("Error updating risk percentage")
    
    async def set_risk_usd(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk as fixed USD amount"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /risk_usd 100")
                return
            
            value = float(ctx.args[0])
            
            # Validate
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than $0")
                return
            elif value > 1000:
                await update.message.reply_text("⚠️ Risk cannot exceed $1000 per trade")
                return
            
            # Check if this is too high relative to balance
            percent_info = ""
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    percent = (value / balance) * 100
                    percent_info = f" (≈{percent:.2f}% of account)"
                    
                    if percent > 5:
                        await update.message.reply_text(
                            f"⚠️ *High Risk Warning*\n\n"
                            f"${value} is {percent:.1f}% of your ${balance:.0f} account.\n"
                            f"Confirm with: `/set_risk {value}`",
                            parse_mode='Markdown'
                        )
                        return
            
            # Apply the change
            self.shared["risk"].risk_usd = value
            self.shared["risk"].use_percent_risk = False
            
            await update.message.reply_text(
                f"✅ Risk updated to ${value} per trade{percent_info}\n"
                f"Use `/risk` to view full settings"
            )
            logger.info(f"Risk updated to ${value} fixed via Telegram")
            
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /risk_usd 100")
        except Exception as e:
            logger.error(f"Error in set_risk_usd: {e}")
            await update.message.reply_text("Error updating risk amount")
    
    async def set_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk amount per trade - percentage (1%) or fixed USD (50)"""
        try:
            if not ctx.args:
                # Show current settings if no args
                await self.show_risk(update, ctx)
                return
            
            risk_str = ctx.args[0]
            
            if risk_str.endswith('%'):
                # Percentage-based risk
                v = float(risk_str.rstrip('%'))
                if v <= 0 or v > 10:
                    await update.message.reply_text("Risk percentage must be between 0% and 10%")
                    return
                
                self.shared["risk"].risk_percent = v
                self.shared["risk"].use_percent_risk = True
                await update.message.reply_text(f"✅ Risk set to {v}% of account per trade")
                logger.info(f"Risk updated to {v}%")
            else:
                # Fixed USD risk
                v = float(risk_str)
                if v <= 0 or v > 1000:
                    await update.message.reply_text("Risk must be between $0 and $1000")
                    return
                
                self.shared["risk"].risk_usd = v
                self.shared["risk"].use_percent_risk = False
                await update.message.reply_text(f"✅ Risk set to ${v} per trade")
                logger.info(f"Risk updated to ${v} fixed")
        except ValueError:
            await update.message.reply_text("Invalid amount. Usage: /set_risk 50")
        except Exception as e:
            logger.error(f"Error in set_risk: {e}")
            await update.message.reply_text("Error updating risk")

    async def ml_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Enable/disable ML dynamic risk"""
        try:
            risk = self.shared["risk"]
            
            if not ctx.args:
                # Show current ML risk status
                msg = "🤖 *ML Dynamic Risk Status*\n"
                msg += "━" * 20 + "\n\n"
                
                if risk.use_ml_dynamic_risk:
                    msg += "• Status: ✅ *ENABLED*\n"
                    msg += f"• ML Score Range: {risk.ml_risk_min_score} - {risk.ml_risk_max_score}\n"
                    msg += f"• Risk Range: {risk.ml_risk_min_percent}% - {risk.ml_risk_max_percent}%\n\n"
                    
                    # Show example scaling
                    msg += "📊 *Risk Scaling Examples:*\n"
                    for score in [70, 75, 80, 85, 90, 95, 100]:
                        if score < risk.ml_risk_min_score:
                            continue
                        if score > risk.ml_risk_max_score:
                            score = risk.ml_risk_max_score
                        
                        # Calculate risk using same logic as sizer
                        score_range = risk.ml_risk_max_score - risk.ml_risk_min_score
                        risk_range = risk.ml_risk_max_percent - risk.ml_risk_min_percent
                        if score_range > 0:
                            score_position = (score - risk.ml_risk_min_score) / score_range
                            risk_pct = risk.ml_risk_min_percent + (score_position * risk_range)
                        else:
                            risk_pct = risk.ml_risk_min_percent
                            
                        msg += f"• ML Score {score}: {risk_pct:.2f}% risk\n"
                else:
                    msg += "• Status: ❌ *DISABLED*\n"
                    msg += f"• Fixed Risk: {risk.risk_percent}%\n"
                
                msg += "\nUsage:\n"
                msg += "`/ml_risk on` - Enable ML dynamic risk\n"
                msg += "`/ml_risk off` - Disable ML dynamic risk\n"
                msg += "`/ml_risk_range 1 5` - Set risk range"
                
                await self.safe_reply(update, msg)
                return
            
            action = ctx.args[0].lower()
            
            if action == "on":
                risk.use_ml_dynamic_risk = True
                msg = "✅ ML Dynamic Risk *ENABLED*\n\n"
                msg += f"Risk will scale from {risk.ml_risk_min_percent}% to {risk.ml_risk_max_percent}%\n"
                msg += f"Based on ML scores {risk.ml_risk_min_score} to {risk.ml_risk_max_score}\n\n"
                msg += "_Higher ML confidence = Higher position size_"
                await self.safe_reply(update, msg)
                logger.info("ML dynamic risk enabled")
                
            elif action == "off":
                risk.use_ml_dynamic_risk = False
                msg = "❌ ML Dynamic Risk *DISABLED*\n\n"
                msg += f"All trades will use fixed {risk.risk_percent}% risk"
                await self.safe_reply(update, msg)
                logger.info("ML dynamic risk disabled")
                
            else:
                await update.message.reply_text("Usage: /ml_risk on|off")
                
        except Exception as e:
            logger.error(f"Error in ml_risk: {e}")
            await update.message.reply_text("Error updating ML risk settings")

    async def ml_risk_range(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set ML dynamic risk range"""
        try:
            if len(ctx.args) != 2:
                await update.message.reply_text(
                    "Usage: `/ml_risk_range 1 5`\n"
                    "Sets risk range from 1% to 5%",
                    parse_mode='Markdown'
                )
                return
            
            min_risk = float(ctx.args[0])
            max_risk = float(ctx.args[1])
            
            # Validate
            if min_risk <= 0 or min_risk > 10:
                await update.message.reply_text("Minimum risk must be between 0% and 10%")
                return
            
            if max_risk <= min_risk:
                await update.message.reply_text("Maximum risk must be greater than minimum risk")
                return
                
            if max_risk > 10:
                await update.message.reply_text("Maximum risk must not exceed 10%")
                return
            
            # Update
            risk = self.shared["risk"]
            risk.ml_risk_min_percent = min_risk
            risk.ml_risk_max_percent = max_risk
            
            msg = f"✅ ML Risk Range Updated\n\n"
            msg += f"• Minimum Risk: {min_risk}%\n"
            msg += f"• Maximum Risk: {max_risk}%\n\n"
            
            if risk.use_ml_dynamic_risk:
                msg += "_ML Dynamic Risk is currently ENABLED_"
            else:
                msg += "_ML Dynamic Risk is currently DISABLED_\n"
                msg += "Use `/ml_risk on` to enable"
            
            await self.safe_reply(update, msg)
            logger.info(f"ML risk range updated: {min_risk}% - {max_risk}%")
            
        except ValueError:
            await update.message.reply_text("Invalid values. Use numbers only.")
        except Exception as e:
            logger.error(f"Error in ml_risk_range: {e}")
            await update.message.reply_text("Error updating ML risk range")

    async def scalp_set_risk_percent(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set risk percentage specifically for scalp executions."""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /scalp_set_risk_percent 1.5")
                return

            value = float(ctx.args[0])

            # Validate
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than 0%")
                return

            # Warning for high risk (but allow it)
            warning_msg = ""
            if value > 5:
                warning_msg = f"\n⚠️ Risk >{5}% is aggressive. Use with caution."

            # Update in-memory config
            try:
                bot = self.shared.get('bot_instance')
                if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                    bot.config.setdefault('scalp', {}).setdefault('exec', {})['risk_percent'] = value
                    logger.info(f"Updated bot.config scalp risk to {value}%")
            except Exception as e:
                logger.error(f"Error updating in-memory config: {e}")

            # Write to config.yaml for persistence
            try:
                import yaml
                config_path = 'config.yaml'
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

                config.setdefault('scalp', {}).setdefault('exec', {})['risk_percent'] = value

                # Map to combos[].risk_percent if defined
                try:
                    combos = config.get('scalp', {}).get('exec', {}).get('combos')
                    if isinstance(combos, list) and combos:
                        for c in combos:
                            if int(c.get('id', 0)) == 1:
                                c['risk_percent'] = float(base_risk * combo1_multiplier)
                            else:
                                c['risk_percent'] = float(base_risk)
                except Exception:
                    pass

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                logger.info(f"Wrote scalp risk {value}% to config.yaml")
            except Exception as e:
                logger.error(f"Error writing to config.yaml: {e}")
                await update.message.reply_text(
                    f"✅ Scalp risk updated to {value}% (in-memory only)\n"
                    f"⚠️ Could not save to config.yaml: {e}{warning_msg}"
                )
                return

            # Calculate USD amount if balance available
            usd_info = ""
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    usd_amount = balance * (value / 100)
                    usd_info = f" (≈${usd_amount:.2f} per trade)"

            await update.message.reply_text(
                f"✅ Scalp execution risk updated to {value}%{usd_info}{warning_msg}\n"
                f"Use `/scalp_get_risk` to view scalp settings"
            )
            logger.info(f"Scalp risk updated to {value}% via Telegram")

        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /scalp_set_risk_percent 1.5")
        except Exception as e:
            logger.error(f"Error in scalp_set_risk_percent: {e}")
            await update.message.reply_text("Error updating scalp risk percentage")

    async def scalp_get_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current scalp execution risk settings."""
        try:
            # Get from bot config
            risk_percent = None
            try:
                bot = self.shared.get('bot_instance')
                if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                    risk_percent = bot.config.get('scalp', {}).get('exec', {}).get('risk_percent')
            except Exception:
                pass

            # Fallback to config.yaml
            if risk_percent is None:
                try:
                    import yaml
                    with open('config.yaml', 'r') as f:
                        config = yaml.safe_load(f) or {}
                    risk_percent = config.get('scalp', {}).get('exec', {}).get('risk_percent')
                except Exception:
                    pass

            if risk_percent is None:
                await update.message.reply_text("⚠️ Scalp risk not configured")
                return

            # Calculate USD amount if balance available
            usd_info = ""
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    usd_amount = balance * (risk_percent / 100)
                    usd_info = f"\n• USD per trade: ≈${usd_amount:.2f}"

            msg = "🩳 *Scalp Execution Risk*\n"
            msg += "━" * 20 + "\n\n"
            msg += f"• Risk: {risk_percent}%{usd_info}\n\n"
            msg += f"_Use `/scalp_set_risk_percent` to change risk_"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in scalp_get_risk: {e}")
            await update.message.reply_text("Error retrieving scalp risk settings")

    async def manual_combo_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set risk percentage for manual A-tier combo execution."""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /manualcombo_risk 1.0")
                return

            value = float(ctx.args[0])
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than 0%")
                return

            # Update in-memory config
            try:
                bot = self.shared.get('bot_instance')
                if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                    mce = bot.config.setdefault('scalp', {}).setdefault('exec', {}).setdefault('manual_combo_exec', {})
                    mce['risk_mode'] = 'percent'
                    mce['risk_percent'] = float(value)
                    logger.info(f"Updated manual_combo_exec risk_percent to {value}%")
            except Exception as e:
                logger.error(f"Error updating in-memory manual combo risk: {e}")

            # Persist to config.yaml
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                mce_cfg = cfg.setdefault('scalp', {}).setdefault('exec', {}).setdefault('manual_combo_exec', {})
                mce_cfg['risk_mode'] = 'percent'
                mce_cfg['risk_percent'] = float(value)
                with open('config.yaml', 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                logger.error(f"Error writing manual combo risk to config.yaml: {e}")

            # Approximate USD per trade using current balance if available
            usd_info = ""
            try:
                broker = self.shared.get("broker")
                if broker and hasattr(broker, "get_balance"):
                    bal = broker.get_balance()
                    if bal:
                        usd_amount = float(bal) * (value / 100.0)
                        usd_info = f" (≈${usd_amount:.2f} per trade)"
            except Exception:
                pass

            await update.message.reply_text(
                f"✅ Manual combo risk set to {value:.2f}%{usd_info}\n"
                f"_Pattern: RSI:40-60 MACD:bull VWAP:1.2+ Fib:50-61 noMTF_"
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /manualcombo_risk 1.0")
        except Exception as e:
            logger.error(f"Error in manual_combo_risk: {e}")
            await update.message.reply_text("Error updating manual combo risk %")

    async def manual_combo_risk_usd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set fixed USD risk for manual A-tier combo execution."""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /manualcombo_riskusd 5")
                return

            value = float(ctx.args[0])
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than 0 USDT")
                return

            # Update in-memory config
            try:
                bot = self.shared.get('bot_instance')
                if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                    mce = bot.config.setdefault('scalp', {}).setdefault('exec', {}).setdefault('manual_combo_exec', {})
                    mce['risk_mode'] = 'usd'
                    mce['risk_usd'] = float(value)
                    logger.info(f"Updated manual_combo_exec risk_usd to {value} USDT")
            except Exception as e:
                logger.error(f"Error updating in-memory manual combo USD risk: {e}")

            # Persist to config.yaml
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                mce_cfg = cfg.setdefault('scalp', {}).setdefault('exec', {}).setdefault('manual_combo_exec', {})
                mce_cfg['risk_mode'] = 'usd'
                mce_cfg['risk_usd'] = float(value)
                with open('config.yaml', 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                logger.error(f"Error writing manual combo USD risk to config.yaml: {e}")

            await update.message.reply_text(
                f"✅ Manual combo risk set to ≈${value:.2f} per trade\n"
                f"_Pattern: RSI:40-60 MACD:bull VWAP:1.2+ Fib:50-61 noMTF_"
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /manualcombo_riskusd 5")
        except Exception as e:
            logger.error(f"Error in manual_combo_risk_usd: {e}")
            await update.message.reply_text("Error updating manual combo USD risk")

    async def ml90_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Get or set ML≥90 bypass risk percentage"""
        try:
            # Get current value (default 0.5%)
            current_risk = self.shared.get("ml90_bypass_risk", 0.5)

            if not ctx.args:
                # Show current setting
                msg = "🌟 *ML≥90 Bypass Risk*\n"
                msg += "━" * 20 + "\n\n"
                msg += f"• Current Risk: *{current_risk}%*\n"
                msg += f"• Applies to: ML scores ≥90\n"
                msg += f"• Historical WR: ~79%\n"
                msg += f"  (90-99: 80.0% N=5 | 100-109: 78.8% N=132)\n\n"
                msg += "Usage:\n"
                msg += "`/ml90_risk 0.5` - Set to 0.5%\n"
                msg += "`/ml90_risk 1.0` - Set to 1.0%\n\n"
                msg += "⚠️ *Note:* ML≥90 trades bypass ALL gates\n"
                msg += "and execute immediately with this risk %"

                await self.safe_reply(update, msg)
                return

            # Set new value
            new_risk = float(ctx.args[0])

            # Validate
            if new_risk <= 0:
                await update.message.reply_text("Risk must be greater than 0%")
                return

            if new_risk > 5:
                await update.message.reply_text("Risk must not exceed 5% for ML≥90 bypass")
                return

            # Update
            self.shared["ml90_bypass_risk"] = new_risk

            msg = f"✅ ML≥90 Bypass Risk Updated\n\n"
            msg += f"• New Risk: *{new_risk}%*\n"
            msg += f"• Applies to: All ML scores ≥90\n\n"
            msg += f"_All ML≥90 signals will now execute with {new_risk}% risk_"

            await self.safe_reply(update, msg)
            logger.info(f"ML≥90 bypass risk updated to {new_risk}%")

        except ValueError:
            await update.message.reply_text("Invalid value. Usage: /ml90_risk 0.5")
        except Exception as e:
            logger.error(f"Error in ml90_risk: {e}")
            await update.message.reply_text("Error updating ML≥90 bypass risk")

    async def status(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show current positions"""
        try:
            book = self.shared.get("book")
            positions = book.positions if book else {}
            frames = self.shared.get("frames", {})
            per_trade_risk, risk_label = self._compute_risk_snapshot()

            header = ["📊 *Open Positions*", "━━━━━━━━━━━━━━━━━━━━━━"]

            if not positions:
                header.append("")
                header.append("No open positions.")
                header.append(f"*Risk per trade:* {risk_label}")
                header.append(f"*Symbols scanning:* {len(frames)}")
                await self.safe_reply(update, "\n".join(header))
                return

            now = datetime.now(timezone.utc)
            total_pnl = 0.0
            lines = header + [""]

            for idx, (sym, pos) in enumerate(positions.items()):
                if idx >= 10:
                    lines.append(f"…and {len(positions) - idx} more symbols")
                    break

                emoji = "🟢" if pos.side == "long" else "🔴"
                lines.append(f"{emoji} *{sym}* ({pos.side.upper()})")

                strategy = getattr(pos, 'strategy_name', 'unknown')
                if isinstance(strategy, str):
                    strategy = strategy.replace('_', ' ').title()
                lines.append(f"  Strategy: {strategy}")
                lines.append(f"  Entry: {pos.entry:.4f} | Size: {pos.qty}")
                lines.append(f"  SL: {pos.sl:.4f} | TP: {pos.tp:.4f}")

                # Hold duration
                if getattr(pos, 'entry_time', None):
                    entry_time = pos.entry_time
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    held_minutes = max(0, int((now - entry_time).total_seconds() // 60))
                    lines.append(f"  Held: {held_minutes}m")

                # Live PnL snapshot
                if sym in frames and len(frames[sym]) > 0:
                    current_price = frames[sym]['close'].iloc[-1]
                    if pos.side == "long":
                        pnl = (current_price - pos.entry) * pos.qty
                        pnl_pct = ((current_price - pos.entry) / pos.entry) * 100
                    else:
                        pnl = (pos.entry - current_price) * pos.qty
                        pnl_pct = ((pos.entry - current_price) / pos.entry) * 100
                    total_pnl += pnl
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    lines.append(f"  P&L: {pnl_emoji} ${pnl:.2f} ({pnl_pct:+.2f}%)")

                lines.append("")

            total_positions = len(positions)
            estimated_risk = per_trade_risk * total_positions
            lines.append(f"*Positions:* {total_positions} | *Estimated risk:* ${estimated_risk:.2f}")
            lines.append(f"*Risk per trade:* {risk_label}")
            lines.append(f"*Unrealised P&L:* ${total_pnl:.2f}")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.exception("Error in status: %s", e)
            await update.message.reply_text("Error getting status")

    async def balance(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show account balance"""
        try:
            broker = self.shared.get("broker")
            if broker:
                balance = broker.get_balance()
                if balance:
                    self.shared["last_balance"] = balance
                    await self.safe_reply(update, f"💰 *Balance:* ${balance:.2f} USDT")
                else:
                    await update.message.reply_text("Unable to fetch balance")
            else:
                await update.message.reply_text("Broker not initialized")
        except Exception as e:
            logger.exception("Error in balance: %s", e)
            await update.message.reply_text("Error getting balance")

    async def panic_close(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Emergency close position"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /panic_close BTCUSDT")
                return
                
            sym = ctx.args[0].upper()
            
            if sym not in self.shared["book"].positions:
                await update.message.reply_text(f"No position found for {sym}")
                return
                
            self.shared["panic"].append(sym)
            await update.message.reply_text(
                f"⚠️ *Panic close requested for {sym}*\n"
                f"Position will be closed at next tick.",
                parse_mode='Markdown'
            )
            logger.warning(f"Panic close requested for {sym}")
            
        except Exception as e:
            logger.error(f"Error in panic_close: {e}")
            await update.message.reply_text("Error processing panic close")
    
    async def health(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show bot health and analysis status"""
        try:
            import datetime
            
            # Get shared data
            frames = self.shared.get("frames", {})
            last_analysis = self.shared.get("last_analysis", {})
            
            msg = "*🤖 Bot Health Status*\n\n"
            
            # Check if bot is receiving data
            if frames:
                msg += "✅ *Data Reception:* Active\n"
                msg += f"📊 *Symbols Tracked:* {len(frames)}\n\n"
                
                msg += "*Candle Data:*\n"
                for symbol, df in list(frames.items())[:5]:  # Show first 5
                    if df is not None and len(df) > 0:
                        last_time = df.index[-1]
                        candle_count = len(df)
                        msg += f"• {symbol}: {candle_count} candles, last: {last_time.strftime('%H:%M:%S')}\n"
                
                # Show last analysis times
                if last_analysis:
                    msg += "\n*Last Analysis:*\n"
                    now = datetime.datetime.now()
                    for sym, timestamp in list(last_analysis.items())[:5]:
                        time_ago = (now - timestamp).total_seconds()
                        if time_ago < 60:
                            msg += f"• {sym}: {int(time_ago)}s ago\n"
                        else:
                            msg += f"• {sym}: {int(time_ago/60)}m ago\n"
                else:
                    msg += "\n⏳ *Waiting for first candle close to analyze*"
            else:
                msg += "⚠️ *Data Reception:* No data yet\n"
                msg += "Bot is starting up..."
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in health: {e}")
            await update.message.reply_text("Error getting health status")
    
    async def symbols(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show list of active trading symbols"""
        try:
            frames = self.shared.get("frames", {})
            configured = self.shared.get("symbols_config")

            if configured:
                symbols_list = list(configured)
            elif frames:
                symbols_list = list(frames.keys())
            else:
                await update.message.reply_text("No symbols loaded yet")
                return

            msg = "📈 *Active Trading Pairs*\n\n"

            # Show in groups of 5
            for i in range(0, len(symbols_list), 5):
                group = symbols_list[i:i+5]
                msg += " • ".join(group) + "\n"
            
            msg += f"\n*Total:* {len(symbols_list)} symbols"
            timeframe = self.shared.get("timeframe")
            if timeframe:
                msg += f"\n*Timeframe:* {timeframe} minutes"
            msg += "\n*Strategies:* Trend ML / Mean Reversion"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.exception("Error in symbols: %s", e)
            await update.message.reply_text("Error getting symbols")
    
    async def dashboard(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show complete bot dashboard"""
        try:
            # Best-effort: refresh closed positions before computing snapshots so counts reflect latest state
            try:
                bot = self.shared.get('bot_instance')
                book = self.shared.get('book')
                meta = self.shared.get('meta')
                ml_scorer = self.shared.get('ml_scorer')
                if bot and book:
                    await bot.check_closed_positions(book, meta, ml_scorer, None, None)
            except Exception:
                pass
            # simple rate-limit to avoid Telegram flood control on heavy dashboard
            if not self._cooldown_ok('dashboard'):
                await self.safe_reply(update, "⏳ Please wait before using /dashboard again")
                return
            # Pick dashboard based on enabled strategies; show Scalp-only when others are disabled
            cfg = self.shared.get('config', {}) or {}
            trend_enabled = bool(((cfg.get('trend', {}) or {}).get('enabled', True)))
            range_enabled = bool(((cfg.get('range', {}) or {}).get('enabled', True)))
            if (not trend_enabled) and (not range_enabled):
                text, kb = self._build_scalp_dashboard()
            else:
                # Default to Trend-centric dashboard when any other strategy is active
                text, kb = self._build_trend_dashboard()

            # Send message with keyboard buttons via robust safe_reply
            await self.safe_reply(update, text, parse_mode='Markdown', reply_markup=kb)

        except Exception as e:
            logger.exception("Error in dashboard: %s", e)
            await self.safe_reply(update, "Error getting dashboard")

    async def ui_callback(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Handle inline UI callbacks"""
        try:
            query = update.callback_query
            data = query.data or ""
            # Trend-only UI routes (take precedence)
            if data in ("ui:dash:refresh", "ui:dash:refresh:trend"):
                await query.answer("Refreshing…")
                text, kb = self._build_trend_dashboard()
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception:
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                return
            if data == "ui:exec:wr":
                # Provide quick tactile feedback and post a fresh Exec WR panel
                try:
                    await query.answer("Computing…")
                except Exception:
                    await query.answer()
                await self.exec_winrates(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:dash:refresh:scalp":
                await query.answer("Refreshing…")
                text, kb = self._build_scalp_dashboard(False)
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception as _e_edit:
                    logger.debug(f"ui:dash:refresh:scalp edit failed ({_e_edit}), sending new message")
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                return
            if data == "ui:dash:more":
                try:
                    await query.answer("More…")
                except Exception:
                    await query.answer()
                text, kb = self._build_scalp_dashboard(True)
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception as _e_edit:
                    logger.debug(f"ui:dash:more edit failed ({_e_edit}), sending new message")
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                return
            if data == "ui:dash:less":
                try:
                    await query.answer("Back…")
                except Exception:
                    await query.answer()
                text, kb = self._build_scalp_dashboard(False)
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception as _e_edit:
                    logger.debug(f"ui:dash:less edit failed ({_e_edit}), sending new message")
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                return
            if data == "ui:scalp:qa":
                await query.answer()
                await self.scalp_qa(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:gates":
                await query.answer()
                await self.scalp_gate_analysis(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:risk":
                try:
                    await query.answer("Loading risk…")
                except Exception:
                    await query.answer()
                text, kb = self._build_scalp_dashboard(False)
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception:
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                return
            if data == "ui:scalp:combos":
                try:
                    await query.answer("Loading combos…")
                except Exception:
                    await query.answer()
                text, kb = self._build_scalp_dashboard(False)
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception:
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                return
            if data == "ui:scalp:comp":
                await query.answer()
                await self.scalp_comprehensive_analysis(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:patterns":
                await query.answer()
                await self.scalp_patterns(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:qwr":
                await query.answer()
                await self.scalp_qscore_wr(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:mlwr":
                await query.answer()
                await self.scalp_mlscore_wr(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:timewr":
                await query.answer()
                await self.scalp_time_wr(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:emaslopes":
                await query.answer()
                await self.scalp_ema_slopes(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:open_phantoms":
                await query.answer()
                await self.scalp_open_phantoms(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:slopesevci":
                await query.answer()
                await self.scalp_ema_slopes_evci(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:advancedcombos":
                await query.answer()
                await self.scalp_advanced_combos(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:combosevci":
                await query.answer()
                await self.scalp_advanced_combos_evci(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:ultimate":
                await query.answer()
                await self.scalp_ultimate(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:proanalytics":
                await query.answer()
                await self.scalp_pro_analytics(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:buildstrategy":
                await query.answer()
                await self.scalp_build_strategy(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data == "ui:scalp:gatefeat":
                await query.answer()
                await self.scalp_gate_feature_combos(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data.startswith("ui:scalp:timewr_vars:session:"):
                await query.answer()
                key = data.split(":")[-1]
                await self._scalp_time_vars(type('obj', (object,), {'message': query.message}), ctx, kind='session', key=key)
                return
            if data.startswith("ui:scalp:timewr_vars:day:"):
                await query.answer()
                key = data.split(":")[-1]
                await self._scalp_time_vars(type('obj', (object,), {'message': query.message}), ctx, kind='day', key=key)
                return
            if data == "ui:scalp:promote":
                await query.answer()
                await self.scalp_promotion_status(type('obj', (object,), {'message': query.message}), ctx)
                return
            if data in ("ui:trend:states", "ui:positions", "ui:recent", "ui:phantom:trend", "ui:phantom:range", "ui:ml:trend", "ui:ml:patterns", "ui:events", "ui:settings") or data.startswith("ui:settings:toggle:") or data.startswith("ui:settings:set:"):
                # Re-dispatch into the simpler trend-only handlers by faking a small switch
                if data == "ui:trend:states":
                    await query.answer()
                    try:
                        from strategy_pullback import get_trend_states_snapshot
                        snap = get_trend_states_snapshot() or {}
                        lines = ["📐 *Trend States*", ""]
                        for sym, st in sorted(snap.items()):
                            try:
                                s = st.get('state','?')
                                executed = bool(st.get('executed', False))
                                # Display mapping: avoid 'SENT' unless executed
                                disp_state = s
                                if s == 'SIGNAL_SENT' and not executed:
                                    disp_state = 'SIGNAL'
                                bl = float(st.get('breakout_level', 0.0) or 0.0)
                                conf = int(st.get('confirm_progress', 0) or 0)
                                micro = st.get('micro_state') or ""
                                pivot = float(st.get('last_counter_pivot') or 0.0)
                                if s in ("RESISTANCE_BROKEN", "HL_FORMED"):
                                    pivot_label = "LH"
                                elif s in ("SUPPORT_BROKEN", "LH_FORMED"):
                                    pivot_label = "HL"
                                else:
                                    pivot_label = "PV"
                                parts = [f"{disp_state}"]
                                if micro:
                                    parts.append(micro)
                                if pivot > 0:
                                    parts.append(f"{pivot_label}={pivot:.4f}")
                                try:
                                    div_ok = st.get('divergence_ok', False)
                                    div_type = st.get('divergence_type','NONE')
                                    div_score = float(st.get('divergence_score',0.0) or 0.0)
                                    parts.append(f"Div={'✅' if div_ok else '—'} {div_type}{(' '+str(round(div_score,2))) if div_score else ''}")
                                except Exception:
                                    pass
                                # Gates and execution indicators
                                try:
                                    gates = st.get('gates', {}) or {}
                                    htf_ok = gates.get('htf_ok')
                                    sr_ok = gates.get('sr_ok')
                                    parts.append(
                                        f"Gates: HTF={'✅' if htf_ok else ('—' if htf_ok is not None else '?')} SR={'✅' if sr_ok else ('—' if sr_ok is not None else '?')}"
                                    )
                                except Exception:
                                    pass
                                if executed:
                                    parts.append('EXEC')
                                try:
                                    if st.get('bos_crossed'):
                                        wr = st.get('waiting_reason') or 'WAIT_PIVOT'
                                        parts.append(f"BOS=armed({wr})")
                                except Exception:
                                    pass
                                state_line = " | ".join(parts)
                                lines.append(f"• {sym}: {state_line} | lvl {bl:.4f} | conf {conf}")
                            except Exception:
                                lines.append(f"• {sym}: {st}")
                        text = "\n".join(lines)
                        # Handle Telegram 4096-char limit by truncating and hinting
                        if len(text) > 3800:
                            # Keep header + first ~40 items
                            header = lines[:2]
                            body = lines[2:42]
                            rest = len(lines) - len(body) - 2
                            body.append(f"… (+{rest} more, use /trend_states)")
                            text = "\n".join(header + body)
                        try:
                            await query.edit_message_text(text, parse_mode='Markdown')
                        except Exception:
                            # Fallback: send as a new message (avoid losing content)
                            try:
                                await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
                            except Exception:
                                await query.edit_message_text("📐 Trend states unavailable", parse_mode='Markdown')
                    except Exception:
                        await query.edit_message_text("📐 Trend states unavailable", parse_mode='Markdown')
                    return
                if data == "ui:positions":
                    await query.answer()
                    book = self.shared.get('book')
                    positions = (book.positions if book else {})
                    lines = ["📊 *Open Positions*", ""]
                    if not positions:
                        lines.append("None")
                    else:
                        for sym, p in positions.items():
                            try:
                                be = ''
                                so = getattr(self.shared.get('bot_instance', None), '_scaleout', {}) if self.shared.get('bot_instance') else {}
                                if isinstance(so, dict) and sym in so and so.get('be_moved'):
                                    be = ' | BE moved'
                                lines.append(f"• {sym} {p.side.upper()} qty={p.qty} entry={p.entry:.4f} sl={p.sl:.4f} tp={p.tp:.4f}{be}")
                            except Exception:
                                continue
                    await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:recent":
                    await query.answer()
                    tt = self.shared.get('trade_tracker')
                    rec = getattr(tt, 'trades', []) or []
                    rec = [t for t in rec if 'trend' in (getattr(t, 'strategy_name', '') or '').lower()]
                    rec.sort(key=lambda t: getattr(t, 'exit_time', None) or getattr(t, 'entry_time', None), reverse=True)
                    last10 = rec[:10]
                    lines = ["📜 *Recent Executed (Trend)*", ""]
                    for t in last10:
                        try:
                            sym = t.symbol; side = t.side
                            et = t.entry_time.strftime('%m-%d %H:%M') if t.entry_time else ''
                            out = f" | PnL ${float(t.pnl_usd):.2f}" if getattr(t, 'exit_price', None) else ''
                            lines.append(f"• {sym} {side.upper()} {et}{out}")
                        except Exception:
                            continue
                    if len(lines) <= 2:
                        lines.append("No trades yet")
                    await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:phantom:trend":
                    await query.answer()
                    pt = self.shared.get('phantom_tracker')
                    total = wins = losses = timeouts = 0
                    open_cnt = 0
                    if pt:
                        for trades in getattr(pt, 'phantom_trades', {}).values():
                            for p in trades:
                                if (getattr(p, 'strategy_name', '') or '').startswith('trend'):
                                    oc = getattr(p, 'outcome', None)
                                    if oc in ('win','loss'):
                                        total += 1
                                        wins += (1 if oc == 'win' else 0)
                                        losses += (1 if oc == 'loss' else 0)
                                    if (not getattr(p, 'was_executed', False)) and getattr(p, 'exit_reason', None) == 'timeout':
                                        timeouts += 1
                        # Count open (in-flight) trend phantoms
                        try:
                            for lst in getattr(pt, 'active_phantoms', {}).values():
                                for p in (lst or []):
                                    if (getattr(p, 'strategy_name', '') or '').startswith('trend'):
                                        if not getattr(p, 'exit_time', None):
                                            open_cnt += 1
                        except Exception:
                            pass
                    wr = (wins/total*100.0) if total else 0.0
                    lines = ["👻 *Trend Phantom*", "", f"Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}"]
                    await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:phantom:range":
                    await query.answer()
                    pt = self.shared.get('phantom_tracker')
                    total = wins = losses = timeouts = 0
                    open_cnt = 0
                    if pt:
                        for trades in getattr(pt, 'phantom_trades', {}).values():
                            for p in trades:
                                if (getattr(p, 'strategy_name', '') or '').startswith('range'):
                                    oc = getattr(p, 'outcome', None)
                                    if oc in ('win','loss'):
                                        total += 1
                                        wins += (1 if oc == 'win' else 0)
                                        losses += (1 if oc == 'loss' else 0)
                                    if (not getattr(p, 'was_executed', False)) and getattr(p, 'exit_reason', None) == 'timeout':
                                        timeouts += 1
                        # Count open (in-flight) range phantoms
                        try:
                            for lst in getattr(pt, 'active_phantoms', {}).values():
                                for p in (lst or []):
                                    if (getattr(p, 'strategy_name', '') or '').startswith('range'):
                                        if not getattr(p, 'exit_time', None):
                                            open_cnt += 1
                        except Exception:
                            pass
                    wr = (wins/total*100.0) if total else 0.0
                    lines = ["📦 *Range Phantom*", "", f"Tracked: {total} | Open: {open_cnt} | WR: {wr:.1f}% (W/L {wins}/{losses}) | Timeouts: {timeouts}"]
                    await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:ml:trend":
                    await query.answer()
                    ml = self.shared.get('ml_scorer')
                    if not ml:
                        await query.edit_message_text("🤖 Trend ML unavailable", parse_mode='Markdown')
                    else:
                        st = ml.get_stats()
                        lines = ["🤖 *Trend ML*", "",
                                 f"Status: {st.get('status','⏳')}",
                                 f"Threshold: {st.get('current_threshold','?'):.0f}",
                                 f"Completed trades: {st.get('completed_trades',0)}",
                                 f"Recent WR: {st.get('recent_win_rate',0.0):.1f}%"]
                        await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:ml:patterns":
                    await query.answer()
                    ml = self.shared.get('ml_scorer')
                    if not ml:
                        await query.edit_message_text("🧠 Patterns unavailable", parse_mode='Markdown')
                    else:
                        pats = ml.get_learned_patterns() or {}
                        fi = pats.get('feature_importance', {})
                        lines = ["🧠 *Trend ML Patterns*", ""]
                        if fi:
                            for k, v in list(fi.items())[:8]:
                                lines.append(f"• {k}: {float(v):.1f}%")
                        else:
                            lines.append("Collecting data…")
                        await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:events":
                    await query.answer()
                    evts = self.shared.get('trend_events') or []
                    lines = ["🧭 *Recent State Activity*", ""]
                    if not evts:
                        lines.append("No recent events")
                    else:
                        for e in evts[-30:]:
                            sym = e.get('symbol','?'); txt = e.get('text','')
                            lines.append(f"• {sym}: {txt}")
                    await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
                    return
                if data == "ui:settings":
                    await query.answer()
                    cfg = self.shared.get('config', {}) or {}
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    sc = (tr_exec.get('scaleout',{}) or {})
                    div = (tr_exec.get('divergence',{}) or {})
                    # Read current RR and timeouts
                    rr_val = None
                    conf_bars = None
                    try:
                        rr_val = float(self.shared.get('trend_settings').rr)
                        conf_bars = int(self.shared.get('trend_settings').confirmation_timeout_bars)
                    except Exception:
                        pass
                    ph = (cfg.get('phantom', {}) or {})
                    # HTF min trend strength & SL mode/buffer
                    try:
                        htf_cfg = ((cfg.get('router', {}) or {}).get('htf_bias', {}) or {})
                        min_ts = float((htf_cfg.get('trend', {}) or {}).get('min_trend_strength', 60.0))
                    except Exception:
                        min_ts = 60.0
                    try:
                        sl_mode = str((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('sl_mode', 'breakout')))
                        sl_buf = float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('breakout_sl_buffer_atr', 0.30)))
                    except Exception:
                        sl_mode = 'breakout'; sl_buf = 0.30
                    htf_gate = ((((cfg.get('trend',{}) or {}).get('exec',{}) or {}).get('htf_gate',{}) or {}))
                    # Rule-mode thresholds snapshot (Trend/Range/Scalp)
                    try:
                        rm_tr = ((cfg.get('trend', {}) or {}).get('rule_mode', {}) or {})
                        tr_exec_q = float(rm_tr.get('execute_q_min', 78))
                        tr_ph_q = float(rm_tr.get('phantom_q_min', 65))
                    except Exception:
                        tr_exec_q = 78.0; tr_ph_q = 65.0
                    try:
                        rm_rg = ((cfg.get('range', {}) or {}).get('rule_mode', {}) or {})
                        rg_exec_q = float(rm_rg.get('execute_q_min', 78))
                        rg_ph_q = float(rm_rg.get('phantom_q_min', 65))
                    except Exception:
                        rg_exec_q = 78.0; rg_ph_q = 65.0
                    try:
                        rm_sc = ((cfg.get('scalp', {}) or {}).get('rule_mode', {}) or {})
                        sc_exec_q = float(rm_sc.get('execute_q_min', 60))
                        sc_ph_q = float(rm_sc.get('phantom_q_min', 80))
                    except Exception:
                        sc_exec_q = 88.0; sc_ph_q = 80.0
                    # Adapters status
                    try:
                        ad_tr = bool(((cfg.get('trend', {}) or {}).get('rule_mode', {}) or {}).get('adapter_enabled', True))
                    except Exception:
                        ad_tr = True
                    try:
                        ad_rg = bool(((cfg.get('range', {}) or {}).get('rule_mode', {}) or {}).get('adapter_enabled', True))
                    except Exception:
                        ad_rg = True
                    try:
                        ad_sc = bool(((cfg.get('scalp', {}) or {}).get('rule_mode', {}) or {}).get('adapter_enabled', True))
                    except Exception:
                        ad_sc = True
                    lines = ["⚙️ *Settings*", "",
                             f"Rule‑Mode thresholds:",
                             f"• Trend: Exec≥{tr_exec_q:.0f} | Phantom≥{tr_ph_q:.0f}",
                             f"• Range: Exec≥{rg_exec_q:.0f} | Phantom≥{rg_ph_q:.0f}",
                             f"• Scalp: Exec≥{sc_exec_q:.0f} | Phantom≥{sc_ph_q:.0f}",
                             f"Adapters: Trend {'On' if ad_tr else 'Off'} | Range {'On' if ad_rg else 'Off'} | Scalp {'On' if ad_sc else 'Off'}",
                             f"Stream entry: {'On' if tr_exec.get('allow_stream_entry', True) else 'Off'}",
                             f"Scale‑out: {'On' if sc.get('enabled', False) else 'Off'}",
                             f"BE move: {'On' if sc.get('move_sl_to_be', True) else 'Off'}",
                             f"R:R: 1:{rr_val if rr_val is not None else '2.5'}",
                             f"TP1 R: {sc.get('tp1_r',1.6)} | TP2 R: {sc.get('tp2_r',3.0)} | Fraction: {sc.get('fraction',0.5):.2f}",
                             f"Confirm timeout bars: {conf_bars if conf_bars is not None else '6'} | Phantom timeout h: {(getattr(self.shared.get('phantom_tracker'), 'timeout_hours', 0) or 0)}",
                             # HL/LH 3m timeouts from live trend_settings
                             (lambda ts=self.shared.get('trend_settings'): (
                                 f"HL→PB timeout: {getattr(ts, 'breakout_to_pullback_bars_3m', 25)} bars | PB→BOS timeout: {getattr(ts, 'pullback_to_bos_bars_3m', 25)} bars"
                               ) if ts else "HL→PB timeout: 25 bars | PB→BOS timeout: 25 bars")(),
                             f"SL Mode: {sl_mode} | SL Buffer ATR: {sl_buf:.2f}",
                             f"HTF min trend strength: {min_ts:.1f}",
                             f"Recovery: BE reconcile: {'On' if ((cfg.get('trend',{}) or {}).get('exec',{}) or {}).get('recovery_reconcile_be', False) else 'Off'}",
                             "",
                             "🧱 *Exec S/R Gate*",
                             f"Enabled: {(((cfg.get('trend',{}) or {}).get('exec',{}) or {}).get('sr',{}) or {}).get('enabled', True)} | Min strength: {(((cfg.get('trend',{}) or {}).get('exec',{}) or {}).get('sr',{}) or {}).get('min_strength', 2.8)}",
                             f"Confluence tol: {(((cfg.get('trend',{}) or {}).get('exec',{}) or {}).get('sr',{}) or {}).get('confluence_tolerance_pct', 0.0025)} | Clearance ATR: {(((cfg.get('trend',{}) or {}).get('exec',{}) or {}).get('sr',{}) or {}).get('min_break_clear_atr', 0.10)}",
                             "",
                             f"Phantoms: {'On' if ph.get('enabled', True) else 'Off'} | Weight: {ph.get('weight', 0.8)}",
                             "",
                             "📈 *Symbol HTF Gate*",
                             f"Enabled: {bool(htf_gate.get('enabled', True))} | Mode: {htf_gate.get('mode','gated')}",
                             f"TS1H min: {float(htf_gate.get('min_trend_strength_1h',60.0)):.1f} | TS4H min: {float(htf_gate.get('min_trend_strength_4h',55.0)):.1f}",
                             f"EMA align: {'On' if bool(htf_gate.get('ema_alignment', True)) else 'Off'} | ADX1H min: {float(htf_gate.get('adx_min_1h',0.0)):.1f}",
                             f"Structure: {'On' if bool(htf_gate.get('structure_confluence', False)) else 'Off'} | Soft Δ: {float(htf_gate.get('soft_delta',5)):.0f}",
                             "",
                             "📐 *Divergence (3m)*",
                             f"Mode: {div.get('mode','off')} | Require: {div.get('require','any')} | Osc: {', '.join(div.get('oscillators', ['rsi','tsi']))}",
                             f"RSI len: {div.get('rsi_len',14)} | TSI: {div.get('tsi_long',25)}/{div.get('tsi_short',13)} | Window: {div.get('confirm_window_bars_3m',6)} bars",
                             f"Min strength — RSI: {((div.get('min_strength',{}) or {}).get('rsi',2.0))} | TSI: {((div.get('min_strength',{}) or {}).get('tsi',0.3))}",
                             "",
                             "Use the buttons below to toggle or set values."]
                    kb = InlineKeyboardMarkup([
                        # Strategy selector for Q thresholds submenus
                        [InlineKeyboardButton("Trend Q", callback_data="ui:settings:q:trend"), InlineKeyboardButton("Range Q", callback_data="ui:settings:q:range"), InlineKeyboardButton("Scalp Q", callback_data="ui:settings:q:scalp")],
                        [InlineKeyboardButton("Stream Entry", callback_data="ui:settings:toggle:stream")],
                        [InlineKeyboardButton("Scale‑out", callback_data="ui:settings:toggle:scaleout")],
                        [InlineKeyboardButton("BE Move", callback_data="ui:settings:toggle:be")],
                        [InlineKeyboardButton("SL Mode", callback_data="ui:settings:toggle:sl_mode"), InlineKeyboardButton("Set SL Buffer", callback_data="ui:settings:set:sl_buffer")],
                        [InlineKeyboardButton("Set R:R", callback_data="ui:settings:set:rr"), InlineKeyboardButton("Set TP1 R", callback_data="ui:settings:set:tp1_r")],
                        [InlineKeyboardButton("Set TP2 R", callback_data="ui:settings:set:tp2_r"), InlineKeyboardButton("Set Fraction", callback_data="ui:settings:set:fraction")],
                        [InlineKeyboardButton("Set Confirm Bars", callback_data="ui:settings:set:confirm_bars"), InlineKeyboardButton("Set Phantom Hours", callback_data="ui:settings:set:phantom_hours")],
                        [InlineKeyboardButton("Set HL→PB Bars", callback_data="ui:settings:set:timeout_hl"), InlineKeyboardButton("Set PB→BOS Bars", callback_data="ui:settings:set:timeout_bos")],
                        [InlineKeyboardButton("Set HTF Min TS", callback_data="ui:settings:set:htf_min_ts")],
                        [InlineKeyboardButton("HTF Gate", callback_data="ui:settings:toggle:htf_gate"), InlineKeyboardButton("Mode", callback_data="ui:settings:toggle:htf_mode")],
                        [InlineKeyboardButton("Set TS1H", callback_data="ui:settings:set:htf_ts1h"), InlineKeyboardButton("Set TS4H", callback_data="ui:settings:set:htf_ts4h")],
                        [InlineKeyboardButton("EMA Align", callback_data="ui:settings:toggle:htf_ema"), InlineKeyboardButton("Set ADX1H", callback_data="ui:settings:set:htf_adx1h")],
                        [InlineKeyboardButton("Structure", callback_data="ui:settings:toggle:htf_struct"), InlineKeyboardButton("Soft Δ", callback_data="ui:settings:set:htf_soft_delta")],
                        [InlineKeyboardButton("Recovery BE", callback_data="ui:settings:toggle:reconcile_be")],
                        [InlineKeyboardButton("SR Gate", callback_data="ui:settings:toggle:sr_gate"), InlineKeyboardButton("Set SR Strength", callback_data="ui:settings:set:sr_strength")],
                        [InlineKeyboardButton("Set SR Confluence", callback_data="ui:settings:set:sr_confluence"), InlineKeyboardButton("Set SR Clear ATR", callback_data="ui:settings:set:sr_clear")],
                        
                        [InlineKeyboardButton("Phantoms On/Off", callback_data="ui:settings:toggle:phantom"), InlineKeyboardButton("Set Phantom Weight", callback_data="ui:settings:set:phantom_weight")],
                        [InlineKeyboardButton("Div Mode", callback_data="ui:settings:toggle:div_mode"), InlineKeyboardButton("Div Require", callback_data="ui:settings:toggle:div_require")],
                        [InlineKeyboardButton("Osc RSI", callback_data="ui:settings:toggle:div_osc_rsi"), InlineKeyboardButton("Osc TSI", callback_data="ui:settings:toggle:div_osc_tsi")],
                        [InlineKeyboardButton("Set RSI Len", callback_data="ui:settings:set:div_rsi_len"), InlineKeyboardButton("Set TSI Params", callback_data="ui:settings:set:div_tsi_params")],
                        [InlineKeyboardButton("Set Div Window", callback_data="ui:settings:set:div_window"), InlineKeyboardButton("Set Min RSI", callback_data="ui:settings:set:div_min_rsi")],
                        [InlineKeyboardButton("Set Min TSI", callback_data="ui:settings:set:div_min_tsi")],
                        [InlineKeyboardButton("🔙 Back", callback_data="ui:dash:refresh")]
                    ])
                    await query.edit_message_text("\n".join(lines), reply_markup=kb, parse_mode='Markdown')
                    return
                if data.startswith("ui:settings:q:"):
                    await query.answer()
                    strat = data.split(':')[-1]  # trend|range|scalp
                    cfg = self.shared.get('config', {}) or {}
                    rm = ((cfg.get(strat, {}) or {}).get('rule_mode', {}) or {})
                    exec_q = float(rm.get('execute_q_min', 78 if strat != 'scalp' else 40))
                    ph_q = float(rm.get('phantom_q_min', 65 if strat != 'scalp' else 80))
                    header = f"⚙️ *{strat.title()} Thresholds*\n\nExec≥{exec_q:.0f} | Phantom≥{ph_q:.0f}"
                    # Map to prompts keys
                    key_exec = 'exec_q' if strat == 'trend' else ('exec_q_range' if strat == 'range' else 'exec_q_scalp')
                    key_ph = 'phantom_q' if strat == 'trend' else ('phantom_q_range' if strat == 'range' else 'phantom_q_scalp')
                    kb = InlineKeyboardMarkup([
                        [InlineKeyboardButton("Set Exec Q", callback_data=f"ui:settings:set:{key_exec}"), InlineKeyboardButton("Set Phantom Q", callback_data=f"ui:settings:set:{key_ph}")],
                        [InlineKeyboardButton("🔙 Back", callback_data="ui:settings")]
                    ])
                    await query.edit_message_text(header, reply_markup=kb, parse_mode='Markdown')
                    return
                if data.startswith("ui:settings:toggle:"):
                    await query.answer()
                    key = data.rsplit(':',1)[-1]
                    bot = self.shared.get('bot_instance')
                    cfg = self.shared.get('config', {}) or {}
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    sc = (tr_exec.get('scaleout',{}) or {})
                    div = (tr_exec.get('divergence',{}) or {})
                    ph = (cfg.get('phantom', {}) or {})
                    ts = self.shared.get('trend_settings')
                    if key == 'stream':
                        tr_exec['allow_stream_entry'] = not bool(tr_exec.get('allow_stream_entry', True))
                    elif key == 'scaleout':
                        sc['enabled'] = not bool(sc.get('enabled', False))
                        tr_exec['scaleout'] = sc
                    elif key == 'be':
                        sc['move_sl_to_be'] = not bool(sc.get('move_sl_to_be', True))
                        tr_exec['scaleout'] = sc
                    elif key == 'sl_mode':
                        cur = str(tr_exec.get('sl_mode', 'breakout'))
                        nxt = 'hybrid' if cur == 'breakout' else 'breakout'
                        tr_exec['sl_mode'] = nxt
                        ts = self.shared.get('trend_settings')
                        if ts:
                            ts.sl_mode = nxt
                    elif key == 'reconcile_be':
                        try:
                            cur = bool(tr_exec.get('recovery_reconcile_be', False))
                        except Exception:
                            cur = False
                        tr_exec['recovery_reconcile_be'] = (not cur)
                    elif key == 'sr_gate':
                        sr = (tr_exec.get('sr', {}) or {})
                        sr['enabled'] = not bool(sr.get('enabled', True))
                        tr_exec['sr'] = sr
                        ts = self.shared.get('trend_settings')
                        if ts:
                            ts.sr_exec_enabled = bool(sr['enabled'])
                    
                    elif key == 'htf_gate':
                        hg = (tr_exec.get('htf_gate', {}) or {})
                        hg['enabled'] = not bool(hg.get('enabled', True))
                        tr_exec['htf_gate'] = hg
                    elif key == 'htf_mode':
                        hg = (tr_exec.get('htf_gate', {}) or {})
                        cur = str(hg.get('mode', 'gated')).lower()
                        hg['mode'] = 'soft' if cur == 'gated' else 'gated'
                        tr_exec['htf_gate'] = hg
                    elif key == 'htf_ema':
                        hg = (tr_exec.get('htf_gate', {}) or {})
                        hg['ema_alignment'] = not bool(hg.get('ema_alignment', True))
                        tr_exec['htf_gate'] = hg
                    elif key == 'htf_struct':
                        hg = (tr_exec.get('htf_gate', {}) or {})
                        hg['structure_confluence'] = not bool(hg.get('structure_confluence', False))
                        tr_exec['htf_gate'] = hg
                    elif key == 'div_mode':
                        # Cycle: off -> optional -> strict -> off
                        mode = str((div.get('mode') or 'off')).lower()
                        nxt = 'optional' if mode == 'off' else 'strict' if mode == 'optional' else 'off'
                        div['mode'] = nxt; div['enabled'] = (nxt != 'off')
                        tr_exec['divergence'] = div
                        if ts:
                            ts.div_mode = nxt; ts.div_enabled = (nxt != 'off')
                    elif key == 'div_require':
                        req = str(div.get('require','any')).lower()
                        div['require'] = 'all' if req == 'any' else 'any'
                        tr_exec['divergence'] = div
                        if ts:
                            ts.div_require = div['require']
                    elif key == 'div_osc_rsi':
                        osc = list(div.get('oscillators', ['rsi','tsi']))
                        if 'rsi' in osc:
                            osc = [o for o in osc if o != 'rsi']
                        else:
                            osc.append('rsi')
                        div['oscillators'] = osc
                        tr_exec['divergence'] = div
                        if ts:
                            ts.div_use_rsi = bool('rsi' in osc)
                    elif key == 'div_osc_tsi':
                        osc = list(div.get('oscillators', ['rsi','tsi']))
                        if 'tsi' in osc:
                            osc = [o for o in osc if o != 'tsi']
                        else:
                            osc.append('tsi')
                        div['oscillators'] = osc
                        tr_exec['divergence'] = div
                        if ts:
                            ts.div_use_tsi = bool('tsi' in osc)
                    elif key == 'phantom':
                        ph['enabled'] = not bool(ph.get('enabled', True))
                        cfg['phantom'] = ph
                    try:
                        cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                        if bot is not None:
                            bot.config = cfg
                        self.shared['config'] = cfg
                    except Exception:
                        pass
                    # Back to settings snapshot
                    await self.ui_callback(update, ctx)
                    return
                if data.startswith("ui:settings:set:"):
                    await query.answer()
                    key = data.rsplit(':',1)[-1]
                    # Prompt for value and store awaiting state
                    chat_id = query.message.chat_id
                    self._ui_state[chat_id] = {'await': key}
                    prompts = {
                        'rr': "Send a number for base R:R (e.g., 2.5)",
                        'tp1_r': "Send TP1 R (e.g., 1.6)",
                        'tp2_r': "Send TP2 R (e.g., 3.0)",
                        'fraction': "Send scale‑out fraction between 0.1 and 0.9 (e.g., 0.5)",
                        'confirm_bars': "Send confirmation timeout bars (integer, e.g., 6)",
                        'phantom_hours': "Send phantom timeout hours (integer, e.g., 100)",
                        'phantom_weight': "Send phantom training weight between 0.3 and 1.0 (e.g., 0.8)",
                        'sl_buffer': "Send breakout SL buffer ATR (e.g., 0.40)",
                        'exec_q': "Send Exec Q threshold for Trend (e.g., 78)",
                        'phantom_q': "Send Phantom Q threshold for Trend (e.g., 65)",
                        'exec_q_range': "Send Exec Q threshold for Range (e.g., 78)",
                        'phantom_q_range': "Send Phantom Q threshold for Range (e.g., 65)",
                        'exec_q_scalp': "Send Exec Q threshold for Scalp (e.g., 75)",
                        'phantom_q_scalp': "Send Phantom Q threshold for Scalp (e.g., 65-80)",
                        'timeout_hl': "Send HL→PB timeout bars (integer, e.g., 25)",
                        'timeout_bos': "Send PB→BOS timeout bars (integer, e.g., 25)",
                        'htf_min_ts': "Send HTF min trend strength (e.g., 60)",
                        'sr_strength': "Send SR min strength (e.g., 2.8)",
                        'sr_confluence': "Send SR confluence tolerance pct (e.g., 0.0025)",
                        'sr_clear': "Send SR min break clearance in ATR (e.g., 0.10)",
                        
                        'htf_ts1h': "Send min trend strength 1H (e.g., 60)",
                        'htf_ts4h': "Send min trend strength 4H (e.g., 55)",
                        'htf_adx1h': "Send ADX(14) minimum on 1H (e.g., 20; 0 to disable)",
                        'htf_soft_delta': "Send soft gate ML threshold delta (e.g., 5)",
                        'div_rsi_len': "Send RSI length (e.g., 14)",
                        'div_tsi_params': "Send TSI params as long,short (e.g., 25,13)",
                        'div_window': "Send divergence confirm window in 3m bars (e.g., 6)",
                        'div_min_rsi': "Send minimum RSI delta (e.g., 2.0)",
                        'div_min_tsi': "Send minimum TSI delta (e.g., 0.3)"
                    }
                    await query.edit_message_text(f"✍️ {prompts.get(key,'Send a value')}")
                    return
            if data.startswith("ui:dash:refresh"):
                # Build fresh dashboard and edit in place (no cooldown)
                await query.answer("Refreshing…")
                text, kb = self._build_trend_dashboard()
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception:
                    # Fallback to sending a new message if edit fails
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
            elif data.startswith("ui:symbols:"):
                idx = int(data.split(":")[-1])
                frames = self.shared.get("frames", {})
                symbols = list(frames.keys())
                if not symbols:
                    await query.answer("No symbols")
                    return
                idx = max(0, min(len(symbols)-1, idx))
                sym = symbols[idx]
                df = frames.get(sym)
                msg = f"📈 *{sym}*\n"
                if df is not None and not df.empty:
                    last = df['close'].iloc[-1]
                    msg += f"• Last: {last:.4f}\n• Candles: {len(df)}\n"
                kb = [[
                    InlineKeyboardButton("◀️ Prev", callback_data=f"ui:symbols:{idx-1}"),
                    InlineKeyboardButton("Next ▶️", callback_data=f"ui:symbols:{idx+1}")
                ],[
                    InlineKeyboardButton("🔙 Back", callback_data="ui:dash:refresh")
                ]]
                await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
            elif data.startswith("ui:phantom:main"):
                # Quick phantom QA reuse
                await query.answer()
                fake_update = type('obj', (object,), {'message': query.message})
                await self.phantom_qa(fake_update, ctx)
            elif data.startswith("ui:scalp:qa"):
                await query.answer()
                fake_update = type('obj', (object,), {'message': query.message})
                await self.scalp_qa(fake_update, ctx)
            elif data.startswith("ui:scalp:qscore"):
                await query.answer()
                fake_update = type('obj', (object,), {'message': query.message})
                await self.scalp_qscore_report(fake_update, ctx)
            elif data.startswith("ui:qscore:all"):
                await query.answer()
                fake_update = type('obj', (object,), {'message': query.message})
                await self.qscore_all_report(fake_update, ctx)
            elif data.startswith("ui:ml:stats"):
                await query.answer()
                fake_update = type('obj', (object,), {'message': query.message})
                await self.ml_all_report(fake_update, ctx)
            # Scalp promotion UI removed (feature disabled)
            elif data.startswith("ui:ml:main"):
                await query.answer()
                ml = self.shared.get("ml_scorer")
                msg = "🤖 *ML Overview*\n"
                if ml:
                    st = ml.get_stats()
                    msg += f"• Trend: {st.get('status')} | Thresh: {st.get('current_threshold')}\n"
                    calib = None
                    try:
                        import os, redis, json
                        r = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
                        calib = r.get('iml:calibration')
                    except Exception:
                        pass
                    if calib:
                        msg += "• Calibration: bins saved\n"
                emr = self.shared.get("enhanced_mr_scorer")
                if emr:
                    try:
                        info = emr.get_retrain_info()
                        status = "Ready" if info.get('is_ml_ready') else 'Training'
                        msg += f"• Enhanced MR: {status} | Thresh: {getattr(emr, 'min_score', 'N/A')}\n"
                        msg += f"  Trades: {info.get('total_combined', 0)} | Next retrain in: {info.get('trades_until_next_retrain', 0)}\n"
                        msg += f"  Executed: {info.get('completed_trades', 0)} | Phantom: {info.get('phantom_count', 0)}\n"
                    except Exception:
                        pass
                await query.edit_message_text(msg, parse_mode='Markdown')
            elif data.startswith("ui:shadow:stats"):
                await query.answer()
                try:
                    from shadow_trade_simulator import get_shadow_tracker
                    st = get_shadow_tracker()
                    s_stats = st.get_stats()
                except Exception:
                    s_stats = {}
                # Baseline from executed trades
                tt = self.shared.get('trade_tracker')
                trades = getattr(tt, 'trades', []) or []
                def _baseline_for(key: str):
                    wins = losses = 0
                    for t in trades:
                        strat = (getattr(t, 'strategy_name', '') or '').lower()
                        grp = 'trend' if ('trend' in strat or 'pullback' in strat) else 'enhanced_mr' if ('mr' in strat or 'reversion' in strat) else None
                        if grp == key:
                            pnl = float(getattr(t, 'pnl_usd', 0))
                            if pnl > 0:
                                wins += 1
                            else:
                                losses += 1
                    total = wins + losses
                    wr = (wins/total*100.0) if total else 0.0
                    return wins, losses, total, wr
                pbw, pbl, pbt, pbwr = _baseline_for('trend')
                mrw, mrl, mrt, mrwr = _baseline_for('enhanced_mr')
                msg = [
                    "🧪 *Shadow vs Baseline*",
                    "",
                    "🔵 Trend",
                    f"• Baseline: W {pbw} / L {pbl} (WR {pbwr:.1f}%)",
                    f"• Shadow:   W {s_stats.get('trend',{}).get('wins',0)} / L {s_stats.get('trend',{}).get('losses',0)} (WR {s_stats.get('trend',{}).get('wr',0.0):.1f}%)",
                    "",
                    "🌀 Mean Reversion",
                    f"• Baseline: W {mrw} / L {mrl} (WR {mrwr:.1f}%)",
                    f"• Shadow:   W {s_stats.get('enhanced_mr',{}).get('wins',0)} / L {s_stats.get('enhanced_mr',{}).get('losses',0)} (WR {s_stats.get('enhanced_mr',{}).get('wr',0.0):.1f}%)",
                ]
                await query.edit_message_text("\n".join(msg), parse_mode='Markdown')
            elif data.startswith("ui:risk:main"):
                await query.answer()
                risk = self.shared.get('risk')
                msg = "⚙️ *Risk Settings*\n"
                if risk:
                    msg += f"• Risk: {risk.risk_percent:.2f}% | Leverage: {risk.max_leverage}x\n"
                await query.edit_message_text(msg, parse_mode='Markdown')
            elif data.startswith("ui:regime:main"):
                await query.answer()
                frames = self.shared.get('frames', {})
                lines = ["🧭 *Market Regime*", ""]
                if not frames:
                    lines.append("No market data available yet")
                else:
                    # Show summaries for up to 5 most recently analysed symbols
                    last_analysis = self.shared.get("last_analysis", {})
                    if last_analysis:
                        symbols = [s for s, _ in sorted(last_analysis.items(), key=lambda kv: kv[1], reverse=True)[:5]]
                    else:
                        symbols = list(frames.keys())[:5]
                    for sym in symbols:
                        df = frames.get(sym)
                        if df is None or df.empty:
                            continue
                        try:
                            summary = get_regime_summary(df.tail(200), sym)
                            lines.append(f"• {sym}: {summary}")
                        except Exception:
                            lines.append(f"• {sym}: unable to analyse")
                await query.edit_message_text("\n".join(lines), parse_mode='Markdown')
            elif data.startswith("ui:htf:status"):
                await query.answer()
                await query.edit_message_text("HTF S/R module disabled.")
            elif data.startswith("ui:htf:update"):
                await query.answer("Disabled")
            else:
                await query.answer("Unknown action")
        except Exception as e:
            logger.error(f"Error in ui_callback: {e}")
            try:
                await update.callback_query.answer("Error")
            except Exception:
                pass
    


    async def regime_single(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show enhanced regime snapshot for one or more symbols."""
        try:
            frames = self.shared.get("frames", {})
            if not frames:
                await update.message.reply_text("No market data available yet")
                return

            if ctx.args:
                symbols = [ctx.args[0].upper()]
            else:
                last_analysis = self.shared.get("last_analysis", {})
                if last_analysis:
                    symbols = [sym for sym, _ in sorted(last_analysis.items(), key=lambda kv: kv[1], reverse=True)[:5]]
                else:
                    symbols = list(frames.keys())[:5]

            if not symbols:
                await update.message.reply_text("No symbols to analyse")
                return

            lines = ["🔍 *Market Regime Snapshot*", ""]

            for sym in symbols:
                df = frames.get(sym)
                if df is None or len(df) < 50:
                    lines.append(f"• {sym}: insufficient data")
                    continue

                try:
                    analysis = get_enhanced_market_regime(df.tail(200), sym)
                    confidence = analysis.regime_confidence * 100
                    lines.append(f"• *{sym}*: {analysis.primary_regime.title()} ({confidence:.0f}% conf)")

                    detail = [f"Vol: {analysis.volatility_level}"]
                    if analysis.primary_regime == "ranging":
                        detail.append(f"Range: {analysis.range_quality}")
                    if analysis.recommended_strategy and analysis.recommended_strategy != "none":
                        detail.append(f"Strat: {analysis.recommended_strategy.replace('_', ' ').title()}")
                    lines.append("  " + " | ".join(detail))
                except Exception as exc:
                    logger.debug(f"Regime analysis error for {sym}: {exc}")
                    lines.append(f"• {sym}: unable to analyse")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.exception("Error in regime command: %s", e)
            await update.message.reply_text("Error retrieving regime information")


    async def analysis(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show recent analysis details for symbols."""
        try:
            frames = self.shared.get("frames", {})
            last_analysis = self.shared.get("last_analysis", {})

            if not frames:
                await update.message.reply_text("No market data available yet")
                return

            if ctx.args:
                symbols = [ctx.args[0].upper()]
            elif last_analysis:
                symbols = [sym for sym, _ in sorted(last_analysis.items(), key=lambda kv: kv[1], reverse=True)[:5]]
            else:
                symbols = list(frames.keys())[:5]

            msg = "🔍 *Recent Analysis*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

            for symbol in symbols:
                df = frames.get(symbol)
                if df is None or len(df) < 20:
                    msg += f"*{symbol}*: insufficient data\n\n"
                    continue

                msg += f"*{symbol}*\n"

                if symbol in last_analysis:
                    analysed_at = last_analysis[symbol]
                    if analysed_at.tzinfo is None:
                        analysed_at = analysed_at.replace(tzinfo=timezone.utc)
                    age_minutes = max(0, int((datetime.now(timezone.utc) - analysed_at).total_seconds() // 60))
                    msg += f"• Last analysed: {age_minutes}m ago\n"
                else:
                    msg += "• Last analysed: n/a\n"

                last_price = df['close'].iloc[-1]
                msg += f"• Last price: {last_price:.4f}\n"
                msg += f"• Candles loaded: {len(df)}\n\n"

            msg += "_Use /analysis SYMBOL for a specific market_"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.exception("Error in analysis: %s", e)
            await update.message.reply_text("Error getting analysis details")
    
    async def stats(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show trading statistics"""
        try:
            # Get trade tracker from shared
            tracker = self.shared.get("trade_tracker")
            if not tracker:
                await update.message.reply_text("Statistics tracking not initialized yet")
                return
            
            # Parse arguments for time period
            days = None
            if ctx.args:
                try:
                    days = int(ctx.args[0])
                except ValueError:
                    pass
            
            # Get formatted statistics
            msg = tracker.format_stats_message(days)
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in stats: {e}")
            await update.message.reply_text("Error getting statistics")
    
    async def recent_trades(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show recent trades"""
        try:
            # Get trade tracker from shared
            tracker = self.shared.get("trade_tracker")
            if not tracker:
                await update.message.reply_text("Trade tracking not initialized yet")
                return
            
            # Parse arguments for limit
            limit = 5
            if ctx.args:
                try:
                    limit = min(20, int(ctx.args[0]))  # Max 20 recent trades
                except ValueError:
                    pass
            
            # Get formatted recent trades
            if hasattr(tracker, 'format_recent_trades'):
                msg = tracker.format_recent_trades(limit)
            else:
                # Fallback for TradeTrackerPostgres without format_recent_trades
                msg = "📜 *Recent Trades*\n"
                msg += "━" * 20 + "\n\n"
                
                # Get trades
                trades = []
                if hasattr(tracker, 'trades'):
                    trades = tracker.trades[-limit:] if tracker.trades else []
                elif hasattr(tracker, 'get_recent_trades'):
                    trades = tracker.get_recent_trades(limit)
                
                if not trades:
                    msg += "_No trades recorded yet_"
                else:
                    for i, trade in enumerate(reversed(trades[-limit:]), 1):
                        # Format each trade
                        symbol = getattr(trade, 'symbol', 'N/A')
                        side = getattr(trade, 'side', 'N/A').upper()
                        pnl = float(getattr(trade, 'pnl_usd', 0))
                        pnl_pct = float(getattr(trade, 'pnl_percent', 0))
                        exit_time = getattr(trade, 'exit_time', None)
                        
                        # Format time
                        time_str = ""
                        if exit_time:
                            if isinstance(exit_time, str):
                                time_str = exit_time[:16]  # Keep YYYY-MM-DD HH:MM
                            else:
                                time_str = exit_time.strftime("%Y-%m-%d %H:%M")
                        
                        # Build trade line
                        result_emoji = "✅" if pnl > 0 else "❌"
                        msg += f"{i}. {result_emoji} *{symbol}* {side}\n"
                        msg += f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                        if time_str:
                            msg += f"   Time: {time_str}\n"
                        msg += "\n"
                
                msg += f"\n_Showing last {min(limit, len(trades))} trades_"
                
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in recent_trades: {e}")
            await update.message.reply_text("Error getting recent trades")

    # Removed: trend_high_ml method (Trend strategy disabled)


    # Removed: trend_states method (Trend strategy disabled)
        """Show current Trend Pullback state per symbol"""
        try:
            from strategy_pullback import get_trend_states_snapshot
            snap = get_trend_states_snapshot()
            if not snap:
                await self.safe_reply(update, "🔎 Trend Pullback States\n• No states tracked yet")
                return
            lines = ["🔎 *Trend Pullback States*", "━━━━━━━━━━━━━━━━━━━━━", ""]
            for sym in sorted(snap.keys()):
                st = snap[sym]
                state = st.get('state','?')
                brk = st.get('breakout_level')
                ext = st.get('pullback_extreme')
                conf = st.get('confirm_progress', 0)
                age = st.get('pullback_age_bars')
                micro = st.get('micro_state') or ""
                pivot = float(st.get('last_counter_pivot') or 0.0)
                # Decide pivot label by macro path: long path expects LH, short path expects HL
                if state in ("RESISTANCE_BROKEN", "HL_FORMED"):
                    pivot_label = "LH"
                elif state in ("SUPPORT_BROKEN", "LH_FORMED"):
                    pivot_label = "HL"
                else:
                    pivot_label = "PV"
                detail = []
                if brk:
                    detail.append(f"break {brk:.4f}")
                if ext:
                    detail.append(f"pb {ext:.4f}")
                detail.append(f"conf {conf}")
                if age is not None:
                    detail.append(f"age {age}b")
                suffix = f" | {micro}" if micro else ""
                if pivot > 0:
                    suffix += f" | {pivot_label}={pivot:.4f}"
                try:
                    div_ok = st.get('divergence_ok', False)
                    div_type = st.get('divergence_type','NONE')
                    div_score = float(st.get('divergence_score',0.0) or 0.0)
                    suffix += f" | Div={'✅' if div_ok else '—'} {div_type}{(' '+str(round(div_score,2))) if div_score else ''}"
                except Exception:
                    pass
                try:
                    if st.get('bos_crossed'):
                        wr = st.get('waiting_reason') or 'WAIT_PIVOT'
                        suffix += f" | BOS=armed ({wr})"
                except Exception:
                    pass
                lines.append(f"• {sym}: {state}{suffix} ({', '.join(detail)})")
            await self.safe_reply(update, "\n".join(lines))
        except Exception as exc:
            logger.warning(f"trend_states failed: {exc}")
            await self.safe_reply(update, "⚠️ Failed to fetch trend states")
    
    async def ml_stats(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show ML system statistics and status for a specific strategy."""
        try:
            strategy_arg = 'trend' # Default strategy
            if ctx.args:
                strategy_arg = ctx.args[0].lower()
                if strategy_arg not in ['trend', 'reversion']:
                    await self.safe_reply(update, "Invalid strategy. Choose `trend` or `reversion`.")
                    return

            msg = f"🤖 *ML Status: {strategy_arg.title()} Strategy*\n"
            msg += "━" * 25 + "\n\n"

            if strategy_arg == 'trend':
                ml_scorer = self.shared.get("ml_scorer")
            else:
                # Placeholder for the future mean reversion scorer
                ml_scorer = self.shared.get("ml_scorer_reversion") 

            if not ml_scorer:
                msg += f"❌ *ML System Not Available for {strategy_arg.title()} Strategy*\n"
                if strategy_arg == 'reversion':
                    msg += "This model will be trained after enough data is collected from the rule-based strategy."
                await self.safe_reply(update, msg)
                return

            # Get and display stats from the selected scorer
            stats = ml_scorer.get_stats()
            if stats.get('is_ml_ready'):
                msg += "✅ *Status: Active & Learning*\n"
                msg += f"• Model trained on: {stats.get('last_train_count', stats.get('completed_trades', 0))} trades\n"
            else:
                msg += "📊 *Status: Collecting Data*\n"
                trades_needed = stats.get('trades_needed', 200)
                msg += f"• Trades needed for training: {trades_needed}\n"

            msg += f"• Completed trades (live): {stats.get('completed_trades', 0)}\n"
            msg += f"• Current threshold: {stats.get('current_threshold', 70):.0f}\n"
            msg += f"• Active models: {len(stats.get('models_active', []))}\n"

            # Show retrain info for mean reversion
            if strategy_arg == 'reversion' and 'next_retrain_in' in stats:
                msg += f"• Next retrain in: {stats['next_retrain_in']} trades\n"

            # Add strategy-specific notes
            if strategy_arg == 'reversion':
                msg += "\n📝 *Mean Reversion Features:*\n"
                msg += "• Range characteristics\n"
                msg += "• Oscillator extremes\n"
                msg += "• Volume confirmation\n"
                msg += "• Reversal strength\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in ml_stats: {e}")
            await update.message.reply_text("Error getting ML statistics")
    
    async def reset_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Reset trade statistics"""
        try:
            import os
            import json
            from datetime import datetime
            
            # Check if user is authorized
            if update.effective_user.id != self.chat_id:
                await update.message.reply_text("❌ Unauthorized")
                return
            
            reset_count = 0
            backups = []
            
            # 1. Reset TradeTracker history
            if os.path.exists("trade_history.json"):
                try:
                    # Read existing data
                    with open("trade_history.json", 'r') as f:
                        data = json.load(f)
                        trade_count = len(data)
                    
                    if trade_count > 0:
                        # Create backup
                        backup_name = f"trade_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(backup_name, 'w') as f:
                            json.dump(data, f, indent=2)
                        backups.append(f"• {trade_count} trades → {backup_name}")
                        reset_count += trade_count
                        
                        # Create empty file
                        with open("trade_history.json", 'w') as f:
                            json.dump([], f)
                except Exception as e:
                    logger.error(f"Error backing up trade history: {e}")
            
            # 2. Reset any cached stats in shared tracker
            if "tracker" in self.shared and self.shared["tracker"]:
                try:
                    self.shared["tracker"].trades = []
                    self.shared["tracker"].save_trades()
                    backups.append("• Tracker cache cleared")
                except:
                    pass
            
            # 3. Reset ML trade count (keep model but reset counter)
            ml_reset_info = ""
            if "ml_scorer" in self.shared and self.shared["ml_scorer"]:
                try:
                    ml_scorer = self.shared["ml_scorer"]
                    old_count = ml_scorer.completed_trades_count if hasattr(ml_scorer, 'completed_trades_count') else 0
                    
                    # Reset counters
                    if hasattr(ml_scorer, 'completed_trades_count'):
                        ml_scorer.completed_trades_count = 0
                    if hasattr(ml_scorer, 'last_train_count'):
                        ml_scorer.last_train_count = 0
                    
                    # Clear Redis data if available
                    if hasattr(ml_scorer, 'redis_client') and ml_scorer.redis_client:
                        try:
                            ml_scorer.redis_client.delete('ml_completed_trades')
                            ml_scorer.redis_client.delete('ml_enhanced_completed_trades')
                            ml_scorer.redis_client.delete('ml_v2_completed_trades')
                            ml_scorer.redis_client.set('ml_trades_count', 0)
                            ml_scorer.redis_client.set('ml_enhanced_trades_count', 0)
                            ml_scorer.redis_client.set('ml_v2_trades_count', 0)
                        except:
                            pass
                    
                    ml_reset_info = f"\n🤖 **ML Status:**\n"
                    ml_reset_info += f"• Reset {old_count} trade counter\n"
                    ml_reset_info += f"• Model kept (if trained)\n"
                    ml_reset_info += f"• Will retrain after 200 new trades"
                except Exception as e:
                    logger.error(f"Error resetting ML stats: {e}")
            
            # Build response
            if reset_count > 0 or backups:
                response = "✅ **Statistics Reset Complete!**\n\n"
                
                if backups:
                    response += "**Backed up:**\n"
                    response += "\n".join(backups) + "\n"
                
                response += ml_reset_info + "\n"
                
                response += "\n**What happens now:**\n"
                response += "• Trade history: Starting fresh at 0\n"
                response += "• Win rate: Will recalculate from new trades\n"
                response += "• P&L: Reset to $0.00\n"
                response += "• New trades will build fresh statistics\n\n"
                response += "📊 Use /stats to see fresh statistics\n"
                response += "🤖 Use /ml to check ML status"
            else:
                response = "ℹ️ No statistics to reset - already clean\n\n"
                response += "📊 /stats - View statistics\n"
                response += "🤖 /ml - Check ML status"
            
            await self.safe_reply(update, response)
            
        except Exception as e:
            logger.error(f"Error resetting stats: {e}")
            await update.message.reply_text(f"❌ Error resetting stats: {str(e)[:200]}")
    
    async def ml_rankings(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML rankings for all symbols"""
        try:
            # Get trade tracker for historical data
            trade_tracker = self.shared.get("trade_tracker")
            if not trade_tracker:
                await update.message.reply_text("Trade tracker not available")
                return
            
            # Get all trades
            all_trades = trade_tracker.trades if hasattr(trade_tracker, 'trades') else []
            
            if not all_trades:
                await update.message.reply_text(
                    "📊 *ML Symbol Rankings*\n\n"
                    "No completed trades yet.\n"
                    "Rankings will appear after trades complete.",
                    parse_mode='Markdown'
                )
                return
            
            # Calculate stats per symbol
            symbol_stats = {}
            
            for trade in all_trades:
                symbol = trade.symbol
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0,
                        'best_trade': 0,
                        'worst_trade': 0,
                        'last_5_trades': []
                    }
                
                stats = symbol_stats[symbol]
                stats['trades'] += 1
                
                # Convert Decimal to float for calculations
                pnl = float(trade.pnl_usd)
                
                if pnl > 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
                
                stats['total_pnl'] += pnl
                stats['best_trade'] = max(stats['best_trade'], pnl)
                stats['worst_trade'] = min(stats['worst_trade'], pnl)
                
                # Track last 5 trades for trend
                stats['last_5_trades'].append(1 if pnl > 0 else 0)
                if len(stats['last_5_trades']) > 5:
                    stats['last_5_trades'].pop(0)
            
            # Calculate rankings
            rankings = []
            for symbol, stats in symbol_stats.items():
                win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
                
                # Recent performance (last 5 trades)
                if len(stats['last_5_trades']) >= 3:
                    recent_wr = sum(stats['last_5_trades']) / len(stats['last_5_trades']) * 100
                else:
                    recent_wr = win_rate
                
                # Score combines win rate, profitability, and recent performance
                # 50% win rate, 30% profitability, 20% recent performance
                normalized_pnl = max(-100, min(100, avg_pnl * 10))  # Normalize P&L
                score = (win_rate * 0.5) + (normalized_pnl * 0.3) + (recent_wr * 0.2)
                
                # Training data indicator
                data_quality = "🟢" if stats['trades'] >= 10 else "🟡" if stats['trades'] >= 5 else "🔴"
                
                rankings.append({
                    'symbol': symbol,
                    'win_rate': win_rate,
                    'trades': stats['trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'total_pnl': stats['total_pnl'],
                    'avg_pnl': avg_pnl,
                    'score': score,
                    'recent_wr': recent_wr,
                    'data_quality': data_quality
                })
            
            # Sort by score
            rankings.sort(key=lambda x: x['score'], reverse=True)
            
            # Format message
            msg = "🏆 *ML Symbol Performance Rankings*\n"
            msg += "=" * 30 + "\n\n"
            
            # Summary
            total_symbols = len(rankings)
            profitable_symbols = sum(1 for r in rankings if r['total_pnl'] > 0)
            high_wr_symbols = sum(1 for r in rankings if r['win_rate'] >= 50)
            well_tested = sum(1 for r in rankings if r['trades'] >= 10)
            
            msg += f"📊 *Overview*\n"
            msg += f"Total Symbols: {total_symbols}\n"
            msg += f"Profitable: {profitable_symbols} ({profitable_symbols/total_symbols*100:.0f}%)\n" if total_symbols > 0 else ""
            msg += f"Win Rate ≥50%: {high_wr_symbols}\n"
            msg += f"Well Tested (10+ trades): {well_tested}\n\n"
            
            # Data quality legend
            msg += "📈 *Data Quality*\n"
            msg += "🟢 10+ trades (reliable)\n"
            msg += "🟡 5-9 trades (moderate)\n"
            msg += "🔴 <5 trades (limited)\n\n"
            
            # Top performers
            msg += "✅ *Top 10 Performers*\n"
            msg += "```\n"
            msg += f"{'#':<3} {'Symbol':<10} {'WR%':>6} {'Trades':>7} {'PnL':>8} {'Q'}\n"
            msg += "-" * 40 + "\n"
            
            for i, r in enumerate(rankings[:10], 1):
                msg += f"{i:<3} {r['symbol']:<10} {r['win_rate']:>5.1f}% {r['trades']:>7} ${r['total_pnl']:>7.2f} {r['data_quality']}\n"
            msg += "```\n\n"
            
            # Bottom performers (if more than 10)
            if len(rankings) > 10:
                msg += "❌ *Bottom 5 Performers*\n"
                msg += "```\n"
                msg += f"{'Symbol':<10} {'WR%':>6} {'Trades':>7} {'PnL':>8}\n"
                msg += "-" * 35 + "\n"
                
                bottom = rankings[-5:] if len(rankings) > 5 else []
                for r in bottom:
                    msg += f"{r['symbol']:<10} {r['win_rate']:>5.1f}% {r['trades']:>7} ${r['total_pnl']:>7.2f}\n"
                msg += "```\n\n"
            
            # Trending symbols
            trending_up = [r for r in rankings if r['recent_wr'] > r['win_rate'] + 10 and r['trades'] >= 5]
            trending_down = [r for r in rankings if r['recent_wr'] < r['win_rate'] - 10 and r['trades'] >= 5]
            
            if trending_up or trending_down:
                msg += "📈 *Trending*\n"
                if trending_up:
                    msg += "⬆️ Improving: " + ", ".join([r['symbol'] for r in trending_up[:3]]) + "\n"
                if trending_down:
                    msg += "⬇️ Declining: " + ", ".join([r['symbol'] for r in trending_down[:3]]) + "\n"
                msg += "\n"
            
            # ML recommendations
            msg += "🎯 *ML Recommendations*\n"
            
            # Find best reliable performer
            reliable = [r for r in rankings if r['trades'] >= 10]
            if reliable:
                best_reliable = reliable[0]
                msg += f"Most Reliable: {best_reliable['symbol']} "
                msg += f"({best_reliable['win_rate']:.1f}% in {best_reliable['trades']} trades)\n"
            
            # Find most profitable
            if rankings:
                most_profitable = max(rankings, key=lambda x: x['total_pnl'])
                if most_profitable['total_pnl'] > 0:
                    msg += f"Most Profitable: {most_profitable['symbol']} "
                    msg += f"(${most_profitable['total_pnl']:.2f})\n"
            
            # Symbols to watch (good WR but limited data)
            watch_list = [r for r in rankings if r['win_rate'] >= 60 and 3 <= r['trades'] < 10]
            if watch_list:
                msg += f"Watch List: " + ", ".join([r['symbol'] for r in watch_list[:5]]) + "\n"
            
            # Symbols to avoid
            avoid = [r for r in rankings if r['win_rate'] < 30 and r['trades'] >= 5]
            if avoid:
                msg += f"Consider Avoiding: " + ", ".join([r['symbol'] for r in avoid[:3]]) + "\n"
            
            msg += "\n_Refresh with /mlrankings_"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in ml_rankings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error generating rankings: {str(e)[:100]}")
    
    async def phantom_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show phantom trade statistics, with an optional strategy filter."""
        try:
            strategy_filter = None
            if ctx.args:
                strategy_filter = ctx.args[0].lower()
                if strategy_filter not in ['trend', 'reversion']:
                    await update.message.reply_text("Invalid strategy. Choose `trend` or `reversion`.")
                    return

            msg = "👻 *Phantom Trade Statistics*\n"
            msg += f"_Strategy: {strategy_filter.title() if strategy_filter else 'All'}_\n"
            msg += "━" * 25 + "\n\n"

            from phantom_trade_tracker import get_phantom_tracker
            phantom_tracker = get_phantom_tracker()

            all_phantoms = []
            for trades in phantom_tracker.phantom_trades.values():
                all_phantoms.extend(trades)

            if strategy_filter:
                all_phantoms = [p for p in all_phantoms if hasattr(p, 'strategy_name') and p.strategy_name.lower() == strategy_filter]

            if not all_phantoms:
                msg += "No phantom trades recorded for this filter."
                await self.safe_reply(update, msg)
                return

            # Calculate stats for the filtered set
            executed = [p for p in all_phantoms if p.was_executed]
            rejected = [p for p in all_phantoms if not p.was_executed]
            rejected_wins = [p for p in rejected if p.outcome == 'win']
            rejected_losses = [p for p in rejected if p.outcome == 'loss']
            missed_profit = sum(p.pnl_percent for p in rejected_wins if p.pnl_percent)
            avoided_loss = sum(abs(p.pnl_percent) for p in rejected_losses if p.pnl_percent)

            msg += "📊 *Overview*\n"
            msg += f"• Total signals tracked: {len(all_phantoms)}\n"
            msg += f"• Executed trades: {len(executed)}\n"
            msg += f"• Phantom trades: {len(rejected)}\n"
            
            # Verify counts add up
            if len(all_phantoms) != (len(executed) + len(rejected)):
                msg += f"⚠️ *Count mismatch: {len(all_phantoms)} ≠ {len(executed) + len(rejected)}*\n"
            msg += "\n"

            if rejected:
                msg += "🚫 *Rejected Trade Analysis*\n"
                rejected_wr = (len(rejected_wins) / len(rejected)) * 100 if rejected else 0
                msg += f"• Rejected Win Rate: {rejected_wr:.1f}% ({len(rejected_wins)}/{len(rejected)})\n"
                msg += f"• Missed Profit: +{missed_profit:.2f}%\n"
                msg += f"• Avoided Loss: -{avoided_loss:.2f}%\n"
                net_impact = missed_profit - avoided_loss
                msg += f"• *Net Impact: {net_impact:+.2f}%*\n"
            
            # Add executed trade analysis if available
            if executed:
                executed_wins = [p for p in executed if p.outcome == 'win']
                executed_losses = [p for p in executed if p.outcome == 'loss']
                if executed_wins or executed_losses:
                    msg += "\n✅ *Executed Trade Analysis*\n"
                    executed_wr = (len(executed_wins) / len(executed)) * 100 if executed else 0
                    msg += f"• Executed Win Rate: {executed_wr:.1f}% ({len(executed_wins)}/{len(executed)})\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in phantom_stats: {e}")
            await update.message.reply_text("Error getting phantom statistics")
    
    async def phantom_detail(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show detailed phantom statistics for a symbol"""
        try:
            # Get symbol from command
            if not ctx.args:
                await update.message.reply_text(
                    "Please specify a symbol\n"
                    "Usage: `/phantom\\_detail BTCUSDT`",
                    parse_mode='Markdown'
                )
                return
            
            symbol = ctx.args[0].upper()
            
            msg = f"👻 *Phantom Stats: {symbol}*\n"
            msg += "━" * 25 + "\n\n"
            
            # Get phantom tracker
            try:
                from phantom_trade_tracker import get_phantom_tracker
                phantom_tracker = get_phantom_tracker()
            except Exception as e:
                logger.error(f"Error importing phantom tracker: {e}")
                await update.message.reply_text("⚠️ Phantom tracker not available")
                return
            
            # Get symbol-specific statistics
            stats = phantom_tracker.get_phantom_stats(symbol)
            
            if stats['total'] == 0:
                msg += f"No phantom trades recorded for {symbol}\n"
                msg += "\n_Try another symbol or wait for more signals_"
                await self.safe_reply(update, msg)
                return
            
            # Overview for this symbol
            msg += "📊 *Overview*\n"
            msg += f"• Total signals: {stats['total']}\n"
            msg += f"• Executed: {stats['executed']}\n"
            msg += f"• Phantoms: {stats['rejected']}\n"
            if stats['total'] > 0:
                execution_rate = (stats['executed'] / stats['total']) * 100
                msg += f"• Execution rate: {execution_rate:.1f}%\n"
            msg += "\n"
            
            # Rejection analysis
            rejection_stats = stats['rejection_stats']
            if rejection_stats['total_rejected'] > 0:
                msg += "🚫 *Rejection Analysis*\n"
                msg += f"• Rejected trades: {rejection_stats['total_rejected']}\n"
                msg += f"• Would have won: {rejection_stats['would_have_won']}\n"
                msg += f"• Would have lost: {rejection_stats['would_have_lost']}\n"
                
                # Win rate of rejected trades
                if rejection_stats['total_rejected'] > 0:
                    rejected_wr = (rejection_stats['would_have_won'] / rejection_stats['total_rejected']) * 100
                    msg += f"• Rejected win rate: {rejected_wr:.1f}%\n"
                
                # Financial impact
                if rejection_stats['missed_profit_pct'] > 0:
                    msg += f"• Missed profit: +{rejection_stats['missed_profit_pct']:.2f}%\n"
                if rejection_stats['avoided_loss_pct'] > 0:
                    msg += f"• Avoided loss: -{rejection_stats['avoided_loss_pct']:.2f}%\n"
                
                # Net impact
                net_impact = rejection_stats['missed_profit_pct'] - rejection_stats['avoided_loss_pct']
                if net_impact != 0:
                    msg += f"• Net impact: {net_impact:+.2f}%\n"
                msg += "\n"
            
            # Recent phantom trades for this symbol
            if symbol in phantom_tracker.phantom_trades:
                recent_phantoms = phantom_tracker.phantom_trades[symbol][-5:]
                if recent_phantoms:
                    msg += "📜 *Recent Phantoms*\n"
                    for phantom in recent_phantoms:
                        if phantom.outcome:
                            outcome_emoji = "✅" if phantom.outcome == "win" else "❌"
                            msg += f"• Score {phantom.ml_score:.0f}: {outcome_emoji} "
                            msg += f"{phantom.side.upper()} {phantom.pnl_percent:+.2f}%\n"
                    msg += "\n"
            
            # Active phantom for this symbol
            if symbol in phantom_tracker.active_phantoms:
                phantom = phantom_tracker.active_phantoms[symbol]
                msg += "👀 *Currently Tracking*\n"
                msg += f"• {phantom.side.upper()} position\n"
                msg += f"• Entry: {phantom.entry_price:.4f}\n"
                msg += f"• ML Score: {phantom.ml_score:.1f}\n"
                msg += f"• Target: {phantom.take_profit:.4f}\n"
                msg += f"• Stop: {phantom.stop_loss:.4f}\n"
                msg += "\n"
            
            # ML insights
            msg += "💡 *ML Insights*\n"
            if rejection_stats['total_rejected'] > 0 and rejection_stats['would_have_won'] > rejection_stats['would_have_lost']:
                msg += "• ML may be too conservative\n"
                msg += "• Consider threshold adjustment\n"
            elif rejection_stats['total_rejected'] > 0 and rejection_stats['would_have_lost'] > rejection_stats['would_have_won']:
                msg += "• ML filtering effectively\n"
                msg += "• Avoiding losing trades\n"
            else:
                msg += "• Gathering more data\n"
                msg += "• Patterns emerging\n"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in phantom_detail: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error getting phantom details: {str(e)[:100]}")
    
    async def evolution_performance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML Evolution shadow performance"""
        try:
            msg = "🧬 *ML Evolution Performance*\n"
            msg += "━" * 25 + "\n\n"
            
            try:
                from ml_evolution_tracker import get_evolution_tracker
                tracker = get_evolution_tracker()
                summary = tracker.get_performance_summary()
            except Exception as e:
                logger.error(f"Error getting evolution tracker: {e}")
                await update.message.reply_text("Evolution tracking not available")
                return
            
            if 'status' in summary:
                msg += summary['status']
            else:
                # Overview
                msg += "📊 *Shadow Mode Performance*\n"
                msg += f"• Total signals analyzed: {summary['total_signals']}\n"
                msg += f"• Agreement rate: {summary['agreement_rate']:.1f}%\n"
                msg += f"• Completed comparisons: {summary['completed_comparisons']}\n"
                msg += "\n"
                
                # Performance comparison
                if summary['completed_comparisons'] > 0:
                    msg += "🎯 *Win Rate Comparison*\n"
                    msg += f"• General model: {summary['general_win_rate']:.1f}%\n"
                    msg += f"• Evolution model: {summary['evolution_win_rate']:.1f}%\n"
                    
                    diff = summary['evolution_win_rate'] - summary['general_win_rate']
                    if diff > 0:
                        msg += f"• Evolution advantage: +{diff:.1f}%\n"
                    else:
                        msg += f"• General advantage: {abs(diff):.1f}%\n"
                    msg += "\n"
                
                # Symbol insights
                insights = summary.get('symbol_insights', {})
                if insights:
                    msg += "🔍 *Top Symbol Benefits*\n"
                    sorted_symbols = sorted(insights.items(), 
                                          key=lambda x: x[1]['evolution_advantage'], 
                                          reverse=True)[:5]
                    
                    for symbol, data in sorted_symbols:
                        advantage = data['evolution_advantage']
                        if advantage != 0:
                            msg += f"• {symbol}: "
                            if advantage > 0:
                                msg += f"+{advantage} better decisions\n"
                            else:
                                msg += f"{advantage} worse decisions\n"
                    msg += "\n"
                
                # Recommendation
                msg += "💡 *Recommendation*\n"
                msg += f"{summary['recommendation']}\n\n"
                
                msg += "_Shadow mode continues learning..._"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in evolution_performance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error getting evolution performance: {str(e)[:100]}")
    
    async def force_retrain_ml(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force retrain ML models to reset feature expectations"""
        try:
            msg = "🔧 *ML Force Retrain*\n"
            msg += "━" * 25 + "\n\n"
            
            # Get ML scorer
            ml_scorer = self.shared.get("ml_scorer")
            if not ml_scorer:
                await update.message.reply_text("⚠️ ML scorer not available")
                return
            
            # Get current status before reset
            stats_before = ml_scorer.get_stats()
            
            msg += "📊 *Current Status*\n"
            msg += f"• Models: {', '.join(stats_before['models_active']) if stats_before['models_active'] else 'None'}\n"
            msg += f"• Feature version: {stats_before.get('model_feature_version', 'unknown')}\n"
            msg += f"• Feature count: {stats_before.get('feature_count', 'unknown')}\n"
            msg += f"• Completed trades: {stats_before['completed_trades']}\n\n"
            
            # Force retrain
            ml_scorer.force_retrain_models()
            
            msg += "✅ *Actions Taken*\n"
            msg += "• Cleared existing models\n"
            msg += "• Reset scaler\n"
            msg += "• Cleared Redis cache\n"
            msg += "• Reset to original features (22)\n\n"
            
            msg += "📝 *What Happens Next*\n"
            msg += "• Models will use rule-based scoring\n"
            msg += "• Will retrain on next trade completion\n"
            msg += "• Will detect available features automatically\n"
            msg += "• No interruption to trading\n\n"
            
            msg += "⚡ *Commands*\n"
            msg += "• `/ml` - Check ML status\n"
            msg += "• `/stats` - View trading stats"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in force_retrain_ml: {e}")
            await update.message.reply_text(f"Error forcing ML retrain: {str(e)[:100]}")
    
    async def ml_patterns(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show detailed ML patterns and insights for all strategies (when trained)"""
        try:
            def esc(s: object) -> str:
                text = str(s)
                return (
                    text.replace("\\", "\\\\")
                        .replace("_", "\\_")
                        .replace("*", "\\*")
                        .replace("[", "\\[")
                        .replace("`", "\\`")
                )

            response_text = "🧠 *ML Pattern Analysis*\n"
            response_text += "━" * 25 + "\n\n"

            # Get ML scorers
            trend_scorer_inst = None
            mean_reversion_scorer = None
            enhanced_mr_scorer = None
            scalp_scorer = None
            try:
                trend_scorer_inst = get_trend_scorer()
            except Exception:
                pass
            try:
                from enhanced_mr_scorer import get_enhanced_mr_scorer
                enhanced_mr_scorer = get_enhanced_mr_scorer()
            except Exception:
                pass
            try:
                from autobot.strategies.scalp.scorer import get_scalp_scorer
                scalp_scorer = get_scalp_scorer()
            except Exception:
                pass

            scorers = {}
            if trend_scorer_inst:
                scorers["Trend Pullback"] = trend_scorer_inst
            # Original MR disabled per configuration; do not include in patterns

            for strategy_name, scorer in scorers.items():
                response_text += f"*{strategy_name}:*\n"
                if not scorer or not hasattr(scorer, 'get_learned_patterns'):
                    response_text += "  ❌ *Pattern Analysis Not Available*\n"
                    response_text += "  _ML system not yet initialized or trained._\n\n"
                    continue

                patterns = scorer.get_learned_patterns()

                if not patterns or all(not v for v in patterns.values()):
                    stats = scorer.get_stats()
                    response_text += f"  📊 *Collecting Data...*\n"
                    response_text += f"  • Completed trades: {stats.get('completed_trades', 0)}\n"
                    response_text += f"  • Status: {stats.get('status', 'Learning')}\n"
                    response_text += "  _Patterns will emerge after more trades._\n\n"
                    continue

                # Feature Importance (Top 10)
                if patterns.get('feature_importance'):
                    response_text += "  📊 *Feature Importance (Top 10)*\n"
                    response_text += "  _What drives winning trades_\n"
                    
                    for i, (feat, imp) in enumerate(list(patterns['feature_importance'].items())[:10], 1):
                        feat_name = feat.replace('_', ' ').title()
                        bar_length = int(imp / 10)
                        bar = '█' * bar_length + '░' * (10 - bar_length)
                        response_text += f"  {i}. {esc(feat_name)}\n"
                        response_text += f"     {esc(bar)} {imp:.1f}%\n"
                    response_text += "\n"
                
                # Time Analysis
                time_patterns = patterns.get('time_patterns', {})
                if time_patterns:
                    response_text += "  ⏰ *Time-Based Insights*\n"
                    
                    if time_patterns.get('best_hours'):
                        response_text += "  🌟 *Golden Hours*\n"
                        for hour, stats in list(time_patterns['best_hours'].items())[:5]:
                            response_text += f"  • {esc(hour)} → {esc(stats)}\n"
                        response_text += "\n"
                    
                    if time_patterns.get('worst_hours'):
                        response_text += "  ⚠️ *Danger Hours*\n"
                        for hour, stats in list(time_patterns['worst_hours'].items())[:5]:
                            response_text += f"  • {esc(hour)} → {esc(stats)}\n"
                        response_text += "\n"
                    
                    if time_patterns.get('session_performance'):
                        response_text += "  🌍 *Market Sessions*\n"
                        for session, perf in time_patterns['session_performance'].items():
                            if 'WR' in perf:
                                wr = float(perf.split('%')[0].split()[-1])
                                emoji = '🟢' if wr >= 50 else '🔴'
                            else:
                                emoji = '⚪'
                            response_text += f"  {emoji} {esc(session)}: {esc(perf)}\n"
                        response_text += "\n"
                
                # Market Conditions
                market_conditions = patterns.get('market_conditions', {})
                if market_conditions:
                    response_text += "  📈 *Market Condition Analysis*\n"
                    
                    if market_conditions.get('volatility_impact'):
                        response_text += "  🌊 *Volatility Performance*\n"
                        for vol_type, stats in market_conditions['volatility_impact'].items():
                            if 'WR' in stats:
                                wr = float(stats.split('%')[0].split()[-1])
                                emoji = '✅' if wr >= 50 else '❌'
                            else:
                                emoji = '➖'
                            response_text += f"  {emoji} {esc(vol_type.title())}: {esc(stats)}\n"
                        response_text += "\n"
                    
                    if market_conditions.get('volume_impact'):
                        response_text += "  📊 *Volume Analysis*\n"
                        for vol_type, stats in market_conditions['volume_impact'].items():
                            vol_name = vol_type.replace('_', ' ').title()
                            if 'WR' in stats:
                                wr = float(stats.split('%')[0].split()[-1])
                                emoji = '✅' if wr >= 50 else '❌'
                            else:
                                emoji = '➖'
                            response_text += f"  {emoji} {esc(vol_name)}: {esc(stats)}\n"
                        response_text += "\n"
                    
                    if market_conditions.get('trend_impact'):
                        response_text += "  📉 *Trend Analysis*\n"
                        for trend_type, stats in market_conditions['trend_impact'].items():
                            trend_name = trend_type.replace('_', ' ').title()
                            if 'WR' in stats:
                                wr = float(stats.split('%')[0].split()[-1])
                                emoji = '✅' if wr >= 50 else '❌'
                            else:
                                emoji = '➖'
                            response_text += f"  {emoji} {esc(trend_name)}: {esc(stats)}\n"
                        response_text += "\n"
                
                # Winning vs Losing Patterns
                if patterns.get('winning_patterns') or patterns.get('losing_patterns'):
                    response_text += "  🎯 *Trade Outcome Patterns*\n"
                    
                    if patterns.get('winning_patterns'):
                        response_text += "  ✅ *Common in Winners*\n"
                        for pattern in patterns['winning_patterns']:
                            response_text += f"  • {esc(pattern)}\n"
                        response_text += "\n"
                    
                    if patterns.get('losing_patterns'):
                        response_text += "  ❌ *Common in Losers*\n"
                        for pattern in patterns['losing_patterns']:
                            response_text += f"  • {esc(pattern)}\n"
                        response_text += "\n"
                
                # Summary insights
                response_text += "  💡 *Key Takeaways*\n"
                response_text += "  • Focus on high-importance features\n"
                response_text += "  • Trade during golden hours\n"
                response_text += "  • Adapt to market conditions\n"
                response_text += "  • Avoid danger patterns\n\n"
            
            # End of Trend (classic). Enhanced MR follows.

            # Enhanced MR insights (now with patterns)
            if enhanced_mr_scorer:
                try:
                    info = enhanced_mr_scorer.get_enhanced_stats()
                    response_text += "🧠 *Enhanced MR (Ensemble) Insights*\n"
                    response_text += f"• Status: {info.get('status')}\n"
                    response_text += f"• Executed: {info.get('completed_trades',0)} | Phantom: {info.get('phantom_count',0)} | Total: {info.get('total_combined',0)}\n"
                    response_text += f"• Threshold: {info.get('current_threshold','?')}\n"
                    # Patterns
                    emr_patterns = enhanced_mr_scorer.get_enhanced_patterns()
                    fi = emr_patterns.get('feature_importance', {})
                    if fi:
                        response_text += "\n  📊 *Feature Importance*\n"
                        for i, (feat, imp) in enumerate(list(fi.items())[:10], 1):
                            feat_name = feat.replace('_', ' ').title()
                            bar_len = max(1, min(10, int(float(imp)/10)))
                            bar = '█' * bar_len + '░' * (10 - bar_len)
                            response_text += f"  {i}. {esc(feat_name)}\n     {esc(bar)} {float(imp):.1f}%\n"
                    tp = emr_patterns.get('time_patterns', {})
                    if tp:
                        response_text += "\n  ⏰ *Time-Based Insights*\n"
                        if tp.get('best_hours'):
                            response_text += "  🌟 *Golden Hours*\n"
                            for h, txt in list(tp['best_hours'].items())[:5]:
                                response_text += f"  • {esc(h)}: {esc(txt)}\n"
                        if tp.get('worst_hours'):
                            response_text += "  ⚠️ *Danger Hours*\n"
                            for h, txt in list(tp['worst_hours'].items())[:5]:
                                response_text += f"  • {esc(h)}: {esc(txt)}\n"
                        if tp.get('session_performance'):
                            response_text += "  🌍 *Market Sessions*\n"
                            for s, txt in tp['session_performance'].items():
                                response_text += f"  • {esc(s)}: {esc(txt)}\n"
                    mc = emr_patterns.get('market_conditions', {})
                    if mc:
                        response_text += "\n  🌡️ *Market Condition Patterns*\n"
                        for k, v in mc.items():
                            title = k.replace('_', ' ').title()
                            response_text += f"  {esc(title)}:\n"
                            for bk, txt in v.items():
                                response_text += f"   • {esc(bk)}: {esc(txt)}\n"
                    wp = emr_patterns.get('winning_patterns', [])
                    if wp:
                        response_text += "\n  ✅ *Common in Winners*\n"
                        for p in wp[:5]:
                            response_text += f"  • {esc(p)}\n"
                    lp = emr_patterns.get('losing_patterns', [])
                    if lp:
                        response_text += "\n  ❌ *Common in Losers*\n"
                        for p in lp[:5]:
                            response_text += f"  • {esc(p)}\n"
                    response_text += "\n"
                except Exception:
                    response_text += "🧠 *Enhanced MR (Ensemble) Insights*\n  ❌ Not available\n\n"

            # Scalp insights (status + patterns when available)
            if scalp_scorer:
                try:
                    response_text += "🩳 *Scalp ML Insights*\n"
                    ready = getattr(scalp_scorer,'is_ml_ready',False)
                    response_text += f"• Status: {'✅ Ready' if ready else '⏳ Training'}\n"
                    try:
                        ri = scalp_scorer.get_retrain_info()
                        response_text += f"• Records: {ri.get('total_records',0)} | Trainable: {ri.get('trainable_size',0)}\n"
                        response_text += f"• Next retrain in: {ri.get('trades_until_next_retrain',0)} trades\n"
                    except Exception:
                        response_text += f"• Samples: {getattr(scalp_scorer,'completed_trades',0)}\n"
                    response_text += f"• Threshold: {getattr(scalp_scorer,'min_score',75)}\n"

                    # Patterns (when RF trained)
                    sp = scalp_scorer.get_patterns() if ready else {}
                    fi = (sp or {}).get('feature_importance', {})
                    if fi:
                        response_text += "\n  📊 *Feature Importance (Scalp)*\n"
                        for i, (feat, imp) in enumerate(list(fi.items())[:8], 1):
                            feat_name = feat.replace('_',' ').title()
                            bar_len = max(1, min(10, int(float(imp)/10)))
                            bar = '█'*bar_len + '░'*(10-bar_len)
                            response_text += f"  {i}. {esc(feat_name)}\n     {esc(bar)} {float(imp):.1f}%\n"
                    tp = (sp or {}).get('time_patterns', {})
                    if tp:
                        response_text += "\n  ⏰ *Time-Based (Scalp)*\n"
                        if tp.get('best_hours'):
                            response_text += "  🌟 *Best Hours*\n"
                            for h, txt in list(tp['best_hours'].items())[:5]:
                                response_text += f"  • {esc(h)}: {esc(txt)}\n"
                        if tp.get('session_performance'):
                            response_text += "  🌍 *Sessions*\n"
                            for s, txt in tp['session_performance'].items():
                                response_text += f"  • {esc(s)}: {esc(txt)}\n"
                    mc = (sp or {}).get('market_conditions', {})
                    if mc:
                        response_text += "\n  🌡️ *Conditions (Scalp)*\n"
                        for k, v in mc.items():
                            title = k.replace('_',' ').title()
                            response_text += f"  {esc(title)}:\n"
                            for bk, txt in v.items():
                                response_text += f"   • {esc(bk)}: {esc(txt)}\n"
                    response_text += "\n"
                except Exception:
                    response_text += "🩳 *Scalp ML Insights*\n  ❌ Not available\n\n"

            await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"Error in ml_patterns: {e}")
            await update.message.reply_text("Error getting ML patterns")

    async def shadow_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Compare baseline executed vs shadow-simulated ML nudges per strategy."""
        try:
            try:
                from shadow_trade_simulator import get_shadow_tracker
                st = get_shadow_tracker()
                s_stats = st.get_stats()
            except Exception:
                s_stats = {}

            # Baseline from executed trades
            tt = self.shared.get('trade_tracker')
            trades = getattr(tt, 'trades', []) or []
            def _baseline_for(key: str):
                wins = losses = 0
                for t in trades:
                    strat = (getattr(t, 'strategy_name', '') or '').lower()
                    grp = 'trend' if ('trend' in strat or 'pullback' in strat) else 'enhanced_mr' if ('mr' in strat or 'reversion' in strat) else None
                    if grp == key:
                        pnl = float(getattr(t, 'pnl_usd', 0))
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                total = wins + losses
                wr = (wins/total*100.0) if total else 0.0
                return wins, losses, total, wr

                pbw, pbl, pbt, pbwr = _baseline_for('trend')
            mrw, mrl, mrt, mrwr = _baseline_for('enhanced_mr')

            lines = [
                "🧪 *Shadow vs Baseline*",
                "",
                    "🔵 Trend",
                f"• Baseline: W {pbw} / L {pbl} (WR {pbwr:.1f}%)",
                    f"• Shadow:   W {s_stats.get('trend',{}).get('wins',0)} / L {s_stats.get('trend',{}).get('losses',0)} (WR {s_stats.get('trend',{}).get('wr',0.0):.1f}%)",
                "",
                "🌀 Mean Reversion",
                f"• Baseline: W {mrw} / L {mrl} (WR {mrwr:.1f}%)",
                f"• Shadow:   W {s_stats.get('enhanced_mr',{}).get('wins',0)} / L {s_stats.get('enhanced_mr',{}).get('losses',0)} (WR {s_stats.get('enhanced_mr',{}).get('wr',0.0):.1f}%)",
            ]
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in shadow_stats: {e}")
            await update.message.reply_text("Error getting shadow stats")
    
    async def ml_retrain_info(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML retrain countdown information"""
        try:
            msg = "🔄 *ML Retrain Status*\n"
            msg += "━" * 25 + "\n\n"
            
            # Check if ML scorer is available
            ml_scorer = self.shared.get("ml_scorer")
            
            if not ml_scorer or not hasattr(ml_scorer, 'get_retrain_info'):
                msg += "❌ *ML System Not Available*\n\n"
                msg += "ML retraining info requires:\n"
                msg += "• ML system enabled\n"
                msg += "• Bot running with ML\n"
                await self.safe_reply(update, msg)
                return
            
            # Get retrain info
            info = ml_scorer.get_retrain_info()
            
            # Current status
            msg += "📊 *Current Status*\n"
            msg += f"• ML Ready: {'✅ Yes' if info['is_ml_ready'] else '❌ No'}\n"
            msg += f"• Executed trades: {info['completed_trades']}\n"
            msg += f"• Phantom trades: {info['phantom_count']}\n"
            msg += f"• Combined total: {info['total_combined']}\n"
            msg += "\n"
            
            # Training history
            if info['is_ml_ready']:
                msg += "📈 *Training History*\n"
                msg += f"• Last trained at: {info['last_train_count']} trades\n"
                trades_since = info['total_combined'] - info['last_train_count']
                msg += f"• Trades since last: {trades_since}\n"
                msg += "\n"
            
            # Next retrain countdown
            msg += "⏳ *Next Retrain*\n"
            if info['trades_until_next_retrain'] == 0:
                if info['is_ml_ready']:
                    msg += "🟢 **Ready to retrain!**\n"
                    msg += "Will retrain on next trade completion\n"
                else:
                    msg += "🟢 **Ready for initial training!**\n"
                    msg += "Will train on next trade completion\n"
            else:
                msg += f"• Trades needed: **{info['trades_until_next_retrain']}**\n"
                msg += f"• Will retrain at: {info['next_retrain_at']} total trades\n"
                
                # Progress bar - calculate based on actual retrain interval
                if info['is_ml_ready']:
                    # For retrain: how far through the 100-trade cycle
                    trades_in_cycle = ml_scorer.RETRAIN_INTERVAL - info['trades_until_next_retrain']
                    progress = (trades_in_cycle / ml_scorer.RETRAIN_INTERVAL) * 100
                else:
                    # For initial training: progress toward MIN_TRADES_FOR_ML
                    trades_so_far = ml_scorer.MIN_TRADES_FOR_ML - info['trades_until_next_retrain']
                    progress = (trades_so_far / ml_scorer.MIN_TRADES_FOR_ML) * 100
                
                progress = max(0, min(100, progress))
                filled = int(progress / 10)
                bar = '█' * filled + '░' * (10 - filled)
                msg += f"• Progress: {bar} {progress:.0f}%\n"
            
            msg += "\n"
            
            # Info about retraining
            msg += "ℹ️ *Retrain Info*\n"
            if not info['is_ml_ready']:
                msg += f"• Initial training after {ml_scorer.MIN_TRADES_FOR_ML} trades\n"
            msg += f"• Retrain interval: Every {ml_scorer.RETRAIN_INTERVAL} trades\n"
            msg += "• Both executed and phantom trades count\n"
            msg += "• Models improve with each retrain\n"
            msg += "\n"
            
            # Commands
            msg += "⚡ *Commands*\n"
            msg += "• `/force_retrain` - Force immediate retrain\n"
            msg += "• `/ml` - View ML status\n"
            msg += "• `/phantom` - View phantom trades"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in ml_retrain_info: {e}")
            await update.message.reply_text("Error getting ML retrain info")
    
    async def cluster_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show symbol cluster status using hardcoded clusters"""
        try:
            msg = "🎯 *Symbol Cluster Status*\n"
            msg += "━" * 25 + "\n\n"
            
            # Use hardcoded clusters
            try:
                from hardcoded_clusters import (get_hardcoded_clusters, get_cluster_name, 
                                               get_symbol_cluster, get_cluster_description,
                                               BLUE_CHIP, STABLE, MEME_VOLATILE, MID_CAP)
                
                # Get all hardcoded clusters
                all_clusters = get_hardcoded_clusters()
                
                # Count hardcoded symbols
                cluster_counts = {
                    1: len(BLUE_CHIP),
                    2: len(STABLE),
                    3: len(MEME_VOLATILE),
                    4: len(MID_CAP),
                    5: 0  # Will be calculated from active symbols
                }
                
                # Count small cap symbols from trading symbols
                trading_symbols = self.shared.get("frames", {}).keys()
                known_symbols = set(all_clusters.keys())
                small_cap_count = len([s for s in trading_symbols if s not in known_symbols])
                cluster_counts[5] = small_cap_count
                
                msg += "📊 *Hardcoded Cluster Distribution*\n"
                for i in range(1, 6):
                    name = get_cluster_name(i)
                    count = cluster_counts[i]
                    if count > 0:
                        msg += f"• {name}: {count} symbols\n"
                
                msg += "\n🔍 *Sample Symbols by Cluster*\n"
                
                # Show examples from each cluster
                msg += f"\n*{get_cluster_name(1)}:*\n"
                for symbol in BLUE_CHIP[:5]:
                    msg += f"• {symbol}\n"
                    
                msg += f"\n*{get_cluster_name(3)}:*\n"
                for symbol in MEME_VOLATILE[:5]:
                    msg += f"• {symbol}\n"
                    
                msg += f"\n*{get_cluster_name(4)}:*\n"  
                for symbol in MID_CAP[:5]:
                    msg += f"• {symbol}\n"
                
                # Show current positions with clusters
                positions = self.shared.get("book", {}).positions
                if positions:
                    msg += "\n📈 *Your Open Positions:*\n"
                    for symbol in list(positions.keys())[:10]:
                        cluster_id = get_symbol_cluster(symbol)
                        cluster_name = get_cluster_name(cluster_id)
                        msg += f"• {symbol}: {cluster_name}\n"
                
                msg += "\n✅ *Using hardcoded clusters*\n"
                msg += "_No generation needed - always available_"
                    
            except Exception as e:
                logger.error(f"Error loading clusters: {e}")
                msg += "❌ Error loading cluster data\n"
                msg += "Basic clustering may still be active\n"
            
            msg += "\n💡 *Commands*\n"
            msg += "• `/update_clusters` - Update clusters\n"
            msg += "• `/ml` - View ML status"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in cluster_status: {e}")
            await update.message.reply_text("Error getting cluster status")
    
    async def update_clusters(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Inform user that hardcoded clusters are always up to date"""
        try:
            from hardcoded_clusters import get_hardcoded_clusters, BLUE_CHIP, STABLE, MEME_VOLATILE, MID_CAP
            
            msg = "✅ *Cluster Update Status*\n"
            msg += "━" * 25 + "\n\n"
            msg += "📌 *Using Hardcoded Clusters*\n"
            msg += "No update needed - clusters are hardcoded!\n\n"
            
            msg += "📊 *Current Distribution:*\n"
            msg += f"• Blue Chip: {len(BLUE_CHIP)} symbols\n"
            msg += f"• Stable: {len(STABLE)} symbols\n"
            msg += f"• Meme/Volatile: {len(MEME_VOLATILE)} symbols\n"
            msg += f"• Mid-Cap: {len(MID_CAP)} symbols\n"
            msg += f"• Small Cap: All others\n\n"
            
            msg += "💡 *Benefits:*\n"
            msg += "• Always available\n"
            msg += "• No generation needed\n"
            msg += "• Consistent classification\n"
            msg += "• Based on market research\n\n"
            
            msg += "Use `/clusters` to view full details"
            
            await update.message.reply_text(msg)
                
        except Exception as e:
            logger.error(f"Error in update_clusters: {e}")
            await update.message.reply_text("Error triggering cluster update")
    
    async def set_ml_threshold(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set ML threshold for signal acceptance"""
        try:
            # Get ML scorer
            ml_scorer = self.shared.get("ml_scorer")
            if not ml_scorer:
                await update.message.reply_text("❌ ML system not available")
                return
            
            # Check if arguments provided
            if not ctx.args:
                # Show current threshold
                msg = "🤖 *ML Threshold Settings*\n"
                msg += "━" * 25 + "\n\n"
                msg += f"Current threshold: {ml_scorer.min_score}\n\n"
                msg += "📊 *Threshold Guide:*\n"
                msg += "• 50-60: Very lenient (more signals)\n"
                msg += "• 60-70: Moderate\n"
                msg += "• 70-80: Standard (default)\n" 
                msg += "• 80-90: Conservative\n"
                msg += "• 90-100: Very strict (fewer signals)\n\n"
                msg += "Usage: `/set_ml_threshold 75`"
                await self.safe_reply(update, msg)
                return
            
            try:
                new_threshold = float(ctx.args[0])
                
                # Validate threshold
                if new_threshold < 0 or new_threshold > 100:
                    await update.message.reply_text("❌ Threshold must be between 0 and 100")
                    return
                
                # Update threshold
                old_threshold = ml_scorer.min_score
                ml_scorer.min_score = new_threshold
                
                # Save to Redis if available
                if ml_scorer.redis_client:
                    try:
                        ml_scorer.redis_client.set('iml:threshold', new_threshold)
                    except:
                        pass
                
                # Prepare response
                msg = f"✅ *ML Threshold Updated*\n\n"
                msg += f"• Old threshold: {old_threshold}\n"
                msg += f"• New threshold: {new_threshold}\n\n"
                
                # Add interpretation
                if new_threshold < 60:
                    msg += "⚡ Very lenient - Expect more signals with higher risk\n"
                elif new_threshold < 70:
                    msg += "📊 Moderate - Balanced approach\n"
                elif new_threshold < 80:
                    msg += "✅ Standard - Good balance of quality and quantity\n"
                elif new_threshold < 90:
                    msg += "🛡️ Conservative - Higher quality signals only\n"
                else:
                    msg += "🏆 Very strict - Only the best signals\n"
                
                # Add stats if available
                stats = ml_scorer.get_stats()
                if stats.get('completed_trades', 0) > 0:
                    msg += f"\nBased on {stats['completed_trades']} completed trades"
                
                await self.safe_reply(update, msg)
                logger.info(f"ML threshold updated from {old_threshold} to {new_threshold}")
                
            except ValueError:
                await update.message.reply_text("❌ Invalid threshold. Please provide a number between 0 and 100")
            
        except Exception as e:
            logger.error(f"Error in set_ml_threshold: {e}")
            await update.message.reply_text("Error updating ML threshold")

    async def _set_highml_common(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, strat_key: str, default_key_path: list[str]):
        try:
            if not ctx.args:
                await update.message.reply_text(f"Usage: /{strat_key}_highml <value> (0-100)")
                return
            val = float(ctx.args[0])
            if val < 0 or val > 100:
                await update.message.reply_text("❌ Value must be between 0 and 100")
                return
            # Update in-memory config
            cfg = self.shared.get("config", {}) or {}
            # Navigate/create path like ['scalp','exec','high_ml_force']
            node = cfg
            for ki in default_key_path[:-1]:
                if ki not in node or not isinstance(node[ki], dict):
                    node[ki] = {}
                node = node[ki]
            old = node.get(default_key_path[-1])
            node[default_key_path[-1]] = float(val)
            # Keep reference parity with bot instance
            try:
                bot = self.shared.get('bot_instance')
                if bot is not None:
                    bot.config = cfg
            except Exception:
                pass
            # Best-effort persist hint in Redis for visibility across restarts
            try:
                r = self.shared.get('bot_instance')._redis if self.shared.get('bot_instance') else None
                if r is not None:
                    r.set(f"cfg:{strat_key}:high_ml_force", str(val))
            except Exception:
                pass
            await self.safe_reply(update, f"✅ {strat_key.upper()} high‑ML threshold updated: {old} → {val}")
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: 80")
        except Exception as e:
            logger.error(f"set_highml_common error: {e}")
            await update.message.reply_text("Error updating threshold")

    async def set_scalp_highml(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._set_highml_common(update, ctx, 'scalp', ['scalp','exec','high_ml_force'])

    async def set_mr_highml(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._set_highml_common(update, ctx, 'mr', ['mr','exec','high_ml_force'])

    async def set_trend_highml(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # Ensure Trend execution is enabled when user updates threshold
        try:
            cfg = self.shared.get("config", {}) or {}
            if not isinstance(cfg.get('trend', {}).get('exec', {}), dict):
                cfg.setdefault('trend',{}).setdefault('exec',{})
            cfg['trend']['exec']['enabled'] = True
            bot = self.shared.get('bot_instance')
            if bot is not None:
                bot.config = cfg
        except Exception:
            pass
        await self._set_highml_common(update, ctx, 'trend', ['trend','exec','high_ml_force'])
    
    async def htf_sr_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show HTF support/resistance status (disabled)."""
        await self.safe_reply(update, "HTF S/R module disabled.")
        return
        try:
            from autobot.utils.sr_levels import mtf_sr
            
            msg = "📊 *HTF Support/Resistance Status*\n"
            msg += "━" * 30 + "\n\n"
            
            # Count symbols with levels
            symbols_with_levels = 0
            total_levels = 0
            
            for symbol, levels in mtf_sr.sr_levels.items():
                if levels:
                    symbols_with_levels += 1
                    total_levels += len(levels)
            
            msg += f"📈 *Overview:*\n"
            msg += f"• Symbols analyzed: {len(mtf_sr.sr_levels)}\n"
            msg += f"• Symbols with levels: {symbols_with_levels}\n"
            msg += f"• Total levels tracked: {total_levels}\n"
            msg += f"• Update interval: {mtf_sr.update_interval} candles\n\n"
            
            # Show specific symbol if provided
            if ctx.args:
                symbol = ctx.args[0].upper()
                if symbol in mtf_sr.sr_levels:
                    # Get current price from frames
                    frames = self.shared.get("frames", {})
                    current_price = None
                    if symbol in frames and not frames[symbol].empty:
                        current_price = frames[symbol]['close'].iloc[-1]
                    
                    if current_price:
                        msg += f"📍 *{symbol} Levels:*\n"
                        msg += f"Current Price: {current_price:.4f}\n"
                        
                        # Get price-validated levels
                        validated_levels = mtf_sr.get_price_validated_levels(symbol, current_price)
                        
                        # Group by type
                        resistance_levels = [(l, s) for l, s, t in validated_levels if t == 'resistance']
                        support_levels = [(l, s) for l, s, t in validated_levels if t == 'support']
                        
                        # Show top 5 of each
                        if resistance_levels:
                            msg += "\n🔴 *Resistance (above price):*\n"
                            for level, strength in resistance_levels[:5]:
                                distance_pct = ((level - current_price) / current_price) * 100
                                msg += f"• {level:.4f} (strength: {strength:.1f}, +{distance_pct:.2f}%)\n"
                        else:
                            msg += "\n🔴 *Resistance:* None above current price\n"
                        
                        if support_levels:
                            msg += "\n🟢 *Support (below price):*\n"
                            for level, strength in support_levels[:5]:
                                distance_pct = ((current_price - level) / level) * 100
                                msg += f"• {level:.4f} (strength: {strength:.1f}, -{distance_pct:.2f}%)\n"
                        else:
                            msg += "\n🟢 *Support:* None below current price\n"
                    else:
                        msg += f"❌ No price data available for {symbol}"
                else:
                    msg += f"❌ No HTF levels found for {symbol}"
            else:
                # Show example usage
                msg += "💡 *Usage:*\n"
                msg += "`/htf_sr BTCUSDT` - Show levels for specific symbol\n"
                msg += "`/update_htf_sr` - Force update all HTF levels"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in htf_sr_status: {e}")
            await update.message.reply_text("Error fetching HTF S/R status")
    
    async def update_htf_sr(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force update HTF support/resistance levels (disabled)."""
        await self.safe_reply(update, "HTF S/R module disabled.")
        return
        try:
            # Get frames data
            frames = self.shared.get("frames", {})
            if not frames:
                await update.message.reply_text("❌ No candle data available")
                return
            
            # Send initial message
            msg = await update.message.reply_text("🔄 Updating HTF S/R levels for all symbols...")
            
            # Update HTF levels
            from autobot.utils.sr_levels import initialize_all_sr_levels
            results = initialize_all_sr_levels(frames)
            
            # Update message with results
            symbols_with_levels = [sym for sym, count in results.items() if count > 0]
            
            result_msg = "✅ *HTF S/R Update Complete*\n"
            result_msg += "━" * 25 + "\n\n"
            result_msg += f"📊 *Results:*\n"
            result_msg += f"• Symbols analyzed: {len(results)}\n"
            result_msg += f"• Found levels: {len(symbols_with_levels)} symbols\n"
            result_msg += f"• Total levels: {sum(results.values())}\n\n"
            
            # Show top 5 symbols by level count
            if symbols_with_levels:
                top_symbols = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
                result_msg += "🏆 *Top Symbols by Level Count:*\n"
                for sym, count in top_symbols:
                    if count > 0:
                        result_msg += f"• {sym}: {count} levels\n"
            
            result_msg += "\nUse `/htf_sr [symbol]` to view specific levels"
            
            await msg.edit_text(result_msg)
            
        except Exception as e:
            logger.error(f"Error in update_htf_sr: {e}")
            await update.message.reply_text("Error updating HTF S/R levels")

    # Removed: mr_ml_stats method (MR strategy disabled)
        """Show detailed Mean Reversion ML statistics"""
        # MR disabled UX guard
        try:
            cfg = self.shared.get('config') or {}
            if bool(((cfg.get('modes', {}) or {}).get('disable_mr', True))):
                await self.safe_reply(update, "🌀 Mean Reversion is disabled in this build.")
                return
        except Exception:
            pass
        try:
            msg = "🔄 *Mean Reversion ML Status*\n"
            msg += "━" * 30 + "\n\n"

            # Get mean reversion scorer
            try:
                mean_reversion_scorer = get_mean_reversion_scorer()
            except Exception as e:
                msg += f"❌ *Error getting Mean Reversion ML:* {e}\n"
                await self.safe_reply(update, msg)
                return

            # Get comprehensive stats
            stats = mean_reversion_scorer.get_stats()

            # Status section
            if stats.get('is_ml_ready'):
                msg += "✅ *Status: ACTIVE & LEARNING*\n"
                msg += f"• Models trained: {len(stats.get('models_active', []))}/3\n"
                msg += f"• Model types: {', '.join(stats.get('models_active', []))}\n"
            else:
                msg += "📊 *Status: COLLECTING DATA*\n"
                remaining = max(0, stats.get('min_trades_for_ml', 50) - stats.get('completed_trades', 0))
                msg += f"• Trades needed: {remaining} more\n"

            msg += "\n📊 *Trade Statistics:*\n"
            msg += f"• Total trades: {stats.get('completed_trades', 0)}\n"
            msg += f"• Last training: {stats.get('last_train_count', 0)} trades\n"

            if 'recent_win_rate' in stats and stats['recent_trades'] > 0:
                msg += f"• Recent win rate: {stats['recent_win_rate']:.1f}% ({stats['recent_trades']} trades)\n"

            # Retrain info
            msg += "\n🔄 *Retrain Schedule:*\n"
            msg += f"• Retrain interval: {stats.get('retrain_interval', 25)} trades\n"
            next_retrain = stats.get('next_retrain_in', 0)
            if next_retrain > 0:
                msg += f"• Next retrain in: {next_retrain} trades\n"
            else:
                msg += "• Ready for retrain! \ud83c\udf86\n"

            # Scoring configuration
            msg += "\n⚙️ *ML Configuration:*\n"
            msg += f"• Score threshold: {stats.get('current_threshold', 70):.0f}\n"
            msg += f"• Min trades for ML: {stats.get('min_trades_for_ml', 50)}\n"

            # Feature info
            msg += "\n🧪 *Features Used:*\n"
            msg += "• Range width & strength\n"
            msg += "• RSI & Stochastic extremes\n"
            msg += "• Volume confirmation\n"
            msg += "• Reversal candle quality\n"
            msg += "• Session & time context\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in mr_ml_stats: {e}")
            await update.message.reply_text("Error getting Mean Reversion ML statistics")

    # Removed: mr_retrain method (MR strategy disabled)
        """Force retrain Mean Reversion ML models"""
        # MR disabled UX guard
        try:
            cfg = self.shared.get('config') or {}
            if bool(((cfg.get('modes', {}) or {}).get('disable_mr', True))):
                await self.safe_reply(update, "🌀 Mean Reversion is disabled in this build.")
                return
        except Exception:
            pass
        try:
            msg = "🔄 *Mean Reversion ML Retrain*\n"
            msg += "━" * 25 + "\n\n"

            # Get mean reversion scorer
            try:
                mean_reversion_scorer = get_mean_reversion_scorer()
            except Exception as e:
                msg += f"❌ *Error:* {e}\n"
                await self.safe_reply(update, msg)
                return

            # Check if retrain is possible
            retrain_info = mean_reversion_scorer.get_retrain_info()

            if not retrain_info['can_train']:
                msg += "⚠️ *Cannot Retrain Yet*\n"
                remaining = max(0, 50 - retrain_info['total_trades'])
                msg += f"Need {remaining} more trades before first training.\n"
                await self.safe_reply(update, msg)
                return

            # Show pre-retrain status
            msg += "📊 *Pre-Retrain Status:*\n"
            msg += f"• Available trades: {retrain_info['total_trades']}\n"
            msg += f"• Last training: {retrain_info['last_train_at']} trades\n"

            # Attempt retrain
            msg += "\n🔄 *Starting Retrain...*\n"
            temp_msg = await update.message.reply_text(msg)

            try:
                success = mean_reversion_scorer.startup_retrain()

                if success:
                    msg += "✅ *Retrain Successful!*\n"
                    msg += "\n�\udf86 *Post-Retrain Status:*\n"

                    # Get updated stats
                    updated_stats = mean_reversion_scorer.get_stats()
                    msg += f"• Models active: {len(updated_stats.get('models_active', []))}\n"
                    msg += f"• Trained on: {updated_stats.get('last_train_count', 0)} trades\n"
                    msg += f"• Status: {updated_stats.get('status', 'Unknown')}\n"

                else:
                    msg += "❌ *Retrain Failed*\n"
                    msg += "Check logs for details.\n"

            except Exception as retrain_error:
                msg += f"❌ *Retrain Error:* {retrain_error}\n"

            await temp_msg.edit_text(msg)

        except Exception as e:
            logger.error(f"Error in mr_retrain: {e}")
            await update.message.reply_text("Error during Mean Reversion ML retrain")

    async def enhanced_mr_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Enhanced Mean Reversion ML statistics"""
        # MR disabled UX guard
        try:
            cfg = self.shared.get('config') or {}
            if bool(((cfg.get('modes', {}) or {}).get('disable_mr', True))):
                await self.safe_reply(update, "🌀 Mean Reversion is disabled in this build.")
                return
        except Exception:
            pass
        try:
            msg = "🧠 *Enhanced Mean Reversion ML Status*\n"
            msg += "━" * 35 + "\n\n"

            # Check if enhanced system is available
            try:
                from enhanced_mr_scorer import get_enhanced_mr_scorer
                enhanced_mr_scorer = get_enhanced_mr_scorer()
            except ImportError:
                msg += "❌ *Enhanced MR ML not available*\n"
                msg += "Please check if enhanced_mr_scorer.py is installed.\n"
                await self.safe_reply(update, msg)
                return

            # Get enhanced stats
            stats = enhanced_mr_scorer.get_enhanced_stats()

            # Status section
            if stats.get('is_ml_ready'):
                msg += "✅ *Status: ADVANCED ML ACTIVE*\n"
                msg += f"• Strategy: {stats.get('strategy', 'Enhanced Mean Reversion')}\n"
                msg += f"• Models: {stats.get('model_count', 0)}/4 active\n"
                msg += f"• Features: {stats.get('feature_count', 30)}+ enhanced features\n"
            else:
                msg += f"📚 *Status: {stats.get('status', 'Learning')}*\n"
                msg += f"• Trades needed: {stats.get('min_trades_for_ml', 30)}\n"
                msg += f"• Progress: {stats.get('completed_trades', 0)}/{stats.get('min_trades_for_ml', 30)}\n"

            msg += "\n📊 *Performance Metrics:*\n"
            msg += f"• Completed trades: {stats.get('completed_trades', 0)}\n"
            msg += f"• Current threshold: {stats.get('current_threshold', 72):.0f}%\n"
            msg += f"• Threshold range: {stats.get('min_threshold', 65)}-{stats.get('max_threshold', 88)}%\n"

            if stats.get('recent_win_rate', 0) > 0:
                msg += f"• Recent win rate: {stats.get('recent_win_rate', 0):.1f}%\n"
                msg += f"• Sample size: {stats.get('recent_trades', 0)} trades\n"

            # Model details
            if stats.get('models_active'):
                msg += "\n🤖 *Active Models:*\n"
                for model in stats.get('models_active', []):
                    msg += f"• {model.replace('_', ' ').title()}\n"

            # Training info
            msg += "\n🔄 *Training Info:*\n"
            msg += f"• Last trained: {stats.get('last_train_count', 0)} trades\n"
            msg += f"• Retrain interval: {stats.get('retrain_interval', 50)} trades\n"
            msg += f"• Next retrain in: {stats.get('trades_until_retrain', 'N/A')} trades\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in enhanced_mr_stats: {e}")
            await update.message.reply_text("Error getting Enhanced MR ML stats")

    # Removed: mr_phantom_stats method (MR strategy disabled)
        """Show Mean Reversion phantom trade statistics"""
        # MR disabled UX guard
        try:
            cfg = self.shared.get('config') or {}
            if bool(((cfg.get('modes', {}) or {}).get('disable_mr', True))):
                await self.safe_reply(update, "🌀 Mean Reversion is disabled in this build.")
                return
        except Exception:
            pass
        try:
            msg = "👻 *Mean Reversion Phantom Trades*\n"
            msg += "━" * 30 + "\n\n"

            # Get MR phantom tracker
            try:
                from mr_phantom_tracker import get_mr_phantom_tracker
                mr_phantom_tracker = get_mr_phantom_tracker()
            except ImportError:
                msg += "❌ *MR Phantom Tracker not available*\n"
                await self.safe_reply(update, msg)
                return

            # Get phantom stats
            phantom_stats = mr_phantom_tracker.get_mr_phantom_stats()

            msg += f"📈 *Overall Statistics:*\n"
            total_trades = phantom_stats.get('total_mr_trades', 0)
            executed_trades = phantom_stats.get('executed', 0)
            rejected_trades = phantom_stats.get('rejected', 0)
            
            msg += f"• Total MR signals: {total_trades}\n"
            msg += f"• Executed: {executed_trades}\n"
            msg += f"• Rejected: {rejected_trades}\n"
            
            # Add verification that counts add up
            if total_trades != (executed_trades + rejected_trades):
                msg += f"⚠️ *Count mismatch detected: {total_trades} ≠ {executed_trades + rejected_trades}*\n"

            # Outcome analysis - show all rates, not just non-zero
            outcome = phantom_stats.get('outcome_analysis', {})
            if phantom_stats.get('executed', 0) > 0 or phantom_stats.get('rejected', 0) > 0:
                msg += f"\n📊 *Performance Analysis:*\n"
                
                # Show executed trades performance
                executed_count = phantom_stats.get('executed', 0)
                if executed_count > 0:
                    executed_wr = outcome.get('executed_win_rate', 0)
                    msg += f"• Executed trades: {executed_count} (Win rate: {executed_wr:.1f}%)\n"
                
                # Show rejected trades performance
                rejected_count = phantom_stats.get('rejected', 0)
                if rejected_count > 0:
                    rejected_wr = outcome.get('rejected_would_win_rate', 0)
                    msg += f"• Rejected trades: {rejected_count} (Would-be win rate: {rejected_wr:.1f}%)\n"

            # MR-specific metrics
            mr_metrics = phantom_stats.get('mr_specific_metrics', {})
            if mr_metrics:
                msg += f"\n📉 *Mean Reversion Specific:*\n"
                msg += f"• Range breakouts during trade: {mr_metrics.get('range_breakout_during_trade', 0)}\n"
                msg += f"• Timeout closures: {mr_metrics.get('timeout_closures', 0)}\n"
                msg += f"• High confidence ranges: {mr_metrics.get('high_confidence_ranges', 0)}\n"
                msg += f"• Boundary entries: {mr_metrics.get('boundary_entries', 0)}\n"

                if mr_metrics.get('boundary_entry_win_rate'):
                    msg += f"• Boundary entry win rate: {mr_metrics.get('boundary_entry_win_rate', 0):.1f}%\n"

            # Range performance breakdown
            range_perf = phantom_stats.get('range_performance', {})
            if range_perf:
                msg += f"\n🎯 *Range Quality Performance:*\n"
                for quality, data in range_perf.items():
                    if isinstance(data, dict) and data.get('wins') is not None:
                        total = data.get('wins', 0) + data.get('losses', 0)
                        if total > 0:
                            wr = (data.get('wins', 0) / total) * 100
                            msg += f"• {quality.replace('_', ' ').title()}: {wr:.1f}% ({total} trades)\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in mr_phantom_stats: {e}")
            await update.message.reply_text("Error getting MR phantom stats")

    async def parallel_performance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show parallel strategy system performance comparison"""
        if not self._cooldown_ok('parallel_performance'):
            await self.safe_reply(update, "⏳ Please wait before using /parallel_performance again")
            return
        try:
            msg = "⚡ *Parallel Strategy Performance*\n"
            msg += "━" * 35 + "\n\n"

            # Check if enhanced parallel system is available
            try:
                from enhanced_mr_scorer import get_enhanced_mr_scorer
                from ml_scorer_trend import get_trend_scorer
                enhanced_mr = get_enhanced_mr_scorer()
                trend_ml = get_trend_scorer()
            except ImportError:
                msg += "❌ *Enhanced parallel system not available*\n"
                await self.safe_reply(update, msg)
                return

            # Get stats from both systems
            trend_stats = trend_ml.get_stats()
            mr_stats = enhanced_mr.get_enhanced_stats()

            msg += "🎯 *Trend Strategy (Breakout):*\n"
            msg += f"• Status: {trend_stats.get('status', 'Unknown')}\n"
            t_exec = trend_stats.get('executed_count', trend_stats.get('completed_trades', 0))
            t_ph = trend_stats.get('phantom_count', 0)
            msg += f"• Trades: Executed {t_exec} | Phantom {t_ph}\n"
            msg += f"• Threshold: {trend_stats.get('current_threshold', 70):.0f}%\n"
            if trend_stats.get('recent_win_rate', 0) > 0:
                msg += f"• Recent WR: {trend_stats.get('recent_win_rate', 0):.1f}%\n"

            msg += "\n📉 *Mean Reversion Strategy (Ranging Markets):*\n"
            msg += f"• Status: {mr_stats.get('status', 'Unknown')}\n"
            msg += f"• Trades: {mr_stats.get('completed_trades', 0)}\n"
            msg += f"• Threshold: {mr_stats.get('current_threshold', 72):.0f}%\n"
            if mr_stats.get('recent_win_rate', 0) > 0:
                msg += f"• Recent WR: {mr_stats.get('recent_win_rate', 0):.1f}%\n"

            # Combined performance
            total_trades = trend_stats.get('completed_trades', 0) + mr_stats.get('completed_trades', 0)
            msg += f"\n📊 *Combined System:*\n"
            msg += f"• Total trades: {total_trades}\n"
            msg += f"• Strategy coverage: Full market conditions\n"
            msg += f"• Adaptive routing: Regime-based selection\n"

            # Active models summary
            trend_models = len(trend_stats.get('models_active', []))
            mr_models = mr_stats.get('model_count', 0)
            msg += f"• Active ML models: {trend_models + mr_models} total\n"
            msg += f"  - Trend: {trend_models}/2 models\n"
            msg += f"  - Mean Reversion: {mr_models}/4 models\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in parallel_performance: {e}")
            await update.message.reply_text("Error getting parallel performance stats")

    async def regime_analysis(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current market regime analysis for top symbols"""
        if not self._cooldown_ok('regime_analysis'):
            await self.safe_reply(update, "⏳ Please wait before using /regimeanalysis again")
            return
        try:
            msg = "🌐 *Market Regime Analysis*\n"
            msg += "━" * 30 + "\n\n"

            # Check if enhanced regime detection is available
            try:
                from autobot.utils.regime import get_enhanced_market_regime, get_regime_summary
            except ImportError:
                msg += "❌ *Enhanced regime detection not available*\n"
                await self.safe_reply(update, msg)
                return

            # Get current frames from shared data (if available)
            book = self.shared.get("book")
            if not book or not hasattr(book, 'positions'):
                msg += "❌ *No market data available*\n"
                await self.safe_reply(update, msg)
                return

            # Analyze regime for symbols with positions or top symbols
            frames = self.shared.get("frames", {})
            symbols_to_analyze = list(book.positions.keys()) if book.positions else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

            regime_summary = {}
            for symbol in symbols_to_analyze[:8]:  # Limit to 8 symbols
                if symbol in frames and not frames[symbol].empty:
                    try:
                        regime_analysis = get_enhanced_market_regime(frames[symbol], symbol)
                        regime_summary[symbol] = regime_analysis
                    except Exception as e:
                        logger.debug(f"Regime analysis failed for {symbol}: {e}")

            if regime_summary:
                msg += "📊 *Current Regime Analysis:*\n"
                for symbol, analysis in regime_summary.items():
                    msg += f"\n**{symbol}:**\n"
                    msg += f"• Regime: {analysis.primary_regime.title()}\n"
                    msg += f"• Confidence: {analysis.regime_confidence:.0%}\n"
                    msg += f"• Strategy: {analysis.recommended_strategy.replace('_', ' ').title()}\n"

                    if analysis.primary_regime == "ranging":
                        msg += f"• Range quality: {analysis.range_quality}\n"
                    elif analysis.primary_regime == "trending":
                        msg += f"• Trend strength: {analysis.trend_strength:.0f}%\n"

                    msg += f"• Volatility: {analysis.volatility_level}\n"

                # Overall summary
                regimes = [analysis.primary_regime for analysis in regime_summary.values()]
                trending_count = regimes.count('trending')
                ranging_count = regimes.count('ranging')
                volatile_count = regimes.count('volatile')

                msg += f"\n🔍 *Market Summary:*\n"
                msg += f"• Trending: {trending_count} symbols\n"
                msg += f"• Ranging: {ranging_count} symbols\n"
                msg += f"• Volatile: {volatile_count} symbols\n"

            else:
                msg += "❌ *No regime data available*\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in regime_analysis: {e}")
            await update.message.reply_text("Error getting regime analysis")

    async def strategy_comparison(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Compare strategy performance and show regime accuracy"""
        if not self._cooldown_ok('strategy_comparison'):
            await self.safe_reply(update, "⏳ Please wait before using /strategycomparison again")
            return
        try:
            msg = "⚖️ *Strategy Comparison Analysis*\n"
            msg += "━" * 35 + "\n\n"

            # Get trade tracker for historical performance
            trade_tracker = self.shared.get("trade_tracker")
            if not trade_tracker:
                msg += "❌ *No trade history available*\n"
                await self.safe_reply(update, msg)
                return

            try:
                # Get recent trades by strategy
                all_trades = trade_tracker.get_all_trades()

                # Filter recent trades (last 50)
                recent_trades = all_trades[-50:] if len(all_trades) > 50 else all_trades

                # Group by strategy
                trend_trades = [t for t in recent_trades if t.strategy_name in ("trend_pullback", "pullback")]
                mr_trades = [t for t in recent_trades if t.strategy_name in ["mean_reversion", "enhanced_mr"]]

                msg += f"📈 *Recent Performance (Last 50 trades):*\n"

                if trend_trades:
                    trend_wins = sum(1 for t in trend_trades if t.pnl_usd > 0)
                    trend_wr = (trend_wins / len(trend_trades)) * 100
                    trend_pnl = sum(t.pnl_usd for t in trend_trades)

                    msg += f"\n🎯 *Trend Strategy:*\n"
                    msg += f"• Trades: {len(trend_trades)}\n"
                    msg += f"• Win rate: {trend_wr:.1f}%\n"
                    msg += f"• Total P&L: ${trend_pnl:.2f}\n"
                    if len(trend_trades) > 0:
                        avg_pnl = trend_pnl / len(trend_trades)
                        msg += f"• Avg P&L: ${avg_pnl:.2f}\n"

                if mr_trades:
                    mr_wins = sum(1 for t in mr_trades if t.pnl_usd > 0)
                    mr_wr = (mr_wins / len(mr_trades)) * 100
                    mr_pnl = sum(t.pnl_usd for t in mr_trades)

                    msg += f"\n📉 *Mean Reversion Strategy:*\n"
                    msg += f"• Trades: {len(mr_trades)}\n"
                    msg += f"• Win rate: {mr_wr:.1f}%\n"
                    msg += f"• Total P&L: ${mr_pnl:.2f}\n"
                    if len(mr_trades) > 0:
                        avg_pnl = mr_pnl / len(mr_trades)
                        msg += f"• Avg P&L: ${avg_pnl:.2f}\n"

                # Combined stats
                if trend_trades or mr_trades:
                    total_trades = len(trend_trades) + len(mr_trades)
                    total_wins = (len([t for t in trend_trades if t.pnl_usd > 0]) +
                                 len([t for t in mr_trades if t.pnl_usd > 0]))
                    total_pnl = (sum(t.pnl_usd for t in trend_trades) +
                                sum(t.pnl_usd for t in mr_trades))

                    msg += f"\n📊 *Combined Performance:*\n"
                    msg += f"• Total trades: {total_trades}\n"
                    if total_trades > 0:
                        combined_wr = (total_wins / total_trades) * 100
                        msg += f"• Combined win rate: {combined_wr:.1f}%\n"
                        msg += f"• Combined P&L: ${total_pnl:.2f}\n"
                        msg += f"• Avg per trade: ${total_pnl/total_trades:.2f}\n"

                    # Strategy distribution
                    trend_pct = (len(trend_trades) / total_trades) * 100 if total_trades > 0 else 0
                    mr_pct = (len(mr_trades) / total_trades) * 100 if total_trades > 0 else 0

                    msg += f"\n📋 *Strategy Distribution:*\n"
                    msg += f"• Trend: {trend_pct:.1f}% of trades\n"
                    msg += f"• Mean Reversion: {mr_pct:.1f}% of trades\n"

                else:
                    msg += "❌ *No recent strategy trades found*\n"

            except Exception as e:
                logger.error(f"Error analyzing trade history: {e}")
                msg += f"❌ *Error analyzing trades: {e}*\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in strategy_comparison: {e}")
            await update.message.reply_text("Error comparing strategies")

    async def system_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show enhanced parallel system status and configuration"""
        try:
            msg = "🤖 *Enhanced Parallel System Status*\n"
            msg += "━" * 40 + "\n\n"

            # System Architecture
            msg += "🏗️ *System Architecture:*\n"
            msg += "• 🔄 Parallel Strategy Routing\n"
            msg += "• 🧠 Enhanced ML Scorers (Trend + MR)\n"
            msg += "• 👻 Independent Phantom Tracking\n"
            msg += "• 🎯 Regime-Based Strategy Selection\n\n"

            # Check system availability
            bot_instance = self.shared.get("bot_instance")

            msg += "⚡ *Component Status:*\n"

            # Enhanced MR System
            try:
                if bot_instance and hasattr(bot_instance, 'enhanced_mr_scorer') and bot_instance.enhanced_mr_scorer:
                    msg += "• ✅ Enhanced Mean Reversion ML\n"
                else:
                    msg += "• ⏳ Enhanced Mean Reversion ML (Loading)\n"
            except:
                msg += "• ❓ Enhanced Mean Reversion ML (Unknown)\n"

            # Trend System
            ml_scorer = self.shared.get("ml_scorer")
            if ml_scorer:
                msg += "• ✅ Trend ML System\n"
            else:
                msg += "• ⏳ Trend ML System (Loading)\n"

            # Trend Promotion status summary
            try:
                cfg = self.shared.get('config', {}) or {}
                tr_cfg = (cfg.get('trend', {}) or {}).get('promotion', {})
                tp = self.shared.get('trend_promotion', {}) or {}
                cap = int(tr_cfg.get('daily_exec_cap', 20))
                msg += f"• 🚀 Trend Promotion: {'Active' if tp.get('active') else 'Off'} ({tp.get('count',0)}/{cap})\n"
            except Exception:
                pass

            # Market Regime Detection
            try:
                from autobot.utils.regime import get_enhanced_market_regime
                msg += "• ✅ Enhanced Regime Detection\n"
            except:
                msg += "• ❌ Enhanced Regime Detection (Error)\n"

            # Phantom Trackers
            phantom_tracker = self.shared.get("phantom_tracker")
            if phantom_tracker:
                msg += "• ✅ Trend Phantom Tracker\n"
            else:
                msg += "• ⏳ Trend Phantom Tracker\n"

            try:
                if bot_instance and hasattr(bot_instance, 'mr_phantom_tracker') and bot_instance.mr_phantom_tracker:
                    msg += "• ✅ MR Phantom Tracker\n"
                else:
                    msg += "• ⏳ MR Phantom Tracker\n"
            except:
                msg += "• ❓ MR Phantom Tracker\n"

            msg += "\n🎯 *Strategy Selection Logic:*\n"
            msg += "• 📊 Trending Markets → Trend Strategy\n"
            msg += "• 📦 High-Quality Ranges → Enhanced MR\n"
            msg += "• 🌪️ Volatile Markets → Skip Trading\n"
            msg += "• ⚖️ Independent ML Scoring Per Strategy\n\n"

            msg += "📈 *Performance Features:*\n"
            msg += "• 🎯 Consistent 2.5:1 Risk:Reward\n"
            msg += "• 💰 Fee-Adjusted Take Profits\n"
            msg += "• 🛡️ Hybrid Stop Loss Calculation\n"
            msg += "• 🔄 Volatility-Adaptive Buffers\n"
            msg += "• 📊 Real-Time Regime Analysis\n\n"

            msg += "⚙️ *Quick Access Commands:*\n"
            msg += "`/enhanced_mr` - MR ML status\n"
            msg += "`/mr_phantom` - MR phantom trades\n"
            msg += "`/parallel_performance` - Strategy comparison\n"
            msg += "`/regime_analysis` - Current market regimes\n"
            msg += "`/strategy_comparison` - Detailed performance\n\n"

            msg += "_Enhanced system provides complete market coverage with specialized strategies_"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in system_status: {e}")
            await update.message.reply_text("Error getting system status")

    async def phantom_qa(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show phantom sampling QA: caps, usage, dedup hits, basic WR by routing."""
        try:
            from datetime import datetime
            day = datetime.utcnow().strftime('%Y%m%d')
            # Load config caps
            cfg = self.shared.get('config') or {}
            ph = cfg.get('phantom', {})
            none_cap = ph.get('none_cap', 50)
            cl3_cap = ph.get('cluster3_cap', 20)
            off_cap = ph.get('offhours_cap', 15)

            # Read counters from Redis if available
            none_used = cl3_used = off_used = dedup_hits = 0
            blocked_total = blocked_trend = blocked_mr = blocked_scalp = 0
            try:
                import os, redis
                r = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
                none_used = int(r.get(f'phantom:daily:none_count:{day}') or 0)
                cl3_used = int(r.get(f'phantom:daily:cluster3_count:{day}') or 0)
                off_used = int(r.get(f'phantom:daily:offhours_count:{day}') or 0)
                dedup_hits = int(r.get(f'phantom:dedup_hits:{day}') or 0)
                blocked_total = int(r.get(f'phantom:blocked:{day}') or 0)
                blocked_trend = int(r.get(f'phantom:blocked:{day}:trend') or 0)
                blocked_mr = int(r.get(f'phantom:blocked:{day}:mr') or 0)
                blocked_scalp = int(r.get(f'phantom:blocked:{day}:scalp') or 0)
                # Flow controller stats (accepted and relax per strategy)
                tr_acc = int(r.get(f'phantom:flow:{day}:trend:accepted') or 0)
                mr_acc = int(r.get(f'phantom:flow:{day}:mr:accepted') or 0)
                sc_acc = int(r.get(f'phantom:flow:{day}:scalp:accepted') or 0)
                tr_relax = float(r.get(f'phantom:flow:{day}:trend:relax') or 0.0)
                mr_relax = float(r.get(f'phantom:flow:{day}:mr:relax') or 0.0)
                sc_relax = float(r.get(f'phantom:flow:{day}:scalp:relax') or 0.0)
            except Exception:
                pass

            # Fallbacks when Redis is unavailable or counters are zero
            # Derive usage and blocked from in-memory trackers
            try:
                get_phantom_tracker = None
                get_mr_phantom_tracker = None
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                from autobot.utils.symbols import load_symbol_clusters
                clusters_map = load_symbol_clusters()
            except Exception:
                get_phantom_tracker = get_mr_phantom_tracker = get_scalp_phantom_tracker = None  # type: ignore
                clusters_map = {}

            # Helper: check if timestamp is today UTC off-hours
            def _is_off_hours(ts: datetime) -> bool:
                h = ts.hour
                return (h >= 22 or h < 2)

            # Build a list of all phantom signals recorded today (PB, MR, Scalp)
            derived_none_used = derived_cl3_used = derived_off_used = 0
            derived_tr_acc = derived_mr_acc = derived_sc_acc = 0
            derived_blk_total = derived_blk_tr = derived_blk_mr = derived_blk_sc = 0
            # Trend
            try:
                pt = get_phantom_tracker() if get_phantom_tracker else None
                if pt:
                    d_blk_total = 0; d_blk_tr = 0
                    # Completed
                    for trades in pt.phantom_trades.values():
                        for p in trades:
                            if hasattr(p, 'signal_time') and p.signal_time.strftime('%Y%m%d') == day and not getattr(p, 'was_executed', False):
                                derived_none_used += 1
                                derived_tr_acc += 1
                                if clusters_map.get(p.symbol, 0) == 3:
                                    derived_cl3_used += 1
                                if _is_off_hours(p.signal_time):
                                    derived_off_used += 1
                    # Active
                    for p in pt.active_phantoms.values():
                        if p.signal_time.strftime('%Y%m%d') == day:
                            derived_none_used += 1
                            derived_tr_acc += 1
                            if clusters_map.get(p.symbol, 0) == 3:
                                derived_cl3_used += 1
                            if _is_off_hours(p.signal_time):
                                derived_off_used += 1
                    # Blocked (local fallback)
                    bl = pt.get_blocked_counts(day)
                    derived_blk_total += bl.get('total', 0)
                    derived_blk_tr += bl.get('trend', 0)
            except Exception:
                pass
            # MR
            try:
                mrpt = get_mr_phantom_tracker() if get_mr_phantom_tracker else None
                if mrpt:
                    d_blk_total = 0; d_blk_mr = 0
                    for trades in mrpt.mr_phantom_trades.values():
                        for p in trades:
                            if hasattr(p, 'signal_time') and p.signal_time.strftime('%Y%m%d') == day and not getattr(p, 'was_executed', False):
                                derived_none_used += 1
                                derived_mr_acc += 1
                                if clusters_map.get(p.symbol, 0) == 3:
                                    derived_cl3_used += 1
                                if _is_off_hours(p.signal_time):
                                    derived_off_used += 1
                    for p in mrpt.active_mr_phantoms.values():
                        if p.signal_time.strftime('%Y%m%d') == day:
                            derived_none_used += 1
                            derived_mr_acc += 1
                            if clusters_map.get(p.symbol, 0) == 3:
                                derived_cl3_used += 1
                            if _is_off_hours(p.signal_time):
                                derived_off_used += 1
                    bl = mrpt.get_blocked_counts(day)
                    derived_blk_total += bl.get('total', 0)
                    derived_blk_mr += bl.get('mr', 0)
            except Exception:
                pass
            # Scalp
            try:
                scpt = get_scalp_phantom_tracker() if get_scalp_phantom_tracker else None
                if scpt:
                    d_blk_total = 0; d_blk_sc = 0
                    for trades in scpt.completed.values():
                        for p in trades:
                            if p.signal_time.strftime('%Y%m%d') == day and not getattr(p, 'was_executed', False):
                                derived_none_used += 1
                                derived_sc_acc += 1
                                if clusters_map.get(p.symbol, 0) == 3:
                                    derived_cl3_used += 1
                                if _is_off_hours(p.signal_time):
                                    derived_off_used += 1
                    for p in scpt.active.values():
                        if p.signal_time.strftime('%Y%m%d') == day:
                            derived_none_used += 1
                            derived_sc_acc += 1
                            if clusters_map.get(p.symbol, 0) == 3:
                                derived_cl3_used += 1
                            if _is_off_hours(p.signal_time):
                                derived_off_used += 1
                    bl = scpt.get_blocked_counts(day)
                    derived_blk_total += bl.get('total', 0)
                    derived_blk_sc += bl.get('scalp', 0)
            except Exception:
                pass

            # Prefer Redis values if present; otherwise use derived
            if none_used == 0 and (derived_none_used > 0):
                none_used = derived_none_used
            if cl3_used == 0 and (derived_cl3_used > 0):
                cl3_used = derived_cl3_used
            if off_used == 0 and (derived_off_used > 0):
                off_used = derived_off_used
            if blocked_total == 0 and derived_blk_total > 0:
                blocked_total = derived_blk_total
            if blocked_trend == 0 and derived_blk_tr > 0:
                blocked_trend = derived_blk_tr
            if blocked_mr == 0 and derived_blk_mr > 0:
                blocked_mr = derived_blk_mr
            if blocked_scalp == 0 and derived_blk_sc > 0:
                blocked_scalp = derived_blk_sc

            # Flow controller: accepted & relax fallback
            targets = (cfg.get('phantom_flow', {}) or {}).get('daily_target', {'trend':40,'mr':40,'scalp':40})
            def _relax_from(accepted:int, target:int) -> float:
                try:
                    h = datetime.utcnow().hour
                    pace_target = float(target) * min(1.0, max(1, h) / 24.0)
                    deficit = max(0.0, pace_target - float(accepted))
                    base_r = min(1.0, deficit / max(1.0, target * 0.5))
                    return base_r
                except Exception:
                    return 0.0
            if 'tr_acc' not in locals() or locals().get('tr_acc', 0) == 0:
                tr_acc = derived_tr_acc
            if 'mr_acc' not in locals() or mr_acc == 0:
                mr_acc = derived_mr_acc
            if 'sc_acc' not in locals() or sc_acc == 0:
                sc_acc = derived_sc_acc
            if 'tr_relax' not in locals() or locals().get('tr_relax', 0.0) == 0.0:
                tr_relax = _relax_from(tr_acc, int(targets.get('trend',40)))
            if 'mr_relax' not in locals() or mr_relax == 0.0:
                mr_relax = _relax_from(mr_acc, int(targets.get('mr',40)))
            if 'sc_relax' not in locals() or sc_relax == 0.0:
                sc_relax = _relax_from(sc_acc, int(targets.get('scalp',40)))

            # WR by routing across PB + MR + Scalp trackers
            wr_none = wr_allowed = 0.0
            try:
                total_none = wins_none = 0
                total_allowed = wins_allowed = 0
                # Trend
                if get_phantom_tracker:
                    pt = get_phantom_tracker()
                    for r in pt.get_learning_data():
                        routed_none = isinstance(r.get('features'), dict) and r['features'].get('routing') == 'none'
                        if routed_none:
                            total_none += 1; wins_none += int(r.get('outcome', 0))
                        else:
                            total_allowed += 1; wins_allowed += int(r.get('outcome', 0))
                # MR
                if get_mr_phantom_tracker:
                    mrpt = get_mr_phantom_tracker()
                    for r in mrpt.get_learning_data():
                        routed_none = False
                        try:
                            f = r.get('features') or {}
                            ef = r.get('enhanced_features') or {}
                            routed_none = (isinstance(f, dict) and f.get('routing') == 'none') or (isinstance(ef, dict) and ef.get('routing') == 'none')
                        except Exception:
                            pass
                        if routed_none:
                            total_none += 1; wins_none += int(r.get('outcome', 0))
                        else:
                            total_allowed += 1; wins_allowed += int(r.get('outcome', 0))
                # Scalp
                if get_scalp_phantom_tracker:
                    scpt = get_scalp_phantom_tracker()
                    for r in scpt.get_learning_data():
                        f = r.get('features') or {}
                        if isinstance(f, dict) and f.get('routing') == 'none':
                            total_none += 1; wins_none += int(r.get('outcome', 0))
                        else:
                            total_allowed += 1; wins_allowed += int(r.get('outcome', 0))
                if total_none:
                    wr_none = wins_none / total_none * 100.0
                if total_allowed:
                    wr_allowed = wins_allowed / total_allowed * 100.0
            except Exception:
                pass

            lines = [
                "🧪 *Phantom QA*",
                f"• Caps (none/cluster3/off): {none_cap}/{cl3_cap}/{off_cap}",
                f"• Used today: {none_used}/{cl3_used}/{off_used}",
                f"• Dedup hits today: {dedup_hits}",
                f"• WR routing=none: {wr_none:.1f}%",
                f"• WR routing=allowed: {wr_allowed:.1f}%",
                f"• Blocked today: {blocked_total} (Trend {blocked_trend}, MR {blocked_mr}, Scalp {blocked_scalp})",
            ]
            # Append flow controller section if available
            try:
                pf = cfg.get('phantom_flow', {})
                if pf.get('enabled', False):
                    targets = pf.get('daily_target', {'trend':40,'mr':40,'scalp':40})
                    lines.extend([
                        "\n🎛️ *Flow Controller* (phantom-only)",
                        f"• Trend: {locals().get('tr_acc',0)}/{targets.get('trend',0)} (relax {locals().get('tr_relax',0.0)*100:.0f}%)",
                        f"• Mean Reversion: {locals().get('mr_acc',0)}/{targets.get('mr',0)} (relax {locals().get('mr_relax',0.0)*100:.0f}%)",
                        f"• Scalp: {locals().get('sc_acc',0)}/{targets.get('scalp',0)} (relax {locals().get('sc_relax',0.0)*100:.0f}%)",
                    ])
            except Exception:
                pass
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in phantom_qa: {e}")
            await update.message.reply_text("Error getting phantom QA")

    async def scalp_qa(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Quick summary of scalp phantom stats using the dedicated scalp tracker."""
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            st = scpt.get_scalp_phantom_stats()
            total = st.get('total', 0)
            wins = st.get('wins', 0)
            losses = st.get('losses', 0)
            wr = st.get('wr', 0.0)
            # Scalp ML stats
            ml_ready = False; thr = 75; samples = 0; nxt = None; trainable = None; total_rec = None
            try:
                from autobot.strategies.scalp.scorer import get_scalp_scorer
                s = get_scalp_scorer()
                ml_ready = bool(getattr(s, 'is_ml_ready', False))
                thr = getattr(s, 'min_score', 75)
                samples = getattr(s, 'completed_trades', 0)
                try:
                    ri = s.get_retrain_info()
                    nxt = ri.get('trades_until_next_retrain')
                    trainable = ri.get('trainable_size')
                    total_rec = ri.get('total_records')
                except Exception:
                    nxt = None
            except Exception:
                pass
            # Scalp shadow stats
            sh_total = sh_wr = 0.0; sh_w = sh_l = 0
            try:
                from shadow_trade_simulator import get_shadow_tracker
                sstats = get_shadow_tracker().get_stats().get('scalp', {})
                sh_total = sstats.get('total', 0)
                sh_w = sstats.get('wins', 0)
                sh_l = sstats.get('losses', 0)
                sh_wr = sstats.get('wr', 0.0)
            except Exception:
                pass
            # Gate pass rate
            gate_pass_count = 0
            gate_pass_pct = 0.0
            try:
                phantoms = [p for arr in scpt.completed.values() for p in arr if p.outcome in ('win', 'loss') and not getattr(p, 'was_executed', False)]
                if phantoms:
                    gate_pass_count = sum(1 for p in phantoms if scpt.compute_gate_status(p)['all'])
                    gate_pass_pct = (gate_pass_count / len(phantoms) * 100) if len(phantoms) > 0 else 0.0
            except Exception:
                pass
            msg = [
                "🩳 *Scalp QA*",
                f"• ML: {'✅ Ready' if ml_ready else '⏳ Training'} | Records: {total_rec if total_rec is not None else samples} | Trainable: {trainable if trainable is not None else '-'} | Thr: {thr}",
                f"• Next retrain in: {nxt if nxt is not None else '-'} trades",
                f"• Phantom recorded: {total} | WR: {wr:.1f}% (W/L {wins}/{losses})",
                f"• All gates pass: {gate_pass_count}/{total} ({gate_pass_pct:.1f}%)",
                f"• Shadow (ML-based): {int(sh_total)} | WR: {sh_wr:.1f}% (W/L {sh_w}/{sh_l})",
                "_Scalp runs phantom-only; shadow sim reflects ML decision quality_"
            ]
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"Error in scalp_qa: {e}")
            await update.message.reply_text("Error getting scalp QA")

    async def scalp_qscore_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """As-if execute report for Scalp based on phantom Qscore thresholds.

        Computes WR and EV (mean realized R) for phantoms with qscore ≥ thresholds over recent windows.
        """
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker, ScalpPhantomTrade
            scpt = get_scalp_phantom_tracker()
            # Gather completed phantom trades with qscore and outcome (exclude timeouts)
            trades: list = []
            for arr in (scpt.completed or {}).values():
                for t in arr:
                    try:
                        if getattr(t, 'outcome', None) in ('win','loss') and not getattr(t, 'was_executed', False):
                            feats = getattr(t, 'features', {}) or {}
                            q = feats.get('qscore', None)
                            if isinstance(q, (int,float)):
                                trades.append(t)
                    except Exception:
                        continue
            if not trades:
                await self.safe_reply(update, "🩳 *Scalp Qscore*\nNo scored phantoms yet.")
                return
            # Windows
            import datetime as _dt
            now = _dt.datetime.utcnow()
            def within_days(t: ScalpPhantomTrade, days: int) -> bool:
                try:
                    et = getattr(t, 'exit_time', None)
                    if not et:
                        return False
                    return (now - et).days <= days
                except Exception:
                    return False
            # Thresholds to probe: around configured execute_q_min
            cfg = self.shared.get('config') or {}
            sc_cfg = (cfg.get('scalp', {}) or {})
            rm = (sc_cfg.get('rule_mode', {}) or {})
            base_thr = int(float(rm.get('execute_q_min', 60)))
            thrs = sorted(set([base_thr - 5, base_thr - 3, base_thr, base_thr + 2, base_thr + 5]))
            def agg(sample: list) -> dict:
                tot = len(sample)
                wins = sum(1 for t in sample if getattr(t, 'outcome', None) == 'win')
                losses = tot - wins
                wr = (wins / tot * 100.0) if tot else 0.0
                # EV in R using realized_rr (timeouts excluded already)
                try:
                    ev = sum(float(getattr(t, 'realized_rr', 0.0) or 0.0) for t in sample) / tot if tot else 0.0
                except Exception:
                    ev = 0.0
                return {'n': tot, 'wr': wr, 'evr': ev, 'w': wins, 'l': losses}
            # Build report for 7d and 30d windows
            def lines_for(days: int) -> list[str]:
                sub = [t for t in trades if within_days(t, days)]
                out = [f"• Window: {days}d ({len(sub)})"]
                if not sub:
                    out.append("  No data")
                    return out
                for thr in thrs:
                    sample = []
                    for t in sub:
                        try:
                            qv = float((getattr(t, 'features', {}) or {}).get('qscore', 0.0) or 0.0)
                            if qv >= thr:
                                sample.append(t)
                        except Exception:
                            continue
                    s = agg(sample)
                    out.append(f"  ≥{thr}: N={s['n']} WR={s['wr']:.1f}% EV={s['evr']:.2f}R (W/L {s['w']}/{s['l']})")
                return out
            msg = [
                "📈 *Scalp Qscore (as-if execute)*",
                f"Base thr: {base_thr}",
            ]
            msg += lines_for(7)
            msg.append("")
            msg += lines_for(30)
            msg.append("")
            msg.append("_EV uses realized R from phantoms; excludes timeouts_")
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"Error in scalp_qscore_report: {e}")
            await update.message.reply_text("Error computing Scalp Qscore report")

    async def scalp_gate_analysis(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Analyze Scalp phantom performance by hard gate pass/fail combinations."""
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()

            # Parse days argument
            days = 30
            if ctx.args and len(ctx.args) > 0:
                try:
                    days = int(ctx.args[0])
                    days = max(1, min(365, days))  # Clamp to 1-365 days
                except Exception:
                    await self.safe_reply(update, "Usage: /scalp_gate_analysis [days]\nExample: /scalp_gate_analysis 30")
                    return

            # Get gate analysis
            analysis = scpt.get_gate_analysis(days=days, min_samples=20)

            if analysis.get('error'):
                await self.safe_reply(update, f"🚪 *Scalp Gate Analysis*\n\n{analysis['error']}")
                return

            # Build message
            total = analysis['total_phantoms']
            baseline_wr = analysis['baseline_wr']
            all_gates = analysis['all_gates_pass']
            sorted_vars = analysis.get('sorted_variables', [])
            combos = analysis['top_combinations']

            msg = [
                f"🚪 *Scalp Variable Analysis* ({days}d)\n",
                f"📊 Baseline: {total} phantoms, {baseline_wr:.1f}% WR\n",
            ]

            # Show top 10 positive-impact variables
            if sorted_vars:
                positive_vars = [(k, v) for k, v in sorted_vars if v['delta'] and v['delta'] > 0]
                if positive_vars:
                    msg.append("*🟢 Top Filters (Improve WR):*")
                    for var_name, stats in positive_vars[:10]:
                        msg.append(
                            f"✅ {var_name}: {stats['pass_total']} trades, "
                            f"{stats['pass_wr']:.1f}% WR ({stats['delta']:+.1f}%)"
                        )

                # Show top 10 negative-impact variables
                negative_vars = [(k, v) for k, v in sorted_vars if v['delta'] and v['delta'] < 0]
                if negative_vars:
                    msg.append("\n*🔴 Worst Filters (Hurt WR):*")
                    for var_name, stats in reversed(negative_vars[-10:]):
                        msg.append(
                            f"❌ {var_name}: {stats['pass_total']} trades, "
                            f"{stats['pass_wr']:.1f}% WR ({stats['delta']:+.1f}%)"
                        )

                # Show neutral/insufficient data
                insufficient = len([v for k, v in analysis['variable_stats'].items() if not v['sufficient_samples']])
                if insufficient > 0:
                    msg.append(f"\n⚠️ {insufficient} variables with insufficient samples (<20)")

            # Top combinations
            if combos:
                msg.append("\n*Top Gate Combinations:*")
                msg.append("(H=HTF, V=Vol, B=Body, A=Align)")
                for combo in combos[:5]:
                    bitmap = combo['bitmap']
                    visual = ''.join(['🟢' if c == '1' else '🔴' for c in bitmap])
                    msg.append(
                        f"{visual} {combo['wins']}/{combo['total']} ({combo['wr']:.1f}% WR)"
                    )

            msg.append(f"\n_Min samples: {analysis['min_samples']} per gate status_")

            await self.safe_reply(update, "\n".join(msg))

        except Exception as e:
            logger.error(f"Error in scalp_gate_analysis: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def scalp_comprehensive_analysis(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Comprehensive Scalp analysis: all variables, combinations, recommendations."""
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()

            # Parse month argument
            month = None
            if ctx.args and len(ctx.args) > 0:
                month = ctx.args[0]
                # Validate format YYYY-MM
                try:
                    year, mon = month.split('-')
                    int(year), int(mon)
                except Exception:
                    await self.safe_reply(update, "Usage: /scalpcomprehensive [YYYY-MM]\\nExample: /scalpcomprehensive 2025-10")
                    return

            # Get comprehensive analysis
            analysis = scpt.get_comprehensive_analysis(month=month, top_n=10, min_samples=20)

            if analysis.get('error'):
                await self.safe_reply(update, f"🔍 <b>Scalp Comprehensive Analysis</b>\n\n{analysis['error']}", parse_mode='HTML')
                return

            # Build message
            total = analysis['total_phantoms']
            baseline_wr = analysis['baseline_wr']
            period = analysis['period']
            ranked_vars = analysis.get('ranked_variables', [])
            pair_analysis = analysis.get('pair_analysis', {})
            triplet_analysis = analysis.get('triplet_analysis', {})

            msg = [
                f"🔍 <b>Scalp Comprehensive Analysis</b> ({period})\n",
                f"📊 Dataset: {total} phantoms, {baseline_wr:.1f}% baseline WR\n",
            ]

            # Top 10 solo variables
            if ranked_vars:
                msg.append("━━━ <b>SOLO VARIABLES (Top 10)</b> ━━━")
                for var_name, stats in ranked_vars[:10]:
                    emoji = "✅" if stats['delta'] > 0 else "⚠️"
                    msg.append(
                        f"{emoji} {var_name}: {stats['pass_wr']:.1f}% WR "
                        f"({stats['delta']:+.1f}%) [{stats['pass_total']} trades]"
                    )

            # Best pairs
            if pair_analysis:
                msg.append("\n━━━ <b>BEST PAIRS</b> ━━━")
                for (v1, v2), stats in list(pair_analysis.items())[:5]:
                    synergy_str = f" synergy{stats.get('synergy', 0):+.1f}%" if stats.get('synergy') else ""
                    msg.append(
                        f"🎯 {v1} + {v2}: {stats['wr']:.1f}% WR "
                        f"[{stats['total']} trades] {stats['delta']:+.1f}%{synergy_str}"
                    )

            # Best triplets
            if triplet_analysis:
                msg.append("\n━━━ <b>BEST TRIPLETS</b> ━━━")
                for (v1, v2, v3), stats in list(triplet_analysis.items())[:3]:
                    msg.append(
                        f"🚀 {v1} + {v2} + {v3}: {stats['wr']:.1f}% WR "
                        f"[{stats['total']} trades]"
                    )

            msg.append("\n💡 Use /scalprecommend for config snippet")

            await self.safe_reply(update, "\n".join(msg), parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error in scalp_comprehensive_analysis: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def scalp_gate_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Adjust per-gate risk percentages for Scalp gate-based execution.

        Usage:
          /scalpgaterisk              → show current values
          /scalpgaterisk wick 2.5     → set Wick pass risk to 2.5%
          /scalpgaterisk htf 10       → set HTF pass risk to 10%
          /scalpgaterisk both 15      → set Both pass risk to 15%
        Aliases: /scalprisk
        """
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            shared = self.shared
            if 'scalp_gate_risk' not in shared or not isinstance(shared['scalp_gate_risk'], dict):
                shared['scalp_gate_risk'] = {'wick': 2.0, 'htf': 10.0, 'both': 15.0, 'vol': 0.5}
            rmap = shared['scalp_gate_risk']
            if not args or len(args) == 0:
                await self.safe_reply(update,
                    "🩳 *Scalp Gate Risk*\n"+
                    f"• Wick+Vol+Slope (active): {float(rmap.get('wick',2.0)):.2f}%\n"+
                    f"• Vol+Slope (disabled): {float(rmap.get('vol',0.5)):.2f}%\n"+
                    f"• HTF+Slope (disabled): {float(rmap.get('htf',10.0)):.2f}%\n"+
                    f"• ALL aligned (disabled): {float(rmap.get('both',15.0)):.2f}%\n\n"+
                    "• /scalpgaterisk [wick] <percent> — Set risk% for gate-based executes"
                )
                return
            if len(args) != 2:
                await self.safe_reply(update, "Usage: /scalpgaterisk [wick] <percent>")
                return
            gate = args[0].strip().lower()
            if gate not in ('wick',):
                await self.safe_reply(update, "Gate must be: wick")
                return
            try:
                pct = float(args[1])
            except Exception:
                await self.safe_reply(update, "Percent must be a number (e.g., 2.5)")
                return
            # Clamp reasonable bounds
            if pct < 0.01 or pct > 25.0:
                await self.safe_reply(update, "Percent out of bounds (0.01–25)")
                return
            rmap[gate] = float(pct)
            self.shared['scalp_gate_risk'] = rmap
            await self.safe_reply(update,
                "✅ *Scalp Gate Risk Updated*\n"+
                f"• Wick+Vol+Slope: {float(rmap.get('wick',2.0)):.2f}%\n"+
                f"• Vol+Slope: {float(rmap.get('vol',0.5)):.2f}%\n"+
                f"• HTF+Slope: {float(rmap.get('htf',10.0)):.2f}%\n"+
                f"• ALL aligned: {float(rmap.get('both',15.0)):.2f}%"
            )
        except Exception as e:
            logger.error(f"Error in scalp_gate_risk: {e}")
            await update.message.reply_text("Error updating scalp gate risk")

    async def stream_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Report live stream health and last kline timestamps.

        Shows:
        - Main stream idle seconds (time since last kline observed)
        - Per-symbol last 3m kline age and timestamp (oldest first, top 15)
        """
        try:
            from datetime import datetime, timezone
            import time as _t
            bot = (self.shared or {}).get('bot_instance') if hasattr(self, 'shared') else None
            symbols = list((self.shared or {}).get('symbols_config') or [])
            if bot is None:
                await self.safe_reply(update, "❌ Bot instance not available")
                return
            # Main idle seconds via watchdog timestamp (monotonic based)
            try:
                last_mono = float(getattr(bot, '_last_main_kline', 0.0) or 0.0)
                idle_s = int(_t.monotonic() - last_mono) if last_mono else None
            except Exception:
                idle_s = None
            # Collect per-symbol ages from 3m frames
            rows = []
            now = datetime.now(timezone.utc)
            try:
                frames_3m = getattr(bot, 'frames_3m', {}) or {}
            except Exception:
                frames_3m = {}
            for s in symbols:
                try:
                    df = frames_3m.get(s)
                    if df is None or len(df) == 0:
                        rows.append((s, None, None))
                        continue
                    ts = df.index[-1]
                    age = (now - ts).total_seconds() if ts.tzinfo else (now - ts.tz_localize('UTC')).total_seconds()
                    rows.append((s, int(age), ts))
                except Exception:
                    rows.append((s, None, None))
            # Order by age desc (oldest first)
            rows.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
            # Build message
            lines = ["🔌 Stream Status", ""]
            if idle_s is not None:
                lines.append(f"Main idle: {idle_s}s")
            else:
                lines.append("Main idle: n/a")
            # Show top 15 oldest symbols for brevity
            lines.append("")
            lines.append("Oldest symbols (3m):")
            show = rows[:15]
            for sym, age, ts in show:
                if age is None or ts is None:
                    lines.append(f"• {sym}: n/a")
                else:
                    try:
                        ts_str = ts.strftime('%H:%M:%SZ') if ts.tzinfo else ts.tz_localize('UTC').strftime('%H:%M:%SZ')
                    except Exception:
                        ts_str = str(ts)
                    lines.append(f"• {sym}: {age}s ago (ts {ts_str})")
            # Tail summary
            known = [r for r in rows if r[1] is not None]
            if known:
                max_age = max(r[1] for r in known)
                min_age = min(r[1] for r in known)
                lines.append("")
                lines.append(f"Symbols: {len(rows)} | min age: {min_age}s | max age: {max_age}s")
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in stream_status: {e}")
            try:
                await update.message.reply_text(f"❌ Error: {e}")
            except Exception:
                pass

    async def scalp_reset(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force reset Scalp state counters and summary snapshot.

        Clears per-symbol MOM/PULL/VWAP/Q flags, reasons histogram, and the Redis summary key.
        Does not touch any open phantoms or positions.
        """
        try:
            bot = (self.shared or {}).get('bot_instance') if hasattr(self, 'shared') else None
            cleared = False
            if bot is not None:
                try:
                    # Clear in-memory flags
                    if hasattr(bot, '_scalp_symbol_state'):
                        bot._scalp_symbol_state = {}
                    if hasattr(bot, '_scalp_reasons'):
                        bot._scalp_reasons = {}
                    if hasattr(bot, '_scalp_state_flush_ts'):
                        bot._scalp_state_flush_ts = 0.0
                    cleared = True
                except Exception:
                    pass
                # Clear Redis snapshot
                try:
                    if hasattr(bot, '_redis') and bot._redis is not None:
                        bot._redis.delete('state:scalp:summary')
                except Exception:
                    pass
                # Update shared summary to zeroed counters
                try:
                    from datetime import datetime as _dt
                    snapshot = {'ts': _dt.utcnow().isoformat()+'Z', 'mom': 0, 'pull': 0, 'vwap': 0, 'q_ge_thr': 0, 'exec_today': 0, 'phantom_open': 0, 'reasons': {}}
                    self.shared['scalp_states'] = snapshot
                except Exception:
                    pass
            await self.safe_reply(update, "✅ Scalp states reset. Counters will repopulate with fresh data.")
        except Exception as e:
            logger.error(f"Error in scalp_reset: {e}")
            await update.message.reply_text("❌ Error resetting scalp states")

    async def scalp_patterns(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Scalp ML patterns: feature importances, time patterns, and condition buckets."""
        try:
            from autobot.strategies.scalp.scorer import get_scalp_scorer
            s = get_scalp_scorer()
            pat = s.get_patterns() if s else {}
            lines = ["🤖 *Scalp ML Patterns*", ""]
            # Feature importance
            fi = pat.get('feature_importance', {}) or {}
            if fi:
                # Normalize feature names to compact style to match user preference
                name_map = {
                    'ema_slope_fast': 'emaslopefast',
                    'ema_slope_slow': 'emaslopeslow',
                    'atr_pct': 'atrpct',
                    'bb_width_pct': 'bbwidthpct',
                    'impulse_ratio': 'impulseratio',
                    'volume_ratio': 'volumeratio',
                    'upper_wick_ratio': 'upperwickratio',
                    'lower_wick_ratio': 'lowerwickratio',
                    'vwap_dist_atr': 'vwapdistatr',
                    'session': 'session',
                    'volatility_regime': 'volatilityregime',
                    'symbol_cluster': 'symbolcluster'
                }
                lines.append("🔧 Feature Importance (RF)")
                for k, v in fi.items():
                    disp = name_map.get(k, k.replace('_',''))
                    lines.append(f"• {disp}: {float(v):.1f}%")
                lines.append("")
            # Time patterns
            tp = pat.get('time_patterns', {}) or {}
            sp = tp.get('session_performance', {}) or {}
            if sp:
                lines.append("🕒 Sessions")
                for k, v in sp.items():
                    lines.append(f"• {k}: {v}")
                lines.append("")
            # Market conditions
            mc = pat.get('market_conditions', {}) or {}
            for title, mp in mc.items():
                if mp:
                    label = {
                        'volatility_regime': '🌪️ Volatility',
                        'vwap_dist_atr': '📏 VWAP Dist (ATR)',
                        'bb_width_pct': '📦 BB Width'
                    }.get(title, title)
                    lines.append(label)
                    for k, v in mp.items():
                        lines.append(f"• {k}: {v}")
                    lines.append("")
            # Simple narrative
            for n in (pat.get('winning_patterns') or []):
                lines.append(f"✅ {n}")
            for n in (pat.get('losing_patterns') or []):
                lines.append(f"❌ {n}")
            if len(lines) <= 2:
                lines.append("(no ML patterns available yet)")
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in scalp_patterns: {e}")
            await update.message.reply_text("Error fetching Scalp ML patterns")

    async def scalp_qscore_wr(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show 30d Qscore bucket win rates for Scalp phantoms (decisive outcomes only)."""
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)
            # Collect decisive phantoms with qscore
            items = []
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        q = (getattr(p, 'features', {}) or {}).get('qscore', None)
                        if isinstance(q, (int,float)):
                            items.append((float(q), oc))
                    except Exception:
                        continue
            if not items:
                await self.safe_reply(update, "📈 *Scalp Qscore WR (30d)*\nNo data yet.")
                return
            # 10-pt bins from 40 to 100
            bins = list(range(40, 101, 10))
            def _bkey(q: float) -> str:
                if q < 40:
                    return "<40"
                for b in bins:
                    if q < b+10:
                        return f"{b}-{b+9}"
                return "100+"
            agg = {}
            for q, oc in items:
                k = _bkey(q)
                s = agg.setdefault(k, {'w':0,'n':0})
                s['n'] += 1
                if oc == 'win':
                    s['w'] += 1
            ordered = [k for k in ["<40"] + [f"{b}-{b+9}" for b in bins]] if any(k in agg for k in ["<40"]) else [f"{b}-{b+9}" for b in bins]
            msg = ["📈 *Scalp Qscore WR (30d)*", ""]
            for k in ordered:
                if k in agg:
                    s = agg[k]
                    wr = (s['w']/s['n']*100.0) if s['n'] else 0.0
                    msg.append(f"• {k}: WR {wr:.1f}% (N={s['n']})")
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"Error in scalp_qscore_wr: {e}")
            await update.message.reply_text("Error computing Qscore WR")

    async def scalp_mlscore_wr(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show 30d ML-score bucket win rates for Scalp phantoms (decisive outcomes only)."""
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)
            # Collect decisive phantoms with ml score
            items = []
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        ml = (getattr(p, 'features', {}) or {}).get('ml', None)
                        if isinstance(ml, (int,float)):
                            items.append((float(ml), oc))
                    except Exception:
                        continue
            if not items:
                await self.safe_reply(update, "📈 *Scalp ML WR (30d)*\nNo data yet.")
                return
            bins = list(range(40, 101, 10))
            def _bkey(v: float) -> str:
                if v < 40:
                    return "<40"
                for b in bins:
                    if v < b+10:
                        return f"{b}-{b+9}"
                return "100+"
            agg = {}
            for v, oc in items:
                k = _bkey(v)
                s = agg.setdefault(k, {'w':0,'n':0})
                s['n'] += 1
                if oc == 'win':
                    s['w'] += 1
            ordered = [k for k in ["<40"] + [f"{b}-{b+9}" for b in bins]] if any(k in agg for k in ["<40"]) else [f"{b}-{b+9}" for b in bins]
            msg = ["📈 *Scalp ML WR (30d)*", ""]
            for k in ordered:
                if k in agg:
                    s = agg[k]
                    wr = (s['w']/s['n']*100.0) if s['n'] else 0.0
                    msg.append(f"• {k}: WR {wr:.1f}% (N={s['n']})")
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"Error in scalp_mlscore_wr: {e}")
            await update.message.reply_text("Error computing ML WR")

    async def scalp_time_wr(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show 30d WR by session and day of week for Scalp phantoms (decisive only)."""
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)
            # Aggregate
            sess_map = {'asian': {'w':0,'n':0}, 'european': {'w':0,'n':0}, 'us': {'w':0,'n':0}, 'off_hours': {'w':0,'n':0}}
            dow_map = {i: {'w':0,'n':0} for i in range(7)}  # 0=Mon..6=Sun
            def _session_fallback(dt: datetime) -> str:
                hr = dt.hour if dt else 0
                if 0 <= hr < 8: return 'asian'
                if 8 <= hr < 16: return 'european'
                return 'us'
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        # Session
                        feats = getattr(p, 'features', {}) or {}
                        sess = str(feats.get('session')) if feats.get('session') else _session_fallback(et)
                        if sess not in sess_map:
                            sess = 'off_hours'
                        sess_map[sess]['n'] += 1
                        if oc == 'win': sess_map[sess]['w'] += 1
                        # Day of week
                        d = et.weekday()
                        if d in dow_map:
                            dow_map[d]['n'] += 1
                            if oc == 'win': dow_map[d]['w'] += 1
                    except Exception:
                        continue
            # Build message
            lines = ["🗓 *Scalp Sessions/Days (30d)*", ""]
            # Sessions
            lines.append("Sessions")
            order = ['asian','european','us','off_hours']
            for k in order:
                s = sess_map.get(k, {'w':0,'n':0}); wr = (s['w']/s['n']*100.0) if s['n'] else 0.0
                low = " (low N)" if s['n'] and s['n'] < 5 else ""
                lines.append(f"• {k}: WR {wr:.1f}% (N={s['n']}){low}")
            # Days of week
            lines.append("")
            lines.append("Days")
            day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            best = None
            for i in range(7):
                s = dow_map[i]; wr = (s['w']/s['n']*100.0) if s['n'] else 0.0
                low = " (low N)" if s['n'] and s['n'] < 5 else ""
                lines.append(f"• {day_names[i]}: WR {wr:.1f}% (N={s['n']}){low}")
                if s['n'] >= 5:
                    if best is None or wr > best[1]:
                        best = (i, wr, s['n'])
            if best:
                lines.append(f"\nBest day: {day_names[best[0]]} {best[1]:.1f}% (N={best[2]})")
            # Inline keyboard: direct links to Vars-by-Session/Day views
            kb_rows = []
            try:
                from telegram import InlineKeyboardMarkup, InlineKeyboardButton
                kb_rows.append([
                    InlineKeyboardButton("Vars: asian", callback_data="ui:scalp:timewr_vars:session:asian"),
                    InlineKeyboardButton("Vars: european", callback_data="ui:scalp:timewr_vars:session:european")
                ])
                kb_rows.append([
                    InlineKeyboardButton("Vars: us", callback_data="ui:scalp:timewr_vars:session:us"),
                    InlineKeyboardButton("Vars: off_hours", callback_data="ui:scalp:timewr_vars:session:off_hours")
                ])
                # Days (Mon..Sun)
                day_keys = ['mon','tue','wed','thu','fri','sat','sun']
                row = []
                for i, dk in enumerate(day_keys):
                    row.append(InlineKeyboardButton(dk.title(), callback_data=f"ui:scalp:timewr_vars:day:{dk}"))
                    if (i % 4) == 3:
                        kb_rows.append(row)
                        row = []
                if row:
                    kb_rows.append(row)
                kb = InlineKeyboardMarkup(kb_rows)
                try:
                    await update.message.reply_text("\n".join(lines), reply_markup=kb)
                except Exception:
                    # Fallback without keyboard
                    await self.safe_reply(update, "\n".join(lines))
            except Exception:
                await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in scalp_time_wr: {e}")
            await update.message.reply_text("Error computing Sessions/Days WR")

    async def scalp_ema_slopes(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Detailed EMA slope analysis showing win rates by value ranges and combinations.

        Shows individual and combined win rates for emaslopefast and emaslopeslow features.
        User specified these are the two most important features (19.8% and 15.2% importance).
        """
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)

            # Collect decisive phantoms with EMA slope data AND side
            items = []
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        side = getattr(p, 'side', None)
                        if not side:
                            continue
                        feats = getattr(p, 'features', {}) or {}
                        fast = feats.get('ema_slope_fast', None)
                        slow = feats.get('ema_slope_slow', None)
                        if isinstance(fast, (int, float)) and isinstance(slow, (int, float)):
                            items.append((float(fast), float(slow), oc, str(side).lower()))
                    except Exception:
                        continue

            if not items:
                await self.safe_reply(update, "📉 *EMA Slopes Analysis (30d)*\nNo data yet.")
                return

            # Separate longs and shorts
            longs = [(f, s, oc) for f, s, oc, side in items if side == 'long']
            shorts = [(f, s, oc) for f, s, oc, side in items if side == 'short']

            # Define value ranges for binning (percentage values)
            fast_bins = [
                ("<-0.10", lambda x: x < -0.10),
                ("-0.10--0.05", lambda x: -0.10 <= x < -0.05),
                ("-0.05--0.03", lambda x: -0.05 <= x < -0.03),
                ("-0.03--0.01", lambda x: -0.03 <= x < -0.01),
                ("-0.01-0.00", lambda x: -0.01 <= x < 0.00),
                ("0.00-0.03", lambda x: 0.00 <= x < 0.03),
                ("0.03+", lambda x: x >= 0.03),
            ]

            slow_bins = [
                ("<-0.05", lambda x: x < -0.05),
                ("-0.05--0.03", lambda x: -0.05 <= x < -0.03),
                ("-0.03--0.015", lambda x: -0.03 <= x < -0.015),
                ("-0.015-0.00", lambda x: -0.015 <= x < 0.00),
                ("0.00-0.015", lambda x: 0.00 <= x < 0.015),
                ("0.015+", lambda x: x >= 0.015),
            ]

            # Helper function to analyze slopes for a dataset
            def analyze_slopes(dataset, direction_name):
                if not dataset:
                    return None

                # Aggregate individual feature win rates
                fast_agg = {}
                slow_agg = {}

                for fast, slow, oc in dataset:
                    # Fast EMA slope
                    for label, test in fast_bins:
                        if test(fast):
                            s = fast_agg.setdefault(label, {'w': 0, 'n': 0})
                            s['n'] += 1
                            if oc == 'win':
                                s['w'] += 1
                            break

                    # Slow EMA slope
                    for label, test in slow_bins:
                        if test(slow):
                            s = slow_agg.setdefault(label, {'w': 0, 'n': 0})
                            s['n'] += 1
                            if oc == 'win':
                                s['w'] += 1
                            break

                # Aggregate combination win rates
                combo_agg = {}
                for fast, slow, oc in dataset:
                    fast_label = None
                    slow_label = None

                    for label, test in fast_bins:
                        if test(fast):
                            fast_label = label
                            break

                    for label, test in slow_bins:
                        if test(slow):
                            slow_label = label
                            break

                    if fast_label and slow_label:
                        combo_key = f"F:{fast_label} × S:{slow_label}"
                        s = combo_agg.setdefault(combo_key, {'w': 0, 'n': 0})
                        s['n'] += 1
                        if oc == 'win':
                            s['w'] += 1

                return {
                    'fast_agg': fast_agg,
                    'slow_agg': slow_agg,
                    'combo_agg': combo_agg,
                    'count': len(dataset)
                }

            # Build message
            msg = [
                "📉 *EMA Slopes Analysis (30d)*",
                "━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "🔢 *Feature Importance:*",
                "• emaslopefast: 19.8%",
                "• emaslopeslow: 15.2%",
                ""
            ]

            # Analyze LONGS
            if longs:
                long_analysis = analyze_slopes(longs, "LONG")
                msg.extend([
                    "═══════════════════════════",
                    "🟢 *LONG TRADES*",
                    "═══════════════════════════",
                    "",
                    "📊 *Strategy Semantics:*",
                    "• ✅ POSITIVE slopes = Trend Following (breakouts)",
                    "• ⚠️ NEGATIVE slopes = Mean Reversion (buy dips)",
                    "",
                    "⚡ *Fast EMA Slope % (Individual)*",
                    ""
                ])

                # Sort fast by win rate descending
                fast_sorted = sorted(
                    [(k, v) for k, v in long_analysis['fast_agg'].items()],
                    key=lambda x: (x[1]['w'] / x[1]['n']) if x[1]['n'] else 0,
                    reverse=True
                )

                for label, stats in fast_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    if wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 20.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    msg.append(f"• {label:>15}: {indicator} WR {wr:5.1f}% (N={stats['n']:>3})")

                msg.extend(["", "🐌 *Slow EMA Slope % (Individual)*", ""])

                # Sort slow by win rate descending
                slow_sorted = sorted(
                    [(k, v) for k, v in long_analysis['slow_agg'].items()],
                    key=lambda x: (x[1]['w'] / x[1]['n']) if x[1]['n'] else 0,
                    reverse=True
                )

                for label, stats in slow_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    if wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 20.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    msg.append(f"• {label:>15}: {indicator} WR {wr:5.1f}% (N={stats['n']:>3})")

                msg.extend(["", "🔁 *Top 10 Long Combinations*", ""])

                # Sort combinations by win rate descending
                combo_sorted = sorted(
                    [(k, v) for k, v in long_analysis['combo_agg'].items() if v['n'] >= 3],
                    key=lambda x: ((x[1]['w'] / x[1]['n']), x[1]['n']) if x[1]['n'] else (0, 0),
                    reverse=True
                )[:10]

                for combo_key, stats in combo_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    if wr >= 50.0:
                        indicator = "🟢"
                    elif wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 30.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    msg.append(f"• {indicator} WR {wr:5.1f}% (N={stats['n']:>3}) | {combo_key}")

                if not combo_sorted:
                    msg.append("(No combinations with N≥3)")

                msg.extend(["", f"📊 *Long Samples: {long_analysis['count']}*", ""])

            # Analyze SHORTS
            if shorts:
                short_analysis = analyze_slopes(shorts, "SHORT")
                msg.extend([
                    "═══════════════════════════",
                    "🔴 *SHORT TRADES*",
                    "═══════════════════════════",
                    "",
                    "📊 *Strategy Semantics:*",
                    "• ✅ NEGATIVE slopes = Trend Following (breakdowns)",
                    "• ⚠️ POSITIVE slopes = Mean Reversion (sell rips)",
                    "",
                    "⚡ *Fast EMA Slope % (Individual)*",
                    ""
                ])

                # Sort fast by win rate descending
                fast_sorted = sorted(
                    [(k, v) for k, v in short_analysis['fast_agg'].items()],
                    key=lambda x: (x[1]['w'] / x[1]['n']) if x[1]['n'] else 0,
                    reverse=True
                )

                for label, stats in fast_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    if wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 20.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    msg.append(f"• {label:>15}: {indicator} WR {wr:5.1f}% (N={stats['n']:>3})")

                msg.extend(["", "🐌 *Slow EMA Slope % (Individual)*", ""])

                # Sort slow by win rate descending
                slow_sorted = sorted(
                    [(k, v) for k, v in short_analysis['slow_agg'].items()],
                    key=lambda x: (x[1]['w'] / x[1]['n']) if x[1]['n'] else 0,
                    reverse=True
                )

                for label, stats in slow_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    if wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 20.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    msg.append(f"• {label:>15}: {indicator} WR {wr:5.1f}% (N={stats['n']:>3})")

                msg.extend(["", "🔁 *Top 10 Short Combinations*", ""])

                # Sort combinations by win rate descending
                combo_sorted = sorted(
                    [(k, v) for k, v in short_analysis['combo_agg'].items() if v['n'] >= 3],
                    key=lambda x: ((x[1]['w'] / x[1]['n']), x[1]['n']) if x[1]['n'] else (0, 0),
                    reverse=True
                )[:10]

                for combo_key, stats in combo_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    if wr >= 50.0:
                        indicator = "🟢"
                    elif wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 30.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    msg.append(f"• {indicator} WR {wr:5.1f}% (N={stats['n']:>3}) | {combo_key}")

                if not combo_sorted:
                    msg.append("(No combinations with N≥3)")

                msg.extend(["", f"📊 *Short Samples: {short_analysis['count']}*", ""])

            msg.extend(["", f"📊 *Total Samples: {len(items)} (L:{len(longs)} | S:{len(shorts)})*"])

            await self.safe_reply(update, "\n".join(msg))

        except Exception as e:
            logger.error(f"Error in scalp_ema_slopes: {e}")
            await update.message.reply_text("Error computing EMA slope analysis")

    async def scalp_ema_slopes_evci(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """EMA slope analysis with EV (realized R), Wilson CI and Lift vs baseline.

        Strategy-aware view:
        - Filters by acceptance_path based on scalp.signal.vwap_pattern (bounce|reject|revert)
        - 30d decisive outcomes (win/loss), phantoms + executed mirrors
        - Solo fast/slow bins ranked by EV lift
        - Top fast×slow combos by EV_R (N≥50)
        - ASCII heatmap of WR% for fast×slow (cells with N<20 shown as —)
        """
        try:
            from datetime import datetime, timedelta
            from math import sqrt
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker

            def wilson_ci(w:int, n:int, z:float=1.96):
                if n <= 0:
                    return (0.0, 0.0)
                p = w / n
                denom = 1.0 + (z*z)/n
                center = (p + (z*z)/(2*n)) / denom
                margin = z * sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
                lo, hi = max(0.0, center - margin), min(1.0, center + margin)
                return (lo*100.0, hi*100.0)

            scpt = get_scalp_phantom_tracker()
            # Read current strategy pattern for filtering and title
            try:
                cfg = self.shared.get('config', {}) or {}
                pat = str(((cfg.get('scalp', {}) or {}).get('signal', {}) or {}).get('vwap_pattern', 'revert')).lower()
            except Exception:
                pat = 'revert'
            cutoff = datetime.utcnow() - timedelta(days=30)

            # Collect decisive scalps with required features
            items = []  # (fast, slow, win, rr, pnl)
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        if getattr(p, 'outcome', None) not in ('win','loss'):
                            continue
                        feats = getattr(p, 'features', {}) or {}
                        # Filter by acceptance_path if set (strategy-aware)
                        ap = str(feats.get('acceptance_path', '') or '')
                        if pat == 'bounce' and ap and ap != 'vwap_bounce':
                            continue
                        if pat == 'reject' and ap and ap != 'vwap_reject':
                            continue
                        fast = feats.get('ema_slope_fast', None)
                        slow = feats.get('ema_slope_slow', None)
                        if not (isinstance(fast, (int,float)) and isinstance(slow, (int,float))):
                            continue
                        rr = getattr(p, 'realized_rr', None)
                        pnl = getattr(p, 'pnl_percent', None)
                        rr = float(rr) if isinstance(rr, (int,float)) else 0.0
                        pnl = float(pnl) if isinstance(pnl, (int,float)) else 0.0
                        win = 1 if getattr(p, 'outcome', None) == 'win' else 0
                        side = str(getattr(p, 'side', ''))
                        items.append((float(fast), float(slow), win, rr, pnl, side))
                    except Exception:
                        continue

            if not items:
                await self.safe_reply(update, "EMA Slopes (EV+CI) 30d\nNo data yet.", parse_mode=None)
                return

            # Baseline metrics
            n_total = len(items)
            w_total = sum(1 for t in items if t[2]==1)
            wr_base = (w_total/n_total*100.0) if n_total else 0.0
            evr_base = (sum(t[3] for t in items) / n_total) if n_total else 0.0

            # Fixed bins (percent units like existing view)
            fast_bins = [
                ("<-0.10", lambda x: x < -0.10),
                ("-0.10--0.05", lambda x: -0.10 <= x < -0.05),
                ("-0.05--0.03", lambda x: -0.05 <= x < -0.03),
                ("-0.03--0.01", lambda x: -0.03 <= x < -0.01),
                ("-0.01-0.00", lambda x: -0.01 <= x < 0.00),
                ("0.00-0.03", lambda x: 0.00 <= x < 0.03),
                ("0.03+", lambda x: x >= 0.03),
            ]
            slow_bins = [
                ("<-0.05", lambda x: x < -0.05),
                ("-0.05--0.03", lambda x: -0.05 <= x < -0.03),
                ("-0.03--0.015", lambda x: -0.03 <= x < -0.015),
                ("-0.015-0.00", lambda x: -0.015 <= x < 0.00),
                ("0.00-0.015", lambda x: 0.00 <= x < 0.015),
                ("0.015+", lambda x: x >= 0.015),
            ]

            def bin_label(val: float, bins):
                for label, test in bins:
                    if test(val):
                        return label
                return None

            # Aggregations
            fast_agg = {}
            slow_agg = {}
            combo_agg = {}
            for f, s, w, rr, pnl, side in items:
                fl = bin_label(f, fast_bins)
                sl = bin_label(s, slow_bins)
                if fl:
                    a = fast_agg.setdefault(fl, {'w':0,'n':0,'rr':0.0,'pnl':0.0})
                    a['n'] += 1; a['w'] += w; a['rr'] += rr; a['pnl'] += pnl
                if sl:
                    a = slow_agg.setdefault(sl, {'w':0,'n':0,'rr':0.0,'pnl':0.0})
                    a['n'] += 1; a['w'] += w; a['rr'] += rr; a['pnl'] += pnl
                if fl and sl:
                    key = f"F:{fl} × S:{sl}"
                    a = combo_agg.setdefault(key, {'w':0,'n':0,'rr':0.0,'pnl':0.0})
                    a['n'] += 1; a['w'] += w; a['rr'] += rr; a['pnl'] += pnl

            def fmt_bin(name: str, agg: dict):
                n = agg['n']; w = agg['w']
                wr = (w/n*100.0) if n else 0.0
                lo, hi = wilson_ci(w, n)
                evr = (agg['rr']/n) if n else 0.0
                lift_wr = wr - wr_base
                lift_evr = evr - evr_base
                return f"• {name:>12}: WR {wr:5.1f}% [{lo:4.0f}-{hi:4.0f}] N={n:>3} | EV_R {evr:+.2f} | Lift_WR {lift_wr:+.1f} | Lift_EV {lift_evr:+.2f}"

            # Build message with chunking
            lines = []
            title_suffix = " (Bounce)" if pat=='bounce' else (" (Reject)" if pat=='reject' else "")
            lines.append(f"📉 EMA Slopes (EV+CI) — 30d{title_suffix}")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")
            # Strategy hint text
            if pat=='bounce':
                lines.append("Trend-continuation (VWAP bounce): expect slow>0 with fast ~0–0.03 near VWAP; view is filtered to bounce signals.")
            elif pat=='reject':
                lines.append("Mean-reversion (VWAP rejection): expect negative slopes to lead; view filtered to rejection signals.")
            lines.append(f"Baseline: WR {wr_base:.1f}% | EV_R {evr_base:+.2f} (N={n_total})")
            lines.append("")
            lines.append("⚡ Fast EMA Slope bins (ranked by Lift_EV)")
            fast_sorted = sorted(fast_agg.items(), key=lambda kv: ((kv[1]['rr']/kv[1]['n']) if kv[1]['n'] else -1e9), reverse=True)
            for name, agg in fast_sorted:
                if agg['n'] >= 30:
                    lines.append(fmt_bin(name, agg))

            lines.append("")
            lines.append("🐌 Slow EMA Slope bins (ranked by Lift_EV)")
            slow_sorted = sorted(slow_agg.items(), key=lambda kv: ((kv[1]['rr']/kv[1]['n']) if kv[1]['n'] else -1e9), reverse=True)
            for name, agg in slow_sorted:
                if agg['n'] >= 30:
                    lines.append(fmt_bin(name, agg))

            # Top combos by EV_R
            lines.append("")
            lines.append("🔁 Top Fast×Slow combinations by EV_R (N≥50)")
            combo_sorted = sorted(combo_agg.items(), key=lambda kv: ((kv[1]['rr']/kv[1]['n']) if kv[1]['n'] else -1e9), reverse=True)
            shown = 0
            for key, agg in combo_sorted:
                if agg['n'] < 50:
                    continue
                n = agg['n']; w = agg['w']
                wr = (w/n*100.0) if n else 0.0
                lo, hi = wilson_ci(w, n)
                evr = (agg['rr']/n) if n else 0.0
                lift_wr = wr - wr_base
                lift_evr = evr - evr_base
                lines.append(f"• WR {wr:5.1f}% [{lo:4.0f}-{hi:4.0f}] N={n:>3} | EV_R {evr:+.2f} | Lift_WR {lift_wr:+.1f} | Lift_EV {lift_evr:+.2f} | {key}")
                shown += 1
                if shown >= 10:
                    break

            # Per-side slope bins (to guide thresholds)
            try:
                lines.append("")
                def side_bins(side_key: str):
                    arr = [t for t in items if str(t[-1]).lower()==side_key]
                    if not arr:
                        return []
                    n_side = len(arr)
                    w_side = sum(1 for t in arr if t[2]==1)
                    wr_side = (w_side/n_side*100.0) if n_side else 0.0

                    # Add semantic context based on direction
                    if side_key == 'long':
                        out = [
                            "═══════════════════════════",
                            f"🟢 LONG TRADES (N={n_side}, WR={wr_side:.1f}%)",
                            "═══════════════════════════",
                            "• ✅ POSITIVE slopes = Trend (breakouts)",
                            "• ⚠️ NEGATIVE slopes = Mean Reversion (buy dips)"
                        ]
                    else:
                        out = [
                            "═══════════════════════════",
                            f"🔴 SHORT TRADES (N={n_side}, WR={wr_side:.1f}%)",
                            "═══════════════════════════",
                            "• ✅ NEGATIVE slopes = Trend (breakdowns)",
                            "• ⚠️ POSITIVE slopes = Mean Reversion (sell rips)"
                        ]

                    # Fast
                    out.append("")
                    out.append("⚡ Fast EMA Slope")
                    for label, test in fast_bins:
                        n = w = 0
                        for t in arr:
                            if test(t[0]):
                                n += 1; w += int(t[2])
                        if n:
                            wr = w/n*100.0
                            mark = "🟢" if wr >= 50 else ("✅" if wr >= 40 else ("⚠️" if wr >= 30 else "❌"))
                            out.append(f"• {label:>12}: {mark} WR {wr:5.1f}% (N={n:>3})")
                    # Slow
                    out.append("")
                    out.append("🐌 Slow EMA Slope")
                    for label, test in slow_bins:
                        n = w = 0
                        for t in arr:
                            if test(t[1]):
                                n += 1; w += int(t[2])
                        if n:
                            wr = w/n*100.0
                            mark = "🟢" if wr >= 50 else ("✅" if wr >= 40 else ("⚠️" if wr >= 30 else "❌"))
                            out.append(f"• {label:>12}: {mark} WR {wr:5.1f}% (N={n:>3})")
                    return out
                lines.extend(side_bins('long'))
                lines.append("")
                lines.extend(side_bins('short'))
            except Exception:
                pass

            # Heatmap (WR%)
            lines.append("")
            lines.append("🗺 Heatmap WR% (rows=Slow, cols=Fast; '—' if N<20)")
            fast_labels = [fb[0] for fb in fast_bins]
            header = "          " + "  ".join([f"{lbl:>11}" for lbl in fast_labels])
            lines.append(header)
            for sl, _ in slow_bins:
                row_vals = []
                for fl, _ in fast_bins:
                    key = f"F:{fl} × S:{sl}"
                    agg = combo_agg.get(key)
                    if not agg or agg['n'] < 20:
                        row_vals.append(f"{'-':>11}")
                        continue
                    wr = (agg['w']/agg['n']*100.0) if agg['n'] else 0.0
                    row_vals.append(f"{wr:5.1f}%{(' (N)') if False else '':>5}")
                lines.append(f"{sl:>10}  " + "  ".join(row_vals))

            # Send in chunks
            MAX = 3500
            buf = []
            cur = 0
            for ln in lines:
                if cur + len(ln) + 1 > MAX and buf:
                    await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                    buf = []
                    cur = 0
                buf.append(ln)
                cur += len(ln) + 1
            if buf:
                await self.safe_reply(update, "\n".join(buf), parse_mode=None)

        except Exception as e:
            logger.error(f"Error in scalp_ema_slopes_evci: {e}")
            try:
                await self.safe_reply(update, "Error computing Slopes (EV+CI)", parse_mode=None)
            except Exception:
                pass

    async def scalp_advanced_combos(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Multi-feature combination analysis showing proven high-WR patterns.

        Analyzes top 5 features (66.5% total importance) across 30 days:
        - ema_slope_fast (16.4%), ema_slope_slow (18.9%)
        - atr_pct (12.5%), bb_width_pct (10.3%), vwap_dist_atr (8.4%)

        Only shows combinations with N≥50 for statistical reliability.
        """
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)

            # Collect decisive phantoms with all 5 features
            items = []
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        feats = getattr(p, 'features', {}) or {}
                        ap = str(feats.get('acceptance_path', '') or '')
                        if pat == 'bounce' and ap and ap != 'vwap_bounce':
                            continue
                        if pat == 'reject' and ap and ap != 'vwap_reject':
                            continue

                        # Extract all 5 features
                        fast = feats.get('ema_slope_fast', None)
                        slow = feats.get('ema_slope_slow', None)
                        atr = feats.get('atr_pct', None)
                        bbw = feats.get('bb_width_pct', None)
                        vwap = feats.get('vwap_dist_atr', None)

                        # Only include if all features present
                        if all(isinstance(x, (int, float)) for x in [fast, slow, atr, bbw, vwap]):
                            items.append((
                                float(fast), float(slow), float(atr),
                                float(bbw), float(vwap), oc
                            ))
                    except Exception:
                        continue

            if not items:
                await self.safe_reply(update, "📊 *Advanced Combos Analysis (30d)*\nNo data yet.")
                return

            # Define bins for each feature
            fast_bins = [
                ("<-0.05", lambda x: x < -0.05),
                ("-0.05--0.01", lambda x: -0.05 <= x < -0.01),
                ("-0.01-0.01", lambda x: -0.01 <= x < 0.01),
                ("0.01-0.03", lambda x: 0.01 <= x < 0.03),
                ("0.03+", lambda x: x >= 0.03),
            ]

            slow_bins = [
                ("<-0.03", lambda x: x < -0.03),
                ("-0.03-0.00", lambda x: -0.03 <= x < 0.00),
                ("0.00-0.015", lambda x: 0.00 <= x < 0.015),
                ("0.015+", lambda x: x >= 0.015),
            ]

            atr_bins = [
                ("<0.5%", lambda x: x < 0.5),
                ("0.5-1.5%", lambda x: 0.5 <= x < 1.5),
                ("1.5%+", lambda x: x >= 1.5),
            ]

            bbw_bins = [
                ("<1.2%", lambda x: x < 1.2),
                ("1.2-2.0%", lambda x: 1.2 <= x < 2.0),
                ("2.0%+", lambda x: x >= 2.0),
            ]

            vwap_bins = [
                ("<0.6", lambda x: x < 0.6),
                ("0.6-1.0", lambda x: 0.6 <= x < 1.0),
                ("1.0+", lambda x: x >= 1.0),
            ]

            # Aggregate combination win rates
            combo_agg = {}
            for fast, slow, atr, bbw, vwap, oc in items:
                # Find labels for each feature
                fast_label = None
                slow_label = None
                atr_label = None
                bbw_label = None
                vwap_label = None

                for label, test in fast_bins:
                    if test(fast):
                        fast_label = label
                        break

                for label, test in slow_bins:
                    if test(slow):
                        slow_label = label
                        break

                for label, test in atr_bins:
                    if test(atr):
                        atr_label = label
                        break

                for label, test in bbw_bins:
                    if test(bbw):
                        bbw_label = label
                        break

                for label, test in vwap_bins:
                    if test(vwap):
                        vwap_label = label
                        break

                if all([fast_label, slow_label, atr_label, bbw_label, vwap_label]):
                    combo_key = f"F:{fast_label} S:{slow_label} ATR:{atr_label} BBW:{bbw_label} VWAP:{vwap_label}"
                    s = combo_agg.setdefault(combo_key, {'w': 0, 'n': 0, 'winners': [], 'losers': []})
                    s['n'] += 1
                    if oc == 'win':
                        s['w'] += 1
                        s['winners'].append((fast, slow, atr, bbw, vwap))  # Store raw values
                    else:
                        s['losers'].append((fast, slow, atr, bbw, vwap))

            # Filter for N≥50 and sort by WR
            combo_sorted = sorted(
                [(k, v) for k, v in combo_agg.items() if v['n'] >= 50],
                key=lambda x: (x[1]['w'] / x[1]['n']) if x[1]['n'] else 0,
                reverse=True
            )[:20]  # Top 20

            # Build message
            suffix = " (Bounce)" if pat=='bounce' else (" (Reject)" if pat=='reject' else "")
            msg = [
                f"📊 *Advanced Combos Analysis (30d){suffix}*",
                "━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "🎯 *Top 5 Features: 66.5% total importance*",
                "✅ Only combinations with N≥50 (proven patterns)",
                "",
                "🔝 *Top 20 High-WR Combinations*",
                ""
            ]

            if combo_sorted:
                import numpy as np

                for combo_key, stats in combo_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    # Visual indicator for combination quality
                    if wr >= 50.0:
                        indicator = "🟢"  # Excellent
                    elif wr >= 40.0:
                        indicator = "✅"  # Good
                    elif wr >= 30.0:
                        indicator = "⚠️"  # Medium
                    else:
                        indicator = "❌"  # Poor
                    msg.append(f"• {indicator} WR {wr:5.1f}% (N={stats['n']:>3}) | {combo_key}")

                    # Add winner statistics (averages and percentiles)
                    if stats['winners']:
                        try:
                            winners = np.array(stats['winners'])
                            means = winners.mean(axis=0)
                            p25 = np.percentile(winners, 25, axis=0)
                            p75 = np.percentile(winners, 75, axis=0)

                            # Display averages
                            msg.append(f"  ↳ Avg: F={means[0]:.3f} S={means[1]:.3f} ATR={means[2]:.2f}% BBW={means[3]:.2f}% VWAP={means[4]:.2f}")
                            # Display percentile ranges (compact format) - DISABLED for cleaner output
                            # msg.append(f"  ↳ P25-P75: F={p25[0]:.3f}-{p75[0]:.3f} S={p25[1]:.3f}-{p75[1]:.3f}")
                        except Exception:
                            pass  # Skip if numpy calc fails
            else:
                msg.append("(No combinations with N≥50 found)")
                msg.append("")
                msg.append(f"_Try lowering threshold to N≥20 if no results_")

            msg.extend(["", f"📊 *Total qualifying combos:* {len(combo_sorted)}", f"📊 *Total samples analyzed:* {len(items)}"])

            await self.safe_reply(update, "\n".join(msg))

        except Exception as e:
            logger.error(f"Error in scalp_advanced_combos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text("Error computing advanced combinations analysis")

    async def scalp_advanced_combos_evci(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Advanced Combos analysis with EV (realized R), Wilson CI and Lift vs baseline.

        - 30d decisive (win/loss) outcomes from Scalp phantoms + executed
        - Strategy-aware: filters by acceptance_path based on vwap_pattern when present
        - Binning across 5 features: fast/slow slopes, atr_pct, bb_width_pct, vwap_dist_atr
        - Top combos by EV_R (N≥50), with WR [95% CI], Lift_WR, Lift_EV
        """
        try:
            from datetime import datetime, timedelta
            from math import sqrt
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker

            def wilson_ci(w:int, n:int, z:float=1.96):
                if n <= 0:
                    return (0.0, 0.0)
                p = w / n
                denom = 1.0 + (z*z)/n
                center = (p + (z*z)/(2*n)) / denom
                margin = z * sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
                lo, hi = max(0.0, center - margin), min(1.0, center + margin)
                return (lo*100.0, hi*100.0)

            scpt = get_scalp_phantom_tracker()
            try:
                cfg = self.shared.get('config', {}) or {}
                pat = str(((cfg.get('scalp', {}) or {}).get('signal', {}) or {}).get('vwap_pattern', 'revert')).lower()
            except Exception:
                pat = 'revert'
            cutoff = datetime.utcnow() - timedelta(days=30)

            # Collect decisive phantoms with all 5 features and realized_rr
            items = []  # (fast, slow, atr, bbw, vwap, win, rr)
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        if getattr(p, 'outcome', None) not in ('win','loss'):
                            continue
                        feats = getattr(p, 'features', {}) or {}
                        # Strategy-aware acceptance filter when available
                        ap = str(feats.get('acceptance_path', '') or '')
                        if pat == 'bounce' and ap and ap != 'vwap_bounce':
                            continue
                        if pat == 'reject' and ap and ap != 'vwap_reject':
                            continue
                        fast = feats.get('ema_slope_fast'); slow = feats.get('ema_slope_slow')
                        atr = feats.get('atr_pct'); bbw = feats.get('bb_width_pct'); vwap = feats.get('vwap_dist_atr')
                        if not all(isinstance(x, (int, float)) for x in [fast, slow, atr, bbw, vwap]):
                            continue
                        rr = getattr(p, 'realized_rr', None)
                        rr = float(rr) if isinstance(rr, (int,float)) else 0.0
                        win = 1 if getattr(p, 'outcome', None) == 'win' else 0
                        items.append((float(fast), float(slow), float(atr), float(bbw), float(vwap), win, rr))
                    except Exception:
                        continue

            if not items:
                await self.safe_reply(update, "📊 Combos (EV+CI) — 30d\nNo data yet.", parse_mode=None)
                return

            # Baseline metrics
            n_total = len(items); w_total = sum(1 for *_, win, _ in items if win == 1)
            wr_base = (w_total/n_total*100.0) if n_total else 0.0
            evr_base = sum(rr for *_, rr in items) / n_total

            # Define bins
            fast_bins = [("<-0.05", lambda x: x < -0.05), ("-0.05--0.01", lambda x: -0.05 <= x < -0.01), ("-0.01-0.01", lambda x: -0.01 <= x < 0.01), ("0.01-0.03", lambda x: 0.01 <= x < 0.03), ("0.03+", lambda x: x >= 0.03)]
            slow_bins = [("<-0.03", lambda x: x < -0.03), ("-0.03-0.00", lambda x: -0.03 <= x < 0.00), ("0.00-0.015", lambda x: 0.00 <= x < 0.015), ("0.015+", lambda x: x >= 0.015)]
            atr_bins  = [("<0.5%", lambda x: x < 0.5), ("0.5-1.5%", lambda x: 0.5 <= x < 1.5), ("1.5%+", lambda x: x >= 1.5)]
            bbw_bins  = [("<1.2%", lambda x: x < 1.2), ("1.2-2.0%", lambda x: 1.2 <= x < 2.0), ("2.0%+", lambda x: x >= 2.0)]
            vwap_bins = [("<0.6", lambda x: x < 0.6), ("0.6-1.0", lambda x: 0.6 <= x < 1.0), ("1.0+", lambda x: x >= 1.0)]

            def label(val: float, bins):
                for lb, test in bins:
                    if test(val):
                        return lb
                return None

            # Aggregate combos
            combos = {}
            for f, s, a, b, v, win, rr in items:
                lf = label(f, fast_bins); ls = label(s, slow_bins); la = label(a, atr_bins); lb = label(b, bbw_bins); lv = label(v, vwap_bins)
                if not all([lf, ls, la, lb, lv]):
                    continue
                key = f"F:{lf} S:{ls} ATR:{la} BBW:{lb} VWAP:{lv}"
                c = combos.setdefault(key, {'n':0,'w':0,'rr':0.0})
                c['n'] += 1; c['w'] += int(win); c['rr'] += rr

            # Rank by EV_R
            ranked = sorted([(k,v) for k,v in combos.items() if v['n'] >= 50], key=lambda kv: (kv[1]['rr']/kv[1]['n']), reverse=True)

            lines = []
            suffix = " (Bounce)" if pat=='bounce' else (" (Reject)" if pat=='reject' else "")
            lines.append(f"📊 Combos (EV+CI) — 30d{suffix}")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")
            lines.append(f"Baseline: WR {wr_base:.1f}% | EV_R {evr_base:+.2f} (N={n_total})")
            lines.append("")
            lines.append("🔝 Top combos by EV_R (N≥50)")
            shown = 0
            for combo_key, agg in ranked:
                n = agg['n']; w = agg['w']
                wr = (w/n*100.0) if n else 0.0
                lo, hi = wilson_ci(w, n)
                evr = (agg['rr']/n) if n else 0.0
                lift_wr = wr - wr_base; lift_evr = evr - evr_base
                lines.append(f"• WR {wr:5.1f}% [{lo:4.0f}-{hi:4.0f}] N={n:>3} | EV_R {evr:+.2f} | Lift_WR {lift_wr:+.1f} | Lift_EV {lift_evr:+.2f} | {combo_key}")
                shown += 1
                if shown >= 12:
                    break

            # Send in chunks
            MAX = 3500
            buf = []
            cur = 0
            for ln in lines:
                if cur + len(ln) + 1 > MAX and buf:
                    await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                    buf = []
                    cur = 0
                buf.append(ln)
                cur += len(ln) + 1
            if buf:
                await self.safe_reply(update, "\n".join(buf), parse_mode=None)

        except Exception as e:
            logger.error(f"Error in scalp_advanced_combos_evci: {e}")
            try:
                await self.safe_reply(update, "Error computing Combos (EV+CI)", parse_mode=None)
            except Exception:
                pass
    async def scalp_gate_feature_combos(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Individual Gate + Feature combination analysis.

        Analyzes each gate variable individually (not combinations) with
        top 5 features (slopes, ATR%, BBW%, VWAP) to find proven high-WR patterns.

        Shows which individual gates + feature combinations produce highest WR.
        Only shows combinations with N≥50 for reliable statistical significance.
        """
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)

            # Collect decisive phantoms (was_executed=False) with all features
            items = []
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        # Only true phantoms (not executed mirrors)
                        if getattr(p, 'was_executed', False):
                            continue

                        feats = getattr(p, 'features', {}) or {}

                        # Extract 5 features
                        fast = feats.get('ema_slope_fast', None)
                        slow = feats.get('ema_slope_slow', None)
                        atr = feats.get('atr_pct', None)
                        bbw = feats.get('bb_width_pct', None)
                        vwap = feats.get('vwap_dist_atr', None)

                        # Only include if all features present
                        if not all(isinstance(x, (int, float)) for x in [fast, slow, atr, bbw, vwap]):
                            continue

                        # Compute gate status for this phantom
                        try:
                            gate_status = scpt.compute_gate_status(p)
                            if not gate_status:
                                continue
                        except Exception:
                            continue

                        # OPTION 3: Analyze each gate individually (not combinations)
                        # For each passing gate, create separate entry
                        key_gates = [
                            'body', 'body040', 'body045', 'body050', 'body060',
                            'impulse040', 'impulse050', 'impulse060', 'impulse080',
                            'vwap045', 'vwap060', 'vwap080', 'vwap100',
                            'htf', 'vol', 'align_15m',
                            'bbwidth60p', 'bbwidth70p', 'bbwidth80p',
                            'vol110', 'vol120', 'vol150',
                            'q040', 'q050', 'q060', 'q070'
                        ]

                        # Create one item per passing gate
                        passed_any = False
                        for gate_var in key_gates:
                            if gate_status.get(gate_var, False):
                                items.append((
                                    gate_var,
                                    float(fast), float(slow), float(atr),
                                    float(bbw), float(vwap), oc
                                ))
                                passed_any = True

                        # Also track phantoms with NO gates passing
                        if not passed_any:
                            items.append((
                                'none',
                                float(fast), float(slow), float(atr),
                                float(bbw), float(vwap), oc
                            ))
                    except Exception as e:
                        continue

            if not items:
                await self.safe_reply(update, "🚪 *Gate+Feature Analysis (30d)*\nNo data yet.")
                return

            # Define bins for features (same as Advanced Combos)
            fast_bins = [
                ("<-0.05", lambda x: x < -0.05),
                ("-0.05--0.01", lambda x: -0.05 <= x < -0.01),
                ("-0.01-0.01", lambda x: -0.01 <= x < 0.01),
                ("0.01-0.03", lambda x: 0.01 <= x < 0.03),
                ("0.03+", lambda x: x >= 0.03),
            ]

            slow_bins = [
                ("<-0.03", lambda x: x < -0.03),
                ("-0.03-0.00", lambda x: -0.03 <= x < 0.00),
                ("0.00-0.015", lambda x: 0.00 <= x < 0.015),
                ("0.015+", lambda x: x >= 0.015),
            ]

            atr_bins = [
                ("<0.5%", lambda x: x < 0.5),
                ("0.5-1.5%", lambda x: 0.5 <= x < 1.5),
                ("1.5%+", lambda x: x >= 1.5),
            ]

            bbw_bins = [
                ("<1.2%", lambda x: x < 1.2),
                ("1.2-2.0%", lambda x: 1.2 <= x < 2.0),
                ("2.0%+", lambda x: x >= 2.0),
            ]

            vwap_bins = [
                ("<0.6", lambda x: x < 0.6),
                ("0.6-1.0", lambda x: 0.6 <= x < 1.0),
                ("1.0+", lambda x: x >= 1.0),
            ]

            # Aggregate gate+feature combination win rates
            combo_agg = {}
            for gate_combo, fast, slow, atr, bbw, vwap, oc in items:
                # Find labels for each feature
                fast_label = None
                slow_label = None
                atr_label = None
                bbw_label = None
                vwap_label = None

                for label, test in fast_bins:
                    if test(fast):
                        fast_label = label
                        break

                for label, test in slow_bins:
                    if test(slow):
                        slow_label = label
                        break

                for label, test in atr_bins:
                    if test(atr):
                        atr_label = label
                        break

                for label, test in bbw_bins:
                    if test(bbw):
                        bbw_label = label
                        break

                for label, test in vwap_bins:
                    if test(vwap):
                        vwap_label = label
                        break

                if all([fast_label, slow_label, atr_label, bbw_label, vwap_label]):
                    combo_key = f"{gate_combo} | F:{fast_label} S:{slow_label} ATR:{atr_label} BBW:{bbw_label} VWAP:{vwap_label}"
                    s = combo_agg.setdefault(combo_key, {'w': 0, 'n': 0})
                    s['n'] += 1
                    if oc == 'win':
                        s['w'] += 1

            # Filter for N≥50 (balanced) and sort by WR
            combo_sorted = sorted(
                [(k, v) for k, v in combo_agg.items() if v['n'] >= 50],
                key=lambda x: (x[1]['w'] / x[1]['n']) if x[1]['n'] else 0,
                reverse=True
            )[:20]  # Top 20

            # Build message
            msg = [
                "🚪 *Gate+Feature Analysis (30d, N≥50)*",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "🎯 *Individual Gates × Slopes × ATR% × BBW% × VWAP*",
                "✅ Only proven patterns (N≥50)",
                "",
                "🔝 *Top 20 High-WR Combinations*",
                ""
            ]

            if combo_sorted:
                for combo_key, stats in combo_sorted:
                    wr = (stats['w'] / stats['n'] * 100.0) if stats['n'] else 0.0
                    # Visual indicator
                    if wr >= 50.0:
                        indicator = "🟢"
                    elif wr >= 40.0:
                        indicator = "✅"
                    elif wr >= 30.0:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"

                    # Format: "gate | F:... S:... ATR:... BBW:... VWAP:..."
                    msg.append(f"• {indicator} WR {wr:5.1f}% (N={stats['n']:>3}) | {combo_key}")
            else:
                msg.append("(No combinations with N≥50 found)")
                msg.append("")
                msg.append("_Insufficient data. Need more phantom samples._")
                msg.append(f"_Total samples: {len(items)}, unique combos: {len(combo_agg)}_")

            msg.extend([
                "",
                f"📊 *Total combos with N≥50:* {len(combo_sorted)}",
                f"📊 *Total samples analyzed:* {len(items)}"
            ])

            await self.safe_reply(update, "\n".join(msg))

        except Exception as e:
            logger.error(f"Error in scalp_gate_feature_combos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text("Error computing gate+feature combinations analysis")

    async def scalp_pro_analytics(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Pro Analytics: RSI/MACD/MAs/VWAP/Fibonacci/Structure/MTF with EV and CI.

        - 30d decisive outcomes (Scalp phantoms + executed mirrors)
        - Top combos across: RSI bin × MACD state × VWAP bin × Fib zone × MTF agree
        - Wilson 95% CI, EV_R, Lift vs baseline; N≥50
        """
        try:
            from datetime import datetime, timedelta
            from math import sqrt
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker

            def wilson_ci(w:int, n:int, z:float=1.96):
                if n <= 0:
                    return (0.0, 0.0)
                p = w / n
                denom = 1.0 + (z*z)/n
                center = (p + (z*z)/(2*n)) / denom
                margin = z * sqrt((p*(1-p)/n) + (z*z)/(4*n*n)) / denom
                lo, hi = max(0.0, center - margin), min(1.0, center + margin)
                return (lo*100.0, hi*100.0)

            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)

            # Prefer aggregated counters from Redis for speed; fall back to raw scan
            items = []  # (rsi, macd_hist, vwap, fib_zone, mtf15, win, rr, side)
            try:
                r = getattr(scpt, 'redis_client', None)
                if r is None:
                    raise RuntimeError('no_redis')
                from collections import defaultdict
                agg = {'long': defaultdict(lambda: {'n':0,'w':0,'rr':0.0}), 'short': defaultdict(lambda: {'n':0,'w':0,'rr':0.0})}
                for i in range(0, 30):
                    day = (datetime.utcnow() - timedelta(days=i)).strftime('%Y%m%d')
                    for side in ('long','short'):
                        n_key = f"combos:scalp:n:{day}:{side}"
                        w_key = f"combos:scalp:w:{day}:{side}"
                        rr_key = f"combos:scalp:rr:{day}:{side}"
                        n_map = r.hgetall(n_key) or {}
                        if not n_map:
                            continue
                        w_map = r.hgetall(w_key) or {}
                        rr_map = r.hgetall(rr_key) or {}
                        for k, nv in n_map.items():
                            try:
                                agg[side][k]['n'] += int(nv)
                                agg[side][k]['w'] += int(w_map.get(k, 0) or 0)
                                agg[side][k]['rr'] += float(rr_map.get(k, 0.0) or 0.0)
                            except Exception:
                                continue
                # Convert to items list (use bin midpoints; ranking unaffected)
                def _mid_rsi(lbl:str)->float:
                    return 25.0 if lbl=='<30' else 35.0 if lbl=='30-40' else 50.0 if lbl=='40-60' else 65.0 if lbl=='60-70' else 75.0
                def _macd_val(lbl:str)->float:
                    return 1.0 if lbl=='bull' else -1.0
                def _vwap_mid(lbl:str)->float:
                    return 0.5 if lbl=='<0.6' else 0.9 if lbl=='0.6-1.2' else 1.3
                for side in ('long','short'):
                    for key, a in agg[side].items():
                        try:
                            parts = key.split()
                            rsi_lbl = parts[0].split(':',1)[1]
                            macd_lbl = parts[1].split(':',1)[1]
                            vwap_lbl = parts[2].split(':',1)[1]
                            fibz = parts[3].split(':',1)[1]
                            ma = parts[4]
                            mtf = True if ma == 'MTF' else False
                            rsi_mid = _mid_rsi(rsi_lbl); macd_val = _macd_val(macd_lbl); vwap_mid = _vwap_mid(vwap_lbl)
                            n = a['n']; w = a['w']; rr_sum = a['rr']
                            if n <= 0:
                                continue
                            # Represent as a single aggregate row (N captured later by combos map)
                            # Put 'w' (wins) and 'rr_sum' in the win/rr slots to be aggregated downstream
                            items.append((rsi_mid, macd_val, vwap_mid, fibz, mtf, w, rr_sum, side))
                        except Exception:
                            continue
            except Exception:
                # Fallback: scan completed in memory
                for arr in (getattr(scpt, 'completed', {}) or {}).values():
                    for p in arr:
                        try:
                            et = getattr(p, 'exit_time', None)
                            if not et or et < cutoff:
                                continue
                            if getattr(p, 'outcome', None) not in ('win','loss'):
                                continue
                            side = getattr(p, 'side', None)
                            if not side:
                                continue
                            f = getattr(p, 'features', {}) or {}
                            rsi = f.get('rsi_14'); mh = f.get('macd_hist')
                            vwap = f.get('vwap_dist_atr'); fibz = f.get('fib_zone'); mtf = f.get('mtf_agree_15')
                            if not (isinstance(rsi,(int,float)) and isinstance(mh,(int,float)) and isinstance(vwap,(int,float)) and isinstance(fibz, str) and isinstance(mtf, (bool,int))):
                                continue
                            rr = getattr(p, 'realized_rr', None); rr = float(rr) if isinstance(rr,(int,float)) else 0.0
                            win = 1 if getattr(p,'outcome',None)=='win' else 0
                            items.append((float(rsi), float(mh), float(vwap), str(fibz), bool(mtf), win, rr, str(side).lower()))
                        except Exception:
                            continue

            if not items:
                await self.safe_reply(update, "📈 Pro Analytics (EV+CI) — 30d\nNo data yet.", parse_mode=None)
                return

            # Separate longs and shorts
            longs = [(rsi, mh, vwap, fz, mtf, win, rr) for rsi, mh, vwap, fz, mtf, win, rr, side in items if side == 'long']
            shorts = [(rsi, mh, vwap, fz, mtf, win, rr) for rsi, mh, vwap, fz, mtf, win, rr, side in items if side == 'short']

            # Bins
            rsi_bins = [("<30", lambda x: x < 30), ("30-40", lambda x: 30 <= x < 40), ("40-60", lambda x: 40 <= x < 60), ("60-70", lambda x: 60 <= x < 70), ("70+", lambda x: x >= 70)]
            macd_bins = [("bull", lambda h: h > 0), ("bear", lambda h: h <= 0)]
            vwap_bins = [("<0.6", lambda x: x < 0.6), ("0.6-1.2", lambda x: 0.6 <= x < 1.2), ("1.2+", lambda x: x >= 1.2)]
            fib_bins = ["0-23","23-38","38-50","50-61","61-78","78-100"]

            def lab(val, bins):
                for lb, fn in bins:
                    if fn(val):
                        return lb
                return None

            # Helper function to analyze combos for a dataset
            def analyze_combos(dataset, direction_name):
                if not dataset:
                    return None

                combos = {}
                for rsi, mh, vwap, fibz, mtf, win, rr in dataset:
                    r = lab(rsi, rsi_bins); m = lab(mh, macd_bins); v = lab(vwap, vwap_bins)
                    fz = fibz if fibz in fib_bins else None; ma = 'MTF' if bool(mtf) else 'noMTF'
                    if not all([r, m, v, fz, ma]):
                        continue
                    key = f"RSI:{r} MACD:{m} VWAP:{v} Fib:{fz} {ma}"
                    agg = combos.setdefault(key, {'n':0,'w':0,'rr':0.0})
                    agg['n'] += 1; agg['w'] += int(win); agg['rr'] += rr

                return {'combos': combos, 'count': len(dataset)}

            # Build message
            lines = [
                "📈 Pro Analytics (EV+CI) — 30d",
                "━━━━━━━━━━━━━━━━━━━━━━━━",
                ""
            ]

            # Analyze LONGS
            if longs:
                long_analysis = analyze_combos(longs, "LONG")
                n_long = len(longs)
                # Baseline from raw rows (fallback) — aggregated path approximates, but we prefer raw baseline when available
                try:
                    w_long = sum(1 for *_, win, _ in longs if isinstance(win, (int, float)) and win == 1)
                    wr_long = (w_long/n_long*100.0) if n_long else 0.0
                    evr_long = sum((rr if isinstance(rr,(int,float)) else 0.0) for *_, rr in longs) / n_long if n_long else 0.0
                except Exception:
                    wr_long = 0.0; evr_long = 0.0

                lines.extend([
                    "═══════════════════════════",
                    f"🟢 LONG TRADES (N={n_long})",
                    "═══════════════════════════",
                    "",
                    f"Baseline: WR {wr_long:.1f}% | EV_R {evr_long:+.2f}",
                    "",
                    "🔝 Top 10 Long Combos by EV_R (N≥20)"
                ])

                ranked_long = sorted([(k,v) for k,v in long_analysis['combos'].items() if v['n'] >= 20],
                                    key=lambda kv: (kv[1]['rr']/kv[1]['n']), reverse=True)[:10]

                for key, agg in ranked_long:
                    n = agg['n']; w = agg['w']
                    wr = (w/n*100.0) if n else 0.0
                    lo, hi = wilson_ci(w, n)
                    evr = (agg['rr']/n) if n else 0.0
                    lift_wr = wr - wr_long; lift_evr = evr - evr_long
                    lines.append(f"• WR {wr:5.1f}% [{lo:4.0f}-{hi:4.0f}] N={n:>3} | EV_R {evr:+.2f} | Lift_WR {lift_wr:+.1f} | Lift_EV {lift_evr:+.2f} | {key}")

                if not ranked_long:
                    lines.append("(No combos with N≥20)")

                lines.append("")

            # Analyze SHORTS
            if shorts:
                short_analysis = analyze_combos(shorts, "SHORT")
                n_short = len(shorts)
                try:
                    w_short = sum(1 for *_, win, _ in shorts if isinstance(win,(int,float)) and win == 1)
                    wr_short = (w_short/n_short*100.0) if n_short else 0.0
                    evr_short = sum((rr if isinstance(rr,(int,float)) else 0.0) for *_, rr in shorts) / n_short if n_short else 0.0
                except Exception:
                    wr_short = 0.0; evr_short = 0.0

                lines.extend([
                    "═══════════════════════════",
                    f"🔴 SHORT TRADES (N={n_short})",
                    "═══════════════════════════",
                    "",
                    f"Baseline: WR {wr_short:.1f}% | EV_R {evr_short:+.2f}",
                    "",
                    "🔝 Top 10 Short Combos by EV_R (N≥20)"
                ])

                ranked_short = sorted([(k,v) for k,v in short_analysis['combos'].items() if v['n'] >= 20],
                                     key=lambda kv: (kv[1]['rr']/kv[1]['n']), reverse=True)[:10]

                for key, agg in ranked_short:
                    n = agg['n']; w = agg['w']
                    wr = (w/n*100.0) if n else 0.0
                    lo, hi = wilson_ci(w, n)
                    evr = (agg['rr']/n) if n else 0.0
                    lift_wr = wr - wr_short; lift_evr = evr - evr_short
                    lines.append(f"• WR {wr:5.1f}% [{lo:4.0f}-{hi:4.0f}] N={n:>3} | EV_R {evr:+.2f} | Lift_WR {lift_wr:+.1f} | Lift_EV {lift_evr:+.2f} | {key}")

                if not ranked_short:
                    lines.append("(No combos with N≥20)")

                lines.append("")

            lines.append(f"📊 Total: {len(items)} (L:{len(longs)} | S:{len(shorts)})")

            # Send
            MAX = 3500
            buf = []
            cur = 0
            for ln in lines:
                if cur + len(ln) + 1 > MAX and buf:
                    await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                    buf = []
                    cur = 0
                buf.append(ln)
                cur += len(ln) + 1
            if buf:
                await self.safe_reply(update, "\n".join(buf), parse_mode=None)

        except Exception as e:
            logger.error(f"Error in scalp_pro_analytics: {e}")
            try:
                await self.safe_reply(update, "Error computing Pro Analytics", parse_mode=None)
            except Exception:
                pass

    async def scalp_build_strategy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Build a candidate Scalp strategy (combos) from last 30d data.

        Uses enriched features (RSI/MACD/VWAP/Fib/MTF/Volume/BBW) to propose
        top N combinations by WR (N≥50). Outputs a YAML snippet you can paste
        into config under `scalp.exec.pro_exec` (execution path to be wired after approval).
        """
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker

            scpt = get_scalp_phantom_tracker()
            cutoff = datetime.utcnow() - timedelta(days=30)

            items = []  # (rsi, macd_hist, vwap, fib, mtf, vol, bbw, win)
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        f = getattr(p, 'features', {}) or {}
                        rsi = f.get('rsi_14'); mh = f.get('macd_hist')
                        vwap = f.get('vwap_dist_atr'); fibz = f.get('fib_zone')
                        mtf = f.get('mtf_agree_15'); vol = f.get('volume_ratio'); bbw = f.get('bb_width_pct')
                        if not (isinstance(rsi,(int,float)) and isinstance(mh,(int,float)) and isinstance(vwap,(int,float)) and isinstance(fibz,str) and isinstance(mtf,(bool,int)) and isinstance(vol,(int,float)) and isinstance(bbw,(int,float))):
                            continue
                        win = 1 if oc=='win' else 0
                        items.append((float(rsi), float(mh), float(vwap), fibz, bool(mtf), float(vol), float(bbw), win))
                    except Exception:
                        continue

            if not items:
                await self.safe_reply(update, "🧠 Build Strategy (30d)\n\nNo data yet.", parse_mode=None)
                return

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

            # Rank by WR (N≥50)
            ranked = sorted([(k,v) for k,v in combos.items() if v['n'] >= 50], key=lambda kv: (kv[1]['w']/kv[1]['n']), reverse=True)[:12]

            if not ranked:
                await self.safe_reply(update, "🧠 Build Strategy (30d)\n\nNo combinations with N≥50 yet.", parse_mode=None)
                return

            # Emit YAML snippet for pro_exec.combos
            lines = ["🧠 Build Strategy (30d)", "━━━━━━━━━━━━━━━━━━━━━━━━", "", "Paste into config.yaml under scalp.exec.pro_exec:", "", "scalp:", "  exec:", "    pro_exec:", "      combos:"]
            cid = 200
            for (rsi_name, macd_name, vwap_name, fib_name, mtf_ok, vol_name, bbw_name), agg in ranked:
                cid += 1
                rsi_bin = next(b for b in rsi_bins if b[0]==rsi_name)
                macd_val = next(b for b in macd_bins if b[0]==macd_name)[2]
                vwap_bin = next(b for b in vwap_bins if b[0]==vwap_name)
                vol_min = next(b for b in vol_bins if b[0]==vol_name)[2]
                bbw_max = next(b for b in bbw_bins if b[0]==bbw_name)[2]
                wr = (agg['w']/agg['n']*100.0)
                lines.append(f"        - id: {cid}")
                lines.append(f"          rsi_min: {rsi_bin[2]}")
                lines.append(f"          rsi_max: {rsi_bin[3]}")
                lines.append(f"          macd_hist: {macd_val}")
                lines.append(f"          vwap_min: {vwap_bin[2]}")
                lines.append(f"          vwap_max: {vwap_bin[3]}")
                lines.append(f"          fib_zone: \"{fib_name}\"")
                lines.append(f"          mtf_agree_15: {str(bool(mtf_ok)).lower()}")
                lines.append(f"          volume_ratio_min: {vol_min}")
                lines.append(f"          bbw_max: {bbw_max}")
                lines.append(f"          risk_percent: 1.0")
                lines.append(f"          enabled: true  # WR {wr:.1f}% (N={agg['n']})")

            MAX = 3500
            buf=[]; cur=0
            for ln in lines:
                if cur + len(ln) + 1 > MAX and buf:
                    await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                    buf=[]; cur=0
                buf.append(ln); cur += len(ln) + 1
            if buf:
                await self.safe_reply(update, "\n".join(buf), parse_mode=None)

        except Exception as e:
            logger.error(f"Error in scalp_build_strategy: {e}")
            try:
                await self.safe_reply(update, "Error building strategy from data", parse_mode=None)
            except Exception:
                pass
    async def scalp_open_phantoms(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """List all currently open Scalp phantoms (non-executed, no exit)."""
        try:
            from datetime import datetime
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            open_list = []
            for sym, lst in (getattr(scpt, 'active', {}) or {}).items():
                for p in (lst or []):
                    try:
                        if getattr(p, 'exit_time', None):
                            continue
                        if bool(getattr(p, 'was_executed', False)):
                            # Only show pure phantoms here
                            continue
                        st = getattr(p, 'signal_time', None)
                        age_min = 0
                        if st:
                            try:
                                age_min = int((datetime.utcnow() - st).total_seconds() // 60)
                            except Exception:
                                age_min = 0
                        feats = getattr(p, 'features', {}) or {}
                        ml = float(getattr(p, 'ml_score', 0.0) or 0.0)
                        open_list.append({
                            'symbol': p.symbol,
                            'side': p.side,
                            'entry': p.entry_price,
                            'tp': p.take_profit,
                            'sl': p.stop_loss,
                            'age_min': age_min,
                            'ml': ml,
                            'f_fast': feats.get('ema_slope_fast', None),
                            'f_slow': feats.get('ema_slope_slow', None),
                            'atr': feats.get('atr_pct', None),
                            'bbw': feats.get('bb_width_pct', None),
                            'vwap': feats.get('vwap_dist_atr', None)
                        })
                    except Exception:
                        continue

            if not open_list:
                await self.safe_reply(update, "👻 Scalp Open Phantoms\n\n(none)", parse_mode=None)
                return

            # Sort by age desc, then symbol
            open_list.sort(key=lambda x: (-x['age_min'], x['symbol']))

            MAX = 3500
            header = [f"👻 Scalp Open Phantoms ({len(open_list)})", "━━━━━━━━━━━━━━━━━━━━━━━━", ""]
            buf = header[:]
            cur = sum(len(s)+1 for s in buf)
            for o in open_list:
                try:
                    line = (
                        f"• {o['symbol']} {str(o['side']).upper()} @ {float(o['entry']):.4f} | TP {float(o['tp']):.4f} | SL {float(o['sl']):.4f} | age {o['age_min']}m"
                    )
                    # Append short feature snapshot if available
                    def _fmt(x,prec=3):
                        try:
                            return f"{float(x):.{prec}f}"
                        except Exception:
                            return "—"
                    line2 = (
                        f"    fast { _fmt(o['f_fast']) }% | slow { _fmt(o['f_slow']) }% | ATR { _fmt(o['atr']) }% | BBW { _fmt((o['bbw'] or 0.0)*100,2) }% | VWAP { _fmt(o['vwap']) }σ | ML {o['ml']:.1f}"
                    )
                    for ln in (line, line2):
                        if cur + len(ln) + 1 > MAX and buf:
                            await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                            buf = []
                            cur = 0
                        buf.append(ln)
                        cur += len(ln) + 1
                except Exception:
                    continue
            if buf:
                await self.safe_reply(update, "\n".join(buf), parse_mode=None)

        except Exception as e:
            logger.error(f"Error in scalp_open_phantoms: {e}")
            try:
                await self.safe_reply(update, "Error listing open Scalp phantoms", parse_mode=None)
            except Exception:
                pass

    async def scalp_ml_recommend(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """ML‑driven strategy recommendations from last 30d training data.

        Outputs top combos and a YAML snippet for pro_exec.combos.
        """
        try:
            from autobot.strategies.scalp.scorer import get_scalp_scorer
            scorer = get_scalp_scorer()
            res = scorer.recommend_strategies(days=30, min_n=50, top_n=12)
            if res.get('error'):
                await self.safe_reply(update, f"🧠 ML Strategy Recommend\n\nNo data yet ({res['error']})", parse_mode=None)
                return
            ranked = res.get('ranked', [])
            yaml_lines = res.get('yaml', [])
            lines = ["🧠 ML Strategy Recommendations (30d)", "━━━━━━━━━━━━━━━━━━━━━━━━", ""]
            for key, agg in ranked:
                wr = (agg['w']/agg['n']*100.0) if agg['n'] else 0.0
                lines.append(f"• WR {wr:5.1f}% (N={agg['n']:>3}) | {key}")
            lines.append("")
            lines.append("Paste into config.yaml:")
            lines.extend(yaml_lines)
            # Chunked send
            MAX=3500; buf=[]; cur=0
            for ln in lines:
                if cur + len(ln) + 1 > MAX and buf:
                    await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                    buf=[]; cur=0
                buf.append(ln); cur += len(ln) + 1
            if buf:
                await self.safe_reply(update, "\n".join(buf), parse_mode=None)
        except Exception as e:
            logger.error(f"Error in scalp_ml_recommend: {e}")
            try:
                await self.safe_reply(update, "Error computing ML recommendations", parse_mode=None)
            except Exception:
                pass

    async def scalp_ultimate(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Ultimate analysis: All 50+ variables, phantoms + executed trades"""
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()

            # Kickoff without Markdown to avoid parse noise in logs
            await self.safe_reply(update, "🔍 Running ultimate analysis (50+ variables)...", parse_mode=None)

            # Call comprehensive analysis (analyzes all variables, phantoms + executed)
            # Offload to thread to avoid blocking the event loop for large datasets
            try:
                import asyncio as _asyncio
                result = await _asyncio.to_thread(scpt.get_comprehensive_analysis)
            except Exception:
                # Fallback for older Python versions
                loop = _asyncio.get_running_loop()
                result = await loop.run_in_executor(None, scpt.get_comprehensive_analysis)

            if 'error' in result:
                await self.safe_reply(update, f"❌ {result['error']}", parse_mode=None)
                return

            # Format output
            msg = [
                "🎯 *ULTIMATE ANALYSIS* (30d)",
                "━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                f"📊 Analyzed: {result.get('total', 0)} trades (phantoms + executed)",
                f"✅ Overall WR: {result.get('overall_wr', 0):.1f}%",
                "",
                "🏆 *TOP 10 SOLO VARIABLES*",
                ""
            ]

            # Top 10 solo variables by delta impact
            solo_vars = result.get('solo_rankings', [])[:10]
            if solo_vars:
                for i, var in enumerate(solo_vars, 1):
                    name = var.get('var', 'unknown')
                    delta = var.get('delta', 0)
                    wr = var.get('wr', 0)
                    n = var.get('n', 0)
                    msg.append(f"{i}. {name}: ΔWR +{delta:.1f}% | WR {wr:.1f}% (N={n})")
            else:
                msg.append("_No solo variables found_")

            msg.extend(["", "🔗 *TOP 10 PAIRS*", ""])

            # Top 10 pairs by WR
            pairs = result.get('pair_rankings', [])[:10]
            if pairs:
                for i, pair in enumerate(pairs, 1):
                    combo = pair.get('combo', 'unknown')
                    wr = pair.get('wr', 0)
                    n = pair.get('n', 0)
                    msg.append(f"{i}. {combo}: WR {wr:.1f}% (N={n})")
            else:
                msg.append("_No pairs with N≥30 found_")

            msg.extend(["", "🎲 *TOP 5 TRIPLETS*", ""])

            # Top 5 triplets by WR
            triplets = result.get('triplet_rankings', [])[:5]
            if triplets:
                for i, trip in enumerate(triplets, 1):
                    combo = trip.get('combo', 'unknown')
                    wr = trip.get('wr', 0)
                    n = trip.get('n', 0)
                    msg.append(f"{i}. {combo}: WR {wr:.1f}% (N={n})")
            else:
                msg.append("_No triplets with N≥20 found_")

            msg.extend([
                "",
                "💡 *Use these combinations to improve signal filtering!*",
                f"📈 Analyzed: {len(solo_vars)} solos, {len(pairs)} pairs, {len(triplets)} triplets"
            ])

            # Join and send in chunks to avoid Telegram "message is too long" errors
            full = "\n".join(msg)
            MAX = 3500
            if len(full) <= MAX:
                await self.safe_reply(update, full, parse_mode=None)
            else:
                # Split on line boundaries
                buf = []
                cur = 0
                for line in msg:
                    if cur + len(line) + 1 > MAX and buf:
                        await self.safe_reply(update, "\n".join(buf), parse_mode=None)
                        buf = []
                        cur = 0
                    buf.append(line)
                    cur += len(line) + 1
                if buf:
                    await self.safe_reply(update, "\n".join(buf), parse_mode=None)

        except Exception as e:
            logger.error(f"scalp_ultimate error: {e}", exc_info=True)
            await self.safe_reply(update, f"❌ Error: {e}")

    async def scalp_combos_toggle(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Toggle combos-only execution mode for Scalp.

        Usage:
        /scalpcombos on  -> combos_only=true, block_noncombo=true
        /scalpcombos off -> combos_only=false, block_noncombo=false
        /scalpcombos     -> show current status
        """
        try:
            state: Optional[str] = None
            if ctx.args:
                state = str(ctx.args[0]).strip().lower()

            # Resolve in-memory config
            bot = self.shared.get('bot_instance')
            mem_cfg = bot.config if bot and hasattr(bot, 'config') else None

            def _get_exec(cfg: dict | None) -> dict:
                try:
                    return ((cfg.get('scalp', {}) or {}).get('exec', {}) or {}) if isinstance(cfg, dict) else {}
                except Exception:
                    return {}

            cur = _get_exec(mem_cfg)
            cur_file = {}
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    y = yaml.safe_load(f) or {}
                cur_file = _get_exec(y)
            except Exception:
                cur_file = {}

            if not state:
                combos_only = bool(cur.get('combos_only', cur_file.get('combos_only', False)))
                block_noncombo = bool(cur.get('block_noncombo', cur_file.get('block_noncombo', False)))
                await self.safe_reply(update, f"🎯 Scalp combos-only status\n• combos_only: {combos_only}\n• block_noncombo: {block_noncombo}")
                return

            if state not in ('on', 'off'):
                await self.safe_reply(update, "Usage: /scalpcombos on|off")
                return

            enable = (state == 'on')

            # Update in-memory
            try:
                if isinstance(mem_cfg, dict):
                    mem_cfg.setdefault('scalp', {}).setdefault('exec', {})['combos_only'] = enable
                    mem_cfg['scalp']['exec']['block_noncombo'] = enable
            except Exception:
                pass

            # Persist to config.yaml
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    y = yaml.safe_load(f) or {}
                y.setdefault('scalp', {}).setdefault('exec', {})['combos_only'] = enable
                y['scalp']['exec']['block_noncombo'] = enable
                with open('config.yaml', 'w') as f:
                    yaml.dump(y, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                logger.error(f"combos_only persist failed: {e}")

            await self.safe_reply(update, f"✅ Scalp combos-only set to {enable} (block_noncombo={enable})")
        except Exception as e:
            logger.error(f"Error in scalp_combos_toggle: {e}")
            try:
                await self.safe_reply(update, f"❌ Error: {e}")
            except Exception:
                pass

    async def scalp_combos_mute(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Toggle muting for disabled paths when combos_only is active.

        Usage:
        /scalpcombosmute on|off  (no args -> show status)
        """
        try:
            state: Optional[str] = None
            if ctx.args:
                state = str(ctx.args[0]).strip().lower()

            # Resolve in-memory config
            bot = self.shared.get('bot_instance')
            mem_cfg = bot.config if bot and hasattr(bot, 'config') else None

            def _get_exec(cfg: dict | None) -> dict:
                try:
                    return ((cfg.get('scalp', {}) or {}).get('exec', {}) or {}) if isinstance(cfg, dict) else {}
                except Exception:
                    return {}

            cur = _get_exec(mem_cfg)
            cur_file = {}
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    y = yaml.safe_load(f) or {}
                cur_file = _get_exec(y)
            except Exception:
                cur_file = {}

            if not state:
                mute = bool(cur.get('mute_disabled_paths', cur_file.get('mute_disabled_paths', False)))
                await self.safe_reply(update, f"🔕 combos_only mute_disabled_paths: {mute}\nUsage: /scalpcombosmute on|off")
                return

            if state not in ('on', 'off'):
                await self.safe_reply(update, "Usage: /scalpcombosmute on|off")
                return

            enable = (state == 'on')

            # Update in-memory
            try:
                if isinstance(mem_cfg, dict):
                    mem_cfg.setdefault('scalp', {}).setdefault('exec', {})['mute_disabled_paths'] = enable
            except Exception:
                pass

            # Persist to config.yaml
            try:
                import yaml
                with open('config.yaml', 'r') as f:
                    y = yaml.safe_load(f) or {}
                y.setdefault('scalp', {}).setdefault('exec', {})['mute_disabled_paths'] = enable
                with open('config.yaml', 'w') as f:
                    yaml.dump(y, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                logger.error(f"mute_disabled_paths persist failed: {e}")

            await self.safe_reply(update, f"✅ combos_only mute_disabled_paths set to {enable}")
        except Exception as e:
            logger.error(f"Error in scalp_combos_mute: {e}")
            try:
                await self.safe_reply(update, f"❌ Error: {e}")
            except Exception:
                pass

    async def combo_status_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current adaptive combo filter status.

        Displays:
        - Enabled/disabled combo counts
        - Threshold settings
        - Last update time
        - Recent combo changes (promotions/demotions)
        - Top 10 enabled combos by WR
        """
        try:
            mgr = self.shared.get('adaptive_combo_mgr')
            if not mgr:
                await self.safe_reply(update, "❌ Adaptive combo manager not available")
                return

            stats = mgr.get_stats_summary()

            thr_l = float(stats.get('min_wr_threshold_long', stats.get('min_wr_threshold', 0.0)) or 0.0)
            thr_s = float(stats.get('min_wr_threshold_short', stats.get('min_wr_threshold', 0.0)) or 0.0)

            lines = [
                "📊 *Adaptive Combo Filter Status*",
                "",
                f"Status: {'🟢 ENABLED' if stats['enabled'] else '🔴 DISABLED'}",
                f"Thresholds: Long LB WR ≥{thr_l:.1f}%, Short LB WR ≥{thr_s:.1f}%, N ≥{stats['min_sample_size']}",
                f"Lookback: {stats['lookback_days']} days",
                "",
                "*Current State*",
                f"🟢 Longs: {stats['long_enabled']} enabled | {stats['long_disabled']} disabled",
                f"🔴 Shorts: {stats['short_enabled']} enabled | {stats['short_disabled']} disabled",
                f"Total: {stats['total_combos']} combos tracked",
                "",
                f"Last Update: {stats.get('last_update', 'Never')}",
                f"Total Updates: {stats['update_count']}",
            ]

            # 30d aggregate totals (exec vs phantom) per side
            try:
                lt = stats.get('long_totals', {}) or {}
                st = stats.get('short_totals', {}) or {}
                if (lt.get('n', 0) or 0) > 0 or (st.get('n', 0) or 0) > 0:
                    lines.append("")
                    lines.append("📈 *30d Combo Totals*")
                    if lt.get('n', 0):
                        lines.append(
                            f"🟢 Longs: N={lt.get('n',0)} "
                            f"(Exec {lt.get('n_exec',0)}, Phantom {lt.get('n_phantom',0)}) "
                            f"| 24h: +{lt.get('n_24h',0)} "
                            f"(Exec {lt.get('n_exec_24h',0)}, Phantom {lt.get('n_phantom_24h',0)})"
                        )
                    if st.get('n', 0):
                        lines.append(
                            f"🔴 Shorts: N={st.get('n',0)} "
                            f"(Exec {st.get('n_exec',0)}, Phantom {st.get('n_phantom',0)}) "
                            f"| 24h: +{st.get('n_24h',0)} "
                            f"(Exec {st.get('n_exec_24h',0)}, Phantom {st.get('n_phantom_24h',0)})"
                        )
            except Exception:
                pass

            if stats.get('recent_changes'):
                lines.append("")
                lines.append("*Recent Changes (last 10)*")
                for change in stats['recent_changes']:
                    lines.append(f"• {change}")

            # Show active combos with exec/phantom breakdown
            active_longs = mgr.get_active_combos('long')
            active_shorts = mgr.get_active_combos('short')
            if active_longs or active_shorts:
                lines.append("")
                lines.append("🎯 *Active Combos*")

                def _format_active_side(combos, header, emoji):
                    if not combos:
                        return
                    # Sort by WR then N
                    combos_sorted = sorted(
                        combos,
                        key=lambda c: (float(c.get('wr', 0.0) or 0.0), int(c.get('n', 0) or 0)),
                        reverse=True
                    )[:10]
                    lines.append(f"{emoji} {header} ({len(combos)} enabled)")
                    for idx, c in enumerate(combos_sorted, start=1):
                        try:
                            wr = float(c.get('wr', 0.0) or 0.0)
                            n = int(c.get('n', 0) or 0)
                            ev = c.get('ev_r', None)
                            ev_str = ""
                            if ev is not None:
                                try:
                                    ev_str = f", EV_R {float(ev):+.2f}R"
                                except Exception:
                                    ev_str = ""
                            n_exec = int(c.get('n_exec', 0) or 0)
                            n_ph = int(c.get('n_phantom', 0) or 0)
                            n_24 = int(c.get('n_24h', 0) or 0)
                            n_exec_24 = int(c.get('n_exec_24h', 0) or 0)
                            n_ph_24 = int(c.get('n_phantom_24h', 0) or 0)
                            lines.append(
                                f"{idx}) {emoji} WR {wr:.1f}% (N={n}{ev_str})"
                            )
                            lines.append(
                                f"   Exec {n_exec}, Phantom {n_ph} | 24h: +{n_24} "
                                f"(Exec {n_exec_24}, Phantom {n_ph_24})"
                            )
                            lines.append(f"   {c.get('combo_id','')}")
                        except Exception:
                            continue

                if active_longs:
                    _format_active_side(active_longs, "*Longs*", "🟢")
                if active_shorts:
                    lines.append("")
                    _format_active_side(active_shorts, "*Shorts*", "🔴")

            # Show disabled combos + top-per-side combos if requested
            if ctx.args and 'all' in ctx.args:
                lines.append("")
                lines.append("*Top Combos by Win Rate*")

                # Load per-side state to avoid cross-side overwrites
                try:
                    long_state = mgr._load_combo_state('long')
                except Exception:
                    long_state = mgr._load_combo_state()
                    long_state = {k: v for k, v in long_state.items() if v.get('side') == 'long'}
                try:
                    short_state = mgr._load_combo_state('short')
                except Exception:
                    short_state = mgr._load_combo_state()
                    short_state = {k: v for k, v in short_state.items() if v.get('side') == 'short'}

                def _build_top_wr_side(state: dict, side_label: str) -> list[str]:
                    side_lines: list[str] = []
                    combos = []
                    for data in (state or {}).values():
                        try:
                            wr = float(data.get('wr', 0.0) or 0.0)
                            n = int(data.get('n', 0) or 0)
                        except Exception:
                            continue
                        # Require a minimum sample size so small-N 100% WR patterns
                        # don't dominate the ranking (focus on proven combos).
                        if n < 30:
                            continue
                        combos.append({
                            'combo_id': data.get('combo_id') or '',
                            'wr': wr,
                            'n': n,
                            'enabled': bool(data.get('enabled', False)),
                            'ev_r': data.get('ev_r'),
                        })

                    if not combos:
                        side_lines.append(f"{side_label} No combos tracked.")
                        return side_lines

                    # Sort by WR descending, then by sample size N descending
                    top_wr = sorted(
                        combos,
                        key=lambda c: (c['wr'], c['n']),
                        reverse=True
                    )[:10]

                    side_lines.append(side_label)
                    for i, c in enumerate(top_wr, start=1):
                        status_emoji = "🟢" if c.get('enabled') else "🔴"
                        ev = c.get('ev_r')
                        ev_line = ""
                        if ev is not None:
                            try:
                                ev_line = f", EV {float(ev):+.2f}R"
                            except Exception:
                                ev_line = ""
                        side_lines.append(
                            f"{i}) {status_emoji} WR {c['wr']:.1f}% (N={c['n']}{ev_line})"
                        )
                        side_lines.append(f"   {c['combo_id']}")
                    return side_lines

                # Longs
                lines.extend(_build_top_wr_side(long_state, "🟢 *Top Long Combos (Top 10 by WR)*"))
                lines.append("")
                # Shorts
                lines.extend(_build_top_wr_side(short_state, "🔴 *Top Short Combos (Top 10 by WR)*"))

                # Preserve original disabled-by-N view for reference
                lines.append("")
                lines.append("*Disabled Combos (Top 10 by N)*")
                all_state = mgr._load_combo_state()
                disabled_list = [v for v in all_state.values() if not v.get('enabled')]
                disabled_sorted = sorted(disabled_list, key=lambda x: x.get('n', 0), reverse=True)[:10]
                for combo in disabled_sorted:
                    side_emoji = "🟢" if combo.get('side') == 'long' else "🔴"
                    wr = float(combo.get('wr', 0.0))
                    n = int(combo.get('n', 0))
                    lines.append(f"{side_emoji} WR {wr:.1f}% (N={n})")
                    lines.append(f"   {combo['combo_id']}")
            else:
                lines.append("")
                lines.append("No active combos. Use `/combo_status all` to see disabled ones.")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.error(f"Error in combo_status_cmd: {e}", exc_info=True)
            await self.safe_reply(update, f"❌ Error: {e}")

    async def force_adaptive_combos(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force enable adaptive combos with custom N and disable fallback.

        Usage: /force_adaptive [min_n]
        Example: /force_adaptive 5  -> Sets min_sample_size=5, fallback='off', updates filters.
        """
        try:
            mgr = self.shared.get('adaptive_combo_mgr')
            if not mgr:
                await self.safe_reply(update, "❌ Adaptive combo manager not available")
                return

            # Parse N (default to 5 if not provided)
            target_n = 5
            if ctx.args:
                try:
                    target_n = int(ctx.args[0])
                except ValueError:
                    pass
            
            # Update Manager Settings
            old_n = mgr.min_sample_size
            mgr.min_sample_size = target_n
            
            # Update Config (Runtime)
            cfg = self.shared.get('config', {})
            if 'scalp' not in cfg: cfg['scalp'] = {}
            if 'exec' not in cfg['scalp']: cfg['scalp']['exec'] = {}
            
            old_fallback = cfg['scalp']['exec'].get('fallback_until_ready', 'pro')
            cfg['scalp']['exec']['fallback_until_ready'] = 'off'
            
            # Update adaptive config in memory
            if 'adaptive_combos' not in cfg['scalp']['exec']: cfg['scalp']['exec']['adaptive_combos'] = {}
            cfg['scalp']['exec']['adaptive_combos']['min_sample_size'] = target_n
            
            # Force Update
            enabled, disabled, changes = mgr.update_combo_filters(force=True)
            
            msg = (
                f"✅ *Adaptive Combos Forced*\n\n"
                f"• Min Sample Size: {old_n} → {target_n}\n"
                f"• Fallback Mode: {old_fallback} → off\n"
                f"• Filter Update: {enabled} enabled, {disabled} disabled\n\n"
                f"⚠️ *Warning*: This overrides safety defaults. Monitor closely.\n"
                f"Use `/combo_threshold` to adjust WR if needed."
            )
            await self.safe_reply(update, msg)
            logger.info(f"Force adaptive combos: N={target_n}, fallback=off, enabled={enabled}")

        except Exception as e:
            logger.error(f"Error in force_adaptive_combos: {e}", exc_info=True)
            await self.safe_reply(update, f"❌ Error: {e}")

    async def combo_refresh_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force immediate recalculation of adaptive combo filters.

        Triggers update and reports changes.
        """
        try:
            mgr = self.shared.get('adaptive_combo_mgr')
            if not mgr:
                await self.safe_reply(update, "❌ Adaptive combo manager not available")
                return

            if not mgr.enabled:
                await self.safe_reply(update, "⚠️ Adaptive combo filtering is disabled in config")
                return

            await self.safe_reply(update, "🔄 Refreshing combo filters...")

            enabled_count, disabled_count, changes = mgr.update_combo_filters(force=True)

            lines = [
                "✅ *Combo Filter Refresh Complete*",
                "",
                f"🟢 Enabled: {enabled_count}",
                f"🔴 Disabled: {disabled_count}",
                f"Changes: {len(changes)}",
            ]

            if changes:
                lines.append("")
                lines.append("*Changes*")
                for change in changes:
                    lines.append(f"• {change}")
            else:
                lines.append("")
                lines.append("_No changes from previous state_")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.error(f"Error in combo_refresh_cmd: {e}", exc_info=True)
            await self.safe_reply(update, f"❌ Error: {e}")

    async def combo_threshold_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Adjust the LB WR thresholds for combo filtering.

        Usage:
        /combo_threshold              - Show current thresholds
        /combo_threshold X            - Set both long/short LB WR to X (e.g., 40)
        /combo_threshold L S          - Set long LB WR=L and short LB WR=S
        """
        try:
            mgr = self.shared.get('adaptive_combo_mgr')
            if not mgr:
                await self.safe_reply(update, "❌ Adaptive combo manager not available")
                return

            if not ctx.args:
                # Show current thresholds
                thr_l = getattr(mgr, 'min_wr_threshold_long', getattr(mgr, 'min_wr_threshold', 0.0))
                thr_s = getattr(mgr, 'min_wr_threshold_short', getattr(mgr, 'min_wr_threshold', 0.0))
                await self.safe_reply(update,
                    f"📊 Current Threshold\n"
                    f"Long LB WR ≥{thr_l:.1f}%\n"
                    f"Short LB WR ≥{thr_s:.1f}%\n"
                    f"N ≥{mgr.min_sample_size}\n\n"
                    f"Gating uses Wilson lower-bound WR.\n"
                    f"Usage: /combo_threshold <L> [S]"
                )
                return

            # Parse thresholds
            args = ctx.args
            try:
                if len(args) == 1:
                    thr_long = float(args[0])
                    thr_short = float(args[0])
                else:
                    thr_long = float(args[0])
                    thr_short = float(args[1])
            except ValueError:
                await self.safe_reply(update, "❌ Invalid threshold(s). Must be numbers (e.g., 45 or 45 30)")
                return

            if not (0.0 <= thr_long <= 100.0 and 0.0 <= thr_short <= 100.0):
                await self.safe_reply(update, "❌ Thresholds must be between 0 and 100")
                return

            old_long = getattr(mgr, 'min_wr_threshold_long', getattr(mgr, 'min_wr_threshold', 0.0))
            old_short = getattr(mgr, 'min_wr_threshold_short', getattr(mgr, 'min_wr_threshold', 0.0))

            # Update manager thresholds (side-specific) and keep global as fallback
            mgr.min_wr_threshold_long = thr_long
            mgr.min_wr_threshold_short = thr_short
            try:
                mgr.min_wr_threshold = min(thr_long, thr_short)
            except Exception:
                pass

            # Update runtime config snapshot for consistency (non-persistent)
            cfg = self.shared.get('config', {}) or {}
            acfg = (((cfg.setdefault('scalp', {}).setdefault('exec', {})).setdefault('adaptive_combos', {})))
            acfg['min_wr_threshold_long'] = thr_long
            acfg['min_wr_threshold_short'] = thr_short
            acfg['use_wilson_lb'] = True

            # Trigger refresh with new thresholds
            enabled_count, disabled_count, changes = mgr.update_combo_filters(force=True)

            await self.safe_reply(update,
                f"✅ *Threshold Updated*\n\n"
                f"Old Long: {old_long:.1f}% → New Long: {thr_long:.1f}%\n"
                f"Old Short: {old_short:.1f}% → New Short: {thr_short:.1f}%\n\n"
                f"🟢 Enabled: {enabled_count}\n"
                f"🔴 Disabled: {disabled_count}\n"
                f"Changes: {len(changes)}\n\n"
                f"_Note: This is a runtime change only. Update config.yaml to persist._"
            )

        except Exception as e:
            logger.error(f"Error in combo_threshold_cmd: {e}", exc_info=True)
            await self.safe_reply(update, f"❌ Error: {e}")

    async def scalp_risk_state(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            cfg = self.shared.get('config', {}) or {}
            ar = (((cfg.get('scalp', {}) or {}).get('exec', {}) or {}).get('adaptive_risk', {}) or {})
            enabled = bool(ar.get('enabled', True))
            base = float(ar.get('base_percent', 1.0))
            rmin = float(ar.get('min_percent', 0.5))
            rmax = float(ar.get('max_percent', 3.0))
            min_n = int(ar.get('min_samples', 30))
            ev_floor = float(ar.get('ev_floor_r', 0.0))
            ladder = ar.get('wr_lb_ladder') or {'45':1.0,'55':1.5,'65':2.0,'70':2.5}
            lines = [
                "⚙️ Adaptive Risk (Scalp)",
                f"• enabled: {enabled}",
                f"• base/min/max: {base:.2f}% / {rmin:.2f}% / {rmax:.2f}%",
                f"• min_samples: {min_n}",
                f"• ev_floor_r: {ev_floor:+.2f}",
                f"• ladder: {ladder}",
            ]
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in scalp_risk_state: {e}")
            await self.safe_reply(update, f"❌ Error: {e}")

    async def scalp_risk_limits(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            if len(args) != 3:
                await self.safe_reply(update, "Usage: /scalprisklimits <base> <min> <max>")
                return
            base, rmin, rmax = map(float, args)
            cfg = self.shared.get('config', {}) or {}
            mem = cfg.setdefault('scalp', {}).setdefault('exec', {}).setdefault('adaptive_risk', {})
            mem['enabled'] = True
            mem['base_percent'] = base
            mem['min_percent'] = rmin
            mem['max_percent'] = rmax
            await self.safe_reply(update, f"✅ Adaptive risk limits set: base {base:.2f}%, min {rmin:.2f}%, max {rmax:.2f}% (runtime only)")
        except Exception as e:
            logger.error(f"Error in scalp_risk_limits: {e}")
            await self.safe_reply(update, f"❌ Error: {e}")

    async def scalp_risk_ladder(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            if not args:
                await self.safe_reply(update, "Usage: /scalpriskladder <wr:mult,wr:mult,...>  e.g., 45:1.0,55:1.5,65:2.0,70:2.5")
                return
            spec = args[0]
            parts = [p.strip() for p in spec.split(',') if p.strip()]
            ladder = {}
            for p in parts:
                try:
                    k, v = p.split(':', 1)
                    ladder[str(float(k))] = float(v)
                except Exception:
                    continue
            if not ladder:
                await self.safe_reply(update, "❌ Invalid ladder format")
                return
            cfg = self.shared.get('config', {}) or {}
            mem = cfg.setdefault('scalp', {}).setdefault('exec', {}).setdefault('adaptive_risk', {})
            mem['enabled'] = True
            mem['wr_lb_ladder'] = ladder
            await self.safe_reply(update, f"✅ Ladder set: {ladder} (runtime only)")
        except Exception as e:
            logger.error(f"Error in scalp_risk_ladder: {e}")
            await self.safe_reply(update, f"❌ Error: {e}")

    async def exec_winrates(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, days_sessions: int = 30):
        """Show execution-only win rates: Today, Yesterday, 7-day daily, and 30d sessions (asian/european/us).

        Uses exit_time for bucketing and counts only decisive closed trades.
        """
        try:
            tracker = self.shared.get("trade_tracker")
            trades = []

            # Prefer DB for freshness when available, else fallback to memory list
            if tracker and getattr(tracker, 'use_db', False) and getattr(tracker, 'conn', None):
                try:
                    from datetime import timedelta
                    cutoff = datetime.utcnow() - timedelta(days=60)
                    rows = []
                    with tracker.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT symbol, exit_time, pnl_usd
                            FROM trades
                            WHERE exit_time >= %s
                            ORDER BY exit_time DESC
                            """,
                            (cutoff,)
                        )
                        rows = cur.fetchall()
                    class _Row:
                        def __init__(self, sym, et, pnl):
                            self.symbol = sym
                            # Ensure exit_time is timezone-naive UTC
                            if et and hasattr(et, 'tzinfo') and et.tzinfo is not None:
                                self.exit_time = et.replace(tzinfo=None)
                            else:
                                self.exit_time = et
                            self.pnl_usd = float(pnl)
                    trades = [_Row(r[0], r[1], r[2]) for r in rows]
                except Exception as _db_e:
                    logger.debug(f"Exec WR DB query failed, falling back to memory: {_db_e}")
                    trades = getattr(tracker, 'trades', []) if tracker else []
            else:
                trades = getattr(tracker, 'trades', []) if tracker else []

            if not trades:
                await self.safe_reply(update, "📈 *Execution WR*\n\n_No executed trades yet_")
                return

            from datetime import datetime, timedelta
            now = datetime.utcnow()

            def _is_win(t):
                try:
                    return float(getattr(t, 'pnl_usd', 0.0) or 0.0) > 0.0
                except Exception:
                    return False

            def _normalize_exit_time(t):
                """Normalize exit_time to timezone-naive UTC datetime"""
                et = getattr(t, 'exit_time', None)
                if not et:
                    return None
                if hasattr(et, 'tzinfo') and et.tzinfo is not None:
                    # Convert timezone-aware to UTC and remove timezone
                    return et.replace(tzinfo=None) if et.tzinfo.utcoffset(et) == timedelta(0) else et.astimezone(None).replace(tzinfo=None)
                return et

            # Filter executed trades with valid exit_time and normalize timezone
            execd = []
            for t in trades:
                et = _normalize_exit_time(t)
                if et:
                    # Store normalized exit_time back on the object for consistent access
                    t.exit_time = et
                    execd.append(t)

            # Diagnostic logging
            logger.info(f"[exec_winrates] Total trades: {len(trades)}, With valid exit_time: {len(execd)}")
            if len(trades) > 0 and len(execd) == 0:
                logger.warning(f"[exec_winrates] All {len(trades)} trades missing or invalid exit_time!")

            if len(execd) > 0:
                # Log sample exit times for debugging
                sample_times = [t.exit_time.isoformat() for t in execd[:3]]
                logger.info(f"[exec_winrates] Sample exit_times (UTC): {sample_times}")
                logger.info(f"[exec_winrates] Current UTC time: {now.isoformat()}")

            # Daily WR calculator with robust date filtering
            def _day_wr(target_date):
                """Calculate WR for a specific date (timezone-naive UTC)"""
                # Create UTC datetime range for the entire day
                day_start = datetime.combine(target_date, datetime.min.time())
                # Use timedelta to get the start of next day, then subtract 1 microsecond
                day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)

                # Filter trades within this day
                arr = []
                for t in execd:
                    et = t.exit_time
                    if day_start <= et <= day_end:
                        arr.append(t)

                n = len(arr)
                w = sum(1 for t in arr if _is_win(t))
                wr = (w / n * 100.0) if n else 0.0
                return wr, n, w

            # Today / Yesterday
            today = now.date()
            yday = (now - timedelta(days=1)).date()
            t_wr, t_n, t_w = _day_wr(today)
            y_wr, y_n, y_w = _day_wr(yday)

            # Diagnostic logging for today's trades
            if t_n == 0 and len(execd) > 0:
                recent_dates = sorted(set([t.exit_time.date().isoformat() for t in execd]), reverse=True)[:7]
                logger.warning(f"[exec_winrates] Today {today.isoformat()} has N=0 but {len(execd)} executed trades exist. Recent dates: {recent_dates}")

            # Last 7 days breakdown
            daily_lines = []
            day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            for i in range(7):
                d = (now - timedelta(days=i)).date()
                wr, n, w = _day_wr(d)
                name = day_names[d.weekday()]
                low = " (low N)" if 0 < n < 5 else ""
                daily_lines.append(f"• {name} {d.isoformat()}: WR {wr:.1f}% (N={n}){low}")
            daily_lines = list(reversed(daily_lines))

            # Aggregate WR windows
            def _agg_wr(days: int) -> tuple:
                cutoff = now - timedelta(days=days)
                arr = [t for t in execd if t.exit_time >= cutoff]
                n = len(arr)
                w = sum(1 for t in arr if _is_win(t))
                wr = (w / n * 100.0) if n else 0.0
                return wr, n, w

            wr30, n30, w30 = _agg_wr(30)
            wr60, n60, w60 = _agg_wr(60)

            # Sessions (30d) - Asian: 0-8, European: 8-16, US: 16-24 UTC
            cutoff = now - timedelta(days=days_sessions)
            sess_map = {'asian': {'w':0,'n':0}, 'european': {'w':0,'n':0}, 'us': {'w':0,'n':0}}

            def _session(dt: datetime) -> str:
                h = dt.hour
                if 0 <= h < 8:
                    return 'asian'
                elif 8 <= h < 16:
                    return 'european'
                else:
                    return 'us'

            for t in execd:
                try:
                    et = t.exit_time
                    if et < cutoff:
                        continue
                    s = _session(et)
                    sess_map[s]['n'] += 1
                    if _is_win(t):
                        sess_map[s]['w'] += 1
                except Exception as sess_err:
                    logger.debug(f"Session calculation error: {sess_err}")
                    continue

            # Build output message
            lines = [
                "📈 *Execution WR*",
                "",
                f"Today: WR {t_wr:.1f}% (N={t_n})",
                f"Yesterday: WR {y_wr:.1f}% (N={y_n})",
                "",
                "🗓 *Last 7 days*",
                *daily_lines,
                "",
                f"📆 Last 30d: WR {wr30:.1f}% (N={n30})",
                f"📆 Last 60d: WR {wr60:.1f}% (N={n60})",
                "",
                f"🕘 *Sessions ({days_sessions}d)*",
            ]

            for k in ['asian','european','us']:
                s = sess_map[k]
                wr = (s['w']/s['n']*100.0) if s['n'] else 0.0
                low = " (low N)" if 0 < s['n'] < 10 else ""
                lines.append(f"• {k}: WR {wr:.1f}% (N={s['n']}){low}")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.error(f"Error in exec_winrates: {e}", exc_info=True)
            await self.safe_reply(update, "❌ Error computing execution win rates")

    async def scalp_time_vars_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Command wrapper: /scalptimewrvars <sessions|session|days|day> <key>

        Examples:
          /scalptimewrvars sessions asian
          /scalptimewrvars days fri
        """
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            if not args or len(args) < 2:
                await self.safe_reply(update, "Usage: /scalptimewrvars <sessions|days> <asian|european|us|off_hours|mon..sun>")
                return
            kind = args[0].strip().lower()
            key = args[1].strip().lower()
            if kind in ('sessions','session'):
                await self._scalp_time_vars(update, ctx, kind='session', key=key)
                return
            if kind in ('days','day'):
                await self._scalp_time_vars(update, ctx, kind='day', key=key)
                return
            await self.safe_reply(update, "First arg must be 'sessions' or 'days'")
        except Exception as e:
            logger.error(f"Error in scalp_time_vars_cmd: {e}")
            await update.message.reply_text("Error computing Vars-by-time view")

    async def _scalp_time_vars(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, *, kind: str, key: str, days: int = 30, min_samples: int = 20):
        """Compute Scalp variable/gate WR for a subset filtered by session or day.

        kind: 'session' or 'day'
        key:  session in {'asian','european','us','off_hours'} or day in {'mon'..'sun'|'monday'..}
        """
        try:
            from datetime import datetime, timedelta
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()

            cutoff = datetime.utcnow() - timedelta(days=days)

            def _session_fallback(dt: datetime) -> str:
                hr = dt.hour if dt else 0
                if 0 <= hr < 8: return 'asian'
                if 8 <= hr < 16: return 'european'
                return 'us'

            # Normalize key
            k = key.strip().lower()
            dow_map = {
                'mon': 0, 'monday': 0,
                'tue': 1, 'tuesday': 1,
                'wed': 2, 'wednesday': 2,
                'thu': 3, 'thursday': 3,
                'fri': 4, 'friday': 4,
                'sat': 5, 'saturday': 5,
                'sun': 6, 'sunday': 6,
            }

            # Collect filtered decisive phantoms
            phantoms = []
            for arr in (getattr(scpt, 'completed', {}) or {}).values():
                for p in arr:
                    try:
                        et = getattr(p, 'exit_time', None)
                        if not et or et < cutoff:
                            continue
                        oc = getattr(p, 'outcome', None)
                        if oc not in ('win','loss'):
                            continue
                        if kind == 'session':
                            feats = getattr(p, 'features', {}) or {}
                            sess = str(feats.get('session')) if feats.get('session') else _session_fallback(et)
                            if sess != k:
                                continue
                        elif kind == 'day':
                            didx = dow_map.get(k)
                            if didx is None:
                                continue
                            if et.weekday() != didx:
                                continue
                        else:
                            continue
                        phantoms.append(p)
                    except Exception:
                        continue

            total = len(phantoms)
            if total == 0:
                await self.safe_reply(update, f"🚪 *Scalp Variable Analysis* ({days}d)\nNo data for {kind}={key}.")
                return
            wins = sum(1 for p in phantoms if getattr(p, 'outcome', None) == 'win')
            baseline_wr = (wins / total * 100.0) if total else 0.0

            # Compute gate statuses once
            gate_statuses = [scpt.compute_gate_status(p) for p in phantoms]

            # Variables list (align with tracker)
            all_variables = [
                'htf', 'vol', 'body', 'align_15m',
                'body_040', 'body_045', 'body_050', 'body_060', 'wick_align',
                'vwap_045', 'vwap_060', 'vwap_080', 'vwap_100',
                'vol_110', 'vol_120', 'vol_150',
                'bb_width_60p', 'bb_width_70p', 'bb_width_80p',
                'q_040', 'q_050', 'q_060', 'q_070',
                'impulse_040', 'impulse_060',
                'micro_seq',
                'htf_070', 'htf_080',
            ]

            variable_stats = {}
            for var in all_variables:
                var_pass_idx = [i for i, gs in enumerate(gate_statuses) if bool(gs.get(var, False))]
                var_fail_idx = [i for i, gs in enumerate(gate_statuses) if not bool(gs.get(var, False))]

                pass_total = len(var_pass_idx)
                fail_total = len(var_fail_idx)
                pass_wins = sum(1 for i in var_pass_idx if getattr(phantoms[i], 'outcome', None) == 'win')
                fail_wins = sum(1 for i in var_fail_idx if getattr(phantoms[i], 'outcome', None) == 'win')
                pass_wr = (pass_wins / pass_total * 100.0) if pass_total else 0.0
                fail_wr = (fail_wins / fail_total * 100.0) if fail_total else 0.0
                # Delta vs fail_wr (consistent with get_gate_analysis)
                delta = (pass_wr - fail_wr) if (pass_total >= min_samples and fail_total >= min_samples) else None
                variable_stats[var] = {
                    'pass_wins': pass_wins,
                    'pass_total': pass_total,
                    'pass_wr': pass_wr,
                    'fail_wins': fail_wins,
                    'fail_total': fail_total,
                    'fail_wr': fail_wr,
                    'delta': delta,
                    'sufficient_samples': (pass_total >= min_samples and fail_total >= min_samples)
                }

            # Ranked variables by delta
            sorted_variables = sorted(
                [(k, v) for k, v in variable_stats.items() if v['sufficient_samples']],
                key=lambda x: x[1]['delta'], reverse=True
            )

            # Combinations for original 4 gates (HVBA)
            gates = ['htf', 'vol', 'body', 'align_15m']
            from collections import defaultdict
            combo_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
            for i, p in enumerate(phantoms):
                gs = gate_statuses[i]
                bitmap = ''.join(['1' if gs.get(g, False) else '0' for g in gates])
                combo_stats[bitmap]['total'] += 1
                if getattr(p, 'outcome', None) == 'win':
                    combo_stats[bitmap]['wins'] += 1
            sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]['total'], reverse=True)

            def _pretty(v: str) -> str:
                try:
                    return v.replace('bb_width_', 'bbwidth').replace('_', '')
                except Exception:
                    return v

            # Build message
            scope = f"Session: {key}" if kind == 'session' else f"Day: {key.title()}"
            msg = [
                f"🚪 *Scalp Variable Analysis* ({days}d)",
                f"📊 Baseline: {total} phantoms, {baseline_wr:.1f}% WR",
                "",
            ]
            # Top and Worst filters
            if sorted_variables:
                positive = [(k, v) for k, v in sorted_variables if v['delta'] and v['delta'] > 0]
                negative = [(k, v) for k, v in sorted_variables if v['delta'] and v['delta'] < 0]
                if positive:
                    msg.append("🟢 Top Filters (Improve WR):")
                    for var_name, stats in positive[:10]:
                        msg.append(
                            f"✅ {_pretty(var_name)}: {stats['pass_total']} trades, {stats['pass_wr']:.1f}% WR ({stats['delta']:+.1f}%)"
                        )
                    msg.append("")
                if negative:
                    msg.append("*🔴 Worst Filters (Hurt WR):*")
                    # Show worst 10
                    for var_name, stats in list(reversed(negative[-10:])):
                        msg.append(
                            f"❌ {_pretty(var_name)}: {stats['pass_total']} trades, {stats['pass_wr']:.1f}% WR ({stats['delta']:+.1f}%)"
                        )
                    msg.append("")
                # Insufficient samples
                insuff = len([1 for _, v in variable_stats.items() if not v['sufficient_samples']])
                if insuff:
                    msg.append(f"⚠️ {insuff} variables with insufficient samples (<{min_samples})")
            # Top combinations
            tops = []
            for bitmap, stats in sorted_combos[:5]:
                if stats['total'] >= min_samples:
                    wr = (stats['wins']/stats['total']*100.0) if stats['total'] else 0.0
                    visual = ''.join(['🟢' if c == '1' else '🔴' for c in bitmap])
                    tops.append(f"{visual} {stats['wins']}/{stats['total']} ({wr:.1f}% WR)")
            if tops:
                msg.append("")
                msg.append("Top Gate Combinations:")
                msg.append("(H=HTF, V=Vol, B=Body, A=Align)")
                msg.extend(tops)
            # Append scope and footer
            msg.insert(1, scope)
            msg.append(f"\nMin samples: {min_samples} per gate status")

            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"Error in _scalp_time_vars: {e}")
            await update.message.reply_text("Error computing Variables by Session/Day")

    async def scalp_offhours_toggle(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Toggle off-hours execution block for Scalp (phantoms continue). Usage: /scalpoffhours on|off"""
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            if not args or args[0].lower() not in ('on','off'):
                state = bool(((self.shared.get('scalp_offhours') or {}).get('enabled', False)))
                await self.safe_reply(update, f"🕘 Off-hours exec block is {'ON' if state else 'OFF'}\nUsage: /scalpoffhours on|off")
                return
            state = args[0].lower() == 'on'
            cfg = self.shared.get('scalp_offhours') or {}
            cfg['enabled'] = state
            cfg.setdefault('windows', [])
            cfg.setdefault('allow_htf', False)
            cfg.setdefault('allow_both', False)
            self.shared['scalp_offhours'] = cfg
            await self.safe_reply(update, f"✅ Off-hours exec block set to {'ON' if state else 'OFF'}")
        except Exception as e:
            logger.error(f"Error in scalp_offhours_toggle: {e}")
            await update.message.reply_text("Error toggling off-hours")

    async def scalp_offhours_window(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Manage off-hours windows. Usage: /scalpoffhourswindow add HH:MM-HH:MM | remove HH:MM-HH:MM | list"""
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            cfg = self.shared.get('scalp_offhours') or {'enabled': False, 'windows': [], 'allow_htf': False, 'allow_both': False}
            windows = cfg.get('windows', []) or []
            if not args or args[0].lower() == 'list':
                await self.safe_reply(update, "🕘 *Off-hours windows*\n"+ ("\n".join([f"• {w}" for w in windows]) if windows else "(none)"))
                self.shared['scalp_offhours'] = cfg
                return
            action = args[0].lower()
            if action not in ('add','remove') or len(args) < 2:
                await self.safe_reply(update, "Usage: /scalpoffhourswindow add HH:MM-HH:MM | remove HH:MM-HH:MM | list")
                return
            win = args[1]
            if action == 'add':
                if win not in windows:
                    windows.append(win)
            else:
                if win in windows:
                    windows.remove(win)
            cfg['windows'] = windows
            self.shared['scalp_offhours'] = cfg
            await self.safe_reply(update, f"✅ Off-hours windows: {', '.join(windows) if windows else '(none)'}")
        except Exception as e:
            logger.error(f"Error in scalp_offhours_window: {e}")
            await update.message.reply_text("Error updating off-hours windows")

    async def scalp_offhours_exception(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Toggle exceptions during off-hours. Usage: /scalpoffhoursexception htf on|off | both on|off | status"""
        try:
            args = ctx.args if hasattr(ctx, 'args') else []
            cfg = self.shared.get('scalp_offhours') or {'enabled': False, 'windows': [], 'allow_htf': False, 'allow_both': False}
            if not args or args[0].lower() == 'status':
                await self.safe_reply(update, f"🕘 Off-hours exceptions\n• allow_htf: {cfg.get('allow_htf', False)}\n• allow_both: {cfg.get('allow_both', False)}")
                self.shared['scalp_offhours'] = cfg
                return
            if len(args) != 2 or args[0].lower() not in ('htf','both') or args[1].lower() not in ('on','off'):
                await self.safe_reply(update, "Usage: /scalpoffhoursexception htf on|off | both on|off | status")
                return
            key = 'allow_htf' if args[0].lower() == 'htf' else 'allow_both'
            cfg[key] = (args[1].lower() == 'on')
            self.shared['scalp_offhours'] = cfg
            await self.safe_reply(update, f"✅ Off-hours {args[0].lower()} set to {args[1].lower()}")
        except Exception as e:
            logger.error(f"Error in scalp_offhours_exception: {e}")
            await update.message.reply_text("Error updating off-hours exceptions")

    async def scalp_recommendations(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Generate actionable recommendations with config snippet."""
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()

            # Get analysis and recommendations
            analysis = scpt.get_comprehensive_analysis(month=None, top_n=10, min_samples=20)

            if analysis.get('error'):
                await self.safe_reply(update, f"🎯 <b>Scalp Recommendations</b>\n\n{analysis['error']}", parse_mode='HTML')
                return

            recs = scpt.generate_recommendations(analysis, min_wr=60.0, min_samples=30)

            if recs.get('error'):
                await self.safe_reply(update, f"🎯 <b>Scalp Recommendations</b>\n\n{recs['error']}", parse_mode='HTML')
                return

            # Build message
            msg = [
                f"🎯 <b>Scalp Config Recommendations</b>\n",
                f"📊 Based on {analysis['total_phantoms']} phantoms ({analysis['period']})\n",
            ]

            # Enable recommendations
            if recs['enable']:
                msg.append("━━━ <b>ENABLE (High WR)</b> ━━━")
                for rec in recs['enable'][:5]:
                    msg.append(
                        f"✅ {rec['variable']}: {rec['reason']}, {rec['count']} trades"
                    )

            # Disable recommendations
            if recs['disable']:
                msg.append("\n━━━ <b>DISABLE (Low WR)</b> ━━━")
                for rec in recs['disable'][:5]:
                    msg.append(
                        f"❌ {rec['variable']}: {rec['reason']}, {rec['count']} trades"
                    )

            # Best combinations
            if recs['best_pairs']:
                msg.append("\n━━━ <b>BEST COMBINATIONS</b> ━━━")
                for combo in recs['best_pairs'][:3]:
                    msg.append(
                        f"🎯 {' + '.join(combo['variables'])}: {combo['wr']:.1f}% WR "
                        f"[{combo['count']} trades]"
                    )

            # Config snippet
            if recs['config_snippet']:
                msg.append(f"\n━━━ <b>CONFIG SNIPPET</b> ━━━")
                msg.append(f"<pre>\n{recs['config_snippet']}\n</pre>")
                msg.append("\n<i>Copy/paste to config.yaml</i>")

            await self.safe_reply(update, "\n".join(msg), parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error in scalp_recommendations: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def scalp_monthly_trends(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Analyze Scalp variable performance trends across months."""
        try:
            from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()

            # Parse months argument (optional)
            months = None
            if ctx.args:
                months = ctx.args  # e.g., ['2025-10', '2025-09', '2025-08']

            # Get monthly trends
            trends = scpt.get_monthly_trends(months=months)

            if trends.get('error'):
                await self.safe_reply(update, f"📈 *Scalp Monthly Trends*\\n\\n{trends['error']}")
                return

            # Build message
            months_str = ', '.join(trends['months'])
            summary = trends.get('summary', {})
            var_trends = trends.get('variable_trends', {})

            msg = [
                f"📈 *Scalp Variable Trends* ({len(trends['months'])} months)\\n",
                f"📅 Months: {months_str}\\n",
                f"📊 Summary: {summary.get('improving_count', 0)} improving, "
                f"{summary.get('degrading_count', 0)} degrading, "
                f"{summary.get('stable_count', 0)} stable\\n",
            ]

            # Show top improving variables
            improving = summary.get('improving_vars', [])
            if improving:
                msg.append("━━━ *📈 IMPROVING* ━━━")
                for var in improving[:5]:
                    if var in var_trends:
                        t = var_trends[var]
                        monthly = t.get('monthly_data', {})
                        msg.append(f"✅ {var}:")
                        for month in sorted(monthly.keys())[-3:]:  # Last 3 months
                            data = monthly[month]
                            msg.append(f"  {month}: {data['wr']:.1f}% ({data['delta']:+.1f}%) [{data['count']}]")
                        msg.append(f"  📈 +{t['wr_change']:.1f}% over period\\n")

            # Show top degrading variables
            degrading = summary.get('degrading_vars', [])
            if degrading:
                msg.append("━━━ *📉 DEGRADING* ━━━")
                for var in degrading[:5]:
                    if var in var_trends:
                        t = var_trends[var]
                        monthly = t.get('monthly_data', {})
                        msg.append(f"⚠️ {var}:")
                        for month in sorted(monthly.keys())[-3:]:
                            data = monthly[month]
                            msg.append(f"  {month}: {data['wr']:.1f}% ({data['delta']:+.1f}%) [{data['count']}]")
                        msg.append(f"  📉 {t['wr_change']:.1f}% over period\\n")

            msg.append("\\n💡 Focus on improving variables for config")

            await self.safe_reply(update, "\\n".join(msg))

        except Exception as e:
            logger.error(f"Error in scalp_monthly_trends: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def qscore_all_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Qscore win/loss buckets for Trend, Range, and Scalp with simple recommendations.

        Buckets in 5-pt steps from 50 to 100. Uses persisted phantom stores so it survives restarts.
        """
        try:
            import math
            # Helpers
            def _bucket(q: float) -> int:
                try:
                    return max(50, min(100, int(math.floor(q/5.0)*5)))
                except Exception:
                    return 0
            def _agg_map():
                return {b: {'w':0,'l':0} for b in range(50, 101, 5)}
            def _lines(title: str, agg: dict, exec_thr: int) -> list:
                out = [f"{title} (thr {exec_thr})"]
                # compute WR and rec
                best_below = None; above = None
                for b in range(50, 101, 5):
                    d = agg.get(b, {'w':0,'l':0}); n = d['w']+d['l']
                    if n==0: continue
                    wr = (d['w']/n*100.0) if n else 0.0
                    out.append(f"  {b:>2}+ : N={n} WR={wr:.1f}% (W/L {d['w']}/{d['l']})")
                # recommendation around threshold
                try:
                    def _wr(d):
                        n=d['w']+d['l']; return (d['w']/n*100.0) if n else 0.0
                    below_bin = 5*int(math.floor((exec_thr-1)/5.0))
                    above_bin = 5*int(math.floor(exec_thr/5.0))
                    below = agg.get(below_bin, {'w':0,'l':0}); above = agg.get(above_bin, {'w':0,'l':0})
                    wr_b = _wr(below); wr_a = _wr(above)
                    nb = below['w']+below['l']; na = above['w']+above['l']
                    rec = "Keep threshold"
                    if nb >= 10 and wr_b >= wr_a - 2.0:
                        rec = "Consider -3 thr (near-miss strong)"
                    elif na >= 10 and wr_a < 45.0:
                        rec = "Consider +3 thr (weak above cut)"
                    out.append(f"  Recommendation: {rec}")
                except Exception:
                    pass
                return out
            # Build aggregates
            trend_agg = _agg_map(); range_agg = _agg_map(); scalp_agg = _agg_map()
            # Trend/Range from PhantomTradeTracker
            try:
                pt = self.shared.get('phantom_tracker')
                for arr in (getattr(pt, 'phantom_trades', {}) or {}).values():
                    for p in arr:
                        try:
                            oc = getattr(p, 'outcome', None)
                            if oc not in ('win','loss'): continue
                            feats = getattr(p, 'features', {}) or {}
                            q = feats.get('qscore', None)
                            if not isinstance(q, (int,float)): continue
                            b = _bucket(float(q))
                            strat = str(getattr(p, 'strategy_name','') or '').lower()
                            if strat.startswith('range'):
                                if oc == 'win':
                                    range_agg[b]['w'] += 1
                                else:
                                    range_agg[b]['l'] += 1
                            elif ('trend' in strat) or ('pullback' in strat):
                                if oc == 'win':
                                    trend_agg[b]['w'] += 1
                                else:
                                    trend_agg[b]['l'] += 1
                        except Exception:
                            continue
            except Exception:
                pass
            # Scalp from ScalpPhantomTracker
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                for arr in (getattr(scpt, 'completed', {}) or {}).values():
                    for p in arr:
                        try:
                            oc = getattr(p, 'outcome', None)
                            if oc not in ('win','loss'): continue
                            feats = getattr(p, 'features', {}) or {}
                            q = feats.get('qscore', None)
                            if not isinstance(q, (int,float)): continue
                            b = _bucket(float(q))
                            if oc=='win': scalp_agg[b]['w'] += 1
                            else: scalp_agg[b]['l'] += 1
                        except Exception:
                            continue
            except Exception:
                pass
            # Thresholds from config
            cfg = self.shared.get('config') or {}
            t_thr = int(float(((cfg.get('trend',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 78)))
            r_thr = int(float(((cfg.get('range',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 78)))
            s_thr = int(float(((cfg.get('scalp',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 60)))
            # Assemble message
            msg = ["📊 *Qscore Buckets*", ""]
            msg += _lines("Trend", trend_agg, t_thr); msg.append("")
            msg += _lines("Range", range_agg, r_thr); msg.append("")
            msg += _lines("Scalp", scalp_agg, s_thr)
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"qscore_all_report error: {e}")
            await update.message.reply_text("Error computing Qscore stats")

    async def ml_all_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML-score win/loss buckets for Trend, Range, and Scalp.

        Buckets in 10-pt steps from 50 to 100. Uses phantom stores so it survives restarts.
        Excludes timeouts.
        """
        try:
            import math
            def _bucket(s: float) -> int:
                try:
                    return max(50, min(100, int(math.floor(s/10.0)*10)))
                except Exception:
                    return 0
            def _agg_map():
                return {b: {'w':0,'l':0} for b in range(50, 101, 10)}
            def _lines(title: str, agg: dict) -> list:
                out = [title]
                for b in range(50, 101, 10):
                    d = agg.get(b, {'w':0,'l':0}); n = d['w']+d['l']
                    if n==0: continue
                    wr = (d['w']/n*100.0) if n else 0.0
                    out.append(f"  {b:>2}+ : N={n} WR={wr:.1f}% (W/L {d['w']}/{d['l']})")
                return out
            # Aggregates
            trend_agg = _agg_map(); range_agg = _agg_map(); scalp_agg = _agg_map()
            # Trend/Range from PhantomTradeTracker
            try:
                pt = self.shared.get('phantom_tracker')
                for arr in (getattr(pt, 'phantom_trades', {}) or {}).values():
                    for p in arr:
                        try:
                            oc = getattr(p, 'outcome', None)
                            if oc not in ('win','loss'): continue
                            ms = getattr(p, 'ml_score', None)
                            if not isinstance(ms, (int,float)): continue
                            b = _bucket(float(ms))
                            strat = str(getattr(p, 'strategy_name','') or '').lower()
                            if strat.startswith('range'):
                                if oc=='win': range_agg[b]['w'] += 1
                                else: range_agg[b]['l'] += 1
                            elif ('trend' in strat) or ('pullback' in strat):
                                if oc=='win': trend_agg[b]['w'] += 1
                                else: trend_agg[b]['l'] += 1
                        except Exception:
                            continue
            except Exception:
                pass
            # Scalp from ScalpPhantomTracker
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                for arr in (getattr(scpt, 'completed', {}) or {}).values():
                    for p in arr:
                        try:
                            oc = getattr(p, 'outcome', None)
                            if oc not in ('win','loss'): continue
                            ms = getattr(p, 'ml_score', None)
                            if not isinstance(ms, (int,float)): continue
                            b = _bucket(float(ms))
                            if oc=='win': scalp_agg[b]['w'] += 1
                            else: scalp_agg[b]['l'] += 1
                        except Exception:
                            continue
            except Exception:
                pass
            # Assemble
            msg = ["🧠 *ML Buckets (All Strategies)*", ""]
            msg += _lines("Trend", trend_agg); msg.append("")
            msg += _lines("Range", range_agg); msg.append("")
            msg += _lines("Scalp", scalp_agg)
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"ml_all_report error: {e}")
            await update.message.reply_text("Error computing ML report")

    async def scalp_promotion_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Summarize Scalp promotion readiness (WR-based only)."""
        try:
            # Config thresholds (defaults)
            target_wr = 50.0
            cfg = self.shared.get('config') or {}
            scalp_cfg = cfg.get('scalp', {})
            thr = scalp_cfg.get('threshold', 75)
            # Override from config if provided
            target_wr = float(scalp_cfg.get('promote_min_wr', target_wr))
            promote_enabled = bool(scalp_cfg.get('promote_enabled', False))

            # Stats from dedicated scalp tracker
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                st = scpt.get_scalp_phantom_stats()
            except Exception:
                st = {'total': 0, 'wins': 0, 'losses': 0, 'wr': 0.0}

            # Scalp scorer readiness
            ml_ready = False
            try:
                from autobot.strategies.scalp.scorer import get_scalp_scorer
                s = get_scalp_scorer()
                ml_ready = bool(getattr(s, 'is_ml_ready', False))
                thr = getattr(s, 'min_score', thr)
            except Exception:
                pass

            total = st.get('total', 0)
            wr = st.get('wr', 0.0)
            # Recent WR(window)
            recent_wr = wr
            try:
                from autobot.strategies.scalp.phantom import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                recents = []
                for trades in getattr(scpt, 'completed', {}).values():
                    for p in trades:
                        if getattr(p, 'outcome', None) in ('win','loss'):
                            recents.append(p)
                recents.sort(key=lambda x: getattr(x, 'exit_time', None) or getattr(x, 'signal_time', None))
                window = int(scalp_cfg.get('promote_window', 50))
                recents = recents[-window:]
                if recents:
                    rw = sum(1 for p in recents if getattr(p, 'outcome', None) == 'win')
                    recent_wr = (rw / len(recents)) * 100.0
            except Exception:
                pass
            metric = str(scalp_cfg.get('promote_metric', 'recent')).lower()
            wr_for_gate = recent_wr if metric == 'recent' else wr
            ready = (wr_for_gate >= target_wr)

            lines = [
                "🩳 *Scalp Promotion Status*",
                f"• Phantom recorded: {total} (W/L {st.get('wins',0)}/{st.get('losses',0)})",
                f"• Phantom WR: {wr:.1f}% | Recent({int(scalp_cfg.get('promote_window',50))}): {recent_wr:.1f}%",
                f"• ML Ready: {'✅' if ml_ready else '⏳'} | Threshold: {thr}",
                f"• Gate: {'Recent' if metric=='recent' else 'Overall'} WR ≥ {target_wr:.1f}% (no sample gate)",
                f"• Promotion toggle: {'ON' if promote_enabled else 'OFF'}",
                f"• Recommendation: {'🟢 Ready' if ready else '🟡 Not ready'}",
                "_Promotion executes when phantom WR meets target; micro-context enforced._"
            ]
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in scalp_promotion_status: {e}")
            await update.message.reply_text("Error getting scalp promotion status")

    # Removed: trend_promotion_status method (Trend strategy disabled)
        """Summarize Trend promotion (corking) status and readiness."""
        try:
            cfg = self.shared.get('config') or {}
            tr_cfg = (cfg.get('trend', {}) or {}).get('promotion', {})
            tp = self.shared.get('trend_promotion', {}) or {}
            cap = int(tr_cfg.get('daily_exec_cap', 20))
            promote_wr = float(tr_cfg.get('promote_wr', 55.0))
            demote_wr = float(tr_cfg.get('demote_wr', 35.0))
            min_recent = int(tr_cfg.get('min_recent', 30))
            min_total = int(tr_cfg.get('min_total_trades', 100))
            block_extreme = bool(tr_cfg.get('block_extreme_vol', True))

            # Trend ML stats
            tr_wr = 0.0; tr_recent = 0; total_exec = 0
            try:
                tr_scorer = self.shared.get('trend_scorer') or get_trend_scorer()
                stats = tr_scorer.get_stats() if tr_scorer else {}
                tr_wr = float(stats.get('recent_win_rate', 0.0))
                tr_recent = int(stats.get('recent_trades', 0))
                total_exec = int(stats.get('executed_count', 0))
            except Exception:
                pass

            ready = (tr_recent >= min_recent) and (total_exec >= min_total) and (tr_wr >= promote_wr)
            lines = [
                "📈 *Trend Promotion (Corking)*",
                f"• Status: {'✅ Active' if tp.get('active') else 'Off'} | Used: {tp.get('count',0)}/{cap}",
                f"• Promote/Demote: {promote_wr:.0f}%/{demote_wr:.0f}%",
                f"• Recent WR: {tr_wr:.1f}% (N={tr_recent}) | Total exec: {total_exec}",
                f"• Blocks: Extreme Volatility={'ON' if block_extreme else 'OFF'}",
                f"• Gate: N ≥ {min_recent}, Total ≥ {min_total}, WR ≥ {promote_wr:.0f}%",
                f"• Recommendation: {'🟢 Ready' if ready else '🟡 Not ready'}",
            ]
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in trend_promotion_status: {e}")
            await update.message.reply_text("Error getting trend promotion status")

    async def training_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show background ML training status"""
        try:
            msg = "🎯 *Background ML Training Status*\n"
            msg += "━" * 35 + "\n\n"

            # Get background trainer status
            try:
                from background_initial_trainer import get_background_trainer
                trainer = get_background_trainer()
                status = trainer.get_status()
                
                current_status = status.get('status', 'unknown')
                
                if current_status == 'not_started':
                    msg += "⏳ *Status: Not Started*\n"
                    msg += "Training will begin if no existing ML models are detected.\n\n"
                    
                    # Check if models already exist
                    try:
                        import redis
                        import os
                        redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
                        
                        trend_model = redis_client.get('tml:model')
                        mr_model = redis_client.get('enhanced_mr:model_data')
                        
                        if trend_model and mr_model:
                            msg += "✅ *Existing ML Models Found:*\n"
                            msg += "• Trend ML Model: ✅ Trained\n"
                            msg += "• Enhanced MR Model: ✅ Trained\n\n"
                            msg += "🔄 Live bot handles automatic retraining as trades accumulate.\n"
                        else:
                            msg += "❌ *Missing ML Models:*\n"
                            if not trend_model:
                                msg += "• Trend ML Model: ⏳ Missing\n"
                            if not mr_model:
                                msg += "• Enhanced MR Model: ⏳ Missing\n"
                            msg += "\n📝 Training should start automatically on next bot restart.\n"
                    except:
                        msg += "❓ Unable to check existing models.\n"
                
                elif current_status == 'running':
                    msg += "🚀 *Status: Training In Progress*\n\n"
                    
                    stage = status.get('stage', 'Unknown')
                    symbol = status.get('symbol', '')
                    progress = status.get('progress', 0)
                    total = status.get('total', 0)
                    
                    msg += f"📊 *Current Stage:* {stage}\n"
                    if symbol:
                        msg += f"🔍 *Current Symbol:* {symbol}\n"
                    if total > 0:
                        percentage = (progress / total) * 100
                        msg += f"📈 *Progress:* {progress}/{total} ({percentage:.1f}%)\n"
                    
                    msg += f"\n⏰ *Last Updated:* {status.get('timestamp', 'Unknown')}\n\n"
                    msg += "💡 Training runs in background - live trading continues normally.\n"
                
                elif current_status == 'completed':
                    msg += "🎉 *Status: Training Complete!*\n\n"
                    
                    trend_signals = status.get('trend_signals', 0)
                    mr_signals = status.get('mr_signals', 0)
                    total_symbols = status.get('total_symbols', 0)
                    
                    msg += f"✅ *Results:*\n"
                    msg += f"• Trend Signals: {trend_signals:,}\n"
                    msg += f"• MR Signals: {mr_signals:,}\n"
                    msg += f"• Total Symbols: {total_symbols}\n\n"
                    
                    msg += f"⏰ *Completed:* {status.get('timestamp', 'Unknown')}\n\n"
                    msg += "🔄 *Next Steps:*\n"
                    msg += "• Live bot now handles automatic retraining\n"
                    msg += "• Use `/ml` and `/enhanced_mr` to check model status\n"
                    msg += "• Models retrain automatically as trades accumulate\n"
                
                elif current_status == 'error':
                    msg += "❌ *Status: Training Error*\n\n"
                    
                    error = status.get('error', 'Unknown error')
                    msg += f"🚨 *Error:* {error}\n\n"
                    msg += f"⏰ *Error Time:* {status.get('timestamp', 'Unknown')}\n\n"
                    msg += "🔄 *Recovery:*\n"
                    msg += "• Training will retry on next bot restart\n"
                    msg += "• Check logs for detailed error information\n"
                    msg += "• Ensure sufficient disk space and memory\n"
                
                else:
                    msg += f"❓ *Status: {current_status}*\n"
                    msg += "Unknown training status.\n"
                    
            except ImportError:
                msg += "❌ *Background Trainer Not Available*\n"
                msg += "Background training module not found.\n\n"
                msg += "💡 Use the existing `/ml` commands to check model status.\n"
            except Exception as e:
                msg += f"❌ *Error Getting Status*\n"
                msg += f"Error: {str(e)[:100]}...\n\n"
                msg += "Try again in a few moments.\n"

            msg += "\n" + "━" * 35 + "\n"
            msg += "📋 *Available Commands:*\n"
            msg += "`/ml` - Trend ML status\n"
            msg += "`/enhanced_mr` - Enhanced MR status\n"
            msg += "`/phantom` - Phantom tracking stats\n"
            msg += "`/mr_phantom` - MR phantom stats\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in training_status: {e}")
            await update.message.reply_text("Error getting training status")

    async def set_qscore(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set Qscore thresholds at runtime.
        Usage: /set_qscore <trend|range|scalp> <execQ> [phantomQ]
        Examples: /set_qscore trend 80 65 | /set_qscore scalp 60
        """
        try:
            txt = (getattr(update, 'message', None) and update.message.text) or ''
            parts = txt.strip().split()
            if len(parts) < 3:
                await self.safe_reply(update, "Usage: /set_qscore <trend|range|scalp> <execQ> [phantomQ]")
                return
            _, strat, exec_q, *rest = parts
            strat = strat.lower()
            if strat not in ('trend','range','scalp'):
                await self.safe_reply(update, "Strategy must be: trend, range, scalp")
                return
            try:
                exec_q = float(exec_q)
            except Exception:
                await self.safe_reply(update, "Exec Q must be a number")
                return
            ph_q = None
            if rest:
                try:
                    ph_q = float(rest[0])
                except Exception:
                    await self.safe_reply(update, "Phantom Q must be a number")
                    return
            # Update shared config
            cfg = self.shared.get('config', {})
            cfg.setdefault(strat, {}).setdefault('rule_mode', {})
            cfg[strat]['rule_mode']['execute_q_min'] = exec_q
            if ph_q is not None:
                cfg[strat]['rule_mode']['phantom_q_min'] = ph_q
            # Update live bot config
            bot = self.shared.get('bot_instance')
            if bot and hasattr(bot, 'config') and isinstance(bot.config, dict):
                bot.config.setdefault(strat, {}).setdefault('rule_mode', {})
                bot.config[strat]['rule_mode']['execute_q_min'] = exec_q
                if ph_q is not None:
                    bot.config[strat]['rule_mode']['phantom_q_min'] = ph_q
            # Persist in Redis for restarts
            try:
                import os, redis
                url = os.getenv('REDIS_URL')
                if url:
                    r = redis.from_url(url, decode_responses=True)
                    r.set(f'config_override:{strat}:rule_mode:execute_q_min', str(exec_q))
                    if ph_q is not None:
                        r.set(f'config_override:{strat}:rule_mode:phantom_q_min', str(ph_q))
            except Exception:
                pass
            msg = [f"✅ Set {strat} execute_q_min={exec_q:.0f}"]
            if ph_q is not None:
                msg.append(f"phantom_q_min={ph_q:.0f}")
            await self.safe_reply(update, " ".join(msg))
        except Exception as e:
            logger.error(f"set_qscore error: {e}")
            await update.message.reply_text("Error setting Qscore")

    async def flow_debug(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Flow Controller status: accepted, targets, relax per strategy."""
        try:
            fc = self.shared.get('flow_controller')
            if not fc or not getattr(fc, 'enabled', False):
                await self.safe_reply(update, "🎛️ Flow Controller disabled or unavailable")
                return
            st = fc.get_status() if hasattr(fc, 'get_status') else {}
            targets = st.get('targets', {})
            acc = st.get('accepted', {})
            rx = st.get('relax', {})
            comps = st.get('components', {}) or {}
            # Compute instant (raw) relax based on accepted and time-of-day pace
            try:
                from datetime import datetime as _dt
                hour = _dt.utcnow().hour
            except Exception:
                hour = 12
            def _inst_rel(ax: int, tgt: int) -> float:
                try:
                    pace_target = float(tgt) * min(1.0, max(1, hour) / 24.0)
                    deficit = max(0.0, pace_target - float(ax))
                    base_r = min(1.0, deficit / max(1.0, tgt * 0.5))
                    return base_r
                except Exception:
                    return 0.0
            ir_pb = _inst_rel(int(acc.get('trend',0)), int(targets.get('trend',0) or 1))
            ir_mr = _inst_rel(int(acc.get('mr',0)), int(targets.get('mr',0) or 1))
            ir_sc = _inst_rel(int(acc.get('scalp',0)), int(targets.get('scalp',0) or 1))
            def _fmt_line(name_key, label, inst_rel):
                try:
                    c = comps.get(name_key) or {}
                    if c:
                        wr_txt = ""
                        try:
                            if c.get('wr') is not None:
                                wr_txt = f", wr {float(c.get('wr',0.0))*100:.0f}%"
                                if bool(c.get('guard_active', False)):
                                    wr_txt += f" (guard {float(c.get('guard_cap',0.0))*100:.0f}% active)"
                        except Exception:
                            pass
                        return (
                            f"• {label}: {acc.get(name_key,0)}/{targets.get(name_key,0)} "
                            f"(relax {float(rx.get(name_key,0.0))*100:.0f}%, "
                            f"pace {float(c.get('pace',0.0))*100:.0f}%, def {float(c.get('deficit',0.0))*100:.0f}%, "
                            f"boost +{float(c.get('boost',0.0))*100:.0f}%, min {float(c.get('min',0.0))*100:.0f}%{wr_txt})"
                        )
                except Exception:
                    pass
                return (
                    f"• {label}: {acc.get(name_key,0)}/{targets.get(name_key,0)} "
                    f"(relax {float(rx.get(name_key,0.0))*100:.0f}%, inst {inst_rel*100:.0f}%)"
                )

            lines = [
                "🎛️ *Flow Controller Status*",
                f"• Enabled: {st.get('enabled', False)}",
                f"• Smoothing hours: {st.get('smoothing_hours', '?')}",
                "",
                _fmt_line('trend', 'Trend', ir_pb),
                _fmt_line('mr', 'Mean Reversion', ir_mr),
                _fmt_line('scalp', 'Scalp', ir_sc),
                "",
                "_Relax raises sampling when behind pace; deficit/boost help catch up; guards enforce minimum quality._"
            ]
            await self.safe_reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"Error in flow_debug: {e}")
            await update.message.reply_text("Error getting flow status")
    async def _on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Handle numeric inputs for settings when a prompt is active."""
        try:
            chat = update.effective_chat
            if not chat or chat.id != self.chat_id:
                return
            state = self._ui_state.get(chat.id)
            if not state or 'await' not in state:
                return
            key = state['await']
            text = (update.message.text or '').strip()
            cfg = self.shared.get('config', {}) or {}
            tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
            sc = (tr_exec.get('scaleout',{}) or {})
            div = (tr_exec.get('divergence',{}) or {})
            ts = self.shared.get('trend_settings')
            pt = self.shared.get('phantom_tracker')

            def _ok(msg: str):
                self._ui_state.pop(chat.id, None)
                return ctx.application.create_task(self.send_message(msg))

            # Parse and apply
            if key == 'exec_q':
                try:
                    val = float(text)
                    cfg.setdefault('trend', {}).setdefault('rule_mode', {})
                    cfg['trend']['rule_mode']['execute_q_min'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    # Persist in Redis for restarts
                    try:
                        import os, redis
                        url = os.getenv('REDIS_URL')
                        if url:
                            r = redis.from_url(url, decode_responses=True)
                            r.set('config_override:trend:rule_mode:execute_q_min', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Trend Exec≥Q set to {val:.0f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 78")
                return

            if key == 'phantom_q':
                try:
                    val = float(text)
                    cfg.setdefault('trend', {}).setdefault('rule_mode', {})
                    cfg['trend']['rule_mode']['phantom_q_min'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    try:
                        import os, redis
                        url = os.getenv('REDIS_URL')
                        if url:
                            r = redis.from_url(url, decode_responses=True)
                            r.set('config_override:trend:rule_mode:phantom_q_min', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Trend Phantom≥Q set to {val:.0f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 65")
                return

            if key == 'timeout_hl':
                try:
                    val = int(text)
                    if val < 1 or val > 60:
                        await update.message.reply_text("Bars must be 1–60")
                        return
                    ts = self.shared.get('trend_settings')
                    if ts:
                        ts.breakout_to_pullback_bars_3m = val
                    cfg.setdefault('trend', {}).setdefault('exec', {}).setdefault('timeouts', {})
                    cfg['trend']['exec']['timeouts']['breakout_to_pullback_bars_3m'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    await _ok(f"✅ HL→PB timeout set to {val} bars")
                except Exception:
                    await update.message.reply_text("Please send an integer, e.g., 25")
                return

            if key == 'timeout_bos':
                try:
                    val = int(text)
                    if val < 1 or val > 60:
                        await update.message.reply_text("Bars must be 1–60")
                        return
                    ts = self.shared.get('trend_settings')
                    if ts:
                        ts.pullback_to_bos_bars_3m = val
                    cfg.setdefault('trend', {}).setdefault('exec', {}).setdefault('timeouts', {})
                    cfg['trend']['exec']['timeouts']['pullback_to_bos_bars_3m'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    await _ok(f"✅ PB→BOS timeout set to {val} bars")
                except Exception:
                    await update.message.reply_text("Please send an integer, e.g., 25")
                return
            if key == 'rr':
                try:
                    val = float(text)
                    if val < 1.2 or val > 5.0:
                        await update.message.reply_text("Value out of range (1.2–5.0)")
                        return
                    if ts:
                        ts.rr = val
                    self.shared['risk_reward'] = val
                    await _ok(f"✅ R:R set to 1:{val}")
                except Exception:
                    await update.message.reply_text("Please send a valid number, e.g., 2.5")
                return

            if key == 'tp1_r':
                try:
                    val = float(text)
                    if val < 1.1 or val > 3.0:
                        await update.message.reply_text("Value out of range (1.1–3.0)")
                        return
                    sc['tp1_r'] = val
                    tr_exec['scaleout'] = sc
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ TP1 R set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a valid number, e.g., 1.6")
                return

            if key == 'tp2_r':
                try:
                    val = float(text)
                    if val < 1.5 or val > 6.0:
                        await update.message.reply_text("Value out of range (1.5–6.0)")
                        return
                    sc['tp2_r'] = val
                    tr_exec['scaleout'] = sc
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ TP2 R set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a valid number, e.g., 3.0")
                return

            if key == 'fraction':
                try:
                    val = float(text)
                    if val < 0.1 or val > 0.9:
                        await update.message.reply_text("Fraction must be between 0.1 and 0.9")
                        return
                    sc['fraction'] = val
                    tr_exec['scaleout'] = sc
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ Scale‑out fraction set to {val:.2f}")
                except Exception:
                    await update.message.reply_text("Please send a valid number, e.g., 0.5")
                return

            if key == 'confirm_bars':
                try:
                    val = int(text)
                    if val < 1 or val > 24:
                        await update.message.reply_text("Bars must be 1–24")
                        return
                    if ts:
                        ts.confirmation_timeout_bars = val
                    await _ok(f"✅ Confirmation timeout set to {val} bars")
                except Exception:
                    await update.message.reply_text("Please send an integer, e.g., 6")
                return

            if key == 'phantom_hours':
                try:
                    val = int(text)
                    if val < 1 or val > 240:
                        await update.message.reply_text("Hours must be 1–240")
                        return
                    if pt:
                        pt.timeout_hours = val
                    await _ok(f"✅ Phantom timeout set to {val}h")
                except Exception:
                    await update.message.reply_text("Please send an integer, e.g., 100")
                return

            if key == 'phantom_weight':
                try:
                    val = float(text)
                    if val < 0.3 or val > 1.5:
                        await update.message.reply_text("Weight must be 0.3–1.5")
                        return
                    ph = (cfg.get('phantom', {}) or {})
                    ph['weight'] = float(val)
                    cfg['phantom'] = ph
                    self.shared['config'] = cfg
                    try:
                        bot_instance = self.shared.get('bot_instance')
                        if bot_instance:
                            bot_instance.config = cfg
                    except Exception:
                        pass
                    # Also persist to Redis so ML scorer can pick it up live
                    try:
                        bot_instance = self.shared.get('bot_instance')
                        r = getattr(bot_instance, '_redis', None)
                        if r:
                            r.set('ml:trend:phantom_weight', str(val))
                            r.set('phantom:weight', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Phantom training weight set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a valid number, e.g., 0.8")
                return

            if key == 'sr_strength':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    sr = (tr_exec.get('sr', {}) or {})
                    sr['min_strength'] = float(val)
                    tr_exec['sr'] = sr
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    ts = self.shared.get('trend_settings')
                    if ts:
                        ts.sr_min_strength = float(val)
                    await _ok(f"✅ SR min strength set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 2.8")
                return

            if key == 'sr_confluence':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    sr = (tr_exec.get('sr', {}) or {})
                    sr['confluence_tolerance_pct'] = float(val)
                    tr_exec['sr'] = sr
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    ts = self.shared.get('trend_settings')
                    if ts:
                        ts.sr_confluence_tolerance_pct = float(val)
                    await _ok(f"✅ SR confluence tolerance set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 0.0025")
                return

            if key == 'sr_clear':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    sr = (tr_exec.get('sr', {}) or {})
                    sr['min_break_clear_atr'] = float(val)
                    tr_exec['sr'] = sr
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    ts = self.shared.get('trend_settings')
                    if ts:
                        ts.sr_min_break_clear_atr = float(val)
                    await _ok(f"✅ SR min clearance set to {val} ATR")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 0.10")
                return

            

            if key == 'htf_ts1h':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    hg = (tr_exec.get('htf_gate', {}) or {})
                    hg['min_trend_strength_1h'] = float(val)
                    tr_exec['htf_gate'] = hg
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ HTF min 1H trend strength set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 60")
                return

            if key == 'htf_ts4h':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    hg = (tr_exec.get('htf_gate', {}) or {})
                    hg['min_trend_strength_4h'] = float(val)
                    tr_exec['htf_gate'] = hg
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ HTF min 4H trend strength set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 55")
                return

            if key == 'htf_adx1h':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    hg = (tr_exec.get('htf_gate', {}) or {})
                    hg['adx_min_1h'] = float(val)
                    tr_exec['htf_gate'] = hg
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ HTF 1H ADX minimum set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 20")
                return

            if key == 'htf_soft_delta':
                try:
                    val = float(text)
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    hg = (tr_exec.get('htf_gate', {}) or {})
                    hg['soft_delta'] = float(val)
                    tr_exec['htf_gate'] = hg
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    await _ok(f"✅ HTF soft delta set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 5")
                return

            if key == 'sl_buffer':
                try:
                    val = float(text)
                    if val < 0.05 or val > 1.00:
                        await update.message.reply_text("Buffer must be 0.05–1.00 ATR")
                        return
                    tr_exec = ((cfg.get('trend',{}) or {}).get('exec',{}) or {})
                    tr_exec['breakout_sl_buffer_atr'] = float(val)
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    try:
                        bot_instance = self.shared.get('bot_instance')
                        if bot_instance:
                            bot_instance.config = cfg
                    except Exception:
                        pass
                    ts = self.shared.get('trend_settings')
                    if ts:
                        ts.breakout_sl_buffer_atr = float(val)
                    await _ok(f"✅ SL breakout buffer set to {val:.2f} ATR")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 0.40")
                return

            if key == 'htf_min_ts':
                try:
                    val = float(text)
                    if val < 40 or val > 90:
                        await update.message.reply_text("Min trend strength must be 40–90")
                        return
                    router = cfg.setdefault('router', {}).setdefault('htf_bias', {}).setdefault('trend', {})
                    router['min_trend_strength'] = float(val)
                    self.shared['config'] = cfg
                    try:
                        bot_instance = self.shared.get('bot_instance')
                        if bot_instance:
                            bot_instance.config = cfg
                    except Exception:
                        pass
                    await _ok(f"✅ HTF min trend strength set to {val:.1f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 60")
                return

            # Divergence settings
            if key == 'div_rsi_len':
                try:
                    val = int(text)
                    if val < 2 or val > 100:
                        await update.message.reply_text("RSI length must be 2–100")
                        return
                    div['rsi_len'] = val
                    tr_exec['divergence'] = div
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    if ts:
                        ts.div_rsi_len = val
                    await _ok(f"✅ RSI length set to {val}")
                except Exception:
                    await update.message.reply_text("Please send an integer, e.g., 14")
                return

            if key == 'div_tsi_params':
                try:
                    parts = [int(x.strip()) for x in text.split(',')]
                    if len(parts) != 2:
                        await update.message.reply_text("Send as long,short e.g., 25,13")
                        return
                    lo, sh = parts
                    if lo < 2 or lo > 200 or sh < 2 or sh > 200:
                        await update.message.reply_text("TSI params must be between 2 and 200, e.g., 25,13")
                        return
                    div['tsi_long'] = lo; div['tsi_short'] = sh
                    tr_exec['divergence'] = div
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    if ts:
                        ts.div_tsi_long = lo; ts.div_tsi_short = sh
                    await _ok(f"✅ TSI params set to {lo},{sh}")
                except Exception:
                    await update.message.reply_text("Please send as long,short e.g., 25,13")
                return

            if key == 'div_window':
                try:
                    val = int(text)
                    if val < 0 or val > 50:
                        await update.message.reply_text("Window must be 0–50 bars")
                        return
                    div['confirm_window_bars_3m'] = val
                    tr_exec['divergence'] = div
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    if ts:
                        ts.div_window_bars_3m = val
                    await _ok(f"✅ Divergence window set to {val} bars")
                except Exception:
                    await update.message.reply_text("Please send an integer, e.g., 6")
                return

            if key == 'div_min_rsi':
                try:
                    val = float(text)
                    if 'min_strength' not in div or not isinstance(div.get('min_strength'), dict):
                        div['min_strength'] = {}
                    div['min_strength']['rsi'] = val
                    tr_exec['divergence'] = div
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    if ts:
                        ts.div_min_strength_rsi = val
                    await _ok(f"✅ Minimum RSI delta set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 2.0")
                return

            if key == 'div_min_tsi':
                try:
                    val = float(text)
                    if 'min_strength' not in div or not isinstance(div.get('min_strength'), dict):
                        div['min_strength'] = {}
                    div['min_strength']['tsi'] = val
                    tr_exec['divergence'] = div
                    cfg.setdefault('trend', {}).setdefault('exec', {}).update(tr_exec)
                    self.shared['config'] = cfg
                    if ts:
                        ts.div_min_strength_tsi = val
                    await _ok(f"✅ Minimum TSI delta set to {val}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 0.3")
                return

        except Exception:
            # swallow
            pass
            if key == 'exec_q_range':
                try:
                    val = float(text)
                    cfg.setdefault('range', {}).setdefault('rule_mode', {})
                    cfg['range']['rule_mode']['execute_q_min'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    try:
                        import os, redis
                        url = os.getenv('REDIS_URL')
                        if url:
                            r = redis.from_url(url, decode_responses=True)
                            r.set('config_override:range:rule_mode:execute_q_min', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Range Exec≥Q set to {val:.0f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 78")
                return

            if key == 'phantom_q_range':
                try:
                    val = float(text)
                    cfg.setdefault('range', {}).setdefault('rule_mode', {})
                    cfg['range']['rule_mode']['phantom_q_min'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    try:
                        import os, redis
                        url = os.getenv('REDIS_URL')
                        if url:
                            r = redis.from_url(url, decode_responses=True)
                            r.set('config_override:range:rule_mode:phantom_q_min', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Range Phantom≥Q set to {val:.0f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 65")
                return

            if key == 'exec_q_scalp':
                try:
                    val = float(text)
                    cfg.setdefault('scalp', {}).setdefault('rule_mode', {})
                    cfg['scalp']['rule_mode']['execute_q_min'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    try:
                        import os, redis
                        url = os.getenv('REDIS_URL')
                        if url:
                            r = redis.from_url(url, decode_responses=True)
                            r.set('config_override:scalp:rule_mode:execute_q_min', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Scalp Exec≥Q set to {val:.0f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 90")
                return

            if key == 'phantom_q_scalp':
                try:
                    val = float(text)
                    cfg.setdefault('scalp', {}).setdefault('rule_mode', {})
                    cfg['scalp']['rule_mode']['phantom_q_min'] = val
                    self.shared['config'] = cfg
                    bot_instance = self.shared.get('bot_instance')
                    if bot_instance and hasattr(bot_instance, 'config'):
                        bot_instance.config = cfg
                    try:
                        import os, redis
                        url = os.getenv('REDIS_URL')
                        if url:
                            r = redis.from_url(url, decode_responses=True)
                            r.set('config_override:scalp:rule_mode:phantom_q_min', str(val))
                    except Exception:
                        pass
                    await _ok(f"✅ Scalp Phantom≥Q set to {val:.0f}")
                except Exception:
                    await update.message.reply_text("Please send a number, e.g., 80")
                return
