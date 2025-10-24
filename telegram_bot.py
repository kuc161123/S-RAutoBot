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

from ml_scorer_trend import get_trend_scorer
from ml_scorer_mean_reversion import get_mean_reversion_scorer
from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary

logger = logging.getLogger(__name__)

class TGBot:
    def __init__(self, token:str, chat_id:int, shared:dict):
        # shared contains {"risk": RiskConfig, "book": Book, "panic": list, "meta": dict}
        # Build Application with a resilient HTTPX client when available
        if HTTPXRequest is not None:
            try:
                request = HTTPXRequest(
                    connect_timeout=10.0,
                    read_timeout=35.0,
                    write_timeout=35.0,
                    pool_timeout=10.0
                )
                self.app = Application.builder().token(token).request(request).build()
            except Exception:
                self.app = Application.builder().token(token).build()
        else:
            self.app = Application.builder().token(token).build()
        self.chat_id = chat_id
        self.shared = shared
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
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("panic_close", self.panic_close))
        self.app.add_handler(CommandHandler("balance", self.balance))
        self.app.add_handler(CommandHandler("health", self.health))
        self.app.add_handler(CommandHandler("symbols", self.symbols))
        self.app.add_handler(CommandHandler("dashboard", self.dashboard))
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
        self.app.add_handler(CommandHandler("mr_ml", self.mr_ml_stats))
        self.app.add_handler(CommandHandler("mr_retrain", self.mr_retrain))
        self.app.add_handler(CommandHandler("enhanced_mr", self.enhanced_mr_stats))
        self.app.add_handler(CommandHandler("enhancedmr", self.enhanced_mr_stats))  # Alternative command name
        self.app.add_handler(CommandHandler("mr_phantom", self.mr_phantom_stats))
        self.app.add_handler(CommandHandler("mrphantom", self.mr_phantom_stats))  # Alternative command name
        self.app.add_handler(CommandHandler("parallel_performance", self.parallel_performance))
        self.app.add_handler(CommandHandler("parallelperformance", self.parallel_performance))  # Alternative command name
        # Trend pullback state snapshot
        self.app.add_handler(CommandHandler("trend_states", self.trend_states))
        self.app.add_handler(CommandHandler("trendstates", self.trend_states))  # Alternative command name
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
        self.app.add_handler(CommandHandler("scalppromote", self.scalp_promotion_status))
        self.app.add_handler(CommandHandler("trendpromote", self.trend_promotion_status))
        # Trend ML high-ML threshold changer
        self.app.add_handler(CommandHandler("trendhighml", self.trend_high_ml))
        self.app.add_handler(CallbackQueryHandler(self.ui_callback, pattern=r"^ui:"))
        # Capture numeric input replies when a settings prompt is active
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_text))
        self.app.add_handler(CommandHandler("mlstatus", self.ml_stats))
        self.app.add_handler(CommandHandler("panicclose", self.panic_close))
        self.app.add_handler(CommandHandler("forceretrain", self.force_retrain_ml))
        self.app.add_handler(CommandHandler("shadowstats", self.shadow_stats))
        self.app.add_handler(CommandHandler("flowdebug", self.flow_debug))
        self.app.add_handler(CommandHandler("flowstatus", self.flow_debug))
        # High-ML threshold controls per strategy (with and without underscores)
        self.app.add_handler(CommandHandler("scalp_highml", self.set_scalp_highml))
        self.app.add_handler(CommandHandler("mr_highml", self.set_mr_highml))
        self.app.add_handler(CommandHandler("trend_highml", self.set_trend_highml))
        # Aliases without underscore as requested
        self.app.add_handler(CommandHandler("scalphighml", self.set_scalp_highml))
        self.app.add_handler(CommandHandler("mrhighml", self.set_mr_highml))
        self.app.add_handler(CommandHandler("trendhighml", self.set_trend_highml))

        self.running = False

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
        except Exception:
            lines.append("(unavailable)")

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
        except Exception:
            pass

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Refresh", callback_data="ui:dash:refresh")],
            [InlineKeyboardButton("📐 Trend States", callback_data="ui:trend:states"), InlineKeyboardButton("📊 Positions", callback_data="ui:positions")],
            [InlineKeyboardButton("📜 Recent", callback_data="ui:recent"), InlineKeyboardButton("👻 Phantom", callback_data="ui:phantom:trend"), InlineKeyboardButton("📦 Range", callback_data="ui:phantom:range")],
            [InlineKeyboardButton("🤖 ML", callback_data="ui:ml:trend"), InlineKeyboardButton("🧠 Patterns", callback_data="ui:ml:patterns")],
            [InlineKeyboardButton("🧭 Events", callback_data="ui:events"), InlineKeyboardButton("⚙️ Settings", callback_data="ui:settings")]
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
            from ml_scorer_scalp import get_scalp_scorer
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
            from scalp_phantom_tracker import get_scalp_phantom_tracker
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
                from scalp_phantom_tracker import get_scalp_phantom_tracker
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

        # MR Phantom
        mr_phantom = self.shared.get("mr_phantom_tracker")
        if mr_phantom:
            try:
                mr_stats = mr_phantom.get_mr_phantom_stats()
                lines.append("")
                lines.append("🌀 *MR Phantom*")
                lines.append(f"• Tracked: {mr_stats.get('total_mr_trades', 0)}")
                try:
                    timeouts = mr_stats.get('mr_specific_metrics', {}).get('timeout_closures', 0)
                    lines.append(f"• Timeouts: {timeouts}")
                except Exception:
                    pass
            except Exception as exc:
                logger.debug(f"Unable to fetch MR phantom stats: {exc}")

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
                from scalp_phantom_tracker import get_scalp_phantom_tracker
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
            # Mean Reversion (phantom)
            lines.append(f"• 🌀 Mean Reversion")
            lines.append(f"  ✅ W: {mr_wins}   ❌ L: {mr_losses}   🎯 WR: {mr_wr:.1f}%")
            lines.append(f"  🟢 Open: {mr_open_trades} ({mr_open_syms} syms)   🔒 Closed: {mr_closed}")
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
            # Executed MR
            mrx = exec_stats['mr']
            mrx_wr = (mrx['wins'] / (mrx['wins'] + mrx['losses']) * 100.0) if (mrx['wins'] + mrx['losses']) else 0.0
            lines.append("• 🌀 Mean Reversion")
            lines.append(f"  ✅ W: {mrx['wins']}   ❌ L: {mrx['losses']}   🎯 WR: {mrx_wr:.1f}%")
            lines.append(f"  🔓 Open: {mrx['open']}   🔒 Closed: {mrx['closed']}")
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
             InlineKeyboardButton("🧪 Scalp Promote", callback_data="ui:scalp:promote")],
            [InlineKeyboardButton("🧱 HTF S/R", callback_data="ui:htf:status"),
             InlineKeyboardButton("🔄 Update S/R", callback_data="ui:htf:update")],
            [InlineKeyboardButton("⚙️ Risk", callback_data="ui:risk:main"),
             InlineKeyboardButton("🧭 Regime", callback_data="ui:regime:main")]
        ])

        return "\n".join(lines), kb

    async def start_polling(self):
        """Start the bot polling"""
        if not self.running:
            await self.app.initialize()
            await self.app.start()
            self.running = True
            logger.info("Telegram bot started polling")
            # Start polling in background, drop any pending updates to avoid conflicts
            await self.app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=list(UpdateType)
            )

    async def stop(self):
        """Stop the bot"""
        if self.running:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.running = False
            logger.info("Telegram bot stopped")

    async def send_message(self, text:str):
        """Send message to configured chat with retry on network errors"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Try with Markdown first
                await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown')
                return  # Success, exit
            except telegram.error.BadRequest as e:
                if "can't parse entities" in str(e).lower():
                    # Markdown parsing failed, try with better escaping
                    logger.warning("Markdown parsing failed, trying with escaped text")
                    try:
                        # Escape common problematic characters
                        escaped_text = text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')
                        await self.app.bot.send_message(chat_id=self.chat_id, text=escaped_text, parse_mode='Markdown')
                        return  # Success, exit
                    except:
                        # If still fails, send as plain text
                        logger.warning("Escaped markdown also failed, sending as plain text")
                        try:
                            await self.app.bot.send_message(chat_id=self.chat_id, text=text)
                            return  # Success, exit
                        except Exception as plain_e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Plain text send failed (attempt {attempt + 1}/{max_retries}): {plain_e}")
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                logger.error(f"Failed to send message after {max_retries} attempts: {plain_e}")
                else:
                    logger.error(f"Failed to send message: {e}")
                    return  # Don't retry on non-network errors
            except (telegram.error.NetworkError, telegram.error.TimedOut) as e:
                # Network-related errors, retry
                if attempt < max_retries - 1:
                    logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to send message after {max_retries} attempts: {e}")
            except Exception as e:
                # Check if it's a network-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['httpx.readerror', 'network', 'timeout', 'connection']):
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to send message after {max_retries} attempts: {e}")
                else:
                    logger.error(f"Failed to send message: {e}")
                    return  # Don't retry on non-network errors
    
    async def safe_reply(self, update: Update, text: str, parse_mode: str = 'Markdown'):
        """Safely reply to a message with automatic fallback and retry"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                await update.message.reply_text(text, parse_mode=parse_mode)
                return  # Success, exit
            except telegram.error.BadRequest as e:
                if "can't parse entities" in str(e).lower():
                    logger.warning(f"Markdown parsing failed in reply, trying escaped")
                    try:
                        # More comprehensive escaping
                        escaped_text = text
                        # Escape special markdown characters
                        for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                            escaped_text = escaped_text.replace(char, f'\\{char}')
                        await update.message.reply_text(escaped_text, parse_mode=parse_mode)
                        return  # Success, exit
                    except Exception as e2:
                        # Final fallback to plain text
                        logger.warning(f"Escaped markdown also failed ({e2}), replying as plain text")
                        # Remove all markdown formatting
                        plain_text = text
                        for char in ['*', '_', '`', '~']:
                            plain_text = plain_text.replace(char, '')
                        try:
                            await update.message.reply_text(plain_text, parse_mode=None)
                            return  # Success, exit
                        except Exception as plain_e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Plain text reply failed (attempt {attempt + 1}/{max_retries}): {plain_e}")
                                await asyncio.sleep(retry_delay)
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
                    logger.warning(f"Network error in reply (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to reply after {max_retries} attempts: {e}")
            except Exception as e:
                # Check if it's a network-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['httpx.readerror', 'network', 'timeout', 'connection']):
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error in reply (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay)
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

    async def help(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        per_trade, risk_label = self._compute_risk_snapshot()
        timeframe = self.shared.get("timeframe", "15")
        lines = [
            "📚 *Bot Help*",
            "",
            "Monitoring",
            "• /dashboard — Dashboard (Trend + Range phantom)",
            "• /trend_states — Trend state per symbol",
            "• /recent — Recent executed (trend)",
            "",
            "ML",
            "• /ml — Trend ML status",
            "• /mlpatterns — Learned patterns",
            "",
            "Risk",
            "• /risk — Show current risk",
            "• /risk_percent <V> or /riskpercent <V> — Set percent risk (e.g., 2.5)",
            "• /risk_usd <V> or /riskusd <V> — Set fixed USD risk (e.g., 100)",
            "• /set_risk <amount> — Flexible (e.g., 3% or 50)",
            "",
            "Settings (via dashboard → Settings)",
            "• Rule‑Mode thresholds (Exec≥Q, Phantom≥Q)",
            "• Stream entry On/Off",
            "• Scale‑out (TP1/TP2, fraction) and BE move",
            "• Timeouts (HL/LH, confirmations, phantom)",
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
                from scalp_phantom_tracker import get_scalp_phantom_tracker
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
                    from scalp_phantom_tracker import get_scalp_phantom_tracker
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
            
            msg = f"""📊 *Risk Management Settings*
            
{balance_text}⚙️ *Mode:* {mode}
💸 *Risk per trade:* {risk_amount}
📈 *Risk/Reward Ratio:* 1:{risk.rr if hasattr(risk, 'rr') else 2.5}

*Commands:*
`/risk_percent 2.5` - Set to 2.5%
`/risk_usd 100` - Set to $100
`/set_risk 3%` or `/set_risk 50` - Flexible"""
            
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
            
            # Apply the change
            self.shared["risk"].risk_percent = value
            self.shared["risk"].use_percent_risk = True
            
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
            # simple rate-limit to avoid Telegram flood control on heavy dashboard
            if not self._cooldown_ok('dashboard'):
                await self.safe_reply(update, "⏳ Please wait before using /dashboard again")
                return
            # Use Trend-only dashboard with inline Settings for consistency
            text, kb = self._build_trend_dashboard()
            await update.message.reply_text(text, reply_markup=kb)

        except Exception as e:
            logger.exception("Error in dashboard: %s", e)
            await update.message.reply_text("Error getting dashboard")

    async def ui_callback(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Handle inline UI callbacks"""
        try:
            query = update.callback_query
            data = query.data or ""
            # Trend-only UI routes (take precedence)
            if data == "ui:dash:refresh":
                await query.answer("Refreshing…")
                text, kb = self._build_trend_dashboard()
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode='Markdown')
                except Exception:
                    await self.safe_reply(type('obj', (object,), {'message': query.message}), text)
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
                    lines = ["⚙️ *Settings*", "",
                             f"Stream entry: {'On' if tr_exec.get('allow_stream_entry', True) else 'Off'}",
                             f"Scale‑out: {'On' if sc.get('enabled', False) else 'Off'}",
                             f"BE move: {'On' if sc.get('move_sl_to_be', True) else 'Off'}",
                             f"R:R: 1:{rr_val if rr_val is not None else '2.5'}",
                             f"TP1 R: {sc.get('tp1_r',1.6)} | TP2 R: {sc.get('tp2_r',3.0)} | Fraction: {sc.get('fraction',0.5):.2f}",
                             f"Confirm timeout bars: {conf_bars if conf_bars is not None else '6'} | Phantom timeout h: {(getattr(self.shared.get('phantom_tracker'), 'timeout_hours', 0) or 0)}",
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
                        [InlineKeyboardButton("Stream Entry", callback_data="ui:settings:toggle:stream")],
                        [InlineKeyboardButton("Scale‑out", callback_data="ui:settings:toggle:scaleout")],
                        [InlineKeyboardButton("BE Move", callback_data="ui:settings:toggle:be")],
                        [InlineKeyboardButton("SL Mode", callback_data="ui:settings:toggle:sl_mode"), InlineKeyboardButton("Set SL Buffer", callback_data="ui:settings:set:sl_buffer")],
                        [InlineKeyboardButton("Set R:R", callback_data="ui:settings:set:rr"), InlineKeyboardButton("Set TP1 R", callback_data="ui:settings:set:tp1_r")],
                        [InlineKeyboardButton("Set TP2 R", callback_data="ui:settings:set:tp2_r"), InlineKeyboardButton("Set Fraction", callback_data="ui:settings:set:fraction")],
                        [InlineKeyboardButton("Set Confirm Bars", callback_data="ui:settings:set:confirm_bars"), InlineKeyboardButton("Set Phantom Hours", callback_data="ui:settings:set:phantom_hours")],
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
            elif data.startswith("ui:scalp:promote"):
                await query.answer()
                fake_update = type('obj', (object,), {'message': query.message})
                await self.scalp_promotion_status(fake_update, ctx)
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

    async def trend_high_ml(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set Trend high-ML execution threshold (also updates min_ml). Usage: /trendhighml 90"""
        try:
            if not ctx.args:
                await self.safe_reply(update, "Usage: /trendhighml <threshold>")
                return
            val = float(ctx.args[0])
            cfg = self.shared.get('config', {}) or {}
            tr_exec = ((cfg.setdefault('trend', {})).setdefault('exec', {}))
            old_hi = tr_exec.get('high_ml_force', None)
            old_min = tr_exec.get('min_ml', None)
            tr_exec['high_ml_force'] = val
            tr_exec['min_ml'] = min(val, tr_exec.get('min_ml', val))
            await self.safe_reply(update, f"✅ Trend high‑ML threshold updated: {old_hi} → {val} (min_ml: {old_min} → {tr_exec['min_ml']})")
        except Exception as exc:
            logger.warning(f"trend_high_ml failed: {exc}")
            await self.safe_reply(update, "⚠️ Failed to update Trend high‑ML threshold")

    async def trend_states(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
                from ml_scorer_scalp import get_scalp_scorer
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
            from multi_timeframe_sr import mtf_sr
            
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
            from multi_timeframe_sr import initialize_all_sr_levels
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

    async def mr_ml_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show detailed Mean Reversion ML statistics"""
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

    async def mr_retrain(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force retrain Mean Reversion ML models"""
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

    async def mr_phantom_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Mean Reversion phantom trade statistics"""
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
                from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary
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
                from enhanced_market_regime import get_enhanced_market_regime
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
                from phantom_trade_tracker import get_phantom_tracker
                from mr_phantom_tracker import get_mr_phantom_tracker
                from scalp_phantom_tracker import get_scalp_phantom_tracker
                from symbol_clustering import load_symbol_clusters
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
            from scalp_phantom_tracker import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            st = scpt.get_scalp_phantom_stats()
            total = st.get('total', 0)
            wins = st.get('wins', 0)
            losses = st.get('losses', 0)
            wr = st.get('wr', 0.0)
            # Scalp ML stats
            ml_ready = False; thr = 75; samples = 0; nxt = None; trainable = None; total_rec = None
            try:
                from ml_scorer_scalp import get_scalp_scorer
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
            msg = [
                "🩳 *Scalp QA*",
                f"• ML: {'✅ Ready' if ml_ready else '⏳ Training'} | Records: {total_rec if total_rec is not None else samples} | Trainable: {trainable if trainable is not None else '-'} | Thr: {thr}",
                f"• Next retrain in: {nxt if nxt is not None else '-'} trades",
                f"• Phantom recorded: {total} | WR: {wr:.1f}% (W/L {wins}/{losses})",
                f"• Shadow (ML-based): {int(sh_total)} | WR: {sh_wr:.1f}% (W/L {sh_w}/{sh_l})",
                "_Scalp runs phantom-only; shadow sim reflects ML decision quality_"
            ]
            await self.safe_reply(update, "\n".join(msg))
        except Exception as e:
            logger.error(f"Error in scalp_qa: {e}")
            await update.message.reply_text("Error getting scalp QA")

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
                from scalp_phantom_tracker import get_scalp_phantom_tracker
                scpt = get_scalp_phantom_tracker()
                st = scpt.get_scalp_phantom_stats()
            except Exception:
                st = {'total': 0, 'wins': 0, 'losses': 0, 'wr': 0.0}

            # Scalp scorer readiness
            ml_ready = False
            try:
                from ml_scorer_scalp import get_scalp_scorer
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
                from scalp_phantom_tracker import get_scalp_phantom_tracker
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

    async def trend_promotion_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
