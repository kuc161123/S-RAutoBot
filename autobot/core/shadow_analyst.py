"""
Shadow Analyst (Phase 1c) — READ-ONLY intelligence over shadow_signals.

Turns the shadow learner's raw observations into statistically honest views:
  - combo_edges():   per-combo cost-adjusted Wilson bounds + stability + neighbors
  - rolling_edge():  per-family edge over 7/21/60 day windows
  - weekly_r():      the flagship time-clustering signal (validated -30..-40R band)
  - rr_grid():       counterfactual RR table from mfe_to_sl
  - regime_edges():  edge by regime x BTC trend
  - proposals_check(): combos/families meeting Phase-2 evidence gates (REPORT ONLY)
  - alert_tick():    proactive Telegram alerts with cooldowns + daily cap

DESIGN CONTRACT: pure reads + the tiny shadow_alerts dedupe table. This module
must NEVER affect trading — it measures, it never gates. Global signals (weekly
R) are ADVISORY by validated design: an automatic global kill-switch was tested
and rejected (it clipped the jackpot months that pay for the whole system).

Statistical guards (why the thresholds look strict): naive per-combo
promote/blacklist was validated to OVERFIT (-873R OOS). Every verdict here
therefore requires cost adjustment + Wilson bounds + time-stability, and
negative verdicts additionally require neighborhood agreement.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from autobot.core.shadow_logger import wilson_lower_bound, wilson_upper_bound

logger = logging.getLogger(__name__)

ROUND_TRIP_COST = 0.0015          # fee 2x0.0006 + slippage ~0.0003, fraction of notional
WEEKLY_WARN_R = -25.0             # validated breaker band is -30..-40; alert slightly early
WEEKLY_CRIT_R = -35.0
ALERT_DAILY_CAP = 3
ALERT_COOLDOWNS_H = {             # per alert kind
    'weekly_r_warn': 48, 'weekly_r_crit': 24,
    'combo_negative': 14 * 24, 'family_candidate': 7 * 24,
    'learner_health': 24,
}


def _cost_r(entry, sl):
    """Round-trip fee+slippage expressed in R units for one signal."""
    try:
        risk = abs(entry - sl)
        if risk <= 0:
            return 0.0
        return ROUND_TRIP_COST * entry / risk
    except Exception:
        return 0.0


def _net_r(row):
    """Cost-adjusted R for a resolved row. expired counts 0R (conservative)."""
    if row['status'] == 'expired':
        return 0.0
    return (row['r_result'] or 0.0) - _cost_r(row['entry'], row['sl'])


class ShadowAnalyst:
    def __init__(self, shadow_logger):
        self.sl = shadow_logger
        self.enabled = bool(getattr(shadow_logger, 'enabled', False))
        self._last_alert_tick = 0.0
        if self.enabled:
            self.sl._safe(self._ensure_alert_table)

    # ---- infra ----
    def _rows(self, days=None, family=None, statuses=("win", "loss", "expired")):
        """Fetch resolved rows (small volumes; aggregate in Python)."""
        def _q():
            import psycopg2.extras
            where = ["status IN %s"]
            params = [tuple(statuses)]
            if days:
                where.append("ts > now() - interval '%s days'" % int(days))
            if family:
                where.append("family = %s")
                params.append(family)
            with self.sl._conn() as conn, conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, ts, sig_time_ms, symbol, side, div_type, in_universe,
                           entry, sl, tp, rr, atr_mult, regime, btc_bull, turnover,
                           executed, status, r_result, mfe, mae, mfe_to_sl,
                           bars_to_outcome, family, source, resolved_ts
                    FROM shadow_signals WHERE {' AND '.join(where)};
                """, params)
                return [dict(r) for r in cur.fetchall()]
        return self.sl._safe(_q, default=[]) or []

    @staticmethod
    def _outcome_ts(row):
        """Best-effort outcome time: signal bar + bars_to_outcome hours,
        falling back to resolved_ts (which is biased late)."""
        try:
            if row.get('bars_to_outcome') and row.get('sig_time_ms'):
                return datetime.fromtimestamp(
                    (row['sig_time_ms'] + row['bars_to_outcome'] * 3_600_000) / 1000,
                    tz=timezone.utc)
        except Exception:
            pass
        return row.get('resolved_ts')

    # ---- per-combo edges with the full statistical guard stack ----
    def combo_edges(self, min_n=30, family='DIV', in_universe=None):
        rows = [r for r in self._rows(family=family)
                if in_universe is None or r['in_universe'] == in_universe]
        groups = {}
        for r in rows:
            groups.setdefault((r['symbol'], r['side'], r['div_type']), []).append(r)
        # per-(div_type, side) negatives for neighborhood checks
        out = []
        NEIGHBOR_FLOOR = 20   # combos this size join the neighborhood pool
        for (sym, side, div), rs in groups.items():
            wl = [r for r in rs if r['status'] in ('win', 'loss')]
            n = len(wl)
            if n < NEIGHBOR_FLOOR:
                continue  # too small even as neighborhood evidence
            wins = sum(1 for r in wl if r['status'] == 'win')
            rr = sum(r['rr'] or 0 for r in wl) / n
            cost = sum(_cost_r(r['entry'], r['sl']) for r in wl) / n
            net = [_net_r(r) for r in rs]
            avg_net = sum(net) / len(net) if net else 0.0
            lb = wilson_lower_bound(wins, n)
            ub = wilson_upper_bound(wins, n)
            breakeven_c = (1.0 + cost) / (1.0 + rr) if rr > 0 else 1.0
            # time-half stability: both halves' net R share the sign of the total
            rs_sorted = sorted(rs, key=lambda r: r['sig_time_ms'] or 0)
            half = len(rs_sorted) // 2
            h1 = sum(_net_r(r) for r in rs_sorted[:half])
            h2 = sum(_net_r(r) for r in rs_sorted[half:])
            tot = h1 + h2
            stable = (h1 > 0 and h2 > 0) if tot > 0 else (h1 < 0 and h2 < 0)
            out.append(dict(symbol=sym, side=side, div_type=div, n=n, wins=wins,
                            wr=wins / n, lb=lb, ub=ub, rr=rr, cost_r=cost,
                            breakeven_c=breakeven_c, avg_net_r=avg_net,
                            sum_net_r=sum(net), stable=stable,
                            n_expired=len(rs) - n))
        # neighborhood agreement for negative verdicts
        by_symside = {}
        by_divside = {}
        for x in out:
            by_symside.setdefault((x['symbol'], x['side']), []).append(x)
            by_divside.setdefault((x['div_type'], x['side']), []).append(x)
        for x in out:
            neigh_neg = False
            sibs = [s for s in by_symside.get((x['symbol'], x['side']), [])
                    if s['div_type'] != x['div_type']]
            if any(s['avg_net_r'] < 0 for s in sibs):
                neigh_neg = True
            else:
                peers = [p for p in by_divside.get((x['div_type'], x['side']), [])
                         if p['symbol'] != x['symbol']]
                if sum(1 for p in peers if p['avg_net_r'] < 0) >= 3:
                    neigh_neg = True
            x['neighbors_negative'] = neigh_neg
            # verdict ladder
            if x['lb'] > x['breakeven_c'] and x['stable']:
                x['verdict'] = 'edge'
            elif x['wr'] > x['breakeven_c']:
                x['verdict'] = 'promising'
            elif x['ub'] < x['breakeven_c'] and x['stable'] and neigh_neg and x['n'] >= 60:
                x['verdict'] = 'condemn'          # confident negative
            elif x['avg_net_r'] < 0:
                x['verdict'] = 'negative'
            else:
                x['verdict'] = 'neutral'
        # small combos served as neighborhood evidence; report only n >= min_n
        return [x for x in out if x['n'] >= min_n]

    # ---- rolling per-family edge ----
    def rolling_edge(self, windows=(7, 21, 60)):
        rows = self._rows(days=max(windows))
        now = datetime.now(timezone.utc)
        buckets = {}
        for r in rows:
            fam = r.get('family') or 'DIV'
            key = f"{fam}{'‡' if not r['in_universe'] else ''}"
            buckets.setdefault(key, []).append(r)
        out = []
        for key, rs in sorted(buckets.items()):
            item = {'family': key}
            for w in windows:
                sub = []
                for r in rs:
                    ots = self._outcome_ts(r)
                    if ots is None:
                        continue
                    try:
                        age = (now - ots).total_seconds() / 86400
                    except TypeError:
                        continue
                    if age <= w:
                        sub.append(r)
                wl = [r for r in sub if r['status'] in ('win', 'loss')]
                net = [_net_r(r) for r in sub]
                gp = sum(x for x in net if x > 0)
                gl = -sum(x for x in net if x < 0)
                item[f'n_{w}d'] = len(wl)
                item[f'r_{w}d'] = round(sum(net), 1)
                item[f'pf_{w}d'] = round(gp / gl, 2) if gl > 0 else None
            # coverage span
            times = [r['sig_time_ms'] for r in rs if r['sig_time_ms']]
            item['span_days'] = round((max(times) - min(times)) / 86_400_000, 1) if times else 0
            item['n_total'] = len(rs)
            out.append(item)
        return out

    # ---- flagship: weekly shadow R (the validated clustering signal) ----
    def weekly_r(self, weeks=8):
        rows = self._rows(days=weeks * 7 + 10, family='DIV')
        rows = [r for r in rows if r['in_universe']]
        now = datetime.now(timezone.utc)
        series = {}
        t7 = t21 = 0.0
        for r in rows:
            ots = self._outcome_ts(r)
            if ots is None:
                continue
            try:
                age_d = (now - ots).total_seconds() / 86400
            except TypeError:
                continue
            nr = _net_r(r)
            iso = ots.isocalendar()
            wk = f"{iso[0]}-W{iso[1]:02d}"
            series[wk] = series.get(wk, 0.0) + nr
            if age_d <= 7:
                t7 += nr
            if age_d <= 21:
                t21 += nr
        if t7 <= WEEKLY_CRIT_R:
            state, suggested = 'CRIT', 0.5
        elif t7 <= WEEKLY_WARN_R:
            state, suggested = 'WARN', 0.7
        else:
            state, suggested = 'OK', 1.0
        return {'series': dict(sorted(series.items())[-weeks:]),
                'trailing_7d_r': round(t7, 1), 'trailing_21d_r': round(t21, 1),
                'state': state, 'suggested_mult': suggested,
                'warn_at': WEEKLY_WARN_R, 'crit_at': WEEKLY_CRIT_R}

    # ---- counterfactual RR from mfe_to_sl ----
    def rr_grid(self, rr_values=(3, 4, 5, 6, 8, 10), symbol=None, family='DIV'):
        rows = [r for r in self._rows(family=family)
                if r.get('mfe_to_sl') is not None
                and (symbol is None or r['symbol'] == symbol)]
        if not rows:
            return []
        out = []
        for rrp in rr_values:
            wins = sum(1 for r in rows if (r['mfe_to_sl'] or 0) >= rrp)
            n = len(rows)
            avg_cost = sum(_cost_r(r['entry'], r['sl']) for r in rows) / n
            wr = wins / n
            exp = wr * (rrp - avg_cost) - (1 - wr) * (1 + avg_cost)
            gp = wins * (rrp - avg_cost)
            gl = (n - wins) * (1 + avg_cost)
            out.append({'rr': rrp, 'n': n, 'wins': wins, 'wr': round(wr, 3),
                        'exp_r': round(exp, 3),
                        'pf': round(gp / gl, 2) if gl > 0 else None})
        return out

    # ---- regime-conditional breakdown ----
    def regime_edges(self):
        rows = [r for r in self._rows(family='DIV') if r['status'] in ('win', 'loss')]
        groups = {}
        for r in rows:
            key = (r.get('regime') or 'unknown',
                   'bull' if r.get('btc_bull') else 'bear/na')
            groups.setdefault(key, []).append(r)
        out = []
        for (regime, btc), rs in sorted(groups.items()):
            n = len(rs)
            wins = sum(1 for r in rs if r['status'] == 'win')
            net = [_net_r(r) for r in rs]
            gp = sum(x for x in net if x > 0)
            gl = -sum(x for x in net if x < 0)
            out.append({'regime': regime, 'btc': btc, 'n': n, 'wr': round(wins / n, 3),
                        'sum_r': round(sum(net), 1),
                        'pf': round(gp / gl, 2) if gl > 0 else None})
        return out

    # ---- Phase-2 evidence gates (REPORT ONLY here) ----
    def proposals_check(self):
        found = []
        # family promotion candidates
        for fam in self.rolling_edge(windows=(60,)):
            if fam['family'].startswith('DIV') and '‡' not in fam['family']:
                continue  # in-universe DIV is already live
            if (fam['span_days'] >= 60 and fam['n_60d'] >= 100
                    and (fam['pf_60d'] or 0) >= 1.3):
                found.append({'kind': 'promote_family', 'target': fam['family'],
                              'evidence': fam,
                              'note': 'meets shadow gates; ALSO requires offline backtest before capital'})
        # bench candidates (condemned combos on live configs)
        for x in self.combo_edges(min_n=60, family='DIV', in_universe=True):
            if x['verdict'] == 'condemn':
                found.append({'kind': 'bench_config',
                              'target': f"{x['symbol']} {x['side']} {x['div_type']}",
                              'evidence': {k: x[k] for k in
                                           ('n', 'wr', 'ub', 'breakeven_c', 'avg_net_r',
                                            'stable', 'neighbors_negative')},
                              'note': 'confident negative (UB<breakeven, stable, neighbors agree)'})
        return found

    # ---- alert engine ----
    def _ensure_alert_table(self):
        with self.sl._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shadow_alerts (
                    alert_key TEXT PRIMARY KEY,
                    last_sent TIMESTAMPTZ,
                    times_sent INT DEFAULT 0
                );
            """)
            conn.commit()

    def _alert_allowed(self, key, kind):
        def _q():
            cd_h = ALERT_COOLDOWNS_H.get(kind, 24)
            with self.sl._conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM shadow_alerts WHERE last_sent > now() - interval '1 day';")
                if cur.fetchone()[0] >= ALERT_DAILY_CAP:
                    return False
                cur.execute("SELECT last_sent FROM shadow_alerts WHERE alert_key=%s;", (key,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    age_h = (datetime.now(timezone.utc) - row[0]).total_seconds() / 3600
                    if age_h < cd_h:
                        return False
                return True
        return bool(self.sl._safe(_q, default=False))

    def _alert_mark(self, key):
        def _q():
            with self.sl._conn() as conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO shadow_alerts (alert_key, last_sent, times_sent)
                    VALUES (%s, now(), 1)
                    ON CONFLICT (alert_key) DO UPDATE
                    SET last_sent = now(), times_sent = shadow_alerts.times_sent + 1;
                """, (key,))
                conn.commit()
        self.sl._safe(_q)

    def alert_tick(self, alerts_enabled=True):
        """Evaluate all alert rules. Returns a list of Telegram-ready messages
        (the CALLER sends them). Internally throttled to ~hourly. Never raises."""
        if not self.enabled or not alerts_enabled:
            return []
        import time as _t
        now = _t.time()
        if now - self._last_alert_tick < 3300:
            return []
        self._last_alert_tick = now
        try:
            return self._alert_tick()
        except Exception as e:
            logger.debug(f"[SHADOW-ANALYST] alert_tick failed: {e}")
            return []

    def _alert_tick(self):
        msgs = []
        # 1) weekly shadow R (advisory — the validated clustering signal)
        wk = self.weekly_r()
        if wk['trailing_7d_r'] <= WEEKLY_CRIT_R and self._alert_allowed('weekly_r_crit', 'weekly_r_crit'):
            self._alert_mark('weekly_r_crit')
            msgs.append(
                f"🔴 **SHADOW WEEK CRITICAL** (advisory)\n"
                f"├ 7d shadow R: {wk['trailing_7d_r']:+.1f}R (crit ≤ {WEEKLY_CRIT_R:.0f})\n"
                f"├ Losses are clustering — the validated week-brake band is hit\n"
                f"└ Consider /stop or reduced risk. Details: /learn week")
        elif wk['trailing_7d_r'] <= WEEKLY_WARN_R and self._alert_allowed('weekly_r_warn', 'weekly_r_warn'):
            self._alert_mark('weekly_r_warn')
            msgs.append(
                f"🟠 **SHADOW WEEK COLD** (advisory)\n"
                f"├ 7d shadow R: {wk['trailing_7d_r']:+.1f}R (warn ≤ {WEEKLY_WARN_R:.0f})\n"
                f"└ Edge is cold this week. Details: /learn week")
        # 2) condemned combos on live configs
        for x in self.combo_edges(min_n=60, family='DIV', in_universe=True):
            if x['verdict'] != 'condemn':
                continue
            key = f"combo_negative:{x['symbol']}:{x['side']}:{x['div_type']}"
            if self._alert_allowed(key, 'combo_negative'):
                self._alert_mark(key)
                msgs.append(
                    f"🚫 **CONFIG DRAG** (evidence-backed)\n"
                    f"├ {x['symbol']} {x['side']} {x['div_type']}\n"
                    f"├ {x['n']} signals · WR {x['wr']*100:.0f}% · UB {x['ub']*100:.0f}% "
                    f"< breakeven {x['breakeven_c']*100:.0f}%\n"
                    f"├ Stable negative in both time halves, neighbors agree\n"
                    f"└ Bench candidate — /learn sym {x['symbol']}")
        # 3) family promotion candidates
        for p in self.proposals_check():
            if p['kind'] != 'promote_family':
                continue
            key = f"family_candidate:{p['target']}"
            if self._alert_allowed(key, 'family_candidate'):
                self._alert_mark(key)
                ev = p['evidence']
                msgs.append(
                    f"🌱 **FAMILY CANDIDATE** (shadow track record)\n"
                    f"├ {p['target']}: PF {ev.get('pf_60d')} over {ev.get('n_60d')} signals / 60d\n"
                    f"├ Meets shadow gates — still needs offline backtest\n"
                    f"└ Details: /learn family")
        # 4) learner health
        def _health():
            with self.sl._conn() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT count(*), min(ts) FROM shadow_signals
                    WHERE status='pending' AND ts < now() - interval '48 hours';
                """)
                return cur.fetchone()
        h = self.sl._safe(_health)
        if h and (h[0] or 0) > 50 and self._alert_allowed('learner_health', 'learner_health'):
            self._alert_mark('learner_health')
            msgs.append(
                f"🩺 **SHADOW LEARNER BACKLOG**\n"
                f"├ {h[0]} signals pending >48h (oldest {h[1]:%Y-%m-%d})\n"
                f"└ Resolution may be falling behind — check logs")
        return msgs
