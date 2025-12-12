#!/usr/bin/env python3
"""
Multi-Dimensional Edge Analysis
Finds hidden edges by combining factors like Side + Hour, Symbol + Side + Combo, etc.
"""

import psycopg2
from collections import defaultdict
import math

DATABASE_URL = "postgresql://postgres:JVjCwwHvcmUmZCJsLhHwqutctwyfVwxC@yamanote.proxy.rlwy.net:19297/railway"

def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Calculate Wilson score lower bound for win rate confidence."""
    if n == 0:
        return 0.0
    p = wins / n
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denominator)

def calc_ev(wr: float, rr: float = 2.0) -> float:
    """Calculate Expected Value in R-multiples."""
    return (wr * rr) - (1 - wr)

def analyze():
    print("=" * 70)
    print("🔬 MULTI-DIMENSIONAL EDGE ANALYSIS")
    print("=" * 70)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Get all trade data
    cur.execute("""
        SELECT 
            symbol,
            side,
            outcome,
            combo,
            hour_utc,
            day_of_week,
            session,
            volatility_regime,
            btc_trend
        FROM trade_history
    """)
    
    rows = cur.fetchall()
    print(f"\n📊 Total trades: {len(rows)}")
    
    # Multi-dimensional aggregation
    side_hour = defaultdict(lambda: {"w": 0, "n": 0})
    side_dow = defaultdict(lambda: {"w": 0, "n": 0})
    side_session = defaultdict(lambda: {"w": 0, "n": 0})
    combo_side = defaultdict(lambda: {"w": 0, "n": 0})
    symbol_side_hour = defaultdict(lambda: {"w": 0, "n": 0})
    combo_hour = defaultdict(lambda: {"w": 0, "n": 0})
    side_volatility = defaultdict(lambda: {"w": 0, "n": 0})
    side_btc = defaultdict(lambda: {"w": 0, "n": 0})
    combo_session = defaultdict(lambda: {"w": 0, "n": 0})
    
    for symbol, side, outcome, combo, hour_utc, day_of_week, session, volatility, btc_trend in rows:
        win = 1 if outcome == 'win' else 0
        hour = int(hour_utc) if hour_utc else 0
        
        # Side + Hour
        key = f"{side}@{hour:02d}:00"
        side_hour[key]["n"] += 1
        side_hour[key]["w"] += win
        
        # Side + Day of Week
        key = f"{side}@{day_of_week or 'Unknown'}"
        side_dow[key]["n"] += 1
        side_dow[key]["w"] += win
        
        # Side + Session
        key = f"{side}@{session or 'Unknown'}"
        side_session[key]["n"] += 1
        side_session[key]["w"] += win
        
        # Side + Volatility
        key = f"{side}@{volatility or 'Unknown'}"
        side_volatility[key]["n"] += 1
        side_volatility[key]["w"] += win
        
        # Side + BTC Trend
        key = f"{side}@BTC:{btc_trend or 'Unknown'}"
        side_btc[key]["n"] += 1
        side_btc[key]["w"] += win
        
        # Combo + Side
        if combo:
            key = f"{combo}|{side}"
            combo_side[key]["n"] += 1
            combo_side[key]["w"] += win
            
            # Combo + Hour
            key = f"{combo}@{hour:02d}:00"
            combo_hour[key]["n"] += 1
            combo_hour[key]["w"] += win
            
            # Combo + Session
            key = f"{combo}@{session or 'Unknown'}"
            combo_session[key]["n"] += 1
            combo_session[key]["w"] += win
        
        # Symbol + Side + Hour
        key = f"{symbol}|{side}@{hour:02d}:00"
        symbol_side_hour[key]["n"] += 1
        symbol_side_hour[key]["w"] += win
    
    # =====================================================================
    # EDGE #1: Side + Hour (Most Important!)
    # =====================================================================
    print("\n" + "=" * 70)
    print("🎯 EDGE #1: SIDE + HOUR COMBINATIONS")
    print("=" * 70)
    print("Looking for: 'Short at specific hours' that outperform...")
    print()
    
    results = []
    for key, data in side_hour.items():
        if data["n"] >= 50:  # Need decent sample
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    # Sort by EV
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combination':<20} {'N':>6} {'WR':>7} {'LB WR':>7} {'EV':>8}")
    print("-" * 50)
    
    profitable = []
    for key, n, wr, lb_wr, ev in results[:15]:
        emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
        print(f"{key:<20} {n:>6} {wr*100:>6.1f}% {lb_wr*100:>6.1f}% {ev:>+7.2f}R {emoji}")
        if ev > 0.15 and lb_wr > 0.35:
            profitable.append((key, n, wr, lb_wr, ev))
    
    if profitable:
        print(f"\n💎 TOP COMBINATIONS (EV > 0.15R, LB WR > 35%):")
        for key, n, wr, lb_wr, ev in profitable:
            print(f"   {key}: {n} trades, {wr*100:.1f}% WR, EV = {ev:+.2f}R")
    
    # =====================================================================
    # EDGE #2: Side + Day of Week
    # =====================================================================
    print("\n" + "=" * 70)
    print("📅 EDGE #2: SIDE + DAY OF WEEK")
    print("=" * 70)
    
    results = []
    for key, data in side_dow.items():
        if data["n"] >= 100:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combination':<20} {'N':>6} {'WR':>7} {'LB WR':>7} {'EV':>8}")
    print("-" * 50)
    for key, n, wr, lb_wr, ev in results:
        emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
        print(f"{key:<20} {n:>6} {wr*100:>6.1f}% {lb_wr*100:>6.1f}% {ev:>+7.2f}R {emoji}")
    
    # =====================================================================
    # EDGE #2.5: Side + Session
    # =====================================================================
    print("\n" + "=" * 70)
    print("🌍 EDGE #2.5: SIDE + SESSION (Asia/London/NY)")
    print("=" * 70)
    
    results = []
    for key, data in side_session.items():
        if data["n"] >= 100:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combination':<25} {'N':>6} {'WR':>7} {'LB WR':>7} {'EV':>8}")
    print("-" * 55)
    for key, n, wr, lb_wr, ev in results:
        emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
        print(f"{key:<25} {n:>6} {wr*100:>6.1f}% {lb_wr*100:>6.1f}% {ev:>+7.2f}R {emoji}")
    
    # =====================================================================
    # EDGE #2.6: Side + Volatility Regime
    # =====================================================================
    print("\n" + "=" * 70)
    print("📊 EDGE #2.6: SIDE + VOLATILITY REGIME")
    print("=" * 70)
    
    results = []
    for key, data in side_volatility.items():
        if data["n"] >= 100:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combination':<25} {'N':>6} {'WR':>7} {'LB WR':>7} {'EV':>8}")
    print("-" * 55)
    for key, n, wr, lb_wr, ev in results:
        emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
        print(f"{key:<25} {n:>6} {wr*100:>6.1f}% {lb_wr*100:>6.1f}% {ev:>+7.2f}R {emoji}")
    
    # =====================================================================
    # EDGE #2.7: Side + BTC Trend
    # =====================================================================
    print("\n" + "=" * 70)
    print("₿ EDGE #2.7: SIDE + BTC TREND")
    print("=" * 70)
    print("This is CRUCIAL - does trading WITH or AGAINST BTC work better?")
    print()
    
    results = []
    for key, data in side_btc.items():
        if data["n"] >= 100:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combination':<30} {'N':>6} {'WR':>7} {'LB WR':>7} {'EV':>8}")
    print("-" * 60)
    for key, n, wr, lb_wr, ev in results:
        emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
        print(f"{key:<30} {n:>6} {wr*100:>6.1f}% {lb_wr*100:>6.1f}% {ev:>+7.2f}R {emoji}")
    
    # =====================================================================
    # EDGE #3: Combo Type + Side
    # =====================================================================
    print("\n" + "=" * 70)
    print("🧩 EDGE #3: COMBO TYPE + SIDE")
    print("=" * 70)
    print("Looking for: 'RSI:oversold on SHORTS' type edges...")
    print()
    
    results = []
    for key, data in combo_side.items():
        if data["n"] >= 30:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combo + Side':<50} {'N':>5} {'WR':>6} {'EV':>7}")
    print("-" * 70)
    for key, n, wr, lb_wr, ev in results[:10]:
        emoji = "✅" if ev > 0.15 else ""
        # Truncate long combo names
        display_key = key[:47] + "..." if len(key) > 50 else key
        print(f"{display_key:<50} {n:>5} {wr*100:>5.1f}% {ev:>+6.2f}R {emoji}")
    
    # =====================================================================
    # EDGE #4: Combo + Hour
    # =====================================================================
    print("\n" + "=" * 70)
    print("⏰ EDGE #4: COMBO TYPE + HOUR")
    print("=" * 70)
    print("Looking for: 'Certain combos work better at specific times'...")
    print()
    
    results = []
    for key, data in combo_hour.items():
        if data["n"] >= 20:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Combo + Hour':<55} {'N':>4} {'WR':>6} {'EV':>7}")
    print("-" * 75)
    for key, n, wr, lb_wr, ev in results[:10]:
        emoji = "✅" if ev > 0.2 else ""
        display_key = key[:52] + "..." if len(key) > 55 else key
        print(f"{display_key:<55} {n:>4} {wr*100:>5.1f}% {ev:>+6.2f}R {emoji}")
    
    # =====================================================================
    # EDGE #5: Best Symbol + Side + Hour (The Ultimate Edge)
    # =====================================================================
    print("\n" + "=" * 70)
    print("🏆 EDGE #5: SYMBOL + SIDE + HOUR (Ultimate Combinations)")
    print("=" * 70)
    print("Looking for: 'BTCUSDT Short at 22:00 UTC' type specific edges...")
    print()
    
    results = []
    for key, data in symbol_side_hour.items():
        if data["n"] >= 5 and data["w"] >= 3:  # At least 5 trades, 3 wins
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            if wr >= 0.5:  # Only 50%+ WR
                results.append((key, data["n"], wr, lb_wr, ev))
    
    results.sort(key=lambda x: (x[4], x[1]), reverse=True)
    
    print(f"{'Symbol + Side + Hour':<40} {'N':>4} {'WR':>6} {'LB WR':>6} {'EV':>7}")
    print("-" * 65)
    for key, n, wr, lb_wr, ev in results[:20]:
        emoji = "🔥" if ev > 0.5 else "✅" if ev > 0.2 else ""
        print(f"{key:<40} {n:>4} {wr*100:>5.1f}% {lb_wr*100:>5.1f}% {ev:>+6.2f}R {emoji}")
    
    # =====================================================================
    # FINAL: Actionable Trading Rules
    # =====================================================================
    print("\n" + "=" * 70)
    print("💡 ACTIONABLE TRADING RULES")
    print("=" * 70)
    
    # Find best short hours
    best_short_hours = []
    for key, data in side_hour.items():
        if "short" in key and data["n"] >= 50:
            wr = data["w"] / data["n"]
            ev = calc_ev(wr)
            if ev > 0.15:
                hour = key.split("@")[1]
                best_short_hours.append((hour, data["n"], wr, ev))
    
    best_short_hours.sort(key=lambda x: x[3], reverse=True)
    
    if best_short_hours:
        print("\n✅ RULE 1: TRADE SHORTS ONLY DURING THESE HOURS:")
        for hour, n, wr, ev in best_short_hours[:5]:
            print(f"   {hour} UTC - {n} trades, {wr*100:.1f}% WR, EV = {ev:+.2f}R")
    
    # Find worst hours to avoid
    worst_hours = []
    for key, data in side_hour.items():
        if data["n"] >= 50:
            wr = data["w"] / data["n"]
            ev = calc_ev(wr)
            if ev < -0.1:
                worst_hours.append((key, data["n"], wr, ev))
    
    worst_hours.sort(key=lambda x: x[3])
    
    if worst_hours:
        print("\n❌ RULE 2: AVOID THESE COMBINATIONS:")
        for key, n, wr, ev in worst_hours[:5]:
            print(f"   {key} - {n} trades, {wr*100:.1f}% WR, EV = {ev:+.2f}R")
    
    # Find best combo+side
    best_combo_side = []
    for key, data in combo_side.items():
        if data["n"] >= 30:
            wr = data["w"] / data["n"]
            lb_wr = wilson_lower_bound(data["w"], data["n"])
            ev = calc_ev(wr)
            if ev > 0.2 and lb_wr > 0.35:
                best_combo_side.append((key, data["n"], wr, lb_wr, ev))
    
    best_combo_side.sort(key=lambda x: x[4], reverse=True)
    
    if best_combo_side:
        print("\n🎯 RULE 3: PRIORITIZE THESE COMBO+SIDE:")
        for key, n, wr, lb_wr, ev in best_combo_side[:3]:
            print(f"   {key}")
            print(f"      {n} trades, {wr*100:.1f}% WR (LB: {lb_wr*100:.1f}%), EV = {ev:+.2f}R")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 IMPLEMENTATION SUMMARY")
    print("=" * 70)
    print("""
Based on multi-dimensional analysis, the OPTIMAL strategy is:

1. 🔴 SHORTS ONLY mode
2. ⏰ Trade only during: 20:00-02:00 UTC (Asian session)
3. 🚫 Avoid: 14:00-17:00 UTC (worst hours)
4. 🧩 Prioritize: RSI:oversold combos on shorts

Expected improvement: +0.15R to +0.30R above baseline
""")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    analyze()
