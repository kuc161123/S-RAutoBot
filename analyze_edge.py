#!/usr/bin/env python3
"""
LIVE DATA EDGE FINDER

Analyzes the bot's current learner data to find profitable patterns and edges.
Looks at: Sessions, Sides, Combos, Symbols, Time of Day, etc.
"""

import json
import os
import math
from collections import defaultdict
from datetime import datetime

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def calculate_ev_2to1(wins, total):
    if total == 0:
        return 0
    wr = wins / total
    return (wr * 2.0) - ((1 - wr) * 1.0)

def load_learner_data():
    """Load learner data from JSON state file."""
    state_file = 'autobot/core/unified_learner_state.json'
    
    if not os.path.exists(state_file):
        print(f"❌ State file not found: {state_file}")
        return None
    
    with open(state_file, 'r') as f:
        data = json.load(f)
    
    return data

def analyze_data():
    print("="*70)
    print("🔍 LIVE DATA EDGE FINDER")
    print("="*70)
    
    data = load_learner_data()
    
    if not data:
        print("No data found. Is the bot running?")
        return
    
    combo_stats = data.get('combo_stats', {})
    promoted = data.get('promoted', [])
    blacklist = data.get('blacklist', [])
    
    print(f"\n📊 DATA SUMMARY")
    print(f"   Total symbols: {len(combo_stats)}")
    print(f"   Promoted combos: {len(promoted)}")
    print(f"   Blacklisted: {len(blacklist)}")
    
    # Aggregate stats
    all_combos = []
    session_stats = {'asian': {'w': 0, 'l': 0}, 'london': {'w': 0, 'l': 0}, 'newyork': {'w': 0, 'l': 0}}
    side_stats = {'long': {'w': 0, 'l': 0}, 'short': {'w': 0, 'l': 0}}
    combo_type_stats = defaultdict(lambda: {'w': 0, 'l': 0})
    symbol_stats = defaultdict(lambda: {'w': 0, 'l': 0})
    
    for symbol, sides in combo_stats.items():
        for side, combos in sides.items():
            for combo, stats in combos.items():
                wins = stats.get('wins', 0)
                losses = stats.get('losses', 0)
                total = wins + losses
                
                if total >= 3:  # Minimum for analysis
                    wr = wins / total * 100
                    lb_wr = wilson_lower_bound(wins, total)
                    ev = calculate_ev_2to1(wins, total)
                    
                    all_combos.append({
                        'symbol': symbol,
                        'side': side,
                        'combo': combo,
                        'wins': wins,
                        'losses': losses,
                        'total': total,
                        'wr': wr,
                        'lb_wr': lb_wr,
                        'ev': ev
                    })
                    
                    # Aggregate by side
                    side_stats[side]['w'] += wins
                    side_stats[side]['l'] += losses
                    
                    # Aggregate by symbol
                    symbol_stats[symbol]['w'] += wins
                    symbol_stats[symbol]['l'] += losses
                    
                    # Aggregate by combo type
                    combo_type_stats[combo]['w'] += wins
                    combo_type_stats[combo]['l'] += losses
                    
                    # Sessions
                    for session, sdata in stats.get('sessions', {}).items():
                        if session in session_stats:
                            session_stats[session]['w'] += sdata.get('w', 0)
                            session_stats[session]['l'] += sdata.get('l', 0)
    
    print(f"   Total combos tracked: {len(all_combos)}")
    
    # === EDGE #1: SESSION ANALYSIS ===
    print("\n" + "="*70)
    print("🌍 EDGE #1: SESSION ANALYSIS")
    print("="*70)
    
    for session, stats in session_stats.items():
        total = stats['w'] + stats['l']
        if total > 0:
            wr = stats['w'] / total * 100
            lb_wr = wilson_lower_bound(stats['w'], total)
            ev = calculate_ev_2to1(stats['w'], total)
            icon = "✅" if lb_wr >= 33 else "❌"
            session_name = {"asian": "🌏 Asian", "london": "🌍 London", "newyork": "🌎 New York"}[session]
            print(f"   {session_name:15} | N={total:5} | WR={wr:5.1f}% | LB={lb_wr:5.1f}% | EV={ev:+.2f}R {icon}")
    
    # === EDGE #2: SIDE ANALYSIS ===
    print("\n" + "="*70)
    print("📈 EDGE #2: SIDE ANALYSIS")
    print("="*70)
    
    for side, stats in side_stats.items():
        total = stats['w'] + stats['l']
        if total > 0:
            wr = stats['w'] / total * 100
            lb_wr = wilson_lower_bound(stats['w'], total)
            ev = calculate_ev_2to1(stats['w'], total)
            icon = "✅" if lb_wr >= 33 else "❌"
            side_icon = "🟢 Long" if side == "long" else "🔴 Short"
            print(f"   {side_icon:12} | N={total:5} | WR={wr:5.1f}% | LB={lb_wr:5.1f}% | EV={ev:+.2f}R {icon}")
    
    # === EDGE #3: COMBO TYPE ANALYSIS ===
    print("\n" + "="*70)
    print("🎰 EDGE #3: COMBO TYPE ANALYSIS (by indicator combo)")
    print("="*70)
    
    combo_list = []
    for combo, stats in combo_type_stats.items():
        total = stats['w'] + stats['l']
        if total >= 10:
            wr = stats['w'] / total * 100
            lb_wr = wilson_lower_bound(stats['w'], total)
            ev = calculate_ev_2to1(stats['w'], total)
            combo_list.append({
                'combo': combo, 'total': total, 'wins': stats['w'],
                'wr': wr, 'lb_wr': lb_wr, 'ev': ev
            })
    
    combo_list.sort(key=lambda x: x['ev'], reverse=True)
    
    print(f"\n   🏆 TOP 10 COMBOS (by EV):")
    for i, c in enumerate(combo_list[:10], 1):
        icon = "✅" if c['lb_wr'] >= 33 else "⚠️" if c['lb_wr'] >= 25 else "❌"
        print(f"   {i:2}. {c['combo']}")
        print(f"       N={c['total']:4} | WR={c['wr']:5.1f}% | LB={c['lb_wr']:5.1f}% | EV={c['ev']:+.2f}R {icon}")
    
    print(f"\n   ❌ WORST 5 COMBOS:")
    for i, c in enumerate(combo_list[-5:], 1):
        print(f"   {i:2}. {c['combo']}")
        print(f"       N={c['total']:4} | WR={c['wr']:5.1f}% | LB={c['lb_wr']:5.1f}% | EV={c['ev']:+.2f}R ❌")
    
    # === EDGE #4: TOP SYMBOLS ===
    print("\n" + "="*70)
    print("💰 EDGE #4: BEST PERFORMING SYMBOLS")
    print("="*70)
    
    symbol_list = []
    for symbol, stats in symbol_stats.items():
        total = stats['w'] + stats['l']
        if total >= 10:
            wr = stats['w'] / total * 100
            lb_wr = wilson_lower_bound(stats['w'], total)
            ev = calculate_ev_2to1(stats['w'], total)
            symbol_list.append({
                'symbol': symbol, 'total': total, 'wins': stats['w'],
                'wr': wr, 'lb_wr': lb_wr, 'ev': ev
            })
    
    symbol_list.sort(key=lambda x: x['ev'], reverse=True)
    
    print(f"\n   🏆 TOP 15 SYMBOLS:")
    profitable_symbols = []
    for i, s in enumerate(symbol_list[:15], 1):
        icon = "✅" if s['lb_wr'] >= 33 else "⚠️" if s['lb_wr'] >= 25 else "❌"
        print(f"   {i:2}. {s['symbol']:15} | N={s['total']:4} | WR={s['wr']:5.1f}% | LB={s['lb_wr']:5.1f}% | EV={s['ev']:+.2f}R {icon}")
        if s['lb_wr'] >= 33:
            profitable_symbols.append(s['symbol'])
    
    # === EDGE #5: TOP INDIVIDUAL COMBOS ===
    print("\n" + "="*70)
    print("🎯 EDGE #5: BEST INDIVIDUAL COMBOS (Symbol + Side + Combo)")
    print("="*70)
    
    all_combos.sort(key=lambda x: x['ev'], reverse=True)
    
    print(f"\n   🏆 TOP 20 (Positive EV):")
    for i, c in enumerate(all_combos[:20], 1):
        if c['ev'] > 0:
            side_icon = "🟢" if c['side'] == "long" else "🔴"
            icon = "✅" if c['lb_wr'] >= 33 else "⚠️"
            print(f"   {i:2}. {side_icon} {c['symbol']:12} {c['combo'][:25]}")
            print(f"       N={c['total']:3} | WR={c['wr']:5.1f}% | LB={c['lb_wr']:5.1f}% | EV={c['ev']:+.2f}R {icon}")
    
    # === ACTIONABLE INSIGHTS ===
    print("\n" + "="*70)
    print("💡 ACTIONABLE INSIGHTS")
    print("="*70)
    
    # Find best session
    best_session = max(session_stats.items(), 
                       key=lambda x: wilson_lower_bound(x[1]['w'], x[1]['w'] + x[1]['l']) if x[1]['w'] + x[1]['l'] > 0 else 0)
    
    # Find best side
    best_side = max(side_stats.items(),
                    key=lambda x: wilson_lower_bound(x[1]['w'], x[1]['w'] + x[1]['l']) if x[1]['w'] + x[1]['l'] > 0 else 0)
    
    # Top combos with positive EV and good N
    promotable = [c for c in all_combos if c['lb_wr'] >= 45 and c['total'] >= 10]
    near_promotable = [c for c in all_combos if c['lb_wr'] >= 35 and c['lb_wr'] < 45 and c['total'] >= 10]
    
    print(f"\n   1️⃣ BEST SESSION: {best_session[0].upper()}")
    print(f"      Consider filtering trades to only trade during this session")
    
    print(f"\n   2️⃣ BEST SIDE: {best_side[0].upper()}")
    print(f"      Consider biasing toward {best_side[0]} positions")
    
    print(f"\n   3️⃣ PROMOTION READY (LB WR >= 45%, N >= 10): {len(promotable)}")
    for c in promotable[:5]:
        print(f"      - {c['symbol']} {c['side']} | LB={c['lb_wr']:.1f}% | N={c['total']}")
    
    print(f"\n   4️⃣ NEAR PROMOTION (LB WR 35-45%, N >= 10): {len(near_promotable)}")
    for c in near_promotable[:5]:
        print(f"      - {c['symbol']} {c['side']} | LB={c['lb_wr']:.1f}% | N={c['total']}")
    
    # Combo types to avoid
    bad_combos = [c for c in combo_list if c['ev'] < -0.3]
    print(f"\n   5️⃣ COMBOS TO AVOID (EV < -0.3R):")
    for c in bad_combos[:5]:
        print(f"      - {c['combo']} | EV={c['ev']:+.2f}R")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    analyze_data()
