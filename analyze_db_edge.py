#!/usr/bin/env python3
"""
DATABASE EDGE FINDER

Connects to the live PostgreSQL database and analyzes VWAP trade data
to find edges in: Sessions, Sides, Combos, Symbols, Time of Day
"""

import psycopg2
import math
from collections import defaultdict
from datetime import datetime

DATABASE_URL = "postgresql://postgres:JVjCwwHvcmUmZCJsLhHwqutctwyfVwxC@yamanote.proxy.rlwy.net:19297/railway"

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def calculate_ev(wins, total, rr=2.0):
    if total == 0:
        return 0
    wr = wins / total
    return (wr * rr) - ((1 - wr) * 1.0)

def analyze_database():
    print("="*70)
    print("🔍 DATABASE EDGE FINDER - VWAP STRATEGY")
    print("="*70)
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("✅ Connected to database")
        
        # First, let's see what tables exist
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        print(f"\n📋 Tables found: {[t[0] for t in tables]}")
        
        # Check if there's a trades or signals table
        for table in tables:
            table_name = table[0]
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"   {table_name}: {count} rows")
            
            # Show columns
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """)
            columns = cur.fetchall()
            print(f"      Columns: {[c[0] for c in columns]}")
        
        # Try to find the main trade/signal table and analyze it
        # Common names: trades, signals, phantom_trades, learner_signals, etc.
        possible_tables = ['trades', 'signals', 'phantom_trades', 'learner_signals', 
                          'trade_history', 'signal_history', 'learning_signals']
        
        main_table = None
        for t in possible_tables:
            if t in [table[0] for table in tables]:
                main_table = t
                break
        
        if not main_table:
            # Use the first table with the most columns
            main_table = tables[0][0] if tables else None
        
        if main_table:
            print(f"\n📊 Analyzing table: {main_table}")
            
            # Get sample data
            cur.execute(f"SELECT * FROM {main_table} LIMIT 5")
            sample = cur.fetchall()
            
            # Get column names
            cur.execute(f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{main_table}'
            """)
            columns = [c[0] for c in cur.fetchall()]
            print(f"   Columns: {columns}")
            
            # Analyze based on available columns
            analyze_table(cur, main_table, columns)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

def analyze_table(cur, table_name, columns):
    """Analyze a table for edges."""
    
    print("\n" + "="*70)
    print(f"📈 EDGE ANALYSIS: {table_name}")
    print("="*70)
    
    # Try to find outcome column (win/loss)
    outcome_cols = ['outcome', 'result', 'status', 'is_win', 'won']
    outcome_col = None
    for c in outcome_cols:
        if c in columns:
            outcome_col = c
            break
    
    # Try to find symbol column
    symbol_cols = ['symbol', 'sym', 'pair', 'asset']
    symbol_col = None
    for c in symbol_cols:
        if c in columns:
            symbol_col = c
            break
    
    # Try to find side column
    side_cols = ['side', 'direction', 'type', 'position']
    side_col = None
    for c in side_cols:
        if c in columns:
            side_col = c
            break
    
    # Try to find combo column
    combo_cols = ['combo', 'combo_key', 'signal_type', 'strategy']
    combo_col = None
    for c in combo_cols:
        if c in columns:
            combo_col = c
            break
    
    # Try to find timestamp
    time_cols = ['timestamp', 'created_at', 'time', 'date', 'resolved_at', 'entry_time']
    time_col = None
    for c in time_cols:
        if c in columns:
            time_col = c
            break
    
    print(f"\n   Detected columns:")
    print(f"   - Outcome: {outcome_col}")
    print(f"   - Symbol: {symbol_col}")
    print(f"   - Side: {side_col}")
    print(f"   - Combo: {combo_col}")
    print(f"   - Time: {time_col}")
    
    # === OVERALL STATS ===
    if outcome_col:
        # Check what values outcome can have
        cur.execute(f"SELECT DISTINCT {outcome_col} FROM {table_name}")
        outcomes = [o[0] for o in cur.fetchall()]
        print(f"\n   Outcome values: {outcomes}")
        
        # Try to determine win condition
        win_values = ['win', 'won', 'profit', 'tp', 'take_profit', True, 1, 'W', 'w']
        win_condition = None
        for wv in win_values:
            if wv in outcomes:
                win_condition = wv
                break
        
        if win_condition is None and 'win' in str(outcomes).lower():
            win_condition = [o for o in outcomes if 'win' in str(o).lower()][0] if any('win' in str(o).lower() for o in outcomes) else None
        
        if win_condition:
            print(f"   Win value detected: {win_condition}")
            
            # Total stats
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                FROM {table_name}
            """, (win_condition,))
            total, wins = cur.fetchone()
            wins = wins or 0
            
            wr = wins / total * 100 if total > 0 else 0
            lb_wr = wilson_lower_bound(wins, total)
            ev = calculate_ev(wins, total)
            
            print(f"\n📊 OVERALL STATS")
            print(f"   Total trades: {total}")
            print(f"   Wins: {wins}")
            print(f"   Win Rate: {wr:.1f}%")
            print(f"   LB Win Rate: {lb_wr:.1f}%")
            print(f"   EV (2:1 R:R): {ev:+.2f}R")
            
            # === BY SIDE ===
            if side_col:
                print("\n" + "-"*50)
                print("📈 BY SIDE")
                cur.execute(f"""
                    SELECT 
                        {side_col},
                        COUNT(*) as total,
                        SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                    FROM {table_name}
                    GROUP BY {side_col}
                    ORDER BY COUNT(*) DESC
                """, (win_condition,))
                
                for row in cur.fetchall():
                    side, total, wins = row
                    wins = wins or 0
                    if total >= 5:
                        wr = wins / total * 100
                        lb = wilson_lower_bound(wins, total)
                        ev = calculate_ev(wins, total)
                        icon = "✅" if lb >= 33 else "❌"
                        print(f"   {str(side):10} | N={total:5} | WR={wr:5.1f}% | LB={lb:5.1f}% | EV={ev:+.2f}R {icon}")
            
            # === BY SYMBOL (TOP 20) ===
            if symbol_col:
                print("\n" + "-"*50)
                print("💰 TOP 20 SYMBOLS (by EV)")
                cur.execute(f"""
                    SELECT 
                        {symbol_col},
                        COUNT(*) as total,
                        SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                    FROM {table_name}
                    GROUP BY {symbol_col}
                    HAVING COUNT(*) >= 10
                    ORDER BY SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END)::float / COUNT(*) DESC
                    LIMIT 20
                """, (win_condition, win_condition))
                
                results = []
                for row in cur.fetchall():
                    sym, total, wins = row
                    wins = wins or 0
                    wr = wins / total * 100
                    lb = wilson_lower_bound(wins, total)
                    ev = calculate_ev(wins, total)
                    results.append((sym, total, wins, wr, lb, ev))
                
                results.sort(key=lambda x: x[5], reverse=True)
                for i, (sym, total, wins, wr, lb, ev) in enumerate(results[:20], 1):
                    icon = "✅" if lb >= 33 else "❌"
                    print(f"   {i:2}. {str(sym):15} | N={total:4} | WR={wr:5.1f}% | LB={lb:5.1f}% | EV={ev:+.2f}R {icon}")
            
            # === BY COMBO (TOP 15) ===
            if combo_col:
                print("\n" + "-"*50)
                print("🎰 TOP 15 COMBOS (by EV)")
                cur.execute(f"""
                    SELECT 
                        {combo_col},
                        COUNT(*) as total,
                        SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                    FROM {table_name}
                    GROUP BY {combo_col}
                    HAVING COUNT(*) >= 10
                    ORDER BY SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END)::float / COUNT(*) DESC
                    LIMIT 15
                """, (win_condition, win_condition))
                
                results = []
                for row in cur.fetchall():
                    combo, total, wins = row
                    wins = wins or 0
                    wr = wins / total * 100
                    lb = wilson_lower_bound(wins, total)
                    ev = calculate_ev(wins, total)
                    results.append((combo, total, wins, wr, lb, ev))
                
                results.sort(key=lambda x: x[5], reverse=True)
                for i, (combo, total, wins, wr, lb, ev) in enumerate(results[:15], 1):
                    icon = "✅" if lb >= 33 else "❌"
                    print(f"   {i:2}. {str(combo)[:40]:40} | N={total:4} | WR={wr:5.1f}% | LB={lb:5.1f}% | EV={ev:+.2f}R {icon}")
            
            # === BY SYMBOL + SIDE ===
            if symbol_col and side_col:
                print("\n" + "-"*50)
                print("🎯 TOP 20 SYMBOL + SIDE COMBOS (by EV)")
                cur.execute(f"""
                    SELECT 
                        {symbol_col}, {side_col},
                        COUNT(*) as total,
                        SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                    FROM {table_name}
                    GROUP BY {symbol_col}, {side_col}
                    HAVING COUNT(*) >= 5
                    ORDER BY SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END)::float / COUNT(*) DESC
                    LIMIT 20
                """, (win_condition, win_condition))
                
                results = []
                for row in cur.fetchall():
                    sym, side, total, wins = row
                    wins = wins or 0
                    wr = wins / total * 100
                    lb = wilson_lower_bound(wins, total)
                    ev = calculate_ev(wins, total)
                    results.append((sym, side, total, wins, wr, lb, ev))
                
                results.sort(key=lambda x: x[6], reverse=True)
                for i, (sym, side, total, wins, wr, lb, ev) in enumerate(results[:20], 1):
                    icon = "✅" if lb >= 33 else "❌"
                    side_icon = "🟢" if str(side).lower() == 'long' else "🔴"
                    print(f"   {i:2}. {side_icon} {str(sym):12} | N={total:4} | WR={wr:5.1f}% | LB={lb:5.1f}% | EV={ev:+.2f}R {icon}")
            
            # === BY HOUR (if timestamp available) ===
            if time_col:
                print("\n" + "-"*50)
                print("⏰ BY HOUR (UTC)")
                try:
                    cur.execute(f"""
                        SELECT 
                            EXTRACT(HOUR FROM {time_col}::timestamp) as hour,
                            COUNT(*) as total,
                            SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                        FROM {table_name}
                        WHERE {time_col} IS NOT NULL
                        GROUP BY EXTRACT(HOUR FROM {time_col}::timestamp)
                        HAVING COUNT(*) >= 10
                        ORDER BY EXTRACT(HOUR FROM {time_col}::timestamp)
                    """, (win_condition,))
                    
                    for row in cur.fetchall():
                        hour, total, wins = row
                        wins = wins or 0
                        wr = wins / total * 100
                        lb = wilson_lower_bound(wins, total)
                        ev = calculate_ev(wins, total)
                        icon = "✅" if lb >= 33 else "❌"
                        print(f"   {int(hour):02d}:00 UTC | N={total:4} | WR={wr:5.1f}% | LB={lb:5.1f}% | EV={ev:+.2f}R {icon}")
                except Exception as e:
                    print(f"   ⚠️ Could not parse timestamp: {e}")
            
            # === ACTIONABLE INSIGHTS ===
            print("\n" + "="*70)
            print("💡 ACTIONABLE INSIGHTS")
            print("="*70)
            
            # Find profitable combos
            if symbol_col and side_col:
                cur.execute(f"""
                    SELECT 
                        {symbol_col}, {side_col},
                        COUNT(*) as total,
                        SUM(CASE WHEN {outcome_col} = %s THEN 1 ELSE 0 END) as wins
                    FROM {table_name}
                    GROUP BY {symbol_col}, {side_col}
                    HAVING COUNT(*) >= 10
                """, (win_condition,))
                
                profitable = []
                for row in cur.fetchall():
                    sym, side, total, wins = row
                    wins = wins or 0
                    lb = wilson_lower_bound(wins, total)
                    ev = calculate_ev(wins, total)
                    if lb >= 45:
                        profitable.append((sym, side, total, lb, ev))
                
                if profitable:
                    profitable.sort(key=lambda x: x[4], reverse=True)
                    print(f"\n   🏆 READY TO PROMOTE (LB WR >= 45%):")
                    for sym, side, total, lb, ev in profitable[:10]:
                        print(f"      {sym} {side} | LB={lb:.1f}% | EV={ev:+.2f}R")
                else:
                    print(f"\n   ⚠️ No combos meet promotion criteria (LB WR >= 45%)")
                    print(f"      Need more trades or better performance")
        else:
            print(f"   ⚠️ Could not detect win condition from outcomes: {outcomes}")
    else:
        print("   ⚠️ No outcome column found")

if __name__ == "__main__":
    analyze_database()
