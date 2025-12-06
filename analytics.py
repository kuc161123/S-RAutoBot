#!/usr/bin/env python3
"""
VWAP Bot Analytics Add-On
=========================
Standalone analytics tool that reads from the bot's Postgres database
to find patterns in wins and losses.

Usage:
    python analytics.py [--days 30] [--output report.md]
    
Requires: DATABASE_URL environment variable
"""

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️ pandas not installed. Some features disabled.")

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db_connection():
    """Connect to the bot's Postgres database"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not set. Export it or run with:")
        print("   DATABASE_URL='postgresql://...' python analytics.py")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_trade_history(conn, days=30):
    """Fetch trade history from the last N days"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT * FROM trade_history 
            WHERE created_at > NOW() - INTERVAL '%s days'
            ORDER BY created_at DESC
        """, (days,))
        return cur.fetchall()

def fetch_combo_stats(conn):
    """Fetch aggregated combo stats"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM combo_stats ORDER BY wins DESC")
        return cur.fetchall()

# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

def analyze_by_dimension(trades, dimension_fn, dimension_name):
    """Analyze win rate by a given dimension (e.g., day, hour, symbol)"""
    stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    
    for trade in trades:
        key = dimension_fn(trade)
        if trade['outcome'] == 'win':
            stats[key]['wins'] += 1
        else:
            stats[key]['losses'] += 1
    
    results = []
    for key, data in stats.items():
        total = data['wins'] + data['losses']
        wr = (data['wins'] / total * 100) if total > 0 else 0
        results.append({
            'key': key,
            'wins': data['wins'],
            'losses': data['losses'],
            'total': total,
            'wr': wr
        })
    
    return sorted(results, key=lambda x: x['wr'], reverse=True)

def analyze_by_day(trades):
    """Analyze win rate by day of week"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return analyze_by_dimension(
        trades,
        lambda t: days[t['created_at'].weekday()],
        'Day'
    )

def analyze_by_hour(trades):
    """Analyze win rate by hour of day"""
    return analyze_by_dimension(
        trades,
        lambda t: f"{t['created_at'].hour:02d}:00",
        'Hour'
    )

def analyze_by_symbol(trades):
    """Analyze win rate by symbol"""
    return analyze_by_dimension(trades, lambda t: t['symbol'], 'Symbol')

def analyze_by_side(trades):
    """Analyze win rate by side (long/short)"""
    return analyze_by_dimension(trades, lambda t: t['side'], 'Side')

def analyze_by_combo(trades):
    """Analyze win rate by combo pattern"""
    return analyze_by_dimension(trades, lambda t: t['combo'], 'Combo')

def analyze_rr_performance(trades):
    """Analyze counterfactual R:R performance"""
    rr_stats = {1.5: {'w': 0, 'l': 0}, 2.0: {'w': 0, 'l': 0}, 
                2.5: {'w': 0, 'l': 0}, 3.0: {'w': 0, 'l': 0}}
    
    for trade in trades:
        max_r = trade.get('max_r_reached', 0) or 0
        for rr in [1.5, 2.0, 2.5, 3.0]:
            if max_r >= rr:
                rr_stats[rr]['w'] += 1
            else:
                rr_stats[rr]['l'] += 1
    
    results = []
    for rr, data in rr_stats.items():
        total = data['w'] + data['l']
        if total > 0:
            wr = data['w'] / total * 100
            ev = (wr/100 * rr) - ((100-wr)/100 * 1.0)
            results.append({'rr': rr, 'wr': wr, 'ev': ev, 'total': total})
    
    return results

# ============================================================================
# PATTERN DISCOVERY
# ============================================================================

def find_winning_patterns(trades, min_trades=5, min_wr=50):
    """Find patterns that consistently win"""
    # Group by symbol+side+combo
    patterns = defaultdict(lambda: {'wins': 0, 'losses': 0, 'trades': []})
    
    for trade in trades:
        key = f"{trade['symbol']}|{trade['side']}|{trade['combo']}"
        if trade['outcome'] == 'win':
            patterns[key]['wins'] += 1
        else:
            patterns[key]['losses'] += 1
        patterns[key]['trades'].append(trade)
    
    winners = []
    for key, data in patterns.items():
        total = data['wins'] + data['losses']
        if total >= min_trades:
            wr = data['wins'] / total * 100
            if wr >= min_wr:
                symbol, side, combo = key.split('|')
                winners.append({
                    'symbol': symbol,
                    'side': side,
                    'combo': combo,
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'total': total,
                    'wr': wr
                })
    
    return sorted(winners, key=lambda x: (x['wr'], x['total']), reverse=True)

def find_losing_patterns(trades, min_trades=5, max_wr=40):
    """Find patterns that consistently lose"""
    patterns = defaultdict(lambda: {'wins': 0, 'losses': 0})
    
    for trade in trades:
        key = f"{trade['symbol']}|{trade['side']}|{trade['combo']}"
        if trade['outcome'] == 'win':
            patterns[key]['wins'] += 1
        else:
            patterns[key]['losses'] += 1
    
    losers = []
    for key, data in patterns.items():
        total = data['wins'] + data['losses']
        if total >= min_trades:
            wr = data['wins'] / total * 100
            if wr <= max_wr:
                symbol, side, combo = key.split('|')
                losers.append({
                    'symbol': symbol,
                    'side': side,
                    'combo': combo,
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'total': total,
                    'wr': wr
                })
    
    return sorted(losers, key=lambda x: x['wr'])

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(trades, days=30):
    """Generate comprehensive analytics report"""
    total = len(trades)
    wins = sum(1 for t in trades if t['outcome'] == 'win')
    losses = total - wins
    wr = (wins / total * 100) if total > 0 else 0
    
    report = []
    report.append(f"# 📊 VWAP Bot Analytics Report")
    report.append(f"")
    report.append(f"**Period**: Last {days} days")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"")
    report.append(f"## Overview")
    report.append(f"- **Total Trades**: {total}")
    report.append(f"- **Win Rate**: {wr:.1f}% ({wins}W / {losses}L)")
    report.append(f"")
    
    # Day of Week Analysis
    report.append(f"## 📅 Performance by Day")
    report.append(f"| Day | Win Rate | Trades |")
    report.append(f"|-----|----------|--------|")
    for item in analyze_by_day(trades):
        report.append(f"| {item['key']} | {item['wr']:.0f}% | {item['total']} |")
    report.append(f"")
    
    # Hour Analysis
    report.append(f"## ⏰ Performance by Hour (UTC)")
    report.append(f"| Hour | Win Rate | Trades |")
    report.append(f"|------|----------|--------|")
    for item in sorted(analyze_by_hour(trades), key=lambda x: x['key'])[:12]:
        if item['total'] >= 3:
            report.append(f"| {item['key']} | {item['wr']:.0f}% | {item['total']} |")
    report.append(f"")
    
    # Side Analysis
    report.append(f"## 📈 Performance by Side")
    for item in analyze_by_side(trades):
        icon = "🟢" if item['key'] == 'long' else "🔴"
        report.append(f"- {icon} **{item['key'].upper()}**: {item['wr']:.0f}% ({item['wins']}/{item['total']})")
    report.append(f"")
    
    # R:R Analysis
    report.append(f"## 💹 Optimal R:R Target")
    report.append(f"| R:R | Win Rate | EV | Trades |")
    report.append(f"|-----|----------|-----|--------|")
    for item in analyze_rr_performance(trades):
        report.append(f"| {item['rr']}:1 | {item['wr']:.0f}% | {item['ev']:+.2f}R | {item['total']} |")
    report.append(f"")
    
    # Best Patterns
    winners = find_winning_patterns(trades, min_trades=3, min_wr=50)[:10]
    if winners:
        report.append(f"## 🏆 Top Winning Patterns")
        report.append(f"| Symbol | Side | Combo | WR | Trades |")
        report.append(f"|--------|------|-------|-----|--------|")
        for p in winners:
            report.append(f"| {p['symbol']} | {p['side']} | {p['combo'][:20]} | {p['wr']:.0f}% | {p['total']} |")
        report.append(f"")
    
    # Worst Patterns
    losers = find_losing_patterns(trades, min_trades=3, max_wr=35)[:10]
    if losers:
        report.append(f"## 🚫 Worst Losing Patterns")
        report.append(f"| Symbol | Side | Combo | WR | Trades |")
        report.append(f"|--------|------|-------|-----|--------|")
        for p in losers:
            report.append(f"| {p['symbol']} | {p['side']} | {p['combo'][:20]} | {p['wr']:.0f}% | {p['total']} |")
        report.append(f"")
    
    return "\n".join(report)

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VWAP Bot Analytics')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze (default: 30)')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: print to console)')
    args = parser.parse_args()
    
    print(f"🔍 VWAP Bot Analytics")
    print(f"━━━━━━━━━━━━━━━━━━━━━━")
    
    conn = get_db_connection()
    print(f"✅ Connected to database")
    
    trades = fetch_trade_history(conn, days=args.days)
    print(f"📊 Found {len(trades)} trades in last {args.days} days")
    
    if len(trades) == 0:
        print("⚠️ No trades found. The bot needs to run and resolve signals first.")
        return
    
    report = generate_report(trades, days=args.days)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {args.output}")
    else:
        print("\n" + report)
    
    conn.close()

if __name__ == '__main__':
    main()
