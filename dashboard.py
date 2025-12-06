#!/usr/bin/env python3
"""
AutoBot Dashboard - Web interface for monitoring the trading bot
"""

from flask import Flask, render_template_string, jsonify
import yaml
import os
import re
from datetime import datetime
from collections import Counter

app = Flask(__name__)

# Dashboard HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoBot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 30px 0;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header .status {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 20px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 20px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 212, 255, 0.2);
        }
        
        .card-title {
            font-size: 1.2em;
            color: #00d4ff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-title .icon {
            font-size: 1.5em;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .stat-box {
            background: rgba(0, 212, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #00ff88;
        }
        
        .stat-label {
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }
        
        .combo-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .combo-table th, .combo-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .combo-table th {
            color: #00d4ff;
            font-weight: 600;
        }
        
        .combo-table tr:hover {
            background: rgba(0, 212, 255, 0.1);
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .badge-long {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }
        
        .badge-short {
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
        }
        
        .combo-tag {
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            background: rgba(123, 44, 191, 0.2);
            border: 1px solid rgba(123, 44, 191, 0.5);
            border-radius: 6px;
            font-size: 0.75em;
            font-family: monospace;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            border-radius: 10px;
            transition: width 0.5s;
        }
        
        .analytics-chart {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        
        .chart-bar {
            flex: 1;
            min-width: 60px;
            text-align: center;
        }
        
        .chart-bar-fill {
            height: var(--height);
            background: linear-gradient(180deg, #00d4ff, #7b2cbf);
            border-radius: 5px 5px 0 0;
            margin-bottom: 5px;
            min-height: 20px;
        }
        
        .chart-bar-label {
            font-size: 0.7em;
            color: #888;
        }
        
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #00d4ff, #7b2cbf);
            border: none;
            border-radius: 50%;
            color: white;
            font-size: 1.5em;
            cursor: pointer;
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
            transition: transform 0.3s;
        }
        
        .refresh-btn:hover {
            transform: scale(1.1);
        }
        
        .card.full-width {
            grid-column: 1 / -1;
        }
        
        .last-update {
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AutoBot Dashboard</h1>
        <div class="status">
            <span class="status-dot"></span>
            <span>VWAP Combo Strategy</span>
        </div>
        <div style="margin-top: 20px;">
            <a href="/analytics" style="display: inline-block; padding: 10px 20px; background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 20px; color: #00d4ff; text-decoration: none; transition: all 0.3s;">
                📊 View Analytics
            </a>
        </div>
    </div>
    
    <div class="grid">
        <!-- Stats Overview -->
        <div class="card">
            <div class="card-title"><span class="icon">📊</span> Overview</div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{{ stats.total_symbols }}</div>
                    <div class="stat-label">Active Symbols</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ stats.total_combos }}</div>
                    <div class="stat-label">Total Combos</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ stats.long_combos }}</div>
                    <div class="stat-label">Long Combos</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ stats.short_combos }}</div>
                    <div class="stat-label">Short Combos</div>
                </div>
            </div>
        </div>
        
        <!-- RSI Distribution -->
        <div class="card">
            <div class="card-title"><span class="icon">📈</span> RSI Distribution</div>
            <div class="analytics-chart">
                {% for rsi, count in analytics.rsi.items() %}
                <div class="chart-bar">
                    <div class="chart-bar-fill" style="--height: {{ (count / analytics.max_rsi * 100)|int }}px"></div>
                    <div class="chart-bar-label">{{ rsi }}<br>{{ count }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- MACD Distribution -->
        <div class="card">
            <div class="card-title"><span class="icon">📉</span> MACD Trend</div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value" style="color: #00ff88">{{ analytics.macd.bull }}</div>
                    <div class="stat-label">Bullish</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #ff6b6b">{{ analytics.macd.bear }}</div>
                    <div class="stat-label">Bearish</div>
                </div>
            </div>
        </div>
        
        <!-- Fib Distribution -->
        <div class="card">
            <div class="card-title"><span class="icon">🎯</span> Fib Zones</div>
            <div class="analytics-chart">
                {% for fib, count in analytics.fib.items() %}
                <div class="chart-bar">
                    <div class="chart-bar-fill" style="--height: {{ (count / analytics.max_fib * 100)|int }}px"></div>
                    <div class="chart-bar-label">{{ fib }}<br>{{ count }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Top Performers -->
        <div class="card full-width">
            <div class="card-title"><span class="icon">🏆</span> Top Performers (30d)</div>
            <table class="combo-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Combo</th>
                        <th>WR</th>
                        <th>EV</th>
                        <th>Trades</th>
                    </tr>
                </thead>
                <tbody>
                    {% for p in top_performers %}
                    <tr>
                        <td><strong>{{ p.symbol }}</strong></td>
                        <td>
                            <span class="badge {{ 'badge-long' if p.side == 'long' else 'badge-short' }}">
                                {{ p.side|upper }}
                            </span>
                        </td>
                        <td><span class="combo-tag">{{ p.combo }}</span></td>
                        <td style="color: #00ff88; font-weight: bold;">{{ p.wr|round(0) }}%</td>
                        <td>{{ p.ev|default(0)|round(2) }}R</td>
                        <td>{{ p.total }}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="6" style="text-align: center; color: #666;">No trade data yet</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Active Combos Table -->
        <div class="card full-width">
            <div class="card-title"><span class="icon">🎰</span> Active Combos by Symbol</div>
            <table class="combo-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Combos</th>
                    </tr>
                </thead>
                <tbody>
                    {% for symbol, data in combos.items() %}
                    {% if data.long %}
                    <tr>
                        <td><strong>{{ symbol }}</strong></td>
                        <td><span class="badge badge-long">LONG</span></td>
                        <td>
                            {% for combo in data.long %}
                            <span class="combo-tag">{{ combo }}</span>
                            {% endfor %}
                        </td>
                    </tr>
                    {% endif %}
                    {% if data.short %}
                    <tr>
                        <td><strong>{{ symbol }}</strong></td>
                        <td><span class="badge badge-short">SHORT</span></td>
                        <td>
                            {% for combo in data.short %}
                            <span class="combo-tag">{{ combo }}</span>
                            {% endfor %}
                        </td>
                    </tr>
                    {% endif %}
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="last-update">
        Last updated: {{ last_update }}
    </div>
    
    <button class="refresh-btn" onclick="location.reload()">🔄</button>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

def load_combos():
    """Load combos from YAML file"""
    try:
        with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def calculate_stats(combos):
    """Calculate overview statistics"""
    total_symbols = len(combos)
    long_combos = sum(len(data.get('long', [])) for data in combos.values())
    short_combos = sum(len(data.get('short', [])) for data in combos.values())
    
    return {
        'total_symbols': total_symbols,
        'total_combos': long_combos + short_combos,
        'long_combos': long_combos,
        'short_combos': short_combos
    }

def analyze_combos(combos):
    """Analyze combo distributions"""
    rsi_counter = Counter()
    macd_counter = Counter()
    fib_counter = Counter()
    
    # Parse all combos
    for symbol, data in combos.items():
        all_combos = data.get('long', []) + data.get('short', [])
        for combo in all_combos:
            # Parse RSI:X MACD:Y Fib:Z format
            parts = combo.split()
            for part in parts:
                if part.startswith('RSI:'):
                    rsi_counter[part.replace('RSI:', '')] += 1
                elif part.startswith('MACD:'):
                    macd_counter[part.replace('MACD:', '')] += 1
                elif part.startswith('Fib:'):
                    fib_counter[part.replace('Fib:', '')] += 1
    
    # Sort RSI by level order
    rsi_order = ['<30', '30-40', '40-60', '60-70', '70+']
    rsi_sorted = {k: rsi_counter.get(k, 0) for k in rsi_order}
    
    # Sort Fib by level order
    fib_order = ['0-23', '23-38', '38-50', '50-61', '61-78', '78-100', '100+']
    fib_sorted = {k: fib_counter.get(k, 0) for k in fib_order}
    
    return {
        'rsi': rsi_sorted,
        'max_rsi': max(rsi_sorted.values()) if rsi_sorted.values() else 1,
        'macd': {
            'bull': macd_counter.get('bull', 0),
            'bear': macd_counter.get('bear', 0)
        },
        'fib': fib_sorted,
        'max_fib': max(fib_sorted.values()) if fib_sorted.values() else 1
    }

# Analytics HTML Template
ANALYTICS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoBot Analytics</title>
    <style>
        /* Reuse main dashboard styles */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .header { text-align: center; padding: 30px 0; margin-bottom: 30px; }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #ff00cc, #333399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .nav-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            color: white;
            text-decoration: none;
            margin-top: 10px;
            transition: background 0.3s;
        }
        .nav-btn:hover { background: rgba(255, 255, 255, 0.2); }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
        }
        .card-title { font-size: 1.2em; color: #ff00cc; margin-bottom: 20px; }
        .full-width { grid-column: 1 / -1; }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
        th { color: #ff00cc; }
        
        .bar-container { display: flex; align-items: center; gap: 10px; }
        .bar { height: 8px; background: #333399; border-radius: 4px; flex-grow: 1; }
        .bar-fill { height: 100%; background: linear-gradient(90deg, #ff00cc, #333399); border-radius: 4px; }
        
        .win { color: #00ff88; }
        .loss { color: #ff6b6b; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Performance Analytics</h1>
        <a href="/" class="nav-btn">← Back to Dashboard</a>
    </div>
    
    <div class="grid">
        <!-- Overview -->
        <div class="card">
            <div class="card-title">📈 Overview (Last {{ days }} Days)</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; text-align: center;">
                <div>
                    <div style="font-size: 2.5em; color: {{ 'white' if wr >= 50 else '#ff6b6b' }}">{{ wr|round(1) }}%</div>
                    <div style="color: #888">Win Rate</div>
                </div>
                <div>
                    <div style="font-size: 2.5em;">{{ total }}</div>
                    <div style="color: #888">Total Trades</div>
                </div>
            </div>
        </div>
        
        <!-- Day Performance -->
        <div class="card">
            <div class="card-title">📅 Best Days</div>
            <table>
                <tr><th>Day</th><th>WR</th><th>Trades</th></tr>
                {% for day in days_stats %}
                <tr>
                    <td>{{ day.key }}</td>
                    <td class="{{ 'win' if day.wr >= 50 else 'loss' }}">{{ day.wr|round(0) }}%</td>
                    <td>{{ day.total }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <!-- Hour Performance -->
        <div class="card">
            <div class="card-title">⏰ Best Hours (UTC)</div>
            <table>
                <tr><th>Hour</th><th>WR</th><th>Trades</th></tr>
                {% for hour in hours_stats %}
                {% if hour.total >= 3 %}
                <tr>
                    <td>{{ hour.key }}</td>
                    <td class="{{ 'win' if hour.wr >= 50 else 'loss' }}">{{ hour.wr|round(0) }}%</td>
                    <td>{{ hour.total }}</td>
                </tr>
                {% endif %}
                {% endfor %}
            </table>
        </div>
        
        <!-- R:R Performance -->
        <div class="card">
            <div class="card-title">🎯 Optimal R:R</div>
            <table>
                <tr><th>Target</th><th>WR</th><th>EV</th></tr>
                {% for rr in rr_stats %}
                <tr>
                    <td>{{ rr.rr }}R</td>
                    <td>{{ rr.wr|round(0) }}%</td>
                    <td class="{{ 'win' if rr.ev > 0 else 'loss' }}">{{ rr.ev|round(2) }}R</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <!-- Winning Patterns -->
        <div class="card full-width">
            <div class="card-title">🏆 Top Winning Patterns</div>
            <table>
                <tr><th>Symbol</th><th>Side</th><th>Combo</th><th>WR</th><th>Trades</th></tr>
                {% for p in winners %}
                <tr>
                    <td>{{ p.symbol }}</td>
                    <td>{{ p.side|upper }}</td>
                    <td style="font-family: monospace; font-size: 0.9em;">{{ p.combo }}</td>
                    <td class="win">{{ p.wr|round(0) }}%</td>
                    <td>{{ p.total }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <!-- Losing Patterns -->
        <div class="card full-width">
            <div class="card-title">🚫 Worst Losing Patterns</div>
            <table>
                <tr><th>Symbol</th><th>Side</th><th>Combo</th><th>WR</th><th>Trades</th></tr>
                {% for p in losers %}
                <tr>
                    <td>{{ p.symbol }}</td>
                    <td>{{ p.side|upper }}</td>
                    <td style="font-family: monospace; font-size: 0.9em;">{{ p.combo }}</td>
                    <td class="loss">{{ p.wr|round(0) }}%</td>
                    <td>{{ p.total }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
</body>
</html>
"""

@app.route('/analytics')
def analytics_view():
    try:
        from analytics import get_db_connection, fetch_trade_history, analyze_by_day, analyze_by_hour, analyze_rr_performance, find_winning_patterns, find_losing_patterns
        
        conn = get_db_connection()
        trades = fetch_trade_history(conn, days=30)
        conn.close()
        
        total = len(trades)
        wins = sum(1 for t in trades if t['outcome'] == 'win')
        wr = (wins / total * 100) if total > 0 else 0
        
        return render_template_string(
            ANALYTICS_HTML,
            days=30,
            total=total,
            wr=wr,
            days_stats=analyze_by_day(trades),
            hours_stats=analyze_by_hour(trades),
            rr_stats=analyze_rr_performance(trades),
            winners=find_winning_patterns(trades, min_trades=3),
            losers=find_losing_patterns(trades, min_trades=3)
        )
    except Exception as e:
        return f"<h1>Analytics Error</h1><p>{e}</p><p>Make sure DATABASE_URL is set and analytics.py is present.</p>"

@app.route('/')
def dashboard():
    combos = load_combos()
    stats = calculate_stats(combos)
    analytics = analyze_combos(combos)
    
    # Fetch top performers from DB
    top_performers = []
    try:
        from analytics import get_db_connection, fetch_trade_history, find_winning_patterns
        conn = get_db_connection()
        trades = fetch_trade_history(conn, days=30)
        conn.close()
        
        # Get top 5 winners with at least 3 trades
        winners = find_winning_patterns(trades, min_trades=3)
        top_performers = winners[:5]
        
        # Calculate EV for display (simplified)
        for p in top_performers:
            # Assuming 2:1 RR for now as default, or we could calculate actual EV if we had it
            # analytics.py find_winning_patterns returns 'wr' and 'total'
            # Let's estimate EV based on 2:1 RR for display purposes if not present
            if 'ev' not in p:
                wr_dec = p['wr'] / 100
                p['ev'] = (wr_dec * 2.0) - ((1 - wr_dec) * 1.0)
                
    except Exception as e:
        print(f"Failed to fetch top performers: {e}")
    
    return render_template_string(
        DASHBOARD_HTML,
        combos=combos,
        stats=stats,
        analytics=analytics,
        top_performers=top_performers,
        last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route('/api/combos')
def api_combos():
    """API endpoint for combos data"""
    combos = load_combos()
    return jsonify(combos)

@app.route('/api/stats')
def api_stats():
    """API endpoint for stats"""
    combos = load_combos()
    stats = calculate_stats(combos)
    analytics = analyze_combos(combos)
    return jsonify({
        'stats': stats,
        'analytics': analytics
    })

if __name__ == '__main__':
    print("🚀 Starting AutoBot Dashboard...")
    print("📊 Open http://localhost:8888 in your browser")
    app.run(host='0.0.0.0', port=8888, debug=False)
