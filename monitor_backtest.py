#!/usr/bin/env python3
"""Live monitoring script for 400-symbol backtest"""
import os
import re
import time
from datetime import datetime
import subprocess

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def parse_log():
    log_file = 'backtest_400.log'
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    processed = 0
    passed = 0
    failed = 0
    start_time = None
    latest_symbol = None
    recent_results = []
    
    for line in lines:
        if 'Started:' in line:
            try:
                start_time = datetime.strptime(line.split('Started: ')[1].strip(), '%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        if re.search(r'\[\d+/400\]', line):
            match = re.search(r'\[(\d+)/400\] (\w+)', line)
            if match:
                processed = int(match.group(1))
                latest_symbol = match.group(2)
        
        if '✅' in line and '[' in line:
            passed += 1
            recent_results.append(line.strip())
        elif '⚠️' in line and '[' in line:
            failed += 1
            recent_results.append(line.strip())
    
    return {
        'processed': processed,
        'passed': passed,
        'failed': failed,
        'start_time': start_time,
        'latest_symbol': latest_symbol,
        'recent': recent_results[-10:]  # Last 10 results
    }

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def display_progress(data):
    if not data:
        print("❌ Backtest log not found")
        return
    
    total = 400
    processed = data['processed']
    passed = data['passed']
    failed = data['failed']
    remaining = total - processed
    progress_pct = (processed / total * 100) if total > 0 else 0
    pass_rate = (passed / processed * 100) if processed > 0 else 0
    
    clear_screen()
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🔬 LIVE BACKTEST MONITOR" + " " * 35 + "║")
    print("╠" + "═" * 78 + "╣")
    
    # Time info
    if data['start_time']:
        elapsed = (datetime.now() - data['start_time']).total_seconds()
        avg_per_symbol = elapsed / processed if processed > 0 else 0
        remaining_seconds = avg_per_symbol * remaining
        
        print(f"║ Started: {data['start_time'].strftime('%H:%M:%S')}" + " " * 20 + 
              f"Elapsed: {format_time(elapsed)}" + " " * (27 - len(format_time(elapsed))) + "║")
        print(f"║ ETA: {format_time(remaining_seconds)}" + " " * 29 + 
              f"Speed: {60/avg_per_symbol:.1f} sym/min" + " " * (20 - len(f"{60/avg_per_symbol:.1f}")) + "║")
    
    print("╠" + "═" * 78 + "╣")
    
    # Progress bar
    bar_width = 50
    filled = int(bar_width * processed / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"║ Progress: [{bar}] {progress_pct:.1f}%" + " " * (25 - len(f"{progress_pct:.1f}%")) + "║")
    print(f"║ {processed}/{total} symbols" + " " * (66 - len(f"{processed}/{total} symbols")) + "║")
    
    print("╠" + "═" * 78 + "╣")
    
    # Stats
    print(f"║ ✅ Passed:    {passed:3d} symbols ({pass_rate:.1f}% success rate)" + 
          " " * (35 - len(f"{passed:3d} symbols ({pass_rate:.1f}% success rate)")) + "║")
    print(f"║ ⚠️  Failed:    {failed:3d} symbols" + " " * (53 - len(f"{failed:3d} symbols")) + "║")
    print(f"║ 🎯 Remaining: {remaining:3d} symbols" + " " * (53 - len(f"{remaining:3d} symbols")) + "║")
    
    print("╠" + "═" * 78 + "╣")
    
    # Latest symbol
    if data['latest_symbol']:
        print(f"║ Current: {data['latest_symbol']}" + " " * (67 - len(data['latest_symbol'])) + "║")
    
    # Recent results
    print("╠" + "═" * 78 + "╣")
    print("║ Recent Results:" + " " * 63 + "║")
    for result in data['recent'][-5:]:
        # Extract just the symbol and result
        if '✅' in result:
            parts = result.split(']')
            if len(parts) > 1:
                info = parts[1].strip()
                display = info[:74]  # Truncate if too long
                print(f"║ {display}" + " " * (77 - len(display)) + "║")
        elif '⚠️' in result:
            parts = result.split(']')
            if len(parts) > 1:
                info = parts[1].strip()
                display = info[:74]
                print(f"║ {display}" + " " * (77 - len(display)) + "║")
    
    print("╚" + "═" * 78 + "╝")
    print("\nPress Ctrl+C to exit monitor (backtest continues in background)")

def main():
    print("Starting live monitor...")
    print("Refreshing every 5 seconds...")
    time.sleep(2)
    
    try:
        while True:
            data = parse_log()
            display_progress(data)
            
            # Check if complete
            if data and data['processed'] >= 400:
                print("\n🎉 BACKTEST COMPLETE!")
                print(f"Results: {data['passed']} symbols passed validation")
                break
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped. Backtest continues in background.")
        print("Run this script again to resume monitoring.")

if __name__ == "__main__":
    main()
