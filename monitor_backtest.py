#!/usr/bin/env python3
"""
Real-time monitor for backtest progress
Shows live updates of which symbol is being processed and results
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_info():
    """Get info about running backtest processes"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        processes = []
        for line in lines:
            if 'backtest_pro_rules_full.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9],
                        'command': ' '.join(parts[10:])
                    })
        return processes
    except:
        return []

def get_results_info():
    """Get information from results file - returns (mod_time, count, results_dict, valid_count)"""
    results_file = 'backtest_pro_rules_results.json'
    if not os.path.exists(results_file):
        return None, 0, {}, 0
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Get file modification time
        mtime = os.path.getmtime(results_file)
        mod_time = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
        
        # Count valid configs
        valid_count = sum(1 for v in data.values() if v.get('long') or v.get('short'))
        
        return mod_time, len(data), data, valid_count
    except Exception as e:
        # Return 4 values even on error
        return None, 0, {}, 0

def get_progress_info():
    """Get information from progress file"""
    progress_file = 'backtest_pro_rules_progress.json'
    if not os.path.exists(progress_file):
        return None
    
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except:
        return None

def format_size(size_bytes):
    """Format file size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def main():
    print("Starting real-time backtest monitor...")
    print("Press Ctrl+C to stop monitoring\n")
    time.sleep(2)
    
    last_results_count = 0
    last_mod_time = None
    
    try:
        while True:
            clear_screen()
            
            print("=" * 80)
            print("BACKTEST PRO RULES - REAL-TIME MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Process status
            print("1. PROCESS STATUS")
            print("-" * 80)
            processes = get_process_info()
            if processes:
                print(f"   ✓ Script is RUNNING ({len(processes)} process(es))")
                for p in processes:
                    print(f"   PID: {p['pid']} | CPU: {p['cpu']}% | MEM: {p['mem']}% | Time: {p['time']}")
            else:
                print("   ✗ Script is NOT running")
            print()
            
            # Results file
            print("2. RESULTS FILE")
            print("-" * 80)
            mod_time, count, results, valid_count = get_results_info()
            
            if mod_time:
                file_size = os.path.getsize('backtest_pro_rules_results.json')
                print(f"   ✓ File exists: {format_size(file_size)}")
                print(f"   Last updated: {mod_time}")
                print(f"   Symbols completed: {count}")
                print(f"   Symbols with valid configs (WR >= 40%): {valid_count}")
                
                # Show if new symbols were added
                if count > last_results_count:
                    print(f"   🎉 NEW: {count - last_results_count} symbol(s) completed!")
                    last_results_count = count
                
                # Show recent symbols
                if results:
                    print(f"\n   Recent symbols:")
                    for symbol in list(results.keys())[-5:]:
                        result = results[symbol]
                        status = "✓" if result.get('long') or result.get('short') else "✗"
                        long_wr = result.get('long', {}).get('win_rate', 0) if result.get('long') else 0
                        short_wr = result.get('short', {}).get('win_rate', 0) if result.get('short') else 0
                        print(f"     {status} {symbol}: LONG={long_wr:.1f}% SHORT={short_wr:.1f}%")
            else:
                print("   ✗ No results file yet (still processing first symbol)")
                print("   This is normal - first symbol takes longest (~30-60 min)")
            print()
            
            # Progress file
            print("3. PROGRESS")
            print("-" * 80)
            progress = get_progress_info()
            if progress:
                current = progress.get('current', 0)
                total = progress.get('total', 0)
                completed = progress.get('completed', [])
                valid = progress.get('valid_count', 0)
                
                if total > 0:
                    pct = (current / total) * 100
                    print(f"   Progress: {current}/{total} symbols ({pct:.1f}%)")
                    print(f"   Valid configs found: {valid}")
                    
                    if completed:
                        print(f"   Last completed: {completed[-1] if completed else 'None'}")
            else:
                print("   No progress file yet")
            print()
            
            # Estimated time
            print("4. ESTIMATED TIME")
            print("-" * 80)
            if mod_time and count > 0:
                # Rough estimate: if we have results, estimate based on time per symbol
                elapsed = time.time() - os.path.getmtime('backtest_pro_rules_results.json')
                if count > 0:
                    time_per_symbol = elapsed / count
                    remaining = 50 - count  # Assuming ~50 symbols
                    if remaining > 0:
                        est_minutes = (time_per_symbol * remaining) / 60
                        print(f"   Avg time per symbol: ~{time_per_symbol/60:.1f} minutes")
                        print(f"   Estimated remaining: ~{est_minutes:.1f} minutes ({est_minutes/60:.1f} hours)")
            else:
                print("   Still processing first symbol...")
                print("   First symbol typically takes 30-60 minutes")
            print()
            
            print("=" * 80)
            print("Refreshing every 5 seconds... (Ctrl+C to stop)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()

