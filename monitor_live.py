#!/usr/bin/env python3
"""
Live Bot Monitor
Reads bot_output.log and displays real-time status.
"""
import time
import os
import re
from datetime import datetime
from collections import deque

LOG_FILE = 'bot_output.log'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_file_tail(filepath, n=1000):
    """Read last n lines of file"""
    try:
        # Simple implementation for tail
        # For very large files, seek is better, but this is fine for now
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except Exception:
        return []

def parse_logs():
    lines = get_file_tail(LOG_FILE, 2000)
    
    stats = {
        'heartbeat': None,
        'ws_connected': False,
        'active_symbols': set(),
        'phantoms': deque(maxlen=5),
        'executions': deque(maxlen=5),
        'errors': deque(maxlen=5),
        'balance': 'Unknown',
        'risk': 'Unknown'
    }
    
    for line in lines:
        # Heartbeat
        if 'Bot heartbeat:' in line:
            stats['heartbeat'] = line.split(' - ')[0]
            if 'ws_connected=True' in line:
                stats['ws_connected'] = True
            else:
                stats['ws_connected'] = False
        
        # Balance
        if 'Bybit balance refreshed' in line:
            match = re.search(r'\$(\d+\.\d+)', line)
            if match:
                stats['balance'] = f"${match.group(1)}"
        
        # Risk
        if 'Risk per trade:' in line:
            match = re.search(r'Risk per trade: (.*?) \|', line)
            if match:
                stats['risk'] = match.group(1)

        # Active Symbols (from 3m bar close)
        if '3m BAR CLOSED' in line:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                stats['active_symbols'].add(match.group(1))
        
        # Phantoms
        if 'Phantom Recorded' in line:
            # Format: 2025... - INFO - 👻 Phantom Recorded [SYMBOL]: side | Combo: ...
            parts = line.split('Phantom Recorded')
            if len(parts) > 1:
                timestamp = line.split(' - ')[0].split(' ')[1]
                content = parts[1].strip()
                stats['phantoms'].append(f"[{timestamp}] {content}")
        
        # Executions
        if '🚀 EXECUTING' in line:
            parts = line.split('EXECUTING')
            if len(parts) > 1:
                timestamp = line.split(' - ')[0].split(' ')[1]
                content = parts[1].strip()
                stats['executions'].append(f"[{timestamp}] {content}")
                
        # Errors
        if 'ERROR -' in line:
            timestamp = line.split(' - ')[0].split(' ')[1]
            content = line.split('ERROR -')[1].strip()
            stats['errors'].append(f"[{timestamp}] {content}")

    return stats

def main():
    print("Starting Live Monitor...")
    time.sleep(1)
    
    while True:
        try:
            stats = parse_logs()
            clear_screen()
            
            # Header
            print("╔" + "═" * 78 + "╗")
            print("║" + " " * 28 + "🤖 LIVE BOT MONITOR" + " " * 29 + "║")
            print("╠" + "═" * 78 + "╣")
            
            # Status
            hb_status = "🟢 Online" if stats['ws_connected'] else "🔴 Disconnected"
            last_hb = stats['heartbeat'] if stats['heartbeat'] else "Waiting..."
            print(f"║ Status:  {hb_status:<20} Last Heartbeat: {last_hb:<25} ║")
            print(f"║ Balance: {stats['balance']:<20} Risk: {stats['risk']:<35} ║")
            print(f"║ Symbols: {len(stats['active_symbols']):<20} (Active in last 2000 lines)           ║")
            print("╠" + "═" * 78 + "╣")
            
            # Executions
            print("║ 🚀 Recent Executions:" + " " * 54 + "║")
            if not stats['executions']:
                print("║    (No recent executions)" + " " * 51 + "║")
            for ex in list(stats['executions'])[::-1]:
                print(f"║    {ex[:72]:<72} ║")
            
            print("╠" + "═" * 78 + "╣")
            
            # Phantoms
            print("║ 👻 Recent Phantoms:" + " " * 56 + "║")
            if not stats['phantoms']:
                print("║    (No recent phantoms)" + " " * 53 + "║")
            for ph in list(stats['phantoms'])[::-1]:
                print(f"║    {ph[:72]:<72} ║")
                
            print("╠" + "═" * 78 + "╣")
            
            # Errors
            if stats['errors']:
                print("║ ⚠️  Recent Errors:" + " " * 58 + "║")
                for err in list(stats['errors'])[::-1]:
                    print(f"║    {err[:72]:<72} ║")
                print("╚" + "═" * 78 + "╝")
            else:
                print("║ ✅ System Healthy (No recent errors)" + " " * 40 + "║")
                print("╚" + "═" * 78 + "╝")
            
            print("\nPress Ctrl+C to exit monitor (Bot continues running)")
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
