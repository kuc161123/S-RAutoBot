#!/usr/bin/env python3
"""Monitor realistic backtest progress"""
import time
import os

last_size = 0
output_file = 'symbol_overrides_REALISTIC.yaml'

print("Monitoring realistic backtest progress...")
print("Output file: symbol_overrides_REALISTIC.yaml")
print("="*60)

while True:
    try:
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Count symbols processed
            symbols_found = sum(1 for line in lines if line.strip() and not line.strip().startswith('#') and ':' in line and 'USDT:' in line)
            
            if size != last_size:
                print(f"{time.strftime('%H:%M:%S')} | Symbols validated: {symbols_found} | File size: {size:,} bytes")
                last_size = size
        else:
            print(f"{time.strftime('%H:%M:%S')} | Waiting for output file...")
        
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
