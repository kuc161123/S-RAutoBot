#!/usr/bin/env python3
"""
Automated Symbol Processing Workflow:
1. Fetch historical data for symbol
2. Run backtest (768 combinations)
3. If WR >= 50%: Add to Pro Rules in bot.py, commit, push
4. Move to next symbol

This will run continuously until all symbols are processed.
"""
import json
import os
import subprocess
import sys
import time
import requests
from datetime import datetime

def fetch_candles_for_symbol(symbol, limit=1000):
    """Fetch historical 15m candles for a symbol"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": "15",
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"  ✗ API Error: {data['retMsg']}")
            return None
        
        candles = []
        for item in data['result']['list']:
            candles.append({
                'timestamp': int(item[0]),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            })
        
        # Reverse to get chronological order
        candles.reverse()
        return candles
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def run_backtest(symbol, data):
    """Run backtest for a symbol using the data"""
    # Save data to temp file
    temp_file = f'temp_data_{symbol}.json'
    with open(temp_file, 'w') as f:
        json.dump({symbol: data}, f)
    
    # Import and run backtest
    try:
        sys.path.insert(0, os.getcwd())
        from backtest_single_symbol import SingleSymbolOptimizer
        
        optimizer = SingleSymbolOptimizer(data_file=temp_file)
        result = optimizer.optimize_symbol(symbol)
        
        # Clean up temp file
        os.remove(temp_file)
        
        return result
    except Exception as e:
        print(f"  ✗ Backtest error: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None

def add_symbol_to_pro_rules(symbol, params):
    """Add symbol's optimal parameters to bot.py Pro Rules"""
    # Read current bot.py
    with open('autobot/core/bot.py', 'r') as f:
        lines = f.readlines()
    
    # Find where to insert (after line 100, before class definitions)
    insert_line = None
    for i, line in enumerate(lines):
        if i > 100 and 'PER_SYMBOL_PRO_RULES' in line:
            # Rules dict already exists, find the closing brace
            for j in range(i, len(lines)):
                if lines[j].strip() == '}':
                    insert_line = j
                    break
            break
        elif i > 100 and ('class ' in line or 'def ' in line):
            # Need to create the dict before this line
            insert_line = i
            # Check if dict exists a few lines before
            dict_exists = False
            for k in range(max(0, i-20), i):
                if 'PER_SYMBOL_PRO_RULES' in lines[k]:
                    dict_exists = True
                    break
            
            if not dict_exists:
                # Create the dict
                new_lines = [
                    "\n",
                    "# Per-Symbol Optimized Pro Rules (WR >= 50%)\n",
                    "# Auto-generated from backtesting - DO NOT EDIT MANUALLY\n",
                    "PER_SYMBOL_PRO_RULES = {\n",
                    f"    '{symbol}': {{\n",
                    f"        'rsi_min_long': {params['rsi_min_long']},\n",
                    f"        'rsi_max_long': {params['rsi_max_long']},\n",
                    f"        'rsi_min_short': {params['rsi_min_short']},\n",
                    f"        'rsi_max_short': {params['rsi_max_short']},\n",
                    f"        'vwap_dist_max': {params['vwap_dist_max']},\n",
                    f"        'vol_ratio_min': {params['vol_ratio_min']},\n",
                    f"        'macd_hist_min': {params['macd_hist_min']},\n",
                    f"        'bb_width_min': {params['bb_width_min']}\n",
                    "    },\n",
                    "}\n",
                    "\n"
                ]
                lines = lines[:insert_line] + new_lines + lines[insert_line:]
            break
    
    if insert_line is None:
        # Fallback: insert at line 120
        insert_line = 120
        new_lines = [
            "\n",
            "# Per-Symbol Optimized Pro Rules (WR >= 50%)\n",
            "PER_SYMBOL_PRO_RULES = {\n",
            f"    '{symbol}': {{\n",
            f"        'rsi_min_long': {params['rsi_min_long']},\n",
            f"        'rsi_max_long': {params['rsi_max_long']},\n",
            f"        'rsi_min_short': {params['rsi_min_short']},\n",
            f"        'rsi_max_short': {params['rsi_max_short']},\n",
            f"        'vwap_dist_max': {params['vwap_dist_max']},\n",
            f"        'vol_ratio_min': {params['vol_ratio_min']},\n",
            f"        'macd_hist_min': {params['macd_hist_min']},\n",
            f"        'bb_width_min': {params['bb_width_min']}\n",
            "    },\n",
            "}\n",
            "\n"
        ]
        lines = lines[:insert_line] + new_lines + lines[insert_line:]
    else:
        # Dict exists, add entry before closing brace
        new_entry = [
            f"    '{symbol}': {{\n",
            f"        'rsi_min_long': {params['rsi_min_long']},\n",
            f"        'rsi_max_long': {params['rsi_max_long']},\n",
            f"        'rsi_min_short': {params['rsi_min_short']},\n",
            f"        'rsi_max_short': {params['rsi_max_short']},\n",
            f"        'vwap_dist_max': {params['vwap_dist_max']},\n",
            f"        'vol_ratio_min': {params['vol_ratio_min']},\n",
            f"        'macd_hist_min': {params['macd_hist_min']},\n",
            f"        'bb_width_min': {params['bb_width_min']}\n",
            "    },\n"
        ]
        lines = lines[:insert_line] + new_entry + lines[insert_line:]
    
    # Write back
    with open('autobot/core/bot.py', 'w') as f:
        f.writelines(lines)
    
    return True

def commit_and_push(symbol, wr, trades):
    """Commit and push the changes"""
    try:
        # Git add
        subprocess.run(['git', 'add', 'autobot/core/bot.py'], check=True, cwd=os.getcwd())
        
        # Commit
        msg = f"Add {symbol} to Pro Rules (WR: {wr:.1f}%, Trades: {trades})"
        subprocess.run(['git', 'commit', '-m', msg], check=True, cwd=os.getcwd())
        
        # Push
        subprocess.run(['git', 'push', 'origin', 'main'], check=True, cwd=os.getcwd())
        
        return True
    except Exception as e:
        print(f"  ✗ Git error: {e}")
        return False

def main():
    # Load all symbols
    with open('all_bybit_symbols.json', 'r') as f:
        data = json.load(f)
    symbols = data['symbols']
    
    print(f"{'='*60}")
    print(f"🚀 AUTOMATED SYMBOL PROCESSING")
    print(f"{'='*60}")
    print(f"Total Symbols: {len(symbols)}")
    print(f"Threshold: WR >= 50%")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Load progress if exists
    progress_file = 'processing_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        processed = set(progress.get('processed', []))
        profitable = progress.get('profitable', [])
    else:
        processed = set()
        profitable = []
    
    # Process each symbol
    for i, symbol in enumerate(symbols, 1):
        if symbol in processed:
            print(f"[{i}/{len(symbols)}] {symbol} - Already processed, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(symbols)}] Processing: {symbol}")
        print(f"{'='*60}")
        
        # 1. Fetch data
        print(f"  📥 Fetching historical data...")
        candles = fetch_candles_for_symbol(symbol)
        
        if not candles:
            print(f"  ⚠️  Skipping {symbol} - No data available")
            processed.add(symbol)
            continue
        
        print(f"  ✓ Fetched {len(candles)} candles")
        
        # 2. Run backtest
        print(f"  🧪 Running backtest (768 combinations)...")
        result = run_backtest(symbol, candles)
        
        if not result:
            print(f"  ⚠️  Skipping {symbol} - Backtest failed")
            processed.add(symbol)
            continue
        
        wr = result['win_rate']
        trades = result['trades']
        
        # 3. Check if profitable
        if wr >= 50.0:
            print(f"  ✅ PROFITABLE: WR {wr:.1f}% ({trades} trades)")
            print(f"  🔧 Adding to Pro Rules...")
            
            if add_symbol_to_pro_rules(symbol, result['params']):
                print(f"  ✓ Added to bot.py")
                
                print(f"  📤 Committing and pushing...")
                if commit_and_push(symbol, wr, trades):
                    print(f"  ✓ Committed and pushed")
                    profitable.append({
                        'symbol': symbol,
                        'win_rate': wr,
                        'trades': trades
                    })
                else:
                    print(f"  ✗ Failed to commit")
        else:
            print(f"  ❌ REJECTED: WR {wr:.1f}% < 50%")
        
        # Mark as processed
        processed.add(symbol)
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump({
                'processed': list(processed),
                'profitable': profitable,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Processed: {len(processed)}")
    print(f"Profitable (WR >= 50%): {len(profitable)}")
    
    if profitable:
        print(f"\n🏆 PROFITABLE SYMBOLS:")
        for item in profitable:
            print(f"  • {item['symbol']}: {item['win_rate']:.1f}% WR ({item['trades']} trades)")
    else:
        print(f"\n⚠️  No symbols met the 50% WR threshold")
    
    print(f"\n✅ Processing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
