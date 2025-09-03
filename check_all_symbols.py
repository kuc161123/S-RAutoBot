#!/usr/bin/env python3
"""
Check all available Bybit futures symbols and analyze feasibility
"""
import requests
import json
import yaml

def get_all_bybit_futures():
    """Get all available USDT perpetuals on Bybit"""
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {
        "category": "linear",
        "status": "Trading"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error: {data['retMsg']}")
            return None
        
        symbols = []
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol.endswith('USDT') and not symbol.endswith('-'):
                # Filter out obvious stablecoins and weird pairs
                if not any(x in symbol for x in ['USDC', 'BUSD', 'TUSD', 'DAI', 'EUR', 'GBP', 'BRZ']):
                    symbols.append({
                        'symbol': symbol,
                        'max_leverage': float(item['leverageFilter']['maxLeverage']),
                        'volume_24h': 0  # Will fetch separately
                    })
        
        return symbols
        
    except Exception as e:
        print(f"Failed to fetch from Bybit: {e}")
        return None

def get_volume_data(symbols):
    """Get 24h volume for symbols"""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear"}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        volume_map = {}
        for item in data['result']['list']:
            if 'volume24h' in item:
                volume_map[item['symbol']] = float(item.get('turnover24h', 0))
        
        # Update symbols with volume
        for sym in symbols:
            sym['volume_24h'] = volume_map.get(sym['symbol'], 0)
        
        return symbols
    except:
        return symbols

def analyze_symbol_requirements(symbols):
    """Analyze requirements for monitoring all symbols"""
    
    # Filter by minimum volume ($100k daily volume minimum)
    active_symbols = [s for s in symbols if s['volume_24h'] > 100000]
    
    # Calculate WebSocket connections needed (190 per connection)
    max_per_ws = 190
    total_connections = (len(active_symbols) - 1) // max_per_ws + 1
    
    # Estimate memory usage (rough estimates)
    memory_per_symbol = 0.5  # MB (klines + state)
    total_memory = len(active_symbols) * memory_per_symbol
    
    # Estimate processing load
    candles_per_minute = len(active_symbols) * (60 / 15)  # 15-min timeframe
    
    return {
        'total_available': len(symbols),
        'active_tradeable': len(active_symbols),
        'ws_connections_needed': total_connections,
        'estimated_memory_mb': total_memory,
        'candles_per_minute': candles_per_minute,
        'symbols': active_symbols
    }

def create_all_symbols_config(analysis):
    """Create config with all symbols"""
    
    # Read current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Sort by volume (highest first)
    symbols = sorted(analysis['symbols'], key=lambda x: x['volume_24h'], reverse=True)
    
    # Take all symbols with decent volume
    symbol_list = [s['symbol'] for s in symbols]
    
    # Update config
    config['trade']['symbols'] = symbol_list
    
    # Save as new file
    with open('config_all_symbols.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return symbol_list

def main():
    print("=" * 80)
    print("🔍 Analyzing ALL Bybit Futures Symbols")
    print("=" * 80)
    
    # Get all symbols
    print("\n📊 Fetching all available symbols...")
    symbols = get_all_bybit_futures()
    
    if not symbols:
        print("Failed to fetch symbols")
        return
    
    print(f"Found {len(symbols)} total USDT perpetuals")
    
    # Get volume data
    print("\n📈 Fetching volume data...")
    symbols = get_volume_data(symbols)
    
    # Analyze requirements
    print("\n🔬 Analyzing requirements...")
    analysis = analyze_symbol_requirements(symbols)
    
    print(f"""
📊 ANALYSIS RESULTS:
{'=' * 50}
Total Available Symbols: {analysis['total_available']}
Active (>$100k volume): {analysis['active_tradeable']}
Very Active (>$1M volume): {len([s for s in analysis['symbols'] if s['volume_24h'] > 1000000])}
High Volume (>$10M volume): {len([s for s in analysis['symbols'] if s['volume_24h'] > 10000000])}

🔧 TECHNICAL REQUIREMENTS:
{'=' * 50}
WebSocket Connections: {analysis['ws_connections_needed']}
Estimated Memory: {analysis['estimated_memory_mb']:.0f} MB
Candles per minute: {analysis['candles_per_minute']:.0f}
Signals to process: ~{analysis['active_tradeable'] * 2} per day

⚠️ IMPACT ASSESSMENT:
{'=' * 50}""")
    
    if analysis['active_tradeable'] > 500:
        print("""
❌ HIGH RISK - NOT RECOMMENDED:
• Too many symbols ({} > 500)
• WebSocket connections: {} (may hit rate limits)
• Memory usage: {:.0f}MB (may cause OOM on Railway)
• Processing load: Very High
• Signal quality: Will decrease (too much noise)

RECOMMENDATION: Stick with top 250-400 by volume
""".format(analysis['active_tradeable'], analysis['ws_connections_needed'], analysis['estimated_memory_mb']))
    elif analysis['active_tradeable'] > 300:
        print("""
⚠️ MODERATE RISK - PROCEED WITH CAUTION:
• Manageable symbols ({})
• WebSocket connections: {} (within limits)
• Memory usage: {:.0f}MB (acceptable)
• Processing load: High but manageable
• Signal quality: May have more false signals

RECOMMENDATION: Could work, but monitor closely
""".format(analysis['active_tradeable'], analysis['ws_connections_needed'], analysis['estimated_memory_mb']))
    else:
        print("""
✅ LOW RISK - FEASIBLE:
• Good symbol count ({})
• WebSocket connections: {} (safe)
• Memory usage: {:.0f}MB (fine)
• Processing load: Manageable
• Signal quality: Should maintain quality

RECOMMENDATION: Safe to proceed
""".format(analysis['active_tradeable'], analysis['ws_connections_needed'], analysis['estimated_memory_mb']))
    
    # Show top symbols by volume
    print(f"\n📊 Top 10 Symbols by Volume:")
    print("=" * 50)
    top_symbols = sorted(analysis['symbols'], key=lambda x: x['volume_24h'], reverse=True)[:10]
    for i, sym in enumerate(top_symbols, 1):
        print(f"{i:2}. {sym['symbol']:15} ${sym['volume_24h']/1e6:,.1f}M")
    
    # Ask for confirmation
    print(f"""
🤔 DECISION TIME:
{'=' * 50}
Current: 250 symbols (safe, tested)
Proposed: {analysis['active_tradeable']} symbols (all active)

Benefits of ALL symbols:
✅ More trading opportunities
✅ Catch new listings early
✅ No missing potential winners

Risks of ALL symbols:
❌ Higher resource usage
❌ More false signals
❌ Potential stability issues
❌ Harder to track/debug

Would you like to:
1. Create config_all_symbols.yaml ({analysis['active_tradeable']} symbols)
2. Create config_top_400.yaml (compromise)
3. Stay with current 250 (recommended)
""")
    
    # Create the config file
    if analysis['active_tradeable'] <= 500:
        print(f"\n📝 Creating config_all_symbols.yaml with {analysis['active_tradeable']} symbols...")
        symbol_list = create_all_symbols_config(analysis)
        print(f"✅ Created config_all_symbols.yaml")
        
        # Create top 400 version too
        analysis_400 = analysis.copy()
        analysis_400['symbols'] = sorted(analysis['symbols'], key=lambda x: x['volume_24h'], reverse=True)[:400]
        create_all_symbols_config(analysis_400)
        with open('config_all_symbols.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['trade']['symbols'] = [s['symbol'] for s in analysis_400['symbols']]
        with open('config_top_400.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Created config_top_400.yaml with 400 symbols")
        
        # Backup current config
        print(f"\n📦 Creating backup...")
        import shutil
        shutil.copy('config.yaml', 'config_backup_250.yaml')
        print(f"✅ Backup saved as config_backup_250.yaml")
        
        print(f"""
📋 FILES CREATED:
• config_backup_250.yaml - Your current safe config (250 symbols)
• config_top_400.yaml - Moderate expansion (400 symbols)  
• config_all_symbols.yaml - All active symbols ({analysis['active_tradeable']})

TO SWITCH:
• Test 400: cp config_top_400.yaml config.yaml
• Use all: cp config_all_symbols.yaml config.yaml
• Revert: cp config_backup_250.yaml config.yaml

⚠️ IMPORTANT: Test gradually!
1. Try 400 first for a day
2. If stable, try all symbols
3. Monitor memory and performance
""")

if __name__ == "__main__":
    main()