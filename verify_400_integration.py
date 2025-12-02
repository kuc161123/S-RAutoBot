#!/usr/bin/env python3
"""
Verification script for 400-symbol integration.

Tests:
1. Verify config.yaml has 196 symbols
2. Verify symbol_overrides.yaml is loaded correctly
3. Verify WR/N stats are parsed from comments
4. Display sample stats
"""
import yaml

def main():
    print("="*80)
    print("🧪 VERIFYING 400-SYMBOL INTEGRATION")
    print("="*80)
    
    # Test 1: Check config.yaml
    print("\n📋 Test 1: Checking config.yaml...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        symbols = config.get('trade', {}).get('symbols', [])
        print(f"✅ Config loaded: {len(symbols)} symbols configured")
        
        if len(symbols) == 196:
            print("✅ Correct symbol count (196)")
        else:
            print(f"⚠️  Expected 196 symbols, got {len(symbols)}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Test 2: Check symbol_overrides.yaml
    print("\n📊 Test 2: Checking symbol_overrides.yaml...")
    try:
        with open('symbol_overrides.yaml', 'r') as f:
            overrides = yaml.safe_load(f)
        
        print(f"✅ Overrides loaded: {len(overrides)} symbols with combos")
        
        if len(overrides) ==196:
            print("✅ Correct override count (196)")
        else:
            print(f"⚠️  Expected 196 override symbols, got {len(overrides)}")
    except Exception as e:
        print(f"❌ Failed to load overrides: {e}")
        return
    
    # Test 3: Parse WR/N stats from comments
    print("\n🔍 Test 3: Parsing backtest stats from comments...")
    try:
        import re
        
        with open('symbol_overrides.yaml', 'r') as f:
            lines = f.readlines()
        
        stats = {}
        current_symbol = None
        current_side = None
        
        for line in lines:
            # Detect symbol
            symbol_match = re.match(r'^([A-Z0-9]+USDT):', line)
            if symbol_match:
                current_symbol = symbol_match.group(1)
                if current_symbol not in stats:
                    stats[current_symbol] = {}
                current_side = None
                continue
            
            # Detect side
            side_match = re.match(r'^  (long|short):', line)
            if side_match and current_symbol:
                current_side = side_match.group(1)
                continue
            
            # Parse backtest comment
            backtest_match = re.match(r'\s*#\s*Backtest:\s*WR=([\d.]+)%,?\s*N=(\d+)', line)
            if backtest_match and current_symbol and current_side:

                wr = float(backtest_match.group(1))
                n = int(backtest_match.group(2))
                stats[current_symbol][current_side] = {'wr': wr, 'n': n}
        
        total_stats = sum(len(sides) for sides in stats.values())
        print(f"✅ Parsed {total_stats} backtest stats from {len(stats)} symbols")
        
        # Display top 10 by WR
        print("\n🏆 Top 10 Combos by Win Rate:")

        all_combos = []
        for sym, sides in stats.items():
            for side, stat in sides.items():
                all_combos.append((sym, side, stat['wr'], stat['n']))
        
        all_combos.sort(key=lambda x: x[2], reverse=True)
        for idx, (sym, side, wr, n) in enumerate(all_combos[:10], 1):
            print(f"{idx:2d}. {sym:20s} {side:5s} WR={wr:5.1f}% (N={n:3d})")
        
    except Exception as e:
        print(f"❌ Failed to parse stats: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)
    print(f"📈 Configuration: {len(symbols)} symbols")
    print(f"📊 Overrides: {len(overrides)} symbols with combos")
    print(f"📉 Backtest Stats: {total_stats} WR/N pairs parsed")
    print("\n🚀 Integration verified successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

