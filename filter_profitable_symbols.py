#!/usr/bin/env python3
"""
Process all symbols and filter for Win Rate > 50%
Only symbols meeting this threshold will be added to the tradable list
"""
import json
from backtest_single_symbol import SingleSymbolOptimizer

def main():
    optimizer = SingleSymbolOptimizer(data_file='backtest_data_all50.json')
    
    # Load all available symbols
    with open('backtest_data_all50.json', 'r') as f:
        data = json.load(f)
    
    symbols = list(data.keys())
    print(f"{'='*60}")
    print(f"Testing {len(symbols)} symbols for WR > 50% threshold")
    print(f"{'='*60}\n")
    
    profitable_symbols = []
    all_results = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        try:
            result = optimizer.optimize_symbol(symbol)
            
            if result:
                all_results.append({
                    'symbol': symbol,
                    'win_rate': result['win_rate'],
                    'trades': result['trades'],
                    'score': result['score'],
                    'params': result['params']
                })
                
                # Check WR threshold
                if result['win_rate'] >= 50.0:
                    profitable_symbols.append(symbol)
                    print(f"✅ {symbol}: WR {result['win_rate']:.1f}% - PASSED (Trades: {result['trades']})")
                else:
                    print(f"❌ {symbol}: WR {result['win_rate']:.1f}% - REJECTED (< 50%)")
            else:
                print(f"⚠️  {symbol}: No valid configuration found")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    # Save all results
    with open('all_symbol_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save profitable symbols list
    with open('profitable_symbols.json', 'w') as f:
        json.dump({
            'symbols': profitable_symbols,
            'count': len(profitable_symbols),
            'threshold': '50%'
        }, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tested: {len(all_results)}")
    print(f"Passed (WR ≥ 50%): {len(profitable_symbols)}")
    print(f"Rejected (WR < 50%): {len(all_results) - len(profitable_symbols)}")
    
    if profitable_symbols:
        print(f"\n🏆 PROFITABLE SYMBOLS (WR ≥ 50%):")
        for sym in profitable_symbols:
            res = next(r for r in all_results if r['symbol'] == sym)
            print(f"  • {sym}: {res['win_rate']:.1f}% WR ({res['trades']} trades)")
    else:
        print(f"\n⚠️  No symbols met the 50% WR threshold")
        print(f"\nTop 5 by Win Rate:")
        sorted_results = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)
        for i, res in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {res['symbol']}: {res['win_rate']:.1f}% WR ({res['trades']} trades)")
    
    print(f"\n✅ Results saved to:")
    print(f"  - all_symbol_results.json")
    print(f"  - profitable_symbols.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
