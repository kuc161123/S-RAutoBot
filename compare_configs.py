#!/usr/bin/env python3
"""
COMPARE CONFIGS (SMART VERSION)
===============================
Compares current config.yaml with market_wide_results.csv.
CRITICAL ADDITION: verification_mode.
It runs a simulation for the CURRENT config on the 150-day data to get a TRUE baseline.
"""

import pandas as pd
import yaml
import sys
import os
import concurrent.futures

# Import simulation logic from the research script
import research_market_wide as research

def load_yaml_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_results(path='market_wide_results.csv'):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def verify_baseline(symbol, current_cfg):
    """
    Runs a simulation for the CURRENT config on the same data
    to get the actual 'Current R' for the last 150 days.
    """
    try:
        # Fetch data (same parameters as research)
        df_1h = research.fetch_klines(symbol, research.SIGNAL_TF, research.DAYS)
        df_5m = research.fetch_klines(symbol, research.EXECUTION_TF, research.DAYS)
        
        if df_1h.empty or df_5m.empty:
            return None

        df_1h = research.prepare_1h_data(df_1h)
        all_signals = research.detect_signals(df_1h)
        
        # Filter for current config's signal type
        target_signals = [s for s in all_signals if s['type'] == current_cfg['divergence_type']]
        
        if not target_signals:
            return 0.0 # No signals triggers = 0 R

        trades = []
        rr = float(current_cfg['rr'])
        atr = float(current_cfg['atr_mult'])

        for sig in target_signals:
            res = research.execute_trade(sig, df_1h, df_5m, rr, atr)
            if res is not None:
                trades.append(res)
        
        return sum(trades)
    except Exception as e:
        print(f"Error checking baseline for {symbol}: {e}")
        return None

def main():
    print("🔍 SMARTER Comparison: Current Config vs. New Research...")

    # Load data
    try:
        config = load_yaml_config()
        new_results = load_results()
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    if new_results.empty:
        print("⚠️ No research results found yet.")
        return

    current_symbols = config.get('symbols', {})
    print(f"📄 Current Portfolio: {len(current_symbols)} symbols")
    print(f"📊 New Research: {len(new_results)} configurations found")

    # Filter to only symbols where we have research results
    # (No point comparing if we haven't scanned that symbol yet)
    researched_symbols = set(new_results['symbol'].unique())
    relevant_current = {k: v for k, v in current_symbols.items() if k in researched_symbols}
    
    print(f"✅ Intersected: {len(relevant_current)} symbols have fresh research data.")

    comparison = []
    
    # We will only verify baseline for symbols where a strictly BETTER config exists
    # to save time, or we can do it for all 'relevant' symbols to get a full picture.
    # Let's do it for the top candidates first.
    
    print("\n🚀 Verifying Baselines (Parallel Execution)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {}
        
        for sym, current_cfg in relevant_current.items():
            symbol_results = new_results[new_results['symbol'] == sym]
            if symbol_results.empty: continue
            
            best_new = symbol_results.sort_values('total_r', ascending=False).iloc[0]
            
            # Check if parameters are already identical
            match = (
                current_cfg['divergence_type'] == best_new['div_type'] and
                float(current_cfg['rr']) == float(best_new['rr']) and 
                float(current_cfg['atr_mult']) == float(best_new['atr'])
            )
            
            if match:
                comparison.append({
                    'symbol': sym,
                    'status': 'MATCH',
                    'current_r': best_new['total_r'], # Same logic = Same R
                    'new_r': best_new['total_r'],
                    'diff': 0,
                    'details': f"ALREADY OPTIMIZED ({best_new['div_type']} {best_new['rr']}R)"
                })
            else:
                # Need to verify baseline
                future = executor.submit(verify_baseline, sym, current_cfg)
                future_to_sym[future] = (sym, best_new)

        completed = 0
        for future in concurrent.futures.as_completed(future_to_sym):
            sym, best_new = future_to_sym[future]
            try:
                baseline_r = future.result()
                if baseline_r is None:
                    status = "ERROR"
                    baseline_r = 0
                else:
                    status = "IMPROVED"
                
                diff = best_new['total_r'] - baseline_r
                
                comparison.append({
                    'symbol': sym,
                    'status': status,
                    'current_r': round(baseline_r, 2),
                    'new_r': best_new['total_r'],
                    'diff': round(diff, 2),
                    'details': f"Old: {baseline_r:.1f}R -> New: {best_new['total_r']:.1f}R ({best_new['div_type']} {best_new['rr']}R)"
                })
                
            except Exception as e:
                print(f"Failed {sym}: {e}")
            
            completed += 1
            sys.stdout.write(f"\rValidated {completed}/{len(future_to_sym)}")
            sys.stdout.flush()

    print("\n")
    
    # Analyze Output
    df = pd.DataFrame(comparison)
    if df.empty:
        print("No comparisons made yet.")
        return

    # Sort by positive difference
    df = df.sort_values('diff', ascending=False)
    
    print("\n🏆 Verified Improvements (Apples-to-Apples):")
    print(df[df['diff'] > 5].head(20).to_string(index=False))
    
    print(f"\n📉 Degraded/Same: {len(df[df['diff'] <= 0])}")
    
    # Save detailed report
    df.to_csv('verified_comparison.csv', index=False)
    print("\n💾 Saved to 'verified_comparison.csv'")

if __name__ == "__main__":
    main()
