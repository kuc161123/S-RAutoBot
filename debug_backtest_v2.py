import asyncio
import aiohttp
import pandas as pd
import pandas_ta as ta
import logging
from backtest_wf_v2 import BacktestEngine, fetch_klines

# Configure Logging to Console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def debug_symbol(symbol):
    print(f"🔍 DEBUGGING {symbol}...")
    
    async with aiohttp.ClientSession() as session:
        df = await fetch_klines(session, symbol)
        
    if df.empty:
        print("❌ No data fetched")
        return

    print(f"📊 Data: {len(df)} candles")
    
    engine = BacktestEngine()
    df = engine.calculate_indicators(df)
    
    # Split
    n = len(df)
    train_end = int(n * 0.60)
    test_end = int(n * 0.80)
    
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:test_end]
    
    print(f"🔹 Train: {len(train_df)} | Test: {len(test_df)}", flush=True)
    
    # 1. Discovery
    print("\n--- STEP 1: DISCOVERY (TRAIN) ---", flush=True)
    combos = {}
    sigs = (train_df['bbw_pct'] > 0.45) & (train_df['vol_ratio'] > 0.8)
    
    for idx in train_df[sigs].index:
        if idx >= train_df.index[-100]: continue
        row = train_df.loc[idx]
        combo = engine.get_combo(row)
        
        # Check Short (since most elites are short)
        if row['close'] < row['open']:
            entry = train_df.loc[idx, 'close']
            atr = row['atr']
            tp = entry - 2*atr
            sl = entry + 2*atr
            
            future = train_df.loc[idx:].iloc[1:100]
            win = False
            for _, f in future.iterrows():
                if f['high'] >= sl: break
                if f['low'] <= tp: win = True; break
            
            k = ('short', combo)
            if k not in combos: combos[k] = {'wins': 0, 'total': 0}
            combos[k]['total'] += 1
            if win: combos[k]['wins'] += 1

    candidates = []
    for (side, combo), stats in combos.items():
        wr = (stats['wins']/stats['total']) * 100
        if stats['total'] >= 30 and wr > 60:
            print(f"✅ CANDIDATE: {side} {combo} | WR: {wr:.1f}% ({stats['wins']}/{stats['total']})", flush=True)
            candidates.append({'side': side, 'combo': combo})
        elif stats['total'] >= 30:
            print(f"❌ FAILED TRAIN: {side} {combo} | WR: {wr:.1f}% (Need >60%)", flush=True)

    if not candidates:
        print("❌ NO CANDIDATES FOUND IN TRAIN")
        return

    # 2. Validation
    print("\n--- STEP 2: VALIDATION (TEST) ---")
    for cand in candidates:
        side = cand['side']
        combo = cand['combo']
        print(f"\nTesting {side} {combo}...")
        
        trades = []
        t_sigs = (test_df['bbw_pct'] > 0.45) & (test_df['vol_ratio'] > 0.8) & (test_df['close'] < test_df['open'])
        
        for idx in test_df[t_sigs].index:
            if idx >= test_df.index[-100]: continue
            row = test_df.loc[idx]
            if engine.get_combo(row) != combo: continue
            
            next_idx = test_df.index.get_loc(idx) + 1
            if next_idx >= len(test_df): continue
            
            entry_price = test_df.iloc[next_idx]['open']
            atr = row['atr']
            tp = entry_price - 2*atr
            sl = entry_price + 2*atr
            
            future = test_df.iloc[next_idx:].iloc[1:100]
            pnl, outcome = engine.simulate_trade(entry_price, side, sl, tp, future, stress_test=False)
            trades.append({'pnl': pnl, 'outcome': outcome})
            
        if not trades:
            print("  ❌ No trades in Test")
            continue
            
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        
        print(f"  📊 Test Result: PnL={total_pnl:.2f}% | WR={win_rate:.1f}% | Trades={len(trades)}")
        
        if total_pnl <= 0: print("  ❌ FAILED: Negative PnL")
        elif win_rate <= 50: print("  ❌ FAILED: WR <= 50%")
        elif len(trades) < 15: print("  ❌ FAILED: Not enough trades (<15)")
        else:
            print("  ✅ PASSED PnL Check")
            mc_pass, mc_dd = engine.monte_carlo_check(trades)
            print(f"  🎲 Monte Carlo MaxDD: {mc_dd*100:.1f}% (Limit 30%)")
            if mc_pass:
                print("  ✅ PASSED Monte Carlo")
            else:
                print("  ❌ FAILED Monte Carlo")

if __name__ == "__main__":
    asyncio.run(debug_symbol("ETHUSDT"))
