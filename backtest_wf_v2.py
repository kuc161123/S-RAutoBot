import asyncio
import aiohttp
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import yaml
from collections import defaultdict
import random

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_wf_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
TIMEFRAME = '3'
LIMIT = 1000
DAYS = 180  # Increased history for better Holdout split
TRAIN_SPLIT = 0.60  # 60% Optimization
TEST_SPLIT = 0.20   # 20% Walk-Forward Validation
HOLDOUT_SPLIT = 0.20 # 20% The Vault (Final Exam)

# Realistic Costs
FEE_RATE = 0.00055  # 0.055% Taker Fee
SLIPPAGE_BASE = 0.0003 # 0.03% Base Slippage
SPREAD_BASE = 0.0002   # 0.02% Base Spread

# Stress Test Config
MONTE_CARLO_RUNS = 1000
MAX_DD_THRESHOLD = 0.30  # 30% Max Drawdown allowed
BAD_LUCK_PROB = 0.05     # 5% chance of missed fill
SPREAD_SPIKE_PROB = 0.10 # 10% chance of double spread

class BacktestEngine:
    def __init__(self):
        pass

    def calculate_indicators(self, df):
        try:
            # RSI 14
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD (12, 26, 9)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # VWAP (Rolling 20)
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['vp'] = df['tp'] * df['volume']
            df['vwap'] = df['vp'].rolling(20).sum() / df['volume'].rolling(20).sum()
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['vwap_dist_atr'] = (df['close'] - df['vwap']).abs() / df['atr']
            
            # MTF (EMA 20 vs 50)
            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            
            # Fib Zone (50 bar lookback)
            df['roll_max'] = df['high'].rolling(50).max()
            df['roll_min'] = df['low'].rolling(50).min()
            df['fib_ret'] = (df['roll_max'] - df['close']) / (df['roll_max'] - df['roll_min']) * 100
            
            # BB Width Pct
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_w'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['close']
            df['bbw_pct'] = df['bb_w'].rolling(100).rank(pct=True)
            
            # Volume Ratio
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
            
            return df.dropna()
        except Exception:
            return pd.DataFrame()

    def get_combo(self, row):
        # RSI Bin
        rsi = row['rsi']
        if rsi < 30: r_bin = '<30'
        elif rsi < 40: r_bin = '30-40'
        elif rsi < 60: r_bin = '40-60'
        elif rsi < 70: r_bin = '60-70'
        else: r_bin = '70+'
        
        # MACD Bin
        m_bin = 'bull' if row['macd_hist'] > 0 else 'bear'
        
        # VWAP Bin
        v = row['vwap_dist_atr']
        if v < 0.6: v_bin = '<0.6'
        elif v < 1.2: v_bin = '0.6-1.2'
        else: v_bin = '1.2+'
        
        # Fib Bin
        f = row['fib_ret']
        if f < 23.6: f_bin = '0-23'
        elif f < 38.2: f_bin = '23-38'
        elif f < 50: f_bin = '38-50'
        elif f < 61.8: f_bin = '50-61'
        elif f < 78.6: f_bin = '61-78'
        else: f_bin = '78-100'
        
        # MTF
        mtf = 'MTF' if row['ema20'] > row['ema50'] else 'noMTF'
        
        return f"RSI:{r_bin} MACD:{m_bin} VWAP:{v_bin} Fib:{f_bin} {mtf}"

    def simulate_trade(self, entry_price, side, sl, tp, future_candles, stress_test=False):
        """
        Simulates a trade with realistic execution.
        Returns: (pnl_percent, outcome_str)
        """
        # 1. Stress Test: Missed Fill?
        if stress_test and random.random() < BAD_LUCK_PROB:
            return 0.0, 'missed'

        # 2. Calculate Costs
        spread = SPREAD_BASE * 2 if (stress_test and random.random() < SPREAD_SPIKE_PROB) else SPREAD_BASE
        slippage = SLIPPAGE_BASE
        
        # Adjust Entry for Spread/Slippage
        # Long: Buy at Ask (Price + 0.5*Spread) + Slippage
        # Short: Sell at Bid (Price - 0.5*Spread) - Slippage
        if side == 'long':
            real_entry = entry_price * (1 + (spread/2) + slippage)
        else:
            real_entry = entry_price * (1 - (spread/2) - slippage)

        # 3. Check Outcome
        outcome = 'timeout'
        exit_price = real_entry # Default if timeout
        
        for _, row in future_candles.iterrows():
            if side == 'long':
                if row['low'] <= sl:
                    outcome = 'loss'
                    # Exit at SL - Slippage
                    exit_price = sl * (1 - slippage)
                    break
                if row['high'] >= tp:
                    outcome = 'win'
                    # Exit at TP - Slippage (limit order might slip if market moves fast through it, but usually 0 for limit)
                    # Let's assume TP is a limit order, so no negative slippage, maybe positive? 
                    # To be conservative/realistic for a taker bot or market exit:
                    exit_price = tp * (1 - slippage) 
                    break
            else:
                if row['high'] >= sl:
                    outcome = 'loss'
                    exit_price = sl * (1 + slippage)
                    break
                if row['low'] <= tp:
                    outcome = 'win'
                    exit_price = tp * (1 + slippage)
                    break
        
        # 4. Calculate Gross PnL
        if side == 'long':
            gross_pnl = (exit_price - real_entry) / real_entry
        else:
            gross_pnl = (real_entry - exit_price) / real_entry
            
        # 5. Subtract Fees (Round Trip)
        net_pnl = gross_pnl - (FEE_RATE * 2)
        
        return net_pnl * 100, outcome

    def monte_carlo_check(self, trades):
        """
        Run Monte Carlo simulation on a list of trade PnLs.
        Returns: (pass_boolean, max_dd_95th_percentile)
        """
        if len(trades) < 30: return False, 0.0
        
        pnls = [t['pnl'] for t in trades]
        max_dds = []
        
        for _ in range(MONTE_CARLO_RUNS):
            random.shuffle(pnls)
            equity = 100.0
            peak = 100.0
            max_dd = 0.0
            
            for p in pnls:
                equity *= (1 + p/100)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            
            max_dds.append(max_dd)
            
        # Check 95th percentile MaxDD (worst 5% of cases)
        max_dds.sort()
        idx_95 = int(MONTE_CARLO_RUNS * 0.95)
        dd_95 = max_dds[idx_95]
        
        return dd_95 < MAX_DD_THRESHOLD, dd_95

    def process_symbol(self, df, symbol):
        df = self.calculate_indicators(df)
        if len(df) < 1000: return None
        
        # --- SPLIT DATA ---
        n = len(df)
        train_end = int(n * TRAIN_SPLIT)
        test_end = int(n * (TRAIN_SPLIT + TEST_SPLIT))
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        holdout_df = df.iloc[test_end:]
        
        # --- STEP 1: DISCOVERY (TRAIN) ---
        combos = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        # Vectorized signal scan (simplified for speed, then detailed check)
        # Using same logic as bot: BBW > 0.45, VolRatio > 0.8
        sigs = (train_df['bbw_pct'] > 0.45) & (train_df['vol_ratio'] > 0.8)
        
        for idx in train_df[sigs].index:
            if idx >= train_df.index[-100]: continue # Skip end
            
            row = train_df.loc[idx]
            combo = self.get_combo(row)
            
            # Check Long
            if row['close'] > row['open']:
                # Simulate simple outcome for discovery (fast)
                entry = train_df.loc[idx, 'close'] # Use close for rough discovery
                atr = row['atr']
                tp = entry + 2*atr
                sl = entry - 2*atr
                
                future = train_df.loc[idx:].iloc[1:100]
                win = False
                for _, f in future.iterrows():
                    if f['low'] <= sl: break
                    if f['high'] >= tp: win = True; break
                
                k = ('long', combo)
                combos[k]['total'] += 1
                if win: combos[k]['wins'] += 1
                
            # Check Short
            elif row['close'] < row['open']:
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
                combos[k]['total'] += 1
                if win: combos[k]['wins'] += 1

        # Select Candidates (WR > 60% in Train)
        candidates = []
        for (side, combo), stats in combos.items():
            if stats['total'] >= 30 and (stats['wins']/stats['total']) > 0.60:
                candidates.append({'side': side, 'combo': combo, 'train_stats': stats})
        
        if not candidates: return None

        # --- STEP 2: VALIDATION (TEST) ---
        # Strict execution simulation
        validated = []
        
        for cand in candidates:
            side = cand['side']
            combo = cand['combo']
            
            trades = []
            
            # Scan Test Data
            t_sigs = (test_df['bbw_pct'] > 0.45) & (test_df['vol_ratio'] > 0.8)
            if side == 'long':
                t_sigs = t_sigs & (test_df['close'] > test_df['open'])
            else:
                t_sigs = t_sigs & (test_df['close'] < test_df['open'])
                
            for idx in test_df[t_sigs].index:
                if idx >= test_df.index[-100]: continue
                row = test_df.loc[idx]
                if self.get_combo(row) != combo: continue
                
                # REALISTIC ENTRY: Next Open
                next_idx = test_df.index.get_loc(idx) + 1
                if next_idx >= len(test_df): continue
                
                entry_price = test_df.iloc[next_idx]['open']
                atr = row['atr']
                
                if side == 'long':
                    tp = entry_price + 2*atr
                    sl = entry_price - 2*atr
                else:
                    tp = entry_price - 2*atr
                    sl = entry_price + 2*atr
                
                future = test_df.iloc[next_idx:].iloc[1:100]
                
                pnl, outcome = self.simulate_trade(entry_price, side, sl, tp, future, stress_test=False)
                trades.append({'pnl': pnl, 'outcome': outcome})
            
            # Filter: Must be profitable in Test
            if not trades: continue
            
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
            
            if total_pnl > 0 and win_rate > 0.50 and len(trades) >= 15:
                # --- STEP 3: STRESS TEST ---
                # Run Monte Carlo
                mc_pass, mc_dd = self.monte_carlo_check(trades)
                
                if mc_pass:
                    cand['test_stats'] = {'n': len(trades), 'pnl': total_pnl, 'wr': win_rate, 'mc_dd': mc_dd}
                    validated.append(cand)

        if not validated: return None

        # --- STEP 4: THE VAULT (HOLDOUT) ---
        # Final Exam for survivors
        final_survivors = []
        
        for cand in validated:
            side = cand['side']
            combo = cand['combo']
            trades = []
            
            h_sigs = (holdout_df['bbw_pct'] > 0.45) & (holdout_df['vol_ratio'] > 0.8)
            if side == 'long':
                h_sigs = h_sigs & (holdout_df['close'] > holdout_df['open'])
            else:
                h_sigs = h_sigs & (holdout_df['close'] < holdout_df['open'])
                
            for idx in holdout_df[h_sigs].index:
                if idx >= holdout_df.index[-100]: continue
                row = holdout_df.loc[idx]
                if self.get_combo(row) != combo: continue
                
                next_idx = holdout_df.index.get_loc(idx) + 1
                if next_idx >= len(holdout_df): continue
                
                entry_price = holdout_df.iloc[next_idx]['open']
                atr = row['atr']
                
                if side == 'long':
                    tp = entry_price + 2*atr
                    sl = entry_price - 2*atr
                else:
                    tp = entry_price - 2*atr
                    sl = entry_price + 2*atr
                
                future = holdout_df.iloc[next_idx:].iloc[1:100]
                
                # STRESS TEST ENABLED FOR HOLDOUT
                pnl, outcome = self.simulate_trade(entry_price, side, sl, tp, future, stress_test=True)
                trades.append({'pnl': pnl, 'outcome': outcome})
            
            if not trades: continue
            
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
            
            # Final Criteria: Profitable even with stress
            if total_pnl > 0 and win_rate > 0.50:
                cand['holdout_stats'] = {'n': len(trades), 'pnl': total_pnl, 'wr': win_rate}
                final_survivors.append(cand)
                
        if not final_survivors: return None
        
        return {'symbol': symbol, 'strategies': final_survivors}

async def fetch_klines(session, symbol):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=DAYS)).timestamp() * 1000)
    all_klines = []
    
    while True:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": TIMEFRAME, "limit": LIMIT, "end": end_time}
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data['retCode'] != 0: break
                klines = data['result']['list']
                if not klines: break
                all_klines.extend(klines)
                last_ts = int(klines[-1][0])
                if last_ts <= start_time: break
                end_time = last_ts - 1
        except Exception: break
            
    if not all_klines: return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['startTime'] = pd.to_numeric(df['startTime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df.sort_values('startTime').reset_index(drop=True)
    return df

async def worker(queue, session, results, engine):
    while True:
        symbol = await queue.get()
        try:
            df = await fetch_klines(session, symbol)
            if not df.empty:
                res = engine.process_symbol(df, symbol)
                if res:
                    results.append(res)
                    # Log summary
                    for s in res['strategies']:
                        logger.info(f"✅ {symbol} {s['side'].upper()} {s['combo']} | TestWR: {s['test_stats']['wr']*100:.1f}% | HoldoutPNL: {s['holdout_stats']['pnl']:.2f}%")
        except Exception as e:
            logger.error(f"Error {symbol}: {e}")
        finally:
            queue.task_done()

async def run():
    # Load Symbols
    try:
        with open('symbols_400.yaml', 'r') as f:
            sym_data = yaml.safe_load(f)
            symbols = sym_data['symbols']
    except FileNotFoundError:
        logger.error("symbols_400.yaml not found!")
        return

    logger.info(f"🚀 Starting V2 Realistic Backtest on {len(symbols)} symbols")
    logger.info(f"Config: Fee={FEE_RATE*100}%, Slip={SLIPPAGE_BASE*100}%, MC_Runs={MONTE_CARLO_RUNS}")
    
    engine = BacktestEngine()
    queue = asyncio.Queue()
    for s in symbols: queue.put_nowait(s)
    
    results = []
    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(queue, session, results, engine)) for _ in range(20)]
        await queue.join()
        for w in workers: w.cancel()
        
    # Generate Report & Overrides
    logger.info("Generating V2 Report...")
    
    yaml_out = {}
    csv_lines = ["Symbol,Side,Combo,TrainWR,TestWR,TestPnL,HoldoutWR,HoldoutPnL,MC_DD"]
    
    for r in results:
        sym = r['symbol']
        yaml_out[sym] = {}
        
        for s in r['strategies']:
            side = s['side']
            combo = s['combo']
            
            if side not in yaml_out[sym]: yaml_out[sym][side] = []
            yaml_out[sym][side].append(combo)
            
            csv_lines.append(f"{sym},{side},{combo},{s['train_stats']['wins']/s['train_stats']['total']*100:.1f},{s['test_stats']['wr']*100:.1f},{s['test_stats']['pnl']:.2f},{s['holdout_stats']['wr']*100:.1f},{s['holdout_stats']['pnl']:.2f},{s['test_stats']['mc_dd']*100:.1f}")
            
    # Save CSV
    with open('backtest_v2_results.csv', 'w') as f:
        f.write("\n".join(csv_lines))
        
    # Save YAML
    with open('symbol_overrides_v2.yaml', 'w') as f:
        f.write("# V2 Realistic Backtest Results (Strict Execution + Stress Test)\n")
        f.write(yaml.dump(yaml_out))
        
    logger.info(f"✅ V2 Backtest Complete. {len(results)} symbols survived.")
    logger.info("Saved to backtest_v2_results.csv and symbol_overrides_v2.yaml")

if __name__ == "__main__":
    asyncio.run(run())
