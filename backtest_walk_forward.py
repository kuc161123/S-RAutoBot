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

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_walk_forward.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TIMEFRAME = '3'  # 3m
LIMIT = 1000     # Candles per chunk
DAYS = 120       # History depth (Increased from 60)
SLIPPAGE = 0.0004 # 0.04%
FEES = 0.0012     # 0.12% roundtrip
TRAIN_SPLIT = 0.7 # 70% Train, 30% Test

# Indicators (Same as Bot)
def calculate_indicators(df):
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

def get_combo(row):
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

async def fetch_klines(session, symbol):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=DAYS)).timestamp() * 1000)
    all_klines = []
    
    while True:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": TIMEFRAME,
            "limit": LIMIT,
            "end": end_time
        }
        
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data['retCode'] != 0:
                    break
                
                klines = data['result']['list']
                if not klines:
                    break
                    
                all_klines.extend(klines)
                last_ts = int(klines[-1][0])
                if last_ts <= start_time:
                    break
                end_time = last_ts - 1
                
        except Exception:
            break
            
    if not all_klines:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_klines, columns=['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['startTime'] = pd.to_numeric(df['startTime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df.sort_values('startTime').reset_index(drop=True)
    return df

def backtest_symbol(df, symbol):
    df = calculate_indicators(df)
    if len(df) < 500:
        return None
        
    # Split Data
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # --- TRAIN PHASE ---
    # Find best combo
    combos = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0.0})
    
    # Vectorized Signal Detection (Train)
    long_sigs = (train_df['bbw_pct'] > 0.45) & (train_df['vol_ratio'] > 0.8) & (train_df['close'] > train_df['open'])
    short_sigs = (train_df['bbw_pct'] > 0.45) & (train_df['vol_ratio'] > 0.8) & (train_df['close'] < train_df['open'])
    
    # Evaluate Train Signals
    for idx in train_df[long_sigs | short_sigs].index:
        try:
            row = train_df.loc[idx]
            combo = get_combo(row)
            side = 'long' if row['close'] > row['open'] else 'short'
            
            # Outcome
            sig_iloc = train_df.index.get_loc(idx)
            if sig_iloc >= len(train_df) - 1: continue
            
            entry = train_df.iloc[sig_iloc+1]['open'] * (1 + SLIPPAGE)
            atr = train_df.iloc[sig_iloc]['atr']
            
            if side == 'long':
                tp = entry + 2 * atr
                sl = entry - 2 * atr
            else:
                tp = entry - 2 * atr
                sl = entry + 2 * atr
                
            # Check future
            future = train_df.iloc[sig_iloc+1:].iloc[1:100] # Max 100 bars hold
            win = False
            for _, f_row in future.iterrows():
                if side == 'long':
                    if f_row['low'] <= sl: break
                    if f_row['high'] >= tp: win = True; break
                else:
                    if f_row['high'] >= sl: break
                    if f_row['low'] <= tp: win = True; break
            
            key = (side, combo)
            combos[key]['total'] += 1
            if win: combos[key]['wins'] += 1
            
        except Exception:
            continue
            
    # Select Best Train Combos
    best_long = None
    best_short = None
    
    for (side, combo), stats in combos.items():
        if stats['total'] < 40: continue # Min 40 trades in Train (Increased from 20)
        wr = (stats['wins'] / stats['total']) * 100
        if wr > 60:
            if side == 'long':
                if not best_long or wr > best_long['wr']:
                    best_long = {'combo': combo, 'wr': wr, 'n': stats['total']}
            else:
                if not best_short or wr > best_short['wr']:
                    best_short = {'combo': combo, 'wr': wr, 'n': stats['total']}
    
    result = {'symbol': symbol, 'long': None, 'short': None}
    
    # --- TEST PHASE ---
    # Validate Best Combos on Test Data
    
    def run_test(target_combo, target_side):
        wins = 0
        total = 0
        
        # Filter Test Signals
        if target_side == 'long':
            sigs = (test_df['bbw_pct'] > 0.45) & (test_df['vol_ratio'] > 0.8) & (test_df['close'] > test_df['open'])
        else:
            sigs = (test_df['bbw_pct'] > 0.45) & (test_df['vol_ratio'] > 0.8) & (test_df['close'] < test_df['open'])
            
        for idx in test_df[sigs].index:
            try:
                row = test_df.loc[idx]
                if get_combo(row) != target_combo: continue
                
                sig_iloc = test_df.index.get_loc(idx)
                if sig_iloc >= len(test_df) - 1: continue
                
                entry = test_df.iloc[sig_iloc+1]['open'] * (1 + SLIPPAGE)
                atr = test_df.iloc[sig_iloc]['atr']
                
                if target_side == 'long':
                    tp = entry + 2 * atr
                    sl = entry - 2 * atr
                else:
                    tp = entry - 2 * atr
                    sl = entry + 2 * atr
                    
                future = test_df.iloc[sig_iloc+1:].iloc[1:100]
                win = False
                for _, f_row in future.iterrows():
                    if target_side == 'long':
                        if f_row['low'] <= sl: break
                        if f_row['high'] >= tp: win = True; break
                    else:
                        if f_row['high'] >= sl: break
                        if f_row['low'] <= tp: win = True; break
                
                total += 1
                if win: wins += 1
            except: continue
            
        return {'wr': (wins/total*100) if total > 0 else 0.0, 'n': total}

    if best_long:
        test_res = run_test(best_long['combo'], 'long')
        result['long'] = {
            'combo': best_long['combo'],
            'train_wr': best_long['wr'],
            'train_n': best_long['n'],
            'test_wr': test_res['wr'],
            'test_n': test_res['n']
        }
        
    if best_short:
        test_res = run_test(best_short['combo'], 'short')
        result['short'] = {
            'combo': best_short['combo'],
            'train_wr': best_short['wr'],
            'train_n': best_short['n'],
            'test_wr': test_res['wr'],
            'test_n': test_res['n']
        }
        
    return result

async def worker(queue, session, results):
    while True:
        symbol = await queue.get()
        try:
            df = await fetch_klines(session, symbol)
            if not df.empty:
                res = backtest_symbol(df, symbol)
                if res and (res['long'] or res['short']):
                    results.append(res)
                    
                    # Log progress
                    log_msg = f"[{len(results)}] {symbol} "
                    if res['long']:
                        l = res['long']
                        log_msg += f"LONG: Train={l['train_wr']:.1f}%({l['train_n']}) -> Test={l['test_wr']:.1f}%({l['test_n']}) "
                    if res['short']:
                        s = res['short']
                        log_msg += f"SHORT: Train={s['train_wr']:.1f}%({s['train_n']}) -> Test={s['test_wr']:.1f}%({s['test_n']})"
                    logger.info(log_msg)
                    
        except Exception as e:
            logger.error(f"Error {symbol}: {e}")
        finally:
            queue.task_done()

async def run():
    # Load symbols from the generated file
    try:
        with open('symbols_400.yaml', 'r') as f:
            sym_data = yaml.safe_load(f)
            symbols = sym_data['symbols']
    except FileNotFoundError:
        logger.error("symbols_400.yaml not found!")
        return

    # Load config for API keys (needed for fetch_klines)
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"Starting Walk-Forward Validation on {len(symbols)} symbols...")
    logger.info(f"Split: {TRAIN_SPLIT*100}% Train / {(1-TRAIN_SPLIT)*100}% Test")
    
    queue = asyncio.Queue()
    for s in symbols:
        queue.put_nowait(s)
        
    results = []
    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(queue, session, results)) for _ in range(10)]
        await queue.join()
        for w in workers: w.cancel()
        
    # Save Report
    logger.info("Generating Report...")
    csv_lines = ["Symbol,Side,Combo,Train_WR,Train_N,Test_WR,Test_N,Delta_WR"]
    
    pass_count = 0
    total_strategies = 0
    
    for r in results:
        if r['long']:
            l = r['long']
            delta = l['test_wr'] - l['train_wr']
            csv_lines.append(f"{r['symbol']},LONG,\"{l['combo']}\",{l['train_wr']:.1f},{l['train_n']},{l['test_wr']:.1f},{l['test_n']},{delta:.1f}")
            total_strategies += 1
            if l['test_wr'] > 50: pass_count += 1
            
        if r['short']:
            s = r['short']
            delta = s['test_wr'] - s['train_wr']
            csv_lines.append(f"{r['symbol']},SHORT,\"{s['combo']}\",{s['train_wr']:.1f},{s['train_n']},{s['test_wr']:.1f},{s['test_n']},{delta:.1f}")
            total_strategies += 1
            if s['test_wr'] > 50: pass_count += 1
            
    with open('walk_forward_results.csv', 'w') as f:
        f.write("\n".join(csv_lines))
        
    logger.info(f"✅ Walk-Forward Complete")
    logger.info(f"Strategies Tested: {total_strategies}")
    logger.info(f"Robust Strategies (>50% Test WR): {pass_count} ({pass_count/total_strategies*100:.1f}%)")
    logger.info(f"Saved to walk_forward_results.csv")

if __name__ == "__main__":
    asyncio.run(run())
