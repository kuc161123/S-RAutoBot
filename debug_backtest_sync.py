import pandas as pd
import pandas_ta as ta
import logging
import requests
from datetime import datetime, timedelta
import time
import random

# Configure Logging to Console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Mock Engine Logic (copied from backtest_wf_v2.py)
class BacktestEngine:
    def calculate_indicators(self, df):
        try:
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['vp'] = df['tp'] * df['volume']
            df['vwap'] = df['vp'].rolling(20).sum() / df['volume'].rolling(20).sum()
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['vwap_dist_atr'] = (df['close'] - df['vwap']).abs() / df['atr']
            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            df['roll_max'] = df['high'].rolling(50).max()
            df['roll_min'] = df['low'].rolling(50).min()
            df['fib_ret'] = (df['roll_max'] - df['close']) / (df['roll_max'] - df['roll_min']) * 100
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_w'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['close']
            df['bbw_pct'] = df['bb_w'].rolling(100).rank(pct=True)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
            return df.dropna()
        except Exception:
            return pd.DataFrame()

    def get_combo(self, row):
        rsi = row['rsi']
        if rsi < 30: r_bin = '<30'
        elif rsi < 40: r_bin = '30-40'
        elif rsi < 60: r_bin = '40-60'
        elif rsi < 70: r_bin = '60-70'
        else: r_bin = '70+'
        m_bin = 'bull' if row['macd_hist'] > 0 else 'bear'
        v = row['vwap_dist_atr']
        if v < 0.6: v_bin = '<0.6'
        elif v < 1.2: v_bin = '0.6-1.2'
        else: v_bin = '1.2+'
        f = row['fib_ret']
        if f < 23.6: f_bin = '0-23'
        elif f < 38.2: f_bin = '23-38'
        elif f < 50: f_bin = '38-50'
        elif f < 61.8: f_bin = '50-61'
        elif f < 78.6: f_bin = '61-78'
        else: f_bin = '78-100'
        mtf = 'MTF' if row['ema20'] > row['ema50'] else 'noMTF'
        return f"RSI:{r_bin} MACD:{m_bin} VWAP:{v_bin} Fib:{f_bin} {mtf}"

    def simulate_trade(self, entry_price, side, sl, tp, future_candles, stress_test=False):
        if stress_test and random.random() < 0.05: return 0.0, 'missed'
        spread = 0.0004 if (stress_test and random.random() < 0.10) else 0.0002
        slippage = 0.0003
        if side == 'long': real_entry = entry_price * (1 + (spread/2) + slippage)
        else: real_entry = entry_price * (1 - (spread/2) - slippage)
        outcome = 'timeout'
        exit_price = real_entry
        for _, row in future_candles.iterrows():
            if side == 'long':
                if row['low'] <= sl:
                    outcome = 'loss'
                    exit_price = sl * (1 - slippage)
                    break
                if row['high'] >= tp:
                    outcome = 'win'
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
        if side == 'long': gross_pnl = (exit_price - real_entry) / real_entry
        else: gross_pnl = (real_entry - exit_price) / real_entry
        net_pnl = gross_pnl - (0.00055 * 2)
        return net_pnl * 100, outcome

def fetch_klines_sync(symbol):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=60)).timestamp() * 1000)
    all_klines = []
    
    print(f"Fetching data for {symbol}...", end="", flush=True)
    while True:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": "3", "limit": 1000, "end": end_time}
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            if data['retCode'] != 0: break
            klines = data['result']['list']
            if not klines: break
            all_klines.extend(klines)
            last_ts = int(klines[-1][0])
            if last_ts <= start_time: break
            end_time = last_ts - 1
            print(".", end="", flush=True)
            time.sleep(0.1) # Rate limit nice
        except Exception as e:
            print(f" Error: {e}")
            break
            
    print(" Done.")
    if not all_klines: return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['startTime'] = pd.to_numeric(df['startTime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df.sort_values('startTime').reset_index(drop=True)
    return df

def debug_symbol_sync(symbol):
    df = fetch_klines_sync(symbol)
    if df.empty:
        print("❌ No data fetched")
        return

    print(f"📊 Data: {len(df)} candles")
    engine = BacktestEngine()
    df = engine.calculate_indicators(df)
    
    n = len(df)
    train_end = int(n * 0.60)
    test_end = int(n * 0.80)
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:test_end]
    
    print(f"🔹 Train: {len(train_df)} | Test: {len(test_df)}")
    
    # Discovery
    combos = {}
    sigs = (train_df['bbw_pct'] > 0.45) & (train_df['vol_ratio'] > 0.8)
    for idx in train_df[sigs].index:
        if idx >= train_df.index[-100]: continue
        row = train_df.loc[idx]
        combo = engine.get_combo(row)
        if row['close'] < row['open']: # Short only for debug
            k = ('short', combo)
            if k not in combos: combos[k] = {'wins': 0, 'total': 0}
            combos[k]['total'] += 1
            # Simple outcome check
            entry = train_df.loc[idx, 'close']
            atr = row['atr']
            tp = entry - 2*atr
            sl = entry + 2*atr
            future = train_df.loc[idx:].iloc[1:100]
            win = False
            for _, f in future.iterrows():
                if f['high'] >= sl: break
                if f['low'] <= tp: win = True; break
            if win: combos[k]['wins'] += 1

    candidates = []
    for (side, combo), stats in combos.items():
        wr = (stats['wins']/stats['total']) * 100
        if stats['total'] >= 30 and wr > 60:
            print(f"✅ CANDIDATE: {side} {combo} | WR: {wr:.1f}%")
            candidates.append({'side': side, 'combo': combo})

    if not candidates:
        print("❌ NO CANDIDATES IN TRAIN")
        return

    # Validation
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
            pnl, outcome = engine.simulate_trade(entry_price, side, sl, tp, future)
            trades.append({'pnl': pnl})
            
        if not trades:
            print("  ❌ No trades in Test")
            continue
            
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        print(f"  📊 Test Result: PnL={total_pnl:.2f}% | WR={win_rate:.1f}% | Trades={len(trades)}")

if __name__ == "__main__":
    debug_symbol_sync("ETHUSDT")
