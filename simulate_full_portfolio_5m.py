import yaml
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
import concurrent.futures

# CONFIG
RISK_PCT = 0.001
INITIAL_CAPITAL = 700.0
BASE_URL = "https://api.bybit.com"

# MONTH DEFINITIONS
MONTHS = [
    ("JAN 2026", "2026-01-01 00:00:00", "2026-01-14 23:59:59"),
]

START_DATE = "2026-01-01 00:00:00"
END_DATE = "2026-01-14 23:59:59"

def fetch_klines(symbol, interval):
    # Fetch ample data covering Jan 2026
    end_req = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp() * 1000)
    
    # Needs buffer for 1H indicators (Dec 2025)
    if interval == '60':
        start_req = int(datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp() * 1000)
    else:
        # 5M needs to cover Jan 1
        start_req = int(datetime(2025, 12, 25, tzinfo=timezone.utc).timestamp() * 1000)

    current_end = end_req
    all_candles = []
    
    while current_end > start_req:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=5)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
            time.sleep(0.05) # Rate limit protection
        except: break
        
    if not all_candles: return None
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = df[c].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def prepare_1h(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ema'] = df['close'].ewm(span=200, adjust=False).mean()
    h, l, c_prev = df['high'], df['low'], df['close'].shift()
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df

def find_pivots(data, left=3, right=3):
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    for i in range(left, len(data) - right):
        window = data[i-left : i+right+1]
        center = data[i]
        if len(window) != (left + right + 1): continue
        if center == max(window) and list(window).count(center) == 1: pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1: pivot_lows[i] = center
    return pivot_highs, pivot_lows

def detect_signals(df):
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    rsi = df['rsi'].values; ema = df['ema'].values
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    used_pivots = set() 
    
    for i in range(205, len(df) - 3):
        curr_price = close[i]; curr_ema = ema[i]
        
        # BULLISH
        if curr_price > curr_ema:
            p_lows = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2: break
            if len(p_lows) == 2:
                curr_idx, curr_val = p_lows[0]; prev_idx, prev_val = p_lows[1]
                dedup_key = (curr_idx, prev_idx, 'BULL')
                if (i - curr_idx) <= 10:
                    if dedup_key not in used_pivots:
                         added = False
                         if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'long', 'type': 'REG_BULL', 'swing': max(high[curr_idx:i+1])})
                            added = True
                         if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'long', 'type': 'HID_BULL', 'swing': max(high[curr_idx:i+1])})
                            added = True
                         if added: used_pivots.add(dedup_key)

        # BEARISH
        if curr_price < curr_ema:
             p_highs = []
             for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_ph[j]):
                    p_highs.append((j, price_ph[j]))
                    if len(p_highs) >= 2: break
             if len(p_highs) == 2:
                curr_idx, curr_val = p_highs[0]; prev_idx, prev_val = p_highs[1]
                dedup_key = (curr_idx, prev_idx, 'BEAR')
                if (i - curr_idx) <= 10:
                    if dedup_key not in used_pivots:
                        added = False
                        if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'short', 'type': 'REG_BEAR', 'swing': min(low[curr_idx:i+1])})
                            added = True
                        if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'short', 'type': 'HID_BEAR', 'swing': min(low[curr_idx:i+1])})
                            added = True
                        if added: used_pivots.add(dedup_key)
    return signals

def process_symbol(args):
    sym, cfg = args
    
    # 1. Fetch 1H Data
    df_1h = fetch_klines(sym, '60')
    if df_1h is None or len(df_1h) < 300: return []
    df_1h = prepare_1h(df_1h)
    
    # 2. Fetch 5M Data (Heavy)
    df_5m = fetch_klines(sym, '5')
    if df_5m is None: return [] # 5M missing means no verification possible
    
    signals = detect_signals(df_1h)
    trades = []
    last_exit_time = 0
    
    StartTS = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000)
    EndTS = int(datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000)
    
    for sig in signals:
        if sig['type'] != cfg['divergence_type']: continue
        
        candle = df_1h.iloc[sig['conf_idx']]
        start_ts = int(candle['start'].timestamp()*1000)
        
        if start_ts < StartTS or start_ts > EndTS: continue
        if start_ts < last_exit_time: continue
        
        # Execution Check
        conf_idx = sig['conf_idx']
        side = sig['side']
        bos = sig['swing']
        
        entry_data = None
        for i in range(1, 7):
            if conf_idx+i+1 >= len(df_1h): break
            c = df_1h.iloc[conf_idx+i]
            triggered = (c['close'] > bos) if side == 'long' else (c['close'] < bos)
            if triggered:
                entry_c = df_1h.iloc[conf_idx+i+1]
                entry_data = {
                    'price': entry_c['open'], 
                    'ts': int(entry_c['start'].timestamp()*1000), 
                    'atr': c['atr']
                }
                break
        
        if not entry_data: continue
        if entry_data['ts'] < last_exit_time: continue
        
        # 5M PRECISION CHECK
        rr = cfg['rr']
        sl_dist = entry_data['atr'] * cfg['atr_mult']
        entry_price = entry_data['price']
        
        tp = entry_price + (sl_dist * rr) if side == 'long' else entry_price - (sl_dist * rr)
        sl = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
        
        outcome = -1.0
        exit_ts = entry_data['ts'] + (24*3600000) 
        
        sub = df_5m[df_5m['start'] >= pd.to_datetime(entry_data['ts'], unit='ms', utc=True)]
        
        hit = False
        for row in sub.itertuples():
            curr_ts = int(row.start.timestamp()*1000)
            if side == 'long':
                if row.low <= sl: outcome=-1.0; exit_ts=curr_ts; hit=True; break
                if row.high >= tp: outcome=rr; exit_ts=curr_ts; hit=True; break
            else:
                if row.high >= sl: outcome=-1.0; exit_ts=curr_ts; hit=True; break
                if row.low <= tp: outcome=rr; exit_ts=curr_ts; hit=True; break
        
        if not hit: outcome = 0 # Running
        
        last_exit_time = exit_ts
        trades.append({'sym': sym, 'side': side, 'price': entry_price, 'entry_time': entry_data['ts'], 'exit_time': exit_ts, 'r': outcome})
        
    return trades

def main():
    print(f"🔥 JAN 2026 MTD SIMULATION (235 SYMBOLS, 5M PRECISION)")
    start_time = time.time()
    
    with open('config_combined.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    tasks = list(conf['symbols'].items())
    print(f"Adding {len(tasks)} symbols to queue...")
    
    all_trades = []
    
    # Needs throttle because 5M data is huge and Bybit rate limits are 10/s roughly
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as exc:
        results = exc.map(process_symbol, tasks)
        processed = 0
        for r in results:
            all_trades.extend(r)
            processed += 1
            if processed % 10 == 0: print(f"Processed {processed}/{len(tasks)}...", end='\r')
            
    print(f"\nTotal Trades Found: {len(all_trades)}")
    
    # SIMULATE PORTFOLIO
    all_trades.sort(key=lambda x: x['entry_time'])
    
    balance = INITIAL_CAPITAL
    peak_balance = INITIAL_CAPITAL
    max_drawdown = 0
    
    monthly_stats = {m[0]: {'wins': 0, 'losses': 0, 'net_r': 0, 'start_bal': 0, 'end_bal': 0} for m in MONTHS}
    
    # Populate Start Balances roughly
    # Actually, we iterate linearly.
    
    # We need to map timestamps to months
    def get_month_key(ts):
         dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
         k = f"{dt.strftime('%b').upper()} {dt.year}"
         return k

    # Set initial start bal for first month
    monthly_stats[MONTHS[0][0]]['start_bal'] = INITIAL_CAPITAL
    
    current_month = None
    
    # Equity Curve
    equity = [INITIAL_CAPITAL]
    
    for t in all_trades:
        m_key = get_month_key(t['entry_time'])
        if m_key not in monthly_stats: continue # Out of bounds (Aug or Jan)
        
        if current_month != m_key:
             # Month switched
             if current_month: monthly_stats[current_month]['end_bal'] = balance
             monthly_stats[m_key]['start_bal'] = balance
             current_month = m_key
             
        risk_amt = balance * RISK_PCT
        realized_r = t['r'] - 0.1 # Fee
        pnl = realized_r * risk_amt
        balance += pnl
        
        equity.append(balance)
        if balance > peak_balance: peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        if dd > max_drawdown: max_drawdown = dd
        
        monthly_stats[m_key]['net_r'] += realized_r
        if t['r'] > 0: monthly_stats[m_key]['wins'] += 1
        else: monthly_stats[m_key]['losses'] += 1
        
    if current_month: monthly_stats[current_month]['end_bal'] = balance

    target_symbols = ['BICOUSDT', 'HEMIUSDT', 'LAUSDT', 'MIRAUSDT', 'MOVEUSDT', 'UBUSDT', 'BNBUSDT', 'FARTCOINUSDT']
    
    print("\n" + "="*80)
    print(f"📊 SPECIFIC SYMBOL CHECK (USER OPEN POSITIONS)")
    print(f"{'TIME (UTC)':<20} | {'SYMBOL':<12} | {'SIDE':<5} | {'PRICE':<10} | {'RESULT':<10}")
    print("-" * 80)
    
    found_any = False
    for t in all_trades:
        if t['sym'] in target_symbols:
            dt = datetime.fromtimestamp(t['entry_time']/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            status = "OPEN/RUNNING" if t['r'] == 0 else f"{t['r']} R"
            print(f"{dt:<20} | {t['sym']:<12} | {t['side'].upper():<5} | {t['price']:<10.4f} | {status:<10}")
            found_any = True
            
    if not found_any:
        print("No trades found for these symbols in the simulation period.")
        
    print("="*80)
    print(f"{'MONTH':<10} | {'TRADES (W/L)':<15} | {'WR%':<6} | {'NET R':<10} | {'START BAL':<12} | {'END BAL':<12} | {'CHANGE'}")
    print("="*80)
    
    for m_name, start, end in MONTHS:
        s = monthly_stats[m_name]
        total = s['wins'] + s['losses']
        wr = (s['wins']/total*100) if total > 0 else 0
        change = s['end_bal'] - s['start_bal']
        pct = (change / s['start_bal'] * 100) if s['start_bal'] > 0 else 0
        
        print(f"{m_name:<10} | {total:<4} ({s['wins']}/{s['losses']}){'':<5} | {wr:<6.1f} | {s['net_r']:<10.1f} | ${s['start_bal']:<12.2f} | ${s['end_bal']:<12.2f} | {pct:+.1f}%")
        
    print("="*80)
    print(f"FINAL BALANCE: ${balance:.2f}")
    print(f"TOTAL RETURN: {(balance - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.1f}%")
    print(f"MAX DRAWDOWN: {max_drawdown:.1f}%")
    print(f"TIME TAKEN: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
