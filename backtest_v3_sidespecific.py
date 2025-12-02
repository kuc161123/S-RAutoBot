import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
from autobot.brokers.bybit import Bybit, BybitConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

def calculate_indicators(df):
    """Vectorized calculation of all features"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI 14
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    df['rsi'] = rsi
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    
    # VWAP (Rolling 20)
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()
    atr = (high - low).rolling(14).mean()
    df['atr'] = atr
    df['vwap_dist_atr'] = (close - vwap).abs() / atr
    
    # MTF (EMA 20 vs 50)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df['mtf_agree'] = ema20 > ema50
    
    # Fib Zone (50 bar lookback)
    roll_max = high.rolling(50).max()
    roll_min = low.rolling(50).min()
    fib_ret = (roll_max - close) / (roll_max - roll_min)
    df['fib_ret'] = fib_ret * 100
    
    # BB Width Pct
    std = close.rolling(20).std()
    ma = close.rolling(20).mean()
    upper = ma + 2*std
    lower = ma - 2*std
    bbw = (upper - lower) / close
    df['bbw_pct'] = bbw.rolling(100).rank(pct=True)
    
    # Volume Ratio
    vol_ma = volume.rolling(20).mean()
    df['vol_ratio'] = volume / vol_ma
    
    return df

def get_combo(row):
    # RSI Bin
    r = row['rsi']
    if r < 30: rb = '<30'
    elif r < 40: rb = '30-40'
    elif r < 60: rb = '40-60'
    elif r < 70: rb = '60-70'
    else: rb = '70+'
    
    # MACD Bin
    mb = 'bull' if row['macd_hist'] > 0 else 'bear'
    
    # VWAP Bin
    v = row['vwap_dist_atr']
    if v < 0.6: vb = '<0.6'
    elif v < 1.2: vb = '0.6-1.2'
    else: vb = '1.2+'
    
    # Fib Bin
    f = row['fib_ret']
    if f < 23.6: fb = '0-23'
    elif f < 38.2: fb = '23-38'
    elif f < 50.0: fb = '38-50'
    elif f < 61.8: fb = '50-61'
    elif f < 78.6: fb = '61-78'
    else: fb = '78-100'
    
    # MTF
    ma = 'MTF' if row['mtf_agree'] else 'noMTF'
    
    return f"RSI:{rb} MACD:{mb} VWAP:{vb} Fib:{fb} {ma}"

async def run():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    symbols = cfg['trade']['symbols']
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    final_results = []
    
    print(f"🔬 ULTRA-DEEP Backtest: Processing {len(symbols)} symbols...")
    print("Fetching ~10,000 candles per symbol (50 API requests)")
    print("Filter: WR > 60% AND N >= 30 (SIDE-SPECIFIC)\n")
    
    for idx, sym in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] {sym}...", end=" ", flush=True)
        try:
            # Fetch MORE Data (50 requests = ~10,000 candles)
            klines = []
            end_ts = None
            for i in range(50):
                k = bybit.get_klines(sym, '3', limit=200, end=end_ts)
                if not k: break
                klines = k + klines
                end_ts = int(k[0][0]) - 1
                await asyncio.sleep(0.05)
                if i % 10 == 9:
                    print(f"{(i+1)*200}...", end=" ", flush=True)
                
            if len(klines) < 2000:
                print("❌ Insufficient data")
                continue
                
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
            df.set_index('start', inplace=True)
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            
            # Calculate Indicators
            df = calculate_indicators(df)
            df = df.dropna()
            
            # Simulate Signals (SIDE-SPECIFIC)
            long_sigs = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] > df['open'])
            short_sigs = (df['bbw_pct'] > 0.45) & (df['vol_ratio'] > 0.8) & (df['close'] < df['open'])
            
            long_signals = []
            short_signals = []
            
            # Longs
            for ridx in df[long_sigs].index:
                row = df.loc[ridx]
                entry = row['close']
                atr = row['atr']
                sl = entry - 2 * atr
                tp = entry + 2 * atr
                
                outcome = 'loss'
                future = df.loc[ridx:].iloc[1:100]
                for _, f_row in future.iterrows():
                    if f_row['low'] <= sl:
                        outcome = 'loss'; break
                    if f_row['high'] >= tp:
                        outcome = 'win'; break
                
                long_signals.append({'combo': get_combo(row), 'outcome': outcome})
                
            # Shorts
            for ridx in df[short_sigs].index:
                row = df.loc[ridx]
                entry = row['close']
                atr = row['atr']
                sl = entry + 2 * atr
                tp = entry - 2 * atr
                
                outcome = 'loss'
                future = df.loc[ridx:].iloc[1:100]
                for _, f_row in future.iterrows():
                    if f_row['high'] >= sl:
                        outcome = 'loss'; break
                    if f_row['low'] <= tp:
                        outcome = 'win'; break
                        
                short_signals.append({'combo': get_combo(row), 'outcome': outcome})
                
            # Stats PER SIDE
            best_long = None
            best_short = None
            
            if long_signals:
                res_df = pd.DataFrame(long_signals)
                stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
                if 'win' not in stats.columns: stats['win'] = 0
                stats['total'] = stats.sum(axis=1)
                stats['wr'] = (stats['win'] / stats['total']) * 100
                
                # STRICTER: WR > 60% AND N >= 30
                best = stats[(stats['wr'] > 60) & (stats['total'] >= 30)].sort_values('wr', ascending=False)
                if not best.empty:
                    top = best.iloc[0]
                    best_long = {'combo': best.index[0], 'wr': top['wr'], 'n': int(top['total'])}
            
            if short_signals:
                res_df = pd.DataFrame(short_signals)
                stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
                if 'win' not in stats.columns: stats['win'] = 0
                stats['total'] = stats.sum(axis=1)
                stats['wr'] = (stats['win'] / stats['total']) * 100
                
                best = stats[(stats['wr'] > 60) & (stats['total'] >= 30)].sort_values('wr', ascending=False)
                if not best.empty:
                    top = best.iloc[0]
                    best_short = {'combo': best.index[0], 'wr': top['wr'], 'n': int(top['total'])}
            
            # Report
            if best_long or best_short:
                msg = []
                if best_long:
                    msg.append(f"LONG: {best_long['combo'][:30]}... (WR={best_long['wr']:.1f}%, N={best_long['n']})")
                if best_short:
                    msg.append(f"SHORT: {best_short['combo'][:30]}... (WR={best_short['wr']:.1f}%, N={best_short['n']})")
                print("✅ " + " | ".join(msg))
                
                final_results.append({
                    'symbol': sym,
                    'long': best_long,
                    'short': best_short
                })
            else:
                print("⚠️ No high-prob combo (WR>60%, N>=30)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Generate YAML output with SIDE-SPECIFIC combos
    print(f"\n{'='*60}")
    print("RESULTS - symbol_overrides_v3.yaml (SIDE-SPECIFIC):")
    print(f"{'='*60}\n")
    
    yaml_lines = ["# Symbol-Specific Golden Combos (ULTRA-DEEP, SIDE-SPECIFIC)", "# Filter: WR > 60% AND N >= 30", ""]
    for r in final_results:
        yaml_lines.append(f"{r['symbol']}:")
        
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    - \"{r['long']['combo']}\"")
            yaml_lines.append(f"    # Backtest: WR={r['long']['wr']:.1f}%, N={r['long']['n']}")
        
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    - \"{r['short']['combo']}\"")
            yaml_lines.append(f"    # Backtest: WR={r['short']['wr']:.1f}%, N={r['short']['n']}")
        
        yaml_lines.append("")
    
    output = "\n".join(yaml_lines)
    print(output)
    
    with open('symbol_overrides_v3.yaml', 'w') as f:
        f.write(output)
    
    print(f"\n✅ Saved to symbol_overrides_v3.yaml ({len(final_results)} symbols with side-specific combos)")

if __name__ == "__main__":
    asyncio.run(run())
