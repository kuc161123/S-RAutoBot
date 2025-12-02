import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
from autobot.brokers.bybit import Bybit, BybitConfig
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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

async def backtest_symbol(bybit, sym, idx, total):
    """Backtest a single symbol"""
    try:
        logger.info(f"[{idx}/{total}] {sym}...")
        
        # Fetch 50 requests = ~10,000 candles
        klines = []
        end_ts = None
        
        for i in range(50):
            k = bybit.get_klines(sym, '3', limit=200, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.03)  # Rate limiting
                
        if len(klines) < 2000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data ({len(klines)} candles)")
            return None
            
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
        
        # Costs
        SLIPPAGE = 0.0004  # 0.04%
        FEES = 0.0012      # 0.12% round trip
        TOTAL_COST = SLIPPAGE + FEES

        # Longs
        for ridx in df[long_sigs].index:
            # Get integer location of the signal
            try:
                sig_iloc = df.index.get_loc(ridx)
                if sig_iloc >= len(df) - 1:
                    continue
                
                # Entry on NEXT candle open
                entry_candle = df.iloc[sig_iloc + 1]
                entry_raw = entry_candle['open']
                atr = df.iloc[sig_iloc]['atr'] # ATR from signal candle
                
                # Apply slippage to entry
                entry_real = entry_raw * (1 + SLIPPAGE)
                
                sl_raw = entry_raw - 2 * atr
                tp_raw = entry_raw + 2 * atr
                
                outcome = 'loss'
                # Future from entry candle onwards
                future = df.iloc[sig_iloc + 1:].iloc[1:100]
                
                for _, f_row in future.iterrows():
                    if f_row['low'] <= sl_raw:
                        outcome = 'loss'
                        break
                    if f_row['high'] >= tp_raw:
                        # Check if profitable after costs
                        exit_real = tp_raw * (1 - SLIPPAGE)
                        pnl_pct = (exit_real - entry_real) / entry_real
                        if pnl_pct > FEES:
                            outcome = 'win'
                        else:
                            outcome = 'loss' 
                        break
                
                long_signals.append({'combo': get_combo(df.loc[ridx]), 'outcome': outcome})
            except Exception:
                continue
            
        # Shorts
        for ridx in df[short_sigs].index:
            try:
                sig_iloc = df.index.get_loc(ridx)
                if sig_iloc >= len(df) - 1:
                    continue
                    
                # Entry on NEXT candle open
                entry_candle = df.iloc[sig_iloc + 1]
                entry_raw = entry_candle['open']
                atr = df.iloc[sig_iloc]['atr']
                
                # Apply slippage to entry
                entry_real = entry_raw * (1 - SLIPPAGE)
                
                sl_raw = entry_raw + 2 * atr
                tp_raw = entry_raw - 2 * atr
                
                outcome = 'loss'
                future = df.iloc[sig_iloc + 1:].iloc[1:100]
                
                for _, f_row in future.iterrows():
                    if f_row['high'] >= sl_raw:
                        outcome = 'loss'
                        break
                    if f_row['low'] <= tp_raw:
                        # Check if profitable after costs
                        exit_real = tp_raw * (1 + SLIPPAGE)
                        pnl_pct = (entry_real - exit_real) / entry_real
                        if pnl_pct > FEES:
                            outcome = 'win'
                        else:
                            outcome = 'loss'
                        break
                        
                short_signals.append({'combo': get_combo(df.loc[ridx]), 'outcome': outcome})
            except Exception:
                continue
            
        # Stats PER SIDE (STRICTER: WR > 60%, N >= 30)
        best_long = None
        best_short = None
        
        if long_signals:
            res_df = pd.DataFrame(long_signals)
            stats = res_df.groupby('combo')['outcome'].value_counts().unstack(fill_value=0)
            if 'win' not in stats.columns: stats['win'] = 0
            stats['total'] = stats.sum(axis=1)
            stats['wr'] = (stats['win'] / stats['total']) * 100
            
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
                msg.append(f"LONG: WR={best_long['wr']:.1f}%, N={best_long['n']}")
            if best_short:
                msg.append(f"SHORT: WR={best_short['wr']:.1f}%, N={best_short['n']}")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            return {
                'symbol': sym,
                'long': best_long,
                'short': best_short
            }
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No high-prob combo (WR>60%, N>=30)")
            return None
            
    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

async def run():
    # Load symbols from the generated file
    try:
        with open('symbols_400.yaml', 'r') as f:
            sym_data = yaml.safe_load(f)
            symbols = sym_data['symbols']
    except FileNotFoundError:
        print("❌ symbols_400.yaml not found. Running fetch_symbols.py first...")
        # Fallback or exit
        return

    # Load config for API keys
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    print(f"{'='*80}")
    print(f"🔬 MASSIVE BACKTEST: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Data: ~10,000 candles per symbol (50 API requests)")
    print(f"Filter: WR > 60% AND N >= 30 (SIDE-SPECIFIC)")
    print(f"Costs: Slippage 0.04% + Fees 0.12% = 0.16% per trade")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Process in batches of 10 for parallelization
    BATCH_SIZE = 10
    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[batch_start:batch_start + BATCH_SIZE]
        
        # Create tasks for this batch
        tasks = []
        for i, sym in enumerate(batch_symbols):
            idx = batch_start + i + 1
            tasks.append(backtest_symbol(bybit, sym, idx, len(symbols)))
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect valid results
        for result in batch_results:
            if result and not isinstance(result, Exception):
                results.append(result)
        
        # Progress update
        print(f"\n{'='*80}")
        print(f"Progress: {min(batch_start + BATCH_SIZE, len(symbols))}/{len(symbols)} symbols processed | {len(results)} combos found")
        print(f"{'='*80}\n")
    
    # Generate YAML output
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(results)} symbols with validated combos")
    print(f"{'='*80}\n")
    
    yaml_lines = [
        "# Ultra-Deep Backtest Results (400 Symbols)",
        "# Filter: WR > 60% AND N >= 30 (Side-Specific)",
        f"# Processed: {len(symbols)} symbols",
        f"# Passed: {len(results)} symbols",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
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
    
    with open('symbol_overrides_400.yaml', 'w') as f:
        f.write(output)
    
    print(f"✅ Saved to symbol_overrides_400.yaml")
    print(f"✅ {len(results)} symbols with validated combos")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(run())
