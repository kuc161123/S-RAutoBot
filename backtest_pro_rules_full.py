#!/usr/bin/env python3
"""
Full Pro Rules Backtest
- Fetches all tradeable Bybit symbols (by market cap)
- Tests parameter combinations for LONG and SHORT separately per symbol
- Uses 3m timeframe with 5000 candles
- Filters results with WR > 40%
- Outputs per-symbol results
"""

import json
import pandas as pd
import numpy as np
import itertools
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import Bybit broker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autobot.brokers.bybit import Bybit, BybitConfig

class ProRulesBacktester:
    def __init__(self):
        # Initialize Bybit client
        bybit_cfg = BybitConfig(
            base_url=os.getenv('BYBIT_BASE_URL', 'https://api.bybit.com'),
            api_key=os.getenv('BYBIT_API_KEY', ''),
            api_secret=os.getenv('BYBIT_API_SECRET', ''),
            alt_base_url=os.getenv('BYBIT_ALT_BASE_URL'),
            use_alt_on_fail=True
        )
        self.bybit = Bybit(bybit_cfg)
        self.results = {}
        
    def get_all_symbols(self) -> List[str]:
        """Get all tradeable USDT linear symbols from Bybit, ordered by market cap"""
        print("Fetching all tradeable symbols from Bybit...")
        
        # Try to get symbols from config first (they're already ordered by market cap)
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                cfg = yaml.safe_load(f)
                config_symbols = cfg.get('trade', {}).get('symbols', [])
                if config_symbols:
                    print(f"Using {len(config_symbols)} symbols from config.yaml")
                    return config_symbols
        except Exception as e:
            print(f"Could not read config: {e}")
        
        # Fallback: try API
        try:
            # Get tickers - no auth needed for public endpoint
            import requests
            resp = requests.get("https://api.bybit.com/v5/market/tickers", params={
                "category": "linear",
                "settleCoin": "USDT"
            }, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("result"):
                    tickers = data["result"].get("list", [])
                    
                    # Filter for USDT pairs
                    symbols = []
                    for ticker in tickers:
                        symbol = ticker.get("symbol", "")
                        if symbol.endswith("USDT"):
                            # Get turnover24h as proxy for liquidity/market cap
                            turnover = float(ticker.get("turnover24h", 0) or 0)
                            symbols.append((symbol, turnover))
                    
                    # Sort by turnover descending
                    symbols.sort(key=lambda x: x[1], reverse=True)
                    symbol_list = [s[0] for s in symbols]
                    
                    print(f"Found {len(symbol_list)} tradeable symbols from API")
                    return symbol_list
        except Exception as e:
            print(f"Error fetching symbols from API: {e}")
        
        # Final fallback: return empty list
        print("No symbols found, returning empty list")
        return []
    
    def fetch_klines(self, symbol: str, timeframe: str = "3", limit: int = 5000) -> pd.DataFrame:
        """Fetch historical klines with pagination to get 5000 candles"""
        print(f"  Fetching {limit} candles for {symbol}...")
        all_klines = []
        
        # Bybit max is 200 per request, need to paginate
        batch_size = 200
        batches_needed = (limit + batch_size - 1) // batch_size
        
        import requests
        
        end_time = None
        for batch in range(batches_needed):
            try:
                batch_limit = min(batch_size, limit - len(all_klines))
                if batch_limit <= 0:
                    break
                
                # Use public API directly (no auth needed)
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": str(batch_limit)
                }
                if end_time is not None:
                    params["end"] = str(int(end_time))
                
                resp = requests.get("https://api.bybit.com/v5/market/kline", params=params, timeout=15)
                
                if resp.status_code != 200:
                    print(f"    API error: {resp.status_code}")
                    break
                
                data = resp.json()
                if not data or not data.get("result"):
                    break
                
                klines = data["result"].get("list", [])
                if not klines:
                    break
                
                # Bybit returns newest first, reverse for chronological order
                klines = list(reversed(klines))
                all_klines.extend(klines)
                
                # Set end_time for next batch (oldest candle timestamp - 1)
                if len(klines) > 0:
                    end_time = int(klines[0][0]) - 1
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"    Error fetching batch {batch+1}: {e}")
                break
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep only needed columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"    Fetched {len(df)} candles")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for pro rules"""
        if len(df) < 50:
            return df
        
        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # Bollinger Bands (20, 2)
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # ATR 14
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # VWAP (session-based approximation - using rolling 50 bars)
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['tpv'] = df['tp'] * df['volume']
        df['cum_tpv'] = df['tpv'].rolling(window=50).sum()
        df['cum_vol'] = df['volume'].rolling(window=50).sum()
        df['vwap'] = df['cum_tpv'] / df['cum_vol']
        df['vwap_dist'] = abs(df['close'] - df['vwap'])
        df['vwap_dist_atr'] = df['vwap_dist'] / df['atr']
        
        # Volume Ratio (vs 20 SMA)
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # Wick calculations
        df['body_top'] = np.maximum(df['open'], df['close'])
        df['body_bottom'] = np.minimum(df['open'], df['close'])
        df['upper_wick'] = df['high'] - df['body_top']
        df['lower_wick'] = df['body_bottom'] - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['upper_wick_ratio'] = df['upper_wick'] / df['candle_range']
        df['lower_wick_ratio'] = df['lower_wick'] / df['candle_range']
        
        # Fibonacci levels (simplified - using recent high/low)
        lookback = 50
        df['recent_high'] = df['high'].rolling(window=lookback).max()
        df['recent_low'] = df['low'].rolling(window=lookback).min()
        df['range_size'] = df['recent_high'] - df['recent_low']
        
        # Calculate fib retracement
        df['fib_ret'] = (df['close'] - df['recent_low']) / df['range_size']
        df['fib_pct'] = df['fib_ret'] * 100
        
        # Assign fib zones
        def get_fib_zone(pct):
            if pd.isna(pct):
                return 'none'
            if pct < 23.6:
                return '0-23'
            elif pct < 38.2:
                return '23-38'
            elif pct < 50.0:
                return '38-50'
            elif pct < 61.8:
                return '50-61'
            elif pct < 78.6:
                return '61-78'
            else:
                return '78-100'
        
        df['fib_zone'] = df['fib_pct'].apply(get_fib_zone)
        
        # MTF alignment (simplified - using 15m trend)
        # For backtest, we'll use a proxy: if price is above 20 EMA on 3m, consider it aligned
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['mtf_agree_15'] = (df['close'] > df['ema_20']).astype(int)  # Simplified proxy
        
        return df.dropna()
    
    def check_pro_rules(self, row: pd.Series, params: dict, side: str) -> bool:
        """Check if signal passes pro rules with all optimized parameters"""
        s = str(side).lower()
        
        # Extract values
        try:
            rsi = float(row['rsi'])
            macd_hist = float(row['macd_hist'])
            vwap_dist_atr = float(row['vwap_dist_atr'])
            vol_ratio = float(row['vol_ratio'])
            bb_width_pct = float(row['bb_width_pct'])
            atr_pct = float(row['atr_pct'])
            fib_zone = str(row['fib_zone'])
            upper_wick_ratio = float(row['upper_wick_ratio'])
            lower_wick_ratio = float(row['lower_wick_ratio'])
        except:
            return False
        
        # RSI check (optimized ranges)
        if s == 'long':
            rsi_ok = params['rsi_min_long'] <= rsi <= params['rsi_max_long']
        else:
            rsi_ok = params['rsi_min_short'] <= rsi <= params['rsi_max_short']
        
        # MACD check (optimized minimum)
        macd_ok = False
        if s == 'long':
            macd_ok = (macd_hist > 0) and (abs(macd_hist) >= params['macd_hist_min'])
        else:
            macd_ok = (macd_hist < 0) and (abs(macd_hist) >= params['macd_hist_min'])
        
        # VWAP check (optimized maximum)
        vwap_ok = vwap_dist_atr <= params['vwap_dist_max']
        
        # Volume check (optimized minimum)
        vol_ok = vol_ratio >= params['vol_ratio_min']
        
        # BB Width check (optimized minimum)
        bbw_ok = bb_width_pct >= params['bb_width_min']
        
        # ATR check (optimized minimum)
        atr_ok = atr_pct >= params['atr_pct_min']
        
        # Wick check (optimized parameters)
        wdelta = params['wick_delta_min']
        wick_ratio_min = params['wick_ratio_min']
        if s == 'long':
            wick_ok = (lower_wick_ratio >= upper_wick_ratio + wdelta) and (lower_wick_ratio >= wick_ratio_min)
        else:
            wick_ok = (upper_wick_ratio >= lower_wick_ratio + wdelta) and (upper_wick_ratio >= wick_ratio_min)
        
        # Fib check (optimized zones - can be list of allowed zones)
        allowed_fib_zones = params.get('fib_zones', ['23-38', '38-50', '50-61'])
        fib_ok = fib_zone in allowed_fib_zones
        
        # MTF check (simplified - using our proxy)
        mtf_ok = bool(row['mtf_agree_15']) if s == 'long' else True  # Simplified
        
        # All must pass
        return bool(rsi_ok and macd_ok and vwap_ok and vol_ok and bbw_ok and atr_ok and wick_ok and fib_ok and mtf_ok)
    
    def run_backtest(self, df: pd.DataFrame, params: dict, side: str) -> Tuple[int, int, float]:
        """Run backtest for given parameters and side"""
        if len(df) < 100:
            return 0, 0, 0.0
        
        total_trades = 0
        wins = 0
        
        # Scalp R:R from config (2.1:1)
        rr = 2.1
        tp_pct = 0.01  # 1% TP
        sl_pct = tp_pct / rr  # ~0.476% SL
        
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        
        for i in range(50, len(df) - 1):  # Start after indicators warm up
            row = df.iloc[i]
            
            if in_position:
                # Check exit conditions
                if side == 'long':
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                    
                    # Check if TP or SL hit
                    if row['high'] >= tp_price:
                        wins += 1
                        total_trades += 1
                        in_position = False
                    elif row['low'] <= sl_price:
                        total_trades += 1
                        in_position = False
                    # Timeout after 96 candles (24 hours of 15m, but we're on 3m so ~5 hours)
                    elif i - entry_idx >= 96:
                        # Check final outcome
                        if row['close'] >= entry_price * (1 + tp_pct * 0.5):  # Partial win
                            wins += 1
                        total_trades += 1
                        in_position = False
                else:  # short
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)
                    
                    if row['low'] <= tp_price:
                        wins += 1
                        total_trades += 1
                        in_position = False
                    elif row['high'] >= sl_price:
                        total_trades += 1
                        in_position = False
                    elif i - entry_idx >= 96:
                        if row['close'] <= entry_price * (1 - tp_pct * 0.5):
                            wins += 1
                        total_trades += 1
                        in_position = False
            else:
                # Check for entry signal
                if self.check_pro_rules(row, params, side):
                    in_position = True
                    entry_price = float(row['close'])
                    entry_idx = i
        
        # Close any remaining position
        if in_position:
            final_row = df.iloc[-1]
            if side == 'long':
                if final_row['close'] >= entry_price * (1 + tp_pct * 0.5):
                    wins += 1
            else:
                if final_row['close'] <= entry_price * (1 - tp_pct * 0.5):
                    wins += 1
            total_trades += 1
        
        wr = (wins / total_trades * 100) if total_trades > 0 else 0.0
        return total_trades, wins, wr
    
    def optimize_symbol(self, symbol: str) -> Dict:
        """Optimize pro rules parameters for a symbol (separate for long/short)"""
        print(f"\n{'='*60}")
        print(f"Optimizing: {symbol}")
        print(f"{'='*60}")
        
        # Fetch data
        df = self.fetch_klines(symbol, timeframe="3", limit=5000)
        if len(df) < 100:
            print(f"  ⚠️  Insufficient data: {len(df)} candles")
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        if len(df) < 100:
            print(f"  ⚠️  Insufficient data after indicators: {len(df)} candles")
            return None
        
        print(f"  ✓ Data ready: {len(df)} candles")
        
        # Expanded parameter grid - testing ALL pro rules parameters
        # Reduced ranges to keep search space manageable while still comprehensive
        # RSI ranges
        rsi_long_opts = [
            (35, 65), (38, 62), (40, 60), (42, 58)
        ]
        rsi_short_opts = [
            (30, 60), (33, 57), (35, 55), (38, 52)
        ]
        
        # VWAP distance (ATR)
        vwap_opts = [1.0, 1.2, 1.3, 1.5, 1.8]
        
        # Volume ratio
        vol_opts = [1.2, 1.5, 1.8, 2.0]
        
        # MACD histogram minimum
        macd_opts = [0.0003, 0.0005, 0.001, 0.0015]
        
        # BB width minimum
        bb_width_opts = [0.008, 0.010, 0.012, 0.015]
        
        # ATR percentage minimum
        atr_pct_opts = [0.04, 0.05, 0.06, 0.08]
        
        # Wick delta minimum
        wick_delta_opts = [0.08, 0.10, 0.12, 0.15]
        
        # Wick ratio minimum
        wick_ratio_opts = [0.20, 0.25, 0.30]
        
        # Fib zones (combinations of allowed zones)
        fib_zone_combos = [
            ['23-38', '38-50', '50-61'],  # Standard
            ['38-50', '50-61'],           # Higher zones
            ['23-38', '38-50'],           # Lower zones
            ['50-61', '61-78'],           # Upper zones
            ['38-50'],                    # Single zone
            ['50-61'],                    # Single zone
        ]
        
        # Test LONG - all parameter combinations
        print(f"  Testing LONG combinations (all parameters)...")
        long_combinations = list(itertools.product(
            rsi_long_opts, vwap_opts, vol_opts, macd_opts, 
            bb_width_opts, atr_pct_opts, wick_delta_opts, 
            wick_ratio_opts, fib_zone_combos
        ))
        print(f"    Total combinations: {len(long_combinations)}")
        long_results = []
        
        for i, (rsi_l, vwap, vol, macd, bbw, atr, wd, wr, fibz) in enumerate(long_combinations):
            params = {
                'rsi_min_long': rsi_l[0], 'rsi_max_long': rsi_l[1],
                'rsi_min_short': 35, 'rsi_max_short': 55,  # Dummy for long test
                'vwap_dist_max': vwap,
                'vol_ratio_min': vol,
                'macd_hist_min': macd,
                'bb_width_min': bbw,
                'atr_pct_min': atr,
                'wick_delta_min': wd,
                'wick_ratio_min': wr,
                'fib_zones': fibz
            }
            
            trades, wins, wr_val = self.run_backtest(df, params, 'long')
            
            if trades >= 10:  # Minimum trades threshold
                long_results.append({
                    'params': params,
                    'trades': trades,
                    'wins': wins,
                    'win_rate': wr_val,
                    'score': trades * wr_val  # Volume * Quality
                })
            
            if (i + 1) % 500 == 0:
                pct = (i + 1) / len(long_combinations) * 100
                print(f"    LONG Progress: {i+1}/{len(long_combinations)} ({pct:.1f}%) | Valid: {len(long_results)}")
        
        print(f"    LONG: Found {len(long_results)} valid configurations")
        
        # Test SHORT - all parameter combinations
        print(f"  Testing SHORT combinations (all parameters)...")
        short_combinations = list(itertools.product(
            rsi_short_opts, vwap_opts, vol_opts, macd_opts,
            bb_width_opts, atr_pct_opts, wick_delta_opts,
            wick_ratio_opts, fib_zone_combos
        ))
        print(f"    Total combinations: {len(short_combinations)}")
        short_results = []
        
        for i, (rsi_s, vwap, vol, macd, bbw, atr, wd, wr, fibz) in enumerate(short_combinations):
            params = {
                'rsi_min_long': 40, 'rsi_max_long': 60,  # Dummy for short test
                'rsi_min_short': rsi_s[0], 'rsi_max_short': rsi_s[1],
                'vwap_dist_max': vwap,
                'vol_ratio_min': vol,
                'macd_hist_min': macd,
                'bb_width_min': bbw,
                'atr_pct_min': atr,
                'wick_delta_min': wd,
                'wick_ratio_min': wr,
                'fib_zones': fibz
            }
            
            trades, wins, wr_val = self.run_backtest(df, params, 'short')
            
            if trades >= 10:
                short_results.append({
                    'params': params,
                    'trades': trades,
                    'wins': wins,
                    'win_rate': wr_val,
                    'score': trades * wr_val
                })
            
            if (i + 1) % 500 == 0:
                pct = (i + 1) / len(short_combinations) * 100
                print(f"    SHORT Progress: {i+1}/{len(short_combinations)} ({pct:.1f}%) | Valid: {len(short_results)}")
        
        print(f"    SHORT: Found {len(short_results)} valid configurations")
        
        # Find best for each side
        best_long = None
        if long_results:
            long_results.sort(key=lambda x: x['score'], reverse=True)
            best_long = long_results[0]
            if best_long['win_rate'] < 40.0:
                best_long = None
        
        best_short = None
        if short_results:
            short_results.sort(key=lambda x: x['score'], reverse=True)
            best_short = short_results[0]
            if best_short['win_rate'] < 40.0:
                best_short = None
        
        result = {
            'symbol': symbol,
            'long': best_long,
            'short': best_short
        }
        
        # Print summary
        print(f"\n  Results:")
        if best_long:
            p = best_long['params']
            print(f"    LONG:  WR={best_long['win_rate']:.1f}% | Trades={best_long['trades']} | Score={best_long['score']:.1f}")
            print(f"      RSI: {p['rsi_min_long']}-{p['rsi_max_long']} | VWAP: {p['vwap_dist_max']} | Vol: {p['vol_ratio_min']}x")
            print(f"      MACD: {p['macd_hist_min']} | BBW: {p['bb_width_min']} | ATR: {p['atr_pct_min']}%")
            print(f"      Wick Δ: {p['wick_delta_min']} | Wick Min: {p['wick_ratio_min']} | Fib: {p['fib_zones']}")
        else:
            print(f"    LONG:  No valid configuration (WR < 40% or <10 trades)")
        
        if best_short:
            p = best_short['params']
            print(f"    SHORT: WR={best_short['win_rate']:.1f}% | Trades={best_short['trades']} | Score={best_short['score']:.1f}")
            print(f"      RSI: {p['rsi_min_short']}-{p['rsi_max_short']} | VWAP: {p['vwap_dist_max']} | Vol: {p['vol_ratio_min']}x")
            print(f"      MACD: {p['macd_hist_min']} | BBW: {p['bb_width_min']} | ATR: {p['atr_pct_min']}%")
            print(f"      Wick Δ: {p['wick_delta_min']} | Wick Min: {p['wick_ratio_min']} | Fib: {p['fib_zones']}")
        else:
            print(f"    SHORT: No valid configuration (WR < 40% or <10 trades)")
        
        return result
    
    def run_full_backtest(self, symbols: List[str] = None, min_wr: float = 40.0, resume: bool = True):
        """Run full backtest on all symbols - processes one at a time and saves incrementally"""
        if symbols is None:
            symbols = self.get_all_symbols()
        
        if not symbols:
            print("No symbols to test")
            return
        
        # Results file
        output_file = "backtest_pro_rules_results.json"
        progress_file = "backtest_pro_rules_progress.json"
        
        # Load existing results if resuming
        results = {}
        completed_symbols = set()
        if resume:
            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        results = json.load(f)
                    completed_symbols = set(results.keys())
                    print(f"Loaded {len(completed_symbols)} previously completed symbols")
            except Exception as e:
                print(f"Could not load previous results: {e}")
        
        print(f"\n{'='*80}")
        print(f"Starting Full Pro Rules Backtest")
        print(f"Total symbols: {len(symbols)}")
        print(f"Already completed: {len(completed_symbols)}")
        print(f"Remaining: {len(symbols) - len(completed_symbols)}")
        print(f"Timeframe: 3m")
        print(f"Candles: 5000")
        print(f"Min WR: {min_wr}%")
        print(f"Results file: {output_file}")
        print(f"{'='*80}\n")
        
        valid_count = sum(1 for r in results.values() if r.get('long') or r.get('short'))
        
        # Process symbols one by one
        for idx, symbol in enumerate(symbols, 1):
            # Skip if already completed
            if symbol in completed_symbols:
                print(f"\n[{idx}/{len(symbols)}] Skipping {symbol} (already completed)")
                continue
            
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(symbols)}] Processing {symbol}...")
            print(f"{'='*80}")
            
            try:
                result = self.optimize_symbol(symbol)
                
                # Save result immediately
                if result:
                    results[symbol] = result
                    if result.get('long') or result.get('short'):
                        valid_count += 1
                else:
                    # Still save even if no valid config (mark as failed)
                    results[symbol] = {'symbol': symbol, 'long': None, 'short': None, 'error': 'No valid configuration'}
                
                # Save to file after each symbol
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Save progress
                with open(progress_file, 'w') as f:
                    json.dump({
                        'completed': list(results.keys()),
                        'total': len(symbols),
                        'current': idx,
                        'valid_count': valid_count
                    }, f, indent=2)
                
                print(f"\n✓ Saved results for {symbol}")
                print(f"  Progress: {idx}/{len(symbols)} | Valid configs: {valid_count}")
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Interrupted by user")
                print(f"Progress saved. Resume by running the script again.")
                break
            except Exception as e:
                print(f"  ❌ Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                
                # Save error result
                results[symbol] = {'symbol': symbol, 'long': None, 'short': None, 'error': str(e)}
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Rate limiting between symbols
            time.sleep(1)
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total symbols tested: {len(results)}")
        print(f"Symbols with valid configs (WR >= {min_wr}%): {valid_count}")
        print(f"\nResults saved to: {output_file}")
        
        # Print per-symbol results
        print(f"\n{'='*80}")
        print(f"PER-SYMBOL RESULTS (WR >= {min_wr}%)")
        print(f"{'='*80}")
        
        for symbol in sorted(results.keys()):
            result = results[symbol]
            if not (result.get('long') or result.get('short')):
                continue
            
            print(f"\n{symbol}:")
            if result.get('long'):
                l = result['long']
                p = l['params']
                print(f"  LONG:  WR={l['win_rate']:.1f}% | Trades={l['trades']}")
                print(f"    RSI: {p['rsi_min_long']}-{p['rsi_max_long']} | VWAP: {p['vwap_dist_max']} | Vol: {p['vol_ratio_min']}x")
                print(f"    MACD: {p['macd_hist_min']} | BBW: {p['bb_width_min']} | ATR: {p['atr_pct_min']}%")
                print(f"    Wick Δ: {p['wick_delta_min']} | Wick Min: {p['wick_ratio_min']} | Fib: {p['fib_zones']}")
            
            if result.get('short'):
                s = result['short']
                p = s['params']
                print(f"  SHORT: WR={s['win_rate']:.1f}% | Trades={s['trades']}")
                print(f"    RSI: {p['rsi_min_short']}-{p['rsi_max_short']} | VWAP: {p['vwap_dist_max']} | Vol: {p['vol_ratio_min']}x")
                print(f"    MACD: {p['macd_hist_min']} | BBW: {p['bb_width_min']} | ATR: {p['atr_pct_min']}%")
                print(f"    Wick Δ: {p['wick_delta_min']} | Wick Min: {p['wick_ratio_min']} | Fib: {p['fib_zones']}")

if __name__ == "__main__":
    backtester = ProRulesBacktester()
    backtester.run_full_backtest()

