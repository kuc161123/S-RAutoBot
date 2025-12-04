import asyncio
import logging
import yaml
import os
import pandas as pd
import pandas_ta as ta
import aiohttp
from datetime import datetime
from autobot.brokers.bybit import Bybit, BybitConfig

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vwap_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VWAPBot")

class VWAPBot:
    def __init__(self):
        self.load_config()
        self.setup_broker()
        self.vwap_combos = {}
        self.active_positions = {} # Cache to avoid API spam
        
    def load_config(self):
        # Load .env manually
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        os.environ[k] = v
        except FileNotFoundError:
            pass

        with open('config.yaml', 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        # Env Var Replacement
        for k, v in self.cfg['bybit'].items():
            if isinstance(v, str) and v.startswith("${"):
                var = v[2:-1]
                val = os.getenv(var)
                if val: self.cfg['bybit'][k] = val
                
        for k, v in self.cfg['telegram'].items():
            if isinstance(v, str) and v.startswith("${"):
                var = v[2:-1]
                val = os.getenv(var)
                if val: self.cfg['telegram'][k] = val

    def setup_broker(self):
        self.broker = Bybit(BybitConfig(
            base_url=self.cfg['bybit']['base_url'],
            api_key=self.cfg['bybit']['api_key'],
            api_secret=self.cfg['bybit']['api_secret']
        ))
        
    def load_overrides(self):
        """Reload overrides to pick up new backtest findings"""
        try:
            with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
                self.vwap_combos = yaml.safe_load(f) or {}
            # logger.info(f"Loaded {len(self.vwap_combos)} VWAP Combo overrides")
        except FileNotFoundError:
            self.vwap_combos = {}
            
    async def send_telegram(self, msg):
        token = self.cfg['telegram']['token']
        chat_id = self.cfg['telegram']['chat_id']
        
        # Debug Log
        if token:
            logger.info(f"TG Token loaded: {token[:5]}...{token[-5:]}")
        else:
            logger.error("TG Token is EMPTY")
            
        if chat_id:
            logger.info(f"TG Chat ID loaded: {chat_id}")
        else:
            logger.error("TG Chat ID is EMPTY")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram failed: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def calculate_indicators(self, df):
        if len(df) < 50: return df
        
        # ATR
        df['atr'] = df.ta.atr(length=14)
        
        # RSI
        df['rsi'] = df.ta.rsi(length=14)
        
        # MACD
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        
        # VWAP
        try:
            vwap = df.ta.vwap(high='high', low='low', close='close', volume='volume')
            if isinstance(vwap, pd.DataFrame):
                df['vwap'] = vwap.iloc[:, 0]
            else:
                df['vwap'] = vwap
        except Exception:
            # Fallback VWAP
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).rolling(1440//3).sum() / df['volume'].rolling(1440//3).sum()

        # Fib (Rolling 50 High/Low)
        df['roll_high'] = df['high'].rolling(50).max()
        df['roll_low'] = df['low'].rolling(50).min()
        
        return df.dropna()

    def get_combo(self, row):
        # RSI Bin
        rsi = row.rsi
        if rsi < 30: r_bin = '<30'
        elif rsi < 40: r_bin = '30-40'
        elif rsi < 60: r_bin = '40-60'
        elif rsi < 70: r_bin = '60-70'
        else: r_bin = '70+'
        
        # MACD Bin
        m_bin = 'bull' if row.macd > row.macd_signal else 'bear'
        
        # Fib Bin
        high = row.roll_high
        low = row.roll_low
        close = row.close
        if high == low: f_bin = '0-23'
        else:
            fib = (high - close) / (high - low) * 100
            if fib < 23.6: f_bin = '0-23'
            elif fib < 38.2: f_bin = '23-38'
            elif fib < 50.0: f_bin = '38-50'
            elif fib < 61.8: f_bin = '50-61'
            elif fib < 78.6: f_bin = '61-78'
            elif fib < 100: f_bin = '78-100'
            else: f_bin = '100+'
            
        return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

    async def process_symbol(self, sym):
        try:
            # 1. Check Overrides First (Optimization)
            if sym not in self.vwap_combos:
                return
                
            # 2. Fetch Data
            klines = self.broker.get_klines(sym, '3', limit=200)
            if not klines: return
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            
            df = self.calculate_indicators(df)
            if df.empty: return
            
            row = df.iloc[-1] # Current candle (or last closed? Strategy uses CLOSE logic, so we check last CLOSED candle usually, but for live we check current if it just closed? 
            # Backtest uses: if row.low <= vwap and row.close > vwap. This implies the candle HAS closed.
            # So we should look at iloc[-2] if iloc[-1] is the currently forming candle?
            # Bybit get_klines returns the latest candle which is usually incomplete.
            # So we must use iloc[-2] (the last completed candle).
            
            last_candle = df.iloc[-2]
            
            # 3. Check Signal
            signal = None
            side = None
            
            # Long: Low touched VWAP, Close > VWAP
            if last_candle.low <= last_candle.vwap and last_candle.close > last_candle.vwap:
                side = 'long'
            # Short: High touched VWAP, Close < VWAP
            elif last_candle.high >= last_candle.vwap and last_candle.close < last_candle.vwap:
                side = 'short'
                
            if side:
                combo = self.get_combo(last_candle)
                
                # Check if allowed
                allowed = self.vwap_combos.get(sym, {}).get(side, [])
                if combo in allowed:
                    # VALID SIGNAL
                    logger.info(f"🚀 SIGNAL: {sym} {side} {combo}")
                    await self.execute_trade(sym, side, last_candle, combo)
                else:
                    # Phantom
                    logger.info(f"👻 Phantom: {sym} {side} {combo} (Not in allowed list)")
                    
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")

    async def execute_trade(self, sym, side, row, combo):
        # Check if already in position
        pos = self.broker.get_position(sym)
        if pos and float(pos['size']) > 0:
            logger.info(f"Skipping {sym}: Already in position")
            return

        # Calculate Risk
        balance = self.broker.get_balance() or 0
        risk_pct = self.cfg.get('risk', {}).get('risk_percent', 0.5) / 100
        risk_amt = balance * risk_pct
        
        # R:R 1:2
        atr = row.atr
        entry = row.close
        
        if side == 'long':
            sl = entry - (2.0 * atr)
            tp = entry + (4.0 * atr)
            dist = entry - sl
        else:
            sl = entry + (2.0 * atr)
            tp = entry - (4.0 * atr)
            dist = sl - entry
            
        if dist == 0: return
        
        qty = risk_amt / dist
        # Round qty (simple logic, ideally use instrument info)
        # For now, just round to 0 decimals for simplicity or 3 sig figs?
        # Bybit needs specific precision. 
        # Let's try to be safe: use a simple rounding or fetch precision.
        # To be safe/fast: just try to execute. If fail, log it.
        # Better: use a helper to round.
        
        # Simple rounding based on price
        if entry > 1000: qty = round(qty, 3)
        elif entry > 1: qty = round(qty, 1)
        else: qty = round(qty, 0)
        
        if qty <= 0: return

        logger.info(f"Executing {side} on {sym}: Size {qty}, SL {sl}, TP {tp}")
        
        # Execute
        # 1. Market Order
        res = self.broker.place_market(sym, side, qty)
        if res and res.get('retCode') == 0:
            # 2. Set TP/SL
            self.broker.set_tpsl(sym, tp, sl, qty)
            
            # 3. Notify
            msg = (
                f"🚀 **VWAP SNIPER ENTRY**\n"
                f"Symbol: `{sym}`\n"
                f"Side: **{side.upper()}**\n"
                f"Combo: `{combo}`\n"
                f"Price: {entry}\n"
                f"Size: {qty}\n"
                f"Risk: ${risk_amt:.2f} ({risk_pct*100}%)\n"
                f"TP: {tp:.4f} (4ATR)\n"
                f"SL: {sl:.4f} (2ATR)"
            )
            await self.send_telegram(msg)
        else:
            logger.error(f"Order failed: {res}")

    async def run(self):
        logger.info("🤖 VWAP Bot Started")
        await self.send_telegram("🤖 **VWAP Bot Started**\nMode: Aggressive\nStrategy: VWAP Bounce + Golden Combos")
        
        # Load symbols
        with open('symbols_400.yaml', 'r') as f:
            symbols = yaml.safe_load(f)['symbols']
            
        while True:
            self.load_overrides()
            
            # Process in chunks to avoid blocking too long?
            # Or just sequential await
            for sym in symbols:
                await self.process_symbol(sym)
                await asyncio.sleep(0.1) # Rate limit protection
                
            logger.info("Loop complete. Sleeping...")
            await asyncio.sleep(10) # 10s sleep between full loops

