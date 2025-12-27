import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

class QuantEdgeBacktester:
    def __init__(self, df, fee=0.0005, slippage=0.0003, funding=0.0001):
        self.df = df.copy()
        self.fee = fee
        self.slippage = slippage
        self.funding = funding # Approximate periodic impact
        
    def calculate_metrics(self, trades_df):
        if trades_df.empty:
            return {"sharpe": 0, "max_dd": 0, "win_rate": 0, "total_pnl": 0}
        
        pnl = trades_df['pnl']
        total_pnl = pnl.sum()
        win_rate = (pnl > 0).mean()
        
        # Simple daily-equivalent Sharpe (assuming ~10 trades/day on average)
        sharpe = (pnl.mean() / pnl.std() * np.sqrt(365 * 10)) if pnl.std() > 0 else 0
        
        # Max Drawdown
        equity = (1 + pnl).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        
        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trades": len(trades_df)
        }

    def run_backtest(self, signals, rr=2.0, sl_atr_mult=1.5):
        """
        Signals: Series or Array of (-1, 0, 1)
        Execution: Signal at close[t], Fill at open[t+1]
        """
        df = self.df.copy()
        df['signal'] = signals
        
        # Shift signals to represent entry at open[t+1]
        df['actual_entry_sig'] = df['signal'].shift(1).fillna(0)
        
        trades = []
        in_position = False
        pos_side = 0
        entry_price = 0
        entry_time = None
        sl = 0
        tp = 0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # --- EXIT LOGIC ---
            if in_position:
                exit_price = 0
                exit_reason = None
                
                # Check for SL/TP within the bar (Aggressive assumption: High hits TP, Low hits SL)
                if pos_side == 1:
                    if row['low'] <= sl:
                        exit_price = sl
                        exit_reason = "SL"
                    elif row['high'] >= tp:
                        exit_price = tp
                        exit_reason = "TP"
                elif pos_side == -1:
                    if row['high'] >= sl:
                        exit_price = sl
                        exit_reason = "SL"
                    elif row['low'] <= tp:
                        exit_price = tp
                        exit_reason = "TP"
                
                if exit_reason:
                    # Apply slippage/fee to exit
                    realized_exit = exit_price * (1 - self.slippage if pos_side == 1 else 1 + self.slippage)
                    pnl_pct = (realized_exit - entry_price) / entry_price if pos_side == 1 else (entry_price - realized_exit) / entry_price
                    # Net of fees (Exit fee)
                    pnl_pct -= self.fee
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': row['timestamp'],
                        'side': pos_side,
                        'entry_price': entry_price,
                        'exit_price': realized_exit,
                        'pnl': pnl_pct,
                        'reason': exit_reason
                    })
                    in_position = False
                    pos_side = 0
            
            # --- ENTRY LOGIC ---
            if not in_position and row['actual_entry_sig'] != 0:
                pos_side = row['actual_entry_sig']
                # Realistic entry at Open + Slippage + Fee
                entry_price = row['open'] * (1 + self.slippage if pos_side == 1 else 1 - self.slippage)
                # Deduct entry fee
                entry_cost_adj = 1 + self.fee # Entry cost in percentage
                
                atr = row['atr'] if 'atr' in row else row['open'] * 0.01 # Fallback
                sl_dist = atr * sl_atr_mult
                
                if pos_side == 1:
                    sl = entry_price - sl_dist
                    tp = entry_price + (sl_dist * rr)
                else:
                    sl = entry_price + sl_dist
                    tp = entry_price - (sl_dist * rr)
                
                entry_time = row['timestamp']
                in_position = True
                
        return pd.DataFrame(trades)

    def monte_carlo(self, trades_pnl, n_sims=1000, n_trades=100):
        results = []
        for _ in range(n_sims):
            sample = np.random.choice(trades_pnl, size=n_trades, replace=True)
            equity = np.cumprod(1 + sample)
            results.append(equity[-1])
        return np.array(results)

    def walk_forward(self, strategy_func, param_grid, n_folds=6):
        """
        Simplified WFO logic
        """
        total_len = len(self.df)
        fold_size = total_len // (n_folds + 1)
        
        oos_results = []
        
        for f in range(n_folds):
            train_start = f * fold_size
            train_end = train_start + fold_size * 2
            test_end = train_end + fold_size
            
            if test_end > total_len: break
            
            train_df = self.df.iloc[train_start:train_end]
            test_df = self.df.iloc[train_end:test_end]
            
            # Optimization (Simplistic)
            best_params = None
            max_pnl = -np.inf
            
            # Divide params in grid: Strategy vs Backtest
            strat_keys = ['rsi_thresh', 'bb_std', 'lookback', 'trend_ema', 'sq_mult', 'kc_mult']
            bt_keys = ['rr', 'sl_atr_mult']
            
            for params in param_grid:
                s_params = {k: v for k, v in params.items() if k in strat_keys}
                b_params = {k: v for k, v in params.items() if k in bt_keys}
                
                train_signals = strategy_func(train_df, **s_params)
                res = self.run_backtest(train_signals, **b_params)
                if not res.empty:
                    total_pnl = res['pnl'].sum()
                    if total_pnl > max_pnl:
                        max_pnl = total_pnl
                        best_params = params
            
            # OOS Test
            if best_params:
                s_params = {k: v for k, v in best_params.items() if k in strat_keys}
                b_params = {k: v for k, v in best_params.items() if k in bt_keys}
                
                test_signals = strategy_func(test_df, **s_params)
                oos_res = self.run_backtest(test_signals, **b_params)
                oos_results.append({
                    'fold': f,
                    'params': best_params,
                    'trades': oos_res
                })
                
        return oos_results
