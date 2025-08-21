import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import structlog
from dataclasses import dataclass, field

from ..api.bybit_client import BybitClient
from ..strategy.supply_demand import SupplyDemandStrategy, Zone, ZoneType
from ..config import settings
from ..utils.rounding import (
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profits,
    round_to_tick,
    round_to_qty_step
)

logger = structlog.get_logger(__name__)

@dataclass
class BacktestTrade:
    """Represents a trade in backtesting"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: float
    pnl: float = 0
    pnl_percent: float = 0
    fees: float = 0
    zone_type: ZoneType = None
    zone_score: float = 0
    exit_reason: str = ""
    tp1_hit: bool = False
    is_open: bool = True

class BacktestEngine:
    """Backtesting engine for supply and demand strategy"""
    
    def __init__(self, bybit_client: BybitClient, strategy: SupplyDemandStrategy):
        self.client = bybit_client
        self.strategy = strategy
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.initial_capital = 10000  # Default starting capital
        
    async def run_backtest(
        self, 
        symbol: str, 
        timeframe: str,
        lookback_days: int = 30,
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """Run backtest for a symbol"""
        
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = [initial_capital]
        
        try:
            # Fetch historical data
            logger.info(f"Fetching {lookback_days} days of {timeframe} data for {symbol}")
            df = await self._fetch_historical_data(symbol, timeframe, lookback_days)
            
            if df.empty:
                raise ValueError("No historical data available")
            
            # Get instrument info
            instrument = self.client.get_instrument(symbol)
            if not instrument:
                raise ValueError(f"Unknown symbol: {symbol}")
            
            # Run the backtest
            logger.info("Running backtest simulation...")
            await self._simulate_trading(df, symbol, timeframe, instrument)
            
            # Calculate metrics
            logger.info("Calculating performance metrics...")
            metrics = self._calculate_metrics()
            
            # Generate equity curve chart
            chart_path = self._generate_charts(df, symbol, timeframe)
            metrics['chart_path'] = chart_path
            
            # Add time period info
            metrics['start_date'] = df.index[0].strftime('%Y-%m-%d')
            metrics['end_date'] = df.index[-1].strftime('%Y-%m-%d')
            metrics['symbol'] = symbol
            metrics['timeframe'] = timeframe
            
            logger.info(f"Backtest complete: {metrics['total_trades']} trades, "
                       f"{metrics['win_rate']:.1f}% win rate, "
                       f"{metrics['total_return']:.2f}% return")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            raise
    
    async def _fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str,
        lookback_days: int
    ) -> pd.DataFrame:
        """Fetch historical kline data"""
        
        # Convert timeframe to minutes
        timeframe_map = {
            '1': 1, '3': 3, '5': 5, '15': 15, '30': 30,
            '60': 60, '120': 120, '240': 240, 'D': 1440, 'W': 10080
        }
        
        interval_minutes = timeframe_map.get(timeframe, 15)
        
        # Calculate number of candles needed
        candles_per_day = 1440 / interval_minutes
        total_candles = int(candles_per_day * lookback_days)
        
        # Fetch in chunks (max 200 per request)
        all_data = []
        limit = min(200, total_candles)
        
        for i in range(0, total_candles, limit):
            chunk = await self.client.get_klines(symbol, timeframe, limit)
            if not chunk.empty:
                all_data.append(chunk)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        if all_data:
            df = pd.concat(all_data)
            df = df.drop_duplicates()
            df = df.sort_index()
            return df
        
        return pd.DataFrame()
    
    async def _simulate_trading(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        instrument: Dict
    ) -> None:
        """Simulate trading on historical data"""
        
        current_capital = self.initial_capital
        open_trade: Optional[BacktestTrade] = None
        zones: List[Zone] = []
        
        # Window for zone detection
        window_size = 50
        
        for i in range(window_size, len(df)):
            current_time = df.index[i]
            current_candle = df.iloc[i]
            
            # Get window of data for zone detection
            window_df = df.iloc[max(0, i-window_size):i+1]
            
            # Detect zones periodically (every 10 candles)
            if i % 10 == 0:
                new_zones = self.strategy.detect_zones(window_df, symbol, timeframe)
                
                # Add new zones, remove old ones
                zones = [z for z in zones if z.age_hours < settings.sd_zone_max_age_hours]
                zones.extend(new_zones)
                zones = zones[-50:]  # Keep max 50 zones
            
            # Update zones with current price
            self.strategy.zones[symbol] = zones
            self.strategy.update_zones(symbol, current_candle['close'])
            
            # Check open position
            if open_trade:
                # Check exit conditions
                exit_price, exit_reason = self._check_exit_conditions(
                    open_trade,
                    current_candle
                )
                
                if exit_price:
                    # Close the trade
                    open_trade.exit_time = current_time
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = exit_reason
                    open_trade.is_open = False
                    
                    # Calculate PnL
                    pnl = self._calculate_trade_pnl(open_trade, instrument)
                    open_trade.pnl = pnl
                    open_trade.pnl_percent = (pnl / (open_trade.entry_price * open_trade.position_size)) * 100
                    
                    # Update capital
                    current_capital += pnl
                    self.equity_curve.append(current_capital)
                    
                    # Reset open trade
                    open_trade = None
            
            # Check for new entry signal if no open position
            if not open_trade and current_capital > 0:
                signal = self.strategy.check_entry_signal(
                    symbol,
                    current_candle['close'],
                    current_capital,
                    settings.default_risk_percent,
                    instrument
                )
                
                if signal:
                    # Create new trade
                    open_trade = BacktestTrade(
                        entry_time=current_time,
                        exit_time=None,
                        symbol=symbol,
                        side=signal.side,
                        entry_price=signal.entry_price,
                        exit_price=None,
                        stop_loss=signal.stop_loss,
                        take_profit_1=signal.take_profit_1,
                        take_profit_2=signal.take_profit_2,
                        position_size=signal.position_size,
                        zone_type=signal.zone.zone_type,
                        zone_score=signal.zone.score
                    )
                    
                    # Add to trades list
                    self.trades.append(open_trade)
        
        # Close any remaining open trade at market close
        if open_trade:
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = df.iloc[-1]['close']
            open_trade.exit_reason = "End of backtest"
            open_trade.is_open = False
            
            pnl = self._calculate_trade_pnl(open_trade, instrument)
            open_trade.pnl = pnl
            open_trade.pnl_percent = (pnl / (open_trade.entry_price * open_trade.position_size)) * 100
            
            current_capital += pnl
            self.equity_curve.append(current_capital)
    
    def _check_exit_conditions(
        self,
        trade: BacktestTrade,
        candle: pd.Series
    ) -> Tuple[Optional[float], str]:
        """Check if trade should be exited"""
        
        # Check stop loss
        if trade.side == "Buy":
            if candle['low'] <= trade.stop_loss:
                return trade.stop_loss, "Stop Loss"
        else:
            if candle['high'] >= trade.stop_loss:
                return trade.stop_loss, "Stop Loss"
        
        # Check TP1 (partial exit)
        if not trade.tp1_hit:
            if trade.side == "Buy":
                if candle['high'] >= trade.take_profit_1:
                    trade.tp1_hit = True
                    # In real trading, we'd partial close here
                    # For backtest, we'll move stop to breakeven
                    if settings.move_stop_to_breakeven_at_tp1:
                        trade.stop_loss = trade.entry_price
            else:
                if candle['low'] <= trade.take_profit_1:
                    trade.tp1_hit = True
                    if settings.move_stop_to_breakeven_at_tp1:
                        trade.stop_loss = trade.entry_price
        
        # Check TP2 (full exit)
        if trade.side == "Buy":
            if candle['high'] >= trade.take_profit_2:
                return trade.take_profit_2, "Take Profit 2"
        else:
            if candle['low'] <= trade.take_profit_2:
                return trade.take_profit_2, "Take Profit 2"
        
        return None, ""
    
    def _calculate_trade_pnl(self, trade: BacktestTrade, instrument: Dict) -> float:
        """Calculate P&L for a trade including fees"""
        
        # Calculate gross PnL
        if trade.side == "Buy":
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.position_size
        else:
            gross_pnl = (trade.entry_price - trade.exit_price) * trade.position_size
        
        # Calculate fees (entry + exit)
        entry_fee = trade.entry_price * trade.position_size * 0.0006  # 0.06% taker
        exit_fee = trade.exit_price * trade.position_size * 0.0006
        total_fees = entry_fee + exit_fee
        
        trade.fees = total_fees
        
        # Net PnL
        net_pnl = gross_pnl - total_fees
        
        return net_pnl
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (total_pnl / self.initial_capital) * 100
        
        # Average metrics
        avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0
        
        # Best/worst trades
        best_trade = max(t.pnl_percent for t in self.trades) if self.trades else 0
        worst_trade = min(t.pnl_percent for t in self.trades) if self.trades else 0
        
        # Drawdown calculation
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t.pnl_percent for t in self.trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Zone performance analysis
        zone_stats = self._analyze_zone_performance()
        
        # Probability scores by zone score buckets
        probability_scores = self._calculate_probability_scores()
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'zone_stats': zone_stats,
            'probability_scores': probability_scores
        }
    
    def _analyze_zone_performance(self) -> Dict[str, Any]:
        """Analyze performance by zone characteristics"""
        
        demand_trades = [t for t in self.trades if t.zone_type == ZoneType.DEMAND]
        supply_trades = [t for t in self.trades if t.zone_type == ZoneType.SUPPLY]
        
        demand_wins = [t for t in demand_trades if t.pnl > 0]
        supply_wins = [t for t in supply_trades if t.pnl > 0]
        
        demand_win_rate = (len(demand_wins) / len(demand_trades) * 100) if demand_trades else 0
        supply_win_rate = (len(supply_wins) / len(supply_trades) * 100) if supply_trades else 0
        
        # Fresh zone analysis (assuming first touch)
        fresh_zone_trades = [t for t in self.trades if t.zone_score >= 70]
        fresh_zone_wins = [t for t in fresh_zone_trades if t.pnl > 0]
        fresh_zone_win_rate = (len(fresh_zone_wins) / len(fresh_zone_trades) * 100) if fresh_zone_trades else 0
        
        return {
            'demand_win_rate': demand_win_rate,
            'supply_win_rate': supply_win_rate,
            'fresh_zone_win_rate': fresh_zone_win_rate,
            'demand_trades': len(demand_trades),
            'supply_trades': len(supply_trades)
        }
    
    def _calculate_probability_scores(self) -> Dict[str, Dict[str, Any]]:
        """Calculate win rates by zone score buckets"""
        
        buckets = {
            '60_70': {'min': 60, 'max': 70},
            '70_80': {'min': 70, 'max': 80},
            '80_90': {'min': 80, 'max': 90},
            '90_100': {'min': 90, 'max': 100}
        }
        
        results = {}
        
        for bucket_name, bucket_range in buckets.items():
            bucket_trades = [
                t for t in self.trades 
                if bucket_range['min'] <= t.zone_score < bucket_range['max']
            ]
            
            if bucket_trades:
                wins = [t for t in bucket_trades if t.pnl > 0]
                win_rate = (len(wins) / len(bucket_trades)) * 100
                avg_pnl = np.mean([t.pnl_percent for t in bucket_trades])
            else:
                win_rate = 0
                avg_pnl = 0
            
            results[bucket_name] = {
                'win_rate': win_rate,
                'count': len(bucket_trades),
                'avg_pnl': avg_pnl
            }
        
        return results
    
    def _generate_charts(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Generate equity curve and trade visualization charts"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} Price & Trades', 'Equity Curve'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add trade markers
        for trade in self.trades:
            # Entry marker
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.side == 'Buy' else 'triangle-down',
                        size=10,
                        color='green' if trade.side == 'Buy' else 'red'
                    ),
                    name=f'{trade.side} Entry',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Exit marker
            if trade.exit_time:
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color='blue' if trade.pnl > 0 else 'orange'
                        ),
                        name='Exit',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Equity curve
        equity_times = []
        for i, trade in enumerate(self.trades):
            if trade.exit_time:
                equity_times.append(trade.exit_time)
        
        if len(equity_times) == len(self.equity_curve[1:]):
            fig.add_trace(
                go.Scatter(
                    x=equity_times,
                    y=self.equity_curve[1:],
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
        
        fig.update_layout(
            title=f"Backtest Results - {symbol} {timeframe}",
            height=800,
            showlegend=False
        )
        
        # Save chart
        chart_path = f"/tmp/backtest_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.write_image(chart_path)
        
        return chart_path