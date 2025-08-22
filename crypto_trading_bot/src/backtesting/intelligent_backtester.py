"""
Intelligent Backtesting System
Complete validation of trading strategies before live deployment
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog
from dataclasses import dataclass, field
import json

logger = structlog.get_logger(__name__)


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    direction: str  # 'long' or 'short'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    trade_duration: Optional[timedelta] = None
    entry_reason: str = ""
    exit_reason: str = ""
    ml_confidence: float = 0.0
    risk_score: float = 0.0


@dataclass
class BacktestResults:
    """Complete backtest results"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: timedelta = timedelta()
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_trade_duration: timedelta = timedelta()
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    # ML specific metrics
    ml_accuracy: float = 0.0
    ml_precision: float = 0.0
    ml_recall: float = 0.0
    ml_f1_score: float = 0.0
    
    # Strategy specific
    best_performing_symbol: str = ""
    worst_performing_symbol: str = ""
    best_time_of_day: str = ""
    worst_time_of_day: str = ""
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0


class IntelligentBacktester:
    """
    Complete backtesting system with ML validation
    """
    
    def __init__(self, bybit_client, intelligent_engine):
        self.client = bybit_client
        self.engine = intelligent_engine
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.backtest_trades: List[BacktestTrade] = []
        self.open_positions: Dict[str, BacktestTrade] = {}
        
    async def run_complete_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02
    ) -> BacktestResults:
        """Run complete backtest with all strategies"""
        try:
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Load historical data
            await self._load_historical_data(symbols, start_date, end_date)
            
            # Initialize backtest state
            current_capital = initial_capital
            equity_curve = [initial_capital]
            daily_returns = []
            
            # Run backtest day by day
            current_date = start_date
            while current_date <= end_date:
                
                # Process each symbol
                for symbol in symbols:
                    if symbol not in self.historical_data:
                        continue
                    
                    # Get data for current day
                    data = self._get_data_for_date(symbol, current_date)
                    if data is None:
                        continue
                    
                    # Check for signals
                    signal = await self._generate_signal(
                        symbol, data, current_capital, risk_per_trade
                    )
                    
                    if signal:
                        # Execute trade in backtest
                        trade = await self._execute_backtest_trade(
                            symbol, signal, data, current_capital * risk_per_trade
                        )
                        if trade:
                            self.backtest_trades.append(trade)
                            current_capital += trade.pnl
                    
                    # Update open positions
                    await self._update_open_positions(symbol, data)
                
                # Update equity curve
                equity_curve.append(current_capital)
                
                # Calculate daily return
                if len(equity_curve) >= 2:
                    daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                    daily_returns.append(daily_return)
                
                # Move to next day
                current_date += timedelta(days=1)
            
            # Close all remaining positions
            await self._close_all_positions()
            
            # Calculate results
            results = self._calculate_results(
                self.backtest_trades,
                equity_curve,
                daily_returns,
                initial_capital
            )
            
            # Validate ML performance
            await self._validate_ml_performance(results)
            
            logger.info(f"Backtest complete: {results.win_rate:.2%} win rate, "
                       f"Sharpe: {results.sharpe_ratio:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return BacktestResults()
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """Load historical data for all symbols"""
        for symbol in symbols:
            try:
                # Get historical klines
                klines = await self.client.get_historical_klines(
                    symbol=symbol,
                    interval='15',  # 15 minute candles
                    start=start_date,
                    end=end_date
                )
                
                if klines:
                    df = pd.DataFrame(klines)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Calculate indicators
                    df = await self._calculate_indicators(df)
                    
                    self.historical_data[symbol] = df
                    logger.info(f"Loaded {len(df)} candles for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
    
    async def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _get_data_for_date(
        self,
        symbol: str,
        date: datetime
    ) -> Optional[pd.DataFrame]:
        """Get historical data for specific date"""
        if symbol not in self.historical_data:
            return None
        
        df = self.historical_data[symbol]
        
        # Get data up to current date
        mask = df.index.date <= date.date()
        return df[mask].tail(100)  # Last 100 candles
    
    async def _generate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        capital: float,
        risk_per_trade: float
    ) -> Optional[Dict]:
        """Generate trading signal using ML"""
        try:
            if len(data) < 50:
                return None
            
            latest = data.iloc[-1]
            
            # Get ML prediction
            ml_prediction = await self._get_ml_prediction(symbol, data)
            
            # Check technical conditions
            conditions = {
                'trend': latest['sma_20'] > latest['sma_50'],
                'momentum': latest['rsi'] > 30 and latest['rsi'] < 70,
                'macd': latest['macd_hist'] > 0,
                'volume': latest['volume_ratio'] > 1.2,
                'ml_bullish': ml_prediction.get('direction') == 'long',
                'ml_confidence': ml_prediction.get('confidence', 0) > 0.65
            }
            
            # Count positive conditions
            score = sum(conditions.values())
            
            if score >= 4:
                # Calculate position size
                position_size = (capital * risk_per_trade) / latest['atr']
                
                return {
                    'symbol': symbol,
                    'direction': 'long' if conditions['trend'] else 'short',
                    'price': latest['close'],
                    'size': position_size,
                    'stop_loss': latest['close'] - (2 * latest['atr']),
                    'take_profit': latest['close'] + (3 * latest['atr']),
                    'confidence': ml_prediction.get('confidence', 0.5),
                    'reason': f"Score: {score}/6"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    async def _get_ml_prediction(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Dict:
        """Get ML prediction for backtesting"""
        try:
            # Prepare features
            features = {
                'price_change': (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20],
                'volume_ratio': data['volume_ratio'].iloc[-1],
                'rsi': data['rsi'].iloc[-1],
                'macd_hist': data['macd_hist'].iloc[-1],
                'bb_position': (data['close'].iloc[-1] - data['bb_lower'].iloc[-1]) / 
                              (data['bb_upper'].iloc[-1] - data['bb_lower'].iloc[-1]),
                'trend_strength': abs(data['sma_20'].iloc[-1] - data['sma_50'].iloc[-1]) / data['close'].iloc[-1]
            }
            
            # Simulate ML prediction (in real system, use actual model)
            score = (
                features['price_change'] * 0.3 +
                features['volume_ratio'] * 0.2 +
                (features['rsi'] - 50) / 100 * 0.2 +
                features['macd_hist'] * 0.15 +
                features['bb_position'] * 0.15
            )
            
            return {
                'direction': 'long' if score > 0 else 'short',
                'confidence': min(abs(score), 1.0),
                'predicted_move': score * 100  # Percentage
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {'direction': 'neutral', 'confidence': 0}
    
    async def _execute_backtest_trade(
        self,
        symbol: str,
        signal: Dict,
        data: pd.DataFrame,
        position_value: float
    ) -> Optional[BacktestTrade]:
        """Execute trade in backtest"""
        try:
            # Check if already in position
            if symbol in self.open_positions:
                return None
            
            # Create trade
            trade = BacktestTrade(
                symbol=symbol,
                entry_time=data.index[-1],
                exit_time=None,
                entry_price=signal['price'],
                exit_price=None,
                size=signal['size'],
                direction=signal['direction'],
                entry_reason=signal['reason'],
                ml_confidence=signal['confidence']
            )
            
            # Store in open positions
            self.open_positions[symbol] = trade
            
            logger.debug(f"Opened {trade.direction} position in {symbol} at {trade.entry_price}")
            
            return None  # Will return complete trade when closed
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    async def _update_open_positions(self, symbol: str, data: pd.DataFrame):
        """Update open positions with current data"""
        if symbol not in self.open_positions:
            return
        
        trade = self.open_positions[symbol]
        current_price = data['close'].iloc[-1]
        
        # Calculate current PnL
        if trade.direction == 'long':
            pnl_percent = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_percent = (trade.entry_price - current_price) / trade.entry_price
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Stop loss (2 ATR)
        atr = data['atr'].iloc[-1]
        if trade.direction == 'long':
            stop_loss = trade.entry_price - (2 * atr)
            if current_price <= stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
        else:
            stop_loss = trade.entry_price + (2 * atr)
            if current_price >= stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
        
        # Take profit (3 ATR)
        if trade.direction == 'long':
            take_profit = trade.entry_price + (3 * atr)
            if current_price >= take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        else:
            take_profit = trade.entry_price - (3 * atr)
            if current_price <= take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        
        # Time-based exit (hold max 5 days)
        holding_time = data.index[-1] - trade.entry_time
        if holding_time > timedelta(days=5):
            should_exit = True
            exit_reason = "Max holding time"
        
        # Exit if needed
        if should_exit:
            trade.exit_time = data.index[-1]
            trade.exit_price = current_price
            trade.exit_reason = exit_reason
            trade.trade_duration = trade.exit_time - trade.entry_time
            
            # Calculate final PnL
            if trade.direction == 'long':
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
                trade.pnl_percent = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.size
                trade.pnl_percent = (trade.entry_price - trade.exit_price) / trade.entry_price
            
            # Remove from open positions and add to completed trades
            del self.open_positions[symbol]
            self.backtest_trades.append(trade)
            
            logger.debug(f"Closed {trade.direction} in {symbol}: "
                        f"PnL: {trade.pnl:.2f} ({trade.pnl_percent:.2%}), "
                        f"Reason: {exit_reason}")
    
    async def _close_all_positions(self):
        """Close all remaining open positions at end of backtest"""
        for symbol, trade in list(self.open_positions.items()):
            # Get last known price
            if symbol in self.historical_data:
                last_price = self.historical_data[symbol]['close'].iloc[-1]
                trade.exit_price = last_price
                trade.exit_time = self.historical_data[symbol].index[-1]
                trade.exit_reason = "Backtest End"
                trade.trade_duration = trade.exit_time - trade.entry_time
                
                # Calculate PnL
                if trade.direction == 'long':
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
                    trade.pnl_percent = (trade.exit_price - trade.entry_price) / trade.entry_price
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.size
                    trade.pnl_percent = (trade.entry_price - trade.exit_price) / trade.entry_price
                
                self.backtest_trades.append(trade)
        
        self.open_positions.clear()
    
    def _calculate_results(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        daily_returns: List[float],
        initial_capital: float
    ) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        results = BacktestResults()
        
        if not trades:
            return results
        
        # Basic metrics
        results.total_trades = len(trades)
        results.winning_trades = sum(1 for t in trades if t.pnl > 0)
        results.losing_trades = sum(1 for t in trades if t.pnl < 0)
        results.win_rate = results.winning_trades / results.total_trades if results.total_trades > 0 else 0
        
        # PnL metrics
        results.total_pnl = sum(t.pnl for t in trades)
        results.total_pnl_percent = results.total_pnl / initial_capital
        
        # Average metrics
        winning_pnls = [t.pnl for t in trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in trades if t.pnl < 0]
        
        results.average_win = np.mean(winning_pnls) if winning_pnls else 0
        results.average_loss = np.mean(losing_pnls) if losing_pnls else 0
        results.largest_win = max(winning_pnls) if winning_pnls else 0
        results.largest_loss = min(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        if daily_returns:
            returns_array = np.array(daily_returns)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if returns_array.std() > 0:
                results.sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    results.sortino_ratio = (returns_array.mean() / downside_std) * np.sqrt(252)
            
            # Value at Risk
            results.var_95 = np.percentile(returns_array, 5)
            results.cvar_95 = returns_array[returns_array <= results.var_95].mean()
        
        # Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        results.max_drawdown = drawdown.min()
        
        # Calmar ratio
        if results.max_drawdown < 0:
            annual_return = results.total_pnl_percent * (252 / len(daily_returns)) if daily_returns else 0
            results.calmar_ratio = annual_return / abs(results.max_drawdown)
        
        # Trade duration
        durations = [t.trade_duration for t in trades if t.trade_duration]
        if durations:
            results.average_trade_duration = sum(durations, timedelta()) / len(durations)
        
        # Symbol performance
        symbol_pnls = {}
        for trade in trades:
            if trade.symbol not in symbol_pnls:
                symbol_pnls[trade.symbol] = 0
            symbol_pnls[trade.symbol] += trade.pnl
        
        if symbol_pnls:
            results.best_performing_symbol = max(symbol_pnls, key=symbol_pnls.get)
            results.worst_performing_symbol = min(symbol_pnls, key=symbol_pnls.get)
        
        # Store trades and curves
        results.trades = trades
        results.equity_curve = equity_curve
        results.daily_returns = daily_returns
        
        return results
    
    async def _validate_ml_performance(self, results: BacktestResults):
        """Validate ML model performance"""
        if not results.trades:
            return
        
        # Calculate ML metrics
        true_positives = sum(1 for t in results.trades 
                            if t.ml_confidence > 0.6 and t.pnl > 0)
        false_positives = sum(1 for t in results.trades 
                             if t.ml_confidence > 0.6 and t.pnl <= 0)
        true_negatives = sum(1 for t in results.trades 
                            if t.ml_confidence <= 0.6 and t.pnl <= 0)
        false_negatives = sum(1 for t in results.trades 
                             if t.ml_confidence <= 0.6 and t.pnl > 0)
        
        # Accuracy
        total = true_positives + false_positives + true_negatives + false_negatives
        if total > 0:
            results.ml_accuracy = (true_positives + true_negatives) / total
        
        # Precision
        if true_positives + false_positives > 0:
            results.ml_precision = true_positives / (true_positives + false_positives)
        
        # Recall
        if true_positives + false_negatives > 0:
            results.ml_recall = true_positives / (true_positives + false_negatives)
        
        # F1 Score
        if results.ml_precision + results.ml_recall > 0:
            results.ml_f1_score = (2 * results.ml_precision * results.ml_recall) / (results.ml_precision + results.ml_recall)
    
    async def validate_strategy(
        self,
        strategy_name: str,
        min_win_rate: float = 0.45,
        min_sharpe: float = 1.0,
        min_profit_factor: float = 1.2
    ) -> bool:
        """Validate if strategy meets minimum requirements"""
        try:
            # Run backtest
            results = await self.run_complete_backtest(
                symbols=self.engine.symbol_manager.get_active_symbols()[:10],
                start_date=datetime.now() - timedelta(days=180),
                end_date=datetime.now(),
                initial_capital=10000
            )
            
            # Check requirements
            passed = (
                results.win_rate >= min_win_rate and
                results.sharpe_ratio >= min_sharpe and
                results.profit_factor >= min_profit_factor and
                results.max_drawdown > -0.20  # Max 20% drawdown
            )
            
            if passed:
                logger.info(f"Strategy {strategy_name} PASSED validation: "
                           f"Win Rate: {results.win_rate:.2%}, "
                           f"Sharpe: {results.sharpe_ratio:.2f}, "
                           f"Profit Factor: {results.profit_factor:.2f}")
            else:
                logger.warning(f"Strategy {strategy_name} FAILED validation: "
                              f"Win Rate: {results.win_rate:.2%}, "
                              f"Sharpe: {results.sharpe_ratio:.2f}, "
                              f"Profit Factor: {results.profit_factor:.2f}")
            
            # Save results
            await self._save_backtest_results(strategy_name, results)
            
            return passed
            
        except Exception as e:
            logger.error(f"Strategy validation error: {e}")
            return False
    
    async def _save_backtest_results(self, strategy_name: str, results: BacktestResults):
        """Save backtest results to file"""
        try:
            results_dict = {
                'strategy': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'total_trades': results.total_trades,
                    'win_rate': results.win_rate,
                    'sharpe_ratio': results.sharpe_ratio,
                    'sortino_ratio': results.sortino_ratio,
                    'profit_factor': results.profit_factor,
                    'max_drawdown': results.max_drawdown,
                    'total_pnl': results.total_pnl,
                    'total_pnl_percent': results.total_pnl_percent,
                    'ml_accuracy': results.ml_accuracy,
                    'ml_f1_score': results.ml_f1_score
                },
                'trades': [
                    {
                        'symbol': t.symbol,
                        'direction': t.direction,
                        'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                        'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                        'pnl': t.pnl,
                        'pnl_percent': t.pnl_percent
                    }
                    for t in results.trades[:100]  # Save first 100 trades
                ]
            }
            
            # Save to file
            filename = f"backtest_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(f"backtests/{filename}", 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    async def run_walk_forward_analysis(
        self,
        symbols: List[str],
        total_period_days: int = 365,
        in_sample_days: int = 180,
        out_sample_days: int = 30
    ) -> List[BacktestResults]:
        """Run walk-forward analysis for robust validation"""
        results = []
        
        start_date = datetime.now() - timedelta(days=total_period_days)
        current_date = start_date
        
        while current_date + timedelta(days=in_sample_days + out_sample_days) <= datetime.now():
            # In-sample period
            in_sample_start = current_date
            in_sample_end = current_date + timedelta(days=in_sample_days)
            
            # Train on in-sample
            logger.info(f"Training on {in_sample_start.date()} to {in_sample_end.date()}")
            
            # Out-of-sample period
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + timedelta(days=out_sample_days)
            
            # Test on out-of-sample
            logger.info(f"Testing on {out_sample_start.date()} to {out_sample_end.date()}")
            
            result = await self.run_complete_backtest(
                symbols=symbols,
                start_date=out_sample_start,
                end_date=out_sample_end,
                initial_capital=10000
            )
            
            results.append(result)
            
            # Move forward
            current_date += timedelta(days=out_sample_days)
        
        # Calculate average performance
        if results:
            avg_win_rate = np.mean([r.win_rate for r in results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            
            logger.info(f"Walk-forward complete: "
                       f"Avg Win Rate: {avg_win_rate:.2%}, "
                       f"Avg Sharpe: {avg_sharpe:.2f}")
        
        return results