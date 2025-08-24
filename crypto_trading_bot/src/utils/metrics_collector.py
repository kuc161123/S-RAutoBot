"""
Prometheus Metrics Collector
Tracks and exports trading bot performance metrics
"""
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
import time
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)

# Define metrics
# Counters
trades_total = Counter('trading_bot_trades_total', 'Total number of trades executed', ['symbol', 'side'])
signals_generated = Counter('trading_bot_signals_total', 'Total signals generated', ['symbol', 'signal_type'])
errors_total = Counter('trading_bot_errors_total', 'Total errors encountered', ['component', 'error_type'])
websocket_reconnects = Counter('trading_bot_ws_reconnects_total', 'WebSocket reconnection count')

# Gauges
account_balance = Gauge('trading_bot_account_balance', 'Current account balance in USD')
open_positions = Gauge('trading_bot_open_positions', 'Number of open positions')
active_symbols = Gauge('trading_bot_active_symbols', 'Number of actively monitored symbols')
ml_model_accuracy = Gauge('trading_bot_ml_accuracy', 'ML model accuracy', ['model_name'])
circuit_breaker_state = Gauge('trading_bot_circuit_breaker', 'Circuit breaker state (0=closed, 1=open, 2=half-open)', ['component'])

# Histograms
trade_pnl = Histogram('trading_bot_trade_pnl', 'Trade P&L distribution', buckets=(-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, float('inf')))
order_latency = Histogram('trading_bot_order_latency_seconds', 'Order execution latency', buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')))
signal_processing_time = Histogram('trading_bot_signal_processing_seconds', 'Signal processing time', buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, float('inf')))

# Summaries
api_request_duration = Summary('trading_bot_api_request_duration_seconds', 'API request duration', ['endpoint'])
ml_prediction_time = Summary('trading_bot_ml_prediction_seconds', 'ML prediction time')


class MetricsCollector:
    """
    Collects and manages Prometheus metrics
    """
    
    def __init__(self):
        self.start_time = time.time()
        
    def record_trade(self, symbol: str, side: str, pnl: float):
        """Record trade execution metrics"""
        trades_total.labels(symbol=symbol, side=side).inc()
        trade_pnl.observe(pnl)
        
    def record_signal(self, symbol: str, signal_type: str):
        """Record signal generation"""
        signals_generated.labels(symbol=symbol, signal_type=signal_type).inc()
        
    def record_error(self, component: str, error_type: str):
        """Record error occurrence"""
        errors_total.labels(component=component, error_type=error_type).inc()
        
    def update_account_balance(self, balance: float):
        """Update account balance gauge"""
        account_balance.set(balance)
        
    def update_open_positions(self, count: int):
        """Update open positions count"""
        open_positions.set(count)
        
    def update_active_symbols(self, count: int):
        """Update active symbols count"""
        active_symbols.set(count)
        
    def update_ml_accuracy(self, model_name: str, accuracy: float):
        """Update ML model accuracy"""
        ml_model_accuracy.labels(model_name=model_name).set(accuracy)
        
    def update_circuit_breaker(self, component: str, state: str):
        """Update circuit breaker state"""
        state_value = {'closed': 0, 'open': 1, 'half_open': 2}.get(state, 0)
        circuit_breaker_state.labels(component=component).set(state_value)
        
    def record_order_latency(self, latency_seconds: float):
        """Record order execution latency"""
        order_latency.observe(latency_seconds)
        
    def record_signal_processing(self, processing_time: float):
        """Record signal processing time"""
        signal_processing_time.observe(processing_time)
        
    def record_websocket_reconnect(self):
        """Record WebSocket reconnection"""
        websocket_reconnects.inc()
        
    @api_request_duration.time()
    def track_api_request(self, endpoint: str):
        """Context manager to track API request duration"""
        return api_request_duration.labels(endpoint=endpoint).time()
        
    @ml_prediction_time.time()
    def track_ml_prediction(self):
        """Context manager to track ML prediction time"""
        return ml_prediction_time.time()
    
    def get_metrics(self) -> bytes:
        """
        Generate Prometheus metrics in text format
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest()
    
    def get_uptime(self) -> float:
        """Get bot uptime in seconds"""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary for logging
        
        Returns:
            Dictionary with key metrics
        """
        return {
            'uptime_hours': self.get_uptime() / 3600,
            'total_trades': sum(trades_total._metrics.values()) if trades_total._metrics else 0,
            'total_signals': sum(signals_generated._metrics.values()) if signals_generated._metrics else 0,
            'total_errors': sum(errors_total._metrics.values()) if errors_total._metrics else 0,
            'current_balance': account_balance._value.get() if hasattr(account_balance, '_value') else 0,
            'open_positions': open_positions._value.get() if hasattr(open_positions, '_value') else 0
        }


# Global instance
metrics_collector = MetricsCollector()