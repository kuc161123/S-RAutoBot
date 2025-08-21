import structlog
import logging
import sys
from pathlib import Path
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import json

from ..config import settings

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Prometheus metrics
trade_counter = Counter('trades_total', 'Total number of trades', ['symbol', 'side', 'result'])
trade_pnl = Histogram('trade_pnl', 'Trade PnL distribution', ['symbol'])
active_positions = Gauge('active_positions', 'Number of active positions')
account_balance = Gauge('account_balance', 'Account balance in USD')
win_rate = Gauge('win_rate', 'Current win rate percentage')
zone_detection_counter = Counter('zones_detected', 'Number of zones detected', ['symbol', 'type'])
api_request_counter = Counter('api_requests', 'API requests', ['endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])
error_counter = Counter('errors_total', 'Total errors', ['type'])

def setup_logging():
    """Configure structured logging"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging
    log_level = getattr(logging, settings.log_level.upper())
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # File handler - main log
    file_handler = logging.FileHandler(
        LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(log_level)
    
    # Error file handler
    error_handler = logging.FileHandler(
        LOG_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    )
    error_handler.setLevel(logging.ERROR)
    
    # Trade log handler
    trade_handler = logging.FileHandler(
        LOG_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
    )
    trade_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler, error_handler],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create specialized loggers
    trade_logger = logging.getLogger('trades')
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)
    
    return structlog.get_logger()

class TradingLogger:
    """Specialized logger for trading events"""
    
    def __init__(self):
        self.logger = structlog.get_logger('trading')
        self.trade_logger = logging.getLogger('trades')
    
    def log_trade_opened(self, trade_data: dict):
        """Log trade opening"""
        self.trade_logger.info(json.dumps({
            'event': 'trade_opened',
            'timestamp': datetime.utcnow().isoformat(),
            **trade_data
        }))
        
        # Update metrics
        trade_counter.labels(
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            result='opened'
        ).inc()
        active_positions.inc()
    
    def log_trade_closed(self, trade_data: dict):
        """Log trade closing"""
        self.trade_logger.info(json.dumps({
            'event': 'trade_closed',
            'timestamp': datetime.utcnow().isoformat(),
            **trade_data
        }))
        
        # Update metrics
        result = 'win' if trade_data['pnl'] > 0 else 'loss'
        trade_counter.labels(
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            result=result
        ).inc()
        trade_pnl.labels(symbol=trade_data['symbol']).observe(trade_data['pnl'])
        active_positions.dec()
    
    def log_zone_detected(self, zone_data: dict):
        """Log zone detection"""
        self.logger.info(
            "Zone detected",
            symbol=zone_data['symbol'],
            zone_type=zone_data['type'],
            score=zone_data['score']
        )
        
        zone_detection_counter.labels(
            symbol=zone_data['symbol'],
            type=zone_data['type']
        ).inc()
    
    def log_api_call(self, endpoint: str, duration: float, status: str):
        """Log API call"""
        api_request_counter.labels(endpoint=endpoint, status=status).inc()
        api_request_duration.labels(endpoint=endpoint).observe(duration)
    
    def log_error(self, error_type: str, error_msg: str, context: dict = None):
        """Log error"""
        self.logger.error(
            error_msg,
            error_type=error_type,
            context=context
        )
        error_counter.labels(type=error_type).inc()
    
    def update_account_metrics(self, balance: float, current_win_rate: float):
        """Update account metrics"""
        account_balance.set(balance)
        win_rate.set(current_win_rate)

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.logger = structlog.get_logger('performance')
        self.metrics = {
            'api_latency': [],
            'zone_detection_time': [],
            'order_execution_time': []
        }
    
    def record_latency(self, operation: str, duration: float):
        """Record operation latency"""
        if operation in self.metrics:
            self.metrics[operation].append(duration)
            
            # Keep only last 100 measurements
            if len(self.metrics[operation]) > 100:
                self.metrics[operation] = self.metrics[operation][-100:]
        
        self.logger.info(
            f"Operation {operation} completed",
            duration_ms=duration * 1000
        )
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        import numpy as np
        
        stats = {}
        for operation, durations in self.metrics.items():
            if durations:
                stats[operation] = {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'p95': np.percentile(durations, 95),
                    'p99': np.percentile(durations, 99)
                }
        
        return stats

def get_prometheus_metrics():
    """Get Prometheus metrics in text format"""
    return generate_latest()

# Initialize loggers
logger = setup_logging()
trading_logger = TradingLogger()
performance_monitor = PerformanceMonitor()