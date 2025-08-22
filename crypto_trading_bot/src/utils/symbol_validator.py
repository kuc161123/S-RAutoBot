"""
Symbol Validator and Manager
Ensures only valid symbols are used and handles symbol-specific errors
"""
import asyncio
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class SymbolValidator:
    """Validates and manages trading symbols"""
    
    def __init__(self, bybit_client):
        self.client = bybit_client
        self.valid_symbols: Set[str] = set()
        self.invalid_symbols: Set[str] = set()
        self.symbol_info: Dict[str, Dict] = {}
        self.last_validation = None
        self.validation_interval = timedelta(hours=1)
        
    async def initialize(self):
        """Initialize and validate all symbols"""
        await self.validate_all_symbols()
        
    async def validate_all_symbols(self):
        """Validate all symbols from settings"""
        try:
            logger.info("Validating trading symbols...")
            
            # Get all active symbols from exchange
            exchange_symbols = await self._get_exchange_symbols()
            
            # Get symbols from settings
            from ..config import settings
            configured_symbols = settings.default_symbols
            
            # Validate each symbol
            for symbol in configured_symbols:
                if await self.validate_symbol(symbol, exchange_symbols):
                    self.valid_symbols.add(symbol)
                else:
                    self.invalid_symbols.add(symbol)
                    logger.warning(f"Invalid symbol removed: {symbol}")
            
            self.last_validation = datetime.now()
            
            logger.info(f"Symbol validation complete: {len(self.valid_symbols)} valid, {len(self.invalid_symbols)} invalid")
            
        except Exception as e:
            logger.error(f"Error validating symbols: {e}")
    
    async def _get_exchange_symbols(self) -> Set[str]:
        """Get all active symbols from exchange"""
        try:
            # Refresh instruments
            await self.client.refresh_instruments()
            
            # Get all symbols
            exchange_symbols = set(self.client.instruments.keys())
            
            return exchange_symbols
            
        except Exception as e:
            logger.error(f"Error getting exchange symbols: {e}")
            return set()
    
    async def validate_symbol(self, symbol: str, exchange_symbols: Optional[Set[str]] = None) -> bool:
        """Validate a single symbol"""
        try:
            # Check if symbol exists on exchange
            if exchange_symbols is None:
                exchange_symbols = await self._get_exchange_symbols()
            
            if symbol not in exchange_symbols:
                return False
            
            # Get symbol info
            info = self.client.get_instrument(symbol)
            if not info:
                return False
            
            # Check if symbol is active
            status = info.get('status', '')
            if status != 'Trading':
                return False
            
            # Check contract type (only USDT perpetuals)
            if not symbol.endswith('USDT'):
                return False
            
            # Store symbol info
            self.symbol_info[symbol] = {
                'min_qty': float(info.get('min_qty', 0)),
                'max_qty': float(info.get('max_qty', 0)),
                'qty_step': float(info.get('qty_step', 0)),
                'min_price': float(info.get('min_price', 0)),
                'max_price': float(info.get('max_price', 0)),
                'tick_size': float(info.get('tick_size', 0)),
                'status': status,
                'base_currency': info.get('base_currency', ''),
                'quote_currency': info.get('quote_currency', 'USDT')
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_valid_symbols(self) -> List[str]:
        """Get list of valid symbols"""
        # Re-validate if needed
        if self.last_validation is None or \
           datetime.now() - self.last_validation > self.validation_interval:
            asyncio.create_task(self.validate_all_symbols())
        
        return list(self.valid_symbols)
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol in self.valid_symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        return self.symbol_info.get(symbol)
    
    def remove_invalid_symbol(self, symbol: str):
        """Remove a symbol from valid list"""
        if symbol in self.valid_symbols:
            self.valid_symbols.remove(symbol)
            self.invalid_symbols.add(symbol)
            logger.info(f"Symbol {symbol} marked as invalid")


class SymbolErrorHandler:
    """Handles symbol-specific errors"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.max_errors = 5
        self.blacklisted_symbols: Set[str] = set()
        
    def handle_symbol_error(self, symbol: str, error: Exception) -> bool:
        """
        Handle symbol error and return whether to continue using symbol
        """
        error_str = str(error)
        
        # Check for specific error types
        if "Invalid symbol" in error_str or "not found" in error_str:
            self.blacklist_symbol(symbol, "Invalid symbol")
            return False
        
        if "Insufficient balance" in error_str:
            logger.warning(f"Insufficient balance for {symbol}")
            return True  # Don't blacklist, just skip this trade
        
        if "Position already exists" in error_str:
            logger.debug(f"Position already exists for {symbol}")
            return True  # Normal condition
        
        # Count errors
        self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
        
        # Blacklist if too many errors
        if self.error_counts[symbol] >= self.max_errors:
            self.blacklist_symbol(symbol, f"Too many errors ({self.error_counts[symbol]})")
            return False
        
        return True
    
    def blacklist_symbol(self, symbol: str, reason: str):
        """Blacklist a symbol"""
        self.blacklisted_symbols.add(symbol)
        logger.warning(f"Symbol {symbol} blacklisted: {reason}")
    
    def is_blacklisted(self, symbol: str) -> bool:
        """Check if symbol is blacklisted"""
        return symbol in self.blacklisted_symbols
    
    def reset_error_count(self, symbol: str):
        """Reset error count for a symbol"""
        if symbol in self.error_counts:
            del self.error_counts[symbol]
    
    def get_error_summary(self) -> Dict:
        """Get error summary"""
        return {
            'error_counts': self.error_counts,
            'blacklisted': list(self.blacklisted_symbols),
            'total_errors': sum(self.error_counts.values())
        }


# Global instances
symbol_validator = None
symbol_error_handler = SymbolErrorHandler()


def get_symbol_validator(bybit_client) -> SymbolValidator:
    """Get or create symbol validator instance"""
    global symbol_validator
    if symbol_validator is None:
        symbol_validator = SymbolValidator(bybit_client)
    return symbol_validator