"""
Symbol R:R Mapping Helper
==========================
Provides utilities for loading and accessing per-symbol R:R ratios.
"""

import yaml
from typing import Dict, Optional


class SymbolRRConfig:
    """Manages per-symbol Risk:Reward ratios"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize symbol R:R configuration
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.symbols: Dict[str, dict] = {}
        self.load_config()
        
    def load_config(self):
        """Load symbol configuration from config.yaml"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            self.symbols = config.get('symbols', {})
            
            # Validate
            enabled_count = sum(1 for s in self.symbols.values() if s.get('enabled', False))
            print(f"[SymbolRRConfig] Loaded {len(self.symbols)} symbols ({enabled_count} enabled)")
            
        except Exception as e:
            print(f"[SymbolRRConfig] Error loading config: {e}")
            self.symbols = {}
    
    def get_rr_for_symbol(self, symbol: str) -> Optional[float]:
        """
        Get R:R ratio for a specific symbol
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            R:R ratio or None if symbol not found/disabled
        """
        if symbol not in self.symbols:
            return None
            
        symbol_config = self.symbols[symbol]
        
        if not symbol_config.get('enabled', False):
            return None
            
        return symbol_config.get('rr')
    
    def is_symbol_enabled(self, symbol: str) -> bool:
        """
        Check if symbol is enabled for trading
        
        Args:
            symbol: Trading pair
            
        Returns:
            True if enabled, False otherwise
        """
        if symbol not in self.symbols:
            return False
            
        return self.symbols[symbol].get('enabled', False)
    
    def get_enabled_symbols(self) -> list[str]:
        """
        Get list of all enabled symbols
        
        Returns:
            List of enabled symbol names
        """
        return [
            symbol for symbol, config in self.symbols.items()
            if config.get('enabled', False)
        ]
    
    def get_symbols_by_rr(self, rr: float) -> list[str]:
        """
        Get symbols with a specific R:R ratio
        
        Args:
            rr: R:R ratio to filter by
            
        Returns:
            List of symbols with that R:R
        """
        return [
            symbol for symbol, config in self.symbols.items()
            if config.get('enabled', False) and config.get('rr') == rr
        ]
    
    def get_rr_distribution(self) -> Dict[float, int]:
        """
        Get distribution of R:R ratios across enabled symbols
        
        Returns:
            Dictionary of {rr: count}
        """
        distribution = {}
        for symbol, config in self.symbols.items():
            if config.get('enabled', False):
                rr = config.get('rr')
                if rr:
                    distribution[rr] = distribution.get(rr, 0) + 1
        return distribution


# Convenience function for quick access
_config_instance = None

def get_symbol_rr(symbol: str) -> Optional[float]:
    """
    Get R:R ratio for a symbol (convenience function)
    
    Args:
        symbol: Trading pair
        
    Returns:
        R:R ratio or None
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = SymbolRRConfig()
    return _config_instance.get_rr_for_symbol(symbol)


def get_enabled_symbols() -> list[str]:
    """
    Get all enabled symbols (convenience function)
    
    Returns:
        List of enabled symbols
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = SymbolRRConfig()
    return _config_instance.get_enabled_symbols()


if __name__ == "__main__":
    # Test
    config = SymbolRRConfig()
    
    print("\n=== Symbol R:R Configuration ===")
    print(f"Total symbols: {len(config.symbols)}")
    print(f"Enabled symbols: {len(config.get_enabled_symbols())}")
    
    print("\nR:R Distribution:")
    for rr, count in sorted(config.get_rr_distribution().items()):
        print(f"  {rr}:1 → {count} symbols")
    
    print("\nSample symbols:")
    for symbol in ['BTCUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT']:
        rr = config.get_rr_for_symbol(symbol)
        enabled = config.is_symbol_enabled(symbol)
        status = "✅" if enabled else "❌"
        print(f"  {status} {symbol}: {rr}:1 R:R" if rr else f"  {status} {symbol}: Not configured")
