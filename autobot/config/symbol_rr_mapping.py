"""
Symbol Configuration Manager
============================
Manages per-symbol settings including:
- Enabled/disabled status
- R:R ratio
- Divergence type (REG_BULL, REG_BEAR, HID_BULL, HID_BEAR)

Supports the multi-divergence strategy with 271 validated symbols.
"""

import yaml
from typing import Dict, Optional, List


class SymbolRRConfig:
    """Manages per-symbol Risk:Reward ratios and divergence types"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize symbol configuration
        
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
            
            # Count by divergence type
            div_counts = {}
            for sym, cfg in self.symbols.items():
                if cfg.get('enabled', True):  # Default to True if missing
                    div = cfg.get('divergence_type', cfg.get('divergence', 'UNKNOWN'))
                    div_counts[div] = div_counts.get(div, 0) + 1
            
            enabled_count = sum(1 for s in self.symbols.values() if s.get('enabled', True))
            print(f"[SymbolConfig] Loaded {len(self.symbols)} symbols ({enabled_count} enabled)")
            
            # Print divergence breakdown
            if div_counts:
                print(f"[SymbolConfig] By divergence: {div_counts}")
            
        except Exception as e:
            print(f"[SymbolConfig] Error loading config: {e}")
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
        
        if not symbol_config.get('enabled', True):
            return None
            
        return symbol_config.get('rr')
    
    def get_divergence_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get allowed divergence type for a symbol
        
        Args:
            symbol: Trading pair
            
        Returns:
            Divergence code (REG_BULL, REG_BEAR, HID_BULL, HID_BEAR) or None
        """
        if symbol not in self.symbols:
            return None
            
        symbol_config = self.symbols[symbol]
        
        if not symbol_config.get('enabled', True):
            return None
            
        return symbol_config.get('divergence_type', symbol_config.get('divergence'))
    
    def get_symbol_config(self, symbol: str) -> Optional[dict]:
        """
        Get full configuration for a symbol
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict with enabled, rr, divergence or None
        """
        if symbol not in self.symbols:
            return None
            
        return self.symbols[symbol]
    
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
            
        return self.symbols[symbol].get('enabled', True)
    
    def is_divergence_allowed(self, symbol: str, divergence_code: str) -> bool:
        """
        Check if a specific divergence type is allowed for a symbol
        
        Args:
            symbol: Trading pair
            divergence_code: One of REG_BULL, REG_BEAR, HID_BULL, HID_BEAR
            
        Returns:
            True if this divergence type is allowed for this symbol
        """
        if symbol not in self.symbols:
            return False
            
        symbol_config = self.symbols[symbol]
        
        if not symbol_config.get('enabled', True):
            return False
            
        allowed_div = symbol_config.get('divergence_type', symbol_config.get('divergence'))
        return allowed_div == divergence_code
    
    def get_enabled_symbols(self) -> List[str]:
        """
        Get list of all enabled symbols
        
        Returns:
            List of enabled symbol names
        """
        return [
            symbol for symbol, config in self.symbols.items()
            if config.get('enabled', True)
        ]
    
    def get_symbols_by_divergence(self, divergence_code: str) -> List[str]:
        """
        Get symbols with a specific divergence type
        
        Args:
            divergence_code: One of REG_BULL, REG_BEAR, HID_BULL, HID_BEAR
            
        Returns:
            List of symbols with that divergence type
        """
        return [
            symbol for symbol, config in self.symbols.items()
            if config.get('enabled', False) and config.get('divergence') == divergence_code
        ]
    
    def get_symbols_by_rr(self, rr: float) -> List[str]:
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
    
    def get_divergence_summary(self) -> Dict[str, int]:
        """
        Get count of enabled symbols by divergence type
        
        Returns:
            Dict of {divergence_code: count}
        """
        counts = {}
        for sym, cfg in self.symbols.items():
            if cfg.get('enabled', True):
                div = cfg.get('divergence_type', cfg.get('divergence', 'UNKNOWN'))
                counts[div] = counts.get(div, 0) + 1
        return counts
    
    def get_total_enabled(self) -> int:
        """Get count of enabled symbols"""
        return sum(1 for s in self.symbols.values() if s.get('enabled', True))
    
    def get_rr_summary(self) -> Dict[float, int]:
        """
        Get count of enabled symbols by R:R ratio
        
        Returns:
            Dict of {rr: count}
        """
        counts = {}
        for sym, cfg in self.symbols.items():
            if cfg.get('enabled', True):
                rr = cfg.get('rr', 0)
                counts[rr] = counts.get(rr, 0) + 1
        return counts
