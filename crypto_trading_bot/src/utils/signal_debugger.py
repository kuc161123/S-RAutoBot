"""
Signal Generation Debugger
Helps diagnose why signals aren't being generated
"""
import structlog
from datetime import datetime
from typing import Dict, List, Optional

logger = structlog.get_logger(__name__)

class SignalDebugger:
    """
    Tracks and logs signal generation process to identify issues
    """
    
    def __init__(self):
        self.scan_count = 0
        self.zone_count = 0
        self.structure_count = 0
        self.signal_count = 0
        self.last_scan = None
        self.last_signal = None
        self.reasons_no_signal = []
        
    def log_scan_start(self, symbol: str):
        """Log when scanning starts for a symbol"""
        self.scan_count += 1
        self.last_scan = datetime.now()
        logger.debug(f"🔍 Scanning {symbol} (scan #{self.scan_count})")
        
    def log_htf_zones(self, symbol: str, zones: List):
        """Log HTF zones found"""
        if zones:
            self.zone_count += len(zones)
            logger.info(f"📊 {symbol}: Found {len(zones)} HTF zones")
            for i, zone in enumerate(zones[:3]):  # Log top 3
                logger.debug(f"  Zone {i+1}: {zone.get('type')} [{zone.get('lower'):.2f}-{zone.get('upper'):.2f}] strength={zone.get('strength'):.1f}")
        else:
            logger.debug(f"❌ {symbol}: No HTF zones found")
            self.reasons_no_signal.append(f"{symbol}: No HTF zones")
            
    def log_ltf_structure(self, symbol: str, structure):
        """Log LTF structure analysis"""
        if structure:
            self.structure_count += 1
            logger.info(f"📈 {symbol}: LTF structure = {structure.trend.value if hasattr(structure, 'trend') else 'unknown'}")
            if hasattr(structure, 'structure_pattern'):
                patterns = [p.value if hasattr(p, 'value') else str(p) for p in structure.structure_pattern[-3:]]
                logger.debug(f"  Patterns: {patterns}")
        else:
            logger.debug(f"❌ {symbol}: No LTF structure detected")
            self.reasons_no_signal.append(f"{symbol}: No LTF structure")
            
    def log_confluence_check(self, symbol: str, has_confluence: bool, reason: str = ""):
        """Log confluence checking result"""
        if has_confluence:
            logger.info(f"✅ {symbol}: HTF/LTF confluence found!")
        else:
            logger.debug(f"❌ {symbol}: No confluence - {reason}")
            self.reasons_no_signal.append(f"{symbol}: No confluence - {reason}")
            
    def log_signal_generated(self, symbol: str, signal: Dict):
        """Log when a signal is generated"""
        self.signal_count += 1
        self.last_signal = datetime.now()
        
        logger.info(f"🎯 SIGNAL GENERATED for {symbol}!")
        logger.info(f"  Direction: {signal.get('direction', 'unknown')}")
        logger.info(f"  Entry: {signal.get('entry_price', 0):.2f}")
        logger.info(f"  Stop: {signal.get('stop_loss', 0):.2f}")
        logger.info(f"  Confidence: {signal.get('confidence', 0):.1f}%")
        
    def log_no_signal(self, symbol: str, reason: str):
        """Log why no signal was generated"""
        logger.debug(f"⚠️ {symbol}: No signal - {reason}")
        self.reasons_no_signal.append(f"{symbol}: {reason}")
        
    def get_summary(self) -> str:
        """Get summary of signal generation"""
        time_since_scan = (datetime.now() - self.last_scan).total_seconds() if self.last_scan else 999999
        time_since_signal = (datetime.now() - self.last_signal).total_seconds() if self.last_signal else 999999
        
        summary = [
            "=== SIGNAL GENERATION SUMMARY ===",
            f"Scans performed: {self.scan_count}",
            f"HTF zones found: {self.zone_count}",
            f"LTF structures analyzed: {self.structure_count}",
            f"Signals generated: {self.signal_count}",
            f"Last scan: {time_since_scan:.0f}s ago",
            f"Last signal: {time_since_signal:.0f}s ago" if self.last_signal else "Last signal: Never",
            "",
            "Recent no-signal reasons:"
        ]
        
        # Show last 10 reasons
        for reason in self.reasons_no_signal[-10:]:
            summary.append(f"  - {reason}")
            
        if self.signal_count == 0 and self.scan_count > 0:
            summary.append("")
            summary.append("⚠️ NO SIGNALS GENERATED - Check:")
            summary.append("  1. Zone detection thresholds")
            summary.append("  2. Market structure requirements")
            summary.append("  3. Confluence requirements")
            summary.append("  4. Risk parameters")
            
        return "\n".join(summary)
        
    def reset_stats(self):
        """Reset statistics"""
        self.scan_count = 0
        self.zone_count = 0
        self.structure_count = 0
        self.signal_count = 0
        self.reasons_no_signal = []

# Global instance
signal_debugger = SignalDebugger()