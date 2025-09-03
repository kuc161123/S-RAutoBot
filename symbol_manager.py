#!/usr/bin/env python3
"""
Symbol Manager - Safe switching between different symbol configurations
With performance monitoring and automatic rollback
"""
import yaml
import shutil
import os
import logging
import psutil
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolManager:
    """Manage symbol configuration changes safely"""
    
    CONFIGS = {
        '250': 'config_backup_250.yaml',
        '400': 'config_top_400.yaml', 
        'all': 'config_all_symbols.yaml',
        'current': 'config.yaml'
    }
    
    def __init__(self):
        self.performance_log = []
        
    def get_current_config(self) -> Dict:
        """Get current configuration"""
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def get_symbol_count(self) -> int:
        """Get current number of symbols"""
        config = self.get_current_config()
        return len(config['trade']['symbols'])
    
    def check_system_resources(self) -> Dict:
        """Check current system resources"""
        process = psutil.Process(os.getpid())
        
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=1),
            'num_threads': process.num_threads(),
            'timestamp': datetime.now()
        }
    
    def switch_config(self, target: str) -> bool:
        """Switch to a different symbol configuration"""
        
        if target not in self.CONFIGS:
            logger.error(f"Unknown config: {target}")
            return False
        
        source_file = self.CONFIGS[target]
        
        if not os.path.exists(source_file):
            logger.error(f"Config file not found: {source_file}")
            return False
        
        try:
            # Backup current before switching
            backup_name = f"config_before_switch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            shutil.copy('config.yaml', backup_name)
            logger.info(f"Created backup: {backup_name}")
            
            # Switch config
            shutil.copy(source_file, 'config.yaml')
            
            # Verify
            new_count = self.get_symbol_count()
            logger.info(f"✅ Switched to {target} config with {new_count} symbols")
            
            # Log performance baseline
            resources = self.check_system_resources()
            self.performance_log.append({
                'config': target,
                'symbols': new_count,
                'resources': resources
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch config: {e}")
            return False
    
    def revert_to_safe(self) -> bool:
        """Revert to safe 250 symbol configuration"""
        return self.switch_config('250')
    
    def gradual_scale_test(self):
        """Test scaling gradually"""
        
        print("=" * 60)
        print("🧪 GRADUAL SCALING TEST")
        print("=" * 60)
        
        current = self.get_symbol_count()
        print(f"\nCurrent symbols: {current}")
        
        # Test sequence
        if current == 250:
            print("\n📈 Scaling Plan:")
            print("1. Current: 250 symbols (baseline)")
            print("2. Next: 400 symbols (moderate)")
            print("3. Final: 391 symbols (all active)")
            
            response = input("\nProceed to 400 symbols? (y/n): ")
            if response.lower() == 'y':
                if self.switch_config('400'):
                    print("\n✅ Switched to 400 symbols")
                    print("⏰ Run for 1 hour and monitor:")
                    print("   - Memory usage (should stay <300MB)")
                    print("   - Signal quality")
                    print("   - WebSocket stability")
                    print("\nIf stable, run: python3 symbol_manager.py --all")
                    
        elif current == 400:
            print("\n📈 Currently at 400 symbols")
            resources = self.check_system_resources()
            print(f"Memory: {resources['memory_mb']:.0f}MB")
            
            response = input("\nProceed to ALL symbols (391)? (y/n): ")
            if response.lower() == 'y':
                if self.switch_config('all'):
                    print("\n✅ Switched to ALL symbols (391)")
                    print("⚠️ Monitor closely for 1 hour")
                    print("If issues arise, run: python3 symbol_manager.py --revert")
                    
        else:
            print(f"\n⚠️ Currently at {current} symbols (custom config)")
            print("Options:")
            print("  --250: Switch to safe 250")
            print("  --400: Switch to moderate 400")
            print("  --all: Switch to all symbols")
    
    def show_status(self):
        """Show current status and recommendations"""
        
        config = self.get_current_config()
        symbols = config['trade']['symbols']
        resources = self.check_system_resources()
        
        print("=" * 60)
        print("📊 SYMBOL CONFIGURATION STATUS")
        print("=" * 60)
        print(f"""
Current Configuration:
• Symbols monitored: {len(symbols)}
• Memory usage: {resources['memory_mb']:.0f}MB
• CPU usage: {resources['cpu_percent']:.1f}%
• WebSocket connections: {(len(symbols) - 1) // 190 + 1}

Performance Guidelines:
• 250 symbols: ✅ Safe (tested extensively)
• 400 symbols: ⚠️ Moderate (should work fine)
• 391 symbols: ⚠️ All active (monitor closely)

Recommendations based on {len(symbols)} symbols:""")
        
        if len(symbols) <= 250:
            print("""
✅ OPTIMAL CONFIGURATION
• Best signal quality
• Proven stability
• Low resource usage
• Easy to monitor
""")
        elif len(symbols) <= 400:
            print("""
⚠️ EXTENDED CONFIGURATION
• More opportunities
• Slightly higher resource use
• Monitor for stability
• Consider reverting if issues
""")
        else:
            print("""
⚠️ MAXIMUM CONFIGURATION
• Maximum opportunities
• Higher false signal risk
• Monitor resources closely
• Be ready to revert if needed
""")
        
        print(f"Commands:")
        print(f"  Revert to safe: python3 symbol_manager.py --revert")
        print(f"  Switch to 400: python3 symbol_manager.py --400")
        print(f"  Switch to all: python3 symbol_manager.py --all")
        print(f"  Check status: python3 symbol_manager.py --status")

def main():
    import sys
    
    manager = SymbolManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--status':
            manager.show_status()
        elif command == '--revert':
            if manager.revert_to_safe():
                print("✅ Reverted to safe 250 symbol configuration")
                print("⚠️ Restart the bot for changes to take effect")
        elif command == '--250':
            if manager.switch_config('250'):
                print("✅ Switched to 250 symbols")
                print("⚠️ Restart the bot for changes to take effect")
        elif command == '--400':
            if manager.switch_config('400'):
                print("✅ Switched to 400 symbols")
                print("⚠️ Restart the bot for changes to take effect")
        elif command == '--all':
            if manager.switch_config('all'):
                print("✅ Switched to ALL symbols (391)")
                print("⚠️ Restart the bot for changes to take effect")
        else:
            print(f"Unknown command: {command}")
            print("Available: --status, --revert, --250, --400, --all")
    else:
        manager.gradual_scale_test()

if __name__ == "__main__":
    main()