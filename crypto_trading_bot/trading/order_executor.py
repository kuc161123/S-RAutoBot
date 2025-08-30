"""
Order execution module
"""
from typing import Optional
import asyncio
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)

class OrderExecutor:
    """Execute trading orders"""
    
    def __init__(self, exchange_client, position_manager, telegram_notifier=None):
        self.exchange = exchange_client
        self.position_manager = position_manager
        self.telegram = telegram_notifier
        self.pending_orders = {}
        
        logger.info("Order executor initialized")
    
    async def execute_signal(self, signal) -> bool:
        """Execute a trading signal"""
        try:
            symbol = signal.symbol
            
            # Check if we can open position
            if not self.position_manager.can_open_position(symbol):
                logger.warning(f"Cannot open position for {symbol}")
                return False
            
            # Get account balance
            balance = self.exchange.get_account_balance()
            if not balance or balance < 100:  # Minimum $100 balance
                logger.error(f"Insufficient balance: ${balance}")
                await self._notify(f"❌ Insufficient balance: ${balance:.2f}")
                return False
            
            # Determine leverage based on signal type (scalping vs swing)
            leverage = 5 if hasattr(signal, 'signal_type') and signal.signal_type == "SCALP" else 10
            
            # Calculate position size
            position_size = self.position_manager.calculate_position_size(
                balance=balance,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                leverage=leverage
            )
            
            if position_size <= 0:
                logger.error(f"Invalid position size calculated for {symbol}")
                return False
            
            # Set leverage for the symbol
            self.exchange.set_leverage(symbol, leverage)
            
            # Place the order
            side = "Buy" if signal.action == "BUY" else "Sell"
            
            order_result = self.exchange.place_order(
                symbol=symbol,
                side=side,
                qty=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_result:
                # Add to position manager
                self.position_manager.add_position(
                    symbol=symbol,
                    side=signal.action,
                    entry_price=signal.price,
                    size=position_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                
                # Send notification with signal type
                signal_type = getattr(signal, 'signal_type', 'TRADE')
                risk_reward = getattr(signal, 'risk_reward', 0)
                
                message = (
                    f"✅ **{signal_type} Position Opened**\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {signal.action}\n"
                    f"Entry: ${signal.price:.4f}\n"
                    f"Size: {position_size:.4f}\n"
                    f"Stop Loss: ${signal.stop_loss:.4f}\n"
                    f"Take Profit: ${signal.take_profit:.4f}\n"
                    f"R:R Ratio: {risk_reward:.2f}\n"
                    f"Reason: {signal.reason}\n"
                    f"Confidence: {signal.confidence*100:.1f}%"
                )
                
                await self._notify(message)
                
                logger.info(f"Order executed successfully for {symbol}")
                return True
            else:
                logger.error(f"Failed to execute order for {symbol}")
                await self._notify(f"❌ Failed to execute order for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            await self._notify(f"❌ Error executing signal: {e}")
            return False
    
    async def check_positions(self):
        """Check and manage existing positions"""
        try:
            # Get positions from exchange
            exchange_positions = self.exchange.get_positions()
            
            for position in exchange_positions:
                symbol = position['symbol']
                current_price = position.get('entry_price', 0)  # Use mark price in real implementation
                
                # Update position P&L
                self.position_manager.update_position(symbol, current_price)
                
                # Check if should close
                close_reason = self.position_manager.should_close_position(symbol, current_price)
                
                if close_reason:
                    await self.close_position(symbol, close_reason)
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    async def close_position(self, symbol: str, reason: str = "MANUAL") -> bool:
        """Close a position"""
        try:
            # Close on exchange
            if self.exchange.close_position(symbol):
                # Get final price
                positions = self.exchange.get_positions()
                exit_price = next((p['entry_price'] for p in positions if p['symbol'] == symbol), 0)
                
                # Update position manager
                self.position_manager.close_position(symbol, exit_price, reason)
                
                # Get position details for notification
                position = self.position_manager.positions.get(symbol)
                if position:
                    pnl_emoji = "🟢" if position.pnl > 0 else "🔴"
                    
                    message = (
                        f"{pnl_emoji} **Position Closed**\n"
                        f"Symbol: {symbol}\n"
                        f"Exit: ${exit_price:.4f}\n"
                        f"P&L: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)\n"
                        f"Reason: {reason}"
                    )
                    
                    await self._notify(message)
                
                logger.info(f"Position closed: {symbol} ({reason})")
                return True
            else:
                logger.error(f"Failed to close position: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def close_all_positions(self):
        """Close all open positions"""
        positions = list(self.position_manager.positions.keys())
        
        for symbol in positions:
            await self.close_position(symbol, "CLOSE_ALL")
            await asyncio.sleep(1)  # Avoid rate limiting
        
        logger.info(f"Closed {len(positions)} positions")
    
    async def _notify(self, message: str):
        """Send notification via Telegram"""
        if self.telegram:
            try:
                await self.telegram.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")