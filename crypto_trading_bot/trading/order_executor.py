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
    
    def __init__(self, exchange_client, position_manager, config, telegram_notifier=None):
        self.exchange = exchange_client
        self.position_manager = position_manager
        self.config = config
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
            
            # Use the single leverage value from config (scalping-focused)
            leverage = self.config.leverage  # Single leverage for all trades
            
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
            
            # CRITICAL SAFETY CHECK: Log and validate position size
            position_value_usd = position_size * signal.price
            potential_loss = position_size * abs(signal.price - signal.stop_loss)
            
            logger.info(f"=== ORDER VALIDATION for {symbol} ===")
            logger.info(f"Position size: {position_size:.8f} coins")
            logger.info(f"Position value: ${position_value_usd:.2f}")
            logger.info(f"Potential loss: ${potential_loss:.2f}")
            logger.info(f"Balance: ${balance:.2f}")
            
            # ULTRA STRICT SAFETY LIMITS - HARDCODED
            # For $250 balance with 0.5% risk = $1.25 risk
            # Max position should be small to prevent blowouts
            max_position_value = balance * 0.1  # MAX 10% of balance (e.g., $25 on $250)
            max_risk = balance * 0.005  # Never risk more than 0.5% (matches RISK_PER_TRADE)
            
            if position_value_usd > max_position_value:
                logger.error(f"BLOCKED: Position value ${position_value_usd:.2f} exceeds safety limit ${max_position_value:.2f}")
                await self._notify(f"❌ Order blocked for {symbol}: Position too large (${position_value_usd:.2f})")
                return False
            
            if potential_loss > max_risk:
                logger.error(f"BLOCKED: Risk ${potential_loss:.2f} exceeds safety limit ${max_risk:.2f}")
                await self._notify(f"❌ Order blocked for {symbol}: Risk too high (${potential_loss:.2f})")
                return False
            
            # FINAL CHECK: Ensure position size in USD makes sense
            # Recalculate to be absolutely sure
            final_position_value = position_size * signal.price
            if final_position_value > balance * 0.15:  # Absolute max 15% of balance
                logger.error(f"FINAL BLOCK: Position value ${final_position_value:.2f} still too large")
                # Force reduce position size
                max_safe_position_size = (balance * 0.1) / signal.price
                logger.info(f"Forcing position size from {position_size:.8f} to {max_safe_position_size:.8f}")
                position_size = max_safe_position_size
            
            # Set leverage for the symbol
            self.exchange.set_leverage(symbol, leverage)
            
            # Place the order
            side = "Buy" if signal.action == "BUY" else "Sell"
            
            logger.info(f"Placing order: {side} {position_size:.8f} {symbol}")
            
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
            # Escape special characters for Telegram markdown
            error_msg = str(e).replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("]", "\\]")
            await self._notify(f"❌ Error executing signal: {error_msg}")
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