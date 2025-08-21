import asyncio
import signal
import sys
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from telegram import Update
import structlog
from datetime import datetime, timedelta
import json

from .config import settings
from .api.bybit_client import BybitClient
from .strategy.supply_demand import SupplyDemandStrategy
from .telegram.bot import TradingBot
from .trading.order_manager import OrderManager
from .db.database import init_db, close_db, DatabaseManager
from .utils.logging import logger, trading_logger, get_prometheus_metrics
from .trading.trading_engine import TradingEngine

# Initialize logger
logger = structlog.get_logger(__name__)

# Global instances
bybit_client = None
strategy = None
telegram_bot = None
order_manager = None
trading_engine = None
db_manager = DatabaseManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global bybit_client, strategy, telegram_bot, order_manager, trading_engine
    
    try:
        logger.info("Starting Crypto Trading Bot...")
        
        # Initialize database
        await init_db()
        
        # Initialize Bybit client
        bybit_client = BybitClient()
        await bybit_client.initialize()
        
        # Initialize strategy
        strategy = SupplyDemandStrategy()
        
        # Initialize order manager
        order_manager = OrderManager(bybit_client)
        
        # Initialize Telegram bot
        telegram_bot = TradingBot(bybit_client, strategy)
        await telegram_bot.initialize()
        
        # Initialize trading engine
        trading_engine = TradingEngine(
            bybit_client=bybit_client,
            strategy=strategy,
            order_manager=order_manager,
            telegram_bot=telegram_bot
        )
        
        # Start background tasks
        asyncio.create_task(trading_engine.run())
        asyncio.create_task(instrument_refresh_task())
        asyncio.create_task(daily_reset_task())
        
        # Subscribe to WebSocket streams
        setup_websocket_subscriptions()
        
        # Start Telegram bot polling if not using webhook
        if not settings.telegram_webhook_url:
            asyncio.create_task(telegram_bot.run_polling())
        
        logger.info("Bot initialization complete")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise
    
    finally:
        logger.info("Shutting down bot...")
        
        # Stop trading engine
        if trading_engine:
            await trading_engine.stop()
        
        # Close connections
        if bybit_client:
            await bybit_client.close()
        
        # Close database
        await close_db()
        
        logger.info("Bot shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Bot",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security dependency
async def verify_telegram_secret(request: Request):
    """Verify Telegram webhook secret"""
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret_token != settings.telegram_secret_token:
        raise HTTPException(status_code=403, detail="Invalid secret token")
    return True

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Crypto Trading Bot",
        "status": "running",
        "version": "1.0.0",
        "environment": settings.environment
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global trading_engine
    
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "healthy",
            "bybit_api": "healthy" if bybit_client else "unhealthy",
            "trading_engine": "running" if trading_engine and trading_engine.is_running else "stopped",
            "telegram_bot": "healthy" if telegram_bot else "unhealthy"
        }
    }
    
    # Check if any component is unhealthy
    if any(v == "unhealthy" for v in status["components"].values()):
        status["status"] = "degraded"
    
    return status

@app.post("/telegram", dependencies=[Depends(verify_telegram_secret)])
async def telegram_webhook(request: Request):
    """Telegram webhook endpoint"""
    global telegram_bot
    
    try:
        # Parse update
        data = await request.json()
        update = Update.de_json(data, telegram_bot.application.bot)
        
        # Process update
        await telegram_bot.application.process_update(update)
        
        return {"ok": True}
        
    except Exception as e:
        logger.error(f"Error processing telegram update: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    metrics_text = get_prometheus_metrics()
    return Response(content=metrics_text, media_type="text/plain")

@app.get("/api/status")
async def get_status():
    """Get bot status"""
    global trading_engine, order_manager
    
    if not trading_engine:
        return {"error": "Trading engine not initialized"}
    
    account_info = await bybit_client.get_account_info()
    positions = await bybit_client.get_positions()
    daily_stats = order_manager.get_daily_stats()
    
    return {
        "trading_enabled": trading_engine.is_running,
        "account": {
            "balance": float(account_info.get('totalWalletBalance', 0)),
            "available": float(account_info.get('availableBalance', 0)),
            "unrealized_pnl": float(account_info.get('totalPerpUPL', 0))
        },
        "positions": len([p for p in positions if float(p['size']) > 0]),
        "daily_stats": daily_stats,
        "monitored_symbols": len(trading_engine.monitored_symbols),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/enable")
async def enable_trading():
    """Enable trading"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    await trading_engine.start()
    
    # Notify via Telegram
    for chat_id in settings.telegram_allowed_chat_ids:
        await telegram_bot.send_notification(
            chat_id,
            "✅ Trading enabled via API"
        )
    
    return {"success": True, "message": "Trading enabled"}

@app.post("/api/disable")
async def disable_trading():
    """Disable trading"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    await trading_engine.stop()
    
    # Notify via Telegram
    for chat_id in settings.telegram_allowed_chat_ids:
        await telegram_bot.send_notification(
            chat_id,
            "🛑 Trading disabled via API"
        )
    
    return {"success": True, "message": "Trading disabled"}

@app.post("/api/emergency_stop")
async def emergency_stop():
    """Emergency stop - close all positions"""
    global order_manager, trading_engine
    
    if not order_manager:
        raise HTTPException(status_code=500, detail="Order manager not initialized")
    
    # Stop trading
    if trading_engine:
        await trading_engine.stop()
    
    # Close all positions
    await order_manager.emergency_stop()
    
    # Notify via Telegram
    for chat_id in settings.telegram_allowed_chat_ids:
        await telegram_bot.send_notification(
            chat_id,
            "🚨 EMERGENCY STOP EXECUTED - All positions closed"
        )
    
    return {"success": True, "message": "Emergency stop executed"}

@app.get("/api/positions")
async def get_positions():
    """Get open positions"""
    global bybit_client
    
    if not bybit_client:
        raise HTTPException(status_code=500, detail="Bybit client not initialized")
    
    positions = await bybit_client.get_positions()
    open_positions = [p for p in positions if float(p['size']) > 0]
    
    return {
        "positions": open_positions,
        "count": len(open_positions)
    }

@app.get("/api/zones/{symbol}")
async def get_zones(symbol: str):
    """Get active zones for a symbol"""
    global strategy
    
    if not strategy:
        raise HTTPException(status_code=500, detail="Strategy not initialized")
    
    zones = strategy.get_active_zones(symbol.upper())
    
    return {
        "symbol": symbol.upper(),
        "zones": [
            {
                "type": z.zone_type.value,
                "upper": z.upper_bound,
                "lower": z.lower_bound,
                "score": z.score,
                "touches": z.touches,
                "age_hours": z.age_hours
            }
            for z in zones
        ],
        "count": len(zones)
    }

# Background tasks
async def instrument_refresh_task():
    """Periodically refresh instrument data"""
    while True:
        try:
            await asyncio.sleep(settings.instrument_refresh_interval_hours * 3600)
            
            if bybit_client:
                await bybit_client.refresh_instruments()
                logger.info("Instruments refreshed")
                
        except Exception as e:
            logger.error(f"Error refreshing instruments: {e}")

async def daily_reset_task():
    """Reset daily statistics at UTC midnight"""
    while True:
        try:
            # Calculate seconds until midnight UTC
            now = datetime.utcnow()
            midnight = datetime(now.year, now.month, now.day) + timedelta(days=1)
            seconds_until_midnight = (midnight - now).total_seconds()
            
            await asyncio.sleep(seconds_until_midnight)
            
            if order_manager:
                await order_manager.reset_daily_stats()
                
            # Send daily summary
            await send_daily_summary()
            
            logger.info("Daily statistics reset")
            
        except Exception as e:
            logger.error(f"Error in daily reset: {e}")

async def send_daily_summary():
    """Send daily trading summary to users"""
    global order_manager, telegram_bot, db_manager
    
    if not order_manager or not telegram_bot:
        return
    
    stats = order_manager.get_daily_stats()
    
    # Get additional stats from database
    daily_trades = db_manager.get_daily_trades()
    
    summary = {
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'total_trades': stats['trades'],
        'winning_trades': len([t for t in daily_trades if t.pnl > 0]),
        'losing_trades': len([t for t in daily_trades if t.pnl < 0]),
        'total_pnl': stats['pnl'],
        'total_fees': stats['fees']
    }
    
    # Send to all authorized users
    from .telegram.formatters import format_daily_summary
    message = format_daily_summary(summary)
    
    for chat_id in settings.telegram_allowed_chat_ids:
        await telegram_bot.send_notification(chat_id, message)

def setup_websocket_subscriptions():
    """Setup WebSocket subscriptions"""
    global bybit_client, order_manager
    
    if not bybit_client or not order_manager:
        return
    
    # Subscribe to private streams
    bybit_client.subscribe_positions(order_manager.handle_position_update)
    bybit_client.subscribe_orders(order_manager.handle_order_update)
    
    logger.info("WebSocket subscriptions setup complete")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )