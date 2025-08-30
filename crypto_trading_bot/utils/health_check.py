"""
Health check server for monitoring
"""
import asyncio
from aiohttp import web
import structlog

logger = structlog.get_logger(__name__)

class HealthCheckServer:
    """Simple HTTP server for health checks"""
    
    def __init__(self, bot_instance, port=8080):
        self.bot = bot_instance
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.status_check)
    
    async def health_check(self, request):
        """Simple health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'service': 'crypto-trading-bot'
        })
    
    async def status_check(self, request):
        """Detailed status check"""
        try:
            status = {
                'status': 'running' if self.bot and self.bot.is_running else 'stopped',
                'exchange_connected': bool(self.bot.exchange) if self.bot else False,
                'telegram_connected': bool(self.bot.telegram_bot) if self.bot else False,
                'positions': len(self.bot.position_manager.positions) if self.bot and self.bot.position_manager else 0
            }
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error in status check: {e}")
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def start(self):
        """Start health check server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()
            logger.info(f"Health check server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")
    
    async def stop(self):
        """Stop health check server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Health check server stopped")