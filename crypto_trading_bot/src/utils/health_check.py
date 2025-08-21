"""
Health check and monitoring system
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import psutil
import structlog
from enum import Enum

logger = structlog.get_logger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentHealth:
    """Track health of a single component"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.HEALTHY
        self.last_check = datetime.now()
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.error_message = None
        self.metadata = {}
    
    def update(self, healthy: bool, message: str = None, metadata: Dict = None):
        """Update component health status"""
        self.last_check = datetime.now()
        
        if healthy:
            self.status = HealthStatus.HEALTHY
            self.last_success = datetime.now()
            self.consecutive_failures = 0
            self.error_message = None
        else:
            self.consecutive_failures += 1
            self.error_message = message
            
            if self.consecutive_failures >= 3:
                self.status = HealthStatus.UNHEALTHY
            else:
                self.status = HealthStatus.DEGRADED
        
        if metadata:
            self.metadata.update(metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        uptime = (datetime.now() - self.last_success).total_seconds()
        
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "uptime_seconds": uptime,
            "consecutive_failures": self.consecutive_failures,
            "error": self.error_message,
            "metadata": self.metadata
        }

class HealthMonitor:
    """System-wide health monitoring"""
    
    def __init__(self):
        self.components = {
            "database": ComponentHealth("PostgreSQL Database"),
            "redis": ComponentHealth("Redis Cache"),
            "bybit_api": ComponentHealth("Bybit API"),
            "bybit_websocket": ComponentHealth("Bybit WebSocket"),
            "telegram_bot": ComponentHealth("Telegram Bot"),
            "trading_engine": ComponentHealth("Trading Engine"),
            "strategy": ComponentHealth("Strategy Engine")
        }
        self.start_time = datetime.now()
        self.check_interval = 30  # seconds
        self.monitoring_task = None
    
    async def start_monitoring(self):
        """Start background health monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await self.check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_all_components(self):
        """Check health of all components"""
        tasks = [
            self.check_database(),
            self.check_redis(),
            self.check_bybit_api(),
            self.check_telegram(),
            self.check_trading_engine(),
            self.check_system_resources()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def check_database(self) -> bool:
        """Check database connectivity"""
        try:
            from ..db.database import get_db
            
            with get_db() as db:
                result = db.execute("SELECT 1").scalar()
                self.components["database"].update(True, metadata={"connected": True})
                return True
                
        except Exception as e:
            self.components["database"].update(False, str(e))
            return False
    
    async def check_redis(self) -> bool:
        """Check Redis connectivity"""
        try:
            import redis.asyncio as redis
            from ..config import settings
            
            r = redis.from_url(settings.redis_url)
            await r.ping()
            info = await r.info()
            
            self.components["redis"].update(True, metadata={
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A")
            })
            
            await r.close()
            return True
            
        except Exception as e:
            self.components["redis"].update(False, str(e))
            return False
    
    async def check_bybit_api(self) -> bool:
        """Check Bybit API connectivity"""
        try:
            from pybit.unified_trading import HTTP
            from ..config import settings
            
            session = HTTP(
                testnet=settings.bybit_testnet,
                api_key=settings.bybit_api_key,
                api_secret=settings.bybit_api_secret
            )
            
            result = session.get_server_time()
            
            if result["retCode"] == 0:
                server_time = result["result"]["timeSecond"]
                local_time = datetime.now().timestamp()
                time_diff = abs(server_time - local_time)
                
                self.components["bybit_api"].update(True, metadata={
                    "time_sync_diff": time_diff,
                    "testnet": settings.bybit_testnet
                })
                return True
            else:
                self.components["bybit_api"].update(False, result.get("retMsg"))
                return False
                
        except Exception as e:
            self.components["bybit_api"].update(False, str(e))
            return False
    
    async def check_telegram(self) -> bool:
        """Check Telegram bot connectivity"""
        try:
            import httpx
            from ..config import settings
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        self.components["telegram_bot"].update(True, metadata={
                            "bot_username": data["result"].get("username"),
                            "bot_id": data["result"].get("id")
                        })
                        return True
                
                self.components["telegram_bot"].update(False, f"API returned {response.status_code}")
                return False
                
        except Exception as e:
            self.components["telegram_bot"].update(False, str(e))
            return False
    
    async def check_trading_engine(self) -> bool:
        """Check if trading engine is running"""
        from ..main import trading_engine
        
        if trading_engine and trading_engine.is_running:
            status = trading_engine.get_status()
            self.components["trading_engine"].update(True, metadata={
                "active_positions": status.get("active_positions", 0),
                "monitored_symbols": len(status.get("monitored_symbols", [])),
                "daily_trades": status.get("daily_stats", {}).get("trades", 0)
            })
            return True
        else:
            self.components["trading_engine"].update(False, "Engine not running")
            return False
    
    async def check_system_resources(self):
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Warn if resources are high
            warnings = []
            if cpu_percent > 80:
                warnings.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 80:
                warnings.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                warnings.append(f"Low disk space: {disk.percent}% used")
            
            if warnings:
                logger.warning(f"System resource warnings: {', '.join(warnings)}")
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {}
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        unhealthy_count = sum(1 for c in self.components.values() 
                            if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in self.components.values() 
                           if c.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": self.get_overall_status().value,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "uptime_human": self._format_uptime(uptime),
            "components": {
                name: comp.to_dict() 
                for name, comp in self.components.items()
            },
            "system_resources": asyncio.run(self.check_system_resources())
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        
        return " ".join(parts) if parts else "< 1m"

# Global health monitor instance
health_monitor = HealthMonitor()