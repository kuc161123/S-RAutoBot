"""
Environment and configuration validation utilities
"""
import os
import re
from typing import Dict, List, Tuple, Optional
import structlog
from datetime import datetime
import asyncio
import httpx

logger = structlog.get_logger(__name__)

class ConfigValidator:
    """Validate configuration and environment variables"""
    
    @staticmethod
    def validate_environment() -> Tuple[bool, List[str]]:
        """
        Validate all required environment variables
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        warnings = []
        
        # Check Bybit API credentials
        bybit_key = os.getenv('BYBIT_API_KEY', '')
        bybit_secret = os.getenv('BYBIT_API_SECRET', '')
        
        if not bybit_key or 'YOUR_' in bybit_key.upper():
            errors.append("❌ BYBIT_API_KEY is not set or contains placeholder")
        elif len(bybit_key) < 10:
            errors.append("❌ BYBIT_API_KEY appears to be invalid (too short)")
            
        if not bybit_secret or 'YOUR_' in bybit_secret.upper():
            errors.append("❌ BYBIT_API_SECRET is not set or contains placeholder")
        elif len(bybit_secret) < 20:
            errors.append("❌ BYBIT_API_SECRET appears to be invalid (too short)")
        
        # Check Telegram configuration
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        telegram_chat_ids = os.getenv('TELEGRAM_ALLOWED_CHAT_IDS', '')
        
        if not telegram_token or 'YOUR_' in telegram_token.upper():
            errors.append("❌ TELEGRAM_BOT_TOKEN is not set or contains placeholder")
        elif not re.match(r'^\d+:[A-Za-z0-9_-]+$', telegram_token):
            errors.append("❌ TELEGRAM_BOT_TOKEN format is invalid")
            
        if not telegram_chat_ids:
            warnings.append("⚠️ TELEGRAM_ALLOWED_CHAT_IDS is empty - bot will accept commands from anyone")
        
        # Check database URLs
        database_url = os.getenv('DATABASE_URL', '')
        redis_url = os.getenv('REDIS_URL', '')
        
        if not database_url:
            warnings.append("⚠️ DATABASE_URL not set - using default (may not work in production)")
        elif not (database_url.startswith('postgresql://') or database_url.startswith('postgres://')):
            errors.append("❌ DATABASE_URL must start with postgresql:// or postgres://")
            
        if not redis_url:
            warnings.append("⚠️ REDIS_URL not set - using default (may not work in production)")
        elif not redis_url.startswith('redis://'):
            errors.append("❌ REDIS_URL must start with redis://")
        
        # Check security
        secret_key = os.getenv('SECRET_KEY', '')
        if not secret_key or 'GENERATE' in secret_key.upper() or len(secret_key) < 32:
            errors.append("❌ SECRET_KEY must be at least 32 characters (use: openssl rand -hex 32)")
        
        # Check trading parameters
        risk_percent = os.getenv('DEFAULT_RISK_PERCENT', '1.0')
        try:
            risk = float(risk_percent)
            if risk <= 0 or risk > 10:
                warnings.append(f"⚠️ DEFAULT_RISK_PERCENT={risk}% is unusual (recommended: 0.5-2%)")
        except:
            errors.append("❌ DEFAULT_RISK_PERCENT must be a number")
        
        # Log results
        for error in errors:
            logger.error(error)
        for warning in warnings:
            logger.warning(warning)
            
        return (len(errors) == 0, errors, warnings)
    
    @staticmethod
    async def test_bybit_connection(api_key: str, api_secret: str, testnet: bool = True) -> Tuple[bool, str]:
        """Test Bybit API connection"""
        try:
            from pybit.unified_trading import HTTP
            
            session = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret
            )
            
            # Try to get account info
            result = session.get_wallet_balance(
                accountType="UNIFIED" if not testnet else "CONTRACT"
            )
            
            if result["retCode"] == 0:
                balance = result["result"]["list"][0].get("totalWalletBalance", 0)
                return True, f"✅ Bybit connection successful (Balance: ${balance})"
            else:
                return False, f"❌ Bybit API error: {result.get('retMsg', 'Unknown error')}"
                
        except Exception as e:
            return False, f"❌ Bybit connection failed: {str(e)}"
    
    @staticmethod
    async def test_telegram_bot(token: str) -> Tuple[bool, str]:
        """Test Telegram bot token"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://api.telegram.org/bot{token}/getMe")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        bot_name = data["result"].get("username", "Unknown")
                        return True, f"✅ Telegram bot connected (@{bot_name})"
                    else:
                        return False, f"❌ Telegram API error: {data.get('description', 'Unknown error')}"
                elif response.status_code == 401:
                    return False, "❌ Invalid Telegram bot token"
                else:
                    return False, f"❌ Telegram API returned status {response.status_code}"
                    
        except Exception as e:
            return False, f"❌ Telegram connection failed: {str(e)}"
    
    @staticmethod
    async def test_database_connection(database_url: str) -> Tuple[bool, str]:
        """Test PostgreSQL connection"""
        try:
            import asyncpg
            
            conn = await asyncpg.connect(database_url)
            version = await conn.fetchval('SELECT version()')
            await conn.close()
            
            return True, f"✅ PostgreSQL connected ({version.split(',')[0]})"
            
        except Exception as e:
            return False, f"❌ Database connection failed: {str(e)}"
    
    @staticmethod
    async def test_redis_connection(redis_url: str) -> Tuple[bool, str]:
        """Test Redis connection"""
        try:
            import redis.asyncio as redis
            
            r = redis.from_url(redis_url)
            await r.ping()
            info = await r.info('server')
            version = info.get('redis_version', 'Unknown')
            await r.close()
            
            return True, f"✅ Redis connected (v{version})"
            
        except Exception as e:
            return False, f"❌ Redis connection failed: {str(e)}"

class StartupValidator:
    """Comprehensive startup validation"""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.results = {
            'environment': False,
            'bybit': False,
            'telegram': False,
            'database': False,
            'redis': False
        }
        self.errors = []
        self.warnings = []
    
    async def run_all_checks(self) -> bool:
        """Run all validation checks"""
        logger.info("=" * 60)
        logger.info("🚀 CRYPTO TRADING BOT - STARTUP VALIDATION")
        logger.info("=" * 60)
        
        # 1. Environment variables
        logger.info("\n📋 Checking Environment Variables...")
        env_valid, errors, warnings = self.validator.validate_environment()
        self.results['environment'] = env_valid
        self.errors.extend(errors)
        self.warnings.extend(warnings)
        
        if not env_valid:
            logger.error("Environment validation failed - check errors above")
            self._print_summary()
            return False
        
        # 2. Bybit connection
        logger.info("\n🔌 Testing Bybit API Connection...")
        bybit_key = os.getenv('BYBIT_API_KEY', '')
        bybit_secret = os.getenv('BYBIT_API_SECRET', '')
        bybit_testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        
        if bybit_key and not 'YOUR_' in bybit_key:
            success, message = await self.validator.test_bybit_connection(
                bybit_key, bybit_secret, bybit_testnet
            )
            self.results['bybit'] = success
            logger.info(message)
            if not success:
                self.errors.append(message)
        
        # 3. Telegram bot
        logger.info("\n💬 Testing Telegram Bot...")
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        
        if telegram_token and not 'YOUR_' in telegram_token:
            success, message = await self.validator.test_telegram_bot(telegram_token)
            self.results['telegram'] = success
            logger.info(message)
            if not success:
                self.errors.append(message)
        
        # 4. Database
        logger.info("\n🗄️ Testing Database Connection...")
        database_url = os.getenv('DATABASE_URL', '')
        
        if database_url:
            success, message = await self.validator.test_database_connection(database_url)
            self.results['database'] = success
            logger.info(message)
            if not success:
                self.warnings.append(message)  # Non-critical
        
        # 5. Redis
        logger.info("\n📦 Testing Redis Connection...")
        redis_url = os.getenv('REDIS_URL', '')
        
        if redis_url:
            success, message = await self.validator.test_redis_connection(redis_url)
            self.results['redis'] = success
            logger.info(message)
            if not success:
                self.warnings.append(message)  # Non-critical
        
        # Print summary
        self._print_summary()
        
        # Determine if we can start
        critical_ok = self.results['environment'] and self.results['bybit'] and self.results['telegram']
        
        if critical_ok:
            logger.info("\n✅ All critical systems passed - Bot can start!")
            return True
        else:
            logger.error("\n❌ Critical systems failed - Bot cannot start")
            logger.error("Please fix the errors above and restart")
            return False
    
    def _print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        # Status table
        for component, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{component.upper():15} {status}")
        
        # Errors
        if self.errors:
            logger.info("\n🚨 ERRORS (Must Fix):")
            for error in self.errors:
                logger.error(f"  {error}")
        
        # Warnings
        if self.warnings:
            logger.info("\n⚠️ WARNINGS (Should Review):")
            for warning in self.warnings:
                logger.warning(f"  {warning}")
        
        logger.info("=" * 60)

async def validate_startup() -> bool:
    """Main validation entry point"""
    validator = StartupValidator()
    return await validator.run_all_checks()