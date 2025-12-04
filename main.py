import asyncio
import logging
from autobot.core.bot import VWAPBot

if __name__ == "__main__":
    try:
        bot = VWAPBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
