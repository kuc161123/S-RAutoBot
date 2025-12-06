import asyncio
import logging
import threading
from autobot.core.bot import VWAPBot
from dashboard import app

def run_dashboard():
    """Run the web dashboard in a separate thread"""
    try:
        print("📊 Dashboard starting at http://localhost:8888")
        app.run(host='0.0.0.0', port=8888, debug=False, use_reloader=False)
    except Exception as e:
        logging.error(f"Dashboard failed to start: {e}")

if __name__ == "__main__":
    try:
        # Start dashboard in background thread
        dash_thread = threading.Thread(target=run_dashboard, daemon=True)
        dash_thread.start()
        
        # Start main bot
        bot = VWAPBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
