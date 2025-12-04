from autobot.data.storage import PostgresCandleStorage
import logging

logging.basicConfig(level=logging.INFO)

def run():
    print("Initializing Database...")
    try:
        storage = PostgresCandleStorage()
        print("✅ Database initialized successfully.")
        print(f"Stats: {storage.get_stats()}")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    run()
