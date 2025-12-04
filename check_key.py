import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

print(f"Testing Key: {api_key}")
print(f"Testing Secret: {api_secret[:5]}...")

try:
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    print(session.get_wallet_balance(accountType="UNIFIED"))
    print("✅ Key is VALID")
except Exception as e:
    print(f"❌ Key is INVALID: {e}")
