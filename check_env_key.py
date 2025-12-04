import os
import sys

def check_keys():
    print("--- Environment Variable Check ---")
    
    # Check OS Environment
    api_key = os.environ.get('BYBIT_API_KEY')
    if api_key:
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        print(f"OS Env 'BYBIT_API_KEY': {masked} (Length: {len(api_key)})")
    else:
        print("OS Env 'BYBIT_API_KEY': NOT FOUND")

    # Check .env file if it exists
    if os.path.exists('.env'):
        print("\n--- .env File Check ---")
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.strip().startswith('BYBIT_API_KEY'):
                        parts = line.split('=')
                        if len(parts) >= 2:
                            val = parts[1].strip().strip('"').strip("'")
                            masked = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "***"
                            print(f"Found in .env: {masked} (Length: {len(val)})")
        except Exception as e:
            print(f"Error reading .env: {e}")
    else:
        print("\n.env file not found.")

if __name__ == "__main__":
    check_keys()
