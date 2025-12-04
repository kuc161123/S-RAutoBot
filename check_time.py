import time
import requests

def check_time():
    try:
        local_time_ms = int(time.time() * 1000)
        print(f"Local Time: {local_time_ms}")
        
        resp = requests.get("https://api.bybit.com/v5/market/time")
        data = resp.json()
        
        if data['retCode'] == 0:
            server_time_ms = int(data['result']['timeSecond']) * 1000
            # Note: timeSecond is in seconds, but result might have nano/micro. 
            # Actually v5 returns timeSecond (string) and timeNano (string).
            # Let's use timeNano for precision if available, or just timeSecond.
            # Wait, let's check the raw response.
            print(f"Server Response: {data}")
            
            # Usually 'timeSecond' is a string of seconds.
            server_time_ms = int(data['result']['timeSecond']) * 1000
            
            diff = server_time_ms - local_time_ms
            print(f"Server Time: {server_time_ms}")
            print(f"Difference: {diff} ms ({diff/1000:.2f} seconds)")
            
            if abs(diff) > 5000:
                print("❌ CRITICAL: Time drift is huge!")
            else:
                print("✅ Time sync is within acceptable limits.")
        else:
            print(f"Failed to get server time: {data}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_time()
