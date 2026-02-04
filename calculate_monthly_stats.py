
import pandas as pd
import datetime

def analyze_monthly_profitability(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Convert entry_time to datetime
    if 'entry_time' not in df.columns:
        print("Column 'entry_time' not found.")
        return
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    # Extract Month-Year
    df['month_year'] = df['entry_time'].dt.to_period('M')
    
    # Calculate monthly R sum
    if 'r_result' not in df.columns:
        print("Column 'r_result' not found.")
        return
    
    monthly_stats = df.groupby('month_year')['r_result'].sum().reset_index()
    monthly_stats['month_year'] = monthly_stats['month_year'].astype(str)
    
    # Sort by date
    monthly_stats = monthly_stats.sort_values('month_year')
    
    print("Monthly Profitability (R):")
    print(monthly_stats.to_string(index=False))
    
    # Total
    total_r = df['r_result'].sum()
    print(f"\nTotal R: {total_r:.2f}")

    # Also breakdown by symbol if needed, but user asked for "each month"
    # let's only provide monthly total for now.

if __name__ == "__main__":
    analyze_monthly_profitability("/Users/lualakol/AutoTrading Bot/validation_6month_trades.csv")
