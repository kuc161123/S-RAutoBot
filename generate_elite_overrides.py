import pandas as pd
import yaml
import os

def generate_elite():
    # Read CSV
    try:
        df = pd.read_csv('walk_forward_results.csv')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter
    # Criteria: Test WR > 50% AND Test Trades >= 10
    elite = df[(df['Test_WR'] > 50) & (df['Test_N'] >= 10)].copy()
    
    print(f"Total Strategies: {len(df)}")
    print(f"Elite Strategies: {len(elite)}")
    
    # Group by Symbol
    overrides = {}
    
    for _, row in elite.iterrows():
        sym = row['Symbol']
        side = row['Side'].lower() # 'long' or 'short'
        combo = row['Combo']
        
        if sym not in overrides:
            overrides[sym] = {}
            
        if side not in overrides[sym]:
            overrides[sym][side] = []
            
        # Add combo and comment
        overrides[sym][side].append(combo)
        # We can't easily add comments in the structure for YAML dump, 
        # so we will format it manually or just dump the structure.
        # Let's dump structure first.
    
    # Generate YAML manually to include comments
    yaml_lines = [
        "# Elite Strategies (Walk-Forward Validated)",
        "# Criteria: Test WR > 50% AND Test N >= 10",
        f"# Generated: {pd.Timestamp.now()}",
        ""
    ]
    
    # Sort symbols
    sorted_syms = sorted(overrides.keys())
    
    for sym in sorted_syms:
        yaml_lines.append(f"{sym}:")
        data = overrides[sym]
        
        if 'long' in data:
            yaml_lines.append("  long:")
            for combo in data['long']:
                # Find stats for comment
                stats = elite[(elite['Symbol'] == sym) & (elite['Side'] == 'LONG') & (elite['Combo'] == combo)].iloc[0]
                yaml_lines.append(f"    - \"{combo}\"")
                yaml_lines.append(f"    # Train: {stats['Train_WR']:.1f}%({stats['Train_N']}) -> Test: {stats['Test_WR']:.1f}%({stats['Test_N']})")
                
        if 'short' in data:
            yaml_lines.append("  short:")
            for combo in data['short']:
                # Find stats for comment
                stats = elite[(elite['Symbol'] == sym) & (elite['Side'] == 'SHORT') & (elite['Combo'] == combo)].iloc[0]
                yaml_lines.append(f"    - \"{combo}\"")
                yaml_lines.append(f"    # Train: {stats['Train_WR']:.1f}%({stats['Train_N']}) -> Test: {stats['Test_WR']:.1f}%({stats['Test_N']})")
        
        yaml_lines.append("")
        
    with open('symbol_overrides_400.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
        
    print(f"✅ Generated symbol_overrides_400.yaml with {len(sorted_syms)} symbols")

if __name__ == "__main__":
    generate_elite()
