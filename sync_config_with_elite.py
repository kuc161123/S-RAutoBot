import yaml

# Read elite symbols from overrides
with open('symbol_overrides_400.yaml', 'r') as f:
    content = f.read()
    
elite_symbols = []
for line in content.split('\n'):
    if line and not line.startswith('#') and not line.startswith(' ') and ':' in line:
        symbol = line.split(':')[0].strip()
        if symbol:
            elite_symbols.append(symbol)

print(f"Found {len(elite_symbols)} elite symbols")

# Read config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update symbols
config['trade']['symbols'] = sorted(elite_symbols)

# Write back
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config.yaml with {len(elite_symbols)} elite symbols")
