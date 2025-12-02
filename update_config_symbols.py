import yaml

# Load the 400 symbols
with open('symbols_400.yaml', 'r') as f:
    sym_data = yaml.safe_load(f)
    
# Load current config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update symbols
old_count = len(config['trade']['symbols'])
config['trade']['symbols'] = sym_data['symbols']

# Save updated config
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config.yaml:")
print(f"   Old symbol count: {old_count}")
print(f"   New symbol count: {len(config['trade']['symbols'])}")
print(f"   First 10: {config['trade']['symbols'][:10]}")
