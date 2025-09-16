"""
Hardcoded Symbol Clusters for Bybit USDT Perpetuals
Based on market cap, category, and trading characteristics
"""

# Cluster 1: Blue Chip (Top tier, high market cap, established)
BLUE_CHIP = [
    'BTCUSDT',      # Bitcoin - #1 market cap
    'ETHUSDT',      # Ethereum - #2 market cap
    'BNBUSDT',      # Binance Coin - Top 5
    'SOLUSDT',      # Solana - Major Layer 1
    'XRPUSDT',      # Ripple - Top 10
    'ADAUSDT',      # Cardano - Major Layer 1
    'AVAXUSDT',     # Avalanche - Major Layer 1
    'MATICUSDT',    # Polygon - Major Layer 2
    'DOTUSDT',      # Polkadot - Major Layer 1
    'LINKUSDT',     # Chainlink - Leading Oracle
    'LTCUSDT',      # Litecoin - Old guard
    'UNIUSDT',      # Uniswap - Leading DEX
    'ATOMUSDT',     # Cosmos - Major interoperability
    'NEARUSDT',     # NEAR Protocol - Major Layer 1
]

# Cluster 2: Stable/Low Volatility (Stablecoins and low volatility assets)
STABLE = [
    'USDTUSDT',     # Tether (if exists)
    'USDCUSDT',     # USD Coin
    'BUSDUSDT',     # Binance USD
    'DAIUSDT',      # DAI Stablecoin
    'TUSDUSDT',     # TrueUSD
    'USDPUSDT',     # Pax Dollar
    'FRAXUSDT',     # Frax
    'LDOUSDT',      # Lido DAO - Liquid staking
    'RPLLUSDT',     # Rocket Pool - Liquid staking
]

# Cluster 3: Meme/High Volatility
MEME_VOLATILE = [
    'DOGEUSDT',     # Dogecoin
    'SHIBUSDT',     # Shiba Inu
    '1000SHIBUSDT', # Shiba Inu (1000x)
    'PEPEUSDT',     # Pepe
    '1000PEPEUSDT', # Pepe (1000x)
    'FLOKIUSDT',    # Floki Inu
    '1000FLOKIUSDT',# Floki (1000x)
    'BONKUSDT',     # Bonk
    '1000BONKUSDT', # Bonk (1000x)
    'WIFUSDT',      # dogwifhat
    'BOMEUSDT',     # Book of Meme
    'MEMEUSDT',     # Memecoin
    'BABYDOGEUSDT', # Baby Doge
    'ELON1000USDT', # Dogelon Mars
    'SATSUSDT',     # 1000SATS
    '1000SATSUSDT', # 1000SATS
    'ORDIUSDT',     # ORDI (BRC-20)
    'WLDUSDT',      # Worldcoin - High volatility
    'AKITAUSDT',    # Akita Inu
    'KISHUUSDT',    # Kishu Inu
]

# Cluster 4: Mid-Cap Alts (Established projects, follow BTC)
MID_CAP = [
    'AAVEUSDT',     # Aave - DeFi Blue Chip
    'MKRUSDT',      # Maker - DeFi
    'SNXUSDT',      # Synthetix - DeFi
    'CRVUSDT',      # Curve - DeFi
    'COMPUSDT',     # Compound - DeFi
    'FILUSDT',      # Filecoin - Storage
    'FTMUSDT',      # Fantom - Layer 1
    'SANDUSDT',     # Sandbox - Metaverse
    'MANAUSDT',     # Decentraland - Metaverse
    'AXSUSDT',      # Axie Infinity - Gaming
    'THETAUSDT',    # Theta - Video
    'ICPUSDT',      # Internet Computer
    'VECHOUSDT',    # VeChain - Supply Chain
    'ALGOUSDT',     # Algorand - Layer 1
    'XLMUSDT',      # Stellar
    'EGLDUSDT',     # MultiversX
    'HBARUSDT',     # Hedera
    'APTUSDT',      # Aptos - New Layer 1
    'ARBUSDT',      # Arbitrum - Layer 2
    'OPUSDT',       # Optimism - Layer 2
    'STXUSDT',      # Stacks - Bitcoin L2
    'QNTUSDT',      # Quant
    'GRTUSDT',      # The Graph
    'CHZUSDT',      # Chiliz - Sports
    'ENJUSDT',      # Enjin - Gaming
    'FLOWUSDT',     # Flow - NFTs
    'IMXUSDT',      # Immutable X - Gaming L2
    'LDOUSDT',      # Lido DAO
    'RPLLUSDT',     # Rocket Pool
    'INJUSDT',      # Injective
    'SUIUSDT',      # Sui - New Layer 1
    'SEIUSDT',      # Sei - New Layer 1
    'TONUSDT',      # Toncoin
    'KASUSDT',      # Kaspa
]

# Cluster 5: Small Cap/Others (Everything else, newer projects, low liquidity)
# This will be the default for any symbol not in the above lists

def get_hardcoded_clusters():
    """
    Returns a complete mapping of symbols to cluster IDs
    """
    clusters = {}
    
    # Assign cluster 1 (Blue Chip)
    for symbol in BLUE_CHIP:
        clusters[symbol] = 1
    
    # Assign cluster 2 (Stable)
    for symbol in STABLE:
        clusters[symbol] = 2
        
    # Assign cluster 3 (Meme/Volatile)
    for symbol in MEME_VOLATILE:
        clusters[symbol] = 3
        
    # Assign cluster 4 (Mid-Cap)
    for symbol in MID_CAP:
        clusters[symbol] = 4
    
    # Note: Cluster 5 (Small Cap) is assigned by default to any unlisted symbol
    
    return clusters


def get_symbol_cluster(symbol: str) -> int:
    """
    Get the cluster ID for a specific symbol
    Returns 5 (Small Cap) if not found in hardcoded lists
    """
    if symbol in BLUE_CHIP:
        return 1
    elif symbol in STABLE:
        return 2
    elif symbol in MEME_VOLATILE:
        return 3
    elif symbol in MID_CAP:
        return 4
    else:
        return 5  # Small Cap/Others


def get_cluster_name(cluster_id: int) -> str:
    """Get human-readable cluster name"""
    names = {
        1: "Blue Chip",
        2: "Stable/Low Volatility", 
        3: "Meme/High Volatility",
        4: "Mid-Cap Alts",
        5: "Small Cap/Others"
    }
    return names.get(cluster_id, "Unknown")


def get_cluster_description(cluster_id: int) -> str:
    """Get detailed cluster description"""
    descriptions = {
        1: "Blue Chip (Top market cap, established projects like BTC, ETH)",
        2: "Stable/Low Volatility (Stablecoins and liquid staking tokens)",
        3: "Meme/High Volatility (Meme coins and highly volatile assets)",
        4: "Mid-Cap Alts (Established DeFi, Gaming, L1/L2 projects)",
        5: "Small Cap/Others (Newer projects, low liquidity tokens)"
    }
    return descriptions.get(cluster_id, "Unknown cluster")


# Additional categorization for future use
CATEGORIES = {
    'layer1': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 
               'NEARUSDT', 'FTMUSDT', 'ALGOUSDT', 'ICPUSDT', 'APTUSDT', 'SUIUSDT', 
               'SEIUSDT', 'TONUSDT', 'KASUSDT'],
    'layer2': ['MATICUSDT', 'ARBUSDT', 'OPUSDT', 'STXUSDT', 'IMXUSDT'],
    'defi': ['UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'SNXUSDT', 'CRVUSDT', 'COMPUSDT',
             'LDOUSDT', 'RPLLUSDT'],
    'oracle': ['LINKUSDT'],
    'metaverse': ['SANDUSDT', 'MANAUSDT', 'ENJUSDT'],
    'gaming': ['AXSUSDT', 'ENJUSDT', 'FLOWUSDT', 'IMXUSDT'],
    'meme': MEME_VOLATILE,
    'stablecoin': STABLE,
    'exchange': ['BNBUSDT'],
    'storage': ['FILUSDT'],
    'interoperability': ['ATOMUSDT', 'DOTUSDT'],
}