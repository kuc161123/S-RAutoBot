"""
Fear & Greed Index Fetcher

Fetches crypto market sentiment from Alternative.me API.
- Free API (no authentication required)
- Updates daily
- Range: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
- Caches value for 24 hours to minimize API calls

Used to block trades during extreme sentiment conditions.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
import aiohttp

logger = logging.getLogger(__name__)


class FearGreedFetcher:
    """Fetches and caches Fear & Greed Index from Alternative.me."""

    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=24)

    async def get_index(self) -> Optional[int]:
        """
        Get current Fear & Greed Index value (0-100).

        Returns:
            int: 0-100 (0=extreme fear, 100=extreme greed)
            None: if API fails or data unavailable
        """
        # Check cache
        if self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache.get('value')

        # Fetch from API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data and len(data['data']) > 0:
                            value = int(data['data'][0]['value'])
                            classification = data['data'][0]['value_classification']

                            # Update cache
                            self._cache = {
                                'value': value,
                                'classification': classification,
                                'timestamp': data['data'][0]['timestamp']
                            }
                            self._cache_time = datetime.now()

                            logger.info(f"Fear & Greed Index: {value} ({classification})")
                            return value
                    else:
                        logger.warning(f"Fear & Greed API returned status {resp.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching Fear & Greed Index: {e}")

        return None

    def get_classification(self, value: int) -> str:
        """
        Get text classification for a given index value.

        Args:
            value: Fear & Greed Index value (0-100)

        Returns:
            str: Classification ('Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed')
        """
        if value <= 20:
            return "Extreme Fear"
        elif value <= 40:
            return "Fear"
        elif value <= 60:
            return "Neutral"
        elif value <= 80:
            return "Greed"
        else:
            return "Extreme Greed"

    def should_block_trade(self, value: Optional[int], min_value: int = 20, max_value: int = 80) -> bool:
        """
        Determine if trade should be blocked based on Fear & Greed Index.

        Args:
            value: Current F&G index (None = don't block if unavailable)
            min_value: Block if index < this (default 20 = extreme fear)
            max_value: Block if index > this (default 80 = extreme greed)

        Returns:
            bool: True if trade should be blocked, False otherwise
        """
        if value is None:
            # If API is down, don't block (fail-open)
            return False

        if value < min_value:
            logger.info(f"Trade blocked: Extreme Fear (F&G={value} < {min_value})")
            return True

        if value > max_value:
            logger.info(f"Trade blocked: Extreme Greed (F&G={value} > {max_value})")
            return True

        return False

    def clear_cache(self):
        """Clear cached data (useful for testing or manual refresh)."""
        self._cache = None
        self._cache_time = None
        logger.info("Fear & Greed cache cleared")
