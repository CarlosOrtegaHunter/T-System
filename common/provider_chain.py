"""
Provider chain for trying multiple providers in sequence.

This allows the system to try multiple data sources in order until one succeeds.
"""
from typing import List, Optional, Union
from datetime import datetime
import polars as pl
from common.config import logger
from common.providers import MarketDataProvider, ProviderError


class ProviderChain:
    """
    Chains multiple providers together, trying each in sequence until one succeeds.
    
    This allows fallback behavior: try local files first, then Polygon.io, then yfinance.
    """
    
    def __init__(self, providers: List[MarketDataProvider]):
        """
        Initialize provider chain.
        
        :param providers: List of providers to try in order
        """
        self.providers = providers
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        adjusted: bool = True
    ) -> pl.DataFrame:
        """
        Try each provider in sequence until one succeeds.
        
        :param ticker: Stock ticker symbol
        :param start: Start date
        :param end: End date
        :param interval: Data interval
        :param adjusted: Whether to return adjusted prices
        :return: Polars DataFrame with OHLCV data
        :raises: ProviderError if all providers fail
        """
        last_error = None
        
        for provider in self.providers:
            if not provider.is_available():
                logger.debug(f"Provider {provider.name} is not available, skipping...")
                continue
            
            try:
                logger.debug(f"Trying provider {provider.name} for {ticker}...")
                result = provider.fetch_ohlcv(ticker, start, end, interval, adjusted)
                
                if not result.is_empty():
                    logger.info(f"Successfully fetched {ticker} data using {provider.name}")
                    return result
                else:
                    logger.debug(f"Provider {provider.name} returned empty data for {ticker}")
                    
            except ProviderError as e:
                logger.debug(f"Provider {provider.name} failed for {ticker}: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.warning(f"Unexpected error from provider {provider.name} for {ticker}: {e}")
                last_error = ProviderError(f"Unexpected error from {provider.name}: {e}")
                continue
        
        # All providers failed
        error_msg = f"All providers failed to fetch data for {ticker}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise ProviderError(error_msg)


def get_default_provider_chain() -> ProviderChain:
    """
    Get the default provider chain: ParquetProvider -> PolygonProvider -> YFinanceProvider.
    
    This tries local files first, then Polygon.io, then yfinance as fallback.
    """
    from common.parquet_provider import ParquetProvider
    from common.polygon_provider import get_polygon_provider
    from common.yfinance_provider import YFinanceProvider
    
    providers = [
        ParquetProvider(),  # Try local files first
        get_polygon_provider(),  # Then Polygon.io
        YFinanceProvider(),  # Finally yfinance as fallback
    ]
    
    return ProviderChain(providers)
