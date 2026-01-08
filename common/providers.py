"""
Market data provider abstraction.

This module defines the interface for market data providers, allowing the system
to switch between different data sources (Polygon.io, yfinance, local files, etc.)
without changing analysis or strategy code.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Union
import polars as pl
import pandas as pd


class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.
    
    All providers must implement the `fetch_ohlcv` method, which returns
    historical OHLCV (Open, High, Low, Close, Volume) data in a standardized format.
    """
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        adjusted: bool = True
    ) -> pl.DataFrame:
        """
        Fetch historical OHLCV data for a ticker.
        
        :param ticker: Stock ticker symbol (e.g., "AAPL", "GME")
        :param start: Start date in format "YYYY-MM-DD" or datetime object
        :param end: End date in format "YYYY-MM-DD" or datetime object
        :param interval: Data interval - "1d" (daily), "1h" (hourly), etc.
        :param adjusted: Whether to return adjusted prices
        :return: Polars DataFrame with columns: date, open, high, low, close, volume
                 (and optionally 'adj_close' if adjusted=True)
        :raises: ProviderError or appropriate exception if fetch fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and configured.
        
        :return: True if provider can be used, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the provider.
        
        :return: Provider name (e.g., "Polygon.io", "yfinance", "Parquet")
        """
        pass


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider is not available or not configured."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when a provider's rate limit is exceeded."""
    pass
