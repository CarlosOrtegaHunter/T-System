"""
Cache store abstraction for market data.

This module defines the interface for caching market data, allowing the system
to persist fetched data for faster subsequent access and deterministic backtesting.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Union
import polars as pl


class CacheStore(ABC):
    """
    Abstract base class for cache stores.
    
    Cache stores persist fetched market data to avoid repeated API calls and
    enable deterministic, reproducible backtests.
    """
    
    @abstractmethod
    def get(
        self,
        provider_name: str,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        adjusted: bool = True,
        schema_version: str = "1.0"
    ) -> Optional[pl.DataFrame]:
        """
        Retrieve cached data if available.
        
        :param provider_name: Name of the provider that originally fetched the data
        :param ticker: Stock ticker symbol
        :param start: Start date
        :param end: End date
        :param interval: Data interval
        :param adjusted: Whether data is adjusted
        :param schema_version: Schema version for cache key (allows cache invalidation on schema changes)
        :return: Cached DataFrame if available, None otherwise
        """
        pass
    
    @abstractmethod
    def put(
        self,
        provider_name: str,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        data: pl.DataFrame,
        interval: str = "1d",
        adjusted: bool = True,
        schema_version: str = "1.0"
    ) -> bool:
        """
        Store data in cache.
        
        :param provider_name: Name of the provider that fetched the data
        :param ticker: Stock ticker symbol
        :param start: Start date
        :param end: End date
        :param data: DataFrame to cache
        :param interval: Data interval
        :param adjusted: Whether data is adjusted
        :param schema_version: Schema version for cache key
        :return: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(
        self,
        ticker: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> bool:
        """
        Clear cached data.
        
        :param ticker: If provided, only clear cache for this ticker
        :param provider_name: If provided, only clear cache for this provider
        :return: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the cache store is available and configured.
        
        :return: True if cache can be used, False otherwise
        """
        pass


# TODO: Implement SQL-based cache store
# 
# class SQLCacheStore(CacheStore):
#     """
#     SQL-based cache store for market data.
#     
#     Stores cached data in a SQL database, allowing for efficient querying
#     and integration with existing database infrastructure.
#     
#     Can leverage existing OHLCVDatabase class (common/db_ohlcv.py) for storage.
#     Cache keys are stored as:
#     - provider_name: VARCHAR
#     - ticker: VARCHAR
#     - start_date: DATE
#     - end_date: DATE
#     - interval: VARCHAR
#     - adjusted: BOOLEAN
#     - schema_version: VARCHAR
#     
#     Data can be stored:
#     - In normalized table format using OHLCVDatabase.insert_ohlcv_optimized()
#     - Or as Parquet blobs in a separate cache table
#     
#     Implementation should:
#     1. Check cache before providers fetch (in ProviderChain or individual providers)
#     2. Store fetched data after successful provider fetch
#     3. Use OHLCVDatabase for efficient bulk inserts and queries
#     """
#     pass


# TODO: Implement Parquet-based cache store
# 
# class ParquetCacheStore(CacheStore):
#     """
#     Parquet-based cache store for market data.
#     
#     Stores cached data as Parquet files on disk, organized by:
#     cache/
#       {provider_name}/
#         {ticker}/
#           {start_date}_{end_date}_{interval}_{adjusted}_{schema_version}.parquet
#     
#     Uses Polars to read/write Parquet files efficiently.
#     """
#     pass


class NoOpCacheStore(CacheStore):
    """
    No-op cache store that doesn't actually cache anything.
    
    Useful for testing or when caching is not desired.
    """
    
    def get(
        self,
        provider_name: str,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        adjusted: bool = True,
        schema_version: str = "1.0"
    ) -> Optional[pl.DataFrame]:
        """No-op: always returns None."""
        return None
    
    def put(
        self,
        provider_name: str,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        data: pl.DataFrame,
        interval: str = "1d",
        adjusted: bool = True,
        schema_version: str = "1.0"
    ) -> bool:
        """No-op: always returns True."""
        return True
    
    def clear(
        self,
        ticker: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> bool:
        """No-op: always returns True."""
        return True
    
    def is_available(self) -> bool:
        """Always available (but does nothing)."""
        return True
