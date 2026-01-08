"""
Parquet/CSV file-based data provider implementation.

This provider loads historical data from local CSV or Parquet files.
It searches for files in the resources/equity/{ticker}/ directory.
"""
from datetime import datetime
from typing import Union, Optional
import polars as pl
from common import readers
from common.config import logger
from common.utils import datetime_index
from common.providers import MarketDataProvider, ProviderError


class ParquetProvider(MarketDataProvider):
    """
    Local file-based data provider for fetching historical stock data.
    
    Searches for CSV or Parquet files in resources/equity/{ticker}/ directory.
    Supports common file naming patterns like BATS_{TICKER}, 1D.csv or {ticker}_*.csv
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize Parquet provider.
        
        :param base_path: Base path for data files. If None, uses default resources/equity/ structure.
        """
        self.base_path = base_path
    
    @property
    def name(self) -> str:
        """Human-readable name of the provider."""
        return "Parquet/CSV"
    
    def is_available(self) -> bool:
        """Check if Parquet provider is available (always True, as it reads local files)."""
        return True
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        adjusted: bool = True
    ) -> pl.DataFrame:
        """
        Fetch historical OHLCV data for a ticker from local CSV/Parquet files.
        
        Returns Polars DataFrame with standardized columns: date, open, high, low, close, volume
        (and optionally 'adj_close' if adjusted=True).
        """
        try:
            ticker_lower = ticker.lower()
            
            # Look for equity data in equity/{ticker}/ folder
            # Try common patterns: BATS_{TICKER}, 1D.csv or {ticker}_*.csv
            pattern = f'equity/{ticker_lower}/*.csv'
            equity_files = readers.CSV.from_pattern(pattern)
            
            if not equity_files:
                logger.debug(f"No local CSV files found for {ticker} using pattern {pattern}")
                return pl.DataFrame(schema={
                    'date': pl.Datetime,
                    'open': pl.Float64,
                    'high': pl.Float64,
                    'low': pl.Float64,
                    'close': pl.Float64,
                    'volume': pl.Int64,
                    'adj_close': pl.Float64
                })
            
            # Concatenate all equity files for this ticker
            data = readers.CSV.from_pattern_concatenate(pattern)
            
            if data.is_empty():
                logger.debug(f"Empty data loaded from CSV files for {ticker}")
                return pl.DataFrame(schema={
                    'date': pl.Datetime,
                    'open': pl.Float64,
                    'high': pl.Float64,
                    'low': pl.Float64,
                    'close': pl.Float64,
                    'volume': pl.Int64,
                    'adj_close': pl.Float64
                })
            
            # Ensure date column exists and is datetime
            data = datetime_index(data)
            
            # Standardize column names (handle Volume vs volume, etc.)
            if 'volume' not in data.columns and 'Volume' in data.columns:
                data = data.rename({'Volume': 'volume'})
            
            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in data.columns]
                raise ProviderError(f"Missing required columns in data file: {missing}")
            
            # Convert start_date and end_date to datetime for comparison
            if isinstance(start, str):
                start_date_dt = datetime.strptime(start, "%Y-%m-%d")
            elif isinstance(start, datetime):
                start_date_dt = start
            else:
                start_date_dt = start
            
            if isinstance(end, str):
                end_date_dt = datetime.strptime(end, "%Y-%m-%d")
            elif isinstance(end, datetime):
                end_date_dt = end
            else:
                end_date_dt = end
            
            # Filter to requested date range
            filtered = data.filter(
                (pl.col("date") >= start_date_dt) & (pl.col("date") <= end_date_dt)
            )
            
            # Handle adjusted close
            if adjusted and 'adj_close' not in filtered.columns and 'close' in filtered.columns:
                # If no adj_close column, use close as adj_close
                filtered = filtered.with_columns(pl.col('close').alias('adj_close'))
            
            # Select and order columns
            expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if adjusted and 'adj_close' in filtered.columns:
                expected_cols.append('adj_close')
            
            # Only select columns that exist
            available_cols = [col for col in expected_cols if col in filtered.columns]
            filtered = filtered.select(available_cols)
            
            logger.info(f"Successfully loaded {len(filtered)} records for {ticker} from local CSV files")
            return filtered
            
        except Exception as e:
            logger.error(f"Error loading data from local files for {ticker}: {e}")
            raise ProviderError(f"Failed to load data from local files: {e}")
