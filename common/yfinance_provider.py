"""
yfinance data provider implementation.

This provider uses yfinance as a fallback when Polygon.io is not available.
"""
from datetime import datetime
from typing import Union
import polars as pl
from common.config import logger
from common.providers import MarketDataProvider, ProviderError, ProviderRateLimitError

try:
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class YFinanceProvider(MarketDataProvider):
    """
    yfinance data provider for fetching historical stock data.
    
    This is typically used as a fallback when Polygon.io is not available.
    Note: yfinance has daily rate limits.
    """
    
    def __init__(self):
        """Initialize yfinance provider."""
        pass
    
    @property
    def name(self) -> str:
        """Human-readable name of the provider."""
        return "yfinance"
    
    def is_available(self) -> bool:
        """Check if yfinance provider is available."""
        return YFINANCE_AVAILABLE
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        adjusted: bool = True
    ) -> pl.DataFrame:
        """
        Fetch historical OHLCV data for a ticker using yfinance.
        
        Returns Polars DataFrame with standardized columns: date, open, high, low, close, volume
        (and optionally 'adj_close' if adjusted=True).
        """
        if not self.is_available():
            raise ProviderError("yfinance provider is not available. Install with: pip install yfinance")
        
        try:
            # Convert datetime to string if needed
            if isinstance(start, datetime):
                start_str = start.strftime("%Y-%m-%d")
            else:
                start_str = str(start) if start else None
            
            if isinstance(end, datetime):
                end_str = end.strftime("%Y-%m-%d")
            else:
                end_str = str(end) if end else None
            
            logger.debug(f"Fetching {ticker} data from yfinance for range {start_str} to {end_str}")
            
            # Download data using yfinance
            df = yf.download(ticker, start=start_str, end=end_str, interval=interval, auto_adjust=adjusted)
            
            if df.empty:
                logger.warning(f"No data returned from yfinance for {ticker}")
                return pl.DataFrame(schema={
                    'date': pl.Datetime,
                    'open': pl.Float64,
                    'high': pl.Float64,
                    'low': pl.Float64,
                    'close': pl.Float64,
                    'volume': pl.Int64,
                    'adj_close': pl.Float64
                })
            
            # Reset index to get Date as a column
            df = df.reset_index()
            
            # Standardize column names to lowercase
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() if isinstance(col, tuple) else col 
                         for col in df.columns]
            
            # Rename Date to date if needed
            if 'date' not in df.columns and 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            
            # Convert to Polars
            df_pl = pl.from_pandas(df)
            
            # Ensure date column is datetime
            if 'date' in df_pl.columns:
                df_pl = df_pl.with_columns(pl.col('date').cast(pl.Datetime))
            
            # Handle adjusted close
            if adjusted and 'adj close' in df_pl.columns:
                df_pl = df_pl.rename({'adj close': 'adj_close'})
            elif adjusted and 'adj_close' not in df_pl.columns and 'close' in df_pl.columns:
                # If no adj close column, use close as adj_close
                df_pl = df_pl.with_columns(pl.col('close').alias('adj_close'))
            
            # Select and order columns
            expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if adjusted and 'adj_close' in df_pl.columns:
                expected_cols.append('adj_close')
            
            # Only select columns that exist
            available_cols = [col for col in expected_cols if col in df_pl.columns]
            df_pl = df_pl.select(available_cols)
            
            logger.info(f"Successfully fetched {len(df_pl)} records for {ticker} from yfinance")
            return df_pl
            
        except YFRateLimitError:
            logger.error("Yahoo Finance rate limit reached. Please try again later or use locally stored data.")
            raise ProviderRateLimitError("Yahoo Finance rate limit reached. Please try again later.")
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {ticker}: {e}")
            raise ProviderError(f"Failed to fetch data from yfinance: {e}")
