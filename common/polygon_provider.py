"""
Polygon.io data provider to replace yfinance functionality.
Supports both API calls and file downloads for historical OHLCV data.
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import polars as pl
from common.config import logger

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    logger.warning("polygon-api-client not installed. Install with: pip install polygon-api-client")


class PolygonProvider:
    """
    Polygon.io data provider for fetching historical stock data.
    Can use API calls or download files directly.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon.io provider.
        
        :param api_key: Polygon.io API key. If None, will try to get from:
                       1. Environment variable POLYGON_API_KEY
                       2. polygon_config.json file in project root
        """
        # Try to get API key from various sources
        if api_key:
            self.api_key = api_key
        else:
            # First try environment variable
            self.api_key = os.getenv("POLYGON_API_KEY")
            
            # If not found, try config file
            if not self.api_key:
                config_path = Path(__file__).parent.parent / "polygon_config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            self.api_key = config.get("api_key")
                            logger.info("Loaded Polygon.io API key from polygon_config.json")
                    except Exception as e:
                        logger.warning(f"Failed to load API key from config file: {e}")
        
        self.client = None
        
        if POLYGON_AVAILABLE and self.api_key:
            try:
                self.client = RESTClient(api_key=self.api_key)
                logger.info("Polygon.io client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon.io client: {e}")
                self.client = None
        elif not POLYGON_AVAILABLE:
            logger.warning("Polygon.io client library not available. Install with: pip install polygon-api-client")
        elif not self.api_key:
            logger.warning("Polygon.io API key not provided. Set POLYGON_API_KEY environment variable or pass api_key parameter.")
    
    def download(self, ticker: str, start: str = None, end: str = None, 
                 interval: str = "1d", adjusted: bool = True) -> pd.DataFrame:
        """
        Download historical OHLCV data for a ticker.
        Mimics yfinance.download() interface for easy replacement.
        
        :param ticker: Stock ticker symbol (e.g., "AAPL", "GME")
        :param start: Start date in format "YYYY-MM-DD" or datetime object
        :param end: End date in format "YYYY-MM-DD" or datetime object
        :param interval: Data interval - "1d" (daily), "1h" (hourly), etc.
        :param adjusted: Whether to return adjusted prices
        :return: pandas DataFrame with OHLCV data, matching yfinance format
        """
        if not self.client:
            logger.error("Polygon.io client not available. Cannot download data.")
            return self._empty_dataframe()
        
        try:
            # Convert dates to strings if needed (Polygon.io expects YYYY-MM-DD format)
            if isinstance(start, datetime):
                start_str = start.strftime("%Y-%m-%d")
            elif start is None:
                start_str = "2020-01-01"  # Default start date
            else:
                start_str = str(start)
            
            if isinstance(end, datetime):
                end_str = end.strftime("%Y-%m-%d")
            elif end is None:
                end_str = datetime.today().strftime("%Y-%m-%d")
            else:
                end_str = str(end)
            
            # Store original dates for filtering later
            start_date_for_filter = start
            end_date_for_filter = end
            
            # Map interval to Polygon.io timespan and multiplier
            timespan, multiplier = self._parse_interval(interval)
            
            logger.debug(f"Fetching {ticker} data from Polygon.io: {start_str} to {end_str}, interval={interval}")
            
            # Fetch aggregates from Polygon.io
            aggs = []
            cursor = None
            
            # Polygon.io has a limit of 50000 records per request, so we may need to paginate
            while True:
                params = {
                    "ticker": ticker,
                    "multiplier": multiplier,
                    "timespan": timespan,
                    "from_": start_str,
                    "to": end_str,
                    "adjusted": adjusted,
                    "sort": "asc",
                    "limit": 50000
                }
                
                if cursor:
                    params["cursor"] = cursor
                
                response = self.client.get_aggs(**params)
                
                # Handle response - Polygon.io returns results as a list or in response.results
                results = getattr(response, 'results', None)
                if results is None:
                    # If response is directly iterable (list), use it
                    if isinstance(response, list):
                        results = response
                    else:
                        # Try to get results from response attributes
                        results = []
                
                if not results:
                    break
                
                aggs.extend(results)
                
                # Check if there's more data (pagination)
                next_cursor = getattr(response, 'next_cursor', None)
                if not next_cursor:
                    break
                
                cursor = next_cursor
            
            if not aggs:
                logger.warning(f"No data returned from Polygon.io for {ticker}")
                return self._empty_dataframe()
            
            # Convert Aggregate objects to dictionary format
            # Polygon.io returns Aggregate objects with attributes: timestamp, open, high, low, close, volume, etc.
            records = []
            for agg in aggs:
                if hasattr(agg, '__dict__'):
                    # If it's an object, convert to dict
                    record = {
                        'timestamp': getattr(agg, 'timestamp', None),
                        'open': getattr(agg, 'open', None),
                        'high': getattr(agg, 'high', None),
                        'low': getattr(agg, 'low', None),
                        'close': getattr(agg, 'close', None),
                        'volume': getattr(agg, 'volume', None),
                        'vwap': getattr(agg, 'vwap', None),
                    }
                else:
                    # If it's already a dict, use it directly
                    record = agg
                records.append(record)
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Convert timestamp (milliseconds) to datetime
            if 'timestamp' in df.columns:
                # Convert from milliseconds to datetime (UTC), then normalize to date-only
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                # Convert UTC to naive datetime (date only, no time component)
                df['Date'] = df['timestamp'].dt.tz_localize(None).dt.normalize()
                df = df.set_index('Date')
                # Drop the original timestamp column
                if 'timestamp' in df.columns:
                    df = df.drop(columns=['timestamp'])
            else:
                logger.warning("No timestamp column found in Polygon.io response")
                return self._empty_dataframe()
            
            # Filter to requested date range (Polygon.io might return more data)
            if start_date_for_filter:
                if isinstance(start_date_for_filter, str):
                    start_date_obj = pd.to_datetime(start_date_for_filter).normalize()
                elif isinstance(start_date_for_filter, datetime):
                    start_date_obj = start_date_for_filter.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    start_date_obj = pd.to_datetime(start_date_for_filter).normalize()
                df = df[df.index >= start_date_obj]
            
            if end_date_for_filter:
                if isinstance(end_date_for_filter, str):
                    end_date_obj = pd.to_datetime(end_date_for_filter).normalize()
                elif isinstance(end_date_for_filter, datetime):
                    end_date_obj = end_date_for_filter.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    end_date_obj = pd.to_datetime(end_date_for_filter).normalize()
                df = df[df.index <= end_date_obj]
            
            logger.info(f"Successfully fetched {len(df)} records for {ticker} (filtered to {start_str} - {end_str})")
            
            # Rename columns to match yfinance format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'vwap': 'VWAP'  # Volume weighted average price
            })
            
            # Add adjusted close if adjusted=True (Polygon returns adjusted values in 'close' when adjusted=True)
            if adjusted:
                df['Adj Close'] = df['Close']
            else:
                df['Adj Close'] = df['Close']
            
            # Select columns in yfinance order
            columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df[[col for col in columns if col in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon.io for {ticker}: {e}")
            return self._empty_dataframe()
    
    def _parse_interval(self, interval: str) -> tuple:
        """
        Parse interval string to Polygon.io timespan and multiplier.
        
        :param interval: Interval string like "1d", "5d", "1h", "15m"
        :return: Tuple of (timespan, multiplier)
        """
        interval_map = {
            "1m": ("minute", 1),
            "5m": ("minute", 5),
            "15m": ("minute", 15),
            "30m": ("minute", 30),
            "1h": ("hour", 1),
            "4h": ("hour", 4),
            "1d": ("day", 1),
            "1w": ("week", 1),
            "1mo": ("month", 1),
        }
        
        # Handle numeric prefixes
        if interval not in interval_map:
            # Try to parse custom intervals
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                return ("minute", minutes)
            elif interval.endswith('h'):
                hours = int(interval[:-1])
                return ("hour", hours)
            elif interval.endswith('d'):
                days = int(interval[:-1])
                return ("day", days)
            else:
                logger.warning(f"Unknown interval {interval}, defaulting to 1d")
                return ("day", 1)
        
        return interval_map[interval]
    
    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame matching yfinance format."""
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])


# Global instance for convenience
_polygon_provider = None


def get_polygon_provider(api_key: Optional[str] = None) -> PolygonProvider:
    """
    Get or create a global Polygon.io provider instance.
    
    :param api_key: Optional API key. If not provided, uses existing instance or creates new one.
    :return: PolygonProvider instance
    """
    global _polygon_provider
    if _polygon_provider is None or api_key:
        _polygon_provider = PolygonProvider(api_key=api_key)
    return _polygon_provider


def download(ticker: str, start: str = None, end: str = None, 
             interval: str = "1d", adjusted: bool = True) -> pd.DataFrame:
    """
    Convenience function that mimics yfinance.download() interface.
    
    :param ticker: Stock ticker symbol
    :param start: Start date
    :param end: End date
    :param interval: Data interval
    :param adjusted: Whether to return adjusted prices
    :return: pandas DataFrame with OHLCV data
    """
    provider = get_polygon_provider()
    return provider.download(ticker, start, end, interval, adjusted)

