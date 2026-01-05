from datetime import datetime
from typing import Union

from sortedcontainers import SortedList
import polars as pl

#todo: the debugger slows openbb imports WAY too much
#from openbb import obb
from common import readers
from common.config import logger
from common.utils import datetime_index
from common.polygon_provider import get_polygon_provider
from core.Event import Event
from core.ContinuousSignal import ContinuousSignal

# Fallback to yfinance if Polygon.io is not available
try:
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Using Polygon.io only.")

class Equity:
    _instances = {}

    # TODO: Implement extensive and intensive data.
    #  Choice to load options data dependent on the outstanding shares or float...

    def __new__(cls, ticker: str, name: str = None, data: dict = None, earliest_date: str = None):
        """
        Ensures only one instance per ticker.
        """
        ticker = ticker.upper()
        data = dict() if data is None else data
        if ticker in cls._instances:
            cls._instances[ticker].name = name if name else ticker
            cls._instances[ticker].add_data(data)
            return cls._instances[ticker]
        else:
            instance = super().__new__(cls)
            cls._instances[ticker] = instance
            return instance

    def __init__(self, ticker: str, name: str = None, data: dict = None, earliest_date: str = None):
        """
        Initialize an Equity object.

        :param ticker: Stock symbol (e.g., "GME").
        :param name: Full name of the equity.
        :param data: Dictionary where keys are names ("price", "volume") and values are ContinuousSignals.
        :param earliest_date: Earliest date for data loading.
        """
        if hasattr(self, 'initialized') and self.initialized:
            return  # Prevent re-initialization

        self.ticker = ticker.upper()
        self.name = name if name else ticker
        self.data = data if data else {}
        self.events = SortedList(key=lambda event: event.date)
        self.continuousSignals = []
        self.earliest_datetime = datetime.strptime(earliest_date, "%Y-%m-%d") if earliest_date else datetime.strptime("2020-01-01", "%Y-%m-%d")

        self.get_historical_volumes(earliest_date)
        self.get_historical_price(earliest_date)

        self.initialized = True
        logger.debug(self.__repr__()+' created.')

    def get_historical_price(self, start_date, end_date=None) -> pl.DataFrame:
        """
        Fetches historical daily price data using Polygon.io (with yfinance fallback).
        """
        start_date, end_date = self.time_window(start_date, end_date)
        
        # If we have cached data, check if it covers the requested range
        if "price" in self.data and not self.data['price'].is_empty():
            cached_data = self.data['price']
            # Check if cached data covers the requested range
            cached_start = cached_data['date'].min()
            cached_end = cached_data['date'].max()
            requested_start = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
            requested_end = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
            
            # Convert to comparable format
            if hasattr(cached_start, 'date'):
                cached_start = cached_start.date() if hasattr(cached_start, 'date') else cached_start
            if hasattr(cached_end, 'date'):
                cached_end = cached_end.date() if hasattr(cached_end, 'date') else cached_end
            if isinstance(requested_start, datetime):
                requested_start = requested_start.date()
            if isinstance(requested_end, datetime):
                requested_end = requested_end.date()
            
            # If cached data covers the requested range, filter and return it
            if cached_start <= requested_start and cached_end >= requested_end:
                filtered = cached_data.filter(
                    (pl.col("date") >= requested_start) & (pl.col("date") <= requested_end)
                )
                if not filtered.is_empty():
                    return filtered
        
        # Fetch fresh data for the requested range
        # First, try to load from local equity CSV files
        try:
            # Look for equity data in equity/{ticker}/ folder
            # Try common patterns: BATS_{TICKER}, 1D.csv or {ticker}_*.csv
            equity_files = readers.CSV.from_pattern(f'equity/{self.ticker.lower()}/*.csv')
            if equity_files:
                # Concatenate all equity files for this ticker
                data = readers.CSV.from_pattern_concatenate(f'equity/{self.ticker.lower()}/*.csv')
                if not data.is_empty():
                    # Ensure date column exists and is datetime
                    data = datetime_index(data)
                    # Standardize column names (handle Volume vs volume, etc.)
                    if 'volume' not in data.columns and 'Volume' in data.columns:
                        data = data.rename({'Volume': 'volume'})
                    # Ensure we have required columns
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        # Convert start_date and end_date to datetime for comparison
                        if isinstance(start_date, str):
                            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        elif isinstance(start_date, datetime):
                            start_date_dt = start_date
                        else:
                            start_date_dt = start_date
                        
                        if isinstance(end_date, str):
                            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        elif isinstance(end_date, datetime):
                            end_date_dt = end_date
                        else:
                            end_date_dt = end_date
                        
                        # Filter to requested date range (compare datetime to datetime)
                        filtered = data.filter(
                            (pl.col("date") >= start_date_dt) & (pl.col("date") <= end_date_dt)
                        )
                        if not filtered.is_empty():
                            # Cache the full dataset
                            if "price" not in self.data or self.data['price'].is_empty():
                                self.data['price'] = data
                            logger.info(f"Successfully loaded {self.ticker} data from local CSV files")
                            return filtered
        except Exception as e:
            logger.debug(f"No local equity CSV found for {self.ticker}: {e}")
        
        #data = obb.equity.price.historical(symbol=self.ticker, start_date=start_date, end_date=end_date)
        
        # Try Polygon.io first
        polygon_provider = get_polygon_provider()
        data = None
        
        if polygon_provider.client:
            try:
                logger.debug(f"Fetching {self.ticker} data from Polygon.io for range {start_date} to {end_date}")
                data = polygon_provider.download(self.ticker, start=start_date, end=end_date)
                if not data.empty:
                    data = data.reset_index()
                    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
                    # Rename 'date' column if it exists, otherwise use index
                    if 'date' not in data.columns and 'Date' in data.columns:
                        data = data.rename(columns={'Date': 'date'})
                    elif 'date' not in data.columns:
                        data = data.reset_index()
                        if 'timestamp' in data.columns:
                            data = data.rename(columns={'timestamp': 'date'})
                    # Cache the data
                    if "price" not in self.data or self.data['price'].is_empty():
                        self.data['price'] = pl.from_pandas(data)
                    logger.info(f"Successfully fetched {self.ticker} data from Polygon.io")
                    return pl.from_pandas(data)
            except Exception as e:
                logger.warning(f"Polygon.io fetch failed for {self.ticker}: {e}. Falling back to yfinance.")
            
            # Fallback to yfinance if Polygon.io fails or is not available
            if YFINANCE_AVAILABLE:
                try:
                    logger.debug(f"Fetching {self.ticker} data from yfinance")
                    data = yf.download(self.ticker, start=start_date, end=end_date)
                    data = data.reset_index()
                    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
                    if 'date' not in data.columns and 'Date' in data.columns:
                        data = data.rename(columns={'Date': 'date'})
                    self.data['price'] = pl.from_pandas(data)
                except YFRateLimitError:
                    self.data['price'] = pl.DataFrame({"date": [], "open": [], "high": [], "low": [], "close": [], "adj close": [], "volume": []})
                    logger.error("Yahoo Finance rate limit reached. Please try again later or use locally stored data.")
                except Exception as e:
                    logger.error(f"Yahoo Finance exception when fetching data: {e}")
                    self.data['price'] = pl.DataFrame({"date": [], "open": [], "high": [], "low": [], "close": [], "adj close": [], "volume": []})
            else:
                logger.error("Neither Polygon.io nor yfinance available. Cannot fetch price data.")
                self.data['price'] = pl.DataFrame({"date": [], "open": [], "high": [], "low": [], "close": [], "adj close": [], "volume": []})
            
            return self.data['price']

    def get_historical_volumes(self, start_date, end_date=None) -> pl.DataFrame:
        """
        TODO: might refactor DataFrame approach
        Fetches historical daily volume data.
        """
        if "volume" in self.data:
            data = self.data['volume']
        else:
            # Look for volume data in volume/{ticker}.volume_by_exchange.csv
            data = readers.CSV(f'volume/{self.ticker.lower()}.volume_by_exchange.csv')
            self.data['volume'] = datetime_index(data)

        start_date, end_date = self.time_window(start_date, end_date)

        return self.data['volume'].filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

    def time_window(self, start_date, end_date):
        """
        Converts start_date and end_date to datetime objects.
        """
        if start_date is None:
            start_date = self.earliest_datetime.strftime('%Y-%m-%d')
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        return start_date, end_date

    def attach(self, signal: Union[Event, ContinuousSignal, list]):
        """
        Attaches Event(s) or ContinuousSignal(s) to the Equity object.
        """
        if isinstance(signal, list):
            for s in signal:
                self.attach(s)
        if isinstance(signal, Event):
            self.events.add(signal)
        elif isinstance(signal, ContinuousSignal):
            signal.ticker = self #todo: improve attaching model
            self.continuousSignals.append(signal)

    def add_data(self, data: dict):
        """
        Merges new data into the existing dataset.
        """
        for key, value in data.items():
            self.data[key] = pl.from_pandas(value) if hasattr(value, "to_pandas") else value

    @property
    def Options(self):
        from core.Options import Options
        return Options(self)

    def get_event(self, label: str) -> Event:
        """
        Retrieves events by label.
        """
        return next((event for event in self.events if event.__label == label), None)

    def get_continuous_signal(self, label: str) -> ContinuousSignal:
        """
        Retrieves an attached ContinuousSignal by label.
        # TODO: Can this be optimized?
        """
        return next((signal for signal in self.continuousSignals if signal.__label == label), None)

    def __getitem__(self, key) -> pl.DataFrame:
        """
        Retrieves a dataframe with data.
        # TODO: safely retrieving a single ContinuousSignal instead of multiple columns
        """
        results = []

        # Collect from ContinuousSignals
        for signal in self.continuousSignals:
            if key.lower() in signal.df.columns:
                results.append(signal.df.select(["date", key.lower()]))

        for label, df in self.data.items():
            if key in df.columns:
                results.append(df.select(["date", key]))

        if results:
            df = pl.concat(results)
            return df

        raise KeyError(f"Key '{key}' not found.")

    def __str__(self):
        return f"Equity({self.ticker}, name={self.name})" if self.name != self.ticker else f"Equity({self.ticker})"

    @classmethod
    def get_instance(cls, ticker: str):
        """
        Retrieves an Equity instance by ticker.
        """
        return cls._instances.get(ticker.upper(), None)
