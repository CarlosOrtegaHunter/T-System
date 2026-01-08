from datetime import datetime
from typing import Union, Optional

from sortedcontainers import SortedList
import polars as pl

#todo: the debugger slows openbb imports WAY too much
#from openbb import obb
from common import readers
from common.config import logger
from common.utils import datetime_index
from common.provider_chain import ProviderChain, get_default_provider_chain
from common.providers import ProviderError
from core.Event import Event
from core.ContinuousSignal import ContinuousSignal

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

    def __init__(self, ticker: str, name: str = None, data: dict = None, earliest_date: str = None, provider: Optional[ProviderChain] = None):
        """
        Initialize an Equity object.

        :param ticker: Stock symbol (e.g., "GME").
        :param name: Full name of the equity.
        :param data: Dictionary where keys are names ("price", "volume") and values are ContinuousSignals.
        :param earliest_date: Earliest date for data loading.
        :param provider: ProviderChain to use for fetching data. If None, uses default chain (Parquet -> Polygon -> yfinance).
        """
        if hasattr(self, 'initialized') and self.initialized:
            return  # Prevent re-initialization

        self.ticker = ticker.upper()
        self.name = name if name else ticker
        self.data = data if data else {}
        self.events = SortedList(key=lambda event: event.date)
        self.continuousSignals = []
        self.earliest_datetime = datetime.strptime(earliest_date, "%Y-%m-%d") if earliest_date else datetime.strptime("2020-01-01", "%Y-%m-%d")
        
        # Use provided provider chain or default
        self.provider = provider if provider is not None else get_default_provider_chain()

        self.get_historical_volumes(earliest_date)
        self.get_historical_price(earliest_date)

        self.initialized = True
        logger.debug(self.__repr__()+' created.')

    def get_historical_price(self, start_date, end_date=None) -> pl.DataFrame:
        """
        Fetches historical daily price data using the configured provider chain.
        
        The provider chain tries providers in order until one succeeds:
        1. ParquetProvider (local CSV/Parquet files)
        2. PolygonProvider (Polygon.io API)
        3. YFinanceProvider (yfinance as fallback)
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
        
        # Fetch fresh data using provider chain
        try:
            data = self.provider.fetch_ohlcv(
                self.ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                adjusted=True
            )
            
            # Cache the data if we got results
            if not data.is_empty():
                if "price" not in self.data or self.data['price'].is_empty():
                    self.data['price'] = data
                return data
            else:
                # Return empty DataFrame with correct schema
                return pl.DataFrame(schema={
                    'date': pl.Datetime,
                    'open': pl.Float64,
                    'high': pl.Float64,
                    'low': pl.Float64,
                    'close': pl.Float64,
                    'volume': pl.Int64,
                    'adj_close': pl.Float64
                })
                
        except ProviderError as e:
            logger.error(f"Failed to fetch price data for {self.ticker}: {e}")
            # Return empty DataFrame with correct schema
            return pl.DataFrame(schema={
                'date': pl.Datetime,
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'volume': pl.Int64,
                'adj_close': pl.Float64
            })

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
