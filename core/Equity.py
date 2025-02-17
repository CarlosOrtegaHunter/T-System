from datetime import datetime
from typing import Union

from sortedcontainers import SortedList
import polars as pl
import yfinance as yf
#todo: the debugger slows openbb imports WAY too much
#from openbb import obb
from common import readers
from common.config import logger
from common.utils import datetime_index
from core.Event import Event
from core.ContinuousSignal import ContinuousSignal

class Equity:
    _instances = {}

    def __new__(cls, ticker: str, name: str, data: dict = None, earliest_date: str = None):
        """
        Ensures only one instance per ticker.
        """
        ticker = ticker.upper()

        if ticker in cls._instances:
            cls._instances[ticker].name = name
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
        Fetches historical daily price data.
        """
        #todo: refactor
        if start_date is None:
            start_date = self.earliest_datetime.strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        #data = obb.equity.price.historical(symbol=self.ticker, start_date=start_date, end_date=end_date)
        data = yf.download(self.ticker, start=start_date, end=end_date)
        data = data.reset_index()
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        self.data['price'] = pl.from_pandas(data)
        return self.data['price']

    def get_historical_volumes(self, start_date, end_date=None) -> pl.DataFrame:
        """
        TODO: might refactor DataFrame approach
        Fetches historical daily volume data.
        """
        if "volume" in self.data:
            data = self.data['volume']
        else:
            data = readers.CSV(f'nyse-{self.ticker.lower()}.volume_by_exchange.csv')
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
