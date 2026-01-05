from datetime import datetime

#import pandas as pd
import polars as pl
from common import readers
from common.config import logger
from common.utils import datetime_index, join_on_date
from core import Equity, ContinuousSignal

class Options:

    # Class-level dictionary to store instances of Option objects by ticker
    _instances = {}

    def __new__(cls, equity: Equity, data: dict = None):
        """
        Ensure only one instance per ticker.
        """
        ticker = equity.ticker.upper()

        # Check if the Option object for the ticker already exists
        if ticker in cls._instances:
            cls._instances[ticker].add_data(data)
            return cls._instances[ticker]
        else:
            # If no object exists, create a new one
            instance = super().__new__(cls)
            cls._instances[ticker] = instance
            return instance

    def __init__(self, underlying: Equity, data: dict = None, earliest_date = None):
        """
        Initialize an Option object linked to an Equity.
        :param underlying: The Equity object underlying the Option.
        """
        if hasattr(self, 'initialized') and self.initialized:
            return  # Prevent re-initialization if the object already exists

        self.underlying = underlying
        self.data = data if data else dict() #todo: feeling cute, might delete it later
        self.historical_data = None
        self.earliest_datetime = datetime.strptime(earliest_date, "%Y-%m-%d") if earliest_date else datetime.strptime("2020-01-01", "%Y-%m-%d")
        self._overview_pattern = '_options-overview-history-*.csv'

        self.get_historical_overview()

        self.initialized = True
        logger.debug(self.__repr__()+' created.')

    def get_historical_overview(self, start_date: str = None, end_date: str = None):
        """
        Retrieve the historical option data by reading multiple CSVs and joining them into a single DataFrame.
        The files should be in a directory and named according to the pattern:
        '{ticker}_options-overview-history-{time}.csv'.

        :param start_date: Start date for the historical data (e.g., "2020-01-01").
        :param end_date: End date for the historical data (e.g., "2022-12-31").
        :return: A polars DataFrame containing the historical option data.
        """
        start_date, end_date = self.underlying.time_window(start_date, end_date)

        # TODO: check whether the data is available within time window, for now assuming the data are the same
        if "overview" in self.data:
            df = self.data['overview']
            df = df.filter(
                (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
            )
            return df

        options_overview = readers.CSV.from_pattern_concatenate(f'options/{self.underlying.ticker.lower()}/{self.underlying.ticker.lower()}' + self._overview_pattern)
        options_overview = datetime_index(options_overview.sort("date"))
        options_overview = options_overview.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        self.data["overview"] = options_overview
        if options_overview.is_empty():
            logger.warning(f"No data found in the date range from {start_date} to {end_date}.")
        return self.data['overview']

    def add_data(self, data: dict):
        if data is not None:
            for key, value in data.items():
                self.data[key] = value

    @property
    def Chain(self):
        pass # This public version does not support the Chain class
        from core.Chain import Chain
        return Chain(self.underlying)

    def __getitem__(self, key) -> ContinuousSignal:
        results = []

        for label, df in self.data.items():
            if key.lower() in df.columns:
                results.append(df.select(["date", key.lower()]))

        if results:
            df = join_on_date(results)
            return ContinuousSignal(f"{self.underlying.ticker} {key}", signal=df)

        raise KeyError(f"Key '{key}' not found.")

    def __str__(self):
        return f"Options({self.underlying.ticker})"

    @classmethod
    def get_instance(cls, equity: Equity):
        """Retrieve the existing instance of an Option object by ticker."""
        ticker = equity.ticker.upper()
        return cls._instances.get(ticker, None)
