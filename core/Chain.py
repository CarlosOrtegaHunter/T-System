import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import RealDictCursor

from common.config import logger
from core import Options
from core import Equity

#import Options
#import Equity

class Chain:
    def __init__(self, equity: Equity, db_params: dict = None):
        """
        Initialize the Chain object for the given equity.
        :param equity: The Equity object corresponding to the option chain.
        :param db_params: A dictionary containing database connection parameters (host, dbname, user, password, port).
        """
        self.equity = equity
        self.db_params = db_params
        self.connection = None
        self.chain_data = self._download_chain()
        logger.debug(self.__repr__()+' created.')

    def _download_chain(self):
        """
        Fetches the options chain data for the underlying equity from Yahoo Finance.

        :return: A dictionary containing data for calls and puts, as DataFrames.
        """

        # TODO: I should create the yf ticket in the Equity class... have one object per ticker.
        #   i might need something similar for options.

        # Fetch options data using yfinance
        ticker = yf.Ticker(self.equity.ticker)

        # Get expiration dates (to ensure some data exists)
        try:
            expiration_dates = ticker.options
            if not expiration_dates:
                logger.warning(f"No options data available for {self.equity.ticker}.")
                return None
        except Exception as e:
            logger.exception(f"Failed to fetch options data: {e}")
            return None

        # Fetch options data for the first expiration date
        first_expiry = expiration_dates[0]
        try:
            calls = ticker.option_chain(first_expiry).calls
            puts = ticker.option_chain(first_expiry).puts
        except Exception as e:
            logger.exception(f"Failed to fetch options chain for {self.equity.ticker}: {e}")
            return None

        # Combine data into a dictionary
        chain = {
            "CALL": calls,
            "PUT": puts
        }

        return chain

    def get_calls(self):
        """
        Retrieve the calls data from the options chain.

        :return: A DataFrame containing call options data.
        """
        return self.chain_data.get("calls") if self.chain_data else None

    def get_puts(self):
        """
        Retrieve the puts data from the options chain.

        :return: A DataFrame containing put options data.
        """
        return self.chain_data.get("puts") if self.chain_data else None

    def __str__(self):
        return f"Options Chain for {self.equity.ticker}"

    def _connect_to_db(self):
        """
        Establish a connection to the TimescaleDB/Postgres database.
        """
        if not self.connection or self.connection.closed != 0:
            try:
                self.connection = psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor)
            except Exception as e:
                raise ConnectionError(f"Failed to connect to database: {e}")

    def get_historical_chain(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        pass
        """
        Retrieve historical option chain data for the given equity between start_date and end_date.

        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format (optional).
        :return: A pandas DataFrame containing the historical option chain data.
        """
        self._connect_to_db()

        # If end_date is not provided, use today's date
        end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')

        query = f"""
            SELECT * 
            FROM options_chain 
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date DESC;
        """

        params = (self.equity.ticker.upper(), start_date, end_date)

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                if not rows:
                    logger.warning(f"No data found for {self.equity.ticker} between {start_date} and {end_date}.")
                    return pd.DataFrame()
                else:
                    return pd.DataFrame(rows)

        except Exception as e:
            raise RuntimeError(f"Failed to execute query: {e}")

    def get_option(self, expiration_date: str, strike_price: float, option_type):
        pass
        """
        Retrieve a specific option from the chain based on expiration date, strike price, and option type.

        :param expiration_date: The expiration date of the option (YYYY-MM-DD).
        :param strike_price: The strike price of the option.
        :param option_type: The type of the option (CALL or PUT).
        :return: A pandas Series representing the selected option data.
        """
        self._connect_to_db()

        query = f"""
            SELECT * 
            FROM options_chain 
            WHERE ticker = %s AND expiration_date = %s AND strike_price = %s AND option_type = %s;
        """
        params = (self.equity.ticker.upper(), expiration_date, strike_price, option_type.value)

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"No option found for {self.equity.ticker} with expiration {expiration_date}, "
                          f"strike {strike_price}, and type {option_type}.")
                    return None
                else:
                    return pd.Series(row)

        except Exception as e:
            raise RuntimeError(f"Failed to execute query: {e}")

    def __del__(self):
        """
        Ensure the database connection is closed when the object is deleted.
        """
        if self.connection:
            self.connection.close()
