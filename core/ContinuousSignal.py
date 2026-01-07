import numbers
from datetime import timedelta, datetime
from types import MappingProxyType

import numpy as np
import polars as pl
import pandas as pd
from common import utils
from common.config import logger
from common.utils import string_index, datetime_index
from core import Equity

_TEMP_OBJECT_LABEL = 'temporaryContinuousSignalObject'
_TIME_SYMBOL = 'date'
_DATA_SYMBOL = 'data'
#TODO: might the _TEMP_OBJECT_LABEL approach for a parameter based one

class ContinuousSignal:
    """
    ContinuousSignal is a high-performance wrapper around Polars LazyFrame for efficient time-series data handling.
    It provides seamless integration with pandas, supports lazy evaluation, and enables optimized slicing, filtering,
    and arithmetic operations on datetime-indexed financial or signal data.
    Key Features:
    - Efficient lazy evaluation with Polars LazyFrame.
    - Supports arithmetic and logical operations while preserving data integrity.
    - Seamless compatibility with pandas DataFrame and Series.
    - Optimized for financial and trading applications.
    - Temporary object handling for intermediate computations.
    """
    _SIGNALS_BY_TICKER = {}  # todo: implement deletion of signals
    __slots__ = ("__signal", "__label", "__ticker", "__metadata", "__is_temporary", "__data_col_name", "metadata", "keep_when_dropping_duplicates")

    def __init__(self, label: str, signal = None, ticker: Equity = None, metadata: dict = None, keep: str = None, by_ref = False):
        """
        Initialize a ContinuousSignal object.
        The label _TEMP_OBJECT_LABEL ('temporaryContinuousSignalObject') is reserved for proxy objects that might be discarded during an operation.
        :param label: The label for the signal (e.g., sentiment analysis).
        :param signal: A Polars DataFrame or Pandas DataFrame with a _TIME_SYMBOL ('date') column and a _DATA_SYMBOL ('data') column.
        :param ticker: The Equity object associated with the signal.
        :param metadata: Optional dictionary for additional event details.
        :param keep: 'all' (to keep duplicates), 'first' or 'last' (when dropping duplicate dates). None defaults to 'last'.
        :param by_ref: Create object copying data by reference instead of by value.
        """
        self.__label = label
        self.__ticker = None
        self.__is_temporary = False
        self.__data_col_name = _DATA_SYMBOL
        self.metadata = None
        self.keep_when_dropping_duplicates = None

        # I bring thou the rvalue reference equivalent
        if self.__label == _TEMP_OBJECT_LABEL:
            self.__is_temporary = True
            self.__signal = signal
            self.__ticker = ticker
            self.metadata = metadata
            self.keep_when_dropping_duplicates = keep
            return # IN THE NAME OF PERFORMANCE, I HEREBY DECLARE TEMPORARY OBJECTS CAN CHEAT CHECKS

        if isinstance(signal, ContinuousSignal):
            if by_ref:
                if signal.__signal: self.__signal = signal.__signal
                if signal.__ticker: self.__ticker = signal.__ticker
                if signal.metadata: self.metadata = signal.metadata
                if signal.keep_when_dropping_duplicates: self.keep_when_dropping_duplicates = signal.keep_when_dropping_duplicates
            else:
                self.__signal = signal.__signal.clone()
                self.__ticker = signal.__ticker
                self.metadata = signal.metadata.copy() if signal.metadata else None
                self.keep_when_dropping_duplicates = signal.keep_when_dropping_duplicates
        else:
            self.__signal = self._signal_to_lazyframe(signal)

        if metadata:
            self.metadata = metadata
        if ticker:
            self.__ticker = ticker
        if keep:
            self.keep_when_dropping_duplicates = keep
        else:
            self.keep_when_dropping_duplicates = 'last'

        if not isinstance(signal, ContinuousSignal):
            self._validate_signal_dataframe()
            self._validate_no_conflicting_duplicates()

        if self.__ticker:
            self.register_signal(self)

        logger.debug(self.__repr__() + ' created.')

    @staticmethod
    def _normalize_to_lazyframe(data, allow_pandas=True):
        """
        Universal converter: Normalizes pandas/polars inputs to polars LazyFrame.
        Handles: pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray
        
        :param data: Input data in any supported format
        :param allow_pandas: Whether to allow pandas inputs (for universal compatibility)
        :return: polars LazyFrame
        """
        if data is None:
            return pl.DataFrame(schema={_TIME_SYMBOL: pl.Datetime, _DATA_SYMBOL: pl.Float64}).lazy()
        
        if isinstance(data, pl.LazyFrame):
            return data
        elif isinstance(data, pl.DataFrame):
            return data.lazy()
        elif isinstance(data, pl.Series):
            return data.to_frame(name=_DATA_SYMBOL).lazy()
        elif allow_pandas:
            if isinstance(data, pd.DataFrame):
                return pl.from_pandas(data).lazy()
            elif isinstance(data, pd.Series):
                return pl.from_pandas(data.to_frame(name=_DATA_SYMBOL)).lazy()
        elif hasattr(data, "to_pandas"):
            # Generic pandas-like object
            return pl.from_pandas(data).lazy()
        elif isinstance(data, np.ndarray):
            return pl.DataFrame({_DATA_SYMBOL: data.flatten()}).lazy()
        else:
            raise TypeError(f"Cannot convert {type(data)} to polars LazyFrame. "
                          f"Supported types: pandas/polars DataFrame/Series, numpy arrays")

    @staticmethod
    def _infer_shift_map_intent(lazy_frame, time_symbol=_TIME_SYMBOL, data_symbol=_DATA_SYMBOL):
        """
        Infers intent from schema: determines if this is a shift (numeric) or map (datetime) operation,
        and which column to use. Universal for pandas/polars.
        
        Returns: (column_name, column_expr, is_mapping, operation_type)
        - column_name: string name of the column to use (for joins)
        - column_expr: polars expression for the column to use
        - is_mapping: True if this is a date-to-date mapping (requires join)
        - operation_type: 'shift' (numeric), 'map_dates' (datetime mapping), 'map_values' (value mapping)
        """
        schema = lazy_frame.collect_schema()
        schema_keys = set(schema.keys())
        schema_types = {k: schema[k] for k in schema_keys}
        
        # Case 1: Single column - infer from type
        if len(schema_keys) == 1:
            col_name = schema_keys.pop()
            col_type = schema_types[col_name]
            
            if col_type in (pl.Date, pl.Datetime):
                # Single datetime column = date mapping
                return (col_name, pl.col(col_name), True, 'map_dates')
            elif col_type in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, int, float):
                # Single numeric column = shift values
                return (col_name, pl.col(col_name), False, 'shift')
            else:
                # Try to use as data column
                return (col_name, pl.col(col_name), False, 'shift')
        
        # Case 2: Has both TIME_SYMBOL and DATA_SYMBOL
        elif time_symbol in schema_keys and data_symbol in schema_keys:
            data_type = schema_types[data_symbol]
            if data_type in (pl.Date, pl.Datetime):
                # DATA_SYMBOL is datetime = date mapping (use DATA_SYMBOL as new dates)
                return (data_symbol, pl.col(data_symbol), True, 'map_dates')
            else:
                # DATA_SYMBOL is numeric = value mapping (join on dates, use data values)
                return (data_symbol, pl.col(data_symbol), True, 'map_values')
        
        # Case 3: Only TIME_SYMBOL = date mapping
        elif time_symbol in schema_keys:
            return (time_symbol, pl.col(time_symbol), True, 'map_dates')
        
        # Case 4: Only DATA_SYMBOL = value shift/mapping
        elif data_symbol in schema_keys:
            data_type = schema_types[data_symbol]
            if data_type in (pl.Date, pl.Datetime):
                # DATA_SYMBOL is datetime but no TIME_SYMBOL = ambiguous, treat as date mapping
                return (data_symbol, pl.col(data_symbol), True, 'map_dates')
            else:
                # DATA_SYMBOL is numeric = sequential shift
                return (data_symbol, pl.col(data_symbol), False, 'shift')
        
        # Case 5: Ambiguous - multiple columns but no recognized symbols
        else:
            # Try to find a datetime column
            datetime_cols = [k for k, v in schema_types.items() if v in (pl.Date, pl.Datetime)]
            if datetime_cols:
                col_name = datetime_cols[0]
                return (col_name, pl.col(col_name), True, 'map_dates')
            # Otherwise use first column
            first_col = list(schema_keys)[0]
            return (first_col, pl.col(first_col), False, 'shift')

    def _signal_to_lazyframe(self, signal):
        if signal is None:
            return pl.DataFrame(schema={_TIME_SYMBOL: pl.Datetime, _DATA_SYMBOL: pl.Float64}).lazy()
        elif isinstance(signal, pl.LazyFrame):
            return signal
        elif isinstance(signal, pl.Series):
            # Series should be allowed for date shifting operations!
            # Handle Series by converting to DataFrame with _DATA_SYMBOL column
            # Note: Series doesn't have a date column, so it will be used for sequential operations
            # For left join operations, the Series values will be aligned by index/position
            if self.label != _TEMP_OBJECT_LABEL:
                logger.error("Only temporary ContinuousSignal objects are allowed to wrap a Series signal. ")
                raise ValueError("Only temporary ContinuousSignal objects are allowed to wrap a Series signal. ")
            # Convert Series to LazyFrame with _DATA_SYMBOL column
            # The Series will be used for operations where length matching is handled by the operation itself
            return signal.to_frame(name=_DATA_SYMBOL).lazy()
        elif isinstance(signal, pl.DataFrame):
            # dimensions are checked in _validate_signal_dataframe
            return signal.lazy()
        elif hasattr(signal, "to_pandas"):
            return pl.from_pandas(signal).lazy()
        else:
            raise TypeError("Signal must be a Polars DataFrame or a Pandas DataFrame.")

    @classmethod
    def register_signal(cls, csignal):
        """Keeps dictionary of signals by ticker."""
        from core import Equity
        if not isinstance(csignal.__ticker, Equity):
            logger.error("Ticker must be of Equity type.")
            raise ValueError("Ticker must be of Equity type.")
        ticker = csignal.__ticker.ticker
        if ticker not in cls._SIGNALS_BY_TICKER:
            cls._SIGNALS_BY_TICKER[ticker] = []
        cls._SIGNALS_BY_TICKER[ticker].append(csignal)

    def _validate_signal_dataframe(self):
        """
        Ensures that the DataFrame has a valid datetime column called _DATE_SYMBOL and one data column called _DATA_SYMBOL.
        """
        if _TIME_SYMBOL not in self.__signal.collect_schema().names():
            raise ValueError(f"Signal DataFrame must contain a '{_TIME_SYMBOL}' column.")

        self.__signal = self.__signal.select([_TIME_SYMBOL] + [col for col in self.__signal.collect_schema().keys() if col != _TIME_SYMBOL])
        try:
            self.__signal = datetime_index(self.__signal)
        except (ValueError, TypeError, pl.exceptions.ComputeError) as e:
            logger.error(f"Failed to convert '{_TIME_SYMBOL}' column to datetime: {e}")
            raise ValueError(f"Failed to convert '{_TIME_SYMBOL}' column to datetime. "
                           f"Please ensure the '{_TIME_SYMBOL}' column contains valid datetime values. "
                           f"Original error: {e}") from e

        column_n = len(self.__signal.collect_schema().names())
        if column_n != 2:
            raise ValueError(f"Signal DataFrame must contain exactly two columns, found {column_n}.")
        self.__data_col_name = [col for col in self.__signal.collect_schema().names() if col != _TIME_SYMBOL][0]
        self.__signal = self.__signal.rename({self.__data_col_name: _DATA_SYMBOL})
        self.__signal = utils.lowercase_columns(self.__signal)

    def _validate_no_conflicting_duplicates(self):
        """
        Ensures that all rows with the same date have identical values.
        """
        logger.debug("Collecting LazyFrames... ("+self.label+") @_validate_no_conflicting_duplicates")
        try:
            self.__signal = string_index(self.__signal).collect()
        except (ValueError, pl.exceptions.ComputeError) as e:
            # If collection fails, it might be due to invalid datetime conversion
            # Re-raise as ValueError with context
            if isinstance(e, pl.exceptions.ComputeError):
                raise ValueError(f"Failed to process date column during duplicate validation. "
                               f"Ensure date column contains valid datetime values. Original error: {e}") from e
            raise
        # Check for duplicates: compare total rows with unique date count
        # pl.len() and n_unique() return DataFrames when used in select, so collect first
        if isinstance(self.__signal, pl.LazyFrame):
            total_rows = self.__signal.select(pl.len()).collect().item(0, 0)
            unique_dates = self.__signal.select(pl.col(_TIME_SYMBOL).n_unique()).collect().item(0, 0)
        else:
            # Already a DataFrame
            total_rows = len(self.__signal)
            unique_dates = self.__signal.select(pl.col(_TIME_SYMBOL).n_unique()).item(0, 0)
        has_duplicates = total_rows > unique_dates

        if has_duplicates:
            # Check if there are conflicting values (same date, different data)
            # Group by date and data, then check if any date appears with multiple data values
            if isinstance(self.__signal, pl.LazyFrame):
                grouped = self.__signal.group_by(_TIME_SYMBOL).agg(pl.col(_DATA_SYMBOL).n_unique().alias("data_unique")).collect()
            else:
                grouped = self.__signal.group_by(_TIME_SYMBOL).agg(pl.col(_DATA_SYMBOL).n_unique().alias("data_unique"))
            conflicts_number = (grouped["data_unique"] > 1).any()
            if conflicts_number:
                if self.keep_when_dropping_duplicates is None:
                    self.keep_when_dropping_duplicates = "last"
                if self.keep_when_dropping_duplicates in ["first", "last"]:
                    logger.warning(
                        f"Conflicting data found for {self.__label}. Using keep='{self.keep_when_dropping_duplicates}'."
                    )
                else:
                    logger.error("Conflicting data found. Hint: use SuperpositionSignal instead.")
                    raise ValueError("Conflicting data found. Hint: use SuperpositionSignal instead.")
            self.__signal = self.__signal.unique(
                subset=_TIME_SYMBOL,
                keep="first" if self.keep_when_dropping_duplicates == "first" else "last"
            ).lazy()
        self.__signal = datetime_index(self.__signal)
        self.__signal = self.__signal.sort(_TIME_SYMBOL).lazy()

    def __getitem__(self, key):
        """
        Mimics Pandas-like selection. Returns a new ContinuousSignal object:
        - `signal["value"]` -> Returns a temporary ContinuousSignal with the selected column.
        - `signal[start:end]` -> Filters by date range.
        - `signal[mask]` -> Boolean filtering.
        """
        logger.debug("__getitem__ "+self.__label)

        if isinstance(key, str):
            # Check if it's a date string (e.g., "2020-01-01", "2020-01-01 12:00:00")
            try:
                # Try to parse as date/datetime string
                date_key = pd.to_datetime(key)
                # If successful, treat as date lookup - keep lazy until we need the scalar
                result_lazy = self.__signal.filter(pl.col(_TIME_SYMBOL) == pl.lit(date_key)).select(_DATA_SYMBOL)
                # Only materialize once to get both count and value
                result_df = result_lazy.collect()
                if result_df.height == 0:
                    raise KeyError(f"No data found for date: {key}")
                return result_df.item(0, 0)
            except (ValueError, TypeError):
                # Not a date string, treat as column name - keep lazy
                if key.lower() == self.__data_col_name:
                    key = _DATA_SYMBOL
                # Return lazy ContinuousSignal - no materialization
                return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal.select(pl.col(key.lower())), by_ref=True)

        elif isinstance(key, slice):
            start, end = key.start, key.stop
            # Only materialize min/max if needed (None bounds) - necessary for slice bounds
            if start is None:
                start = self.__signal.select(pl.col(_TIME_SYMBOL).min()).collect().item(0, 0)
            if end is None:
                end = self.__signal.select(pl.col(_TIME_SYMBOL).max()).collect().item(0, 0)
            # Return lazy - no materialization of the filtered result
            # Convert string dates to datetime and use literals
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)
            return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal.filter(pl.col(_TIME_SYMBOL).is_between(pl.lit(start), pl.lit(end))), by_ref=True)

        elif isinstance(key, pl.Series):
            return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal.filter(key), by_ref=True)

        elif isinstance(key, int):
            # For int indexing, we need height and the value - materialize only what's necessary
            # Could cache height, but for now materialize once for the value
            height = self.__signal.select(pl.len()).collect().item(0, 0)
            # Materialize only the single row we need
            idx = key % height if key >= 0 else (key % height + height) % height
            return self.__signal.select(pl.col(_DATA_SYMBOL)).slice(idx, 1).collect().item(0, 0)

        elif isinstance(key, (pl.Datetime, datetime)):
            # Handle both polars Datetime and Python datetime objects
            # Keep filter lazy until we need the scalar value
            date_lit = pl.lit(key)
            result_lazy = self.__signal.filter(pl.col(_TIME_SYMBOL) == date_lit).select(_DATA_SYMBOL)
            # Only materialize once to get both count and value
            result_df = result_lazy.collect()
            if result_df.height == 0:
                raise KeyError(f"No data found for timestamp: {key}")
            return result_df.item(0, 0)

        else:
            raise TypeError(f"Invalid selection type: {type(key)}.")

    def __setitem__(self, key, value):
        """
        Allows setting values like Pandas.
        Takes a column-like or 2-column dataframe-like value.
        Performs a left join if value is a two-column dataframe-like with a datetime index.
        Otherwise, it copies values sequentially up to the available rows.
        - `signal["column"] = new_data` -> Updates a column.
        - `signal[start:end] = new_data` -> Updates rows by date range.
        - `signal[mask] = new_data` -> Updates rows where the mask is True.
        """
        logger.debug("__setitem__ "+self.__label)

        _tempContinuousSignalFlag = False
        _tempCS = None

        if self.__is_temporary:
            msg = "Temporary objects resulting from ContinuousSignal operations must be settled by calling settle."
            logger.error(msg)
            raise PermissionError(msg)
        # Unwrap dataframe from ContinuousSignal
        if isinstance(value, ContinuousSignal):
            if value.label == _TEMP_OBJECT_LABEL:
                _tempCS = value
                _tempContinuousSignalFlag = True
                value = value.__signal
            else:
                logger.error(f"Cannot set ContinuousSignal data equal to unaccessed ContinuousSignal object.")
                raise TypeError("Cannot set ContinuousSignal data equal to unaccessed ContinuousSignal object.")

        # Check value type, convert to pl.LazyFrame using universal converter
        # This handles pandas/polars seamlessly
        if not isinstance(value, (int, float, pl.LazyFrame, ContinuousSignal)):
            value = self._normalize_to_lazyframe(value, allow_pandas=True)

        if not isinstance(value, pl.LazyFrame | pl.Datetime | int | float):
            logger.error(f"Unsupported type for value: {type(value)}")
            raise TypeError(f"Unsupported type for value: {type(value)}")

        # Create mask of dates on which to operate
        mask = pl.lit(True)

        if isinstance(key, str):
            # Check if string is a date/datetime string, not a column name
            try:
                # Try to parse as date/datetime string
                date_key = pd.to_datetime(key)
                # If successful, treat as date range (single date = point in time)
                # For setitem with a single date, we'll use it as both start and end
                mask = pl.col(_TIME_SYMBOL) == pl.lit(date_key)
            except (ValueError, TypeError):
                # Not a date string, treat as column name
                if key.lower() != self.__data_col_name and key.lower() != _DATA_SYMBOL:
                    raise IndexError(f"No such column as '{key.lower()}'. Only columns are '{_DATA_SYMBOL}' and '{self.__data_col_name}' (aka '{_DATA_SYMBOL}'). ")
                # mask remains everywhere true
                mask = pl.lit(True)

        elif isinstance(key, slice):
            start, end = key.start, key.stop
            if start is None:
                start = self.__signal.select(pl.col(_TIME_SYMBOL).min()).collect().item(0, 0)
            else:
                # Convert string dates to datetime objects
                if isinstance(start, str):
                    start = pd.to_datetime(start)
            if end is None:
                end = self.__signal.select(pl.col(_TIME_SYMBOL).max()).collect().item(0, 0)
            else:
                # Convert string dates to datetime objects
                if isinstance(end, str):
                    end = pd.to_datetime(end)

            # Use literals for date comparison to avoid column name interpretation
            mask = pl.col(_TIME_SYMBOL).is_between(pl.lit(start), pl.lit(end))

        elif isinstance(key, pl.Series):
            # Validate boolean Series mask
            if key.dtype != pl.Boolean:
                raise TypeError(f"Series mask must be boolean type, got {key.dtype}")
            # Validate mask length matches signal length
            signal_len = self.__signal.select(pl.len()).collect().item(0, 0)
            key_len = len(key)
            if key_len != signal_len:
                raise ValueError(f"Boolean mask length ({key_len}) must match signal length ({signal_len})")
            mask = key

        elif isinstance(key, pl.Datetime):
            raise NotImplementedError("Datetime indexing not implemented")

        else:
            logger.exception(f"Invalid key type: {type(key)}.")
            raise TypeError(f"Invalid key type: {type(key)}.")

        # Perform data assignment

        if isinstance(value, (int, float)):
            # Lazy operation - no materialization
            self.__signal = self.__signal.with_columns(
                pl.when(mask).then(value).otherwise(pl.col(_DATA_SYMBOL)).alias(_DATA_SYMBOL)
            )
            # Return early - int/float assignment is complete
            if _tempContinuousSignalFlag:
                logger.debug("Temporary ContinuousSignal object absorbed by " + self.label+"... ")
                del _tempCS
            return self

        if isinstance(value, pl.LazyFrame):

            if _TIME_SYMBOL in value.collect_schema().names() and _DATA_SYMBOL in value.collect_schema().names():
                # Left join to align on date, replacing where available
                self.__signal = (
                    self.__signal.join(value, on=_TIME_SYMBOL, how="left")
                    .with_columns(
                        pl.when(mask)
                        .then(pl.col(_DATA_SYMBOL + "_right"))  # Replaced values from the joined DataFrame
                        .otherwise(pl.col(_DATA_SYMBOL))  # Keep existing values otherwise
                        .alias(_DATA_SYMBOL)
                    )
                    .select([_TIME_SYMBOL, _DATA_SYMBOL])  # Drop unnecessary right-side columns
                )

            elif _DATA_SYMBOL in value.collect_schema().names():
                # Sequential value assignment: align by row position within the mask
                # Use row index joining for positional alignment
                signal_with_idx = self.__signal.with_row_index("__row_idx")
                value_with_idx = value.with_row_index("__row_idx")
                
                # Join on row index and update values sequentially
                joined = signal_with_idx.join(value_with_idx, on="__row_idx", how="left")
                
                # Replace data values where mask is True, using joined values
                self.__signal = joined.with_columns(
                    pl.when(mask)
                    .then(pl.col(_DATA_SYMBOL + "_right"))
                    .otherwise(pl.col(_DATA_SYMBOL))
                    .alias(_DATA_SYMBOL)
                ).select([_TIME_SYMBOL, _DATA_SYMBOL])

            elif _TIME_SYMBOL in value.collect_schema().names():
                # Date-to-date mapping: remap dates in signal based on mapping provided
                # Universal support for pandas/polars inputs
                value_schema = value.collect_schema()
                value_cols = set(value_schema.keys())
                value_types = {k: value_schema[k] for k in value_cols}
                
                if len(value_cols) == 1 and _TIME_SYMBOL in value_cols:
                    # Single TIME_SYMBOL column: sequential date mapping by position/index
                    # This maps dates positionally: signal row i gets value row i's date
                    # We need to align by row position, not by date value
                    signal_len = self.__signal.select(pl.len()).collect().item(0, 0)
                    value_len = value.select(pl.len()).collect().item(0, 0)
                    max_len = min(signal_len, value_len)
                    
                    # Create row numbers for alignment
                    signal_with_idx = self.__signal.with_row_index("__row_idx")
                    value_with_idx = value.with_row_index("__row_idx")
                    
                    # Join on row index for positional mapping
                    self.__signal = (
                        signal_with_idx.join(value_with_idx, on="__row_idx", how="left")
                        .with_columns(
                            pl.when(mask & (pl.col("__row_idx") < max_len))
                            .then(pl.col(_TIME_SYMBOL + "_right"))  # Use mapped date from value
                            .otherwise(pl.col(_TIME_SYMBOL))  # Keep original date
                            .alias(_TIME_SYMBOL)
                        )
                        .select([_TIME_SYMBOL, _DATA_SYMBOL])  # Drop row index
                    )
                    
                elif len(value_cols) == 2:
                    # Two columns: date mapping table
                    # One column should be old dates (to join on), other is new dates (to map to)
                    other_cols = [c for c in value_cols if c != _TIME_SYMBOL]
                    other_col = other_cols[0]
                    other_col_type = value_types[other_col]
                    
                    # Determine which column is old dates and which is new dates
                    # If other_col is datetime, it's likely the new dates
                    # If other_col is not datetime, TIME_SYMBOL is old dates, other_col might be something else
                    if other_col_type in (pl.Date, pl.Datetime):
                        # TIME_SYMBOL = old dates, other_col = new dates
                        # Join on TIME_SYMBOL, map to other_col
                        # Ensure datetime types match for join
                        signal_time_type = self.__signal.collect_schema()[_TIME_SYMBOL]
                        value_time_type = value_schema[_TIME_SYMBOL]
                        
                        # Normalize datetime types to match
                        if signal_time_type != value_time_type:
                            # Cast value's TIME_SYMBOL to match signal's type
                            value = value.with_columns(
                                pl.col(_TIME_SYMBOL).cast(signal_time_type)
                            )
                        
                        # After join, right side columns get _right suffix (except join key if names match)
                        # Since we're joining on TIME_SYMBOL, the right side's other_col will become other_col_right
                        mapped_col_name = other_col + "_right"
                        
                        # Perform join and mapping
                        # Join the value dataframe - right side columns get _right suffix
                        joined = self.__signal.join(value, on=_TIME_SYMBOL, how="left")
                        
                        # Get the actual column name after join (might be other_col_right or just other_col)
                        joined_schema = joined.collect_schema()
                        mapped_col_name = None
                        if other_col + "_right" in joined_schema:
                            mapped_col_name = other_col + "_right"
                        elif other_col in joined_schema and other_col != _TIME_SYMBOL:
                            mapped_col_name = other_col
                        else:
                            # Find any column that's not TIME_SYMBOL or DATA_SYMBOL
                            available = [c for c in joined_schema.keys() if c not in [_TIME_SYMBOL, _DATA_SYMBOL]]
                            if available:
                                mapped_col_name = available[0]
                            else:
                                raise ValueError(f"Could not find mapped date column after join. Available: {list(joined_schema.keys())}")
                        
                        # Update TIME_SYMBOL where mask is True and mapped date exists
                        self.__signal = (
                            joined
                            .with_columns(
                                pl.when(mask & pl.col(mapped_col_name).is_not_null())
                                .then(pl.col(mapped_col_name))  # Map to new date from value
                                .otherwise(pl.col(_TIME_SYMBOL))  # Keep original date
                                .alias(_TIME_SYMBOL)
                            )
                            .select([_TIME_SYMBOL, _DATA_SYMBOL])  # Drop the _right columns
                        )
                    else:
                        # Ambiguous - assume TIME_SYMBOL is join key, other_col might be an ID
                        # For now, treat TIME_SYMBOL as the mapping target
                        # This case might need more context from user
                        raise ValueError(f"Ambiguous date mapping: value has {_TIME_SYMBOL} and non-datetime column '{other_col}'. "
                                       f"Please provide a mapping table with old_date and new_date columns, or use a ContinuousSignal.")
                else:
                    raise ValueError(f"Date mapping requires 1-2 columns in value. Got {len(value_cols)} columns: {value_cols}")

            else:
                logger.error(f"Value must contain at least a '{_DATA_SYMBOL}' column.")
                raise ValueError(f"Value must contain at least a '{_DATA_SYMBOL}' column.")

        else:
            raise TypeError(f"Unsupported value type: {type(value)}")

        if _tempContinuousSignalFlag:
            logger.debug("Temporary ContinuousSignal object absorbed by " + self.label+"... ")
            del _tempCS

        return self

    def copy(self):
        """Creates a copy of the signal."""
        return ContinuousSignal(self.__label, self.__signal, self.__ticker, getattr(self, 'metadata', None),
                                self.keep_when_dropping_duplicates, by_ref=False)

    def to_pandas(self):
        """Converts to a Pandas DataFrame for compatibility."""
        logger.debug("Collecting LazyFrames... ("+self.label+") @to_pandas")
        return datetime_index(string_index(self.__signal).collect()).to_pandas()

    def __repr__(self):
        return f"ContinuousSignal(name={self.__label!r})"#, rows={self.__signal.collect().height})"

    # TODO: should return time and data in pandas or numpy?
    @property
    def time(self):
        """Returns time column lazy frame """
        if _TIME_SYMBOL in self.__signal.collect_schema().names():
            return self.__signal.select(_TIME_SYMBOL)
        elif self.__label == _TEMP_OBJECT_LABEL:
            logger.error(f"Temporary object has no {_TIME_SYMBOL} column. ")
            raise PermissionError(f"Temporary object has no {_TIME_SYMBOL} column. ")
        else:
            logger.exception()
            raise RuntimeError()

    @property
    def data(self):
        """Returns data column lazy frame"""
        if _DATA_SYMBOL in self.__signal.collect_schema().names():
            return self.__signal.select(_DATA_SYMBOL)
        elif self.__label == _TEMP_OBJECT_LABEL:
            logger.error(f"Temporary object has no {_DATA_SYMBOL} column. ")
            raise PermissionError(f"Temporary object has no {_DATA_SYMBOL} column. ")
        else:
            logger.exception()
            raise RuntimeError()

    @property
    def df(self):
        return self.__signal

    @property
    def cumulative(self):
        """Computes and returns a ContinuousSignal with the cumulative sum."""
        _df = self.__signal.with_columns(pl.col(_DATA_SYMBOL).cum_sum().alias(_DATA_SYMBOL))
        _signal = ContinuousSignal(_TEMP_OBJECT_LABEL, _df)
        _signal.__data_col_name = self.__data_col_name+"_cumulative"
        return _signal

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, name: str):
        self.__label = name

    @classmethod
    def get_signals(cls):
        """Returns a read-only copy of the dictionary."""
        return MappingProxyType(cls._SIGNALS_BY_TICKER)

    def temp_ref(self):
        return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal, self.__ticker, self.metadata,
                                self.keep_when_dropping_duplicates, by_ref=True)

    def settle(self, _label: str = 'result', ticker: Equity = None, metadata: dict = None, keep: str = None):
        """
        Settles an operation result (temp reference) object into a proper ContinuousSignal object.
        Will fail if the dataframe does not follow the required format.
        Use example: (csignal1 * csignal2).settle('row wise multiplied signal')
        """
        self.__is_temporary = False
        self.__label = _label

        if ticker: self.__ticker = ticker
        if metadata: self.metadata = metadata
        if keep: self.keep_when_dropping_duplicates = keep

        self._validate_signal_dataframe()
        self._validate_no_conflicting_duplicates()
        if self.__ticker:
            self.register_signal(self)

        return self

    # Time shift operations
    def __lshift__(self, other):
        """
        Shift dates backward: cs << 5 (shifts 5 days back)
        Returns a new temporary ContinuousSignal (soft copy with by_ref=True).
        Original signal is NOT modified (follows same pattern as +, -, *, etc.).
        """
        return self._time_shift(other, reverse=True)

    def __rshift__(self, other):
        """
        Shift dates forward: cs >> 5 (shifts 5 days forward)
        Returns a new temporary ContinuousSignal (soft copy with by_ref=True).
        Original signal is NOT modified (follows same pattern as +, -, *, etc.).
        """
        return self._time_shift(other, reverse=False)

    # Arithmetic Operations
    def __mul__(self, other):
        return self._apply_operation(self, other, '*')

    def __add__(self, other):
        return self._apply_operation(self, other, '+')

    def __sub__(self, other):
        return self._apply_operation(self, other, '-')

    def __truediv__(self, other):
        return self._apply_operation(self, other, '/')

    def __floordiv__(self, other):
        return self._apply_operation(self, other, '//')

    def __mod__(self, other):
        return self._apply_operation(self, other, '%')

    def __pow__(self, other):
        return self._apply_operation(self, other, '**')

    # Reverse Arithmetic Operations
    def __rtruediv__(self, other):
        return self._apply_operation(self, other, '/', reverse=True)

    def __rfloordiv__(self, other):
        return self._apply_operation(self, other, '//', reverse=True)

    def __rmod__(self, other):
        return self._apply_operation(self, other, '%', reverse=True)

    def __rpow__(self, other):
        return self._apply_operation(self, other, '**', reverse=True)

    def __rsub__(self, other):
        return self._apply_operation(self, other, '-', reverse=True)

    def __rmul__(self, other):
        return self._apply_operation(self, other, '*', reverse=True)

    def __radd__(self, other):
        return self._apply_operation(self, other, '+', reverse=True)

    # Unary Operations
    def __neg__(self):
        return self._apply_operation(self, -1, '*')

    def __pos__(self):
        return self

    def __abs__(self):
        return self._apply_operation(self, pl.col(_DATA_SYMBOL).abs(), '*')

    # In-Place Arithmetic Operations
    def __imul__(self, other):
        return self._apply_operation(self, other, '*')

    def __iadd__(self, other):
        return self._apply_operation(self, other, '+')

    def __isub__(self, other):
        return self._apply_operation(self, other, '-')

    def __itruediv__(self, other):
        return self._apply_operation(self, other, '/')

    def __ifloordiv__(self, other):
        return self._apply_operation(self, other, '//')

    def __imod__(self, other):
        return self._apply_operation(self, other, '%')

    def __ipow__(self, other):
        return self._apply_operation(self, other, '**')

    # Bitwise Operations
    def __and__(self, other):
        return self._apply_operation(self, other, '&')

    def __or__(self, other):
        return self._apply_operation(self, other, '|')

    def __xor__(self, other):
        return self._apply_operation(self, other, '^')

    def __invert__(self):
        return ContinuousSignal(_TEMP_OBJECT_LABEL,
                                self.__signal.with_columns((~pl.col(_DATA_SYMBOL)).alias(_DATA_SYMBOL)),
                                by_ref=True)

    # Comparison Operations
    def __eq__(self, other):
        return self._apply_operation(self, other, '==')

    def __ne__(self, other):
        return self._apply_operation(self, other, '!=')

    def __lt__(self, other):
        return self._apply_operation(self, other, '<')

    def __le__(self, other):
        return self._apply_operation(self, other, '<=')

    def __gt__(self, other):
        return self._apply_operation(self, other, '>')

    def __ge__(self, other):
        return self._apply_operation(self, other, '>=')

    _OPS_CACHE = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "//": lambda x, y: x // y,
        "%": lambda x, y: x % y,
        "**": lambda x, y: x ** y,
        "&": lambda x, y: x & y,
        "|": lambda x, y: x | y,
        "^": lambda x, y: x ^ y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
        # << and >> handled by _time_shift
    }

    @classmethod
    def _apply_operation(cls, self, other, operator, reverse=False):
        col = pl.col(_DATA_SYMBOL)
        op_func = cls._OPS_CACHE[operator]

        if isinstance(other, ContinuousSignal):
            other = other.__signal
            self.__signal = self.__signal.join(other, on=_TIME_SYMBOL, how="left")
            return ContinuousSignal(
                _TEMP_OBJECT_LABEL,
                self.__signal.with_columns(
                    op_func(col, pl.col(_DATA_SYMBOL + "_right")).alias(_DATA_SYMBOL)
                ).select([_TIME_SYMBOL, _DATA_SYMBOL]),
                by_ref=True
            )

        elif isinstance(other, numbers.Number):
            expr = op_func(pl.lit(other), col) if reverse else op_func(col, pl.lit(other))
            return ContinuousSignal(
                _TEMP_OBJECT_LABEL,
                self.__signal.with_columns(expr.alias(_DATA_SYMBOL)).with_columns(col.alias(_DATA_SYMBOL)),
                by_ref=True
            )

        elif isinstance(other, pl.Expr):
            expr = op_func(other, col) if reverse else op_func(col, other)
            return ContinuousSignal(
                _TEMP_OBJECT_LABEL,
                self.__signal.with_columns(expr.alias(_DATA_SYMBOL)).with_columns(col.alias(_DATA_SYMBOL)),
                by_ref=True
            )

        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __matmul__(self, other):
        """
        Date mapping operator: cs @ mapping_df
        
        Maps dates in the signal based on a mapping table.
        MODIFIES IN PLACE for performance (no copying).
        Use cs.copy() @ mapping if you need a copy.
        
        The mapping can be:
        - DataFrame with 'date' and another datetime column: old_date -> new_date
        - DataFrame with only 'date': positional mapping
        - ContinuousSignal: same as DataFrame
        
        Returns self (modified in place).
        
        Example:
            mapping = pl.DataFrame({'date': old_dates, 'new_date': new_dates})
            cs @ mapping  # Maps dates in place
            new_cs = cs.copy() @ mapping  # Explicit copy if needed
        """
        
        # Normalize input to LazyFrame
        if isinstance(other, ContinuousSignal):
            mapping = other.__signal
        else:
            mapping = self._normalize_to_lazyframe(other, allow_pandas=True)
        
        # Check if this is a date mapping operation
        mapping_schema = mapping.collect_schema()
        mapping_cols = set(mapping_schema.keys())
        
        if _TIME_SYMBOL not in mapping_cols:
            raise ValueError(f"Date mapping requires '{_TIME_SYMBOL}' column in mapping. Got columns: {mapping_cols}")
        
        # Create new LazyFrame (immutable operations) - don't modify self.__signal
        # Follows same pattern as _apply_operation: create new LazyFrame, return temp object
        if len(mapping_cols) == 1 and _TIME_SYMBOL in mapping_cols:
            # Sequential date mapping by position
            signal_len = self.__signal.select(pl.len()).collect().item(0, 0)
            mapping_len = mapping.select(pl.len()).collect().item(0, 0)
            max_len = min(signal_len, mapping_len)
            
            signal_with_idx = self.__signal.with_row_index("__row_idx")
            mapping_with_idx = mapping.with_row_index("__row_idx")
            
            new_signal = (
                signal_with_idx.join(mapping_with_idx, on="__row_idx", how="left")
                .with_columns(
                    pl.when(pl.col("__row_idx") < max_len)
                    .then(pl.col(_TIME_SYMBOL + "_right"))
                    .otherwise(pl.col(_TIME_SYMBOL))
                    .alias(_TIME_SYMBOL)
                )
                .select([_TIME_SYMBOL, _DATA_SYMBOL])
            )
            
        elif len(mapping_cols) == 2:
            # Date mapping table: TIME_SYMBOL = old dates, other_col = new dates
            other_cols = [c for c in mapping_cols if c != _TIME_SYMBOL]
            other_col = other_cols[0]
            other_col_type = mapping_schema[other_col]
            
            if other_col_type in (pl.Date, pl.Datetime):
                    # Ensure datetime types match
                    signal_time_type = self.__signal.collect_schema()[_TIME_SYMBOL]
                    mapping_time_type = mapping_schema[_TIME_SYMBOL]
                    
                    if signal_time_type != mapping_time_type:
                        mapping = mapping.with_columns(
                            pl.col(_TIME_SYMBOL).cast(signal_time_type)
                        )
                    
                    # Perform join - creates new LazyFrame (immutable)
                    joined = self.__signal.join(mapping, on=_TIME_SYMBOL, how="left")
                    
                    # Get the actual column name after join
                    joined_schema = joined.collect_schema()
                    mapped_col_name = None
                    if other_col + "_right" in joined_schema:
                        mapped_col_name = other_col + "_right"
                    elif other_col in joined_schema and other_col != _TIME_SYMBOL:
                        mapped_col_name = other_col
                    else:
                        # Find any column that's not TIME_SYMBOL or DATA_SYMBOL
                        available = [c for c in joined_schema.keys() if c not in [_TIME_SYMBOL, _DATA_SYMBOL]]
                        if available:
                            mapped_col_name = available[0]
                        else:
                            raise ValueError(f"Could not find mapped date column after join. Available: {list(joined_schema.keys())}")
                    
                    new_signal = (
                        joined
                        .with_columns(
                            pl.when(pl.col(mapped_col_name).is_not_null())
                            .then(pl.col(mapped_col_name))
                            .otherwise(pl.col(_TIME_SYMBOL))
                            .alias(_TIME_SYMBOL)
                        )
                        .select([_TIME_SYMBOL, _DATA_SYMBOL])
                    )
            else:
                raise ValueError(f"Date mapping requires datetime column for new dates. Got type: {other_col_type}")
        else:
            raise ValueError(f"Date mapping requires 1-2 columns. Got {len(mapping_cols)} columns: {mapping_cols}")
        
        # Return temp object with by_ref=True (soft copy - shares LazyFrame reference)
        # Original self.__signal is NOT modified (LazyFrames are immutable)
        return ContinuousSignal(
            _TEMP_OBJECT_LABEL,
            new_signal,
            self.__ticker,
            self.metadata,
            self.keep_when_dropping_duplicates,
            by_ref=True
        )
    
    def __imatmul__(self, other):
        """
        In-place date mapping operator: cs @= mapping_df
        
        Modifies the signal in place by reassigning self.__signal.
        """
        result = self @ other
        self.__signal = result.__signal
        return self

    def _time_shift(self, other, reverse=False, resolution="days"):
        """
        Changes the time column (_TIME_SYMBOL) values of self.__signal based on `other`.
        Returns a new temporary ContinuousSignal (soft copy with by_ref=True).
        Original signal is NOT modified (follows same pattern as +, -, *, etc.).

        :param other: Can be a timedelta, number, array of shifts, or another ContinuousSignal.
        :param reverse: If True, shifts backward instead of forward.
        :param resolution: Defines the unit of shift for numeric values (e.g., "days", "hours").
        """
        time_col = pl.col(_TIME_SYMBOL)
        shift_expr = None
        mask = pl.lit(True) # Will serve to filter the rows to be changed in the case of date mappings

        # First see if the shift is a timedelta/number

        if isinstance(other, (timedelta, pd.Timedelta, pl.Duration)):
            # Convert timedelta to polars duration
            if isinstance(other, timedelta):
                shift_expr = pl.duration(days=other.days, seconds=other.seconds, microseconds=other.microseconds)
            elif isinstance(other, pd.Timedelta):
                # Convert pandas timedelta to polars duration
                total_seconds = other.total_seconds()
                days = int(total_seconds // 86400)
                seconds = int(total_seconds % 86400)
                microseconds = int((total_seconds % 1) * 1_000_000)
                shift_expr = pl.duration(days=days, seconds=seconds, microseconds=microseconds)
            else:
                shift_expr = other  # Already a pl.Duration
            
            # Create new LazyFrame (immutable) - return temp object like regular operators
            if reverse:
                new_signal = self.__signal.with_columns(
                    (pl.col(_TIME_SYMBOL) - shift_expr).alias(_TIME_SYMBOL)
                )
            else:
                new_signal = self.__signal.with_columns(
                    (pl.col(_TIME_SYMBOL) + shift_expr).alias(_TIME_SYMBOL)
                )
            
            return ContinuousSignal(
                _TEMP_OBJECT_LABEL,
                new_signal,
                self.__ticker,
                self.metadata,
                self.keep_when_dropping_duplicates,
                by_ref=True
            )

        elif isinstance(other, (int, float)):
            # Create duration expression for the shift
            if resolution == "days":
                shift_expr = pl.duration(days=other)
            elif resolution == "hours":
                shift_expr = pl.duration(hours=other)
            elif resolution == "minutes":
                shift_expr = pl.duration(minutes=other)
            elif resolution == "seconds":
                shift_expr = pl.duration(seconds=other)
            else:
                shift_expr = pl.duration(**{resolution: other})
            
            # Create new LazyFrame (immutable) - return temp object like regular operators
            if reverse:
                # Shift backward (subtract time)
                new_signal = self.__signal.with_columns(
                    (pl.col(_TIME_SYMBOL) - shift_expr).alias(_TIME_SYMBOL)
                )
            else:
                # Shift forward (add time)
                new_signal = self.__signal.with_columns(
                    (pl.col(_TIME_SYMBOL) + shift_expr).alias(_TIME_SYMBOL)
                )
            
            return ContinuousSignal(
                _TEMP_OBJECT_LABEL,
                new_signal,
                self.__ticker,
                self.metadata,
                self.keep_when_dropping_duplicates,
                by_ref=True
            )

        # If shift by numbers or time deltas, go on to the end...
        else:

            # Normalize other to LazyFrame (handles pandas/polars/ContinuousSignal)
            if isinstance(other, ContinuousSignal):
                other_signal = other.__signal
            else:
                # Use universal converter for pandas/polars
                other_signal = self._normalize_to_lazyframe(other, allow_pandas=True)
            
            # Infer intent from schema using universal inference
            col_name, shift_map_col, is_mapping, operation_type = self._infer_shift_map_intent(other_signal)
            
            my_signal_height = self.__signal.select(pl.len()).collect().item(0, 0)
            
            # Handle based on inferred operation type
            if operation_type == 'map_dates':
                # Date-to-date mapping: join and replace TIME_SYMBOL
                # Create new LazyFrame (immutable) - return temp object like regular operators
                joined = self.__signal.join(other_signal, on=_TIME_SYMBOL, how="left")
                # After join, the column will have _right suffix
                mapped_col_name = col_name + "_right"
                # Verify it exists, fallback to TIME_SYMBOL_right if col_name was TIME_SYMBOL
                joined_schema = joined.collect_schema()
                if mapped_col_name not in joined_schema and _TIME_SYMBOL + "_right" in joined_schema:
                    mapped_col_name = _TIME_SYMBOL + "_right"
                
                new_signal = joined.with_columns(
                    pl.when(mask)
                    .then(pl.col(mapped_col_name))
                    .otherwise(pl.col(_TIME_SYMBOL))
                    .alias(_TIME_SYMBOL)
                ).select([_TIME_SYMBOL, _DATA_SYMBOL])
                
                return ContinuousSignal(
                    _TEMP_OBJECT_LABEL,
                    new_signal,
                    self.__ticker,
                    self.metadata,
                    self.keep_when_dropping_duplicates,
                    by_ref=True
                )
                
            elif operation_type == 'map_values':
                # Value mapping: join on dates, replace data values
                # Create new LazyFrame (immutable) - return temp object like regular operators
                mask_height = self.__signal.filter(mask).select(pl.len()).collect().item(0, 0)
                max_len = min(my_signal_height, mask_height)
                
                # After join, column will have _right suffix
                joined = self.__signal.join(other_signal, on=_TIME_SYMBOL, how="left")
                mapped_col_name = col_name + "_right"
                joined_schema = joined.collect_schema()
                if mapped_col_name not in joined_schema:
                    # Fallback: try to find the right column
                    right_cols = [k for k in joined_schema.keys() if k.endswith("_right")]
                    if right_cols:
                        mapped_col_name = right_cols[0]
                
                new_signal = (
                    joined
                    .with_columns(
                        pl.when(mask)
                        .then(pl.col(mapped_col_name))
                        .otherwise(pl.col(_DATA_SYMBOL))
                        .alias(_DATA_SYMBOL)
                    )
                    .select([_TIME_SYMBOL, _DATA_SYMBOL])
                )
                
                return ContinuousSignal(
                    _TEMP_OBJECT_LABEL,
                    new_signal,
                    self.__ticker,
                    self.metadata,
                    self.keep_when_dropping_duplicates,
                    by_ref=True
                )
                
            elif operation_type == 'shift':
                # Numeric shift: use the column expression directly
                shift_expr = shift_map_col
                
                if is_mapping:
                    # MAP BY ORDER with join (date-aligned shift)
                    # Create new LazyFrame (immutable) - return temp object like regular operators
                    mask_height = self.__signal.filter(mask).select(pl.len()).collect().item(0, 0)
                    max_len = min(my_signal_height, mask_height)
                    
                    joined = self.__signal.join(other_signal, on=_TIME_SYMBOL, how="left")
                    mapped_col_name = col_name + "_right"
                    new_signal = (
                        joined
                        .with_columns(
                            pl.when(mask)
                            .then(pl.col(mapped_col_name))
                            .otherwise(pl.col(_DATA_SYMBOL))
                            .alias(_DATA_SYMBOL)
                        )
                        .select([_TIME_SYMBOL, _DATA_SYMBOL])
                    )
                    
                    return ContinuousSignal(
                        _TEMP_OBJECT_LABEL,
                        new_signal,
                        self.__ticker,
                        self.metadata,
                        self.keep_when_dropping_duplicates,
                        by_ref=True
                    )
                # SHIFT BY ORDER (no join) - apply shift expression directly
                # Create new LazyFrame (immutable) - return temp object like regular operators
                if reverse:
                    new_signal = self.__signal.with_columns(
                        (pl.col(_TIME_SYMBOL) - shift_expr).alias(_TIME_SYMBOL)
                    )
                else:
                    new_signal = self.__signal.with_columns(
                        (pl.col(_TIME_SYMBOL) + shift_expr).alias(_TIME_SYMBOL)
                    )
                
                return ContinuousSignal(
                    _TEMP_OBJECT_LABEL,
                    new_signal,
                    self.__ticker,
                    self.metadata,
                    self.keep_when_dropping_duplicates,
                    by_ref=True
                )
            else:
                raise TypeError(f"Unsupported type for time shifting: {type(other)}")
