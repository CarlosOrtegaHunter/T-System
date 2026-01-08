import numbers
from types import MappingProxyType
from datetime import timedelta

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
                self.__signal = signal.__signal
                self.__ticker = signal.__ticker
                self.metadata = signal.metadata
                self.keep_when_dropping_duplicates = signal.keep_when_dropping_duplicates
            else:
                self.__signal = signal.__signal.clone()
                self.__ticker = signal.__ticker
                self.metadata = signal.metadata.copy()
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

    def _signal_to_lazyframe(self, signal):
        if signal is None:
            return pl.DataFrame(schema={_TIME_SYMBOL: pl.Datetime, _DATA_SYMBOL: pl.Float64}).lazy()
        elif isinstance(signal, pl.LazyFrame):
            return signal
        elif isinstance(signal, pl.Series):
            # Series should be allowed for date shifting operations!
            # todo: check this works correctly^ handle non matching column lengths as a left join
            if self.label != _TEMP_OBJECT_LABEL:
                logger.error("Only temporary ContinuousSignal objects are allowed to wrap a Series signal. ")
                raise ValueError("Only temporary ContinuousSignal objects are allowed to wrap a Series signal. ")
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
        self.__signal = datetime_index(self.__signal) #TODO: handle possible casting exceptions

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
        self.__signal = string_index(self.__signal).collect()
        duplicate_mask = self.__signal.select(_TIME_SYMBOL).is_duplicated()

        if duplicate_mask.any():
            conflicts_number = self.__signal.filter(duplicate_mask).n_unique(subset=_TIME_SYMBOL) > 1
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
        Mimics Pandas-like selection.
        - `signal["value"]` -> Returns a Polars Series (not LazyFrame).
        - `signal[start:end]` -> Filters by date range.
        - `signal[mask]` -> Boolean filtering.
        """
        logger.debug("__getitem__ "+self.__label)

        if isinstance(key, str):
            if key.lower() == self.__data_col_name:
                key = _DATA_SYMBOL
            # Ensure it returns a Series instead of LazyFrame
            #TODO: handle getting individual datetimes, string dates!
            return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal.select(pl.col(key.lower())), by_ref=True)

        elif isinstance(key, slice):
            start, end = key.start, key.stop
            if start is None:
                start = self.__signal.select(pl.col(_TIME_SYMBOL).min()).collect().item(0, 0)
            if end is None:
                end = self.__signal.select(pl.col(_TIME_SYMBOL).max()).collect().item(0, 0)

            return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal.filter(pl.col(_TIME_SYMBOL).is_between(start, end)), by_ref=True)

        elif isinstance(key, pl.Series):
            return ContinuousSignal(_TEMP_OBJECT_LABEL, self.__signal.filter(key), by_ref=True)

        elif isinstance(key, int):
            height = self.__signal.select(pl.len()).collect().item(0, 0)
            return self.__signal.select(pl.col(_DATA_SYMBOL)).collect().item(key % height, 0)

        elif isinstance(key, pl.Datetime):
            result = self.__signal.filter(pl.col(_TIME_SYMBOL) == pl.lit(key)).select(_DATA_SYMBOL)
            if result.select(pl.len()).collect().item(0, 0) == 0:
                raise KeyError(f"No data found for timestamp: {key}")
            return result.collect().item(0, 0)

        else:
            raise TypeError(f"Invalid selection type: {type(key)}.")

    def __setitem__(self, key, value):
        """
        Allows setting values like Pandas.
        - `signal["column"] = new_series` -> Updates a column.
        - `signal[start:end] = new_values` -> Updates rows by date range.
        - `signal[mask] = new_value` -> Updates rows where the mask is True.
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

        # Check value type, convert to pl.LazyFrame

        if isinstance(value, np.ndarray):
            value = pl.DataFrame({ _DATA_SYMBOL: value.flatten() }).lazy()

        if isinstance(value, pd.DataFrame):
            value = pl.from_pandas(value).lazy()

        if isinstance(value, pd.Series):
            value = pl.Series(value)

        if isinstance(value, pl.Series):
            value.to_frame(name=_DATA_SYMBOL).lazy()

        if isinstance(value, pl.DataFrame):
            value = value.lazy()

        if not isinstance(value, pl.LazyFrame | pl.Datetime | int | float):
            logger.error(f"Unsupported type for value: {type(value)}")
            raise TypeError(f"Unsupported type for value: {type(value)}")

        # Create mask of dates on which to operate
        mask = pl.lit(True)

        if isinstance(key, str):
            # Todo: I am assuming key is not a string date
            if key.lower() != self.__data_col_name and key.lower() != _DATA_SYMBOL:
                raise IndexError(f"No such column as '{key.lower()}'. Only columns are '{_DATA_SYMBOL}' and '{self.__data_col_name}' (aka '{_DATA_SYMBOL}'). ")
            # mask remains everywhere true

        elif isinstance(key, slice):
            start, end = key.start, key.stop
            if start is None:
                start = self.__signal.select(pl.col(_TIME_SYMBOL).min()).collect().item(0, 0)
            if end is None:
                end = self.__signal.select(pl.col(_TIME_SYMBOL).max()).collect().item(0, 0)

            mask = pl.col(_TIME_SYMBOL).is_between(start, end)

        # TODO: check this works
        elif isinstance(key, pl.Series) and key.dtype == pl.Boolean:
            mask = key

        elif isinstance(key, pl.Datetime):
            raise NotImplementedError("Datetime indexing not implemented")

        else:
            logger.exception(f"Invalid key type: {type(key)}.")
            raise TypeError(f"Invalid key type: {type(key)}.")

        # Perform data assignment

        if isinstance(value, (int, float)):
            self.__signal = self.__signal.with_columns(
                pl.when(mask).then(value).otherwise(pl.col(_DATA_SYMBOL)).alias(_DATA_SYMBOL)
            )

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

                value_height = value.select(pl.len()).collect().item(0, 0)
                mask_height = self.__signal.filter(mask).select(pl.len()).collect().item(0, 0)
                max_len = min(mask_height, value_height)

                # Copy sequentially and replace only within mask's range
                self.__signal = self.__signal.with_columns(
                    pl.when(mask)
                    .then(pl.col(_DATA_SYMBOL).limit(max_len))  # Replace only up to the available rows
                    .otherwise(pl.col(_DATA_SYMBOL))
                    .alias(_DATA_SYMBOL)
                )

            elif _TIME_SYMBOL in value.collect_schema().names():
                #TODO: fix this...? in order to have mappings of dates! (date operations... write functions too)
                # I should also implement a QFT-like sort of class for advanced calendar operations
                raise NotImplementedError(f"Only column cannot be a non-temporary '{_TIME_SYMBOL}' column dataframe.")

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
        return ContinuousSignal(self.__label, self.__signal, self.__ticker, self.metadata,
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
        """Returns collected time column lazy frame """
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

    # todo: must be scrapped
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

    # Time shift operations
    def __lshift__(self, other):
        """
        Index-based date shift backward.
        `cs << n` shifts each observation to the timestamp that is `n` rows earlier
        in the sorted time index, independent of the underlying calendar.
        """
        return self._time_shift(other, reverse=True)

    def __rshift__(self, other):
        """
        Index-based date shift forward.
        `cs >> n` shifts each observation to the timestamp that is `n` rows later
        in the sorted time index, independent of the underlying calendar.
        """
        return self._time_shift(other, reverse=False)

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

        # TODO: DAY SHIFTING OPERATIONS
        #  <<
        #  >>
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

    def _time_shift(self, other, reverse: bool = False):
        """
        Index/slot-based time shift.

        - Integer/float `other` is interpreted as a shift in *rows* of the sorted time index.
          This is calendar-agnostic and works for any market/calendar (US, EU, crypto, intraday, etc.).

            cs >> 35  # move each observation to the timestamp 35 rows later
            cs << 10  # move each observation to the timestamp 10 rows earlier

        - timedelta / Timedelta / pl.Duration perform true calendar/time shifts and are
          handled via datetime arithmetic on the `_TIME_SYMBOL` column.
        """
        # Calendar/time-based shift
        if isinstance(other, (timedelta, pd.Timedelta, pl.Duration)):
            if isinstance(other, timedelta):
                shift_expr = pl.duration(days=other.days,
                                         seconds=other.seconds,
                                         microseconds=other.microseconds)
            elif isinstance(other, pd.Timedelta):
                total_seconds = other.total_seconds()
                days = int(total_seconds // 86400)
                seconds = int(total_seconds % 86400)
                microseconds = int((total_seconds % 1) * 1_000_000)
                shift_expr = pl.duration(days=days,
                                         seconds=seconds,
                                         microseconds=microseconds)
            else:
                # Already a pl.Duration
                shift_expr = other

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

        # Index/slot-based shift
        if isinstance(other, (int, float)):
            offset = int(other)

            if offset == 0:
                return ContinuousSignal(
                    _TEMP_OBJECT_LABEL,
                    self.__signal,
                    self.__ticker,
                    self.metadata,
                    self.keep_when_dropping_duplicates,
                    by_ref=True
                )

            # Use row-wise shift on the timestamp column (no join, purely index-based)
            if reverse:
                # cs << n  => map each row to timestamp at row_idx - n
                shifted_dates = pl.col(_TIME_SYMBOL).shift(offset)
            else:
                # cs >> n  => map each row to timestamp at row_idx + n
                shifted_dates = pl.col(_TIME_SYMBOL).shift(-offset)

            new_signal = self.__signal.with_columns(
                pl.when(shifted_dates.is_not_null())
                .then(shifted_dates)
                .otherwise(pl.col(_TIME_SYMBOL))
                .alias(_TIME_SYMBOL)
            )

            return ContinuousSignal(
                _TEMP_OBJECT_LABEL,
                new_signal,
                getattr(self, "_ContinuousSignal__ticker", None),
                getattr(self, "metadata", None),
                getattr(self, "keep_when_dropping_duplicates", None),
                by_ref=True
            )

        raise TypeError(f"Unsupported type for time shifting: {type(other)}")
