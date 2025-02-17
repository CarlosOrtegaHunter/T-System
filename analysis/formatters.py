import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import pandas as pd
from common.config import logger
from common.utils import are_columns_numeric

#TODO: PercentageFormatter 

class BigNumberFormatter:
    """A utility class for formatting numerical values with order of magnitude labels (k, M)."""

    def __init__(self, data: np.ndarray | pl.DataFrame | pl.LazyFrame | pd.DataFrame, scale_factor = None):
        """
        Computes a scale factor for automatic formatting.
        :param data: Numpy array or DataFrame containing data which will be formatted.
        """
        self.scale_factor = scale_factor
        self.scaled_data = None

        self.compute_scaled(data)

        if not self.scale_factor:
            self.scale_factor = 1

    def format(self, x, pos=None):
        """
        Format a number based on the scale factor.
        :param x: The numerical value to format.
        :param pos: Position argument required by Matplotlib formatters (not used directly).
        :return: Formatted string with appropriate suffix ('k', 'M', ...).
        """
        if self.scale_factor >= 1e18:
            return f'{x :.1f}Qn'  # Convert to quintillions
        if self.scale_factor >= 1e15:
            return f'{x :.1f}Qd'  # Convert to quadrillions
        if self.scale_factor >= 1e12:
            return f'{x :.1f}T'  # Convert to trillions
        if self.scale_factor >= 1e9:
            return f'{x :.1f}B'  # Convert to billions
        if self.scale_factor >= 1e6:
            return f'{x :.1f}M'  # Convert to millions
        elif self.scale_factor >= 1e3:
            return f'{x :.1f}k'  # Convert to thousands
        else:
            return f'{x:.1f}'  # Keep as is for smaller values

    def get_formatter(self):
        """
        :return: A function compatible with Matplotlib tick formatters.
        """
        return mticker.FuncFormatter(self.format)

    def compute_scaled(self, data: np.ndarray | pl.DataFrame | pl.LazyFrame | pd.DataFrame) -> np.ndarray:
        """ If the input is a pl.LazyFrame or pl.DataFrame, returns a pl.DataFrame.
            If the input is a pd.DataFrame, returns a pd.DataFrame. """
        # todo: implement zoom?
        err_msg = "BigNumberFormatter argument must be a one-dimensional numpy array, pandas or polars DataFrame of numeric type."

        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        if isinstance(data, pd.DataFrame):
            if len(data.columns) != 1:
                logger.error(err_msg)
                ValueError(err_msg)
            data = data.to_numpy()

        if isinstance(data, pl.DataFrame):
            if len(data.collect_schema().names()) != 1:
                logger.error(err_msg)
                ValueError(err_msg)
            data = data.to_numpy()

        if not isinstance(data, np.ndarray):
            logger.error(err_msg)
            raise TypeError(err_msg)

        data = data.flatten()

        if not are_columns_numeric(data):
            TypeError(err_msg)

        if not self.scale_factor:
            magnitude = np.floor(np.log10(np.abs(data.mean())))
            scale_factor = 10 ** magnitude
            if scale_factor >= 1e18:
                scale_factor = 1e18  # Quintillions
            elif scale_factor >= 1e15:
                scale_factor = 1e15  # Quadrillions
            elif scale_factor >= 1e12:
                scale_factor = 1e12  # Trillions
            elif scale_factor >= 1e9:
                scale_factor = 1e9  # Billions
            elif scale_factor >= 1e6:
                scale_factor = 1e6  # Millions
            elif scale_factor >= 1e3:
                scale_factor = 1e3  # thousands
            elif scale_factor < 1e3:
                scale_factor = 1

            # todo: ughh.. i think this should handle different time windows...
            self.scale_factor = scale_factor

        self.scaled_data = data / self.scale_factor
        return self.scaled_data