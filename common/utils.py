from typing import Union

import numpy as np
import polars as pl
import pandas as pd

def lowercase_columns(df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]):
    if isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame):
        return df.rename({col: col.lower() for col in df.collect_schema().names()}) if df.collect_schema().len() > 0 else df
    elif isinstance(df, pd.DataFrame):
        df.columns = df.columns.str.lower()
        return df
    return df

def datetime_index(df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
    if df.collect_schema()['date'] == pl.Datetime: return df
    try:
        return df.with_columns(pl.col("date").str.to_datetime())
    except pl.exceptions.ComputeError as e:
        raise ValueError(f"Failed to convert 'date' column to datetime: {e}") from e

def string_index(df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
    if df.collect_schema()['date'] == pl.Utf8: return df
    try:
        return df.with_columns(pl.col("date").cast(pl.Utf8()))
    except pl.exceptions.ComputeError as e:
        # If date column can't be cast, it might be because it's not valid datetime
        raise ValueError(f"Failed to convert 'date' column to string. Ensure date column is valid datetime first: {e}") from e

def join_on_date(dfs: Union[list[pl.DataFrame] | list[pl.LazyFrame]]) -> pl.DataFrame | pl.LazyFrame:
    """Performs outer join of a dataframe list."""
    if not dfs:
        raise ValueError("The list of DataFrames/LazyFrames is empty.")
    is_lazy = any(isinstance(df, pl.LazyFrame) for df in dfs)
    if is_lazy:
        dfs = [df.lazy() if isinstance(df, pl.DataFrame) else df for df in dfs]
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.join(df, on="date", how="outer")
    return df_merged if is_lazy else df_merged

def are_columns_numeric(df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]) -> bool:
    """Check if all columns in a Pandas DataFrame, Polars DataFrame, or LazyFrame are numeric."""
    numeric_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64)
    if isinstance(df, np.ndarray):
        return np.issubdtype(df.dtype, np.number) and np.all(np.isfinite(df))
    if isinstance(df, pd.DataFrame):
        return all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
    elif isinstance(df, pl.DataFrame):
        return all(dtype in numeric_types for dtype in df.collect_schema().values())
    elif isinstance(df, pl.LazyFrame):
        return all(dtype in numeric_types for dtype in df.collect_schema().values())
    else:
        return False
