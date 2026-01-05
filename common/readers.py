from common.config import logger
from pathlib import Path
import pandas as pd
import polars as pl

class CSV:
    """Utility for retrieving CSV data with a simple function-like interface."""

    _resources_path = Path(__file__).resolve().parent.parent / "resources"

    def __new__(cls, filename: str) -> pl.DataFrame:
        """
        Loads a single CSV file as a DataFrame.
        :param filename: Relative path of the CSV file within the resources folder.
        :return: Polars DataFrame containing the CSV data.
        """
        file_path = cls._resolve_path(filename)
        return cls._read_csv(file_path)

    @classmethod
    def from_pattern(cls, pattern: str) -> list:
        """
        Reads multiple CSV files matching a pattern and returns them as a list of DataFrames.
        :param pattern: File pattern to match, using '/' as a separator (e.g., 'options/gme/*.csv').
        :return: List of Polars DataFrames, one per matched file. Returns an empty list if no files are found.
        """
        matched_files = list(cls._resolve_path(pattern).parent.glob(cls._resolve_path(pattern).name))

        if not matched_files:
            logger.warning(f"Warning: No files found matching pattern '{pattern}'")
            return []

        results = []
        for file in matched_files:
            df = cls._read_csv(file)
            results.append(df)

        return results

    @classmethod
    def from_pattern_concatenate(cls, pattern: str) -> pl.DataFrame:
        """
        Reads multiple CSV files matching a pattern and concatenates them into a single DataFrame.
        :param pattern: File pattern to match, using '/' as a separator.
        :return: A single Polars DataFrame containing concatenated data from all matched files.
                 Returns an empty DataFrame if no files are found.
        """
        dataframes = cls.from_pattern(pattern)
        return pl.concat(dataframes) if dataframes else pl.DataFrame()

    @classmethod
    def _resolve_path(cls, relative_path: str) -> Path:
        """
        Converts a relative file pattern into an OS-independent absolute path.
        :param relative_path: A relative path string using '/' as separator (e.g., 'options/gme/*.csv').
        :return: An absolute Path object compatible with the current operating system.
        """
        return cls._resources_path / Path(*relative_path.split('/'))

    @staticmethod
    def _read_csv(file_path: Path) -> pl.DataFrame:
        """
        Reads a CSV file safely with error handling, returning a polars DataFrame.
        Handles both 'date' and 'time' columns (time can be Unix timestamp).
        :param file_path: Absolute Path to the CSV file.
        :return: A Polars DataFrame containing the file's data.
                 Returns an empty DataFrame if the file is missing or cannot be read.
        """
        try:
            df = pd.read_csv(file_path, dtype=str) #pandas read_csv works better for now
            original_columns = df.columns.tolist()
            df.columns = [col.lower() for col in df.columns]
            percentage_columns = []

            # Handle time column (Unix timestamp) - convert to date
            if "time" in df.columns and "date" not in df.columns:
                # Check if time values look like Unix timestamps (large numbers)
                sample_time = df["time"].iloc[0] if len(df) > 0 else None
                if sample_time and sample_time.isdigit() and len(sample_time) >= 10:
                    # Convert Unix timestamp to datetime, then normalize to date (remove time component)
                    df["date"] = pd.to_datetime(df["time"].astype(int), unit='s').dt.normalize()
                    df = df.drop(columns=["time"])
                else:
                    # Treat as regular datetime string
                    df["date"] = pd.to_datetime(df["time"], errors='coerce').dt.normalize()
                    df = df.drop(columns=["time"])

            for col in df.columns:
                if col != "date" and col != "time":
                    if df[col].astype(str).str.contains("%").any():
                        percentage_columns.append(col)
                        df[col] = df[col].str.replace("%", "", regex=True).str.strip()
                        df[col] = pd.to_numeric(df[col], errors="coerce") / 100
                    else:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna()
            pl_df = pl.from_pandas(df)
            #pl_df = pl_df.with_columns(pl.col("date").cast(pl.Datetime))
            return pl_df

        except FileNotFoundError:
            logger.exception(f"Error: File not found - {file_path}")
            return pl.DataFrame()

        except pl.exceptions.NoDataError:
            logger.exception(f"Error: Empty file - {file_path}")
            return pl.DataFrame()

        except Exception as e:
                logger.exception(f"Error reading {file_path}: {e}")
                return pl.DataFrame()
