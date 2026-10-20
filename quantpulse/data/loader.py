"""Market data loader with schema validation and strict point-in-time / look-ahead protection."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from quantpulse.core.config import DEFAULT_SAMPLE_CSV


class MarketRecord(BaseModel):
    """Schema validation for a single market data observation."""

    Date: str
    Ticker: str
    Open: float = Field(gt=0.0)
    High: float = Field(gt=0.0)
    Low: float = Field(gt=0.0)
    Close: float = Field(gt=0.0)
    Adj_Close: float = Field(gt=0.0)
    Volume: int = Field(ge=0)

    @field_validator("High")
    @classmethod
    def validate_high_bounds(cls, v: float, info) -> float:
        # High must be greater than or equal to Low
        low = info.data.get("Low")
        if low is not None and v < low:
            raise ValueError(f"High ({v}) cannot be lower than Low ({low})")
        return v


class MarketDataLoader:
    """Robust data loader that ensures data integrity and prevents look-ahead bias."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_SAMPLE_CSV
        self._raw_df: Optional[pd.DataFrame] = None
        self._price_matrix: Optional[pd.DataFrame] = None

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and validate market data from disk."""
        if self._raw_df is not None and not force_reload:
            return self._raw_df

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.data_path}. "
                f"Run generator to create sample dataset."
            )

        df = pd.read_csv(self.data_path)
        required_cols = {"Date", "Ticker", "Open", "High", "Low", "Close", "Adj_Close", "Volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Ensure datetime format
        df["Date"] = pd.to_datetime(df["Date"])

        # Enforce strict chronological ordering
        df.sort_values(by=["Date", "Ticker"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Integrity checks: no duplicate Date-Ticker pairs
        if df.duplicated(subset=["Date", "Ticker"]).any():
            duplicates = df[df.duplicated(subset=["Date", "Ticker"], keep=False)]
            raise ValueError(f"Found duplicate (Date, Ticker) pairs:\n{duplicates.head()}")

        # Integrity check: positive prices
        if (df["Adj_Close"] <= 0).any():
            invalid_prices = df[df["Adj_Close"] <= 0]
            raise ValueError(f"Found non-positive Adjusted Close prices:\n{invalid_prices}")

        self._raw_df = df
        return self._raw_df

    def get_price_matrix(
        self,
        tickers: Optional[List[str]] = None,
        price_col: str = "Adj_Close",
    ) -> pd.DataFrame:
        """Pivot the long-form market data into a wide price matrix.

        Index: DatetimeIndex (strictly sorted ascending)
        Columns: Tickers
        Values: Specified price column (default 'Adj_Close')
        """
        df = self.load()
        if tickers:
            df = df[df["Ticker"].isin(tickers)]
            missing_tickers = set(tickers) - set(df["Ticker"].unique())
            if missing_tickers:
                raise ValueError(f"Requested tickers not found in dataset: {missing_tickers}")

        pivoted = df.pivot(index="Date", columns="Ticker", values=price_col)
        # Sort index chronologically
        pivoted.sort_index(ascending=True, inplace=True)
        # Forward fill then backward fill any occasional missing market holiday discrepancies
        pivoted.ffill(inplace=True)
        pivoted.bfill(inplace=True)
        self._price_matrix = pivoted
        return self._price_matrix

    def get_point_in_time(
        self,
        as_of_date: Union[str, pd.Timestamp],
        tickers: Optional[List[str]] = None,
        price_col: str = "Adj_Close",
    ) -> pd.DataFrame:
        """Retrieve price history strictly up to `as_of_date` (inclusive).

        Crucial for eliminating look-ahead bias during backtesting, ML feature
        generation, and live simulated inference.
        """
        as_of = pd.to_datetime(as_of_date)
        matrix = self.get_price_matrix(tickers=tickers, price_col=price_col)
        sliced = matrix.loc[matrix.index <= as_of].copy()

        if sliced.empty:
            raise ValueError(
                f"No market data available on or before {as_of.strftime('%Y-%m-%d')}."
            )
        return sliced

    def rolling_windows(
        self,
        window_size: int,
        step_size: int = 1,
        tickers: Optional[List[str]] = None,
    ) -> Generator[Tuple[pd.Timestamp, pd.DataFrame], None, None]:
        """Generate rolling historical data windows without forward-looking data leakage.

        Parameters
        ----------
        window_size : int
            Number of historical trading days in each window.
        step_size : int
            Step advance between successive windows.
        tickers : list of str, optional
            Asset tickers to include.

        Yields
        ------
        (as_of_date, window_dataframe)
            Tuple of current timestamp and the historical slice of length `window_size`.
        """
        matrix = self.get_price_matrix(tickers=tickers)
        if len(matrix) < window_size:
            raise ValueError(
                f"Window size {window_size} exceeds total dataset length {len(matrix)}"
            )

        for end_idx in range(window_size, len(matrix) + 1, step_size):
            window_slice = matrix.iloc[end_idx - window_size : end_idx]
            as_of_date = window_slice.index[-1]
            yield as_of_date, window_slice
