"""Simple historical-volatility baselines for benchmarking machine learning models."""

from typing import Union
import numpy as np
import pandas as pd


class HistoricalVolatilityBaseline:
    """Non-parametric baseline forecasting future volatility using historical empirical observations.

    Supports:
    - "rolling": Uses the most recent rolling realized volatility (e.g., 21-day or 5-day window).
    - "expanding": Uses the expanding historical average volatility up to day t.
    """

    def __init__(self, method: str = "rolling", window_col: str = "vol_21d"):
        self.method = method
        self.window_col = window_col

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Generate baseline volatility predictions directly from historical feature observations.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Unscaled or original features containing rolling volatility columns.

        Returns
        -------
        np.ndarray of shape (N,)
            Predicted volatility values.
        """
        if self.window_col in feature_df.columns:
            series = feature_df[self.window_col]
        elif "vol_5d" in feature_df.columns:
            series = feature_df["vol_5d"]
        else:
            raise ValueError(f"Feature DataFrame does not contain rolling volatility column '{self.window_col}'.")

        if self.method == "rolling":
            # Forecast is simply the latest realized rolling volatility
            preds = series.values
        elif self.method == "expanding":
            preds = series.expanding(min_periods=5).mean().values
        else:
            raise ValueError(f"Unknown baseline method: {self.method}")

        return np.asarray(preds, dtype=np.float32)
