"""Feature engineering, look-ahead safe chronological splitting, and PyTorch dataset generation."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from quantpulse.core.config import TRADING_DAYS_PER_YEAR


class RobustFeatureScaler:
    """Standard z-score scaler fitted strictly on training observations to eliminate data leakage."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, data: pd.DataFrame) -> "RobustFeatureScaler":
        """Compute mean and standard deviation strictly on the training partition."""
        self.feature_names = list(data.columns)
        values = data.values.astype(np.float32)
        self.means = np.nanmean(values, axis=0)
        self.stds = np.nanstd(values, axis=0)
        # Prevent division by zero on static features
        self.stds[self.stds < self.epsilon] = 1.0
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Apply z-score normalization using fitted training statistics."""
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        values = data.values.astype(np.float32)
        return (values - self.means) / self.stds

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)

    def to_dict(self) -> Dict:
        """Serialize scaler parameters for model persistence."""
        return {
            "means": self.means.tolist() if self.means is not None else None,
            "stds": self.stds.tolist() if self.stds is not None else None,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RobustFeatureScaler":
        """Reconstruct scaler from dictionary."""
        scaler = cls()
        scaler.means = np.array(data["means"], dtype=np.float32)
        scaler.stds = np.array(data["stds"], dtype=np.float32)
        scaler.feature_names = data["feature_names"]
        return scaler


class VolatilityFeaturePipeline:
    """Constructs lagged return and rolling volatility features with forward target labels."""

    def __init__(
        self,
        forward_horizon: int = 5,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ):
        self.forward_horizon = forward_horizon
        self.periods_per_year = periods_per_year

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate feature matrix and forward realized volatility target from market bars.

        Parameters
        ----------
        df : pd.DataFrame
            Market data with columns: ['Close'] (and optionally ['Open', 'High', 'Low', 'Volume']).

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (Feature DataFrame, Target Series representing annualized forward realized volatility)
        """
        data = df.copy()
        if "Close" not in data.columns and "Adj_Close" in data.columns:
            data["Close"] = data["Adj_Close"]
        if "Close" not in data.columns:
            raise ValueError("Input DataFrame must contain 'Close' or 'Adj_Close'.")

        # 1. Base returns
        data["return_1d"] = data["Close"].pct_change()
        data["return_lag1"] = data["return_1d"].shift(1)
        data["return_lag2"] = data["return_1d"].shift(2)
        data["return_lag5"] = data["return_1d"].shift(4)

        # 2. Rolling realized volatilities (annualized, strictly backwards)
        scale = np.sqrt(self.periods_per_year)
        data["vol_5d"] = data["return_1d"].rolling(5).std() * scale
        data["vol_10d"] = data["return_1d"].rolling(10).std() * scale
        data["vol_21d"] = data["return_1d"].rolling(21).std() * scale

        # 3. High-Low range (Parkinson proxy)
        if "High" in data.columns and "Low" in data.columns:
            data["hl_spread"] = np.log(np.maximum(data["High"], 1e-4) / np.maximum(data["Low"], 1e-4))
        else:
            data["hl_spread"] = np.abs(data["return_1d"]) * 2.0

        # 4. Volume relative shock
        if "Volume" in data.columns:
            vol_sma = data["Volume"].rolling(20).mean().replace(0, np.nan)
            data["vol_shock"] = (data["Volume"] / vol_sma).fillna(1.0)
        else:
            data["vol_shock"] = 1.0

        # 5. Forward target: Realized volatility over next H days
        # Target at day t: std(R_{t+1 ... t+H}) * sqrt(252)
        # Using backwards rolling on shifted returns
        future_returns = data["return_1d"].shift(-self.forward_horizon)
        # Rolling forward window
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.forward_horizon)
        target = data["return_1d"].rolling(window=indexer).std() * scale
        # Shift target by 1 so target at t uses returns from t+1 onwards
        data["target_vol"] = target.shift(-1)

        feature_cols = [
            "return_1d",
            "return_lag1",
            "return_lag2",
            "return_lag5",
            "vol_5d",
            "vol_10d",
            "vol_21d",
            "hl_spread",
            "vol_shock",
        ]

        # Clean NaN rows created by lags and forward horizons
        valid_mask = data[feature_cols + ["target_vol"]].notna().all(axis=1)
        clean_features = data.loc[valid_mask, feature_cols].copy()
        clean_target = data.loc[valid_mask, "target_vol"].copy()

        return clean_features, clean_target


def create_chronological_splits(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """Split time-series features and targets chronologically to prevent look-ahead leakage.

    Split partitions:
    - Train: [0, train_end]
    - Validation: [train_end, val_end]
    - Test: [val_end, total]
    """
    n = len(features)
    if n < 40:
        raise ValueError(f"Insufficient samples ({n}) for train/val/test splits.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_X, train_y = features.iloc[:train_end], target.iloc[:train_end]
    val_X, val_y = features.iloc[train_end:val_end], target.iloc[train_end:val_end]
    test_X, test_y = features.iloc[val_end:], target.iloc[val_end:]

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


class VolatilitySequenceDataset(Dataset):
    """PyTorch Dataset that extracts sliding sequential historical windows for LSTM training."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 20,
    ):
        """
        Parameters
        ----------
        features : np.ndarray of shape (N, num_features)
            Scaled feature values.
        targets : np.ndarray of shape (N,)
            Target volatility values.
        sequence_length : int, default 20
            Number of historical timesteps per sequence window (e.g. 20 trading days).
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have identical length.")
        if len(features) < sequence_length:
            raise ValueError(f"Dataset length ({len(features)}) < sequence_length ({sequence_length}).")

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sequence: idx to idx + sequence_length
        end_idx = idx + self.sequence_length
        x_seq = self.features[idx:end_idx]
        # Target corresponds to the forecast made at the final step of the sequence
        y = self.targets[end_idx - 1]
        return x_seq, y.unsqueeze(0)
