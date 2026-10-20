"""Shared fixtures for testing QuantPulse AI analytics and data loader."""

import numpy as np
import pandas as pd
import pytest

from quantpulse.core.config import DEFAULT_SAMPLE_CSV
from quantpulse.data.loader import MarketDataLoader


@pytest.fixture
def sample_csv_path():
    """Path to the generated sample market dataset."""
    return DEFAULT_SAMPLE_CSV


@pytest.fixture
def sample_loader(sample_csv_path):
    """Initialized MarketDataLoader instance."""
    return MarketDataLoader(data_path=sample_csv_path)


@pytest.fixture
def known_price_series():
    """Deterministic price series for analytical verification.

    Starts at 100, drops to 80, rises to 120, drops to 90.
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    prices = [100.0, 80.0, 120.0, 90.0]
    return pd.Series(prices, index=dates, name="TEST_ASSET")


@pytest.fixture
def constant_price_series():
    """Deterministic flat price series (zero return)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.Series(100.0, index=dates, name="FLAT_ASSET")


@pytest.fixture
def synthetic_return_series():
    """Normally distributed return series with known mean and variance."""
    rng = np.random.default_rng(12345)
    dates = pd.date_range("2022-01-01", periods=252, freq="B")
    # Daily mean return approx 0.0005 (~12.6% ann), daily std 0.01 (~15.8% ann vol)
    returns = rng.normal(loc=0.0005, scale=0.01, size=len(dates))
    return pd.Series(returns, index=dates, name="SIM_STOCK")


@pytest.fixture
def multi_asset_price_df():
    """Small multi-asset DataFrame for testing matrix-level vectorization."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    data = {
        "ASSET_A": [100.0, 105.0, 102.0, 108.0, 110.0],
        "ASSET_B": [50.0, 48.0, 49.0, 47.0, 46.0],
    }
    return pd.DataFrame(data, index=dates)
