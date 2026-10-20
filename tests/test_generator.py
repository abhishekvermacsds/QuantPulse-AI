"""Unit tests for synthetic market data generator."""

import numpy as np
import pytest

from quantpulse.data.generator import (
    DEFAULT_ASSET_UNIVERSE,
    generate_synthetic_market_data,
    save_sample_market_data,
)


def test_generate_synthetic_market_data_shape_and_columns():
    """Verify generated dataset contains correct dimensions and columns."""
    periods = 20
    df = generate_synthetic_market_data(periods=periods, seed=123)

    num_assets = len(DEFAULT_ASSET_UNIVERSE)
    expected_rows = periods * num_assets

    assert len(df) == expected_rows
    assert set(df.columns) == {
        "Date", "Ticker", "Open", "High", "Low", "Close", "Adj_Close", "Volume"
    }
    assert (df["Open"] > 0).all()
    assert (df["High"] >= df["Low"]).all()
    assert (df["Volume"] >= 0).all()


def test_generator_mismatched_correlation_matrix_raises():
    """Correlation matrix of improper shape must raise ValueError."""
    bad_corr = np.eye(2)
    with pytest.raises(ValueError, match="Correlation matrix shape"):
        generate_synthetic_market_data(correlation_matrix=bad_corr)


def test_save_sample_market_data_persists_to_file(tmp_path):
    """Test saving generated sample data to disk."""
    dest = tmp_path / "test_sample.csv"
    saved_path = save_sample_market_data(destination=dest, seed=999)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
