"""Unit tests for MarketDataLoader and look-ahead bias prevention."""

from pathlib import Path
import pandas as pd
import pytest

from quantpulse.data.loader import MarketDataLoader


def test_market_data_loader_basic(sample_loader):
    """Test loading valid sample dataset."""
    df = sample_loader.load()
    assert not df.empty
    assert "Date" in df.columns
    assert "Ticker" in df.columns
    assert "Adj_Close" in df.columns
    assert df["Date"].is_monotonic_increasing


def test_price_matrix_pivot(sample_loader):
    """Test pivoted wide matrix format."""
    matrix = sample_loader.get_price_matrix()
    assert isinstance(matrix, pd.DataFrame)
    assert set(matrix.columns) == {"AAPL", "MSFT", "JPM", "XOM", "SPY", "TLT"}
    assert matrix.index.is_monotonic_increasing
    assert not matrix.isna().any().any()


def test_get_point_in_time_prevents_lookahead(sample_loader):
    """Ensure get_point_in_time strictly excludes future dates."""
    as_of = "2022-06-30"
    pit_data = sample_loader.get_point_in_time(as_of_date=as_of)

    # Max date in sliced data must be <= as_of
    assert pit_data.index.max() <= pd.to_datetime(as_of)
    # Ensure there are no dates beyond as_of
    assert len(pit_data.loc[pit_data.index > pd.to_datetime(as_of)]) == 0


def test_rolling_windows_chronology(sample_loader):
    """Verify rolling_windows generator enforces backwards-looking slices."""
    window_size = 50
    step_size = 10
    windows = list(sample_loader.rolling_windows(window_size=window_size, step_size=step_size))

    assert len(windows) > 0
    prev_as_of = None

    for as_of_date, window_df in windows:
        assert len(window_df) == window_size
        assert window_df.index[-1] == as_of_date
        assert window_df.index.is_monotonic_increasing
        if prev_as_of is not None:
            assert as_of_date > prev_as_of
        prev_as_of = as_of_date


def test_data_loader_detects_missing_columns(tmp_path):
    """Loader must raise ValueError if required columns are missing."""
    bad_csv = tmp_path / "bad_data.csv"
    bad_df = pd.DataFrame({"Date": ["2023-01-01"], "Ticker": ["AAPL"]})
    bad_df.to_csv(bad_csv, index=False)

    loader = MarketDataLoader(data_path=bad_csv)
    with pytest.raises(ValueError, match="missing required columns"):
        loader.load()


def test_data_loader_detects_non_positive_prices(tmp_path):
    """Loader must reject negative or zero prices."""
    invalid_csv = tmp_path / "invalid_price.csv"
    invalid_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Ticker": ["AAPL", "AAPL"],
        "Open": [100.0, 100.0],
        "High": [105.0, 105.0],
        "Low": [95.0, 95.0],
        "Close": [100.0, -10.0],
        "Adj_Close": [100.0, -10.0],
        "Volume": [1000, 1000],
    })
    invalid_df.to_csv(invalid_csv, index=False)

    loader = MarketDataLoader(data_path=invalid_csv)
    with pytest.raises(ValueError, match="non-positive"):
        loader.load()
