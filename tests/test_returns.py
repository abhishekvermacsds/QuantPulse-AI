"""Unit tests for returns calculations."""

import numpy as np
import pandas as pd
import pytest

from quantpulse.analytics.returns import (
    annualize_returns,
    calculate_cumulative_returns,
    calculate_returns,
)


def test_simple_returns_calculation(known_price_series):
    """Test standard simple percentage returns against known manual calculation."""
    # Prices: [100.0, 80.0, 120.0, 90.0]
    # Simple Returns:
    # (80 - 100) / 100 = -0.20
    # (120 - 80) / 80  = +0.50
    # (90 - 120) / 120 = -0.25
    expected = [-0.20, 0.50, -0.25]
    result = calculate_returns(known_price_series, method="simple", drop_first=True)

    assert len(result) == 3
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def test_log_returns_calculation(known_price_series):
    """Test log returns: r_t = ln(P_t / P_{t-1})."""
    # Prices: [100.0, 80.0, 120.0, 90.0]
    expected = [
        np.log(80.0 / 100.0),
        np.log(120.0 / 80.0),
        np.log(90.0 / 120.0),
    ]
    result = calculate_returns(known_price_series, method="log", drop_first=True)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def test_returns_drop_first_false(known_price_series):
    """Ensure drop_first=False preserves length with first item as NaN."""
    result = calculate_returns(known_price_series, method="simple", drop_first=False)
    assert len(result) == len(known_price_series)
    assert np.isnan(result.iloc[0])


def test_cumulative_returns():
    """Verify cumulative returns calculation."""
    dates = pd.date_range("2023-01-01", periods=3)
    rets = pd.Series([0.10, -0.05, 0.20], index=dates)

    # Cumulative:
    # Day 1: (1 + 0.10) - 1 = 0.10
    # Day 2: (1 + 0.10) * (1 - 0.05) - 1 = 1.10 * 0.95 - 1 = 1.045 - 1 = 0.045
    # Day 3: 1.045 * 1.20 - 1 = 1.254 - 1 = 0.254
    expected = [0.10, 0.045, 0.254]
    cum_rets = calculate_cumulative_returns(rets, method="simple")
    np.testing.assert_allclose(cum_rets.values, expected, rtol=1e-5)


def test_annualized_returns_cagr():
    """Test Compound Annual Growth Rate (CAGR) formula."""
    # 252 periods of 0.1% daily return
    rets = pd.Series([0.001] * 252)
    # Total wealth: (1 + 0.001)^252
    cagr = annualize_returns(rets, periods_per_year=252, compound=True)
    expected_cagr = (1.001 ** 252) - 1.0
    assert pytest.approx(cagr, rel=1e-5) == expected_cagr


def test_annualized_returns_arithmetic():
    """Test arithmetic annualization: mean(R) * 252."""
    rets = pd.Series([0.001] * 252)
    arithmetic = annualize_returns(rets, periods_per_year=252, compound=False)
    assert pytest.approx(arithmetic, rel=1e-5) == 0.001 * 252


def test_invalid_price_raises_error():
    """Non-positive prices must be rejected."""
    bad_prices = pd.Series([100.0, -5.0, 50.0])
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_returns(bad_prices)


def test_invalid_method_raises_error(known_price_series):
    """Unsupported calculation method must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported method"):
        calculate_returns(known_price_series, method="geometric")
