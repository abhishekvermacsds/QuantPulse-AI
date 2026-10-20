"""Unit tests for financial risk metrics: Volatility, Downside Deviation, Sharpe, Sortino, Drawdowns, Calmar."""

import numpy as np
import pandas as pd
import pytest

from quantpulse.analytics.risk import (
    annualized_volatility,
    calmar_ratio,
    downside_deviation,
    drawdown_series,
    max_drawdown,
    rolling_risk_metrics,
    sharpe_ratio,
    sortino_ratio,
)


def test_annualized_volatility_known_series():
    """Verify annualized volatility scales correctly by sqrt(252)."""
    # Daily returns alternating between +0.01 and -0.01
    rets = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01] * 40)
    expected_vol = rets.std(ddof=1) * np.sqrt(252)
    calculated_vol = annualized_volatility(rets, periods_per_year=252)
    assert pytest.approx(calculated_vol, rel=1e-5) == expected_vol


def test_annualized_volatility_flat_series(constant_price_series):
    """Flat returns should yield zero volatility."""
    rets = constant_price_series.pct_change().dropna()
    assert annualized_volatility(rets) == 0.0


def test_downside_deviation_only_penalizes_downside():
    """Downside deviation should only account for negative deviations from target."""
    # Series with strictly positive returns vs target 0.0
    all_positive = pd.Series([0.01, 0.02, 0.015, 0.03, 0.005])
    down_dev = downside_deviation(all_positive, target_return=0.0)
    assert down_dev == 0.0


def test_sharpe_ratio_behavior(synthetic_return_series):
    """Verify Sharpe Ratio against explicit definition."""
    rf = 0.04
    sr = sharpe_ratio(synthetic_return_series, risk_free_rate=rf)

    wealth = (1.0 + synthetic_return_series).prod()
    n = len(synthetic_return_series)
    cagr = (wealth ** (252 / n)) - 1.0
    vol = synthetic_return_series.std(ddof=1) * np.sqrt(252)
    expected_sr = (cagr - rf) / vol

    assert pytest.approx(sr, rel=1e-5) == expected_sr


def test_sortino_ratio_greater_than_sharpe_with_positive_skew():
    """When negative returns are infrequent, Sortino should be substantially higher than Sharpe."""
    # Skewed series: mostly positive, few small dips
    rets = pd.Series([0.02, 0.03, 0.01, -0.002, 0.025, 0.015, -0.001, 0.02] * 30)
    sr = sharpe_ratio(rets, risk_free_rate=0.02)
    sortino = sortino_ratio(rets, risk_free_rate=0.02)
    assert sortino > sr


def test_drawdown_series_and_max_drawdown():
    """Verify underwater curve and MDD calculation with an exact scenario.

    Path:
    Day 0: P0 = 100
    Day 1: +20% -> 120 (Peak)
    Day 2: -25% -> 90  (Trough relative to peak: (90 - 120) / 120 = -0.25)
    Day 3: +10% -> 99  (Drawdown relative to peak: (99 - 120) / 120 = -0.175)
    Day 4: +30% -> 128.7 (New Peak, Drawdown = 0.0)
    """
    rets = pd.Series([0.20, -0.25, 0.10, 0.30])
    dd = drawdown_series(rets)

    expected_drawdowns = [
        0.0,                    # (1.20 - 1.20) / 1.20 = 0.0
        -0.25,                  # (0.90 - 1.20) / 1.20 = -0.25 (-25%)
        (99.0 - 120.0) / 120.0, # -0.175 (-17.5%)
        0.0,                    # new peak
    ]
    np.testing.assert_allclose(dd.values, expected_drawdowns, rtol=1e-5)

    mdd = max_drawdown(rets)
    assert pytest.approx(mdd, rel=1e-5) == -0.25


def test_calmar_ratio():
    """Verify Calmar Ratio: CAGR / abs(MDD)."""
    rets = pd.Series([0.05, -0.02, 0.04, -0.01] * 20)
    calmar = calmar_ratio(rets, periods_per_year=80)
    mdd = max_drawdown(rets)
    cagr = ((1.0 + rets).prod() ** (80 / len(rets))) - 1.0
    expected_calmar = cagr / abs(mdd)
    assert pytest.approx(calmar, rel=1e-5) == expected_calmar


def test_rolling_risk_metrics_no_lookahead(synthetic_return_series):
    """Ensure rolling metrics respect backwards windowing without look-ahead bias."""
    window = 30
    rolling_df = rolling_risk_metrics(synthetic_return_series, window=window)

    # First window-1 rows must be NaN
    assert rolling_df.iloc[: window - 1]["rolling_volatility"].isna().all()
    # From index window-1 onwards, metrics should be populated
    assert not rolling_df.iloc[window - 1:]["rolling_volatility"].isna().any()

    # Verify a specific rolling window manually
    test_slice = synthetic_return_series.iloc[0:window]
    expected_vol = annualized_volatility(test_slice)
    actual_vol = rolling_df.iloc[window - 1]["rolling_volatility"]
    assert pytest.approx(actual_vol, rel=1e-5) == expected_vol
