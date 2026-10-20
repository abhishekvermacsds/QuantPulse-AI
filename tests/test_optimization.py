"""Unit tests for portfolio optimization, diversification metrics, and allocation comparisons."""

import numpy as np
import pandas as pd
import pytest

from quantpulse.optimization import (
    FrontierPoint,
    OptimizationResult,
    PortfolioOptimizer,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_expected_returns,
    calculate_portfolio_returns,
    compare_portfolios,
    diversification_ratio,
    diversification_report,
    effective_number_of_assets,
    evaluate_portfolio_metrics,
    herfindahl_index,
    out_of_sample_validation,
    validate_covariance_matrix,
)


@pytest.fixture
def test_returns_df():
    """Deterministic return DataFrame for 4 assets over 200 days."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    tickers = ["TECH", "ENERGY", "FINANCE", "BONDS"]
    # Mean returns and volatilities
    mus = np.array([0.15, 0.10, 0.12, 0.04]) / 252.0
    sigmas = np.array([0.22, 0.25, 0.18, 0.08]) / np.sqrt(252.0)

    # Correlation matrix
    corr = np.array([
        [1.0, 0.3, 0.5, -0.1],
        [0.3, 1.0, 0.4, -0.2],
        [0.5, 0.4, 1.0, -0.15],
        [-0.1, -0.2, -0.15, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    normals = rng.standard_normal(size=(200, 4)) @ L.T
    rets = mus + sigmas * normals
    return pd.DataFrame(rets, index=dates, columns=tickers)


def test_expected_returns_and_covariance_methods(test_returns_df):
    """Test expected returns and covariance estimation methods."""
    mu_mean = calculate_expected_returns(test_returns_df, method="mean")
    assert len(mu_mean) == 4
    assert not mu_mean.isna().any()

    cov_sample = calculate_covariance_matrix(test_returns_df, method="sample")
    cov_lw = calculate_covariance_matrix(test_returns_df, method="ledoit_wolf")

    assert cov_sample.shape == (4, 4)
    assert cov_lw.shape == (4, 4)
    assert np.allclose(cov_sample.values, cov_sample.values.T)
    assert np.allclose(cov_lw.values, cov_lw.values.T)


def test_validate_covariance_matrix_repairs_small_negative_eigenvals():
    """Verify spectral projection repairs non-PSD matrices and rejects when fix_non_psd=False."""
    # Create a symmetric matrix with a small negative eigenvalue
    A = np.array([
        [1.0, 0.999, 0.999],
        [0.999, 1.0, 0.999],
        [0.999, 0.999, 1.0],
    ])
    # Perturb to create slightly negative eigenvalue
    A[0, 1] = 1.01
    A[1, 0] = 1.01
    df_non_psd = pd.DataFrame(A, index=["A", "B", "C"], columns=["A", "B", "C"])

    # Without fix, it must raise ValueError
    with pytest.raises(ValueError, match="non-positive semi-definite"):
        validate_covariance_matrix(df_non_psd, fix_non_psd=False)

    # With fix, it projects onto PSD cone
    df_fixed = validate_covariance_matrix(df_non_psd, fix_non_psd=True)
    eigenvals = np.linalg.eigvalsh(df_fixed.values)
    assert (eigenvals >= 0.0).all()


def test_diversification_metrics(test_returns_df):
    """Verify Diversification Ratio, HHI, and Effective Assets calculations."""
    cov = calculate_covariance_matrix(test_returns_df)

    # 1. Equal-weighted weights
    w_eq = pd.Series([0.25, 0.25, 0.25, 0.25], index=test_returns_df.columns)
    hhi_eq = herfindahl_index(w_eq)
    n_eff_eq = effective_number_of_assets(w_eq)

    # Equal weights for 4 assets: HHI = 4 * (0.25^2) = 0.25, N_eff = 1 / 0.25 = 4.0
    assert pytest.approx(hhi_eq, rel=1e-5) == 0.25
    assert pytest.approx(n_eff_eq, rel=1e-5) == 4.0

    dr_eq = diversification_ratio(w_eq, cov)
    # Diversification ratio must be >= 1.0 for non-negative weights
    assert dr_eq >= 1.0

    # 2. Concentrated 100% single asset
    w_conc = pd.Series([1.0, 0.0, 0.0, 0.0], index=test_returns_df.columns)
    assert pytest.approx(herfindahl_index(w_conc), rel=1e-5) == 1.0
    assert pytest.approx(effective_number_of_assets(w_conc), rel=1e-5) == 1.0
    assert pytest.approx(diversification_ratio(w_conc, cov), rel=1e-5) == 1.0

    # 3. Comprehensive report
    rep = diversification_report(w_eq, cov, returns=test_returns_df)
    assert rep["total_assets"] == 4
    assert rep["active_assets_count"] == 4
    assert "avg_active_correlation" in rep


def test_equal_weight_optimization(test_returns_df):
    """Test equal-weighted benchmark generation."""
    opt = PortfolioOptimizer.from_returns(test_returns_df)
    res = opt.optimize_equal_weight()

    assert res.objective == "equal_weight"
    assert res.success is True
    assert len(res.weights) == 4
    for w in res.weights.values():
        assert pytest.approx(w, rel=1e-4) == 0.25
    assert pytest.approx(sum(res.weights.values()), rel=1e-4) == 1.0


def test_minimum_volatility_optimizes_variance(test_returns_df):
    """Minimum volatility portfolio must achieve lower volatility than equal weighting."""
    opt = PortfolioOptimizer.from_returns(test_returns_df)
    eq_res = opt.optimize_equal_weight()
    min_vol_res = opt.optimize_minimum_volatility()

    assert min_vol_res.success is True
    assert min_vol_res.volatility <= eq_res.volatility + 1e-6
    assert pytest.approx(sum(min_vol_res.weights.values()), rel=1e-4) == 1.0
    for w in min_vol_res.weights.values():
        assert w >= -1e-5  # long-only


def test_maximum_sharpe_optimizes_risk_adjusted_return(test_returns_df):
    """Maximum Sharpe portfolio must achieve higher or equal Sharpe ratio than equal weighting."""
    opt = PortfolioOptimizer.from_returns(test_returns_df, risk_free_rate=0.03)
    eq_res = opt.optimize_equal_weight()
    max_sharpe_res = opt.optimize_max_sharpe()

    assert max_sharpe_res.success is True
    assert max_sharpe_res.sharpe_ratio >= eq_res.sharpe_ratio - 1e-4
    assert pytest.approx(sum(max_sharpe_res.weights.values()), rel=1e-4) == 1.0
    for w in max_sharpe_res.weights.values():
        assert w >= -1e-5


def test_max_weight_constraint_enforced(test_returns_df):
    """Test that max_weight constraint caps every asset's allocation."""
    max_weight_limit = 0.35
    opt = PortfolioOptimizer.from_returns(test_returns_df, max_weight=max_weight_limit)

    min_vol = opt.optimize_minimum_volatility()
    for ticker, w in min_vol.weights.items():
        assert w <= max_weight_limit + 1e-4

    max_sharpe = opt.optimize_max_sharpe()
    for ticker, w in max_sharpe.weights.items():
        assert w <= max_weight_limit + 1e-4


def test_infeasible_max_weight_raises_error(test_returns_df):
    """When max_weight < 1/N, an exception must be raised immediately."""
    # 4 assets => 1/4 = 0.25. max_weight=0.20 is mathematically infeasible.
    with pytest.raises(ValueError, match="Infeasible constraint"):
        PortfolioOptimizer.from_returns(test_returns_df, max_weight=0.20)


def test_max_sharpe_fallback_with_negative_excess_returns():
    """When all asset expected returns are <= Rf, max Sharpe falls back gracefully to min vol."""
    # Create returns with negative drift
    rng = np.random.default_rng(10)
    rets = pd.DataFrame(
        rng.normal(loc=-0.001, scale=0.01, size=(100, 3)),
        columns=["A", "B", "C"],
        index=pd.date_range("2023-01-01", periods=100),
    )
    opt = PortfolioOptimizer.from_returns(rets, risk_free_rate=0.05)
    res = opt.optimize_max_sharpe()

    assert res.success is True
    assert "fell back" in res.message.lower()
    assert pytest.approx(sum(res.weights.values()), rel=1e-4) == 1.0


def test_efficient_frontier_generation(test_returns_df):
    """Verify points along the efficient frontier curve."""
    opt = PortfolioOptimizer.from_returns(test_returns_df)
    frontier = opt.compute_efficient_frontier(points=10)

    assert len(frontier) >= 2
    for pt in frontier:
        assert isinstance(pt, FrontierPoint)
        assert pt.volatility > 0.0
        assert pytest.approx(sum(pt.weights.values()), rel=1e-4) == 1.0


def test_compare_portfolios(test_returns_df):
    """Test side-by-side portfolio risk and return comparison."""
    opt = PortfolioOptimizer.from_returns(test_returns_df)
    eq = opt.optimize_equal_weight()
    min_vol = opt.optimize_minimum_volatility()
    max_sharpe = opt.optimize_max_sharpe()

    portfolios = {
        "Equal_Weight": eq.weights_series,
        "Min_Vol": min_vol.weights_series,
        "Max_Sharpe": max_sharpe.weights_series,
    }

    comp_df = compare_portfolios(test_returns_df, portfolios)
    assert isinstance(comp_df, pd.DataFrame)
    assert set(comp_df.columns) == {"Equal_Weight", "Min_Vol", "Max_Sharpe"}
    assert "Sharpe" in comp_df.index
    assert "CAGR" in comp_df.index
    assert "Max_Drawdown" in comp_df.index
    assert "Diversification_Ratio" in comp_df.index


def test_out_of_sample_validation_no_lookahead(test_returns_df):
    """Verify that out-of-sample backtesting does not leak future data."""
    res = out_of_sample_validation(test_returns_df, train_ratio=0.70)

    assert "in_sample_metrics" in res
    assert "out_of_sample_metrics" in res
    assert "allocations" in res

    in_sample = res["in_sample_metrics"]
    out_of_sample = res["out_of_sample_metrics"]

    assert set(in_sample.columns) == {"Equal_Weight", "Min_Volatility", "Max_Sharpe"}
    assert set(out_of_sample.columns) == {"Equal_Weight", "Min_Volatility", "Max_Sharpe"}
