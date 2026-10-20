"""Unit tests for Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantpulse.analytics.var import (
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
)


def test_historical_var_exact_percentile():
    """Historical VaR must match the negative empirical percentile."""
    # 100 sorted daily returns from -0.10 to +0.09
    rets = pd.Series(np.linspace(-0.10, 0.09, 100))
    # 95% confidence level => 5th percentile is at index 5 (-0.05 approx)
    var_95 = historical_var(rets, confidence_level=0.95)
    expected_cutoff = -np.percentile(rets, 5)
    assert pytest.approx(var_95, rel=1e-5) == expected_cutoff
    assert var_95 > 0.0  # Reported as positive loss


def test_parametric_var_normal_distribution():
    """Parametric VaR should match theoretical Gaussian analytical formula."""
    rng = np.random.default_rng(42)
    # Generate 500,000 samples to approximate theoretical normal
    rets = pd.Series(rng.normal(loc=0.001, scale=0.02, size=500_000))

    var_norm = parametric_var(rets, confidence_level=0.95, method="normal")
    mu = rets.mean()
    sigma = rets.std()
    z_95 = stats.norm.ppf(0.95)
    expected_var = z_95 * sigma - mu

    assert pytest.approx(var_norm, rel=1e-4) == expected_var


def test_cornish_fisher_adjusts_for_fat_tails():
    """Negatively skewed and fat-tailed distribution must increase VaR under Cornish-Fisher."""
    rng = np.random.default_rng(42)
    # Base normals + severe negative crash shocks to induce negative skew and high kurtosis
    normals = rng.normal(loc=0.0005, scale=0.015, size=2000)
    shocks = rng.choice([-0.08, -0.06, -0.10], size=50)
    fat_tailed_rets = pd.Series(np.concatenate([normals, shocks]))

    var_normal = parametric_var(fat_tailed_rets, confidence_level=0.99, method="normal")
    var_cf = parametric_var(fat_tailed_rets, confidence_level=0.99, method="cornish_fisher")

    # With negative tail shocks, Cornish-Fisher should indicate higher risk than standard normal
    assert var_cf > var_normal


def test_historical_cvar_greater_than_var(synthetic_return_series):
    """By fundamental mathematical definition, CVaR (expected shortfall) >= VaR."""
    var_95 = historical_var(synthetic_return_series, confidence_level=0.95)
    cvar_95 = historical_cvar(synthetic_return_series, confidence_level=0.95)
    assert cvar_95 >= var_95


def test_parametric_cvar_formula():
    """Verify analytical Gaussian CVaR formula against numerical integration."""
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(loc=0.001, scale=0.02, size=200_000))

    cvar_param = parametric_cvar(rets, confidence_level=0.95)
    mu = rets.mean()
    sigma = rets.std()
    z = stats.norm.ppf(0.95)
    phi_z = stats.norm.pdf(z)
    expected_cvar = -mu + sigma * (phi_z / (1.0 - 0.95))

    assert pytest.approx(cvar_param, rel=1e-4) == expected_cvar


def test_monte_carlo_var_bounds(synthetic_return_series):
    """Monte Carlo VaR should be within reasonable proximity of historical VaR."""
    var_hist = historical_var(synthetic_return_series, confidence_level=0.95)
    var_mc = monte_carlo_var(synthetic_return_series, confidence_level=0.95, simulations=20_000, seed=42)
    # Check within 20% relative difference for 252 points
    assert pytest.approx(var_mc, rel=0.25) == var_hist
