"""Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall) analytics."""

from typing import Union
import numpy as np
import pandas as pd
from scipy import stats

from quantpulse.core.config import DEFAULT_CONFIDENCE_LEVEL_VAR


def historical_var(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL_VAR,
) -> float:
    """Compute Historical Value at Risk (VaR) using empirical quantiles.

    Represents the expected maximum loss at the given confidence level.
    Expressed as a positive value (e.g., 0.023 indicates a 2.3% loss).

    Formula:
        VaR_alpha = - Quantile(returns, 1 - alpha)

    Parameters
    ----------
    returns : pd.Series
        Historical periodic returns.
    confidence_level : float, default 0.95
        Probability threshold (e.g., 0.95 or 0.99).

    Returns
    -------
    float
        VaR as a positive fraction of portfolio value.
    """
    if len(returns) == 0:
        return 0.0

    alpha = confidence_level
    cutoff_percentile = (1.0 - alpha) * 100.0
    quantile_val = float(np.percentile(returns.dropna(), cutoff_percentile))
    # Return as positive loss magnitude
    return float(-quantile_val)


def parametric_var(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL_VAR,
    method: str = "normal",
) -> float:
    """Compute Parametric Value at Risk (Gaussian or Cornish-Fisher modified).

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    confidence_level : float, default 0.95
        Confidence level.
    method : {"normal", "cornish_fisher"}, default "normal"
        - "normal": Assumes Gaussian distribution: VaR = z_alpha * sigma - mu
        - "cornish_fisher": Adjusts normal critical value for observed skewness & kurtosis.

    Returns
    -------
    float
        VaR as a positive fraction.
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        return 0.0

    mu = float(clean_rets.mean())
    sigma = float(clean_rets.std(ddof=1))
    if sigma <= 1e-8:
        return 0.0

    p = 1.0 - confidence_level  # e.g., 0.05 for 95% confidence
    z_p = float(stats.norm.ppf(p))  # negative value, e.g. -1.6449 for 95%

    if method == "cornish_fisher":
        s = float(stats.skew(clean_rets))
        # stats.kurtosis calculates excess kurtosis (Fisher definition)
        k = float(stats.kurtosis(clean_rets))

        # Cornish-Fisher expansion on the lower quantile z_p
        z_cf = (
            z_p
            + (z_p**2 - 1.0) * s / 6.0
            + (z_p**3 - 3.0 * z_p) * k / 24.0
            - (2.0 * z_p**3 - 5.0 * z_p) * (s**2) / 36.0
        )
        z_p = z_cf

    # Return threshold is mu + z_p * sigma
    # VaR as positive loss: - (mu + z_p * sigma) = -z_p * sigma - mu
    var_val = -z_p * sigma - mu
    return float(var_val)


def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL_VAR,
    simulations: int = 10_000,
    seed: int = 42,
) -> float:
    """Estimate Value at Risk using Monte Carlo simulations sampled from fitted student-t distribution.

    Captures fat tails more effectively than standard Gaussian assumption.
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 4:
        return 0.0

    # Fit a Student-t distribution (df, loc, scale)
    df, loc, scale = stats.t.fit(clean_rets)
    rng = np.random.default_rng(seed)
    simulated_rets = stats.t.rvs(df, loc=loc, scale=scale, size=simulations, random_state=rng)

    cutoff = (1.0 - confidence_level) * 100.0
    sim_quantile = float(np.percentile(simulated_rets, cutoff))
    return float(-sim_quantile)


def historical_cvar(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL_VAR,
) -> float:
    """Compute Historical Conditional Value at Risk (CVaR / Expected Shortfall).

    Average loss incurred when returns breach the VaR threshold.

    Formula:
        CVaR_alpha = - E[R | R <= -VaR_alpha]

    Parameters
    ----------
    returns : pd.Series
        Historical periodic returns.
    confidence_level : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        CVaR as a positive fraction.
    """
    clean_rets = returns.dropna()
    if len(clean_rets) == 0:
        return 0.0

    var_threshold = historical_var(clean_rets, confidence_level=confidence_level)
    # Filter returns strictly worse than -VaR threshold
    tail_losses = clean_rets[clean_rets <= -var_threshold]

    if len(tail_losses) == 0:
        return var_threshold

    return float(-tail_losses.mean())


def parametric_cvar(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL_VAR,
) -> float:
    """Calculate Analytical Gaussian Conditional Value at Risk (Expected Shortfall).

    Closed-form solution for Normal distribution:
        CVaR_alpha = -mu + sigma * [ phi(z_alpha) / (1 - alpha) ]

    where phi is the standard normal probability density function (PDF),
    and z_alpha is the normal inverse CDF at confidence_level.
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        return 0.0

    mu = float(clean_rets.mean())
    sigma = float(clean_rets.std(ddof=1))
    if sigma <= 1e-8:
        return 0.0

    alpha = confidence_level
    z = float(stats.norm.ppf(alpha))
    phi_z = float(stats.norm.pdf(z))

    es = -mu + sigma * (phi_z / (1.0 - alpha))
    return float(es)
