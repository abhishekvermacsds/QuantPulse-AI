"""Asset correlation and portfolio diversification analytics."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the correlation matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset periodic returns.

    Returns
    -------
    pd.DataFrame
        Pairwise correlation matrix.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Input 'returns' must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("Input returns DataFrame cannot be empty.")

    clean_rets = returns.dropna()
    return clean_rets.corr()


def diversification_ratio(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> float:
    """Calculate the Diversification Ratio (DR) of a portfolio (Choueifaty & Coignard, 2008).

    Formula:
        DR(w) = sum(w_i * sigma_i) / sqrt(w.T @ Sigma @ w)

    Measures the ratio of the weighted average asset volatilities to the diversified portfolio volatility.
    DR >= 1.0, where higher values indicate stronger diversification benefits.

    Parameters
    ----------
    weights : pd.Series
        Asset allocation weights summing to 1.0.
    cov_matrix : pd.DataFrame
        Annualized asset covariance matrix.

    Returns
    -------
    float
        Diversification Ratio >= 1.0.
    """
    # Align weights with covariance index
    common_assets = [a for a in cov_matrix.index if a in weights.index]
    if not common_assets:
        raise ValueError("No matching assets found between weights and covariance matrix.")

    w = weights[common_assets].values.astype(float)
    sigma_mat = cov_matrix.loc[common_assets, common_assets].values.astype(float)

    # Individual annualized asset volatilities: sqrt(diag(Sigma))
    individual_vols = np.sqrt(np.maximum(np.diag(sigma_mat), 0.0))
    weighted_vol = float(np.dot(w, individual_vols))

    # Portfolio volatility: sqrt(w.T @ Sigma @ w)
    port_var = float(w @ sigma_mat @ w)
    if port_var <= 1e-12:
        return 1.0

    port_vol = np.sqrt(port_var)
    return float(weighted_vol / port_vol)


def herfindahl_index(weights: pd.Series) -> float:
    """Calculate the Herfindahl-Hirschman Index (HHI) of portfolio weights.

    Formula:
        HHI = sum(w_i^2)

    Measures portfolio concentration:
    - 1 / N indicates perfect equal-weighted diversification.
    - 1.0 indicates 100% concentration in a single asset.
    """
    w = weights.values.astype(float)
    return float(np.sum(w**2))


def effective_number_of_assets(weights: pd.Series) -> float:
    """Calculate the Effective Number of Assets (N_eff = 1 / HHI).

    Equal weighting across N assets gives N_eff = N.
    A single-asset concentration gives N_eff = 1.
    """
    hhi = herfindahl_index(weights)
    if hhi <= 1e-12:
        return float(len(weights))
    return float(1.0 / hhi)


def diversification_report(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive portfolio diversification breakdown."""
    dr = diversification_ratio(weights, cov_matrix)
    hhi = herfindahl_index(weights)
    n_eff = effective_number_of_assets(weights)

    active_weights = weights[weights > 1e-4]
    zero_weights = weights[weights <= 1e-4]

    report = {
        "diversification_ratio": round(dr, 4),
        "herfindahl_index": round(hhi, 4),
        "effective_number_of_assets": round(n_eff, 2),
        "total_assets": len(weights),
        "active_assets_count": len(active_weights),
        "zero_weight_assets": list(zero_weights.index),
        "max_asset_weight": round(float(weights.max()), 4),
        "min_active_weight": round(float(active_weights.min()), 4) if not active_weights.empty else 0.0,
    }

    if returns is not None:
        corr = calculate_correlation_matrix(returns)
        # Average pairwise correlation of active assets
        if len(active_weights) > 1:
            active_tickers = list(active_weights.index)
            sub_corr = corr.loc[active_tickers, active_tickers]
            n = len(active_tickers)
            # Upper triangle off-diagonal elements
            triu_indices = np.triu_indices(n, k=1)
            avg_corr = float(np.mean(sub_corr.values[triu_indices]))
            report["avg_active_correlation"] = round(avg_corr, 4)

    return report
