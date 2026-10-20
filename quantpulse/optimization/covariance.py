"""Expected return estimation, covariance matrix modeling, and positive semi-definite (PSD) validation."""

from typing import Optional, Union
import numpy as np
import pandas as pd

from quantpulse.core.config import TRADING_DAYS_PER_YEAR


def calculate_expected_returns(
    returns: pd.DataFrame,
    method: str = "mean",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate annualized expected returns for a universe of assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical periodic returns of assets.
    method : {"mean", "exponential"}, default "mean"
        - "mean": Annualized arithmetic mean of returns: mean(R) * periods_per_year
        - "exponential": Exponentially weighted moving average with 60-day half-life.
    periods_per_year : int, default 252
        Number of trading periods per year.

    Returns
    -------
    pd.Series
        Annualized expected returns indexed by asset ticker.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Input 'returns' must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("Input returns DataFrame cannot be empty.")

    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        raise ValueError("At least 2 observations are required to estimate expected returns.")

    if method == "mean":
        mu = clean_rets.mean() * periods_per_year
    elif method == "exponential":
        # 60-day half-life exponential weighting
        ewm_rets = clean_rets.ewm(halflife=60).mean().iloc[-1]
        mu = ewm_rets * periods_per_year
    else:
        raise ValueError(f"Unsupported expected return method '{method}'. Choose 'mean' or 'exponential'.")

    return mu


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    fix_non_psd: bool = True,
) -> pd.DataFrame:
    """Calculate annualized asset covariance matrix with shrinkage and robustness options.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset periodic returns.
    method : {"sample", "ledoit_wolf"}, default "sample"
        - "sample": Standard sample covariance matrix: cov(R) * periods_per_year
        - "ledoit_wolf": Ledoit-Wolf shrinkage towards constant correlation / variance.
    periods_per_year : int, default 252
        Trading periods per year.
    fix_non_psd : bool, default True
        Whether to adjust small negative eigenvalues resulting from numerical precision.

    Returns
    -------
    pd.DataFrame
        Annualized covariance matrix.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Input 'returns' must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("Input returns DataFrame cannot be empty.")

    clean_rets = returns.dropna()
    tickers = list(clean_rets.columns)
    n_obs, n_assets = clean_rets.shape

    if n_obs < n_assets:
        # High dimensionality warning / fallback to shrinkage
        method = "ledoit_wolf"

    if method == "sample":
        daily_cov = clean_rets.cov()
    elif method == "ledoit_wolf":
        # Ledoit-Wolf shrinkage to structured constant variance target F = (mean_var * I)
        X = clean_rets.values - clean_rets.mean().values
        sample_cov = (X.T @ X) / (n_obs - 1)
        mean_var = np.trace(sample_cov) / n_assets
        prior = mean_var * np.eye(n_assets)

        # Shrinkage intensity computation
        y = X**2
        phi_mat = (y.T @ y) / n_obs - sample_cov**2
        phi = np.sum(phi_mat)
        gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2
        kappa = phi / gamma if gamma > 0 else 0.0
        delta = max(0.0, min(1.0, kappa / n_obs))

        shrunk_cov = delta * prior + (1.0 - delta) * sample_cov
        daily_cov = pd.DataFrame(shrunk_cov, index=tickers, columns=tickers)
    else:
        raise ValueError(f"Unsupported covariance method '{method}'. Choose 'sample' or 'ledoit_wolf'.")

    ann_cov = daily_cov * periods_per_year
    return validate_covariance_matrix(ann_cov, fix_non_psd=fix_non_psd)


def validate_covariance_matrix(
    cov_matrix: pd.DataFrame,
    fix_non_psd: bool = True,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Validate that covariance matrix is symmetric, finite, and positive semi-definite (PSD).

    If matrix has tiny negative eigenvalues due to floating-point approximations,
    replaces them with `epsilon` (spectral projection).
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("Covariance matrix must be a pandas DataFrame.")

    if cov_matrix.isna().any().any():
        raise ValueError("Covariance matrix contains NaN values.")
    if np.isinf(cov_matrix.values).any():
        raise ValueError("Covariance matrix contains infinite values.")

    n_rows, n_cols = cov_matrix.shape
    if n_rows != n_cols:
        raise ValueError(f"Covariance matrix must be square, got shape {cov_matrix.shape}.")

    # Enforce exact symmetry: (A + A.T) / 2
    sym_matrix = (cov_matrix.values + cov_matrix.values.T) / 2.0

    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(sym_matrix)

    min_eigenval = np.min(eigenvals)
    if not fix_non_psd and min_eigenval < -1e-4:
        raise ValueError(
            f"Covariance matrix is significantly non-positive semi-definite (min eigenvalue = {min_eigenval:.6f})."
        )

    if fix_non_psd and min_eigenval < epsilon:
        # Clip eigenvalues below epsilon (spectral projection onto PSD cone)
        clipped_eigenvals = np.maximum(eigenvals, epsilon)
        # Reconstruct PSD matrix: V * diag(clipped) * V^T
        reconstructed = eigenvecs @ np.diag(clipped_eigenvals) @ eigenvecs.T
        # Re-ensure exact symmetry
        reconstructed = (reconstructed + reconstructed.T) / 2.0
        return pd.DataFrame(reconstructed, index=cov_matrix.index, columns=cov_matrix.columns)

    return pd.DataFrame(sym_matrix, index=cov_matrix.index, columns=cov_matrix.columns)
