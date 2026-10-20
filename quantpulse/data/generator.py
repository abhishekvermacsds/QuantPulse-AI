"""Synthetic financial market data generator using correlated Geometric Brownian Motion with Merton Jump-Diffusion."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from quantpulse.core.config import SAMPLE_DATA_DIR, TRADING_DAYS_PER_YEAR


DEFAULT_ASSET_UNIVERSE: Dict[str, Dict[str, float]] = {
    "AAPL": {"initial_price": 150.0, "mu": 0.18, "sigma": 0.25, "sector": "Technology"},
    "MSFT": {"initial_price": 280.0, "mu": 0.16, "sigma": 0.22, "sector": "Technology"},
    "JPM":  {"initial_price": 130.0, "mu": 0.12, "sigma": 0.20, "sector": "Financials"},
    "XOM":  {"initial_price": 95.0,  "mu": 0.10, "sigma": 0.24, "sector": "Energy"},
    "SPY":  {"initial_price": 420.0, "mu": 0.11, "sigma": 0.16, "sector": "Index ETF"},
    "TLT":  {"initial_price": 105.0, "mu": 0.04, "sigma": 0.14, "sector": "Fixed Income ETF"},
}

# Empirical correlation matrix reflecting realistic cross-asset dependencies
DEFAULT_CORRELATION_MATRIX = np.array([
    # AAPL   MSFT    JPM    XOM    SPY    TLT
    [ 1.00,  0.72,  0.45,  0.30,  0.82, -0.25],  # AAPL
    [ 0.72,  1.00,  0.42,  0.28,  0.80, -0.22],  # MSFT
    [ 0.45,  0.42,  1.00,  0.50,  0.68, -0.35],  # JPM
    [ 0.30,  0.28,  0.50,  1.00,  0.55, -0.15],  # XOM
    [ 0.82,  0.80,  0.68,  0.55,  1.00, -0.38],  # SPY
    [-0.25, -0.22, -0.35, -0.15, -0.38,  1.00],  # TLT
])


def generate_synthetic_market_data(
    start_date: str = "2021-01-04",
    periods: int = 756,  # 3 full trading years (approx 252 * 3)
    universe: Optional[Dict[str, Dict[str, float]]] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    seed: int = 42,
    jump_lambda: float = 0.02,  # Poisson jump intensity (2% prob of shock per day)
    jump_mean: float = -0.015,  # Average jump effect
    jump_std: float = 0.03,     # Jump standard deviation
) -> pd.DataFrame:
    """Generate realistic OHLCV market data for a basket of assets.

    Employs correlated Geometric Brownian Motion with Merton Jump-Diffusion
    to simulate realistic financial asset characteristics:
    - Drift and volatility scaling
    - Cross-asset correlation via Cholesky factorization
    - Heavy tails / kurtosis from Poisson-driven market shocks
    - Realistic intra-day High, Low, Open, Volume relations

    Parameters
    ----------
    start_date : str
        First trading date (ISO format YYYY-MM-DD).
    periods : int
        Number of consecutive trading days to generate.
    universe : dict, optional
        Dictionary mapping ticker to parameters (initial_price, mu, sigma).
    correlation_matrix : np.ndarray, optional
        Symmetric positive-definite correlation matrix.
    seed : int
        Random seed for deterministic, reproducible output.
    jump_lambda : float
        Annualized Poisson rate of jump events.
    jump_mean : float
        Mean magnitude of jumps.
    jump_std : float
        Standard deviation of jumps.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    """
    rng = np.random.default_rng(seed)
    assets = universe or DEFAULT_ASSET_UNIVERSE
    tickers = list(assets.keys())
    num_assets = len(tickers)

    corr = correlation_matrix if correlation_matrix is not None else DEFAULT_CORRELATION_MATRIX
    if corr.shape != (num_assets, num_assets):
        raise ValueError(
            f"Correlation matrix shape {corr.shape} does not match asset count {num_assets}"
        )

    # Cholesky decomposition: L @ L.T = Correlation Matrix
    cholesky_matrix = np.linalg.cholesky(corr)

    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # Pre-extract asset drift and annual volatility
    mus = np.array([assets[t]["mu"] for t in tickers])
    sigmas = np.array([assets[t]["sigma"] for t in tickers])
    initial_prices = np.array([assets[t]["initial_price"] for t in tickers])

    # Uncorrelated standard normal increments (periods - 1, num_assets)
    uncorrelated_normals = rng.standard_normal(size=(periods - 1, num_assets))
    # Correlated standard normal increments: (uncorrelated @ L.T)
    correlated_normals = uncorrelated_normals @ cholesky_matrix.T

    # Generate Poisson-driven jumps (Merton jump-diffusion)
    jump_occurrences = rng.poisson(lam=jump_lambda, size=(periods - 1, num_assets))
    jump_magnitudes = rng.normal(loc=jump_mean, scale=jump_std, size=(periods - 1, num_assets))
    jumps = jump_occurrences * jump_magnitudes

    # Log returns: d(ln S) = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z + jumps
    drift = (mus - 0.5 * sigmas**2) * dt
    diffusion = sigmas * np.sqrt(dt) * correlated_normals
    log_returns = drift + diffusion + jumps

    # Build prices: P_t = P_0 * exp(cumsum(log_returns))
    all_log_returns = np.vstack([np.zeros((1, num_assets)), log_returns])
    cum_log_returns = np.cumsum(all_log_returns, axis=0)
    close_prices = initial_prices * np.exp(cum_log_returns)

    # Generate business dates
    date_range = pd.bdate_range(start=start_date, periods=periods)

    records: List[Dict] = []
    for asset_idx, ticker in enumerate(tickers):
        prices = close_prices[:, asset_idx]
        for t in range(periods):
            curr_date = date_range[t]
            close_p = float(prices[t])

            # Generate synthetic Open, High, Low relative to Close
            if t == 0:
                open_p = close_p * (1.0 + rng.normal(0, 0.002))
            else:
                # Open tends to be close to previous Close with small gap
                prev_p = float(prices[t - 1])
                open_p = prev_p * (1.0 + rng.normal(0, 0.003))

            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            intra_spread = abs(close_p - open_p)
            high_addon = max(intra_spread * 0.4, close_p * abs(rng.normal(0.005, 0.003)))
            low_deduct = max(intra_spread * 0.4, close_p * abs(rng.normal(0.005, 0.003)))

            high_p = max(open_p, close_p) + high_addon
            low_p = max(0.01, min(open_p, close_p) - low_deduct)

            # Realistic daily volume (higher on high volatility days)
            daily_volatility_shock = abs(close_p - open_p) / open_p
            base_vol = 1_500_000 if ticker not in ["SPY", "AAPL"] else 25_000_000
            volume = int(base_vol * (1.0 + 10.0 * daily_volatility_shock) * rng.uniform(0.7, 1.4))

            records.append({
                "Date": curr_date.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Open": round(open_p, 4),
                "High": round(high_p, 4),
                "Low": round(low_p, 4),
                "Close": round(close_p, 4),
                "Adj_Close": round(close_p, 4),
                "Volume": volume,
            })

    df = pd.DataFrame(records)
    # Sort strictly by Date ascending, then Ticker
    df.sort_values(by=["Date", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_sample_market_data(destination: Optional[Path] = None, seed: int = 42) -> Path:
    """Generate and persist the sample market dataset to CSV."""
    target_path = destination or (SAMPLE_DATA_DIR / "market_data.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_market_data(seed=seed)
    df.to_csv(target_path, index=False)
    return target_path


if __name__ == "__main__":
    out_file = save_sample_market_data()
    print(f"Sample market data written to {out_file}")
