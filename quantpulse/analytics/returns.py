"""Return calculation module: simple, logarithmic, cumulative, and annualized returns."""

from typing import Union
import numpy as np
import pandas as pd

from quantpulse.core.config import TRADING_DAYS_PER_YEAR

SeriesOrDataFrame = Union[pd.Series, pd.DataFrame]


def calculate_returns(
    prices: SeriesOrDataFrame,
    method: str = "simple",
    drop_first: bool = True,
) -> SeriesOrDataFrame:
    """Calculate periodic asset returns from price levels.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Historical asset prices indexed by timestamp.
    method : {"simple", "log"}, default "simple"
        - "simple": R_t = (P_t - P_{t-1}) / P_{t-1}
        - "log": r_t = ln(P_t / P_{t-1})
    drop_first : bool, default True
        Whether to drop the initial NaN resulting from the period shift.

    Returns
    -------
    pd.Series or pd.DataFrame
        Calculated return series.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("Input 'prices' must be a pandas Series or DataFrame.")

    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to calculate returns.")

    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'simple' or 'log'.")

    if drop_first:
        returns = returns.dropna()

    return returns


def calculate_cumulative_returns(
    returns: SeriesOrDataFrame,
    method: str = "simple",
) -> SeriesOrDataFrame:
    """Compute cumulative wealth trajectory starting from baseline 0.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns.
    method : {"simple", "log"}, default "simple"
        Format of input returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Cumulative return series starting from 0.0 at t=0.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("Input 'returns' must be a pandas Series or DataFrame.")

    if method == "simple":
        return (1.0 + returns).cumprod() - 1.0
    elif method == "log":
        return np.exp(returns.cumsum()) - 1.0
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'simple' or 'log'.")


def annualize_returns(
    returns: SeriesOrDataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    compound: bool = True,
) -> Union[float, pd.Series]:
    """Calculate the annualized return (CAGR or arithmetic).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic return series (simple returns).
    periods_per_year : int, default 252
        Number of trading periods per year.
    compound : bool, default True
        If True, calculates Compound Annual Growth Rate (CAGR): (1 + R_total)^(252 / N) - 1.
        If False, calculates arithmetic annualized mean: mean(R) * 252.

    Returns
    -------
    float or pd.Series
        Annualized return.
    """
    if len(returns) == 0:
        return np.nan if isinstance(returns, pd.Series) else pd.Series(dtype=float)

    n_periods = len(returns)

    if not compound:
        return returns.mean() * periods_per_year

    # CAGR: (1 + total_return) ** (periods_per_year / n_periods) - 1
    total_wealth = (1.0 + returns).prod()
    if isinstance(total_wealth, pd.Series):
        # Handle cases where total wealth is non-positive (100% loss)
        result = np.where(
            total_wealth > 0,
            (total_wealth ** (periods_per_year / n_periods)) - 1.0,
            -1.0,
        )
        return pd.Series(result, index=returns.columns)
    else:
        if total_wealth <= 0:
            return -1.0
        return float((total_wealth ** (periods_per_year / n_periods)) - 1.0)
