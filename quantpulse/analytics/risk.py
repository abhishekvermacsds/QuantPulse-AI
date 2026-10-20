"""Financial risk metrics: volatility, downside deviation, Sharpe, Sortino, Drawdown, and Calmar."""

from typing import Union
import numpy as np
import pandas as pd

from quantpulse.core.config import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from quantpulse.analytics.returns import annualize_returns

SeriesOrDataFrame = Union[pd.Series, pd.DataFrame]


def annualized_volatility(
    returns: SeriesOrDataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
) -> Union[float, pd.Series]:
    """Calculate the annualized standard deviation of returns.

    Formula:
        sigma_ann = std(returns) * sqrt(periods_per_year)

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns.
    periods_per_year : int, default 252
        Number of trading days per year.
    ddof : int, default 1
        Delta degrees of freedom for sample standard deviation.

    Returns
    -------
    float or pd.Series
        Annualized volatility.
    """
    if len(returns) <= ddof:
        return np.nan if isinstance(returns, pd.Series) else pd.Series(np.nan, index=returns.columns)

    daily_std = returns.std(ddof=ddof)
    scale = np.sqrt(periods_per_year)
    return float(daily_std * scale) if isinstance(returns, pd.Series) else daily_std * scale


def downside_deviation(
    returns: SeriesOrDataFrame,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
) -> Union[float, pd.Series]:
    """Calculate the annualized downside deviation (semi-deviation) below a target threshold.

    Formula:
        downside_diff = min(returns - target_return, 0)
        semi_dev = sqrt(sum(downside_diff^2) / (N - ddof)) * sqrt(periods_per_year)

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns.
    target_return : float, default 0.0
        Benchmark threshold (e.g. 0.0 or daily risk-free rate).
    periods_per_year : int, default 252
        Trading periods per year.
    ddof : int, default 1
        Degrees of freedom.

    Returns
    -------
    float or pd.Series
        Annualized downside deviation.
    """
    if len(returns) <= ddof:
        return np.nan if isinstance(returns, pd.Series) else pd.Series(np.nan, index=returns.columns)

    # Only penalize negative excess returns
    excess = returns - target_return
    underperformance = np.minimum(excess, 0.0)
    sum_sq = (underperformance ** 2).sum()
    n_effective = len(returns) - ddof

    variance = sum_sq / max(n_effective, 1)
    daily_semi_dev = np.sqrt(variance)
    scale = np.sqrt(periods_per_year)
    return float(daily_semi_dev * scale) if isinstance(returns, pd.Series) else daily_semi_dev * scale


def sharpe_ratio(
    returns: SeriesOrDataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    compound_returns: bool = True,
) -> Union[float, pd.Series]:
    """Calculate the annualized Sharpe Ratio.

    Formula:
        Sharpe = (Annualized_Return - Risk_Free_Rate) / Annualized_Volatility

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns.
    risk_free_rate : float, default 0.04
        Annualized risk-free rate.
    periods_per_year : int, default 252
        Trading periods per year.
    compound_returns : bool, default True
        Whether to use CAGR (True) or arithmetic annualized return (False).

    Returns
    -------
    float or pd.Series
        Annualized Sharpe Ratio.
    """
    ann_ret = annualize_returns(returns, periods_per_year=periods_per_year, compound=compound_returns)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)

    if isinstance(returns, pd.Series):
        if np.isnan(ann_vol) or ann_vol <= 1e-8:
            return 0.0
        return float((ann_ret - risk_free_rate) / ann_vol)
    else:
        # Vectorized for DataFrame columns
        diff = ann_ret - risk_free_rate
        ratio = diff / ann_vol.replace(0.0, np.nan)
        return ratio.fillna(0.0)


def sortino_ratio(
    returns: SeriesOrDataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    compound_returns: bool = True,
) -> Union[float, pd.Series]:
    """Calculate the annualized Sortino Ratio.

    Focuses exclusively on downside variance, without penalizing upside volatility.

    Formula:
        Sortino = (Annualized_Return - Risk_Free_Rate) / Downside_Deviation

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns.
    risk_free_rate : float, default 0.04
        Annualized risk-free rate.
    target_return : float, default 0.0
        Minimum acceptable daily return.
    periods_per_year : int, default 252
        Trading periods per year.
    compound_returns : bool, default True
        Whether to use CAGR.

    Returns
    -------
    float or pd.Series
        Annualized Sortino Ratio.
    """
    ann_ret = annualize_returns(returns, periods_per_year=periods_per_year, compound=compound_returns)
    downside_vol = downside_deviation(returns, target_return=target_return, periods_per_year=periods_per_year)

    if isinstance(returns, pd.Series):
        if np.isnan(downside_vol) or downside_vol <= 1e-8:
            return 0.0
        return float((ann_ret - risk_free_rate) / downside_vol)
    else:
        diff = ann_ret - risk_free_rate
        ratio = diff / downside_vol.replace(0.0, np.nan)
        return ratio.fillna(0.0)


def drawdown_series(returns: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Compute the drawdown trajectory (underwater series) over time.

    Formula:
        Wealth_t = cumprod(1 + R_t)
        Peak_t = max_{s <= t}(Wealth_s)
        Drawdown_t = (Wealth_t - Peak_t) / Peak_t <= 0

    Returns
    -------
    pd.Series or pd.DataFrame
        Time series of percentage drawdown values <= 0.0.
    """
    wealth = (1.0 + returns).cumprod()
    peaks = wealth.cummax()
    drawdowns = (wealth - peaks) / peaks
    return drawdowns


def max_drawdown(returns: SeriesOrDataFrame) -> Union[float, pd.Series]:
    """Calculate the maximum peak-to-trough drop (Maximum Drawdown, MDD).

    Formula:
        MDD = min(Drawdown_t)

    Returns
    -------
    float or pd.Series
        Maximum drawdown expressed as a negative float (e.g. -0.22 for -22%).
    """
    dd = drawdown_series(returns)
    if isinstance(returns, pd.Series):
        val = dd.min()
        return 0.0 if np.isnan(val) else float(val)
    else:
        return dd.min().fillna(0.0)


def calmar_ratio(
    returns: SeriesOrDataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    compound_returns: bool = True,
) -> Union[float, pd.Series]:
    """Calculate the Calmar Ratio: annualized return divided by absolute maximum drawdown.

    Formula:
        Calmar = Annualized_Return / |Maximum_Drawdown|
    """
    ann_ret = annualize_returns(returns, periods_per_year=periods_per_year, compound=compound_returns)
    mdd = max_drawdown(returns)

    if isinstance(returns, pd.Series):
        abs_mdd = abs(mdd)
        if abs_mdd <= 1e-8:
            return 0.0
        return float(ann_ret / abs_mdd)
    else:
        abs_mdd = mdd.abs().replace(0.0, np.nan)
        ratio = ann_ret / abs_mdd
        return ratio.fillna(0.0)


def rolling_risk_metrics(
    returns: pd.Series,
    window: int = 63,  # Approx 3 months of trading days
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate rolling risk metrics across a sliding historical window.

    Guarantees no look-ahead bias by evaluating metrics strictly on backwards-looking windows.

    Returns DataFrame with columns:
    - 'rolling_return' (annualized)
    - 'rolling_volatility' (annualized)
    - 'rolling_sharpe'
    - 'rolling_max_drawdown'
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("rolling_risk_metrics currently requires a single pd.Series.")

    roll_ret = returns.rolling(window).apply(
        lambda w: annualize_returns(pd.Series(w), periods_per_year=periods_per_year),
        raw=False,
    )
    roll_vol = returns.rolling(window).apply(
        lambda w: annualized_volatility(pd.Series(w), periods_per_year=periods_per_year),
        raw=False,
    )
    roll_mdd = returns.rolling(window).apply(
        lambda w: max_drawdown(pd.Series(w)),
        raw=False,
    )

    roll_sharpe = (roll_ret - risk_free_rate) / roll_vol.replace(0.0, np.nan)

    df_rolling = pd.DataFrame({
        "rolling_return": roll_ret,
        "rolling_volatility": roll_vol,
        "rolling_sharpe": roll_sharpe.fillna(0.0),
        "rolling_max_drawdown": roll_mdd,
    }, index=returns.index)

    return df_rolling
