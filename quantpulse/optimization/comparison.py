"""Portfolio comparison, performance attribution, and look-ahead safe out-of-sample evaluation."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from quantpulse.core.config import DEFAULT_RISK_FREE_RATE
from quantpulse.analytics.returns import annualize_returns, calculate_cumulative_returns
from quantpulse.analytics.risk import (
    annualized_volatility,
    calmar_ratio,
    downside_deviation,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quantpulse.analytics.var import (
    historical_cvar,
    historical_var,
    parametric_cvar,
    parametric_var,
)
from quantpulse.optimization.covariance import calculate_covariance_matrix
from quantpulse.optimization.diversification import (
    diversification_ratio,
    effective_number_of_assets,
    herfindahl_index,
)


def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Union[pd.Series, Dict[str, float]],
) -> pd.Series:
    """Calculate the time series of aggregate portfolio returns from asset weights.

    Formula:
        R_p(t) = sum_{i=1}^N w_i * R_{i,t}
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    common_assets = [c for c in asset_returns.columns if c in weights.index]
    if not common_assets:
        raise ValueError("No common assets between asset returns and portfolio weights.")

    w = weights[common_assets]
    w = w / w.sum()  # ensure exact sum to 1.0

    port_returns = asset_returns[common_assets].dot(w)
    port_returns.name = "Portfolio_Returns"
    return port_returns


def evaluate_portfolio_metrics(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    cov_matrix: Optional[pd.DataFrame] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Any]:
    """Calculate a comprehensive risk, return, and diversification profile for a portfolio."""
    ann_ret = annualize_returns(portfolio_returns, compound=True)
    ann_vol = annualized_volatility(portfolio_returns)
    sr = sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate)
    sortino = sortino_ratio(portfolio_returns, risk_free_rate=risk_free_rate)
    mdd = max_drawdown(portfolio_returns)
    calmar = calmar_ratio(portfolio_returns)

    var_95_h = historical_var(portfolio_returns, confidence_level=0.95)
    var_95_cf = parametric_var(portfolio_returns, confidence_level=0.95, method="cornish_fisher")
    cvar_95_h = historical_cvar(portfolio_returns, confidence_level=0.95)

    hhi = herfindahl_index(weights)
    n_eff = effective_number_of_assets(weights)

    dr = None
    if cov_matrix is not None:
        try:
            dr = diversification_ratio(weights, cov_matrix)
        except Exception:
            pass

    return {
        "CAGR": round(ann_ret * 100.0, 2),
        "Annualized_Vol": round(ann_vol * 100.0, 2),
        "Sharpe": round(sr, 2),
        "Sortino": round(sortino, 2),
        "Max_Drawdown": round(mdd * 100.0, 2),
        "Calmar": round(calmar, 2),
        "VaR_95_Historical": round(var_95_h * 100.0, 2),
        "VaR_95_CornishFisher": round(var_95_cf * 100.0, 2),
        "CVaR_95_Historical": round(cvar_95_h * 100.0, 2),
        "Diversification_Ratio": round(dr, 3) if dr is not None else np.nan,
        "Effective_Assets": round(n_eff, 2),
    }


def compare_portfolios(
    returns: pd.DataFrame,
    portfolios: Dict[str, Union[pd.Series, Dict[str, float]]],
    cov_matrix: Optional[pd.DataFrame] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compare multiple portfolio allocations side-by-side across key risk and return metrics.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.
    portfolios : dict of (str, weights)
        Mapping of portfolio names (e.g. 'Equal_Weight', 'Min_Vol', 'Max_Sharpe') to weights.
    cov_matrix : pd.DataFrame, optional
        Annualized covariance matrix for diversification ratio.
    risk_free_rate : float
        Benchmark risk-free rate.

    Returns
    -------
    pd.DataFrame
        Tabular comparison with metric rows and portfolio columns.
    """
    if cov_matrix is None:
        cov_matrix = calculate_covariance_matrix(returns)

    profiles: Dict[str, Dict[str, Any]] = {}
    for name, w_input in portfolios.items():
        w_series = pd.Series(w_input) if isinstance(w_input, dict) else w_input
        port_ret = calculate_portfolio_returns(returns, w_series)
        profiles[name] = evaluate_portfolio_metrics(
            port_ret,
            weights=w_series,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
        )

    df_comp = pd.DataFrame(profiles)
    return df_comp


def out_of_sample_validation(
    returns: pd.DataFrame,
    train_ratio: float = 0.70,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = 0.40,
) -> Dict[str, Any]:
    """Perform look-ahead safe in-sample vs out-of-sample portfolio evaluation.

    1. Splits asset returns chronologically into train and test sets.
    2. Optimizes Equal-Weight, Min-Vol, and Max-Sharpe strictly on train data.
    3. Evaluates allocations on unseen out-of-sample test data.

    Returns dictionary with in-sample and out-of-sample comparison tables.
    """
    from quantpulse.optimization.optimizer import PortfolioOptimizer

    clean_rets = returns.dropna()
    split_idx = int(len(clean_rets) * train_ratio)
    if split_idx < 30 or (len(clean_rets) - split_idx) < 30:
        raise ValueError("Insufficient observations for train/test split (need at least 30 periods per split).")

    train_rets = clean_rets.iloc[:split_idx]
    test_rets = clean_rets.iloc[split_idx:]

    optimizer = PortfolioOptimizer.from_returns(
        train_rets,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )

    eq_res = optimizer.optimize_equal_weight()
    min_vol_res = optimizer.optimize_minimum_volatility()
    max_sharpe_res = optimizer.optimize_max_sharpe()

    portfolios = {
        "Equal_Weight": eq_res.weights_series,
        "Min_Volatility": min_vol_res.weights_series,
        "Max_Sharpe": max_sharpe_res.weights_series,
    }

    in_sample_comp = compare_portfolios(train_rets, portfolios, risk_free_rate=risk_free_rate)
    out_of_sample_comp = compare_portfolios(test_rets, portfolios, risk_free_rate=risk_free_rate)

    return {
        "train_range": (train_rets.index.min().strftime("%Y-%m-%d"), train_rets.index.max().strftime("%Y-%m-%d")),
        "test_range": (test_rets.index.min().strftime("%Y-%m-%d"), test_rets.index.max().strftime("%Y-%m-%d")),
        "in_sample_metrics": in_sample_comp,
        "out_of_sample_metrics": out_of_sample_comp,
        "allocations": {name: res.weights for name, res in [
            ("Equal_Weight", eq_res),
            ("Min_Volatility", min_vol_res),
            ("Max_Sharpe", max_sharpe_res),
        ]},
    }
