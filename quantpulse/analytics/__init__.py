"""Quantitative finance analytics: returns, volatility, risk, and tail loss metrics."""

from quantpulse.analytics.returns import (
    calculate_returns,
    calculate_cumulative_returns,
    annualize_returns,
)
from quantpulse.analytics.risk import (
    annualized_volatility,
    downside_deviation,
    sharpe_ratio,
    sortino_ratio,
    drawdown_series,
    max_drawdown,
    calmar_ratio,
    rolling_risk_metrics,
)
from quantpulse.analytics.var import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    historical_cvar,
    parametric_cvar,
)

__all__ = [
    "calculate_returns",
    "calculate_cumulative_returns",
    "annualize_returns",
    "annualized_volatility",
    "downside_deviation",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_series",
    "max_drawdown",
    "calmar_ratio",
    "rolling_risk_metrics",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "historical_cvar",
    "parametric_cvar",
]
