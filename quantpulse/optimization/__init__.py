"""Portfolio optimization, asset allocation, diversification, and comparison module."""

from quantpulse.optimization.covariance import (
    calculate_covariance_matrix,
    calculate_expected_returns,
    validate_covariance_matrix,
)
from quantpulse.optimization.diversification import (
    calculate_correlation_matrix,
    diversification_ratio,
    diversification_report,
    effective_number_of_assets,
    herfindahl_index,
)
from quantpulse.optimization.optimizer import (
    FrontierPoint,
    OptimizationResult,
    PortfolioOptimizer,
)
from quantpulse.optimization.comparison import (
    calculate_portfolio_returns,
    compare_portfolios,
    evaluate_portfolio_metrics,
    out_of_sample_validation,
)

__all__ = [
    "calculate_covariance_matrix",
    "calculate_expected_returns",
    "validate_covariance_matrix",
    "calculate_correlation_matrix",
    "diversification_ratio",
    "diversification_report",
    "effective_number_of_assets",
    "herfindahl_index",
    "FrontierPoint",
    "OptimizationResult",
    "PortfolioOptimizer",
    "calculate_portfolio_returns",
    "compare_portfolios",
    "evaluate_portfolio_metrics",
    "out_of_sample_validation",
]
