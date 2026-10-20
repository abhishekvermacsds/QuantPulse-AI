"""Portfolio optimization engine supporting Equal-Weight, Minimum Volatility, Maximum Sharpe, and Efficient Frontier."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import cvxpy as cp
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError

from quantpulse.core.config import DEFAULT_RISK_FREE_RATE
from quantpulse.optimization.covariance import (
    calculate_covariance_matrix,
    calculate_expected_returns,
    validate_covariance_matrix,
)


class OptimizationResult(BaseModel):
    """Container for portfolio optimization results."""

    objective: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool = True
    message: str = "Optimization completed successfully."

    @property
    def weights_series(self) -> pd.Series:
        """Return weights as a pandas Series."""
        return pd.Series(self.weights, name=self.objective)


class FrontierPoint(BaseModel):
    """Single coordinate point along the Efficient Frontier."""

    target_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]


class PortfolioOptimizer:
    """Quantitative portfolio optimizer using PyPortfolioOpt and CVXPY.

    Supports:
    - Equal-Weighted (1/N) Baseline
    - Minimum Volatility Portfolio
    - Maximum Sharpe Ratio Portfolio
    - Long-only constraint (w_i >= 0, sum(w) = 1)
    - Configurable maximum allocation limit per asset (w_i <= max_weight)
    - Efficient Frontier curve generation
    - Infeasible constraint detection & graceful solver fallbacks
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = 1.0,
    ):
        self.risk_free_rate = float(risk_free_rate)
        self.max_weight = float(max_weight)

        # Validate inputs
        if not isinstance(expected_returns, pd.Series):
            raise TypeError("expected_returns must be a pandas Series.")
        if not isinstance(cov_matrix, pd.DataFrame):
            raise TypeError("cov_matrix must be a pandas DataFrame.")

        # Ensure index matching
        common_tickers = [t for t in expected_returns.index if t in cov_matrix.index]
        if len(common_tickers) == 0:
            raise ValueError("No common tickers found between expected_returns and cov_matrix.")

        self.tickers = common_tickers
        self.n_assets = len(self.tickers)
        self.expected_returns = expected_returns[self.tickers].astype(float)
        self.cov_matrix = validate_covariance_matrix(self.cov_matrix_subset(cov_matrix))

        # Check for constraint feasibility: max_weight must be >= 1 / N
        min_feasible_weight = 1.0 / self.n_assets
        if self.max_weight < (min_feasible_weight - 1e-6):
            raise ValueError(
                f"Infeasible constraint: max_weight ({self.max_weight:.4f}) cannot be less than "
                f"1 / N ({min_feasible_weight:.4f}) for {self.n_assets} assets."
            )

    def cov_matrix_subset(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Extract square submatrix for aligned tickers."""
        return cov_matrix.loc[self.tickers, self.tickers].astype(float)

    @classmethod
    def from_returns(
        cls,
        returns: pd.DataFrame,
        expected_return_method: str = "mean",
        cov_method: str = "sample",
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = 1.0,
    ) -> "PortfolioOptimizer":
        """Instantiate optimizer directly from historical asset returns."""
        mu = calculate_expected_returns(returns, method=expected_return_method)
        cov = calculate_covariance_matrix(returns, method=cov_method)
        return cls(
            expected_returns=mu,
            cov_matrix=cov,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        )

    def _calc_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Compute expected annual return, volatility, and Sharpe ratio for a weight vector."""
        w = np.asarray(weights, dtype=float)
        mu = self.expected_returns.values
        cov = self.cov_matrix.values

        port_return = float(np.dot(w, mu))
        port_variance = float(w @ cov @ w)
        port_volatility = float(np.sqrt(max(port_variance, 0.0)))

        sharpe = (
            (port_return - self.risk_free_rate) / port_volatility
            if port_volatility > 1e-8
            else 0.0
        )
        return port_return, port_volatility, sharpe

    def optimize_equal_weight(self) -> OptimizationResult:
        """Generate Equal-Weighted (1/N) benchmark allocation."""
        w = np.full(self.n_assets, 1.0 / self.n_assets)
        ret, vol, sr = self._calc_stats(w)
        weights_dict = {ticker: round(float(w[i]), 6) for i, ticker in enumerate(self.tickers)}

        return OptimizationResult(
            objective="equal_weight",
            weights=weights_dict,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            success=True,
            message="Equal-weighted allocation calculated.",
        )

    def optimize_minimum_volatility(self) -> OptimizationResult:
        """Find the Global Minimum Volatility portfolio on the efficient frontier.

        Solves:
            minimize   w.T @ Sigma @ w
            subject to sum(w) == 1, 0 <= w_i <= max_weight
        """
        try:
            ef = EfficientFrontier(
                self.expected_returns,
                self.cov_matrix,
                weight_bounds=(0.0, self.max_weight),
            )
            raw_weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights(cutoff=1e-5)
            weights_array = np.array([cleaned_weights[t] for t in self.tickers])
            # Re-normalize to strictly sum to 1.0
            weights_array = weights_array / np.sum(weights_array)
            ret, vol, sr = self._calc_stats(weights_array)

            return OptimizationResult(
                objective="minimum_volatility",
                weights={t: round(float(weights_array[i]), 6) for i, t in enumerate(self.tickers)},
                expected_return=ret,
                volatility=vol,
                sharpe_ratio=sr,
                success=True,
                message="Minimum volatility optimization succeeded via PyPortfolioOpt.",
            )
        except Exception as exc:
            # Fallback to direct CVXPY solver
            return self._solve_min_vol_cvxpy(fallback_reason=str(exc))

    def _solve_min_vol_cvxpy(self, fallback_reason: str = "") -> OptimizationResult:
        """Direct CVXPY solver for minimum volatility portfolio."""
        w = cp.Variable(self.n_assets)
        cov = self.cov_matrix.values
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= self.max_weight,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # Last resort fallback: equal weight
            eq = self.optimize_equal_weight()
            eq.success = False
            eq.message = f"MinVol solver failed ({prob.status}). Fallback to equal weight. Reason: {fallback_reason}"
            return eq

        w_val = np.maximum(w.value, 0.0)
        w_val = w_val / np.sum(w_val)
        ret, vol, sr = self._calc_stats(w_val)

        return OptimizationResult(
            objective="minimum_volatility",
            weights={t: round(float(w_val[i]), 6) for i, t in enumerate(self.tickers)},
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            success=True,
            message="Minimum volatility optimization succeeded via direct CVXPY fallback.",
        )

    def optimize_max_sharpe(self) -> OptimizationResult:
        """Find the Tangency / Maximum Sharpe Ratio portfolio.

        Solves:
            maximize   (w.T @ mu - Rf) / sqrt(w.T @ Sigma @ w)
            subject to sum(w) == 1, 0 <= w_i <= max_weight

        Handles the case where all expected returns are below the risk-free rate
        by falling back gracefully to Minimum Volatility.
        """
        # Check if any expected return exceeds risk-free rate
        if (self.expected_returns <= self.risk_free_rate).all():
            min_vol_res = self.optimize_minimum_volatility()
            return OptimizationResult(
                objective="max_sharpe",
                weights=min_vol_res.weights,
                expected_return=min_vol_res.expected_return,
                volatility=min_vol_res.volatility,
                sharpe_ratio=min_vol_res.sharpe_ratio,
                success=True,
                message=(
                    "All asset expected returns are <= risk_free_rate. "
                    "Max Sharpe is undefined with positive slope; gracefully fell back to Minimum Volatility."
                ),
            )

        try:
            ef = EfficientFrontier(
                self.expected_returns,
                self.cov_matrix,
                weight_bounds=(0.0, self.max_weight),
            )
            raw_weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights = ef.clean_weights(cutoff=1e-5)
            weights_array = np.array([cleaned_weights[t] for t in self.tickers])
            weights_array = weights_array / np.sum(weights_array)
            ret, vol, sr = self._calc_stats(weights_array)

            return OptimizationResult(
                objective="max_sharpe",
                weights={t: round(float(weights_array[i]), 6) for i, t in enumerate(self.tickers)},
                expected_return=ret,
                volatility=vol,
                sharpe_ratio=sr,
                success=True,
                message="Maximum Sharpe optimization succeeded via PyPortfolioOpt.",
            )
        except Exception as exc:
            # Fallback to direct CVXPY quadratic utility approximation
            return self._solve_max_sharpe_cvxpy(fallback_reason=str(exc))

    def _solve_max_sharpe_cvxpy(self, fallback_reason: str = "") -> OptimizationResult:
        """Direct CVXPY solver for Sharpe ratio maximization via variable transformation (Stutzer/Korn)."""
        try:
            # Transformation: y = w / (w^T mu - Rf), kappa = 1 / (w^T mu - Rf)
            y = cp.Variable(self.n_assets)
            kappa = cp.Variable()
            mu_diff = self.expected_returns.values - self.risk_free_rate
            cov = self.cov_matrix.values

            objective = cp.Minimize(cp.quad_form(y, cov))
            constraints = [
                mu_diff @ y == 1.0,
                cp.sum(y) == kappa,
                y >= 0.0,
                y <= self.max_weight * kappa,
                kappa >= 1e-4,
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL)

            if prob.status in ["optimal", "optimal_inaccurate"] and kappa.value > 0:
                w_val = y.value / kappa.value
                w_val = np.maximum(w_val, 0.0)
                w_val = w_val / np.sum(w_val)
                ret, vol, sr = self._calc_stats(w_val)

                return OptimizationResult(
                    objective="max_sharpe",
                    weights={t: round(float(w_val[i]), 6) for i, t in enumerate(self.tickers)},
                    expected_return=ret,
                    volatility=vol,
                    sharpe_ratio=sr,
                    success=True,
                    message="Max Sharpe optimization succeeded via direct CVXPY fallback.",
                )
        except Exception:
            pass

        # Final fallback to minimum volatility
        min_vol = self.optimize_minimum_volatility()
        min_vol.objective = "max_sharpe"
        min_vol.message = f"Max Sharpe fallback to minimum volatility due to solver issue: {fallback_reason}"
        return min_vol

    def compute_efficient_frontier(self, points: int = 25) -> List[FrontierPoint]:
        """Compute discrete points along the Markowitz Efficient Frontier.

        Spans from the Minimum Volatility return level to the Maximum achievable return.

        Parameters
        ----------
        points : int, default 25
            Number of points along the frontier.

        Returns
        -------
        list of FrontierPoint
        """
        min_vol_res = self.optimize_minimum_volatility()
        max_possible_return = float(self.expected_returns.max())
        min_return = float(min_vol_res.expected_return)

        if max_possible_return <= min_return:
            # Flat frontier
            return [
                FrontierPoint(
                    target_return=min_return,
                    volatility=min_vol_res.volatility,
                    sharpe_ratio=min_vol_res.sharpe_ratio,
                    weights=min_vol_res.weights,
                )
            ]

        target_returns = np.linspace(min_return, max_possible_return * 0.999, points)
        frontier_points: List[FrontierPoint] = []

        for target_ret in target_returns:
            try:
                ef = EfficientFrontier(
                    self.expected_returns,
                    self.cov_matrix,
                    weight_bounds=(0.0, self.max_weight),
                )
                ef.efficient_return(target_return=target_ret)
                raw_weights = ef.clean_weights(cutoff=1e-5)
                w_arr = np.array([raw_weights[t] for t in self.tickers])
                w_arr = w_arr / np.sum(w_arr)
                ret, vol, sr = self._calc_stats(w_arr)

                frontier_points.append(
                    FrontierPoint(
                        target_return=ret,
                        volatility=vol,
                        sharpe_ratio=sr,
                        weights={t: round(float(w_arr[i]), 6) for i, t in enumerate(self.tickers)},
                    )
                )
            except Exception:
                continue

        # If empty or only 1 point, at least return min_vol and equal_weight
        if not frontier_points:
            eq = self.optimize_equal_weight()
            frontier_points = [
                FrontierPoint(
                    target_return=min_vol_res.expected_return,
                    volatility=min_vol_res.volatility,
                    sharpe_ratio=min_vol_res.sharpe_ratio,
                    weights=min_vol_res.weights,
                ),
                FrontierPoint(
                    target_return=eq.expected_return,
                    volatility=eq.volatility,
                    sharpe_ratio=eq.sharpe_ratio,
                    weights=eq.weights,
                ),
            ]

        return frontier_points
