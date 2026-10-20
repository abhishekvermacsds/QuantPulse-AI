"""Portfolio metrics, asset correlations, and convex optimization endpoints."""

import logging
from typing import Dict, List
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from quantpulse.analytics import (
    annualize_returns,
    annualized_volatility,
    calmar_ratio,
    calculate_returns,
    historical_cvar,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)
from quantpulse.api.schemas import (
    AssetMetricsItem,
    CorrelationResponse,
    OptimizationRequest,
    OptimizationResponse,
    PortfolioAllocationProfile,
    PortfolioMetricsResponse,
)
from quantpulse.core.cache import (
    cache_response,
    generate_cache_key,
    get_cached_value,
    set_cached_value,
)
from quantpulse.data.loader import MarketDataLoader
from quantpulse.db.models import OptimizationRecord
from quantpulse.db.session import SessionLocal
from quantpulse.optimization import (
    PortfolioOptimizer,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_portfolio_returns,
    diversification_ratio,
    effective_number_of_assets,
    evaluate_portfolio_metrics,
)

logger = logging.getLogger(__name__)
router = APIRouter()
_loader = MarketDataLoader()


@router.get("/portfolio/metrics", response_model=PortfolioMetricsResponse)
@cache_response(prefix="portfolio_metrics", ttl=300)
def get_portfolio_metrics() -> PortfolioMetricsResponse:
    """Retrieve full risk, volatility, and performance profile for all universe assets."""
    price_matrix = _loader.get_price_matrix()
    daily_returns = calculate_returns(price_matrix, method="simple")

    metrics_list: List[AssetMetricsItem] = []
    for ticker in price_matrix.columns:
        rets = daily_returns[ticker]
        metrics_list.append(
            AssetMetricsItem(
                ticker=ticker,
                cagr=round(float(annualize_returns(rets, compound=True)) * 100.0, 2),
                annualized_volatility=round(float(annualized_volatility(rets)) * 100.0, 2),
                sharpe_ratio=round(float(sharpe_ratio(rets, risk_free_rate=0.04)), 2),
                sortino_ratio=round(float(sortino_ratio(rets, risk_free_rate=0.04)), 2),
                max_drawdown=round(float(max_drawdown(rets)) * 100.0, 2),
                calmar_ratio=round(float(calmar_ratio(rets)), 2),
                var_95_historical=round(float(historical_var(rets, 0.95)) * 100.0, 2),
                var_95_cornish_fisher=round(float(parametric_var(rets, 0.95, method="cornish_fisher")) * 100.0, 2),
                cvar_95_historical=round(float(historical_cvar(rets, 0.95)) * 100.0, 2),
            )
        )

    return PortfolioMetricsResponse(
        start_date=price_matrix.index.min().strftime("%Y-%m-%d"),
        end_date=price_matrix.index.max().strftime("%Y-%m-%d"),
        trading_days=len(price_matrix),
        asset_universe=list(price_matrix.columns),
        metrics=metrics_list,
    )


@router.get("/portfolio/correlation", response_model=CorrelationResponse)
@cache_response(prefix="portfolio_correlation", ttl=300)
def get_correlation_matrix() -> CorrelationResponse:
    """Retrieve pairwise asset return correlation matrix."""
    price_matrix = _loader.get_price_matrix()
    daily_returns = calculate_returns(price_matrix, method="simple")
    corr_df = calculate_correlation_matrix(daily_returns)

    # Format into nested dictionary
    tickers = list(corr_df.columns)
    matrix_dict = {
        row_ticker: {col_ticker: round(float(corr_df.loc[row_ticker, col_ticker]), 4) for col_ticker in tickers}
        for row_ticker in tickers
    }

    return CorrelationResponse(tickers=tickers, matrix=matrix_dict)


@router.post("/portfolio/optimize", response_model=OptimizationResponse)
def optimize_portfolio(req: OptimizationRequest) -> OptimizationResponse:
    """Compute optimal portfolio allocations subject to long-only and maximum weight constraints."""
    cache_key = generate_cache_key(
        "portfolio_optimize",
        req.tickers,
        req.objective,
        req.max_weight,
        req.risk_free_rate,
        req.include_frontier,
    )
    cached_result = get_cached_value(cache_key)
    if cached_result:
        return OptimizationResponse(**cached_result)

    price_matrix = _loader.get_price_matrix(tickers=req.tickers)
    daily_returns = calculate_returns(price_matrix, method="simple")
    tickers = list(price_matrix.columns)
    n_assets = len(tickers)

    min_feasible_cap = 1.0 / n_assets
    if req.max_weight < (min_feasible_cap - 1e-6):
        raise HTTPException(
            status_code=400,
            detail=f"Infeasible constraint: max_weight ({req.max_weight:.2f}) cannot be less than 1 / N ({min_feasible_cap:.2f}) for {n_assets} assets.",
        )

    try:
        optimizer = PortfolioOptimizer.from_returns(
            daily_returns,
            risk_free_rate=req.risk_free_rate,
            max_weight=req.max_weight,
        )

        allocations: Dict[str, PortfolioAllocationProfile] = {}
        cov = calculate_covariance_matrix(daily_returns)

        # 1. Equal-Weight Baseline
        if req.objective in ["equal_weight", "all"]:
            eq_res = optimizer.optimize_equal_weight()
            eq_metrics = evaluate_portfolio_metrics(
                calculate_portfolio_returns(daily_returns, eq_res.weights),
                weights=eq_res.weights_series,
                cov_matrix=cov,
                risk_free_rate=req.risk_free_rate,
            )
            allocations["equal_weight"] = PortfolioAllocationProfile(
                objective="equal_weight",
                weights=eq_res.weights,
                expected_return=round(eq_res.expected_return * 100.0, 2),
                volatility=round(eq_res.volatility * 100.0, 2),
                sharpe_ratio=round(eq_res.sharpe_ratio, 2),
                diversification_ratio=eq_metrics["Diversification_Ratio"],
                effective_assets=eq_metrics["Effective_Assets"],
                max_drawdown=eq_metrics["Max_Drawdown"],
                var_95_historical=eq_metrics["VaR_95_Historical"],
                cvar_95_historical=eq_metrics["CVaR_95_Historical"],
                message=eq_res.message,
            )

        # 2. Minimum Volatility
        if req.objective in ["min_volatility", "all"]:
            min_vol_res = optimizer.optimize_minimum_volatility()
            mv_metrics = evaluate_portfolio_metrics(
                calculate_portfolio_returns(daily_returns, min_vol_res.weights),
                weights=min_vol_res.weights_series,
                cov_matrix=cov,
                risk_free_rate=req.risk_free_rate,
            )
            allocations["min_volatility"] = PortfolioAllocationProfile(
                objective="min_volatility",
                weights=min_vol_res.weights,
                expected_return=round(min_vol_res.expected_return * 100.0, 2),
                volatility=round(min_vol_res.volatility * 100.0, 2),
                sharpe_ratio=round(min_vol_res.sharpe_ratio, 2),
                diversification_ratio=mv_metrics["Diversification_Ratio"],
                effective_assets=mv_metrics["Effective_Assets"],
                max_drawdown=mv_metrics["Max_Drawdown"],
                var_95_historical=mv_metrics["VaR_95_Historical"],
                cvar_95_historical=mv_metrics["CVaR_95_Historical"],
                message=min_vol_res.message,
            )

        # 3. Maximum Sharpe
        if req.objective in ["max_sharpe", "all"]:
            ms_res = optimizer.optimize_max_sharpe()
            ms_metrics = evaluate_portfolio_metrics(
                calculate_portfolio_returns(daily_returns, ms_res.weights),
                weights=ms_res.weights_series,
                cov_matrix=cov,
                risk_free_rate=req.risk_free_rate,
            )
            allocations["max_sharpe"] = PortfolioAllocationProfile(
                objective="max_sharpe",
                weights=ms_res.weights,
                expected_return=round(ms_res.expected_return * 100.0, 2),
                volatility=round(ms_res.volatility * 100.0, 2),
                sharpe_ratio=round(ms_res.sharpe_ratio, 2),
                diversification_ratio=ms_metrics["Diversification_Ratio"],
                effective_assets=ms_metrics["Effective_Assets"],
                max_drawdown=ms_metrics["Max_Drawdown"],
                var_95_historical=ms_metrics["VaR_95_Historical"],
                cvar_95_historical=ms_metrics["CVaR_95_Historical"],
                message=ms_res.message,
            )

        frontier_points = None
        if req.include_frontier:
            raw_frontier = optimizer.compute_efficient_frontier(points=12)
            frontier_points = [
                {
                    "return": round(pt.target_return * 100.0, 2),
                    "volatility": round(pt.volatility * 100.0, 2),
                    "sharpe": round(pt.sharpe_ratio, 2),
                }
                for pt in raw_frontier
            ]

        response = OptimizationResponse(allocations=allocations, frontier=frontier_points)

        # Store in cache
        try:
            set_cached_value(cache_key, response.model_dump(), ttl=600)
        except Exception as exc:
            logger.debug("Failed to set optimization cache: %s", exc)

        # Optionally persist optimization run to PostgreSQL
        try:
            primary_profile = allocations.get("max_sharpe") or allocations.get("min_volatility") or allocations.get("equal_weight")
            if primary_profile:
                db = SessionLocal()
                record = OptimizationRecord(
                    portfolio_id=None,
                    objective=req.objective,
                    target_return=None,
                    max_weight=req.max_weight,
                    optimal_weights=primary_profile.weights,
                    expected_return=primary_profile.expected_return,
                    volatility=primary_profile.volatility,
                    sharpe_ratio=primary_profile.sharpe_ratio,
                    diversification_ratio=primary_profile.diversification_ratio,
                    effective_assets=primary_profile.effective_assets,
                )
                db.add(record)
                db.commit()
                db.close()
        except Exception as exc:
            logger.debug("Could not record optimization to database: %s", exc)

        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(exc)}") from exc
