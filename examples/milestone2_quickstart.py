"""Milestone 2 Quickstart Demonstration: Portfolio Optimization & Allocation Comparison."""

import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quantpulse.analytics.returns import calculate_returns
from quantpulse.data.loader import MarketDataLoader
from quantpulse.optimization import (
    PortfolioOptimizer,
    calculate_correlation_matrix,
    compare_portfolios,
    diversification_report,
    out_of_sample_validation,
)


def main():
    print("=" * 85)
    print("   QUANTPULSE AI — Autonomous Portfolio Risk & Volatility Optimization")
    print("   Milestone 2: PyPortfolioOpt & CVXPY Optimization Engine")
    print("=" * 85)

    # 1. Ingest Market Data
    loader = MarketDataLoader()
    price_matrix = loader.get_price_matrix()
    daily_returns = calculate_returns(price_matrix, method="simple")

    print(f"\n[1] Ingested Asset Universe:")
    print(f"    - Assets: {list(price_matrix.columns)}")
    print(f"    - Historical Range: {price_matrix.index.min().date()} to {price_matrix.index.max().date()}")
    print(f"    - Observations: {len(daily_returns)} trading days")

    # 2. Correlation Matrix
    corr_matrix = calculate_correlation_matrix(daily_returns)
    print("\n[2] Asset Pairwise Correlation Matrix:")
    print(corr_matrix.round(3).to_string())

    # 3. Optimize Portfolios with Max Weight Constraint
    max_weight_limit = 0.35
    print(f"\n[3] Running Convex Portfolio Optimizations (Max Allocation Limit: {max_weight_limit * 100:.0f}%):")

    optimizer = PortfolioOptimizer.from_returns(
        daily_returns,
        risk_free_rate=0.04,
        max_weight=max_weight_limit,
    )

    eq_res = optimizer.optimize_equal_weight()
    min_vol_res = optimizer.optimize_minimum_volatility()
    max_sharpe_res = optimizer.optimize_max_sharpe()

    # Weight Table
    weights_df = pd.DataFrame({
        "Equal_Weight (1/N)": pd.Series(eq_res.weights),
        f"Min_Volatility (<= {max_weight_limit*100:.0f}%)": pd.Series(min_vol_res.weights),
        f"Max_Sharpe (<= {max_weight_limit*100:.0f}%)": pd.Series(max_sharpe_res.weights),
    }) * 100.0

    print("\n[+] Asset Allocation Weights (%):")
    print(weights_df.round(2).to_string())

    # 4. Comprehensive Performance & Risk Comparison Table
    portfolios = {
        "Equal_Weight": eq_res.weights_series,
        "Min_Volatility": min_vol_res.weights_series,
        "Max_Sharpe": max_sharpe_res.weights_series,
    }

    comparison_df = compare_portfolios(daily_returns, portfolios, risk_free_rate=0.04)
    print("\n[4] Full Portfolio Risk & Performance Attribution (In-Sample):")
    print(comparison_df.to_string())

    # 5. Efficient Frontier Summary
    frontier = optimizer.compute_efficient_frontier(points=5)
    print("\n[5] Sample Points along Efficient Frontier Curve:")
    frontier_records = []
    for i, pt in enumerate(frontier):
        frontier_records.append({
            "Point": f"#{i+1}",
            "Exp. Return": f"{pt.target_return * 100:.2f}%",
            "Volatility": f"{pt.volatility * 100:.2f}%",
            "Sharpe (4%)": f"{pt.sharpe_ratio:.2f}",
        })
    print(pd.DataFrame(frontier_records).to_string(index=False))

    # 6. Look-Ahead Safe Out-of-Sample Backtest Comparison
    print("\n[6] Look-Ahead Safe Validation (70% Train / 30% Unseen Out-of-Sample Test):")
    oos_result = out_of_sample_validation(
        daily_returns,
        train_ratio=0.70,
        risk_free_rate=0.04,
        max_weight=max_weight_limit,
    )
    print(f"    - Training Window (In-Sample): {oos_result['train_range'][0]} -> {oos_result['train_range'][1]}")
    print(f"    - Testing Window (Out-of-Sample): {oos_result['test_range'][0]} -> {oos_result['test_range'][1]}")
    print("\n--- Out-of-Sample Test Performance ---")
    print(oos_result["out_of_sample_metrics"].to_string())

    print("\n" + "=" * 85)
    print("   Milestone 2 Verification: SUCCESSFUL")
    print("=" * 85)


if __name__ == "__main__":
    main()
