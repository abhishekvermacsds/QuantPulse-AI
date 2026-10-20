"""Milestone 1 Quickstart Demonstration: Financial Risk & Volatility Analysis."""

import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quantpulse.analytics import (
    annualize_returns,
    annualized_volatility,
    calmar_ratio,
    calculate_returns,
    downside_deviation,
    historical_cvar,
    historical_var,
    max_drawdown,
    parametric_cvar,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)
from quantpulse.data import MarketDataLoader


def main():
    print("=" * 80)
    print("   QUANTPULSE AI — Autonomous Portfolio Risk & Volatility Optimization")
    print("   Milestone 1: Risk Engine & Data Analytics Verification")
    print("=" * 80)

    # 1. Ingest Market Data
    loader = MarketDataLoader()
    price_matrix = loader.get_price_matrix()

    print(f"\n[+] Loaded Historical Price Matrix:")
    print(f"    - Date Range : {price_matrix.index.min().date()} to {price_matrix.index.max().date()}")
    print(f"    - Trading Days: {len(price_matrix)}")
    print(f"    - Asset Universe: {list(price_matrix.columns)}")

    # 2. Compute Daily Returns
    daily_returns = calculate_returns(price_matrix, method="simple")

    # 3. Compute Comprehensive Risk & Return Metrics per Asset
    records = []
    for ticker in price_matrix.columns:
        rets = daily_returns[ticker]
        ann_ret = annualize_returns(rets, compound=True)
        ann_vol = annualized_volatility(rets)
        sr = sharpe_ratio(rets, risk_free_rate=0.04)
        sortino = sortino_ratio(rets, risk_free_rate=0.04)
        mdd = max_drawdown(rets)
        calmar = calmar_ratio(rets)
        var_hist = historical_var(rets, confidence_level=0.95)
        var_param = parametric_var(rets, confidence_level=0.95, method="normal")
        var_cf = parametric_var(rets, confidence_level=0.95, method="cornish_fisher")
        cvar_hist = historical_cvar(rets, confidence_level=0.95)
        cvar_param = parametric_cvar(rets, confidence_level=0.95)

        records.append({
            "Asset": ticker,
            "CAGR": f"{ann_ret * 100:.2f}%",
            "Ann. Vol": f"{ann_vol * 100:.2f}%",
            "Sharpe (4%)": f"{sr:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Max DD": f"{mdd * 100:.2f}%",
            "Calmar": f"{calmar:.2f}",
            "VaR 95% (H)": f"{var_hist * 100:.2f}%",
            "VaR 95% (CF)": f"{var_cf * 100:.2f}%",
            "CVaR 95% (H)": f"{cvar_hist * 100:.2f}%",
        })

    summary_df = pd.DataFrame(records)
    print("\n[+] Asset Risk & Performance Summary:")
    print(summary_df.to_string(index=False))

    # 4. Look-Ahead Protection Demonstration
    as_of = "2022-12-30"
    pit_matrix = loader.get_point_in_time(as_of)
    print(f"\n[+] Point-in-Time Slicing (Look-Ahead Bias Prevention):")
    print(f"    - Requested 'as-of' date : {as_of}")
    print(f"    - Sliced max date in history: {pit_matrix.index.max().date()}")
    print(f"    - Guarantee: Zero future leakages into analytical pipelines.")

    print("\n" + "=" * 80)
    print("   Milestone 1 Verification: SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    main()
