import React from "react";
import { Scale } from "lucide-react";
import { OptimizationData } from "../types";

interface OptimizationComparisonProps {
  optimization: OptimizationData | null;
}

export const OptimizationComparison: React.FC<OptimizationComparisonProps> = ({
  optimization,
}) => {
  if (!optimization) {
    return (
      <div className="fintech-card p-6 flex flex-col justify-center items-center min-h-[160px] text-center">
        <Scale className="h-8 w-8 text-slate-600 mb-2" />
        <h3 className="text-sm font-semibold text-slate-300">Strategy Comparison Matrix</h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1">
          Side-by-side performance comparison will appear once allocations are computed.
        </p>
      </div>
    );
  }

  const { equal_weight, min_volatility, max_sharpe } = optimization.allocations;

  const rows = [
    {
      metric: "Expected Return (CAGR)",
      eq: `${equal_weight.expected_return.toFixed(2)}%`,
      mv: `${min_volatility.expected_return.toFixed(2)}%`,
      ms: `${max_sharpe.expected_return.toFixed(2)}%`,
      better: "high",
    },
    {
      metric: "Annualized Volatility",
      eq: `${equal_weight.volatility.toFixed(2)}%`,
      mv: `${min_volatility.volatility.toFixed(2)}%`,
      ms: `${max_sharpe.volatility.toFixed(2)}%`,
      better: "low",
    },
    {
      metric: "Sharpe Ratio (Rf = 4.0%)",
      eq: equal_weight.sharpe_ratio.toFixed(2),
      mv: min_volatility.sharpe_ratio.toFixed(2),
      ms: max_sharpe.sharpe_ratio.toFixed(2),
      better: "high",
    },
    {
      metric: "Max Drawdown (MDD)",
      eq: `${equal_weight.max_drawdown.toFixed(2)}%`,
      mv: `${min_volatility.max_drawdown.toFixed(2)}%`,
      ms: `${max_sharpe.max_drawdown.toFixed(2)}%`,
      better: "high", // closer to 0 is better
    },
    {
      metric: "Historical VaR (95%)",
      eq: `${equal_weight.var_95_historical.toFixed(2)}%`,
      mv: `${min_volatility.var_95_historical.toFixed(2)}%`,
      ms: `${max_sharpe.var_95_historical.toFixed(2)}%`,
      better: "low",
    },
    {
      metric: "Historical CVaR (95%)",
      eq: `${equal_weight.cvar_95_historical.toFixed(2)}%`,
      mv: `${min_volatility.cvar_95_historical.toFixed(2)}%`,
      ms: `${max_sharpe.cvar_95_historical.toFixed(2)}%`,
      better: "low",
    },
    {
      metric: "Diversification Ratio",
      eq: equal_weight.diversification_ratio.toFixed(2),
      mv: min_volatility.diversification_ratio.toFixed(2),
      ms: max_sharpe.diversification_ratio.toFixed(2),
      better: "high",
    },
    {
      metric: "Effective Assets (N_eff)",
      eq: equal_weight.effective_assets.toFixed(1),
      mv: min_volatility.effective_assets.toFixed(1),
      ms: max_sharpe.effective_assets.toFixed(1),
      better: "neutral",
    },
  ];

  return (
    <div className="fintech-card p-6">
      <div className="flex items-center space-x-2.5 mb-5">
        <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
          <Scale className="h-4 w-4" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Portfolio Allocation & Risk Attribution Comparison
          </h3>
          <p className="text-xs text-slate-400">
            Side-by-side performance comparison of 1/N baseline vs convex optimal portfolios
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="py-3 px-4 font-semibold">Risk & Performance Metric</th>
              <th className="py-3 px-4 font-semibold text-right">Equal-Weight (1/N)</th>
              <th className="py-3 px-4 font-semibold text-right text-cyan-400">
                Minimum Volatility
              </th>
              <th className="py-3 px-4 font-semibold text-right text-indigo-400">
                Maximum Sharpe
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-slate-900/40 transition">
                <td className="py-3 px-4 text-slate-300 font-medium">{r.metric}</td>
                <td className="py-3 px-4 mono text-right text-slate-400">{r.eq}</td>
                <td className="py-3 px-4 mono text-right font-bold text-cyan-300">
                  {r.mv}
                </td>
                <td className="py-3 px-4 mono text-right font-bold text-indigo-300">
                  {r.ms}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
