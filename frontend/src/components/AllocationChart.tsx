import React from "react";
import { PieChart, Sliders } from "lucide-react";
import { OptimizationData } from "../types";

interface AllocationChartProps {
  optimization: OptimizationData | null;
  selectedStrategy: "min_volatility" | "max_sharpe" | "equal_weight";
  onSelectStrategy: (strat: "min_volatility" | "max_sharpe" | "equal_weight") => void;
  maxWeight: number;
  onMaxWeightChange: (val: number) => void;
  onReoptimize: () => void;
  loading: boolean;
}

const ASSET_COLORS: Record<string, string> = {
  AAPL: "#38bdf8", // Sky blue
  MSFT: "#818cf8", // Indigo
  JPM: "#34d399",  // Emerald
  XOM: "#fbbf24",  // Amber
  SPY: "#f43f5e",  // Rose
  TLT: "#a78bfa",  // Purple
};

export const AllocationChart: React.FC<AllocationChartProps> = ({
  optimization,
  selectedStrategy,
  onSelectStrategy,
  maxWeight,
  onMaxWeightChange,
  onReoptimize,
  loading,
}) => {
  if (!optimization) {
    return (
      <div className="fintech-card p-6 flex flex-col justify-center items-center min-h-[360px] text-center">
        <PieChart className="h-10 w-10 text-slate-600 mb-3" />
        <h3 className="text-sm font-semibold text-slate-300">Asset Allocation Weights</h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1">
          {loading ? "Calculating optimal portfolio weights..." : "No allocation data available. Start the backend to compute weights."}
        </p>
      </div>
    );
  }

  const activeAlloc = optimization.allocations[selectedStrategy];
  const weights = activeAlloc ? activeAlloc.weights : {};
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);

  return (
    <div className="fintech-card p-6 flex flex-col justify-between">
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
            <PieChart className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Asset Allocation Weights
            </h3>
            <p className="text-xs text-slate-400">
              Convex weight optimization with configurable asset caps
            </p>
          </div>
        </div>

        {/* Strategy Switcher */}
        <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
          {(["equal_weight", "min_volatility", "max_sharpe"] as const).map((strat) => (
            <button
              key={strat}
              onClick={() => onSelectStrategy(strat)}
              className={`px-3 py-1.5 rounded-md font-medium transition cursor-pointer ${
                selectedStrategy === strat
                  ? "bg-cyan-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {strat === "equal_weight" && "Equal-Weight"}
              {strat === "min_volatility" && "Min-Vol"}
              {strat === "max_sharpe" && "Max-Sharpe"}
            </button>
          ))}
        </div>
      </div>

      {/* Allocation Stack Bar */}
      <div className="mb-6">
        <div className="h-6 w-full rounded-lg overflow-hidden flex bg-slate-900 border border-slate-800">
          {entries.map(([ticker, weight]) => {
            const pct = (weight * 100);
            if (pct <= 0.5) return null;
            return (
              <div
                key={ticker}
                style={{
                  width: `${pct}%`,
                  backgroundColor: ASSET_COLORS[ticker] || "#64748b",
                }}
                className="h-full transition-all duration-500 relative group flex items-center justify-center text-[10px] font-bold text-slate-950 truncate px-1"
                title={`${ticker}: ${pct.toFixed(1)}%`}
              >
                {pct >= 8 && ticker}
              </div>
            );
          })}
        </div>
      </div>

      {/* Asset Breakdown List */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-6">
        {entries.map(([ticker, weight]) => {
          const pct = (weight * 100);
          const color = ASSET_COLORS[ticker] || "#64748b";
          return (
            <div
              key={ticker}
              className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-center justify-between"
            >
              <div className="flex items-center space-x-2">
                <span
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs font-bold text-white">{ticker}</span>
              </div>
              <span className="text-xs font-bold mono text-slate-200">
                {pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Live Constraint Slider */}
      <div className="pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3 text-xs">
          <Sliders className="h-4 w-4 text-slate-400" />
          <span className="text-slate-300 font-medium">Max Asset Cap:</span>
          <span className="mono font-bold text-cyan-400">
            {(maxWeight * 100).toFixed(0)}%
          </span>
          <input
            type="range"
            min={0.20}
            max={0.50}
            step={0.05}
            value={maxWeight}
            onChange={(e) => onMaxWeightChange(parseFloat(e.target.value))}
            className="w-28 accent-cyan-500 cursor-pointer"
          />
        </div>

        <button
          onClick={onReoptimize}
          disabled={loading}
          className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 rounded-lg text-xs font-semibold transition border border-slate-700 cursor-pointer"
        >
          {loading ? "Optimizing..." : "Recompute Optimization"}
        </button>
      </div>
    </div>
  );
};
