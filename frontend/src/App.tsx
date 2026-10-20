import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  Percent,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { Navbar } from "./components/Navbar";
import { KpiCard } from "./components/KpiCard";
import { AllocationChart } from "./components/AllocationChart";
import { CorrelationHeatmap } from "./components/CorrelationHeatmap";
import { VolatilityCard } from "./components/VolatilityCard";
import { SentimentCard } from "./components/SentimentCard";
import { OptimizationComparison } from "./components/OptimizationComparison";
import { api } from "./services/api";
import {
  CorrelationData,
  HealthStatus,
  OptimizationData,
  PortfolioMetricsData,
  SentimentOverview,
  VolatilityForecast,
} from "./types";

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [metrics, setMetrics] = useState<PortfolioMetricsData | null>(null);
  const [correlation, setCorrelation] = useState<CorrelationData | null>(null);
  const [optimization, setOptimization] = useState<OptimizationData | null>(null);
  const [volatility, setVolatility] = useState<VolatilityForecast | null>(null);
  const [sentiment, setSentiment] = useState<SentimentOverview | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedStrategy, setSelectedStrategy] = useState<
    "min_volatility" | "max_sharpe" | "equal_weight"
  >("min_volatility");
  const [selectedTicker, setSelectedTicker] = useState("SPY");
  const [maxWeight, setMaxWeight] = useState(0.35);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [hRes, mRes, cRes, optRes, vRes, sRes] = await Promise.allSettled([
        api.getHealth(),
        api.getPortfolioMetrics(),
        api.getCorrelation(),
        api.optimizePortfolio(maxWeight),
        api.getVolatilityForecast(selectedTicker),
        api.getSentimentOverview(),
      ]);

      if (hRes.status === "fulfilled") {
        setHealth(hRes.value);
      } else {
        setHealth(null);
      }

      if (mRes.status === "fulfilled") {
        setMetrics(mRes.value);
      }

      if (cRes.status === "fulfilled") {
        setCorrelation(cRes.value);
      }

      if (optRes.status === "fulfilled") {
        setOptimization(optRes.value);
      }

      if (vRes.status === "fulfilled") {
        setVolatility(vRes.value);
      }

      if (sRes.status === "fulfilled") {
        setSentiment(sRes.value);
      }

      // Check if backend is completely unreachable
      const failures = [hRes, mRes, cRes, optRes, vRes, sRes].filter(
        (r) => r.status === "rejected"
      ) as PromiseRejectedResult[];

      if (failures.length === 6) {
        setError(
          "Backend Offline: Unable to connect to QuantPulse API. Please ensure the backend is running at http://127.0.0.1:8000 (run: uvicorn quantpulse.api.app:app --host 127.0.0.1 --port 8000 --reload)."
        );
      } else if (failures.length > 0) {
        setError(
          `Partial data loaded: ${failures.length} service(s) reported an issue (${failures[0].reason?.message || "Check server log"}).`
        );
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to QuantPulse API.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleTickerChange = async (ticker: string) => {
    setSelectedTicker(ticker);
    try {
      const v = await api.getVolatilityForecast(ticker);
      setVolatility(v);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleReoptimize = async () => {
    setLoading(true);
    try {
      const opt = await api.optimizePortfolio(maxWeight);
      setOptimization(opt);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Derive KPI metrics from optimization
  const minVolProfile = optimization?.allocations.min_volatility;
  const eqProfile = optimization?.allocations.equal_weight;
  const volReduction =
    minVolProfile && eqProfile
      ? ((eqProfile.volatility - minVolProfile.volatility) / eqProfile.volatility) * 100
      : 0;

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      <Navbar health={health} loading={loading} onRefresh={loadData} />

      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        {/* Error Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-200 flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2.5">
              <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
            <button
              onClick={loadData}
              className="px-3 py-1 bg-rose-800 hover:bg-rose-700 text-white rounded font-medium cursor-pointer"
            >
              Retry
            </button>
          </div>
        )}

        {/* Top KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          <KpiCard
            title="Min-Vol Portfolio Volatility"
            value={minVolProfile ? `${minVolProfile.volatility.toFixed(2)}%` : "..."}
            subtext={
              eqProfile
                ? `Equal-Weight benchmark: ${eqProfile.volatility.toFixed(2)}%`
                : undefined
            }
            icon={TrendingDown}
            badge={volReduction > 0 ? `-${volReduction.toFixed(1)}% Vol` : undefined}
            badgeColor="green"
          />

          <KpiCard
            title="Max Sharpe Ratio"
            value={
              optimization?.allocations.max_sharpe
                ? optimization.allocations.max_sharpe.sharpe_ratio.toFixed(2)
                : "..."
            }
            subtext="Annualized excess return per unit risk (Rf = 4.0%)"
            icon={TrendingUp}
            badge="Tangency"
            badgeColor="cyan"
          />

          <KpiCard
            title="Value at Risk (95% Daily)"
            value={
              minVolProfile
                ? `${minVolProfile.var_95_historical.toFixed(2)}%`
                : "..."
            }
            subtext="Non-parametric historical tail loss limit"
            icon={ShieldAlert}
            badge="Tail Risk"
            badgeColor="slate"
          />

          <KpiCard
            title="Diversification Ratio"
            value={
              minVolProfile
                ? minVolProfile.diversification_ratio.toFixed(2)
                : "..."
            }
            subtext={
              minVolProfile
                ? `${minVolProfile.effective_assets.toFixed(1)} Effective Assets (N_eff)`
                : undefined
            }
            icon={Percent}
            badge="Choueifaty DR"
            badgeColor="cyan"
          />
        </div>

        {/* 2-Column Analytics Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Allocations & Correlations */}
          <div className="space-y-8">
            <AllocationChart
              optimization={optimization}
              selectedStrategy={selectedStrategy}
              onSelectStrategy={setSelectedStrategy}
              maxWeight={maxWeight}
              onMaxWeightChange={setMaxWeight}
              onReoptimize={handleReoptimize}
              loading={loading}
            />

            <CorrelationHeatmap correlation={correlation} />
          </div>

          {/* Right Column: Volatility & News Sentiment */}
          <div className="space-y-8">
            <VolatilityCard
              volatility={volatility}
              selectedTicker={selectedTicker}
              onSelectTicker={handleTickerChange}
              availableTickers={metrics?.asset_universe || ["SPY", "AAPL", "MSFT", "JPM", "XOM", "TLT"]}
              loading={loading}
            />

            <SentimentCard sentiment={sentiment} />
          </div>
        </div>

        {/* Bottom Matrix: Full Side-by-Side Attribution Table */}
        <OptimizationComparison optimization={optimization} />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 px-6 text-center text-xs text-slate-500">
        <p className="max-w-3xl mx-auto">
          QuantPulse AI is an algorithmic quantitative finance portfolio analysis and simulation engine.
          Metrics and simulated allocations are for research and portfolio risk engineering.
          No automated real-money trading orders are executed.
        </p>
      </footer>
    </div>
  );
};
