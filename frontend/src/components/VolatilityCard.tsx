import React from "react";
import { TrendingUp, Clock } from "lucide-react";
import { VolatilityForecast } from "../types";

interface VolatilityCardProps {
  volatility: VolatilityForecast | null;
  selectedTicker: string;
  onSelectTicker: (ticker: string) => void;
  availableTickers: string[];
  loading: boolean;
}

export const VolatilityCard: React.FC<VolatilityCardProps> = ({
  volatility,
  selectedTicker,
  onSelectTicker,
  availableTickers,
  loading,
}) => {
  return (
    <div className="fintech-card p-6 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
            <TrendingUp className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              LSTM Volatility Forecast
            </h3>
            <p className="text-xs text-slate-400">
              5-Day forward realized volatility prediction
            </p>
          </div>
        </div>

        {/* Ticker Selector */}
        <select
          value={selectedTicker}
          onChange={(e) => onSelectTicker(e.target.value)}
          disabled={loading}
          className="bg-slate-900 border border-slate-800 text-xs text-white rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500 cursor-pointer"
        >
          {availableTickers.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {volatility ? (
        <div>
          {/* Main Forecast Metric */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-950/40 to-slate-900 border border-cyan-800/30 mb-4">
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-xs uppercase font-semibold text-cyan-400">
                Forward 5-Day Realized Volatility
              </span>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/50">
                {volatility.model}
              </span>
            </div>
            <div className="text-3xl font-extrabold mono text-white">
              {volatility.forecasted_volatility.toFixed(2)}%
            </div>
            <div className="text-xs text-slate-400 mt-1 flex items-center space-x-1">
              <Clock className="h-3.5 w-3.5 text-slate-500" />
              <span>As of {volatility.as_of_date} ({volatility.context_bars} trading bars)</span>
            </div>
          </div>

          {/* Baseline Comparisons */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] font-medium text-slate-400 block mb-1">
                21-Day Historical Vol
              </span>
              <span className="text-base font-bold mono text-slate-200">
                {volatility.historical_vol_21d.toFixed(2)}%
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] font-medium text-slate-400 block mb-1">
                5-Day Historical Vol
              </span>
              <span className="text-base font-bold mono text-slate-200">
                {volatility.historical_vol_5d.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-xs text-slate-500 py-8 text-center">
          {loading ? "Loading volatility forecasts..." : "Volatility forecast unavailable. Connect backend to compute."}
        </div>
      )}
    </div>
  );
};
