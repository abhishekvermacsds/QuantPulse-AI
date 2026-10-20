import React from "react";
import { Grid } from "lucide-react";
import { CorrelationData } from "../types";

interface CorrelationHeatmapProps {
  correlation: CorrelationData | null;
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ correlation }) => {
  if (!correlation) {
    return (
      <div className="fintech-card p-6 flex flex-col justify-center items-center min-h-[300px] text-center">
        <Grid className="h-10 w-10 text-slate-600 mb-3" />
        <h3 className="text-sm font-semibold text-slate-300">Asset Correlation Heatmap</h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1">
          No correlation data available. Connect backend to compute pairwise returns.
        </p>
      </div>
    );
  }

  const { tickers, matrix } = correlation;

  const getCellColor = (val: number) => {
    if (val >= 0.99) return "bg-slate-800 text-slate-400";
    if (val > 0.6) return "bg-cyan-950/80 text-cyan-300 border-cyan-800/40";
    if (val > 0.3) return "bg-indigo-950/60 text-indigo-300 border-indigo-800/40";
    if (val > 0.0) return "bg-slate-900 text-slate-300";
    if (val > -0.2) return "bg-amber-950/60 text-amber-300 border-amber-800/40";
    return "bg-emerald-950/80 text-emerald-300 border-emerald-800/40 font-bold"; // Strong negative correlation (diversifier)
  };

  return (
    <div className="fintech-card p-6 flex flex-col justify-between">
      <div className="flex items-center space-x-2.5 mb-5">
        <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
          <Grid className="h-4 w-4" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Asset Correlation Heatmap
          </h3>
          <p className="text-xs text-slate-400">
            Pairwise Pearson return correlations (green indicates negative correlation hedge)
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-center text-xs border-collapse">
          <thead>
            <tr>
              <th className="p-2 text-left text-slate-500 font-semibold"></th>
              {tickers.map((t) => (
                <th key={t} className="p-2 font-bold text-slate-300">
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((rowTicker) => (
              <tr key={rowTicker}>
                <td className="p-2 text-left font-bold text-slate-300">
                  {rowTicker}
                </td>
                {tickers.map((colTicker) => {
                  const val = matrix[rowTicker]?.[colTicker] ?? 0.0;
                  return (
                    <td key={colTicker} className="p-1">
                      <div
                        className={`py-2 px-2.5 rounded border text-xs mono transition ${getCellColor(
                          val
                        )}`}
                        title={`${rowTicker} ↔ ${colTicker}: ${val.toFixed(3)}`}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400">
        <span className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded bg-emerald-500 inline-block" />
          <span>Hedge (&lt; -0.2)</span>
        </span>
        <span className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded bg-slate-700 inline-block" />
          <span>Uncorrelated (0.0)</span>
        </span>
        <span className="flex items-center space-x-1.5">
          <span className="w-2.5 h-2.5 rounded bg-cyan-500 inline-block" />
          <span>Coupled (&gt; 0.6)</span>
        </span>
      </div>
    </div>
  );
};
