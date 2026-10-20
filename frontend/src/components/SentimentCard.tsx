import React, { useState } from "react";
import { MessageSquare, Send, Sparkles } from "lucide-react";
import { SentimentOverview } from "../types";
import { api } from "../services/api";

interface SentimentCardProps {
  sentiment: SentimentOverview | null;
}

export const SentimentCard: React.FC<SentimentCardProps> = ({ sentiment }) => {
  const [customText, setCustomText] = useState("");
  const [customTicker, setCustomTicker] = useState("AAPL");
  const [classifying, setClassifying] = useState(false);
  const [customResult, setCustomResult] = useState<any | null>(null);
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const handleClassify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!customText.trim()) return;

    setClassifying(true);
    setClassifyError(null);
    try {
      const res = await api.classifyHeadline(customText.trim(), customTicker);
      setCustomResult(res.item);
    } catch (err: any) {
      setClassifyError(err.message || "Failed to classify headline.");
    } finally {
      setClassifying(false);
    }
  };

  const getLabelBadge = (label: string) => {
    switch (label.toLowerCase()) {
      case "positive":
        return "bg-emerald-950/80 text-emerald-400 border-emerald-800/40";
      case "negative":
        return "bg-rose-950/80 text-rose-400 border-rose-800/40";
      default:
        return "bg-slate-800 text-slate-300 border-slate-700/50";
    }
  };

  return (
    <div className="fintech-card p-6 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center space-x-2.5 mb-5">
        <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
          <MessageSquare className="h-4 w-4" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            FinBERT Financial News Sentiment
          </h3>
          <p className="text-xs text-slate-400">
            Transformer-based NLP classification & asset sentiment attribution
          </p>
        </div>
      </div>

      {/* Asset Breakdown Badges */}
      {sentiment && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 mb-5">
          {sentiment.asset_summaries.map((s) => (
            <div
              key={s.ticker}
              className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-bold text-white">{s.ticker}</span>
                <span
                  className={`mono font-bold text-xs ${
                    s.mean_sentiment_score > 0
                      ? "text-emerald-400"
                      : s.mean_sentiment_score < 0
                      ? "text-rose-400"
                      : "text-slate-400"
                  }`}
                >
                  {s.mean_sentiment_score > 0 ? "+" : ""}
                  {s.mean_sentiment_score.toFixed(2)}
                </span>
              </div>
              <div className="text-[10px] text-slate-400 flex justify-between">
                <span className="text-emerald-400">{s.bullish_ratio.toFixed(0)}% Bull</span>
                <span className="text-rose-400">{s.bearish_ratio.toFixed(0)}% Bear</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Recent Sample Headlines */}
      {sentiment && (
        <div className="space-y-2 mb-5 max-h-48 overflow-y-auto pr-1">
          {sentiment.recent_headlines.slice(0, 5).map((h, i) => (
            <div
              key={i}
              className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/80 flex items-start justify-between gap-3 text-xs"
            >
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  {h.ticker && (
                    <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-slate-800 text-slate-200">
                      {h.ticker}
                    </span>
                  )}
                  <span
                    className={`text-[10px] font-semibold px-1.5 py-0.2 rounded border uppercase ${getLabelBadge(
                      h.label
                    )}`}
                  >
                    {h.label} ({((h.confidence || 0) * 100).toFixed(0)}%)
                  </span>
                </div>
                <p className="text-slate-300 leading-relaxed text-[11px]">{h.headline}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {!sentiment && (
        <div className="text-xs text-slate-500 py-8 text-center border-b border-slate-800/80 mb-5">
          News sentiment feed unavailable. Start backend to load financial headlines.
        </div>
      )}

      {/* Interactive Live Headline Classifier */}
      <form onSubmit={handleClassify} className="pt-4 border-t border-slate-800">
        <span className="text-xs font-semibold text-slate-300 block mb-2 flex items-center space-x-1.5">
          <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
          <span>Test Live Headline Classification</span>
        </span>
        <div className="flex gap-2">
          <select
            value={customTicker}
            onChange={(e) => setCustomTicker(e.target.value)}
            className="bg-slate-900 border border-slate-800 text-xs text-white rounded-lg px-2.5 focus:outline-none focus:border-cyan-500"
          >
            {["AAPL", "MSFT", "JPM", "XOM", "SPY", "TLT"].map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Type or paste financial headline..."
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            className="flex-1 bg-slate-900 border border-slate-800 text-xs text-white rounded-lg px-3 py-2 focus:outline-none focus:border-cyan-500 placeholder:text-slate-600"
          />
          <button
            type="submit"
            disabled={classifying || !customText.trim()}
            className="px-3.5 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold transition flex items-center space-x-1 cursor-pointer"
          >
            <Send className="h-3 w-3" />
            <span>{classifying ? "Classifying..." : "Score"}</span>
          </button>
        </div>

        {classifyError && (
          <p className="text-xs text-rose-400 mt-2">{classifyError}</p>
        )}

        {customResult && (
          <div className="mt-3 p-2.5 rounded-lg bg-slate-900/90 border border-cyan-800/40 text-xs flex items-center justify-between">
            <span className="text-slate-300">
              Result:{" "}
              <strong className="text-white uppercase">{customResult.label}</strong>{" "}
              (Confidence: {(customResult.confidence * 100).toFixed(1)}%)
            </span>
            <span className="mono font-bold text-cyan-400">
              Score: {customResult.sentiment_score > 0 ? "+" : ""}
              {customResult.sentiment_score.toFixed(2)}
            </span>
          </div>
        )}
      </form>
    </div>
  );
};
