export interface HealthStatus {
  status: string;
  version: string;
  device: string;
  memory_mb: number;
  timestamp: string;
}

export interface AssetMetrics {
  ticker: string;
  cagr: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  var_95_historical: number;
  var_95_cornish_fisher: number;
  cvar_95_historical: number;
}

export interface PortfolioMetricsData {
  start_date: string;
  end_date: string;
  trading_days: number;
  asset_universe: string[];
  metrics: AssetMetrics[];
}

export interface CorrelationData {
  tickers: string[];
  matrix: Record<string, Record<string, number>>;
}

export interface AllocationProfile {
  objective: string;
  weights: Record<string, number>;
  expected_return: number;
  volatility: number;
  sharpe_ratio: number;
  diversification_ratio: number;
  effective_assets: number;
  max_drawdown: number;
  var_95_historical: number;
  cvar_95_historical: number;
  message: string;
}

export interface OptimizationData {
  allocations: {
    equal_weight: AllocationProfile;
    min_volatility: AllocationProfile;
    max_sharpe: AllocationProfile;
  };
  frontier?: Array<{
    return: number;
    volatility: number;
    sharpe: number;
  }>;
}

export interface VolatilityForecast {
  ticker: string;
  as_of_date: string;
  forecasted_volatility: number;
  historical_vol_21d: number;
  historical_vol_5d: number;
  model: string;
  context_bars: number;
}

export interface HeadlineItem {
  headline: string;
  ticker?: string;
  timestamp?: string;
  label: "positive" | "negative" | "neutral" | string;
  confidence: number;
  sentiment_score: number;
  probabilities: Record<string, number>;
}

export interface AssetSentimentSummary {
  ticker: string;
  headline_count: number;
  mean_sentiment_score: number;
  bullish_ratio: number;
  bearish_ratio: number;
  neutral_ratio: number;
}

export interface SentimentOverview {
  total_headlines: number;
  mean_score: number;
  asset_summaries: AssetSentimentSummary[];
  recent_headlines: HeadlineItem[];
}
