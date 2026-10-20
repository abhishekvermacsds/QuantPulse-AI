import {
  CorrelationData,
  HealthStatus,
  OptimizationData,
  PortfolioMetricsData,
  SentimentOverview,
  VolatilityForecast,
} from "../types";

// Prefer relative '/api' in dev & production to leverage Vite & Nginx reverse proxies,
// or fallback to configured VITE_API_URL or localhost.
const rawApiUrl = import.meta.env.VITE_API_URL;
const API_BASE = rawApiUrl !== undefined && rawApiUrl !== "" ? rawApiUrl : "/api";

const DEFAULT_TIMEOUT_MS = 8000;

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
    return res;
  } catch (error: any) {
    if (error.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s. Verify the FastAPI backend is running.`);
    }
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        `Unable to reach QuantPulse API at ${url}. Ensure the backend is active at http://127.0.0.1:8000.`
      );
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchJson<T>(url: string, options?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const res = await fetchWithTimeout(url, options, timeoutMs);

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // ignore JSON parse error
    }
    throw new Error(errorDetail);
  }

  return (await res.json()) as T;
}

export const api = {
  getHealth: () => fetchJson<HealthStatus>(`${API_BASE}/health`, undefined, 3000),

  getPortfolioMetrics: () =>
    fetchJson<PortfolioMetricsData>(`${API_BASE}/portfolio/metrics`),

  getCorrelation: () =>
    fetchJson<CorrelationData>(`${API_BASE}/portfolio/correlation`),

  optimizePortfolio: (maxWeight = 0.35, riskFreeRate = 0.04) =>
    fetchJson<OptimizationData>(`${API_BASE}/portfolio/optimize`, {
      method: "POST",
      body: JSON.stringify({
        objective: "all",
        max_weight: maxWeight,
        risk_free_rate: riskFreeRate,
        include_frontier: true,
      }),
    }),

  getVolatilityForecast: (ticker = "SPY") =>
    fetchJson<VolatilityForecast>(`${API_BASE}/volatility/predict?ticker=${ticker}`),

  getSentimentOverview: () =>
    fetchJson<SentimentOverview>(`${API_BASE}/sentiment/headlines`),

  classifyHeadline: (headline: string, ticker?: string) =>
    fetchJson<{ item: any }>(`${API_BASE}/sentiment/classify`, {
      method: "POST",
      body: JSON.stringify({ headline, ticker }),
    }),
};
