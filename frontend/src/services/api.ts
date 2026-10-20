import {
  CorrelationData,
  HealthStatus,
  OptimizationData,
  PortfolioMetricsData,
  SentimentOverview,
  VolatilityForecast,
} from "../types";

// Resolve API_BASE from VITE_API_URL environment variable, query param, or intelligent fallback
function resolveApiBase(): string {
  // 1. Check URL query param (?api=https://...) for live debugging / override
  if (typeof window !== "undefined" && window.location) {
    const params = new URLSearchParams(window.location.search);
    const queryApi = params.get("api");
    if (queryApi) {
      const clean = queryApi.trim().replace(/\/+$/, "");
      return clean.endsWith("/api") ? clean : `${clean}/api`;
    }
  }

  // 2. Check build-time VITE_API_URL environment variable
  const envUrl = (import.meta.env.VITE_API_URL || "").trim();
  if (envUrl) {
    const clean = envUrl.replace(/\/+$/, "");
    return clean.endsWith("/api") ? clean : `${clean}/api`;
  }

  // 3. Fallback for production deployment on Render (*.onrender.com)
  if (typeof window !== "undefined" && window.location && window.location.hostname.includes("onrender.com")) {
    return "https://quantpulse-api.onrender.com/api";
  }

  // 4. Fallback to local Vite dev server proxy
  return "/api";
}

const API_BASE = resolveApiBase();

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
        `Unable to reach QuantPulse API at ${url}. Ensure the backend is active at ${API_BASE}.`
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
