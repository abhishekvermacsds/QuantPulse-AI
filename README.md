# QuantPulse AI — Autonomous Portfolio Risk & Volatility Optimization Engine

QuantPulse AI is an enterprise-grade quantitative portfolio risk, volatility forecasting, and allocation engine. Built for quantitative analysts, portfolio managers, and risk engineers, it bridges mathematical finance, deep learning, financial news sentiment, distributed background queues, persistent storage, and modern web analytics.

---

## 🏛️ System Architecture

```text
                               ┌─────────────────────────────────────────────────────────┐
                               │               PostgreSQL 16 Persistence                 │
                               │  - Portfolios & Allocations    - Asset Price Bars       │
                               │  - Optimization Runs           - Backtest Records       │
                               │  - Volatility Forecasts        - FinBERT Sentiments     │
                               └────────────────────────────┬────────────────────────────┘
                                                            │ SQLAlchemy 2.0 / Alembic
                               ┌────────────────────────────┴────────────────────────────┐
                               ▼                                                         ▼
                [ Market Price Engine ]                                       [ News & Headline Feed ]
           (OHLCV, Merton Jump-Diffusion,                                 (Timestamped Financial Headlines)
             Point-in-Time Slicing, No Leakage)                                          │
                               │                                                         ▼
                               ▼                                              [ NLP Sentiment Engine ]
                [ Risk & Volatility Analytics ]                               (FinBERT Financial Sentiment)
           (CAGR, Vol, Sharpe, Sortino, MDD,                                             │
            Parametric/Historical VaR, CVaR)                                             │
                               │                                                         │
                               ├─────────────────────────────┬───────────────────────────┘
                               ▼                             ▼
                    [ ML Volatility Engine ]           [ Sentiment Score ]
                     (PyTorch LSTM Forecast)                 │
                               │                             │
                               └──────────────┬──────────────┘
                                              ▼
                               [ Portfolio Optimization Engine ]
                               (PyPortfolioOpt / CVXPY Solvers)
                               - Mean-Variance Efficient Frontier
                               - Minimum Volatility & Max Sharpe
                               - Dynamic Risk-Budgeting / CVaR
                                              │
                                              ▼
                                   [ Execution / API Layer ]
                            FastAPI REST Endpoints + Redis Cache
                               (TTL Caching & Parameter Hashing)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Asynchronous Task Worker ]                       [ Analytics Dashboard ]
          Celery + Redis Broker                              React + TypeScript + Vite
       (Model Training, Batch News)                         (Dark Fintech Dashboard)
```

---

## ✨ Core Features

1. **Synthetic Market Generation & Look-Ahead Safe Loader**:
   - Merton Jump-Diffusion process simulating fat-tailed returns, volatility clustering, and macroeconomic market shocks.
   - Look-ahead bias prevention via strictly timestamped point-in-time window slicing.
2. **Comprehensive Risk & Performance Analytics**:
   - CAGR, Annualized Volatility, Downside Semi-Deviation, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Calmar Ratio.
   - Non-parametric Historical Value at Risk (VaR 95%), Parametric Cornish-Fisher VaR (skewness/kurtosis adjusted), and Conditional VaR (Expected Shortfall / CVaR).
3. **Convex Portfolio Optimization**:
   - Equal-Weighted baseline reference portfolio.
   - Minimum Volatility optimization (global variance minimization on convex boundary).
   - Maximum Sharpe Ratio optimization (tangency portfolio with Ledoit-Wolf covariance shrinkage and spectral positive semi-definite projection).
   - Configurable maximum allocation per asset and diversification analytics (Diversification Ratio, Herfindahl-Hirschman Index, Effective Number of Assets $N_{eff}$).
4. **Deep Time-Series Volatility Forecasting**:
   - PyTorch LSTM neural network predicting 5-day forward realized volatility.
   - Chronological train/validation/test splits, in-sample robust feature scaling, and Softplus activation for strictly non-negative outputs.
   - Benchmarked out-of-sample against 21-day historical rolling volatility baselines using MAE and RMSE.
5. **Financial News Sentiment Analysis (FinBERT)**:
   - NLP headline classification into `positive`, `negative`, and `neutral` using `ProsusAI/finbert`.
   - Continuous compound sentiment scores: $P(\text{Positive}) - P(\text{Negative}) \in [-1.0, 1.0]$.
   - Lazy loading and explicit memory release (`unload_model()`) to preserve RAM on 8 GB MacBook M1.
6. **Distributed Systems & Persistence**:
   - PostgreSQL 16 persistence for portfolios, asset prices, optimization history, forecast records, and backtest results.
   - SQLAlchemy 2.0 and Alembic database migrations.
   - Redis caching with MD5 parameter hashing and graceful in-memory offline fallback.
   - Celery background workers for asynchronous model training and batch news ingestion.
7. **Modern Web Analytics Dashboard**:
   - React 18, Vite 5, TypeScript, Tailwind CSS, and Lucide React.
   - Interactive weight allocation charts, correlation heatmaps, live headline tester, and side-by-side strategy attribution matrices.

---

## 🖥️ UI & Analytics Dashboard Preview

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  QuantPulse AI  v0.1.0        [ API Online | MPS | 309 MB ]             [ Refresh Data ]│
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [ Min-Vol Volatility ]    [ Max Sharpe Ratio ]     [ Value at Risk (95%) ]  [ Diversification Ratio ] │
│         12.18%                    1.42                      -1.84%                    1.62             │
│   -24.1% vs Benchmark      Rf = 4.0% Tangency          Historical Tail Loss     5.1 Effective Assets   │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│  ASSET ALLOCATION WEIGHTS                │  LSTM VOLATILITY FORECAST                   │
│  [Equal-Weight] [Min-Vol] [Max-Sharpe]   │  Asset: [ SPY ▼ ]                           │
│  AAPL: 22.0% ■■■■■■■■                    │  Forward 5-Day Realized Volatility: 14.12%   │
│  MSFT: 24.0% ■■■■■■■■■                   │  21-Day Historical Vol: 16.20%              │
│  JPM:  14.0% ■■■■■                       │  5-Day Historical Vol:  15.08%              │
│  XOM:  10.0% ■■■■                        ├─────────────────────────────────────────────┤
│  SPY:  15.0% ■■■■■                       │  FINBERT FINANCIAL NEWS SENTIMENT           │
│  TLT:  15.0% ■■■■■                       │  AAPL (+0.75): 85% Bullish | 15% Bearish    │
│  Max Weight Cap: [───●──────] 35%        │  MSFT (+0.62): 78% Bullish | 22% Bearish    │
├──────────────────────────────────────────┴─────────────────────────────────────────────┤
│  STRATEGY COMPARISON MATRIX                                                            │
│  Metric                        Equal-Weight         Min-Volatility        Max-Sharpe   │
│  Expected Return (CAGR)           14.20%                12.45%              18.60%     │
│  Annualized Volatility            16.05%                12.18%              14.35%     │
│  Sharpe Ratio (Rf = 4.0%)          0.64                  0.69                1.02      │
│  Max Drawdown (MDD)              -19.45%               -14.20%             -16.10%     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Directory Layout

```text
QuantPulse-AI/
├── .dockerignore                   # Docker build exclusion rules
├── .env.example                    # Environment configuration template
├── .gitignore                      # Git exclusion rules
├── Dockerfile.api                  # FastAPI backend container
├── Dockerfile.frontend             # Multi-stage React + Vite + Nginx container
├── Dockerfile.worker               # Celery background worker container
├── README.md                       # Architecture & setup guide
├── alembic/                        # Database migration scripts
│   ├── env.py                      # Dynamic database configuration
│   └── versions/                   # Versioned DDL migration scripts
├── alembic.ini                     # Alembic migration configuration
├── docker-compose.yml              # Multi-container orchestration (Postgres, Redis, API, Worker, UI)
├── pyproject.toml                  # PEP 621 package specification
├── requirements.txt                # Production Python dependencies
├── setup.py                        # Setuptools package configuration
├── data/
│   ├── sample/
│   │   ├── market_data.csv         # 3-year multi-asset synthetic market dataset
│   │   └── financial_headlines.csv # Sample financial news headlines
│   ├── raw/                        # Ingested raw market feeds
│   └── processed/                  # Normalized feature datasets
├── frontend/                       # React + Vite + TypeScript dashboard
│   ├── nginx.conf                  # Production Nginx reverse-proxy configuration
│   ├── vite.config.ts              # Vite config with /api reverse proxy
│   ├── src/
│   │   ├── components/             # KPI Cards, Allocations, Heatmap, Sentiment, Volatility
│   │   ├── services/api.ts         # Resilient fetch API with timeouts and proxy routing
│   │   └── types.ts                # TypeScript domain interfaces
├── quantpulse/                     # Core Python package
│   ├── analytics/                  # Financial risk, returns, and VaR/CVaR calculations
│   ├── api/                        # FastAPI REST API routes and Pydantic schemas
│   ├── core/                       # Config and Redis caching with offline fallbacks
│   ├── data/                       # Synthetic jump-diffusion generator & point-in-time loader
│   ├── db/                         # SQLAlchemy 2.0 persistence models, sessions & seeders
│   ├── ml/                         # PyTorch LSTM volatility forecaster & trainer
│   ├── optimization/               # PyPortfolioOpt & CVXPY convex portfolio solvers
│   ├── sentiment/                  # FinBERT financial NLP sentiment classifier
│   └── tasks/                      # Celery distributed tasks (training, batch sentiment)
├── examples/                       # Standalone runnable verification scripts
└── tests/                          # Automated Pytest suite (83 tests, 100% passing)
```

---

## 📡 REST API Endpoints Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health, memory footprint (MB), device (`mps`/`cpu`), database and cache status |
| `GET` | `/api/portfolio/metrics` | Full returns, risk, and VaR/CVaR profile across asset universe |
| `GET` | `/api/portfolio/correlation` | Pairwise Pearson return correlation matrix |
| `POST` | `/api/portfolio/optimize` | Convex portfolio optimization (`min_volatility`, `max_sharpe`, `equal_weight`) |
| `GET` | `/api/volatility/predict` | 5-day forward realized volatility forecast using PyTorch LSTM |
| `GET` | `/api/sentiment/headlines` | FinBERT financial news sentiment aggregation by ticker |
| `POST` | `/api/sentiment/classify` | On-demand NLP classification for custom user headlines |
| `GET` | `/api/portfolios` | List all persisted portfolios from PostgreSQL/SQLite |
| `POST` | `/api/portfolios` | Create and store a new portfolio |
| `GET` | `/api/portfolios/{id}/optimizations` | Retrieve historical optimization runs for a portfolio |
| `POST` | `/api/db/seed` | Seed database tables with historical sample price bars and news |
| `POST` | `/api/tasks/volatility/train` | Dispatch background LSTM model training via Celery |
| `POST` | `/api/tasks/sentiment/batch` | Dispatch background FinBERT batch sentiment analysis |
| `GET` | `/api/tasks/{task_id}` | Poll asynchronous task status (`PENDING`, `PROGRESS`, `SUCCESS`, `FAILURE`) |

---

## 🚀 Local macOS Apple Silicon (M1, 8 GB RAM) Run Instructions

### Step 1: Open Terminal 1 — Start the FastAPI Backend

```bash
cd /Users/abhishekverma/projects/QuantPulse-AI

# 1. Activate Python virtual environment
source .venv/bin/activate

# 2. Copy the environment configuration template
cp -n .env.example .env

# 3. Apply database migrations
alembic upgrade head

# 4. Seed historical price bars and default portfolio
python -m quantpulse.db.seed

# 5. Start FastAPI backend with hot-reload
uvicorn quantpulse.api.app:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- API Diagnostics: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### Step 2: Open Terminal 2 — Start the React Dashboard

```bash
cd /Users/abhishekverma/projects/QuantPulse-AI/frontend

# Launch the Vite development server
npm run dev
```
- Open [http://localhost:5173](http://localhost:5173) in your browser.
*(Note: If port 5173 is occupied, Vite will automatically start on port 5174; the configured reverse proxy in `vite.config.ts` ensures uninterrupted backend communication).*

---

## 🐳 Docker Compose Deployment (Multi-Container)

When Docker Desktop is installed, you can launch the complete production environment with a single command:

```bash
docker compose up --build
```

**Services:**
- `postgres`: PostgreSQL 16 on port `5432`
- `redis`: Redis 7 on port `6379`
- `api`: FastAPI backend on port `8000`
- `worker`: Celery asynchronous worker
- `frontend`: React / Vite application served via Nginx reverse proxy on port `3000`

---

## 🧪 Automated Testing

QuantPulse AI includes a test suite covering financial risk formulas, convex optimization solvers, deep learning time-series pipelines, NLP sentiment, Redis caching, and database persistence.

```bash
source .venv/bin/activate
pytest -v
```
**Test Results**: **83 passed**, 0 failed in 6.19 seconds.

---

## ⚠️ Limitations & Educational Disclaimer

1. **Analytical & Research Purpose**: QuantPulse AI is an algorithmic quantitative finance portfolio analysis and simulation engine. It does **not** execute live market orders, provide financial advice, or connect to brokerage accounts.
2. **Model Horizon**: The PyTorch LSTM model currently produces single-horizon 5-day forward volatility forecasts. Multi-horizon autoregressive forecasts will be introduced in subsequent updates.
3. **Macroeconomic Shocks**: Merton Jump-Diffusion synthetic bars simulate fat-tailed statistical properties but do not reflect unexpected geopolitical or interest rate announcements.
