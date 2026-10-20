"""Time-series volatility forecasting endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
import numpy as np

from quantpulse.analytics.risk import annualized_volatility
from quantpulse.api.schemas import VolatilityForecastResponse
from quantpulse.core.config import PROJECT_ROOT
from quantpulse.data.loader import MarketDataLoader
from quantpulse.ml.inference import VolatilityInferenceEngine

router = APIRouter()
_loader = MarketDataLoader()
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved"


@router.get("/volatility/predict", response_model=VolatilityForecastResponse)
def predict_volatility(
    ticker: str = Query(default="SPY", description="Asset ticker symbol"),
) -> VolatilityForecastResponse:
    """Generate forward 5-day realized volatility prediction using PyTorch LSTM."""
    raw_df = _loader.load()
    symbol = ticker.upper()

    asset_df = raw_df[raw_df["Ticker"] == symbol].copy()
    if asset_df.empty:
        available = list(raw_df["Ticker"].unique())
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{symbol}' not found in dataset. Available assets: {available}",
        )

    asset_df.set_index("Date", inplace=True)
    rets = asset_df["Close"].pct_change().dropna()

    scale = np.sqrt(252.0)
    vol_21d = float(rets.iloc[-21:].std() * scale) if len(rets) >= 21 else 0.0
    vol_5d = float(rets.iloc[-5:].std() * scale) if len(rets) >= 5 else 0.0

    # Locate model checkpoint or fallback
    model_path = SAVED_MODELS_DIR / f"lstm_volatility_{symbol.lower()}.pt"
    if not model_path.exists():
        # Fallback to default spy model if available
        model_path = SAVED_MODELS_DIR / "lstm_volatility_spy.pt"

    forecast_val = vol_21d  # baseline default
    model_name = "Historical_21d_Baseline"

    if model_path.exists():
        try:
            inference = VolatilityInferenceEngine(checkpoint_path=model_path, device="cpu")
            forecast_val = inference.predict_latest(asset_df)
            model_name = "PyTorch_LSTM_Forecaster"
        except Exception:
            pass

    return VolatilityForecastResponse(
        ticker=symbol,
        as_of_date=asset_df.index.max().strftime("%Y-%m-%d"),
        forecasted_volatility=round(float(forecast_val) * 100.0, 2),
        historical_vol_21d=round(float(vol_21d) * 100.0, 2),
        historical_vol_5d=round(float(vol_5d) * 100.0, 2),
        model=model_name,
        context_bars=len(asset_df),
    )
