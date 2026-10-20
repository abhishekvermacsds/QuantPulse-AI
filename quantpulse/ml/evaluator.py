"""Evaluation metrics and comparative benchmarking for volatility regression models."""

from typing import Dict, Union
import numpy as np
import pandas as pd


def calculate_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculate standard regression performance metrics: MAE, RMSE, MAPE, and R2.

    Parameters
    ----------
    y_true : array-like
        Actual ground-truth forward realized volatilities.
    y_pred : array-like
        Model volatility forecasts.

    Returns
    -------
    dict
        Dictionary containing calculated metrics rounded to 5 decimal places.
    """
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(yt) != len(yp):
        raise ValueError(f"Shape mismatch: y_true ({len(yt)}) and y_pred ({len(yp)}).")
    if len(yt) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    errors = yp - yt
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))

    # MAPE: avoid division by zero
    non_zero_mask = yt > 1e-6
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(errors[non_zero_mask] / yt[non_zero_mask])) * 100.0)
    else:
        mape = np.nan

    # R-squared
    ss_tot = float(np.sum((yt - np.mean(yt))**2))
    ss_res = float(np.sum(errors**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "MAE": round(mae, 5),
        "RMSE": round(rmse, 5),
        "MAPE(%)": round(mape, 2),
        "R2": round(r2, 4),
    }


def compare_forecast_models(
    y_true: Union[np.ndarray, pd.Series],
    model_predictions: Dict[str, Union[np.ndarray, pd.Series]],
    baseline_name: str = "Historical_Baseline",
) -> pd.DataFrame:
    """Compare multiple volatility forecasting models side-by-side against a baseline."""
    rows = {}
    for model_name, preds in model_predictions.items():
        rows[model_name] = calculate_regression_metrics(y_true, preds)

    df_comp = pd.DataFrame(rows).T

    # If baseline is present, compute percentage improvement in RMSE and MAE
    if baseline_name in df_comp.index:
        base_rmse = df_comp.loc[baseline_name, "RMSE"]
        base_mae = df_comp.loc[baseline_name, "MAE"]

        df_comp["RMSE_Improvement(%)"] = (
            (base_rmse - df_comp["RMSE"]) / base_rmse * 100.0
        ).round(2)
        df_comp["MAE_Improvement(%)"] = (
            (base_mae - df_comp["MAE"]) / base_mae * 100.0
        ).round(2)

    return df_comp
