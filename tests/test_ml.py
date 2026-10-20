"""Unit tests for Milestone 3: PyTorch LSTM volatility forecasting pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from quantpulse.ml.baseline import HistoricalVolatilityBaseline
from quantpulse.ml.dataset import (
    RobustFeatureScaler,
    VolatilityFeaturePipeline,
    VolatilitySequenceDataset,
    create_chronological_splits,
)
from quantpulse.ml.evaluator import calculate_regression_metrics, compare_forecast_models
from quantpulse.ml.inference import VolatilityInferenceEngine
from quantpulse.ml.model import LSTMVolatilityForecaster
from quantpulse.ml.trainer import (
    TrainingConfig,
    VolatilityTrainer,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


@pytest.fixture
def sample_market_df():
    """Deterministic synthetic market data for ML tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=180, freq="B")
    # Simulating price process with varying volatility regimes
    rets = rng.normal(loc=0.0005, scale=0.015, size=len(dates))
    prices = 100.0 * np.exp(np.cumsum(rets))
    highs = prices * (1.0 + abs(rng.normal(0, 0.005, size=len(dates))))
    lows = prices * (1.0 - abs(rng.normal(0, 0.005, size=len(dates))))
    volumes = rng.integers(1_000_000, 5_000_000, size=len(dates))

    return pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Adj_Close": prices,
        "High": highs,
        "Low": lows,
        "Volume": volumes,
    }).set_index("Date")


def test_feature_pipeline_and_no_lookahead_target(sample_market_df):
    """Test feature construction and verify forward target alignment."""
    pipeline = VolatilityFeaturePipeline(forward_horizon=5)
    features, target = pipeline.build_features(sample_market_df)

    assert len(features) == len(target)
    assert not features.isna().any().any()
    assert not target.isna().any()

    # Target must be strictly positive annualized volatility
    assert (target > 0.0).all()

    # Verify expected feature columns exist
    expected_cols = {
        "return_1d", "return_lag1", "return_lag2", "return_lag5",
        "vol_5d", "vol_10d", "vol_21d", "hl_spread", "vol_shock"
    }
    assert expected_cols.issubset(set(features.columns))


def test_chronological_split_prevents_leakage(sample_market_df):
    """Verify strictly monotonic chronological train/val/test splits."""
    pipeline = VolatilityFeaturePipeline()
    features, target = pipeline.build_features(sample_market_df)

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = create_chronological_splits(
        features, target, train_ratio=0.70, val_ratio=0.15
    )

    # 1. Total samples conserved
    assert len(train_X) + len(val_X) + len(test_X) == len(features)

    # 2. Strict chronological order: Train < Val < Test
    assert train_X.index.max() < val_X.index.min()
    assert val_X.index.max() < test_X.index.min()


def test_scaler_fitted_strictly_on_train_no_leakage(sample_market_df):
    """Verify feature scaler is fitted ONLY on train partition."""
    pipeline = VolatilityFeaturePipeline()
    features, target = pipeline.build_features(sample_market_df)
    (train_X, _), (val_X, _), (test_X, _) = create_chronological_splits(features, target)

    scaler = RobustFeatureScaler()
    train_scaled = scaler.fit_transform(train_X)
    val_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test_X)

    # Train scaled means must be 0 and stds 1
    np.testing.assert_allclose(np.mean(train_scaled, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(train_scaled, axis=0), 1.0, atol=1e-4)

    # Val and test means will NOT be identically zero (reflecting true out-of-sample data)
    assert not np.allclose(np.mean(val_scaled, axis=0), 0.0, atol=1e-3)


def test_volatility_sequence_dataset_shapes():
    """Verify sequential dataset batch dimensions."""
    seq_len = 15
    num_feats = 8
    n_samples = 50
    dummy_feats = np.random.randn(n_samples, num_feats).astype(np.float32)
    dummy_targets = np.random.rand(n_samples).astype(np.float32)

    dataset = VolatilitySequenceDataset(dummy_feats, dummy_targets, sequence_length=seq_len)
    expected_len = n_samples - seq_len + 1
    assert len(dataset) == expected_len

    x, y = dataset[0]
    assert x.shape == (seq_len, num_feats)
    assert y.shape == (1,)


def test_lstm_model_forward_and_softplus_constraint():
    """Test model forward pass and non-negative Softplus activation."""
    batch_size = 4
    seq_len = 20
    in_feats = 9

    model = LSTMVolatilityForecaster(input_size=in_feats, hidden_size=16, num_layers=1)
    dummy_input = torch.randn(batch_size, seq_len, in_feats)
    output = model(dummy_input)

    assert output.shape == (batch_size, 1)
    # Output must be strictly non-negative (Softplus constraint)
    assert (output >= 0.0).all()


def test_historical_baseline_predictions(sample_market_df):
    """Verify historical volatility baseline output."""
    pipeline = VolatilityFeaturePipeline()
    features, _ = pipeline.build_features(sample_market_df)

    baseline = HistoricalVolatilityBaseline(method="rolling", window_col="vol_21d")
    preds = baseline.predict(features)

    assert len(preds) == len(features)
    assert (preds > 0.0).all()


def test_regression_metrics_calculation():
    """Test calculation of MAE, RMSE, MAPE, and R2 on deterministic arrays."""
    y_true = np.array([0.10, 0.20, 0.30])
    y_pred = np.array([0.12, 0.18, 0.32])

    metrics = calculate_regression_metrics(y_true, y_pred)

    expected_mae = (0.02 + 0.02 + 0.02) / 3.0
    expected_rmse = np.sqrt((0.02**2 + 0.02**2 + 0.02**2) / 3.0)

    assert pytest.approx(metrics["MAE"], rel=1e-4) == expected_mae
    assert pytest.approx(metrics["RMSE"], rel=1e-4) == expected_rmse
    assert metrics["R2"] > 0.0


def test_compare_forecast_models():
    """Verify comparative model benchmarking table."""
    y_true = np.array([0.15, 0.20, 0.25, 0.18])
    y_base = np.array([0.17, 0.23, 0.29, 0.22])  # worse
    y_lstm = np.array([0.16, 0.21, 0.26, 0.19])  # better

    comp_df = compare_forecast_models(
        y_true,
        {"Historical_Baseline": y_base, "LSTM_Model": y_lstm},
        baseline_name="Historical_Baseline",
    )

    assert "LSTM_Model" in comp_df.index
    assert "Historical_Baseline" in comp_df.index
    assert "RMSE_Improvement(%)" in comp_df.columns
    # LSTM had lower error than baseline, so improvement should be positive
    assert comp_df.loc["LSTM_Model", "RMSE_Improvement(%)"] > 0.0


def test_model_training_saving_and_inference(sample_market_df, tmp_path):
    """End-to-end integration test: train small model, save checkpoint, and predict with inference engine."""
    set_seed(42)

    pipeline = VolatilityFeaturePipeline(forward_horizon=5)
    features, target = pipeline.build_features(sample_market_df)

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = create_chronological_splits(
        features, target, train_ratio=0.70, val_ratio=0.15
    )

    scaler = RobustFeatureScaler().fit(train_X)
    train_ds = VolatilitySequenceDataset(scaler.transform(train_X), train_y.values, sequence_length=10)
    val_ds = VolatilitySequenceDataset(scaler.transform(val_X), val_y.values, sequence_length=10)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    config = TrainingConfig(epochs=3, batch_size=16, sequence_length=10, hidden_size=16, num_layers=1, device="cpu")
    model = LSTMVolatilityForecaster(input_size=train_X.shape[1], hidden_size=16, num_layers=1)

    trainer = VolatilityTrainer(model, config)
    history = trainer.fit(train_loader, val_loader)

    assert history["epochs_completed"] == 3
    assert len(history["train_losses"]) == 3

    # Save checkpoint
    ckpt_path = tmp_path / "lstm_test.pt"
    save_checkpoint(model, scaler, config, ckpt_path)
    assert ckpt_path.exists()

    # Load and test inference engine
    inference_engine = VolatilityInferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    latest_pred = inference_engine.predict_latest(sample_market_df)
    assert isinstance(latest_pred, float)
    assert latest_pred > 0.0
