"""Inference engine providing clean production prediction pipelines decoupled from training code."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch

from quantpulse.ml.dataset import VolatilityFeaturePipeline
from quantpulse.ml.model import LSTMVolatilityForecaster
from quantpulse.ml.trainer import load_checkpoint


class VolatilityInferenceEngine:
    """Production inference engine for forward volatility forecasting.

    Decoupled from training:
    - Loads pre-trained model weights and feature scaler from a checkpoint.
    - Operates on live or historical price bars without training code or label requirements.
    - Memory-efficient forward pass on CPU or Apple Silicon MPS.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model, self.scaler, self.config = load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.pipeline = VolatilityFeaturePipeline()
        self.sequence_length = self.config.sequence_length

    def predict_sequence(self, recent_market_data: pd.DataFrame) -> np.ndarray:
        """Generate forward volatility forecasts from recent market bars.

        Parameters
        ----------
        recent_market_data : pd.DataFrame
            Market data containing at least (sequence_length + 25) historical bars
            to compute rolling technical indicators.

        Returns
        -------
        np.ndarray of shape (num_forecasts,)
            Forecasted forward annualized volatility values.
        """
        features_df, _ = self.pipeline.build_features(recent_market_data)
        if len(features_df) < self.sequence_length:
            raise ValueError(
                f"Insufficient historical bars ({len(features_df)}) after feature extraction. "
                f"Need at least sequence_length ({self.sequence_length}) bars."
            )

        scaled_features = self.scaler.transform(features_df)
        num_sequences = len(scaled_features) - self.sequence_length + 1

        sequences = []
        for i in range(num_sequences):
            sequences.append(scaled_features[i : i + self.sequence_length])

        seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds_tensor = self.model(seq_tensor)
            preds = preds_tensor.cpu().numpy().flatten()

        return preds

    def predict_latest(self, recent_market_data: pd.DataFrame) -> float:
        """Generate the single most recent forward volatility prediction."""
        preds = self.predict_sequence(recent_market_data)
        return float(preds[-1])
