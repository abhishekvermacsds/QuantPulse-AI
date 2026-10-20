"""PyTorch LSTM architecture for financial volatility forecasting."""

from typing import Optional
import torch
import torch.nn as nn


class LSTMVolatilityForecaster(nn.Module):
    """Deep Recurrent LSTM model designed for time-series volatility regression.

    Architecture:
    - Multi-layer LSTM capturing temporal dependencies across sequential trading days.
    - Fully connected regression head with non-linear hidden activation.
    - Softplus activation output constraint ensuring strictly positive volatility forecasts: sigma > 0.
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Regression Head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # Softplus guarantees smooth, positive volatility forecasts: log(1 + exp(x))
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
            Predicted annualized forward volatility.
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the representation at the final timestep: (batch_size, hidden_size)
        last_step_repr = lstm_out[:, -1, :]
        out = self.fc_head(last_step_repr)
        return out

    def count_parameters(self) -> int:
        """Count total trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_default_device(prefer_mps: bool = True) -> torch.device:
    """Detect optimal compute device: Apple Silicon MPS or CPU."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
