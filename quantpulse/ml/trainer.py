"""Reproducible training engine with validation tracking, early stopping, and checkpointing."""

from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quantpulse.ml.dataset import (
    RobustFeatureScaler,
    VolatilitySequenceDataset,
    create_chronological_splits,
)
from quantpulse.ml.model import LSTMVolatilityForecaster, get_default_device


def set_seed(seed: int = 42) -> None:
    """Enforce strict determinism across Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Note: some MPS operations have deterministic mode
        pass


class TrainingConfig(BaseModel):
    """Hyperparameters and configuration for volatility model training."""

    epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.002, gt=0.0)
    sequence_length: int = Field(default=20, gt=0)
    hidden_size: int = Field(default=32, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    patience: int = Field(default=8, gt=0, description="Early stopping patience")
    seed: int = Field(default=42)
    device: str = Field(default="auto")


class VolatilityTrainer:
    """Manages training execution, early stopping, and loss history tracking."""

    def __init__(
        self,
        model: LSTMVolatilityForecaster,
        config: TrainingConfig,
    ):
        self.config = config
        set_seed(self.config.seed)

        if self.config.device == "auto":
            self.device = get_default_device(prefer_mps=True)
        else:
            self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )
        # Huber Loss (Smooth L1) provides robust gradients against market shock outliers
        self.criterion = nn.HuberLoss(delta=0.01)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.best_model_weights: Optional[Dict] = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * len(batch_x)

        return running_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate validation or test loss."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                running_loss += loss.item() * len(batch_x)

        return running_loss / len(dataloader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Execute full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save best weights
                self.best_model_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        # Restore optimal checkpoint
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_completed": len(self.train_losses),
        }


def save_checkpoint(
    model: LSTMVolatilityForecaster,
    scaler: RobustFeatureScaler,
    config: TrainingConfig,
    destination: Union[str, Path],
) -> Path:
    """Save trained model weights, feature scaler parameters, and metadata."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "scaler_dict": scaler.to_dict(),
        "config_dict": config.model_dump(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
    }
    torch.save(payload, dest)
    return dest


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[LSTMVolatilityForecaster, RobustFeatureScaler, TrainingConfig]:
    """Load model and feature scaler from disk."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    target_device = device or torch.device("cpu")
    payload = torch.load(path, map_location=target_device)

    model = LSTMVolatilityForecaster(
        input_size=payload["input_size"],
        hidden_size=payload["hidden_size"],
        num_layers=payload["num_layers"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(target_device)
    model.eval()

    scaler = RobustFeatureScaler.from_dict(payload["scaler_dict"])
    config = TrainingConfig(**payload["config_dict"])

    return model, scaler, config
