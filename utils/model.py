"""Autoencoder architecture matching Colab_Train.ipynb / best_model.pth."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from config import BEST_MODEL_PTH, DEFAULT_INPUT_DIM, DEFAULT_LATENT_DIM


class ImprovedAutoencoder(nn.Module):
    """Deep autoencoder: input → 1024 → 512 → 256 → latent → reverse decode."""

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def load_autoencoder(
    path: Optional[Union[str, Path]] = None,
    input_dim: int = DEFAULT_INPUT_DIM,
    latent_dim: int = DEFAULT_LATENT_DIM,
    device: Optional[torch.device] = None,
) -> ImprovedAutoencoder:
    """Load trained weights; falls back to random init if file missing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    weight_path = Path(path) if path else BEST_MODEL_PTH

    if weight_path.exists():
        try:
            state = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(f"Warning: model weights not found at {weight_path}; using random init")

    model.to(device)
    model.eval()
    return model
