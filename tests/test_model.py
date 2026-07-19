"""Smoke tests for autoencoder architecture."""
from __future__ import annotations

import torch

from utils.model import ImprovedAutoencoder


def test_autoencoder_shapes():
    model = ImprovedAutoencoder(input_dim=1289, latent_dim=128)
    model.eval()
    x = torch.randn(4, 1289)
    with torch.no_grad():
        recon = model(x)
        z = model.encode(x)
    assert recon.shape == (4, 1289)
    assert z.shape == (4, 128)
    # Sigmoid decoder outputs in [0, 1]
    assert float(recon.min()) >= 0.0
    assert float(recon.max()) <= 1.0
