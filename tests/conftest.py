"""Shared fixtures for unit tests (lightweight synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.recommender import RecommendationEngine


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    rows = [
        {
            "Name": "Alpha Shooter",
            "Header image": "http://img/a.jpg",
            "Short description": "Fast FPS multiplayer shooter",
            "Genres": "['Action', 'Free to Play']",
            "Tags": "['FPS', 'Shooter', 'Multiplayer', 'Competitive']",
            "Categories": "['Multi-player', 'Online Multi-Player']",
            "Positive": 9000,
            "Total Reviews": 10000,
            "Movies": "",
            "Link Game": "http://store/a",
            "Price": 0.0,
        },
        {
            "Name": "Beta Tactics",
            "Header image": "http://img/b.jpg",
            "Short description": "Tactical FPS team game",
            "Genres": "['Action', 'Strategy']",
            "Tags": "['FPS', 'Tactical', 'Multiplayer', 'Shooter']",
            "Categories": "['Multi-player']",
            "Positive": 8000,
            "Total Reviews": 9000,
            "Movies": "",
            "Link Game": "http://store/b",
            "Price": 19.99,
        },
        {
            "Name": "Cozy Farm",
            "Header image": "http://img/c.jpg",
            "Short description": "Relaxing farming simulator",
            "Genres": "['Simulation', 'Casual']",
            "Tags": "['Farming', 'Relaxing', 'Singleplayer', 'Cute']",
            "Categories": "['Single-player']",
            "Positive": 7000,
            "Total Reviews": 8000,
            "Movies": "",
            "Link Game": "http://store/c",
            "Price": 14.99,
        },
        {
            "Name": "Dungeon Quest",
            "Header image": "http://img/d.jpg",
            "Short description": "Action RPG dungeon crawler",
            "Genres": "['Action', 'RPG']",
            "Tags": "['RPG', 'Dungeon Crawler', 'Singleplayer', 'Fantasy']",
            "Categories": "['Single-player']",
            "Positive": 6000,
            "Total Reviews": 7000,
            "Movies": "",
            "Link Game": "http://store/d",
            "Price": 29.99,
        },
        {
            "Name": "Arena FPS",
            "Header image": "http://img/e.jpg",
            "Short description": "Arena shooter competitive",
            "Genres": "['Action']",
            "Tags": "['FPS', 'Arena', 'Multiplayer', 'Competitive']",
            "Categories": "['Multi-player', 'Online Multi-Player']",
            "Positive": 5000,
            "Total Reviews": 6000,
            "Movies": "",
            "Link Game": "http://store/e",
            "Price": 9.99,
        },
        {
            "Name": "Obscure Indie",
            "Header image": "http://img/f.jpg",
            "Short description": "Tiny experimental game",
            "Genres": "['Indie', 'Adventure']",
            "Tags": "['Experimental', 'Short', 'Indie']",
            "Categories": "['Single-player']",
            "Positive": 50,
            "Total Reviews": 80,
            "Movies": "",
            "Link Game": "http://store/f",
            "Price": 4.99,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_latent(tiny_df: pd.DataFrame) -> np.ndarray:
    """Hand-crafted latent vectors so FPS games cluster together."""
    rng = np.random.default_rng(0)
    dim = 16
    base_fps = np.ones(dim, dtype=np.float32)
    base_farm = np.zeros(dim, dtype=np.float32)
    base_farm[dim // 2 :] = 1.0
    base_rpg = np.zeros(dim, dtype=np.float32)
    base_rpg[::2] = 1.0

    vectors = [
        base_fps + rng.normal(0, 0.05, dim),  # Alpha
        base_fps + rng.normal(0, 0.05, dim),  # Beta
        base_farm + rng.normal(0, 0.05, dim),  # Cozy
        base_rpg + rng.normal(0, 0.05, dim),  # Dungeon
        base_fps + rng.normal(0, 0.05, dim),  # Arena
        rng.normal(0, 1, dim),  # Obscure
    ]
    return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def engine(tiny_df: pd.DataFrame, tiny_latent: np.ndarray) -> RecommendationEngine:
    return RecommendationEngine(df=tiny_df, latent_reps=tiny_latent, use_faiss=False)
