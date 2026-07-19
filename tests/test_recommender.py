"""Unit tests for RecommendationEngine."""
from __future__ import annotations

import numpy as np
import pytest

from utils.parsing import jaccard, parse_list_field
from utils.recommender import RecommendationEngine


def test_parse_list_field():
    assert parse_list_field("['Action', 'RPG']") == ["Action", "RPG"]
    assert parse_list_field("Action, RPG") == ["Action", "RPG"]
    assert parse_list_field(None) == []
    assert parse_list_field(float("nan")) == []


def test_jaccard():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 0.0


def test_recommend_basic(engine: RecommendationEngine):
    recs = engine.recommend(
        game_name="Alpha Shooter",
        num_recommendations=3,
        min_reviews=1000,
        min_rating=0.5,
    )
    assert recs is not None
    assert len(recs) == 3
    names = [r["Name"] for r in recs]
    # Seed title must never appear in results
    assert "Alpha Shooter" not in names
    # Should prefer other FPS-like titles over farm/rpg
    assert "Cozy Farm" not in names[:2]
    assert all("explanation" in r for r in recs)
    assert all("Similarity" in r for r in recs)


def test_duplicate_name_excluded(tiny_df, tiny_latent):
    """If catalog has two rows with the same title, both must be excluded."""
    import pandas as pd

    df = pd.concat([tiny_df, tiny_df.iloc[[0]].assign(AppID=999)], ignore_index=True)
    # extend latent with a near-copy of the first vector
    lat = np.vstack([tiny_latent, tiny_latent[0] + 0.01])
    eng = RecommendationEngine(df=df, latent_reps=lat, use_faiss=False)
    recs = eng.recommend(
        game_name="Alpha Shooter",
        num_recommendations=5,
        min_reviews=0,
        min_rating=0.0,
    )
    assert recs is not None
    assert all(r["Name"] != "Alpha Shooter" for r in recs)


def test_game_not_found(engine: RecommendationEngine):
    assert engine.recommend(game_name="Does Not Exist 999") is None


def test_quality_filter_excludes_obscure(engine: RecommendationEngine):
    recs = engine.recommend(
        game_name="Alpha Shooter",
        num_recommendations=10,
        min_reviews=5000,
        min_rating=0.65,
    )
    assert recs is not None
    names = [r["Name"] for r in recs]
    assert "Obscure Indie" not in names


def test_multi_seed(engine: RecommendationEngine):
    recs = engine.recommend(
        game_names=["Alpha Shooter", "Beta Tactics"],
        num_recommendations=2,
        min_reviews=1000,
        min_rating=0.5,
    )
    assert recs is not None
    assert len(recs) >= 1
    # Seeds excluded
    names = {r["Name"] for r in recs}
    assert "Alpha Shooter" not in names
    assert "Beta Tactics" not in names


def test_genre_filter(engine: RecommendationEngine):
    recs = engine.recommend(
        game_name="Alpha Shooter",
        num_recommendations=5,
        min_reviews=100,
        min_rating=0.0,
        genres=["RPG"],
    )
    assert recs is not None
    assert any("RPG" in str(r["Genres"]) for r in recs)


def test_multiplayer_filter(engine: RecommendationEngine):
    recs = engine.recommend(
        game_name="Alpha Shooter",
        num_recommendations=5,
        min_reviews=100,
        min_rating=0.0,
        multiplayer_only=True,
    )
    assert recs is not None
    for r in recs:
        assert r["Name"] != "Cozy Farm"


def test_max_price_filter(engine: RecommendationEngine):
    recs = engine.recommend(
        game_name="Alpha Shooter",
        num_recommendations=5,
        min_reviews=100,
        min_rating=0.0,
        max_price=15.0,
    )
    assert recs is not None
    assert all(r["Price"] <= 15.0 for r in recs)


def test_latent_vector_query(engine: RecommendationEngine):
    vec = engine.latent_raw[0]
    recs = engine.recommend(
        latent_vector=vec,
        num_recommendations=2,
        min_reviews=100,
        min_rating=0.0,
    )
    assert recs is not None
    assert len(recs) == 2


def test_search_and_popular(engine: RecommendationEngine):
    hits = engine.search("shoot", limit=5)
    assert any("Shooter" in h["Name"] or "FPS" in h["Name"] or "Alpha" in h["Name"] for h in hits)
    popular = engine.get_popular_games(3)
    assert len(popular) == 3
    info = engine.get_game_info("Cozy Farm")
    assert info is not None
    assert info["Name"] == "Cozy Farm"


def test_row_count_mismatch():
    df_small = __import__("pandas").DataFrame(
        {
            "Name": ["A"],
            "Header image": ["x"],
            "Short description": ["y"],
            "Genres": ["['Action']"],
            "Tags": ["['FPS']"],
            "Categories": ["['Single-player']"],
            "Positive": [1],
            "Total Reviews": [1],
            "Movies": [""],
            "Link Game": ["u"],
            "Price": [0],
        }
    )
    with pytest.raises(ValueError):
        RecommendationEngine(df=df_small, latent_reps=np.zeros((2, 8), dtype=np.float32), use_faiss=False)
