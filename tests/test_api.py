"""Integration and unit tests for Flask API routes."""
from __future__ import annotations

import pytest


@pytest.fixture
def client(engine):
    import app as app_module

    app_module.rec_engine = engine
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "games" in data
    assert "latent_dim" in data


def test_game_names_endpoint(client):
    res = client.get("/game_names")
    assert res.status_code == 200
    names = res.get_json()
    assert isinstance(names, list)
    assert len(names) > 0


def test_genres_endpoint(client):
    res = client.get("/genres")
    assert res.status_code == 200
    genres = res.get_json()
    assert isinstance(genres, list)
    assert len(genres) > 0


def test_get_game_info_post(client):
    res = client.post("/get_game_info", json={"game_name": "Alpha Shooter"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["Name"] == "Alpha Shooter"
    assert "Genres" in data


def test_get_game_info_get(client):
    res = client.get("/get_game_info?game_name=Alpha Shooter")
    assert res.status_code == 200
    data = res.get_json()
    assert data["Name"] == "Alpha Shooter"


def test_get_game_info_empty(client):
    res = client.post("/get_game_info", json={"game_name": "   "})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_get_game_info_not_found(client):
    res = client.post("/get_game_info", json={"game_name": "Nonexistent Title 999"})
    assert res.status_code == 404
    assert "error" in res.get_json()


def test_recommend_missing_seed(client):
    res = client.post("/recommend", json={})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_recommend_with_filters(client):
    res = client.post(
        "/recommend",
        json={
            "game_name": "Alpha Shooter",
            "num_recommendations": 3,
            "min_reviews": 100,
            "min_rating": 0.5,
            "genres": ["Action", ""],
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)


def test_popular_games_endpoint(client):
    res = client.get("/popular_games?num=2")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)
    assert len(data) <= 2


def test_search_endpoint(client):
    res = client.get("/search?q=alpha&limit=3")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)
    assert len(data) <= 3


def test_random_game_endpoint(client):
    res = client.get("/random_game")
    assert res.status_code == 200
    data = res.get_json()
    assert "Name" in data
