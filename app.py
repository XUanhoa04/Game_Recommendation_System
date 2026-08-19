"""Flask application for Game Recommender Pro."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

import numpy as np
from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from utils.recommender import RecommendationEngine

app = Flask(__name__)


def _load_engine() -> RecommendationEngine:
    try:
        engine = RecommendationEngine(use_faiss=True)
        print(f"Loaded {len(engine.df)} games (FAISS={engine.use_faiss})")
        return engine
    except Exception as exc:
        print(f"Notice: Using fallback test catalog ({exc})")
        import pandas as pd

        fallback_df = pd.DataFrame(
            [
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
            ]
        )
        fallback_latent = np.ones((len(fallback_df), 128), dtype=np.float32)
        return RecommendationEngine(df=fallback_df, latent_reps=fallback_latent, use_faiss=False)


print("Loading recommendation engine...")
rec_engine = _load_engine()

# Lazy optional encoder for online cold-start path
_feature_builder = None


def get_feature_builder():
    global _feature_builder
    if _feature_builder is None:
        from utils.features import FeatureBuilder

        _feature_builder = FeatureBuilder(load_bert=False)
    return _feature_builder


def _parse_filters(data: Dict[str, Any]) -> Dict[str, Any]:
    genres = data.get("genres") or data.get("genre")
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]
    elif isinstance(genres, (list, tuple, set)):
        genres = [str(g).strip() for g in genres if str(g).strip()]
    else:
        genres = None

    max_price = data.get("max_price")
    if max_price is not None and max_price != "":
        try:
            max_price = max(0.0, float(max_price))
        except (ValueError, TypeError):
            max_price = None
    else:
        max_price = None

    try:
        num_recs = max(1, int(data.get("num_recommendations", 9)))
    except (ValueError, TypeError):
        num_recs = 9

    try:
        min_revs = max(0, int(data.get("min_reviews", 5000)))
    except (ValueError, TypeError):
        min_revs = 5000

    try:
        min_rat = max(0.0, min(1.0, float(data.get("min_rating", 0.65))))
    except (ValueError, TypeError):
        min_rat = 0.65

    return {
        "num_recommendations": num_recs,
        "min_reviews": min_revs,
        "min_rating": min_rat,
        "genres": genres if genres else None,
        "max_price": max_price,
        "multiplayer_only": bool(data.get("multiplayer_only", False)),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "games": len(rec_engine.df),
            "faiss": rec_engine.use_faiss,
            "latent_dim": rec_engine.dim,
        }
    )


@app.route("/game_names", methods=["GET"])
def game_names():
    try:
        return jsonify(rec_engine.names)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/genres", methods=["GET"])
def genres():
    return jsonify(rec_engine.available_genres())


@app.route("/get_game_info", methods=["GET", "POST"])
def get_game_info():
    if request.method == "GET":
        game_name = request.args.get("game_name") or request.args.get("name") or ""
    else:
        data = request.get_json(silent=True) or {}
        game_name = data.get("game_name", "")

    game_name = str(game_name).strip()
    if not game_name:
        return jsonify({"error": "game_name is required"}), 400

    info = rec_engine.get_game_info(game_name)
    if info is None:
        return jsonify({"error": "Game not found"}), 404
    return jsonify(info)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    filters = _parse_filters(data)

    # Multi-seed support: game_names list or single game_name
    game_names: Optional[List[str]] = data.get("game_names")
    game_name = data.get("game_name")
    if game_names and isinstance(game_names, list):
        seeds = [str(n).strip() for n in game_names if str(n).strip()]
    elif game_name:
        seeds = [str(game_name).strip()]
    else:
        seeds = []

    if not seeds:
        return jsonify({"error": "game_name or game_names required"}), 400

    recommendations = rec_engine.recommend(game_names=seeds, **filters)
    if recommendations is None:
        return jsonify([])
    return jsonify(recommendations)


@app.route("/recommend_encode", methods=["POST"])
def recommend_encode():
    """Cold-start: encode free-form game metadata and recommend similar titles.

    Body: genres, tags, categories, description, short_description + filters.
    Requires transformers + torch model weights (heavy; optional).
    """
    data = request.get_json(silent=True) or {}
    filters = _parse_filters(data)
    try:
        builder = get_feature_builder()
        latent = builder.encode_game(
            genres=data.get("genres", []),
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            description=data.get("description", ""),
            short_description=data.get("short_description", ""),
        )
        results = rec_engine.recommend_from_features(latent_vector=latent[0], **filters)
        return jsonify(results)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/random_game", methods=["GET"])
def random_game():
    return jsonify(rec_engine.random_game())


@app.route("/popular_games", methods=["GET"])
def popular_games():
    num_games = request.args.get("num", 20, type=int)
    if num_games is None or num_games <= 0:
        num_games = 20
    num_games = min(num_games, 100)
    return jsonify(rec_engine.get_popular_games(num_games))


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    limit = request.args.get("limit", 10, type=int)
    if limit is None or limit <= 0:
        limit = 10
    limit = min(limit, 50)
    return jsonify(rec_engine.search(query, limit=limit))


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
