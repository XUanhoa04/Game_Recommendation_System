"""Flask application for Game Recommender Pro."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from utils.recommender import RecommendationEngine

app = Flask(__name__)

print("Loading recommendation engine...")
rec_engine = RecommendationEngine(use_faiss=True)
print(f"Loaded {len(rec_engine.df)} games (FAISS={rec_engine.use_faiss})")

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
    max_price = data.get("max_price")
    if max_price is not None and max_price != "":
        max_price = float(max_price)
    else:
        max_price = None
    return {
        "num_recommendations": int(data.get("num_recommendations", 9)),
        "min_reviews": int(data.get("min_reviews", 5000)),
        "min_rating": float(data.get("min_rating", 0.65)),
        "genres": genres,
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


@app.route("/get_game_info", methods=["POST"])
def get_game_info():
    data = request.get_json(silent=True) or {}
    game_name = data.get("game_name", "")
    info = rec_engine.get_game_info(game_name)
    if info is None:
        return jsonify({}), 404
    return jsonify(info)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    filters = _parse_filters(data)

    # Multi-seed support: game_names list or single game_name
    game_names: Optional[List[str]] = data.get("game_names")
    game_name = data.get("game_name")
    if game_names and isinstance(game_names, list):
        seeds = [str(n) for n in game_names if n]
    elif game_name:
        seeds = [str(game_name)]
    else:
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
    return jsonify(rec_engine.get_popular_games(num_games))


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    limit = request.args.get("limit", 10, type=int)
    return jsonify(rec_engine.search(query, limit=limit))


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
