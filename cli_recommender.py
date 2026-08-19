"""CLI for Game Recommender Pro.

Usage:
  python cli_recommender.py "Counter-Strike: Global Offensive"
  python cli_recommender.py "Dota 2" "Team Fortress 2" --k 5
  python cli_recommender.py "Hades" --genre Action --min-rating 0.7
  python cli_recommender.py --search "portal"
  python cli_recommender.py --genres
  python cli_recommender.py --random
  python cli_recommender.py "Portal 2" --export-json results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from app import _load_engine
from utils.recommender import RecommendationEngine


def main(argv: list[str] | None = None, engine: RecommendationEngine | None = None) -> int:
    parser = argparse.ArgumentParser(description="Content-based Steam game recommender")
    parser.add_argument("games", nargs="*", default=[], help="One or more seed game names")
    parser.add_argument("-k", "--num", type=int, default=9, help="Number of recommendations")
    parser.add_argument("--min-reviews", type=int, default=5000)
    parser.add_argument("--min-rating", type=float, default=0.65)
    parser.add_argument("--genre", action="append", dest="genres", default=None)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--multiplayer", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--export-json", type=str, default=None, help="Save recommendations to JSON file")
    parser.add_argument("--search", type=str, default=None, help="Search for games matching query")
    parser.add_argument("--genres", action="store_true", help="List top available genres")
    parser.add_argument("--random", action="store_true", help="Pick a random popular game as seed")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args(argv)

    if engine is None:
        try:
            engine = RecommendationEngine(use_faiss=not args.no_faiss)
        except Exception:
            engine = _load_engine()

    if args.genres:
        top_genres = engine.available_genres(30)
        print("Available genres:")
        for g in top_genres:
            print(f" - {g}")
        return 0

    if args.search:
        results = engine.search(args.search, limit=args.num)
        if not results:
            print(f"No games found matching: '{args.search}'")
            return 0
        print(f"Search results for '{args.search}':\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['Name']} ({res['Genres']})")
        return 0

    seeds = list(args.games)
    if args.random:
        rand_game = engine.random_game()
        if rand_game and rand_game.get("Name"):
            seeds.append(rand_game["Name"])
            print(f"Randomly selected seed: {rand_game['Name']}\n")

    if not seeds:
        parser.print_help()
        return 1

    recs = engine.recommend(
        game_names=seeds,
        num_recommendations=args.num,
        min_reviews=args.min_reviews,
        min_rating=args.min_rating,
        genres=args.genres,
        max_price=args.max_price,
        multiplayer_only=args.multiplayer,
    )

    if recs is None:
        print(f"Game(s) not found: {seeds}", file=sys.stderr)
        return 1

    if args.export_json:
        export_path = Path(args.export_json)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(recs)} recommendations to {export_path}")

    if args.json:
        print(json.dumps(recs, indent=2, ensure_ascii=False))
        return 0

    seeds_str = ", ".join(seeds)
    print(f"Recommendations for [{seeds_str}] ({len(recs)} results):\n")
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['Name']}")
        print(f"   Similarity: {rec['Similarity']:.4f} | Quality: {rec['Quality Score']:.4f}")
        print(f"   Rating: {rec['Rating']} ({rec['Total Reviews']:,} reviews)")
        print(f"   Genres: {rec['Genres']}")
        if rec.get("explanation"):
            print(f"   Why: {rec['explanation'].get('summary', '')}")
        print(f"   Link: {rec['Link Game']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
