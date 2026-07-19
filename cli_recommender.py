"""CLI for Game Recommender Pro.

Usage:
  python cli_recommender.py "Counter-Strike: Global Offensive"
  python cli_recommender.py "Dota 2" "Team Fortress 2" --k 5
  python cli_recommender.py "Hades" --genre Action --min-rating 0.7
"""
from __future__ import annotations

import argparse
import json
import sys

from utils.recommender import RecommendationEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Content-based Steam game recommender")
    parser.add_argument("games", nargs="+", help="One or more seed game names")
    parser.add_argument("-k", "--num", type=int, default=9, help="Number of recommendations")
    parser.add_argument("--min-reviews", type=int, default=5000)
    parser.add_argument("--min-rating", type=float, default=0.65)
    parser.add_argument("--genre", action="append", dest="genres", default=None)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--multiplayer", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args(argv)

    engine = RecommendationEngine(use_faiss=not args.no_faiss)
    recs = engine.recommend(
        game_names=args.games,
        num_recommendations=args.num,
        min_reviews=args.min_reviews,
        min_rating=args.min_rating,
        genres=args.genres,
        max_price=args.max_price,
        multiplayer_only=args.multiplayer,
    )

    if recs is None:
        print(f"Game(s) not found: {args.games}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(recs, indent=2, ensure_ascii=False))
        return 0

    seeds = ", ".join(args.games)
    print(f"Recommendations for [{seeds}] ({len(recs)} results):\n")
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
