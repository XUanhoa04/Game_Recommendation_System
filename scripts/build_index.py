"""Build and persist FAISS index from latent representations.

Usage:
  python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FAISS_INDEX_PATH  # noqa: E402
from utils.recommender import RecommendationEngine  # noqa: E402


def main() -> int:
    engine = RecommendationEngine(use_faiss=True)
    if not engine.use_faiss or engine._faiss_index is None:
        print("FAISS is not available in this environment.")
        return 1
    engine.save_faiss_index(FAISS_INDEX_PATH)
    print(f"Saved FAISS index ({len(engine.df)} vectors, dim={engine.dim}) → {FAISS_INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
