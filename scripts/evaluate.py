"""Offline evaluation: autoencoder latent vs baselines.

Proxy ground truth = high tag/genre Jaccard overlap.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --sample 200 --k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set

import numpy as np
import pandas as pd

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LATENT_REPS_NPY, PROCESSED_GAMES_CSV  # noqa: E402
from utils.metrics import (  # noqa: E402
    aggregate_metrics,
    average_precision,
    build_proxy_relevant,
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from utils.parsing import parse_list_field  # noqa: E402
from utils.recommender import RecommendationEngine, _l2_normalize  # noqa: E402


def _tfidf_matrix(texts: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    # Convert to dense L2-normalized rows for cosine via matmul
    dense = matrix.astype(np.float32).toarray()
    return _l2_normalize(dense)


def _raw_feature_proxy(df: pd.DataFrame) -> np.ndarray:
    """Lightweight multi-hot genre+tag proxy (no BERT) for a baseline."""
    from sklearn.preprocessing import MultiLabelBinarizer

    genres = [parse_list_field(v) for v in df["Genres"]]
    tags = [parse_list_field(v) for v in df["Tags"]]
    mlb_g = MultiLabelBinarizer(sparse_output=False)
    mlb_t = MultiLabelBinarizer(sparse_output=False)
    g = mlb_g.fit_transform(genres).astype(np.float32)
    t = mlb_t.fit_transform(tags).astype(np.float32)
    # Weight genres higher (match training intent)
    feats = np.hstack([g * 2.0, t * 1.5]).astype(np.float32)
    return _l2_normalize(feats)


def _topk_from_matrix(
    matrix: np.ndarray,
    query_idx: int,
    k: int,
    exclude: Set[int],
) -> List[int]:
    sims = matrix @ matrix[query_idx]
    # mask excluded
    sims = sims.copy()
    for e in exclude:
        sims[e] = -np.inf
    if k >= len(sims):
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k)[:k]
        order = part[np.argsort(-sims[part])]
    return [int(i) for i in order[:k] if np.isfinite(sims[i])]


def _popular_baseline(
    popularity: np.ndarray,
    query_idx: int,
    k: int,
) -> List[int]:
    order = np.argsort(-popularity)
    return [int(i) for i in order if i != query_idx][:k]


def evaluate(
    sample_size: int = 150,
    k: int = 10,
    seed: int = 42,
    min_reviews: int = 0,
    min_rating: float = 0.0,
) -> Dict:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(PROCESSED_GAMES_CSV)
    latent = np.load(LATENT_REPS_NPY)

    engine = RecommendationEngine(df=df, latent_reps=latent, use_faiss=True)

    genre_sets = engine.genre_sets
    tag_sets = engine.tag_sets

    # Baselines matrices (built once)
    print("Building TF-IDF baseline...")
    desc = (
        df["Short description"].fillna("").astype(str)
        + " "
        + df.get("combined_desc", pd.Series([""] * len(df))).fillna("").astype(str)
    )
    tfidf = _tfidf_matrix(desc.tolist())

    print("Building raw multi-hot baseline...")
    raw = _raw_feature_proxy(df)

    n = len(df)
    # Prefer well-tagged games as queries
    tag_counts = np.array([len(t) for t in tag_sets])
    eligible = np.where(tag_counts >= 5)[0]
    if len(eligible) < sample_size:
        eligible = np.arange(n)
    query_ids = rng.choice(eligible, size=min(sample_size, len(eligible)), replace=False)

    latent_norm = engine.latent  # already L2-normalized

    def ae_pure_cosine(q: int) -> List[int]:
        """Pure latent cosine (fair vs other content baselines)."""
        return _topk_from_matrix(latent_norm, q, k, {q})

    def ae_quality_rank(q: int) -> List[int]:
        """Product ranking: similarity + rating prior + optional filters."""
        return engine.recommend_indices_only(
            q,
            k=k,
            min_reviews=min_reviews,
            min_rating=min_rating,
            candidate_pool=max(500, k * 20),
        )

    methods: Dict[str, Callable[[int], List[int]]] = {
        "popular": lambda q: _popular_baseline(engine.popularity, q, k),
        "tfidf_desc": lambda q: _topk_from_matrix(tfidf, q, k, {q}),
        "raw_multihot": lambda q: _topk_from_matrix(raw, q, k, {q}),
        "autoencoder_cosine": ae_pure_cosine,
        "autoencoder_quality": ae_quality_rank,
    }

    results: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    all_rec_ids: Dict[str, List[int]] = {m: [] for m in methods}
    latencies: Dict[str, List[float]] = {m: [] for m in methods}

    print(f"Evaluating {len(query_ids)} queries @K={k}...")
    for qi, q in enumerate(query_ids):
        relevant = build_proxy_relevant(int(q), genre_sets, tag_sets)
        if len(relevant) < 3:
            continue

        for name, fn in methods.items():
            t0 = time.perf_counter()
            recs = fn(int(q))
            latencies[name].append((time.perf_counter() - t0) * 1000)

            rec_tag_sets = [tag_sets[i] for i in recs if i < len(tag_sets)]
            row = {
                f"precision@{k}": precision_at_k(relevant, recs, k),
                f"recall@{k}": recall_at_k(relevant, recs, k),
                f"map@{k}": average_precision(relevant, recs, k),
                f"ndcg@{k}": ndcg_at_k(relevant, recs, k),
                "diversity": intra_list_diversity(rec_tag_sets),
            }
            results[name].append(row)
            all_rec_ids[name].extend(recs)

        if (qi + 1) % 25 == 0:
            print(f"  processed {qi + 1}/{len(query_ids)}")

    summary = {}
    for name in methods:
        metrics = aggregate_metrics(results[name])
        metrics["coverage"] = catalog_coverage(all_rec_ids[name], n)
        metrics["latency_ms_mean"] = float(np.mean(latencies[name])) if latencies[name] else 0.0
        metrics["n_queries"] = len(results[name])
        summary[name] = metrics

    return {
        "k": k,
        "sample_size": sample_size,
        "seed": seed,
        "methods": summary,
    }


def print_table(report: Dict) -> None:
    methods = report["methods"]
    if not methods:
        print("No results.")
        return

    metric_keys = [
        k
        for k in next(iter(methods.values())).keys()
        if k not in ("n_queries",)
    ]
    col_w = 16
    header = f"{'method':22s}" + "".join(f"{m:>{col_w}s}" for m in metric_keys)
    print(header)
    print("-" * len(header))
    for name, metrics in methods.items():
        row = f"{name:22s}"
        for m in metric_keys:
            val = metrics.get(m, 0.0)
            if "latency" in m:
                row += f"{val:{col_w}.2f}"
            else:
                row += f"{val:{col_w}.4f}"
        print(row)
    nq = next(iter(methods.values())).get("n_queries", 0)
    print(f"\nQueries used: {nq} | K={report['k']} | seed={report['seed']}")
    print(
        "Note: proxy labels use tag/genre Jaccard overlap "
        "(not user clicks). Content-based methods are expected to score well."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline recommender evaluation")
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-reviews", type=int, default=0)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--json-out", type=str, default="data/evaluation_report.json")
    args = parser.parse_args()

    report = evaluate(
        sample_size=args.sample,
        k=args.k,
        seed=args.seed,
        min_reviews=args.min_reviews,
        min_rating=args.min_rating,
    )
    print_table(report)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
