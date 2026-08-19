"""Offline recommendation metrics and proxy-label helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from utils.parsing import jaccard


def precision_at_k(relevant: Set[int], recommended: Sequence[int], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = sum(1 for i in top if i in relevant)
    return hits / k


def recall_at_k(relevant: Set[int], recommended: Sequence[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = sum(1 for i in top if i in relevant)
    return hits / len(relevant)


def average_precision(relevant: Set[int], recommended: Sequence[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = 0
    score = 0.0
    for i, idx in enumerate(top, start=1):
        if idx in relevant:
            hits += 1
            score += hits / i
    denom = min(len(relevant), k)
    return score / denom if denom > 0 else 0.0


def ndcg_at_k(relevant: Set[int], recommended: Sequence[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    dcg = 0.0
    for i, idx in enumerate(top, start=1):
        if idx in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def intra_list_diversity(tag_sets: Sequence[Set[str]]) -> float:
    """Mean pairwise (1 - Jaccard) over tag sets in a recommendation list."""
    n = len(tag_sets)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - jaccard(tag_sets[i], tag_sets[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def catalog_coverage(all_recommended: Iterable[int], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    unique_items = set(all_recommended)
    if not unique_items:
        return 0.0
    return min(1.0, len(unique_items) / catalog_size)


def build_proxy_relevant(
    query_idx: int,
    genre_sets: Sequence[Set[str]],
    tag_sets: Sequence[Set[str]],
    min_tag_jaccard: float = 0.25,
    min_genre_jaccard: float = 0.5,
    top_n_by_tag: int = 50,
) -> Set[int]:
    """Proxy ground-truth: games with high tag/genre overlap (excluding self)."""
    if query_idx < 0 or query_idx >= len(tag_sets) or query_idx >= len(genre_sets):
        return set()
    if top_n_by_tag <= 0:
        return set()

    q_tags = tag_sets[query_idx]
    q_genres = genre_sets[query_idx]
    if not q_tags and not q_genres:
        return set()

    scored: List[tuple] = []
    num_items = min(len(tag_sets), len(genre_sets))

    for i in range(num_items):
        if i == query_idx:
            continue
        tj = jaccard(q_tags, tag_sets[i])
        gj = jaccard(q_genres, genre_sets[i])
        if tj >= min_tag_jaccard or gj >= min_genre_jaccard:
            scored.append((0.7 * tj + 0.3 * gj, i))

    scored.sort(reverse=True)
    return {i for _, i in scored[:top_n_by_tag]}


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
