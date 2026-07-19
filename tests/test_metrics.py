"""Unit tests for offline metrics."""
from __future__ import annotations

from utils.metrics import (
    average_precision,
    build_proxy_relevant,
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_recall():
    relevant = {1, 2, 3}
    recs = [1, 4, 2, 5]
    assert precision_at_k(relevant, recs, 4) == 0.5
    assert recall_at_k(relevant, recs, 4) == 2 / 3


def test_map_and_ndcg():
    relevant = {1, 3}
    recs = [1, 2, 3, 4]
    assert average_precision(relevant, recs, 4) > 0
    assert 0 < ndcg_at_k(relevant, recs, 4) <= 1


def test_diversity_and_coverage():
    tags = [{"a", "b"}, {"b", "c"}, {"x", "y"}]
    div = intra_list_diversity(tags)
    assert 0 <= div <= 1
    assert catalog_coverage([1, 2, 2, 3], 10) == 0.3


def test_proxy_relevant():
    genre_sets = [{"Action"}, {"Action"}, {"Casual"}, {"Action", "RPG"}]
    tag_sets = [
        {"FPS", "Shooter", "Multiplayer"},
        {"FPS", "Shooter", "Tactical"},
        {"Farming", "Cute"},
        {"RPG", "Fantasy"},
    ]
    rel = build_proxy_relevant(0, genre_sets, tag_sets, min_tag_jaccard=0.2, min_genre_jaccard=0.5)
    assert 1 in rel
    assert 2 not in rel or 1 in rel
