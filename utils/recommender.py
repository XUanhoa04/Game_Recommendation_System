"""Content-based recommendation engine with FAISS ANN, quality ranking, and explanations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MIN_RATING,
    DEFAULT_MIN_REVIEWS,
    DEFAULT_NUM_RECS,
    FAISS_INDEX_PATH,
    LATENT_REPS_NPY,
    PROCESSED_GAMES_CSV,
    RATING_WEIGHT,
    SIM_WEIGHT,
)
from utils.parsing import parse_list_field, top_overlap


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


@dataclass
class SearchFilters:
    min_reviews: int = DEFAULT_MIN_REVIEWS
    min_rating: float = DEFAULT_MIN_RATING
    genres: Optional[List[str]] = None  # any match
    max_price: Optional[float] = None
    multiplayer_only: bool = False


class RecommendationEngine:
    """Precomputed-latent content-based recommender."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        latent_reps: Optional[np.ndarray] = None,
        use_faiss: bool = True,
        data_csv: Union[str, Path] = PROCESSED_GAMES_CSV,
        latent_path: Union[str, Path] = LATENT_REPS_NPY,
    ):
        if df is None:
            df = pd.read_csv(data_csv)
        if latent_reps is None:
            latent_reps = np.load(latent_path)

        if len(df) != len(latent_reps):
            raise ValueError(
                f"Row count mismatch: df={len(df)} vs latent={len(latent_reps)}"
            )

        self.df = df.reset_index(drop=True)
        self.latent_raw = np.asarray(latent_reps, dtype=np.float32)
        self.latent = _l2_normalize(self.latent_raw)
        self.dim = self.latent.shape[1]

        # Indexes
        self.game_index: Dict[str, int] = {
            str(name).lower(): idx for idx, name in enumerate(self.df["Name"])
        }
        self.names = self.df["Name"].astype(str).tolist()

        # Pre-parse metadata for filters / explanations
        self.genre_lists: List[List[str]] = [
            parse_list_field(v) for v in self.df["Genres"]
        ]
        self.tag_lists: List[List[str]] = [parse_list_field(v) for v in self.df["Tags"]]
        self.category_lists: List[List[str]] = [
            parse_list_field(v) for v in self.df["Categories"]
        ]
        self.genre_sets: List[Set[str]] = [set(g) for g in self.genre_lists]
        self.tag_sets: List[Set[str]] = [set(t) for t in self.tag_lists]

        self.total_reviews = self.df["Total Reviews"].fillna(0).astype(np.float64).to_numpy()
        self.positive = self.df["Positive"].fillna(0).astype(np.float64).to_numpy()
        self.positive_ratio = self.positive / np.maximum(self.total_reviews, 1.0)

        if "Price" in self.df.columns:
            self.prices = pd.to_numeric(self.df["Price"], errors="coerce").fillna(0).to_numpy()
        else:
            self.prices = np.zeros(len(self.df), dtype=np.float64)

        multiplayer_tokens = {
            "multi-player",
            "multiplayer",
            "online multi-player",
            "online multiplayer",
            "co-op",
            "online co-op",
            "mmo",
            "massively multiplayer",
        }
        self.is_multiplayer = np.array(
            [
                any(c.lower() in multiplayer_tokens for c in cats)
                or any(t.lower() in multiplayer_tokens for t in tags)
                or any(g.lower() in multiplayer_tokens for g in genres)
                for genres, tags, cats in zip(
                    self.genre_lists, self.tag_lists, self.category_lists
                )
            ],
            dtype=bool,
        )

        self.use_faiss = use_faiss
        self._faiss_index = None
        if use_faiss:
            self._init_faiss()

        # Popularity score for homepage / random
        self.popularity = self.positive_ratio * np.log1p(self.total_reviews)

    def _init_faiss(self) -> None:
        try:
            import faiss

            index = faiss.IndexFlatIP(self.dim)
            index.add(self.latent.astype(np.float32))
            self._faiss_index = index
        except Exception as exc:  # noqa: BLE001
            print(f"FAISS unavailable ({exc}); using numpy similarity")
            self._faiss_index = None
            self.use_faiss = False

    def save_faiss_index(self, path: Union[str, Path] = FAISS_INDEX_PATH) -> None:
        if self._faiss_index is None:
            raise RuntimeError("FAISS index not initialized")
        import faiss

        faiss.write_index(self._faiss_index, str(path))

    def resolve_name(self, game_name: str) -> Optional[int]:
        return self.game_index.get(game_name.lower().strip())

    def _query_vector(
        self,
        game_names: Optional[Sequence[str]] = None,
        indices: Optional[Sequence[int]] = None,
        latent_vector: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Build a single L2-normalized query vector from seeds or raw latent."""
        seed_indices: List[int] = []

        if latent_vector is not None:
            vec = np.asarray(latent_vector, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self.dim:
                raise ValueError(f"Expected latent dim {self.dim}, got {vec.shape[0]}")
            if not np.all(np.isfinite(vec)):
                raise ValueError("Latent vector contains NaN or Inf values")
            return _l2_normalize(vec.reshape(1, -1))[0], seed_indices

        if indices is not None:
            seed_indices = [int(i) for i in indices if 0 <= int(i) < len(self.df)]
        elif game_names is not None:
            for name in game_names:
                idx = self.resolve_name(name)
                if idx is None:
                    raise KeyError(f"Game not found: {name}")
                seed_indices.append(idx)
        else:
            raise ValueError("Provide game_names, indices, or latent_vector")

        if not seed_indices:
            raise ValueError("No valid seed games provided")

        # Average raw latents then re-normalize (standard multi-seed fusion)
        mean_vec = self.latent_raw[seed_indices].mean(axis=0, keepdims=True)
        return _l2_normalize(mean_vec)[0], seed_indices

    def _similarities(self, query: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, similarities) of top_n nearest neighbors (unsorted among candidates)."""
        query = query.astype(np.float32).reshape(1, -1)
        top_n = min(top_n, len(self.latent))

        if self._faiss_index is not None:
            sims, idxs = self._faiss_index.search(query, top_n)
            return idxs[0], sims[0]

        sims = (self.latent @ query.T).ravel()
        if top_n >= len(sims):
            order = np.argsort(-sims)
        else:
            # argpartition for speed, then sort the slice
            part = np.argpartition(-sims, top_n)[:top_n]
            order = part[np.argsort(-sims[part])]
        return order, sims[order]

    def _passes_filters(self, idx: int, filters: SearchFilters) -> bool:
        if self.total_reviews[idx] < filters.min_reviews:
            return False
        if self.positive_ratio[idx] < filters.min_rating:
            return False
        if filters.max_price is not None and self.prices[idx] > filters.max_price:
            return False
        if filters.multiplayer_only and not self.is_multiplayer[idx]:
            return False
        if filters.genres:
            wanted = {str(g).strip().lower() for g in filters.genres if str(g).strip()}
            if wanted:
                game_genres = {g.lower() for g in self.genre_lists[idx]}
                if not wanted.intersection(game_genres):
                    return False
        return True

    def _explain(
        self,
        seed_indices: Sequence[int],
        candidate_idx: int,
        similarity: float,
    ) -> Dict[str, Any]:
        if not seed_indices:
            return {
                "shared_genres": [],
                "shared_tags": [],
                "reasons": [f"Similarity: {similarity:.3f}"],
                "summary": f"Similarity: {similarity:.3f}",
            }

        # Aggregate seed tags/genres for multi-seed
        seed_genres: Set[str] = set()
        seed_tags: Set[str] = set()
        for s in seed_indices:
            if 0 <= s < len(self.genre_sets):
                seed_genres |= self.genre_sets[s]
                seed_tags |= self.tag_sets[s]

        shared_genres = top_overlap(seed_genres, self.genre_sets[candidate_idx], limit=6)
        shared_tags = top_overlap(seed_tags, self.tag_sets[candidate_idx], limit=8)

        reasons: List[str] = []
        if shared_genres:
            reasons.append("Shared genres: " + ", ".join(shared_genres))
        if shared_tags:
            reasons.append("Shared tags: " + ", ".join(shared_tags[:5]))
        reasons.append(f"Latent similarity: {similarity:.3f}")
        reasons.append(
            f"Community rating: {self.positive_ratio[candidate_idx] * 100:.1f}%"
        )

        return {
            "shared_genres": shared_genres,
            "shared_tags": shared_tags,
            "reasons": reasons,
            "summary": " · ".join(reasons[:3]),
        }

    def _row_to_dict(
        self,
        idx: int,
        similarity: float,
        quality_score: float,
        seed_indices: Sequence[int],
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        game = self.df.iloc[idx]
        movies = game["Movies"] if "Movies" in self.df.columns else ""
        if pd.isna(movies):
            movies = ""

        item: Dict[str, Any] = {
            "Name": game["Name"],
            "Header image": game["Header image"],
            "Short description": game["Short description"] if pd.notna(game["Short description"]) else "",
            "Genres": game["Genres"],
            "Tags": game["Tags"] if "Tags" in self.df.columns else "",
            "Movies": movies,
            "Link Game": game["Link Game"],
            "Positive": int(self.positive[idx]),
            "Total Reviews": int(self.total_reviews[idx]),
            "Price": float(self.prices[idx]),
            "Similarity": float(similarity),
            "Rating": f"{self.positive_ratio[idx] * 100:.1f}%",
            "Quality Score": float(quality_score),
            "index": int(idx),
        }
        if include_explanation and seed_indices:
            item["explanation"] = self._explain(seed_indices, idx, similarity)
        return item

    def recommend(
        self,
        game_name: Optional[str] = None,
        game_names: Optional[Sequence[str]] = None,
        indices: Optional[Sequence[int]] = None,
        latent_vector: Optional[np.ndarray] = None,
        num_recommendations: int = DEFAULT_NUM_RECS,
        min_reviews: int = DEFAULT_MIN_REVIEWS,
        min_rating: float = DEFAULT_MIN_RATING,
        genres: Optional[List[str]] = None,
        max_price: Optional[float] = None,
        multiplayer_only: bool = False,
        sim_weight: float = SIM_WEIGHT,
        rating_weight: float = RATING_WEIGHT,
        include_explanation: bool = True,
        candidate_pool: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Recommend similar games with quality ranking and optional filters.

        Accepts a single game_name, multiple game_names (multi-seed), raw indices,
        or an external latent_vector (online encode path).
        """
        if game_name is not None and game_names is None:
            game_names = [game_name]

        try:
            query, seed_indices = self._query_vector(
                game_names=game_names,
                indices=indices,
                latent_vector=latent_vector,
            )
        except KeyError:
            return None

        filters = SearchFilters(
            min_reviews=min_reviews,
            min_rating=min_rating,
            genres=genres,
            max_price=max_price,
            multiplayer_only=multiplayer_only,
        )

        # Exclude seed indices and any rows that share a seed title
        # (catalog has ~158 duplicate names: remasters / re-listings)
        exclude = set(seed_indices)
        seed_name_keys = {self.names[i].lower() for i in seed_indices}
        if seed_name_keys:
            for i, name in enumerate(self.names):
                if name.lower() in seed_name_keys:
                    exclude.add(i)

        # Oversample ANN pool so filters still yield enough results
        pool = candidate_pool or max(num_recommendations * 50, 500)
        pool = min(pool, len(self.latent))
        nn_idx, nn_sim = self._similarities(query, pool)

        sim_map = {int(i): float(s) for i, s in zip(nn_idx, nn_sim) if i >= 0}

        candidates: List[Tuple[int, float, float]] = []

        for i, sim in sim_map.items():
            if i in exclude or i < 0:
                continue
            if not self._passes_filters(i, filters):
                continue
            quality = sim_weight * sim + rating_weight * self.positive_ratio[i]
            candidates.append((i, sim, quality))

        # If ANN pool is too small after filters, fall back to full cosine scan
        if len(candidates) < num_recommendations:
            all_sims = (self.latent @ query.reshape(-1, 1)).ravel()
            seen = {c[0] for c in candidates} | exclude
            for i, sim in enumerate(all_sims):
                if i in seen:
                    continue
                if not self._passes_filters(i, filters):
                    continue
                quality = sim_weight * float(sim) + rating_weight * self.positive_ratio[i]
                candidates.append((i, float(sim), quality))

        candidates.sort(key=lambda x: x[2], reverse=True)

        # Deduplicate by display name so remasters don't flood the list
        top: List[Tuple[int, float, float]] = []
        used_names: Set[str] = set()
        for i, sim, q in candidates:
            key = self.names[i].lower()
            if key in used_names:
                continue
            used_names.add(key)
            top.append((i, sim, q))
            if len(top) >= num_recommendations:
                break

        return [
            self._row_to_dict(i, sim, q, seed_indices, include_explanation)
            for i, sim, q in top
        ]

    def recommend_indices_only(
        self,
        query_idx: int,
        k: int = 10,
        min_reviews: int = 0,
        min_rating: float = 0.0,
        candidate_pool: int = 200,
    ) -> List[int]:
        """Lightweight path for evaluation (returns indices only)."""
        results = self.recommend(
            indices=[query_idx],
            num_recommendations=k,
            min_reviews=min_reviews,
            min_rating=min_rating,
            include_explanation=False,
            candidate_pool=candidate_pool,
        )
        if not results:
            return []
        return [r["index"] for r in results]

    def get_game_info(self, game_name: str) -> Optional[Dict[str, Any]]:
        idx = self.resolve_name(game_name)
        if idx is None:
            return None
        game = self.df.iloc[idx]
        return {
            "Name": game["Name"],
            "Header image": game["Header image"],
            "Link Game": game["Link Game"],
            "Short description": game["Short description"] if pd.notna(game["Short description"]) else "",
            "Genres": game["Genres"],
            "Tags": game["Tags"] if "Tags" in self.df.columns else "",
            "Total Reviews": int(self.total_reviews[idx]),
            "Rating": f"{self.positive_ratio[idx] * 100:.1f}%",
            "Price": float(self.prices[idx]),
        }

    def get_popular_games(self, num_games: int = 20) -> List[Dict[str, Any]]:
        order = np.argsort(-self.popularity)[:num_games]
        results = []
        for idx in order:
            game = self.df.iloc[int(idx)]
            results.append(
                {
                    "Name": game["Name"],
                    "Header image": game["Header image"],
                    "Short description": game["Short description"]
                    if pd.notna(game["Short description"])
                    else "",
                    "Genres": game["Genres"],
                    "Total Reviews": int(self.total_reviews[idx]),
                    "Rating": f"{self.positive_ratio[idx] * 100:.1f}%",
                }
            )
        return results

    def random_game(self, top_n: int = 1000) -> Dict[str, Any]:
        if len(self.df) == 0:
            return {"Name": "", "Header image": "", "Link Game": ""}
        pool_size = max(1, min(top_n, len(self.df)))
        order = np.argsort(-self.total_reviews)[:pool_size]
        if len(order) == 0:
            return {"Name": "", "Header image": "", "Link Game": ""}
        idx = int(np.random.choice(order))
        game = self.df.iloc[idx]
        return {
            "Name": game["Name"],
            "Header image": game["Header image"],
            "Link Game": game["Link Game"],
        }

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = query.lower().strip()
        if not q or limit <= 0:
            return []

        matches: List[Tuple[int, float, str]] = []
        seen_names: Set[str] = set()

        for idx, name in enumerate(self.names):
            lower = name.lower()
            if lower in seen_names:
                continue

            if lower == q:
                tier = 0  # Exact match
            elif lower.startswith(q):
                tier = 1  # Prefix match
            elif f" {q}" in lower or f":{q}" in lower or f"-{q}" in lower:
                tier = 2  # Word boundary match
            elif q in lower:
                tier = 3  # Substring match
            else:
                continue

            seen_names.add(lower)
            pop = float(self.popularity[idx]) if idx < len(self.popularity) else 0.0
            matches.append((tier, -pop, name))

        matches.sort(key=lambda x: (x[0], x[1], x[2].lower()))
        results = []
        for _, _, name in matches[:limit]:
            idx = self.game_index[name.lower()]
            game = self.df.iloc[idx]
            results.append(
                {
                    "Name": game["Name"],
                    "Header image": game["Header image"],
                    "Genres": game["Genres"],
                }
            )
        return results

    def available_genres(self, top_n: int = 40) -> List[str]:
        from collections import Counter

        counts: Counter = Counter()
        for genres in self.genre_lists:
            counts.update(genres)
        return [g for g, _ in counts.most_common(top_n)]

    def recommend_from_features(
        self,
        latent_vector: np.ndarray,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Online encode path: recommend from an external latent vector."""
        results = self.recommend(latent_vector=latent_vector, **kwargs)
        return results or []
