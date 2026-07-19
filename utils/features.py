"""Feature construction and optional online encoding for new games."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from config import (
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    CATEGORY_WEIGHT,
    DESC_WEIGHT,
    FEATURE_INFO_PKL,
    GENRE_WEIGHT,
    MLB_CATEGORIES_PKL,
    MLB_GENRES_PKL,
    MLB_TAGS_PKL,
    SCALER_PKL,
    TAG_WEIGHT,
)
from utils.model import ImprovedAutoencoder, load_autoencoder
from utils.parsing import parse_list_field


class FeatureBuilder:
    """Build scaled feature vectors matching the training pipeline.

    BERT is loaded lazily only when description embeddings are needed.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
        load_bert: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(FEATURE_INFO_PKL, "rb") as f:
            self.feature_info: Dict[str, Any] = pickle.load(f)

        with open(MLB_GENRES_PKL, "rb") as f:
            self.mlb_genres = pickle.load(f)
        with open(MLB_TAGS_PKL, "rb") as f:
            self.mlb_tags = pickle.load(f)
        with open(MLB_CATEGORIES_PKL, "rb") as f:
            self.mlb_categories = pickle.load(f)
        with open(SCALER_PKL, "rb") as f:
            self.scaler = pickle.load(f)

        self.genre_weight = float(self.feature_info.get("genre_weight", GENRE_WEIGHT))
        self.tag_weight = float(self.feature_info.get("tag_weight", TAG_WEIGHT))
        self.category_weight = float(self.feature_info.get("category_weight", CATEGORY_WEIGHT))
        self.desc_weight = float(self.feature_info.get("desc_weight", DESC_WEIGHT))
        self.input_dim = int(self.feature_info.get("input_dim", 1289))
        self.latent_dim = int(self.feature_info.get("latent_dim", 128))

        self._tokenizer = None
        self._bert = None
        self._encoder: Optional[ImprovedAutoencoder] = None

        if load_bert:
            self._ensure_bert()

    def _ensure_bert(self) -> None:
        if self._bert is not None:
            return
        from transformers import BertModel, BertTokenizer

        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self._bert = BertModel.from_pretrained(BERT_MODEL_NAME).to(self.device)
        self._bert.eval()

    def _ensure_encoder(self) -> ImprovedAutoencoder:
        if self._encoder is None:
            self._encoder = load_autoencoder(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                device=self.device,
            )
        return self._encoder

    def embed_texts(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """BERT [CLS] embeddings for a list of texts."""
        self._ensure_bert()
        assert self._tokenizer is not None and self._bert is not None

        embeddings: List[np.ndarray] = []
        clean = [t if t and str(t).strip() else " " for t in texts]

        with torch.no_grad():
            for i in range(0, len(clean), batch_size):
                batch = clean[i : i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=BERT_MAX_LENGTH,
                ).to(self.device)
                outputs = self._bert(**inputs)
                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls)

        return np.vstack(embeddings).astype(np.float32)

    def build_features(
        self,
        genres: Sequence[Union[str, Sequence[str]]],
        tags: Sequence[Union[str, Sequence[str]]],
        categories: Sequence[Union[str, Sequence[str]]],
        descriptions: Sequence[str],
    ) -> np.ndarray:
        """Build scaled feature matrix for one or more games."""
        genre_lists = [parse_list_field(g) for g in genres]
        tag_lists = [parse_list_field(t) for t in tags]
        cat_lists = [parse_list_field(c) for c in categories]

        genres_matrix = self.mlb_genres.transform(genre_lists).astype(np.float32)
        tags_matrix = self.mlb_tags.transform(tag_lists).astype(np.float32)
        cats_matrix = self.mlb_categories.transform(cat_lists).astype(np.float32)
        desc_emb = self.embed_texts(list(descriptions))

        features = np.hstack(
            [
                genres_matrix * self.genre_weight,
                tags_matrix * self.tag_weight,
                cats_matrix * self.category_weight,
                desc_emb * self.desc_weight,
            ]
        ).astype(np.float32)

        return self.scaler.transform(features).astype(np.float32)

    def encode_to_latent(self, features: np.ndarray) -> np.ndarray:
        """Encode scaled features through the trained autoencoder."""
        encoder = self._ensure_encoder()
        tensor = torch.FloatTensor(features).to(self.device)
        # BatchNorm needs batch>1 or eval mode (already eval)
        if tensor.shape[0] == 1:
            # eval mode handles batch of 1 for BN
            pass
        with torch.no_grad():
            latent = encoder.encode(tensor).cpu().numpy()
        return latent.astype(np.float32)

    def encode_game(
        self,
        genres: Union[str, Sequence[str]],
        tags: Union[str, Sequence[str]],
        categories: Union[str, Sequence[str]],
        description: str,
        short_description: str = "",
    ) -> np.ndarray:
        """End-to-end encode a single new game into latent space (shape 1 x D)."""
        combined = f"{description} {short_description}".strip() or " "
        features = self.build_features(
            genres=[genres],
            tags=[tags],
            categories=[categories],
            descriptions=[combined],
        )
        return self.encode_to_latent(features)
