"""Central configuration for Game Recommender."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Artifact paths
PROCESSED_GAMES_CSV = DATA_DIR / "processed_games.csv"
LATENT_REPS_NPY = DATA_DIR / "latent_reps.npy"
FEATURE_INFO_PKL = DATA_DIR / "feature_info.pkl"
BEST_MODEL_PTH = DATA_DIR / "best_model.pth"
SCALER_PKL = DATA_DIR / "scaler.pkl"
MLB_GENRES_PKL = DATA_DIR / "mlb_genres.pkl"
MLB_TAGS_PKL = DATA_DIR / "mlb_tags.pkl"
MLB_CATEGORIES_PKL = DATA_DIR / "mlb_categories.pkl"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

# Model defaults (must match Colab_Train.ipynb / best_model.pth)
DEFAULT_INPUT_DIM = 1289
DEFAULT_LATENT_DIM = 128
GENRE_WEIGHT = 2.0
TAG_WEIGHT = 1.5
CATEGORY_WEIGHT = 1.0
DESC_WEIGHT = 1.0
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128

# Ranking defaults
DEFAULT_NUM_RECS = 9
DEFAULT_MIN_REVIEWS = 5000
DEFAULT_MIN_RATING = 0.65
SIM_WEIGHT = 0.7
RATING_WEIGHT = 0.3

# Flask
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "1") not in ("0", "false", "False")
