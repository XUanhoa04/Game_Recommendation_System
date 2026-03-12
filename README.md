# Game Recommender Pro

A content-based game recommendation system built with a deep autoencoder model, served via a Flask web application.

## Overview

The system encodes Steam game features into a latent space using an autoencoder neural network, then computes cosine similarity between latent vectors to find similar games. A quality score combining similarity and user review ratio is used to rank recommendations.
## Dataset
https://drive.google.com/file/d/1XyYrLodKYIgrWhpDpU2lIAwq9KPv0gKz/view?usp=sharing

## Tech Stack

- **Backend**: Python, Flask
- **ML**: PyTorch (Autoencoder), scikit-learn (cosine similarity)
- **Data**: pandas, numpy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5, jQuery

## Project Structure

```
├── app.py                  # Flask application and recommendation engine
├── recommender.py          # Standalone version of app.py
├── cli_recommender.py      # CLI interface for testing recommendations
├── Colab_Train.ipynb       # Model training notebook (Google Colab)
├── data/
│   ├── processed_games.csv # Preprocessed game dataset
│   ├── latent_reps.npy     # Encoded latent representations
│   └── feature_info.pkl    # Feature metadata
├── templates/
│   └── index.html          # Main UI template
└── static/
    ├── css/styles.css
    └── js/scripts.js
```

## Model Architecture

The autoencoder compresses game feature vectors into a 128-dimensional latent space:

- **Encoder**: Linear(input → 512 → 256 → 128) with BatchNorm, ReLU, Dropout
- **Decoder**: Linear(128 → 256 → 512 → input) with BatchNorm, ReLU, Dropout

Recommendation quality score: `score = similarity * 0.7 + positive_ratio * 0.3`

## Setup

### Prerequisites

```bash
pip install flask pandas numpy torch scikit-learn
```

### Running the App

1. Ensure the `data/` directory contains `processed_games.csv`, `latent_reps.npy`, and `feature_info.pkl`.
2. Start the server:

```bash
python app.py
```

3. Open `http://localhost:5000` in your browser.

### CLI Usage

```bash
python cli_recommender.py "Counter-Strike 2"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/game_names` | All game names for autocomplete |
| POST | `/get_game_info` | Game details by name |
| POST | `/recommend` | Get recommendations for a game |
| GET | `/popular_games` | Top games by popularity score |
| GET | `/random_game` | Random game from top 1000 |
| GET | `/search` | Search games by name (q parameter) |

## Training

Open `Colab_Train.ipynb` in Google Colab to retrain the autoencoder on updated data. The notebook handles data preprocessing, model training, and export of `latent_reps.npy` and `feature_info.pkl`.

## Filters

Recommendations are filtered by default to games with at least 5,000 total reviews and a minimum 65% positive review ratio to ensure quality results.

## Screenshots
