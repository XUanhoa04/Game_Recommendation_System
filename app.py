from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

DATA_DIR = 'data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(ImprovedAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, input_dim),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)

# ==================== LOAD DATA ====================
print("Loading data...")
df = pd.read_csv(f'{DATA_DIR}/processed_games.csv')
latent_reps = np.load(f'{DATA_DIR}/latent_reps.npy')

with open(f'{DATA_DIR}/feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

print(f"Loaded {len(df)} games")

# ==================== RECOMMENDATION ENGINE ====================
class RecommendationEngine:
    def __init__(self, df, latent_reps):
        self.df = df
        self.latent_reps = latent_reps
        self.game_index = {name.lower(): idx for idx, name in enumerate(df['Name'])}
    
    def recommend(self, game_name, num_recommendations=9, min_reviews=5000, min_rating=0.65):
 
        game_name_lower = game_name.lower()
        
        # Find game
        if game_name_lower not in self.game_index:
            return None
        
        idx = self.game_index[game_name_lower]
        
        # Calculate similarities
        similarities = cosine_similarity([self.latent_reps[idx]], self.latent_reps)[0]
        
        # Create candidates with scores
        candidates = []
        for i, sim in enumerate(similarities):
            if i == idx:  # Skip the game itself
                continue
            
            game = self.df.iloc[i]
            total_reviews = game['Total Reviews']
            positive_ratio = game['Positive'] / max(total_reviews, 1)
            
            # Quality score combining similarity and ratings
            quality_score = sim * 0.7 + positive_ratio * 0.3
            
            candidates.append({
                'index': i,
                'similarity': sim,
                'positive_ratio': positive_ratio,
                'total_reviews': total_reviews,
                'quality_score': quality_score
            })
        
        # Filter by quality
        filtered_candidates = [
            c for c in candidates 
            if c['total_reviews'] >= min_reviews and c['positive_ratio'] >= min_rating
        ]
        
        # Sort by quality score
        filtered_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Get top recommendations
        top_candidates = filtered_candidates[:num_recommendations]
        
        # Build response
        recommendations = []
        for candidate in top_candidates:
            game = self.df.iloc[candidate['index']]
            recommendations.append({
                'Name': game['Name'],
                'Header image': game['Header image'],
                'Short description': game['Short description'],
                'Genres': game['Genres'],
                'Movies': game['Movies'] if pd.notna(game['Movies']) else '',
                'Link Game': game['Link Game'],
                'Positive': int(game['Positive']),
                'Total Reviews': int(game['Total Reviews']),
                'Similarity': float(candidate['similarity']),
                'Rating': f"{candidate['positive_ratio']*100:.1f}%"
            })
        
        return recommendations
    
    def get_popular_games(self, num_games=20):
        """Get popular games for homepage"""
        # Calculate popularity score
        df_copy = self.df.copy()
        df_copy['popularity'] = (
            df_copy['Positive'] / df_copy['Total Reviews'].clip(lower=1)
        ) * np.log1p(df_copy['Total Reviews'])
        
        # Get top games
        top_games = df_copy.nlargest(num_games, 'popularity')
        
        return [{
            'Name': row['Name'],
            'Header image': row['Header image'],
            'Short description': row['Short description'],
            'Genres': row['Genres'],
            'Total Reviews': int(row['Total Reviews']),
            'Rating': f"{row['Positive'] / row['Total Reviews'] * 100:.1f}%"
        } for _, row in top_games.iterrows()]

# Initialize recommendation engine
rec_engine = RecommendationEngine(df, latent_reps)

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_names', methods=['GET'])
def game_names():
    """Return all game names for autocomplete"""
    try:
        return jsonify(df['Name'].tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_game_info', methods=['POST'])
def get_game_info():
    """Get information about a specific game"""
    data = request.get_json()
    game_name = data.get('game_name', '')
    
    game_row = df[df['Name'].str.lower() == game_name.lower()]
    
    if game_row.empty:
        return jsonify({}), 404
    
    game = game_row.iloc[0]
    return jsonify({
        'Name': game['Name'],
        'Header image': game['Header image'],
        'Link Game': game['Link Game'],
        'Short description': game['Short description'],
        'Genres': game['Genres'],
        'Total Reviews': int(game['Total Reviews']),
        'Rating': f"{game['Positive'] / game['Total Reviews'] * 100:.1f}%"
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get game recommendations"""
    data = request.get_json()
    game_name = data.get('game_name', '')
    num_recommendations = data.get('num_recommendations', 9)
    
    recommendations = rec_engine.recommend(game_name, num_recommendations)
    
    if recommendations is None:
        return jsonify([])
    
    return jsonify(recommendations)

@app.route('/random_game', methods=['GET'])
def random_game():
    """Get a random popular game"""
    # Sample from top 1000 popular games
    df_popular = df.nlargest(1000, 'Total Reviews')
    random_game = df_popular.sample(1).iloc[0]
    
    return jsonify({
        'Name': random_game['Name'],
        'Header image': random_game['Header image'],
        'Link Game': random_game['Link Game']
    })

@app.route('/popular_games', methods=['GET'])
def popular_games():
    """Get popular games for homepage"""
    num_games = request.args.get('num', 20, type=int)
    games = rec_engine.get_popular_games(num_games)
    return jsonify(games)

@app.route('/search', methods=['GET'])
def search():
    """Search games by name"""
    query = request.args.get('q', '').lower()
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify([])
    
    # Fuzzy search
    matches = df[df['Name'].str.lower().str.contains(query, na=False)]
    results = matches.head(limit)
    
    return jsonify([{
        'Name': row['Name'],
        'Header image': row['Header image'],
        'Genres': row['Genres']
    } for _, row in results.iterrows()])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)