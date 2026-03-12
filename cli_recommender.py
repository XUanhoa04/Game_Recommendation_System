import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys

# Đường dẫn data (thay đổi nếu cần)
DATA_DIR = 'data'

# Load data
try:
    df = pd.read_csv(f'{DATA_DIR}/processed_games.csv')
    latent_reps = np.load(f'{DATA_DIR}/latent_reps.npy')
    with open(f'{DATA_DIR}/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
except FileNotFoundError:
    print("Error: Data files not found in 'data/' directory.")
    sys.exit(1)

# Class RecommendationEngine (copy từ app.py)
class RecommendationEngine:
    def __init__(self, df, latent_reps):
        self.df = df
        self.latent_reps = latent_reps
        self.game_index = {name.lower(): idx for idx, name in enumerate(df['Name'])}
    
    def recommend(self, game_name, num_recommendations=9, min_reviews=5000, min_rating=0.65):
        game_name_lower = game_name.lower()
        
        if game_name_lower not in self.game_index:
            return None
        
        idx = self.game_index[game_name_lower]
        
        similarities = cosine_similarity([self.latent_reps[idx]], self.latent_reps)[0]
        
        candidates = []
        for i, sim in enumerate(similarities):
            if i == idx:  # Skip game gốc
                continue
            
            game = self.df.iloc[i]
            total_reviews = game['Total Reviews']
            positive_ratio = game['Positive'] / max(total_reviews, 1)
            
            # Chỉ số quality_score = similarity * 0.7 + positive_ratio * 0.3
            quality_score = sim * 0.7 + positive_ratio * 0.3
            
            candidates.append({
                'index': i,
                'similarity': sim,
                'positive_ratio': positive_ratio,
                'total_reviews': total_reviews,
                'quality_score': quality_score
            })
        
        # Filter theo min_reviews và min_rating
        filtered_candidates = [
            c for c in candidates 
            if c['total_reviews'] >= min_reviews and c['positive_ratio'] >= min_rating
        ]
        
        # Sort theo quality_score descending
        filtered_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        
        top_candidates = filtered_candidates[:num_recommendations]
        
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
                'Rating': f"{candidate['positive_ratio']*100:.1f}%",
                'Quality Score': float(candidate['quality_score'])
            })
        
        return recommendations

# Khởi tạo và chạy CLI
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cli_recommender.py \"Game Name\"")
        sys.exit(1)
    
    game_name = sys.argv[1]
    rec_engine = RecommendationEngine(df, latent_reps)
    recommendations = rec_engine.recommend(game_name)
    
    if recommendations is None:
        print(f"Game '{game_name}' not found.")
    else:
        print(f"Recommendations for '{game_name}':")
        for rec in recommendations:
            print(f"\nName: {rec['Name']}")
            print(f"Similarity: {rec['Similarity']:.4f} (độ tương đồng latent vector từ autoencoder)")
            print(f"Positive Ratio: {rec['positive_ratio']:.4f} (tỷ lệ review positive / total reviews)")
            print(f"Quality Score: {rec['Quality Score']:.4f} (tính bằng similarity * 0.7 + positive_ratio * 0.3)")
            print(f"Genres: {rec['Genres']}")
            print(f"Rating: {rec['Rating']}")
            print(f"Total Reviews: {rec['Total Reviews']}")
            print(f"Link: {rec['Link Game']}")