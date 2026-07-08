import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 1. Load data with downcasted types to save RAM immediately
# We only load the columns we actually need.
print('Lê')
animelist = pd.read_csv('./data/rating_complete.csv', 
                        usecols=['user_id', 'anime_id', 'rating'],
                        dtype={'user_id': 'int32', 'anime_id': 'int32', 'rating': 'int8'})

animes = pd.read_csv('./data/anime.csv', usecols=['MAL_ID', 'Name'])

print('Filtra 1')
# Remove 0 ratings (unrated)
animelist = animelist[animelist['rating'] > 0]

# 2. Aggressive Filtering
# Keep users with > 150 ratings and animes with > 500 ratings
print('Conta')
user_counts = animelist['user_id'].value_counts()
anime_counts = animelist['anime_id'].value_counts()

print('Filtra 2')
animelist = animelist[
    (animelist['user_id'].isin(user_counts[user_counts >= 150].index)) & 
    (animelist['anime_id'].isin(anime_counts[anime_counts >= 500].index))
]

print('Índice')
# 3. Map IDs to continuous indices for the sparse matrix
# This prevents out-of-bounds errors when creating the matrix
user_cat = animelist['user_id'].astype('category')
anime_cat = animelist['anime_id'].astype('category')

animelist['user_idx'] = user_cat.cat.codes
animelist['anime_idx'] = anime_cat.cat.codes

print('Dicionário')
# Keep dictionaries to map indices back to real IDs
idx_to_user = dict(enumerate(user_cat.cat.categories))
idx_to_anime = dict(enumerate(anime_cat.cat.categories))
anime_to_idx = {v: k for k, v in idx_to_anime.items()}

print('Merge')
# Merge names for display purposes
anime_mapping = animes.set_index('MAL_ID')['Name'].to_dict()

# 4. Create the Sparse Matrix (Rows: Users, Cols: Animes)
# This uses fractions of the RAM compared to a Pandas pivot table
sparse_user_item = csr_matrix(
    (animelist['rating'], (animelist['user_idx'], animelist['anime_idx']))
)

print(f"Sparse Matrix created with shape: {sparse_user_item.shape}")

# Fit the NearestNeighbors model on the sparse matrix
# algorithm='brute' works best for sparse cosine similarity
print('Nearest Neighbors')
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(sparse_user_item)

def recommend_collab_optimized(user_id, matrix, knn_model, top_n=5):
    # Find the internal index for the user
    try:
        user_idx = list(idx_to_user.keys())[list(idx_to_user.values()).index(user_id)]
    except ValueError:
        return "User not found in filtered data."

    # Get distances and indices of nearest neighbors
    distances, indices = knn_model.kneighbors(matrix[user_idx], n_neighbors=20)
    
    # Flatten arrays (skip the first one, which is the user themselves)
    neighbor_indices = indices.flatten()[1:]
    neighbor_distances = distances.flatten()[1:]
    
    # Convert distances to similarities (1 - distance)
    similarities = 1 - neighbor_distances
    
    # Get the items the target user has already rated
    user_rated_indices = matrix[user_idx].indices
    
    recommendations = {}
    
    # Iterate through neighbors to find items
    for i, neighbor_idx in enumerate(neighbor_indices):
        neighbor_ratings = matrix[neighbor_idx]
        sim_score = similarities[i]
        
        for anime_idx, rating in zip(neighbor_ratings.indices, neighbor_ratings.data):
            if anime_idx not in user_rated_indices: # If target user hasn't seen it
                if anime_idx not in recommendations:
                    recommendations[anime_idx] = {'score': 0, 'weight': 0}
                recommendations[anime_idx]['score'] += rating * sim_score
                recommendations[anime_idx]['weight'] += sim_score
                
    # Calculate weighted average and sort
    final_recs = []
    for anime_idx, data in recommendations.items():
        if data['weight'] > 0:
            final_score = data['score'] / data['weight']
            real_anime_id = idx_to_anime[anime_idx]
            anime_name = anime_mapping.get(real_anime_id, f"Unknown ID {real_anime_id}")
            final_recs.append((anime_name, final_score))
            
    return sorted(final_recs, key=lambda x: x[1], reverse=True)[:top_n]

# Test it using a valid user ID from the filtered dataset
sample_user_id = idx_to_user[0] 
print(recommend_collab_optimized(sample_user_id, sparse_user_item, model_knn))