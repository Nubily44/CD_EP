import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds

# 1. Load data with downcasted types to save RAM immediately
# We only load the columns we actually need.
animelist = pd.read_csv('./data/animelist.csv', 
                        usecols=['user_id', 'anime_id', 'rating'],
                        dtype={'user_id': 'int32', 'anime_id': 'int32', 'rating': 'int8'})

animes = pd.read_csv('./data/anime.csv', usecols=['MAL_ID', 'Name'])

# Remove 0 ratings (unrated)
animelist = animelist[animelist['rating'] > 0]

# 2. Aggressive Filtering
# Keep users with > 150 ratings and animes with > 500 ratings
user_counts = animelist['user_id'].value_counts()
anime_counts = animelist['anime_id'].value_counts()

animelist = animelist[
    (animelist['user_id'].isin(user_counts[user_counts >= 150].index)) & 
    (animelist['anime_id'].isin(anime_counts[anime_counts >= 500].index))
]

# 3. Map IDs to continuous indices for the sparse matrix
# This prevents out-of-bounds errors when creating the matrix
user_cat = animelist['user_id'].astype('category')
anime_cat = animelist['anime_id'].astype('category')

animelist['user_idx'] = user_cat.cat.codes
animelist['anime_idx'] = anime_cat.cat.codes

# Keep dictionaries to map indices back to real IDs
idx_to_user = dict(enumerate(user_cat.cat.categories))
idx_to_anime = dict(enumerate(anime_cat.cat.categories))
anime_to_idx = {v: k for k, v in idx_to_anime.items()}

# Merge names for display purposes
anime_mapping = animes.set_index('MAL_ID')['Name'].to_dict()

# 4. Create the Sparse Matrix (Rows: Users, Cols: Animes)
# This uses fractions of the RAM compared to a Pandas pivot table
sparse_user_item = csr_matrix(
    (animelist['rating'], (animelist['user_idx'], animelist['anime_idx']))
)

print(f"Sparse Matrix created with shape: {sparse_user_item.shape}")

# Convert to float for SVD and normalize
sparse_float = sparse_user_item.astype('float64')

# Calculate user means
# We sum the ratings and divide by the number of non-zero ratings per user
user_sums = sparse_float.sum(axis=1).A1
user_counts_nonzero = sparse_float.getnnz(axis=1)
# Avoid division by zero
user_counts_nonzero[user_counts_nonzero == 0] = 1 
user_means = user_sums / user_counts_nonzero

# Subtract mean from non-zero elements
# We do this directly on the sparse matrix data to save memory
normalized_sparse = sparse_float.copy()
for i in range(normalized_sparse.shape[0]):
    start = normalized_sparse.indptr[i]
    end = normalized_sparse.indptr[i+1]
    normalized_sparse.data[start:end] -= user_means[i]

# Run SVD directly on the sparse matrix
# k is the number of latent factors
U, sigma, Vt = svds(normalized_sparse, k=20)
sigma_diag = np.diag(sigma)

def recommend_svd_optimized(user_id, U, sigma_diag, Vt, user_means, original_matrix, top_n=5):
    try:
        user_idx = list(idx_to_user.keys())[list(idx_to_user.values()).index(user_id)]
    except ValueError:
        return "User not found in filtered data."

    # Reconstruct predictions ONLY for this specific user
    # Matrix multiplication: U[user] * Sigma * Vt
    user_predictions = np.dot(np.dot(U[user_idx, :], sigma_diag), Vt) + user_means[user_idx]
    
    # Get items the user already rated
    user_rated_indices = original_matrix[user_idx].indices
    
    # Create list of unrated animes with predicted scores
    recs = []
    for anime_idx, pred_score in enumerate(user_predictions):
        if anime_idx not in user_rated_indices:
            real_anime_id = idx_to_anime[anime_idx]
            anime_name = anime_mapping.get(real_anime_id, f"Unknown ID {real_anime_id}")
            recs.append((anime_name, pred_score))
            
    return sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]


sample_user_id = idx_to_user[0] 
print(recommend_svd_optimized(sample_user_id, U, sigma_diag, Vt, user_means, sparse_user_item))