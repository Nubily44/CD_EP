import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    print("--- Loading and Preprocessing Data ---")
    animelist = pd.read_csv('./data/animelist.csv', 
                            usecols=['user_id', 'anime_id', 'rating'],
                            dtype={'user_id': 'int32', 'anime_id': 'int32', 'rating': 'int8'})

    animelist = animelist[animelist['rating'] > 0]

    user_counts = animelist['user_id'].value_counts()
    anime_counts = animelist['anime_id'].value_counts()

    animelist = animelist[
        (animelist['user_id'].isin(user_counts[user_counts >= 150].index)) & 
        (animelist['anime_id'].isin(anime_counts[anime_counts >= 500].index))
    ]

    animelist['user_idx'] = animelist['user_id'].astype('category').cat.codes
    animelist['anime_idx'] = animelist['anime_id'].astype('category').cat.codes

    n_users = animelist['user_idx'].nunique()
    n_animes = animelist['anime_idx'].nunique()
    matrix_shape = (n_users, n_animes)

    # Stratified Train (80%) / Val (10%) / Test (10%) Split
    train_df, temp_df = train_test_split(
        animelist, 
        test_size=0.20, 
        stratify=animelist['user_idx'], 
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['user_idx'], 
        random_state=42
    )

    # Build the training matrix (the foundation for similarity)
    train_matrix = csr_matrix((train_df['rating'], (train_df['user_idx'], train_df['anime_idx'])), shape=matrix_shape)

    print("\n--- Fitting Neighborhood Model ---")
    # Fit KNN using cosine distance on the training matrix
    k_neighbors = 20
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k_neighbors + 1)
    knn.fit(train_matrix)

    # Calculate training user means as a fallback baseline
    train_float = train_matrix.astype('float64')
    user_sums = train_float.sum(axis=1).A1
    user_counts_nonzero = train_float.getnnz(axis=1)
    user_counts_nonzero[user_counts_nonzero == 0] = 1
    user_means = user_sums / user_counts_nonzero

    def evaluate_knn(eval_df, label):
        print(f"\nEvaluating User-KNN on {label}...")
        
        # Find neighbors for all users in one batch call to keep it fast
        distances, indices = knn.kneighbors(train_matrix)
        
        true_ratings = []
        pred_ratings = []
        
        # Group evaluations by user to avoid redundant lookups
        grouped = eval_df.groupby('user_idx')
        
        for user_idx, group in grouped:
            # Extract neighbor metrics (excluding the user's self-match)
            user_nn_indices = indices[user_idx][1:]
            user_nn_distances = distances[user_idx][1:]
            user_nn_similarities = 1.0 - user_nn_distances
            
            # Fetch neighbor interactions from the training matrix
            # Doing this prevents loading the full matrix into dense memory
            for _, row in group.iterrows():
                anime_idx = row['anime_idx']
                actual_rating = row['rating']
                
                weighted_sum = 0.0
                similarity_sum = 0.0
                
                for idx, sim in zip(user_nn_indices, user_nn_similarities):
                    neighbor_rating = train_matrix[idx, anime_idx]
                    if neighbor_rating > 0:
                        weighted_sum += neighbor_rating * sim
                        similarity_sum += sim
                
                # If neighbors have rated the item, compute weighted average. 
                # Otherwise, fall back to the user's personal mean rating.
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                else:
                    predicted_rating = user_means[user_idx]
                    
                true_ratings.append(actual_rating)
                pred_ratings.append(np.clip(predicted_rating, 1, 10))
                
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae = mean_absolute_error(true_ratings, pred_ratings)
        
        print(f"[{label} Metrics]")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")

    evaluate_knn(val_df, "Validation Set")
    evaluate_knn(test_df, "Test Set")

if __name__ == "__main__":
    main()