import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    print("--- Loading and Preprocessing Data ---")
    # Load dataset with memory-optimized data types
    animelist = pd.read_csv('./data/animelist.csv', 
                            usecols=['user_id', 'anime_id', 'rating'],
                            dtype={'user_id': 'int32', 'anime_id': 'int32', 'rating': 'int8'})
    animes = pd.read_csv('./data/animes.csv', usecols=['MAL_ID', 'Name'])

    # Remove unrated entries (0 scores)
    animelist = animelist[animelist['rating'] > 0]

    # Filter for active users and popular items to ensure split stability
    user_counts = animelist['user_id'].value_counts()
    anime_counts = animelist['anime_id'].value_counts()

    animelist = animelist[
        (animelist['user_id'].isin(user_counts[user_counts >= 150].index)) & 
        (animelist['anime_id'].isin(anime_counts[anime_counts >= 500].index))
    ]

    # Map IDs to continuous index values
    animelist['user_idx'] = animelist['user_id'].astype('category').cat.codes
    animelist['anime_idx'] = animelist['anime_id'].astype('category').cat.codes

    n_users = animelist['user_idx'].nunique()
    n_animes = animelist['anime_idx'].nunique()
    matrix_shape = (n_users, n_animes)

    print(f"Dataset unique counts -> Users: {n_users}, Animes: {n_animes}")

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

    print(f"Split completed -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Construct Sparse Matrices
    train_matrix = csr_matrix((train_df['rating'], (train_df['user_idx'], train_df['anime_idx'])), shape=matrix_shape)
    val_matrix = csr_matrix((val_df['rating'], (val_df['user_idx'], val_df['anime_idx'])), shape=matrix_shape)
    test_matrix = csr_matrix((test_df['rating'], (test_df['user_idx'], test_df['anime_idx'])), shape=matrix_shape)

    print("\n--- Training SVD Model ---")
    train_float = train_matrix.astype('float64')

    # Calculate user baseline averages from training data exclusively
    user_sums = train_float.sum(axis=1).A1
    user_counts_nonzero = train_float.getnnz(axis=1)
    user_counts_nonzero[user_counts_nonzero == 0] = 1 
    user_means = user_sums / user_counts_nonzero

    # Center training data around user averages
    normalized_train = train_float.copy()
    for i in range(normalized_train.shape[0]):
        start = normalized_train.indptr[i]
        end = normalized_train.indptr[i+1]
        normalized_train.data[start:end] -= user_means[i]

    # Compute SVD components (k = latent factors)
    U, sigma, Vt = svds(normalized_train, k=20)
    sigma_diag = np.diag(sigma)

    # Reconstruct predictions matrix and add back user baselines
    all_predictions = np.dot(np.dot(U, sigma_diag), Vt) + user_means.reshape(-1, 1)
    all_predictions = np.clip(all_predictions, 1, 10)

    # Evaluation Routine
    def evaluate(pred_matrix, true_sparse, label):
        rows, cols = true_sparse.nonzero()
        true_ratings = true_sparse.data
        predicted_ratings = pred_matrix[rows, cols]
        
        rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
        mae = mean_absolute_error(true_ratings, predicted_ratings)
        
        print(f"\n[{label} Metrics]")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")

    evaluate(all_predictions, val_matrix, "Validation Set")
    evaluate(all_predictions, test_matrix, "Test Set")

if __name__ == "__main__":
    main()