import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
import gc
import os
import time

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0.0
        val_loss = logs.get('val_loss') if logs else None
        loss = logs.get('loss') if logs else None
        print(
            f"  Epoch {epoch + 1} finished in {epoch_time:.3f}s "
            f"(avg step {avg_batch_time:.3f}s, {len(self.batch_times)} steps) "
            f"loss={loss:.4f} val_loss={val_loss:.4f}"
        )

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        print(f"Training finished in {total_time:.3f}s")


def build_model_architecture(num_users, num_animes, num_genres, num_studios, num_ratings, 
                             genre_mapping, studio_mapping, rating_mapping):
    """Creates a fresh, uncompiled model architecture for each fold."""
    embedding_size = 32

    user_input = Input(shape=(1,), name='user_input')
    anime_input = Input(shape=(1,), name='anime_input')

    # ADDED NAMES: We need these names to extract the embeddings later!
    user_embed = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten()(user_embed)

    anime_embed = Embedding(input_dim=num_animes, output_dim=embedding_size, name='anime_embedding')(anime_input)
    anime_vec = Flatten()(anime_embed)

    genre_lookup = Embedding(input_dim=num_animes, output_dim=num_genres, 
                             weights=[genre_mapping], trainable=False, name='genre_lookup')(anime_input)
    genre_vec = Flatten()(genre_lookup)
    genre_dense = Dense(16, activation='relu')(genre_vec)

    studio_lookup = Embedding(input_dim=num_animes, output_dim=num_studios, 
                              weights=[studio_mapping], trainable=False, name='studio_lookup')(anime_input)
    studio_vec = Flatten()(studio_lookup)
    studio_dense = Dense(16, activation='relu')(studio_vec)

    rating_lookup = Embedding(input_dim=num_animes, output_dim=num_ratings, 
                              weights=[rating_mapping], trainable=False, name='rating_lookup')(anime_input)
    rating_vec = Flatten()(rating_lookup)
    rating_dense = Dense(8, activation='relu')(rating_vec)

    concat = Concatenate()([user_vec, anime_vec, genre_dense, studio_dense, rating_dense])

    fc1 = Dense(128, activation='relu')(concat)
    dropout1 = Dropout(0.2)(fc1)
    fc2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(fc2)
    
    output = Dense(1)(dropout2)

    model = Model(inputs=[user_input, anime_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    # Create a directory to store all our saved models and weights
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    print("--- 1. Loading and Preparing Data ---")
    anime_df = pd.read_csv('./data/anime.csv', 
                           usecols=['MAL_ID', 'Name', 'Genres', 'Rating', 'Studios'])
    ratings_df = pd.read_csv('./data/rating_complete.csv', 
                             usecols=['user_id', 'anime_id', 'rating'],
                             dtype={'user_id': 'int32', 'anime_id': 'int32', 'rating': 'float32'})

    ratings_df = ratings_df[ratings_df['rating'] > 0]

    user_ids = ratings_df['user_id'].unique()
    user_to_idx = {x: i for i, x in enumerate(user_ids)}
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx).astype('int32')

    anime_ids = anime_df['MAL_ID'].unique()
    anime_to_idx = {x: i for i, x in enumerate(anime_ids)}
    
    ratings_df = ratings_df[ratings_df['anime_id'].isin(anime_to_idx.keys())].copy()
    ratings_df['anime_idx'] = ratings_df['anime_id'].map(anime_to_idx).astype('int32')

    num_users = len(user_to_idx)
    num_animes = len(anime_to_idx)

    print("--- Injecting Negative Samples ---")
    num_positives = len(ratings_df)
    neg_users = np.random.choice(ratings_df['user_idx'].unique(), size=num_positives)
    neg_animes = np.random.choice(ratings_df['anime_idx'].unique(), size=num_positives)
    
    negatives_df = pd.DataFrame({
        'user_idx': neg_users,
        'anime_idx': neg_animes,
        'rating': 0.0  
    })
    
    ratings_df = pd.concat([ratings_df, negatives_df], ignore_index=True)
    ratings_df = ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("--- 2. Processing Metadata ---")
    anime_df['Genres'] = anime_df['Genres'].fillna('')
    anime_df['Studios'] = anime_df['Studios'].fillna('')
    anime_df['Rating'] = anime_df['Rating'].fillna('Unknown')

    genres_encoded = anime_df['Genres'].str.get_dummies(sep=', ')
    studios_encoded = anime_df['Studios'].str.get_dummies(sep=', ')
    rating_encoded = pd.get_dummies(anime_df['Rating'], prefix='Rating')

    genre_mapping = np.zeros((num_animes, genres_encoded.shape[1]), dtype='float32')
    studio_mapping = np.zeros((num_animes, studios_encoded.shape[1]), dtype='float32')
    rating_mapping = np.zeros((num_animes, rating_encoded.shape[1]), dtype='float32')

    for idx, row in anime_df.iterrows():
        a_idx = anime_to_idx.get(row['MAL_ID'])
        if a_idx is not None:
            genre_mapping[a_idx] = genres_encoded.iloc[idx].values
            studio_mapping[a_idx] = studios_encoded.iloc[idx].values
            rating_mapping[a_idx] = rating_encoded.iloc[idx].values

    X_users = ratings_df['user_idx'].values
    X_animes = ratings_df['anime_idx'].values
    y_ratings = ratings_df['rating'].values
    del ratings_df
    gc.collect() 

    print("--- 3. Starting 5-Fold Cross Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold = 1
    # We only need to pass X_users so KFold knows how many rows to split
    for train_index, val_index in kf.split(X_users):    
        print(f"\n========== TRAINING FOLD {fold} ==========")
        
        user_train, user_val = X_users[train_index], X_users[val_index]
        anime_train, anime_val = X_animes[train_index], X_animes[val_index]
        y_train, y_val = y_ratings[train_index], y_ratings[val_index]

        model = build_model_architecture(
            num_users, num_animes, 
            genres_encoded.shape[1], studios_encoded.shape[1], rating_encoded.shape[1],
            genre_mapping, studio_mapping, rating_mapping
        )

        # METHOD 1 PREP: We save the entire model as the training progresses via Checkpoint
        model_save_path = f'./saved_models/entire_model_fold_{fold}.keras'
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
        timing_callback = TimingCallback()

        print("Training in progress...")
        fold_start_time = time.time()
        model.fit(
            [user_train, anime_train], y_train,
            batch_size=8192, 
            epochs=20, 
            validation_data=([user_val, anime_val], y_val),
            callbacks=[early_stop, reduce_lr, checkpoint, timing_callback]
        )
        fold_time = time.time() - fold_start_time
        print(f"Fold {fold} completed in {fold_time:.3f}s")
        
        # ==========================================
        # IMPLEMENTING THE 3 SAVING METHODS
        # ==========================================
        print(f"\n--- Extracting Best Weights for Fold {fold} ---")
        
        # 1. METHOD 1 is already done! (Saved to 'model_save_path' by the Checkpoint callback)
        # We load it back to ensure we are extracting the peak performance weights, not the final overfit epoch.
        best_model = tf.keras.models.load_model(model_save_path)

        # 2. METHOD 2: Save ONLY the Weights
        weights_save_path = f'./saved_models/pure_weights_fold_{fold}.weights.h5'
        best_model.save_weights(weights_save_path)
        print(f"Saved pure weights to: {weights_save_path}")

        # 3. METHOD 3: Extract and Save Specific Embeddings
        # Find the layers by the exact names we assigned in `build_model_architecture`
        user_layer = best_model.get_layer('user_embedding')
        anime_layer = best_model.get_layer('anime_embedding')

        user_weights = user_layer.get_weights()[0]
        anime_weights = anime_layer.get_weights()[0]

        np.save(f'./saved_models/user_vectors_fold_{fold}.npy', user_weights)
        np.save(f'./saved_models/anime_vectors_fold_{fold}.npy', anime_weights)
        print(f"Saved {user_weights.shape[0]} User Vectors and {anime_weights.shape[0]} Anime Vectors to Numpy arrays.")
        
        # Clean up memory to prevent a crash on the next fold
        tf.keras.backend.clear_session()
        del model
        del best_model
        gc.collect()
        
        fold += 1

    print("\n--- Training Complete! All 3 formats have been successfully saved to the ./saved_models directory. ---")

if __name__ == "__main__":
    main()