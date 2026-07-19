import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
import gc
import os

def build_model_architecture(num_users, num_animes, num_genres, num_studios, num_ratings, 
                             num_types, num_premiered, num_sources,
                             genre_mapping, studio_mapping, rating_mapping,
                             type_mapping, premiered_mapping, source_mapping, member_mapping):
    """Creates a fresh, uncompiled model architecture for each fold."""
    embedding_size = 32

    user_input = Input(shape=(1,), name='user_input')
    anime_input = Input(shape=(1,), name='anime_input')

    # Core Collaborative Embeddings
    user_embed = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten()(user_embed)

    anime_embed = Embedding(input_dim=num_animes, output_dim=embedding_size, name='anime_embedding')(anime_input)
    anime_vec = Flatten()(anime_embed)

    # --- CATEGORICAL METADATA LOOKUPS ---
    genre_lookup = Embedding(input_dim=num_animes, output_dim=num_genres, weights=[genre_mapping], trainable=False, name='genre_lookup')(anime_input)
    genre_vec = Flatten()(genre_lookup)
    genre_dense = Dense(16, activation='relu')(genre_vec)

    studio_lookup = Embedding(input_dim=num_animes, output_dim=num_studios, weights=[studio_mapping], trainable=False, name='studio_lookup')(anime_input)
    studio_vec = Flatten()(studio_lookup)
    studio_dense = Dense(16, activation='relu')(studio_vec)

    rating_lookup = Embedding(input_dim=num_animes, output_dim=num_ratings, weights=[rating_mapping], trainable=False, name='rating_lookup')(anime_input)
    rating_vec = Flatten()(rating_lookup)
    rating_dense = Dense(8, activation='relu')(rating_vec)

    type_lookup = Embedding(input_dim=num_animes, output_dim=num_types, weights=[type_mapping], trainable=False, name='type_lookup')(anime_input)
    type_vec = Flatten()(type_lookup)
    type_dense = Dense(8, activation='relu')(type_vec)

    premiered_lookup = Embedding(input_dim=num_animes, output_dim=num_premiered, weights=[premiered_mapping], trainable=False, name='premiered_lookup')(anime_input)
    premiered_vec = Flatten()(premiered_lookup)
    premiered_dense = Dense(16, activation='relu')(premiered_vec)

    source_lookup = Embedding(input_dim=num_animes, output_dim=num_sources, weights=[source_mapping], trainable=False, name='source_lookup')(anime_input)
    source_vec = Flatten()(source_lookup)
    source_dense = Dense(8, activation='relu')(source_vec)

    # --- NUMERICAL METADATA LOOKUP ---
    member_lookup = Embedding(input_dim=num_animes, output_dim=1, weights=[member_mapping], trainable=False, name='member_lookup')(anime_input)
    member_vec = Flatten()(member_lookup) 

    # Combine all branches
    concat = Concatenate()([
        user_vec, anime_vec, 
        genre_dense, studio_dense, rating_dense, 
        type_dense, premiered_dense, source_dense, 
        member_vec
    ])

    fc1 = Dense(128, activation='relu')(concat)
    dropout1 = Dropout(0.2)(fc1)
    fc2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(fc2)
    
    output = Dense(1)(dropout2)

    model = Model(inputs=[user_input, anime_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    print("--- 1. Loading and Preparing Data ---")
    anime_df = pd.read_csv('./data/anime.csv', 
                           usecols=['MAL_ID', 'Name', 'Genres', 'Rating', 'Studios', 'Type', 'Premiered', 'Source', 'Members'])
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

    """print("--- Injecting Negative Samples ---")
    num_positives = len(ratings_df)
    neg_users = np.random.choice(ratings_df['user_idx'].unique(), size=num_positives)
    neg_animes = np.random.choice(ratings_df['anime_idx'].unique(), size=num_positives)
    
    negatives_df = pd.DataFrame({'user_idx': neg_users, 'anime_idx': neg_animes, 'rating': 0.0})
    ratings_df = pd.concat([ratings_df, negatives_df], ignore_index=True)
    ratings_df = ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)"""

    print("--- 2. Processing Metadata (Categorical and Numerical) ---")
    anime_df['Genres'] = anime_df['Genres'].fillna('')
    anime_df['Studios'] = anime_df['Studios'].fillna('')
    anime_df['Rating'] = anime_df['Rating'].fillna('Unknown')
    anime_df['Type'] = anime_df['Type'].fillna('Unknown')
    anime_df['Premiered'] = anime_df['Premiered'].fillna('Unknown')
    anime_df['Source'] = anime_df['Source'].fillna('Unknown')

    genres_encoded = anime_df['Genres'].str.get_dummies(sep=', ')
    studios_encoded = anime_df['Studios'].str.get_dummies(sep=', ')
    rating_encoded = pd.get_dummies(anime_df['Rating'], prefix='Rating')
    type_encoded = pd.get_dummies(anime_df['Type'], prefix='Type')
    premiered_encoded = pd.get_dummies(anime_df['Premiered'], prefix='Premiered')
    source_encoded = pd.get_dummies(anime_df['Source'], prefix='Source')

    anime_df['Members'] = pd.to_numeric(anime_df['Members'], errors='coerce').fillna(0)
    members_log = np.log1p(anime_df['Members'].values)
    members_normalized = (members_log - members_log.min()) / (members_log.max() - members_log.min() + 1e-9)

    genre_mapping = np.zeros((num_animes, genres_encoded.shape[1]), dtype='float32')
    studio_mapping = np.zeros((num_animes, studios_encoded.shape[1]), dtype='float32')
    rating_mapping = np.zeros((num_animes, rating_encoded.shape[1]), dtype='float32')
    type_mapping = np.zeros((num_animes, type_encoded.shape[1]), dtype='float32')
    premiered_mapping = np.zeros((num_animes, premiered_encoded.shape[1]), dtype='float32')
    source_mapping = np.zeros((num_animes, source_encoded.shape[1]), dtype='float32')
    member_mapping = np.zeros((num_animes, 1), dtype='float32')

    for idx, row in anime_df.iterrows():
        a_idx = anime_to_idx.get(row['MAL_ID'])
        if a_idx is not None:
            genre_mapping[a_idx] = genres_encoded.iloc[idx].values
            studio_mapping[a_idx] = studios_encoded.iloc[idx].values
            rating_mapping[a_idx] = rating_encoded.iloc[idx].values
            type_mapping[a_idx] = type_encoded.iloc[idx].values
            premiered_mapping[a_idx] = premiered_encoded.iloc[idx].values
            source_mapping[a_idx] = source_encoded.iloc[idx].values
            member_mapping[a_idx] = members_normalized[idx]

    X_users = ratings_df['user_idx'].values
    X_animes = ratings_df['anime_idx'].values
    y_ratings = ratings_df['rating'].values
    del ratings_df
    gc.collect() 

    print("\n--- 2.5 Setting Aside the Unseen Test Set ---")
    X_users_cv, X_users_test, X_animes_cv, X_animes_test, y_ratings_cv, y_ratings_test = train_test_split(
        X_users, X_animes, y_ratings, test_size=0.10, random_state=42
    )
    
    del X_users, X_animes, y_ratings
    gc.collect()
    
    BATCH_SIZE = 4096 
    test_dataset = tf.data.Dataset.from_tensor_slices(
        ({'user_input': X_users_test, 'anime_input': X_animes_test}, y_ratings_test)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"Cross-Validation Set: {len(y_ratings_cv)} rows")
    print(f"Locked Test Set:      {len(y_ratings_test)} rows")

    print("\n--- 3. Starting 5-Fold Cross Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold = 1
    for train_index, val_index in kf.split(X_users_cv):
        print(f"\n========== TRAINING FOLD {fold} ==========")
        
        user_train, user_val = X_users_cv[train_index], X_users_cv[val_index]
        anime_train, anime_val = X_animes_cv[train_index], X_animes_cv[val_index]
        y_train, y_val = y_ratings_cv[train_index], y_ratings_cv[val_index]

        model = build_model_architecture(
            num_users, num_animes, 
            genres_encoded.shape[1], studios_encoded.shape[1], rating_encoded.shape[1],
            type_encoded.shape[1], premiered_encoded.shape[1], source_encoded.shape[1],
            genre_mapping, studio_mapping, rating_mapping,
            type_mapping, premiered_mapping, source_mapping, member_mapping
        )

        model_save_path = f'./saved_models/entire_model_fold_{fold}.keras'
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)

        print("Preparing high-speed tf.data pipelines...")
        # We need a buffer size for shuffling. For huge datasets, 100,000 is a 
        # sweet spot that shuffles well without crashing your RAM.
        SHUFFLE_BUFFER = 100000 
        
        # Notice we added .shuffle() right before .batch()
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'user_input': user_train, 'anime_input': anime_train}, y_train)
        ).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Validation data does NOT need to be shuffled, so we leave it as is
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'user_input': user_val, 'anime_input': anime_val}, y_val)
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        print("Training in progress...")
        history = model.fit(
            train_dataset,
            epochs=20, 
            validation_data=val_dataset,
            shuffle=False, # <-- This explicitly tells Keras to stop warning you!
            callbacks=[early_stop, reduce_lr, checkpoint]
        )
        
        print(f"\n--- Saving Training History for Fold {fold} ---")
        hist_df = pd.DataFrame(history.history)
        hist_df.insert(0, 'epoch', range(1, len(hist_df) + 1)) 
        hist_df.to_csv(f'./saved_models/training_history_fold_{fold}.csv', index=False)
        
        print(f"\n--- Extracting Best Weights for Fold {fold} ---")
        best_model = tf.keras.models.load_model(model_save_path)

        print(f"--- Evaluating Fold {fold} on the True Test Set ---")
        test_loss, test_mae = best_model.evaluate(test_dataset, verbose=1)
        print(f"-> FOLD {fold} OBJECTIVE MAE: {test_mae:.4f}")

        weights_save_path = f'./saved_models/pure_weights_fold_{fold}.weights.h5'
        best_model.save_weights(weights_save_path)

        user_weights = best_model.get_layer('user_embedding').get_weights()[0]
        anime_weights = best_model.get_layer('anime_embedding').get_weights()[0]

        np.save(f'./saved_models/user_vectors_fold_{fold}.npy', user_weights)
        np.save(f'./saved_models/anime_vectors_fold_{fold}.npy', anime_weights)
        
        tf.keras.backend.clear_session()
        del model, best_model, train_dataset, val_dataset
        gc.collect()
        
        fold += 1

    print("\n--- Training Complete! All models, history, and test metrics have been successfully saved. ---")

if __name__ == "__main__":
    main()