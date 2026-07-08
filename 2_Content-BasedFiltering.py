import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Load Data
# We only need the anime metadata for this algorithm
animes = pd.read_csv('./data/anime.csv', usecols=['MAL_ID', 'Name', 'Genres'])

# 2. Preprocess
# Replace NaN values with empty strings to avoid vectorizer errors
animes['Genres'] = animes['Genres'].fillna('')

# 3. Build the TF-IDF Matrix
# stop_words='english' removes common words like 'and', 'the', etc.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(animes['Genres'])

print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
# Example output: (17562, 46) -> 17,562 animes described by 46 unique genres

# 4. Compute Similarity
# We use linear_kernel instead of cosine_similarity because it is significantly 
# faster and uses less memory when working with TF-IDF matrices.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of indices and anime titles for quick lookups
indices = pd.Series(animes.index, index=animes['Name']).drop_duplicates()

def get_content_recommendations(title, df=animes, cosine_sim=cosine_sim, top_n=10):
    if title not in indices:
        return "Anime not found in database."
        
    # Get the index of the anime that matches the title
    idx = indices[title]

    # Get pairwise similarity scores of all anime with that anime
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the anime based on the similarity scores (Highest to lowest)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar anime (Skip index 0, which is the anime itself)
    sim_scores = sim_scores[1:top_n+1]

    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Return the top N most similar anime along with their genres to verify the logic
    return df[['Name', 'Genres']].iloc[anime_indices]

# Test the function
print("Recommendations for 'Trigun':")
print(get_content_recommendations('Trigun'))