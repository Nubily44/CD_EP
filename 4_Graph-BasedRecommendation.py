import pandas as pd
import networkx as nx

# 1. Load Data (Downcasted for memory)
animelist = pd.read_csv('./data/animelist.csv', 
                        usecols=['user_id', 'anime_id', 'rating'],
                        dtype={'user_id': 'int32', 'anime_id': 'int32', 'rating': 'int8'})
animes = pd.read_csv('./data/anime.csv', usecols=['MAL_ID', 'Name'])

# 2. Aggressive Graph Optimization
# ONLY map relationships where the user gave a 9 or 10 (strongest possible link)
strong_links = animelist[animelist['rating'] >= 9]

# Filter for active users and popular anime to reduce the number of solitary nodes
user_counts = strong_links['user_id'].value_counts()
anime_counts = strong_links['anime_id'].value_counts()

# Keep users who gave at least 50 nines/tens, and anime that received at least 100
active_users = user_counts[user_counts >= 50].index
popular_animes = anime_counts[anime_counts >= 100].index

filtered_links = strong_links[
    (strong_links['user_id'].isin(active_users)) & 
    (strong_links['anime_id'].isin(popular_animes))
]

# Merge names for the final output
filtered_links = filtered_links.merge(animes, left_on='anime_id', right_on='MAL_ID')

# 3. Build the Bipartite Graph
G = nx.Graph()

# Add a 'U_' prefix to user IDs to prevent overlap with Anime IDs
users = ['U_' + str(u) for u in filtered_links['user_id'].unique()]
anime_nodes = filtered_links['Name'].unique().tolist()

G.add_nodes_from(users, bipartite=0)
G.add_nodes_from(anime_nodes, bipartite=1)

# Map the edges (User -> Anime)
edges = [('U_' + str(row.user_id), row.Name) for _, row in filtered_links.iterrows()]
G.add_edges_from(edges)
print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

def get_graph_recommendations(target_user_id, graph, top_n=5):
    target_node = 'U_' + str(target_user_id)
    
    if target_node not in graph:
        return "User not found in the highly-filtered graph."
        
    # Set up Personalized PageRank
    # We assign a weight of 1.0 to our target user, and 0.0 to everyone else.
    # The algorithm will "walk" the graph starting from this user to find related nodes.
    personalization = {node: 0 for node in graph.nodes()}
    personalization[target_node] = 1.0
    
    # Calculate PageRank
    pagerank_scores = nx.pagerank(graph, personalization=personalization)
    
    # Get anime the user has already strongly rated so we don't recommend them
    user_watched = set(graph.neighbors(target_node))
    
    # Filter results: Keep only anime nodes (bipartite=1) that the user hasn't seen
    recommendations = {
        node: score for node, score in pagerank_scores.items() 
        if graph.nodes[node].get('bipartite') == 1 and node not in user_watched
    }
    
    # Sort by the highest PageRank score
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Test with a user ID that exists in our filtered users
sample_user = active_users[0]
print(f"\nGraph Recommendations for User {sample_user}:")
for anime, score in get_graph_recommendations(sample_user, G):
    print(f"{anime} (Network Score: {score:.6f})")