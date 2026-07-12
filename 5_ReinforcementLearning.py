import pandas as pd
import numpy as np

# 1. Load Data and set up the simulation environment
animes = pd.read_csv('./data/anime.csv')
animes['Score'] = pd.to_numeric(animes['Score'], errors='coerce')

# Drop animes with missing scores and get the Top 20 most popular
candidates = animes.dropna(subset=['Score']).sort_values(by='Popularity').head(20).copy()

# Create a "true reward probability" based on the actual MAL score
# (e.g., Score 8.78 -> 0.878 chance the simulated user clicks/watches)
candidates['True_Reward_Prob'] = candidates['Score'].astype(float) / 10.0

anime_names = candidates['Name'].tolist()
true_probabilities = candidates['True_Reward_Prob'].tolist()

# 2. Build the RL Agent
class EpsilonGreedyBandit:
    def __init__(self, item_names, epsilon=0.1):
        self.items = item_names
        self.n_arms = len(item_names)
        self.epsilon = epsilon
        
        # Trackers
        self.counts = np.zeros(self.n_arms)   # How many times an anime was shown
        self.rewards = np.zeros(self.n_arms)  # How many clicks it got
        self.q_values = np.zeros(self.n_arms) # Estimated value (Rewards / Counts)

    def select_item(self):
        # EXPLORE: Epsilon % of the time, pick a completely random anime
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        # EXPLOIT: The rest of the time, pick the anime with the highest current Q-value
        else:
            return np.argmax(self.q_values)

    def update_knowledge(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.rewards[chosen_arm] += reward
        # Recalculate the expected reward for this anime
        self.q_values[chosen_arm] = self.rewards[chosen_arm] / self.counts[chosen_arm]

# 3. Run the Simulation
# We initialize the bandit with a 15% exploration rate
agent = EpsilonGreedyBandit(anime_names, epsilon=0.15)
total_simulated_users = 10000

print(f"Simulating {total_simulated_users} user interactions...\n")

for _ in range(total_simulated_users):
    # The agent decides which anime to show the user
    chosen_idx = agent.select_item()
    
    # The simulated user decides whether to click, based on the hidden probability
    hidden_prob = true_probabilities[chosen_idx]
    user_clicked = 1 if np.random.random() < hidden_prob else 0
    
    # The agent updates its mathematical model based on the user's action
    agent.update_knowledge(chosen_idx, user_clicked)

# 4. Results Display
results = pd.DataFrame({
    'Anime': anime_names,
    'Times_Shown': agent.counts,
    'Learned_Q_Value (Est. Click Rate)': agent.q_values,
    'Actual_MAL_Score_Prob': true_probabilities
})

# Sort by how many times the AI chose to show it
results = results.sort_values(by='Times_Shown', ascending=False)

print("Final Knowledge State of the Reinforcement Learning Agent:")
print("Notice how the agent heavily favors showing the highest-scored animes.")
print("-" * 75)
print(results.to_string(index=False))