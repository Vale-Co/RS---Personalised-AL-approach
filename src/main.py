# ============================================================
# Popularity, Random, Poperror strategies for Cold-Start User Simulation
# ============================================================

# STEP 1: Upload dataset file (user-item interactions)
# This step allows you to upload a CSV file (e.g., useritemmatrix.csv) in Google Colab.
from google.colab import files
uploaded = files.upload()

# STEP 2: Import necessary libraries
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
import random
from collections import Counter

# STEP 3: Fix randomness for reproducibility
# This ensures consistent results every time the code is run.
random.seed(1)
np.random.seed(1)

# STEP 4: Load the interaction data
# Assumes a CSV file with columns: userId, itemId, interaction (binary or ratings)
data = pd.read_csv("useritemmatrix.csv")

# STEP 5: Remove any users with no interactions
# Ensures we only consider active users who have interacted with items
data = data.groupby('userId').filter(lambda x: len(x) > 0)

# STEP 6: Encode users and items as numeric indexes
# This makes the data compatible with machine learning models like SVD
data['user_idx'] = data['userId'].astype('category').cat.codes
data['item_idx'] = data['itemId'].astype('category').cat.codes

# STEP 7: Simulate Cold-Start Users
# Define a fraction of users to be treated as new/cold users (never seen by the model)
cold_user_fraction = 0.25
all_users  = data['user_idx'].unique()

# Randomly select 25% of the users to simulate cold-start scenarios
cold_users = np.random.choice(all_users,
                              size=int(len(all_users)*cold_user_fraction),
                              replace=False)

# STEP 8: Filter dataset to exclude cold users for model training
# The training data includes only "warm" users (i.e., seen by the system)
warm_data = data[~data['user_idx'].isin(cold_users)]

# STEP 9: Count how many times each item was interacted with
# Useful for popularity-based selection
item_counts = warm_data['itemId'].value_counts()

# STEP 10: Filter items with at least 10 interactions
# To ensure item quality and avoid noise or sparse data
eligible_items = item_counts[item_counts >= 10].index.tolist()

# ------------------------------------------------------------
# STEP 11: Misclassification Error Function
# Used to calculate how ambiguous or diverse the feedback is for a given item.
# High error indicates users disagree on liking/disliking the item.
def misclassification_error(labels):
    n = len(labels)
    if n == 0: return 0
    probs = np.bincount(labels, minlength=2)/n
    return 1 - np.max(probs)

# STEP 12: Compute error-based and hybrid (poperror) scores
# These will help define item selection strategies.
error_scores   = {}         # Tracks ambiguity of item interactions
poperror_scores = {}       # Combines popularity and ambiguity

for item in eligible_items:
    labels = data[data['itemId'] == item]['interaction'].values
    err = misclassification_error(labels)         # Ambiguity in feedback
    freq = item_counts[item]                      # Popularity (frequency)
    error_scores[item]   = err
    poperror_scores[item] = 0.9*np.log10(freq) + 1*err  # Weighted hybrid score

# ------------------------------------------------------------
# STEP 13: Item Selection Strategies
# Used to simulate what cold users are shown initially
# - 'random': randomly pick items
# - 'popularity': most interacted items
# - 'poperror': hybrid score that balances popularity and ambiguity
def select_items(strategy, k):
    if strategy == 'random':
        return random.sample(eligible_items, k)
    elif strategy == 'popularity':
        return list(item_counts.loc[eligible_items].sort_values(ascending=False)
                                                  .head(k).index)
    elif strategy == 'poperror':
        return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
    else:
        raise ValueError("Invalid strategy.")

# ------------------------------------------------------------
# STEP 14: Create train and test splits for cold users
# Cold users only see selected "shown" items in the training data
# The rest of their interactions are placed into the test set
def create_train_test(df, cold, shown):
    # Train includes all interactions from warm users,
    # and only shown items for cold users
    train = df[~df['user_idx'].isin(cold) |
              ((df['user_idx'].isin(cold)) & (df['itemId'].isin(shown)))]
    
    # Get only cold users who have interacted with the shown items
    interacted = train[train['user_idx'].isin(cold)]['user_idx'].unique()
    
    # Test includes remaining items for these cold users
    test = df[(df['user_idx'].isin(interacted)) & 
              (~df['itemId'].isin(shown))]
    
    return train, test

# ------------------------------------------------------------
# STEP 15: Experiment Loop
# Test each strategy with different values of k (number of shown items)
results = []
reader = Reader(rating_scale=(0, 1))  # For binary interaction (0/1) dataset

# Loop through strategies and different values of k
for strat in ['random', 'popularity', 'poperror']:
    for k in [10, 25, 50, 100]:
        # Select k items using the current strategy
        shown = select_items(strat, k)
        
        # Create training and testing data based on shown items
        train_df, test_df = create_train_test(data, cold_users, shown)
        
        # Prepare the training data for the Surprise library
        dset = Dataset.load_from_df(train_df[['user_idx', 'item_idx', 'interaction']],
                                    reader)
        trainset = dset.build_full_trainset()

        # Train a collaborative filtering model using SVD (Matrix Factorization)
        # SVD helps predict user-item interactions based on latent features
        model = SVD(n_factors=200, reg_all=1e-6, biased=True,
                    random_state=1)

        # (Further steps such as model training and evaluation would go here)

