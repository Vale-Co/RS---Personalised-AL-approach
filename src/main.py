# ============================================================
# Popularity, Random, Poperror strategies
# ============================================================

# 1 - Install required packages 
# !pip install numpy==1.24.4
# !pip install scikit-surprise
from google.colab import files
uploaded = files.upload()

# 2 - Import libraries
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
import random
from collections import Counter

random.seed(1)
np.random.seed(1)

data = pd.read_csv("useritemmatrix.csv")
data = data.groupby('userId').filter(lambda x: len(x) > 0)
data['user_idx'] = data['userId'].astype('category').cat.codes
data['item_idx'] = data['itemId'].astype('category').cat.codes

cold_user_fraction = 0.25
all_users  = data['user_idx'].unique()
cold_users = np.random.choice(all_users,
                              size=int(len(all_users)*cold_user_fraction),
                              replace=False)

warm_data   = data[~data['user_idx'].isin(cold_users)]
item_counts = warm_data['itemId'].value_counts()
eligible_items = item_counts[item_counts >= 10].index.tolist()

def misclassification_error(labels):
    n = len(labels)
    if n == 0: return 0
    probs = np.bincount(labels, minlength=2)/n
    return 1 - np.max(probs)

error_scores   = {}
poperror_scores = {}
for item in eligible_items:
    labels = data[data['itemId']==item]['interaction'].values
    err = misclassification_error(labels)
    freq = item_counts[item]
    error_scores[item]   = err
    poperror_scores[item]= 0.9*np.log10(freq) + 1*err

def select_items(strategy,k):
    if strategy=='random':
        return random.sample(eligible_items,k)
    elif strategy=='popularity':
        return list(item_counts.loc[eligible_items].sort_values(ascending=False)
                                                  .head(k).index)
    elif strategy=='poperror':
        return sorted(poperror_scores,key=poperror_scores.get,reverse=True)[:k]
    else:
        raise ValueError("Invalid strategy.")

def create_train_test(df,cold,shown):
    train = df[~df['user_idx'].isin(cold) |
              ((df['user_idx'].isin(cold)) & (df['itemId'].isin(shown)))]
    interacted = train[train['user_idx'].isin(cold)]['user_idx'].unique()
    test = df[(df['user_idx'].isin(interacted)) &
              (~df['itemId'].isin(shown))]
    return train,test

results = []
reader = Reader(rating_scale=(0,1))

for strat in ['random','popularity','poperror']:
    for k in [10,25,50,100]:
        shown = select_items(strat,k)
        train_df,test_df = create_train_test(data,cold_users,shown)
        dset = Dataset.load_from_df(train_df[['user_idx','item_idx','interaction']],
                                    reader)
        trainset = dset.build_full_trainset()
        model = SVD(n_factors=200, reg_all=1e-6, biased=True,
                    random_state=1, n_epochs=50)
        model.fit(trainset)
        preds = [model.predict(uid=row.user_idx, iid=row.item_idx, r_ui=row.interaction)
                 for row in test_df.itertuples()]
        rmse = accuracy.rmse(preds, verbose=False)
        results.append((strat,k,len(cold_users),rmse))

df_results = pd.DataFrame(results,
                          columns=['Strategy','ItemsShown','ColdUsers','RMSE'])
print("=== Global strategies ===")
print(df_results)



# ============================================================
#  SHHP 
# ============================================================

from tqdm import tqdm

reader = Reader(rating_scale=(0,1))
most_popular_iid = item_counts.idxmax()

shhp_records = []

for k_shhp in [10,25,50,100]:
    print(f"\n-- SHHP for k={k_shhp} --")
    per_user_rmse = []

    # first 100 cold users (same sample for every k)
    for u in tqdm(cold_users[:100]):
        shown = [most_popular_iid]

        while len(shown) < k_shhp:
            train_df = data[~data['user_idx'].isin(cold_users) |
                            ((data['user_idx']==u) & (data['itemId'].isin(shown)))]

            trainset = Dataset.load_from_df(train_df[['user_idx','item_idx','interaction']],
                                            reader).build_full_trainset()
            svd = SVD(n_factors=200, reg_all=1e-6, biased=True)
            svd.fit(trainset)

            # score all candidate items
            best_item = None
            best_est  = -1
            for iid in eligible_items:
                if iid in shown: continue
                est = svd.predict(uid=u, iid=data.loc[data['itemId']==iid,'item_idx'].iloc[0]).est
                if est > best_est:
                    best_est, best_item = est, iid
            if best_item is None: break
            shown.append(best_item)

        # evaluate on unseen for that user
        test_df = data[(data['user_idx']==u) & (~data['itemId'].isin(shown))]
        if len(test_df):
            preds = [svd.predict(uid=row.user_idx, iid=row.item_idx, r_ui=row.interaction)
                     for row in test_df.itertuples()]
            rmse = accuracy.rmse(preds, verbose=False)
            per_user_rmse.append(rmse)

    avg_rmse = np.mean(per_user_rmse) if per_user_rmse else np.nan
    shhp_records.append(('SHHP',k_shhp,len(per_user_rmse),avg_rmse))
    print(f"Average RMSE  (k={k_shhp}): {avg_rmse:.4f}")

df_shhp_all = pd.DataFrame(shhp_records,
                            columns=['Strategy','ItemsShown','UsersEvaluated','RMSE'])
print("\n=== SHHP summary ===")
print(df_shhp_all)





