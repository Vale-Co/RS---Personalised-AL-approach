# 🎓 Active Learning for Cold-Start Recommendation

This repository contains the code used in the Master's thesis project _“Personalised Active Learning Strategies for Cold-Start Recommender Systems”_ by Valentina Conz.

The goal of the project was to explore whether **personalised active learning strategies** can outperform non-personalised ones in solving the **cold start problem** in collaborative filtering recommender systems.

---

## 🧠 Project Description

Recommender systems struggle with cold users—new users who have no interaction history. To address this, the thesis evaluates different **active learning strategies** that determine which items to show to cold users in the early stages.

Four strategies were compared:
- `random` – selects random items
- `popularity` – selects the most popular items
- `poperror` – a hybrid that combines item popularity and uncertainty
- `shhp` – **Single Heuristic Highest Predicted**, a novel sequential strategy that personalises item selection based on the user's responses

The results show that **SHHP**, though more computationally intensive, can outperform non-personalised strategies in terms of recommendation quality.

---

## 📂 Files

- `main.py` — Main script with all experiments
- `useritemmatrix.csv` — The user-item interaction dataset


---

## 📊 Results

The SHHP strategy was evaluated on **100 cold users** (due to compute constraints), while the other strategies used 25% of the cold users. Results show that **personalised item selection leads to better predictions**, validating the value of adaptive cold-start handling.

