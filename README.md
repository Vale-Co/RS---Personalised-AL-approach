# Recommender Systems-Personalised-AL-approach
<!-- ============================================================== -->
<!--  Active‑Learning for Cold‑Start Recommendation (Thesis Project) -->
<!-- ============================================================== -->

# Active‑Learning for **Cold‑Start Recommender Systems**  
*A Python (NumPy + Surprise) replication & sandbox for thesis experiments*

> **Goal of the thesis project**  
> 1. **Replicate** the results from **“Active Learning Strategies for Solving the Cold‑User Problem in Model‑Based Recommender Systems”**  
>    (T. Geurts et al., *Web Intelligence* 2020).  
> 2. **Compare** several non‑personalised active‑learning strategies with a personalised approach (SHHP).  
> 3. Provide a clean, single‑file baseline (`main.py`) that can be easily *extended* with new heuristics, models, or datasets.

The current script supports **two experiment tracks**:

| Section in&nbsp;`main.py` | Strategy / logic                                     | Personalised? |  *k* values asked | Notes |
|---------------------------|------------------------------------------------------|---------------|-------------------|-------|
| **5 – 6** (*global pass*) | `random`, `popularity`, `poperror`                   | **No**  | **10, 25, 50, 100** | Matches Table 1 of the paper |
| **7** (*SHHP loop*)       | `shhp` – *Single‑Heuristic Highest‑Predicted* (ask 1‑by‑1) | **Yes**  | **10** (default, editable) | Runs only on the **first 100** cold users, exactly as in WI‑2020 |

---

*Why two tracks?*  
*Global* strategies simulate a shop banner that shows the **same** 10/25/50/100 items to every newcomer.  
*SHHP* imitates a **short interview** – the system adapts the next question based on what the current user just rated, stopping after *k* items.

Feel free to fork this repo and plug in:

* different rating models (e.g. LightFM, implicit ALS, VAEs),  
* alternative interview heuristics,  
* richer evaluation metrics (Precision@k, NDCG, coverage).

Happy experimenting!
