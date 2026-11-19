# Machine Learning Labs

A curated collection of hands-on Machine Learning experiments and comparative studies developed during the **M.Sc. in Computer Science (Artificial Intelligence)** at the University of Padua. Each notebook combines mathematical formulation, reproducible Python implementations, and quantitative/visual evaluation.

## Experiments Overview

| # | Notebook | Domain | Methods & Algorithms | Dataset | Key Highlights |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | [Hybrid Recommender](notebooks/01_hybrid_recommender_movielens.ipynb) | Recommender Systems | TF-IDF + Cosine Similarity, User-Based CF, Weighted Fusion ($\alpha = 0.5$) | MovieLens 100k | Balances content relevance and social validation to mitigate Cold-Start and Filter Bubbles |
| 2 | [Cold-Start Classification](notebooks/02_cold_start_random_forest.ipynb) | Supervised Learning | Random Forest (100 trees), One-Hot Encoding, Gini Feature Importance | MovieLens 100k | Reframes Cold-Start recommendation as binary classification (**0.651 ROC-AUC**, **0.62 accuracy**) |
| 3 | [EM / GMM vs. K-Means](notebooks/03_em_gmm_vs_kmeans_clustering.ipynb) | Unsupervised Learning | K-Means, Full-Covariance GMM (EM), AIC / BIC Model Selection | Anisotropic Synthetic Blobs | Full-covariance GMM achieves **1.000 ARI** vs. **0.558 ARI** for K-Means; AIC/BIC recover optimal $k = 3$ |
| 4 | [Bayesian Networks](notebooks/04_bayesian_networks_structure_learning.ipynb) | Probabilistic Graphical Models | Categorical Naive Bayes, Hill Climbing + K2 Structure Learning (`bnlearn`) | UCI Heart Disease | Compares conditional independence baseline (**86.7% accuracy**) against learned DAG dependencies |
| 5 | [Bias-Variance & K-Fold CV](notebooks/05_bias_variance_kfold_cv.ipynb) | Model Selection | Polynomial Regression ($p \in \{1, 3, 5, 9, 15, 30\}$), 5-Fold Cross-Validation | Noisy Sinusoidal Synthetic Data | Identifies optimal degree $p = 5$ (**CV MSE: 0.0104**, **Test MSE: 0.0096**) |

## Tech Stack

- **Language:** Python 3
- **Core Libraries:** `scikit-learn`, `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- **Probabilistic Graphical Models:** `bnlearn`

## Repository Structure

```text
Machine-Learning-Labs/
├── notebooks/
│   ├── 01_hybrid_recommender_movielens.ipynb
│   ├── 02_cold_start_random_forest.ipynb
│   ├── 03_em_gmm_vs_kmeans_clustering.ipynb
│   ├── 04_bayesian_networks_structure_learning.ipynb
│   └── 05_bias_variance_kfold_cv.ipynb
└── README.md
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone [https://github.com/huraken00/Machine-Learning-Labs.git](https://github.com/huraken00/Machine-Learning-Labs.git)
   cd Machine-Learning-Labs
   ```
2. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn bnlearn requests jupyter
   ```
3. **Launch Jupyter**
   ```bash
   jupyter notebook notebooks/
   ```
   *(All datasets are downloaded automatically or generated synthetically within each notebook for full reproducibility.)*
