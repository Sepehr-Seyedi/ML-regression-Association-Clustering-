# ml-regression-association-clustering

A compact educational collection demonstrating three foundational machine learning techniques:
- **Linear regression** (manual normal equation and scikit-learn)
- **Association rule mining** (support and confidence)
- **KMeans clustering** (visualization and evaluation)

This repository is intended for learning, experimentation, and small demos. Each project is self-contained and includes scripts, example usage, and notes on extensions.

## Repository layout

- `regression/`  
  - `multiple_regression_numpy.py` — multiple linear regression using the normal equation and 3D regression plane visualization.  
  - `simple_regression_sklearn.py` — simple linear regression with train/test split using scikit-learn and 2D plot.  
  - `comparison_manual_vs_sklearn.py` — compares manual normal-equation solution with scikit-learn on a single feature.

- `association_rules/`  
  - `apriori_bruteforce.py` — brute-force frequent itemset enumeration and association rule generation using support and confidence thresholds.

- `clustering/`  
  - `kmeans_evaluation.py` — synthetic data generation, KMeans clustering, 2D visualization, and evaluation metrics (purity and entropy).

- `data/`  
  - Optional sample dataset(s). Scripts assume a CSV at `data/dataset.csv` or use synthetic data where noted.

- `requirements.txt`  
  - Lists Python dependencies.

## Quickstart

1. Clone the repository
```bash
git clone https://github.com/<your-username>/ml-regression-association-clustering.git
cd ml-regression-association-clustering
