# ml-regression-association-clustering

A compact educational collection demonstrating three foundational machine learning techniques: **Linear Regression** (NumPy normal equation and scikit‑learn), **Association Rule Mining** (support and confidence brute force), and **KMeans Clustering** (synthetic data, visualization, purity and entropy). Each project is self contained with scripts, example usage, and visual outputs to help you learn, compare implementations, and extend the methods for real datasets.

---

## Project overview

**What this repository contains**

- **Linear Regression Demo**  
  - Manual multiple linear regression using the normal equation with a 3D regression plane visualization.  
  - Simple linear regression using scikit‑learn with train test split and 2D visualization.  
  - A comparison script that validates the manual solution against scikit‑learn.

- **Association Rule Mining Demo**  
  - Brute force frequent itemset enumeration and association rule generation using **support** and **confidence** thresholds.  
  - Educational implementation suitable for small transactional datasets.

- **KMeans Clustering Evaluation**  
  - Synthetic data generation with `make_blobs`, KMeans clustering, 2D visualization, and evaluation using **purity** and **entropy** metrics.

**Who this is for**

- Students and beginners learning core ML algorithms and their implementations.  
- Practitioners who want compact, readable examples to adapt for teaching or quick experiments.  
- Anyone who wants to compare manual algorithmic implementations with scikit‑learn.

---

## Repository structure and files

```
ml-regression-association-clustering/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ data/
│  └─ dataset.csv   # optional sample or instructions to add your own
├─ regression/
│  ├─ multiple_regression_numpy.py
│  ├─ simple_regression_sklearn.py
│  └─ comparison_manual_vs_sklearn.py
├─ association_rules/
│  └─ apriori_bruteforce.py
└─ clustering/
   └─ kmeans_evaluation.py
```

**Key files explained**

- **multiple_regression_numpy.py**  
  Computes weights using the normal equation \(w = (X^T X)^{-1} X^T y\) and plots a 3D regression plane for two features.

- **simple_regression_sklearn.py**  
  Uses scikit‑learn `LinearRegression` with a train test split, prints coefficients, and plots predictions vs actuals.

- **comparison_manual_vs_sklearn.py**  
  Compares manual normal equation results with scikit‑learn on a single feature and overlays fitted lines.

- **apriori_bruteforce.py**  
  Enumerates all nonempty itemsets, computes support, filters by minimum support, and generates association rules that meet a minimum confidence threshold.

- **kmeans_evaluation.py**  
  Generates synthetic clusters, runs KMeans, visualizes cluster assignments, and computes purity and entropy to evaluate clustering quality.

---

## Quickstart and installation

**Requirements**

- **Python** 3.8 or newer  
- **Core packages**  
  - `numpy`  
  - `pandas`  
  - `matplotlib`  
  - `scikit-learn`  
  - `scipy`

**Install dependencies**

```bash
python -m venv .venv
# Activate the virtual environment
# macOS Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Run examples**

- Multiple linear regression
```bash
python regression/multiple_regression_numpy.py
```

- Simple linear regression with scikit‑learn
```bash
python regression/simple_regression_sklearn.py
```

- Manual vs scikit‑learn comparison
```bash
python regression/comparison_manual_vs_sklearn.py
```

- Association rules brute force
```bash
python association_rules/apriori_bruteforce.py
```

- KMeans clustering evaluation
```bash
python clustering/kmeans_evaluation.py
```

---

## Usage notes and data expectations

**Data placement**

- Place your CSV dataset in `data/dataset.csv` or update the script file paths to point to your dataset.  
- Regression scripts expect at least two feature columns and one target column. Update column names in the scripts if your CSV uses different headers.

**Common pitfalls and fixes**

- **Curly quotes** Replace curly quotes with straight ASCII quotes to avoid syntax errors.  
- **Singular matrix** If `np.linalg.inv(X.T @ X)` fails, use the pseudo inverse `np.linalg.pinv` or add regularization.  
- **KMeans reproducibility** Set `n_init` explicitly for reproducible results, for example `KMeans(n_clusters=3, random_state=42, n_init=10)`.  
- **Scalability** The association rules script enumerates all subsets and is exponential in the number of unique items. Use Apriori or FP‑Growth for larger datasets.

**Outputs**

- Scripts print model parameters and metrics to the console and open Matplotlib plots for visual inspection. Save plots manually if needed.

---

## Extensions and ideas

- **Regression** Add evaluation metrics such as MSE and R², cross validation, polynomial features, and regularization (Ridge, Lasso).  
- **Association rules** Implement Apriori pruning or FP‑Growth for scalability, compute additional metrics such as lift and leverage, and export rules to CSV.  
- **Clustering** Add internal metrics such as silhouette score, adjusted Rand index, and visualize high dimensional data with PCA or t‑SNE.  
- **Notebooks** Convert scripts into Jupyter notebooks with step‑by‑step explanations and interactive plots.  
- **CLI and tests** Add a small command line interface to set thresholds and add unit tests for core functions.

---

## License and contribution

**License recommendation**  
This repository is suitable for the **MIT License**. Add a `LICENSE` file with the MIT text and your name and year.

**Contributing guidelines**

- Fork the repository and create a feature branch.  
- Open a pull request with a clear description of changes.  
- Include tests or example outputs for new functionality.  
- Keep changes focused and document any new dependencies in `requirements.txt`.

---

## Final tips

- Keep sample data small and version only tiny example files. For larger or proprietary datasets, provide instructions to download or generate data instead of committing it.  
- Use clear variable names and add inline comments when you adapt scripts for your own datasets.  
- If you want, I can generate ready‑to‑paste `requirements.txt`, `.gitignore`, and `LICENSE` files or produce cleaned, runnable versions of each script.

---

**Copy this README into your repository root as README.md and update file paths, author name, and license year as needed.**
