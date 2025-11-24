import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy

data, labels = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

def calc_purity(y_true, y_pred):
    cont_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)

def calc_entropy(y_true, y_pred):
    cont_matrix = confusion_matrix(y_true, y_pred)
    cont_matrix = np.array(cont_matrix, dtype=float) / np.sum(cont_matrix)
    return np.sum(-entropy(cont_matrix, base=2, axis=0))

purity = calc_purity(labels, clusters)
cluster_entropy = calc_entropy(labels, clusters)

plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', edgecolors='k', s=50)
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(f'Purity of clustering: {purity}')
print(f'Entropy of clustering: {cluster_entropy}')
