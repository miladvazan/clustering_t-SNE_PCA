# clustering_t-SNE_PCA
clustering using t-SNE (t-Distributed Stochastic Neighbor Embedding) and PCA

## clustering using t-SNE
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# Load the Iris dataset (or you can use your own dataset)
data = load_iris()
X = data.data
y = data.target

# Apply t-SNE to obtain a lower-dimensional representation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Perform clustering directly on the t-SNE embedding
k = 3  # Number of clusters
centroids = X_tsne[np.random.choice(X_tsne.shape[0], k, replace=False)]

# Assign each point to the closest centroid
labels = np.argmin(np.linalg.norm(X_tsne[:, np.newaxis] - centroids, axis=-1), axis=-1)

# Visualize the clustering results
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.title('t-SNE Clustering')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
```
## clustering using PCA
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform clustering directly on the PCA-transformed data
k = 3  # Number of clusters
centroids = X_pca[np.random.choice(X_pca.shape[0], k, replace=False)]

# Assign each point to the closest centroid
labels = np.argmin(np.linalg.norm(X_pca[:, np.newaxis] - centroids, axis=-1), axis=-1)

# Visualize the clustering results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('PCA Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
```
