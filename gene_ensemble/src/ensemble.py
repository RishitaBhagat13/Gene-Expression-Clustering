import numpy as np
from sklearn.cluster import KMeans

print("ensemble loaded (final improved)")   # debug


def fuzzy_ensemble(clusterings, n_clusters=2):
    n = len(clusterings[0])
    
    # Initialize co-association matrix
    co_matrix = np.zeros((n, n))
    
    # 🔥 Stronger weights (Agglomerative helps a lot)
    weights = np.array([1.0, 1.5, 0.8, 1.8])[:len(clusterings)]

    # Build weighted co-association matrix
    for w, labels in zip(weights, clusterings):
        for i in range(n):
            for j in range(n):
                # Ignore DBSCAN noise (-1)
                if labels[i] == -1 or labels[j] == -1:
                    continue
                
                if labels[i] == labels[j]:
                    co_matrix[i][j] += w

    # 🔥 Safe normalization (avoid division by zero)
    max_val = np.max(co_matrix)
    if max_val > 0:
        co_matrix = co_matrix / max_val

    # 🔥 Add symmetry (important)
    co_matrix = (co_matrix + co_matrix.T) / 2

    # 🔥 Add small noise for stability
    co_matrix += 1e-5 * np.random.rand(n, n)

    # 🔥 Final clustering (stronger KMeans)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    final_labels = model.fit_predict(co_matrix)

    return final_labels