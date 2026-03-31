import numpy as np
from sklearn.cluster import KMeans

print("ensemble loaded ")   


def fuzzy_ensemble(clusterings, n_clusters=2):
    n = len(clusterings[0])
    
    # Initialize co-association matrix
    co_matrix = np.zeros((n, n))
    
    # weights
    weights = np.array([1.0, 1.5, 0.8, 1.8])[:len(clusterings)]

    # Build weighted co-association matrix
    for w, labels in zip(weights, clusterings):
        for i in range(n):
            for j in range(n):
                
                if labels[i] == -1 or labels[j] == -1:
                    continue
                
                if labels[i] == labels[j]:
                    co_matrix[i][j] += w

    
    max_val = np.max(co_matrix)
    if max_val > 0:
        co_matrix = co_matrix / max_val

    co_matrix = (co_matrix + co_matrix.T) / 2

    co_matrix += 1e-5 * np.random.rand(n, n)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    final_labels = model.fit_predict(co_matrix)

    return final_labels
