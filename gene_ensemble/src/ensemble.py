import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


print(" ensemble loaded ")


def compute_weights(X, clusterings):
    """
    Automatically compute weights using silhouette score
    """
    weights = []
    
    for labels in clusterings:
        # Ignore noise (-1)
        mask = labels != -1
        
        if len(set(labels[mask])) < 2:
            weights.append(0.1)  # very low weight
            continue
        
        try:
            score = silhouette_score(X[mask], labels[mask])
            weights.append(max(score, 0.01))  # avoid zero weight
        except:
            weights.append(0.1)
    
    return np.array(weights)


def build_co_matrix(clusterings, weights):
    """
    Fast vectorized co-association matrix
    """
    n = len(clusterings[0])
    co_matrix = np.zeros((n, n))

    for w, labels in zip(weights, clusterings):
        valid = labels != -1
        
        # Broadcasting comparison (FAST ⚡)
        same_cluster = (labels[:, None] == labels[None, :])
        valid_mask = (valid[:, None] & valid[None, :])
        
        co_matrix += w * (same_cluster & valid_mask)

    return co_matrix


def fuzzy_ensemble(X, clusterings, n_clusters=2):
    """
    Improved ensemble clustering
    """
    
    # Step 1: Compute weights automatically
    weights = compute_weights(X, clusterings)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Step 2: Build co-association matrix
    co_matrix = build_co_matrix(clusterings, weights)
    
    # Step 3: Normalize matrix
    if np.max(co_matrix) > 0:
        co_matrix = co_matrix / np.max(co_matrix)
    
    # Step 4: Symmetrize
    co_matrix = (co_matrix + co_matrix.T) / 2
    
    # Step 5: Add tiny noise
    np.fill_diagonal(co_matrix, 1)
    co_matrix += 1e-6 * np.random.rand(*co_matrix.shape)
    
    # Step 6: Final clustering
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=50
    )
    
    final_labels = model.fit_predict(co_matrix)
    
    return final_labels
