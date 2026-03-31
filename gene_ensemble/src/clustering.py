import numpy as np
from sklearn.metrics import silhouette_score

# ============================================
# 1. K-MEANS 
# ============================================
def run_kmeans(X, k=2, max_iter=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels


# ============================================
# 2. DBSCAN 
# ============================================
def run_dbscan(X, eps=0.5, min_samples=3):
    n = len(X)
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def region_query(point):
        return np.where(np.linalg.norm(X - point, axis=1) <= eps)[0]

    def expand_cluster(i, neighbors):
        labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            pt = neighbors[j]

            if not visited[pt]:
                visited[pt] = True
                new_neighbors = region_query(X[pt])

                if len(new_neighbors) >= min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            if labels[pt] == -1:
                labels[pt] = cluster_id

            j += 1

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = region_query(X[i])

        if len(neighbors) >= min_samples:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels


# ============================================
# 3. AGGLOMERATIVE 
# ============================================
def run_agglomerative(X, k=2):
    clusters = [[i] for i in range(len(X))]

    def cluster_distance(c1, c2):
        return min(
            np.linalg.norm(X[i] - X[j])
            for i in c1 for j in c2
        )

    while len(clusters) > k:
        min_dist = float('inf')
        merge_pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance(clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)

        i, j = merge_pair
        clusters[i] += clusters[j]
        del clusters[j]

    labels = np.zeros(len(X))
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            labels[point] = idx

    return labels.astype(int)


# ============================================
# 4. FUZZY C-MEANS
# ============================================
def fuzzy_c_means(X, c=2, m=2, max_iter=300):
    n = len(X)
    U = np.random.dirichlet(np.ones(c) * 20, size=n)

    for _ in range(max_iter):
        centers = (U.T @ X) / np.sum(U.T, axis=1)[:, None]
        dist = np.linalg.norm(X[:, None] - centers, axis=2) + 1e-6

        U = 1 / (dist ** (2 / (m - 1)))
        U = U / np.sum(U, axis=1, keepdims=True)

    return np.argmax(U, axis=1)


# ============================================
# 5. BEST K SELECTION
# ============================================
def find_best_k(X, k_range=(2,6)):
    print("\nFinding best K...")

    best_k = 2
    best_score = -1

    for k in range(k_range[0], k_range[1]+1):
        labels = run_kmeans(X, k)
        score = silhouette_score(X, labels)

        print(f"K={k}, Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Best K selected: {best_k}")
    return best_k
