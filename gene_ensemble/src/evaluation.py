from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

print("evaluation module loaded ")


def evaluate(X, y_true, labels, name):
    print(f"\n{name}")

    # Handle DBSCAN noise (-1)
    if -1 in labels:
        mask = labels != -1
        labels = labels[mask]
        X = X[mask]

        if y_true is not None:
            y_true = y_true[mask]

    # Check valid clustering
    if len(set(labels)) < 2:
        print("Only one cluster ")
        return

    # Metrics
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)

    print("Silhouette:", round(sil, 4))
    print("DB Index:", round(dbi, 4))

    # ARI (only if true labels available)
    if y_true is not None and len(set(y_true)) > 1:
        ari = adjusted_rand_score(y_true, labels)
        print("ARI:", round(ari, 4))
    else:
        print("ARI: Not valid (single class or missing labels)")
