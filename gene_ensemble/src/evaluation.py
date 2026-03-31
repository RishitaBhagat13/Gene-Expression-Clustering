from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

print("evaluation module loaded")   # debug

def evaluate(X, y_true, labels, name):
    print(f"\n{name}")
    
    # Check if clustering is valid
    if len(set(labels)) < 2:
        print("Only one cluster ❌")
        return
    
    # Metrics
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ari = adjusted_rand_score(y_true, labels)
    
    print("Silhouette:", round(sil, 4))
    print("DB Index:", round(db, 4))
    print("ARI:", round(ari, 4))