import sys, os, importlib

# Fix path
sys.path.insert(0, os.path.abspath("../src"))

# Import modules
import clustering
import preprocessing
import ensemble
from evaluation import evaluate

# Reload clustering (important)
importlib.reload(clustering)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ============================================
# LOAD DATA
# ============================================
X, y = preprocessing.load_data("../data/data_set_ALL_AML_train.csv")
X_scaled = preprocessing.preprocess(X)

print("Label distribution:", np.unique(y, return_counts=True))


# ============================================
# PCA
# ============================================
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)


# ============================================
# BEST K
# ============================================
from sklearn.metrics import silhouette_score

def find_best_k(X, k_range=(2,6)):
    best_k, best_score = 2, -1
    for k in range(k_range[0], k_range[1]+1):
        labels = clustering.run_kmeans(X, k)
        score = silhouette_score(X, labels)
        print(f"K={k}, Silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    print("Best K:", best_k)
    return best_k

best_k = find_best_k(X_reduced)


# ============================================
# CLUSTERING
# ============================================
kmeans_labels = clustering.run_kmeans(X_reduced, best_k)
dbscan_labels = clustering.run_dbscan(X_reduced)
fuzzy_labels = clustering.fuzzy_c_means(X_reduced)
agg_labels = clustering.run_agglomerative(X_reduced)


# ============================================
# DEBUG
# ============================================
print("\nCluster diversity:")
print("KMeans:", np.unique(kmeans_labels))
print("DBSCAN:", np.unique(dbscan_labels))
print("Fuzzy:", np.unique(fuzzy_labels))
print("Agg:", np.unique(agg_labels))


# ============================================
# ENSEMBLE
# ============================================
ensemble_labels = ensemble.fuzzy_ensemble([
    kmeans_labels,
    fuzzy_labels,
    agg_labels
])


# ============================================
# EVALUATION (WITH ARI)
# ============================================
evaluate(X_reduced, y, kmeans_labels, "KMeans")
evaluate(X_reduced, y, dbscan_labels, "DBSCAN")
evaluate(X_reduced, y, fuzzy_labels, "Fuzzy")
evaluate(X_reduced, y, agg_labels, "Agglomerative")
evaluate(X_reduced, y, ensemble_labels, "Ensemble")


# ============================================
# CLASSIFIER VALIDATION
# ============================================
print("\n=== CLASSIFIER VALIDATION ===")

def evaluate_classifier(X, labels, name):
    print(f"\n{name}")

    if len(set(labels)) < 2:
        print("Skipping ❌")
        return

    print("RF:", round(cross_val_score(RandomForestClassifier(), X, labels, cv=5).mean(),4))
    print("SVM:", round(cross_val_score(SVC(), X, labels, cv=5).mean(),4))
    print("KNN:", round(cross_val_score(KNeighborsClassifier(3), X, labels, cv=5).mean(),4))


evaluate_classifier(X_reduced, kmeans_labels, "KMeans")
evaluate_classifier(X_reduced, agg_labels, "Agglomerative")
evaluate_classifier(X_reduced, ensemble_labels, "Ensemble")


# ============================================
# SAVE RESULTS
# ============================================
os.makedirs("results", exist_ok=True)


# ============================================
# PCA PLOT
# ============================================
plt.figure()
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=ensemble_labels)
plt.title("PCA - Ensemble")
plt.savefig("results/pca_plot.png")
plt.close()


# ============================================
# TSNE 
# ============================================
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=ensemble_labels)
plt.title("t-SNE - Ensemble")
plt.savefig("results/tsne_plot.png")
plt.close()


# ============================================
# SILHOUETTE
# ============================================
if len(set(ensemble_labels)) > 1:
    sil_vals = silhouette_samples(X_reduced, ensemble_labels)

    plt.figure()
    plt.hist(sil_vals, bins=20)
    plt.title("Silhouette Distribution")
    plt.savefig("results/silhouette.png")
    plt.close()


# ============================================
# HEATMAP
# ============================================
plt.figure(figsize=(8,6))
sns.heatmap(X_scaled[:50], cmap="viridis")
plt.title("Heatmap")
plt.savefig("results/heatmap.png")
plt.close()


# ============================================
# ABLATION
# ============================================
def safe_score(X, labels):
    if len(set(labels)) < 2:
        return 0
    return silhouette_score(X, labels)

ablation = {
    "KMeans": safe_score(X_reduced, kmeans_labels),
    "DBSCAN": safe_score(X_reduced, dbscan_labels),
    "Fuzzy": safe_score(X_reduced, fuzzy_labels),
    "Agg": safe_score(X_reduced, agg_labels),
    "Ensemble": safe_score(X_reduced, ensemble_labels),
}

print("\nAblation:", ablation)

plt.figure()
plt.bar(ablation.keys(), ablation.values())
plt.title("Ablation")
plt.savefig("results/ablation.png")
plt.close()


# ============================================
# ALL CLUSTERS VISUAL
# ============================================
plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=kmeans_labels)
plt.title("KMeans")

plt.subplot(2,3,2)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=dbscan_labels)
plt.title("DBSCAN")

plt.subplot(2,3,3)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=fuzzy_labels)
plt.title("Fuzzy")

plt.subplot(2,3,4)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=agg_labels)
plt.title("Agglomerative")

plt.subplot(2,3,5)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=ensemble_labels)
plt.title("Ensemble ⭐")

plt.tight_layout()
plt.savefig("results/all_clusters.png")
plt.close()

print("\nALL DONE - Results saved in /results")