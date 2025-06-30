import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# --- Configuration ---
input_file = "DATA/global_ehr_biobank_results.csv"
output_dir = "ANALYSIS/SEMANTIC-CLUSTERS"
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
df = pd.read_csv(input_file)
df = df[df["MeSH Terms"].notnull()]
df = df[df["Year"].notnull()]
df["Year"] = df["Year"].astype(int)
df = df[df["Year"] <= 2024]

# --- Extract MeSH terms per article ---
df["MeSH Terms"] = df["MeSH Terms"].str.lower().str.strip()
df["MeSH List"] = df["MeSH Terms"].str.split(";")
df = df[df["MeSH List"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
df["MeSH List"] = df["MeSH List"].apply(lambda lst: [term.strip() for term in lst])

# --- Build term-document matrix ---
corpus = df["MeSH List"].tolist()
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, binary=True)
X = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()
pmids = df["PMID"].tolist()

# --- PCA dimensionality reduction ---
n_components = min(50, X.shape[1])
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X.toarray())

# --- Optimize K ---
best_k, best_score, best_labels = None, -1, None
for k in range(4, 16):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

print(f"✅ Optimal K: {best_k} (silhouette: {best_score:.3f})")

# --- Final clustering ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# --- Save clustered article metadata ---
df_out = df[["PMID", "Year"]].copy()
df_out["Cluster"] = clusters
df_out.to_csv(os.path.join(output_dir, "clustered_articles.csv"), index=False)

# --- Compute top terms per cluster ---
cluster_term_scores = []
X_bin = X.toarray()
for cluster_id in range(best_k):
    cluster_idx = np.where(clusters == cluster_id)[0]
    term_sums = X_bin[cluster_idx].sum(axis=0)
    df_total = X_bin.sum(axis=0)
    ipf = np.log((1 + X_bin.shape[0]) / (1 + df_total))  # inverse population freq
    scores = term_sums * ipf
    top_indices = np.argsort(scores)[::-1][:10]
    cluster_term_scores.append([terms[i] for i in top_indices])

with open(os.path.join(output_dir, "top_terms_by_cluster.csv"), "w") as f:
    f.write("Cluster,Top Terms\n")
    for i, terms_ in enumerate(cluster_term_scores):
        f.write(f"{i},\"{', '.join(terms_)}\"\n")

# --- PCA plot (cluster centroids only) ---
centroids = kmeans.cluster_centers_
centroid_labels = [f"Cluster {i}" for i in range(best_k)]

plt.figure(figsize=(10, 8))
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], hue=centroid_labels,
                palette="tab10", s=200, legend="full")
plt.title("Cluster Centroids (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_centroids_pca.png"))
plt.close()

# --- UMAP plot (cluster centroids only) ---
umap_coords = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(centroids)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=umap_coords[:, 0], y=umap_coords[:, 1], hue=centroid_labels,
                palette="tab10", s=200, legend="full")
plt.title("Cluster Centroids (UMAP Projection)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_centroids_umap.png"))
plt.close()

print("✅ Clustering with centroid visualization complete.")
print(f"📁 Outputs saved to: {output_dir}")
