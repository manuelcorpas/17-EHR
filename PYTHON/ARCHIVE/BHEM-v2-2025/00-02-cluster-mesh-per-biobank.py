import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import cosine_similarity

try:
    import networkx as nx
except ImportError:
    print("NetworkX not found, skipping network visualization")
    has_networkx = False
else:
    has_networkx = True
from collections import Counter

# Setup
input_file = "DATA/00-00-ehr_biobank_articles.csv"
output_dir = "ANALYSIS/00-02-THEMATIC-ANALYSIS-BIOBANKS"
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading and preprocessing data...")
df = pd.read_csv(input_file)
df = df[df["Year"] <= 2024]
df = df.dropna(subset=["Biobank", "MeSH Terms"])

# Expand Biobank and MeSH columns
df["Biobank"] = df["Biobank"].str.split(";")
df["MeSH Terms"] = df["MeSH Terms"].str.split(";")
df = df.explode("Biobank").explode("MeSH Terms")
df["Biobank"] = df["Biobank"].str.strip()
df["MeSH Terms"] = df["MeSH Terms"].str.strip().str.lower()

# Get all biobanks with at least 5 papers
biobank_counts = df["Biobank"].value_counts()
biobanks = biobank_counts[biobank_counts >= 5].index.tolist()

# ------ PART 1: IDENTIFY THEMES PER BIOBANK USING TOPIC MODELING ------

# Dictionary to store themes for each biobank
biobank_themes = {}
biobank_mesh_doc = {}

# Number of themes to extract per biobank
n_themes = 5

for biobank in biobanks:
    print(f"\n🔍 Identifying themes for: {biobank}")
    biobank_df = df[df["Biobank"] == biobank]
    
    # Create document-term matrix from MeSH terms
    mesh_by_paper = biobank_df.groupby("PMID")["MeSH Terms"].apply(lambda x: " ".join(set(x))).reset_index()
    documents = mesh_by_paper["MeSH Terms"].tolist()
    biobank_mesh_doc[biobank] = documents
    
    # Skip if too few documents
    if len(documents) < 5:
        print(f"Skipping {biobank}: insufficient data ({len(documents)} documents)")
        continue
    
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    # Skip if matrix is too small
    if tfidf_matrix.shape[1] < n_themes:
        print(f"Skipping {biobank}: insufficient features ({tfidf_matrix.shape[1]} terms)")
        continue
    
    # Topic modeling with NMF (Non-negative Matrix Factorization)
    nmf_model = NMF(n_components=n_themes, random_state=42)
    nmf_model.fit(tfidf_matrix)
    
    # Get top terms for each theme
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Store themes
    themes = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_terms_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 terms
        top_terms = [feature_names[i] for i in top_terms_idx]
        themes.append(top_terms)
    
    biobank_themes[biobank] = themes
    
    # Save themes to CSV
    themes_df = pd.DataFrame(themes, columns=[f"Term_{i+1}" for i in range(10)])
    themes_df.index.name = "Theme"
    themes_df.to_csv(os.path.join(output_dir, f"{biobank}_themes.csv"))
    
    # Visualize top terms for each theme
    plt.figure(figsize=(15, 10))
    for i, theme in enumerate(themes):
        plt.subplot(n_themes, 1, i+1)
        y_pos = np.arange(len(theme))
        plt.barh(y_pos, [nmf_model.components_[i][topic.argsort()[:-11:-1][j]] for j in range(10)], align='center')
        plt.yticks(y_pos, theme)
        plt.title(f"Theme {i+1}")
    
    plt.tight_layout()
    plt.suptitle(f"Top Themes for {biobank}", y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, f"{biobank}_themes_visualization.png"), bbox_inches='tight')
    plt.close()

# ------ PART 2: COMPARE THEMES BETWEEN BIOBANKS ------

print("\n🔍 Comparing themes between biobanks...")

# Method 1: Cross-biobank theme similarity matrix using overall MeSH term distributions
biobank_vectors = {}

for biobank in biobank_themes.keys():
    # Flatten all themes into one list and count term frequencies
    all_terms = [term for theme in biobank_themes[biobank] for term in theme]
    term_counter = Counter(all_terms)
    biobank_vectors[biobank] = term_counter

# Create similarity matrix
similarity_matrix = pd.DataFrame(index=biobank_vectors.keys(), columns=biobank_vectors.keys(), dtype=float)

for b1 in biobank_vectors.keys():
    for b2 in biobank_vectors.keys():
        # Calculate cosine similarity between theme term distributions
        v1 = np.array([biobank_vectors[b1].get(term, 0) for term in set(biobank_vectors[b1].keys()) | set(biobank_vectors[b2].keys())])
        v2 = np.array([biobank_vectors[b2].get(term, 0) for term in set(biobank_vectors[b1].keys()) | set(biobank_vectors[b2].keys())])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            cos_sim = np.dot(v1, v2) / (v1_norm * v2_norm)
        else:
            cos_sim = 0
            
        similarity_matrix.loc[b1, b2] = cos_sim

# Save similarity matrix
similarity_matrix.to_csv(os.path.join(output_dir, "biobank_theme_similarity_matrix.csv"))

# Visualize similarity matrix as heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Theme Similarity Between Biobanks", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "biobank_theme_similarity_heatmap.png"))
plt.close()

# Method 2: Hierarchical clustering of biobanks based on theme similarity
# Convert similarity matrix to distance matrix (1 - similarity)
distance_matrix = 1 - similarity_matrix

# Perform hierarchical clustering
# Use scipy's linkage directly instead of AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
Z = linkage(distance_matrix.values, method='average')
# We'll use the linkage matrix directly for the dendrogram

# Plot dendrogram
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(12, 8))
# Z is already defined above
dendrogram(
    Z,
    labels=distance_matrix.index,
    orientation='right',
    leaf_font_size=12
)
plt.title("Hierarchical Clustering of Biobanks by Research Themes", fontsize=16)
plt.xlabel("Distance (Theme Dissimilarity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "biobank_theme_dendrogram.png"))
plt.close()

# Method 3: Network visualization of biobank theme relationships
if has_networkx:
    try:
        G = nx.Graph()
        
        # Add nodes (biobanks)
        for biobank in similarity_matrix.index:
            G.add_node(biobank)
        
        # Add edges with weight based on similarity (only if above threshold)
        threshold = 0.3
        for b1 in similarity_matrix.index:
            for b2 in similarity_matrix.columns:
                if b1 != b2 and similarity_matrix.loc[b1, b2] > threshold:
                    G.add_edge(b1, b2, weight=similarity_matrix.loc[b1, b2])
        
        # Create positions using spring layout (based on weights)
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
        
        # Plot
        plt.figure(figsize=(14, 12))
        node_sizes = [biobank_counts.get(b, 10) * 10 for b in G.nodes()]  # Size by article count, with fallback
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color="lightgray")
        
        # Draw nodes with labels
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        
        plt.title("Network of Biobanks with Similar Research Themes", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "biobank_theme_network.png"))
        plt.close()
    except Exception as e:
        print(f"Error in network visualization: {e}")
        print("Skipping network visualization")
else:
    print("Skipping network visualization (NetworkX not available)")

# ------ PART 3: GENERATE A SUMMARY REPORT ------

print("\n📊 Generating summary report...")

# Get top themes for each biobank
top_themes_summary = {}
for biobank in biobank_themes:
    # For each theme, join the first 5 terms with commas
    theme_descriptions = [", ".join(theme[:5]) for theme in biobank_themes[biobank]]
    top_themes_summary[biobank] = theme_descriptions

# Create summary dataframe
summary_df = pd.DataFrame.from_dict(top_themes_summary, orient='index')
summary_df.columns = [f"Theme {i+1}" for i in range(summary_df.shape[1])]
summary_df.index.name = "Biobank"
summary_df.to_csv(os.path.join(output_dir, "biobank_theme_summary.csv"))

# Calculate theme uniqueness score (inverse of average similarity to other biobanks)
uniqueness_scores = {}
for biobank in similarity_matrix.index:
    # Average similarity to all other biobanks
    avg_similarity = (similarity_matrix.loc[biobank].sum() - 1) / (len(similarity_matrix) - 1)
    uniqueness_scores[biobank] = 1 - avg_similarity

# Create uniqueness dataframe and sort by uniqueness
uniqueness_df = pd.DataFrame.from_dict(uniqueness_scores, orient='index', columns=["Uniqueness Score"])
uniqueness_df = uniqueness_df.sort_values("Uniqueness Score", ascending=False)
uniqueness_df.index.name = "Biobank"
uniqueness_df.to_csv(os.path.join(output_dir, "biobank_theme_uniqueness.csv"))

# Visualize uniqueness scores
plt.figure(figsize=(12, 8))
sns.barplot(x=uniqueness_df.index, y="Uniqueness Score", data=uniqueness_df)
plt.xticks(rotation=90)
plt.title("Thematic Uniqueness of Biobanks", fontsize=16)
plt.ylabel("Uniqueness Score (0-1)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "biobank_theme_uniqueness.png"))
plt.close()

print("✅ Completed thematic analysis and comparison for all biobanks.")
print(f"Results saved to: {output_dir}")