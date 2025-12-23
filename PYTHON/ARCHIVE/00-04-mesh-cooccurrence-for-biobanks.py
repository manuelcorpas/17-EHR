#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MeSH Co-occurrence Analysis and Clustering for Biobanks

This script analyzes MeSH term co-occurrences across different biobanks to identify
predominant research themes and enable comparison between biobanks. It uses an LLM-inspired
approach similar to the reference code but adapted specifically for MeSH terms.

Usage:
  python3 mesh_cooccurrence_analysis.py --input_file DATA/00-00-ehr_biobank_articles.csv
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import torch
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
import networkx as nx
from itertools import combinations

#############################################################################
# Data Loading & Preprocessing
#############################################################################

def load_biobank_mesh_data(input_file, min_articles=5):
    """
    Load and preprocess biobank data with MeSH terms
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df = df[df["Year"] <= 2024]
    df = df.dropna(subset=["Biobank", "MeSH Terms"])
    
    # Expand Biobank and MeSH columns
    df["Biobank"] = df["Biobank"].str.split(";")
    df["MeSH Terms"] = df["MeSH Terms"].str.split(";")
    df = df.explode("Biobank").explode("MeSH Terms")
    df["Biobank"] = df["Biobank"].str.strip()
    df["MeSH Terms"] = df["MeSH Terms"].str.strip().str.lower()
    
    # Filter biobanks with minimum number of articles
    biobank_counts = df["Biobank"].value_counts()
    valid_biobanks = biobank_counts[biobank_counts >= min_articles].index.tolist()
    df = df[df["Biobank"].isin(valid_biobanks)]
    
    print(f"Preprocessed data contains {df['PMID'].nunique()} articles across {len(valid_biobanks)} biobanks")
    return df, valid_biobanks

#############################################################################
# Co-occurrence Matrix Generation
#############################################################################

def generate_cooccurrence_matrices(df, biobanks):
    """
    Generate co-occurrence matrices for MeSH terms, both overall and per biobank
    """
    # Get all unique MeSH terms
    all_mesh_terms = sorted(df["MeSH Terms"].unique())
    n_terms = len(all_mesh_terms)
    term_to_idx = {term: i for i, term in enumerate(all_mesh_terms)}
    
    print(f"Building co-occurrence matrices for {n_terms} unique MeSH terms...")
    
    # Create overall co-occurrence matrix
    overall_matrix = np.zeros((n_terms, n_terms))
    
    # Create per-biobank matrices
    biobank_matrices = {biobank: np.zeros((n_terms, n_terms)) for biobank in biobanks}
    
    # Group MeSH terms by PMID
    pmid_meshes = df.groupby("PMID")["MeSH Terms"].apply(list).to_dict()
    pmid_biobanks = df.groupby("PMID")["Biobank"].apply(lambda x: list(set(x))).to_dict()
    
    # Compute co-occurrences
    for pmid, mesh_terms in tqdm(pmid_meshes.items(), desc="Computing co-occurrences"):
        # Ensure unique terms per article
        mesh_terms = list(set(mesh_terms))
        if len(mesh_terms) < 2:
            continue
            
        # Update co-occurrence counts in overall matrix
        for i, term1 in enumerate(mesh_terms):
            idx1 = term_to_idx[term1]
            for term2 in mesh_terms[i+1:]:
                idx2 = term_to_idx[term2]
                overall_matrix[idx1, idx2] += 1
                overall_matrix[idx2, idx1] += 1  # Mirror the matrix
        
        # Update biobank-specific matrices
        article_biobanks = pmid_biobanks.get(pmid, [])
        for biobank in article_biobanks:
            if biobank in biobank_matrices:
                for i, term1 in enumerate(mesh_terms):
                    idx1 = term_to_idx[term1]
                    for term2 in mesh_terms[i+1:]:
                        idx2 = term_to_idx[term2]
                        biobank_matrices[biobank][idx1, idx2] += 1
                        biobank_matrices[biobank][idx2, idx1] += 1
    
    return overall_matrix, biobank_matrices, all_mesh_terms, term_to_idx

#############################################################################
# TF-IDF Normalization for Co-occurrence
#############################################################################

def compute_tfipf_matrices(overall_matrix, biobank_matrices, all_mesh_terms):
    """
    Convert raw co-occurrence matrices to TF-IPF (Term Frequency-Inverse Paper Frequency)
    """
    n_terms = len(all_mesh_terms)
    n_biobanks = len(biobank_matrices)
    
    # Calculate term document frequency (in how many biobanks each term appears)
    term_biobank_freq = np.zeros(n_terms)
    for biobank, matrix in biobank_matrices.items():
        # If a term has any co-occurrence in this biobank, count it
        term_has_cooc = (np.sum(matrix, axis=1) > 0).astype(int)
        term_biobank_freq += term_has_cooc
    
    # Calculate IPF (Inverse Paper Frequency)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    ipf = np.log((n_biobanks + epsilon) / (term_biobank_freq + epsilon))
    
    # Convert to TF-IPF matrices
    tfipf_matrices = {}
    for biobank, matrix in biobank_matrices.items():
        # Get TF (normalize co-occurrences within this biobank)
        row_sums = np.sum(matrix, axis=1)
        tf = np.zeros_like(matrix)
        for i in range(n_terms):
            if row_sums[i] > 0:
                tf[i, :] = matrix[i, :] / row_sums[i]
        
        # Calculate TF-IPF
        tfipf = np.zeros_like(tf)
        for i in range(n_terms):
            tfipf[i, :] = tf[i, :] * ipf
        
        tfipf_matrices[biobank] = tfipf
    
    return tfipf_matrices

#############################################################################
# Theme Extraction
#############################################################################

def extract_themes(tfipf_matrices, all_mesh_terms, n_themes=5):
    """
    Extract dominant themes from TF-IPF matrices using NMF
    """
    biobank_themes = {}
    
    for biobank, matrix in tfipf_matrices.items():
        print(f"\nExtracting themes for: {biobank}")
        
        # SVD for dimensionality reduction before NMF
        svd = TruncatedSVD(n_components=min(50, matrix.shape[1]-1))
        X_reduced = svd.fit_transform(matrix)
        
        # Ensure non-negativity
        X_reduced = np.maximum(X_reduced, 0)
        
        # Apply NMF for theme extraction
        try:
            nmf = NMF(n_components=n_themes, random_state=42, max_iter=500)
            W = nmf.fit_transform(X_reduced)  # article-theme matrix
            H = nmf.components_  # theme-term matrix
            
            # Get top terms for each theme
            themes = []
            for i in range(n_themes):
                theme_vector = np.zeros(len(all_mesh_terms))
                for j in range(X_reduced.shape[1]):
                    theme_vector += H[i, j] * svd.components_[j, :]
                
                # Get top 10 terms
                top_indices = np.argsort(-theme_vector)[:10]
                top_terms = [all_mesh_terms[idx] for idx in top_indices]
                themes.append(top_terms)
            
            biobank_themes[biobank] = themes
            
            # Print top 3 terms for each theme
            print(f"  Top themes for {biobank}:")
            for i, theme in enumerate(themes):
                print(f"    Theme {i+1}: {', '.join(theme[:3])}")
                
        except Exception as e:
            print(f"Error extracting themes for {biobank}: {e}")
            continue
    
    return biobank_themes

#############################################################################
# Theme Similarity and Clustering
#############################################################################

def compute_theme_similarity(biobank_themes):
    """
    Compute similarity between biobanks based on their themes
    """
    biobanks = list(biobank_themes.keys())
    n_biobanks = len(biobanks)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n_biobanks, n_biobanks))
    
    # Function to calculate Jaccard similarity between two term lists
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    # Calculate similarity
    for i, bank1 in enumerate(biobanks):
        for j, bank2 in enumerate(biobanks):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
                
            # Flatten themes into term lists
            terms1 = [term for theme in biobank_themes[bank1] for term in theme]
            terms2 = [term for theme in biobank_themes[bank2] for term in theme]
            
            # Calculate Jaccard similarity
            similarity_matrix[i, j] = jaccard_similarity(terms1, terms2)
    
    return pd.DataFrame(similarity_matrix, index=biobanks, columns=biobanks)

def cluster_biobanks(similarity_matrix):
    """
    Perform hierarchical clustering on biobanks based on theme similarity
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    Z = linkage(squareform(distance_matrix), method='average')
    
    # Determine optimal number of clusters
    silhouette_scores = []
    for n_clusters in range(2, min(10, len(similarity_matrix)+1)):
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        if len(np.unique(labels)) > 1:  # Ensure we have multiple clusters
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores.append((n_clusters, score))
    
    # Get optimal number of clusters
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else 2
    
    # Get cluster assignments
    labels = fcluster(Z, optimal_n_clusters, criterion='maxclust')
    
    return Z, labels, optimal_n_clusters

#############################################################################
# Visualization Functions
#############################################################################

def visualize_cooccurrence_network(matrix, mesh_terms, biobank_name, top_n=100, output_dir="ANALYSIS/MESH-COOCCURRENCE"):
    """
    Create a network visualization of the most important co-occurrences
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top co-occurrences
    flat_indices = np.argsort(matrix.flatten())[-top_n:]
    row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges for top co-occurrences
    for i, j in zip(row_indices, col_indices):
        if i != j:  # Skip self-loops
            term1 = mesh_terms[i]
            term2 = mesh_terms[j]
            weight = matrix[i, j]
            if weight > 0:
                G.add_edge(term1, term2, weight=weight)
    
    # Calculate node importance (sum of edge weights)
    node_importance = {}
    for node in G.nodes():
        node_importance[node] = sum(G[node][neighbor]['weight'] for neighbor in G[node])
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Create positions using Fruchterman-Reingold layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw edges with varying thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.3, edge_color="gray")
    
    # Draw nodes with varying size based on importance
    node_sizes = [node_importance[node] * 100 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    
    # Draw labels only for important nodes
    important_nodes = {node: node_importance[node] for node in G.nodes()}
    top_nodes = sorted(important_nodes.items(), key=lambda x: x[1], reverse=True)[:25]
    labels = {node: node for node, _ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")
    
    plt.title(f"Top MeSH Co-occurrences for {biobank_name}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{biobank_name.replace(' ', '_')}_cooccurrence_network.png"), dpi=300)
    plt.close()

def visualize_all_results(biobank_themes, similarity_matrix, Z, labels, output_dir="ANALYSIS/MESH-COOCCURRENCE"):
    """
    Create visualizations of themes and clustering results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Theme heatmap for each biobank
    for biobank, themes in biobank_themes.items():
        plt.figure(figsize=(14, 10))
        theme_matrix = np.zeros((len(themes), 10))
        theme_labels = []
        term_labels = []
        
        for i, theme in enumerate(themes):
            theme_labels.append(f"Theme {i+1}")
            for j, term in enumerate(theme[:10]):
                if j >= len(term_labels):
                    term_labels.append(term)
                theme_matrix[i, j] = 10 - j  # Higher value for more important terms
        
        sns.heatmap(theme_matrix, annot=False, cmap="YlGnBu", 
                    xticklabels=term_labels, yticklabels=theme_labels)
        plt.title(f"Themes for {biobank}", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{biobank.replace(' ', '_')}_themes.png"))
        plt.close()
    
    # 2. Similarity heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Theme Similarity Between Biobanks", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "biobank_theme_similarity_heatmap.png"))
    plt.close()
    
    # 3. Dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(
        Z,
        labels=similarity_matrix.index,
        orientation='right',
        leaf_font_size=12
    )
    plt.title("Hierarchical Clustering of Biobanks by Research Themes", fontsize=16)
    plt.xlabel("Distance (Theme Dissimilarity)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "biobank_theme_dendrogram.png"))
    plt.close()
    
    # 4. Network visualization of biobank similarity
    plt.figure(figsize=(14, 12))
    G = nx.Graph()
    
    # Add nodes (biobanks)
    biobanks = similarity_matrix.index.tolist()
    for i, biobank in enumerate(biobanks):
        G.add_node(biobank, cluster=labels[i])
    
    # Add edges with weight based on similarity (only if above threshold)
    threshold = 0.3
    for i, bank1 in enumerate(biobanks):
        for j, bank2 in enumerate(biobanks):
            if i != j and similarity_matrix.iloc[i, j] > threshold:
                G.add_edge(bank1, bank2, weight=similarity_matrix.iloc[i, j])
    
    # Create positions using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Draw edges with varying thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color="lightgray")
    
    # Draw nodes colored by cluster
    cluster_colors = plt.cm.rainbow(np.linspace(0, 1, max(labels)))
    node_colors = [cluster_colors[label-1] for label in labels]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    plt.title("Network of Biobanks with Similar Research Themes", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "biobank_theme_network.png"))
    plt.close()
    
    # 5. UMAP visualization of biobanks
    try:
        # Convert similarity matrix to distance matrix
        dist_matrix = 1 - similarity_matrix.values
        
        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, metric='precomputed', random_state=42)
        embedding = reducer.fit_transform(dist_matrix)
        
        # Plot
        plt.figure(figsize=(12, 10))
        for i, label in enumerate(np.unique(labels)):
            mask = (labels == label)
            plt.scatter(
                embedding[mask, 0], 
                embedding[mask, 1], 
                s=200, 
                c=[cluster_colors[label-1]], 
                label=f"Cluster {label}"
            )
        
        # Add labels for each point
        for i, biobank in enumerate(biobanks):
            plt.annotate(biobank, (embedding[i, 0], embedding[i, 1]), fontsize=8)
        
        plt.title("UMAP Projection of Biobanks by Theme Similarity", fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "biobank_theme_umap.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating UMAP visualization: {e}")

#############################################################################
# Reporting Functions
#############################################################################

def generate_summary_report(biobank_themes, labels, output_dir="ANALYSIS/MESH-COOCCURRENCE"):
    """
    Generate summary reports of themes and clusters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create theme summary CSV
    theme_rows = []
    for biobank, themes in biobank_themes.items():
        for i, theme in enumerate(themes):
            theme_rows.append({
                'Biobank': biobank,
                'Theme': i+1,
                'Top Terms': ', '.join(theme[:5]),
                'All Terms': ', '.join(theme)
            })
    
    theme_df = pd.DataFrame(theme_rows)
    theme_df.to_csv(os.path.join(output_dir, "biobank_theme_summary.csv"), index=False)
    
    # 2. Create cluster report
    biobanks = list(biobank_themes.keys())
    cluster_df = pd.DataFrame({
        'Biobank': biobanks,
        'Cluster': labels
    })
    
    # Group by cluster
    cluster_report = []
    for cluster in sorted(set(labels)):
        cluster_biobanks = cluster_df[cluster_df['Cluster'] == cluster]['Biobank'].tolist()
        cluster_report.append({
            'Cluster': cluster,
            'Size': len(cluster_biobanks),
            'Biobanks': ', '.join(cluster_biobanks)
        })
    
    cluster_summary = pd.DataFrame(cluster_report)
    cluster_summary.to_csv(os.path.join(output_dir, "biobank_cluster_summary.csv"), index=False)
    
    # 3. Create a textual summary
    with open(os.path.join(output_dir, "theme_analysis_summary.txt"), 'w') as f:
        f.write("# MeSH Co-occurrence Theme Analysis Summary\n\n")
        
        f.write("## Biobank Clusters\n\n")
        for row in cluster_report:
            f.write(f"### Cluster {row['Cluster']} ({row['Size']} biobanks)\n")
            f.write(f"Biobanks: {row['Biobanks']}\n\n")
            
            # Find common themes in this cluster
            cluster_biobanks = row['Biobanks'].split(', ')
            all_themes = []
            for biobank in cluster_biobanks:
                if biobank in biobank_themes:
                    themes = biobank_themes[biobank]
                    for i, theme in enumerate(themes):
                        all_themes.append(set(theme[:5]))  # Top 5 terms
            
            # Find terms that appear frequently in this cluster
            if all_themes:
                term_counts = Counter()
                for theme_set in all_themes:
                    for term in theme_set:
                        term_counts[term] += 1
                
                common_terms = [term for term, count in term_counts.most_common(10)]
                f.write(f"Common research themes: {', '.join(common_terms)}\n\n")
        
        f.write("\n## Biobank-Specific Themes\n\n")
        for biobank, themes in biobank_themes.items():
            f.write(f"### {biobank}\n")
            for i, theme in enumerate(themes):
                f.write(f"Theme {i+1}: {', '.join(theme[:5])}\n")
            f.write("\n")

#############################################################################
# Main Script
#############################################################################

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='MeSH Co-occurrence Analysis and Clustering for Biobanks')
    parser.add_argument('--input_file', default='DATA/00-00-ehr_biobank_articles.csv', help='Input CSV file with biobank and MeSH data')
    parser.add_argument('--output_dir', default='ANALYSIS/MESH-COOCCURRENCE', help='Output directory for results')
    parser.add_argument('--min_articles', type=int, default=5, help='Minimum articles for a biobank to be included')
    parser.add_argument('--n_themes', type=int, default=5, help='Number of themes to extract per biobank')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    df, biobanks = load_biobank_mesh_data(args.input_file, args.min_articles)
    
    # Step 2: Generate co-occurrence matrices
    overall_matrix, biobank_matrices, all_mesh_terms, term_to_idx = generate_cooccurrence_matrices(df, biobanks)
    
    # Step 3: Compute TF-IPF matrices
    tfipf_matrices = compute_tfipf_matrices(overall_matrix, biobank_matrices, all_mesh_terms)
    
    # Step 4: Create network visualizations
    print("Creating co-occurrence network visualizations...")
    for biobank in biobanks:
        visualize_cooccurrence_network(
            biobank_matrices[biobank], 
            all_mesh_terms, 
            biobank, 
            output_dir=args.output_dir
        )
    
    # Step 5: Extract themes
    print("Extracting themes from co-occurrence patterns...")
    biobank_themes = extract_themes(tfipf_matrices, all_mesh_terms, n_themes=args.n_themes)
    
    # Step 6: Compute theme similarity
    print("Computing similarity between biobank themes...")
    similarity_matrix = compute_theme_similarity(biobank_themes)
    
    # Step 7: Cluster biobanks
    print("Clustering biobanks based on theme similarity...")
    Z, labels, n_clusters = cluster_biobanks(similarity_matrix)
    print(f"Identified {n_clusters} clusters of biobanks with similar research themes")
    
    # Step 8: Visualize results
    print("Generating visualizations...")
    visualize_all_results(biobank_themes, similarity_matrix, Z, labels, output_dir=args.output_dir)
    
    # Step 9: Generate reports
    print("Creating summary reports...")
    generate_summary_report(biobank_themes, labels, output_dir=args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Found {n_clusters} distinct clusters of biobanks based on MeSH term co-occurrences")

if __name__ == "__main__":
    main()