def create_individual_biobank_plots(all_projection_data):
    """Create individual detailed plots for each biobank (for specific analysis)"""
    logger.info("Creating individual biobank plots for detailed analysis")
    
    for proj_data in all_projection_data:
        biobank_name = proj_data['biobank_name']
        pca_coords = proj_data['pca_coords']
        umap_coords = proj_data['umap_coords']
        cluster_sizes = proj_data['cluster_sizes']
        explained_var = proj_data['pca_explained_variance']
        
        # Create detailed PCA plot
        plt.figure(figsize=(12, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, len(pca_coords)))
        
        for i, (x, y) in enumerate(pca_coords):
            cluster_size = cluster_sizes[i]
            plt.scatter(x, y, c=[colors[i]], s=100 + cluster_size * 2, 
                       alpha=0.8, edgecolors='black', linewidth=1)
            plt.annotate(f'C{i}\n(n={cluster_size})', (x, y), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.title(f'PCA: MeSH Term Clusters within {biobank_name}\n'
                  f'({len(pca_coords)} semantic clusters from TF-IDF analysis)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.figtext(0.02, 0.02, 
                    f'Each point = 1 cluster of publications within {biobank_name}\n'
                    f'Point size ∝ number of publications in cluster\n'
                    f'Distance = semantic similarity of MeSH terms',
                    fontsize=9, style='italic')
        
        pca_file = os.path.join(analysis_dir, f'pca_clusters_{biobank_name.lower().replace(" ", "_")}.png')
        plt.savefig(pca_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed UMAP plot (if available)
        if umap_coords is not None:
            plt.figure(figsize=(12, 9))
            
            for i, (x, y) in enumerate(umap_coords):
                cluster_size = cluster_sizes[i]
                plt.scatter(x, y, c=[colors[i]], s=100 + cluster_size * 2, 
                           alpha=0.8, edgecolors='black', linewidth=1)
                plt.annotate(f'C{i}\n(n={cluster_size})', (x, y), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=10, ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            plt.title(f'UMAP: MeSH Term Clusters within {biobank_name}\n'
                      f'({len(umap_coords)} semantic clusters from TF-IDF analysis)', 
                      fontsize=14, fontweight='bold')
            plt.xlabel('UMAP1', fontsize=12)
            plt.ylabel('UMAP2', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.figtext(0.02, 0.02, 
                        f'Each point = 1 cluster of publications within {biobank_name}\n'
                        f'Point size ∝ number of publications in cluster\n'
                        f'Proximity = semantic similarity of MeSH terms',
                        fontsize=9, style='italic')
            
            umap_file = os.path.join(analysis_dir, f'umap_clusters_{biobank_name.lower().replace(" ", "_")}.png')
            plt.savefig(umap_file, dpi=300, bbox_inches='tight')
            plt.close()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BIOBANK MESH TERM CLUSTERING PIPELINE

Clusters biomedical publications from different biobanks based on their MeSH terms.
Identifies semantic clusters of publications within each biobank using TF-IDF and K-means.

PIPELINE:
1. Load and preprocess MeSH terms from biobank publications
2. **PER-BIOBANK ANALYSIS** (each biobank analyzed independently):
   a. Create TF-IDF matrix from that biobank's publications only
   b. Bootstrap optimal K selection (silhouette scoring)
   c. K-means clustering within that biobank
   d. c-DF-IPF scoring for top MeSH terms per cluster
   e. Semantic summaries using TF-IDF weights
   f. 2D projections (PCA/UMAP) of cluster centroids
3. Save biobank-specific results and visualizations

NOTE: Each biobank is analyzed separately because they represent different research 
communities with distinct publication patterns and MeSH term distributions.

INPUT CSV COLUMNS (from biobank data retrieval):
- Biobank: Which biobank the paper refers to
- PMID: PubMed ID
- MeSH_Terms: Medical Subject Headings (semicolon separated)

OUTPUT FILES:
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/clustering_results_<biobank>.csv: Publications with cluster assignments
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/cluster_summaries_<biobank>.csv: Top MeSH terms per cluster
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/biobank_clustering_summary.csv: Overall summary across all biobanks
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/composite_pca_all_biobanks.png: INTEGRATED PCA showing all biobanks
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/composite_umap_all_biobanks.png: INTEGRATED UMAP showing all biobanks
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/pca_clusters_<biobank>.png: Individual detailed PCA per biobank
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/umap_clusters_<biobank>.png: Individual detailed UMAP per biobank

USAGE:
1. Place this script in PYTHON/ directory as 00-02-biobank-mesh-clustering.py
2. Run from root directory: python PYTHON/00-02-biobank-mesh-clustering.py
3. Ensure biobank_research_data.csv exists in DATA/ directory

REQUIREMENTS:
- pip install scikit-learn pandas numpy matplotlib seaborn umap-learn
"""

import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from datetime import datetime

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. UMAP visualizations will be skipped.")
    print("Install with: pip install umap-learn")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")  # Input data location
analysis_dir = os.path.join(current_dir, "ANALYSIS", "00-02-BIOBANK-MESH-CLUSTERING")
os.makedirs(analysis_dir, exist_ok=True)

#############################################################################
# 1. Data Loading and Preprocessing
#############################################################################

def load_biobank_data():
    """Load biobank research data and validate required columns"""
    input_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    
    # Validate required columns
    required_cols = ['Biobank', 'PMID', 'MeSH_Terms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows without MeSH terms
    df = df.dropna(subset=['MeSH_Terms'])
    df = df[df['MeSH_Terms'].str.strip() != '']
    
    logger.info(f"Loaded {len(df):,} publications with MeSH terms from {df['Biobank'].nunique()} biobanks")
    
    return df

def preprocess_mesh_terms(mesh_string):
    """Clean and preprocess MeSH terms string"""
    if pd.isna(mesh_string) or mesh_string.strip() == '':
        return []
    
    # Split by semicolon (as per biobank retrieval format)
    terms = [term.strip() for term in str(mesh_string).split(';')]
    
    # Clean each term
    cleaned_terms = []
    for term in terms:
        if term:
            # Convert to lowercase, replace spaces with underscores, remove special chars
            cleaned = re.sub(r'[^\w\s]', '', term.lower())
            cleaned = re.sub(r'\s+', '_', cleaned.strip())
            if cleaned:
                cleaned_terms.append(cleaned)
    
    return cleaned_terms

def create_tfidf_matrix(publications_df):
    """Create TF-IDF matrix from MeSH terms"""
    # Preprocess all MeSH terms
    mesh_docs = []
    valid_indices = []
    
    for idx, row in publications_df.iterrows():
        terms = preprocess_mesh_terms(row['MeSH_Terms'])
        if terms:  # Only include publications with valid MeSH terms
            mesh_docs.append(' '.join(terms))
            valid_indices.append(idx)
    
    if len(mesh_docs) < 2:
        logger.warning("Insufficient publications with valid MeSH terms for TF-IDF")
        return None, None, None
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit vocabulary size
        min_df=2,          # Term must appear in at least 2 documents
        max_df=0.8,        # Term must appear in less than 80% of documents
        ngram_range=(1, 2), # Include both unigrams and bigrams
        token_pattern=r'\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform(mesh_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Filter dataframe to only valid indices
    filtered_df = publications_df.loc[valid_indices].copy().reset_index(drop=True)
    
    logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape[0]} documents x {tfidf_matrix.shape[1]} features")
    
    return tfidf_matrix, feature_names, filtered_df

#############################################################################
# 2. Bootstrap Optimal K Selection
#############################################################################

def bootstrap_optimal_k(tfidf_matrix, k_range=(2, 10), n_bootstrap=50, sample_frac=0.8):
    """Bootstrap silhouette scoring to find optimal number of clusters"""
    if tfidf_matrix.shape[0] < 10:
        logger.warning("Too few publications for bootstrap K selection, using K=3")
        return 3
    
    k_scores = defaultdict(list)
    n_samples = max(10, int(tfidf_matrix.shape[0] * sample_frac))
    
    logger.info(f"Bootstrap K selection: trying K={k_range[0]} to {k_range[1]} with {n_bootstrap} iterations")
    
    for iteration in range(n_bootstrap):
        # Random sample
        sample_indices = np.random.choice(tfidf_matrix.shape[0], size=n_samples, replace=False)
        sample_matrix = tfidf_matrix[sample_indices]
        
        for k in range(k_range[0], k_range[1] + 1):
            if k >= sample_matrix.shape[0]:
                continue
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sample_matrix.toarray())
                
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(sample_matrix.toarray(), labels)
                    k_scores[k].append(score)
            except Exception as e:
                logger.warning(f"Error in K={k}, iteration {iteration}: {e}")
                continue
    
    # Find K with highest average silhouette score
    best_k = k_range[0]
    best_score = -1
    
    for k in range(k_range[0], k_range[1] + 1):
        if k_scores[k]:
            avg_score = np.mean(k_scores[k])
            logger.info(f"K={k}: avg silhouette = {avg_score:.4f} (n={len(k_scores[k])})")
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
    
    logger.info(f"Selected optimal K={best_k} with silhouette score {best_score:.4f}")
    return best_k

#############################################################################
# 3. K-means Clustering
#############################################################################

def perform_kmeans_clustering(tfidf_matrix, k):
    """Perform K-means clustering on the full TF-IDF matrix"""
    logger.info(f"Performing K-means clustering with K={k}")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
    
    # Calculate final silhouette score
    if len(np.unique(cluster_labels)) > 1:
        silhouette = silhouette_score(tfidf_matrix.toarray(), cluster_labels)
        logger.info(f"Final silhouette score: {silhouette:.4f}")
    
    return cluster_labels, kmeans

#############################################################################
# 4. c-DF-IPF Scoring
#############################################################################

def compute_cdf_ipf(publications_df, feature_names, tfidf_matrix, cluster_labels):
    """Compute c-DF-IPF scores for each cluster"""
    logger.info("Computing c-DF-IPF scores for clusters")
    
    n_clusters = len(np.unique(cluster_labels))
    n_total_pubs = len(publications_df)
    
    # Count term occurrences across all publications
    term_doc_counts = np.array((tfidf_matrix > 0).sum(axis=0)).flatten()
    
    cluster_summaries = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_pubs = np.sum(cluster_mask)
        
        if cluster_pubs == 0:
            continue
        
        # Get TF-IDF matrix for this cluster
        cluster_tfidf = tfidf_matrix[cluster_mask]
        
        # Calculate DF: proportion of publications in cluster that have each term
        cluster_term_counts = np.array((cluster_tfidf > 0).sum(axis=0)).flatten()
        df_scores = cluster_term_counts / cluster_pubs
        
        # Calculate IPF: log(total publications / publications with term)
        ipf_scores = np.log(n_total_pubs / (term_doc_counts + 1e-8))  # Add small constant to avoid division by zero
        
        # Calculate c-DF-IPF
        cdf_ipf_scores = df_scores * ipf_scores
        
        # Get top terms
        top_indices = np.argsort(cdf_ipf_scores)[::-1][:10]  # Top 10
        
        cluster_summaries[cluster_id] = {
            'n_publications': cluster_pubs,
            'top_terms_cdf_ipf': [
                {
                    'term': feature_names[idx],
                    'cdf_ipf_score': cdf_ipf_scores[idx],
                    'df_score': df_scores[idx],
                    'ipf_score': ipf_scores[idx]
                }
                for idx in top_indices
            ]
        }
        
        # Log top 5 terms
        logger.info(f"Cluster {cluster_id} ({cluster_pubs} pubs) - Top 5 c-DF-IPF terms:")
        for i, term_info in enumerate(cluster_summaries[cluster_id]['top_terms_cdf_ipf'][:5]):
            logger.info(f"  {i+1}. {term_info['term']}: {term_info['cdf_ipf_score']:.4f}")
    
    return cluster_summaries

#############################################################################
# 5. Semantic Summaries
#############################################################################

def compute_semantic_summaries(tfidf_matrix, feature_names, cluster_labels):
    """Compute semantic summaries using mean TF-IDF vectors"""
    logger.info("Computing semantic summaries")
    
    n_clusters = len(np.unique(cluster_labels))
    semantic_summaries = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        
        if np.sum(cluster_mask) == 0:
            continue
        
        # Calculate mean TF-IDF vector for this cluster
        cluster_tfidf = tfidf_matrix[cluster_mask]
        mean_tfidf = np.array(cluster_tfidf.mean(axis=0)).flatten()
        
        # Get top weighted terms
        top_indices = np.argsort(mean_tfidf)[::-1][:10]
        
        semantic_summaries[cluster_id] = [
            {
                'term': feature_names[idx],
                'mean_tfidf': mean_tfidf[idx]
            }
            for idx in top_indices if mean_tfidf[idx] > 0
        ]
    
    return semantic_summaries

#############################################################################
# 6. 2D Projections and Visualization
#############################################################################

def create_2d_projections(tfidf_matrix, cluster_labels, biobank_name):
    """Create PCA and UMAP 2D projections of cluster centroids for a SINGLE biobank
    Returns projection data for later composite visualization"""
    logger.info(f"Computing biobank-specific 2D projections for {biobank_name}")
    
    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
    
    logger.info(f"  {biobank_name}: {n_clusters} clusters with sizes {cluster_sizes}")
    
    # Calculate cluster centroids (mean TF-IDF vectors per cluster)
    centroids = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) > 0:
            centroid = np.array(tfidf_matrix[cluster_mask].mean(axis=0)).flatten()
            centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    if len(centroids) < 2:
        logger.warning(f"Too few centroids for 2D projection in {biobank_name}")
        return None
    
    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(centroids)
    
    # UMAP projection (if available)
    umap_coords = None
    if UMAP_AVAILABLE and len(centroids) >= 3:
        try:
            umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(3, len(centroids)-1))
            umap_coords = umap_reducer.fit_transform(centroids)
        except Exception as e:
            logger.warning(f"Error creating UMAP projection for {biobank_name}: {e}")
            umap_coords = None
    
    # Return projection data for composite visualization
    projection_data = {
        'biobank_name': biobank_name,
        'pca_coords': pca_coords,
        'umap_coords': umap_coords,
        'cluster_sizes': cluster_sizes,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'n_clusters': n_clusters
    }
    
    return projection_data

def create_composite_visualizations(all_projection_data):
    """Create composite PCA and UMAP figures showing all biobanks together"""
    if not all_projection_data:
        logger.warning("No projection data available for composite visualization")
        return
    
    # Filter out None entries
    valid_projections = [p for p in all_projection_data if p is not None]
    if not valid_projections:
        return
    
    n_biobanks = len(valid_projections)
    
    # Calculate grid layout for subplots
    cols = min(3, n_biobanks)  # Max 3 columns
    rows = math.ceil(n_biobanks / cols)
    
    # Create composite PCA figure
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_biobanks == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('PCA: MeSH Term Clusters Across All Biobanks\n'
                 'Each panel shows semantic clusters within one biobank', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for i, proj_data in enumerate(valid_projections):
        ax = axes[i]
        biobank_name = proj_data['biobank_name']
        pca_coords = proj_data['pca_coords']
        cluster_sizes = proj_data['cluster_sizes']
        explained_var = proj_data['pca_explained_variance']
        
        # Create colors for this biobank
        colors = plt.cm.tab10(np.linspace(0, 1, len(pca_coords)))
        
        # Plot each cluster
        for j, (x, y) in enumerate(pca_coords):
            cluster_size = cluster_sizes[j]
            ax.scatter(x, y, c=[colors[j]], s=80 + cluster_size * 1.5, 
                      alpha=0.8, edgecolors='black', linewidth=0.8)
            ax.annotate(f'C{j}', (x, y), xytext=(3, 3), textcoords='offset points', 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.7))
        
        # Customize subplot
        ax.set_title(f'{biobank_name}\n({proj_data["n_clusters"]} clusters)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=9)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for i in range(n_biobanks, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    pca_composite_file = os.path.join(analysis_dir, 'composite_pca_all_biobanks.png')
    plt.savefig(pca_composite_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Composite PCA plot saved: {pca_composite_file}")
    
    # Create composite UMAP figure (if data available)
    umap_data_available = [p for p in valid_projections if p['umap_coords'] is not None]
    
    if umap_data_available and UMAP_AVAILABLE:
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if len(umap_data_available) == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('UMAP: MeSH Term Clusters Across All Biobanks\n'
                     'Each panel shows semantic clusters within one biobank', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for i, proj_data in enumerate(umap_data_available):
            ax = axes[i]
            biobank_name = proj_data['biobank_name']
            umap_coords = proj_data['umap_coords']
            cluster_sizes = proj_data['cluster_sizes']
            
            # Create colors for this biobank
            colors = plt.cm.tab10(np.linspace(0, 1, len(umap_coords)))
            
            # Plot each cluster
            for j, (x, y) in enumerate(umap_coords):
                cluster_size = cluster_sizes[j]
                ax.scatter(x, y, c=[colors[j]], s=80 + cluster_size * 1.5, 
                          alpha=0.8, edgecolors='black', linewidth=0.8)
                ax.annotate(f'C{j}', (x, y), xytext=(3, 3), textcoords='offset points', 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.7))
            
            # Customize subplot
            ax.set_title(f'{biobank_name}\n({proj_data["n_clusters"]} clusters)', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('UMAP1', fontsize=9)
            ax.set_ylabel('UMAP2', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for i in range(len(umap_data_available), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        umap_composite_file = os.path.join(analysis_dir, 'composite_umap_all_biobanks.png')
        plt.savefig(umap_composite_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Composite UMAP plot saved: {umap_composite_file}")
    
    # Create individual biobank plots for detailed analysis
    create_individual_biobank_plots(valid_projections)

#############################################################################
# 7. Save Results
#############################################################################

def save_biobank_results(biobank_name, publications_df, cluster_labels, 
                        cluster_summaries, semantic_summaries):
    """Save clustering results for a single biobank"""
    biobank_clean = biobank_name.lower().replace(" ", "_")
    
    # Save publications with cluster assignments
    results_df = publications_df.copy()
    results_df['cluster'] = cluster_labels
    
    results_file = os.path.join(analysis_dir, f'clustering_results_{biobank_clean}.csv')
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved: {results_file}")
    
    # Save cluster summaries
    summary_rows = []
    for cluster_id, summary in cluster_summaries.items():
        # Top 5 c-DF-IPF terms
        for i, term_info in enumerate(summary['top_terms_cdf_ipf'][:5]):
            summary_rows.append({
                'biobank': biobank_name,
                'cluster': cluster_id,
                'n_publications': summary['n_publications'],
                'rank_cdf_ipf': i + 1,
                'term_cdf_ipf': term_info['term'],
                'cdf_ipf_score': term_info['cdf_ipf_score'],
                'df_score': term_info['df_score'],
                'ipf_score': term_info['ipf_score']
            })
        
        # Top 5 TF-IDF terms
        if cluster_id in semantic_summaries:
            for i, term_info in enumerate(semantic_summaries[cluster_id][:5]):
                # Find corresponding row to add TF-IDF info
                if i < len(summary_rows) and summary_rows[-(5-i)]['cluster'] == cluster_id:
                    row_idx = -(5-i)
                    summary_rows[row_idx]['rank_tfidf'] = i + 1
                    summary_rows[row_idx]['term_tfidf'] = term_info['term']
                    summary_rows[row_idx]['mean_tfidf_score'] = term_info['mean_tfidf']
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(analysis_dir, f'cluster_summaries_{biobank_clean}.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Cluster summaries saved: {summary_file}")
    
    return summary_rows

def save_overall_summary(all_summaries):
    """Save overall summary across all biobanks"""
    if not all_summaries:
        return
    
    overall_df = pd.DataFrame(all_summaries)
    summary_file = os.path.join(analysis_dir, 'biobank_clustering_summary.csv')
    overall_df.to_csv(summary_file, index=False)
    logger.info(f"Overall summary saved: {summary_file}")

#############################################################################
# 8. Main Pipeline
#############################################################################

def process_biobank(biobank_name, biobank_df):
    """Process a single biobank through the complete pipeline (INDEPENDENT ANALYSIS)
    Returns summary rows and projection data for composite visualization"""
    logger.info(f"\n{'='*60}")
    logger.info(f"INDEPENDENT ANALYSIS: {biobank_name} ({len(biobank_df):,} publications)")
    logger.info(f"Clustering publications within {biobank_name} based on MeSH semantic similarity")
    logger.info(f"{'='*60}")
    
    # Minimum publications threshold
    if len(biobank_df) < 10:
        logger.warning(f"Skipping {biobank_name}: too few publications ({len(biobank_df)})")
        return [], None
    
    # 1. Create TF-IDF matrix (biobank-specific)
    logger.info(f"1. Creating TF-IDF matrix from {biobank_name} publications only")
    tfidf_matrix, feature_names, filtered_df = create_tfidf_matrix(biobank_df)
    
    if tfidf_matrix is None:
        logger.warning(f"Skipping {biobank_name}: insufficient valid MeSH terms")
        return [], None
    
    # 2. Bootstrap optimal K selection (within biobank)
    logger.info(f"2. Finding optimal clusters within {biobank_name}")
    optimal_k = bootstrap_optimal_k(tfidf_matrix)
    
    # 3. K-means clustering (biobank-specific)
    logger.info(f"3. Clustering {biobank_name} publications into {optimal_k} semantic groups")
    cluster_labels, kmeans_model = perform_kmeans_clustering(tfidf_matrix, optimal_k)
    
    # 4. c-DF-IPF scoring
    logger.info(f"4. Computing c-DF-IPF scores for {biobank_name} clusters")
    cluster_summaries = compute_cdf_ipf(filtered_df, feature_names, tfidf_matrix, cluster_labels)
    
    # 5. Semantic summaries
    logger.info(f"5. Generating semantic summaries for {biobank_name}")
    semantic_summaries = compute_semantic_summaries(tfidf_matrix, feature_names, cluster_labels)
    
    # 6. 2D projections (biobank-specific) - collect data for composite visualization
    logger.info(f"6. Computing projections for {biobank_name}")
    projection_data = create_2d_projections(tfidf_matrix, cluster_labels, biobank_name)
    
    # 7. Save results
    logger.info(f"7. Saving {biobank_name} results")
    summary_rows = save_biobank_results(biobank_name, filtered_df, cluster_labels, 
                                       cluster_summaries, semantic_summaries)
    
    return summary_rows, projection_data

def main():
    """Main execution function"""
    print("=" * 80)
    print("BIOBANK MESH TERM CLUSTERING PIPELINE")
    print("Per-biobank semantic clustering of publications by MeSH terms")
    print("(Each biobank analyzed independently)")
    print("=" * 80)
    
    try:
        # Load data
        df = load_biobank_data()
        
        # Print biobank statistics
        biobank_stats = df['Biobank'].value_counts()
        print(f"\n📊 Biobank publication counts:")
        for biobank, count in biobank_stats.items():
            print(f"   {biobank}: {count:,} publications")
        
        print(f"\n🎯 Processing pipeline (PER-BIOBANK ANALYSIS):")
        print(f"   Each biobank analyzed independently because:")
        print(f"   - Different research communities & focus areas")
        print(f"   - Distinct MeSH term distributions")
        print(f"   - Separate publication patterns")
        print(f"")
        print(f"   Steps per biobank:")
        print(f"   1. TF-IDF vectorization of MeSH terms (biobank-specific)")
        print(f"   2. Bootstrap optimal K selection (K=2-10)")
        print(f"   3. K-means clustering (within-biobank)")
        print(f"   4. c-DF-IPF scoring")
        print(f"   5. Semantic summaries")
        print(f"   6. 2D projections (PCA + UMAP)")
        print(f"   7. Results output")
        print(f"")
        print(f"   📊 VISUALIZATION STRATEGY:")
        print(f"   - COMPOSITE plots: All biobanks in one figure for comparison")
        print(f"   - INDIVIDUAL plots: Detailed analysis per biobank")
        
        # Process each biobank and collect projection data
        all_summaries = []
        all_projection_data = []
        
        for biobank_name in sorted(df['Biobank'].unique()):
            biobank_df = df[df['Biobank'] == biobank_name].copy()
            summary_rows, projection_data = process_biobank(biobank_name, biobank_df)
            all_summaries.extend(summary_rows)
            if projection_data is not None:
                all_projection_data.append(projection_data)
        
        # Create composite visualizations showing all biobanks together
        logger.info("\n" + "="*60)
        logger.info("CREATING COMPOSITE VISUALIZATIONS")
        logger.info("="*60)
        create_composite_visualizations(all_projection_data)
        
        # Save overall summary
        save_overall_summary(all_summaries)
        
        print(f"\n✅ Clustering pipeline complete!")
        print(f"📂 Results saved to ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/:")
        print(f"   📊 COMPOSITE VISUALIZATIONS (all biobanks in one figure):")
        print(f"   - composite_pca_all_biobanks.png")
        print(f"   - composite_umap_all_biobanks.png")
        print(f"   📈 INDIVIDUAL BIOBANK FILES:")
        print(f"   - clustering_results_<biobank>.csv (per-biobank clustering)")
        print(f"   - cluster_summaries_<biobank>.csv (biobank-specific top MeSH terms)")
        print(f"   - pca_clusters_<biobank>.png (detailed individual PCA)")
        print(f"   - umap_clusters_<biobank>.png (detailed individual UMAP)")
        print(f"   📋 SUMMARY:")
        print(f"   - biobank_clustering_summary.csv (combined summary)")
        
        print(f"\n🎯 Key insights from composite visualizations:")
        print(f"    📊 COMPOSITE PLOTS: Compare cluster patterns across biobanks side-by-side")
        print(f"    📈 INDIVIDUAL PLOTS: Detailed analysis of each biobank's semantic clusters")
        print(f"    🔍 Look for:")
        print(f"       - Shared research themes across biobanks")
        print(f"       - Biobank-specific research specializations")
        print(f"       - Cluster complexity differences between communities")
        print(f"       - Emerging research trends per biobank")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main()