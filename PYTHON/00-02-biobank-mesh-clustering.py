#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BIOBANK MESH TERM CLUSTERING PIPELINE - ENHANCED VERSION WITH UNIFIED VISUALIZATION

Clusters biomedical publications from different biobanks based on their MeSH terms.
Identifies semantic clusters of publications within each biobank using TF-IDF and K-means.
ENHANCED: Includes comprehensive supplementary table generation for cluster interpretation.
NEW: Adds Figure 3B showing all biobank clusters in unified semantic space for overlap analysis.

NOTE: Now applies EXACT same filtering logic as 00-01-biobank-analysis.py to ensure consistent counts:
- Year filtering (2000-2024, excluding 2025 as incomplete)
- Preprint exclusion using the same identifiers and patterns
- Same data cleaning procedures
- Reports total published papers (matching analysis script)
- BUT only clusters papers that have MeSH terms available

FILTERING APPROACH:
1. Apply exact same filtering as analysis script to get total published papers
2. Report these totals (should match analysis script exactly)
3. For clustering: use only the subset that has MeSH terms
4. This ensures consistency while still enabling meaningful clustering

PIPELINE:
1. Load and preprocess data with consistent filtering (same as analysis script)
2. **PER-BIOBANK ANALYSIS** (each biobank analyzed independently):
   a. Create TF-IDF matrix from that biobank's publications with MeSH terms
   b. Bootstrap optimal K selection (silhouette scoring)
   c. K-means clustering within that biobank
   d. c-DF-IPF scoring for top MeSH terms per cluster
   e. Semantic summaries using TF-IDF weights
   f. 2D projections (PCA/UMAP) of cluster centroids
3. **NEW - UNIFIED ANALYSIS** (all biobanks in common semantic space):
   a. Create unified TF-IDF matrix using all publications
   b. Project each biobank's clusters into this common space
   c. Generate Figure 3B showing all clusters together
4. Save biobank-specific results and visualizations
5. Generate comprehensive supplementary table with cluster characteristics

INPUT CSV COLUMNS (from biobank data retrieval):
- Biobank: Which biobank the paper refers to
- PMID: PubMed ID
- MeSH_Terms: Medical Subject Headings (semicolon separated)
- Year: Publication year
- Journal: Journal name

OUTPUT FILES:
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/clustering_results_<biobank>.csv: Publications with cluster assignments
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/cluster_summaries_<biobank>.csv: Top MeSH terms per cluster
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/biobank_clustering_summary.csv: Overall summary across all biobanks
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/composite_pca_all_biobanks.png/.pdf: INTEGRATED PCA showing all biobanks (Figure 3)
- **NEW** ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/unified_pca_all_clusters.png/.pdf: UNIFIED PCA showing overlaps (Figure 3B)
- **NEW** ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/unified_umap_all_clusters.png/.pdf: UNIFIED UMAP showing overlaps
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/composite_umap_all_biobanks.png/.pdf: INTEGRATED UMAP showing all biobanks
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/pca_clusters_<biobank>.png/.pdf: Individual detailed PCA per biobank
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/umap_clusters_<biobank>.png/.pdf: Individual detailed UMAP per biobank
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/supplementary_cluster_characteristics_table.csv: COMPREHENSIVE cluster details
- ANALYSIS/00-02-BIOBANK-MESH-CLUSTERING/cluster_summary_overview.csv: Simplified overview table

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
import matplotlib.patches as mpatches
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

# Define preprint servers and patterns to exclude (same as analysis script)
PREPRINT_IDENTIFIERS = [
    'medRxiv', 'bioRxiv', 'Research Square', 'arXiv', 'ChemRxiv',
    'PeerJ Preprints', 'F1000Research', 'Authorea', 'Preprints.org',
    'SSRN', 'RePEc', 'OSF Preprints', 'SocArXiv', 'PsyArXiv',
    'EarthArXiv', 'engrXiv', 'TechRxiv'
]

#############################################################################
# 1. Data Loading and Preprocessing (CONSISTENT WITH ANALYSIS SCRIPT)
#############################################################################

def load_biobank_data():
    """Load biobank research data and apply EXACT same filtering as analysis script"""
    input_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Loading data from {input_file}")
    df_raw = pd.read_csv(input_file, low_memory=False)
    
    logger.info(f"Loaded {len(df_raw):,} total records from {df_raw['Biobank'].nunique()} biobanks")
    
    # Apply the EXACT same filtering logic as 00-01-biobank-analysis.py
    logger.info("Applying EXACT same filtering logic as analysis script...")
    
    # Step 1: Clean and prepare basic data
    df = df_raw.copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Step 2: Remove records with invalid years
    df_valid_years = df.dropna(subset=['Year']).copy()
    df_valid_years['Year'] = df_valid_years['Year'].astype(int)
    logger.info(f"After removing invalid years: {len(df_valid_years):,} records")
    
    # Step 3: Apply year range filter (2000-2024, exclude 2025 as incomplete)
    df_year_filtered = df_valid_years[(df_valid_years['Year'] >= 2000) & (df_valid_years['Year'] <= 2024)].copy()
    logger.info(f"After year filtering (2000-2024): {len(df_year_filtered):,} records")
    
    # Step 4: Clean MeSH terms and Journal names
    df_year_filtered['MeSH_Terms'] = df_year_filtered['MeSH_Terms'].fillna('')
    df_year_filtered['Journal'] = df_year_filtered['Journal'].fillna('Unknown Journal')
    
    # Step 5: Identify preprints (same logic as analysis script)
    logger.info("Identifying preprints...")
    df_year_filtered['is_preprint'] = False
    
    # Check journal names for preprint identifiers
    for identifier in PREPRINT_IDENTIFIERS:
        mask = df_year_filtered['Journal'].str.contains(identifier, case=False, na=False)
        df_year_filtered.loc[mask, 'is_preprint'] = True
    
    # Additional checks for preprint patterns
    preprint_patterns = [
        r'preprint',
        r'pre-print', 
        r'working paper',
        r'discussion paper'
    ]
    
    for pattern in preprint_patterns:
        mask = df_year_filtered['Journal'].str.contains(pattern, case=False, na=False)
        df_year_filtered.loc[mask, 'is_preprint'] = True
    
    # Step 6: Separate preprints and published papers (SAME AS ANALYSIS SCRIPT)
    df_preprints = df_year_filtered[df_year_filtered['is_preprint'] == True].copy()
    df_published = df_year_filtered[df_year_filtered['is_preprint'] == False].copy()
    
    # Step 7: Print comprehensive filtering statistics (SAME TOTALS AS ANALYSIS SCRIPT)
    total_raw = len(df_raw)
    total_year_filtered = len(df_year_filtered)
    preprint_count = len(df_preprints)
    published_count = len(df_published)
    
    logger.info(f"\n📊 FILTERING RESULTS (SHOULD MATCH ANALYSIS SCRIPT EXACTLY):")
    logger.info(f"   📚 Raw dataset: {total_raw:,} records")
    logger.info(f"   📅 After year filtering (2000-2024): {total_year_filtered:,} records")
    logger.info(f"   📑 Preprints identified: {preprint_count:,} records ({preprint_count/total_year_filtered*100:.1f}%)")
    logger.info(f"   📖 Published papers (ALL): {published_count:,} records ({published_count/total_year_filtered*100:.1f}%)")
    
    # Print biobank distribution for ALL published papers (should match analysis script EXACTLY)
    logger.info(f"\n📋 Published papers by biobank (SHOULD MATCH ANALYSIS SCRIPT):")
    biobank_counts = df_published['Biobank'].value_counts()
    total_published = len(df_published)
    for biobank, count in biobank_counts.items():
        percentage = (count / total_published) * 100
        logger.info(f"   • {biobank}: {count:,} papers ({percentage:.1f}%)")
    logger.info(f"   📊 Total published papers: {biobank_counts.sum():,}")
    
    # NOW check how many have MeSH terms for clustering (but don't exclude from total)
    df_with_mesh = df_published.dropna(subset=['MeSH_Terms'])
    df_with_mesh = df_with_mesh[df_with_mesh['MeSH_Terms'].str.strip() != '']
    
    logger.info(f"\n🔬 MeSH TERM AVAILABILITY FOR CLUSTERING:")
    logger.info(f"   📖 Published papers with MeSH terms: {len(df_with_mesh):,} ({len(df_with_mesh)/published_count*100:.1f}%)")
    logger.info(f"   📖 Published papers without MeSH terms: {published_count - len(df_with_mesh):,} ({(published_count - len(df_with_mesh))/published_count*100:.1f}%)")
    
    mesh_biobank_counts = df_with_mesh['Biobank'].value_counts()
    logger.info(f"\n📋 Papers WITH MeSH terms by biobank (for clustering):")
    for biobank, count in mesh_biobank_counts.items():
        total_biobank = biobank_counts[biobank]
        percentage = (count / total_biobank) * 100
        logger.info(f"   • {biobank}: {count:,} / {total_biobank:,} papers ({percentage:.1f}% have MeSH)")
    
    # Validate required columns
    required_cols = ['Biobank', 'PMID', 'MeSH_Terms']
    missing_cols = [col for col in required_cols if col not in df_with_mesh.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Return ONLY papers with MeSH terms for clustering, but report total published counts
    return df_with_mesh

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
# 6. NEW - UNIFIED SEMANTIC SPACE ANALYSIS
#############################################################################

def create_unified_semantic_space(all_publications_df):
    """Create a unified TF-IDF matrix across ALL biobanks for common semantic space"""
    logger.info("\n" + "="*60)
    logger.info("CREATING UNIFIED SEMANTIC SPACE")
    logger.info("="*60)
    logger.info("Building common TF-IDF vocabulary across all biobanks...")
    
    # Preprocess all MeSH terms from all biobanks
    mesh_docs = []
    doc_biobanks = []
    doc_indices = []
    
    for idx, row in all_publications_df.iterrows():
        terms = preprocess_mesh_terms(row['MeSH_Terms'])
        if terms:
            mesh_docs.append(' '.join(terms))
            doc_biobanks.append(row['Biobank'])
            doc_indices.append(idx)
    
    logger.info(f"Total documents for unified space: {len(mesh_docs):,}")
    
    # Create unified TF-IDF matrix with common vocabulary
    unified_vectorizer = TfidfVectorizer(
        max_features=2000,  # Larger vocabulary for unified space
        min_df=5,           # Higher threshold for unified space
        max_df=0.9,         # Allow more common terms
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    unified_tfidf = unified_vectorizer.fit_transform(mesh_docs)
    unified_features = unified_vectorizer.get_feature_names_out()
    
    logger.info(f"Unified TF-IDF matrix: {unified_tfidf.shape[0]} documents x {unified_tfidf.shape[1]} features")
    
    return unified_tfidf, unified_features, doc_biobanks, doc_indices

def compute_cluster_centroids_in_unified_space(unified_tfidf, doc_biobanks, biobank_cluster_data):
    """Compute cluster centroids for each biobank in the unified semantic space"""
    logger.info("Computing cluster centroids in unified space...")
    
    all_centroids_data = []
    
    for data in biobank_cluster_data:
        biobank_name = data['biobank_name']
        cluster_labels = data['cluster_labels']
        publications_df = data['publications_df']
        
        logger.info(f"Processing {biobank_name} clusters in unified space...")
        
        # Map publications to unified space indices
        biobank_unified_indices = []
        for idx, row in publications_df.iterrows():
            # Find this publication in the unified space
            for i, (biobank, orig_idx) in enumerate(zip(doc_biobanks, range(len(doc_biobanks)))):
                if biobank == biobank_name and i < len(doc_biobanks):
                    biobank_unified_indices.append(i)
                    break
        
        # Compute centroids for each cluster
        n_clusters = len(np.unique(cluster_labels))
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                # Get indices in unified space for this cluster
                cluster_unified_indices = [biobank_unified_indices[i] 
                                          for i in range(len(cluster_mask)) 
                                          if cluster_mask[i] and i < len(biobank_unified_indices)]
                
                if cluster_unified_indices:
                    # Calculate centroid in unified space
                    cluster_vectors = unified_tfidf[cluster_unified_indices]
                    centroid = np.array(cluster_vectors.mean(axis=0)).flatten()
                    
                    all_centroids_data.append({
                        'biobank': biobank_name,
                        'cluster_id': cluster_id,
                        'cluster_size': cluster_size,
                        'centroid': centroid
                    })
    
    return all_centroids_data

def create_unified_visualization(all_centroids_data, save_projections=True, load_existing=True):
    """
    Create Figure 3B: unified visualization showing all clusters together
    
    Parameters:
    -----------
    all_centroids_data : list
        List of dictionaries containing centroid data for each cluster
    save_projections : bool
        Whether to save projection coordinates for reuse
    load_existing : bool
        Whether to load existing projections if available
    """
    logger.info("\n" + "="*60)
    logger.info("CREATING FIGURE 3B: UNIFIED CLUSTER VISUALIZATION")
    logger.info("="*60)
    
    if not all_centroids_data:
        logger.warning("No centroids data available for unified visualization")
        return
    
    # Extract centroids matrix
    centroids_matrix = np.array([d['centroid'] for d in all_centroids_data])
    biobanks = [d['biobank'] for d in all_centroids_data]
    cluster_ids = [d['cluster_id'] for d in all_centroids_data]
    cluster_sizes = [d['cluster_size'] for d in all_centroids_data]
    
    # Define colors for each biobank - using distinct, colorblind-friendly palette
    biobank_colors = {
        'UK Biobank': '#2E86AB',      # Blue
        'FinnGen': '#F24236',          # Red/Orange
        'All of Us': '#73AB84',        # Green
        'Estonian Biobank': '#F6AE2D', # Gold/Yellow
        'Million Veteran Program': '#9B5DE5'  # Purple
    }
    
    # Check for saved projections
    pca_coords_file = os.path.join(analysis_dir, 'unified_pca_coordinates.npy')
    umap_coords_file = os.path.join(analysis_dir, 'unified_umap_coordinates.npy')
    
    # PCA projection
    if load_existing and os.path.exists(pca_coords_file):
        logger.info("Loading existing PCA coordinates...")
        pca_coords = np.load(pca_coords_file)
        # Load explained variance if saved
        pca_var_file = os.path.join(analysis_dir, 'pca_explained_variance.npy')
        if os.path.exists(pca_var_file):
            explained_variance_ratio = np.load(pca_var_file)
        else:
            explained_variance_ratio = [0.0, 0.0]  # Default if not available
    else:
        logger.info("Computing PCA projection for unified space...")
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(centroids_matrix)
        explained_variance_ratio = pca.explained_variance_ratio_
        
        if save_projections:
            np.save(pca_coords_file, pca_coords)
            np.save(os.path.join(analysis_dir, 'pca_explained_variance.npy'), explained_variance_ratio)
            logger.info(f"Saved PCA coordinates to {pca_coords_file}")
    
    # Create Figure 3B - PCA with improved layout
    fig = plt.figure(figsize=(16, 12))
    
    # Main plot
    ax = fig.add_subplot(111)
    
    # Get unique biobanks for legend
    unique_biobanks = sorted(list(set(biobanks)))
    
    # Plot each biobank's clusters
    for biobank in unique_biobanks:
        # Get indices for this biobank
        biobank_indices = [i for i, b in enumerate(biobanks) if b == biobank]
        
        if not biobank_indices:
            continue
            
        # Extract coordinates and sizes for this biobank
        biobank_x = [pca_coords[i, 0] for i in biobank_indices]
        biobank_y = [pca_coords[i, 1] for i in biobank_indices]
        biobank_sizes = [100 + (cluster_sizes[i] * 2) for i in biobank_indices]  # Scale for visibility
        
        # Plot all clusters for this biobank
        scatter = ax.scatter(biobank_x, biobank_y,
                           c=biobank_colors.get(biobank, '#333333'),
                           s=biobank_sizes,
                           alpha=0.6,
                           edgecolors='white',
                           linewidth=2,
                           label=biobank)
        
        # Add cluster labels with better visibility
        for idx, i in enumerate(biobank_indices):
            cluster_id = cluster_ids[i]
            x, y = pca_coords[i]
            
            # Add white background for better visibility
            ax.annotate(f'C{cluster_id}',
                       (x, y),
                       ha='center',
                       va='center',
                       fontsize=9,
                       fontweight='bold',
                       color='black',
                       bbox=dict(boxstyle='circle,pad=0.3',
                               facecolor='white',
                               edgecolor=biobank_colors.get(biobank, '#333333'),
                               alpha=0.9,
                               linewidth=1.5))
    
    # Customize plot appearance
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance explained)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance explained)', fontsize=13, fontweight='bold')
    
    # Title with better spacing
    ax.set_title('Figure 3B: Unified Semantic Space - All Biobank Clusters Together\n' +
                'Revealing Overlaps and Distinct Research Territories',
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid for better readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add spines styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#333333')
    
    # Create legend for biobanks with better positioning
    biobank_legend = ax.legend(loc='upper left',
                              frameon=True,
                              fancybox=True,
                              shadow=True,
                              title='Biobank',
                              title_fontsize=12,
                              fontsize=11,
                              markerscale=1.5,
                              framealpha=0.95,
                              edgecolor='black')
    
    # Add biobank legend
    ax.add_artist(biobank_legend)
    
    # Create size legend - using dummy points
    size_legend_handles = []
    for size, label in [(100, 'Small (<50 papers)'), 
                        (300, 'Medium (50-150 papers)'), 
                        (600, 'Large (>150 papers)')]:
        size_legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', markersize=np.sqrt(size/10),
                      alpha=0.6, markeredgecolor='white', markeredgewidth=2,
                      label=label)
        )
    
    # Add size legend
    size_legend = ax.legend(handles=size_legend_handles,
                          loc='upper right',
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          title='Cluster Size',
                          title_fontsize=12,
                          fontsize=11,
                          framealpha=0.95,
                          edgecolor='black')
    
    # Keep both legends visible
    ax.add_artist(biobank_legend)
    
    # Add informative text box instead of bottom text
    textstr = ('Interpretation Guide:\n'
              '• Each circle = one semantic cluster from a biobank\n'
              '• Circle size = number of publications in cluster\n'
              '• Proximity = semantic similarity (shared MeSH terms)\n'
              '• Overlapping circles = shared research themes')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=props)
    
    # Adjust layout to prevent overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save Figure 3B
    pca_unified_file = os.path.join(analysis_dir, 'unified_pca_all_clusters.png')
    pca_unified_pdf = os.path.join(analysis_dir, 'unified_pca_all_clusters.pdf')
    plt.savefig(pca_unified_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pca_unified_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ Figure 3B (PCA) saved: {pca_unified_file} and {pca_unified_pdf}")
    
    # Create UMAP version if available
    if UMAP_AVAILABLE and len(centroids_matrix) >= 10:
        # Check for saved UMAP coordinates
        if load_existing and os.path.exists(umap_coords_file):
            logger.info("Loading existing UMAP coordinates...")
            umap_coords = np.load(umap_coords_file)
        else:
            logger.info("Computing UMAP projection for unified space...")
            try:
                umap_reducer = umap.UMAP(n_components=2,
                                        random_state=42,
                                        n_neighbors=min(15, len(centroids_matrix)-1),
                                        min_dist=0.1,
                                        metric='cosine')
                umap_coords = umap_reducer.fit_transform(centroids_matrix)
                
                if save_projections:
                    np.save(umap_coords_file, umap_coords)
                    logger.info(f"Saved UMAP coordinates to {umap_coords_file}")
            except Exception as e:
                logger.warning(f"Error creating unified UMAP: {e}")
                return
        
        # Create UMAP figure with same improved layout
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        
        # Plot each biobank's clusters
        for biobank in unique_biobanks:
            biobank_indices = [i for i, b in enumerate(biobanks) if b == biobank]
            
            if not biobank_indices:
                continue
                
            biobank_x = [umap_coords[i, 0] for i in biobank_indices]
            biobank_y = [umap_coords[i, 1] for i in biobank_indices]
            biobank_sizes = [100 + (cluster_sizes[i] * 2) for i in biobank_indices]
            
            ax.scatter(biobank_x, biobank_y,
                      c=biobank_colors.get(biobank, '#333333'),
                      s=biobank_sizes,
                      alpha=0.6,
                      edgecolors='white',
                      linewidth=2,
                      label=biobank)
            
            # Add cluster labels
            for idx, i in enumerate(biobank_indices):
                cluster_id = cluster_ids[i]
                x, y = umap_coords[i]
                
                ax.annotate(f'C{cluster_id}',
                          (x, y),
                          ha='center',
                          va='center',
                          fontsize=9,
                          fontweight='bold',
                          color='black',
                          bbox=dict(boxstyle='circle,pad=0.3',
                                  facecolor='white',
                                  edgecolor=biobank_colors.get(biobank, '#333333'),
                                  alpha=0.9,
                                  linewidth=1.5))
        
        # Customize UMAP plot
        ax.set_xlabel('UMAP1', fontsize=13, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=13, fontweight='bold')
        ax.set_title('Figure 3B (UMAP): Unified Semantic Space - All Biobank Clusters Together\n' +
                    'Non-linear Projection Revealing Research Territories',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#333333')
        
        # Add legends (same as PCA)
        biobank_legend = ax.legend(loc='upper left',
                                  frameon=True,
                                  fancybox=True,
                                  shadow=True,
                                  title='Biobank',
                                  title_fontsize=12,
                                  fontsize=11,
                                  markerscale=1.5,
                                  framealpha=0.95,
                                  edgecolor='black')
        
        ax.add_artist(biobank_legend)
        
        size_legend = ax.legend(handles=size_legend_handles,
                              loc='upper right',
                              frameon=True,
                              fancybox=True,
                              shadow=True,
                              title='Cluster Size',
                              title_fontsize=12,
                              fontsize=11,
                              framealpha=0.95,
                              edgecolor='black')
        
        ax.add_artist(biobank_legend)
        
        # Add interpretation guide
        textstr = ('Interpretation Guide:\n'
                  '• UMAP preserves local structure better than PCA\n'
                  '• Tight groups = highly similar research themes\n'
                  '• Distance reflects semantic dissimilarity\n'
                  '• Isolated clusters = unique research focus')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', bbox=props)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save UMAP figure
        umap_unified_file = os.path.join(analysis_dir, 'unified_umap_all_clusters.png')
        umap_unified_pdf = os.path.join(analysis_dir, 'unified_umap_all_clusters.pdf')
        plt.savefig(umap_unified_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(umap_unified_pdf, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"✅ Figure 3B (UMAP) saved: {umap_unified_file} and {umap_unified_pdf}")
    
    # Generate overlap analysis report
    generate_overlap_analysis(all_centroids_data, pca_coords)
    
    # Print summary of what's visible
    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("="*60)
    for biobank in unique_biobanks:
        biobank_clusters = [i for i, b in enumerate(biobanks) if b == biobank]
        total_pubs = sum([cluster_sizes[i] for i in biobank_clusters])
        logger.info(f"{biobank}: {len(biobank_clusters)} clusters, {total_pubs} total publications")
    
    return pca_coords, umap_coords if 'umap_coords' in locals() else None

def create_annotated_unified_visualization(all_centroids_data, all_cluster_data, pca_coords=None, show_top_terms=True):
    """
    Create an annotated version of Figure 3B with cluster descriptions
    
    Parameters:
    -----------
    all_centroids_data : list
        List of dictionaries containing centroid data
    all_cluster_data : list
        List of dictionaries containing full cluster information including top terms
    pca_coords : numpy array, optional
        Pre-computed PCA coordinates (to avoid recomputation)
    show_top_terms : bool
        Whether to show top MeSH terms for selected clusters
    """
    logger.info("\n" + "="*60)
    logger.info("CREATING ANNOTATED UNIFIED VISUALIZATION")
    logger.info("="*60)
    
    if not all_centroids_data or not all_cluster_data:
        logger.warning("Missing data for annotated visualization")
        return
    
    # Extract data
    centroids_matrix = np.array([d['centroid'] for d in all_centroids_data])
    biobanks = [d['biobank'] for d in all_centroids_data]
    cluster_ids = [d['cluster_id'] for d in all_centroids_data]
    cluster_sizes = [d['cluster_size'] for d in all_centroids_data]
    
    # Define colors
    biobank_colors = {
        'UK Biobank': '#2E86AB',
        'FinnGen': '#F24236',
        'All of Us': '#73AB84',
        'Estonian Biobank': '#F6AE2D',
        'Million Veteran Program': '#9B5DE5'
    }
    
    # Use provided PCA coords or load/compute
    if pca_coords is None:
        pca_coords_file = os.path.join(analysis_dir, 'unified_pca_coordinates.npy')
        if os.path.exists(pca_coords_file):
            pca_coords = np.load(pca_coords_file)
            pca_var_file = os.path.join(analysis_dir, 'pca_explained_variance.npy')
            explained_variance_ratio = np.load(pca_var_file) if os.path.exists(pca_var_file) else [0.0, 0.0]
        else:
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(centroids_matrix)
            explained_variance_ratio = pca.explained_variance_ratio_
    else:
        pca_var_file = os.path.join(analysis_dir, 'pca_explained_variance.npy')
        explained_variance_ratio = np.load(pca_var_file) if os.path.exists(pca_var_file) else [0.0, 0.0]
    
    # Create mapping of biobank-cluster to top terms
    cluster_terms_map = {}
    for data in all_cluster_data:
        biobank_name = data['biobank_name']
        cluster_summaries = data['cluster_summaries']
        for cluster_id, summary in cluster_summaries.items():
            key = f"{biobank_name}_C{cluster_id}"
            top_terms = summary['top_terms_cdf_ipf'][:3]  # Top 3 terms
            cluster_terms_map[key] = [term['term'].replace('_', ' ').title() for term in top_terms]
    
    # Create figure with annotations
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    
    unique_biobanks = sorted(list(set(biobanks)))
    
    # Find clusters to annotate (largest from each biobank + overlapping ones)
    clusters_to_annotate = []
    
    # Get largest cluster from each biobank
    for biobank in unique_biobanks:
        biobank_indices = [i for i, b in enumerate(biobanks) if b == biobank]
        if biobank_indices:
            largest_idx = max(biobank_indices, key=lambda i: cluster_sizes[i])
            clusters_to_annotate.append(largest_idx)
    
    # Find overlapping clusters (close in PCA space)
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(pca_coords, metric='euclidean'))
    
    for i in range(len(biobanks)):
        for j in range(i+1, len(biobanks)):
            if biobanks[i] != biobanks[j] and distances[i, j] < 0.2:  # Threshold for "close"
                clusters_to_annotate.extend([i, j])
    
    clusters_to_annotate = list(set(clusters_to_annotate))  # Remove duplicates
    
    # Plot clusters
    for biobank in unique_biobanks:
        biobank_indices = [i for i, b in enumerate(biobanks) if b == biobank]
        
        if not biobank_indices:
            continue
            
        biobank_x = [pca_coords[i, 0] for i in biobank_indices]
        biobank_y = [pca_coords[i, 1] for i in biobank_indices]
        biobank_sizes = [100 + (cluster_sizes[i] * 2) for i in biobank_indices]
        
        ax.scatter(biobank_x, biobank_y,
                  c=biobank_colors.get(biobank, '#333333'),
                  s=biobank_sizes,
                  alpha=0.6,
                  edgecolors='white',
                  linewidth=2,
                  label=biobank)
        
        # Add cluster labels
        for idx, i in enumerate(biobank_indices):
            cluster_id = cluster_ids[i]
            x, y = pca_coords[i]
            
            # Basic cluster label
            ax.annotate(f'C{cluster_id}',
                       (x, y),
                       ha='center',
                       va='center',
                       fontsize=9,
                       fontweight='bold',
                       color='black',
                       bbox=dict(boxstyle='circle,pad=0.3',
                               facecolor='white',
                               edgecolor=biobank_colors.get(biobank, '#333333'),
                               alpha=0.9,
                               linewidth=1.5))
            
            # Add detailed annotation for selected clusters
            if show_top_terms and i in clusters_to_annotate:
                key = f"{biobank}_C{cluster_id}"
                if key in cluster_terms_map:
                    terms_text = '\n'.join(cluster_terms_map[key][:2])  # Show top 2 terms
                    
                    # Position annotation to avoid overlap
                    offset_x = 0.15 if x > 0 else -0.15
                    offset_y = 0.1 if y > 0 else -0.1
                    
                    ax.annotate(terms_text,
                              xy=(x, y),
                              xytext=(x + offset_x, y + offset_y),
                              fontsize=8,
                              ha='center',
                              bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='lightyellow',
                                      edgecolor=biobank_colors.get(biobank, '#333333'),
                                      alpha=0.8),
                              arrowprops=dict(arrowstyle='-',
                                            connectionstyle='arc3,rad=0.3',
                                            color=biobank_colors.get(biobank, '#333333'),
                                            alpha=0.5))
    
    # Plot styling
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance explained)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance explained)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 3B (Annotated): Unified Semantic Space with Cluster Descriptions\n' +
                'Top MeSH Terms Shown for Key Clusters',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#333333')
    
    # Legends
    biobank_legend = ax.legend(loc='upper left',
                              frameon=True,
                              fancybox=True,
                              shadow=True,
                              title='Biobank',
                              title_fontsize=12,
                              fontsize=11,
                              markerscale=1.5,
                              framealpha=0.95,
                              edgecolor='black')
    
    ax.add_artist(biobank_legend)
    
    # Size legend
    size_legend_handles = []
    for size, label in [(100, 'Small (<50 papers)'),
                        (300, 'Medium (50-150 papers)'),
                        (600, 'Large (>150 papers)')]:
        size_legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='gray', markersize=np.sqrt(size/10),
                      alpha=0.6, markeredgecolor='white', markeredgewidth=2,
                      label=label)
        )
    
    size_legend = ax.legend(handles=size_legend_handles,
                          loc='upper right',
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          title='Cluster Size',
                          title_fontsize=12,
                          fontsize=11,
                          framealpha=0.95,
                          edgecolor='black')
    
    ax.add_artist(biobank_legend)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save annotated version
    annotated_file = os.path.join(analysis_dir, 'unified_pca_annotated.png')
    annotated_pdf = os.path.join(analysis_dir, 'unified_pca_annotated.pdf')
    plt.savefig(annotated_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(annotated_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ Annotated Figure 3B saved: {annotated_file} and {annotated_pdf}")
    
    return pca_coords


def create_overlap_heatmap(all_centroids_data):
    """
    Create a heatmap showing semantic similarity between all cluster pairs
    """
    from scipy.spatial.distance import cdist
    import seaborn as sns
    
    logger.info("Creating cluster overlap heatmap...")
    
    # Extract data
    centroids_matrix = np.array([d['centroid'] for d in all_centroids_data])
    biobanks = [d['biobank'] for d in all_centroids_data]
    cluster_ids = [d['cluster_id'] for d in all_centroids_data]
    
    # Calculate cosine similarity (1 - cosine distance)
    distances = cdist(centroids_matrix, centroids_matrix, metric='cosine')
    similarities = 1 - distances
    
    # Create labels for clusters
    cluster_labels = [f"{biobank[:3]}-C{cluster_id}" for biobank, cluster_id in zip(biobanks, cluster_ids)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(similarities,
                xticklabels=cluster_labels,
                yticklabels=cluster_labels,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Semantic Similarity (1 - Cosine Distance)'},
                ax=ax)
    
    # Customize
    ax.set_title('Semantic Similarity Matrix Between All Biobank Clusters\n' +
                'Darker = More Similar Research Themes',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = os.path.join(analysis_dir, 'cluster_similarity_heatmap.png')
    heatmap_pdf = os.path.join(analysis_dir, 'cluster_similarity_heatmap.pdf')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(heatmap_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ Similarity heatmap saved: {heatmap_file} and {heatmap_pdf}")
    
    return similarities, cluster_labels

def generate_overlap_analysis(all_centroids_data, pca_coords):
    """Analyze and report cluster overlaps in semantic space"""
    logger.info("\n" + "="*60)
    logger.info("ANALYZING SEMANTIC OVERLAPS")
    logger.info("="*60)
    
    # Calculate pairwise distances between clusters
    from scipy.spatial.distance import cdist
    
    centroids_matrix = np.array([d['centroid'] for d in all_centroids_data])
    distances = cdist(centroids_matrix, centroids_matrix, metric='cosine')
    
    # Find closest clusters across different biobanks
    overlap_threshold = 0.3  # Cosine distance threshold for "overlap"
    overlaps = []
    
    for i in range(len(all_centroids_data)):
        for j in range(i+1, len(all_centroids_data)):
            if all_centroids_data[i]['biobank'] != all_centroids_data[j]['biobank']:
                dist = distances[i, j]
                if dist < overlap_threshold:
                    overlaps.append({
                        'biobank1': all_centroids_data[i]['biobank'],
                        'cluster1': all_centroids_data[i]['cluster_id'],
                        'biobank2': all_centroids_data[j]['biobank'],
                        'cluster2': all_centroids_data[j]['cluster_id'],
                        'distance': dist
                    })
    
    # Sort by distance (closest first)
    overlaps.sort(key=lambda x: x['distance'])
    
    # Report top overlaps
    logger.info(f"\nTop semantic overlaps (distance < {overlap_threshold}):")
    for i, overlap in enumerate(overlaps[:10]):
        logger.info(f"{i+1}. {overlap['biobank1']} C{overlap['cluster1']} <-> " +
                   f"{overlap['biobank2']} C{overlap['cluster2']}: " +
                   f"distance = {overlap['distance']:.3f}")
    
    if not overlaps:
        logger.info("No strong semantic overlaps detected between biobanks.")
    
    # Save overlap analysis
    if overlaps:
        overlap_df = pd.DataFrame(overlaps)
        overlap_file = os.path.join(analysis_dir, 'semantic_overlap_analysis.csv')
        overlap_df.to_csv(overlap_file, index=False)
        logger.info(f"\n✅ Overlap analysis saved: {overlap_file}")

#############################################################################
# 7. Original 2D Projections and Visualization (Figure 3)
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
    """Create composite PCA and UMAP figures showing all biobanks together (Figure 3)"""
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
    
    # Create composite PCA figure (Figure 3)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_biobanks == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Figure 3: PCA - MeSH Term Clusters Across All Biobanks\n'
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
    pca_composite_pdf = os.path.join(analysis_dir, 'composite_pca_all_biobanks.pdf')
    plt.savefig(pca_composite_file, dpi=300, bbox_inches='tight')
    plt.savefig(pca_composite_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure 3 (PCA) saved: {pca_composite_file} and {pca_composite_pdf}")
    
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
        umap_composite_pdf = os.path.join(analysis_dir, 'composite_umap_all_biobanks.pdf')
        plt.savefig(umap_composite_file, dpi=300, bbox_inches='tight')
        plt.savefig(umap_composite_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Composite UMAP plot saved: {umap_composite_file} and {umap_composite_pdf}")
    
    # Create individual biobank plots for detailed analysis
    create_individual_biobank_plots(valid_projections)

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
        pca_file_pdf = os.path.join(analysis_dir, f'pca_clusters_{biobank_name.lower().replace(" ", "_")}.pdf')
        plt.savefig(pca_file, dpi=300, bbox_inches='tight')
        plt.savefig(pca_file_pdf, dpi=300, bbox_inches='tight')
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
            umap_file_pdf = os.path.join(analysis_dir, f'umap_clusters_{biobank_name.lower().replace(" ", "_")}.pdf')
            plt.savefig(umap_file, dpi=300, bbox_inches='tight')
            plt.savefig(umap_file_pdf, dpi=300, bbox_inches='tight')
            plt.close()

#############################################################################
# 8. Save Results
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
# 9. Supplementary Table Generation
#############################################################################

def generate_cluster_description(top_terms):
    """Generate a human-readable description of the cluster based on top terms"""
    if not top_terms:
        return "Undefined cluster"
    
    # Convert MeSH terms to readable format
    readable_terms = [term['term'].replace('_', ' ').title() for term in top_terms]
    
    if len(readable_terms) == 1:
        return f"Research focused on {readable_terms[0]}"
    elif len(readable_terms) == 2:
        return f"Research combining {readable_terms[0]} and {readable_terms[1]}"
    else:
        return f"Research spanning {readable_terms[0]}, {readable_terms[1]}, and {readable_terms[2]}"

def infer_research_theme(top_terms):
    """Infer the primary research theme based on top terms"""
    if not top_terms:
        return "Unknown"
    
    # Common research theme mappings
    theme_keywords = {
        'Genomics': ['genome', 'genetic', 'dna', 'snp', 'gwas', 'polymorphism', 'allele', 'genotype'],
        'Cardiovascular': ['heart', 'cardiovascular', 'cardiac', 'blood_pressure', 'hypertension', 'coronary'],
        'Neurological': ['brain', 'neurological', 'cognitive', 'alzheimer', 'dementia', 'neural'],
        'Metabolic': ['diabetes', 'obesity', 'metabolism', 'glucose', 'insulin', 'lipid'],
        'Cancer': ['cancer', 'tumor', 'oncology', 'malignant', 'carcinoma', 'neoplasm'],
        'Imaging': ['imaging', 'mri', 'scan', 'radiological', 'tomography'],
        'Environmental': ['environment', 'pollution', 'exposure', 'air_quality', 'toxicity'],
        'Epidemiological': ['epidemiology', 'cohort', 'longitudinal', 'prospective', 'risk_factors'],
        'Pregnancy/Maternal': ['pregnancy', 'maternal', 'prenatal', 'birth', 'fetal'],
        'Pediatric': ['infant', 'child', 'pediatric', 'adolescent', 'development']
    }
    
    # Check top terms against theme keywords
    top_term_text = ' '.join([term['term'].lower() for term in top_terms])
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in top_term_text for keyword in keywords):
            return theme
    
    # If no specific theme found, use the most prominent term
    return top_terms[0]['term'].replace('_', ' ').title()

def create_supplementary_cluster_table(all_cluster_data):
    """Create comprehensive supplementary table summarizing all cluster characteristics"""
    logger.info("Creating supplementary cluster characteristics table")
    
    # Ensure analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)
    logger.info(f"Output directory: {analysis_dir}")
    
    supplementary_rows = []
    
    for biobank_data in all_cluster_data:
        biobank_name = biobank_data['biobank_name']
        cluster_summaries = biobank_data['cluster_summaries']
        publications_df = biobank_data['publications_df']
        cluster_labels = biobank_data['cluster_labels']
        feature_names = biobank_data['feature_names']
        tfidf_matrix = biobank_data['tfidf_matrix']
        
        for cluster_id, summary in cluster_summaries.items():
            # Get cluster-specific data
            cluster_mask = cluster_labels == cluster_id
            cluster_pubs_count = summary['n_publications']
            
            # Count unique MeSH terms in this cluster
            cluster_tfidf = tfidf_matrix[cluster_mask]
            unique_terms_in_cluster = np.sum(np.array((cluster_tfidf > 0).sum(axis=0)).flatten() > 0)
            
            # Get top 5 c-DF-IPF terms with their semantic descriptors
            top_terms = summary['top_terms_cdf_ipf'][:5]
            
            # Format top terms for display
            top_terms_formatted = []
            for i, term_info in enumerate(top_terms):
                term = term_info['term']
                score = term_info['cdf_ipf_score']
                
                # Create semantic descriptor (convert underscore back to readable format)
                semantic_descriptor = term.replace('_', ' ').title()
                
                top_terms_formatted.append(f"{term} ({semantic_descriptor}): {score:.4f}")
            
            # Create row for supplementary table
            supplementary_row = {
                'Biobank': biobank_name,
                'Cluster_ID': f"C{cluster_id}",
                'Number_of_Publications': cluster_pubs_count,
                'Total_Unique_MeSH_Terms': unique_terms_in_cluster,
                'Top_5_Semantic_Terms_cDF_IPF': ' | '.join(top_terms_formatted),
                'Cluster_Description': generate_cluster_description(top_terms[:3]),
                'Primary_Research_Theme': infer_research_theme(top_terms[:3]),
                'c_DF_IPF_Score_Range': f"{top_terms[0]['cdf_ipf_score']:.4f} - {top_terms[-1]['cdf_ipf_score']:.4f}" if top_terms else "N/A"
            }
            
            # Add individual top terms as separate columns for easier analysis
            for i, term_info in enumerate(top_terms[:5]):
                col_prefix = f"Term_{i+1}"
                supplementary_row[f"{col_prefix}_MeSH"] = term_info['term']
                supplementary_row[f"{col_prefix}_Semantic_Descriptor"] = term_info['term'].replace('_', ' ').title()
                supplementary_row[f"{col_prefix}_cDF_IPF_Score"] = term_info['cdf_ipf_score']
                supplementary_row[f"{col_prefix}_DF_Score"] = term_info['df_score']
                supplementary_row[f"{col_prefix}_IPF_Score"] = term_info['ipf_score']
            
            supplementary_rows.append(supplementary_row)
    
    # Create DataFrame and save
    supplementary_df = pd.DataFrame(supplementary_rows)
    
    # Sort by biobank and cluster size for better readability
    supplementary_df = supplementary_df.sort_values(['Biobank', 'Number_of_Publications'], ascending=[True, False])
    
    # Save supplementary table
    supplementary_file = os.path.join(analysis_dir, 'supplementary_cluster_characteristics_table.csv')
    supplementary_df.to_csv(supplementary_file, index=False)
    logger.info(f"✅ Supplementary cluster characteristics table saved: {supplementary_file}")
    
    # Also create a summary version with just key information
    summary_columns = ['Biobank', 'Cluster_ID', 'Number_of_Publications', 
                      'Total_Unique_MeSH_Terms', 'Primary_Research_Theme',
                      'Top_5_Semantic_Terms_cDF_IPF']
    
    summary_df = supplementary_df[summary_columns].copy()
    summary_file = os.path.join(analysis_dir, 'cluster_summary_overview.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"✅ Cluster summary overview saved: {summary_file}")
    
    # Verify files exist and log their sizes
    if os.path.exists(supplementary_file):
        size_kb = os.path.getsize(supplementary_file) / 1024
        logger.info(f"📊 Supplementary table: {len(supplementary_df)} rows, {len(supplementary_df.columns)} columns ({size_kb:.1f} KB)")
    
    if os.path.exists(summary_file):
        size_kb = os.path.getsize(summary_file) / 1024
        logger.info(f"📊 Summary overview: {len(summary_df)} rows, {len(summary_df.columns)} columns ({size_kb:.1f} KB)")
    
    return supplementary_df

#############################################################################
# 10. Process Individual Biobank
#############################################################################

def process_biobank_enhanced(biobank_name, biobank_df):
    """Enhanced version of process_biobank that collects data for supplementary table"""
    logger.info(f"\n{'='*60}")
    logger.info(f"INDEPENDENT ANALYSIS: {biobank_name} ({len(biobank_df):,} publications)")
    logger.info(f"Clustering publications within {biobank_name} based on MeSH semantic similarity")
    logger.info(f"{'='*60}")
    
    # Minimum publications threshold
    if len(biobank_df) < 10:
        logger.warning(f"Skipping {biobank_name}: too few publications ({len(biobank_df)})")
        return [], None, None
    
    # 1. Create TF-IDF matrix (biobank-specific)
    logger.info(f"1. Creating TF-IDF matrix from {biobank_name} publications only")
    tfidf_matrix, feature_names, filtered_df = create_tfidf_matrix(biobank_df)
    
    if tfidf_matrix is None:
        logger.warning(f"Skipping {biobank_name}: insufficient valid MeSH terms")
        return [], None, None
    
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
    
    # 8. Collect data for supplementary table and unified analysis
    cluster_data = {
        'biobank_name': biobank_name,
        'cluster_summaries': cluster_summaries,
        'publications_df': filtered_df,
        'cluster_labels': cluster_labels,
        'feature_names': feature_names,
        'tfidf_matrix': tfidf_matrix,
        'semantic_summaries': semantic_summaries
    }
    
    return summary_rows, projection_data, cluster_data

#############################################################################
# 11. Main Enhanced Pipeline
#############################################################################

def main_enhanced():
    """Enhanced main execution function with unified visualization (Figure 3B)"""
    print("=" * 80)
    print("BIOBANK MESH TERM CLUSTERING PIPELINE - ENHANCED WITH UNIFIED VISUALIZATION")
    print("Per-biobank semantic clustering + unified semantic space analysis")
    print("Generates Figure 3 (separate panels) and Figure 3B (unified space)")
    print("=" * 80)
    
    try:
        # Load data with EXACT same filtering as analysis script
        df_with_mesh = load_biobank_data()
        
        print(f"\n🎯 Processing pipeline:")
        print(f"   📊 Figure 3: Each biobank analyzed independently (separate panels)")
        print(f"   🔄 Figure 3B: All clusters in unified semantic space (shows overlaps)")
        print(f"   📋 Supplementary tables with cluster characteristics")
        print(f"   ✅ EXACT FILTERING: Same totals as 00-01-biobank-analysis.py")
        print(f"   🔬 CLUSTERING: Only papers with MeSH terms ({len(df_with_mesh):,} papers)")
        
        # Process each biobank and collect data
        all_summaries = []
        all_projection_data = []
        all_cluster_data = []
        
        for biobank_name in sorted(df_with_mesh['Biobank'].unique()):
            biobank_df = df_with_mesh[df_with_mesh['Biobank'] == biobank_name].copy()
            summary_rows, projection_data, cluster_data = process_biobank_enhanced(biobank_name, biobank_df)
            
            all_summaries.extend(summary_rows)
            if projection_data is not None:
                all_projection_data.append(projection_data)
            if cluster_data is not None:
                all_cluster_data.append(cluster_data)
        
        # Create Figure 3: composite visualizations (separate panels)
        logger.info("\n" + "="*60)
        logger.info("CREATING FIGURE 3: COMPOSITE VISUALIZATIONS")
        logger.info("="*60)
        create_composite_visualizations(all_projection_data)
        
        # NEW: Create unified semantic space and Figure 3B
        logger.info("\n" + "="*60)
        logger.info("CREATING UNIFIED SEMANTIC SPACE FOR FIGURE 3B")
        logger.info("="*60)
        
        # Create unified TF-IDF space
        unified_tfidf, unified_features, doc_biobanks, doc_indices = create_unified_semantic_space(df_with_mesh)
        
        # Compute cluster centroids in unified space
        all_centroids_data = compute_cluster_centroids_in_unified_space(
            unified_tfidf, doc_biobanks, all_cluster_data
        )
        
        # Create Figure 3B: unified visualization
        create_unified_visualization(all_centroids_data)
        
        # Create supplementary cluster characteristics table
        logger.info("\n" + "="*60)
        logger.info("CREATING SUPPLEMENTARY CLUSTER CHARACTERISTICS TABLE")
        logger.info("="*60)
        supplementary_df = create_supplementary_cluster_table(all_cluster_data)
        
        # Save overall summary
        save_overall_summary(all_summaries)
        
        # Print summary statistics
        print(f"\n✅ Enhanced clustering pipeline complete!")
        print(f"📂 All results saved to: {analysis_dir}")
        
        print(f"\n🎨 FIGURE 3 COMPONENTS:")
        print(f"   📊 Figure 3: composite_pca_all_biobanks.png/.pdf")
        print(f"      - Shows each biobank's clusters independently")
        print(f"      - Separate panels for comparison")
        print(f"   🔄 Figure 3B: unified_pca_all_clusters.png/.pdf")
        print(f"      - All clusters in shared semantic space")
        print(f"      - Reveals overlaps and distinct territories")
        print(f"      - Color = biobank, size = cluster size")
        
        print(f"\n📊 SUPPLEMENTARY TABLE STATISTICS:")
        print(f"   Total clusters analyzed: {len(supplementary_df)}")
        print(f"   Biobanks included: {supplementary_df['Biobank'].nunique()}")
        print(f"   Largest cluster: {supplementary_df['Number_of_Publications'].max()} publications")
        print(f"   Average cluster size: {supplementary_df['Number_of_Publications'].mean():.1f} publications")
        
        print(f"\n📂 ALL OUTPUT FILES:")
        print(f"   🎨 FIGURE 3 VISUALIZATIONS:")
        print(f"      - composite_pca_all_biobanks.png/.pdf (Figure 3)")
        print(f"      - unified_pca_all_clusters.png/.pdf (Figure 3B)")
        print(f"      - unified_umap_all_clusters.png/.pdf (Figure 3B - UMAP)")
        print(f"      - semantic_overlap_analysis.csv (cluster distances)")
        print(f"   📋 SUPPLEMENTARY TABLES:")
        print(f"      - supplementary_cluster_characteristics_table.csv")
        print(f"      - cluster_summary_overview.csv")
        print(f"   📈 INDIVIDUAL BIOBANK FILES:")
        print(f"      - clustering_results_<biobank>.csv")
        print(f"      - cluster_summaries_<biobank>.csv")
        print(f"      - pca_clusters_<biobank>.png/.pdf")
        print(f"      - umap_clusters_<biobank>.png/.pdf")
        print(f"   📋 SUMMARY:")
        print(f"      - biobank_clustering_summary.csv")
        
        print(f"\n💡 KEY INSIGHTS FROM FIGURE 3B:")
        print(f"   - Overlapping clusters indicate shared research themes")
        print(f"   - Isolated clusters show unique biobank specializations")
        print(f"   - Distance reflects semantic dissimilarity")
        print(f"   - Check semantic_overlap_analysis.csv for quantified overlaps")
        
        return supplementary_df
        
    except Exception as e:
        logger.error(f"Error in enhanced pipeline: {e}")
        raise

if __name__ == "__main__":
    # Run the enhanced version with Figure 3B
    supplementary_table = main_enhanced()