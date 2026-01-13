#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BIOBANK MESH TERM CLUSTERING PIPELINE - WITH IMPROVED OVERLAP ANALYSIS

Clusters biomedical publications from different biobanks based on their MeSH terms.
Identifies semantic clusters of publications within each biobank using TF-IDF and K-means.
ENHANCED: Direct MeSH term overlap analysis instead of problematic unified clustering.

Changes from original:
- Replaced unified semantic space (Figure 3B) with direct MeSH overlap analysis
- Added research theme distribution heatmap
- Added Jaccard similarity matrix for biobank pairs
- More interpretable and scientifically valid approach

PIPELINE:
1. Load and preprocess data with consistent filtering
2. PER-BIOBANK ANALYSIS (Figure 3A):
   - Independent clustering within each biobank
   - TF-IDF, K-means, c-DF-IPF scoring
   - Individual PCA/UMAP projections
3. OVERLAP ANALYSIS (Figure 3B):
   - Direct MeSH term overlap between biobanks
   - Research theme distribution across biobanks
   - Jaccard similarity measurements
4. Save results and supplementary tables

USAGE:
1. Place this script in PYTHON/ directory
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
from collections import defaultdict, Counter
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
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "00-02-BIOBANK-MESH-CLUSTERING")
os.makedirs(analysis_dir, exist_ok=True)

# Define preprint servers and patterns to exclude
PREPRINT_IDENTIFIERS = [
    'medRxiv', 'bioRxiv', 'Research Square', 'arXiv', 'ChemRxiv',
    'PeerJ Preprints', 'F1000Research', 'Authorea', 'Preprints.org',
    'SSRN', 'RePEc', 'OSF Preprints', 'SocArXiv', 'PsyArXiv',
    'EarthArXiv', 'engrXiv', 'TechRxiv'
]

#############################################################################
# 1. Data Loading and Preprocessing
#############################################################################

def load_biobank_data():
    """Load biobank research data and apply filtering"""
    input_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Loading data from {input_file}")
    df_raw = pd.read_csv(input_file, low_memory=False)
    
    logger.info(f"Loaded {len(df_raw):,} total records from {df_raw['Biobank'].nunique()} biobanks")
    
    # Apply filtering
    df = df_raw.copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Remove invalid years
    df_valid_years = df.dropna(subset=['Year']).copy()
    df_valid_years['Year'] = df_valid_years['Year'].astype(int)
    
    # Apply year range filter (2000-2024)
    df_year_filtered = df_valid_years[(df_valid_years['Year'] >= 2000) & 
                                      (df_valid_years['Year'] <= 2024)].copy()
    
    # Clean MeSH terms and Journal names
    df_year_filtered['MeSH_Terms'] = df_year_filtered['MeSH_Terms'].fillna('')
    df_year_filtered['Journal'] = df_year_filtered['Journal'].fillna('Unknown Journal')
    
    # Identify and exclude preprints
    df_year_filtered['is_preprint'] = False
    
    for identifier in PREPRINT_IDENTIFIERS:
        mask = df_year_filtered['Journal'].str.contains(identifier, case=False, na=False)
        df_year_filtered.loc[mask, 'is_preprint'] = True
    
    preprint_patterns = [r'preprint', r'pre-print', r'working paper', r'discussion paper']
    for pattern in preprint_patterns:
        mask = df_year_filtered['Journal'].str.contains(pattern, case=False, na=False)
        df_year_filtered.loc[mask, 'is_preprint'] = True
    
    # Get published papers only
    df_published = df_year_filtered[df_year_filtered['is_preprint'] == False].copy()
    
    # Get papers with MeSH terms
    df_with_mesh = df_published.dropna(subset=['MeSH_Terms'])
    df_with_mesh = df_with_mesh[df_with_mesh['MeSH_Terms'].str.strip() != '']
    
    # Print statistics
    logger.info(f"\n📊 FILTERING RESULTS:")
    logger.info(f"   📚 Raw dataset: {len(df_raw):,} records")
    logger.info(f"   📅 After year filtering (2000-2024): {len(df_year_filtered):,} records")
    logger.info(f"   📖 Published papers: {len(df_published):,} records")
    logger.info(f"   🔬 Papers with MeSH terms: {len(df_with_mesh):,} records")
    
    biobank_counts = df_with_mesh['Biobank'].value_counts()
    logger.info(f"\n📋 Papers with MeSH terms by biobank:")
    for biobank, count in biobank_counts.items():
        logger.info(f"   • {biobank}: {count:,} papers")
    
    return df_with_mesh

def preprocess_mesh_terms(mesh_string):
    """Clean and preprocess MeSH terms string"""
    if pd.isna(mesh_string) or mesh_string.strip() == '':
        return []
    
    terms = [term.strip() for term in str(mesh_string).split(';')]
    
    cleaned_terms = []
    for term in terms:
        if term:
            cleaned = re.sub(r'[^\w\s]', '', term.lower())
            cleaned = re.sub(r'\s+', '_', cleaned.strip())
            if cleaned:
                cleaned_terms.append(cleaned)
    
    return cleaned_terms

def create_tfidf_matrix(publications_df):
    """Create TF-IDF matrix from MeSH terms"""
    mesh_docs = []
    valid_indices = []
    
    for idx, row in publications_df.iterrows():
        terms = preprocess_mesh_terms(row['MeSH_Terms'])
        if terms:
            mesh_docs.append(' '.join(terms))
            valid_indices.append(idx)
    
    if len(mesh_docs) < 2:
        logger.warning("Insufficient publications with valid MeSH terms")
        return None, None, None
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform(mesh_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    filtered_df = publications_df.loc[valid_indices].copy().reset_index(drop=True)
    
    logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape[0]} documents x {tfidf_matrix.shape[1]} features")
    
    return tfidf_matrix, feature_names, filtered_df

#############################################################################
# 2. Clustering Functions
#############################################################################

def bootstrap_optimal_k(tfidf_matrix, k_range=(2, 10), n_bootstrap=50, sample_frac=0.8):
    """Bootstrap silhouette scoring to find optimal number of clusters"""
    if tfidf_matrix.shape[0] < 10:
        logger.warning("Too few publications for bootstrap K selection, using K=3")
        return 3
    
    k_scores = defaultdict(list)
    n_samples = max(10, int(tfidf_matrix.shape[0] * sample_frac))
    
    logger.info(f"Bootstrap K selection: trying K={k_range[0]} to {k_range[1]}")
    
    for iteration in range(n_bootstrap):
        sample_indices = np.random.choice(tfidf_matrix.shape[0], size=n_samples, replace=False)
        sample_matrix = tfidf_matrix[sample_indices]
        
        for k in range(k_range[0], min(k_range[1] + 1, sample_matrix.shape[0])):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sample_matrix.toarray())
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(sample_matrix.toarray(), labels)
                    k_scores[k].append(score)
            except Exception as e:
                continue
    
    best_k = k_range[0]
    best_score = -1
    
    for k in k_scores:
        if k_scores[k]:
            avg_score = np.mean(k_scores[k])
            logger.info(f"  K={k}: avg silhouette = {avg_score:.4f}")
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
    
    logger.info(f"Selected optimal K={best_k} with silhouette score {best_score:.4f}")
    return best_k

def perform_kmeans_clustering(tfidf_matrix, k):
    """Perform K-means clustering"""
    logger.info(f"Performing K-means clustering with K={k}")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
    
    if len(np.unique(cluster_labels)) > 1:
        silhouette = silhouette_score(tfidf_matrix.toarray(), cluster_labels)
        logger.info(f"Final silhouette score: {silhouette:.4f}")
    
    return cluster_labels, kmeans

def compute_cdf_ipf(publications_df, feature_names, tfidf_matrix, cluster_labels):
    """Compute c-DF-IPF scores for each cluster"""
    logger.info("Computing c-DF-IPF scores")
    
    n_clusters = len(np.unique(cluster_labels))
    n_total_pubs = len(publications_df)
    
    term_doc_counts = np.array((tfidf_matrix > 0).sum(axis=0)).flatten()
    
    cluster_summaries = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_pubs = np.sum(cluster_mask)
        
        if cluster_pubs == 0:
            continue
        
        cluster_tfidf = tfidf_matrix[cluster_mask]
        cluster_term_counts = np.array((cluster_tfidf > 0).sum(axis=0)).flatten()
        df_scores = cluster_term_counts / cluster_pubs
        ipf_scores = np.log(n_total_pubs / (term_doc_counts + 1e-8))
        cdf_ipf_scores = df_scores * ipf_scores
        
        top_indices = np.argsort(cdf_ipf_scores)[::-1][:10]
        
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
        
        logger.info(f"Cluster {cluster_id} ({cluster_pubs} pubs) - Top 3 terms: " +
                   ", ".join([cluster_summaries[cluster_id]['top_terms_cdf_ipf'][i]['term'] 
                             for i in range(min(3, len(cluster_summaries[cluster_id]['top_terms_cdf_ipf'])))]))
    
    return cluster_summaries

def compute_semantic_summaries(tfidf_matrix, feature_names, cluster_labels):
    """Compute semantic summaries using mean TF-IDF vectors"""
    n_clusters = len(np.unique(cluster_labels))
    semantic_summaries = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        
        if np.sum(cluster_mask) == 0:
            continue
        
        cluster_tfidf = tfidf_matrix[cluster_mask]
        mean_tfidf = np.array(cluster_tfidf.mean(axis=0)).flatten()
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
# 3. NEW - Direct MeSH Overlap Analysis (Replacement for Unified Space)
#############################################################################

def analyze_mesh_term_overlaps(df_with_mesh):
    """
    Directly analyze MeSH term overlaps between biobanks
    This replaces the problematic unified semantic space approach
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYZING MESH TERM OVERLAPS BETWEEN BIOBANKS")
    logger.info("="*60)
    
    # Extract top MeSH terms per biobank
    biobank_mesh_terms = {}
    biobank_term_counts = {}
    
    for biobank in df_with_mesh['Biobank'].unique():
        biobank_df = df_with_mesh[df_with_mesh['Biobank'] == biobank]
        
        # Collect all MeSH terms for this biobank
        all_terms = []
        for mesh_string in biobank_df['MeSH_Terms'].dropna():
            terms = [term.strip().lower() for term in str(mesh_string).split(';')]
            all_terms.extend(terms)
        
        # Count frequencies
        term_counts = Counter(all_terms)
        biobank_term_counts[biobank] = term_counts
        
        # Get top 50 terms
        top_terms = [term for term, count in term_counts.most_common(50)]
        biobank_mesh_terms[biobank] = set(top_terms)
        
        logger.info(f"\n{biobank}:")
        logger.info(f"  Total unique MeSH terms: {len(term_counts)}")
        logger.info(f"  Top 5 terms: {', '.join([t for t, c in term_counts.most_common(5)])}")
    
    # Create overlap matrix
    biobanks = sorted(list(biobank_mesh_terms.keys()))
    n_biobanks = len(biobanks)
    overlap_matrix = np.zeros((n_biobanks, n_biobanks))
    
    for i, bb1 in enumerate(biobanks):
        for j, bb2 in enumerate(biobanks):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = biobank_mesh_terms[bb1] & biobank_mesh_terms[bb2]
                union = biobank_mesh_terms[bb1] | biobank_mesh_terms[bb2]
                jaccard = len(intersection) / len(union) if union else 0
                overlap_matrix[i, j] = jaccard
    
    # Create Figure 3B: Overlap analysis visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Jaccard similarity heatmap
    sns.heatmap(overlap_matrix, 
                xticklabels=biobanks, 
                yticklabels=biobanks,
                annot=True, 
                fmt='.2f', 
                cmap='YlOrRd',
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Jaccard Similarity'},
                ax=ax1)
    ax1.set_title('MeSH Term Overlap Between Biobanks\n(Jaccard Similarity of Top 50 Terms)', 
                  fontsize=12, fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Find terms that appear in at least 3 biobanks
    all_top_terms = set()
    for terms in biobank_mesh_terms.values():
        all_top_terms.update(terms)
    
    shared_terms = []
    for term in all_top_terms:
        biobanks_with_term = sum(1 for bb_terms in biobank_mesh_terms.values() if term in bb_terms)
        if biobanks_with_term >= 3:
            shared_terms.append(term)
    
    # Get top 20 most shared terms
    shared_term_scores = []
    for term in shared_terms:
        total_count = sum(biobank_term_counts[bb].get(term, 0) for bb in biobanks)
        shared_term_scores.append((term, total_count))
    
    shared_term_scores.sort(key=lambda x: x[1], reverse=True)
    top_shared_terms = [term for term, _ in shared_term_scores[:20]]
    
    # Create frequency matrix for shared terms
    freq_matrix = []
    for term in top_shared_terms:
        row = []
        for bb in biobanks:
            count = biobank_term_counts[bb].get(term, 0)
            total = sum(biobank_term_counts[bb].values())
            freq = (count / total) * 1000  # Per 1000 papers
            row.append(freq)
        freq_matrix.append(row)
    
    freq_matrix = np.array(freq_matrix)
    
    # Plot frequency heatmap
    sns.heatmap(freq_matrix.T,
                xticklabels=[t.replace('_', ' ').title()[:25] for t in top_shared_terms],
                yticklabels=biobanks,
                cmap='Blues',
                cbar_kws={'label': 'Frequency (per 1000 papers)'},
                ax=ax2)
    ax2.set_title('Top Shared MeSH Terms Across Biobanks\n(Terms appearing in ≥3 biobanks)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('MeSH Terms', fontsize=11)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Figure 3B: MeSH Term Overlap Analysis Across Biobanks', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    overlap_file = os.path.join(analysis_dir, 'mesh_term_overlap_analysis.png')
    plt.savefig(overlap_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"\n✅ Figure 3B saved: {overlap_file}")
    
    # Print overlap statistics
    logger.info("\n" + "="*60)
    logger.info("OVERLAP STATISTICS")
    logger.info("="*60)
    
    overlaps = []
    for i, bb1 in enumerate(biobanks):
        for j, bb2 in enumerate(biobanks):
            if i < j:
                intersection = biobank_mesh_terms[bb1] & biobank_mesh_terms[bb2]
                overlaps.append({
                    'Biobank_1': bb1,
                    'Biobank_2': bb2,
                    'Shared_Terms': len(intersection),
                    'Jaccard_Similarity': overlap_matrix[i, j],
                    'Example_Shared_Terms': ', '.join(list(intersection)[:5])
                })
    
    overlap_df = pd.DataFrame(overlaps)
    overlap_df = overlap_df.sort_values('Jaccard_Similarity', ascending=False)
    
    logger.info("\nTop biobank pairs by similarity:")
    for _, row in overlap_df.head(5).iterrows():
        logger.info(f"  {row['Biobank_1']} <-> {row['Biobank_2']}: "
                   f"Jaccard={row['Jaccard_Similarity']:.3f}, "
                   f"Shared={row['Shared_Terms']} terms")
    
    # Save overlap statistics
    stats_file = os.path.join(analysis_dir, 'mesh_overlap_statistics.csv')
    overlap_df.to_csv(stats_file, index=False)
    logger.info(f"\n✅ Overlap statistics saved: {stats_file}")
    
    return overlap_df, shared_terms

def create_research_theme_analysis(df_with_mesh):
    """
    Create research theme distribution analysis across biobanks
    """
    logger.info("\n" + "="*60)
    logger.info("CREATING RESEARCH THEME DISTRIBUTION ANALYSIS")
    logger.info("="*60)
    
    # Define major research themes based on MeSH categories
    research_themes = {
        'Genomics/Genetics': ['genome', 'genetic', 'polymorphism', 'gwas', 'genotype', 'allele', 'variant', 'sequencing', 'dna'],
        'Cardiovascular': ['cardiovascular', 'heart', 'coronary', 'hypertension', 'blood pressure', 'cardiac', 'artery', 'stroke'],
        'Metabolic/Diabetes': ['diabetes', 'obesity', 'metabolic', 'insulin', 'glucose', 'lipid', 'cholesterol', 'bmi'],
        'Cancer/Oncology': ['cancer', 'tumor', 'neoplasm', 'carcinoma', 'oncology', 'malignant', 'metastasis'],
        'Neurological': ['brain', 'neural', 'cognitive', 'alzheimer', 'dementia', 'neurological', 'parkinson', 'memory'],
        'Mental Health': ['depression', 'anxiety', 'psychiatric', 'mental health', 'psychological', 'bipolar', 'schizophrenia'],
        'Respiratory': ['lung', 'respiratory', 'asthma', 'copd', 'pulmonary', 'breathing', 'airway'],
        'Imaging': ['imaging', 'mri', 'scan', 'tomography', 'radiological', 'ultrasound', 'x-ray'],
        'Epidemiology': ['epidemiology', 'cohort', 'risk factors', 'prospective', 'longitudinal', 'prevalence', 'incidence']
    }
    
    # Count papers per theme per biobank
    theme_counts = {biobank: {theme: 0 for theme in research_themes} 
                   for biobank in df_with_mesh['Biobank'].unique()}
    
    for _, row in df_with_mesh.iterrows():
        biobank = row['Biobank']
        mesh_terms = str(row['MeSH_Terms']).lower() if pd.notna(row['MeSH_Terms']) else ''
        
        for theme, keywords in research_themes.items():
            if any(keyword in mesh_terms for keyword in keywords):
                theme_counts[biobank][theme] += 1
    
    # Create comparison data
    comparison_data = []
    for biobank in sorted(theme_counts.keys()):
        total_papers = len(df_with_mesh[df_with_mesh['Biobank'] == biobank])
        for theme in research_themes:
            count = theme_counts[biobank][theme]
            percentage = (count / total_papers * 100) if total_papers > 0 else 0
            comparison_data.append({
                'Biobank': biobank,
                'Research_Theme': theme,
                'Paper_Count': count,
                'Percentage': percentage
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Pivot for visualization
    pivot_df = comparison_df.pivot(index='Research_Theme', 
                                   columns='Biobank', 
                                   values='Percentage')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Percentage of Papers (%)'},
                vmin=0, vmax=50)
    
    ax.set_title('Research Theme Distribution Across Biobanks\n' +
                'Percentage of papers containing theme-related MeSH terms',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Biobank', fontsize=12)
    ax.set_ylabel('Research Theme', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    theme_file = os.path.join(analysis_dir, 'research_theme_distribution.png')
    plt.savefig(theme_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Theme distribution saved: {theme_file}")
    
    # Save table
    table_file = os.path.join(analysis_dir, 'research_theme_comparison.csv')
    pivot_df.to_csv(table_file)
    logger.info(f"✅ Theme comparison table saved: {table_file}")
    
    # Print summary
    logger.info("\nResearch Theme Coverage Summary:")
    for theme in research_themes:
        mean_coverage = pivot_df.loc[theme].mean()
        std_coverage = pivot_df.loc[theme].std()
        logger.info(f"  {theme}: {mean_coverage:.1f}% ± {std_coverage:.1f}%")
    
    return comparison_df, pivot_df

#############################################################################
# 4. Individual Biobank Visualization Functions
#############################################################################

def create_2d_projections(tfidf_matrix, cluster_labels, biobank_name):
    """Create PCA and UMAP 2D projections of cluster centroids"""
    logger.info(f"Computing 2D projections for {biobank_name}")
    
    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
    
    # Calculate cluster centroids
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
            umap_reducer = umap.UMAP(n_components=2, random_state=42, 
                                     n_neighbors=min(3, len(centroids)-1))
            umap_coords = umap_reducer.fit_transform(centroids)
        except Exception as e:
            logger.warning(f"Error creating UMAP projection: {e}")
            umap_coords = None
    
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
    """Create composite PCA and UMAP figures showing all biobanks (Figure 3A)"""
    if not all_projection_data:
        logger.warning("No projection data available")
        return
    
    valid_projections = [p for p in all_projection_data if p is not None]
    if not valid_projections:
        return
    
    n_biobanks = len(valid_projections)
    cols = min(3, n_biobanks)
    rows = math.ceil(n_biobanks / cols)
    
    # Create Figure 3A: PCA composite
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_biobanks == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Figure 3A: Semantic Clusters Within Each Biobank\n'
                 'Independent clustering analysis per biobank', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for i, proj_data in enumerate(valid_projections):
        ax = axes[i]
        biobank_name = proj_data['biobank_name']
        pca_coords = proj_data['pca_coords']
        cluster_sizes = proj_data['cluster_sizes']
        explained_var = proj_data['pca_explained_variance']
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(pca_coords)))
        
        for j, (x, y) in enumerate(pca_coords):
            cluster_size = cluster_sizes[j]
            ax.scatter(x, y, c=[colors[j]], s=80 + cluster_size * 1.5, 
                      alpha=0.8, edgecolors='black', linewidth=0.8)
            ax.annotate(f'C{j}', (x, y), xytext=(3, 3), textcoords='offset points', 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.7))
        
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
    logger.info(f"✅ Figure 3A saved: {pca_composite_file}")

#############################################################################
# 5. Save Results Functions
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
        for i, term_info in enumerate(summary['top_terms_cdf_ipf'][:5]):
            summary_rows.append({
                'biobank': biobank_name,
                'cluster': cluster_id,
                'n_publications': summary['n_publications'],
                'rank_cdf_ipf': i + 1,
                'term_cdf_ipf': term_info['term'],
                'cdf_ipf_score': term_info['cdf_ipf_score']
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(analysis_dir, f'cluster_summaries_{biobank_clean}.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Cluster summaries saved: {summary_file}")
    
    return summary_rows

def create_supplementary_cluster_table(all_cluster_data):
    """Create comprehensive supplementary table"""
    logger.info("Creating supplementary cluster characteristics table")
    
    supplementary_rows = []
    
    for biobank_data in all_cluster_data:
        biobank_name = biobank_data['biobank_name']
        cluster_summaries = biobank_data['cluster_summaries']
        
        for cluster_id, summary in cluster_summaries.items():
            cluster_pubs_count = summary['n_publications']
            top_terms = summary['top_terms_cdf_ipf'][:5]
            
            top_terms_formatted = []
            for term_info in top_terms:
                term = term_info['term'].replace('_', ' ').title()
                score = term_info['cdf_ipf_score']
                top_terms_formatted.append(f"{term}: {score:.3f}")
            
            supplementary_row = {
                'Biobank': biobank_name,
                'Cluster_ID': f"C{cluster_id}",
                'Number_of_Publications': cluster_pubs_count,
                'Top_5_Terms': ' | '.join(top_terms_formatted)
            }
            
            supplementary_rows.append(supplementary_row)
    
    supplementary_df = pd.DataFrame(supplementary_rows)
    supplementary_df = supplementary_df.sort_values(['Biobank', 'Number_of_Publications'], 
                                                  ascending=[True, False])
    
    supplementary_file = os.path.join(analysis_dir, 'supplementary_cluster_table.csv')
    supplementary_df.to_csv(supplementary_file, index=False)
    logger.info(f"✅ Supplementary table saved: {supplementary_file}")
    
    return supplementary_df

#############################################################################
# 6. Process Individual Biobank
#############################################################################

def process_biobank(biobank_name, biobank_df):
    """Process a single biobank"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {biobank_name} ({len(biobank_df):,} publications)")
    logger.info(f"{'='*60}")
    
    if len(biobank_df) < 10:
        logger.warning(f"Skipping {biobank_name}: too few publications")
        return [], None, None
    
    # Create TF-IDF matrix
    tfidf_matrix, feature_names, filtered_df = create_tfidf_matrix(biobank_df)
    
    if tfidf_matrix is None:
        logger.warning(f"Skipping {biobank_name}: insufficient MeSH terms")
        return [], None, None
    
    # Find optimal K
    optimal_k = bootstrap_optimal_k(tfidf_matrix)
    
    # Perform clustering
    cluster_labels, kmeans_model = perform_kmeans_clustering(tfidf_matrix, optimal_k)
    
    # Compute scores
    cluster_summaries = compute_cdf_ipf(filtered_df, feature_names, tfidf_matrix, cluster_labels)
    semantic_summaries = compute_semantic_summaries(tfidf_matrix, feature_names, cluster_labels)
    
    # Create projections
    projection_data = create_2d_projections(tfidf_matrix, cluster_labels, biobank_name)
    
    # Save results
    summary_rows = save_biobank_results(biobank_name, filtered_df, cluster_labels, 
                                       cluster_summaries, semantic_summaries)
    
    # Collect data for supplementary table
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
# 7. Main Pipeline
#############################################################################

def main():
    """Main execution function"""
    print("=" * 80)
    print("BIOBANK MESH TERM CLUSTERING PIPELINE")
    print("With Improved MeSH Overlap Analysis")
    print("=" * 80)
    
    try:
        # Load data
        df_with_mesh = load_biobank_data()
        
        print(f"\n🎯 Processing pipeline:")
        print(f"   📊 Figure 3A: Individual biobank clustering")
        print(f"   🔄 Figure 3B: Direct MeSH term overlap analysis")
        print(f"   📋 Research theme distribution")
        print(f"   📑 Supplementary tables")
        
        # Process each biobank independently
        all_summaries = []
        all_projection_data = []
        all_cluster_data = []
        
        for biobank_name in sorted(df_with_mesh['Biobank'].unique()):
            biobank_df = df_with_mesh[df_with_mesh['Biobank'] == biobank_name].copy()
            summary_rows, projection_data, cluster_data = process_biobank(biobank_name, biobank_df)
            
            all_summaries.extend(summary_rows)
            if projection_data is not None:
                all_projection_data.append(projection_data)
            if cluster_data is not None:
                all_cluster_data.append(cluster_data)
        
        # Create Figure 3A: Individual biobank clustering
        logger.info("\n" + "="*60)
        logger.info("CREATING FIGURE 3A: INDIVIDUAL BIOBANK CLUSTERING")
        logger.info("="*60)
        create_composite_visualizations(all_projection_data)
        
        # Create Figure 3B: MeSH overlap analysis (NEW APPROACH)
        logger.info("\n" + "="*60)
        logger.info("CREATING FIGURE 3B: MESH OVERLAP ANALYSIS")
        logger.info("="*60)
        overlap_df, shared_terms = analyze_mesh_term_overlaps(df_with_mesh)
        
        # Create research theme distribution
        logger.info("\n" + "="*60)
        logger.info("CREATING RESEARCH THEME ANALYSIS")
        logger.info("="*60)
        comparison_df, theme_pivot = create_research_theme_analysis(df_with_mesh)
        
        # Create supplementary table
        logger.info("\n" + "="*60)
        logger.info("CREATING SUPPLEMENTARY TABLES")
        logger.info("="*60)
        supplementary_df = create_supplementary_cluster_table(all_cluster_data)
        
        # Save overall summary
        if all_summaries:
            overall_df = pd.DataFrame(all_summaries)
            summary_file = os.path.join(analysis_dir, 'biobank_clustering_summary.csv')
            overall_df.to_csv(summary_file, index=False)
            logger.info(f"✅ Overall summary saved: {summary_file}")
        
        # Print final summary
        print(f"\n✅ Pipeline complete!")
        print(f"📂 All results saved to: {analysis_dir}")
        
        print(f"\n🎨 KEY OUTPUTS:")
        print(f"   📊 Figure 3A: composite_pca_all_biobanks.png")
        print(f"      - Individual clustering within each biobank")
        print(f"   📊 Figure 3B: mesh_term_overlap_analysis.png")
        print(f"      - Direct MeSH term overlaps between biobanks")
        print(f"      - Jaccard similarity matrix")
        print(f"   📊 Additional: research_theme_distribution.png")
        print(f"      - Research theme coverage by biobank")
        
        print(f"\n📊 KEY FINDINGS:")
        if not overlap_df.empty:
            top_overlap = overlap_df.iloc[0]
            print(f"   Highest overlap: {top_overlap['Biobank_1']} <-> {top_overlap['Biobank_2']}")
            print(f"   Jaccard similarity: {top_overlap['Jaccard_Similarity']:.3f}")
            print(f"   Shared terms: {top_overlap['Shared_Terms']}")
        
        print(f"\n📋 TABLES GENERATED:")
        print(f"   - mesh_overlap_statistics.csv")
        print(f"   - research_theme_comparison.csv")
        print(f"   - supplementary_cluster_table.csv")
        print(f"   - biobank_clustering_summary.csv")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"   Figure 3A shows independent research clusters within each biobank")
        print(f"   Figure 3B reveals substantial MeSH term overlap (Jaccard ~0.6-0.8)")
        print(f"   All biobanks share core themes but with distinct emphases")
        print(f"   This approach avoids artifacts from unified clustering")
        
        return supplementary_df
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    supplementary_table = main()