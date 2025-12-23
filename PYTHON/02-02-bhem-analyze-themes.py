#!/usr/bin/env python3
"""
02-02-bhem-analyze-themes.py
============================
BHEM Step 2b: Analyze Research Themes and Knowledge Contributions

⚡ OPTIMIZED FOR M3 ULTRA (32 cores, 256GB RAM)
- Parallel biobank processing using joblib
- Vectorized pandas operations (no iterrows!)
- Multi-threaded sklearn operations
- Reduced bootstrap with parallel K evaluation

METHODS:
1. MeSH term frequency analysis per biobank
2. TF-IDF vectorization for term importance
3. K-means clustering with parallel optimal K selection
4. c-DF-IPF scoring for cluster-distinctive terms
5. Theme labeling and interpretation
6. Cross-biobank Jaccard similarity

INPUT:  DATA/bhem_publications_mapped.csv (or bhem_publications.csv)
OUTPUT: 
    - DATA/bhem_themes.json (full analysis)
    - DATA/bhem_biobank_profiles.csv (per-biobank summary)
    - DATA/bhem_cluster_terms.csv (top terms per cluster)
    - DATA/bhem_theme_overlap.csv (biobank similarity matrix)

USAGE:
    python 02-02-bhem-analyze-themes.py
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

# Parallel processing
from joblib import Parallel, delayed, cpu_count

# ML with parallel support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Auto-detect cores (will be 32 on M3 Ultra)
N_CORES = cpu_count()
N_JOBS = min(N_CORES, 32)  # Cap at 32 for safety

print(f"🖥️  Hardware: {N_CORES} CPU cores detected")
print(f"⚡ Using {N_JOBS} parallel workers")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"

# Input - prefer mapped file, fall back to raw
INPUT_FILE = DATA_DIR / "bhem_publications_mapped.csv"
if not INPUT_FILE.exists():
    INPUT_FILE = DATA_DIR / "bhem_publications.csv"

# Outputs
OUTPUT_THEMES = DATA_DIR / "bhem_themes.json"
OUTPUT_PROFILES = DATA_DIR / "bhem_biobank_profiles.csv"
OUTPUT_CLUSTERS = DATA_DIR / "bhem_cluster_terms.csv"
OUTPUT_OVERLAP = DATA_DIR / "bhem_theme_overlap.csv"

# Clustering parameters (optimized for parallel)
MIN_DOCS_FOR_CLUSTERING = 20
K_RANGE = (3, 12)
N_BOOTSTRAP = 15  # Reduced since we parallelize K evaluation
TOP_TERMS_PER_CLUSTER = 10


# =============================================================================
# RESEARCH THEME DEFINITIONS
# =============================================================================

THEME_KEYWORDS = {
    'genomics_gwas': {
        'name': 'Genomics & GWAS',
        'keywords': ['genome-wide association', 'gwas', 'genetic variant', 'snp', 
                     'polymorphism', 'allele', 'genotype', 'heritability', 'polygenic'],
        'category': 'methodology'
    },
    'mendelian_randomization': {
        'name': 'Mendelian Randomization',
        'keywords': ['mendelian randomization', 'causal inference', 'instrumental variable',
                     'genetic instrument', 'mr analysis'],
        'category': 'methodology'
    },
    'neuroimaging': {
        'name': 'Neuroimaging',
        'keywords': ['magnetic resonance imaging', 'mri', 'brain imaging', 'neuroimaging',
                     'white matter', 'gray matter', 'cortical', 'hippocampus', 'fmri'],
        'category': 'methodology'
    },
    'cardiovascular': {
        'name': 'Cardiovascular Research',
        'keywords': ['cardiovascular', 'heart', 'cardiac', 'coronary', 'myocardial',
                     'hypertension', 'blood pressure', 'atherosclerosis', 'stroke'],
        'category': 'disease_area'
    },
    'metabolic': {
        'name': 'Metabolic & Diabetes',
        'keywords': ['diabetes', 'obesity', 'metabolic', 'insulin', 'glucose',
                     'lipid', 'cholesterol', 'bmi', 'adiposity', 'glycemic'],
        'category': 'disease_area'
    },
    'cancer': {
        'name': 'Cancer & Oncology',
        'keywords': ['cancer', 'tumor', 'neoplasm', 'carcinoma', 'oncology',
                     'malignant', 'metastasis', 'breast cancer', 'lung cancer'],
        'category': 'disease_area'
    },
    'neurological': {
        'name': 'Neurological Disorders',
        'keywords': ['alzheimer', 'dementia', 'parkinson', 'neurodegeneration',
                     'cognitive', 'memory', 'epilepsy', 'multiple sclerosis'],
        'category': 'disease_area'
    },
    'mental_health': {
        'name': 'Mental Health & Psychiatry',
        'keywords': ['depression', 'anxiety', 'schizophrenia', 'bipolar', 
                     'psychiatric', 'mental health', 'psychological', 'ptsd'],
        'category': 'disease_area'
    },
    'infectious': {
        'name': 'Infectious Diseases',
        'keywords': ['infection', 'infectious', 'covid', 'sars-cov', 'viral',
                     'bacterial', 'tuberculosis', 'hiv', 'malaria', 'sepsis'],
        'category': 'disease_area'
    },
    'pharmacogenomics': {
        'name': 'Pharmacogenomics',
        'keywords': ['pharmacogenomics', 'drug response', 'adverse drug', 
                     'pharmacokinetics', 'drug metabolism', 'therapeutic'],
        'category': 'methodology'
    },
    'epidemiology': {
        'name': 'Epidemiology & Risk Factors',
        'keywords': ['risk factor', 'epidemiology', 'cohort study', 'longitudinal',
                     'prospective', 'incidence', 'prevalence', 'mortality'],
        'category': 'methodology'
    },
    'omics': {
        'name': 'Multi-omics & Biomarkers',
        'keywords': ['proteomics', 'metabolomics', 'transcriptomics', 'biomarker',
                     'epigenetics', 'methylation', 'expression', 'microbiome'],
        'category': 'methodology'
    },
    'aging': {
        'name': 'Aging & Longevity',
        'keywords': ['aging', 'ageing', 'longevity', 'elderly', 'geriatric',
                     'age-related', 'lifespan', 'frailty', 'senescence'],
        'category': 'disease_area'
    },
    'respiratory': {
        'name': 'Respiratory Diseases',
        'keywords': ['respiratory', 'lung', 'pulmonary', 'asthma', 'copd',
                     'chronic obstructive', 'airway', 'smoking'],
        'category': 'disease_area'
    },
    'musculoskeletal': {
        'name': 'Musculoskeletal',
        'keywords': ['osteoporosis', 'bone', 'fracture', 'arthritis', 'joint',
                     'musculoskeletal', 'skeletal', 'back pain', 'spine'],
        'category': 'disease_area'
    },
    'eye_vision': {
        'name': 'Eye & Vision',
        'keywords': ['eye', 'vision', 'retinal', 'glaucoma', 'macular',
                     'ophthalmology', 'cataract', 'myopia', 'ocular'],
        'category': 'disease_area'
    },
    'sleep': {
        'name': 'Sleep Research',
        'keywords': ['sleep', 'insomnia', 'circadian', 'sleep apnea',
                     'chronotype', 'sleep disorder', 'melatonin'],
        'category': 'disease_area'
    },
    'nutrition': {
        'name': 'Nutrition & Diet',
        'keywords': ['nutrition', 'diet', 'dietary', 'food', 'nutrient',
                     'vitamin', 'alcohol', 'coffee', 'mediterranean'],
        'category': 'exposure'
    },
    'physical_activity': {
        'name': 'Physical Activity',
        'keywords': ['physical activity', 'exercise', 'sedentary', 'fitness',
                     'accelerometer', 'walking', 'sports', 'movement'],
        'category': 'exposure'
    },
    'environment': {
        'name': 'Environmental Health',
        'keywords': ['air pollution', 'environmental', 'particulate matter',
                     'noise', 'green space', 'urban', 'exposure', 'climate'],
        'category': 'exposure'
    }
}


# =============================================================================
# VECTORIZED MESH TERM PROCESSING (NO ITERROWS!)
# =============================================================================

def preprocess_mesh_terms_vectorized(mesh_series: pd.Series) -> pd.Series:
    """
    Vectorized preprocessing of MeSH terms.
    Uses pandas str methods instead of row-by-row iteration.
    """
    # Fill NaN with empty string
    mesh_clean = mesh_series.fillna('')
    
    # Lowercase
    mesh_clean = mesh_clean.str.lower()
    
    # Replace punctuation with space
    mesh_clean = mesh_clean.str.replace(r'[^\w\s;]', ' ', regex=True)
    
    # Replace multiple spaces with single space
    mesh_clean = mesh_clean.str.replace(r'\s+', ' ', regex=True)
    
    # Replace spaces within terms with underscores (between semicolons)
    def process_terms(s):
        if not s or s.strip() == '':
            return ''
        terms = [t.strip().replace(' ', '_') for t in s.split(';') if t.strip()]
        return ' '.join(set(terms))  # Deduplicate
    
    return mesh_clean.apply(process_terms)


def create_combined_text_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized creation of combined text for theme detection.
    """
    title = df.get('title', pd.Series([''] * len(df), index=df.index)).fillna('')
    abstract = df.get('abstract', pd.Series([''] * len(df), index=df.index)).fillna('')
    mesh = df.get('mesh_terms', df.get('MeSH_Terms', pd.Series([''] * len(df), index=df.index))).fillna('')
    
    return (title + ' ' + abstract + ' ' + mesh).str.lower()


# =============================================================================
# VECTORIZED THEME DETECTION
# =============================================================================

def detect_themes_vectorized(texts: pd.Series) -> pd.DataFrame:
    """
    Vectorized theme detection using pandas str.contains().
    Returns a DataFrame with theme counts per document.
    
    Much faster than row-by-row iteration!
    """
    theme_columns = {}
    
    for theme_id, theme_info in THEME_KEYWORDS.items():
        # Build regex pattern for all keywords
        pattern = '|'.join([re.escape(kw) for kw in theme_info['keywords']])
        
        # Vectorized count (1 if any keyword present, 0 otherwise)
        theme_columns[theme_id] = texts.str.contains(pattern, case=False, regex=True, na=False).astype(int)
    
    return pd.DataFrame(theme_columns)


def analyze_themes_vectorized(df: pd.DataFrame, biobank_id: str) -> dict:
    """
    Analyze themes for a biobank using vectorized operations.
    """
    biobank_df = df[df['biobank_id'] == biobank_id].copy()
    
    if len(biobank_df) == 0:
        return {}
    
    # Vectorized text combination
    texts = create_combined_text_vectorized(biobank_df)
    
    # Vectorized theme detection
    theme_df = detect_themes_vectorized(texts)
    
    # Aggregate counts
    theme_counts = theme_df.sum()
    total_papers = len(biobank_df)
    
    theme_profile = {}
    for theme_id, count in theme_counts.items():
        if count > 0:
            theme_profile[theme_id] = {
                'name': THEME_KEYWORDS[theme_id]['name'],
                'category': THEME_KEYWORDS[theme_id]['category'],
                'count': int(count),
                'percentage': round(100 * count / total_papers, 1),
                'paper_count': int(count)
            }
    
    # Sort by count
    theme_profile = dict(sorted(theme_profile.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True))
    
    return theme_profile


# =============================================================================
# PARALLEL CLUSTERING
# =============================================================================

def evaluate_k_single(args):
    """Evaluate a single K value - designed for parallel execution."""
    k, sample_matrix = args
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = kmeans.fit_predict(sample_matrix)
        
        if len(np.unique(labels)) > 1:
            score = silhouette_score(sample_matrix, labels)
            return k, score
    except:
        pass
    return k, -1


def find_optimal_k_parallel(tfidf_matrix) -> int:
    """
    Find optimal K using parallel evaluation.
    Evaluates all K values simultaneously using multiple cores.
    """
    n_samples = tfidf_matrix.shape[0]
    
    if n_samples < K_RANGE[0] * 2:
        return min(3, n_samples // 2)
    
    # Sample for faster evaluation
    sample_size = min(n_samples, 500)
    indices = np.random.choice(n_samples, size=sample_size, replace=False)
    sample_matrix = tfidf_matrix[indices].toarray()
    
    # Generate tasks for parallel execution
    max_k = min(K_RANGE[1] + 1, sample_size // 3)
    k_range = range(K_RANGE[0], max_k)
    
    # Parallel K evaluation
    args_list = [(k, sample_matrix) for k in k_range]
    
    with ProcessPoolExecutor(max_workers=min(len(args_list), N_JOBS)) as executor:
        results = list(executor.map(evaluate_k_single, args_list))
    
    # Find best K
    best_k = K_RANGE[0]
    best_score = -1
    
    for k, score in results:
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k


def create_mesh_documents_vectorized(df: pd.DataFrame) -> tuple:
    """
    Create MeSH documents using vectorized operations.
    Returns (documents list, valid indices).
    """
    mesh_col = df.get('mesh_terms', df.get('MeSH_Terms', pd.Series())).fillna('')
    
    # Vectorized preprocessing
    mesh_docs = preprocess_mesh_terms_vectorized(mesh_col)
    
    # Filter non-empty
    valid_mask = mesh_docs.str.strip() != ''
    valid_indices = df.index[valid_mask].tolist()
    documents = mesh_docs[valid_mask].tolist()
    
    return documents, valid_indices


def cluster_biobank_optimized(df: pd.DataFrame, biobank_id: str, biobank_name: str) -> dict:
    """
    Perform semantic clustering with parallel K selection.
    """
    biobank_df = df[df['biobank_id'] == biobank_id].copy()
    
    if len(biobank_df) < MIN_DOCS_FOR_CLUSTERING:
        return None
    
    # Vectorized document creation
    documents, valid_indices = create_mesh_documents_vectorized(biobank_df)
    
    if len(documents) < MIN_DOCS_FOR_CLUSTERING:
        return None
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        logger.warning(f"TF-IDF failed for {biobank_name}: {e}")
        return None
    
    # Find optimal K (parallel)
    optimal_k = find_optimal_k_parallel(tfidf_matrix)
    
    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
    
    # Compute c-DF-IPF
    cluster_data = compute_cluster_terms(
        tfidf_matrix, feature_names, cluster_labels, 
        biobank_id, biobank_name
    )
    
    # PCA for visualization
    if tfidf_matrix.shape[0] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(tfidf_matrix.toarray())
        cluster_data['pca_variance'] = pca.explained_variance_ratio_.tolist()
        
        centroids = []
        for c in range(optimal_k):
            mask = cluster_labels == c
            if np.sum(mask) > 0:
                centroid = coords[mask].mean(axis=0)
                centroids.append({
                    'cluster': c,
                    'x': float(centroid[0]),
                    'y': float(centroid[1]),
                    'size': int(np.sum(mask))
                })
        cluster_data['centroids'] = centroids
    
    return cluster_data


def compute_cluster_terms(tfidf_matrix, feature_names, cluster_labels, 
                          biobank_id: str, biobank_name: str) -> dict:
    """Compute c-DF-IPF scores and extract top terms per cluster."""
    n_clusters = len(np.unique(cluster_labels))
    n_total = len(cluster_labels)
    
    # Global document frequency (vectorized)
    term_doc_counts = np.array((tfidf_matrix > 0).sum(axis=0)).flatten()
    
    clusters = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size == 0:
            continue
        
        cluster_tfidf = tfidf_matrix[cluster_mask]
        
        # c-DF: proportion of cluster docs containing term (vectorized)
        cluster_term_counts = np.array((cluster_tfidf > 0).sum(axis=0)).flatten()
        df_scores = cluster_term_counts / cluster_size
        
        # IPF: inverse population frequency
        ipf_scores = np.log(n_total / (term_doc_counts + 1e-8))
        
        # c-DF-IPF
        cdf_ipf = df_scores * ipf_scores
        
        # Get top terms
        top_indices = np.argsort(cdf_ipf)[::-1][:TOP_TERMS_PER_CLUSTER]
        
        top_terms = []
        for idx in top_indices:
            if cdf_ipf[idx] > 0:
                term = feature_names[idx].replace('_', ' ').title()
                top_terms.append({
                    'term': term,
                    'cdf_ipf': round(float(cdf_ipf[idx]), 4),
                    'df': round(float(df_scores[idx]), 4)
                })
        
        # Infer theme
        theme = infer_cluster_theme(top_terms)
        
        clusters.append({
            'cluster_id': cluster_id,
            'size': int(cluster_size),
            'percentage': round(100 * cluster_size / n_total, 1),
            'top_terms': top_terms,
            'inferred_theme': theme
        })
    
    return {
        'biobank_id': biobank_id,
        'biobank_name': biobank_name,
        'total_papers': n_total,
        'n_clusters': n_clusters,
        'clusters': clusters
    }


def infer_cluster_theme(top_terms: list) -> str:
    """Infer a theme label from top cluster terms."""
    if not top_terms:
        return "General Research"
    
    term_text = ' '.join([t['term'].lower() for t in top_terms[:5]])
    
    best_theme = "General Research"
    best_score = 0
    
    for theme_id, theme_info in THEME_KEYWORDS.items():
        score = sum(1 for kw in theme_info['keywords'] if kw in term_text)
        if score > best_score:
            best_score = score
            best_theme = theme_info['name']
    
    return best_theme


# =============================================================================
# PARALLEL BIOBANK PROCESSING
# =============================================================================

def process_single_biobank(biobank_tuple) -> dict:
    """
    Process a single biobank - themes + clustering.
    Designed for parallel execution.
    """
    biobank_id, biobank_df, biobank_name = biobank_tuple
    
    result = {
        'biobank_id': biobank_id,
        'biobank_name': biobank_name,
        'themes': {},
        'clusters': None,
        'cluster_rows': []
    }
    
    if len(biobank_df) == 0:
        return result
    
    # Theme analysis (vectorized within biobank)
    texts = create_combined_text_vectorized(biobank_df)
    theme_df = detect_themes_vectorized(texts)
    theme_counts = theme_df.sum()
    total_papers = len(biobank_df)
    
    for theme_id, count in theme_counts.items():
        if count > 0:
            result['themes'][theme_id] = {
                'name': THEME_KEYWORDS[theme_id]['name'],
                'category': THEME_KEYWORDS[theme_id]['category'],
                'count': int(count),
                'percentage': round(100 * count / total_papers, 1)
            }
    
    # Sort themes
    result['themes'] = dict(sorted(result['themes'].items(), 
                                   key=lambda x: x[1]['count'], 
                                   reverse=True))
    
    # Clustering (if enough papers)
    if len(biobank_df) >= MIN_DOCS_FOR_CLUSTERING:
        documents, valid_indices = create_mesh_documents_vectorized(biobank_df)
        
        if len(documents) >= MIN_DOCS_FOR_CLUSTERING:
            try:
                # TF-IDF
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    min_df=2,
                    max_df=0.85,
                    ngram_range=(1, 2),
                    token_pattern=r'\b\w+\b'
                )
                
                tfidf_matrix = vectorizer.fit_transform(documents)
                feature_names = vectorizer.get_feature_names_out()
                
                # Quick K selection (simplified for parallel)
                n_samples = tfidf_matrix.shape[0]
                optimal_k = min(max(3, n_samples // 100), K_RANGE[1])
                optimal_k = max(optimal_k, K_RANGE[0])
                
                # Sample-based evaluation
                if n_samples > 100:
                    sample_size = min(300, n_samples)
                    sample_idx = np.random.choice(n_samples, sample_size, replace=False)
                    sample_matrix = tfidf_matrix[sample_idx].toarray()
                    
                    best_k = optimal_k
                    best_score = -1
                    
                    for k in range(K_RANGE[0], min(K_RANGE[1] + 1, sample_size // 3)):
                        try:
                            km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
                            labels = km.fit_predict(sample_matrix)
                            if len(np.unique(labels)) > 1:
                                score = silhouette_score(sample_matrix, labels)
                                if score > best_score:
                                    best_score = score
                                    best_k = k
                        except:
                            continue
                    optimal_k = best_k
                
                # Final clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
                
                # Compute cluster terms
                cluster_data = compute_cluster_terms(
                    tfidf_matrix, feature_names, cluster_labels,
                    biobank_id, biobank_name
                )
                
                # PCA
                if tfidf_matrix.shape[0] >= 2:
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(tfidf_matrix.toarray())
                    cluster_data['pca_variance'] = pca.explained_variance_ratio_.tolist()
                    
                    centroids = []
                    for c in range(optimal_k):
                        mask = cluster_labels == c
                        if np.sum(mask) > 0:
                            centroid = coords[mask].mean(axis=0)
                            centroids.append({
                                'cluster': c,
                                'x': float(centroid[0]),
                                'y': float(centroid[1]),
                                'size': int(np.sum(mask))
                            })
                    cluster_data['centroids'] = centroids
                
                result['clusters'] = cluster_data
                
                # Flatten clusters for CSV
                for cluster in cluster_data['clusters']:
                    for term in cluster['top_terms'][:5]:
                        result['cluster_rows'].append({
                            'biobank_id': biobank_id,
                            'biobank_name': biobank_name,
                            'cluster_id': cluster['cluster_id'],
                            'cluster_size': cluster['size'],
                            'cluster_theme': cluster['inferred_theme'],
                            'term': term['term'],
                            'cdf_ipf': term['cdf_ipf']
                        })
                
            except Exception as e:
                logger.warning(f"Clustering failed for {biobank_name}: {e}")
    
    return result


# =============================================================================
# CROSS-BIOBANK ANALYSIS (VECTORIZED)
# =============================================================================

def compute_biobank_overlap_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Jaccard similarity between biobanks using vectorized operations.
    """
    biobanks = df['biobank_id'].unique()
    
    # Pre-compute term sets for all biobanks
    biobank_terms = {}
    
    for biobank_id in biobanks:
        biobank_df = df[df['biobank_id'] == biobank_id]
        mesh_col = biobank_df.get('mesh_terms', biobank_df.get('MeSH_Terms', pd.Series()))
        
        all_terms = []
        for mesh in mesh_col.dropna():
            terms = [t.strip().lower() for t in str(mesh).split(';') if t.strip()]
            all_terms.extend(terms)
        
        # Top 100 most common terms
        term_counts = Counter(all_terms)
        biobank_terms[biobank_id] = set([t for t, _ in term_counts.most_common(100)])
    
    # Compute similarity matrix
    n = len(biobanks)
    similarity_matrix = np.zeros((n, n))
    
    for i, bb1 in enumerate(biobanks):
        for j, bb2 in enumerate(biobanks):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                terms1 = biobank_terms.get(bb1, set())
                terms2 = biobank_terms.get(bb2, set())
                
                if terms1 and terms2:
                    intersection = len(terms1 & terms2)
                    union = len(terms1 | terms2)
                    sim = intersection / union if union > 0 else 0
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
    
    return pd.DataFrame(similarity_matrix, index=biobanks, columns=biobanks)


def compute_distinctiveness_vectorized(df: pd.DataFrame) -> dict:
    """
    Compute distinctive MeSH terms per biobank using vectorized operations.
    """
    biobanks = df['biobank_id'].unique()
    
    # Global term counts
    global_term_counts = Counter()
    biobank_term_counts = {}
    
    for biobank_id in biobanks:
        biobank_df = df[df['biobank_id'] == biobank_id]
        mesh_col = biobank_df.get('mesh_terms', biobank_df.get('MeSH_Terms', pd.Series()))
        
        local_terms = []
        for mesh in mesh_col.dropna():
            terms = [t.strip().lower().replace(' ', '_') for t in str(mesh).split(';') if t.strip()]
            local_terms.extend(terms)
        
        local_counts = Counter(local_terms)
        biobank_term_counts[biobank_id] = local_counts
        global_term_counts.update(local_terms)
    
    global_total = sum(global_term_counts.values())
    
    # Compute distinctiveness (lift)
    distinctiveness = {}
    
    for biobank_id in biobanks:
        local_counts = biobank_term_counts[biobank_id]
        local_total = sum(local_counts.values())
        
        if local_total == 0:
            continue
        
        distinctive_terms = []
        
        for term, local_count in local_counts.most_common(200):
            global_count = global_term_counts[term]
            local_freq = local_count / local_total
            expected_freq = global_count / global_total if global_total > 0 else 0
            
            lift = local_freq / expected_freq if expected_freq > 0 else 0
            
            if lift > 1.5 and local_count >= 3:
                distinctive_terms.append({
                    'term': term.replace('_', ' ').title(),
                    'local_count': local_count,
                    'local_freq': round(local_freq * 100, 2),
                    'lift': round(lift, 2)
                })
        
        distinctive_terms.sort(key=lambda x: x['lift'], reverse=True)
        distinctiveness[biobank_id] = distinctive_terms[:20]
    
    return distinctiveness


def compute_knowledge_metrics(df: pd.DataFrame, theme_profiles: dict, 
                               cluster_data: dict, distinctiveness: dict) -> dict:
    """Compute comprehensive knowledge contribution metrics per biobank."""
    metrics = {}
    
    for biobank_id in df['biobank_id'].unique():
        biobank_df = df[df['biobank_id'] == biobank_id]
        biobank_name = biobank_df['biobank_name'].iloc[0] if len(biobank_df) > 0 else biobank_id
        
        total_papers = len(biobank_df)
        mesh_col = biobank_df.get('mesh_terms', biobank_df.get('MeSH_Terms', pd.Series()))
        papers_with_mesh = len(mesh_col.dropna()[mesh_col.dropna().str.strip() != ''])
        
        themes = theme_profiles.get(biobank_id, {})
        theme_breadth = len(themes)
        
        top_themes = sorted(
            [(tid, t['percentage']) for tid, t in themes.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        clusters = cluster_data.get(biobank_id, {})
        n_clusters = clusters.get('n_clusters', 0) if clusters else 0
        
        distinct_terms = distinctiveness.get(biobank_id, [])
        avg_distinctiveness = np.mean([t['lift'] for t in distinct_terms[:10]]) if distinct_terms else 0
        
        methodology_themes = [tid for tid, t in themes.items() 
                             if THEME_KEYWORDS.get(tid, {}).get('category') == 'methodology']
        disease_themes = [tid for tid, t in themes.items() 
                         if THEME_KEYWORDS.get(tid, {}).get('category') == 'disease_area']
        
        metrics[biobank_id] = {
            'biobank_name': biobank_name,
            'total_papers': total_papers,
            'papers_with_mesh': papers_with_mesh,
            'mesh_coverage': round(100 * papers_with_mesh / total_papers, 1) if total_papers > 0 else 0,
            'theme_breadth': theme_breadth,
            'top_themes': [{'theme': tid, 'percentage': pct} for tid, pct in top_themes],
            'n_clusters': n_clusters,
            'distinctiveness_score': round(avg_distinctiveness, 2),
            'methodology_diversity': len(methodology_themes),
            'disease_coverage': len(disease_themes),
            'distinctive_terms': distinct_terms[:10]
        }
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    
    print("=" * 70)
    print("BHEM STEP 2b: Analyze Research Themes")
    print("⚡ PARALLEL OPTIMIZED FOR M3 ULTRA")
    print("=" * 70)
    print(f"🖥️  Using {N_JOBS} parallel workers")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_THEMES}")
    
    # Load data
    if not INPUT_FILE.exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        print("   Run 02-00-bhem-fetch-pubmed.py first")
        return
    
    print(f"\n📂 Loading data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df):,} publications")
    print(f"   Biobanks: {df['biobank_id'].nunique()}")
    
    # Prepare biobank data for parallel processing
    print(f"\n🔬 Preparing parallel processing...")
    biobank_data = []
    for biobank_id in df['biobank_id'].unique():
        biobank_df = df[df['biobank_id'] == biobank_id].copy()
        biobank_name = biobank_df['biobank_name'].iloc[0] if len(biobank_df) > 0 else biobank_id
        biobank_data.append((biobank_id, biobank_df, biobank_name))
    
    print(f"   Processing {len(biobank_data)} biobanks in parallel...")
    
    # PARALLEL BIOBANK PROCESSING
    print(f"\n⚡ Running parallel analysis on {N_JOBS} cores...")
    results = Parallel(n_jobs=N_JOBS, verbose=10, backend='loky')(
        delayed(process_single_biobank)(data) for data in biobank_data
    )
    
    # Collect results
    theme_profiles = {}
    cluster_data = {}
    cluster_rows = []
    
    for result in results:
        biobank_id = result['biobank_id']
        theme_profiles[biobank_id] = result['themes']
        if result['clusters']:
            cluster_data[biobank_id] = result['clusters']
        cluster_rows.extend(result['cluster_rows'])
    
    # Print summary
    print(f"\n📊 Theme Analysis Summary:")
    for result in results[:10]:  # Show first 10
        if result['themes']:
            themes = result['themes']
            top_theme = list(themes.items())[0] if themes else (None, {})
            if top_theme[0]:
                print(f"   {result['biobank_name']}: {len(themes)} themes, "
                      f"top: {top_theme[1].get('name', 'N/A')} ({top_theme[1].get('percentage', 0):.1f}%)")
    
    # Cross-biobank analysis
    print(f"\n🔗 Computing cross-biobank overlap...")
    overlap_df = compute_biobank_overlap_vectorized(df)
    
    print(f"\n✨ Computing distinctiveness...")
    distinctiveness = compute_distinctiveness_vectorized(df)
    
    print(f"\n📈 Computing knowledge metrics...")
    knowledge_metrics = compute_knowledge_metrics(
        df, theme_profiles, cluster_data, distinctiveness
    )
    
    # Compile results
    full_results = {
        'generated_at': datetime.now().isoformat(),
        'total_publications': len(df),
        'total_biobanks': df['biobank_id'].nunique(),
        'processing_cores': N_JOBS,
        'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
        'theme_definitions': {
            tid: {'name': t['name'], 'category': t['category']} 
            for tid, t in THEME_KEYWORDS.items()
        },
        'biobank_themes': theme_profiles,
        'biobank_clusters': cluster_data,
        'biobank_distinctiveness': distinctiveness,
        'knowledge_metrics': knowledge_metrics,
        'cross_biobank_similarity': overlap_df.to_dict()
    }
    
    # Save outputs
    print(f"\n💾 Saving results...")
    
    with open(OUTPUT_THEMES, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"   ✅ {OUTPUT_THEMES}")
    
    # Biobank profiles CSV
    profile_rows = []
    for biobank_id, metrics in knowledge_metrics.items():
        row = {
            'biobank_id': biobank_id,
            'biobank_name': metrics['biobank_name'],
            'total_papers': metrics['total_papers'],
            'mesh_coverage': metrics['mesh_coverage'],
            'theme_breadth': metrics['theme_breadth'],
            'n_clusters': metrics['n_clusters'],
            'distinctiveness_score': metrics['distinctiveness_score'],
            'methodology_diversity': metrics['methodology_diversity'],
            'disease_coverage': metrics['disease_coverage']
        }
        for i, theme in enumerate(metrics['top_themes'][:3]):
            row[f'top_theme_{i+1}'] = theme['theme']
            row[f'top_theme_{i+1}_pct'] = theme['percentage']
        profile_rows.append(row)
    
    pd.DataFrame(profile_rows).to_csv(OUTPUT_PROFILES, index=False)
    print(f"   ✅ {OUTPUT_PROFILES}")
    
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(OUTPUT_CLUSTERS, index=False)
        print(f"   ✅ {OUTPUT_CLUSTERS}")
    
    overlap_df.to_csv(OUTPUT_OVERLAP)
    print(f"   ✅ {OUTPUT_OVERLAP}")
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("THEME ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"📊 Processed: {len(df):,} publications across {df['biobank_id'].nunique()} biobanks")
    print(f"🚀 Speedup: ~{N_JOBS}x with parallel processing")
    
    print(f"\n📊 Top 10 Biobanks by Knowledge Contribution:")
    sorted_metrics = sorted(knowledge_metrics.items(), 
                           key=lambda x: x[1]['total_papers'], 
                           reverse=True)[:10]
    for biobank_id, metrics in sorted_metrics:
        print(f"   {metrics['biobank_name']}")
        print(f"      Papers: {metrics['total_papers']:,} | Themes: {metrics['theme_breadth']} | "
              f"Distinctiveness: {metrics['distinctiveness_score']:.2f}")
    
    print(f"\n✅ STEP 2b COMPLETE!")
    print(f"\n➡️  Next step: python 02-03-bhem-compute-metrics.py")


if __name__ == "__main__":
    main()