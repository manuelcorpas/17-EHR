#!/usr/bin/env python3
"""
HEIM SEMANTIC INTEGRATION - METRIC COMPUTATION
===============================================

Computes semantic equity metrics from PubMedBERT embeddings.

METRICS:
1. Semantic Isolation Index (SII)
   - Average cosine distance to k-nearest neighbors
   - Higher = more isolated, less connected research

2. Knowledge Transfer Potential (KTP)
   - Cross-disease centroid similarity
   - Higher = more potential for research spillover

3. Research Clustering Coefficient (RCC)
   - Within-disease embedding variance
   - Higher = more dispersed research topics

4. Temporal Semantic Drift
   - Cosine distance between yearly centroids
   - Measures how research focus evolves over time

OUTPUTS:
- DATA/05-SEMANTIC/heim_sem_metrics.json
- ANALYSIS/05-03-SEMANTIC-METRICS/disease_metrics.csv
- ANALYSIS/05-03-SEMANTIC-METRICS/cross_disease_similarity.csv
- ANALYSIS/05-03-SEMANTIC-METRICS/temporal_drift.csv

USAGE:
    python 05-03-heim-sem-compute-metrics.py

REQUIREMENTS:
    pip install numpy pandas h5py scikit-learn tqdm
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
EMBEDDINGS_DIR = SEMANTIC_DIR / "EMBEDDINGS"
ANALYSIS_DIR = BASE_DIR / "ANALYSIS" / "05-03-SEMANTIC-METRICS"
LOGS_DIR = BASE_DIR / "LOGS"

# Marker files
EMBED_COMPLETE = SEMANTIC_DIR / ".embed_complete"

# Output files
METRICS_FILE = SEMANTIC_DIR / "heim_sem_metrics.json"

# Metric parameters
K_NEIGHBORS = 100  # For SII computation
MIN_PAPERS_FOR_METRICS = 20  # Minimum papers for reliable metrics

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"05-03-metrics-{timestamp}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("heim_metrics")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# =============================================================================
# EMBEDDING LOADING
# =============================================================================

def load_embeddings(disease_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings from HDF5 file."""
    h5_file = disease_dir / "embeddings.h5"

    with h5py.File(h5_file, 'r') as f:
        embeddings = f['embeddings'][:]
        pmids = f['pmids'][:]
        years = f['years'][:]

    # Decode bytes to strings
    pmids = np.array([p.decode() if isinstance(p, bytes) else p for p in pmids])
    years = np.array([y.decode() if isinstance(y, bytes) else y for y in years])

    return embeddings, pmids, years

def load_all_embeddings(embeddings_dir: Path, logger: logging.Logger) -> Dict:
    """Load embeddings for all diseases."""
    disease_data = {}

    disease_dirs = sorted([
        d for d in embeddings_dir.iterdir()
        if d.is_dir() and (d / "embeddings.h5").exists()
    ])

    logger.info(f"  Loading embeddings from {len(disease_dirs)} diseases...")

    for disease_dir in tqdm(disease_dirs, desc="  Loading"):
        disease_name = disease_dir.name
        try:
            embeddings, pmids, years = load_embeddings(disease_dir)

            if len(embeddings) >= MIN_PAPERS_FOR_METRICS:
                disease_data[disease_name] = {
                    "embeddings": embeddings,
                    "pmids": pmids,
                    "years": years,
                    "centroid": np.mean(embeddings, axis=0),
                    "n_papers": len(embeddings)
                }
            else:
                logger.warning(
                    f"  {disease_name}: Only {len(embeddings)} papers, "
                    f"skipping (min: {MIN_PAPERS_FOR_METRICS})"
                )
        except Exception as e:
            logger.error(f"  Failed to load {disease_name}: {e}")

    logger.info(f"  Loaded {len(disease_data)} diseases with sufficient data")
    return disease_data

# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_semantic_isolation_index(
    disease_data: Dict,
    k: int = K_NEIGHBORS,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Compute Semantic Isolation Index (SII) for each disease.

    SII measures how isolated a disease's research is from all other research.
    Higher values indicate more isolated, potentially neglected areas.
    """
    if logger:
        logger.info("\n  Computing Semantic Isolation Index (SII)...")

    # Collect all centroids
    diseases = list(disease_data.keys())
    centroids = np.array([disease_data[d]["centroid"] for d in diseases])

    # Compute pairwise distances
    distances = cosine_distances(centroids)

    sii_scores = {}
    for i, disease in enumerate(tqdm(diseases, desc="    SII", leave=False)):
        # Get distances to other diseases (exclude self)
        other_distances = np.delete(distances[i], i)

        # SII = mean distance to k nearest neighbors
        k_actual = min(k, len(other_distances))
        nearest_k = np.partition(other_distances, k_actual - 1)[:k_actual]
        sii_scores[disease] = float(np.mean(nearest_k))

    return sii_scores

def compute_knowledge_transfer_potential(
    disease_data: Dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Compute Knowledge Transfer Potential (KTP) for each disease.

    KTP measures how much a disease's research could benefit from
    or contribute to other disease research.
    """
    if logger:
        logger.info("\n  Computing Knowledge Transfer Potential (KTP)...")

    diseases = list(disease_data.keys())
    centroids = np.array([disease_data[d]["centroid"] for d in diseases])

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(centroids)

    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=diseases,
        columns=diseases
    )

    # KTP = mean similarity to top 10% most similar diseases (excluding self)
    ktp_scores = {}
    top_k = max(1, len(diseases) // 10)

    for i, disease in enumerate(diseases):
        sims = similarity_matrix[i].copy()
        sims[i] = -1  # Exclude self
        top_similar = np.partition(sims, -top_k)[-top_k:]
        ktp_scores[disease] = float(np.mean(top_similar))

    return ktp_scores, similarity_df

def compute_research_clustering_coefficient(
    disease_data: Dict,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Compute Research Clustering Coefficient (RCC) for each disease.

    RCC measures the dispersion of research within a disease.
    Higher values indicate more diverse research approaches.
    """
    if logger:
        logger.info("\n  Computing Research Clustering Coefficient (RCC)...")

    rcc_scores = {}

    for disease in tqdm(disease_data.keys(), desc="    RCC", leave=False):
        embeddings = disease_data[disease]["embeddings"]
        centroid = disease_data[disease]["centroid"]

        # RCC = mean cosine distance from centroid
        distances = cosine_distances(embeddings, centroid.reshape(1, -1))
        rcc_scores[disease] = float(np.mean(distances))

    return rcc_scores

def compute_temporal_drift(
    disease_data: Dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """
    Compute Temporal Semantic Drift for each disease.

    Measures how research focus has evolved over time by tracking
    centroid movement between years.
    """
    if logger:
        logger.info("\n  Computing Temporal Semantic Drift...")

    drift_data = {}
    drift_records = []

    for disease in tqdm(disease_data.keys(), desc="    Drift", leave=False):
        embeddings = disease_data[disease]["embeddings"]
        years = disease_data[disease]["years"]

        # Group by year
        year_centroids = {}
        unique_years = sorted(set(y for y in years if y != "unknown"))

        for year in unique_years:
            year_mask = years == year
            if np.sum(year_mask) >= 5:  # Minimum papers per year
                year_embeddings = embeddings[year_mask]
                year_centroids[year] = np.mean(year_embeddings, axis=0)

        if len(year_centroids) < 2:
            continue

        # Compute year-over-year drift
        sorted_years = sorted(year_centroids.keys())
        year_drifts = []

        for i in range(1, len(sorted_years)):
            prev_year = sorted_years[i - 1]
            curr_year = sorted_years[i]

            drift = cosine_distances(
                year_centroids[prev_year].reshape(1, -1),
                year_centroids[curr_year].reshape(1, -1)
            )[0, 0]

            year_drifts.append(drift)
            drift_records.append({
                "disease": disease,
                "from_year": prev_year,
                "to_year": curr_year,
                "drift": drift
            })

        # Aggregate metrics
        if year_drifts:
            drift_data[disease] = {
                "mean_drift": float(np.mean(year_drifts)),
                "max_drift": float(np.max(year_drifts)),
                "total_drift": float(np.sum(year_drifts)),
                "n_years": len(year_centroids)
            }

    drift_df = pd.DataFrame(drift_records)
    return drift_data, drift_df

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HEIM Semantic Pipeline - Metric Computation"
    )
    args = parser.parse_args()

    logger = setup_logging()

    print("\n" + "=" * 70)
    print(" HEIM SEMANTIC PIPELINE - METRIC COMPUTATION")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not EMBED_COMPLETE.exists():
        print("\n  ERROR: Embeddings not complete. Run 05-02-heim-sem-embed.py first.")
        sys.exit(1)

    # Create output directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all embeddings
    disease_data = load_all_embeddings(EMBEDDINGS_DIR, logger)

    if not disease_data:
        print("\n  ERROR: No valid embeddings found.")
        sys.exit(1)

    # Compute metrics
    print("\n" + "-" * 70)

    # 1. Semantic Isolation Index
    sii_scores = compute_semantic_isolation_index(disease_data, logger=logger)

    # 2. Knowledge Transfer Potential
    ktp_scores, similarity_df = compute_knowledge_transfer_potential(disease_data, logger=logger)

    # 3. Research Clustering Coefficient
    rcc_scores = compute_research_clustering_coefficient(disease_data, logger=logger)

    # 4. Temporal Semantic Drift
    drift_data, drift_df = compute_temporal_drift(disease_data, logger=logger)

    # Compile all metrics
    logger.info("\n  Compiling metrics...")

    all_metrics = {
        "generated_at": datetime.now().isoformat(),
        "n_diseases": len(disease_data),
        "total_papers": sum(d["n_papers"] for d in disease_data.values()),
        "metrics": {}
    }

    for disease in disease_data.keys():
        all_metrics["metrics"][disease] = {
            "n_papers": disease_data[disease]["n_papers"],
            "sii": sii_scores.get(disease),
            "ktp": ktp_scores.get(disease),
            "rcc": rcc_scores.get(disease),
            "temporal_drift": drift_data.get(disease, {})
        }

    # Add summary statistics
    all_metrics["summary"] = {
        "sii": {
            "mean": float(np.mean(list(sii_scores.values()))),
            "std": float(np.std(list(sii_scores.values()))),
            "min": float(np.min(list(sii_scores.values()))),
            "max": float(np.max(list(sii_scores.values())))
        },
        "ktp": {
            "mean": float(np.mean(list(ktp_scores.values()))),
            "std": float(np.std(list(ktp_scores.values()))),
            "min": float(np.min(list(ktp_scores.values()))),
            "max": float(np.max(list(ktp_scores.values())))
        },
        "rcc": {
            "mean": float(np.mean(list(rcc_scores.values()))),
            "std": float(np.std(list(rcc_scores.values()))),
            "min": float(np.min(list(rcc_scores.values()))),
            "max": float(np.max(list(rcc_scores.values())))
        }
    }

    # Save outputs
    logger.info("\n  Saving outputs...")

    # Main metrics file
    with open(METRICS_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"    {METRICS_FILE.relative_to(BASE_DIR)}")

    # Disease metrics CSV
    disease_df = pd.DataFrame([
        {
            "disease": d,
            "n_papers": disease_data[d]["n_papers"],
            "sii": sii_scores.get(d),
            "ktp": ktp_scores.get(d),
            "rcc": rcc_scores.get(d),
            "mean_drift": drift_data.get(d, {}).get("mean_drift")
        }
        for d in disease_data.keys()
    ])
    disease_df = disease_df.sort_values("sii", ascending=False)
    disease_csv = ANALYSIS_DIR / "disease_metrics.csv"
    disease_df.to_csv(disease_csv, index=False)
    logger.info(f"    {disease_csv.relative_to(BASE_DIR)}")

    # Cross-disease similarity matrix
    similarity_csv = ANALYSIS_DIR / "cross_disease_similarity.csv"
    similarity_df.to_csv(similarity_csv)
    logger.info(f"    {similarity_csv.relative_to(BASE_DIR)}")

    # Temporal drift data
    if not drift_df.empty:
        drift_csv = ANALYSIS_DIR / "temporal_drift.csv"
        drift_df.to_csv(drift_csv, index=False)
        logger.info(f"    {drift_csv.relative_to(BASE_DIR)}")

    # Summary
    print("\n" + "=" * 70)
    print(" METRIC COMPUTATION COMPLETE")
    print("=" * 70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Diseases analyzed: {len(disease_data)}")
    print(f"  Total papers: {all_metrics['total_papers']:,}")
    print(f"\n  Metric Ranges:")
    print(f"    SII: {all_metrics['summary']['sii']['min']:.4f} - {all_metrics['summary']['sii']['max']:.4f}")
    print(f"    KTP: {all_metrics['summary']['ktp']['min']:.4f} - {all_metrics['summary']['ktp']['max']:.4f}")
    print(f"    RCC: {all_metrics['summary']['rcc']['min']:.4f} - {all_metrics['summary']['rcc']['max']:.4f}")
    print(f"\n  Next step: python 05-04-heim-sem-generate-figures.py")

if __name__ == "__main__":
    main()
