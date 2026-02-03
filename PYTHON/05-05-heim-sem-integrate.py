#!/usr/bin/env python3
"""
HEIM SEMANTIC INTEGRATION - FINAL INTEGRATION
==============================================

Merges semantic metrics with HEIM v6 biobank and clinical trial data
to create unified 3-dimensional equity analysis.

INTEGRATION:
1. Load biobank metrics (Discovery dimension)
2. Load clinical trial metrics (Translation dimension)
3. Load semantic metrics (Knowledge dimension)
4. Merge on GBD disease taxonomy
5. Compute unified equity scores
6. Generate manuscript-ready tables

OUTPUTS:
- DATA/05-SEMANTIC/heim_integrated_metrics.json
- DATA/05-SEMANTIC/heim_integrated_metrics.csv
- ANALYSIS/05-04-HEIM-SEM-FIGURES/table_integrated_scores.csv
- ANALYSIS/05-04-HEIM-SEM-FIGURES/table_top_neglected.csv

USAGE:
    python 05-05-heim-sem-integrate.py

REQUIREMENTS:
    pip install pandas numpy
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
ANALYSIS_DIR = BASE_DIR / "ANALYSIS" / "05-04-HEIM-SEM-FIGURES"
LOGS_DIR = BASE_DIR / "LOGS"

# Input files
BHEM_METRICS_FILE = DATA_DIR / "bhem_metrics.json"
BHEM_DISEASE_FILE = DATA_DIR / "bhem_disease_metrics.csv"
CT_METRICS_FILE = DATA_DIR / "heim_ct_metrics.json"
CT_DISEASE_FILE = DATA_DIR / "heim_ct_disease_equity.csv"
SEM_METRICS_FILE = SEMANTIC_DIR / "heim_sem_metrics.json"

# Output files
INTEGRATED_JSON = SEMANTIC_DIR / "heim_integrated_metrics.json"
INTEGRATED_CSV = SEMANTIC_DIR / "heim_integrated_metrics.csv"

# Weights for unified score (configurable)
DEFAULT_WEIGHTS = {
    "discovery": 0.33,    # Biobank/Gap Score
    "translation": 0.33,  # Clinical trials
    "knowledge": 0.34     # Semantic isolation
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"05-05-integrate-{timestamp}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("heim_integrate")
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
# DATA LOADING
# =============================================================================

def normalize_disease_name(name: str) -> str:
    """Normalize disease name for matching across data sources."""
    n = name.lower()
    n = n.replace("_", " ").replace("-", " ").replace("/", " ")
    # Remove commas and everything after for short-form matching
    n = ' '.join(n.split())  # collapse whitespace
    return n.strip()

def load_biobank_metrics(logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load HEIM biobank metrics."""
    metrics_dict = {}

    # Try JSON first
    if BHEM_METRICS_FILE.exists():
        try:
            with open(BHEM_METRICS_FILE, 'r') as f:
                metrics_dict = json.load(f)
            logger.info(f"    Loaded biobank JSON metrics")
        except Exception as e:
            logger.warning(f"    Failed to load biobank JSON: {e}")

    # Load CSV for detailed disease data
    df = None
    if BHEM_DISEASE_FILE.exists():
        try:
            df = pd.read_csv(BHEM_DISEASE_FILE)
            df['disease_normalized'] = df['Condition'].apply(normalize_disease_name) if 'Condition' in df.columns else df.iloc[:, 0].apply(normalize_disease_name)
            logger.info(f"    Loaded biobank CSV: {len(df)} diseases")
        except Exception as e:
            logger.warning(f"    Failed to load biobank CSV: {e}")

    return df, metrics_dict

def load_clinical_trial_metrics(logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load HEIM clinical trial metrics."""
    metrics_dict = {}

    if CT_METRICS_FILE.exists():
        try:
            with open(CT_METRICS_FILE, 'r') as f:
                metrics_dict = json.load(f)
            logger.info(f"    Loaded CT JSON metrics")
        except Exception as e:
            logger.warning(f"    Failed to load CT JSON: {e}")

    df = None
    if CT_DISEASE_FILE.exists():
        try:
            df = pd.read_csv(CT_DISEASE_FILE)
            # Try to find disease column
            disease_col = None
            for col in ['disease', 'Disease', 'condition', 'Condition', 'gbd_cause']:
                if col in df.columns:
                    disease_col = col
                    break

            if disease_col:
                df['disease_normalized'] = df[disease_col].apply(normalize_disease_name)
                logger.info(f"    Loaded CT CSV: {len(df)} diseases")
            else:
                logger.warning(f"    CT CSV has no disease column: {df.columns.tolist()}")
        except Exception as e:
            logger.warning(f"    Failed to load CT CSV: {e}")

    return df, metrics_dict

def load_semantic_metrics(logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load semantic metrics."""
    metrics_dict = {}

    if not SEM_METRICS_FILE.exists():
        logger.error("    Semantic metrics file not found!")
        return None, {}

    try:
        with open(SEM_METRICS_FILE, 'r') as f:
            metrics_dict = json.load(f)

        # Convert to DataFrame
        records = []
        for disease, data in metrics_dict.get('metrics', {}).items():
            records.append({
                'disease': disease,
                'disease_normalized': normalize_disease_name(disease),
                'n_papers': data.get('n_papers', 0),
                'sii': data.get('sii'),
                'ktp': data.get('ktp'),
                'rcc': data.get('rcc'),
                'mean_drift': data.get('temporal_drift', {}).get('mean_drift')
            })

        df = pd.DataFrame(records)
        logger.info(f"    Loaded semantic metrics: {len(df)} diseases")
        return df, metrics_dict

    except Exception as e:
        logger.error(f"    Failed to load semantic metrics: {e}")
        return None, {}

# =============================================================================
# METRIC INTEGRATION
# =============================================================================

def compute_unified_score(
    gap_score: Optional[float],
    ct_equity: Optional[float],
    sii: Optional[float],
    weights: Dict[str, float]
) -> Optional[float]:
    """
    Compute unified equity score from three dimensions.

    Higher score = more neglected/under-researched
    """
    values = []
    weight_sum = 0

    # Discovery (Gap Score) - higher = more neglected
    if gap_score is not None and not np.isnan(gap_score):
        # Normalize to 0-1 (assuming Gap Score is 0-100)
        values.append(gap_score / 100 * weights['discovery'])
        weight_sum += weights['discovery']

    # Translation (CT Equity) - need to invert if higher = better
    if ct_equity is not None and not np.isnan(ct_equity):
        # Assuming ct_equity is a positive metric where higher = better represented
        # Invert: 1 - ct_equity/max to make higher = more neglected
        values.append((1 - ct_equity) * weights['translation'])
        weight_sum += weights['translation']

    # Knowledge (SII) - higher = more isolated = more neglected
    if sii is not None and not np.isnan(sii):
        # SII typically 0-1, higher = more isolated
        values.append(sii * weights['knowledge'])
        weight_sum += weights['knowledge']

    if weight_sum == 0:
        return None

    # Normalize by actual weights used
    return sum(values) / weight_sum * 100  # Scale to 0-100

def integrate_metrics(
    biobank_df: Optional[pd.DataFrame],
    ct_df: Optional[pd.DataFrame],
    semantic_df: pd.DataFrame,
    weights: Dict[str, float],
    logger: logging.Logger
) -> pd.DataFrame:
    """Integrate all metrics into unified DataFrame."""
    logger.info("\n  Integrating metrics...")

    # Start with semantic data as base
    integrated = semantic_df.copy()

    # Merge biobank data
    if biobank_df is not None:
        # Find matching columns
        gap_col = None
        for col in ['Gap_Score', 'gap_score', 'GapScore']:
            if col in biobank_df.columns:
                gap_col = col
                break

        if gap_col:
            merge_cols = ['disease_normalized', gap_col]
            biobank_subset = biobank_df[merge_cols].drop_duplicates(subset=['disease_normalized'])
            integrated = integrated.merge(
                biobank_subset,
                on='disease_normalized',
                how='left'
            )
            integrated.rename(columns={gap_col: 'gap_score'}, inplace=True)

            # Substring fallback for unmatched diseases (e.g., "tracheal" in "tracheal, bronchus, and lung cancer")
            unmatched = integrated[integrated['gap_score'].isna()].index
            if len(unmatched) > 0:
                biobank_lookup = dict(zip(biobank_subset['disease_normalized'], biobank_subset[gap_col]))
                for idx in unmatched:
                    sem_name = integrated.loc[idx, 'disease_normalized']
                    for bhem_name, gap_val in biobank_lookup.items():
                        if sem_name in bhem_name or bhem_name in sem_name:
                            integrated.loc[idx, 'gap_score'] = gap_val
                            break

            logger.info(f"    Merged biobank data: {integrated['gap_score'].notna().sum()} matches")
        else:
            integrated['gap_score'] = np.nan
            logger.warning(f"    No Gap Score column found in biobank data")
    else:
        integrated['gap_score'] = np.nan

    # Merge CT data
    if ct_df is not None:
        # Find equity-related columns
        equity_col = None
        for col in ['equity_score', 'Equity_Score', 'enrollment_equity', 'equity']:
            if col in ct_df.columns:
                equity_col = col
                break

        if equity_col:
            merge_cols = ['disease_normalized', equity_col]
            ct_subset = ct_df[merge_cols].drop_duplicates(subset=['disease_normalized'])
            integrated = integrated.merge(
                ct_subset,
                on='disease_normalized',
                how='left'
            )
            integrated.rename(columns={equity_col: 'ct_equity'}, inplace=True)
            logger.info(f"    Merged CT data: {integrated['ct_equity'].notna().sum()} matches")
        else:
            integrated['ct_equity'] = np.nan
            logger.warning(f"    No equity column found in CT data")
    else:
        integrated['ct_equity'] = np.nan

    # Compute unified score
    integrated['unified_score'] = integrated.apply(
        lambda row: compute_unified_score(
            row.get('gap_score'),
            row.get('ct_equity'),
            row.get('sii'),
            weights
        ),
        axis=1
    )

    # Compute dimension availability
    integrated['dimensions_available'] = integrated.apply(
        lambda row: sum([
            1 if pd.notna(row.get('gap_score')) else 0,
            1 if pd.notna(row.get('ct_equity')) else 0,
            1 if pd.notna(row.get('sii')) else 0
        ]),
        axis=1
    )

    logger.info(f"    Unified scores computed: {integrated['unified_score'].notna().sum()}")

    return integrated

# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_outputs(
    integrated_df: pd.DataFrame,
    weights: Dict[str, float],
    logger: logging.Logger
):
    """Generate all output files."""
    logger.info("\n  Generating outputs...")

    # Prepare clean DataFrame for export
    export_cols = [
        'disease', 'n_papers', 'gap_score', 'ct_equity', 'sii', 'ktp', 'rcc',
        'mean_drift', 'unified_score', 'dimensions_available'
    ]
    export_df = integrated_df[[c for c in export_cols if c in integrated_df.columns]].copy()
    export_df = export_df.sort_values('unified_score', ascending=False)

    # 1. Save integrated CSV
    export_df.to_csv(INTEGRATED_CSV, index=False)
    logger.info(f"    Saved: {INTEGRATED_CSV.relative_to(BASE_DIR)}")

    # 2. Save integrated JSON
    json_output = {
        "generated_at": datetime.now().isoformat(),
        "weights": weights,
        "n_diseases": len(export_df),
        "n_with_unified_score": int(export_df['unified_score'].notna().sum()),
        "summary": {
            "unified_score": {
                "mean": float(export_df['unified_score'].mean()) if export_df['unified_score'].notna().any() else None,
                "std": float(export_df['unified_score'].std()) if export_df['unified_score'].notna().any() else None,
                "min": float(export_df['unified_score'].min()) if export_df['unified_score'].notna().any() else None,
                "max": float(export_df['unified_score'].max()) if export_df['unified_score'].notna().any() else None
            }
        },
        "diseases": export_df.to_dict(orient='records')
    }

    with open(INTEGRATED_JSON, 'w') as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"    Saved: {INTEGRATED_JSON.relative_to(BASE_DIR)}")

    # 3. Generate manuscript tables
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Table: Integrated scores (all diseases)
    table_all = export_df[['disease', 'gap_score', 'ct_equity', 'sii', 'unified_score']].copy()
    table_all.columns = ['Disease', 'Gap Score', 'CT Equity', 'Semantic Isolation', 'Unified Score']
    table_all['Disease'] = table_all['Disease'].str.replace('_', ' ')
    table_all.to_csv(ANALYSIS_DIR / "table_integrated_scores.csv", index=False)
    logger.info(f"    Saved: table_integrated_scores.csv")

    # Table: Top 20 most neglected
    top_neglected = export_df.nlargest(20, 'unified_score')[
        ['disease', 'n_papers', 'gap_score', 'sii', 'unified_score']
    ].copy()
    top_neglected.columns = ['Disease', 'Papers', 'Gap Score', 'Semantic Isolation', 'Unified Score']
    top_neglected['Disease'] = top_neglected['Disease'].str.replace('_', ' ')
    top_neglected.to_csv(ANALYSIS_DIR / "table_top_neglected.csv", index=False)
    logger.info(f"    Saved: table_top_neglected.csv")

    return export_df

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HEIM Semantic Pipeline - Final Integration"
    )
    parser.add_argument(
        "--discovery-weight", type=float, default=DEFAULT_WEIGHTS['discovery'],
        help="Weight for discovery (biobank) dimension"
    )
    parser.add_argument(
        "--translation-weight", type=float, default=DEFAULT_WEIGHTS['translation'],
        help="Weight for translation (CT) dimension"
    )
    parser.add_argument(
        "--knowledge-weight", type=float, default=DEFAULT_WEIGHTS['knowledge'],
        help="Weight for knowledge (semantic) dimension"
    )

    args = parser.parse_args()

    # Set weights
    weights = {
        "discovery": args.discovery_weight,
        "translation": args.translation_weight,
        "knowledge": args.knowledge_weight
    }

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    logger = setup_logging()

    print("\n" + "=" * 70)
    print(" HEIM SEMANTIC PIPELINE - FINAL INTEGRATION")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Dimension Weights:")
    print(f"    Discovery (Biobank): {weights['discovery']:.1%}")
    print(f"    Translation (CT):    {weights['translation']:.1%}")
    print(f"    Knowledge (Semantic): {weights['knowledge']:.1%}")

    # Load all data sources
    print("\n" + "-" * 70)
    logger.info("  Loading data sources...")

    biobank_df, biobank_dict = load_biobank_metrics(logger)
    ct_df, ct_dict = load_clinical_trial_metrics(logger)
    semantic_df, semantic_dict = load_semantic_metrics(logger)

    if semantic_df is None or semantic_df.empty:
        logger.error("  Cannot proceed without semantic metrics!")
        sys.exit(1)

    # Integrate metrics
    integrated_df = integrate_metrics(
        biobank_df, ct_df, semantic_df, weights, logger
    )

    # Generate outputs
    export_df = generate_outputs(integrated_df, weights, logger)

    # Summary statistics
    print("\n" + "=" * 70)
    print(" INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Summary:")
    print(f"    Total diseases: {len(export_df)}")
    print(f"    With unified score: {export_df['unified_score'].notna().sum()}")

    if export_df['unified_score'].notna().any():
        print(f"\n  Unified Score Distribution:")
        print(f"    Mean: {export_df['unified_score'].mean():.1f}")
        print(f"    Min:  {export_df['unified_score'].min():.1f}")
        print(f"    Max:  {export_df['unified_score'].max():.1f}")

        print(f"\n  Top 5 Most Neglected (by Unified Score):")
        top5 = export_df.nlargest(5, 'unified_score')
        for _, row in top5.iterrows():
            disease = row['disease'].replace('_', ' ')
            score = row['unified_score']
            print(f"    {disease[:35]:35} {score:.1f}")

    print(f"\n  Output files:")
    print(f"    {INTEGRATED_CSV.relative_to(BASE_DIR)}")
    print(f"    {INTEGRATED_JSON.relative_to(BASE_DIR)}")
    print(f"\n  HEIM Semantic Integration Pipeline Complete!")

if __name__ == "__main__":
    main()
