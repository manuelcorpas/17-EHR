#!/usr/bin/env python3
"""
02-03-bhem-compute-metrics.py
=============================
HEIM-Biobank v1.0: Compute Health Equity Metrics

ADAPTED to use FULL GBD registry from 02-01 (all 175+ GBD causes) instead of
hardcoded 25-disease subset. Preserves ALL sophisticated metric formulas from
Corpas et al. (2025).

Implements four core equity metrics:
1. Burden Score - Composite disease burden from GBD 2021
2. Research Gap Score - Mismatch between burden and research attention
3. Research Opportunity Score - Unrealized potential per biobank
4. Equity Alignment Score - Overall equity performance per biobank

INPUT:  
    DATA/bhem_publications_mapped.csv
    DATA/gbd_disease_registry.json (from 02-01)
    
OUTPUT: 
    DATA/bhem_metrics.json
    DATA/bhem_biobank_metrics.csv
    DATA/bhem_disease_metrics.csv

USAGE:
    python 02-03-bhem-compute-metrics.py

METHODOLOGY:
    Corpas M, et al. (2025). EHR-Linked Biobank Expansion Reveals Global
    Health Inequities. Annual Review of Biomedical Data Science.

VERSION: HEIM-Biobank v1.0
DATE: 2025-12-25
"""

import json
import logging
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Version metadata
VERSION = "HEIM-Biobank v1.0"
VERSION_DATE = "2025-12-25"
METHODOLOGY_SOURCE = "Corpas et al. 2025, Annual Review of Biomedical Data Science"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"

# Input files
INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"
INPUT_GBD_REGISTRY = DATA_DIR / "gbd_disease_registry.json"

# Output files
OUTPUT_METRICS_JSON = DATA_DIR / "bhem_metrics.json"
OUTPUT_BIOBANK_CSV = DATA_DIR / "bhem_biobank_metrics.csv"
OUTPUT_DISEASE_CSV = DATA_DIR / "bhem_disease_metrics.csv"


# =============================================================================
# GBD LEVEL 2 TO CATEGORY MAPPINGS
# Maps GBD Level 2 categories to simplified disease categories for gap scoring
# =============================================================================

GBD_LEVEL2_TO_CATEGORY = {
    # Infectious diseases - stricter thresholds
    "HIV/AIDS and sexually transmitted infections": "Infectious",
    "HIV/AIDS and STIs": "Infectious",
    "Respiratory infections and tuberculosis": "Infectious",
    "Respiratory infections": "Infectious",
    "Enteric infections": "Infectious",
    "Other infectious diseases": "Infectious",
    "Tuberculosis": "Infectious",
    
    # Neglected diseases - strictest thresholds
    "Neglected tropical diseases and malaria": "Neglected",
    "NTDs and malaria": "Neglected",
    
    # Maternal and child - priority diseases
    "Maternal and neonatal disorders": "Maternal/Child",
    "Maternal disorders": "Maternal/Child",
    "Neonatal disorders": "Maternal/Child",
    "Nutritional deficiencies": "Nutritional",
    
    # Non-communicable diseases
    "Cardiovascular diseases": "Cardiovascular",
    "Neoplasms": "Cancer",
    "Chronic respiratory diseases": "Respiratory",
    "Digestive diseases": "Digestive",
    "Neurological disorders": "Neurological",
    "Mental disorders": "Mental Health",
    "Substance use disorders": "Mental Health",
    "Diabetes and kidney diseases": "Metabolic",
    "Skin and subcutaneous diseases": "Other NCD",
    "Sense organ diseases": "Other NCD",
    "Musculoskeletal disorders": "Musculoskeletal",
    "Other non-communicable diseases": "Other NCD",
    
    # Injuries
    "Transport injuries": "Injuries",
    "Unintentional injuries": "Injuries",
    "Self-harm and interpersonal violence": "Injuries",
}

# Categories requiring stricter gap thresholds (from original methodology)
INFECTIOUS_CATEGORIES = {"Infectious", "Neglected"}
NEGLECTED_CATEGORIES = {"Neglected", "Maternal/Child", "Nutritional"}


# =============================================================================
# GLOBAL SOUTH CLASSIFICATION (from original)
# =============================================================================

GLOBAL_SOUTH_REGIONS = {"AFR", "SEAR", "EMR"}

GLOBAL_SOUTH_COUNTRIES = {
    # Africa
    "Uganda", "Nigeria", "Kenya", "South Africa", "Ghana", "Ethiopia", "Tanzania",
    "Rwanda", "Senegal", "Cameroon", "Egypt", "Morocco", "Tunisia",
    # Latin America
    "Mexico", "Brazil", "Argentina", "Chile", "Colombia", "Peru", "Ecuador",
    "Venezuela", "Cuba", "Costa Rica",
    # Asia
    "India", "Bangladesh", "Pakistan", "Indonesia", "Vietnam", "Philippines",
    "Thailand", "Malaysia", "Nepal", "Sri Lanka",
    # China (considered separately but included for completeness)
    "China"
}

# GBD Level 2 categories that are Global South priorities
GLOBAL_SOUTH_GBD_CATEGORIES = {
    "HIV/AIDS and sexually transmitted infections",
    "HIV/AIDS and STIs",
    "Respiratory infections and tuberculosis",
    "Respiratory infections",
    "Enteric infections",
    "Neglected tropical diseases and malaria",
    "NTDs and malaria",
    "Other infectious diseases",
    "Maternal and neonatal disorders",
    "Maternal disorders",
    "Neonatal disorders",
    "Nutritional deficiencies",
    "Tuberculosis",
}


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# All formulas from Corpas et al. (2025) preserved exactly
# =============================================================================

def compute_burden_score(dalys_millions: float, deaths_millions: float = None, 
                         prevalence_millions: float = None) -> float:
    """
    Compute composite Burden Score for a disease.
    
    ORIGINAL Formula (Corpas et al. 2025):
        Burden_Score = (0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]
    
    ADAPTED Formula (when only DALYs available):
        Burden_Score = 10 × log₁₀(DALYs_millions × 1e6 + 1)
        
    This log-scaled fallback produces comparable scores:
    - DALYs ~1M → score ~60-70
    - DALYs ~10M → score ~70-80  
    - DALYs ~100M → score ~80-90
    
    Args:
        dalys_millions: Disability-adjusted life years in millions
        deaths_millions: Annual deaths in millions (optional)
        prevalence_millions: Total cases in millions (optional)
    
    Returns:
        Composite burden score (higher = greater burden)
    """
    # If we have all three components, use original formula
    if deaths_millions is not None and prevalence_millions is not None:
        # Handle edge case: prevalence = 0
        if prevalence_millions <= 0:
            prevalence_millions = 1.0  # Use 1.0 to avoid log(0)
        
        daly_component = 0.5 * dalys_millions
        death_component = 50.0 * deaths_millions
        prevalence_component = 10.0 * math.log10(prevalence_millions)
        
        burden_score = daly_component + death_component + prevalence_component
    else:
        # Fallback: DALYs-only formula (log-scaled for comparable range)
        if dalys_millions <= 0:
            return 0.0
        
        # Convert to raw DALYs and log-scale
        dalys_raw = dalys_millions * 1_000_000
        burden_score = 10.0 * math.log10(dalys_raw + 1)
    
    return round(burden_score, 2)


def compute_research_gap_score(disease_id: str, publications: int, 
                                burden_score: float, total_dalys_millions: float,
                                global_south_priority: bool,
                                category: str) -> Tuple[float, str]:
    """
    Compute Research Gap Score for a disease (0-100 scale).
    
    Three-tier scoring system (Corpas et al. 2025):
    1. Zero-publication penalty: Gap = 95 (Critical)
    2. Threshold adjustments for infectious/neglected diseases
    3. Burden-normalized intensity for others
    
    Args:
        disease_id: Disease identifier
        publications: Number of biobank publications
        burden_score: Computed burden score
        total_dalys_millions: Total DALYs for this disease (in millions)
        global_south_priority: Whether disease is Global South priority
        category: Disease category (Infectious, Neglected, etc.)
    
    Returns:
        Tuple of (gap_score, gap_severity_category)
    """
    # Tier 1: Zero publications = Critical gap
    if publications == 0:
        gap_score = 95.0
        return gap_score, "Critical"
    
    # Tier 2: Category-specific thresholds for infectious/neglected diseases
    if category in INFECTIOUS_CATEGORIES:
        if publications < 10:
            gap_score = 90.0
        elif publications < 25:
            gap_score = 80.0
        elif publications < 50:
            gap_score = 70.0
        elif publications < 100:
            gap_score = 60.0
        else:
            # Move to burden-normalized calculation
            gap_score = None
            
    elif category in NEGLECTED_CATEGORIES:
        if publications < 10:
            gap_score = 92.0
        elif publications < 25:
            gap_score = 82.0
        elif publications < 50:
            gap_score = 72.0
        elif publications < 100:
            gap_score = 65.0
        else:
            gap_score = None
            
    else:
        gap_score = None
    
    # Tier 3: Burden-normalized intensity for other cases
    if gap_score is None:
        # Publications per million DALYs
        if total_dalys_millions > 0:
            pubs_per_million_dalys = publications / total_dalys_millions
        else:
            pubs_per_million_dalys = publications
        
        # Decile-based scoring (higher intensity = lower gap)
        if pubs_per_million_dalys >= 100:
            gap_score = 10.0
        elif pubs_per_million_dalys >= 50:
            gap_score = 20.0
        elif pubs_per_million_dalys >= 25:
            gap_score = 30.0
        elif pubs_per_million_dalys >= 10:
            gap_score = 40.0
        elif pubs_per_million_dalys >= 5:
            gap_score = 50.0
        elif pubs_per_million_dalys >= 2:
            gap_score = 60.0
        elif pubs_per_million_dalys >= 1:
            gap_score = 70.0
        elif pubs_per_million_dalys >= 0.5:
            gap_score = 80.0
        else:
            gap_score = 85.0
    
    # Global South priority penalty: +10 if under-researched
    if global_south_priority and publications < 50:
        gap_score = min(95.0, gap_score + 10.0)
    
    # Classify severity
    if gap_score > 70:
        severity = "Critical"
    elif gap_score > 50:
        severity = "High"
    elif gap_score > 30:
        severity = "Moderate"
    else:
        severity = "Low"
    
    return round(gap_score, 1), severity


def compute_research_opportunity_score(disease_publications: Dict[str, int],
                                       disease_metrics: Dict[str, Dict]) -> float:
    """
    Compute Research Opportunity Score for a biobank.
    
    Formula (Corpas et al. 2025):
        ROS_b = Σ Burden_Score(d) for diseases where Publications(b,d) ≤ 2
    
    Higher ROS indicates greater unrealized potential for equity-aligned research.
    
    Args:
        disease_publications: Dict mapping disease_id -> publication count for this biobank
        disease_metrics: Dict with burden scores for all diseases
    
    Returns:
        Research Opportunity Score (sum of burden scores for underexplored diseases)
    """
    ros = 0.0
    
    for disease_id, metrics in disease_metrics.items():
        pubs = disease_publications.get(disease_id, 0)
        
        # Include diseases with 0, 1, or 2 publications
        if pubs <= 2:
            ros += metrics['burden_score']
    
    return round(ros, 2)


def compute_equity_alignment_score(disease_publications: Dict[str, int],
                                   disease_metrics: Dict[str, Dict],
                                   total_publications: int) -> Tuple[float, str, Dict]:
    """
    Compute Equity Alignment Score for a biobank (0-100 scale).
    
    Formula (Corpas et al. 2025 / HEIM-Biobank v1.0):
        EAS = 100 - [(0.4 × Gap_Severity) + (0.3 × Burden_Miss) + (0.3 × Capacity_Penalty)]
    
    Components:
        - Gap_Severity: Weighted count of gap categories, normalized to 0-100
          (Critical=4, High=2, Moderate=1)
        - Burden_Miss: Proportion of DALYs in diseases with ≤2 publications
        - Capacity_Penalty: Inverse of publications per disease coverage
    
    Args:
        disease_publications: Dict mapping disease_id -> publication count
        disease_metrics: Dict with burden scores and gap severities
        total_publications: Total publications for this biobank
    
    Returns:
        Tuple of (eas_score, category, components_dict)
    """
    # Count gap severities for this biobank's coverage
    n_critical = 0
    n_high = 0
    n_moderate = 0
    n_low = 0
    
    missed_dalys = 0.0
    total_dalys = 0.0
    diseases_covered = 0
    
    for disease_id, metrics in disease_metrics.items():
        pubs = disease_publications.get(disease_id, 0)
        dalys = metrics['dalys_millions']
        total_dalys += dalys
        
        if pubs > 0:
            diseases_covered += 1
        
        # Count gaps based on this biobank's coverage
        if pubs == 0:
            n_critical += 1
            missed_dalys += dalys
        elif pubs <= 2:
            # Still consider as under-researched
            gap_sev = metrics['gap_severity']
            if gap_sev == "Critical":
                n_critical += 1
            elif gap_sev == "High":
                n_high += 1
            else:
                n_moderate += 1
            missed_dalys += dalys
        elif pubs <= 10:
            gap_sev = metrics['gap_severity']
            if gap_sev == "High":
                n_high += 1
            elif gap_sev == "Moderate":
                n_moderate += 1
            else:
                n_low += 1
        else:
            n_low += 1
    
    # Component 1: Gap Severity (0-100)
    # Weighted gaps: Critical=4, High=2, Moderate=1
    weighted_gaps = (4 * n_critical) + (2 * n_high) + (1 * n_moderate)
    max_possible_gaps = 4 * len(disease_metrics)  # All critical
    gap_severity_component = min(100.0, (weighted_gaps / max_possible_gaps) * 100) if max_possible_gaps > 0 else 0
    
    # Component 2: Burden Miss (0-100)
    burden_miss_component = (missed_dalys / total_dalys * 100) if total_dalys > 0 else 0
    
    # Component 3: Capacity Penalty (0-100)
    # Based on average publications per disease
    n_diseases = len(disease_metrics)
    pubs_per_disease = total_publications / n_diseases if n_diseases > 0 else 0
    capacity_penalty = 100 - min(pubs_per_disease, 100)
    
    # Compute EAS
    eas = 100 - (
        (0.4 * gap_severity_component) +
        (0.3 * burden_miss_component) +
        (0.3 * capacity_penalty)
    )
    
    # Clamp to 0-100
    eas = max(0.0, min(100.0, eas))
    
    # Categorize
    if eas >= 80:
        category = "Strong"
    elif eas >= 60:
        category = "Moderate"
    elif eas >= 40:
        category = "Weak"
    else:
        category = "Poor"
    
    components = {
        'gap_severity_component': round(gap_severity_component, 2),
        'burden_miss_component': round(burden_miss_component, 2),
        'capacity_penalty': round(capacity_penalty, 2),
        'n_critical': n_critical,
        'n_high': n_high,
        'n_moderate': n_moderate,
        'n_low': n_low,
        'diseases_covered': diseases_covered,
        'missed_dalys': round(missed_dalys, 2)
    }
    
    return round(eas, 1), category, components


def compute_equity_ratio(hic_publications: int, hic_dalys: float,
                         lmic_publications: int, lmic_dalys: float) -> float:
    """
    Compute global equity ratio comparing HIC vs LMIC research intensity.
    
    Formula (Corpas et al. 2025):
        Equity_Ratio = (Pubs_HIC / DALYs_HIC) / (Pubs_LMIC / DALYs_LMIC)
    
    Ratio > 1 indicates HIC-focused disparity.
    
    Args:
        hic_publications: Publications from HIC biobanks
        hic_dalys: Estimated DALYs in HIC regions
        lmic_publications: Publications from Global South biobanks
        lmic_dalys: Estimated DALYs in LMIC regions
    
    Returns:
        Equity ratio (>1 = HIC bias, <1 = LMIC bias, 1 = balanced)
    """
    # Avoid division by zero
    if hic_dalys <= 0 or lmic_dalys <= 0:
        return float('inf')
    if hic_publications <= 0 or lmic_publications <= 0:
        return float('inf')
    
    hic_intensity = hic_publications / hic_dalys
    lmic_intensity = lmic_publications / lmic_dalys
    
    if lmic_intensity <= 0:
        return float('inf')
    
    ratio = hic_intensity / lmic_intensity
    
    return round(ratio, 2)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_gbd_registry() -> Dict[str, Dict]:
    """
    Load the GBD disease registry created by 02-01.
    
    Expected registry structure from 02-01:
    {
        "cause_name": {
            "name": "cause_name",
            "gbd_level2": "GBD Level 2 Category",
            "global_south_priority": true/false,
            "dalys": 123456789,  # raw DALYs (not millions)
            "publications": 42
        },
        ...
    }
    """
    logger.info(f"Loading GBD registry from {INPUT_GBD_REGISTRY}")
    
    if not INPUT_GBD_REGISTRY.exists():
        raise FileNotFoundError(
            f"GBD registry not found: {INPUT_GBD_REGISTRY}\n"
            f"Run 02-01-bhem-map-diseases.py first"
        )
    
    with open(INPUT_GBD_REGISTRY, 'r') as f:
        registry = json.load(f)
    
    # Count statistics
    with_dalys = sum(1 for d in registry.values() if d.get('dalys', 0) > 0)
    gs_priority = sum(1 for d in registry.values() if d.get('global_south_priority', False))
    total_dalys = sum(d.get('dalys', 0) for d in registry.values())
    
    logger.info(f"Loaded GBD registry: {len(registry)} causes")
    logger.info(f"  Causes with DALYs: {with_dalys}")
    logger.info(f"  Global South priority: {gs_priority}")
    logger.info(f"  Total DALYs: {total_dalys/1e9:.2f} billion")
    
    return registry


def load_publications(filepath: Path) -> pd.DataFrame:
    """Load mapped publications data."""
    logger.info(f"Loading publications from {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} publications")
    
    return df


# =============================================================================
# MAIN COMPUTATION FUNCTIONS
# =============================================================================

def compute_all_disease_metrics(df: pd.DataFrame, 
                                 gbd_registry: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute metrics for ALL diseases in the GBD registry.
    
    Uses gbd_causes_str column from 02-01 to count publications per disease.
    
    Returns dict mapping disease_id -> {burden_score, gap_score, gap_severity, ...}
    """
    logger.info("Computing disease-level metrics...")
    
    disease_metrics = {}
    
    # Count publications per disease from gbd_causes_str column
    disease_pubs = defaultdict(int)
    
    if 'gbd_causes_str' in df.columns:
        for disease_str in df['gbd_causes_str'].dropna():
            for disease_id in str(disease_str).split('|'):
                disease_id = disease_id.strip()
                if disease_id:
                    disease_pubs[disease_id] += 1
        logger.info(f"  Counted publications from gbd_causes_str column")
        logger.info(f"  Found publications for {len(disease_pubs)} causes")
    else:
        # Fallback: use registry publication counts
        logger.warning("  No gbd_causes_str column - using registry publication counts")
        for disease_id, data in gbd_registry.items():
            disease_pubs[disease_id] = data.get('publications', 0)
    
    # Compute metrics for each disease in registry
    for disease_id, disease_data in gbd_registry.items():
        # Get publication count (prefer fresh count, fallback to registry)
        pubs = disease_pubs.get(disease_id, disease_data.get('publications', 0))
        
        # Get DALYs (raw number from IHME, convert to millions)
        dalys_raw = disease_data.get('dalys', 0)
        dalys_millions = dalys_raw / 1_000_000 if dalys_raw > 0 else 0
        
        # Get optional deaths/prevalence if available
        deaths_millions = disease_data.get('deaths_millions', None)
        prevalence_millions = disease_data.get('prevalence_millions', None)
        
        # Get GBD Level 2 category
        gbd_level2 = disease_data.get('gbd_level2', 'Unknown')
        
        # Map gbd_level2 to simplified category for gap scoring
        category = GBD_LEVEL2_TO_CATEGORY.get(gbd_level2, "Other NCD")
        
        # Determine Global South priority
        global_south = disease_data.get('global_south_priority', False)
        # Also flag if in relevant GBD category
        if gbd_level2 in GLOBAL_SOUTH_GBD_CATEGORIES:
            global_south = True
        
        # Compute burden score
        burden_score = compute_burden_score(
            dalys_millions=dalys_millions,
            deaths_millions=deaths_millions,
            prevalence_millions=prevalence_millions
        )
        
        # Compute gap score
        gap_score, gap_severity = compute_research_gap_score(
            disease_id=disease_id,
            publications=pubs,
            burden_score=burden_score,
            total_dalys_millions=dalys_millions,
            global_south_priority=global_south,
            category=category
        )
        
        # Research intensity: pubs per million DALYs
        research_intensity = pubs / dalys_millions if dalys_millions > 0 else 0
        
        disease_metrics[disease_id] = {
            'name': disease_data.get('name', disease_id),
            'category': category,
            'gbd_level2': gbd_level2,
            'dalys_millions': round(dalys_millions, 2),
            'deaths_millions': deaths_millions,
            'prevalence_millions': prevalence_millions,
            'global_south_priority': global_south,
            'publications': pubs,
            'burden_score': burden_score,
            'gap_score': gap_score,
            'gap_severity': gap_severity,
            'research_intensity': round(research_intensity, 3)
        }
    
    # Log summary
    severities = defaultdict(int)
    for dm in disease_metrics.values():
        severities[dm['gap_severity']] += 1
    
    total_pubs = sum(dm['publications'] for dm in disease_metrics.values())
    
    logger.info(f"Disease metrics computed: {len(disease_metrics)} causes")
    logger.info(f"  Total publications matched: {total_pubs:,}")
    logger.info(f"  Gap distribution: Critical={severities['Critical']}, "
                f"High={severities['High']}, Moderate={severities['Moderate']}, "
                f"Low={severities['Low']}")
    
    return disease_metrics


def compute_all_biobank_metrics(df: pd.DataFrame, 
                                 disease_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute metrics for all biobanks in the dataset.
    
    Returns dict mapping biobank_id -> {ros, eas, disease_publications, ...}
    """
    logger.info("Computing biobank-level metrics...")
    
    biobank_metrics = {}
    
    # Get unique biobanks
    if 'biobank_id' not in df.columns:
        logger.warning("No biobank_id column found")
        return biobank_metrics
    
    biobank_ids = df['biobank_id'].unique()
    
    for biobank_id in biobank_ids:
        biobank_df = df[df['biobank_id'] == biobank_id]
        
        # Count publications per disease for this biobank
        disease_pubs = defaultdict(int)
        
        if 'gbd_causes_str' in biobank_df.columns:
            for disease_str in biobank_df['gbd_causes_str'].dropna():
                for disease_id in str(disease_str).split('|'):
                    disease_id = disease_id.strip()
                    if disease_id:
                        disease_pubs[disease_id] += 1
        
        # Get biobank metadata
        biobank_name = biobank_df['biobank_name'].iloc[0] if 'biobank_name' in biobank_df.columns else biobank_id
        country = biobank_df['country'].iloc[0] if 'country' in biobank_df.columns else "Unknown"
        region = biobank_df['region'].iloc[0] if 'region' in biobank_df.columns else "Unknown"
        
        total_pubs = len(biobank_df)
        
        # Compute ROS
        ros = compute_research_opportunity_score(disease_pubs, disease_metrics)
        
        # Compute EAS
        eas, eas_category, eas_components = compute_equity_alignment_score(
            disease_pubs, disease_metrics, total_pubs
        )
        
        # Count critical gaps (diseases with 0 publications from this biobank)
        critical_gap_diseases = []
        for disease_id, metrics in disease_metrics.items():
            if disease_pubs.get(disease_id, 0) == 0:
                critical_gap_diseases.append(disease_id)
        
        # Global South percentage
        gs_pubs = 0
        if 'global_south_priority' in biobank_df.columns:
            gs_pubs = biobank_df['global_south_priority'].sum()
        gs_pct = (gs_pubs / total_pubs * 100) if total_pubs > 0 else 0
        
        # Year distribution
        year_dist = {}
        if 'year' in biobank_df.columns:
            year_counts = biobank_df['year'].value_counts()
            year_dist = {str(int(y)): int(c) for y, c in year_counts.items() if pd.notna(y)}
        
        # Determine if Global South biobank
        is_global_south = (
            region in GLOBAL_SOUTH_REGIONS or 
            country in GLOBAL_SOUTH_COUNTRIES
        )
        
        biobank_metrics[biobank_id] = {
            'name': biobank_name,
            'country': country,
            'region': region,
            'is_global_south': is_global_south,
            'total_publications': total_pubs,
            'diseases_covered': len([d for d in disease_pubs.values() if d > 0]),
            'research_opportunity_score': ros,
            'equity_alignment_score': eas,
            'equity_alignment_category': eas_category,
            'equity_alignment_components': eas_components,
            'critical_gap_count': len(critical_gap_diseases),
            'critical_gaps': critical_gap_diseases[:20],  # Limit for JSON size
            'global_south_percentage': round(gs_pct, 1),
            'disease_publications': dict(disease_pubs),
            'year_distribution': year_dist
        }
    
    # Log summary
    eas_dist = defaultdict(int)
    for bm in biobank_metrics.values():
        eas_dist[bm['equity_alignment_category']] += 1
    
    logger.info(f"Biobank metrics computed: {len(biobank_metrics)} biobanks")
    logger.info(f"  EAS distribution: Strong={eas_dist['Strong']}, "
                f"Moderate={eas_dist['Moderate']}, Weak={eas_dist['Weak']}, "
                f"Poor={eas_dist['Poor']}")
    
    return biobank_metrics


def compute_global_metrics(df: pd.DataFrame,
                           disease_metrics: Dict[str, Dict],
                           biobank_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compute global/system-wide metrics.
    """
    logger.info("Computing global metrics...")
    
    total_pubs = len(df)
    
    # Compute equity ratio
    # Estimate: 80% of DALYs in LMICs, 20% in HICs (WHO estimate)
    total_dalys = sum(dm['dalys_millions'] for dm in disease_metrics.values())
    lmic_dalys = total_dalys * 0.80
    hic_dalys = total_dalys * 0.20
    
    # Count publications by region
    hic_pubs = 0
    gs_pubs = 0
    
    for bm in biobank_metrics.values():
        if bm['is_global_south']:
            gs_pubs += bm['total_publications']
        else:
            hic_pubs += bm['total_publications']
    
    equity_ratio = compute_equity_ratio(hic_pubs, hic_dalys, gs_pubs, lmic_dalys)
    
    # Gap distribution
    gap_dist = defaultdict(int)
    for dm in disease_metrics.values():
        gap_dist[dm['gap_severity']] += 1
    
    # EAS distribution
    eas_dist = defaultdict(int)
    for bm in biobank_metrics.values():
        eas_dist[bm['equity_alignment_category']] += 1
    
    # Averages
    avg_ros = np.mean([bm['research_opportunity_score'] for bm in biobank_metrics.values()]) if biobank_metrics else 0
    avg_eas = np.mean([bm['equity_alignment_score'] for bm in biobank_metrics.values()]) if biobank_metrics else 0
    avg_gap = np.mean([dm['gap_score'] for dm in disease_metrics.values()]) if disease_metrics else 0
    
    global_metrics = {
        'version': VERSION,
        'version_date': VERSION_DATE,
        'generated_at': datetime.now().isoformat(),
        'total_publications': total_pubs,
        'total_biobanks': len(biobank_metrics),
        'total_diseases': len(disease_metrics),
        'equity_ratio': equity_ratio,
        'hic_publications': hic_pubs,
        'global_south_publications': gs_pubs,
        'gap_distribution': dict(gap_dist),
        'eas_distribution': dict(eas_dist),
        'average_ros': round(avg_ros, 2),
        'average_eas': round(avg_eas, 1),
        'average_gap_score': round(avg_gap, 1),
        'total_dalys_millions': round(total_dalys, 1),
        'methodology': {
            'burden_score': "0.5 × DALYs + 50 × Deaths + 10 × log₁₀(Prevalence) [or 10 × log₁₀(DALYs) if Deaths/Prevalence unavailable]",
            'gap_score': "Three-tier: zero-pub penalty (95), category thresholds (Infectious/Neglected), burden-normalized intensity",
            'ros': "Σ Burden_Score for diseases with ≤2 publications",
            'eas': "100 - (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)",
            'equity_ratio': "(Pubs_HIC / DALYs_HIC) / (Pubs_LMIC / DALYs_LMIC)",
            'source': METHODOLOGY_SOURCE
        }
    }
    
    logger.info(f"Global metrics: {total_pubs:,} publications, "
                f"equity ratio = {equity_ratio:.1f}x, "
                f"avg EAS = {avg_eas:.1f}")
    
    return global_metrics


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_metrics_json(global_metrics: Dict, biobank_metrics: Dict, 
                      disease_metrics: Dict, filepath: Path) -> None:
    """Save all metrics to JSON file."""
    output = {
        'global': global_metrics,
        'biobanks': biobank_metrics,
        'diseases': disease_metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Saved metrics JSON: {filepath}")


def save_biobank_csv(biobank_metrics: Dict, filepath: Path) -> None:
    """Save biobank metrics to CSV file."""
    rows = []
    
    for biobank_id, bm in biobank_metrics.items():
        row = {
            'biobank_id': biobank_id,
            'name': bm['name'],
            'country': bm['country'],
            'region': bm['region'],
            'is_global_south': bm['is_global_south'],
            'total_publications': bm['total_publications'],
            'diseases_covered': bm['diseases_covered'],
            'research_opportunity_score': bm['research_opportunity_score'],
            'equity_alignment_score': bm['equity_alignment_score'],
            'equity_alignment_category': bm['equity_alignment_category'],
            'critical_gap_count': bm['critical_gap_count'],
            'global_south_percentage': bm['global_south_percentage'],
            'gap_severity_component': bm['equity_alignment_components']['gap_severity_component'],
            'burden_miss_component': bm['equity_alignment_components']['burden_miss_component'],
            'capacity_penalty': bm['equity_alignment_components']['capacity_penalty']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('equity_alignment_score', ascending=False)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved biobank CSV: {filepath}")


def save_disease_csv(disease_metrics: Dict, filepath: Path) -> None:
    """Save disease metrics to CSV file."""
    rows = []
    
    for disease_id, dm in disease_metrics.items():
        row = {
            'disease_id': disease_id,
            'name': dm['name'],
            'category': dm['category'],
            'gbd_level2': dm['gbd_level2'],
            'dalys_millions': dm['dalys_millions'],
            'deaths_millions': dm['deaths_millions'],
            'prevalence_millions': dm['prevalence_millions'],
            'global_south_priority': dm['global_south_priority'],
            'publications': dm['publications'],
            'burden_score': dm['burden_score'],
            'gap_score': dm['gap_score'],
            'gap_severity': dm['gap_severity'],
            'research_intensity': dm['research_intensity']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('gap_score', ascending=False)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved disease CSV: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-Biobank v1.0: Compute Health Equity Metrics")
    print(f"Methodology: {METHODOLOGY_SOURCE}")
    print(f"Using FULL GBD registry (all causes, not arbitrary 25-disease subset)")
    print("=" * 70)
    
    # Check input files
    if not INPUT_PUBLICATIONS.exists():
        print(f"\n❌ Input file not found: {INPUT_PUBLICATIONS}")
        print(f"   Run 02-01-bhem-map-diseases.py first")
        return
    
    if not INPUT_GBD_REGISTRY.exists():
        print(f"\n❌ GBD registry not found: {INPUT_GBD_REGISTRY}")
        print(f"   Run 02-01-bhem-map-diseases.py first")
        return
    
    # Load GBD registry
    print(f"\n📂 Loading GBD registry...")
    gbd_registry = load_gbd_registry()
    print(f"   GBD causes loaded: {len(gbd_registry)}")
    
    # Load publications
    print(f"\n📂 Loading publications...")
    df = load_publications(INPUT_PUBLICATIONS)
    print(f"   Publications loaded: {len(df):,}")
    
    # Compute disease metrics
    print(f"\n📊 Computing disease metrics...")
    disease_metrics = compute_all_disease_metrics(df, gbd_registry)
    print(f"   Causes processed: {len(disease_metrics)}")
    
    # Compute biobank metrics
    print(f"\n🏦 Computing biobank metrics...")
    biobank_metrics = compute_all_biobank_metrics(df, disease_metrics)
    print(f"   Biobanks processed: {len(biobank_metrics)}")
    
    # Compute global metrics
    print(f"\n🌍 Computing global metrics...")
    global_metrics = compute_global_metrics(df, disease_metrics, biobank_metrics)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    save_metrics_json(global_metrics, biobank_metrics, disease_metrics, OUTPUT_METRICS_JSON)
    save_biobank_csv(biobank_metrics, OUTPUT_BIOBANK_CSV)
    save_disease_csv(disease_metrics, OUTPUT_DISEASE_CSV)
    
    # Print summary
    print(f"\n" + "=" * 70)
    print(f"📊 HEIM-Biobank v1.0 METRICS SUMMARY")
    print(f"=" * 70)
    
    print(f"\n🔢 Global Statistics:")
    print(f"   Total publications: {global_metrics['total_publications']:,}")
    print(f"   Total biobanks: {global_metrics['total_biobanks']}")
    print(f"   Total GBD causes: {global_metrics['total_diseases']}")
    print(f"   Total DALYs: {global_metrics['total_dalys_millions']:.1f}M")
    print(f"   Equity ratio: {global_metrics['equity_ratio']:.1f}x (>1 = HIC bias)")
    
    print(f"\n📈 Gap Distribution:")
    for severity, count in sorted(global_metrics['gap_distribution'].items()):
        print(f"   {severity}: {count} causes")
    
    print(f"\n🏆 Equity Alignment Distribution:")
    for category, count in sorted(global_metrics['eas_distribution'].items()):
        print(f"   {category}: {count} biobanks")
    
    print(f"\n📁 Output Files:")
    print(f"   {OUTPUT_METRICS_JSON}")
    print(f"   {OUTPUT_BIOBANK_CSV}")
    print(f"   {OUTPUT_DISEASE_CSV}")
    
    # Top 5 biobanks by EAS
    if biobank_metrics:
        print(f"\n🏆 Top 5 Biobanks by Equity Alignment Score:")
        sorted_biobanks = sorted(biobank_metrics.items(), 
                                  key=lambda x: x[1]['equity_alignment_score'], 
                                  reverse=True)[:5]
        for i, (bid, bm) in enumerate(sorted_biobanks, 1):
            print(f"   {i}. {bm['name']}: EAS={bm['equity_alignment_score']:.1f} "
                  f"({bm['equity_alignment_category']}), {bm['diseases_covered']} causes")
    
    # Top 10 critical gap diseases (by burden)
    print(f"\n⚠️  Top 10 Critical Gap Diseases (by burden):")
    critical_diseases = [(did, dm) for did, dm in disease_metrics.items() 
                         if dm['gap_severity'] == 'Critical']
    sorted_critical = sorted(critical_diseases, key=lambda x: x[1]['dalys_millions'], reverse=True)[:10]
    for i, (did, dm) in enumerate(sorted_critical, 1):
        gs_flag = "🌍" if dm['global_south_priority'] else "  "
        print(f"   {i}. {gs_flag} {dm['name']}: Gap={dm['gap_score']:.0f}, "
              f"Pubs={dm['publications']}, DALYs={dm['dalys_millions']:.1f}M")
    
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Next step: python 02-05-bhem-generate-json.py")


if __name__ == "__main__":
    main()