#!/usr/bin/env python3
"""
04-03-heim-ct-compute-metrics.py
================================
HEIM-CT: Compute Clinical Trial Enrollment Equity Metrics

Computes enrollment equity metrics aligned with the HEIM framework,
enabling direct comparison between biobank research output and
clinical trial enrollment patterns.

METRICS COMPUTED:
    1. Enrollment Representation Index (ERI) - Observed % / Reference %
    2. CT Gap Score (0-100) - Trial coverage gap vs disease burden
    3. HIC/LMIC Trial Ratio - Geographic concentration
    4. Sponsor Diversity - By agency class
    5. Temporal Diversity Trends - Year-over-year changes
    6. Disease-Biobank Integration Score - HEIM alignment

INPUT:
    DATA/heim_ct_studies_mapped.csv
    DATA/heim_ct_baseline.csv
    DATA/heim_ct_countries.csv
    DATA/heim_ct_sponsors.csv
    DATA/heim_ct_disease_registry.json
    DATA/bhem_disease_metrics.csv (from 03-03, optional)
    
OUTPUT:
    DATA/heim_ct_enrollment_equity.csv
    DATA/heim_ct_disease_equity.csv
    DATA/heim_ct_geographic_equity.csv
    DATA/heim_ct_metrics.json
    ANALYSIS/HEIM-CT/equity_report.txt

USAGE:
    python 04-03-heim-ct-compute-metrics.py

VERSION: HEIM-CT v1.0
DATE: 2026-01-13
"""

import os
import re
import json
import logging
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional

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

VERSION = "HEIM-CT v1.0"
VERSION_DATE = "2026-01-13"
METHODOLOGY_SOURCE = "HEIM Framework (Corpas et al. 2025)"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
ANALYSIS_DIR = BASE_DIR / "ANALYSIS" / "04-04-HEIM-CT-FIGURES"

# Input files
INPUT_STUDIES = DATA_DIR / "heim_ct_studies_mapped.csv"
INPUT_BASELINE = DATA_DIR / "heim_ct_baseline.csv"
INPUT_COUNTRIES = DATA_DIR / "heim_ct_countries.csv"
INPUT_SPONSORS = DATA_DIR / "heim_ct_sponsors.csv"
INPUT_CT_REGISTRY = DATA_DIR / "heim_ct_disease_registry.json"
INPUT_BIOBANK_METRICS = DATA_DIR / "bhem_disease_metrics.csv"  # From 03-03

# Output files
OUTPUT_ENROLLMENT = DATA_DIR / "heim_ct_enrollment_equity.csv"
OUTPUT_DISEASE = DATA_DIR / "heim_ct_disease_equity.csv"
OUTPUT_GEOGRAPHIC = DATA_DIR / "heim_ct_geographic_equity.csv"
OUTPUT_METRICS_JSON = DATA_DIR / "heim_ct_metrics.json"
OUTPUT_REPORT = ANALYSIS_DIR / "04-04-00_equity_report.txt"


# =============================================================================
# REFERENCE POPULATIONS
# =============================================================================

# US Census 2020 racial/ethnic distribution
US_CENSUS_2020 = {
    'white': 0.578,
    'black': 0.121,
    'asian': 0.059,
    'hispanic': 0.187,
    'american_indian': 0.007,
    'pacific_islander': 0.002,
    'multiple': 0.107,
    'other': 0.039
}

# Global population distribution (approximate)
GLOBAL_POPULATION = {
    'white': 0.11,
    'black': 0.15,
    'asian': 0.60,
    'hispanic': 0.08,
    'american_indian': 0.01,
    'pacific_islander': 0.01,
    'multiple': 0.02,
    'other': 0.02
}


# =============================================================================
# COUNTRY CLASSIFICATION
# =============================================================================

HIC_COUNTRIES = {
    'United States', 'United Kingdom', 'Germany', 'France', 'Canada',
    'Australia', 'Japan', 'Italy', 'Spain', 'Netherlands', 'Belgium',
    'Switzerland', 'Austria', 'Sweden', 'Norway', 'Denmark', 'Finland',
    'Ireland', 'Israel', 'Singapore', 'South Korea', 'New Zealand',
    'Luxembourg', 'Iceland', 'Portugal', 'Greece', 'Czech Republic',
    'Slovenia', 'Estonia', 'Poland', 'Hungary', 'Slovakia', 'Lithuania',
    'Latvia', 'Croatia', 'Cyprus', 'Malta', 'Taiwan', 'Hong Kong',
    'United Arab Emirates', 'Qatar', 'Kuwait', 'Bahrain', 'Saudi Arabia'
}

LMIC_COUNTRIES = {
    'India', 'China', 'Brazil', 'Mexico', 'Indonesia', 'Pakistan',
    'Bangladesh', 'Nigeria', 'Ethiopia', 'Philippines', 'Egypt',
    'Vietnam', 'Turkey', 'Iran', 'Thailand', 'South Africa', 'Colombia',
    'Argentina', 'Kenya', 'Uganda', 'Tanzania', 'Ghana', 'Peru',
    'Venezuela', 'Malaysia', 'Morocco', 'Algeria', 'Sudan', 'Iraq',
    'Afghanistan', 'Nepal', 'Sri Lanka', 'Myanmar', 'Cambodia', 'Laos',
    'Zimbabwe', 'Zambia', 'Malawi', 'Mozambique', 'Rwanda', 'Senegal',
    'Mali', 'Niger', 'Burkina Faso', 'Guinea', 'Sierra Leone', 'Liberia',
    'Central African Republic', 'Democratic Republic of the Congo',
    'Haiti', 'Honduras', 'Guatemala', 'Nicaragua', 'Bolivia', 'Paraguay',
    'Ecuador', 'Dominican Republic', 'Cuba', 'Jamaica', 'Chile', 'Uruguay'
}


# =============================================================================
# RACE/ETHNICITY EXTRACTION
# =============================================================================

RACE_PATTERNS = {
    'white': [
        r'\bwhite\b', r'\bcaucasian\b', r'\beuropean\b', 
        r'\bnon.?hispanic\s+white\b'
    ],
    'black': [
        r'\bblack\b', r'\bafrican\s*american\b', r'\bafro\b',
        r'\bafrican\b'
    ],
    'asian': [
        r'\basian\b', r'\bchinese\b', r'\bjapanese\b', r'\bkorean\b',
        r'\bvietnamese\b', r'\bfilipino\b', r'\bindian\b', r'\bsouth\s*asian\b',
        r'\beast\s*asian\b', r'\bsoutheast\s*asian\b'
    ],
    'hispanic': [
        r'\bhispanic\b', r'\blatino\b', r'\blatina\b', r'\blatinx\b',
        r'\bmexican\b', r'\bpuerto\s*rican\b', r'\bcuban\b', r'\bspanish\b'
    ],
    'american_indian': [
        r'\bamerican\s*indian\b', r'\balaska\s*native\b', 
        r'\bnative\s*american\b', r'\bindigenous\b', r'\bfirst\s*nations\b'
    ],
    'pacific_islander': [
        r'\bpacific\s*islander\b', r'\bhawaiian\b', r'\bsamoan\b',
        r'\bguamanian\b', r'\bchamorro\b'
    ],
    'multiple': [
        r'\bmixed\b', r'\bmultiracial\b', r'\btwo\s*or\s*more\b',
        r'\bmultiple\s*races?\b', r'\bbiracial\b'
    ],
    'other': [
        r'\bother\b', r'\bunknown\b', r'\bnot\s*reported\b',
        r'\bnot\s*specified\b', r'\bdeclined\b'
    ]
}


def classify_race_category(text: str) -> Optional[str]:
    """Classify a text string into a race/ethnicity category."""
    if not text or pd.isna(text):
        return None
    
    text_lower = str(text).lower().strip()
    
    for category, patterns in RACE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category
    
    return 'other'


def extract_enrollment_by_race(df_baseline: pd.DataFrame) -> Dict[str, Dict]:
    """Extract enrollment counts by race/ethnicity from baseline measurements."""
    logger.info("   Extracting enrollment by race/ethnicity...")
    
    # Filter for race/ethnicity rows
    race_mask = df_baseline['title'].fillna('').str.lower().str.contains(
        'race|ethnic|racial|ancestry', regex=True
    )
    df_race = df_baseline[race_mask].copy()
    
    if len(df_race) == 0:
        logger.warning("   No race/ethnicity data found in baseline measurements")
        return {}
    
    logger.info(f"   Found {len(df_race):,} race/ethnicity rows")
    
    # Aggregate by category
    enrollment = defaultdict(lambda: {'count': 0, 'trials': set()})
    
    for _, row in df_race.iterrows():
        category_text = str(row.get('category', ''))
        race_cat = classify_race_category(category_text)
        
        if race_cat:
            # Try to get numeric value
            value = row.get('param_value_num')
            if pd.isna(value):
                try:
                    value = float(str(row.get('param_value', '0')).replace(',', ''))
                except:
                    value = 0
            
            enrollment[race_cat]['count'] += value
            enrollment[race_cat]['trials'].add(row.get('nct_id'))
    
    # Convert sets to counts
    result = {}
    for cat, data in enrollment.items():
        result[cat] = {
            'count': data['count'],
            'trial_count': len(data['trials'])
        }
    
    return result


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_enrollment_representation_index(
    observed: Dict[str, Dict],
    reference: Dict[str, float] = US_CENSUS_2020
) -> Dict[str, Dict]:
    """
    Compute Enrollment Representation Index (ERI) for each race/ethnicity.
    
    ERI = (Observed % in trials) / (Reference population %)
    
    Interpretation:
        ERI < 0.5: Severely underrepresented
        0.5 ≤ ERI < 0.8: Underrepresented
        0.8 ≤ ERI ≤ 1.2: Proportionally represented
        ERI > 1.2: Overrepresented
    """
    total_enrollment = sum(d['count'] for d in observed.values())
    
    if total_enrollment == 0:
        return {}
    
    eri = {}
    for category, ref_pct in reference.items():
        obs_count = observed.get(category, {}).get('count', 0)
        obs_pct = obs_count / total_enrollment if total_enrollment > 0 else 0
        
        index = obs_pct / ref_pct if ref_pct > 0 else 0
        
        # Classify representation
        if index < 0.5:
            status = "Severely underrepresented"
        elif index < 0.8:
            status = "Underrepresented"
        elif index <= 1.2:
            status = "Proportional"
        else:
            status = "Overrepresented"
        
        eri[category] = {
            'observed_count': obs_count,
            'observed_pct': round(obs_pct * 100, 2),
            'reference_pct': round(ref_pct * 100, 2),
            'eri': round(index, 3),
            'status': status,
            'trial_count': observed.get(category, {}).get('trial_count', 0)
        }
    
    return eri


def compute_ct_gap_score(
    trials: int,
    dalys_millions: float,
    is_global_south_priority: bool = False
) -> Tuple[float, str]:
    """
    Compute Clinical Trial Gap Score (0-100).
    
    Analogous to HEIM-Biobank Research Gap Score.
    Higher score = larger gap between disease burden and trial activity.
    
    Formula adapted from Corpas et al. (2025):
        Base score from trials per million DALYs
        Adjusted for Global South priority diseases
    """
    # Zero trial penalty
    if trials == 0:
        return 95.0, "Critical"
    
    if dalys_millions <= 0:
        dalys_millions = 0.1  # Minimum for scoring
    
    # Trials per million DALYs
    trials_per_daly = trials / dalys_millions
    
    # Score thresholds (adapted from biobank methodology)
    if is_global_south_priority:
        # Stricter thresholds for Global South priority diseases
        if trials_per_daly < 1:
            gap_score = 90 - (trials_per_daly * 10)
        elif trials_per_daly < 5:
            gap_score = 80 - ((trials_per_daly - 1) * 5)
        elif trials_per_daly < 20:
            gap_score = 60 - ((trials_per_daly - 5) * 2)
        elif trials_per_daly < 50:
            gap_score = 30 - ((trials_per_daly - 20) * 0.5)
        else:
            gap_score = max(5, 15 - (trials_per_daly - 50) * 0.1)
    else:
        # Standard thresholds
        if trials_per_daly < 5:
            gap_score = 85 - (trials_per_daly * 5)
        elif trials_per_daly < 20:
            gap_score = 60 - ((trials_per_daly - 5) * 2)
        elif trials_per_daly < 50:
            gap_score = 30 - ((trials_per_daly - 20) * 0.5)
        elif trials_per_daly < 100:
            gap_score = 15 - ((trials_per_daly - 50) * 0.1)
        else:
            gap_score = max(5, 10 - (trials_per_daly - 100) * 0.05)
    
    gap_score = max(0, min(100, gap_score))
    
    # Severity classification
    if gap_score >= 70:
        severity = "Critical"
    elif gap_score >= 50:
        severity = "High"
    elif gap_score >= 30:
        severity = "Moderate"
    else:
        severity = "Low"
    
    return round(gap_score, 1), severity


def compute_geographic_concentration(df_countries: pd.DataFrame) -> Dict:
    """Compute HIC/LMIC trial distribution."""
    logger.info("   Computing geographic concentration...")
    
    # Count trials by country
    country_trials = df_countries.groupby('name')['nct_id'].nunique().to_dict()
    
    hic_trials = 0
    lmic_trials = 0
    other_trials = 0
    
    for country, count in country_trials.items():
        if country in HIC_COUNTRIES:
            hic_trials += count
        elif country in LMIC_COUNTRIES:
            lmic_trials += count
        else:
            other_trials += count
    
    total = hic_trials + lmic_trials + other_trials
    
    # HIC/LMIC ratio
    ratio = hic_trials / lmic_trials if lmic_trials > 0 else float('inf')
    
    # Top countries
    top_countries = sorted(country_trials.items(), key=lambda x: -x[1])[:20]
    
    return {
        'hic_trials': hic_trials,
        'lmic_trials': lmic_trials,
        'other_trials': other_trials,
        'total_trials': total,
        'hic_percentage': round(100 * hic_trials / total, 1) if total > 0 else 0,
        'lmic_percentage': round(100 * lmic_trials / total, 1) if total > 0 else 0,
        'hic_lmic_ratio': round(ratio, 1) if ratio != float('inf') else 'inf',
        'top_countries': [{'country': c, 'trials': t} for c, t in top_countries]
    }


def compute_sponsor_diversity(df_sponsors: pd.DataFrame, df_baseline: pd.DataFrame) -> Dict:
    """Compute diversity metrics by sponsor type."""
    logger.info("   Computing sponsor diversity...")
    
    # Get lead sponsors
    lead_sponsors = df_sponsors[df_sponsors['lead_or_collaborator'] == 'lead'].copy()
    
    sponsor_stats = {}
    for agency_class in ['NIH', 'Industry', 'Other']:
        mask = lead_sponsors['agency_class'] == agency_class
        nct_ids = lead_sponsors[mask]['nct_id'].unique()
        
        # Get baseline data for these trials
        trial_baseline = df_baseline[df_baseline['nct_id'].isin(nct_ids)]
        
        sponsor_stats[agency_class] = {
            'trial_count': len(nct_ids),
            'with_demographics': trial_baseline['nct_id'].nunique()
        }
    
    return sponsor_stats


def compute_disease_equity_metrics(
    df_studies: pd.DataFrame,
    ct_registry: Dict,
    biobank_metrics: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """Compute equity metrics for each disease."""
    logger.info("   Computing disease-level equity metrics...")
    
    disease_metrics = []
    
    # Count trials per disease
    disease_trial_counts = defaultdict(int)
    for _, row in df_studies.iterrows():
        causes = row.get('gbd_causes', [])
        if isinstance(causes, str):
            causes = causes.split('|') if causes else []
        for cause in causes:
            disease_trial_counts[cause] += 1
    
    # Load biobank metrics for integration
    biobank_gap = {}
    if biobank_metrics is not None and len(biobank_metrics) > 0:
        for _, row in biobank_metrics.iterrows():
            disease_name = row.get('name', row.get('disease_id', ''))
            biobank_gap[disease_name] = {
                'gap_score': row.get('gap_score', None),
                'publications': row.get('publications', 0)
            }
    
    for disease, info in ct_registry.items():
        trials = disease_trial_counts.get(disease, 0)
        dalys = info.get('dalys', 0) / 1e6  # Convert to millions
        is_gs = info.get('global_south_priority', False)
        
        ct_gap, severity = compute_ct_gap_score(trials, dalys, is_gs)
        
        # Integration with biobank metrics
        bb_data = biobank_gap.get(disease, {})
        bb_gap = bb_data.get('gap_score')
        bb_pubs = bb_data.get('publications', 0)
        
        # Combined pipeline score
        if bb_gap is not None:
            # Average of biobank and CT gap scores
            combined_gap = (ct_gap + bb_gap) / 2
        else:
            combined_gap = ct_gap
        
        disease_metrics.append({
            'disease': disease,
            'gbd_level2': info.get('gbd_level2', 'Unknown'),
            'global_south_priority': is_gs,
            'dalys_millions': round(dalys, 2),
            'trial_count': trials,
            'ct_gap_score': ct_gap,
            'ct_gap_severity': severity,
            'biobank_publications': bb_pubs,
            'biobank_gap_score': bb_gap,
            'combined_gap_score': round(combined_gap, 1),
            'trials_per_million_dalys': round(trials / dalys, 2) if dalys > 0 else 0
        })
    
    return disease_metrics


def compute_temporal_trends(df_studies: pd.DataFrame) -> Dict:
    """Compute year-over-year trends in trial characteristics."""
    logger.info("   Computing temporal trends...")
    
    # Use start_year or posted_year
    year_col = 'start_year' if 'start_year' in df_studies.columns else 'posted_year'
    
    df_studies = df_studies.copy()
    df_studies[year_col] = pd.to_numeric(df_studies[year_col], errors='coerce')
    
    trends = {}
    
    for year in range(2010, 2025):
        year_data = df_studies[df_studies[year_col] == year]
        
        if len(year_data) == 0:
            continue
        
        gs_count = year_data['global_south_priority'].sum() if 'global_south_priority' in year_data.columns else 0
        
        trends[year] = {
            'total_trials': len(year_data),
            'global_south_priority_trials': int(gs_count),
            'gs_percentage': round(100 * gs_count / len(year_data), 1) if len(year_data) > 0 else 0
        }
    
    return trends


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def write_equity_report(
    eri: Dict,
    geographic: Dict,
    disease_metrics: List[Dict],
    temporal: Dict,
    output_path: Path
) -> None:
    """Write human-readable equity report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HEIM-CT: Clinical Trial Enrollment Equity Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Version: {VERSION}\n")
        f.write(f"Methodology: {METHODOLOGY_SOURCE}\n")
        f.write("=" * 80 + "\n\n")
        
        # Enrollment Representation
        f.write("ENROLLMENT REPRESENTATION INDEX (ERI)\n")
        f.write("-" * 40 + "\n")
        f.write("ERI = Observed% / Reference% (US Census 2020)\n")
        f.write("< 0.5: Severely underrepresented | 0.5-0.8: Underrepresented\n")
        f.write("0.8-1.2: Proportional | > 1.2: Overrepresented\n\n")
        
        for category, data in sorted(eri.items(), key=lambda x: x[1].get('eri', 0)):
            f.write(f"  {category:20} ERI={data['eri']:5.2f}  "
                    f"Obs={data['observed_pct']:5.1f}%  Ref={data['reference_pct']:5.1f}%  "
                    f"[{data['status']}]\n")
        f.write("\n")
        
        # Geographic Distribution
        f.write("GEOGRAPHIC DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"HIC trials: {geographic['hic_trials']:,} ({geographic['hic_percentage']:.1f}%)\n")
        f.write(f"LMIC trials: {geographic['lmic_trials']:,} ({geographic['lmic_percentage']:.1f}%)\n")
        ratio_str = str(geographic['hic_lmic_ratio']) if geographic['hic_lmic_ratio'] != 'inf' else '∞'
        f.write(f"HIC:LMIC ratio: {ratio_str}:1\n\n")
        
        f.write("Top 10 countries:\n")
        for item in geographic['top_countries'][:10]:
            f.write(f"  {item['country']:30} {item['trials']:>8,} trials\n")
        f.write("\n")
        
        # Disease Gap Analysis
        f.write("DISEASE GAP ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Gap distribution
        gap_dist = defaultdict(int)
        for dm in disease_metrics:
            gap_dist[dm['ct_gap_severity']] += 1
        
        f.write("Gap severity distribution:\n")
        for severity in ['Critical', 'High', 'Moderate', 'Low']:
            f.write(f"  {severity:12} {gap_dist[severity]:3} diseases\n")
        f.write("\n")
        
        # Top critical gaps
        critical = [d for d in disease_metrics if d['ct_gap_severity'] == 'Critical']
        critical_sorted = sorted(critical, key=lambda x: x['dalys_millions'], reverse=True)
        
        f.write("Top 15 Critical Gap Diseases (by burden):\n")
        for dm in critical_sorted[:15]:
            gs = "🌍" if dm['global_south_priority'] else "  "
            f.write(f"  {gs} {dm['disease'][:40]:40} Gap={dm['ct_gap_score']:4.0f}  "
                    f"Trials={dm['trial_count']:5}  DALYs={dm['dalys_millions']:6.1f}M\n")
        f.write("\n")
        
        # Global South priority analysis
        gs_diseases = [d for d in disease_metrics if d['global_south_priority']]
        gs_critical = len([d for d in gs_diseases if d['ct_gap_severity'] == 'Critical'])
        
        f.write("GLOBAL SOUTH PRIORITY DISEASES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total GS priority diseases: {len(gs_diseases)}\n")
        f.write(f"With Critical gaps: {gs_critical} ({100*gs_critical/len(gs_diseases):.1f}%)\n")
        f.write(f"Average CT Gap Score: {np.mean([d['ct_gap_score'] for d in gs_diseases]):.1f}\n\n")
        
        # Temporal trends
        if temporal:
            f.write("TEMPORAL TRENDS\n")
            f.write("-" * 40 + "\n")
            for year in sorted(temporal.keys()):
                data = temporal[year]
                f.write(f"  {year}: {data['total_trials']:>6,} trials, "
                        f"{data['gs_percentage']:5.1f}% GS priority\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"   Report saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-CT: Compute Clinical Trial Enrollment Equity Metrics")
    print(f"Version: {VERSION}")
    print(f"Methodology: {METHODOLOGY_SOURCE}")
    print("=" * 70)
    
    # Ensure directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check inputs
    if not INPUT_STUDIES.exists():
        print(f"\n❌ Studies file not found: {INPUT_STUDIES}")
        print(f"   Run 04-02-heim-ct-map-diseases.py first")
        return
    
    # Load data
    print(f"\n📂 Loading data...")
    
    df_studies = pd.read_csv(INPUT_STUDIES)
    print(f"   Studies: {len(df_studies):,}")
    
    df_baseline = pd.read_csv(INPUT_BASELINE) if INPUT_BASELINE.exists() else pd.DataFrame()
    print(f"   Baseline measurements: {len(df_baseline):,}")
    
    df_countries = pd.read_csv(INPUT_COUNTRIES) if INPUT_COUNTRIES.exists() else pd.DataFrame()
    print(f"   Countries: {len(df_countries):,}")
    
    df_sponsors = pd.read_csv(INPUT_SPONSORS) if INPUT_SPONSORS.exists() else pd.DataFrame()
    print(f"   Sponsors: {len(df_sponsors):,}")
    
    # Load CT disease registry
    ct_registry = {}
    if INPUT_CT_REGISTRY.exists():
        with open(INPUT_CT_REGISTRY, 'r') as f:
            ct_registry = json.load(f)
        print(f"   Disease registry: {len(ct_registry)} diseases")
    
    # Load biobank metrics for integration
    biobank_metrics = None
    if INPUT_BIOBANK_METRICS.exists():
        biobank_metrics = pd.read_csv(INPUT_BIOBANK_METRICS)
        print(f"   Biobank metrics: {len(biobank_metrics)} diseases")
    else:
        print(f"   ⚠️ Biobank metrics not found; integration scores unavailable")
    
    # Compute metrics
    print(f"\n📊 Computing equity metrics...")
    
    # 1. Enrollment representation
    print(f"\n   1. Enrollment Representation Index...")
    enrollment_by_race = extract_enrollment_by_race(df_baseline)
    eri = compute_enrollment_representation_index(enrollment_by_race, US_CENSUS_2020)
    
    if eri:
        for cat, data in sorted(eri.items(), key=lambda x: x[1].get('eri', 0)):
            print(f"      {cat:20} ERI={data['eri']:.2f} ({data['status']})")
    else:
        print(f"      ⚠️ No enrollment data available")
    
    # 2. Geographic concentration
    print(f"\n   2. Geographic Concentration...")
    geographic = compute_geographic_concentration(df_countries) if len(df_countries) > 0 else {}
    
    if geographic:
        print(f"      HIC trials: {geographic.get('hic_trials', 0):,} ({geographic.get('hic_percentage', 0):.1f}%)")
        print(f"      LMIC trials: {geographic.get('lmic_trials', 0):,} ({geographic.get('lmic_percentage', 0):.1f}%)")
        print(f"      HIC:LMIC ratio: {geographic.get('hic_lmic_ratio', 'N/A')}:1")
    
    # 3. Disease equity metrics
    print(f"\n   3. Disease Equity Metrics...")
    disease_metrics = compute_disease_equity_metrics(df_studies, ct_registry, biobank_metrics)
    
    gap_dist = defaultdict(int)
    for dm in disease_metrics:
        gap_dist[dm['ct_gap_severity']] += 1
    print(f"      Critical: {gap_dist['Critical']}, High: {gap_dist['High']}, "
          f"Moderate: {gap_dist['Moderate']}, Low: {gap_dist['Low']}")
    
    # 4. Sponsor diversity
    print(f"\n   4. Sponsor Diversity...")
    sponsor_diversity = compute_sponsor_diversity(df_sponsors, df_baseline) if len(df_sponsors) > 0 else {}
    
    for sponsor, stats in sponsor_diversity.items():
        print(f"      {sponsor:12} {stats['trial_count']:>6,} trials")
    
    # 5. Temporal trends
    print(f"\n   5. Temporal Trends...")
    temporal = compute_temporal_trends(df_studies)
    
    if temporal:
        recent_years = sorted(temporal.keys())[-3:]
        for year in recent_years:
            print(f"      {year}: {temporal[year]['total_trials']:,} trials, "
                  f"{temporal[year]['gs_percentage']:.1f}% GS priority")
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Enrollment equity CSV
    if eri:
        eri_rows = [{'category': k, **v} for k, v in eri.items()]
        pd.DataFrame(eri_rows).to_csv(OUTPUT_ENROLLMENT, index=False)
        print(f"   {OUTPUT_ENROLLMENT.name}")
    
    # Disease equity CSV
    df_disease = pd.DataFrame(disease_metrics)
    df_disease = df_disease.sort_values('ct_gap_score', ascending=False)
    df_disease.to_csv(OUTPUT_DISEASE, index=False)
    print(f"   {OUTPUT_DISEASE.name}: {len(df_disease)} diseases")
    
    # Geographic equity CSV
    if geographic.get('top_countries'):
        pd.DataFrame(geographic['top_countries']).to_csv(OUTPUT_GEOGRAPHIC, index=False)
        print(f"   {OUTPUT_GEOGRAPHIC.name}")
    
    # Comprehensive JSON
    metrics_output = {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'methodology': METHODOLOGY_SOURCE,
        'summary': {
            'total_trials': len(df_studies),
            'trials_with_demographics': df_baseline['nct_id'].nunique() if len(df_baseline) > 0 else 0,
            'diseases_analyzed': len(disease_metrics),
            'hic_lmic_ratio': geographic.get('hic_lmic_ratio', 'N/A'),
            'gap_distribution': dict(gap_dist)
        },
        'enrollment_representation': eri,
        'geographic_distribution': geographic,
        'sponsor_diversity': sponsor_diversity,
        'temporal_trends': temporal,
        'disease_metrics_summary': {
            'critical_gap_count': gap_dist['Critical'],
            'high_gap_count': gap_dist['High'],
            'avg_ct_gap_score': round(np.mean([d['ct_gap_score'] for d in disease_metrics]), 1),
            'global_south_priority_avg_gap': round(
                np.mean([d['ct_gap_score'] for d in disease_metrics if d['global_south_priority']]), 1
            ) if any(d['global_south_priority'] for d in disease_metrics) else 0
        }
    }
    
    with open(OUTPUT_METRICS_JSON, 'w') as f:
        json.dump(metrics_output, f, indent=2, default=str)
    print(f"   {OUTPUT_METRICS_JSON.name}")
    
    # Equity report
    write_equity_report(eri, geographic, disease_metrics, temporal, OUTPUT_REPORT)
    print(f"   {OUTPUT_REPORT.name}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ METRICS COMPUTATION COMPLETE")
    print(f"=" * 70)
    
    print(f"\n📊 Key Findings:")
    print(f"   Total trials analyzed: {len(df_studies):,}")
    print(f"   Trials with demographics: {df_baseline['nct_id'].nunique() if len(df_baseline) > 0 else 0:,}")
    
    if geographic:
        print(f"   HIC:LMIC trial ratio: {geographic.get('hic_lmic_ratio', 'N/A')}:1")
    
    print(f"   Critical gap diseases: {gap_dist['Critical']}")
    
    # Global South priority summary
    gs_diseases = [d for d in disease_metrics if d['global_south_priority']]
    if gs_diseases:
        gs_avg_gap = np.mean([d['ct_gap_score'] for d in gs_diseases])
        print(f"   Global South priority avg gap: {gs_avg_gap:.1f}")
    
    print(f"\n📁 Output files:")
    print(f"   {OUTPUT_ENROLLMENT}")
    print(f"   {OUTPUT_DISEASE}")
    print(f"   {OUTPUT_GEOGRAPHIC}")
    print(f"   {OUTPUT_METRICS_JSON}")
    print(f"   {OUTPUT_REPORT}")
    
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Integration: Compare CT metrics with biobank metrics from 03-03")


if __name__ == "__main__":
    main()
