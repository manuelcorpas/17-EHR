#!/usr/bin/env python3
"""
02-02-bhem-compute-metrics.py
=============================
BHEM Step 3: Compute research gap scores and equity metrics

Calculates burden scores, gap scores, Research Opportunity Scores (ROS),
and equity metrics for each biobank and disease.

INPUT:  DATA/bhem_publications_mapped.csv
        DATA/disease_registry.json
OUTPUT: DATA/bhem_metrics.json
        DATA/bhem_biobank_metrics.csv
        DATA/bhem_disease_metrics.csv

USAGE:
    python 02-02-bhem-compute-metrics.py
"""

import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"

INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"
INPUT_DISEASES = DATA_DIR / "disease_registry.json"

OUTPUT_METRICS = DATA_DIR / "bhem_metrics.json"
OUTPUT_BIOBANK_CSV = DATA_DIR / "bhem_biobank_metrics.csv"
OUTPUT_DISEASE_CSV = DATA_DIR / "bhem_disease_metrics.csv"

# Gap score thresholds
GAP_CRITICAL = 70
GAP_HIGH = 50
GAP_MODERATE = 30

# WHO Regions
GLOBAL_SOUTH_REGIONS = ['AFR', 'SEAR', 'EMR']  # Africa, SE Asia, Eastern Mediterranean
HIC_REGIONS = ['EUR', 'AMR', 'WPR']  # Europe, Americas, Western Pacific (mixed)


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_burden_score(dalys: float, deaths: float, prevalence: float) -> float:
    """
    Compute composite burden score from GBD metrics.
    
    Formula: (0.5 × DALYs) + (50 × Deaths) + (10 × log₁₀(Prevalence))
    
    Weights balance:
    - DALYs: comprehensive but can be dominated by high-prevalence conditions
    - Deaths: emphasizes mortality for fatal diseases
    - Prevalence: log-scaled to prevent dominance by common conditions
    """
    burden = 0.0
    
    # DALYs component (millions)
    burden += 0.5 * dalys
    
    # Deaths component (millions)
    burden += 50.0 * deaths
    
    # Prevalence component (log-scaled, millions)
    if prevalence > 0:
        burden += 10.0 * math.log10(prevalence + 1)
    
    return round(burden, 2)


def compute_gap_score(publications: int, burden_score: float, 
                      category: str, global_south_priority: bool) -> float:
    """
    Compute research gap score (0-100).
    
    Higher score = larger gap between burden and research attention.
    """
    # Zero publications = critical gap
    if publications == 0:
        return 95.0
    
    # Very few publications for infectious/neglected diseases
    if category in ['Infectious', 'Neglected'] and publications < 50:
        return 90.0
    
    # Global South priority with low coverage
    if global_south_priority and publications < 100:
        return 85.0
    
    # Calculate intensity-based score
    if burden_score > 0:
        intensity = publications / burden_score
        
        # Intensity thresholds (publications per burden unit)
        if intensity < 0.1:
            base_score = 80
        elif intensity < 0.5:
            base_score = 65
        elif intensity < 1.0:
            base_score = 50
        elif intensity < 2.0:
            base_score = 35
        elif intensity < 5.0:
            base_score = 20
        else:
            base_score = 10
    else:
        base_score = 50
    
    # Category adjustments
    if category in ['Infectious', 'Neglected', 'Maternal']:
        base_score = min(100, base_score + 10)
    
    if global_south_priority:
        base_score = min(100, base_score + 5)
    
    return round(base_score, 1)


def classify_gap_severity(gap_score: float) -> str:
    """Classify gap score into severity category."""
    if gap_score > GAP_CRITICAL:
        return 'Critical'
    elif gap_score > GAP_HIGH:
        return 'High'
    elif gap_score > GAP_MODERATE:
        return 'Moderate'
    else:
        return 'Low'


def compute_research_opportunity_score(biobank_diseases: dict, 
                                        disease_registry: dict,
                                        disease_gap_scores: dict) -> float:
    """
    Compute Research Opportunity Score (ROS) for a biobank.
    
    ROS = sum of burden scores for diseases with ≤2 publications
    Higher ROS = more unrealized potential to address high-burden conditions
    """
    ros = 0.0
    
    for disease_id, disease in disease_registry.items():
        pub_count = biobank_diseases.get(disease_id, 0)
        gap_score = disease_gap_scores.get(disease_id, 0)
        
        # Only count under-researched diseases
        if pub_count <= 2 and gap_score > GAP_HIGH:
            burden = compute_burden_score(
                disease.get('dalys_millions', 0),
                disease.get('deaths_millions', 0),
                disease.get('prevalence_millions', 0)
            )
            ros += burden
    
    return round(ros, 2)


def compute_equity_ratio(hic_intensity: float, gs_intensity: float) -> float:
    """
    Compute equity ratio between HIC and Global South research intensity.
    
    Ratio > 1 indicates HIC has more research per burden unit.
    """
    if gs_intensity > 0:
        return round(hic_intensity / gs_intensity, 2)
    elif hic_intensity > 0:
        return float('inf')
    else:
        return 1.0


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def main():
    print("=" * 70)
    print("BHEM STEP 3: Compute Research Gap & Equity Metrics")
    print("=" * 70)
    
    # Check input files
    if not INPUT_PUBLICATIONS.exists():
        print(f"❌ Input file not found: {INPUT_PUBLICATIONS}")
        print(f"   Run 02-01-bhem-map-diseases.py first")
        return
    
    if not INPUT_DISEASES.exists():
        print(f"❌ Disease registry not found: {INPUT_DISEASES}")
        print(f"   Run 02-01-bhem-map-diseases.py first")
        return
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(INPUT_PUBLICATIONS)
    print(f"   Publications: {len(df):,}")
    
    with open(INPUT_DISEASES, 'r') as f:
        disease_registry = json.load(f)
    print(f"   Diseases: {len(disease_registry)}")
    
    # Get unique biobanks
    biobanks = df['biobank_id'].unique()
    print(f"   Biobanks: {len(biobanks)}")
    
    # ==========================================================================
    # DISEASE-LEVEL METRICS
    # ==========================================================================
    print(f"\n🔬 Computing disease-level metrics...")
    
    disease_metrics = {}
    
    for disease_id, disease in disease_registry.items():
        # Count publications for this disease
        pub_count = 0
        for disease_list_str in df['disease_ids_str'].dropna():
            if disease_id in disease_list_str.split('|'):
                pub_count += 1
        
        # Compute burden score
        burden = compute_burden_score(
            disease.get('dalys_millions', 0),
            disease.get('deaths_millions', 0),
            disease.get('prevalence_millions', 0)
        )
        
        # Compute gap score
        gap = compute_gap_score(
            pub_count,
            burden,
            disease.get('category', 'Other'),
            disease.get('global_south_priority', False)
        )
        
        # Research intensity
        intensity = pub_count / burden if burden > 0 else 0
        
        disease_metrics[disease_id] = {
            'name': disease['name'],
            'category': disease.get('category', 'Other'),
            'publications': pub_count,
            'dalys_millions': disease.get('dalys_millions', 0),
            'deaths_millions': disease.get('deaths_millions', 0),
            'prevalence_millions': disease.get('prevalence_millions', 0),
            'burden_score': burden,
            'gap_score': gap,
            'gap_severity': classify_gap_severity(gap),
            'research_intensity': round(intensity, 4),
            'global_south_priority': disease.get('global_south_priority', False)
        }
    
    # Print disease summary
    print(f"\n📊 Disease Gap Analysis:")
    gap_counts = defaultdict(int)
    for dm in disease_metrics.values():
        gap_counts[dm['gap_severity']] += 1
    
    for severity in ['Critical', 'High', 'Moderate', 'Low']:
        count = gap_counts.get(severity, 0)
        pct = count / len(disease_metrics) * 100
        print(f"   {severity}: {count} diseases ({pct:.1f}%)")
    
    # Top gaps
    print(f"\n🚨 Top 5 Critical Gap Diseases:")
    sorted_diseases = sorted(disease_metrics.items(), 
                            key=lambda x: x[1]['gap_score'], reverse=True)
    for disease_id, dm in sorted_diseases[:5]:
        print(f"   {dm['name']}: gap={dm['gap_score']}, pubs={dm['publications']}")
    
    # ==========================================================================
    # BIOBANK-LEVEL METRICS
    # ==========================================================================
    print(f"\n🏥 Computing biobank-level metrics...")
    
    biobank_metrics = {}
    
    for biobank_id in biobanks:
        biobank_df = df[df['biobank_id'] == biobank_id]
        
        # Basic counts
        total_pubs = len(biobank_df)
        
        # Disease coverage
        biobank_diseases = defaultdict(int)
        for disease_list_str in biobank_df['disease_ids_str'].dropna():
            for disease_id in disease_list_str.split('|'):
                if disease_id:
                    biobank_diseases[disease_id] += 1
        
        diseases_covered = len([d for d in biobank_diseases if biobank_diseases[d] > 0])
        
        # Critical gaps for this biobank
        critical_gaps = []
        for disease_id, dm in disease_metrics.items():
            if biobank_diseases.get(disease_id, 0) <= 2 and dm['gap_score'] > GAP_CRITICAL:
                critical_gaps.append(disease_id)
        
        # Research Opportunity Score
        ros = compute_research_opportunity_score(
            biobank_diseases, disease_registry, 
            {d: dm['gap_score'] for d, dm in disease_metrics.items()}
        )
        
        # Global South priority coverage
        gs_pubs = biobank_df['global_south_priority'].sum()
        gs_pct = (gs_pubs / total_pubs * 100) if total_pubs > 0 else 0
        
        # Year distribution
        year_dist = biobank_df.groupby('year').size().to_dict()
        
        # Get biobank info from first row
        first_row = biobank_df.iloc[0]
        
        biobank_metrics[biobank_id] = {
            'name': first_row.get('biobank_name', biobank_id),
            'country': first_row.get('country', 'Unknown'),
            'region': first_row.get('region', 'Unknown'),
            'total_publications': total_pubs,
            'diseases_covered': diseases_covered,
            'disease_publications': dict(biobank_diseases),
            'critical_gaps': critical_gaps,
            'critical_gap_count': len(critical_gaps),
            'research_opportunity_score': ros,
            'global_south_publications': int(gs_pubs),
            'global_south_percentage': round(gs_pct, 1),
            'year_distribution': year_dist
        }
    
    # Print biobank summary
    print(f"\n📊 Biobank Summary:")
    sorted_biobanks = sorted(biobank_metrics.items(), 
                            key=lambda x: x[1]['total_publications'], reverse=True)
    for biobank_id, bm in sorted_biobanks[:10]:
        print(f"   {bm['name']}: {bm['total_publications']:,} pubs, "
              f"ROS={bm['research_opportunity_score']:.1f}, "
              f"gaps={bm['critical_gap_count']}")
    
    # ==========================================================================
    # GLOBAL METRICS
    # ==========================================================================
    print(f"\n🌍 Computing global equity metrics...")
    
    # Publications by region
    region_pubs = df.groupby('region').size().to_dict()
    
    # Calculate equity ratio
    hic_pubs = sum(region_pubs.get(r, 0) for r in ['EUR', 'AMR'])
    gs_pubs = sum(region_pubs.get(r, 0) for r in GLOBAL_SOUTH_REGIONS)
    
    # Get total burden for each region type (simplified)
    hic_diseases = [d for d, dm in disease_metrics.items() if not dm['global_south_priority']]
    gs_diseases = [d for d, dm in disease_metrics.items() if dm['global_south_priority']]
    
    hic_burden = sum(disease_metrics[d]['burden_score'] for d in hic_diseases)
    gs_burden = sum(disease_metrics[d]['burden_score'] for d in gs_diseases)
    
    hic_intensity = hic_pubs / hic_burden if hic_burden > 0 else 0
    gs_intensity = gs_pubs / gs_burden if gs_burden > 0 else 0
    
    equity_ratio = compute_equity_ratio(hic_intensity, gs_intensity)
    
    # Year trends
    year_totals = df.groupby('year').size().to_dict()
    
    # Summary statistics
    global_metrics = {
        'total_publications': len(df),
        'total_biobanks': len(biobanks),
        'total_diseases': len(disease_registry),
        'publications_by_region': region_pubs,
        'hic_publications': hic_pubs,
        'global_south_publications': gs_pubs,
        'equity_ratio': equity_ratio,
        'gap_distribution': dict(gap_counts),
        'year_totals': year_totals,
        'critical_gap_diseases': [d for d, dm in disease_metrics.items() 
                                  if dm['gap_severity'] == 'Critical'],
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n📊 Global Summary:")
    print(f"   Total publications: {global_metrics['total_publications']:,}")
    print(f"   HIC publications: {hic_pubs:,}")
    print(f"   Global South publications: {gs_pubs:,}")
    print(f"   Equity ratio: {equity_ratio:.1f}x")
    print(f"   Critical gap diseases: {len(global_metrics['critical_gap_diseases'])}")
    
    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================
    print(f"\n💾 Saving metrics...")
    
    # Save comprehensive JSON
    all_metrics = {
        'global': global_metrics,
        'diseases': disease_metrics,
        'biobanks': biobank_metrics
    }
    
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"   ✅ {OUTPUT_METRICS}")
    
    # Save disease metrics CSV
    disease_df = pd.DataFrame([
        {**{'disease_id': d}, **dm} 
        for d, dm in disease_metrics.items()
    ])
    disease_df.to_csv(OUTPUT_DISEASE_CSV, index=False)
    print(f"   ✅ {OUTPUT_DISEASE_CSV}")
    
    # Save biobank metrics CSV
    biobank_rows = []
    for biobank_id, bm in biobank_metrics.items():
        row = {
            'biobank_id': biobank_id,
            'name': bm['name'],
            'country': bm['country'],
            'region': bm['region'],
            'total_publications': bm['total_publications'],
            'diseases_covered': bm['diseases_covered'],
            'critical_gap_count': bm['critical_gap_count'],
            'research_opportunity_score': bm['research_opportunity_score'],
            'global_south_percentage': bm['global_south_percentage']
        }
        biobank_rows.append(row)
    
    biobank_df = pd.DataFrame(biobank_rows)
    biobank_df.to_csv(OUTPUT_BIOBANK_CSV, index=False)
    print(f"   ✅ {OUTPUT_BIOBANK_CSV}")
    
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Next step: python 02-04-bhem-generate-json.py")


if __name__ == "__main__":
    main()
