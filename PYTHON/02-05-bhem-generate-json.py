#!/usr/bin/env python3
"""
02-03-bhem-generate-json.py
===========================
BHEM Step 4: Generate JSON files for static GitHub Pages site

Creates optimized JSON files that power the interactive dashboard.
These files are served directly from GitHub Pages.

INPUT:  DATA/bhem_publications_mapped.csv
        DATA/bhem_metrics.json
OUTPUT: docs/data/biobanks.json
        docs/data/diseases.json
        docs/data/matrix.json
        docs/data/trends.json
        docs/data/summary.json

USAGE:
    python 02-03-bhem-generate-json.py
"""

import json
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
DOCS_DIR = BASE_DIR / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

# Create output directories
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input files
INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"
INPUT_METRICS = DATA_DIR / "bhem_metrics.json"

# Output files
OUTPUT_BIOBANKS = DOCS_DATA_DIR / "biobanks.json"
OUTPUT_DISEASES = DOCS_DATA_DIR / "diseases.json"
OUTPUT_MATRIX = DOCS_DATA_DIR / "matrix.json"
OUTPUT_TRENDS = DOCS_DATA_DIR / "trends.json"
OUTPUT_SUMMARY = DOCS_DATA_DIR / "summary.json"


# =============================================================================
# JSON GENERATION FUNCTIONS
# =============================================================================

def generate_biobanks_json(metrics: dict, df: pd.DataFrame) -> dict:
    """Generate biobanks.json with all biobank data."""
    biobanks = []
    
    for biobank_id, bm in metrics['biobanks'].items():
        biobank_df = df[df['biobank_id'] == biobank_id]
        
        # Get top diseases for this biobank
        disease_pubs = bm.get('disease_publications', {})
        top_diseases = sorted(disease_pubs.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        biobank = {
            'id': biobank_id,
            'name': bm['name'],
            'country': bm['country'],
            'region': bm['region'],
            'stats': {
                'totalPublications': bm['total_publications'],
                'diseasesCovered': bm['diseases_covered'],
                'criticalGaps': bm['critical_gap_count'],
                'ros': bm['research_opportunity_score'],
                'globalSouthPct': bm['global_south_percentage']
            },
            'topDiseases': [
                {'id': d, 'count': c} for d, c in top_diseases
            ],
            'criticalGapDiseases': bm.get('critical_gaps', []),
            'yearTrend': [
                {'year': int(y), 'count': c} 
                for y, c in sorted(bm.get('year_distribution', {}).items())
            ]
        }
        biobanks.append(biobank)
    
    # Sort by publications descending
    biobanks.sort(key=lambda x: x['stats']['totalPublications'], reverse=True)
    
    return {
        'biobanks': biobanks,
        'totalCount': len(biobanks),
        'generated': datetime.now().isoformat()
    }


def generate_diseases_json(metrics: dict) -> dict:
    """Generate diseases.json with all disease data."""
    diseases = []
    
    for disease_id, dm in metrics['diseases'].items():
        disease = {
            'id': disease_id,
            'name': dm['name'],
            'category': dm['category'],
            'burden': {
                'dalys': dm['dalys_millions'],
                'deaths': dm['deaths_millions'],
                'prevalence': dm['prevalence_millions'],
                'score': dm['burden_score']
            },
            'research': {
                'publications': dm['publications'],
                'intensity': dm['research_intensity']
            },
            'gap': {
                'score': dm['gap_score'],
                'severity': dm['gap_severity']
            },
            'globalSouthPriority': dm['global_south_priority']
        }
        diseases.append(disease)
    
    # Sort by gap score descending
    diseases.sort(key=lambda x: x['gap']['score'], reverse=True)
    
    # Group by category
    categories = defaultdict(list)
    for d in diseases:
        categories[d['category']].append(d['id'])
    
    return {
        'diseases': diseases,
        'categories': dict(categories),
        'totalCount': len(diseases),
        'generated': datetime.now().isoformat()
    }


def generate_matrix_json(metrics: dict, df: pd.DataFrame) -> dict:
    """Generate matrix.json with disease-biobank publication matrix."""
    # Get all biobanks and diseases
    biobank_ids = list(metrics['biobanks'].keys())
    disease_ids = list(metrics['diseases'].keys())
    
    # Build matrix
    matrix = []
    
    for biobank_id in biobank_ids:
        biobank_df = df[df['biobank_id'] == biobank_id]
        
        row = {'biobank': biobank_id, 'values': {}}
        
        for disease_id in disease_ids:
            # Count publications for this disease
            count = 0
            for disease_list_str in biobank_df['disease_ids_str'].dropna():
                if disease_id in disease_list_str.split('|'):
                    count += 1
            row['values'][disease_id] = count
        
        matrix.append(row)
    
    # Sort by total publications
    matrix.sort(key=lambda x: sum(x['values'].values()), reverse=True)
    
    return {
        'matrix': matrix,
        'biobanks': biobank_ids,
        'diseases': disease_ids,
        'generated': datetime.now().isoformat()
    }


def generate_trends_json(metrics: dict, df: pd.DataFrame) -> dict:
    """Generate trends.json with time series data."""
    # Global yearly totals
    year_totals = df.groupby('year').size().to_dict()
    
    # Cumulative totals
    cumulative = {}
    running_total = 0
    for year in sorted(year_totals.keys()):
        running_total += year_totals[year]
        cumulative[year] = running_total
    
    # By biobank
    biobank_trends = {}
    for biobank_id in metrics['biobanks'].keys():
        biobank_df = df[df['biobank_id'] == biobank_id]
        biobank_trends[biobank_id] = biobank_df.groupby('year').size().to_dict()
    
    # By region
    region_trends = {}
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        region_trends[region] = region_df.groupby('year').size().to_dict()
    
    # By disease category (for global south priority)
    gs_trend = df[df['global_south_priority'] == True].groupby('year').size().to_dict()
    
    return {
        'yearly': [
            {'year': int(y), 'count': c, 'cumulative': cumulative[y]}
            for y, c in sorted(year_totals.items())
        ],
        'byBiobank': {
            bid: [{'year': int(y), 'count': c} for y, c in sorted(vals.items())]
            for bid, vals in biobank_trends.items()
        },
        'byRegion': {
            region: [{'year': int(y), 'count': c} for y, c in sorted(vals.items())]
            for region, vals in region_trends.items()
        },
        'globalSouth': [
            {'year': int(y), 'count': c} for y, c in sorted(gs_trend.items())
        ],
        'yearRange': {
            'min': int(df['year'].min()),
            'max': int(df['year'].max())
        },
        'generated': datetime.now().isoformat()
    }


def generate_summary_json(metrics: dict, df: pd.DataFrame) -> dict:
    """Generate summary.json with dashboard summary statistics."""
    global_metrics = metrics['global']
    disease_metrics = metrics['diseases']
    biobank_metrics = metrics['biobanks']
    
    # Gap distribution
    gap_dist = defaultdict(int)
    for dm in disease_metrics.values():
        gap_dist[dm['gap_severity']] += 1
    
    # Top biobanks
    top_biobanks = sorted(
        biobank_metrics.items(),
        key=lambda x: x[1]['total_publications'],
        reverse=True
    )[:5]
    
    # Top gaps
    top_gaps = sorted(
        disease_metrics.items(),
        key=lambda x: x[1]['gap_score'],
        reverse=True
    )[:5]
    
    # Region distribution
    region_counts = df['region'].value_counts().to_dict()
    
    return {
        'totals': {
            'publications': len(df),
            'biobanks': len(biobank_metrics),
            'diseases': len(disease_metrics),
            'countries': df['country'].nunique(),
            'yearsCovered': int(df['year'].max()) - int(df['year'].min()) + 1
        },
        'equity': {
            'ratio': global_metrics.get('equity_ratio', 1.0),
            'hicPublications': global_metrics.get('hic_publications', 0),
            'globalSouthPublications': global_metrics.get('global_south_publications', 0)
        },
        'gaps': {
            'distribution': dict(gap_dist),
            'criticalCount': gap_dist.get('Critical', 0),
            'topGaps': [
                {'id': d, 'name': dm['name'], 'score': dm['gap_score']}
                for d, dm in top_gaps
            ]
        },
        'topBiobanks': [
            {
                'id': bid,
                'name': bm['name'],
                'publications': bm['total_publications'],
                'ros': bm['research_opportunity_score']
            }
            for bid, bm in top_biobanks
        ],
        'regions': region_counts,
        'lastUpdate': datetime.now().isoformat()
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BHEM STEP 4: Generate JSON Files for Static Site")
    print("=" * 70)
    
    # Check input files
    if not INPUT_PUBLICATIONS.exists():
        print(f"❌ Input file not found: {INPUT_PUBLICATIONS}")
        print(f"   Run previous steps first")
        return
    
    if not INPUT_METRICS.exists():
        print(f"❌ Metrics file not found: {INPUT_METRICS}")
        print(f"   Run 02-02-bhem-compute-metrics.py first")
        return
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(INPUT_PUBLICATIONS)
    print(f"   Publications: {len(df):,}")
    
    with open(INPUT_METRICS, 'r') as f:
        metrics = json.load(f)
    print(f"   Biobanks: {len(metrics['biobanks'])}")
    print(f"   Diseases: {len(metrics['diseases'])}")
    
    # Generate JSON files
    print(f"\n📝 Generating JSON files...")
    
    # 1. Biobanks
    print(f"   Generating biobanks.json...")
    biobanks_data = generate_biobanks_json(metrics, df)
    with open(OUTPUT_BIOBANKS, 'w') as f:
        json.dump(biobanks_data, f, indent=2)
    print(f"   ✅ {OUTPUT_BIOBANKS} ({len(biobanks_data['biobanks'])} biobanks)")
    
    # 2. Diseases
    print(f"   Generating diseases.json...")
    diseases_data = generate_diseases_json(metrics)
    with open(OUTPUT_DISEASES, 'w') as f:
        json.dump(diseases_data, f, indent=2)
    print(f"   ✅ {OUTPUT_DISEASES} ({len(diseases_data['diseases'])} diseases)")
    
    # 3. Matrix
    print(f"   Generating matrix.json...")
    matrix_data = generate_matrix_json(metrics, df)
    with open(OUTPUT_MATRIX, 'w') as f:
        json.dump(matrix_data, f, indent=2)
    print(f"   ✅ {OUTPUT_MATRIX}")
    
    # 4. Trends
    print(f"   Generating trends.json...")
    trends_data = generate_trends_json(metrics, df)
    with open(OUTPUT_TRENDS, 'w') as f:
        json.dump(trends_data, f, indent=2)
    print(f"   ✅ {OUTPUT_TRENDS}")
    
    # 5. Summary
    print(f"   Generating summary.json...")
    summary_data = generate_summary_json(metrics, df)
    with open(OUTPUT_SUMMARY, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"   ✅ {OUTPUT_SUMMARY}")
    
    # Print summary
    print(f"\n📊 Summary Statistics:")
    print(f"   Total publications: {summary_data['totals']['publications']:,}")
    print(f"   Biobanks: {summary_data['totals']['biobanks']}")
    print(f"   Countries: {summary_data['totals']['countries']}")
    print(f"   Critical gaps: {summary_data['gaps']['criticalCount']}")
    print(f"   Equity ratio: {summary_data['equity']['ratio']:.1f}x")
    
    print(f"\n📁 Output directory: {DOCS_DATA_DIR}")
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Next step: python 02-04-bhem-build-site.py")


if __name__ == "__main__":
    main()
