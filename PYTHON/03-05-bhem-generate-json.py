#!/usr/bin/env python3
"""
03-05-bhem-generate-json.py
===========================
HEIM-Biobank v2.0 (IHCC): Generate Dashboard JSON Files

Transforms computed metrics into 8 JSON files optimized for the web dashboard.

INPUT:
    DATA/bhem_metrics.json
    DATA/bhem_publications_mapped.csv
    DATA/bhem_themes.json (optional)

OUTPUT (docs/data/):
    1. summary.json     - Global statistics and overview
    2. biobanks.json    - Per-biobank detailed data
    3. diseases.json    - Per-disease detailed data
    4. matrix.json      - Biobank × Disease publication matrix
    5. trends.json      - Publication trends over time
    6. themes.json      - MeSH theme analysis
    7. comparison.json  - Biobank comparison pairs
    8. equity.json      - HIC vs LMIC equity analysis

USAGE:
    python 03-05-bhem-generate-json.py

VERSION: HEIM-Biobank v2.0 (IHCC)
DATE: 2025-12-24
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

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

VERSION = "HEIM-Biobank v2.0 (IHCC)"
VERSION_DATE = "2026-01-07"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = DOCS_DIR / "data"

# Year filtering
MIN_YEAR = 2000
MAX_YEAR = 2025
EXCLUDE_YEARS = [2026]

# Input files
INPUT_METRICS = DATA_DIR / "bhem_metrics.json"
INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"
INPUT_THEMES = DATA_DIR / "bhem_themes.json"

# WHO regions
WHO_REGIONS = {
    "AFR": "Africa",
    "AMR": "Americas",
    "SEAR": "South-East Asia",
    "EUR": "Europe",
    "EMR": "Eastern Mediterranean",
    "WPR": "Western Pacific"
}

# Income classification (simplified)
HIC_REGIONS = {"EUR", "WPR"}  # Primarily HIC
LMIC_REGIONS = {"AFR", "SEAR", "EMR"}  # Primarily LMIC
MIXED_REGIONS = {"AMR"}  # Mixed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_json(filepath: Path) -> Optional[Dict]:
    """Load JSON file if exists."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def load_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Load CSV file if exists, with year filtering."""
    if filepath.exists():
        df = pd.read_csv(filepath)
        
        # Apply year filter
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[
                (df['year'] >= MIN_YEAR) & 
                (df['year'] <= MAX_YEAR) & 
                (~df['year'].isin(EXCLUDE_YEARS))
            ]
        
        # Handle IHCC cohort columns
        if 'cohort_id' in df.columns and 'biobank_id' not in df.columns:
            df['biobank_id'] = df['cohort_id']
        if 'cohort_name' in df.columns and 'biobank_name' not in df.columns:
            df['biobank_name'] = df['cohort_name']
        
        return df
    return None


def save_json(data: Dict, filepath: Path) -> None:
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {filepath}")


def get_severity_color(severity: str) -> str:
    """Get color code for gap severity."""
    colors = {
        "Critical": "#dc3545",  # Red
        "High": "#fd7e14",      # Orange
        "Moderate": "#ffc107",  # Yellow
        "Low": "#28a745"        # Green
    }
    return colors.get(severity, "#6c757d")


def get_eas_color(category: str) -> str:
    """Get color code for EAS category."""
    colors = {
        "Strong": "#28a745",    # Green
        "Moderate": "#17a2b8",  # Teal
        "Weak": "#ffc107",      # Yellow
        "Poor": "#dc3545"       # Red
    }
    return colors.get(category, "#6c757d")


# =============================================================================
# JSON GENERATORS
# =============================================================================

def generate_summary_json(metrics: Dict, pubs_df: pd.DataFrame) -> Dict:
    """
    Generate summary.json - Global overview and key statistics.
    Used by: Overview tab (landing page)
    """
    logger.info("Generating summary.json...")
    
    global_data = metrics.get('global', {})
    biobanks = metrics.get('biobanks', {})
    diseases = metrics.get('diseases', {})
    
    # Count by region
    region_counts = defaultdict(int)
    region_pubs = defaultdict(int)
    for bid, bdata in biobanks.items():
        region = bdata.get('region', 'Unknown')
        region_counts[region] += 1
        region_pubs[region] += bdata.get('total_publications', 0)
    
    # Count by country
    country_counts = defaultdict(int)
    for bid, bdata in biobanks.items():
        country = bdata.get('country', 'Unknown')
        country_counts[country] += 1
    
    # Top critical gaps
    critical_gaps = []
    for did, ddata in diseases.items():
        if ddata.get('gap_severity') == 'Critical':
            critical_gaps.append({
                'id': did,
                'name': ddata.get('name', did),
                'gapScore': ddata.get('gap_score', 0),
                'dalys': ddata.get('dalys_millions', 0),
                'publications': ddata.get('publications', 0)
            })
    critical_gaps.sort(key=lambda x: x['gapScore'], reverse=True)
    
    # Top biobanks by EAS
    top_biobanks = []
    for bid, bdata in sorted(biobanks.items(), 
                              key=lambda x: x[1].get('equity_alignment_score', 0), 
                              reverse=True)[:10]:
        top_biobanks.append({
            'id': bid,
            'name': bdata.get('name', bid),
            'eas': bdata.get('equity_alignment_score', 0),
            'category': bdata.get('equity_alignment_category', 'Unknown'),
            'publications': bdata.get('total_publications', 0)
        })
    
    summary = {
        'version': VERSION,
        'versionDate': VERSION_DATE,
        'generated': datetime.now().isoformat(),
        'overview': {
            'totalBiobanks': len(biobanks),
            'totalPublications': global_data.get('total_publications', 0),
            'totalDiseases': len(diseases),
            'totalCountries': len(country_counts),
            'equityRatio': global_data.get('equity_ratio', 0)
        },
        'gapDistribution': global_data.get('gap_distribution', {}),
        'easDistribution': global_data.get('eas_distribution', {}),
        'regionStats': [
            {
                'region': region,
                'name': WHO_REGIONS.get(region, region),
                'biobanks': count,
                'publications': region_pubs.get(region, 0)
            }
            for region, count in sorted(region_counts.items())
        ],
        'criticalGaps': critical_gaps[:10],
        'topBiobanks': top_biobanks,
        'averages': {
            'eas': global_data.get('average_eas', 0),
            'ros': global_data.get('average_ros', 0),
            'gapScore': global_data.get('average_gap_score', 0)
        },
        'methodology': global_data.get('methodology', {})
    }
    
    return summary


def generate_biobanks_json(metrics: Dict, pubs_df: pd.DataFrame) -> Dict:
    """
    Generate biobanks.json - Detailed per-biobank data.
    Used by: Biobanks tab
    """
    logger.info("Generating biobanks.json...")
    
    biobanks = metrics.get('biobanks', {})
    diseases = metrics.get('diseases', {})
    
    biobank_list = []
    
    for bid, bdata in biobanks.items():
        # Get disease-level stats for this biobank
        disease_pubs = bdata.get('disease_publications', {})
        
        critical_gaps = []
        high_gaps = []
        
        for did, ddata in diseases.items():
            pubs = disease_pubs.get(did, 0)
            if pubs == 0:
                critical_gaps.append(did)
            elif pubs <= 2 and ddata.get('gap_severity') in ['Critical', 'High']:
                high_gaps.append(did)
        
        # Determine data sufficiency
        total_pubs = bdata.get('total_publications', 0)
        if total_pubs >= 500:
            data_sufficiency = "Adequate"
        elif total_pubs >= 50:
            data_sufficiency = "Moderate"
        else:
            data_sufficiency = "Low"
        
        biobank_record = {
            'id': bid,
            'name': bdata.get('name', bid),
            'country': bdata.get('country', 'Unknown'),
            'region': bdata.get('region', 'Unknown'),
            'regionName': WHO_REGIONS.get(bdata.get('region', ''), 'Unknown'),
            'isGlobalSouth': bdata.get('is_global_south', False),
            'stats': {
                'totalPublications': total_pubs,
                'diseasesCovered': bdata.get('diseases_covered', 0),
                'criticalGaps': len(critical_gaps),
                'highGaps': len(high_gaps)
            },
            'scores': {
                'equityAlignment': bdata.get('equity_alignment_score', 0),
                'equityCategory': bdata.get('equity_alignment_category', 'Unknown'),
                'researchOpportunity': bdata.get('research_opportunity_score', 0)
            },
            'components': bdata.get('equity_alignment_components', {}),
            'criticalGapDiseases': critical_gaps[:5],
            'dataSufficiency': data_sufficiency,
            'color': get_eas_color(bdata.get('equity_alignment_category', 'Unknown'))
        }
        
        biobank_list.append(biobank_record)
    
    # Sort by EAS descending
    biobank_list.sort(key=lambda x: x['scores']['equityAlignment'], reverse=True)
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'count': len(biobank_list),
        'biobanks': biobank_list
    }


def load_research_appropriateness() -> Dict:
    """Load research appropriateness classifications."""
    appropriateness_file = DATA_DIR / "research_appropriateness.json"
    if appropriateness_file.exists():
        with open(appropriateness_file) as f:
            data = json.load(f)
            return data.get('disease_classifications', {})
    return {}


def generate_diseases_json(metrics: Dict) -> Dict:
    """
    Generate diseases.json - Detailed per-disease data.
    Used by: Diseases tab

    Now includes research appropriateness classification.
    """
    logger.info("Generating diseases.json...")

    diseases = metrics.get('diseases', {})
    biobanks = metrics.get('biobanks', {})

    # Load research appropriateness
    appropriateness = load_research_appropriateness()
    
    disease_list = []
    
    for did, ddata in diseases.items():
        # Count biobanks with publications for this disease
        biobanks_engaged = 0
        biobank_pubs = []
        
        for bid, bdata in biobanks.items():
            pubs = bdata.get('disease_publications', {}).get(did, 0)
            if pubs > 0:
                biobanks_engaged += 1
                biobank_pubs.append({
                    'id': bid,
                    'name': bdata.get('name', bid),
                    'publications': pubs
                })
        
        # Sort by publications
        biobank_pubs.sort(key=lambda x: x['publications'], reverse=True)
        
        # Get research appropriateness
        research_approp = appropriateness.get(did, 'high')

        # Adjust gap interpretation based on appropriateness
        raw_gap_score = ddata.get('gap_score', 0)
        if research_approp == 'limited':
            adjusted_gap_note = "Gap score reduced: disease not well-suited for biobank research"
        elif research_approp == 'moderate':
            adjusted_gap_note = "Moderate biobank relevance"
        else:
            adjusted_gap_note = None

        disease_record = {
            'id': did,
            'name': ddata.get('name', did),
            'category': ddata.get('category', 'Unknown'),
            'globalSouthPriority': ddata.get('global_south_priority', False),
            'researchAppropriateness': research_approp,
            'burden': {
                'dalysMillions': ddata.get('dalys_millions', 0),
                'deathsMillions': ddata.get('deaths_millions', 0),
                'prevalenceMillions': ddata.get('prevalence_millions', 0),
                'burdenScore': ddata.get('burden_score', 0)
            },
            'research': {
                'globalPublications': ddata.get('publications', 0),
                'biobanksEngaged': biobanks_engaged,
                'researchIntensity': ddata.get('research_intensity', 0)
            },
            'gap': {
                'score': raw_gap_score,
                'severity': ddata.get('gap_severity', 'Unknown'),
                'color': get_severity_color(ddata.get('gap_severity', 'Unknown')),
                'adjustedNote': adjusted_gap_note
            },
            'topBiobanks': biobank_pubs[:5]
        }
        
        disease_list.append(disease_record)
    
    # Sort by gap score descending (highest gaps first)
    disease_list.sort(key=lambda x: x['gap']['score'], reverse=True)
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'count': len(disease_list),
        'diseases': disease_list
    }


def generate_matrix_json(metrics: Dict) -> Dict:
    """
    Generate matrix.json - Biobank × Disease publication counts.
    Used by: Matrix tab (heatmap visualization)
    """
    logger.info("Generating matrix.json...")
    
    biobanks = metrics.get('biobanks', {})
    diseases = metrics.get('diseases', {})
    
    # Get ordered lists
    biobank_ids = sorted(biobanks.keys())
    disease_ids = sorted(diseases.keys())
    
    # Build matrix
    values = []
    gap_categories = []
    
    for bid in biobank_ids:
        row_values = []
        row_gaps = []
        bdata = biobanks[bid]
        disease_pubs = bdata.get('disease_publications', {})
        
        for did in disease_ids:
            pubs = disease_pubs.get(did, 0)
            row_values.append(pubs)
            
            # Determine gap category for this cell
            if pubs == 0:
                row_gaps.append("critical")
            elif pubs <= 2:
                row_gaps.append("high")
            elif pubs <= 10:
                row_gaps.append("moderate")
            else:
                row_gaps.append("low")
        
        values.append(row_values)
        gap_categories.append(row_gaps)
    
    # Build metadata
    biobank_meta = [
        {
            'id': bid,
            'name': biobanks[bid].get('name', bid),
            'eas': biobanks[bid].get('equity_alignment_score', 0)
        }
        for bid in biobank_ids
    ]
    
    disease_meta = [
        {
            'id': did,
            'name': diseases[did].get('name', did),
            'gapScore': diseases[did].get('gap_score', 0),
            'category': diseases[did].get('category', 'Unknown')
        }
        for did in disease_ids
    ]
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'dimensions': {
            'biobanks': len(biobank_ids),
            'diseases': len(disease_ids)
        },
        'biobanks': biobank_meta,
        'diseases': disease_meta,
        'matrix': {
            'values': values,
            'gapCategories': gap_categories
        },
        'colorScale': {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'moderate': '#ffc107',
            'low': '#28a745'
        }
    }


def generate_trends_json(pubs_df: pd.DataFrame, metrics: Dict) -> Dict:
    """
    Generate trends.json - Publication trends over time.
    Used by: Trends tab
    """
    logger.info("Generating trends.json...")
    
    biobanks = metrics.get('biobanks', {})
    
    # Overall trends by year
    if pubs_df is not None and 'year' in pubs_df.columns:
        yearly_total = pubs_df.groupby('year').size().to_dict()
        yearly_total = {int(k): int(v) for k, v in yearly_total.items() if pd.notna(k)}
    else:
        yearly_total = {}
    
    # Trends by biobank
    biobank_trends = {}
    if pubs_df is not None and 'biobank_id' in pubs_df.columns and 'year' in pubs_df.columns:
        for bid in biobanks.keys():
            bdf = pubs_df[pubs_df['biobank_id'] == bid]
            if len(bdf) > 0:
                yearly = bdf.groupby('year').size().to_dict()
                biobank_trends[bid] = {int(k): int(v) for k, v in yearly.items() if pd.notna(k)}
    
    # Use year distribution from biobank metrics as fallback
    if not biobank_trends:
        for bid, bdata in biobanks.items():
            year_dist = bdata.get('year_distribution', {})
            if year_dist:
                biobank_trends[bid] = {int(k): int(v) for k, v in year_dist.items()}
    
    # Aggregate yearly total if not from pubs_df
    if not yearly_total and biobank_trends:
        for bid, years in biobank_trends.items():
            for year, count in years.items():
                yearly_total[year] = yearly_total.get(year, 0) + count
    
    # Get all years
    all_years = sorted(set(yearly_total.keys()))
    
    # Calculate growth rates
    growth_rates = {}
    for i, year in enumerate(all_years[1:], 1):
        prev_year = all_years[i-1]
        if yearly_total.get(prev_year, 0) > 0:
            rate = ((yearly_total.get(year, 0) - yearly_total.get(prev_year, 0)) / 
                    yearly_total.get(prev_year, 1)) * 100
            growth_rates[year] = round(rate, 1)
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'yearRange': {
            'start': min(all_years) if all_years else 2000,
            'end': max(all_years) if all_years else 2024
        },
        'global': {
            'yearly': yearly_total,
            'total': sum(yearly_total.values()),
            'growthRates': growth_rates
        },
        'byBiobank': {
            bid: {
                'name': biobanks.get(bid, {}).get('name', bid),
                'yearly': years
            }
            for bid, years in biobank_trends.items()
        }
    }


def generate_themes_json(themes_data: Optional[Dict], metrics: Dict) -> Dict:
    """
    Generate themes.json - MeSH theme analysis.
    Used by: Themes tab
    
    Dashboard expects:
    {
        "themes": [{"id": ..., "name": ..., "publications": ..., "diseaseCount": ...}],
        "byBiobank": {...}
    }
    
    02-02 produces:
    {
        "theme_definitions": {...},
        "biobank_themes": {...},
        "knowledge_metrics": {...}
    }
    
    This function transforms 02-02 format OR generates from disease categories.
    """
    logger.info("Generating themes.json...")
    
    biobanks = metrics.get('biobanks', {})
    diseases = metrics.get('diseases', {})
    
    # Transform themes data from 02-02 format if available
    if themes_data and 'theme_definitions' in themes_data:
        logger.info("  Transforming 02-02 theme analysis format...")
        
        # Get theme definitions from 02-02
        theme_defs = themes_data.get('theme_definitions', {})
        biobank_themes = themes_data.get('biobank_themes', {})
        knowledge_metrics = themes_data.get('knowledge_metrics', {})
        
        # Aggregate theme publications across all biobanks
        # 02-02 structure: biobank_themes[biobank_id][theme_id] = {'name': ..., 'count': N, 'percentage': X}
        theme_pubs_total = defaultdict(int)
        theme_biobank_count = defaultdict(set)
        
        for biobank_id, themes_dict in biobank_themes.items():
            # themes_dict is like: {'genomics_gwas': {'name': '...', 'count': 123, 'percentage': 45.2}, ...}
            for theme_id, theme_info in themes_dict.items():
                if isinstance(theme_info, dict):
                    count = theme_info.get('count', 0)
                    theme_pubs_total[theme_id] += count
                    if count > 0:
                        theme_biobank_count[theme_id].add(biobank_id)
        
        # Build themes array in dashboard format
        themes = []
        for theme_id, theme_info in theme_defs.items():
            themes.append({
                'id': theme_id,
                'name': theme_info.get('name', theme_id),
                'category': theme_info.get('category', 'unknown'),
                'publications': theme_pubs_total.get(theme_id, 0),
                'diseaseCount': len(theme_biobank_count.get(theme_id, set()))  # Number of biobanks covering this theme
            })
        
        # Sort by publications
        themes.sort(key=lambda x: x['publications'], reverse=True)
        
        # Build byBiobank coverage
        biobank_theme_coverage = {}
        for biobank_id, themes_dict in biobank_themes.items():
            biobank_name = biobank_id
            if biobank_id in knowledge_metrics:
                biobank_name = knowledge_metrics[biobank_id].get('biobank_name', biobank_id)
            
            # Extract theme counts
            theme_pubs = {}
            for theme_id, theme_info in themes_dict.items():
                if isinstance(theme_info, dict):
                    theme_pubs[theme_id] = theme_info.get('count', 0)
            
            biobank_theme_coverage[biobank_id] = {
                'name': biobank_name,
                'themes': theme_pubs
            }
        
        return {
            'version': VERSION,
            'generated': datetime.now().isoformat(),
            'source': 'bhem_themes.json (transformed)',
            'themes': themes,
            'byBiobank': biobank_theme_coverage,
            'raw': {
                'theme_definitions': theme_defs,
                'total_biobanks': len(biobank_themes)
            }
        }
    
    # FALLBACK: Generate themes from disease categories
    logger.info("  Generating themes from disease categories...")
    
    category_counts = defaultdict(int)
    category_diseases = defaultdict(int)
    
    for did, ddata in diseases.items():
        cat = ddata.get('category', 'Other')
        category_counts[cat] += ddata.get('publications', 0)
        category_diseases[cat] += 1
    
    themes = []
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        themes.append({
            'id': cat.lower().replace(' ', '_').replace('/', '_'),
            'name': cat,
            'publications': count,
            'diseaseCount': category_diseases[cat]
        })
    
    # Theme coverage by biobank
    biobank_theme_coverage = {}
    for bid, bdata in biobanks.items():
        disease_pubs = bdata.get('disease_publications', {})
        theme_pubs = defaultdict(int)
        
        for did, pubs in disease_pubs.items():
            if did in diseases:
                cat = diseases[did].get('category', 'Other')
                theme_pubs[cat] += pubs
        
        biobank_theme_coverage[bid] = {
            'name': bdata.get('name', bid),
            'themes': dict(theme_pubs)
        }
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'source': 'generated_from_disease_categories',
        'themes': themes,
        'byBiobank': biobank_theme_coverage
    }


def generate_comparison_json(metrics: Dict) -> Dict:
    """
    Generate comparison.json - Biobank comparison data.
    Used by: Compare tab (side-by-side comparison)
    """
    logger.info("Generating comparison.json...")
    
    biobanks = metrics.get('biobanks', {})
    diseases = metrics.get('diseases', {})
    
    # Pre-compute comparison data for each biobank
    comparison_data = []
    
    for bid, bdata in biobanks.items():
        disease_pubs = bdata.get('disease_publications', {})
        
        # Radar chart dimensions
        total_pubs = bdata.get('total_publications', 0)
        max_pubs = max(b.get('total_publications', 1) for b in biobanks.values())
        
        radar = {
            'coverage': min(100, bdata.get('diseases_covered', 0) / 25 * 100),
            'volume': min(100, total_pubs / max_pubs * 100) if max_pubs > 0 else 0,
            'equity': bdata.get('equity_alignment_score', 0),
            'depth': min(100, 100 - bdata.get('equity_alignment_components', {}).get('capacity_penalty', 100)),
            'gsEngagement': bdata.get('global_south_percentage', 0)
        }
        
        comparison_data.append({
            'id': bid,
            'name': bdata.get('name', bid),
            'country': bdata.get('country', 'Unknown'),
            'region': bdata.get('region', 'Unknown'),
            'stats': {
                'publications': total_pubs,
                'diseases': bdata.get('diseases_covered', 0),
                'eas': bdata.get('equity_alignment_score', 0),
                'ros': bdata.get('research_opportunity_score', 0),
                'criticalGaps': bdata.get('critical_gap_count', 0)
            },
            'radar': radar,
            'diseasePublications': disease_pubs
        })
    
    # Find similar biobank pairs
    similar_pairs = []
    biobank_list = list(comparison_data)
    
    for i, b1 in enumerate(biobank_list):
        for b2 in biobank_list[i+1:]:
            # Calculate similarity based on radar chart
            similarity = 0
            for key in ['coverage', 'volume', 'equity', 'depth', 'gsEngagement']:
                diff = abs(b1['radar'][key] - b2['radar'][key])
                similarity += (100 - diff)
            similarity /= 5  # Average similarity
            
            if similarity > 60:  # Threshold for "similar"
                similar_pairs.append({
                    'biobank1': b1['id'],
                    'biobank2': b2['id'],
                    'similarity': round(similarity, 1)
                })
    
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'biobanks': comparison_data,
        'similarPairs': similar_pairs[:20],
        'radarDimensions': [
            {'key': 'coverage', 'label': 'Disease Coverage', 'description': 'Percentage of 25 diseases with publications'},
            {'key': 'volume', 'label': 'Publication Volume', 'description': 'Relative publication count'},
            {'key': 'equity', 'label': 'Equity Alignment', 'description': 'Equity Alignment Score (0-100)'},
            {'key': 'depth', 'label': 'Research Depth', 'description': 'Average publications per disease'},
            {'key': 'gsEngagement', 'label': 'Global South Focus', 'description': 'Percentage of Global South priority diseases'}
        ]
    }


def generate_equity_json(metrics: Dict) -> Dict:
    """
    Generate equity.json - HIC vs LMIC equity analysis.
    Used by: Equity tab
    
    Now uses income_level column from IHCC cohort registry if available.
    """
    logger.info("Generating equity.json...")
    
    global_data = metrics.get('global', {})
    biobanks = metrics.get('biobanks', {})
    diseases = metrics.get('diseases', {})
    
    # Categorize biobanks by income level
    hic_biobanks = []
    lmic_biobanks = []
    
    for bid, bdata in biobanks.items():
        region = bdata.get('region', 'Unknown')
        is_gs = bdata.get('is_global_south', False)
        income_level = bdata.get('income_level', '')  # From IHCC registry
        
        entry = {
            'id': bid,
            'name': bdata.get('name', bid),
            'country': bdata.get('country', 'Unknown'),
            'region': region,
            'incomeLevel': income_level,
            'publications': bdata.get('total_publications', 0),
            'eas': bdata.get('equity_alignment_score', 0)
        }
        
        # Use income_level if available, otherwise fall back to region-based classification
        if income_level:
            if income_level in ['HIC']:
                hic_biobanks.append(entry)
            else:  # UMIC, LMIC, LIC
                lmic_biobanks.append(entry)
        elif is_gs or region in LMIC_REGIONS:
            lmic_biobanks.append(entry)
        else:
            hic_biobanks.append(entry)
    
    # Aggregate statistics
    hic_pubs = sum(b['publications'] for b in hic_biobanks)
    lmic_pubs = sum(b['publications'] for b in lmic_biobanks)
    total_pubs = hic_pubs + lmic_pubs
    
    hic_avg_eas = np.mean([b['eas'] for b in hic_biobanks]) if hic_biobanks else 0
    lmic_avg_eas = np.mean([b['eas'] for b in lmic_biobanks]) if lmic_biobanks else 0
    
    # Global South priority diseases coverage
    gs_diseases = []
    for did, ddata in diseases.items():
        if ddata.get('global_south_priority', False):
            gs_diseases.append({
                'id': did,
                'name': ddata.get('name', did),
                'gapScore': ddata.get('gap_score', 0),
                'severity': ddata.get('gap_severity', 'Unknown'),
                'publications': ddata.get('publications', 0),
                'dalys': ddata.get('dalys_millions', 0)
            })
    
    gs_diseases.sort(key=lambda x: x['gapScore'], reverse=True)
    
    # Regional breakdown
    region_stats = defaultdict(lambda: {'biobanks': 0, 'publications': 0, 'avgEas': []})
    
    for bid, bdata in biobanks.items():
        region = bdata.get('region', 'Unknown')
        region_stats[region]['biobanks'] += 1
        region_stats[region]['publications'] += bdata.get('total_publications', 0)
        region_stats[region]['avgEas'].append(bdata.get('equity_alignment_score', 0))
    
    region_data = []
    for region, stats in region_stats.items():
        avg_eas = np.mean(stats['avgEas']) if stats['avgEas'] else 0
        region_data.append({
            'region': region,
            'name': WHO_REGIONS.get(region, region),
            'incomeCategory': 'LMIC' if region in LMIC_REGIONS else ('Mixed' if region in MIXED_REGIONS else 'HIC'),
            'biobanks': stats['biobanks'],
            'publications': stats['publications'],
            'averageEas': round(avg_eas, 1)
        })
    
    region_data.sort(key=lambda x: x['publications'], reverse=True)
    
    return {
        'version': VERSION,
        'generated': datetime.now().isoformat(),
        'equityRatio': global_data.get('equity_ratio', 0),
        'equityInterpretation': 'HIC-biased' if global_data.get('equity_ratio', 1) > 1.5 else 'Balanced',
        'summary': {
            'hic': {
                'biobanks': len(hic_biobanks),
                'publications': hic_pubs,
                'publicationShare': round(hic_pubs / total_pubs * 100, 1) if total_pubs > 0 else 0,
                'averageEas': round(hic_avg_eas, 1)
            },
            'lmic': {
                'biobanks': len(lmic_biobanks),
                'publications': lmic_pubs,
                'publicationShare': round(lmic_pubs / total_pubs * 100, 1) if total_pubs > 0 else 0,
                'averageEas': round(lmic_avg_eas, 1)
            }
        },
        'globalSouthDiseases': gs_diseases,
        'byRegion': region_data,
        'hicBiobanks': sorted(hic_biobanks, key=lambda x: x['publications'], reverse=True),
        'lmicBiobanks': sorted(lmic_biobanks, key=lambda x: x['publications'], reverse=True)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-Biobank v2.0 (IHCC): Generate Dashboard JSON Files")
    print("=" * 70)
    
    # Check input files
    if not INPUT_METRICS.exists():
        print(f"\n❌ Input file not found: {INPUT_METRICS}")
        print(f"   Run 03-03-bhem-compute-metrics.py first")
        return
    
    # Load data
    print(f"\n📂 Loading data...")
    
    metrics = load_json(INPUT_METRICS)
    if not metrics:
        print(f"❌ Failed to load metrics")
        return
    print(f"   Metrics loaded: {len(metrics.get('biobanks', {}))} biobanks, "
          f"{len(metrics.get('diseases', {}))} diseases")
    
    pubs_df = load_csv(INPUT_PUBLICATIONS)
    if pubs_df is not None:
        print(f"   Publications loaded: {len(pubs_df):,} records")
    else:
        print(f"   Publications file not found (trends will use fallback data)")
    
    themes_data = load_json(INPUT_THEMES)
    if themes_data:
        print(f"   Themes data loaded")
    else:
        print(f"   Themes file not found (will generate from disease categories)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # Generate all JSON files
    print(f"\n🔧 Generating JSON files...")
    
    # 1. Summary
    summary = generate_summary_json(metrics, pubs_df)
    save_json(summary, OUTPUT_DIR / "summary.json")
    
    # 2. Biobanks
    biobanks = generate_biobanks_json(metrics, pubs_df)
    save_json(biobanks, OUTPUT_DIR / "biobanks.json")
    
    # 3. Diseases
    diseases = generate_diseases_json(metrics)
    save_json(diseases, OUTPUT_DIR / "diseases.json")
    
    # 4. Matrix
    matrix = generate_matrix_json(metrics)
    save_json(matrix, OUTPUT_DIR / "matrix.json")
    
    # 5. Trends
    trends = generate_trends_json(pubs_df, metrics)
    save_json(trends, OUTPUT_DIR / "trends.json")
    
    # 6. Themes
    themes = generate_themes_json(themes_data, metrics)
    save_json(themes, OUTPUT_DIR / "themes.json")
    
    # 7. Comparison
    comparison = generate_comparison_json(metrics)
    save_json(comparison, OUTPUT_DIR / "comparison.json")
    
    # 8. Equity
    equity = generate_equity_json(metrics)
    save_json(equity, OUTPUT_DIR / "equity.json")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ JSON GENERATION COMPLETE")
    print(f"=" * 70)
    
    print(f"\n📁 Output Files:")
    for json_file in OUTPUT_DIR.glob("*.json"):
        size = json_file.stat().st_size
        print(f"   {json_file.name}: {size:,} bytes")
    
    print(f"\n➡️  Next step: python 03-04-bhem-build-site.py")


if __name__ == "__main__":
    main()