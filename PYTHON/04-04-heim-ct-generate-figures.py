#!/usr/bin/env python3
"""
04-04-heim-ct-generate-figures.py
=================================
Generate Publication-Quality Figures for HEIM-CT Clinical Trials Equity Analysis.

Produces 8 figures for Lancet submission:
  Fig CT1: HEIM-CT Framework Overview
  Fig CT2: Research Intensity Disparity (trials per million DALYs)
  Fig CT3: Global South Priority Diseases - Trial Coverage
  Fig CT4: Geographic Concentration (HIC vs LMIC)
  Fig CT5: Temporal Trends (trial volume and GS priority share)
  Fig CT6: Double Jeopardy (burden vs research attention)
  Fig CT7: Disease Category Coverage
  Fig CT8: Top 15 High-Burden Diseases by Trial Coverage

VERSION: HEIM-CT v2.0 (Publication Quality)
DATE: 2026-01-13
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VERSION = "HEIM-CT v2.0"
VERSION_DATE = "2026-01-13"

BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
OUTPUT_DIR = BASE_DIR / "ANALYSIS" / "04-04-HEIM-CT-FIGURES"

# =============================================================================
# PUBLICATION-QUALITY STYLE (Lancet Standard)
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# COLOR SCHEMES (Consistent with HEIM-Biobank)
# =============================================================================

# Gap severity colors
SEVERITY_COLORS = {
    'Critical': '#dc3545',
    'High': '#fd7e14',
    'Moderate': '#ffc107',
    'Low': '#28a745'
}

# Geographic colors
COLOR_HIC = '#2166ac'       # Blue - High-Income Countries
COLOR_LMIC = '#b2182b'      # Red - Low/Middle-Income Countries

# Disease type colors
COLOR_GS_PRIORITY = '#d73027'   # Red - Global South Priority
COLOR_NON_GS = '#4575b4'        # Blue - Other diseases
COLOR_CANCER = '#762a83'        # Purple - Cancer
COLOR_NCD = '#1b7837'           # Green - Non-communicable
COLOR_INFECTIOUS = '#e66101'    # Orange - Infectious

# Intensity colors
COLOR_HIGH_INTENSITY = '#28a745'    # Green
COLOR_MODERATE_INTENSITY = '#17a2b8' # Teal
COLOR_LOW_INTENSITY = '#dc3545'     # Red

# GBD Level 2 category colors
CATEGORY_COLORS = {
    'Neoplasms': '#762a83',
    'Cardiovascular diseases': '#e31a1c',
    'Mental disorders': '#1f78b4',
    'Neurological disorders': '#33a02c',
    'Respiratory infections': '#ff7f00',
    'NTDs and Malaria': '#6a3d9a',
    'Chronic respiratory diseases': '#b15928',
    'Digestive diseases': '#a6cee3',
    'Musculoskeletal disorders': '#fb9a99',
    'HIV/AIDS and STIs': '#fdbf6f',
    'Diabetes and kidney diseases': '#cab2d6',
    'Other': '#999999'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_intensity_category(trials_per_m_dalys: float) -> str:
    """Categorize research intensity."""
    if trials_per_m_dalys >= 1000:
        return 'High'
    elif trials_per_m_dalys >= 100:
        return 'Moderate'
    else:
        return 'Low'


def get_intensity_color(trials_per_m_dalys: float) -> str:
    """Get color based on research intensity."""
    if trials_per_m_dalys >= 1000:
        return COLOR_HIGH_INTENSITY
    elif trials_per_m_dalys >= 100:
        return COLOR_MODERATE_INTENSITY
    else:
        return COLOR_LOW_INTENSITY


def wrap_label(text: str, max_len: int = 30) -> str:
    """Wrap long labels for better display."""
    if len(text) <= max_len:
        return text

    # Try to break at natural points
    words = text.split()
    lines = []
    current = []

    for word in words:
        if len(' '.join(current + [word])) <= max_len:
            current.append(word)
        else:
            if current:
                lines.append(' '.join(current))
            current = [word]

    if current:
        lines.append(' '.join(current))

    return '\n'.join(lines)


def abbreviate_disease(name: str) -> str:
    """Create sensible abbreviations for long disease names."""
    abbreviations = {
        'Tracheal, bronchus, and lung cancer': 'Lung cancer',
        'Brain and central nervous system cancer': 'CNS cancer',
        'Interstitial lung disease and pulmonary sarcoidosis': 'ILD/Sarcoidosis',
        'Chronic obstructive pulmonary disease': 'COPD',
        'Alzheimer\'s disease and other dementias': 'Dementia',
        'Sexually transmitted infections excluding HIV': 'STIs (non-HIV)',
        'Other cardiovascular and circulatory diseases': 'Other CVD',
        'Hemoglobinopathies and hemolytic anemias': 'Hemoglobinopathies',
        'Cirrhosis and other chronic liver diseases': 'Liver cirrhosis',
        'Attention-deficit/hyperactivity disorder': 'ADHD',
        'Atrial fibrillation and flutter': 'Atrial fibrillation',
    }
    return abbreviations.get(name, name)


def save_figure(fig, name: str, output_dir: Path):
    """Save figure in both PDF and PNG formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'{name}.{ext}')
    logger.info(f"  Saved: {name}.pdf/png")
    plt.close(fig)


def load_data() -> Dict:
    """Load all required data files."""
    data = {}

    # Disease trial matrix (main analysis file)
    matrix_file = DATA_DIR / "heim_ct_disease_trial_matrix.csv"
    if matrix_file.exists():
        df = pd.read_csv(matrix_file)
        # Calculate trials per million DALYs
        df['trials_per_m_dalys'] = np.where(
            df['dalys_millions'] > 0,
            df['trial_count'] / df['dalys_millions'],
            0
        )
        data['matrix'] = df
        logger.info(f"   Loaded disease matrix: {len(df)} diseases")

    # Studies for temporal analysis
    studies_file = DATA_DIR / "heim_ct_studies_mapped.csv"
    if studies_file.exists():
        df = pd.read_csv(studies_file)
        data['studies'] = df
        logger.info(f"   Loaded studies: {len(df):,} trials")

    # Countries for geographic analysis
    countries_file = DATA_DIR / "heim_ct_countries.csv"
    if countries_file.exists():
        df = pd.read_csv(countries_file)
        data['countries'] = df
        logger.info(f"   Loaded countries: {len(df):,} site records")

    # Biobank metrics for comparison
    biobank_file = DATA_DIR / "bhem_disease_metrics.csv"
    if biobank_file.exists():
        df = pd.read_csv(biobank_file)
        data['biobank'] = df
        logger.info(f"   Loaded biobank data: {len(df)} diseases")

    return data


# =============================================================================
# FIGURE CT1: HEIM-CT FRAMEWORK
# =============================================================================

def fig_ct1_framework(output_dir: Path) -> None:
    """Figure CT1: HEIM-CT Framework Overview."""
    logger.info("Generating Figure CT1: HEIM-CT Framework...")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(7, 6.5, 'HEIM Clinical Trials Extension: Research-to-Translation Gap',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Main concept box
    concept_box = mpatches.FancyBboxPatch(
        (0.5, 4.2), 13, 1.8,
        boxstyle="round,pad=0.1",
        facecolor='#e8f4f8', edgecolor='#2c3e50', linewidth=2
    )
    ax.add_patch(concept_box)

    ax.text(7, 5.4, 'Clinical Trial Equity Index (CTEI)',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(7, 4.7,
            'CTEI  =  f( Trial Volume, Disease Burden Alignment, Geographic Distribution )',
            ha='center', va='center', fontsize=9, family='monospace')

    # Three pillars
    pillars = [
        ('Trial Volume\nAnalysis', '#3498db',
         '500K+ trials\n2000-2025\nClinicalTrials.gov', 2.0),
        ('Burden-Research\nMismatch', '#e74c3c',
         'GBD 2021 DALYs\nvs Trial counts\n89 disease categories', 7.0),
        ('Geographic\nConcentration', '#27ae60',
         'HIC vs LMIC sites\nGlobal South\npriority diseases', 12.0)
    ]

    for name, color, desc, x in pillars:
        box = mpatches.FancyBboxPatch(
            (x - 1.5, 1.0), 3, 2.5,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='#2c3e50', alpha=0.25, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, 3.1, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#2c3e50')
        ax.text(x, 1.9, desc, ha='center', va='center',
                fontsize=8, color='#2c3e50')
        ax.annotate('', xy=(x, 3.5), xytext=(x, 4.2),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # Key findings summary
    ax.text(7, 0.4,
            'Key Finding: 30 Global South Priority diseases receive disproportionately fewer trials relative to disease burden',
            ha='center', va='center', fontsize=9, style='italic', color='#555555')

    save_figure(fig, '04-04-01_heim_ct_framework', output_dir)


# =============================================================================
# FIGURE CT2: RESEARCH INTENSITY DISPARITY
# =============================================================================

def fig_ct2_research_intensity(data: Dict, output_dir: Path) -> None:
    """Figure CT2: Research intensity disparity by disease."""
    logger.info("Generating Figure CT2: Research Intensity Disparity...")

    df = data['matrix'].copy()
    df = df[df['trial_count'] > 0].copy()
    df['display_name'] = df['gbd_cause'].apply(abbreviate_disease)

    # Top 12 highest intensity (mostly cancers)
    top_12 = df.nlargest(12, 'trials_per_m_dalys')

    # Bottom 12 with significant burden (>5M DALYs)
    bottom_12 = df[df['dalys_millions'] > 5].nsmallest(12, 'trials_per_m_dalys')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: Highest research intensity
    colors1 = [COLOR_CANCER if 'cancer' in d.lower() or d in ['Leukemia', 'Melanoma', 'Multiple myeloma']
               else COLOR_NCD for d in top_12['gbd_cause']]

    y_pos1 = range(len(top_12))
    bars1 = ax1.barh(y_pos1, top_12['trials_per_m_dalys'], color=colors1, edgecolor='white', height=0.7)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(top_12['display_name'], fontsize=9)
    ax1.set_xlabel('Clinical Trials per Million DALYs')
    ax1.set_title('(A) Highest Research Intensity', fontweight='bold', pad=10)
    ax1.invert_yaxis()

    # Add trial count annotations
    for i, (_, row) in enumerate(top_12.iterrows()):
        ax1.text(row['trials_per_m_dalys'] + 500, i,
                f'{row["trial_count"]:,} trials',
                va='center', fontsize=8, color='#555555')

    # Panel B: Lowest research intensity (high burden)
    colors2 = [COLOR_GS_PRIORITY if gs else COLOR_NON_GS
               for gs in bottom_12['global_south_priority']]

    y_pos2 = range(len(bottom_12))
    bars2 = ax2.barh(y_pos2, bottom_12['trials_per_m_dalys'], color=colors2, edgecolor='white', height=0.7)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(bottom_12['display_name'], fontsize=9)
    ax2.set_xlabel('Clinical Trials per Million DALYs')
    ax2.set_title('(B) Lowest Research Intensity\n(High-Burden Diseases, >5M DALYs)', fontweight='bold', pad=10)
    ax2.invert_yaxis()

    # Add DALY annotations
    max_intensity = bottom_12['trials_per_m_dalys'].max()
    for i, (_, row) in enumerate(bottom_12.iterrows()):
        ax2.text(row['trials_per_m_dalys'] + max_intensity * 0.05, i,
                f'{row["dalys_millions"]:.0f}M DALYs',
                va='center', fontsize=8, color='#555555')

    # Legends
    cancer_patch = mpatches.Patch(color=COLOR_CANCER, label='Cancer')
    ncd_patch = mpatches.Patch(color=COLOR_NCD, label='Other NCD')
    ax1.legend(handles=[cancer_patch, ncd_patch], loc='lower right', framealpha=0.95)

    gs_patch = mpatches.Patch(color=COLOR_GS_PRIORITY, label='Global South Priority')
    other_patch = mpatches.Patch(color=COLOR_NON_GS, label='Other Diseases')
    ax2.legend(handles=[gs_patch, other_patch], loc='lower right', framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, '04-04-02_research_intensity_disparity', output_dir)


# =============================================================================
# FIGURE CT3: GLOBAL SOUTH PRIORITY DISEASES
# =============================================================================

def fig_ct3_global_south_diseases(data: Dict, output_dir: Path) -> None:
    """Figure CT3: Global South priority diseases - trials vs burden."""
    logger.info("Generating Figure CT3: Global South Priority Diseases...")

    df = data['matrix'].copy()
    gs_diseases = df[df['global_south_priority'] == True].copy()
    gs_diseases['display_name'] = gs_diseases['gbd_cause'].apply(abbreviate_disease)
    gs_diseases = gs_diseases.sort_values('dalys_millions', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Panel A: Trial counts (log scale for visibility)
    y_pos = range(len(gs_diseases))

    # Color by trial count (darker = fewer trials = more neglected)
    colors = []
    for tc in gs_diseases['trial_count']:
        if tc < 500:
            colors.append('#67001f')  # Dark red - severely neglected
        elif tc < 2000:
            colors.append('#d6604d')  # Medium red
        elif tc < 10000:
            colors.append('#f4a582')  # Light red
        else:
            colors.append('#92c5de')  # Blue - well represented

    bars1 = ax1.barh(y_pos, gs_diseases['trial_count'], color=colors, edgecolor='white', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(gs_diseases['display_name'], fontsize=9)
    ax1.set_xlabel('Number of Clinical Trials (2000-2025)')
    ax1.set_title('(A) Clinical Trial Coverage\n30 Global South Priority Diseases', fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.set_xscale('log')
    ax1.set_xlim(10, 200000)

    # Add threshold lines
    ax1.axvline(x=500, color='#67001f', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axvline(x=2000, color='#d6604d', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=10000, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    ax1.text(500, len(gs_diseases) + 0.5, '<500\n(Severely neglected)', fontsize=7,
             ha='center', color='#67001f', fontweight='bold')

    # Panel B: Disease burden (DALYs)
    bars2 = ax2.barh(y_pos, gs_diseases['dalys_millions'], color='#4575b4', edgecolor='white', height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([''] * len(gs_diseases))  # Hide labels (shown in panel A)
    ax2.set_xlabel('Disease Burden (Million DALYs, GBD 2021)')
    ax2.set_title('(B) Global Disease Burden', fontweight='bold', pad=10)
    ax2.invert_yaxis()

    # Add DALY values
    for i, (_, row) in enumerate(gs_diseases.iterrows()):
        ax2.text(row['dalys_millions'] + 2, i,
                f'{row["dalys_millions"]:.1f}M',
                va='center', fontsize=8, color='#555555')

    # Legend for Panel A
    legend_elements = [
        mpatches.Patch(color='#67001f', label='<500 trials (Severely neglected)'),
        mpatches.Patch(color='#d6604d', label='500-2,000 trials'),
        mpatches.Patch(color='#f4a582', label='2,000-10,000 trials'),
        mpatches.Patch(color='#92c5de', label='>10,000 trials')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    save_figure(fig, '04-04-03_global_south_priority_diseases', output_dir)


# =============================================================================
# FIGURE CT4: GEOGRAPHIC CONCENTRATION
# =============================================================================

def fig_ct4_geographic_concentration(data: Dict, output_dir: Path) -> None:
    """Figure CT4: Geographic concentration of clinical trials."""
    logger.info("Generating Figure CT4: Geographic Concentration...")

    if 'countries' not in data:
        logger.warning("  Skipped: No countries data")
        return

    df = data['countries'].copy()

    # World Bank High-Income Countries
    HIC_COUNTRIES = {
        'United States', 'Canada', 'Germany', 'France', 'United Kingdom',
        'Italy', 'Spain', 'Japan', 'Australia', 'Netherlands', 'Belgium',
        'Switzerland', 'Austria', 'Sweden', 'Denmark', 'Norway', 'Finland',
        'Ireland', 'Israel', 'South Korea', 'Singapore', 'New Zealand',
        'Portugal', 'Greece', 'Czech Republic', 'Poland', 'Hungary',
        'Slovenia', 'Estonia', 'Latvia', 'Lithuania', 'Iceland', 'Luxembourg'
    }

    # Count trials per country
    country_counts = df['name'].value_counts().head(20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.5, 1]})

    # Panel A: Top 20 countries
    colors = [COLOR_HIC if c in HIC_COUNTRIES else COLOR_LMIC for c in country_counts.index]
    y_pos = range(len(country_counts))

    bars = ax1.barh(y_pos, country_counts.values, color=colors, edgecolor='white', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(country_counts.index, fontsize=9)
    ax1.set_xlabel('Number of Clinical Trial Sites')
    ax1.set_title('(A) Top 20 Countries by Trial Sites', fontweight='bold', pad=10)
    ax1.invert_yaxis()

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, country_counts.values)):
        ax1.text(count + 1000, i, f'{count:,}', va='center', fontsize=8, color='#555555')

    # Legend for Panel A
    hic_patch = mpatches.Patch(color=COLOR_HIC, label='High-Income (HIC)')
    lmic_patch = mpatches.Patch(color=COLOR_LMIC, label='Low/Middle-Income (LMIC)')
    ax1.legend(handles=[hic_patch, lmic_patch], loc='lower right', framealpha=0.95)

    # Panel B: HIC vs LMIC distribution
    df['income'] = df['name'].apply(lambda x: 'HIC' if x in HIC_COUNTRIES else 'LMIC')
    income_counts = df['income'].value_counts()

    hic_count = income_counts.get('HIC', 0)
    lmic_count = income_counts.get('LMIC', 0)
    ratio = hic_count / lmic_count if lmic_count > 0 else 0

    # Donut chart
    sizes = [hic_count, lmic_count]
    colors_pie = [COLOR_HIC, COLOR_LMIC]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=None, colors=colors_pie,
        autopct='%1.1f%%', startangle=90,
        explode=(0, 0.03), pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white')
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax2.set_title(f'(B) Geographic Distribution\nHIC:LMIC Ratio = {ratio:.1f}:1',
                  fontweight='bold', pad=10)

    # Add legend with counts
    legend_labels = [f'HIC ({hic_count:,} sites)', f'LMIC ({lmic_count:,} sites)']
    ax2.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, '04-04-04_geographic_concentration', output_dir)


# =============================================================================
# FIGURE CT5: TEMPORAL TRENDS
# =============================================================================

def fig_ct5_temporal_trends(data: Dict, output_dir: Path) -> None:
    """Figure CT5: Temporal trends in clinical trials."""
    logger.info("Generating Figure CT5: Temporal Trends...")

    if 'studies' not in data:
        logger.warning("  Skipped: No studies data")
        return

    df = data['studies'].copy()

    # Filter to valid years
    if 'start_year' in df.columns:
        df = df[df['start_year'].notna() & (df['start_year'] >= 2000) & (df['start_year'] <= 2025)]
    else:
        logger.warning("  Skipped: No start_year column")
        return

    # Aggregate by year
    yearly = df.groupby('start_year').agg({
        'nct_id': 'count',
        'global_south_priority': 'sum'
    }).reset_index()
    yearly.columns = ['year', 'total_trials', 'gs_priority_trials']
    yearly['gs_share'] = 100 * yearly['gs_priority_trials'] / yearly['total_trials']
    yearly['non_gs_trials'] = yearly['total_trials'] - yearly['gs_priority_trials']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: Stacked bar - total trials over time
    ax1.bar(yearly['year'], yearly['gs_priority_trials'],
            color=COLOR_GS_PRIORITY, alpha=0.9, label='Global South Priority', edgecolor='white')
    ax1.bar(yearly['year'], yearly['non_gs_trials'],
            bottom=yearly['gs_priority_trials'],
            color=COLOR_NON_GS, alpha=0.9, label='Other Diseases', edgecolor='white')

    ax1.set_ylabel('Number of Trials Started')
    ax1.set_title('(A) Clinical Trials Over Time (2000-2025)\nStacked by Disease Priority',
                  fontweight='bold', pad=10)

    # COVID-19 marker
    ax1.axvline(x=2020, color='#333333', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(2020.2, yearly['total_trials'].max() * 0.95, 'COVID-19',
             fontsize=9, color='#333333', fontweight='bold', va='top')

    ax1.legend(loc='upper left', framealpha=0.95)

    # Add total annotation
    total_trials = yearly['total_trials'].sum()
    ax1.text(0.98, 0.95, f'Total: {total_trials:,} trials',
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=10, fontweight='bold', color='#333333')

    # Panel B: Global South priority share trend
    ax2.plot(yearly['year'], yearly['gs_share'], 'o-',
             color=COLOR_GS_PRIORITY, linewidth=2.5, markersize=6)
    ax2.fill_between(yearly['year'], yearly['gs_share'],
                     alpha=0.2, color=COLOR_GS_PRIORITY)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('% of Trials on GS Priority Diseases')
    ax2.set_title('(B) Share of Trials Addressing Global South Priority Diseases',
                  fontweight='bold', pad=10)

    # Add average line
    avg_share = yearly['gs_share'].mean()
    ax2.axhline(y=avg_share, color='#555555', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.text(2024, avg_share + 1, f'Average: {avg_share:.1f}%',
             fontsize=9, color='#555555', ha='right')

    ax2.set_ylim(0, 55)
    ax2.set_xlim(1999, 2026)

    # Trend annotation
    early_avg = yearly[yearly['year'] <= 2005]['gs_share'].mean()
    recent_avg = yearly[yearly['year'] >= 2020]['gs_share'].mean()
    change = recent_avg - early_avg

    ax2.annotate(f'Trend: {change:+.1f} percentage points\n(2000-05 vs 2020-25)',
                 xy=(2015, 35), fontsize=9, color='#555555',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))

    plt.tight_layout()
    save_figure(fig, '04-04-05_temporal_trends', output_dir)


# =============================================================================
# FIGURE CT6: DOUBLE JEOPARDY (BURDEN VS TRIALS)
# =============================================================================

def fig_ct6_double_jeopardy(data: Dict, output_dir: Path) -> None:
    """Figure CT6: Double jeopardy - high burden + few trials."""
    logger.info("Generating Figure CT6: Double Jeopardy...")

    df = data['matrix'].copy()
    df['display_name'] = df['gbd_cause'].apply(abbreviate_disease)
    df = df[df['dalys_millions'] > 0]  # Only diseases with burden data

    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot with different markers
    gs_mask = df['global_south_priority'] == True

    # Non-GS diseases (circles)
    ax.scatter(df[~gs_mask]['trial_count'], df[~gs_mask]['dalys_millions'],
               s=80, alpha=0.6, c=COLOR_NON_GS, label='Other Diseases',
               edgecolors='white', linewidth=0.5)

    # GS Priority diseases (squares, larger)
    ax.scatter(df[gs_mask]['trial_count'], df[gs_mask]['dalys_millions'],
               s=120, alpha=0.8, c=COLOR_GS_PRIORITY, marker='s',
               label='Global South Priority', edgecolors='white', linewidth=0.5)

    # Log scale for trials
    ax.set_xscale('log')
    ax.set_xlabel('Number of Clinical Trials (log scale)')
    ax.set_ylabel('Disease Burden (Million DALYs)')
    ax.set_title('Clinical Trial Coverage vs Disease Burden\n"Double Jeopardy" Zone: High Burden + Few Trials',
                 fontweight='bold', pad=15)

    # Quadrant lines
    ax.axhline(y=50, color='#888888', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=5000, color='#888888', linestyle='--', alpha=0.4, linewidth=1)

    # Quadrant labels
    ax.text(150, 200, 'DOUBLE JEOPARDY\nHigh Burden, Few Trials',
            fontsize=11, ha='center', va='center', color='#b2182b',
            fontweight='bold', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fee0d2', edgecolor='none', alpha=0.7))

    ax.text(40000, 200, 'APPROPRIATE ATTENTION\nHigh Burden, Many Trials',
            fontsize=11, ha='center', va='center', color='#2166ac',
            fontweight='bold', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d1e5f0', edgecolor='none', alpha=0.7))

    # Label key diseases (high burden or GS priority with significant burden)
    label_threshold_dalys = 40  # Only label diseases with >40M DALYs
    label_threshold_gs = 20    # Label GS priority with >20M DALYs

    labeled_diseases = df[
        (df['dalys_millions'] > label_threshold_dalys) |
        ((df['global_south_priority']) & (df['dalys_millions'] > label_threshold_gs))
    ]

    # Smart label positioning
    for _, row in labeled_diseases.iterrows():
        x, y = row['trial_count'], row['dalys_millions']
        label = row['display_name'][:20]

        # Determine offset based on position
        if x < 5000:
            offset_x, ha = 10, 'left'
        else:
            offset_x, ha = -10, 'right'

        offset_y = 3 if y < 150 else -5

        ax.annotate(label, xy=(x, y), xytext=(offset_x, offset_y),
                    textcoords='offset points', fontsize=7.5,
                    ha=ha, va='center', color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor='#cccccc', alpha=0.8))

    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(20, 200000)
    ax.set_ylim(0, 230)

    plt.tight_layout()
    save_figure(fig, '04-04-06_double_jeopardy', output_dir)


# =============================================================================
# FIGURE CT7: DISEASE CATEGORY COVERAGE
# =============================================================================

def fig_ct7_disease_categories(data: Dict, output_dir: Path) -> None:
    """Figure CT7: Disease category coverage analysis."""
    logger.info("Generating Figure CT7: Disease Category Coverage...")

    df = data['matrix'].copy()

    # Aggregate by GBD Level 2 category
    cat_data = df.groupby('gbd_level2').agg({
        'trial_count': 'sum',
        'dalys_millions': 'sum',
        'global_south_priority': 'sum',
        'gbd_cause': 'count'
    }).reset_index()
    cat_data.columns = ['category', 'trials', 'dalys', 'gs_diseases', 'n_diseases']
    cat_data['trials_per_m_dalys'] = cat_data['trials'] / cat_data['dalys'].replace(0, np.nan)
    cat_data = cat_data.sort_values('trials', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: Trials by category
    y_pos = range(len(cat_data))
    colors = [CATEGORY_COLORS.get(c, CATEGORY_COLORS['Other']) for c in cat_data['category']]

    bars1 = ax1.barh(y_pos, cat_data['trials'], color=colors, edgecolor='white', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cat_data['category'], fontsize=9)
    ax1.set_xlabel('Number of Clinical Trials')
    ax1.set_title('(A) Clinical Trials by Disease Category', fontweight='bold', pad=10)
    ax1.invert_yaxis()

    # Add trial count labels
    for i, (_, row) in enumerate(cat_data.iterrows()):
        ax1.text(row['trials'] + 5000, i, f'{row["trials"]:,}',
                va='center', fontsize=8, color='#555555')

    # Panel B: Research intensity by category
    cat_data_sorted = cat_data.sort_values('trials_per_m_dalys', ascending=True)
    y_pos2 = range(len(cat_data_sorted))

    # Color by intensity
    intensity_colors = [get_intensity_color(i) for i in cat_data_sorted['trials_per_m_dalys']]

    bars2 = ax2.barh(y_pos2, cat_data_sorted['trials_per_m_dalys'],
                     color=intensity_colors, edgecolor='white', height=0.7)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(cat_data_sorted['category'], fontsize=9)
    ax2.set_xlabel('Trials per Million DALYs')
    ax2.set_title('(B) Research Intensity by Category', fontweight='bold', pad=10)

    # Add threshold lines
    ax2.axvline(x=100, color='#555555', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=1000, color='#555555', linestyle='--', alpha=0.5, linewidth=1)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_HIGH_INTENSITY, label='High (>1000)'),
        mpatches.Patch(color=COLOR_MODERATE_INTENSITY, label='Moderate (100-1000)'),
        mpatches.Patch(color=COLOR_LOW_INTENSITY, label='Low (<100)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', title='Intensity', framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, '04-04-07_disease_category_coverage', output_dir)


# =============================================================================
# FIGURE CT8: TOP HIGH-BURDEN DISEASES
# =============================================================================

def fig_ct8_top_burden_diseases(data: Dict, output_dir: Path) -> None:
    """Figure CT8: Top 15 high-burden diseases by trial coverage."""
    logger.info("Generating Figure CT8: Top High-Burden Diseases...")

    df = data['matrix'].copy()
    df['display_name'] = df['gbd_cause'].apply(abbreviate_disease)

    # Top 15 by disease burden
    top_burden = df.nlargest(15, 'dalys_millions').copy()
    top_burden = top_burden.sort_values('dalys_millions', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 9))

    y_pos = range(len(top_burden))

    # Colors by Global South priority
    colors = [COLOR_GS_PRIORITY if gs else COLOR_NON_GS
              for gs in top_burden['global_south_priority']]

    # Horizontal bars for trials
    bars = ax.barh(y_pos, top_burden['trial_count'], color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{name} ({dalys:.0f}M DALYs)"
                        for name, dalys in zip(top_burden['display_name'], top_burden['dalys_millions'])],
                       fontsize=9)
    ax.set_xlabel('Number of Clinical Trials (2000-2025)')
    ax.set_title('Top 15 Diseases by Global Burden:\nClinical Trial Coverage',
                 fontweight='bold', fontsize=13, pad=15)

    # Add trial intensity annotation
    for i, (_, row) in enumerate(top_burden.iterrows()):
        intensity = row['trial_count'] / row['dalys_millions'] if row['dalys_millions'] > 0 else 0
        intensity_label = f'{intensity:.0f} trials/M DALY'

        # Position based on bar width
        ax.text(row['trial_count'] + 500, i, intensity_label,
                va='center', fontsize=8, color='#555555')

    # Add reference line for "adequate" coverage
    ax.axvline(x=10000, color='#888888', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(10000, len(top_burden) - 0.5, '10,000 trials', fontsize=8,
            color='#888888', ha='center', va='bottom')

    # Legend
    gs_patch = mpatches.Patch(color=COLOR_GS_PRIORITY, label='Global South Priority')
    other_patch = mpatches.Patch(color=COLOR_NON_GS, label='Other Diseases')
    ax.legend(handles=[gs_patch, other_patch], loc='lower right', framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, '04-04-08_top_burden_diseases', output_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-CT: Generate Publication-Quality Figures")
    print(f"Version: {VERSION} ({VERSION_DATE})")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {OUTPUT_DIR}")

    print(f"\n📂 Loading data...")
    data = load_data()

    if 'matrix' not in data:
        print("❌ Error: Disease matrix not found")
        return

    print(f"\n🎨 Generating figures (8 total)...")

    # Generate all figures
    fig_ct1_framework(OUTPUT_DIR)
    fig_ct2_research_intensity(data, OUTPUT_DIR)
    fig_ct3_global_south_diseases(data, OUTPUT_DIR)
    fig_ct4_geographic_concentration(data, OUTPUT_DIR)
    fig_ct5_temporal_trends(data, OUTPUT_DIR)
    fig_ct6_double_jeopardy(data, OUTPUT_DIR)
    fig_ct7_disease_categories(data, OUTPUT_DIR)
    fig_ct8_top_burden_diseases(data, OUTPUT_DIR)

    print(f"\n✅ COMPLETE! Generated 8 publication-quality figures.")
    print(f"   Format: PDF + PNG (300 DPI)")
    print(f"   Style: Times New Roman, Lancet standard")

    # Summary statistics
    df = data['matrix']
    total_trials = df['trial_count'].sum()
    gs_trials = df[df['global_south_priority']]['trial_count'].sum()
    gs_share = 100 * gs_trials / total_trials

    print(f"\n📊 Key Statistics:")
    print(f"   Total trials mapped: {total_trials:,}")
    print(f"   Global South Priority: {gs_trials:,} ({gs_share:.1f}%)")
    print(f"   Diseases analyzed: {len(df)}")


if __name__ == "__main__":
    main()
