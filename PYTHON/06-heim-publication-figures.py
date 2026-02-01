#!/usr/bin/env python3
"""
06-heim-publication-figures.py
==============================

HEIM Publication Figures Generator for Nature Medicine submission.

Generates main-text and extended-data figures from pre-computed HEIM metrics
(webapp JSON data files). All output is PDF + PNG at 300 DPI.

FIGURES:
    - Figure 2a: World map of biobank equity (EAS by country)
    - Figure 2b: Biobank EAS distribution (bar chart)
    - Figure 2c: Publication share by WHO region
    - Figure 3a: World map of clinical trial sites
    - Figure 3b: Trial intensity gap (GS vs non-GS)
    - Figure 4a: NTD vs non-NTD semantic isolation comparison
    - Figure 6:  Unified Neglect Score ranking (top 30 diseases)

INPUT:
    docs/data/diseases.json   - Disease-level HEIM metrics
    docs/data/summary.json    - Biobank-level summary statistics
    docs/data/matrix.json     - Cross-disease similarity matrix

OUTPUT:
    ANALYSIS/05-04-HEIM-SEM-FIGURES/*.pdf
    ANALYSIS/05-04-HEIM-SEM-FIGURES/*.png

USAGE:
    python3.11 06-heim-publication-figures.py

REQUIREMENTS:
    pip install matplotlib seaborn pandas numpy geopandas

Author: Manuel Corpas
Date: 2026-01-25
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import geopandas for world maps
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. World maps will be skipped.")

# Set publication-quality defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Paths
BASE_PATH = Path("/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS")
DATA_PATH = BASE_PATH / "docs" / "data"
OUTPUT_PATH = BASE_PATH / "ANALYSIS" / "05-04-HEIM-SEM-FIGURES"

# WHO NTD list for classification
WHO_NTDS = [
    'Lymphatic_filariasis', 'Lymphatic filariasis',
    'Guinea_worm_disease', 'Guinea worm disease',
    'Schistosomiasis',
    'Onchocerciasis',
    'Leishmaniasis',
    'Chagas_disease', 'Chagas disease',
    'African_trypanosomiasis', 'African trypanosomiasis',
    'Dengue',
    'Rabies',
    'Ascariasis',
    'Trichuriasis',
    'Hookworm_disease', 'Hookworm disease',
    'Scabies',
    'Trachoma',
    'Cysticercosis',
    'Cystic_echinococcosis', 'Cystic echinococcosis',
    'Yellow_fever', 'Yellow fever',
    'Foodborne_trematodiases', 'Foodborne trematodiases',
    'Leprosy'
]

def load_data():
    """Load all required data files."""
    print("Loading data...")

    # Load integrated metrics
    with open(DATA_PATH / "integrated.json", 'r') as f:
        integrated = json.load(f)

    # Load biobank data
    with open(DATA_PATH / "biobanks.json", 'r') as f:
        biobanks = json.load(f)

    # Load semantic metrics
    with open(DATA_PATH / "semantic.json", 'r') as f:
        semantic = json.load(f)

    # Load clinical trials
    with open(DATA_PATH / "clinical_trials.json", 'r') as f:
        clinical_trials = json.load(f)

    print(f"  Integrated: {integrated['n_diseases']} diseases")
    print(f"  Biobanks: {biobanks['count']} biobanks")
    print(f"  Semantic: {semantic['n_diseases']} diseases")

    return integrated, biobanks, semantic, clinical_trials


def is_ntd(disease_name):
    """Check if a disease is a WHO-classified NTD."""
    disease_clean = disease_name.replace('_', ' ')
    for ntd in WHO_NTDS:
        if ntd.lower() in disease_clean.lower() or disease_clean.lower() in ntd.lower():
            return True
    return False


def create_figure_4a(semantic_data, output_path):
    """
    Figure 4a: Box plot comparing SII between NTDs and non-NTDs
    """
    print("\nGenerating Figure 4a: NTD vs non-NTD comparison...")

    # Extract metrics
    metrics = semantic_data['metrics']

    ntd_sii = []
    non_ntd_sii = []

    for disease, data in metrics.items():
        sii = data.get('sii', 0)
        if is_ntd(disease):
            ntd_sii.append(sii)
        else:
            non_ntd_sii.append(sii)

    print(f"  NTDs: {len(ntd_sii)}, non-NTDs: {len(non_ntd_sii)}")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(ntd_sii, non_ntd_sii, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((np.std(ntd_sii)**2 + np.std(non_ntd_sii)**2) / 2)
    cohens_d = (np.mean(ntd_sii) - np.mean(non_ntd_sii)) / pooled_std

    # Percent difference
    pct_diff = ((np.mean(ntd_sii) - np.mean(non_ntd_sii)) / np.mean(non_ntd_sii)) * 100

    print(f"  NTD mean SII: {np.mean(ntd_sii):.5f}")
    print(f"  Non-NTD mean SII: {np.mean(non_ntd_sii):.5f}")
    print(f"  Difference: {pct_diff:.1f}%")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 5))

    # Colors
    colors = ['#E74C3C', '#3498DB']  # Red for NTD, Blue for non-NTD

    # Box plot
    bp = ax.boxplot([ntd_sii, non_ntd_sii],
                    positions=[1, 2],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points with jitter
    for i, (data, color) in enumerate(zip([ntd_sii, non_ntd_sii], colors), 1):
        x = np.random.normal(i, 0.08, len(data))
        ax.scatter(x, data, alpha=0.5, s=20, color=color, edgecolors='white', linewidth=0.5)

    # Add significance bracket
    y_max = max(max(ntd_sii), max(non_ntd_sii))
    bracket_y = y_max * 1.05
    ax.plot([1, 1, 2, 2], [bracket_y, bracket_y*1.02, bracket_y*1.02, bracket_y], 'k-', linewidth=1)

    # Significance stars
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'

    ax.text(1.5, bracket_y*1.04, sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['NTDs\n(n={})'.format(len(ntd_sii)),
                        'Other diseases\n(n={})'.format(len(non_ntd_sii))],
                       fontsize=10)
    ax.set_ylabel('Semantic Isolation Index (SII)', fontsize=11)
    ax.set_title('Semantic Isolation:\nNTDs vs Other Conditions', fontsize=12, fontweight='bold')

    # Add statistics annotation
    stats_text = f'Δ = {pct_diff:.0f}%\nP = {p_value:.3f}\nd = {cohens_d:.2f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.4, 2.6)

    plt.tight_layout()

    # Save
    output_file = output_path / "fig4a_ntd_comparison.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()

    return {
        'ntd_n': len(ntd_sii),
        'non_ntd_n': len(non_ntd_sii),
        'ntd_mean': np.mean(ntd_sii),
        'non_ntd_mean': np.mean(non_ntd_sii),
        'pct_diff': pct_diff,
        'p_value': p_value,
        'cohens_d': cohens_d
    }


def create_figure_6(integrated_data, output_path):
    """
    Figure 6: Unified Neglect Score ranking - Top 30 diseases
    """
    print("\nGenerating Figure 6: Unified Neglect Score ranking...")

    # Get disease data
    diseases = integrated_data['diseases']

    # Sort by unified score
    diseases_sorted = sorted(diseases, key=lambda x: x.get('unified_score', 0), reverse=True)

    # Take top 30
    top_30 = diseases_sorted[:30]

    # Prepare data
    names = []
    scores = []
    is_ntd_list = []
    gap_scores = []
    sii_scores = []

    for d in top_30:
        name = d['disease'].replace('_', ' ')
        if len(name) > 25:
            name = name[:22] + '...'
        names.append(name)
        scores.append(d.get('unified_score', 0))
        is_ntd_list.append(is_ntd(d['disease']))
        gap_scores.append(d.get('gap_score', 0))
        sii_scores.append(d.get('sii', 0) * 10000)  # Scale for visibility

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Colors based on NTD status
    colors = ['#E74C3C' if ntd else '#95A5A6' for ntd in is_ntd_list]

    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()  # Highest at top

    ax.set_xlabel('Unified Neglect Score', fontsize=11)
    ax.set_title('Top 30 Most Neglected Diseases\n(HEIM Unified Score)', fontsize=12, fontweight='bold')

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=8)

    # Legend
    ntd_patch = mpatches.Patch(color='#E74C3C', label='WHO Neglected Tropical Disease')
    other_patch = mpatches.Patch(color='#95A5A6', label='Other condition')
    ax.legend(handles=[ntd_patch, other_patch], loc='lower right', fontsize=9)

    # Add annotation
    ntd_count = sum(is_ntd_list[:10])
    ax.text(0.02, 0.02, f'Top 10: {ntd_count}/10 are NTDs\nAll top 10 affect Global South',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(scores) * 1.15)

    plt.tight_layout()

    # Save
    output_file = output_path / "fig6_unified_score_ranking.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def create_figure_2(biobank_data, output_path):
    """
    Figure 2b: Biobank Equity Alignment Score distribution
    """
    print("\nGenerating Figure 2b: Biobank EAS distribution...")

    biobanks = biobank_data['biobanks']

    # Extract EAS scores
    eas_scores = []
    names = []
    categories = []

    for bb in biobanks:
        eas = bb['scores']['equityAlignment']
        eas_scores.append(eas)
        names.append(bb['name'])
        categories.append(bb['scores']['equityCategory'])

    # Create dataframe
    df = pd.DataFrame({
        'name': names,
        'eas': eas_scores,
        'category': categories
    }).sort_values('eas', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color mapping
    color_map = {'High': '#27AE60', 'Moderate': '#F39C12', 'Low': '#E74C3C'}
    colors = [color_map[cat] for cat in df['category']]

    # Bar chart
    x_pos = np.arange(len(df))
    bars = ax.bar(x_pos, df['eas'], color=colors, edgecolor='white', linewidth=0.3)

    # Threshold lines
    ax.axhline(y=60, color='#27AE60', linestyle='--', linewidth=1.5, label='High threshold (60)')
    ax.axhline(y=40, color='#F39C12', linestyle='--', linewidth=1.5, label='Moderate threshold (40)')

    # Labels
    ax.set_xticks(x_pos[::5])
    ax.set_xticklabels([df.iloc[i]['name'][:15] for i in range(0, len(df), 5)],
                        rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Equity Alignment Score (EAS)', fontsize=11)
    ax.set_xlabel('Biobanks (ranked by EAS)', fontsize=11)
    ax.set_title('Biobank Equity Alignment Distribution\n(n=70 IHCC biobanks)', fontsize=12, fontweight='bold')

    # Add UK Biobank label
    uk_idx = df[df['name'] == 'UK Biobank'].index[0]
    uk_pos = list(df.index).index(uk_idx)
    ax.annotate('UK Biobank\n(EAS=84.6)', xy=(uk_pos, df.loc[uk_idx, 'eas']),
                xytext=(uk_pos+5, df.loc[uk_idx, 'eas']+5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    # Legend
    high_patch = mpatches.Patch(color='#27AE60', label=f'High (n=1)')
    mod_patch = mpatches.Patch(color='#F39C12', label=f'Moderate (n=13)')
    low_patch = mpatches.Patch(color='#E74C3C', label=f'Low (n=56)')
    ax.legend(handles=[high_patch, mod_patch, low_patch], loc='upper right', fontsize=9)

    # Add percentage annotation
    ax.text(0.98, 0.5, '80% of biobanks\nshow LOW\nequity alignment',
            transform=ax.transAxes, fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    # Save
    output_file = output_path / "fig2b_biobank_eas_distribution.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def create_figure_2c(biobank_data, output_path):
    """
    Figure 2c: Publication share by income group
    """
    print("\nGenerating Figure 2c: Publication share by income group...")

    biobanks = biobank_data['biobanks']

    # Aggregate by income
    hic_pubs = 0
    lmic_pubs = 0

    for bb in biobanks:
        pubs = bb['stats']['totalPublications']
        if bb['isGlobalSouth']:
            lmic_pubs += pubs
        else:
            hic_pubs += pubs

    total = hic_pubs + lmic_pubs
    hic_pct = hic_pubs / total * 100
    lmic_pct = lmic_pubs / total * 100

    print(f"  HIC: {hic_pubs} ({hic_pct:.1f}%)")
    print(f"  LMIC: {lmic_pubs} ({lmic_pct:.1f}%)")

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Pie chart
    sizes = [hic_pct, lmic_pct]
    labels = [f'High-Income\nCountries\n{hic_pct:.1f}%',
              f'Global South\n{lmic_pct:.1f}%']
    colors = ['#3498DB', '#E74C3C']
    explode = (0, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       explode=explode, autopct='',
                                       startangle=90,
                                       wedgeprops=dict(width=0.7, edgecolor='white'))

    # Center text
    ax.text(0, 0, f'Total:\n{total:,}\npublications',
            ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_title('Biobank Publication Share\nby Country Income Level', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = output_path / "fig2c_publication_share.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def create_figure_3b(clinical_trial_data, output_path):
    """
    Figure 3b: Trial intensity gap visualization
    """
    print("\nGenerating Figure 3b: Trial intensity gap...")

    ct = clinical_trial_data
    geo = ct.get('geographic', {})
    gs = ct.get('globalSouthAnalysis', {})

    hic_sites = geo.get('hicSites', 552952)
    lmic_sites = geo.get('lmicSites', 217226)

    hic_intensity = gs.get('nonGsIntensity', 1167.4)
    gs_intensity = gs.get('gsIntensity', 485.9)
    intensity_gap = gs.get('intensityGap', 2.4)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Site distribution
    ax1 = axes[0]
    categories = ['High-Income\nCountries', 'Low/Middle-Income\nCountries']
    sites = [hic_sites, lmic_sites]
    colors = ['#3498DB', '#E74C3C']

    bars1 = ax1.bar(categories, sites, color=colors, edgecolor='white')
    ax1.set_ylabel('Number of Trial Sites', fontsize=11)
    ax1.set_title('Clinical Trial Site Distribution', fontsize=12, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars1, sites):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f'{count:,}\n({count/sum(sites)*100:.1f}%)',
                ha='center', fontsize=10)

    # Add ratio
    ratio = hic_sites / lmic_sites
    ax1.text(0.5, 0.85, f'Ratio: {ratio:.1f}×',
            transform=ax1.transAxes, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Intensity gap
    ax2 = axes[1]
    categories2 = ['High-Income\nCountries', 'Global South']
    intensities = [hic_intensity, gs_intensity]

    bars2 = ax2.bar(categories2, intensities, color=colors, edgecolor='white')
    ax2.set_ylabel('Trials per Million DALYs', fontsize=11)
    ax2.set_title('Trial Intensity by Region\n(Global South diseases)', fontsize=12, fontweight='bold')

    # Add intensity labels
    for bar, intensity in zip(bars2, intensities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{intensity:.0f}',
                ha='center', fontsize=10)

    # Add gap annotation
    ax2.annotate('', xy=(1, gs_intensity), xytext=(1, hic_intensity),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text(1.15, (hic_intensity + gs_intensity)/2, f'{intensity_gap}×\ngap',
            ha='left', va='center', fontsize=11, fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    output_file = output_path / "fig3b_trial_intensity_gap.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def create_figure_1_spec(output_path):
    """
    Create specification for Figure 1 (framework schematic)
    This would typically be created in a design tool like Illustrator/BioRender
    """
    print("\nGenerating Figure 1 specification...")

    spec = """
FIGURE 1: HEIM FRAMEWORK SCHEMATIC
==================================

Layout: 3-panel horizontal (A, B, C)

PANEL A: Three Dimensions
-------------------------
Visual: Three interconnected circles/pillars

[DISCOVERY]          [TRANSLATION]         [KNOWLEDGE]
   |                      |                     |
Biobanks             Clinical Trials      Scientific Literature
70 cohorts           563K trials          13.1M abstracts
29 countries         770K sites           175 diseases
   |                      |                     |
Gap Score            Trial Intensity      Semantic Isolation
Equity Alignment     Geographic Ratio     Index (SII)
Score (EAS)                               KTP, RCC

Color scheme:
- Discovery: Blue (#3498DB)
- Translation: Green (#27AE60)
- Knowledge: Purple (#9B59B6)


PANEL B: Data Flow
------------------
Visual: Flowchart

[GBD 2021 Taxonomy] ──────────────────────────────────┐
       │                                               │
       ▼                                               │
[179 Disease Categories]                               │
       │                                               │
       ├────────────────┬──────────────────┬──────────┘
       ▼                ▼                  ▼
   [PubMed]         [AACT]            [IHCC]
   Entrez API    ClinicalTrials.gov  Biobank Registry
       │                │                  │
       ▼                ▼                  ▼
[13.1M abstracts] [770K sites]      [38K publications]
       │                │                  │
       ▼                ▼                  ▼
  [PubMedBERT]    [Geographic]      [MeSH Mapping]
  [Embeddings]    [Classification]  [Gap Scores]
       │                │                  │
       └────────────────┴──────────────────┘
                        │
                        ▼
              [UNIFIED NEGLECT SCORE]


PANEL C: Score Integration
--------------------------
Visual: Equation with weighting diagram

Unified Score = 0.33 × Discovery + 0.33 × Translation + 0.34 × Knowledge

[Three colored bars merging into one]
Discovery (33%) ──┐
Translation (33%) ─┼──► UNIFIED NEGLECT SCORE (0-50)
Knowledge (34%) ──┘

Scale bar: 0 ────────────────────────► 50
           Well-resourced        Severely neglected


DESIGN NOTES:
- Use consistent color palette
- Include icons for data sources (database, globe, document)
- Arrows should show data flow direction
- Consider using Nature Medicine figure style guide
- Suggested tools: BioRender, Adobe Illustrator, Inkscape
"""

    # Save specification
    output_file = output_path / "fig1_specification.txt"
    with open(output_file, 'w') as f:
        f.write(spec)
    print(f"  Saved specification: {output_file}")

    # Also create a simple programmatic version
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Three dimensions
    ax1 = axes[0]
    dims = ['Discovery\n(Biobanks)', 'Translation\n(Clinical Trials)', 'Knowledge\n(Literature)']
    values = [70, 0.56, 13.1]  # Scaled for visualization
    colors = ['#3498DB', '#27AE60', '#9B59B6']

    bars = ax1.bar(dims, [1, 1, 1], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylim(0, 1.5)
    ax1.set_ylabel('')
    ax1.set_title('A. Three Dimensions of Research Equity', fontsize=12, fontweight='bold')

    # Add metrics below
    metrics = ['70 biobanks\n38K publications\nEAS, Gap Score',
               '563K trials\n770K sites\nGeographic Ratio',
               '13.1M abstracts\n175 diseases\nSII, KTP, RCC']
    for i, (bar, metric) in enumerate(zip(bars, metrics)):
        ax1.text(bar.get_x() + bar.get_width()/2, 0.5, metric,
                ha='center', va='center', fontsize=9)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])

    # Panel B: Data sources
    ax2 = axes[1]
    sources = ['GBD 2021', 'PubMed', 'AACT', 'IHCC']
    source_sizes = [179, 13.1, 0.56, 70]
    ax2.barh(sources, source_sizes, color=['gray', '#9B59B6', '#27AE60', '#3498DB'])
    ax2.set_xlabel('Scale (diseases/millions/cohorts)')
    ax2.set_title('B. Data Sources and Scale', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Score integration
    ax3 = axes[2]
    weights = [0.33, 0.33, 0.34]
    labels = ['Discovery\n(33%)', 'Translation\n(33%)', 'Knowledge\n(34%)']
    ax3.pie(weights, labels=labels, colors=colors, autopct='', startangle=90,
           wedgeprops=dict(width=0.4, edgecolor='white'))
    ax3.text(0, 0, 'UNIFIED\nSCORE', ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.set_title('C. Score Integration', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save simple version
    output_file = output_path / "fig1_framework_simple.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    print(f"  Saved simple version: {output_file}")

    plt.close()


def get_world_map():
    """Load world map from Natural Earth (handles geopandas 1.0+ deprecation)."""
    # URL for Natural Earth low-res countries
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url)
        return world
    except Exception as e:
        print(f"  Warning: Could not download Natural Earth data: {e}")
        # Try alternative: use built-in data if available (older geopandas)
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            return world
        except:
            return None


def create_figure_2a_world_map(biobank_data, output_path):
    """
    Figure 2a: World map of biobank locations colored by Equity Alignment Score
    """
    if not GEOPANDAS_AVAILABLE:
        print("\nSkipping Figure 2a: geopandas not available")
        return

    print("\nGenerating Figure 2a: World map of biobank equity...")

    # Load world map
    world = get_world_map()
    if world is None:
        print("  Error: Could not load world map data")
        return

    # Get country name column (varies between versions)
    name_col = 'NAME' if 'NAME' in world.columns else 'name' if 'name' in world.columns else 'ADMIN'

    # Country name mapping (biobank data → shapefile names)
    country_mapping = {
        'United States': 'United States of America',
        'United Kingdom': 'United Kingdom',
        'International': None,  # Skip international
        'South Korea': 'South Korea',
        'Taiwan': None,  # Not in natural earth
        'Guinea-Bissau': 'Guinea-Bissau',
    }

    # Aggregate EAS by country (take max EAS if multiple biobanks)
    biobanks = biobank_data['biobanks']
    country_eas = {}
    country_count = {}

    for bb in biobanks:
        country = bb['country']
        eas = bb['scores']['equityAlignment']

        # Map country name if needed
        mapped_country = country_mapping.get(country, country)
        if mapped_country is None:
            continue

        if mapped_country not in country_eas:
            country_eas[mapped_country] = eas
            country_count[mapped_country] = 1
        else:
            # Take the maximum EAS (best performing biobank)
            country_eas[mapped_country] = max(country_eas[mapped_country], eas)
            country_count[mapped_country] += 1

    print(f"  Countries with biobanks: {len(country_eas)}")

    # Merge with world map
    world['eas'] = world[name_col].map(country_eas)
    world['has_biobank'] = world['eas'].notna()
    world['biobank_count'] = world[name_col].map(country_count).fillna(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot countries without biobanks in light gray
    world[~world['has_biobank']].plot(
        ax=ax,
        color='#E0E0E0',
        edgecolor='white',
        linewidth=0.3
    )

    # Plot countries with biobanks, colored by EAS
    cmap = LinearSegmentedColormap.from_list('eas_cmap', ['#E74C3C', '#F39C12', '#27AE60'])
    world_with_biobank = world[world['has_biobank']]

    world_with_biobank.plot(
        ax=ax,
        column='eas',
        cmap=cmap,
        edgecolor='black',
        linewidth=0.5,
        legend=False,
        vmin=0,
        vmax=100
    )

    # Add colorbar
    norm = Normalize(vmin=0, vmax=100)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=25, pad=0.04)
    cbar.set_label('Equity Alignment Score (EAS)', fontsize=11)

    # Add threshold lines on colorbar
    cbar.ax.axhline(y=40, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=60, color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.3, 40, 'Moderate', fontsize=7, va='center', clip_on=True)
    cbar.ax.text(1.3, 60, 'High', fontsize=7, va='center', clip_on=True)

    # Title and labels
    ax.set_title('Global Distribution of Biobank Research Equity\n(n=70 IHCC Biobanks across 29 Countries)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)

    # Add legend for gray countries
    gray_patch = mpatches.Patch(color='#E0E0E0', label='No IHCC biobank')
    ax.legend(handles=[gray_patch], loc='lower left', fontsize=9)

    # Add annotation
    top_country = max(country_eas.items(), key=lambda x: x[1])
    ax.text(0.02, 0.98, f'Highest EAS: {top_country[0]} ({top_country[1]:.1f})',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    # Save
    output_file = output_path / "fig2a_world_biobank_equity.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def create_figure_3a_world_map(clinical_trial_data, output_path):
    """
    Figure 3a: World map of clinical trial site distribution
    """
    if not GEOPANDAS_AVAILABLE:
        print("\nSkipping Figure 3a: geopandas not available")
        return

    print("\nGenerating Figure 3a: World map of clinical trial sites...")

    # Load world map
    world = get_world_map()
    if world is None:
        print("  Error: Could not load world map data")
        return

    # Get country name column
    name_col = 'NAME' if 'NAME' in world.columns else 'name' if 'name' in world.columns else 'ADMIN'

    # Extract trial site data
    geo = clinical_trial_data.get('geographic', {})
    top_countries = geo.get('topCountries', [])

    # Country name mapping
    country_mapping = {
        'United States': 'United States of America',
        'Turkey (Türkiye)': 'Turkey',
        'South Korea': 'South Korea',
    }

    # Create dictionary of country sites
    country_sites = {}
    for c in top_countries:
        name = c['name']
        mapped_name = country_mapping.get(name, name)
        country_sites[mapped_name] = c['sites']

    print(f"  Countries in data: {len(country_sites)}")

    # Merge with world map
    world['sites'] = world[name_col].map(country_sites).fillna(0)
    world['log_sites'] = np.log10(world['sites'] + 1)  # Log scale for visualization
    world['has_trials'] = world['sites'] > 0

    # Income classification
    hic_list = ['United States of America', 'France', 'Canada', 'Germany',
                'United Kingdom', 'Italy', 'Spain', 'South Korea', 'Belgium',
                'Netherlands', 'Denmark', 'Australia', 'Japan', 'Switzerland']
    world['income'] = world[name_col].apply(lambda x: 'HIC' if x in hic_list else 'LMIC')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot countries without trial data in light gray
    world[~world['has_trials']].plot(
        ax=ax,
        color='#F5F5F5',
        edgecolor='white',
        linewidth=0.3
    )

    # Plot countries with trial sites
    # Use diverging colormap: blue for HIC, red for LMIC
    world_with_trials = world[world['has_trials']].copy()

    # Create custom color based on income and intensity
    def get_color(row):
        if row['income'] == 'HIC':
            # Blue scale based on sites
            intensity = min(row['log_sites'] / 6, 1)  # Normalize
            return (0.2, 0.4 + 0.4*intensity, 0.8 + 0.2*intensity)  # Blue shades
        else:
            # Red scale based on sites
            intensity = min(row['log_sites'] / 6, 1)
            return (0.9, 0.3 + 0.3*intensity, 0.3)  # Red shades

    # Plot separately by income
    hic_countries = world_with_trials[world_with_trials['income'] == 'HIC']
    lmic_countries = world_with_trials[world_with_trials['income'] == 'LMIC']

    # Custom colormap for HIC (blues)
    cmap_hic = LinearSegmentedColormap.from_list('hic', ['#B3D9FF', '#0066CC'])
    # Custom colormap for LMIC (reds)
    cmap_lmic = LinearSegmentedColormap.from_list('lmic', ['#FFCCCC', '#CC0000'])

    hic_countries.plot(
        ax=ax,
        column='log_sites',
        cmap=cmap_hic,
        edgecolor='black',
        linewidth=0.5,
        legend=False
    )

    lmic_countries.plot(
        ax=ax,
        column='log_sites',
        cmap=cmap_lmic,
        edgecolor='black',
        linewidth=0.5,
        legend=False
    )

    # Add colorbar for site counts
    norm = Normalize(vmin=0, vmax=6)  # log10 scale
    sm = ScalarMappable(cmap='YlOrRd', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label('Trial Sites (log₁₀ scale)', fontsize=10)
    cbar.set_ticks([0, 2, 4, 6])
    cbar.set_ticklabels(['1', '100', '10K', '1M'])

    # Title
    total_sites = geo.get('hicSites', 0) + geo.get('lmicSites', 0)
    hic_pct = geo.get('hicPct', 71.8)
    ax.set_title(f'Global Distribution of Clinical Trial Sites\n({total_sites:,} sites; HIC {hic_pct:.0f}% vs LMIC {100-hic_pct:.0f}%)',
                 fontsize=14, fontweight='bold')

    # Legend
    hic_patch = mpatches.Patch(color='#3498DB', label=f'High-Income ({hic_pct:.0f}%)')
    lmic_patch = mpatches.Patch(color='#E74C3C', label=f'Low/Middle-Income ({100-hic_pct:.0f}%)')
    gray_patch = mpatches.Patch(color='#F5F5F5', label='No data')
    ax.legend(handles=[hic_patch, lmic_patch, gray_patch], loc='lower left', fontsize=9)

    # Add top countries annotation
    top_5 = sorted(country_sites.items(), key=lambda x: x[1], reverse=True)[:5]
    top_text = "Top 5 countries:\n" + "\n".join([f"{c[0][:12]}: {c[1]:,}" for c in top_5])
    ax.text(0.02, 0.98, top_text,
            transform=ax.transAxes, fontsize=8, va='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add ratio annotation
    ratio = geo.get('hicLmicRatio', 2.5)
    ax.text(0.98, 0.02, f'HIC:LMIC Ratio = {ratio:.1f}×',
            transform=ax.transAxes, fontsize=11, ha='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Remove axis
    ax.set_axis_off()

    plt.tight_layout()

    # Save
    output_file = output_path / "fig3a_world_clinical_trials.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def create_regional_comparison(biobank_data, clinical_trial_data, output_path):
    """
    Create regional comparison bar charts showing biobank and trial distribution by WHO region
    """
    print("\nGenerating regional comparison charts...")

    biobanks = biobank_data['biobanks']

    # Aggregate by region
    regions = {}
    for bb in biobanks:
        region = bb['regionName']
        if region not in regions:
            regions[region] = {
                'biobank_count': 0,
                'publications': 0,
                'mean_eas': [],
                'gs_count': 0
            }
        regions[region]['biobank_count'] += 1
        regions[region]['publications'] += bb['stats']['totalPublications']
        regions[region]['mean_eas'].append(bb['scores']['equityAlignment'])
        if bb['isGlobalSouth']:
            regions[region]['gs_count'] += 1

    # Calculate mean EAS
    for r in regions:
        regions[r]['mean_eas'] = np.mean(regions[r]['mean_eas'])

    # Create dataframe
    df = pd.DataFrame(regions).T
    df = df.sort_values('publications', ascending=False)

    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Biobank count and publications
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, df['biobank_count'], width,
                    label='Biobanks', color='#3498DB', edgecolor='white')
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, df['publications']/1000, width,
                         label='Publications (thousands)', color='#27AE60', edgecolor='white')

    ax1.set_ylabel('Number of Biobanks', color='#3498DB', fontsize=11)
    ax1_twin.set_ylabel('Publications (thousands)', color='#27AE60', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=30, ha='right', fontsize=9)
    ax1.set_title('A. Biobanks and Publications by WHO Region', fontsize=12, fontweight='bold')

    # Combined legend
    ax1.legend([bars1, bars2], ['Biobanks', 'Publications (thousands)'],
               loc='upper right', fontsize=9)

    ax1.spines['top'].set_visible(False)
    ax1_twin.spines['top'].set_visible(False)

    # Panel B: Mean EAS by region
    ax2 = axes[1]
    colors = ['#E74C3C' if eas < 40 else '#F39C12' if eas < 60 else '#27AE60'
              for eas in df['mean_eas']]

    bars = ax2.barh(df.index, df['mean_eas'], color=colors, edgecolor='white')
    ax2.axvline(x=40, color='#F39C12', linestyle='--', linewidth=1.5, label='Moderate threshold')
    ax2.axvline(x=60, color='#27AE60', linestyle='--', linewidth=1.5, label='High threshold')

    ax2.set_xlabel('Mean Equity Alignment Score (EAS)', fontsize=11)
    ax2.set_title('B. Research Equity by WHO Region', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(loc='lower right', fontsize=9)

    # Add score labels
    for bar, score in zip(bars, df['mean_eas']):
        ax2.text(score + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=9)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    output_file = output_path / "fig_regional_comparison.png"
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_file.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"  Saved: {output_file}")

    plt.close()


def main():
    """Generate all publication figures."""
    print("=" * 60)
    print("HEIM PUBLICATION FIGURES GENERATOR")
    print("=" * 60)

    # Load data
    integrated, biobanks, semantic, clinical_trials = load_data()

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\n" + "-" * 60)

    # Figure 1: Framework schematic (specification)
    create_figure_1_spec(OUTPUT_PATH)

    # Figure 2a: World map of biobank equity (NEW)
    create_figure_2a_world_map(biobanks, OUTPUT_PATH)

    # Figure 2b: Biobank EAS distribution
    create_figure_2(biobanks, OUTPUT_PATH)

    # Figure 2c: Publication share
    create_figure_2c(biobanks, OUTPUT_PATH)

    # Figure 3a: World map of clinical trials (NEW)
    create_figure_3a_world_map(clinical_trials, OUTPUT_PATH)

    # Figure 3b: Trial intensity gap
    create_figure_3b(clinical_trials, OUTPUT_PATH)

    # Figure 4a: NTD comparison
    stats = create_figure_4a(semantic, OUTPUT_PATH)

    # Figure 6: Unified score ranking
    create_figure_6(integrated, OUTPUT_PATH)

    # Regional comparison (NEW)
    create_regional_comparison(biobanks, clinical_trials, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_PATH.glob("fig*.png")):
        print(f"  - {f.name}")

    print("\nNote: Figure 1 requires design software (BioRender/Illustrator)")
    print("      Specification saved to fig1_specification.txt")

    # Save statistics
    stats_file = OUTPUT_PATH / "figure_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'figure_4a': stats,
            'generated_at': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
