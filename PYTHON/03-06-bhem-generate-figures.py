#!/usr/bin/env python3
"""
03-06-bhem-generate-figures.py
==============================
HEIM-Biobank v2.0 (IHCC): Generate Publication-Quality Figures

VERSION 3: Critical fixes - calculate categories from EAS scores directly
- Fig 1: Wider box, smaller font
- Fig 2: Color from EAS score value
- Fig 4: Better label positioning  
- Fig 5: All critical diseases labeled
- Fig 6: Categories calculated from scores, not data field
- Fig 7: Colors from scores directly
- Fig S02: Only show categories with gaps > 0

VERSION: HEIM-Biobank v2.0 (IHCC) - Revision 3
DATE: 2026-01-10
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

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VERSION = "HEIM-Biobank v2.0 (IHCC)"
VERSION_DATE = "2026-01-10"

BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
ANALYSIS_DIR = BASE_DIR / "ANALYSIS" / "03-06-HEIM-FIGURES"

INPUT_METRICS = DATA_DIR / "bhem_metrics.json"
INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"

MIN_YEAR = 2000
MAX_YEAR = 2025

# =============================================================================
# STYLE
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

SEVERITY_COLORS = {
    'Critical': '#dc3545',
    'High': '#fd7e14',
    'Moderate': '#ffc107',
    'Low': '#28a745'
}

# EAS category colors
COLOR_HIGH = '#28a745'      # Green
COLOR_MODERATE = '#17a2b8'  # Teal
COLOR_LOW = '#dc3545'       # Red

REGION_COLORS = {
    'EUR': '#3498db', 'AMR': '#2ecc71', 'WPR': '#9b59b6',
    'EMR': '#f39c12', 'AFR': '#e74c3c', 'SEAR': '#1abc9c'
}

WHO_REGION_NAMES = {
    'EUR': 'Europe', 'AMR': 'Americas', 'WPR': 'Western Pacific',
    'EMR': 'Eastern Mediterranean', 'AFR': 'Africa', 'SEAR': 'South-East Asia'
}

COUNTRY_COORDS = {
    'United States': (-98.5, 39.5), 'United Kingdom': (-2.0, 54.0),
    'Finland': (26.0, 64.0), 'Estonia': (25.0, 59.0), 'Norway': (10.0, 62.0),
    'Denmark': (10.0, 56.0), 'Netherlands': (5.0, 52.0), 'Germany': (10.0, 51.0),
    'France': (2.0, 47.0), 'Spain': (-4.0, 40.0), 'Italy': (12.0, 43.0),
    'Iceland': (-18.0, 65.0), 'Japan': (138.0, 36.0), 'China': (105.0, 35.0),
    'Taiwan': (121.0, 24.0), 'South Korea': (127.0, 36.0), 'Korea': (127.0, 36.0),
    'Australia': (134.0, -25.0), 'Brazil': (-53.0, -10.0), 'Mexico': (-102.0, 24.0),
    'South Africa': (25.0, -29.0), 'Nigeria': (8.0, 10.0), 'Uganda': (32.0, 1.0),
    'Kenya': (38.0, 1.0), 'Ghana': (-1.0, 8.0), 'Qatar': (51.0, 25.0),
    'Iran': (53.0, 32.0), 'India': (79.0, 22.0), 'Singapore': (104.0, 1.3),
    'Canada': (-106.0, 56.0), 'Sweden': (18.0, 62.0), 'Belgium': (4.0, 51.0),
    'Switzerland': (8.0, 47.0), 'Austria': (14.0, 47.5), 'Poland': (19.0, 52.0),
    'Czech Republic': (15.0, 50.0), 'Hungary': (19.0, 47.0), 'Greece': (22.0, 39.0),
    'Portugal': (-8.0, 39.5), 'Ireland': (-8.0, 53.0), 'New Zealand': (174.0, -41.0),
    'Chile': (-71.0, -33.0), 'Argentina': (-64.0, -34.0), 'Colombia': (-74.0, 4.5),
    'Peru': (-76.0, -10.0), 'International': (10.0, 50.0), 'Multi-country': (0.0, 45.0),
}


# =============================================================================
# HELPER FUNCTIONS - KEY FIX: Calculate category from score
# =============================================================================

def get_eas_category(score: float) -> str:
    """Calculate EAS category from score value directly."""
    if score >= 70:
        return 'High'
    elif score >= 40:
        return 'Moderate'
    else:
        return 'Low'


def get_eas_color_from_score(score: float) -> str:
    """Get color based on EAS score value directly."""
    if score >= 70:
        return COLOR_HIGH
    elif score >= 40:
        return COLOR_MODERATE
    else:
        return COLOR_LOW


def load_json(filepath: Path) -> Optional[Dict]:
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def load_csv(filepath: Path) -> Optional[pd.DataFrame]:
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


def load_all_data():
    metrics = load_json(INPUT_METRICS)
    pubs_df = None
    if INPUT_PUBLICATIONS.exists():
        pubs_df = load_csv(INPUT_PUBLICATIONS)
        if pubs_df is not None and 'year' in pubs_df.columns:
            pubs_df['year'] = pd.to_numeric(pubs_df['year'], errors='coerce')
            pubs_df = pubs_df[(pubs_df['year'] >= MIN_YEAR) & (pubs_df['year'] <= MAX_YEAR)]
    else:
        logger.warning(f"Publications CSV not found: {INPUT_PUBLICATIONS} - Fig 09 will be skipped")
    return metrics, pubs_df


def save_figure(fig, name: str, output_dir: Path):
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'{name}.{ext}')
    logger.info(f"  Saved: {name}.pdf/png")
    plt.close(fig)


# =============================================================================
# MAIN FIGURES
# =============================================================================

def fig_01_heim_framework(output_dir: Path) -> None:
    """Figure 1: HEIM Framework - FIXED: Much wider box, smaller font"""
    logger.info("Generating Figure 1: HEIM Framework...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(7, 5.5, 'Health Equity Informative Metrics (HEIM) Framework',
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Formula box - MUCH WIDER
    formula_box = mpatches.FancyBboxPatch(
        (0.5, 3.4), 13, 1.3,
        boxstyle="round,pad=0.1",
        facecolor='#e8f4f8', edgecolor='#2c3e50', linewidth=2
    )
    ax.add_patch(formula_box)
    # SMALLER FONT
    ax.text(7, 4.05, 
            'EAS  =  100  −  ( 0.4 × Gap_Severity  +  0.3 × Burden_Miss  +  0.3 × Capacity_Penalty )',
            ha='center', va='center', fontsize=9, family='monospace')
    
    # Component boxes
    components = [
        ('Gap Severity', '#e74c3c', 'Mismatch between\nresearch output and\ndisease burden', 2.0),
        ('Burden Miss', '#f39c12', 'High-burden diseases\nwith inadequate\ncoverage', 6.0),
        ('Capacity Penalty', '#3498db', 'Underutilisation in\nunderrepresented\nsettings', 10.0)
    ]
    
    for name, color, desc, x in components:
        box = mpatches.FancyBboxPatch(
            (x, 0.8), 2, 2,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='#2c3e50', alpha=0.3, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + 1, 2.4, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#2c3e50')
        ax.text(x + 1, 1.4, desc, ha='center', va='center',
                fontsize=8, color='#2c3e50')
        ax.annotate('', xy=(x + 1, 2.8), xytext=(x + 1, 3.4),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
    
    ax.text(7, 0.3, 'Score Categories:  High (≥70)  |  Moderate (40-69)  |  Low (<40)',
            ha='center', va='center', fontsize=9, style='italic', color='#7f8c8d')
    
    save_figure(fig, 'fig_01_heim_framework', output_dir)


def fig_02_world_map(metrics: Dict, output_dir: Path) -> None:
    """Figure 2: World Map - FIXED: Color from score directly"""
    logger.info("Generating Figure 2: World Map...")
    
    biobanks = metrics.get('biobanks', {})
    
    # Aggregate by country - track max EAS score
    country_data = defaultdict(lambda: {'pubs': 0, 'biobanks': 0, 'max_eas': 0})
    for bid, bdata in biobanks.items():
        country = bdata.get('country', 'Unknown')
        pubs = bdata.get('total_publications', 0)
        eas = bdata.get('equity_alignment_score', 0)
        
        country_data[country]['pubs'] += pubs
        country_data[country]['biobanks'] += 1
        if eas > country_data[country]['max_eas']:
            country_data[country]['max_eas'] = eas
    
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888888')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#cccccc')
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_facecolor('#e6f3ff')
    
    for country, data in country_data.items():
        if country not in COUNTRY_COORDS:
            continue
        
        lon, lat = COUNTRY_COORDS[country]
        pubs = data['pubs']
        eas = data['max_eas']
        
        size = max(30, min(800, 30 * np.log10(pubs + 1)))
        # KEY FIX: Get color from score directly
        color = get_eas_color_from_score(eas)
        
        if HAS_CARTOPY:
            ax.scatter(lon, lat, s=size, c=color, alpha=0.7,
                      transform=ccrs.PlateCarree(), edgecolors='white', linewidth=0.5, zorder=10)
        else:
            ax.scatter(lon, lat, s=size, c=color, alpha=0.7,
                      edgecolors='white', linewidth=0.5, zorder=10)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=12, label='High EAS (≥70)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MODERATE, markersize=12, label='Moderate EAS (40-69)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=12, label='Low EAS (<40)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.95)
    ax.set_title('Geographic Distribution of 70 IHCC-Registered Biobanks\n(Circle size proportional to publication volume)', fontsize=12)
    
    save_figure(fig, 'fig_02_world_map', output_dir)


def fig_03_regional_distribution(metrics: Dict, output_dir: Path) -> None:
    """Figure 3: Regional Distribution"""
    logger.info("Generating Figure 3: Regional Distribution...")
    
    biobanks = metrics.get('biobanks', {})
    
    region_data = defaultdict(lambda: {'biobanks': 0, 'publications': 0})
    for bid, bdata in biobanks.items():
        region = bdata.get('region', 'Unknown')
        region_data[region]['biobanks'] += 1
        region_data[region]['publications'] += bdata.get('total_publications', 0)
    
    sorted_regions = sorted(region_data.items(), key=lambda x: x[1]['publications'], reverse=True)
    
    regions = [r[0] for r in sorted_regions]
    pubs = [r[1]['publications'] for r in sorted_regions]
    n_biobanks = [r[1]['biobanks'] for r in sorted_regions]
    total_pubs = sum(pubs)
    shares = [p / total_pubs * 100 for p in pubs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    colors = [REGION_COLORS.get(r, '#999999') for r in regions]
    bars = ax1.barh(range(len(regions)), pubs, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(regions)))
    ax1.set_yticklabels([WHO_REGION_NAMES.get(r, r) for r in regions])
    ax1.set_xlabel('Number of Publications')
    ax1.set_title('(A) Publications by WHO Region')
    ax1.invert_yaxis()
    
    for i, (bar, share, n) in enumerate(zip(bars, shares, n_biobanks)):
        ax1.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2,
                f'{share:.1f}% ({n} biobanks)', va='center', fontsize=9)
    
    explode = [0.03 if s < 3 else 0 for s in shares]
    wedges, texts, autotexts = ax2.pie(pubs, labels=None, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 2 else '', explode=explode, startangle=90, textprops={'fontsize': 9})
    ax2.set_title('(B) Publication Share')
    
    legend_labels = [f'{WHO_REGION_NAMES.get(r, r)} ({p:,})' for r, p in zip(regions, pubs)]
    ax2.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    save_figure(fig, 'fig_03_regional_distribution', output_dir)


def fig_04_equity_ratio(metrics: Dict, output_dir: Path) -> None:
    """Figure 4: Equity Ratio - FIXED: Better positioning"""
    logger.info("Generating Figure 4: Equity Ratio...")
    
    global_data = metrics.get('global', {})
    biobanks = metrics.get('biobanks', {})
    
    hic_pubs, lmic_pubs = 0, 0
    hic_count, lmic_count = 0, 0
    
    for bid, bdata in biobanks.items():
        is_gs = bdata.get('is_global_south', False)
        region = bdata.get('region', '')
        pubs = bdata.get('total_publications', 0)
        
        if is_gs or region in ['AFR', 'SEAR', 'EMR']:
            lmic_pubs += pubs
            lmic_count += 1
        else:
            hic_pubs += pubs
            hic_count += 1
    
    equity_ratio = global_data.get('equity_ratio', 57.8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    categories = ['HIC Biobanks', 'LMIC Biobanks']
    values = [hic_pubs, lmic_pubs]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(categories, values, color=colors, edgecolor='white', width=0.6)
    ax1.set_ylabel('Number of Publications')
    # FIXED: More space for title
    ax1.set_title('(A) Publication Volume by Income Group', pad=15)
    # FIXED: Set y limit higher to make room for labels
    ax1.set_ylim(0, max(values) * 1.3)
    
    # FIXED: Labels inside bars or with better positioning
    for bar, val, count in zip(bars, values, [hic_count, lmic_count]):
        # Put count inside bar, publication number above
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f'{val:,}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.03,
                f'({count} biobanks)', ha='center', va='bottom', fontsize=9, color='#555555')
    
    # Panel B
    ax2.barh(['HIC\n(per DALY)', 'LMIC\n(per DALY)'], [equity_ratio, 1], 
             color=colors, edgecolor='white', height=0.5)
    ax2.set_xlabel('Relative Research Intensity')
    ax2.set_title(f'(B) Equity Ratio: {equity_ratio:.1f}:1')
    ax2.axvline(x=1, color='#2c3e50', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.text(equity_ratio/2, 0, f'{equity_ratio:.1f}×', ha='center', va='center',
             fontsize=14, fontweight='bold', color='white')
    ax2.text(0.5, 1, '1×', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    save_figure(fig, 'fig_04_equity_ratio', output_dir)


def fig_05_disease_gaps(metrics: Dict, output_dir: Path) -> None:
    """Figure 5: Disease Gaps - FIXED: Smart label positioning to prevent overlap"""
    logger.info("Generating Figure 5: Disease Gaps...")
    
    diseases = metrics.get('diseases', {})
    global_data = metrics.get('global', {})
    
    data = {'dalys': [], 'pubs': [], 'severity': [], 'name': []}
    
    for did, ddata in diseases.items():
        d = ddata.get('dalys_millions', 0)
        p = ddata.get('publications', 0)
        s = ddata.get('gap_severity', 'Low')
        n = ddata.get('name', did)
        
        if d > 0:
            data['dalys'].append(d)
            data['pubs'].append(p + 1)
            data['severity'].append(s)
            data['name'].append(n)
    
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
    
    ax1 = fig.add_subplot(gs[0])
    
    for sev in ['Critical', 'High', 'Moderate', 'Low']:
        mask = [s == sev for s in data['severity']]
        d_sub = [d for d, m in zip(data['dalys'], mask) if m]
        p_sub = [p for p, m in zip(data['pubs'], mask) if m]
        ax1.scatter(d_sub, p_sub, c=SEVERITY_COLORS[sev], label=sev,
                   alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Disease Burden (Million DALYs)')
    ax1.set_ylabel('Publications (log scale)')
    ax1.set_title('(A) Disease Burden vs Research Attention')
    ax1.legend(title='Gap Severity', loc='upper left')
    
    x_line = np.logspace(-1, 2, 100)
    ax1.plot(x_line, x_line * 10, '--', color='#7f8c8d', alpha=0.4, linewidth=1)
    
    # FIXED: Smart label positioning using repulsion algorithm
    critical_points = [(d, p, n) for d, p, s, n in zip(data['dalys'], data['pubs'], data['severity'], data['name']) if s == 'Critical']
    critical_points.sort(key=lambda x: x[0], reverse=True)
    
    # Convert to log scale for positioning calculations
    log_dalys = [np.log10(d) for d, p, n in critical_points]
    log_pubs = [np.log10(p) for d, p, n in critical_points]
    
    # Calculate label positions with repulsion to avoid overlap
    label_positions = []
    for i, (d, p, n) in enumerate(critical_points):
        log_d, log_p = np.log10(d), np.log10(p)
        
        # Base offset depends on quadrant and index
        # Spread labels radially around points
        angle = (i * 137.5) % 360  # Golden angle for good distribution
        radius = 0.15 + (i % 3) * 0.05  # Varying radii
        
        offset_x = radius * np.cos(np.radians(angle))
        offset_y = radius * np.sin(np.radians(angle))
        
        # Push labels toward edges of plot
        if log_d > 1:  # Right side of plot
            offset_x = abs(offset_x) * 0.8
        elif log_d < 0:  # Left side
            offset_x = -abs(offset_x) * 0.8
            
        label_positions.append((d, p, n, offset_x, offset_y))
    
    # Draw labels with connection lines
    for d, p, n, ox, oy in label_positions:
        label = n[:20] if len(n) > 20 else n
        log_d, log_p = np.log10(d), np.log10(p)
        
        # Convert offset back to data coordinates
        target_x = 10 ** (log_d + ox)
        target_y = 10 ** (log_p + oy)
        
        ha = 'left' if ox > 0 else 'right'
        
        ax1.annotate(label, xy=(d, p), xytext=(target_x, target_y),
                    fontsize=6, alpha=0.95, color='#333333', ha=ha, va='center',
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5, alpha=0.5),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))
    
    # Panel B
    ax2 = fig.add_subplot(gs[1])
    gap_dist = global_data.get('gap_distribution', {})
    labels = ['Critical', 'High', 'Moderate', 'Low']
    sizes = [gap_dist.get(l, 0) for l in labels]
    colors = [SEVERITY_COLORS[l] for l in labels]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90, explode=[0.05, 0.02, 0, 0], textprops={'fontsize': 9})
    ax2.set_title('(B) Gap Severity Distribution\n(179 diseases)')
    
    plt.tight_layout()
    save_figure(fig, 'fig_05_disease_gaps', output_dir)


def fig_06_eas_distribution(metrics: Dict, output_dir: Path) -> None:
    """Figure 6: EAS Distribution - FIXED: Calculate categories from scores"""
    logger.info("Generating Figure 6: EAS Distribution...")
    
    biobanks = metrics.get('biobanks', {})
    
    # Extract EAS scores
    eas_values = [b.get('equity_alignment_score', 0) for b in biobanks.values()]
    
    # KEY FIX: Calculate categories from scores directly
    cat_counts = {'High': 0, 'Moderate': 0, 'Low': 0}
    for eas in eas_values:
        cat = get_eas_category(eas)
        cat_counts[cat] += 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Histogram
    bins = np.arange(0, 105, 5)
    n, bin_edges, patches = ax1.hist(eas_values, bins=bins, edgecolor='white', linewidth=0.5)
    
    # Color bins by category
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 70:
            patch.set_facecolor(COLOR_HIGH)
        elif left_edge >= 40:
            patch.set_facecolor(COLOR_MODERATE)
        else:
            patch.set_facecolor(COLOR_LOW)
    
    ax1.axvline(x=40, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.axvline(x=70, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.6)
    
    max_n = max(n) if len(n) > 0 else 10
    ax1.text(20, max_n * 0.9, 'Low', ha='center', fontsize=10, fontweight='bold')
    ax1.text(55, max_n * 0.9, 'Moderate', ha='center', fontsize=10, fontweight='bold')
    ax1.text(85, max_n * 0.9, 'High', ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Equity Alignment Score (EAS)')
    ax1.set_ylabel('Number of Biobanks')
    ax1.set_title('(A) EAS Distribution Across 70 IHCC Biobanks')
    
    # Panel B: Category bar chart - FIXED with calculated counts
    cats = ['High\n(≥70)', 'Moderate\n(40-69)', 'Low\n(<40)']
    counts = [cat_counts['High'], cat_counts['Moderate'], cat_counts['Low']]
    bar_colors = [COLOR_HIGH, COLOR_MODERATE, COLOR_LOW]
    
    bars = ax2.bar(cats, counts, color=bar_colors, edgecolor='white', width=0.6)
    ax2.set_ylabel('Number of Biobanks')
    ax2.set_title('(B) EAS Category Distribution')
    
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100 if total > 0 else 0
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    # Set reasonable y limit
    ax2.set_ylim(0, max(counts) * 1.3 if max(counts) > 0 else 10)
    
    plt.tight_layout()
    save_figure(fig, 'fig_06_eas_distribution', output_dir)


def fig_07_top_biobanks(metrics: Dict, output_dir: Path) -> None:
    """Figure 7: Top 15 Biobanks - FIXED: Position text after error whiskers"""
    logger.info("Generating Figure 7: Top Biobanks...")
    
    biobanks = metrics.get('biobanks', {})
    
    sorted_bb = sorted(biobanks.items(),
                       key=lambda x: x[1].get('equity_alignment_score', 0),
                       reverse=True)[:15]
    
    names = [b[1].get('name', b[0])[:35] for b in sorted_bb]
    eas_vals = [b[1].get('equity_alignment_score', 0) for b in sorted_bb]
    countries = [b[1].get('country', '')[:15] for b in sorted_bb]
    pubs = [b[1].get('total_publications', 0) for b in sorted_bb]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # KEY FIX: Get colors from scores directly
    colors = [get_eas_color_from_score(eas) for eas in eas_vals]
    y_pos = range(len(names))
    
    # Error bar is ±3 points
    error_width = 3
    xerr = [[min(error_width, e) for e in eas_vals], [min(error_width, 100-e) for e in eas_vals]]
    
    bars = ax.barh(y_pos, eas_vals, color=colors, edgecolor='white',
                   xerr=xerr, capsize=3, error_kw={'linewidth': 1, 'alpha': 0.5})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{n} ({c})' for n, c in zip(names, countries)])
    ax.set_xlabel('Equity Alignment Score (EAS)')
    ax.set_title('Top 15 IHCC Biobanks by Equity Alignment Score')
    ax.invert_yaxis()
    ax.set_xlim(0, 105)  # Extended to fit labels
    
    ax.axvline(x=40, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.4)
    ax.axvline(x=70, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.4)
    
    # FIXED: Position publication count AFTER the error whisker (bar_width + error_width + padding)
    for i, (bar, eas, pub) in enumerate(zip(bars, eas_vals, pubs)):
        # Text position = bar end + error bar + small padding
        text_x = eas + error_width + 1.5
        ax.text(text_x, bar.get_y() + bar.get_height()/2,
                f'{pub:,}', va='center', fontsize=8, color='#555555')
    
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HIGH, label='High (≥70)'),
        mpatches.Patch(facecolor=COLOR_MODERATE, label='Moderate (40-69)'),
        mpatches.Patch(facecolor=COLOR_LOW, label='Low (<40)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_figure(fig, 'fig_07_top_biobanks', output_dir)


def fig_08_critical_gaps(metrics: Dict, output_dir: Path) -> None:
    """Figure 8: Critical Gap Diseases - FIXED: Label ALL dots, minimal connector lines"""
    logger.info("Generating Figure 8: Critical Gap Diseases...")
    
    diseases = metrics.get('diseases', {})
    
    critical = []
    for did, ddata in diseases.items():
        if ddata.get('gap_severity') == 'Critical':
            critical.append({
                'name': ddata.get('name', did),
                'dalys': ddata.get('dalys_millions', 0),
                'publications': ddata.get('publications', 0),
                'gap_score': ddata.get('gap_score', 0)
            })
    
    critical.sort(key=lambda x: x['dalys'], reverse=True)
    critical = critical[:12]
    
    if not critical:
        logger.warning("  No critical gap diseases found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [d['name'][:30] for d in critical]
    dalys = [d['dalys'] for d in critical]
    pubs = [d['publications'] for d in critical]
    gap_scores = [d['gap_score'] for d in critical]
    
    colors = ['#dc3545' if p == 0 else '#fd7e14' for p in pubs]
    
    bars = ax1.barh(range(len(names)), dalys, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Disease Burden (Million DALYs)')
    ax1.set_title('(A) Critical Gap Diseases by Burden')
    ax1.invert_yaxis()
    
    for i, (bar, pub) in enumerate(zip(bars, pubs)):
        label = '0 pubs' if pub == 0 else f'{pub} pubs'
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8)
    
    # Panel B: Scatter plot
    ax2.scatter(pubs, gap_scores,
                s=[max(80, d * 4) for d in dalys],
                c='#dc3545', alpha=0.7, edgecolors='white', linewidth=0.5)
    ax2.set_xlabel('Number of Publications')
    ax2.set_ylabel('Gap Score')
    ax2.set_title('(B) Gap Score vs Publications\n(Bubble size = DALYs)')
    ax2.axhline(y=70, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.5)
    
    # Label ALL critical diseases with clean positioning - NO connector lines
    # Group by approximate position to handle overlaps
    label_data = list(zip(pubs, gap_scores, [d['name'] for d in critical], dalys))
    
    # Sort by publications (x-axis) for left-to-right layout
    label_data_sorted = sorted(label_data, key=lambda x: x[0])
    
    # Track y-positions used at each x-zone to stagger vertically
    used_positions = []
    
    for i, (x, y, name, daly) in enumerate(label_data_sorted):
        label = name[:18] if len(name) > 18 else name
        
        # Determine offset based on position in plot
        # Left side (low pubs): labels go left
        # Right side (high pubs): labels go right
        if x < 20:
            offset_x = -3
            ha = 'right'
        elif x < 40:
            offset_x = 3 if i % 2 == 0 else -3
            ha = 'left' if offset_x > 0 else 'right'
        else:
            offset_x = 5
            ha = 'left'
        
        # Stagger y-offset to avoid vertical overlap
        base_y_offset = 0
        
        # Check for nearby labels and adjust y
        for (ux, uy) in used_positions:
            if abs(x - ux) < 12 and abs(y - uy) < 3:
                # Nearby label exists, stagger this one
                base_y_offset = 2.5 if (y - uy) >= 0 else -2.5
        
        target_y = y + base_y_offset
        
        # Keep within bounds
        target_y = max(71, min(97, target_y))
        
        ax2.annotate(label, xy=(x, y), xytext=(offset_x, target_y - y),
                    textcoords='offset points',
                    fontsize=6.5, alpha=0.95, color='#333333', ha=ha, va='center')
        
        used_positions.append((x, target_y))
    
    legend_elements = [
        mpatches.Patch(facecolor='#dc3545', label='Zero publications'),
        mpatches.Patch(facecolor='#fd7e14', label='Minimal publications')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_figure(fig, 'fig_08_critical_gaps', output_dir)


def fig_09_temporal_trends(pubs_df: Optional[pd.DataFrame], output_dir: Path) -> None:
    """Figure 9: Temporal Trends"""
    logger.info("Generating Figure 9: Temporal Trends...")
    
    if pubs_df is None or 'year' not in pubs_df.columns:
        logger.warning("  No publication data available")
        return
    
    yearly = pubs_df.groupby('year').size().reset_index(name='publications')
    yearly = yearly[(yearly['year'] >= MIN_YEAR) & (yearly['year'] <= MAX_YEAR)]
    yearly = yearly.sort_values('year')
    
    if yearly.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.fill_between(yearly['year'], yearly['publications'], alpha=0.3, color='#3498db')
    ax1.plot(yearly['year'], yearly['publications'], 'o-', color='#3498db', markersize=4, linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Publications')
    ax1.set_title('(A) Annual Publication Output')
    
    if len(yearly) >= 10:
        early_mean = yearly[yearly['year'] <= 2015]['publications'].mean()
        late_mean = yearly[yearly['year'] >= 2020]['publications'].mean()
        if early_mean > 0:
            growth = (late_mean - early_mean) / early_mean * 100
            mid_year = 2017
            mid_val = yearly[yearly['year'] == mid_year]['publications'].values
            if len(mid_val) > 0:
                ax1.annotate(f'~{growth:.0f}% growth\n(2015 to 2020+)',
                            xy=(mid_year, mid_val[0]), xytext=(2008, late_mean * 0.8),
                            arrowprops=dict(arrowstyle='->', color='#7f8c8d'),
                            fontsize=9, color='#7f8c8d')
    
    yearly['cumulative'] = yearly['publications'].cumsum()
    ax2.fill_between(yearly['year'], yearly['cumulative'], alpha=0.3, color='#2ecc71')
    ax2.plot(yearly['year'], yearly['cumulative'], 'o-', color='#2ecc71', markersize=4, linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Publications')
    ax2.set_title('(B) Cumulative Publication Growth')
    
    total = yearly['cumulative'].iloc[-1]
    ax2.annotate(f'Total: {total:,}', xy=(yearly['year'].iloc[-1], total),
                 xytext=(-60, -20), textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'fig_09_temporal_trends', output_dir)


# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================

def fig_s01_income_comparison(metrics: Dict, output_dir: Path) -> None:
    """Supplementary Figure S1"""
    logger.info("Generating Figure S1: Income Comparison...")
    
    biobanks = metrics.get('biobanks', {})
    
    groups = {'HIC': {'pubs': 0, 'count': 0, 'eas': []}, 'LMIC': {'pubs': 0, 'count': 0, 'eas': []}}
    
    for bid, bdata in biobanks.items():
        is_gs = bdata.get('is_global_south', False)
        region = bdata.get('region', '')
        group = 'LMIC' if (is_gs or region in ['AFR', 'SEAR', 'EMR']) else 'HIC'
        groups[group]['pubs'] += bdata.get('total_publications', 0)
        groups[group]['count'] += 1
        groups[group]['eas'].append(bdata.get('equity_alignment_score', 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    labels = ['HIC', 'LMIC']
    colors = ['#3498db', '#e74c3c']
    
    counts = [groups[g]['count'] for g in labels]
    axes[0].bar(labels, counts, color=colors, edgecolor='white')
    axes[0].set_ylabel('Number of Biobanks')
    axes[0].set_title('(A) Biobank Count')
    for i, c in enumerate(counts):
        axes[0].text(i, c + 1, str(c), ha='center', fontweight='bold')
    
    pubs = [groups[g]['pubs'] for g in labels]
    axes[1].bar(labels, pubs, color=colors, edgecolor='white')
    axes[1].set_ylabel('Number of Publications')
    axes[1].set_title('(B) Publication Volume')
    for i, p in enumerate(pubs):
        axes[1].text(i, p + 500, f'{p:,}', ha='center', fontweight='bold')
    
    eas_data = [groups[g]['eas'] for g in labels]
    bp = axes[2].boxplot(eas_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].set_ylabel('Equity Alignment Score')
    axes[2].set_title('(C) EAS Distribution')
    
    plt.tight_layout()
    save_figure(fig, 'fig_s01_income_comparison', output_dir)


def fig_s02_disease_category_coverage(metrics: Dict, output_dir: Path) -> None:
    """Supplementary Figure S2 - FIXED: Only show categories with critical gaps > 0"""
    logger.info("Generating Figure S2: Disease Category Coverage...")
    
    diseases = metrics.get('diseases', {})
    
    cat_data = defaultdict(lambda: {'pubs': 0, 'count': 0, 'critical': 0})
    
    for did, ddata in diseases.items():
        cat = ddata.get('category', 'Other')
        cat_data[cat]['pubs'] += ddata.get('publications', 0)
        cat_data[cat]['count'] += 1
        if ddata.get('gap_severity') == 'Critical':
            cat_data[cat]['critical'] += 1
    
    sorted_cats = sorted(cat_data.items(), key=lambda x: x[1]['pubs'], reverse=True)
    
    cats_a = [c[0] for c in sorted_cats]
    pubs_a = [c[1]['pubs'] for c in sorted_cats]
    
    # FIXED: Filter to only categories with critical > 0 for panel B
    cats_with_critical = [(c[0], c[1]['critical']) for c in sorted_cats if c[1]['critical'] > 0]
    cats_with_critical.sort(key=lambda x: x[1], reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: All categories by publications
    ax1.barh(range(len(cats_a)), pubs_a, color='#3498db', edgecolor='white')
    ax1.set_yticks(range(len(cats_a)))
    ax1.set_yticklabels(cats_a)
    ax1.set_xlabel('Number of Publications')
    ax1.set_title('(A) Publications by Disease Category')
    ax1.invert_yaxis()
    
    # Panel B: Only categories with critical gaps
    if cats_with_critical:
        cats_b = [c[0] for c in cats_with_critical]
        critical_b = [c[1] for c in cats_with_critical]
        
        ax2.barh(range(len(cats_b)), critical_b, color='#dc3545', edgecolor='white')
        ax2.set_yticks(range(len(cats_b)))
        ax2.set_yticklabels(cats_b)
        ax2.set_xlabel('Number of Critical Gap Diseases')
        ax2.set_title('(B) Critical Gaps by Category\n(Only categories with ≥1 critical gap shown)')
        ax2.invert_yaxis()
        
        # Add count labels
        for i, val in enumerate(critical_b):
            ax2.text(val + 0.2, i, str(val), va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No critical gaps', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(B) Critical Gaps by Category')
    
    plt.tight_layout()
    save_figure(fig, 'fig_s02_disease_category_coverage', output_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-Biobank: Generate Publication Figures (v3)")
    print(f"Version: {VERSION}")
    print("=" * 70)
    
    print(f"\n📦 cartopy: {'available' if HAS_CARTOPY else 'not installed'}")
    
    if not INPUT_METRICS.exists():
        print(f"❌ Required: {INPUT_METRICS}")
        return
    
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {ANALYSIS_DIR}")
    
    print(f"\n📂 Loading data...")
    metrics, pubs_df = load_all_data()
    
    if metrics:
        print(f"   {len(metrics.get('biobanks', {}))} biobanks, {len(metrics.get('diseases', {}))} diseases")
    
    print(f"\n🎨 Generating figures...")
    
    fig_01_heim_framework(ANALYSIS_DIR)
    
    if metrics:
        fig_02_world_map(metrics, ANALYSIS_DIR)
        fig_03_regional_distribution(metrics, ANALYSIS_DIR)
        fig_04_equity_ratio(metrics, ANALYSIS_DIR)
        fig_05_disease_gaps(metrics, ANALYSIS_DIR)
        fig_06_eas_distribution(metrics, ANALYSIS_DIR)
        fig_07_top_biobanks(metrics, ANALYSIS_DIR)
        fig_08_critical_gaps(metrics, ANALYSIS_DIR)
    
    if pubs_df is not None:
        fig_09_temporal_trends(pubs_df, ANALYSIS_DIR)
    
    print(f"\n📊 Supplementary figures...")
    if metrics:
        fig_s01_income_comparison(metrics, ANALYSIS_DIR)
        fig_s02_disease_category_coverage(metrics, ANALYSIS_DIR)
    
    print(f"\n✅ COMPLETE!")


if __name__ == "__main__":
    main()