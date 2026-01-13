#!/usr/bin/env python3
"""
WHO Genomic Studies Global Map - Publication Quality (Cartopy Version)
=======================================================================
True Natural Earth projection with accurate country boundaries.
Designed for Mac with network access to download shapefiles.

REQUIREMENTS:
    pip install cartopy matplotlib numpy

USAGE:
    python PYTHON/04-00-WHO-genomic-map.py

OUTPUT:
    ANALYSIS/04-00-WHO-GENOMIC-MAP/who_genomic_map_publication.png
    ANALYSIS/04-00-WHO-GENOMIC-MAP/who_genomic_map_publication.pdf
    ANALYSIS/04-00-WHO-GENOMIC-MAP/heim_equity_panel.png
    ANALYSIS/04-00-WHO-GENOMIC-MAP/heim_equity_panel.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

# Setup paths - works from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'ANALYSIS', '04-00-WHO-GENOMIC-MAP')
os.makedirs(output_dir, exist_ok=True)

# Publication settings
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# ============================================================================
# DATA - WHO 2024 Report
# ============================================================================

TOP_10_DATA = {
    'China': {'studies': 1420, 'pct': 21.8, 'lon': 104.0, 'lat': 35.0, 'income': 'UMIC'},
    'United States': {'studies': 980, 'pct': 15.1, 'lon': -98.0, 'lat': 39.0, 'income': 'HIC'},
    'Italy': {'studies': 520, 'pct': 8.0, 'lon': 12.6, 'lat': 42.5, 'income': 'HIC'},
    'France': {'studies': 380, 'pct': 5.8, 'lon': 2.2, 'lat': 46.2, 'income': 'HIC'},
    'Germany': {'studies': 340, 'pct': 5.2, 'lon': 10.5, 'lat': 51.2, 'income': 'HIC'},
    'United Kingdom': {'studies': 310, 'pct': 4.8, 'lon': -2.0, 'lat': 54.0, 'income': 'HIC'},
    'Spain': {'studies': 240, 'pct': 3.7, 'lon': -3.7, 'lat': 40.0, 'income': 'HIC'},
    'Japan': {'studies': 180, 'pct': 2.8, 'lon': 138.0, 'lat': 36.0, 'income': 'HIC'},
    'India': {'studies': 150, 'pct': 2.3, 'lon': 78.9, 'lat': 22.0, 'income': 'LMIC'},
    'Netherlands': {'studies': 120, 'pct': 1.8, 'lon': 5.3, 'lat': 52.1, 'income': 'HIC'},
}

# World Bank income groups (ISO A3 codes)
INCOME_GROUPS = {
    'HIC': ['USA', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'JPN', 'NLD', 'CAN', 'AUS',
            'KOR', 'CHE', 'AUT', 'BEL', 'SWE', 'NOR', 'DNK', 'FIN', 'IRL', 'NZL',
            'SGP', 'ISR', 'PRT', 'GRC', 'CZE', 'POL', 'HUN', 'SVK', 'SVN', 'EST',
            'LVA', 'LTU', 'CHL', 'URY', 'PAN', 'HRV', 'ARE', 'SAU', 'QAT', 'KWT'],
    'UMIC': ['CHN', 'BRA', 'MEX', 'RUS', 'TUR', 'ARG', 'ZAF', 'MYS', 'THA', 'COL',
             'PER', 'ROU', 'BGR', 'SRB', 'KAZ', 'AZE', 'IRN', 'IRQ', 'JOR', 'CUB'],
    'LMIC': ['IND', 'IDN', 'PHL', 'VNM', 'EGY', 'PAK', 'BGD', 'NGA', 'UKR', 'MAR',
             'KEN', 'GHA', 'CMR', 'TZA', 'UGA', 'MMR', 'NPL', 'LKA', 'KHM', 'BOL'],
    'LIC': ['ETH', 'COD', 'SDN', 'AFG', 'YEM', 'MOZ', 'MDG', 'MWI', 'MLI', 'BFA',
            'NER', 'TCD', 'SOM', 'CAF', 'SSD', 'BDI', 'RWA', 'SLE', 'LBR', 'HTI'],
}

# Colors
COLORS = {
    'ocean': '#F5F8FA',
    'land_hic': '#E8F4F8',
    'land_umic': '#E8EDF2',
    'land_lmic': '#F0EDE8',
    'land_lic': '#F5F0EB',
    'land_default': '#EDEDED',
    'border_hic': '#1A5276',
    'border_umic': '#2874A6',
    'border_lmic': '#D4AC0D',
    'border_lic': '#BA4A00',
    'border_default': '#AAAAAA',
}

STUDY_CMAP = plt.cm.Blues


def get_income_group(iso_a3):
    """Get income group for country."""
    for group, countries in INCOME_GROUPS.items():
        if iso_a3 in countries:
            return group
    return 'Unknown'


def create_publication_map():
    """Create publication-quality world map with Natural Earth projection."""
    print("Creating publication-quality world map...")
    
    # Robinson projection - excellent for thematic world maps
    projection = ccrs.Robinson()
    
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.82], projection=projection)
    ax.set_global()
    
    # Ocean
    ax.add_feature(cfeature.OCEAN, facecolor=COLORS['ocean'], zorder=0)
    
    # Get Natural Earth countries (110m resolution for clean rendering)
    shpfilename = shapereader.natural_earth(
        resolution='110m',
        category='cultural',
        name='admin_0_countries'
    )
    reader = shapereader.Reader(shpfilename)
    
    # Normalize study counts for coloring
    max_studies = max(d['studies'] for d in TOP_10_DATA.values())
    norm = Normalize(vmin=0, vmax=max_studies)
    
    # Country name mapping (handles variations)
    name_mapping = {
        'United States': ['United States of America', 'United States'],
        'United Kingdom': ['United Kingdom', 'United Kingdom of Great Britain and Northern Ireland'],
    }
    
    country_studies = {}
    for name, data in TOP_10_DATA.items():
        country_studies[name] = data['studies']
        if name in name_mapping:
            for variant in name_mapping[name]:
                country_studies[variant] = data['studies']
    
    # Draw all countries
    for country in reader.records():
        name = country.attributes['NAME']
        name_long = country.attributes.get('NAME_LONG', name)
        iso_a3 = country.attributes.get('ISO_A3', '')
        
        # Get study count
        studies = country_studies.get(name, country_studies.get(name_long, 0))
        
        # Get income group
        income = get_income_group(iso_a3)
        
        # Determine colors
        if studies > 0:
            facecolor = STUDY_CMAP(norm(studies))
            edgecolor = '#1A5276'
            linewidth = 0.8
        else:
            # Color by income group for underrepresented countries
            facecolor = COLORS.get(f'land_{income.lower()}', COLORS['land_default'])
            edgecolor = COLORS.get(f'border_{income.lower()}', COLORS['border_default'])
            linewidth = 0.3
        
        ax.add_geometries(
            [country.geometry],
            ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=1
        )
    
    # Label offsets (in projection coordinates, roughly)
    label_config = {
        'China': {'dx': 2500000, 'dy': 2000000, 'ha': 'left'},
        'United States': {'dx': -3500000, 'dy': 1500000, 'ha': 'right'},
        'Italy': {'dx': 1500000, 'dy': -1000000, 'ha': 'left'},
        'France': {'dx': -2000000, 'dy': 1200000, 'ha': 'right'},
        'Germany': {'dx': 1500000, 'dy': 1000000, 'ha': 'left'},
        'United Kingdom': {'dx': -2500000, 'dy': 1500000, 'ha': 'right'},
        'Spain': {'dx': -2500000, 'dy': -800000, 'ha': 'right'},
        'Japan': {'dx': 2000000, 'dy': 1000000, 'ha': 'left'},
        'India': {'dx': 2000000, 'dy': -1500000, 'ha': 'left'},
        'Netherlands': {'dx': 1500000, 'dy': 1200000, 'ha': 'left'},
    }
    
    # Draw proportional symbols
    for rank, (country, data) in enumerate(TOP_10_DATA.items(), 1):
        # Transform coordinates
        x, y = projection.transform_point(data['lon'], data['lat'], ccrs.PlateCarree())
        studies = data['studies']
        
        # Area-proportional radius (in projection units)
        max_radius = 1200000
        radius = np.sqrt(studies / max_studies) * max_radius
        
        color = STUDY_CMAP(norm(studies))
        
        # Glow effect
        glow = Circle((x, y), radius * 1.25,
                     facecolor=color, alpha=0.3,
                     transform=ax.transData, zorder=3)
        ax.add_patch(glow)
        
        # Main symbol
        circle = Circle((x, y), radius,
                        facecolor=color,
                        edgecolor='#1A5276',
                        linewidth=1.5,
                        transform=ax.transData, zorder=4)
        ax.add_patch(circle)
        
        # Rank number
        txt = ax.text(x, y, str(rank),
                     fontsize=9 if rank <= 2 else 8,
                     fontweight='bold',
                     color='white',
                     ha='center', va='center',
                     transform=ax.transData, zorder=5)
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2.5, foreground='#1A5276'),
            path_effects.Normal()
        ])
        
        # Label with leader line
        cfg = label_config.get(country, {'dx': 1500000, 'dy': 500000, 'ha': 'left'})
        lx = x + cfg['dx']
        ly = y + cfg['dy']
        
        # Leader line
        ax.plot([x, lx], [y, ly],
               color='#555555', linewidth=0.8,
               transform=ax.transData, zorder=2,
               solid_capstyle='round')
        
        # Label
        label_text = f"{country}\n{studies:,}"
        ax.text(lx, ly, label_text,
               fontsize=7.5, ha=cfg['ha'], va='center',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='#CCCCCC',
                        linewidth=0.5,
                        alpha=0.95),
               transform=ax.transData, zorder=6)
    
    # Title
    ax.set_title(
        'Global Distribution of Genomic Clinical Studies (1990–2024)\n'
        'Top 10 Countries Account for ~70% of All 6,500+ Studies Worldwide',
        fontsize=12, fontweight='bold', pad=15, color='#263238'
    )
    
    # === LEGENDS ===
    
    # Colorbar
    cbar_ax = fig.add_axes([0.32, 0.055, 0.25, 0.018])
    sm = ScalarMappable(cmap=STUDY_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Number of Studies', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    
    # Size legend
    legend_ax = fig.add_axes([0.02, 0.01, 0.14, 0.09])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')
    
    legend_ax.text(0.5, 0.95, 'Symbol Size (area = studies)',
                   fontsize=7.5, ha='center', va='top', fontweight='bold')
    
    for i, size in enumerate([200, 700, 1400]):
        x_pos = 0.18 + i * 0.28
        r = np.sqrt(size / max_studies) * 0.14
        circle = Circle((x_pos, 0.50), r,
                        facecolor=STUDY_CMAP(norm(size)),
                        edgecolor='#1A5276', linewidth=0.8)
        legend_ax.add_patch(circle)
        legend_ax.text(x_pos, 0.18, f'{size}', fontsize=6.5, ha='center')
    
    # Income group legend
    income_ax = fig.add_axes([0.62, 0.01, 0.20, 0.09])
    income_ax.set_xlim(0, 1)
    income_ax.set_ylim(0, 1)
    income_ax.axis('off')
    
    income_ax.text(0.5, 0.95, 'World Bank Income Groups',
                   fontsize=7.5, ha='center', va='top', fontweight='bold')
    
    income_legend = [
        ('HIC', COLORS['land_hic'], COLORS['border_hic'], '68%'),
        ('UMIC', COLORS['land_umic'], COLORS['border_umic'], '22%'),
        ('LMIC', COLORS['land_lmic'], COLORS['border_lmic'], '9.5%'),
        ('LIC', COLORS['land_lic'], COLORS['border_lic'], '<0.5%'),
    ]
    
    for i, (code, fill, edge, pct) in enumerate(income_legend):
        x = 0.08 + (i % 2) * 0.45
        y = 0.60 - (i // 2) * 0.35
        income_ax.add_patch(FancyBboxPatch(
            (x, y - 0.08), 0.08, 0.16,
            boxstyle="round,pad=0.02",
            facecolor=fill, edgecolor=edge, linewidth=1.5
        ))
        income_ax.text(x + 0.12, y, f'{code}: {pct}', fontsize=6.5, va='center')
    
    # Caption
    caption = (
        "Top 10 countries account for ~70% of global genomic clinical studies. "
        "High-income to low-income publication ratio: 322:1."
    )
    fig.text(0.5, 0.003, caption, fontsize=8, ha='center', va='bottom',
             style='italic', color='#546E7A')
    
    fig.text(0.98, 0.003, 'Source: WHO 2025 Report',
             fontsize=6, ha='right', va='bottom', color='#90A4AE')
    
    return fig


def create_heim_panel():
    """
    Create HEIM equity panel - Publication-grade analytical figure (Refined Layout v2).
    Strict spacing, alignment, and non-overlap for all render sizes.
    """
    print("Creating HEIM equity panel...")
    
    # Restrained color palette
    HEIM_COLORS = {
        'primary': '#2C3E50',
        'secondary': '#5D6D7E',
        'accent': '#1A5276',
        'bar_research': '#2874A6',
        'bar_burden': '#85929E',
        'alert': '#C0392B',
        'alert_bg': '#FDECEA',
        'border': '#D5D8DC',
        'bg_panel': '#F8F9FA',
        'lic_highlight': '#E74C3C',
    }
    
    from matplotlib.patches import Rectangle
    
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ========================================================================
    # HEADER (with increased vertical padding)
    # ========================================================================
    
    ax.text(0.5, 0.970, 'HEIM: Health Equity Informative Metrics',
            fontsize=16, fontweight='bold', ha='center', va='top',
            color=HEIM_COLORS['primary'])
    
    ax.text(0.5, 0.930, 'Framework for Evaluating Alignment Between Genomic Research Output and Global Disease Burden',
            fontsize=9, ha='center', va='top', color=HEIM_COLORS['secondary'])
    
    ax.axhline(y=0.900, xmin=0.05, xmax=0.95, color=HEIM_COLORS['border'], linewidth=0.8)
    
    # ========================================================================
    # SECTION A: CORE FORMULATION (Left column - fixed containers)
    # ========================================================================
    
    section_header_y = 0.870
    
    ax.text(0.035, section_header_y, 'A', fontsize=12, fontweight='bold', 
            color=HEIM_COLORS['primary'], va='center')
    ax.text(0.060, section_header_y, 'Core Formulation', fontsize=10, fontweight='bold', 
            color=HEIM_COLORS['primary'], va='center')
    
    ax.text(0.035, 0.810, 'Equity Alignment Score (EAS)',
            fontsize=10, fontweight='bold', ha='left', va='center',
            color=HEIM_COLORS['accent'])
    
    formula_box = FancyBboxPatch((0.035, 0.745), 0.26, 0.045,
                                  boxstyle='round,pad=0.012',
                                  facecolor=HEIM_COLORS['bg_panel'],
                                  edgecolor=HEIM_COLORS['border'],
                                  linewidth=0.8)
    ax.add_patch(formula_box)
    ax.text(0.165, 0.767, 'EAS = f ( Gap, Burden, Capacity )',
            fontsize=10, ha='center', va='center',
            family='monospace', color=HEIM_COLORS['primary'])
    
    components = [
        ('Gap Severity', 'Mismatch between research output and disease burden'),
        ('Burden Miss', 'High-burden diseases with little or no genomic research coverage'),
        ('Capacity Penalty', 'Underutilization of research capacity in underrepresented settings'),
    ]
    bullet_colors = ['#3498DB', '#F39C12', '#27AE60']
    
    def_start_y = 0.685
    def_spacing = 0.080
    
    for i, (term, definition) in enumerate(components):
        block_y = def_start_y - i * def_spacing
        
        ax.add_patch(Rectangle((0.035, block_y - 0.008), 0.012, 0.016,
                               facecolor=bullet_colors[i], edgecolor='none'))
        
        ax.text(0.058, block_y, term, fontsize=9, fontweight='bold',
               color=HEIM_COLORS['primary'], va='center')
        
        ax.text(0.058, block_y - 0.030, definition, fontsize=7.5,
               color=HEIM_COLORS['secondary'], va='center')
    
    # ========================================================================
    # SECTION B: INCOME GROUP DISTRIBUTION (Center - precise bar positioning)
    # ========================================================================
    
    ax.text(0.335, section_header_y, 'B', fontsize=12, fontweight='bold', 
            color=HEIM_COLORS['primary'], va='center')
    ax.text(0.360, section_header_y, 'Distribution of Genomic Clinical Studies by Income Group',
            fontsize=10, fontweight='bold', color=HEIM_COLORS['primary'], va='center')
    
    categories = ['HIC', 'UMIC', 'LMIC', 'LIC']
    research_pct = [68.0, 22.0, 9.5, 0.5]
    burden_pct = [18.0, 32.0, 35.0, 15.0]
    
    chart_left = 0.38
    chart_bottom = 0.34
    chart_width = 0.30
    chart_height = 0.42
    bar_group_width = chart_width / 4
    bar_width = bar_group_width * 0.38
    bar_gap = bar_group_width * 0.06
    max_val = 80
    
    for grid_val in [0, 20, 40, 60, 80]:
        grid_y = chart_bottom + (grid_val / max_val) * chart_height
        ax.axhline(y=grid_y, xmin=0.355, xmax=0.70, color='#EAECEE', linewidth=0.5, zorder=0)
        ax.text(0.350, grid_y, f'{grid_val}%', fontsize=7, ha='right', va='center',
               color=HEIM_COLORS['secondary'])
    
    ax.text(0.330, chart_bottom + chart_height/2, 'Share (%)',
            fontsize=8, ha='center', va='center', rotation=90, color=HEIM_COLORS['secondary'])
    
    for i, (cat, res, bur) in enumerate(zip(categories, research_pct, burden_pct)):
        group_center = chart_left + (i + 0.5) * bar_group_width
        
        res_x = group_center - bar_width - bar_gap/2
        res_height = (res / max_val) * chart_height
        ax.add_patch(Rectangle((res_x, chart_bottom), bar_width, res_height,
                               facecolor=HEIM_COLORS['bar_research'], edgecolor='none', alpha=0.9))
        
        bur_x = group_center + bar_gap/2
        bur_height = (bur / max_val) * chart_height
        ax.add_patch(Rectangle((bur_x, chart_bottom), bar_width, bur_height,
                               facecolor=HEIM_COLORS['bar_burden'], edgecolor='none', alpha=0.7))
        
        label_y = chart_bottom + res_height + 0.015
        label_text = f'{res:.0f}%' if res >= 1 else f'{res:.1f}%'
        label_color = HEIM_COLORS['lic_highlight'] if cat == 'LIC' else HEIM_COLORS['bar_research']
        ax.text(res_x + bar_width/2, label_y, label_text,
               fontsize=8, fontweight='bold', ha='center', va='bottom', color=label_color)
        
        cat_color = HEIM_COLORS['lic_highlight'] if cat == 'LIC' else HEIM_COLORS['primary']
        cat_weight = 'bold' if cat in ['HIC', 'LIC'] else 'normal'
        ax.text(group_center, chart_bottom - 0.040, cat,
               fontsize=9, fontweight=cat_weight, ha='center', va='top', color=cat_color)
    
    legend_y = 0.820
    ax.add_patch(Rectangle((0.40, legend_y - 0.007), 0.022, 0.014,
                           facecolor=HEIM_COLORS['bar_research'], edgecolor='none'))
    ax.text(0.430, legend_y, 'Genomic Research Output', fontsize=7.5, va='center',
           color=HEIM_COLORS['primary'])
    
    ax.add_patch(Rectangle((0.56, legend_y - 0.007), 0.022, 0.014,
                           facecolor=HEIM_COLORS['bar_burden'], edgecolor='none', alpha=0.7))
    ax.text(0.590, legend_y, 'Global Disease Burden', fontsize=7.5, va='center',
           color=HEIM_COLORS['primary'])
    
    callout_x = 0.72
    callout_y = 0.48
    callout_width = 0.085
    callout_height = 0.095
    
    callout_box = FancyBboxPatch((callout_x - callout_width/2, callout_y - callout_height/2), 
                                  callout_width, callout_height,
                                  boxstyle='round,pad=0.010',
                                  facecolor='white',
                                  edgecolor=HEIM_COLORS['lic_highlight'],
                                  linewidth=1.2,
                                  alpha=0.97)
    ax.add_patch(callout_box)
    
    ax.text(callout_x, callout_y + 0.025, 'LIC: 0.5%', fontsize=8, fontweight='bold',
           ha='center', va='center', color=HEIM_COLORS['lic_highlight'])
    ax.text(callout_x, callout_y, 'of research', fontsize=7, ha='center', va='center',
           color=HEIM_COLORS['lic_highlight'])
    ax.text(callout_x, callout_y - 0.025, 'vs 15% burden', fontsize=7, ha='center', va='center',
           color=HEIM_COLORS['lic_highlight'])
    
    lic_group_center = chart_left + 3.5 * bar_group_width
    ax.annotate('', xy=(lic_group_center + 0.01, chart_bottom + 0.10),
               xytext=(callout_x - callout_width/2 - 0.005, callout_y),
               arrowprops=dict(arrowstyle='->', color=HEIM_COLORS['lic_highlight'],
                              lw=1.2, connectionstyle='arc3,rad=0.15'))
    
    # ========================================================================
    # SECTION C: KEY FINDING (Right - single grouped container)
    # ========================================================================
    
    ax.text(0.795, section_header_y, 'C', fontsize=12, fontweight='bold', 
            color=HEIM_COLORS['primary'], va='center')
    ax.text(0.820, section_header_y, 'Key Finding', fontsize=10, fontweight='bold',
            color=HEIM_COLORS['primary'], va='center')
    
    kf_left = 0.795
    kf_bottom = 0.42
    kf_width = 0.175
    kf_height = 0.40
    
    kf_box = FancyBboxPatch((kf_left, kf_bottom), kf_width, kf_height,
                             boxstyle='round,pad=0.018',
                             facecolor=HEIM_COLORS['alert_bg'],
                             edgecolor=HEIM_COLORS['alert'],
                             linewidth=1.5)
    ax.add_patch(kf_box)
    
    kf_center_x = kf_left + kf_width / 2
    
    ax.text(kf_center_x, kf_bottom + kf_height * 0.75, '322:1',
            fontsize=32, fontweight='bold', ha='center', va='center',
            color=HEIM_COLORS['alert'])
    
    ax.text(kf_center_x, kf_bottom + kf_height * 0.50, 'HIC to LIC',
            fontsize=10, ha='center', va='center', color=HEIM_COLORS['primary'])
    ax.text(kf_center_x, kf_bottom + kf_height * 0.38, 'Publication Ratio',
            fontsize=10, ha='center', va='center', color=HEIM_COLORS['primary'])
    
    ax.text(kf_center_x, kf_bottom + kf_height * 0.20, 'High-income countries',
            fontsize=7.5, ha='center', va='center', color=HEIM_COLORS['secondary'], style='italic')
    ax.text(kf_center_x, kf_bottom + kf_height * 0.10, 'produce 322× more',
            fontsize=7.5, ha='center', va='center', color=HEIM_COLORS['secondary'], style='italic')
    ax.text(kf_center_x, kf_bottom + kf_height * 0.00 + 0.02, 'genomic studies',
            fontsize=7.5, ha='center', va='center', color=HEIM_COLORS['secondary'], style='italic')
    
    # ========================================================================
    # SECTION D: SUMMARY STATISTICS (Bottom - equal columns, baseline aligned)
    # ========================================================================
    
    ax.axhline(y=0.255, xmin=0.035, xmax=0.965, color=HEIM_COLORS['border'], linewidth=0.8)
    
    ax.text(0.035, 0.220, 'D', fontsize=12, fontweight='bold', color=HEIM_COLORS['primary'], va='center')
    ax.text(0.060, 0.220, 'Summary: WHO Genomic Clinical Studies (1990–2024)',
            fontsize=10, fontweight='bold', color=HEIM_COLORS['primary'], va='center')
    
    stats = [
        ('6,513', 'Total Studies'),
        ('70%', 'Top 10 Countries'),
        ('68%', 'High-Income Countries'),
        ('<0.5%', 'Low-Income Countries'),
    ]
    
    n_cols = len(stats)
    col_width = 0.90 / n_cols
    col_start = 0.06
    
    stat_value_y = 0.150
    stat_label_y = 0.095
    
    for i, (value, label) in enumerate(stats):
        col_center = col_start + (i + 0.5) * col_width
        
        color = HEIM_COLORS['alert'] if '<0.5%' in value else HEIM_COLORS['accent']
        ax.text(col_center, stat_value_y, value,
               fontsize=20, fontweight='bold', ha='center', va='center',
               color=color)
        
        ax.text(col_center, stat_label_y, label,
               fontsize=8, ha='center', va='center', color=HEIM_COLORS['secondary'])
    
    # ========================================================================
    # FOOTER (Two distinct lines with fixed positions)
    # ========================================================================
    
    footer_line1_y = 0.038
    footer_line2_y = 0.015
    
    ax.text(0.035, footer_line1_y,
            'Data source: World Health Organization (2025). Human genomics technologies in clinical studies – '
            'the research landscape: report on the 1990–2024 period.',
            fontsize=7, ha='left', va='center', color=HEIM_COLORS['secondary'])
    
    ax.text(0.035, footer_line2_y,
            'Framework note: HEIM (Health Equity Informative Metrics) methodology — manuscript currently under revision.',
            fontsize=7, ha='left', va='center', color=HEIM_COLORS['secondary'], style='italic')
    
    return fig


def main():
    print("=" * 60)
    print("WHO GENOMIC STUDIES - PUBLICATION QUALITY GRAPHICS")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    
    fig_map = create_publication_map()
    map_png = os.path.join(output_dir, 'who_genomic_map_publication.png')
    map_pdf = os.path.join(output_dir, 'who_genomic_map_publication.pdf')
    fig_map.savefig(map_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_map.savefig(map_pdf, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {map_png}")
    print(f"  Saved: {map_pdf}")
    plt.close(fig_map)
    
    fig_heim = create_heim_panel()
    heim_png = os.path.join(output_dir, 'heim_equity_panel.png')
    heim_pdf = os.path.join(output_dir, 'heim_equity_panel.pdf')
    fig_heim.savefig(heim_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_heim.savefig(heim_pdf, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {heim_png}")
    print(f"  Saved: {heim_pdf}")
    plt.close(fig_heim)
    
    print("\nComplete. Key findings:")
    for rank, (country, data) in enumerate(TOP_10_DATA.items(), 1):
        print(f"  {rank:2}. {country:18} {data['studies']:,} ({data['pct']}%) [{data['income']}]")


if __name__ == "__main__":
    main()