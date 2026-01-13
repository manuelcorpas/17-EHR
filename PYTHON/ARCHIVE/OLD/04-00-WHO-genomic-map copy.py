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
    
    fig.text(0.98, 0.003, 'Source: WHO 2024 Report',
             fontsize=6, ha='right', va='bottom', color='#90A4AE')
    
    return fig


def create_heim_panel():
    """
    Create HEIM equity panel - Publication-grade analytical figure.
    Minimalist, metric-driven visualization for policy and journal audiences.
    """
    print("Creating HEIM equity panel...")
    
    # Restrained color palette (perceptually uniform)
    HEIM_COLORS = {
        'primary': '#2C3E50',      # Dark slate for text
        'secondary': '#5D6D7E',    # Medium gray for secondary text
        'accent': '#1A5276',       # Deep blue for emphasis
        'bar_research': '#2874A6', # Blue for research bars
        'bar_burden': '#85929E',   # Gray for burden bars
        'alert': '#C0392B',        # Restrained red for key finding
        'alert_bg': '#FDECEA',     # Very light red background
        'border': '#D5D8DC',       # Light border
        'bg_panel': '#F8F9FA',     # Panel background
        'lic_highlight': '#E74C3C', # LIC emphasis
    }
    
    from matplotlib.patches import Rectangle
    
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ========================================================================
    # HEADER
    # ========================================================================
    ax.text(0.5, 0.96, 'HEIM: Health Equity Informative Metrics',
            fontsize=16, fontweight='bold', ha='center', va='top',
            color=HEIM_COLORS['primary'])
    
    ax.text(0.5, 0.915, 'Framework for Evaluating Alignment Between Genomic Research Output and Global Disease Burden',
            fontsize=9, ha='center', va='top', color=HEIM_COLORS['secondary'])
    
    # Subtle separator
    ax.axhline(y=0.885, xmin=0.08, xmax=0.92, color=HEIM_COLORS['border'], linewidth=0.8)
    
    # ========================================================================
    # SECTION A: CORE FORMULATION (Left)
    # ========================================================================
    
    # Panel A title
    ax.text(0.05, 0.86, 'A', fontsize=12, fontweight='bold', color=HEIM_COLORS['primary'])
    ax.text(0.08, 0.86, 'Core Formulation', fontsize=10, fontweight='bold', 
            color=HEIM_COLORS['primary'], va='top')
    
    # EAS formula - prominent display
    formula_y = 0.78
    ax.text(0.18, formula_y, 'Equity Alignment Score (EAS)',
            fontsize=10, fontweight='bold', ha='left', va='center',
            color=HEIM_COLORS['accent'])
    
    ax.text(0.18, formula_y - 0.04, 'EAS = f ( Gap, Burden, Capacity )',
            fontsize=11, ha='left', va='center',
            family='monospace', color=HEIM_COLORS['primary'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=HEIM_COLORS['bg_panel'],
                     edgecolor=HEIM_COLORS['border'], linewidth=0.8))
    
    # Component definitions - clean list format
    components = [
        ('Gap Severity', 'Mismatch between research output and disease burden'),
        ('Burden Miss', 'High-burden diseases with little or no genomic research coverage'),
        ('Capacity Penalty', 'Underutilization of research capacity in underrepresented settings'),
    ]
    
    # Simple bullet indicators
    bullet_colors = ['#3498DB', '#F39C12', '#27AE60']
    
    start_y = 0.62
    for i, (term, definition) in enumerate(components):
        y = start_y - i * 0.065
        
        # Small square bullet
        ax.add_patch(Rectangle((0.06, y - 0.008), 0.012, 0.016,
                               facecolor=bullet_colors[i], edgecolor='none'))
        
        # Term (bold) and definition
        ax.text(0.085, y, term, fontsize=9, fontweight='bold',
               color=HEIM_COLORS['primary'], va='center')
        ax.text(0.085, y - 0.025, definition, fontsize=8,
               color=HEIM_COLORS['secondary'], va='center')
    
    # ========================================================================
    # SECTION B: INCOME GROUP DISTRIBUTION (Center)
    # ========================================================================
    
    ax.text(0.42, 0.86, 'B', fontsize=12, fontweight='bold', color=HEIM_COLORS['primary'])
    ax.text(0.45, 0.86, 'Distribution of Genomic Clinical Studies by Income Group',
            fontsize=10, fontweight='bold', color=HEIM_COLORS['primary'], va='top')
    
    # Data
    categories = ['HIC', 'UMIC', 'LMIC', 'LIC']
    research_pct = [68.0, 22.0, 9.5, 0.5]  # Share of genomic studies
    burden_pct = [18.0, 32.0, 35.0, 15.0]   # Approximate share of global disease burden
    
    # Bar chart parameters
    bar_width = 0.055
    bar_spacing = 0.09
    chart_left = 0.48
    chart_bottom = 0.42
    chart_height = 0.32
    
    # Y-axis scale
    max_val = 80
    
    # Draw gridlines first
    for grid_val in [0, 20, 40, 60, 80]:
        grid_y = chart_bottom + (grid_val / max_val) * chart_height
        ax.axhline(y=grid_y, xmin=0.44, xmax=0.87, color='#EAECEE', linewidth=0.5, zorder=0)
        ax.text(0.435, grid_y, f'{grid_val}%', fontsize=7, ha='right', va='center',
               color=HEIM_COLORS['secondary'])
    
    # Y-axis label
    ax.text(0.41, chart_bottom + chart_height/2, 'Share (%)',
            fontsize=8, ha='center', va='center', rotation=90, color=HEIM_COLORS['secondary'])
    
    # Draw bars
    for i, (cat, res, bur) in enumerate(zip(categories, research_pct, burden_pct)):
        x_center = chart_left + i * bar_spacing
        
        # Research bar (blue)
        res_height = (res / max_val) * chart_height
        ax.add_patch(Rectangle((x_center - bar_width/2, chart_bottom),
                               bar_width * 0.9, res_height,
                               facecolor=HEIM_COLORS['bar_research'],
                               edgecolor='none', alpha=0.9))
        
        # Burden bar (gray)
        bur_height = (bur / max_val) * chart_height
        ax.add_patch(Rectangle((x_center + bar_width/2 * 0.1, chart_bottom),
                               bar_width * 0.9, bur_height,
                               facecolor=HEIM_COLORS['bar_burden'],
                               edgecolor='none', alpha=0.7))
        
        # Value labels on research bars
        if res >= 5:
            ax.text(x_center - bar_width/4, chart_bottom + res_height - 0.02,
                   f'{res:.0f}%' if res >= 1 else f'{res:.1f}%',
                   fontsize=7, fontweight='bold', ha='center', va='top', color='white')
        else:
            # For small values, label above
            ax.text(x_center - bar_width/4, chart_bottom + res_height + 0.01,
                   f'{res:.1f}%', fontsize=7, fontweight='bold', ha='center', va='bottom',
                   color=HEIM_COLORS['lic_highlight'] if cat == 'LIC' else HEIM_COLORS['bar_research'])
        
        # Category labels
        ax.text(x_center, chart_bottom - 0.025, cat,
               fontsize=9, fontweight='bold' if cat in ['HIC', 'LIC'] else 'normal',
               ha='center', va='top',
               color=HEIM_COLORS['lic_highlight'] if cat == 'LIC' else HEIM_COLORS['primary'])
    
    # Legend for bars
    legend_y = 0.80
    ax.add_patch(Rectangle((0.48, legend_y), 0.025, 0.015,
                           facecolor=HEIM_COLORS['bar_research'], edgecolor='none'))
    ax.text(0.515, legend_y + 0.0075, 'Genomic Research Output', fontsize=8, va='center',
           color=HEIM_COLORS['primary'])
    
    ax.add_patch(Rectangle((0.66, legend_y), 0.025, 0.015,
                           facecolor=HEIM_COLORS['bar_burden'], edgecolor='none', alpha=0.7))
    ax.text(0.695, legend_y + 0.0075, 'Global Disease Burden', fontsize=8, va='center',
           color=HEIM_COLORS['primary'])
    
    # Annotation for LIC - subtle arrow
    lic_x = chart_left + 3 * bar_spacing
    ax.annotate('', xy=(lic_x - 0.01, chart_bottom + 0.06),
               xytext=(lic_x - 0.04, chart_bottom + 0.12),
               arrowprops=dict(arrowstyle='->', color=HEIM_COLORS['lic_highlight'],
                              lw=1.2, connectionstyle='arc3,rad=-0.2'))
    ax.text(lic_x - 0.055, chart_bottom + 0.135, 'LIC: 0.5%\nof research\nvs 15% of\nburden',
           fontsize=7, ha='center', va='bottom', color=HEIM_COLORS['lic_highlight'],
           linespacing=1.1)
    
    # ========================================================================
    # SECTION C: KEY FINDING (Right)
    # ========================================================================
    
    ax.text(0.77, 0.86, 'C', fontsize=12, fontweight='bold', color=HEIM_COLORS['primary'])
    ax.text(0.80, 0.86, 'Key Finding', fontsize=10, fontweight='bold',
            color=HEIM_COLORS['primary'], va='top')
    
    # Callout panel - restrained
    callout_box = FancyBboxPatch((0.77, 0.52), 0.20, 0.28,
                                  boxstyle='round,pad=0.015',
                                  facecolor=HEIM_COLORS['alert_bg'],
                                  edgecolor=HEIM_COLORS['alert'],
                                  linewidth=1.5)
    ax.add_patch(callout_box)
    
    # The ratio - large but restrained
    ax.text(0.87, 0.70, '322:1',
            fontsize=32, fontweight='bold', ha='center', va='center',
            color=HEIM_COLORS['alert'])
    
    ax.text(0.87, 0.60, 'HIC to LIC\nPublication Ratio',
            fontsize=9, ha='center', va='center', color=HEIM_COLORS['primary'],
            linespacing=1.3)
    
    ax.text(0.87, 0.545, 'High-income countries produce 322×\nmore genomic studies than\nlow-income countries',
            fontsize=7.5, ha='center', va='top', color=HEIM_COLORS['secondary'],
            linespacing=1.2, style='italic')
    
    # ========================================================================
    # SECTION D: SUMMARY STATISTICS (Bottom)
    # ========================================================================
    
    ax.axhline(y=0.28, xmin=0.05, xmax=0.95, color=HEIM_COLORS['border'], linewidth=0.8)
    
    ax.text(0.05, 0.25, 'D', fontsize=12, fontweight='bold', color=HEIM_COLORS['primary'])
    ax.text(0.08, 0.25, 'Summary: WHO Genomic Clinical Studies (1990–2024)',
            fontsize=10, fontweight='bold', color=HEIM_COLORS['primary'], va='top')
    
    # Statistics in clean grid
    stats = [
        ('6,513', 'Total Studies'),
        ('70%', 'Top 10 Countries'),
        ('68%', 'High-Income Countries'),
        ('<0.5%', 'Low-Income Countries'),
    ]
    
    stat_y = 0.155
    for i, (value, label) in enumerate(stats):
        x = 0.12 + i * 0.22
        
        # Value
        color = HEIM_COLORS['alert'] if '<0.5%' in value else HEIM_COLORS['accent']
        ax.text(x, stat_y + 0.04, value,
               fontsize=18, fontweight='bold', ha='center', va='center',
               color=color)
        
        # Label
        ax.text(x, stat_y - 0.02, label,
               fontsize=8, ha='center', va='center', color=HEIM_COLORS['secondary'])
    
    # ========================================================================
    # SOURCE ATTRIBUTION (Footer)
    # ========================================================================
    
    footer_y = 0.035
    
    ax.text(0.05, footer_y,
            'Data source: World Health Organization (2025). Human genomics technologies in clinical studies – '
            'the research landscape: report on the 1990–2024 period.',
            fontsize=7, ha='left', va='center', color=HEIM_COLORS['secondary'])
    
    ax.text(0.05, footer_y - 0.022,
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