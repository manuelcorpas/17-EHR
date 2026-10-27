#!/usr/bin/env python3
"""
Fig 3a world map, LEGIBILITY-FIXED (Reviewer #1/#2).
Self-contained: reads clinical_trials.json + a LOCAL Natural Earth shapefile
(Studio is offline). Same choropleth logic as 06-heim-publication-figures.py's
create_figure_3a_world_map; only enlarges fonts (inset 8->12, colorbar 10->13,
legend 9->12, title 14->17, base 10->13) and the canvas. No data changed.
Output: docs/figures/Fig_03a_Trial_Map_LEGIBLE.png/.pdf
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

def main():
    base = Path(__file__).resolve().parent.parent
    data = json.load(open(base / "docs" / "data" / "clinical_trials.json"))
    shp = base / "DATA" / "naturalearth" / "ne_110m_admin_0_countries.shp"
    out = base / "docs" / "figures" / "Fig_03a_Trial_Map_LEGIBLE.png"

    world = gpd.read_file(shp)
    name_col = 'NAME' if 'NAME' in world.columns else ('name' if 'name' in world.columns else 'ADMIN')

    geo = data.get('geographic', {})
    top_countries = geo.get('topCountries', [])
    country_mapping = {'United States': 'United States of America',
                       'Turkey (Türkiye)': 'Turkey', 'South Korea': 'South Korea'}
    country_sites = {country_mapping.get(c['name'], c['name']): c['sites'] for c in top_countries}
    print(f"  countries in data: {len(country_sites)}")

    world['sites'] = world[name_col].map(country_sites).fillna(0)
    world['log_sites'] = np.log10(world['sites'] + 1)
    world['has_trials'] = world['sites'] > 0
    hic_list = ['United States of America','France','Canada','Germany','United Kingdom','Italy',
                'Spain','South Korea','Belgium','Netherlands','Denmark','Australia','Japan','Switzerland']
    world['income'] = world[name_col].apply(lambda x: 'HIC' if x in hic_list else 'LMIC')

    fig, ax = plt.subplots(figsize=(16, 9))
    world[~world['has_trials']].plot(ax=ax, color='#F5F5F5', edgecolor='white', linewidth=0.3)
    wt = world[world['has_trials']].copy()
    cmap_hic = LinearSegmentedColormap.from_list('hic', ['#B3D9FF', '#0066CC'])
    cmap_lmic = LinearSegmentedColormap.from_list('lmic', ['#FFCCCC', '#CC0000'])
    wt[wt['income'] == 'HIC'].plot(ax=ax, column='log_sites', cmap=cmap_hic, edgecolor='black', linewidth=0.5)
    wt[wt['income'] == 'LMIC'].plot(ax=ax, column='log_sites', cmap=cmap_lmic, edgecolor='black', linewidth=0.5)

    sm = ScalarMappable(cmap='YlOrRd', norm=Normalize(vmin=0, vmax=6)); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Trial Sites (log10 scale)', fontsize=14)
    cbar.set_ticks([0, 2, 4, 6]); cbar.set_ticklabels(['1', '100', '10K', '1M'])
    cbar.ax.tick_params(labelsize=12)

    total = geo.get('hicSites', 0) + geo.get('lmicSites', 0)
    hic_pct = geo.get('hicPct', 71.8)
    ax.set_title(f'Global Distribution of Clinical Trial Sites\n({total:,} sites; HIC {hic_pct:.0f}% vs LMIC {100-hic_pct:.0f}%)',
                 fontsize=18, fontweight='bold')

    hic_patch = mpatches.Patch(color='#3498DB', label=f'High-Income ({hic_pct:.0f}%)')
    lmic_patch = mpatches.Patch(color='#E74C3C', label=f'Low/Middle-Income ({100-hic_pct:.0f}%)')
    gray_patch = mpatches.Patch(color='#F5F5F5', label='No data')
    ax.legend(handles=[hic_patch, lmic_patch, gray_patch], loc='lower left', fontsize=13, framealpha=0.9)

    top_5 = sorted(country_sites.items(), key=lambda x: x[1], reverse=True)[:5]
    top_text = "Top 5 countries:\n" + "\n".join([f"{c[0][:14]}: {c[1]:,}" for c in top_5])
    ax.text(0.02, 0.98, top_text, transform=ax.transAxes, fontsize=13, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, pad=0.6))
    ratio = geo.get('hicLmicRatio', 2.5)
    ax.text(0.98, 0.04, f'HIC:LMIC Ratio = {ratio:.1f}×', transform=ax.transAxes, fontsize=15,
            ha='right', fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, pad=0.6))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(out.with_suffix('.pdf'), facecolor='white', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

if __name__ == '__main__':
    main()
