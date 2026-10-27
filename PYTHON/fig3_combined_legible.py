#!/usr/bin/env python3
"""Fig 3 Translation Dimension, LEGIBILITY-FIXED composite (Reviewer #1/#2).
Top panel (a): enlarged world map of trial sites (local Natural Earth shapefile).
Bottom panels (b): trial-site distribution + intensity gap, enlarged fonts.
Same data/logic as 06-heim-publication-figures.py; only rendering enlarged.
Output: docs/figures/Fig_03_Translation_Dimension_LEGIBLE.png/.pdf
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
from matplotlib.gridspec import GridSpec
import geopandas as gpd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

def main():
    base = Path(__file__).resolve().parent.parent
    data = json.load(open(base / "docs" / "data" / "clinical_trials.json"))
    shp = base / "DATA" / "naturalearth" / "ne_110m_admin_0_countries.shp"
    out = base / "docs" / "figures" / "Fig_03_Translation_Dimension_LEGIBLE.png"

    geo = data.get('geographic', {}); gs = data.get('globalSouthAnalysis', {})
    fig = plt.figure(figsize=(16, 15))
    grid = GridSpec(2, 2, height_ratios=[2.25, 1.0], hspace=0.16, wspace=0.16)
    ax_map = fig.add_subplot(grid[0, :])
    ax_b1 = fig.add_subplot(grid[1, 0]); ax_b2 = fig.add_subplot(grid[1, 1])

    # ---- Panel (a): world map ----
    world = gpd.read_file(shp)
    name_col = 'NAME' if 'NAME' in world.columns else ('name' if 'name' in world.columns else 'ADMIN')
    cmap = {'United States': 'United States of America', 'Turkey (Türkiye)': 'Turkey'}
    top_countries = geo.get('topCountries', [])
    country_sites = {cmap.get(c['name'], c['name']): c['sites'] for c in top_countries}
    world['sites'] = world[name_col].map(country_sites).fillna(0)
    world['log_sites'] = np.log10(world['sites'] + 1)
    world['has_trials'] = world['sites'] > 0
    hic_list = ['United States of America','France','Canada','Germany','United Kingdom','Italy','Spain',
                'South Korea','Belgium','Netherlands','Denmark','Australia','Japan','Switzerland']
    world['income'] = world[name_col].apply(lambda x: 'HIC' if x in hic_list else 'LMIC')
    world[~world['has_trials']].plot(ax=ax_map, color='#F5F5F5', edgecolor='white', linewidth=0.3)
    wt = world[world['has_trials']].copy()
    wt[wt['income']=='HIC'].plot(ax=ax_map, column='log_sites',
        cmap=LinearSegmentedColormap.from_list('h',['#B3D9FF','#0066CC']), edgecolor='black', linewidth=0.5)
    wt[wt['income']=='LMIC'].plot(ax=ax_map, column='log_sites',
        cmap=LinearSegmentedColormap.from_list('l',['#FFCCCC','#CC0000']), edgecolor='black', linewidth=0.5)
    sm = ScalarMappable(cmap='YlOrRd', norm=Normalize(vmin=0, vmax=6)); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_map, shrink=0.6, aspect=22, pad=0.01)
    cbar.set_label('Trial Sites (log10 scale)', fontsize=14); cbar.set_ticks([0,2,4,6])
    cbar.set_ticklabels(['1','100','10K','1M']); cbar.ax.tick_params(labelsize=12)
    hic_pct = geo.get('hicPct', 71.8); total = geo.get('hicSites',0)+geo.get('lmicSites',0)
    ax_map.set_title(f'Global Distribution of Clinical Trial Sites  ({total:,} sites; HIC {hic_pct:.0f}% vs LMIC {100-hic_pct:.0f}%)',
                     fontsize=17, fontweight='bold')
    ax_map.legend(handles=[mpatches.Patch(color='#3498DB', label=f'High-Income ({hic_pct:.0f}%)'),
                           mpatches.Patch(color='#E74C3C', label=f'Low/Middle-Income ({100-hic_pct:.0f}%)'),
                           mpatches.Patch(color='#F5F5F5', label='No data')],
                  loc='lower left', fontsize=13, framealpha=0.9)
    top5 = sorted(country_sites.items(), key=lambda x: x[1], reverse=True)[:5]
    ax_map.text(0.02, 0.98, "Top 5 countries:\n"+"\n".join(f"{c[0][:14]}: {c[1]:,}" for c in top5),
                transform=ax_map.transAxes, fontsize=13, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, pad=0.6))
    ax_map.text(0.98, 0.04, f"HIC:LMIC Ratio = {geo.get('hicLmicRatio',2.5):.1f}×", transform=ax_map.transAxes,
                fontsize=15, ha='right', fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, pad=0.6))
    ax_map.set_axis_off()

    # ---- Panel (b): bars ----
    hic_sites = geo.get('hicSites',552952); lmic_sites = geo.get('lmicSites',217226)
    colors = ['#3498DB', '#E74C3C']
    b1 = ax_b1.bar(['High-Income\nCountries','Low/Middle-Income\nCountries'], [hic_sites,lmic_sites], color=colors, edgecolor='white')
    ax_b1.set_ylabel('Number of Trial Sites', fontsize=14)
    ax_b1.set_title('Clinical Trial Site Distribution', fontsize=15, fontweight='bold')
    for bar,c in zip(b1,[hic_sites,lmic_sites]):
        ax_b1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10000, f'{c:,}\n({c/(hic_sites+lmic_sites)*100:.1f}%)', ha='center', fontsize=13)
    ax_b1.text(0.5, 0.85, f'Ratio: {hic_sites/lmic_sites:.1f}×', transform=ax_b1.transAxes, ha='center',
               fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax_b1.tick_params(labelsize=12); ax_b1.spines[['top','right']].set_visible(False)
    ax_b1.set_ylim(0, max(hic_sites,lmic_sites)*1.28)

    hic_i = gs.get('nonGsIntensity',1167.4); gs_i = gs.get('gsIntensity',485.9); gap = gs.get('intensityGap',2.4)
    b2 = ax_b2.bar(['High-Income\nCountries','Global South'], [hic_i,gs_i], color=colors, edgecolor='white')
    ax_b2.set_ylabel('Trials per Million DALYs', fontsize=14)
    ax_b2.set_title('Trial Intensity by Region\n(Global South diseases)', fontsize=15, fontweight='bold')
    for bar,v in zip(b2,[hic_i,gs_i]):
        ax_b2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f'{v:.0f}', ha='center', fontsize=13)
    ax_b2.annotate('', xy=(1,gs_i), xytext=(1,hic_i), arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax_b2.text(1.15, (hic_i+gs_i)/2, f'{gap}×\ngap', ha='left', va='center', fontsize=14, fontweight='bold')
    ax_b2.tick_params(labelsize=12); ax_b2.spines[['top','right']].set_visible(False)
    ax_b2.set_ylim(0, max(hic_i,gs_i)*1.22)

    # panel labels
    ax_map.text(-0.02, 1.04, 'a', transform=ax_map.transAxes, fontsize=26, fontweight='bold', va='top')
    ax_b1.text(-0.18, 1.08, 'b', transform=ax_b1.transAxes, fontsize=26, fontweight='bold', va='top')

    plt.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(out.with_suffix('.pdf'), facecolor='white', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

if __name__ == '__main__':
    main()
