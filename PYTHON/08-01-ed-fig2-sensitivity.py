"""Extended Data Fig 2 (NCOMMS-26-040266-T round 2): ranking robustness.
a) without the +10 Global South penalty; b) two-dimension fallback on complete cases;
c) published vs corrected ranks (2026-10-25 category correction).
Reads ANALYSIS/03-11-NATCOMMS-R2-SENSITIVITY outputs and docs/data; writes ANALYSIS/08-NATCOMMS-R2-FIGURES.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
SENS = BASE / 'ANALYSIS' / '03-11-NATCOMMS-R2-SENSITIVITY'
OUT = BASE / 'ANALYSIS' / '08-NATCOMMS-R2-FIGURES'
BLUE, SKY, VERM, GREY = '#0072B2', '#56B4E9', '#D55E00', '#9E9E9E'


def _rank(x):
    return pd.Series(x).rank(ascending=False, method='min').values


def compute():
    sc = pd.read_csv(SENS / '02-r2-sensitivity-scores.csv')
    w = json.load(open(BASE / 'docs' / 'data' / 'integrated.json'))['weights']
    sc['r_base'] = _rank(sc.u_base); sc['r_nopen'] = _rank(sc.u_nopen_fixedw)
    cc = sc[sc.has_ct].copy()
    g, s = sc.gap, sc.sii  # normalisation ranges as in the scoring (non-injury frame matches report run)
    full = json.load(open(BASE / 'docs' / 'data' / 'integrated.json'))['diseases']
    allg = [d['gap_score'] for d in full if d['gap_score'] is not None]; alls = [d['sii'] for d in full if d['sii'] is not None]
    n = lambda v, a: (v - min(a)) / (max(a) - min(a))
    cc['u_dk'] = 100 * (w['discovery'] * n(cc.gap, allg) + w['knowledge'] * n(cc.sii, alls)) / (w['discovery'] + w['knowledge'])
    cc['r_full'] = _rank(cc.u_base); cc['r_dk'] = _rank(cc.u_dk)
    corr = pd.read_csv(SENS / '08-supp-table-category-correction.csv').dropna(subset=['rank_published', 'rank_corrected'])
    top = lambda d, a, b: len(set(d.nsmallest(10, a).disease) & set(d.nsmallest(10, b).disease))
    stats = dict(
        rho_nopen=spearmanr(sc.u_base, sc.u_nopen_fixedw).correlation, top10_nopen=top(sc, 'r_base', 'r_nopen'),
        rho_fallback=spearmanr(cc.u_base, cc.u_dk).correlation, top10_fallback=top(cc, 'r_full', 'r_dk'),
        rho_correction=spearmanr(corr.unified_score_published, corr.unified_score_corrected).correlation,
        top10_correction=top(corr, 'rank_published', 'rank_corrected'), n_all=len(sc), n_cc=len(cc))
    return dict(sc=sc, cc=cc, corr=corr, stats=stats)


def _panel(ax, x, y, colors, title, xl, yl, note):
    m = max(np.max(x), np.max(y)) + 3
    ax.plot([1, m], [1, m], color='0.6', lw=1, ls='--', zorder=0)
    ax.scatter(x, y, c=colors, s=38, edgecolor='white', linewidth=0.4)
    ax.set_xlim(m, 0); ax.set_ylim(m, 0)
    ax.set_title(title, fontsize=15, fontweight='bold'); ax.set_xlabel(xl, fontsize=14); ax.set_ylabel(yl, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(0.97, 0.03, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.6'))
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)


def main():
    d = compute(); s = d['stats']; sc, cc, corr = d['sc'], d['cc'], d['corr']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2))
    gs_col = np.where(sc.gsp == True, VERM, GREY)
    _panel(axes[0], sc.r_base, sc.r_nopen, gs_col, 'Without +10 Global South penalty',
           'Rank, main analysis', 'Rank, penalty removed',
           f"Spearman ρ = {s['rho_nopen']:.3f}\nTop-10 overlap {s['top10_nopen']}/10\nn = {s['n_all']}")
    _panel(axes[1], cc.r_full, cc.r_dk, np.where(cc.gsp == True, VERM, GREY), 'Two-dimension fallback',
           'Rank, three dimensions', 'Rank, Discovery + Knowledge only',
           f"Spearman ρ = {s['rho_fallback']:.3f}\nTop-10 overlap {s['top10_fallback']}/10\nn = {s['n_cc']} complete cases")
    _panel(axes[2], corr.rank_published, corr.rank_corrected, np.where(corr.category_changed, BLUE, GREY),
           'Category correction', 'Rank, previous version', 'Rank, corrected',
           f"Spearman ρ = {s['rho_correction']:.3f}\nTop-10 overlap {s['top10_correction']}/10\nn = {len(corr)}")
    for ax, l in zip(axes, 'abc'):
        ax.text(-0.12, 1.06, l, transform=ax.transAxes, fontsize=22, fontweight='bold')
    from matplotlib.lines import Line2D
    mk = lambda c, t: Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=t)
    axes[0].legend(handles=[mk(VERM, 'Global South priority'), mk(GREY, 'Other')], loc='upper left', fontsize=12, frameon=False)
    axes[2].legend(handles=[mk(BLUE, 'Category corrected'), mk(GREY, 'Unchanged')], loc='upper left', fontsize=12, frameon=False)
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'Extended_Data_Fig_02_Sensitivity_Analysis.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()},
              open(OUT / 'Extended_Data_Fig_02_stats.json', 'w'), indent=2)
    print(s)


if __name__ == '__main__':
    main()
