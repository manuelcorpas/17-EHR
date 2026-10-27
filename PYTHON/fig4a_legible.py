#!/usr/bin/env python3.11
"""
Fig 4a UMAP, LEGIBILITY-FIXED version (Reviewer #1/#2).
Based on generate_fig4a_umap_standalone.py. Changes ONLY rendering:
larger tiered label fonts, adjustText repulsion to de-overlap, bigger
figure + axis/title/legend/colorbar fonts. No data or positions changed
(same RANDOM_SEED, same centroids -> identical scatter).
Output: docs/figures/Fig_04_UMAP_Semantic_Landscape_LEGIBLE.png/.pdf
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata
import h5py
import umap
try:
    from adjustText import adjust_text
    HAVE_ADJUST = True
except ImportError:
    HAVE_ADJUST = False
    print("WARN: adjustText missing; labels may overlap")

RANDOM_SEED = 42

def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "DATA" / "05-SEMANTIC"
    out_png = base / "docs" / "figures" / "Fig_04_UMAP_Semantic_Landscape_LEGIBLE.png"

    with open(data_dir / "heim_sem_metrics.json") as f:
        metrics = json.load(f)

    print("Loading centroids...")
    embeddings_dir = data_dir / "EMBEDDINGS"
    centroids = {}
    for disease_dir in sorted(embeddings_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        h5_files = list(disease_dir.glob("*.h5")) + list(disease_dir.glob("*.hdf5"))
        if not h5_files:
            continue
        try:
            with h5py.File(h5_files[0], 'r') as hf:
                if 'embeddings' in hf:
                    centroids[disease_dir.name] = hf['embeddings'][:].mean(axis=0)
        except Exception as e:
            print(f"  warn {disease_dir.name}: {e}")
    print(f"  {len(centroids)} centroids")

    diseases = [d for d in centroids if d in metrics['metrics']]
    centroid_matrix = np.array([centroids[d] for d in diseases])

    print("UMAP...")
    np.random.seed(RANDOM_SEED)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric='cosine', random_state=RANDOM_SEED)
    emb = reducer.fit_transform(centroid_matrix)

    paper_counts = np.array([metrics['metrics'][d].get('n_papers', 100) for d in diseases])
    sizes = 40 + (paper_counts - paper_counts.min()) / (paper_counts.max() - paper_counts.min() + 1) * 320
    sii_values = np.array([metrics['metrics'][d].get('sii', 0) for d in diseases])
    sii_ranks = rankdata(sii_values, method='average') / len(sii_values)

    # Bigger canvas gives labels room
    fig, ax = plt.subplots(figsize=(15, 11))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=sii_ranks, s=sizes,
                         cmap='RdYlBu_r', vmin=0, vmax=1, alpha=0.75,
                         edgecolors='white', linewidths=0.6)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('SII rank percentile', fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    pct = [0, 0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(pct)
    cbar.set_ticklabels([f'{np.percentile(sii_values, p*100):.4f}' for p in pct])

    ntd_names = {'Leishmaniasis','Schistosomiasis','Onchocerciasis','Dengue','Rabies',
        'Trachoma','Leprosy','African_trypanosomiasis','Chagas_disease','Yellow_fever',
        'Lymphatic_filariasis','Hookworm_disease','Ascariasis'}
    high_burden = {'Ischemic_heart_disease','Stroke','Diabetes_mellitus',
        'Chronic_obstructive_pulmonary_disease','Lower_respiratory_infections','HIV/AIDS',
        'Tuberculosis','Malaria','Neonatal_disorders','Diarrheal_diseases','Breast_cancer',
        'Lung_cancer','Depressive_disorders','Road_injuries','Hypertensive_heart_disease',
        'Alzheimer_disease_and_other_dementias'}

    label_indices = {}
    sii_sorted = sorted(zip(diseases, sii_values, range(len(diseases))),
                        key=lambda x: x[1], reverse=True)
    for d, s, idx in sii_sorted[:5]:
        label_indices[idx] = (d.replace("_", " "), 'high_sii')
    for d, s, idx in sii_sorted[-3:]:
        label_indices[idx] = (d.replace("_", " "), 'low_sii')
    for i, d in enumerate(diseases):
        if d in ntd_names and i not in label_indices:
            label_indices[i] = (d.replace("_", " "), 'ntd')
    for i, d in enumerate(diseases):
        if d in high_burden and i not in label_indices:
            label_indices[i] = (d.replace("_", " "), 'burden')
    # trim volume labels 5 -> 3 to cut clutter
    for idx, _ in sorted(enumerate(paper_counts), key=lambda x: x[1], reverse=True)[:3]:
        if idx not in label_indices:
            label_indices[idx] = (diseases[idx].replace("_", " "), 'volume')

    # Larger, tiered fonts
    style_map = {
        'high_sii': dict(fontsize=14, fontweight='bold',   color='#C0392B'),
        'low_sii':  dict(fontsize=13, fontweight='normal', color='#2980B9'),
        'ntd':      dict(fontsize=13, fontweight='bold',   color='#8E44AD'),
        'burden':   dict(fontsize=12, fontweight='normal', color='#2C3E50'),
        'volume':   dict(fontsize=11, fontweight='normal', color='#7F8C8D'),
    }

    texts = []
    for idx, (label, key) in label_indices.items():
        st = style_map[key]
        texts.append(ax.text(emb[idx, 0], emb[idx, 1], label,
                     fontsize=st['fontsize'], fontweight=st['fontweight'],
                     color=st['color'], alpha=0.97, zorder=5))

    if HAVE_ADJUST:
        print(f"adjustText de-overlap on {len(texts)} labels...")
        arrow = dict(arrowstyle='-', color='gray', alpha=0.55, lw=0.7)
        try:
            adjust_text(texts, ax=ax, expand=(1.4, 1.5), arrowprops=arrow)
        except TypeError:
            adjust_text(texts, ax=ax, arrowprops=arrow)

    ax.set_xlabel('UMAP Dimension 1', fontsize=15)
    ax.set_ylabel('UMAP Dimension 2', fontsize=15)
    ax.tick_params(labelsize=12)
    ax.set_title('Disease Research Semantic Landscape  (Size = Research Volume, Colour = Isolation Index)',
                 fontsize=17, fontweight='bold')

    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#C0392B', markersize=11, label='Most isolated'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#8E44AD', markersize=11, label='NTDs'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2C3E50', markersize=11, label='High-burden'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2980B9', markersize=11, label='Least isolated'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=13, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(out_png.with_suffix('.pdf'), facecolor='white', bbox_inches='tight')
    print(f"  Saved: {out_png}")
    plt.close()

if __name__ == '__main__':
    main()
