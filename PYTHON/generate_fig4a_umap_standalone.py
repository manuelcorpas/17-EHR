#!/usr/bin/env python3.11
"""
Generate standalone Figure 4a: UMAP Disease Research Semantic Landscape.
Extracts panel (a) from the Nature Health Fig_04_Semantic_Isolation composite.
Output: docs/figures/Fig_04_UMAP_Semantic_Landscape.png
"""
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import rankdata

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Run: pip install umap-learn")
    sys.exit(1)

RANDOM_SEED = 42

# NTD list (matches 05-04 script)
WHO_NTDS = {
    'African trypanosomiasis', 'Chagas disease', 'Dengue',
    'Leishmaniasis', 'Leprosy', 'Lymphatic filariasis',
    'Onchocerciasis', 'Rabies', 'Schistosomiasis',
    'Trachoma', 'Yellow fever', 'Guinea worm disease',
    'Cystic echinococcosis', 'Hookworm disease', 'Ascariasis',
    'Food-borne trematodiases', 'Ebola', 'Zika virus'
}

def is_ntd(disease_name):
    clean = disease_name.replace('_', ' ')
    return clean in WHO_NTDS

def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "DATA" / "05-SEMANTIC"
    output_file = base / "docs" / "figures" / "Fig_04_UMAP_Semantic_Landscape.png"

    # Load data
    print("Loading semantic metrics...")
    with open(data_dir / "heim_sem_metrics.json") as f:
        metrics = json.load(f)

    # Load centroids from embeddings
    print("Loading disease centroids...")
    import h5py
    embeddings_dir = data_dir / "EMBEDDINGS"
    centroids = {}
    for disease_dir in sorted(embeddings_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        h5_files = list(disease_dir.glob("*.h5")) + list(disease_dir.glob("*.hdf5"))
        if not h5_files:
            continue
        disease_name = disease_dir.name
        try:
            with h5py.File(h5_files[0], 'r') as hf:
                if 'embeddings' in hf:
                    emb = hf['embeddings'][:]
                    centroids[disease_name] = emb.mean(axis=0)
        except Exception as e:
            print(f"  Warning: Could not load {disease_name}: {e}")

    print(f"  Loaded {len(centroids)} disease centroids")

    # Filter to diseases with both centroids and metrics
    diseases = [d for d in centroids.keys() if d in metrics['metrics']]
    print(f"  Diseases with centroids + metrics: {len(diseases)}")

    # Build matrices
    centroid_matrix = np.array([centroids[d] for d in diseases])

    # UMAP
    print("Running UMAP...")
    np.random.seed(RANDOM_SEED)
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric='cosine', random_state=RANDOM_SEED
    )
    embedding_2d = reducer.fit_transform(centroid_matrix)

    # Paper counts for sizing
    paper_counts = np.array([
        metrics['metrics'][d].get('n_papers', 100) for d in diseases
    ])
    sizes = 30 + (paper_counts - paper_counts.min()) / (paper_counts.max() - paper_counts.min() + 1) * 250

    # SII for colouring (rank normalised)
    sii_values = np.array([metrics['metrics'][d].get('sii', 0) for d in diseases])
    sii_ranks = rankdata(sii_values, method='average') / len(sii_values)

    # Create figure
    print("Generating UMAP plot...")
    fig, ax = plt.subplots(figsize=(12, 9))

    scatter = ax.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=sii_ranks, s=sizes,
        cmap='RdYlBu_r', vmin=0, vmax=1,
        alpha=0.7, edgecolors='white', linewidths=0.5
    )

    # Colorbar with actual SII values at percentile positions
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('SII rank percentile')
    pct_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    sii_at_pct = [np.percentile(sii_values, p * 100) for p in pct_ticks]
    cbar.set_ticks(pct_ticks)
    cbar.set_ticklabels([f'{v:.4f}' for v in sii_at_pct])

    # Label selection
    ntd_names = {
        'Leishmaniasis', 'Schistosomiasis', 'Onchocerciasis', 'Dengue',
        'Rabies', 'Trachoma', 'Leprosy', 'African_trypanosomiasis',
        'Chagas_disease', 'Yellow_fever', 'Lymphatic_filariasis',
        'Hookworm_disease', 'Ascariasis'
    }
    high_burden = {
        'Ischemic_heart_disease', 'Stroke', 'Diabetes_mellitus',
        'Chronic_obstructive_pulmonary_disease', 'Lower_respiratory_infections',
        'HIV/AIDS', 'Tuberculosis', 'Malaria', 'Neonatal_disorders',
        'Diarrheal_diseases', 'Breast_cancer', 'Lung_cancer',
        'Depressive_disorders', 'Road_injuries', 'Hypertensive_heart_disease',
        'Alzheimer_disease_and_other_dementias'
    }

    label_indices = {}
    sii_sorted = sorted(zip(diseases, sii_values, range(len(diseases))),
                        key=lambda x: x[1], reverse=True)

    for disease, sii, idx in sii_sorted[:5]:
        label_indices[idx] = (disease.replace("_", " "), 'high_sii')
    for disease, sii, idx in sii_sorted[-3:]:
        label_indices[idx] = (disease.replace("_", " "), 'low_sii')
    for i, d in enumerate(diseases):
        if d in ntd_names and i not in label_indices:
            label_indices[i] = (d.replace("_", " "), 'ntd')
    for i, d in enumerate(diseases):
        if d in high_burden and i not in label_indices:
            label_indices[i] = (d.replace("_", " "), 'burden')
    vol_sorted = sorted(enumerate(paper_counts), key=lambda x: x[1], reverse=True)
    for idx, _ in vol_sorted[:5]:
        if idx not in label_indices:
            label_indices[idx] = (diseases[idx].replace("_", " "), 'volume')

    style_map = {
        'high_sii': dict(fontsize=10, fontweight='bold', color='#C0392B'),
        'low_sii':  dict(fontsize=9, fontweight='normal', color='#2980B9'),
        'ntd':      dict(fontsize=9, fontweight='bold', color='#8E44AD'),
        'burden':   dict(fontsize=9, fontweight='normal', color='#2C3E50'),
        'volume':   dict(fontsize=8, fontweight='normal', color='#7F8C8D'),
    }

    offset_angles = [(12, 8), (-12, 12), (12, -10), (-14, -8),
                     (18, 0), (-18, 0), (0, 14), (0, -14)]
    for i, (idx, (label, style_key)) in enumerate(label_indices.items()):
        style = style_map[style_key]
        offset = offset_angles[i % len(offset_angles)]
        ax.annotate(
            label,
            xy=(embedding_2d[idx, 0], embedding_2d[idx, 1]),
            xytext=offset, textcoords='offset points',
            fontsize=style['fontsize'], fontweight=style['fontweight'],
            color=style['color'], alpha=0.95,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5),
        )

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Disease Research Semantic Landscape (Size = Volume, Colour = SII)',
                 fontsize=14, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
               markersize=8, label='Most isolated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8E44AD',
               markersize=8, label='NTDs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50',
               markersize=8, label='High-burden'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9',
               markersize=8, label='Least isolated'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    print(f"  Saved: {output_file}")

    plt.close()

if __name__ == '__main__':
    main()
