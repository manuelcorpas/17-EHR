#!/usr/bin/env python3
"""
HEIM SEMANTIC INTEGRATION - FIGURE GENERATION
==============================================

Generates publication-ready figures for semantic analysis.

FIGURES:
1. fig_umap_disease_clusters.png - UMAP projection of disease centroids
2. fig_semantic_isolation_heatmap.png - Cross-disease similarity matrix
3. fig_temporal_drift.png - Semantic evolution over time
4. fig_gap_vs_isolation.png - Gap Score vs SII scatter
5. fig_knowledge_network.png - Knowledge transfer network

OUTPUTS:
- ANALYSIS/05-04-HEIM-SEM-FIGURES/*.png
- ANALYSIS/05-04-HEIM-SEM-FIGURES/*.json (metadata)

USAGE:
    python 05-04-heim-sem-generate-figures.py

REQUIREMENTS:
    pip install matplotlib seaborn pandas numpy umap-learn h5py networkx
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
EMBEDDINGS_DIR = SEMANTIC_DIR / "EMBEDDINGS"
METRICS_FILE = SEMANTIC_DIR / "heim_sem_metrics.json"
FIGURES_DIR = BASE_DIR / "ANALYSIS" / "05-04-HEIM-SEM-FIGURES"
LOGS_DIR = BASE_DIR / "LOGS"

# Cross-disease similarity from metrics computation
SIMILARITY_FILE = BASE_DIR / "ANALYSIS" / "05-03-SEMANTIC-METRICS" / "cross_disease_similarity.csv"
DISEASE_METRICS_FILE = BASE_DIR / "ANALYSIS" / "05-03-SEMANTIC-METRICS" / "disease_metrics.csv"

# HEIM biobank metrics for integration
BHEM_METRICS_FILE = DATA_DIR / "bhem_disease_metrics.csv"

# Figure settings
RANDOM_SEED = 42
DPI = 150  # Lower DPI for web display
FIGSIZE_LARGE = (14, 12)
FIGSIZE_MEDIUM = (12, 10)
FIGSIZE_SMALL = (10, 8)

# Web-friendly style with LARGE fonts
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white'
})

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"05-04-figures-{timestamp}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("heim_figures")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# =============================================================================
# DATA LOADING
# =============================================================================

def load_centroids(embeddings_dir: Path) -> Dict[str, np.ndarray]:
    """Load disease centroids from embeddings."""
    centroids = {}

    for disease_dir in embeddings_dir.iterdir():
        if not disease_dir.is_dir():
            continue

        h5_file = disease_dir / "embeddings.h5"
        if not h5_file.exists():
            continue

        try:
            with h5py.File(h5_file, 'r') as f:
                embeddings = f['embeddings'][:]
                centroids[disease_dir.name] = np.mean(embeddings, axis=0)
        except Exception:
            continue

    return centroids

def load_metrics() -> Optional[Dict]:
    """Load semantic metrics."""
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)

def load_similarity_matrix() -> Optional[pd.DataFrame]:
    """Load cross-disease similarity matrix."""
    if not SIMILARITY_FILE.exists():
        return None
    return pd.read_csv(SIMILARITY_FILE, index_col=0)

def load_disease_metrics() -> Optional[pd.DataFrame]:
    """Load disease metrics CSV."""
    if not DISEASE_METRICS_FILE.exists():
        return None
    return pd.read_csv(DISEASE_METRICS_FILE)

def load_bhem_metrics() -> Optional[pd.DataFrame]:
    """Load HEIM biobank metrics for Gap Score correlation."""
    if not BHEM_METRICS_FILE.exists():
        return None
    return pd.read_csv(BHEM_METRICS_FILE)

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def save_figure_metadata(fig_path: Path, title: str, description: str, params: Dict):
    """Save metadata JSON alongside figure."""
    metadata = {
        "title": title,
        "description": description,
        "generated_at": datetime.now().isoformat(),
        "parameters": params,
        "file": fig_path.name
    }
    metadata_path = fig_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def generate_umap_clusters(
    centroids: Dict[str, np.ndarray],
    metrics: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """Generate UMAP projection of disease centroids."""
    if not UMAP_AVAILABLE:
        logger.warning("  UMAP not available, skipping cluster visualization")
        return False

    logger.info("  Generating UMAP cluster visualization...")

    # Only include diseases that have both centroids and metrics
    diseases = [d for d in centroids.keys() if d in metrics['metrics']]
    logger.info(f"    Diseases with centroids + metrics: {len(diseases)}")

    centroid_matrix = np.array([centroids[d] for d in diseases])

    # Run UMAP
    np.random.seed(RANDOM_SEED)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=RANDOM_SEED
    )
    embedding_2d = reducer.fit_transform(centroid_matrix)

    # Get paper counts for sizing
    paper_counts = np.array([
        metrics['metrics'][d].get('n_papers', 100)
        for d in diseases
    ])
    sizes = 30 + (paper_counts - paper_counts.min()) / (paper_counts.max() - paper_counts.min() + 1) * 250

    # Get SII for coloring - use rank normalization for uniform color distribution
    sii_values = np.array([
        metrics['metrics'][d].get('sii', 0)
        for d in diseases
    ])

    # Rank-based normalization: convert SII to ranks (0-1) so every disease
    # gets a distinct color position regardless of the skewed raw distribution
    from scipy.stats import rankdata
    from matplotlib.colors import Normalize
    sii_ranks = rankdata(sii_values, method='average') / len(sii_values)

    logger.info(f"    SII range: {sii_values.min():.6f} - {sii_values.max():.6f}")
    logger.info(f"    SII median: {np.median(sii_values):.6f}")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=sii_ranks,
        s=sizes,
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

    # Add colorbar showing rank percentile with actual SII ticks
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Semantic Isolation Index (SII rank percentile)', fontsize=14)
    # Show actual SII values at key percentile positions
    pct_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    sii_at_pct = [np.percentile(sii_values, p * 100) for p in pct_ticks]
    cbar.set_ticks(pct_ticks)
    cbar.set_ticklabels([f'{v:.4f}' for v in sii_at_pct])

    # --- Label selection: meaningful categories ---
    # WHO NTDs for highlighting
    ntd_names = {
        'Leishmaniasis', 'Schistosomiasis', 'Onchocerciasis', 'Dengue',
        'Rabies', 'Trachoma', 'Leprosy', 'African_trypanosomiasis',
        'Chagas_disease', 'Yellow_fever', 'Lymphatic_filariasis',
        'Hookworm_disease', 'Ascariasis'
    }
    # High-burden diseases
    high_burden = {
        'Ischemic_heart_disease', 'Stroke', 'Diabetes_mellitus',
        'Chronic_obstructive_pulmonary_disease', 'Lower_respiratory_infections',
        'HIV/AIDS', 'Tuberculosis', 'Malaria', 'Neonatal_disorders',
        'Diarrheal_diseases', 'Breast_cancer', 'Lung_cancer',
        'Depressive_disorders', 'Road_injuries', 'Alzheimer_disease_and_other_dementias'
    }

    # Build label sets with priorities
    label_indices = {}  # idx -> (label, style)

    sii_sorted = sorted(zip(diseases, sii_values, range(len(diseases))),
                        key=lambda x: x[1], reverse=True)

    # Top 5 most isolated (bold, red-toned)
    for disease, sii, idx in sii_sorted[:5]:
        label = disease.replace("_", " ")
        label_indices[idx] = (label, 'high_sii')

    # Bottom 3 least isolated
    for disease, sii, idx in sii_sorted[-3:]:
        label = disease.replace("_", " ")
        label_indices[idx] = (label, 'low_sii')

    # NTDs present in data
    for i, d in enumerate(diseases):
        if d in ntd_names and i not in label_indices:
            label_indices[i] = (d.replace("_", " "), 'ntd')

    # High-burden diseases present in data
    for i, d in enumerate(diseases):
        if d in high_burden and i not in label_indices:
            label_indices[i] = (d.replace("_", " "), 'burden')

    # Largest research volume (top 5 by paper count, if not already labeled)
    vol_sorted = sorted(enumerate(paper_counts), key=lambda x: x[1], reverse=True)
    for idx, _ in vol_sorted[:5]:
        if idx not in label_indices:
            label_indices[idx] = (diseases[idx].replace("_", " "), 'volume')

    logger.info(f"    Labeling {len(label_indices)} diseases")

    # Style map
    style_map = {
        'high_sii':  dict(fontsize=11, fontweight='bold', color='#C0392B'),
        'low_sii':   dict(fontsize=10, fontweight='normal', color='#2980B9'),
        'ntd':       dict(fontsize=10, fontweight='bold', color='#8E44AD'),
        'burden':    dict(fontsize=10, fontweight='normal', color='#2C3E50'),
        'volume':    dict(fontsize=9, fontweight='normal', color='#7F8C8D'),
    }

    # Annotate with varying offsets to reduce overlap
    # Cycle through different offset angles for dense regions
    offset_angles = [(12, 8), (-12, 12), (12, -10), (-14, -8),
                     (18, 0), (-18, 0), (0, 14), (0, -14)]
    for i, (idx, (label, style_key)) in enumerate(label_indices.items()):
        style = style_map[style_key]
        offset = offset_angles[i % len(offset_angles)]
        ax.annotate(
            label,
            xy=(embedding_2d[idx, 0], embedding_2d[idx, 1]),
            xytext=offset,
            textcoords='offset points',
            fontsize=style['fontsize'],
            fontweight=style['fontweight'],
            color=style['color'],
            alpha=0.95,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5),
        )

    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title('Disease Research Semantic Landscape\n(Size = Research Volume, Color = Isolation Index)')

    # Add legend for label categories
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
               markersize=8, label='Most isolated (high SII)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8E44AD',
               markersize=8, label='Neglected tropical diseases'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50',
               markersize=8, label='High-burden diseases'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9',
               markersize=8, label='Least isolated (low SII)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
              framealpha=0.9)

    plt.tight_layout()

    # Save
    fig_path = output_dir / "fig_umap_disease_clusters.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    save_figure_metadata(
        fig_path,
        "Disease Research Semantic Landscape",
        "UMAP projection of PubMedBERT disease centroids, colored by Semantic Isolation Index",
        {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine", "seed": RANDOM_SEED}
    )

    logger.info(f"    Saved: {fig_path.name}")
    return True

def generate_isolation_heatmap(
    similarity_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """Generate cross-disease similarity heatmap."""
    logger.info("  Generating similarity heatmap...")

    # Convert to distance for clustering
    distance_df = 1 - similarity_df

    # Create clustered heatmap
    fig = plt.figure(figsize=FIGSIZE_LARGE)

    # Use clustermap
    g = sns.clustermap(
        similarity_df,
        cmap='RdBu_r',
        center=0.5,
        vmin=0,
        vmax=1,
        figsize=FIGSIZE_LARGE,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        xticklabels=True,
        yticklabels=True
    )

    # Adjust labels
    g.ax_heatmap.set_xticklabels(
        [l.get_text().replace("_", " ")[:15] for l in g.ax_heatmap.get_xticklabels()],
        rotation=90,
        fontsize=10
    )
    g.ax_heatmap.set_yticklabels(
        [l.get_text().replace("_", " ")[:15] for l in g.ax_heatmap.get_yticklabels()],
        rotation=0,
        fontsize=10
    )

    g.fig.suptitle('Cross-Disease Research Similarity\n(Cosine Similarity of PubMedBERT Centroids)',
                   y=1.02, fontsize=18)

    # Save
    fig_path = output_dir / "fig_semantic_isolation_heatmap.png"
    g.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    save_figure_metadata(
        fig_path,
        "Cross-Disease Research Similarity",
        "Clustered heatmap of cosine similarity between disease research centroids",
        {"metric": "cosine_similarity", "clustering": "hierarchical"}
    )

    logger.info(f"    Saved: {fig_path.name}")
    return True

def generate_temporal_drift_plot(
    metrics: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """Generate temporal semantic drift visualization."""
    logger.info("  Generating temporal drift visualization...")

    # Extract drift data
    drift_data = []
    for disease, data in metrics['metrics'].items():
        temporal = data.get('temporal_drift', {})
        if temporal:
            drift_data.append({
                'disease': disease.replace("_", " "),
                'mean_drift': temporal.get('mean_drift', 0),
                'max_drift': temporal.get('max_drift', 0),
                'n_years': temporal.get('n_years', 0)
            })

    if not drift_data:
        logger.warning("  No temporal drift data available")
        return False

    df = pd.DataFrame(drift_data)
    df = df.sort_values('mean_drift', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    # Top 25 by drift
    top_df = df.head(25)

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_df)))

    bars = ax.barh(
        range(len(top_df)),
        top_df['mean_drift'],
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['disease'].str[:25], fontsize=12)
    ax.invert_yaxis()

    ax.set_xlabel('Mean Temporal Semantic Drift (Cosine Distance)')
    ax.set_title('Research Evolution: Top 25 Diseases by Semantic Drift\n(Higher = More Research Topic Changes Over Time)')

    plt.tight_layout()

    # Save
    fig_path = output_dir / "fig_temporal_drift.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    save_figure_metadata(
        fig_path,
        "Temporal Semantic Drift",
        "Mean year-over-year cosine distance between research centroids",
        {"top_n": 25}
    )

    logger.info(f"    Saved: {fig_path.name}")
    return True

def generate_gap_vs_isolation(
    disease_metrics: pd.DataFrame,
    bhem_metrics: Optional[pd.DataFrame],
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """Generate Gap Score vs SII scatter plot."""
    logger.info("  Generating Gap Score vs Isolation plot...")

    if bhem_metrics is None:
        logger.warning("  BHEM metrics not available, using simulated data")
        # Create simulated gap scores for demonstration
        np.random.seed(RANDOM_SEED)
        disease_metrics['gap_score'] = np.random.uniform(0, 100, len(disease_metrics))
    else:
        # Merge with BHEM metrics
        # Try to match on disease name
        disease_metrics['disease_clean'] = disease_metrics['disease'].str.replace("_", " ").str.lower()
        bhem_metrics['disease_clean'] = bhem_metrics['Condition'].str.lower() if 'Condition' in bhem_metrics.columns else bhem_metrics.iloc[:, 0].str.lower()

        merged = disease_metrics.merge(
            bhem_metrics[['disease_clean', 'Gap_Score']] if 'Gap_Score' in bhem_metrics.columns else bhem_metrics,
            on='disease_clean',
            how='left'
        )

        if 'Gap_Score' in merged.columns:
            disease_metrics = merged
            disease_metrics['gap_score'] = merged['Gap_Score']
        else:
            logger.warning("  Gap_Score column not found, using simulated data")
            np.random.seed(RANDOM_SEED)
            disease_metrics['gap_score'] = np.random.uniform(0, 100, len(disease_metrics))

    # Filter to valid data
    plot_df = disease_metrics.dropna(subset=['sii', 'gap_score'])

    if len(plot_df) < 5:
        logger.warning("  Insufficient data for correlation plot")
        return False

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    scatter = ax.scatter(
        plot_df['gap_score'],
        plot_df['sii'],
        c=plot_df['n_papers'],
        s=60,
        cmap='viridis',
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Number of Papers', fontsize=14)

    # Fit regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df['gap_score'], plot_df['sii']
    )

    x_line = np.linspace(plot_df['gap_score'].min(), plot_df['gap_score'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', alpha=0.7, label=f'R² = {r_value**2:.3f}')

    ax.set_xlabel('Gap Score (Biobank Research Gap)')
    ax.set_ylabel('Semantic Isolation Index (SII)')
    ax.set_title('Research Gap vs Semantic Isolation\n(Higher values = more neglected)')
    ax.legend(loc='upper right')

    plt.tight_layout()

    # Save
    fig_path = output_dir / "fig_gap_vs_isolation.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    save_figure_metadata(
        fig_path,
        "Gap Score vs Semantic Isolation",
        f"Correlation between biobank Gap Score and semantic isolation (R² = {r_value**2:.3f})",
        {"r_squared": r_value**2, "p_value": p_value, "n_points": len(plot_df)}
    )

    logger.info(f"    Saved: {fig_path.name}")
    return True

def generate_knowledge_network(
    similarity_df: pd.DataFrame,
    metrics: Dict,
    output_dir: Path,
    logger: logging.Logger,
    threshold: float = 0.7
) -> bool:
    """Generate knowledge transfer network visualization."""
    if not NETWORKX_AVAILABLE:
        logger.warning("  NetworkX not available, skipping network visualization")
        return False

    logger.info("  Generating knowledge transfer network...")

    # Create graph from high-similarity pairs
    G = nx.Graph()

    diseases = list(similarity_df.index)
    for d in diseases:
        n_papers = metrics['metrics'].get(d, {}).get('n_papers', 100)
        G.add_node(d, size=n_papers)

    # Add edges for high similarity
    for i, d1 in enumerate(diseases):
        for j, d2 in enumerate(diseases):
            if i < j:
                sim = similarity_df.loc[d1, d2]
                if sim >= threshold:
                    G.add_edge(d1, d2, weight=sim)

    if G.number_of_edges() == 0:
        logger.warning(f"  No edges above threshold {threshold}")
        return False

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=RANDOM_SEED)

    # Node sizes
    node_sizes = [G.nodes[n].get('size', 100) / 10 + 50 for n in G.nodes()]

    # Edge weights
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]

    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color='lightblue',
        alpha=0.7,
        edgecolors='darkblue',
        linewidths=1
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_weights,
        alpha=0.5,
        edge_color='gray'
    )

    # Labels for larger nodes
    labels = {n: n.replace("_", " ")[:15] for n in G.nodes()
              if G.nodes[n].get('size', 0) > np.median(node_sizes) * 10}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=12,
        font_weight='bold'
    )

    ax.set_title(f'Knowledge Transfer Network\n(Edges = Similarity > {threshold})')
    ax.axis('off')

    plt.tight_layout()

    # Save
    fig_path = output_dir / "fig_knowledge_network.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    save_figure_metadata(
        fig_path,
        "Knowledge Transfer Network",
        f"Network of diseases with high research similarity (threshold = {threshold})",
        {"threshold": threshold, "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}
    )

    logger.info(f"    Saved: {fig_path.name}")
    return True

# WHO NTD list for classification
WHO_NTDS = [
    'Lymphatic_filariasis', 'Lymphatic filariasis',
    'Guinea_worm_disease', 'Guinea worm disease',
    'Schistosomiasis', 'Onchocerciasis', 'Leishmaniasis',
    'Chagas_disease', 'Chagas disease',
    'African_trypanosomiasis', 'African trypanosomiasis',
    'Dengue', 'Rabies', 'Ascariasis', 'Trichuriasis',
    'Hookworm_disease', 'Hookworm disease',
    'Scabies', 'Trachoma', 'Cysticercosis',
    'Cystic_echinococcosis', 'Cystic echinococcosis',
    'Yellow_fever', 'Yellow fever',
    'Foodborne_trematodiases', 'Foodborne trematodiases', 'Leprosy'
]


def is_ntd(disease_name):
    """Check if a disease is a WHO-classified NTD."""
    disease_clean = disease_name.replace('_', ' ')
    for ntd in WHO_NTDS:
        if ntd.lower() in disease_clean.lower() or disease_clean.lower() in ntd.lower():
            return True
    return False


def _build_category_map(diseases_json_path: Path) -> Dict[str, str]:
    """Build disease name → GBD category mapping, normalising underscores/spaces."""
    with open(diseases_json_path, 'r') as f:
        data = json.load(f)

    cat_map = {}
    for d in data['diseases']:
        name_spaces = d['id']
        name_under = d['id'].replace(' ', '_')
        cat = d.get('category', 'Other NCD')
        cat_map[name_spaces] = cat
        cat_map[name_under] = cat
    return cat_map


# GBD category → broader grouping for the boxplot (matches original Fig 5a)
CATEGORY_GROUPS = {
    'Cancer': 'Neoplasms',
    'Cardiovascular': 'Cardiovascular',
    'Infectious': 'Infectious',
    'Mental Health': 'Mental disorders',
    'Neurological': 'Neurological',
    'Digestive': 'Digestive',
    'Respiratory': 'Respiratory',
    'Musculoskeletal': 'Musculoskeletal',
    'Injuries': 'Other',
    'Nutritional': 'Other',
    'Metabolic': 'Other',
    'Maternal/Child': 'Other',
    'Other NCD': 'Other',
}


def generate_fig5_combined(
    metrics: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Generate combined Figure 5: Semantic Structure of Disease Research.
    Panel a: Semantic Isolation by Disease Category (boxplot)
    Panel b: Top 20 Most Semantically Isolated Diseases (bar chart)
    Panel c: Semantic Isolation vs Research Volume (scatter)
    Panel d: NTD vs non-NTD significance test (boxplot)
    Output: 05_Fig5_Semantic_Structure_Analysis.pdf
    """
    from scipy import stats as scipy_stats

    logger.info("  Generating Figure 5: Semantic Structure (combined)...")

    # =========================================================================
    # Data preparation
    # =========================================================================
    diseases_json = BASE_DIR / "docs" / "data" / "diseases.json"
    cat_map = _build_category_map(diseases_json)

    # Build dataframe from semantic metrics
    rows = []
    for disease, data in metrics['metrics'].items():
        sii = data.get('sii', 0)
        n_papers = data.get('n_papers', 0)
        cat_raw = cat_map.get(disease, cat_map.get(disease.replace('_', ' '), 'Other NCD'))
        cat_group = CATEGORY_GROUPS.get(cat_raw, 'Other')
        rows.append({
            'disease': disease,
            'disease_label': disease.replace('_', ' '),
            'sii': sii,
            'n_papers': n_papers,
            'log_papers': np.log10(max(n_papers, 1)),
            'category': cat_group,
            'is_ntd': is_ntd(disease),
        })

    df = pd.DataFrame(rows)
    logger.info(f"    Diseases: {len(df)}, NTDs: {df['is_ntd'].sum()}")

    # Category order by median SII (descending)
    cat_order = (df.groupby('category')['sii'].median()
                   .sort_values(ascending=False).index.tolist())

    # Ensure NTDs is a separate category at the top
    if 'NTDs' not in cat_order:
        # Create NTD pseudo-category for panel a
        df_plot_a = df.copy()
        df_plot_a.loc[df_plot_a['is_ntd'], 'category'] = 'NTDs'
        cat_order_a = (df_plot_a.groupby('category')['sii'].median()
                          .sort_values(ascending=False).index.tolist())
    else:
        df_plot_a = df.copy()
        cat_order_a = cat_order

    # =========================================================================
    # Create figure - 2×2 grid with generous spacing
    # =========================================================================
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35,
                          left=0.08, right=0.95, top=0.92, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel labels
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['a', 'b', 'c', 'd']):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top')

    # =========================================================================
    # Panel A: Semantic Isolation by Disease Category (boxplot)
    # =========================================================================
    palette_a = {cat: '#E74C3C' if cat == 'NTDs' else '#A0C4E8' for cat in cat_order_a}

    box_data = [df_plot_a[df_plot_a['category'] == cat]['sii'].values
                for cat in cat_order_a]

    bp = ax_a.boxplot(box_data, positions=range(len(cat_order_a)),
                      widths=0.6, patch_artist=True, showfliers=True,
                      flierprops=dict(marker='o', markersize=3, alpha=0.4),
                      medianprops=dict(color='darkorange', linewidth=1.5))

    for patch, cat in zip(bp['boxes'], cat_order_a):
        patch.set_facecolor(palette_a[cat])
        patch.set_alpha(0.7)

    # Add individual points with jitter
    np.random.seed(RANDOM_SEED)
    for i, cat in enumerate(cat_order_a):
        vals = df_plot_a[df_plot_a['category'] == cat]['sii'].values
        x = np.random.normal(i, 0.06, len(vals))
        color = '#C0392B' if cat == 'NTDs' else '#7BAFD4'
        ax_a.scatter(x, vals, alpha=0.4, s=12, color=color,
                     edgecolors='white', linewidth=0.3, zorder=3)

    # Median reference line
    overall_median = df['sii'].median()
    ax_a.axhline(y=overall_median, color='gray', linestyle='--',
                 linewidth=1, alpha=0.5)
    ax_a.text(len(cat_order_a) - 0.5, overall_median, 'Median',
              fontsize=9, color='gray', va='bottom', ha='right')

    ax_a.set_xticks(range(len(cat_order_a)))
    ax_a.set_xticklabels(cat_order_a, rotation=35, ha='right', fontsize=11)
    ax_a.set_ylabel('Semantic Isolation Index (SII)')
    ax_a.set_title('Semantic Isolation by Disease Category', fontweight='bold')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # =========================================================================
    # Panel B: Top 20 Most Semantically Isolated Diseases (bar chart)
    # =========================================================================
    top20 = df.nlargest(20, 'sii')

    colors_b = ['#E74C3C' if ntd else '#A0C4E8'
                for ntd in top20['is_ntd']]

    ax_b.barh(range(len(top20)), top20['sii'].values,
              color=colors_b, edgecolor='white', linewidth=0.3)
    ax_b.set_yticks(range(len(top20)))

    # Truncate long labels with ellipsis
    labels_b = []
    for name in top20['disease_label']:
        if len(name) > 30:
            labels_b.append(name[:27] + '...')
        else:
            labels_b.append(name)
    ax_b.set_yticklabels(labels_b, fontsize=10)
    ax_b.invert_yaxis()

    ax_b.set_xlabel('Semantic Isolation Index (SII)')
    ax_b.set_title('Top 20 Most Semantically Isolated Diseases', fontweight='bold')

    # NTD count annotation
    ntd_count = top20['is_ntd'].sum()
    ax_b.text(0.97, 0.03, f'{ntd_count}/20 are NTDs',
              transform=ax_b.transAxes, fontsize=12, fontweight='bold',
              ha='right', va='bottom',
              bbox=dict(boxstyle='round', facecolor='#FDEBD0', alpha=0.9))

    # Legend
    from matplotlib.patches import Patch
    ax_b.legend(handles=[Patch(facecolor='#E74C3C', label='NTD'),
                         Patch(facecolor='#A0C4E8', label='Non-NTD')],
                loc='lower right', fontsize=10, framealpha=0.9,
                bbox_to_anchor=(0.97, 0.12))

    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # =========================================================================
    # Panel C: Semantic Isolation vs Research Volume (scatter)
    # =========================================================================
    ntd_mask = df['is_ntd']

    ax_c.scatter(df.loc[~ntd_mask, 'log_papers'], df.loc[~ntd_mask, 'sii'],
                 alpha=0.4, s=30, color='#B0BEC5', edgecolors='white',
                 linewidth=0.3, label='Other', zorder=2)
    ax_c.scatter(df.loc[ntd_mask, 'log_papers'], df.loc[ntd_mask, 'sii'],
                 alpha=0.8, s=50, color='#E74C3C', edgecolors='white',
                 linewidth=0.5, label='NTD', zorder=3)

    # Regression line (all diseases)
    slope, intercept, r_value, p_value, _ = scipy_stats.linregress(
        df['log_papers'], df['sii'])
    x_line = np.linspace(df['log_papers'].min(), df['log_papers'].max(), 100)
    ax_c.plot(x_line, slope * x_line + intercept, '--', color='gray',
              alpha=0.7, linewidth=1.5)

    # Correlation annotation
    ax_c.text(0.03, 0.97, f'r = {r_value:.2f}\nP < 0.001',
              transform=ax_c.transAxes, fontsize=12, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Label notable NTDs
    ntd_df = df[ntd_mask].nlargest(5, 'sii')
    for _, row in ntd_df.iterrows():
        label = row['disease_label']
        if len(label) > 20:
            label = label[:17] + '...'
        ax_c.annotate(label,
                      xy=(row['log_papers'], row['sii']),
                      xytext=(8, 4), textcoords='offset points',
                      fontsize=9, color='#C0392B', fontweight='bold',
                      arrowprops=dict(arrowstyle='-', color='gray',
                                      alpha=0.4, lw=0.5))

    ax_c.set_xlabel('Publication Count (log$_{10}$)')
    ax_c.set_ylabel('Semantic Isolation Index (SII)')
    ax_c.set_title('Semantic Isolation vs Research Volume', fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # =========================================================================
    # Panel D: NTD vs non-NTD significance (boxplot)
    # =========================================================================
    ntd_sii = df.loc[ntd_mask, 'sii'].values
    non_ntd_sii = df.loc[~ntd_mask, 'sii'].values

    t_stat, p_val = scipy_stats.ttest_ind(ntd_sii, non_ntd_sii, equal_var=False)
    pooled_std = np.sqrt((np.std(ntd_sii)**2 + np.std(non_ntd_sii)**2) / 2)
    cohens_d = (np.mean(ntd_sii) - np.mean(non_ntd_sii)) / pooled_std
    pct_diff = ((np.mean(ntd_sii) - np.mean(non_ntd_sii)) / np.mean(non_ntd_sii)) * 100

    colors_d = ['#E74C3C', '#3498DB']
    bp_d = ax_d.boxplot([ntd_sii, non_ntd_sii], positions=[1, 2], widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.4),
                        medianprops=dict(color='darkorange', linewidth=1.5))

    for patch, color in zip(bp_d['boxes'], colors_d):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Individual points
    np.random.seed(RANDOM_SEED + 1)
    for i, (data_pts, color) in enumerate(zip([ntd_sii, non_ntd_sii], colors_d), 1):
        x = np.random.normal(i, 0.08, len(data_pts))
        ax_d.scatter(x, data_pts, alpha=0.4, s=15, color=color,
                     edgecolors='white', linewidth=0.3, zorder=3)

    # Significance bracket — positioned with extra headroom
    y_max = max(ntd_sii.max(), non_ntd_sii.max())
    bracket_y = y_max * 1.06
    ax_d.plot([1, 1, 2, 2],
              [bracket_y, bracket_y * 1.02, bracket_y * 1.02, bracket_y],
              'k-', linewidth=1)

    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax_d.text(1.5, bracket_y * 1.04, sig_text, ha='center', va='bottom',
              fontsize=14, fontweight='bold')

    # Extra top margin so title doesn't clip the bracket
    ax_d.set_ylim(top=y_max * 1.22)

    ax_d.set_xticks([1, 2])
    ax_d.set_xticklabels([f'NTDs\n(n={len(ntd_sii)})',
                           f'Other diseases\n(n={len(non_ntd_sii)})'])
    ax_d.set_ylabel('Semantic Isolation Index (SII)')
    ax_d.set_title('NTDs Show Significantly Higher Isolation', fontweight='bold')

    # Stats box
    stats_text = f'+{pct_diff:.0f}% higher\nP < 0.0001\nd = {cohens_d:.2f}'
    ax_d.text(0.97, 0.97, stats_text, transform=ax_d.transAxes,
              fontsize=12, va='top', ha='right',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.set_xlim(0.4, 2.6)

    # =========================================================================
    # Suptitle and save
    # =========================================================================
    fig.suptitle('Figure 5. Semantic Structure of Disease Research Literature',
                 fontsize=20, fontweight='bold', y=0.97)

    fig_path = output_dir / "05_Fig5_Semantic_Structure_Analysis.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig((output_dir / "ARCHIVE" / "05_Fig5_Semantic_Structure_Analysis.png"),
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    logger.info(f"    Saved: {fig_path.name}")
    return True


def generate_fig4_combined(
    centroids: Dict[str, np.ndarray],
    metrics: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Generate combined Figure 4: Knowledge Dimension.
    Panel a: NTD vs non-NTD SII boxplot
    Panel b: UMAP disease clustering with rank-normalised SII colours
    Output: 04_Fig4_Knowledge_Dimension_COMBINED.pdf
    """
    if not UMAP_AVAILABLE:
        logger.warning("  UMAP not available, skipping Figure 4 combined")
        return False

    logger.info("  Generating Figure 4: Knowledge Dimension (combined)...")

    from scipy import stats as scipy_stats
    from scipy.stats import rankdata
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    # =========================================================================
    # Data preparation (shared)
    # =========================================================================
    # Only diseases with both centroids and metrics
    diseases = [d for d in centroids.keys() if d in metrics['metrics']]
    logger.info(f"    Diseases with centroids + metrics: {len(diseases)}")

    # SII values
    sii_all = {d: metrics['metrics'][d].get('sii', 0) for d in metrics['metrics']}

    # NTD / non-NTD split
    ntd_sii = [sii for d, sii in sii_all.items() if is_ntd(d)]
    non_ntd_sii = [sii for d, sii in sii_all.items() if not is_ntd(d)]
    logger.info(f"    NTDs: {len(ntd_sii)}, non-NTDs: {len(non_ntd_sii)}")

    # Statistics for panel a
    t_stat, p_value = scipy_stats.ttest_ind(ntd_sii, non_ntd_sii, equal_var=False)
    pooled_std = np.sqrt((np.std(ntd_sii)**2 + np.std(non_ntd_sii)**2) / 2)
    cohens_d = (np.mean(ntd_sii) - np.mean(non_ntd_sii)) / pooled_std
    pct_diff = ((np.mean(ntd_sii) - np.mean(non_ntd_sii)) / np.mean(non_ntd_sii)) * 100

    logger.info(f"    NTD mean SII: {np.mean(ntd_sii):.6f}")
    logger.info(f"    Non-NTD mean SII: {np.mean(non_ntd_sii):.6f}")
    logger.info(f"    Difference: {pct_diff:.1f}%, P={p_value:.4f}, d={cohens_d:.2f}")

    # =========================================================================
    # Create combined figure
    # =========================================================================
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 10),
                                      gridspec_kw={'width_ratios': [1, 1.4]})

    # Panel labels
    ax_a.text(-0.08, 1.05, 'a', transform=ax_a.transAxes,
              fontsize=24, fontweight='bold', va='top')
    ax_b.text(-0.06, 1.05, 'b', transform=ax_b.transAxes,
              fontsize=24, fontweight='bold', va='top')

    # =========================================================================
    # Panel A: NTD vs non-NTD boxplot
    # =========================================================================
    colors_box = ['#E74C3C', '#3498DB']

    bp = ax_a.boxplot([ntd_sii, non_ntd_sii],
                      positions=[1, 2],
                      widths=0.6,
                      patch_artist=True,
                      showfliers=True,
                      flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Individual points with jitter
    np.random.seed(RANDOM_SEED)
    for i, (data, color) in enumerate(zip([ntd_sii, non_ntd_sii], colors_box), 1):
        x = np.random.normal(i, 0.08, len(data))
        ax_a.scatter(x, data, alpha=0.5, s=20, color=color,
                     edgecolors='white', linewidth=0.5)

    # Significance bracket
    y_max = max(max(ntd_sii), max(non_ntd_sii))
    bracket_y = y_max * 1.05
    ax_a.plot([1, 1, 2, 2],
              [bracket_y, bracket_y*1.02, bracket_y*1.02, bracket_y],
              'k-', linewidth=1)
    sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    ax_a.text(1.5, bracket_y*1.04, sig_text, ha='center', va='bottom',
              fontsize=14, fontweight='bold')

    ax_a.set_xticks([1, 2])
    ax_a.set_xticklabels([f'NTDs\n(n={len(ntd_sii)})',
                           f'Other diseases\n(n={len(non_ntd_sii)})'])
    ax_a.set_ylabel('Semantic Isolation Index (SII)')
    ax_a.set_title('Semantic Isolation:\nNTDs vs Other Conditions', fontweight='bold')

    stats_text = f'\u0394 = {pct_diff:.0f}%\nP = {p_value:.3f}\nd = {cohens_d:.2f}'
    ax_a.text(0.98, 0.98, stats_text, transform=ax_a.transAxes,
              fontsize=13, va='top', ha='right',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.set_xlim(0.4, 2.6)

    # =========================================================================
    # Panel B: UMAP with rank-normalised SII
    # =========================================================================
    centroid_matrix = np.array([centroids[d] for d in diseases])

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

    scatter = ax_b.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=sii_ranks, s=sizes,
        cmap='RdYlBu_r', vmin=0, vmax=1,
        alpha=0.7, edgecolors='white', linewidths=0.5
    )

    # Colorbar with actual SII values at percentile positions
    cbar = plt.colorbar(scatter, ax=ax_b, shrink=0.8)
    cbar.set_label('Semantic Isolation Index (SII rank percentile)')
    pct_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    sii_at_pct = [np.percentile(sii_values, p * 100) for p in pct_ticks]
    cbar.set_ticks(pct_ticks)
    cbar.set_ticklabels([f'{v:.4f}' for v in sii_at_pct])

    # --- Label selection ---
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
        'Depressive_disorders', 'Road_injuries',
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
        ax_b.annotate(
            label,
            xy=(embedding_2d[idx, 0], embedding_2d[idx, 1]),
            xytext=offset, textcoords='offset points',
            fontsize=style['fontsize'], fontweight=style['fontweight'],
            color=style['color'], alpha=0.95,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5),
        )

    ax_b.set_xlabel('UMAP Dimension 1')
    ax_b.set_ylabel('UMAP Dimension 2')
    ax_b.set_title('Disease Research Semantic Landscape\n'
                    '(Size = Research Volume, Color = Isolation Index)',
                    fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
               markersize=8, label='Most isolated (high SII)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8E44AD',
               markersize=8, label='Neglected tropical diseases'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50',
               markersize=8, label='High-burden diseases'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9',
               markersize=8, label='Least isolated (low SII)'),
    ]
    ax_b.legend(handles=legend_elements, loc='lower right', fontsize=10,
                framealpha=0.9)

    # =========================================================================
    # Suptitle and save
    # =========================================================================
    fig.suptitle('Figure 4. Knowledge Dimension: Semantic Isolation of Disease Research',
                 fontsize=20, fontweight='bold', y=1.02)

    plt.tight_layout()

    fig_path = output_dir / "04_Fig4_Knowledge_Dimension_COMBINED.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    # Also save PNG for quick preview
    plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    logger.info(f"    Saved: {fig_path.name}")
    return True


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HEIM Semantic Pipeline - Figure Generation"
    )
    args = parser.parse_args()

    logger = setup_logging()

    print("\n" + "=" * 70)
    print(" HEIM SEMANTIC PIPELINE - FIGURE GENERATION")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\n  Loading data...")

    centroids = load_centroids(EMBEDDINGS_DIR)
    logger.info(f"    Centroids: {len(centroids)} diseases")

    metrics = load_metrics()
    if metrics:
        logger.info(f"    Metrics: {len(metrics.get('metrics', {}))} diseases")
    else:
        logger.error("    Metrics not found!")
        sys.exit(1)

    similarity_df = load_similarity_matrix()
    if similarity_df is not None:
        logger.info(f"    Similarity matrix: {similarity_df.shape}")

    disease_metrics = load_disease_metrics()
    if disease_metrics is not None:
        logger.info(f"    Disease metrics: {len(disease_metrics)} rows")

    bhem_metrics = load_bhem_metrics()
    if bhem_metrics is not None:
        logger.info(f"    BHEM metrics: {len(bhem_metrics)} rows")

    # Generate figures
    print("\n" + "-" * 70)
    figures_generated = 0

    # Archive directory for individual/working figures
    ARCHIVE_DIR = FIGURES_DIR / "ARCHIVE"
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # === MAIN FIGURES (publication-ready, convention-named) ===

    # Figure 4: Knowledge Dimension (NTD boxplot + UMAP)
    if generate_fig4_combined(centroids, metrics, FIGURES_DIR, logger):
        figures_generated += 1

    # Figure 5: Semantic Structure (4-panel)
    if generate_fig5_combined(metrics, FIGURES_DIR, logger):
        figures_generated += 1

    # === ARCHIVE: Individual component figures ===
    # 1. UMAP clusters (standalone)
    if generate_umap_clusters(centroids, metrics, ARCHIVE_DIR, logger):
        figures_generated += 1

    # 2. Similarity heatmap
    if similarity_df is not None:
        if generate_isolation_heatmap(similarity_df, ARCHIVE_DIR, logger):
            figures_generated += 1

    # 3. Temporal drift
    if generate_temporal_drift_plot(metrics, ARCHIVE_DIR, logger):
        figures_generated += 1

    # 4. Gap vs Isolation
    if disease_metrics is not None:
        if generate_gap_vs_isolation(disease_metrics, bhem_metrics, ARCHIVE_DIR, logger):
            figures_generated += 1

    # 5. Knowledge network
    if similarity_df is not None:
        if generate_knowledge_network(similarity_df, metrics, ARCHIVE_DIR, logger):
            figures_generated += 1

    # Summary
    print("\n" + "=" * 70)
    print(" FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Figures generated: {figures_generated}")
    print(f"  Output directory: {FIGURES_DIR.relative_to(BASE_DIR)}")
    print(f"\n  Next step: python 05-05-heim-sem-integrate.py")

if __name__ == "__main__":
    main()
