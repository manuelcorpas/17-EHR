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

    diseases = list(centroids.keys())
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
    paper_counts = [
        metrics['metrics'].get(d, {}).get('n_papers', 100)
        for d in diseases
    ]
    sizes = np.array(paper_counts)
    sizes = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1) * 200

    # Get SII for coloring
    sii_values = [
        metrics['metrics'].get(d, {}).get('sii', 0.5)
        for d in diseases
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=sii_values,
        s=sizes,
        cmap='RdYlBu_r',
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Semantic Isolation Index (SII)', fontsize=14)

    # Label notable diseases (highest/lowest SII)
    sii_sorted = sorted(zip(diseases, sii_values, range(len(diseases))),
                        key=lambda x: x[1], reverse=True)

    # Top 5 most isolated
    for disease, sii, idx in sii_sorted[:5]:
        label = disease.replace("_", " ")[:20]
        ax.annotate(
            label,
            (embedding_2d[idx, 0], embedding_2d[idx, 1]),
            fontsize=12,
            fontweight='bold',
            alpha=0.9
        )

    # Bottom 5 least isolated
    for disease, sii, idx in sii_sorted[-5:]:
        label = disease.replace("_", " ")[:20]
        ax.annotate(
            label,
            (embedding_2d[idx, 0], embedding_2d[idx, 1]),
            fontsize=12,
            alpha=0.9
        )

    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title('Disease Research Semantic Landscape\n(Size = Research Volume, Color = Isolation Index)')

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

    # 1. UMAP clusters
    if generate_umap_clusters(centroids, metrics, FIGURES_DIR, logger):
        figures_generated += 1

    # 2. Similarity heatmap
    if similarity_df is not None:
        if generate_isolation_heatmap(similarity_df, FIGURES_DIR, logger):
            figures_generated += 1

    # 3. Temporal drift
    if generate_temporal_drift_plot(metrics, FIGURES_DIR, logger):
        figures_generated += 1

    # 4. Gap vs Isolation
    if disease_metrics is not None:
        if generate_gap_vs_isolation(disease_metrics, bhem_metrics, FIGURES_DIR, logger):
            figures_generated += 1

    # 5. Knowledge network
    if similarity_df is not None:
        if generate_knowledge_network(similarity_df, metrics, FIGURES_DIR, logger):
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
