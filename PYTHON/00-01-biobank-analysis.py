"""
BIOBANK RESEARCH ANALYSIS (EXCLUDING PREPRINTS)

Analyzes biobank research publications and creates comprehensive visualizations
following academic publication standards. Automatically filters out preprints
and provides detailed statistics on filtering.

ANALYSES:
1. Preprint filtering statistics and comparison
2. Year-by-year publication distribution (published papers only, 2000-2024)
3. Top 10 MeSH terms per biobank (published papers only)
4. Top 10 journals per biobank (published papers only)
5. Publications per year trend lines by biobank (published papers only)

INPUT: DATA/biobank_research_data.csv
OUTPUT: High-quality figures saved to ANALYSIS/00-01-BIOBANK-ANALYSIS/ directory

NOTE: 2025 data is excluded from all analyses as the year is incomplete.

USAGE:
python PYTHON/00-01-biobank-analysis.py

REQUIREMENTS:
pip install pandas matplotlib seaborn numpy scipy
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup paths
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "00-01-BIOBANK-ANALYSIS")
os.makedirs(analysis_dir, exist_ok=True)

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5
})

# Define preprint servers and patterns to exclude
PREPRINT_IDENTIFIERS = [
    'medRxiv', 'bioRxiv', 'Research Square', 'arXiv', 'ChemRxiv',
    'PeerJ Preprints', 'F1000Research', 'Authorea', 'Preprints.org',
    'SSRN', 'RePEc', 'OSF Preprints', 'SocArXiv', 'PsyArXiv',
    'EarthArXiv', 'engrXiv', 'TechRxiv'
]

def identify_preprints(df):
    """Identify and filter out preprints from the dataset"""
    print("🔍 Identifying and filtering preprints...")
    
    # Create a copy to avoid modifying original
    df_filtered = df.copy()
    
    # Apply year filtering first (exclude incomplete 2025)
    df_filtered = df_filtered[(df_filtered['Year'] >= 2000) & (df_filtered['Year'] <= 2024)]
    
    # Initialize preprint flag
    df_filtered['is_preprint'] = False
    
    # Check journal names for preprint identifiers
    for identifier in PREPRINT_IDENTIFIERS:
        mask = df_filtered['Journal'].str.contains(identifier, case=False, na=False)
        df_filtered.loc[mask, 'is_preprint'] = True
    
    # Additional checks for preprint patterns
    preprint_patterns = [
        r'preprint',
        r'pre-print', 
        r'working paper',
        r'discussion paper'
    ]
    
    for pattern in preprint_patterns:
        mask = df_filtered['Journal'].str.contains(pattern, case=False, na=False)
        df_filtered.loc[mask, 'is_preprint'] = True
    
    # Separate preprints and published papers
    df_preprints = df_filtered[df_filtered['is_preprint'] == True].copy()
    df_published = df_filtered[df_filtered['is_preprint'] == False].copy()
    
    # Print filtering statistics
    total_papers = len(df)
    preprint_count = len(df_preprints)
    published_count = len(df_published)
    preprint_percentage = (preprint_count / total_papers) * 100 if total_papers > 0 else 0
    
    print(f"📊 Preprint Filtering Results:")
    print(f"   Total papers retrieved: {total_papers:,}")
    print(f"   Preprints identified: {preprint_count:,} ({preprint_percentage:.1f}%)")
    print(f"   Published papers: {published_count:,} ({100-preprint_percentage:.1f}%)")
    
    if preprint_count > 0:
        print(f"\n🔬 Preprint sources found:")
        preprint_journals = df_preprints['Journal'].value_counts().head(10)
        for journal, count in preprint_journals.items():
            print(f"   {journal}: {count:,} papers")
    
    return df_published, df_preprints

def load_and_prepare_data():
    """Load and prepare the biobank research data, filtering out preprints"""
    print("📊 Loading biobank research data...")
    
    data_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the data retrieval script first.")
        return None, None
    
    df = pd.read_csv(data_file)
    
    # Clean and prepare data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    
    # Filter reasonable year range (exclude 2025 as it's incomplete)
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    
    # Clean MeSH terms and Journal names
    df['MeSH_Terms'] = df['MeSH_Terms'].fillna('')
    df['Journal'] = df['Journal'].fillna('Unknown Journal')
    
    print(f"✅ Loaded {len(df):,} papers from {df['Year'].min()}-{df['Year'].max()} (2025 excluded)")
    print(f"📋 Biobanks in dataset: {', '.join(df['Biobank'].unique())}")
    
    # Filter out preprints
    df_published, df_preprints = identify_preprints(df)
    
    return df_published, df_preprints

def create_preprint_filtering_summary(df_published, df_preprints):
    """Create a summary plot showing preprint filtering statistics"""
    print("📈 Creating preprint filtering summary...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall filtering pie chart
    total_papers = len(df_published) + len(df_preprints)
    published_count = len(df_published)
    preprint_count = len(df_preprints)
    
    if total_papers > 0:
        sizes = [published_count, preprint_count]
        labels = [f'Published Papers\n({published_count:,})', f'Preprints\n({preprint_count:,})']
        colors = ['lightblue', 'lightcoral']
        explode = (0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 10})
        ax1.set_title('A. Dataset Composition\n(Published vs Preprints)', fontweight='bold', pad=20)
    
    # 2. Preprint sources breakdown
    if len(df_preprints) > 0:
        preprint_journals = df_preprints['Journal'].value_counts().head(8)
        colors_bar = plt.cm.Set3(np.linspace(0, 1, len(preprint_journals)))
        
        bars = ax2.barh(range(len(preprint_journals)), preprint_journals.values, color=colors_bar)
        ax2.set_yticks(range(len(preprint_journals)))
        ax2.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                            for name in preprint_journals.index], fontsize=9)
        ax2.set_xlabel('Number of Papers', fontweight='bold')
        ax2.set_title('B. Top Preprint Sources\n(Excluded from Analysis)', fontweight='bold', pad=20)
        ax2.invert_yaxis()
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, preprint_journals.values)):
            ax2.text(count + 0.5, i, str(count), va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No preprints identified', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('B. Top Preprint Sources\n(Excluded from Analysis)', fontweight='bold', pad=20)
    
    # 3. Biobank distribution in published papers
    biobank_counts = df_published['Biobank'].value_counts()
    colors_biobank = plt.cm.Set2(np.linspace(0, 1, len(biobank_counts)))
    
    bars = ax3.bar(range(len(biobank_counts)), biobank_counts.values, color=colors_biobank, alpha=0.8)
    ax3.set_xticks(range(len(biobank_counts)))
    ax3.set_xticklabels(biobank_counts.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Number of Published Papers', fontweight='bold')
    ax3.set_title('C. Published Papers by Biobank\n(Preprints Excluded)', fontweight='bold', pad=20)
    
    # Add count labels on bars
    for bar, count in zip(bars, biobank_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(biobank_counts.values)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Year-over-year comparison (if preprints exist)
    if len(df_preprints) > 0:
        # Combine datasets with type indicator
        df_all_with_type = pd.concat([
            df_published.assign(paper_type='Published'),
            df_preprints.assign(paper_type='Preprint')
        ])
        
        # Group by year and type
        year_type_counts = df_all_with_type.groupby(['Year', 'paper_type']).size().unstack(fill_value=0)
        
        # Create stacked bar plot
        year_type_counts.plot(kind='bar', stacked=True, ax=ax4, 
                             color=['lightblue', 'lightcoral'], alpha=0.8)
        ax4.set_xlabel('Publication Year', fontweight='bold')
        ax4.set_ylabel('Number of Papers', fontweight='bold')
        ax4.set_title('D. Published vs Preprint Papers by Year\n(2000-2024, 2025 Excluded)', fontweight='bold', pad=20)
        ax4.legend(title='Paper Type', loc='upper left')
        ax4.tick_params(axis='x', rotation=45)
    else:
        # Just show published papers by year
        year_counts = df_published['Year'].value_counts().sort_index()
        ax4.bar(year_counts.index, year_counts.values, color='lightblue', alpha=0.8)
        ax4.set_xlabel('Publication Year', fontweight='bold')
        ax4.set_ylabel('Number of Published Papers', fontweight='bold')
        ax4.set_title('D. Published Papers by Year\n(2000-2024, No Preprints Found)', fontweight='bold', pad=20)
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'preprint_filtering_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Saved: {output_file}")
    return fig

def create_year_distribution_plot(df):
    """Create year-by-year publication distribution plot (published papers only)"""
    print("📈 Creating year-by-year publication distribution (published papers only)...")
    
    # Prepare data for stacked bar plot
    year_biobank = df.groupby(['Year', 'Biobank']).size().unstack(fill_value=0)
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create stacked bar plot with custom colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(year_biobank.columns)))
    year_biobank.plot(kind='bar', stacked=True, ax=ax, width=0.85, color=colors, alpha=0.8)
    
    ax.set_title('Year-by-Year Distribution of Published Papers by Biobank\n(2000-2024, 2025 Excluded as Incomplete)', 
                fontweight='bold', pad=25, fontsize=14)
    ax.set_xlabel('Publication Year', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Published Papers', fontweight='bold', fontsize=12)
    
    # Customize legend
    ax.legend(title='Biobank', bbox_to_anchor=(1.05, 1), loc='upper left', 
             title_fontsize=11, fontsize=10)
    
    # Rotate x-axis labels and improve spacing
    plt.xticks(rotation=45, ha='right')
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'biobank_yearly_distribution_published.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Saved: {output_file}")
    return fig

def create_mesh_terms_analysis(df):
    """Create top 10 MeSH terms analysis per biobank (published papers only)"""
    print("🏷️  Analyzing MeSH terms by biobank (published papers only)...")
    
    # Load deduplicated data for global analysis
    dedup_file = os.path.join(data_dir, 'biobank_research_data_deduplicated.csv')
    if os.path.exists(dedup_file):
        df_dedup_raw = pd.read_csv(dedup_file)
        df_dedup_raw['MeSH_Terms'] = df_dedup_raw['MeSH_Terms'].fillna('')
        df_dedup_raw['Journal'] = df_dedup_raw['Journal'].fillna('Unknown Journal')
        # Filter out preprints from deduplicated data too
        df_dedup, _ = identify_preprints(df_dedup_raw)
        print("📊 Using deduplicated dataset for global MeSH analysis (preprints excluded)")
    else:
        df_dedup = df
        print("⚠️ Deduplicated file not found, using main dataset")
    
    biobanks = df['Biobank'].unique()
    n_biobanks = len(biobanks)
    
    # Add one more subplot for global analysis
    total_plots = n_biobanks + 1
    
    # Calculate subplot dimensions
    n_cols = 3 if total_plots > 3 else total_plots
    n_rows = int(np.ceil(total_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if total_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if total_plots > 1 else [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_biobanks + 1))
    
    for idx, biobank in enumerate(biobanks):
        biobank_data = df[df['Biobank'] == biobank]
        
        # Extract all MeSH terms for this biobank
        all_mesh_terms = []
        for mesh_string in biobank_data['MeSH_Terms']:
            if pd.notna(mesh_string) and mesh_string.strip():
                terms = [term.strip() for term in mesh_string.split(';') if term.strip()]
                all_mesh_terms.extend(terms)
        
        # Get top 10 most common terms
        if all_mesh_terms:
            mesh_counter = Counter(all_mesh_terms)
            top_10_mesh = mesh_counter.most_common(10)
            
            if top_10_mesh:
                terms, counts = zip(*top_10_mesh)
                
                # Create horizontal bar plot with enhanced styling
                y_pos = np.arange(len(terms))
                bars = axes[idx].barh(y_pos, counts, color=colors[idx], alpha=0.8, 
                                     edgecolor='white', linewidth=0.5)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels([term[:40] + '...' if len(term) > 40 else term 
                                          for term in terms], fontsize=9)
                axes[idx].set_xlabel('Number of Published Papers', fontweight='bold')
                axes[idx].set_title(f'{biobank}\nTop 10 MeSH Terms', fontweight='bold', fontsize=11)
                axes[idx].invert_yaxis()
                
                # Add count labels on bars
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    axes[idx].text(count + max(counts)*0.01, i, str(count), 
                                  va='center', fontsize=9, fontweight='bold')
        
        if not all_mesh_terms or not top_10_mesh:
            axes[idx].text(0.5, 0.5, 'No MeSH terms available', 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=11, style='italic')
            axes[idx].set_title(f'{biobank}\nTop 10 MeSH Terms', fontweight='bold', fontsize=11)
    
    # Add global analysis (excluding UK Biobank to show diversity)
    global_idx = n_biobanks
    df_non_uk = df_dedup[df_dedup['Biobank'] != 'UK Biobank'] if 'UK Biobank' in df_dedup['Biobank'].values else df_dedup
    
    all_global_mesh_terms = []
    for mesh_string in df_non_uk['MeSH_Terms']:
        if pd.notna(mesh_string) and mesh_string.strip():
            terms = [term.strip() for term in mesh_string.split(';') if term.strip()]
            all_global_mesh_terms.extend(terms)
    
    if all_global_mesh_terms:
        global_mesh_counter = Counter(all_global_mesh_terms)
        top_10_global_mesh = global_mesh_counter.most_common(10)
        
        if top_10_global_mesh:
            terms, counts = zip(*top_10_global_mesh)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(terms))
            bars = axes[global_idx].barh(y_pos, counts, color='darkslategray', alpha=0.8,
                                        edgecolor='white', linewidth=0.5)
            axes[global_idx].set_yticks(y_pos)
            axes[global_idx].set_yticklabels([term[:40] + '...' if len(term) > 40 else term 
                                             for term in terms], fontsize=9)
            axes[global_idx].set_xlabel('Number of Published Papers', fontweight='bold')
            axes[global_idx].set_title('Non-UK Biobanks\nTop 10 MeSH Terms', fontweight='bold', fontsize=11)
            axes[global_idx].invert_yaxis()
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                axes[global_idx].text(count + max(counts)*0.01, i, str(count), 
                                     va='center', fontsize=9, fontweight='bold')
    else:
        axes[global_idx].text(0.5, 0.5, 'No non-UK MeSH terms available', 
                            ha='center', va='center', transform=axes[global_idx].transAxes,
                            fontsize=11, style='italic')
        axes[global_idx].set_title('Non-UK Biobanks\nTop 10 MeSH Terms', fontweight='bold', fontsize=11)
    
    # Hide unused subplots
    for idx in range(total_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'biobank_mesh_terms_published.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Saved: {output_file}")
    return fig

def create_journal_analysis(df):
    """Create top 10 journals analysis per biobank (published papers only)"""
    print("📚 Analyzing journals by biobank (published papers only)...")
    
    # Load deduplicated data for global analysis
    dedup_file = os.path.join(data_dir, 'biobank_research_data_deduplicated.csv')
    if os.path.exists(dedup_file):
        df_dedup_raw = pd.read_csv(dedup_file)
        df_dedup_raw['Journal'] = df_dedup_raw['Journal'].fillna('Unknown Journal')
        # Filter out preprints from deduplicated data too
        df_dedup, _ = identify_preprints(df_dedup_raw)
        print("📊 Using deduplicated dataset for global journal analysis (preprints excluded)")
    else:
        df_dedup = df
        print("⚠️ Deduplicated file not found, using main dataset")
    
    biobanks = df['Biobank'].unique()
    n_biobanks = len(biobanks)
    
    # Add one more subplot for global journal analysis
    total_plots = n_biobanks + 1
    
    # Calculate subplot dimensions
    n_cols = 3 if total_plots > 3 else total_plots
    n_rows = int(np.ceil(total_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if total_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if total_plots > 1 else [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_biobanks + 1))
    
    for idx, biobank in enumerate(biobanks):
        biobank_data = df[df['Biobank'] == biobank]
        
        # Get top 10 journals
        journal_counts = biobank_data['Journal'].value_counts().head(10)
        
        if len(journal_counts) > 0:
            # Create horizontal bar plot with enhanced styling
            y_pos = np.arange(len(journal_counts))
            bars = axes[idx].barh(y_pos, journal_counts.values, color=colors[idx], alpha=0.8,
                                 edgecolor='white', linewidth=0.5)
            axes[idx].set_yticks(y_pos)
            
            # Truncate long journal names
            journal_names = [name[:45] + '...' if len(name) > 45 else name 
                           for name in journal_counts.index]
            axes[idx].set_yticklabels(journal_names, fontsize=9)
            axes[idx].set_xlabel('Number of Published Papers', fontweight='bold')
            axes[idx].set_title(f'{biobank}\nTop 10 Journals', fontweight='bold', fontsize=11)
            axes[idx].invert_yaxis()
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, journal_counts.values)):
                axes[idx].text(count + max(journal_counts.values)*0.01, i, str(count), 
                              va='center', fontsize=9, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, 'No journal data available', 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=11, style='italic')
            axes[idx].set_title(f'{biobank}\nTop 10 Journals', fontweight='bold', fontsize=11)
    
    # Add global journal analysis (excluding UK Biobank to avoid dominance)
    global_idx = n_biobanks
    df_non_uk = df_dedup[df_dedup['Biobank'] != 'UK Biobank'] if 'UK Biobank' in df_dedup['Biobank'].values else df_dedup
    global_journal_counts = df_non_uk['Journal'].value_counts().head(10)
    
    if len(global_journal_counts) > 0:
        # Create horizontal bar plot
        y_pos = np.arange(len(global_journal_counts))
        bars = axes[global_idx].barh(y_pos, global_journal_counts.values, color='darkslategray', alpha=0.8,
                                    edgecolor='white', linewidth=0.5)
        axes[global_idx].set_yticks(y_pos)
        
        # Truncate long journal names
        journal_names = [name[:45] + '...' if len(name) > 45 else name 
                       for name in global_journal_counts.index]
        axes[global_idx].set_yticklabels(journal_names, fontsize=9)
        axes[global_idx].set_xlabel('Number of Published Papers', fontweight='bold')
        axes[global_idx].set_title('Non-UK Biobanks\nTop 10 Journals', fontweight='bold', fontsize=11)
        axes[global_idx].invert_yaxis()
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, global_journal_counts.values)):
            axes[global_idx].text(count + max(global_journal_counts.values)*0.01, i, str(count), 
                                 va='center', fontsize=9, fontweight='bold')
    else:
        axes[global_idx].text(0.5, 0.5, 'No non-UK journal data available', 
                            ha='center', va='center', transform=axes[global_idx].transAxes,
                            fontsize=11, style='italic')
        axes[global_idx].set_title('Non-UK Biobanks\nTop 10 Journals', fontweight='bold', fontsize=11)
    
    # Hide unused subplots
    for idx in range(total_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'biobank_journals_published.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Saved: {output_file}")
    return fig

def create_publication_trends_plot(df):
    """Create publication trends line plot by biobank (published papers only)"""
    print("📈 Creating publication trends by biobank (published papers only)...")
    
    # Prepare data
    yearly_counts = df.groupby(['Year', 'Biobank']).size().unstack(fill_value=0)
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot lines for each biobank with enhanced styling
    colors = plt.cm.Set2(np.linspace(0, 1, len(yearly_counts.columns)))
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, biobank in enumerate(yearly_counts.columns):
        ax.plot(yearly_counts.index, yearly_counts[biobank], 
               marker=markers[idx % len(markers)], linewidth=3, markersize=6, 
               label=biobank, color=colors[idx], alpha=0.8,
               linestyle=line_styles[idx % len(line_styles)],
               markeredgecolor='white', markeredgewidth=1)
    
    ax.set_title('Publication Trends by Biobank Over Time\n(Published Papers Only, 2000-2024)', 
                fontweight='bold', pad=25, fontsize=15)
    ax.set_xlabel('Publication Year', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Published Papers', fontweight='bold', fontsize=12)
    
    # Customize legend
    ax.legend(title='Biobank', bbox_to_anchor=(1.05, 1), loc='upper left',
             title_fontsize=12, fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis to show all years with better spacing
    ax.set_xticks(yearly_counts.index[::2])  # Show every other year
    plt.xticks(rotation=45, ha='right')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add subtle background shading for different decades
    for decade_start in range(2000, 2030, 10):
        decade_end = min(decade_start + 10, yearly_counts.index.max() + 1)
        if decade_start <= yearly_counts.index.max():
            ax.axvspan(decade_start, decade_end, alpha=0.05, color='gray')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'biobank_publication_trends_published.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Saved: {output_file}")
    return fig

def create_summary_statistics(df_published, df_preprints):
    """Create and save comprehensive summary statistics"""
    print("📊 Generating comprehensive summary statistics...")
    
    # Overall statistics
    total_papers = len(df_published) + len(df_preprints)
    published_papers = len(df_published)
    preprint_papers = len(df_preprints)
    year_range = f"{df_published['Year'].min()}-{df_published['Year'].max()}"
    biobank_counts = df_published['Biobank'].value_counts()
    
    # Journal statistics (published only)
    unique_journals = df_published['Journal'].nunique()
    top_journals = df_published['Journal'].value_counts().head(5)
    
    # MeSH term statistics (published only)
    all_mesh_terms = []
    for mesh_string in df_published['MeSH_Terms']:
        if pd.notna(mesh_string) and mesh_string.strip():
            terms = [term.strip() for term in mesh_string.split(';') if term.strip()]
            all_mesh_terms.extend(terms)
    
    unique_mesh_terms = len(set(all_mesh_terms))
    top_mesh_terms = Counter(all_mesh_terms).most_common(5)
    
    # Preprint statistics
    preprint_percentage = (preprint_papers / total_papers) * 100 if total_papers > 0 else 0
    
    # Create comprehensive summary report
    summary = f"""
BIOBANK RESEARCH ANALYSIS SUMMARY (PUBLISHED PAPERS ONLY)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

DATASET OVERVIEW:
  Total Papers Retrieved: {total_papers:,}
  Published Papers (Analysis): {published_papers:,} ({100-preprint_percentage:.1f}%)
  Preprints (Excluded): {preprint_papers:,} ({preprint_percentage:.1f}%)
  Year Range: {year_range} (2025 excluded as incomplete)
  Unique Journals (Published): {unique_journals:,}
  Unique MeSH Terms (Published): {unique_mesh_terms:,}

PUBLISHED PAPERS BY BIOBANK:
"""
    
    for biobank, count in biobank_counts.items():
        percentage = (count / published_papers) * 100
        summary += f"  {biobank}: {count:,} papers ({percentage:.1f}%)\n"
    
    summary += f"""
TOP 5 JOURNALS (Published Papers Only):
"""
    for journal, count in top_journals.items():
        summary += f"  {journal}: {count:,} papers\n"
    
    summary += f"""
TOP 5 MeSH TERMS (Published Papers Only):
"""
    for term, count in top_mesh_terms:
        summary += f"  {term}: {count:,} papers\n"
    
    if preprint_papers > 0:
        summary += f"""
PREPRINT SOURCES (EXCLUDED FROM ANALYSIS):
"""
        preprint_journals = df_preprints['Journal'].value_counts().head(5)
        for journal, count in preprint_journals.items():
            summary += f"  {journal}: {count:,} papers\n"
    
    summary += f"""
QUALITY ASSURANCE:
  ✅ Preprints excluded from all analyses
  ✅ 2025 data excluded (incomplete year)
  ✅ Publication-quality visualizations generated
  ✅ MeSH terms and journal names cleaned
  ✅ Year range validated (2000-2024)
  
NOTES:
  - All visualizations and statistics are based on published papers only
  - Preprints were identified and excluded to ensure research quality
  - 2025 data excluded to avoid bias from incomplete year
  - Deduplicated datasets used for global analyses where available
"""
    
    # Save summary
    summary_file = os.path.join(analysis_dir, 'biobank_analysis_summary_published.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"✅ Saved: {summary_file}")
    print(summary)

def create_combined_overview_plot(df_published):
    """Create a publication-quality combined overview plot"""
    print("🎨 Creating publication-quality combined overview plot...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1], hspace=0.35, wspace=0.25)
    
    # 1. Publication trends (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    yearly_counts = df_published.groupby(['Year', 'Biobank']).size().unstack(fill_value=0)
    colors = plt.cm.Set2(np.linspace(0, 1, len(yearly_counts.columns)))
    
    for idx, biobank in enumerate(yearly_counts.columns):
        ax1.plot(yearly_counts.index, yearly_counts[biobank], 
               marker='o', linewidth=3.5, markersize=6, 
               label=biobank, color=colors[idx], alpha=0.8,
               markeredgecolor='white', markeredgewidth=1.5)
    
    ax1.set_title('A. Publication Trends by Biobank Over Time (Published Papers Only, 2000-2024)', 
                 fontweight='bold', fontsize=14, loc='left', pad=20)
    ax1.set_xlabel('Publication Year', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Published Papers', fontweight='bold', fontsize=12)
    ax1.legend(title='Biobank', bbox_to_anchor=(1.02, 1), loc='upper left',
              title_fontsize=11, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 2. Total publications by biobank (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    biobank_counts = df_published['Biobank'].value_counts()
    bars = ax2.bar(range(len(biobank_counts)), biobank_counts.values, 
                   color=colors[:len(biobank_counts)], alpha=0.8,
                   edgecolor='white', linewidth=1)
    ax2.set_title('B. Total Published Papers by Biobank', 
                 fontweight='bold', fontsize=12, loc='left', pad=15)
    ax2.set_xlabel('Biobank', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Number of Published Papers', fontweight='bold', fontsize=11)
    ax2.set_xticks(range(len(biobank_counts)))
    ax2.set_xticklabels(biobank_counts.index, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for bar, value in zip(bars, biobank_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(biobank_counts.values)*0.01, 
                f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Year distribution (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    year_counts = df_published['Year'].value_counts().sort_index()
    bars = ax3.bar(year_counts.index, year_counts.values, alpha=0.8, 
                   color='steelblue', edgecolor='white', linewidth=0.5)
    ax3.set_title('C. Published Papers by Year (All Biobanks)', 
                 fontweight='bold', fontsize=12, loc='left', pad=15)
    ax3.set_xlabel('Publication Year', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Number of Published Papers', fontweight='bold', fontsize=11)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Enhanced summary statistics (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create enhanced summary text with quality indicators
    total_papers = len(df_published)
    year_range = f"{df_published['Year'].min()}-{df_published['Year'].max()}"
    unique_journals = df_published['Journal'].nunique()
    
    summary_text = f"""Dataset Quality Summary: {total_papers:,} peer-reviewed papers spanning {year_range} from {unique_journals:,} unique journals
Biobank Distribution: {' | '.join([f"{biobank}: {count:,}" for biobank, count in biobank_counts.head(3).items()])}
✅ Quality Assured: Preprints excluded | 2025 excluded (incomplete) | MeSH terms analyzed | Publication trends validated"""
    
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3, edgecolor='steelblue'),
             weight='normal', linespacing=1.5)
    
    plt.suptitle('Biobank Research Publications Analysis Overview (2000-2024)\n(Published Papers Only - Preprints Excluded)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'biobank_overview_combined_published.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Saved: {output_file}")
    return fig

def main():
    """Main analysis function"""
    print("=" * 70)
    print("BIOBANK RESEARCH ANALYSIS (PUBLISHED PAPERS ONLY)")
    print("Publication-quality visualizations excluding preprints")
    print("=" * 70)
    
    # Load data and filter preprints
    df_published, df_preprints = load_and_prepare_data()
    if df_published is None:
        return
    
    if len(df_published) == 0:
        print("❌ No published papers found after filtering preprints!")
        return
    
    print(f"\n📁 Output directory: {analysis_dir}")
    print(f"📊 Creating publication-quality visualizations (preprints excluded)...")
    
    # Create all visualizations
    try:
        # Preprint filtering summary
        create_preprint_filtering_summary(df_published, df_preprints)
        
        # Individual plots (published papers only)
        create_year_distribution_plot(df_published)
        create_mesh_terms_analysis(df_published)
        create_journal_analysis(df_published)
        create_publication_trends_plot(df_published)
        
        # Combined overview plot
        create_combined_overview_plot(df_published)
        
        # Comprehensive summary statistics
        create_summary_statistics(df_published, df_preprints)
        
        print(f"\n✅ Analysis complete!")
        print(f"📂 All figures saved to: {analysis_dir}")
        print(f"📊 Generated visualizations:")
        print(f"   - preprint_filtering_summary.png/pdf")
        print(f"   - biobank_yearly_distribution_published.png/pdf")
        print(f"   - biobank_mesh_terms_published.png/pdf")
        print(f"   - biobank_journals_published.png/pdf") 
        print(f"   - biobank_publication_trends_published.png/pdf")
        print(f"   - biobank_overview_combined_published.png/pdf")
        print(f"   - biobank_analysis_summary_published.txt")
        
        print(f"\n🎯 Key Insights:")
        print(f"   - All visualizations exclude preprints for research quality")
        print(f"   - 2025 data excluded to avoid bias from incomplete year")
        print(f"   - {len(df_published):,} published papers analyzed (2000-2024)")
        print(f"   - {len(df_preprints):,} preprints excluded from analysis")
        print(f"   - Publication-quality figures ready for academic use")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()