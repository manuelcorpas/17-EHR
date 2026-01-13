#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MULTI-BIOBANK EQUITY GAP ANALYSIS PIPELINE
Comparing Health Equity Coverage Across Major Biobanks Including Genomics England

Script: PYTHON/01-00-biobank-equity-gap-discovery.py
Input: DATA/biobank_research_data.csv (including Genomics England)
Output: ANALYSIS/01-00-BIOBANK-EQUITY-GAPS/

PURPOSE:
Extends the biobank MeSH clustering pipeline to analyze health equity gaps across
multiple biobanks including Genomics England. Compares how different biobanks
address high-burden global diseases using GBD 2021 data.

PIPELINE:
1. Load biobank data (UK Biobank, All of Us, FinnGen, Genomics England, etc.)
2. Perform standard MeSH clustering per biobank
3. EQUITY GAP ANALYSIS:
   - Map publications to equity-sensitive diseases via MeSH
   - Calculate coverage scores per disease per biobank
   - Compute Equity Gap Scores using DALYs
   - Compare biobank performance on equity metrics
4. Generate comparative visualizations and opportunity indices

USAGE:
1. Ensure biobank_research_data.csv includes Genomics England data
2. Run: python PYTHON/01-00-biobank-equity-gap-discovery.py
3. Outputs saved to ANALYSIS/01-00-BIOBANK-EQUITY-GAPS/

Author: Health Equity Analysis Team
Date: September 2025
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict, Counter
import logging
from datetime import datetime
from matplotlib.gridspec import GridSpec

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-00-BIOBANK-EQUITY-GAPS")
clustering_dir = os.path.join(current_dir, "ANALYSIS", "00-02-BIOBANK-MESH-CLUSTERING")
os.makedirs(analysis_dir, exist_ok=True)

# Equity-sensitive disease set with GBD 2021 data (DALYs in millions)
EQUITY_DISEASES = {
    'Tuberculosis': {
        'mesh_terms': ['tuberculosis', 'mycobacterium tuberculosis', 'tb infection'],
        'mesh_codes': ['D014376', 'C001625'],
        'dalys': 46.2,
        'deaths': 1200,
        'category': 'Infectious'
    },
    'HIV/AIDS': {
        'mesh_terms': ['hiv', 'aids', 'acquired immunodeficiency syndrome', 'hiv infections'],
        'mesh_codes': ['D015658', 'D006678'],
        'dalys': 42.5,
        'deaths': 650,
        'category': 'Infectious'
    },
    'Malaria': {
        'mesh_terms': ['malaria', 'plasmodium', 'plasmodium falciparum'],
        'mesh_codes': ['D008288'],
        'dalys': 57.3,
        'deaths': 619,
        'category': 'Infectious'
    },
    'Diarrhoeal diseases': {
        'mesh_terms': ['diarrhea', 'dysentery', 'gastroenteritis', 'diarrhoea'],
        'mesh_codes': ['D003967', 'D003968'],
        'dalys': 71.8,
        'deaths': 1530,
        'category': 'Infectious'
    },
    'Lower respiratory infections': {
        'mesh_terms': ['pneumonia', 'respiratory tract infections', 'bronchitis', 'lower respiratory'],
        'mesh_codes': ['D012141', 'D011014'],
        'dalys': 97.2,
        'deaths': 2180,
        'category': 'Infectious'
    },
    'Neonatal disorders': {
        'mesh_terms': ['infant newborn diseases', 'neonatal', 'premature birth', 'preterm'],
        'mesh_codes': ['D007232', 'D007231'],
        'dalys': 96.8,
        'deaths': 1700,
        'category': 'Maternal/Child'
    },
    'Maternal disorders': {
        'mesh_terms': ['pregnancy complications', 'maternal death', 'maternal health'],
        'mesh_codes': ['D011248', 'D000052938'],
        'dalys': 17.6,
        'deaths': 287,
        'category': 'Maternal/Child'
    },
    'COPD': {
        'mesh_terms': ['pulmonary disease chronic obstructive', 'copd', 'emphysema', 'chronic bronchitis'],
        'mesh_codes': ['D029424', 'D001991'],
        'dalys': 74.4,
        'deaths': 3230,
        'category': 'NCD'
    },
    'Diabetes': {
        'mesh_terms': ['diabetes mellitus', 'diabetes', 'type 2 diabetes', 'type 1 diabetes'],
        'mesh_codes': ['D003920', 'D003924'],
        'dalys': 66.3,
        'deaths': 1660,
        'category': 'NCD'
    },
    'Cardiovascular diseases': {
        'mesh_terms': ['cardiovascular diseases', 'heart disease', 'coronary disease', 'myocardial infarction'],
        'mesh_codes': ['D002318', 'D003327'],
        'dalys': 393.1,
        'deaths': 17790,
        'category': 'NCD'
    },
    'Stroke': {
        'mesh_terms': ['stroke', 'cerebrovascular', 'brain ischemia'],
        'mesh_codes': ['D020521', 'D002561'],
        'dalys': 143.0,
        'deaths': 6550,
        'category': 'NCD'
    },
    'Cancer': {
        'mesh_terms': ['neoplasms', 'cancer', 'carcinoma', 'tumor', 'malignant'],
        'mesh_codes': ['D009369', 'D002277'],
        'dalys': 250.5,
        'deaths': 10000,
        'category': 'NCD'
    },
    'Mental disorders': {
        'mesh_terms': ['mental disorders', 'depression', 'anxiety', 'psychiatric'],
        'mesh_codes': ['D001523', 'D003863'],
        'dalys': 125.5,
        'deaths': 0,
        'category': 'Mental Health'
    },
    'Alzheimer disease': {
        'mesh_terms': ['alzheimer disease', 'dementia', 'alzheimer'],
        'mesh_codes': ['D000544', 'D003704'],
        'dalys': 28.8,
        'deaths': 1880,
        'category': 'Neurological'
    },
    'Asthma': {
        'mesh_terms': ['asthma', 'bronchial asthma'],
        'mesh_codes': ['D001249'],
        'dalys': 21.6,
        'deaths': 455,
        'category': 'Respiratory'
    },
    'Sickle cell disease': {
        'mesh_terms': ['anemia sickle cell', 'sickle cell', 'hemoglobin s disease'],
        'mesh_codes': ['D000755'],
        'dalys': 4.8,
        'deaths': 34,
        'category': 'Genetic'
    },
    'Rheumatic heart disease': {
        'mesh_terms': ['rheumatic heart disease', 'rheumatic fever'],
        'mesh_codes': ['D012214', 'D012213'],
        'dalys': 10.7,
        'deaths': 306,
        'category': 'NCD'
    },
    'Protein-energy malnutrition': {
        'mesh_terms': ['protein energy malnutrition', 'malnutrition', 'kwashiorkor', 'marasmus'],
        'mesh_codes': ['D011502', 'D007732'],
        'dalys': 18.9,
        'deaths': 232,
        'category': 'Nutrition'
    },
    'Iron deficiency anaemia': {
        'mesh_terms': ['anemia iron deficiency', 'iron deficiency'],
        'mesh_codes': ['D018798', 'D000740'],
        'dalys': 34.7,
        'deaths': 42,
        'category': 'Nutrition'
    },
    'Schistosomiasis': {
        'mesh_terms': ['schistosomiasis', 'bilharzia'],
        'mesh_codes': ['D012552'],
        'dalys': 1.9,
        'deaths': 11,
        'category': 'NTD'
    }
}

# Preprint identifiers to exclude
PREPRINT_IDENTIFIERS = [
    'medRxiv', 'bioRxiv', 'Research Square', 'arXiv', 'ChemRxiv'
]

#############################################################################
# 1. Data Loading Functions
#############################################################################

def load_biobank_data():
    """Load biobank research data including Genomics England"""
    input_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(input_file):
        # If data doesn't exist, create simulated data including GE
        logger.info("Creating simulated biobank data including Genomics England")
        df = create_simulated_biobank_data()
    else:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, low_memory=False)
    
    # Apply filtering
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    
    # Clean MeSH terms
    df['MeSH_Terms'] = df['MeSH_Terms'].fillna('')
    df['Journal'] = df['Journal'].fillna('Unknown Journal')
    
    # Exclude preprints
    df['is_preprint'] = False
    for identifier in PREPRINT_IDENTIFIERS:
        mask = df['Journal'].str.contains(identifier, case=False, na=False)
        df.loc[mask, 'is_preprint'] = True
    
    df = df[df['is_preprint'] == False]
    df = df.dropna(subset=['MeSH_Terms'])
    df = df[df['MeSH_Terms'].str.strip() != '']
    
    logger.info(f"Loaded {len(df):,} publications from {df['Biobank'].nunique()} biobanks")
    for biobank in df['Biobank'].unique():
        count = len(df[df['Biobank'] == biobank])
        logger.info(f"  • {biobank}: {count:,} papers")
    
    return df

def create_simulated_biobank_data():
    """Create simulated data for demonstration if real data unavailable"""
    np.random.seed(42)
    
    biobanks = ['UK Biobank', 'All of Us', 'FinnGen', 'Genomics England', 
                'China Kadoorie', 'BioBank Japan']
    
    data = []
    for biobank in biobanks:
        n_papers = np.random.randint(300, 800)
        
        for i in range(n_papers):
            # Simulate MeSH terms based on biobank focus
            if biobank == 'Genomics England':
                # GE focuses on rare diseases, cancer, some population genomics
                disease_pool = ['neoplasms', 'genetic diseases inborn', 'rare diseases',
                               'whole genome sequencing', 'pediatrics', 'pharmacogenomics']
                weight = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
            elif biobank == 'UK Biobank':
                # UKB has broad coverage
                disease_pool = ['cardiovascular diseases', 'diabetes mellitus', 'neoplasms',
                               'mental disorders', 'respiratory tract diseases', 'obesity']
                weight = [0.25, 0.15, 0.2, 0.15, 0.15, 0.1]
            elif biobank == 'All of Us':
                # All of Us emphasizes diversity and common diseases
                disease_pool = ['diabetes mellitus', 'hypertension', 'asthma', 
                               'mental disorders', 'health disparities', 'covid-19']
                weight = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
            else:
                # Other biobanks
                disease_pool = ['cardiovascular diseases', 'diabetes', 'cancer',
                               'genetics', 'epidemiology', 'biomarkers']
                weight = [0.2, 0.15, 0.2, 0.15, 0.15, 0.15]
            
            # Select 1-3 MeSH terms
            n_terms = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            selected_terms = np.random.choice(disease_pool, size=n_terms, replace=False, p=weight/np.sum(weight[:len(disease_pool)]))
            
            data.append({
                'Biobank': biobank,
                'PMID': f"{biobank[:2]}_{i:05d}",
                'Year': np.random.choice(range(2010, 2025)),
                'Journal': np.random.choice(['Nature', 'Science', 'Cell', 'NEJM', 'Lancet', 'BMJ']),
                'MeSH_Terms': '; '.join(selected_terms)
            })
    
    return pd.DataFrame(data)

#############################################################################
# 2. Equity Gap Analysis Functions
#############################################################################

def map_publications_to_diseases(df):
    """Map publications to equity-sensitive diseases based on MeSH terms"""
    logger.info("\nMapping publications to equity-sensitive diseases...")
    
    # Create disease-publication mapping for each biobank
    biobank_disease_mapping = defaultdict(lambda: defaultdict(list))
    
    for idx, row in df.iterrows():
        biobank = row['Biobank']
        mesh_terms = str(row['MeSH_Terms']).lower()
        pmid = row.get('PMID', f"paper_{idx}")
        
        for disease_name, disease_info in EQUITY_DISEASES.items():
            # Check if any disease MeSH term appears in publication
            matched = False
            for term in disease_info['mesh_terms']:
                if term.lower() in mesh_terms:
                    matched = True
                    break
            
            if matched:
                biobank_disease_mapping[biobank][disease_name].append(pmid)
    
    return biobank_disease_mapping

def calculate_biobank_equity_scores(biobank_disease_mapping, df):
    """Calculate equity gap scores for each biobank"""
    logger.info("\nCalculating equity gap scores per biobank...")
    
    # Get max DALYs for normalization
    max_dalys = max(d['dalys'] for d in EQUITY_DISEASES.values())
    
    equity_scores_data = []
    
    # Get unique biobanks
    biobanks = df['Biobank'].unique()
    
    for biobank in biobanks:
        biobank_total = len(df[df['Biobank'] == biobank])
        
        for disease_name, disease_info in EQUITY_DISEASES.items():
            # Count publications for this disease
            pub_count = len(biobank_disease_mapping[biobank][disease_name])
            
            # Calculate coverage score (0-1) using log scale
            coverage_score = min(1.0, np.log1p(pub_count) / np.log1p(50))
            
            # Calculate burden score (0-1)
            burden_score = disease_info['dalys'] / max_dalys
            
            # Calculate Equity Gap Score (0-100)
            egs = burden_score * (1 - coverage_score) * 100
            
            # Classify gap severity
            if egs >= 70:
                gap_category = 'Critical'
            elif egs >= 50:
                gap_category = 'High'
            elif egs >= 30:
                gap_category = 'Moderate'
            elif egs >= 10:
                gap_category = 'Low'
            else:
                gap_category = 'Adequate'
            
            equity_scores_data.append({
                'biobank': biobank,
                'disease': disease_name,
                'category': disease_info['category'],
                'dalys_millions': disease_info['dalys'],
                'deaths_thousands': disease_info['deaths'],
                'publications': pub_count,
                'pubs_per_1000': (pub_count / biobank_total) * 1000 if biobank_total > 0 else 0,
                'coverage_score': coverage_score,
                'burden_score': burden_score,
                'equity_gap_score': egs,
                'gap_category': gap_category
            })
    
    equity_df = pd.DataFrame(equity_scores_data)
    
    # Save detailed scores
    scores_file = os.path.join(analysis_dir, 'biobank_equity_gap_scores.csv')
    equity_df.to_csv(scores_file, index=False)
    logger.info(f"✓ Equity gap scores saved to {scores_file}")
    
    return equity_df

def compute_comparative_opportunity_index(equity_df):
    """Compute opportunity index comparing all biobanks"""
    logger.info("\nComputing comparative opportunity indices...")
    
    opportunity_data = []
    
    for biobank in equity_df['biobank'].unique():
        biobank_data = equity_df[equity_df['biobank'] == biobank]
        
        # Focus on neglected diseases (≤2 publications)
        neglected = biobank_data[biobank_data['publications'] <= 2]
        
        if len(neglected) > 0:
            # Calculate opportunity score
            total_opportunity = neglected['dalys_millions'].sum()
            n_neglected = len(neglected)
            
            # Top 3 opportunities
            top_opps = neglected.nlargest(3, 'dalys_millions')['disease'].tolist()
            
            opportunity_data.append({
                'biobank': biobank,
                'n_neglected_diseases': n_neglected,
                'total_opportunity_dalys': total_opportunity,
                'avg_gap_score': neglected['equity_gap_score'].mean(),
                'top_opportunities': ', '.join(top_opps[:3])
            })
    
    opp_df = pd.DataFrame(opportunity_data)
    opp_df = opp_df.sort_values('total_opportunity_dalys', ascending=False)
    
    # Save opportunity index
    opp_file = os.path.join(analysis_dir, 'biobank_opportunity_index.csv')
    opp_df.to_csv(opp_file, index=False)
    logger.info(f"✓ Opportunity index saved to {opp_file}")
    
    return opp_df

#############################################################################
# 3. ORIGINAL Visualization Functions (Dashboard & Matrix)
#############################################################################

def create_equity_comparison_dashboard(equity_df, opp_df):
    """Create comprehensive equity comparison dashboard"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Multi-Biobank Health Equity Gap Analysis\nComparing Coverage of High-Burden Global Diseases',
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Heatmap: Disease coverage by biobank
    ax1 = fig.add_subplot(gs[0, :])
    plot_coverage_heatmap(ax1, equity_df)
    
    # 2. Scatter: Genomics England focus
    ax2 = fig.add_subplot(gs[1, 0])
    plot_ge_scatter(ax2, equity_df)
    
    # 3. Bar chart: Comparative gap scores
    ax3 = fig.add_subplot(gs[1, 1])
    plot_comparative_gaps(ax3, equity_df)
    
    # 4. Radar chart: Biobank profiles
    ax4 = fig.add_subplot(gs[1, 2], projection='polar')
    plot_biobank_profiles(ax4, equity_df)
    
    # 5. Opportunity comparison
    ax5 = fig.add_subplot(gs[2, :2])
    plot_opportunity_comparison(ax5, opp_df)
    
    # 6. GE specific insights
    ax6 = fig.add_subplot(gs[2, 2])
    plot_ge_insights(ax6, equity_df)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_file = os.path.join(analysis_dir, 'Biobank_Equity_Comparison_Dashboard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Dashboard saved to {output_file}")

def plot_coverage_heatmap(ax, equity_df):
    """FIXED: Heatmap showing disease coverage across all biobanks"""
    
    # Pivot data for heatmap
    pivot_df = equity_df.pivot_table(
        values='publications',
        index='disease',
        columns='biobank',
        fill_value=0
    )
    
    # Sort diseases by total DALYs
    disease_dalys = {d: EQUITY_DISEASES[d]['dalys'] for d in pivot_df.index if d in EQUITY_DISEASES}
    sorted_diseases = sorted(disease_dalys.keys(), key=lambda x: disease_dalys[x], reverse=True)
    # Filter to only diseases that exist in both pivot_df and EQUITY_DISEASES
    sorted_diseases = [d for d in sorted_diseases if d in pivot_df.index]
    pivot_df = pivot_df.loc[sorted_diseases]
    
    # Convert values to integers for annotation
    annot_values = pivot_df.values.astype(int)
    
    # Create heatmap with log scale for colors but original values for annotations
    sns.heatmap(np.log1p(pivot_df), 
                annot=annot_values,  # Use integer values for annotations
                fmt='d',  # Now this will work with integers
                cmap='YlOrRd',
                cbar_kws={'label': 'log(Publications + 1)'},
                ax=ax)
    
    ax.set_title('Disease Coverage Across Biobanks (Number of Publications)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Biobank', fontsize=11)
    ax.set_ylabel('Disease (sorted by DALYs)', fontsize=11)
    
    # Highlight Genomics England column if present
    if 'Genomics England' in pivot_df.columns:
        ge_idx = list(pivot_df.columns).index('Genomics England')
        ax.add_patch(plt.Rectangle((ge_idx, 0), 1, len(pivot_df), 
                                  fill=False, edgecolor='blue', lw=3))
        ax.text(ge_idx + 0.5, -0.5, 'GE', fontsize=10, fontweight='bold',
               ha='center', color='blue')
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=8)

def plot_ge_scatter(ax, equity_df):
    """Scatter plot focused on Genomics England"""
    
    ge_data = equity_df[equity_df['biobank'] == 'Genomics England']
    
    if len(ge_data) == 0:
        ax.text(0.5, 0.5, 'No Genomics England data', ha='center', va='center')
        return
    
    x = ge_data['dalys_millions']
    y = ge_data['publications']
    colors = ['red' if gap == 'Critical' else 'orange' if gap == 'High' 
              else 'yellow' if gap == 'Moderate' else 'green' 
              for gap in ge_data['gap_category']]
    
    scatter = ax.scatter(x, y, s=x*5, c=colors, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Annotate critical gaps
    for _, row in ge_data[ge_data['gap_category'] == 'Critical'].iterrows():
        ax.annotate(row['disease'][:10], 
                   (row['dalys_millions'], row['publications']),
                   xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax.set_xscale('log')
    ax.set_xlabel('Disease Burden (DALYs millions, log scale)', fontsize=10)
    ax.set_ylabel('GE Publications', fontsize=10)
    ax.set_title('Genomics England: Burden vs Coverage', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Critical Gap', alpha=0.6),
        Patch(facecolor='orange', label='High Gap', alpha=0.6),
        Patch(facecolor='green', label='Low/Adequate', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

def plot_comparative_gaps(ax, equity_df):
    """Compare average gap scores across biobanks"""
    
    # Calculate mean gap scores per biobank
    gap_summary = equity_df.groupby('biobank').agg({
        'equity_gap_score': 'mean',
        'publications': 'sum'
    }).reset_index()
    gap_summary = gap_summary.sort_values('equity_gap_score', ascending=False)
    
    # Create bar chart
    colors = ['#E74C3C' if bb == 'Genomics England' else '#3498DB' 
              for bb in gap_summary['biobank']]
    
    bars = ax.bar(range(len(gap_summary)), gap_summary['equity_gap_score'], 
                  color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(range(len(gap_summary)))
    ax.set_xticklabels(gap_summary['biobank'], rotation=45, ha='right')
    ax.set_ylabel('Mean Equity Gap Score', fontsize=10)
    ax.set_title('Average Equity Gap by Biobank', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, gap_summary['equity_gap_score']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{score:.1f}', ha='center', fontsize=8)

def plot_biobank_profiles(ax, equity_df):
    """Radar chart showing biobank profiles across disease categories"""
    
    # Calculate coverage by category
    category_coverage = equity_df.groupby(['biobank', 'category']).agg({
        'coverage_score': 'mean'
    }).reset_index()
    
    categories = category_coverage['category'].unique()
    biobanks = category_coverage['biobank'].unique()[:4]  # Top 4 biobanks for clarity
    
    # Setup radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot each biobank
    colors = plt.cm.Set3(np.linspace(0, 1, len(biobanks)))
    
    for i, biobank in enumerate(biobanks):
        biobank_data = category_coverage[category_coverage['biobank'] == biobank]
        values = []
        for cat in categories:
            cat_data = biobank_data[biobank_data['category'] == cat]
            values.append(cat_data['coverage_score'].mean() if len(cat_data) > 0 else 0)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=biobank, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title('Disease Category Coverage Profiles', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.3)

def plot_opportunity_comparison(ax, opp_df):
    """Compare opportunity indices across biobanks"""
    
    if len(opp_df) == 0:
        ax.text(0.5, 0.5, 'No opportunity data available', ha='center', va='center')
        return
    
    # Create grouped bar chart
    x = np.arange(len(opp_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, opp_df['n_neglected_diseases'], width,
                  label='Neglected Diseases (n)', color='#E74C3C', alpha=0.7)
    bars2 = ax.bar(x + width/2, opp_df['total_opportunity_dalys'], width,
                  label='Opportunity (DALYs)', color='#3498DB', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(opp_df['biobank'], rotation=45, ha='right')
    ax.set_ylabel('Count / DALYs (millions)', fontsize=10)
    ax.set_title('Investment Opportunities by Biobank', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight Genomics England
    if 'Genomics England' in opp_df['biobank'].values:
        ge_idx = opp_df[opp_df['biobank'] == 'Genomics England'].index[0]
        ax.add_patch(plt.Rectangle((ge_idx - 0.5, 0), 1, ax.get_ylim()[1],
                                  fill=False, edgecolor='blue', lw=2, linestyle='--'))

def plot_ge_insights(ax, equity_df):
    """Specific insights for Genomics England"""
    
    ge_data = equity_df[equity_df['biobank'] == 'Genomics England']
    
    if len(ge_data) == 0:
        ax.text(0.5, 0.5, 'No Genomics England data', ha='center', va='center')
        return
    
    ax.axis('off')
    
    # Calculate key metrics
    critical_gaps = len(ge_data[ge_data['gap_category'] == 'Critical'])
    high_gaps = len(ge_data[ge_data['gap_category'] == 'High'])
    total_coverage = ge_data['publications'].sum()
    mean_gap = ge_data['equity_gap_score'].mean()
    
    # Get top gaps
    top_gaps = ge_data.nlargest(3, 'equity_gap_score')[['disease', 'equity_gap_score']]
    
    # Display insights
    insights_text = f"""GENOMICS ENGLAND EQUITY PROFILE
    
Key Metrics:
• Critical Gaps: {critical_gaps} diseases
• High Gaps: {high_gaps} diseases  
• Total Publications: {total_coverage}
• Mean Gap Score: {mean_gap:.1f}

Top Equity Gaps:"""
    
    ax.text(0.1, 0.9, insights_text, fontsize=10, transform=ax.transAxes,
           verticalalignment='top', fontweight='bold')
    
    y_pos = 0.5
    for _, row in top_gaps.iterrows():
        gap_text = f"• {row['disease']}: {row['equity_gap_score']:.1f}"
        ax.text(0.1, y_pos, gap_text, fontsize=9, transform=ax.transAxes)
        y_pos -= 0.08
    
    # Comparison to other biobanks
    all_means = equity_df.groupby('biobank')['equity_gap_score'].mean()
    ge_rank = len(all_means) - all_means.rank()['Genomics England'] + 1
    
    comparison_text = f"\nRanking: #{int(ge_rank)} of {len(all_means)} biobanks\n(1 = lowest gaps)"
    ax.text(0.1, 0.2, comparison_text, fontsize=10, transform=ax.transAxes,
           style='italic', color='blue')

def create_detailed_comparison_matrix(equity_df):
    """FIXED: Create detailed disease-by-biobank comparison matrix"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    fig.suptitle('Detailed Disease Coverage Comparison Across Biobanks',
                fontsize=16, fontweight='bold')
    
    # Left panel: Absolute coverage
    pivot_abs = equity_df.pivot_table(
        values='publications',
        index='disease',
        columns='biobank',
        fill_value=0
    )
    
    # Convert to integers for annotation
    annot_values = pivot_abs.values.astype(int)
    
    sns.heatmap(pivot_abs, annot=annot_values, fmt='d', cmap='Blues',
               cbar_kws={'label': 'Number of Publications'},
               ax=ax1)
    ax1.set_title('Absolute Publication Counts', fontsize=12)
    
    # Right panel: Gap scores
    pivot_gap = equity_df.pivot_table(
        values='equity_gap_score',
        index='disease',
        columns='biobank',
        fill_value=0
    )
    
    sns.heatmap(pivot_gap, annot=True, fmt='.0f', cmap='RdYlGn_r',
               vmin=0, vmax=100,
               cbar_kws={'label': 'Equity Gap Score (0-100)'},
               ax=ax2)
    ax2.set_title('Equity Gap Scores (higher = worse)', fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(analysis_dir, 'Detailed_Coverage_Matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Detailed matrix saved to {output_file}")

#############################################################################
# 4. NEW IMPROVED Visualization Functions (Separated & Clean)
#############################################################################

def create_clean_disease_coverage_heatmap(equity_df):
    """Create a standalone, properly formatted disease coverage heatmap"""
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Pivot data for heatmap
    pivot_df = equity_df.pivot_table(
        values='publications',
        index='disease',
        columns='biobank',
        fill_value=0
    )
    
    # Sort diseases by total burden (DALYs) for better visualization
    disease_dalys = equity_df.groupby('disease')['dalys_millions'].first().to_dict()
    sorted_diseases = sorted(pivot_df.index, key=lambda x: disease_dalys.get(x, 0), reverse=True)
    pivot_df = pivot_df.loc[sorted_diseases]
    
    # Create disease labels with DALYs included
    disease_labels_with_dalys = []
    for disease in pivot_df.index:
        dalys = disease_dalys.get(disease, 0)
        # Add DALY value to disease name with consistent formatting
        label = f"{disease} ({dalys:.0f}M DALYs)"
        disease_labels_with_dalys.append(label)
    
    # Create annotations with integer values
    annot_df = pivot_df.astype(int)
    
    # Create heatmap with log scale for colors but actual values for annotations
    sns.heatmap(
        np.log1p(pivot_df),  # Log scale for better color distribution
        annot=annot_df,      # Show actual publication counts
        fmt='d',             # Integer format
        cmap='YlOrRd',
        cbar_kws={'label': 'Log(Publications + 1)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        square=False,
        vmin=0,
        vmax=np.log1p(pivot_df.max().max()),
        yticklabels=disease_labels_with_dalys  # Use custom labels with DALYs
    )
    
    # Improve labels
    ax.set_title('Disease Coverage Across Biobanks\n(Number of Publications per Disease)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Biobank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Disease (sorted by global burden)', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Color code the y-axis labels based on DALY burden
    for i, label in enumerate(ax.get_yticklabels()):
        dalys = disease_dalys.get(sorted_diseases[i], 0)
        if dalys > 100:
            label.set_color('darkred')
            label.set_fontweight('bold')
        elif dalys > 50:
            label.set_color('darkorange')
            label.set_fontweight('bold')
        else:
            label.set_color('darkgreen')
        label.set_fontsize(9)
    
    # Highlight Genomics England if present
    if 'Genomics England' in pivot_df.columns:
        ge_idx = list(pivot_df.columns).index('Genomics England')
        ax.add_patch(plt.Rectangle((ge_idx, 0), 1, len(pivot_df), 
                                  fill=False, edgecolor='blue', lw=2))
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'Disease_Coverage_Heatmap_Clean.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Clean heatmap saved to {output_file}")
    
    return pivot_df

def create_corrected_equity_gap_heatmap(equity_df):
    """Create a corrected equity gap score heatmap with proper calculations"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create pivot table for heatmap
    pivot_gap = equity_df.pivot_table(
        values='equity_gap_score',
        index='disease',
        columns='biobank',
        fill_value=0
    )
    
    # Sort by average gap score across biobanks
    avg_gaps = pivot_gap.mean(axis=1)
    pivot_gap = pivot_gap.loc[avg_gaps.sort_values(ascending=False).index]
    
    # Create heatmap with corrected scores
    sns.heatmap(
        pivot_gap,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',  # Red = high gap (bad), Green = low gap (good)
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Equity Gap Score (0-100)\nHigher = Greater Research-Burden Mismatch'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        square=False
    )
    
    ax.set_title('Equity Gap Scores Across Biobanks\n(100 = Maximum Gap, 0 = No Gap)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Biobank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Disease (sorted by average equity gap)', fontsize=12, fontweight='bold')
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=9)
    
    # Add average gap scores on the right
    for i, disease in enumerate(pivot_gap.index):
        avg_gap = pivot_gap.loc[disease].mean()
        color = 'red' if avg_gap > 70 else 'orange' if avg_gap > 40 else 'green'
        ax.text(len(pivot_gap.columns) + 0.1, i + 0.5, f'{avg_gap:.0f}', 
               fontsize=8, ha='left', va='center', color=color, fontweight='bold')
    
    # Add column label
    ax.text(len(pivot_gap.columns) + 0.1, -1, 'Avg', fontsize=8, ha='left', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'Equity_Gap_Scores_Clean.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Clean equity gap heatmap saved to {output_file}")
    
    return pivot_gap

def create_biobank_comparison_chart(equity_df):
    """Create a clear comparison chart of biobank performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average equity gap by biobank
    ax1 = axes[0, 0]
    biobank_gaps = equity_df.groupby('biobank').agg({
        'equity_gap_score': 'mean',
        'publications': 'sum'
    }).reset_index()
    biobank_gaps = biobank_gaps.sort_values('equity_gap_score', ascending=True)
    
    colors = ['green' if gap < 40 else 'orange' if gap < 60 else 'red' 
              for gap in biobank_gaps['equity_gap_score']]
    
    bars = ax1.barh(range(len(biobank_gaps)), biobank_gaps['equity_gap_score'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(biobank_gaps)))
    ax1.set_yticklabels(biobank_gaps['biobank'])
    ax1.set_xlabel('Mean Equity Gap Score', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Biobank Performance\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, biobank_gaps['equity_gap_score'])):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                va='center', fontsize=9)
    
    # 2. Coverage of high-burden diseases (>50M DALYs)
    ax2 = axes[0, 1]
    high_burden = equity_df[equity_df['dalys_millions'] > 50]
    coverage_high = high_burden.groupby('biobank')['publications'].sum().sort_values(ascending=False)
    
    ax2.bar(range(len(coverage_high)), coverage_high.values, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(coverage_high)))
    ax2.set_xticklabels(coverage_high.index, rotation=45, ha='right')
    ax2.set_ylabel('Total Publications', fontsize=11, fontweight='bold')
    ax2.set_title('Coverage of High-Burden Diseases\n(>50M DALYs)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Number of neglected diseases per biobank
    ax3 = axes[1, 0]
    neglected = equity_df[equity_df['publications'] <= 2]
    neglected_counts = neglected.groupby('biobank')['disease'].count().sort_values(ascending=True)
    
    colors = ['red' if count > 15 else 'orange' if count > 10 else 'green' 
              for count in neglected_counts.values]
    
    bars = ax3.barh(range(len(neglected_counts)), neglected_counts.values, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(neglected_counts)))
    ax3.set_yticklabels(neglected_counts.index)
    ax3.set_xlabel('Number of Neglected Diseases (≤2 publications)', fontsize=11, fontweight='bold')
    ax3.set_title('Research Gaps by Biobank', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Distribution of gap categories
    ax4 = axes[1, 1]
    gap_categories = equity_df.groupby(['biobank', 'gap_category']).size().unstack(fill_value=0)
    
    # Ensure we have all categories in correct order
    category_order = ['Critical', 'High', 'Moderate', 'Low', 'Adequate']
    existing_cats = [cat for cat in category_order if cat in gap_categories.columns]
    gap_categories = gap_categories[existing_cats]
    
    gap_categories.plot(kind='bar', stacked=True, ax=ax4, 
                        color=['darkred', 'red', 'orange', 'yellow', 'green'][:len(existing_cats)],
                        alpha=0.7)
    ax4.set_xlabel('Biobank', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of Diseases', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution of Equity Gap Categories', fontsize=12, fontweight='bold')
    ax4.legend(title='Gap Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Biobank Health Equity Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'Biobank_Performance_Comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Performance comparison saved to {output_file}")

#############################################################################
# 5. Clustering Functions
#############################################################################

def perform_clustering_analysis(df):
    """Perform standard MeSH clustering for each biobank"""
    logger.info("\n" + "="*60)
    logger.info("PERFORMING MESH CLUSTERING ANALYSIS")
    logger.info("="*60)
    
    clustering_results = {}
    
    for biobank in df['Biobank'].unique():
        biobank_df = df[df['Biobank'] == biobank]
        logger.info(f"\nClustering {biobank}: {len(biobank_df)} papers")
        
        if len(biobank_df) < 20:
            logger.warning(f"  Skipping {biobank}: too few papers")
            continue
        
        # Create TF-IDF matrix
        mesh_docs = []
        for _, row in biobank_df.iterrows():
            terms = str(row['MeSH_Terms']).lower().replace(';', ' ')
            mesh_docs.append(terms)
        
        vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(mesh_docs)
        
        # Simple K-means with K=5 for consistency
        kmeans = KMeans(n_clusters=min(5, len(biobank_df)//10), random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix.toarray())
        
        clustering_results[biobank] = {
            'labels': labels,
            'n_clusters': len(np.unique(labels)),
            'silhouette': silhouette_score(tfidf_matrix.toarray(), labels) if len(np.unique(labels)) > 1 else 0
        }
        
        logger.info(f"  Created {clustering_results[biobank]['n_clusters']} clusters, "
                   f"silhouette = {clustering_results[biobank]['silhouette']:.3f}")
    
    return clustering_results

def analyze_mesh_overlaps(df):
    """Analyze MeSH term overlaps between biobanks"""
    logger.info("\nAnalyzing MeSH term overlaps...")
    
    biobank_terms = {}
    
    for biobank in df['Biobank'].unique():
        biobank_df = df[df['Biobank'] == biobank]
        all_terms = []
        
        for mesh_string in biobank_df['MeSH_Terms'].dropna():
            terms = [t.strip().lower() for t in str(mesh_string).split(';')]
            all_terms.extend(terms)
        
        term_counts = Counter(all_terms)
        top_terms = set([term for term, _ in term_counts.most_common(50)])
        biobank_terms[biobank] = top_terms
    
    # Calculate Jaccard similarities
    biobanks = sorted(list(biobank_terms.keys()))
    n = len(biobanks)
    overlap_matrix = np.zeros((n, n))
    
    for i, bb1 in enumerate(biobanks):
        for j, bb2 in enumerate(biobanks):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = biobank_terms[bb1] & biobank_terms[bb2]
                union = biobank_terms[bb1] | biobank_terms[bb2]
                jaccard = len(intersection) / len(union) if union else 0
                overlap_matrix[i, j] = jaccard
    
    # Save overlap matrix
    overlap_df = pd.DataFrame(overlap_matrix, index=biobanks, columns=biobanks)
    overlap_file = os.path.join(analysis_dir, 'mesh_overlap_matrix.csv')
    overlap_df.to_csv(overlap_file)
    logger.info(f"✓ Overlap matrix saved to {overlap_file}")
    
    return overlap_df

#############################################################################
# 6. Report Generation
#############################################################################

def generate_equity_report(equity_df, opp_df):
    """Generate comprehensive text report"""
    
    report = []
    report.append("="*80)
    report.append("MULTI-BIOBANK HEALTH EQUITY GAP ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    report.append(f"Biobanks Analyzed: {', '.join(equity_df['biobank'].unique())}")
    report.append(f"Diseases Analyzed: {len(EQUITY_DISEASES)}")
    
    # Overall statistics
    report.append("\n" + "="*60)
    report.append("OVERALL STATISTICS")
    report.append("="*60)
    
    summary = equity_df.groupby('biobank').agg({
        'publications': 'sum',
        'equity_gap_score': 'mean',
        'coverage_score': 'mean'
    }).round(2)
    
    report.append("\nBiobank Performance Summary:")
    report.append(summary.to_string())
    
    # Genomics England specific
    report.append("\n" + "="*60)
    report.append("GENOMICS ENGLAND PROFILE")
    report.append("="*60)
    
    ge_data = equity_df[equity_df['biobank'] == 'Genomics England']
    if len(ge_data) > 0:
        critical = len(ge_data[ge_data['gap_category'] == 'Critical'])
        high = len(ge_data[ge_data['gap_category'] == 'High'])
        
        report.append(f"\nCritical Gaps: {critical} diseases")
        report.append(f"High Gaps: {high} diseases")
        report.append(f"Mean Gap Score: {ge_data['equity_gap_score'].mean():.1f}")
        
        report.append("\nTop 5 Gaps:")
        top_gaps = ge_data.nlargest(5, 'equity_gap_score')[['disease', 'equity_gap_score', 'dalys_millions', 'publications']]
        report.append(top_gaps.to_string(index=False))
        
        report.append("\nStrengths (lowest gaps):")
        strengths = ge_data.nsmallest(3, 'equity_gap_score')[['disease', 'publications']]
        report.append(strengths.to_string(index=False))
    
    # Comparative analysis
    report.append("\n" + "="*60)
    report.append("COMPARATIVE ANALYSIS")
    report.append("="*60)
    
    report.append("\nMost Neglected Diseases Across All Biobanks:")
    total_pubs = equity_df.groupby('disease')['publications'].sum().sort_values()
    most_neglected = total_pubs.head(5)
    for disease, count in most_neglected.items():
        dalys = EQUITY_DISEASES[disease]['dalys']
        report.append(f"  • {disease}: {count} total pubs, {dalys:.1f}M DALYs")
    
    # Opportunities
    report.append("\n" + "="*60)
    report.append("INVESTMENT OPPORTUNITIES")
    report.append("="*60)
    
    if len(opp_df) > 0:
        report.append("\nBiobanks Ranked by Total Opportunity:")
        report.append(opp_df[['biobank', 'n_neglected_diseases', 'total_opportunity_dalys']].to_string(index=False))
    
    # Strategic recommendations
    report.append("\n" + "="*60)
    report.append("STRATEGIC RECOMMENDATIONS")
    report.append("="*60)
    
    report.append("""
1. IMMEDIATE PRIORITIES:
   • Address infectious disease gaps (TB, malaria, HIV)
   • Strengthen maternal/child health research
   • Expand coverage of neglected tropical diseases

2. GENOMICS ENGLAND SPECIFIC:
   • Leverage rare disease expertise for equity conditions
   • Expand paediatric cluster to address neonatal disorders
   • Consider partnerships for global health initiatives

3. COLLABORATIVE OPPORTUNITIES:
   • Joint initiatives for high-burden diseases
   • Data sharing for underrepresented populations
   • Coordinated approach to NTDs

4. LONG-TERM STRATEGY:
   • Develop equity metrics for funding decisions
   • Create targeted calls for neglected areas
   • Build capacity in low-resource settings
    """)
    
    # Save report
    report_text = '\n'.join(report)
    report_file = os.path.join(analysis_dir, 'equity_analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"✓ Report saved to {report_file}")
    print("\n" + "="*40)
    print("REPORT EXCERPT")
    print("="*40)
    print('\n'.join(report[:30]))

#############################################################################
# 7. Main Pipeline
#############################################################################

def main():
    """Main execution function"""
    print("="*80)
    print("MULTI-BIOBANK HEALTH EQUITY GAP ANALYSIS")
    print("Including Genomics England Comparative Assessment")
    print("="*80)
    
    try:
        # Load biobank data
        df = load_biobank_data()
        
        # Perform clustering analysis
        clustering_results = perform_clustering_analysis(df)
        
        # Analyze MeSH overlaps
        overlap_matrix = analyze_mesh_overlaps(df)
        
        # Map publications to diseases
        biobank_disease_mapping = map_publications_to_diseases(df)
        
        # Calculate equity gap scores
        equity_df = calculate_biobank_equity_scores(biobank_disease_mapping, df)
        
        # Compute opportunity indices
        opp_df = compute_comparative_opportunity_index(equity_df)
        
        # Generate ALL visualizations
        logger.info("\n" + "="*60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        # Original visualizations
        logger.info("\nGenerating original dashboard and matrix...")
        create_equity_comparison_dashboard(equity_df, opp_df)
        create_detailed_comparison_matrix(equity_df)
        
        # NEW: Generate improved separated visualizations
        logger.info("\nGenerating improved separated visualizations...")
        create_clean_disease_coverage_heatmap(equity_df)
        create_corrected_equity_gap_heatmap(equity_df)
        create_biobank_comparison_chart(equity_df)
        
        # Generate report
        generate_equity_report(equity_df, opp_df)
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to: {analysis_dir}")
        print("\nKey Files Generated:")
        print("\nORIGINAL VISUALIZATIONS:")
        print("  ✓ Biobank_Equity_Comparison_Dashboard.png")
        print("  ✓ Detailed_Coverage_Matrix.png")
        print("\nIMPROVED VISUALIZATIONS:")
        print("  ✓ Disease_Coverage_Heatmap_Clean.png")
        print("  ✓ Equity_Gap_Scores_Clean.png")
        print("  ✓ Biobank_Performance_Comparison.png")
        print("\nDATA FILES:")
        print("  ✓ biobank_equity_gap_scores.csv")
        print("  ✓ biobank_opportunity_index.csv")
        print("  ✓ mesh_overlap_matrix.csv")
        print("  ✓ equity_analysis_report.txt")
        
        # Key findings summary
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        # Find biobank with lowest/highest gaps
        mean_gaps = equity_df.groupby('biobank')['equity_gap_score'].mean()
        best_biobank = mean_gaps.idxmin()
        worst_biobank = mean_gaps.idxmax()
        
        print(f"\nBest Equity Performance: {best_biobank} (mean gap: {mean_gaps[best_biobank]:.1f})")
        print(f"Highest Gaps: {worst_biobank} (mean gap: {mean_gaps[worst_biobank]:.1f})")
        
        if 'Genomics England' in mean_gaps.index:
            ge_score = mean_gaps['Genomics England']
            ge_rank = len(mean_gaps) - mean_gaps.rank()['Genomics England'] + 1
            print(f"\nGenomics England:")
            print(f"  • Mean gap score: {ge_score:.1f}")
            print(f"  • Ranking: {int(ge_rank)} of {len(mean_gaps)} biobanks")
            
            ge_data = equity_df[equity_df['biobank'] == 'Genomics England']
            critical = len(ge_data[ge_data['gap_category'] == 'Critical'])
            print(f"  • Critical gaps: {critical} diseases")
        
        # Most neglected diseases
        total_coverage = equity_df.groupby('disease')['publications'].sum()
        most_neglected = total_coverage.nsmallest(3)
        print(f"\nMost Neglected Diseases (across all biobanks):")
        for disease, pubs in most_neglected.items():
            dalys = EQUITY_DISEASES[disease]['dalys']
            print(f"  • {disease}: {pubs} pubs, {dalys:.1f}M DALYs")
        
        return equity_df, opp_df
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    equity_results, opportunity_results = main()