#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BIOBANK BIAS DETECTION PIPELINE
================================

Detects and analyzes various types of biases in biobank research publications:
- Population stratification (ancestry-related confounding)
- Batch effects (temporal processing variations)
- Missing data patterns (systematic missingness)
- Sampling biases (demographic underrepresentation)
- Technical variation (platform-specific biases)

This script analyzes abstracts and MeSH terms to identify bias-related discussions,
quantifies bias awareness trends, and provides comprehensive reporting.

INPUT: DATA/biobank_research_data.csv (from 00-00-biobank-data-retrieval.py)
OUTPUT: Bias detection results, visualizations, and comprehensive report

USAGE:
    python PYTHON/02-00-biobank-bias-detection.py

REQUIREMENTS:
    pip install pandas numpy matplotlib seaborn scikit-learn nltk
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import logging
import json
from pathlib import Path

# NLP libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "02-00-BIOBANK-BIAS-DETECTION")
os.makedirs(analysis_dir, exist_ok=True)

# Download NLTK data if needed
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

#############################################################################
# 1. BIAS DEFINITIONS AND PATTERNS
#############################################################################

BIAS_CATEGORIES = {
    'population_stratification': {
        'name': 'Population Stratification',
        'description': 'Ancestry-related confounding in genetic studies',
        'keywords': [
            'population stratification', 'ancestry', 'population structure',
            'genetic ancestry', 'principal component', 'eigenstrat', 'admixture',
            'population substructure', 'ethnic', 'ethnicity', 'race', 'racial',
            'european ancestry', 'african ancestry', 'asian ancestry', 
            'hispanic', 'latino', 'diverse population', 'homogeneous population',
            'gwas', 'genetic association', 'confounding', 'stratification'
        ],
        'mesh_terms': [
            'Genetics, Population', 'Ethnic Groups', 'Continental Population Groups',
            'Genome-Wide Association Study', 'Genetic Variation', 'Polymorphism, Single Nucleotide'
        ],
        'severity_indicators': {
            'high': ['confounding', 'false positive', 'spurious association'],
            'medium': ['adjust', 'control', 'account for'],
            'low': ['consider', 'potential', 'may affect']
        }
    },
    
    'batch_effects': {
        'name': 'Batch Effects',
        'description': 'Technical variation from temporal processing',
        'keywords': [
            'batch effect', 'batch correction', 'combat', 'technical variation',
            'processing batch', 'experimental batch', 'batch variability',
            'batch-to-batch', 'systematic bias', 'technical artifact',
            'normalization', 'batch adjustment', 'plate effect', 'run effect',
            'temporal variation', 'processing time', 'batch confounding'
        ],
        'mesh_terms': [
            'Artifact', 'Data Accuracy', 'Quality Control', 'Reproducibility of Results',
            'Bias', 'Research Design'
        ],
        'severity_indicators': {
            'high': ['severe batch', 'substantial batch', 'significant batch'],
            'medium': ['moderate batch', 'some batch', 'batch variation'],
            'low': ['minimal batch', 'slight batch', 'minor batch']
        }
    },
    
    'missing_data': {
        'name': 'Missing Data Patterns',
        'description': 'Non-random missingness indicating systematic bias',
        'keywords': [
            'missing data', 'missingness', 'incomplete data', 'data completeness',
            'missing at random', 'missing not at random', 'mcar', 'mar', 'mnar',
            'imputation', 'missing value', 'incomplete case', 'dropout',
            'loss to follow-up', 'attrition', 'non-response', 'data quality',
            'complete case', 'listwise deletion', 'multiple imputation'
        ],
        'mesh_terms': [
            'Data Collection', 'Data Accuracy', 'Lost to Follow-Up', 
            'Patient Dropouts', 'Data Interpretation, Statistical'
        ],
        'severity_indicators': {
            'high': ['substantial missing', 'extensive missing', '>30% missing'],
            'medium': ['moderate missing', '10-30% missing', 'some missing'],
            'low': ['minimal missing', '<10% missing', 'few missing']
        }
    },
    
    'sampling_bias': {
        'name': 'Sampling Bias',
        'description': 'Demographic underrepresentation',
        'keywords': [
            'sampling bias', 'selection bias', 'recruitment bias', 'underrepresented',
            'overrepresented', 'representative sample', 'generalizability',
            'external validity', 'diversity', 'inclusion', 'exclusion criteria',
            'healthy volunteer', 'convenience sample', 'demographic bias',
            'socioeconomic', 'geographic bias', 'urban rural', 'age bias',
            'gender bias', 'sex bias', 'minority', 'vulnerable population'
        ],
        'mesh_terms': [
            'Selection Bias', 'Patient Selection', 'Sampling Studies',
            'Health Status Disparities', 'Healthcare Disparities', 'Minority Groups'
        ],
        'severity_indicators': {
            'high': ['severe underrepresentation', 'lack of diversity', 'homogeneous sample'],
            'medium': ['limited diversity', 'some underrepresentation', 'moderate bias'],
            'low': ['slight underrepresentation', 'mostly representative', 'minor bias']
        }
    },
    
    'technical_variation': {
        'name': 'Technical Variation',
        'description': 'Platform-specific measurement biases',
        'keywords': [
            'technical variation', 'platform bias', 'measurement error',
            'instrument variation', 'assay variability', 'platform effect',
            'technical noise', 'measurement bias', 'calibration', 'standardization',
            'quality control', 'technical replicate', 'coefficient of variation',
            'inter-assay', 'intra-assay', 'platform comparison', 'method comparison',
            'analytical variation', 'pre-analytical'
        ],
        'mesh_terms': [
            'Equipment and Supplies', 'Calibration', 'Reference Standards',
            'Quality Control', 'Reproducibility of Results', 'Observer Variation'
        ],
        'severity_indicators': {
            'high': ['high variability', 'poor reproducibility', 'significant variation'],
            'medium': ['moderate variability', 'acceptable reproducibility', 'some variation'],
            'low': ['low variability', 'good reproducibility', 'minimal variation']
        }
    }
}

#############################################################################
# 2. DATA LOADING AND PREPROCESSING
#############################################################################

def load_biobank_data():
    """Load and preprocess biobank research data."""
    input_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run 00-00-biobank-data-retrieval.py first to generate the data.")
        return None
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    
    # Clean data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    
    # Filter years
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    
    # Clean text fields
    df['Abstract'] = df['Abstract'].fillna('')
    df['Title'] = df['Title'].fillna('')
    df['MeSH_Terms'] = df['MeSH_Terms'].fillna('')
    
    # Combine text for analysis
    df['combined_text'] = df['Title'] + ' ' + df['Abstract']
    
    logger.info(f"Loaded {len(df):,} papers from {df['Biobank'].nunique()} biobanks")
    logger.info(f"Year range: {df['Year'].min()}-{df['Year'].max()}")
    
    return df

#############################################################################
# 3. BIAS DETECTION FUNCTIONS
#############################################################################

def detect_bias_mentions(text, bias_category):
    """Detect mentions of a specific bias type in text."""
    if not text:
        return False, [], 0
    
    text_lower = text.lower()
    keywords = BIAS_CATEGORIES[bias_category]['keywords']
    
    found_keywords = []
    for keyword in keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Calculate confidence score based on number of matches
    confidence = min(len(found_keywords) / 3.0, 1.0)  # Normalize to 0-1
    
    return len(found_keywords) > 0, found_keywords, confidence

def analyze_bias_severity(text, bias_category):
    """Analyze the severity of bias discussion in text."""
    if not text:
        return 'none'
    
    text_lower = text.lower()
    severity_indicators = BIAS_CATEGORIES[bias_category]['severity_indicators']
    
    for severity_level in ['high', 'medium', 'low']:
        for indicator in severity_indicators[severity_level]:
            if indicator in text_lower:
                return severity_level
    
    return 'mentioned'  # Bias mentioned but severity unclear

def check_mesh_terms_for_bias(mesh_terms, bias_category):
    """Check if MeSH terms indicate bias-related content."""
    if not mesh_terms:
        return False
    
    mesh_list = [term.strip() for term in str(mesh_terms).split(';')]
    bias_mesh = BIAS_CATEGORIES[bias_category]['mesh_terms']
    
    for mesh_term in mesh_list:
        for bias_term in bias_mesh:
            if bias_term.lower() in mesh_term.lower():
                return True
    
    return False

def analyze_all_biases(df):
    """Analyze all papers for each type of bias."""
    logger.info("Analyzing papers for bias mentions...")
    
    results = []
    total_papers = len(df)
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"Processing paper {idx}/{total_papers} ({idx/total_papers*100:.1f}%)")
        
        paper_result = {
            'PMID': row['PMID'],
            'Biobank': row['Biobank'],
            'Year': row['Year'],
            'Title': row['Title']
        }
        
        # Analyze each bias type
        for bias_type in BIAS_CATEGORIES:
            # Check abstract
            mentioned, keywords, confidence = detect_bias_mentions(
                row['combined_text'], bias_type
            )
            
            # Check MeSH terms
            mesh_indicates = check_mesh_terms_for_bias(row['MeSH_Terms'], bias_type)
            
            # Determine severity if mentioned
            severity = analyze_bias_severity(row['combined_text'], bias_type) if mentioned else 'none'
            
            # Store results
            paper_result[f'{bias_type}_mentioned'] = int(mentioned)
            paper_result[f'{bias_type}_confidence'] = confidence
            paper_result[f'{bias_type}_severity'] = severity
            paper_result[f'{bias_type}_mesh'] = int(mesh_indicates)
            paper_result[f'{bias_type}_keywords'] = '|'.join(keywords) if keywords else ''
        
        results.append(paper_result)
    
    return pd.DataFrame(results)

#############################################################################
# 4. ANALYSIS FUNCTIONS
#############################################################################

def analyze_bias_cooccurrence(bias_df):
    """Analyze which biases tend to be discussed together."""
    logger.info("Analyzing bias co-occurrence patterns...")
    
    bias_types = list(BIAS_CATEGORIES.keys())
    n_biases = len(bias_types)
    
    # Create co-occurrence matrix
    cooccurrence_matrix = np.zeros((n_biases, n_biases))
    
    for i, bias1 in enumerate(bias_types):
        for j, bias2 in enumerate(bias_types):
            if i <= j:
                # Count papers mentioning both biases
                both_mentioned = ((bias_df[f'{bias1}_mentioned'] == 1) & 
                                 (bias_df[f'{bias2}_mentioned'] == 1)).sum()
                cooccurrence_matrix[i, j] = both_mentioned
                cooccurrence_matrix[j, i] = both_mentioned
    
    # Normalize by diagonal (self-occurrence)
    normalized_matrix = np.zeros_like(cooccurrence_matrix)
    for i in range(n_biases):
        for j in range(n_biases):
            if cooccurrence_matrix[i, i] > 0:
                normalized_matrix[i, j] = cooccurrence_matrix[i, j] / cooccurrence_matrix[i, i]
    
    return cooccurrence_matrix, normalized_matrix, bias_types

def analyze_temporal_trends(bias_df):
    """Analyze how bias awareness has changed over time."""
    logger.info("Analyzing temporal trends in bias awareness...")
    
    yearly_stats = []
    
    for year in sorted(bias_df['Year'].unique()):
        year_data = bias_df[bias_df['Year'] == year]
        year_stat = {'Year': year, 'Total_Papers': len(year_data)}
        
        # Calculate percentage of papers mentioning each bias
        for bias_type in BIAS_CATEGORIES:
            mentioned_count = (year_data[f'{bias_type}_mentioned'] == 1).sum()
            percentage = (mentioned_count / len(year_data)) * 100 if len(year_data) > 0 else 0
            year_stat[f'{bias_type}_percentage'] = percentage
        
        yearly_stats.append(year_stat)
    
    return pd.DataFrame(yearly_stats)

def analyze_by_biobank(bias_df):
    """Analyze bias patterns by biobank."""
    logger.info("Analyzing bias patterns by biobank...")
    
    biobank_stats = []
    
    for biobank in bias_df['Biobank'].unique():
        biobank_data = bias_df[bias_df['Biobank'] == biobank]
        
        stat = {
            'Biobank': biobank,
            'Total_Papers': len(biobank_data),
            'Years_Active': f"{biobank_data['Year'].min()}-{biobank_data['Year'].max()}"
        }
        
        # Calculate bias awareness metrics
        for bias_type in BIAS_CATEGORIES:
            mentioned_count = (biobank_data[f'{bias_type}_mentioned'] == 1).sum()
            percentage = (mentioned_count / len(biobank_data)) * 100 if len(biobank_data) > 0 else 0
            
            # Calculate average confidence
            mentioned_papers = biobank_data[biobank_data[f'{bias_type}_mentioned'] == 1]
            avg_confidence = mentioned_papers[f'{bias_type}_confidence'].mean() if len(mentioned_papers) > 0 else 0
            
            stat[f'{bias_type}_percentage'] = percentage
            stat[f'{bias_type}_avg_confidence'] = avg_confidence
        
        # Calculate overall bias awareness score
        stat['Overall_Bias_Awareness'] = np.mean([stat[f'{bias_type}_percentage'] 
                                                   for bias_type in BIAS_CATEGORIES])
        
        biobank_stats.append(stat)
    
    return pd.DataFrame(biobank_stats)

#############################################################################
# 5. VISUALIZATION FUNCTIONS
#############################################################################

def create_bias_overview_visualization(bias_df, temporal_df, biobank_df):
    """Create comprehensive bias overview visualization."""
    logger.info("Creating bias overview visualization...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall bias prevalence
    ax1 = fig.add_subplot(gs[0, :2])
    bias_prevalence = []
    for bias_type in BIAS_CATEGORIES:
        prevalence = (bias_df[f'{bias_type}_mentioned'] == 1).sum() / len(bias_df) * 100
        bias_prevalence.append({
            'Bias': BIAS_CATEGORIES[bias_type]['name'],
            'Prevalence': prevalence
        })
    
    prev_df = pd.DataFrame(bias_prevalence).sort_values('Prevalence', ascending=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(prev_df)))
    
    bars = ax1.barh(prev_df['Bias'], prev_df['Prevalence'], color=colors, edgecolor='black')
    ax1.set_xlabel('% of Papers Mentioning Bias', fontweight='bold')
    ax1.set_title('A. Bias Awareness in Biobank Research Literature', fontweight='bold', fontsize=14)
    
    # Add percentage labels
    for bar, val in zip(bars, prev_df['Prevalence']):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontweight='bold')
    
    # 2. Temporal trends
    ax2 = fig.add_subplot(gs[0, 2])
    recent_years = temporal_df[temporal_df['Year'] >= 2015]
    
    for bias_type in BIAS_CATEGORIES:
        ax2.plot(recent_years['Year'], recent_years[f'{bias_type}_percentage'],
                marker='o', label=BIAS_CATEGORIES[bias_type]['name'][:15], linewidth=2)
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('% Papers Discussing Bias', fontweight='bold')
    ax2.set_title('B. Temporal Trends (2015-2024)', fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Bias co-occurrence heatmap
    ax3 = fig.add_subplot(gs[1, :])
    cooccur_matrix, norm_matrix, bias_labels = analyze_bias_cooccurrence(bias_df)
    
    # Use short names for heatmap
    short_names = [BIAS_CATEGORIES[b]['name'].split()[0] for b in bias_labels]
    
    sns.heatmap(norm_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=short_names, yticklabels=short_names,
                ax=ax3, cbar_kws={'label': 'Co-occurrence Rate'})
    ax3.set_title('C. Bias Co-occurrence Matrix (Normalized)', fontweight='bold', fontsize=14)
    
    # 4. Biobank comparison
    ax4 = fig.add_subplot(gs[2, :2])
    top_biobanks = biobank_df.nlargest(5, 'Total_Papers')
    
    x = np.arange(len(top_biobanks))
    width = 0.15
    
    for i, bias_type in enumerate(list(BIAS_CATEGORIES.keys())[:5]):
        values = top_biobanks[f'{bias_type}_percentage'].values
        offset = (i - 2) * width
        ax4.bar(x + offset, values, width, label=BIAS_CATEGORIES[bias_type]['name'][:15])
    
    ax4.set_xlabel('Biobank', fontweight='bold')
    ax4.set_ylabel('% Papers Mentioning Bias', fontweight='bold')
    ax4.set_title('D. Bias Awareness by Biobank', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_biobanks['Biobank'].values, rotation=45, ha='right')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Severity distribution
    ax5 = fig.add_subplot(gs[2, 2])
    severity_data = []
    for bias_type in BIAS_CATEGORIES:
        mentioned_papers = bias_df[bias_df[f'{bias_type}_mentioned'] == 1]
        if len(mentioned_papers) > 0:
            severity_counts = mentioned_papers[f'{bias_type}_severity'].value_counts()
            for severity, count in severity_counts.items():
                if severity != 'none':
                    severity_data.append({
                        'Bias': BIAS_CATEGORIES[bias_type]['name'][:10],
                        'Severity': severity,
                        'Count': count
                    })
    
    if severity_data:
        sev_df = pd.DataFrame(severity_data)
        sev_pivot = sev_df.pivot(index='Bias', columns='Severity', values='Count').fillna(0)
        
        sev_pivot.plot(kind='bar', stacked=True, ax=ax5, 
                      color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
        ax5.set_xlabel('Bias Type', fontweight='bold')
        ax5.set_ylabel('Number of Papers', fontweight='bold')
        ax5.set_title('E. Bias Severity Distribution', fontweight='bold')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
        ax5.legend(title='Severity', fontsize=8)
    
    plt.suptitle('BIAS DETECTION IN BIOBANK RESEARCH: Comprehensive Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'bias_overview_comprehensive.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"✅ Saved comprehensive overview: {output_file}")
    
    return fig

#############################################################################
# 6. REPORT GENERATION
#############################################################################

def generate_comprehensive_report(bias_df, temporal_df, biobank_df):
    """Generate comprehensive bias detection report."""
    logger.info("Generating comprehensive report...")
    
    report = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_papers': len(bias_df),
            'total_biobanks': bias_df['Biobank'].nunique(),
            'year_range': f"{bias_df['Year'].min()}-{bias_df['Year'].max()}"
        },
        'summary_statistics': {},
        'temporal_trends': {},
        'biobank_analysis': {}
    }
    
    # Summary statistics for each bias
    for bias_type in BIAS_CATEGORIES:
        mentioned_count = (bias_df[f'{bias_type}_mentioned'] == 1).sum()
        prevalence = (mentioned_count / len(bias_df)) * 100
        
        mentioned_papers = bias_df[bias_df[f'{bias_type}_mentioned'] == 1]
        avg_confidence = mentioned_papers[f'{bias_type}_confidence'].mean() if len(mentioned_papers) > 0 else 0
        
        severity_dist = mentioned_papers[f'{bias_type}_severity'].value_counts().to_dict() if len(mentioned_papers) > 0 else {}
        
        report['summary_statistics'][bias_type] = {
            'name': BIAS_CATEGORIES[bias_type]['name'],
            'papers_mentioning': int(mentioned_count),
            'prevalence_percentage': round(prevalence, 2),
            'average_confidence': round(avg_confidence, 3),
            'severity_distribution': severity_dist
        }
    
    # Temporal trends
    recent_trends = temporal_df[temporal_df['Year'] >= 2019]
    for bias_type in BIAS_CATEGORIES:
        trend_data = recent_trends[['Year', f'{bias_type}_percentage']].to_dict('records')
        report['temporal_trends'][bias_type] = trend_data
    
    # Top biobanks by bias awareness
    top_biobanks = biobank_df.nlargest(5, 'Overall_Bias_Awareness')
    report['biobank_analysis']['top_aware_biobanks'] = top_biobanks[
        ['Biobank', 'Total_Papers', 'Overall_Bias_Awareness']
    ].to_dict('records')
    
    # Save JSON report
    report_file = os.path.join(analysis_dir, 'bias_detection_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"✅ Saved comprehensive report: {report_file}")
    
    # Generate text summary
    summary_text = f"""
BIOBANK BIAS DETECTION REPORT
=============================
Generated: {report['metadata']['analysis_date']}

DATASET OVERVIEW:
- Total papers analyzed: {report['metadata']['total_papers']:,}
- Biobanks included: {report['metadata']['total_biobanks']}
- Time period: {report['metadata']['year_range']}

BIAS PREVALENCE SUMMARY:
"""
    
    for bias_type, stats in report['summary_statistics'].items():
        summary_text += f"""
{stats['name'].upper()}:
- Papers mentioning: {stats['papers_mentioning']:,} ({stats['prevalence_percentage']:.1f}%)
- Average confidence: {stats['average_confidence']:.2f}
- Severity: {stats.get('severity_distribution', {})}
"""
    
    # Most aware biobanks
    summary_text += "\nTOP BIOBANKS BY BIAS AWARENESS:\n"
    for biobank_info in report['biobank_analysis']['top_aware_biobanks']:
        summary_text += f"• {biobank_info['Biobank']}: {biobank_info['Overall_Bias_Awareness']:.1f}% awareness\n"
    
    # Key findings
    most_discussed = max(report['summary_statistics'].items(), 
                        key=lambda x: x[1]['prevalence_percentage'])
    least_discussed = min(report['summary_statistics'].items(), 
                         key=lambda x: x[1]['prevalence_percentage'])
    
    summary_text += f"""
KEY FINDINGS:
- Most discussed bias: {most_discussed[1]['name']} ({most_discussed[1]['prevalence_percentage']:.1f}%)
- Least discussed bias: {least_discussed[1]['name']} ({least_discussed[1]['prevalence_percentage']:.1f}%)
- Gap in awareness: {most_discussed[1]['prevalence_percentage'] - least_discussed[1]['prevalence_percentage']:.1f} percentage points

RECOMMENDATIONS:
1. Increase awareness of underrepresented bias types, especially {least_discussed[1]['name']}
2. Standardize bias reporting across all biobanks
3. Implement systematic bias assessment protocols
4. Enhance diversity and inclusion in recruitment strategies
5. Develop bias mitigation guidelines specific to biobank research
"""
    
    # Save text summary
    summary_file = os.path.join(analysis_dir, 'bias_detection_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    logger.info(f"✅ Saved text summary: {summary_file}")
    
    print(summary_text)
    
    return report

#############################################################################
# 7. MAIN PIPELINE
#############################################################################

def main():
    """Main execution pipeline for bias detection."""
    print("=" * 80)
    print("BIOBANK BIAS DETECTION PIPELINE")
    print("Comprehensive Analysis of Bias Awareness in Biobank Research")
    print("=" * 80)
    
    try:
        # Load data
        print("\n📊 Loading biobank research data...")
        df = load_biobank_data()
        
        if df is None:
            print("❌ Could not load data. Please ensure biobank_research_data.csv exists.")
            return None, None
        
        # Detect biases
        print("\n🔍 Detecting bias mentions in abstracts...")
        print(f"   Analyzing {len(df):,} papers for {len(BIAS_CATEGORIES)} bias types...")
        bias_df = analyze_all_biases(df)
        
        # Save raw results
        bias_results_file = os.path.join(analysis_dir, 'bias_detection_results.csv')
        bias_df.to_csv(bias_results_file, index=False)
        print(f"✅ Saved bias detection results: {bias_results_file}")
        
        # Temporal analysis
        print("\n📈 Analyzing temporal trends...")
        temporal_df = analyze_temporal_trends(bias_df)
        temporal_file = os.path.join(analysis_dir, 'temporal_trends.csv')
        temporal_df.to_csv(temporal_file, index=False)
        
        # Biobank analysis
        print("\n🏥 Analyzing by biobank...")
        biobank_df = analyze_by_biobank(bias_df)
        biobank_file = os.path.join(analysis_dir, 'biobank_bias_analysis.csv')
        biobank_df.to_csv(biobank_file, index=False)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        create_bias_overview_visualization(bias_df, temporal_df, biobank_df)
        
        # Generate comprehensive report
        print("\n📋 Generating comprehensive report...")
        report = generate_comprehensive_report(bias_df, temporal_df, biobank_df)
        
        print("\n" + "=" * 80)
        print("✅ BIAS DETECTION PIPELINE COMPLETE!")
        print(f"📂 All results saved to: {analysis_dir}")
        print("\n📊 KEY OUTPUTS:")
        print("   • bias_detection_results.csv - Raw detection results")
        print("   • bias_overview_comprehensive.png - Main visualization")
        print("   • bias_detection_report.json - Detailed JSON report")
        print("   • bias_detection_summary.txt - Executive summary")
        
        # Print quick stats
        print("\n📈 QUICK STATISTICS:")
        for bias_type in BIAS_CATEGORIES:
            prevalence = (bias_df[f'{bias_type}_mentioned'] == 1).sum() / len(bias_df) * 100
            print(f"   • {BIAS_CATEGORIES[bias_type]['name']}: {prevalence:.1f}% of papers")
        
        # Try to run Fairlearn analysis if available
        try:
            from PYTHON.fairlearn_enhanced import run_enhanced_fairness_analysis
            print("\n🎯 Running enhanced Fairlearn analysis...")
            fairlearn_results = run_enhanced_fairness_analysis(bias_df, analysis_dir)
            print("✅ Fairlearn analysis complete!")
        except ImportError:
            print("\n💡 To run enhanced fairness analysis:")
            print("   1. Install Fairlearn: pip install fairlearn")
            print("   2. Use the enhanced Fairlearn script you have")
        
        return bias_df, report
        
    except Exception as e:
        logger.error(f"Error in bias detection pipeline: {e}")
        print(f"\n❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    bias_results, final_report = main()