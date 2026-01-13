import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
input_file = os.path.join(parent_dir, "ANALYSIS", "00-01-LITERATURE-ANALYSIS", "00-01-biobank_statistics.txt")
csv_input_file = os.path.join(parent_dir, "DATA", "00-00-ehr_biobank_articles_with_citations.csv")
output_dir = os.path.join(parent_dir, "ANALYSIS", "00-02-STAT-VISUALIZATION")
os.makedirs(output_dir, exist_ok=True)

# Configure plot styling for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# Enhanced color palette for publication
BIOBANK_COLORS = {
    'UK Biobank': '#1f77b4',
    'Million Veteran Program': '#ff7f0e', 
    'FinnGen': '#2ca02c',
    'All of Us': '#d62728',
    'Estonian Biobank': '#9467bd'
}

# ENHANCED KEYWORD NORMALIZATION
def create_keyword_normalization_mapping():
    """Create comprehensive keyword normalization mapping to address all identified issues"""
    
    # Base normalization mapping
    normalization_map = {
        # Biobank variations
        'biobank': 'Biobank',
        'biobanks': 'Biobank',
        'bio-bank': 'Biobank',
        'bio-banks': 'Biobank',
        
        # GWAS variations
        'gwas': 'GWAS (Genome-wide association studies)',
        'genome-wide association study': 'GWAS (Genome-wide association studies)',
        'genome-wide association studies': 'GWAS (Genome-wide association studies)',
        'genome wide association study': 'GWAS (Genome-wide association studies)',
        'genome wide association studies': 'GWAS (Genome-wide association studies)',
        'genomewide association study': 'GWAS (Genome-wide association studies)',
        'genomewide association studies': 'GWAS (Genome-wide association studies)',
        
        # Mendelian randomization variations
        'mendelian randomization': 'Mendelian randomization',
        'mendelian randomisation': 'Mendelian randomization',
        'mendelian randomization study': 'Mendelian randomization',
        'mendelian randomisation study': 'Mendelian randomization',
        'mendelian randomization analysis': 'Mendelian randomization',
        'mendelian randomisation analysis': 'Mendelian randomization',
        'mr study': 'Mendelian randomization',
        'mr analysis': 'Mendelian randomization',
        
        # Causality variations
        'causality': 'Causality and causal inference',
        'causal relationship': 'Causality and causal inference',
        'causal effect': 'Causality and causal inference',
        'causal association': 'Causality and causal inference',
        'causal inference': 'Causality and causal inference',
        'causal analysis': 'Causality and causal inference',
        
        # MVP variations
        'mvp': 'Million Veteran Program (MVP)',
        'million veteran program': 'Million Veteran Program (MVP)',
        'million veteran programme': 'Million Veteran Program (MVP)',
        'veterans affairs': 'Million Veteran Program (MVP)',
        
        # Cardiovascular conditions
        'mitral valve prolapse': 'Mitral valve prolapse',
        'mitral annular disjunction': 'Mitral annular disjunction',
        'mitral regurgitation': 'Mitral regurgitation',
        'cardiovascular disease': 'Cardiovascular disease',
        'cardiovascular diseases': 'Cardiovascular disease',
        
        # Other common variations
        'machine learning': 'Machine learning',
        'artificial intelligence': 'Artificial intelligence',
        'deep learning': 'Deep learning',
        'precision medicine': 'Precision medicine',
        'personalized medicine': 'Precision medicine',
        'electronic health record': 'Electronic health records',
        'electronic health records': 'Electronic health records',
        'ehr': 'Electronic health records',
        'ehrs': 'Electronic health records',
        
        # Disease terms
        'diabetes mellitus': 'Diabetes',
        'type 2 diabetes': 'Type 2 diabetes',
        'type 1 diabetes': 'Type 1 diabetes',
        'alzheimer disease': "Alzheimer's disease",
        'alzheimer\'s disease': "Alzheimer's disease",
        'alzheimers disease': "Alzheimer's disease",
        'dementia': 'Dementia',
        'depression': 'Depression',
        'obesity': 'Obesity',
        'hypertension': 'Hypertension',
        'cancer': 'Cancer',
        
        # Methodology terms
        'cohort study': 'Cohort study',
        'case-control study': 'Case-control study',
        'longitudinal study': 'Longitudinal study',
        'cross-sectional study': 'Cross-sectional study',
        'meta-analysis': 'Meta-analysis',
        'systematic review': 'Systematic review',
        'clinical trial': 'Clinical trial',
        'randomized controlled trial': 'Randomized controlled trial',
        
        # Genetics terms
        'genetics': 'Genetics',
        'genomics': 'Genomics',
        'pharmacogenetics': 'Pharmacogenetics',
        'pharmacogenomics': 'Pharmacogenomics',
        'polygenic score': 'Polygenic risk score',
        'polygenic risk score': 'Polygenic risk score',
        'genetic variant': 'Genetic variants',
        'genetic variants': 'Genetic variants',
        'single nucleotide polymorphism': 'Single nucleotide polymorphisms',
        'single nucleotide polymorphisms': 'Single nucleotide polymorphisms',
        'snp': 'Single nucleotide polymorphisms',
        'snps': 'Single nucleotide polymorphisms',
    }
    
    return normalization_map

def normalize_keyword(keyword, normalization_map):
    """
    Enhanced keyword normalization with comprehensive mapping
    """
    if not keyword or keyword.isspace():
        return ""
    
    # Clean and lowercase for comparison
    clean_keyword = keyword.strip().lower()
    
    # Direct mapping lookup
    if clean_keyword in normalization_map:
        return normalization_map[clean_keyword]
    
    # Partial matching for complex terms
    for pattern, normalized_form in normalization_map.items():
        if pattern in clean_keyword and len(pattern) > 3:  # Avoid short false matches
            return normalized_form
    
    # If no mapping found, return properly capitalized version
    return ' '.join(word.capitalize() for word in keyword.strip().split())

def normalize_mesh_terms(mesh_dict, normalization_map):
    """Normalize MeSH terms using the same mapping"""
    normalized_mesh = {}
    
    for term, count in mesh_dict.items():
        normalized_term = normalize_keyword(term, normalization_map)
        if normalized_term in normalized_mesh:
            normalized_mesh[normalized_term] += count
        else:
            normalized_mesh[normalized_term] = count
    
    return normalized_mesh

def normalize_keywords_dict(keywords_dict, normalization_map):
    """Normalize keywords dictionary"""
    normalized_keywords = {}
    
    for keyword, count in keywords_dict.items():
        normalized_keyword = normalize_keyword(keyword, normalization_map)
        if normalized_keyword in normalized_keywords:
            normalized_keywords[normalized_keyword] += count
        else:
            normalized_keywords[normalized_keyword] = count
    
    return normalized_keywords

# ENHANCED DATA PARSING WITH NORMALIZATION
def parse_biobank_statistics_enhanced(file_path):
    """Enhanced parsing with keyword normalization"""
    print(f"Parsing statistics file with enhanced normalization: {file_path}")
    
    # Get normalization mapping
    normalization_map = create_keyword_normalization_mapping()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract total paper counts
    total_papers_match = re.search(r'Total papers analyzed: (\d+)', content)
    total_papers = int(total_papers_match.group(1)) if total_papers_match else None
    
    # Extract paper counts by biobank
    papers_by_biobank = {}
    biobank_papers_section = re.search(r'Papers by biobank:(.*?)(?=\n\n)', content, re.DOTALL)
    if biobank_papers_section:
        for line in biobank_papers_section.group(1).strip().split('\n'):
            match = re.search(r'\s*-\s*(.*?):\s*(\d+)\s*papers', line)
            if match:
                biobank, count = match.groups()
                papers_by_biobank[biobank.strip()] = int(count)
    
    # Extract statistics for each biobank
    biobank_stats = {}
    biobank_sections = re.findall(r'==================================================\n(.*?) STATISTICS\n==================================================\n\n(.*?)(?=\n\n==================================================|$)', content, re.DOTALL)
    
    for biobank_name, section in biobank_sections:
        biobank_name = biobank_name.strip()
        biobank_stats[biobank_name] = {}
        
        # Extract and normalize MeSH terms
        mesh_terms = {}
        mesh_section = re.search(r'1\. MOST COMMON MeSH TERMS\n----------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if mesh_section:
            for line in mesh_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*occurrences', line)
                if match:
                    term, count = match.groups()
                    mesh_terms[term.strip()] = int(count)
        
        # Normalize MeSH terms
        biobank_stats[biobank_name]['mesh_terms'] = normalize_mesh_terms(mesh_terms, normalization_map)
        
        # Extract and normalize keywords
        keywords = {}
        keywords_section = re.search(r'2\. MOST COMMON AUTHOR KEYWORDS\n----------------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if keywords_section:
            for line in keywords_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*occurrences', line)
                if match:
                    keyword, count = match.groups()
                    keywords[keyword.strip()] = int(count)
        
        # Normalize keywords
        biobank_stats[biobank_name]['keywords'] = normalize_keywords_dict(keywords, normalization_map)
        
        # Extract authors (no normalization needed)
        authors = {}
        authors_section = re.search(r'3\. MOST PROLIFIC AUTHORS\n-----------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if authors_section:
            for line in authors_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*publications', line)
                if match:
                    author, count = match.groups()
                    authors[author.strip()] = int(count)
        biobank_stats[biobank_name]['authors'] = authors
        
        # Extract institutions (no normalization needed)
        institutions = {}
        institutions_section = re.search(r'4\. LEADING INSTITUTIONS\n----------------------\n(.*?)(?=\n\n|$)', section, re.DOTALL)
        if institutions_section:
            for line in institutions_section.group(1).strip().split('\n'):
                match = re.search(r'\s*\d+\.\s*(.*?):\s*(\d+)\s*publications', line)
                if match:
                    institution, count = match.groups()
                    institutions[institution.strip()] = int(count)
        biobank_stats[biobank_name]['institutions'] = institutions
    
    return {
        'total_papers': total_papers,
        'papers_by_biobank': papers_by_biobank,
        'biobank_stats': biobank_stats
    }

# TOP CITED ARTICLES ANALYSIS
def analyze_top_cited_articles(csv_file_path):
    """Analyze top cited articles for each biobank - true all-time most cited (no filters)"""
    print("Analyzing most cited articles of all time for each biobank (no filtering)...")
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return {}
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Ensure Citation Count column exists and is numeric
    if 'Citation Count' not in df.columns:
        print("Citation Count column not found in CSV")
        return {}
    
    df['Citation Count'] = pd.to_numeric(df['Citation Count'], errors='coerce').fillna(0)
    
    # Convert Year to numeric for display
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Define biobanks
    biobanks = ['UK Biobank', 'Million Veteran Program', 'FinnGen', 'All of Us', 'Estonian Biobank']
    
    top_cited_by_biobank = {}
    
    for biobank in biobanks:
        print(f"Processing {biobank}...")
        
        # Filter articles for this biobank
        biobank_articles = df[df['Biobank'].str.contains(biobank, na=False, case=False)]
        
        if len(biobank_articles) == 0:
            print(f"  No articles found for {biobank}")
            continue
        
        # Take top 10 articles by citation count (including 0 citations)
        # Secondary sort by year (descending) to break ties
        top_cited = biobank_articles.nlargest(10, ['Citation Count', 'Year'])
        
        print(f"  Total articles for {biobank}: {len(biobank_articles)}")
        print(f"  Top 10 citation range: {top_cited['Citation Count'].min()} - {top_cited['Citation Count'].max()}")
        
        if len(top_cited) > 0:
            year_min = top_cited['Year'].min()
            year_max = top_cited['Year'].max()
            print(f"  Years represented: {int(year_min) if not pd.isna(year_min) else 'N/A'} - {int(year_max) if not pd.isna(year_max) else 'N/A'}")
        
        # Extract main authors (first 3 authors)
        def extract_main_authors(authors_str):
            if pd.isna(authors_str):
                return "Unknown"
            authors = str(authors_str).split(';')
            main_authors = [author.strip() for author in authors[:3]]  # First 3 authors, stripped
            if len(authors) > 3:
                return '; '.join(main_authors) + ' et al.'
            else:
                return '; '.join(main_authors)
        
        top_cited = top_cited.copy()  # Avoid SettingWithCopyWarning
        top_cited['Main Authors'] = top_cited['Authors'].apply(extract_main_authors)
        
        # Create clean table
        citation_table = top_cited[['Title', 'Main Authors', 'Journal', 'Year', 'Citation Count']].copy()
        citation_table['Rank'] = range(1, len(citation_table) + 1)
        citation_table = citation_table[['Rank', 'Title', 'Main Authors', 'Journal', 'Year', 'Citation Count']]
        
        top_cited_by_biobank[biobank] = citation_table
    
    return top_cited_by_biobank

def create_top_cited_tables(top_cited_data):
    """Create publication-quality tables for top cited articles"""
    print("Creating top cited articles tables...")
    
    for biobank, citation_table in top_cited_data.items():
        if citation_table.empty:
            print(f"  Skipping {biobank} - no cited articles found")
            continue
        
        print(f"  Creating table for {biobank} with {len(citation_table)} articles")
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(16, max(8, len(citation_table) * 0.8)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        headers = ['Rank', 'Title', 'Main Authors', 'Journal', 'Year', 'Citations']
        
        for _, row in citation_table.iterrows():
            # Truncate title if too long
            title = row['Title'][:80] + '...' if len(str(row['Title'])) > 80 else row['Title']
            
            # Ensure we have valid data
            year_display = int(row['Year']) if not pd.isna(row['Year']) else 'N/A'
            citations_display = int(row['Citation Count']) if not pd.isna(row['Citation Count']) else 0
            
            table_data.append([
                int(row['Rank']),
                title,
                row['Main Authors'],
                row['Journal'],
                year_display,
                citations_display
            ])
        
        # Create table
        table = ax.table(cellText=table_data, 
                        colLabels=headers,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.06, 0.4, 0.25, 0.15, 0.06, 0.08])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        # Add subtitle with citation info
        min_citations = citation_table['Citation Count'].min()
        max_citations = citation_table['Citation Count'].max()
        min_year = citation_table['Year'].min()
        max_year = citation_table['Year'].max()
        
        title_text = f'Top {len(citation_table)} Most Cited Articles of All Time - {biobank}'
        if min_citations == max_citations == 0:
            subtitle_text = f'(All articles have 0 citations | Years: {int(min_year) if not pd.isna(min_year) else "N/A"} - {int(max_year) if not pd.isna(max_year) else "N/A"})'
        else:
            subtitle_text = f'(Citations: {int(min_citations)} - {int(max_citations)} | Years: {int(min_year) if not pd.isna(min_year) else "N/A"} - {int(max_year) if not pd.isna(max_year) else "N/A"})'
        
        plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.95)
        plt.title(subtitle_text, fontsize=12, style='italic', pad=10)
        
        plt.savefig(os.path.join(output_dir, f'{biobank.replace(" ", "_")}_top_cited_articles.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save as CSV
        citation_table.to_csv(
            os.path.join(output_dir, f'{biobank.replace(" ", "_")}_top_cited_articles.csv'), 
            index=False
        )

# ENHANCED INSTITUTION ANALYSIS - BAR PLOTS INSTEAD OF PIE CHARTS
def create_institution_bar_plots(data):
    """Create bar plots for top institutions (replacing pie charts)"""
    print("Creating institution bar plots...")
    
    for biobank, stats in data['biobank_stats'].items():
        if not stats['institutions']:
            continue
        
        # Get top 10 institutions
        top_institutions = dict(sorted(stats['institutions'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10])
        
        if not top_institutions:
            continue
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        institutions = list(top_institutions.keys())
        counts = list(top_institutions.values())
        
        # Truncate long institution names
        institutions_short = [inst[:50] + '...' if len(inst) > 50 else inst for inst in institutions]
        
        bars = ax.barh(institutions_short, counts, 
                      color=BIOBANK_COLORS.get(biobank, '#1f77b4'), alpha=0.8)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + max(counts)*0.01, i, f'{count}', 
                   va='center', ha='left', fontweight='bold')
        
        ax.set_xlabel('Number of Publications', fontsize=14, fontweight='bold')
        ax.set_title(f'Top 10 Leading Institutions - {biobank}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{biobank.replace(" ", "_")}_top_institutions_bar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# ENHANCED KEYWORD ANALYSIS FOR ANNUAL REVIEW
def create_research_themes_analysis(data):
    """Create comprehensive research themes analysis for Annual Review"""
    print("Creating research themes analysis...")
    
    # Combined keyword analysis across all biobanks
    all_keywords = Counter()
    biobank_keyword_matrix = {}
    
    for biobank, stats in data['biobank_stats'].items():
        biobank_keyword_matrix[biobank] = stats['keywords']
        for keyword, count in stats['keywords'].items():
            all_keywords[keyword] += count
    
    # Get top 20 keywords overall
    top_keywords = dict(all_keywords.most_common(20))
    
    # Create heatmap of top keywords across biobanks
    keyword_df = pd.DataFrame(biobank_keyword_matrix).fillna(0)
    top_keyword_df = keyword_df.loc[keyword_df.index.isin(top_keywords.keys())]
    
    # Sort by total frequency
    top_keyword_df['Total'] = top_keyword_df.sum(axis=1)
    top_keyword_df = top_keyword_df.sort_values('Total', ascending=False).drop('Total', axis=1)
    
    # Create publication-quality heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(top_keyword_df, 
                annot=True, 
                fmt='g',
                cmap='YlOrRd',
                linewidths=0.5,
                ax=ax,
                cbar_kws={'label': 'Keyword Frequency'})
    
    ax.set_title('Research Keywords Distribution Across Biobanks', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Biobank', fontsize=14, fontweight='bold')
    ax.set_ylabel('Research Keywords', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'research_keywords_heatmap.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save keyword data
    top_keyword_df.to_csv(os.path.join(output_dir, 'research_keywords_matrix.csv'))

# DIAGNOSTIC FUNCTION FOR CITATION DATA
def analyze_citation_data_quality(csv_file_path):
    """Analyze the quality of citation data to understand the issues"""
    print("Analyzing citation data quality...")
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return
    
    df = pd.read_csv(csv_file_path)
    df['Citation Count'] = pd.to_numeric(df['Citation Count'], errors='coerce').fillna(0)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    biobanks = ['UK Biobank', 'Million Veteran Program', 'FinnGen', 'All of Us', 'Estonian Biobank']
    
    citation_summary_path = os.path.join(output_dir, "citation_data_summary.txt")
    
    with open(citation_summary_path, 'w') as f:
        f.write("CITATION DATA QUALITY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for biobank in biobanks:
            biobank_articles = df[df['Biobank'].str.contains(biobank, na=False, case=False)]
            
            if len(biobank_articles) == 0:
                continue
                
            f.write(f"{biobank.upper()}:\n")
            f.write(f"  Total articles: {len(biobank_articles)}\n")
            
            # Citation statistics
            total_with_citations = len(biobank_articles[biobank_articles['Citation Count'] > 0])
            total_zero_citations = len(biobank_articles[biobank_articles['Citation Count'] == 0])
            
            f.write(f"  Articles with citations > 0: {total_with_citations}\n")
            f.write(f"  Articles with 0 citations: {total_zero_citations}\n")
            
            if total_with_citations > 0:
                cited_articles = biobank_articles[biobank_articles['Citation Count'] > 0]
                f.write(f"  Citation range: {cited_articles['Citation Count'].min()} - {cited_articles['Citation Count'].max()}\n")
                f.write(f"  Mean citations: {cited_articles['Citation Count'].mean():.1f}\n")
                f.write(f"  Median citations: {cited_articles['Citation Count'].median():.1f}\n")
            
            # Year distribution
            f.write(f"  Year range: {biobank_articles['Year'].min()} - {biobank_articles['Year'].max()}\n")
            
            # Articles by year
            year_counts = biobank_articles['Year'].value_counts().sort_index()
            f.write(f"  Articles by year: ")
            for year, count in year_counts.tail(5).items():  # Last 5 years
                if not pd.isna(year):
                    f.write(f"{int(year)}({count}) ")
            f.write(f"\n")
            
            # High-citation articles
            high_cited = biobank_articles[biobank_articles['Citation Count'] >= 100]
            f.write(f"  Articles with 100+ citations: {len(high_cited)}\n")
            
            f.write(f"\n")
    
    print(f"Citation data summary saved to: {citation_summary_path}")

# Updated research questions analysis
def answer_research_questions(data, top_cited_data):
    """Answer the four specific research questions with improved citation analysis"""
    print("Generating answers to research questions...")
    
    report_path = os.path.join(output_dir, "research_questions_analysis.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BIOBANK RESEARCH ANALYSIS - ANNUAL REVIEW\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("NOTE: Citation analysis shows the top 10 articles by citation count\n")
        f.write("for each biobank (including articles with 0 citations if necessary).\n")
        f.write("Articles are ranked by citation count, with publication year as tiebreaker.\n\n")
        
        # Question 1: Most common keywords subjects
        f.write("1. SUBJECT OF MOST COMMONLY OCCURRING KEYWORDS BY BIOBANK\n")
        f.write("-" * 60 + "\n\n")
        
        for biobank, stats in data['biobank_stats'].items():
            f.write(f"{biobank.upper()}:\n")
            top_keywords = dict(sorted(stats['keywords'].items(), 
                                     key=lambda x: x[1], reverse=True)[:15])
            
            # Categorize keywords by theme
            genetic_terms = []
            clinical_terms = []
            methodology_terms = []
            disease_terms = []
            
            for keyword, count in top_keywords.items():
                keyword_lower = keyword.lower()
                if any(term in keyword_lower for term in ['genetic', 'gwas', 'mendelian', 'genome', 'snp', 'polygenic']):
                    genetic_terms.append(f"{keyword} ({count})")
                elif any(term in keyword_lower for term in ['disease', 'clinical', 'patient', 'treatment', 'diagnosis']):
                    clinical_terms.append(f"{keyword} ({count})")
                elif any(term in keyword_lower for term in ['study', 'analysis', 'method', 'cohort', 'population']):
                    methodology_terms.append(f"{keyword} ({count})")
                elif any(term in keyword_lower for term in ['diabetes', 'cancer', 'cardiovascular', 'depression', 'dementia']):
                    disease_terms.append(f"{keyword} ({count})")
            
            if genetic_terms:
                f.write(f"  Genetic Research: {'; '.join(genetic_terms[:5])}\n")
            if clinical_terms:
                f.write(f"  Clinical Applications: {'; '.join(clinical_terms[:5])}\n")
            if disease_terms:
                f.write(f"  Disease Focus: {'; '.join(disease_terms[:5])}\n")
            if methodology_terms:
                f.write(f"  Methodology: {'; '.join(methodology_terms[:3])}\n")
            f.write(f"\n")
        
        # Question 2: Most cited papers subjects (top 10 by citation count)
        f.write("\n2. SUBJECT OF MOST CITED PAPERS BY BIOBANK (TOP 10 BY CITATION COUNT)\n")
        f.write("-" * 60 + "\n\n")
        
        for biobank, citation_table in top_cited_data.items():
            if citation_table.empty:
                f.write(f"{biobank.upper()}: No articles found\n\n")
                continue
                
            f.write(f"{biobank.upper()} (Top {len(citation_table)} articles by citation count):\n")
            f.write(f"  Most cited article ({citation_table.iloc[0]['Citation Count']} citations):\n")
            f.write(f"    {citation_table.iloc[0]['Title'][:100]}...\n")
            f.write(f"    Authors: {citation_table.iloc[0]['Main Authors']}\n")
            f.write(f"    Journal: {citation_table.iloc[0]['Journal']} ({citation_table.iloc[0]['Year']})\n")
            
            # Add citation range and year span info
            min_cit = citation_table['Citation Count'].min()
            max_cit = citation_table['Citation Count'].max()
            min_year = citation_table['Year'].min()
            max_year = citation_table['Year'].max()
            f.write(f"  Citation range: {int(min_cit)} - {int(max_cit)}\n")
            f.write(f"  Publication years: {int(min_year) if not pd.isna(min_year) else 'N/A'} - {int(max_year) if not pd.isna(max_year) else 'N/A'}\n\n")
        
        # Question 3: Top 20 prolific authors
        f.write("\n3. TOP 20 MOST PROLIFIC AUTHORS BY BIOBANK\n")
        f.write("-" * 60 + "\n\n")
        
        for biobank, stats in data['biobank_stats'].items():
            f.write(f"{biobank.upper()}:\n")
            top_authors = dict(sorted(stats['authors'].items(), 
                                    key=lambda x: x[1], reverse=True)[:20])
            for i, (author, count) in enumerate(top_authors.items(), 1):
                f.write(f"  {i:2d}. {author}: {count} publications\n")
            f.write(f"\n")
        
        # Question 4: Top 10 leading institutions
        f.write("\n4. TOP 10 LEADING INSTITUTIONS BY BIOBANK\n")
        f.write("-" * 60 + "\n\n")
        
        for biobank, stats in data['biobank_stats'].items():
            f.write(f"{biobank.upper()}:\n")
            top_institutions = dict(sorted(stats['institutions'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10])
            for i, (institution, count) in enumerate(top_institutions.items(), 1):
                f.write(f"  {i:2d}. {institution}: {count} publications\n")
            f.write(f"\n")

# MAIN EXECUTION
def main():
    """Main execution function for Annual Review analysis"""
    print("=" * 60)
    print("ANNUAL REVIEW BIOBANK ANALYSIS")
    print("Challenges and Opportunities Using Global EHR Linked Biobanks")
    print("=" * 60)
    
    # Check input files
    if not os.path.exists(input_file):
        print(f"Error: Statistics file not found at {input_file}")
        print("Please run the analysis script first.")
        return
    
    # Parse enhanced biobank statistics
    data = parse_biobank_statistics_enhanced(input_file)
    print(f"Successfully parsed data for {len(data['biobank_stats'])} biobanks")
    
    # Analyze citation data quality first
    analyze_citation_data_quality(csv_input_file)
    
    # Analyze top cited articles (with improved filtering)
    top_cited_data = analyze_top_cited_articles(csv_input_file)
    
    # Create all analyses
    create_top_cited_tables(top_cited_data)
    create_institution_bar_plots(data)
    create_research_themes_analysis(data)
    answer_research_questions(data, top_cited_data)
    
    print(f"\n✅ Annual Review analysis complete!")
    print(f"📊 Results saved to: ANALYSIS/00-02-STAT-VISUALIZATION")
    print(f"\nGenerated files:")
    print(f"  - Top cited articles tables (PNG & CSV) - Top 10 by citation count")
    print(f"  - Institution bar plots (replacing pie charts)")
    print(f"  - Research themes heatmap")
    print(f"  - Research questions analysis report")
    print(f"  - Citation data quality summary")
    print(f"  - Keyword normalization applied throughout")
    print(f"\nNote: Now shows top 10 articles by citation count for each biobank")
    print(f"      (includes articles with 0 citations if they rank in top 10).")

if __name__ == "__main__":
    main()