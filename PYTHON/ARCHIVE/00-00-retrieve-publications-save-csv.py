import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import warnings
import time
from Bio import Entrez
import xml.etree.ElementTree as ET
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "ANALYSIS", "BIOBANK-RESULTS")
os.makedirs(output_dir, exist_ok=True)

# Configure Entrez (REPLACE WITH YOUR EMAIL)
Entrez.email = "mc.admin@manuelcorpas.com"  # REQUIRED: Replace with your email
Entrez.tool = "BiobankAnalysis"

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

# Define the five biobanks and their search terms
BIOBANK_DEFINITIONS = {
    'UK Biobank': [
        'UK Biobank', 'United Kingdom Biobank', 'U.K. Biobank', 
        'UK-Biobank', 'British Biobank'
    ],
    'Million Veteran Program': [
        'Million Veteran Program', 'MVP', 'Veterans Affairs', 
        'VA Million Veteran', 'Million Veterans Program'
    ],
    'FinnGen': [
        'FinnGen', 'Finnish Genome', 'Finland Genome', 
        'Finnish Biobank', 'FinnGen biobank'
    ],
    'All of Us': [
        'All of Us', 'All of Us Research Program', 'AoU', 
        'Precision Medicine Initiative', 'PMI Cohort'
    ],
    'Estonian Biobank': [
        'Estonian Biobank', 'Estonia Biobank', 'Estonian Genome Center',
        'Tartu Biobank', 'Estonian Health'
    ]
}

def create_keyword_normalization_mapping():
    """Create comprehensive keyword normalization mapping"""
    normalization_map = {
        # Biobank variations
        'biobank': 'Biobank',
        'biobanks': 'Biobank', 
        'bio-bank': 'Biobank',
        'bio-banks': 'Biobank',
        
        # GWAS variations
        'gwas': 'GWAS',
        'genome-wide association study': 'GWAS',
        'genome-wide association studies': 'GWAS',
        'genome wide association study': 'GWAS',
        'genome wide association studies': 'GWAS',
        'genomewide association study': 'GWAS',
        'genomewide association studies': 'GWAS',
        
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
        'mvp': 'Million Veteran Program',
        'million veteran program': 'Million Veteran Program',
        'million veteran programme': 'Million Veteran Program',
        'veterans affairs': 'Million Veteran Program',
        
        # Cardiovascular conditions
        'mitral valve prolapse': 'Mitral valve prolapse',
        'mitral annular disjunction': 'Mitral annular disjunction',
        'mitral regurgitation': 'Mitral regurgitation',
        'cardiovascular disease': 'Cardiovascular disease',
        'cardiovascular diseases': 'Cardiovascular disease',
        
        # EHR variations
        'electronic health record': 'Electronic health records',
        'electronic health records': 'Electronic health records',
        'ehr': 'Electronic health records',
        'ehrs': 'Electronic health records',
        'electronic medical record': 'Electronic health records',
        'electronic medical records': 'Electronic health records',
        'emr': 'Electronic health records',
        'emrs': 'Electronic health records',
        
        # Other common variations
        'machine learning': 'Machine learning',
        'artificial intelligence': 'Artificial intelligence',
        'deep learning': 'Deep learning',
        'precision medicine': 'Precision medicine',
        'personalized medicine': 'Precision medicine',
        'personalised medicine': 'Precision medicine',
        
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
    """Enhanced keyword normalization with comprehensive mapping"""
    if not keyword or keyword.isspace():
        return ""
    
    clean_keyword = keyword.strip().lower()
    
    # Direct mapping lookup
    if clean_keyword in normalization_map:
        return normalization_map[clean_keyword]
    
    # Partial matching for complex terms
    for pattern, normalized_form in normalization_map.items():
        if pattern in clean_keyword and len(pattern) > 3:
            return normalized_form
    
    # If no mapping found, return properly capitalized version
    return ' '.join(word.capitalize() for word in keyword.strip().split())

def create_biobank_search_query(biobank_name):
    """Create a PubMed search query for a specific biobank + EHR"""
    biobank_terms = BIOBANK_DEFINITIONS[biobank_name]
    
    # Create biobank part of query
    biobank_query = ' OR '.join([f'"{term}"[All Fields]' for term in biobank_terms])
    
    # EHR terms
    ehr_query = '("electronic health record"[All Fields] OR "electronic health records"[All Fields] OR "EHR"[All Fields] OR "EHRs"[All Fields] OR "electronic medical record"[All Fields] OR "electronic medical records"[All Fields] OR "EMR"[All Fields] OR "EMRs"[All Fields])'
    
    # Combine with AND
    full_query = f'({biobank_query}) AND {ehr_query} AND ("2000"[PDAT] : "2025"[PDAT])'
    
    return full_query

def search_pubmed_for_biobank(biobank_name, max_results=1000):
    """Search PubMed for articles mentioning a specific biobank + EHR"""
    logger.info(f"Searching PubMed for {biobank_name}...")
    
    query = create_biobank_search_query(biobank_name)
    logger.info(f"Query: {query}")
    
    try:
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        search_results = Entrez.read(handle)
        handle.close()
        
        pmids = search_results["IdList"]
        logger.info(f"Found {len(pmids)} articles for {biobank_name}")
        
        if not pmids:
            return []
        
        # Fetch detailed information in batches
        batch_size = 100
        articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1} for {biobank_name}")
            
            try:
                # Fetch article details
                handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="xml", retmode="xml")
                records = handle.read()
                handle.close()
                
                # Parse XML
                root = ET.fromstring(records)
                
                for article in root.findall('.//PubmedArticle'):
                    article_data = parse_pubmed_article(article, biobank_name)
                    if article_data:
                        articles.append(article_data)
                
                # Be respectful to NCBI servers
                time.sleep(0.34)  # ~3 requests per second
                
            except Exception as e:
                logger.error(f"Error fetching batch for {biobank_name}: {e}")
                continue
        
        return articles
        
    except Exception as e:
        logger.error(f"Error searching PubMed for {biobank_name}: {e}")
        return []

def parse_pubmed_article(article_xml, biobank_name):
    """Parse a single PubMed article XML"""
    try:
        article_data = {'Biobank': biobank_name}
        
        # PMID
        pmid_elem = article_xml.find('.//PMID')
        article_data['PMID'] = pmid_elem.text if pmid_elem is not None else ''
        
        # Title
        title_elem = article_xml.find('.//ArticleTitle')
        article_data['Title'] = title_elem.text if title_elem is not None else ''
        
        # Abstract
        abstract_parts = []
        for abstract_text in article_xml.findall('.//AbstractText'):
            if abstract_text.text:
                abstract_parts.append(abstract_text.text)
        article_data['Abstract'] = ' '.join(abstract_parts)
        
        # Authors
        authors = []
        for author in article_xml.findall('.//Author'):
            lastname = author.find('.//LastName')
            forename = author.find('.//ForeName')
            if lastname is not None and forename is not None:
                authors.append(f"{lastname.text}, {forename.text}")
            elif lastname is not None:
                authors.append(lastname.text)
        article_data['Authors'] = '; '.join(authors)
        
        # Journal
        journal_elem = article_xml.find('.//Journal/Title')
        if journal_elem is None:
            journal_elem = article_xml.find('.//Journal/ISOAbbreviation')
        article_data['Journal'] = journal_elem.text if journal_elem is not None else ''
        
        # Publication Year
        year_elem = article_xml.find('.//PubDate/Year')
        if year_elem is None:
            year_elem = article_xml.find('.//PubDate/MedlineDate')
            if year_elem is not None and year_elem.text:
                # Extract year from MedlineDate (e.g., "2020 Jan-Feb" -> "2020")
                year_match = re.search(r'(\d{4})', year_elem.text)
                article_data['Year'] = year_match.group(1) if year_match else ''
            else:
                article_data['Year'] = ''
        else:
            article_data['Year'] = year_elem.text
        
        # MeSH Terms
        mesh_terms = []
        for mesh in article_xml.findall('.//MeshHeading/DescriptorName'):
            if mesh.text:
                mesh_terms.append(mesh.text)
        article_data['MeSH_Terms'] = '; '.join(mesh_terms)
        
        # Keywords
        keywords = []
        for keyword in article_xml.findall('.//Keyword'):
            if keyword.text:
                keywords.append(keyword.text)
        article_data['Keywords'] = '; '.join(keywords)
        
        # Affiliations
        affiliations = []
        for affiliation in article_xml.findall('.//Affiliation'):
            if affiliation.text:
                affiliations.append(affiliation.text)
        article_data['Affiliations'] = '; '.join(affiliations)
        
        # DOI
        doi_elem = article_xml.find('.//ArticleId[@IdType="doi"]')
        article_data['DOI'] = doi_elem.text if doi_elem is not None else ''
        
        return article_data
        
    except Exception as e:
        logger.error(f"Error parsing article: {e}")
        return None

def retrieve_all_biobank_data():
    """Retrieve data for all biobanks from PubMed"""
    logger.info("Starting comprehensive biobank data retrieval from PubMed...")
    
    all_articles = []
    biobank_counts = {}
    
    for biobank_name in BIOBANK_DEFINITIONS.keys():
        articles = search_pubmed_for_biobank(biobank_name, max_results=2000)
        biobank_counts[biobank_name] = len(articles)
        all_articles.extend(articles)
        
        # Save individual biobank data
        if articles:
            df = pd.DataFrame(articles)
            filename = f"{biobank_name.replace(' ', '_')}_articles.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)
            logger.info(f"Saved {len(articles)} articles for {biobank_name}")
    
    # Save combined dataset
    if all_articles:
        combined_df = pd.DataFrame(all_articles)
        combined_df.to_csv(os.path.join(output_dir, "all_biobank_articles.csv"), index=False)
        logger.info(f"Saved combined dataset with {len(all_articles)} total articles")
    
    return combined_df, biobank_counts

def analyze_biobank_statistics(df):
    """Analyze biobank data and create statistics"""
    logger.info("Analyzing biobank statistics...")
    
    normalization_map = create_keyword_normalization_mapping()
    biobank_stats = {}
    
    for biobank in BIOBANK_DEFINITIONS.keys():
        biobank_articles = df[df['Biobank'] == biobank]
        
        if len(biobank_articles) == 0:
            continue
        
        logger.info(f"Analyzing {len(biobank_articles)} articles for {biobank}")
        
        # Process MeSH terms
        mesh_counter = Counter()
        for mesh_string in biobank_articles['MeSH_Terms'].dropna():
            if mesh_string.strip():
                terms = [term.strip() for term in mesh_string.split(';')]
                for term in terms:
                    if term:
                        normalized_term = normalize_keyword(term, normalization_map)
                        mesh_counter[normalized_term] += 1
        
        # Process Keywords
        keyword_counter = Counter()
        for keyword_string in biobank_articles['Keywords'].dropna():
            if keyword_string.strip():
                terms = [term.strip() for term in keyword_string.split(';')]
                for term in terms:
                    if term:
                        normalized_term = normalize_keyword(term, normalization_map)
                        keyword_counter[normalized_term] += 1
        
        # Process Authors
        author_counter = Counter()
        for author_string in biobank_articles['Authors'].dropna():
            if author_string.strip():
                authors = [author.strip() for author in author_string.split(';')]
                for author in authors:
                    if author and len(author) > 2:  # Filter out initials
                        author_counter[author] += 1
        
        # Process Institutions from affiliations
        institution_counter = Counter()
        for affiliation_string in biobank_articles['Affiliations'].dropna():
            if affiliation_string.strip():
                affiliations = [aff.strip() for aff in affiliation_string.split(';')]
                for affiliation in affiliations:
                    if affiliation:
                        # Extract institution name (simplified)
                        institution = extract_institution_name(affiliation)
                        if institution:
                            institution_counter[institution] += 1
        
        biobank_stats[biobank] = {
            'total_articles': len(biobank_articles),
            'mesh_terms': dict(mesh_counter.most_common(50)),
            'keywords': dict(keyword_counter.most_common(50)),
            'authors': dict(author_counter.most_common(50)),
            'institutions': dict(institution_counter.most_common(50))
        }
    
    return biobank_stats

def extract_institution_name(affiliation):
    """Extract institution name from affiliation string"""
    if not affiliation:
        return None
    
    # Common patterns for institutions
    patterns = [
        r'University of ([^,]+)',
        r'([^,]+University)',
        r'([^,]+Institute)',
        r'([^,]+Hospital)',
        r'([^,]+Medical Center)',
        r'([^,]+College)',
        r'([^,]+School of Medicine)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, affiliation, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # Fallback: take first part before comma
    parts = affiliation.split(',')
    if parts:
        return parts[0].strip()
    
    return None

def create_publication_trends(df):
    """Analyze publication trends over time"""
    logger.info("Creating publication trends analysis...")
    
    # Convert year to numeric
    df['Year_Numeric'] = pd.to_numeric(df['Year'], errors='coerce')
    df_years = df[(df['Year_Numeric'] >= 2000) & (df['Year_Numeric'] <= 2025)]
    
    # Create trends by biobank
    biobanks = list(BIOBANK_DEFINITIONS.keys())
    years = range(2000, 2026)
    
    trend_data = {'Year': list(years)}
    
    for biobank in biobanks:
        biobank_articles = df_years[df_years['Biobank'] == biobank]
        yearly_counts = biobank_articles['Year_Numeric'].value_counts().sort_index()
        
        # Fill missing years with 0
        biobank_yearly = [yearly_counts.get(year, 0) for year in years]
        trend_data[biobank] = biobank_yearly
    
    trends_df = pd.DataFrame(trend_data)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Full timeline
    for biobank in biobanks:
        ax1.plot(trends_df['Year'], trends_df[biobank], 
                marker='o', linewidth=2, markersize=4,
                label=biobank, color=BIOBANK_COLORS.get(biobank, '#gray'))
    
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Number of Publications', fontweight='bold')
    ax1.set_title('Publication Trends by Biobank (2000-2025)', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recent trends (2015-2025)
    recent_trends = trends_df[trends_df['Year'] >= 2015]
    
    for biobank in biobanks:
        ax2.plot(recent_trends['Year'], recent_trends[biobank], 
                marker='o', linewidth=3, markersize=6,
                label=biobank, color=BIOBANK_COLORS.get(biobank, '#gray'))
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Number of Publications', fontweight='bold')
    ax2.set_title('Recent Publication Growth (2015-2025)', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "publication_trends.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    trends_df.to_csv(os.path.join(output_dir, "publication_trends.csv"), index=False)
    
    return trends_df

def create_keyword_heatmap(biobank_stats):
    """Create keyword heatmap across biobanks"""
    logger.info("Creating keyword heatmap...")
    
    # Combine all keywords
    all_keywords = Counter()
    for biobank, stats in biobank_stats.items():
        for keyword, count in stats['keywords'].items():
            all_keywords[keyword] += count
    
    # Get top 20 keywords
    top_keywords = [kw for kw, _ in all_keywords.most_common(20)]
    
    # Create matrix
    matrix_data = []
    biobank_names = list(biobank_stats.keys())
    
    for keyword in top_keywords:
        row = []
        for biobank in biobank_names:
            count = biobank_stats[biobank]['keywords'].get(keyword, 0)
            row.append(count)
        matrix_data.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(matrix_data, 
                xticklabels=[bn.replace(' ', '\n') for bn in biobank_names],
                yticklabels=top_keywords,
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                linewidths=0.5,
                ax=ax,
                cbar_kws={'label': 'Keyword Frequency'})
    
    ax.set_title('Research Keywords Distribution Across Biobanks', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Biobank', fontsize=14, fontweight='bold')
    ax.set_ylabel('Research Keywords', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_heatmap.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_institution_bar_plots(biobank_stats):
    """Create bar plots for top institutions by biobank"""
    logger.info("Creating institution bar plots...")
    
    for biobank, stats in biobank_stats.items():
        if not stats['institutions']:
            continue
        
        # Get top 10 institutions
        top_institutions = dict(list(stats['institutions'].items())[:10])
        
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
        plt.savefig(os.path.join(output_dir, f'{biobank.replace(" ", "_")}_institutions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_report(df, biobank_stats, biobank_counts):
    """Create comprehensive summary report"""
    logger.info("Creating summary report...")
    
    report_path = os.path.join(output_dir, "biobank_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BIOBANK RESEARCH ANALYSIS - COMPREHENSIVE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Articles Retrieved: {len(df)}\n\n")
        
        # Articles by biobank
        f.write("ARTICLES BY BIOBANK\n")
        f.write("-" * 30 + "\n")
        for biobank, count in biobank_counts.items():
            f.write(f"{biobank}: {count} articles\n")
        f.write(f"\n")
        
        # Question 1: Most common keywords by biobank
        f.write("1. MOST COMMONLY OCCURRING KEYWORDS BY BIOBANK\n")
        f.write("-" * 50 + "\n\n")
        
        for biobank, stats in biobank_stats.items():
            f.write(f"{biobank.upper()}:\n")
            top_keywords = dict(list(stats['keywords'].items())[:15])
            
            for i, (keyword, count) in enumerate(top_keywords.items(), 1):
                f.write(f"  {i:2d}. {keyword}: {count} occurrences\n")
            f.write(f"\n")
        
        # Question 2: Top authors
        f.write("2. TOP 20 MOST PROLIFIC AUTHORS BY BIOBANK\n")
        f.write("-" * 45 + "\n\n")
        
        for biobank, stats in biobank_stats.items():
            f.write(f"{biobank.upper()}:\n")
            top_authors = dict(list(stats['authors'].items())[:20])
            
            for i, (author, count) in enumerate(top_authors.items(), 1):
                f.write(f"  {i:2d}. {author}: {count} publications\n")
            f.write(f"\n")
        
        # Question 3: Top institutions
        f.write("3. TOP 10 LEADING INSTITUTIONS BY BIOBANK\n")
        f.write("-" * 42 + "\n\n")
        
        for biobank, stats in biobank_stats.items():
            f.write(f"{biobank.upper()}:\n")
            top_institutions = dict(list(stats['institutions'].items())[:10])
            
            for i, (institution, count) in enumerate(top_institutions.items(), 1):
                f.write(f"  {i:2d}. {institution}: {count} publications\n")
            f.write(f"\n")
        
        # Question 4: Most common MeSH terms
        f.write("4. MOST COMMON MESH TERMS BY BIOBANK\n")
        f.write("-" * 38 + "\n\n")
        
        for biobank, stats in biobank_stats.items():
            f.write(f"{biobank.upper()}:\n")
            top_mesh = dict(list(stats['mesh_terms'].items())[:15])
            
            for i, (mesh_term, count) in enumerate(top_mesh.items(), 1):
                f.write(f"  {i:2d}. {mesh_term}: {count} occurrences\n")
            f.write(f"\n")

def main():
    """Main execution function"""
    print("=" * 60)
    print("BIOBANK RESEARCH ANALYSIS - PUBMED DATA RETRIEVAL")
    print("Challenges and Opportunities Using Global EHR Linked Biobanks")
    print("=" * 60)
    
    # IMPORTANT: Check email configuration
    if Entrez.email == "your.email@institution.edu":
        print("\n❌ ERROR: Please set your email address in Entrez.email")
        print("This is required by NCBI for API access.")
        print("Edit the script and replace 'your.email@institution.edu' with your actual email.")
        return
    
    print(f"\nUsing email: {Entrez.email}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Retrieve data from PubMed
    print("\n📡 Retrieving data from PubMed...")
    df, biobank_counts = retrieve_all_biobank_data()
    
    if df.empty:
        print("❌ No data retrieved. Please check your internet connection and try again.")
        return
    
    print(f"✅ Retrieved {len(df)} total articles")
    
    # Step 2: Analyze biobank statistics
    print("\n📊 Analyzing biobank statistics...")
    biobank_stats = analyze_biobank_statistics(df)
    
    # Step 3: Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Publication trends
    trends_df = create_publication_trends(df)
    
    # Keyword heatmap
    create_keyword_heatmap(biobank_stats)
    
    # Institution bar plots
    create_institution_bar_plots(biobank_stats)
    
    # Step 4: Create comprehensive report
    print("\n📝 Creating comprehensive report...")
    create_summary_report(df, biobank_stats, biobank_counts)
    
    print(f"\n✅ Analysis complete!")
    print(f"📂 All results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - all_biobank_articles.csv (complete dataset)")
    print(f"  - Individual biobank CSV files")
    print(f"  - publication_trends.png & .csv")
    print(f"  - keyword_heatmap.png")
    print(f"  - Institution bar plots for each biobank")
    print(f"  - biobank_analysis_report.txt (comprehensive report)")
    print(f"\n📋 Summary:")
    for biobank, count in biobank_counts.items():
        print(f"  {biobank}: {count} articles")

if __name__ == "__main__":
    main()