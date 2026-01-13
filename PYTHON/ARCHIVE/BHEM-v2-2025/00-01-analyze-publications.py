import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re

# Setup paths - make all paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up one level

# Define the input/output locations
input_dir = os.path.join(parent_dir, "DATA")
# Change to read the new file with citation counts
csv_file = os.path.join(input_dir, "00-00-ehr_biobank_articles_with_citations.csv")
output_dir = os.path.join(parent_dir, "ANALYSIS", "00-01-LITERATURE-ANALYSIS")
os.makedirs(output_dir, exist_ok=True)
stats_file = os.path.join(output_dir, "00-01-biobank_statistics.txt")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
tables_dir = os.path.join(output_dir, "tables")
os.makedirs(tables_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "analysis_script.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function to parse existing biobank statistics if available
def parse_biobank_statistics(file_path):
    """Parse the biobank statistics file if it exists"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse the content (simplified example)
            data = {"exists": True}
            # Add parsing logic here if needed
            return data
        else:
            logger.info(f"Statistics file not found at {file_path}. Will generate new analysis.")
            return None
    except Exception as e:
        logger.error(f"Error reading statistics file: {e}")
        print(f"Error reading statistics file: {e}")
        return None

# Get current year for plotting
current_year = datetime.now().year

# Biobank aliases for reference - ensure all possible variations are included
biobanks = {
    "UK Biobank": ["UK Biobank", "UKBiobank", "UK-Biobank", "UK biobank", "uk biobank", "Uk Biobank"],
    "All of Us": ["All of Us", "AllofUs", "All-of-Us", "all of us"],
    "FinnGen": ["FinnGen", "Finn-Gen", "finngen"],
    "Estonian Biobank": ["Estonian Biobank", "Estonia Biobank", "Estonian-Biobank", "estonian biobank"],
    "Million Veteran Program": ["Million Veteran Program", "MVP", "Million Veteran Programme", "million veteran program"]
}

# Data structures for analysis
biobank_papers = defaultdict(list)  # {biobank: [pmids]}
biobank_mesh_terms = defaultdict(lambda: Counter())  # {biobank: Counter(mesh_terms)}
biobank_keywords = defaultdict(lambda: Counter())  # {biobank: Counter(keywords)}
biobank_authors = defaultdict(lambda: Counter())  # {biobank: Counter(authors)}
biobank_institutions = defaultdict(lambda: Counter())  # {biobank: Counter(institutions)}
paper_citations = defaultdict(int)  # {pmid: citation_count}
paper_titles = {}  # {pmid: title}
paper_years = {}   # {pmid: year}
paper_journals = {}  # {pmid: journal}
paper_full_authors = {}  # {pmid: full_authors_list}

# Function to normalize keywords to handle variations
def normalize_keyword(keyword):
    """
    Normalize keywords with a direct approach to handle capitalization issues.
    """
    if not keyword or keyword.isspace():
        return ""
    
    # Strip whitespace
    original_keyword = keyword.strip()
    lower_keyword = original_keyword.lower()
    
    # Dictionary of special complex cases that need custom handling (regex patterns)
    special_patterns = {
        r'mendel(i|ia)n\s+random(i[sz]|is|iz)ation': 'Mendelian randomization',
        r'uk\s*bio\s*bank': 'UK Biobank',
        r'gwas': 'GWAS (Genome-wide association studies)',
        r'genome[\s-]wide\s+association\s+stud': 'GWAS (Genome-wide association studies)',
    }
    
    # Check special regex patterns first
    for pattern, normalized_form in special_patterns.items():
        if re.search(pattern, lower_keyword):
            return normalized_form
    
    # Direct mapping of terms to their preferred forms - LOWERCASE keys for comparison
    # This handles simple capitalization differences directly
    term_mapping = {
        'dementia': 'Dementia',
        'alzheimer\'s disease': 'Alzheimer\'s disease',
        'biobank': 'Biobank',
        'cardiovascular disease': 'Cardiovascular disease',
        'epidemiology': 'Epidemiology',
        'genetics': 'Genetics',
        'gut microbiota': 'Gut microbiota', 
        'physical activity': 'Physical activity',
        'depression': 'Depression',
        'mortality': 'Mortality',
        'covid-19': 'COVID-19',
        'cohort study': 'Cohort study',
        'causality': 'Causality and causal inference',
        'causal relationship': 'Causality and causal inference',
        'causal effect': 'Causality and causal inference',
        'causal association': 'Causality and causal inference',
        'causal inference': 'Causality and causal inference',
        'mvp': 'Million Veteran Program (MVP)',
        'million veteran program': 'Million Veteran Program (MVP)',
        'mitral valve prolapse': 'Mitral valve prolapse',
        'mitral annular disjunction': 'Mitral annular disjunction',
    }
    
    # Direct lookup - exactly match the lowercase version of keyword
    if lower_keyword in term_mapping:
        return term_mapping[lower_keyword]
    
    # Try partial matching for longer terms
    for term, normalized in term_mapping.items():
        if term in lower_keyword:
            return normalized
    
    # Return original with first letter of each word capitalized if no match
    words = original_keyword.split()
    if words:
        return ' '.join(word.capitalize() if not word.isupper() else word for word in words)
    
    # If all else fails, return original
    return original_keyword

# -------------------------------
# Function to process the CSV data
# -------------------------------
def process_csv_data(df):
    """Process the CSV data and populate analysis structures"""
    logger.info(f"Processing {len(df)} articles...")
    
    # Extract biobank information
    for _, row in df.iterrows():
        pmid = str(row['PMID'])
        title = row['Title']
        year = str(row['Year'])
        journal = row['Journal'] if 'Journal' in row else ''
        authors_list = row['Authors'].split('; ') if not pd.isna(row['Authors']) else []
        
        # Updated to handle 'Citation Count' column from first script
        citation_count = int(row['Citation Count']) if 'Citation Count' in row and not pd.isna(row['Citation Count']) else 0
        
        # Store basic paper info
        paper_titles[pmid] = title
        paper_years[pmid] = year
        paper_journals[pmid] = journal
        paper_full_authors[pmid] = authors_list
        paper_citations[pmid] = citation_count
        
        # Process biobanks mentioned in the paper
        biobanks_mentioned = row['Biobank'].split('; ') if not pd.isna(row['Biobank']) else []
        
        for biobank_name in biobanks_mentioned:
            biobank_name = biobank_name.strip()
            for canonical_name, aliases in biobanks.items():
                if biobank_name in aliases:
                    # Found a match, process this biobank data
                    biobank_papers[canonical_name].append({
                        'pmid': pmid,
                        'title': title,
                        'citations': citation_count,
                        'year': year,
                        'journal': journal,
                        'authors': authors_list[:3] if len(authors_list) > 3 else authors_list
                    })
                    
                    # Process MeSH terms
                    mesh_terms = row['MeSH Terms'].split('; ') if not pd.isna(row['MeSH Terms']) else []
                    for term in mesh_terms:
                        if term:
                            # Apply normalization to MeSH terms as well
                            normalized_term = normalize_keyword(term)
                            biobank_mesh_terms[canonical_name][normalized_term] += 1
                    
                    # Process keywords
                    keywords = row['Keywords'].split('; ') if not pd.isna(row['Keywords']) else []
                    
                    normalized_keywords = []
                    for keyword in keywords:
                        if keyword:
                            # Apply our new normalization function
                            normalized_keyword = normalize_keyword(keyword)
                            normalized_keywords.append(normalized_keyword)
                    
                    # Add normalized keywords to counter
                    for keyword in normalized_keywords:
                        if keyword:
                            biobank_keywords[canonical_name][keyword] += 1
                    
                    # Process authors
                    for author in authors_list:
                        if author:
                            biobank_authors[canonical_name][author] += 1
                    
                    # Process institutions - extract from affiliations
                    affiliations = row['Affiliations'].split('; ') if not pd.isna(row['Affiliations']) else []
                    for affiliation in affiliations:
                        # Simple extraction for analysis
                        parts = affiliation.split(',')
                        for part in parts:
                            part = part.strip()
                            if len(part) > 3 and any(indicator in part.lower() for indicator in 
                                   ['university', 'college', 'institute', 'hospital', 'center', 'centre']):
                                biobank_institutions[canonical_name][part] += 1
                    
                    # Once we've found a matching biobank, we can break the inner loop
                    break

# -------------------------------
# Function to create visualizations - UPDATED
# -------------------------------
def create_visualizations():
    """Create visualizations for the data analysis"""
    
    logger.info("Creating visualizations...")
    
    # Create citation distribution plot for each biobank
    for biobank, papers in biobank_papers.items():
        if not papers:
            continue
            
        # Extract citation counts
        citation_counts = [paper['citations'] for paper in papers]
        
        if citation_counts:
            plt.figure(figsize=(10, 6))
            
            # Create histogram with log scale
            plt.hist(citation_counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f'Citation Distribution for {biobank}', fontsize=16)
            plt.xlabel('Citation Count', fontsize=14)
            plt.ylabel('Number of Papers', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add log scale if there are high citation counts
            if max(citation_counts) > 100:
                plt.xscale('log')
                plt.xlabel('Citation Count (log scale)', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_citation_distribution.png'), dpi=300)
            plt.close()
    
    # Create bar charts for top MeSH terms and keywords for each biobank
    for biobank in biobank_papers.keys():
        if biobank not in biobank_mesh_terms or not biobank_mesh_terms[biobank]:
            continue
        
        # MeSH terms
        plt.figure(figsize=(12, 8))
        terms = [term for term, _ in biobank_mesh_terms[biobank].most_common(15)]
        counts = [count for _, count in biobank_mesh_terms[biobank].most_common(15)]
        
        if terms:  # Make sure we have data
            plt.barh(terms, counts, color='skyblue')
            plt.title(f'Top 15 MeSH Terms for {biobank}', fontsize=16)
            plt.xlabel('Frequency', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_mesh_terms.png'), dpi=300)
            plt.close()
        
        # Keywords
        if biobank_keywords[biobank]:
            # Get the top keywords - since normalization is already done during processing,
            # we can just take the top 15 directly
            keywords = [kw for kw, _ in biobank_keywords[biobank].most_common(15)]
            kw_counts = [count for _, count in biobank_keywords[biobank].most_common(15)]
            
            if keywords:  # Make sure we have data
                plt.figure(figsize=(12, 8))
                plt.barh(keywords, kw_counts, color='lightgreen')
                plt.title(f'Top 15 Author Keywords for {biobank}', fontsize=16)
                plt.xlabel('Frequency', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_keywords.png'), dpi=300)
                plt.close()
    
    # Institution visualization - CONVERTED FROM PIE CHART TO BAR CHART
    for biobank in biobank_papers.keys():
        if biobank not in biobank_institutions or not biobank_institutions[biobank]:
            continue
        
        # Create a bar chart of top institutions (instead of pie chart)
        institutions = [inst for inst, _ in biobank_institutions[biobank].most_common(10)]
        inst_counts = [count for _, count in biobank_institutions[biobank].most_common(10)]
        
        if institutions and inst_counts:  # Make sure we have data
            plt.figure(figsize=(12, 8))
            
            # Truncate long institution names
            short_labels = [f"{inst[:30]}..." if len(inst) > 30 else inst for inst in institutions]
            
            # Create horizontal bar chart
            plt.barh(short_labels, inst_counts, color='skyblue')
            plt.xlabel('Number of Publications', fontsize=14)
            plt.title(f'Top 10 Institutions for {biobank}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_institutions.png'), dpi=300)
            plt.close()
            
    # Create a publications per year visualization for each biobank
    for biobank, papers in biobank_papers.items():
        if not papers:
            continue
            
        # Count papers by year
        years_counter = Counter([paper['year'] for paper in papers if paper['year'].isdigit()])
        
        # Sort years and get counts
        sorted_years = sorted([int(year) for year in years_counter.keys() if year.isdigit()])
        counts = [years_counter[str(year)] for year in sorted_years]
        
        if sorted_years:
            plt.figure(figsize=(12, 6))
            plt.bar(sorted_years, counts, color='darkblue', alpha=0.7)
            plt.title(f'Publications per Year for {biobank}', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of Publications', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(sorted_years, rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_pubs_per_year.png'), dpi=300)
            plt.close()

# -------------------------------
# Function to perform advanced analytics
# -------------------------------
def perform_advanced_analytics(df):
    """Perform advanced analytics including clustering of MeSH terms"""
    logger.info("Performing advanced analytics (MeSH term clustering)...")
    
    # Process each biobank separately
    for biobank in biobank_papers.keys():
        if biobank not in biobank_mesh_terms or len(biobank_mesh_terms[biobank]) < 10:
            logger.info(f"Skipping {biobank} - insufficient data")
            continue
            
        logger.info(f"Processing MeSH term clusters for {biobank}")
        
        # Get the most common MeSH terms for this biobank (top 100)
        top_terms = [term for term, _ in biobank_mesh_terms[biobank].most_common(100)]
        if not top_terms:
            logger.info(f"No MeSH terms found for {biobank}")
            continue
            
        # Create a term-to-index mapping
        term_to_idx = {term: i for i, term in enumerate(top_terms)}
        
        # Initialize co-occurrence matrix
        cooccurrence_matrix = np.zeros((len(top_terms), len(top_terms)))
        
        # For each paper, find its MeSH terms and update the co-occurrence matrix
        for paper in biobank_papers[biobank]:
            pmid = paper['pmid']
            
            # Get MeSH terms for this paper from the dataframe
            paper_mesh = []
            try:
                paper_row = df[df['PMID'] == pmid]
                if not paper_row.empty:
                    paper_mesh = paper_row['MeSH Terms'].iloc[0].split('; ') if not pd.isna(paper_row['MeSH Terms'].iloc[0]) else []
                    
                    # Normalize the MeSH terms
                    normalized_paper_mesh = [normalize_keyword(term) for term in paper_mesh]
                    paper_mesh = [term for term in normalized_paper_mesh if term in top_terms]
                    
                    # Update co-occurrence counts
                    for i, term1 in enumerate(paper_mesh):
                        idx1 = term_to_idx.get(term1)
                        if idx1 is not None:
                            for term2 in paper_mesh:
                                idx2 = term_to_idx.get(term2)
                                if idx2 is not None:
                                    cooccurrence_matrix[idx1, idx2] += 1
            except Exception as e:
                logger.error(f"Error processing MeSH terms for PMID {pmid}: {e}")
                continue
        
        # Perform clustering (without visualization)
        if len(top_terms) > 2:
            try:
                # Perform PCA to reduce dimensionality (for clustering only, not for visualization)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(cooccurrence_matrix)
                
                # Perform K-means clustering
                n_clusters = min(10, len(top_terms))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(pca_result)
                
                # For each cluster, identify the most common terms
                cluster_terms = {}
                for cluster_id in range(n_clusters):
                    terms_in_cluster = [top_terms[i] for i in range(len(top_terms)) if clusters[i] == cluster_id]
                    # Sort by frequency in the biobank
                    terms_in_cluster.sort(key=lambda x: biobank_mesh_terms[biobank][x], reverse=True)
                    cluster_terms[cluster_id] = terms_in_cluster
                
                # Save the cluster analysis to a text file
                with open(os.path.join(output_dir, f"{biobank.replace(' ', '_')}_mesh_clusters.txt"), 'w', encoding='utf-8') as f:
                    f.write(f"MESH TERM CLUSTER ANALYSIS FOR {biobank}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for cluster_id, terms in cluster_terms.items():
                        f.write(f"Cluster {cluster_id}:\n")
                        f.write("-" * 30 + "\n")
                        for term in terms[:10]:  # Top 10 terms in each cluster
                            f.write(f"  {term}: {biobank_mesh_terms[biobank][term]} occurrences\n")
                        f.write("\n")
                        
                        # Try to interpret the cluster
                        f.write("Potential interpretation: ")
                        if len(terms) >= 3:
                            top_3_terms = terms[:3]
                            f.write(f"This cluster appears to focus on {', '.join(top_3_terms)}\n")
                        else:
                            f.write("Insufficient terms to interpret\n")
                        f.write("\n" + "=" * 50 + "\n\n")
            except Exception as e:
                logger.error(f"Error in PCA/clustering for {biobank}: {e}")
                print(f"Error in PCA/clustering for {biobank}: {e}")

# -------------------------------
# Function to generate top cited article tables
# -------------------------------
def generate_top_cited_article_tables():
    """Generate HTML tables for the top 10 most cited articles for each biobank"""
    logger.info("Generating top cited article tables...")
    
    for biobank, papers in biobank_papers.items():
        if not papers:
            continue
            
        # Sort papers by citation count (descending)
        cited_papers = sorted(papers, key=lambda x: (x['citations'], x['year']), reverse=True)
        
        # Take top 10 or fewer if there aren't enough
        top_papers = cited_papers[:10]
        
        # Create HTML table
        html = f"""
        <html>
        <head>
            <title>Top Cited Articles for {biobank}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e6f7ff; }}
            </style>
        </head>
        <body>
            <h1>Top 10 Most Cited Articles for {biobank}</h1>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Title</th>
                    <th>Journal</th>
                    <th>Year</th>
                    <th>Main Authors</th>
                    <th>Citations</th>
                    <th>PMID</th>
                </tr>
        """
        
        for i, paper in enumerate(top_papers, 1):
            # Format authors as a comma-separated list
            authors_str = ", ".join(paper['authors'])
            if len(paper['authors']) < len(paper_full_authors.get(paper['pmid'], [])):
                authors_str += " et al."
                
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{paper['title']}</td>
                    <td>{paper['journal']}</td>
                    <td>{paper['year']}</td>
                    <td>{authors_str}</td>
                    <td>{paper['citations']}</td>
                    <td>{paper['pmid']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save the HTML table to file
        output_file = os.path.join(tables_dir, f"{biobank.replace(' ', '_')}_top_cited_articles.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Generated top cited articles table for {biobank}")
        
        # Also generate CSV for easier data manipulation
        csv_data = "Rank,Title,Journal,Year,Main Authors,Citations,PMID\n"
        for i, paper in enumerate(top_papers, 1):
            authors_str = ", ".join(paper['authors'])
            if len(paper['authors']) < len(paper_full_authors.get(paper['pmid'], [])):
                authors_str += " et al."
            
            # Replace commas in title and authors to prevent CSV parsing issues
            title_clean = paper['title'].replace(",", ";")
            authors_clean = authors_str.replace(",", ";")
            journal_clean = paper['journal'].replace(",", ";")
            
            csv_data += f"{i},{title_clean},{journal_clean},{paper['year']},{authors_clean},{paper['citations']},{paper['pmid']}\n"
        
        # Save the CSV file
        csv_file = os.path.join(tables_dir, f"{biobank.replace(' ', '_')}_top_cited_articles.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(csv_data)

# -------------------------------
# Function to generate statistics
# -------------------------------
def generate_statistics():
    """Generate descriptive statistics about the biobank literature"""
    logger.info("Generating descriptive statistics...")
    
    # Generate the statistics file
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("===============================================\n")
        f.write("BIOBANK EHR LITERATURE DESCRIPTIVE STATISTICS\n")
        f.write("===============================================\n\n")
        
        # Overall statistics
        f.write(f"Total papers analyzed: {len(paper_titles)}\n")
        f.write(f"Papers by biobank:\n")
        for biobank, papers in biobank_papers.items():
            f.write(f"  - {biobank}: {len(papers)} papers\n")
        f.write("\n")
        
        # For each biobank
        for biobank in biobank_papers.keys():
            if not biobank_papers[biobank]:
                continue
                
            f.write(f"\n{'=' * 50}\n")
            f.write(f"{biobank} STATISTICS\n")
            f.write(f"{'=' * 50}\n\n")
            
            # 1. Most Common MeSH Terms
            f.write("1. MOST COMMON MeSH TERMS\n")
            f.write("----------------------\n")
            for term, count in biobank_mesh_terms[biobank].most_common(20):
                f.write(f"  {term}: {count} occurrences\n")
            f.write("\n")
            
            # 2. Most Common Author Keywords
            f.write("2. MOST COMMON AUTHOR KEYWORDS\n")
            f.write("----------------------------\n")
            
            # We don't need special handling for FinnGen anymore since keyword normalization
            # is now done during initial processing for all biobanks
            for keyword, count in biobank_keywords[biobank].most_common(20):
                f.write(f"  {keyword}: {count} occurrences\n")
            f.write("\n")
            
            # 3. Most Cited Papers
            f.write("3. MOST CITED PAPERS (Based on available data)\n")
            f.write("------------------------------------\n")
            cited_papers = sorted(biobank_papers[biobank], key=lambda x: (x['citations'], x['year']), reverse=True)
            for i, paper in enumerate(cited_papers[:20], 1):
                f.write(f"  {i}. PMID {paper['pmid']}: {paper['title']} ({paper['year']}) - {paper['citations']} citations\n")
            f.write("\n")
            
            # 4. Most Prolific Authors
            f.write("4. MOST PROLIFIC AUTHORS\n")
            f.write("-----------------------\n")
            for author, count in biobank_authors[biobank].most_common(20):
                f.write(f"  {author}: {count} publications\n")
            f.write("\n")
            
            # 5. Leading Institutions
            f.write("5. LEADING INSTITUTIONS\n")
            f.write("----------------------\n")
            for i, (institution, count) in enumerate(biobank_institutions[biobank].most_common(20), 1):
                # Format institution name appropriately
                inst_name = institution[:100] + "..." if len(institution) > 100 else institution
                f.write(f"  {i}. {inst_name}: {count} publications\n")
            f.write("\n\n")
            
            # 6. Citation Analysis (new section)
            f.write("6. CITATION ANALYSIS\n")
            f.write("-------------------\n")
            citation_counts = [paper['citations'] for paper in biobank_papers[biobank]]
            if citation_counts:
                total_citations = sum(citation_counts)
                avg_citations = total_citations / len(citation_counts) if citation_counts else 0
                median_citations = sorted(citation_counts)[len(citation_counts)//2] if citation_counts else 0
                max_citations = max(citation_counts) if citation_counts else 0
                
                f.write(f"  Total citations: {total_citations}\n")
                f.write(f"  Average citations per paper: {avg_citations:.2f}\n")
                f.write(f"  Median citations per paper: {median_citations}\n")
                f.write(f"  Maximum citations: {max_citations}\n")
                
                # Count papers with 0, 1-10, 11-50, 51-100, >100 citations
                citation_ranges = {
                    "0 citations": sum(1 for c in citation_counts if c == 0),
                    "1-10 citations": sum(1 for c in citation_counts if 1 <= c <= 10),
                    "11-50 citations": sum(1 for c in citation_counts if 11 <= c <= 50),
                    "51-100 citations": sum(1 for c in citation_counts if 51 <= c <= 100),
                    ">100 citations": sum(1 for c in citation_counts if c > 100)
                }
                
                f.write("\n  Citation distribution:\n")
                for range_label, count in citation_ranges.items():
                    percentage = (count / len(citation_counts)) * 100 if citation_counts else 0
                    f.write(f"    {range_label}: {count} papers ({percentage:.1f}%)\n")
            else:
                f.write("  No citation data available\n")
            f.write("\n\n")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    try:
        # Print working directory and path info for debugging
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Looking for CSV file at: {csv_file}")
        print(f"Output directory: {output_dir}")
        
        # Try to parse existing statistics first
        input_file = stats_file
        data = parse_biobank_statistics(input_file)
        
        # If statistics file doesn't exist, generate it
        if data is None:
            logger.info(f"No existing statistics file found at {input_file}. Generating from CSV...")
            
            # Check if CSV file exists
            if not os.path.exists(csv_file):
                logger.error(f"Input CSV file not found: {csv_file}")
                print(f"Error: Input CSV file not found: {csv_file}")
                print("Please run the retrieval script first to generate the CSV file.")
                sys.exit(1)
                
            # Read the CSV data
            logger.info("Reading CSV data...")
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded {len(df)} articles from CSV")
                
                # Check for the correct Citation Count column
                if 'Citation Count' not in df.columns:
                    # Look for alternative column names
                    if 'Citations' in df.columns:
                        df.rename(columns={'Citations': 'Citation Count'}, inplace=True)
                    else:
                        logger.warning("No citation count column found. Adding a default column with zero values.")
                        df['Citation Count'] = 0
                
            except Exception as e:
                logger.error(f"Error reading CSV file: {e}")
                print(f"Error reading CSV file: {e}")
                sys.exit(1)
            
            # Process the CSV data
            process_csv_data(df)
            
            # Create visualizations
            create_visualizations()
            logger.info("Basic visualizations completed")
            
            # Perform advanced analysis - mesh term clustering without visualizations
            perform_advanced_analytics(df)
            logger.info("Advanced analytics completed")
            
            # Generate top cited article tables
            generate_top_cited_article_tables()
            logger.info("Top cited article tables generated")
            
            # Generate statistics
            generate_statistics()
            logger.info("Statistics generation completed")
            
            print(f"\n✅ Analysis complete!")
            print(f"📊 Statistics file: {stats_file}")
            print(f"📊 Visualizations saved to: {plots_dir}")
            print(f"📊 Top cited article tables saved to: {tables_dir}")
        else:
            # Use the parsed data for further analysis
            logger.info("Using existing statistics file for analysis...")
            print(f"Found existing statistics file. Further analysis would go here.")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)