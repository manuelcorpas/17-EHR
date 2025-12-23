import os
import sys
import pandas as pd
import numpy as np
import logging
from collections import Counter, defaultdict
from datetime import datetime
import re

# Setup paths - make all paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up one level

# Define the input/output locations
input_dir = os.path.join(parent_dir, "DATA")
csv_file = os.path.join(input_dir, "00-00-ehr_biobank_articles_with_citations.csv")
output_dir = os.path.join(parent_dir, "ANALYSIS", "00-01-LITERATURE-ANALYSIS")
os.makedirs(output_dir, exist_ok=True)
stats_file = os.path.join(output_dir, "00-01-biobank_statistics.txt")

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

# Get current year for analysis
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
        
        # Store basic paper info
        paper_titles[pmid] = title
        paper_years[pmid] = year
        paper_journals[pmid] = journal
        paper_full_authors[pmid] = authors_list
        
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
            for keyword, count in biobank_keywords[biobank].most_common(20):
                f.write(f"  {keyword}: {count} occurrences\n")
            f.write("\n")
            
            # 3. Most Prolific Authors
            f.write("3. MOST PROLIFIC AUTHORS\n")
            f.write("-----------------------\n")
            for author, count in biobank_authors[biobank].most_common(20):
                f.write(f"  {author}: {count} publications\n")
            f.write("\n")
            
            # 4. Leading Institutions
            f.write("4. LEADING INSTITUTIONS\n")
            f.write("----------------------\n")
            for i, (institution, count) in enumerate(biobank_institutions[biobank].most_common(20), 1):
                # Format institution name appropriately
                inst_name = institution[:100] + "..." if len(institution) > 100 else institution
                f.write(f"  {i}. {inst_name}: {count} publications\n")
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
                
            except Exception as e:
                logger.error(f"Error reading CSV file: {e}")
                print(f"Error reading CSV file: {e}")
                sys.exit(1)
            
            # Process the CSV data
            process_csv_data(df)
            
            # Generate statistics
            generate_statistics()
            logger.info("Statistics generation completed")
            
            print(f"\n✅ Analysis complete!")
            print(f"📊 Statistics file: {stats_file}")
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