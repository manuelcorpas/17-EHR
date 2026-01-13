from Bio import Entrez
import csv
import time
import os
import sys
import re
import json
from collections import Counter, defaultdict
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import pandas as pd

# Set Entrez parameters
Entrez.email = "your.email@example.com"
Entrez.tool = "BiopythonClient"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("DATA", "biobank_script.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 500
MAX_RETRY = 3
NER_BATCH_SIZE = 100  # Process NER in batches to manage memory

seen_pmids = set()
processed_count = 0
current_year = datetime.now().year

# Output location
output_dir = "DATA"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "00-00-ehr_biobank_articles.csv")
stats_file = os.path.join("00-01-LITERATURE-ANALYSIS/00-00-biobank_statistics.txt")
plots_dir = os.path.join("00-01-LITERATURE-ANALYSIS/plots")
os.makedirs(plots_dir, exist_ok=True)

# Tracking statistics
biobank_papers = defaultdict(list)  # {biobank: [pmids]}
biobank_mesh_terms = defaultdict(lambda: Counter())  # {biobank: Counter(mesh_terms)}
biobank_keywords = defaultdict(lambda: Counter())  # {biobank: Counter(keywords)}
biobank_authors = defaultdict(lambda: Counter())  # {biobank: Counter(authors)}
biobank_institutions = defaultdict(lambda: Counter())  # {biobank: Counter(institutions)}
paper_citations = defaultdict(int)  # {pmid: citation_count}
paper_titles = {}  # {pmid: title}
paper_years = {}   # {pmid: year}

# Institution normalization mapping
institution_mapping = {
    # UK-based institutions
    "university of oxford": "University of Oxford",
    "oxford university": "University of Oxford",
    "university of cambridge": "University of Cambridge",
    "cambridge university": "University of Cambridge",
    "imperial college london": "Imperial College London",
    "university college london": "University College London",
    "king's college london": "King's College London",
    "kings college london": "King's College London",
    
    # US-based institutions
    "harvard medical school": "Harvard Medical School",
    "harvard university": "Harvard University",
    "stanford university": "Stanford University",
    "university of california": "University of California",
    "ucla": "University of California, Los Angeles",
    "ucsf": "University of California, San Francisco",
    "uc berkeley": "University of California, Berkeley",
    "johns hopkins university": "Johns Hopkins University",
    "yale university": "Yale University",
    "yale school of medicine": "Yale School of Medicine",
    "massachusetts general hospital": "Massachusetts General Hospital",
    "mayo clinic": "Mayo Clinic",
    "nih": "National Institutes of Health",
    "national institutes of health": "National Institutes of Health",
    
    # Finnish institutions
    "university of helsinki": "University of Helsinki",
    "institute for molecular medicine finland": "Institute for Molecular Medicine Finland (FIMM)",
    "fimm": "Institute for Molecular Medicine Finland (FIMM)",
    
    # Estonian institutions
    "estonian biobank": "Estonian Biobank",
    "estonian genome center": "Estonian Genome Center",
    "university of tartu": "University of Tartu"
}

# Biobank aliases
biobanks = {
    "UK Biobank": ["UK Biobank"],
    "All of Us": ["All of Us"],
    "FinnGen": ["FinnGen"],
    "Estonian Biobank": ["Estonian Biobank"],
    "Million Veteran Program": ["Million Veteran Program", "MVP"]
}

# Build biobank query string
biobank_query = " OR ".join([f'"{alias}"' for names in biobanks.values() for alias in names])

# -------------------------------
# Advanced institution extraction
# -------------------------------
def extract_institutions(affiliation_text):
    """
    Extract and normalize institution names from affiliation text
    """
    if not affiliation_text:
        return []
    
    institutions = []
    
    # Split by commas or semicolons
    parts = re.split(r'[,;]', affiliation_text)
    
    # Extract candidate institution names
    for part in parts:
        part = part.strip()
        
        # Skip too short parts or those that are likely not institutions
        if len(part) < 4 or part.isdigit():
            continue
        
        # Check for common institution indicators
        inst_indicators = ['university', 'college', 'institute', 'school', 'center', 'centre', 
                          'hospital', 'clinic', 'laboratory', 'department', 'faculty']
        
        if any(indicator in part.lower() for indicator in inst_indicators):
            # Extract the institution name
            institutions.append(part)
        elif 'biobank' in part.lower() or any(biobank_name.lower() in part.lower() for biobank_name in biobanks.keys()):
            # Specific handling for biobank names
            institutions.append(part)
    
    # Normalize institution names
    normalized_institutions = []
    for inst in institutions:
        inst_lower = inst.lower()
        
        # Try to match with known institutions
        matched = False
        for pattern, normalized_name in institution_mapping.items():
            if pattern in inst_lower:
                normalized_institutions.append(normalized_name)
                matched = True
                break
        
        # If no match found, use the original institution name
        if not matched:
            normalized_institutions.append(inst)
    
    return list(set(normalized_institutions))  # Remove duplicates

# -------------------------------
# Function to save article records
# -------------------------------
def save_records_to_csv(records, csv_file):
    batch_count = 0
    with open(csv_file, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for article in records.get('PubmedArticle', []):
            pmid = str(article['MedlineCitation']['PMID'])
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            article_data = article['MedlineCitation']['Article']
            title = article_data.get('ArticleTitle', '')
            # Store title for reference
            paper_titles[pmid] = title

            abstract_parts = article_data.get('Abstract', {}).get('AbstractText', [''])
            if isinstance(abstract_parts, list):
                abstract = ' '.join(str(part) for part in abstract_parts)
            else:
                abstract = str(abstract_parts)

            authors = []
            affiliations = set()
            for author in article_data.get('AuthorList', []):
                if 'LastName' in author and 'Initials' in author:
                    author_name = f"{author['LastName']} {author['Initials']}"
                    authors.append(author_name)
                for aff in author.get('AffiliationInfo', []):
                    affiliations.add(aff.get('Affiliation', ''))
            author_str = "; ".join(authors)
            affiliation_str = "; ".join(affiliations)

            journal = article_data.get('Journal', {}).get('Title', '')
            pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            paper_years[pmid] = year
            date_str = f"{year}-{pub_date.get('Month', '')}"

            doi = ''
            for id_item in article_data.get('ELocationID', []):
                if id_item.attributes.get('EIdType') == 'doi':
                    doi = str(id_item)

            # Extract MeSH terms separately from keywords
            mesh_terms = []
            for mesh_heading in article['MedlineCitation'].get('MeshHeadingList', []):
                descriptor = str(mesh_heading['DescriptorName'])
                mesh_terms.append(descriptor)
            mesh_str = "; ".join(mesh_terms)
            
            # Extract author-provided keywords separately
            keyword_list = []
            for kw_group in article['MedlineCitation'].get('KeywordList', []):
                for k in kw_group:
                    keyword_list.append(str(k))
            keywords = "; ".join(keyword_list)

            # Get citation count from PubMed
            citation_count = 0
            if 'CommentsCorrectionsList' in article['MedlineCitation']:
                for comment in article['MedlineCitation']['CommentsCorrectionsList']:
                    if comment.attributes.get('RefType') == 'Cites':
                        citation_count += 1
            paper_citations[pmid] = citation_count

            full_text = f"{title} {abstract} {keywords}".lower()
            matched_biobanks = {biobank for biobank, aliases in biobanks.items()
                                if any(alias.lower() in full_text for alias in aliases)}
            biobank_str = "; ".join(sorted(matched_biobanks))

            # Extract institutions using improved method
            extracted_institutions = []
            for affiliation in affiliations:
                institutions = extract_institutions(affiliation)
                extracted_institutions.extend(institutions)
            
            # Update statistics
            for biobank in matched_biobanks:
                biobank_papers[biobank].append({
                    'pmid': pmid,
                    'title': title,
                    'citations': citation_count,
                    'year': year
                })
                
                # Add MeSH terms
                for term in mesh_terms:
                    biobank_mesh_terms[biobank][term] += 1
                
                # Add keywords separately
                for keyword in keyword_list:
                    biobank_keywords[biobank][keyword] += 1
                
                # Add authors
                for author in authors:
                    biobank_authors[biobank][author] += 1
                
                # Add institutions - using improved extraction
                for institution in extracted_institutions:
                    if institution and len(institution) > 3:  # Filter out very short institution names
                        biobank_institutions[biobank][institution] += 1

            writer.writerow([
                pmid, title, abstract, author_str, journal, year, date_str, doi,
                mesh_str, keywords, affiliation_str, biobank_str
            ])
            batch_count += 1
    return batch_count

# -------------------------------
# Function to process a search query
# -------------------------------
def process_records(query, count):
    global processed_count, seen_pmids
    
    # For each batch, we need to get the right portion of results
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        retry_count, success = 0, False

        while not success and retry_count < MAX_RETRY:
            try:
                print(f"    ⏬ Fetching records {start + 1} to {end}")
                
                # For each batch, do a new search with the correct retstart
                search_handle = Entrez.esearch(
                    db="pubmed", 
                    term=query, 
                    retstart=start,
                    retmax=BATCH_SIZE
                )
                search_results = Entrez.read(search_handle)
                search_handle.close()
                
                # Get the PMIDs for this batch
                pmid_list = search_results["IdList"]
                
                if not pmid_list:
                    print("    ⚠️ No PMIDs returned in this batch")
                    success = True
                    continue
                
                # Fetch the full records using the PMIDs
                fetch_handle = Entrez.efetch(
                    db="pubmed", 
                    rettype="xml", 
                    retmode="xml",
                    id=",".join(pmid_list)
                )
                records = Entrez.read(fetch_handle)
                fetch_handle.close()

                batch_count = save_records_to_csv(records, csv_file)
                processed_count += batch_count
                success = True
                print(f"    ✓ Saved {batch_count} articles. Total: {processed_count}")
                
            except Exception as e:
                retry_count += 1
                print(f"    ⚠️ Error: {e} (Attempt {retry_count}/{MAX_RETRY})")
                if retry_count < MAX_RETRY:
                    wait_time = 2 ** retry_count
                    print(f"    ⏱️ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("    ❌ Max retries reached. Skipping batch...")
                    
        # Sleep between batches to respect NCBI's rate limits
        time.sleep(1)

# -------------------------------
# Function to create visualizations
# -------------------------------
def create_visualizations():
    """Create visualizations for the data analysis"""
    
    logger.info("Creating visualizations...")
    
    # Create a figure for publication trends over time
    plt.figure(figsize=(12, 8))
    
    # Prepare data for time series
    years_range = range(2010, current_year + 1)
    biobank_pub_by_year = {}
    
    for biobank in biobanks.keys():
        pub_counts = []
        for year in years_range:
            year_str = str(year)
            count = sum(1 for paper in biobank_papers[biobank] if paper['year'] == year_str)
            pub_counts.append(count)
        biobank_pub_by_year[biobank] = pub_counts
    
    # Plot the time series
    for biobank, counts in biobank_pub_by_year.items():
        plt.plot(years_range, counts, marker='o', linewidth=2, label=biobank)
    
    plt.title('Publication Trends for Biobanks Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Publications', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'publication_trends.png'), dpi=300)
    plt.close()
    
    # Create bar charts for top MeSH terms and keywords for each biobank
    for biobank in biobanks.keys():
        if biobank not in biobank_mesh_terms or not biobank_mesh_terms[biobank]:
            continue
        
        # MeSH terms
        plt.figure(figsize=(12, 8))
        terms = [term for term, _ in biobank_mesh_terms[biobank].most_common(15)]
        counts = [count for _, count in biobank_mesh_terms[biobank].most_common(15)]
        
        plt.barh(terms, counts, color='skyblue')
        plt.title(f'Top 15 MeSH Terms for {biobank}', fontsize=16)
        plt.xlabel('Frequency', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_mesh_terms.png'), dpi=300)
        plt.close()
        
        # Keywords
        plt.figure(figsize=(12, 8))
        keywords = [kw for kw, _ in biobank_keywords[biobank].most_common(15)]
        kw_counts = [count for _, count in biobank_keywords[biobank].most_common(15)]
        
        plt.barh(keywords, kw_counts, color='lightgreen')
        plt.title(f'Top 15 Author Keywords for {biobank}', fontsize=16)
        plt.xlabel('Frequency', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_keywords.png'), dpi=300)
        plt.close()
    
    # Institution network visualization
    for biobank in biobanks.keys():
        if biobank not in biobank_institutions or not biobank_institutions[biobank]:
            continue
        
        # Create a pie chart of top institutions
        plt.figure(figsize=(12, 10))
        institutions = [inst for inst, _ in biobank_institutions[biobank].most_common(10)]
        inst_counts = [count for _, count in biobank_institutions[biobank].most_common(10)]
        
        # Calculate percentages
        total = sum(inst_counts)
        percentages = [count/total*100 for count in inst_counts]
        
        # Truncate long institution names
        short_labels = [f"{inst[:30]}..." if len(inst) > 30 else inst for inst in institutions]
        
        plt.pie(inst_counts, labels=short_labels, autopct='%1.1f%%', 
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title(f'Top 10 Institutions for {biobank}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{biobank.replace(" ", "_")}_institutions.png'), dpi=300)
        plt.close()

# -------------------------------
# Function to generate statistics
# -------------------------------
def generate_statistics():
    logger.info("Generating descriptive statistics...")
    
    # Create visualizations
    create_visualizations()
    
    # Generate the statistics file
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("===============================================\n")
        f.write("BIOBANK EHR LITERATURE DESCRIPTIVE STATISTICS\n")
        f.write("===============================================\n\n")
        
        # Overall statistics
        f.write(f"Total papers analyzed: {len(seen_pmids)}\n")
        f.write(f"Papers by biobank:\n")
        for biobank, papers in biobank_papers.items():
            f.write(f"  - {biobank}: {len(papers)} papers\n")
        f.write("\n")
        
        # For each biobank
        for biobank in biobanks.keys():
            if biobank not in biobank_papers or not biobank_papers[biobank]:
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
            
            # 3. Most Cited Papers (using PubMed citations only)
            f.write("3. MOST CITED PAPERS (PubMed Citations)\n")
            f.write("------------------------------------\n")
            cited_papers = sorted(biobank_papers[biobank], key=lambda x: (x['citations'], x['year']), reverse=True)
            for i, paper in enumerate(cited_papers[:20], 1):
                f.write(f"  {i}. PMID {paper['pmid']}: {paper['title']} ({paper['year']}) - {paper['citations']} citations\n")
            f.write("\n")
            
            # 4. Most Prolific Authors (specific to this biobank)
            f.write("4. MOST PROLIFIC AUTHORS\n")
            f.write("-----------------------\n")
            for author, count in biobank_authors[biobank].most_common(20):
                f.write(f"  {author}: {count} publications\n")
            f.write("\n")
            
            # 5. Leading Institutions (showing actual organization names)
            f.write("5. LEADING INSTITUTIONS\n")
            f.write("----------------------\n")
            for i, (institution, count) in enumerate(biobank_institutions[biobank].most_common(20), 1):
                # Format institution name appropriately
                inst_name = institution[:100] + "..." if len(institution) > 100 else institution
                f.write(f"  {i}. {inst_name}: {count} publications\n")
            f.write("\n\n")
    
    logger.info(f"Statistics saved to: {stats_file}")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    try:
        # Write CSV header once
        with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "PMID", "Title", "Abstract", "Authors", "Journal", "Year", "PubDate", "DOI",
                "MeSH Terms", "Keywords", "Affiliations", "Biobank"
            ])

        print("🔄 Using year-by-year search to handle large result sets...")
        
        # Process each year (we'll go backwards from most recent)
        for year in range(current_year, 1999, -1):
            year_query = f"({biobank_query}) AND ({year}[PDAT])"
            print(f"\n📅 Processing year {year}...")

            search_handle = Entrez.esearch(db="pubmed", term=year_query, retmax=0)
            search_results = Entrez.read(search_handle)
            search_handle.close()

            year_count = int(search_results["Count"])
            print(f"  📄 Found {year_count} articles for {year}")
            if year_count == 0:
                continue

            if year_count > 9000:
                print(f"  ⚠️ Year {year} >9000, processing by month...")
                for month in range(1, 13):
                    month_name = datetime(year, month, 1).strftime("%b")
                    month_query = f"({biobank_query}) AND ({year}/{month_name}[PDAT])"
                    print(f"    🗓️ {month_name} {year}...")

                    month_search = Entrez.esearch(db="pubmed", term=month_query, retmax=0)
                    month_results = Entrez.read(month_search)
                    month_search.close()

                    month_count = int(month_results["Count"])
                    print(f"      📄 Found {month_count} articles")
                    if month_count == 0:
                        continue
                    process_records(month_query, month_count)
            else:
                process_records(year_query, year_count)

        # Generate statistics at the end
        generate_statistics()

        print(f"\n✅ Done! {processed_count} articles saved to: {csv_file}")
        print(f"📊 Unique PMIDs processed: {len(seen_pmids)}")
        print(f"📊 Statistics file: {stats_file}")
        print(f"📊 Visualizations saved to: {plots_dir}")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)