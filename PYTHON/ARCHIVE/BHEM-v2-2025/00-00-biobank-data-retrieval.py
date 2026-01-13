"""
BIOBANK DATA RETRIEVAL

Retrieves ALL papers that mention major biobanks and saves to CSV.
No limits, no topic filtering - gets complete dataset of papers mentioning these biobanks.

OUTPUT CSV COLUMNS:
- Biobank: Which biobank the paper refers to
- PMID: PubMed ID
- Title: Article title
- Abstract: Full abstract text
- Journal: Journal name
- Year: Publication year
- MeSH_Terms: Medical Subject Headings (semicolon separated)

USAGE:
1. Place this script in PYTHON/ directory as 00-00-biobank_data_retrieval.py
2. Run from root directory: python PYTHON/00-00-biobank_data_retrieval.py
3. Data will be saved to DATA/biobank_research_data.csv

REQUIREMENTS:
- pip install biopython pandas
- Email is pre-configured: mc.admin@manuelcorpas.com

NOTE: To exclude preprints, you can filter the results afterwards by journal name
(e.g., exclude medRxiv, bioRxiv, Research Square from the final CSV)
"""

import os
import re
import pandas as pd
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

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
os.makedirs(data_dir, exist_ok=True)

# Configure Entrez
Entrez.email = "mc.admin@manuelcorpas.com"
Entrez.tool = "BiobankDataRetrieval"

# Define biobanks with specific search terms
BIOBANK_DEFINITIONS = {
    'UK Biobank': [
        'UK Biobank', 'United Kingdom Biobank', 'U.K. Biobank', 
        'UK-Biobank'
    ],
    'Million Veteran Program': [
        'Million Veteran Program', 'Million Veterans Program',
        'MVP biobank', 'MVP cohort', 'MVP genomics',
        'Veterans Affairs Million Veteran Program',
        'VA Million Veteran Program'
    ],
    'FinnGen': [
        'FinnGen', 'FinnGen biobank', 'FinnGen study',
        'FinnGen cohort', 'FinnGen consortium'
    ],
    'All of Us': [
        'All of Us Research Program', 'All of Us cohort',
        'All of Us biobank', 'AoU Research Program',
        'Precision Medicine Initiative cohort'
    ],
    'Estonian Biobank': [
        'Estonian Biobank', 'Estonia Biobank', 'Estonian Genome Center',
        'Estonian Health Cohort', 'Tartu Biobank'
    ]
}

def create_biobank_search_query(biobank_name):
    """Create search query to find papers that mention this biobank"""
    biobank_terms = BIOBANK_DEFINITIONS[biobank_name]
    biobank_query = ' OR '.join([f'"{term}"[All Fields]' for term in biobank_terms])
    
    # Just biobank name and date range - no exclusions for now
    full_query = f'({biobank_query}) AND ("2000"[PDAT] : "2025"[PDAT])'
    
    return full_query

def search_pubmed_for_biobank(biobank_name):
    """Search PubMed for ALL published papers that mention this biobank using date-based chunking"""
    logger.info(f"Searching for ALL papers mentioning {biobank_name}...")
    
    all_pmids = []
    
    # Break down by year to avoid 9,999 limit
    years = range(2000, 2026)  # 2000 to 2025
    
    for year in years:
        # Build year-specific query  
        biobank_terms = BIOBANK_DEFINITIONS[biobank_name]
        biobank_query = ' OR '.join([f'"{term}"[All Fields]' for term in biobank_terms])
        
        year_query = f'({biobank_query}) AND ("{year}"[PDAT])'
        
        try:
            # Get total count for this year
            handle = Entrez.esearch(db="pubmed", term=year_query, retmax=0)
            search_results = Entrez.read(handle)
            handle.close()
            
            year_count = int(search_results["Count"])
            
            if year_count == 0:
                continue
                
            logger.info(f"Year {year}: {year_count:,} papers for {biobank_name}")
                
            if year_count > 9999:
                # If still too many for a year, break down by month
                logger.info(f"Year {year} has {year_count:,} papers - breaking down by month")
                year_pmids = search_year_by_month(biobank_name, year)
            else:
                # Get all PMIDs for this year
                handle = Entrez.esearch(db="pubmed", term=year_query, retmax=year_count, sort="relevance")
                search_results = Entrez.read(handle)
                handle.close()
                
                year_pmids = search_results["IdList"]
            
            all_pmids.extend(year_pmids)
            time.sleep(0.34)  # Respect rate limits
            
        except Exception as e:
            logger.error(f"Error searching year {year} for {biobank_name}: {e}")
            continue
    
    logger.info(f"Retrieved {len(all_pmids):,} total PMIDs for {biobank_name}")
    
    # Now fetch the full article data in batches
    return fetch_article_data(all_pmids, biobank_name)

def search_year_by_month(biobank_name, year):
    """Search a specific year by month if yearly results exceed 9,999"""
    biobank_terms = BIOBANK_DEFINITIONS[biobank_name]
    biobank_query = ' OR '.join([f'"{term}"[All Fields]' for term in biobank_terms])
    
    year_pmids = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        month_query = f'({biobank_query}) AND ("{year}/{month_str}"[PDAT])'
        
        try:
            handle = Entrez.esearch(db="pubmed", term=month_query, retmax=9999, sort="relevance")
            search_results = Entrez.read(handle)
            handle.close()
            
            month_pmids = search_results["IdList"]
            year_pmids.extend(month_pmids)
            
            if len(month_pmids) > 0:
                logger.info(f"  {year}/{month_str}: {len(month_pmids):,} papers")
            
            time.sleep(0.34)
            
        except Exception as e:
            logger.error(f"Error searching {year}/{month_str} for {biobank_name}: {e}")
            continue
    
    return year_pmids

def fetch_article_data(pmids, biobank_name):
    """Fetch full article data for a list of PMIDs"""
    if not pmids:
        return []
    
    logger.info(f"Fetching article data for {len(pmids):,} papers for {biobank_name}")
    
    batch_size = 100
    articles = []
    
    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(pmids) - 1) // batch_size + 1
        
        logger.info(f"Fetching article data batch {batch_num:,}/{total_batches:,} for {biobank_name}")
        
        try:
            handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="xml", retmode="xml")
            records = handle.read()
            handle.close()
            
            root = ET.fromstring(records)
            
            for article in root.findall('.//PubmedArticle'):
                article_data = parse_pubmed_article(article, biobank_name)
                if article_data:
                    articles.append(article_data)
            
            time.sleep(0.34)  # Respect NCBI rate limits
            
        except Exception as e:
            logger.error(f"Error fetching batch {batch_num} for {biobank_name}: {e}")
            continue
    
    logger.info(f"Successfully parsed {len(articles):,} articles for {biobank_name}")
    return articles

def parse_pubmed_article(article_xml, biobank_name):
    """Parse PubMed article and extract required fields only"""
    try:
        article_data = {}
        
        # Biobank
        article_data['Biobank'] = biobank_name
        
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
        
        return article_data
        
    except Exception as e:
        logger.error(f"Error parsing article: {e}")
        return None

def check_existing_data():
    """Check if we already have data to avoid re-downloading"""
    output_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            logger.info(f"Found existing dataset with {len(df):,} papers")
            
            # Check if we have data for all biobanks
            existing_biobanks = set(df['Biobank'].unique())
            required_biobanks = set(BIOBANK_DEFINITIONS.keys())
            
            if required_biobanks.issubset(existing_biobanks):
                biobank_counts = df['Biobank'].value_counts()
                
                print(f"\n📁 Found existing dataset:")
                for biobank, count in biobank_counts.items():
                    if biobank in BIOBANK_DEFINITIONS:
                        print(f"   {biobank}: {count:,} papers")
                
                use_existing = input(f"\nUse existing data? (y/n): ").strip().lower()
                if use_existing in ['y', 'yes']:
                    return df
            
        except Exception as e:
            logger.error(f"Error reading existing data: {e}")
    
    return None

def save_progress(all_articles, biobank_name):
    """Save progress after each biobank to avoid losing data"""
    if all_articles:
        df = pd.DataFrame(all_articles)
        temp_file = os.path.join(data_dir, f'biobank_research_data_temp.csv')
        df.to_csv(temp_file, index=False)
        logger.info(f"Progress saved after {biobank_name}")

def main():
    """Main execution function - data retrieval only"""
    print("=" * 60)
    print("BIOBANK DATA RETRIEVAL") 
    print("All papers mentioning major biobanks")
    print("=" * 60)
    
    # Email is already configured
    print(f"Using email: {Entrez.email}")
    print(f"Data will be saved to: {data_dir}")
    
    print(f"\n🎯 Search strategy:")
    print(f"   Simple biobank name search only")
    print(f"   No topic filtering - ALL papers mentioning these biobanks")
    print(f"   Date-based chunking to bypass 9,999 record limit")
    print(f"   Searches year-by-year (or month-by-month if needed)")
    print(f"   Date range: 2000-2025")
    
    print(f"\n📋 Target biobanks:")
    for biobank in BIOBANK_DEFINITIONS.keys():
        print(f"   - {biobank}")
    
    # Check for existing data
    existing_df = check_existing_data()
    if existing_df is not None:
        print(f"\n✅ Using existing data with {len(existing_df):,} papers")
        print(f"📊 Data ready for analysis!")
        return
    
    # Retrieve fresh data
    print(f"\n📡 Retrieving fresh data from PubMed...")
    
    all_articles = []
    biobank_counts = {}
    
    for biobank_name in BIOBANK_DEFINITIONS.keys():
        print(f"\n🔍 Searching {biobank_name}...")
        articles = search_pubmed_for_biobank(biobank_name)
        
        if articles:
            biobank_counts[biobank_name] = len(articles)
            all_articles.extend(articles)
            print(f"   ✅ Retrieved {len(articles):,} papers mentioning {biobank_name}")
            
            # Save progress after each biobank
            save_progress(all_articles, biobank_name)
        else:
            print(f"   ⚠️  No papers found mentioning {biobank_name}")
    
    if not all_articles:
        print("\n❌ No papers retrieved. Check your internet connection.")
        return
    
    # Create final dataset
    print(f"\n💾 Saving final dataset...")
    df = pd.DataFrame(all_articles)
    
    # Ensure we have the required columns
    required_columns = ['Biobank', 'PMID', 'Title', 'Abstract', 'Journal', 'Year', 'MeSH_Terms']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[required_columns]
    
    # Remove duplicates based on PMID (same paper might mention multiple biobanks)
    df_dedup = df.drop_duplicates(subset=['PMID'], keep='first')
    
    # Save final dataset
    output_file = os.path.join(data_dir, 'biobank_research_data.csv')
    df.to_csv(output_file, index=False)
    
    # Also save deduplicated version
    dedup_file = os.path.join(data_dir, 'biobank_research_data_deduplicated.csv')
    df_dedup.to_csv(dedup_file, index=False)
    
    # Clean up temp file
    temp_file = os.path.join(data_dir, f'biobank_research_data_temp.csv')
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\n✅ Data retrieval complete!")
    print(f"📂 Files saved:")
    print(f"   - {output_file}")
    print(f"   - {dedup_file} (duplicates removed)")
    
    print(f"\n📊 Summary:")
    print(f"   Total papers: {len(df):,}")
    print(f"   Unique papers: {len(df_dedup):,} (after deduplication)")
    
    if biobank_counts:
        print(f"\n📋 Papers by biobank:")
        for biobank, count in biobank_counts.items():
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"   {biobank}: {count:,} papers ({percentage:.1f}%)")
    
    print(f"\n🎯 Ready for analysis!")
    print(f"   Your CSV contains: Biobank, PMID, Title, Abstract, Journal, Year, MeSH_Terms")
    print(f"   Use this data in your next analysis script.")

if __name__ == "__main__":
    main()