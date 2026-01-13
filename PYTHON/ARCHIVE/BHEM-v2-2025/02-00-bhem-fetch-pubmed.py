#!/usr/bin/env python3
"""
02-00-bhem-fetch-pubmed.py
==========================
BHEM Step 1: Fetch PubMed Publications (PRECISION VERSION)

Retrieves publications that actually USE biobank data, not just mention biobanks.
Uses [Title/Abstract] field constraints and requires exact phrase matching.

KEY PRECISION FEATURES:
- Uses [Title/Abstract] instead of [All Fields] to avoid metadata noise
- Requires exact biobank name phrases in quotes
- Only searches from realistic operational dates (e.g., UK Biobank from 2010+)
- Excludes generic terms that cause false positives

INPUT:  PubMed API
OUTPUT: DATA/bhem_publications.csv

USAGE:
    python 02-00-bhem-fetch-pubmed.py
"""

import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Entrez configuration
Entrez.email = "mc.admin@manuelcorpas.com"
Entrez.tool = "BHEM-DataRetrieval"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "bhem_publications.csv"
PROGRESS_FILE = DATA_DIR / "bhem_fetch_progress.csv"

# Rate limiting
REQUESTS_PER_SECOND = 3
SLEEP_TIME = 1.0 / REQUESTS_PER_SECOND

# Preprint servers to exclude
PREPRINT_PATTERNS = [
    'medRxiv', 'bioRxiv', 'Research Square', 'arXiv', 'ChemRxiv',
    'PeerJ Preprints', 'F1000Research', 'Preprints.org', 'SSRN'
]


# =============================================================================
# BIOBANK REGISTRY - PRECISION QUERIES
# =============================================================================
# Each biobank has:
#   - query: PubMed search string using [Title/Abstract] for precision
#   - start_year: When the biobank became operational for research output
#   - country, region: For equity analysis
#
# IMPORTANT: start_year reflects when publications USING the biobank data
# would realistically appear, not when the biobank was founded.

BIOBANK_REGISTRY = {
    # =========================================================================
    # TIER 1: Major established biobanks
    # =========================================================================
    'uk_biobank': {
        'name': 'UK Biobank',
        # Exact phrase only - no aliases that could match other biobanks
        'query': '"UK Biobank"[Title/Abstract]',
        # Data access began ~2012, significant output from 2014+
        'start_year': 2012,
        'country': 'United Kingdom',
        'region': 'EUR',
        'cohort_size': 500000
    },
    'mvp': {
        'name': 'Million Veteran Program',
        'query': '"Million Veteran Program"[Title/Abstract]',
        # MVP launched 2011, first major papers ~2016
        'start_year': 2015,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 1000000
    },
    'finngen': {
        'name': 'FinnGen',
        'query': '"FinnGen"[Title/Abstract]',
        # FinnGen launched 2017, papers from 2019+
        'start_year': 2019,
        'country': 'Finland',
        'region': 'EUR',
        'cohort_size': 500000
    },
    'all_of_us': {
        'name': 'All of Us Research Program',
        'query': '"All of Us Research Program"[Title/Abstract]',
        # All of Us launched 2018, data available 2020+
        'start_year': 2019,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 750000
    },
    'estonian_biobank': {
        'name': 'Estonian Biobank',
        'query': '("Estonian Biobank"[Title/Abstract] OR "Estonian Genome Center"[Title/Abstract])',
        'start_year': 2004,
        'country': 'Estonia',
        'region': 'EUR',
        'cohort_size': 200000
    },
    
    # =========================================================================
    # TIER 2: Major Asian biobanks
    # =========================================================================
    'biobank_japan': {
        'name': 'BioBank Japan',
        'query': '"BioBank Japan"[Title/Abstract]',
        'start_year': 2005,
        'country': 'Japan',
        'region': 'WPR',
        'cohort_size': 270000
    },
    'china_kadoorie': {
        'name': 'China Kadoorie Biobank',
        'query': '"China Kadoorie Biobank"[Title/Abstract]',
        'start_year': 2008,
        'country': 'China',
        'region': 'WPR',
        'cohort_size': 512000
    },
    'taiwan_biobank': {
        'name': 'Taiwan Biobank',
        'query': '"Taiwan Biobank"[Title/Abstract]',
        'start_year': 2014,
        'country': 'Taiwan',
        'region': 'WPR',
        'cohort_size': 200000
    },
    
    # =========================================================================
    # TIER 3: Nordic biobanks
    # =========================================================================
    'hunt': {
        'name': 'HUNT Study',
        'query': '("HUNT Study"[Title/Abstract] OR "Nord-Trøndelag Health Study"[Title/Abstract])',
        'start_year': 1995,
        'country': 'Norway',
        'region': 'EUR',
        'cohort_size': 230000
    },
    'decode': {
        'name': 'deCODE Genetics',
        'query': '"deCODE"[Title/Abstract]',
        'start_year': 2000,
        'country': 'Iceland',
        'region': 'EUR',
        'cohort_size': 160000
    },
    'danish_national': {
        'name': 'Danish National Biobank',
        'query': '"Danish National Biobank"[Title/Abstract]',
        'start_year': 2014,
        'country': 'Denmark',
        'region': 'EUR',
        'cohort_size': 800000
    },
    'swedish_twin': {
        'name': 'Swedish Twin Registry',
        'query': '"Swedish Twin Registry"[Title/Abstract]',
        'start_year': 1990,
        'country': 'Sweden',
        'region': 'EUR',
        'cohort_size': 200000
    },
    
    # =========================================================================
    # TIER 4: European biobanks
    # =========================================================================
    'generation_scotland': {
        'name': 'Generation Scotland',
        'query': '"Generation Scotland"[Title/Abstract]',
        'start_year': 2010,
        'country': 'United Kingdom',
        'region': 'EUR',
        'cohort_size': 24000
    },
    'lifelines': {
        'name': 'LifeLines',
        'query': '"LifeLines"[Title/Abstract]',
        'start_year': 2010,
        'country': 'Netherlands',
        'region': 'EUR',
        'cohort_size': 167000
    },
    'constances': {
        'name': 'CONSTANCES',
        'query': '"CONSTANCES"[Title/Abstract]',
        'start_year': 2015,
        'country': 'France',
        'region': 'EUR',
        'cohort_size': 220000
    },
    'nako': {
        'name': 'German National Cohort (NAKO)',
        'query': '("NAKO"[Title/Abstract] OR "German National Cohort"[Title/Abstract])',
        'start_year': 2016,
        'country': 'Germany',
        'region': 'EUR',
        'cohort_size': 205000
    },
    
    # =========================================================================
    # TIER 5: US healthcare biobanks
    # =========================================================================
    'biovu': {
        'name': 'BioVU',
        'query': '"BioVU"[Title/Abstract]',
        'start_year': 2009,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 300000
    },
    'geisinger_mycode': {
        'name': 'Geisinger MyCode',
        'query': '"MyCode"[Title/Abstract]',
        'start_year': 2012,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 300000
    },
    'mgb_biobank': {
        'name': 'Mass General Brigham Biobank',
        'query': '("Mass General Brigham Biobank"[Title/Abstract] OR "Partners Biobank"[Title/Abstract])',
        'start_year': 2012,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 140000
    },
    
    # =========================================================================
    # TIER 6: Middle East biobanks
    # =========================================================================
    'qatar_biobank': {
        'name': 'Qatar Biobank',
        'query': '"Qatar Biobank"[Title/Abstract]',
        'start_year': 2014,
        'country': 'Qatar',
        'region': 'EMR',
        'cohort_size': 80000
    },
    
    # =========================================================================
    # TIER 7: African biobanks (EQUITY CRITICAL)
    # =========================================================================
    'h3africa': {
        'name': 'H3Africa',
        'query': '("H3Africa"[Title/Abstract] OR "Human Heredity and Health in Africa"[Title/Abstract])',
        'start_year': 2013,
        'country': 'Pan-African',
        'region': 'AFR',
        'cohort_size': 75000
    },
    'uganda_genome': {
        'name': 'Uganda Genome Resource',
        'query': '"Uganda Genome Resource"[Title/Abstract]',
        'start_year': 2017,
        'country': 'Uganda',
        'region': 'AFR',
        'cohort_size': 10000
    },
    'awigen': {
        'name': 'AWI-Gen',
        'query': '"AWI-Gen"[Title/Abstract]',
        'start_year': 2016,
        'country': 'Pan-African',
        'region': 'AFR',
        'cohort_size': 12000
    },
    
    # =========================================================================
    # TIER 8: Latin American biobanks (EQUITY CRITICAL)
    # =========================================================================
    'mexico_city': {
        'name': 'Mexico City Prospective Study',
        'query': '"Mexico City Prospective Study"[Title/Abstract]',
        'start_year': 2003,
        'country': 'Mexico',
        'region': 'AMR',
        'cohort_size': 160000
    },
    'epigen_brazil': {
        'name': 'EPIGEN-Brazil',
        'query': '"EPIGEN-Brazil"[Title/Abstract]',
        'start_year': 2013,
        'country': 'Brazil',
        'region': 'AMR',
        'cohort_size': 6500
    },
    
    # =========================================================================
    # TIER 9: Consortia and networks
    # =========================================================================
    'emerge': {
        'name': 'eMERGE Network',
        'query': '"eMERGE"[Title/Abstract]',
        'start_year': 2009,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 300000
    },
    'topmed': {
        'name': 'TOPMed',
        'query': '("TOPMed"[Title/Abstract] OR "Trans-Omics for Precision Medicine"[Title/Abstract])',
        'start_year': 2016,
        'country': 'United States',
        'region': 'AMR',
        'cohort_size': 180000
    },
}


# =============================================================================
# PUBMED SEARCH FUNCTIONS
# =============================================================================

def search_biobank(biobank_id: str, biobank_info: dict) -> list:
    """
    Search PubMed for publications from a specific biobank.
    Only searches from the biobank's operational start year.
    """
    name = biobank_info['name']
    query = biobank_info['query']
    start_year = biobank_info['start_year']
    
    logger.info(f"Searching: {name}")
    logger.info(f"  Query: {query}")
    logger.info(f"  Start year: {start_year}")
    
    all_pmids = []
    current_year = datetime.now().year
    
    # Search year by year from start_year
    for year in range(start_year, current_year + 1):
        year_query = f'({query}) AND ("{year}"[PDAT])'
        
        try:
            # Get count first
            handle = Entrez.esearch(db="pubmed", term=year_query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            
            count = int(result["Count"])
            
            if count == 0:
                continue
            
            logger.info(f"  {year}: {count:,} papers")
            
            # Fetch PMIDs
            handle = Entrez.esearch(db="pubmed", term=year_query, retmax=count)
            result = Entrez.read(handle)
            handle.close()
            
            all_pmids.extend(result["IdList"])
            
            time.sleep(SLEEP_TIME)
            
        except Exception as e:
            logger.error(f"  Error searching {year}: {e}")
            continue
    
    logger.info(f"  Total PMIDs: {len(all_pmids):,}")
    return list(set(all_pmids))  # Deduplicate


def fetch_article_details(pmids: list, biobank_id: str, biobank_info: dict) -> list:
    """Fetch detailed article data for a list of PMIDs."""
    if not pmids:
        return []
    
    logger.info(f"Fetching details for {len(pmids):,} articles...")
    
    articles = []
    batch_size = 100
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pmids) - 1) // batch_size + 1
        
        if batch_num % 10 == 0 or batch_num == total_batches:
            logger.info(f"  Batch {batch_num}/{total_batches}")
        
        try:
            handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
            
            root = ET.fromstring(xml_data)
            
            for article_xml in root.findall('.//PubmedArticle'):
                article = parse_article(article_xml, biobank_id, biobank_info)
                if article:
                    articles.append(article)
            
            time.sleep(SLEEP_TIME)
            
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            continue
    
    # Filter preprints
    before_filter = len(articles)
    articles = [a for a in articles if not is_preprint(a.get('journal', ''))]
    preprints_removed = before_filter - len(articles)
    
    if preprints_removed > 0:
        logger.info(f"  Removed {preprints_removed} preprints")
    
    logger.info(f"  Final: {len(articles):,} peer-reviewed articles")
    return articles


def parse_article(article_xml, biobank_id: str, biobank_info: dict) -> dict:
    """Parse a PubMed article XML element."""
    try:
        # PMID
        pmid_elem = article_xml.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else None
        if not pmid:
            return None
        
        # Title
        title_elem = article_xml.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else ''
        
        # Abstract
        abstract_parts = []
        for abstract_text in article_xml.findall('.//AbstractText'):
            if abstract_text.text:
                abstract_parts.append(abstract_text.text)
        abstract = ' '.join(abstract_parts)
        
        # Journal
        journal_elem = article_xml.find('.//Journal/Title')
        if journal_elem is None:
            journal_elem = article_xml.find('.//Journal/ISOAbbreviation')
        journal = journal_elem.text if journal_elem is not None else ''
        
        # Year
        year = None
        year_elem = article_xml.find('.//PubDate/Year')
        if year_elem is not None:
            year = year_elem.text
        else:
            medline_date = article_xml.find('.//PubDate/MedlineDate')
            if medline_date is not None and medline_date.text:
                match = re.search(r'(\d{4})', medline_date.text)
                if match:
                    year = match.group(1)
        
        # MeSH Terms
        mesh_terms = []
        for mesh in article_xml.findall('.//MeshHeading/DescriptorName'):
            if mesh.text:
                mesh_terms.append(mesh.text)
        
        return {
            'pmid': pmid,
            'biobank_id': biobank_id,
            'biobank_name': biobank_info['name'],
            'title': title,
            'abstract': abstract,
            'journal': journal,
            'year': int(year) if year else None,
            'mesh_terms': '; '.join(mesh_terms),
            'country': biobank_info['country'],
            'region': biobank_info['region']
        }
        
    except Exception as e:
        logger.error(f"Error parsing article: {e}")
        return None


def is_preprint(journal: str) -> bool:
    """Check if a journal is a preprint server."""
    if not journal:
        return False
    journal_lower = journal.lower()
    for pattern in PREPRINT_PATTERNS:
        if pattern.lower() in journal_lower:
            return True
    return False


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

def load_progress() -> set:
    """Load set of already-processed biobank IDs."""
    if PROGRESS_FILE.exists():
        df = pd.read_csv(PROGRESS_FILE)
        return set(df['biobank_id'].unique())
    return set()


def save_progress(biobank_id: str, article_count: int):
    """Save progress after processing a biobank."""
    progress_data = {
        'biobank_id': [biobank_id],
        'articles': [article_count],
        'timestamp': [datetime.now().isoformat()]
    }
    
    df = pd.DataFrame(progress_data)
    
    if PROGRESS_FILE.exists():
        existing = pd.read_csv(PROGRESS_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(PROGRESS_FILE, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BHEM STEP 1: Fetch PubMed Publications")
    print("PRECISION VERSION - Title/Abstract search with date constraints")
    print("=" * 70)
    print(f"Email: {Entrez.email}")
    print(f"Biobanks: {len(BIOBANK_REGISTRY)}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Load existing progress
    completed = load_progress()
    if completed:
        print(f"\n📋 Resuming: {len(completed)} biobanks already processed")
    
    # Load existing articles if resuming
    all_articles = []
    if OUTPUT_FILE.exists() and completed:
        existing_df = pd.read_csv(OUTPUT_FILE)
        all_articles = existing_df.to_dict('records')
        print(f"   Loaded {len(all_articles):,} existing articles")
    
    # Process each biobank
    for biobank_id, biobank_info in BIOBANK_REGISTRY.items():
        if biobank_id in completed:
            print(f"\n⏭️  Skipping {biobank_info['name']} (already done)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Fetching: {biobank_info['name']} ({biobank_info['country']})")
        print(f"{'='*60}")
        
        # Search
        pmids = search_biobank(biobank_id, biobank_info)
        
        if not pmids:
            logger.info(f"No articles found for {biobank_info['name']}")
            save_progress(biobank_id, 0)
            continue
        
        # Fetch details
        articles = fetch_article_details(pmids, biobank_id, biobank_info)
        all_articles.extend(articles)
        
        # Save progress
        save_progress(biobank_id, len(articles))
        
        # Save intermediate results
        df = pd.DataFrame(all_articles)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(articles):,} articles for {biobank_info['name']}")
        print(f"   Total so far: {len(all_articles):,}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FETCH COMPLETE")
    print(f"{'='*70}")
    
    df = pd.DataFrame(all_articles)
    
    if len(df) > 0:
        print(f"\n📊 Summary:")
        print(f"   Total articles: {len(df):,}")
        print(f"   Unique PMIDs: {df['pmid'].nunique():,}")
        print(f"   Biobanks: {df['biobank_id'].nunique()}")
        
        years = df['year'].dropna()
        if len(years) > 0:
            print(f"   Year range: {int(years.min())}-{int(years.max())}")
        
        print(f"\n📋 Articles per biobank:")
        for biobank_id, count in df['biobank_id'].value_counts().head(15).items():
            name = BIOBANK_REGISTRY.get(biobank_id, {}).get('name', biobank_id)
            print(f"   {name}: {count:,}")
        
        remaining = len(df['biobank_id'].unique()) - 15
        if remaining > 0:
            print(f"   ... and {remaining} more biobanks")
        
        print(f"\n📍 Articles by region:")
        for region, count in df['region'].value_counts().items():
            print(f"   {region}: {count:,}")
    
    print(f"\n💾 Output: {OUTPUT_FILE}")
    print(f"\n✅ STEP 1 COMPLETE!")
    print(f"\n➡️  Next step: python 02-01-bhem-map-diseases.py")


if __name__ == "__main__":
    main()