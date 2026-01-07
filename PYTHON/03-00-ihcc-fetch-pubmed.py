#!/usr/bin/env python3
"""
03-00-bhem-fetch-pubmed.py
HEIM-Biobank v3.0: PubMed Publication Fetcher

Uses comprehensive search terms with multiple aliases per biobank.
Year-by-year chunking to bypass 9,999 record limit.
Month-by-month fallback for high-volume years.

Input:  DATA/ihcc_cohort_registry.json
Output: DATA/bhem_publications.csv
        DATA/query_log.json
        DATA/fetch_summary.json

Reference: Connolly et al. (2025) Communications Medicine 5:366
"""

import json
import csv
import time
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

try:
    from Bio import Entrez
except ImportError:
    print("ERROR: pip install biopython")
    sys.exit(1)

# =============================================================================
# NCBI CONFIGURATION
# =============================================================================
Entrez.email = "mc@manuelcorpas.com"
Entrez.tool = "HEIM-Biobank-v3"
Entrez.api_key = "44271e8e8b6d39627a80dc93092a718c6808"

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "DATA")
REGISTRY_PATH = os.path.join(DATA_DIR, "ihcc_cohort_registry.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "bhem_publications.csv")
QUERY_LOG_PATH = os.path.join(DATA_DIR, "query_log.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "fetch_summary.json")

# =============================================================================
# SETTINGS
# =============================================================================
BATCH_SIZE = 100          # Records per efetch call
RATE_LIMIT = 0.34         # ~3 req/sec (safe rate)
MAX_RETRIES = 3           # Retry attempts for NCBI 500 errors
RETRY_DELAY = 5           # Seconds to wait before retry
START_YEAR = 2000
END_YEAR = datetime.now().year

# =============================================================================
# COMPREHENSIVE SEARCH TERMS (multiple aliases per biobank)
# =============================================================================
BIOBANK_SEARCH_TERMS = {
    'ukb': ['UK Biobank', 'United Kingdom Biobank', 'U.K. Biobank', 'UK-Biobank'],
    'mvp': ['Million Veteran Program', 'Million Veterans Program', 'MVP biobank', 
            'MVP cohort', 'MVP genomics', 'Veterans Affairs Million Veteran Program',
            'VA Million Veteran Program'],
    'aou': ['All of Us Research Program', 'All of Us cohort', 'All of Us biobank',
            'AoU Research Program', 'Precision Medicine Initiative cohort'],
    'finngen': ['FinnGen', 'FinnGen biobank', 'FinnGen study', 'FinnGen cohort',
                'FinnGen consortium'],
    'estbb': ['Estonian Biobank', 'Estonia Biobank', 'Estonian Genome Center',
              'Estonian Health Cohort', 'Tartu Biobank', 'Estonian Genome Project'],
    'bbj': ['BioBank Japan', 'Biobank Japan', 'BBJ cohort'],
    'twb': ['Taiwan Biobank', 'TWB cohort'],
    'ckb': ['China Kadoorie Biobank', 'CKB cohort', 'Kadoorie Biobank'],
    'kgp': ['Korean Genome Project', 'Korea Biobank', 'Korean Biobank', 'KoGES'],
    'hunt': ['HUNT Study', 'HUNT cohort', 'Nord-Trondelag Health Study', 'HUNT4'],
    'decode': ['deCODE genetics', 'deCODE Genetics', 'Icelandic population study'],
    'gs': ['Generation Scotland', 'GS:SFHS'],
    'lifelines': ['LifeLines', 'Lifelines Cohort', 'LifeLines cohort study'],
    'constances': ['CONSTANCES', 'CONSTANCES cohort'],
    'nako': ['German National Cohort', 'NAKO Gesundheitsstudie', 'NAKO cohort'],
    'biovu': ['BioVU', 'Vanderbilt BioVU'],
    'qbb': ['Qatar Biobank', 'QBB cohort'],
    'emerge': ['eMERGE Network', 'eMERGE Consortium', 'eMERGE cohort'],
    'topmed': ['TOPMed', 'Trans-Omics for Precision Medicine'],
    'h3africa': ['H3Africa', 'Human Heredity and Health in Africa'],
    'awigen': ['AWI-Gen', 'Africa Wits-INDEPTH', 'AWI-Gen cohort'],
    'ugr': ['Uganda Genome Resource', 'Ugandan Genome', 'UGR cohort'],
    'gel': ['Genomics England', '100000 Genomes Project', '100,000 Genomes'],
    'epic': ['EPIC cohort', 'European Prospective Investigation into Cancer'],
    'framingham': ['Framingham Heart Study', 'Framingham cohort'],
    'whi': ["Women's Health Initiative", 'WHI cohort'],
    'aric': ['Atherosclerosis Risk in Communities', 'ARIC study', 'ARIC cohort'],
    'mesa': ['Multi-Ethnic Study of Atherosclerosis', 'MESA study', 'MESA cohort'],
    'jackson': ['Jackson Heart Study', 'JHS cohort'],
    'hchs_sol': ['Hispanic Community Health Study', 'HCHS/SOL', 'Study of Latinos'],
    'rotterdam': ['Rotterdam Study', 'Rotterdam cohort', 'ERGO study'],
    'twins_uk': ['TwinsUK', 'UK Twins Registry', 'St Thomas Twin Registry'],
    'alspac': ['ALSPAC', 'Avon Longitudinal Study', 'Children of the 90s'],
    'whitehall': ['Whitehall II', 'Whitehall study', 'Whitehall cohort'],
}

PREPRINT_SOURCES = [
    "medrxiv", "biorxiv", "arxiv", "research square", 
    "ssrn", "preprints", "chemrxiv", "f1000research"
]


def load_registry() -> Dict:
    """Load the IHCC cohort registry."""
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)


def safe_str(value) -> str:
    """Safely convert value to string, handling None."""
    if value is None:
        return ''
    return str(value)


def is_preprint(record: Dict) -> bool:
    """Check if a publication is a preprint."""
    source = safe_str(record.get('source')).lower()
    title = safe_str(record.get('title')).lower()
    journal = safe_str(record.get('journal')).lower()
    
    for preprint in PREPRINT_SOURCES:
        if preprint in source or preprint in title or preprint in journal:
            return True
    return False


def build_search_query(cohort_id: str, cohort: Dict) -> str:
    """Build comprehensive search query with multiple aliases."""
    # Check if we have predefined search terms
    if cohort_id in BIOBANK_SEARCH_TERMS:
        terms = BIOBANK_SEARCH_TERMS[cohort_id]
        query = ' OR '.join([f'"{term}"[All Fields]' for term in terms])
        return f'({query})'
    else:
        # Use the pubmed_query from registry
        return cohort['pubmed_query']


def search_pubmed_year(query: str, year: int) -> List[str]:
    """Search PubMed for a specific year with retry logic for server errors."""
    full_query = f'{query} AND ("{year}"[PDAT])'
    
    for attempt in range(MAX_RETRIES):
        try:
            handle = Entrez.esearch(db="pubmed", term=full_query, retmax=0)
            results = Entrez.read(handle)
            handle.close()
            
            count = int(results.get("Count", 0))
            
            if count == 0:
                return []
            
            if count > 9999:
                # Break down by month
                return search_year_by_month(query, year)
            
            # Get all PMIDs for this year
            handle = Entrez.esearch(db="pubmed", term=full_query, retmax=count)
            results = Entrez.read(handle)
            handle.close()
            
            return results.get("IdList", [])
            
        except Exception as e:
            error_msg = str(e)
            # Check for NCBI server errors (500, temporarily unavailable)
            if "500" in error_msg or "temporarily unavailable" in error_msg.lower() or "Backend failed" in error_msg:
                if attempt < MAX_RETRIES - 1:
                    print(f"      Server error {year}, retrying in {RETRY_DELAY}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                    continue
            print(f"    ERROR searching year {year}: {e}")
            return []
    
    return []


def search_year_by_month(query: str, year: int) -> List[str]:
    """Search a year month-by-month if results exceed 9,999."""
    all_pmids = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        month_query = f'{query} AND ("{year}/{month_str}"[PDAT])'
        
        for attempt in range(MAX_RETRIES):
            try:
                handle = Entrez.esearch(db="pubmed", term=month_query, retmax=9999)
                results = Entrez.read(handle)
                handle.close()
                
                pmids = results.get("IdList", [])
                all_pmids.extend(pmids)
                
                time.sleep(RATE_LIMIT)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                if "500" in error_msg or "temporarily unavailable" in error_msg.lower() or "Backend failed" in error_msg:
                    if attempt < MAX_RETRIES - 1:
                        print(f"      Server error {year}/{month_str}, retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                print(f"    ERROR searching {year}/{month_str}: {e}")
                break
    
    return all_pmids


def fetch_details(pmids: List[str], cohort_name: str) -> List[Dict]:
    """Fetch detailed records for a list of PMIDs."""
    if not pmids:
        return []
    
    records = []
    total_batches = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        for attempt in range(MAX_RETRIES):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    rettype="xml",
                    retmode="xml"
                )
                xml_data = handle.read()
                handle.close()
                
                root = ET.fromstring(xml_data)
                
                for article in root.findall('.//PubmedArticle'):
                    record = parse_article(article)
                    if record:
                        records.append(record)
                
                if total_batches > 5 and batch_num % 5 == 0:
                    print(f"    {cohort_name}: batch {batch_num}/{total_batches}")
                
                time.sleep(RATE_LIMIT)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                if "500" in error_msg or "temporarily unavailable" in error_msg.lower() or "Backend failed" in error_msg:
                    if attempt < MAX_RETRIES - 1:
                        print(f"      Server error batch {batch_num}, retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                print(f"    ERROR fetching batch {batch_num}/{total_batches}: {e}")
                time.sleep(1)
                break
    
    return records


def parse_article(article: ET.Element) -> Optional[Dict]:
    """Parse a PubMed article XML element into a dictionary."""
    try:
        # PMID
        pmid_elem = article.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else None
        if not pmid:
            return None
        
        # Title
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else ''
        
        # Abstract
        abstract_parts = []
        for abs_text in article.findall('.//AbstractText'):
            if abs_text.text:
                abstract_parts.append(abs_text.text)
        abstract = ' '.join(abstract_parts)
        
        # Journal
        journal_elem = article.find('.//Journal/Title')
        if journal_elem is None:
            journal_elem = article.find('.//Journal/ISOAbbreviation')
        journal = journal_elem.text if journal_elem is not None else ''
        
        # Year
        year = None
        year_elem = article.find('.//PubDate/Year')
        if year_elem is not None:
            year = year_elem.text
        else:
            medline_date = article.find('.//PubDate/MedlineDate')
            if medline_date is not None and medline_date.text:
                # Extract year from formats like "2020 Jan-Feb"
                import re
                match = re.search(r'(\d{4})', medline_date.text)
                if match:
                    year = match.group(1)
        
        # MeSH terms
        mesh_terms = []
        for mesh in article.findall('.//MeshHeading/DescriptorName'):
            if mesh.text:
                mesh_terms.append(mesh.text)
        
        return {
            'pmid': pmid,
            'title': title or '',
            'abstract': abstract or '',
            'journal': journal or '',
            'year': year or '',
            'mesh_terms': '; '.join(mesh_terms),
            'source': (journal or '').lower()
        }
    except Exception as e:
        return None


def fetch_cohort_publications(cohort: Dict, index: int, total: int) -> Tuple[str, List[Dict], Dict]:
    """Fetch all publications for a single cohort using year-by-year chunking."""
    cohort_id = cohort['cohort_id']
    cohort_name = cohort['name']
    start_year = max(cohort.get('operational_year', START_YEAR), START_YEAR)
    
    print(f"[{index}/{total}] {cohort_name}...", flush=True)
    
    # Build comprehensive search query
    query = build_search_query(cohort_id, cohort)
    
    timestamp = datetime.now().isoformat()
    all_pmids = []
    
    # Search year by year
    for year in range(start_year, END_YEAR + 1):
        pmids = search_pubmed_year(query, year)
        if pmids:
            print(f"    {year}: {len(pmids)} papers")
            all_pmids.extend(pmids)
        time.sleep(RATE_LIMIT)
    
    # Remove duplicates
    all_pmids = list(set(all_pmids))
    
    if not all_pmids:
        log_entry = {
            'cohort_id': cohort_id,
            'cohort_name': cohort_name,
            'query': query,
            'timestamp': timestamp,
            'pmids_found': 0,
            'records_retained': 0,
        }
        print(f"    Total: 0 publications")
        return cohort_id, [], log_entry
    
    print(f"    Fetching {len(all_pmids)} article details...")
    
    # Fetch details
    records = fetch_details(all_pmids, cohort_name)
    
    # Filter preprints and add metadata
    filtered = []
    preprint_count = 0
    
    for rec in records:
        if is_preprint(rec):
            preprint_count += 1
            continue
        
        rec['cohort_id'] = cohort_id
        rec['cohort_name'] = cohort_name
        rec['income_level'] = cohort['income_level']
        rec['region'] = cohort['region']
        rec['country'] = cohort['country']
        filtered.append(rec)
    
    log_entry = {
        'cohort_id': cohort_id,
        'cohort_name': cohort_name,
        'query': query,
        'timestamp': timestamp,
        'pmids_found': len(all_pmids),
        'records_fetched': len(records),
        'records_retained': len(filtered),
        'preprints_excluded': preprint_count,
    }
    
    print(f"    Total: {len(filtered)} publications" + 
          (f" (excl. {preprint_count} preprints)" if preprint_count else ""))
    
    return cohort_id, filtered, log_entry


def calculate_summary(publications: List[Dict], registry: Dict, elapsed: float) -> Dict:
    """Calculate summary statistics."""
    income_counts = {'HIC': 0, 'UMIC': 0, 'LMIC': 0, 'LIC': 0}
    cohort_counts = {}
    region_counts = {}
    year_counts = {}
    
    for pub in publications:
        income = pub.get('income_level', 'Unknown')
        if income in income_counts:
            income_counts[income] += 1
        
        cohort_ids = pub.get('cohort_id', 'Unknown').split('; ')
        for cid in cohort_ids:
            cohort_counts[cid] = cohort_counts.get(cid, 0) + 1
        
        region = pub.get('region', 'Unknown')
        region_counts[region] = region_counts.get(region, 0) + 1
        
        year = pub.get('year')
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1
    
    hic = income_counts['HIC']
    lmic_lic = income_counts['LMIC'] + income_counts['LIC']
    hic_lmic_ratio = round(hic / lmic_lic, 1) if lmic_lic > 0 else float('inf')
    
    return {
        'timestamp': datetime.now().isoformat(),
        'version': '3.0',
        'source': 'IHCC Global Cohort Atlas',
        'total_publications': len(publications),
        'total_cohorts': len(registry.get('cohorts', [])),
        'cohorts_with_publications': len(cohort_counts),
        'elapsed_seconds': round(elapsed, 1),
        'by_income_level': income_counts,
        'hic_lmic_ratio': hic_lmic_ratio,
        'by_region': region_counts,
        'by_year': dict(sorted(year_counts.items())),
        'top_cohorts': dict(sorted(cohort_counts.items(), key=lambda x: -x[1])[:20])
    }


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fetch PubMed publications for IHCC cohorts')
    parser.add_argument('--start-from', type=int, default=1, 
                        help='Start from cohort number (1-indexed), e.g. --start-from 71')
    args = parser.parse_args()
    start_from = args.start_from
    
    print("="*70)
    print("HEIM-Biobank v3.0: PubMed Publication Fetcher")
    print("="*70)
    print(f"Email: {Entrez.email}")
    print(f"API Key: {'Active' if Entrez.api_key else 'None'}")
    print(f"Search strategy: Year-by-year with month fallback")
    print(f"Date range: {START_YEAR}-{END_YEAR}")
    if start_from > 1:
        print(f"Resuming from cohort: {start_from}")
    print("="*70)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load registry
    print(f"\nLoading: {REGISTRY_PATH}")
    registry = load_registry()
    cohorts = registry['cohorts']
    print(f"Found {len(cohorts)} cohorts")
    
    income_summary = {}
    for c in cohorts:
        inc = c.get('income_level', 'Unknown')
        income_summary[inc] = income_summary.get(inc, 0) + 1
    print(f"  HIC: {income_summary.get('HIC', 0)}, UMIC: {income_summary.get('UMIC', 0)}, "
          f"LMIC: {income_summary.get('LMIC', 0)}, LIC: {income_summary.get('LIC', 0)}")
    
    # Load existing data if resuming
    all_publications = []
    query_log = []
    
    if start_from > 1 and os.path.exists(OUTPUT_PATH):
        print(f"\nLoading existing data for resume...")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_publications = list(reader)
        print(f"  Loaded {len(all_publications)} existing records")
    
    print(f"\nFetching publications...")
    print("-"*70)
    
    start_time = time.time()
    
    for i, cohort in enumerate(cohorts, 1):
        # Skip cohorts before start_from
        if i < start_from:
            continue
        cohort_id, pubs, log_entry = fetch_cohort_publications(cohort, i, len(cohorts))
        all_publications.extend(pubs)
        query_log.append(log_entry)
        
        # Save progress periodically
        if i % 10 == 0:
            temp_file = os.path.join(DATA_DIR, 'bhem_publications_temp.csv')
            fieldnames = ['pmid', 'title', 'abstract', 'journal', 'year', 'mesh_terms',
                          'cohort_id', 'cohort_name', 'income_level', 'region', 'country']
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_publications)
            print(f"    [Progress saved: {len(all_publications)} records]")
    
    elapsed = time.time() - start_time
    
    print("-"*70)
    print(f"Fetch complete in {elapsed/60:.1f} minutes")
    
    # Deduplicate
    print("\nDeduplicating...")
    seen_pmids = {}
    unique_publications = []
    
    for pub in all_publications:
        pmid = pub['pmid']
        if pmid not in seen_pmids:
            seen_pmids[pmid] = pub
            unique_publications.append(pub)
        else:
            existing = seen_pmids[pmid]
            if existing['cohort_id'] != pub['cohort_id']:
                existing['cohort_id'] += f"; {pub['cohort_id']}"
                existing['cohort_name'] += f"; {pub['cohort_name']}"
    
    print(f"Unique publications: {len(unique_publications)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary = calculate_summary(unique_publications, registry, elapsed)
    
    print(f"Total: {summary['total_publications']:,}")
    print(f"HIC:LMIC ratio: {summary['hic_lmic_ratio']}:1")
    print(f"\nBy income:")
    for income, count in summary['by_income_level'].items():
        pct = round(100 * count / len(unique_publications), 1) if unique_publications else 0
        print(f"  {income}: {count:,} ({pct}%)")
    
    print(f"\nTop 10 cohorts:")
    for i, (cohort, count) in enumerate(list(summary['top_cohorts'].items())[:10]):
        print(f"  {i+1}. {cohort}: {count:,}")
    
    # Save
    print(f"\n" + "-"*70)
    print(f"Saving: {OUTPUT_PATH}")
    
    fieldnames = ['pmid', 'title', 'abstract', 'journal', 'year', 'mesh_terms',
                  'cohort_id', 'cohort_name', 'income_level', 'region', 'country']
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(unique_publications)
    
    print(f"Saving: {QUERY_LOG_PATH}")
    with open(QUERY_LOG_PATH, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'total_queries': len(query_log),
            'queries': query_log
        }, f, indent=2)
    
    print(f"Saving: {SUMMARY_PATH}")
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Clean up temp file
    temp_file = os.path.join(DATA_DIR, 'bhem_publications_temp.csv')
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\n" + "="*70)
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"="*70)


if __name__ == "__main__":
    main()