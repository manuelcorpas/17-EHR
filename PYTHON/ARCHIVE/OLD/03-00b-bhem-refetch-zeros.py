#!/usr/bin/env python3
"""
03-00b-bhem-refetch-zeros.py
HEIM-Biobank v3.0: Re-fetch cohorts with 0 publications

Loads existing data, identifies 0-publication cohorts, and re-fetches
with broader/corrected search queries. Appends to existing CSV.

Input:  DATA/bhem_publications.csv (existing)
        DATA/query_log.json (to identify zeros)
Output: DATA/bhem_publications.csv (updated with new records)
        DATA/refetch_log.json
"""

import json
import csv
import time
import os
import sys
import re
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
Entrez.tool = "HEIM-Biobank-v3-refetch"
Entrez.api_key = "44271e8e8b6d39627a80dc93092a718c6808"

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "DATA")
REGISTRY_PATH = os.path.join(DATA_DIR, "ihcc_cohort_registry.json")
EXISTING_CSV = os.path.join(DATA_DIR, "bhem_publications.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "bhem_publications.csv")  # Will append
QUERY_LOG_PATH = os.path.join(DATA_DIR, "query_log.json")
REFETCH_LOG_PATH = os.path.join(DATA_DIR, "refetch_log.json")

# =============================================================================
# SETTINGS
# =============================================================================
BATCH_SIZE = 100
RATE_LIMIT = 0.34
START_YEAR = 2000
END_YEAR = datetime.now().year

PREPRINT_SOURCES = ["medrxiv", "biorxiv", "arxiv", "research square", "ssrn", "preprints"]

# =============================================================================
# BROADER SEARCH TERMS FOR LMIC COHORTS
# These use country-level genetics terms to catch more publications
# =============================================================================
BROAD_SEARCH_TERMS = {
    # African cohorts
    'kenyagenome': [
        'Kenya genome', 'Kenyan genetic', 'Kenya population genetic', 
        'Kenya genomic', 'East Africa genome Kenya'
    ],
    'tanzaniagen': [
        'Tanzania genome', 'Tanzanian genetic', 'Tanzania population genetic',
        'Tanzania genomic'
    ],
    'botswana': [
        'Botswana genome', 'Botswana genetic', 'Botswana population genetic',
        'Botswana genomic', 'Botswana HIV genetic'
    ],
    'ethiopian': [
        'Ethiopian genome', 'Ethiopia genetic', 'Ethiopian population genetic',
        'Addis Ababa genetic', 'Ethiopian genomic'
    ],
    'nigerian': [
        'Nigerian genome', '54gene', 'Nigeria genetic', 'Nigerian population genetic',
        'Lagos genetic', 'Ibadan genetic', 'Nigerian genomic'
    ],
    'ghana': [
        'Ghana genome', 'Ghanaian genetic', 'Ghana population genetic',
        'Accra genetic', 'Ghana genomic'
    ],
    'rwandan': [
        'Rwanda genome', 'Rwandan genetic', 'Rwanda population genetic',
        'Kigali genetic', 'Rwanda genomic', 'Rwanda Biomedical'
    ],
    'zambian': [
        'Zambia genome', 'Zambian genetic', 'Zambia population genetic',
        'Lusaka genetic', 'Zambia genomic'
    ],
    
    # Asian cohorts  
    'genome_india': [
        'GenomeIndia', 'Genome India', 'IndiGenomes', 'Indian genome project',
        'India population genetic', 'Indian genetic diversity', 'CSIR genome India',
        'Indian Biological Data Centre'
    ],
    'indigen': [
        'IndiGen', 'IndiGen program', 'CSIR IndiGen', 'Indian pilot genome'
    ],
    'vietnameses': [
        'Vietnam genome', 'Vietnamese genetic', 'Vietnam population genetic',
        'Hanoi genetic', 'Vietnamese genomic'
    ],
    'indonesian': [
        'Indonesia genome', 'Indonesian genetic', 'Indonesia population genetic',
        'Indonesian genomic', 'Eijkman genome', 'Indonesian diversity'
    ],
    'bangladeshi': [
        'Bangladesh genome', 'Bangladeshi genetic', 'Bangladesh population genetic',
        'Dhaka genetic', 'Bangladeshi genomic'
    ],
    'pakistani': [
        'Pakistan genome', 'Pakistani genetic', 'Pakistan population genetic',
        'Lahore genetic', 'Karachi genetic', 'Pakistani genomic'
    ],
    'filipino': [
        'Philippine genome', 'Filipino genetic', 'Philippines population genetic',
        'Manila genetic', 'Philippine Genome Center', 'Filipino genomic'
    ],
    'malaysian': [
        'Malaysian Cohort', 'Malaysia genome', 'Malaysian genetic',
        'Malaysia population genetic', 'Malaysian genomic', 'MyHDW'
    ],
    
    # Middle Eastern cohorts
    'emirati': [
        'Emirati genome', 'UAE genome', 'United Arab Emirates genetic',
        'Emirati population genetic', 'Abu Dhabi genetic', 'Dubai genetic'
    ],
    'egyptian': [
        'Egyptian genome', 'Egypt genetic', 'Egyptian population genetic',
        'Cairo genetic', 'Egyptian genomic'
    ],
    'moroccan': [
        'Moroccan genome', 'Morocco genetic', 'Moroccan population genetic',
        'Rabat genetic', 'Moroccan genomic'
    ],
    
    # Latin American cohorts
    'pgp': [
        'Peruvian Genome Project', 'Peru genome', 'Peruvian genetic',
        'Peru population genetic', 'Lima genetic', 'Andean genetic',
        'Peruvian genomic', 'Peru indigenous genetic'
    ],
    'colombian': [
        'Colombian genome', 'Colombia genetic', 'Colombian population genetic',
        'Bogota genetic', 'Colombian genomic', 'Colombia indigenous'
    ],
    'argentine': [
        'Argentine genome', 'Argentina genetic', 'Argentine population genetic',
        'Buenos Aires genetic', 'Argentine genomic'
    ],
    'chilean': [
        'Chilean genome', 'Chile genetic', 'Chilean population genetic',
        'Santiago genetic', 'Chilean genomic', 'Chile biobank'
    ],
    
    # Singapore (had 0)
    'sgbb': [
        'Singapore Biobank', 'SingHealth Tissue Repository', 
        'National University Hospital Singapore biobank',
        'Singapore population genetic', 'Singapore Health Study'
    ],
}


def load_existing_publications() -> Tuple[List[Dict], set]:
    """Load existing publications and return list + set of PMIDs."""
    publications = []
    pmids = set()
    
    if os.path.exists(EXISTING_CSV):
        with open(EXISTING_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                publications.append(row)
                pmids.add(row.get('pmid', ''))
    
    return publications, pmids


def load_zero_cohorts() -> List[str]:
    """Load cohort IDs that had 0 publications from query log."""
    zeros = []
    
    if os.path.exists(QUERY_LOG_PATH):
        with open(QUERY_LOG_PATH, 'r') as f:
            data = json.load(f)
            for q in data.get('queries', []):
                if q.get('records_retained', 0) == 0:
                    zeros.append(q.get('cohort_id'))
    
    return zeros


def load_registry() -> Dict:
    """Load cohort registry."""
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)


def safe_str(value) -> str:
    if value is None:
        return ''
    return str(value)


def is_preprint(record: Dict) -> bool:
    source = safe_str(record.get('source')).lower()
    title = safe_str(record.get('title')).lower()
    journal = safe_str(record.get('journal')).lower()
    for preprint in PREPRINT_SOURCES:
        if preprint in source or preprint in title or preprint in journal:
            return True
    return False


def build_broad_query(cohort_id: str) -> str:
    """Build broader search query."""
    if cohort_id in BROAD_SEARCH_TERMS:
        terms = BROAD_SEARCH_TERMS[cohort_id]
        query = ' OR '.join([f'"{term}"[All Fields]' for term in terms])
        return f'({query})'
    return None


def search_pubmed_year(query: str, year: int) -> List[str]:
    """Search PubMed for a specific year."""
    full_query = f'{query} AND ("{year}"[PDAT])'
    
    try:
        handle = Entrez.esearch(db="pubmed", term=full_query, retmax=0)
        results = Entrez.read(handle)
        handle.close()
        
        count = int(results.get("Count", 0))
        if count == 0:
            return []
        
        handle = Entrez.esearch(db="pubmed", term=full_query, retmax=min(count, 9999))
        results = Entrez.read(handle)
        handle.close()
        
        return results.get("IdList", [])
        
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


def fetch_details(pmids: List[str]) -> List[Dict]:
    """Fetch article details."""
    if not pmids:
        return []
    
    records = []
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i+BATCH_SIZE]
        
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
            
            root = ET.fromstring(xml_data)
            for article in root.findall('.//PubmedArticle'):
                record = parse_article(article)
                if record:
                    records.append(record)
            
            time.sleep(RATE_LIMIT)
        except Exception as e:
            print(f"      Fetch error: {e}")
    
    return records


def parse_article(article: ET.Element) -> Optional[Dict]:
    """Parse PubMed XML."""
    try:
        pmid_elem = article.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else None
        if not pmid:
            return None
        
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else ''
        
        abstract_parts = []
        for abs_text in article.findall('.//AbstractText'):
            if abs_text.text:
                abstract_parts.append(abs_text.text)
        abstract = ' '.join(abstract_parts)
        
        journal_elem = article.find('.//Journal/Title')
        if journal_elem is None:
            journal_elem = article.find('.//Journal/ISOAbbreviation')
        journal = journal_elem.text if journal_elem is not None else ''
        
        year = None
        year_elem = article.find('.//PubDate/Year')
        if year_elem is not None:
            year = year_elem.text
        else:
            medline_date = article.find('.//PubDate/MedlineDate')
            if medline_date is not None and medline_date.text:
                match = re.search(r'(\d{4})', medline_date.text)
                if match:
                    year = match.group(1)
        
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
    except:
        return None


def refetch_cohort(cohort: Dict, existing_pmids: set) -> Tuple[List[Dict], Dict]:
    """Re-fetch a cohort with broader search terms."""
    cohort_id = cohort['cohort_id']
    cohort_name = cohort['name']
    start_year = max(cohort.get('operational_year', START_YEAR), START_YEAR)
    
    query = build_broad_query(cohort_id)
    if not query:
        return [], {'cohort_id': cohort_id, 'status': 'no_broad_terms', 'new_pubs': 0}
    
    print(f"  {cohort_name}...")
    print(f"    Query: {query[:80]}...")
    
    all_pmids = []
    for year in range(start_year, END_YEAR + 1):
        pmids = search_pubmed_year(query, year)
        if pmids:
            # Filter out already-fetched PMIDs
            new_pmids = [p for p in pmids if p not in existing_pmids]
            if new_pmids:
                print(f"    {year}: {len(new_pmids)} new papers")
                all_pmids.extend(new_pmids)
        time.sleep(RATE_LIMIT)
    
    all_pmids = list(set(all_pmids))
    
    if not all_pmids:
        print(f"    No new publications found")
        return [], {'cohort_id': cohort_id, 'query': query, 'status': 'no_new', 'new_pubs': 0}
    
    print(f"    Fetching {len(all_pmids)} new articles...")
    records = fetch_details(all_pmids)
    
    # Filter and add metadata
    filtered = []
    for rec in records:
        if is_preprint(rec):
            continue
        rec['cohort_id'] = cohort_id
        rec['cohort_name'] = cohort_name
        rec['income_level'] = cohort['income_level']
        rec['region'] = cohort['region']
        rec['country'] = cohort['country']
        filtered.append(rec)
    
    print(f"    Added: {len(filtered)} publications")
    
    log = {
        'cohort_id': cohort_id,
        'cohort_name': cohort_name,
        'query': query,
        'status': 'success',
        'pmids_found': len(all_pmids),
        'new_pubs': len(filtered)
    }
    
    return filtered, log


def main():
    print("="*70)
    print("HEIM-Biobank v3.0: Re-fetch Zero-Publication Cohorts")
    print("="*70)
    
    # Load existing data
    print(f"\nLoading existing publications...")
    existing_pubs, existing_pmids = load_existing_publications()
    print(f"  Existing: {len(existing_pubs):,} publications, {len(existing_pmids):,} PMIDs")
    
    # Load zero cohorts
    print(f"\nIdentifying zero-publication cohorts...")
    zero_cohorts = load_zero_cohorts()
    print(f"  Found: {len(zero_cohorts)} cohorts with 0 publications")
    
    # Filter to only those we have broad terms for
    to_refetch = [c for c in zero_cohorts if c in BROAD_SEARCH_TERMS]
    print(f"  With broad search terms: {len(to_refetch)}")
    
    if not to_refetch:
        print("\nNo cohorts to re-fetch. Done.")
        return
    
    # Load registry
    registry = load_registry()
    cohort_map = {c['cohort_id']: c for c in registry['cohorts']}
    
    # Re-fetch
    print(f"\n" + "-"*70)
    print("Re-fetching with broader queries...")
    print("-"*70)
    
    new_publications = []
    refetch_log = []
    
    for cohort_id in to_refetch:
        if cohort_id not in cohort_map:
            continue
        cohort = cohort_map[cohort_id]
        pubs, log = refetch_cohort(cohort, existing_pmids)
        new_publications.extend(pubs)
        refetch_log.append(log)
        
        # Add new PMIDs to existing set to avoid duplicates
        for p in pubs:
            existing_pmids.add(p['pmid'])
    
    print(f"\n" + "-"*70)
    print(f"Total new publications: {len(new_publications)}")
    
    if new_publications:
        # Append to existing CSV
        print(f"\nAppending to: {OUTPUT_PATH}")
        
        fieldnames = ['pmid', 'title', 'abstract', 'journal', 'year', 'mesh_terms',
                      'cohort_id', 'cohort_name', 'income_level', 'region', 'country']
        
        with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerows(new_publications)
        
        print(f"  Appended {len(new_publications)} new records")
        print(f"  New total: {len(existing_pubs) + len(new_publications):,} publications")
    
    # Save refetch log
    print(f"Saving: {REFETCH_LOG_PATH}")
    with open(REFETCH_LOG_PATH, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cohorts_refetched': len(to_refetch),
            'new_publications': len(new_publications),
            'log': refetch_log
        }, f, indent=2)
    
    # Summary by cohort
    print(f"\n" + "="*70)
    print("SUMMARY BY COHORT")
    print("="*70)
    for log in refetch_log:
        print(f"  {log.get('cohort_id')}: {log.get('new_pubs', 0)} new publications")
    
    print(f"\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()