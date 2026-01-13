#!/usr/bin/env python3
"""
03-00c-bhem-validate-refetch.py
HEIM-Biobank v3.0: Validate and smart re-fetch

1. Validates suspicious results (Rwanda) by checking MeSH terms for genomics relevance
2. Uses smarter queries for still-missing cohorts (author names, institutions)
3. Reports which cohorts genuinely have no indexed publications
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
Entrez.tool = "HEIM-Biobank-v3-validate"
Entrez.api_key = "44271e8e8b6d39627a80dc93092a718c6808"

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "DATA")
EXISTING_CSV = os.path.join(DATA_DIR, "bhem_publications.csv")
VALIDATED_CSV = os.path.join(DATA_DIR, "bhem_publications_validated.csv")
VALIDATION_LOG = os.path.join(DATA_DIR, "validation_log.json")

BATCH_SIZE = 100
RATE_LIMIT = 0.34
START_YEAR = 2000
END_YEAR = datetime.now().year

PREPRINT_SOURCES = ["medrxiv", "biorxiv", "arxiv", "research square", "ssrn", "preprints"]

# =============================================================================
# GENOMICS-RELEVANT MESH TERMS (for validation)
# =============================================================================
GENOMICS_MESH = [
    'genome', 'genomic', 'genetic', 'genotype', 'allele', 'polymorphism',
    'sequencing', 'exome', 'variant', 'mutation', 'SNP', 'GWAS', 
    'whole genome', 'DNA', 'RNA', 'transcriptome', 'epigenetic',
    'population genetics', 'ancestry', 'admixture', 'haplotype',
    'biobank', 'cohort studies', 'prospective studies'
]

# =============================================================================
# SMARTER SEARCH STRATEGIES
# Known PIs, institutions, grant numbers, specific project names
# =============================================================================
SMART_SEARCHES = {
    'ethiopian': {
        'name': 'Ethiopian Genome Project',
        'queries': [
            '"Ethiopian" AND "genome sequencing"',
            '"Addis Ababa University" AND "genetics"',
            '"Ethiopian" AND ("whole genome" OR "exome")',
            '"Ethiopia" AND "population genetics"',
            '"Gurja" AND "Ethiopian"',  # Known PI
        ]
    },
    'ghana': {
        'name': 'Ghana Genome Project', 
        'queries': [
            '"Ghana" AND "genome sequencing"',
            '"University of Ghana" AND "genetics"',
            '"Ghanaian" AND ("whole genome" OR "exome")',
            '"Ghana" AND "population genetics"',
            '"Noguchi Memorial" AND "genetics"',
        ]
    },
    'kenyagenome': {
        'name': 'Kenya Genome Project',
        'queries': [
            '"Kenya" AND "genome sequencing"',
            '"KEMRI" AND "genetics"',
            '"Kenyan" AND ("whole genome" OR "exome")',
            '"Kenya" AND "population genetics"',
            '"Kilifi" AND "genomic"',
        ]
    },
    'tanzaniagen': {
        'name': 'Tanzania Human Genetics',
        'queries': [
            '"Tanzania" AND "genome sequencing"',
            '"Muhimbili" AND "genetics"',
            '"Tanzanian" AND ("whole genome" OR "exome")',
            '"Tanzania" AND "population genetics"',
        ]
    },
    'emirati': {
        'name': 'Emirati Genome Program',
        'queries': [
            '"United Arab Emirates" AND "genome"',
            '"Emirati" AND "genetic"',
            '"UAE" AND "population genetics"',
            '"Khalifa University" AND "genome"',
            '"MBZUAI" AND "genome"',
        ]
    },
    'botswana': {
        'name': 'Botswana Genome Project',
        'queries': [
            '"Botswana" AND "genome sequencing"',
            '"Botswana Harvard" AND "genetic"',
            '"Botswana" AND "population genetics"',
        ]
    },
    'bangladeshi': {
        'name': 'Bangladesh Genome Project',
        'queries': [
            '"Bangladesh" AND "genome sequencing"',
            '"Bangladeshi" AND ("whole genome" OR "exome")',
            '"icddr,b" AND "genetic"',
            '"Dhaka" AND "population genetics"',
        ]
    },
    'colombian': {
        'name': 'Colombian Genome Project',
        'queries': [
            '"Colombia" AND "genome sequencing"',
            '"Colombian" AND "population genetics"',
            '"Universidad de Antioquia" AND "genetics"',
            '"Colombian" AND "ancestry"',
        ]
    },
    'argentine': {
        'name': 'Argentine Genomic Medicine',
        'queries': [
            '"Argentina" AND "genome sequencing"',
            '"Argentine" AND "population genetics"',
            '"Buenos Aires" AND "genomic"',
            '"PoblAr"',  # Argentine population genetics project
        ]
    },
    'chilean': {
        'name': 'Chilean National Biobank',
        'queries': [
            '"Chile" AND "genome sequencing"',
            '"Chilean" AND "population genetics"',
            '"Universidad de Chile" AND "genetics"',
            '"Mapuche" AND "genetic"',  # Indigenous Chilean genetics
        ]
    },
    'zambian': {
        'name': 'Zambia Genome Project',
        'queries': [
            '"Zambia" AND "genome sequencing"',
            '"Zambian" AND "genetic"',
            '"University of Zambia" AND "genetics"',
            '"Lusaka" AND "genomic"',
        ]
    },
    'moroccan': {
        'name': 'Moroccan Biobank',
        'queries': [
            '"Morocco" AND "genome"',
            '"Moroccan" AND "population genetics"',
            '"Rabat" AND "genetics"',
        ]
    },
}


def is_genomics_relevant(record: Dict) -> bool:
    """Check if a publication is genomics/biobank relevant based on MeSH and title."""
    mesh = record.get('mesh_terms', '').lower()
    title = record.get('title', '').lower()
    abstract = record.get('abstract', '').lower()
    
    text = f"{mesh} {title} {abstract}"
    
    relevance_score = 0
    for term in GENOMICS_MESH:
        if term.lower() in text:
            relevance_score += 1
    
    return relevance_score >= 2  # At least 2 genomics-relevant terms


def load_existing_data() -> Tuple[List[Dict], set]:
    """Load existing publications."""
    publications = []
    pmids = set()
    
    with open(EXISTING_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            publications.append(row)
            pmids.add(row.get('pmid', ''))
    
    return publications, pmids


def validate_cohort_publications(publications: List[Dict], cohort_id: str) -> Tuple[List[Dict], int, int]:
    """Validate publications for a cohort, filtering out non-genomics papers."""
    cohort_pubs = [p for p in publications if p.get('cohort_id') == cohort_id]
    
    valid = []
    removed = 0
    
    for pub in cohort_pubs:
        if is_genomics_relevant(pub):
            valid.append(pub)
        else:
            removed += 1
    
    return valid, len(cohort_pubs), removed


def search_pubmed(query: str) -> List[str]:
    """Search PubMed with a query."""
    full_query = f'{query} AND ("{START_YEAR}"[PDAT] : "{END_YEAR}"[PDAT])'
    
    try:
        handle = Entrez.esearch(db="pubmed", term=full_query, retmax=500)
        results = Entrez.read(handle)
        handle.close()
        return results.get("IdList", [])
    except Exception as e:
        print(f"      Search error: {e}")
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
        }
    except:
        return None


def smart_search_cohort(cohort_id: str, cohort_info: Dict, existing_pmids: set, registry: Dict) -> Tuple[List[Dict], Dict]:
    """Smart search using multiple targeted queries."""
    name = cohort_info['name']
    queries = cohort_info['queries']
    
    print(f"\n  {name}...")
    
    all_pmids = set()
    query_results = {}
    
    for query in queries:
        pmids = search_pubmed(query)
        new_pmids = [p for p in pmids if p not in existing_pmids and p not in all_pmids]
        if new_pmids:
            print(f"    '{query[:50]}...': {len(new_pmids)} new")
            query_results[query] = len(new_pmids)
            all_pmids.update(new_pmids)
        time.sleep(RATE_LIMIT)
    
    if not all_pmids:
        print(f"    No publications found")
        return [], {'cohort_id': cohort_id, 'status': 'no_results', 'queries_tried': len(queries)}
    
    # Fetch and validate
    print(f"    Fetching {len(all_pmids)} candidates...")
    records = fetch_details(list(all_pmids))
    
    # Filter for genomics relevance
    valid = []
    for rec in records:
        if is_genomics_relevant(rec):
            # Get cohort metadata from registry
            cohort_data = next((c for c in registry['cohorts'] if c['cohort_id'] == cohort_id), None)
            if cohort_data:
                rec['cohort_id'] = cohort_id
                rec['cohort_name'] = cohort_data['name']
                rec['income_level'] = cohort_data['income_level']
                rec['region'] = cohort_data['region']
                rec['country'] = cohort_data['country']
                valid.append(rec)
    
    print(f"    Validated: {len(valid)} genomics-relevant publications")
    
    return valid, {
        'cohort_id': cohort_id,
        'name': name,
        'status': 'success' if valid else 'no_valid',
        'candidates': len(all_pmids),
        'valid': len(valid),
        'query_results': query_results
    }


def main():
    print("="*70)
    print("HEIM-Biobank v3.0: Validate and Smart Re-fetch")
    print("="*70)
    
    # Load existing data
    print(f"\nLoading existing publications...")
    publications, existing_pmids = load_existing_data()
    print(f"  Total: {len(publications):,} publications")
    
    # =========================================================================
    # STEP 1: Validate Rwanda (suspicious high count)
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 1: Validating Rwanda Biomedical Center Cohort")
    print("="*70)
    
    valid_rwanda, original, removed = validate_cohort_publications(publications, 'rwandan')
    print(f"  Original: {original} papers")
    print(f"  Genomics-relevant: {len(valid_rwanda)} papers")
    print(f"  Removed (non-genomics): {removed} papers")
    
    # Sample some removed papers
    rwanda_pubs = [p for p in publications if p.get('cohort_id') == 'rwandan']
    non_genomics = [p for p in rwanda_pubs if not is_genomics_relevant(p)]
    
    if non_genomics:
        print(f"\n  Sample of removed non-genomics papers:")
        for p in non_genomics[:5]:
            print(f"    - {p.get('title', '')[:70]}...")
    
    # =========================================================================
    # STEP 2: Smart search for still-missing cohorts
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 2: Smart search for missing cohorts")
    print("="*70)
    
    # Load registry for metadata
    with open(os.path.join(DATA_DIR, "ihcc_cohort_registry.json")) as f:
        registry = json.load(f)
    
    new_publications = []
    search_log = []
    
    for cohort_id, cohort_info in SMART_SEARCHES.items():
        pubs, log = smart_search_cohort(cohort_id, cohort_info, existing_pmids, registry)
        new_publications.extend(pubs)
        search_log.append(log)
        
        for p in pubs:
            existing_pmids.add(p['pmid'])
    
    # =========================================================================
    # STEP 3: Create validated dataset
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 3: Creating validated dataset")
    print("="*70)
    
    # Remove non-genomics Rwanda papers from main dataset
    validated_pubs = [p for p in publications if not (p.get('cohort_id') == 'rwandan' and not is_genomics_relevant(p))]
    
    # Add new validated publications
    validated_pubs.extend(new_publications)
    
    print(f"  Original dataset: {len(publications):,}")
    print(f"  Removed non-genomics Rwanda: {removed}")
    print(f"  Added new validated: {len(new_publications)}")
    print(f"  Final validated: {len(validated_pubs):,}")
    
    # Save validated dataset
    print(f"\nSaving: {VALIDATED_CSV}")
    fieldnames = ['pmid', 'title', 'abstract', 'journal', 'year', 'mesh_terms',
                  'cohort_id', 'cohort_name', 'income_level', 'region', 'country']
    
    with open(VALIDATED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(validated_pubs)
    
    # Save validation log
    print(f"Saving: {VALIDATION_LOG}")
    with open(VALIDATION_LOG, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'original_count': len(publications),
            'validated_count': len(validated_pubs),
            'rwanda_validation': {
                'original': original,
                'valid': len(valid_rwanda),
                'removed': removed
            },
            'smart_searches': search_log,
            'new_publications': len(new_publications)
        }, f, indent=2)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nRwanda validation:")
    print(f"  {original} → {len(valid_rwanda)} (removed {removed} non-genomics)")
    
    print(f"\nSmart search results:")
    for log in search_log:
        status = "✓" if log.get('valid', 0) > 0 else "✗"
        print(f"  {status} {log.get('name', log.get('cohort_id'))}: {log.get('valid', 0)} papers")
    
    print(f"\nCohorts with genuinely 0 indexed publications:")
    genuinely_zero = [log for log in search_log if log.get('valid', 0) == 0]
    for log in genuinely_zero:
        print(f"  - {log.get('name', log.get('cohort_id'))}")
    
    print(f"\n  Note: These projects may be too new for PubMed-indexed publications,")
    print(f"  or publish under different names not captured by our queries.")
    
    print(f"\n" + "="*70)
    print("COMPLETE")
    print(f"Use {VALIDATED_CSV} for downstream analysis")
    print("="*70)


if __name__ == "__main__":
    main()