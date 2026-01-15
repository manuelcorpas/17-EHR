#!/usr/bin/env python3
"""
HEIM SEMANTIC INTEGRATION - PUBMED DATA RETRIEVAL
==================================================

Retrieves PubMed abstracts for all GBD diseases (2000-2025) with robust
error handling, checkpointing, and quality filtering.

FEATURES:
- Exponential backoff (1s → 2s → 4s → 8s → 16s max)
- 0.4s minimum delay between requests (NCBI rate limit)
- Checkpoint after each disease completes
- Atomic checkpoint writes (temp → rename)
- Quality filtering based on thresholds
- Graceful handling of malformed responses
- Sleep prevention via subprocess caffeinate

QUALITY FILTERS:
- Year Coverage ≥ 70% (18+ years with data)
- Abstract Coverage ≥ 95%
- Minimum Papers ≥ 50
- Composite Quality Score ≥ 80.0

OUTPUTS:
- DATA/05-SEMANTIC/PUBMED-RAW/{Disease}/pubmed_{year}.json
- DATA/05-SEMANTIC/heim_sem_quality_scores.csv
- DATA/05-SEMANTIC/heim_sem_disease_registry.json
- DATA/05-SEMANTIC/.fetch_complete marker

USAGE:
    python 05-01-heim-sem-fetch.py                    # Full run
    python 05-01-heim-sem-fetch.py --resume           # Resume from checkpoint
    python 05-01-heim-sem-fetch.py --fresh            # Ignore checkpoint, restart
    python 05-01-heim-sem-fetch.py --diseases "Malaria,Dengue"  # Specific diseases only

REQUIREMENTS:
    pip install biopython pandas tqdm requests
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from Bio import Entrez
import xml.etree.ElementTree as ET

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
PUBMED_RAW_DIR = SEMANTIC_DIR / "PUBMED-RAW"
CHECKPOINTS_DIR = SEMANTIC_DIR / "CHECKPOINTS"
LOGS_DIR = BASE_DIR / "LOGS"

# Marker files
SETUP_COMPLETE = SEMANTIC_DIR / ".setup_complete"
FETCH_COMPLETE = SEMANTIC_DIR / ".fetch_complete"
CHECKPOINT_FILE = CHECKPOINTS_DIR / "checkpoint_fetch.json"
CHECKPOINT_BACKUP = CHECKPOINTS_DIR / "checkpoint_fetch.json.bak"

# Output files
QUALITY_SCORES_FILE = SEMANTIC_DIR / "heim_sem_quality_scores.csv"
DISEASE_REGISTRY_FILE = SEMANTIC_DIR / "heim_sem_disease_registry.json"
MESH_MAPPING_FILE = SEMANTIC_DIR / "gbd_mesh_mapping.json"

# NCBI Configuration
NCBI_EMAIL = "mc@manuelcorpas.com"
NCBI_TOOL = "HEIM-Semantic-Fetch"
NCBI_API_KEY = "44271e8e8b6d39627a80dc93092a718c6808"  # Optional but recommended

# Rate limiting
MIN_REQUEST_DELAY = 0.4  # seconds between requests
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds

# Time range
START_YEAR = 2000
END_YEAR = 2025

# Quality thresholds
MIN_YEAR_COVERAGE = 0.70  # 70% of years must have data
MIN_ABSTRACT_COVERAGE = 0.95  # 95% of papers must have abstracts
MIN_PAPERS = 50  # Minimum papers for inclusion
MIN_QUALITY_SCORE = 80.0  # Composite quality score

# GBD Disease List (from HEIM v6 biobank - 175 diseases)
GBD_DISEASES = [
    "Scabies",
    "Drowning",
    "Guinea worm disease",
    "Animal contact",
    "Cysticercosis",
    "Schistosomiasis",
    "Dengue",
    "Malaria",
    "Lymphatic filariasis",
    "Encephalitis",
    "Typhoid and paratyphoid",
    "Conflict and terrorism",
    "Viral skin diseases",
    "Paralytic ileus and intestinal obstruction",
    "Iodine deficiency",
    "Leishmaniasis",
    "Otitis media",
    "Invasive Non-typhoidal Salmonella (iNTS)",
    "Road injuries",
    "Urticaria",
    "Diarrheal diseases",
    "Tuberculosis",
    "Pertussis",
    "Hemoglobinopathies and hemolytic anemias",
    "Measles",
    "Meningitis",
    "Foreign body",
    "Fungal skin diseases",
    "Fire, heat, and hot substances",
    "Protein-energy malnutrition",
    "Vitamin A deficiency",
    "Exposure to mechanical forces",
    "Pruritus",
    "Headache disorders",
    "Dietary iron deficiency",
    "Acne vulgaris",
    "HIV/AIDS",
    "Tetanus",
    "Chronic obstructive pulmonary disease",
    "Lower respiratory infections",
    "Upper respiratory infections",
    "COVID-19",
    "Appendicitis",
    "Rabies",
    "Onchocerciasis",
    "Cystic echinococcosis",
    "Self-harm",
    "Low back pain",
    "Pneumoconiosis",
    "Leukemia",
    "Cardiomyopathy and myocarditis",
    "Inguinal, femoral, and abdominal hernia",
    "Decubitus ulcer",
    "Age-related and other hearing loss",
    "Drug use disorders",
    "Neck pain",
    "Oral disorders",
    "Congenital birth defects",
    "Exposure to forces of nature",
    "Intestinal nematode infections",
    "Falls",
    "Neonatal disorders",
    "Osteoarthritis",
    "Schizophrenia",
    "Other neglected tropical diseases",
    "Rheumatic heart disease",
    "Leprosy",
    "Sexually transmitted infections excluding HIV",
    "Idiopathic developmental intellectual disability",
    "Autism spectrum disorders",
    "Asthma",
    "Cirrhosis and other chronic liver diseases",
    "Police conflict and executions",
    "Bipolar disorder",
    "Ischemic heart disease",
    "Upper digestive system diseases",
    "Anxiety disorders",
    "Gallbladder and biliary diseases",
    "Pancreatitis",
    "Vascular intestinal disorders",
    "Other musculoskeletal disorders",
    "Stroke",
    "Non-Hodgkin lymphoma",
    "Diphtheria",
    "Other sense organ diseases",
    "Other transport injuries",
    "Blindness and vision loss",
    "Poisonings",
    "Adverse effects of medical treatment",
    "Varicella and herpes zoster",
    "Chronic kidney disease",
    "Depressive disorders",
    "Acute hepatitis",
    "Other unintentional injuries",
    "Dermatitis",
    "Psoriasis",
    "Aortic aneurysm",
    "Bacterial skin diseases",
    "Other skin and subcutaneous diseases",
    "Tracheal, bronchus, and lung cancer",
    "Atrial fibrillation and flutter",
    "Yellow fever",
    "Diabetes mellitus",
    "Eating disorders",
    "Interpersonal violence",
    "Alcohol use disorders",
    "Alopecia areata",
    "Acute glomerulonephritis",
    "Endocarditis",
    "Stomach cancer",
    "Non-rheumatic valvular heart disease",
    "Hodgkin lymphoma",
    "Hypertensive heart disease",
    "Other digestive diseases",
    "Trachoma",
    "Malignant skin melanoma",
    "Interstitial lung disease and pulmonary sarcoidosis",
    "Zika virus",
    "Ebola",
    "Testicular cancer",
    "Endocrine, metabolic, blood, and immune disorders",
    "Colon and rectum cancer",
    "Urinary diseases and male infertility",
    "Gynecological diseases",
    "Ovarian cancer",
    "Sudden infant death syndrome",
    "Lip and oral cavity cancer",
    "Nasopharynx cancer",
    "Other pharynx cancer",
    "Gout",
    "Prostate cancer",
    "Parkinson's disease",
    "Uterine cancer",
    "Cervical cancer",
    "Breast cancer",
    "Rheumatoid arthritis",
    "Maternal disorders",
    "Other unspecified infectious diseases",
    "Environmental heat and cold exposure",
    "Food-borne trematodiases",
    "African trypanosomiasis",
    "Chagas disease",
    "Other intestinal infectious diseases",
    "Other nutritional deficiencies",
    "Gallbladder and biliary tract cancer",
    "Esophageal cancer",
    "Eye cancer",
    "Multiple sclerosis",
    "Motor neuron disease",
    "Alzheimer's disease and other dementias",
    "Other neurological disorders",
    "Inflammatory bowel disease",
    "Other chronic respiratory diseases",
    "Pulmonary Arterial Hypertension",
    "Other cardiovascular and circulatory diseases",
    "Attention-deficit/hyperactivity disorder",
    "Conduct disorder",
    "Other mental disorders",
    "Lower extremity peripheral arterial disease",
    "Neuroblastoma and other peripheral nervous cell tumors",
    "Soft tissue and other extraosseous sarcomas",
    "Liver cancer",
    "Malignant neoplasm of bone and articular cartilage",
    "Other malignant neoplasms",
    "Other neoplasms",
    "Multiple myeloma",
    "Mesothelioma",
    "Thyroid cancer",
    "Brain and central nervous system cancer",
    "Bladder cancer",
    "Kidney cancer",
    "Non-melanoma skin cancer",
    "Larynx cancer",
    "Pancreatic cancer",
    "Idiopathic epilepsy"
]

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with file and console handlers."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"05-01-fetch-{timestamp}.log"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("heim_fetch")
    logger.setLevel(logging.DEBUG)

    # File handler - detailed
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    # Console handler - info and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resumable processing."""

    def __init__(self, checkpoint_file: Path, backup_file: Path):
        self.checkpoint_file = checkpoint_file
        self.backup_file = backup_file
        self.data = self._load_or_create()

    def _load_or_create(self) -> Dict:
        """Load existing checkpoint or create new one."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                # Validate checksum
                stored_checksum = data.get("checksum", "")
                data_copy = {k: v for k, v in data.items() if k != "checksum"}
                computed_checksum = hashlib.sha256(
                    json.dumps(data_copy, sort_keys=True).encode()
                ).hexdigest()[:16]
                if stored_checksum == computed_checksum:
                    return data
                else:
                    print("  Warning: Checkpoint checksum mismatch, starting fresh")
            except (json.JSONDecodeError, KeyError):
                print("  Warning: Corrupt checkpoint, starting fresh")

        return {
            "script": "05-01-heim-sem-fetch.py",
            "version": "1.0",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_diseases": [],
            "in_progress": None,
            "failed_diseases": [],
            "disease_stats": {},
            "total_papers": 0,
        }

    def save(self):
        """Save checkpoint with atomic write."""
        self.data["last_updated"] = datetime.now().isoformat()

        # Compute checksum
        data_copy = {k: v for k, v in self.data.items() if k != "checksum"}
        self.data["checksum"] = hashlib.sha256(
            json.dumps(data_copy, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Backup existing checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(self.backup_file)

        # Write to temp file then rename (atomic)
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        temp_file.rename(self.checkpoint_file)

    def mark_in_progress(self, disease: str):
        """Mark disease as in progress."""
        self.data["in_progress"] = disease
        self.save()

    def mark_completed(self, disease: str, stats: Dict):
        """Mark disease as completed."""
        if disease not in self.data["completed_diseases"]:
            self.data["completed_diseases"].append(disease)
        self.data["disease_stats"][disease] = stats
        self.data["total_papers"] += stats.get("total_papers", 0)
        self.data["in_progress"] = None
        self.save()

    def mark_failed(self, disease: str, error: str):
        """Mark disease as failed."""
        self.data["failed_diseases"].append({
            "disease": disease,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.data["in_progress"] = None
        self.save()

    def is_completed(self, disease: str) -> bool:
        """Check if disease is already completed."""
        return disease in self.data["completed_diseases"]

    def get_pending_diseases(self, all_diseases: List[str]) -> List[str]:
        """Get list of diseases not yet completed."""
        completed = set(self.data["completed_diseases"])
        return [d for d in all_diseases if d not in completed]

# =============================================================================
# PUBMED RETRIEVAL
# =============================================================================

class PubMedFetcher:
    """Fetch PubMed data with rate limiting and error handling."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_request_time = 0
        self.mesh_mapping = self._load_mesh_mapping()

        # Configure Entrez
        Entrez.email = NCBI_EMAIL
        Entrez.tool = NCBI_TOOL
        if NCBI_API_KEY:
            Entrez.api_key = NCBI_API_KEY

    def _load_mesh_mapping(self) -> Dict:
        """Load GBD to MeSH mapping file."""
        if MESH_MAPPING_FILE.exists():
            with open(MESH_MAPPING_FILE, 'r') as f:
                data = json.load(f)
                return data.get('mappings', {})
        return {}

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_REQUEST_DELAY:
            time.sleep(MIN_REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff on failure."""
        backoff = INITIAL_BACKOFF
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)[:100]}"
                )
                if attempt < MAX_RETRIES - 1:
                    self.logger.debug(f"Waiting {backoff}s before retry...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)  # Max 16s backoff

        raise last_error

    def search_disease_year(self, disease: str, year: int) -> List[str]:
        """Search PubMed for disease in a specific year."""
        # Check for MeSH mapping first
        if disease in self.mesh_mapping:
            mapping = self.mesh_mapping[disease]
            base_query = mapping.get('combined', f'"{disease}"[Title/Abstract]')
            self.logger.debug(f"  Using MeSH mapping for {disease}")
        else:
            # Fallback to simple title/abstract search
            base_query = f'"{disease}"[Title/Abstract]'

        # Add year filter
        query = f'({base_query}) AND ("{year}"[PDAT])'

        def do_search():
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=100000,
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()
            return record

        record = self._retry_with_backoff(do_search)
        pmids = record.get("IdList", [])
        self.logger.debug(f"  {disease} {year}: {len(pmids)} PMIDs found")
        return pmids

    def fetch_abstracts(self, pmids: List[str], batch_size: int = 500) -> List[Dict]:
        """Fetch abstract data for PMIDs in batches."""
        all_papers = []

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]

            def do_fetch():
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="xml",
                    retmode="xml"
                )
                xml_data = handle.read()
                handle.close()
                return xml_data

            try:
                xml_data = self._retry_with_backoff(do_fetch)
                papers = self._parse_xml(xml_data)
                all_papers.extend(papers)
            except Exception as e:
                self.logger.error(f"Failed to fetch batch: {e}")
                continue

        return all_papers

    def _parse_xml(self, xml_data: bytes) -> List[Dict]:
        """Parse PubMed XML response into paper records."""
        papers = []

        try:
            root = ET.fromstring(xml_data)

            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)

        except ET.ParseError as e:
            self.logger.error(f"XML parse error: {e}")

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[Dict]:
        """Parse single PubMed article into dict."""
        try:
            medline = article.find('.//MedlineCitation')
            if medline is None:
                return None

            pmid_elem = medline.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                return None

            # Title
            title_elem = medline.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_parts = []
            for abs_text in medline.findall('.//AbstractText'):
                if abs_text.text:
                    label = abs_text.get('Label', '')
                    if label:
                        abstract_parts.append(f"{label}: {abs_text.text}")
                    else:
                        abstract_parts.append(abs_text.text)
            abstract = " ".join(abstract_parts)

            # Year
            year = None
            pub_date = medline.find('.//PubDate')
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                if year_elem is not None:
                    year = year_elem.text

            # MeSH terms
            mesh_terms = []
            for mesh in medline.findall('.//MeshHeading/DescriptorName'):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "mesh_terms": mesh_terms,
                "has_abstract": len(abstract) > 0
            }

        except Exception as e:
            self.logger.debug(f"Error parsing article: {e}")
            return None

# =============================================================================
# QUALITY METRICS
# =============================================================================

def compute_quality_metrics(disease_data: Dict[int, List[Dict]]) -> Dict:
    """Compute quality metrics for a disease."""
    years_with_data = sum(1 for papers in disease_data.values() if papers)
    total_years = END_YEAR - START_YEAR + 1
    year_coverage = years_with_data / total_years

    total_papers = sum(len(papers) for papers in disease_data.values())
    papers_with_abstracts = sum(
        sum(1 for p in papers if p.get("has_abstract"))
        for papers in disease_data.values()
    )
    abstract_coverage = papers_with_abstracts / total_papers if total_papers > 0 else 0

    # Unique PMIDs (de-duplication check)
    all_pmids = []
    for papers in disease_data.values():
        all_pmids.extend(p["pmid"] for p in papers)
    unique_pmids = len(set(all_pmids))
    duplicate_rate = 1 - (unique_pmids / len(all_pmids)) if all_pmids else 0

    # Compute composite quality score (0-100)
    quality_score = (
        (year_coverage * 40) +
        (abstract_coverage * 40) +
        ((1 - duplicate_rate) * 20)
    )

    return {
        "total_papers": total_papers,
        "unique_pmids": unique_pmids,
        "papers_with_abstracts": papers_with_abstracts,
        "year_coverage": year_coverage,
        "abstract_coverage": abstract_coverage,
        "duplicate_rate": duplicate_rate,
        "quality_score": quality_score,
        "years_with_data": years_with_data,
        "total_years": total_years,
        "passes_filters": (
            year_coverage >= MIN_YEAR_COVERAGE and
            abstract_coverage >= MIN_ABSTRACT_COVERAGE and
            total_papers >= MIN_PAPERS and
            quality_score >= MIN_QUALITY_SCORE
        )
    }

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_disease(
    disease: str,
    fetcher: PubMedFetcher,
    output_dir: Path,
    logger: logging.Logger
) -> Dict:
    """Process a single disease: search, fetch, save."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {disease}")
    logger.info(f"{'='*60}")

    # Create disease directory
    disease_folder = disease.replace("/", "_").replace(" ", "_")
    disease_dir = output_dir / disease_folder
    disease_dir.mkdir(parents=True, exist_ok=True)

    disease_data = {}
    years = list(range(START_YEAR, END_YEAR + 1))

    # Progress bar for years
    for year in tqdm(years, desc=f"  {disease[:30]}", leave=False):
        try:
            # Search for PMIDs
            pmids = fetcher.search_disease_year(disease, year)

            if pmids:
                # Fetch abstracts
                papers = fetcher.fetch_abstracts(pmids)

                # Save to file
                output_file = disease_dir / f"pubmed_{year}.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        "disease": disease,
                        "year": year,
                        "retrieved_at": datetime.now().isoformat(),
                        "pmid_count": len(pmids),
                        "paper_count": len(papers),
                        "papers": papers
                    }, f, indent=2)

                disease_data[year] = papers
                logger.debug(f"    {year}: {len(papers)} papers saved")
            else:
                disease_data[year] = []
                logger.debug(f"    {year}: no papers found")

        except Exception as e:
            logger.error(f"    {year}: failed - {e}")
            disease_data[year] = []

    # Compute quality metrics
    metrics = compute_quality_metrics(disease_data)
    metrics["disease"] = disease
    metrics["disease_folder"] = disease_folder

    # Log summary
    logger.info(f"  Total papers: {metrics['total_papers']:,}")
    logger.info(f"  Unique PMIDs: {metrics['unique_pmids']:,}")
    logger.info(f"  Year coverage: {metrics['year_coverage']:.1%}")
    logger.info(f"  Abstract coverage: {metrics['abstract_coverage']:.1%}")
    logger.info(f"  Quality score: {metrics['quality_score']:.1f}")
    logger.info(f"  Passes filters: {'Yes' if metrics['passes_filters'] else 'No'}")

    return metrics

def save_quality_scores(all_metrics: List[Dict], output_file: Path):
    """Save quality scores to CSV."""
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("quality_score", ascending=False)
    df.to_csv(output_file, index=False)

def save_disease_registry(all_metrics: List[Dict], output_file: Path):
    """Save disease registry with metadata."""
    registry = {
        "generated_at": datetime.now().isoformat(),
        "time_range": {"start": START_YEAR, "end": END_YEAR},
        "quality_thresholds": {
            "min_year_coverage": MIN_YEAR_COVERAGE,
            "min_abstract_coverage": MIN_ABSTRACT_COVERAGE,
            "min_papers": MIN_PAPERS,
            "min_quality_score": MIN_QUALITY_SCORE
        },
        "diseases": {
            m["disease"]: {
                "folder": m["disease_folder"],
                "passes_filters": m["passes_filters"],
                "quality_score": m["quality_score"],
                "total_papers": m["total_papers"]
            }
            for m in all_metrics
        },
        "summary": {
            "total_diseases": len(all_metrics),
            "diseases_passing": sum(1 for m in all_metrics if m["passes_filters"]),
            "total_papers": sum(m["total_papers"] for m in all_metrics),
            "total_unique_pmids": sum(m["unique_pmids"] for m in all_metrics)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(registry, f, indent=2)

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HEIM Semantic Pipeline - PubMed Data Retrieval"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (default behavior)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore checkpoint and start fresh"
    )
    parser.add_argument(
        "--diseases", type=str,
        help="Comma-separated list of specific diseases to process"
    )
    parser.add_argument(
        "--diseases-file", type=str,
        help="File with disease names (one per line) to process"
    )
    parser.add_argument(
        "--no-caffeinate", action="store_true",
        help="Don't use caffeinate (for subprocess calls)"
    )

    args = parser.parse_args()

    # Handle caffeinate wrapper
    if not args.no_caffeinate and sys.platform == "darwin":
        # Re-run with caffeinate
        print("Starting with sleep prevention enabled (caffeinate)")
        cmd = ["caffeinate", "-i", "-s", "-d", sys.executable, __file__, "--no-caffeinate"]
        if args.fresh:
            cmd.append("--fresh")
        if args.diseases_file:
            cmd.extend(["--diseases-file", args.diseases_file])
        elif args.diseases:
            cmd.extend(["--diseases", args.diseases])
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # Setup logging
    logger = setup_logging()

    print("\n" + "=" * 70)
    print(" HEIM SEMANTIC PIPELINE - PUBMED DATA RETRIEVAL")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Time range: {START_YEAR}-{END_YEAR}")
    print(f"  Output: {PUBMED_RAW_DIR}")

    # Check setup complete
    if not SETUP_COMPLETE.exists():
        print("\n  ERROR: Setup not complete. Run 05-00-heim-sem-setup.py first.")
        sys.exit(1)

    # Initialize checkpoint
    if args.fresh and CHECKPOINT_FILE.exists():
        print("  Removing existing checkpoint (--fresh flag)")
        CHECKPOINT_FILE.unlink()

    checkpoint = CheckpointManager(CHECKPOINT_FILE, CHECKPOINT_BACKUP)

    # Determine diseases to process
    if args.diseases_file:
        with open(args.diseases_file, 'r') as f:
            diseases = [line.strip() for line in f if line.strip()]
        print(f"  Processing diseases from file: {len(diseases)}")
    elif args.diseases:
        diseases = [d.strip() for d in args.diseases.split(",")]
        print(f"  Processing specific diseases: {len(diseases)}")
    else:
        diseases = GBD_DISEASES
        print(f"  Processing all GBD diseases: {len(diseases)}")

    pending = checkpoint.get_pending_diseases(diseases)
    print(f"  Already completed: {len(diseases) - len(pending)}")
    print(f"  Pending: {len(pending)}")

    if not pending:
        print("\n  All diseases already processed!")
    else:
        # Initialize fetcher
        fetcher = PubMedFetcher(logger)

        # Process each disease
        all_metrics = []

        # Load existing metrics from checkpoint
        for disease in checkpoint.data["completed_diseases"]:
            if disease in checkpoint.data["disease_stats"]:
                all_metrics.append(checkpoint.data["disease_stats"][disease])

        print(f"\n  Starting retrieval...")
        print(f"  Estimated time: {len(pending) * 3} - {len(pending) * 8} minutes")
        print("-" * 70)

        for i, disease in enumerate(pending, 1):
            logger.info(f"\n[{i}/{len(pending)}] {disease}")

            checkpoint.mark_in_progress(disease)

            try:
                metrics = process_disease(
                    disease, fetcher, PUBMED_RAW_DIR, logger
                )
                all_metrics.append(metrics)
                checkpoint.mark_completed(disease, metrics)

            except KeyboardInterrupt:
                logger.warning("\nInterrupted by user. Progress saved to checkpoint.")
                print("\n  Interrupted. Resume with: python 05-01-heim-sem-fetch.py --resume")
                sys.exit(130)

            except Exception as e:
                logger.error(f"Failed to process {disease}: {e}")
                checkpoint.mark_failed(disease, str(e))
                continue

        # Save final outputs
        print("\n" + "-" * 70)
        print("  Saving final outputs...")

        save_quality_scores(all_metrics, QUALITY_SCORES_FILE)
        logger.info(f"  Quality scores saved: {QUALITY_SCORES_FILE}")

        save_disease_registry(all_metrics, DISEASE_REGISTRY_FILE)
        logger.info(f"  Disease registry saved: {DISEASE_REGISTRY_FILE}")

        # Create completion marker
        with open(FETCH_COMPLETE, 'w') as f:
            json.dump({
                "completed_at": datetime.now().isoformat(),
                "diseases_processed": len(all_metrics),
                "diseases_passing": sum(1 for m in all_metrics if m["passes_filters"]),
                "total_papers": sum(m["total_papers"] for m in all_metrics)
            }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print(" RETRIEVAL COMPLETE")
    print("=" * 70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total diseases: {len(checkpoint.data['completed_diseases'])}")
    print(f"  Total papers: {checkpoint.data['total_papers']:,}")
    if checkpoint.data["failed_diseases"]:
        print(f"  Failed: {len(checkpoint.data['failed_diseases'])}")
    print(f"\n  Next step: python 05-02-heim-sem-embed.py")

if __name__ == "__main__":
    main()
