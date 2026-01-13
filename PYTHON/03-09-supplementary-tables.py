#!/usr/bin/env python3
"""
03-09-supplementary-tables.py
=============================
HEIM-Biobank: Generate Supplementary Tables for Manuscript

Addresses multiple reviewer comments requiring documentation:
- Comment 3: Burden Score calibration documentation
- Comment 5: Full Capacity_Penalty specification
- Comment 8: Exact search strings per biobank
- Comment 17-18: Software versions and methodology transparency

USAGE:
    python 03-09-supplementary-tables.py
"""

import json
import logging
import sys
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# =============================================================================
# HARDWARE OPTIMIZATION (M3 Ultra: 32 cores, 256GB RAM)
# =============================================================================

N_CORES = min(multiprocessing.cpu_count() - 4, 28)

# NumPy/Pandas optimization
pd.set_option('compute.use_numexpr', True)
pd.set_option('mode.copy_on_write', True)

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Detect project root: DATA should be at project root, not in PYTHON/
if (SCRIPT_DIR / "DATA").exists():
    BASE_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "DATA").exists():
    BASE_DIR = SCRIPT_DIR.parent
else:
    # Fallback: assume current working directory
    BASE_DIR = Path.cwd()
    if not (BASE_DIR / "DATA").exists():
        BASE_DIR = SCRIPT_DIR.parent

DATA_DIR = BASE_DIR / "DATA"
OUTPUT_DIR = BASE_DIR / "ANALYSIS" / "03-09-SUPPLEMENTARY-TABLES"

# Input files
INPUT_IHCC_REGISTRY = DATA_DIR / "ihcc_cohort_registry.json"
INPUT_GBD_REGISTRY = DATA_DIR / "gbd_disease_registry.json"
INPUT_QUERY_LOG = DATA_DIR / "query_log.json"
INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"


# =============================================================================
# BIOBANK SEARCH TERMS (from 03-00-ihcc-fetch-pubmed.py)
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


# =============================================================================
# TABLE S1: SEARCH QUERIES
# =============================================================================

def generate_search_query_table(ihcc_registry: Dict, query_log: Dict) -> pd.DataFrame:
    """
    Generate Supplementary Table S1: PubMed Search Queries
    
    Addresses Comment 8: "Please add to Methods or Supplement:
    - The exact search strings or template used per biobank
    - How you handled ambiguous acronyms"
    """
    logger.info("Generating Table S1: Search Queries...")
    
    rows = []
    
    # From IHCC registry
    cohorts = ihcc_registry.get('cohorts', [])
    for cohort in cohorts:
        cid = cohort.get('id', cohort.get('cohort_id', ''))
        
        # Get predefined search terms if available
        search_terms = BIOBANK_SEARCH_TERMS.get(cid, [])
        
        # Get query from registry
        registry_query = cohort.get('pubmed_query', '')
        
        # Build full query
        if search_terms:
            aliases = '; '.join(search_terms)
            query_template = ' OR '.join([f'"{term}"[All Fields]' for term in search_terms])
        else:
            aliases = cohort.get('name', cid)
            query_template = registry_query
        
        # Get results from query log
        records_found = 0
        for q in query_log.get('queries', []):
            if q.get('cohort_id') == cid or q.get('biobank_id') == cid:
                records_found = q.get('records_retained', 0)
                break
        
        rows.append({
            'cohort_id': cid,
            'cohort_name': cohort.get('name', cohort.get('cohort_name', cid)),
            'country': cohort.get('country', ''),
            'income_level': cohort.get('income_level', ''),
            'search_aliases': aliases,
            'field_restriction': '[All Fields]',
            'date_range': '2000-2025',
            'query_template': query_template[:300],  # Truncate long queries
            'publications_retrieved': records_found,
            'disambiguation_notes': 'Multiple aliases used; full name + acronym' if len(search_terms) > 1 else ''
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('publications_retrieved', ascending=False)
    
    return df


# =============================================================================
# TABLE S2: BURDEN SCORE CALIBRATION
# =============================================================================

def generate_burden_calibration_table(gbd_registry: Dict) -> pd.DataFrame:
    """
    Generate Supplementary Table S2: Burden Score Calibration
    
    OPTIMIZED: Vectorized numpy operations.
    
    Addresses Comment 3: "Please describe the objective function or 
    criteria used for calibration"
    """
    logger.info("Generating Table S2: Burden Score Calibration...")
    
    # GBD priority diseases for calibration reference
    CALIBRATION_DISEASES = [
        'Ischemic heart disease',
        'Stroke',
        'Chronic obstructive pulmonary disease',
        'Lower respiratory infections',
        'Neonatal disorders',
        'Tracheal, bronchus, and lung cancer',
        'Diabetes mellitus',
        'Diarrheal diseases',
        'Road injuries',
        'HIV/AIDS',
        'Tuberculosis',
        'Malaria',
        'Alzheimer disease and other dementias',
        'Depressive disorders',
        'Back pain',
        'Cirrhosis and other chronic liver diseases'
    ]
    
    # Extract data into arrays for vectorized computation
    disease_ids = list(gbd_registry.keys())
    n_diseases = len(disease_ids)
    
    dalys_raw = np.array([gbd_registry[d].get('dalys', 0) for d in disease_ids])
    dalys_millions = np.where(dalys_raw > 1000, dalys_raw / 1_000_000, dalys_raw)
    
    # Vectorized burden score computation
    dalys_for_log = dalys_millions * 1_000_000
    burden_scores = np.where(dalys_for_log > 0, 10.0 * np.log10(dalys_for_log + 1), 0)
    
    # Check calibration diseases (vectorized string matching)
    calibration_lower = [c.lower() for c in CALIBRATION_DISEASES]
    is_calibration = np.array([
        any(cal in d.lower() or d.lower() in cal for cal in calibration_lower)
        for d in disease_ids
    ])
    
    # Get GBD level2 and global south priority
    gbd_level2 = [gbd_registry[d].get('gbd_level2', '') for d in disease_ids]
    gs_priority = [gbd_registry[d].get('global_south_priority', False) for d in disease_ids]
    
    # Build dataframe efficiently
    df = pd.DataFrame({
        'disease': disease_ids,
        'gbd_level2': gbd_level2,
        'dalys_millions': np.round(dalys_millions, 2),
        'burden_score': np.round(burden_scores, 2),
        'is_calibration_disease': is_calibration,
        'global_south_priority': gs_priority
    })
    
    # Compute ranks (vectorized)
    df = df.sort_values('dalys_millions', ascending=False)
    df['daly_rank'] = np.arange(1, len(df) + 1)
    
    df = df.sort_values('burden_score', ascending=False)
    df['burden_score_rank'] = np.arange(1, len(df) + 1)
    
    # Compute rank correlation for calibration diseases
    calibration_df = df[df['is_calibration_disease']]
    if len(calibration_df) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(calibration_df['daly_rank'], calibration_df['burden_score_rank'])
        logger.info(f"   Calibration correlation (DALY vs Burden rank): ρ = {corr:.4f}")
    
    return df.sort_values('burden_score', ascending=False)


# =============================================================================
# TABLE S3: FORMULA SPECIFICATIONS
# =============================================================================

def generate_formula_specifications() -> str:
    """
    Generate Supplementary Table S3: Complete Formula Specifications
    
    Addresses Comment 5: "The exact computation (inputs, scaling, and 
    thresholds) should be fully specified"
    """
    logger.info("Generating Table S3: Formula Specifications...")
    
    content = """# Supplementary Table S3: HEIM Formula Specifications

## 1. Burden Score

### Primary Formula (when deaths and prevalence available)

```
Burden_Score = (w_d × DALYs) + (w_m × Deaths) + [w_p × log₁₀(Prevalence)]
```

**Parameters:**
| Symbol | Description | Default Value | Units |
|--------|-------------|---------------|-------|
| w_d | DALY weight | 0.5 | - |
| w_m | Mortality weight | 50.0 | - |
| w_p | Prevalence weight | 10.0 | - |
| DALYs | Disability-adjusted life years | - | Millions |
| Deaths | Annual mortality | - | Millions |
| Prevalence | Total cases | - | Millions |

### Fallback Formula (DALYs only)

```
Burden_Score = 10 × log₁₀(DALYs_raw + 1)
```

Where DALYs_raw is the raw count (not millions).

**Rationale:** Log-transformation prevents extreme burden values from 
dominating and produces comparable scores across different burden magnitudes.

---

## 2. Research Gap Score

### Three-Tier Scoring System

**Tier 1: Zero Publications**
```
If Publications = 0:
    Gap_Score = 95 (Critical)
```

**Tier 2: Category-Specific Thresholds**

For Infectious diseases (HIV, TB, respiratory infections, etc.):
| Publications | Gap Score |
|--------------|-----------|
| < 10 | 90 |
| < 25 | 80 |
| < 50 | 70 |
| < 100 | 60 |
| ≥ 100 | Use Tier 3 |

For Neglected/Maternal diseases:
| Publications | Gap Score |
|--------------|-----------|
| < 10 | 92 |
| < 25 | 82 |
| < 50 | 72 |
| < 100 | 65 |
| ≥ 100 | Use Tier 3 |

**Tier 3: Burden-Normalized Intensity**

```
Pubs_per_Million_DALYs = Publications / DALYs_millions
```

| Pubs per Million DALYs | Gap Score |
|-----------------------|-----------|
| ≥ 100 | 10 |
| ≥ 50 | 20 |
| ≥ 25 | 30 |
| ≥ 10 | 40 |
| ≥ 5 | 50 |
| ≥ 2 | 60 |
| ≥ 1 | 70 |
| ≥ 0.5 | 80 |
| < 0.5 | 85 |

**Global South Penalty:**
```
If Global_South_Priority AND Publications < 50:
    Gap_Score = min(95, Gap_Score + 10)
```

### Gap Severity Classification

| Category | Gap Score Range |
|----------|----------------|
| Critical | > 70 |
| High | 50-70 |
| Moderate | 30-50 |
| Low | < 30 |

---

## 3. Research Opportunity Score (ROS)

```
ROS_b = Σ Burden_Score(d) for all diseases where Publications(b,d) ≤ 2
```

**Interpretation:** Higher ROS indicates greater unrealized potential for 
equity-aligned research. The threshold of ≤2 publications identifies diseases 
effectively unaddressed by a given biobank.

---

## 4. Equity Alignment Score (EAS)

### Formula

```
EAS = 100 - [(w_g × Gap_Severity) + (w_b × Burden_Miss) + (w_c × Capacity_Penalty)]
```

**Weights:**
| Symbol | Description | Default Value |
|--------|-------------|---------------|
| w_g | Gap severity weight | 0.4 |
| w_b | Burden miss weight | 0.3 |
| w_c | Capacity penalty weight | 0.3 |

### Component Calculations

**Gap_Severity (0-100):**
```
Weighted_Gaps = (4 × N_critical) + (2 × N_high) + (1 × N_moderate)
Max_Gaps = 4 × N_diseases
Gap_Severity = (Weighted_Gaps / Max_Gaps) × 100
```

Where N_critical, N_high, N_moderate are counts of diseases in each gap 
category for this biobank's coverage.

**Burden_Miss (0-100):**
```
Missed_DALYs = Σ DALYs for diseases with Publications ≤ 2
Total_DALYs = Σ DALYs for all diseases
Burden_Miss = (Missed_DALYs / Total_DALYs) × 100
```

**Capacity_Penalty (0-100):**
```
Pubs_per_Disease = Total_Publications / N_diseases
Capacity_Penalty = 100 - min(Pubs_per_Disease, 100)
```

**Interpretation:** This penalizes biobanks with low average publications 
per disease, reflecting limited research capacity to address the full 
burden spectrum. The penalty is capped at 100 (i.e., no credit below 
1 publication per disease on average).

### EAS Categories

| Category | EAS Range |
|----------|-----------|
| Strong | ≥ 80 |
| Moderate | 60-79 |
| Developing | 40-59 |
| Low | < 40 |

---

## 5. HIC:LMIC Equity Ratio

### Publication-Based Ratio

```
Equity_Ratio = Publications_HIC / Publications_LMIC
```

**Interpretation:** Ratio > 1 indicates HIC dominance in research output.

### Burden-Adjusted Ratio (Alternative)

```
Equity_Ratio_adj = (Pubs_HIC / DALYs_HIC) / (Pubs_LMIC / DALYs_LMIC)
```

This accounts for differential disease burden between income groups.

---

## 6. Data Sources

| Data Element | Source | Version |
|--------------|--------|---------|
| DALYs, Deaths, Prevalence | IHME Global Burden of Disease | 2021 |
| Biobank Registry | IHCC Global Cohort Atlas | 2024 |
| Publications | PubMed/MEDLINE | Retrieved 2025 |
| Income Classifications | World Bank | 2024 |
| MeSH Terms | NLM | 2024 |

"""
    return content


# =============================================================================
# TABLE S4: SOFTWARE VERSIONS
# =============================================================================

def generate_software_versions() -> str:
    """
    Generate Supplementary Table S4: Software and Library Versions
    
    Addresses Comment 18: "Provide software versions and key libraries"
    """
    logger.info("Generating Table S4: Software Versions...")
    
    # Get versions
    python_version = platform.python_version()
    
    # Try to get library versions
    versions = {}
    
    libraries = [
        'pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib', 
        'seaborn', 'biopython', 'sentence_transformers'
    ]
    
    for lib in libraries:
        try:
            if lib == 'biopython':
                import Bio
                versions[lib] = Bio.__version__
            elif lib == 'sklearn':
                import sklearn
                versions[lib] = sklearn.__version__
            elif lib == 'sentence_transformers':
                import sentence_transformers
                versions[lib] = sentence_transformers.__version__
            else:
                module = __import__(lib)
                versions[lib] = module.__version__
        except (ImportError, AttributeError):
            versions[lib] = 'Not installed'
    
    content = f"""# Supplementary Table S4: Software and Library Versions

## Computational Environment

| Component | Version |
|-----------|---------|
| Python | {python_version} |
| Operating System | {platform.system()} {platform.release()} |
| Architecture | {platform.machine()} |

## Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | {versions.get('pandas', 'N/A')} | Data manipulation |
| numpy | {versions.get('numpy', 'N/A')} | Numerical computing |
| scipy | {versions.get('scipy', 'N/A')} | Statistical analysis |
| scikit-learn | {versions.get('sklearn', 'N/A')} | Machine learning |
| matplotlib | {versions.get('matplotlib', 'N/A')} | Visualization |
| seaborn | {versions.get('seaborn', 'N/A')} | Statistical visualization |
| biopython | {versions.get('biopython', 'N/A')} | NCBI API access |
| sentence-transformers | {versions.get('sentence_transformers', 'N/A')} | Semantic similarity |

## External APIs

| Service | Purpose |
|---------|---------|
| NCBI Entrez | PubMed publication retrieval |
| IHME GBD Results Tool | Disease burden data |

## Reproducibility

All analysis code is available at: https://github.com/manuelcorpas/17-EHR

Analysis was performed on: {datetime.now().strftime('%Y-%m-%d')}

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Year range | 2000-2025 |
| Random seed | 42 |
| Bootstrap iterations | 1000 |
| Confidence interval | 95% (percentile method) |
"""
    
    return content


# =============================================================================
# TABLE S5: MESH-GBD MAPPING
# =============================================================================

def generate_mesh_gbd_mapping(gbd_registry: Dict) -> pd.DataFrame:
    """
    Generate Supplementary Table S5: MeSH to GBD Mapping
    
    Addresses Comment 11: Documentation of disease mapping
    """
    logger.info("Generating Table S5: MeSH-GBD Mapping...")
    
    rows = []
    
    for disease_id, info in gbd_registry.items():
        rows.append({
            'gbd_cause': disease_id,
            'gbd_level2': info.get('gbd_level2', ''),
            'global_south_priority': info.get('global_south_priority', False),
            'dalys_millions': round(info.get('dalys', 0) / 1_000_000, 2) if info.get('dalys', 0) > 1000 else info.get('dalys', 0),
            'publications': info.get('publications', 0)
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('dalys_millions', ascending=False)
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("HEIM-Biobank: Generate Supplementary Tables")
    print(f"Hardware: {N_CORES} cores available")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    print(f"   Base directory: {BASE_DIR}")
    print(f"   Data directory: {DATA_DIR}")
    
    ihcc_registry = {}
    if INPUT_IHCC_REGISTRY.exists():
        with open(INPUT_IHCC_REGISTRY, 'r') as f:
            ihcc_registry = json.load(f)
        print(f"   IHCC cohorts: {len(ihcc_registry.get('cohorts', []))}")
    else:
        print(f"   ⚠️  IHCC registry not found: {INPUT_IHCC_REGISTRY}")
    
    gbd_registry = {}
    if INPUT_GBD_REGISTRY.exists():
        with open(INPUT_GBD_REGISTRY, 'r') as f:
            gbd_registry = json.load(f)
        print(f"   GBD diseases: {len(gbd_registry)}")
    else:
        print(f"   ⚠️  GBD registry not found: {INPUT_GBD_REGISTRY}")
    
    query_log = {}
    if INPUT_QUERY_LOG.exists():
        with open(INPUT_QUERY_LOG, 'r') as f:
            query_log = json.load(f)
        print(f"   Query log entries: {len(query_log.get('queries', []))}")
    else:
        print(f"   ⚠️  Query log not found: {INPUT_QUERY_LOG}")
    
    # ==========================================================================
    # Generate Tables
    # ==========================================================================
    
    # Table S1: Search Queries
    print("\n" + "-" * 70)
    print("Generating Table S1: Search Queries...")
    
    if ihcc_registry:
        search_df = generate_search_query_table(ihcc_registry, query_log)
        search_file = OUTPUT_DIR / "03-09-01-search_queries.csv"
        search_df.to_csv(search_file, index=False)
        print(f"   Saved: {search_file}")
        print(f"   Cohorts documented: {len(search_df)}")
    else:
        print("   ⚠️  Skipped (no IHCC registry)")
    
    # Table S2: Burden Score Calibration
    print("\n" + "-" * 70)
    print("Generating Table S2: Burden Score Calibration...")
    
    if gbd_registry:
        calibration_df = generate_burden_calibration_table(gbd_registry)
        calibration_file = OUTPUT_DIR / "03-09-02-burden_score_calibration.csv"
        calibration_df.to_csv(calibration_file, index=False)
        print(f"   Saved: {calibration_file}")
        print(f"   Diseases documented: {len(calibration_df)}")
    else:
        print("   ⚠️  Skipped (no GBD registry)")
    
    # Table S3: Formula Specifications
    print("\n" + "-" * 70)
    print("Generating Table S3: Formula Specifications...")
    
    formula_content = generate_formula_specifications()
    formula_file = OUTPUT_DIR / "03-09-03-formula_specifications.md"
    with open(formula_file, 'w') as f:
        f.write(formula_content)
    print(f"   Saved: {formula_file}")
    
    # Table S4: Software Versions
    print("\n" + "-" * 70)
    print("Generating Table S4: Software Versions...")
    
    software_content = generate_software_versions()
    software_file = OUTPUT_DIR / "03-09-04-software_versions.md"
    with open(software_file, 'w') as f:
        f.write(software_content)
    print(f"   Saved: {software_file}")
    
    # Table S5: MeSH-GBD Mapping
    print("\n" + "-" * 70)
    print("Generating Table S5: MeSH-GBD Mapping...")
    
    if gbd_registry:
        mapping_df = generate_mesh_gbd_mapping(gbd_registry)
        mapping_file = OUTPUT_DIR / "03-09-05-mesh_gbd_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        print(f"   Saved: {mapping_file}")
        print(f"   Mappings documented: {len(mapping_df)}")
    else:
        print("   ⚠️  Skipped (no GBD registry)")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY TABLES COMPLETE")
    print("=" * 70)
    
    print(f"\n📁 Output files in: {OUTPUT_DIR}")
    print(f"   - 03-09-01-search_queries.csv")
    print(f"   - 03-09-02-burden_score_calibration.csv")
    print(f"   - 03-09-03-formula_specifications.md")
    print(f"   - 03-09-04-software_versions.md")
    print(f"   - 03-09-05-mesh_gbd_mapping.csv")
    
    print("\n✅ Include these tables in Supplementary Materials")


if __name__ == "__main__":
    main()