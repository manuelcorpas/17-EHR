# Data Directory

This directory contains input data, computed metrics, and intermediate outputs for the HEIM (Health Equity Informative Metrics) pipeline. The analysis covers 70 IHCC biobanks, 563,725 clinical trials, and 13.1 million PubMed abstracts across 175 GBD 2021 Level 3 disease categories.

## Primary Datasets

### Discovery Dimension (Biobank Research)

| File | Description | Records |
|------|-------------|---------|
| `bhem_publications_mapped.csv` | All biobank-linked publications mapped to GBD diseases | 38,595 |
| `bhem_publications.csv` | Raw PubMed retrieval (pre-mapping) | 38,595 |
| `bhem_metrics.json` | Disease-level metrics (Burden Score, Gap Score, EAS) | 179 diseases |
| `bhem_biobank_metrics.csv` | Biobank-level EAS and component scores | 70 biobanks |
| `bhem_disease_metrics.csv` | Disease-level metrics in tabular form | 179 diseases |
| `bhem_biobank_profiles.csv` | Biobank publication profiles by disease | 70 biobanks |
| `bhem_themes.json` | Research theme analysis per biobank | 70 biobanks |
| `bhem_theme_overlap.csv` | Cross-biobank theme overlap matrix | -- |
| `bhem_cluster_terms.csv` | MeSH term clusters | -- |
| `bhem_fetch_progress.csv` | PubMed retrieval log | 70 entries |
| `fetch_summary.json` | Retrieval summary statistics | -- |

**Key columns in `bhem_publications_mapped.csv`:**
- `biobank`: IHCC biobank name (70 unique values)
- `pmid`: PubMed identifier
- `title`, `abstract`: Article metadata
- `journal`: Journal name
- `year`: Publication year (2000-2025)
- `mesh_terms`: MeSH descriptors (semicolon-separated)
- `gbd_disease`: Mapped GBD Level 3 disease category
- `gbd_level2`: GBD Level 2 grouping

### Translation Dimension (Clinical Trials)

| File | Description | Size |
|------|-------------|------|
| `heim_ct_studies.csv` | Raw trial records from AACT | 224 MB |
| `heim_ct_studies_mapped.csv` | Trials mapped to GBD diseases | 370 MB |
| `heim_ct_conditions.csv` | Trial condition fields | 55 MB |
| `heim_ct_mesh_conditions.csv` | MeSH-mapped conditions | 288 MB |
| `heim_ct_countries.csv` | Trial facility locations | 21 MB |
| `heim_ct_sponsors.csv` | Trial sponsor data | 52 MB |
| `heim_ct_baseline.csv` | Baseline characteristic data | 370 MB |
| `heim_ct_eligibilities.csv` | Eligibility criteria | 837 MB |
| `heim_ct_metrics.json` | Computed equity metrics | 6 KB |
| `heim_ct_disease_trial_matrix.csv` | Disease-by-trial mapping | 6 KB |
| `heim_ct_disease_equity.csv` | Per-disease equity scores | 8 KB |
| `heim_ct_disease_registry.json` | GBD disease registry for CT mapping | 18 KB |
| `heim_ct_geographic_equity.csv` | HIC vs LMIC trial distribution | <1 KB |
| `heim_ct_enrollment_equity.csv` | Enrollment equity data | <1 KB |
| `heim_ct_demographic_categories.csv` | Demographic breakdown | 95 KB |
| `heim_ct_fetch_manifest.json` | AACT fetch log | 2 KB |
| `heim_ct_schema_profile.json` | AACT schema documentation | 117 KB |

### Knowledge Dimension (Semantic Analysis)

Located in `05-SEMANTIC/`:

| File/Directory | Description |
|----------------|-------------|
| `PUBMED-RAW/` | Raw PubMed abstract XMLs (175 disease directories) |
| `EMBEDDINGS/` | PubMedBERT embeddings in gzip-compressed HDF5 (175 diseases) |
| `CHECKPOINTS/` | Embedding generation checkpoints |
| `AUDIT/` | Data integrity audit reports |
| `heim_sem_metrics.json` | SII, KTP, RCC, temporal drift per disease |
| `heim_sem_disease_registry.json` | Disease registry for semantic pipeline |
| `heim_sem_quality_scores.csv` | Abstract quality scores |
| `heim_integrated_metrics.json` | All three dimensions merged; Unified Neglect Score |
| `heim_integrated_metrics.csv` | Same, in tabular form |
| `gbd_mesh_mapping.json` | GBD-to-MeSH mapping used for abstract retrieval |

### Disease Taxonomy

| File | Description |
|------|-------------|
| `gbd_disease_registry.json` | GBD 2021 Level 3 taxonomy (175 diseases with DALYs, deaths, prevalence) |
| `gbd_cause_mapping.json` | GBD cause hierarchy mapping |
| `disease_registry.json` | Legacy disease registry |

### GBD Source Data

| File | Description | Source |
|------|-------------|--------|
| `IHMEGBD_2021_DATA*.csv` | Global Burden of Disease 2021 estimates | [IHME GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) |
| `GBD-Results-tool-citation.txt` | Citation for GBD data | IHME |

**GBD query parameters:**
- Location: Global
- Age: All ages
- Sex: Both
- Metric: Number (not rate)
- Year: 2021
- Measures: DALYs, deaths, prevalence

---

## Data Provenance

| Source | Access | Study Period |
|--------|--------|--------------|
| IHCC Global Cohort Atlas | [ihccglobal.org](https://ihccglobal.org/) | -- |
| PubMed/MEDLINE | [NCBI Entrez API](https://www.ncbi.nlm.nih.gov/home/develop/api/) | 2000-2025 |
| AACT (ClinicalTrials.gov) | [aact.ctti-clinicaltrials.org](https://aact.ctti-clinicaltrials.org/) | 2000-2025 |
| IHME GBD 2021 | [ghdx.healthdata.org](https://ghdx.healthdata.org/) | 2021 |

---

## Git-tracked vs Gitignored Files

Large intermediate files (>100 MB) are **gitignored** to keep the repository manageable. These can be regenerated by running the pipeline scripts.

**Gitignored (regenerate via pipeline):**
- `heim_ct_studies.csv`, `heim_ct_studies_mapped.csv`, `heim_ct_baseline.csv`, `heim_ct_eligibilities.csv`, `heim_ct_mesh_conditions.csv`
- `05-SEMANTIC/EMBEDDINGS/` (HDF5 files, ~50 GB total)
- `05-SEMANTIC/PUBMED-RAW/` (raw abstracts)

**Git-tracked (small, essential):**
- All `*_metrics.json` files
- `bhem_publications_mapped.csv`
- `gbd_disease_registry.json`
- `heim_ct_metrics.json`, `heim_ct_disease_trial_matrix.csv`
- `05-SEMANTIC/heim_sem_metrics.json`, `05-SEMANTIC/heim_integrated_metrics.json`

---

## Regeneration Instructions

Each data source can be regenerated independently:

```bash
# Discovery dimension (biobank publications from PubMed)
python3.11 PYTHON/03-00-ihcc-fetch-pubmed.py     # Fetches 38,595 publications (~2-3 hours)
python3.11 PYTHON/03-01-bhem-map-diseases.py      # Maps to GBD taxonomy
python3.11 PYTHON/03-03-bhem-compute-metrics.py   # Computes Gap Score, EAS

# Translation dimension (clinical trials from AACT)
# Requires AACT credentials in .env (see .env.example)
python3.11 PYTHON/04-01-heim-ct-fetch.py          # Fetches trial records (~30-45 min)
python3.11 PYTHON/04-02-heim-ct-map-diseases.py   # Maps to GBD taxonomy (~15 min)
python3.11 PYTHON/04-03-heim-ct-compute-metrics.py

# Knowledge dimension (PubMed abstracts + embeddings)
python3.11 PYTHON/05-01-heim-sem-fetch.py         # Fetches 13.1M abstracts (~4-6 hours)
python3.11 PYTHON/05-02-heim-sem-embed.py          # PubMedBERT embeddings (~8-12 hours GPU)
python3.11 PYTHON/05-03-heim-sem-compute-metrics.py

# Integration (all three dimensions)
python3.11 PYTHON/05-05-heim-sem-integrate.py
```

**Note:** PubMed data is dynamic. Running retrieval scripts after the analysis date (January 2026) may yield additional records.

---

## Archive

The `ARCHIVE/` subdirectory contains deprecated data files from earlier analysis iterations (original 5-biobank dataset). These are retained for reproducibility of exploratory analyses but are not used in the current pipeline.

---

## Citation

If using this data, please cite:

1. The preprint (see main [README](../README.md))
2. Original data sources:

```bibtex
@article{gbd2021,
  author  = {{GBD 2021 Diseases and Injuries Collaborators}},
  title   = {Global incidence, prevalence, years lived with disability (YLDs),
             disability-adjusted life-years (DALYs), and healthy life expectancy
             (HALE) for 371 diseases and injuries in 204 countries and territories},
  journal = {The Lancet},
  year    = {2024},
  volume  = {403},
  pages   = {2133--2161}
}
```
