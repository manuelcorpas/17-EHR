# Three Dimensions of Neglect: How Biobanks, Clinical Trials, and Scientific Literature Systematically Exclude the Global South

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Interactive Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen.svg)](https://manuelcorpas.github.io/17-EHR/)

## Overview

This repository contains the complete code, data, and analysis pipeline for the **Health Equity Informative Metrics (HEIM)** framework, a three-dimensional analysis of research equity across biobanks, clinical trials, and scientific literature.

> **Corpas M, Freidin MB, Valdivia-Silva J, Baker S, Fatumo S, Guio H** (2026)
>
> Diseases affecting 1.5 billion people in the Global South are systematically excluded from biomedical research infrastructure. We show that this exclusion operates across three stages of the research enterprise: biobanks, clinical trials, and the scientific literature. Using HEIM, we analysed 70 biobanks, ~0.6 million clinical trials, and 13.1 million PubMed articles spanning 175 diseases. Only 1 of 70 biobanks produces research proportionate to global disease burden. The ten most neglected diseases are exclusively conditions of the Global South, facing disadvantage at every stage.

### Key Findings

| Finding | Detail |
|---------|--------|
| Biobank equity | Only 1 of 70 biobanks achieves high equity alignment (EAS >= 70) |
| Clinical trial concentration | Trial sites concentrate 2.5-fold in high-income countries |
| Semantic isolation | NTDs are 44% more isolated in the knowledge landscape (P < 0.0001, Cohen's d = 1.80) |
| Unified neglect | The 10 most neglected diseases are exclusively Global South conditions |
| Zero publications | Lymphatic filariasis, dengue, and schistosomiasis have 0 biobank publications across 70 cohorts |

---

## Data Sources

| Source | Description | Records | Access |
|--------|-------------|---------|--------|
| IHME GBD 2021 | DALYs, deaths, prevalence | 179 Level 3 disease categories, 204 countries | [IHME GBD Results](https://vizhub.healthdata.org/gbd-results/) |
| IHCC Global Cohort Atlas | Biobank registry | 70 biobanks, 29 countries | [IHCC](https://ihccglobal.org/) |
| PubMed/MEDLINE | Biobank-linked publications | 38,595 articles (2000-2025) | [NCBI Entrez API](https://www.ncbi.nlm.nih.gov/home/develop/api/) |
| AACT (ClinicalTrials.gov) | Clinical trial records | 563,725 trials; 770,178 facility records | [AACT](https://aact.ctti-clinicaltrials.org/) |
| PubMed abstracts | For semantic embeddings | 13,100,113 unique abstracts | NCBI Entrez API |

---

## Repository Structure

```
17-EHR/
├── PYTHON/                              # Analysis pipeline (numbered sequentially)
│   ├── 03-00  to 03-10                  # Discovery dimension (biobanks)
│   ├── 04-00  to 04-04                  # Translation dimension (clinical trials)
│   ├── 05-00  to 05-05                  # Knowledge dimension (semantic analysis)
│   └── 06-heim-publication-figures.py   # Main-text and extended data figures
│
├── DATA/                                # Input data and computed metrics
│   ├── gbd_disease_registry.json        # GBD 2021 disease taxonomy
│   ├── bhem_metrics.json                # Discovery dimension metrics
│   ├── heim_ct_metrics.json             # Translation dimension metrics
│   ├── 05-SEMANTIC/                     # Embedding outputs and semantic metrics
│   └── ARCHIVE/                         # Raw retrieval data
│
├── ANALYSIS/                            # Generated figures and tables
│   ├── 03-06-HEIM-FIGURES/              # Discovery dimension figures
│   ├── 04-04-HEIM-CT-FIGURES/           # Translation dimension figures
│   ├── 05-04-HEIM-SEM-FIGURES/          # Knowledge dimension + publication figures
│   ├── 03-07-SENSITIVITY-ANALYSIS/      # Parameter sensitivity results
│   └── 03-08-VALIDATION-METRICS/        # Validation outputs
│
├── docs/                                # Interactive dashboard (GitHub Pages)
│   ├── index.html
│   ├── js/app.js, js/charts.js
│   ├── css/style.css
│   └── data/*.json                      # Dashboard data files
│
├── SCRIPTS/                             # Shell scripts for pipeline orchestration
└── requirements.txt                     # Python dependencies
```

---

## Analysis Pipeline

The pipeline is organised into three dimensions, each with numbered scripts that should be executed sequentially within their group. All scripts use `python3.11`.

### Discovery Dimension (Biobank Research)

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `03-00-ihcc-fetch-pubmed.py` | Retrieve publications for 70 IHCC biobanks from PubMed | `DATA/bhem_publications_mapped.csv` |
| `03-00b-bhem-build-gbd-map.py` | Build GBD-to-MeSH disease mapping | `DATA/gbd_disease_registry.json` |
| `03-01-bhem-map-diseases.py` | Map publications to GBD taxonomy | Mapped publication records |
| `03-02-bhem-analyze-themes.py` | Analyse research themes per biobank | `DATA/bhem_themes.json` |
| `03-03-bhem-compute-metrics.py` | Compute Burden Score, Gap Score, EAS | `DATA/bhem_metrics.json` |
| `03-04-bhem-build-site.py` | Generate interactive dashboard | `docs/` |
| `03-05-bhem-generate-json.py` | Export dashboard data files | `docs/data/*.json` |
| `03-06-bhem-generate-figures.py` | Generate Discovery dimension figures | `ANALYSIS/03-06-HEIM-FIGURES/` |
| `03-07-sensitivity-analysis.py` | Parameter perturbation (weight +-20%) | `ANALYSIS/03-07-SENSITIVITY-ANALYSIS/` |
| `03-08-validation-metrics.py` | Cross-validation of metrics | `ANALYSIS/03-08-VALIDATION-METRICS/` |
| `03-09-supplementary-tables.py` | Supplementary tables | `ANALYSIS/03-09-SUPPLEMENTARY-TABLES/` |
| `03-10-methodology-enhancements.py` | Additional methodological analyses | `ANALYSIS/03-10-REVISION-GUIDE/` |

### Translation Dimension (Clinical Trials)

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `04-00-heim-ct-setup.py` | Validate environment and AACT connection | Connection test |
| `04-01-heim-ct-fetch.py` | Fetch trial records from AACT database | `DATA/heim_ct_*.csv` |
| `04-02-heim-ct-map-diseases.py` | Map trials to GBD taxonomy (multiprocessing) | `DATA/heim_ct_disease_trial_matrix.csv` |
| `04-03-heim-ct-compute-metrics.py` | Compute research intensity, HIC:LMIC ratios | `DATA/heim_ct_metrics.json` |
| `04-04-heim-ct-generate-figures.py` | Generate Translation dimension figures | `ANALYSIS/04-04-HEIM-CT-FIGURES/` |

### Knowledge Dimension (Semantic Analysis)

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `05-00-heim-sem-setup.py` | Validate environment and dependencies | Setup report |
| `05-01-heim-sem-fetch.py` | Retrieve 13.1M PubMed abstracts | `DATA/05-SEMANTIC/ABSTRACTS/` |
| `05-02-heim-sem-embed.py` | Generate PubMedBERT embeddings (768-dim) | `DATA/05-SEMANTIC/EMBEDDINGS/` |
| `05-03-heim-sem-compute-metrics.py` | Compute SII, KTP, RCC, temporal drift | `DATA/05-SEMANTIC/heim_sem_metrics.json` |
| `05-04-heim-sem-generate-figures.py` | Generate Knowledge dimension figures | `ANALYSIS/05-04-HEIM-SEM-FIGURES/` |
| `05-05-heim-sem-integrate.py` | Merge all three dimensions; Unified Score | `DATA/05-SEMANTIC/heim_integrated_metrics.json` |
| `05-05-heim-sem-audit.py` | Data integrity audit | Audit report |

### Publication Figures

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `06-heim-publication-figures.py` | Main-text and extended data figures | `ANALYSIS/05-04-HEIM-SEM-FIGURES/` |

---

## Metrics

### Burden Score
Composite measure of disease severity:
```
Burden Score = (0.5 x DALYs) + (50 x Deaths) + [10 x log10(Prevalence)]
```

### Gap Score (0-100)
Three-tier system measuring the mismatch between disease burden and research attention:
1. Zero-publication penalty (Gap = 95)
2. Category-specific thresholds (stricter for infectious/NTDs)
3. Burden-normalised research intensity

| Category | Threshold |
|----------|-----------|
| Critical | > 70 |
| High | 50-70 |
| Moderate | 30-50 |
| Low | < 30 |

### Equity Alignment Score (EAS, 0-100)
Biobank-level equity performance:
```
EAS = 100 - (0.4 x Gap_Severity + 0.3 x Burden_Miss + 0.3 x Capacity_Penalty)
```

| Category | Threshold |
|----------|-----------|
| High | >= 70 |
| Moderate | 40-69 |
| Low | < 40 |

### Semantic Isolation Index (SII)
Measures how disconnected a disease's research literature is from the broader biomedical knowledge base, computed as the mean cosine distance between a disease's publication embeddings and the centroids of its 100 nearest-neighbour diseases in PubMedBERT vector space.

### Knowledge Transfer Potential (KTP)
Mean cosine similarity between a disease centroid and the centroids of its top 10% most similar conditions, reflecting potential for cross-disciplinary spillover.

### Research Clustering Coefficient (RCC)
Mean cosine distance from individual abstracts to their disease centroid, quantifying within-disease research dispersion.

### Unified Neglect Score
Integrates all three dimensions using PCA-derived weights (PC1 explains 63.3% of variance across 86 diseases with complete data):
```
Unified Score = (0.50 x Discovery) + (0.29 x Translation) + (0.21 x Knowledge)
```
For diseases lacking clinical trial data, the score uses Discovery and Knowledge only (weights rescaled to 0.71 and 0.29 respectively).

---

## Reproducing the Analysis

### Prerequisites

```bash
git clone https://github.com/manuelcorpas/17-EHR.git
cd 17-EHR
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### AACT Database Access (Translation Dimension)

The clinical trials pipeline (scripts `04-*`) requires access to the AACT PostgreSQL database:

1. Register at https://aact.ctti-clinicaltrials.org/users/sign_up (free; approval typically within 24 hours)
2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
3. Test the connection:
   ```bash
   python3.11 PYTHON/04-00-heim-ct-setup.py
   ```

### Running the Full Pipeline

Approximate runtimes are listed for each script (measured on Apple M3 Max, 36 GB RAM, 100 Mbps connection). Actual times will vary with hardware and network speed.

```bash
# Discovery dimension
python3.11 PYTHON/03-00-ihcc-fetch-pubmed.py       # ~2-3 hours (PubMed API, rate-limited)
python3.11 PYTHON/03-00b-bhem-build-gbd-map.py      # ~1 min
python3.11 PYTHON/03-01-bhem-map-diseases.py         # ~5 min
python3.11 PYTHON/03-02-bhem-analyze-themes.py       # ~3 min
python3.11 PYTHON/03-03-bhem-compute-metrics.py      # ~2 min
python3.11 PYTHON/03-06-bhem-generate-figures.py     # ~2 min
python3.11 PYTHON/03-07-sensitivity-analysis.py      # ~5 min
python3.11 PYTHON/03-08-validation-metrics.py        # ~1 min

# Translation dimension (requires AACT database access; see above)
python3.11 PYTHON/04-00-heim-ct-setup.py             # ~10 sec (connection test)
python3.11 PYTHON/04-01-heim-ct-fetch.py             # ~30-45 min (downloads ~2M trial records)
python3.11 PYTHON/04-02-heim-ct-map-diseases.py      # ~15 min (24-core multiprocessing)
python3.11 PYTHON/04-03-heim-ct-compute-metrics.py   # ~2 min
python3.11 PYTHON/04-04-heim-ct-generate-figures.py  # ~2 min

# Knowledge dimension (GPU strongly recommended; CPU fallback available)
python3.11 PYTHON/05-00-heim-sem-setup.py            # ~10 sec (dependency check)
python3.11 PYTHON/05-01-heim-sem-fetch.py            # ~4-6 hours (13.1M PubMed abstracts)
python3.11 PYTHON/05-02-heim-sem-embed.py            # ~8-12 hours GPU / days on CPU
python3.11 PYTHON/05-03-heim-sem-compute-metrics.py  # ~10 min
python3.11 PYTHON/05-04-heim-sem-generate-figures.py # ~3 min

# Integration and publication figures
python3.11 PYTHON/05-05-heim-sem-integrate.py        # ~1 min
python3.11 PYTHON/06-heim-publication-figures.py      # ~2 min
```

**Notes:**
- PubMed retrieval (`03-00`, `05-01`) requires internet access and is rate-limited to 3 requests/second
- AACT access (`04-01`) requires database credentials (see AACT Database Access section above)
- Embedding generation (`05-02`) processes 13.1M abstracts; GPU (MPS/CUDA) strongly recommended
- Pre-computed metrics and data are included in `DATA/` for analysis scripts that do not require raw retrieval
- Total pipeline runtime: ~16-24 hours with GPU, longer on CPU-only hardware

### Sensitivity Analysis

Weighting parameters in the EAS and Unified Score formulas were perturbed by +-20% across 51 total schemes. Rank-order stability was assessed using Spearman's rho (all rho > 0.975). See `ANALYSIS/03-07-SENSITIVITY-ANALYSIS/` for full results.

### Statistical Methods

- Group comparisons: Welch's t-test with Cohen's d effect sizes
- Correlations: Pearson's r
- Multiple comparisons: Benjamini-Hochberg FDR correction (q < 0.05)
- Dimensionality reduction: UMAP (n_neighbours=15, min_dist=0.1, cosine metric)

---

## Interactive Dashboard

An interactive dashboard for exploring HEIM metrics across all three dimensions is available at:

**https://manuelcorpas.github.io/17-EHR/**

The dashboard source is in `docs/` and is served via GitHub Pages.

---

## Authors

- **Manuel Corpas** - University of Westminster; Alan Turing Institute; GENEQ Global
- **Maxim B. Freidin** - King's College London
- **Julio Valdivia-Silva** - UTEC Lima
- **Simeon Baker** - University of Bath; GENEQ Global
- **Segun Fatumo** - MRC/UVRI and LSHTM Uganda Research Unit; Queen Mary University of London
- **Heinner Guio** - UTEC Lima; Instituto Nacional de Salud, Peru; GENEQ Global

**Corresponding author:** m.corpas@westminster.ac.uk (ORCID: 0000-0002-5765-9627)

---

## How to Cite

If you use this code, data, or the HEIM framework, please cite:

```bibtex
@article{corpas2026heim,
  author  = {Corpas, Manuel and Freidin, Maxim B. and Valdivia-Silva, Julio
             and Baker, Simeon and Fatumo, Segun and Guio, Heinner},
  title   = {Three Dimensions of Neglect: How Biobanks, Clinical Trials,
             and Scientific Literature Systematically Underserve
             Global South Diseases},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.1101/2026.01.04.26343419},
  url     = {https://doi.org/10.1101/2026.01.04.26343419}
}
```

A machine-readable citation is also available in [CITATION.cff](CITATION.cff).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the International Health Cohorts Consortium (IHCC) for biobank registry data, the Clinical Trials Transformation Initiative for AACT database access, the Institute for Health Metrics and Evaluation for GBD 2021 data, and NCBI for PubMed API access.
