# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-XX-XX

### 🎉 Initial Release - Paper Publication Version

This release accompanies the publication:
> "EHR-Linked Biobank Expansion Reveals Global Health Inequities"  
> Corpas M, Ojewunmi O, Guio H, Fatumo S (2025)  
> Annual Review of Biomedical Data Science

### Added

- **Data retrieval pipeline** (`00-00-biobank-data-retrieval.py`)
  - Automated PubMed search for 5 major EHR-linked biobanks
  - Date-based chunking to handle >9,999 records
  - Preprint identification and exclusion

- **Publication analysis** (`00-01-biobank-analysis.py`)
  - Year-by-year publication distribution (2000-2024)
  - Top 10 MeSH terms per biobank
  - Top 10 journals per biobank
  - Preprint filtering statistics

- **MeSH term clustering** (`00-02-biobank-mesh-clustering.py`)
  - TF-IDF vectorization of MeSH terms
  - Bootstrap-based optimal K selection
  - K-means clustering with c-DF-IPF scoring
  - PCA and UMAP projections

- **Research gap discovery** (`01-00-research-gap-discovery.py`)
  - Integration with GBD 2021 disease burden data
  - Composite Burden Score calculation
  - Research Gap Score (0-100) for 25 priority diseases
  - Biobank-specific Research Opportunity Scores

- **Gap visualization** (`01-01-data-driven-viz.py`)
  - Burden vs. research effort scatter plots
  - Gap severity heatmaps
  - Equity panel visualizations

### Data

- Curated corpus of 14,142 peer-reviewed publications
- MeSH-to-disease mappings for 25 high-burden conditions
- GBD 2021 DALYs, deaths, and prevalence data

### Figures

- Figure 1: Publication trends by biobank (2000-2024)
- Figure 2: Top MeSH terms across biobanks
- Figure 3: Semantic cluster PCA projections
- Figure 4: Disease-biobank research heatmap
- Figure 5: Research gap summary (25 diseases)
- Supplementary Figures S1-S4
- Supplementary Table S1: Cluster characteristics

---

## [Unreleased]

### Planned
- Biobank Health Equity Monitor (BHEM) dashboard
- Expanded global biobank registry (50+ biobanks)
- Automated weekly data updates
- Interactive web interface

---

## Notes

### Versioning Scheme

- **Major**: Breaking changes to pipeline or methodology
- **Minor**: New analyses or biobanks added
- **Patch**: Bug fixes, documentation updates

### Reproducibility

All releases are tagged and can be reproduced by:
1. Checking out the release tag
2. Installing dependencies from `requirements.txt`
3. Running scripts in numerical order

### Data Availability

Due to size constraints, raw PubMed data is not included in releases. 
Run `00-00-biobank-data-retrieval.py` to regenerate, or use the 
pre-processed CSV files in `DATA/`.
