# EHR-Linked Biobank Expansion Reveals Global Health Inequities

[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue.svg)](https://doi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the code, data, and analysis pipeline for the paper:

> **EHR-Linked Biobank Expansion Reveals Global Health Inequities**  
> Corpas M, Ojewunmi O, Guio H, Fatumo S (2025)  
> *Annual Review of Biomedical Data Science*

We present a comprehensive, semantics-based benchmark of the research footprint of five globally established EHR-linked biobanks: **UK Biobank**, **Million Veteran Program (MVP)**, **FinnGen**, **All of Us Research Program**, and **Estonian Biobank**.

### Key Findings

- Analysis of **14,142 peer-reviewed publications** (2000-2024)
- Identification of distinct thematic profiles for each biobank using MeSH term clustering
- Burden-adjusted gap scores reveal **critical underrepresentation** of conditions like malaria, tuberculosis, and diarrheal diseases
- Development of **Research Opportunity Scores** to quantify unrealized potential for each biobank

---

## Repository Structure

```
17-EHR/
├── PYTHON/                          # Analysis scripts (run in order)
│   ├── 00-00-biobank-data-retrieval.py    # Step 1: PubMed data retrieval
│   ├── 00-01-biobank-analysis.py          # Step 2: Publication analysis & figures
│   ├── 00-02-biobank-mesh-clustering.py   # Step 3: MeSH term clustering (Fig 3)
│   ├── 01-00-research-gap-discovery.py    # Step 4: Disease burden gap analysis
│   ├── 01-01-data-driven-viz.py           # Step 5: Gap visualization (Fig 4-5)
│   └── ARCHIVE/                           # Deprecated/exploratory scripts
│
├── DATA/                            # Input data
│   ├── biobank_research_data.csv          # Main dataset (14,655 records)
│   ├── biobank_research_data_deduplicated.csv
│   ├── IHMEGBD_2021_DATA*.csv             # GBD 2021 disease burden data
│   └── README.md                          # Data documentation
│
├── ANALYSIS/                        # Generated outputs
│   ├── 00-01-BIOBANK-ANALYSIS/            # Publication trends & MeSH frequencies
│   ├── 00-02-BIOBANK-MESH-CLUSTERING/     # Semantic clustering results
│   ├── 01-00-RESEARCH-GAP-DISCOVERY/      # Gap analysis & heatmaps
│   ├── 01-01-DATA-VIZ/                    # Final gap visualizations
│   └── ARCHIVE/                           # Superseded analysis outputs
│
├── FIGS/                            # Publication-ready figures
│   ├── FIG-1-*.png                        # Overview combined plot
│   ├── FIG-2-*.png                        # MeSH terms by biobank
│   ├── FIG-3-*.png                        # Semantic cluster PCA
│   ├── FIG-4-*.png                        # Disease-biobank heatmap
│   ├── FIG-5-*.png                        # Research gap summary
│   └── SUPPL-*.png/csv                    # Supplementary materials
│
├── PRODUCTION/                      # Print-ready PDFs for journal
│   └── *.pdf
│
├── DOCX/                            # Manuscript versions
│   └── EHR-Linked-biobanks_v6.docx        # Submitted manuscript
│
├── REVISION_1/                      # Revision materials
│   ├── EHR-Linked-biobanks_v7.docx        # Revised manuscript
│   ├── Response-To-Reviewers_v1.docx
│   └── BD9_Fatumo_ReviewerComments.docx
│
└── CONTEXT/                         # Reference literature
```

---

## Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/manuelcorpas/17-EHR.git
cd 17-EHR

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

Execute scripts in numerical order from the repository root:

```bash
# Step 1: Retrieve PubMed data (requires internet, ~30 min)
python PYTHON/00-00-biobank-data-retrieval.py

# Step 2: Generate publication analysis figures
python PYTHON/00-01-biobank-analysis.py

# Step 3: Perform MeSH term clustering
python PYTHON/00-02-biobank-mesh-clustering.py

# Step 4: Run research gap discovery analysis
python PYTHON/01-00-research-gap-discovery.py

# Step 5: Create final gap visualizations
python PYTHON/01-01-data-driven-viz.py
```

> **Note**: Step 1 queries PubMed and may take 20-30 minutes. Pre-computed data is available in `DATA/`.

---

## Reproducibility

### Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| PubMed | 14,142 biobank-linked publications | Via Entrez API |
| GBD 2021 | DALYs, deaths, prevalence for 25 diseases | [IHME GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) |
| WHO GHO | Global health indicators | [WHO Data Portal](https://www.who.int/data/gho) |

### Key Parameters

- **Year range**: 2000-2024 (2025 excluded as incomplete)
- **Preprint exclusion**: medRxiv, bioRxiv, Research Square, etc.
- **Disease set**: 25 priority diseases based on DALYs, mortality, and prevalence
- **Burden Score formula**: `(0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]`

### Validated Outputs

All figures and tables in the manuscript can be reproduced by running the pipeline. Expected outputs:

| Script | Key Outputs | Manuscript Location |
|--------|-------------|---------------------|
| `00-01-biobank-analysis.py` | Publication trends, MeSH frequencies | Fig 1, Fig 2 |
| `00-02-biobank-mesh-clustering.py` | Semantic cluster PCA | Fig 3 |
| `01-00-research-gap-discovery.py` | Disease-biobank heatmap | Fig 4 |
| `01-01-data-driven-viz.py` | Gap summary visualization | Fig 5 |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{corpas2025ehrlinked,
  title={EHR-Linked Biobank Expansion Reveals Global Health Inequities},
  author={Corpas, Manuel and Ojewunmi, Oyesola and Guio, Heinner and Fatumo, Segun},
  journal={Annual Review of Biomedical Data Science},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

---

## Authors

- **Manuel Corpas** - University of Westminster, Alan Turing Institute, Cambridge Precision Medicine
- **Oyesola Ojewunmi** - Queen Mary University of London
- **Heinner Guio** - UTEC Lima, INBIOMEDIC
- **Segun Fatumo** - Queen Mary University of London, MRC/UVRI Uganda

**Corresponding authors**: m.corpas@westminster.ac.uk, s.fatumo@qmul.ac.uk

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- UK Biobank, Million Veteran Program, FinnGen, All of Us Research Program, Estonian Biobank
- Global Burden of Disease Study 2021 collaborators
- NCBI PubMed for data access

---

## Version History

- **v1.0.0** (2025-XX-XX): Initial release accompanying publication
- See [CHANGELOG.md](CHANGELOG.md) for full version history
