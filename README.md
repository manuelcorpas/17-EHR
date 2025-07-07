# EHR-Linked Biobanks Research Analysis Framework

This repository provides a comprehensive analytical framework for systematically examining the global research landscape of Electronic Health Record (EHR)-linked biobanks. The framework combines bibliometric analysis, semantic clustering, and research gap discovery to understand research themes, temporal trends, and alignment with global health priorities.

## 🎯 Framework Overview

This multi-component framework supports the analysis presented in **"The Research Footprint of Global EHR-Linked Biobanks"**, providing both the analytical pipeline and research gap discovery tools for understanding biobank research landscapes and their alignment with global disease burden.

### Target Biobanks
* **UK Biobank** (United Kingdom)
* **All of Us Research Program** (United States)
* **FinnGen** (Finland)
* **Estonian Biobank** (Estonia)
* **Million Veteran Program (MVP)** (United States)

### Framework Components

**Component 1: Biobank Literature Analysis**
- Systematic PubMed data retrieval for biobank-linked publications
- MeSH-based semantic clustering and thematic analysis
- Publication trends and biobank-specific research profiling
- Cross-biobank comparative analysis

**Component 2: Research Gap Discovery**
- Global disease burden analysis using GBD 2021 data (25 diseases)
- Research effort quantification via publication mapping
- Burden-adjusted gap scoring and opportunity identification
- Global health equity assessment

**Component 3: Data-Driven Visualization**
- Publication-quality figures for academic publication
- Interactive research gap visualizations
- Comprehensive equity and priority analysis dashboards

## 🔍 Methodology

### Biobank Literature Retrieval

**PubMed Query Strategy**: Systematic search of peer-reviewed literature (2000-2024)

**Search Methodology**:
- **Biobank-Specific Searches**: Exact names and validated aliases for each target biobank
- **Comprehensive Coverage**: No thematic filtering - captures all research domains
- **Quality Assurance**: Systematic preprint exclusion and data validation

**Search Terms by Biobank**:
- **UK Biobank**: "UK Biobank", "United Kingdom Biobank", "U.K. Biobank"
- **Million Veteran Program**: "Million Veteran Program", "MVP biobank", "Veterans Affairs Million Veteran Program"
- **FinnGen**: "FinnGen", "FinnGen biobank", "FinnGen study", "FinnGen consortium"
- **All of Us**: "All of Us Research Program", "All of Us cohort", "AoU Research Program"
- **Estonian Biobank**: "Estonian Biobank", "Estonian Genome Center", "Tartu Biobank"

**Data Quality Standards**:
- ✅ **Published Papers Only**: Systematic preprint exclusion (medRxiv, bioRxiv, Research Square, etc.)
- ✅ **Complete Temporal Coverage**: 2000-2024 (2025 excluded as incomplete)
- ✅ **API Compliance**: NCBI rate limiting and batch processing
- ✅ **Deduplication**: PMID-based unique paper identification

**Final Dataset**: 14,142 peer-reviewed publications with MeSH annotations

### Research Gap Analysis Methodology

**Global Disease Burden Integration**:
- **Data Source**: Global Burden of Disease Study 2021 (IHME)
- **Coverage**: 25 high-priority diseases across major health domains
- **Metrics**: DALYs, mortality, and prevalence data
- **Scope**: Global estimates (both sexes, all ages, 2021)

**25-Disease Analytical Framework**:
```
Cardiovascular: Ischemic Heart Disease, Stroke
Mental Health: Depression, Anxiety Disorders, Bipolar Disorder
Infectious: Tuberculosis, HIV/AIDS, Malaria, Diarrheal Diseases
Neoplasms: Lung Cancer, Breast Cancer, Colorectal Cancer
Metabolic: Diabetes Mellitus Type 2
Neurological: Alzheimer Disease
Respiratory: COPD, Asthma
Musculoskeletal: Low Back Pain, Rheumatoid Arthritis
Kidney: Chronic Kidney Disease
Digestive: Cirrhosis
Maternal/Child: Preterm Birth Complications
Sensory: Cataracts
Injuries: Road Traffic Accidents
Neglected: Neglected Tropical Diseases
Endocrine: Thyroid Disorders
```

**Gap Scoring Algorithm**:
- **Composite Burden Score**: `(0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]`
- **Research Intensity**: Publications per million DALYs
- **Gap Classification**: Critical (>70), High (50-70), Moderate (30-50), Low (<30)
- **Opportunity Scoring**: Biobank-specific unrealized research potential

### Semantic Clustering Pipeline

**MeSH-Based Analysis**:
- **TF-IDF Vectorization**: MeSH terms converted to semantic vectors
- **Per-Biobank Clustering**: Independent analysis preserving biobank research identities
- **Bootstrap K-Selection**: Optimal cluster number via silhouette scoring (50 iterations)
- **c-DF-IPF Scoring**: Cluster characterization using term importance weighting

**Dimensionality Reduction**:
- **PCA Projections**: Variance-preserving 2D cluster visualization
- **UMAP Projections**: Non-linear manifold learning for semantic similarity

## 📊 Analytical Outputs

### Biobank Research Analysis
- **Publication Trends**: Temporal patterns and growth trajectories (2000-2024)
- **Thematic Profiling**: MeSH frequency analysis and research specializations
- **Semantic Clustering**: Research theme identification within each biobank
- **Cross-Biobank Comparison**: Convergent and divergent research patterns

### Research Gap Discovery
- **Burden vs Effort Analysis**: Disease burden plotted against research investment
- **Critical Gap Identification**: High-burden, low-research disease areas
- **Biobank Opportunity Scores**: Unrealized research potential quantification
- **Global Health Equity Assessment**: Resource allocation disparities

### Key Findings
- **Research Imbalance**: Profound misalignment between disease burden and research attention
- **Critical Gaps**: Diseases affecting Global South populations severely underrepresented
- **Biobank Specialization**: Distinct research signatures reflecting institutional mandates
- **Methodological Contribution**: Scalable framework for research priority assessment

## 🗂️ Repository Structure

```
├── PYTHON/
│   ├── 00-00-biobank-data-retrieval.py      # PubMed systematic retrieval
│   ├── 00-01-biobank-analysis.py            # Publication analysis & visualization
│   ├── 00-02-biobank-mesh-clustering.py     # MeSH semantic clustering
│   ├── 01-00-research-gap-discovery.py      # 25-disease gap analysis
│   └── 01-01-data-driven-viz.py             # Advanced visualization suite
├── DATA/
│   ├── biobank_research_data.csv            # Raw publication dataset
│   ├── biobank_research_data_deduplicated.csv
│   └── IHMEGBD_2021_DATA*.csv               # GBD 2021 disease burden data
├── ANALYSIS/
│   ├── 00-01-BIOBANK-ANALYSIS/             # Publication analysis outputs
│   ├── 00-02-BIOBANK-MESH-CLUSTERING/      # Semantic clustering results
│   ├── 01-00-RESEARCH-GAP-DISCOVERY/       # Gap analysis results
│   └── 01-01-DATA-VIZ/                     # Advanced visualizations
└── README.md                                # This framework documentation
```

## 🚀 Usage Pipeline

### 1. Data Retrieval
```bash
# Retrieve all biobank-linked publications from PubMed (2000-2024)
python PYTHON/00-00-biobank-data-retrieval.py
```

### 2. Biobank Literature Analysis
```bash
# Generate publication trends, MeSH analysis, and quality-assured statistics
python PYTHON/00-01-biobank-analysis.py
```

### 3. Semantic Clustering
```bash
# Perform MeSH-based clustering within each biobank
python PYTHON/00-02-biobank-mesh-clustering.py
```

### 4. Research Gap Discovery
```bash
# Analyze disease burden vs research effort (25 diseases)
python PYTHON/01-00-research-gap-discovery.py
```

### 5. Advanced Visualization
```bash
# Generate publication-quality research gap visualizations
python PYTHON/01-01-data-driven-viz.py
```

## 📈 Key Visualizations

### Biobank Analysis Suite
- **Publication Growth Trajectories**: Annual publication trends by biobank
- **Research Theme Profiles**: Top MeSH terms and specialization patterns
- **Semantic Cluster Maps**: PCA/UMAP projections of research themes
- **Cross-Biobank Comparisons**: Integrated research landscape analysis

### Research Gap Analysis Suite
- **Disease Burden vs Research Matrix**: 25-disease priority mapping
- **Critical Gap Identification**: High-burden, low-research visualization
- **Biobank Opportunity Heatmaps**: Unrealized research potential by institution
- **Global Health Equity Dashboard**: Resource allocation disparity analysis

### Publication-Quality Standards
- **Resolution**: 300 DPI for academic publication
- **Formats**: PNG and PDF outputs
- **Typography**: Professional academic formatting
- **Color Palettes**: Accessibility-compliant scientific visualization

## 🔬 Technical Requirements

### Core Dependencies
```bash
# Data retrieval and analysis
pip install pandas numpy matplotlib seaborn scipy biopython

# Machine learning and clustering
pip install scikit-learn umap-learn

# Statistical analysis
pip install statsmodels
```

### System Requirements
- **Python**: 3.8+ recommended
- **Memory**: 4GB+ for large dataset processing
- **Storage**: 1GB for full dataset and outputs
- **Network**: Stable connection for PubMed API access

## 📊 Output Highlights

### Research Gap Discovery Results
- **Critical Gaps Identified**: Malaria, tuberculosis, diarrheal diseases, neglected tropical diseases
- **Research Opportunity Scores**: Quantified potential by biobank
- **Global Health Equity Metrics**: 175M+ DALYs in underrepresented conditions
- **Burden-Adjusted Recommendations**: Data-driven priority identification

### Biobank Research Signatures
- **UK Biobank**: Broadest thematic diversity, environmental epidemiology leadership
- **FinnGen**: Mendelian randomization and registry-based phenotyping focus
- **All of Us**: Diversity, equity, and population health emphasis
- **Million Veteran Program**: Veteran-specific health conditions specialization
- **Estonian Biobank**: Regulatory genomics and cardiometabolic traits

### Methodological Innovations
- **Semantic Clustering Framework**: MeSH-based biobank research profiling
- **Burden-Adjusted Gap Scoring**: Objective research priority quantification
- **Cross-Biobank Comparative Analysis**: Institutional research signature characterization
- **Real-Time Data Integration**: GBD 2021 authoritative burden data utilization

## 📚 Scientific Impact

### Research Applications
- **Funding Priority Setting**: Evidence-based resource allocation guidance
- **Biobank Strategic Planning**: Research portfolio diversification recommendations
- **Global Health Policy**: Equity-informed research investment strategies
- **Academic Research**: Comprehensive biobank landscape characterization

### Methodological Contributions
- **Scalable Gap Analysis**: Reproducible framework for research priority assessment
- **Semantic Research Profiling**: MeSH-based institutional characterization methodology
- **Burden-Research Alignment**: Quantitative equity assessment tools
- **Publication-Quality Automation**: Standardized academic visualization pipeline

## ✍️ Citation and Acknowledgments

This framework supports the research presented in **"The Research Footprint of Global EHR-Linked Biobanks"** and provides reproducible tools for biobank research landscape analysis and global health research gap discovery.

**Data Sources**:
- PubMed/MEDLINE (National Library of Medicine)
- Global Burden of Disease Study 2021 (IHME)
- Medical Subject Headings (MeSH) Thesaurus

**Methodological Framework**: Transparent, reproducible, and academically rigorous approach to biobank research analysis with systematic quality assurance and global health equity assessment.

---

**Principal Investigator**: Dr. Manuel Corpas  
**Institution**: University of Westminster, London  
**Contact**: [m.corpas@westminster.ac.uk](mailto:m.corpas@westminster.ac.uk)  
**Repository**: [https://github.com/manuelcorpas/17-EHR](https://github.com/manuelcorpas/17-EHR)

**Quality Standards**: Published papers only (2000-2024) | Systematic preprint exclusion | Authoritative disease burden data | Reproducible analytical pipeline