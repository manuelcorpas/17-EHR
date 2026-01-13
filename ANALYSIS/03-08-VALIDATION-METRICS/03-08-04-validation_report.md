# HEIM Validation Report

Generated: 2026-01-11 15:23

## Overview

This report addresses Reviewer Comments 7-11 regarding validation and 
within-group analyses.

---

## 1. Within-Income-Group EAS Analysis (Comment 7)

### Rationale
To enable fair comparison of biobanks operating under different resource 
constraints, we computed EAS percentiles within each World Bank income group.

### Summary Statistics by Income Group

| Income | N | Mean Pubs | Mean Diseases | Mean EAS | Std EAS |
|--------|---|-----------|---------------|----------|---------|
| HIC | 61 | 597 | 68 | 38.2 | 23.1 |
| UMIC | 10 | 174 | 43 | 24.1 | 22.9 |
| LMIC | 4 | 91 | 38 | 21.1 | 16.0 |
| LIC | 3 | 47 | 27 | 15.3 | 7.1 |

### Interpretation

Within-income-group percentiles allow identification of high-performing 
biobanks relative to their peers. A biobank with low global EAS but high 
within-income percentile may represent best-in-class performance given 
available resources.

---

## 2. Disease Mapping Validation (Comment 11)

### Methodology

- Sample size: 200 randomly selected publications
- Random seed: 42 (for reproducibility)

### Results

| Metric | Value |
|--------|-------|
| Publications with mapping | 200 (100.0%) |
| Publications without mapping | 0 |
| Mean diseases per publication | 5.03 |
| Median diseases per publication | 5.0 |
| Max diseases per publication | 35 |
| Single-disease publications | 37 |
| Multi-disease publications | 163 |

### Multi-Morbidity Handling

Publications are mapped to ALL applicable GBD causes based on MeSH term 
matching. Each publication-disease pair is counted once (i.e., a publication 
mapped to 3 diseases contributes 1 publication to each disease's count).

### Disease Distribution

| Diseases per Pub | N Publications |
|------------------|----------------|
| 1 | 37 |
| 2 | 6 |
| 3 | 27 |
| 4 | 27 |
| 5 | 29 |
| 6 | 37 |
| 7 | 15 |
| 8 | 8 |
| 9 | 5 |
| 10 | 5 |
| 33 | 1 |
| 34 | 1 |
| 35 | 2 |


---

## 3. Disease Coverage Analysis


| Metric | Value |
|--------|-------|
| GBD causes in registry | 179 |
| Causes with ≥1 publication | 172 |
| Coverage rate | 96.1% |
| Causes with zero publications | 7 |

### Top 10 Most-Researched Diseases

| Rank | Disease | Publications | % of Total |
|------|---------|--------------|------------|
| 1 | Gynecological diseases | 23,966 | 65.81% |
| 2 | _BIOBANK_METHODS | 22,295 | 61.22% |
| 3 | _AGING | 18,662 | 51.25% |
| 4 | Urinary diseases and male infertility | 16,129 | 44.29% |
| 5 | _GENOMICS | 8,657 | 23.77% |
| 6 | Other neoplasms | 6,925 | 19.02% |
| 7 | Endocrine, metabolic, blood, and immune  | 3,898 | 10.7% |
| 8 | Diabetes mellitus | 3,255 | 8.94% |
| 9 | Maternal disorders | 3,235 | 8.88% |
| 10 | Other cardiovascular and circulatory dis | 3,137 | 8.61% |


### Diseases with Zero Publications (Examples)

- Dengue
- Schistosomiasis
- Lymphatic filariasis
- Cysticercosis
- Guinea worm disease
- Drowning
- Animal contact


---

## 4. PubMed Search Coverage (Comment 8)

### Methodology


- Total cohorts searched: 17
- Cohorts with publications: 15
- Total publications retrieved: 19,111
- Preprints excluded: 621
- Mean publications per cohort: 1124.2
- Mean retention rate: 85.9%

### Search Strategy

Each biobank was searched using multiple aliases to maximize recall:
- Primary name (e.g., "UK Biobank")
- Known aliases (e.g., "United Kingdom Biobank", "UK-Biobank")
- Consortium names where applicable

Field restriction: [All Fields] was used to capture publications 
mentioning the biobank in title, abstract, or full text.

### Disambiguation Approach

Ambiguous acronyms (e.g., "MVP" for Million Veteran Program) were 
handled by including the full name in combination with the acronym 
using OR operators, and manual verification of high-volume acronym 
matches during quality control.

### Top 5 Cohorts by Publication Count

| Cohort | Publications |
|--------|-------------|
| UK Biobank | 13,902 |
| Women's Health Initiative | 3,436 |
| Taiwan Biobank | 560 |
| SIMPLER | 280 |
| Tohoku Medical Megabank | 245 |


---

## 5. Limitations and Caveats

1. **PubMed Indexing Bias**: Regional databases (LILACS, African Index Medicus) 
   may contain additional LMIC publications not captured in PubMed.

2. **MeSH Annotation Lag**: Recently published articles may lack complete 
   MeSH annotations, potentially underestimating disease coverage.

3. **Multi-mapping Assumptions**: The fractional contribution approach assumes 
   equal relevance of a publication to all mapped diseases, which may not 
   reflect actual research emphasis.

4. **Within-Income Limitations**: Small sample sizes in LIC and LMIC groups 
   may limit the stability of percentile estimates.

---

## 6. Data Availability

All validation data and scripts are available at:
https://github.com/manuelcorpas/17-EHR

