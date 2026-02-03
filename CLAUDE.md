# HEIM Project Context

## Project Overview
**Health Equity Informative Metrics (HEIM)**: A framework quantifying research alignment between biobank output and global disease burden.

**Webapp**: https://manuelcorpas.github.io/17-EHR/
**Repository**: https://github.com/manuelcorpas/17-EHR

## Manuscript (v4 - medRxiv submission)
**Title**: Health Equity Informative Metrics (HEIM): Quantifying Research Alignment Across 70 Global Biobanks

**Authors**: Manuel Corpas, Julio Valdivia-Silva, Segun Fatumo, Heinner Guio

**Status**: medRxiv preprint (not yet published in journal)

**Location**: `/DOCX/HEIM/MEDRXIV_v2/HEIM_Equity_Index_v4.docx`

## Key Statistics (must match webapp)

| Metric | Value |
|--------|-------|
| Biobanks | 70 (from 29 countries) |
| Publications | 38,595 (2000-2025) |
| Disease categories | 179 (GBD 2021) |
| HIC biobanks | 52 (36,096 pubs, 93.5%) |
| LMIC biobanks | 18 (2,499 pubs, 6.5%) |
| Alignment ratio | 57.8:1 |

## EAS Categories (Equity Alignment Score)
- **High (≥70)**: 1 biobank (1.4%) - UK Biobank only (EAS 84.6)
- **Moderate (40-69)**: 13 biobanks (18.6%)
- **Low (<40)**: 56 biobanks (80.0%)

**Color coding** (semantic - High=good):
- High = Green (#28a745)
- Moderate = Yellow (#ffc107)
- Low = Red (#dc3545)

## Gap Score Categories (disease-level)
- **Critical (>70)**: 22 diseases (12%)
- **High (50-70)**: 26 diseases (15%)
- **Moderate (30-50)**: 47 diseases (26%)
- **Low (<30)**: 84 diseases (47%)

**Color coding** (semantic - Critical=bad):
- Critical = Red
- High = Orange
- Moderate = Yellow
- Low = Green

## Regional Distribution
| Region | Publications | % | Biobanks |
|--------|-------------|---|----------|
| Europe | 21,482 | 55.7% | 25 |
| Americas | 12,433 | 32.2% | 20 |
| Western Pacific | 3,819 | 9.9% | 13 |
| Eastern Mediterranean | 574 | 1.5% | 5 |
| Africa | 277 | 0.7% | 6 |
| South-East Asia | 10 | <0.1% | 1 |

## Top Biobanks
1. UK Biobank: EAS 84.6, 13,785 publications, 163 disease categories
2. Nurses' Health Study: EAS 55.2
3. Women's Health Initiative: EAS 53.4

**Top LMIC biobanks**:
- China Kadoorie Biobank: EAS 41.6, 583 publications
- ELSA-Brasil: EAS 38.0
- Qatar Biobank: EAS 21.2
- AWI-Gen: EAS 10.8

## Critical Gap Diseases (examples)
| Disease | DALYs (millions) | Publications |
|---------|------------------|--------------|
| Malaria | 55.2 | 17 |
| Tuberculosis | 43.1 | 42 |
| Diarrheal diseases | 46.8 | 46 |
| Lower respiratory infections | 63.4 | 89 |
| Neonatal disorders | 35.2 | 156 |

## HEIM Formulas

**Burden Score**:
```
Burden Score = (0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]
```

**EAS (Equity Alignment Score)**:
```
EAS = 100 − (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)
```

## Webapp Structure
- `/docs/index.html` - Main HTML
- `/docs/js/app.js` - Application logic
- `/docs/js/charts.js` - Chart visualizations
- `/docs/css/style.css` - Styles
- `/docs/data/*.json` - Data files (biobanks, diseases, summary, etc.)

## Important Notes
1. EAS categories use "High/Moderate/Low" (not Strong/Weak/Poor)
2. EAS thresholds: ≥70 (High), 40-69 (Moderate), <40 (Low)
3. Gap Score thresholds: >70 (Critical), 50-70 (High), 30-50 (Moderate), <30 (Low)
4. Citation should say "medRxiv preprint" until published
5. Compare Biobanks dropdown is sorted alphabetically
6. Color semantics differ between EAS (High=green=good) and Gap (High=orange=bad)

## Data Sources
- Publications: PubMed/MEDLINE (2000-2025)
- Disease burden: IHME GBD 2021
- Biobank registry: IHCC Global Cohort Atlas (79 cohorts, 70 with publications)
- Clinical trials: ClinicalTrials.gov via AACT database (2000-2025)

---

# HEIM-CT: Clinical Trials Extension (Added January 2026)

## Overview
Extended HEIM framework to analyze clinical trials from ClinicalTrials.gov, revealing the "Discovery-to-Translation Gap" - how health inequities persist from biobanks to clinical trials.

## Key Statistics (Clinical Trials)

| Metric | Value |
|--------|-------|
| Total trials analyzed | 2,189,930 |
| Date range | 2000-2025 |
| Disease categories | 89 (GBD 2021) |
| Trial site records | 770,178 |
| Countries | 194 |

## Global South Priority Analysis

| Metric | GS Priority | Non-GS | Gap |
|--------|-------------|--------|-----|
| Diseases | 30 | 59 | - |
| Trials | 444,502 (20.3%) | 1,745,428 (79.7%) | 1:3.9 |
| DALYs | 914.8M (38.0%) | 1,495.2M (62.0%) | 1:1.6 |
| **Research Intensity** | 486 trials/M DALY | 1,167 trials/M DALY | **2.4x gap** |

## Key Findings

1. **2.4x Intensity Gap**: GS diseases receive 2.4x less research per DALY
2. **Cancer Dominance**: 61.5% of trials for 8.4% of burden (7.3x overrepresentation)
3. **Geographic**: HIC:LMIC ratio = 2.5:1 (trial sites)
4. **Temporal Decline**: GS share fell from 47.7% (2000-05) to 35.5% (2020-25) = -12.3pp
5. **Severely Neglected**: 9 GS diseases with <500 trials in 26 years

## Clinical Trials Pipeline Scripts

```
PYTHON/
├── 04-00-heim-ct-setup.py          # Environment validation, AACT connection
├── 04-01-heim-ct-fetch.py          # Fetch from ClinicalTrials.gov
├── 04-02-heim-ct-map-diseases.py   # Map to GBD taxonomy (multiprocessing)
├── 04-03-heim-ct-compute-metrics.py # Calculate equity metrics
├── 04-04-heim-ct-generate-figures.py # Generate 8 publication figures
```

## AACT Database Credentials
Stored in `.env` (gitignored):
```
AACT_USER=corpas
AACT_PASSWORD=CTTImobfjmc2
```

## Generated Figures (Publication Quality)

Location: `ANALYSIS/04-04-HEIM-CT-FIGURES/`

| Figure | File | Description |
|--------|------|-------------|
| CT1 | 04-04-01_heim_ct_framework | Conceptual framework |
| CT2 | 04-04-02_research_intensity_disparity | Highest vs lowest intensity |
| CT3 | 04-04-03_global_south_priority_diseases | 30 GS diseases trial coverage |
| CT4 | 04-04-04_geographic_concentration | HIC vs LMIC distribution |
| CT5 | 04-04-05_temporal_trends | Trial growth + GS share decline |
| CT6 | 04-04-06_double_jeopardy | Burden vs trials scatter |
| CT7 | 04-04-07_disease_category_coverage | By GBD Level 2 category |
| CT8 | 04-04-08_top_burden_diseases | Top 15 high-burden diseases |

Format: PDF + PNG, 300 DPI, Times New Roman

## Webapp Updates (January 2026)

Added to https://manuelcorpas.github.io/17-EHR/:
- **Clinical Trials tab**: Full CT analysis
- **Pipeline Gap tab**: Biobank vs CT comparison
- **Trends tab**: Temporal analysis
- **Equity tab**: HIC:LMIC distribution

New data file: `docs/data/clinical_trials.json`

## Paper Extension Documents

Location: `DOCX/HEIM/MEDRXIV_v2/`
- `HEIM-CT-PAPER-BRIEFING.md` - All statistics and narrative for paper
- `PROMPT-LANCET-PAPER-EXTENSION.md` - What's done vs what remains

## Lowest Intensity Diseases (>5M DALYs)

| Disease | Trials/M DALY | Trials | DALYs | GS |
|---------|---------------|--------|-------|-----|
| Malaria | 25.4 | 1,402 | 55.2M | Yes |
| Tuberculosis | 32.1 | 1,507 | 47.0M | Yes |
| Meningitis | 42.0 | 610 | 14.5M | Yes |
| COVID-19 | 43.6 | 9,239 | 212.0M | Yes |
| Neonatal disorders | 59.7 | 11,118 | 186.4M | Yes |

## Severely Neglected Diseases (<500 trials in 26 years)

- Typhoid/paratyphoid: 483 trials
- Yellow fever: 459 trials
- Leishmaniasis: 358 trials
- Dengue: 273 trials
- Rabies: 107 trials
- Schistosomiasis: 94 trials
- Trachoma: 91 trials
- Lymphatic filariasis: 90 trials
- Onchocerciasis: 34 trials (~1/year globally)

## Quotable Statistics

> "For every clinical trial on malaria, there are 72 trials on breast cancer—despite malaria causing 2.7x more disability."

> "Global South Priority diseases' share of clinical trials has declined from 48% to 35% since 2000, despite a 15-fold increase in trial volume."

> "Onchocerciasis has received 34 clinical trials in 26 years—approximately one trial per year for a disease affecting 21 million people."

## Next Steps (Paper Revision)

1. Add ~300 words Methods section (AACT source, mapping methodology)
2. Add ~800-1000 words Results section (7 key findings)
3. Add ~500 words Discussion (pipeline gap, policy implications)
4. Select 3-4 figures for main text (CT2, CT3, CT5, CT6 recommended)
5. Format for target journal submission

## Important Notes

1. Citation format: `Corpas et al. (2026) In preparation` (no journal name publicly)
2. Use python3.11 for running scripts
3. Large data files (>100MB) excluded from git - regenerate via pipeline if needed
4. Multiprocessing optimization in 04-02 script uses 24 workers

---

# Nature Medicine Manuscript (Added January 2026)

## Overview
Full three-dimensional HEIM manuscript targeting Nature Medicine, integrating Discovery, Translation, and Knowledge dimensions.

**Title**: Three Dimensions of Neglect: How Biobanks, Clinical Trials, and Scientific Literature Systematically Underserve Global South Diseases

**Authors**: Manuel Corpas, Maxim Freydin, Julio Valdivia-Silva, Simeon Baker, Segun Fatumo, Heinner Guio

**Location**: `/DOCX/NATURE-MEDICINE/00-HEIM_Nature_Medicine_Manuscript_v1.docx`

**Status**: In preparation (v1)

## Document Structure (paragraph indices)

| Section | Para Index | Style |
|---------|-----------|-------|
| Title/Authors | 1–17 | Normal |
| Abstract | 21 | Heading 1 |
| Introduction | 24 | Heading 1 |
| Results | 35 | Heading 1 |
| Discussion | 96 | Heading 1 |
| Extended Data Figures | 118 | Heading 1 |
| Methods | 135 | Heading 1 |
| Acknowledgements | 145 | Heading 1 |
| Author Contributions | 147 | Heading 1 |
| Competing Interests | 149 | Heading 1 |
| Additional Information | 151 | Heading 1 |

## Methods Section (expanded 1 Feb 2026)

Expanded from 138 words to ~1,166 words across 9 subsections (paras 136–144):

| # | Subsection | Words |
|---|-----------|-------|
| 1 | Study Design and Overview | 120 |
| 2 | Data Sources | 159 |
| 3 | Disease Mapping | 95 |
| 4 | Discovery Dimension Metrics | 211 |
| 5 | Translation Dimension Metrics | 129 |
| 6 | Knowledge Dimension Metrics | 173 |
| 7 | Unified Neglect Score | 97 |
| 8 | Statistical Analysis | 106 |
| 9 | Code and Data Availability | 76 |

Format: Bold subsection title followed by body text, Times New Roman 10pt.

**Script**: `PYTHON/06-01-nature-medicine-expand-methods.py` — idempotent; can be re-run to replace Methods section content.

## Key Metrics Defined in Methods

- **Burden Score**: (0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]
- **Gap Score**: Three-tier (zero-publication penalty, category thresholds, burden-normalised intensity)
- **EAS**: 100 − (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)
- **SII (Semantic Isolation Index)**: Mean cosine distance to k=100 nearest-neighbour disease centroids
- **KTP (Knowledge Transfer Potential)**: Mean cosine similarity to top 10% most similar diseases
- **RCC (Research Clustering Coefficient)**: Mean cosine distance from abstracts to their disease centroid
- **Unified Neglect Score**: (0.33 × Discovery) + (0.33 × Translation) + (0.34 × Knowledge)

## Semantic Isolation Index (SII) — Plain Language

The SII measures how disconnected a disease's research is from the rest of medical science by comparing the language used in its publications to the language used in publications about other diseases — diseases with higher scores have research that "speaks its own language," cut off from mainstream biomedical knowledge. It is computed from PubMedBERT embeddings (768-dimensional vectors) and added to Corpas Core embeddings.
