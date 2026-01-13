# Prompt: Extending HEIM Paper with Clinical Trials Analysis for Lancet Global Health

## Context

You are helping to extend an existing manuscript on health equity in biomedical research. The original paper introduces the **Health Equity Informative Metrics (HEIM)** framework, which quantifies disparities in biobank research output relative to global disease burden. The paper is being prepared for **Lancet Global Health**.

## What Has Been Completed

### 1. Data Pipeline (Complete)

A complete clinical trials analysis pipeline has been built and executed:

```
PYTHON/
├── 04-00-heim-ct-setup.py      # Environment validation, AACT connection test
├── 04-01-heim-ct-fetch.py      # Fetches data from ClinicalTrials.gov (AACT PostgreSQL)
├── 04-02-heim-ct-map-diseases.py   # Maps trials to GBD 2021 disease taxonomy
├── 04-03-heim-ct-compute-metrics.py # Calculates equity metrics
├── 04-04-heim-ct-generate-figures.py # Generates 8 publication-quality figures
```

**Data Retrieved:**
- 563,725 clinical trials from ClinicalTrials.gov (2000-2025)
- 770,178 trial site location records
- Mapped to 89 GBD 2021 disease categories
- 2,189,930 disease-trial mappings (trials can map to multiple diseases)

### 2. Analysis Results (Complete)

**Key Findings:**
- **2.19 million clinical trials** analyzed (2000-2025)
- **Global South Priority diseases** (30 conditions) receive only **20.3% of trials** despite representing **38% of disease burden**
- **Research intensity gap: 2.4x** (486 vs 1,167 trials per million DALYs)
- **Cancer dominance:** 61.5% of all trials for 8.4% of burden
- **Geographic concentration:** HIC:LMIC ratio of 2.5:1 for trial sites
- **Temporal decline:** GS priority share fell from 47.7% (2000-05) to 35.5% (2020-25), a **-12.3 percentage point decline**
- **9 severely neglected diseases** with <500 trials in 26 years (including dengue, schistosomiasis, lymphatic filariasis)

### 3. Figures Generated (Complete)

Eight publication-quality figures in PDF and PNG (300 DPI, Times New Roman):

| Figure | File | Description |
|--------|------|-------------|
| CT1 | `04-04-01_heim_ct_framework.pdf` | Conceptual framework diagram |
| CT2 | `04-04-02_research_intensity_disparity.pdf` | Highest vs lowest intensity diseases |
| CT3 | `04-04-03_global_south_priority_diseases.pdf` | 30 GS diseases: trials vs burden |
| CT4 | `04-04-04_geographic_concentration.pdf` | HIC vs LMIC distribution |
| CT5 | `04-04-05_temporal_trends.pdf` | Trial growth and GS share decline |
| CT6 | `04-04-06_double_jeopardy.pdf` | Burden vs trials scatter plot |
| CT7 | `04-04-07_disease_category_coverage.pdf` | By GBD Level 2 category |
| CT8 | `04-04-08_top_burden_diseases.pdf` | Top 15 high-burden diseases |

**Location:** `ANALYSIS/04-04-HEIM-CT-FIGURES/`

### 4. Briefing Document (Complete)

A comprehensive briefing document with all statistics, interpretations, and suggested narrative:

**Location:** `DOCS/HEIM-CT-PAPER-BRIEFING.md`

Contains:
- Executive summary
- All key statistics with tables
- Figure descriptions and their narrative purpose
- Comparison with biobank findings
- Discussion points
- Suggested paper structure
- Quotable one-liners

### 5. Data Files (Complete)

```
DATA/
├── heim_ct_studies.csv           # Raw trial data
├── heim_ct_studies_mapped.csv    # Trials with disease mappings
├── heim_ct_disease_trial_matrix.csv  # Disease × trial counts with DALYs
├── heim_ct_countries.csv         # Geographic distribution
├── heim_ct_conditions.csv        # Condition text mappings
```

---

## What Needs To Be Done

### 1. Manuscript Writing

**The existing paper needs to be extended with a new section on clinical trials.** The current manuscript (`DOCX/HEIM/MEDRXIV_v2/HEIM_Equity_Index_v4.docx`) focuses on biobank research. It needs:

#### A. Abstract Revision
Add clinical trials findings to abstract. Suggested addition:
> "Extending the HEIM framework to clinical trials reveals a Research-to-Translation Gap: while 2.19 million trials were conducted between 2000-2025, Global South Priority diseases received 2.4x less research intensity than other conditions, and their share of trials has declined by 12.3 percentage points."

#### B. New Methods Section
Add ~300 words describing:
- Data source: ClinicalTrials.gov AACT database
- Date range: 2000-2025
- Disease mapping methodology to GBD 2021
- Research intensity calculation (trials per million DALYs)
- Geographic classification (World Bank income groups)

#### C. New Results Section
Add ~800-1000 words covering:
1. Overall clinical trial landscape (volume, temporal trends)
2. Research intensity disparity (2.4x gap finding)
3. Cancer dominance problem (61.5% for 8.4% burden)
4. Geographic concentration (2.5:1 HIC:LMIC)
5. Temporal trends (declining GS priority share)
6. Severely neglected diseases (<500 trials)
7. Double jeopardy analysis

#### D. Extended Discussion
Add ~500 words discussing:
- Comparison between biobank and clinical trial patterns
- The "pipeline leakage" hypothesis (discoveries don't translate equitably)
- Why the gap is worsening despite growth
- Policy recommendations
- Limitations of clinical trials analysis

#### E. Figure Integration
Select 3-4 figures from the 8 generated to include in main text. Suggested:
- **Figure CT2** (Research Intensity Disparity) - central finding
- **Figure CT3** (Global South Priority) - comprehensive view
- **Figure CT5** (Temporal Trends) - shows worsening pattern
- **Figure CT6** (Double Jeopardy) - powerful visualization

Remaining figures can go to supplementary materials.

### 2. Figure Refinement (If Needed)

The figures are publication-ready but may need minor adjustments based on:
- Lancet Global Health specific formatting requirements
- Editor/reviewer feedback
- Final paper narrative decisions

### 3. Supplementary Materials

Create supplementary appendix containing:
- Full methodology for disease mapping
- Complete disease-by-disease table (89 diseases)
- Additional figures not in main text
- Data availability statement
- Code availability statement

### 4. Data and Code Sharing

Prepare for submission:
- Deposit analysis code to GitHub/Zenodo
- Create DOI for reproducibility
- Prepare data dictionary
- Document AACT database access

### 5. Author Contributions

Update author contributions to reflect clinical trials analysis work.

### 6. Lancet Global Health Formatting

Ensure compliance with journal requirements:
- Word limits (typically 3,500 words for Articles)
- Reference style
- Figure specifications
- Structured abstract format
- STROBE/RECORD checklist if applicable

---

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Original manuscript | `DOCX/HEIM/MEDRXIV_v2/HEIM_Equity_Index_v4.docx` |
| Briefing document | `DOCS/HEIM-CT-PAPER-BRIEFING.md` |
| Figures | `ANALYSIS/04-04-HEIM-CT-FIGURES/` |
| Core data | `DATA/heim_ct_disease_trial_matrix.csv` |
| Analysis scripts | `PYTHON/04-*.py` |
| Biobank figures | `ANALYSIS/03-06-HEIM-FIGURES/` |

---

## Central Narrative to Maintain

The paper should tell this story:

1. **HEIM Framework** identifies health equity gaps in biomedical research
2. **Biobank analysis** reveals a 57.8:1 HIC:LMIC publication ratio (discovery stage)
3. **NEW: Clinical trials analysis** reveals these gaps persist at translation stage
4. **Key insight:** The system isn't self-correcting—despite 15x growth in trials, equity is declining
5. **Implication:** Active policy intervention required to address structural inequities

---

## Suggested Title Update

**Original:** "Health Equity Informative Metrics (HEIM): A Framework for Quantifying Research Equity in Biobank Studies"

**Suggested revision:** "Health Equity Informative Metrics (HEIM): Quantifying the Discovery-to-Translation Gap in Global Health Research"

---

## Contact/Technical Notes

- AACT database credentials stored in `.env` (gitignored)
- Python 3.11 environment required
- Analysis fully reproducible via scripts in `PYTHON/04-*.py`
- Total runtime: ~15 minutes on M3 Ultra Mac

---

*This prompt was generated on 2026-01-13 to document the HEIM clinical trials extension project status.*
