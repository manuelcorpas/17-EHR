# HEIM-CT: Clinical Trials Equity Extension

## Overview

HEIM-CT extends the Health Equity Informative Metrics (HEIM) framework to integrate clinical trial enrollment analysis with biobank research output equity metrics. This creates a unified view of health research equity across the entire biomedical pipeline.

## Integrated Framework Concept

```
Discovery → Translation → Implementation
(Biobanks)   (Clinical Trials)   (Clinical Practice)
    │              │                    │
    ▼              ▼                    ▼
What diseases   Who participates     Who benefits
are studied?    in the research?     from findings?
```

## Scripts

### 04-00-heim-ct-setup.py
Tests AACT database connection and profiles schema for demographic data.

**Requirements:**
```bash
export AACT_USER='your_username'
export AACT_PASSWORD='your_password'
pip install psycopg2-binary --break-system-packages
```

**Register at:** https://aact.ctti-clinicaltrials.org/users/sign_up

**Outputs:**
- `DATA/heim_ct_schema_profile.json`
- `DATA/heim_ct_demographic_categories.csv`
- `ANALYSIS/HEIM-CT/setup_diagnostic_report.txt`

### 04-01-heim-ct-fetch.py
Fetches clinical trial data from the AACT PostgreSQL database.

**Usage:**
```bash
python 04-01-heim-ct-fetch.py [--start-year 2010] [--end-year 2024] [--interventional-only]
```

**Outputs:**
- `DATA/heim_ct_studies.csv`
- `DATA/heim_ct_baseline.csv`
- `DATA/heim_ct_conditions.csv`
- `DATA/heim_ct_mesh_conditions.csv`
- `DATA/heim_ct_countries.csv`
- `DATA/heim_ct_sponsors.csv`
- `DATA/heim_ct_eligibilities.csv`
- `DATA/heim_ct_fetch_manifest.json`

### 04-02-heim-ct-map-diseases.py
Maps clinical trial conditions to GBD 2021 disease categories (same taxonomy as HEIM-Biobank).

**Outputs:**
- `DATA/heim_ct_studies_mapped.csv`
- `DATA/heim_ct_disease_registry.json`
- `DATA/heim_ct_disease_trial_matrix.csv`

### 04-03-heim-ct-compute-metrics.py
Computes enrollment equity metrics aligned with the HEIM framework.

**Metrics Computed:**
1. **Enrollment Representation Index (ERI)**: Observed % / Reference %
   - < 0.5: Severely underrepresented
   - 0.5-0.8: Underrepresented
   - 0.8-1.2: Proportional
   - > 1.2: Overrepresented

2. **CT Gap Score (0-100)**: Trial coverage gap vs disease burden
   - Critical (≥70), High (50-69), Moderate (30-49), Low (<30)

3. **HIC/LMIC Trial Ratio**: Geographic concentration (analogous to biobank 57.8:1 ratio)

4. **Sponsor Diversity**: By agency class (NIH, Industry, Other)

5. **Temporal Diversity Trends**: Year-over-year changes

6. **Combined Gap Score**: Integration with biobank metrics from 03-03

**Outputs:**
- `DATA/heim_ct_enrollment_equity.csv`
- `DATA/heim_ct_disease_equity.csv`
- `DATA/heim_ct_geographic_equity.csv`
- `DATA/heim_ct_metrics.json`
- `ANALYSIS/HEIM-CT/equity_report.txt`

## Execution Sequence

```bash
# Prerequisites
export AACT_USER='your_username'
export AACT_PASSWORD='your_password'

# Pipeline
python3.11 PYTHON/04-00-heim-ct-setup.py
python3.11 PYTHON/04-01-heim-ct-fetch.py --start-year 2010 --end-year 2024
python3.11 PYTHON/04-02-heim-ct-map-diseases.py
python3.11 PYTHON/04-03-heim-ct-compute-metrics.py
```

## Integration with HEIM-Biobank

Both pipelines use the same GBD 2021 disease taxonomy, enabling direct comparison:

| HEIM-Biobank | HEIM-CT |
|--------------|---------|
| 179 GBD diseases | Same 179 diseases |
| Gap Score (0-100) | CT Gap Score (0-100) |
| Equity Alignment Score (EAS) | Enrollment Representation Index (ERI) |
| HIC:LMIC ratio (57.8:1) | HIC:LMIC trial ratio |
| Publications per million DALYs | Trials per million DALYs |

## Key Integration Metrics

When biobank metrics (from 03-03) are available:

- **Combined Gap Score**: Average of biobank and CT gap scores per disease
- **Pipeline Equity Index**: Identifies diseases with gaps at both discovery and translation stages
- **Double Jeopardy Diseases**: Global South priority diseases with critical gaps in BOTH biobanks AND clinical trials

## Reference Populations

**US Census 2020** (default for ERI):
- White (non-Hispanic): 57.8%
- Black/African American: 12.1%
- Asian: 5.9%
- Hispanic/Latino: 18.7%
- American Indian/Alaska Native: 0.7%
- Native Hawaiian/Pacific Islander: 0.2%
- Multiple races: 10.7%

## Methodology

Metrics adapted from:
> Corpas M, et al. (2025). EHR-Linked Biobank Expansion Reveals Global Health Inequities. Annual Review of Biomedical Data Science.

## Known Limitations

1. **Demographics posting rate**: Only ~15-20% of trials post demographic data
2. **Race/ethnicity standardization**: Categories vary across trials
3. **Geographic bias**: Majority of trials are US-based
4. **Disease mapping precision**: Keyword matching may miss complex conditions

## Version

HEIM-CT v1.0 (2026-01-13)
