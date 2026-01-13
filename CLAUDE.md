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
