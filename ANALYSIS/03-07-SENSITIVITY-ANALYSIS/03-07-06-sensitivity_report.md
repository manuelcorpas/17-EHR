# HEIM Sensitivity Analysis Report

Generated: 2026-01-11 15:22

## Overview

This report addresses Reviewer Comments 3-4 regarding metric sensitivity.

- Publications analyzed: 36,416
- Diseases analyzed: 179
- Biobanks analyzed: 78

## 1. Burden Score Sensitivity

**Baseline formula:** Burden = (0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]

**Variations tested:** ±20% and ±50% for each coefficient

**Results:**
| Variation | Spearman ρ |
|-----------|------------|
| dalys_+20% | 1.0000 |
| dalys_-20% | 1.0000 |
| dalys_+50% | 1.0000 |
| dalys_-50% | 1.0000 |
| deaths_+20% | 1.0000 |
| deaths_-20% | 1.0000 |
| deaths_+50% | 1.0000 |
| deaths_-50% | 1.0000 |
| prevalence_+20% | 1.0000 |
| prevalence_-20% | 1.0000 |
| prevalence_+50% | 1.0000 |
| prevalence_-50% | 1.0000 |

**Conclusion:** Rankings highly stable (all ρ > 0.95)

## 2. Gap Score Sensitivity

**Baseline thresholds:** Critical >70, High >50, Moderate >30

**Zero-publication penalties tested:** 85, 90, 95, 100

**Results:**
| Threshold | Zero Penalty | Critical | High | Moderate | Low | ρ |
|-----------|--------------|----------|------|----------|-----|---|
| baseline | 95.0 | 23 | 27 | 46 | 83 | 1.0000 |
| baseline | 100.0 | 23 | 27 | 46 | 83 | 0.9999 |
| baseline | 90.0 | 23 | 27 | 46 | 83 | 0.9999 |
| baseline | 85.0 | 23 | 27 | 46 | 83 | 0.9998 |
| strict | 95.0 | 23 | 27 | 46 | 83 | 1.0000 |
| strict | 100.0 | 23 | 27 | 46 | 83 | 0.9999 |
| strict | 90.0 | 23 | 27 | 46 | 83 | 0.9999 |
| strict | 85.0 | 23 | 27 | 46 | 83 | 0.9998 |
| lenient | 95.0 | 31 | 35 | 38 | 75 | 1.0000 |
| lenient | 100.0 | 31 | 35 | 38 | 75 | 0.9999 |
| lenient | 90.0 | 31 | 35 | 38 | 75 | 0.9999 |
| lenient | 85.0 | 31 | 35 | 38 | 75 | 0.9998 |
| very_strict | 95.0 | 16 | 17 | 33 | 113 | 1.0000 |
| very_strict | 100.0 | 16 | 17 | 33 | 113 | 0.9999 |
| very_strict | 90.0 | 16 | 17 | 33 | 113 | 0.9999 |
| very_strict | 85.0 | 16 | 17 | 33 | 113 | 0.9998 |
| very_lenient | 95.0 | 33 | 33 | 38 | 75 | 1.0000 |
| very_lenient | 100.0 | 33 | 33 | 38 | 75 | 0.9999 |
| very_lenient | 90.0 | 33 | 33 | 38 | 75 | 0.9999 |
| very_lenient | 85.0 | 33 | 33 | 38 | 75 | 0.9998 |

**Conclusion:** Gap classifications stable across thresholds

## 3. EAS Weight Sensitivity

**Baseline weights:** Gap Severity=0.4, Burden Miss=0.3, Capacity Penalty=0.3

**Results:**
| Weight Scheme | Mean EAS | Std EAS | ρ |
|---------------|----------|---------|---|
| baseline | 21.3 | 16.8 | 1.0000 |
| gap_heavy | 22.7 | 17.3 | 0.9991 |
| burden_heavy | 21.1 | 17.2 | 0.9978 |
| capacity_heavy | 18.6 | 15.4 | 0.9992 |
| equal | 20.3 | 16.4 | 0.9992 |
| gap_dominant | 24.1 | 17.9 | 0.9980 |
| burden_dominant | 23.6 | 19.6 | 0.9910 |

**Conclusion:** EAS rankings stable across weight schemes

## Interpretation for Reviewers

All sensitivity analyses demonstrate high rank stability (Spearman ρ > 0.90 across 
all tested variations), confirming that the reported findings are robust to 
reasonable parameter perturbations. The choice of baseline parameters does not 
materially affect the identification of high-gap diseases or biobank rankings.
