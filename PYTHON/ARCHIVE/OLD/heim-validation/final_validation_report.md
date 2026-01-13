# HEIM-Biobank Data Validation Report

**Generated:** December 24, 2025  
**Data Files Analyzed:** 8 JSON files from dashboard

---

## Executive Summary

The HEIM-Biobank dashboard data has **6 critical issues** that need to be addressed before the dashboard can be considered production-ready.

### Critical Issues Found

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | **3 diseases with 0 publications** | 🔴 CRITICAL | Type 2 Diabetes (67M DALYs), Chronic Kidney Disease (35M DALYs), Road Traffic Injuries (85.5M DALYs) show zero publications |
| 2 | **Matrix only 30% complete** | 🔴 CRITICAL | Matrix contains 22,353 pubs but biobanks.json claims 75,356 |
| 3 | **Total doesn't match manuscript** | 🟡 WARNING | Dashboard: 75,356 vs Manuscript: 14,142 publications |
| 4 | **UK Biobank count inflated** | 🟡 WARNING | Dashboard: 13,341 vs Manuscript: 10,752 |
| 5 | **Estonian Biobank count deflated** | 🟡 WARNING | Dashboard: 116 vs Manuscript: 663 |
| 6 | **27 biobanks vs manuscript's 5** | 🟡 CLARIFY | Scope needs clarification |

---

## Detailed Findings

### 1. Zero-Publication Diseases (CRITICAL)

These major diseases show **0 publications** which is clearly incorrect:

| Disease | Global Burden (DALYs) | Publications | Expected |
|---------|----------------------|--------------|----------|
| Type 2 Diabetes Mellitus | 67.0 million | 0 | >1,000 |
| Road Traffic Injuries | 85.5 million | 0 | >200 |
| Chronic Kidney Disease | 35.0 million | 0 | >500 |

**Root Cause:** MeSH term mapping likely failed for these disease categories.

### 2. Matrix Completeness Issue

```
Matrix grand total:     22,353 publications
biobanks.json claims:   75,356 publications
MISSING:                53,003 publications (70%)
```

The matrix only covers disease-specific publications, but the gap is too large to explain by methodology papers alone.

### 3. Manuscript vs Dashboard Comparison

**Manuscript (5 biobanks):**
| Biobank | Manuscript | Dashboard | Difference |
|---------|------------|-----------|------------|
| UK Biobank | 10,752 | 13,341 | +2,589 |
| FinnGen | 1,806 | 2,683 | +877 |
| Estonian Biobank | 663 | 116 | -547 |
| All of Us | 535 | 613 | +78 |
| Million Veteran Program | 386 | 327 | -59 |
| **TOTAL** | **14,142** | **17,080** | +2,938 |

### 4. Extra Biobanks (22 not in manuscript)

The dashboard includes 22 additional biobanks totaling 58,276 extra publications:

| Biobank | Publications |
|---------|--------------|
| eMERGE Network | 47,755 |
| deCODE Genetics | 5,604 |
| HUNT Study | 1,152 |
| LifeLines | 730 |
| China Kadoorie Biobank | 516 |
| ... and 17 more | ... |

---

## Root Cause Analysis

### Hypothesis 1: MeSH Mapping Failure
The zero-publication diseases suggest the MeSH term queries failed to match:
- "Diabetes Mellitus, Type 2" MeSH terms
- "Renal Insufficiency, Chronic" MeSH terms  
- "Accidents, Traffic" MeSH terms

### Hypothesis 2: Different Data Sources
The dashboard appears to use a different (expanded) dataset than the manuscript analyzed.

### Hypothesis 3: Matrix Generation Bug
The matrix generation script may have:
- Used wrong column indices
- Failed to aggregate some disease categories
- Applied different filtering criteria

---

## Recommendations

### Immediate Actions

1. **Fix Zero-Publication Diseases**
   - Review MeSH term mappings for Type 2 Diabetes, CKD, Road Traffic Injuries
   - Regenerate disease publication counts from verified PubMed queries

2. **Reconcile Matrix vs Total**
   - Clarify what the matrix represents (disease-specific only?)
   - Either expand matrix to cover all publications OR adjust biobank totals

3. **Clarify Dashboard Scope**
   - Is dashboard meant to show 5 biobanks (manuscript) or 27 biobanks (expanded)?
   - Document the data source and scope clearly

### Data Quality Checklist

- [ ] Verify MeSH mappings for all 25 diseases
- [ ] Cross-validate UK Biobank count against PubMed query
- [ ] Ensure matrix row sums match biobank totals
- [ ] Ensure matrix column sums match disease totals
- [ ] Add data provenance documentation
- [ ] Implement automated validation in data pipeline

---

## Validation Scripts

The following diagnostic scripts were created:

1. `validate_heim_data.py` - Basic validation of all JSON files
2. `deep_diagnostic.py` - Traces data inconsistencies  
3. `matrix_investigation.py` - Matrix completeness analysis
4. `comprehensive_audit.py` - Cross-reference with manuscript

Run all validations:
```bash
python validate_heim_data.py
python deep_diagnostic.py
python matrix_investigation.py
python comprehensive_audit.py
```
