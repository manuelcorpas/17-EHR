# HEIM Manuscript Revision Guide

Generated: 2026-01-13 06:22

## Overview

This guide provides specific text recommendations for addressing reviewer 
comments. Each section corresponds to numbered reviewer comments.

---

## Comment-by-Comment Response Plan

### Comments 1-2: Conceptual Framing of "Equity"

**Action required:** Add new "Normative Framework" subsection to Methods

**Key points to address:**
1. Clarify that HEIM measures disease-area alignment, not comprehensive equity
2. Acknowledge global vs. local burden tension
3. State explicitly what HEIM does NOT measure

**See:** 03-10-01-methodology_enhancements.md, Section 1-2

---

### Comments 3-4: HEIM Metric Transparency

**Action required:** 
1. Run 03-07-sensitivity-analysis.py
2. Include sensitivity results in Supplementary Materials
3. Add calibration documentation

**See:** 
- ANALYSIS/03-07-SENSITIVITY-ANALYSIS/03-07-06-sensitivity_report.md
- ANALYSIS/03-09-SUPPLEMENTARY-TABLES/03-09-02-burden_score_calibration.csv

---

### Comment 5: Capacity_Penalty Specification

**Action required:** Add full formula specification to Methods

**Text to add:**
```
Capacity_Penalty = 100 - min(P_b / N_d, 100)

Where P_b = Total publications from biobank b, and N_d = Number of 
GBD diseases in registry.
```

**See:** 03-10-01-methodology_enhancements.md, Section "Capacity_Penalty Specification"

---

### Comments 6-7: Equity Ratio and EAS Categories

**Action required:**
1. Run 03-08-validation-metrics.py for within-income percentiles
2. Add threshold justification to Methods
3. Consider noting GDP-adjustment as limitation

**See:**
- ANALYSIS/03-08-VALIDATION-METRICS/03-08-01-within_income_eas.csv
- 03-10-01-methodology_enhancements.md, Section "EAS Threshold Justification"

---

### Comments 8-10: Data Completeness

**Action required:**
1. Run 03-09-supplementary-tables.py for search query documentation
2. Add IHCC coverage estimate to Discussion
3. Address PubMed indexing limitation

**See:**
- ANALYSIS/03-09-SUPPLEMENTARY-TABLES/03-09-01-search_queries.csv
- ANALYSIS/03-08-VALIDATION-METRICS/03-08-03-search_coverage_analysis.csv

---

### Comment 11: Disease Mapping

**Action required:**
1. Clarify multi-disease handling in Methods
2. Document 8% discordance in Supplement

**See:** ANALYSIS/03-08-VALIDATION-METRICS/03-08-02-disease_mapping_validation.csv

---

### Comments 13-14: Causal Language

**Action required:** Global find-and-replace using language_replacements.csv

**Key changes:**
- "systematic exclusion" → "patterns consistent with underrepresentation"
- "demonstrates" → "is consistent with"
- "proves" → "suggests"
- Remove unsupported causal attributions about UK Biobank success factors

**See:** language_replacements.csv

---

### Comment 15: Abstract Revision

**Action required:** Revise abstract per abstract_revision.md

**Key additions:**
- Explicit scope statement about what HEIM measures/doesn't measure
- Soften causal language
- Acknowledge alternative explanations

---

### Comments 17-18: Technical Documentation

**Action required:**
1. State analysis was not pre-registered
2. Add software versions to Methods or Supplement
3. Specify bootstrap CI type

**See:** ANALYSIS/SUPPLEMENTARY/Table_S4_software_versions.md

---

## Checklist for Revision

- [ ] Add "Normative Framework" subsection (Comments 1-2)
- [ ] Add "Scope and Limitations" subsection (Comment 2)
- [ ] Document Capacity_Penalty formula (Comment 5)
- [ ] Include sensitivity analyses in Supplement (Comments 3-4)
- [ ] Add within-income percentiles table (Comment 7)
- [ ] Include search query documentation (Comment 8)
- [ ] Clarify multi-disease mapping (Comment 11)
- [ ] Apply language replacements throughout (Comments 13-14)
- [ ] Revise abstract (Comment 15)
- [ ] Add software versions (Comment 18)
- [ ] State pre-registration status (Comment 17)
- [ ] Verify numeric consistency (Comment 26)

