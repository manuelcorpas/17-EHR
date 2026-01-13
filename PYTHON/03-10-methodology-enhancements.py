#!/usr/bin/env python3
"""
03-10-methodology-enhancements.py
=================================
HEIM-Biobank: Methodology Enhancements for Manuscript Revision

Generates text recommendations addressing:
- Comments 1-2: Equity conceptual framing
- Comments 13-14: Causal language softening
- Comment 15: Abstract clarifications


USAGE:
    python 03-10-methodology-enhancements.py
"""

import json
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Detect project root: DATA should be at project root, not in PYTHON/
if (SCRIPT_DIR / "DATA").exists():
    BASE_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "DATA").exists():
    BASE_DIR = SCRIPT_DIR.parent
else:
    # Fallback: assume current working directory
    BASE_DIR = Path.cwd()
    if not (BASE_DIR / "DATA").exists():
        BASE_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = BASE_DIR / "ANALYSIS" / "03-10-REVISION-GUIDE"


# =============================================================================
# LANGUAGE REPLACEMENTS (Comment 13-14)
# =============================================================================

LANGUAGE_REPLACEMENTS = [
    # Causal language softening
    {
        'original': 'systematic exclusion of populations',
        'replacement': 'patterns consistent with systematic underrepresentation of populations',
        'comment': '13',
        'rationale': 'Avoids implying intentional exclusion; acknowledges pattern without causal claim'
    },
    {
        'original': 'reveals structural inequity',
        'replacement': 'identifies structural patterns in research allocation',
        'comment': '13',
        'rationale': 'Descriptive rather than normative framing'
    },
    {
        'original': 'demonstrates what sustained investment, open data policies, and global collaboration can achieve',
        'replacement': 'correlates with higher EAS, consistent with the hypothesis that sustained investment and open data policies contribute to equity alignment',
        'comment': '14',
        'rationale': 'Removes unsupported causal attribution; presents as hypothesis'
    },
    {
        'original': 'confirms the systematic neglect',
        'replacement': 'is consistent with systematic neglect',
        'comment': '13',
        'rationale': 'Epistemic hedge; data consistent with but does not prove'
    },
    {
        'original': 'proves that',
        'replacement': 'suggests that',
        'comment': '13',
        'rationale': 'Observational data cannot prove causation'
    },
    {
        'original': 'reflects deep-rooted bias',
        'replacement': 'may reflect structural factors including resource constraints, research priorities, and institutional capacity',
        'comment': '13',
        'rationale': 'Acknowledges alternative explanations'
    },
    {
        'original': 'directly demonstrates exclusion',
        'replacement': 'reveals patterns consistent with differential attention',
        'comment': '13',
        'rationale': 'Pattern description rather than mechanism claim'
    },
    # Equity framing clarifications
    {
        'original': 'measures equity in biobank research',
        'replacement': 'measures alignment between biobank research output and global disease burden',
        'comment': '1-2',
        'rationale': 'Clarifies that HEIM measures topic allocation, not broader equity dimensions'
    },
    {
        'original': 'comprehensive equity metric',
        'replacement': 'disease-burden alignment metric',
        'comment': '1-2',
        'rationale': 'Avoids overstating scope; equity is multidimensional'
    },
    {
        'original': 'global biobank equity',
        'replacement': 'global biobank research-burden alignment',
        'comment': '1-2',
        'rationale': 'Precise terminology for what HEIM actually measures'
    },
]


# =============================================================================
# METHODOLOGY ENHANCEMENTS
# =============================================================================

def generate_methodology_section():
    """Generate enhanced methodology section addressing Comments 1-2, 5."""
    
    content = """# Enhanced Methodology Section

## Normative Framework: Global Burden as Benchmark

*[NEW SECTION - addresses Comment 1]*

This analysis adopts the position that global biobank infrastructure constitutes 
a global public good, particularly given the demonstrated portability of genomic 
discoveries across populations. We therefore evaluate research alignment against 
**global** disease burden rather than biobank-specific national burden.

We acknowledge this represents one of two defensible normative perspectives:

**(a) Global alignment perspective:** Biobank-enabled research should 
proportionally address diseases by global DALYs, given the translational 
potential of genomic discoveries across populations and the increasing 
interconnection of global health research networks.

**(b) Local responsiveness perspective:** Biobanks should primarily serve 
their source populations' health priorities, reflecting national funding 
mandates and the principle that research infrastructure should benefit 
contributing communities.

HEIM quantifies perspective (a) and makes no claims about perspective (b). 
Future iterations of the framework could incorporate local burden metrics 
to enable dual-perspective analysis.

---

## Scope and Limitations of HEIM

*[NEW SUBSECTION - addresses Comment 2]*

HEIM captures one dimension of research equity: **disease-area alignment with 
global burden**. It does not assess:

- **Research quality**: Publication counts are agnostic to study design, 
  sample size, or methodological rigor
- **Participant-level representation**: HEIM does not measure whether study 
  populations reflect global genetic diversity
- **Governance equity**: Access policies, benefit-sharing arrangements, and 
  authorship patterns are beyond the current scope
- **Translational impact**: Clinical or policy uptake of research findings 
  is not captured

These dimensions warrant complementary frameworks and should not be inferred 
from HEIM scores alone.

---

## Capacity_Penalty Specification

*[ENHANCED - addresses Comment 5]*

The Capacity_Penalty component of EAS quantifies publication intensity relative 
to disease coverage:

```
Capacity_Penalty = 100 - min(P_b / N_d, 100)
```

Where:
- **P_b** = Total publications from biobank *b*
- **N_d** = Number of GBD diseases in registry (currently 175)

**Interpretation:** This penalizes biobanks with low average publications per 
disease, reflecting limited research capacity to address the full burden 
spectrum. A biobank with 175 publications across 175 diseases (1 per disease 
average) receives zero penalty; a biobank with 17.5 publications receives a 
penalty of 90.

**Design rationale:** The penalty addresses Comment 5's concern about whether 
it "penalises HIC biobanks for not working on high-burden diseases that 
predominantly occur in LMICs, or penalises LMIC biobanks for low output." 
The answer is **both**: any biobank with low per-disease publication intensity 
receives a penalty, regardless of income classification. This reflects the 
principle that equity-aligned research requires both breadth (disease coverage) 
and depth (publication intensity).

The 100-point cap prevents extreme penalties for very small biobanks and 
ensures the component remains comparable to Gap_Severity and Burden_Miss.

---

## EAS Threshold Justification

*[NEW SUBSECTION - addresses Comment 7]*

EAS category thresholds were established *a priori* based on distributional 
considerations:

| Category | EAS Range | Distributional Basis |
|----------|-----------|---------------------|
| Strong | ≥ 80 | Upper quartile of possible scores |
| Moderate | 60-79 | Upper-middle range |
| Developing | 40-59 | Lower-middle range |
| Low | < 40 | Lower quartile |

Under current biobank landscape conditions, achieving Strong EAS requires:
- Publication presence across most GBD disease categories
- Research intensity above 1 publication per disease on average
- Coverage of high-burden infectious and neglected diseases

We acknowledge that these thresholds reflect structural geography and scale 
constraints. To enable fairer comparison, we report **within-income-group 
percentiles** (Supplementary Table SX) alongside global EAS scores.

---

## Alternative Interpretations

*[NEW SUBSECTION - addresses Comment 13]*

The patterns identified by HEIM are **consistent with**, but do not directly 
demonstrate, systematic neglect of high-burden diseases in LMIC settings. 
Alternative explanations include:

1. **Differential research capacity**: LMIC biobanks operate under resource 
   constraints that limit publication output regardless of disease priorities

2. **Funding alignment**: National funding agencies may prioritize domestic 
   health burdens over global burden metrics

3. **Disease epidemiology**: Some high-burden conditions (e.g., diarrheal 
   diseases, road injuries) may be less amenable to biobank-based study 
   designs than chronic NCDs

4. **Data infrastructure**: Diseases requiring longitudinal follow-up may 
   be systematically underrepresented in newer biobanks regardless of intent

These factors are not mutually exclusive with equity concerns but warrant 
acknowledgment when interpreting burden-alignment gaps.

"""
    return content


def generate_abstract_revision():
    """Generate revised abstract addressing Comment 15."""
    
    content = """# Revised Abstract

*[Addresses Comment 15: "The Summary is strong, but could be more precise 
about what HEIM does not measure"]*

---

## Original Abstract (excerpt)

> "We present the Health Equity Informative Metrics (HEIM) framework, 
> a comprehensive approach to quantifying equity in global biobank research..."

## Revised Abstract

**Background:** The rapid proliferation of population biobanks has transformed 
genomic research, yet it remains unclear whether research outputs align with 
global health priorities. We present the Health Equity Informative Metrics 
(HEIM) framework for quantifying alignment between biobank research output 
and global disease burden.

**Methods:** We analyzed [N] publications from [N] biobanks registered with 
the International Health Cohorts Consortium (IHCC), mapping research output 
to 175 disease categories from the Global Burden of Disease 2021 study. We 
computed burden-weighted research gap scores for each disease and Equity 
Alignment Scores (EAS) for each biobank. Bootstrap 95% confidence intervals 
were calculated using [N] iterations with percentile method.

**Findings:** [Key findings with specific numbers]

**Interpretation:** These metrics quantify disease-area alignment with global 
burden; they do not assess research quality, participant-level diversity, 
governance equity, or translational impact. The observed patterns are 
consistent with systematic underrepresentation of high-burden diseases 
prevalent in low- and middle-income settings, though alternative explanations 
including differential research capacity and funding priorities warrant 
consideration. HEIM provides a foundation for monitoring progress toward 
more globally aligned biobank research agendas.

**Funding:** [Funding sources]

---

## Key Changes

1. Changed "quantifying equity" to "quantifying alignment between research 
   output and global disease burden"

2. Added explicit scope statement: "These metrics quantify disease-area 
   alignment with global burden; they do not assess research quality, 
   participant-level diversity, governance equity, or translational impact"

3. Softened causal language: "consistent with systematic underrepresentation" 
   instead of "demonstrates systematic neglect"

4. Acknowledged alternative explanations in the Interpretation

"""
    return content


def generate_language_csv():
    """Generate CSV of language replacements."""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=['original', 'replacement', 'comment', 'rationale'])
    writer.writeheader()
    for row in LANGUAGE_REPLACEMENTS:
        writer.writerow(row)
    
    return output.getvalue()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("HEIM-Biobank: Methodology Enhancements")
    print("Addressing Reviewer Comments 1-2, 5, 13-15")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # Generate methodology enhancements
    print("\n📝 Generating methodology enhancements...")
    methodology = generate_methodology_section()
    methodology_file = OUTPUT_DIR / "03-10-01-methodology_enhancements.md"
    with open(methodology_file, 'w') as f:
        f.write(methodology)
    print(f"   Saved: {methodology_file}")
    
    # Generate language replacements
    print("\n📝 Generating language replacement guide...")
    language_csv = generate_language_csv()
    language_file = OUTPUT_DIR / "03-10-02-language_replacements.csv"
    with open(language_file, 'w') as f:
        f.write(language_csv)
    print(f"   Saved: {language_file}")
    print(f"   Replacements documented: {len(LANGUAGE_REPLACEMENTS)}")
    
    # Generate abstract revision
    print("\n📝 Generating abstract revision...")
    abstract = generate_abstract_revision()
    abstract_file = OUTPUT_DIR / "03-10-03-abstract_revision.md"
    with open(abstract_file, 'w') as f:
        f.write(abstract)
    print(f"   Saved: {abstract_file}")
    
    # Generate comprehensive revision guide
    print("\n📝 Generating comprehensive revision guide...")
    
    guide = f"""# HEIM Manuscript Revision Guide

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

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

"""
    
    guide_file = OUTPUT_DIR / "03-10-04-revision_guide.md"
    with open(guide_file, 'w') as f:
        f.write(guide)
    print(f"   Saved: {guide_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("METHODOLOGY ENHANCEMENTS COMPLETE")
    print("=" * 70)
    
    print(f"\n📁 Output files in: {OUTPUT_DIR}")
    print(f"   - 03-10-01-methodology_enhancements.md")
    print(f"   - 03-10-02-language_replacements.csv")
    print(f"   - 03-10-03-abstract_revision.md")
    print(f"   - 03-10-04-revision_guide.md")
    
    print("\n📋 Next steps:")
    print("   1. Run all analysis scripts (03-07, 03-08, 03-09)")
    print("   2. Review 03-10-04-revision_guide.md for complete checklist")
    print("   3. Apply 03-10-02-language_replacements.csv throughout manuscript")
    print("   4. Incorporate 03-10-01-methodology_enhancements.md into Methods section")


if __name__ == "__main__":
    main()