# Enhanced Methodology Section

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

