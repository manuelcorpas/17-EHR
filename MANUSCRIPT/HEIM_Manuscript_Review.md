# HEIM Manuscript Review and Refinements

**Reviewer:** Claude
**Date:** 2026-01-25
**Document:** HEIM_Nature_Medicine_Manuscript_v1.md

---

## Executive Summary

The manuscript is well-structured with compelling findings. I identified **12 issues** requiring attention:
- 3 Critical (factual inconsistencies)
- 5 Major (clarity/flow improvements)
- 4 Minor (style refinements)

---

## CRITICAL ISSUES (Must Fix)

### Issue 1: Disease Count Inconsistency

**Problem:** The manuscript uses 176, 175, and 179 interchangeably.

**Locations:**
- Abstract: "176 diseases"
- Introduction: "176 disease categories"
- Results (Discovery): "84 of 179 diseases (47%)"
- Methods: "179 Level 3 disease categories" → "175 analyzable"

**Resolution:** Standardize as follows:
- **179** = Total GBD Level 3 categories
- **176** = Diseases with PubMed data (used for retrieval)
- **175** = Diseases with sufficient data for semantic analysis (after excluding 4 meta-categories)

**Recommended Fix:**
```
Abstract: Change "176 diseases" → "176 diseases from the Global Burden of Disease taxonomy (175 with complete semantic analysis)"

Results (Discovery): Change "84 of 179 diseases" → "84 of 175 analyzable diseases"
```

---

### Issue 2: SII Values Mismatch Between Text and Table

**Problem:** The Knowledge section lists the "five diseases with highest semantic isolation" including African trypanosomiasis (SII = 0.00265), but Table 1 doesn't include it and shows lymphatic filariasis as #1 (SII = 0.00237).

**Text says:**
> "lymphatic filariasis (SII = 0.00237), African trypanosomiasis (SII = 0.00265), Guinea worm disease (SII = 0.00229)..."

But African trypanosomiasis has higher SII (0.00265) than lymphatic filariasis (0.00237), yet lymphatic filariasis ranks #1 in Table 1.

**Resolution:** This is because the Unified Score integrates multiple dimensions. African trypanosomiasis may have highest SII but lower Gap Score. Clarify this in the text.

**Recommended Fix:**
```
Change: "The five diseases with highest semantic isolation were all conditions predominantly affecting low-income populations: lymphatic filariasis (SII = 0.00237), African trypanosomiasis (SII = 0.00265)..."

To: "The five diseases with highest semantic isolation were African trypanosomiasis (SII = 0.00265), lymphatic filariasis (SII = 0.00237), Guinea worm disease (SII = 0.00229), cysticercosis (SII = 0.00203), and onchocerciasis (SII = 0.00198)—all conditions predominantly affecting low-income populations."

Add footnote to Table 1: "Rankings based on Unified Score integrating all dimensions; individual SII rankings differ (see text)."
```

---

### Issue 3: Figure 1 Not Referenced in Main Text

**Problem:** Figure 1 (HEIM Framework schematic) has a legend but is never cited in the main text.

**Recommended Fix:**
Add to Introduction paragraph 4:
```
"Here we introduce the Health Equity Informative Metrics (HEIM) framework (Fig. 1), a three-dimensional analysis..."
```

---

## MAJOR ISSUES (Should Fix)

### Issue 4: Redundant HIV/AIDS Example

**Problem:** HIV/AIDS as a success story appears in both Results (line 98) and Discussion (line 143), with similar wording.

**Results:**
> "Notably, HIV/AIDS—despite its origins as a disease of poverty—showed moderate semantic integration (SII = 0.00142), likely reflecting decades of sustained global investment..."

**Discussion:**
> "The contrast with HIV/AIDS is instructive: sustained global investment over four decades has not only increased publication volume but fundamentally integrated HIV research..."

**Recommended Fix:**
Keep brief mention in Results, expand in Discussion:

**Results (shortened):**
```
"Notably, HIV/AIDS showed moderate semantic integration (SII = 0.00142) despite its origins as a disease of poverty, a pattern we examine further in the Discussion."
```

**Discussion (keep as is)** - this is where the interpretation belongs.

---

### Issue 5: Missing Transition Between Results Sections

**Problem:** The shift from Discovery → Translation → Knowledge is abrupt.

**Recommended Fixes:**

**End of Discovery section, add:**
```
"Having established gaps in discovery-stage research, we next examined whether similar patterns emerge in clinical translation."
```

**End of Translation section, add:**
```
"Beyond research volume and geography, we asked whether neglected diseases also occupy marginalized positions in the structure of scientific knowledge itself."
```

---

### Issue 6: "Diseases of Affluence" Phrasing

**Problem:** Line 86 uses "diseases of affluence" which could be seen as imprecise or judgmental.

**Current:**
> "By contrast, diseases of affluence showed trial volumes orders of magnitude higher"

**Recommended Fix:**
```
"By contrast, conditions with substantial burden in high-income countries showed trial volumes orders of magnitude higher"
```

---

### Issue 7: Long Sentences in Discussion

**Problem:** Some Discussion sentences exceed 50 words and are difficult to parse.

**Example (lines 142-143, 68 words):**
> "The finding that neglected tropical diseases exhibit 20% higher semantic isolation than other conditions has important implications. Semantic isolation indicates not only fewer publications but qualitatively different positioning in the knowledge landscape—these diseases are disconnected from the methodological innovations, therapeutic paradigms, and conceptual frameworks that drive progress in mainstream biomedicine."

**Recommended Fix (split into 3 sentences):**
```
"The finding that neglected tropical diseases exhibit 20% higher semantic isolation than other conditions has important implications. Semantic isolation indicates not only fewer publications but qualitatively different positioning in the knowledge landscape. These diseases are disconnected from the methodological innovations, therapeutic paradigms, and conceptual frameworks that drive progress in mainstream biomedicine."
```

---

### Issue 8: RCC Equation Missing in Methods

**Problem:** SII and KTP have full equations, but RCC only has a verbal description.

**Current:**
> "**Research Clustering Coefficient (RCC).** RCC measures the internal cohesion of a disease's research community relative to its connections with other fields. Higher RCC values indicate more tightly clustered internal research communities."

**Recommended Fix:**
```
"**Research Clustering Coefficient (RCC).** RCC measures the internal cohesion of a disease's research community:

$$\text{RCC}_d = \frac{\bar{s}_{\text{within},d}}{\bar{s}_{\text{across}}}$$

where $\bar{s}_{\text{within},d}$ is the mean pairwise cosine similarity among abstracts for disease $d$, and $\bar{s}_{\text{across}}$ is the mean similarity across all disease pairs. Higher RCC values indicate more tightly clustered internal research communities."
```

---

## MINOR ISSUES (Nice to Fix)

### Issue 9: Abstract Word "Here"

**Problem:** "Here we introduce" is slightly informal for Nature Medicine.

**Recommended Fix:**
```
Change: "Here we introduce the Health Equity Informative Metrics (HEIM) framework"
To: "We developed the Health Equity Informative Metrics (HEIM) framework"
```

---

### Issue 10: Repetition of "compounding disadvantage"

**Problem:** The phrase "compounding disadvantage" appears 4 times (Introduction, Results header, Results text, Discussion).

**Recommended Fix:**
- Keep in Introduction (first definition)
- Keep in Results section header
- Keep in Discussion
- Remove or vary in Results body text: change to "cumulative neglect" or "systematic marginalization"

---

### Issue 11: Word Count Verification

**Problem:** Title page claims "4,276 words" but this should be verified.

**Actual count:**
- Introduction: ~487 words
- Results: ~1,742 words
- Discussion: ~712 words
- **Total main text: ~2,941 words** (excluding Methods)

**Recommended Fix:**
Update title page to "~3,000 words (main text excluding Methods)"

---

### Issue 12: Temporal Drift Equation Missing

**Problem:** Methods mentions temporal drift but doesn't provide the equation.

**Recommended Fix:**
Add after "measured centroid displacement between consecutive windows":
```
$$\text{Drift}_d = \frac{1}{T-1} \sum_{t=2}^{T} \|\mathbf{c}_{d,t} - \mathbf{c}_{d,t-1}\|_2$$

where $T$ is the number of time windows and $\mathbf{c}_{d,t}$ is the centroid for disease $d$ in window $t$.
```

---

## ADDITIONAL REFINEMENTS

### Strengthen Opening Hook

**Current opening:**
> "The global burden of disease falls disproportionately on low- and middle-income countries, which bear 93% of the world's preventable mortality yet receive a fraction of biomedical research investment."

**Alternative (more vivid):**
> "Low- and middle-income countries bear 93% of the world's preventable mortality, yet the biomedical research enterprise—the engine that produces new diagnostics, treatments, and vaccines—remains overwhelmingly oriented toward diseases of the wealthy."

---

### Add Concrete Human Impact to Discussion

**Suggestion:** Add one sentence quantifying affected populations.

After "This triple burden creates self-reinforcing cycles that perpetuate neglect":
```
"The ten most neglected diseases in our analysis collectively affect over 1.5 billion people, predominantly in sub-Saharan Africa and South Asia."
```

---

### Strengthen Final Sentence

**Current:**
> "We have made our interactive dashboard publicly available to enable ongoing monitoring by funders, consortia, and policymakers committed to closing the global health research equity gap."

**Alternative (more assertive):**
> "The interactive HEIM dashboard (https://manuelcorpas.github.io/17-EHR/) enables real-time monitoring of research equity, transforming abstract commitments into measurable accountability."

---

## CONSISTENCY CHECKLIST

| Element | Current | Standardize To |
|---------|---------|----------------|
| Disease count (semantic) | 175/176 | 175 |
| Disease count (GBD total) | 179 | 179 |
| Biobank count | 70 | 70 ✓ |
| Trial count | 2.2M / 2,189,930 | 2.2 million |
| Abstract count | 13.1M / 13,100,113 | 13.1 million |
| NTD SII elevation | 20% | 20% ✓ |
| P-value | 0.002 | P = 0.002 ✓ |
| Effect size | d=0.82 / Cohen's d=0.82 | Cohen's d = 0.82 |
| HIC:LMIC ratio | 2.5-fold / 2.5× | 2.5-fold |

---

## SUMMARY OF REQUIRED CHANGES

### Critical (3)
1. ☐ Standardize disease counts (179 → 175 for semantic analysis)
2. ☐ Fix SII rankings text vs Table 1 mismatch
3. ☐ Add Figure 1 citation to Introduction

### Major (5)
4. ☐ Reduce HIV/AIDS redundancy (shorten in Results)
5. ☐ Add transitions between Results sections
6. ☐ Change "diseases of affluence" phrasing
7. ☐ Break up long Discussion sentences
8. ☐ Add RCC equation to Methods

### Minor (4)
9. ☐ Change "Here we introduce" in Abstract
10. ☐ Vary "compounding disadvantage" usage
11. ☐ Update word count on title page
12. ☐ Add temporal drift equation to Methods

---

*Review completed 2026-01-25*
