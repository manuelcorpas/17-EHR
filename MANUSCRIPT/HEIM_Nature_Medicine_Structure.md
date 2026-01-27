# HEIM Framework — Nature Medicine Manuscript Structure

**Target Journal:** Nature Medicine
**Article Type:** Analysis
**Word Limit:** ~3,500 words (main text) + Methods
**Figures:** 6 main + Extended Data
**Authors:** Corpas M, Valdivia-Silva J, Baker S, Fatumo S, Guio H
**Affiliations:** University of Westminster | The Alan Turing Institute | GENEQ Global

---

## TITLE OPTIONS

**Option A (Framework-focused):**
> Health Equity Informative Metrics: A three-dimensional framework reveals structural inequities across 13 million biomedical publications

**Option B (Finding-focused):**
> Neglected tropical diseases are semantically isolated in global health research: Evidence from 13 million publications

**Option C (Action-focused):**
> Quantifying the discovery-to-translation gap: A framework for prioritizing global health research investments

---

## ABSTRACT (150 words)

[STRUCTURE]

**Background (1-2 sentences):**
Global health inequities persist despite decades of attention, yet the structural biases embedded in research infrastructure remain poorly quantified.

**Methods (2-3 sentences):**
We developed the Health Equity Informative Metrics (HEIM) framework to quantify research neglect across three dimensions: Discovery (70 IHCC biobanks, 38,595 publications), Translation (2.2 million clinical trials, 770,000 site records), and Knowledge (13.1 million PubMed abstracts with semantic embeddings). Using the Global Burden of Disease 2021 taxonomy, we analyzed 176 disease categories spanning 25 years (2000-2025).

**Results (3-4 sentences):**
We found that neglected tropical diseases show 20% higher semantic isolation than other disease categories (p=0.002, Cohen's d=0.82). Only 1 of 70 major biobanks achieves high equity alignment. Clinical trial sites are 2.5-fold concentrated in high-income countries. The most neglected diseases—lymphatic filariasis, cysticercosis, dengue—show compounding disadvantage across all three dimensions.

**Conclusion (1-2 sentences):**
HEIM provides actionable metrics to guide research prioritization and funding allocation toward closing the global health equity gap.

---

## INTRODUCTION (400-500 words)

### Paragraph 1: The Problem
[PLACEHOLDER - Frame the global health inequity problem]

- Health outcomes vary dramatically by geography and income
- The "10/90 gap" (10% of research addresses 90% of disease burden) persists
- Research infrastructure itself may perpetuate inequities
- Cite: GBD 2021, WHO reports, Lancet Commission on Global Health

### Paragraph 2: Current Limitations
[PLACEHOLDER - Why existing approaches are insufficient]

- Previous frameworks focus on single dimensions (funding, publications, trials)
- No unified metric exists to compare diseases across research stages
- Semantic structure of research literature has not been analyzed at scale
- Cannot identify "compounding neglect" where diseases are disadvantaged at every stage

### Paragraph 3: Our Contribution
[PLACEHOLDER - What HEIM offers]

- Three-dimensional framework: Discovery → Translation → Knowledge
- Unified scoring system enabling cross-disease comparison
- Novel semantic metrics (SII, KTP, RCC) revealing hidden isolation
- Scale: 13.1M papers, 2.2M trials, 70 biobanks, 176 diseases

### Paragraph 4: Key Questions
[PLACEHOLDER - Research questions addressed]

1. How equitably is biobank research distributed across the global disease burden?
2. Are clinical trials geographically representative of disease prevalence?
3. Do neglected diseases occupy isolated positions in the semantic landscape of research?
4. Which diseases show compounding disadvantage across all dimensions?

---

## RESULTS (1,500-1,800 words)

### Section 1: The Discovery Dimension — Biobank Research Equity (300 words)

**Key Data:**
- 70 IHCC biobanks from 29 countries
- 38,595 publications analyzed
- 179 GBD disease categories mapped

**Key Findings:**
- Only 1 biobank (UK Biobank) achieves "High" equity alignment (EAS=84.6)
- 13 biobanks: Moderate alignment (EAS 40-60)
- 56 biobanks: Low alignment (EAS <40)
- 93.5% of publications from HIC biobanks
- 6.5% from Global South biobanks

**Gap Score Distribution:**
- 22 diseases: Critical gaps (Gap Score ≥90)
- 26 diseases: High gaps (Gap Score 70-89)
- 47 diseases: Moderate gaps (Gap Score 50-69)
- 84 diseases: Low gaps (Gap Score <50)

[FIGURE 2: Global map of biobank equity alignment]

### Section 2: The Translation Dimension — Clinical Trial Geography (300 words)

**Key Data:**
- 2,189,930 clinical trials (ClinicalTrials.gov via AACT)
- 770,178 trial site records with geographic data
- 89 disease categories with sufficient data

**Key Findings:**
- HIC sites: 552,952 (71.8%)
- LMIC sites: 217,226 (28.2%)
- HIC:LMIC ratio: 2.5x
- Top countries: USA (192,501), China (48,028), France (41,913)

**Global South Analysis:**
- 30 diseases predominantly affecting Global South
- Global South receives 20.3% of trials but bears 38.0% of DALYs
- Trial intensity gap: 2.4x (HIC vs Global South)

[FIGURE 3: Clinical trial site distribution by WHO region and income level]

### Section 3: The Knowledge Dimension — Semantic Isolation of Disease Research (400 words)

**Key Data:**
- 13,100,113 PubMed abstracts (2000-2025)
- 176 diseases with sufficient publications
- PubMedBERT embeddings (768 dimensions)
- MeSH Major Topics [Majr] for precision retrieval

**Novel Metrics:**
1. **Semantic Isolation Index (SII):** Measures how distant a disease's research is from the centroid of all biomedical research
2. **Knowledge Transfer Potential (KTP):** Quantifies cross-disease knowledge flow based on semantic similarity
3. **Research Clustering Coefficient (RCC):** Measures density of research community structure
4. **Temporal Drift:** Tracks semantic evolution over 25 years

**Key Findings:**
- NTDs show 20% higher SII than non-NTDs (p=0.002, Cohen's d=0.82)
- Top 5 isolated diseases: Lymphatic filariasis, Guinea worm, Cysticercosis, Dengue, Scabies
- Isolation correlates with Gap Score (r=0.67, p<0.001)
- NTDs form distinct cluster in UMAP projection

[FIGURE 4: UMAP projection of disease semantic space with NTD cluster highlighted]
[FIGURE 5: Semantic isolation heatmap (175×175 diseases)]

### Section 4: Unified Neglect Score — Identifying Compounding Disadvantage (300 words)

**Methodology:**
- Unified Score = 0.33×Discovery + 0.33×Translation + 0.34×Knowledge
- Normalized to 0-100 scale
- Sensitivity analysis with alternative weightings

**Key Findings:**
- Mean Unified Score: 18.7 (SD=13.3)
- Range: 0.1 (Ischemic heart disease) to 46.9 (Lymphatic filariasis)

**Top 10 Most Neglected Diseases:**

| Rank | Disease | Unified Score | Discovery | Translation | Knowledge |
|------|---------|---------------|-----------|-------------|-----------|
| 1 | Lymphatic filariasis | 46.9 | Critical | N/A | High SII |
| 2 | Guinea worm disease | 46.9 | Critical | N/A | High SII |
| 3 | Cysticercosis | 46.9 | Critical | N/A | High SII |
| 4 | Dengue | 46.9 | Critical | N/A | High SII |
| 5 | Scabies | 46.9 | Critical | N/A | High SII |
| 6 | Animal contact | 46.9 | Critical | N/A | High SII |
| 7 | Malaria | 46.9 | Critical | N/A | High SII |
| 8 | Ascariasis | 46.8 | Critical | N/A | High SII |
| 9 | Rabies | 46.8 | Critical | N/A | High SII |
| 10 | Yellow fever | 46.7 | Critical | N/A | High SII |

**Pattern:** All top 10 are NTDs or infectious diseases predominantly affecting the Global South

[FIGURE 6: Unified neglect score ranking with dimension breakdown]

### Section 5: Temporal Trends — Is the Gap Closing? (200 words)

**Analysis Period:** 2000-2025 (5-year windows)

**Key Findings:**
- Overall semantic isolation of NTDs has not decreased
- Biobank publications increasingly concentrated in UK Biobank
- Clinical trial site distribution unchanged over 25 years
- Mean temporal drift lower for NTDs (less knowledge evolution)

[Extended Data Figure: Temporal trends panel]

---

## DISCUSSION (600-700 words)

### Paragraph 1: Summary of Findings
[PLACEHOLDER]

- HEIM reveals compounding disadvantage for NTDs across all three dimensions
- Semantic isolation is a novel and important indicator of research neglect
- The gap persists despite global health initiatives

### Paragraph 2: Implications for Funding Agencies
[PLACEHOLDER]

- HEIM provides quantitative targets for research investment
- Unified Score can guide portfolio allocation
- Semantic metrics reveal where knowledge transfer is blocked

### Paragraph 3: Implications for Biobank Consortia
[PLACEHOLDER]

- IHCC and other consortia can use EAS for strategic planning
- Global South biobanks need infrastructure investment
- Cross-biobank collaboration on neglected diseases

### Paragraph 4: Implications for Clinical Trial Design
[PLACEHOLDER]

- Geographic distribution should match disease burden
- Trial site selection criteria should include equity considerations
- Regulatory frameworks may need adaptation

### Paragraph 5: Limitations
[PLACEHOLDER]

- PubMed coverage bias (English-language, indexed journals only)
- Clinical trial registration completeness varies by country
- Biobank publication counts may underestimate unpublished research
- Causal relationships cannot be established

### Paragraph 6: Future Directions
[PLACEHOLDER]

- Integration with funding data (NIH Reporter, Wellcome, Gates)
- Expansion to preprints and grey literature
- Prospective tracking of HEIM metrics
- Development of intervention studies

---

## METHODS (1,000-1,200 words)

### Data Sources

**Global Burden of Disease 2021**
- Source: Institute for Health Metrics and Evaluation (IHME)
- Diseases: 179 Level 3 causes
- Metrics: DALYs, deaths, prevalence by country and year

**IHCC Biobanks**
- Source: International Health Cohorts Consortium
- Biobanks: 70 registered cohorts
- Publications: PubMed search by biobank name + disease MeSH terms

**Clinical Trials**
- Source: AACT (Aggregate Analysis of ClinicalTrials.gov)
- Trials: 2,189,930 registered trials (2000-2025)
- Site records: 770,178 with geographic coordinates

**PubMed Literature**
- Source: PubMed/MEDLINE via Entrez API
- Abstracts: 13,100,113 (2000-2025)
- Query strategy: MeSH Major Topics [Majr] + title/abstract keywords

### GBD-to-MeSH Mapping

[PLACEHOLDER - Describe the mapping methodology]
- Manual curation of 179 GBD diseases to MeSH terms
- Validation by clinical experts
- Handling of composite categories

### Metric Definitions

**Gap Score (Discovery Dimension)**
```
Gap_Score =
  - 95 if zero publications
  - Category-specific threshold if infectious/neglected
  - 100 - (Burden-normalized publication intensity) otherwise
```

**Equity Alignment Score (EAS)**
```
EAS = 100 - (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)

Where:
- Gap_Severity = Σ(Gap_Score × disease_weight)
- Burden_Miss = DALYs for diseases with ≤2 publications
- Capacity_Penalty = f(critical_gaps, high_gaps)
```

**Semantic Isolation Index (SII)**
```
SII = ||centroid_disease - centroid_all|| / max_distance

Where:
- centroid_disease = mean embedding of all papers for disease
- centroid_all = mean embedding of all 13.1M papers
```

**Knowledge Transfer Potential (KTP)**
```
KTP = 1 - mean(cosine_distance(disease_i, disease_j)) for all j ≠ i
```

**Research Clustering Coefficient (RCC)**
```
RCC = mean(cosine_similarity) within disease / mean(cosine_similarity) across diseases
```

**Unified Neglect Score**
```
Unified_Score = w1 × norm(Gap_Score) + w2 × norm(CT_Equity) + w3 × norm(SII)

Default weights: w1=0.33, w2=0.33, w3=0.34
```

### Embedding Generation

- Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
- Output: 768-dimensional dense vectors
- Hardware: Apple M-series with Metal Performance Shaders
- Abstracts processed: 13,100,113

### Statistical Analysis

- Group comparisons: Welch's t-test with Cohen's d effect size
- Correlations: Pearson's r with 95% CI
- Multiple comparisons: Benjamini-Hochberg FDR correction
- Sensitivity analysis: Alternative weighting schemes (equal, burden-weighted)

### Code and Data Availability

- Interactive webapp: https://manuelcorpas.github.io/17-EHR/
- Code repository: https://github.com/manuelcorpas/17-EHR
- Processed data: [Zenodo DOI to be assigned]
- Raw data: GBD (IHME), AACT (CTTI), PubMed (NLM)

---

## FIGURES

### Main Figures (6)

**Figure 1: HEIM Framework Schematic**
- Panel A: Three-dimensional framework (Discovery → Translation → Knowledge)
- Panel B: Data flow diagram
- Panel C: Unified Score calculation
- [TO CREATE: Illustrator/BioRender schematic]

**Figure 2: Global Biobank Equity Map**
- World map with biobank locations
- Color-coded by EAS category (High/Moderate/Low)
- Bubble size = publication count
- [SOURCE: docs/data/biobanks.json]

**Figure 3: Clinical Trial Geographic Distribution**
- Panel A: World map of trial site density
- Panel B: HIC vs LMIC bar chart by WHO region
- Panel C: Intensity gap visualization
- [SOURCE: docs/data/clinical_trials.json]

**Figure 4: Semantic Landscape of Disease Research**
- UMAP projection of 175 diseases
- NTD cluster highlighted
- Interactive version in webapp
- [SOURCE: ANALYSIS/05-04-HEIM-SEM-FIGURES/fig_umap_disease_clusters.png]

**Figure 5: Semantic Isolation Matrix**
- 175×175 heatmap with hierarchical clustering
- Disease categories annotated
- NTD block highlighted
- [SOURCE: ANALYSIS/05-04-HEIM-SEM-FIGURES/fig_semantic_isolation_heatmap.png]

**Figure 6: Unified Neglect Score Ranking**
- Horizontal bar chart of top 30 neglected diseases
- Stacked by dimension contribution
- NTDs highlighted
- [TO CREATE: Based on docs/data/integrated.json]

### Extended Data Figures (4-6)

**Extended Data Fig 1: GBD-to-MeSH Mapping Validation**
- Sankey diagram of disease category mappings
- Coverage statistics

**Extended Data Fig 2: Sensitivity Analysis**
- Alternative weighting schemes
- Robustness of rankings
- [SOURCE: docs/data/sensitivity_analysis.json]

**Extended Data Fig 3: Temporal Trends**
- Panel A: Biobank publication growth by region
- Panel B: Trial site distribution over time
- Panel C: Semantic drift by disease category
- [SOURCE: ANALYSIS/05-04-HEIM-SEM-FIGURES/fig_temporal_drift.png]

**Extended Data Fig 4: Gap Score vs Semantic Isolation**
- Scatter plot with regression line
- Confidence interval
- [SOURCE: ANALYSIS/05-04-HEIM-SEM-FIGURES/fig_gap_vs_isolation.png]

**Extended Data Fig 5: Knowledge Transfer Network**
- Network visualization of cross-disease knowledge flow
- Node size = publication count
- Edge weight = KTP
- [SOURCE: ANALYSIS/05-04-HEIM-SEM-FIGURES/fig_knowledge_network.png]

---

## SUPPLEMENTARY MATERIALS

### Supplementary Tables

**Table S1:** Complete GBD-to-MeSH mapping (179 diseases)
**Table S2:** Biobank characteristics and EAS scores (70 biobanks)
**Table S3:** Clinical trial summary by disease category (89 diseases)
**Table S4:** Semantic metrics for all diseases (175 diseases)
**Table S5:** Unified neglect scores with dimension breakdown (175 diseases)
**Table S6:** Sensitivity analysis results

### Supplementary Methods

**S1:** Detailed PubMed query construction
**S2:** Embedding generation pipeline
**S3:** Metric normalization procedures
**S4:** Statistical analysis code

### Supplementary Data

- All processed datasets (CSV/JSON)
- Embedding centroids (not full embeddings due to size)
- Reproducibility scripts

---

## AUTHOR CONTRIBUTIONS

**MC:** Conceptualization, methodology, software, formal analysis, data curation, writing—original draft, visualization, project administration

**JVS:** Methodology, validation, writing—review & editing

**SB:** Methodology, writing—review & editing

**SF:** Methodology, writing—review & editing, supervision

**HG:** Conceptualization, writing—review & editing, supervision

---

## COMPETING INTERESTS

[PLACEHOLDER - Declare any conflicts]

---

## ACKNOWLEDGMENTS

[PLACEHOLDER]
- IHCC consortium for biobank data access
- CTTI for AACT database
- Compute resources
- Funding sources

---

## REFERENCES

[Target: 40-50 references]

### Key Citations to Include

1. GBD 2021 Collaborators. Global burden of disease study 2021. Lancet 2024.
2. WHO. World Health Statistics 2025.
3. Kilama WL. The 10/90 gap in sub-Saharan Africa. Lancet 2009.
4. [IHCC consortium paper]
5. [AACT database paper]
6. Gu Y et al. Domain-specific language model pretraining for biomedical NLP. ACM CHIL 2021. [PubMedBERT]
7. [Previous health equity frameworks]
8. [NTD research gap papers]
9. [Clinical trial equity papers]
10. [Biobank equity papers]

---

## CHECKLIST FOR SUBMISSION

- [ ] Title page with all author information
- [ ] Abstract (150 words)
- [ ] Main text (3,500 words)
- [ ] Methods section
- [ ] References (40-50)
- [ ] Figure legends
- [ ] 6 main figures (high resolution)
- [ ] Extended data figures
- [ ] Supplementary tables
- [ ] Data availability statement
- [ ] Code availability statement
- [ ] Ethics statement (if applicable)
- [ ] Cover letter
- [ ] Suggested reviewers (3-5)
- [ ] Excluded reviewers (if any)

---

## TIMELINE

| Task | Status |
|------|--------|
| Structure complete | ✓ |
| Introduction draft | |
| Results draft | |
| Discussion draft | |
| Methods draft | |
| Figure preparation | |
| Co-author review | |
| Final revision | |
| Submission | |

---

*Document created: 2026-01-25*
*Last updated: 2026-01-25*
