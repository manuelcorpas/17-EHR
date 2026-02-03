# HEIM Framework — Methods Section Draft

**Target:** Nature Medicine
**Word count target:** 1,000-1,200 words
**Date:** 2026-01-25

---

## METHODS (1,187 words)

### Study Design and Overview

We developed the Health Equity Informative Metrics (HEIM) framework to quantify research equity across three dimensions of the biomedical research lifecycle: Discovery (biobank-derived research), Translation (clinical trials), and Knowledge (semantic structure of scientific literature). We used the Global Burden of Disease (GBD) 2021 taxonomy as a common ontology to enable cross-dimensional comparison. The study period spanned January 2000 to December 2025.

### Data Sources

**Global Burden of Disease 2021.** We obtained disease burden estimates from the Institute for Health Metrics and Evaluation (IHME) GBD 2021 study, including disability-adjusted life years (DALYs), deaths, and prevalence for 179 Level 3 disease categories across 204 countries and territories. Burden estimates were aggregated by World Bank income classification (high-income countries, HIC; low- and middle-income countries, LMIC) and WHO region.

**International Health Cohorts Consortium (IHCC).** We analyzed 70 biobanks registered with the IHCC as of January 2026, representing 29 countries across six WHO regions. For each biobank, we retrieved disease-specific publications from PubMed using structured queries combining biobank name variants with disease-specific Medical Subject Headings (MeSH) terms. The retrieval period was January 2000 to December 2025.

**Aggregate Analysis of ClinicalTrials.gov (AACT).** Clinical trial data were obtained from the AACT database, a PostgreSQL mirror of ClinicalTrials.gov maintained by the Clinical Trials Transformation Initiative. We extracted 2,189,930 registered trials with study start dates between January 2000 and December 2025, along with 770,178 facility records containing geographic coordinates. Trial-disease mapping used condition terms matched to GBD categories via MeSH cross-references.

**PubMed/MEDLINE.** We retrieved abstracts for 176 GBD disease categories from PubMed using the Entrez E-utilities API. Queries employed MeSH Major Topics [Majr] to ensure retrieved articles had the disease as a primary focus, combined with title/abstract keyword searches for diseases lacking adequate MeSH coverage. The final corpus comprised 13,100,113 unique abstracts.

### GBD-to-MeSH Disease Mapping

We developed a systematic mapping between 179 GBD Level 3 disease categories and corresponding MeSH terms. For each GBD category, we identified: (1) exact MeSH matches where available; (2) combinations of related MeSH terms for composite categories; and (3) title/abstract keyword supplements for conditions with incomplete MeSH coverage. Two investigators (MC, JVS) independently curated mappings, with discrepancies resolved by consensus. The complete mapping is provided in Supplementary Table S1. Four GBD categories representing methodological meta-categories rather than specific diseases were excluded, yielding 175 analyzable disease categories for semantic analysis.

### Discovery Dimension Metrics

**Gap Score.** For each biobank-disease pair, we computed a Gap Score reflecting research deficit relative to disease burden:

```
Gap_Score =
  95                                    if publications = 0
  Category_threshold                    if infectious/neglected disease below threshold
  100 - (Publication_intensity × 100)   otherwise

Where:
  Publication_intensity = (Publications / Burden_Score) / max(Publications / Burden_Score)
  Burden_Score = 0.5 × DALYs + 50 × Deaths + 10 × log₁₀(Prevalence)
```

Gap Scores range from 0 (no gap) to 95 (complete absence of research). Diseases with zero publications received the maximum penalty (95) regardless of burden.

**Equity Alignment Score (EAS).** For each biobank, we computed an overall equity alignment score:

```
EAS = 100 - (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)

Where:
  Gap_Severity = Σ(Gap_Score_i × Disease_Weight_i) / Σ(Disease_Weight_i)
  Burden_Miss = Total DALYs for diseases with ≤2 publications
  Capacity_Penalty = f(n_critical_gaps, n_high_gaps, total_publications)
```

EAS ranges from 0 to 100, with higher scores indicating better alignment between research output and global disease burden. We categorized biobanks as High (EAS ≥ 60), Moderate (40 ≤ EAS < 60), or Low (EAS < 40) equity alignment.

### Translation Dimension Metrics

**Geographic Distribution.** Trial sites were geocoded using facility location data from AACT and classified by country income level (World Bank 2024 classifications) and WHO region. We computed the HIC:LMIC site ratio as the number of trial sites in high-income countries divided by sites in low- and middle-income countries.

**Trial Intensity.** For disease categories with ≥50 registered trials, we computed trial intensity as the ratio of trial count to disease DALYs (trials per million DALYs). The intensity gap was defined as the ratio of HIC trial intensity to Global South trial intensity for diseases with primary burden in low-income regions.

### Knowledge Dimension Metrics

**Semantic Embeddings.** We generated dense vector representations for each PubMed abstract using PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext), a transformer model pre-trained on PubMed abstracts and full-text articles. Each abstract was encoded as a 768-dimensional vector. Embeddings were computed using PyTorch with Apple Metal Performance Shaders (MPS) acceleration. For each disease, we computed the centroid (mean embedding) across all associated abstracts.

**Semantic Isolation Index (SII).** SII measures the Euclidean distance between a disease's centroid and the global centroid of all biomedical literature:

```
SII_d = ||centroid_d - centroid_global|| / max_distance

Where:
  centroid_d = mean embedding of all abstracts for disease d
  centroid_global = mean embedding of all 13.1M abstracts
  max_distance = maximum observed centroid distance across all diseases
```

Higher SII values indicate greater semantic isolation from mainstream biomedical research.

**Knowledge Transfer Potential (KTP).** KTP quantifies the average semantic similarity between a disease and all other diseases:

```
KTP_d = 1 - mean(cosine_distance(centroid_d, centroid_j)) for all j ≠ d
```

Higher KTP values indicate greater potential for cross-disease knowledge transfer.

**Research Clustering Coefficient (RCC).** RCC measures the internal cohesion of a disease's research community:

```
RCC_d = mean(cosine_similarity within disease d) / mean(cosine_similarity across all diseases)
```

Higher RCC values indicate more tightly clustered internal research communities.

**Temporal Drift.** We computed embeddings separately for five-year windows (2000–2004, 2005–2009, 2010–2014, 2015–2019, 2020–2025) and measured centroid displacement between consecutive windows:

```
Temporal_Drift_d = mean(||centroid_d,t - centroid_d,t-1||) across all time windows t
```

Higher temporal drift indicates greater semantic evolution over time.

### Unified Neglect Score

We integrated the three dimensions into a single Unified Neglect Score:

```
Unified_Score = w₁ × norm(Gap_Score) + w₂ × norm(CT_Equity) + w₃ × norm(SII)

Where:
  w₁ = 0.33 (Discovery weight)
  w₂ = 0.33 (Translation weight)
  w₃ = 0.34 (Knowledge weight)
  norm(x) = (x - min(x)) / (max(x) - min(x)) × 100
```

For diseases lacking clinical trial data (Translation dimension), the Unified Score was computed using available dimensions with proportionally adjusted weights. Sensitivity analyses examined alternative weighting schemes (equal weights, burden-proportional weights, single-dimension rankings).

### Statistical Analysis

Group comparisons between NTDs and non-NTDs used Welch's t-test with Cohen's d effect size. Correlations were assessed using Pearson's r with 95% confidence intervals computed via Fisher z-transformation. Multiple comparisons across disease categories were corrected using the Benjamini-Hochberg procedure with false discovery rate (FDR) < 0.05. Temporal trends were assessed using linear regression with year as predictor. All analyses were conducted in Python 3.11 using pandas, numpy, scipy, and statsmodels.

### Visualization

Dimensionality reduction for disease semantic space visualization used Uniform Manifold Approximation and Projection (UMAP) with n_neighbors=15, min_dist=0.1, and random_state=42 for reproducibility. Hierarchical clustering for heatmaps used Ward's method with Euclidean distance. All figures were generated using matplotlib and seaborn.

### Code and Data Availability

The interactive HEIM dashboard is available at https://manuelcorpas.github.io/17-EHR/. Analysis code is available at https://github.com/manuelcorpas/17-EHR. Processed metrics and disease mappings are deposited at [Zenodo DOI]. Raw data are available from original sources: GBD 2021 (IHME), AACT (CTTI), and PubMed (NLM/NCBI). PubMedBERT is available from Hugging Face (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).

### Ethics Statement

This study used only publicly available, de-identified aggregate data and did not require ethics approval.

---

## METHODS SUMMARY (for main text, ~200 words)

We developed the HEIM framework to quantify research equity across Discovery (biobank research), Translation (clinical trials), and Knowledge (scientific literature) dimensions. Using the GBD 2021 taxonomy, we mapped 176 disease categories to MeSH terms and retrieved disease-specific publications for 70 IHCC biobanks (38,595 publications), clinical trial records from AACT (2.2 million trials, 770,000 sites), and PubMed abstracts (13.1 million). For the Knowledge dimension, we generated 768-dimensional semantic embeddings using PubMedBERT and computed novel metrics: Semantic Isolation Index (distance from global research centroid), Knowledge Transfer Potential (cross-disease similarity), and Research Clustering Coefficient (internal cohesion). We integrated dimensions into a Unified Neglect Score with equal weighting. Group comparisons used Welch's t-test with Cohen's d; correlations used Pearson's r with FDR correction. Full methods, code, and an interactive dashboard are available online.

---

## SUPPLEMENTARY METHODS OUTLINE

**Supplementary Methods S1: PubMed Query Construction**
- Detailed query syntax for each disease category
- Handling of MeSH term combinations
- Date restrictions and field specifications
- Deduplication procedures

**Supplementary Methods S2: Embedding Generation Pipeline**
- Hardware specifications (Apple M-series, MPS)
- Batch processing parameters
- Memory management for 13.1M abstracts
- Quality control checks

**Supplementary Methods S3: Metric Normalization**
- Min-max scaling procedures
- Handling of missing data
- Outlier treatment
- Sensitivity to normalization choices

**Supplementary Methods S4: Sensitivity Analyses**
- Alternative dimension weights (0.5/0.25/0.25, 0.25/0.25/0.5, etc.)
- Exclusion of diseases with limited data
- Bootstrap confidence intervals for Unified Scores
- Leave-one-out stability analysis for biobank rankings

---

## KEY TECHNICAL DETAILS FOR REVIEWERS

### Why PubMedBERT?
- Domain-specific pre-training on biomedical text
- Superior performance on biomedical NLP benchmarks vs. general BERT
- 768 dimensions balance expressiveness and computational tractability
- Citation: Gu et al. 2021, Domain-Specific Language Model Pretraining for Biomedical NLP

### Why MeSH Major Topics [Majr]?
- Ensures disease is primary focus of article, not incidental mention
- Reduces noise from tangentially related publications
- Standard practice in systematic reviews and bibliometric analyses
- Trade-off: may undercount some relevant research (addressed in Limitations)

### Why Equal Dimension Weights?
- No a priori reason to privilege one dimension
- Sensitivity analysis shows rankings robust to alternative schemes
- Allows transparent, reproducible scoring
- Users can apply custom weights via interactive dashboard

### Handling Missing Clinical Trial Data
- 86 of 175 diseases lack sufficient trial data for Translation score
- For these diseases, Unified Score = (Discovery + Knowledge) / 2
- Clearly flagged in outputs ("dimensions_available" field)
- Sensitivity analysis confirms NTD rankings stable regardless

---

## EQUATIONS FORMATTED FOR PUBLICATION

**Equation 1: Gap Score**
$$
\text{Gap}_d = \begin{cases}
95 & \text{if } P_d = 0 \\
\tau_c & \text{if disease } d \in \text{category } c \text{ and } P_d < \theta_c \\
100 - 100 \times \frac{P_d / B_d}{\max_j(P_j / B_j)} & \text{otherwise}
\end{cases}
$$

**Equation 2: Equity Alignment Score**
$$
\text{EAS}_b = 100 - (0.4 \times S_b + 0.3 \times M_b + 0.3 \times C_b)
$$

**Equation 3: Semantic Isolation Index**
$$
\text{SII}_d = \frac{\|\mathbf{c}_d - \mathbf{c}_{\text{global}}\|_2}{\max_j \|\mathbf{c}_j - \mathbf{c}_{\text{global}}\|_2}
$$

**Equation 4: Knowledge Transfer Potential**
$$
\text{KTP}_d = 1 - \frac{1}{N-1} \sum_{j \neq d} \left(1 - \frac{\mathbf{c}_d \cdot \mathbf{c}_j}{\|\mathbf{c}_d\| \|\mathbf{c}_j\|}\right)
$$

**Equation 5: Unified Neglect Score**
$$
U_d = \sum_{i=1}^{3} w_i \times \frac{x_{d,i} - \min(x_i)}{\max(x_i) - \min(x_i)} \times 100
$$

---

## CITATIONS FOR METHODS

[M1] GBD 2021 Collaborators. Global burden of 369 diseases and injuries. Lancet 2024.

[M2] IHCC Consortium. International Health Cohorts Consortium: a global network. Nat Med 2022.

[M3] Tasneem A et al. The database for aggregate analysis of ClinicalTrials.gov (AACT). PLoS One 2012.

[M4] Gu Y et al. Domain-specific language model pretraining for biomedical NLP. ACM CHIL 2021.

[M5] McInnes L et al. UMAP: Uniform Manifold Approximation and Projection. JOSS 2018.

[M6] Benjamini Y, Hochberg Y. Controlling the false discovery rate. J R Stat Soc B 1995.
