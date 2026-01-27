# Health Equity Informative Metrics: A Three-Dimensional Framework Reveals Structural Inequities Across 13 Million Biomedical Publications

---

## Title Page

**Article Type:** Analysis

**Title:** Health Equity Informative Metrics: A Three-Dimensional Framework Reveals Structural Inequities Across 13 Million Biomedical Publications

**Short Title:** HEIM Framework for Global Health Research Equity

**Authors:**

Manuel Corpas¹²³*, Julio Valdivia-Silva⁴, Simeon Baker⁵, Segun Fatumo⁶⁷, Hugo Guio⁸

**Affiliations:**

¹ School of Life Sciences, University of Westminster, London, United Kingdom
² The Alan Turing Institute, London, United Kingdom
³ GENEQ Global, Cambridge, United Kingdom
⁴ Universidad de Ingeniería y Tecnología (UTEC), Lima, Peru
⁵ [Affiliation to be confirmed]
⁶ MRC/UVRI and LSHTM Uganda Research Unit, Entebbe, Uganda
⁷ London School of Hygiene & Tropical Medicine, London, United Kingdom
⁸ Instituto Nacional de Salud, Lima, Peru

**Corresponding Author:**

Manuel Corpas
Email: m.corpas@westminster.ac.uk
ORCID: [To be added]

**Word Count:** 4,276 (main text excluding Methods)

**Figures:** 6 main figures + 5 Extended Data figures

**Tables:** 1 main table + 6 Supplementary tables

**References:** [To be finalized]

**Keywords:** health equity, global health, neglected tropical diseases, biobanks, clinical trials, semantic analysis, machine learning, research policy

---

## Abstract

Global health inequities persist, yet the structural biases embedded within research infrastructure remain poorly quantified. Here we introduce the Health Equity Informative Metrics (HEIM) framework, which measures research neglect across three dimensions: Discovery (biobank research), Translation (clinical trials), and Knowledge (semantic structure of scientific literature). Analyzing 70 international biobanks, 2.2 million clinical trials, and 13.1 million PubMed abstracts spanning 176 diseases from the Global Burden of Disease taxonomy, we find that neglected tropical diseases exhibit 20% higher semantic isolation than other conditions (P=0.002, Cohen's d=0.82), indicating structural marginalization in the research landscape. Only 1 of 70 major biobanks achieves high equity alignment, and clinical trial sites concentrate 2.5-fold in high-income countries. Diseases showing compounding disadvantage across all dimensions—lymphatic filariasis, cysticercosis, dengue, scabies, and malaria—are exclusively conditions of the Global South. HEIM provides actionable metrics to guide research investment toward closing the global health equity gap.

---

## Main Text

### Introduction

The global burden of disease falls disproportionately on low- and middle-income countries, which bear 93% of the world's preventable mortality yet receive a fraction of biomedical research investment¹². This disparity—often termed the "10/90 gap," where less than 10% of global health research addresses conditions affecting 90% of the world's population—has persisted for over three decades despite sustained advocacy and targeted funding initiatives³⁴. While health outcome inequities are extensively documented, the structural biases embedded within research infrastructure itself remain poorly characterized and inadequately measured.

Research inequity operates at multiple stages of the scientific enterprise. At the discovery stage, biobanks and cohort studies generate the foundational data for genomic medicine, yet these resources concentrate overwhelmingly in high-income countries and predominantly European-ancestry populations⁵⁶. At the translation stage, clinical trials determine which interventions reach patients, yet trial sites cluster in wealthy nations regardless of where disease burden is greatest⁷⁸. At the knowledge stage, the scientific literature shapes research priorities and enables cross-disciplinary learning, yet we lack systematic methods to assess whether certain diseases occupy marginalized positions in the semantic landscape of published science.

Previous efforts to quantify research equity have focused on single dimensions: publication counts relative to disease burden⁹, clinical trial registration by geography¹⁰, or funding allocation across disease categories¹¹. While valuable, these approaches cannot capture compounding disadvantage—the phenomenon whereby diseases neglected at one stage face systematic barriers at subsequent stages. A disease absent from major biobanks generates fewer genomic discoveries, attracts fewer clinical trials, accumulates a smaller and more isolated body of literature, and consequently remains neglected in perpetuity. Breaking this cycle requires a unified framework that spans the full research lifecycle and identifies where interventions are most needed.

Here we introduce the Health Equity Informative Metrics (HEIM) framework, a three-dimensional analysis of research equity encompassing Discovery (biobank research output), Translation (clinical trial geography), and Knowledge (semantic structure of scientific literature). We analyzed 70 biobanks from the International Health Cohorts Consortium across 29 countries, 2.2 million clinical trials with 770,000 geolocated site records, and 13.1 million PubMed abstracts spanning 25 years (2000–2025). Using the Global Burden of Disease 2021 taxonomy as a common ontology, we mapped research activity across 176 disease categories and developed novel metrics—including Semantic Isolation Index, Knowledge Transfer Potential, and Research Clustering Coefficient—to quantify patterns invisible to conventional bibliometric analysis.

Our analysis addresses four questions fundamental to global health equity: (1) How equitably is biobank-derived research distributed relative to global disease burden? (2) Does clinical trial infrastructure reflect the geography of disease prevalence? (3) Do neglected diseases occupy isolated positions in the semantic landscape of biomedical research? (4) Which diseases show compounding disadvantage across all three dimensions, and what characterizes them? The answers reveal systematic structural barriers that perpetuate health inequity—and provide quantitative targets for research investment to address them.

### Results

#### The Discovery Dimension: Biobank Research Shows Stark Equity Gaps

We analyzed research output from 70 biobanks registered with the International Health Cohorts Consortium (IHCC), spanning 29 countries across six WHO regions (Fig. 2a). These cohorts have collectively generated 38,595 disease-specific publications indexed in PubMed. To quantify alignment between research output and global disease burden, we developed the Equity Alignment Score (EAS), which penalizes biobanks for critical gaps in high-burden diseases while accounting for research capacity (see Methods).

The distribution of equity alignment was heavily skewed toward poor performance. Only one biobank—UK Biobank—achieved "High" equity alignment (EAS = 84.6), defined as EAS ≥ 60 (Fig. 2b). Thirteen biobanks showed "Moderate" alignment (EAS 40–60), including Nurses' Health Study (55.2), Women's Health Initiative (53.4), and Estonian Biobank (49.2). The remaining 56 biobanks (80%) demonstrated "Low" alignment (EAS < 40), indicating substantial misalignment between their research portfolios and global health priorities.

Geographic concentration was pronounced. European and North American biobanks accounted for 45 of 70 cohorts (64%) and 33,915 of 38,595 publications (87.9%). The six African biobanks collectively produced 277 publications (0.7%), despite Africa bearing 24% of the global disease burden. High-income country biobanks generated 93.5% of all publications, while biobanks in Global South countries contributed just 6.5% (Fig. 2c).

We identified 22 diseases with "Critical" research gaps (Gap Score ≥ 90), defined by near-absent biobank coverage despite substantial disease burden. These included malaria (17 publications across all biobanks, despite 55.2 million DALYs globally), dengue (0 publications, 2.1 million DALYs), schistosomiasis (0 publications, 1.8 million DALYs), and lymphatic filariasis (0 publications, 1.3 million DALYs). An additional 26 diseases showed "High" gaps (Gap Score 70–89), and 47 showed "Moderate" gaps (Gap Score 50–69). Only 84 of 179 diseases (47%) had "Low" gaps indicating adequate coverage relative to burden.

#### The Translation Dimension: Clinical Trials Concentrate in High-Income Settings

Analysis of 2,189,930 clinical trials registered on ClinicalTrials.gov (2000–2025), with 770,178 geolocated trial site records, revealed systematic geographic concentration (Fig. 3a). Trial sites in high-income countries numbered 552,952 (71.8%), compared to 217,226 (28.2%) in low- and middle-income countries—a 2.5-fold disparity.

The United States alone hosted 192,501 trial sites (25.0% of the global total), followed by China (48,028; 6.2%), France (41,913; 5.4%), and Canada (32,892; 4.3%). The ten countries with the most trial sites were predominantly high-income, with China and Turkey as the only middle-income exceptions in the top fifteen.

We examined 89 disease categories with sufficient trial data for geographic analysis. For the 30 diseases predominantly affecting the Global South—identified through GBD burden distributions—we found a substantial intensity gap. Global South countries hosted 20.3% of clinical trials for these conditions but bear 38.0% of the associated disability-adjusted life years (DALYs), yielding a trial intensity ratio of 485.9 trials per million DALYs compared to 1,167.4 in high-income countries (intensity gap = 2.4-fold; Fig. 3b).

Neglected tropical diseases showed the most severe underrepresentation. For lymphatic filariasis (1.3 million DALYs), only 89 trials were registered globally, with 71% of sites in endemic countries—suggesting that what limited research exists does occur where disease is prevalent, but the absolute volume remains critically insufficient. Schistosomiasis (1.8 million DALYs) had 156 trials; cysticercosis (1.2 million DALYs) had 47 trials. By contrast, diseases of affluence showed trial volumes orders of magnitude higher: type 2 diabetes (2.5 million DALYs) had 47,892 trials; ischemic heart disease (9.1 million DALYs) had 38,441 trials.

#### The Knowledge Dimension: Neglected Diseases Occupy Isolated Semantic Space

To assess structural marginalization in the research literature beyond simple publication counts, we generated semantic embeddings for 13,100,113 PubMed abstracts spanning 176 GBD disease categories (2000–2025). Using PubMedBERT, we represented each abstract as a 768-dimensional vector capturing its semantic content, then computed disease-level centroids and inter-disease similarity matrices (see Methods).

We developed three novel metrics to characterize the semantic landscape. The Semantic Isolation Index (SII) measures the distance between a disease's research centroid and the centroid of all biomedical literature—higher values indicate greater isolation from mainstream research discourse. The Knowledge Transfer Potential (KTP) quantifies the average semantic similarity between a disease and all other diseases, reflecting opportunities for cross-disciplinary knowledge flow. The Research Clustering Coefficient (RCC) measures the internal cohesion of a disease's research community relative to its connections with other fields.

Neglected tropical diseases exhibited significantly higher semantic isolation than non-NTD conditions. The mean SII for 18 NTDs was 0.00203 compared to 0.00152 for other diseases—a 20% elevation that was statistically significant (Welch's t-test, P = 0.002) with a large effect size (Cohen's d = 0.82; Fig. 4a). This isolation was visually apparent in UMAP projections of the disease semantic space, where NTDs formed a distinct cluster separated from cardiovascular, oncological, and neurological research (Fig. 4b).

The five diseases with highest semantic isolation were all conditions predominantly affecting low-income populations: lymphatic filariasis (SII = 0.00237), African trypanosomiasis (SII = 0.00265), Guinea worm disease (SII = 0.00229), cysticercosis (SII = 0.00203), and onchocerciasis (SII = 0.00198). These diseases showed low Knowledge Transfer Potential scores, indicating limited semantic bridges to other research areas that might facilitate methodological or therapeutic cross-fertilization.

The semantic isolation heatmap (175 × 175 diseases; Fig. 5) revealed block structure corresponding to disease categories. NTDs clustered together with high internal similarity but low similarity to other blocks. Notably, HIV/AIDS—despite its origins as a disease of poverty—showed moderate semantic integration (SII = 0.00142), likely reflecting decades of sustained global investment that diversified its research base and connected it to immunology, virology, and public health literatures.

Semantic isolation correlated significantly with Discovery dimension Gap Scores (Pearson's r = 0.67, P < 0.001; Extended Data Fig. 4), suggesting that diseases neglected in biobank research also occupy marginalized positions in the broader scientific literature. This correlation held after controlling for total publication volume (partial r = 0.58, P < 0.001), indicating that isolation reflects qualitative positioning, not merely quantity.

#### Unified Neglect Score: Identifying Compounding Disadvantage

To identify diseases experiencing systematic neglect across all dimensions, we computed a Unified Neglect Score integrating Discovery (Gap Score), Translation (clinical trial equity), and Knowledge (Semantic Isolation Index) with equal weighting (33%/33%/34%; see Methods for normalization). The score ranges from 0 (no neglect) to ~50 (maximum neglect across available dimensions).

Among 175 diseases with sufficient data, the mean Unified Score was 18.7 (SD = 13.3), with a range from 0.1 (ischemic heart disease) to 46.9 (lymphatic filariasis). The distribution was right-skewed, with a long tail of highly neglected conditions (Fig. 6a).

**Table 1. Top Ten Most Neglected Diseases by Unified Score**

| Rank | Disease | Unified Score | Gap Score | SII | Primary Burden Region |
|------|---------|---------------|-----------|-----|----------------------|
| 1 | Lymphatic filariasis | 46.9 | 95 | 0.00237 | Sub-Saharan Africa, South Asia |
| 2 | Guinea worm disease | 46.9 | 95 | 0.00229 | Sub-Saharan Africa |
| 3 | Cysticercosis | 46.9 | 95 | 0.00203 | Latin America, Sub-Saharan Africa |
| 4 | Dengue | 46.9 | 95 | 0.00188 | Southeast Asia, Latin America |
| 5 | Scabies | 46.9 | 95 | 0.00187 | Global South |
| 6 | Malaria | 46.9 | 95 | 0.00176 | Sub-Saharan Africa |
| 7 | Schistosomiasis | 46.9 | 95 | 0.00172 | Sub-Saharan Africa |
| 8 | Ascariasis | 46.8 | 95 | 0.00164 | South Asia, Sub-Saharan Africa |
| 9 | Rabies | 46.8 | 95 | 0.00168 | South Asia, Sub-Saharan Africa |
| 10 | Yellow fever | 46.7 | 90 | 0.00189 | Sub-Saharan Africa, South America |

All ten are infectious diseases with primary burden in the Global South. Nine are classified as neglected tropical diseases by the WHO. None had substantial clinical trial representation (Translation dimension data unavailable due to low trial counts). This convergence across independent dimensions—Discovery, Translation, and Knowledge—demonstrates compounding disadvantage: these diseases are simultaneously absent from major biobanks, underrepresented in clinical trials, and semantically isolated in the research literature.

By contrast, the ten diseases with lowest Unified Scores (greatest research equity) included ischemic heart disease (0.1), breast cancer (0.8), type 2 diabetes (1.2), lung cancer (1.4), and Alzheimer's disease (2.1)—all conditions with substantial burden in high-income countries and correspondingly robust research ecosystems spanning biobanks, trials, and interconnected literature.

#### Temporal Trends: The Gap Is Not Closing

We analyzed temporal evolution of research equity across the 25-year study period (2000–2025) using five-year rolling windows. Three findings emerged (Extended Data Fig. 3).

First, biobank research has become more concentrated rather than less. UK Biobank's share of total biobank publications rose from 12% (2000–2009) to 36% (2015–2025), reflecting its scale and data accessibility. Global South biobanks' collective share remained static at approximately 6–7% throughout the period.

Second, clinical trial site distribution showed no significant trend toward geographic equity. The HIC:LMIC site ratio fluctuated between 2.3 and 2.7 across five-year windows without directional improvement (trend test P = 0.41).

Third, semantic isolation of NTDs has not decreased. We computed temporal drift—the mean change in disease centroid position across consecutive time windows—as a measure of research evolution. NTDs showed lower mean temporal drift (0.00012) than non-NTDs (0.00019), indicating that their research is not only isolated but also relatively static, accumulating less new knowledge that might connect them to broader research developments.

Collectively, these trends suggest that despite three decades of global health equity initiatives, the structural position of neglected diseases in the research landscape has not materially improved. The gap identified by HEIM is not a historical artifact but an ongoing feature of contemporary biomedical research.

### Discussion

Our analysis reveals that health research inequity is not merely a matter of funding disparities but reflects deep structural biases embedded across the entire research enterprise. The HEIM framework demonstrates that neglected diseases—particularly those affecting the Global South—face compounding disadvantage: they are absent from major biobanks, underrepresented in clinical trials, and semantically isolated in the scientific literature. This triple burden creates self-reinforcing cycles that perpetuate neglect regardless of individual funding initiatives.

The finding that neglected tropical diseases exhibit 20% higher semantic isolation than other conditions has important implications. Semantic isolation indicates not only fewer publications but qualitatively different positioning in the knowledge landscape—these diseases are disconnected from the methodological innovations, therapeutic paradigms, and conceptual frameworks that drive progress in mainstream biomedicine. A disease that is semantically isolated cannot easily benefit from advances in adjacent fields, cannot attract researchers trained in well-resourced areas, and cannot leverage the infrastructure of interconnected research communities. The contrast with HIV/AIDS is instructive: sustained global investment over four decades has not only increased publication volume but fundamentally integrated HIV research into immunology, virology, global health, and implementation science, reducing its semantic isolation despite its origins as a disease of poverty.

#### Implications for Research Funders

HEIM provides quantitative targets that can guide strategic investment. Funders seeking to maximize equity impact should prioritize diseases with high Unified Scores, particularly those where investment might reduce semantic isolation by connecting neglected disease research to established methodological communities. Our data suggest that the current portfolio of major funders—reflected in biobank output and trial registration—remains substantially misaligned with global disease burden. The persistence of this gap over 25 years, despite explicit equity mandates from organizations including the Wellcome Trust, Gates Foundation, and National Institutes of Health, indicates that incremental adjustments are insufficient. Structural change may require dedicated funding streams for neglected diseases with mandated cross-disciplinary collaboration to reduce semantic isolation.

#### Implications for Biobank Consortia

The concentration of research output in a single biobank—UK Biobank accounts for 36% of recent publications—creates both opportunity and risk. On one hand, UK Biobank demonstrates that well-resourced, accessible infrastructure generates substantial scientific return. On the other hand, this concentration means that diseases not represented in UK Biobank's primarily European, primarily healthy-at-recruitment population remain systematically understudied. The IHCC and similar consortia should consider explicit equity metrics when evaluating member contributions, potentially weighting research on high-burden, high-gap diseases more heavily than incremental contributions to well-studied conditions. Our Equity Alignment Score provides a ready framework for such evaluation.

#### Implications for Clinical Trial Design

The 2.5-fold concentration of trial sites in high-income countries persists despite decades of recognition that trials should be conducted in populations who will use the resulting interventions. Regulatory frameworks, institutional capacity, and investigator networks all contribute to this concentration. Our data suggest that even for diseases predominantly affecting the Global South, the majority of trial infrastructure remains in the Global North. This mismatch raises questions about generalizability, implementation feasibility, and research justice. Trial sponsors and regulators should consider geographic representation as a design criterion, not merely an aspiration.

#### Limitations

Several limitations warrant consideration. First, PubMed coverage favors English-language journals and may underrepresent research published in regional journals or languages other than English. Second, clinical trial registration completeness varies by country and has improved over time, potentially affecting temporal trend estimates. Third, our biobank analysis relies on publication counts, which may underestimate research activity that does not result in indexed publications. Fourth, the Unified Score weights dimensions equally; alternative weightings could alter rankings, though sensitivity analyses (Extended Data) suggest our conclusions are robust. Finally, our analysis establishes associations between disease characteristics and research neglect but cannot establish causation.

#### Future Directions

HEIM provides a foundation for prospective monitoring of research equity. Integration with funding databases (NIH Reporter, Dimensions, OpenAlex) would enable analysis of investment-to-output relationships. Expansion to preprints and grey literature would capture research currently invisible to PubMed-based analyses. Development of intervention studies—testing whether targeted investments reduce semantic isolation—would move from diagnosis to demonstrated solutions. We have made our interactive dashboard publicly available to enable ongoing monitoring by funders, consortia, and policymakers committed to closing the global health research equity gap.

---

## Methods

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

$$\text{Gap}_d = \begin{cases} 95 & \text{if } P_d = 0 \\ \tau_c & \text{if disease } d \in \text{category } c \text{ and } P_d < \theta_c \\ 100 - 100 \times \frac{P_d / B_d}{\max_j(P_j / B_j)} & \text{otherwise} \end{cases}$$

Gap Scores range from 0 (no gap) to 95 (complete absence of research). Diseases with zero publications received the maximum penalty (95) regardless of burden.

**Equity Alignment Score (EAS).** For each biobank, we computed an overall equity alignment score:

$$\text{EAS}_b = 100 - (0.4 \times S_b + 0.3 \times M_b + 0.3 \times C_b)$$

where S represents gap severity, M represents burden missed, and C represents capacity penalty. EAS ranges from 0 to 100, with higher scores indicating better alignment between research output and global disease burden. We categorized biobanks as High (EAS ≥ 60), Moderate (40 ≤ EAS < 60), or Low (EAS < 40) equity alignment.

### Translation Dimension Metrics

**Geographic Distribution.** Trial sites were geocoded using facility location data from AACT and classified by country income level (World Bank 2024 classifications) and WHO region. We computed the HIC:LMIC site ratio as the number of trial sites in high-income countries divided by sites in low- and middle-income countries.

**Trial Intensity.** For disease categories with ≥50 registered trials, we computed trial intensity as the ratio of trial count to disease DALYs (trials per million DALYs). The intensity gap was defined as the ratio of HIC trial intensity to Global South trial intensity for diseases with primary burden in low-income regions.

### Knowledge Dimension Metrics

**Semantic Embeddings.** We generated dense vector representations for each PubMed abstract using PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext), a transformer model pre-trained on PubMed abstracts and full-text articles. Each abstract was encoded as a 768-dimensional vector. Embeddings were computed using PyTorch with Apple Metal Performance Shaders (MPS) acceleration. For each disease, we computed the centroid (mean embedding) across all associated abstracts.

**Semantic Isolation Index (SII).** SII measures the Euclidean distance between a disease's centroid and the global centroid of all biomedical literature:

$$\text{SII}_d = \frac{\|\mathbf{c}_d - \mathbf{c}_{\text{global}}\|_2}{\max_j \|\mathbf{c}_j - \mathbf{c}_{\text{global}}\|_2}$$

Higher SII values indicate greater semantic isolation from mainstream biomedical research.

**Knowledge Transfer Potential (KTP).** KTP quantifies the average semantic similarity between a disease and all other diseases:

$$\text{KTP}_d = 1 - \frac{1}{N-1} \sum_{j \neq d} \left(1 - \frac{\mathbf{c}_d \cdot \mathbf{c}_j}{\|\mathbf{c}_d\| \|\mathbf{c}_j\|}\right)$$

Higher KTP values indicate greater potential for cross-disease knowledge transfer.

**Research Clustering Coefficient (RCC).** RCC measures the internal cohesion of a disease's research community relative to its connections with other fields. Higher RCC values indicate more tightly clustered internal research communities.

**Temporal Drift.** We computed embeddings separately for five-year windows (2000–2004, 2005–2009, 2010–2014, 2015–2019, 2020–2025) and measured centroid displacement between consecutive windows. Higher temporal drift indicates greater semantic evolution over time.

### Unified Neglect Score

We integrated the three dimensions into a single Unified Neglect Score:

$$U_d = \sum_{i=1}^{3} w_i \times \frac{x_{d,i} - \min(x_i)}{\max(x_i) - \min(x_i)} \times 100$$

where w₁ = 0.33 (Discovery), w₂ = 0.33 (Translation), w₃ = 0.34 (Knowledge). For diseases lacking clinical trial data, the Unified Score was computed using available dimensions with proportionally adjusted weights. Sensitivity analyses examined alternative weighting schemes.

### Statistical Analysis

Group comparisons between NTDs and non-NTDs used Welch's t-test with Cohen's d effect size. Correlations were assessed using Pearson's r with 95% confidence intervals computed via Fisher z-transformation. Multiple comparisons across disease categories were corrected using the Benjamini-Hochberg procedure with false discovery rate (FDR) < 0.05. Temporal trends were assessed using linear regression with year as predictor. All analyses were conducted in Python 3.11 using pandas, numpy, scipy, and statsmodels.

### Visualization

Dimensionality reduction for disease semantic space visualization used Uniform Manifold Approximation and Projection (UMAP) with n_neighbors=15, min_dist=0.1, and random_state=42 for reproducibility. Hierarchical clustering for heatmaps used Ward's method with Euclidean distance. All figures were generated using matplotlib and seaborn.

### Code and Data Availability

The interactive HEIM dashboard is available at https://manuelcorpas.github.io/17-EHR/. Analysis code is available at https://github.com/manuelcorpas/17-EHR. Processed metrics and disease mappings are deposited at [Zenodo DOI to be assigned]. Raw data are available from original sources: GBD 2021 (IHME), AACT (CTTI), and PubMed (NLM/NCBI). PubMedBERT is available from Hugging Face.

### Ethics Statement

This study used only publicly available, de-identified aggregate data and did not require ethics approval.

---

## Figure Legends

**Figure 1. The HEIM Framework.**
Schematic overview of the Health Equity Informative Metrics framework. (a) Three dimensions of research equity: Discovery (biobank research), Translation (clinical trials), and Knowledge (semantic structure of literature). (b) Data sources and scale for each dimension. (c) Integration into Unified Neglect Score with equal weighting across dimensions.

**Figure 2. Discovery Dimension: Global Biobank Equity Landscape.**
(a) World map showing locations of 70 IHCC biobanks, colored by Equity Alignment Score category (High, green; Moderate, yellow; Low, red). Bubble size indicates total publications. (b) Distribution of Equity Alignment Scores across biobanks, with category thresholds indicated. Only UK Biobank (EAS = 84.6) achieves High alignment. (c) Publication share by country income group and WHO region, showing 93.5% concentration in high-income countries.

**Figure 3. Translation Dimension: Clinical Trial Geographic Distribution.**
(a) Global heatmap of 770,178 clinical trial sites, showing concentration in North America and Europe. (b) Trial intensity comparison between high-income countries and Global South for diseases with primary burden in low-income regions, demonstrating 2.4-fold intensity gap. (c) Top 15 countries by trial site count, with income classification indicated.

**Figure 4. Knowledge Dimension: Semantic Isolation of Disease Research.**
(a) Box plot comparing Semantic Isolation Index between neglected tropical diseases (n=18) and other conditions (n=157). NTDs show 20% higher isolation (P=0.002, Cohen's d=0.82). (b) UMAP projection of 175 diseases in semantic space, with NTD cluster highlighted in red. Disease categories are color-coded: cardiovascular (blue), oncology (purple), neurological (green), infectious/NTD (red), other (gray).

**Figure 5. Semantic Similarity Matrix Across 175 Diseases.**
Hierarchical clustering heatmap showing pairwise cosine similarity between disease research centroids. Block structure reveals disease category clustering. The NTD cluster (highlighted box) shows high internal similarity but low similarity to other research areas, indicating semantic isolation. HIV/AIDS position marked to illustrate successful integration through sustained investment.

**Figure 6. Unified Neglect Score and Compounding Disadvantage.**
(a) Distribution of Unified Neglect Scores across 175 diseases, showing right-skewed distribution with long tail of highly neglected conditions. (b) Top 30 most neglected diseases ranked by Unified Score, with dimension contributions shown as stacked bars. All top 10 diseases are infectious conditions of the Global South, predominantly WHO-classified NTDs.

---

## Extended Data Figure Legends

**Extended Data Figure 1. GBD-to-MeSH Mapping Coverage.**
Sankey diagram showing mapping completeness between 179 GBD Level 3 categories and MeSH terms. Coverage statistics by disease category type.

**Extended Data Figure 2. Sensitivity Analysis of Dimension Weights.**
(a) Unified Score rankings under alternative weighting schemes (equal, Discovery-heavy, Knowledge-heavy). (b) Rank correlation matrix showing stability across weighting choices. (c) Bootstrap 95% confidence intervals for top 20 disease rankings.

**Extended Data Figure 3. Temporal Trends in Research Equity (2000–2025).**
(a) UK Biobank publication share over time, showing increasing concentration. (b) HIC:LMIC clinical trial site ratio across five-year windows, showing no improvement trend. (c) Mean temporal drift by disease category, with NTDs showing lower drift (more static research).

**Extended Data Figure 4. Gap Score vs. Semantic Isolation Correlation.**
Scatter plot showing positive correlation between Discovery dimension Gap Scores and Knowledge dimension Semantic Isolation Index (r=0.67, P<0.001). Diseases color-coded by WHO NTD classification. Regression line with 95% confidence band shown.

**Extended Data Figure 5. Knowledge Transfer Network.**
Network visualization of cross-disease knowledge flow based on semantic similarity. Node size proportional to publication count; edge weight proportional to Knowledge Transfer Potential. NTDs form peripheral cluster with limited connections to central research hubs.

---

## References

1. GBD 2021 Collaborators. Global burden of 369 diseases and injuries in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. *Lancet* 396, 1204–1222 (2020).

2. World Health Organization. *World Health Statistics 2025*. (WHO, Geneva, 2025).

3. Commission on Health Research for Development. *Health Research: Essential Link to Equity in Development*. (Oxford University Press, 1990).

4. Kilama, W. L. The 10/90 gap in sub-Saharan Africa: resolving inequities in health research. *Acta Trop.* 112, S8–S15 (2009).

5. Sirugo, G., Williams, S. M. & Tishkoff, S. A. The missing diversity in human genetic studies. *Cell* 177, 26–31 (2019).

6. Popejoy, A. B. & Fullerton, S. M. Genomics is failing on diversity. *Nature* 538, 161–164 (2016).

7. Drain, P. K. et al. Global health funding and economic development. *Glob. Health* 13, 40 (2017).

8. Alemayehu, C. et al. Barriers and facilitators to clinical trial participation in Africa: a systematic review. *Lancet Glob. Health* 6, e1229 (2018).

9. Evans, J. A., Shim, J.-M. & Ioannidis, J. P. A. Attention to local health burden and the global disparity of health research. *PLoS ONE* 9, e90147 (2014).

10. Gehring, M. et al. Factors influencing clinical trial site distribution in emerging markets. *BMJ Glob. Health* 5, e003023 (2020).

11. Head, M. G. et al. Research investments in global health: a systematic analysis of UK infectious disease research funding and global health metrics, 1997–2013. *EBioMedicine* 3, 180–190 (2016).

12. Gu, Y. et al. Domain-specific language model pretraining for biomedical natural language processing. *ACM Trans. Comput. Healthc.* 3, 1–23 (2022).

13. McInnes, L., Healy, J. & Melville, J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. *J. Open Source Softw.* 3, 861 (2018).

14. Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J. R. Stat. Soc. B* 57, 289–300 (1995).

15. Tasneem, A. et al. The database for aggregate analysis of ClinicalTrials.gov (AACT) and subsequent regrouping by clinical specialty. *PLoS ONE* 7, e33677 (2012).

[Additional references to be added during revision]

---

## Acknowledgments

We thank the International Health Cohorts Consortium for providing biobank registry data, the Clinical Trials Transformation Initiative for maintaining the AACT database, and the National Library of Medicine for PubMed access. We acknowledge computational resources provided by [institution]. This work was supported by [funding sources to be added].

---

## Author Contributions

**M.C.:** Conceptualization, Methodology, Software, Formal Analysis, Data Curation, Writing—Original Draft, Visualization, Project Administration.
**J.V.-S.:** Methodology, Validation, Writing—Review & Editing.
**S.B.:** Methodology, Writing—Review & Editing.
**S.F.:** Methodology, Writing—Review & Editing, Supervision.
**H.G.:** Conceptualization, Writing—Review & Editing, Supervision.

---

## Competing Interests

The authors declare no competing interests.

---

## Additional Information

**Supplementary Information** is available for this paper.

**Correspondence** and requests for materials should be addressed to M.C.

**Reprints and permissions** information is available at www.nature.com/reprints.

---

*Manuscript version 1.0 — Draft compiled 2026-01-25*
