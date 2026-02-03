# Health Equity Informative Metrics (HEIM) v2.0: Equity Analysis of 70 International Health Cohorts Consortium Biobanks

**Interactive Dashboard and Registry-Based Assessment**

Manuel Corpas^1,2*^

^1^ School of Life Sciences, University of Westminster, London, United Kingdom
^2^ The Alan Turing Institute, London, United Kingdom

\* Corresponding author: m.corpas@westminster.ac.uk

**Preprint version 2.0 | January 2026**

---

## Abstract

**Background:** Despite representing approximately 85% of the global population, low- and middle-income countries (LMICs) contribute less than 10% of participants to genome-wide association studies and biobank research. This disparity has profound implications for the generalisability of precision medicine. However, no standardised framework exists to quantify research equity at the biobank level or track progress over time.

**Methods:** We developed the Health Equity Informative Metrics (HEIM) framework to quantify alignment between biobank research output and global disease burden. We analysed 38,595 PubMed-indexed publications (2000-2025) from 70 IHCC-member biobanks (of 79 registered cohorts; 9 excluded for lacking PubMed-indexed publications) across 29 countries, mapping each to 179 disease categories from the Global Burden of Disease Study 2021. We calculated disease-specific Gap Scores measuring the mismatch between burden (disability-adjusted life years, DALYs) and research attention, biobank-level Equity Alignment Scores (EAS), and regional equity ratios comparing high-income (HIC) to LMIC research intensity.

**Findings:** Within the 70-biobank IHCC cohort, HIC biobanks produced substantially higher research output per DALY compared to LMIC biobanks (equity ratio: 57.8:1). Regional concentration was marked: the Americas and Europe accounted for 87.9% of publications, while Africa, Eastern Mediterranean, and South-East Asia combined contributed 2.2%. Of 179 disease categories, 48 (27%) exhibited critical or high-severity research gaps despite substantial global burden; 160 diseases were classified as realistically addressable by biobank research. Only 1 of 70 biobanks (1.4%) achieved 'Strong' equity alignment; 13 (18.6%) were 'Weak'; 56 (80%) were rated 'Poor'. Twenty-two diseases showed critical gaps, including malaria (55.2 million DALYs, 17 publications), dengue (2.1 million DALYs, 0 publications), and schistosomiasis (1.8 million DALYs, 0 publications).

**Interpretation:** The HEIM framework reveals substantial disparities in how biobank research capacity is distributed relative to global disease burden across IHCC-registered cohorts. UK Biobank emerges as an exceptional performer, contributing 35.7% of all publications and achieving the only 'Strong' EAS rating. These findings establish standardised baseline measurements for tracking progress toward more equitable genomic research and identify high-priority targets for intervention.

**Note:** This study analyses IHCC-registered biobanks using a standardised registry-based approach. A previous framework validation study (medRxiv DOI: 10.1101/2026.01.04.26343419) examined 27 high-output biobanks including eMERGE and deCODE. See Supplementary Information for detailed comparison.

**Keywords:** biobanks; health equity; genomics; global health; disability-adjusted life years; precision medicine; low- and middle-income countries; IHCC

---

## Introduction

The global genomics research enterprise faces a fundamental equity challenge. Despite representing approximately 85% of the world's population, low- and middle-income countries (LMICs) contribute less than 10% of participants across genome-wide association studies (GWAS), pharmacogenomics research, direct-to-consumer genomics, and clinical trials.^1-4^ This disparity has far-reaching consequences for the generalisability and clinical utility of precision medicine approaches.

The consequences of this representation gap are increasingly well-documented. Polygenic risk scores (PRS) developed primarily in European-ancestry populations show reduced predictive accuracy when applied to other populations, with performance decrements of 50% or greater reported in African-ancestry cohorts.^5-6^ Variant pathogenicity assessments in clinical databases such as ClinVar exhibit systematic biases toward well-studied populations, with variants common in African and Asian populations disproportionately classified as 'variants of uncertain significance'.^7-8^ These are not merely academic concerns; they represent systematic barriers to delivering equitable precision medicine globally.

Global health organisations, including the World Health Organization, have increasingly called for measurable frameworks to track progress toward equitable genomic research.^9^ A recent WHO report examining 6,513 genomic clinical studies conducted between 1990 and 2024 found that high-income countries (HICs) accounted for 68% of all studies, while low-income countries (LICs) contributed less than 0.5%.^14^ However, a significant methodological gap remains: while the problem of underrepresentation has been extensively documented, no standardised framework exists to quantify research equity at the biobank level, benchmark institutions against global health priorities, or track progress over time.

Biobanks represent critical infrastructure for genomic research, linking biological samples with electronic health records and enabling the large-scale studies necessary for genomic discovery.^10-12^ The International Health Cohorts Consortium (IHCC) provides a standardised registry of population cohorts worldwide, offering a unique opportunity to assess equity across a defined set of research resources. The geographic distribution of biobank capacity, however, mirrors broader inequities in research infrastructure.

This study introduces the Health Equity Informative Metrics (HEIM) framework to address this measurement gap. HEIM operationalises equity assessment by comparing biobank research output against global disease burden measured in disability-adjusted life years (DALYs) from the Global Burden of Disease Study 2021.^13^ HEIM treats equity as an engineering problem amenable to systematic measurement, tracking, and targeted intervention.

We present findings from the first comprehensive HEIM analysis of the IHCC cohort registry, examining 38,595 publications from 70 biobanks across 29 countries. Our objectives were to: (1) quantify the alignment between biobank research output and global disease burden; (2) identify diseases with critical research gaps despite high burden; (3) benchmark individual biobanks using standardised equity metrics; and (4) characterise regional disparities in research capacity. These baseline measurements establish a foundation for monitoring progress and informing strategic investments in genomic research equity.

---

## Methods

### Data Sources

#### Publication Data

We retrieved publications linked to biobanks registered with the International Health Cohorts Consortium (IHCC) from PubMed using the NCBI Entrez API. Biobank-specific queries were constructed using institution names, acronyms, and known variants (full query strings available at the HEIM Equity Tool: https://manuelcorpas.github.io/17-EHR/). The search period spanned January 2000 to December 2025, with a data cut-off date of December 31, 2025. Each publication's Medical Subject Headings (MeSH) terms were extracted and mapped to Global Burden of Disease (GBD) 2021 disease categories using a manually curated mapping dictionary. Manual review of 200 randomly sampled publications showed 92% agreement with automated MeSH-to-GBD mapping; discrepancies primarily occurred with multi-disease publications where secondary conditions were missed.

#### Disease Burden Data

Disability-adjusted life years (DALYs) were obtained from the Institute for Health Metrics and Evaluation (IHME) Global Burden of Disease Study 2021.^13^ DALYs combine years of life lost (YLL) due to premature mortality with years lived with disability (YLD), providing a comprehensive measure of disease burden. We used global, all-age, both-sexes estimates for the 179 disease categories included in our analysis.

#### Biobank Selection

We identified all cohorts in the International Health Cohorts Consortium (IHCC) Global Cohort Atlas as of December 2025 (N = 79). Of these:

- **70 had PubMed-indexed publications** and were included in analysis
- **9 were excluded** due to:
  - No PubMed-indexed publications (n = 7)
  - Insufficient metadata for analysis (n = 2)

All 70 analysed biobanks met the minimum inclusion criterion of ≥1 peer-reviewed publication. No additional publication threshold was applied to maximise representativeness. This standardised inclusion criterion ensures reproducibility and enables future updates as the IHCC registry expands.

The 70 analysed biobanks span 29 countries across six WHO regions (Table 1). Biobanks were classified by income status using World Bank 2024 classifications: 52 from high-income countries (HIC) and 18 from low- and middle-income countries (LMIC).

#### Research Appropriateness Classification

We classified diseases by their appropriateness for biobank-based research:

- **Highly appropriate (n=137):** Chronic diseases with genetic/environmental components amenable to cohort study (e.g., diabetes, cardiovascular disease, cancer)
- **Moderately appropriate (n=23):** Diseases where biobank data provides some insights but not primary research modality
- **Limited appropriateness (n=15):** Acute conditions, injuries, or external causes (e.g., drowning, road injuries, conflict)
- **Methodology (n=4):** Meta-categories for publications about methodology rather than specific diseases

This classification ensures biobanks are not penalised for low research on diseases outside their methodological scope. Of 179 GBD categories, 160 are realistically addressable by biobank research.

### Core Metric Definitions

The HEIM framework comprises three core components that together quantify research equity: Gap Severity (mismatch between research output and disease burden), Burden Miss (high-burden diseases with inadequate coverage), and Capacity Penalty (underutilisation of research capacity in underrepresented settings).

#### Equity Alignment Score (EAS)

The EAS provides an overall assessment of how well a biobank's research portfolio aligns with global disease burden:

> **EAS = 100 - (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)**

Scores range from 0-100 and are categorised as:

- **Strong (≥70):** Excellent alignment with global health priorities
- **Weak (40-69):** Moderate alignment, some gaps in priority diseases
- **Poor (<40):** Significant misalignment with global health priorities

EAS scoring uses global DALYs rather than region-specific burden; this is a deliberate normative choice to assess how well each biobank serves global rather than local health priorities.

#### Research Gap Score

The Gap Score measures the mismatch between disease burden and research attention using a three-tier system:

- **Tier 1 (Score 95):** Diseases with zero biobank-linked publications
- **Tier 2:** Category-based thresholds (stricter for infectious/neglected diseases due to higher Global South burden)
- **Tier 3:** Burden-normalised intensity for diseases above threshold

Gap severity was classified as: Critical (>70), High (50-70), Moderate (30-50), or Low (<30).

#### Equity Ratio

The equity ratio compares research intensity (publications per DALY) between high-income country (HIC) and low- and middle-income country (LMIC) biobanks:

> **Equity Ratio = (Publications_HIC / DALYs_HIC) / (Publications_LMIC / DALYs_LMIC)**

A ratio of 1.0 would indicate perfect equity; higher values indicate greater HIC advantage.

### Confidence Intervals

We calculated 95% confidence intervals for EAS scores based on publication sampling variability:

> **CI_width = 2.5 × (1 + log₁₀(max_pubs / biobank_pubs))**

Biobanks with higher publication volumes have narrower confidence intervals and more reliable EAS estimates.

### UK Biobank Sensitivity Analysis

Given UK Biobank's dominant position (35.7% of all publications), we conducted sensitivity analyses excluding UK Biobank to assess its influence on aggregate statistics.

### Data Availability and Reproducibility

All analysis code, mapping dictionaries, and aggregated data are available at the HEIM Equity Tool (https://manuelcorpas.github.io/17-EHR/). The tool displays this study's 70-biobank IHCC dataset. Original framework validation data (27 biobanks) are available in the v1.0 preprint (medRxiv DOI: 10.1101/2026.01.04.26343419). Raw publication data are derived from PubMed and subject to NLM terms of use. Disease burden data are available from the IHME Global Health Data Exchange.

---

## Results

### Overview of Dataset

We analysed 38,595 PubMed-indexed publications from 70 IHCC-registered biobanks across 29 countries (Table 1). Publications were mapped to 179 disease categories based on MeSH term annotations. The included biobanks span six WHO regions, with representation from both high-income (n=52) and low- and middle-income (n=18) countries.

### Regional Distribution of Research Output

Research output was highly concentrated in specific regions (Table 2). Europe contributed 55.7% of all publications (21,482 of 38,595), driven primarily by UK Biobank (13,785 publications, 35.7% of total). The Americas contributed 32.2% (12,433 publications), with the Nurses' Health Study (3,197), Women's Health Initiative (3,433), and All of Us Research Program (837) as leading contributors.

The Western Pacific region contributed 9.9% (3,819 publications), led by Biobank Japan (492), China Kadoorie Biobank (624), and Taiwan Biobank (312). Critically, Africa (277 publications, 0.7%), Eastern Mediterranean (574 publications, 1.5%), and South-East Asia (10 publications, <0.1%) combined contributed only 2.2% of total publications.

### Equity Ratio

Within the 70-biobank IHCC cohort, HIC biobanks produced substantially higher research output per DALY compared to LMIC biobanks (equity ratio: 57.8:1). This ratio indicates that for every unit of disease burden, HIC biobanks generated 57.8 times more research output than LMIC biobanks. While lower than the ratio observed in our previous framework validation study (322:1 in a 27-biobank sample including eMERGE Network), the finding confirms substantial inequity across the standardised IHCC registry.

### Disease-Level Research Gaps

Of 179 disease categories analysed, 22 (12.3%) showed critical research gaps, 26 (14.5%) showed high gaps, 47 (26.3%) showed moderate gaps, and 84 (46.9%) showed low gaps (adequate coverage). The 48 diseases with critical or high gaps represent priority targets for equity-focused investment.

Diseases with critical gaps (Gap Score >70) included:

| Disease | DALYs (millions) | Publications | Gap Score | Appropriateness |
|---------|------------------|--------------|-----------|-----------------|
| Malaria | 55.2 | 17 | 95 | High |
| Dengue | 2.1 | 0 | 95 | High |
| Schistosomiasis | 1.8 | 0 | 95 | High |
| Lymphatic filariasis | 1.3 | 0 | 95 | High |
| Cysticercosis | 1.2 | 0 | 95 | High |
| Scabies | 5.3 | 1 | 95 | Moderate |
| Typhoid and paratyphoid | 8.1 | 21 | 90 | High |
| Drowning | 15.7 | 0 | 95 | Limited* |
| Animal contact | 4.9 | 0 | 95 | Limited* |

*Limited appropriateness diseases are flagged but not used to penalise biobanks, as they fall outside typical biobank research scope.

**Interpretation note:** Diseases with limited appropriateness (drowning, road injuries, conflict/terrorism, animal contact) represent conditions where biobank-based research has inherent limitations. The 160 diseases with high or moderate appropriateness constitute the realistic target space for biobank equity assessment.

### Biobank-Level Equity Alignment

Equity Alignment Scores ranged from 1.7 (Civil Service Hospital Nepal) to 84.6 (UK Biobank). The distribution was heavily skewed toward lower scores (Table 3):

| Category | Score Range | Number of Biobanks | Percentage |
|----------|-------------|-------------------|------------|
| Strong | ≥70 | 1 | 1.4% |
| Weak | 40-69 | 13 | 18.6% |
| Poor | <40 | 56 | 80.0% |
| **Total** | - | **70** | **100%** |

Only UK Biobank achieved 'Strong' alignment (EAS 84.6, 95% CI: 82.1-87.1). The next highest-scoring biobanks were:

| Rank | Biobank | EAS | 95% CI | Publications | Confidence |
|------|---------|-----|--------|--------------|------------|
| 1 | UK Biobank | 84.6 | 82.1-87.1 | 13,785 | High |
| 2 | Nurses' Health Study | 55.2 | 53.0-57.4 | 3,197 | High |
| 3 | Women's Health Initiative | 53.4 | 51.3-55.5 | 3,433 | High |
| 4 | Estonian Biobank | 49.2 | 46.3-52.1 | 675 | Moderate |
| 5 | All of Us Research Program | 49.1 | 46.3-51.9 | 837 | Moderate |
| 6 | HUNT Study | 48.1 | 45.4-50.8 | 1,218 | Moderate |
| 7 | EPIC | 45.2 | 42.8-47.6 | 1,870 | High |
| 8 | Danish National Birth Cohort | 44.4 | 41.5-47.3 | 636 | Moderate |
| 9 | Genomics England | 43.6 | 40.8-46.4 | 700 | Moderate |
| 10 | MoBa | 43.1 | 40.2-46.0 | 962 | Moderate |

The concentration in the 'Poor' category (80%) reflects systemic factors including resource constraints, infrastructure limitations, and research agendas driven by local rather than global priorities.

### UK Biobank: Benchmark Institution Analysis

UK Biobank's exceptional position warrants specific analysis:

| Metric | With UK Biobank | Without UK Biobank |
|--------|-----------------|-------------------|
| Total biobanks | 70 | 69 |
| Total publications | 38,595 | 24,810 |
| Mean EAS | 24.1 | 23.2 |
| Median EAS | 19.7 | 19.5 |
| Strong category | 1 | 0 |
| Top EAS | 84.6 (UKB) | 55.2 (NHS) |

Key observations:

- UK Biobank contributes 35.7% of all IHCC-linked publications
- The gap between UK Biobank (84.6) and second-place Nurses' Health Study (55.2) is 29.4 points—exceeding the entire range among other biobanks
- Without UK Biobank, no biobank achieves 'Strong' alignment
- UK Biobank accounts for 64.2% of all European region publications

**Interpretation:** UK Biobank's exceptional performance reflects its design, funding, and infrastructure—it sets a benchmark rather than distorts the analysis. For policy purposes, comparing biobanks to the 'Weak' threshold (40-60) may be more actionable than comparison to UK Biobank.

### Global South and LMIC Biobanks

Eighteen biobanks in our sample serve Global South or LMIC populations. Despite resource constraints, several are emerging as important contributors:

| Biobank | Country | Region | Publications | Diseases | EAS |
|---------|---------|--------|--------------|----------|-----|
| China Kadoorie Biobank | China | WPR | 624 | 109 | 41.6 |
| ELSA-Brasil | Brazil | AMR | 418 | 89 | 38.0 |
| Qatar Biobank | Qatar | EMR | 287 | 72 | 21.2 |
| PERSIAN Cohort | Iran | EMR | 156 | 58 | 19.5 |
| AWI-Gen | Pan-African | AFR | 89 | 43 | 10.8 |
| Uganda Genome Resource | Uganda | AFR | 31 | 21 | 2.7 |

These biobanks face systematic challenges including limited funding and infrastructure constraints but provide critical representation for populations historically excluded from genomic research.

---

## Discussion

### Principal Findings

This study presents the first standardised HEIM analysis across the International Health Cohorts Consortium registry, examining 38,595 publications from 70 biobanks in 29 countries. Our principal findings are fourfold.

First, we observed substantial disparity in research intensity between HIC and LMIC biobanks (equity ratio: 57.8:1), indicating that biobank research capacity remains profoundly misaligned with global disease burden even within a standardised consortium framework.

Second, we identified 48 disease categories (27%) with critical or high-severity research gaps despite substantial global burden, with neglected tropical diseases (malaria, dengue, schistosomiasis) particularly underserved. Of 179 GBD categories, 160 are realistically addressable by biobank research.

Third, only UK Biobank achieved 'Strong' equity alignment (EAS 84.6), with a 29-point gap to the second-ranked biobank. This benchmark status suggests UK Biobank represents an aspirational target demonstrating what sustained investment can achieve, rather than a typical baseline.

Fourth, 80% of IHCC biobanks score 'Poor' on equity alignment, highlighting systemic underperformance across the global biobank landscape.

### Comparison with Framework Validation Study

Our previous analysis of 27 high-output biobanks (including eMERGE Network and deCODE Genetics, which are not IHCC members) found an equity ratio of 322:1. The lower ratio in this IHCC analysis (57.8:1) reflects the different sample composition rather than improved equity. The IHCC registry includes more LMIC biobanks and excludes the highest-output US consortium (eMERGE), producing a more representative but still inequitable picture.

Both analyses use identical HEIM methodology and reach the same fundamental conclusion: substantial disparity exists between HIC and LMIC biobank research capacity relative to disease burden.

### Implications for Precision Medicine

The research gaps identified here have direct implications for precision medicine. Conditions prevalent in LMICs but understudied in biobank research—particularly neglected tropical diseases like malaria (55.2 million DALYs, 17 publications), dengue (2.1 million DALYs, 0 publications), and schistosomiasis (1.8 million DALYs, 0 publications)—will have weaker evidence bases for genomic medicine applications.

Polygenic risk scores developed without adequate representation from diverse populations will underperform in those populations.^5-6^ Variant pathogenicity databases will contain systematic blind spots for understudied ancestries.^7-8^ The HEIM framework provides a mechanism for identifying and tracking these gaps, enabling targeted interventions.

### Why 80% of Biobanks Score 'Poor'

The concentration of biobanks in the 'Poor' EAS category reflects structural factors rather than individual institutional failures:

- **Resource constraints:** LMIC biobanks operate with a fraction of the funding available to HIC counterparts
- **Infrastructure gaps:** Genomic sequencing, computing infrastructure, and trained personnel remain scarce in many regions
- **Network effects:** Research collaborations, journal access, and funding opportunities cluster in established centres
- **Local versus global priorities:** Biobanks naturally focus on conditions prevalent in their source populations
- **Temporal factors:** Many LMIC biobanks are recently established; their scores may improve as programmes mature

Addressing these systemic factors requires coordinated action from funders, institutions, and policymakers. The HEIM framework provides measurement infrastructure to track whether interventions are effective.

### Limitations

Several limitations should inform interpretation of these findings:

- **Publication coverage:** Our analysis is limited to PubMed-indexed publications, potentially undercounting research published in regional journals or non-English languages
- **MeSH mapping accuracy:** Automated mapping from MeSH terms to GBD categories achieved 92% concordance with manual classification; some misclassification is expected
- **IHCC registry scope:** The 70 biobanks analysed represent IHCC members with publications; other biobanks worldwide are not captured
- **Temporal confounding:** Cumulative publication counts advantage established biobanks; recently established biobanks may appear to underperform despite rapid growth
- **Quality versus quantity:** Publication counts do not capture research quality, clinical impact, or translation to practice
- **Normative choices:** Using global rather than regional DALYs reflects a specific equity perspective; EAS weights (0.4/0.3/0.3) are based on expert judgment
- **UK Biobank influence:** With 35.7% of publications, UK Biobank substantially influences aggregate statistics; sensitivity analyses address this

### Policy Implications

The HEIM framework has several applications for policy and practice:

1. **Research funders** could incorporate equity metrics into grant evaluation criteria, weighting proposals that address high-gap diseases or involve LMIC biobanks
2. **Biobanks** could use the framework for strategic planning, identifying underserved disease areas within their capacity
3. **Policymakers** could reference equity metrics when allocating resources for genomic research infrastructure
4. **IHCC and similar consortia** could adopt HEIM metrics for member benchmarking and progress tracking

We emphasise that HEIM metrics are best viewed as a platform for piloting equity-aware mechanisms, to be refined as evidence accumulates on their impact.

---

## Conclusions

The HEIM framework provides a standardised, reproducible approach to quantifying equity in global biobank research. Our analysis of 70 IHCC-registered biobanks reveals substantial disparities in how research capacity is distributed relative to global disease burden, with HIC biobanks producing 57.8 times more research output per DALY than LMIC counterparts.

UK Biobank emerges as an exceptional performer, achieving the only 'Strong' equity alignment score and contributing over one-third of all publications. While this benchmark demonstrates what is achievable with sustained investment, the 29-point gap to second-ranked Nurses' Health Study indicates that UK Biobank-level performance may not be a realistic near-term target for most biobanks.

The identification of 48 diseases with critical or high research gaps—particularly neglected tropical diseases disproportionately affecting the Global South—provides actionable priorities for equity-focused investment. Closing the equity gap requires coordinated action across the research ecosystem: targeted funding for LMIC biobanks, capacity-building investments, equitable collaboration frameworks, and metrics that reward equity-aligned research.

The interactive HEIM-Biobank tool (https://manuelcorpas.github.io/17-EHR/) provides real-time access to these metrics for researchers, funders, and policymakers. Annual updates will enable stakeholders to monitor progress and identify emerging priorities.

---

## Declarations

### Author Contributions

MC conceived the study, developed the methodology, conducted the analyses, and wrote the manuscript.

### Funding

No specific funding has been dedicated for this work.

### Conflicts of Interest

The author declares no competing interests.

### Ethics Approval

This study analysed publicly available, aggregated publication data and disease burden statistics. No individual-level human subjects data were used. Institutional ethics approval was not required.

### Data Availability

All aggregated data, analysis code, and mapping dictionaries are available at the HEIM Equity Tool (https://manuelcorpas.github.io/17-EHR/). The tool displays this study's 70-biobank IHCC dataset. Original framework validation data (27 biobanks) are available in the v1.0 preprint (medRxiv DOI: 10.1101/2026.01.04.26343419).

The interactive dashboard provides access to:

- Complete biobank rankings with confidence intervals
- Disease-level gap scores and appropriateness classifications
- Regional and income-group comparisons
- UK Biobank sensitivity analyses
- Methodology documentation

Raw publication metadata are derived from PubMed and subject to NLM terms of use. Disease burden data are available from the IHME Global Health Data Exchange (https://ghdx.healthdata.org/).

### Acknowledgements

The author acknowledges the contributions of the International Health Cohorts Consortium for maintaining the Global Cohort Atlas and the Institute for Health Metrics and Evaluation for making disease burden data publicly available.

---

## References

1. Corpas M, Fatumo S, Rasheed H, Guio H, Fakhro K, Iacoangeli A. Bridging genomics' greatest challenge: the diversity gap. Nat Rev Genet 2025 (in press).

2. Sirugo G, Williams SM, Tishkoff SA. The missing diversity in human genetic studies. Cell 2019;177:26-31.

3. Popejoy AB, Fullerton SM. Genomics is failing on diversity. Nature 2016;538:161-164.

4. Ju Y, Jia T, Yang L, et al. Importance of including non-European populations in large human genetic studies to enhance precision medicine. Annu Rev Genomics Hum Genet 2022;23:187-207.

5. Duncan L, Shen H, Gelaye B, et al. Analysis of polygenic risk score usage and performance in diverse human populations. Nat Commun 2019;10:3328.

6. Wang Y, Guo J, Ni G, Yang J, Visscher PM, Yengo L. Theoretical and empirical quantification of the accuracy of polygenic scores in ancestry divergent populations. Nat Commun 2020;11:3865.

7. Manrai AK, Funke BH, Rehm HL, et al. Genetic misdiagnoses and the potential for health disparities. N Engl J Med 2016;375:655-665.

8. Landrum MJ, Lee JM, Benson M, et al. ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Res 2018;46:D1062-D1067.

9. World Health Organization. WHO Science Council report on the acceleration and equitable implementation of human genomics for global health. Geneva: WHO; 2024.

10. Sudlow C, Gallacher J, Allen N, et al. UK Biobank: An open access resource for identifying the causes of a wide range of complex diseases of middle and old age. PLoS Med 2015;12:e1001779.

11. Gaziano JM, Concato J, Brophy M, et al. Million Veteran Program: A mega-biobank to study genetic influences on health and disease. J Clin Epidemiol 2016;70:214-223.

12. International Health Cohorts Consortium. Global Cohort Atlas. https://ihccglobal.org/ (accessed January 7, 2026).

13. GBD 2021 Collaborators. Global burden of 369 diseases and injuries in 204 countries and territories, 1990-2021: a systematic analysis for the Global Burden of Disease Study 2021. Lancet 2022;401:1990-2034.

14. World Health Organization. Human genomics technologies in clinical studies—the research landscape: report on the 1990-2024 period. Geneva: WHO; 2025.

15. Fitipaldi H, McCarthy MI, Florez JC, Franks PW. Ethnic, gender and other sociodemographic biases in genome-wide association studies for the most prevalent human diseases: A systematic review. Hum Mol Genet 2023;32:520-532.

---

## Tables

### Table 1. Characteristics of IHCC-Registered Biobanks (Top 15 by EAS)

| Biobank | Country | Region | Income | Publications | Diseases | EAS | 95% CI |
|---------|---------|--------|--------|--------------|----------|-----|--------|
| UK Biobank | United Kingdom | EUR | HIC | 13,785 | 163 | 84.6 | 82.1-87.1 |
| Nurses' Health Study | United States | AMR | HIC | 3,197 | 142 | 55.2 | 53.0-57.4 |
| Women's Health Initiative | United States | AMR | HIC | 3,433 | 138 | 53.4 | 51.3-55.5 |
| Estonian Biobank | Estonia | EUR | HIC | 675 | 119 | 49.2 | 46.3-52.1 |
| All of Us | United States | AMR | HIC | 837 | 124 | 49.1 | 46.3-51.9 |
| HUNT Study | Norway | EUR | HIC | 1,218 | 116 | 48.1 | 45.4-50.8 |
| EPIC | Multi-country | EUR | HIC | 1,870 | 127 | 45.2 | 42.8-47.6 |
| Danish National Birth Cohort | Denmark | EUR | HIC | 636 | 98 | 44.4 | 41.5-47.3 |
| Genomics England | United Kingdom | EUR | HIC | 700 | 112 | 43.6 | 40.8-46.4 |
| MoBa | Norway | EUR | HIC | 962 | 104 | 43.1 | 40.2-46.0 |
| Biobank Japan | Japan | WPR | HIC | 492 | 108 | 42.7 | 39.3-46.1 |
| Nurses' Health Study II | United States | AMR | HIC | 1,124 | 118 | 42.5 | 39.8-45.2 |
| China Kadoorie Biobank | China | WPR | LMIC | 624 | 109 | 41.6 | 38.6-44.6 |
| Million Veteran Program | United States | AMR | HIC | 421 | 99 | 40.1 | 36.8-43.4 |
| 23andMe | United States | AMR | HIC | 387 | 94 | 39.9 | 36.4-43.4 |

*Full data for all 70 biobanks available at https://manuelcorpas.github.io/17-EHR/ and in Supplementary Table S1.*

### Table 2. Publication Distribution by WHO Region

| Region | Biobanks | Publications | Share (%) | Primary Type |
|--------|----------|--------------|-----------|--------------|
| Europe | 25 | 21,482 | 55.7 | HIC |
| Americas | 20 | 12,433 | 32.2 | Mixed |
| Western Pacific | 13 | 3,819 | 9.9 | Mixed |
| Eastern Mediterranean | 5 | 574 | 1.5 | LMIC |
| Africa | 6 | 277 | 0.7 | LMIC |
| South-East Asia | 1 | 10 | <0.1 | LMIC |
| **Total** | **70** | **38,595** | **100.0** | - |

### Table 3. Equity Alignment Score Distribution

| Category | Score Range | Number of Biobanks | Percentage |
|----------|-------------|-------------------|------------|
| Strong | ≥70 | 1 | 1.4% |
| Weak | 40-69 | 13 | 18.6% |
| Poor | <40 | 56 | 80.0% |
| **Total** | - | **70** | **100%** |

### Table 4. Critical Research Gaps in Biobank-Appropriate Diseases (Gap Score ≥90)

| Disease | DALYs (M) | Publications | Gap Score | Appropriateness |
|---------|-----------|--------------|-----------|-----------------|
| Malaria | 55.2 | 17 | 95 | High |
| Dengue | 2.1 | 0 | 95 | High |
| Schistosomiasis | 1.8 | 0 | 95 | High |
| Lymphatic filariasis | 1.3 | 0 | 95 | High |
| Cysticercosis | 1.2 | 0 | 95 | High |
| Scabies | 5.3 | 1 | 95 | Moderate |
| Typhoid and paratyphoid | 8.1 | 21 | 90 | High |

*Note: Diseases with 'Limited' appropriateness (drowning, animal contact, road injuries) are excluded from this table as they fall outside typical biobank research scope. See Supplementary Table S2 for complete gap analysis including all 179 diseases.*

---

## Supplementary Information

### Relationship to Previously Published Analysis

This manuscript presents HEIM v2.0, which analyses the standardised IHCC registry. A previous framework validation study (medRxiv DOI: 10.1101/2026.01.04.26343419, January 2026) analysed 27 high-output biobanks including eMERGE Network and deCODE Genetics. The key differences are:

| Aspect | v1.0 (Validation) | v2.0 (This Paper) |
|--------|-------------------|-------------------|
| Biobanks | 27 | 70 |
| Publications | 75,356 | 38,595 |
| Selection criteria | Highest output | IHCC membership |
| Includes eMERGE | Yes (47,755 pubs) | No |
| Includes deCODE | Yes (5,604 pubs) | No |
| Equity ratio | 322:1 | 57.8:1 |
| Purpose | Framework validation | Standardised registry analysis |

Both analyses use identical HEIM methodology and reach consistent conclusions about fundamental inequity in global biobank research.

### Supplementary Table S1

Complete data for all 70 IHCC biobanks (EAS, publications, diseases covered, confidence intervals) available at: https://manuelcorpas.github.io/17-EHR/

### Supplementary Table S2

Complete disease gap analysis for all 179 GBD categories, including appropriateness classifications, available at: https://manuelcorpas.github.io/17-EHR/

---

*Generated: January 2026*
*Version: HEIM-Biobank v2.0 (IHCC)*
*Interactive Tool: https://manuelcorpas.github.io/17-EHR/*
