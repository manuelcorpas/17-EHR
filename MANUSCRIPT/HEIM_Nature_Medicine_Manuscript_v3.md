# Three Dimensions of Neglect: How Biobanks, Clinical Trials, and Scientific Literature Systematically Exclude the Global South

---

## Title Page

**Article Type:** Analysis

**Title:** Three Dimensions of Neglect: How Biobanks, Clinical Trials, and Scientific Literature Systematically Exclude the Global South

**Short Title:** Three-Dimensional Research Equity Analysis

**Authors:**

Manuel Corpas¹²³*, Maxim Freydin⁴, Julio Valdivia-Silva⁵, Simeon Baker³⁶, Segun Fatumo⁷⁸, Hugo Guio⁹

**Affiliations:**

¹ School of Life Sciences, University of Westminster, London, United Kingdom
² The Alan Turing Institute, London, United Kingdom
³ GENEQ Global, London, United Kingdom
⁴ Department of Twin Research and Genetic Epidemiology, School of Life Course and Population Sciences, King's College London, London, United Kingdom
⁵ Universidad de Ingeniería y Tecnología (UTEC), Lima, Peru
⁶ University of Bath, Bath, United Kingdom
⁷ MRC/UVRI and LSHTM Uganda Research Unit, Entebbe, Uganda
⁸ London School of Hygiene & Tropical Medicine, London, United Kingdom
⁹ Instituto Nacional de Salud, Lima, Peru

**Corresponding Author:**

Manuel Corpas
Email: m.corpas@westminster.ac.uk
ORCID: 0000-0002-5765-9627

**Author ORCIDs:**
- Manuel Corpas: 0000-0002-5765-9627
- Maxim Freydin: 0000-0002-1439-6259
- Julio Valdivia-Silva: [To be added prior to submission]
- Simeon Baker: [To be added prior to submission]
- Segun Fatumo: 0000-0002-0529-6291
- Hugo Guio: [To be added prior to submission]

*Note: All author ORCIDs will be confirmed prior to final submission.*

**Word Count:** ~3,000 (main text excluding Methods)

**Figures:** 6 main figures + 5 Extended Data figures

**Tables:** 1 main table + 6 Supplementary tables

**References:** 50

**Keywords:** health equity, global health, neglected tropical diseases, biobanks, clinical trials, semantic analysis, machine learning, research policy

---

## Abstract

Diseases affecting 1.5 billion people in the Global South are systematically excluded from biomedical research infrastructure. Here we show that neglected tropical diseases exhibit 44% higher semantic isolation in the scientific literature than other conditions (P < 0.0001, Cohen's d = 1.80), are virtually absent from major biobanks (0 publications for lymphatic filariasis, dengue, and schistosomiasis), and face a 2.5-fold deficit in clinical trial sites compared to high-income countries. Using the Health Equity Informative Metrics (HEIM) framework, we analysed 70 international biobanks, 2.2 million clinical trials, and 13.1 million PubMed abstracts spanning 175 diseases. Only 1 of 70 biobanks achieves high equity alignment. The ten most neglected diseases are exclusively conditions of the Global South, experiencing compounding disadvantage across all three dimensions. These findings reveal that research inequity is not merely a funding problem but a structural feature of the biomedical enterprise, with quantifiable targets for intervention.

---

## Main Text

### Introduction

Lymphatic filariasis affects 120 million people worldwide, causing debilitating swelling that traps individuals in cycles of poverty and stigma. Yet across 70 major international biobanks, we found zero publications on this disease. Zero genomic studies. Zero translational research pipelines. This is not an isolated failure: it reflects a systematic pattern of exclusion embedded in the architecture of biomedical research itself (Fig. 1).

Low- and middle-income countries bear 93% of the world's preventable mortality, yet the biomedical research enterprise remains overwhelmingly oriented toward conditions prevalent in wealthy nations¹⁻⁴. This persistent structural exclusion, where the diseases of poverty remain invisible to the research infrastructure that could address them, has continued for decades despite sustained advocacy and targeted funding initiatives⁵⁻⁸. While health outcome inequities are extensively documented, the structural biases embedded within research infrastructure itself remain poorly characterised and inadequately measured. Critically, the HIV/AIDS experience demonstrates that sustained investment can fundamentally reverse structural marginalisation: a disease once semantically isolated from mainstream biomedicine is now deeply integrated into immunology, virology, and global health. This suggests that the patterns we identify are not inevitable but are amenable to strategic intervention.

Research inequity operates at multiple stages of the scientific enterprise. At the discovery stage, biobanks and cohort studies generate the foundational data for genomic medicine, yet these resources concentrate overwhelmingly in high-income countries and predominantly European-ancestry populations¹⁵⁻¹⁸. At the translation stage, clinical trials determine which interventions reach patients, yet trial sites cluster in wealthy nations regardless of where disease burden is greatest²³⁻²⁷. At the knowledge stage, the scientific literature shapes research priorities and enables cross-disciplinary learning, yet we lack systematic methods to assess whether certain diseases occupy marginalised positions in the semantic landscape of published science.

Previous efforts to quantify research equity have focused on single dimensions: publication counts relative to disease burden⁹, clinical trial registration by geography²⁵˒²⁹, or funding allocation across disease categories³⁰⁻³³. While valuable, these approaches cannot capture compounding disadvantage, the phenomenon whereby diseases neglected at one stage face systematic barriers at subsequent stages. A disease absent from major biobanks generates fewer genomic discoveries, attracts fewer clinical trials, accumulates a smaller and more isolated body of literature, and consequently remains neglected in perpetuity. Breaking this cycle requires a unified framework that spans the full research lifecycle and identifies where interventions are most needed.

Here we introduce the Health Equity Informative Metrics (HEIM) framework (Fig. 1), a three-dimensional analysis of research equity encompassing Discovery (biobank research output), Translation (clinical trial geography), and Knowledge (semantic structure of scientific literature). We analysed 70 biobanks from the International Health Cohorts Consortium across 29 countries, 2.2 million clinical trials with 770,000 geolocated site records, and 13.1 million PubMed abstracts spanning 25 years (2000–2025). Using the Global Burden of Disease 2021 taxonomy as a common ontology, we mapped research activity across 175 disease categories and developed novel metrics (including Semantic Isolation Index, Knowledge Transfer Potential, and Research Clustering Coefficient) to quantify patterns invisible to conventional bibliometric analysis.

Our analysis addresses four questions fundamental to global health equity: (1) How equitably is biobank-derived research distributed relative to global disease burden? (2) Does clinical trial infrastructure reflect the geography of disease prevalence? (3) Do neglected diseases occupy isolated positions in the semantic landscape of biomedical research? (4) Which diseases show compounding disadvantage across all three dimensions, and what characterises them? The answers reveal systematic structural barriers that perpetuate health inequity, and provide quantitative targets for research investment to address them.

### Results

#### The Discovery Dimension: Biobank Research Shows Stark Equity Gaps

Only 1 of 70 major international biobanks achieves high equity alignment with global disease burden. This stark finding emerged from our analysis of 70 biobanks registered with the International Health Cohorts Consortium (IHCC)⁴⁸, spanning 29 countries across six WHO regions (Fig. 2a). These cohorts have collectively generated 38,595 disease-specific publications indexed in PubMed. To quantify alignment between research output and global disease burden, we developed the Equity Alignment Score (EAS), which penalises biobanks for critical gaps in high-burden diseases while accounting for research capacity (see Methods).

The distribution of equity alignment was heavily skewed toward poor performance. Only UK Biobank¹⁹˒²⁰ achieved "High" equity alignment (EAS = 84.6), defined as EAS ≥ 60 (Fig. 2b). Thirteen biobanks showed "Moderate" alignment (EAS 40–60), including Nurses' Health Study (55.2), Women's Health Initiative (53.4), and Estonian Biobank (49.2). The remaining 56 biobanks (80%) demonstrated "Low" alignment (EAS < 40), indicating substantial misalignment between their research portfolios and global health priorities.

Geographic concentration was pronounced. European and North American biobanks accounted for 45 of 70 cohorts (64%) and 33,915 of 38,595 publications (87.9%). The six African biobanks collectively produced 277 publications (0.7%), despite Africa bearing 24% of the global disease burden: a 34-fold disparity between research output and disease burden. High-income country biobanks generated 93.5% of all publications, while biobanks in Global South countries contributed just 6.5% (Fig. 2c).

We identified 22 diseases with "Critical" research gaps (Gap Score ≥ 90), a threshold corresponding to fewer than five publications across all 70 biobanks despite substantial disease burden. These included malaria (17 publications across all biobanks, despite 55.2 million disability-adjusted life years [DALYs] globally), dengue (0 publications, 2.1 million DALYs), schistosomiasis (0 publications, 1.8 million DALYs), and lymphatic filariasis (0 publications, 1.3 million DALYs). An additional 26 diseases showed "High" gaps (Gap Score 70–89), and 47 showed "Moderate" gaps (Gap Score 50–69). Only 84 of 175 analysable diseases (48%) had "Low" gaps indicating adequate coverage relative to burden.

Having established these gaps in discovery-stage research, we next examined whether similar patterns emerge in clinical translation.

#### The Translation Dimension: Clinical Trials Concentrate in High-Income Settings

Analysis of 2.2 million clinical trials registered on ClinicalTrials.gov (2000–2025), with 770,178 geolocated trial site records, revealed systematic geographic concentration (Fig. 3a). Trial sites in high-income countries numbered 552,952 (71.8%), compared to 217,226 (28.2%) in low- and middle-income countries, representing a 2.5-fold disparity.

The United States alone hosted 192,501 trial sites (25.0% of the global total), followed by China (48,028; 6.2%), France (41,913; 5.4%), and Canada (32,892; 4.3%). The ten countries with the most trial sites were predominantly high-income, with China and Turkey as the only middle-income exceptions in the top fifteen. China's position reflects its status as an emerging global research power with rapidly expanding clinical trial infrastructure, though this capacity has not translated into proportional representation for neglected diseases affecting other LMIC regions.

We examined 89 disease categories with sufficient trial data for geographic analysis. For the 30 diseases predominantly affecting the Global South—identified through GBD burden distributions—we found a substantial intensity gap. Global South countries hosted 20.3% of clinical trials for these conditions but bear 38.0% of the associated disability-adjusted life years (DALYs), yielding a trial intensity ratio of 485.9 trials per million DALYs compared to 1,167.4 in high-income countries (intensity gap = 2.4-fold; Fig. 3b).

Neglected tropical diseases (NTDs) showed the most severe underrepresentation¹⁰⁻¹⁴. For lymphatic filariasis (1.3 million DALYs), only 89 trials were registered globally—compared to 47,892 for type 2 diabetes despite similar global burden (2.5 million DALYs), a 538-fold disparity. What limited NTD research exists does occur where disease is prevalent (71% of lymphatic filariasis sites in endemic countries), but absolute volumes remain critically insufficient. Schistosomiasis (1.8 million DALYs) had 156 trials; cysticercosis (1.2 million DALYs) had 47 trials. By contrast, ischemic heart disease (9.1 million DALYs) had 38,441 trials—representing approximately 4,200 trials per million DALYs compared to just 68 for lymphatic filariasis.

Beyond research volume and geography, we asked whether neglected diseases also occupy marginalised positions in the structure of scientific knowledge itself.

#### The Knowledge Dimension: Neglected Diseases Occupy Isolated Semantic Space

To assess structural marginalisation in the research literature beyond simple publication counts³⁹⁻⁴², we generated semantic embeddings for 13.1 million PubMed abstracts spanning 175 GBD disease categories (2000–2025). Using PubMedBERT, we represented each abstract as a 768-dimensional vector capturing its semantic content, then computed disease-level centroids and inter-disease similarity matrices (see Methods).

We developed three novel metrics to characterise the semantic landscape. The Semantic Isolation Index (SII) measures the distance between a disease's research centroid and the centroid of all biomedical literature—higher values indicate greater isolation from mainstream research discourse. In practical terms, a semantically isolated disease cannot easily benefit from methodological advances in adjacent fields; when immunology develops new therapeutic paradigms, isolated diseases remain disconnected from these innovations. The Knowledge Transfer Potential (KTP) quantifies the average semantic similarity between a disease and all other diseases, reflecting opportunities for cross-disciplinary knowledge flow. The Research Clustering Coefficient (RCC) measures the internal cohesion of a disease's research community relative to its connections with other fields.

Neglected tropical diseases exhibited significantly higher semantic isolation than non-NTD conditions. The mean SII for 15 NTDs was 0.00211 compared to 0.00146 for other diseases, a 44% elevation that was highly statistically significant (Welch's t-test, P < 0.0001) with a very large effect size (Cohen's d = 1.80, exceeding the threshold typically considered "large" by more than twofold; Fig. 4a). This isolation was visually apparent in UMAP projections of the disease semantic space, where NTDs formed a distinct cluster separated from cardiovascular, oncological, and neurological research (Fig. 4b).

The five diseases with highest semantic isolation were African trypanosomiasis (SII = 0.00265), lymphatic filariasis (SII = 0.00237), Guinea worm disease (SII = 0.00229), cysticercosis (SII = 0.00203), and onchocerciasis (SII = 0.00198). All five are conditions predominantly affecting low-income populations. These diseases showed low Knowledge Transfer Potential scores, indicating limited semantic bridges to other research areas that might facilitate methodological or therapeutic cross-fertilisation.

The semantic structure analysis (Fig. 5) revealed systematic patterns across disease categories. NTDs exhibited higher isolation across all metrics (Fig. 5a), with the top 20 most isolated diseases dominated by Global South conditions (Fig. 5b). Critically, isolation was not simply a function of research volume: diseases with fewer publications were not uniformly isolated, suggesting that semantic positioning reflects structural factors beyond publication count (Fig. 5c). The NTD versus non-NTD comparison demonstrated clear distributional separation (Fig. 5d). Notably, HIV/AIDS showed moderate semantic integration (SII = 0.00142) despite its origins as a disease of poverty, a pattern we examine further in the Discussion.

Semantic isolation correlated significantly with Discovery dimension Gap Scores (Pearson's r = 0.67, P < 0.001; Extended Data Fig. 4), suggesting that diseases neglected in biobank research also occupy marginalised positions in the broader scientific literature. This correlation held after controlling for total publication volume (partial r = 0.58, P < 0.001), indicating that isolation reflects qualitative positioning, not merely quantity.

#### HIV/AIDS: Empirical Evidence That Semantic Isolation Can Be Reversed

HIV/AIDS provides a natural experiment demonstrating that sustained investment can reverse semantic isolation. We computed SII values for HIV/AIDS across five-year windows from 2000 to 2025. In 2000–2004, HIV/AIDS had an SII of 0.00187, placing it among the more isolated conditions. By 2020–2025, this had decreased to 0.00142, a 24% reduction in semantic isolation. This trajectory contrasts sharply with NTDs, which showed no significant change in SII over the same period (mean change: +2.1%, P = 0.71).

The semantic integration of HIV/AIDS was not uniform across knowledge domains. HIV research showed the strongest connectivity gains with immunology (KTP increase: 0.034), clinical trial methodology (0.028), and global health systems research (0.025). These are precisely the domains where PEPFAR, the Global Fund, and other coordinated investments concentrated capacity-building efforts. By contrast, HIV's connectivity with neglected disease research remained low (KTP: 0.011), suggesting that integration benefits did not diffuse to other diseases of poverty. This pattern indicates that strategic investment can reduce isolation for targeted diseases but does not automatically benefit adjacent neglected conditions.

#### Unified Neglect Score: Identifying Compounding Disadvantage

To identify diseases experiencing systematic neglect across all dimensions, we computed a Unified Neglect Score integrating Discovery (Gap Score), Translation (clinical trial equity), and Knowledge (Semantic Isolation Index) with equal weighting (33%/33%/34%; see Methods for normalisation). The score ranges from 0 (no neglect) to ~50 (maximum neglect across available dimensions).

Among 175 diseases with sufficient data, the mean Unified Score was 18.7 (SD = 13.3), with a range from 0.1 (ischemic heart disease) to 46.9 (lymphatic filariasis). The distribution was right-skewed, with a long tail of highly neglected conditions (Fig. 6a).

**Table 1. Top Ten Most Neglected Diseases by Unified Score**

| Rank | Disease | WHO NTD | Unified Score | Gap Score | SII | WHO Region(s) | Primary Burden Region |
|------|---------|---------|---------------|-----------|-----|---------------|----------------------|
| 1 | Lymphatic filariasis | Yes | 46.9 | 95 | 0.00237 | AFR, SEAR | Sub-Saharan Africa, South Asia |
| 2 | Guinea worm disease | Yes | 46.9 | 95 | 0.00229 | AFR | Sub-Saharan Africa |
| 3 | Cysticercosis | Yes | 46.9 | 95 | 0.00203 | AMR, AFR | Latin America, Sub-Saharan Africa |
| 4 | Dengue | Yes | 46.9 | 95 | 0.00188 | SEAR, AMR | Southeast Asia, Latin America |
| 5 | Scabies | Yes | 46.9 | 95 | 0.00187 | Multiple | Global South |
| 6 | Malaria | — | 46.9 | 95 | 0.00176 | AFR | Sub-Saharan Africa |
| 7 | Schistosomiasis | Yes | 46.9 | 95 | 0.00172 | AFR | Sub-Saharan Africa |
| 8 | Ascariasis | Yes | 46.8 | 95 | 0.00164 | SEAR, AFR | South Asia, Sub-Saharan Africa |
| 9 | Rabies | Yes | 46.8 | 95 | 0.00168 | SEAR, AFR | South Asia, Sub-Saharan Africa |
| 10 | Yellow fever | Yes | 46.7 | 90 | 0.00189 | AFR, AMR | Sub-Saharan Africa, South America |

*Note: WHO NTD = World Health Organization Neglected Tropical Disease classification. WHO Regions: AFR = African Region, SEAR = South-East Asia Region, AMR = Region of the Americas. Nine of ten are WHO-classified NTDs; malaria is addressed separately by the WHO Global Malaria Programme. Rankings reflect Unified Score integrating all dimensions; individual SII rankings differ slightly (see text). African trypanosomiasis has the highest SII (0.00265) but lower Gap Score places it outside the top 10. The clustering of top scores at ~46.9 reflects that these diseases lack sufficient clinical trial data for the Translation dimension; their Unified Score is computed from Discovery and Knowledge dimensions only (see Methods), creating a ceiling effect when both dimensions indicate maximum neglect.*

All ten diseases in Table 1 are infectious diseases with primary burden in the Global South. Nine are classified as neglected tropical diseases by the WHO. None had substantial clinical trial representation (Translation dimension data unavailable due to low trial counts). This convergence across independent dimensions—Discovery, Translation, and Knowledge—demonstrates cumulative neglect: these diseases are simultaneously absent from major biobanks, underrepresented in clinical trials, and semantically isolated in the research literature.

By contrast, the ten diseases with lowest Unified Scores (greatest research equity) included ischemic heart disease (0.1), breast cancer (0.8), type 2 diabetes (1.2), lung cancer (1.4), and Alzheimer's disease (2.1). These are all conditions with substantial burden in high-income countries and correspondingly robust research ecosystems spanning biobanks, trials, and interconnected literature.

#### Temporal Trends: The Gap Is Not Closing

We analysed temporal evolution of research equity across the 25-year study period (2000–2025) using five-year rolling windows. Three findings emerged (Extended Data Fig. 3).

First, biobank research has become more concentrated rather than less. UK Biobank's share of total biobank publications rose from 12% (2000–2009) to 36% (2015–2025), reflecting its scale and data accessibility. Global South biobanks' collective share remained static at approximately 6–7% throughout the period.

Second, clinical trial site distribution showed no significant trend toward geographic equity. The HIC:LMIC site ratio fluctuated between 2.3 and 2.7 across five-year windows without directional improvement (trend test P = 0.41).

Third, semantic isolation of NTDs has not decreased. We computed temporal drift—the mean change in disease centroid position across consecutive time windows—as a measure of research evolution. NTDs showed lower mean temporal drift (0.00012) than non-NTDs (0.00019). While low drift could theoretically indicate a mature, stable research field, the combination of low drift with high isolation suggests stagnation rather than maturity: NTD research is not accumulating new knowledge that might connect it to broader methodological developments in adjacent fields.

Collectively, these trends demonstrate a sobering conclusion: despite three decades of global health equity initiatives, the structural position of neglected diseases in the research landscape has not materially improved. The gap identified by HEIM is not a historical artefact but an ongoing feature of contemporary biomedical research. Incremental improvements in individual programmes have not translated into systemic change.

### Discussion

Our analysis reveals that health research inequity is not merely a matter of funding disparities but reflects deep structural biases embedded across the entire research enterprise. The HEIM framework demonstrates that neglected diseases—particularly those affecting the Global South—face compounding disadvantage: they are absent from major biobanks, underrepresented in clinical trials, and semantically isolated in the scientific literature. This triple burden creates self-reinforcing cycles that perpetuate neglect regardless of individual funding initiatives. The ten most neglected diseases in our analysis collectively affect over 1.5 billion people, predominantly in sub-Saharan Africa and South Asia.

The finding that neglected tropical diseases exhibit 44% higher semantic isolation than other conditions has important implications. Semantic isolation indicates not only fewer publications but qualitatively different positioning in the knowledge landscape. These diseases are disconnected from the methodological innovations, therapeutic paradigms, and conceptual frameworks that drive progress in mainstream biomedicine. A disease that is semantically isolated cannot easily benefit from advances in adjacent fields, cannot attract researchers trained in well-resourced areas, and cannot leverage the infrastructure of interconnected research communities.

The contrast with HIV/AIDS is instructive⁴⁹˒⁵⁰. Sustained global investment over four decades has not only increased publication volume but fundamentally integrated HIV research into immunology, virology, global health, and implementation science, reducing its semantic isolation despite its origins as a disease of poverty. This transformation occurred through specific mechanisms: dedicated funding streams (PEPFAR, Global Fund) that required collaboration with established research institutions; training pipelines that built capacity in endemic regions while connecting researchers to mainstream methodological communities; and deliberate efforts to publish in high-impact journals that reach broad audiences. The result is that HIV/AIDS research today is semantically connected to cancer immunotherapy, vaccine development, and health systems research in ways that NTD research is not. This demonstrates that structural marginalization is not inevitable—it can be reversed through strategic, sustained investment that deliberately connects neglected disease research to established methodological communities.

#### Implications for Research Funders

HEIM provides quantitative targets that can guide strategic investment. Funders seeking to maximize equity impact should prioritize diseases with high Unified Scores, particularly those where investment might reduce semantic isolation by connecting neglected disease research to established methodological communities. Our data suggest that the current portfolio of major funders—reflected in biobank output and trial registration—remains substantially misaligned with global disease burden. The persistence of this gap over 25 years, despite explicit equity mandates from organizations including the Wellcome Trust, Gates Foundation, and National Institutes of Health, indicates that incremental adjustments are insufficient. Structural change may require dedicated funding streams for neglected diseases with mandated cross-disciplinary collaboration to reduce semantic isolation.

#### Implications for Biobank Consortia

The concentration of research output in a single biobank—UK Biobank accounts for 36% of recent publications—creates both opportunity and risk. On one hand, UK Biobank demonstrates that well-resourced, accessible infrastructure generates substantial scientific return. On the other hand, this concentration means that diseases not represented in UK Biobank's primarily European, primarily healthy-at-recruitment population remain systematically understudied. The IHCC and similar consortia should consider explicit equity metrics when evaluating member contributions, potentially weighting research on high-burden, high-gap diseases more heavily than incremental contributions to well-studied conditions. Our Equity Alignment Score provides a ready framework for such evaluation.

#### Implications for Clinical Trial Design

The 2.5-fold concentration of trial sites in high-income countries persists despite decades of recognition that trials should be conducted in populations who will use the resulting interventions. Regulatory frameworks, institutional capacity, and investigator networks all contribute to this concentration. Our data suggest that even for diseases predominantly affecting the Global South, the majority of trial infrastructure remains in the Global North. This mismatch raises questions about generalizability, implementation feasibility, and research justice. Trial sponsors and regulators should consider geographic representation as a design criterion, not merely an aspiration.

#### Actionable Policy Targets

Based on our findings, we propose three measurable targets for structural change:

1. **Funders**: Allocate a minimum of 15% of global health research portfolios to diseases with Unified Neglect Scores above 40, with explicit requirements for cross-disciplinary collaboration to reduce semantic isolation.

2. **Biobank Consortia**: Adopt standardized Equity Alignment Score reporting as a condition of consortium membership, with a target of achieving EAS ≥ 40 (Moderate alignment) for all member biobanks within five years.

3. **Clinical Trial Sponsors**: Require that trials for diseases with primary burden in the Global South include at least 50% of sites in endemic regions, moving from the current 29% toward proportional representation.

These targets are informed by HEIM metrics and can be monitored through the interactive dashboard, enabling accountability and progress tracking.

#### Limitations

Several limitations warrant consideration. First, IHCC membership is voluntary, and non-member biobanks may differ systematically in their disease coverage; our findings reflect the largest coordinated biobank network but not all global biobank activity. Second, PubMed coverage favours English-language journals and may underrepresent research published in regional journals or non-English languages; this bias likely underestimates research activity in the Global South, meaning our findings may be conservative. Third, clinical trial registration completeness varies by country and has improved over time, particularly following the ICMJE requirement (2005), potentially affecting temporal trend estimates; pre-2005 trials may be underrepresented. Fourth, our biobank analysis relies on publication counts, which may underestimate research activity that does not result in indexed publications, including internal reports and grey literature. Fifth, the Unified Score weights dimensions equally; alternative weightings could alter rankings, though sensitivity analyses (Extended Data Fig. 2) suggest our conclusions are robust (Spearman's ρ > 0.92 across schemes). Finally, our analysis establishes associations between disease characteristics and research neglect but cannot establish causation; the observed patterns may reflect multiple interacting factors including historical funding decisions, institutional capacity, and disease characteristics.

#### Future Directions

HEIM provides a foundation for prospective monitoring of research equity. Integration with funding databases (NIH Reporter, Dimensions, OpenAlex) would enable analysis of investment-to-output relationships. Expansion to preprints and grey literature would capture research currently invisible to PubMed-based analyses. Development of intervention studies—testing whether targeted investments reduce semantic isolation—would move from diagnosis to demonstrated solutions. The interactive HEIM dashboard (https://manuelcorpas.github.io/17-EHR/) enables real-time monitoring of research equity, transforming abstract commitments into measurable accountability.

---

## Methods

### Study Design and Overview

We developed the Health Equity Informative Metrics (HEIM) framework to quantify research equity across three dimensions of the biomedical research lifecycle: Discovery (biobank-derived research), Translation (clinical trials), and Knowledge (semantic structure of scientific literature). We used the Global Burden of Disease (GBD) 2021 taxonomy as a common ontology to enable cross-dimensional comparison. The study period spanned January 2000 to December 2025.

### Data Sources

**Global Burden of Disease 2021.** We obtained disease burden estimates from the Institute for Health Metrics and Evaluation (IHME) GBD 2021 study, including disability-adjusted life years (DALYs), deaths, and prevalence for 179 Level 3 disease categories across 204 countries and territories. Burden estimates were aggregated by World Bank income classification (high-income countries, HIC; low- and middle-income countries, LMIC) and WHO region.

**International Health Cohorts Consortium (IHCC).** We analysed 70 biobanks registered with the IHCC as of January 2026, representing 29 countries across six WHO regions. For each biobank, we retrieved disease-specific publications from PubMed using structured queries combining biobank name variants with disease-specific Medical Subject Headings (MeSH) terms. The retrieval period was January 2000 to December 2025.

**Aggregate Analysis of ClinicalTrials.gov (AACT).** Clinical trial data were obtained from the AACT database, a PostgreSQL mirror of ClinicalTrials.gov maintained by the Clinical Trials Transformation Initiative. We extracted 2,189,930 registered trials with study start dates between January 2000 and December 2025, along with 770,178 facility records containing geographic coordinates. Trial-disease mapping used condition terms matched to GBD categories via MeSH cross-references.

**PubMed/MEDLINE.** We retrieved abstracts for 175 GBD disease categories from PubMed using the Entrez E-utilities API. Queries employed MeSH Major Topics [Majr] to ensure retrieved articles had the disease as a primary focus, combined with title/abstract keyword searches for diseases lacking adequate MeSH coverage. The final corpus comprised 13,100,113 unique abstracts.

### GBD-to-MeSH Disease Mapping

We developed a systematic mapping between 179 GBD Level 3 disease categories and corresponding MeSH terms. For each GBD category, we identified: (1) exact MeSH matches where available; (2) combinations of related MeSH terms for composite categories; and (3) title/abstract keyword supplements for conditions with incomplete MeSH coverage. Two investigators (MC, JVS) independently curated mappings, achieving 94.2% initial agreement (Cohen's κ = 0.91, indicating excellent reliability); discrepancies were resolved by consensus. The complete mapping is provided in Supplementary Table S1. Four GBD categories representing methodological meta-categories rather than specific diseases were excluded, yielding 175 analyzable disease categories.

### Discovery Dimension Metrics

**Gap Score.** For each biobank-disease pair, we computed a Gap Score reflecting research deficit relative to disease burden:

$$\text{Gap}_d = \begin{cases} 95 & \text{if } P_d = 0 \\ \tau_c & \text{if disease } d \in \text{category } c \text{ and } P_d < \theta_c \\ 100 - 100 \times \frac{P_d / B_d}{\max_j(P_j / B_j)} & \text{otherwise} \end{cases}$$

where $P_d$ is the publication count for disease $d$, $B_d$ is the disease burden, and $\tau_c$ and $\theta_c$ are category-specific thresholds for infectious and neglected diseases. Gap Scores range from 0 (no gap) to 95 (complete absence of research).

**Equity Alignment Score (EAS).** For each biobank, we computed an overall equity alignment score:

$$\text{EAS}_b = 100 - (0.4 \times S_b + 0.3 \times M_b + 0.3 \times C_b)$$

where $S_b$ represents gap severity (weighted mean of Gap Scores), $M_b$ represents burden missed (total DALYs for diseases with ≤2 publications), and $C_b$ represents capacity penalty (function of critical and high gaps relative to total output). EAS ranges from 0 to 100, with higher scores indicating better alignment. We categorized biobanks as High (EAS ≥ 60), Moderate (40 ≤ EAS < 60), or Low (EAS < 40) equity alignment.

### Translation Dimension Metrics

**Geographic Distribution.** Trial sites were geocoded using facility location data from AACT and classified by country income level (World Bank 2024 classifications) and WHO region. We computed the HIC:LMIC site ratio as the number of trial sites in high-income countries divided by sites in low- and middle-income countries.

**Trial Intensity.** For disease categories with ≥50 registered trials, we computed trial intensity as the ratio of trial count to disease DALYs (trials per million DALYs). The intensity gap was defined as the ratio of HIC trial intensity to Global South trial intensity for diseases with primary burden in low-income regions.

### Knowledge Dimension Metrics

**Semantic Embeddings.** We generated dense vector representations for each PubMed abstract using PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, version 1.1, commit hash: `bf5939a`, released October 2022)³⁴, a transformer model pre-trained on PubMed abstracts and full-text articles³⁵⁻³⁷. Each abstract was encoded as a 768-dimensional vector. Embeddings were computed using PyTorch 2.0 with batch processing (batch size = 256) on consumer hardware (Apple M2 Max, 64GB RAM) using Metal Performance Shaders (MPS) acceleration; total computation time was approximately 72 hours for 13.1 million abstracts. The pipeline is designed for reproducibility on standard workstations without requiring specialized GPU clusters. For each disease, we computed the centroid (mean embedding) across all associated abstracts.

**Semantic Isolation Index (SII).** SII measures the Euclidean distance between a disease's centroid and the global centroid of all biomedical literature:

$$\text{SII}_d = \frac{\|\mathbf{c}_d - \mathbf{c}_{\text{global}}\|_2}{\max_j \|\mathbf{c}_j - \mathbf{c}_{\text{global}}\|_2}$$

where $\mathbf{c}_d$ is the centroid for disease $d$ and $\mathbf{c}_{\text{global}}$ is the centroid of all 13.1 million abstracts. Higher SII values indicate greater semantic isolation from mainstream biomedical research.

**Knowledge Transfer Potential (KTP).** KTP quantifies the average semantic similarity between a disease and all other diseases:

$$\text{KTP}_d = 1 - \frac{1}{N-1} \sum_{j \neq d} \left(1 - \frac{\mathbf{c}_d \cdot \mathbf{c}_j}{\|\mathbf{c}_d\| \|\mathbf{c}_j\|}\right)$$

Higher KTP values indicate greater potential for cross-disease knowledge transfer.

**Research Clustering Coefficient (RCC).** RCC measures the internal cohesion of a disease's research community:

$$\text{RCC}_d = \frac{\bar{s}_{\text{within},d}}{\bar{s}_{\text{across}}}$$

where $\bar{s}_{\text{within},d}$ is the mean pairwise cosine similarity among abstracts for disease $d$, and $\bar{s}_{\text{across}}$ is the mean similarity across all disease pairs. Higher RCC values indicate more tightly clustered internal research communities.

**Temporal Drift.** We computed embeddings separately for five-year windows (2000–2004, 2005–2009, 2010–2014, 2015–2019, 2020–2025) and measured centroid displacement between consecutive windows:

$$\text{Drift}_d = \frac{1}{T-1} \sum_{t=2}^{T} \|\mathbf{c}_{d,t} - \mathbf{c}_{d,t-1}\|_2$$

where $T$ is the number of time windows. Higher temporal drift indicates greater semantic evolution over time.

### Unified Neglect Score

We integrated the three dimensions into a single Unified Neglect Score:

$$U_d = \sum_{i=1}^{3} w_i \times \frac{x_{d,i} - \min(x_i)}{\max(x_i) - \min(x_i)} \times 100$$

where $w_1 = 0.33$ (Discovery), $w_2 = 0.33$ (Translation), $w_3 = 0.34$ (Knowledge), and $x_{d,i}$ is the raw score for disease $d$ on dimension $i$. Equal weighting was chosen because (1) no theoretical framework exists to privilege one dimension over others, (2) this approach ensures transparent, reproducible scoring, and (3) the interactive dashboard allows users to apply custom weights. For 86 diseases lacking sufficient clinical trial data (Translation dimension), the Unified Score was computed as $U_d = 0.5 \times \text{norm}(\text{Gap}) + 0.5 \times \text{norm}(\text{SII})$, preserving equal weighting across available dimensions. Sensitivity analyses examined alternative weighting schemes (Extended Data Fig. 2) and confirmed that NTD rankings were stable (Spearman's ρ > 0.92 across all schemes).

### Statistical Analysis

Group comparisons between NTDs and non-NTDs used Welch's t-test with Cohen's d effect size⁴⁶. Correlations were assessed using Pearson's r with 95% confidence intervals computed via Fisher z-transformation. Multiple comparisons across disease categories were corrected using the Benjamini-Hochberg procedure⁴⁵ with false discovery rate (FDR) < 0.05. Temporal trends were assessed using linear regression with year as predictor. All analyses were conducted in Python 3.11 using pandas, numpy, scipy, and statsmodels.

### Visualisation

Dimensionality reduction for disease semantic space visualisation used Uniform Manifold Approximation and Projection (UMAP)⁴³˒⁴⁴ with n_neighbors=15, min_dist=0.1, and random_state=42 for reproducibility. Hierarchical clustering for heatmaps used Ward's method with Euclidean distance. All figures were generated using matplotlib and seaborn.

### Code Availability

All analysis code is publicly available at https://github.com/manuelcorpas/17-EHR (archived at Zenodo: DOI 10.5281/zenodo.XXXXXXX [to be assigned upon acceptance]). The repository includes Python scripts for data retrieval, embedding generation, metric computation, and figure generation. The interactive HEIM dashboard is available at https://manuelcorpas.github.io/17-EHR/.

### Data Availability

Processed datasets generated during this study are available as follows:
- **Disease metrics** (Unified Scores, Gap Scores, SII values for 175 diseases): deposited at Zenodo (DOI to be assigned upon acceptance)
- **Biobank equity scores** (EAS for 70 IHCC biobanks): deposited at Zenodo
- **GBD-to-MeSH mappings** (Supplementary Table S1): included in Supplementary Information

Raw data are available from original sources under their respective terms:
- GBD 2021 burden estimates: https://vizhub.healthdata.org/gbd-results/ (IHME)
- Clinical trial data: https://aact.ctti-clinicaltrials.org/ (AACT/CTTI)
- PubMed abstracts: https://pubmed.ncbi.nlm.nih.gov/ (NLM/NCBI)
- PubMedBERT model: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

### Ethics Statement

This study used only publicly available, de-identified aggregate data and did not require ethics approval.

---

## Figure Legends

**Figure 1. The HEIM Framework.**
Schematic overview of the Health Equity Informative Metrics framework. (a) Three dimensions of research equity: Discovery (biobank research), Translation (clinical trials), and Knowledge (semantic structure of literature). (b) Data sources and scale for each dimension. (c) Integration into Unified Neglect Score with equal weighting across dimensions.

**Figure 2. Discovery Dimension: Global Biobank Equity Landscape.**
(a) World map showing locations of 70 IHCC biobanks, coloured by Equity Alignment Score category (High, green; Moderate, yellow; Low, red). Bubble size indicates total publications. (b) Distribution of Equity Alignment Scores across biobanks, with category thresholds indicated. Only UK Biobank (EAS = 84.6) achieves High alignment. (c) Publication share by country income group and WHO region, showing 93.5% concentration in high-income countries.

**Figure 3. Translation Dimension: Clinical Trial Geographic Distribution.**
(a) Global heatmap of 770,178 clinical trial sites, showing concentration in North America and Europe. (b) Trial intensity comparison between high-income countries and Global South for diseases with primary burden in low-income regions, demonstrating 2.4-fold intensity gap. (c) Top 15 countries by trial site count, with income classification indicated.

**Figure 4. Knowledge Dimension: Semantic Isolation of Disease Research.**
(a) Box plot comparing Semantic Isolation Index between neglected tropical diseases (n = 15) and other conditions (n = 160). NTDs show 44% higher isolation (P < 0.0001, Cohen's d = 1.80). (b) UMAP projection of 175 diseases in semantic space. Key NTDs labelled: lymphatic filariasis, African trypanosomiasis, schistosomiasis, dengue, onchocerciasis. Well-studied comparators labelled: breast cancer, ischemic heart disease, type 2 diabetes, Alzheimer's disease. Disease categories are colour-coded: cardiovascular (blue), oncology (purple), neurological (green), infectious/NTD (red), other (gray). Note the spatial separation between the NTD cluster (upper left) and well-connected diseases (centre).

**Figure 5. Semantic Structure Analysis of Disease Research.**
(a) Distribution of Semantic Isolation Index by disease category, showing NTDs as statistical outliers. (b) Top 20 most semantically isolated diseases, with NTDs highlighted. (c) Relationship between semantic isolation and research volume (log scale), demonstrating that isolation is not merely a function of publication count. (d) Direct comparison of NTD versus non-NTD distributions with statistical summary. Reading guide: Higher SII values indicate greater distance from mainstream biomedical research discourse. Diseases in the upper left of panel (c) are most neglected: few publications AND semantically disconnected from methodological advances in adjacent fields.

**Figure 6. Unified Neglect Score and Compounding Disadvantage.**
(a) Distribution of Unified Neglect Scores across 175 diseases, showing right-skewed distribution with long tail of highly neglected conditions. (b) Top 30 most neglected diseases ranked by Unified Score, with dimension contributions shown as stacked bars. All top 10 diseases are infectious conditions of the Global South, predominantly WHO-classified NTDs.

---

## Extended Data Figure Legends

**Extended Data Figure 1. GBD-to-MeSH Mapping Coverage.**
Sankey diagram showing mapping completeness between 179 GBD Level 3 categories and MeSH terms. Four methodological meta-categories were excluded, yielding 175 analysable diseases. Coverage statistics by disease category type.

**Extended Data Figure 2. Sensitivity Analysis of Dimension Weights.**
(a) Unified Score rankings under alternative weighting schemes (equal, Discovery-heavy, Knowledge-heavy). (b) Rank correlation matrix showing stability across weighting choices (all pairwise Spearman's ρ > 0.92). (c) Bootstrap 95% confidence intervals for top 20 disease rankings.

**Extended Data Figure 3. Temporal Trends in Research Equity (2000–2025).**
(a) UK Biobank publication share over time, showing increasing concentration from 12% to 36%. (b) HIC:LMIC clinical trial site ratio across five-year windows, showing no improvement trend (P = 0.41). (c) Mean temporal drift by disease category, with NTDs showing lower drift (more static research).

**Extended Data Figure 4. Gap Score vs. Semantic Isolation Correlation.**
Scatter plot showing positive correlation between Discovery dimension Gap Scores and Knowledge dimension Semantic Isolation Index (r = 0.67, P < 0.001). Diseases colour-coded by WHO NTD classification. Regression line with 95% confidence band shown. Partial correlation controlling for publication volume: r = 0.58.

**Extended Data Figure 5. Knowledge Transfer Network.**
Network visualization of cross-disease knowledge flow based on semantic similarity. Node size proportional to publication count; edge weight proportional to Knowledge Transfer Potential. NTDs form peripheral cluster with limited connections to central research hubs.

---

## References

### Global Burden of Disease and Health Metrics

1. GBD 2021 Collaborators. Global burden of 369 diseases and injuries in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. *Lancet* 396, 1204–1222 (2020).

2. World Health Organization. *World Health Statistics 2025*. (WHO, Geneva, 2025).

3. Murray, C. J. L. et al. Disability-adjusted life years (DALYs) for 291 diseases and injuries in 21 regions, 1990–2010: a systematic analysis for the Global Burden of Disease Study 2010. *Lancet* 380, 2197–2223 (2012).

4. Vos, T. et al. Global burden of 369 diseases and injuries in 204 countries and territories, 1990–2019. *Lancet* 396, 1204–1222 (2020).

### The 10/90 Gap and Research Equity

5. Commission on Health Research for Development. *Health Research: Essential Link to Equity in Development*. (Oxford University Press, 1990).

6. Kilama, W. L. The 10/90 gap in sub-Saharan Africa: resolving inequities in health research. *Acta Trop.* 112, S8–S15 (2009).

7. Røttingen, J.-A. et al. Mapping of available health research and development data: what's there, what's missing, and what role is there for a global observatory? *Lancet* 382, 1286–1307 (2013).

8. Viergever, R. F. & Hendriks, T. C. C. The 10 largest public and philanthropic funders of health research in the world: what they fund and how they distribute their funds. *Health Res. Policy Syst.* 14, 12 (2016).

9. Evans, J. A., Shim, J.-M. & Ioannidis, J. P. A. Attention to local health burden and the global disparity of health research. *PLoS ONE* 9, e90147 (2014).

### Neglected Tropical Diseases

10. Hotez, P. J. et al. The global burden of disease study 2010: interpretation and implications for the neglected tropical diseases. *PLoS Negl. Trop. Dis.* 8, e2865 (2014).

11. World Health Organization. *Ending the Neglect to Attain the Sustainable Development Goals: A Road Map for Neglected Tropical Diseases 2021–2030*. (WHO, Geneva, 2020).

12. Molyneux, D. H. et al. Neglected tropical diseases: progress towards addressing the chronic pandemic. *Lancet* 389, 312–325 (2017).

13. Hotez, P. J. & Kamath, A. Neglected tropical diseases in sub-Saharan Africa: review of their prevalence, distribution, and disease burden. *PLoS Negl. Trop. Dis.* 3, e412 (2009).

14. Moran, M. et al. Neglected disease research and development: the public divide. *Policy Cures Res.* G-FINDER (2021).

### Genomic Diversity and Biobank Equity

15. Sirugo, G., Williams, S. M. & Tishkoff, S. A. The missing diversity in human genetic studies. *Cell* 177, 26–31 (2019).

16. Popejoy, A. B. & Fullerton, S. M. Genomics is failing on diversity. *Nature* 538, 161–164 (2016).

17. Martin, A. R. et al. Clinical use of current polygenic risk scores may exacerbate health disparities. *Nat. Genet.* 51, 584–591 (2019).

18. Fatumo, S. et al. A roadmap to increase diversity in genomic studies. *Nat. Med.* 28, 243–250 (2022).

19. Bycroft, C. et al. The UK Biobank resource with deep phenotyping and genomic data. *Nature* 562, 203–209 (2018).

20. Sudlow, C. et al. UK Biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. *PLoS Med.* 12, e1001779 (2015).

21. All of Us Research Program Investigators. The "All of Us" Research Program. *N. Engl. J. Med.* 381, 668–676 (2019).

22. Gurdasani, D. et al. Uganda Genome Resource enables insights into population history and genomic discovery in Africa. *Cell* 179, 984–1002.e36 (2019).

### Clinical Trial Equity and Geographic Distribution

23. Drain, P. K. et al. Global health funding and economic development. *Glob. Health* 13, 40 (2017).

24. Alemayehu, C. et al. Barriers and facilitators to clinical trial participation in Africa: a systematic review. *Lancet Glob. Health* 6, e1229 (2018).

25. Gehring, M. et al. Factors influencing clinical trial site distribution in emerging markets. *BMJ Glob. Health* 5, e003023 (2020).

26. Thiers, F. A. et al. Trends in the globalization of clinical trials. *Nat. Rev. Drug Discov.* 7, 13–14 (2008).

27. Glickman, S. W. et al. Ethical and scientific implications of the globalization of clinical research. *N. Engl. J. Med.* 360, 816–823 (2009).

28. Tasneem, A. et al. The database for aggregate analysis of ClinicalTrials.gov (AACT) and subsequent regrouping by clinical specialty. *PLoS ONE* 7, e33677 (2012).

29. Califf, R. M. et al. Characteristics of clinical trials registered in ClinicalTrials.gov, 2007–2010. *JAMA* 307, 1838–1847 (2012).

### Health Research Funding

30. Head, M. G. et al. Research investments in global health: a systematic analysis of UK infectious disease research funding and global health metrics, 1997–2013. *EBioMedicine* 3, 180–190 (2016).

31. Rottingen, J.-A. et al. New vaccines against epidemic infectious diseases. *N. Engl. J. Med.* 376, 610–613 (2017).

32. Yegros-Yegros, A. et al. Exploring why global health needs are unmet by research efforts: the potential influences of geography, industry and publication incentives. *Health Res. Policy Syst.* 18, 47 (2020).

33. Chapman, N. et al. Neglected disease research and development: uneven progress. *G-FINDER Rep.* (Policy Cures Research, 2019).

### Natural Language Processing and Biomedical Text Mining

34. Gu, Y. et al. Domain-specific language model pretraining for biomedical natural language processing. *ACM Trans. Comput. Healthc.* 3, 1–23 (2022).

35. Lee, J. et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics* 36, 1234–1240 (2020).

36. Beltagy, I. et al. SciBERT: a pretrained language model for scientific text. In *Proc. 2019 Conf. Empirical Methods in Natural Language Processing* (EMNLP, 2019).

37. Devlin, J. et al. BERT: pre-training of deep bidirectional transformers for language understanding. In *Proc. 2019 Conf. North American Chapter of the Association for Computational Linguistics* (NAACL, 2019).

38. Mikolov, T. et al. Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems* 26 (NIPS, 2013).

### Bibliometrics and Research Measurement

39. Börner, K. et al. Design and update of a classification system: the UCSD map of science. *PLoS ONE* 7, e39464 (2012).

40. Leydesdorff, L. & Rafols, I. A global map of science based on the ISI Subject Categories. *J. Am. Soc. Inf. Sci. Technol.* 60, 348–362 (2009).

41. Waltman, L. et al. A unified approach to mapping and clustering of bibliometric networks. *J. Informetr.* 4, 629–635 (2010).

42. Boyack, K. W. et al. Mapping the backbone of science. *Scientometrics* 64, 351–374 (2005).

### Dimensionality Reduction and Visualization

43. McInnes, L., Healy, J. & Melville, J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. *J. Open Source Softw.* 3, 861 (2018).

44. van der Maaten, L. & Hinton, G. Visualizing data using t-SNE. *J. Mach. Learn. Res.* 9, 2579–2605 (2008).

### Statistical Methods

45. Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J. R. Stat. Soc. B* 57, 289–300 (1995).

46. Cohen, J. Statistical power analysis for the behavioral sciences. 2nd ed. (Lawrence Erlbaum, 1988).

### International Health Cohorts and Consortia

47. Ollier, W. et al. UK Biobank: from concept to reality. *Pharmacogenomics* 6, 639–646 (2005).

48. International Health Cohorts Consortium. About IHCC: Mission and vision. *IHCC* https://ihccglobal.org/ (2024).

### HIV/AIDS Research as Model for Success

49. Fauci, A. S. & Lane, H. C. Four decades of HIV/AIDS — much accomplished, much to do. *N. Engl. J. Med.* 383, 1–4 (2020).

50. UNAIDS. *Global AIDS Update 2023*. (UNAIDS, Geneva, 2023).

---

## Acknowledgments

We thank the International Health Cohorts Consortium for providing biobank registry data, the Clinical Trials Transformation Initiative for maintaining the AACT database, and the National Library of Medicine for PubMed access. We acknowledge computational resources provided by [institution]. This work was supported by [funding sources to be added].

---

## Author Contributions

**M.C.:** Conceptualization, Methodology, Software, Formal Analysis, Data Curation, Writing—Original Draft, Visualisation, Project Administration.
**M.F.:** Methodology, Validation, Writing—Review & Editing.
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

*Manuscript version 3.0 — Statistics corrected 2026-01-25*

---

## Change Log (v2 → v3)

**Critical Statistics Update:** Corrected NTD semantic isolation statistics based on verified data analysis.

| Statistic | v2 (incorrect) | v3 (verified) |
|-----------|----------------|---------------|
| NTD SII elevation | 20% | **44%** |
| P-value | 0.002 | **< 0.0001** |
| Cohen's d | 0.82 | **1.80** |
| NTD sample size | n = 18 | **n = 15** |
| Non-NTD sample size | n = 157 | **n = 160** |
| NTD mean SII | 0.00203 | **0.00211** |
| Non-NTD mean SII | 0.00152 | **0.00146** |

**Sections updated:**
1. Abstract
2. Results (Knowledge Dimension)
3. Discussion (paragraph 2)
4. Figure 4 legend

**Impact:** The corrected statistics show an even stronger effect than previously reported. The 44% elevation with Cohen's d = 1.80 represents a "very large" effect size, strengthening the paper's central finding.

---

## Change Log (v1 → v2)

1. **Disease count standardized:** Now consistently 175 for semantic analysis (179 GBD - 4 excluded)
2. **SII rankings clarified:** Added note to Table 1 explaining African trypanosomiasis has highest SII but Unified Score rankings differ
3. **Figure 1 citation added:** Introduction now references Fig. 1
4. **HIV/AIDS redundancy reduced:** Shortened in Results, expanded interpretation in Discussion
5. **Transitions added:** Bridging sentences between Discovery→Translation and Translation→Knowledge
6. **"Diseases of affluence" changed:** Now "conditions with substantial burden in high-income countries"
7. **Long sentences split:** Discussion paragraph 2 now three sentences instead of two
8. **RCC equation added:** Full mathematical definition in Methods
9. **Temporal drift equation added:** Full mathematical definition in Methods
10. **Abstract refined:** "Here we introduce" → "We developed"
11. **"Compounding disadvantage" varied:** Changed to "cumulative neglect" in one instance
12. **Word count corrected:** ~3,000 words (was incorrectly listed as 4,276)
13. **Opening strengthened:** More vivid framing of the research-mortality paradox
14. **Human impact added:** "1.5 billion people" affected statistic in Discussion
15. **Closing strengthened:** Dashboard as "measurable accountability"
