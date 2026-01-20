# Methods: PubMed Literature Retrieval

## Disease Taxonomy

We used the Global Burden of Disease (GBD) 2021 taxonomy as our classification framework, encompassing 176 disease categories. This taxonomy provides standardised definitions enabling systematic comparison across the full spectrum of human disease.

## The Terminology Gap

GBD disease names are optimised for epidemiological consistency, not literature retrieval. Researchers rarely use GBD terminology in their publications. For example, the GBD category "Tracheal, bronchus, and lung cancer" yields only 20 publications when searched as an exact phrase, whereas the literature on lung cancer comprises over 200,000 papers. This terminology mismatch affects nearly all GBD categories to varying degrees.

## Query Strategy

To bridge this gap, we constructed PubMed queries using two complementary approaches:

**1. MeSH Major Topics.** Medical Subject Headings (MeSH) are a controlled vocabulary assigned to PubMed articles by trained indexers at the National Library of Medicine. Indexers read each article and assign standardised descriptors regardless of the authors' terminology—a paper on "bronchogenic adenocarcinoma" and one on "NSCLC" both receive the MeSH term "Lung Neoplasms."

We restricted queries to MeSH **Major Topics** ([Majr]), a designation indicating the term represents a central focus of the article rather than a peripheral mention. This distinction, made by human indexers based on each term's prominence in the paper, provides a validated relevance filter at the point of indexing.

**2. Title/Abstract Keywords.** MeSH indexing has known limitations: recently published articles may not yet be indexed, and indexing depth varies across journals. We therefore supplemented Major Topic queries with keyword searches in titles and abstracts ([tiab]), capturing common synonyms and abbreviations.

## Query Structure

Each GBD disease category was mapped to one or more MeSH Major Topics and relevant keywords, combined with Boolean OR logic:

```
(MeSH_term[Majr] OR keyword1[tiab] OR keyword2[tiab] OR ...)
```

Examples:

| GBD Category | PubMed Query |
|--------------|--------------|
| Tracheal, bronchus, and lung cancer | `("Lung Neoplasms"[Majr] OR "lung cancer"[tiab] OR "NSCLC"[tiab])` |
| Alzheimer's disease and other dementias | `("Alzheimer Disease"[Majr] OR "Dementia"[Majr] OR "Alzheimer"[tiab])` |
| Intestinal nematode infections | `("Nematode Infections"[Majr] OR "soil-transmitted helminth"[tiab])` |

This mapping was applied uniformly across all 176 disease categories. The complete mapping specification is provided in Supplementary Data.

## Temporal Scope

We retrieved publications indexed between 2000 and 2025, stratified by year to enable temporal trend analysis. For each disease-year combination, we extracted PubMed identifiers, titles, abstracts, and MeSH annotations.

## Precision Rationale

The Major Topic restriction addresses a key methodological concern: ensuring retrieved publications are genuinely focused on the target disease. Standard MeSH queries return papers where a term appears anywhere in the indexing, including tangential mentions. Major Topic designation requires the concept to be central to the article's content—a judgement made by trained human indexers who have read the paper.

This human-in-the-loop validation at the indexing stage provides precision assurance that would otherwise require manual review of retrieved abstracts. The title/abstract keyword component ensures recall for papers not yet fully indexed or using non-standard terminology.
