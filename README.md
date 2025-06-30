# 17-EHR

This repository supports a high-impact review article titled **"Challenges and Opportunities Using Global EHR Linked Biobanks"**, invited for Volume 9 of the *Annual Review of Biomedical Data Science (2026)*. It includes code and data to characterize, cluster, and visualize global literature on biobanks integrated with electronic health records (EHRs).

## 🧠 Project Goal

To systematically examine the scientific landscape at the intersection of biobanking and EHRs, with a focus on global diversity, interoperability, and translational application. Special attention is given to five named biobank initiatives:

* UK Biobank
* All of Us (US)
* FinnGen (Finland)
* Estonian Biobank
* Million Veteran Program (MVP, US)


## 🔍 Methods Overview

To generate the literature corpus used throughout this repository, we conducted a structured search of PubMed using the NCBI Entrez API between 2000 and 2025. The search was organized into eight thematically distinct subqueries, each corresponding to a core conceptual domain such as real-world evidence, FAIR data principles, or machine learning applications in EHR-biobank settings. Articles were deduplicated across subqueries based on PMIDs. Metadata extracted for each record included titles, abstracts, journals, authors, DOIs, publication years, MeSH terms, author-supplied keywords, affiliations, citation counts, and named biobank mentions. The resulting dataset of 3,652 unique articles served as the basis for bibliometric, clustering, and equity-related analyses documented in this repository.

PubMed was queried from 2000–2025 using the NCBI Entrez API across 8 thematically defined subqueries:

1. Global EHR-linked biobanks
2. Data harmonization & record linkage
3. LMICs and underserved populations
4. ML/NLP in EHR-biobanks
5. Ethics, governance, and consent
6. Real-world evidence
7. Precision medicine
8. FAIR data principles

Each article was parsed to extract:

* Title, abstract, journal, authors
* MeSH terms and keywords
* Year, citation count, DOI, affiliations
* Biobank mentions (named entities + fuzzy matching)

## 🧩 Analytical Pipeline

* Yearly trends and biobank-specific histograms
* Top authors, top institutions, citation leaders per biobank
* Keyword normalization and consolidation (e.g., "Mendelian randomisation" variants)
* MeSH term co-occurrence clustering (PCA, UMAP)
* Filtering of dominant MeSH for conceptual summaries

## 📚 Curation Notes

* Citation counts retrieved using Entrez + fallback to CrossRef
* Biobank aliases and keywords are resolved (e.g., “MVP” == “Million Veteran Program”)
* Institutional normalization resolves variants like “Harvard Med Sch” and “Harvard University”

## 🔬 Extensions Underway

* Clustering MeSH terms into thematic domains
* Stratifying clusters by biobank for conceptual contrast
* Equity lens: tracking LMIC mentions and underrepresented populations
* LLM classification layer for manual cluster validation

## ✍️ Citation

This repository accompanies the invited review for *Annual Review of Biomedical Data Science (Vol. 9, 2026)*.

---

**Author**: Dr. Manuel Corpas
**GitHub**: [manuelcorpas](https://github.com/manuelcorpas)
**Contact**: [m.corpas@westminster.ac.uk](mailto:m.corpas@westminster.ac.uk)

