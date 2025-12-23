# Data Directory

This directory contains the input data for the EHR-Linked Biobank analysis.

## Files

### Primary Dataset

| File | Description | Size | Records |
|------|-------------|------|---------|
| `biobank_research_data.csv` | Main PubMed dataset with all biobank-linked publications | ~15 MB | 14,655 |
| `biobank_research_data_deduplicated.csv` | Deduplicated version (unique PMIDs) | ~12 MB | ~14,142 |

**Columns:**
- `Biobank`: Which biobank the paper refers to (UK Biobank, FinnGen, All of Us, MVP, Estonian Biobank)
- `PMID`: PubMed identifier
- `Title`: Article title
- `Abstract`: Full abstract text
- `Journal`: Journal name
- `Year`: Publication year (2000-2024)
- `MeSH_Terms`: Medical Subject Headings (semicolon-separated)

### Disease Burden Data

| File | Description | Source |
|------|-------------|--------|
| `IHMEGBD_2021_DATA*.csv` | Global Burden of Disease 2021 estimates | [IHME GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) |
| `GBD-Results-tool-citation.txt` | Proper citation for GBD data | IHME |

**GBD Metrics Used:**
- DALYs (Disability-Adjusted Life Years)
- Deaths (annual mortality)
- Prevalence (disease cases)

**Filters Applied:**
- Location: Global
- Age: All ages
- Sex: Both
- Metric: Number (not rate)
- Year: 2021

---

## Regenerating the Data

To regenerate `biobank_research_data.csv` from PubMed:

```bash
python PYTHON/00-00-biobank-data-retrieval.py
```

**Requirements:**
- Internet connection
- Entrez email configured (see script)
- ~30 minutes runtime

**Note:** PubMed data is dynamic. Running the retrieval script after the paper publication date may yield additional records.

---

## Biobank Search Terms

Each biobank was searched using the following aliases:

| Biobank | Search Terms |
|---------|--------------|
| UK Biobank | "UK Biobank", "United Kingdom Biobank", "U.K. Biobank", "UK-Biobank" |
| Million Veteran Program | "Million Veteran Program", "Million Veterans Program", "MVP biobank", "MVP cohort", "VA Million Veteran Program" |
| FinnGen | "FinnGen", "FinnGen biobank", "FinnGen study", "FinnGen cohort", "FinnGen consortium" |
| All of Us | "All of Us Research Program", "All of Us cohort", "All of Us biobank", "AoU Research Program" |
| Estonian Biobank | "Estonian Biobank", "Estonia Biobank", "Estonian Genome Center", "Tartu Biobank" |

---

## Data Cleaning Notes

1. **Year filtering**: 2000-2024 (2025 excluded as incomplete at analysis time)
2. **Preprint exclusion**: medRxiv, bioRxiv, Research Square, arXiv, etc.
3. **MeSH requirement**: Only papers with MeSH annotations used for clustering
4. **Deduplication**: Based on PMID (same paper may mention multiple biobanks)

---

## Archive

The `ARCHIVE/` subdirectory contains deprecated data files from earlier analysis iterations. These are retained for reproducibility of exploratory analyses but are not used in the final paper.

---

## Citation

If using this data, please cite both:

1. The paper (see main README)
2. Original data sources:
   - PubMed/MEDLINE: National Library of Medicine
   - GBD 2021: IHME, University of Washington

```bibtex
@misc{gbd2021,
  author = {{GBD 2021 Diseases and Injuries Collaborators}},
  title = {Global incidence, prevalence, years lived with disability (YLDs), disability-adjusted life-years (DALYs), and healthy life expectancy (HALE) for 371 diseases and injuries in 204 countries and territories},
  journal = {The Lancet},
  year = {2024},
  volume = {403},
  pages = {2133-2161}
}
```
