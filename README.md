# 17-EHR

This repository supports a high-impact review article titled **"Challenges and Opportunities Using Global EHR Linked Biobanks"**, invited for Volume 9 of the *Annual Review of Biomedical Data Science (2026)*. It includes code and data to characterize and visualize the global literature landscape of major biobank initiatives.

## 🧠 Project Goal

To systematically examine the scientific landscape of research utilizing five major biobank initiatives, with a focus on understanding research themes, temporal trends, and comparative research activity. This analysis provides foundational data for understanding the global impact and research directions of major biobanking efforts.

* UK Biobank
* All of Us (US)
* FinnGen (Finland)
* Estonian Biobank
* Million Veteran Program (MVP, US)

## 🔍 Methods Overview

To generate the literature corpus used throughout this repository, we conducted a systematic search of PubMed using the NCBI Entrez API between 2000 and 2024. The search strategy focused specifically on identifying all published research that mentions any of the five major biobank initiatives by name, prioritizing published, peer-reviewed articles while systematically excluding preprints to ensure research quality and validity.

### Data Collection Strategy

**PubMed Query Period**: 2000–2024 (2025 excluded as incomplete year)

**Search Methodology**: 
- **Biobank-Specific Name Searches**: Individual searches for each target biobank using exact names and known aliases
- **No Thematic Filtering**: Comprehensive retrieval of ALL papers mentioning target biobanks regardless of research topic
- **Simple Boolean Logic**: Biobank names OR aliases + date range constraints only

**Target Biobanks and Search Terms**:
- **UK Biobank**: "UK Biobank", "United Kingdom Biobank", "U.K. Biobank", "UK-Biobank"
- **Million Veteran Program**: "Million Veteran Program", "Million Veterans Program", "MVP biobank", "MVP cohort", "MVP genomics", "Veterans Affairs Million Veteran Program", "VA Million Veteran Program"
- **FinnGen**: "FinnGen", "FinnGen biobank", "FinnGen study", "FinnGen cohort", "FinnGen consortium"
- **All of Us**: "All of Us Research Program", "All of Us cohort", "All of Us biobank", "AoU Research Program", "Precision Medicine Initiative cohort"
- **Estonian Biobank**: "Estonian Biobank", "Estonia Biobank", "Estonian Genome Center", "Estonian Health Cohort", "Tartu Biobank"

**Quality Assurance**:
- ✅ **Preprint Exclusion**: Systematic filtering of medRxiv, bioRxiv, Research Square, arXiv, and other preprint servers (applied post-retrieval)
- ✅ **Published Papers Only**: Focus on peer-reviewed, formally published research
- ✅ **Complete Year Analysis**: 2025 excluded to prevent bias from incomplete data
- ✅ **Date-based Chunking**: Year-by-year (and month-by-month when needed) retrieval to bypass PubMed's 9,999 record limit
- ✅ **Comprehensive Coverage**: No topic restrictions - captures all research domains mentioning target biobanks

### Technical Implementation

**Search Query Structure**:
```
(biobank_terms) AND ("YEAR"[PDAT])
```
Where `biobank_terms` = `"Term1"[All Fields] OR "Term2"[All Fields] OR ...`

**Scalability Approach**:
- **Primary Strategy**: Year-by-year searches (2000-2024)
- **Fallback Strategy**: Month-by-month searches when yearly results exceed 9,999 records
- **Rate Limiting**: 0.34-second delays between API calls to respect NCBI guidelines
- **Batch Processing**: 100-record batches for metadata retrieval

### Data Processing Pipeline

**Deduplication Strategy**: 
- Articles mentioning multiple biobanks appear once per biobank in raw data
- PMID-based deduplication creates unique article dataset
- Both raw and deduplicated datasets preserved for different analyses

**Metadata Extraction**:
- **Bibliographic Data**: Titles, abstracts, journals, publication years
- **Indexing Terms**: MeSH terms (Medical Subject Headings)
- **Identifiers**: PubMed IDs (PMIDs) for cross-referencing
- **Biobank Attribution**: Which biobank(s) each paper mentions

**Output Datasets**:
- `biobank_research_data.csv`: Raw data with potential duplicates across biobanks
- `biobank_research_data_deduplicated.csv`: Unique papers (PMID-based deduplication)

The resulting dataset captures the complete landscape of published research mentioning these major biobank initiatives, providing comprehensive coverage for bibliometric and trend analyses.

### Preprint Filtering Methodology

To ensure research quality, we implemented comprehensive preprint detection and exclusion:

**Excluded Sources**:
- medRxiv, bioRxiv, Research Square
- arXiv, ChemRxiv, PeerJ Preprints
- F1000Research, OSF Preprints
- Any journal containing "preprint", "working paper", or similar terms

**Filtering Statistics**: 
- Detailed breakdown of excluded preprints by source
- Transparency reporting on filtering impact
- Comparative analysis of published vs. preprint distributions

## 🧩 Analytical Pipeline

### Publication Analysis
* **Temporal Trends**: Yearly publication patterns (2000-2024)
* **Biobank-Specific Analysis**: Individual biobank research landscapes
* **Quality Metrics**: Impact factors, citation patterns, journal prestige

### Content Analysis
* **MeSH Term Analysis**: Top medical subject headings per biobank and globally
* **Journal Distribution**: Publication venue preferences and patterns
* **Biobank Coverage**: Research volume and trends for each target biobank
* **Temporal Patterns**: Publication trends and growth rates over time (2000-2024)

### Network Analysis
* **MeSH Co-occurrence**: Identification of research theme relationships
* **Cross-Biobank Comparisons**: Research focus similarities and differences between biobanks
* **Temporal Evolution**: How research themes have evolved over 25 years

### Bibliometric Analysis
* **Publication Volume**: Papers per biobank, year, and journal
* **Research Diversity**: Breadth of topics covered per biobank
* **Growth Patterns**: Temporal trends in biobank research activity

## 📊 Visualization Outputs

### Publication-Quality Figures
All visualizations are generated in both PNG and PDF formats suitable for academic publication:

* **Preprint Filtering Summary**: Transparency in data quality assurance
* **Yearly Distribution Plots**: Publication trends by biobank (2000-2024)
* **MeSH Terms Analysis**: Top research themes per biobank
* **Journal Analysis**: Publishing venue preferences
* **Publication Trends**: Multi-year trend lines with statistical analysis
* **Combined Overview**: Comprehensive research landscape summary

### Quality Standards
* High-resolution outputs (300 DPI)
* Professional color palettes and typography
* Clear legends and axis labels
* Academic publication formatting standards

## 📚 Curation Notes

### Data Quality Assurance
* **Metadata Completeness**: Systematic extraction of available PubMed fields (Title, Abstract, Journal, Year, MeSH Terms, PMID)
* **Biobank Entity Recognition**: Exact string matching for biobank names and established aliases
* **Temporal Filtering**: Year validation ensuring data completeness and accuracy (2000-2024)
* **Deduplication**: PMID-based removal of duplicate articles across biobank searches

### Data Limitations
* **Citation Metrics**: Not retrieved in current implementation
* **Author Information**: Not extracted from PubMed records  
* **Institutional Affiliations**: Not included in current data collection
* **Full-Text Access**: Limited to PubMed abstracts and metadata

### Search Strategy Validation
* **Comprehensive Coverage**: Simple name-based search ensures maximum recall for target biobanks
* **No False Positives**: Exact string matching minimizes irrelevant results
* **Temporal Completeness**: Year-by-year approach ensures no missing time periods
* **API Compliance**: Rate limiting and batch processing respect NCBI guidelines

### Reproducibility
* **Version Control**: All scripts and parameters documented
* **Data Provenance**: Clear tracking of data sources and transformations
* **Parameter Documentation**: Configurable analysis parameters
* **Error Handling**: Robust exception handling and progress monitoring

## 🔬 Extensions Underway

### Advanced Analytics
* **Semantic Clustering**: MeSH terms grouped into thematic research domains
* **Biobank Research Profiling**: Characterization of research focus per biobank using MeSH analysis
* **Temporal Theme Evolution**: Tracking how research themes change over time (2000-2024)
* **Cross-Biobank Theme Analysis**: Comparative analysis of research focus between biobanks

### Text Mining and NLP
* **Abstract Analysis**: Natural language processing of abstract content for deeper insights
* **Topic Modeling**: Unsupervised discovery of research themes beyond MeSH terms
* **Keyword Extraction**: Identification of emerging terminology and research areas
* **Comparative Biobank Vocabularies**: Analysis of language and terminology differences

### Methodological Enhancements
* **Search Strategy Optimization**: Refinement of biobank name detection and alias expansion
* **Metadata Enrichment**: Integration with external databases for additional paper metrics
* **Validation Studies**: Manual review of samples to assess search precision and recall
* **Temporal Bias Assessment**: Analysis of search completeness across different time periods

## 📁 Repository Structure

```
├── PYTHON/
│   ├── 00-00-biobank-data-retrieval.py    # PubMed data collection
│   └── 00-01-biobank-analysis.py          # Analysis and visualization
├── DATA/
│   ├── biobank_research_data.csv          # Raw retrieved data
│   └── biobank_research_data_deduplicated.csv
├── ANALYSIS/
│   └── 00-01-BIOBANK-ANALYSIS/           # Generated figures and reports
└── README.md                              # This file
```

## 🚀 Usage

### Data Retrieval
```bash
# Retrieve papers mentioning target biobanks from PubMed (2000-2025)
# Note: Analysis script will filter to 2000-2024 and exclude preprints
python PYTHON/00-00-biobank-data-retrieval.py
```

### Analysis and Visualization
```bash
# Generate publication-quality figures and statistics
# Automatically excludes 2025 data and preprints for analysis
python PYTHON/00-01-biobank-analysis.py
```

### Requirements
```bash
pip install pandas matplotlib seaborn numpy scipy biopython
```

## ✍️ Citation

This repository accompanies the invited review for *Annual Review of Biomedical Data Science (Vol. 9, 2026)*. The methodology prioritizes research quality through systematic preprint exclusion and temporal data completeness.

---

**Author**: Dr. Manuel Corpas  
**Affiliation**: University of Westminster  
**GitHub**: [manuelcorpas](https://github.com/manuelcorpas)  
**Contact**: [m.corpas@westminster.ac.uk](mailto:m.corpas@westminster.ac.uk)

**Data Quality Standards**: Published papers only (2000-2024) | Preprints excluded | Name-based biobank identification | Comprehensive temporal coverage