# EHR-Linked Biobank Analysis Pipeline - Qatar Biobank Update

## Overview
This update adds **Qatar Biobank** to the complete biobank analysis pipeline, expanding coverage from 6 to **7 major global biobanks**.

## Biobanks Now Analyzed (7 total)
1. **UK Biobank** (founded 2006)
2. **Million Veteran Program** (founded 2011)
3. **FinnGen** (founded 2017)
4. **All of Us Research Program** (founded 2015)
5. **Estonian Biobank** (founded 2000)
6. **Genomics England** (founded 2013)
7. **Qatar Biobank** (founded 2012) - *NEW*

## Qatar Biobank Details
- **Founding Year**: 2012
- **Search Terms Added**:
  - "Qatar Biobank"
  - "QBB"
  - "Qatar Biobank cohort"
  - "Qatar Genome Programme"
  - "Qatar Genome Project"
  - "QBB cohort"
  - "Qatar Biobank study"
  - "Qatar National Biobank"
  - "Qatari Biobank"
  - "Qatar biobank participants"
  - "Qatar Biobank data"

## Scripts Updated

### 00-00-biobank-data-retrieval.py
- Added Qatar Biobank to `BIOBANK_DEFINITIONS` dictionary
- Includes search terms and founding year (2012)
- Updated documentation to reflect 7 biobanks

### 00-01-biobank-analysis.py
- Updated NUM_BIOBANKS constant to 7
- Added Qatar Biobank highlighting in output statistics
- Updated all figure titles and descriptions
- Updated summary text to include Qatar Biobank

### 00-02-biobank-mesh-clustering.py
- Updated documentation header to list 7 biobanks
- Added Qatar Biobank to biobank list
- Script already processes biobanks dynamically from data

### 00-03-ge-cluster-analysis.py
- *No changes needed* - This is a Genomics England-specific strategic analysis script
- For Qatar Biobank-specific analysis, a similar script could be created (00-03-qb-cluster-analysis.py)

### 01-00-research-gap-discovery.py
- Updated documentation to include Qatar Biobank
- Script already processes biobanks dynamically from data

### 02-00-ge-fairness.py
- *No changes needed* - Script already processes all biobanks dynamically from data

## Pipeline Execution Order
```bash
# 1. Data Retrieval (now retrieves 7 biobanks)
python PYTHON/00-00-biobank-data-retrieval.py

# 2. Basic Analysis
python PYTHON/00-01-biobank-analysis.py

# 3. MeSH Clustering
python PYTHON/00-02-biobank-mesh-clustering.py

# 4. Genomics England Strategic Analysis (optional)
python PYTHON/00-03-ge-cluster-analysis.py

# 5. Equity Gap Analysis
python PYTHON/01-00-research-gap-discovery.py

# 6. Fairness/Bias Detection
python PYTHON/02-00-ge-fairness.py
```

## Expected Output Changes
With Qatar Biobank included, you will see:
- Additional rows in all biobank comparison tables
- Qatar Biobank publications in trend analyses
- Qatar Biobank clusters in semantic space visualizations
- Qatar Biobank equity gap scores
- Qatar Biobank bias profiles

## Notes
- The pipeline automatically discovers and processes all biobanks present in the input CSV
- Qatar Biobank provides important Middle Eastern population representation
- Adds genomic diversity from Qatari and broader Gulf populations
- Complements existing European, American, and Finnish cohorts

## Contact
For questions about the Qatar Biobank integration, consult:
- Dr. Wadha Al-Muftah (Qatar Biobank)
- Manuel Corpas (pipeline development)
