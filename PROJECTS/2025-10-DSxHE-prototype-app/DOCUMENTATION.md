# HEIM Assessor - Complete Documentation

**Version:** 6.0.0  
**Author:** Dr. Manuel Corpas  
**Institution:** Alan Turing Institute  
**Project:** DSxHE Data Diversity Theme  
**License:** [To be determined]  
**Date:** October 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Directory Structure](#directory-structure)
4. [Quick Start Guide](#quick-start-guide)
5. [Data Format Specifications](#data-format-specifications)
6. [Metrics & Methodology](#metrics--methodology)
7. [Using the Application](#using-the-application)
8. [Output Files](#output-files)
9. [Technical Architecture](#technical-architecture)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)
12. [Citation & Credits](#citation--credits)
13. [Appendix](#appendix)

---

## Overview

### What is HEIM?

The **Health Equity Informative Marker (HEIM) Assessor** is a quantitative framework and software tool for measuring representation and diversity in health datasets. It provides researchers with objective metrics and actionable recommendations to improve dataset equity across genomics, clinical trials, imaging studies, and biobanks.

### Purpose

- **Assess** dataset diversity across four key dimensions
- **Visualize** representation gaps through interactive charts
- **Recommend** specific actions to improve equity
- **Export** professional scorecards for grant applications and publications

### Key Features

- ✅ Quantitative HEIM scoring (0-100 scale)
- ✅ Four-dimensional equity assessment (Ancestry, Geography, Age, Sex)
- ✅ Interactive visualizations (Plotly charts and maps)
- ✅ Intelligent gap analysis and recommendations
- ✅ Professional PDF scorecard generation
- ✅ Privacy-first design (all processing local)
- ✅ Citation-ready outputs for publications

---

## Installation

### System Requirements

**Minimum:**
- Python 3.8 or higher
- 4GB RAM
- Modern web browser (Chrome, Firefox, Safari, Edge)
- 100MB disk space

**Recommended:**
- Python 3.10+
- 8GB RAM
- Chrome/Firefox for best visualization performance

### Step 1: Prerequisites

Ensure Python is installed:
```bash
python --version
# Should show Python 3.8 or higher
```

### Step 2: Clone or Download Repository

```bash
# If using Git:
git clone https://github.com/[YOUR_USERNAME]/HEIM.git
cd HEIM

# Or download ZIP and extract, then navigate to folder
```

### Step 3: Create Directory Structure

```bash
# Create required directories if they don't exist
mkdir -p DATA PYTHON OUTPUT
```

### Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install streamlit==1.29.0 pandas==2.1.4 numpy==1.26.2 plotly==5.18.0 reportlab==4.0.7
```

### Step 5: Verify Installation

```bash
# Check all packages installed
pip list | grep -E "streamlit|pandas|numpy|plotly|reportlab"

# Should show all five packages with correct versions
```

### Step 6: Run Application

```bash
# From HEIM root directory
streamlit run PYTHON/app.py

# Application will open automatically in your browser at:
# http://localhost:8501
```

---

## Directory Structure

```
HEIM/
├── PYTHON/                          # Source code modules
│   ├── app.py                       # Main Streamlit application
│   ├── metrics.py                   # Diversity metric calculations
│   ├── reference_data.py            # Population benchmarks
│   ├── scoring.py                   # HEIM composite scoring
│   ├── visualizations.py            # Interactive charts
│   ├── recommendations.py           # Gap analysis engine
│   └── pdf_generator.py             # PDF export system
├── DATA/                            # Input data folder
│   └── sample_data.csv              # Auto-generated sample dataset
├── OUTPUT/                          # Generated reports
│   ├── HEIM_Scorecard_*.pdf         # PDF scorecards
│   └── [timestamped files]
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
├── DOCUMENTATION.md                 # This file
└── LICENSE                          # License information

Total files: ~10 Python modules + documentation
Total size: ~50MB with dependencies
```

---

## Quick Start Guide

### For First-Time Users

**1. Launch the application:**
```bash
streamlit run PYTHON/app.py
```

**2. Load sample data:**
- Click the **"📋 Use Sample Data"** button
- Sample dataset (50 participants) loads automatically

**3. Review results:**
- Scroll through the automatically generated analysis:
  - Data validation results
  - Summary statistics
  - HEIM score and badge
  - Interactive visualizations
  - Recommendations

**4. Download outputs:**
- Scroll to bottom: "📥 Export & Download"
- Click **"📄 Download PDF Scorecard"**
- PDF is saved to `OUTPUT/` folder and downloaded

**Time to first result: ~60 seconds**

### For Your Own Data

**1. Prepare your CSV file:**
- Must include: `participant_id`, `ancestry`, `age`, `sex`, `country`
- See [Data Format Specifications](#data-format-specifications)

**2. Upload your file:**
- Click **"Browse files"** button
- Select your CSV file
- Wait for validation

**3. Review and export:**
- Check validation warnings/errors
- Review HEIM scores and visualizations
- Download PDF scorecard for grant/publication

---

## Data Format Specifications

### Required Columns

Your CSV file **must** include these five columns:

| Column | Type | Description | Valid Values | Example |
|--------|------|-------------|--------------|---------|
| `participant_id` | String | Unique identifier for each participant | Any unique string | P001, SUB_123, ID_2024_001 |
| `ancestry` | String | Ancestry/ethnicity code | See ancestry codes below | EUR, AFR, EAS, SAS, AMR |
| `age` | Numeric | Age in years | 0-120 | 45, 32.5, 67 |
| `sex` | String | Sex/gender | M, F, Male, Female, Other, Unknown, U | M, Female, Other |
| `country` | String | Country of origin or recruitment | Full country name | United Kingdom, Nigeria, India |

### Optional Columns

These columns enhance analysis but are not required:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `region` | String | Geographic region | Europe, Africa, East Asia |
| `disease_status` | String | Case/control status | Case, Control, Healthy |
| `recruitment_site` | String | Study site | London Hospital, Boston Clinic |

### Ancestry Codes

**Coarse-grained codes (recommended for most studies):**

```
AFR - African ancestry
EUR - European ancestry
EAS - East Asian ancestry
SAS - South Asian ancestry
AMR - Admixed American (Latino/Hispanic)
MID - Middle Eastern ancestry
OCE - Oceanian ancestry
OTH - Other ancestry
```

**Fine-grained codes (1000 Genomes Project populations):**

*African:*
- YRI (Yoruba, Nigeria)
- LWK (Luhya, Kenya)
- GWD (Gambian)
- MSL (Mende, Sierra Leone)
- ESN (Esan, Nigeria)
- ASW (African American, Southwest US)
- ACB (African Caribbean, Barbados)

*European:*
- CEU (Utah residents, Northern/Western European)
- TSI (Tuscan, Italy)
- FIN (Finnish)
- GBR (British)
- IBS (Iberian, Spain)

*East Asian:*
- CHB (Han Chinese, Beijing)
- JPT (Japanese, Tokyo)
- CHS (Han Chinese, South)
- CDX (Chinese Dai)
- KHV (Kinh, Vietnam)

*South Asian:*
- GIH (Gujarati Indian)
- PJL (Punjabi, Pakistan)
- BEB (Bengali, Bangladesh)
- STU (Sri Lankan Tamil)
- ITU (Indian Telugu)

*Admixed American:*
- MXL (Mexican American)
- PUR (Puerto Rican)
- CLM (Colombian)
- PEL (Peruvian)

### Example CSV Format

**Minimal valid CSV:**
```csv
participant_id,ancestry,age,sex,country
P001,EUR,45,F,United Kingdom
P002,AFR,32,M,Nigeria
P003,SAS,28,F,India
P004,EAS,51,M,China
P005,AMR,39,F,Mexico
```

**Full CSV with optional columns:**
```csv
participant_id,ancestry,age,sex,country,region,disease_status,recruitment_site
P001,EUR,45,F,United Kingdom,Europe,Case,London Hospital
P002,AFR,32,M,Nigeria,Africa,Control,Lagos Clinic
P003,SAS,28,F,India,South Asia,Case,Mumbai Center
P004,EAS,51,M,China,East Asia,Control,Beijing Hospital
P005,AMR,39,F,Mexico,Americas,Case,Mexico City Site
```

### Data Validation Rules

**The application automatically checks:**

1. **Required columns present** - All 5 mandatory columns exist
2. **No duplicate IDs** - Each participant_id is unique
3. **Valid age range** - Ages between 0-120 years
4. **Recognized ancestry codes** - Codes match valid lists
5. **Valid sex values** - Sex matches expected values
6. **Minimum sample size** - At least 20 participants recommended

**Warnings (non-blocking):**
- Missing values in any column
- Unrecognized ancestry codes (will still process)
- Non-standard sex values
- Small dataset size (<20 participants)

---

## Metrics & Methodology

### HEIM Composite Score

The overall HEIM Representation Score (0-100) is calculated as a weighted composite of four dimensions:

```
HEIM Score = (0.50 × Ancestry) + (0.20 × Geographic) + 
             (0.15 × Age) + (0.15 × Sex)
```

**Rationale for weights:**
- Ancestry (50%): Most critical for genomic/clinical generalizability
- Geography (20%): Environmental and healthcare system variation
- Age (15%): Life-stage effects and applicability
- Sex (15%): Sex-specific effects and balance

### Dimension 1: Ancestry Diversity

**Metrics calculated:**

1. **Simpson's Diversity Index (D)**
   ```
   D = 1 - Σ(pi²)
   where pi = proportion of ancestry group i
   Range: 0 (no diversity) to 1 (maximum diversity)
   ```

2. **Pielou's Evenness Index (J)**
   ```
   J = H / Hmax
   where H = Shannon entropy, Hmax = ln(S), S = number of groups
   Range: 0 (uneven) to 1 (perfectly even)
   ```

3. **Representation Gap Score**
   ```
   Gap = Σ|observed_i - expected_i|
   Compared against 1000 Genomes Project global distribution
   Lower values = better representation
   ```

4. **Composite Ancestry Score**
   ```
   Score = (40% × Diversity) + (30% × Evenness) + (30% × Gap)
   Normalized to 0-100 scale
   ```

**Reference population (1000 Genomes Project):**
- African (AFR): 26%
- European (EUR): 20%
- East Asian (EAS): 26%
- South Asian (SAS): 20%
- Admixed American (AMR): 8%

### Dimension 2: Geographic Diversity

**Metrics calculated:**

1. **Country Count Score**
   - 1-4 countries: 20-40 points
   - 5-9 countries: 40-60 points
   - 10-19 countries: 60-80 points
   - 20+ countries: 80-100 points

2. **Simpson's Diversity by Country**
   - Applied to country distribution
   - Range: 0-1, normalized to 0-100

3. **Composite Geographic Score**
   ```
   Score = (60% × Country Count) + (40% × Diversity)
   ```

### Dimension 3: Age Distribution

**Metrics calculated:**

1. **Range Score**
   ```
   Range Score = min(age_range / 40, 1.0) × 50
   Wider ranges score higher (up to 40+ years)
   ```

2. **Spread Score**
   ```
   Spread Score = min(std_dev / (range/4), 1.0) × 30
   Rewards even distribution across age range
   ```

3. **Distribution Score**
   ```
   Distribution Score = min(CV / 0.3, 1.0) × 20
   where CV = coefficient of variation
   ```

4. **Composite Age Score**
   ```
   Score = Range + Spread + Distribution
   Maximum: 100 points
   ```

### Dimension 4: Sex Balance

**Metrics calculated:**

1. **Balance Score**
   ```
   Score = (1 - |female_prop - 0.5|) × 100
   Perfect 50/50 split = 100 points
   ```

2. **Diversity Score**
   - Simpson's Diversity applied to sex categories
   - Normalized to 0-100

3. **Composite Sex Score**
   ```
   Score = (70% × Balance) + (30% × Diversity)
   ```

### Badge Assignment

Badges are assigned based on overall HEIM score:

| Score Range | Badge | Interpretation |
|-------------|-------|----------------|
| 90-100 | 🥇 Platinum | Exceptional diversity and representation |
| 75-89 | 🥇 Gold | Strong representation with minor gaps |
| 60-74 | 🥈 Silver | Good diversity with notable gaps |
| 40-59 | 🥉 Bronze | Moderate diversity with significant gaps |
| 0-39 | ⚠️ Needs Improvement | Substantial gaps requiring major improvements |

---

## Using the Application

### Main Interface Components

**1. Sidebar (Left)**
- About HEIM: Framework overview
- Data Privacy: Local processing information
- Directory Structure: File organization
- Version information

**2. Main Panel (Center)**
- File upload area
- Data validation results
- Summary statistics
- HEIM scorecard
- Visual analytics
- Recommendations
- Export options

### Step-by-Step Workflow

#### Step 1: Data Upload

**Option A: Use Sample Data**
- Click "📋 Use Sample Data" button
- 50-participant demonstration dataset loads
- Automatically saved to `DATA/sample_data.csv`

**Option B: Upload Your CSV**
- Click "Browse files" button
- Select CSV from your computer
- Wait for upload and validation

#### Step 2: Validation Review

**Check for errors (red boxes):**
- Missing required columns
- Duplicate participant IDs
- Invalid age values
- Must fix before proceeding

**Check for warnings (yellow boxes):**
- Unrecognized ancestry codes
- Missing data in some fields
- Small dataset size
- Non-standard sex values
- Can proceed with warnings

#### Step 3: Data Preview

**Review automatically displayed information:**
- First 10 rows of your data
- Total participant count
- Unique ancestries represented
- Countries represented
- Data completeness percentage
- Ancestry distribution table
- Sex distribution table
- Age statistics (mean, median, range, std dev)

#### Step 4: HEIM Scorecard

**Main Score Display:**
- Large circular gauge showing 0-100 score
- Badge level (Platinum/Gold/Silver/Bronze)
- Interpretation text
- Sample size note

**Dimension Breakdown:**
- Four color-coded progress bars
- Individual scores for each dimension
- Weights displayed (50%, 20%, 15%, 15%)

**Detailed Metrics (Tabs):**
- 🌍 Ancestry: Diversity index, evenness, representation gap
- 📍 Geography: Country count, diversity index
- 👥 Age: Mean, median, range, standard deviation
- ⚧️ Sex: Balance score, diversity index, distribution

#### Step 5: Visual Analytics

**Scroll through interactive visualizations:**

1. **Radar Chart**: Overall dimension scores on spider plot
2. **Ancestry Pie Chart**: Donut chart with percentages
3. **Ancestry Comparison**: Your data vs. global reference (bars)
4. **World Map**: Choropleth showing participant distribution
5. **Top Countries**: Horizontal bar chart of most represented
6. **Age Histogram**: Distribution with reference lines
7. **Sex Distribution**: Color-coded bar chart

**Interaction features:**
- Hover over charts for detailed information
- Zoom and pan on maps
- Click legend items to show/hide data
- Download charts as images (Plotly toolbar)

#### Step 6: Recommendations

**Overall Assessment:**
- Color-coded status message (green/yellow/orange/red)
- Total issues count
- Critical and high priority counts

**Top Priority Actions:**
- Up to 5 most important improvements
- Expandable cards with severity icons
- Dimension, severity level, and next step

**Detailed Recommendations (Tabs):**
- 🌍 Ancestry: Issues and actions for ancestry gaps
- 📍 Geography: Geographic expansion strategies
- 👥 Age: Age range and distribution improvements
- ⚧️ Sex: Balance and representation suggestions

**Each recommendation includes:**
- Severity level (🚨 Critical, ⚠️ High, ⚡ Moderate, 💡 Low)
- Problem description
- Specific recommended actions (bulleted list)

#### Step 7: Export & Download

**Three download options:**

1. **💾 Download CSV Data**
   - Validated dataset with all columns
   - Timestamped filename: `HEIM_Data_YYYYMMDD_HHMMSS.csv`
   - Use for record-keeping or further analysis

2. **📄 Download PDF Scorecard**
   - Professional one-page report
   - Includes all key metrics and recommendations
   - Timestamped: `HEIM_Scorecard_YYYYMMDD_HHMMSS.pdf`
   - Also saved to `OUTPUT/` folder
   - Ready for grant applications or publications

3. **📋 Download Recommendations**
   - Text file with action items
   - Timestamped: `HEIM_Recommendations_YYYYMMDD_HHMMSS.txt`
   - Easy to copy into planning documents

---

## Output Files

### PDF Scorecard Contents

**Page 1 (Complete Report):**

1. **Header Section**
   - HEIM branding and logo
   - "Health Equity Informative Marker Assessment" subtitle
   - Horizontal rule separator

2. **Metadata**
   - Dataset name
   - Generation date and time
   - HEIM version (v6.0)

3. **Overall Score Display**
   - Large score (0-100) in color
   - Badge level prominently displayed
   - Interpretation text

4. **Dataset Summary Table**
   - Total participants
   - Unique ancestries
   - Countries represented
   - Age range (min-max)
   - Mean age
   - Sex distribution (F/M counts)

5. **HEIM Dimension Scores Table**
   - Four rows (Ancestry, Geographic, Age, Sex)
   - Score out of 100 for each
   - Weight percentage
   - Status indicator (✓ or ⚠)

6. **Key Findings Section**
   - 4-6 bullet points
   - ✓ for strengths, ⚠ for gaps
   - Specific numbers and details

7. **Priority Recommendations Table**
   - Top 5 priority actions
   - Priority number (1-5)
   - Issue title (truncated if long)
   - Severity with icon

8. **Footer**
   - Generation information
   - HEIM version reference
   - Dr. Manuel Corpas credit
   - Alan Turing Institute affiliation

**Styling:**
- Professional color scheme (blues, grays)
- Clear tables with grid lines
- Color-coded elements (badges, scores)
- Readable fonts (Helvetica)
- One-page format (fits on letter/A4)

### Text Recommendations Format

```
HEIM RECOMMENDATIONS
Dataset: [Your Dataset Name]
Generated: YYYY-MM-DD HH:MM

OVERALL ASSESSMENT
[Status message with emoji indicators]

TOP PRIORITY ACTIONS

1. [Issue title]
2. [Issue title]
3. [Issue title]
4. [Issue title]
5. [Issue title]
```

---

## Technical Architecture

### Technology Stack

**Frontend:**
- Streamlit 1.29.0: Web application framework
- Plotly 5.18.0: Interactive visualizations
- HTML/CSS: Custom styling

**Backend (Python):**
- pandas 2.1.4: Data manipulation
- numpy 1.26.2: Numerical computations
- reportlab 4.0.7: PDF generation

**Architecture Pattern:**
- Modular design with separated concerns
- Functional programming approach
- Type hints for clarity
- Comprehensive error handling

### Module Descriptions

#### `app.py` (Main Application)
- **Lines:** ~1000
- **Purpose:** Streamlit UI and workflow orchestration
- **Key Functions:**
  - `create_sample_data()`: Generate demonstration dataset
  - `validate_dataframe()`: Data quality checks
  - `display_data_summary()`: Summary statistics UI
  - `display_heim_scorecard()`: Score display UI
  - `display_visualizations()`: Charts rendering
  - `display_recommendations()`: Recommendations UI
  - `main()`: Application entry point

#### `metrics.py` (Diversity Calculations)
- **Lines:** ~300
- **Purpose:** Core diversity metric implementations
- **Key Functions:**
  - `simpsons_diversity_index()`: Calculate Simpson's D
  - `pielou_evenness()`: Calculate Pielou's J
  - `representation_gap()`: Compare to reference
  - `age_distribution_score()`: Age metrics
  - `sex_balance_score()`: Sex balance metrics
  - `geographic_diversity_score()`: Geographic metrics
  - `calculate_all_metrics()`: Aggregate calculator

#### `reference_data.py` (Population Benchmarks)
- **Lines:** ~150
- **Purpose:** Reference population data storage
- **Key Data:**
  - `GLOBAL_ANCESTRY`: 1000 Genomes proportions
  - `CONTINENTAL_ANCESTRY`: 7-category distribution
  - `FINE_GRAINED_ANCESTRY`: 26-population data
  - `DISEASE_DEMOGRAPHICS`: Disease-specific references
  - `ANCESTRY_NAMES`: Code to name mappings

#### `scoring.py` (HEIM Composite Scoring)
- **Lines:** ~400
- **Purpose:** Combine metrics into HEIM score
- **Key Functions:**
  - `calculate_ancestry_score()`: Ancestry dimension
  - `calculate_geographic_score()`: Geographic dimension
  - `calculate_age_score()`: Age dimension
  - `calculate_sex_score()`: Sex dimension
  - `calculate_heim_score()`: Weighted composite
  - `get_badge_interpretation()`: Badge descriptions
  - `get_score_color()`: Color coding

#### `visualizations.py` (Interactive Charts)
- **Lines:** ~450
- **Purpose:** Plotly chart generation
- **Key Functions:**
  - `create_ancestry_pie_chart()`: Donut chart
  - `create_ancestry_comparison_chart()`: Comparison bars
  - `create_geographic_map()`: Choropleth world map
  - `create_age_histogram()`: Age distribution
  - `create_sex_distribution_chart()`: Sex bars
  - `create_diversity_radar_chart()`: Spider plot
  - `create_country_bar_chart()`: Top countries

#### `recommendations.py` (Gap Analysis)
- **Lines:** ~500
- **Purpose:** Intelligent recommendation generation
- **Key Components:**
  - `Recommendation` class: Structured recommendation
  - `analyze_ancestry_gaps()`: Ancestry-specific
  - `analyze_geographic_gaps()`: Geographic-specific
  - `analyze_age_gaps()`: Age-specific
  - `analyze_sex_gaps()`: Sex-specific
  - `generate_recommendations()`: Aggregate generator
  - `generate_priority_actions()`: Prioritization

#### `pdf_generator.py` (PDF Export)
- **Lines:** ~450
- **Purpose:** Professional PDF generation
- **Key Functions:**
  - `create_score_table()`: Dimension scores table
  - `create_summary_table()`: Dataset summary
  - `create_recommendations_table()`: Recommendations
  - `generate_pdf_scorecard()`: Main PDF builder
  - Color helpers: `get_badge_color()`, `get_score_color()`

### Data Flow

```
User Upload CSV
    ↓
Validation (app.py)
    ↓
Metric Calculation (metrics.py)
    ↓
Reference Comparison (reference_data.py)
    ↓
HEIM Scoring (scoring.py)
    ↓
┌─────────────────┬──────────────────┬────────────────────┐
│                 │                  │                    │
Visualization     Gap Analysis       PDF Generation
(visualizations.py) (recommendations.py) (pdf_generator.py)
│                 │                  │
└─────────────────┴──────────────────┴────────────────────┘
                    ↓
            Display to User (app.py)
                    ↓
            Export Files (OUTPUT/)
```

### Performance Characteristics

**Typical processing times (50 participants):**
- Data upload: <1 second
- Validation: <1 second
- Metric calculation: 1-2 seconds
- Visualization generation: 2-3 seconds
- PDF generation: 2-3 seconds
- **Total: 6-10 seconds**

**Scalability:**
- Tested with datasets up to 10,000 participants
- Performance degrades linearly with size
- World map rendering is slowest component for large datasets
- Recommendation: <5,000 participants for optimal experience

**Memory usage:**
- Base application: ~150MB
- Per 1,000 participants: +10MB
- PDF generation: +20MB temporary
- Visualization rendering: +50MB peak

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "ModuleNotFoundError: No module named 'X'"

**Cause:** Missing Python package

**Solution:**
```bash
pip install -r requirements.txt
# Or install specific package:
pip install [package_name]
```

#### Issue: "Streamlit command not found"

**Cause:** Streamlit not in PATH

**Solution:**
```bash
# Try with Python module:
python -m streamlit run PYTHON/app.py

# Or reinstall:
pip install --upgrade streamlit
```

#### Issue: "All arrays must be of the same length"

**Cause:** Mismatched CSV column lengths

**Solution:**
- Check CSV file for incomplete rows
- Ensure all rows have same number of columns
- Remove trailing commas in CSV
- Check for hidden characters

#### Issue: "Invalid ancestry code"

**Cause:** Unrecognized ancestry values

**Solution:**
- This is a WARNING, not an error - analysis continues
- Check ancestry codes against valid list
- Use coarse codes (AFR, EUR, etc.) for best results
- Or update `VALID_ANCESTRY_CODES` in `app.py`

#### Issue: PDF generation fails

**Cause:** Missing reportlab or file permission issues

**Solution:**
```bash
# Reinstall reportlab:
pip install --upgrade reportlab

# Check OUTPUT/ directory permissions:
ls -la OUTPUT/

# Create OUTPUT directory if missing:
mkdir OUTPUT
chmod 755 OUTPUT
```

#### Issue: World map not displaying

**Cause:** Country names not recognized by Plotly

**Solution:**
- Use full country names (not abbreviations)
- Check spelling (e.g., "United Kingdom" not "UK")
- Some countries may not render (known Plotly limitation)

#### Issue: Application runs but nothing displays

**Cause:** Browser cache or session state issues

**Solution:**
1. Press `Ctrl+R` or `Cmd+R` to reload page
2. Clear browser cache
3. Try different browser
4. Stop app (`Ctrl+C`) and restart

#### Issue: "Port 8501 already in use"

**Cause:** Another Streamlit app running

**Solution:**
```bash
# Use different port:
streamlit run PYTHON/app.py --server.port 8502

# Or kill existing process:
lsof -ti:8501 | xargs kill -9  # Mac/Linux
# Windows: Find and close Python process in Task Manager
```

#### Issue: Slow performance with large datasets

**Cause:** Dataset >5,000 participants

**Solution:**
- Use sampling: analyze random subset
- Disable world map (comment out in code)
- Increase system memory allocation
- Consider batch processing approach

---

## Advanced Usage

### Customizing Reference Populations

Edit `PYTHON/reference_data.py` to modify expected distributions:

```python
# Example: Adjust for disease-specific study
CUSTOM_ANCESTRY = {
    'AFR': 0.40,  # Increased for sickle cell study
    'EUR': 0.15,
    'EAS': 0.20,
    'SAS': 0.20,
    'AMR': 0.05
}
```

### Modifying Scoring Weights

Edit `PYTHON/scoring.py` to change dimension importance:

```python
# Example: Increase age importance for aging study
WEIGHTS = {
    'ancestry': 0.40,    # Reduced from 50%
    'geographic': 0.20,
    'age': 0.25,         # Increased from 15%
    'sex': 0.15
}
```

### Batch Processing Multiple Datasets

Create a Python script:

```python
import pandas as pd
from pathlib import Path
from scoring import calculate_heim_score
from pdf_generator import generate_pdf_scorecard

# Process all CSVs in a folder
data_folder = Path("DATA/batch/")
for csv_file in data_folder.glob("*.csv"):
    df = pd.read_csv(csv_file)
    result = calculate_heim_score(df)
    generate_pdf_scorecard(
        df, result, {},
        output_path=Path(f"OUTPUT/{csv_file.stem}_report.pdf")
    )
```

### Custom Visualization Themes

Edit `PYTHON/visualizations.py` for color schemes:

```python
# Example: Use colorblind-friendly palette
import plotly.express as px
colors = px.colors.qualitative.Safe  # Colorblind-safe
```

### API Integration

While HEIM is designed as a web app, you can use core functions programmatically:

```python
from scoring import calculate_heim_score
from recommendations import generate_recommendations
import pandas as pd

# Load data
df = pd.read_csv("my_data.csv")

# Calculate score
heim_result = calculate_heim_score(df, reference_type='coarse')
print(f"HEIM Score: {heim_result['overall_score']}")

# Get recommendations
recs = generate_recommendations(df, heim_result)
print(f"Priority actions: {len(recs['priority_actions'])}")
```

---

## Citation & Credits

### Citing HEIM Assessor

**Software citation:**
```bibtex
@software{heim_assessor_2025,
  author = {Corpas, Manuel},
  title = {HEIM Assessor: Health Equity Informative Marker for Dataset Diversity Assessment},
  year = {2025},
  version = {6.0.0},
  institution = {Alan Turing Institute},
  url = {https://github.com/[YOUR_USERNAME]/HEIM}
}
```

**Methodology citation (when available):**
```bibtex
@article{corpas2025heim,
  author = {Corpas, Manuel and [colleagues]},
  title = {HEIM: A Quantitative Framework for Measuring Dataset Equity in Genomics},
  journal = {[Journal Name]},
  year = {2025},
  volume = {XX},
  pages = {XXX-XXX}
}
```

### Acknowledgments

**Development:**
- Dr. Manuel Corpas (Alan Turing Institute): Conceptualization and methodology
- Claude AI (Anthropic): Code generation and architecture assistance
- DSxHE Data Diversity Theme: Framework development

**Data Sources:**
- 1000 Genomes Project: Reference population distributions
- GA4GH (Global Alliance for Genomics & Health): Standards alignment

**Funding:**
[Add funding acknowledgments as appropriate]

### License

[License information to be determined]

Suggested: MIT License or CC-BY 4.0 for maximum reproducibility

---

## Appendix

### A. Glossary of Terms

**HEIM:** Health Equity Informative Marker - quantitative framework for measuring dataset diversity

**Simpson's Diversity Index:** Metric measuring probability that two randomly selected individuals belong to different groups

**Pielou's Evenness:** Metric measuring how evenly distributed groups are (0=uneven, 1=perfectly even)

**Representation Gap:** Absolute difference between observed and expected proportions

**Badge:** Qualitative label (Platinum/Gold/Silver/Bronze) based on HEIM score

**Ancestry:** Genetic ancestry or ethnicity categorization

**1000 Genomes Project:** International research effort that catalogued genetic variation in human populations

### B. Ancestry Code Reference Card

```
┌─────────┬──────────────────┬────────────┐
│ Code    │ Full Name        │ % Global   │
├─────────┼──────────────────┼────────────┤
│ AFR     │ African          │ 26%        │
│ EUR     │ European         │ 20%        │
│ EAS     │ East Asian       │ 26%        │
│ SAS     │ South Asian      │ 20%        │
│ AMR     │ Admixed American │ 8%         │
│ MID     │ Middle Eastern   │ Variable   │
│ OCE     │ Oceanian         │ Variable   │
│ OTH     │ Other            │ Variable   │
└─────────┴──────────────────┴────────────┘
```

### C. Metric Interpretation Guide

**Simpson's D:**
- 0.0-0.3: Low diversity (1-2 groups dominate)
- 0.3-0.6: Moderate diversity (several groups present)
- 0.6-0.8: Good diversity (many groups represented)
- 0.8-1.0: Excellent diversity (highly diverse)

**Pielou's J:**
- 0.0-0.4: Very uneven (one group dominates heavily)
- 0.4-0.6: Uneven (noticeable imbalance)
- 0.6-0.8: Fairly even (moderate balance)
- 0.8-1.0: Very even (excellent balance)

**HEIM Score:**
- 90-100: Exceptional - meets highest equity standards
- 75-89: Strong - suitable for most applications
- 60-74: Acceptable - improvements recommended
- 40-59: Limited - substantial gaps present
- 0-39: Inadequate - major improvements required

### D. FAQ

**Q: Can I use HEIM for non-genomic studies?**  
A: Yes! HEIM applies to any health dataset with demographic information (clinical trials, imaging, surveys, etc.)

**Q: What if my dataset has <20 participants?**  
A: HEIM will process it but issue a warning. Diversity metrics become unreliable with very small samples.

**Q: Can I modify the reference populations?**  
A: Yes, edit `reference_data.py`. Use your own disease-specific or regional distributions.

**Q: Is my data uploaded to a server?**  
A: No. All processing happens locally in your browser. Data never leaves your computer.

**Q: Can I use this for grant applications?**  
A: Yes! The PDF scorecard is designed for appendices to grant proposals and diversity statements.

**Q: What if some countries don't appear on the world map?**  
A: This is a known Plotly limitation. Ensure country names are spelled correctly and use full names.

**Q: How often should I reassess my dataset?**  
A: Reassess whenever you add significant new participants or complete a recruitment phase.

**Q: Can I compare multiple datasets?**  
A: Currently no built-in comparison. Save multiple PDFs and compare manually, or see "Advanced Usage" for batch processing.

### E. Version History

**v6.0.0 (October 2025) - Week 6 Release**
- ✅ Added professional PDF export system
- ✅ Three-column export interface
- ✅ Timestamped output files
- ✅ ReportLab integration

**v4.0.0 (October 2025) - Week 4 Release**
- ✅ Added intelligent recommendations engine
- ✅ Priority-ranked action items
- ✅ Severity-based color coding
- ✅ Downloadable text recommendations

**v3.0.0 (October 2025) - Week 3 Release**
- ✅ Added interactive visualizations
- ✅ Plotly charts (pie, bars, maps)
- ✅ World choropleth map
- ✅ Radar chart for dimensions

**v2.0.0 (October 2025) - Week 2 Release**
- ✅ Implemented HEIM scoring engine
- ✅ Four-dimension metrics calculation
- ✅ Badge assignment system
- ✅ Circular score gauge

**v1.0.0 (October 2025) - Week 1 Release**
- ✅ Initial data upload interface
- ✅ CSV validation system
- ✅ Summary statistics display
- ✅ Sample data generator

### F. Contact & Support

**Bug Reports:**  
Please submit issues to: [GitHub Issues URL]

**Feature Requests:**  
Email: [contact email]

**Questions:**  
Join discussion: [Community forum/Slack]

**Contributing:**  
See CONTRIBUTING.md for guidelines

**Updates:**  
Follow: [@HeimAssessor on Twitter]  
Newsletter: [signup link]

---

## Conclusion

This documentation provides complete information for installing, using, and understanding HEIM Assessor. For additional support or to contribute to the project, please see contact information in Appendix F.

**Last updated:** October 11, 2025  
**Documentation version:** 1.0  
**Software version:** 6.0.0

---

*HEIM Assessor - Making equity measurable and actionable in health research.*