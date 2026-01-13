# Supplementary Table S3: HEIM Formula Specifications

## 1. Burden Score

### Primary Formula (when deaths and prevalence available)

```
Burden_Score = (w_d × DALYs) + (w_m × Deaths) + [w_p × log₁₀(Prevalence)]
```

**Parameters:**
| Symbol | Description | Default Value | Units |
|--------|-------------|---------------|-------|
| w_d | DALY weight | 0.5 | - |
| w_m | Mortality weight | 50.0 | - |
| w_p | Prevalence weight | 10.0 | - |
| DALYs | Disability-adjusted life years | - | Millions |
| Deaths | Annual mortality | - | Millions |
| Prevalence | Total cases | - | Millions |

### Fallback Formula (DALYs only)

```
Burden_Score = 10 × log₁₀(DALYs_raw + 1)
```

Where DALYs_raw is the raw count (not millions).

**Rationale:** Log-transformation prevents extreme burden values from 
dominating and produces comparable scores across different burden magnitudes.

---

## 2. Research Gap Score

### Three-Tier Scoring System

**Tier 1: Zero Publications**
```
If Publications = 0:
    Gap_Score = 95 (Critical)
```

**Tier 2: Category-Specific Thresholds**

For Infectious diseases (HIV, TB, respiratory infections, etc.):
| Publications | Gap Score |
|--------------|-----------|
| < 10 | 90 |
| < 25 | 80 |
| < 50 | 70 |
| < 100 | 60 |
| ≥ 100 | Use Tier 3 |

For Neglected/Maternal diseases:
| Publications | Gap Score |
|--------------|-----------|
| < 10 | 92 |
| < 25 | 82 |
| < 50 | 72 |
| < 100 | 65 |
| ≥ 100 | Use Tier 3 |

**Tier 3: Burden-Normalized Intensity**

```
Pubs_per_Million_DALYs = Publications / DALYs_millions
```

| Pubs per Million DALYs | Gap Score |
|-----------------------|-----------|
| ≥ 100 | 10 |
| ≥ 50 | 20 |
| ≥ 25 | 30 |
| ≥ 10 | 40 |
| ≥ 5 | 50 |
| ≥ 2 | 60 |
| ≥ 1 | 70 |
| ≥ 0.5 | 80 |
| < 0.5 | 85 |

**Global South Penalty:**
```
If Global_South_Priority AND Publications < 50:
    Gap_Score = min(95, Gap_Score + 10)
```

### Gap Severity Classification

| Category | Gap Score Range |
|----------|----------------|
| Critical | > 70 |
| High | 50-70 |
| Moderate | 30-50 |
| Low | < 30 |

---

## 3. Research Opportunity Score (ROS)

```
ROS_b = Σ Burden_Score(d) for all diseases where Publications(b,d) ≤ 2
```

**Interpretation:** Higher ROS indicates greater unrealized potential for 
equity-aligned research. The threshold of ≤2 publications identifies diseases 
effectively unaddressed by a given biobank.

---

## 4. Equity Alignment Score (EAS)

### Formula

```
EAS = 100 - [(w_g × Gap_Severity) + (w_b × Burden_Miss) + (w_c × Capacity_Penalty)]
```

**Weights:**
| Symbol | Description | Default Value |
|--------|-------------|---------------|
| w_g | Gap severity weight | 0.4 |
| w_b | Burden miss weight | 0.3 |
| w_c | Capacity penalty weight | 0.3 |

### Component Calculations

**Gap_Severity (0-100):**
```
Weighted_Gaps = (4 × N_critical) + (2 × N_high) + (1 × N_moderate)
Max_Gaps = 4 × N_diseases
Gap_Severity = (Weighted_Gaps / Max_Gaps) × 100
```

Where N_critical, N_high, N_moderate are counts of diseases in each gap 
category for this biobank's coverage.

**Burden_Miss (0-100):**
```
Missed_DALYs = Σ DALYs for diseases with Publications ≤ 2
Total_DALYs = Σ DALYs for all diseases
Burden_Miss = (Missed_DALYs / Total_DALYs) × 100
```

**Capacity_Penalty (0-100):**
```
Pubs_per_Disease = Total_Publications / N_diseases
Capacity_Penalty = 100 - min(Pubs_per_Disease, 100)
```

**Interpretation:** This penalizes biobanks with low average publications 
per disease, reflecting limited research capacity to address the full 
burden spectrum. The penalty is capped at 100 (i.e., no credit below 
1 publication per disease on average).

### EAS Categories

| Category | EAS Range |
|----------|-----------|
| Strong | ≥ 80 |
| Moderate | 60-79 |
| Developing | 40-59 |
| Low | < 40 |

---

## 5. HIC:LMIC Equity Ratio

### Publication-Based Ratio

```
Equity_Ratio = Publications_HIC / Publications_LMIC
```

**Interpretation:** Ratio > 1 indicates HIC dominance in research output.

### Burden-Adjusted Ratio (Alternative)

```
Equity_Ratio_adj = (Pubs_HIC / DALYs_HIC) / (Pubs_LMIC / DALYs_LMIC)
```

This accounts for differential disease burden between income groups.

---

## 6. Data Sources

| Data Element | Source | Version |
|--------------|--------|---------|
| DALYs, Deaths, Prevalence | IHME Global Burden of Disease | 2021 |
| Biobank Registry | IHCC Global Cohort Atlas | 2024 |
| Publications | PubMed/MEDLINE | Retrieved 2025 |
| Income Classifications | World Bank | 2024 |
| MeSH Terms | NLM | 2024 |

