# Supplementary Table S4: Software and Library Versions

## Computational Environment

| Component | Version |
|-----------|---------|
| Python | 3.11.14 |
| Operating System | Darwin 24.6.0 |
| Architecture | arm64 |

## Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.4 | Numerical computing |
| scipy | 1.16.3 | Statistical analysis |
| scikit-learn | 1.7.2 | Machine learning |
| matplotlib | 3.10.7 | Visualization |
| seaborn | 0.13.2 | Statistical visualization |
| biopython | 1.86 | NCBI API access |
| sentence-transformers | 5.2.0 | Semantic similarity |

## External APIs

| Service | Purpose |
|---------|---------|
| NCBI Entrez | PubMed publication retrieval |
| IHME GBD Results Tool | Disease burden data |

## Reproducibility

All analysis code is available at: https://github.com/manuelcorpas/17-EHR

Analysis was performed on: 2026-01-11

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Year range | 2000-2025 |
| Random seed | 42 |
| Bootstrap iterations | 1000 |
| Confidence interval | 95% (percentile method) |
