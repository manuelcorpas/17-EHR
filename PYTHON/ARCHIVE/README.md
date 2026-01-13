# ARCHIVE Directory

Reorganized 2026-01-13 to streamline active development.

## Structure

```
ARCHIVE/
├── BHEM-v2-2025/    Working previous version (00-XX, 01-XX, 02-XX series)
└── OLD/             Miscellaneous utilities, backups, validation tools
```

## BHEM-v2-2025/

Contains the functional BHEM v1.0/v2.0 pipeline scripts (36 files):
- **00-XX**: Biobank data retrieval and initial analysis (May-Jul 2025)
- **01-XX**: Research gap discovery, network analysis, impact metrics
- **02-XX**: BHEM v2.0 pipeline - fetch, map, analyze, compute, build site

These were superseded by the 03-XX v3.0 IHCC-aligned pipeline but remain functional if needed for reference or rollback.

## OLD/

Contains:
- Utility scripts (refetch, validation helpers)
- Site backups and methodology archives
- WHO genomic map drafts

## Active Scripts (in parent PYTHON/)

- **03-XX**: HEIM-Biobank v3.0 - IHCC-aligned global equity index
- **04-XX**: HEIM-CT - Clinical trials equity extension
