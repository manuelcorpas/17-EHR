#!/usr/bin/env python3
"""
Comprehensive Data Audit - Cross-reference with manuscript and source scripts
"""

import json
from pathlib import Path

print("="*80)
print("COMPREHENSIVE DATA AUDIT")
print("="*80)

data_dir = Path('/mnt/user-data/uploads')

# Load dashboard data
biobanks = json.load(open(data_dir / 'biobanks.json'))['biobanks']
diseases = json.load(open(data_dir / 'diseases.json'))['diseases']
matrix = json.load(open(data_dir / 'matrix.json'))
summary = json.load(open(data_dir / 'summary.json'))

#############################################################################
# 1. MANUSCRIPT REFERENCE VALUES (from EHR-Linked-biobanks_v7.docx)
#############################################################################
print("\n1. MANUSCRIPT VS DASHBOARD DATA")
print("-"*80)

# From the manuscript:
manuscript_values = {
    'total_publications': 14142,  # "14,142 peer-reviewed publications"
    'biobanks': {
        'UK Biobank': 10752,      # "UK Biobank accounts for more than 75% (10,752 articles)"
        'FinnGen': 1806,          # "FinnGen (1,806)"
        'Estonian Biobank': 663,  # "Estonian Biobank (663)"
        'All of Us': 535,         # "All of Us (535)"
        'Million Veteran Program': 386,  # "Million Veteran Program (386)"
    },
    'year_range': '2000-2024',
    'num_biobanks_analyzed': 5,  # The paper analyzed 5 major biobanks
}

print("\nManuscript says (5 biobanks):")
for bb, count in manuscript_values['biobanks'].items():
    print(f"   {bb}: {count:,} publications")
print(f"   TOTAL: {manuscript_values['total_publications']:,}")

print("\nDashboard biobanks.json says (27 biobanks):")

# Find matching biobanks
for name, expected in manuscript_values['biobanks'].items():
    found = next((b for b in biobanks if name.lower() in b.get('name', '').lower()), None)
    if found:
        actual = found.get('stats', {}).get('totalPublications', 0)
        diff = actual - expected
        status = "✓" if abs(diff) < 100 else "✗"
        print(f"   {status} {name}: {actual:,} (expected {expected:,}, Δ={diff:+,})")
    else:
        print(f"   ✗ {name}: NOT FOUND")

#############################################################################
# 2. DISEASE BURDEN VALIDATION (from manuscript/GBD 2021)
#############################################################################
print("\n\n2. DISEASE BURDEN VALIDATION")
print("-"*80)

# Expected high-burden diseases that MUST have publications
# (Based on common diseases studied in biobank research)
expected_high_pub_diseases = [
    ('stroke', 'Stroke', 1000),
    ('diabetes', 'Type 2 Diabetes', 500),
    ('heart', 'Ischemic Heart Disease', 500),
    ('alzheimer', 'Alzheimer', 500),
    ('cancer', 'Cancer', 500),
    ('depression', 'Depression', 300),
    ('kidney', 'Chronic Kidney Disease', 200),
]

print("\nExpected high-publication diseases:")
for key, display, min_expected in expected_high_pub_diseases:
    found = next((d for d in diseases if key in d.get('name', '').lower()), None)
    if found:
        actual = found.get('research', {}).get('globalPublications', 0)
        status = "✓" if actual >= min_expected else "✗ UNDER-REPORTED"
        print(f"   {status} {found['name']}: {actual:,} (expected ≥{min_expected})")
    else:
        print(f"   ✗ {display}: NOT FOUND IN diseases.json")

#############################################################################
# 3. DATA CONSISTENCY CHECK
#############################################################################
print("\n\n3. INTERNAL CONSISTENCY CHECK")
print("-"*80)

# Total publications from different sources
biobanks_total = sum(b.get('stats', {}).get('totalPublications', 0) for b in biobanks)
diseases_total = sum(d.get('research', {}).get('globalPublications', 0) for d in diseases)
matrix_total = sum(sum(row) for row in matrix['matrix']['values'])
summary_total = summary.get('overview', {}).get('totalPublications', 0)

print(f"\nPublication totals from different sources:")
print(f"   summary.json:    {summary_total:>12,}")
print(f"   biobanks.json:   {biobanks_total:>12,} (sum of all biobanks)")
print(f"   diseases.json:   {diseases_total:>12,} (sum of all diseases)")  
print(f"   matrix.json:     {matrix_total:>12,} (sum of all cells)")
print(f"   manuscript:      {manuscript_values['total_publications']:>12,}")

print("\n   Analysis:")
if biobanks_total == summary_total:
    print("   ✓ biobanks.json matches summary.json")
else:
    print(f"   ✗ biobanks.json ≠ summary.json (Δ={biobanks_total-summary_total:+,})")

if diseases_total == matrix_total:
    print("   ✓ diseases.json matches matrix.json")
else:
    print(f"   ✗ diseases.json ≠ matrix.json (Δ={diseases_total-matrix_total:+,})")

if matrix_total == biobanks_total:
    print("   ✓ matrix covers all biobank publications")
else:
    coverage = 100 * matrix_total / biobanks_total if biobanks_total > 0 else 0
    print(f"   ✗ matrix only covers {coverage:.1f}% of biobank publications")

#############################################################################
# 4. BIOBANK COVERAGE CHECK
#############################################################################
print("\n\n4. BIOBANK COVERAGE ANALYSIS")
print("-"*80)

# The dashboard claims 27 biobanks, but manuscript analyzed only 5
print(f"\nDashboard includes {len(biobanks)} biobanks")
print(f"Manuscript analyzed 5 biobanks")

extra_biobanks = []
for b in biobanks:
    name = b.get('name', '')
    is_core = any(core.lower() in name.lower() for core in manuscript_values['biobanks'].keys())
    if not is_core:
        extra_biobanks.append((name, b.get('stats', {}).get('totalPublications', 0)))

print(f"\nExtra biobanks not in manuscript (n={len(extra_biobanks)}):")
for name, pubs in sorted(extra_biobanks, key=lambda x: -x[1])[:10]:
    print(f"   {name}: {pubs:,} publications")

#############################################################################
# 5. CRITICAL ISSUES SUMMARY
#############################################################################
print("\n" + "="*80)
print("CRITICAL ISSUES SUMMARY")
print("="*80)

issues = []

# Issue 1: Total mismatch
if biobanks_total != manuscript_values['total_publications']:
    issues.append(f"Total publications ({biobanks_total:,}) doesn't match manuscript ({manuscript_values['total_publications']:,})")

# Issue 2: Matrix incomplete
if matrix_total < biobanks_total * 0.5:
    issues.append(f"Matrix only contains {100*matrix_total/biobanks_total:.0f}% of claimed publications")

# Issue 3: Zero-publication diseases
zero_diseases = [d['name'] for d in diseases if d.get('research', {}).get('globalPublications', 0) == 0]
if zero_diseases:
    issues.append(f"{len(zero_diseases)} diseases have 0 publications: {', '.join(zero_diseases)}")

# Issue 4: Core biobank numbers don't match
for name, expected in manuscript_values['biobanks'].items():
    found = next((b for b in biobanks if name.lower() in b.get('name', '').lower()), None)
    if found:
        actual = found.get('stats', {}).get('totalPublications', 0)
        if abs(actual - expected) > expected * 0.2:  # More than 20% off
            issues.append(f"{name}: {actual:,} vs manuscript {expected:,}")

print(f"\n{len(issues)} CRITICAL ISSUES FOUND:\n")
for i, issue in enumerate(issues, 1):
    print(f"   {i}. {issue}")

print("\n" + "="*80)
print("RECOMMENDATIONS")  
print("="*80)

print("""
1. DATA SOURCE MISMATCH
   The dashboard appears to use DIFFERENT data than the manuscript.
   - Manuscript: 14,142 publications from 5 biobanks
   - Dashboard: 75,356 publications from 27 biobanks
   
   QUESTION: Is the dashboard supposed to show:
   a) The same 5 biobanks as the manuscript? 
   b) An expanded set of 27 biobanks?

2. MATRIX INCOMPLETENESS
   The matrix only contains ~30% of claimed publications.
   This suggests either:
   a) Bug in matrix generation
   b) Matrix shows disease-specific pubs only (not total)
   c) Different filtering criteria

3. ZERO-PUBLICATION DISEASES
   Type 2 Diabetes, Chronic Kidney Disease, Road Traffic Injuries
   show 0 publications - this is IMPOSSIBLE for major biobanks.
   
   LIKELY CAUSE: MeSH term mapping failed for these diseases.

4. RECOMMENDED ACTIONS:
   a) Clarify scope: 5 biobanks (manuscript) or 27 biobanks (expanded)?
   b) Regenerate matrix with verified MeSH mappings
   c) Cross-validate all numbers against PubMed queries
   d) Document data provenance
""")

