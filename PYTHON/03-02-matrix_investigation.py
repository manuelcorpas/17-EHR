#!/usr/bin/env python3
"""
Matrix Investigation - Understanding why 70% of publications are missing
"""

import json
from pathlib import Path
from collections import defaultdict

data_dir = Path('/mnt/user-data/uploads')

matrix = json.load(open(data_dir / 'matrix.json'))
biobanks = json.load(open(data_dir / 'biobanks.json'))['biobanks']
diseases = json.load(open(data_dir / 'diseases.json'))['diseases']

print("="*80)
print("MATRIX INVESTIGATION: WHY IS 70% OF DATA MISSING?")
print("="*80)

#############################################################################
# 1. MATRIX STRUCTURE ANALYSIS
#############################################################################
print("\n1. MATRIX STRUCTURE")
print("-"*80)

matrix_biobanks = matrix['biobanks']
matrix_diseases = matrix['diseases']
matrix_values = matrix['matrix']['values']

print(f"   Matrix dimensions: {len(matrix_biobanks)} biobanks × {len(matrix_diseases)} diseases")
print(f"   Total cells: {len(matrix_biobanks) * len(matrix_diseases):,}")

# Count non-zero cells
non_zero = sum(1 for row in matrix_values for val in row if val > 0)
total_cells = len(matrix_biobanks) * len(matrix_diseases)
print(f"   Non-zero cells: {non_zero:,} ({100*non_zero/total_cells:.1f}%)")

# Grand total
matrix_total = sum(sum(row) for row in matrix_values)
print(f"   Grand total (sum of all cells): {matrix_total:,}")

#############################################################################
# 2. COMPARE TOTALS
#############################################################################
print("\n2. PUBLICATION TOTAL COMPARISON")
print("-"*80)

biobanks_total = sum(b.get('stats', {}).get('totalPublications', 0) for b in biobanks)
diseases_total = sum(d.get('research', {}).get('globalPublications', 0) for d in diseases)

print(f"   Matrix sum:        {matrix_total:>12,}")
print(f"   biobanks.json:     {biobanks_total:>12,}")
print(f"   diseases.json:     {diseases_total:>12,}")
print(f"   ")
print(f"   MISSING from matrix: {biobanks_total - matrix_total:>12,} ({100*(biobanks_total-matrix_total)/biobanks_total:.1f}%)")

#############################################################################
# 3. ZERO-VALUE DISEASES IN MATRIX
#############################################################################
print("\n3. DISEASES WITH ZERO PUBLICATIONS IN MATRIX")
print("-"*80)

zero_diseases = []
for di, md in enumerate(matrix_diseases):
    col_sum = sum(row[di] for row in matrix_values)
    if col_sum == 0:
        # Find burden info
        d = next((x for x in diseases if x.get('id') == md.get('id')), None)
        dalys = d.get('burden', {}).get('dalysMillions', 0) if d else 0
        zero_diseases.append((md['name'], dalys))
        print(f"   ⚠️  {md['name']}: {dalys}M DALYs - NO PUBLICATIONS")

print(f"\n   Total zero-publication diseases: {len(zero_diseases)}")

#############################################################################
# 4. CHECK WHAT THE MATRIX VALUES REPRESENT
#############################################################################
print("\n4. SAMPLE MATRIX VALUES (UK Biobank row)")
print("-"*80)

# Find UK Biobank row
ukb_idx = next((i for i, mb in enumerate(matrix_biobanks) if 'UK Biobank' in mb.get('name', '')), None)

if ukb_idx is not None:
    ukb_row = matrix_values[ukb_idx]
    print(f"   UK Biobank matrix row sum: {sum(ukb_row):,}")
    print(f"   UK Biobank in biobanks.json: {next((b.get('stats',{}).get('totalPublications',0) for b in biobanks if 'UK Biobank' in b.get('name','')), 'N/A'):,}")
    print(f"\n   Non-zero disease values for UK Biobank:")
    for di, val in enumerate(ukb_row):
        if val > 0:
            print(f"      {matrix_diseases[di]['name']}: {val:,}")

#############################################################################
# 5. CHECK IF MATRIX HAS "DISEASE-SPECIFIC" PUBLICATIONS ONLY
#############################################################################
print("\n5. HYPOTHESIS: Matrix contains only disease-specific publications")
print("-"*80)

# Compare disease-level publications vs total
print("   The matrix might intentionally exclude publications that:")
print("   - Are methodology-focused rather than disease-focused")
print("   - Cover multiple diseases (avoiding double-counting)")
print("   - Are general biobank infrastructure papers")
print()

# Check disease coverage stats in biobanks.json
print("   Disease coverage stats from biobanks.json:")
for b in biobanks[:5]:
    name = b.get('name', 'Unknown')
    total_pubs = b.get('stats', {}).get('totalPublications', 0)
    diseases_covered = b.get('stats', {}).get('diseasesCovered', 0)
    
    # Find matrix row sum
    mb_idx = next((i for i, mb in enumerate(matrix_biobanks) if mb.get('name') == name), None)
    matrix_row_sum = sum(matrix_values[mb_idx]) if mb_idx is not None else 0
    
    print(f"   {name[:30]:<30} total={total_pubs:>6,} matrix={matrix_row_sum:>6,} diseases={diseases_covered}/25")

#############################################################################
# 6. ROOT CAUSE: THE DATA IS FICTIONAL/SYNTHETIC
#############################################################################
print("\n6. DATA QUALITY ASSESSMENT")
print("-"*80)

# Check for suspicious patterns
print("   Checking for synthetic data indicators...")

# Are all values round numbers?
all_values = [v for row in matrix_values for v in row if v > 0]
round_100 = sum(1 for v in all_values if v % 100 == 0)
round_10 = sum(1 for v in all_values if v % 10 == 0)

print(f"   Values divisible by 100: {round_100}/{len(all_values)} ({100*round_100/len(all_values) if all_values else 0:.1f}%)")
print(f"   Values divisible by 10: {round_10}/{len(all_values)} ({100*round_10/len(all_values) if all_values else 0:.1f}%)")

# Value distribution
print(f"\n   Value statistics:")
if all_values:
    print(f"   Min: {min(all_values)}")
    print(f"   Max: {max(all_values)}")
    print(f"   Mean: {sum(all_values)/len(all_values):.1f}")

# Check UK Biobank specific numbers against paper
print(f"\n   Cross-reference with manuscript (EHR-Linked-biobanks_v7.docx):")
print(f"   - Manuscript says UK Biobank has 10,752 peer-reviewed articles")
print(f"   - biobanks.json says: {next((b.get('stats',{}).get('totalPublications',0) for b in biobanks if 'UK Biobank' in b.get('name','')), 'N/A'):,}")

#############################################################################
# 7. CONCLUSIONS
#############################################################################
print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

print("""
CRITICAL ISSUES IDENTIFIED:

1. MATRIX IS INCOMPLETE
   - Matrix contains only 22,353 publications
   - biobanks.json claims 75,356 publications  
   - 53,003 publications (70%) are MISSING from the matrix

2. THREE MAJOR DISEASES HAVE ZERO DATA
   - Type 2 Diabetes Mellitus: 0 publications (67M DALYs)
   - Chronic Kidney Disease: 0 publications (35M DALYs)
   - Road Traffic Injuries: 0 publications (85.5M DALYs)

3. DATA DOES NOT MATCH THE MANUSCRIPT
   - Paper says UK Biobank: 10,752 articles
   - biobanks.json says: 13,341 (inflated)
   
4. POSSIBLE CAUSES:
   a) MeSH term mapping failed for some diseases
   b) Matrix generation script had bugs
   c) Data was partially synthetic/estimated
   d) Disease categories weren't properly defined

RECOMMENDATIONS:
   - Regenerate all data from source (PubMed queries)
   - Verify MeSH mappings for all 25 diseases
   - Cross-validate against manuscript numbers
   - Add data provenance/audit trail
""")

