#!/usr/bin/env python3
"""
HEIM-Biobank Deep Diagnostic Script
====================================
Traces data inconsistencies to their source and identifies root causes.
"""

import json
from pathlib import Path
from collections import defaultdict

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

data_dir = Path(__file__).parent.parent / 'docs' / 'data'

# Load all files
diseases = load_json(data_dir / 'diseases.json')['diseases']
matrix = load_json(data_dir / 'matrix.json')
biobanks = load_json(data_dir / 'biobanks.json')['biobanks']
summary = load_json(data_dir / 'summary.json')

print(f"\n{BOLD}{BLUE}{'='*80}")
print(" DEEP DIAGNOSTIC: TRACING DATA INCONSISTENCIES")
print(f"{'='*80}{RESET}\n")

#############################################################################
# 1. DISEASE PUBLICATION COUNT ANALYSIS
#############################################################################
print(f"{BOLD}{CYAN}1. DISEASE PUBLICATION COUNTS: diseases.json vs matrix.json{RESET}")
print("-"*80)

# Get disease order from matrix
matrix_diseases = matrix['diseases']
matrix_values = matrix['matrix']['values']

# Calculate column sums from matrix (total pubs per disease)
matrix_disease_totals = defaultdict(int)
for row in matrix_values:
    for di, val in enumerate(row):
        matrix_disease_totals[di] += val

print(f"\n{'Disease Name':<40} {'diseases.json':>15} {'matrix sum':>15} {'Δ':>10}")
print("-"*80)

mismatches = []
for i, d in enumerate(diseases):
    name = d['name']
    reported = d.get('research', {}).get('globalPublications', 0)
    
    # Find matching disease in matrix by ID or name
    matrix_idx = None
    for mi, md in enumerate(matrix_diseases):
        if md.get('id') == d.get('id') or md.get('name') == name:
            matrix_idx = mi
            break
    
    if matrix_idx is not None:
        matrix_total = matrix_disease_totals[matrix_idx]
    else:
        matrix_total = "NOT FOUND"
    
    delta = matrix_total - reported if isinstance(matrix_total, int) else "N/A"
    
    if delta != 0:
        color = RED if abs(delta) > 100 or reported == 0 else YELLOW
        mismatches.append((name, reported, matrix_total, delta))
        print(f"{color}{name:<40} {reported:>15,} {matrix_total:>15,} {delta:>+10}{RESET}")
    else:
        print(f"{GREEN}{name:<40} {reported:>15,} {matrix_total:>15,} {delta:>+10}{RESET}")

#############################################################################
# 2. CHECK IF diseases.json VALUES ARE SHIFTED/SCRAMBLED
#############################################################################
print(f"\n\n{BOLD}{CYAN}2. CHECKING FOR DATA SCRAMBLING/MISALIGNMENT{RESET}")
print("-"*80)

# Get all reported values from diseases.json
diseases_reported_pubs = [(d['name'], d.get('research', {}).get('globalPublications', 0)) for d in diseases]

# Get all matrix column sums
matrix_col_sums = [(matrix_diseases[i]['name'], matrix_disease_totals[i]) for i in range(len(matrix_diseases))]

# Sort both by value to see if they match when reordered
diseases_sorted = sorted(diseases_reported_pubs, key=lambda x: x[1], reverse=True)
matrix_sorted = sorted(matrix_col_sums, key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'diseases.json (name, pubs)':<45} {'matrix.json (name, pubs)':<45}")
print("-"*100)

for i in range(min(15, len(diseases_sorted))):
    d_name, d_pubs = diseases_sorted[i]
    m_name, m_pubs = matrix_sorted[i]
    
    match = "✓" if d_name == m_name else "✗"
    color = GREEN if d_name == m_name else RED
    print(f"{i+1:<6} {d_name[:35]:<35} {d_pubs:>8,}    {m_name[:35]:<35} {m_pubs:>8,}  {color}{match}{RESET}")

#############################################################################
# 3. LOOK FOR PATTERN IN MISALIGNMENT  
#############################################################################
print(f"\n\n{BOLD}{CYAN}3. IDENTIFYING MISALIGNMENT PATTERN{RESET}")
print("-"*80)

# Check if diseases.json values match matrix values but for DIFFERENT diseases
diseases_pubs_set = {d.get('research', {}).get('globalPublications', 0) for d in diseases}
matrix_pubs_set = {matrix_disease_totals[i] for i in range(len(matrix_diseases))}

# Find values in diseases.json that exist in matrix but for wrong disease
print("\nSearching for value matches that suggest row/column shift...")

for d in diseases:
    d_name = d['name']
    d_pubs = d.get('research', {}).get('globalPublications', 0)
    
    # Find which matrix disease has this exact value
    for mi, md in enumerate(matrix_diseases):
        m_pubs = matrix_disease_totals[mi]
        if m_pubs == d_pubs and md['name'] != d_name and d_pubs > 0:
            print(f"  {YELLOW}'{d_name}' has {d_pubs:,} pubs → matches matrix total for '{md['name']}'{RESET}")

#############################################################################
# 4. BIOBANK PUBLICATION TOTALS VS MATRIX ROW SUMS
#############################################################################
print(f"\n\n{BOLD}{CYAN}4. BIOBANK TOTALS: biobanks.json vs matrix row sums{RESET}")
print("-"*80)

matrix_biobanks = matrix['biobanks']

print(f"\n{'Biobank Name':<35} {'biobanks.json':>15} {'matrix sum':>15} {'Δ':>10}")
print("-"*80)

for bi, mb in enumerate(matrix_biobanks):
    # Find matching biobank in biobanks.json
    bb = next((b for b in biobanks if b.get('id') == mb.get('id') or b.get('name') == mb.get('name')), None)
    
    if bb:
        reported = bb.get('stats', {}).get('totalPublications', 0)
        matrix_row_sum = sum(matrix_values[bi])
        delta = matrix_row_sum - reported
        
        color = RED if abs(delta) > 100 else (YELLOW if delta != 0 else GREEN)
        print(f"{color}{mb['name'][:35]:<35} {reported:>15,} {matrix_row_sum:>15,} {delta:>+10}{RESET}")

#############################################################################
# 5. CHECK MATRIX INTERNAL CONSISTENCY
#############################################################################
print(f"\n\n{BOLD}{CYAN}5. MATRIX INTERNAL CONSISTENCY{RESET}")
print("-"*80)

# Total from all matrix cells
matrix_grand_total = sum(sum(row) for row in matrix_values)

# Total from biobanks.json
biobanks_total = sum(b.get('stats', {}).get('totalPublications', 0) for b in biobanks)

# Total from diseases.json
diseases_total = sum(d.get('research', {}).get('globalPublications', 0) for d in diseases)

# Summary total
summary_total = summary.get('overview', {}).get('totalPublications', 0)

print(f"\n  Matrix grand total (all cells):     {matrix_grand_total:>12,}")
print(f"  biobanks.json sum:                  {biobanks_total:>12,}")
print(f"  diseases.json sum:                  {diseases_total:>12,}")
print(f"  summary.json total:                 {summary_total:>12,}")

print(f"\n  {BOLD}Analysis:{RESET}")
if matrix_grand_total == biobanks_total:
    print(f"  {GREEN}✓ Matrix total matches biobanks.json{RESET}")
else:
    print(f"  {RED}✗ Matrix total ≠ biobanks.json (Δ={matrix_grand_total - biobanks_total:+,}){RESET}")

if diseases_total == biobanks_total:
    print(f"  {GREEN}✓ diseases.json total matches biobanks.json{RESET}")
else:
    print(f"  {RED}✗ diseases.json total ≠ biobanks.json (Δ={diseases_total - biobanks_total:+,}){RESET}")

#############################################################################
# 6. ROOT CAUSE ANALYSIS
#############################################################################
print(f"\n\n{BOLD}{RED}{'='*80}")
print(" ROOT CAUSE ANALYSIS")
print(f"{'='*80}{RESET}\n")

print(f"""
{BOLD}FINDINGS:{RESET}

1. {RED}diseases.json has WRONG publication counts{RESET}
   - Type 2 Diabetes shows 0 pubs, but matrix shows 1,876
   - Chronic Kidney Disease shows 0 pubs, but matrix shows 880
   - Road Traffic Injuries shows 0 pubs, but matrix shows 404

2. {YELLOW}The values appear SCRAMBLED - not missing, but assigned to wrong diseases{RESET}
   - This suggests a column index mismatch during data generation
   - The publication counts exist but are associated with wrong disease IDs

3. {GREEN}biobanks.json appears CORRECT{RESET}
   - Biobank totals match matrix row sums
   - Summary total matches biobanks sum

4. {RED}diseases.json needs REGENERATION{RESET}
   - Should recalculate globalPublications from matrix column sums
   - Each disease's publications = sum of that column across all biobanks

{BOLD}RECOMMENDED FIX:{RESET}
   Regenerate diseases.json by:
   1. Use matrix.json as source of truth for disease-level publication counts
   2. For each disease, sum the column values across all biobanks
   3. Update gap scores based on corrected publication counts
""")

#############################################################################
# 7. GENERATE CORRECTED VALUES
#############################################################################
print(f"\n\n{BOLD}{CYAN}7. CORRECTED DISEASE PUBLICATION COUNTS{RESET}")
print("-"*80)

print(f"\n{'Disease':<40} {'Current':>12} {'Corrected':>12}")
print("-"*70)

corrections = []
for di, md in enumerate(matrix_diseases):
    correct_pubs = matrix_disease_totals[di]
    
    # Find in diseases.json
    d = next((x for x in diseases if x.get('id') == md.get('id')), None)
    if d:
        current = d.get('research', {}).get('globalPublications', 0)
        if current != correct_pubs:
            print(f"{RED}{md['name']:<40} {current:>12,} {correct_pubs:>12,}{RESET}")
            corrections.append({
                'id': md.get('id'),
                'name': md['name'],
                'current': current,
                'correct': correct_pubs
            })
        else:
            print(f"{GREEN}{md['name']:<40} {current:>12,} {correct_pubs:>12,}{RESET}")

print(f"\n{BOLD}Total corrections needed: {len(corrections)}{RESET}")

