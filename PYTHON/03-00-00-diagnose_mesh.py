#!/usr/bin/env python3
"""
DIAGNOSTIC: Why are Diabetes, CKD, and Road Traffic showing 0 publications?
============================================================================
This script investigates the MeSH term mapping failures.

Run from project root:
    python PYTHON/diagnose_mesh.py

Author: HEIM Diagnostics
"""
import pandas as pd
from collections import Counter
from pathlib import Path

DATA_DIR = Path("DATA")

def main():
    print("=" * 70)
    print("MESH TERM MAPPING DIAGNOSTIC")
    print("=" * 70)
    
    # Load publications
    pub_file = DATA_DIR / "bhem_publications.csv"
    if not pub_file.exists():
        print(f"❌ File not found: {pub_file}")
        return
    
    df = pd.read_csv(pub_file)
    print(f"\n📂 Loaded {len(df):,} publications")
    print(f"   Columns: {list(df.columns)}")
    
    # Check mesh_terms column
    mesh_col = 'mesh_terms' if 'mesh_terms' in df.columns else 'MeSH_Terms'
    if mesh_col not in df.columns:
        print(f"❌ No mesh_terms column found!")
        return
    
    # Count publications with MeSH terms
    has_mesh = df[mesh_col].notna() & (df[mesh_col].str.strip() != '')
    print(f"\n📊 Publications with MeSH terms: {has_mesh.sum():,} ({has_mesh.sum()/len(df)*100:.1f}%)")
    
    # Sample MeSH terms
    print("\n" + "=" * 70)
    print("SAMPLE MESH TERMS (first 5 publications)")
    print("=" * 70)
    mesh_samples = df[has_mesh][mesh_col].head(5)
    for i, terms in enumerate(mesh_samples, 1):
        print(f"\n{i}. {terms[:300]}...")
    
    # =========================================================================
    # DIABETES INVESTIGATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔍 INVESTIGATING: TYPE 2 DIABETES")
    print("=" * 70)
    
    # Current registry terms (from 02-01-bhem-map-diseases.py)
    registry_diabetes = ['Diabetes Mellitus, Type 2', 'Diabetes Mellitus', 
                        'Diabetic Complications', 'Hyperglycemia', 'Insulin Resistance',
                        'Diabetic Nephropathies', 'Diabetic Retinopathy', 'Diabetic Neuropathies']
    
    print("\nRegistry MeSH terms for diabetes:")
    for term in registry_diabetes:
        print(f"  • {term}")
    
    print("\nSearching for diabetes-related terms in publications:")
    diabetes_searches = ['diabetes', 'diabetic', 'hyperglycemia', 'insulin', 'glucose']
    for search_term in diabetes_searches:
        matches = df[df[mesh_col].str.contains(search_term, case=False, na=False)]
        print(f"  '{search_term}': {len(matches):,} publications")
        if len(matches) > 0 and search_term == 'diabetes':
            # Show actual MeSH term format
            example = matches.iloc[0][mesh_col]
            diabetes_terms = [t for t in example.split(';') if 'diabet' in t.lower()]
            print(f"    → Actual format: {diabetes_terms[:3]}")
    
    # =========================================================================
    # CHRONIC KIDNEY DISEASE INVESTIGATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔍 INVESTIGATING: CHRONIC KIDNEY DISEASE")
    print("=" * 70)
    
    registry_ckd = ['Renal Insufficiency, Chronic', 'Kidney Failure, Chronic', 
                   'Diabetic Nephropathies', 'Glomerulonephritis']
    
    print("\nRegistry MeSH terms for CKD:")
    for term in registry_ckd:
        print(f"  • {term}")
    
    print("\n⚠️  NOTE: 'Chronic Kidney Disease' itself is MISSING from registry!")
    
    print("\nSearching for kidney-related terms in publications:")
    kidney_searches = ['kidney', 'renal', 'nephro', 'ckd', 'glomerul']
    for search_term in kidney_searches:
        matches = df[df[mesh_col].str.contains(search_term, case=False, na=False)]
        print(f"  '{search_term}': {len(matches):,} publications")
        if len(matches) > 0 and search_term == 'kidney':
            example = matches.iloc[0][mesh_col]
            kidney_terms = [t for t in example.split(';') if 'kidney' in t.lower() or 'renal' in t.lower()]
            print(f"    → Actual format: {kidney_terms[:3]}")
    
    # =========================================================================
    # ROAD TRAFFIC INJURIES INVESTIGATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔍 INVESTIGATING: ROAD TRAFFIC INJURIES")
    print("=" * 70)
    
    registry_traffic = ['Accidents, Traffic', 'Wounds and Injuries',
                       'Craniocerebral Trauma', 'Spinal Cord Injuries']
    
    print("\nRegistry MeSH terms for Road Traffic:")
    for term in registry_traffic:
        print(f"  • {term}")
    
    print("\nSearching for traffic/injury terms in publications:")
    traffic_searches = ['traffic', 'accident', 'injury', 'trauma', 'wounds', 'vehicle']
    for search_term in traffic_searches:
        matches = df[df[mesh_col].str.contains(search_term, case=False, na=False)]
        print(f"  '{search_term}': {len(matches):,} publications")
    
    # =========================================================================
    # TOP MESH TERMS ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 TOP 30 MOST COMMON MESH TERMS IN DATASET")
    print("=" * 70)
    
    all_terms = []
    for mesh_str in df[mesh_col].dropna():
        terms = [t.strip() for t in str(mesh_str).split(';')]
        all_terms.extend(terms)
    
    term_counts = Counter(all_terms)
    print(f"\nTotal unique MeSH terms: {len(term_counts):,}")
    print("\nTop 30:")
    for i, (term, count) in enumerate(term_counts.most_common(30), 1):
        # Highlight relevant terms
        highlight = ""
        if any(x in term.lower() for x in ['diabetes', 'kidney', 'renal', 'traffic', 'accident']):
            highlight = " ⭐ RELEVANT"
        print(f"  {i:2}. {term}: {count:,}{highlight}")
    
    # =========================================================================
    # MAPPING FUNCTION TEST
    # =========================================================================
    print("\n" + "=" * 70)
    print("🧪 TESTING MAPPING FUNCTION")
    print("=" * 70)
    
    # Test the normalize function
    def normalize_mesh_term(term):
        return term.lower().strip()
    
    # Test exact matching
    test_mesh = "Diabetes Mellitus; Humans; Insulin; Type 2 Diabetes"
    print(f"\nTest MeSH string: '{test_mesh}'")
    
    mesh_terms = [normalize_mesh_term(t) for t in test_mesh.split(';')]
    mesh_set = set(mesh_terms)
    print(f"Normalized terms: {mesh_set}")
    
    disease_term = normalize_mesh_term('Diabetes Mellitus, Type 2')
    print(f"Looking for: '{disease_term}'")
    print(f"Exact match? {disease_term in mesh_set}")
    
    # Check partial matching
    for article_term in mesh_terms:
        if disease_term in article_term or article_term in disease_term:
            print(f"Partial match found: '{article_term}'")
            break
    else:
        print("No partial match found!")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    print("""
Based on this diagnostic, likely issues are:

1. MISSING MESH TERMS IN REGISTRY:
   - Add 'Chronic Kidney Disease' to CKD
   - Add 'Diabetes Mellitus' (without 'Type 2' qualifier) 
   - Add 'Kidney Diseases' for broader CKD matching
   
2. CASE SENSITIVITY / EXACT MATCHING:
   - MeSH terms may have different formats in PubMed exports
   - E.g., 'Diabetes Mellitus, Type 2' vs 'Type 2 Diabetes Mellitus'
   
3. PARTIAL MATCHING LOGIC:
   - The matching algorithm may have edge cases
   - Need to verify substring matching works correctly

Run the full mapping pipeline again after fixing the registry.
""")
    
    print("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()