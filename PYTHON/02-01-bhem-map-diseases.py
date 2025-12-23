#!/usr/bin/env python3
"""
02-01-bhem-map-diseases.py
==========================
BHEM Step 2: Map publications to diseases using MeSH terms

Maps each publication to the 25 priority diseases based on MeSH term matching.
Adds disease columns to the publications dataset.

INPUT:  DATA/bhem_publications.csv
OUTPUT: DATA/bhem_publications_mapped.csv

USAGE:
    python 02-01-bhem-map-diseases.py
"""

import os
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"

INPUT_FILE = DATA_DIR / "bhem_publications.csv"
OUTPUT_FILE = DATA_DIR / "bhem_publications_mapped.csv"

# =============================================================================
# DISEASE REGISTRY (25 Priority Diseases)
# =============================================================================

DISEASE_REGISTRY = {
    # Cardiovascular
    'ischemic_heart_disease': {
        'name': 'Ischemic Heart Disease',
        'category': 'Cardiovascular',
        'mesh_terms': ['Myocardial Ischemia', 'Coronary Artery Disease', 'Coronary Disease',
                      'Angina Pectoris', 'Myocardial Infarction', 'Acute Coronary Syndrome',
                      'Coronary Stenosis', 'Heart Failure'],
        'dalys_millions': 185.0,
        'deaths_millions': 9.44,
        'prevalence_millions': 197.0,
        'global_south_priority': False
    },
    'stroke': {
        'name': 'Stroke',
        'category': 'Cardiovascular',
        'mesh_terms': ['Stroke', 'Cerebrovascular Disorders', 'Brain Infarction',
                      'Cerebral Infarction', 'Intracranial Hemorrhages', 'Brain Ischemia',
                      'Cerebral Hemorrhage', 'Subarachnoid Hemorrhage'],
        'dalys_millions': 143.0,
        'deaths_millions': 6.55,
        'prevalence_millions': 101.0,
        'global_south_priority': False
    },
    
    # Respiratory
    'copd': {
        'name': 'Chronic Obstructive Pulmonary Disease',
        'category': 'Respiratory',
        'mesh_terms': ['Pulmonary Disease, Chronic Obstructive', 'Chronic Bronchitis', 
                      'Emphysema', 'Bronchitis, Chronic', 'Pulmonary Emphysema'],
        'dalys_millions': 81.0,
        'deaths_millions': 3.23,
        'prevalence_millions': 212.0,
        'global_south_priority': False
    },
    'asthma': {
        'name': 'Asthma',
        'category': 'Respiratory',
        'mesh_terms': ['Asthma', 'Status Asthmaticus', 'Bronchial Hyperreactivity'],
        'dalys_millions': 22.0,
        'deaths_millions': 0.46,
        'prevalence_millions': 262.0,
        'global_south_priority': False
    },
    
    # Metabolic
    'diabetes': {
        'name': 'Type 2 Diabetes',
        'category': 'Metabolic',
        'mesh_terms': ['Diabetes Mellitus, Type 2', 'Diabetes Mellitus', 
                      'Diabetic Complications', 'Hyperglycemia', 'Insulin Resistance',
                      'Diabetic Nephropathies', 'Diabetic Retinopathy', 'Diabetic Neuropathies'],
        'dalys_millions': 67.0,
        'deaths_millions': 1.50,
        'prevalence_millions': 462.0,
        'global_south_priority': False
    },
    
    # Infectious (Global South Priority)
    'malaria': {
        'name': 'Malaria',
        'category': 'Infectious',
        'mesh_terms': ['Malaria', 'Plasmodium falciparum', 'Plasmodium vivax',
                      'Malaria, Cerebral', 'Antimalarials', 'Malaria, Falciparum'],
        'dalys_millions': 62.5,
        'deaths_millions': 0.62,
        'prevalence_millions': 247.0,
        'global_south_priority': True
    },
    'tuberculosis': {
        'name': 'Tuberculosis',
        'category': 'Infectious',
        'mesh_terms': ['Tuberculosis', 'Tuberculosis, Pulmonary', 'Mycobacterium tuberculosis',
                      'Tuberculosis, Multidrug-Resistant', 'Latent Tuberculosis'],
        'dalys_millions': 38.1,
        'deaths_millions': 1.30,
        'prevalence_millions': 10.6,
        'global_south_priority': True
    },
    'hiv_aids': {
        'name': 'HIV/AIDS',
        'category': 'Infectious',
        'mesh_terms': ['HIV Infections', 'Acquired Immunodeficiency Syndrome', 'HIV',
                      'HIV-1', 'AIDS-Related Opportunistic Infections', 'Anti-HIV Agents'],
        'dalys_millions': 42.0,
        'deaths_millions': 0.68,
        'prevalence_millions': 38.4,
        'global_south_priority': True
    },
    'diarrheal_diseases': {
        'name': 'Diarrheal Diseases',
        'category': 'Infectious',
        'mesh_terms': ['Diarrhea', 'Dysentery', 'Cholera', 'Rotavirus Infections',
                      'Gastroenteritis', 'Dehydration', 'Shigella', 'Cryptosporidiosis'],
        'dalys_millions': 44.2,
        'deaths_millions': 1.53,
        'prevalence_millions': 1700.0,
        'global_south_priority': True
    },
    
    # Neglected Tropical Diseases
    'ntds': {
        'name': 'Neglected Tropical Diseases',
        'category': 'Neglected',
        'mesh_terms': ['Neglected Diseases', 'Schistosomiasis', 'Leishmaniasis',
                      'Onchocerciasis', 'Lymphatic Filariasis', 'Chagas Disease',
                      'Dengue', 'Trypanosomiasis', 'Soil-Transmitted Helminthiasis',
                      'Trachoma', 'Leprosy', 'Rabies'],
        'dalys_millions': 28.3,
        'deaths_millions': 0.17,
        'prevalence_millions': 1500.0,
        'global_south_priority': True
    },
    
    # Neurological
    'alzheimers': {
        'name': "Alzheimer's Disease",
        'category': 'Neurological',
        'mesh_terms': ['Alzheimer Disease', 'Dementia', 'Cognitive Dysfunction',
                      'Tauopathies', 'Amyloid beta-Peptides', 'Dementia, Vascular'],
        'dalys_millions': 30.0,
        'deaths_millions': 1.62,
        'prevalence_millions': 55.0,
        'global_south_priority': False
    },
    'epilepsy': {
        'name': 'Epilepsy',
        'category': 'Neurological',
        'mesh_terms': ['Epilepsy', 'Seizures', 'Epilepsy, Generalized',
                      'Epilepsy, Temporal Lobe', 'Status Epilepticus'],
        'dalys_millions': 15.0,
        'deaths_millions': 0.13,
        'prevalence_millions': 50.0,
        'global_south_priority': False
    },
    'parkinsons': {
        'name': "Parkinson's Disease",
        'category': 'Neurological',
        'mesh_terms': ['Parkinson Disease', 'Parkinsonian Disorders', 'Lewy Body Disease'],
        'dalys_millions': 6.0,
        'deaths_millions': 0.33,
        'prevalence_millions': 8.5,
        'global_south_priority': False
    },
    
    # Mental Health
    'depression': {
        'name': 'Depression',
        'category': 'Mental Health',
        'mesh_terms': ['Depressive Disorder', 'Depression', 'Depressive Disorder, Major',
                      'Mood Disorders', 'Depressive Disorder, Treatment-Resistant'],
        'dalys_millions': 50.0,
        'deaths_millions': 0.0,
        'prevalence_millions': 280.0,
        'global_south_priority': False
    },
    'anxiety': {
        'name': 'Anxiety Disorders',
        'category': 'Mental Health',
        'mesh_terms': ['Anxiety Disorders', 'Anxiety', 'Panic Disorder',
                      'Phobic Disorders', 'Obsessive-Compulsive Disorder'],
        'dalys_millions': 28.0,
        'deaths_millions': 0.0,
        'prevalence_millions': 301.0,
        'global_south_priority': False
    },
    'bipolar': {
        'name': 'Bipolar Disorder',
        'category': 'Mental Health',
        'mesh_terms': ['Bipolar Disorder', 'Mania', 'Cyclothymic Disorder'],
        'dalys_millions': 12.0,
        'deaths_millions': 0.0,
        'prevalence_millions': 40.0,
        'global_south_priority': False
    },
    'schizophrenia': {
        'name': 'Schizophrenia',
        'category': 'Mental Health',
        'mesh_terms': ['Schizophrenia', 'Psychotic Disorders', 'Schizophrenia Spectrum and Other Psychotic Disorders'],
        'dalys_millions': 17.0,
        'deaths_millions': 0.0,
        'prevalence_millions': 24.0,
        'global_south_priority': False
    },
    
    # Cancers
    'lung_cancer': {
        'name': 'Lung Cancer',
        'category': 'Cancer',
        'mesh_terms': ['Lung Neoplasms', 'Carcinoma, Non-Small-Cell Lung',
                      'Small Cell Lung Carcinoma', 'Carcinoma, Bronchogenic'],
        'dalys_millions': 45.0,
        'deaths_millions': 1.80,
        'prevalence_millions': 2.2,
        'global_south_priority': False
    },
    'breast_cancer': {
        'name': 'Breast Cancer',
        'category': 'Cancer',
        'mesh_terms': ['Breast Neoplasms', 'Carcinoma, Ductal, Breast',
                      'Triple Negative Breast Neoplasms', 'Breast Carcinoma In Situ'],
        'dalys_millions': 18.0,
        'deaths_millions': 0.68,
        'prevalence_millions': 7.8,
        'global_south_priority': False
    },
    
    # Other NCDs
    'chronic_kidney': {
        'name': 'Chronic Kidney Disease',
        'category': 'Renal',
        'mesh_terms': ['Renal Insufficiency, Chronic', 'Kidney Failure, Chronic', 
                      'Diabetic Nephropathies', 'Glomerulonephritis'],
        'dalys_millions': 35.0,
        'deaths_millions': 1.40,
        'prevalence_millions': 843.0,
        'global_south_priority': False
    },
    'cirrhosis': {
        'name': 'Cirrhosis',
        'category': 'Hepatic',
        'mesh_terms': ['Liver Cirrhosis', 'Fibrosis', 'Fatty Liver',
                      'Non-alcoholic Fatty Liver Disease', 'Hepatitis, Alcoholic'],
        'dalys_millions': 25.0,
        'deaths_millions': 1.32,
        'prevalence_millions': 112.0,
        'global_south_priority': False
    },
    
    # Musculoskeletal
    'low_back_pain': {
        'name': 'Low Back Pain',
        'category': 'Musculoskeletal',
        'mesh_terms': ['Low Back Pain', 'Back Pain', 'Intervertebral Disc Degeneration',
                      'Sciatica', 'Spinal Stenosis'],
        'dalys_millions': 63.0,
        'deaths_millions': 0.0,
        'prevalence_millions': 568.0,
        'global_south_priority': False
    },
    
    # Injuries
    'road_traffic': {
        'name': 'Road Traffic Injuries',
        'category': 'Injuries',
        'mesh_terms': ['Accidents, Traffic', 'Wounds and Injuries',
                      'Craniocerebral Trauma', 'Spinal Cord Injuries'],
        'dalys_millions': 85.5,
        'deaths_millions': 1.35,
        'prevalence_millions': 50.0,
        'global_south_priority': True
    },
    
    # Maternal/Neonatal
    'neonatal_disorders': {
        'name': 'Neonatal Disorders',
        'category': 'Maternal',
        'mesh_terms': ['Infant, Premature, Diseases', 'Neonatal Sepsis',
                      'Respiratory Distress Syndrome, Newborn', 'Asphyxia Neonatorum',
                      'Jaundice, Neonatal', 'Infant, Low Birth Weight'],
        'dalys_millions': 75.0,
        'deaths_millions': 2.40,
        'prevalence_millions': 0.0,
        'global_south_priority': True
    },
    'preterm_birth': {
        'name': 'Preterm Birth Complications',
        'category': 'Maternal',
        'mesh_terms': ['Premature Birth', 'Infant, Premature',
                      'Bronchopulmonary Dysplasia', 'Retinopathy of Prematurity'],
        'dalys_millions': 23.0,
        'deaths_millions': 0.90,
        'prevalence_millions': 15.0,
        'global_south_priority': True
    },
}


# =============================================================================
# DISEASE MAPPING FUNCTIONS
# =============================================================================

def normalize_mesh_term(term: str) -> str:
    """Normalize a MeSH term for matching."""
    return term.lower().strip()


def map_publication_to_diseases(mesh_terms_str: str) -> list:
    """Map a publication's MeSH terms to diseases."""
    if pd.isna(mesh_terms_str) or not mesh_terms_str.strip():
        return []
    
    # Parse MeSH terms
    mesh_terms = [normalize_mesh_term(t) for t in mesh_terms_str.split(';')]
    mesh_set = set(mesh_terms)
    
    matched_diseases = []
    
    for disease_id, disease in DISEASE_REGISTRY.items():
        # Normalize disease MeSH terms
        disease_terms = [normalize_mesh_term(t) for t in disease['mesh_terms']]
        
        # Check for any match
        for disease_term in disease_terms:
            # Exact match
            if disease_term in mesh_set:
                matched_diseases.append(disease_id)
                break
            # Partial match (disease term contained in article term)
            for article_term in mesh_terms:
                if disease_term in article_term or article_term in disease_term:
                    matched_diseases.append(disease_id)
                    break
            else:
                continue
            break
    
    return list(set(matched_diseases))


def get_disease_categories(disease_ids: list) -> list:
    """Get categories for matched diseases."""
    categories = []
    for disease_id in disease_ids:
        if disease_id in DISEASE_REGISTRY:
            categories.append(DISEASE_REGISTRY[disease_id]['category'])
    return list(set(categories))


def is_global_south_priority(disease_ids: list) -> bool:
    """Check if any matched disease is a Global South priority."""
    for disease_id in disease_ids:
        if disease_id in DISEASE_REGISTRY:
            if DISEASE_REGISTRY[disease_id].get('global_south_priority', False):
                return True
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BHEM STEP 2: Map Publications to Diseases")
    print("=" * 70)
    
    # Check input file
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        print(f"   Run 02-00-bhem-fetch-pubmed.py first")
        return
    
    # Load publications
    print(f"\n📂 Loading publications from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df):,} publications")
    
    # Map diseases
    print(f"\n🔬 Mapping publications to {len(DISEASE_REGISTRY)} diseases...")
    
    df['disease_ids'] = df['mesh_terms'].apply(map_publication_to_diseases)
    df['disease_count'] = df['disease_ids'].apply(len)
    df['disease_categories'] = df['disease_ids'].apply(get_disease_categories)
    df['global_south_priority'] = df['disease_ids'].apply(is_global_south_priority)
    
    # Convert lists to strings for CSV storage
    df['disease_ids_str'] = df['disease_ids'].apply(lambda x: '|'.join(x) if x else '')
    df['disease_categories_str'] = df['disease_categories'].apply(lambda x: '|'.join(x) if x else '')
    
    # Statistics
    mapped_count = (df['disease_count'] > 0).sum()
    unmapped_count = (df['disease_count'] == 0).sum()
    global_south_count = df['global_south_priority'].sum()
    
    print(f"\n📊 Mapping Results:")
    print(f"   Publications mapped to diseases: {mapped_count:,} ({mapped_count/len(df)*100:.1f}%)")
    print(f"   Publications not mapped: {unmapped_count:,} ({unmapped_count/len(df)*100:.1f}%)")
    print(f"   Global South priority topics: {global_south_count:,} ({global_south_count/len(df)*100:.1f}%)")
    
    # Disease coverage
    print(f"\n📋 Publications per disease:")
    disease_counts = defaultdict(int)
    for disease_list in df['disease_ids']:
        for disease_id in disease_list:
            disease_counts[disease_id] += 1
    
    # Sort by count
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    for disease_id, count in sorted_diseases:
        name = DISEASE_REGISTRY[disease_id]['name']
        category = DISEASE_REGISTRY[disease_id]['category']
        gs = "🌍" if DISEASE_REGISTRY[disease_id].get('global_south_priority') else ""
        print(f"   {name}: {count:,} {gs}")
    
    # Diseases with zero coverage
    zero_coverage = [d for d in DISEASE_REGISTRY.keys() if disease_counts.get(d, 0) == 0]
    if zero_coverage:
        print(f"\n⚠️  Diseases with ZERO publications:")
        for disease_id in zero_coverage:
            name = DISEASE_REGISTRY[disease_id]['name']
            print(f"   - {name}")
    
    # Category distribution
    print(f"\n📊 Publications by category:")
    category_counts = defaultdict(int)
    for categories in df['disease_categories']:
        for cat in categories:
            category_counts[cat] += 1
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count:,}")
    
    # Save output
    print(f"\n💾 Saving mapped publications to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Also save disease registry as JSON for later steps
    import json
    disease_registry_file = DATA_DIR / "disease_registry.json"
    with open(disease_registry_file, 'w') as f:
        json.dump(DISEASE_REGISTRY, f, indent=2)
    print(f"   Disease registry saved to {disease_registry_file}")
    
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Next step: python 02-02-bhem-compute-metrics.py")


if __name__ == "__main__":
    main()