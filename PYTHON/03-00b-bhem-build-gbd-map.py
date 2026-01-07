#!/usr/bin/env python3
"""
03-00b-bhem-build-gbd-map.py
============================
Build semantic similarity mapping between our GBD cause keys and IHME cause names.

Uses sentence-transformers to compute embeddings and find best matches.
Run this ONCE to generate the mapping file, then use it in 03-01.

INPUT:
    DATA/IHMEGBD_2021_DATA*.csv (IHME GBD export)
    
OUTPUT:
    DATA/gbd_cause_mapping.json (our_key -> ihme_name with similarity scores)

USAGE:
    pip install sentence-transformers
    python 03-00b-bhem-build-gbd-map.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"

OUTPUT_FILE = DATA_DIR / "gbd_cause_mapping.json"

# Minimum similarity threshold for automatic matching
MIN_SIMILARITY = 0.65

# Model for semantic similarity (small, fast, good for medical terms)
MODEL_NAME = "all-MiniLM-L6-v2"

# =============================================================================
# OUR GBD CAUSE KEYS (from GBD_MESH_MAPPING)
# =============================================================================

OUR_CAUSE_KEYS = [
    # HIV/AIDS and STIs
    'HIV/AIDS',
    'Sexually transmitted infections excluding HIV',
    
    # Respiratory Infections
    'Lower respiratory infections',
    'Upper respiratory infections',
    'Otitis media',
    'COVID-19',
    'Tuberculosis',
    
    # Enteric Infections
    'Diarrheal diseases',
    'Typhoid and paratyphoid',
    'Invasive Non-typhoidal Salmonella (iNTS)',
    'Other intestinal infectious diseases',
    
    # NTDs and Malaria
    'Malaria',
    'Dengue',
    'Yellow fever',
    'Rabies',
    'Intestinal nematode infections',
    'Schistosomiasis',
    'Leishmaniasis',
    'Lymphatic filariasis',
    'Onchocerciasis',
    'Trachoma',
    'Chagas disease',
    'African trypanosomiasis',
    'Cysticercosis',
    'Cystic echinococcosis',
    'Leprosy',
    'Food-borne trematodiases',
    'Other neglected tropical diseases',
    'Ebola',
    'Zika virus',
    'Guinea worm disease',
    
    # Other Infectious
    'Meningitis',
    'Encephalitis',
    'Acute hepatitis',
    'Measles',
    'Tetanus',
    'Pertussis',
    'Diphtheria',
    'Varicella and herpes zoster',
    'Other unspecified infectious diseases',
    
    # Maternal/Neonatal
    'Maternal disorders',
    'Neonatal disorders',
    
    # Nutritional
    'Protein-energy malnutrition',
    'Dietary iron deficiency',
    'Iodine deficiency',
    'Vitamin A deficiency',
    'Other nutritional deficiencies',
    
    # Cancers
    'Breast cancer',
    'Cervical cancer',
    'Uterine cancer',
    'Ovarian cancer',
    'Prostate cancer',
    'Testicular cancer',
    'Colon and rectum cancer',
    'Lip and oral cavity cancer',
    'Nasopharynx cancer',
    'Other pharynx cancer',
    'Esophageal cancer',
    'Stomach cancer',
    'Liver cancer',
    'Gallbladder and biliary tract cancer',
    'Pancreatic cancer',
    'Larynx cancer',
    'Tracheal, bronchus, and lung cancer',
    'Malignant skin melanoma',
    'Non-melanoma skin cancer',
    'Kidney cancer',
    'Bladder cancer',
    'Brain and central nervous system cancer',
    'Thyroid cancer',
    'Mesothelioma',
    'Hodgkin lymphoma',
    'Non-Hodgkin lymphoma',
    'Multiple myeloma',
    'Leukemia',
    'Other neoplasms',
    'Other malignant neoplasms',
    'Malignant neoplasm of bone and articular cartilage',
    'Soft tissue and other extraosseous sarcomas',
    'Eye cancer',
    'Neuroblastoma and other peripheral nervous cell tumors',
    
    # Cardiovascular
    'Ischemic heart disease',
    'Stroke',
    'Hypertensive heart disease',
    'Rheumatic heart disease',
    'Non-rheumatic valvular heart disease',
    'Cardiomyopathy and myocarditis',
    'Atrial fibrillation and flutter',
    'Aortic aneurysm',
    'Peripheral artery disease',
    'Endocarditis',
    'Other cardiovascular and circulatory diseases',
    
    # Chronic Respiratory
    'Chronic obstructive pulmonary disease',
    'Asthma',
    'Pneumoconiosis',
    'Interstitial lung disease and pulmonary sarcoidosis',
    'Other chronic respiratory diseases',
    
    # Digestive
    'Cirrhosis and other chronic liver diseases',
    'Upper digestive system diseases',
    'Appendicitis',
    'Paralytic ileus and intestinal obstruction',
    'Inguinal, femoral, and abdominal hernia',
    'Inflammatory bowel disease',
    'Vascular intestinal disorders',
    'Gallbladder and biliary diseases',
    'Pancreatitis',
    'Other digestive diseases',
    
    # Neurological
    'Alzheimer disease and other dementias',
    'Parkinson disease',
    'Epilepsy',
    'Multiple sclerosis',
    'Motor neuron disease',
    'Headache disorders',
    'Other neurological disorders',
    
    # Mental Health
    'Schizophrenia',
    'Depressive disorders',
    'Bipolar disorder',
    'Anxiety disorders',
    'Eating disorders',
    'Autism spectrum disorders',
    'Attention-deficit/hyperactivity disorder',
    'Conduct disorder',
    'Idiopathic developmental intellectual disability',
    'Other mental disorders',
    
    # Substance Use
    'Alcohol use disorders',
    'Drug use disorders',
    
    # Diabetes/Kidney
    'Diabetes mellitus',
    'Chronic kidney disease',
    'Acute glomerulonephritis',
    
    # Skin
    'Dermatitis',
    'Psoriasis',
    'Bacterial skin diseases',
    'Scabies',
    'Fungal skin diseases',
    'Viral skin diseases',
    'Acne vulgaris',
    'Alopecia areata',
    'Pruritus',
    'Urticaria',
    'Decubitus ulcer',
    'Other skin and subcutaneous diseases',
    
    # Sense Organs
    'Blindness and vision loss',
    'Age-related and other hearing loss',
    'Other sense organ diseases',
    
    # Musculoskeletal
    'Rheumatoid arthritis',
    'Osteoarthritis',
    'Low back pain',
    'Neck pain',
    'Gout',
    'Other musculoskeletal disorders',
    
    # Other NCDs
    'Congenital birth defects',
    'Urinary diseases and male infertility',
    'Gynecological diseases',
    'Hemoglobinopathies and hemolytic anemias',
    'Endocrine, metabolic, blood, and immune disorders',
    'Oral disorders',
    'Sudden infant death syndrome',
    
    # Injuries - Transport
    'Road injuries',
    'Other transport injuries',
    
    # Injuries - Unintentional
    'Falls',
    'Drowning',
    'Fire, heat, and hot substances',
    'Poisonings',
    'Exposure to mechanical forces',
    'Adverse effects of medical treatment',
    'Animal contact',
    'Foreign body',
    'Environmental heat and cold exposure',
    'Exposure to forces of nature',
    'Other unintentional injuries',
    
    # Injuries - Self-harm/Violence
    'Self-harm',
    'Interpersonal violence',
    'Conflict and terrorism',
    'Police conflict and executions',
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def find_gbd_file(data_dir: Path) -> Path:
    """Find the IHME GBD data file."""
    patterns = ['IHMEGBD_2021*.csv', 'IHME_GBD*.csv', 'GBD_2021*.csv', 'gbd*.csv']
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            return files[0]
    return None


def load_ihme_causes(gbd_file: Path) -> Dict[str, float]:
    """Load IHME cause names and their DALYs (raw counts, not rates)."""
    logger.info(f"Loading IHME GBD data: {gbd_file.name}")
    
    df = pd.read_csv(gbd_file)
    
    # Filter for global, both sexes, all ages, DALYs, NUMBER (not Rate!)
    mask = (
        (df['location_name'] == 'Global') &
        (df['sex_name'] == 'Both') &
        (df['age_name'] == 'All ages') &
        (df['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
        (df['metric_name'] == 'Number')  # CRITICAL: must be Number, not Rate!
    )
    
    ihme_dalys = {}
    for _, row in df[mask].iterrows():
        cause_name = row['cause_name']
        dalys_val = float(row['val'])
        ihme_dalys[cause_name] = dalys_val
    
    # Sanity check - IHD should be > 100M DALYs
    if 'Ischemic heart disease' in ihme_dalys:
        ihd_dalys = ihme_dalys['Ischemic heart disease']
        if ihd_dalys < 1_000_000:
            logger.warning(f"  WARNING: IHD DALYs={ihd_dalys:.0f} looks like Rate, not Number!")
            logger.warning(f"  Check that metric_name='Number' filter is working")
    
    logger.info(f"  Loaded {len(ihme_dalys)} IHME causes")
    
    # Show sample values
    total = sum(ihme_dalys.values())
    logger.info(f"  Total DALYs: {total/1e9:.2f} billion")
    
    return ihme_dalys


def compute_semantic_mapping(our_keys: List[str], ihme_names: List[str]) -> Dict[str, dict]:
    """
    Compute semantic similarity between our keys and IHME names.
    Returns mapping with best match and similarity score.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers required. Run: pip install sentence-transformers")
    
    logger.info(f"Loading sentence transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info(f"Computing embeddings for {len(our_keys)} our keys...")
    our_embeddings = model.encode(our_keys, convert_to_tensor=True, show_progress_bar=True)
    
    logger.info(f"Computing embeddings for {len(ihme_names)} IHME names...")
    ihme_embeddings = model.encode(ihme_names, convert_to_tensor=True, show_progress_bar=True)
    
    logger.info("Computing similarity matrix...")
    # Compute cosine similarity matrix
    similarity_matrix = util.cos_sim(our_embeddings, ihme_embeddings)
    
    # Convert to numpy for easier manipulation
    sim_np = similarity_matrix.cpu().numpy()
    
    # Build mapping
    mapping = {}
    
    for i, our_key in enumerate(our_keys):
        # Find best match
        best_idx = np.argmax(sim_np[i])
        best_score = float(sim_np[i][best_idx])
        best_match = ihme_names[best_idx]
        
        # Find top 3 matches for review
        top_indices = np.argsort(sim_np[i])[::-1][:3]
        top_matches = [
            {'name': ihme_names[idx], 'score': float(sim_np[i][idx])}
            for idx in top_indices
        ]
        
        mapping[our_key] = {
            'best_match': best_match,
            'similarity': round(best_score, 4),
            'confident': best_score >= MIN_SIMILARITY,
            'top_matches': top_matches
        }
    
    return mapping


def review_and_fix_mapping(mapping: Dict[str, dict], ihme_dalys: Dict[str, float]) -> Dict[str, dict]:
    """
    Review mapping and apply manual fixes for edge cases.
    """
    # Manual overrides for known problem cases
    MANUAL_FIXES = {
        # Exact matches that semantic might miss
        'HIV/AIDS': 'HIV/AIDS',
        'COVID-19': 'COVID-19',
        'Ebola': 'Ebola',
        'Malaria': 'Malaria',
        'Dengue': 'Dengue',
        'Measles': 'Measles',
        'Tetanus': 'Tetanus',
        'Rabies': 'Rabies',
        'Leprosy': 'Leprosy',
        'Trachoma': 'Trachoma',
        'Meningitis': 'Meningitis',
        'Encephalitis': 'Encephalitis',
        'Pertussis': 'Pertussis',
        'Diphtheria': 'Diphtheria',
        'Tuberculosis': 'Tuberculosis',
        'Schistosomiasis': 'Schistosomiasis',
        'Leishmaniasis': 'Leishmaniasis',
        'Onchocerciasis': 'Onchocerciasis',
        'Cysticercosis': 'Cysticercosis',
        'Stroke': 'Stroke',
        'Asthma': 'Asthma',
        'Appendicitis': 'Appendicitis',
        'Pancreatitis': 'Pancreatitis',
        'Schizophrenia': 'Schizophrenia',
        'Psoriasis': 'Psoriasis',
        'Scabies': 'Scabies',
        'Pruritus': 'Pruritus',
        'Urticaria': 'Urticaria',
        'Gout': 'Gout',
        'Endocarditis': 'Endocarditis',
        'Drowning': 'Drowning',
        'Falls': 'Falls',
        'Poisonings': 'Poisonings',
        
        # Apostrophe issues
        'Alzheimer disease and other dementias': "Alzheimer's disease and other dementias",
        'Parkinson disease': "Parkinson's disease",
        
        # Spelling/naming differences
        'Epilepsy': 'Idiopathic epilepsy',
        'Road injuries': 'Road injuries',
        'Peripheral artery disease': 'Lower extremity peripheral arterial disease',
        'Eye cancer': 'Eye cancer',
    }
    
    for our_key, ihme_name in MANUAL_FIXES.items():
        if our_key in mapping and ihme_name in ihme_dalys:
            mapping[our_key]['best_match'] = ihme_name
            mapping[our_key]['similarity'] = 1.0
            mapping[our_key]['confident'] = True
            mapping[our_key]['manual_override'] = True
    
    return mapping


def main():
    print("=" * 70)
    print("GBD Cause Semantic Similarity Mapping")
    print("=" * 70)
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("\n❌ sentence-transformers not installed!")
        print("   Run: pip install sentence-transformers")
        return
    
    # Find GBD file
    gbd_file = find_gbd_file(DATA_DIR)
    if not gbd_file:
        print(f"\n❌ No GBD file found in {DATA_DIR}")
        print("   Place IHMEGBD_2021_DATA*.csv in DATA/ folder")
        return
    
    # Load IHME causes
    ihme_dalys = load_ihme_causes(gbd_file)
    ihme_names = list(ihme_dalys.keys())
    
    print(f"\n📊 Our cause keys: {len(OUR_CAUSE_KEYS)}")
    print(f"📊 IHME causes: {len(ihme_names)}")
    
    # Compute semantic mapping
    print(f"\n🧠 Computing semantic similarity...")
    mapping = compute_semantic_mapping(OUR_CAUSE_KEYS, ihme_names)
    
    # Apply manual fixes
    print(f"\n🔧 Applying manual fixes...")
    mapping = review_and_fix_mapping(mapping, ihme_dalys)
    
    # Add DALYs to mapping
    for our_key, match_info in mapping.items():
        ihme_name = match_info['best_match']
        match_info['dalys'] = ihme_dalys.get(ihme_name, 0)
    
    # Statistics
    confident_count = sum(1 for m in mapping.values() if m['confident'])
    total_dalys = sum(m['dalys'] for m in mapping.values())
    
    print(f"\n📈 Results:")
    print(f"   Confident matches (≥{MIN_SIMILARITY}): {confident_count}/{len(mapping)}")
    print(f"   Total DALYs mapped: {total_dalys/1e9:.2f} billion")
    
    # Show low-confidence matches for review
    low_confidence = [(k, v) for k, v in mapping.items() if not v['confident']]
    if low_confidence:
        print(f"\n⚠️  Low-confidence matches ({len(low_confidence)}):")
        for our_key, match_info in low_confidence[:15]:
            print(f"   {our_key}")
            print(f"      → {match_info['best_match']} (sim={match_info['similarity']:.3f})")
    
    # Save mapping
    output = {
        'metadata': {
            'model': MODEL_NAME,
            'min_similarity': MIN_SIMILARITY,
            'our_keys_count': len(OUR_CAUSE_KEYS),
            'ihme_causes_count': len(ihme_names),
            'confident_matches': confident_count,
            'total_dalys_billions': round(total_dalys / 1e9, 2)
        },
        'mapping': mapping
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved: {OUTPUT_FILE}")
    
    # Show sample matches
    print(f"\n📋 Sample matches:")
    for our_key in ['Ischemic heart disease', 'Stroke', 'Breast cancer', 'Malaria', 
                    'Alzheimer disease and other dementias', 'Depressive disorders'][:6]:
        if our_key in mapping:
            m = mapping[our_key]
            dalys_m = m['dalys'] / 1e6
            print(f"   {our_key}")
            print(f"      → {m['best_match']} ({dalys_m:.1f}M DALYs, sim={m['similarity']:.3f})")
    
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Now run: python 03-01-bhem-map-diseases.py")


if __name__ == "__main__":
    main()