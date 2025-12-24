#!/usr/bin/env python3
"""
02-03-bhem-compute-metrics.py
=============================
HEIM-Biobank v1.0: Compute Health Equity Metrics

Implements four core equity metrics from Corpas et al. (2025):
1. Burden Score - Composite disease burden from GBD 2021
2. Research Gap Score - Mismatch between burden and research attention
3. Research Opportunity Score - Unrealized potential per biobank
4. Equity Alignment Score - Overall equity performance per biobank (NEW)

INPUT:  DATA/bhem_publications_mapped.csv
OUTPUT: DATA/bhem_metrics.json
        DATA/bhem_biobank_metrics.csv
        DATA/bhem_disease_metrics.csv

USAGE:
    python 02-03-bhem-compute-metrics.py

METHODOLOGY:
    Corpas M, et al. (2025). EHR-Linked Biobank Expansion Reveals Global
    Health Inequities. Annual Review of Biomedical Data Science.

VERSION: HEIM-Biobank v1.0
DATE: 2025-12-24
"""

import json
import logging
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

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

# Version metadata
VERSION = "HEIM-Biobank v1.0"
VERSION_DATE = "2025-12-24"
METHODOLOGY_SOURCE = "Corpas et al. 2025, Annual Review of Biomedical Data Science"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"

# Input files
INPUT_PUBLICATIONS = DATA_DIR / "bhem_publications_mapped.csv"

# Output files
OUTPUT_METRICS_JSON = DATA_DIR / "bhem_metrics.json"
OUTPUT_BIOBANK_CSV = DATA_DIR / "bhem_biobank_metrics.csv"
OUTPUT_DISEASE_CSV = DATA_DIR / "bhem_disease_metrics.csv"


# =============================================================================
# DISEASE REGISTRY - 25 HIGH-BURDEN DISEASES WITH GBD 2021 DATA
# =============================================================================

DISEASE_REGISTRY = {
    # Cardiovascular diseases
    "ischemic_heart_disease": {
        "name": "Ischemic Heart Disease",
        "category": "Cardiovascular",
        "dalys_millions": 185.0,
        "deaths_millions": 9.44,
        "prevalence_millions": 244.0,
        "global_south_priority": False,
        "mesh_terms": ["Myocardial Ischemia", "Coronary Disease", "Coronary Artery Disease",
                       "Myocardial Infarction", "Angina Pectoris"]
    },
    "stroke": {
        "name": "Stroke",
        "category": "Cardiovascular",
        "dalys_millions": 143.0,
        "deaths_millions": 7.08,
        "prevalence_millions": 101.0,
        "global_south_priority": False,
        "mesh_terms": ["Stroke", "Cerebrovascular Disorders", "Brain Ischemia",
                       "Intracranial Hemorrhages", "Cerebral Infarction"]
    },
    
    # Respiratory diseases
    "copd": {
        "name": "Chronic Obstructive Pulmonary Disease",
        "category": "Respiratory",
        "dalys_millions": 81.0,
        "deaths_millions": 3.28,
        "prevalence_millions": 212.0,
        "global_south_priority": False,
        "mesh_terms": ["Pulmonary Disease, Chronic Obstructive", "Chronic Obstructive Pulmonary Disease",
                       "COPD", "Emphysema", "Chronic Bronchitis"]
    },
    "asthma": {
        "name": "Asthma",
        "category": "Respiratory",
        "dalys_millions": 22.0,
        "deaths_millions": 0.46,
        "prevalence_millions": 262.0,
        "global_south_priority": False,
        "mesh_terms": ["Asthma", "Asthma, Bronchial", "Bronchial Hyperreactivity"]
    },
    
    # Metabolic diseases
    "type2_diabetes": {
        "name": "Type 2 Diabetes Mellitus",
        "category": "Metabolic",
        "dalys_millions": 67.0,
        "deaths_millions": 1.50,
        "prevalence_millions": 462.0,
        "global_south_priority": False,
        "mesh_terms": ["Diabetes Mellitus, Type 2", "Type 2 Diabetes", "Non-Insulin-Dependent Diabetes",
                       "Adult-Onset Diabetes"]
    },
    
    # Infectious diseases (Global South priority)
    "malaria": {
        "name": "Malaria",
        "category": "Infectious",
        "dalys_millions": 62.5,
        "deaths_millions": 0.62,
        "prevalence_millions": 247.0,
        "global_south_priority": True,
        "mesh_terms": ["Malaria", "Plasmodium falciparum", "Plasmodium vivax",
                       "Malaria, Cerebral", "Malaria, Falciparum"]
    },
    "tuberculosis": {
        "name": "Tuberculosis",
        "category": "Infectious",
        "dalys_millions": 38.1,
        "deaths_millions": 1.30,
        "prevalence_millions": 23.0,
        "global_south_priority": True,
        "mesh_terms": ["Tuberculosis", "Tuberculosis, Pulmonary", "Mycobacterium tuberculosis",
                       "Latent Tuberculosis", "Tuberculosis, Multidrug-Resistant"]
    },
    "hiv_aids": {
        "name": "HIV/AIDS",
        "category": "Infectious",
        "dalys_millions": 42.0,
        "deaths_millions": 0.68,
        "prevalence_millions": 38.4,
        "global_south_priority": True,
        "mesh_terms": ["HIV Infections", "Acquired Immunodeficiency Syndrome", "HIV",
                       "AIDS-Related Opportunistic Infections", "HIV Seropositivity"]
    },
    "diarrheal_diseases": {
        "name": "Diarrheal Diseases",
        "category": "Infectious",
        "dalys_millions": 44.2,
        "deaths_millions": 1.53,
        "prevalence_millions": 1500.0,
        "global_south_priority": True,
        "mesh_terms": ["Diarrhea", "Dysentery", "Cholera", "Gastroenteritis",
                       "Rotavirus Infections", "Cryptosporidiosis"]
    },
    
    # Neglected tropical diseases
    "ntds": {
        "name": "Neglected Tropical Diseases",
        "category": "Neglected",
        "dalys_millions": 28.3,
        "deaths_millions": 0.20,
        "prevalence_millions": 1700.0,
        "global_south_priority": True,
        "mesh_terms": ["Neglected Diseases", "Tropical Medicine", "Schistosomiasis",
                       "Leishmaniasis", "Chagas Disease", "Lymphatic Filariasis",
                       "Onchocerciasis", "Dengue", "Trachoma"]
    },
    
    # Neurological diseases
    "alzheimers": {
        "name": "Alzheimer's Disease and Dementia",
        "category": "Neurological",
        "dalys_millions": 30.0,
        "deaths_millions": 1.80,
        "prevalence_millions": 55.0,
        "global_south_priority": False,
        "mesh_terms": ["Alzheimer Disease", "Dementia", "Cognitive Dysfunction",
                       "Neurodegenerative Diseases", "Tauopathies"]
    },
    "epilepsy": {
        "name": "Epilepsy",
        "category": "Neurological",
        "dalys_millions": 15.0,
        "deaths_millions": 0.13,
        "prevalence_millions": 46.0,
        "global_south_priority": False,
        "mesh_terms": ["Epilepsy", "Seizures", "Status Epilepticus", "Epilepsy, Generalized"]
    },
    "parkinsons": {
        "name": "Parkinson's Disease",
        "category": "Neurological",
        "dalys_millions": 6.0,
        "deaths_millions": 0.33,
        "prevalence_millions": 8.5,
        "global_south_priority": False,
        "mesh_terms": ["Parkinson Disease", "Parkinsonian Disorders", "Lewy Body Disease"]
    },
    
    # Mental health disorders
    "depression": {
        "name": "Depressive Disorders",
        "category": "Mental Health",
        "dalys_millions": 50.0,
        "deaths_millions": 0.0,
        "prevalence_millions": 280.0,
        "global_south_priority": False,
        "mesh_terms": ["Depressive Disorder", "Depression", "Depressive Disorder, Major",
                       "Dysthymic Disorder", "Depressive Disorder, Treatment-Resistant"]
    },
    "anxiety": {
        "name": "Anxiety Disorders",
        "category": "Mental Health",
        "dalys_millions": 28.0,
        "deaths_millions": 0.0,
        "prevalence_millions": 301.0,
        "global_south_priority": False,
        "mesh_terms": ["Anxiety Disorders", "Anxiety", "Generalized Anxiety Disorder",
                       "Panic Disorder", "Phobic Disorders", "Social Anxiety Disorder"]
    },
    "bipolar": {
        "name": "Bipolar Disorder",
        "category": "Mental Health",
        "dalys_millions": 12.0,
        "deaths_millions": 0.0,
        "prevalence_millions": 40.0,
        "global_south_priority": False,
        "mesh_terms": ["Bipolar Disorder", "Bipolar and Related Disorders", "Mania",
                       "Cyclothymic Disorder"]
    },
    "schizophrenia": {
        "name": "Schizophrenia",
        "category": "Mental Health",
        "dalys_millions": 17.0,
        "deaths_millions": 0.0,
        "prevalence_millions": 24.0,
        "global_south_priority": False,
        "mesh_terms": ["Schizophrenia", "Schizophrenia Spectrum and Other Psychotic Disorders",
                       "Psychotic Disorders"]
    },
    
    # Cancer
    "lung_cancer": {
        "name": "Lung Cancer",
        "category": "Cancer",
        "dalys_millions": 45.0,
        "deaths_millions": 1.80,
        "prevalence_millions": 2.2,
        "global_south_priority": False,
        "mesh_terms": ["Lung Neoplasms", "Carcinoma, Non-Small-Cell Lung",
                       "Small Cell Lung Carcinoma", "Adenocarcinoma of Lung"]
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "category": "Cancer",
        "dalys_millions": 18.0,
        "deaths_millions": 0.68,
        "prevalence_millions": 7.8,
        "global_south_priority": False,
        "mesh_terms": ["Breast Neoplasms", "Breast Cancer", "Carcinoma, Ductal, Breast",
                       "Triple Negative Breast Neoplasms"]
    },
    
    # Other NCDs
    "chronic_kidney_disease": {
        "name": "Chronic Kidney Disease",
        "category": "Other NCD",
        "dalys_millions": 35.0,
        "deaths_millions": 1.30,
        "prevalence_millions": 697.0,
        "global_south_priority": False,
        "mesh_terms": ["Renal Insufficiency, Chronic", "Chronic Kidney Disease",
                       "Kidney Failure, Chronic", "Diabetic Nephropathies"]
    },
    "cirrhosis": {
        "name": "Cirrhosis and Liver Disease",
        "category": "Other NCD",
        "dalys_millions": 25.0,
        "deaths_millions": 1.32,
        "prevalence_millions": 112.0,
        "global_south_priority": False,
        "mesh_terms": ["Liver Cirrhosis", "Fatty Liver", "Hepatitis, Chronic",
                       "Non-alcoholic Fatty Liver Disease", "Liver Diseases"]
    },
    
    # Musculoskeletal
    "low_back_pain": {
        "name": "Low Back Pain",
        "category": "Musculoskeletal",
        "dalys_millions": 63.0,
        "deaths_millions": 0.0,
        "prevalence_millions": 568.0,
        "global_south_priority": False,
        "mesh_terms": ["Low Back Pain", "Back Pain", "Intervertebral Disc Degeneration",
                       "Sciatica", "Lumbar Vertebrae"]
    },
    
    # Injuries (Global South priority)
    "road_traffic_injuries": {
        "name": "Road Traffic Injuries",
        "category": "Injuries",
        "dalys_millions": 85.5,
        "deaths_millions": 1.35,
        "prevalence_millions": 50.0,
        "global_south_priority": True,
        "mesh_terms": ["Accidents, Traffic", "Wounds and Injuries", "Craniocerebral Trauma",
                       "Spinal Cord Injuries", "Motor Vehicles"]
    },
    
    # Maternal and neonatal (Global South priority)
    "neonatal_disorders": {
        "name": "Neonatal Disorders",
        "category": "Maternal/Child",
        "dalys_millions": 75.0,
        "deaths_millions": 2.40,
        "prevalence_millions": 30.0,
        "global_south_priority": True,
        "mesh_terms": ["Infant, Newborn, Diseases", "Neonatal Sepsis", "Jaundice, Neonatal",
                       "Respiratory Distress Syndrome, Newborn", "Asphyxia Neonatorum"]
    },
    "preterm_birth": {
        "name": "Preterm Birth Complications",
        "category": "Maternal/Child",
        "dalys_millions": 23.0,
        "deaths_millions": 0.90,
        "prevalence_millions": 15.0,
        "global_south_priority": True,
        "mesh_terms": ["Premature Birth", "Infant, Premature", "Infant, Very Low Birth Weight",
                       "Bronchopulmonary Dysplasia", "Retinopathy of Prematurity"]
    }
}


# =============================================================================
# GLOBAL SOUTH CLASSIFICATION
# =============================================================================

GLOBAL_SOUTH_REGIONS = {"AFR", "SEAR", "EMR"}

GLOBAL_SOUTH_COUNTRIES = {
    # Africa
    "Uganda", "Nigeria", "Kenya", "South Africa", "Ghana", "Ethiopia", "Tanzania",
    "Rwanda", "Senegal", "Cameroon", "Egypt", "Morocco", "Tunisia",
    # Latin America
    "Mexico", "Brazil", "Argentina", "Chile", "Colombia", "Peru", "Ecuador",
    "Venezuela", "Cuba", "Costa Rica",
    # Asia
    "India", "Bangladesh", "Pakistan", "Indonesia", "Vietnam", "Philippines",
    "Thailand", "Malaysia", "Nepal", "Sri Lanka",
    # China (considered separately but included for completeness)
    "China"
}


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_burden_score(dalys_millions: float, deaths_millions: float, 
                         prevalence_millions: float) -> float:
    """
    Compute composite Burden Score for a disease.
    
    Formula (Corpas et al. 2025):
        Burden_Score = (0.5 × DALYs) + (50 × Deaths) + [10 × log₁₀(Prevalence)]
    
    Args:
        dalys_millions: Disability-adjusted life years in millions
        deaths_millions: Annual deaths in millions
        prevalence_millions: Total cases in millions
    
    Returns:
        Composite burden score (higher = greater burden)
    """
    # Handle edge case: prevalence = 0
    if prevalence_millions <= 0:
        prevalence_millions = 1.0  # Use 1.0 to avoid log(0)
    
    daly_component = 0.5 * dalys_millions
    death_component = 50.0 * deaths_millions
    prevalence_component = 10.0 * math.log10(prevalence_millions)
    
    burden_score = daly_component + death_component + prevalence_component
    
    return round(burden_score, 2)


def compute_research_gap_score(disease_id: str, publications: int, 
                                burden_score: float, total_dalys: float,
                                global_south_priority: bool,
                                category: str) -> Tuple[float, str]:
    """
    Compute Research Gap Score for a disease (0-100 scale).
    
    Three-tier scoring system (Corpas et al. 2025):
    1. Zero-publication penalty: Gap = 95 (Critical)
    2. Threshold adjustments for infectious/neglected diseases
    3. Burden-normalized intensity for others
    
    Args:
        disease_id: Disease identifier
        publications: Number of biobank publications
        burden_score: Computed burden score
        total_dalys: Total DALYs for this disease
        global_south_priority: Whether disease is Global South priority
        category: Disease category (Infectious, Neglected, etc.)
    
    Returns:
        Tuple of (gap_score, gap_severity_category)
    """
    # Tier 1: Zero publications = Critical gap
    if publications == 0:
        gap_score = 95.0
        return gap_score, "Critical"
    
    # Tier 2: Category-specific thresholds
    if category == "Infectious":
        if publications < 10:
            gap_score = 90.0
        elif publications < 25:
            gap_score = 80.0
        elif publications < 50:
            gap_score = 70.0
        elif publications < 100:
            gap_score = 60.0
        else:
            # Move to burden-normalized calculation
            gap_score = None
            
    elif category == "Neglected":
        if publications < 10:
            gap_score = 92.0
        elif publications < 25:
            gap_score = 82.0
        elif publications < 50:
            gap_score = 72.0
        elif publications < 100:
            gap_score = 65.0
        else:
            gap_score = None
            
    else:
        gap_score = None
    
    # Tier 3: Burden-normalized intensity for other cases
    if gap_score is None:
        # Publications per million DALYs
        if total_dalys > 0:
            pubs_per_million_dalys = publications / total_dalys
        else:
            pubs_per_million_dalys = publications
        
        # Decile-based scoring (higher intensity = lower gap)
        if pubs_per_million_dalys >= 100:
            gap_score = 10.0
        elif pubs_per_million_dalys >= 50:
            gap_score = 20.0
        elif pubs_per_million_dalys >= 25:
            gap_score = 30.0
        elif pubs_per_million_dalys >= 10:
            gap_score = 40.0
        elif pubs_per_million_dalys >= 5:
            gap_score = 50.0
        elif pubs_per_million_dalys >= 2:
            gap_score = 60.0
        elif pubs_per_million_dalys >= 1:
            gap_score = 70.0
        elif pubs_per_million_dalys >= 0.5:
            gap_score = 80.0
        else:
            gap_score = 85.0
    
    # Global South priority penalty: +10 if under-researched
    if global_south_priority and publications < 50:
        gap_score = min(95.0, gap_score + 10.0)
    
    # Classify severity
    if gap_score > 70:
        severity = "Critical"
    elif gap_score > 50:
        severity = "High"
    elif gap_score > 30:
        severity = "Moderate"
    else:
        severity = "Low"
    
    return round(gap_score, 1), severity


def compute_research_opportunity_score(disease_publications: Dict[str, int],
                                       disease_metrics: Dict[str, Dict]) -> float:
    """
    Compute Research Opportunity Score for a biobank.
    
    Formula (Corpas et al. 2025):
        ROS_b = Σ Burden_Score(d) for diseases where Publications(b,d) ≤ 2
    
    Higher ROS indicates greater unrealized potential for equity-aligned research.
    
    Args:
        disease_publications: Dict mapping disease_id -> publication count for this biobank
        disease_metrics: Dict with burden scores for all diseases
    
    Returns:
        Research Opportunity Score (sum of burden scores for underexplored diseases)
    """
    ros = 0.0
    
    for disease_id, metrics in disease_metrics.items():
        pubs = disease_publications.get(disease_id, 0)
        
        # Include diseases with 0, 1, or 2 publications
        if pubs <= 2:
            ros += metrics['burden_score']
    
    return round(ros, 2)


def compute_equity_alignment_score(disease_publications: Dict[str, int],
                                   disease_metrics: Dict[str, Dict],
                                   total_publications: int) -> Tuple[float, str, Dict]:
    """
    Compute Equity Alignment Score for a biobank (0-100 scale).
    
    NEW in HEIM-Biobank v1.0:
        EAS = 100 - [(0.4 × Gap_Severity) + (0.3 × Burden_Miss) + (0.3 × Capacity_Penalty)]
    
    Components:
        - Gap_Severity: Weighted count of gap categories, normalized to 0-100
        - Burden_Miss: Proportion of DALYs in diseases with ≤2 publications
        - Capacity_Penalty: Inverse of publications per disease coverage
    
    Args:
        disease_publications: Dict mapping disease_id -> publication count
        disease_metrics: Dict with burden scores and gap severities
        total_publications: Total publications for this biobank
    
    Returns:
        Tuple of (eas_score, category, components_dict)
    """
    # Count gap severities for this biobank's coverage
    n_critical = 0
    n_high = 0
    n_moderate = 0
    n_low = 0
    
    missed_dalys = 0.0
    total_dalys = 0.0
    diseases_covered = 0
    
    for disease_id, metrics in disease_metrics.items():
        pubs = disease_publications.get(disease_id, 0)
        dalys = metrics['dalys_millions']
        total_dalys += dalys
        
        if pubs > 0:
            diseases_covered += 1
        
        # Count gaps based on this biobank's coverage
        if pubs == 0:
            n_critical += 1
            missed_dalys += dalys
        elif pubs <= 2:
            # Still consider as under-researched
            gap_sev = metrics['gap_severity']
            if gap_sev == "Critical":
                n_critical += 1
            elif gap_sev == "High":
                n_high += 1
            else:
                n_moderate += 1
            missed_dalys += dalys
        elif pubs <= 10:
            gap_sev = metrics['gap_severity']
            if gap_sev == "High":
                n_high += 1
            elif gap_sev == "Moderate":
                n_moderate += 1
            else:
                n_low += 1
        else:
            n_low += 1
    
    # Component 1: Gap Severity (0-100)
    # Weighted gaps: Critical=4, High=2, Moderate=1
    weighted_gaps = (4 * n_critical) + (2 * n_high) + (1 * n_moderate)
    max_possible_gaps = 4 * len(disease_metrics)  # All critical
    gap_severity_component = min(100.0, (weighted_gaps / max_possible_gaps) * 100) if max_possible_gaps > 0 else 0
    
    # Component 2: Burden Miss (0-100)
    burden_miss_component = (missed_dalys / total_dalys * 100) if total_dalys > 0 else 0
    
    # Component 3: Capacity Penalty (0-100)
    # Based on average publications per disease
    n_diseases = len(disease_metrics)
    pubs_per_disease = total_publications / n_diseases if n_diseases > 0 else 0
    capacity_penalty = 100 - min(pubs_per_disease, 100)
    
    # Compute EAS
    eas = 100 - (
        (0.4 * gap_severity_component) +
        (0.3 * burden_miss_component) +
        (0.3 * capacity_penalty)
    )
    
    # Clamp to 0-100
    eas = max(0.0, min(100.0, eas))
    
    # Categorize
    if eas >= 80:
        category = "Strong"
    elif eas >= 60:
        category = "Moderate"
    elif eas >= 40:
        category = "Weak"
    else:
        category = "Poor"
    
    components = {
        'gap_severity_component': round(gap_severity_component, 2),
        'burden_miss_component': round(burden_miss_component, 2),
        'capacity_penalty': round(capacity_penalty, 2),
        'n_critical': n_critical,
        'n_high': n_high,
        'n_moderate': n_moderate,
        'n_low': n_low,
        'diseases_covered': diseases_covered,
        'missed_dalys': round(missed_dalys, 2)
    }
    
    return round(eas, 1), category, components


def compute_equity_ratio(hic_publications: int, hic_dalys: float,
                         lmic_publications: int, lmic_dalys: float) -> float:
    """
    Compute global equity ratio comparing HIC vs LMIC research intensity.
    
    Formula:
        Equity_Ratio = (Pubs_HIC / DALYs_HIC) / (Pubs_LMIC / DALYs_LMIC)
    
    Ratio > 1 indicates HIC-focused disparity.
    
    Args:
        hic_publications: Publications from HIC biobanks
        hic_dalys: Estimated DALYs in HIC regions
        lmic_publications: Publications from Global South biobanks
        lmic_dalys: Estimated DALYs in LMIC regions
    
    Returns:
        Equity ratio (>1 = HIC bias, <1 = LMIC bias, 1 = balanced)
    """
    # Avoid division by zero
    if hic_dalys <= 0 or lmic_dalys <= 0:
        return float('inf')
    if hic_publications <= 0 or lmic_publications <= 0:
        return float('inf')
    
    hic_intensity = hic_publications / hic_dalys
    lmic_intensity = lmic_publications / lmic_dalys
    
    if lmic_intensity <= 0:
        return float('inf')
    
    ratio = hic_intensity / lmic_intensity
    
    return round(ratio, 2)


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def load_publications(filepath: Path) -> pd.DataFrame:
    """Load mapped publications data."""
    logger.info(f"Loading publications from {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} publications")
    
    return df


def compute_all_disease_metrics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute metrics for all 25 diseases in the registry.
    
    Returns dict mapping disease_id -> {burden_score, gap_score, gap_severity, ...}
    """
    logger.info("Computing disease-level metrics...")
    
    disease_metrics = {}
    
    # Count publications per disease
    disease_pubs = defaultdict(int)
    
    if 'disease_ids_str' in df.columns:
        for disease_str in df['disease_ids_str'].dropna():
            for disease_id in disease_str.split('|'):
                disease_id = disease_id.strip()
                if disease_id:
                    disease_pubs[disease_id] += 1
    
    # Compute metrics for each disease in registry
    for disease_id, disease_data in DISEASE_REGISTRY.items():
        pubs = disease_pubs.get(disease_id, 0)
        
        # Compute burden score
        burden_score = compute_burden_score(
            dalys_millions=disease_data['dalys_millions'],
            deaths_millions=disease_data['deaths_millions'],
            prevalence_millions=disease_data['prevalence_millions']
        )
        
        # Compute gap score
        gap_score, gap_severity = compute_research_gap_score(
            disease_id=disease_id,
            publications=pubs,
            burden_score=burden_score,
            total_dalys=disease_data['dalys_millions'],
            global_south_priority=disease_data['global_south_priority'],
            category=disease_data['category']
        )
        
        # Research intensity: pubs per million DALYs
        research_intensity = pubs / disease_data['dalys_millions'] if disease_data['dalys_millions'] > 0 else 0
        
        disease_metrics[disease_id] = {
            'name': disease_data['name'],
            'category': disease_data['category'],
            'dalys_millions': disease_data['dalys_millions'],
            'deaths_millions': disease_data['deaths_millions'],
            'prevalence_millions': disease_data['prevalence_millions'],
            'global_south_priority': disease_data['global_south_priority'],
            'publications': pubs,
            'burden_score': burden_score,
            'gap_score': gap_score,
            'gap_severity': gap_severity,
            'research_intensity': round(research_intensity, 3)
        }
    
    # Log summary
    severities = defaultdict(int)
    for dm in disease_metrics.values():
        severities[dm['gap_severity']] += 1
    
    logger.info(f"Disease metrics computed: {len(disease_metrics)} diseases")
    logger.info(f"  Gap distribution: Critical={severities['Critical']}, "
                f"High={severities['High']}, Moderate={severities['Moderate']}, "
                f"Low={severities['Low']}")
    
    return disease_metrics


def compute_all_biobank_metrics(df: pd.DataFrame, 
                                 disease_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute metrics for all biobanks in the dataset.
    
    Returns dict mapping biobank_id -> {ros, eas, disease_publications, ...}
    """
    logger.info("Computing biobank-level metrics...")
    
    biobank_metrics = {}
    
    # Get unique biobanks
    if 'biobank_id' not in df.columns:
        logger.warning("No biobank_id column found")
        return biobank_metrics
    
    biobank_ids = df['biobank_id'].unique()
    
    for biobank_id in biobank_ids:
        biobank_df = df[df['biobank_id'] == biobank_id]
        
        # Count publications per disease for this biobank
        disease_pubs = defaultdict(int)
        
        if 'disease_ids_str' in biobank_df.columns:
            for disease_str in biobank_df['disease_ids_str'].dropna():
                for disease_id in disease_str.split('|'):
                    disease_id = disease_id.strip()
                    if disease_id:
                        disease_pubs[disease_id] += 1
        
        # Get biobank metadata
        biobank_name = biobank_df['biobank_name'].iloc[0] if 'biobank_name' in biobank_df.columns else biobank_id
        country = biobank_df['country'].iloc[0] if 'country' in biobank_df.columns else "Unknown"
        region = biobank_df['region'].iloc[0] if 'region' in biobank_df.columns else "Unknown"
        
        total_pubs = len(biobank_df)
        
        # Compute ROS
        ros = compute_research_opportunity_score(disease_pubs, disease_metrics)
        
        # Compute EAS
        eas, eas_category, eas_components = compute_equity_alignment_score(
            disease_pubs, disease_metrics, total_pubs
        )
        
        # Count critical gaps (diseases with 0 publications)
        critical_gap_diseases = []
        for disease_id in disease_metrics.keys():
            if disease_pubs.get(disease_id, 0) == 0:
                critical_gap_diseases.append(disease_id)
        
        # Global South percentage
        gs_pubs = 0
        if 'global_south_priority' in biobank_df.columns:
            gs_pubs = biobank_df['global_south_priority'].sum()
        gs_pct = (gs_pubs / total_pubs * 100) if total_pubs > 0 else 0
        
        # Year distribution
        year_dist = {}
        if 'year' in biobank_df.columns:
            year_counts = biobank_df['year'].value_counts()
            year_dist = {str(int(y)): int(c) for y, c in year_counts.items() if pd.notna(y)}
        
        # Determine if Global South biobank
        is_global_south = (
            region in GLOBAL_SOUTH_REGIONS or 
            country in GLOBAL_SOUTH_COUNTRIES
        )
        
        biobank_metrics[biobank_id] = {
            'name': biobank_name,
            'country': country,
            'region': region,
            'is_global_south': is_global_south,
            'total_publications': total_pubs,
            'diseases_covered': len([d for d in disease_pubs.values() if d > 0]),
            'research_opportunity_score': ros,
            'equity_alignment_score': eas,
            'equity_alignment_category': eas_category,
            'equity_alignment_components': eas_components,
            'critical_gap_count': len(critical_gap_diseases),
            'critical_gaps': critical_gap_diseases,
            'global_south_percentage': round(gs_pct, 1),
            'disease_publications': dict(disease_pubs),
            'year_distribution': year_dist
        }
    
    # Log summary
    eas_dist = defaultdict(int)
    for bm in biobank_metrics.values():
        eas_dist[bm['equity_alignment_category']] += 1
    
    logger.info(f"Biobank metrics computed: {len(biobank_metrics)} biobanks")
    logger.info(f"  EAS distribution: Strong={eas_dist['Strong']}, "
                f"Moderate={eas_dist['Moderate']}, Weak={eas_dist['Weak']}, "
                f"Poor={eas_dist['Poor']}")
    
    return biobank_metrics


def compute_global_metrics(df: pd.DataFrame,
                           disease_metrics: Dict[str, Dict],
                           biobank_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compute global/system-wide metrics.
    """
    logger.info("Computing global metrics...")
    
    total_pubs = len(df)
    
    # Compute equity ratio
    # Estimate: 80% of DALYs in LMICs, 20% in HICs
    total_dalys = sum(dm['dalys_millions'] for dm in disease_metrics.values())
    lmic_dalys = total_dalys * 0.80
    hic_dalys = total_dalys * 0.20
    
    # Count publications by region
    hic_pubs = 0
    gs_pubs = 0
    
    for bm in biobank_metrics.values():
        if bm['is_global_south']:
            gs_pubs += bm['total_publications']
        else:
            hic_pubs += bm['total_publications']
    
    equity_ratio = compute_equity_ratio(hic_pubs, hic_dalys, gs_pubs, lmic_dalys)
    
    # Gap distribution
    gap_dist = defaultdict(int)
    for dm in disease_metrics.values():
        gap_dist[dm['gap_severity']] += 1
    
    # EAS distribution
    eas_dist = defaultdict(int)
    for bm in biobank_metrics.values():
        eas_dist[bm['equity_alignment_category']] += 1
    
    # Averages
    avg_ros = np.mean([bm['research_opportunity_score'] for bm in biobank_metrics.values()]) if biobank_metrics else 0
    avg_eas = np.mean([bm['equity_alignment_score'] for bm in biobank_metrics.values()]) if biobank_metrics else 0
    avg_gap = np.mean([dm['gap_score'] for dm in disease_metrics.values()]) if disease_metrics else 0
    
    global_metrics = {
        'version': VERSION,
        'version_date': VERSION_DATE,
        'generated_at': datetime.now().isoformat(),
        'total_publications': total_pubs,
        'total_biobanks': len(biobank_metrics),
        'total_diseases': len(disease_metrics),
        'equity_ratio': equity_ratio,
        'hic_publications': hic_pubs,
        'global_south_publications': gs_pubs,
        'gap_distribution': dict(gap_dist),
        'eas_distribution': dict(eas_dist),
        'average_ros': round(avg_ros, 2),
        'average_eas': round(avg_eas, 1),
        'average_gap_score': round(avg_gap, 1),
        'methodology': {
            'burden_score': "0.5 × DALYs + 50 × Deaths + 10 × log₁₀(Prevalence)",
            'gap_score': "Three-tier: zero-pub penalty, category thresholds, burden-normalized intensity",
            'ros': "Σ Burden_Score for diseases with ≤2 publications",
            'eas': "100 - (0.4 × Gap_Severity + 0.3 × Burden_Miss + 0.3 × Capacity_Penalty)",
            'source': METHODOLOGY_SOURCE
        }
    }
    
    logger.info(f"Global metrics: {total_pubs:,} publications, "
                f"equity ratio = {equity_ratio:.1f}x, "
                f"avg EAS = {avg_eas:.1f}")
    
    return global_metrics


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_metrics_json(global_metrics: Dict, biobank_metrics: Dict, 
                      disease_metrics: Dict, filepath: Path) -> None:
    """Save all metrics to JSON file."""
    output = {
        'global': global_metrics,
        'biobanks': biobank_metrics,
        'diseases': disease_metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Saved metrics JSON: {filepath}")


def save_biobank_csv(biobank_metrics: Dict, filepath: Path) -> None:
    """Save biobank metrics to CSV file."""
    rows = []
    
    for biobank_id, bm in biobank_metrics.items():
        row = {
            'biobank_id': biobank_id,
            'name': bm['name'],
            'country': bm['country'],
            'region': bm['region'],
            'is_global_south': bm['is_global_south'],
            'total_publications': bm['total_publications'],
            'diseases_covered': bm['diseases_covered'],
            'research_opportunity_score': bm['research_opportunity_score'],
            'equity_alignment_score': bm['equity_alignment_score'],
            'equity_alignment_category': bm['equity_alignment_category'],
            'critical_gap_count': bm['critical_gap_count'],
            'global_south_percentage': bm['global_south_percentage'],
            'gap_severity_component': bm['equity_alignment_components']['gap_severity_component'],
            'burden_miss_component': bm['equity_alignment_components']['burden_miss_component'],
            'capacity_penalty': bm['equity_alignment_components']['capacity_penalty']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('equity_alignment_score', ascending=False)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved biobank CSV: {filepath}")


def save_disease_csv(disease_metrics: Dict, filepath: Path) -> None:
    """Save disease metrics to CSV file."""
    rows = []
    
    for disease_id, dm in disease_metrics.items():
        row = {
            'disease_id': disease_id,
            'name': dm['name'],
            'category': dm['category'],
            'dalys_millions': dm['dalys_millions'],
            'deaths_millions': dm['deaths_millions'],
            'prevalence_millions': dm['prevalence_millions'],
            'global_south_priority': dm['global_south_priority'],
            'publications': dm['publications'],
            'burden_score': dm['burden_score'],
            'gap_score': dm['gap_score'],
            'gap_severity': dm['gap_severity'],
            'research_intensity': dm['research_intensity']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('gap_score', ascending=False)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved disease CSV: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-Biobank v1.0: Compute Health Equity Metrics")
    print(f"Methodology: {METHODOLOGY_SOURCE}")
    print("=" * 70)
    
    # Check input file
    if not INPUT_PUBLICATIONS.exists():
        print(f"\n❌ Input file not found: {INPUT_PUBLICATIONS}")
        print(f"   Run 02-01-bhem-map-diseases.py first")
        return
    
    # Load publications
    print(f"\n📂 Loading data...")
    df = load_publications(INPUT_PUBLICATIONS)
    print(f"   Publications loaded: {len(df):,}")
    
    # Compute disease metrics
    print(f"\n📊 Computing disease metrics...")
    disease_metrics = compute_all_disease_metrics(df)
    print(f"   Diseases processed: {len(disease_metrics)}")
    
    # Compute biobank metrics
    print(f"\n🏦 Computing biobank metrics...")
    biobank_metrics = compute_all_biobank_metrics(df, disease_metrics)
    print(f"   Biobanks processed: {len(biobank_metrics)}")
    
    # Compute global metrics
    print(f"\n🌍 Computing global metrics...")
    global_metrics = compute_global_metrics(df, disease_metrics, biobank_metrics)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    save_metrics_json(global_metrics, biobank_metrics, disease_metrics, OUTPUT_METRICS_JSON)
    save_biobank_csv(biobank_metrics, OUTPUT_BIOBANK_CSV)
    save_disease_csv(disease_metrics, OUTPUT_DISEASE_CSV)
    
    # Print summary
    print(f"\n" + "=" * 70)
    print(f"📊 HEIM-Biobank v1.0 METRICS SUMMARY")
    print(f"=" * 70)
    
    print(f"\n🔢 Global Statistics:")
    print(f"   Total publications: {global_metrics['total_publications']:,}")
    print(f"   Total biobanks: {global_metrics['total_biobanks']}")
    print(f"   Total diseases: {global_metrics['total_diseases']}")
    print(f"   Equity ratio: {global_metrics['equity_ratio']:.1f}x (>1 = HIC bias)")
    
    print(f"\n📈 Gap Distribution:")
    for severity, count in sorted(global_metrics['gap_distribution'].items()):
        print(f"   {severity}: {count} diseases")
    
    print(f"\n🏆 Equity Alignment Distribution:")
    for category, count in sorted(global_metrics['eas_distribution'].items()):
        print(f"   {category}: {count} biobanks")
    
    print(f"\n📁 Output Files:")
    print(f"   {OUTPUT_METRICS_JSON}")
    print(f"   {OUTPUT_BIOBANK_CSV}")
    print(f"   {OUTPUT_DISEASE_CSV}")
    
    # Top 5 biobanks by EAS
    if biobank_metrics:
        print(f"\n🏆 Top 5 Biobanks by Equity Alignment Score:")
        sorted_biobanks = sorted(biobank_metrics.items(), 
                                  key=lambda x: x[1]['equity_alignment_score'], 
                                  reverse=True)[:5]
        for i, (bid, bm) in enumerate(sorted_biobanks, 1):
            print(f"   {i}. {bm['name']}: EAS={bm['equity_alignment_score']:.1f} ({bm['equity_alignment_category']})")
    
    # Top 5 critical gap diseases
    print(f"\n⚠️  Top 5 Critical Gap Diseases:")
    sorted_diseases = sorted(disease_metrics.items(),
                              key=lambda x: x[1]['gap_score'],
                              reverse=True)[:5]
    for i, (did, dm) in enumerate(sorted_diseases, 1):
        print(f"   {i}. {dm['name']}: Gap={dm['gap_score']:.0f} ({dm['gap_severity']}), "
              f"Pubs={dm['publications']}")
    
    print(f"\n✅ COMPLETE!")
    print(f"\n➡️  Next step: python 02-05-bhem-generate-json.py")


if __name__ == "__main__":
    main()