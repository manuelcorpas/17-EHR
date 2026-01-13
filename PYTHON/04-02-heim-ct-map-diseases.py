#!/usr/bin/env python3
"""
04-02-heim-ct-map-diseases.py
=============================
HEIM-CT: Map Clinical Trials to GBD 2021 Disease Categories

Maps ClinicalTrials.gov conditions to the same GBD 2021 disease taxonomy
used in HEIM-Biobank (03-01). Uses both MeSH terms (when available) and
free-text condition name matching.

INPUT:
    DATA/heim_ct_studies.csv
    DATA/heim_ct_conditions.csv
    DATA/heim_ct_mesh_conditions.csv
    DATA/gbd_disease_registry.json (from 03-01, if available)
    DATA/IHMEGBD_2021_DATA*.csv (official GBD data)
    
OUTPUT:
    DATA/heim_ct_studies_mapped.csv
    DATA/heim_ct_disease_registry.json
    DATA/heim_ct_disease_trial_matrix.csv

USAGE:
    python 04-02-heim-ct-map-diseases.py

VERSION: HEIM-CT v1.0
DATE: 2026-01-13
"""

import os
import re
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

# Number of parallel workers (use performance cores on M3 Ultra)
N_WORKERS = min(24, cpu_count())

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VERSION = "HEIM-CT v1.0"
VERSION_DATE = "2026-01-13"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"

# Input files
INPUT_STUDIES = DATA_DIR / "heim_ct_studies.csv"
INPUT_CONDITIONS = DATA_DIR / "heim_ct_conditions.csv"
INPUT_MESH = DATA_DIR / "heim_ct_mesh_conditions.csv"

# Output files
OUTPUT_STUDIES_MAPPED = DATA_DIR / "heim_ct_studies_mapped.csv"
OUTPUT_REGISTRY = DATA_DIR / "heim_ct_disease_registry.json"
OUTPUT_MATRIX = DATA_DIR / "heim_ct_disease_trial_matrix.csv"


# =============================================================================
# GBD DISEASE MAPPING (IDENTICAL TO 03-01 FOR CONSISTENCY)
# =============================================================================
# Maps GBD cause names to MeSH terms and free-text keywords for condition matching.
# This ensures clinical trials map to the same disease categories as biobank publications.

GBD_CONDITION_MAPPING = {
    # =========================================================================
    # COMMUNICABLE, MATERNAL, NEONATAL, AND NUTRITIONAL DISEASES
    # =========================================================================
    
    # HIV/AIDS and STIs
    'HIV/AIDS': {
        'mesh_terms': ['HIV Infections', 'Acquired Immunodeficiency Syndrome', 'HIV', 'HIV-1', 'HIV-2'],
        'keywords': ['hiv', 'aids', 'human immunodeficiency virus', 'antiretroviral'],
        'gbd_level2': 'HIV/AIDS and STIs',
        'global_south_priority': True
    },
    'Sexually transmitted infections excluding HIV': {
        'mesh_terms': ['Sexually Transmitted Diseases', 'Syphilis', 'Gonorrhea', 'Chlamydia Infections', 'Herpes Genitalis', 'Human Papillomavirus'],
        'keywords': ['syphilis', 'gonorrhea', 'chlamydia', 'herpes', 'hpv', 'papillomavirus', 'sexually transmitted'],
        'gbd_level2': 'HIV/AIDS and STIs',
        'global_south_priority': True
    },
    
    # Respiratory Infections
    'Tuberculosis': {
        'mesh_terms': ['Tuberculosis', 'Tuberculosis, Pulmonary', 'Mycobacterium tuberculosis'],
        'keywords': ['tuberculosis', 'tb', 'mycobacterium tuberculosis', 'mdr-tb', 'xdr-tb', 'pulmonary tb'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': True
    },
    'Lower respiratory infections': {
        'mesh_terms': ['Respiratory Tract Infections', 'Pneumonia', 'Bronchitis', 'Bronchiolitis', 'Influenza, Human'],
        'keywords': ['pneumonia', 'bronchitis', 'bronchiolitis', 'respiratory infection', 'influenza', 'rsv', 'respiratory syncytial'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': True
    },
    'Upper respiratory infections': {
        'mesh_terms': ['Common Cold', 'Pharyngitis', 'Tonsillitis', 'Sinusitis', 'Rhinitis'],
        'keywords': ['pharyngitis', 'tonsillitis', 'sinusitis', 'rhinitis', 'common cold', 'upper respiratory'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': False
    },
    'COVID-19': {
        'mesh_terms': ['COVID-19', 'SARS-CoV-2', 'Coronavirus'],
        'keywords': ['covid', 'sars-cov-2', 'coronavirus', 'covid-19', 'covid19', 'long covid'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': True
    },
    
    # Enteric Infections
    'Diarrheal diseases': {
        'mesh_terms': ['Diarrhea', 'Dysentery', 'Gastroenteritis', 'Cholera', 'Rotavirus Infections'],
        'keywords': ['diarrhea', 'dysentery', 'gastroenteritis', 'cholera', 'rotavirus', 'norovirus', 'salmonella'],
        'gbd_level2': 'Enteric infections',
        'global_south_priority': True
    },
    'Typhoid and paratyphoid': {
        'mesh_terms': ['Typhoid Fever', 'Paratyphoid Fever'],
        'keywords': ['typhoid', 'paratyphoid', 'enteric fever'],
        'gbd_level2': 'Enteric infections',
        'global_south_priority': True
    },
    
    # NTDs and Malaria
    'Malaria': {
        'mesh_terms': ['Malaria', 'Plasmodium falciparum', 'Plasmodium vivax'],
        'keywords': ['malaria', 'plasmodium', 'antimalarial', 'cerebral malaria'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Dengue': {
        'mesh_terms': ['Dengue', 'Dengue Virus', 'Severe Dengue'],
        'keywords': ['dengue', 'dengue fever', 'dengue hemorrhagic'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Leishmaniasis': {
        'mesh_terms': ['Leishmaniasis', 'Leishmania', 'Leishmaniasis, Visceral'],
        'keywords': ['leishmaniasis', 'leishmania', 'kala-azar', 'visceral leishmaniasis'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Chagas disease': {
        'mesh_terms': ['Chagas Disease', 'Trypanosoma cruzi'],
        'keywords': ['chagas', 'trypanosoma cruzi', 'american trypanosomiasis'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Schistosomiasis': {
        'mesh_terms': ['Schistosomiasis', 'Schistosoma'],
        'keywords': ['schistosomiasis', 'schistosoma', 'bilharzia'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Lymphatic filariasis': {
        'mesh_terms': ['Elephantiasis, Filarial', 'Filariasis'],
        'keywords': ['filariasis', 'elephantiasis', 'lymphatic filariasis'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Onchocerciasis': {
        'mesh_terms': ['Onchocerciasis', 'Onchocerca volvulus'],
        'keywords': ['onchocerciasis', 'river blindness', 'onchocerca'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Trachoma': {
        'mesh_terms': ['Trachoma', 'Chlamydia trachomatis'],
        'keywords': ['trachoma'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Rabies': {
        'mesh_terms': ['Rabies', 'Rabies virus'],
        'keywords': ['rabies'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Yellow fever': {
        'mesh_terms': ['Yellow Fever'],
        'keywords': ['yellow fever'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Intestinal nematode infections': {
        'mesh_terms': ['Nematode Infections', 'Ascariasis', 'Hookworm Infections'],
        'keywords': ['ascariasis', 'hookworm', 'trichuriasis', 'strongyloidiasis', 'nematode'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Other neglected tropical diseases': {
        'mesh_terms': ['Neglected Diseases', 'Tropical Medicine'],
        'keywords': ['neglected tropical', 'ntd'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    
    # Other Infectious
    'Meningitis': {
        'mesh_terms': ['Meningitis', 'Meningitis, Bacterial', 'Meningitis, Viral'],
        'keywords': ['meningitis', 'meningococcal'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Acute hepatitis': {
        'mesh_terms': ['Hepatitis', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis E'],
        'keywords': ['hepatitis', 'hbv', 'hcv', 'viral hepatitis'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    
    # Maternal/Neonatal
    'Maternal disorders': {
        'mesh_terms': ['Pregnancy Complications', 'Pre-Eclampsia', 'Eclampsia', 'Postpartum Hemorrhage'],
        'keywords': ['preeclampsia', 'eclampsia', 'postpartum', 'maternal', 'pregnancy complication', 'gestational'],
        'gbd_level2': 'Maternal disorders',
        'global_south_priority': True
    },
    'Neonatal disorders': {
        'mesh_terms': ['Infant, Newborn, Diseases', 'Infant, Premature, Diseases', 'Neonatal Sepsis'],
        'keywords': ['neonatal', 'newborn', 'preterm', 'premature birth', 'low birth weight', 'nicu'],
        'gbd_level2': 'Neonatal disorders',
        'global_south_priority': True
    },
    
    # Nutritional
    'Protein-energy malnutrition': {
        'mesh_terms': ['Malnutrition', 'Protein-Energy Malnutrition', 'Kwashiorkor'],
        'keywords': ['malnutrition', 'undernutrition', 'kwashiorkor', 'marasmus', 'wasting'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    'Dietary iron deficiency': {
        'mesh_terms': ['Anemia, Iron-Deficiency', 'Iron Deficiency'],
        'keywords': ['iron deficiency', 'iron deficiency anemia'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    
    # =========================================================================
    # NON-COMMUNICABLE DISEASES
    # =========================================================================
    
    # Neoplasms
    'Breast cancer': {
        'mesh_terms': ['Breast Neoplasms', 'Carcinoma, Ductal, Breast', 'Triple Negative Breast Neoplasms'],
        'keywords': ['breast cancer', 'breast carcinoma', 'breast neoplasm', 'mammary cancer', 'brca'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Tracheal, bronchus, and lung cancer': {
        'mesh_terms': ['Lung Neoplasms', 'Carcinoma, Non-Small-Cell Lung', 'Small Cell Lung Carcinoma'],
        'keywords': ['lung cancer', 'lung carcinoma', 'nsclc', 'sclc', 'non-small cell', 'small cell lung'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Colon and rectum cancer': {
        'mesh_terms': ['Colorectal Neoplasms', 'Colonic Neoplasms', 'Rectal Neoplasms'],
        'keywords': ['colorectal cancer', 'colon cancer', 'rectal cancer', 'bowel cancer', 'crc'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Prostate cancer': {
        'mesh_terms': ['Prostatic Neoplasms', 'Prostate Cancer'],
        'keywords': ['prostate cancer', 'prostatic cancer', 'prostate carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Liver cancer': {
        'mesh_terms': ['Liver Neoplasms', 'Carcinoma, Hepatocellular'],
        'keywords': ['liver cancer', 'hepatocellular carcinoma', 'hcc', 'hepatoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Stomach cancer': {
        'mesh_terms': ['Stomach Neoplasms', 'Gastric Cancer'],
        'keywords': ['stomach cancer', 'gastric cancer', 'gastric carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Pancreatic cancer': {
        'mesh_terms': ['Pancreatic Neoplasms'],
        'keywords': ['pancreatic cancer', 'pancreas cancer', 'pancreatic carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Cervical cancer': {
        'mesh_terms': ['Uterine Cervical Neoplasms'],
        'keywords': ['cervical cancer', 'cervix cancer', 'cervical carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': True
    },
    'Ovarian cancer': {
        'mesh_terms': ['Ovarian Neoplasms'],
        'keywords': ['ovarian cancer', 'ovary cancer', 'ovarian carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Leukemia': {
        'mesh_terms': ['Leukemia', 'Leukemia, Myeloid, Acute', 'Leukemia, Lymphocytic, Chronic'],
        'keywords': ['leukemia', 'leukaemia', 'aml', 'cml', 'all', 'cll'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Non-Hodgkin lymphoma': {
        'mesh_terms': ['Lymphoma, Non-Hodgkin', 'Lymphoma, B-Cell'],
        'keywords': ['non-hodgkin lymphoma', 'nhl', 'b-cell lymphoma', 't-cell lymphoma', 'diffuse large b-cell'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Multiple myeloma': {
        'mesh_terms': ['Multiple Myeloma'],
        'keywords': ['multiple myeloma', 'myeloma', 'plasma cell neoplasm'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Brain and central nervous system cancer': {
        'mesh_terms': ['Brain Neoplasms', 'Glioma', 'Glioblastoma'],
        'keywords': ['brain cancer', 'brain tumor', 'glioma', 'glioblastoma', 'gbm', 'astrocytoma', 'meningioma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Kidney cancer': {
        'mesh_terms': ['Kidney Neoplasms', 'Carcinoma, Renal Cell'],
        'keywords': ['kidney cancer', 'renal cancer', 'renal cell carcinoma', 'rcc'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Bladder cancer': {
        'mesh_terms': ['Urinary Bladder Neoplasms'],
        'keywords': ['bladder cancer', 'urothelial carcinoma', 'bladder carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Melanoma': {
        'mesh_terms': ['Melanoma', 'Malignant Melanoma'],
        'keywords': ['melanoma', 'malignant melanoma', 'skin melanoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Other malignant neoplasms': {
        'mesh_terms': ['Neoplasms', 'Cancer', 'Carcinoma', 'Sarcoma'],
        'keywords': ['cancer', 'carcinoma', 'tumor', 'tumour', 'neoplasm', 'malignant', 'oncology', 'metastatic'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    
    # Cardiovascular
    'Ischemic heart disease': {
        'mesh_terms': ['Myocardial Ischemia', 'Coronary Artery Disease', 'Myocardial Infarction', 'Angina Pectoris'],
        'keywords': ['coronary artery disease', 'cad', 'myocardial infarction', 'heart attack', 'angina', 'ischemic heart', 'acute coronary'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Stroke': {
        'mesh_terms': ['Stroke', 'Cerebrovascular Disorders', 'Brain Infarction', 'Intracranial Hemorrhages'],
        'keywords': ['stroke', 'cerebrovascular', 'brain infarction', 'cerebral hemorrhage', 'tia', 'transient ischemic'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Hypertensive heart disease': {
        'mesh_terms': ['Hypertension', 'Blood Pressure'],
        'keywords': ['hypertension', 'high blood pressure', 'hypertensive', 'blood pressure'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Atrial fibrillation and flutter': {
        'mesh_terms': ['Atrial Fibrillation', 'Atrial Flutter'],
        'keywords': ['atrial fibrillation', 'afib', 'af', 'atrial flutter', 'arrhythmia'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Heart failure': {
        'mesh_terms': ['Heart Failure'],
        'keywords': ['heart failure', 'cardiac failure', 'congestive heart failure', 'hfref', 'hfpef', 'chf'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Cardiomyopathy and myocarditis': {
        'mesh_terms': ['Cardiomyopathies', 'Myocarditis'],
        'keywords': ['cardiomyopathy', 'myocarditis', 'dilated cardiomyopathy', 'hypertrophic cardiomyopathy'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Rheumatic heart disease': {
        'mesh_terms': ['Rheumatic Heart Disease', 'Rheumatic Fever'],
        'keywords': ['rheumatic heart', 'rheumatic fever'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': True
    },
    'Other cardiovascular and circulatory diseases': {
        'mesh_terms': ['Cardiovascular Diseases', 'Heart Diseases', 'Vascular Diseases'],
        'keywords': ['cardiovascular', 'cardiac', 'heart disease', 'vascular disease', 'thromboembolism', 'dvt', 'pulmonary embolism'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    
    # Chronic Respiratory
    'Chronic obstructive pulmonary disease': {
        'mesh_terms': ['Pulmonary Disease, Chronic Obstructive', 'Emphysema'],
        'keywords': ['copd', 'chronic obstructive pulmonary', 'emphysema', 'chronic bronchitis'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    'Asthma': {
        'mesh_terms': ['Asthma'],
        'keywords': ['asthma', 'bronchial asthma', 'asthmatic'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    'Interstitial lung disease and pulmonary sarcoidosis': {
        'mesh_terms': ['Lung Diseases, Interstitial', 'Pulmonary Fibrosis', 'Sarcoidosis'],
        'keywords': ['interstitial lung', 'pulmonary fibrosis', 'ipf', 'sarcoidosis', 'ild'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    
    # Digestive
    'Cirrhosis and other chronic liver diseases': {
        'mesh_terms': ['Liver Cirrhosis', 'Liver Diseases', 'Fatty Liver'],
        'keywords': ['cirrhosis', 'liver disease', 'fatty liver', 'nafld', 'nash', 'liver fibrosis'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Inflammatory bowel disease': {
        'mesh_terms': ['Inflammatory Bowel Diseases', 'Crohn Disease', 'Colitis, Ulcerative'],
        'keywords': ['inflammatory bowel', 'ibd', 'crohn', 'ulcerative colitis', 'uc'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Upper digestive system diseases': {
        'mesh_terms': ['Peptic Ulcer', 'Gastritis', 'Gastroesophageal Reflux'],
        'keywords': ['peptic ulcer', 'gastritis', 'gerd', 'reflux', 'dyspepsia'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Pancreatitis': {
        'mesh_terms': ['Pancreatitis'],
        'keywords': ['pancreatitis', 'acute pancreatitis', 'chronic pancreatitis'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    
    # Neurological
    "Alzheimer's disease and other dementias": {
        'mesh_terms': ['Alzheimer Disease', 'Dementia', 'Cognitive Dysfunction'],
        'keywords': ['alzheimer', 'dementia', 'cognitive impairment', 'mild cognitive impairment', 'mci'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    "Parkinson's disease": {
        'mesh_terms': ['Parkinson Disease'],
        'keywords': ['parkinson', 'parkinsonian', 'pd'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Idiopathic epilepsy': {
        'mesh_terms': ['Epilepsy', 'Seizures'],
        'keywords': ['epilepsy', 'seizure', 'epileptic', 'convulsion'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Multiple sclerosis': {
        'mesh_terms': ['Multiple Sclerosis'],
        'keywords': ['multiple sclerosis', 'ms'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Motor neuron disease': {
        'mesh_terms': ['Motor Neuron Disease', 'Amyotrophic Lateral Sclerosis'],
        'keywords': ['motor neuron', 'als', 'amyotrophic lateral sclerosis', 'lou gehrig'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Headache disorders': {
        'mesh_terms': ['Headache', 'Migraine Disorders'],
        'keywords': ['migraine', 'headache', 'cluster headache', 'tension headache'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    
    # Mental Disorders
    'Depressive disorders': {
        'mesh_terms': ['Depressive Disorder', 'Depression', 'Depressive Disorder, Major'],
        'keywords': ['depression', 'depressive', 'major depressive', 'mdd'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Anxiety disorders': {
        'mesh_terms': ['Anxiety Disorders', 'Anxiety', 'Panic Disorder'],
        'keywords': ['anxiety', 'panic disorder', 'generalized anxiety', 'gad', 'social anxiety', 'phobia', 'ocd'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Bipolar disorder': {
        'mesh_terms': ['Bipolar Disorder'],
        'keywords': ['bipolar', 'manic', 'mania'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Schizophrenia': {
        'mesh_terms': ['Schizophrenia', 'Psychotic Disorders'],
        'keywords': ['schizophrenia', 'psychosis', 'psychotic'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Eating disorders': {
        'mesh_terms': ['Eating Disorders', 'Anorexia Nervosa', 'Bulimia Nervosa'],
        'keywords': ['anorexia', 'bulimia', 'eating disorder', 'binge eating'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Autism spectrum disorders': {
        'mesh_terms': ['Autism Spectrum Disorder', 'Autistic Disorder'],
        'keywords': ['autism', 'autistic', 'asd', 'asperger'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Attention-deficit/hyperactivity disorder': {
        'mesh_terms': ['Attention Deficit Disorder with Hyperactivity'],
        'keywords': ['adhd', 'attention deficit', 'hyperactivity'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Other mental disorders': {
        'mesh_terms': ['Mental Disorders', 'Stress Disorders, Post-Traumatic'],
        'keywords': ['ptsd', 'post-traumatic', 'mental health', 'psychiatric', 'personality disorder', 'insomnia'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    
    # Substance Use
    'Alcohol use disorders': {
        'mesh_terms': ['Alcoholism', 'Alcohol-Related Disorders'],
        'keywords': ['alcoholism', 'alcohol use disorder', 'alcohol dependence', 'alcohol abuse'],
        'gbd_level2': 'Substance use disorders',
        'global_south_priority': False
    },
    'Drug use disorders': {
        'mesh_terms': ['Substance-Related Disorders', 'Opioid-Related Disorders'],
        'keywords': ['substance use', 'drug addiction', 'opioid', 'cocaine', 'drug dependence', 'substance abuse'],
        'gbd_level2': 'Substance use disorders',
        'global_south_priority': False
    },
    
    # Diabetes/Kidney
    'Diabetes mellitus': {
        'mesh_terms': ['Diabetes Mellitus', 'Diabetes Mellitus, Type 2', 'Diabetes Mellitus, Type 1'],
        'keywords': ['diabetes', 'diabetic', 'type 2 diabetes', 'type 1 diabetes', 't2dm', 't1dm', 'hyperglycemia'],
        'gbd_level2': 'Diabetes and kidney diseases',
        'global_south_priority': False
    },
    'Chronic kidney disease': {
        'mesh_terms': ['Kidney Diseases', 'Renal Insufficiency, Chronic', 'Kidney Failure, Chronic'],
        'keywords': ['chronic kidney', 'ckd', 'renal failure', 'kidney disease', 'esrd', 'dialysis', 'kidney transplant'],
        'gbd_level2': 'Diabetes and kidney diseases',
        'global_south_priority': False
    },
    
    # Musculoskeletal
    'Low back pain': {
        'mesh_terms': ['Low Back Pain', 'Back Pain'],
        'keywords': ['low back pain', 'back pain', 'lumbar pain', 'sciatica'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Osteoarthritis': {
        'mesh_terms': ['Osteoarthritis'],
        'keywords': ['osteoarthritis', 'oa', 'knee arthritis', 'hip arthritis', 'degenerative joint'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Rheumatoid arthritis': {
        'mesh_terms': ['Arthritis, Rheumatoid'],
        'keywords': ['rheumatoid arthritis', 'ra'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Other musculoskeletal disorders': {
        'mesh_terms': ['Musculoskeletal Diseases', 'Fibromyalgia', 'Osteoporosis'],
        'keywords': ['fibromyalgia', 'osteoporosis', 'lupus', 'sle', 'musculoskeletal', 'bone disease'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    
    # Skin
    'Dermatitis': {
        'mesh_terms': ['Dermatitis', 'Eczema', 'Dermatitis, Atopic'],
        'keywords': ['dermatitis', 'eczema', 'atopic dermatitis'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Psoriasis': {
        'mesh_terms': ['Psoriasis'],
        'keywords': ['psoriasis', 'psoriatic'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    
    # Other NCDs
    'Congenital birth defects': {
        'mesh_terms': ['Congenital Abnormalities', 'Heart Defects, Congenital'],
        'keywords': ['congenital', 'birth defect', 'congenital heart', 'down syndrome', 'genetic syndrome'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    'Hemoglobinopathies and hemolytic anemias': {
        'mesh_terms': ['Hemoglobinopathies', 'Anemia, Sickle Cell', 'Thalassemia'],
        'keywords': ['sickle cell', 'thalassemia', 'hemoglobinopathy', 'sickle cell disease'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': True
    },
    'Blindness and vision loss': {
        'mesh_terms': ['Blindness', 'Vision Disorders', 'Cataract', 'Glaucoma', 'Macular Degeneration'],
        'keywords': ['blindness', 'vision loss', 'cataract', 'glaucoma', 'macular degeneration', 'amd', 'diabetic retinopathy'],
        'gbd_level2': 'Sense organ diseases',
        'global_south_priority': False
    },
    
    # =========================================================================
    # INJURIES
    # =========================================================================
    'Road injuries': {
        'mesh_terms': ['Accidents, Traffic', 'Wounds and Injuries'],
        'keywords': ['traffic accident', 'road injury', 'motor vehicle', 'trauma'],
        'gbd_level2': 'Transport injuries',
        'global_south_priority': True
    },
    'Falls': {
        'mesh_terms': ['Accidental Falls', 'Hip Fractures', 'Fractures, Bone'],
        'keywords': ['fall', 'fracture', 'hip fracture', 'fall injury'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Self-harm': {
        'mesh_terms': ['Suicide', 'Self-Injurious Behavior'],
        'keywords': ['suicide', 'self-harm', 'suicidal', 'self-injury'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': False
    },
    'Interpersonal violence': {
        'mesh_terms': ['Violence', 'Domestic Violence'],
        'keywords': ['violence', 'domestic violence', 'assault', 'abuse'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': True
    },
}


# =============================================================================
# MAPPING FUNCTIONS
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    if not text or pd.isna(text):
        return ""
    return re.sub(r'[^\w\s]', ' ', str(text).lower()).strip()


def match_mesh_term(mesh_term: str, mapping: Dict) -> List[str]:
    """Match a MeSH term to GBD causes."""
    if not mesh_term or pd.isna(mesh_term):
        return []
    
    mesh_norm = normalize_text(mesh_term)
    matched = []
    
    for gbd_cause, info in mapping.items():
        for term in info.get('mesh_terms', []):
            term_norm = normalize_text(term)
            if term_norm and (term_norm in mesh_norm or mesh_norm in term_norm):
                matched.append(gbd_cause)
                break
    
    return list(set(matched))


def match_condition_text(condition: str, mapping: Dict) -> List[str]:
    """Match a free-text condition to GBD causes using keyword matching."""
    if not condition or pd.isna(condition):
        return []
    
    cond_norm = normalize_text(condition)
    matched = []
    
    for gbd_cause, info in mapping.items():
        # Check keywords
        for keyword in info.get('keywords', []):
            kw_norm = normalize_text(keyword)
            if kw_norm and len(kw_norm) >= 3:
                # Word boundary matching for short keywords
                if len(kw_norm) <= 4:
                    pattern = r'\b' + re.escape(kw_norm) + r'\b'
                    if re.search(pattern, cond_norm):
                        matched.append(gbd_cause)
                        break
                elif kw_norm in cond_norm:
                    matched.append(gbd_cause)
                    break
    
    return list(set(matched))


def map_trial_to_diseases(
    nct_id: str,
    conditions_df: pd.DataFrame,
    mesh_df: pd.DataFrame,
    mapping: Dict
) -> List[str]:
    """Map a trial to GBD disease categories using conditions and MeSH terms."""
    matched_causes = set()

    # Match MeSH terms
    trial_mesh = mesh_df[mesh_df['nct_id'] == nct_id]
    for _, row in trial_mesh.iterrows():
        mesh_term = row.get('mesh_term', '')
        causes = match_mesh_term(mesh_term, mapping)
        matched_causes.update(causes)

    # Match condition names (free text)
    trial_conditions = conditions_df[conditions_df['nct_id'] == nct_id]
    for _, row in trial_conditions.iterrows():
        condition = row.get('name', '')
        causes = match_condition_text(condition, mapping)
        matched_causes.update(causes)

    return list(matched_causes)


# Pre-indexed lookup structures (set by main before multiprocessing)
_CONDITIONS_BY_NCT = {}
_MESH_BY_NCT = {}


def map_trial_fast(nct_id: str) -> Tuple[str, List[str]]:
    """Fast mapping using pre-indexed lookups. Returns (nct_id, causes)."""
    matched_causes = set()

    # Match MeSH terms (pre-indexed)
    for mesh_term in _MESH_BY_NCT.get(nct_id, []):
        causes = match_mesh_term(mesh_term, GBD_CONDITION_MAPPING)
        matched_causes.update(causes)

    # Match condition names (pre-indexed)
    for condition in _CONDITIONS_BY_NCT.get(nct_id, []):
        causes = match_condition_text(condition, GBD_CONDITION_MAPPING)
        matched_causes.update(causes)

    return (nct_id, list(matched_causes))


def init_worker(conditions_dict, mesh_dict):
    """Initialize worker process with pre-indexed data."""
    global _CONDITIONS_BY_NCT, _MESH_BY_NCT
    _CONDITIONS_BY_NCT = conditions_dict
    _MESH_BY_NCT = mesh_dict


def map_trials_parallel(nct_ids: List[str], conditions_dict: Dict, mesh_dict: Dict) -> Dict[str, List[str]]:
    """Map trials to diseases in parallel using multiprocessing."""
    global _CONDITIONS_BY_NCT, _MESH_BY_NCT
    _CONDITIONS_BY_NCT = conditions_dict
    _MESH_BY_NCT = mesh_dict

    logger.info(f"   Using {N_WORKERS} parallel workers...")

    # Process in parallel
    results = {}
    batch_size = 50000

    for i in range(0, len(nct_ids), batch_size):
        batch = nct_ids[i:i+batch_size]

        with Pool(processes=N_WORKERS, initializer=init_worker,
                  initargs=(conditions_dict, mesh_dict)) as pool:
            batch_results = pool.map(map_trial_fast, batch)

        for nct_id, causes in batch_results:
            results[nct_id] = causes

        logger.info(f"   Processed {min(i+batch_size, len(nct_ids)):,}/{len(nct_ids):,} trials...")

    return results


def find_gbd_file(data_dir: Path) -> Path:
    """Find the official IHME GBD data file."""
    patterns = ['IHMEGBD_2021*.csv', 'IHME_GBD*.csv', 'GBD_2021*.csv']
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            return files[0]
    return None


def load_gbd_dalys(gbd_file: Path) -> Dict[str, float]:
    """Load DALYs from official GBD file."""
    logger.info(f"Loading GBD DALYs from: {gbd_file.name}")
    
    df = pd.read_csv(gbd_file)
    
    # Filter for global, both sexes, all ages, DALYs, Number
    mask = (
        (df['location_name'] == 'Global') &
        (df['sex_name'] == 'Both') &
        (df['age_name'] == 'All ages') &
        (df['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
        (df['metric_name'] == 'Number')
    )
    
    dalys = {}
    for _, row in df[mask].iterrows():
        dalys[row['cause_name']] = float(row['val'])
    
    logger.info(f"   Loaded DALYs for {len(dalys)} causes")
    return dalys


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-CT: Map Clinical Trials to GBD 2021 Disease Categories")
    print(f"Version: {VERSION}")
    print("=" * 70)
    
    # Check inputs
    if not INPUT_STUDIES.exists():
        print(f"\n❌ Studies file not found: {INPUT_STUDIES}")
        print(f"   Run 04-01-heim-ct-fetch.py first")
        return
    
    # Load data
    print(f"\n📂 Loading data...")
    
    df_studies = pd.read_csv(INPUT_STUDIES)
    print(f"   Studies: {len(df_studies):,}")
    
    df_conditions = pd.read_csv(INPUT_CONDITIONS) if INPUT_CONDITIONS.exists() else pd.DataFrame()
    print(f"   Conditions: {len(df_conditions):,}")
    
    df_mesh = pd.read_csv(INPUT_MESH) if INPUT_MESH.exists() else pd.DataFrame()
    print(f"   MeSH conditions: {len(df_mesh):,}")
    
    # Load GBD DALYs
    gbd_file = find_gbd_file(DATA_DIR)
    gbd_dalys = {}
    if gbd_file:
        gbd_dalys = load_gbd_dalys(gbd_file)
    else:
        print(f"   ⚠️ GBD file not found; DALYs will be 0")
    
    # Pre-index conditions and MeSH by nct_id for fast lookup
    print(f"\n📇 Pre-indexing data for fast lookup...")

    conditions_dict = {}
    if len(df_conditions) > 0:
        for nct_id, group in df_conditions.groupby('nct_id'):
            conditions_dict[nct_id] = group['name'].dropna().tolist()
    print(f"   Indexed {len(conditions_dict):,} trials with conditions")

    mesh_dict = {}
    if len(df_mesh) > 0:
        for nct_id, group in df_mesh.groupby('nct_id'):
            mesh_dict[nct_id] = group['mesh_term'].dropna().tolist()
    print(f"   Indexed {len(mesh_dict):,} trials with MeSH terms")

    # Map trials to diseases (parallel)
    print(f"\n🔬 Mapping {len(df_studies):,} trials to {len(GBD_CONDITION_MAPPING)} GBD causes...")

    nct_ids = df_studies['nct_id'].tolist()
    trial_causes = map_trials_parallel(nct_ids, conditions_dict, mesh_dict)

    # Count causes
    cause_counts = defaultdict(int)
    for nct_id, causes in trial_causes.items():
        for cause in causes:
            cause_counts[cause] += 1
    
    # Add mapping to studies dataframe
    df_studies['gbd_causes'] = df_studies['nct_id'].map(lambda x: trial_causes.get(x, []))
    df_studies['cause_count'] = df_studies['gbd_causes'].apply(len)
    df_studies['gbd_causes_str'] = df_studies['gbd_causes'].apply(lambda x: '|'.join(x) if x else '')
    
    # Add GBD level 2
    def get_level2(causes):
        levels = set()
        for c in causes:
            if c in GBD_CONDITION_MAPPING:
                levels.add(GBD_CONDITION_MAPPING[c]['gbd_level2'])
        return list(levels)
    
    df_studies['gbd_level2'] = df_studies['gbd_causes'].apply(get_level2)
    df_studies['gbd_level2_str'] = df_studies['gbd_level2'].apply(lambda x: '|'.join(x) if x else '')
    
    # Global South priority
    def is_gs_priority(causes):
        for c in causes:
            if c in GBD_CONDITION_MAPPING:
                if GBD_CONDITION_MAPPING[c].get('global_south_priority', False):
                    return True
        return False
    
    df_studies['global_south_priority'] = df_studies['gbd_causes'].apply(is_gs_priority)
    
    # Statistics
    mapped = (df_studies['cause_count'] > 0).sum()
    unmapped = (df_studies['cause_count'] == 0).sum()
    gs_count = df_studies['global_south_priority'].sum()
    
    print(f"\n📊 MAPPING RESULTS:")
    print(f"   ✅ Mapped: {mapped:,} ({100*mapped/len(df_studies):.1f}%)")
    print(f"   ○  Unmapped: {unmapped:,} ({100*unmapped/len(df_studies):.1f}%)")
    print(f"   🌍 Global South priority: {gs_count:,}")
    
    # Build disease registry
    registry = {}
    for cause, info in GBD_CONDITION_MAPPING.items():
        dalys = 0
        # Try to match cause name to GBD DALYs
        for gbd_name, daly_val in gbd_dalys.items():
            if normalize_text(cause) in normalize_text(gbd_name) or normalize_text(gbd_name) in normalize_text(cause):
                dalys = daly_val
                break
        
        registry[cause] = {
            'name': cause,
            'gbd_level2': info['gbd_level2'],
            'global_south_priority': info['global_south_priority'],
            'dalys': dalys,
            'trials': cause_counts.get(cause, 0)
        }
    
    # Print top causes
    print(f"\n📋 Top 30 GBD causes by trial count:")
    sorted_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)
    for cause, count in sorted_causes[:30]:
        dalys_m = registry[cause]['dalys'] / 1e6 if registry[cause]['dalys'] else 0
        gs = "🌍" if GBD_CONDITION_MAPPING.get(cause, {}).get('global_south_priority') else "  "
        if dalys_m > 0:
            print(f"   {gs} {cause[:45]:45} {count:>6,} trials ({dalys_m:.1f}M DALYs)")
        else:
            print(f"   {gs} {cause[:45]:45} {count:>6,} trials")
    
    # Causes with 0 trials
    zero_trials = [c for c in GBD_CONDITION_MAPPING if cause_counts.get(c, 0) == 0]
    if zero_trials:
        print(f"\n⚠️  {len(zero_trials)} causes with 0 trials:")
        for c in zero_trials[:15]:
            gs = "🌍" if GBD_CONDITION_MAPPING.get(c, {}).get('global_south_priority') else "  "
            print(f"   {gs} {c}")
    
    # Create disease-trial matrix
    print(f"\n📊 Creating disease-trial matrix...")
    matrix_data = []
    for cause in GBD_CONDITION_MAPPING.keys():
        matrix_data.append({
            'gbd_cause': cause,
            'gbd_level2': GBD_CONDITION_MAPPING[cause]['gbd_level2'],
            'global_south_priority': GBD_CONDITION_MAPPING[cause]['global_south_priority'],
            'trial_count': cause_counts.get(cause, 0),
            'dalys_millions': registry[cause]['dalys'] / 1e6 if registry[cause]['dalys'] else 0
        })
    df_matrix = pd.DataFrame(matrix_data)
    df_matrix = df_matrix.sort_values('trial_count', ascending=False)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    df_studies.to_csv(OUTPUT_STUDIES_MAPPED, index=False)
    print(f"   {OUTPUT_STUDIES_MAPPED.name}: {len(df_studies):,} rows")
    
    with open(OUTPUT_REGISTRY, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"   {OUTPUT_REGISTRY.name}")
    
    df_matrix.to_csv(OUTPUT_MATRIX, index=False)
    print(f"   {OUTPUT_MATRIX.name}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ MAPPING COMPLETE")
    print(f"=" * 70)
    print(f"   Trials mapped: {mapped:,}/{len(df_studies):,} ({100*mapped/len(df_studies):.1f}%)")
    print(f"   GBD causes: {len(GBD_CONDITION_MAPPING)}")
    print(f"   Causes with trials: {len(cause_counts)}")
    print(f"   Global South priority trials: {gs_count:,}")
    
    print(f"\n➡️  Next: python 04-03-heim-ct-compute-metrics.py")


if __name__ == "__main__":
    main()
