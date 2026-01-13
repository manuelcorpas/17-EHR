#!/usr/bin/env python3
"""
02-01-bhem-map-diseases-GBD.py
==============================
BHEM Step 2: Map Publications to Official GBD 2021 Disease Categories

Uses OFFICIAL IHME Global Burden of Disease 2021 data:
- Cause names exactly as in GBD
- DALYs values from official IHME export (NOT invented)
- All 175 GBD causes supported

INPUT:  
    DATA/bhem_publications.csv
    DATA/IHMEGBD_2021_DATA*.csv (official IHME GBD export)
    
OUTPUT: 
    DATA/bhem_publications_mapped.csv
    DATA/gbd_disease_registry.json

USAGE:
    python 02-01-bhem-map-diseases-GBD.py
"""

import os
import json
import logging
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"

INPUT_FILE = DATA_DIR / "bhem_publications.csv"
OUTPUT_FILE = DATA_DIR / "bhem_publications_mapped.csv"
REGISTRY_FILE = DATA_DIR / "gbd_disease_registry.json"

# =============================================================================
# GBD CAUSE -> MESH TERM MAPPING
# =============================================================================
# Maps EXACT GBD cause names (as in IHME data) to MeSH terms
# Organized by GBD hierarchy

GBD_MESH_MAPPING = {
    
    # =========================================================================
    # COMMUNICABLE, MATERNAL, NEONATAL, AND NUTRITIONAL DISEASES
    # =========================================================================
    
    # --- HIV/AIDS and STIs ---
    'HIV/AIDS': {
        'mesh_terms': ['HIV Infections', 'Acquired Immunodeficiency Syndrome', 'HIV', 'HIV-1', 'HIV-2', 'Anti-HIV Agents', 'CD4 Lymphocyte Count', 'Viral Load', 'Antiretroviral Therapy'],
        'gbd_level2': 'HIV/AIDS and STIs',
        'global_south_priority': True
    },
    'Sexually transmitted infections excluding HIV': {
        'mesh_terms': ['Sexually Transmitted Diseases', 'Syphilis', 'Gonorrhea', 'Chlamydia Infections', 'Herpes Genitalis', 'Human Papillomavirus', 'Trichomoniasis'],
        'gbd_level2': 'HIV/AIDS and STIs',
        'global_south_priority': True
    },
    
    # --- Respiratory Infections ---
    'Lower respiratory infections': {
        'mesh_terms': ['Respiratory Tract Infections', 'Pneumonia', 'Bronchitis', 'Bronchiolitis', 'Influenza, Human', 'Respiratory Syncytial Virus'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': True
    },
    'Upper respiratory infections': {
        'mesh_terms': ['Common Cold', 'Pharyngitis', 'Tonsillitis', 'Sinusitis', 'Rhinitis', 'Laryngitis'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': False
    },
    'Otitis media': {
        'mesh_terms': ['Otitis Media', 'Ear Infections', 'Otitis'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': False
    },
    'COVID-19': {
        'mesh_terms': ['COVID-19', 'SARS-CoV-2', 'Coronavirus', 'Coronavirus Infections', 'COVID-19 Vaccines', 'Post-Acute COVID-19 Syndrome'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': True
    },
    'Tuberculosis': {
        'mesh_terms': ['Tuberculosis', 'Tuberculosis, Pulmonary', 'Mycobacterium tuberculosis', 'Tuberculosis, Multidrug-Resistant', 'Latent Tuberculosis'],
        'gbd_level2': 'Respiratory infections',
        'global_south_priority': True
    },
    
    # --- Enteric Infections ---
    'Diarrheal diseases': {
        'mesh_terms': ['Diarrhea', 'Dysentery', 'Gastroenteritis', 'Cholera', 'Rotavirus Infections', 'Norovirus', 'Salmonella Infections', 'Shigella', 'Campylobacter', 'Cryptosporidiosis'],
        'gbd_level2': 'Enteric infections',
        'global_south_priority': True
    },
    'Typhoid and paratyphoid': {
        'mesh_terms': ['Typhoid Fever', 'Paratyphoid Fever', 'Salmonella typhi'],
        'gbd_level2': 'Enteric infections',
        'global_south_priority': True
    },
    'Invasive Non-typhoidal Salmonella (iNTS)': {
        'mesh_terms': ['Salmonella Infections', 'Salmonella', 'Bacteremia'],
        'gbd_level2': 'Enteric infections',
        'global_south_priority': True
    },
    'Other intestinal infectious diseases': {
        'mesh_terms': ['Intestinal Diseases', 'Intestinal Infections'],
        'gbd_level2': 'Enteric infections',
        'global_south_priority': True
    },
    
    # --- NTDs and Malaria ---
    'Malaria': {
        'mesh_terms': ['Malaria', 'Plasmodium falciparum', 'Plasmodium vivax', 'Malaria, Cerebral', 'Antimalarials', 'Anopheles'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Dengue': {
        'mesh_terms': ['Dengue', 'Dengue Virus', 'Severe Dengue'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Yellow fever': {
        'mesh_terms': ['Yellow Fever', 'Yellow Fever Virus'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Rabies': {
        'mesh_terms': ['Rabies', 'Rabies virus', 'Rabies Vaccines'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Intestinal nematode infections': {
        'mesh_terms': ['Nematode Infections', 'Ascariasis', 'Hookworm Infections', 'Trichuriasis', 'Strongyloidiasis'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Schistosomiasis': {
        'mesh_terms': ['Schistosomiasis', 'Schistosoma'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Leishmaniasis': {
        'mesh_terms': ['Leishmaniasis', 'Leishmania', 'Leishmaniasis, Visceral'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Lymphatic filariasis': {
        'mesh_terms': ['Elephantiasis, Filarial', 'Filariasis', 'Wuchereria bancrofti'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Onchocerciasis': {
        'mesh_terms': ['Onchocerciasis', 'Onchocerca volvulus', 'River Blindness'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Trachoma': {
        'mesh_terms': ['Trachoma', 'Chlamydia trachomatis'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Chagas disease': {
        'mesh_terms': ['Chagas Disease', 'Trypanosoma cruzi'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'African trypanosomiasis': {
        'mesh_terms': ['Trypanosomiasis, African', 'Sleeping Sickness'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Cysticercosis': {
        'mesh_terms': ['Cysticercosis', 'Neurocysticercosis', 'Taenia solium'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Cystic echinococcosis': {
        'mesh_terms': ['Echinococcosis', 'Hydatid Cyst'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Leprosy': {
        'mesh_terms': ['Leprosy', 'Mycobacterium leprae'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Food-borne trematodiases': {
        'mesh_terms': ['Trematode Infections', 'Liver Flukes', 'Clonorchiasis'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Other neglected tropical diseases': {
        'mesh_terms': ['Neglected Diseases', 'Tropical Medicine'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Ebola': {
        'mesh_terms': ['Hemorrhagic Fever, Ebola', 'Ebolavirus'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Zika virus': {
        'mesh_terms': ['Zika Virus Infection', 'Zika Virus'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    'Guinea worm disease': {
        'mesh_terms': ['Dracunculiasis', 'Dracunculus'],
        'gbd_level2': 'NTDs and Malaria',
        'global_south_priority': True
    },
    
    # --- Other Infectious Diseases ---
    'Meningitis': {
        'mesh_terms': ['Meningitis', 'Meningitis, Bacterial', 'Meningitis, Viral', 'Meningitis, Meningococcal', 'Meningitis, Pneumococcal'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Encephalitis': {
        'mesh_terms': ['Encephalitis', 'Encephalitis, Viral', 'Meningoencephalitis'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Acute hepatitis': {
        'mesh_terms': ['Hepatitis', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis E', 'Hepatitis, Viral, Human'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Measles': {
        'mesh_terms': ['Measles', 'Measles virus'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Tetanus': {
        'mesh_terms': ['Tetanus', 'Clostridium tetani'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Pertussis': {
        'mesh_terms': ['Whooping Cough', 'Pertussis', 'Bordetella pertussis'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Diphtheria': {
        'mesh_terms': ['Diphtheria', 'Corynebacterium diphtheriae'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': True
    },
    'Varicella and herpes zoster': {
        'mesh_terms': ['Chickenpox', 'Herpes Zoster', 'Varicella'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': False
    },
    'Other unspecified infectious diseases': {
        'mesh_terms': ['Communicable Diseases', 'Infection', 'Bacterial Infections', 'Virus Diseases', 'Sepsis', 'Bacteremia'],
        'gbd_level2': 'Other infectious',
        'global_south_priority': False
    },
    
    # --- Maternal Disorders ---
    'Maternal disorders': {
        'mesh_terms': ['Pregnancy Complications', 'Pre-Eclampsia', 'Eclampsia', 'Postpartum Hemorrhage', 'Obstetric Labor Complications', 'Maternal Mortality', 'Puerperal Disorders', 'Gestational Diabetes', 'Hypertension, Pregnancy-Induced', 'Maternal Health', 'Prenatal Care', 'Cesarean Section'],
        'gbd_level2': 'Maternal disorders',
        'global_south_priority': True
    },
    
    # --- Neonatal Disorders ---
    'Neonatal disorders': {
        'mesh_terms': ['Infant, Newborn, Diseases', 'Infant, Premature, Diseases', 'Neonatal Sepsis', 'Respiratory Distress Syndrome, Newborn', 'Asphyxia Neonatorum', 'Jaundice, Neonatal', 'Infant, Low Birth Weight', 'Infant Mortality', 'Premature Birth', 'Neonatal', 'Newborn', 'Preterm'],
        'gbd_level2': 'Neonatal disorders',
        'global_south_priority': True
    },
    
    # --- Nutritional Deficiencies ---
    'Protein-energy malnutrition': {
        'mesh_terms': ['Malnutrition', 'Protein-Energy Malnutrition', 'Kwashiorkor', 'Marasmus', 'Wasting Syndrome', 'Cachexia', 'Undernutrition'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    'Dietary iron deficiency': {
        'mesh_terms': ['Anemia, Iron-Deficiency', 'Iron Deficiency'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    'Iodine deficiency': {
        'mesh_terms': ['Iodine Deficiency', 'Goiter, Endemic'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    'Vitamin A deficiency': {
        'mesh_terms': ['Vitamin A Deficiency', 'Night Blindness'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    'Other nutritional deficiencies': {
        'mesh_terms': ['Deficiency Diseases', 'Vitamin Deficiency', 'Zinc Deficiency', 'Folate Deficiency', 'Vitamin B 12 Deficiency', 'Vitamin D Deficiency'],
        'gbd_level2': 'Nutritional deficiencies',
        'global_south_priority': True
    },
    
    # =========================================================================
    # NON-COMMUNICABLE DISEASES
    # =========================================================================
    
    # --- Neoplasms ---
    'Breast cancer': {
        'mesh_terms': ['Breast Neoplasms', 'Carcinoma, Ductal, Breast', 'Triple Negative Breast Neoplasms', 'BRCA1', 'BRCA2'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Cervical cancer': {
        'mesh_terms': ['Uterine Cervical Neoplasms', 'Cervical Cancer', 'Cervical Intraepithelial Neoplasia'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': True
    },
    'Uterine cancer': {
        'mesh_terms': ['Uterine Neoplasms', 'Endometrial Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Ovarian cancer': {
        'mesh_terms': ['Ovarian Neoplasms', 'Ovarian Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Prostate cancer': {
        'mesh_terms': ['Prostatic Neoplasms', 'Prostate Cancer', 'Prostate-Specific Antigen'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Testicular cancer': {
        'mesh_terms': ['Testicular Neoplasms', 'Testicular Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Colon and rectum cancer': {
        'mesh_terms': ['Colorectal Neoplasms', 'Colonic Neoplasms', 'Rectal Neoplasms', 'Colorectal Cancer', 'Colon Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Lip and oral cavity cancer': {
        'mesh_terms': ['Mouth Neoplasms', 'Lip Neoplasms', 'Oral Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Nasopharynx cancer': {
        'mesh_terms': ['Nasopharyngeal Neoplasms', 'Nasopharyngeal Carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Other pharynx cancer': {
        'mesh_terms': ['Pharyngeal Neoplasms', 'Oropharyngeal Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Esophageal cancer': {
        'mesh_terms': ['Esophageal Neoplasms', 'Esophageal Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Stomach cancer': {
        'mesh_terms': ['Stomach Neoplasms', 'Gastric Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Liver cancer': {
        'mesh_terms': ['Liver Neoplasms', 'Carcinoma, Hepatocellular', 'Hepatocellular Carcinoma', 'Cholangiocarcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Gallbladder and biliary tract cancer': {
        'mesh_terms': ['Gallbladder Neoplasms', 'Bile Duct Neoplasms', 'Biliary Tract Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Pancreatic cancer': {
        'mesh_terms': ['Pancreatic Neoplasms', 'Pancreatic Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Larynx cancer': {
        'mesh_terms': ['Laryngeal Neoplasms', 'Laryngeal Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Tracheal, bronchus, and lung cancer': {
        'mesh_terms': ['Lung Neoplasms', 'Carcinoma, Non-Small-Cell Lung', 'Small Cell Lung Carcinoma', 'Bronchial Neoplasms', 'Lung Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Malignant skin melanoma': {
        'mesh_terms': ['Melanoma', 'Melanoma, Cutaneous Malignant'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Non-melanoma skin cancer': {
        'mesh_terms': ['Skin Neoplasms', 'Carcinoma, Basal Cell', 'Carcinoma, Squamous Cell'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Kidney cancer': {
        'mesh_terms': ['Kidney Neoplasms', 'Carcinoma, Renal Cell', 'Renal Cell Carcinoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Bladder cancer': {
        'mesh_terms': ['Urinary Bladder Neoplasms', 'Bladder Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Brain and central nervous system cancer': {
        'mesh_terms': ['Brain Neoplasms', 'Glioma', 'Glioblastoma', 'Meningioma', 'Central Nervous System Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Thyroid cancer': {
        'mesh_terms': ['Thyroid Neoplasms', 'Thyroid Cancer'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Mesothelioma': {
        'mesh_terms': ['Mesothelioma', 'Pleural Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Hodgkin lymphoma': {
        'mesh_terms': ['Hodgkin Disease', 'Hodgkin Lymphoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Non-Hodgkin lymphoma': {
        'mesh_terms': ['Lymphoma, Non-Hodgkin', 'Lymphoma, B-Cell', 'Lymphoma, T-Cell', 'Burkitt Lymphoma', 'Lymphoma, Follicular'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Multiple myeloma': {
        'mesh_terms': ['Multiple Myeloma', 'Plasma Cell Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Leukemia': {
        'mesh_terms': ['Leukemia', 'Leukemia, Myeloid, Acute', 'Leukemia, Lymphocytic, Chronic', 'Leukemia, Myelogenous, Chronic'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Other neoplasms': {
        'mesh_terms': ['Neoplasms', 'Neoplasm', 'Tumor'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Other malignant neoplasms': {
        'mesh_terms': ['Cancer', 'Malignant', 'Carcinoma', 'Sarcoma', 'Metastasis', 'Oncology', 'Antineoplastic Agents'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Malignant neoplasm of bone and articular cartilage': {
        'mesh_terms': ['Bone Neoplasms', 'Osteosarcoma', 'Chondrosarcoma'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Soft tissue and other extraosseous sarcomas': {
        'mesh_terms': ['Sarcoma', 'Soft Tissue Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Eye cancer': {
        'mesh_terms': ['Eye Neoplasms', 'Retinoblastoma', 'Uveal Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    'Neuroblastoma and other peripheral nervous cell tumors': {
        'mesh_terms': ['Neuroblastoma', 'Peripheral Nervous System Neoplasms'],
        'gbd_level2': 'Neoplasms',
        'global_south_priority': False
    },
    
    # --- Cardiovascular Diseases ---
    'Ischemic heart disease': {
        'mesh_terms': ['Myocardial Ischemia', 'Coronary Artery Disease', 'Coronary Disease', 'Angina Pectoris', 'Myocardial Infarction', 'Acute Coronary Syndrome', 'Heart Failure', 'Atherosclerosis'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Stroke': {
        'mesh_terms': ['Stroke', 'Cerebrovascular Disorders', 'Brain Infarction', 'Cerebral Infarction', 'Intracranial Hemorrhages', 'Brain Ischemia', 'Cerebral Hemorrhage'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Hypertensive heart disease': {
        'mesh_terms': ['Hypertension', 'Blood Pressure', 'Essential Hypertension', 'Antihypertensive Agents'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Rheumatic heart disease': {
        'mesh_terms': ['Rheumatic Heart Disease', 'Rheumatic Fever'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': True
    },
    'Cardiomyopathy and myocarditis': {
        'mesh_terms': ['Cardiomyopathies', 'Cardiomyopathy, Dilated', 'Cardiomyopathy, Hypertrophic', 'Myocarditis'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Atrial fibrillation and flutter': {
        'mesh_terms': ['Atrial Fibrillation', 'Atrial Flutter', 'Arrhythmias, Cardiac'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Aortic aneurysm': {
        'mesh_terms': ['Aortic Aneurysm', 'Aortic Aneurysm, Abdominal', 'Aortic Aneurysm, Thoracic', 'Aortic Dissection'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Lower extremity peripheral arterial disease': {
        'mesh_terms': ['Peripheral Arterial Disease', 'Peripheral Vascular Diseases', 'Intermittent Claudication'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Endocarditis': {
        'mesh_terms': ['Endocarditis', 'Endocarditis, Bacterial', 'Heart Valve Diseases'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Non-rheumatic valvular heart disease': {
        'mesh_terms': ['Heart Valve Diseases', 'Aortic Valve Stenosis', 'Mitral Valve Insufficiency'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Other cardiovascular and circulatory diseases': {
        'mesh_terms': ['Cardiovascular Diseases', 'Heart Diseases', 'Vascular Diseases', 'Venous Thromboembolism', 'Pulmonary Embolism', 'Deep Vein Thrombosis'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    'Pulmonary Arterial Hypertension': {
        'mesh_terms': ['Hypertension, Pulmonary', 'Pulmonary Arterial Hypertension'],
        'gbd_level2': 'Cardiovascular diseases',
        'global_south_priority': False
    },
    
    # --- Chronic Respiratory Diseases ---
    'Chronic obstructive pulmonary disease': {
        'mesh_terms': ['Pulmonary Disease, Chronic Obstructive', 'COPD', 'Chronic Bronchitis', 'Emphysema', 'Pulmonary Emphysema'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    'Asthma': {
        'mesh_terms': ['Asthma', 'Status Asthmaticus', 'Bronchial Hyperreactivity'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    'Interstitial lung disease and pulmonary sarcoidosis': {
        'mesh_terms': ['Lung Diseases, Interstitial', 'Pulmonary Fibrosis', 'Sarcoidosis, Pulmonary'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    'Pneumoconiosis': {
        'mesh_terms': ['Pneumoconiosis', 'Asbestosis', 'Silicosis'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    'Other chronic respiratory diseases': {
        'mesh_terms': ['Respiratory Tract Diseases', 'Lung Diseases', 'Sleep Apnea Syndromes', 'Bronchiectasis', 'Cystic Fibrosis'],
        'gbd_level2': 'Chronic respiratory diseases',
        'global_south_priority': False
    },
    
    # --- Digestive Diseases ---
    'Cirrhosis and other chronic liver diseases': {
        'mesh_terms': ['Liver Cirrhosis', 'Liver Diseases', 'Fatty Liver', 'Non-alcoholic Fatty Liver Disease', 'End Stage Liver Disease', 'Liver Failure'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Upper digestive system diseases': {
        'mesh_terms': ['Peptic Ulcer', 'Stomach Ulcer', 'Duodenal Ulcer', 'Gastritis', 'Gastroesophageal Reflux', 'Helicobacter pylori'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Inflammatory bowel disease': {
        'mesh_terms': ['Inflammatory Bowel Diseases', 'Crohn Disease', 'Colitis, Ulcerative'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Paralytic ileus and intestinal obstruction': {
        'mesh_terms': ['Intestinal Obstruction', 'Ileus', 'Intussusception'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Gallbladder and biliary diseases': {
        'mesh_terms': ['Gallbladder Diseases', 'Gallstones', 'Cholecystitis', 'Cholelithiasis', 'Cholangitis'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Pancreatitis': {
        'mesh_terms': ['Pancreatitis', 'Pancreatitis, Acute Necrotizing', 'Pancreatitis, Chronic'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Appendicitis': {
        'mesh_terms': ['Appendicitis', 'Appendectomy'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Vascular intestinal disorders': {
        'mesh_terms': ['Mesenteric Ischemia', 'Colitis, Ischemic'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Inguinal, femoral, and abdominal hernia': {
        'mesh_terms': ['Hernia', 'Hernia, Inguinal', 'Hernia, Ventral', 'Hernia, Abdominal'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    'Other digestive diseases': {
        'mesh_terms': ['Gastrointestinal Diseases', 'Digestive System Diseases', 'Irritable Bowel Syndrome', 'Diverticulitis', 'Celiac Disease'],
        'gbd_level2': 'Digestive diseases',
        'global_south_priority': False
    },
    
    # --- Neurological Disorders ---
    "Alzheimer's disease and other dementias": {
        'mesh_terms': ['Alzheimer Disease', 'Dementia', 'Cognitive Dysfunction', 'Dementia, Vascular', 'Frontotemporal Dementia', 'Lewy Body Disease', 'Mild Cognitive Impairment'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    "Parkinson's disease": {
        'mesh_terms': ['Parkinson Disease', 'Parkinsonian Disorders'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Idiopathic epilepsy': {
        'mesh_terms': ['Epilepsy', 'Seizures', 'Status Epilepticus', 'Anticonvulsants'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Multiple sclerosis': {
        'mesh_terms': ['Multiple Sclerosis', 'Demyelinating Diseases'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Motor neuron disease': {
        'mesh_terms': ['Motor Neuron Disease', 'Amyotrophic Lateral Sclerosis'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Headache disorders': {
        'mesh_terms': ['Headache', 'Migraine Disorders', 'Tension-Type Headache', 'Cluster Headache'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    'Other neurological disorders': {
        'mesh_terms': ['Nervous System Diseases', 'Brain Diseases', 'Neurodegenerative Diseases', 'Polyneuropathies', 'Guillain-Barre Syndrome', 'Myasthenia Gravis', 'Huntington Disease'],
        'gbd_level2': 'Neurological disorders',
        'global_south_priority': False
    },
    
    # --- Mental Disorders ---
    'Depressive disorders': {
        'mesh_terms': ['Depressive Disorder', 'Depression', 'Depressive Disorder, Major', 'Dysthymic Disorder', 'Mood Disorders', 'Antidepressive Agents'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Anxiety disorders': {
        'mesh_terms': ['Anxiety Disorders', 'Anxiety', 'Panic Disorder', 'Phobic Disorders', 'Obsessive-Compulsive Disorder'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Bipolar disorder': {
        'mesh_terms': ['Bipolar Disorder', 'Mania', 'Cyclothymic Disorder'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Schizophrenia': {
        'mesh_terms': ['Schizophrenia', 'Psychotic Disorders', 'Antipsychotic Agents'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Eating disorders': {
        'mesh_terms': ['Eating Disorders', 'Anorexia Nervosa', 'Bulimia Nervosa', 'Binge-Eating Disorder'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Autism spectrum disorders': {
        'mesh_terms': ['Autism Spectrum Disorder', 'Autistic Disorder', 'Asperger Syndrome'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Attention-deficit/hyperactivity disorder': {
        'mesh_terms': ['Attention Deficit Disorder with Hyperactivity', 'ADHD'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Conduct disorder': {
        'mesh_terms': ['Conduct Disorder', 'Child Behavior Disorders'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Idiopathic developmental intellectual disability': {
        'mesh_terms': ['Intellectual Disability', 'Mental Retardation'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    'Other mental disorders': {
        'mesh_terms': ['Mental Disorders', 'Stress Disorders, Post-Traumatic', 'PTSD', 'Personality Disorders', 'Sleep Wake Disorders', 'Insomnia', 'Mental Health'],
        'gbd_level2': 'Mental disorders',
        'global_south_priority': False
    },
    
    # --- Substance Use Disorders ---
    'Alcohol use disorders': {
        'mesh_terms': ['Alcoholism', 'Alcohol-Related Disorders', 'Alcohol Drinking', 'Binge Drinking'],
        'gbd_level2': 'Substance use disorders',
        'global_south_priority': False
    },
    'Drug use disorders': {
        'mesh_terms': ['Substance-Related Disorders', 'Opioid-Related Disorders', 'Drug Overdose', 'Cocaine-Related Disorders', 'Drug Addiction'],
        'gbd_level2': 'Substance use disorders',
        'global_south_priority': False
    },
    
    # --- Diabetes and Kidney Diseases ---
    'Diabetes mellitus': {
        'mesh_terms': ['Diabetes Mellitus', 'Diabetes Mellitus, Type 2', 'Diabetes Mellitus, Type 1', 'Diabetic Complications', 'Diabetic Nephropathies', 'Diabetic Retinopathy', 'Hyperglycemia', 'Insulin Resistance', 'Hemoglobin A, Glycosylated', 'Blood Glucose', 'Insulin', 'Metformin'],
        'gbd_level2': 'Diabetes and kidney diseases',
        'global_south_priority': False
    },
    'Chronic kidney disease': {
        'mesh_terms': ['Kidney Diseases', 'Renal Insufficiency, Chronic', 'Kidney Failure, Chronic', 'Diabetic Nephropathies', 'Glomerulonephritis', 'Renal Dialysis', 'Kidney Transplantation', 'Glomerular Filtration Rate', 'Proteinuria', 'Albuminuria', 'End-Stage Renal Disease', 'Hemodialysis'],
        'gbd_level2': 'Diabetes and kidney diseases',
        'global_south_priority': False
    },
    'Acute glomerulonephritis': {
        'mesh_terms': ['Glomerulonephritis', 'Nephritis'],
        'gbd_level2': 'Diabetes and kidney diseases',
        'global_south_priority': False
    },
    
    # --- Skin Diseases ---
    'Dermatitis': {
        'mesh_terms': ['Dermatitis', 'Eczema', 'Dermatitis, Atopic'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Psoriasis': {
        'mesh_terms': ['Psoriasis', 'Psoriatic Arthritis'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Scabies': {
        'mesh_terms': ['Scabies', 'Sarcoptes scabiei'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': True
    },
    'Fungal skin diseases': {
        'mesh_terms': ['Dermatomycoses', 'Tinea', 'Candidiasis, Cutaneous'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Viral skin diseases': {
        'mesh_terms': ['Warts', 'Molluscum Contagiosum'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Acne vulgaris': {
        'mesh_terms': ['Acne Vulgaris', 'Acne'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Alopecia areata': {
        'mesh_terms': ['Alopecia Areata', 'Alopecia'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Pruritus': {
        'mesh_terms': ['Pruritus', 'Itching'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Urticaria': {
        'mesh_terms': ['Urticaria', 'Hives'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Decubitus ulcer': {
        'mesh_terms': ['Pressure Ulcer', 'Decubitus Ulcer'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Bacterial skin diseases': {
        'mesh_terms': ['Skin Diseases, Bacterial', 'Impetigo', 'Cellulitis'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    'Other skin and subcutaneous diseases': {
        'mesh_terms': ['Skin Diseases', 'Vitiligo', 'Skin Ulcer'],
        'gbd_level2': 'Skin and subcutaneous diseases',
        'global_south_priority': False
    },
    
    # --- Sense Organ Diseases ---
    'Blindness and vision loss': {
        'mesh_terms': ['Blindness', 'Vision Disorders', 'Cataract', 'Glaucoma', 'Macular Degeneration', 'Retinal Diseases', 'Refractive Errors', 'Myopia', 'Eye Diseases'],
        'gbd_level2': 'Sense organ diseases',
        'global_south_priority': False
    },
    'Age-related and other hearing loss': {
        'mesh_terms': ['Hearing Loss', 'Deafness', 'Hearing Loss, Sensorineural', 'Presbycusis', 'Tinnitus', 'Ear Diseases'],
        'gbd_level2': 'Sense organ diseases',
        'global_south_priority': False
    },
    'Other sense organ diseases': {
        'mesh_terms': ['Sensation Disorders', 'Vestibular Diseases', 'Vertigo'],
        'gbd_level2': 'Sense organ diseases',
        'global_south_priority': False
    },
    
    # --- Musculoskeletal Disorders ---
    'Low back pain': {
        'mesh_terms': ['Low Back Pain', 'Back Pain', 'Intervertebral Disc Degeneration', 'Sciatica', 'Spinal Stenosis'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Neck pain': {
        'mesh_terms': ['Neck Pain', 'Cervical Vertebrae', 'Whiplash Injuries'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Osteoarthritis': {
        'mesh_terms': ['Osteoarthritis', 'Osteoarthritis, Knee', 'Osteoarthritis, Hip', 'Joint Diseases'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Rheumatoid arthritis': {
        'mesh_terms': ['Arthritis, Rheumatoid', 'Arthritis', 'Rheumatoid Factor'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Gout': {
        'mesh_terms': ['Gout', 'Hyperuricemia', 'Uric Acid'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    'Other musculoskeletal disorders': {
        'mesh_terms': ['Musculoskeletal Diseases', 'Fibromyalgia', 'Connective Tissue Diseases', 'Lupus Erythematosus, Systemic', 'Osteoporosis', 'Bone Density'],
        'gbd_level2': 'Musculoskeletal disorders',
        'global_south_priority': False
    },
    
    # --- Other NCDs ---
    'Congenital birth defects': {
        'mesh_terms': ['Congenital Abnormalities', 'Birth Defects', 'Heart Defects, Congenital', 'Neural Tube Defects', 'Down Syndrome', 'Cleft Lip', 'Cleft Palate', 'Genetic Diseases, Inborn'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    'Oral disorders': {
        'mesh_terms': ['Mouth Diseases', 'Dental Caries', 'Periodontal Diseases', 'Tooth Loss', 'Gingivitis', 'Periodontitis', 'Oral Health'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    'Sudden infant death syndrome': {
        'mesh_terms': ['Sudden Infant Death', 'SIDS'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    'Urinary diseases and male infertility': {
        'mesh_terms': ['Urologic Diseases', 'Urinary Tract Infections', 'Prostatic Hyperplasia', 'Urinary Incontinence', 'Infertility, Male'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    'Gynecological diseases': {
        'mesh_terms': ['Genital Diseases, Female', 'Endometriosis', 'Uterine Diseases', 'Polycystic Ovary Syndrome', 'Infertility, Female', 'Menopause'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    'Hemoglobinopathies and hemolytic anemias': {
        'mesh_terms': ['Hemoglobinopathies', 'Anemia, Sickle Cell', 'Thalassemia', 'Anemia, Hemolytic'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': True
    },
    'Endocrine, metabolic, blood, and immune disorders': {
        'mesh_terms': ['Endocrine System Diseases', 'Thyroid Diseases', 'Hypothyroidism', 'Hyperthyroidism', 'Metabolic Diseases', 'Obesity', 'Metabolic Syndrome', 'Body Mass Index'],
        'gbd_level2': 'Other non-communicable diseases',
        'global_south_priority': False
    },
    
    # =========================================================================
    # INJURIES
    # =========================================================================
    
    'Road injuries': {
        'mesh_terms': ['Accidents, Traffic', 'Motor Vehicles', 'Motorcycles', 'Bicycling', 'Pedestrians', 'Wounds and Injuries', 'Craniocerebral Trauma', 'Traffic'],
        'gbd_level2': 'Transport injuries',
        'global_south_priority': True
    },
    'Other transport injuries': {
        'mesh_terms': ['Transportation', 'Aviation', 'Ships', 'Railroads'],
        'gbd_level2': 'Transport injuries',
        'global_south_priority': False
    },
    'Falls': {
        'mesh_terms': ['Accidental Falls', 'Hip Fractures', 'Fractures, Bone', 'Spinal Fractures', 'Wrist Fractures'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Drowning': {
        'mesh_terms': ['Drowning', 'Near Drowning', 'Submersion'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': True
    },
    'Fire, heat, and hot substances': {
        'mesh_terms': ['Burns', 'Fires', 'Burns, Inhalation', 'Smoke Inhalation Injury'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': True
    },
    'Poisonings': {
        'mesh_terms': ['Poisoning', 'Drug Overdose', 'Carbon Monoxide Poisoning', 'Lead Poisoning'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Exposure to mechanical forces': {
        'mesh_terms': ['Wounds and Injuries', 'Occupational Injuries', 'Sports Injuries', 'Sprains and Strains'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Adverse effects of medical treatment': {
        'mesh_terms': ['Medical Errors', 'Drug-Related Side Effects and Adverse Reactions', 'Iatrogenic Disease'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Animal contact': {
        'mesh_terms': ['Bites and Stings', 'Snake Bites', 'Insect Bites and Stings'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': True
    },
    'Foreign body': {
        'mesh_terms': ['Foreign Bodies', 'Airway Obstruction', 'Choking'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Environmental heat and cold exposure': {
        'mesh_terms': ['Heat Stroke', 'Hypothermia', 'Frostbite'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Other unintentional injuries': {
        'mesh_terms': ['Accidents', 'Trauma', 'Emergency Medicine'],
        'gbd_level2': 'Unintentional injuries',
        'global_south_priority': False
    },
    'Self-harm': {
        'mesh_terms': ['Suicide', 'Self-Injurious Behavior', 'Suicide, Attempted', 'Suicidal Ideation'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': False
    },
    'Interpersonal violence': {
        'mesh_terms': ['Violence', 'Domestic Violence', 'Child Abuse', 'Homicide', 'Wounds, Gunshot', 'Sexual Assault'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': True
    },
    'Conflict and terrorism': {
        'mesh_terms': ['Warfare', 'Terrorism', 'Armed Conflicts', 'War'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': True
    },
    'Police conflict and executions': {
        'mesh_terms': ['Violence', 'Law Enforcement'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': True
    },
    'Exposure to forces of nature': {
        'mesh_terms': ['Natural Disasters', 'Earthquakes', 'Floods', 'Cyclonic Storms'],
        'gbd_level2': 'Self-harm and violence',
        'global_south_priority': True
    },
    
    # =========================================================================
    # RESEARCH METHODOLOGY (to capture methodology papers)
    # =========================================================================
    
    '_GENOMICS': {
        'mesh_terms': ['Genome-Wide Association Study', 'Genetic Predisposition to Disease', 'Polymorphism, Single Nucleotide', 'Genetic Variation', 'Genomics', 'Exome', 'Whole Genome Sequencing', 'Mendelian Randomization Analysis', 'Genotype', 'Phenotype', 'Alleles'],
        'gbd_level2': 'Research methodology',
        'global_south_priority': False
    },
    '_BIOBANK_METHODS': {
        'mesh_terms': ['Biological Specimen Banks', 'Cohort Studies', 'Prospective Studies', 'Longitudinal Studies', 'Risk Factors', 'Epidemiologic Studies', 'Cross-Sectional Studies', 'Case-Control Studies', 'Population Health', 'Registries', 'Biomarkers'],
        'gbd_level2': 'Research methodology',
        'global_south_priority': False
    },
    '_IMAGING': {
        'mesh_terms': ['Magnetic Resonance Imaging', 'Brain', 'Neuroimaging', 'Diagnostic Imaging', 'Ultrasonography', 'Computed Tomography', 'Positron-Emission Tomography'],
        'gbd_level2': 'Research methodology',
        'global_south_priority': False
    },
    '_AGING': {
        'mesh_terms': ['Aging', 'Aged', 'Aged, 80 and over', 'Longevity', 'Life Expectancy', 'Frailty', 'Sarcopenia', 'Telomere'],
        'gbd_level2': 'Research methodology',
        'global_south_priority': False
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_gbd_file(data_dir: Path) -> Path:
    """Find the official GBD data file."""
    patterns = [
        'IHMEGBD_2021*.csv',
        'GBD_2021*.csv',
        'gbd*.csv'
    ]
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            return files[0]
    return None


def load_official_gbd_dalys(gbd_file: Path) -> dict:
    """Load official DALYs from IHME GBD export."""
    logger.info(f"Loading official GBD DALYs from: {gbd_file}")
    
    df = pd.read_csv(gbd_file)
    
    # Filter for DALYs, Number metric
    dalys_df = df[
        (df['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
        (df['metric_name'] == 'Number')
    ]
    
    # Create dict: cause_name -> DALYs value
    gbd_dalys = {}
    for _, row in dalys_df.iterrows():
        cause_name = str(row['cause_name']).strip()
        dalys_val = float(row['val'])
        gbd_dalys[cause_name] = dalys_val
    
    logger.info(f"  Loaded official DALYs for {len(gbd_dalys)} causes")
    return gbd_dalys


def normalize_term(term: str) -> str:
    """Normalize MeSH term for matching."""
    if not term:
        return ''
    return term.lower().strip()


def map_publication_to_causes(mesh_terms_str: str, mapping: dict) -> list:
    """Map a publication's MeSH terms to GBD causes."""
    if pd.isna(mesh_terms_str) or not str(mesh_terms_str).strip():
        return []
    
    # Parse and normalize article's MeSH terms
    article_terms = set()
    for t in str(mesh_terms_str).split(';'):
        normalized = normalize_term(t)
        if normalized:
            article_terms.add(normalized)
    
    if not article_terms:
        return []
    
    matched_causes = []
    
    for gbd_cause, info in mapping.items():
        cause_mesh_terms = [normalize_term(t) for t in info['mesh_terms']]
        
        for mesh_term in cause_mesh_terms:
            if not mesh_term:
                continue
            
            # Exact match
            if mesh_term in article_terms:
                matched_causes.append(gbd_cause)
                break
            
            # Substring match (for flexibility)
            for article_term in article_terms:
                if len(mesh_term) >= 4 and mesh_term in article_term:
                    matched_causes.append(gbd_cause)
                    break
                if len(article_term) >= 4 and article_term in mesh_term:
                    matched_causes.append(gbd_cause)
                    break
            else:
                continue
            break
    
    return list(set(matched_causes))


def get_gbd_level2(causes: list, mapping: dict) -> list:
    """Get GBD Level 2 categories for causes."""
    levels = set()
    for cause in causes:
        if cause in mapping:
            levels.add(mapping[cause].get('gbd_level2', 'Unknown'))
    return list(levels)


def is_global_south_priority(causes: list, mapping: dict) -> bool:
    """Check if any cause is a Global South priority."""
    for cause in causes:
        if cause in mapping:
            if mapping[cause].get('global_south_priority', False):
                return True
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BHEM: Map Publications to Official GBD 2021 Categories")
    print("Using IHME official DALYs - NO invented values")
    print("=" * 70)
    
    # Check input file
    if not INPUT_FILE.exists():
        print(f"\n❌ Input not found: {INPUT_FILE}")
        return
    
    # Load publications
    print(f"\n📂 Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   {len(df):,} publications")
    
    # Find MeSH column
    mesh_col = None
    for col in ['mesh_terms', 'MeSH_Terms', 'MeSH Terms']:
        if col in df.columns:
            mesh_col = col
            break
    
    if not mesh_col:
        print("❌ No MeSH column found!")
        return
    
    print(f"   Using column: '{mesh_col}'")
    
    # Load official GBD DALYs
    gbd_file = find_gbd_file(DATA_DIR)
    gbd_dalys = {}
    if gbd_file:
        gbd_dalys = load_official_gbd_dalys(gbd_file)
    else:
        print("⚠️  GBD file not found - DALYs will be 0")
        print("   Copy IHMEGBD_2021_DATA*.csv to DATA/ folder")
    
    # Count causes in our mapping
    n_causes = len(GBD_MESH_MAPPING)
    print(f"\n🔬 Mapping to {n_causes} GBD causes...")
    
    # Map publications to GBD causes
    df['gbd_causes'] = df[mesh_col].apply(
        lambda x: map_publication_to_causes(x, GBD_MESH_MAPPING)
    )
    df['cause_count'] = df['gbd_causes'].apply(len)
    df['gbd_level2'] = df['gbd_causes'].apply(
        lambda x: get_gbd_level2(x, GBD_MESH_MAPPING)
    )
    df['global_south_priority'] = df['gbd_causes'].apply(
        lambda x: is_global_south_priority(x, GBD_MESH_MAPPING)
    )
    
    # Convert to strings for CSV
    df['gbd_causes_str'] = df['gbd_causes'].apply(lambda x: '|'.join(x) if x else '')
    df['gbd_level2_str'] = df['gbd_level2'].apply(lambda x: '|'.join(x) if x else '')
    
    # Calculate statistics
    mapped = (df['cause_count'] > 0).sum()
    unmapped = (df['cause_count'] == 0).sum()
    gs_count = df['global_south_priority'].sum()
    
    print(f"\n📊 MAPPING RESULTS:")
    print(f"   ✅ Mapped: {mapped:,} ({mapped/len(df)*100:.1f}%)")
    print(f"   ○  Unmapped: {unmapped:,} ({unmapped/len(df)*100:.1f}%)")
    print(f"   🌍 Global South priority: {gs_count:,}")
    
    # Count publications per cause
    cause_counts = defaultdict(int)
    for causes in df['gbd_causes']:
        for cause in causes:
            cause_counts[cause] += 1
    
    # Build registry with OFFICIAL DALYs
    registry = {}
    for cause, info in GBD_MESH_MAPPING.items():
        # Get official DALYs (0 if not found)
        dalys = gbd_dalys.get(cause, 0)
        
        registry[cause] = {
            'name': cause,
            'gbd_level2': info['gbd_level2'],
            'global_south_priority': info['global_south_priority'],
            'dalys': dalys,  # OFFICIAL value from IHME
            'publications': cause_counts.get(cause, 0)
        }
    
    # Print top 40 causes
    print(f"\n📋 Top 40 GBD causes by publications:")
    sorted_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)
    for cause, count in sorted_causes[:40]:
        dalys = gbd_dalys.get(cause, 0) / 1e6
        gs = "🌍" if GBD_MESH_MAPPING.get(cause, {}).get('global_south_priority') else "  "
        if dalys > 0:
            print(f"   {gs} {cause[:50]}: {count:,} pubs ({dalys:.1f}M DALYs)")
        else:
            print(f"   {gs} {cause[:50]}: {count:,} pubs")
    
    # Level 2 distribution
    print(f"\n📊 GBD Level 2 distribution:")
    level2_counts = defaultdict(int)
    for levels in df['gbd_level2']:
        for l in levels:
            level2_counts[l] += 1
    for level, count in sorted(level2_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {level}: {count:,}")
    
    # Causes with 0 pubs
    zero_pubs = [c for c in GBD_MESH_MAPPING if cause_counts.get(c, 0) == 0 and not c.startswith('_')]
    if zero_pubs:
        print(f"\n⚠️  {len(zero_pubs)} causes with 0 publications:")
        for c in zero_pubs[:15]:
            dalys = gbd_dalys.get(c, 0) / 1e6
            if dalys > 0:
                print(f"   - {c} ({dalys:.1f}M DALYs)")
            else:
                print(f"   - {c}")
        if len(zero_pubs) > 15:
            print(f"   ... and {len(zero_pubs) - 15} more")
    
    # Save outputs
    print(f"\n💾 Saving: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"💾 Saving: {REGISTRY_FILE}")
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"✅ {mapped:,}/{len(df):,} publications mapped ({mapped/len(df)*100:.1f}%)")
    print(f"✅ {n_causes} GBD causes defined")
    print(f"✅ {len(cause_counts)} causes have publications")
    if gbd_dalys:
        print(f"✅ Official DALYs loaded for {len(gbd_dalys)} causes")
    
    print(f"\n➡️  Next: python 02-02-bhem-analyze-themes.py")


if __name__ == "__main__":
    main()