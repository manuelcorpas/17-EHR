"""
HEIM Reference Population Data

Contains benchmark distributions for comparing dataset representation.
Based on 1000 Genomes Project and global demographics.
"""

from typing import Dict


# Global ancestry distribution (approximate from 1000 Genomes Project)
GLOBAL_ANCESTRY = {
    'AFR': 0.26,  # African
    'EUR': 0.20,  # European
    'EAS': 0.26,  # East Asian
    'SAS': 0.20,  # South Asian
    'AMR': 0.08,  # Admixed American
}

# Continental ancestry distribution (coarse)
CONTINENTAL_ANCESTRY = {
    'AFR': 0.17,  # Africa population
    'EUR': 0.10,  # Europe population
    'EAS': 0.24,  # East Asia population
    'SAS': 0.24,  # South Asia population
    'AMR': 0.13,  # Americas population
    'MID': 0.06,  # Middle East population
    'OCE': 0.01,  # Oceania population
}

# 1000 Genomes fine-grained populations
FINE_GRAINED_ANCESTRY = {
    # African
    'YRI': 0.031,  # Yoruba in Ibadan, Nigeria
    'LWK': 0.029,  # Luhya in Webuye, Kenya
    'GWD': 0.033,  # Gambian in Western Divisions
    'MSL': 0.025,  # Mende in Sierra Leone
    'ESN': 0.030,  # Esan in Nigeria
    'ASW': 0.018,  # African Ancestry in Southwest US
    'ACB': 0.028,  # African Caribbean in Barbados
    
    # European
    'CEU': 0.029,  # Utah residents (CEPH) with Northern/Western European ancestry
    'TSI': 0.031,  # Toscani in Italia
    'FIN': 0.030,  # Finnish in Finland
    'GBR': 0.027,  # British in England and Scotland
    'IBS': 0.031,  # Iberian populations in Spain
    
    # East Asian
    'CHB': 0.030,  # Han Chinese in Beijing, China
    'JPT': 0.031,  # Japanese in Tokyo, Japan
    'CHS': 0.032,  # Han Chinese South
    'CDX': 0.028,  # Chinese Dai in Xishuangbanna, China
    'KHV': 0.030,  # Kinh in Ho Chi Minh City, Vietnam
    
    # South Asian
    'GIH': 0.031,  # Gujarati Indian in Houston, TX
    'PJL': 0.028,  # Punjabi in Lahore, Pakistan
    'BEB': 0.026,  # Bengali in Bangladesh
    'STU': 0.030,  # Sri Lankan Tamil in the UK
    'ITU': 0.031,  # Indian Telugu in the UK
    
    # Admixed American
    'MXL': 0.019,  # Mexican Ancestry in Los Angeles, CA
    'PUR': 0.032,  # Puerto Rican in Puerto Rico
    'CLM': 0.028,  # Colombian in Medellin, Colombia
    'PEL': 0.025,  # Peruvian in Lima, Peru
}


# Disease-specific demographics (examples)
DISEASE_DEMOGRAPHICS = {
    'breast_cancer': {
        'age_median': 61,
        'age_range': 50,
        'sex_female': 0.99,
        'ancestry_risk': {
            'AFR': 'elevated',
            'EUR': 'baseline',
            'EAS': 'lower',
        }
    },
    'diabetes_t2': {
        'age_median': 58,
        'age_range': 40,
        'sex_female': 0.48,
        'ancestry_risk': {
            'SAS': 'elevated',
            'AMR': 'elevated',
            'AFR': 'elevated',
        }
    },
    'cardiovascular': {
        'age_median': 65,
        'age_range': 30,
        'sex_female': 0.45,
        'ancestry_risk': {
            'AFR': 'elevated',
            'SAS': 'elevated',
        }
    },
    'alzheimers': {
        'age_median': 75,
        'age_range': 20,
        'sex_female': 0.65,
        'ancestry_risk': {
            'AFR': 'elevated',
            'AMR': 'elevated',
        }
    },
}


def get_reference_ancestry(granularity: str = 'coarse') -> Dict[str, float]:
    """
    Get reference ancestry distribution.
    
    Args:
        granularity: 'coarse' for 5 categories, 'continental' for 7, 'fine' for 26
        
    Returns:
        Dictionary mapping ancestry codes to expected proportions
    """
    if granularity == 'fine':
        return FINE_GRAINED_ANCESTRY.copy()
    elif granularity == 'continental':
        return CONTINENTAL_ANCESTRY.copy()
    else:
        return GLOBAL_ANCESTRY.copy()


def get_disease_reference(disease: str) -> Dict:
    """
    Get reference demographics for a specific disease.
    
    Args:
        disease: Disease name (e.g., 'breast_cancer', 'diabetes_t2')
        
    Returns:
        Dictionary with expected demographics
    """
    return DISEASE_DEMOGRAPHICS.get(disease, {
        'age_median': 50,
        'age_range': 40,
        'sex_female': 0.5,
    })


# Ancestry code mappings (for user convenience)
ANCESTRY_NAMES = {
    'AFR': 'African',
    'EUR': 'European',
    'EAS': 'East Asian',
    'SAS': 'South Asian',
    'AMR': 'Admixed American',
    'MID': 'Middle Eastern',
    'OCE': 'Oceanian',
    'OTH': 'Other',
}

# Fine-grained population names
POPULATION_NAMES = {
    'YRI': 'Yoruba (Nigeria)',
    'LWK': 'Luhya (Kenya)',
    'GWD': 'Gambian',
    'MSL': 'Mende (Sierra Leone)',
    'ESN': 'Esan (Nigeria)',
    'ASW': 'African American (Southwest US)',
    'ACB': 'African Caribbean (Barbados)',
    'CEU': 'Northern/Western European',
    'TSI': 'Tuscan (Italy)',
    'FIN': 'Finnish',
    'GBR': 'British',
    'IBS': 'Iberian (Spain)',
    'CHB': 'Han Chinese (Beijing)',
    'JPT': 'Japanese (Tokyo)',
    'CHS': 'Han Chinese (South)',
    'CDX': 'Chinese Dai',
    'KHV': 'Kinh (Vietnam)',
    'GIH': 'Gujarati Indian',
    'PJL': 'Punjabi (Pakistan)',
    'BEB': 'Bengali (Bangladesh)',
    'STU': 'Sri Lankan Tamil',
    'ITU': 'Indian Telugu',
    'MXL': 'Mexican American',
    'PUR': 'Puerto Rican',
    'CLM': 'Colombian',
    'PEL': 'Peruvian',
}