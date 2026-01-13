"""
HEIM Composite Scoring Module

Combines individual diversity metrics into overall HEIM Representation Score
and assigns badge levels (Platinum, Gold, Silver, Bronze).
"""

import pandas as pd
from typing import Dict, Tuple
from metrics import calculate_all_metrics, representation_gap
from reference_data import get_reference_ancestry


# Scoring weights for composite HEIM score
WEIGHTS = {
    'ancestry': 0.50,      # 50% - Most important dimension
    'geographic': 0.20,    # 20% - Geographic spread
    'age': 0.15,          # 15% - Age distribution
    'sex': 0.15,          # 15% - Sex balance
}

# Badge thresholds
BADGE_THRESHOLDS = {
    'Platinum': 90,
    'Gold': 75,
    'Silver': 60,
    'Bronze': 40,
}


def normalize_to_100(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a value to 0-100 scale.
    
    Args:
        value: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value
        
    Returns:
        Normalized value between 0 and 100
    """
    if max_val == min_val:
        return 50.0
    
    normalized = ((value - min_val) / (max_val - min_val)) * 100
    return max(0.0, min(100.0, normalized))


def calculate_ancestry_score(df: pd.DataFrame, reference_type: str = 'coarse') -> Tuple[float, Dict]:
    """
    Calculate ancestry representation score.
    
    Combines Simpson's Diversity, evenness, and representation gap.
    
    Args:
        df: DataFrame with ancestry column
        reference_type: Type of reference population ('coarse', 'continental', 'fine')
        
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if 'ancestry' not in df.columns:
        return 0.0, {'error': 'No ancestry column'}
    
    ancestry = df['ancestry'].dropna()
    
    if len(ancestry) == 0:
        return 0.0, {'error': 'No ancestry data'}
    
    # Get metrics
    from metrics import simpsons_diversity_index, pielou_evenness
    
    diversity = simpsons_diversity_index(ancestry)
    evenness = pielou_evenness(ancestry)
    
    # Calculate representation gap
    reference = get_reference_ancestry(reference_type)
    gap = representation_gap(ancestry, reference)
    
    # Normalize gap (0 = perfect, 2 = worst) to 0-100 (inverted)
    gap_score = normalize_to_100(2 - gap, 0, 2)
    
    # Composite ancestry score
    diversity_score = normalize_to_100(diversity, 0, 1)
    evenness_score = normalize_to_100(evenness, 0, 1)
    
    # Weighted combination
    score = (
        diversity_score * 0.4 +
        evenness_score * 0.3 +
        gap_score * 0.3
    )
    
    details = {
        'diversity': diversity,
        'diversity_score': diversity_score,
        'evenness': evenness,
        'evenness_score': evenness_score,
        'representation_gap': gap,
        'gap_score': gap_score,
        'unique_ancestries': ancestry.nunique()
    }
    
    return score, details


def calculate_geographic_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Calculate geographic diversity score.
    
    Args:
        df: DataFrame with country column
        
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if 'country' not in df.columns:
        return 0.0, {'error': 'No country column'}
    
    from metrics import geographic_diversity_score, simpsons_diversity_index
    
    countries = df['country'].dropna()
    
    if len(countries) == 0:
        return 0.0, {'error': 'No country data'}
    
    geo_score = geographic_diversity_score(countries)
    diversity = simpsons_diversity_index(countries)
    diversity_score = normalize_to_100(diversity, 0, 1)
    
    # Combine
    score = (geo_score * 0.6) + (diversity_score * 0.4)
    
    details = {
        'unique_countries': countries.nunique(),
        'diversity': diversity,
        'geographic_score': geo_score
    }
    
    return score, details


def calculate_age_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Calculate age distribution score.
    
    Args:
        df: DataFrame with age column
        
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if 'age' not in df.columns:
        return 0.0, {'error': 'No age column'}
    
    from metrics import age_distribution_score
    
    ages = df['age'].dropna()
    
    if len(ages) == 0:
        return 0.0, {'error': 'No age data'}
    
    score = age_distribution_score(ages)
    
    details = {
        'mean': float(ages.mean()),
        'median': float(ages.median()),
        'min': float(ages.min()),
        'max': float(ages.max()),
        'range': float(ages.max() - ages.min()),
        'std': float(ages.std())
    }
    
    return score, details


def calculate_sex_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Calculate sex/gender balance score.
    
    Args:
        df: DataFrame with sex column
        
    Returns:
        Tuple of (score 0-100, details dict)
    """
    if 'sex' not in df.columns:
        return 0.0, {'error': 'No sex column'}
    
    from metrics import sex_balance_score, simpsons_diversity_index
    
    sex = df['sex'].dropna()
    
    if len(sex) == 0:
        return 0.0, {'error': 'No sex data'}
    
    balance_score = sex_balance_score(sex)
    diversity = simpsons_diversity_index(sex)
    
    # Normalize and combine
    diversity_score = normalize_to_100(diversity, 0, 1)
    score = (balance_score * 0.7) + (diversity_score * 0.3)
    
    # Calculate proportions
    sex_normalized = sex.str.upper().replace({
        'MALE': 'M',
        'FEMALE': 'F',
        'UNKNOWN': 'U',
        'OTHER': 'O'
    })
    
    sex_counts = sex_normalized.value_counts()
    
    details = {
        'balance_score': balance_score,
        'diversity': diversity,
        'distribution': sex_counts.to_dict()
    }
    
    return score, details


def calculate_heim_score(df: pd.DataFrame, reference_type: str = 'coarse') -> Dict:
    """
    Calculate overall HEIM Representation Score.
    
    Combines all dimension scores into composite 0-100 score.
    
    Args:
        df: DataFrame with demographic columns
        reference_type: Type of reference population for ancestry
        
    Returns:
        Dictionary with overall score, badge, and dimension scores
    """
    # Calculate dimension scores
    ancestry_score, ancestry_details = calculate_ancestry_score(df, reference_type)
    geographic_score, geographic_details = calculate_geographic_score(df)
    age_score, age_details = calculate_age_score(df)
    sex_score, sex_details = calculate_sex_score(df)
    
    # Weighted composite
    composite_score = (
        ancestry_score * WEIGHTS['ancestry'] +
        geographic_score * WEIGHTS['geographic'] +
        age_score * WEIGHTS['age'] +
        sex_score * WEIGHTS['sex']
    )
    
    # Determine badge
    badge = 'Needs Improvement'
    for badge_name, threshold in BADGE_THRESHOLDS.items():
        if composite_score >= threshold:
            badge = badge_name
            break
    
    # Badge color
    badge_colors = {
        'Platinum': '#E5E4E2',
        'Gold': '#FFD700',
        'Silver': '#C0C0C0',
        'Bronze': '#CD7F32',
        'Needs Improvement': '#808080'
    }
    
    result = {
        'overall_score': round(composite_score, 1),
        'badge': badge,
        'badge_color': badge_colors[badge],
        'dimensions': {
            'ancestry': {
                'score': round(ancestry_score, 1),
                'weight': WEIGHTS['ancestry'],
                'details': ancestry_details
            },
            'geographic': {
                'score': round(geographic_score, 1),
                'weight': WEIGHTS['geographic'],
                'details': geographic_details
            },
            'age': {
                'score': round(age_score, 1),
                'weight': WEIGHTS['age'],
                'details': age_details
            },
            'sex': {
                'score': round(sex_score, 1),
                'weight': WEIGHTS['sex'],
                'details': sex_details
            }
        },
        'sample_size': len(df)
    }
    
    return result


def get_badge_interpretation(badge: str) -> str:
    """Get human-readable interpretation of badge level."""
    interpretations = {
        'Platinum': 'Exceptional diversity and representation across all dimensions.',
        'Gold': 'Strong representation with minor gaps in some dimensions.',
        'Silver': 'Good diversity but notable gaps requiring attention.',
        'Bronze': 'Moderate diversity with significant gaps in multiple dimensions.',
        'Needs Improvement': 'Substantial representation gaps requiring major improvements.'
    }
    return interpretations.get(badge, '')


def get_score_color(score: float) -> str:
    """Get color for score visualization."""
    if score >= 90:
        return '#00C851'  # Green
    elif score >= 75:
        return '#FFD700'  # Gold
    elif score >= 60:
        return '#C0C0C0'  # Silver
    elif score >= 40:
        return '#CD7F32'  # Bronze
    else:
        return '#FF4444'  # Red