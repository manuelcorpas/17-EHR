"""
HEIM Diversity Metrics Module

Implements core diversity indices for quantifying representation
and balance across demographic dimensions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import math


def simpsons_diversity_index(values: pd.Series) -> float:
    """
    Calculate Simpson's Diversity Index.
    
    D = 1 - Σ(pi²)
    where pi = proportion of category i
    
    Range: 0 (no diversity) to 1 (maximum diversity)
    
    Args:
        values: Pandas Series of categorical values
        
    Returns:
        Diversity index between 0 and 1
    """
    # Remove null values
    values = values.dropna()
    
    if len(values) == 0:
        return 0.0
    
    # Calculate proportions
    value_counts = values.value_counts()
    proportions = value_counts / len(values)
    
    # Simpson's D = 1 - sum of squared proportions
    diversity = 1 - (proportions ** 2).sum()
    
    return float(diversity)


def pielou_evenness(values: pd.Series) -> float:
    """
    Calculate Pielou's Evenness Index (J).
    
    J = H / Hmax
    where H = Shannon entropy, Hmax = ln(S), S = number of categories
    
    Range: 0 (uneven) to 1 (perfectly even)
    
    Args:
        values: Pandas Series of categorical values
        
    Returns:
        Evenness index between 0 and 1
    """
    # Remove null values
    values = values.dropna()
    
    if len(values) == 0:
        return 0.0
    
    # Get unique categories
    value_counts = values.value_counts()
    num_categories = len(value_counts)
    
    # Need at least 2 categories for evenness
    if num_categories < 2:
        return 1.0
    
    # Calculate Shannon entropy
    proportions = value_counts / len(values)
    # Avoid log(0) by filtering out zero proportions
    proportions = proportions[proportions > 0]
    shannon_entropy = -(proportions * np.log(proportions)).sum()
    
    # Maximum possible entropy
    max_entropy = np.log(num_categories)
    
    # Evenness
    if max_entropy == 0:
        return 1.0
    
    evenness = shannon_entropy / max_entropy
    
    return float(evenness)


def representation_gap(observed: pd.Series, expected: Dict[str, float]) -> float:
    """
    Calculate representation gap between observed and expected distributions.
    
    Gap = Σ|observed_i - expected_i|
    
    Lower values indicate better representation.
    
    Args:
        observed: Pandas Series of observed values
        expected: Dictionary mapping categories to expected proportions (should sum to 1.0)
        
    Returns:
        Total absolute difference (0 = perfect match, 2 = maximum mismatch)
    """
    # Remove null values
    observed = observed.dropna()
    
    if len(observed) == 0:
        return 1.0  # Maximum gap for empty data
    
    # Calculate observed proportions
    observed_counts = observed.value_counts()
    observed_props = observed_counts / len(observed)
    
    # Calculate total absolute difference
    total_gap = 0.0
    
    # Check each expected category
    for category, expected_prop in expected.items():
        observed_prop = observed_props.get(category, 0.0)
        total_gap += abs(observed_prop - expected_prop)
    
    # Also account for categories in observed but not in expected
    for category in observed_props.index:
        if category not in expected:
            total_gap += observed_props[category]
    
    return float(total_gap)


def age_distribution_score(ages: pd.Series, expected_median: float = 50.0, expected_range: float = 40.0) -> float:
    """
    Score age distribution based on range and spread.
    
    Higher scores for wider ranges and better spread.
    
    Args:
        ages: Pandas Series of age values
        expected_median: Expected median age for the population
        expected_range: Expected age range
        
    Returns:
        Score between 0 and 100
    """
    # Remove null values
    ages = ages.dropna()
    
    if len(ages) == 0:
        return 0.0
    
    # Calculate range
    age_range = ages.max() - ages.min()
    
    # Range score (wider is better, up to expected range)
    range_score = min(age_range / expected_range, 1.0) * 50
    
    # Spread score based on standard deviation
    age_std = ages.std()
    expected_std = expected_range / 4  # Rough estimate
    spread_score = min(age_std / expected_std, 1.0) * 30
    
    # Distribution score (higher for more even distribution)
    # Use coefficient of variation
    age_mean = ages.mean()
    if age_mean > 0:
        cv = age_std / age_mean
        distribution_score = min(cv / 0.3, 1.0) * 20  # 0.3 is reasonable CV
    else:
        distribution_score = 0
    
    total_score = range_score + spread_score + distribution_score
    
    return float(min(total_score, 100.0))


def sex_balance_score(sex: pd.Series, expected_female: float = 0.5) -> float:
    """
    Score sex/gender balance.
    
    Perfect balance (50/50) scores highest unless expected distribution differs.
    
    Args:
        sex: Pandas Series of sex/gender values
        expected_female: Expected proportion of females (default 0.5)
        
    Returns:
        Score between 0 and 100
    """
    # Remove null values
    sex = sex.dropna()
    
    if len(sex) == 0:
        return 0.0
    
    # Normalize sex values
    sex_normalized = sex.str.upper().replace({
        'MALE': 'M',
        'FEMALE': 'F',
        'UNKNOWN': 'U',
        'OTHER': 'O'
    })
    
    # Count female proportion
    female_count = sex_normalized.isin(['F', 'FEMALE']).sum()
    female_prop = female_count / len(sex)
    
    # Calculate deviation from expected
    deviation = abs(female_prop - expected_female)
    
    # Score: 100 for perfect match, decreases with deviation
    # Maximum deviation is 1.0 (100% one sex)
    score = (1 - deviation) * 100
    
    return float(score)


def geographic_diversity_score(countries: pd.Series) -> float:
    """
    Score geographic diversity based on number of unique countries
    and continental representation.
    
    Args:
        countries: Pandas Series of country names
        
    Returns:
        Score between 0 and 100
    """
    # Remove null values
    countries = countries.dropna()
    
    if len(countries) == 0:
        return 0.0
    
    # Count unique countries
    unique_countries = countries.nunique()
    
    # Score based on number of countries
    # 1 country = 20, 5 countries = 40, 10+ countries = 60, 20+ countries = 80, 30+ = 100
    if unique_countries >= 30:
        country_score = 100
    elif unique_countries >= 20:
        country_score = 80
    elif unique_countries >= 10:
        country_score = 60 + (unique_countries - 10) * 2
    elif unique_countries >= 5:
        country_score = 40 + (unique_countries - 5) * 4
    else:
        country_score = 20 + unique_countries * 4
    
    return float(min(country_score, 100.0))


def calculate_all_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate all diversity metrics for a dataset.
    
    Args:
        df: DataFrame with columns: ancestry, age, sex, country
        
    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}
    
    # Ancestry metrics
    if 'ancestry' in df.columns:
        metrics['ancestry_diversity'] = simpsons_diversity_index(df['ancestry'])
        metrics['ancestry_evenness'] = pielou_evenness(df['ancestry'])
    
    # Geographic metrics
    if 'country' in df.columns:
        metrics['geographic_diversity'] = simpsons_diversity_index(df['country'])
        metrics['geographic_score'] = geographic_diversity_score(df['country'])
        metrics['unique_countries'] = df['country'].nunique()
    
    # Age metrics
    if 'age' in df.columns:
        metrics['age_score'] = age_distribution_score(df['age'])
        metrics['age_range'] = float(df['age'].max() - df['age'].min()) if len(df['age'].dropna()) > 0 else 0
        metrics['age_mean'] = float(df['age'].mean()) if len(df['age'].dropna()) > 0 else 0
        metrics['age_std'] = float(df['age'].std()) if len(df['age'].dropna()) > 0 else 0
    
    # Sex metrics
    if 'sex' in df.columns:
        metrics['sex_balance'] = sex_balance_score(df['sex'])
        metrics['sex_diversity'] = simpsons_diversity_index(df['sex'])
    
    return metrics