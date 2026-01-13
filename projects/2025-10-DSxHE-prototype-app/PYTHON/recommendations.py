"""
HEIM Recommendations Module

Generates actionable recommendations based on diversity gaps.
"""

import pandas as pd
from typing import Dict, List, Tuple
from reference_data import ANCESTRY_NAMES, get_reference_ancestry


class Recommendation:
    """Structure for a single recommendation."""
    
    def __init__(
        self,
        dimension: str,
        severity: str,
        title: str,
        description: str,
        actions: List[str],
        priority: int
    ):
        self.dimension = dimension
        self.severity = severity  # 'critical', 'high', 'moderate', 'low'
        self.title = title
        self.description = description
        self.actions = actions
        self.priority = priority
    
    def to_dict(self) -> Dict:
        return {
            'dimension': self.dimension,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'actions': self.actions,
            'priority': self.priority
        }


def analyze_ancestry_gaps(df: pd.DataFrame, heim_result: Dict) -> List[Recommendation]:
    """Analyze ancestry representation and generate recommendations."""
    recommendations = []
    
    ancestry_score = heim_result['dimensions']['ancestry']['score']
    ancestry_details = heim_result['dimensions']['ancestry']['details']
    
    # Get observed distribution
    ancestry_counts = df['ancestry'].value_counts()
    total = len(df)
    observed = {code: count/total for code, count in ancestry_counts.items()}
    
    # Get expected distribution
    reference = get_reference_ancestry('coarse')
    
    # Critical gaps (missing or severely underrepresented)
    critical_gaps = []
    for code, expected_prop in reference.items():
        observed_prop = observed.get(code, 0)
        if observed_prop == 0 and expected_prop > 0.05:
            critical_gaps.append((code, expected_prop))
        elif observed_prop < expected_prop * 0.3 and expected_prop > 0.05:
            critical_gaps.append((code, expected_prop))
    
    # Generate recommendations based on gaps
    if critical_gaps:
        missing_ancestries = [ANCESTRY_NAMES.get(code, code) for code, _ in critical_gaps[:3]]
        
        actions = [
            f"Partner with biobanks or research institutions in regions with {', '.join(missing_ancestries)} populations",
            "Implement targeted recruitment strategies in underrepresented communities",
            "Collaborate with community organizations to build trust and increase participation",
            "Consider oversampling strategies to balance representation",
            "Review consent processes to ensure cultural appropriateness for diverse populations"
        ]
        
        recommendations.append(Recommendation(
            dimension='ancestry',
            severity='critical' if len(critical_gaps) > 2 else 'high',
            title=f"Critical Underrepresentation: {len(critical_gaps)} ancestry groups missing or severely underrepresented",
            description=f"Your dataset has significant gaps in {', '.join(missing_ancestries)} representation. This limits generalizability of findings to these populations.",
            actions=actions,
            priority=1
        ))
    
    # Evenness issues
    if ancestry_details.get('evenness', 1) < 0.6:
        dominant_ancestry = ancestry_counts.index[0]
        dominant_pct = (ancestry_counts.iloc[0] / total) * 100
        
        recommendations.append(Recommendation(
            dimension='ancestry',
            severity='moderate',
            title=f"Unbalanced Distribution: {ANCESTRY_NAMES.get(dominant_ancestry, dominant_ancestry)} represents {dominant_pct:.0f}% of dataset",
            description="Highly skewed ancestry distribution can lead to biased models that perform poorly on minority groups.",
            actions=[
                "Implement stratified sampling to balance ancestry groups",
                "Weight models by inverse frequency to account for imbalance",
                "Report performance metrics separately for each ancestry group",
                "Consider ancestry-specific analyses alongside pooled analyses"
            ],
            priority=2
        ))
    
    # Low diversity score
    if ancestry_score < 60:
        recommendations.append(Recommendation(
            dimension='ancestry',
            severity='high',
            title="Low Ancestry Diversity Score",
            description="Overall ancestry diversity is below recommended thresholds for generalizable research.",
            actions=[
                "Expand recruitment to include at least 5 major ancestry groups",
                "Aim for minimum 10% representation in each major group",
                "Document ancestry-specific recruitment challenges and mitigation strategies",
                "Consider multi-site collaborations to increase diversity"
            ],
            priority=1
        ))
    
    return recommendations


def analyze_geographic_gaps(df: pd.DataFrame, heim_result: Dict) -> List[Recommendation]:
    """Analyze geographic distribution and generate recommendations."""
    recommendations = []
    
    geo_score = heim_result['dimensions']['geographic']['score']
    geo_details = heim_result['dimensions']['geographic']['details']
    unique_countries = geo_details.get('unique_countries', 0)
    
    # Limited geographic spread
    if unique_countries < 5:
        recommendations.append(Recommendation(
            dimension='geographic',
            severity='high',
            title=f"Limited Geographic Diversity: Only {unique_countries} countries represented",
            description="Narrow geographic spread limits applicability across different environmental, healthcare, and genetic contexts.",
            actions=[
                "Expand recruitment to multiple continents (aim for at least 3)",
                "Partner with international research consortia",
                "Include both high-income and low-to-middle income countries",
                "Consider environmental and healthcare system variations across regions"
            ],
            priority=2
        ))
    elif unique_countries < 10:
        recommendations.append(Recommendation(
            dimension='geographic',
            severity='moderate',
            title="Moderate Geographic Diversity",
            description="Good start, but expanding to more countries would improve generalizability.",
            actions=[
                "Target expansion to underrepresented continents",
                "Include countries with different healthcare systems",
                "Consider both urban and rural populations within countries"
            ],
            priority=3
        ))
    
    # Check for geographic concentration
    country_counts = df['country'].value_counts()
    if len(country_counts) > 0:
        top_country_pct = (country_counts.iloc[0] / len(df)) * 100
        
        if top_country_pct > 50:
            recommendations.append(Recommendation(
                dimension='geographic',
                severity='moderate',
                title=f"Geographic Concentration: {country_counts.index[0]} represents {top_country_pct:.0f}% of participants",
                description="Heavy concentration in one country may limit generalizability of findings.",
                actions=[
                    "Balance recruitment across multiple countries",
                    "Report country-specific results when feasible",
                    "Consider geographic stratification in analyses",
                    "Document country-specific factors that may affect outcomes"
                ],
                priority=3
            ))
    
    return recommendations


def analyze_age_gaps(df: pd.DataFrame, heim_result: Dict) -> List[Recommendation]:
    """Analyze age distribution and generate recommendations."""
    recommendations = []
    
    age_score = heim_result['dimensions']['age']['score']
    age_details = heim_result['dimensions']['age']['details']
    
    age_range = age_details.get('range', 0)
    age_mean = age_details.get('mean', 0)
    
    # Narrow age range
    if age_range < 30:
        recommendations.append(Recommendation(
            dimension='age',
            severity='high',
            title=f"Narrow Age Range: Only {age_range:.0f} years",
            description="Limited age range reduces applicability across lifespan and may miss age-related effects.",
            actions=[
                "Expand recruitment to include younger adults (18-30) if missing",
                "Include older adults (65+) to capture aging-related factors",
                "Ensure representation across at least 4 age decades",
                "Consider age-stratified analyses to identify age-specific effects"
            ],
            priority=2
        ))
    
    # Age skew
    if age_mean < 35:
        recommendations.append(Recommendation(
            dimension='age',
            severity='moderate',
            title="Young Population Bias",
            description="Dataset skews toward younger participants, limiting generalizability to older adults.",
            actions=[
                "Increase recruitment of participants aged 50+",
                "Partner with geriatric clinics or retirement communities",
                "Ensure findings are validated in older populations before clinical application"
            ],
            priority=3
        ))
    elif age_mean > 60:
        recommendations.append(Recommendation(
            dimension='age',
            severity='moderate',
            title="Older Population Bias",
            description="Dataset skews toward older participants, limiting generalizability to younger adults.",
            actions=[
                "Increase recruitment of participants aged 18-40",
                "Use digital recruitment methods to reach younger populations",
                "Ensure findings are validated in younger populations"
            ],
            priority=3
        ))
    
    return recommendations


def analyze_sex_gaps(df: pd.DataFrame, heim_result: Dict) -> List[Recommendation]:
    """Analyze sex/gender balance and generate recommendations."""
    recommendations = []
    
    sex_score = heim_result['dimensions']['sex']['score']
    sex_details = heim_result['dimensions']['sex']['details']
    
    # Normalize sex values
    sex_normalized = df['sex'].str.upper().replace({
        'MALE': 'M',
        'FEMALE': 'F'
    })
    
    sex_counts = sex_normalized.value_counts()
    
    if len(sex_counts) > 0:
        total = sex_counts.sum()
        
        # Severe imbalance
        if sex_counts.iloc[0] / total > 0.8:
            dominant_sex = 'female' if sex_counts.index[0] == 'F' else 'male'
            dominant_pct = (sex_counts.iloc[0] / total) * 100
            
            recommendations.append(Recommendation(
                dimension='sex',
                severity='high',
                title=f"Severe Sex Imbalance: {dominant_pct:.0f}% {dominant_sex}",
                description=f"Extreme {dominant_sex} bias limits generalizability to other sexes and may miss sex-specific effects.",
                actions=[
                    f"Implement targeted recruitment for underrepresented sex",
                    "Review recruitment materials for unintentional sex bias",
                    "Consider sex-specific recruitment channels",
                    "Report sex-stratified results to identify differential effects",
                    "Ensure informed consent process appeals to all sexes"
                ],
                priority=1
            ))
        
        # Moderate imbalance
        elif sex_counts.iloc[0] / total > 0.65:
            dominant_sex = 'female' if sex_counts.index[0] == 'F' else 'male'
            
            recommendations.append(Recommendation(
                dimension='sex',
                severity='moderate',
                title="Moderate Sex Imbalance",
                description="Noticeable sex skew may affect generalizability of findings.",
                actions=[
                    "Balance recruitment to achieve closer to 50/50 representation",
                    "Use sex-stratified sampling if feasible",
                    "Report sex-specific effects in analyses"
                ],
                priority=3
            ))
    
    return recommendations


def generate_priority_actions(recommendations: List[Recommendation]) -> List[Dict]:
    """Generate prioritized action list from all recommendations."""
    
    # Sort by priority
    sorted_recs = sorted(recommendations, key=lambda x: x.priority)
    
    # Get top 5 actions across all dimensions
    priority_actions = []
    
    for rec in sorted_recs[:5]:
        priority_actions.append({
            'dimension': rec.dimension,
            'severity': rec.severity,
            'title': rec.title,
            'top_action': rec.actions[0] if rec.actions else "Review and improve representation"
        })
    
    return priority_actions


def generate_recommendations(df: pd.DataFrame, heim_result: Dict) -> Dict:
    """
    Generate comprehensive recommendations based on HEIM analysis.
    
    Args:
        df: Dataset DataFrame
        heim_result: HEIM scoring results
        
    Returns:
        Dictionary containing recommendations by dimension and priority actions
    """
    all_recommendations = []
    
    # Analyze each dimension
    all_recommendations.extend(analyze_ancestry_gaps(df, heim_result))
    all_recommendations.extend(analyze_geographic_gaps(df, heim_result))
    all_recommendations.extend(analyze_age_gaps(df, heim_result))
    all_recommendations.extend(analyze_sex_gaps(df, heim_result))
    
    # Get priority actions
    priority_actions = generate_priority_actions(all_recommendations)
    
    # Organize by dimension
    by_dimension = {
        'ancestry': [r for r in all_recommendations if r.dimension == 'ancestry'],
        'geographic': [r for r in all_recommendations if r.dimension == 'geographic'],
        'age': [r for r in all_recommendations if r.dimension == 'age'],
        'sex': [r for r in all_recommendations if r.dimension == 'sex']
    }
    
    # Overall assessment
    critical_count = sum(1 for r in all_recommendations if r.severity == 'critical')
    high_count = sum(1 for r in all_recommendations if r.severity == 'high')
    
    if critical_count > 0:
        overall_status = 'critical'
        overall_message = f"⚠️ {critical_count} critical issues require immediate attention to ensure study validity."
    elif high_count > 2:
        overall_status = 'needs_improvement'
        overall_message = f"⚠️ {high_count} significant gaps detected. Addressing these will substantially improve dataset quality."
    elif len(all_recommendations) > 0:
        overall_status = 'good_with_gaps'
        overall_message = f"✓ Good diversity foundation with {len(all_recommendations)} areas for improvement."
    else:
        overall_status = 'excellent'
        overall_message = "✓ Excellent diversity! Dataset meets recommended standards across all dimensions."
    
    return {
        'all_recommendations': [r.to_dict() for r in all_recommendations],
        'by_dimension': {k: [r.to_dict() for r in v] for k, v in by_dimension.items()},
        'priority_actions': priority_actions,
        'overall_status': overall_status,
        'overall_message': overall_message,
        'total_issues': len(all_recommendations),
        'critical_issues': critical_count,
        'high_priority_issues': high_count
    }


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        'critical': '#FF4444',
        'high': '#FF8C00',
        'moderate': '#FFA500',
        'low': '#FFD700'
    }
    return colors.get(severity, '#808080')


def get_severity_icon(severity: str) -> str:
    """Get icon for severity level."""
    icons = {
        'critical': '🚨',
        'high': '⚠️',
        'moderate': '⚡',
        'low': '💡'
    }
    return icons.get(severity, '•')