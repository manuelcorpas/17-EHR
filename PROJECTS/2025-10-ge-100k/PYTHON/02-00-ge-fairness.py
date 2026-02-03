#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABSTRACT-BASED BIOBANK BIAS DETECTION PIPELINE V3.0
====================================================

Detects bias indicators from abstracts in biobank research:
- Identifies population biases, sampling issues, and limitations
- Extracts demographic and methodological features from abstracts
- Compares bias profiles across biobanks
- Identifies health equity gaps
- Provides actionable insights for improving research diversity

Major features:
1. Abstract-appropriate pattern detection
2. Feature extraction (sample size, demographics, etc.)
3. Comparative bias profiling across biobanks
4. Equity gap analysis
5. Risk scoring based on detectable indicators

USAGE:
    python projects/ge-100k/PYTHON/02-00-ge-fairness-abstract.py

REQUIREMENTS:
    pip install pandas numpy matplotlib seaborn scikit-learn nltk
"""

import argparse
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

warnings.filterwarnings('ignore')

# Setup paths using Pathlib
SCRIPT = Path(__file__).resolve()
GE_ROOT = SCRIPT.parents[1]  # projects/ge-100k
DATA_DIR = GE_ROOT / "DATA"
ANALYSIS_ROOT = GE_ROOT / "ANALYSIS" / "02-00-GE-FAIRNESS"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")

# Parse arguments
ap = argparse.ArgumentParser(description="Abstract-Based Biobank Bias Detection Pipeline")
ap.add_argument("--config", default=str(GE_ROOT / "CONFIG" / "ge.default.yml"))
ap.add_argument("--outdir", default=None, help="Optional output directory; if unset, uses dated run folder.")
args = ap.parse_args()

# Set output directory
if args.outdir:
    OUT_DIR = Path(args.outdir)
else:
    OUT_DIR = ANALYSIS_ROOT / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Output directory: {OUT_DIR}")

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

#############################################################################
# 1. ABSTRACT-BASED BIAS INDICATOR DETECTION
#############################################################################

class AbstractBiasDetector:
    """Detect bias indicators from abstracts rather than mitigation methods."""
    
    def __init__(self):
        self.setup_indicators()
    
    def setup_indicators(self):
        """Define patterns for bias indicators detectable in abstracts."""
        
        self.bias_indicators = {
            'population_bias': {
                'homogeneous': [
                    r'\b(?:UK|British|European|Caucasian|white)\s+(?:only|cohort|population|participants)\b',
                    r'\b(?:mostly|predominantly|primarily)\s+(?:male|female|men|women)\b',
                    r'\b(?:single|one)\s+(?:center|centre|hospital|clinic|site)\b',
                    r'\bhomogeneous\s+(?:population|cohort|sample)\b',
                    r'\b(?:restricted|limited)\s+to\s+(?:UK|European|white)\b'
                ],
                'age_restricted': [
                    r'\b(?:elderly|older\s+adults?|aged\s+\d+[-–]\d+)\s+(?:only|exclusively)\b',
                    r'\bmiddle[-\s]aged\b.*\b(?:only|exclusively)\b',
                    r'\bexclud\w+\s+(?:younger|older|elderly|children)\b',
                    r'\bage\s+(?:range|group)[:]\s*\d+[-–]\d+\b'
                ],
                'selection_bias': [
                    r'\bhealthy\s+volunteers?\b',
                    r'\bself[-\s]selected\b',
                    r'\bconvenience\s+sample\b',
                    r'\bvolunteer\s+bias\b',
                    r'\bparticipants?\s+who\s+(?:agreed|consented|volunteered)\b'
                ]
            },
            
            'sample_size_issues': {
                'very_small': [
                    r'\bn\s*[=:]\s*[1-9]\d?\b',  # n < 100
                    r'\b(?:sample\s+size|n)\s*[=:]\s*[1-9]\d?\b',
                    r'\bvery\s+small\s+sample\b'
                ],
                'small': [
                    r'\bn\s*[=:]\s*[1-9]\d{2}\b',  # n = 100-999
                    r'\bsmall\s+sample\s+size\b',
                    r'\blimited\s+sample\b',
                    r'\bpilot\s+study\b',
                    r'\bexploratory\s+(?:study|analysis)\b',
                    r'\bpreliminary\s+(?:study|findings)\b'
                ],
                'moderate': [
                    r'\bn\s*[=:]\s*[1-9]\d{3,4}\b',  # n = 1,000-99,999
                    r'\bthousands?\s+of\s+participants?\b'
                ],
                'large': [
                    r'\bn\s*[=:]\s*[1-9]\d{5,}\b',  # n >= 100,000
                    r'\b\d+\s*(?:hundred\s+)?thousands?\s+(?:of\s+)?participants?\b',
                    r'\blarge[-\s]scale\b',
                    r'\bpopulation[-\s]based\b',
                    r'\b\d+[,.]?\d*\s*million\s+participants?\b'
                ]
            },
            
            'temporal_limitations': {
                'cross_sectional': [
                    r'\bcross[-\s]sectional\s+(?:study|design|analysis)\b',
                    r'\bsingle\s+time[-\s]point\b',
                    r'\bbaseline\s+(?:data\s+)?only\b',
                    r'\bno\s+(?:longitudinal|follow[-\s]up)\b'
                ],
                'short_followup': [
                    r'\b\d+[-\s](?:month|week)s?\s+(?:follow[-\s]up|duration)\b',
                    r'\bshort[-\s]term\s+(?:follow[-\s]up|study)\b',
                    r'\bmedian\s+follow[-\s]up.*\d+\s+(?:months?|weeks?)\b'
                ],
                'retrospective': [
                    r'\bretrospective\s+(?:study|analysis|cohort)\b',
                    r'\bhistorical\s+(?:data|cohort)\b',
                    r'\blookback\s+period\b'
                ]
            },
            
            'geographic_limitations': {
                'single_country': [
                    r'\b(?:UK|US|United\s+States|Finland|Estonia)\s+(?:biobank|cohort)\b',
                    r'\bnational\s+(?:cohort|study|sample)\b',
                    r'\bsingle\s+country\b',
                    r'\bcountry[-\s]specific\b'
                ],
                'urban_bias': [
                    r'\burban\s+(?:population|cohort|participants|areas?)\b',
                    r'\bcity[-\s]based\b',
                    r'\bmetropolitan\s+areas?\b',
                    r'\bexclud\w+\s+rural\b'
                ],
                'regional': [
                    r'\bregional\s+(?:cohort|sample|study)\b',
                    r'\b(?:single|one)\s+(?:state|province|region)\b',
                    r'\blocal\s+(?:population|community)\b'
                ]
            },
            
            'acknowledged_limitations': {
                'generalizability': [
                    r'\blimited\s+generalizability\b',
                    r'\bmay\s+not\s+(?:generalize|be\s+generalizable)\b',
                    r'\bcaution.*generaliz\w+\b',
                    r'\bnot\s+representative\b',
                    r'\blimitations?\s+include\b',
                    r'\bresults\s+may\s+not\s+apply\b'
                ],
                'confounding': [
                    r'\bunmeasured\s+confound\w+\b',
                    r'\bresidual\s+confound\w+\b',
                    r'\bcannot\s+(?:rule\s+out|exclude)\b',
                    r'\bcausal.*cannot\s+be\s+(?:established|determined)\b',
                    r'\bpotential\s+confound\w+\b'
                ],
                'missing_data': [
                    r'\bmissing\s+(?:data|information)\b',
                    r'\bincomplete\s+(?:data|information|records)\b',
                    r'\bdata\s+(?:not\s+)?available\b',
                    r'\blost\s+to\s+follow[-\s]up\b'
                ]
            },
            
            'diversity_indicators': {
                'mentions_diversity': [
                    r'\bdivers\w+\s+(?:population|cohort|sample)\b',
                    r'\bmulti[-\s]?ethnic\b',
                    r'\brepresentative\s+(?:of|sample)\b',
                    r'\binclusive\s+(?:recruitment|sample)\b',
                    r'\bminority\s+(?:populations?|groups?)\b'
                ],
                'mentions_equity': [
                    r'\bhealth\s+(?:equity|disparities|inequalities)\b',
                    r'\bunderserved\s+(?:populations?|communities)\b',
                    r'\bsocioeconomic\s+(?:diversity|factors)\b',
                    r'\baccess\s+to\s+(?:care|healthcare)\b'
                ]
            }
        }
        
        # Disease focus patterns
        self.disease_patterns = {
            'common_diseases': [
                r'\b(?:diabetes|hypertension|cancer|heart\s+disease|stroke)\b',
                r'\b(?:obesity|depression|anxiety|asthma|COPD)\b',
                r'\b(?:cardiovascular|coronary|metabolic)\s+disease\b'
            ],
            'rare_diseases': [
                r'\brare\s+(?:disease|disorder|condition)\b',
                r'\borphan\s+disease\b',
                r'\b(?:genetic|hereditary)\s+(?:disorder|syndrome)\b',
                r'\bprevalence.*(?:<|less\s+than)\s*(?:1|0\.\d+)%\b'
            ],
            'infectious_diseases': [
                r'\b(?:COVID|SARS|influenza|tuberculosis|HIV|malaria)\b',
                r'\b(?:infectious|communicable)\s+disease\b',
                r'\b(?:viral|bacterial|parasitic)\s+infection\b'
            ]
        }

    def extract_sample_size(self, text: str) -> Optional[int]:
        """Extract sample size from abstract text."""
        
        patterns = [
            r'\bn\s*[=:]\s*([0-9,]+)\b',
            r'\b([0-9,]+)\s+(?:participants?|subjects?|patients?|individuals?)\b',
            r'\bsample\s+size\s*(?:of|[=:])\s*([0-9,]+)\b',
            r'\btotal\s+of\s+([0-9,]+)\s+(?:participants?|subjects?)\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    # Remove commas and convert to int
                    size_str = match.group(1).replace(',', '')
                    return int(size_str)
                except:
                    continue
        
        return None
    
    def extract_demographics(self, text: str) -> Dict:
        """Extract demographic information from abstract."""
        
        demographics = {
            'mean_age': None,
            'age_range': None,
            'percent_female': None,
            'percent_male': None
        }
        
        # Extract mean age
        age_patterns = [
            r'mean\s+age\s*(?:of|[=:])\s*(\d+(?:\.\d+)?)\s*(?:years?)?',
            r'average\s+age\s*(?:of|[=:])\s*(\d+(?:\.\d+)?)\s*(?:years?)?',
            r'aged\s+(\d+(?:\.\d+)?)\s*±\s*\d+(?:\.\d+)?\s*years?'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    demographics['mean_age'] = float(match.group(1))
                    break
                except:
                    continue
        
        # Extract age range
        range_pattern = r'age(?:d|s?)?\s+(\d+)\s*[-–to]\s*(\d+)\s*(?:years?)?'
        match = re.search(range_pattern, text.lower())
        if match:
            try:
                demographics['age_range'] = (int(match.group(1)), int(match.group(2)))
            except:
                pass
        
        # Extract sex distribution
        female_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?(?:female|women)',
            r'(?:female|women)\s*[:]\s*(\d+(?:\.\d+)?)\s*%'
        ]
        
        for pattern in female_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    demographics['percent_female'] = float(match.group(1))
                    demographics['percent_male'] = 100 - demographics['percent_female']
                    break
                except:
                    continue
        
        return demographics
    
    def extract_countries(self, text: str) -> List[str]:
        """Extract country mentions from abstract."""
        
        countries = []
        country_patterns = [
            r'\b(?:United\s+Kingdom|UK|Britain|England|Scotland|Wales)\b',
            r'\b(?:United\s+States|US|USA|America)\b',
            r'\b(?:Finland|Finnish)\b',
            r'\b(?:Estonia|Estonian)\b',
            r'\b(?:Sweden|Swedish|Norway|Norwegian|Denmark|Danish)\b',
            r'\b(?:Germany|German|France|French|Italy|Italian|Spain|Spanish)\b',
            r'\b(?:China|Chinese|Japan|Japanese|Korea|Korean)\b',
            r'\b(?:India|Indian|Brazil|Brazilian|Mexico|Mexican)\b',
            r'\b(?:Canada|Canadian|Australia|Australian)\b'
        ]
        
        for pattern in country_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Extract base country name
                country = re.search(pattern, text, re.IGNORECASE).group(0)
                countries.append(country)
        
        return list(set(countries))
    
    def analyze_abstract(self, abstract: str) -> Dict:
        """Comprehensive analysis of a single abstract for bias indicators."""
        
        if not abstract or pd.isna(abstract):
            return self._empty_results()
        
        abstract_lower = abstract.lower()
        
        results = {
            'indicators': defaultdict(list),
            'features': {},
            'scores': {},
            'risk_factors': []
        }
        
        # Check all bias indicators
        for bias_category, indicator_groups in self.bias_indicators.items():
            for indicator_type, patterns in indicator_groups.items():
                for pattern in patterns:
                    if re.search(pattern, abstract_lower, re.IGNORECASE):
                        results['indicators'][bias_category].append(indicator_type)
                        break
        
        # Extract quantitative features
        results['features']['sample_size'] = self.extract_sample_size(abstract)
        results['features'].update(self.extract_demographics(abstract))
        results['features']['countries'] = self.extract_countries(abstract)
        results['features']['num_countries'] = len(results['features']['countries'])
        
        # Check study design
        results['features']['is_gwas'] = bool(re.search(r'genome[-\s]wide', abstract_lower))
        results['features']['is_longitudinal'] = bool(re.search(r'longitudinal|follow[-\s]up|prospective|cohort', abstract_lower))
        results['features']['is_cross_sectional'] = bool(re.search(r'cross[-\s]sectional', abstract_lower))
        results['features']['is_retrospective'] = bool(re.search(r'retrospective|historical', abstract_lower))
        
        # Check diversity mentions
        results['features']['mentions_diversity'] = any(
            re.search(pattern, abstract_lower) 
            for pattern in self.bias_indicators['diversity_indicators']['mentions_diversity']
        )
        results['features']['mentions_limitations'] = any(
            re.search(pattern, abstract_lower) 
            for pattern in self.bias_indicators['acknowledged_limitations']['generalizability']
        )
        
        # Check disease focus
        for disease_type, patterns in self.disease_patterns.items():
            results['features'][f'has_{disease_type}'] = any(
                re.search(pattern, abstract_lower) for pattern in patterns
            )
        
        # Calculate risk scores
        results['scores'] = self.calculate_risk_scores(results)
        
        return results
    
    def calculate_risk_scores(self, analysis_results: Dict) -> Dict:
        """Calculate various bias risk scores based on detected indicators."""
        
        scores = {}
        features = analysis_results['features']
        indicators = analysis_results['indicators']
        
        # Population homogeneity risk
        homogeneity_factors = [
            'homogeneous' in indicators.get('population_bias', []),
            'age_restricted' in indicators.get('population_bias', []),
            features.get('num_countries', 0) <= 1,
            not features.get('mentions_diversity', False)
        ]
        scores['homogeneity_risk'] = sum(homogeneity_factors) / len(homogeneity_factors)
        
        # Sample size adequacy - FIXED: Handle None values
        sample_size = features.get('sample_size')
        if sample_size is not None and sample_size > 0:
            if sample_size < 100:
                scores['sample_size_score'] = 0.0
            elif sample_size < 500:
                scores['sample_size_score'] = 0.25
            elif sample_size < 5000:
                scores['sample_size_score'] = 0.5
            elif sample_size < 50000:
                scores['sample_size_score'] = 0.75
            else:
                scores['sample_size_score'] = 1.0
        else:
            scores['sample_size_score'] = 0.5  # Unknown
        
        # Temporal robustness
        temporal_factors = [
            features.get('is_longitudinal', False),
            not features.get('is_cross_sectional', False),
            'short_followup' not in indicators.get('temporal_limitations', [])
        ]
        scores['temporal_robustness'] = sum(temporal_factors) / len(temporal_factors)
        
        # Geographic diversity
        geographic_factors = [
            features.get('num_countries', 0) > 1,
            'single_country' not in indicators.get('geographic_limitations', []),
            'urban_bias' not in indicators.get('geographic_limitations', [])
        ]
        scores['geographic_diversity'] = sum(geographic_factors) / len(geographic_factors)
        
        # Transparency score
        transparency_factors = [
            features.get('mentions_limitations', False),
            'generalizability' in indicators.get('acknowledged_limitations', []),
            'confounding' in indicators.get('acknowledged_limitations', [])
        ]
        scores['transparency'] = sum(transparency_factors) / len(transparency_factors)
        
        # Overall bias risk (higher = more risk)
        risk_components = [
            1 - scores['sample_size_score'],
            scores['homogeneity_risk'],
            1 - scores['temporal_robustness'],
            1 - scores['geographic_diversity'],
            1 - scores['transparency']
        ]
        scores['overall_bias_risk'] = np.mean(risk_components)
        
        # Equity consideration score
        equity_factors = [
            features.get('mentions_diversity', False),
            any(key.startswith('has_') and features.get(key, False) 
                for key in ['has_rare_diseases', 'has_infectious_diseases']),
            scores['geographic_diversity'] > 0.5
        ]
        scores['equity_consideration'] = sum(equity_factors) / len(equity_factors)
        
        return scores
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'indicators': {},
            'features': {
                'sample_size': None,
                'mean_age': None,
                'age_range': None,
                'percent_female': None,
                'percent_male': None,
                'countries': [],
                'num_countries': 0,
                'is_gwas': False,
                'is_longitudinal': False,
                'is_cross_sectional': False,
                'mentions_diversity': False,
                'mentions_limitations': False
            },
            'scores': {
                'homogeneity_risk': 0.5,
                'sample_size_score': 0.5,
                'temporal_robustness': 0.5,
                'geographic_diversity': 0.5,
                'transparency': 0.5,
                'overall_bias_risk': 0.5,
                'equity_consideration': 0.0
            },
            'risk_factors': []
        }

#############################################################################
# 2. BIOBANK COMPARISON AND PROFILING
#############################################################################

def analyze_biobank_profiles(df: pd.DataFrame, detector: AbstractBiasDetector) -> Dict:
    """Create comparative bias profiles for each biobank."""
    
    biobank_profiles = {}
    
    for biobank in df['Biobank'].unique():
        biobank_data = df[df['Biobank'] == biobank]
        
        profile = {
            'total_papers': len(biobank_data),
            'temporal_range': (int(biobank_data['Year'].min()), int(biobank_data['Year'].max())),
            'sample_sizes': [],
            'countries_mentioned': set(),
            'study_designs': Counter(),
            'bias_indicators': Counter(),
            'risk_scores': [],
            'equity_scores': [],
            'transparency_scores': []
        }
        
        # Analyze each paper
        for _, paper in biobank_data.iterrows():
            abstract = str(paper.get('Abstract', ''))
            if pd.notna(abstract):
                analysis = detector.analyze_abstract(abstract)
                
                # Collect sample sizes
                if analysis['features']['sample_size']:
                    profile['sample_sizes'].append(analysis['features']['sample_size'])
                
                # Collect countries
                profile['countries_mentioned'].update(analysis['features']['countries'])
                
                # Track study designs
                if analysis['features']['is_longitudinal']:
                    profile['study_designs']['longitudinal'] += 1
                if analysis['features']['is_cross_sectional']:
                    profile['study_designs']['cross_sectional'] += 1
                if analysis['features']['is_gwas']:
                    profile['study_designs']['gwas'] += 1
                
                # Track bias indicators
                for category, indicators in analysis['indicators'].items():
                    for indicator in indicators:
                        profile['bias_indicators'][f'{category}:{indicator}'] += 1
                
                # Collect scores
                profile['risk_scores'].append(analysis['scores']['overall_bias_risk'])
                profile['equity_scores'].append(analysis['scores']['equity_consideration'])
                profile['transparency_scores'].append(analysis['scores']['transparency'])
        
        # Calculate summary statistics
        profile['median_sample_size'] = np.median(profile['sample_sizes']) if profile['sample_sizes'] else None
        profile['geographic_diversity'] = len(profile['countries_mentioned'])
        profile['mean_risk_score'] = np.mean(profile['risk_scores']) if profile['risk_scores'] else 0.5
        profile['mean_equity_score'] = np.mean(profile['equity_scores']) if profile['equity_scores'] else 0.0
        profile['mean_transparency_score'] = np.mean(profile['transparency_scores']) if profile['transparency_scores'] else 0.5
        
        # Calculate proportions
        total = len(biobank_data)
        profile['prop_longitudinal'] = profile['study_designs']['longitudinal'] / total if total > 0 else 0
        profile['prop_cross_sectional'] = profile['study_designs']['cross_sectional'] / total if total > 0 else 0
        profile['prop_gwas'] = profile['study_designs']['gwas'] / total if total > 0 else 0
        
        biobank_profiles[biobank] = profile
    
    return biobank_profiles

#############################################################################
# 3. ENHANCED VISUALIZATION
#############################################################################

def create_bias_landscape_dashboard(results_df: pd.DataFrame, biobank_profiles: Dict):
    """Create comprehensive dashboard showing bias landscape."""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Spider plot comparing biobanks
    ax1 = fig.add_subplot(gs[0:2, 0], projection='polar')
    
    categories = ['Sample Size', 'Geographic\nDiversity', 'Temporal\nRobustness', 
                  'Transparency', 'Equity\nFocus']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(biobank_profiles)))
    
    for idx, (biobank, profile) in enumerate(biobank_profiles.items()):
        values = [
            np.log10(profile['median_sample_size'] + 1) / 6 if profile['median_sample_size'] else 0.3,
            min(profile['geographic_diversity'] / 10, 1),
            profile['prop_longitudinal'],
            profile['mean_transparency_score'],
            profile['mean_equity_score']
        ]
        values += values[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=biobank, color=colors[idx])
        ax1.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax1.set_ylim(0, 1)
    ax1.set_title('A. Biobank Bias Profile Comparison', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 2. Sample size distribution by biobank
    ax2 = fig.add_subplot(gs[0, 1:])
    
    sample_data = []
    for biobank, profile in biobank_profiles.items():
        for size in profile['sample_sizes']:
            sample_data.append({'Biobank': biobank, 'Sample Size': size})
    
    if sample_data:
        sample_df = pd.DataFrame(sample_data)
        sample_df['Log Sample Size'] = np.log10(sample_df['Sample Size'] + 1)
        
        biobanks = sample_df['Biobank'].unique()
        positions = range(len(biobanks))
        
        for pos, biobank in enumerate(biobanks):
            biobank_samples = sample_df[sample_df['Biobank'] == biobank]['Log Sample Size']
            if len(biobank_samples) > 0:
                bp = ax2.boxplot([biobank_samples], positions=[pos], widths=0.6,
                                 patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(colors[pos])
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels(biobanks, rotation=45, ha='right')
        ax2.set_ylabel('Log10(Sample Size)', fontweight='bold')
        ax2.set_title('B. Sample Size Distribution by Biobank', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Geographic diversity comparison
    ax3 = fig.add_subplot(gs[1, 1])
    
    geo_data = [(name, profile['geographic_diversity']) 
                for name, profile in biobank_profiles.items()]
    geo_data.sort(key=lambda x: x[1], reverse=True)
    
    names, diversities = zip(*geo_data)
    bars = ax3.barh(range(len(names)), diversities, color='steelblue')
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names)
    ax3.set_xlabel('Number of Countries Mentioned', fontweight='bold')
    ax3.set_title('C. Geographic Diversity by Biobank', fontweight='bold')
    
    for i, (bar, div) in enumerate(zip(bars, diversities)):
        ax3.text(div + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{div}', va='center')
    
    # 4. Study design proportions
    ax4 = fig.add_subplot(gs[1, 2])
    
    design_data = []
    for biobank, profile in biobank_profiles.items():
        design_data.append({
            'Biobank': biobank,
            'Longitudinal': profile['prop_longitudinal'],
            'Cross-sectional': profile['prop_cross_sectional'],
            'GWAS': profile['prop_gwas']
        })
    
    design_df = pd.DataFrame(design_data)
    design_df.set_index('Biobank')[['Longitudinal', 'Cross-sectional', 'GWAS']].plot(
        kind='bar', stacked=False, ax=ax4, color=['#2ecc71', '#e74c3c', '#3498db']
    )
    ax4.set_ylabel('Proportion of Studies', fontweight='bold')
    ax4.set_title('D. Study Design Distribution', fontweight='bold')
    ax4.legend(title='Design Type')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Risk score distribution
    ax5 = fig.add_subplot(gs[2, :2])
    
    risk_data = []
    for biobank, profile in biobank_profiles.items():
        for score in profile['risk_scores']:
            risk_data.append({'Biobank': biobank, 'Risk Score': score})
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        
        for idx, biobank in enumerate(risk_df['Biobank'].unique()):
            biobank_risks = risk_df[risk_df['Biobank'] == biobank]['Risk Score']
            ax5.hist(biobank_risks, bins=20, alpha=0.5, label=biobank, color=colors[idx])
        
        ax5.set_xlabel('Overall Bias Risk Score', fontweight='bold')
        ax5.set_ylabel('Number of Papers', fontweight='bold')
        ax5.set_title('E. Distribution of Bias Risk Scores', fontweight='bold')
        ax5.legend()
        ax5.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Neutral')
    
    # 6. Equity vs Risk scatter
    ax6 = fig.add_subplot(gs[2, 2])
    
    for idx, (biobank, profile) in enumerate(biobank_profiles.items()):
        if profile['risk_scores'] and profile['equity_scores']:
            ax6.scatter(profile['risk_scores'], profile['equity_scores'], 
                       alpha=0.5, label=biobank, color=colors[idx], s=30)
    
    ax6.set_xlabel('Bias Risk Score', fontweight='bold')
    ax6.set_ylabel('Equity Consideration Score', fontweight='bold')
    ax6.set_title('F. Risk vs Equity Trade-off', fontweight='bold')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # 7. Temporal trends in bias risk
    ax7 = fig.add_subplot(gs[3, :])
    
    yearly_risk = results_df.groupby(['Year', 'Biobank'])['overall_bias_risk'].mean().reset_index()
    
    for idx, biobank in enumerate(yearly_risk['Biobank'].unique()):
        biobank_yearly = yearly_risk[yearly_risk['Biobank'] == biobank]
        biobank_yearly = biobank_yearly[biobank_yearly['Year'] >= 2010]  # Focus on recent years
        if len(biobank_yearly) > 0:
            ax7.plot(biobank_yearly['Year'], biobank_yearly['overall_bias_risk'], 
                    marker='o', label=biobank, color=colors[idx], linewidth=2)
    
    ax7.set_xlabel('Year', fontweight='bold')
    ax7.set_ylabel('Mean Bias Risk Score', fontweight='bold')
    ax7.set_title('G. Temporal Trends in Bias Risk', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('BIOBANK BIAS LANDSCAPE: Abstract-Based Assessment', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    fig_path = OUT_DIR / "bias_overview_comprehensive.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.info(f"Saved bias landscape dashboard: {fig_path}")
    
    return fig

#############################################################################
# 4. MAIN ANALYSIS PIPELINE
#############################################################################

def analyze_papers_abstract_based(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze papers using abstract-based bias detection."""
    
    logger.info("Starting abstract-based bias analysis...")
    detector = AbstractBiasDetector()

    df = df.reset_index(drop=True)
    
    results = []
    total_papers = len(df)
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"Processing paper {idx}/{total_papers} ({idx/total_papers*100:.1f}%)")
        
        abstract = str(row.get('Abstract', ''))
        title = str(row.get('Title', ''))
        
        # Analyze abstract
        analysis = detector.analyze_abstract(abstract)
        
        # Create flat result structure
        paper_results = {
            'PMID': row['PMID'],
            'Biobank': row['Biobank'],
            'Year': row['Year'],
            'Title': title[:100] if pd.notna(title) else ''
        }
        
        # Add features
        for feature_name, feature_value in analysis['features'].items():
            if isinstance(feature_value, list):
                paper_results[feature_name] = '|'.join(map(str, feature_value))
            else:
                paper_results[feature_name] = feature_value
        
        # Add scores
        for score_name, score_value in analysis['scores'].items():
            paper_results[score_name] = score_value
        
        # Add indicator counts
        for category in ['population_bias', 'sample_size_issues', 'temporal_limitations', 
                        'geographic_limitations', 'acknowledged_limitations']:
            indicators = analysis['indicators'].get(category, [])
            paper_results[f'{category}_count'] = len(indicators)
            paper_results[f'{category}_types'] = '|'.join(indicators)
        
        # Overall quality score (inverse of bias risk)
        paper_results['overall_quality'] = 1 - analysis['scores']['overall_bias_risk']
        
        # Count total bias indicators
        total_indicators = sum(len(v) for v in analysis['indicators'].values())
        paper_results['total_bias_indicators'] = total_indicators
        
        # Flag papers with critical issues - FIXED: Handle None sample_size
        sample_size = analysis['features'].get('sample_size')
        paper_results['has_critical_issues'] = (
            analysis['scores']['overall_bias_risk'] > 0.7 or
            (sample_size is not None and sample_size < 100) or
            total_indicators > 5
        )
        
        results.append(paper_results)
    
    return pd.DataFrame(results)
#############################################################################
# 5. REPORT GENERATION
#############################################################################
def generate_bias_report(results_df: pd.DataFrame, biobank_profiles: Dict) -> Dict:
    """Generate comprehensive bias assessment report."""
    
    report = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_papers': len(results_df),
            'analysis_version': '3.0 (Abstract-Based)'
        },
        'overall_summary': {},
        'biobank_comparisons': {},
        'bias_indicators': {},
        'equity_gaps': {},
        'recommendations': []
    }
    
    # Overall summary
    report['overall_summary'] = {
        'mean_bias_risk': float(results_df['overall_bias_risk'].mean()),
        'median_sample_size': float(results_df['sample_size'].median()) if results_df['sample_size'].notna().any() else None,
        'papers_with_critical_issues': int(results_df['has_critical_issues'].sum()),
        'papers_mentioning_diversity': int(results_df['mentions_diversity'].sum()),
        'papers_acknowledging_limitations': int(results_df['mentions_limitations'].sum()),
        'proportion_longitudinal': float(results_df['is_longitudinal'].mean()),
        'proportion_cross_sectional': float(results_df['is_cross_sectional'].mean())
    }
    
    # Biobank comparisons
    for biobank, profile in biobank_profiles.items():
        report['biobank_comparisons'][biobank] = {
            'total_papers': profile['total_papers'],
            'median_sample_size': profile['median_sample_size'],
            'geographic_diversity': profile['geographic_diversity'],
            'mean_risk_score': profile['mean_risk_score'],
            'mean_equity_score': profile['mean_equity_score'],
            'prop_longitudinal': profile['prop_longitudinal']
        }
    
    # Top bias indicators
    indicator_counts = Counter()
    for col in results_df.columns:
        if col.endswith('_types'):
            types = results_df[col].str.split('|', expand=True).stack()
            types = types[types != '']
            indicator_counts.update(types)
    
    report['bias_indicators']['most_common'] = dict(indicator_counts.most_common(10))
    
    # Equity gaps
    report['equity_gaps'] = {
        'low_diversity_papers': int((results_df['equity_consideration'] < 0.3).sum()),
        'single_country_studies': int((results_df['num_countries'] <= 1).sum()),
        'very_small_samples': int((results_df['sample_size'] < 100).sum()),
        'high_risk_papers': int((results_df['overall_bias_risk'] > 0.7).sum())
    }
    
    # Generate recommendations
    if report['overall_summary']['mean_bias_risk'] > 0.6:
        report['recommendations'].append(
            "High overall bias risk detected. Prioritize diverse recruitment and multi-site studies."
        )
    
    if report['overall_summary']['proportion_longitudinal'] < 0.3:
        report['recommendations'].append(
            "Increase longitudinal studies to better capture temporal dynamics and causal relationships."
        )
    
    diversity_papers = results_df['mentions_diversity'].sum()
    if diversity_papers < len(results_df) * 0.2:
        report['recommendations'].append(
            "Only {:.1f}% of papers mention diversity. Implement diversity-focused recruitment strategies.".format(
                diversity_papers / len(results_df) * 100
            )
        )
    
    # Save report
    report_file = OUT_DIR / "bias_detection_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Saved bias report: {report_file}")
    
    # Generate text summary - FIXED f-string syntax
    median_sample = report['overall_summary']['median_sample_size']
    median_sample_str = f"{median_sample:,.0f}" if median_sample else "N/A"
    
    summary_text = f"""
ABSTRACT-BASED BIOBANK BIAS DETECTION REPORT
============================================
Generated: {report['metadata']['analysis_date']}

DATASET OVERVIEW:
- Total papers analyzed: {report['metadata']['total_papers']:,}
- Analysis version: {report['metadata']['analysis_version']}

BIAS RISK SUMMARY:
- Mean bias risk score: {report['overall_summary']['mean_bias_risk']:.2f}
- Papers with critical issues: {report['overall_summary']['papers_with_critical_issues']:,}
- Papers mentioning diversity: {report['overall_summary']['papers_mentioning_diversity']:,}
- Papers acknowledging limitations: {report['overall_summary']['papers_acknowledging_limitations']:,}

STUDY DESIGN PATTERNS:
- Longitudinal studies: {report['overall_summary']['proportion_longitudinal']:.1%}
- Cross-sectional studies: {report['overall_summary']['proportion_cross_sectional']:.1%}
- Median sample size: {median_sample_str}

TOP BIAS INDICATORS:
"""
    
    for indicator, count in list(report['bias_indicators']['most_common'].items())[:5]:
        summary_text += f"- {indicator}: {count} occurrences\n"
    
    summary_text += "\nBIOBANK COMPARISON:\n"
    for biobank, stats in report['biobank_comparisons'].items():
        biobank_median = stats['median_sample_size']
        biobank_median_str = f"{biobank_median:,.0f}" if biobank_median else "N/A"
        
        summary_text += f"\n{biobank}:\n"
        summary_text += f"  - Papers: {stats['total_papers']:,}\n"
        summary_text += f"  - Median sample size: {biobank_median_str}\n"
        summary_text += f"  - Geographic diversity: {stats['geographic_diversity']} countries\n"
        summary_text += f"  - Mean bias risk: {stats['mean_risk_score']:.2f}\n"
        summary_text += f"  - Equity score: {stats['mean_equity_score']:.2f}\n"
    
    summary_text += "\nEQUITY GAPS:\n"
    summary_text += f"- Low diversity papers: {report['equity_gaps']['low_diversity_papers']:,}\n"
    summary_text += f"- Single country studies: {report['equity_gaps']['single_country_studies']:,}\n"
    summary_text += f"- Very small samples (<100): {report['equity_gaps']['very_small_samples']:,}\n"
    summary_text += f"- High risk papers: {report['equity_gaps']['high_risk_papers']:,}\n"
    
    summary_text += "\nRECOMMENDATIONS:\n"
    for rec in report['recommendations']:
        summary_text += f"- {rec}\n"
    
    # Save text summary
    summary_file = OUT_DIR / "bias_detection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Saved text summary: {summary_file}")
    
    print(summary_text)
    
    return report
#############################################################################
# 6. MAIN EXECUTION
#############################################################################

def main():
    """Main execution pipeline."""
    
    print("=" * 80)
    print("ABSTRACT-BASED BIOBANK BIAS DETECTION PIPELINE V3.0")
    print("Identifying Bias Indicators from Publication Abstracts")
    print("=" * 80)
    
    try:
        # Load data
        possible_files = [
            DATA_DIR / "biobank_research_data.csv",
            DATA_DIR / "ge_pubmed_qc.csv"
        ]
        
        input_file = None
        for file_path in possible_files:
            if file_path.exists():
                input_file = file_path
                break
        
        if not input_file:
            print(f"❌ Input file not found in {DATA_DIR}")
            print(f"   Looked for: {[str(f) for f in possible_files]}")
            print("Please ensure data file exists.")
            return None, None
        
        print(f"\n📊 Loading data from {input_file}...")
        df = pd.read_csv(input_file, low_memory=False)
        
        # Clean and filter
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
        
        # Remove papers without abstracts
        initial_count = len(df)
        df = df[df['Abstract'].notna()]
        df = df[df['Abstract'].str.len() > 50]  # Ensure meaningful abstracts
        
        print(f"Loaded {len(df):,} papers with abstracts from {df['Biobank'].nunique()} biobanks")
        print(f"(Filtered out {initial_count - len(df):,} papers without meaningful abstracts)")
        
        # Run abstract-based analysis
        print("\n🔬 Running abstract-based bias detection...")
        print("   - Extracting demographic and methodological features")
        print("   - Identifying bias indicators and limitations")
        print("   - Calculating risk and equity scores")
        
        results_df = analyze_papers_abstract_based(df)
        
        # Save main results
        results_file = OUT_DIR / "bias_detection_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Saved bias detection results: {results_file}")
        
        # Create biobank profiles
        print("\n📊 Creating biobank bias profiles...")
        detector = AbstractBiasDetector()
        biobank_profiles = analyze_biobank_profiles(df, detector)
        
        # Save temporal trends
        temporal_df = results_df.groupby('Year').agg({
            'overall_bias_risk': 'mean',
            'equity_consideration': 'mean',
            'sample_size': 'median',
            'mentions_diversity': 'mean',
            'is_longitudinal': 'mean'
        }).reset_index()
        temporal_df.columns = ['Year', 'mean_bias_risk', 'mean_equity_score', 
                               'median_sample_size', 'prop_diversity', 'prop_longitudinal']
        temporal_file = OUT_DIR / "temporal_trends.csv"
        temporal_df.to_csv(temporal_file, index=False)
        
        # Save biobank-level analysis
        # Save biobank-level analysis
        biobank_summary = []
        for biobank, profile in biobank_profiles.items():
            # Calculate mean bias indicator count - FIXED: values are already counts
            if profile['bias_indicators']:
                mean_indicator_count = np.mean(list(profile['bias_indicators'].values()))
            else:
                mean_indicator_count = 0
            
            biobank_summary.append({
                'Biobank': biobank,
                'overall_quality': 1 - profile['mean_risk_score'],
                'red_flag_count': mean_indicator_count,  # Use the calculated mean
                'biases_addressed': profile['mean_transparency_score']
            })
        biobank_df = pd.DataFrame(biobank_summary)
        biobank_file = OUT_DIR / "biobank_bias_analysis.csv"
        biobank_df.to_csv(biobank_file, index=False)    
        
        # Create visualizations
        print("\n📊 Creating bias landscape dashboard...")
        create_bias_landscape_dashboard(results_df, biobank_profiles)
        
        # Generate report
        print("\n📋 Generating bias assessment report...")
        report = generate_bias_report(results_df, biobank_profiles)
        
        # Print key findings
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - KEY FINDINGS")
        print("=" * 80)
        
        print(f"\n📈 BIAS RISK METRICS:")
        print(f"   • Mean bias risk: {report['overall_summary']['mean_bias_risk']:.2f}")
        print(f"   • Papers with critical issues: {report['overall_summary']['papers_with_critical_issues']:,}")
        median_sample = report['overall_summary']['median_sample_size']
        median_sample_display = f"{median_sample:,.0f}" if median_sample else "N/A"
        print(f"   • Median sample size: {median_sample_display}")

        print(f"\n🌍 DIVERSITY & EQUITY:")
        print(f"   • Papers mentioning diversity: {report['overall_summary']['papers_mentioning_diversity']:,}")
        print(f"   • Single-country studies: {report['equity_gaps']['single_country_studies']:,}")
        print(f"   • Low equity consideration: {report['equity_gaps']['low_diversity_papers']:,}")
        
        print(f"\n🏆 TOP PERFORMING BIOBANK:")
        best_biobank = min(biobank_profiles.items(), key=lambda x: x[1]['mean_risk_score'])
        print(f"   • {best_biobank[0]}: Risk score {best_biobank[1]['mean_risk_score']:.2f}")
        
        print(f"\n⚠️ HIGHEST RISK BIOBANK:")
        worst_biobank = max(biobank_profiles.items(), key=lambda x: x[1]['mean_risk_score'])
        print(f"   • {worst_biobank[0]}: Risk score {worst_biobank[1]['mean_risk_score']:.2f}")
        
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for rec in report['recommendations'][:3]:
            print(f"   • {rec}")
        
        print(f"\n📂 All outputs saved to: {OUT_DIR}")
        
        return results_df, report
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, report = main()