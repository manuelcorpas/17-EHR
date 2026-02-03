import streamlit as st
import pandas as pd
import io
import os
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# UPDATED IMPORTS: All HEIM modules now imported from PYTHON subdirectory
# Since app.py moved from PYTHON/ to root, we need to specify PYTHON. prefix
# ============================================================================
from PYTHON.scoring import calculate_heim_score, get_badge_interpretation, get_score_color
from PYTHON.reference_data import ANCESTRY_NAMES
from PYTHON.visualizations import (
    create_ancestry_pie_chart,
    create_ancestry_comparison_chart,
    create_geographic_map,
    create_age_histogram,
    create_sex_distribution_chart,
    create_diversity_radar_chart,
    create_country_bar_chart
)
from PYTHON.recommendations import generate_recommendations, get_severity_color, get_severity_icon
from PYTHON.pdf_generator import generate_pdf_scorecard

# ============================================================================
# UPDATED PATH CONFIGURATION
# Since app.py is now in root (not PYTHON/), we use parent once, not twice
# OLD: PROJECT_ROOT = Path(__file__).parent.parent  # Was: PYTHON/ -> root
# NEW: PROJECT_ROOT = Path(__file__).parent         # Now: root directly
# ============================================================================
PROJECT_ROOT = Path(__file__).parent  # app.py is now in root, so parent IS root
DATA_DIR = PROJECT_ROOT / "DATA"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="HEIM Dataset Equity Scorecard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Validation constants
REQUIRED_COLUMNS = ['participant_id', 'ancestry', 'age', 'sex', 'country']
OPTIONAL_COLUMNS = ['region', 'disease_status', 'recruitment_site']

VALID_ANCESTRY_CODES = [
    'AFR', 'EUR', 'EAS', 'SAS', 'AMR', 'MID', 'OCE', 'OTH',
    # 1000 Genomes populations
    'YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB',  # African
    'CEU', 'TSI', 'FIN', 'GBR', 'IBS',  # European
    'CHB', 'JPT', 'CHS', 'CDX', 'KHV',  # East Asian
    'GIH', 'PJL', 'BEB', 'STU', 'ITU',  # South Asian
    'MXL', 'PUR', 'CLM', 'PEL'  # Admixed American
]

VALID_SEX_VALUES = ['M', 'F', 'Male', 'Female', 'Other', 'Unknown', 'U']


def create_sample_data() -> pd.DataFrame:
    """Generate sample dataset for testing."""
    sample_data = {
        'participant_id': [f'P{str(i).zfill(3)}' for i in range(1, 51)],
        'ancestry': ['EUR'] * 20 + ['AFR'] * 10 + ['EAS'] * 8 + ['SAS'] * 7 + ['AMR'] * 5,
        'age': [25, 34, 45, 56, 67, 32, 41, 29, 38, 52, 44, 33, 61, 27, 49, 55, 36, 42, 58, 31,
                28, 39, 47, 53, 35, 43, 50, 26, 40, 48, 30, 37, 46, 54, 62, 24, 33, 41, 51, 59,
                29, 38, 44, 52, 34, 42, 48, 56, 32, 40],
        'sex': ['F', 'M', 'F', 'M', 'F'] * 10,
        'country': (
            # EUR: 20 participants
            ['United Kingdom', 'United States', 'Germany', 'France', 'Spain'] * 4 + 
            # AFR: 10 participants
            ['Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Ethiopia'] * 2 +
            # EAS: 8 participants
            ['China', 'Japan', 'South Korea', 'Singapore'] * 2 +
            # SAS: 7 participants
            ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Afghanistan', 'India'] +
            # AMR: 5 participants
            ['Mexico', 'Brazil', 'Argentina', 'Peru', 'Colombia']
        ),
        'region': ['Europe'] * 20 + ['Africa'] * 10 + ['East Asia'] * 8 + ['South Asia'] * 7 + ['Americas'] * 5
    }
    return pd.DataFrame(sample_data)


def save_sample_data():
    """Save sample data to DATA directory."""
    sample_path = DATA_DIR / "sample_data.csv"
    if not sample_path.exists():
        df = create_sample_data()
        df.to_csv(sample_path, index=False)
        return True
    return False


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validate uploaded dataframe.
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"❌ Missing required columns: {', '.join(missing_cols)}")
    
    if errors:  # Can't continue validation without required columns
        return False, errors, warnings
    
    # Check for duplicate participant IDs
    if df['participant_id'].duplicated().any():
        duplicates = df[df['participant_id'].duplicated()]['participant_id'].tolist()
        errors.append(f"❌ Duplicate participant IDs found: {', '.join(map(str, duplicates[:5]))}")
    
    # Validate age column
    if df['age'].dtype not in ['int64', 'float64']:
        errors.append("❌ Age column must contain numeric values")
    else:
        invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)]
        if len(invalid_ages) > 0:
            errors.append(f"❌ {len(invalid_ages)} age values are outside valid range (0-120)")
        
        if df['age'].isnull().any():
            warnings.append(f"⚠️ {df['age'].isnull().sum()} missing age values ({df['age'].isnull().sum()/len(df)*100:.1f}%)")
    
    # Validate ancestry codes
    invalid_ancestry = df[~df['ancestry'].isin(VALID_ANCESTRY_CODES) & df['ancestry'].notna()]
    if len(invalid_ancestry) > 0:
        unique_invalid = invalid_ancestry['ancestry'].unique()
        warnings.append(f"⚠️ Unrecognized ancestry codes: {', '.join(map(str, unique_invalid[:10]))}")
        warnings.append(f"   Valid codes include: {', '.join(VALID_ANCESTRY_CODES[:10])}...")
    
    if df['ancestry'].isnull().any():
        warnings.append(f"⚠️ {df['ancestry'].isnull().sum()} missing ancestry values ({df['ancestry'].isnull().sum()/len(df)*100:.1f}%)")
    
    # Validate sex values
    invalid_sex = df[~df['sex'].isin(VALID_SEX_VALUES) & df['sex'].notna()]
    if len(invalid_sex) > 0:
        unique_invalid_sex = invalid_sex['sex'].unique()
        warnings.append(f"⚠️ Non-standard sex values: {', '.join(map(str, unique_invalid_sex))}")
        warnings.append(f"   Expected values: {', '.join(VALID_SEX_VALUES)}")
    
    if df['sex'].isnull().any():
        warnings.append(f"⚠️ {df['sex'].isnull().sum()} missing sex values ({df['sex'].isnull().sum()/len(df)*100:.1f}%)")
    
    # Check dataset size
    if len(df) < 20:
        warnings.append(f"⚠️ Small dataset (n={len(df)}). Diversity metrics may be unreliable with <20 participants.")
    
    # Check for missing data overall
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 10:
        warnings.append(f"⚠️ High overall missingness: {missing_pct:.1f}% of data is missing")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def display_data_summary(df: pd.DataFrame):
    """Display summary statistics of the uploaded data."""
    st.subheader("📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Participants", len(df))
    
    with col2:
        st.metric("Unique Ancestries", df['ancestry'].nunique())
    
    with col3:
        st.metric("Countries", df['country'].nunique())
    
    with col4:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    # Show distribution breakdowns
    st.subheader("🔍 Distribution Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ancestry Distribution**")
        ancestry_counts = df['ancestry'].value_counts()
        ancestry_df = pd.DataFrame({
            'Ancestry': ancestry_counts.index,
            'Count': ancestry_counts.values,
            'Percentage': (ancestry_counts.values / len(df) * 100).round(1)
        })
        st.dataframe(ancestry_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**Sex Distribution**")
        sex_counts = df['sex'].value_counts()
        sex_df = pd.DataFrame({
            'Sex': sex_counts.index,
            'Count': sex_counts.values,
            'Percentage': (sex_counts.values / len(df) * 100).round(1)
        })
        st.dataframe(sex_df, hide_index=True, use_container_width=True)
    
    # Age statistics
    st.write("**Age Statistics**")
    age_stats = df['age'].describe()
    age_col1, age_col2, age_col3, age_col4 = st.columns(4)
    with age_col1:
        st.metric("Mean Age", f"{age_stats['mean']:.1f}")
    with age_col2:
        st.metric("Median Age", f"{age_stats['50%']:.1f}")
    with age_col3:
        st.metric("Age Range", f"{age_stats['min']:.0f}-{age_stats['max']:.0f}")
    with age_col4:
        st.metric("Std Dev", f"{age_stats['std']:.1f}")


def create_score_gauge(score: float, badge: str, badge_color: str):
    """Create a circular gauge for HEIM score display."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{badge}", 'font': {'size': 24, 'color': badge_color}},
        number = {'suffix': " / 100", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': badge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#FFE6E6'},
                {'range': [40, 60], 'color': '#FFE8CC'},
                {'range': [60, 75], 'color': '#E8E8E8'},
                {'range': [75, 90], 'color': '#FFF8DC'},
                {'range': [90, 100], 'color': '#F0F0F0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig


def display_heim_scorecard(heim_result: Dict):
    """Display comprehensive HEIM scorecard with visualizations."""
    st.markdown("---")
    st.header("🎯 HEIM Representation Score")
    
    # Main score display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Score gauge
        fig = create_score_gauge(
            heim_result['overall_score'],
            heim_result['badge'],
            heim_result['badge_color']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"### Overall HEIM Score: **{heim_result['overall_score']}/100**")
        st.markdown(f"**Badge Level:** {heim_result['badge']}")
        st.info(get_badge_interpretation(heim_result['badge']))
        st.caption(f"Based on analysis of **{heim_result['sample_size']} participants**")
    
    # Dimension breakdown
    st.markdown("---")
    st.subheader("📊 Dimension Breakdown")
    
    dimensions = heim_result['dimensions']
    
    # Create bars for each dimension
    for dim_name, dim_label in [
        ('ancestry', '🌍 Ancestry Diversity'),
        ('geographic', '📍 Geographic Diversity'),
        ('age', '👥 Age Distribution'),
        ('sex', '⚧️ Sex Balance')
    ]:
        dim_data = dimensions[dim_name]
        score = dim_data['score']
        weight = dim_data['weight']
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{dim_label}**")
            # Progress bar
            color = get_score_color(score)
            st.markdown(f"""
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 30px; position: relative;">
                    <div style="background-color: {color}; width: {score}%; height: 100%; border-radius: 10px; 
                         display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {score:.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Weight", f"{weight*100:.0f}%")
    
    # Detailed metrics
    st.markdown("---")
    st.subheader("🔍 Detailed Metrics")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Ancestry",
        "📍 Geography", 
        "👥 Age",
        "⚧️ Sex"
    ])
    
    with tab1:
        ancestry_details = dimensions['ancestry']['details']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Diversity Index", f"{ancestry_details.get('diversity', 0):.3f}")
        with col2:
            st.metric("Evenness Score", f"{ancestry_details.get('evenness', 0):.3f}")
        with col3:
            st.metric("Unique Ancestries", ancestry_details.get('unique_ancestries', 0))
        
        if 'representation_gap' in ancestry_details:
            st.info(f"**Representation Gap:** {ancestry_details['representation_gap']:.3f} (lower is better)")
    
    with tab2:
        geo_details = dimensions['geographic']['details']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Countries", geo_details.get('unique_countries', 0))
        with col2:
            st.metric("Diversity Index", f"{geo_details.get('diversity', 0):.3f}")
    
    with tab3:
        age_details = dimensions['age']['details']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{age_details.get('mean', 0):.1f}")
        with col2:
            st.metric("Median", f"{age_details.get('median', 0):.1f}")
        with col3:
            st.metric("Range", f"{age_details.get('range', 0):.0f} years")
        with col4:
            st.metric("Std Dev", f"{age_details.get('std', 0):.1f}")
    
    with tab4:
        sex_details = dimensions['sex']['details']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Balance Score", f"{sex_details.get('balance_score', 0):.1f}")
        with col2:
            st.metric("Diversity Index", f"{sex_details.get('diversity', 0):.3f}")
        
        if 'distribution' in sex_details:
            st.write("**Distribution:**")
            for sex_cat, count in sex_details['distribution'].items():
                st.write(f"- {sex_cat}: {count}")


def display_visualizations(df: pd.DataFrame, heim_result: Dict):
    """Display interactive visualizations of dataset diversity."""
    st.markdown("---")
    st.header("📈 Visual Analytics")
    
    # Radar chart overview
    st.subheader("🎯 Overall Dimension Scores")
    radar_fig = create_diversity_radar_chart(heim_result)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Ancestry visualizations
    st.markdown("---")
    st.subheader("🌍 Ancestry Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pie_fig = create_ancestry_pie_chart(df)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        comparison_fig = create_ancestry_comparison_chart(df)
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Geographic visualizations
    st.markdown("---")
    st.subheader("📍 Geographic Distribution")
    
    # World map
    map_fig = create_geographic_map(df)
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Top countries bar chart
    if df['country'].nunique() > 5:
        country_bar_fig = create_country_bar_chart(df, top_n=10)
        st.plotly_chart(country_bar_fig, use_container_width=True)
    
    # Age and Sex visualizations
    st.markdown("---")
    st.subheader("👥 Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_fig = create_age_histogram(df)
        st.plotly_chart(age_fig, use_container_width=True)
    
    with col2:
        sex_fig = create_sex_distribution_chart(df)
        st.plotly_chart(sex_fig, use_container_width=True)


def display_recommendations(df: pd.DataFrame, heim_result: Dict):
    """Display actionable recommendations based on gaps."""
    st.markdown("---")
    st.header("💡 Recommendations & Action Plan")
    
    # Generate recommendations
    with st.spinner("Analyzing gaps and generating recommendations..."):
        recs = generate_recommendations(df, heim_result)
    
    # Overall status
    st.subheader("📋 Overall Assessment")
    
    status_colors = {
        'critical': '#FF4444',
        'needs_improvement': '#FF8C00',
        'good_with_gaps': '#FFA500',
        'excellent': '#00C851'
    }
    
    status_color = status_colors.get(recs['overall_status'], '#808080')
    
    st.markdown(f"""
        <div style="background-color: {status_color}20; border-left: 4px solid {status_color}; 
             padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {status_color};">{recs['overall_message']}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Issues Identified", recs['total_issues'])
    with col2:
        st.metric("Critical Priority", recs['critical_issues'], 
                 delta=None if recs['critical_issues'] == 0 else f"-{recs['critical_issues']}", 
                 delta_color="inverse")
    with col3:
        st.metric("High Priority", recs['high_priority_issues'],
                 delta=None if recs['high_priority_issues'] == 0 else f"-{recs['high_priority_issues']}", 
                 delta_color="inverse")
    
    # Priority Actions
    if recs['priority_actions']:
        st.markdown("---")
        st.subheader("🎯 Top Priority Actions")
        st.write("Start with these high-impact improvements:")
        
        for i, action in enumerate(recs['priority_actions'], 1):
            severity_color = get_severity_color(action['severity'])
            severity_icon = get_severity_icon(action['severity'])
            
            with st.expander(f"{severity_icon} **Priority {i}:** {action['title']}", expanded=(i==1)):
                st.markdown(f"**Dimension:** {action['dimension'].title()}")
                st.markdown(f"**Severity:** {action['severity'].title()}")
                st.markdown(f"**Next Step:** {action['top_action']}")
    
    # Detailed Recommendations by Dimension
    st.markdown("---")
    st.subheader("🔎 Detailed Recommendations by Dimension")
    
    tabs = st.tabs([
        f"🌍 Ancestry ({len(recs['by_dimension']['ancestry'])})",
        f"📍 Geography ({len(recs['by_dimension']['geographic'])})",
        f"👥 Age ({len(recs['by_dimension']['age'])})",
        f"⚧️ Sex ({len(recs['by_dimension']['sex'])})"
    ])
    
    for tab, (dim_name, dim_recs) in zip(tabs, [
        ('ancestry', recs['by_dimension']['ancestry']),
        ('geographic', recs['by_dimension']['geographic']),
        ('age', recs['by_dimension']['age']),
        ('sex', recs['by_dimension']['sex'])
    ]):
        with tab:
            if not dim_recs:
                st.success(f"✅ No significant issues detected in {dim_name} dimension!")
                st.info("This dimension meets recommended diversity standards.")
            else:
                for rec in dim_recs:
                    severity_icon = get_severity_icon(rec['severity'])
                    severity_color = get_severity_color(rec['severity'])
                    
                    st.markdown(f"""
                        <div style="background-color: {severity_color}10; border-left: 3px solid {severity_color}; 
                             padding: 10px; border-radius: 3px; margin-bottom: 15px;">
                            <h4 style="margin: 0 0 5px 0;">{severity_icon} {rec['title']}</h4>
                            <p style="color: #666; margin: 5px 0;">{rec['description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Recommended Actions:**")
                    for action in rec['actions']:
                        st.markdown(f"- {action}")
                    
                    st.markdown("")  # Spacing
    
    # Return recommendations for PDF export
    return recs


def main():
    # Header
    st.markdown('<div class="main-header">🧬 HEIM Dataset Equity Scorecard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantifying representation and diversity in health datasets</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About HEIM")
        st.write("""
        **Health Equity Informative Marker (HEIM)** provides quantitative assessment of dataset diversity across:
        
        - 🌍 **Ancestry** representation
        - 📍 **Geographic** diversity  
        - 👥 **Age** distribution
        - ⚧️ **Sex/Gender** balance
        
        Upload your dataset to receive an instant equity scorecard.
        """)
        
        st.divider()
        
        st.header("Data Privacy")
        st.write("""
        🔒 **Your data never leaves your browser.**
        
        All processing happens locally. No data is uploaded to servers or stored.
        """)
        
        st.divider()
        
        st.header("Directory Structure")
        st.code(f"""
DATA/     → Input files
PYTHON/   → Source code
OUTPUT/   → Exported PDFs
        """)
        
        st.divider()
        
        st.info("**Version:** 6.0.0 (Week 6)\n\n**Developer:** Dr. Manuel Corpas")
    
    # Main content area
    st.header("📂 Upload Your Dataset")
    
    # File upload section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with columns: participant_id, ancestry, age, sex, country"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("📋 Use Sample Data", use_container_width=True):
            st.session_state['use_sample'] = True
    
    # Show required format
    with st.expander("📋 Required Data Format", expanded=False):
        st.write("""
        Your CSV must include these columns:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | `participant_id` | Unique identifier | P001, SUB_123 |
        | `ancestry` | Ancestry code | EUR, AFR, EAS, SAS, AMR |
        | `age` | Age in years | 45, 32, 67 |
        | `sex` | Sex/gender | M, F, Other |
        | `country` | Country of origin | United Kingdom, Nigeria |
        
        **Optional columns:** region, disease_status, recruitment_site
        """)
        
        st.write("**Ancestry codes:**")
        st.code("AFR (African), EUR (European), EAS (East Asian), SAS (South Asian), AMR (Admixed American), MID (Middle Eastern), OCE (Oceanian), OTH (Other)")
    
    # Process uploaded file or sample data
    df = None
    
    if 'use_sample' in st.session_state and st.session_state['use_sample']:
        df = create_sample_data()
        # Save sample data to DATA directory
        save_sample_data()
        st.success(f"✅ Sample data loaded successfully! (Also saved to DATA/sample_data.csv)")
        st.session_state['use_sample'] = False
    
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return
    
    # If we have data, validate and display it
    if df is not None:
        st.divider()
        
        # Validation
        is_valid, errors, warnings = validate_dataframe(df)
        
        # Display validation results
        if errors:
            st.error("**Validation Errors:**")
            for error in errors:
                st.error(error)
        
        if warnings:
            st.warning("**Validation Warnings:**")
            for warning in warnings:
                st.warning(warning)
        
        if is_valid:
            st.success("✅ **Data validation passed!** Your dataset is ready for analysis.")
            
            # Show data preview
            st.divider()
            st.subheader("👀 Data Preview")
            st.write("First 10 rows:")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show summary statistics
            st.divider()
            display_data_summary(df)
            
            # Calculate HEIM Score
            st.divider()
            with st.spinner("Calculating HEIM scores..."):
                try:
                    heim_result = calculate_heim_score(df, reference_type='coarse')
                    display_heim_scorecard(heim_result)
                    
                    # Display visualizations
                    display_visualizations(df, heim_result)
                    
                    # Generate and display recommendations (capture return value)
                    recs = display_recommendations(df, heim_result)
                    
                except Exception as e:
                    st.error(f"Error calculating HEIM scores: {str(e)}")
                    st.exception(e)
                    return  # Exit if error occurs
            
            # Download options
            st.divider()
            st.header("📥 Export & Download")
            
            # Generate timestamp once
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            dataset_name = uploaded_file.name if uploaded_file else "Sample Dataset"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💾 Download CSV Data",
                    data=csv_buffer.getvalue(),
                    file_name=f"HEIM_Data_{timestamp}.csv",
                    mime="text/csv",
                    help="Download your validated dataset"
                )
            
            with col2:
                # Generate PDF scorecard
                try:
                    # Create temp PDF in OUTPUT directory
                    pdf_filename = f"HEIM_Scorecard_{timestamp}.pdf"
                    pdf_path = OUTPUT_DIR / pdf_filename
                    
                    # Generate PDF
                    generate_pdf_scorecard(
                        df=df,
                        heim_result=heim_result,
                        recommendations=recs,
                        output_path=pdf_path,
                        dataset_name=dataset_name
                    )
                    
                    # Read PDF for download
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    st.download_button(
                        label="📄 Download PDF Scorecard",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        help="Download comprehensive HEIM scorecard as PDF"
                    )
                    
                    st.caption(f"✅ Saved to OUTPUT/{pdf_filename}")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.caption("💡 Try downloading CSV or TXT instead")
            
            with col3:
                # Recommendations text export
                export_text = f"""HEIM RECOMMENDATIONS
Dataset: {dataset_name}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

OVERALL ASSESSMENT
{recs['overall_message']}

TOP PRIORITY ACTIONS
"""
                for i, action in enumerate(recs['priority_actions'][:5], 1):
                    export_text += f"{i}. {action['title']}\n"
                
                st.download_button(
                    label="📋 Download Recommendations",
                    data=export_text,
                    file_name=f"HEIM_Recommendations_{timestamp}.txt",
                    mime="text/plain",
                    help="Download recommendations as text file"
                )
        
        else:
            st.error("⚠️ Please fix the errors above before proceeding.")


if __name__ == "__main__":
    main()