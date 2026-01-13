"""
Research Impact & Influence Analysis
Evaluates research quality, influence, and knowledge transfer patterns

SCRIPT: PYTHON/01-02-research-impact-&-influence-analysis.py
OUTPUT: ANALYSIS/01-02-RESEARCH-IMPACT/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-02-RESEARCH-IMPACT")
os.makedirs(analysis_dir, exist_ok=True)

def load_journal_metrics():
    """
    Create a mock journal impact factor database
    In practice, you'd load this from SCImago, JCR, or other sources
    """
    
    # High-impact journals commonly used in biobank research
    journal_metrics = {
        'nature': {'impact_factor': 49.962, 'quartile': 'Q1', 'field': 'Multidisciplinary'},
        'nature genetics': {'impact_factor': 31.616, 'quartile': 'Q1', 'field': 'Genetics'},
        'nature medicine': {'impact_factor': 36.13, 'quartile': 'Q1', 'field': 'Medicine'},
        'science': {'impact_factor': 47.728, 'quartile': 'Q1', 'field': 'Multidisciplinary'},
        'cell': {'impact_factor': 41.584, 'quartile': 'Q1', 'field': 'Cell Biology'},
        'lancet': {'impact_factor': 79.32, 'quartile': 'Q1', 'field': 'Medicine'},
        'new england journal of medicine': {'impact_factor': 91.245, 'quartile': 'Q1', 'field': 'Medicine'},
        'american journal of human genetics': {'impact_factor': 9.924, 'quartile': 'Q1', 'field': 'Genetics'},
        'plos genetics': {'impact_factor': 4.452, 'quartile': 'Q1', 'field': 'Genetics'},
        'human molecular genetics': {'impact_factor': 4.901, 'quartile': 'Q1', 'field': 'Genetics'},
        'genome research': {'impact_factor': 6.205, 'quartile': 'Q1', 'field': 'Genetics'},
        'nature communications': {'impact_factor': 14.919, 'quartile': 'Q1', 'field': 'Multidisciplinary'},
        'plos one': {'impact_factor': 3.24, 'quartile': 'Q2', 'field': 'Multidisciplinary'},
        'european journal of human genetics': {'impact_factor': 4.188, 'quartile': 'Q2', 'field': 'Genetics'},
        'human genetics': {'impact_factor': 3.619, 'quartile': 'Q2', 'field': 'Genetics'},
        'bmc medical genomics': {'impact_factor': 3.441, 'quartile': 'Q2', 'field': 'Genetics'}
    }
    
    return journal_metrics

def calculate_impact_metrics(df, biobank_col='Biobank', journal_col='Journal', year_col='Year'):
    """Calculate various impact metrics for each biobank"""
    
    journal_metrics = load_journal_metrics()
    impact_analysis = {}
    
    for biobank in df[biobank_col].unique():
        biobank_data = df[df[biobank_col] == biobank].copy()
        
        # Journal impact analysis
        journal_impacts = []
        high_impact_count = 0
        q1_count = 0
        
        for journal in biobank_data[journal_col]:
            if pd.notna(journal):
                journal_clean = journal.lower().strip()
                if journal_clean in journal_metrics:
                    impact = journal_metrics[journal_clean]['impact_factor']
                    journal_impacts.append(impact)
                    
                    if impact >= 10:  # High impact threshold
                        high_impact_count += 1
                    if journal_metrics[journal_clean]['quartile'] == 'Q1':
                        q1_count += 1
                else:
                    journal_impacts.append(2.0)  # Default for unknown journals
        
        # Calculate metrics
        avg_impact = np.mean(journal_impacts) if journal_impacts else 0
        median_impact = np.median(journal_impacts) if journal_impacts else 0
        high_impact_rate = high_impact_count / len(biobank_data) if len(biobank_data) > 0 else 0
        q1_rate = q1_count / len(biobank_data) if len(biobank_data) > 0 else 0
        
        # Temporal impact trends
        yearly_impacts = defaultdict(list)
        for idx, row in biobank_data.iterrows():
            year = row[year_col]
            journal = str(row[journal_col]).lower().strip()
            if journal in journal_metrics:
                yearly_impacts[year].append(journal_metrics[journal]['impact_factor'])
            else:
                yearly_impacts[year].append(2.0)
        
        yearly_avg_impact = {year: np.mean(impacts) for year, impacts in yearly_impacts.items()}
        
        # Research diversity (journal diversity)
        unique_journals = biobank_data[journal_col].nunique()
        total_papers = len(biobank_data)
        journal_diversity = unique_journals / total_papers if total_papers > 0 else 0
        
        impact_analysis[biobank] = {
            'total_papers': total_papers,
            'avg_impact_factor': avg_impact,
            'median_impact_factor': median_impact,
            'high_impact_rate': high_impact_rate,
            'q1_journal_rate': q1_rate,
            'journal_diversity': journal_diversity,
            'yearly_impact_trends': yearly_avg_impact,
            'impact_distribution': journal_impacts
        }
    
    return impact_analysis

def analyze_research_quality_indicators(df, title_col='Title', abstract_col='Abstract', mesh_col='MeSH_Terms'):
    """Analyze research quality indicators from textual content"""
    
    quality_metrics = {}
    
    for biobank in df['Biobank'].unique():
        biobank_data = df[df['Biobank'] == biobank].copy()
        
        # Title analysis
        titles = biobank_data[title_col].dropna()
        
        if len(titles) > 0:
            title_lengths = [len(str(title)) for title in titles]
            avg_title_length = np.mean(title_lengths)
            
            # Look for methodology indicators in titles
            methodology_keywords = ['randomized', 'systematic review', 'meta-analysis', 
                                   'genome-wide', 'mendelian randomization', 'gwas']
            methodology_count = sum(1 for title in titles 
                                  if any(keyword in str(title).lower() for keyword in methodology_keywords))
            methodology_rate = methodology_count / len(titles)
            
            # International collaboration indicators
            intl_keywords = ['international', 'consortium', 'collaboration', 'multi-', 'global']
            intl_count = sum(1 for title in titles 
                           if any(keyword in str(title).lower() for keyword in intl_keywords))
            intl_collaboration_rate = intl_count / len(titles)
        else:
            avg_title_length = 0
            methodology_rate = 0
            intl_collaboration_rate = 0
        
        # MeSH term complexity analysis
        if mesh_col in biobank_data.columns:
            mesh_data = biobank_data[mesh_col].dropna()
            if len(mesh_data) > 0:
                mesh_counts = [len(str(mesh).split(';')) for mesh in mesh_data]
                avg_mesh_terms = np.mean(mesh_counts)
                mesh_complexity = np.std(mesh_counts)  # Variability as complexity measure
            else:
                avg_mesh_terms = 0
                mesh_complexity = 0
        else:
            avg_mesh_terms = 0
            mesh_complexity = 0
        
        quality_metrics[biobank] = {
            'avg_title_length': avg_title_length,
            'methodology_rate': methodology_rate,
            'international_collaboration_rate': intl_collaboration_rate,
            'avg_mesh_terms': avg_mesh_terms,
            'mesh_complexity': mesh_complexity,
            'total_papers_analyzed': len(biobank_data)
        }
    
    return quality_metrics

def calculate_knowledge_transfer_metrics(df, year_col='Year', biobank_col='Biobank'):
    """Calculate metrics related to knowledge transfer and research acceleration"""
    
    transfer_metrics = {}
    
    for biobank in df[biobank_col].unique():
        biobank_data = df[df[biobank_col] == biobank].copy()
        
        # Publication velocity (papers per year)
        yearly_counts = biobank_data[year_col].value_counts().sort_index()
        
        if len(yearly_counts) > 2:
            # Calculate growth rate
            years = yearly_counts.index.values
            counts = yearly_counts.values
            
            # Linear regression for growth trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
            growth_rate = slope
            growth_acceleration = r_value ** 2  # R-squared as acceleration measure
            
            # Research maturity (time from first to current publications)
            research_span = years.max() - years.min()
            maturity_score = research_span / (2024 - years.min()) if years.min() < 2024 else 0
        else:
            growth_rate = 0
            growth_acceleration = 0
            research_span = 0
            maturity_score = 0
        
        # Recent activity (2020-2024 focus)
        recent_data = biobank_data[biobank_data[year_col] >= 2020]
        recent_activity_rate = len(recent_data) / len(biobank_data) if len(biobank_data) > 0 else 0
        
        transfer_metrics[biobank] = {
            'publication_growth_rate': growth_rate,
            'growth_acceleration': growth_acceleration,
            'research_maturity': maturity_score,
            'recent_activity_rate': recent_activity_rate,
            'research_span_years': research_span,
            'total_publications': len(biobank_data)
        }
    
    return transfer_metrics

def create_impact_visualizations(impact_data, quality_data, transfer_data):
    """Create comprehensive impact analysis visualizations"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Average Journal Impact Factor', 'High-Impact Publication Rate (%)',
                       'Research Quality Indicators', 'Publication Growth Trends',
                       'Journal Diversity vs Impact', 'Research Maturity Assessment'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    biobanks = list(impact_data.keys())
    colors = px.colors.qualitative.Set1[:len(biobanks)]
    
    # 1. Average Impact Factor
    avg_impacts = [impact_data[b]['avg_impact_factor'] for b in biobanks]
    fig.add_trace(go.Bar(
        x=biobanks, y=avg_impacts, name='Avg Impact Factor',
        marker_color=colors, text=[f'{x:.2f}' for x in avg_impacts],
        textposition='outside'
    ), row=1, col=1)
    
    # 2. High-Impact Rate
    high_impact_rates = [impact_data[b]['high_impact_rate'] * 100 for b in biobanks]
    fig.add_trace(go.Bar(
        x=biobanks, y=high_impact_rates, name='High-Impact Rate (%)',
        marker_color=colors, text=[f'{x:.1f}%' for x in high_impact_rates],
        textposition='outside'
    ), row=1, col=2)
    
    # 3. Research Quality Indicators
    methodology_rates = [quality_data[b]['methodology_rate'] * 100 for b in biobanks]
    intl_rates = [quality_data[b]['international_collaboration_rate'] * 100 for b in biobanks]
    
    fig.add_trace(go.Scatter(
        x=methodology_rates, y=intl_rates, mode='markers+text',
        text=biobanks, textposition='top center',
        marker=dict(size=20, color=colors, opacity=0.7),
        name='Quality Indicators'
    ), row=2, col=1)
    
    # 4. Growth Trends
    growth_rates = [transfer_data[b]['publication_growth_rate'] for b in biobanks]
    accelerations = [transfer_data[b]['growth_acceleration'] for b in biobanks]
    
    fig.add_trace(go.Scatter(
        x=growth_rates, y=accelerations, mode='markers+text',
        text=biobanks, textposition='top center',
        marker=dict(size=20, color=colors, opacity=0.7),
        name='Growth Trends'
    ), row=2, col=2)
    
    # 5. Diversity vs Impact
    diversities = [impact_data[b]['journal_diversity'] for b in biobanks]
    
    fig.add_trace(go.Scatter(
        x=diversities, y=avg_impacts, mode='markers+text',
        text=biobanks, textposition='top center',
        marker=dict(size=20, color=colors, opacity=0.7),
        name='Diversity vs Impact'
    ), row=3, col=1)
    
    # 6. Research Maturity
    maturities = [transfer_data[b]['research_maturity'] for b in biobanks]
    fig.add_trace(go.Bar(
        x=biobanks, y=maturities, name='Research Maturity',
        marker_color=colors, text=[f'{x:.2f}' for x in maturities],
        textposition='outside'
    ), row=3, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Biobank", row=1, col=1)
    fig.update_xaxes(title_text="Biobank", row=1, col=2)
    fig.update_xaxes(title_text="Methodology Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Growth Rate (papers/year)", row=2, col=2)
    fig.update_xaxes(title_text="Journal Diversity", row=3, col=1)
    fig.update_xaxes(title_text="Biobank", row=3, col=2)
    
    fig.update_yaxes(title_text="Impact Factor", row=1, col=1)
    fig.update_yaxes(title_text="High-Impact Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="International Collaboration (%)", row=2, col=1)
    fig.update_yaxes(title_text="Growth Acceleration (R²)", row=2, col=2)
    fig.update_yaxes(title_text="Avg Impact Factor", row=3, col=1)
    fig.update_yaxes(title_text="Maturity Score", row=3, col=2)
    
    fig.update_layout(height=1200, showlegend=False, 
                     title_text="Biobank Research Impact & Influence Analysis")
    
    # Save visualization
    output_file = os.path.join(analysis_dir, 'research_impact_analysis.html')
    fig.write_html(output_file)
    print(f"✅ Visualization saved: {output_file}")
    
    return fig

def generate_impact_report(impact_data, quality_data, transfer_data):
    """Generate comprehensive impact analysis report"""
    
    report = """
BIOBANK RESEARCH IMPACT & INFLUENCE ANALYSIS
===========================================

JOURNAL IMPACT METRICS:
"""
    
    # Rank biobanks by different metrics
    impact_ranking = sorted(impact_data.items(), key=lambda x: x[1]['avg_impact_factor'], reverse=True)
    quality_ranking = sorted(quality_data.items(), key=lambda x: x[1]['methodology_rate'], reverse=True)
    growth_ranking = sorted(transfer_data.items(), key=lambda x: x[1]['publication_growth_rate'], reverse=True)
    
    for biobank, data in impact_ranking:
        report += f"""
{biobank}:
  - Average Impact Factor: {data['avg_impact_factor']:.2f}
  - High-Impact Rate: {data['high_impact_rate']*100:.1f}%
  - Q1 Journal Rate: {data['q1_journal_rate']*100:.1f}%
  - Journal Diversity: {data['journal_diversity']:.3f}
"""
    
    report += "\nRESEARCH QUALITY INDICATORS:\n"
    for biobank, data in quality_ranking:
        report += f"""
{biobank}:
  - Methodology Focus: {data['methodology_rate']*100:.1f}%
  - International Collaboration: {data['international_collaboration_rate']*100:.1f}%
  - Average MeSH Terms: {data['avg_mesh_terms']:.1f}
  - Research Complexity: {data['mesh_complexity']:.2f}
"""
    
    report += "\nKNOWLEDGE TRANSFER & GROWTH:\n"
    for biobank, data in growth_ranking:
        report += f"""
{biobank}:
  - Growth Rate: {data['publication_growth_rate']:.1f} papers/year
  - Growth Acceleration: {data['growth_acceleration']:.3f}
  - Research Maturity: {data['research_maturity']:.2f}
  - Recent Activity (2020+): {data['recent_activity_rate']*100:.1f}%
"""
    
    # Overall insights
    report += f"""
SUMMARY INSIGHTS:
- Highest Impact: {impact_ranking[0][0]} (IF: {impact_ranking[0][1]['avg_impact_factor']:.2f})
- Best Quality Indicators: {quality_ranking[0][0]} ({quality_ranking[0][1]['methodology_rate']*100:.1f}% methodology focus)
- Fastest Growing: {growth_ranking[0][0]} ({growth_ranking[0][1]['publication_growth_rate']:.1f} papers/year)
- Most Diverse: {max(impact_data.items(), key=lambda x: x[1]['journal_diversity'])[0]}
"""
    
    return report

def run_impact_analysis(df):
    """Main function to run impact and influence analysis"""
    
    print("📊 Calculating journal impact metrics...")
    impact_data = calculate_impact_metrics(df)
    
    print("🔬 Analyzing research quality indicators...")
    quality_data = analyze_research_quality_indicators(df)
    
    print("📈 Computing knowledge transfer metrics...")
    transfer_data = calculate_knowledge_transfer_metrics(df)
    
    print("📋 Creating visualizations...")
    fig = create_impact_visualizations(impact_data, quality_data, transfer_data)
    
    print("📝 Generating impact report...")
    report = generate_impact_report(impact_data, quality_data, transfer_data)
    
    # Save report
    report_file = os.path.join(analysis_dir, 'impact_analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✅ Report saved: {report_file}")
    
    # Save impact metrics to CSV
    impact_df = pd.DataFrame(impact_data).T
    impact_file = os.path.join(analysis_dir, 'impact_metrics.csv')
    impact_df.to_csv(impact_file)
    print(f"✅ Impact metrics saved: {impact_file}")
    
    # Save quality metrics to CSV
    quality_df = pd.DataFrame(quality_data).T
    quality_file = os.path.join(analysis_dir, 'quality_metrics.csv')
    quality_df.to_csv(quality_file)
    print(f"✅ Quality metrics saved: {quality_file}")
    
    # Save transfer metrics to CSV
    transfer_df = pd.DataFrame(transfer_data).T
    transfer_file = os.path.join(analysis_dir, 'transfer_metrics.csv')
    transfer_df.to_csv(transfer_file)
    print(f"✅ Transfer metrics saved: {transfer_file}")
    
    return {
        'impact_metrics': impact_data,
        'quality_metrics': quality_data,
        'transfer_metrics': transfer_data,
        'visualization': fig,
        'report': report
    }

if __name__ == "__main__":
    # Load and filter data
    df = pd.read_csv(os.path.join(data_dir, 'biobank_research_data.csv'))
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    
    print(f"📊 Loaded {len(df):,} publications for impact analysis")
    
    # Run analysis
    results = run_impact_analysis(df)
    
    # Display results
    print(results['report'])
    results['visualization'].show()
    
    print(f"\n🎯 Analysis complete! Results saved to: {analysis_dir}")