"""
Translational Research & Competitive Landscape Analysis
Evaluates clinical translation potential and competitive positioning of biobank research

SCRIPT: PYTHON/01-03-translational-research-&-competitive-landscape.py
OUTPUT: ANALYSIS/01-03-TRANSLATIONAL-COMPETITIVE/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-03-TRANSLATIONAL-COMPETITIVE")
os.makedirs(analysis_dir, exist_ok=True)

def analyze_translational_potential(df, title_col='Title', abstract_col='Abstract', mesh_col='MeSH_Terms'):
    """Analyze translational research potential using keyword analysis"""
    
    # Define translational research indicators
    basic_research_keywords = [
        'mechanism', 'pathway', 'molecular', 'cellular', 'in vitro', 'animal model',
        'mouse model', 'genetic variant', 'protein function', 'gene expression'
    ]
    
    clinical_research_keywords = [
        'clinical trial', 'patient', 'treatment', 'therapy', 'diagnosis', 'prognosis',
        'clinical outcome', 'healthcare', 'medical practice', 'intervention'
    ]
    
    translational_keywords = [
        'biomarker', 'drug target', 'therapeutic target', 'precision medicine',
        'personalized medicine', 'clinical application', 'translational', 'bench to bedside'
    ]
    
    population_health_keywords = [
        'epidemiology', 'public health', 'population', 'cohort', 'risk factor',
        'prevention', 'screening', 'health policy', 'burden of disease'
    ]
    
    translational_analysis = {}
    
    for biobank in df['Biobank'].unique():
        biobank_data = df[df['Biobank'] == biobank].copy()
        
        # Combine title and abstract for analysis
        combined_text = []
        for idx, row in biobank_data.iterrows():
            text = ""
            if pd.notna(row[title_col]):
                text += str(row[title_col]) + " "
            if abstract_col in row and pd.notna(row[abstract_col]):
                text += str(row[abstract_col]) + " "
            if pd.notna(row[mesh_col]):
                text += str(row[mesh_col]).replace(';', ' ')
            combined_text.append(text.lower())
        
        # Count keyword occurrences
        basic_scores = []
        clinical_scores = []
        translational_scores = []
        population_scores = []
        
        for text in combined_text:
            basic_score = sum(1 for kw in basic_research_keywords if kw in text)
            clinical_score = sum(1 for kw in clinical_research_keywords if kw in text)
            trans_score = sum(1 for kw in translational_keywords if kw in text)
            pop_score = sum(1 for kw in population_health_keywords if kw in text)
            
            basic_scores.append(basic_score)
            clinical_scores.append(clinical_score)
            translational_scores.append(trans_score)
            population_scores.append(pop_score)
        
        # Calculate research profile
        total_papers = len(biobank_data)
        
        basic_focus = np.mean(basic_scores)
        clinical_focus = np.mean(clinical_scores)
        translational_focus = np.mean(translational_scores)
        population_focus = np.mean(population_scores)
        
        # Translational readiness score
        translational_readiness = (clinical_focus + translational_focus) / (basic_focus + 1)
        
        # Clinical impact potential
        clinical_papers = sum(1 for score in clinical_scores if score > 0)
        clinical_impact_rate = clinical_papers / total_papers if total_papers > 0 else 0
        
        translational_analysis[biobank] = {
            'basic_research_focus': basic_focus,
            'clinical_research_focus': clinical_focus,
            'translational_focus': translational_focus,
            'population_health_focus': population_focus,
            'translational_readiness': translational_readiness,
            'clinical_impact_rate': clinical_impact_rate,
            'total_papers': total_papers
        }
    
    return translational_analysis

def analyze_research_uniqueness(df, mesh_col='MeSH_Terms'):
    """Analyze research uniqueness and competitive positioning"""
    
    # Create TF-IDF vectors for each biobank's research portfolio
    biobank_portfolios = {}
    
    for biobank in df['Biobank'].unique():
        biobank_data = df[df['Biobank'] == biobank]
        mesh_terms = biobank_data[mesh_col].dropna()
        
        # Combine all MeSH terms for this biobank
        combined_mesh = ' '.join([str(terms).replace(';', ' ') for terms in mesh_terms])
        biobank_portfolios[biobank] = combined_mesh.lower()
    
    # Calculate similarity matrix
    biobanks = list(biobank_portfolios.keys())
    documents = [biobank_portfolios[b] for b in biobanks]
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate pairwise similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Calculate uniqueness scores (1 - average similarity to others)
    uniqueness_scores = {}
    for i, biobank in enumerate(biobanks):
        # Average similarity to all other biobanks
        other_similarities = [similarity_matrix[i][j] for j in range(len(biobanks)) if i != j]
        avg_similarity = np.mean(other_similarities) if other_similarities else 0
        uniqueness_scores[biobank] = 1 - avg_similarity
    
    # Identify distinctive research areas
    feature_names = vectorizer.get_feature_names_out()
    distinctive_features = {}
    
    for i, biobank in enumerate(biobanks):
        biobank_vector = tfidf_matrix[i].toarray()[0]
        
        # Calculate relative importance compared to other biobanks
        relative_scores = []
        for j, score in enumerate(biobank_vector):
            if score > 0:
                # Compare this biobank's score to others' scores for same term
                other_scores = [tfidf_matrix[k][0, j] for k in range(len(biobanks)) if k != i]
                avg_other_score = np.mean(other_scores) if other_scores else 0
                relative_importance = score / (avg_other_score + 0.001)  # Avoid division by zero
                relative_scores.append((feature_names[j], relative_importance, score))
        
        # Sort by relative importance
        relative_scores.sort(key=lambda x: x[1], reverse=True)
        distinctive_features[biobank] = relative_scores[:10]  # Top 10 distinctive features
    
    return uniqueness_scores, similarity_matrix, distinctive_features, biobanks

def analyze_research_gaps_and_opportunities(df, mesh_col='MeSH_Terms'):
    """Identify research gaps and collaboration opportunities"""
    
    # Extract all unique MeSH terms across biobanks
    all_terms = set()
    biobank_terms = {}
    
    for biobank in df['Biobank'].unique():
        biobank_data = df[df['Biobank'] == biobank]
        terms = set()
        
        for mesh_string in biobank_data[mesh_col].dropna():
            mesh_list = [term.strip().lower() for term in str(mesh_string).split(';')]
            terms.update(mesh_list)
            all_terms.update(mesh_list)
        
        biobank_terms[biobank] = terms
    
    # Find gaps and overlaps
    gaps_analysis = {}
    
    for biobank in biobank_terms:
        biobank_set = biobank_terms[biobank]
        
        # Terms unique to this biobank
        unique_terms = biobank_set.copy()
        for other_biobank, other_terms in biobank_terms.items():
            if other_biobank != biobank:
                unique_terms -= other_terms
        
        # Terms this biobank lacks but others have
        missing_terms = set()
        for other_biobank, other_terms in biobank_terms.items():
            if other_biobank != biobank:
                missing_terms.update(other_terms - biobank_set)
        
        # Potential collaboration opportunities (terms shared with at least one other)
        collaboration_terms = biobank_set.copy()
        for other_biobank, other_terms in biobank_terms.items():
            if other_biobank != biobank:
                collaboration_terms &= other_terms
        
        gaps_analysis[biobank] = {
            'unique_terms': list(unique_terms)[:20],  # Top 20
            'missing_terms': list(missing_terms)[:20],
            'collaboration_opportunities': list(collaboration_terms)[:20],
            'uniqueness_ratio': len(unique_terms) / len(biobank_set) if biobank_set else 0,
            'coverage_ratio': len(biobank_set) / len(all_terms) if all_terms else 0
        }
    
    return gaps_analysis

def create_competitive_visualizations(translational_data, uniqueness_data, gaps_data):
    """Create competitive landscape visualizations"""
    
    uniqueness_scores, similarity_matrix, distinctive_features, biobanks = uniqueness_data
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Translational Research Profile', 'Research Uniqueness Scores',
                       'Biobank Similarity Heatmap', 'Clinical Impact vs Basic Research',
                       'Research Coverage & Uniqueness', 'Translational Readiness'),
        specs=[[{'type': 'radar'}, {'type': 'bar'}],
               [{'type': 'heatmap'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Research Profile Radar Chart
    categories = ['Basic Research', 'Clinical Research', 'Translational', 'Population Health']
    
    for i, biobank in enumerate(translational_data.keys()):
        data = translational_data[biobank]
        values = [
            data['basic_research_focus'],
            data['clinical_research_focus'], 
            data['translational_focus'],
            data['population_health_focus']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=biobank,
            opacity=0.6
        ), row=1, col=1)
    
    # 2. Uniqueness Scores
    biobank_names = list(uniqueness_scores.keys())
    uniqueness_values = list(uniqueness_scores.values())
    
    fig.add_trace(go.Bar(
        x=biobank_names,
        y=uniqueness_values,
        name='Uniqueness Score',
        marker_color=px.colors.qualitative.Set1[:len(biobank_names)]
    ), row=1, col=2)
    
    # 3. Similarity Heatmap
    fig.add_trace(go.Heatmap(
        z=similarity_matrix,
        x=biobanks,
        y=biobanks,
        colorscale='RdYlBu_r',
        name='Similarity Matrix'
    ), row=2, col=1)
    
    # 4. Clinical Impact vs Basic Research
    clinical_rates = [translational_data[b]['clinical_impact_rate'] * 100 for b in biobank_names]
    basic_focus = [translational_data[b]['basic_research_focus'] for b in biobank_names]
    
    fig.add_trace(go.Scatter(
        x=basic_focus,
        y=clinical_rates,
        mode='markers+text',
        text=biobank_names,
        textposition='top center',
        marker=dict(size=20, opacity=0.7),
        name='Clinical vs Basic'
    ), row=2, col=2)
    
    # 5. Coverage vs Uniqueness
    coverage_ratios = [gaps_data[b]['coverage_ratio'] * 100 for b in biobank_names]
    uniqueness_ratios = [gaps_data[b]['uniqueness_ratio'] * 100 for b in biobank_names]
    
    fig.add_trace(go.Scatter(
        x=coverage_ratios,
        y=uniqueness_ratios,
        mode='markers+text',
        text=biobank_names,
        textposition='top center',
        marker=dict(size=20, opacity=0.7),
        name='Coverage vs Uniqueness'
    ), row=3, col=1)
    
    # 6. Translational Readiness
    readiness_scores = [translational_data[b]['translational_readiness'] for b in biobank_names]
    
    fig.add_trace(go.Bar(
        x=biobank_names,
        y=readiness_scores,
        name='Translational Readiness',
        marker_color=px.colors.qualitative.Set2[:len(biobank_names)]
    ), row=3, col=2)
    
    # Update layout
    fig.update_layout(height=1400, showlegend=True,
                     title_text="Translational Research & Competitive Landscape Analysis")
    
    # Save visualization
    output_file = os.path.join(analysis_dir, 'translational_competitive_analysis.html')
    fig.write_html(output_file)
    print(f"✅ Visualization saved: {output_file}")
    
    return fig

def generate_competitive_report(translational_data, uniqueness_data, gaps_data):
    """Generate comprehensive competitive analysis report"""
    
    uniqueness_scores, _, distinctive_features, _ = uniqueness_data
    
    report = """
TRANSLATIONAL RESEARCH & COMPETITIVE LANDSCAPE ANALYSIS
======================================================

TRANSLATIONAL READINESS RANKING:
"""
    
    # Rank by translational readiness
    readiness_ranking = sorted(translational_data.items(), 
                             key=lambda x: x[1]['translational_readiness'], reverse=True)
    
    for biobank, data in readiness_ranking:
        report += f"""
{biobank}:
  - Translational Readiness: {data['translational_readiness']:.2f}
  - Clinical Impact Rate: {data['clinical_impact_rate']*100:.1f}%
  - Basic Research Focus: {data['basic_research_focus']:.2f}
  - Clinical Research Focus: {data['clinical_research_focus']:.2f}
  - Population Health Focus: {data['population_health_focus']:.2f}
"""
    
    report += "\nRESEARCH UNIQUENESS & POSITIONING:\n"
    
    uniqueness_ranking = sorted(uniqueness_scores.items(), key=lambda x: x[1], reverse=True)
    
    for biobank, score in uniqueness_ranking:
        gaps = gaps_data[biobank]
        distinctive = distinctive_features[biobank][:3]  # Top 3
        
        report += f"""
{biobank}:
  - Uniqueness Score: {score:.3f}
  - Research Coverage: {gaps['coverage_ratio']*100:.1f}% of all terms
  - Unique Research Ratio: {gaps['uniqueness_ratio']*100:.1f}%
  - Distinctive Areas: {', '.join([feat[0] for feat in distinctive])}
"""
    
    report += "\nSTRATEGIC INSIGHTS:\n"
    
    # Most unique biobank
    most_unique = max(uniqueness_scores.items(), key=lambda x: x[1])
    
    # Most clinically ready
    most_clinical = max(translational_data.items(), 
                       key=lambda x: x[1]['clinical_impact_rate'])
    
    # Most collaborative (highest similarity to others)
    least_unique = min(uniqueness_scores.items(), key=lambda x: x[1])
    
    report += f"""
- Most Unique Research Portfolio: {most_unique[0]} (Score: {most_unique[1]:.3f})
- Highest Clinical Impact: {most_clinical[0]} ({most_clinical[1]['clinical_impact_rate']*100:.1f}% clinical papers)
- Most Collaborative Potential: {least_unique[0]} (overlaps well with others)
- Best Translational Balance: {readiness_ranking[0][0]} (Readiness: {readiness_ranking[0][1]['translational_readiness']:.2f})

COLLABORATION OPPORTUNITIES:
- Cross-biobank gaps exist in: {len(set().union(*[gaps_data[b]['missing_terms'] for b in gaps_data]))} unique research areas
- Potential synergies through complementary research portfolios
- Standardization opportunities in overlapping research themes
"""
    
    return report

def run_competitive_analysis(df):
    """Main function to run translational and competitive analysis"""
    
    print("🔬 Analyzing translational research potential...")
    translational_data = analyze_translational_potential(df)
    
    print("🎯 Calculating research uniqueness and positioning...")
    uniqueness_data = analyze_research_uniqueness(df)
    
    print("🔍 Identifying research gaps and opportunities...")
    gaps_data = analyze_research_gaps_and_opportunities(df)
    
    print("📊 Creating competitive landscape visualizations...")
    fig = create_competitive_visualizations(translational_data, uniqueness_data, gaps_data)
    
    print("📝 Generating competitive analysis report...")
    report = generate_competitive_report(translational_data, uniqueness_data, gaps_data)
    
    # Save report
    report_file = os.path.join(analysis_dir, 'translational_competitive_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✅ Report saved: {report_file}")
    
    # Save translational metrics to CSV
    translational_df = pd.DataFrame(translational_data).T
    translational_file = os.path.join(analysis_dir, 'translational_metrics.csv')
    translational_df.to_csv(translational_file)
    print(f"✅ Translational metrics saved: {translational_file}")
    
    # Save uniqueness scores to CSV
    uniqueness_scores, similarity_matrix, distinctive_features, biobanks = uniqueness_data
    uniqueness_df = pd.DataFrame([
        {'biobank': biobank, 'uniqueness_score': score}
        for biobank, score in uniqueness_scores.items()
    ])
    uniqueness_file = os.path.join(analysis_dir, 'uniqueness_scores.csv')
    uniqueness_df.to_csv(uniqueness_file, index=False)
    print(f"✅ Uniqueness scores saved: {uniqueness_file}")
    
    # Save gaps analysis to CSV
    gaps_df = pd.DataFrame(gaps_data).T
    gaps_file = os.path.join(analysis_dir, 'research_gaps_analysis.csv')
    gaps_df.to_csv(gaps_file)
    print(f"✅ Gaps analysis saved: {gaps_file}")
    
    return {
        'translational_metrics': translational_data,
        'uniqueness_analysis': uniqueness_data,
        'gaps_analysis': gaps_data,
        'visualization': fig,
        'report': report
    }

if __name__ == "__main__":
    # Load and filter data
    df = pd.read_csv(os.path.join(data_dir, 'biobank_research_data.csv'))
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    df = df.dropna(subset=['MeSH_Terms'])
    
    print(f"📊 Loaded {len(df):,} publications for translational competitive analysis")
    
    # Run analysis
    results = run_competitive_analysis(df)
    
    # Display results
    print(results['report'])
    results['visualization'].show()
    
    print(f"\n🎯 Analysis complete! Results saved to: {analysis_dir}")