"""
Comprehensive Research Footprint Dashboard
Integrates all analyses into a unified evaluation framework

SCRIPT: PYTHON/01-04-comprehensive-research-footprint-dashboard.py
OUTPUT: ANALYSIS/01-04-COMPREHENSIVE-FOOTPRINT/
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-04-COMPREHENSIVE-FOOTPRINT")
os.makedirs(analysis_dir, exist_ok=True)

class BiobankFootprintAnalyzer:
    """Comprehensive analyzer for biobank research footprints"""
    
    def __init__(self, df):
        self.df = df
        self.biobanks = df['Biobank'].unique()
        self.metrics = {}
        self.scores = {}
        
    def calculate_comprehensive_metrics(self):
        """Calculate all key metrics for footprint evaluation"""
        
        print("📊 Calculating comprehensive footprint metrics...")
        
        for biobank in self.biobanks:
            biobank_data = self.df[self.df['Biobank'] == biobank]
            
            # Volume Metrics
            total_papers = len(biobank_data)
            yearly_distribution = biobank_data['Year'].value_counts().sort_index()
            recent_papers = len(biobank_data[biobank_data['Year'] >= 2020])
            
            # Growth Metrics
            if len(yearly_distribution) > 2:
                years = yearly_distribution.index.values
                counts = yearly_distribution.values
                from scipy import stats
                growth_slope, _, growth_r2, _, _ = stats.linregress(years, counts)
            else:
                growth_slope, growth_r2 = 0, 0
            
            # Diversity Metrics
            unique_journals = biobank_data['Journal'].nunique()
            unique_mesh_terms = len(set([term.strip() for mesh_string in biobank_data['MeSH_Terms'].dropna() 
                                       for term in str(mesh_string).split(';')]))
            
            # Quality Indicators
            methodology_terms = ['randomized', 'systematic', 'meta-analysis', 'genome-wide']
            methodology_papers = sum(1 for title in biobank_data['Title'].dropna() 
                                   if any(term in str(title).lower() for term in methodology_terms))
            
            # International Collaboration
            intl_terms = ['international', 'consortium', 'collaboration', 'global']
            intl_papers = sum(1 for title in biobank_data['Title'].dropna() 
                            if any(term in str(title).lower() for term in intl_terms))
            
            # Innovation Indicators
            innovation_terms = ['novel', 'first', 'new', 'innovative', 'breakthrough']
            innovation_papers = sum(1 for title in biobank_data['Title'].dropna() 
                                  if any(term in str(title).lower() for term in innovation_terms))
            
            # Clinical Translation
            clinical_terms = ['clinical', 'patient', 'treatment', 'therapy', 'diagnosis']
            clinical_papers = sum(1 for title in biobank_data['Title'].dropna() 
                                if any(term in str(title).lower() for term in clinical_terms))
            
            self.metrics[biobank] = {
                # Volume & Scale
                'total_papers': total_papers,
                'recent_activity': recent_papers / total_papers if total_papers > 0 else 0,
                'publication_span': yearly_distribution.index.max() - yearly_distribution.index.min() + 1,
                
                # Growth & Momentum
                'growth_rate': growth_slope,
                'growth_consistency': growth_r2,
                'annual_average': total_papers / max(yearly_distribution.index.max() - yearly_distribution.index.min() + 1, 1),
                
                # Diversity & Breadth
                'journal_diversity': unique_journals,
                'research_breadth': unique_mesh_terms,
                'journal_diversity_ratio': unique_journals / total_papers if total_papers > 0 else 0,
                
                # Quality & Rigor
                'methodology_rate': methodology_papers / total_papers if total_papers > 0 else 0,
                'international_rate': intl_papers / total_papers if total_papers > 0 else 0,
                'innovation_rate': innovation_papers / total_papers if total_papers > 0 else 0,
                
                # Impact & Translation
                'clinical_translation_rate': clinical_papers / total_papers if total_papers > 0 else 0,
                'avg_title_length': np.mean([len(str(title)) for title in biobank_data['Title'].dropna()]),
                
                # Temporal Dynamics
                'recent_momentum': recent_papers / 5,  # Papers per recent year
                'research_maturity': (yearly_distribution.index.max() - yearly_distribution.index.min()) / (2024 - yearly_distribution.index.min()) if yearly_distribution.index.min() < 2024 else 0
            }
    
    def calculate_footprint_scores(self):
        """Calculate normalized footprint scores across dimensions"""
        
        print("🎯 Calculating footprint dimension scores...")
        
        # Define scoring dimensions
        dimensions = {
            'Volume & Scale': ['total_papers', 'publication_span', 'annual_average'],
            'Growth & Momentum': ['growth_rate', 'growth_consistency', 'recent_momentum'],
            'Diversity & Breadth': ['journal_diversity', 'research_breadth', 'journal_diversity_ratio'],
            'Quality & Rigor': ['methodology_rate', 'international_rate', 'innovation_rate'],
            'Impact & Translation': ['clinical_translation_rate', 'recent_activity'],
            'Research Maturity': ['research_maturity', 'publication_span']
        }
        
        # Normalize metrics to 0-1 scale
        scaler = MinMaxScaler()
        
        # Create matrix of all metrics
        metric_names = list(next(iter(self.metrics.values())).keys())
        metric_matrix = np.array([[self.metrics[biobank][metric] for metric in metric_names] 
                                 for biobank in self.biobanks])
        
        # Handle missing/infinite values
        metric_matrix = np.nan_to_num(metric_matrix, nan=0, posinf=1, neginf=0)
        
        # Normalize
        normalized_matrix = scaler.fit_transform(metric_matrix)
        
        # Calculate dimension scores
        for i, biobank in enumerate(self.biobanks):
            biobank_scores = dict(zip(metric_names, normalized_matrix[i]))
            
            dimension_scores = {}
            for dimension, metrics_list in dimensions.items():
                scores = [biobank_scores.get(metric, 0) for metric in metrics_list if metric in biobank_scores]
                dimension_scores[dimension] = np.mean(scores) if scores else 0
            
            # Overall footprint score
            overall_score = np.mean(list(dimension_scores.values()))
            
            self.scores[biobank] = {
                'dimensions': dimension_scores,
                'overall': overall_score,
                'raw_metrics': self.metrics[biobank]
            }
    
    def create_footprint_dashboard(self):
        """Create comprehensive dashboard visualization"""
        
        print("📊 Creating footprint dashboard...")
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Overall Footprint Scores', 'Volume & Scale Dimension', 'Growth & Momentum',
                'Quality & Rigor', 'Diversity & Breadth', 'Impact & Translation',
                'Research Maturity', 'Footprint Radar Chart', 'Temporal Footprint Evolution'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'polar'}, {'type': 'scatter'}]
            ]
        )
        
        colors = px.colors.qualitative.Set1[:len(self.biobanks)]
        
        # 1. Overall Footprint Scores
        overall_scores = [self.scores[b]['overall'] for b in self.biobanks]
        fig.add_trace(go.Bar(
            x=self.biobanks, y=overall_scores,
            marker_color=colors,
            text=[f'{score:.3f}' for score in overall_scores],
            textposition='outside',
            name='Overall Score'
        ), row=1, col=1)
        
        # 2-7. Individual Dimension Scores
        dimensions = ['Volume & Scale', 'Growth & Momentum', 'Quality & Rigor', 
                     'Diversity & Breadth', 'Impact & Translation', 'Research Maturity']
        positions = [(1,2), (1,3), (2,1), (2,2), (2,3), (3,1)]
        
        for dim, (row, col) in zip(dimensions, positions):
            dim_scores = [self.scores[b]['dimensions'][dim] for b in self.biobanks]
            fig.add_trace(go.Bar(
                x=self.biobanks, y=dim_scores,
                marker_color=colors,
                text=[f'{score:.2f}' for score in dim_scores],
                textposition='outside',
                name=dim,
                showlegend=False
            ), row=row, col=col)
        
        # 8. Radar Chart
        for i, biobank in enumerate(self.biobanks):
            dim_scores = self.scores[biobank]['dimensions']
            values = list(dim_scores.values())
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=list(dim_scores.keys()) + [list(dim_scores.keys())[0]],
                fill='toself',
                name=biobank,
                opacity=0.6,
                line_color=colors[i]
            ), row=3, col=2)
        
        # 9. Temporal Evolution
        # Calculate yearly footprint scores
        yearly_footprints = defaultdict(lambda: defaultdict(float))
        
        for biobank in self.biobanks:
            biobank_data = self.df[self.df['Biobank'] == biobank]
            yearly_counts = biobank_data['Year'].value_counts().sort_index()
            
            # Simple temporal footprint: papers per year normalized by max
            max_papers = yearly_counts.max() if len(yearly_counts) > 0 else 1
            
            for year, count in yearly_counts.items():
                yearly_footprints[year][biobank] = count / max_papers
        
        for i, biobank in enumerate(self.biobanks):
            years = sorted([year for year in yearly_footprints.keys() 
                          if yearly_footprints[year][biobank] > 0])
            values = [yearly_footprints[year][biobank] for year in years]
            
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name=f'{biobank} Temporal',
                line=dict(color=colors[i], width=3),
                showlegend=False
            ), row=3, col=3)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Biobank Research Footprint Dashboard",
            showlegend=True
        )
        
        # Update polar subplot
        fig.update_polars(radialaxis=dict(range=[0, 1]))
        
        # Save visualization
        output_file = os.path.join(analysis_dir, 'comprehensive_footprint_dashboard.html')
        fig.write_html(output_file)
        print(f"✅ Dashboard saved: {output_file}")
        
        return fig
    
    def generate_footprint_report(self):
        """Generate comprehensive footprint evaluation report"""
        
        print("📝 Generating footprint report...")
        
        report = f"""
COMPREHENSIVE BIOBANK RESEARCH FOOTPRINT EVALUATION
==================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
"""
        
        # Rank biobanks by overall score
        ranked_biobanks = sorted(self.scores.items(), key=lambda x: x[1]['overall'], reverse=True)
        
        report += f"""
Overall Footprint Ranking:
"""
        for rank, (biobank, data) in enumerate(ranked_biobanks, 1):
            report += f"  {rank}. {biobank}: {data['overall']:.3f}\n"
        
        report += "\nDIMENSIONAL ANALYSIS:\n"
        
        # Find leaders in each dimension
        dimensions = ['Volume & Scale', 'Growth & Momentum', 'Quality & Rigor', 
                     'Diversity & Breadth', 'Impact & Translation', 'Research Maturity']
        
        for dimension in dimensions:
            leader = max(self.scores.items(), key=lambda x: x[1]['dimensions'][dimension])
            report += f"  {dimension} Leader: {leader[0]} ({leader[1]['dimensions'][dimension]:.3f})\n"
        
        report += "\nDETAILED PROFILES:\n"
        
        for biobank in self.biobanks:
            data = self.scores[biobank]
            metrics = data['raw_metrics']
            
            report += f"""
{biobank}:
  Overall Score: {data['overall']:.3f}
  
  Key Metrics:
    - Total Papers: {metrics['total_papers']:,}
    - Research Span: {metrics['publication_span']} years
    - Growth Rate: {metrics['growth_rate']:.2f} papers/year
    - Journal Diversity: {metrics['journal_diversity']} unique journals
    - Methodology Rate: {metrics['methodology_rate']*100:.1f}%
    - Clinical Translation: {metrics['clinical_translation_rate']*100:.1f}%
    - International Collaboration: {metrics['international_rate']*100:.1f}%
  
  Dimension Scores:
"""
            for dim, score in data['dimensions'].items():
                report += f"    - {dim}: {score:.3f}\n"
        
        # Strategic insights
        report += f"""
STRATEGIC INSIGHTS:

Research Volume Leader: {ranked_biobanks[0][0]}
  - Commands the largest research footprint with highest overall score
  - Demonstrates sustained research output and broad impact

Growth Champion: {max(self.scores.items(), key=lambda x: x[1]['dimensions']['Growth & Momentum'])[0]}
  - Shows strongest growth momentum and expansion trajectory
  - Positioned for continued footprint expansion

Quality Excellence: {max(self.scores.items(), key=lambda x: x[1]['dimensions']['Quality & Rigor'])[0]}
  - Maintains highest standards in methodology and collaboration
  - Sets benchmark for research rigor in biobank studies

Innovation Focus: {max(self.scores.items(), key=lambda x: x[1]['dimensions']['Impact & Translation'])[0]}
  - Leads in clinical translation and practical impact
  - Demonstrates strongest bench-to-bedside research pipeline

FOOTPRINT EVOLUTION RECOMMENDATIONS:
- Cross-biobank collaboration opportunities exist in complementary research areas
- Standardization of methodologies could enhance collective impact
- Knowledge transfer initiatives could accelerate global biobank research
- Specialized excellence areas should be leveraged for competitive advantage
"""
        
        return report
    
    def export_results(self):
        """Export comprehensive results to files"""
        
        # Ensure directory exists
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Export scores as JSON
        scores_file = os.path.join(analysis_dir, 'footprint_scores.json')
        with open(scores_file, 'w') as f:
            json.dump(self.scores, f, indent=2)
        print(f"✅ Scores saved: {scores_file}")
        
        # Export metrics as CSV
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_file = os.path.join(analysis_dir, 'footprint_metrics.csv')
        metrics_df.to_csv(metrics_file)
        print(f"✅ Metrics saved: {metrics_file}")
        
        # Export dimension scores as CSV
        dimension_data = []
        for biobank, data in self.scores.items():
            row = {'Biobank': biobank, 'Overall_Score': data['overall']}
            row.update(data['dimensions'])
            dimension_data.append(row)
        
        dimensions_df = pd.DataFrame(dimension_data)
        dimensions_file = os.path.join(analysis_dir, 'dimension_scores.csv')
        dimensions_df.to_csv(dimensions_file, index=False)
        print(f"✅ Dimension scores saved: {dimensions_file}")
        
        return analysis_dir

def run_comprehensive_footprint_analysis(df):
    """Main function to run comprehensive footprint analysis"""
    
    print("🚀 Starting comprehensive biobank footprint analysis...")
    
    # Initialize analyzer
    analyzer = BiobankFootprintAnalyzer(df)
    
    # Run all analyses
    analyzer.calculate_comprehensive_metrics()
    analyzer.calculate_footprint_scores()
    
    # Create visualizations
    dashboard_fig = analyzer.create_footprint_dashboard()
    
    # Generate report
    report = analyzer.generate_footprint_report()
    
    # Save report
    report_file = os.path.join(analysis_dir, 'comprehensive_footprint_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✅ Report saved: {report_file}")
    
    # Export results
    output_dir = analyzer.export_results()
    
    print("✅ Comprehensive footprint analysis complete!")
    
    return {
        'analyzer': analyzer,
        'dashboard': dashboard_fig,
        'report': report,
        'output_directory': output_dir
    }

if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv(os.path.join(data_dir, 'biobank_research_data.csv'))
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    df = df.dropna(subset=['Title', 'MeSH_Terms'])
    
    print(f"📊 Loaded {len(df):,} publications for comprehensive footprint analysis")
    
    # Run comprehensive analysis
    results = run_comprehensive_footprint_analysis(df)
    
    # Display results
    print(results['report'])
    results['dashboard'].show()
    
    print(f"\n🎯 Complete analysis results saved to: {results['output_directory']}")