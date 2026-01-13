"""
Temporal Research Evolution & Innovation Cycles Analysis
Tracks how research themes evolve over time and identifies innovation patterns

SCRIPT: PYTHON/01-00-temporal-research-evolution-&-innovation-cycles.py
OUTPUT: ANALYSIS/01-00-TEMPORAL-RESEARCH-EVOLUTION/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import networkx as nx
from collections import defaultdict

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-00-TEMPORAL-RESEARCH-EVOLUTION")
os.makedirs(analysis_dir, exist_ok=True)

def analyze_research_evolution(df, mesh_col='MeSH_Terms', year_col='Year', biobank_col='Biobank'):
    """
    Analyze how research themes evolve over time for each biobank
    """
    
    # 1. Temporal MeSH term emergence analysis
    def detect_emerging_terms(biobank_data, window_size=3):
        """Detect terms that are emerging vs declining"""
        yearly_terms = defaultdict(lambda: defaultdict(int))
        
        for idx, row in biobank_data.iterrows():
            year = row[year_col]
            if pd.notna(row[mesh_col]):
                terms = [t.strip().lower().replace(' ', '_') for t in str(row[mesh_col]).split(';')]
                for term in terms:
                    yearly_terms[year][term] += 1
        
        # Calculate term velocity (rate of growth/decline)
        term_velocities = {}
        years = sorted(yearly_terms.keys())
        
        for term in set().union(*[yearly_terms[y].keys() for y in years]):
            term_counts = [yearly_terms[y].get(term, 0) for y in years]
            if sum(term_counts) >= 5:  # Only terms with sufficient frequency
                # Calculate slope of usage over time
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, term_counts)
                term_velocities[term] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'total_usage': sum(term_counts),
                    'trend': 'emerging' if slope > 0.1 and p_value < 0.05 else 
                            'declining' if slope < -0.1 and p_value < 0.05 else 'stable'
                }
        
        return term_velocities
    
    # 2. Research theme similarity over time
    def calculate_temporal_similarity(biobank_data, window_size=2):
        """Calculate how similar research themes are between consecutive time periods"""
        yearly_docs = defaultdict(list)
        
        for idx, row in biobank_data.iterrows():
            year = row[year_col]
            if pd.notna(row[mesh_col]):
                yearly_docs[year].append(str(row[mesh_col]))
        
        # Create documents for each year
        years = sorted(yearly_docs.keys())
        year_documents = []
        year_labels = []
        
        for year in years:
            if len(yearly_docs[year]) >= 5:  # Minimum papers per year
                combined_doc = ' '.join(yearly_docs[year])
                year_documents.append(combined_doc)
                year_labels.append(year)
        
        if len(year_documents) < 2:
            return None, None
        
        # TF-IDF and similarity calculation
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(year_documents)
        
        # Calculate year-to-year similarity
        similarities = []
        for i in range(len(year_documents)-1):
            sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
            similarities.append(sim)
        
        return year_labels, similarities
    
    # 3. Innovation cycle detection
    def detect_innovation_cycles(term_velocities, threshold=0.2):
        """Identify periods of high innovation (many emerging terms)"""
        emerging_terms = [t for t, data in term_velocities.items() 
                         if data['trend'] == 'emerging' and data['r_squared'] > threshold]
        
        declining_terms = [t for t, data in term_velocities.items() 
                          if data['trend'] == 'declining' and data['r_squared'] > threshold]
        
        return {
            'emerging_count': len(emerging_terms),
            'declining_count': len(declining_terms),
            'innovation_ratio': len(emerging_terms) / max(len(declining_terms), 1),
            'emerging_terms': emerging_terms[:10],  # Top 10
            'declining_terms': declining_terms[:10]
        }
    
    results = {}
    
    for biobank in df[biobank_col].unique():
        biobank_data = df[df[biobank_col] == biobank].copy()
        
        # Analyze evolution
        term_velocities = detect_emerging_terms(biobank_data)
        year_labels, similarities = calculate_temporal_similarity(biobank_data)
        innovation_metrics = detect_innovation_cycles(term_velocities)
        
        results[biobank] = {
            'term_velocities': term_velocities,
            'temporal_similarities': (year_labels, similarities),
            'innovation_metrics': innovation_metrics
        }
    
    return results

def visualize_research_evolution(evolution_results):
    """Create comprehensive visualizations of research evolution"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    biobanks = list(evolution_results.keys())
    
    # 1. Emerging vs Declining Terms
    ax1 = axes[0, 0]
    emerging_counts = [evolution_results[b]['innovation_metrics']['emerging_count'] for b in biobanks]
    declining_counts = [evolution_results[b]['innovation_metrics']['declining_count'] for b in biobanks]
    
    x = np.arange(len(biobanks))
    width = 0.35
    
    ax1.bar(x - width/2, emerging_counts, width, label='Emerging Terms', color='green', alpha=0.7)
    ax1.bar(x + width/2, declining_counts, width, label='Declining Terms', color='red', alpha=0.7)
    ax1.set_xlabel('Biobank')
    ax1.set_ylabel('Number of Terms')
    ax1.set_title('Emerging vs Declining Research Terms')
    ax1.set_xticks(x)
    ax1.set_xticklabels(biobanks, rotation=45)
    ax1.legend()
    
    # 2. Innovation Ratios
    ax2 = axes[0, 1]
    innovation_ratios = [evolution_results[b]['innovation_metrics']['innovation_ratio'] for b in biobanks]
    colors = plt.cm.viridis(np.linspace(0, 1, len(biobanks)))
    
    bars = ax2.bar(biobanks, innovation_ratios, color=colors, alpha=0.7)
    ax2.set_ylabel('Innovation Ratio')
    ax2.set_title('Research Innovation Ratio\n(Emerging/Declining Terms)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, innovation_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom')
    
    # 3. Temporal Similarity Trends
    ax3 = axes[0, 2]
    for i, biobank in enumerate(biobanks):
        year_labels, similarities = evolution_results[biobank]['temporal_similarities']
        if similarities:
            transition_years = [f"{year_labels[j]}-{year_labels[j+1]}" for j in range(len(similarities))]
            ax3.plot(range(len(similarities)), similarities, 
                    marker='o', label=biobank, linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Time Transitions')
    ax3.set_ylabel('Thematic Similarity')
    ax3.set_title('Research Continuity Over Time')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Top Emerging Terms for each biobank (examples)
    for idx, biobank in enumerate(biobanks[:3]):  # Show first 3 biobanks
        ax = axes[1, idx]
        emerging_terms = evolution_results[biobank]['innovation_metrics']['emerging_terms'][:8]
        
        if emerging_terms:
            # Get slopes for these terms
            term_data = evolution_results[biobank]['term_velocities']
            slopes = [term_data[term]['slope'] for term in emerging_terms if term in term_data]
            
            if slopes:
                y_pos = np.arange(len(emerging_terms))
                bars = ax.barh(y_pos, slopes[:len(emerging_terms)], alpha=0.7, color='green')
                ax.set_yticks(y_pos)
                ax.set_yticklabels([term.replace('_', ' ').title()[:20] for term in emerging_terms])
                ax.set_xlabel('Growth Rate (slope)')
                ax.set_title(f'{biobank}\nTop Emerging Terms')
                ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(analysis_dir, 'research_evolution_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    return fig

# Example usage
def run_evolution_analysis(df):
    """Main function to run the temporal evolution analysis"""
    
    print("🔄 Analyzing research theme evolution over time...")
    evolution_results = analyze_research_evolution(df)
    
    print("📊 Creating evolution visualizations...")
    fig = visualize_research_evolution(evolution_results)
    
    # Print summary insights
    print("\n📈 RESEARCH EVOLUTION INSIGHTS:")
    for biobank, results in evolution_results.items():
        metrics = results['innovation_metrics']
        print(f"\n{biobank}:")
        print(f"  📈 Emerging terms: {metrics['emerging_count']}")
        print(f"  📉 Declining terms: {metrics['declining_count']}")
        print(f"  🔄 Innovation ratio: {metrics['innovation_ratio']:.2f}")
        
        if metrics['emerging_terms']:
            print(f"  🆕 Top emerging: {', '.join(metrics['emerging_terms'][:3])}")
    
    # Save results to CSV
    results_data = []
    for biobank, results in evolution_results.items():
        metrics = results['innovation_metrics']
        velocities = results['term_velocities']
        
        # Save emerging terms data
        for term in metrics['emerging_terms'][:10]:
            if term in velocities:
                results_data.append({
                    'biobank': biobank,
                    'term': term,
                    'trend_type': 'emerging',
                    'slope': velocities[term]['slope'],
                    'r_squared': velocities[term]['r_squared'],
                    'p_value': velocities[term]['p_value'],
                    'total_usage': velocities[term]['total_usage']
                })
        
        # Save declining terms data
        for term in metrics['declining_terms'][:10]:
            if term in velocities:
                results_data.append({
                    'biobank': biobank,
                    'term': term,
                    'trend_type': 'declining',
                    'slope': velocities[term]['slope'],
                    'r_squared': velocities[term]['r_squared'],
                    'p_value': velocities[term]['p_value'],
                    'total_usage': velocities[term]['total_usage']
                })
    
    # Save results
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_file = os.path.join(analysis_dir, 'temporal_evolution_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"✅ Results saved: {results_file}")
    
    # Save summary statistics
    summary_data = []
    for biobank, results in evolution_results.items():
        metrics = results['innovation_metrics']
        year_labels, similarities = results['temporal_similarities']
        
        summary_data.append({
            'biobank': biobank,
            'emerging_terms_count': metrics['emerging_count'],
            'declining_terms_count': metrics['declining_count'],
            'innovation_ratio': metrics['innovation_ratio'],
            'avg_temporal_similarity': np.mean(similarities) if similarities else 0,
            'temporal_transitions': len(similarities) if similarities else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(analysis_dir, 'evolution_summary_statistics.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary saved: {summary_file}")
    
    return evolution_results, fig

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(os.path.join(data_dir, 'biobank_research_data.csv'))
    
    # Apply same filtering as your main analysis
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    df = df.dropna(subset=['MeSH_Terms'])
    
    print(f"📊 Loaded {len(df):,} publications for temporal analysis")
    
    # Run analysis
    results, figure = run_evolution_analysis(df)
    
    print(f"\n🎯 Analysis complete! Results saved to: {analysis_dir}")
    plt.show()