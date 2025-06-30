import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
input_file = os.path.join(parent_dir, "ANALYSIS", "00-01-LITERATURE-ANALYSIS", "00-01-biobank_statistics.txt")
output_dir = os.path.join(parent_dir, "ANALYSIS", "00-02-VISUALIZATIONS")
os.makedirs(output_dir, exist_ok=True)

# Configure plot styling
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
COLORS = sns.color_palette("viridis", 10)

# Function to parse the biobank statistics file
def parse_biobank_statistics(file_path):
    """Parse the statistics file to extract structured data"""
    print(f"Parsing statistics file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract total paper counts
    total_papers_match = re.search(r'Total papers analyzed: (\d+)', content)
    total_papers = int(total_papers_match.group(1)) if total_papers_match else None
    
    # Extract paper counts by biobank
    papers_by_biobank = {}
    biobank_papers_section = re.search(r'Papers by biobank:(.*?)(?=\n\n)', content, re.DOTALL)
    if biobank_papers_section:
        for line in biobank_papers_section.group(1).strip().split('\n'):
            match = re.search(r'\s*-\s*(.*?):\s*(\d+)\s*papers', line)
            if match:
                biobank, count = match.groups()
                papers_by_biobank[biobank.strip()] = int(count)
    
    # Extract statistics for each biobank
    biobank_stats = {}
    biobank_sections = re.findall(r'==================================================\n(.*?) STATISTICS\n==================================================\n\n(.*?)(?=\n\n==================================================|$)', content, re.DOTALL)
    
    for biobank_name, section in biobank_sections:
        biobank_name = biobank_name.strip()
        biobank_stats[biobank_name] = {}
        
        # Extract MeSH terms
        mesh_terms = {}
        mesh_section = re.search(r'1\. MOST COMMON MeSH TERMS\n----------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if mesh_section:
            for line in mesh_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*occurrences', line)
                if match:
                    term, count = match.groups()
                    mesh_terms[term.strip()] = int(count)
        biobank_stats[biobank_name]['mesh_terms'] = mesh_terms
        
        # Extract keywords
        keywords = {}
        keywords_section = re.search(r'2\. MOST COMMON AUTHOR KEYWORDS\n----------------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if keywords_section:
            for line in keywords_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*occurrences', line)
                if match:
                    keyword, count = match.groups()
                    keywords[keyword.strip()] = int(count)
        biobank_stats[biobank_name]['keywords'] = keywords
        
        # Extract authors
        authors = {}
        authors_section = re.search(r'3\. MOST PROLIFIC AUTHORS\n-----------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if authors_section:
            for line in authors_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*publications', line)
                if match:
                    author, count = match.groups()
                    authors[author.strip()] = int(count)
        biobank_stats[biobank_name]['authors'] = authors
        
        # Extract institutions
        institutions = {}
        institutions_section = re.search(r'4\. LEADING INSTITUTIONS\n----------------------\n(.*?)(?=\n\n|$)', section, re.DOTALL)
        if institutions_section:
            for line in institutions_section.group(1).strip().split('\n'):
                match = re.search(r'\s*\d+\.\s*(.*?):\s*(\d+)\s*publications', line)
                if match:
                    institution, count = match.groups()
                    institutions[institution.strip()] = int(count)
        biobank_stats[biobank_name]['institutions'] = institutions
    
    return {
        'total_papers': total_papers,
        'papers_by_biobank': papers_by_biobank,
        'biobank_stats': biobank_stats
    }

# Function to visualize publication counts by biobank
def visualize_publication_counts(data):
    """Create a bar chart of publication counts by biobank"""
    print("Creating publication counts visualization...")
    
    biobank_counts = pd.DataFrame(list(data['papers_by_biobank'].items()), 
                                 columns=['Biobank', 'Publication Count'])
    biobank_counts = biobank_counts.sort_values('Publication Count', ascending=False)
    
    # Save to CSV
    biobank_counts.to_csv(os.path.join(output_dir, "biobank_publication_counts.csv"), index=False)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=biobank_counts, x='Biobank', y='Publication Count', palette='viridis')
    
    # Add data labels
    for i, count in enumerate(biobank_counts['Publication Count']):
        ax.text(i, count + 200, f"{count:,}", ha='center')
    
    plt.title('Number of Publications by Biobank', fontsize=16)
    plt.ylabel('Publication Count', fontsize=14)
    plt.xlabel('Biobank', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "biobank_publication_counts.png"), dpi=300)
    plt.close()

# Function to analyze and visualize common MeSH terms
def analyze_mesh_terms(data):
    """Analyze and visualize the common MeSH terms across biobanks"""
    print("Analyzing MeSH terms...")
    
    # Create DataFrame of mesh terms
    mesh_rows = []
    for biobank, stats in data['biobank_stats'].items():
        for term, count in stats['mesh_terms'].items():
            mesh_rows.append({
                'Biobank': biobank,
                'MeSH Term': term,
                'Count': count
            })
    
    mesh_df = pd.DataFrame(mesh_rows)
    mesh_df.to_csv(os.path.join(output_dir, "mesh_terms_by_biobank.csv"), index=False)
    
    # Create a pivot table for top 15 mesh terms overall
    pivot_df = mesh_df.pivot_table(
        index='MeSH Term',
        columns='Biobank',
        values='Count',
        fill_value=0
    )
    
    top_terms = mesh_df.groupby('MeSH Term')['Count'].sum().nlargest(15).index
    top_terms_pivot = pivot_df.loc[pivot_df.index.isin(top_terms)]
    
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        top_terms_pivot,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        linewidths=0.5
    )
    plt.title("Top 15 MeSH Terms Across Biobanks", fontsize=16)
    plt.ylabel("MeSH Term", fontsize=14)
    plt.xlabel("Biobank", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_mesh_terms_heatmap.png"), dpi=300)
    plt.close()

    # Create horizontal bar chart of top 10 MeSH terms for each biobank
    for biobank, stats in data['biobank_stats'].items():
        if not stats['mesh_terms']:
            continue
            
        top_n = 10
        terms = [term for term, _ in sorted(stats['mesh_terms'].items(), 
                                            key=lambda x: x[1], reverse=True)[:top_n]]
        counts = [count for _, count in sorted(stats['mesh_terms'].items(), 
                                              key=lambda x: x[1], reverse=True)[:top_n]]
        
        plt.figure(figsize=(10, 8))
        plt.barh(terms[::-1], counts[::-1], color=COLORS)
        plt.title(f'Top {top_n} MeSH Terms for {biobank}', fontsize=16)
        plt.xlabel('Occurrence Count', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{biobank.replace(' ', '_')}_top_mesh_terms.png"), dpi=300)
        plt.close()

# Function to analyze author keywords
def analyze_keywords(data):
    """Analyze and visualize author keywords"""
    print("Analyzing author keywords...")
    
    # Create DataFrame of keywords
    keyword_rows = []
    for biobank, stats in data['biobank_stats'].items():
        for keyword, count in stats['keywords'].items():
            keyword_rows.append({
                'Biobank': biobank,
                'Keyword': keyword,
                'Count': count
            })
    
    keyword_df = pd.DataFrame(keyword_rows)
    keyword_df.to_csv(os.path.join(output_dir, "keywords_by_biobank.csv"), index=False)
    
    # Create a pivot table for visualization
    pivot_df = keyword_df.pivot_table(
        index='Keyword',
        columns='Biobank',
        values='Count',
        fill_value=0
    )
    
    # Get top 15 keywords overall
    top_keywords = keyword_df.groupby('Keyword')['Count'].sum().nlargest(15).index
    top_keywords_pivot = pivot_df.loc[pivot_df.index.isin(top_keywords)]
    
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        top_keywords_pivot,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        linewidths=0.5
    )
    plt.title("Top 15 Author Keywords Across Biobanks", fontsize=16)
    plt.ylabel("Keyword", fontsize=14)
    plt.xlabel("Biobank", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_keywords_heatmap.png"), dpi=300)
    plt.close()
    
    # Thematic analysis of keywords
    themes = {
        'Genetics': ['genetic', 'genetics', 'genome', 'gwas', 'genom', 'mendelian', 'polygenic', 'randomization'],
        'Clinical': ['disease', 'health', 'clinical', 'treatment', 'patient', 'care', 'medical', 'diagnosis', 'therapy', 'medicine'],
        'Data Science': ['data', 'machine learning', 'computational', 'algorithm', 'artificial intelligence', 'ai', 'informatics', 'big data'],
        'Epidemiology': ['epidemiology', 'cohort', 'population', 'risk', 'prevalence', 'incidence', 'mortality'],
        'Methodology': ['method', 'approach', 'framework', 'protocol', 'technique', 'procedure', 'design', 'analysis'],
        'Diversity & Equity': ['diversity', 'equity', 'inclusion', 'disparities', 'ethnicity', 'race', 'minority', 'underrepresented'],
        'Informatics': ['informatics', 'database', 'algorithm', 'ehr', 'electronic health record', 'database'],
        'Disease Focus': ['cancer', 'diabetes', 'cardiovascular', 'neurological', 'psychiatric', 'obesity', 'hypertension']
    }
    
    def classify_keyword(keyword):
        keyword_lower = keyword.lower()
        for theme, keywords in themes.items():
            for k in keywords:
                if k in keyword_lower:
                    return theme
        return 'Other'
    
    theme_rows = []
    for biobank, stats in data['biobank_stats'].items():
        theme_counts = Counter()
        for keyword, count in stats['keywords'].items():
            theme = classify_keyword(keyword)
            theme_counts[theme] += count
        
        for theme, count in theme_counts.items():
            theme_rows.append({
                'Biobank': biobank,
                'Theme': theme,
                'Count': count
            })
    
    theme_df = pd.DataFrame(theme_rows)
    theme_df.to_csv(os.path.join(output_dir, "keyword_themes_by_biobank.csv"), index=False)
    
    # Create thematic heatmap
    pivot_themes = theme_df.pivot_table(
        index='Theme',
        columns='Biobank',
        values='Count',
        fill_value=0
    )
    
    # Normalize by column totals
    pivot_themes_norm = pivot_themes.div(pivot_themes.sum(axis=0), axis=1) * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_themes_norm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5
    )
    plt.title("Thematic Focus by Biobank (% of Keywords)", fontsize=16)
    plt.ylabel("Theme", fontsize=14)
    plt.xlabel("Biobank", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thematic_focus_heatmap.png"), dpi=300)
    plt.close()

# Function to analyze prolific authors
def analyze_authors(data):
    """Analyze and visualize prolific authors"""
    print("Analyzing prolific authors...")
    
    # Create DataFrame of authors
    author_rows = []
    for biobank, stats in data['biobank_stats'].items():
        for author, count in stats['authors'].items():
            author_rows.append({
                'Biobank': biobank,
                'Author': author,
                'Publications': count
            })
    
    author_df = pd.DataFrame(author_rows)
    author_df.to_csv(os.path.join(output_dir, "top_authors_by_biobank.csv"), index=False)
    
    # Create visualization of top 20 authors across all biobanks
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=author_df.sort_values('Publications', ascending=False).head(20),
        x='Publications',
        y='Author',
        hue='Biobank',
        palette='viridis'
    )
    plt.title("Top 20 Most Prolific Authors Across Biobanks", fontsize=16)
    plt.xlabel("Number of Publications", fontsize=14)
    plt.ylabel("Author", fontsize=14)
    plt.legend(title="Biobank", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_authors_overall.png"), dpi=300)
    plt.close()
    
    # Identify authors who publish across multiple biobanks
    author_biobank_matrix = pd.pivot_table(
        author_df,
        index='Author',
        columns='Biobank',
        values='Publications',
        fill_value=0
    )
    
    # Count number of biobanks each author publishes in
    author_biobank_counts = (author_biobank_matrix > 0).sum(axis=1)
    multi_biobank_authors = author_biobank_counts[author_biobank_counts > 1].index.tolist()
    
    # Create a summary of multi-biobank authors
    multi_biobank_summary = author_biobank_matrix.loc[multi_biobank_authors].copy()
    multi_biobank_summary['Total_Biobanks'] = author_biobank_counts[multi_biobank_authors]
    multi_biobank_summary['Total_Publications'] = author_biobank_matrix.loc[multi_biobank_authors].sum(axis=1)
    multi_biobank_summary = multi_biobank_summary.sort_values(['Total_Biobanks', 'Total_Publications'], ascending=False)
    
    multi_biobank_summary.to_csv(os.path.join(output_dir, "multi_biobank_authors.csv"))
    
    # Visualize top 20 cross-biobank authors
    top_multi_biobank = multi_biobank_summary.head(20)
    
    plt.figure(figsize=(12, 10))
    top_multi_biobank.drop(['Total_Biobanks', 'Total_Publications'], axis=1).plot(
        kind='barh',
        stacked=True,
        figsize=(12, 10),
        colormap='viridis'
    )
    plt.title("Publications by Biobank for Top 20 Cross-Biobank Authors", fontsize=16)
    plt.xlabel("Number of Publications", fontsize=14)
    plt.ylabel("Author", fontsize=14)
    plt.legend(title="Biobank", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_biobank_authors.png"), dpi=300)
    plt.close()

# Function to analyze institutions
def analyze_institutions(data):
    """Analyze and visualize leading institutions"""
    print("Analyzing institutions...")
    
    # Create DataFrame of institutions
    institution_rows = []
    for biobank, stats in data['biobank_stats'].items():
        for institution, count in stats['institutions'].items():
            institution_rows.append({
                'Biobank': biobank,
                'Institution': institution,
                'Publications': count
            })
    
    institution_df = pd.DataFrame(institution_rows)
    institution_df.to_csv(os.path.join(output_dir, "institutions_by_biobank.csv"), index=False)
    
    # Group by institution type
    institution_df['Type'] = institution_df['Institution'].str.split().str[0:2].str.join(' ')
    institution_type_counts = institution_df.groupby('Type')['Publications'].sum().reset_index()
    institution_type_counts = institution_type_counts.sort_values('Publications', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=institution_type_counts,
        x='Publications',
        y='Type',
        palette='viridis'
    )
    plt.title("Top 10 Institution Types by Total Publications", fontsize=16)
    plt.xlabel("Number of Publications", fontsize=14)
    plt.ylabel("Institution Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "institution_types.png"), dpi=300)
    plt.close()
    
    # Create bar chart of top 10 institutions overall
    top_institutions = institution_df.groupby('Institution')['Publications'].sum().nlargest(10).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_institutions,
        x='Publications',
        y='Institution',
        palette='viridis'
    )
    plt.title("Top 10 Institutions by Total Publications", fontsize=16)
    plt.xlabel("Number of Publications", fontsize=14)
    plt.ylabel("Institution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_institutions_overall.png"), dpi=300)
    plt.close()
    
    # Analyze geographical distribution based on institution names
    def extract_country(institution):
        institution = institution.lower()
        countries = {
            'usa': ['university of california', 'harvard', 'stanford', 'mit', 'yale', 'boston', 'johns hopkins', 
                    'vanderbilt', 'mayo', 'penn', 'brigham', 'michigan', 'washington', 'north carolina', 'emory'],
            'uk': ['cambridge', 'oxford', 'london', 'edinburgh', 'glasgow', 'imperial', 'manchester', 'king\'s college'],
            'china': ['fudan', 'peking', 'tsinghua', 'shanghai', 'china', 'chinese', 'beijing', 'central south', 
                     'sichuan', 'xiangya', 'zhejiang', 'tongji'],
            'finland': ['helsinki', 'finland', 'finnish', 'oulu', 'turku', 'tampere'],
            'estonia': ['tartu', 'estonia', 'estonian', 'tallinn'],
            'sweden': ['karolinska', 'uppsala', 'lund', 'sweden', 'swedish'],
            'canada': ['toronto', 'mcgill', 'montreal', 'british columbia', 'canada', 'canadian']
        }
        
        for country, indicators in countries.items():
            if any(indicator in institution for indicator in indicators):
                return country
        return 'other'
    
    institution_df['Country'] = institution_df['Institution'].apply(extract_country)
    country_counts = institution_df.groupby('Country')['Publications'].sum().reset_index()
    country_counts = country_counts.sort_values('Publications', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=country_counts,
        x='Publications',
        y='Country',
        palette='viridis'
    )
    plt.title("Geographical Distribution of Publications by Institution Country", fontsize=16)
    plt.xlabel("Number of Publications", fontsize=14)
    plt.ylabel("Country", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geographical_distribution.png"), dpi=300)
    plt.close()

# Function to analyze distinctive keywords
def analyze_distinctive_keywords(data):
    """Analyze and visualize distinctive keywords for each biobank"""
    print("Analyzing distinctive keywords...")
    
    # Create DataFrame of all keywords
    keyword_rows = []
    for biobank, stats in data['biobank_stats'].items():
        for keyword, count in stats['keywords'].items():
            keyword_rows.append({
                'Biobank': biobank,
                'Keyword': keyword,
                'Count': count
            })
    
    keyword_df = pd.DataFrame(keyword_rows)
    
    # Create a matrix of keyword percentages
    keyword_matrix = pd.pivot_table(
        keyword_df,
        index='Keyword',
        columns='Biobank',
        values='Count',
        fill_value=0
    )
    
    # Calculate the percentage of each keyword within each biobank
    keyword_percentages = keyword_matrix.div(keyword_matrix.sum(axis=0), axis=1) * 100
    
    # Find distinctive keywords
    distinctive_keywords = []
    for biobank in keyword_percentages.columns:
        other_biobanks = [b for b in keyword_percentages.columns if b != biobank]
        
        for keyword in keyword_percentages.index:
            biobank_pct = keyword_percentages.loc[keyword, biobank]
            other_pct_mean = keyword_percentages.loc[keyword, other_biobanks].mean()
            
            if biobank_pct > 0 and biobank_pct > (other_pct_mean * 2) and keyword_matrix.loc[keyword, biobank] > 10:
                distinctive_keywords.append({
                    'Biobank': biobank,
                    'Keyword': keyword,
                    'Count': keyword_matrix.loc[keyword, biobank],
                    'Percentage': biobank_pct,
                    'Other_Biobanks_Mean_Pct': other_pct_mean,
                    'Distinctiveness': biobank_pct / (other_pct_mean + 0.01)  # Add small value to avoid division by zero
                })
    
    distinctive_df = pd.DataFrame(distinctive_keywords)
    distinctive_df = distinctive_df.sort_values(['Biobank', 'Distinctiveness'], ascending=[True, False])
    distinctive_df.to_csv(os.path.join(output_dir, "distinctive_keywords.csv"), index=False)
    
    # Visualize top 5 distinctive keywords for each biobank
    top_distinctive = distinctive_df.groupby('Biobank').head(5)
    
    # Create separate bar charts for each biobank's distinctive keywords
    for biobank in top_distinctive['Biobank'].unique():
        biobank_data = top_distinctive[top_distinctive['Biobank'] == biobank]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=biobank_data,
            x='Distinctiveness',
            y='Keyword',
            palette='viridis'
        )
        plt.title(f"Most Distinctive Keywords for {biobank}", fontsize=16)
        plt.xlabel("Distinctiveness Score", fontsize=14)
        plt.ylabel("Keyword", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{biobank.replace(' ', '_')}_distinctive_keywords.png"), dpi=300)
        plt.close()

    # Create a faceted bar chart
    g = sns.FacetGrid(
        top_distinctive, 
        col='Biobank', 
        col_wrap=3, 
        height=4, 
        aspect=1.5,
        sharey=False
    )
    g.map_dataframe(sns.barplot, x='Distinctiveness', y='Keyword', palette='viridis')
    g.set_axis_labels("Distinctiveness Score", "Keyword")
    g.set_titles("{col_name}")
    g.fig.suptitle("Most Distinctive Keywords by Biobank", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_distinctive_keywords.png"), dpi=300)
    plt.close()

# Main execution flow
if __name__ == "__main__":
    print(f"Starting visualization script...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please run the analysis script first to generate the statistics file.")
        exit(1)
    
    # Parse the statistics file
    data = parse_biobank_statistics(input_file)
    print(f"Successfully parsed data for {len(data['biobank_stats'])} biobanks")
    
    # Create visualizations
    visualize_publication_counts(data)
    analyze_mesh_terms(data)
    analyze_keywords(data)
    analyze_authors(data)
    analyze_institutions(data)
    analyze_distinctive_keywords(data)
    
    print(f"✅ Visualization complete. Results saved to: {output_dir}")