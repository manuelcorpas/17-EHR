import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np
from collections import Counter

# Setup
input_file = "ANALYSIS/00-00-LITERATURE-ANALYSIS/00-00-biobank_statistics.txt"
output_dir = "ANALYSIS/00-01-LITERATURE-ANALYSIS"
os.makedirs(output_dir, exist_ok=True)

# Function to parse the statistics file
def parse_biobank_statistics(file_path):
    with open(file_path, 'r') as f:
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
    
    # Extract detailed statistics for each biobank
    biobank_stats = {}
    biobank_sections = re.findall(r'==================================================\n(.*?) STATISTICS\n==================================================\n\n(.*?)(?=\n\n==================================================|$)', content, re.DOTALL)
    
    for biobank_name, section in biobank_sections:
        biobank_name = biobank_name.strip()
        
        # Parse keywords
        keywords = {}
        keywords_section = re.search(r'1\. MOST COMMON KEYWORDS\n----------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if keywords_section:
            for line in keywords_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*occurrences', line)
                if match:
                    keyword, count = match.groups()
                    keywords[keyword.strip()] = int(count)
        
        # Parse cited papers
        cited_papers = []
        cited_papers_section = re.search(r'2\. MOST CITED PAPERS\n-------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if cited_papers_section:
            for line in cited_papers_section.group(1).strip().split('\n'):
                match = re.search(r'\s*\d+\.\s*PMID\s*(\d+):\s*(.*?)\s*\((\d+)\)\s*-\s*(\d+)\s*citations', line)
                if match:
                    pmid, title, year, citations = match.groups()
                    cited_papers.append({
                        'PMID': pmid,
                        'Title': title.strip(),
                        'Year': int(year),
                        'Citations': int(citations)
                    })
        
        # Parse prolific authors
        authors = {}
        authors_section = re.search(r'3\. MOST PROLIFIC AUTHORS\n-----------------------\n(.*?)(?=\n\n\d\.)', section, re.DOTALL)
        if authors_section:
            for line in authors_section.group(1).strip().split('\n'):
                match = re.search(r'\s*(.*?):\s*(\d+)\s*publications', line)
                if match:
                    author, count = match.groups()
                    authors[author.strip()] = int(count)
        
        # Parse leading institutions
        institutions = {}
        institutions_section = re.search(r'4\. LEADING INSTITUTIONS\n----------------------\n(.*?)(?=\n\n|$)', section, re.DOTALL)
        if institutions_section:
            for line in institutions_section.group(1).strip().split('\n'):
                match = re.search(r'\s*\d+\.\s*(.*?):\s*(\d+)\s*publications', line)
                if match:
                    institution, count = match.groups()
                    institutions[institution.strip()] = int(count)
        
        biobank_stats[biobank_name] = {
            'keywords': keywords,
            'cited_papers': cited_papers,
            'authors': authors,
            'institutions': institutions
        }
    
    return {
        'total_papers': total_papers,
        'papers_by_biobank': papers_by_biobank,
        'biobank_stats': biobank_stats
    }

# Load and parse the data
data = parse_biobank_statistics(input_file)

# ---------------------------------------------
# 1. Article counts per biobank
# ---------------------------------------------
biobank_counts = pd.DataFrame(list(data['papers_by_biobank'].items()), columns=['Biobank', 'Article Count'])
biobank_counts = biobank_counts.sort_values('Article Count', ascending=False)
biobank_counts.to_csv(os.path.join(output_dir, "01-biobank_article_counts.csv"), index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=biobank_counts, x='Biobank', y='Article Count', palette='viridis')
plt.title('Articles per Biobank')
plt.ylabel('Article Count')
plt.xlabel('Biobank')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-biobank_article_counts_plot.png"))
plt.close()

# ---------------------------------------------
# 2. Top keywords by biobank
# ---------------------------------------------
# Create a dataframe of the top 10 keywords for each biobank
keyword_rows = []
for biobank, stats in data['biobank_stats'].items():
    for keyword, count in list(stats['keywords'].items())[:10]:  # Top 10 keywords
        keyword_rows.append({
            'Biobank': biobank,
            'Keyword': keyword,
            'Count': count
        })

keyword_df = pd.DataFrame(keyword_rows)
keyword_df.to_csv(os.path.join(output_dir, "01-top_keywords_by_biobank.csv"), index=False)

# Create a heatmap of the top 5 keywords across biobanks
pivot_df = keyword_df.pivot_table(
    index='Keyword', 
    columns='Biobank', 
    values='Count',
    fill_value=0
)

# Get the top 15 keywords overall
top_keywords = keyword_df.groupby('Keyword')['Count'].sum().nlargest(15).index

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_df.loc[pivot_df.index.isin(top_keywords)], 
    annot=True, 
    fmt=".1f", 
    cmap="Blues"
)
plt.title("Top 15 Keywords Across Biobanks")
plt.ylabel("Keyword")
plt.xlabel("Biobank")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-top_keywords_heatmap.png"))
plt.close()

# ---------------------------------------------
# 3. Most prolific authors across biobanks
# ---------------------------------------------
author_rows = []
for biobank, stats in data['biobank_stats'].items():
    for author, count in list(stats['authors'].items())[:10]:  # Top 10 authors
        author_rows.append({
            'Biobank': biobank,
            'Author': author,
            'Publications': count
        })

author_df = pd.DataFrame(author_rows)
author_df.to_csv(os.path.join(output_dir, "01-top_authors_by_biobank.csv"), index=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=author_df.sort_values('Publications', ascending=False).head(20),
    x='Publications',
    y='Author',
    hue='Biobank',
    palette='viridis'
)
plt.title("Top 20 Most Prolific Authors Across Biobanks")
plt.xlabel("Number of Publications")
plt.ylabel("Author")
plt.legend(title="Biobank", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-top_authors_barplot.png"))
plt.close()

# ---------------------------------------------
# 4. Leading institutions analysis
# ---------------------------------------------
institution_rows = []
for biobank, stats in data['biobank_stats'].items():
    for institution, count in list(stats['institutions'].items())[:10]:  # Top 10 institutions
        institution_rows.append({
            'Biobank': biobank,
            'Institution': institution,
            'Publications': count
        })

institution_df = pd.DataFrame(institution_rows)
institution_df.to_csv(os.path.join(output_dir, "01-top_institutions_by_biobank.csv"), index=False)

# Group by institution type (simple approach - extract first word of institution name)
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
plt.title("Top 10 Institution Types by Total Publications")
plt.xlabel("Number of Publications")
plt.ylabel("Institution Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-institution_types_barplot.png"))
plt.close()

# ---------------------------------------------
# 5. Thematic analysis of keywords by biobank
# ---------------------------------------------
# Group keywords into themes (simplified approach)
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
theme_df.to_csv(os.path.join(output_dir, "01-keyword_themes_by_biobank.csv"), index=False)

pivot_themes = theme_df.pivot_table(
    index='Theme', 
    columns='Biobank', 
    values='Count',
    fill_value=0
)

# Normalize by column (biobank) totals
pivot_themes_norm = pivot_themes.div(pivot_themes.sum(axis=0), axis=1) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_themes_norm,
    annot=True, 
    fmt=".1f", 
    cmap="Blues"
)
plt.title("Thematic Focus by Biobank (% of Keywords)")
plt.ylabel("Theme")
plt.xlabel("Biobank")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-thematic_focus_heatmap.png"))
plt.close()

# ---------------------------------------------
# 6. Network diagram of common authors across biobanks (simplified)
# ---------------------------------------------
# Find authors who publish across multiple biobanks
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
multi_biobank_summary = author_biobank_matrix.loc[multi_biobank_authors]
multi_biobank_summary['Total_Biobanks'] = author_biobank_counts[multi_biobank_authors]
multi_biobank_summary['Total_Publications'] = author_biobank_matrix.loc[multi_biobank_authors].sum(axis=1)
multi_biobank_summary = multi_biobank_summary.sort_values(['Total_Biobanks', 'Total_Publications'], ascending=False)

multi_biobank_summary.to_csv(os.path.join(output_dir, "01-multi_biobank_authors.csv"))

# Get top 20 multi-biobank authors for visualization
top_multi_biobank = multi_biobank_summary.head(20)

# Create a stacked bar chart
plt.figure(figsize=(12, 10))
top_multi_biobank.drop(['Total_Biobanks', 'Total_Publications'], axis=1).plot(
    kind='barh', 
    stacked=True,
    figsize=(12, 10),
    colormap='viridis'
)
plt.title("Publications by Biobank for Top 20 Cross-Biobank Authors")
plt.xlabel("Number of Publications")
plt.ylabel("Author")
plt.legend(title="Biobank", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-cross_biobank_authors_barplot.png"))
plt.close()

# ---------------------------------------------
# 7. Comparative analysis of research focus (based on keywords)
# ---------------------------------------------
# Create a summary of the top distinctive keywords for each biobank
keyword_matrix = pd.pivot_table(
    keyword_df,
    index='Keyword',
    columns='Biobank',
    values='Count',
    fill_value=0
)

# Calculate the percentage of each keyword within each biobank
keyword_percentages = keyword_matrix.div(keyword_matrix.sum(axis=0), axis=1) * 100

# Find distinctive keywords (keywords that have a much higher percentage in one biobank)
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
distinctive_df.to_csv(os.path.join(output_dir, "01-distinctive_keywords_by_biobank.csv"), index=False)

# Create a visualization of the top 5 distinctive keywords for each biobank
top_distinctive = distinctive_df.groupby('Biobank').head(5)

plt.figure(figsize=(14, 10))
g = sns.catplot(
    data=top_distinctive,
    kind='bar',
    x='Keyword',
    y='Distinctiveness',
    hue='Biobank',
    col='Biobank',
    col_wrap=2,
    height=5,
    aspect=1.5,
    palette='viridis',
    sharex=False
)
g.set_xticklabels(rotation=45, ha='right')
g.set_titles("{col_name}")
g.fig.suptitle("Most Distinctive Keywords by Biobank", y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01-distinctive_keywords_plot.png"))
plt.close()

print("✅ Analysis complete. Results saved to:", output_dir)