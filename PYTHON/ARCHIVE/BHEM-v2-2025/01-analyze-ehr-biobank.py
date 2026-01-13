import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
from collections import Counter
import os
from matplotlib_venn import venn2

# Define file path
csv_file = "DATA/global_ehr_biobank_results.csv"
output_dir = "ANALYSIS/PUBMED-RETRIEVAL-DESC-ANALYTICS"
os.makedirs(output_dir, exist_ok=True)

# Check if file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File not found: {csv_file}")

# Load dataset
df = pd.read_csv(csv_file)

# Ensure expected columns exist
expected_cols = ["Year", "Journal", "Affiliations"]
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# Clean and preprocess
df = df[df["Year"].notnull()]
df["Year"] = df["Year"].astype(int)
df = df[df["Year"] <= 2024]
df["Year"] = df["Year"].astype(str)

# --- Analysis 1: Articles per year ---
year_counts = df["Year"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o")
plt.title("Number of Articles per Year")
plt.xlabel("Publication Year")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "articles_per_year.png"))
plt.close()

# --- Analysis 2: Top 20 Journals ---
journal_counts = df["Journal"].value_counts().head(20)
plt.figure(figsize=(10, 6))
sns.barplot(y=journal_counts.index, x=journal_counts.values, palette="Blues_d")
plt.title("Top 20 Journals by Number of Articles")
plt.xlabel("Number of Articles")
plt.ylabel("Journal")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_journals.png"))
plt.close()

# --- Analysis 3: Country mentions in affiliations using pycountry ---
all_countries = {country.name for country in pycountry.countries}
affiliations = df["Affiliations"].dropna().tolist()

country_counter = Counter()
for aff in affiliations:
    for country in all_countries:
        if country in aff:
            country_counter[country] += 1

top_countries = country_counter.most_common(20)
countries, counts = zip(*top_countries)

plt.figure(figsize=(10, 6))
sns.barplot(x=counts, y=countries, palette="viridis")
plt.title("Top 20 Country Mentions in Author Affiliations")
plt.xlabel("Number of Mentions")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_countries.png"))
plt.close()

# --- Analysis 4: Top MeSH Terms ---
mesh_terms = df["MeSH Terms"].dropna().str.split(";").explode().str.strip().str.lower()
mesh_counts = mesh_terms.value_counts().head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=mesh_counts.values, y=mesh_counts.index, palette="crest")
plt.title("Top 20 MeSH Terms")
plt.xlabel("Frequency")
plt.ylabel("MeSH Term")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_mesh_terms.png"))
plt.close()

# --- Analysis 5: Top Keywords ---
keywords = df["Keywords"].dropna().str.split(";").explode().str.strip().str.lower()
keyword_counts = keywords.value_counts().head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=keyword_counts.values, y=keyword_counts.index, palette="flare")
plt.title("Top 20 Author-Supplied Keywords")
plt.xlabel("Frequency")
plt.ylabel("Keyword")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_keywords.png"))
plt.close()

# --- Analysis 6a: Keyword Trends Over Time ---
df_keywords = df[['Year', 'Keywords']].dropna()
df_keywords = df_keywords.assign(Keyword=df_keywords['Keywords'].str.split(';')).explode('Keyword')
df_keywords['Keyword'] = df_keywords['Keyword'].str.strip().str.lower()

top_keywords = df_keywords['Keyword'].value_counts().nlargest(10).index
trend_data = df_keywords[df_keywords['Keyword'].isin(top_keywords)]

plt.figure(figsize=(12, 6))
sns.countplot(data=trend_data, x='Year', hue='Keyword', palette='tab10')
plt.title("Trends Over Time for Top 10 Keywords")
plt.xlabel("Year")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.legend(title="Keyword", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "keyword_trends_over_time.png"))
plt.close()

# --- Analysis 6b: MeSH Trends Over Time ---
df_mesh = df[['Year', 'MeSH Terms']].dropna()
df_mesh = df_mesh.assign(MeSH=df_mesh['MeSH Terms'].str.split(';')).explode('MeSH')
df_mesh['MeSH'] = df_mesh['MeSH'].str.strip().str.lower()

top_mesh_terms = df_mesh['MeSH'].value_counts().nlargest(10).index
mesh_trend_data = df_mesh[df_mesh['MeSH'].isin(top_mesh_terms)]

plt.figure(figsize=(12, 6))
sns.countplot(data=mesh_trend_data, x='Year', hue='MeSH', palette='tab20')
plt.title("Trends Over Time for Top 10 MeSH Terms")
plt.xlabel("Year")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.legend(title="MeSH Term", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mesh_trends_over_time.png"))
plt.close()

# --- Analysis 7: Overlap and Divergence Between MeSH Terms and Keywords ---
mesh_set = set(mesh_terms.dropna().unique())
keyword_set = set(keywords.dropna().unique())

overlap_terms = mesh_set & keyword_set
mesh_only = mesh_set - keyword_set
keyword_only = keyword_set - mesh_set

# Save overlap text
with open(os.path.join(output_dir, "mesh_keyword_overlap.txt"), "w") as f:
    f.write("Overlap Terms (MeSH ∩ Keywords):\n")
    f.write("\n".join(sorted(overlap_terms)))
    f.write("\n\nOnly in MeSH:\n")
    f.write("\n".join(sorted(mesh_only)))
    f.write("\n\nOnly in Keywords:\n")
    f.write("\n".join(sorted(keyword_only)))

# Venn Diagram
plt.figure(figsize=(6, 6))
venn2([mesh_set, keyword_set], set_labels=('MeSH Terms', 'Keywords'))
plt.title("Overlap Between MeSH Terms and Author Keywords")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mesh_keyword_venn.png"))
plt.close()

# --- Export CSV Tables ---
year_counts.to_csv(os.path.join(output_dir, "articles_per_year.csv"), header=["Count"])
journal_counts.to_csv(os.path.join(output_dir, "top_journals.csv"), header=["Count"])
pd.DataFrame(top_countries, columns=["Country", "Mentions"]).to_csv(os.path.join(output_dir, "top_countries.csv"), index=False)
mesh_counts.to_csv(os.path.join(output_dir, "top_mesh_terms.csv"), header=["Count"])
keyword_counts.to_csv(os.path.join(output_dir, "top_keywords.csv"), header=["Count"])

print("✅ Analysis complete. Plots and tables saved to the ANALYSIS/PUBMED-RETRIEVAL-DESC-ANALYTICS folder.")
