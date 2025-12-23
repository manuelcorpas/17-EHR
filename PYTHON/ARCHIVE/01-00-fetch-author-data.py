"""
PubMed Author Data Fetcher
Retrieves author and affiliation information using existing PMIDs

SCRIPT: PYTHON/01-00-fetch-author-data.py
OUTPUT: ANALYSIS/01-00-FETCH-AUTHOR-DATA/
PURPOSE: Augment existing biobank dataset with author information from PubMed
"""

import os
import pandas as pd
import numpy as np
import requests
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
from tqdm import tqdm

# Setup paths (scripts run from root directory)
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-00-FETCH-AUTHOR-DATA")
os.makedirs(analysis_dir, exist_ok=True)

def fetch_pmid_details_batch(pmids, batch_size=200, delay=0.5):
    """
    Fetch author and affiliation details for a batch of PMIDs using PubMed eUtils API
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    all_results = {}
    
    # Process PMIDs in batches
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching author data"):
        batch_pmids = pmids[i:i + batch_size]
        pmid_string = ','.join(map(str, batch_pmids))
        
        params = {
            'db': 'pubmed',
            'id': pmid_string,
            'rettype': 'xml',
            'retmode': 'xml'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract data for each article in the batch
            for article in root.findall('.//PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                if pmid_elem is not None:
                    pmid = pmid_elem.text
                    
                    # Extract authors
                    authors = []
                    affiliations = []
                    
                    author_list = article.find('.//AuthorList')
                    if author_list is not None:
                        for author in author_list.findall('Author'):
                            # Get author name
                            last_name = author.find('LastName')
                            first_name = author.find('ForeName')
                            initials = author.find('Initials')
                            
                            if last_name is not None:
                                name_parts = [last_name.text]
                                if first_name is not None:
                                    name_parts.append(first_name.text)
                                elif initials is not None:
                                    name_parts.append(initials.text)
                                
                                full_name = ' '.join(name_parts)
                                authors.append(full_name)
                            
                            # Get affiliation
                            affiliation_info = author.find('AffiliationInfo')
                            if affiliation_info is not None:
                                affiliation = affiliation_info.find('Affiliation')
                                if affiliation is not None:
                                    affiliations.append(affiliation.text)
                    
                    # Store results
                    all_results[pmid] = {
                        'authors': '; '.join(authors) if authors else '',
                        'affiliations': '; '.join(set(affiliations)) if affiliations else '',
                        'author_count': len(authors),
                        'first_author': authors[0] if authors else '',
                        'last_author': authors[-1] if len(authors) > 1 else ''
                    }
            
            # Rate limiting - be respectful to NCBI servers
            time.sleep(delay)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching batch starting at index {i}: {e}")
            continue
        except ET.ParseError as e:
            print(f"Error parsing XML for batch starting at index {i}: {e}")
            continue
    
    return all_results

def extract_institutions_from_affiliations(affiliations_text):
    """
    Extract institution names from affiliation text
    """
    if not affiliations_text or pd.isna(affiliations_text):
        return []
    
    institutions = []
    affiliations = affiliations_text.split('; ')
    
    for affiliation in affiliations:
        # Common patterns for institutions
        # Look for University, Institute, Hospital, Center, College
        institution_patterns = [
            r'([^,]*(?:University|Institute|Hospital|Center|Centre|College|School)[^,]*)',
            r'([^,]*(?:Medical|Health)[^,]*(?:Center|Centre|System|Institute)[^,]*)',
            r'([^,]*(?:Cancer|Research)[^,]*(?:Center|Centre|Institute)[^,]*)'
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, affiliation, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match) > 5:  # Filter out very short matches
                    institutions.append(clean_match)
                    break  # Take first good match per affiliation
    
    return list(set(institutions))  # Remove duplicates

def augment_biobank_data_with_authors(df, analysis_dir):
    """
    Main function to augment existing biobank dataset with author information
    """
    print(f"📊 Processing {len(df):,} publications for author data...")
    
    # Get unique PMIDs
    pmids = df['PMID'].dropna().astype(str).tolist()
    unique_pmids = list(set(pmids))
    
    print(f"🔍 Fetching author data for {len(unique_pmids):,} unique PMIDs...")
    
    # Fetch author data
    author_data = fetch_pmid_details_batch(unique_pmids)
    
    print(f"✅ Successfully retrieved author data for {len(author_data):,} publications")
    
    # Merge back with original dataframe
    df['PMID_str'] = df['PMID'].astype(str)
    
    # Create author columns
    df['Authors'] = df['PMID_str'].map(lambda x: author_data.get(x, {}).get('authors', ''))
    df['Affiliations'] = df['PMID_str'].map(lambda x: author_data.get(x, {}).get('affiliations', ''))
    df['Author_Count'] = df['PMID_str'].map(lambda x: author_data.get(x, {}).get('author_count', 0))
    df['First_Author'] = df['PMID_str'].map(lambda x: author_data.get(x, {}).get('first_author', ''))
    df['Last_Author'] = df['PMID_str'].map(lambda x: author_data.get(x, {}).get('last_author', ''))
    
    # Extract institutions
    print("🏛️ Extracting institution information...")
    df['Institutions'] = df['Affiliations'].apply(extract_institutions_from_affiliations)
    df['Institution_Count'] = df['Institutions'].apply(len)
    df['Primary_Institution'] = df['Institutions'].apply(lambda x: x[0] if x else '')
    
    # Clean up
    df = df.drop('PMID_str', axis=1)
    
    # Save augmented dataset to analysis directory
    output_file = os.path.join(analysis_dir, 'biobank_research_data_with_authors.csv')
    df.to_csv(output_file, index=False)
    print(f"💾 Augmented dataset saved: {output_file}")
    
    # Save raw author data as backup
    author_data_file = os.path.join(analysis_dir, 'raw_author_data.csv')
    author_df = pd.DataFrame([
        {'pmid': pmid, **data} for pmid, data in author_data.items()
    ])
    author_df.to_csv(author_data_file, index=False)
    print(f"💾 Raw author data saved: {author_data_file}")
    
    # Generate summary report
    print("\n📈 AUTHOR DATA SUMMARY:")
    print(f"Publications with author data: {(df['Authors'] != '').sum():,}")
    print(f"Publications with affiliation data: {(df['Affiliations'] != '').sum():,}")
    print(f"Publications with institution data: {(df['Institution_Count'] > 0).sum():,}")
    
    print("\nBy Biobank:")
    for biobank in df['Biobank'].unique():
        biobank_data = df[df['Biobank'] == biobank]
        with_authors = (biobank_data['Authors'] != '').sum()
        total = len(biobank_data)
        print(f"  {biobank}: {with_authors}/{total} ({with_authors/total*100:.1f}%) with authors")
    
    # Show sample of most prolific authors
    if not df['Authors'].empty:
        print("\n👥 SAMPLE OF FREQUENT AUTHORS:")
        all_authors = []
        for authors_str in df['Authors'].dropna():
            if authors_str:
                authors = [a.strip() for a in authors_str.split(';')]
                all_authors.extend(authors)
        
        if all_authors:
            from collections import Counter
            author_counts = Counter(all_authors)
            print("Top 10 most frequent authors:")
            for author, count in author_counts.most_common(10):
                print(f"  {author}: {count} publications")
    
    return df

def create_collaboration_ready_dataset(original_file, analysis_dir):
    """
    Complete pipeline to create collaboration-analysis-ready dataset
    """
    print("🚀 Creating collaboration-ready dataset...")
    
    # Load original data
    df = pd.read_csv(original_file)
    print(f"📖 Loaded {len(df):,} publications from {original_file}")
    
    # Apply basic filtering
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    df = df.dropna(subset=['PMID'])
    print(f"📊 After filtering: {len(df):,} publications")
    
    # Augment with author data
    df_augmented = augment_biobank_data_with_authors(df, analysis_dir)
    
    return df_augmented

if __name__ == "__main__":
    # Setup paths
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "DATA")
    analysis_dir = os.path.join(current_dir, "ANALYSIS", "01-00-FETCH-AUTHOR-DATA")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Input file (your existing dataset)
    input_file = os.path.join(data_dir, 'biobank_research_data.csv')
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please ensure your biobank dataset is saved as 'biobank_research_data.csv' in the DATA directory")
    else:
        # Create augmented dataset
        df_with_authors = create_collaboration_ready_dataset(input_file, analysis_dir)
        
        print(f"\n🎯 Success! Collaboration-ready dataset created.")
        print(f"📁 Output file: {os.path.join(analysis_dir, 'biobank_research_data_with_authors.csv')}")
        print(f"📊 Ready for collaboration network analysis with {len(df_with_authors):,} publications")
        print(f"📂 All outputs saved to: {analysis_dir}")
        
        # Generate final summary report
        summary_report = f"""
AUTHOR DATA FETCHING SUMMARY REPORT
==================================
Input: {input_file}
Output Directory: {analysis_dir}
Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

DATASET STATISTICS:
- Total publications processed: {len(df_with_authors):,}
- Publications with author data: {(df_with_authors['Authors'] != '').sum():,}
- Publications with affiliations: {(df_with_authors['Affiliations'] != '').sum():,}
- Publications with institutions: {(df_with_authors['Institution_Count'] > 0).sum():,}
- Average authors per publication: {df_with_authors['Author_Count'].mean():.1f}
- Average institutions per publication: {df_with_authors['Institution_Count'].mean():.1f}

OUTPUTS GENERATED:
1. biobank_research_data_with_authors.csv - Main augmented dataset
2. raw_author_data.csv - Raw author data from PubMed
3. author_data_summary_report.txt - This summary report

NEXT STEP:
Run the collaboration network analysis:
python3 PYTHON/01-01-collaboration-network-analysis-updated.py
"""
        
        # Save summary report
        report_file = os.path.join(analysis_dir, 'author_data_summary_report.txt')
        with open(report_file, 'w') as f:
            f.write(summary_report)
        print(f"📄 Summary report saved: {report_file}")