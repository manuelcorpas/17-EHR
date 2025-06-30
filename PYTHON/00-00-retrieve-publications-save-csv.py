from Bio import Entrez
import csv
import time
import os
import sys
import re
from datetime import datetime
import logging

# Set Entrez parameters
Entrez.email = "your.email@example.com"
Entrez.tool = "BiopythonClient"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("DATA", "citation_script.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRY = 3
BATCH_SIZE = 50  # Smaller batch size for better performance

# Define paths - match your file structure
output_dir = "DATA"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "00-00-ehr_biobank_articles.csv")
output_csv = os.path.join(output_dir, "00-00-ehr_biobank_articles_with_citations.csv")

def get_citation_count_batch(pmid_list):
    """
    Get citation counts for a batch of PMIDs using elink.
    This is much faster than checking CommentsCorrectionsList.
    """
    citation_counts = {}
    for pmid in pmid_list:
        citation_counts[pmid] = 0
    
    try:
        # Use elink to get citation data (much more efficient)
        pmids_str = ",".join(pmid_list)
        link_handle = Entrez.elink(
            dbfrom="pubmed", 
            db="pubmed",
            linkname="pubmed_pubmed_citedin", 
            id=pmids_str
        )
        link_results = Entrez.read(link_handle)
        link_handle.close()
        
        # Parse the results
        for i, result in enumerate(link_results):
            if i < len(pmid_list) and result.get('LinkSetDb'):
                pmid = pmid_list[i]
                for linkset in result['LinkSetDb']:
                    if linkset.get('LinkName') == 'pubmed_pubmed_citedin':
                        citation_counts[pmid] = len(linkset.get('Link', []))
                        if citation_counts[pmid] > 0:
                            print(f"📊 PMID {pmid} has {citation_counts[pmid]} citations")
        
    except Exception as e:
        logger.error(f"Error fetching citation counts: {e}")
    
    return citation_counts

def modify_original_csv():
    """
    Run the original script, but modify it to include citation counts.
    """
    # Import the original script's functions and constants
    from Bio import Entrez
    import csv
    import time
    import os
    import sys
    import re
    from datetime import datetime
    
    # Constants
    BATCH_SIZE = 500
    MAX_RETRY = 3

    # Use the same path structure as the original script
    output_dir = "DATA"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "00-00-ehr_biobank_articles_with_citations.csv")
    
    # Biobank aliases
    biobanks = {
        "UK Biobank": ["UK Biobank"],
        "All of Us": ["All of Us"],
        "FinnGen": ["FinnGen"],
        "Estonian Biobank": ["Estonian Biobank"],
        "Million Veteran Program": ["Million Veteran Program", "MVP"]
    }
    
    # Build biobank query string
    biobank_query = " OR ".join([f'"{alias}"' for names in biobanks.values() for alias in names])
    
    # Other global variables
    seen_pmids = set()
    processed_count = 0
    current_year = datetime.now().year
    
    # Institution normalization mapping
    institution_mapping = {
        # UK-based institutions
        "university of oxford": "University of Oxford",
        "oxford university": "University of Oxford",
        "university of cambridge": "University of Cambridge",
        "cambridge university": "University of Cambridge",
        "imperial college london": "Imperial College London",
        "university college london": "University College London",
        "king's college london": "King's College London",
        "kings college london": "King's College London",
        
        # US-based institutions
        "harvard medical school": "Harvard Medical School",
        "harvard university": "Harvard University",
        "stanford university": "Stanford University",
        "university of california": "University of California",
        "ucla": "University of California, Los Angeles",
        "ucsf": "University of California, San Francisco",
        "uc berkeley": "University of California, Berkeley",
        "johns hopkins university": "Johns Hopkins University",
        "yale university": "Yale University",
        "yale school of medicine": "Yale School of Medicine",
        "massachusetts general hospital": "Massachusetts General Hospital",
        "mayo clinic": "Mayo Clinic",
        "nih": "National Institutes of Health",
        "national institutes of health": "National Institutes of Health",
        
        # Finnish institutions
        "university of helsinki": "University of Helsinki",
        "institute for molecular medicine finland": "Institute for Molecular Medicine Finland (FIMM)",
        "fimm": "Institute for Molecular Medicine Finland (FIMM)",
        
        # Estonian institutions
        "estonian biobank": "Estonian Biobank",
        "estonian genome center": "Estonian Genome Center",
        "university of tartu": "University of Tartu"
    }
    
    def extract_institutions(affiliation_text):
        """
        Extract and normalize institution names from affiliation text
        """
        if not affiliation_text:
            return []
        
        institutions = []
        
        # Split by commas or semicolons
        parts = re.split(r'[,;]', affiliation_text)
        
        # Extract candidate institution names
        for part in parts:
            part = part.strip()
            
            # Skip too short parts or those that are likely not institutions
            if len(part) < 4 or part.isdigit():
                continue
            
            # Check for common institution indicators
            inst_indicators = ['university', 'college', 'institute', 'school', 'center', 'centre', 
                              'hospital', 'clinic', 'laboratory', 'department', 'faculty']
            
            if any(indicator in part.lower() for indicator in inst_indicators):
                # Extract the institution name
                institutions.append(part)
            elif 'biobank' in part.lower() or any(biobank_name.lower() in part.lower() for biobank_name in biobanks.keys()):
                # Specific handling for biobank names
                institutions.append(part)
        
        # Normalize institution names
        normalized_institutions = []
        for inst in institutions:
            inst_lower = inst.lower()
            
            # Try to match with known institutions
            matched = False
            for pattern, normalized_name in institution_mapping.items():
                if pattern in inst_lower:
                    normalized_institutions.append(normalized_name)
                    matched = True
                    break
            
            # If no match found, use the original institution name
            if not matched:
                normalized_institutions.append(inst)
        
        return list(set(normalized_institutions))  # Remove duplicates
    
    # Dictionary to store PMID to citation count mapping
    pmid_citations = {}
    
    def save_records_to_csv(records, csv_file):
        nonlocal processed_count
        
        batch_count = 0
        pmid_list = []
        
        # Extract all PMIDs first to batch citation lookup
        for article in records.get('PubmedArticle', []):
            pmid = str(article['MedlineCitation']['PMID'])
            if pmid not in seen_pmids:
                pmid_list.append(pmid)
        
        # Get citation counts for this batch of PMIDs
        if pmid_list:
            batch_citations = get_citation_count_batch(pmid_list)
            pmid_citations.update(batch_citations)
        
        with open(csv_file, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for article in records.get('PubmedArticle', []):
                pmid = str(article['MedlineCitation']['PMID'])
                if pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)

                article_data = article['MedlineCitation']['Article']
                title = article_data.get('ArticleTitle', '')

                abstract_parts = article_data.get('Abstract', {}).get('AbstractText', [''])
                if isinstance(abstract_parts, list):
                    abstract = ' '.join(str(part) for part in abstract_parts)
                else:
                    abstract = str(abstract_parts)

                authors = []
                affiliations = set()
                for author in article_data.get('AuthorList', []):
                    if 'LastName' in author and 'Initials' in author:
                        author_name = f"{author['LastName']} {author['Initials']}"
                        authors.append(author_name)
                    for aff in author.get('AffiliationInfo', []):
                        affiliations.add(aff.get('Affiliation', ''))
                author_str = "; ".join(authors)
                affiliation_str = "; ".join(affiliations)

                journal = article_data.get('Journal', {}).get('Title', '')
                pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                year = pub_date.get('Year', '')
                date_str = f"{year}-{pub_date.get('Month', '')}"

                doi = ''
                for id_item in article_data.get('ELocationID', []):
                    if id_item.attributes.get('EIdType') == 'doi':
                        doi = str(id_item)

                # Extract MeSH terms separately from keywords
                mesh_terms = []
                for mesh_heading in article['MedlineCitation'].get('MeshHeadingList', []):
                    descriptor = str(mesh_heading['DescriptorName'])
                    mesh_terms.append(descriptor)
                mesh_str = "; ".join(mesh_terms)
                
                # Extract author-provided keywords separately
                keyword_list = []
                for kw_group in article['MedlineCitation'].get('KeywordList', []):
                    for k in kw_group:
                        keyword_list.append(str(k))
                keywords = "; ".join(keyword_list)

                # Get citation count from our pre-fetched dictionary
                citation_count = pmid_citations.get(pmid, 0)

                full_text = f"{title} {abstract} {keywords}".lower()
                matched_biobanks = {biobank for biobank, aliases in biobanks.items()
                                    if any(alias.lower() in full_text for alias in aliases)}
                biobank_str = "; ".join(sorted(matched_biobanks))

                # Extract institutions using improved method
                extracted_institutions = []
                for affiliation in affiliations:
                    institutions = extract_institutions(affiliation)
                    extracted_institutions.extend(institutions)

                writer.writerow([
                    pmid, title, abstract, author_str, journal, year, date_str, doi,
                    mesh_str, keywords, affiliation_str, biobank_str, citation_count
                ])
                batch_count += 1
        processed_count += batch_count
        return batch_count
    
    def process_records(query, count):
        nonlocal processed_count, seen_pmids
        
        # For each batch, we need to get the right portion of results
        for start in range(0, count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, count)
            retry_count, success = 0, False

            while not success and retry_count < MAX_RETRY:
                try:
                    print(f"    ⏬ Fetching records {start + 1} to {end}")
                    
                    # For each batch, do a new search with the correct retstart
                    search_handle = Entrez.esearch(
                        db="pubmed", 
                        term=query, 
                        retstart=start,
                        retmax=BATCH_SIZE
                    )
                    search_results = Entrez.read(search_handle)
                    search_handle.close()
                    
                    # Get the PMIDs for this batch
                    pmid_list = search_results["IdList"]
                    
                    if not pmid_list:
                        print("    ⚠️ No PMIDs returned in this batch")
                        success = True
                        continue
                    
                    # Fetch the full records using the PMIDs
                    fetch_handle = Entrez.efetch(
                        db="pubmed", 
                        rettype="xml", 
                        retmode="xml",
                        id=",".join(pmid_list)
                    )
                    records = Entrez.read(fetch_handle)
                    fetch_handle.close()

                    batch_count = save_records_to_csv(records, csv_file)
                    success = True
                    print(f"    ✓ Saved {batch_count} articles. Total: {processed_count}")
                    
                except Exception as e:
                    retry_count += 1
                    print(f"    ⚠️ Error: {e} (Attempt {retry_count}/{MAX_RETRY})")
                    if retry_count < MAX_RETRY:
                        wait_time = 2 ** retry_count
                        print(f"    ⏱️ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print("    ❌ Max retries reached. Skipping batch...")
                        
            # Sleep between batches to respect NCBI's rate limits
            time.sleep(1)
    
    # Main execution
    try:
        # Write CSV header once
        with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "PMID", "Title", "Abstract", "Authors", "Journal", "Year", "PubDate", "DOI",
                "MeSH Terms", "Keywords", "Affiliations", "Biobank", "Citation Count"
            ])

        print("🔄 Using year-by-year search to handle large result sets...")
        
        # Process each year (we'll go backwards from most recent)
        # Changed to process all years from current_year back to 2000
        for year in range(current_year, 1999, -1): 
            year_query = f"({biobank_query}) AND ({year}[PDAT])"
            print(f"\n📅 Processing year {year}...")

            search_handle = Entrez.esearch(db="pubmed", term=year_query, retmax=0)
            search_results = Entrez.read(search_handle)
            search_handle.close()

            year_count = int(search_results["Count"])
            print(f"  📄 Found {year_count} articles for {year}")
            if year_count == 0:
                continue

            if year_count > 9000:
                print(f"  ⚠️ Year {year} >9000, processing by month...")
                for month in range(1, 13):
                    month_name = datetime(year, month, 1).strftime("%b")
                    month_query = f"({biobank_query}) AND ({year}/{month_name}[PDAT])"
                    print(f"    🗓️ {month_name} {year}...")

                    month_search = Entrez.esearch(db="pubmed", term=month_query, retmax=0)
                    month_results = Entrez.read(month_search)
                    month_search.close()

                    month_count = int(month_results["Count"])
                    print(f"      📄 Found {month_count} articles")
                    if month_count == 0:
                        continue
                    process_records(month_query, month_count)
            else:
                process_records(year_query, year_count)

        print(f"\n✅ Done! {processed_count} articles saved to: {csv_file}")
        print(f"📊 Unique PMIDs processed: {len(seen_pmids)}")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        start_time = time.time()
        print("🚀 Starting citation count retrieval with improved method...")
        
        # Run the modified script
        modify_original_csv()
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Total processing time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)