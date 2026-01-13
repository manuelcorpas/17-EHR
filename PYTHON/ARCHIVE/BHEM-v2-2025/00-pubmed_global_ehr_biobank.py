from Bio import Entrez
import csv
import time
import os

# Set your email for NCBI Entrez
Entrez.email = "your.email@example.com"

# Output location
output_dir = "DATA"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "global_ehr_biobank_results.csv")

# Time range for relevance
DATE_RANGE = '("2000"[PDAT] : "2025"[PDAT])'

# Thematically aligned subqueries
queries = [
    f'("biobank*" AND ("electronic health record*" OR "EHR")) AND ("global" OR "international" OR "multi-country" OR "cross-national") AND {DATE_RANGE}',
    f'("data integration" OR "record linkage" OR "data harmonisation" OR "interoperability") AND ("EHR" OR "health records") AND ("biobank*" OR "population cohort*") AND {DATE_RANGE}',
    f'( ("low- and middle-income countries" OR "LMIC" OR "global south") AND ("EHR" OR "biobank*" OR "genomics") ) OR ("underrepresented populations" AND "genomic medicine") AND {DATE_RANGE}',
    f'( ("machine learning" OR "deep learning" OR "natural language processing") AND ("EHR" OR "real-world data") ) AND ("biobank*" OR "genomic cohort*") AND {DATE_RANGE}',
    f'("data governance" OR "dynamic consent" OR "broad consent") AND ("biobank*" OR "EHR") AND {DATE_RANGE}',
    f'("real-world data" OR "real world evidence" OR "translational research") AND ("biobank*" OR "EHR linked cohort*") AND {DATE_RANGE}',
    f'("precision medicine" OR "personalised medicine") AND ("biobank*" OR "EHR") AND {DATE_RANGE}',
    f'("FAIR data" OR "data standards" OR "metadata harmonisation") AND ("biobank*" OR "clinical data") AND {DATE_RANGE}',
]

# Track all PMIDs to avoid duplication
seen_pmids = set()

# Open CSV for writing
with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "PMID", "Title", "Abstract", "Authors", "Journal", "Year", "PubDate", "DOI",
        "MeSH Terms", "Keywords", "Affiliations"
    ])

    for idx, query in enumerate(queries):
        print(f"\n🔎 [{idx+1}/{len(queries)}] Searching PubMed...")
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=10000)
        search_results = Entrez.read(search_handle)
        id_list = search_results.get("IdList", [])
        print(f"📄 Found {len(id_list)} articles")

        for start in range(0, len(id_list), 1000):
            end = min(start + 1000, len(id_list))
            print(f"⏬ Fetching records {start+1} to {end}")
            batch_ids = id_list[start:end]

            fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="xml", retmode="xml")
            records = Entrez.read(fetch_handle)

            for article in records['PubmedArticle']:
                pmid = str(article['MedlineCitation']['PMID'])
                if pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)

                article_data = article['MedlineCitation']['Article']
                title = article_data.get('ArticleTitle', '')
                abstract = article_data.get('Abstract', {}).get('AbstractText', [''])[0]

                authors = []
                affiliations = set()
                for author in article_data.get('AuthorList', []):
                    if 'LastName' in author and 'Initials' in author:
                        authors.append(f"{author['LastName']} {author['Initials']}")
                    for aff in author.get('AffiliationInfo', []):
                        aff_text = aff.get('Affiliation')
                        if aff_text:
                            affiliations.add(aff_text)
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

                # MeSH terms
                mesh_terms = []
                for mesh_heading in article['MedlineCitation'].get('MeshHeadingList', []):
                    descriptor = mesh_heading['DescriptorName']
                    mesh_terms.append(str(descriptor))
                mesh_str = "; ".join(mesh_terms)

                # Keywords
                keyword_list = article['MedlineCitation'].get('KeywordList', [])
                keywords = []
                for kw_group in keyword_list:
                    for keyword in kw_group:
                        keywords.append(str(keyword))
                keyword_str = "; ".join(keywords)

                writer.writerow([
                    pmid, title, abstract, author_str, journal, year, date_str, doi,
                    mesh_str, keyword_str, affiliation_str
                ])

            time.sleep(0.1)

print(f"\n✅ Complete! {len(seen_pmids)} unique articles saved to: {csv_file}")
