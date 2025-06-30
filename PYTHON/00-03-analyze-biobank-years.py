import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

# Define biobanks
biobanks = {
    "UK Biobank": ["UK Biobank"],
    "All of Us": ["All of Us"],
    "FinnGen": ["FinnGen"],
    "Estonian Biobank": ["Estonian Biobank"],
    "Million Veteran Program": ["Million Veteran Program", "MVP"]
}

def analyze_publications_by_year(csv_path, output_dir):
    """
    Analyze biobank publications by year from 2000 to 2024
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df)} articles")
    
    # Initialize data structure for counting publications by biobank and year
    yearly_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each article
    for _, row in df.iterrows():
        # Get the publication year
        year_str = str(row['Year']) if not pd.isna(row['Year']) else ""
        
        # Skip if no year data or if year is not a valid number
        if not year_str.strip() or not year_str.isdigit():
            continue
            
        year = int(year_str)
        
        # Skip years outside our range
        if year < 2000 or year > 2024:
            continue
        
        # Get biobanks mentioned in the paper
        biobanks_mentioned = row['Biobank'].split('; ') if not pd.isna(row['Biobank']) else []
        
        # Count publication for each biobank
        for biobank_name in biobanks_mentioned:
            biobank_name = biobank_name.strip()
            for canonical_name, aliases in biobanks.items():
                if biobank_name in aliases:
                    yearly_counts[canonical_name][year] += 1
                    break
    
    # Create a dataframe for the year-by-year breakdown
    years = range(2000, 2025)
    data = {'Year': list(years)}
    
    for biobank in biobanks.keys():
        counts = [yearly_counts[biobank].get(year, 0) for year in years]
        data[biobank] = counts
    
    yearly_df = pd.DataFrame(data)
    
    # Calculate cumulative counts
    cumulative_data = {'Year': list(years)}
    for biobank in biobanks.keys():
        counts = [yearly_counts[biobank].get(year, 0) for year in years]
        cumulative = np.cumsum(counts)
        cumulative_data[biobank] = cumulative
    
    cumulative_df = pd.DataFrame(cumulative_data)
    
    # Save data to CSV files
    yearly_df.to_csv(os.path.join(output_dir, 'biobank_publications_by_year.csv'), index=False)
    cumulative_df.to_csv(os.path.join(output_dir, 'biobank_cumulative_publications.csv'), index=False)
    
    # Print the yearly breakdown
    print("\nYear-by-Year Publication Counts by Biobank:")
    print(yearly_df)
    
    # Print total counts for verification
    total_counts = {biobank: sum(yearly_counts[biobank].values()) for biobank in biobanks.keys()}
    print("\nTotal publication counts:")
    for biobank, count in total_counts.items():
        print(f"  - {biobank}: {count} papers")
    
    # Create visualizations
    create_visualizations(yearly_df, cumulative_df, output_dir)
    
    # Return the dataframes for further analysis if needed
    return yearly_df, cumulative_df

def create_visualizations(yearly_df, cumulative_df, output_dir):
    """
    Create visualizations for the publication data
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set better visual style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Year-by-year publication counts
    plt.figure(figsize=(16, 10))
    
    for i, biobank in enumerate(biobanks.keys()):
        plt.plot(yearly_df['Year'], yearly_df[biobank], 
                 marker='o', linewidth=2.5, markersize=8, 
                 label=biobank, color=colors[i % len(colors)])
    
    plt.title('Publication Trends for Biobanks by Year (2000-2024)', fontsize=20, pad=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Number of Publications', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(yearly_df['Year'][::2], fontsize=12, rotation=45)  # Show every other year
    plt.yticks(fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to start from 0 with appropriate limits
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'biobank_publications_by_year.png'), dpi=300)
    print(f"Saved yearly publication chart to {os.path.join(plots_dir, 'biobank_publications_by_year.png')}")
    plt.close()
    
    # 2. Cumulative publications over time
    plt.figure(figsize=(16, 10))
    
    for i, biobank in enumerate(biobanks.keys()):
        plt.plot(cumulative_df['Year'], cumulative_df[biobank], 
                 marker='o', linewidth=2.5, markersize=8, 
                 label=biobank, color=colors[i % len(colors)])
    
    plt.title('Cumulative Publications for Biobanks (2000-2024)', fontsize=20, pad=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Cumulative Number of Publications', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(cumulative_df['Year'][::2], fontsize=12, rotation=45)  # Show every other year
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'biobank_cumulative_publications.png'), dpi=300)
    print(f"Saved cumulative publication chart to {os.path.join(plots_dir, 'biobank_cumulative_publications.png')}")
    plt.close()
    
    # 3. Stacked bar chart for yearly publications
    plt.figure(figsize=(16, 10))
    
    # Prepare data for stacked bar chart
    bottom = np.zeros(len(yearly_df))
    
    for i, biobank in enumerate(biobanks.keys()):
        plt.bar(yearly_df['Year'], yearly_df[biobank], bottom=bottom, 
                label=biobank, color=colors[i % len(colors)], alpha=0.8)
        bottom += yearly_df[biobank].values
    
    plt.title('Total Publications by Biobank and Year (2000-2024)', fontsize=20, pad=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Number of Publications', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(yearly_df['Year'][::2], fontsize=12, rotation=45)  # Show every other year
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'biobank_publications_stacked.png'), dpi=300)
    print(f"Saved stacked bar chart to {os.path.join(plots_dir, 'biobank_publications_stacked.png')}")
    plt.close()

def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level if needed
    
    # Define input/output paths
    input_dir = os.path.join(parent_dir, "DATA")
    csv_file = os.path.join(input_dir, "00-00-ehr_biobank_articles.csv")
    output_dir = os.path.join(parent_dir, "ANALYSIS", "biobank-year-analysis")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: Input CSV file not found: {csv_file}")
        print("Please specify the correct path to your CSV file.")
        sys.exit(1)
    
    # Run the analysis
    try:
        yearly_df, cumulative_df = analyze_publications_by_year(csv_file, output_dir)
        print("\n✅ Analysis complete!")
        print(f"📊 Results saved to: {output_dir}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()