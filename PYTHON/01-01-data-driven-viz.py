#!/usr/bin/env python3
"""
Complete Research Gap Visualization - ALL DISEASES LABELED (FIXED POSITIONING)
==============================================================================

Creates the full visualization with main plot + panels A,B,C,D
All 25 diseases labeled in main plot with improved label positioning
Red dot labels positioned much closer to their bubbles

Usage: 
    python research_gap_viz_fixed.py

Output: Complete visualization with all diseases labeled and fixed positioning
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

def load_research_gap_data():
    """Load the research gap analysis data from actual output files (25 diseases)."""
    
    # File paths - match what 01-00-research-gap-discovery.py creates (25 diseases)
    gap_file = 'ANALYSIS/01-00-RESEARCH-GAP-DISCOVERY/research_gaps_comprehensive_25diseases.csv'
    effort_file = 'ANALYSIS/01-00-RESEARCH-GAP-DISCOVERY/research_effort_by_disease_25diseases.csv'
    summary_file = 'ANALYSIS/01-00-RESEARCH-GAP-DISCOVERY/gap_analysis_summary_25diseases.json'
    
    # Check if files exist
    missing_files = []
    for file in [gap_file, effort_file, summary_file]:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files:")
        for file in missing_files:
            print(f"   {file}")
        print("\nPlease run 01-00-research-gap-discovery.py first.")
        return None, None, None
    
    # Load data
    print(f"\nLoading data files from 25-disease analysis...")
    try:
        gap_df = pd.read_csv(gap_file)
        effort_df = pd.read_csv(effort_file)
        
        # Load summary JSON
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        print(f"Data loaded successfully:")
        print(f"   • {len(gap_df)} diseases analyzed")
        print(f"   • Total DALYs: {gap_df['dalys_millions'].sum():.1f}M")
        print(f"   • Total publications: {gap_df['publications_count'].sum():,}")
        
        return gap_df, effort_df, summary_data
        
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None, None, None

def create_complete_labeled_visualization(gap_df, summary_data, output_prefix='research_gap_complete_labeled_fixed'):
    """
    Create complete visualization with main plot + panels A,B,C,D, all diseases labeled with fixed positioning.
    """
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(22, 18))
    
    # Define grid: 3 rows, 4 columns with improved spacing
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.4)
    
    # Crisis visualization takes up top 2 rows, all columns
    ax_crisis = fig.add_subplot(gs[0:2, :])
    
    # Equity panels in bottom row
    ax_equity1 = fig.add_subplot(gs[2, 0])
    ax_equity2 = fig.add_subplot(gs[2, 1]) 
    ax_equity3 = fig.add_subplot(gs[2, 2])
    ax_equity4 = fig.add_subplot(gs[2, 3])
    
    # === MAIN CRISIS VISUALIZATION (TOP) ===
    critical_threshold = 80
    critical_mask = gap_df['research_gap_score'] > critical_threshold
    
    # Color coding
    colors = ['#d63031' if score > critical_threshold else '#0984e3' 
              for score in gap_df['research_gap_score']]
    
    # Bubble sizes proportional to DALYs
    sizes = gap_df['dalys_millions'] * 6
    
    # Create scatter plot
    scatter = ax_crisis.scatter(gap_df['dalys_millions'], gap_df['publications_count'], 
                               s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1.2)
    
    # Add research efficiency target line
    max_dalys = gap_df['dalys_millions'].max()
    target_x = np.linspace(0, max_dalys, 100)
    target_y = target_x * 10
    
    ax_crisis.plot(target_x, target_y, '--', color='green', linewidth=3, alpha=0.8,
                  label='TARGET: 10 publications per million DALYs')
    
    # Highlight priority research zone
    crisis_x = np.array([10, max_dalys, max_dalys, 10])
    crisis_y = np.array([0, 0, 150, 150])
    ax_crisis.fill(crisis_x, crisis_y, color='red', alpha=0.1, 
                  label='PRIORITY ZONE: High burden, low research')
    
    # Position labels next to their dots with smart collision avoidance
    def get_nearby_offset(x_val, y_val, bubble_radius, i, is_critical):
        """Generate position near the actual data point with collision avoidance."""
        
        # Calculate bubble radius in data coordinates (approximate)
        bubble_size_factor = np.sqrt(bubble_radius / (np.pi * 6))  # Reverse of size calculation
        
        if is_critical:
            # Special positioning for overlapping diseases - MUCH closer to dots
            if 'Malaria' in disease_name:
                # Position Malaria to the left and slightly below
                base_offset_x, base_offset_y = (-0.8, -0.3)  # Left and slightly below
                offset_x = base_offset_x * bubble_size_factor * 0.6  # Very close
                offset_y = base_offset_y * bubble_size_factor * 1.5   # Very close
            elif 'Diarrheal' in disease_name:
                # Position Diarrheal Diseases to the right and slightly below
                base_offset_x, base_offset_y = (0.8, -0.3)  # Right and slightly below
                offset_x = base_offset_x * bubble_size_factor * 0.6  # Very close
                offset_y = base_offset_y * bubble_size_factor * 1.5   # Very close
            else:
                # For other critical diseases: position at top, very close
                base_offset_x, base_offset_y = (0, 0.6)  # At top
                offset_x = base_offset_x * bubble_size_factor * 0.3  # Very close
                offset_y = base_offset_y * bubble_size_factor * 1.2   # Very close
        else:
            # For adequate (blue) diseases: use existing pattern
            offset_patterns = [
                (1.5, 0.5),    # Right
                (-1.5, 0.5),   # Left  
                (0, 1.2),      # Top
                (0, -1.2),     # Bottom
                (1.2, 1.0),    # Top-right
                (-1.2, 1.0),   # Top-left
                (1.2, -1.0),   # Bottom-right
                (-1.2, -1.0),  # Bottom-left
            ]
            
            # Choose pattern based on position to avoid plot edges
            pattern_idx = i % len(offset_patterns)
            base_offset_x, base_offset_y = offset_patterns[pattern_idx]
            
            # Scale offset by bubble size
            offset_x = base_offset_x * bubble_size_factor * 2  # For blue diseases
            offset_y = base_offset_y * bubble_size_factor * 5  # For blue diseases
        
        # Adjust based on position in plot to avoid edges
        if x_val < max_dalys * 0.15:  # Left edge
            offset_x = abs(offset_x)  # Force right
        elif x_val > max_dalys * 0.85:  # Right edge
            offset_x = -abs(offset_x)  # Force left
            
        if y_val < max_pubs * 0.15:  # Bottom edge
            offset_y = abs(offset_y)  # Force up
        elif y_val > max_pubs * 0.85:  # Top edge
            if not is_critical:  # Only adjust blue labels for top edge
                offset_y = -abs(offset_y)  # Force down
        
        return x_val + offset_x, y_val + offset_y
    
    # Label ALL diseases next to their dots
    max_pubs = gap_df['publications_count'].max()
    
    for i, (_, disease) in enumerate(gap_df.iterrows()):
        disease_name = disease['disease_subcategory']
        is_critical = disease['research_gap_score'] > critical_threshold
        
        # Shorten very long names
        if len(disease_name) > 20:
            if 'Chronic Obstructive' in disease_name:
                short_name = 'COPD'
            elif 'Neglected Tropical' in disease_name:
                short_name = 'Neglected Tropical Diseases'
            elif 'Preterm Birth' in disease_name:
                short_name = 'Preterm Birth Complications'
            elif 'Road Traffic' in disease_name:
                short_name = 'Road Traffic Accidents'
            elif 'Diabetes Mellitus' in disease_name:
                short_name = 'Type 2 Diabetes'
            else:
                short_name = disease_name[:18] + '..'
        else:
            short_name = disease_name
        
        # Get position near the actual data point
        bubble_size = disease['dalys_millions'] * 6
        label_x, label_y = get_nearby_offset(
            disease['dalys_millions'], disease['publications_count'], 
            bubble_size, i, is_critical
        )
        
        # Choose box color based on gap severity
        if is_critical:
            box_color = '#ffcccb'  # Light red for critical
            border_color = '#d63031'
        else:
            box_color = '#e6f3ff'  # Light blue for adequate
            border_color = '#0984e3'
        
        # Create label next to the dot
        ax_crisis.text(label_x, label_y, short_name,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=box_color, alpha=0.9, 
                               edgecolor=border_color, linewidth=1.2),
                      fontsize=8, fontweight='bold', 
                      ha='center', va='center',
                      color='#1a1a1a')
    
    # Style the main plot
    ax_crisis.set_xlabel('Global Disease Burden (Million DALYs)', fontweight='bold', fontsize=14)
    ax_crisis.set_ylabel('Biobank Research Publications', fontweight='bold', fontsize=14)
    ax_crisis.set_title('CRITICAL RESEARCH GAPS IN GLOBAL HEALTH (25 DISEASES)\nBubble size proportional to disease burden (DALYs)',
                       fontweight='bold', pad=25, fontsize=16)
    
    # Legend for main plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d63031', 
               markersize=12, label=f'CRITICAL GAP (score >{critical_threshold})', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0984e3', 
               markersize=12, label=f'ADEQUATE (score ≤{critical_threshold})', markeredgecolor='black'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='Research efficiency target'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.1, label='Priority zone')
    ]
    
    ax_crisis.legend(handles=legend_elements, loc='upper right', frameon=True, 
                    fancybox=True, shadow=True, fontsize=11)
    
    # Research gap statistics box
    critical_count = critical_mask.sum()
    critical_dalys = gap_df[critical_mask]['dalys_millions'].sum()
    critical_pubs = gap_df[critical_mask]['publications_count'].sum()
    total_diseases = len(gap_df)
    
    gap_text = (f"25-DISEASE ANALYSIS\n"
                f"• {critical_count} diseases with critical gaps\n"
                f"• {critical_dalys:.1f}M DALYs affected\n"
                f"• Only {critical_pubs} total publications\n"
                f"• Efficiency: {critical_pubs/critical_dalys:.2f} studies/M DALYs\n"
                f"• Target: {critical_dalys*10:.0f} studies needed\n"
                f"• Coverage: {total_diseases} diseases analyzed")
    
    ax_crisis.text(0.02, 0.88, gap_text, transform=ax_crisis.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor='#e3f2fd', alpha=0.95, edgecolor='#1976d2'),
                  fontweight='bold')
    
    ax_crisis.grid(True, alpha=0.3)
    ax_crisis.set_xlim(-2, max_dalys + 15)
    ax_crisis.set_ylim(-max_pubs*0.05, max_pubs*1.15)
    
    # === EQUITY ANALYSIS (BOTTOM PANELS) ===
    
    # Define disease categories by global priority
    global_south_diseases = [
        'Malaria', 'Tuberculosis', 'HIV/AIDS', 'Neglected Tropical Diseases', 
        'Diarrheal Diseases', 'Preterm Birth Complications', 'Road Traffic Accidents'
    ]
    
    gap_df['global_priority'] = gap_df['disease_subcategory'].apply(
        lambda x: 'Global South' if x in global_south_diseases else 'Global/High Income'
    )
    
    # Panel A: Research intensity by priority
    priority_stats = gap_df.groupby('global_priority').agg({
        'dalys_millions': 'sum',
        'publications_count': 'sum',
        'research_gap_score': 'mean'
    }).reset_index()
    
    priority_stats['research_intensity'] = priority_stats['publications_count'] / priority_stats['dalys_millions']
    
    bars = ax_equity1.bar(priority_stats['global_priority'], priority_stats['research_intensity'],
                         color=['#e74c3c', '#27ae60'], alpha=0.8, edgecolor='black')
    
    ax_equity1.set_ylabel('Pubs per Million DALYs', fontweight='bold', fontsize=10)
    ax_equity1.set_title('A. Global Health Equity Gap\n(25 Diseases)', fontweight='bold', fontsize=12)
    ax_equity1.set_xticklabels(priority_stats['global_priority'], rotation=45, ha='right', fontsize=9)
    
    # Add inequality ratio
    if len(priority_stats) == 2:
        ratio = priority_stats['research_intensity'].max() / priority_stats['research_intensity'].min()
        ax_equity1.text(0.5, 0.9, f'{ratio:.1f}x\nInequality', transform=ax_equity1.transAxes,
                       ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                       fontweight='bold', fontsize=10)
    
    # Panel B: Top research gaps
    top_gaps = gap_df.nlargest(10, 'research_gap_score')
    
    bars = ax_equity2.barh(range(len(top_gaps)), top_gaps['research_gap_score'],
                          color='#d63031', alpha=0.8, edgecolor='black')
    
    ax_equity2.set_yticks(range(len(top_gaps)))
    ax_equity2.set_yticklabels([name[:12] + '...' if len(name) > 12 else name 
                               for name in top_gaps['disease_subcategory']], fontsize=7)
    ax_equity2.set_xlabel('Research Gap Score', fontweight='bold', fontsize=10)
    ax_equity2.set_title('B. Top Research Gaps\n(25 Diseases)', fontweight='bold', fontsize=12)
    ax_equity2.invert_yaxis()
    
    # Panel C: Research effort by category
    category_data = gap_df.groupby('disease_category').agg({
        'dalys_millions': 'sum',
        'publications_count': 'sum'
    }).reset_index().sort_values('publications_count', ascending=True).tail(10)
    
    y_pos = np.arange(len(category_data))
    bars = ax_equity3.barh(y_pos, category_data['publications_count'], 
                          color='steelblue', alpha=0.8, edgecolor='black')
    
    ax_equity3.set_yticks(y_pos)
    ax_equity3.set_yticklabels([cat[:12] for cat in category_data['disease_category']], fontsize=7)
    ax_equity3.set_xlabel('Total Publications', fontweight='bold', fontsize=10)
    ax_equity3.set_title('C. Research by Category\n(25 Diseases)', fontweight='bold', fontsize=12)
    
    # Panel D: Critical diseases focus
    critical_diseases_panel = gap_df[gap_df['research_gap_score'] > 80].nlargest(8, 'dalys_millions')
    
    if len(critical_diseases_panel) > 0:
        bars = ax_equity4.barh(range(len(critical_diseases_panel)), critical_diseases_panel['dalys_millions'],
                              color='#d63031', alpha=0.8, edgecolor='black')
        
        ax_equity4.set_yticks(range(len(critical_diseases_panel)))
        ax_equity4.set_yticklabels([name[:12] + '...' if len(name) > 12 else name 
                                   for name in critical_diseases_panel['disease_subcategory']], fontsize=7)
        ax_equity4.set_xlabel('Disease Burden (M DALYs)', fontweight='bold', fontsize=10)
        ax_equity4.set_title('D. Critical Gaps Detail\n(25 Diseases)', fontweight='bold', fontsize=12)
        ax_equity4.invert_yaxis()
        
        # Add publication counts
        for i, (bar, pubs) in enumerate(zip(bars, critical_diseases_panel['publications_count'])):
            ax_equity4.text(bar.get_width() + 1, i, f'{pubs}', 
                           va='center', fontweight='bold', fontsize=7)
    
    # Overall title
    fig.suptitle('RESEARCH GAPS IN GLOBAL HEALTH: 25-DISEASE PRIORITY AREAS & EQUITY ANALYSIS', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Save files
    png_file = f'{output_prefix}.png'
    pdf_file = f'{output_prefix}.pdf'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"Complete labeled visualization saved: {png_file}, {pdf_file}")
    
    plt.show()
    return fig

def main():
    """
    Main function to generate the complete labeled visualization with fixed positioning.
    """
    print("Creating Complete Labeled Research Gap Visualization (Fixed Positioning)")
    print("=" * 75)
    
    # Load data
    gap_df, effort_df, summary_data = load_research_gap_data()
    
    if gap_df is None:
        print("Could not load data files. Please run 01-00-research-gap-discovery.py first.")
        return
    
    # Print data summary
    critical_count = len(gap_df[gap_df['research_gap_score'] > 80])
    print(f"\nDATA SUMMARY:")
    print(f"   • Total diseases: {len(gap_df)}")
    print(f"   • Critical gaps: {critical_count}")
    print(f"   • Total DALYs: {gap_df['dalys_millions'].sum():.1f}M")
    print(f"   • Total publications: {gap_df['publications_count'].sum():,}")
    
    # Create output directory
    output_dir = Path('./ANALYSIS/01-01-DATA-VIZ/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving figure to: {output_dir}")
    
    # Generate complete visualization with all diseases labeled and fixed positioning
    output_prefix = str(output_dir / 'research_gap_complete_labeled_fixed')
    
    print("\nCreating complete visualization with fixed label positioning...")
    create_complete_labeled_visualization(gap_df, summary_data, output_prefix)
    
    print(f"\nVISUALIZATION COMPLETE!")
    print(f"File saved: research_gap_complete_labeled_fixed.png")
    print(f"Features:")
    print(f"   • All {len(gap_df)} diseases labeled in main plot")
    print(f"   • RED DOT LABELS positioned much closer to bubbles")
    print(f"   • NO connecting lines (clean labels)")
    print(f"   • Complete A, B, C, D panels included")
    print(f"   • Box colors match bubble colors")

if __name__ == "__main__":
    main()