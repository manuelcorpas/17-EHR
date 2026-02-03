"""
HEIM PDF Export Module

Generates professional PDF scorecards from HEIM analysis results.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import pandas as pd
from datetime import datetime
from typing import Dict, List
import io
from pathlib import Path


def get_badge_color(badge: str) -> colors.Color:
    """Get ReportLab color for badge level."""
    badge_colors = {
        'Platinum': colors.Color(0.898, 0.894, 0.886),  # Light gray
        'Gold': colors.Color(1.0, 0.843, 0.0),  # Gold
        'Silver': colors.Color(0.753, 0.753, 0.753),  # Silver
        'Bronze': colors.Color(0.804, 0.498, 0.196),  # Bronze
        'Needs Improvement': colors.Color(0.5, 0.5, 0.5)  # Gray
    }
    return badge_colors.get(badge, colors.grey)


def get_score_color(score: float) -> colors.Color:
    """Get ReportLab color for score visualization."""
    if score >= 90:
        return colors.Color(0.0, 0.784, 0.318)  # Green
    elif score >= 75:
        return colors.Color(1.0, 0.843, 0.0)  # Gold
    elif score >= 60:
        return colors.Color(0.753, 0.753, 0.753)  # Silver
    elif score >= 40:
        return colors.Color(0.804, 0.498, 0.196)  # Bronze
    else:
        return colors.Color(1.0, 0.267, 0.267)  # Red


def create_score_table(heim_result: Dict) -> Table:
    """Create table showing dimension scores."""
    dimensions = heim_result['dimensions']
    
    # Table data
    data = [
        ['Dimension', 'Score', 'Weight', 'Status'],
        ['Ancestry Diversity', 
         f"{dimensions['ancestry']['score']:.1f}/100",
         f"{dimensions['ancestry']['weight']*100:.0f}%",
         '✓' if dimensions['ancestry']['score'] >= 60 else '⚠'],
        ['Geographic Diversity',
         f"{dimensions['geographic']['score']:.1f}/100",
         f"{dimensions['geographic']['weight']*100:.0f}%",
         '✓' if dimensions['geographic']['score'] >= 60 else '⚠'],
        ['Age Distribution',
         f"{dimensions['age']['score']:.1f}/100",
         f"{dimensions['age']['weight']*100:.0f}%",
         '✓' if dimensions['age']['score'] >= 60 else '⚠'],
        ['Sex Balance',
         f"{dimensions['sex']['score']:.1f}/100",
         f"{dimensions['sex']['weight']*100:.0f}%",
         '✓' if dimensions['sex']['score'] >= 60 else '⚠']
    ]
    
    # Create table
    table = Table(data, colWidths=[2.5*inch, 1.2*inch, 1*inch, 0.8*inch])
    
    # Style
    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.122, 0.467, 0.706)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Body
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    return table


def create_summary_table(df: pd.DataFrame) -> Table:
    """Create table with dataset summary statistics."""
    data = [
        ['Metric', 'Value'],
        ['Total Participants', str(len(df))],
        ['Unique Ancestries', str(df['ancestry'].nunique())],
        ['Countries Represented', str(df['country'].nunique())],
        ['Age Range', f"{df['age'].min():.0f} - {df['age'].max():.0f} years"],
        ['Mean Age', f"{df['age'].mean():.1f} years"],
        ['Sex Distribution', f"F: {(df['sex'].str.upper().isin(['F', 'FEMALE']).sum())} / M: {(df['sex'].str.upper().isin(['M', 'MALE']).sum())}"]
    ]
    
    table = Table(data, colWidths=[2.5*inch, 2*inch])
    
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    return table


def create_recommendations_table(recommendations: Dict) -> List:
    """Create formatted recommendations section."""
    elements = []
    
    # Get priority actions
    priority_actions = recommendations.get('priority_actions', [])
    
    if priority_actions:
        data = [['Priority', 'Issue', 'Severity']]
        
        for i, action in enumerate(priority_actions[:5], 1):
            severity_icon = {
                'critical': '🚨',
                'high': '⚠️',
                'moderate': '⚡',
                'low': '💡'
            }.get(action['severity'], '•')
            
            data.append([
                str(i),
                action['title'][:60] + '...' if len(action['title']) > 60 else action['title'],
                f"{severity_icon} {action['severity'].title()}"
            ])
        
        table = Table(data, colWidths=[0.5*inch, 3.5*inch, 1.5*inch])
        
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.8, 0.2, 0.2)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            
            ('BACKGROUND', (0, 1), (-1, -1), colors.Color(1.0, 0.95, 0.95)),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
    
    return elements


def generate_pdf_scorecard(
    df: pd.DataFrame,
    heim_result: Dict,
    recommendations: Dict,
    output_path: Path,
    dataset_name: str = "Dataset"
) -> Path:
    """
    Generate comprehensive PDF scorecard.
    
    Args:
        df: Dataset DataFrame
        heim_result: HEIM scoring results
        recommendations: Recommendations dictionary
        output_path: Path to save PDF
        dataset_name: Name of the dataset
        
    Returns:
        Path to generated PDF
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.Color(0.122, 0.467, 0.706),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.Color(0.122, 0.467, 0.706),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    # Title
    elements.append(Paragraph("🧬 HEIM Dataset Equity Scorecard", title_style))
    elements.append(Paragraph(
        f"Health Equity Informative Marker Assessment",
        subtitle_style
    ))
    
    # Horizontal rule
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.Color(0.122, 0.467, 0.706)))
    elements.append(Spacer(1, 0.2*inch))
    
    # Dataset information
    elements.append(Paragraph(f"<b>Dataset:</b> {dataset_name}", body_style))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    elements.append(Paragraph(f"<b>Report Version:</b> HEIM v4.0", body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Overall Score - Large and prominent
    badge = heim_result['badge']
    score = heim_result['overall_score']
    badge_color = get_badge_color(badge)
    
    score_data = [[
        Paragraph(f"<font size=36 color='#{int(badge_color.red*255):02x}{int(badge_color.green*255):02x}{int(badge_color.blue*255):02x}'><b>{score:.1f}</b></font>", styles['Normal']),
        Paragraph(f"<font size=18><b>{badge}</b></font><br/><font size=10>Badge Level</font>", styles['Normal'])
    ]]
    
    score_table = Table(score_data, colWidths=[2*inch, 3*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.95, 0.95, 1.0)),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('ALIGN', (1, 0), (1, 0), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 2, colors.Color(0.122, 0.467, 0.706)),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    
    elements.append(score_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Interpretation
    from scoring import get_badge_interpretation
    interpretation = get_badge_interpretation(badge)
    elements.append(Paragraph(f"<i>{interpretation}</i>", body_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Dataset Summary
    elements.append(Paragraph("Dataset Summary", heading_style))
    elements.append(create_summary_table(df))
    elements.append(Spacer(1, 0.3*inch))
    
    # Dimension Scores
    elements.append(Paragraph("HEIM Dimension Scores", heading_style))
    elements.append(create_score_table(heim_result))
    elements.append(Spacer(1, 0.3*inch))
    
    # Key Findings
    elements.append(Paragraph("Key Findings", heading_style))
    
    dimensions = heim_result['dimensions']
    findings = []
    
    # Ancestry findings
    anc_score = dimensions['ancestry']['score']
    anc_details = dimensions['ancestry']['details']
    if anc_score >= 75:
        findings.append(f"✓ <b>Strong ancestry diversity</b> ({anc_details.get('unique_ancestries', 0)} groups represented)")
    elif anc_score >= 60:
        findings.append(f"⚡ <b>Moderate ancestry diversity</b> with room for improvement")
    else:
        findings.append(f"⚠ <b>Limited ancestry diversity</b> - significant gaps detected")
    
    # Geographic findings
    geo_score = dimensions['geographic']['score']
    geo_details = dimensions['geographic']['details']
    countries = geo_details.get('unique_countries', 0)
    if countries >= 10:
        findings.append(f"✓ <b>Good geographic spread</b> ({countries} countries)")
    else:
        findings.append(f"⚠ <b>Limited geographic diversity</b> ({countries} countries)")
    
    # Age findings
    age_details = dimensions['age']['details']
    age_range = age_details.get('range', 0)
    if age_range >= 40:
        findings.append(f"✓ <b>Wide age range</b> ({age_range:.0f} years)")
    else:
        findings.append(f"⚠ <b>Narrow age range</b> ({age_range:.0f} years)")
    
    # Sex findings
    sex_score = dimensions['sex']['score']
    if sex_score >= 70:
        findings.append(f"✓ <b>Balanced sex distribution</b>")
    else:
        findings.append(f"⚠ <b>Sex imbalance detected</b>")
    
    for finding in findings:
        elements.append(Paragraph(f"• {finding}", body_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Priority Recommendations
    if recommendations.get('priority_actions'):
        elements.append(Paragraph("Priority Recommendations", heading_style))
        elements.append(Paragraph(
            f"<i>{recommendations['overall_message']}</i>",
            body_style
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        rec_elements = create_recommendations_table(recommendations)
        elements.extend(rec_elements)
        elements.append(Spacer(1, 0.2*inch))
    
    # Footer note
    elements.append(Spacer(1, 0.3*inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<i>Generated by HEIM Assessor v4.0 | For detailed recommendations, please refer to the full report | "
        "Dr. Manuel Corpas, Alan Turing Institute</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))
    
    # Build PDF
    doc.build(elements)
    
    return output_path


def generate_detailed_pdf_report(
    df: pd.DataFrame,
    heim_result: Dict,
    recommendations: Dict,
    output_path: Path,
    dataset_name: str = "Dataset"
) -> Path:
    """
    Generate detailed multi-page PDF report with all recommendations.
    
    This version includes full recommendation details on additional pages.
    """
    # For now, use the single-page version
    # Can be extended to multi-page in future
    return generate_pdf_scorecard(df, heim_result, recommendations, output_path, dataset_name)