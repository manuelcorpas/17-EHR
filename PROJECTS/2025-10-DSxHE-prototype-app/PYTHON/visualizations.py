"""
HEIM Visualizations Module

Interactive charts and maps for displaying diversity metrics.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
from reference_data import ANCESTRY_NAMES, get_reference_ancestry


def create_ancestry_pie_chart(df: pd.DataFrame, title: str = "Ancestry Distribution") -> go.Figure:
    """
    Create interactive pie chart showing ancestry distribution.
    
    Args:
        df: DataFrame with ancestry column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Get ancestry counts
    ancestry_counts = df['ancestry'].value_counts()
    
    # Map codes to full names
    labels = [ANCESTRY_NAMES.get(code, code) for code in ancestry_counts.index]
    values = ancestry_counts.values
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,  # Donut chart
        textinfo='label+percent',
        textposition='auto',
        marker=dict(
            colors=px.colors.qualitative.Set2,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def create_ancestry_comparison_chart(df: pd.DataFrame, reference_type: str = 'coarse') -> go.Figure:
    """
    Create bar chart comparing observed vs expected ancestry distribution.
    
    Args:
        df: DataFrame with ancestry column
        reference_type: Type of reference population
        
    Returns:
        Plotly figure
    """
    # Get observed proportions
    ancestry_counts = df['ancestry'].value_counts()
    total = len(df)
    observed_props = (ancestry_counts / total * 100).to_dict()
    
    # Get expected proportions
    reference = get_reference_ancestry(reference_type)
    expected_props = {k: v * 100 for k, v in reference.items()}
    
    # Combine all ancestry codes
    all_codes = set(list(observed_props.keys()) + list(expected_props.keys()))
    
    # Prepare data
    codes = sorted(all_codes)
    observed_values = [observed_props.get(code, 0) for code in codes]
    expected_values = [expected_props.get(code, 0) for code in codes]
    labels = [ANCESTRY_NAMES.get(code, code) for code in codes]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Dataset',
        x=labels,
        y=observed_values,
        marker_color='#1f77b4',
        text=[f'{v:.1f}%' for v in observed_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Your Dataset: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Global Reference',
        x=labels,
        y=expected_values,
        marker_color='#ff7f0e',
        text=[f'{v:.1f}%' for v in expected_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Reference: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Ancestry Distribution: Your Dataset vs Global Reference',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Ancestry',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'}
    )
    
    return fig


def create_geographic_map(df: pd.DataFrame) -> go.Figure:
    """
    Create world map showing participant distribution by country.
    
    Args:
        df: DataFrame with country column
        
    Returns:
        Plotly figure
    """
    # Get country counts
    country_counts = df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    # Create choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=country_counts['country'],
        locationmode='country names',
        z=country_counts['count'],
        text=country_counts['country'],
        colorscale='Blues',
        colorbar=dict(
            title='Participants',
            thickness=15,
            len=0.5,
            bgcolor='rgba(255,255,255,0.1)',
            tickfont={'color': 'white'},
            titlefont={'color': 'white'}
        ),
        hovertemplate='<b>%{text}</b><br>Participants: %{z}<extra></extra>',
        marker_line_color='white',
        marker_line_width=0.5
    ))
    
    fig.update_layout(
        title={
            'text': 'Geographic Distribution of Participants',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)',
            landcolor='rgba(50,50,50,0.5)',
            coastlinecolor='white',
            countrycolor='rgba(255,255,255,0.2)'
        ),
        height=500,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def create_age_histogram(df: pd.DataFrame, reference_mean: float = 50.0) -> go.Figure:
    """
    Create age distribution histogram with reference overlay.
    
    Args:
        df: DataFrame with age column
        reference_mean: Reference population mean age
        
    Returns:
        Plotly figure
    """
    ages = df['age'].dropna()
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ages,
        nbinsx=20,
        name='Your Dataset',
        marker_color='#1f77b4',
        opacity=0.7,
        hovertemplate='Age Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add reference line
    fig.add_vline(
        x=reference_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Global Mean: {reference_mean}",
        annotation_position="top right",
        annotation_font_color="red"
    )
    
    # Add dataset mean line
    dataset_mean = ages.mean()
    fig.add_vline(
        x=dataset_mean,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Your Mean: {dataset_mean:.1f}",
        annotation_position="top left",
        annotation_font_color="green"
    )
    
    fig.update_layout(
        title={
            'text': 'Age Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Age (years)',
        yaxis_title='Number of Participants',
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'}
    )
    
    return fig


def create_sex_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing sex/gender distribution.
    
    Args:
        df: DataFrame with sex column
        
    Returns:
        Plotly figure
    """
    # Normalize sex values
    sex_normalized = df['sex'].str.upper().replace({
        'MALE': 'M',
        'FEMALE': 'F',
        'UNKNOWN': 'U',
        'OTHER': 'O'
    })
    
    sex_counts = sex_normalized.value_counts()
    
    # Full labels
    label_map = {'M': 'Male', 'F': 'Female', 'U': 'Unknown', 'O': 'Other'}
    labels = [label_map.get(code, code) for code in sex_counts.index]
    
    # Calculate percentages
    percentages = (sex_counts.values / len(df) * 100)
    
    # Color by category
    colors = {'Male': '#3498db', 'Female': '#e74c3c', 'Unknown': '#95a5a6', 'Other': '#9b59b6'}
    bar_colors = [colors.get(label, '#95a5a6') for label in labels]
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=sex_counts.values,
        text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(sex_counts.values, percentages)],
        textposition='outside',
        marker_color=bar_colors,
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Sex/Gender Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Sex/Gender',
        yaxis_title='Number of Participants',
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'}
    )
    
    return fig


def create_diversity_radar_chart(heim_result: Dict) -> go.Figure:
    """
    Create radar chart showing scores across all HEIM dimensions.
    
    Args:
        heim_result: HEIM scoring result dictionary
        
    Returns:
        Plotly figure
    """
    dimensions = heim_result['dimensions']
    
    categories = ['Ancestry<br>Diversity', 'Geographic<br>Diversity', 'Age<br>Distribution', 'Sex<br>Balance']
    scores = [
        dimensions['ancestry']['score'],
        dimensions['geographic']['score'],
        dimensions['age']['score'],
        dimensions['sex']['score']
    ]
    
    # Close the radar chart
    categories_closed = categories + [categories[0]]
    scores_closed = scores + [scores[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='#1f77b4', width=2),
        name='HEIM Scores',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>'
    ))
    
    # Add reference line at 75 (Gold threshold)
    fig.add_trace(go.Scatterpolar(
        r=[75, 75, 75, 75, 75],
        theta=categories_closed,
        mode='lines',
        line=dict(color='gold', width=1, dash='dash'),
        name='Gold Threshold',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='white',
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                color='white',
                gridcolor='rgba(255,255,255,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        title={
            'text': 'HEIM Dimension Scores',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font={'color': 'white'}
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def create_country_bar_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create horizontal bar chart showing top countries by participant count.
    
    Args:
        df: DataFrame with country column
        top_n: Number of top countries to show
        
    Returns:
        Plotly figure
    """
    country_counts = df['country'].value_counts().head(top_n)
    
    # Reverse order for horizontal bars (highest at top)
    country_counts = country_counts.iloc[::-1]
    
    fig = go.Figure(data=[go.Bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        marker_color='#2ecc71',
        text=country_counts.values,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Participants: %{x}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': f'Top {top_n} Countries by Participant Count',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Number of Participants',
        yaxis_title='Country',
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'color': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'color': 'white'}
    )
    
    return fig