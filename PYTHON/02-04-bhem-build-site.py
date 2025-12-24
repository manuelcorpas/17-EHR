#!/usr/bin/env python3
"""
02-04-bhem-build-site.py
========================
HEIM-Biobank v1.0: Build Static Dashboard Site

Generates a complete static HTML/CSS/JS dashboard for GitHub Pages deployment.

OUTPUT (docs/):
    index.html      - Main dashboard with 8 tabs
    css/style.css   - Responsive styling
    js/app.js       - Tab navigation and data loading
    js/charts.js    - Chart.js visualizations
    .nojekyll       - Bypass Jekyll processing

DASHBOARD TABS:
    1. Overview   - Global statistics, key metrics, critical gaps
    2. Biobanks   - Per-biobank details, sortable table
    3. Diseases   - Per-disease analysis, gap severity
    4. Matrix     - Biobank × Disease heatmap
    5. Trends     - Publication trends over time
    6. Themes     - MeSH research theme analysis
    7. Compare    - Side-by-side biobank comparison
    8. Equity     - HIC vs LMIC analysis

USAGE:
    python 02-04-bhem-build-site.py

VERSION: HEIM-Biobank v1.0
DATE: 2025-12-24
"""

import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VERSION = "HEIM-Biobank v1.0"
VERSION_DATE = "2025-12-24"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HEIM-Biobank v1.0 | Global Equity Index for Biobank Research</title>
    <meta name="description" content="Health Equity Informative Metrics for Biobank research - measuring alignment between research output and global disease burden">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    
    <!-- Styles -->
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <!-- Header Banner -->
    <header class="header">
        <div class="header-content">
            <div class="header-title">
                <h1>HEIM-Biobank <span class="version">v1.0</span></h1>
                <p class="tagline">Global equity index for biobank-linked research vs disease burden</p>
            </div>
            <div class="header-meta">
                <span class="framework-badge">Part of HEIM Framework</span>
            </div>
        </div>
    </header>
    
    <!-- Navigation Tabs -->
    <nav class="nav-tabs">
        <button class="tab-btn active" data-tab="overview">Overview</button>
        <button class="tab-btn" data-tab="biobanks">Biobanks</button>
        <button class="tab-btn" data-tab="diseases">Diseases</button>
        <button class="tab-btn" data-tab="matrix">Matrix</button>
        <button class="tab-btn" data-tab="trends">Trends</button>
        <button class="tab-btn" data-tab="themes">Themes</button>
        <button class="tab-btn" data-tab="compare">Compare</button>
        <button class="tab-btn" data-tab="equity">Equity</button>
    </nav>
    
    <!-- Main Content -->
    <main class="main-content">
        
        <!-- Tab 1: Overview -->
        <section id="overview" class="tab-content active">
            <div class="section-header">
                <h2>Global Overview</h2>
                <p>Key metrics and critical research gaps across all biobanks</p>
            </div>
            
            <!-- Summary Cards -->
            <div class="summary-cards">
                <div class="card summary-card">
                    <div class="card-icon">🏛️</div>
                    <div class="card-value" id="stat-biobanks">--</div>
                    <div class="card-label">Biobanks Scored</div>
                </div>
                <div class="card summary-card">
                    <div class="card-icon">📚</div>
                    <div class="card-value" id="stat-publications">--</div>
                    <div class="card-label">Publications Mapped</div>
                </div>
                <div class="card summary-card">
                    <div class="card-icon">🌍</div>
                    <div class="card-value" id="stat-countries">--</div>
                    <div class="card-label">Countries Represented</div>
                </div>
                <div class="card summary-card">
                    <div class="card-icon">⚠️</div>
                    <div class="card-value" id="stat-critical">--</div>
                    <div class="card-label">Critical Gap Diseases</div>
                </div>
            </div>
            
            <!-- Charts Row -->
            <div class="charts-row">
                <div class="card chart-card">
                    <h3>Equity Alignment Distribution</h3>
                    <div class="chart-container">
                        <canvas id="chart-eas-distribution"></canvas>
                    </div>
                </div>
                <div class="card chart-card">
                    <h3>Critical Research Gaps</h3>
                    <div class="chart-container">
                        <canvas id="chart-critical-gaps"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Top Biobanks Table -->
            <div class="card">
                <h3>Top Biobanks by Equity Alignment</h3>
                <div class="table-container">
                    <table class="data-table" id="table-top-biobanks">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Biobank</th>
                                <th>EAS</th>
                                <th>Category</th>
                                <th>Publications</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
            
            <!-- Methodology Note -->
            <div class="card info-card">
                <h3>About HEIM-Biobank</h3>
                <p>The <strong>Equity Alignment Score (0-100)</strong> measures how well a biobank's research portfolio 
                matches global disease burden. Higher scores indicate better alignment between research output and 
                health needs.</p>
                <p>The <strong>Research Gap Score (0-100)</strong> quantifies the mismatch between a disease's global 
                burden and research attention. Categories: <span class="badge badge-critical">Critical (&gt;70)</span> 
                <span class="badge badge-high">High (50-70)</span> 
                <span class="badge badge-moderate">Moderate (30-50)</span> 
                <span class="badge badge-low">Low (&lt;30)</span></p>
                <p class="methodology-source">Methodology: Corpas et al. (2025), Annual Review of Biomedical Data Science</p>
            </div>
        </section>
        
        <!-- Tab 2: Biobanks -->
        <section id="biobanks" class="tab-content">
            <div class="section-header">
                <h2>Biobank Analysis</h2>
                <p>Detailed equity metrics for each biobank in the registry</p>
            </div>
            
            <!-- Filters -->
            <div class="filters">
                <input type="text" id="biobank-search" class="search-input" placeholder="Search biobanks...">
                <select id="biobank-region-filter" class="filter-select">
                    <option value="">All Regions</option>
                    <option value="AFR">Africa</option>
                    <option value="AMR">Americas</option>
                    <option value="EMR">Eastern Mediterranean</option>
                    <option value="EUR">Europe</option>
                    <option value="SEAR">South-East Asia</option>
                    <option value="WPR">Western Pacific</option>
                </select>
                <select id="biobank-category-filter" class="filter-select">
                    <option value="">All Categories</option>
                    <option value="Strong">Strong Alignment</option>
                    <option value="Moderate">Moderate Alignment</option>
                    <option value="Weak">Weak Alignment</option>
                    <option value="Poor">Poor Alignment</option>
                </select>
            </div>
            
            <!-- Biobanks Table -->
            <div class="card">
                <div class="table-container">
                    <table class="data-table sortable" id="table-biobanks">
                        <thead>
                            <tr>
                                <th data-sort="name">Biobank ↕</th>
                                <th data-sort="country">Country ↕</th>
                                <th data-sort="region">Region ↕</th>
                                <th data-sort="eas">EAS ↕</th>
                                <th data-sort="category">Category ↕</th>
                                <th data-sort="publications">Publications ↕</th>
                                <th data-sort="diseases">Diseases ↕</th>
                                <th data-sort="gaps">Critical Gaps ↕</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
            
            <!-- Download Button -->
            <div class="actions">
                <button class="btn btn-primary" onclick="downloadBiobanksCSV()">Download CSV</button>
            </div>
        </section>
        
        <!-- Tab 3: Diseases -->
        <section id="diseases" class="tab-content">
            <div class="section-header">
                <h2>Disease Analysis</h2>
                <p>Research gaps by disease relative to global burden</p>
            </div>
            
            <!-- Filters -->
            <div class="filters">
                <input type="text" id="disease-search" class="search-input" placeholder="Search diseases...">
                <select id="disease-category-filter" class="filter-select">
                    <option value="">All Categories</option>
                    <option value="Cardiovascular">Cardiovascular</option>
                    <option value="Respiratory">Respiratory</option>
                    <option value="Metabolic">Metabolic</option>
                    <option value="Infectious">Infectious</option>
                    <option value="Neglected">Neglected</option>
                    <option value="Neurological">Neurological</option>
                    <option value="Mental Health">Mental Health</option>
                    <option value="Cancer">Cancer</option>
                    <option value="Maternal/Child">Maternal/Child</option>
                    <option value="Injuries">Injuries</option>
                </select>
                <select id="disease-severity-filter" class="filter-select">
                    <option value="">All Severities</option>
                    <option value="Critical">Critical</option>
                    <option value="High">High</option>
                    <option value="Moderate">Moderate</option>
                    <option value="Low">Low</option>
                </select>
            </div>
            
            <!-- Disease Chart -->
            <div class="card chart-card">
                <h3>Research Gap vs Disease Burden</h3>
                <div class="chart-container chart-large">
                    <canvas id="chart-disease-burden"></canvas>
                </div>
            </div>
            
            <!-- Diseases Table -->
            <div class="card">
                <div class="table-container">
                    <table class="data-table sortable" id="table-diseases">
                        <thead>
                            <tr>
                                <th data-sort="name">Disease ↕</th>
                                <th data-sort="category">Category ↕</th>
                                <th data-sort="dalys">DALYs (M) ↕</th>
                                <th data-sort="publications">Publications ↕</th>
                                <th data-sort="gap">Gap Score ↕</th>
                                <th data-sort="severity">Severity ↕</th>
                                <th data-sort="biobanks">Biobanks ↕</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>
        
        <!-- Tab 4: Matrix -->
        <section id="matrix" class="tab-content">
            <div class="section-header">
                <h2>Publication Matrix</h2>
                <p>Biobank × Disease publication counts heatmap</p>
            </div>
            
            <div class="card">
                <div class="matrix-legend">
                    <span class="legend-item"><span class="legend-color" style="background:#dc3545"></span> Critical (0 pubs)</span>
                    <span class="legend-item"><span class="legend-color" style="background:#fd7e14"></span> High (1-2 pubs)</span>
                    <span class="legend-item"><span class="legend-color" style="background:#ffc107"></span> Moderate (3-10 pubs)</span>
                    <span class="legend-item"><span class="legend-color" style="background:#28a745"></span> Low (&gt;10 pubs)</span>
                </div>
                <div class="matrix-container" id="matrix-container">
                    <!-- Matrix rendered by JS -->
                    <p class="loading">Loading matrix data...</p>
                </div>
            </div>
        </section>
        
        <!-- Tab 5: Trends -->
        <section id="trends" class="tab-content">
            <div class="section-header">
                <h2>Publication Trends</h2>
                <p>How biobank research has grown over time</p>
            </div>
            
            <div class="card chart-card">
                <h3>Global Publication Growth</h3>
                <div class="chart-container chart-large">
                    <canvas id="chart-trends-global"></canvas>
                </div>
            </div>
            
            <div class="card chart-card">
                <h3>Trends by Biobank</h3>
                <div class="filters">
                    <select id="trends-biobank-select" class="filter-select">
                        <option value="">Select biobank...</option>
                    </select>
                </div>
                <div class="chart-container chart-large">
                    <canvas id="chart-trends-biobank"></canvas>
                </div>
            </div>
        </section>
        
        <!-- Tab 6: Themes -->
        <section id="themes" class="tab-content">
            <div class="section-header">
                <h2>Research Themes</h2>
                <p>MeSH-based thematic analysis of biobank research</p>
            </div>
            
            <div class="charts-row">
                <div class="card chart-card">
                    <h3>Theme Distribution</h3>
                    <div class="chart-container">
                        <canvas id="chart-theme-dist"></canvas>
                    </div>
                </div>
                <div class="card chart-card">
                    <h3>Publications by Theme</h3>
                    <div class="chart-container">
                        <canvas id="chart-theme-pubs"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Theme Table -->
            <div class="card">
                <h3>Theme Coverage Across Biobanks</h3>
                <div class="table-container">
                    <table class="data-table" id="table-themes">
                        <thead>
                            <tr>
                                <th>Theme</th>
                                <th>Publications</th>
                                <th>Disease Count</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>
        
        <!-- Tab 7: Compare -->
        <section id="compare" class="tab-content">
            <div class="section-header">
                <h2>Compare Biobanks</h2>
                <p>Side-by-side comparison of biobank metrics</p>
            </div>
            
            <!-- Biobank Selectors -->
            <div class="compare-selectors">
                <div class="selector-group">
                    <label>Biobank 1:</label>
                    <select id="compare-biobank1" class="filter-select"></select>
                </div>
                <div class="selector-group">
                    <label>Biobank 2:</label>
                    <select id="compare-biobank2" class="filter-select"></select>
                </div>
            </div>
            
            <!-- Comparison Cards -->
            <div class="compare-cards">
                <div class="card compare-card" id="compare-card1">
                    <h3 id="compare-name1">Select a biobank</h3>
                    <div class="compare-stats" id="compare-stats1"></div>
                </div>
                <div class="card compare-card" id="compare-card2">
                    <h3 id="compare-name2">Select a biobank</h3>
                    <div class="compare-stats" id="compare-stats2"></div>
                </div>
            </div>
            
            <!-- Radar Chart -->
            <div class="card chart-card">
                <h3>Metric Comparison</h3>
                <div class="chart-container">
                    <canvas id="chart-compare-radar"></canvas>
                </div>
            </div>
            
            <!-- Similar Pairs -->
            <div class="card">
                <h3>Similar Biobank Pairs</h3>
                <div class="table-container">
                    <table class="data-table" id="table-similar">
                        <thead>
                            <tr>
                                <th>Biobank 1</th>
                                <th>Biobank 2</th>
                                <th>Similarity</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>
        
        <!-- Tab 8: Equity -->
        <section id="equity" class="tab-content">
            <div class="section-header">
                <h2>Global Health Equity</h2>
                <p>HIC vs LMIC research distribution analysis</p>
            </div>
            
            <!-- Equity Summary -->
            <div class="summary-cards">
                <div class="card summary-card">
                    <div class="card-value" id="equity-ratio">--</div>
                    <div class="card-label">Equity Ratio</div>
                    <div class="card-sublabel" id="equity-interpretation">--</div>
                </div>
                <div class="card summary-card hic-card">
                    <div class="card-value" id="hic-biobanks">--</div>
                    <div class="card-label">HIC Biobanks</div>
                    <div class="card-sublabel" id="hic-pubs">-- publications</div>
                </div>
                <div class="card summary-card lmic-card">
                    <div class="card-value" id="lmic-biobanks">--</div>
                    <div class="card-label">LMIC Biobanks</div>
                    <div class="card-sublabel" id="lmic-pubs">-- publications</div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="charts-row">
                <div class="card chart-card">
                    <h3>Publication Share: HIC vs LMIC</h3>
                    <div class="chart-container">
                        <canvas id="chart-equity-share"></canvas>
                    </div>
                </div>
                <div class="card chart-card">
                    <h3>Publications by WHO Region</h3>
                    <div class="chart-container">
                        <canvas id="chart-equity-region"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Global South Priority Diseases -->
            <div class="card">
                <h3>Global South Priority Diseases</h3>
                <p class="card-description">Diseases disproportionately affecting low- and middle-income countries</p>
                <div class="table-container">
                    <table class="data-table" id="table-gs-diseases">
                        <thead>
                            <tr>
                                <th>Disease</th>
                                <th>DALYs (M)</th>
                                <th>Publications</th>
                                <th>Gap Score</th>
                                <th>Severity</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>
        
    </main>
    
    <!-- Footer -->
    <footer class="footer">
        <div class="footer-content">
            <p>
                <strong>HEIM-Biobank v1.0</strong> | 
                Specification date: {version_date} | 
                Data: GBD 2021, PubMed | 
                Part of <a href="https://github.com/manuelcorpas/17-EHR" target="_blank">HEIM Framework</a>
            </p>
            <p class="footer-methodology">
                Methodology: Corpas et al. (2025), Annual Review of Biomedical Data Science
            </p>
        </div>
    </footer>
    
    <!-- Scripts -->
    <script src="js/app.js"></script>
    <script src="js/charts.js"></script>
</body>
</html>
'''

# =============================================================================
# CSS STYLESHEET
# =============================================================================

CSS_TEMPLATE = '''/* HEIM-Biobank v1.0 Dashboard Styles */

:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary: #64748b;
    --success: #28a745;
    --warning: #ffc107;
    --danger: #dc3545;
    --info: #17a2b8;
    
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-dark: #1e293b;
    
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-light: #94a3b8;
    
    --border: #e2e8f0;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
    --shadow-lg: 0 4px 6px rgba(0,0,0,0.1);
    
    --critical: #dc3545;
    --high: #fd7e14;
    --moderate: #ffc107;
    --low: #28a745;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

/* Header */
.header {
    background: linear-gradient(135deg, var(--bg-dark) 0%, #334155 100%);
    color: white;
    padding: 1.5rem 2rem;
}

.header-content {
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title h1 {
    font-size: 1.75rem;
    font-weight: 700;
}

.header-title .version {
    font-weight: 400;
    opacity: 0.8;
}

.header-title .tagline {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.25rem;
}

.framework-badge {
    background: rgba(255,255,255,0.1);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
}

/* Navigation Tabs */
.nav-tabs {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    display: flex;
    gap: 0;
    max-width: 100%;
    overflow-x: auto;
}

.tab-btn {
    background: none;
    border: none;
    padding: 1rem 1.5rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
}

.tab-btn:hover {
    color: var(--primary);
    background: rgba(37, 99, 235, 0.05);
}

.tab-btn.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
}

/* Main Content */
.main-content {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.section-header {
    margin-bottom: 1.5rem;
}

.section-header h2 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.section-header p {
    color: var(--text-secondary);
}

/* Cards */
.card {
    background: var(--bg-secondary);
    border-radius: 8px;
    box-shadow: var(--shadow);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.card h3 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

/* Summary Cards */
.summary-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.summary-card {
    text-align: center;
    padding: 1.5rem;
}

.summary-card .card-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.summary-card .card-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

.summary-card .card-label {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

.summary-card .card-sublabel {
    font-size: 0.8rem;
    color: var(--text-light);
}

.hic-card .card-value { color: var(--info); }
.lmic-card .card-value { color: var(--success); }

/* Charts */
.charts-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.chart-card {
    min-height: 350px;
}

.chart-container {
    position: relative;
    height: 280px;
}

.chart-container.chart-large {
    height: 400px;
}

/* Tables */
.table-container {
    overflow-x: auto;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.data-table th,
.data-table td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

.data-table th {
    background: var(--bg-primary);
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
}

.data-table th:hover {
    background: #e2e8f0;
}

.data-table tbody tr:hover {
    background: var(--bg-primary);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-critical { background: var(--critical); color: white; }
.badge-high { background: var(--high); color: white; }
.badge-moderate { background: var(--moderate); color: #000; }
.badge-low { background: var(--low); color: white; }
.badge-strong { background: var(--success); color: white; }
.badge-weak { background: var(--warning); color: #000; }
.badge-poor { background: var(--danger); color: white; }

/* Filters */
.filters {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.search-input,
.filter-select {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.9rem;
    background: var(--bg-secondary);
}

.search-input {
    min-width: 250px;
}

.filter-select {
    min-width: 150px;
}

/* Buttons */
.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary {
    background: var(--primary);
    color: white;
}

.btn-primary:hover {
    background: var(--primary-dark);
}

.actions {
    margin-top: 1rem;
}

/* Matrix */
.matrix-container {
    overflow: auto;
    max-height: 600px;
}

.matrix-legend {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
}

.legend-color {
    width: 16px;
    height: 16px;
    border-radius: 3px;
}

.matrix-table {
    border-collapse: collapse;
    font-size: 0.75rem;
}

.matrix-table th,
.matrix-table td {
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--border);
    text-align: center;
    min-width: 40px;
}

.matrix-table th {
    background: var(--bg-primary);
    font-weight: 500;
    position: sticky;
    top: 0;
}

.matrix-table th:first-child {
    left: 0;
    z-index: 2;
}

.matrix-table td:first-child {
    background: var(--bg-primary);
    font-weight: 500;
    position: sticky;
    left: 0;
}

.matrix-cell-critical { background: var(--critical); color: white; }
.matrix-cell-high { background: var(--high); color: white; }
.matrix-cell-moderate { background: var(--moderate); color: #000; }
.matrix-cell-low { background: var(--low); color: white; }

/* Compare */
.compare-selectors {
    display: flex;
    gap: 2rem;
    margin-bottom: 1.5rem;
}

.selector-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.selector-group label {
    font-weight: 500;
}

.compare-cards {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.compare-card h3 {
    font-size: 1.1rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
}

.compare-stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
}

.stat-item {
    padding: 0.5rem;
    background: var(--bg-primary);
    border-radius: 4px;
}

.stat-item .stat-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.stat-item .stat-value {
    font-size: 1.1rem;
    font-weight: 600;
}

/* Info Card */
.info-card {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border-left: 4px solid var(--primary);
}

.info-card p {
    margin-bottom: 0.75rem;
}

.methodology-source {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-style: italic;
}

.card-description {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* Footer */
.footer {
    background: var(--bg-dark);
    color: rgba(255,255,255,0.7);
    padding: 1.5rem 2rem;
    margin-top: 2rem;
}

.footer-content {
    max-width: 1400px;
    margin: 0 auto;
    text-align: center;
}

.footer a {
    color: var(--primary);
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

.footer-methodology {
    font-size: 0.85rem;
    margin-top: 0.5rem;
    opacity: 0.8;
}

/* Loading State */
.loading {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
}

/* Responsive */
@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        text-align: center;
        gap: 1rem;
    }
    
    .charts-row {
        grid-template-columns: 1fr;
    }
    
    .compare-cards {
        grid-template-columns: 1fr;
    }
    
    .compare-selectors {
        flex-direction: column;
    }
    
    .nav-tabs {
        padding: 0 1rem;
    }
    
    .tab-btn {
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
    }
    
    .main-content {
        padding: 1rem;
    }
}
'''

# =============================================================================
# JAVASCRIPT - APP.JS
# =============================================================================

APP_JS_TEMPLATE = '''// HEIM-Biobank v1.0 Dashboard Application

// Global data store
let DATA = {
    summary: null,
    biobanks: null,
    diseases: null,
    matrix: null,
    trends: null,
    themes: null,
    comparison: null,
    equity: null
};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('HEIM-Biobank v1.0 Dashboard initializing...');
    
    // Setup tab navigation
    setupTabs();
    
    // Load all data
    await loadAllData();
    
    // Render initial view
    renderOverview();
    
    // Setup filters and interactions
    setupFilters();
    
    console.log('Dashboard ready');
});

// Tab Navigation
function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active content
            const tabId = btn.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            // Render tab-specific content
            renderTab(tabId);
        });
    });
}

function renderTab(tabId) {
    switch(tabId) {
        case 'overview': renderOverview(); break;
        case 'biobanks': renderBiobanks(); break;
        case 'diseases': renderDiseases(); break;
        case 'matrix': renderMatrix(); break;
        case 'trends': renderTrends(); break;
        case 'themes': renderThemes(); break;
        case 'compare': renderCompare(); break;
        case 'equity': renderEquity(); break;
    }
}

// Data Loading
async function loadAllData() {
    const files = ['summary', 'biobanks', 'diseases', 'matrix', 'trends', 'themes', 'comparison', 'equity'];
    
    const promises = files.map(async (file) => {
        try {
            const response = await fetch(`data/${file}.json`);
            if (response.ok) {
                DATA[file] = await response.json();
                console.log(`Loaded ${file}.json`);
            } else {
                console.warn(`Failed to load ${file}.json`);
            }
        } catch (err) {
            console.warn(`Error loading ${file}.json:`, err);
        }
    });
    
    await Promise.all(promises);
}

// Overview Tab
function renderOverview() {
    if (!DATA.summary) return;
    
    const s = DATA.summary;
    
    // Update summary stats
    document.getElementById('stat-biobanks').textContent = s.overview?.totalBiobanks || '--';
    document.getElementById('stat-publications').textContent = formatNumber(s.overview?.totalPublications);
    document.getElementById('stat-countries').textContent = s.overview?.totalCountries || '--';
    document.getElementById('stat-critical').textContent = s.gapDistribution?.Critical || '--';
    
    // Render charts
    renderEASDistributionChart();
    renderCriticalGapsChart();
    
    // Render top biobanks table
    renderTopBiobanksTable();
}

function renderTopBiobanksTable() {
    const tbody = document.querySelector('#table-top-biobanks tbody');
    if (!tbody || !DATA.summary?.topBiobanks) return;
    
    tbody.innerHTML = DATA.summary.topBiobanks.map((b, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>${b.name}</td>
            <td>${b.eas?.toFixed(1) || '--'}</td>
            <td><span class="badge badge-${b.category?.toLowerCase().replace(' ', '-')}">${b.category}</span></td>
            <td>${formatNumber(b.publications)}</td>
        </tr>
    `).join('');
}

// Biobanks Tab
function renderBiobanks() {
    if (!DATA.biobanks?.biobanks) return;
    
    const tbody = document.querySelector('#table-biobanks tbody');
    if (!tbody) return;
    
    const biobanks = filterBiobanks(DATA.biobanks.biobanks);
    
    tbody.innerHTML = biobanks.map(b => `
        <tr>
            <td>${b.name}</td>
            <td>${b.country}</td>
            <td>${b.regionName}</td>
            <td>${b.scores?.equityAlignment?.toFixed(1) || '--'}</td>
            <td><span class="badge badge-${getCategoryClass(b.scores?.equityCategory)}">${b.scores?.equityCategory || '--'}</span></td>
            <td>${formatNumber(b.stats?.totalPublications)}</td>
            <td>${b.stats?.diseasesCovered || '--'}/25</td>
            <td>${b.stats?.criticalGaps || '--'}</td>
        </tr>
    `).join('');
}

function filterBiobanks(biobanks) {
    const search = document.getElementById('biobank-search')?.value.toLowerCase() || '';
    const region = document.getElementById('biobank-region-filter')?.value || '';
    const category = document.getElementById('biobank-category-filter')?.value || '';
    
    return biobanks.filter(b => {
        if (search && !b.name.toLowerCase().includes(search)) return false;
        if (region && b.region !== region) return false;
        if (category && b.scores?.equityCategory !== category) return false;
        return true;
    });
}

// Diseases Tab
function renderDiseases() {
    if (!DATA.diseases?.diseases) return;
    
    const tbody = document.querySelector('#table-diseases tbody');
    if (!tbody) return;
    
    const diseases = filterDiseases(DATA.diseases.diseases);
    
    tbody.innerHTML = diseases.map(d => `
        <tr>
            <td>${d.name}</td>
            <td>${d.category}</td>
            <td>${d.burden?.dalysMillions?.toFixed(1) || '--'}</td>
            <td>${formatNumber(d.research?.globalPublications)}</td>
            <td>${d.gap?.score?.toFixed(0) || '--'}</td>
            <td><span class="badge badge-${d.gap?.severity?.toLowerCase()}">${d.gap?.severity || '--'}</span></td>
            <td>${d.research?.biobanksEngaged || '--'}</td>
        </tr>
    `).join('');
    
    // Render burden chart
    renderDiseaseBurdenChart();
}

function filterDiseases(diseases) {
    const search = document.getElementById('disease-search')?.value.toLowerCase() || '';
    const category = document.getElementById('disease-category-filter')?.value || '';
    const severity = document.getElementById('disease-severity-filter')?.value || '';
    
    return diseases.filter(d => {
        if (search && !d.name.toLowerCase().includes(search)) return false;
        if (category && d.category !== category) return false;
        if (severity && d.gap?.severity !== severity) return false;
        return true;
    });
}

// Matrix Tab
function renderMatrix() {
    if (!DATA.matrix) return;
    
    const container = document.getElementById('matrix-container');
    if (!container) return;
    
    const m = DATA.matrix;
    
    // Build matrix table
    let html = '<table class="matrix-table">';
    
    // Header row
    html += '<tr><th></th>';
    m.diseases.forEach(d => {
        html += `<th title="${d.name}">${d.name.substring(0, 8)}</th>`;
    });
    html += '</tr>';
    
    // Data rows
    m.biobanks.forEach((b, bi) => {
        html += `<tr><td title="${b.name}">${b.name.substring(0, 15)}</td>`;
        m.matrix.values[bi].forEach((val, di) => {
            const cat = m.matrix.gapCategories[bi][di];
            html += `<td class="matrix-cell-${cat}" title="${b.name} / ${m.diseases[di].name}: ${val} pubs">${val}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</table>';
    container.innerHTML = html;
}

// Trends Tab
function renderTrends() {
    if (!DATA.trends) return;
    
    renderGlobalTrendsChart();
    populateTrendsBiobankSelect();
}

function populateTrendsBiobankSelect() {
    const select = document.getElementById('trends-biobank-select');
    if (!select || !DATA.trends?.byBiobank) return;
    
    const options = Object.entries(DATA.trends.byBiobank).map(([id, data]) => 
        `<option value="${id}">${data.name}</option>`
    );
    
    select.innerHTML = '<option value="">Select biobank...</option>' + options.join('');
    
    select.addEventListener('change', () => {
        const biobank = select.value;
        if (biobank) {
            renderBiobankTrendsChart(biobank);
        }
    });
}

// Themes Tab
function renderThemes() {
    if (!DATA.themes) return;
    
    renderThemeDistributionChart();
    renderThemePublicationsChart();
    renderThemesTable();
}

function renderThemesTable() {
    const tbody = document.querySelector('#table-themes tbody');
    if (!tbody || !DATA.themes?.themes) return;
    
    tbody.innerHTML = DATA.themes.themes.map(t => `
        <tr>
            <td>${t.name}</td>
            <td>${formatNumber(t.publications)}</td>
            <td>${t.diseaseCount || '--'}</td>
        </tr>
    `).join('');
}

// Compare Tab
function renderCompare() {
    if (!DATA.comparison) return;
    
    populateCompareSelects();
    renderSimilarPairsTable();
}

function populateCompareSelects() {
    const select1 = document.getElementById('compare-biobank1');
    const select2 = document.getElementById('compare-biobank2');
    if (!select1 || !select2 || !DATA.comparison?.biobanks) return;
    
    const options = DATA.comparison.biobanks.map(b => 
        `<option value="${b.id}">${b.name}</option>`
    );
    
    select1.innerHTML = '<option value="">Select biobank...</option>' + options.join('');
    select2.innerHTML = '<option value="">Select biobank...</option>' + options.join('');
    
    select1.addEventListener('change', updateComparison);
    select2.addEventListener('change', updateComparison);
}

function updateComparison() {
    const id1 = document.getElementById('compare-biobank1')?.value;
    const id2 = document.getElementById('compare-biobank2')?.value;
    
    if (id1) renderCompareCard(id1, 1);
    if (id2) renderCompareCard(id2, 2);
    if (id1 && id2) renderComparisonRadar(id1, id2);
}

function renderCompareCard(biobankId, cardNum) {
    const biobank = DATA.comparison?.biobanks?.find(b => b.id === biobankId);
    if (!biobank) return;
    
    document.getElementById(`compare-name${cardNum}`).textContent = biobank.name;
    
    const statsDiv = document.getElementById(`compare-stats${cardNum}`);
    statsDiv.innerHTML = `
        <div class="stat-item">
            <div class="stat-label">Publications</div>
            <div class="stat-value">${formatNumber(biobank.stats?.publications)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Diseases</div>
            <div class="stat-value">${biobank.stats?.diseases}/25</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">EAS</div>
            <div class="stat-value">${biobank.stats?.eas?.toFixed(1)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">ROS</div>
            <div class="stat-value">${biobank.stats?.ros?.toFixed(0)}</div>
        </div>
    `;
}

function renderSimilarPairsTable() {
    const tbody = document.querySelector('#table-similar tbody');
    if (!tbody || !DATA.comparison?.similarPairs) return;
    
    const biobanks = DATA.comparison.biobanks || [];
    const getName = (id) => biobanks.find(b => b.id === id)?.name || id;
    
    tbody.innerHTML = DATA.comparison.similarPairs.slice(0, 10).map(p => `
        <tr>
            <td>${getName(p.biobank1)}</td>
            <td>${getName(p.biobank2)}</td>
            <td>${p.similarity?.toFixed(1)}%</td>
        </tr>
    `).join('');
}

// Equity Tab
function renderEquity() {
    if (!DATA.equity) return;
    
    const e = DATA.equity;
    
    // Update summary
    document.getElementById('equity-ratio').textContent = e.equityRatio?.toFixed(2) || '--';
    document.getElementById('equity-interpretation').textContent = e.equityInterpretation || '--';
    document.getElementById('hic-biobanks').textContent = e.summary?.hic?.biobanks || '--';
    document.getElementById('hic-pubs').textContent = formatNumber(e.summary?.hic?.publications) + ' publications';
    document.getElementById('lmic-biobanks').textContent = e.summary?.lmic?.biobanks || '--';
    document.getElementById('lmic-pubs').textContent = formatNumber(e.summary?.lmic?.publications) + ' publications';
    
    // Render charts
    renderEquityShareChart();
    renderEquityRegionChart();
    
    // Render GS diseases table
    renderGSDiseasesTable();
}

function renderGSDiseasesTable() {
    const tbody = document.querySelector('#table-gs-diseases tbody');
    if (!tbody || !DATA.equity?.globalSouthDiseases) return;
    
    tbody.innerHTML = DATA.equity.globalSouthDiseases.map(d => `
        <tr>
            <td>${d.name}</td>
            <td>${d.dalys?.toFixed(1) || '--'}</td>
            <td>${formatNumber(d.publications)}</td>
            <td>${d.gapScore?.toFixed(0) || '--'}</td>
            <td><span class="badge badge-${d.severity?.toLowerCase()}">${d.severity || '--'}</span></td>
        </tr>
    `).join('');
}

// Filters Setup
function setupFilters() {
    // Biobank filters
    document.getElementById('biobank-search')?.addEventListener('input', renderBiobanks);
    document.getElementById('biobank-region-filter')?.addEventListener('change', renderBiobanks);
    document.getElementById('biobank-category-filter')?.addEventListener('change', renderBiobanks);
    
    // Disease filters
    document.getElementById('disease-search')?.addEventListener('input', renderDiseases);
    document.getElementById('disease-category-filter')?.addEventListener('change', renderDiseases);
    document.getElementById('disease-severity-filter')?.addEventListener('change', renderDiseases);
}

// Utility Functions
function formatNumber(num) {
    if (num === undefined || num === null) return '--';
    return num.toLocaleString();
}

function getCategoryClass(category) {
    if (!category) return '';
    const cat = category.toLowerCase();
    if (cat.includes('strong')) return 'strong';
    if (cat.includes('moderate')) return 'moderate';
    if (cat.includes('weak')) return 'weak';
    if (cat.includes('poor')) return 'poor';
    return cat;
}

// CSV Download
function downloadBiobanksCSV() {
    if (!DATA.biobanks?.biobanks) return;
    
    const headers = ['Name', 'Country', 'Region', 'EAS', 'Category', 'Publications', 'Diseases', 'Critical Gaps'];
    const rows = DATA.biobanks.biobanks.map(b => [
        b.name,
        b.country,
        b.region,
        b.scores?.equityAlignment?.toFixed(1),
        b.scores?.equityCategory,
        b.stats?.totalPublications,
        b.stats?.diseasesCovered,
        b.stats?.criticalGaps
    ]);
    
    let csv = headers.join(',') + '\\n';
    csv += rows.map(r => r.map(v => `"${v || ''}"`).join(',')).join('\\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'heim-biobank-data.csv';
    a.click();
    URL.revokeObjectURL(url);
}
'''

# =============================================================================
# JAVASCRIPT - CHARTS.JS
# =============================================================================

CHARTS_JS_TEMPLATE = '''// HEIM-Biobank v1.0 Chart Visualizations

// Color palettes
const COLORS = {
    primary: '#2563eb',
    critical: '#dc3545',
    high: '#fd7e14',
    moderate: '#ffc107',
    low: '#28a745',
    hic: '#17a2b8',
    lmic: '#28a745'
};

const CHART_COLORS = [
    '#2563eb', '#dc3545', '#28a745', '#ffc107', '#17a2b8',
    '#6f42c1', '#fd7e14', '#20c997', '#e83e8c', '#6c757d'
];

// Chart instances for cleanup
let chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

// EAS Distribution Chart (Overview)
function renderEASDistributionChart() {
    const canvas = document.getElementById('chart-eas-distribution');
    if (!canvas || !DATA.summary?.easDistribution) return;
    
    destroyChart('eas-dist');
    
    const dist = DATA.summary.easDistribution;
    const labels = ['Strong', 'Moderate', 'Weak', 'Poor'];
    const values = labels.map(l => dist[l] || 0);
    const colors = [COLORS.low, COLORS.moderate, COLORS.high, COLORS.critical];
    
    chartInstances['eas-dist'] = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// Critical Gaps Chart (Overview)
function renderCriticalGapsChart() {
    const canvas = document.getElementById('chart-critical-gaps');
    if (!canvas || !DATA.summary?.criticalGaps) return;
    
    destroyChart('critical-gaps');
    
    const gaps = DATA.summary.criticalGaps.slice(0, 8);
    
    chartInstances['critical-gaps'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: gaps.map(g => g.name.substring(0, 15)),
            datasets: [{
                label: 'Gap Score',
                data: gaps.map(g => g.gapScore),
                backgroundColor: COLORS.critical
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { 
                    max: 100,
                    title: { display: true, text: 'Gap Score' }
                }
            }
        }
    });
}

// Disease Burden Chart (Diseases Tab)
function renderDiseaseBurdenChart() {
    const canvas = document.getElementById('chart-disease-burden');
    if (!canvas || !DATA.diseases?.diseases) return;
    
    destroyChart('disease-burden');
    
    const diseases = DATA.diseases.diseases.slice(0, 15);
    
    chartInstances['disease-burden'] = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Diseases',
                data: diseases.map(d => ({
                    x: d.burden?.dalysMillions || 0,
                    y: d.gap?.score || 0
                })),
                backgroundColor: diseases.map(d => {
                    const sev = d.gap?.severity?.toLowerCase();
                    return COLORS[sev] || COLORS.primary;
                }),
                pointRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const d = diseases[ctx.dataIndex];
                            return `${d.name}: ${d.burden?.dalysMillions?.toFixed(1)}M DALYs, Gap: ${d.gap?.score?.toFixed(0)}`;
                        }
                    }
                }
            },
            scales: {
                x: { 
                    title: { display: true, text: 'Disease Burden (Million DALYs)' }
                },
                y: { 
                    title: { display: true, text: 'Research Gap Score' },
                    max: 100
                }
            }
        }
    });
}

// Global Trends Chart (Trends Tab)
function renderGlobalTrendsChart() {
    const canvas = document.getElementById('chart-trends-global');
    if (!canvas || !DATA.trends?.global?.yearly) return;
    
    destroyChart('trends-global');
    
    const yearly = DATA.trends.global.yearly;
    const years = Object.keys(yearly).sort();
    const values = years.map(y => yearly[y]);
    
    chartInstances['trends-global'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Publications',
                data: values,
                borderColor: COLORS.primary,
                backgroundColor: COLORS.primary + '20',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    title: { display: true, text: 'Publications' }
                }
            }
        }
    });
}

// Biobank Trends Chart
function renderBiobankTrendsChart(biobankId) {
    const canvas = document.getElementById('chart-trends-biobank');
    if (!canvas || !DATA.trends?.byBiobank?.[biobankId]) return;
    
    destroyChart('trends-biobank');
    
    const biobank = DATA.trends.byBiobank[biobankId];
    const yearly = biobank.yearly || {};
    const years = Object.keys(yearly).sort();
    const values = years.map(y => yearly[y]);
    
    chartInstances['trends-biobank'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: years,
            datasets: [{
                label: biobank.name,
                data: values,
                backgroundColor: COLORS.primary
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    title: { display: true, text: 'Publications' }
                }
            }
        }
    });
}

// Theme Distribution Chart
function renderThemeDistributionChart() {
    const canvas = document.getElementById('chart-theme-dist');
    if (!canvas || !DATA.themes?.themes) return;
    
    destroyChart('theme-dist');
    
    const themes = DATA.themes.themes.slice(0, 8);
    
    chartInstances['theme-dist'] = new Chart(canvas, {
        type: 'pie',
        data: {
            labels: themes.map(t => t.name),
            datasets: [{
                data: themes.map(t => t.publications),
                backgroundColor: CHART_COLORS
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right' }
            }
        }
    });
}

// Theme Publications Chart
function renderThemePublicationsChart() {
    const canvas = document.getElementById('chart-theme-pubs');
    if (!canvas || !DATA.themes?.themes) return;
    
    destroyChart('theme-pubs');
    
    const themes = DATA.themes.themes.slice(0, 8);
    
    chartInstances['theme-pubs'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: themes.map(t => t.name.substring(0, 12)),
            datasets: [{
                label: 'Publications',
                data: themes.map(t => t.publications),
                backgroundColor: CHART_COLORS
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

// Comparison Radar Chart
function renderComparisonRadar(id1, id2) {
    const canvas = document.getElementById('chart-compare-radar');
    if (!canvas || !DATA.comparison?.biobanks) return;
    
    destroyChart('compare-radar');
    
    const b1 = DATA.comparison.biobanks.find(b => b.id === id1);
    const b2 = DATA.comparison.biobanks.find(b => b.id === id2);
    if (!b1 || !b2) return;
    
    const dims = DATA.comparison.radarDimensions || [];
    const labels = dims.map(d => d.label);
    
    chartInstances['compare-radar'] = new Chart(canvas, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: b1.name,
                    data: dims.map(d => b1.radar?.[d.key] || 0),
                    borderColor: COLORS.primary,
                    backgroundColor: COLORS.primary + '40'
                },
                {
                    label: b2.name,
                    data: dims.map(d => b2.radar?.[d.key] || 0),
                    borderColor: COLORS.critical,
                    backgroundColor: COLORS.critical + '40'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Equity Share Chart
function renderEquityShareChart() {
    const canvas = document.getElementById('chart-equity-share');
    if (!canvas || !DATA.equity?.summary) return;
    
    destroyChart('equity-share');
    
    const s = DATA.equity.summary;
    
    chartInstances['equity-share'] = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: ['HIC', 'LMIC'],
            datasets: [{
                data: [s.hic?.publicationShare || 0, s.lmic?.publicationShare || 0],
                backgroundColor: [COLORS.hic, COLORS.lmic]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// Equity Region Chart
function renderEquityRegionChart() {
    const canvas = document.getElementById('chart-equity-region');
    if (!canvas || !DATA.equity?.byRegion) return;
    
    destroyChart('equity-region');
    
    const regions = DATA.equity.byRegion;
    
    chartInstances['equity-region'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: regions.map(r => r.name),
            datasets: [{
                label: 'Publications',
                data: regions.map(r => r.publications),
                backgroundColor: regions.map(r => 
                    r.incomeCategory === 'LMIC' ? COLORS.lmic : 
                    r.incomeCategory === 'HIC' ? COLORS.hic : COLORS.moderate
                )
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    title: { display: true, text: 'Publications' }
                }
            }
        }
    });
}
'''

# =============================================================================
# MAIN
# =============================================================================

def create_file(path: Path, content: str) -> None:
    """Create file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    logger.info(f"Created: {path}")


def main():
    print("=" * 70)
    print(f"HEIM-Biobank v1.0: Build Static Dashboard Site")
    print("=" * 70)
    
    # Create directory structure
    print(f"\n📁 Creating site structure...")
    (DOCS_DIR / "css").mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "js").mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "data").mkdir(parents=True, exist_ok=True)
    
    # Generate HTML (with version date substitution)
    html_content = HTML_TEMPLATE.replace('{version_date}', VERSION_DATE)
    create_file(DOCS_DIR / "index.html", html_content)
    
    # Generate CSS
    create_file(DOCS_DIR / "css" / "style.css", CSS_TEMPLATE)
    
    # Generate JavaScript
    create_file(DOCS_DIR / "js" / "app.js", APP_JS_TEMPLATE)
    create_file(DOCS_DIR / "js" / "charts.js", CHARTS_JS_TEMPLATE)
    
    # Create .nojekyll for GitHub Pages
    create_file(DOCS_DIR / ".nojekyll", "")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ SITE BUILD COMPLETE")
    print(f"=" * 70)
    
    print(f"\n📁 Output Files:")
    for f in DOCS_DIR.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            rel_path = f.relative_to(DOCS_DIR)
            print(f"   {rel_path}: {size:,} bytes")
    
    print(f"\n🚀 Deployment:")
    print(f"   1. Ensure JSON files exist in docs/data/")
    print(f"   2. git add docs/")
    print(f"   3. git commit -m 'Update HEIM-Biobank dashboard'")
    print(f"   4. git push")
    print(f"   5. Enable GitHub Pages: Settings → Pages → Branch: main, Folder: /docs")
    
    print(f"\n🌐 Your dashboard will be at:")
    print(f"   https://[username].github.io/[repo]/")


if __name__ == "__main__":
    main()