#!/usr/bin/env python3
"""
02-04-bhem-build-site.py
========================
BHEM Step 5: Build static HTML/CSS/JS site for GitHub Pages

Creates a complete static website with interactive dashboard.
The site loads JSON data files and renders visualizations client-side.

INPUT:  docs/data/*.json (from previous step)
OUTPUT: docs/index.html
        docs/css/style.css
        docs/js/app.js
        docs/js/charts.js

USAGE:
    python 02-04-bhem-build-site.py

After running, push to GitHub and enable Pages on the docs/ folder.
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

# Paths
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
CSS_DIR = DOCS_DIR / "css"
JS_DIR = DOCS_DIR / "js"

# Create directories
CSS_DIR.mkdir(parents=True, exist_ok=True)
JS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HTML TEMPLATE
# =============================================================================

INDEX_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BHEM - Biobank Health Equity Monitor</title>
    <link rel="stylesheet" href="css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
</head>
<body>
    <header>
        <div class="container">
            <h1>🏥 BHEM</h1>
            <p class="subtitle">Biobank Health Equity Monitor</p>
            <p class="tagline">Tracking global biobank research alignment with disease burden</p>
        </div>
    </header>

    <nav>
        <div class="container">
            <button class="nav-btn active" data-view="overview">Overview</button>
            <button class="nav-btn" data-view="biobanks">Biobanks</button>
            <button class="nav-btn" data-view="diseases">Diseases</button>
            <button class="nav-btn" data-view="matrix">Matrix</button>
            <button class="nav-btn" data-view="trends">Trends</button>
        </div>
    </nav>

    <main>
        <div class="container">
            <!-- Overview View -->
            <section id="view-overview" class="view active">
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-value" id="stat-publications">-</span>
                        <span class="stat-label">Publications</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value" id="stat-biobanks">-</span>
                        <span class="stat-label">Biobanks</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value" id="stat-diseases">-</span>
                        <span class="stat-label">Diseases Tracked</span>
                    </div>
                    <div class="stat-card alert">
                        <span class="stat-value" id="stat-critical">-</span>
                        <span class="stat-label">Critical Gaps</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value" id="stat-equity">-</span>
                        <span class="stat-label">Equity Ratio</span>
                    </div>
                </div>

                <div class="chart-row">
                    <div class="chart-container">
                        <h3>Research Gap Distribution</h3>
                        <canvas id="chart-gaps"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>Publications by Region</h3>
                        <canvas id="chart-regions"></canvas>
                    </div>
                </div>

                <div class="chart-container full-width">
                    <h3>Top Critical Gap Diseases</h3>
                    <canvas id="chart-top-gaps"></canvas>
                </div>

                <div class="info-box">
                    <h3>About This Dashboard</h3>
                    <p>The Biobank Health Equity Monitor (BHEM) tracks research publications from 50+ global biobanks 
                    and evaluates their alignment with the global disease burden. Higher gap scores indicate 
                    diseases that are under-researched relative to their public health impact.</p>
                    <p><strong>Equity Ratio:</strong> Compares research intensity between high-income countries (HIC) 
                    and the Global South. A ratio > 1 indicates HIC-focused research.</p>
                    <p><strong>Data source:</strong> PubMed publications (2000-2025) linked to major biobanks.</p>
                </div>
            </section>

            <!-- Biobanks View -->
            <section id="view-biobanks" class="view">
                <div class="controls">
                    <input type="text" id="biobank-search" placeholder="Search biobanks...">
                    <select id="biobank-sort">
                        <option value="publications">Sort by Publications</option>
                        <option value="ros">Sort by Research Opportunity</option>
                        <option value="gaps">Sort by Critical Gaps</option>
                        <option value="name">Sort by Name</option>
                    </select>
                </div>
                <div id="biobank-list" class="card-grid"></div>
            </section>

            <!-- Diseases View -->
            <section id="view-diseases" class="view">
                <div class="controls">
                    <select id="disease-category">
                        <option value="all">All Categories</option>
                    </select>
                    <select id="disease-sort">
                        <option value="gap">Sort by Gap Score</option>
                        <option value="burden">Sort by Burden</option>
                        <option value="publications">Sort by Publications</option>
                    </select>
                </div>
                <div id="disease-list" class="card-grid"></div>
            </section>

            <!-- Matrix View -->
            <section id="view-matrix" class="view">
                <h3>Disease-Biobank Publication Matrix</h3>
                <p class="hint">Darker cells indicate more publications. Hover for details.</p>
                <div id="matrix-container" class="matrix-scroll">
                    <div id="matrix-heatmap"></div>
                </div>
            </section>

            <!-- Trends View -->
            <section id="view-trends" class="view">
                <div class="chart-container full-width">
                    <h3>Publication Growth Over Time</h3>
                    <canvas id="chart-trends"></canvas>
                </div>
                <div class="chart-row">
                    <div class="chart-container">
                        <h3>Cumulative Publications</h3>
                        <canvas id="chart-cumulative"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>Global South Priority Research</h3>
                        <canvas id="chart-gs-trend"></canvas>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <footer>
        <div class="container">
            <p>BHEM - Biobank Health Equity Monitor</p>
            <p>Data updated: <span id="last-update">-</span></p>
            <p><a href="https://github.com/manuelcorpas/17-EHR" target="_blank">GitHub Repository</a></p>
        </div>
    </footer>

    <script src="js/charts.js"></script>
    <script src="js/app.js"></script>
</body>
</html>
'''


# =============================================================================
# CSS STYLES
# =============================================================================

STYLE_CSS = '''/* BHEM Dashboard Styles */
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --danger: #dc2626;
    --warning: #f59e0b;
    --success: #10b981;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-600: #4b5563;
    --gray-800: #1f2937;
    --gray-900: #111827;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: var(--gray-50);
    color: var(--gray-800);
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Header */
header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 2rem 0;
    text-align: center;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
}

.tagline {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.5rem;
}

/* Navigation */
nav {
    background: white;
    border-bottom: 1px solid var(--gray-200);
    padding: 0.5rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

nav .container {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
}

.nav-btn {
    padding: 0.75rem 1.5rem;
    border: none;
    background: transparent;
    color: var(--gray-600);
    font-size: 1rem;
    cursor: pointer;
    border-radius: 0.5rem;
    transition: all 0.2s;
    white-space: nowrap;
}

.nav-btn:hover {
    background: var(--gray-100);
}

.nav-btn.active {
    background: var(--primary);
    color: white;
}

/* Main Content */
main {
    padding: 2rem 0;
    min-height: calc(100vh - 300px);
}

.view {
    display: none;
}

.view.active {
    display: block;
}

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stat-card.alert {
    border-left: 4px solid var(--danger);
}

.stat-value {
    display: block;
    font-size: 2rem;
    font-weight: bold;
    color: var(--primary);
}

.stat-card.alert .stat-value {
    color: var(--danger);
}

.stat-label {
    color: var(--gray-600);
    font-size: 0.875rem;
}

/* Charts */
.chart-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.chart-container {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.chart-container.full-width {
    margin-bottom: 1.5rem;
}

.chart-container h3 {
    margin-bottom: 1rem;
    color: var(--gray-800);
    font-size: 1.1rem;
}

.chart-container canvas {
    max-height: 300px;
}

/* Info Box */
.info-box {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    border-left: 4px solid var(--primary);
    margin-top: 2rem;
}

.info-box h3 {
    margin-bottom: 0.75rem;
}

.info-box p {
    color: var(--gray-600);
    margin-bottom: 0.5rem;
}

/* Controls */
.controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.controls input,
.controls select {
    padding: 0.75rem 1rem;
    border: 1px solid var(--gray-300);
    border-radius: 0.5rem;
    font-size: 1rem;
    min-width: 200px;
}

.controls input:focus,
.controls select:focus {
    outline: none;
    border-color: var(--primary);
}

/* Card Grid */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1rem;
}

.card {
    background: white;
    border-radius: 0.75rem;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}

.card-title {
    font-weight: 600;
    color: var(--gray-800);
}

.card-subtitle {
    font-size: 0.85rem;
    color: var(--gray-600);
}

.card-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    text-align: center;
    font-size: 0.85rem;
}

.card-stat-value {
    font-weight: 600;
    color: var(--primary);
}

.card-stat-label {
    color: var(--gray-600);
    font-size: 0.75rem;
}

/* Severity Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-critical {
    background: #fef2f2;
    color: var(--danger);
}

.badge-high {
    background: #fffbeb;
    color: var(--warning);
}

.badge-moderate {
    background: #f0f9ff;
    color: var(--primary);
}

.badge-low {
    background: #ecfdf5;
    color: var(--success);
}

/* Matrix */
.matrix-scroll {
    overflow-x: auto;
    background: white;
    border-radius: 0.75rem;
    padding: 1rem;
}

#matrix-heatmap {
    display: grid;
    gap: 2px;
    font-size: 0.7rem;
}

.matrix-cell {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border-radius: 2px;
}

.matrix-cell:hover {
    outline: 2px solid var(--primary);
}

.matrix-header {
    font-weight: 600;
    background: var(--gray-100);
    color: var(--gray-800);
}

.hint {
    color: var(--gray-600);
    font-size: 0.875rem;
    margin-bottom: 1rem;
}

/* Footer */
footer {
    background: var(--gray-800);
    color: white;
    padding: 2rem 0;
    text-align: center;
    margin-top: 2rem;
}

footer p {
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

footer a {
    color: var(--primary);
}

/* Responsive */
@media (max-width: 768px) {
    header h1 {
        font-size: 1.75rem;
    }
    
    .chart-row {
        grid-template-columns: 1fr;
    }
    
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .card-grid {
        grid-template-columns: 1fr;
    }
}

/* Loading State */
.loading {
    text-align: center;
    padding: 3rem;
    color: var(--gray-600);
}

.loading::after {
    content: '';
    display: inline-block;
    width: 1.5rem;
    height: 1.5rem;
    border: 2px solid var(--gray-300);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-left: 0.5rem;
    vertical-align: middle;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
'''


# =============================================================================
# JAVASCRIPT - CHARTS
# =============================================================================

CHARTS_JS = '''// Chart.js configuration and helpers
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.color = '#4b5563';

const COLORS = {
    primary: '#2563eb',
    danger: '#dc2626',
    warning: '#f59e0b',
    success: '#10b981',
    gray: '#6b7280',
    
    // Gap severity
    critical: '#dc2626',
    high: '#f59e0b',
    moderate: '#3b82f6',
    low: '#10b981',
    
    // Regions
    EUR: '#3b82f6',
    AMR: '#10b981',
    WPR: '#8b5cf6',
    AFR: '#f59e0b',
    EMR: '#ef4444',
    SEAR: '#ec4899',
    INTL: '#6b7280'
};

const chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

function createGapDistributionChart(data) {
    destroyChart('chart-gaps');
    const ctx = document.getElementById('chart-gaps').getContext('2d');
    
    const distribution = data.gaps.distribution;
    
    chartInstances['chart-gaps'] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'High', 'Moderate', 'Low'],
            datasets: [{
                data: [
                    distribution.Critical || 0,
                    distribution.High || 0,
                    distribution.Moderate || 0,
                    distribution.Low || 0
                ],
                backgroundColor: [
                    COLORS.critical,
                    COLORS.high,
                    COLORS.moderate,
                    COLORS.low
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createRegionChart(data) {
    destroyChart('chart-regions');
    const ctx = document.getElementById('chart-regions').getContext('2d');
    
    const regions = data.regions;
    const labels = Object.keys(regions);
    const values = Object.values(regions);
    
    chartInstances['chart-regions'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Publications',
                data: values,
                backgroundColor: labels.map(l => COLORS[l] || COLORS.gray)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: v => v.toLocaleString()
                    }
                }
            }
        }
    });
}

function createTopGapsChart(data) {
    destroyChart('chart-top-gaps');
    const ctx = document.getElementById('chart-top-gaps').getContext('2d');
    
    const topGaps = data.gaps.topGaps.slice(0, 8);
    
    chartInstances['chart-top-gaps'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topGaps.map(d => d.name),
            datasets: [{
                label: 'Gap Score',
                data: topGaps.map(d => d.score),
                backgroundColor: topGaps.map(d => 
                    d.score > 70 ? COLORS.critical :
                    d.score > 50 ? COLORS.high :
                    d.score > 30 ? COLORS.moderate : COLORS.low
                )
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Gap Score (0-100)'
                    }
                }
            }
        }
    });
}

function createTrendsChart(data) {
    destroyChart('chart-trends');
    const ctx = document.getElementById('chart-trends').getContext('2d');
    
    const yearly = data.yearly;
    
    chartInstances['chart-trends'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: yearly.map(d => d.year),
            datasets: [{
                label: 'Publications',
                data: yearly.map(d => d.count),
                borderColor: COLORS.primary,
                backgroundColor: COLORS.primary + '20',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: v => v.toLocaleString()
                    }
                }
            }
        }
    });
}

function createCumulativeChart(data) {
    destroyChart('chart-cumulative');
    const ctx = document.getElementById('chart-cumulative').getContext('2d');
    
    const yearly = data.yearly;
    
    chartInstances['chart-cumulative'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: yearly.map(d => d.year),
            datasets: [{
                label: 'Cumulative Publications',
                data: yearly.map(d => d.cumulative),
                borderColor: COLORS.success,
                backgroundColor: COLORS.success + '20',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: v => v.toLocaleString()
                    }
                }
            }
        }
    });
}

function createGlobalSouthTrendChart(data) {
    destroyChart('chart-gs-trend');
    const ctx = document.getElementById('chart-gs-trend').getContext('2d');
    
    const gsTrend = data.globalSouth;
    
    chartInstances['chart-gs-trend'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: gsTrend.map(d => d.year),
            datasets: [{
                label: 'Global South Priority Publications',
                data: gsTrend.map(d => d.count),
                backgroundColor: COLORS.warning
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
'''


# =============================================================================
# JAVASCRIPT - APP
# =============================================================================

APP_JS = '''// BHEM Dashboard Application
const DATA_PATH = 'data/';

let summaryData = null;
let biobanksData = null;
let diseasesData = null;
let matrixData = null;
let trendsData = null;

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        // Load all data
        [summaryData, biobanksData, diseasesData, matrixData, trendsData] = await Promise.all([
            fetch(DATA_PATH + 'summary.json').then(r => r.json()),
            fetch(DATA_PATH + 'biobanks.json').then(r => r.json()),
            fetch(DATA_PATH + 'diseases.json').then(r => r.json()),
            fetch(DATA_PATH + 'matrix.json').then(r => r.json()),
            fetch(DATA_PATH + 'trends.json').then(r => r.json())
        ]);
        
        // Setup navigation
        setupNavigation();
        
        // Render overview
        renderOverview();
        
        // Setup controls
        setupControls();
        
        // Update last update timestamp
        document.getElementById('last-update').textContent = 
            new Date(summaryData.lastUpdate).toLocaleDateString();
            
    } catch (error) {
        console.error('Error loading data:', error);
        document.querySelector('main .container').innerHTML = 
            '<div class="info-box"><h3>Error Loading Data</h3><p>Could not load dashboard data. Please try refreshing the page.</p></div>';
    }
}

function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show corresponding view
            const viewId = 'view-' + btn.dataset.view;
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById(viewId).classList.add('active');
            
            // Render view content
            switch(btn.dataset.view) {
                case 'overview': renderOverview(); break;
                case 'biobanks': renderBiobanks(); break;
                case 'diseases': renderDiseases(); break;
                case 'matrix': renderMatrix(); break;
                case 'trends': renderTrends(); break;
            }
        });
    });
}

function setupControls() {
    // Biobank search
    document.getElementById('biobank-search').addEventListener('input', renderBiobanks);
    document.getElementById('biobank-sort').addEventListener('change', renderBiobanks);
    
    // Disease filters
    const categorySelect = document.getElementById('disease-category');
    const categories = Object.keys(diseasesData.categories);
    categories.forEach(cat => {
        const option = document.createElement('option');
        option.value = cat;
        option.textContent = cat;
        categorySelect.appendChild(option);
    });
    categorySelect.addEventListener('change', renderDiseases);
    document.getElementById('disease-sort').addEventListener('change', renderDiseases);
}

function renderOverview() {
    // Stats
    document.getElementById('stat-publications').textContent = 
        summaryData.totals.publications.toLocaleString();
    document.getElementById('stat-biobanks').textContent = 
        summaryData.totals.biobanks;
    document.getElementById('stat-diseases').textContent = 
        summaryData.totals.diseases;
    document.getElementById('stat-critical').textContent = 
        summaryData.gaps.criticalCount;
    document.getElementById('stat-equity').textContent = 
        summaryData.equity.ratio.toFixed(1) + 'x';
    
    // Charts
    createGapDistributionChart(summaryData);
    createRegionChart(summaryData);
    createTopGapsChart(summaryData);
}

function renderBiobanks() {
    const search = document.getElementById('biobank-search').value.toLowerCase();
    const sortBy = document.getElementById('biobank-sort').value;
    
    let biobanks = [...biobanksData.biobanks];
    
    // Filter
    if (search) {
        biobanks = biobanks.filter(b => 
            b.name.toLowerCase().includes(search) ||
            b.country.toLowerCase().includes(search)
        );
    }
    
    // Sort
    switch(sortBy) {
        case 'publications':
            biobanks.sort((a, b) => b.stats.totalPublications - a.stats.totalPublications);
            break;
        case 'ros':
            biobanks.sort((a, b) => b.stats.ros - a.stats.ros);
            break;
        case 'gaps':
            biobanks.sort((a, b) => b.stats.criticalGaps - a.stats.criticalGaps);
            break;
        case 'name':
            biobanks.sort((a, b) => a.name.localeCompare(b.name));
            break;
    }
    
    // Render
    const container = document.getElementById('biobank-list');
    container.innerHTML = biobanks.map(b => `
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">${b.name}</div>
                    <div class="card-subtitle">${b.country} (${b.region})</div>
                </div>
            </div>
            <div class="card-stats">
                <div>
                    <div class="card-stat-value">${b.stats.totalPublications.toLocaleString()}</div>
                    <div class="card-stat-label">Publications</div>
                </div>
                <div>
                    <div class="card-stat-value">${b.stats.diseasesCovered}</div>
                    <div class="card-stat-label">Diseases</div>
                </div>
                <div>
                    <div class="card-stat-value">${b.stats.ros.toFixed(0)}</div>
                    <div class="card-stat-label">ROS</div>
                </div>
            </div>
            ${b.stats.criticalGaps > 0 ? 
                `<div style="margin-top: 0.75rem; font-size: 0.85rem; color: #dc2626;">
                    ⚠️ ${b.stats.criticalGaps} critical gap${b.stats.criticalGaps > 1 ? 's' : ''}
                </div>` : ''
            }
        </div>
    `).join('');
}

function renderDiseases() {
    const category = document.getElementById('disease-category').value;
    const sortBy = document.getElementById('disease-sort').value;
    
    let diseases = [...diseasesData.diseases];
    
    // Filter by category
    if (category !== 'all') {
        diseases = diseases.filter(d => d.category === category);
    }
    
    // Sort
    switch(sortBy) {
        case 'gap':
            diseases.sort((a, b) => b.gap.score - a.gap.score);
            break;
        case 'burden':
            diseases.sort((a, b) => b.burden.score - a.burden.score);
            break;
        case 'publications':
            diseases.sort((a, b) => b.research.publications - a.research.publications);
            break;
    }
    
    // Render
    const container = document.getElementById('disease-list');
    container.innerHTML = diseases.map(d => `
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">${d.name}</div>
                    <div class="card-subtitle">${d.category}</div>
                </div>
                <span class="badge badge-${d.gap.severity.toLowerCase()}">${d.gap.severity}</span>
            </div>
            <div class="card-stats">
                <div>
                    <div class="card-stat-value">${d.gap.score.toFixed(0)}</div>
                    <div class="card-stat-label">Gap Score</div>
                </div>
                <div>
                    <div class="card-stat-value">${d.research.publications.toLocaleString()}</div>
                    <div class="card-stat-label">Publications</div>
                </div>
                <div>
                    <div class="card-stat-value">${d.burden.dalys.toFixed(0)}M</div>
                    <div class="card-stat-label">DALYs</div>
                </div>
            </div>
            ${d.globalSouthPriority ? 
                '<div style="margin-top: 0.75rem; font-size: 0.85rem; color: #f59e0b;">🌍 Global South Priority</div>' : ''
            }
        </div>
    `).join('');
}

function renderMatrix() {
    const container = document.getElementById('matrix-heatmap');
    
    // Get top 15 biobanks and all diseases
    const topBiobanks = matrixData.matrix.slice(0, 15);
    const diseases = matrixData.diseases;
    
    // Find max value for color scaling
    let maxVal = 0;
    topBiobanks.forEach(row => {
        Object.values(row.values).forEach(v => {
            if (v > maxVal) maxVal = v;
        });
    });
    
    // Build grid
    let html = '<table style="border-collapse: collapse; font-size: 0.7rem;">';
    
    // Header row
    html += '<tr><th style="padding: 4px; min-width: 120px;"></th>';
    diseases.forEach(d => {
        const disease = diseasesData.diseases.find(x => x.id === d);
        const name = disease ? disease.name.substring(0, 12) : d;
        html += `<th style="padding: 4px; writing-mode: vertical-rl; text-orientation: mixed; height: 100px;">${name}</th>`;
    });
    html += '</tr>';
    
    // Data rows
    topBiobanks.forEach(row => {
        const biobank = biobanksData.biobanks.find(b => b.id === row.biobank);
        const name = biobank ? biobank.name.substring(0, 20) : row.biobank;
        
        html += `<tr><td style="padding: 4px; font-weight: 600;">${name}</td>`;
        diseases.forEach(d => {
            const val = row.values[d] || 0;
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const bg = val === 0 ? '#f9fafb' : 
                `rgba(37, 99, 235, ${0.1 + intensity * 0.9})`;
            const color = intensity > 0.5 ? 'white' : '#1f2937';
            
            html += `<td style="padding: 4px; text-align: center; background: ${bg}; color: ${color}; min-width: 40px;" title="${name}: ${val} publications for ${d}">${val || ''}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</table>';
    container.innerHTML = html;
}

function renderTrends() {
    createTrendsChart(trendsData);
    createCumulativeChart(trendsData);
    createGlobalSouthTrendChart(trendsData);
}
'''


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BHEM STEP 5: Build Static HTML Site")
    print("=" * 70)
    
    # Check for data files
    data_files = ['summary.json', 'biobanks.json', 'diseases.json', 'matrix.json', 'trends.json']
    missing = [f for f in data_files if not (DOCS_DIR / "data" / f).exists()]
    
    if missing:
        print(f"⚠️  Warning: Missing data files: {missing}")
        print(f"   Run 02-03-bhem-generate-json.py first for full functionality")
    
    # Write files
    print(f"\n📝 Generating site files...")
    
    # HTML
    with open(DOCS_DIR / "index.html", 'w') as f:
        f.write(INDEX_HTML)
    print(f"   ✅ index.html")
    
    # CSS
    with open(CSS_DIR / "style.css", 'w') as f:
        f.write(STYLE_CSS)
    print(f"   ✅ css/style.css")
    
    # JavaScript
    with open(JS_DIR / "charts.js", 'w') as f:
        f.write(CHARTS_JS)
    print(f"   ✅ js/charts.js")
    
    with open(JS_DIR / "app.js", 'w') as f:
        f.write(APP_JS)
    print(f"   ✅ js/app.js")
    
    # Create .nojekyll for GitHub Pages
    (DOCS_DIR / ".nojekyll").touch()
    print(f"   ✅ .nojekyll")
    
    print(f"\n📁 Site generated in: {DOCS_DIR}")
    print(f"\n✅ COMPLETE!")
    print(f"\n📋 Next steps:")
    print(f"   1. Commit and push to GitHub")
    print(f"   2. Go to repository Settings > Pages")
    print(f"   3. Set Source to 'Deploy from a branch'")
    print(f"   4. Select 'main' branch and '/docs' folder")
    print(f"   5. Save and wait for deployment")
    print(f"\n🌐 Your site will be at: https://[username].github.io/[repo]/")


if __name__ == "__main__":
    main()
