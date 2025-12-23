// BHEM Dashboard Application
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
