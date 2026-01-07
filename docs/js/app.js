// HEIM-Biobank v2.0 (IHCC) Dashboard Application

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
    console.log('HEIM-Biobank v2.0 (IHCC) Dashboard initializing...');
    
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
    
    let csv = headers.join(',') + '\n';
    csv += rows.map(r => r.map(v => `"${v || ''}"`).join(',')).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'heim-biobank-data.csv';
    a.click();
    URL.revokeObjectURL(url);
}
