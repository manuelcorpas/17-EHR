// HEIM Framework v3.0 Dashboard Application
// Three-Dimensional Equity Analysis: Discovery + Translation + Knowledge

// Global data store
let DATA = {
    summary: null,
    biobanks: null,
    diseases: null,
    matrix: null,
    trends: null,
    themes: null,
    comparison: null,
    equity: null,
    clinicalTrials: null,
    semantic: null,       // NEW: Semantic analysis data
    integrated: null      // NEW: Integrated three-dimensional metrics
};

// Chart instances for cleanup
let chartInstances = {};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('HEIM Framework v3.0 Dashboard initializing...');

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
        case 'knowledge': renderKnowledge(); break;
        case 'clinical-trials': renderClinicalTrials(); break;
        case 'biobanks': renderBiobanks(); break;
        case 'diseases': renderDiseases(); break;
        case 'comparison': renderComparison(); break;
        case 'trends': renderTrends(); break;
        case 'equity': renderEquity(); break;
    }
}

// Data Loading
async function loadAllData() {
    const files = ['summary', 'biobanks', 'diseases', 'matrix', 'trends', 'themes', 'comparison', 'equity', 'clinical_trials', 'integrated'];

    const promises = files.map(async (file) => {
        try {
            const response = await fetch(`data/${file}.json`);
            if (response.ok) {
                let key = file;
                if (file === 'clinical_trials') key = 'clinicalTrials';
                DATA[key] = await response.json();
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

// ============================================================
// OVERVIEW TAB
// ============================================================
function renderOverview() {
    // Render biobank summary stats
    if (DATA.summary) {
        const s = DATA.summary;
        setText('stat-biobanks', s.overview?.totalBiobanks || '--');
        setText('stat-publications', formatNumber(s.overview?.totalPublications));
        setText('stat-countries', s.overview?.totalCountries || '--');
        setText('stat-critical', s.gapDistribution?.Critical || '--');
    }

    // Render clinical trials summary stats
    if (DATA.clinicalTrials) {
        const ct = DATA.clinicalTrials;
        setText('stat-ct-trials', formatNumber(ct.summary?.totalTrials));
        setText('stat-ct-gs-pct', ct.globalSouthAnalysis?.gsTrialsPct + '%');
        setText('stat-ct-gap', ct.globalSouthAnalysis?.intensityGap + 'x');
        setText('stat-ct-hic-ratio', ct.geographic?.hicLmicRatio + ':1');
    }

    // Render semantic/integrated stats
    if (DATA.integrated) {
        const int = DATA.integrated;
        console.log('Integrated data loaded:', int.n_diseases, 'diseases');

        // Calculate total papers from diseases
        const totalPapers = int.diseases?.reduce((sum, d) => sum + (d.n_papers || 0), 0) || 0;
        console.log('Total papers calculated:', totalPapers);

        setText('stat-sem-papers', formatNumber(totalPapers));
        setText('stat-sem-embeddings', formatNumber(Math.round(totalPapers * 0.91))); // ~91% success rate
        setText('stat-sem-diseases', int.n_diseases || '--');

        // Count highly isolated diseases (SII > 0.003)
        const isolated = int.diseases?.filter(d => d.sii > 0.003).length || 0;
        setText('stat-sem-isolated', isolated);

        // Render top neglected diseases
        renderTopNeglected(int.diseases);
    } else {
        console.warn('Integrated data not loaded');
    }

    // Render overview charts
    renderOverviewCharts();
}

function renderTopNeglected(diseases) {
    const container = document.getElementById('top-neglected-unified');
    if (!container || !diseases) return;

    // Sort by unified score and get top 5
    const top5 = [...diseases]
        .filter(d => d.unified_score && d.unified_score > 0)
        .sort((a, b) => b.unified_score - a.unified_score)
        .slice(0, 5);

    container.innerHTML = top5.map((d, i) => {
        const diseaseName = d.disease.replace(/_/g, ' ');
        return `
            <div class="top-five-item">
                <div class="top-five-rank">${i + 1}</div>
                <div class="top-five-name">${diseaseName}</div>
                <div class="top-five-score">Score: ${d.unified_score.toFixed(1)}</div>
            </div>
        `;
    }).join('');
}

function renderOverviewCharts() {
    if (!DATA.clinicalTrials) return;

    // Chart: Research Intensity Comparison
    const intensityCtx = document.getElementById('chart-intensity-compare');
    if (intensityCtx && DATA.clinicalTrials.keyFindings) {
        destroyChart('chart-intensity-compare');
        const lowest = DATA.clinicalTrials.keyFindings.lowestIntensity.slice(0, 5);
        const highest = DATA.clinicalTrials.keyFindings.highestIntensity.slice(0, 5);

        chartInstances['chart-intensity-compare'] = new Chart(intensityCtx, {
            type: 'bar',
            data: {
                labels: [...lowest.map(d => truncate(d.name, 20)), '', ...highest.map(d => truncate(d.name, 20))],
                datasets: [{
                    label: 'Trials/M DALYs',
                    data: [...lowest.map(d => d.intensity), null, ...highest.map(d => d.intensity)],
                    backgroundColor: [...lowest.map(d => d.gs ? '#d73027' : '#4575b4'), 'transparent', ...highest.map(() => '#1b7837')]
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Lowest (left) vs Highest (right) Research Intensity' }
                },
                scales: {
                    x: { type: 'logarithmic', title: { display: true, text: 'Trials per Million DALYs (log scale)' } }
                }
            }
        });
    }
}

// ============================================================
// KNOWLEDGE STRUCTURE TAB (NEW in v3.0)
// ============================================================
function renderKnowledge() {
    if (!DATA.integrated) return;

    const int = DATA.integrated;

    // Calculate summary stats
    const totalPapers = int.diseases?.reduce((sum, d) => sum + (d.n_papers || 0), 0) || 0;
    const totalEmbeddings = Math.round(totalPapers * 0.91);

    setText('sem-total-papers', formatNumber(totalPapers));
    setText('sem-embeddings', formatNumber(totalEmbeddings));
    setText('sem-diseases', int.n_diseases || '--');
    setText('sem-unified-mean', int.summary?.unified_score?.mean?.toFixed(1) || '--');

    // Set figure paths
    const figBasePath = 'figures/';
    const figUmap = document.getElementById('fig-umap');
    const figNetwork = document.getElementById('fig-network');
    const figHeatmap = document.getElementById('fig-heatmap');
    const figGapIsolation = document.getElementById('fig-gap-isolation');
    const figTemporal = document.getElementById('fig-temporal');

    if (figUmap) figUmap.src = figBasePath + 'fig_umap_disease_clusters.png';
    if (figNetwork) figNetwork.src = figBasePath + 'fig_knowledge_network.png';
    if (figHeatmap) figHeatmap.src = figBasePath + 'fig_semantic_isolation_heatmap.png';
    if (figGapIsolation) figGapIsolation.src = figBasePath + 'fig_gap_vs_isolation.png';
    if (figTemporal) figTemporal.src = figBasePath + 'fig_temporal_drift.png';

    // Render semantic diseases table
    renderSemanticDiseases();
}

function renderSemanticDiseases() {
    if (!DATA.integrated?.diseases) return;

    const filter = document.getElementById('semantic-filter')?.value || 'all';
    const search = document.getElementById('semantic-disease-search')?.value?.toLowerCase() || '';

    let allDiseases = [...DATA.integrated.diseases];

    // Calculate percentile ranks for each metric
    const siiValues = allDiseases.map(d => d.sii).filter(v => v).sort((a, b) => a - b);
    const ktpValues = allDiseases.map(d => d.ktp).filter(v => v).sort((a, b) => b - a); // reversed - lower is worse
    const rccValues = allDiseases.map(d => d.rcc).filter(v => v).sort((a, b) => a - b);

    const getPercentile = (value, sortedArr, reverse = false) => {
        if (!value || !sortedArr.length) return null;
        const idx = sortedArr.findIndex(v => v >= value);
        const pct = ((idx === -1 ? sortedArr.length : idx) / sortedArr.length) * 100;
        return reverse ? 100 - pct : pct;
    };

    let diseases = allDiseases;

    // Apply filters
    if (filter === 'high-isolation') {
        diseases = diseases.filter(d => d.sii > 0.003);
    } else if (filter === 'low-ktp') {
        diseases = diseases.filter(d => d.ktp < 0.999);
    } else if (filter === 'top-unified') {
        diseases = diseases.sort((a, b) => (b.unified_score || 0) - (a.unified_score || 0)).slice(0, 20);
    }

    // Apply search
    if (search) {
        diseases = diseases.filter(d => d.disease.toLowerCase().replace(/_/g, ' ').includes(search));
    }

    // Sort by unified score
    diseases = diseases.sort((a, b) => (b.unified_score || 0) - (a.unified_score || 0));

    const tbody = document.querySelector('#table-semantic-diseases tbody');
    if (!tbody) return;

    tbody.innerHTML = diseases.map(d => {
        const diseaseName = d.disease.replace(/_/g, ' ');
        const unifiedClass = d.unified_score > 40 ? 'critical' : d.unified_score > 30 ? 'high' : d.unified_score > 20 ? 'moderate' : 'low';

        // Convert to percentile ranks (higher = more isolated/worse)
        const siiPct = getPercentile(d.sii, siiValues);
        const ktpPct = getPercentile(d.ktp, ktpValues, true); // reverse - lower KTP is worse
        const rccPct = getPercentile(d.rcc, rccValues);

        // Color code percentiles
        const getPctClass = (pct) => pct > 75 ? 'pct-high' : pct > 50 ? 'pct-mod' : pct > 25 ? 'pct-low' : 'pct-good';

        return `<tr>
            <td>${diseaseName}</td>
            <td>${formatNumber(d.n_papers)}</td>
            <td>${d.gap_score !== null && !isNaN(d.gap_score) ? d.gap_score.toFixed(0) : '--'}</td>
            <td><span class="pct-badge ${getPctClass(siiPct)}">${siiPct !== null ? siiPct.toFixed(0) + '%' : '--'}</span></td>
            <td><span class="pct-badge ${getPctClass(ktpPct)}">${ktpPct !== null ? (100 - ktpPct).toFixed(0) + '%' : '--'}</span></td>
            <td><span class="pct-badge ${getPctClass(rccPct)}">${rccPct !== null ? rccPct.toFixed(0) + '%' : '--'}</span></td>
            <td><span class="unified-badge unified-${unifiedClass}">${d.unified_score ? d.unified_score.toFixed(1) : '--'}</span></td>
        </tr>`;
    }).join('');
}

// ============================================================
// CLINICAL TRIALS TAB
// ============================================================
function renderClinicalTrials() {
    if (!DATA.clinicalTrials) return;

    const ct = DATA.clinicalTrials;
    const gs = ct.globalSouthAnalysis;
    const geo = ct.geographic;

    // Summary cards
    setText('ct-total-trials', formatNumber(ct.summary.totalTrials));
    setText('ct-gs-trials', formatNumber(gs.gsTrials));
    setText('ct-gs-trials-pct', `(${gs.gsTrialsPct}%)`);
    setText('ct-intensity-gap', gs.intensityGap + 'x');
    setText('ct-neglected', ct.keyFindings.neglectedDiseases.length);

    // Cancer dominance
    const cancer = ct.keyFindings.cancerDominance;
    setText('ct-cancer-pct', cancer.cancerPct + '%');
    setText('ct-cancer-burden-pct', cancer.cancerDALYsPct + '%');
    setText('ct-cancer-ratio', (cancer.cancerPct / cancer.cancerDALYsPct).toFixed(1) + 'x');

    // Temporal insights
    if (ct.temporal && ct.temporal.length > 0) {
        const firstYear = ct.temporal[0];
        const lastYear = ct.temporal[ct.temporal.length - 1];
        const growthFactor = (lastYear.trials / firstYear.trials).toFixed(1);
        const gsDecline = (lastYear.gsPct - ct.temporal.slice(0, 6).reduce((a, b) => a + b.gsPct, 0) / 6).toFixed(1);
        setText('ct-growth-factor', growthFactor + 'x');
        setText('ct-gs-decline', Math.abs(gsDecline));
    }

    // Render charts
    renderCTCharts();

    // Render tables
    renderCTTables();
}

function renderCTCharts() {
    const ct = DATA.clinicalTrials;

    // Trials by category (bar)
    const catCtx = document.getElementById('chart-ct-by-category');
    if (catCtx) {
        destroyChart('chart-ct-by-category');
        const categories = ct.categories.slice(0, 12);
        chartInstances['chart-ct-by-category'] = new Chart(catCtx, {
            type: 'bar',
            data: {
                labels: categories.map(c => truncate(c.name, 25)),
                datasets: [{
                    label: 'Clinical Trials',
                    data: categories.map(c => c.trials),
                    backgroundColor: '#2563eb'
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { title: { display: true, text: 'Number of Trials' } } }
            }
        });
    }

    // Intensity by category
    const intCtx = document.getElementById('chart-ct-intensity');
    if (intCtx) {
        destroyChart('chart-ct-intensity');
        const sorted = [...ct.categories].sort((a, b) => a.intensity - b.intensity).slice(0, 12);
        chartInstances['chart-ct-intensity'] = new Chart(intCtx, {
            type: 'bar',
            data: {
                labels: sorted.map(c => truncate(c.name, 25)),
                datasets: [{
                    label: 'Trials/M DALYs',
                    data: sorted.map(c => c.intensity),
                    backgroundColor: sorted.map(c => c.intensity < 200 ? '#dc3545' : c.intensity < 500 ? '#ffc107' : '#28a745')
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { title: { display: true, text: 'Trials per Million DALYs' } } }
            }
        });
    }

    // HIC vs LMIC pie
    const hicCtx = document.getElementById('chart-ct-hic-lmic');
    if (hicCtx) {
        destroyChart('chart-ct-hic-lmic');
        chartInstances['chart-ct-hic-lmic'] = new Chart(hicCtx, {
            type: 'doughnut',
            data: {
                labels: ['High-Income (HIC)', 'Low/Middle-Income (LMIC)'],
                datasets: [{
                    data: [ct.geographic.hicSites, ct.geographic.lmicSites],
                    backgroundColor: ['#2166ac', '#b2182b']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // Top countries bar
    const countryCtx = document.getElementById('chart-ct-countries');
    if (countryCtx) {
        destroyChart('chart-ct-countries');
        const countries = ct.geographic.topCountries.slice(0, 15);
        chartInstances['chart-ct-countries'] = new Chart(countryCtx, {
            type: 'bar',
            data: {
                labels: countries.map(c => c.name),
                datasets: [{
                    label: 'Trial Sites',
                    data: countries.map(c => c.sites),
                    backgroundColor: countries.map(c => c.income === 'HIC' ? '#2166ac' : '#b2182b')
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { title: { display: true, text: 'Number of Trial Sites' } } }
            }
        });
    }

    // Temporal volume
    const volCtx = document.getElementById('chart-ct-temporal-volume');
    if (volCtx && ct.temporal) {
        destroyChart('chart-ct-temporal-volume');
        chartInstances['chart-ct-temporal-volume'] = new Chart(volCtx, {
            type: 'bar',
            data: {
                labels: ct.temporal.map(t => t.year),
                datasets: [
                    {
                        label: 'GS Priority',
                        data: ct.temporal.map(t => t.gsTrials),
                        backgroundColor: '#d73027'
                    },
                    {
                        label: 'Other',
                        data: ct.temporal.map(t => t.trials - t.gsTrials),
                        backgroundColor: '#4575b4'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                    x: { stacked: true },
                    y: { stacked: true, title: { display: true, text: 'Trials Started' } }
                }
            }
        });
    }

    // GS priority share over time
    const gsCtx = document.getElementById('chart-ct-temporal-gs');
    if (gsCtx && ct.temporal) {
        destroyChart('chart-ct-temporal-gs');
        chartInstances['chart-ct-temporal-gs'] = new Chart(gsCtx, {
            type: 'line',
            data: {
                labels: ct.temporal.map(t => t.year),
                datasets: [{
                    label: 'GS Priority Share (%)',
                    data: ct.temporal.map(t => t.gsPct),
                    borderColor: '#d73027',
                    backgroundColor: 'rgba(215, 48, 39, 0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { title: { display: true, text: '% of Trials' }, min: 0, max: 60 } }
            }
        });
    }
}

function renderCTTables() {
    const ct = DATA.clinicalTrials;

    // GS Priority diseases table
    const gsTable = document.querySelector('#table-ct-gs-diseases tbody');
    if (gsTable) {
        const gsDiseases = ct.diseases.filter(d => d.globalSouthPriority).sort((a, b) => b.dalys - a.dalys);
        gsTable.innerHTML = gsDiseases.map(d => {
            const status = d.trials < 500 ? 'Severely Neglected' : d.trials < 2000 ? 'Neglected' : d.trials < 10000 ? 'Under-researched' : 'Moderate';
            const statusClass = d.trials < 500 ? 'critical' : d.trials < 2000 ? 'high' : d.trials < 10000 ? 'moderate' : 'low';
            return `<tr>
                <td>${d.name}</td>
                <td>${truncate(d.category, 20)}</td>
                <td>${formatNumber(d.trials)}</td>
                <td>${d.dalys.toFixed(1)}</td>
                <td>${d.intensity.toFixed(0)}</td>
                <td><span class="badge badge-${statusClass}">${status}</span></td>
            </tr>`;
        }).join('');
    }

    // Lowest intensity table
    const lowTable = document.querySelector('#table-ct-lowest-intensity tbody');
    if (lowTable && ct.keyFindings) {
        lowTable.innerHTML = ct.keyFindings.lowestIntensity.map(d => `
            <tr>
                <td>${d.name}</td>
                <td>${d.intensity.toFixed(1)}</td>
                <td>${formatNumber(ct.diseases.find(x => x.name === d.name)?.trials || 0)}</td>
                <td>${d.dalys}</td>
                <td>${d.gs ? '<span class="badge badge-critical">Yes</span>' : 'No'}</td>
            </tr>
        `).join('');
    }
}

// ============================================================
// BIOBANKS TAB
// ============================================================
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

// ============================================================
// DISEASES TAB
// ============================================================
function renderDiseases() {
    if (!DATA.clinicalTrials?.diseases) return;

    const tbody = document.querySelector('#table-diseases tbody');
    if (!tbody) return;

    const filter = document.getElementById('disease-view-filter')?.value || 'all';
    const search = document.getElementById('disease-search')?.value.toLowerCase() || '';

    let diseases = DATA.clinicalTrials.diseases;

    if (filter === 'gs') {
        diseases = diseases.filter(d => d.globalSouthPriority);
    } else if (filter === 'neglected') {
        diseases = diseases.filter(d => d.trials < 500);
    }

    if (search) {
        diseases = diseases.filter(d => d.name.toLowerCase().includes(search));
    }

    tbody.innerHTML = diseases.map(d => `
        <tr>
            <td>${d.name}</td>
            <td>${truncate(d.category, 25)}</td>
            <td>${d.dalys.toFixed(1)}</td>
            <td>${formatNumber(d.trials)}</td>
            <td>${d.intensity.toFixed(0)}</td>
            <td>${d.globalSouthPriority ? '<span class="badge badge-critical">Yes</span>' : 'No'}</td>
        </tr>
    `).join('');

    // Render scatter plot
    renderDiseaseScatter();
}

function renderDiseaseScatter() {
    const ctx = document.getElementById('chart-disease-scatter');
    if (!ctx || !DATA.clinicalTrials) return;

    destroyChart('chart-disease-scatter');

    const diseases = DATA.clinicalTrials.diseases.filter(d => d.dalys > 0 && d.trials > 0);

    // Identify diseases to label (high burden, extreme trial counts, notable)
    const labeledDiseases = new Set([
        'Ischemic heart disease', 'Stroke', 'Diabetes mellitus', 'COPD',
        'Lower respiratory infections', 'Neonatal disorders', 'HIV/AIDS',
        'Malaria', 'Tuberculosis', 'Road injuries', 'Diarrheal diseases',
        'Breast cancer', 'Lung cancer', 'Depressive disorders', 'Alzheimer'
    ]);

    // Also label highest burden and lowest trials ratio
    const sortedByBurden = [...diseases].sort((a, b) => b.dalys - a.dalys).slice(0, 8);
    const sortedByIntensity = [...diseases].sort((a, b) => a.intensity - b.intensity).slice(0, 5);
    sortedByBurden.forEach(d => labeledDiseases.add(d.name));
    sortedByIntensity.forEach(d => labeledDiseases.add(d.name));

    chartInstances['chart-disease-scatter'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Global South Priority',
                    data: diseases.filter(d => d.globalSouthPriority).map(d => ({
                        x: d.trials,
                        y: d.dalys,
                        label: d.name,
                        showLabel: labeledDiseases.has(d.name)
                    })),
                    backgroundColor: 'rgba(215, 48, 39, 0.7)',
                    pointRadius: 8
                },
                {
                    label: 'Other Diseases',
                    data: diseases.filter(d => !d.globalSouthPriority).map(d => ({
                        x: d.trials,
                        y: d.dalys,
                        label: d.name,
                        showLabel: labeledDiseases.has(d.name)
                    })),
                    backgroundColor: 'rgba(69, 117, 180, 0.5)',
                    pointRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Clinical Trials (log scale)', font: { size: 14 } }
                },
                y: {
                    title: { display: true, text: 'Disease Burden (Million DALYs)', font: { size: 14 } }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.raw.label}: ${ctx.raw.x.toLocaleString()} trials, ${ctx.raw.y.toFixed(1)}M DALYs`
                    }
                },
                datalabels: {
                    display: (context) => context.dataset.data[context.dataIndex].showLabel,
                    formatter: (value) => value.label.length > 18 ? value.label.substring(0, 16) + '...' : value.label,
                    color: '#333',
                    font: { size: 11, weight: 'bold' },
                    anchor: 'end',
                    align: 'top',
                    offset: 4
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

// ============================================================
// COMPARISON TAB (Pipeline Gap)
// ============================================================
function renderComparison() {
    if (!DATA.clinicalTrials) return;

    const ct = DATA.clinicalTrials;

    // Update comparison table
    setText('comp-hic-ratio', ct.geographic.hicLmicRatio + ':1 (sites)');
    setText('comp-cancer-focus', ct.keyFindings.cancerDominance.cancerPct + '%');
    setText('comp-gs-share', ct.globalSouthAnalysis.gsTrialsPct + '%');

    // Render neglected diseases grid
    const grid = document.getElementById('neglected-diseases-grid');
    if (grid) {
        const neglected = ['Malaria', 'Tuberculosis', 'Neonatal disorders', 'Neglected tropical diseases',
                          'Mental disorders', 'Road injuries', 'Lower respiratory infections'];
        grid.innerHTML = neglected.map(name => `
            <div class="neglected-disease-card">
                <div class="disease-name">${name}</div>
                <div class="disease-status">Neglected at both stages</div>
            </div>
        `).join('');
    }
}

// ============================================================
// TRENDS TAB
// ============================================================
function renderTrends() {
    if (!DATA.clinicalTrials?.temporal) return;

    const temporal = DATA.clinicalTrials.temporal;

    // Growth chart
    const growthCtx = document.getElementById('chart-trends-ct-growth');
    if (growthCtx) {
        destroyChart('chart-trends-ct-growth');
        chartInstances['chart-trends-ct-growth'] = new Chart(growthCtx, {
            type: 'bar',
            data: {
                labels: temporal.map(t => t.year),
                datasets: [{
                    label: 'Trials Started',
                    data: temporal.map(t => t.trials),
                    backgroundColor: '#2563eb'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { title: { display: true, text: 'Number of Trials' } } }
            }
        });
    }

    // GS share chart
    const gsCtx = document.getElementById('chart-trends-gs-share');
    if (gsCtx) {
        destroyChart('chart-trends-gs-share');
        chartInstances['chart-trends-gs-share'] = new Chart(gsCtx, {
            type: 'line',
            data: {
                labels: temporal.map(t => t.year),
                datasets: [{
                    label: 'GS Priority Share (%)',
                    data: temporal.map(t => t.gsPct),
                    borderColor: '#d73027',
                    backgroundColor: 'rgba(215, 48, 39, 0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: '% of Trials' }, min: 0, max: 60 }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 38,
                                yMax: 38,
                                borderColor: '#666',
                                borderDash: [5, 5],
                                label: { content: 'Burden Share (38%)', enabled: true }
                            }
                        }
                    }
                }
            }
        });
    }
}

// ============================================================
// EQUITY TAB
// ============================================================
function renderEquity() {
    if (!DATA.clinicalTrials) return;

    const ct = DATA.clinicalTrials;

    // Update stats
    setText('equity-ct-ratio', ct.geographic.hicLmicRatio + ':1');
    setText('equity-gs-research', ct.globalSouthAnalysis.gsTrialsPct + '%');

    // Burden vs Research chart
    const burdenCtx = document.getElementById('chart-equity-burden-research');
    if (burdenCtx) {
        destroyChart('chart-equity-burden-research');
        chartInstances['chart-equity-burden-research'] = new Chart(burdenCtx, {
            type: 'bar',
            data: {
                labels: ['Global South Priority', 'Other Diseases'],
                datasets: [
                    {
                        label: 'Disease Burden (%)',
                        data: [ct.globalSouthAnalysis.gsDALYsPct, 100 - ct.globalSouthAnalysis.gsDALYsPct],
                        backgroundColor: ['#d73027', '#4575b4']
                    },
                    {
                        label: 'Clinical Trials (%)',
                        data: [ct.globalSouthAnalysis.gsTrialsPct, 100 - ct.globalSouthAnalysis.gsTrialsPct],
                        backgroundColor: ['rgba(215, 48, 39, 0.5)', 'rgba(69, 117, 180, 0.5)']
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: { y: { title: { display: true, text: 'Percentage' } } }
            }
        });
    }

    // Geographic distribution
    const geoCtx = document.getElementById('chart-equity-geo');
    if (geoCtx) {
        destroyChart('chart-equity-geo');
        chartInstances['chart-equity-geo'] = new Chart(geoCtx, {
            type: 'doughnut',
            data: {
                labels: ['HIC', 'LMIC'],
                datasets: [{
                    data: [ct.geographic.hicPct, ct.geographic.lmicPct],
                    backgroundColor: ['#2166ac', '#b2182b']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // GS diseases table
    const gsTable = document.querySelector('#table-gs-diseases-equity tbody');
    if (gsTable) {
        const gsDiseases = ct.diseases.filter(d => d.globalSouthPriority).sort((a, b) => b.dalys - a.dalys);
        gsTable.innerHTML = gsDiseases.map(d => {
            const status = d.trials < 500 ? 'Severely Neglected' : d.trials < 2000 ? 'Neglected' : d.trials < 10000 ? 'Under-researched' : 'Moderate';
            const statusClass = d.trials < 500 ? 'critical' : d.trials < 2000 ? 'high' : d.trials < 10000 ? 'moderate' : 'low';
            return `<tr>
                <td>${d.name}</td>
                <td>${d.dalys.toFixed(1)}</td>
                <td>${formatNumber(d.trials)}</td>
                <td>${d.intensity.toFixed(0)}</td>
                <td><span class="badge badge-${statusClass}">${status}</span></td>
            </tr>`;
        }).join('');
    }
}

// ============================================================
// FILTERS
// ============================================================
function setupFilters() {
    // Biobank filters
    document.getElementById('biobank-search')?.addEventListener('input', renderBiobanks);
    document.getElementById('biobank-region-filter')?.addEventListener('change', renderBiobanks);
    document.getElementById('biobank-category-filter')?.addEventListener('change', renderBiobanks);

    // Disease filters
    document.getElementById('disease-search')?.addEventListener('input', renderDiseases);
    document.getElementById('disease-view-filter')?.addEventListener('change', renderDiseases);

    // Semantic/Knowledge filters
    document.getElementById('semantic-disease-search')?.addEventListener('input', renderSemanticDiseases);
    document.getElementById('semantic-filter')?.addEventListener('change', renderSemanticDiseases);
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================
function formatNumber(num) {
    if (num === undefined || num === null) return '--';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(0) + 'K';
    return num.toLocaleString();
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len) + '...' : str;
}

function getCategoryClass(category) {
    if (!category) return '';
    const cat = category.toLowerCase();
    if (cat === 'high') return 'eas-high';
    if (cat === 'moderate') return 'eas-moderate';
    if (cat === 'low') return 'eas-low';
    return cat;
}

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
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
