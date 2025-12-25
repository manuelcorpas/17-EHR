// HEIM-Biobank v1.0 Chart Visualizations

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
