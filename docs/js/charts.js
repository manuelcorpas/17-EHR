// HEIM Framework v3.0 — Chart Library (Scroll Narrative)
// Publication-quality charts for the 6-section scroll dashboard
// Dependencies: Chart.js 4.4.1, ChartDataLabels plugin
// Note: chartInstances and destroyChart are defined in app.js (loaded after)

'use strict';

// ============================================================
// CHART 1: Unified Ranking — Top 30 diseases (Section 3)
// Horizontal bar chart, three-colour scheme matching publication Fig 5
// ============================================================

// WHO NTD classification (from 06-heim-publication-figures.py)
const WHO_NTDS = new Set([
    'lymphatic filariasis', 'guinea worm disease', 'schistosomiasis',
    'onchocerciasis', 'leishmaniasis', 'chagas disease',
    'african trypanosomiasis', 'dengue', 'rabies', 'ascariasis',
    'trichuriasis', 'hookworm disease', 'scabies', 'trachoma',
    'cysticercosis', 'cystic echinococcosis', 'yellow fever',
    'foodborne trematodiases', 'food-borne trematodiases', 'leprosy'
]);

// Global South priority diseases (from clinical_trials.json + additional)
const GS_PRIORITY = new Set([
    'covid-19', 'neonatal disorders', 'lower respiratory infections',
    'diarrheal diseases', 'malaria', 'tuberculosis', 'hiv/aids',
    'dietary iron deficiency', 'meningitis', 'rheumatic heart disease',
    'maternal disorders', 'hemoglobinopathies and hemolytic anemias',
    'protein-energy malnutrition', 'cervical cancer',
    'typhoid and paratyphoid', 'sexually transmitted infections excluding hiv',
    'acute hepatitis', 'other neglected tropical diseases',
    'intestinal nematode infections',
    // Additional GS diseases
    'iodine deficiency', 'vitamin a deficiency',
    'invasive non-typhoidal salmonella (ints)', 'encephalitis',
    'measles', 'pertussis', 'tetanus', 'ebola', 'zika virus'
]);

function classifyDisease(diseaseName) {
    const clean = diseaseName.replace(/_/g, ' ').toLowerCase().trim();
    for (const ntd of WHO_NTDS) {
        if (clean.includes(ntd) || ntd.includes(clean)) return 'ntd';
    }
    for (const gs of GS_PRIORITY) {
        if (clean.includes(gs) || gs.includes(clean)) return 'gs';
    }
    return 'other';
}

const DISEASE_COLORS = {
    ntd:   '#E74C3C',  // Red — WHO NTD
    gs:    '#F39C12',  // Orange — Global South priority
    other: '#95A5A6'   // Grey — Other condition
};

function renderUnifiedRankingChart(diseases) {
    const ctx = document.getElementById('chart-unified-ranking');
    if (!ctx || !diseases) return;

    if (typeof destroyChart === 'function') destroyChart('chart-unified-ranking');

    // Filter out injuries and null scores, sort descending, take top 30
    const ranked = diseases
        .filter(d => !HEIMEngine.INJURIES.has(d.disease) && d.unified_score != null)
        .sort((a, b) => b.unified_score - a.unified_score)
        .slice(0, 30);

    // Display name formatting (matching publication)
    const displayName = (d) => {
        let name = d.disease.replace(/_/g, ' ');
        name = name.replace('Paralytic ileus and intestinal obstruction', 'Paralytic ileus / intestinal obstruction');
        name = name.replace('Inguinal, femoral, and abdominal hernia', 'Inguinal/femoral/abdominal hernia');
        name = name.replace('Invasive Non-typhoidal Salmonella (iNTS)', 'Invasive non-typhoidal Salmonella');
        if (name.length > 40) name = name.substring(0, 37) + '...';
        return name;
    };

    const labels = ranked.map(d => displayName(d));
    const scores = ranked.map(d => d.unified_score);
    const categories = ranked.map(d => classifyDisease(d.disease));
    const bgColors = categories.map(c => DISEASE_COLORS[c]);

    // Count NTDs and GS in top 10
    const ntdTop10 = categories.slice(0, 10).filter(c => c === 'ntd').length;
    const gsTop10 = categories.slice(0, 10).filter(c => c !== 'other').length;

    // Build three datasets for the legend
    const ntdData = scores.map((s, i) => categories[i] === 'ntd' ? s : null);
    const gsData = scores.map((s, i) => categories[i] === 'gs' ? s : null);
    const otherData = scores.map((s, i) => categories[i] === 'other' ? s : null);

    chartInstances['chart-unified-ranking'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'WHO Neglected Tropical Disease',
                    data: ntdData,
                    backgroundColor: DISEASE_COLORS.ntd,
                    borderRadius: 2,
                    barThickness: 18,
                    skipNull: true
                },
                {
                    label: 'Global South priority (non-NTD)',
                    data: gsData,
                    backgroundColor: DISEASE_COLORS.gs,
                    borderRadius: 2,
                    barThickness: 18,
                    skipNull: true
                },
                {
                    label: 'Other condition',
                    data: otherData,
                    backgroundColor: DISEASE_COLORS.other,
                    borderRadius: 2,
                    barThickness: 18,
                    skipNull: true
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: { right: 50, bottom: 10 }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        pointStyle: 'rect',
                        padding: 16,
                        font: { size: 11, family: '-apple-system, sans-serif' },
                        color: '#1a1a2e'
                    }
                },
                datalabels: {
                    display: (context) => context.dataset.data[context.dataIndex] !== null,
                    anchor: 'end',
                    align: 'right',
                    color: '#6b7280',
                    font: { size: 10, family: '-apple-system, sans-serif' },
                    formatter: (v) => v !== null ? v.toFixed(1) : ''
                },
                tooltip: {
                    callbacks: {
                        title: (items) => items[0]?.label || '',
                        label: (item) => {
                            const v = item.raw;
                            return v !== null ? `${item.dataset.label}: ${v.toFixed(1)}` : '';
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Unified Neglect Score (PCA-derived)',
                        font: { size: 12, family: '-apple-system, sans-serif' },
                        color: '#6b7280'
                    },
                    grid: { color: '#f1f3f5' },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                },
                y: {
                    stacked: true,
                    grid: { display: false },
                    ticks: {
                        color: '#1a1a2e',
                        font: { size: 10, family: '-apple-system, sans-serif' }
                    }
                }
            }
        },
        plugins: [ChartDataLabels, {
            id: 'annotationBox',
            afterDraw: (chart) => {
                const { ctx: c, chartArea } = chart;
                c.save();
                const x = chartArea.left + 8;
                const y = chartArea.bottom - 58;
                c.fillStyle = 'rgba(255, 255, 240, 0.9)';
                c.strokeStyle = '#d1d5db';
                c.lineWidth = 1;
                c.beginPath();
                c.roundRect(x, y, 260, 52, 4);
                c.fill();
                c.stroke();
                c.fillStyle = '#374151';
                c.font = '10px -apple-system, sans-serif';
                c.fillText(`Top 10: ${gsTop10}/10 Global South burden`, x + 8, y + 16);
                c.fillText(`${ntdTop10}/10 WHO NTDs`, x + 8, y + 30);
                c.fillText('Weights: D=0.50, T=0.29, K=0.21 (PCA)', x + 8, y + 44);
                c.restore();
            }
        }]
    });
}


// ============================================================
// CHART 2: Intensity Gap — Lowest vs Highest (Section 4)
// Redesigned diverging bar for equity section
// ============================================================
function renderIntensityGapChart(clinicalTrials) {
    const ctx = document.getElementById('chart-intensity-gap');
    if (!ctx || !clinicalTrials?.keyFindings) return;

    if (typeof destroyChart === 'function') destroyChart('chart-intensity-gap');

    const lowest = clinicalTrials.keyFindings.lowestIntensity.slice(0, 8);
    const highest = clinicalTrials.keyFindings.highestIntensity.slice(0, 5);

    const allItems = [
        ...lowest.map(d => ({ name: d.name, value: d.intensity, type: 'low', gs: d.gs })),
        ...highest.map(d => ({ name: d.name, value: d.intensity, type: 'high', gs: false }))
    ];

    const labels = allItems.map(d => {
        const name = d.name.length > 28 ? d.name.substring(0, 26) + '...' : d.name;
        return d.gs ? name + ' *' : name;
    });
    const values = allItems.map(d => d.value);
    const colors = allItems.map(d =>
        d.type === 'low'
            ? (d.gs ? 'rgba(220, 38, 38, 0.85)' : 'rgba(234, 88, 12, 0.7)')
            : 'rgba(22, 163, 74, 0.75)'
    );

    chartInstances['chart-intensity-gap'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Trials per Million DALYs',
                data: values,
                backgroundColor: colors,
                borderRadius: 3,
                barThickness: 24
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: 20 } },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Research Intensity: Lowest vs Highest',
                    font: { size: 14, family: 'Georgia, serif', weight: 'normal' },
                    color: '#1a1a2e',
                    padding: { bottom: 16 }
                },
                datalabels: {
                    display: true,
                    anchor: 'end',
                    align: 'right',
                    color: '#6b7280',
                    font: { size: 10 },
                    formatter: (v) => v >= 1000 ? (v/1000).toFixed(1) + 'K' : v.toFixed(0)
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Trials per Million DALYs (log scale)',
                        font: { size: 12 },
                        color: '#6b7280'
                    },
                    grid: { color: '#f1f3f5' },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#1a1a2e',
                        font: { size: 11 }
                    }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}


// ============================================================
// CHART 3: Mini Isolation bar (Section 2 — Knowledge card)
// Tiny inline chart showing NTD vs non-NTD isolation
// ============================================================
function renderMiniIsolationChart(diseases) {
    const ctx = document.getElementById('chart-mini-isolation');
    if (!ctx || !diseases) return;

    if (typeof destroyChart === 'function') destroyChart('chart-mini-isolation');

    // NTD list
    const ntds = new Set([
        'Chagas_disease', 'Dengue', 'Leishmaniasis', 'Lymphatic_filariasis',
        'Onchocerciasis', 'Schistosomiasis', 'Trachoma', 'Rabies',
        'African_trypanosomiasis', 'Guinea_worm_disease', 'Cysticercosis',
        'Yellow_fever', 'Typhoid_and_paratyphoid', 'Malaria',
        'Leprosy', 'Food-borne_trematodiases', 'Other_neglected_tropical_diseases'
    ]);

    const ntdSII = diseases.filter(d => ntds.has(d.disease) && d.sii).map(d => d.sii);
    const otherSII = diseases.filter(d => !ntds.has(d.disease) && d.sii).map(d => d.sii);
    const ntdMean = ntdSII.length ? ntdSII.reduce((a,b) => a+b, 0) / ntdSII.length : 0;
    const otherMean = otherSII.length ? otherSII.reduce((a,b) => a+b, 0) / otherSII.length : 0;

    chartInstances['chart-mini-isolation'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['NTDs', 'Other'],
            datasets: [{
                data: [ntdMean * 10000, otherMean * 10000],
                backgroundColor: ['#7c3aed', '#d8b4fe'],
                borderRadius: 3,
                barThickness: 12
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                datalabels: { display: false }
            },
            scales: {
                x: { display: false, beginAtZero: true },
                y: {
                    display: true,
                    grid: { display: false },
                    ticks: { font: { size: 9 }, color: '#6b7280' }
                }
            }
        }
    });
}


// ============================================================
// CHART 4: Sensitivity Scatter (Section 5c — kept from original)
// ============================================================
// ============================================================
// CHART 5: Biobank Compare Radar (Section 5d — kept from original)
// ============================================================
function renderCompareRadar(biobanks) {
    const ctx = document.getElementById('chart-compare-radar');
    if (!ctx) return;

    if (typeof destroyChart === 'function') destroyChart('chart-compare-radar');

    const labels = ['Equity Alignment Score', 'Publications', 'Diseases Covered', 'Critical Gaps (inv)', 'Global South Coverage'];
    const colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed'];

    const datasets = biobanks.map((b, i) => {
        const maxPubs = Math.max(...biobanks.map(bb => bb.stats?.totalPublications || 0));
        return {
            label: b.name,
            data: [
                b.scores?.equityAlignment || 0,
                maxPubs > 0 ? (b.stats?.totalPublications / maxPubs) * 100 : 0,
                b.stats?.diseasesCovered ? (b.stats.diseasesCovered / 179) * 100 : 0,
                b.stats?.criticalGaps != null ? Math.max(0, 100 - b.stats.criticalGaps) : 50,
                50
            ],
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '25',
            pointRadius: 3
        };
    });

    chartInstances['chart-compare-radar'] = new Chart(ctx, {
        type: 'radar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: '#e5e7eb' },
                    ticks: { color: '#9ca3af', font: { size: 10 } },
                    pointLabels: { color: '#1a1a2e', font: { size: 11 } }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { font: { size: 12 }, color: '#1a1a2e' }
                }
            }
        }
    });
}


// ============================================================
// CHART 6: Trial Sites by Country — Top 15 (Section 4)
// Horizontal bar, HIC blue vs LMIC orange
// ============================================================
function renderTrialSitesChart(clinicalTrials) {
    const ctx = document.getElementById('chart-trial-sites');
    if (!ctx || !clinicalTrials?.geographic?.topCountries) return;

    if (typeof destroyChart === 'function') destroyChart('chart-trial-sites');

    const countries = clinicalTrials.geographic.topCountries.slice(0, 15);
    const labels = countries.map(c => c.name);
    const values = countries.map(c => c.sites);
    const colors = countries.map(c =>
        c.income === 'HIC' ? 'rgba(37, 99, 235, 0.8)' : 'rgba(234, 88, 12, 0.8)'
    );

    chartInstances['chart-trial-sites'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Trial Sites',
                data: values,
                backgroundColor: colors,
                borderRadius: 3,
                barThickness: 20
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: 50 } },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Clinical Trial Sites by Country (Top 15)',
                    font: { size: 14, family: 'Georgia, serif', weight: 'normal' },
                    color: '#1a1a2e',
                    padding: { bottom: 12 }
                },
                subtitle: {
                    display: true,
                    text: 'Blue = HIC  |  Orange = LMIC',
                    font: { size: 11 },
                    color: '#9ca3af',
                    padding: { bottom: 8 }
                },
                datalabels: {
                    display: true,
                    anchor: 'end',
                    align: 'right',
                    color: '#6b7280',
                    font: { size: 10 },
                    formatter: (v) => v >= 1000 ? (v / 1000).toFixed(1) + 'K' : v.toLocaleString()
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Trial Sites',
                        font: { size: 12 },
                        color: '#6b7280'
                    },
                    grid: { color: '#f1f3f5' },
                    ticks: {
                        color: '#9ca3af',
                        font: { size: 10 },
                        callback: (v) => v >= 1000 ? (v / 1000) + 'K' : v
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#1a1a2e',
                        font: { size: 11 }
                    }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}


// ============================================================
// CHART 7: Regional Publication Distribution (Section 4)
// Horizontal bar — 6 WHO regions sorted by publication count
// ============================================================
function renderRegionalPubsChart(summary) {
    const ctx = document.getElementById('chart-regional-pubs');
    if (!ctx || !summary?.regionStats) return;

    if (typeof destroyChart === 'function') destroyChart('chart-regional-pubs');

    const regions = [...summary.regionStats].sort((a, b) => b.publications - a.publications);
    const labels = regions.map(r => r.name);
    const values = regions.map(r => r.publications);
    const biobanks = regions.map(r => r.biobanks);

    const maxVal = Math.max(...values);
    const colors = regions.map(r => {
        const t = r.publications / maxVal;
        return `rgba(37, 99, 235, ${0.3 + t * 0.6})`;
    });

    chartInstances['chart-regional-pubs'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Publications',
                data: values,
                backgroundColor: colors,
                borderRadius: 3,
                barThickness: 22
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: 70 } },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'IHCC Biobank Publications by WHO Region',
                    font: { size: 14, family: 'Georgia, serif', weight: 'normal' },
                    color: '#1a1a2e',
                    padding: { bottom: 4 }
                },
                subtitle: {
                    display: true,
                    text: 'Connolly et al. (2025) Commun Med 5:210',
                    font: { size: 10, style: 'italic' },
                    color: '#9ca3af',
                    padding: { bottom: 8 }
                },
                datalabels: {
                    display: true,
                    anchor: 'end',
                    align: 'right',
                    color: '#6b7280',
                    font: { size: 10 },
                    formatter: (v, context) => {
                        const idx = context.dataIndex;
                        const bb = biobanks[idx];
                        return v.toLocaleString() + ' (' + bb + ' biobanks)';
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Total Publications',
                        font: { size: 12 },
                        color: '#6b7280'
                    },
                    grid: { color: '#f1f3f5' },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#1a1a2e',
                        font: { size: 11 }
                    }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}


// ============================================================
// CHART 8: NTD vs Non-NTD Semantic Isolation (Section 4)
// Two horizontal bars with statistical annotation
// ============================================================
function renderNTDIsolationChart(diseases) {
    const ctx = document.getElementById('chart-ntd-isolation');
    if (!ctx || !diseases) return;

    if (typeof destroyChart === 'function') destroyChart('chart-ntd-isolation');

    const ntds = new Set([
        'Chagas_disease', 'Dengue', 'Leishmaniasis', 'Lymphatic_filariasis',
        'Onchocerciasis', 'Schistosomiasis', 'Trachoma', 'Rabies',
        'African_trypanosomiasis', 'Guinea_worm_disease', 'Cysticercosis',
        'Yellow_fever', 'Typhoid_and_paratyphoid', 'Malaria',
        'Leprosy', 'Food-borne_trematodiases', 'Other_neglected_tropical_diseases'
    ]);

    const ntdSII = diseases.filter(d => ntds.has(d.disease) && d.sii).map(d => d.sii);
    const otherSII = diseases.filter(d => !ntds.has(d.disease) && d.sii).map(d => d.sii);
    const ntdMean = ntdSII.length ? ntdSII.reduce((a, b) => a + b, 0) / ntdSII.length : 0;
    const otherMean = otherSII.length ? otherSII.reduce((a, b) => a + b, 0) / otherSII.length : 0;

    // Scale to readable units (×10,000)
    const ntdVal = ntdMean * 10000;
    const otherVal = otherMean * 10000;

    // Compute isolation difference live
    const isolationPct = otherMean > 0 ? Math.round(((ntdMean - otherMean) / otherMean) * 100) : 0;

    chartInstances['chart-ntd-isolation'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['NTDs (n=' + ntdSII.length + ')', 'Non-NTD (n=' + otherSII.length + ')'],
            datasets: [{
                label: 'Mean SII (×10⁴)',
                data: [ntdVal, otherVal],
                backgroundColor: ['rgba(124, 58, 237, 0.85)', 'rgba(209, 196, 233, 0.7)'],
                borderRadius: 3,
                barThickness: 28
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: 20 } },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'NTD vs Non-NTD Semantic Isolation',
                    font: { size: 14, family: 'Georgia, serif', weight: 'normal' },
                    color: '#1a1a2e',
                    padding: { bottom: 4 }
                },
                subtitle: {
                    display: true,
                    text: isolationPct + '% higher isolation, P < 0.0001, Cohen\u2019s d = 1.80',
                    font: { size: 11, style: 'italic' },
                    color: '#7c3aed',
                    padding: { bottom: 8 }
                },
                datalabels: {
                    display: true,
                    anchor: 'end',
                    align: 'right',
                    color: '#6b7280',
                    font: { size: 11 },
                    formatter: (v) => v.toFixed(2)
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Mean Semantic Isolation Index (×10\u2074)',
                        font: { size: 11 },
                        color: '#6b7280'
                    },
                    grid: { color: '#f1f3f5' },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#1a1a2e',
                        font: { size: 11 }
                    }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}
