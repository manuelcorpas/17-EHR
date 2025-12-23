// Chart.js configuration and helpers
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
