// HEIM Framework v3.0 — Scroll Narrative Application
// Replaces tab-based navigation with vertical scroll narrative
// engine.js must be loaded before this file

'use strict';

// ============================================================
// GLOBAL STATE
// ============================================================
let DATA = {
    summary: null,
    biobanks: null,
    clinicalTrials: null,
    integrated: null
};

let WEIGHTS = {
    unified: { discovery: 0.501, translation: 0.293, knowledge: 0.206 },
    eas: { gap: 0.4, burdenMiss: 0.3, capacity: 0.3 },
    burden: { dalys: 0.5, deaths: 50.0, prevalence: 10.0 }
};

let BASELINE_SCORES = null;
let chartInstances = {};
let sectionsRendered = {};

// ============================================================
// INITIALISATION
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('HEIM Scroll Narrative initialising...');

    setupScrollNavigation();
    await loadAllData();

    if (DATA.integrated?.diseases) {
        BASELINE_SCORES = DATA.integrated.diseases.map(d => ({
            disease: d.disease,
            unified_score: d.unified_score
        }));
    }

    // Render visible sections
    renderSection2Framework();
    renderSection3Rankings();
    renderSection4Equity();

    // Setup interactions
    setupWeightSliders();
    renderBiobankChecklist();

    console.log('Dashboard ready');
});


// ============================================================
// SCROLL NAVIGATION (IntersectionObserver + dot sidebar)
// ============================================================
function setupScrollNavigation() {
    const sections = document.querySelectorAll('.scroll-section');
    const dotItems = document.querySelectorAll('.dot-nav-item');
    const progressBar = document.getElementById('progress-bar');

    // Click handler for dot nav
    dotItems.forEach(item => {
        item.addEventListener('click', () => {
            const sectionId = item.dataset.section;
            const target = document.getElementById(sectionId);
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });

    // IntersectionObserver for active dot
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                dotItems.forEach(d => d.classList.remove('active'));
                const active = document.querySelector(`.dot-nav-item[data-section="${id}"]`);
                if (active) active.classList.add('active');

                // Lazy render sections when they become visible
                lazyRenderSection(id);
            }
        });
    }, {
        threshold: 0.3,
        rootMargin: '-10% 0px -10% 0px'
    });

    sections.forEach(section => observer.observe(section));

    // Mobile progress bar
    window.addEventListener('scroll', () => {
        if (!progressBar) return;
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        progressBar.style.width = progress + '%';
    }, { passive: true });
}

function lazyRenderSection(id) {
    if (sectionsRendered[id]) return;

    switch (id) {
        case 'explorer':
            renderLiveRankings(DATA.integrated?.diseases);
            sectionsRendered[id] = true;
            break;
    }
}


// ============================================================
// DATA LOADING
// ============================================================
async function loadAllData() {
    const files = [
        { name: 'summary', key: 'summary' },
        { name: 'biobanks', key: 'biobanks' },
        { name: 'clinical_trials', key: 'clinicalTrials' },
        { name: 'integrated', key: 'integrated' }
    ];

    const promises = files.map(async ({ name, key }) => {
        try {
            const response = await fetch(`data/${name}.json`);
            if (response.ok) {
                DATA[key] = await response.json();
                console.log(`Loaded ${name}.json`);
            }
        } catch (err) {
            console.warn(`Error loading ${name}.json:`, err);
        }
    });

    await Promise.all(promises);
}


// ============================================================
// SECTION 2: FRAMEWORK RENDERING
// ============================================================
function renderSection2Framework() {
    if (DATA.summary) {
        const o = DATA.summary.overview;
        if (o) {
            setText('fw-discovery-stat', formatNumber(o.totalBiobanks) + ' biobanks');
        }
        const critical = DATA.summary.gapDistribution?.Critical || 22;
        setText('fw-discovery-gaps', critical);

        // Discovery bar: percentage of critical gaps
        const totalDiseases = o?.totalDiseases || 179;
        const critPct = Math.round((critical / totalDiseases) * 100);
        const discoveryBar = document.getElementById('fw-discovery-bar');
        if (discoveryBar) {
            discoveryBar.style.width = critPct + '%';
        }
        setText('fw-discovery-pct', critical + ' diseases with no biobank coverage');
    }

    if (DATA.clinicalTrials) {
        const ct = DATA.clinicalTrials;
        if (ct.summary) {
            setText('fw-translation-stat', formatNumber(ct.summary.totalTrials) + ' trials');
        }
        const gs = ct.globalSouthAnalysis;
        if (gs) {
            setText('fw-translation-gap', gs.intensityGap);
            const gsBar = document.getElementById('fw-gs-bar');
            if (gsBar) {
                gsBar.style.width = gs.gsTrialsPct + '%';
            }
        }
    }

    if (DATA.integrated?.diseases) {
        // Knowledge card stats
        const ntds = new Set([
            'Chagas_disease', 'Dengue', 'Leishmaniasis', 'Lymphatic_filariasis',
            'Onchocerciasis', 'Schistosomiasis', 'Trachoma', 'Rabies',
            'African_trypanosomiasis', 'Guinea_worm_disease', 'Cysticercosis',
            'Yellow_fever', 'Typhoid_and_paratyphoid', 'Malaria',
            'Leprosy', 'Food-borne_trematodiases', 'Other_neglected_tropical_diseases'
        ]);
        const diseases = DATA.integrated.diseases;
        const ntdSII = diseases.filter(d => ntds.has(d.disease) && d.sii).map(d => d.sii);
        const otherSII = diseases.filter(d => !ntds.has(d.disease) && d.sii).map(d => d.sii);
        const ntdMean = ntdSII.length ? ntdSII.reduce((a,b) => a+b, 0) / ntdSII.length : 0;
        const otherMean = otherSII.length ? otherSII.reduce((a,b) => a+b, 0) / otherSII.length : 0;
        const isolationPct = otherMean > 0 ? Math.round(((ntdMean - otherMean) / otherMean) * 100) : 20;
        setText('fw-knowledge-isolation', isolationPct);

        renderMiniIsolationChart(diseases);
    }
}


// ============================================================
// SECTION 3: RANKINGS RENDERING
// ============================================================
function renderSection3Rankings() {
    if (!DATA.integrated?.diseases) return;

    const diseases = DATA.integrated.diseases;
    renderUnifiedRankingChart(diseases);

    // Update summary stats
    const scored = diseases.filter(d => !HEIMEngine.INJURIES.has(d.disease) && d.unified_score != null);
    const scores = scored.map(d => d.unified_score);
    const mean = scores.length ? scores.reduce((a,b) => a+b, 0) / scores.length : 0;
    const min = scores.length ? Math.min(...scores) : 0;
    const max = scores.length ? Math.max(...scores) : 0;
    const threeDim = scored.filter(d => d.dimensions_available === 3).length;

    setText('rank-mean', mean.toFixed(1));
    document.getElementById('rank-range').innerHTML = min.toFixed(1) + ' &ndash; ' + max.toFixed(1);
    setText('rank-3d-count', threeDim);
}


// ============================================================
// SECTION 4: EQUITY RENDERING
// ============================================================
function renderSection4Equity() {
    if (!DATA.clinicalTrials) return;

    const ct = DATA.clinicalTrials;
    renderIntensityGapChart(ct);

    if (ct.geographic) {
        setText('eq-ct-ratio', ct.geographic.hicLmicRatio + ':1');
    }
    if (ct.keyFindings?.cancerDominance) {
        setText('eq-cancer-pct', ct.keyFindings.cancerDominance.cancerPct + '%');
    }

    // New interactive charts
    renderTrialSitesChart(ct);

    if (DATA.summary) {
        renderRegionalPubsChart(DATA.summary);
    }

    if (DATA.integrated?.diseases) {
        renderNTDIsolationChart(DATA.integrated.diseases);
    }
}


// ============================================================
// WEIGHT SLIDERS (Section 5a)
// ============================================================
function setupWeightSliders() {
    ['discovery', 'translation', 'knowledge'].forEach(dim => {
        const slider = document.getElementById(`slider-w-${dim}`);
        if (slider) {
            slider.addEventListener('input', () => onUnifiedWeightChange());
        }
    });
}

function onUnifiedWeightChange() {
    const dRaw = parseInt(document.getElementById('slider-w-discovery').value);
    const tRaw = parseInt(document.getElementById('slider-w-translation').value);
    const kRaw = parseInt(document.getElementById('slider-w-knowledge').value);
    const total = dRaw + tRaw + kRaw;

    if (total === 0) return;

    WEIGHTS.unified.discovery = dRaw / total;
    WEIGHTS.unified.translation = tRaw / total;
    WEIGHTS.unified.knowledge = kRaw / total;

    setText('val-w-discovery', WEIGHTS.unified.discovery.toFixed(3));
    setText('val-w-translation', WEIGHTS.unified.translation.toFixed(3));
    setText('val-w-knowledge', WEIGHTS.unified.knowledge.toFixed(3));
    setText('weight-sum', (WEIGHTS.unified.discovery + WEIGHTS.unified.translation + WEIGHTS.unified.knowledge).toFixed(3));

    updateFormulaDisplay();
    updateWeightStatus();
    recalculateAndUpdate();
}

function updateFormulaDisplay() {
    const el = document.getElementById('formula-unified');
    if (el) {
        el.innerHTML = `<code>Unified = <span class="fw">${WEIGHTS.unified.discovery.toFixed(3)}</span> &times; D_norm + <span class="fw">${WEIGHTS.unified.translation.toFixed(3)}</span> &times; T_norm + <span class="fw">${WEIGHTS.unified.knowledge.toFixed(3)}</span> &times; K_norm</code>`;
    }
}

function updateWeightStatus() {
    const def = HEIMEngine.DEFAULTS.unified;
    const isDefault = (
        Math.abs(WEIGHTS.unified.discovery - def.discovery) < 0.01 &&
        Math.abs(WEIGHTS.unified.translation - def.translation) < 0.01 &&
        Math.abs(WEIGHTS.unified.knowledge - def.knowledge) < 0.01
    );

    const statusEl = document.getElementById('weight-status');
    if (statusEl) {
        statusEl.textContent = isDefault ? 'Using published PCA weights' : 'Custom weights (modified)';
        statusEl.className = 'weight-status-text' + (isDefault ? '' : ' modified');
    }
}

function resetAllWeights() {
    const def = HEIMEngine.DEFAULTS;

    WEIGHTS.unified = { ...def.unified };
    WEIGHTS.eas = { ...def.eas };
    WEIGHTS.burden = { ...def.burden };

    const total = def.unified.discovery + def.unified.translation + def.unified.knowledge;
    document.getElementById('slider-w-discovery').value = Math.round(def.unified.discovery / total * 100);
    document.getElementById('slider-w-translation').value = Math.round(def.unified.translation / total * 100);
    document.getElementById('slider-w-knowledge').value = Math.round(def.unified.knowledge / total * 100);

    setText('val-w-discovery', def.unified.discovery.toFixed(3));
    setText('val-w-translation', def.unified.translation.toFixed(3));
    setText('val-w-knowledge', def.unified.knowledge.toFixed(3));
    setText('weight-sum', '1.000');

    updateFormulaDisplay();
    updateWeightStatus();
    recalculateAndUpdate();
}

function recalculateAndUpdate() {
    if (!DATA.integrated?.diseases) return;

    const result = HEIMEngine.recalculateAll(DATA, WEIGHTS);
    DATA.integrated.diseases = result.diseases;
    DATA.integrated.summary = result.summary;

    renderLiveRankings(result.diseases);
    renderUnifiedRankingChart(result.diseases);
}


// ============================================================
// LIVE RANKINGS TABLE (Section 5a)
// ============================================================
function renderLiveRankings(diseases) {
    const tbody = document.querySelector('#table-live-rankings tbody');
    if (!tbody || !diseases) return;

    const diseaseOnly = diseases
        .filter(d => !HEIMEngine.INJURIES.has(d.disease) && d.unified_score != null)
        .sort((a, b) => b.unified_score - a.unified_score);
    const top15 = diseaseOnly.slice(0, 15);

    const baselineLookup = {};
    if (BASELINE_SCORES) {
        BASELINE_SCORES.forEach(b => { baselineLookup[b.disease] = b.unified_score; });
    }

    // Check if weights have been modified
    const def = HEIMEngine.DEFAULTS.unified;
    const isModified = (
        Math.abs(WEIGHTS.unified.discovery - def.discovery) > 0.01 ||
        Math.abs(WEIGHTS.unified.translation - def.translation) > 0.01 ||
        Math.abs(WEIGHTS.unified.knowledge - def.knowledge) > 0.01
    );

    // Show Change column only when weights differ from defaults
    const thead = document.querySelector('#table-live-rankings thead tr');
    if (thead) {
        thead.innerHTML = isModified
            ? '<th>Rank</th><th>Disease</th><th>Unified Score</th><th>Change</th>'
            : '<th>Rank</th><th>Disease</th><th>Unified Score</th>';
    }

    tbody.innerHTML = top15.map((d, i) => {
        const name = d.disease.replace(/_/g, ' ');
        const score = d.unified_score?.toFixed(1) || '--';

        if (!isModified) {
            return `<tr><td>${i + 1}</td><td>${name}</td><td>${score}</td></tr>`;
        }

        const baseline = baselineLookup[d.disease];
        const change = baseline != null && d.unified_score != null ? d.unified_score - baseline : 0;
        const changeStr = change > 0.1 ? `+${change.toFixed(1)}` : change < -0.1 ? change.toFixed(1) : '-';
        const changeClass = change > 0.1 ? 'change-up' : change < -0.1 ? 'change-down' : '';

        return `<tr><td>${i + 1}</td><td>${name}</td><td>${score}</td><td class="${changeClass}">${changeStr}</td></tr>`;
    }).join('');
}


// ============================================================
// SCENARIO BUILDER (Section 5b)
// ============================================================
function runScenario(scenarioId) {
    if (!DATA.integrated?.diseases) return;

    const scenario = HEIMEngine.SCENARIOS[scenarioId];
    if (!scenario) return;

    // Compute baseline
    const baseline = HEIMEngine.computeUnifiedScores(DATA.integrated.diseases, WEIGHTS.unified);
    const baselineSorted = baseline
        .filter(d => !HEIMEngine.INJURIES.has(d.disease) && d.unified_score != null)
        .sort((a, b) => b.unified_score - a.unified_score);

    // Apply scenario
    const after = HEIMEngine.applyScenario(DATA.integrated.diseases, scenario, WEIGHTS.unified);
    const afterSorted = after
        .filter(d => !HEIMEngine.INJURIES.has(d.disease) && d.unified_score != null)
        .sort((a, b) => b.unified_score - a.unified_score);

    // Show results
    const resultsEl = document.getElementById('scenario-results');
    if (resultsEl) resultsEl.classList.add('visible');

    setText('scenario-name', scenario.name);
    setText('scenario-desc', scenario.description);

    // Highlight active card
    document.querySelectorAll('.scenario-card').forEach(c => c.classList.remove('scenario-active'));
    const activeCard = document.querySelector(`[data-scenario="${scenarioId}"]`);
    if (activeCard) activeCard.classList.add('scenario-active');

    // Before table
    const beforeTbody = document.querySelector('#table-scenario-before tbody');
    if (beforeTbody) {
        beforeTbody.innerHTML = baselineSorted.slice(0, 15).map((d, i) =>
            `<tr><td>${i + 1}</td><td>${d.disease.replace(/_/g, ' ')}</td><td>${d.unified_score.toFixed(1)}</td></tr>`
        ).join('');
    }

    // After table
    const afterTbody = document.querySelector('#table-scenario-after tbody');
    if (afterTbody) {
        const baselineRank = {};
        baselineSorted.forEach((d, i) => { baselineRank[d.disease] = i + 1; });

        // Build lookup of baseline scores
        const baseScoreLookup = {};
        baselineSorted.forEach(d => { baseScoreLookup[d.disease] = d.unified_score; });

        // Sort by largest absolute change, then show top 15 most affected
        const withDelta = afterSorted.map(d => {
            const oldScore = baseScoreLookup[d.disease] || 0;
            return { ...d, delta: d.unified_score - oldScore };
        });
        const byImpact = withDelta
            .filter(d => Math.abs(d.delta) > 0.01)
            .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
            .slice(0, 15);

        afterTbody.innerHTML = byImpact.map((d, i) => {
            const changeStr = d.delta < 0 ? d.delta.toFixed(1) : `+${d.delta.toFixed(1)}`;
            const changeClass = d.delta < -0.01 ? 'change-down' : d.delta > 0.01 ? 'change-up' : '';
            return `<tr><td>${i + 1}</td><td>${d.disease.replace(/_/g, ' ')}</td><td>${d.unified_score.toFixed(1)}</td><td class="${changeClass}">${changeStr}</td></tr>`;
        }).join('');
    }

    // Compute Spearman rho
    const baseNames = baselineSorted.map(d => d.disease);
    const baseScores = baselineSorted.map(d => d.unified_score);
    const afterScoresAligned = baseNames.map(name => {
        const found = afterSorted.find(d => d.disease === name);
        return found ? found.unified_score : 0;
    });
    const rho = HEIMEngine.spearmanRho(baseScores, afterScoresAligned);

    // Count rank changes
    const afterRank = {};
    afterSorted.forEach((d, i) => { afterRank[d.disease] = i + 1; });
    let moved = 0, maxMove = 0;
    baselineSorted.forEach((d, i) => {
        const newRank = afterRank[d.disease] || baselineSorted.length;
        const diff = Math.abs(newRank - (i + 1));
        if (diff > 0) moved++;
        maxMove = Math.max(maxMove, diff);
    });

    setText('scenario-rho', rho.toFixed(3));
    setText('scenario-moved', moved);
    setText('scenario-max-move', maxMove);

    // ---- Verification trace ----
    renderVerificationTrace(scenario, DATA.integrated.diseases, baselineSorted, afterSorted, rho, moved, maxMove);
}


// ============================================================
// VERIFICATION TRACE (appended to scenario results)
// ============================================================
function renderVerificationTrace(scenario, originalDiseases, baselineSorted, afterSorted, rho, moved, maxMove) {
    const body = document.getElementById('verification-body');
    if (!body) return;

    // Build baseline score lookup
    const baseScoreLookup = {};
    baselineSorted.forEach(d => { baseScoreLookup[d.disease] = d.unified_score; });

    // Find the most-affected disease
    const withDelta = afterSorted.map(d => ({
        ...d,
        delta: d.unified_score - (baseScoreLookup[d.disease] || 0)
    }));
    const affected = withDelta.filter(d => Math.abs(d.delta) > 0.01);
    const topAffected = affected.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))[0];

    if (!topAffected) {
        body.innerHTML = '<p>No diseases affected by this scenario.</p>';
        return;
    }

    // Get original data for the most-affected disease
    const orig = originalDiseases.find(d => d.disease === topAffected.disease);
    if (!orig) return;

    const name = topAffected.disease.replace(/_/g, ' ');
    const isTargeted = scenario.filter(orig);

    // Compute expected modified values
    const expectedGap = (orig.gap_score != null && scenario.gapReduction != null && isTargeted)
        ? orig.gap_score * (1 - scenario.gapReduction)
        : orig.gap_score;
    const expectedCt = (orig.ct_equity != null && scenario.pubsMultiplier != null && isTargeted)
        ? orig.ct_equity * scenario.pubsMultiplier
        : orig.ct_equity;

    // Format helpers
    const fmt = (v, dec) => v != null ? v.toFixed(dec) : 'null';
    const fmtDelta = (v) => {
        if (v == null) return '';
        const sign = v > 0 ? '+' : '';
        const cls = v < -0.01 ? 'negative' : v > 0.01 ? 'positive' : '';
        return `<span class="vr-delta ${cls}">${sign}${v.toFixed(1)}</span>`;
    };

    // Build multiplier labels
    const gapLabel = scenario.gapReduction != null
        ? `&times;${(1 - scenario.gapReduction).toFixed(1)}`
        : 'unchanged';
    const ctLabel = scenario.pubsMultiplier != null
        ? `&times;${scenario.pubsMultiplier.toFixed(1)}`
        : 'unchanged';

    body.innerHTML = `
        <div class="vr-section">
            <div class="vr-heading">Manual verification: ${name}</div>
            <div class="vr-row">
                <span class="vr-label">Targeted by scenario</span>
                <span>${isTargeted ? 'Yes' : 'No'}</span>
            </div>
            <div class="vr-row">
                <span class="vr-label">gap_score</span>
                <span>${fmt(orig.gap_score, 1)} <span class="vr-arrow">&rarr;</span> ${fmt(expectedGap, 1)} (${gapLabel})</span>
            </div>
            <div class="vr-row">
                <span class="vr-label">ct_equity</span>
                <span>${orig.ct_equity != null ? fmt(orig.ct_equity, 2) : 'null'} <span class="vr-arrow">&rarr;</span> ${expectedCt != null ? fmt(expectedCt, 2) : 'null'} (${ctLabel})</span>
            </div>
            <div class="vr-row">
                <span class="vr-label">SII</span>
                <span>${orig.sii != null ? fmt(orig.sii, 6) : 'null'} (unchanged)</span>
            </div>
            <div class="vr-row">
                <span class="vr-label">Unified score</span>
                <span>${fmt(baseScoreLookup[topAffected.disease], 1)} <span class="vr-arrow">&rarr;</span> ${fmt(topAffected.unified_score, 1)} ${fmtDelta(topAffected.delta)}</span>
            </div>
        </div>
        <div class="vr-summary">
            <span class="vr-affected-count">${affected.length}</span> of ${afterSorted.length} diseases affected
            &middot; Spearman &rho; = ${rho.toFixed(4)}
            &middot; ${moved} rank changes, largest shift: ${maxMove} positions<br>
            Normalisation ranges fixed to baseline (pre-scenario) values to ensure deltas reflect actual changes.
        </div>
    `;

    // Collapse by default
    const details = document.getElementById('scenario-verification');
    if (details) details.removeAttribute('open');
}


// ============================================================
// BIOBANK COMPARISON (Section 5d)
// ============================================================
function renderBiobankChecklist() {
    if (!DATA.biobanks?.biobanks) return;

    const checklist = document.getElementById('compare-checklist');
    if (!checklist) return;

    // Group biobanks by WHO region, sorted alphabetically
    const biobanks = [...DATA.biobanks.biobanks].sort((a, b) => {
        const regionCmp = (a.regionName || '').localeCompare(b.regionName || '');
        if (regionCmp !== 0) return regionCmp;
        return (a.name || '').localeCompare(b.name || '');
    });

    const regionGroups = {};
    biobanks.forEach(b => {
        const r = b.regionName || 'Unknown';
        if (!regionGroups[r]) regionGroups[r] = [];
        regionGroups[r].push(b);
    });

    const regionOrder = Object.keys(regionGroups).sort();

    checklist.innerHTML = regionOrder.map(region => {
        const members = regionGroups[region];
        const items = members.map(b =>
            `<label class="compare-check-item">
                <input type="checkbox" class="compare-checkbox" value="${b.id}">
                <span>${b.name} (${b.country})</span>
                <span class="compare-check-eas">EAS: ${b.scores?.equityAlignment?.toFixed(1) || '--'}</span>
            </label>`
        ).join('');
        return `<div class="compare-region-group">
            <div class="compare-region-header">${region}<span class="region-count">(${members.length})</span></div>
            ${items}
        </div>`;
    }).join('');

    // Filter handler
    const searchInput = document.getElementById('compare-biobank-search');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            const q = searchInput.value.toLowerCase();
            checklist.querySelectorAll('.compare-region-group').forEach(group => {
                const items = group.querySelectorAll('.compare-check-item');
                let anyVisible = false;
                items.forEach(item => {
                    const match = item.textContent.toLowerCase().includes(q);
                    item.style.display = match ? '' : 'none';
                    if (match) anyVisible = true;
                });
                group.style.display = anyVisible ? '' : 'none';
            });
        });
    }
}

function clearBiobankSelection() {
    document.querySelectorAll('.compare-checkbox').forEach(cb => { cb.checked = false; });
    const results = document.getElementById('compare-results');
    if (results) results.classList.remove('visible');
}

function runBiobankComparison() {
    const checked = Array.from(document.querySelectorAll('.compare-checkbox:checked')).map(cb => cb.value);
    if (checked.length < 2 || checked.length > 5) {
        alert('Please select 2-5 biobanks to compare.');
        return;
    }

    const biobanks = checked.map(id => DATA.biobanks.biobanks.find(b => b.id === id)).filter(Boolean);

    const results = document.getElementById('compare-results');
    if (results) results.classList.add('visible');

    renderCompareRadar(biobanks);
    renderCompareTable(biobanks);
}

function renderCompareTable(biobanks) {
    const headerRow = document.getElementById('compare-header-row');
    const tbody = document.querySelector('#table-compare-biobanks tbody');
    if (!headerRow || !tbody) return;

    headerRow.innerHTML = '<th>Metric</th>' + biobanks.map(b => `<th>${b.name}</th>`).join('');

    // Colour helpers: green = good, red = bad
    const colorGood = '#059669';
    const colorMid = '#d97706';
    const colorBad = '#dc2626';

    function easColor(v) {
        const n = parseFloat(v);
        if (isNaN(n)) return '';
        if (n >= 60) return colorGood;
        if (n >= 30) return colorMid;
        return colorBad;
    }
    function inverseColor(v, lo, hi) {
        // Lower is better (gaps, severity, burden miss)
        const n = parseFloat(v);
        if (isNaN(n)) return '';
        if (n <= lo) return colorGood;
        if (n <= hi) return colorMid;
        return colorBad;
    }
    function pubsColor(b) {
        const n = b.stats?.totalPublications || 0;
        if (n >= 1000) return colorGood;
        if (n >= 100) return colorMid;
        return colorBad;
    }
    function diseasesColor(v) {
        const n = parseInt(v);
        if (isNaN(n)) return '';
        if (n >= 100) return colorGood;
        if (n >= 50) return colorMid;
        return colorBad;
    }
    function categoryLabel(cat) {
        if (!cat || cat === '--') return '--';
        if (cat === 'High') return '<span style="color:#059669;font-weight:600;">High equity</span>';
        if (cat === 'Moderate') return '<span style="color:#d97706;font-weight:600;">Moderate equity</span>';
        return '<span style="color:#dc2626;font-weight:600;">Low equity</span>';
    }

    const metrics = [
        { label: 'Country', fn: b => b.country, color: () => '' },
        { label: 'WHO Region', fn: b => b.regionName, color: () => '' },
        { label: 'Equity Alignment Score', fn: b => b.scores?.equityAlignment?.toFixed(1) || '--', color: (v) => easColor(v) },
        { label: 'Equity Level', fn: b => b.scores?.equityCategory || '--', color: () => '', html: true },
        { label: 'Publications', fn: b => formatNumber(b.stats?.totalPublications), color: (v, b) => pubsColor(b) },
        { label: 'Diseases Covered', fn: b => b.stats?.diseasesCovered || '--', color: (v) => diseasesColor(v) },
        { label: 'Critical Gaps', fn: b => b.stats?.criticalGaps || '--', color: (v) => inverseColor(v, 30, 80) },
        { label: 'Gap Severity', fn: b => b.components?.gap_severity_component?.toFixed(1) || '--', color: (v) => inverseColor(v, 30, 60) },
    ];

    tbody.innerHTML = metrics.map(m => {
        const cells = biobanks.map(b => {
            const v = m.fn(b);
            if (m.html) {
                return `<td>${m.label === 'Equity Level' ? categoryLabel(v) : v}</td>`;
            }
            const c = m.color(v, b);
            return c ? `<td style="color:${c};font-weight:600;">${v}</td>` : `<td>${v}</td>`;
        });
        return `<tr><td><strong>${m.label}</strong></td>${cells.join('')}</tr>`;
    }).join('');
}


// ============================================================
// EXPORT FUNCTIONS
// ============================================================
function exportWeightedCSV() {
    if (!DATA.integrated?.diseases) return;

    const diseases = DATA.integrated.diseases
        .filter(d => d.unified_score != null)
        .sort((a, b) => b.unified_score - a.unified_score);

    const weightHeader = `# Weights: D=${WEIGHTS.unified.discovery.toFixed(3)} T=${WEIGHTS.unified.translation.toFixed(3)} K=${WEIGHTS.unified.knowledge.toFixed(3)}`;
    const headers = ['Rank', 'Disease', 'Gap Score', 'CT Equity', 'SII', 'Unified Score', 'Dimensions'];
    let csv = weightHeader + '\n' + headers.join(',') + '\n';

    diseases.forEach((d, i) => {
        csv += [
            i + 1,
            `"${d.disease.replace(/_/g, ' ')}"`,
            d.gap_score?.toFixed(1) || '',
            d.ct_equity?.toFixed(1) || '',
            d.sii?.toFixed(6) || '',
            d.unified_score?.toFixed(2) || '',
            d.dimensions_available || ''
        ].join(',') + '\n';
    });

    downloadFile(csv, 'heim-weighted-rankings.csv', 'text/csv');
}

function exportPDFReport() {
    const printContent = `
        <html><head><title>HEIM Report</title>
        <style>body{font-family:Georgia,serif;padding:2rem;max-width:800px;margin:0 auto;}
        table{margin:1rem 0;width:100%;border-collapse:collapse;}
        th,td{padding:0.5rem;text-align:left;border-bottom:1px solid #e5e7eb;}
        th{background:#f1f3f5;font-size:0.85rem;}</style></head><body>
        <h1>HEIM Framework - Analysis Report</h1>
        <p><strong>Generated:</strong> ${new Date().toISOString().split('T')[0]}</p>
        <p><strong>Weights:</strong> D=${WEIGHTS.unified.discovery.toFixed(3)}, T=${WEIGHTS.unified.translation.toFixed(3)}, K=${WEIGHTS.unified.knowledge.toFixed(3)}</p>
        <h2>Top 20 Most Neglected Diseases</h2>
        <table>
            <tr><th>Rank</th><th>Disease</th><th>Score</th><th>Dims</th></tr>
            ${(DATA.integrated?.diseases || [])
                .filter(d => !HEIMEngine.INJURIES.has(d.disease) && d.unified_score != null)
                .slice(0, 20)
                .map((d, i) => `<tr><td>${i+1}</td><td>${d.disease.replace(/_/g,' ')}</td><td>${d.unified_score.toFixed(1)}</td><td>${d.dimensions_available}</td></tr>`)
                .join('')}
        </table>
        <p style="margin-top:2rem;font-size:0.8rem;color:#6b7280;">Corpas, Freidin, Valdivia-Silva, Baker, Fatumo &amp; Guio (2026). Three Dimensions of Compounding Neglect. https://manuelcorpas.github.io/17-EHR/</p>
        </body></html>`;

    const printWin = window.open('', '_blank');
    printWin.document.write(printContent);
    printWin.document.close();
    printWin.print();
}

function copyCitation() {
    const text = document.querySelector('#citation-text p')?.textContent || '';
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.citation-copy-btn');
        if (btn) {
            btn.textContent = 'Copied';
            setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
        }
    });
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
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

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}
