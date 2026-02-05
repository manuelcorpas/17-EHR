// HEIM Framework v3.0 — Computation Engine
// Ports all formulas from Python (03-03, 06-10) to JavaScript for client-side recalculation
// All pure arithmetic — full recalculation across 175 diseases × 70 biobanks in <5ms

'use strict';

const HEIMEngine = (() => {

    // =========================================================================
    // DEFAULT WEIGHTS (from PCA analysis — PC1 loadings from 86 complete cases)
    // =========================================================================
    const DEFAULTS = Object.freeze({
        // Unified Score weights (PCA-derived)
        unified: { discovery: 0.501, translation: 0.293, knowledge: 0.206 },
        // EAS component weights
        eas: { gap: 0.4, burdenMiss: 0.3, capacity: 0.3 },
        // Burden Score coefficients
        burden: { dalys: 0.5, deaths: 50.0, prevalence: 10.0 },
        // Gap Score thresholds
        gapThresholds: {
            infectious: [
                { maxPubs: 0,   score: 95 },
                { maxPubs: 10,  score: 90 },
                { maxPubs: 25,  score: 80 },
                { maxPubs: 50,  score: 70 },
                { maxPubs: 100, score: 60 }
            ],
            neglected: [
                { maxPubs: 0,   score: 95 },
                { maxPubs: 10,  score: 92 },
                { maxPubs: 25,  score: 82 },
                { maxPubs: 50,  score: 72 },
                { maxPubs: 100, score: 65 }
            ],
            intensityTiers: [
                { minIntensity: 100, score: 10 },
                { minIntensity: 50,  score: 20 },
                { minIntensity: 25,  score: 30 },
                { minIntensity: 10,  score: 40 },
                { minIntensity: 5,   score: 50 },
                { minIntensity: 2,   score: 60 },
                { minIntensity: 1,   score: 70 },
                { minIntensity: 0.5, score: 80 }
            ],
            defaultScore: 85,
            zeroPubScore: 95,
            gsPenalty: 10,
            gsPenaltyThreshold: 50
        }
    });

    // Category classifications
    const INFECTIOUS_CATEGORIES = new Set(['Infectious', 'Neglected']);
    const NEGLECTED_CATEGORIES = new Set(['Neglected', 'Maternal/Child', 'Nutritional']);

    // GBD injury/external cause categories (excluded from disease rankings)
    const INJURIES = new Set([
        'Conflict_and_terrorism', 'Animal_contact', 'Drowning', 'Road_injuries',
        'Self-harm', 'Interpersonal_violence', 'Falls', 'Fire,_heat,_and_hot_substances',
        'Poisonings', 'Exposure_to_mechanical_forces', 'Foreign_body',
        'Environmental_heat_and_cold_exposure', 'Other_transport_injuries',
        'Executions_and_police_conflict', 'Other_unintentional_injuries',
        'Adverse_effects_of_medical_treatment', 'Paralytic_ileus_and_intestinal_obstruction'
    ]);

    // =========================================================================
    // BURDEN SCORE
    // Formula: 0.5 × DALYs_M + 50 × Deaths_M + 10 × log₁₀(Prevalence_M)
    // Fallback: 10 × log₁₀(DALYs_raw + 1)
    // =========================================================================
    function computeBurdenScore(dalysMil, deathsMil, prevalenceMil, coefficients) {
        const c = coefficients || DEFAULTS.burden;

        if (deathsMil != null && prevalenceMil != null) {
            const prev = prevalenceMil <= 0 ? 1.0 : prevalenceMil;
            return round(c.dalys * dalysMil + c.deaths * deathsMil + c.prevalence * Math.log10(prev), 2);
        }
        // Fallback: DALYs-only
        if (dalysMil <= 0) return 0;
        return round(10.0 * Math.log10(dalysMil * 1e6 + 1), 2);
    }

    // =========================================================================
    // RESEARCH GAP SCORE (three-tier)
    // Tier 1: Zero-pub → 95
    // Tier 2: Category thresholds (Infectious/Neglected)
    // Tier 3: Burden-normalised intensity deciles
    // + Global South penalty (+10 if <50 pubs)
    // =========================================================================
    function computeGapScore(publications, dalysMil, isGS, category, thresholds) {
        const t = thresholds || DEFAULTS.gapThresholds;

        // Tier 1
        if (publications === 0) return { score: t.zeroPubScore, severity: 'Critical' };

        let score = null;

        // Tier 2: category thresholds
        if (INFECTIOUS_CATEGORIES.has(category)) {
            for (const tier of t.infectious) {
                if (publications <= tier.maxPubs) { score = tier.score; break; }
            }
        } else if (NEGLECTED_CATEGORIES.has(category)) {
            for (const tier of t.neglected) {
                if (publications <= tier.maxPubs) { score = tier.score; break; }
            }
        }

        // Tier 3: burden-normalised intensity
        if (score === null) {
            const intensity = dalysMil > 0 ? publications / dalysMil : publications;
            score = t.defaultScore;
            for (const tier of t.intensityTiers) {
                if (intensity >= tier.minIntensity) { score = tier.score; break; }
            }
        }

        // Global South penalty
        if (isGS && publications < (t.gsPenaltyThreshold || 50)) {
            score = Math.min(95, score + (t.gsPenalty || 10));
        }

        const severity = score > 70 ? 'Critical' : score > 50 ? 'High' : score > 30 ? 'Moderate' : 'Low';
        return { score: round(score, 1), severity };
    }

    // =========================================================================
    // EQUITY ALIGNMENT SCORE (per biobank)
    // EAS = 100 - (w_gap × GapSeverity + w_burden × BurdenMiss + w_capacity × CapacityPenalty)
    // =========================================================================
    function computeEAS(biobankDiseasePubs, diseaseMetrics, totalPubs, weights) {
        const w = weights || DEFAULTS.eas;

        let nCritical = 0, nHigh = 0, nModerate = 0, nLow = 0;
        let missedDALYs = 0, totalDALYs = 0;
        let diseasesCovered = 0;
        const nDiseases = Object.keys(diseaseMetrics).length;

        for (const [diseaseId, metrics] of Object.entries(diseaseMetrics)) {
            const pubs = biobankDiseasePubs[diseaseId] || 0;
            const dalys = metrics.dalys_millions || metrics.dalysMil || 0;
            totalDALYs += dalys;

            if (pubs > 0) diseasesCovered++;

            if (pubs === 0) {
                nCritical++;
                missedDALYs += dalys;
            } else if (pubs <= 2) {
                const sev = metrics.gap_severity || metrics.severity || '';
                if (sev === 'Critical') nCritical++;
                else if (sev === 'High') nHigh++;
                else nModerate++;
                missedDALYs += dalys;
            } else if (pubs <= 10) {
                const sev = metrics.gap_severity || metrics.severity || '';
                if (sev === 'High') nHigh++;
                else if (sev === 'Moderate') nModerate++;
                else nLow++;
            } else {
                nLow++;
            }
        }

        const weightedGaps = 4 * nCritical + 2 * nHigh + 1 * nModerate;
        const maxGaps = 4 * nDiseases;
        const gapSeverityComponent = maxGaps > 0 ? Math.min(100, (weightedGaps / maxGaps) * 100) : 0;
        const burdenMissComponent = totalDALYs > 0 ? (missedDALYs / totalDALYs) * 100 : 0;
        const pubsPerDisease = nDiseases > 0 ? totalPubs / nDiseases : 0;
        const capacityPenalty = 100 - Math.min(pubsPerDisease, 100);

        let eas = 100 - (
            w.gap * gapSeverityComponent +
            w.burdenMiss * burdenMissComponent +
            w.capacity * capacityPenalty
        );
        eas = clamp(eas, 0, 100);

        const category = eas >= 70 ? 'High' : eas >= 40 ? 'Moderate' : 'Low';

        return {
            score: round(eas, 1),
            category,
            components: {
                gapSeverity: round(gapSeverityComponent, 2),
                burdenMiss: round(burdenMissComponent, 2),
                capacityPenalty: round(capacityPenalty, 2),
                nCritical, nHigh, nModerate, nLow,
                diseasesCovered,
                missedDALYs: round(missedDALYs, 2)
            }
        };
    }

    // =========================================================================
    // UNIFIED SCORE (PCA-weighted)
    // Three dimensions: D (gap_score), T (1/(ct_intensity+1)), K (sii)
    // Unified = (w_D × D_norm + w_T × T_norm + w_K × K_norm) × 100
    // Two-dim fallback: reweight D and K proportionally
    // =========================================================================
    function computeUnifiedScores(diseases, weights) {
        const w = weights || DEFAULTS.unified;

        // Collect normalisation ranges
        const gaps = [], siis = [], ctInvs = [];
        for (const d of diseases) {
            if (d.gap_score != null) gaps.push(d.gap_score);
            if (d.sii != null) siis.push(d.sii);
            if (d.ct_equity != null) ctInvs.push(1.0 / (d.ct_equity + 1));
        }

        const gapMin = Math.min(...gaps), gapMax = Math.max(...gaps);
        const siiMin = Math.min(...siis), siiMax = Math.max(...siis);
        const ctMin = ctInvs.length ? Math.min(...ctInvs) : 0;
        const ctMax = ctInvs.length ? Math.max(...ctInvs) : 1;

        const results = [];
        for (const d of diseases) {
            if (d.gap_score == null || d.sii == null) {
                results.push({ ...d, unified_score: null, dimensions_available: 0 });
                continue;
            }

            const dN = norm(d.gap_score, gapMin, gapMax);
            const kN = norm(d.sii, siiMin, siiMax);

            let score, dims;
            if (d.ct_equity != null) {
                const tN = norm(1.0 / (d.ct_equity + 1), ctMin, ctMax);
                score = (w.discovery * dN + w.translation * tN + w.knowledge * kN) * 100;
                dims = 3;
            } else {
                // Proportional reweight for 2-dim diseases
                const wDK = w.discovery / (w.discovery + w.knowledge);
                const wKD = w.knowledge / (w.discovery + w.knowledge);
                score = (wDK * dN + wKD * kN) * 100;
                dims = 2;
            }

            results.push({ ...d, unified_score: score, dimensions_available: dims });
        }

        return results;
    }

    // =========================================================================
    // EQUITY RATIO
    // (Pubs_HIC / DALYs_HIC) / (Pubs_LMIC / DALYs_LMIC)
    // =========================================================================
    function computeEquityRatio(hicPubs, hicDALYs, lmicPubs, lmicDALYs) {
        if (hicDALYs <= 0 || lmicDALYs <= 0 || hicPubs <= 0 || lmicPubs <= 0) return Infinity;
        const hicIntensity = hicPubs / hicDALYs;
        const lmicIntensity = lmicPubs / lmicDALYs;
        return lmicIntensity <= 0 ? Infinity : round(hicIntensity / lmicIntensity, 2);
    }

    // =========================================================================
    // SENSITIVITY ANALYSIS — Spearman rho between two rankings
    // =========================================================================
    function spearmanRho(arr1, arr2) {
        if (arr1.length !== arr2.length || arr1.length < 2) return NaN;
        const n = arr1.length;
        const rank1 = rankArray(arr1);
        const rank2 = rankArray(arr2);
        let sumD2 = 0;
        for (let i = 0; i < n; i++) {
            const d = rank1[i] - rank2[i];
            sumD2 += d * d;
        }
        return 1 - (6 * sumD2) / (n * (n * n - 1));
    }

    function rankArray(arr) {
        const indexed = arr.map((v, i) => ({ v, i }));
        indexed.sort((a, b) => b.v - a.v); // descending
        const ranks = new Array(arr.length);
        for (let r = 0; r < indexed.length; r++) {
            ranks[indexed[r].i] = r + 1;
        }
        return ranks;
    }

    // =========================================================================
    // SCENARIO MODELLING
    // Apply a scenario (e.g., +50% publications for NTDs) and return new scores
    // =========================================================================
    function applyScenario(diseases, scenario, weights) {
        const modified = diseases.map(d => {
            const copy = { ...d };
            if (scenario.filter(copy)) {
                if (scenario.pubsMultiplier != null) {
                    // Increase ct_equity (trials per M DALYs) by multiplier
                    if (copy.ct_equity != null) {
                        copy.ct_equity = copy.ct_equity * scenario.pubsMultiplier;
                    }
                }
                if (scenario.gapReduction != null) {
                    if (copy.gap_score != null) {
                        copy.gap_score = Math.max(0, copy.gap_score * (1 - scenario.gapReduction));
                    }
                }
            }
            return copy;
        });
        return computeUnifiedScores(modified, weights);
    }

    // Built-in scenario presets
    const SCENARIOS = {
        increaseNTD50: {
            name: 'Increase NTD research +50%',
            description: 'Simulate 50% increase in clinical trial intensity for all NTDs',
            filter: d => {
                const ntds = new Set([
                    'Chagas_disease', 'Dengue', 'Leishmaniasis', 'Lymphatic_filariasis',
                    'Onchocerciasis', 'Schistosomiasis', 'Trachoma', 'Rabies',
                    'African_trypanosomiasis', 'Guinea_worm_disease', 'Cysticercosis',
                    'Yellow_fever', 'Typhoid_and_paratyphoid', 'Malaria',
                    'Leprosy', 'Food-borne_trematodiases', 'Other_neglected_tropical_diseases'
                ]);
                return ntds.has(d.disease);
            },
            pubsMultiplier: 1.5,
            gapReduction: 0.3
        },
        equaliseHICLMIC: {
            name: 'Equalise HIC/LMIC output',
            description: 'Simulate equalising research intensity between HIC and LMIC diseases',
            filter: d => d.gap_score != null && d.gap_score > 50,
            pubsMultiplier: 2.0,
            gapReduction: 0.4
        },
        doubleInfectious: {
            name: 'Double infectious disease research',
            description: 'Simulate doubling clinical trial intensity for infectious diseases',
            filter: d => {
                const name = (d.disease || '').toLowerCase();
                const infectious = ['malaria', 'tuberculosis', 'hiv', 'dengue', 'typhoid',
                    'leishmaniasis', 'chagas', 'schistosomiasis', 'meningitis',
                    'lower_respiratory_infections', 'diarrheal'];
                return infectious.some(inf => name.includes(inf));
            },
            pubsMultiplier: 2.0,
            gapReduction: 0.5
        },
        reduceCancerBias: {
            name: 'Reduce cancer research bias',
            description: 'Redistribute 30% of cancer trial intensity to neglected diseases',
            filter: d => d.gap_score != null && d.gap_score > 60,
            pubsMultiplier: 1.3,
            gapReduction: 0.2
        }
    };

    // =========================================================================
    // FULL RECALCULATION
    // Called when any weight changes — recomputes unified scores and stats
    // =========================================================================
    function recalculateAll(data, weights) {
        const unifiedWeights = weights?.unified || DEFAULTS.unified;
        const easWeights = weights?.eas || DEFAULTS.eas;
        const burdenCoeffs = weights?.burden || DEFAULTS.burden;

        // Recompute unified scores
        const updatedDiseases = computeUnifiedScores(data.integrated.diseases, unifiedWeights);

        // Sort by unified score (descending)
        updatedDiseases.sort((a, b) => {
            const sa = a.unified_score != null ? a.unified_score : -1;
            const sb = b.unified_score != null ? b.unified_score : -1;
            return sb - sa;
        });

        // Compute summary statistics (excluding injuries)
        const diseaseOnly = updatedDiseases.filter(d => !INJURIES.has(d.disease) && d.unified_score != null);
        const scores = diseaseOnly.map(d => d.unified_score);
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length);

        return {
            diseases: updatedDiseases,
            summary: {
                unified_score: {
                    mean: round(mean, 1),
                    std: round(std, 1),
                    min: round(Math.min(...scores), 1),
                    max: round(Math.max(...scores), 1)
                }
            },
            weights: {
                unified: { ...unifiedWeights },
                eas: { ...easWeights },
                burden: { ...burdenCoeffs }
            }
        };
    }

    // =========================================================================
    // SENSITIVITY SWEEP
    // Dirichlet-style: generate N random weight vectors, compute Spearman rho
    // =========================================================================
    function sensitivitySweep(diseases, nSamples = 200) {
        // Baseline (current default weights)
        const baseline = computeUnifiedScores(diseases, DEFAULTS.unified);
        const baselineScores = baseline
            .filter(d => !INJURIES.has(d.disease) && d.unified_score != null)
            .sort((a, b) => a.disease.localeCompare(b.disease))
            .map(d => d.unified_score);
        const baselineNames = baseline
            .filter(d => !INJURIES.has(d.disease) && d.unified_score != null)
            .sort((a, b) => a.disease.localeCompare(b.disease))
            .map(d => d.disease);

        const results = [];

        for (let i = 0; i < nSamples; i++) {
            // Generate random Dirichlet weights (using gamma distribution approximation)
            const alpha = [1, 1, 1]; // uniform Dirichlet
            const gammas = alpha.map(a => {
                let sum = 0;
                for (let j = 0; j < a; j++) sum += -Math.log(Math.random());
                return sum;
            });
            const total = gammas.reduce((a, b) => a + b, 0);
            const w = {
                discovery: gammas[0] / total,
                translation: gammas[1] / total,
                knowledge: gammas[2] / total
            };

            const recomputed = computeUnifiedScores(diseases, w);
            const reScores = recomputed
                .filter(d => !INJURIES.has(d.disease) && d.unified_score != null)
                .sort((a, b) => a.disease.localeCompare(b.disease))
                .map(d => d.unified_score);

            const rho = spearmanRho(baselineScores, reScores);

            results.push({
                weights: w,
                rho: round(rho, 4),
                distance: Math.sqrt(
                    (w.discovery - DEFAULTS.unified.discovery) ** 2 +
                    (w.translation - DEFAULTS.unified.translation) ** 2 +
                    (w.knowledge - DEFAULTS.unified.knowledge) ** 2
                )
            });
        }

        // Sort by distance from default
        results.sort((a, b) => a.distance - b.distance);

        const rhos = results.map(r => r.rho);
        return {
            samples: results,
            summary: {
                meanRho: round(rhos.reduce((a, b) => a + b, 0) / rhos.length, 4),
                minRho: round(Math.min(...rhos), 4),
                maxRho: round(Math.max(...rhos), 4),
                pctAbove90: round(rhos.filter(r => r >= 0.9).length / rhos.length * 100, 1),
                pctAbove95: round(rhos.filter(r => r >= 0.95).length / rhos.length * 100, 1)
            },
            baselineNames
        };
    }

    // =========================================================================
    // HELPERS
    // =========================================================================
    function norm(v, lo, hi) {
        return hi > lo ? (v - lo) / (hi - lo) : 0.5;
    }

    function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    function round(v, decimals) {
        const f = Math.pow(10, decimals);
        return Math.round(v * f) / f;
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================
    return {
        DEFAULTS,
        INJURIES,
        SCENARIOS,
        computeBurdenScore,
        computeGapScore,
        computeEAS,
        computeUnifiedScores,
        computeEquityRatio,
        spearmanRho,
        applyScenario,
        recalculateAll,
        sensitivitySweep,
        norm,
        clamp,
        round
    };

})();
