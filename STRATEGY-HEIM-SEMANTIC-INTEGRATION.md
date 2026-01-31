# HEIM Semantic Integration Strategy

**Version**: 2.0 (Architecture Validated)
**Date**: 2026-01-13
**Objective**: Integrate PubMedBERT semantic embeddings into HEIM v6 for Nature Medicine submission

---

## PART A: ARCHITECTURE REVIEW

### Critical Failure Points Identified

| # | Failure Point | Risk Level | Impact | Mitigation |
|---|--------------|------------|--------|------------|
| 1 | **PubMed API rate limiting** | HIGH | Job blocked/banned | Exponential backoff, API key, 0.4s delay between requests |
| 2 | **Network disconnection during fetch** | MEDIUM | Lost progress | Atomic checkpoints after each disease-year |
| 3 | **Memory exhaustion during embedding** | HIGH | Process killed | Stream processing, explicit gc.collect(), batch clearing |
| 4 | **MPS backend failures** | MEDIUM | Silent corruption | Validation against CPU reference, fallback mode |
| 5 | **HDF5 file corruption** | MEDIUM | Lost embeddings | Atomic writes (temp→rename), integrity checksums |
| 6 | **System sleep interruption** | HIGH | Job terminates | Built-in caffeinate, checkpoint resumability |
| 7 | **Disk space exhaustion** | MEDIUM | Write failures | Pre-flight disk check (require 100GB free) |
| 8 | **Malformed PubMed responses** | LOW | Bad data | Schema validation, graceful skip with logging |
| 9 | **Checkpoint corruption** | MEDIUM | Cannot resume | JSON schema validation, backup checkpoints |
| 10 | **Embedding drift across batches** | LOW | Inconsistent results | Fixed random seeds, model caching |
| 11 | **Metric calculation errors** | MEDIUM | Invalid results | Unit tests with known-answer validation |
| 12 | **Incomplete data propagation** | HIGH | Missing diseases | End-to-end validation before each phase |

---

## PART B: GUARDRAILS & VALIDATION FRAMEWORK

### B.1 Pre-Flight Checks (05-00-heim-sem-setup.py)

```
MANDATORY CHECKS (all must pass before proceeding):
□ Python ≥3.11
□ PyTorch ≥2.0 with MPS available
□ PubMedBERT model downloads and loads
□ 100GB+ free disk space
□ Required packages installed (with correct versions)
□ Directory structure created
□ NCBI API key configured (optional but recommended)
□ Test embedding on sample text matches expected output
□ HDF5 read/write test passes
□ Logging to file works
```

### B.2 Phase Gate Validation

Each script must validate prerequisites before running and outputs after completion:

```
PHASE 1 (Fetch) - 05-01-heim-sem-fetch.py
├── PRE: Setup complete (check marker file)
├── POST: All diseases have PUBMED-RAW/{Disease}/ directory
├── POST: heim_sem_quality_scores.csv has N rows (one per disease)
├── POST: Total papers ≥ 2,000,000 (sanity check)
└── POST: No disease has 0 papers unless quality-filtered

PHASE 2 (Embed) - 05-02-heim-sem-embed.py
├── PRE: PUBMED-RAW directories exist
├── PRE: Quality scores file exists
├── POST: EMBEDDINGS/{Disease}/ matches PUBMED-RAW/{Disease}/
├── POST: Each HDF5 file has shape (N, 768) where N > 0
├── POST: Embedding checksums stored for integrity
└── POST: Sample embeddings validated against reference

PHASE 3 (Metrics) - 05-03-heim-sem-compute-metrics.py
├── PRE: All EMBEDDINGS directories complete
├── POST: heim_sem_metrics.json exists and is valid JSON
├── POST: All expected metrics present (SII, KTP, RCC, drift)
├── POST: Metrics are within plausible ranges
└── POST: No NaN or Inf values in outputs

PHASE 4 (Figures) - 05-04-heim-sem-generate-figures.py
├── PRE: Metrics file exists
├── POST: All expected figure files generated
├── POST: Figure files are non-empty PNGs
└── POST: Metadata JSON for each figure exists

PHASE 5 (Integrate) - 05-05-heim-sem-integrate.py
├── PRE: bhem_metrics.json exists
├── PRE: heim_ct_metrics.json exists
├── PRE: heim_sem_metrics.json exists
├── POST: Integrated metrics cover all diseases
└── POST: Output tables are manuscript-ready
```

### B.3 Checkpoint & Recovery System

```python
# Checkpoint structure for each script
{
    "script": "05-01-heim-sem-fetch.py",
    "version": "1.0",
    "started_at": "2026-01-13T10:00:00",
    "last_updated": "2026-01-13T14:30:00",
    "completed_items": ["Malaria", "Tuberculosis", ...],
    "in_progress": "Dengue",
    "failed_items": [],
    "total_items": 68,
    "checksum": "sha256:abc123..."
}
```

**Recovery procedure:**
1. Script detects existing checkpoint on startup
2. Validates checkpoint integrity (checksum)
3. Resumes from last completed item
4. Creates backup of checkpoint before each update
5. Writes to temp file, then atomic rename

### B.4 Logging Strategy

```
LOGS/
├── 05-00-setup-{timestamp}.log          # Setup validation
├── 05-01-fetch-{timestamp}.log          # Fetch operations
├── 05-02-embed-{timestamp}.log          # Embedding generation
├── 05-03-metrics-{timestamp}.log        # Metric computation
├── 05-04-figures-{timestamp}.log        # Figure generation
├── 05-05-integrate-{timestamp}.log      # Integration
└── error-summary-{timestamp}.log        # Errors only (for quick review)
```

**Log levels:**
- DEBUG: Individual paper processing
- INFO: Disease completion, batch timing
- WARNING: Retries, skipped items, rate limit waits
- ERROR: Failures requiring attention
- CRITICAL: Pipeline cannot continue

---

## PART C: DETAILED IMPLEMENTATION

### C.1 Directory Architecture

```
DATA/
├── bhem_*.csv/json                          # UNCHANGED
├── heim_ct_*.csv/json                       # UNCHANGED
│
└── 05-SEMANTIC/
    ├── heim_sem_quality_scores.csv
    ├── heim_sem_disease_registry.json
    ├── heim_sem_metrics.json
    ├── .setup_complete                      # Marker file from 05-00
    ├── .fetch_complete                      # Marker file from 05-01
    ├── .embed_complete                      # Marker file from 05-02
    ├── CHECKPOINTS/
    │   ├── checkpoint_fetch.json
    │   ├── checkpoint_fetch.json.bak
    │   ├── checkpoint_embed.json
    │   └── checkpoint_embed.json.bak
    ├── PUBMED-RAW/
    │   └── {Disease_Name}/
    │       ├── pubmed_2000.json
    │       ├── pubmed_2001.json
    │       └── ...pubmed_2025.json
    └── EMBEDDINGS/
        └── {Disease_Name}/
            ├── embeddings.h5               # All years combined
            ├── embeddings.h5.sha256        # Integrity checksum
            └── metadata.json               # Paper count, timestamps

ANALYSIS/
├── 05-01-SEMANTIC-QUALITY/
├── 05-02-EMBEDDING-VALIDATION/
├── 05-03-SEMANTIC-METRICS/
└── 05-04-HEIM-SEM-FIGURES/

LOGS/
└── [timestamped log files]
```

### C.2 Script Specifications

#### 05-00-heim-sem-setup.py
```python
"""
Environment validation and directory setup

VALIDATES:
- Python version ≥3.11
- PyTorch ≥2.0 with MPS backend
- Required packages with minimum versions
- Disk space ≥100GB free
- PubMedBERT model loads correctly
- Test embedding matches reference (cosine sim > 0.99)
- HDF5 read/write operations work
- Logging system functional

CREATES:
- DATA/05-SEMANTIC/ directory structure
- DATA/05-SEMANTIC/.setup_complete marker
- ANALYSIS/05-XX directories
- LOGS/ directory

EXIT CODES:
- 0: All checks passed
- 1: Critical failure (cannot proceed)
- 2: Warning (can proceed with caution)
"""
```

#### 05-01-heim-sem-fetch.py
```python
"""
PubMed abstract retrieval with robust error handling

FEATURES:
- Exponential backoff (1s → 2s → 4s → 8s → 16s max)
- 0.4s minimum delay between requests (NCBI rate limit)
- Checkpoint after each disease-year completes
- Atomic checkpoint writes (temp → rename)
- Quality filtering based on thresholds
- Graceful handling of malformed responses
- Sleep prevention via subprocess caffeinate

QUALITY FILTERS:
- Year Coverage ≥ 70% (18+ years with data)
- Abstract Coverage ≥ 95%
- Minimum Papers ≥ 50
- Composite Quality Score ≥ 80.0

OUTPUTS:
- PUBMED-RAW/{Disease}/pubmed_{year}.json
- heim_sem_quality_scores.csv
- heim_sem_disease_registry.json
- .fetch_complete marker

RESUMABILITY:
- Detects existing checkpoint
- Skips completed diseases
- Resumes from last saved position
"""
```

#### 05-02-heim-sem-embed.py
```python
"""
PubMedBERT embedding generation optimized for M3 Ultra

FEATURES:
- MPS backend with CPU fallback
- Batch size 64 (optimized for 256GB RAM)
- Memory clearing after each batch
- Checkpoint after each disease completes
- HDF5 storage with compression
- Integrity checksums (SHA-256)
- Progress bars with ETA
- Validation: sample embeddings vs CPU reference

MEMORY MANAGEMENT:
- Load model once, reuse for all diseases
- Clear batch tensors immediately after embedding
- Explicit gc.collect() every 1000 batches
- Monitor memory usage, warn if >80%

MPS VALIDATION:
- First 10 embeddings computed on both MPS and CPU
- Cosine similarity must be > 0.9999
- If validation fails, fall back to CPU only

OUTPUTS:
- EMBEDDINGS/{Disease}/embeddings.h5
- EMBEDDINGS/{Disease}/embeddings.h5.sha256
- EMBEDDINGS/{Disease}/metadata.json
- .embed_complete marker

RESUMABILITY:
- Detects existing embeddings by disease
- Validates existing embeddings (checksum)
- Only processes missing diseases
"""
```

#### 05-03-heim-sem-compute-metrics.py
```python
"""
Semantic equity metric computation

METRICS:
1. Semantic Isolation Index (SII)
   - k-NN cosine distance (k=100)
   - Higher = more isolated research

2. Knowledge Transfer Potential (KTP)
   - Cross-disease centroid similarity
   - Higher = more potential for spillover

3. Research Clustering Coefficient (RCC)
   - Within-disease embedding variance
   - Higher = more dispersed research

4. Temporal Semantic Drift
   - Cosine distance between yearly centroids
   - Measures how research focus evolves

VALIDATION:
- All metrics bounded (no Inf/NaN)
- Cross-check: high SII correlates with low KTP
- Unit tests with synthetic embeddings

OUTPUTS:
- heim_sem_metrics.json
- ANALYSIS/05-03-SEMANTIC-METRICS/*.csv
"""
```

#### 05-04-heim-sem-generate-figures.py
```python
"""
Publication-ready figure generation

FIGURES:
1. fig_umap_disease_clusters.png
   - UMAP projection of all disease centroids
   - Colored by disease burden
   - Sized by research volume

2. fig_semantic_isolation_heatmap.png
   - 68×68 disease similarity matrix
   - Clustered by similarity
   - Burden overlay

3. fig_temporal_drift.png
   - Line plot of centroid movement over time
   - Highlight diseases with highest drift

4. fig_gap_vs_isolation.png
   - Scatter: Gap Score (x) vs SII (y)
   - Regression line and R²
   - Key outliers labeled

5. fig_knowledge_network.png
   - Network graph of high-KTP disease pairs
   - Node size = research volume
   - Edge weight = similarity

REPRODUCIBILITY:
- Random seed fixed (42)
- All parameters logged
- Metadata JSON for each figure

OUTPUTS:
- ANALYSIS/05-04-HEIM-SEM-FIGURES/*.png
- ANALYSIS/05-04-HEIM-SEM-FIGURES/*.json (metadata)
"""
```

#### 05-05-heim-sem-integrate.py
```python
"""
Merge semantic metrics with HEIM v6 biobank and clinical trial data

INTEGRATION:
1. Load bhem_metrics.json (Discovery dimension)
2. Load heim_ct_metrics.json (Translation dimension)
3. Load heim_sem_metrics.json (Knowledge dimension)
4. Merge on GBD disease taxonomy
5. Compute unified equity scores

UNIFIED SCORE:
- Weighted combination of all three dimensions
- Weights configurable (default: equal)
- Sensitivity analysis with different weights

OUTPUTS:
- Integrated metrics table
- Manuscript-ready supplementary tables
- Validation report
"""
```

---

## PART D: EXECUTION PLAN

### D.1 Full Pipeline Command

```bash
# Navigate to project directory
cd "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"

# Run with sleep prevention and logging
caffeinate -i -s -d bash -c '
    set -e  # Exit on any error

    echo "=== HEIM SEMANTIC PIPELINE ===" | tee pipeline.log
    echo "Started: $(date)" | tee -a pipeline.log

    # Phase 0: Setup
    echo "Phase 0: Setup validation..." | tee -a pipeline.log
    python PYTHON/05-00-heim-sem-setup.py 2>&1 | tee -a pipeline.log

    # Phase 1: Fetch
    echo "Phase 1: PubMed retrieval..." | tee -a pipeline.log
    python PYTHON/05-01-heim-sem-fetch.py 2>&1 | tee -a pipeline.log

    # Phase 2: Embed
    echo "Phase 2: Embedding generation..." | tee -a pipeline.log
    python PYTHON/05-02-heim-sem-embed.py 2>&1 | tee -a pipeline.log

    # Phase 3: Metrics
    echo "Phase 3: Metric computation..." | tee -a pipeline.log
    python PYTHON/05-03-heim-sem-compute-metrics.py 2>&1 | tee -a pipeline.log

    # Phase 4: Figures
    echo "Phase 4: Figure generation..." | tee -a pipeline.log
    python PYTHON/05-04-heim-sem-generate-figures.py 2>&1 | tee -a pipeline.log

    # Phase 5: Integrate
    echo "Phase 5: Integration..." | tee -a pipeline.log
    python PYTHON/05-05-heim-sem-integrate.py 2>&1 | tee -a pipeline.log

    echo "=== PIPELINE COMPLETE ===" | tee -a pipeline.log
    echo "Finished: $(date)" | tee -a pipeline.log
'
```

### D.2 Individual Script Execution (for debugging)

```bash
# Run single script with sleep prevention
caffeinate -i -s -d python PYTHON/05-01-heim-sem-fetch.py

# Resume from checkpoint
caffeinate -i -s -d python PYTHON/05-01-heim-sem-fetch.py --resume

# Force restart (ignore checkpoint)
caffeinate -i -s -d python PYTHON/05-01-heim-sem-fetch.py --fresh
```

### D.3 Monitoring Progress

```bash
# Watch log file in real-time
tail -f LOGS/05-02-embed-*.log

# Check checkpoint status
cat DATA/05-SEMANTIC/CHECKPOINTS/checkpoint_embed.json | python -m json.tool

# Count completed embeddings
ls -la DATA/05-SEMANTIC/EMBEDDINGS/ | wc -l
```

### D.4 Estimated Timeline

| Phase | Script | Duration | Checkpoint Interval |
|-------|--------|----------|---------------------|
| 0 | 05-00-setup | ~2 min | N/A |
| 1 | 05-01-fetch | 2-4 hours | Per disease-year |
| 2 | 05-02-embed | 8-12 hours | Per disease |
| 3 | 05-03-metrics | ~1 hour | Per metric type |
| 4 | 05-04-figures | ~30 min | Per figure |
| 5 | 05-05-integrate | ~10 min | N/A |
| **Total** | | **12-18 hours** | |

---

## PART E: TESTING & VALIDATION

### E.1 Unit Tests (before full run)

```bash
# Run test suite
python -m pytest PYTHON/tests/test_05_semantic.py -v

# Test coverage:
# - test_pubmed_parsing: Validates JSON schema
# - test_embedding_shape: Checks 768-dim output
# - test_mps_consistency: MPS vs CPU comparison
# - test_checkpoint_recovery: Resume logic
# - test_metric_bounds: No Inf/NaN values
```

### E.2 Smoke Test (small sample)

```bash
# Run on 3 diseases only for validation
python PYTHON/05-01-heim-sem-fetch.py --diseases "Malaria,Tuberculosis,Dengue"
python PYTHON/05-02-heim-sem-embed.py --diseases "Malaria,Tuberculosis,Dengue"
python PYTHON/05-03-heim-sem-compute-metrics.py
```

### E.3 Validation Checkpoints

After each phase, verify outputs before proceeding:

```bash
# After Phase 1 (Fetch)
python -c "
import json
with open('DATA/05-SEMANTIC/heim_sem_quality_scores.csv') as f:
    lines = len(f.readlines())
print(f'Diseases retrieved: {lines - 1}')
assert lines > 50, 'Too few diseases retrieved'
"

# After Phase 2 (Embed)
python -c "
import h5py
import os
embed_dir = 'DATA/05-SEMANTIC/EMBEDDINGS'
diseases = os.listdir(embed_dir)
for d in diseases[:3]:
    with h5py.File(f'{embed_dir}/{d}/embeddings.h5', 'r') as f:
        shape = f['embeddings'].shape
        print(f'{d}: {shape}')
        assert shape[1] == 768, 'Wrong embedding dimension'
"
```

---

## PART F: KNOWN ISSUES & WORKAROUNDS

### F.1 MPS Memory Pressure

**Symptom**: Process killed without error message
**Solution**: Reduce batch size, add memory monitoring

```python
# In 05-02-heim-sem-embed.py
import torch
if torch.mps.current_allocated_memory() / 1e9 > 200:  # >200GB
    gc.collect()
    torch.mps.empty_cache()
```

### F.2 NCBI Rate Limiting

**Symptom**: HTTP 429 or "Too Many Requests"
**Solution**: Built into script - exponential backoff

```python
# Retry logic (already implemented)
for attempt in range(5):
    try:
        result = Entrez.efetch(...)
        break
    except HTTPError as e:
        if e.code == 429:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
```

### F.3 HDF5 Corruption

**Symptom**: "Unable to open file" on resume
**Solution**: Checksum validation, rebuild from raw data

```bash
# Verify integrity
sha256sum -c DATA/05-SEMANTIC/EMBEDDINGS/*/embeddings.h5.sha256

# Rebuild single disease
python PYTHON/05-02-heim-sem-embed.py --disease "Malaria" --force
```

---

## PART G: DEPRECATION & MIGRATION

### G.1 18-SEMANTIC-MAPPING Deprecation

The following project is superseded:
- **`/PUBLICATIONS/18-SEMANTIC-MAPPING/`**

Existing assets to preserve:
- Quality scores (for reference only - will regenerate)
- 3 embedded diseases (Malaria, Schistosomiasis, Diarrheal) - validation comparison

### G.2 Future Data Reorganization

Post-pipeline, consider migrating all DATA/ to nested structure:
- `03-BIOBANK/` for bhem_* files
- `04-CLINICAL-TRIALS/` for heim_ct_* files
- `05-SEMANTIC/` already nested (new pattern)

This is a **separate task** requiring script updates and is not blocking for the semantic integration.

---

## PART H: APPROVAL CHECKLIST

- [ ] Architecture review complete (Part A)
- [ ] Guardrails acceptable (Part B)
- [ ] Directory structure approved (Part C.1)
- [ ] Script specifications approved (Part C.2)
- [ ] Execution plan acceptable (Part D)
- [ ] Testing strategy sufficient (Part E)
- [ ] Known issues documented (Part F)
- [ ] Ready to implement 05-00-heim-sem-setup.py

**Upon approval, I will implement scripts sequentially, starting with 05-00-heim-sem-setup.py.**
