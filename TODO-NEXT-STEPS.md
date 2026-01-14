# HEIM Semantic Analysis - Next Steps

## Summary of Tonight's Session (2025-01-14)

### What Was Done

1. **Webapp Updated to v3.0**
   - Three-dimensional framework (Discovery + Translation + Knowledge)
   - Knowledge tab with semantic analysis figures
   - Statistical validation added to methodology

2. **Data Quality Issue Identified**
   - 10 CRITICAL diseases have wrong paper counts (search term failures)
   - 17 WARNING diseases have very low papers (<100)
   - 119 diseases (82%) appear OK

3. **Audit Tools Created**
   - `PYTHON/05-05-heim-sem-audit.py` - Audit script
   - `DATA/05-SEMANTIC/gbd_mesh_mapping.json` - Correct search terms

4. **Data Quality Warning Added**
   - Yellow banner on Knowledge tab warns users
   - Live at: https://manuelcorpas.github.io/17-EHR/

### CRITICAL Diseases Needing Re-Collection

| Disease | Current Papers | Expected | Trials |
|---------|---------------|----------|--------|
| Lung cancer | 20 | 200,000 | 101,350 |
| Brain cancer | 30 | 80,000 | 100,437 |
| Alzheimer's | 470 | 150,000 | 10,805 |
| Colon cancer | 132 | 100,000 | 100,743 |
| Maternal disorders | 125 | ~50,000 | 39,534 |
| Neonatal disorders | 228 | ~30,000 | 11,118 |

### What Needs To Be Done

1. **Update fetch script** (`05-01-heim-sem-fetch.py`)
   - Modify `search_disease_year()` to use the MeSH mapping
   - Load queries from `gbd_mesh_mapping.json`

2. **Re-run data collection** (CRITICAL diseases only first)
   ```bash
   python 05-01-heim-sem-fetch.py --diseases "Tracheal, bronchus, and lung cancer,Brain and central nervous system cancer,Alzheimer's disease and other dementias,Colon and rectum cancer"
   ```

3. **Re-compute embeddings and metrics**
   ```bash
   python 05-02-heim-sem-embed.py
   python 05-03-heim-sem-metrics.py
   ```

4. **Update webapp data**
   - Copy new `heim_integrated_metrics.json` to `docs/data/`

### Nature Medicine Viability

**Current Status:** NOT READY

**Why:**
- Data quality issues undermine the semantic analysis
- The NTD finding (p=0.002) is based on diseases with VALID data
- Need to re-run with corrected queries to confirm

**After Fix:**
- If NTD isolation finding holds with complete data, it's publishable
- Translation gap (papers vs trials) is another strong angle
- Target: Lancet Global Health or PLOS Medicine more realistic than Nature Medicine

### Files Modified Tonight

- `docs/index.html` - Multiple updates
- `docs/css/style.css` - Figure layout
- `docs/js/app.js` - GS breakdown charts
- `docs/data/clinical_trials.json` - Road injuries fix
- `PYTHON/05-05-heim-sem-audit.py` - NEW
- `DATA/05-SEMANTIC/gbd_mesh_mapping.json` - NEW

### Live Webapp

https://manuelcorpas.github.io/17-EHR/
