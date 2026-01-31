#!/bin/bash
# Post-embedding pipeline: validation, figures, webapp refresh
# Runs after iNTS embedding (PID 66973) completes

cd "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
LOG="LOGS/post_embedding_pipeline.log"

echo "=============================================" > $LOG
echo "POST-EMBEDDING PIPELINE" >> $LOG
echo "Started: $(date)" >> $LOG
echo "=============================================" >> $LOG

# Wait for iNTS embedding script to complete
echo "" >> $LOG
echo "[1/5] Waiting for iNTS embedding (PID 66973) to complete..." >> $LOG
while kill -0 66973 2>/dev/null; do sleep 30; done
echo "iNTS embedding complete at $(date)" >> $LOG

# Load API keys
export $(grep -E "VOYAGE|ANTHROPIC" "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/AGENTIC-AI/.env" | xargs)

# Step 2: Run validation/audit report for ALL diseases
echo "" >> $LOG
echo "[2/5] Running validation audit for all 176 diseases..." >> $LOG
echo "Started: $(date)" >> $LOG
python3 PYTHON/05-05-heim-sem-audit.py >> $LOG 2>&1
if [ $? -eq 0 ]; then
    echo "Validation audit complete at $(date)" >> $LOG
else
    echo "WARNING: Validation audit had errors at $(date)" >> $LOG
fi

# Step 3: Generate publication figures
echo "" >> $LOG
echo "[3/5] Generating publication figures..." >> $LOG
echo "Started: $(date)" >> $LOG
python3 PYTHON/05-04-heim-sem-generate-figures.py >> $LOG 2>&1
if [ $? -eq 0 ]; then
    echo "Figures generated at $(date)" >> $LOG
    echo "Output: ANALYSIS/05-04-HEIM-SEM-FIGURES/" >> $LOG
else
    echo "WARNING: Figure generation had errors at $(date)" >> $LOG
fi

# Step 4: Run integration (combines biobank + clinical trials + semantic)
echo "" >> $LOG
echo "[4/5] Running data integration..." >> $LOG
echo "Started: $(date)" >> $LOG
python3 PYTHON/05-05-heim-sem-integrate.py >> $LOG 2>&1
if [ $? -eq 0 ]; then
    echo "Integration complete at $(date)" >> $LOG
else
    echo "WARNING: Integration had errors at $(date)" >> $LOG
fi

# Step 5: Refresh webapp data
echo "" >> $LOG
echo "[5/5] Refreshing webapp data..." >> $LOG
echo "Started: $(date)" >> $LOG

# Copy integrated metrics to webapp
if [ -f "DATA/05-SEMANTIC/heim_integrated_metrics.json" ]; then
    cp DATA/05-SEMANTIC/heim_integrated_metrics.json docs/data/integrated.json
    echo "Copied integrated metrics to docs/data/integrated.json" >> $LOG
fi

# Copy semantic metrics to webapp
if [ -f "DATA/05-SEMANTIC/heim_sem_metrics.json" ]; then
    cp DATA/05-SEMANTIC/heim_sem_metrics.json docs/data/semantic.json
    echo "Copied semantic metrics to docs/data/semantic.json" >> $LOG
fi

# Copy quality scores
if [ -f "DATA/05-SEMANTIC/heim_sem_quality_scores.csv" ]; then
    cp DATA/05-SEMANTIC/heim_sem_quality_scores.csv docs/data/quality_scores.csv
    echo "Copied quality scores to docs/data/quality_scores.csv" >> $LOG
fi

echo "Webapp data refreshed at $(date)" >> $LOG

# Summary
echo "" >> $LOG
echo "=============================================" >> $LOG
echo "PIPELINE COMPLETE" >> $LOG
echo "Finished: $(date)" >> $LOG
echo "=============================================" >> $LOG
echo "" >> $LOG
echo "OUTPUTS:" >> $LOG
echo "- Validation: DATA/05-SEMANTIC/AUDIT/" >> $LOG
echo "- Figures: ANALYSIS/05-04-HEIM-SEM-FIGURES/" >> $LOG
echo "- Webapp: docs/data/" >> $LOG
echo "" >> $LOG
echo "Check logs for any warnings or errors." >> $LOG
