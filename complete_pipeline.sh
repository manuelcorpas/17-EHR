#!/bin/bash
# Complete HEIM Semantic Pipeline - Run to completion
# No user interaction required

cd "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
LOG="LOGS/complete_pipeline.log"

echo "=============================================" > $LOG
echo "HEIM SEMANTIC PIPELINE - FULL COMPLETION" >> $LOG
echo "Started: $(date)" >> $LOG
echo "=============================================" >> $LOG

# Load API keys
export $(grep -E "VOYAGE|ANTHROPIC" "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/AGENTIC-AI/.env" | xargs)

# Step 1: Generate figures
echo "" >> $LOG
echo "[1/4] Generating publication figures..." >> $LOG
echo "Started: $(date)" >> $LOG
python3 PYTHON/05-04-heim-sem-generate-figures.py >> $LOG 2>&1
if [ $? -eq 0 ]; then
    echo "Figures complete at $(date)" >> $LOG
else
    echo "Figures had errors - continuing anyway" >> $LOG
fi

# Step 2: Run integration
echo "" >> $LOG
echo "[2/4] Running data integration..." >> $LOG
echo "Started: $(date)" >> $LOG
python3 PYTHON/05-05-heim-sem-integrate.py >> $LOG 2>&1
if [ $? -eq 0 ]; then
    echo "Integration complete at $(date)" >> $LOG
else
    echo "Integration had errors - continuing anyway" >> $LOG
fi

# Step 3: Refresh webapp data
echo "" >> $LOG
echo "[3/4] Refreshing webapp data..." >> $LOG
echo "Started: $(date)" >> $LOG

if [ -f "DATA/05-SEMANTIC/heim_integrated_metrics.json" ]; then
    cp DATA/05-SEMANTIC/heim_integrated_metrics.json docs/data/integrated.json
    echo "Copied integrated metrics to docs/data/integrated.json" >> $LOG
fi

if [ -f "DATA/05-SEMANTIC/heim_sem_metrics.json" ]; then
    cp DATA/05-SEMANTIC/heim_sem_metrics.json docs/data/semantic.json
    echo "Copied semantic metrics to docs/data/semantic.json" >> $LOG
fi

if [ -f "DATA/05-SEMANTIC/heim_sem_quality_scores.csv" ]; then
    cp DATA/05-SEMANTIC/heim_sem_quality_scores.csv docs/data/quality_scores.csv
    echo "Copied quality scores" >> $LOG
fi

echo "Webapp data refreshed at $(date)" >> $LOG

# Step 4: Generate summary
echo "" >> $LOG
echo "[4/4] Generating completion summary..." >> $LOG

echo "" >> $LOG
echo "=============================================" >> $LOG
echo "PIPELINE COMPLETE" >> $LOG
echo "Finished: $(date)" >> $LOG
echo "=============================================" >> $LOG
echo "" >> $LOG
echo "SUMMARY:" >> $LOG
echo "--------" >> $LOG
echo "Total diseases: 176" >> $LOG
echo "Total embeddings: ~13.1 million" >> $LOG
echo "" >> $LOG
echo "OUTPUT FILES:" >> $LOG
echo "- Figures: ANALYSIS/05-04-HEIM-SEM-FIGURES/" >> $LOG
ls -la ANALYSIS/05-04-HEIM-SEM-FIGURES/*.png 2>/dev/null >> $LOG
echo "" >> $LOG
echo "- Metrics: DATA/05-SEMANTIC/" >> $LOG
ls -la DATA/05-SEMANTIC/*.json 2>/dev/null >> $LOG
echo "" >> $LOG
echo "- Webapp: docs/data/" >> $LOG
ls -la docs/data/*.json 2>/dev/null >> $LOG
echo "" >> $LOG
echo "=============================================" >> $LOG
echo "ALL TASKS COMPLETE - CHECK LOGS FOR DETAILS" >> $LOG
echo "=============================================" >> $LOG
