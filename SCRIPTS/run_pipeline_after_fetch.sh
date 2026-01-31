#!/bin/bash
# Automatically run embedding and metrics after fetch completes

FETCH_PID=38304
SCRIPT_DIR="/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
LOG_FILE="$SCRIPT_DIR/LOGS/pipeline_completion.log"

echo "============================================================" >> "$LOG_FILE"
echo "Pipeline automation started: $(date)" >> "$LOG_FILE"
echo "Waiting for fetch process (PID $FETCH_PID) to complete..." >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

# Wait for fetch to complete
while kill -0 $FETCH_PID 2>/dev/null; do
    sleep 60
done

echo "" >> "$LOG_FILE"
echo "Fetch completed at: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

cd "$SCRIPT_DIR"

# Step 1: Run embedding script
echo "============================================================" >> "$LOG_FILE"
echo "STEP 1: Running embedding script" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

python3 PYTHON/05-02-heim-sem-embed.py 2>&1 | tee -a "$LOG_FILE"

echo "Embedding completed: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Step 2: Run metrics script
echo "============================================================" >> "$LOG_FILE"
echo "STEP 2: Running metrics script" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

python3 PYTHON/05-03-heim-sem-metrics.py 2>&1 | tee -a "$LOG_FILE"

echo "Metrics completed: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "============================================================" >> "$LOG_FILE"
echo "PIPELINE COMPLETE: $(date)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

# Optional: Send notification (macOS)
osascript -e 'display notification "HEIM pipeline complete" with title "Data Processing"' 2>/dev/null || true
