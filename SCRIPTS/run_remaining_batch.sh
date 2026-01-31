#!/bin/bash
# Run remaining fishy diseases batch after critical batch completes

SCRIPT_DIR="/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
LOG_FILE="$SCRIPT_DIR/LOGS/overnight_fetch_remaining.log"
DISEASES_FILE="$SCRIPT_DIR/DATA/05-SEMANTIC/remaining_fishy_diseases.txt"

echo "============================================================" >> "$LOG_FILE"
echo "Starting REMAINING BATCH: 33 fishy diseases" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

cd "$SCRIPT_DIR"

# Run the fetch script with remaining diseases
python3 PYTHON/05-01-heim-sem-fetch.py \
    --diseases-file "$DISEASES_FILE" \
    --no-caffeinate \
    2>&1 | tee -a "$LOG_FILE"

echo "" >> "$LOG_FILE"
echo "Completed: $(date)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
