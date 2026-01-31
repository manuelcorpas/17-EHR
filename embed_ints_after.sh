#!/bin/bash
# Auto-embed iNTS after main embedding and metrics complete

cd "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
LOG="LOGS/embed_ints.log"

echo "Waiting for main embedding (PID 63471) to complete..." > $LOG

# Wait for main embedding to complete
while kill -0 63471 2>/dev/null; do sleep 60; done
echo "Main embedding complete at $(date)" >> $LOG

# Wait for metrics to complete
echo "Waiting for metrics (PID 63526) to complete..." >> $LOG
while kill -0 63526 2>/dev/null; do sleep 30; done
echo "Metrics complete at $(date)" >> $LOG

# Load API keys
export $(grep -E "VOYAGE|ANTHROPIC" "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/AGENTIC-AI/.env" | xargs)

# Embed iNTS
echo "Starting iNTS embedding at $(date)" >> $LOG
python3 PYTHON/05-02-heim-sem-embed.py --diseases "Invasive Non-typhoidal Salmonella (iNTS)" >> $LOG 2>&1
echo "iNTS embedding complete at $(date)" >> $LOG

# Run metrics again to include iNTS
echo "Running final metrics at $(date)" >> $LOG
python3 PYTHON/05-03-heim-sem-compute-metrics.py >> $LOG 2>&1
echo "All 176 diseases complete at $(date)" >> $LOG
