#!/bin/bash
# Wait for critical batch to complete, then run remaining batch

CRITICAL_PID=14019
SCRIPT_DIR="/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
LOG_FILE="$SCRIPT_DIR/LOGS/overnight_fetch_remaining.log"

echo "Waiting for critical batch (PID $CRITICAL_PID) to complete..."
echo "Started waiting: $(date)"

# Wait for the critical batch to finish
while kill -0 $CRITICAL_PID 2>/dev/null; do
    sleep 60  # Check every minute
done

echo "Critical batch completed at $(date)"
echo "Starting remaining batch of 33 diseases..."

# Now run the remaining batch
cd "$SCRIPT_DIR"
nohup bash SCRIPTS/run_remaining_batch.sh > /dev/null 2>&1 &
REMAINING_PID=$!

echo "Remaining batch started with PID: $REMAINING_PID"
echo "Logs: $LOG_FILE"
