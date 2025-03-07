#!/bin/bash
# model_retraining.sh
# Orchestrates the bi-daily model retraining process for the ERCOT RTLMP spike prediction system.
# This script is designed to be run as a scheduled job (e.g., via cron).
# Version: 1.0.0
# bash version: 4.0+

# Exit on error
set -e

# Global variables
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
PYTHON_ENV=${PYTHON_ENV:-"$PROJECT_ROOT/.venv/bin/python"}
CONFIG_PATH=${CONFIG_PATH:-"$PROJECT_ROOT/config/training_config.yaml"}
LOG_DIR=${LOG_DIR:-"$PROJECT_ROOT/logs"}
LOG_FILE="$LOG_DIR/model_retraining_$(date +%Y-%m-%d).log"
LOCK_FILE="$LOG_DIR/model_retraining.lock"
MAX_RETRIES=3
RETRY_DELAY=600  # 10 minutes in seconds
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-""}
THRESHOLD_VALUES="100 200 300"
NODE_IDS="HB_NORTH HB_SOUTH HB_WEST HB_HOUSTON"
TRAINING_WINDOW_DAYS=730  # 2 years
MODEL_OUTPUT_PATH="$PROJECT_ROOT/models"
RETRAINING_INTERVAL_DAYS=2
LAST_TRAINING_FILE="$LOG_DIR/last_training_date.txt"

# Function to set up the environment
setup_environment() {
    # Create log directory if it doesn't exist
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create log directory: $LOG_DIR"
            return 1
        fi
    fi
    
    # Create model output directory if it doesn't exist
    if [ ! -d "$MODEL_OUTPUT_PATH" ]; then
        mkdir -p "$MODEL_OUTPUT_PATH"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create model output directory: $MODEL_OUTPUT_PATH"
            return 1
        fi
    fi
    
    # Check if Python environment exists and is executable
    if [ ! -x "$PYTHON_ENV" ]; then
        echo "ERROR: Python environment not found or not executable: $PYTHON_ENV"
        return 1
    fi
    
    # Check if configuration file exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Configuration file not found: $CONFIG_PATH"
        return 1
    fi
    
    return 0
}

# Function to log a message
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local log_entry="[$timestamp] [$level] $message"
    
    echo "$log_entry"
    echo "$log_entry" >> "$LOG_FILE"
}

# Function to send a notification email
send_notification() {
    local subject="$1"
    local message="$2"
    
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        echo "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        local status=$?
        log_message "INFO" "Notification email sent to $NOTIFICATION_EMAIL (status: $status)"
        return $status
    else
        log_message "INFO" "No notification email configured, skipping notification"
        return 0
    fi
}

# Function to acquire a lock to prevent concurrent execution
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_message "WARNING" "Another instance of model retraining is already running (PID: $pid)"
            return 1
        else
            log_message "WARNING" "Removing stale lock file from PID $pid"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log_message "INFO" "Lock acquired (PID: $$)"
    return 0
}

# Function to release the lock
release_lock() {
    if [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE")" -eq $$ ]; then
        rm -f "$LOCK_FILE"
        log_message "INFO" "Lock released"
        return 0
    else
        log_message "WARNING" "Lock file does not exist or does not belong to this process"
        return 1
    fi
}

# Function to check if retraining is due
check_retraining_due() {
    if [ ! -f "$LAST_TRAINING_FILE" ]; then
        # If the file doesn't exist, create it with a date far in the past to force retraining
        log_message "INFO" "Last training date file not found, creating with initial date"
        echo "2000-01-01" > "$LAST_TRAINING_FILE"
        return 0
    fi
    
    local last_date=$(cat "$LAST_TRAINING_FILE")
    local last_timestamp=$(date -d "$last_date" +%s)
    local current_timestamp=$(date +%s)
    local days_diff=$(( (current_timestamp - last_timestamp) / 86400 ))
    
    log_message "INFO" "Days since last training: $days_diff (interval: $RETRAINING_INTERVAL_DAYS)"
    
    if [ "$days_diff" -ge "$RETRAINING_INTERVAL_DAYS" ]; then
        log_message "INFO" "Retraining is due"
        return 0
    else
        log_message "INFO" "Retraining is not due yet"
        return 1
    fi
}

# Function to update the last training date
update_last_training_date() {
    local current_date=$(date +%Y-%m-%d)
    echo "$current_date" > "$LAST_TRAINING_FILE"
    local status=$?
    
    if [ $status -eq 0 ]; then
        log_message "INFO" "Updated last training date to $current_date"
    else
        log_message "ERROR" "Failed to update last training date"
    fi
    
    return $status
}

# Function to fetch historical data for training
fetch_historical_data() {
    local end_date=$(date +%Y-%m-%d)
    local start_date=$(date -d "$end_date - $TRAINING_WINDOW_DAYS days" +%Y-%m-%d)
    
    log_message "INFO" "Fetching historical data from $start_date to $end_date"
    
    # Fetch ERCOT data
    log_message "INFO" "Fetching ERCOT RTLMP data..."
    local fetch_cmd="$PYTHON_ENV -m rtlmp_predict fetch-data --config $CONFIG_PATH --start-date $start_date --end-date $end_date --data-type rtlmp"
    local output=$(eval "$fetch_cmd" 2>&1)
    local status=$?
    
    if [ $status -eq 0 ]; then
        log_message "INFO" "Successfully fetched ERCOT RTLMP data"
    else
        log_message "ERROR" "Failed to fetch ERCOT RTLMP data: $output"
        return 1
    fi
    
    # Fetch weather data
    log_message "INFO" "Fetching weather data..."
    fetch_cmd="$PYTHON_ENV -m rtlmp_predict fetch-data --config $CONFIG_PATH --start-date $start_date --end-date $end_date --data-type weather"
    output=$(eval "$fetch_cmd" 2>&1)
    status=$?
    
    if [ $status -eq 0 ]; then
        log_message "INFO" "Successfully fetched weather data"
    else
        log_message "ERROR" "Failed to fetch weather data: $output"
        return 1
    fi
    
    log_message "INFO" "All historical data fetched successfully"
    return 0
}

# Function to run the model training process
run_training() {
    local threshold="$1"
    local node="$2"  # This should be a single node, not a space-separated list
    
    log_message "INFO" "Starting model training for threshold $threshold on node $node"
    
    local end_date=$(date +%Y-%m-%d)
    local start_date=$(date -d "$end_date - $TRAINING_WINDOW_DAYS days" +%Y-%m-%d)
    local output_path="$MODEL_OUTPUT_PATH/model_${threshold}_${node}_$(date +%Y%m%d).joblib"
    
    # Construct and execute the training command
    local train_cmd="$PYTHON_ENV -m rtlmp_predict train --config $CONFIG_PATH --start-date $start_date --end-date $end_date --threshold $threshold --node $node --output $output_path"
    
    log_message "INFO" "Executing: $train_cmd"
    local output=$(eval "$train_cmd" 2>&1)
    local status=$?
    
    if [ $status -eq 0 ]; then
        log_message "INFO" "Training completed successfully for threshold $threshold on node $node"
        log_message "DEBUG" "Training output: $output"
    else
        log_message "ERROR" "Training failed for threshold $threshold on node $node"
        log_message "ERROR" "Command output: $output"
    fi
    
    return $status
}

# Function to run training with retry logic
run_training_with_retry() {
    local threshold="$1"
    local node="$2"
    
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        run_training "$threshold" "$node"
        local status=$?
        
        if [ $status -eq 0 ]; then
            log_message "INFO" "Training succeeded for threshold $threshold on node $node"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log_message "WARNING" "Retry $retry_count/$MAX_RETRIES for threshold $threshold on node $node in $RETRY_DELAY seconds"
                sleep $RETRY_DELAY
            else
                log_message "ERROR" "All retries failed for threshold $threshold on node $node"
            fi
        fi
    done
    
    return 1
}

# Function to validate the trained model
validate_model() {
    local model_path="$1"
    
    if [ ! -f "$model_path" ]; then
        log_message "ERROR" "Model file not found: $model_path"
        return 1
    fi
    
    log_message "INFO" "Validating model: $model_path"
    
    # Run model evaluation
    local eval_cmd="$PYTHON_ENV -m rtlmp_predict evaluate --config $CONFIG_PATH --model $model_path"
    local output=$(eval "$eval_cmd" 2>&1)
    local status=$?
    
    if [ $status -ne 0 ]; then
        log_message "ERROR" "Model evaluation failed: $output"
        return 1
    fi
    
    # Extract and check metrics (assuming output has format "metric_name: value")
    local auc=$(echo "$output" | grep -oP 'AUC-ROC: \K[0-9.]+')
    local brier=$(echo "$output" | grep -oP 'Brier Score: \K[0-9.]+')
    
    log_message "INFO" "Model metrics - AUC-ROC: $auc, Brier Score: $brier"
    
    # Check if metrics meet quality thresholds
    # Try to use bc if available, otherwise fall back to awk for floating point comparison
    if command -v bc > /dev/null; then
        if (( $(echo "$auc < 0.70" | bc -l) )); then
            log_message "ERROR" "Model AUC-ROC score ($auc) below threshold (0.70)"
            return 1
        fi
        
        if (( $(echo "$brier > 0.20" | bc -l) )); then
            log_message "ERROR" "Model Brier score ($brier) above threshold (0.20)"
            return 1
        fi
    else
        # Fallback to awk
        if awk "BEGIN {exit !($auc < 0.70)}"; then
            log_message "ERROR" "Model AUC-ROC score ($auc) below threshold (0.70)"
            return 1
        fi
        
        if awk "BEGIN {exit !($brier > 0.20)}"; then
            log_message "ERROR" "Model Brier score ($brier) above threshold (0.20)"
            return 1
        fi
    fi
    
    log_message "INFO" "Model validation passed"
    return 0
}

# Cleanup function
cleanup() {
    log_message "INFO" "Running cleanup"
    release_lock
    log_message "INFO" "Cleanup completed"
}

# Main function
main() {
    # Set up trap for signals to ensure cleanup
    trap cleanup EXIT INT TERM
    
    log_message "INFO" "==============================================================="
    log_message "INFO" "ERCOT RTLMP Spike Prediction Model Retraining - Starting"
    log_message "INFO" "==============================================================="
    
    # Set up environment
    setup_environment
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Environment setup failed, aborting"
        return 1
    fi
    
    # Check if retraining is due
    check_retraining_due
    if [ $? -ne 0 ]; then
        log_message "INFO" "Retraining not due yet, exiting"
        return 0
    fi
    
    # Acquire lock
    acquire_lock
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to acquire lock, aborting"
        return 1
    fi
    
    # Fetch historical data
    fetch_historical_data
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to fetch historical data, aborting"
        return 1
    fi
    
    # Run training for each threshold and node combination
    local success_count=0
    local failure_count=0
    local failed_combinations=""
    
    for threshold in $THRESHOLD_VALUES; do
        for node in $NODE_IDS; do
            log_message "INFO" "Processing threshold $threshold for node $node"
            
            run_training_with_retry "$threshold" "$node"
            if [ $? -eq 0 ]; then
                success_count=$((success_count + 1))
                log_message "INFO" "Successfully trained model for threshold $threshold on node $node"
                
                # Validate the model
                local model_path="$MODEL_OUTPUT_PATH/model_${threshold}_${node}_$(date +%Y%m%d).joblib"
                validate_model "$model_path"
                if [ $? -ne 0 ]; then
                    failure_count=$((failure_count + 1))
                    failed_combinations="$failed_combinations
- Threshold $threshold, Node $node (validation failed)"
                    log_message "ERROR" "Model validation failed for threshold $threshold on node $node"
                fi
            else
                failure_count=$((failure_count + 1))
                failed_combinations="$failed_combinations
- Threshold $threshold, Node $node (training failed)"
                log_message "ERROR" "Failed to train model for threshold $threshold on node $node"
            fi
        done
    done
    
    # Update last training date
    update_last_training_date
    
    # Release lock
    release_lock
    
    # Log completion
    log_message "INFO" "==============================================================="
    log_message "INFO" "Model retraining completed"
    log_message "INFO" "Successful models: $success_count"
    log_message "INFO" "Failed models: $failure_count"
    log_message "INFO" "==============================================================="
    
    # Send notification if there were failures
    if [ $failure_count -gt 0 ]; then
        local subject="[ALERT] ERCOT RTLMP Model Retraining - $failure_count Failures"
        local message="Model retraining completed with $failure_count failures and $success_count successes.

Failed combinations:$failed_combinations

Please check the logs at $LOG_FILE for details."
        send_notification "$subject" "$message"
    fi
    
    # Return success if all training runs were successful, error otherwise
    if [ $failure_count -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Execute main function
main
exit $?