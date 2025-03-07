#!/bin/bash
# daily_inference.sh
# This script orchestrates the daily inference process for the ERCOT RTLMP spike prediction system
# It ensures timely generation of 72-hour RTLMP spike probability forecasts before day-ahead market closure

# Set strict error handling
set -e

# Global variables
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
PYTHON_ENV=${PYTHON_ENV:-"$PROJECT_ROOT/.venv/bin/python"}
CONFIG_PATH=${CONFIG_PATH:-"$PROJECT_ROOT/config/inference_config.yaml"}
LOG_DIR=${LOG_DIR:-"$PROJECT_ROOT/logs"}
LOG_FILE="$LOG_DIR/daily_inference_$(date +%Y-%m-%d).log"
LOCK_FILE="$LOG_DIR/daily_inference.lock"
MAX_RETRIES=3
RETRY_DELAY=300  # 5 minutes
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-""}
THRESHOLD_VALUES="100 200 300"
NODE_IDS="HB_NORTH HB_SOUTH HB_WEST HB_HOUSTON"
DAM_CLOSURE_TIME="10:00"
FORECAST_OUTPUT_PATH="$PROJECT_ROOT/forecasts/$(date +%Y-%m-%d)"

# Function to set up the environment for the daily inference run
setup_environment() {
    # Create log directory if it doesn't exist
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        log_message "INFO" "Created log directory: $LOG_DIR"
    fi
    
    # Create forecast output directory if it doesn't exist
    if [ ! -d "$FORECAST_OUTPUT_PATH" ]; then
        mkdir -p "$FORECAST_OUTPUT_PATH"
        log_message "INFO" "Created forecast output directory: $FORECAST_OUTPUT_PATH"
    fi
    
    # Check if Python environment exists and is executable
    if [ ! -x "$PYTHON_ENV" ]; then
        log_message "ERROR" "Python environment not found or not executable: $PYTHON_ENV"
        return 1
    fi
    
    # Check if configuration file exists
    if [ ! -f "$CONFIG_PATH" ]; then
        log_message "ERROR" "Configuration file not found: $CONFIG_PATH"
        return 1
    fi
    
    # Validate that required environment variables are set
    if [ -z "$THRESHOLD_VALUES" ] || [ -z "$NODE_IDS" ]; then
        log_message "ERROR" "Required environment variables not set"
        return 1
    }
    
    log_message "INFO" "Environment setup completed successfully"
    return 0
}

# Function to logs a message to both stdout and the log file
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Function to sends a notification email if NOTIFICATION_EMAIL is set
send_notification() {
    local subject="$1"
    local message="$2"
    
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        echo -e "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        local exit_code=$?
        log_message "INFO" "Notification email sent to $NOTIFICATION_EMAIL (exit code: $exit_code)"
        return $exit_code
    else
        log_message "DEBUG" "No notification email address configured, skipping notification"
        return 0
    fi
}

# Function to acquires a lock file to prevent concurrent execution
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if ps -p "$pid" > /dev/null; then
            log_message "WARNING" "Another instance of the script is already running (PID: $pid)"
            return 1
        else
            log_message "WARNING" "Removing stale lock file from PID: $pid"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log_message "DEBUG" "Lock acquired (PID: $$)"
    return 0
}

# Function to releases the lock file
release_lock() {
    if [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE")" = "$$" ]; then
        rm -f "$LOCK_FILE"
        log_message "DEBUG" "Lock released"
        return 0
    else
        log_message "WARNING" "Could not release lock - file not found or owned by another process"
        return 1
    fi
}

# Function to check if there's enough time before DAM closure deadline
check_dam_closure_deadline() {
    local current_time=$(date +%H:%M)
    local deadline_hour=${DAM_CLOSURE_TIME%%:*}
    local deadline_minute=${DAM_CLOSURE_TIME##*:}
    local current_hour=${current_time%%:*}
    local current_minute=${current_time##*:}
    
    # Convert to minutes for easier comparison
    local deadline_minutes=$((deadline_hour * 60 + deadline_minute))
    local current_minutes=$((current_hour * 60 + current_minute))
    
    local time_remaining=$((deadline_minutes - current_minutes))
    
    # If we've crossed midnight, adjust the calculation
    if [ $time_remaining -lt 0 ]; then
        time_remaining=$((time_remaining + 24 * 60))
    fi
    
    log_message "INFO" "Time remaining until DAM closure: $time_remaining minutes"
    
    # Less than 1 hour remaining
    if [ $time_remaining -lt 60 ]; then
        log_message "WARNING" "Less than 1 hour remaining until DAM closure deadline ($DAM_CLOSURE_TIME)"
        # Deadline has passed
        if [ $time_remaining -le 0 ]; then
            log_message "CRITICAL" "DAM closure deadline ($DAM_CLOSURE_TIME) has already passed"
            return 1
        fi
        return 1
    fi
    
    return 0
}

# Function to fetch the latest data required for inference
fetch_latest_data() {
    log_message "INFO" "Starting data fetch process"
    
    # Fetch ERCOT data
    log_message "INFO" "Fetching ERCOT market data"
    local ercot_cmd="$PYTHON_ENV -m rtlmp_predict.cli fetch-data --config $CONFIG_PATH --source ercot --output-dir $PROJECT_ROOT/data/raw/$(date +%Y-%m-%d)"
    local ercot_output=$(eval "$ercot_cmd" 2>&1)
    local ercot_exit_code=$?
    
    if [ $ercot_exit_code -ne 0 ]; then
        log_message "ERROR" "Failed to fetch ERCOT data: $ercot_output"
        return 1
    else
        log_message "INFO" "Successfully fetched ERCOT data"
    fi
    
    # Fetch weather data
    log_message "INFO" "Fetching weather forecast data"
    local weather_cmd="$PYTHON_ENV -m rtlmp_predict.cli fetch-data --config $CONFIG_PATH --source weather --output-dir $PROJECT_ROOT/data/raw/$(date +%Y-%m-%d)"
    local weather_output=$(eval "$weather_cmd" 2>&1)
    local weather_exit_code=$?
    
    if [ $weather_exit_code -ne 0 ]; then
        log_message "ERROR" "Failed to fetch weather data: $weather_output"
        return 1
    else
        log_message "INFO" "Successfully fetched weather data"
    fi
    
    log_message "INFO" "Data fetch process completed successfully"
    return 0
}

# Function to run the inference process using the CLI command
run_inference() {
    local threshold="$1"
    local nodes="$2"
    
    log_message "INFO" "Starting inference for threshold=$threshold, nodes=$nodes"
    
    local output_file="$FORECAST_OUTPUT_PATH/forecast_${nodes}_${threshold}_$(date +%Y%m%d_%H%M%S).parquet"
    local cmd="$PYTHON_ENV -m rtlmp_predict.cli predict --config $CONFIG_PATH --threshold $threshold --node $nodes --output $output_file"
    
    log_message "DEBUG" "Executing command: $cmd"
    local output=$(eval "$cmd" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "INFO" "Inference completed successfully for threshold=$threshold, nodes=$nodes"
        log_message "DEBUG" "Command output: $output"
    else
        log_message "ERROR" "Inference failed for threshold=$threshold, nodes=$nodes with exit code $exit_code"
        log_message "ERROR" "Command output: $output"
    fi
    
    return $exit_code
}

# Function to run inference with retry logic for resilience
run_inference_with_retry() {
    local threshold="$1"
    local nodes="$2"
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        run_inference "$threshold" "$nodes"
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            return 0
        else
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log_message "WARNING" "Retrying inference for threshold=$threshold, nodes=$nodes (attempt $retry_count of $MAX_RETRIES) after $RETRY_DELAY seconds"
                sleep $RETRY_DELAY
            else
                log_message "ERROR" "All retry attempts failed for threshold=$threshold, nodes=$nodes"
                return 1
            fi
        fi
    done
    
    return 1
}

# Function to validate that the generated forecast meets quality requirements
validate_forecast() {
    local forecast_path="$1"
    
    if [ ! -f "$forecast_path" ]; then
        log_message "ERROR" "Forecast file not found: $forecast_path"
        return 1
    fi
    
    # Check forecast using the validation command
    local cmd="$PYTHON_ENV -m rtlmp_predict.cli validate --config $CONFIG_PATH --forecast $forecast_path"
    local output=$(eval "$cmd" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "INFO" "Forecast validation passed for: $forecast_path"
        log_message "DEBUG" "Validation output: $output"
        return 0
    else
        log_message "ERROR" "Forecast validation failed for: $forecast_path"
        log_message "ERROR" "Validation output: $output"
        return 1
    fi
}

# Cleanup function called on script exit or error
cleanup() {
    log_message "INFO" "Performing cleanup"
    release_lock
    log_message "INFO" "Cleanup completed"
}

# Main function that orchestrates the daily inference process
main() {
    # Set up error handling with trap for signals
    trap cleanup EXIT INT TERM
    
    log_message "INFO" "Starting daily inference process ($(date))"
    
    # Set up environment
    setup_environment
    if [ $? -ne 0 ]; then
        log_message "CRITICAL" "Environment setup failed, aborting"
        send_notification "ERCOT RTLMP Inference - Environment Setup Failed" "The daily inference process failed during environment setup. Please check the logs at $LOG_FILE."
        return 1
    fi
    
    # Check DAM closure deadline
    check_dam_closure_deadline
    if [ $? -ne 0 ]; then
        log_message "CRITICAL" "Too close to or past DAM closure deadline, aborting"
        send_notification "ERCOT RTLMP Inference - Deadline Warning" "The daily inference process was aborted because it is too close to or past the DAM closure deadline ($DAM_CLOSURE_TIME)."
        return 1
    fi
    
    # Acquire lock
    acquire_lock
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to acquire lock, another instance may be running"
        send_notification "ERCOT RTLMP Inference - Lock Acquisition Failed" "The daily inference process could not acquire the lock. Another instance may be running."
        return 1
    fi
    
    # Fetch latest data
    fetch_latest_data
    if [ $? -ne 0 ]; then
        log_message "CRITICAL" "Failed to fetch latest data, aborting"
        send_notification "ERCOT RTLMP Inference - Data Fetch Failed" "The daily inference process failed during data fetching. Please check the logs at $LOG_FILE."
        return 1
    fi
    
    # Run inference for each threshold and node
    local success_count=0
    local failure_count=0
    local failures=""
    
    for threshold in $THRESHOLD_VALUES; do
        for node in $NODE_IDS; do
            log_message "INFO" "Processing threshold=$threshold, node=$node"
            
            run_inference_with_retry "$threshold" "$node"
            if [ $? -eq 0 ]; then
                success_count=$((success_count + 1))
                
                # Validate the forecast
                local forecast_file="$FORECAST_OUTPUT_PATH/forecast_${node}_${threshold}_$(date +%Y%m%d)*.parquet"
                local latest_forecast=$(ls -t $forecast_file 2>/dev/null | head -1)
                
                if [ -n "$latest_forecast" ]; then
                    validate_forecast "$latest_forecast"
                    if [ $? -ne 0 ]; then
                        log_message "WARNING" "Forecast validation failed for threshold=$threshold, node=$node"
                        failures="$failures\n- Forecast validation failed for threshold=$threshold, node=$node"
                    fi
                else
                    log_message "WARNING" "No forecast file found for threshold=$threshold, node=$node"
                    failures="$failures\n- No forecast file found for threshold=$threshold, node=$node"
                fi
            else
                failure_count=$((failure_count + 1))
                failures="$failures\n- Inference failed for threshold=$threshold, node=$node"
            fi
        done
    done
    
    # Log completion
    local total_count=$((success_count + failure_count))
    log_message "INFO" "Daily inference process completed with $success_count successes and $failure_count failures out of $total_count total runs"
    
    # Send notification if there were failures
    if [ $failure_count -gt 0 ]; then
        local subject="ERCOT RTLMP Inference - Completed with $failure_count Failures"
        local message="The daily inference process completed with $success_count successes and $failure_count failures out of $total_count total runs.\n\nFailed operations:$failures\n\nPlease check the logs at $LOG_FILE for more details."
        send_notification "$subject" "$message"
        return 1
    else
        log_message "INFO" "All inference runs completed successfully"
        return 0
    fi
}

# Execute main function
main
exit $?