#!/bin/bash
# restore_data.sh - Data restoration script for ERCOT RTLMP spike prediction system
# version: 1.0
#
# This script restores data from backups for the ERCOT RTLMP spike prediction system.
# It handles the restoration of raw data, features, models, forecasts, and logs with 
# validation and error handling.
#
# Usage: ./restore_data.sh [--date BACKUP_DATE] [--raw-data] [--features] [--models] [--forecasts] [--logs] [--all] [--help]

# Exit on error, but allow error handling functions to execute
set -e

# Define global variables
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BACKUP_DIR=${BACKUP_DIR:-"$PROJECT_ROOT/backups"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/data"}
FEATURES_DIR=${FEATURES_DIR:-"$PROJECT_ROOT/data/features"}
MODELS_DIR=${MODELS_DIR:-"$PROJECT_ROOT/models"}
FORECASTS_DIR=${FORECASTS_DIR:-"$PROJECT_ROOT/forecasts"}
LOGS_DIR=${LOGS_DIR:-"$PROJECT_ROOT/logs"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/restore_${TIMESTAMP}.log"
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-""}
LOCK_FILE="$LOGS_DIR/restore_data.lock"

# Default flags
RESTORE_RAW_DATA=false
RESTORE_FEATURES=false
RESTORE_MODELS=false
RESTORE_FORECASTS=false
RESTORE_LOGS=false
BACKUP_DATE=""

# Ensure cleanup happens on exit
trap cleanup EXIT

# Initialize a variable to track if we've acquired a lock
LOCK_ACQUIRED=false

# Function to log messages to stdout and log file
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local log_msg="[$timestamp] [$level] $message"
    
    echo "$log_msg"
    
    # Ensure log directory exists
    if [[ ! -d "$LOGS_DIR" ]]; then
        mkdir -p "$LOGS_DIR"
    fi
    
    echo "$log_msg" >> "$LOG_FILE"
}

# Function to check required directories
check_directories() {
    # Check if PROJECT_ROOT exists and is readable
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log_message "ERROR" "Project root directory '$PROJECT_ROOT' not found"
        return 1
    fi
    
    if [[ ! -r "$PROJECT_ROOT" ]]; then
        log_message "ERROR" "Project root directory '$PROJECT_ROOT' not readable"
        return 1
    fi
    
    # Check if BACKUP_DIR exists and is readable
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_message "ERROR" "Backup directory '$BACKUP_DIR' not found"
        return 1
    fi
    
    if [[ ! -r "$BACKUP_DIR" ]]; then
        log_message "ERROR" "Backup directory '$BACKUP_DIR' not readable"
        return 1
    fi
    
    # Create target directories if they don't exist
    for dir in "$DATA_DIR" "$FEATURES_DIR" "$MODELS_DIR" "$FORECASTS_DIR" "$LOGS_DIR"; do
        if [[ ! -d "$dir" ]]; then
            log_message "INFO" "Creating directory '$dir'"
            mkdir -p "$dir"
        fi
    done
    
    return 0
}

# Function to acquire a lock to prevent concurrent execution
acquire_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local pid=$(cat "$LOCK_FILE")
        
        # Check if process is still running
        if kill -0 "$pid" 2>/dev/null; then
            log_message "WARNING" "Another restore process (PID: $pid) is already running"
            return 1
        else
            log_message "WARNING" "Removing stale lock file from PID: $pid"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file with current PID
    echo $$ > "$LOCK_FILE"
    LOCK_ACQUIRED=true
    log_message "INFO" "Lock acquired (PID: $$)"
    return 0
}

# Function to release the lock
release_lock() {
    if [[ "$LOCK_ACQUIRED" == true && -f "$LOCK_FILE" ]]; then
        local pid=$(cat "$LOCK_FILE")
        
        if [[ "$pid" == "$$" ]]; then
            rm -f "$LOCK_FILE"
            log_message "INFO" "Lock released"
            LOCK_ACQUIRED=false
            return 0
        else
            log_message "WARNING" "Lock file exists but contains different PID: $pid (current: $$)"
            return 1
        fi
    fi
    
    return 0
}

# Function to list available backup dates
list_available_backups() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_message "ERROR" "Backup directory '$BACKUP_DIR' not found"
        return 1
    fi
    
    local backups=$(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -name "2*" | sort -r)
    
    if [[ -z "$backups" ]]; then
        log_message "ERROR" "No backups found in '$BACKUP_DIR'"
        return 1
    fi
    
    log_message "INFO" "Available backups:"
    local count=1
    local backup_list=()
    
    while IFS= read -r backup; do
        local date=$(basename "$backup")
        echo "  $count. $date"
        backup_list+=("$date")
        ((count++))
    done <<< "$backups"
    
    # Interactive selection if script is running in a terminal
    if [[ -t 0 && -z "$BACKUP_DATE" ]]; then
        echo -n "Select backup to restore (1-$((count-1))): "
        read -r selection
        
        if [[ "$selection" =~ ^[0-9]+$ && "$selection" -ge 1 && "$selection" -lt "$count" ]]; then
            BACKUP_DATE="${backup_list[$((selection-1))]}"
            log_message "INFO" "Selected backup date: $BACKUP_DATE"
            return 0
        else
            log_message "ERROR" "Invalid selection: $selection"
            return 1
        fi
    fi
    
    # Return the list of backup dates (newest first)
    echo "${backup_list[@]}"
    return 0
}

# Function to validate a backup
validate_backup() {
    local backup_date="$1"
    local backup_path="$BACKUP_DIR/$backup_date"
    
    if [[ ! -d "$backup_path" ]]; then
        log_message "ERROR" "Backup directory for date '$backup_date' not found"
        return 1
    fi
    
    # Check for essential backup files
    local essential_files=(
        "raw_data.tar.gz"
        "features.tar.gz"
        "models.tar.gz"
        "forecasts.tar.gz"
    )
    
    for file in "${essential_files[@]}"; do
        if [[ ! -f "$backup_path/$file" ]]; then
            log_message "WARNING" "Backup file '$file' not found in backup '$backup_date'"
        fi
    done
    
    # Test archive integrity for existing files
    for file in "$backup_path"/*.tar.gz; do
        if [[ -f "$file" ]]; then
            log_message "INFO" "Validating integrity of '$(basename "$file")'"
            if ! tar --test --file="$file" > /dev/null 2>&1; then
                log_message "ERROR" "Archive '$(basename "$file")' is corrupted"
                return 1
            fi
        fi
    done
    
    log_message "INFO" "Backup '$backup_date' validated successfully"
    return 0
}

# Function to restore raw data
restore_raw_data() {
    local backup_date="$1"
    local backup_file="$BACKUP_DIR/$backup_date/raw_data.tar.gz"
    
    log_message "INFO" "Starting raw data restoration from $backup_date"
    
    if [[ ! -f "$backup_file" ]]; then
        log_message "ERROR" "Raw data backup file not found: $backup_file"
        return 1
    fi
    
    # Ensure target directory exists
    if [[ ! -d "$DATA_DIR" ]]; then
        mkdir -p "$DATA_DIR"
    fi
    
    # Count files before extraction
    local files_before=$(find "$DATA_DIR" -type f | wc -l)
    
    # Extract data
    log_message "INFO" "Extracting raw data to $DATA_DIR"
    if ! tar -xzf "$backup_file" -C "$DATA_DIR"; then
        log_message "ERROR" "Failed to extract raw data backup"
        return 1
    fi
    
    # Count files after extraction
    local files_after=$(find "$DATA_DIR" -type f | wc -l)
    local files_added=$((files_after - files_before))
    
    log_message "INFO" "Raw data restoration complete. $files_added files restored."
    return 0
}

# Function to restore features
restore_features() {
    local backup_date="$1"
    local backup_file="$BACKUP_DIR/$backup_date/features.tar.gz"
    
    log_message "INFO" "Starting features restoration from $backup_date"
    
    if [[ ! -f "$backup_file" ]]; then
        log_message "ERROR" "Features backup file not found: $backup_file"
        return 1
    fi
    
    # Ensure target directory exists
    if [[ ! -d "$FEATURES_DIR" ]]; then
        mkdir -p "$FEATURES_DIR"
    fi
    
    # Count files before extraction
    local files_before=$(find "$FEATURES_DIR" -type f | wc -l)
    
    # Extract data
    log_message "INFO" "Extracting features to $FEATURES_DIR"
    if ! tar -xzf "$backup_file" -C "$FEATURES_DIR"; then
        log_message "ERROR" "Failed to extract features backup"
        return 1
    fi
    
    # Count files after extraction
    local files_after=$(find "$FEATURES_DIR" -type f | wc -l)
    local files_added=$((files_after - files_before))
    
    log_message "INFO" "Features restoration complete. $files_added files restored."
    return 0
}

# Function to restore models
restore_models() {
    local backup_date="$1"
    local backup_file="$BACKUP_DIR/$backup_date/models.tar.gz"
    
    log_message "INFO" "Starting models restoration from $backup_date"
    
    if [[ ! -f "$backup_file" ]]; then
        log_message "ERROR" "Models backup file not found: $backup_file"
        return 1
    fi
    
    # Ensure target directory exists
    if [[ ! -d "$MODELS_DIR" ]]; then
        mkdir -p "$MODELS_DIR"
    fi
    
    # Count files before extraction
    local files_before=$(find "$MODELS_DIR" -type f | wc -l)
    
    # Extract data
    log_message "INFO" "Extracting models to $MODELS_DIR"
    if ! tar -xzf "$backup_file" -C "$MODELS_DIR"; then
        log_message "ERROR" "Failed to extract models backup"
        return 1
    fi
    
    # Count files after extraction
    local files_after=$(find "$MODELS_DIR" -type f | wc -l)
    local files_added=$((files_after - files_before))
    
    log_message "INFO" "Models restoration complete. $files_added files restored."
    return 0
}

# Function to restore forecasts
restore_forecasts() {
    local backup_date="$1"
    local backup_file="$BACKUP_DIR/$backup_date/forecasts.tar.gz"
    
    log_message "INFO" "Starting forecasts restoration from $backup_date"
    
    if [[ ! -f "$backup_file" ]]; then
        log_message "ERROR" "Forecasts backup file not found: $backup_file"
        return 1
    fi
    
    # Ensure target directory exists
    if [[ ! -d "$FORECASTS_DIR" ]]; then
        mkdir -p "$FORECASTS_DIR"
    fi
    
    # Count files before extraction
    local files_before=$(find "$FORECASTS_DIR" -type f | wc -l)
    
    # Extract data
    log_message "INFO" "Extracting forecasts to $FORECASTS_DIR"
    if ! tar -xzf "$backup_file" -C "$FORECASTS_DIR"; then
        log_message "ERROR" "Failed to extract forecasts backup"
        return 1
    fi
    
    # Count files after extraction
    local files_after=$(find "$FORECASTS_DIR" -type f | wc -l)
    local files_added=$((files_after - files_before))
    
    log_message "INFO" "Forecasts restoration complete. $files_added files restored."
    return 0
}

# Function to restore logs
restore_logs() {
    local backup_date="$1"
    local backup_file="$BACKUP_DIR/$backup_date/logs.tar.gz"
    
    log_message "INFO" "Starting logs restoration from $backup_date"
    
    if [[ ! -f "$backup_file" ]]; then
        log_message "ERROR" "Logs backup file not found: $backup_file"
        return 1
    fi
    
    # Ensure target directory exists
    if [[ ! -d "$LOGS_DIR" ]]; then
        mkdir -p "$LOGS_DIR"
    fi
    
    # Count files before extraction
    local files_before=$(find "$LOGS_DIR" -type f | wc -l)
    
    # Extract data
    log_message "INFO" "Extracting logs to $LOGS_DIR"
    if ! tar -xzf "$backup_file" -C "$LOGS_DIR"; then
        log_message "ERROR" "Failed to extract logs backup"
        return 1
    fi
    
    # Count files after extraction
    local files_after=$(find "$LOGS_DIR" -type f | wc -l)
    local files_added=$((files_after - files_before))
    
    log_message "INFO" "Logs restoration complete. $files_added files restored."
    return 0
}

# Function to send a notification about restoration status
send_notification() {
    local status="$1"
    local message="$2"
    
    if [[ -z "$NOTIFICATION_EMAIL" ]]; then
        log_message "INFO" "No notification email set, skipping notification"
        return 0
    fi
    
    local subject="[ERCOT RTLMP] Data Restoration ${status}"
    local body="Data Restoration ${status}\n\nTimestamp: $(date)\n\n${message}\n\nBackup Date: ${BACKUP_DATE}\n"
    
    if command -v mail > /dev/null 2>&1; then
        echo -e "$body" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        log_message "INFO" "Notification sent to $NOTIFICATION_EMAIL"
        return 0
    else
        log_message "WARNING" "mail command not available, notification not sent"
        return 1
    fi
}

# Function to handle errors
handle_error() {
    local error_message="$1"
    local error_code="${2:-1}"
    
    log_message "ERROR" "$error_message"
    send_notification "FAILED" "$error_message"
    
    # Release lock if acquired
    if [[ "$LOCK_ACQUIRED" == true ]]; then
        release_lock
    fi
    
    return "$error_code"
}

# Cleanup function called on script exit or error
cleanup() {
    # Release lock if acquired
    if [[ "$LOCK_ACQUIRED" == true ]]; then
        release_lock
    fi
    
    log_message "INFO" "Cleanup completed"
}

# Function to display usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --date BACKUP_DATE   Specify the backup date to restore from (format: YYYYMMDD)
  --raw-data           Restore only raw data
  --features           Restore only engineered features
  --models             Restore only trained models
  --forecasts          Restore only forecasts
  --logs               Restore only logs
  --all                Restore all components (default if no specific component is selected)
  --help               Display this help message

Environment variables:
  PROJECT_ROOT         Root directory of the project
  BACKUP_DIR           Directory where backups are stored
  DATA_DIR             Directory for raw data
  FEATURES_DIR         Directory for engineered features
  MODELS_DIR           Directory for trained models
  FORECASTS_DIR        Directory for generated forecasts
  LOGS_DIR             Directory for application logs
  NOTIFICATION_EMAIL   Email address for notifications
EOF
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --date)
                BACKUP_DATE="$2"
                shift 2
                ;;
            --raw-data)
                RESTORE_RAW_DATA=true
                shift
                ;;
            --features)
                RESTORE_FEATURES=true
                shift
                ;;
            --models)
                RESTORE_MODELS=true
                shift
                ;;
            --forecasts)
                RESTORE_FORECASTS=true
                shift
                ;;
            --logs)
                RESTORE_LOGS=true
                shift
                ;;
            --all)
                RESTORE_RAW_DATA=true
                RESTORE_FEATURES=true
                RESTORE_MODELS=true
                RESTORE_FORECASTS=true
                RESTORE_LOGS=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_message "ERROR" "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # If no component is specified, restore all
    if ! $RESTORE_RAW_DATA && ! $RESTORE_FEATURES && ! $RESTORE_MODELS && ! $RESTORE_FORECASTS && ! $RESTORE_LOGS; then
        RESTORE_RAW_DATA=true
        RESTORE_FEATURES=true
        RESTORE_MODELS=true
        RESTORE_FORECASTS=true
        RESTORE_LOGS=true
    fi
    
    # Initialize logging
    log_message "INFO" "Starting data restoration process"
    log_message "INFO" "Project root: $PROJECT_ROOT"
    log_message "INFO" "Backup directory: $BACKUP_DIR"
    
    # Check directories
    if ! check_directories; then
        handle_error "Failed to verify required directories" 1
        exit 1
    fi
    
    # Acquire lock
    if ! acquire_lock; then
        handle_error "Failed to acquire lock, another restore process may be running" 2
        exit 2
    fi
    
    # If no backup date specified, list available backups and prompt user to select one
    if [[ -z "$BACKUP_DATE" ]]; then
        available_backups=$(list_available_backups)
        
        if [[ $? -ne 0 ]]; then
            handle_error "Failed to list available backups" 3
            exit 3
        fi
        
        # If still no backup date (non-interactive mode), use the latest
        if [[ -z "$BACKUP_DATE" && ! -t 0 ]]; then
            BACKUP_DATE=$(echo "$available_backups" | head -n1)
            log_message "INFO" "Automatically selected latest backup: $BACKUP_DATE"
        fi
        
        # If still no backup date, exit with error
        if [[ -z "$BACKUP_DATE" ]]; then
            handle_error "No backup date specified or selected" 4
            exit 4
        fi
    fi
    
    # Validate selected backup
    if ! validate_backup "$BACKUP_DATE"; then
        handle_error "Validation failed for backup date: $BACKUP_DATE" 5
        exit 5
    fi
    
    log_message "INFO" "Starting restoration from backup: $BACKUP_DATE"
    
    # Track components restored and any failures
    local components_restored=0
    local components_failed=0
    local restore_summary=""
    
    # Restore raw data if requested
    if $RESTORE_RAW_DATA; then
        if restore_raw_data "$BACKUP_DATE"; then
            log_message "INFO" "Raw data restored successfully"
            ((components_restored++))
            restore_summary="${restore_summary}Raw data: SUCCESS\n"
        else
            log_message "ERROR" "Failed to restore raw data"
            ((components_failed++))
            restore_summary="${restore_summary}Raw data: FAILED\n"
        fi
    fi
    
    # Restore features if requested
    if $RESTORE_FEATURES; then
        if restore_features "$BACKUP_DATE"; then
            log_message "INFO" "Features restored successfully"
            ((components_restored++))
            restore_summary="${restore_summary}Features: SUCCESS\n"
        else
            log_message "ERROR" "Failed to restore features"
            ((components_failed++))
            restore_summary="${restore_summary}Features: FAILED\n"
        fi
    fi
    
    # Restore models if requested
    if $RESTORE_MODELS; then
        if restore_models "$BACKUP_DATE"; then
            log_message "INFO" "Models restored successfully"
            ((components_restored++))
            restore_summary="${restore_summary}Models: SUCCESS\n"
        else
            log_message "ERROR" "Failed to restore models"
            ((components_failed++))
            restore_summary="${restore_summary}Models: FAILED\n"
        fi
    fi
    
    # Restore forecasts if requested
    if $RESTORE_FORECASTS; then
        if restore_forecasts "$BACKUP_DATE"; then
            log_message "INFO" "Forecasts restored successfully"
            ((components_restored++))
            restore_summary="${restore_summary}Forecasts: SUCCESS\n"
        else
            log_message "ERROR" "Failed to restore forecasts"
            ((components_failed++))
            restore_summary="${restore_summary}Forecasts: FAILED\n"
        fi
    fi
    
    # Restore logs if requested
    if $RESTORE_LOGS; then
        if restore_logs "$BACKUP_DATE"; then
            log_message "INFO" "Logs restored successfully"
            ((components_restored++))
            restore_summary="${restore_summary}Logs: SUCCESS\n"
        else
            log_message "ERROR" "Failed to restore logs"
            ((components_failed++))
            restore_summary="${restore_summary}Logs: FAILED\n"
        fi
    fi
    
    # Release lock
    release_lock
    
    # Log restoration completion
    local completion_message="Restoration completed with $components_restored components restored successfully"
    if [[ $components_failed -gt 0 ]]; then
        completion_message="$completion_message and $components_failed components failed"
        log_message "WARNING" "$completion_message"
        send_notification "PARTIALLY COMPLETED" "$completion_message\n\n$restore_summary"
        exit 6
    else
        log_message "INFO" "$completion_message"
        send_notification "COMPLETED" "$completion_message\n\n$restore_summary"
        exit 0
    fi
}

# Execute main function
main "$@"