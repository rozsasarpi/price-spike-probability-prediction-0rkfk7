#!/bin/bash
#
# backup_data.sh - Backup script for ERCOT RTLMP spike prediction system
#
# This script creates compressed archives of raw data, features, models, and forecasts
# with appropriate validation and error handling.
#

# Exit on error, undefined variables, and propagate pipe errors
set -euo pipefail

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

# Default directories and settings (can be overridden by environment variables)
BACKUP_DIR=${BACKUP_DIR:-"$PROJECT_ROOT/backups"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/data"}
FEATURES_DIR=${FEATURES_DIR:-"$PROJECT_ROOT/data/features"}
MODELS_DIR=${MODELS_DIR:-"$PROJECT_ROOT/models"}
FORECASTS_DIR=${FORECASTS_DIR:-"$PROJECT_ROOT/forecasts"}
LOGS_DIR=${LOGS_DIR:-"$PROJECT_ROOT/logs"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DATE=$(date +%Y%m%d)
CURRENT_BACKUP_DIR="$BACKUP_DIR/$BACKUP_DATE"
LOG_FILE="$LOGS_DIR/backup_${TIMESTAMP}.log"
RETENTION_DAYS=${RETENTION_DAYS:-30}
FULL_BACKUP_DAY=${FULL_BACKUP_DAY:-Sunday}
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-""}
LOCK_FILE="$LOGS_DIR/backup_data.lock"

# Default options
FORCE_FULL=0
SKIP_CLEANUP=0

# Usage information
function show_usage {
    echo "Usage: $0 [--full] [--no-cleanup] [--retention DAYS] [--help]"
    echo
    echo "Options:"
    echo "  --full         Force a full backup regardless of the day of week"
    echo "  --no-cleanup   Skip cleanup of old backups"
    echo "  --retention DAYS  Number of days to retain backups (default: $RETENTION_DAYS)"
    echo "  --help         Display this help message"
    echo
    echo "Environment variables:"
    echo "  BACKUP_DIR     Directory where backups will be stored"
    echo "  DATA_DIR       Directory containing raw data"
    echo "  FEATURES_DIR   Directory containing engineered features"
    echo "  MODELS_DIR     Directory containing trained models"
    echo "  FORECASTS_DIR  Directory containing generated forecasts"
    echo "  LOGS_DIR       Directory containing application logs"
    echo "  RETENTION_DAYS Number of days to retain backups"
    echo "  FULL_BACKUP_DAY Day of the week for full backups"
    echo "  NOTIFICATION_EMAIL Email address for notifications"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FORCE_FULL=1
            shift
            ;;
        --no-cleanup)
            SKIP_CLEANUP=1
            shift
            ;;
        --retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Log message to both stdout and log file
function log_message {
    local level="$1"
    local message="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local formatted_message="[$timestamp] [$level] $message"
    
    echo "$formatted_message"
    echo "$formatted_message" >> "$LOG_FILE"
}

# Check if required directories exist and are accessible
function check_directories {
    log_message "INFO" "Checking required directories..."
    
    # Check project root
    if [[ ! -d "$PROJECT_ROOT" || ! -r "$PROJECT_ROOT" ]]; then
        log_message "ERROR" "Project root directory doesn't exist or is not readable: $PROJECT_ROOT"
        return 1
    fi
    
    # Check data directories
    local dirs=("$DATA_DIR" "$FEATURES_DIR" "$MODELS_DIR" "$FORECASTS_DIR" "$LOGS_DIR")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" || ! -r "$dir" ]]; then
            log_message "WARNING" "Directory doesn't exist or is not readable: $dir"
        fi
    done
    
    # Create backup directories if they don't exist
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$CURRENT_BACKUP_DIR"
    
    log_message "INFO" "Directory check completed successfully"
    return 0
}

# Acquire a lock to prevent concurrent execution
function acquire_lock {
    log_message "INFO" "Acquiring lock..."
    
    if [[ -f "$LOCK_FILE" ]]; then
        local pid=$(cat "$LOCK_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_message "WARNING" "Another backup process is already running (PID: $pid)"
            return 1
        else
            log_message "WARNING" "Removing stale lock file (PID: $pid)"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log_message "INFO" "Lock acquired (PID: $$)"
    return 0
}

# Release the lock
function release_lock {
    if [[ -f "$LOCK_FILE" && "$(cat "$LOCK_FILE")" == "$$" ]]; then
        rm -f "$LOCK_FILE"
        log_message "INFO" "Lock released (PID: $$)"
        return 0
    fi
    return 1
}

# Determine if today is the day for a full backup
function is_full_backup_day {
    local current_day=$(date +%A)
    if [[ "$current_day" == "$FULL_BACKUP_DAY" ]]; then
        return 0  # Today is full backup day
    else
        return 1  # Today is not full backup day
    fi
}

# Back up raw data files
function backup_raw_data {
    local backup_type="$1"
    log_message "INFO" "Starting raw data backup (type: $backup_type)..."
    
    local backup_file="$CURRENT_BACKUP_DIR/raw_data_${backup_type}_${TIMESTAMP}.tar.gz"
    
    if [[ "$backup_type" == "full" ]]; then
        # Full backup
        if tar -czf "$backup_file" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")" --exclude="$(basename "$FEATURES_DIR")"; then
            log_message "INFO" "Full raw data backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create full raw data backup"
            return 1
        fi
    else
        # Incremental backup using rsync
        local latest_full=$(find "$BACKUP_DIR" -name "raw_data_full_*.tar.gz" -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2)
        
        if [[ -z "$latest_full" ]]; then
            log_message "WARNING" "No previous full backup found, creating full backup instead"
            backup_raw_data "full"
            return $?
        fi
        
        # Create temporary directory for extracting previous backup
        local temp_dir=$(mktemp -d)
        tar -xzf "$latest_full" -C "$temp_dir"
        
        # Create incremental backup using rsync
        if rsync -a --delete --link-dest="$temp_dir/data" "$DATA_DIR/" "$temp_dir/data_new/" && \
           tar -czf "$backup_file" -C "$temp_dir" "data_new"; then
            log_message "INFO" "Incremental raw data backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create incremental raw data backup"
            rm -rf "$temp_dir"
            return 1
        fi
        
        # Clean up
        rm -rf "$temp_dir"
    fi
    
    # Verify the backup
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_message "INFO" "Raw data backup verified successfully"
        return 0
    else
        log_message "ERROR" "Raw data backup verification failed"
        return 1
    fi
}

# Back up engineered features
function backup_features {
    local backup_type="$1"
    log_message "INFO" "Starting features backup (type: $backup_type)..."
    
    local backup_file="$CURRENT_BACKUP_DIR/features_${backup_type}_${TIMESTAMP}.tar.gz"
    
    if [[ "$backup_type" == "full" ]]; then
        # Full backup
        if tar -czf "$backup_file" -C "$(dirname "$FEATURES_DIR")" "$(basename "$FEATURES_DIR")"; then
            log_message "INFO" "Full features backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create full features backup"
            return 1
        fi
    else
        # Incremental backup using rsync
        local latest_full=$(find "$BACKUP_DIR" -name "features_full_*.tar.gz" -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2)
        
        if [[ -z "$latest_full" ]]; then
            log_message "WARNING" "No previous full features backup found, creating full backup instead"
            backup_features "full"
            return $?
        fi
        
        # Create temporary directory for extracting previous backup
        local temp_dir=$(mktemp -d)
        tar -xzf "$latest_full" -C "$temp_dir"
        
        # Create incremental backup using rsync
        if rsync -a --delete --link-dest="$temp_dir/features" "$FEATURES_DIR/" "$temp_dir/features_new/" && \
           tar -czf "$backup_file" -C "$temp_dir" "features_new"; then
            log_message "INFO" "Incremental features backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create incremental features backup"
            rm -rf "$temp_dir"
            return 1
        fi
        
        # Clean up
        rm -rf "$temp_dir"
    fi
    
    # Verify the backup
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_message "INFO" "Features backup verified successfully"
        return 0
    else
        log_message "ERROR" "Features backup verification failed"
        return 1
    fi
}

# Back up trained model artifacts
function backup_models {
    local backup_type="$1"  # Ignore backup type for models, always do full backup
    log_message "INFO" "Starting models backup..."
    
    local backup_file="$CURRENT_BACKUP_DIR/models_full_${TIMESTAMP}.tar.gz"
    
    # Full backup (always use full backup for models to ensure consistency)
    if tar -czf "$backup_file" -C "$(dirname "$MODELS_DIR")" "$(basename "$MODELS_DIR")"; then
        log_message "INFO" "Models backup created: $backup_file"
    else
        log_message "ERROR" "Failed to create models backup"
        return 1
    fi
    
    # Verify the backup
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_message "INFO" "Models backup verified successfully"
        return 0
    else
        log_message "ERROR" "Models backup verification failed"
        return 1
    fi
}

# Back up forecast data
function backup_forecasts {
    local backup_type="$1"
    log_message "INFO" "Starting forecasts backup (type: $backup_type)..."
    
    local backup_file="$CURRENT_BACKUP_DIR/forecasts_${backup_type}_${TIMESTAMP}.tar.gz"
    
    if [[ "$backup_type" == "full" ]]; then
        # Full backup
        if tar -czf "$backup_file" -C "$(dirname "$FORECASTS_DIR")" "$(basename "$FORECASTS_DIR")"; then
            log_message "INFO" "Full forecasts backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create full forecasts backup"
            return 1
        fi
    else
        # Incremental backup using rsync
        local latest_full=$(find "$BACKUP_DIR" -name "forecasts_full_*.tar.gz" -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2)
        
        if [[ -z "$latest_full" ]]; then
            log_message "WARNING" "No previous full forecasts backup found, creating full backup instead"
            backup_forecasts "full"
            return $?
        fi
        
        # Create temporary directory for extracting previous backup
        local temp_dir=$(mktemp -d)
        tar -xzf "$latest_full" -C "$temp_dir"
        
        # Create incremental backup using rsync
        if rsync -a --delete --link-dest="$temp_dir/forecasts" "$FORECASTS_DIR/" "$temp_dir/forecasts_new/" && \
           tar -czf "$backup_file" -C "$temp_dir" "forecasts_new"; then
            log_message "INFO" "Incremental forecasts backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create incremental forecasts backup"
            rm -rf "$temp_dir"
            return 1
        fi
        
        # Clean up
        rm -rf "$temp_dir"
    fi
    
    # Verify the backup
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_message "INFO" "Forecasts backup verified successfully"
        return 0
    else
        log_message "ERROR" "Forecasts backup verification failed"
        return 1
    fi
}

# Back up application logs
function backup_logs {
    local backup_type="$1"
    log_message "INFO" "Starting logs backup (type: $backup_type)..."
    
    local backup_file="$CURRENT_BACKUP_DIR/logs_${backup_type}_${TIMESTAMP}.tar.gz"
    
    if [[ "$backup_type" == "full" ]]; then
        # Full backup
        if tar -czf "$backup_file" -C "$(dirname "$LOGS_DIR")" "$(basename "$LOGS_DIR")"; then
            log_message "INFO" "Full logs backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create full logs backup"
            return 1
        fi
    else
        # Incremental backup using rsync
        local latest_full=$(find "$BACKUP_DIR" -name "logs_full_*.tar.gz" -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2)
        
        if [[ -z "$latest_full" ]]; then
            log_message "WARNING" "No previous full logs backup found, creating full backup instead"
            backup_logs "full"
            return $?
        fi
        
        # Create temporary directory for extracting previous backup
        local temp_dir=$(mktemp -d)
        tar -xzf "$latest_full" -C "$temp_dir"
        
        # Create incremental backup using rsync
        if rsync -a --delete --link-dest="$temp_dir/logs" "$LOGS_DIR/" "$temp_dir/logs_new/" && \
           tar -czf "$backup_file" -C "$temp_dir" "logs_new"; then
            log_message "INFO" "Incremental logs backup created: $backup_file"
        else
            log_message "ERROR" "Failed to create incremental logs backup"
            rm -rf "$temp_dir"
            return 1
        fi
        
        # Clean up
        rm -rf "$temp_dir"
    fi
    
    # Verify the backup
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_message "INFO" "Logs backup verified successfully"
        return 0
    else
        log_message "ERROR" "Logs backup verification failed"
        return 1
    fi
}

# Remove backups older than RETENTION_DAYS
function cleanup_old_backups {
    log_message "INFO" "Starting cleanup of old backups (retention: $RETENTION_DAYS days)..."
    
    local old_dirs=$(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +$RETENTION_DAYS)
    
    if [[ -z "$old_dirs" ]]; then
        log_message "INFO" "No old backups to clean up"
        return 0
    fi
    
    for dir in $old_dirs; do
        log_message "INFO" "Removing old backup: $dir"
        if rm -rf "$dir"; then
            log_message "INFO" "Removed old backup: $dir"
        else
            log_message "ERROR" "Failed to remove old backup: $dir"
            return 1
        fi
    done
    
    log_message "INFO" "Backup cleanup completed successfully"
    return 0
}

# Send a notification about backup status
function send_notification {
    local status="$1"
    local message="$2"
    
    # Skip if notification email is not set
    if [[ -z "$NOTIFICATION_EMAIL" ]]; then
        return 0
    fi
    
    log_message "INFO" "Sending $status notification to $NOTIFICATION_EMAIL"
    
    local subject="ERCOT RTLMP Backup $status - $(hostname) - $BACKUP_DATE"
    local body="Backup Status: $status\nTimestamp: $(date)\nHostname: $(hostname)\n\n$message\n\nBackup Directory: $CURRENT_BACKUP_DIR"
    
    if echo -e "$body" | mail -s "$subject" "$NOTIFICATION_EMAIL"; then
        log_message "INFO" "Notification sent successfully"
        return 0
    else
        log_message "ERROR" "Failed to send notification"
        return 1
    fi
}

# Handle errors during the backup process
function handle_error {
    local error_message="$1"
    local error_code="${2:-1}"
    
    log_message "ERROR" "$error_message"
    send_notification "FAILED" "$error_message"
    
    release_lock
    
    return $error_code
}

# Cleanup function called on script exit or error
function cleanup {
    release_lock
    log_message "INFO" "Backup script cleanup completed"
}

# Set up trap to ensure cleanup on exit
trap cleanup EXIT

# Main function
function main {
    local overall_status=0
    local summary=""
    
    # Log backup start
    log_message "INFO" "Starting backup process at $(date)"
    summary="Backup process started at $(date)\n"
    
    # Check directories
    if ! check_directories; then
        return $(handle_error "Failed to verify required directories" 1)
    fi
    
    # Acquire lock
    if ! acquire_lock; then
        return $(handle_error "Failed to acquire lock, another backup may be running" 2)
    fi
    
    # Determine backup type (full or incremental)
    local backup_type="incremental"
    if [[ $FORCE_FULL -eq 1 ]] || is_full_backup_day; then
        backup_type="full"
    fi
    log_message "INFO" "Backup type: $backup_type"
    summary+="Backup type: $backup_type\n"
    
    # Back up each data type
    local backup_functions=("backup_raw_data" "backup_features" "backup_models" "backup_forecasts" "backup_logs")
    local status_msgs=()
    
    for func in "${backup_functions[@]}"; do
        # Models are always backed up in full
        if [[ "$func" == "backup_models" ]]; then
            if ! $func "full"; then
                overall_status=1
                status_msgs+=("$func: FAILED")
                continue
            fi
        else
            if ! $func "$backup_type"; then
                overall_status=1
                status_msgs+=("$func: FAILED")
                continue
            fi
        fi
        status_msgs+=("$func: SUCCESS")
    done
    
    # Clean up old backups if not skipped
    if [[ $SKIP_CLEANUP -eq 0 ]]; then
        if ! cleanup_old_backups; then
            overall_status=1
            status_msgs+=("cleanup_old_backups: FAILED")
        else
            status_msgs+=("cleanup_old_backups: SUCCESS")
        fi
    else
        log_message "INFO" "Skipping cleanup of old backups"
        status_msgs+=("cleanup_old_backups: SKIPPED")
    fi
    
    # Add status messages to summary
    for msg in "${status_msgs[@]}"; do
        summary+="$msg\n"
    done
    
    # Log backup completion
    log_message "INFO" "Backup process completed at $(date) with status: $([ $overall_status -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
    summary+="Backup process completed at $(date)\nOverall status: $([ $overall_status -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')\n"
    
    # Send notification
    if [[ $overall_status -eq 0 ]]; then
        send_notification "SUCCESS" "$summary"
    else
        send_notification "FAILED" "$summary"
    fi
    
    return $overall_status
}

# Run the main function
main
exit $?