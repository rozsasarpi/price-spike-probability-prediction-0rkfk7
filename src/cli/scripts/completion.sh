#!/bin/bash
#
# Bash completion script for rtlmp_predict CLI tool
#
# This script provides tab completion for rtlmp_predict commands, options,
# and arguments to improve user experience and productivity.
#
# Installation:
#   1. Place this script in a directory (e.g., ~/.bash_completion.d/)
#   2. Add the following line to your ~/.bashrc or ~/.bash_profile:
#      source ~/.bash_completion.d/completion.sh
#   3. Restart your shell or run: source ~/.bashrc
#
# Usage: Once installed, tab completion will work for rtlmp_predict commands
#        and their respective options.

# Define available commands
COMMANDS=(
    "fetch-data"
    "train"
    "predict"
    "backtest"
    "evaluate"
    "visualize"
)

# Define global options available for all commands
GLOBAL_OPTIONS=(
    "--config"
    "--verbose"
    "--no-verbose"
    "--log-level"
    "--output-dir"
    "--help"
)

# Define command-specific options
FETCH_DATA_OPTIONS=(
    "--data-type"
    "--start-date"
    "--end-date"
    "--nodes"
    "--output-path"
    "--output-format"
    "--force-refresh"
    "--use-cache"
    "--verbose"
    "--no-verbose"
)

TRAIN_OPTIONS=(
    "--start-date"
    "--end-date"
    "--model-type"
    "--threshold"
    "--nodes"
    "--hyperparameters"
    "--output-dir"
    "--verbose"
    "--no-verbose"
)

PREDICT_OPTIONS=(
    "--threshold"
    "--node"
    "--forecast-date"
    "--horizon"
    "--output-path"
    "--output-format"
    "--verbose"
    "--no-verbose"
)

BACKTEST_OPTIONS=(
    "--start-date"
    "--end-date"
    "--threshold"
    "--nodes"
    "--model-version"
    "--output-path"
    "--verbose"
    "--no-verbose"
)

EVALUATE_OPTIONS=(
    "--model-version"
    "--threshold"
    "--start-date"
    "--end-date"
    "--output-path"
    "--verbose"
    "--no-verbose"
)

VISUALIZE_OPTIONS=(
    "--forecast-id"
    "--model-version"
    "--output-path"
    "--output-format"
    "--type"
    "--verbose"
    "--no-verbose"
)

# Function to get command-specific options
_get_command_options() {
    local command="$1"
    
    case "$command" in
        "fetch-data")
            echo "${FETCH_DATA_OPTIONS[@]} ${GLOBAL_OPTIONS[@]}"
            ;;
        "train")
            echo "${TRAIN_OPTIONS[@]} ${GLOBAL_OPTIONS[@]}"
            ;;
        "predict")
            echo "${PREDICT_OPTIONS[@]} ${GLOBAL_OPTIONS[@]}"
            ;;
        "backtest")
            echo "${BACKTEST_OPTIONS[@]} ${GLOBAL_OPTIONS[@]}"
            ;;
        "evaluate")
            echo "${EVALUATE_OPTIONS[@]} ${GLOBAL_OPTIONS[@]}"
            ;;
        "visualize")
            echo "${VISUALIZE_OPTIONS[@]} ${GLOBAL_OPTIONS[@]}"
            ;;
        *)
            echo "${GLOBAL_OPTIONS[@]}"
            ;;
    esac
}

# Main completion function
_rtlmp_predict_completions() {
    # Get current and previous words
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Get the command (if any)
    local command=""
    for ((i=1; i < COMP_CWORD; i++)); do
        if [[ "${COMP_WORDS[i]}" == "fetch-data" || 
               "${COMP_WORDS[i]}" == "train" || 
               "${COMP_WORDS[i]}" == "predict" || 
               "${COMP_WORDS[i]}" == "backtest" || 
               "${COMP_WORDS[i]}" == "evaluate" || 
               "${COMP_WORDS[i]}" == "visualize" ]]; then
            command="${COMP_WORDS[i]}"
            break
        fi
    done
    
    # If completing a command
    if [[ -z "$command" && $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${COMMANDS[*]}" -- "$cur"))
        return 0
    fi
    
    # If completing options for a command
    if [[ -n "$command" ]]; then
        local options=$(_get_command_options "$command")
        COMPREPLY=($(compgen -W "$options" -- "$cur"))
        return 0
    fi
    
    # Default to no completions
    COMPREPLY=()
    return 0
}

# Bash completion function registered with the complete command
_complete_rtlmp_predict() {
    _rtlmp_predict_completions
    return 0
}

# Register completion function for rtlmp_predict
complete -F _complete_rtlmp_predict rtlmp_predict