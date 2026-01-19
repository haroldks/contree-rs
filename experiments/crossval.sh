#!/usr/bin/env bash

# Parallel Benchmark Script for Decision Tree Optimization (Python/pycontree Version)
# Requires GNU Parallel: sudo apt-get install parallel

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Error handling function
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Default configuration
TIMEOUT=600.0
MAX_DEPTHS=(3 4 5 6 7 8)
NBTHREAD=35
TEST_DATA_DIR="test_data"
BASE_RESULT_DIR="results_dt"
EXECUTABLE="python experiments.py"  # Points to your Python script

# Argument parsing with validation
parse_arguments() {
    local custom_input_dir=""
    local custom_output_dir=""
    local custom_executable=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --input-dir)
                custom_input_dir="$2"
                shift ;;
            --output-dir)
                custom_output_dir="$2"
                shift ;;
            --executable)
                custom_executable="$2"
                shift ;;
            --threads)
                NBTHREAD="$2"
                shift ;;
            --max-depths)
                IFS=',' read -ra MAX_DEPTHS <<< "$2"
                shift ;;
            --dry-run)
                dry_run=true ;;
            --overwrite)
                OVERWRITE=true ;;
            --help)
                show_help
                exit 0 ;;
            *) error_exit "Unknown parameter: $1. Use --help for usage information." ;;
        esac
        shift
    done

    # Use custom values if provided
    [[ -n "$custom_input_dir" ]] && TEST_DATA_DIR="$custom_input_dir"
    [[ -n "$custom_output_dir" ]] && BASE_RESULT_DIR="$custom_output_dir"
    [[ -n "$custom_executable" ]] && EXECUTABLE="$custom_executable"

    # Validate input directory
    [[ ! -d "$TEST_DATA_DIR" ]] && error_exit "Input directory not found: $TEST_DATA_DIR"

    # Generate input files list
    INPUT_FILES=($(find "$TEST_DATA_DIR" -type f -name "*.txt" -o -name "*.csv" -o -name "*.data"))
    [[ ${#INPUT_FILES[@]} -eq 0 ]] && error_exit "No dataset files found in $TEST_DATA_DIR"
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]
OPTIONS:
    --input-dir DIR          Input directory (default: test_data)
    --output-dir DIR         Output directory (default: results_dt)
    --executable CMD         Command to run (default: python3 run.py)
    --threads N              Parallel jobs (default: 20)
    --max-depths D1,D2,...   Depths (default: 3,4,5)
    --overwrite              Overwrite existing results
    --dry-run                Print commands without executing
EOF
}

# Function to run an individual benchmark
run_benchmark() {
    local dataset="$1"
    local max_depth="$2"
    local output_base_dir="$3"

    local dataset_name=$(basename "$dataset")
    dataset_name="${dataset_name%.*}"

    # Create dataset-specific subdirectory
    local dataset_dir="${output_base_dir}/${dataset_name}"
    mkdir -p "$dataset_dir"

    # Python will output a JSON file for each depth
    local output_file="${dataset_dir}/${max_depth}.json"

    # Skip if file exists and not overwriting
    if [[ -f "$output_file" && "$OVERWRITE" != "true" ]]; then
        log "Skipping existing: $output_file"
        return 0
    fi

    # Build command to match the Python script arguments
    local cmd="$EXECUTABLE \
        $dataset \
        $max_depth \
        $output_file"

    log "Running: $dataset_name (depth $max_depth)"

    if [[ "$dry_run" == true ]]; then
        echo "DRY RUN: $cmd"
        return 0
    fi

    # Execute
    eval "$cmd"
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log "WARNING: Failed $dataset_name depth $max_depth"
    fi

    return $exit_code
}

export -f run_benchmark
export -f log
export EXECUTABLE OVERWRITE dry_run

run_benchmarks() {
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="${BASE_RESULT_DIR}/run_${timestamp}"
    mkdir -p "$output_dir"

    log "=========================================="
    log "Python/ConTree Benchmark Initialized"
    log "=========================================="
    log "Command: $EXECUTABLE"
    log "Max Depths: ${MAX_DEPTHS[*]}"
    log "Parallel threads: $NBTHREAD"
    log "Input directory: $TEST_DATA_DIR"
    log "Output directory: $output_dir"
    log "=========================================="

    if ! command -v parallel &> /dev/null; then
        error_exit "GNU Parallel is required. Install with: sudo apt-get install parallel"
    fi

    CMDFILE=$(mktemp)
    trap "rm -f $CMDFILE" EXIT

    for input_file in "${INPUT_FILES[@]}"; do
        for depth in "${MAX_DEPTHS[@]}"; do
            echo "run_benchmark $input_file $depth $output_dir" >> "$CMDFILE"
        done
    done

    if [[ "$dry_run" == true ]]; then
        cat "$CMDFILE"
    else
        parallel --bar --progress \
                 --joblog "${output_dir}/parallel.log" \
                 -j "$NBTHREAD" < "$CMDFILE"
    fi

    log "Done. Results located in: $output_dir"
}

main() {
    dry_run=false
    OVERWRITE=false
    parse_arguments "$@"
    run_benchmarks
}

main "$@"
