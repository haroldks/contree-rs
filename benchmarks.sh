#!/usr/bin/env bash

# Parallel Benchmark Script for LDS Rust Example
# Requires GNU Parallel

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
TIMEOUT=600.0  # seconds time limit (default from parser)
SUPPORT=1      # minimum support
DEPTHS=(3 4 5 6 7 8 9)  # Depths from 4 to 9
NBTHREAD=20    # Default number of parallel jobs
TEST_DATA_DIR="test_data"
BASE_RESULT_DIR="results_lds"

# LDS-specific parameters
USE_LDS=true
HEURISTIC_VALUES=("true" "false")  # Sort by heuristic
SPLIT_STRATEGIES=("mid" "first" "random")  # Split selection strategies

# Argument parsing with validation
parse_arguments() {
    local custom_input_dir=""
    local custom_output_dir=""
    local use_heuristic_only=false
    local custom_strategies=()

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --input-dir)
                custom_input_dir="$2"
                shift ;;
            --output-dir)
                custom_output_dir="$2"
                shift ;;
            --threads)
                NBTHREAD="$2"
                shift ;;
            --timeout)
                TIMEOUT="$2"
                shift ;;
            --support)
                SUPPORT="$2"
                shift ;;
            --depths)
                IFS=',' read -ra DEPTHS <<< "$2"
                shift ;;
            --heuristic-only)
                use_heuristic_only=true
                HEURISTIC_VALUES=("true") ;;
            --no-heuristic-only)
                use_heuristic_only=true
                HEURISTIC_VALUES=("false") ;;
            --strategy)
                custom_strategies+=("$2")
                shift ;;
            --no-lds)
                USE_LDS=false ;;
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

    # Use custom strategies if provided
    [[ ${#custom_strategies[@]} -gt 0 ]] && SPLIT_STRATEGIES=("${custom_strategies[@]}")

    # Use custom directories if provided
    [[ -n "$custom_input_dir" ]] && TEST_DATA_DIR="$custom_input_dir"
    [[ -n "$custom_output_dir" ]] && BASE_RESULT_DIR="$custom_output_dir"

    # Validate input directory
    [[ ! -d "$TEST_DATA_DIR" ]] && error_exit "Input directory not found: $TEST_DATA_DIR"

    # Generate input files list
    INPUT_FILES=($(find "$TEST_DATA_DIR" -type f -name "*.txt"))
    [[ ${#INPUT_FILES[@]} -eq 0 ]] && error_exit "No .txt files found in $TEST_DATA_DIR"
}

# Show help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Parallel benchmark script for LDS Rust example.

OPTIONS:
    --input-dir DIR          Input directory containing .txt dataset files (default: test_data)
    --output-dir DIR         Output directory for results (default: results_lds)
    --threads N              Number of parallel jobs (default: 20)
    --timeout SECONDS        Time limit in seconds (default: 600.0)
    --support N              Minimum support (default: 1)
    --depths D1,D2,...       Comma-separated list of depths (default: 4,5,6,7,8,9)
    --heuristic-only         Only run with heuristic enabled
    --no-heuristic-only      Only run without heuristic
    --strategy STRAT         Add split selection strategy (mid, first, random)
                             Can be specified multiple times
    --no-lds                 Disable LDS (default: enabled)
    --overwrite              Overwrite existing result files
    --dry-run                Print commands without executing
    --help                   Show this help message

EXAMPLES:
    # Run with default settings
    $0

    # Run with custom input/output directories
    $0 --input-dir my_data --output-dir my_results

    # Run only with heuristic enabled
    $0 --heuristic-only

    # Run with specific depths and timeout
    $0 --depths 4,5,6 --timeout 300

    # Run with 10 parallel jobs
    $0 --threads 10

    # Dry run to see commands
    $0 --dry-run
EOF
}

# Function to run an individual benchmark
run_benchmark() {
    local dataset="$1"
    local depth="$2"
    local output_dir="$3"
    local heuristic="$4"
    local strategy="$5"

    local dataset_name=$(basename "$dataset" .txt)

    # Build command (fast-d2 always enabled)
    local cmd="cargo run --release --example lds -- \
        --input $dataset \
        --depth $depth \
        --support $SUPPORT \
        --time-limit $TIMEOUT \
        --result-dir $output_dir \
        --split-selection-strategy $strategy \
        --fast-d2"

    # Add conditional flags
    [[ "$heuristic" == "true" ]] && cmd="$cmd --sort-by-heuristic"
    [[ "$USE_LDS" == "true" ]] && cmd="$cmd --use-lds"
    [[ "$OVERWRITE" == "true" ]] && cmd="$cmd --overwrite"

    local config_str="depth=$depth, heuristic=$heuristic, strategy=$strategy"
    log "Running LDS on $dataset_name with $config_str"

    if [[ "$dry_run" == true ]]; then
        echo "DRY RUN: $cmd"
        return 0
    fi

    # Run the actual benchmark
    eval "$cmd"
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log "WARNING: Benchmark failed for $dataset_name with $config_str (exit code: $exit_code)"
    fi

    return $exit_code
}

# Export functions and variables for parallel
export -f run_benchmark
export -f log
export -f error_exit
export TIMEOUT SUPPORT USE_LDS OVERWRITE dry_run

# Main execution function
run_benchmarks() {
    # Create main timestamp-based output directory
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="${BASE_RESULT_DIR}/run_${TIMEOUT}"
    mkdir -p "$output_dir"

    log "=========================================="
    log "LDS Benchmark Configuration"
    log "=========================================="
    log "Timeout: ${TIMEOUT}s"
    log "Support: $SUPPORT"
    log "Depths: ${DEPTHS[*]}"
    log "Fast D2: enabled (always)"
    log "Use LDS: $USE_LDS"
    log "Heuristic values: ${HEURISTIC_VALUES[*]}"
    log "Split strategies: ${SPLIT_STRATEGIES[*]}"
    log "Parallel threads: $NBTHREAD"
    log "Input directory: $TEST_DATA_DIR"
    log "Output directory: $output_dir"
    log "Number of datasets: ${#INPUT_FILES[@]}"
    log "Overwrite: ${OVERWRITE:-false}"
    log "=========================================="

    # Check if GNU Parallel is installed
    if ! command -v parallel &> /dev/null; then
        error_exit "GNU Parallel is required but not installed. Install with: sudo apt-get install parallel"
    fi

    # Check if cargo is available
    if ! command -v cargo &> /dev/null; then
        error_exit "Cargo is required but not installed"
    fi

    # Create command file for parallel
    CMDFILE=$(mktemp)
    trap "rm -f $CMDFILE" EXIT

    # Build all combinations
    local total_combinations=0
    for input_file in "${INPUT_FILES[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            for heuristic in "${HEURISTIC_VALUES[@]}"; do
                for strategy in "${SPLIT_STRATEGIES[@]}"; do
                    echo "run_benchmark $input_file $depth $output_dir $heuristic $strategy" >> "$CMDFILE"
                    ((total_combinations++))
                done
            done
        done
    done

    log "Total benchmark combinations: $total_combinations"
    log "Starting parallel execution..."

    # Run with parallel
    if [[ "$dry_run" == true ]]; then
        log "DRY RUN MODE - Commands to be executed:"
        cat "$CMDFILE"
    else
        parallel --bar --progress \
                 --joblog "${output_dir}/parallel_${TIMEOUT}.log" \
                 --results "${output_dir}/parallel_results" \
                 -j "$NBTHREAD" < "$CMDFILE"

        local parallel_exit=$?

        if [[ $parallel_exit -eq 0 ]]; then
            log "All benchmarks completed successfully"
        else
            log "WARNING: Some benchmarks may have failed (exit code: $parallel_exit)"
        fi
    fi

    log "=========================================="
    log "Benchmark Summary"
    log "=========================================="
    log "Results directory: $output_dir"
    log "Job log: ${output_dir}/parallel_${timestamp}.log"
    log "Total combinations: $total_combinations"

    if [[ "$dry_run" != true ]]; then
        # Count successful runs from joblog
        if [[ -f "${output_dir}/parallel_${timestamp}.log" ]]; then
            local successful=$(awk '$7 == 0 {count++} END {print count}' "${output_dir}/parallel_${timestamp}.log")
            local failed=$(awk '$7 != 0 && NR > 1 {count++} END {print count}' "${output_dir}/parallel_${timestamp}.log")
            log "Successful: ${successful:-0}"
            log "Failed: ${failed:-0}"
        fi
    fi
    log "=========================================="
}

# Main script execution
main() {
    # Set default values
    dry_run=false
    OVERWRITE=false

    parse_arguments "$@"
    run_benchmarks
}

# Execute main with all arguments
main "$@"