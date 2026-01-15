#!/bin/bash
# Performance benchmarking script for Reified RSpaces
#
# This script runs benchmarks with:
# - CPU affinity (pinning to specific cores)
# - Performance frequency governor (max CPU speed)
# - Optional perf profiling and flamegraph generation
#
# Usage:
#   ./scripts/benchmark-with-perf.sh                    # Run all benchmarks
#   ./scripts/benchmark-with-perf.sh --filter "atomic"  # Filter specific benchmarks
#   ./scripts/benchmark-with-perf.sh --profile         # Run with perf profiling
#   ./scripts/benchmark-with-perf.sh --flamegraph      # Generate flamegraphs
#   ./scripts/benchmark-with-perf.sh --quick           # Quick benchmark run

set -euo pipefail

# Configuration
BENCHMARK_NAME="spaces_benchmark"
CPU_CORES="${BENCHMARK_CPUS:-0,1}"  # Default: cores 0 and 1
PROFILE_DIR="target/perf-profile"
FLAMEGRAPH_DIR="target/flamegraphs"
BASELINE_DIR="docs/performance"

# Parse arguments
FILTER=""
PROFILE=false
FLAMEGRAPH=false
QUICK=false
SAVE_BASELINE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --flamegraph)
            FLAMEGRAPH=true
            PROFILE=true  # Flamegraph requires profiling
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --save-baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --cpus)
            CPU_CORES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --filter PATTERN   Run only benchmarks matching PATTERN"
            echo "  --profile          Run with perf profiling"
            echo "  --flamegraph       Generate flamegraphs (implies --profile)"
            echo "  --quick            Quick benchmark run (fewer iterations)"
            echo "  --save-baseline    Save results to docs/performance/"
            echo "  --cpus CORES       CPU cores to pin to (default: 0,1)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_tools() {
    local missing=()

    if ! command -v cargo &> /dev/null; then
        missing+=("cargo")
    fi

    if $PROFILE && ! command -v perf &> /dev/null; then
        missing+=("perf (linux-tools-generic)")
    fi

    if $FLAMEGRAPH && ! command -v flamegraph &> /dev/null; then
        warn "flamegraph not found. Install with: cargo install flamegraph"
        warn "Continuing without flamegraph generation..."
        FLAMEGRAPH=false
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing required tools: ${missing[*]}"
        exit 1
    fi
}

# Try to set CPU to performance mode (requires root or appropriate permissions)
set_cpu_performance() {
    info "Attempting to set CPU governor to performance mode..."

    if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" > "$cpu" 2>/dev/null || true
        done
        success "CPU governor set to performance"
    else
        warn "Cannot set CPU governor (requires root). Benchmarks may have variance."
        warn "Run with: sudo cpupower frequency-set -g performance"
    fi
}

# Build benchmarks
build_benchmarks() {
    info "Building benchmarks in release mode..."
    cargo build --bench "$BENCHMARK_NAME" --release
    success "Benchmarks built"
}

# Run benchmarks with optional CPU affinity
run_benchmarks() {
    local bench_args=("--bench" "$BENCHMARK_NAME")

    if [ -n "$FILTER" ]; then
        bench_args+=("--" "$FILTER")
    fi

    if $QUICK; then
        bench_args+=("--quick")
    fi

    # Build the taskset command for CPU affinity
    local taskset_cmd=""
    if command -v taskset &> /dev/null; then
        taskset_cmd="taskset -c $CPU_CORES"
        info "Pinning benchmarks to CPUs: $CPU_CORES"
    else
        warn "taskset not available, running without CPU affinity"
    fi

    info "Running benchmarks..."

    if $PROFILE; then
        run_with_perf "$taskset_cmd"
    else
        # Run with nice for lower scheduling priority variance
        if [ -n "$taskset_cmd" ]; then
            nice -n -5 $taskset_cmd cargo bench "${bench_args[@]}" 2>/dev/null || \
                $taskset_cmd cargo bench "${bench_args[@]}"
        else
            nice -n -5 cargo bench "${bench_args[@]}" 2>/dev/null || \
                cargo bench "${bench_args[@]}"
        fi
    fi

    success "Benchmarks complete"
}

# Run with perf profiling
run_with_perf() {
    local taskset_cmd="$1"

    mkdir -p "$PROFILE_DIR"

    local bench_binary
    bench_binary=$(find target/release/deps -name "${BENCHMARK_NAME}-*" -executable | head -1)

    if [ -z "$bench_binary" ]; then
        error "Benchmark binary not found. Run build first."
        exit 1
    fi

    info "Profiling with perf..."

    local bench_args=()
    if [ -n "$FILTER" ]; then
        bench_args+=("$FILTER")
    fi
    if $QUICK; then
        bench_args+=("--quick")
    fi
    bench_args+=("--bench")

    # Record with perf
    if [ -n "$taskset_cmd" ]; then
        sudo perf record -g --call-graph dwarf -o "$PROFILE_DIR/perf.data" \
            $taskset_cmd "$bench_binary" "${bench_args[@]}"
    else
        sudo perf record -g --call-graph dwarf -o "$PROFILE_DIR/perf.data" \
            "$bench_binary" "${bench_args[@]}"
    fi

    # Generate report
    info "Generating perf report..."
    sudo perf report -i "$PROFILE_DIR/perf.data" --stdio > "$PROFILE_DIR/perf-report.txt"
    sudo chown "$USER:$USER" "$PROFILE_DIR/perf.data" "$PROFILE_DIR/perf-report.txt"

    success "Perf profile saved to $PROFILE_DIR/"

    # Generate flamegraph if requested
    if $FLAMEGRAPH; then
        generate_flamegraph
    fi
}

# Generate flamegraph from perf data
generate_flamegraph() {
    if ! command -v flamegraph &> /dev/null; then
        warn "flamegraph not installed, skipping flamegraph generation"
        return
    fi

    mkdir -p "$FLAMEGRAPH_DIR"

    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)
    local output_file="$FLAMEGRAPH_DIR/flamegraph-$timestamp.svg"

    info "Generating flamegraph..."

    # Use cargo flamegraph or convert from perf data
    if [ -f "$PROFILE_DIR/perf.data" ]; then
        sudo perf script -i "$PROFILE_DIR/perf.data" | \
            flamegraph --perfdata /dev/stdin -o "$output_file" 2>/dev/null || \
            warn "Flamegraph generation failed"
    fi

    if [ -f "$output_file" ]; then
        sudo chown "$USER:$USER" "$output_file"
        success "Flamegraph saved to $output_file"
    fi
}

# Save baseline results
save_baseline() {
    if ! $SAVE_BASELINE; then
        return
    fi

    mkdir -p "$BASELINE_DIR"

    local timestamp
    timestamp=$(date +%Y-%m-%d)
    local baseline_file="$BASELINE_DIR/baseline-$timestamp.md"

    info "Saving baseline results to $baseline_file..."

    cat > "$baseline_file" << EOF
# Performance Baseline - $timestamp

## System Information

\`\`\`
$(uname -a)
$(lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|MHz|cache" | head -20)
\`\`\`

## Benchmark Results

### Summary

Benchmarks run with:
- CPU cores: $CPU_CORES
- Quick mode: $QUICK

### Results

\`\`\`
$(cat target/criterion/report/index.html 2>/dev/null | grep -oP '(?<=<pre>).*?(?=</pre>)' | head -100 || echo "Results not found - check target/criterion/")
\`\`\`

## Criterion Reports

Full reports available at: \`target/criterion/report/index.html\`

EOF

    success "Baseline saved to $baseline_file"
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  Benchmark Summary"
    echo "=========================================="
    echo ""
    echo "Results location: target/criterion/report/index.html"

    if $PROFILE; then
        echo "Perf profile:     $PROFILE_DIR/perf-report.txt"
    fi

    if $FLAMEGRAPH && [ -d "$FLAMEGRAPH_DIR" ]; then
        echo "Flamegraphs:      $FLAMEGRAPH_DIR/"
    fi

    if $SAVE_BASELINE; then
        echo "Baseline:         $BASELINE_DIR/"
    fi

    echo ""
    echo "View HTML report with:"
    echo "  open target/criterion/report/index.html"
    echo ""
}

# Main
main() {
    info "Reified RSpaces Performance Benchmark"
    info "======================================"
    echo ""

    check_tools
    set_cpu_performance
    build_benchmarks
    run_benchmarks
    save_baseline
    print_summary

    success "All done!"
}

main "$@"
