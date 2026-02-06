# =================================================================
# F1R3FLY RUST NODE - LOCAL DEVELOPMENT COMMANDS
# =================================================================
# Run `just` to see all available commands.
# Run `just <command>` to execute a command.
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - just command runner (included in nix flake)

# Configuration paths
local_dir := justfile_directory() / "run-local"
standalone_conf := local_dir / "conf/standalone.conf"
standalone_genesis := local_dir / "genesis/standalone"
standalone_data := local_dir / "data/standalone"

# Validator credentials (bootstrap wallet)
standalone_private_key := "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657"

# Default recipe - show available commands
default:
    @just --list

# =================================================================
# BUILD
# =================================================================

# Build the Rust node in release mode
build:
    cargo build --release -p node

# Build the Rust node in debug mode (faster compile, slower runtime)
build-debug:
    cargo build -p node

# =================================================================
# STANDALONE NODE
# =================================================================

# Setup standalone node data directory (run once before first start)
setup-standalone:
    @echo "Setting up standalone node data directory..."
    mkdir -p {{standalone_data}}/genesis
    cp {{standalone_genesis}}/bonds.txt {{standalone_data}}/genesis/
    cp {{standalone_genesis}}/wallets.txt {{standalone_data}}/genesis/
    @echo "Done. Data directory: {{standalone_data}}"

# Run standalone node locally
run-standalone: build setup-standalone
    @echo "Starting standalone Rust node..."
    @echo "  Config: {{standalone_conf}}"
    @echo "  Data:   {{standalone_data}}"
    @echo ""
    ./target/release/node run -s \
        --config-file={{standalone_conf}} \
        --validator-private-key={{standalone_private_key}} \
        --host=localhost \
        --no-upnp

# Run standalone node in debug mode (for development)
run-standalone-debug: build-debug setup-standalone
    ./target/debug/node run -s \
        --config-file={{standalone_conf}} \
        --validator-private-key={{standalone_private_key}} \
        --host=localhost \
        --no-upnp

# Clean standalone node data (fresh start from genesis)
clean-standalone:
    @echo "Removing standalone node data..."
    rm -rf {{standalone_data}}
    @echo "Done. Run 'just setup-standalone' or 'just run-standalone' to reinitialize."

# =================================================================
# HELP
# =================================================================

# Show node CLI help
help:
    ./target/release/node --help 2>/dev/null || cargo run --release -p node -- --help

# Show 'run' subcommand options
run-help:
    ./target/release/node run --help 2>/dev/null || cargo run --release -p node -- run --help
