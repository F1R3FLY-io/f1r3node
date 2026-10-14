#!/usr/bin/env bash
# Script to run all PeTTa-related tests
# Usage: ./RUN_PETTA_TESTS.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Check if PETTA_PATH is set, otherwise use default
if [ -z "$PETTA_PATH" ]; then
    PETTA_PATH="$REPO_ROOT/PeTTa"
    echo -e "${YELLOW}PETTA_PATH not set, using default: $PETTA_PATH${NC}"
fi

# Check if PeTTa exists
if [ ! -f "$PETTA_PATH/src/metta.pl" ]; then
    echo -e "${YELLOW}Warning: PeTTa not found at $PETTA_PATH/src/metta.pl${NC}"
    echo "Tests will skip PeTTa-dependent operations"
    echo ""
fi

export PETTA_PATH

echo -e "${BLUE}=== Running PeTTa Test Suite ===${NC}"
echo ""

# Run unit tests
echo -e "${GREEN}[1/4] Running unit tests for value_to_par...${NC}"
cargo test --package rholang --lib swi_prolog_service::tests
echo ""

# Run direct execution tests
echo -e "${GREEN}[2/4] Running direct PeTTa execution tests...${NC}"
cargo test --package rholang --test swipl_petta_execution_spec -- --nocapture
echo ""

# Run integration tests
echo -e "${GREEN}[3/4] Running Rholang integration tests...${NC}"
cargo test --package rholang --test swipl_petta_integration_spec -- --nocapture
echo ""

# Run replay tests
echo -e "${GREEN}[4/4] Running replay tests (non-deterministic operations)...${NC}"
cargo test --package rholang --test swipl_petta_replay_spec -- --nocapture
echo ""

echo -e "${BLUE}=== All PeTTa tests completed successfully! ===${NC}"
