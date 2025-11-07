#!/bin/bash

# test-llm.sh - Test script for LLM system processes
#
# This script tests both OpenAI and Ollama integrations
# and provides clear feedback on what's working.
#
# Usage:
#   ./test-llm.sh [openai|ollama|both]
#
# Examples:
#   ./test-llm.sh openai    # Test OpenAI only
#   ./test-llm.sh ollama    # Test Ollama only
#   ./test-llm.sh both      # Test both (default)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine what to test
TEST_MODE="${1:-both}"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "$1"
}

# ==============================================================================
# Check Prerequisites
# ==============================================================================

check_openai_prereqs() {
    print_header "Checking OpenAI Prerequisites"

    # Check if OPENAI_ENABLED is set
    if [[ -z "${OPENAI_ENABLED}" ]]; then
        print_warning "OPENAI_ENABLED not set, checking if enabled in config..."
        # Continue - may be enabled in config file
    else
        print_success "OPENAI_ENABLED=${OPENAI_ENABLED}"
    fi

    # Check if API key is set
    if [[ -z "${OPENAI_SCALA_CLIENT_API_KEY}" ]]; then
        print_error "OPENAI_SCALA_CLIENT_API_KEY not set"
        print_info "Set it with: export OPENAI_SCALA_CLIENT_API_KEY=sk-your-key-here"
        return 1
    else
        print_success "OPENAI_SCALA_CLIENT_API_KEY is set"
    fi

    print_success "OpenAI prerequisites OK"
    return 0
}

check_ollama_prereqs() {
    print_header "Checking Ollama Prerequisites"

    # Check if Ollama is installed
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama not installed"
        print_info "Install from: https://ollama.com/download"
        return 1
    else
        print_success "Ollama is installed"
    fi

    # Check if Ollama server is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_error "Ollama server not running"
        print_info "Start with: ollama serve"
        return 1
    else
        print_success "Ollama server is running"
    fi

    # Check if llama4 model is available
    if ollama list | grep -q "llama4"; then
        print_success "llama4 model is available"
    else
        print_warning "llama4 model not found"
        print_info "Pull with: ollama pull llama4:latest"
        # Don't fail - may work with other models
    fi

    # Check if OLLAMA_ENABLED is set
    if [[ -z "${OLLAMA_ENABLED}" ]]; then
        print_warning "OLLAMA_ENABLED not set, checking if enabled in config..."
    else
        print_success "OLLAMA_ENABLED=${OLLAMA_ENABLED}"
    fi

    print_success "Ollama prerequisites OK"
    return 0
}

check_rnode() {
    print_header "Checking RNode"

    if ! command -v rnode &> /dev/null; then
        print_error "rnode command not found"
        print_info "Make sure RNode is installed and in your PATH"
        return 1
    else
        print_success "rnode command found"
        rnode --version || true
    fi

    return 0
}

# ==============================================================================
# Test Functions
# ==============================================================================

test_openai() {
    print_header "Testing OpenAI Integration"

    # Enable OpenAI for this test
    export OPENAI_ENABLED=true

    print_info "Running openai-example.rho..."
    echo ""

    if rnode eval "${SCRIPT_DIR}/openai-example.rho"; then
        print_success "OpenAI test completed successfully"
        return 0
    else
        print_error "OpenAI test failed"
        return 1
    fi
}

test_ollama() {
    print_header "Testing Ollama Integration"

    # Enable Ollama for this test
    export OLLAMA_ENABLED=true

    print_info "Running ollama-example.rho..."
    echo ""

    if rnode eval "${SCRIPT_DIR}/ollama-example.rho"; then
        print_success "Ollama test completed successfully"
        return 0
    else
        print_error "Ollama test failed"
        return 1
    fi
}

test_demo_contract() {
    print_header "Testing Demo Contract"

    # Enable Ollama for demo (can switch to OpenAI if preferred)
    export OLLAMA_ENABLED=true

    print_info "Running demo-contract.rho..."
    echo ""

    if rnode eval "${SCRIPT_DIR}/demo-contract.rho"; then
        print_success "Demo contract test completed successfully"
        return 0
    else
        print_error "Demo contract test failed"
        return 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    print_header "LLM System Processes Test Suite"
    print_info "Test mode: ${TEST_MODE}"
    echo ""

    # Check RNode
    if ! check_rnode; then
        exit 1
    fi

    echo ""

    # Run tests based on mode
    case "${TEST_MODE}" in
        openai)
            if check_openai_prereqs; then
                echo ""
                test_openai
            else
                print_error "OpenAI prerequisites not met"
                exit 1
            fi
            ;;

        ollama)
            if check_ollama_prereqs; then
                echo ""
                test_ollama
            else
                print_error "Ollama prerequisites not met"
                exit 1
            fi
            ;;

        both)
            # Test OpenAI
            echo ""
            if check_openai_prereqs; then
                echo ""
                test_openai || print_warning "OpenAI tests skipped or failed"
            else
                print_warning "OpenAI prerequisites not met, skipping"
            fi

            # Test Ollama
            echo ""
            if check_ollama_prereqs; then
                echo ""
                test_ollama || print_warning "Ollama tests skipped or failed"
            else
                print_warning "Ollama prerequisites not met, skipping"
            fi
            ;;

        demo)
            # Test demo contract (uses Ollama by default)
            echo ""
            if check_ollama_prereqs; then
                echo ""
                test_demo_contract
            else
                print_error "Ollama prerequisites not met"
                exit 1
            fi
            ;;

        *)
            print_error "Unknown test mode: ${TEST_MODE}"
            print_info "Usage: $0 [openai|ollama|both|demo]"
            exit 1
            ;;
    esac

    echo ""
    print_header "Test Suite Complete"
}

# Run main function
main
