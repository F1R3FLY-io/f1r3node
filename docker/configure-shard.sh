#!/usr/bin/env bash

# ============================================================================
# F1R3FLY Genesis Key Generation Script
# ============================================================================
# This script generates all cryptographic keys needed for a F1R3FLY shard:
# - Validator keys (bootstrap + validators 1-3)
# - User wallet keys (sample wallets)
# - Genesis files (bonds.txt, wallets.txt)
# - Updated .env file with generated keys
# ============================================================================

set -e  # Exit on error

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_CLI="$HOME/Pyrofex/firefly/rust-client/target/debug/node_cli"
KEYS_DIR="$SCRIPT_DIR/keys"
CERTS_DIR="$SCRIPT_DIR/certs"

# Validator configuration
VALIDATOR_STAKE=1000
INITIAL_WALLET_BALANCE=50000000000000000  # 50000000000000000 dust (50 million REV)

# Check if node_cli exists
if [[ ! -x "$NODE_CLI" ]]; then
    echo "ERROR: node_cli not found at $NODE_CLI"
    echo "Please build rust-client first: cd ~/Pyrofex/firefly/rust-client && cargo build"
    exit 1
fi

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

print_section() {
    echo ""
    echo "=========================================================================="
    echo "$1"
    echo "=========================================================================="
    echo ""
}

print_step() {
    echo ""
    echo "--- $1 ---"
    echo ""
}

# ----------------------------------------------------------------------------
# Setup Directories
# ----------------------------------------------------------------------------

print_section "Setting up directory structure"

mkdir -p "$KEYS_DIR"/{bootstrap,validator1,validator2,validator3,wallets}
echo "Created directory structure in $KEYS_DIR"

# ----------------------------------------------------------------------------
# Generate Validator Keys
# ----------------------------------------------------------------------------

print_section "Generating Validator Keys"

# Array of validators
validators=("bootstrap" "validator1" "validator2" "validator3")

for validator in "${validators[@]}"; do
    print_step "Generating keys for $validator"

    output_dir="$KEYS_DIR/$validator"

    # Generate key pair
    echo "Running: $NODE_CLI generate-key-pair --save --output-dir $output_dir"
    $NODE_CLI generate-key-pair --save --output-dir "$output_dir"

    # Read the generated keys
    private_key=$(cat "$output_dir/private_key.hex")
    public_key=$(cat "$output_dir/public_key.hex")

    # Generate REV address from public key
    echo ""
    echo "Generating REV address for $validator..."
    # Extract only the REV address (after "🔑 REV address: ")
    rev_address=$($NODE_CLI generate-rev-address --public-key "$public_key" | grep "REV address:" | sed 's/.*REV address: //')

    if [[ -z "$rev_address" ]]; then
        echo "ERROR: Failed to extract REV address"
        exit 1
    fi

    # Save just the REV address
    echo "$rev_address" > "$output_dir/rev_address.txt"

    echo ""
    echo "✓ Generated $validator keys:"
    echo "  Private Key: $private_key"
    echo "  Public Key:  $public_key"
    echo "  REV Address: $rev_address"
done

# ----------------------------------------------------------------------------
# Generate TLS Certificates
# ----------------------------------------------------------------------------

print_section "Generating TLS Certificates"

# Array of nodes that need TLS certificates
tls_nodes=("bootstrap" "validator1" "validator2" "validator3")

for node in "${tls_nodes[@]}"; do
    print_step "Generating TLS certificate for $node"

    node_dir="$KEYS_DIR/$node"

    # Generate secp256r1 private key in EC format
    echo "Generating secp256r1 private key..."
    openssl ecparam -name secp256r1 -genkey -noout -out "$node_dir/node.key.ec.pem"

    # Convert to PKCS#8 format (required by Netty SSL)
    echo "Converting to PKCS#8 format..."
    openssl pkcs8 -topk8 -nocrypt -in "$node_dir/node.key.ec.pem" -out "$node_dir/node.key.pem"
    rm "$node_dir/node.key.ec.pem"

    # Compute node ID from the private key BEFORE generating certificate
    echo "Computing node ID from TLS private key..."
    node_id=$($NODE_CLI get-node-id -k "$node_dir/node.key.pem" | grep -oE '[a-f0-9]{40}$' | tail -1)

    if [[ -z "$node_id" ]]; then
        echo "ERROR: Failed to compute node ID from private key"
        exit 1
    fi

    echo "  Node ID: $node_id"

    # Save node ID to file
    echo "$node_id" > "$node_dir/node_id.txt"

    # Generate self-signed certificate with node ID as CN (valid 10 years)
    echo "Generating self-signed X.509 certificate with CN=$node_id..."
    openssl req -new -x509 -key "$node_dir/node.key.pem" \
        -out "$node_dir/node.certificate.pem" \
        -days 3650 -subj "/CN=$node_id"

    echo ""
    echo "✓ Generated TLS certificate for $node"
    echo "  Key:         $node_dir/node.key.pem"
    echo "  Certificate: $node_dir/node.certificate.pem"
    echo "  Node ID:     $node_id"
    echo "  CN:          $node_id"
done

# Store bootstrap node ID for convenience
bootstrap_node_id=$(cat "$KEYS_DIR/bootstrap/node_id.txt")

# ----------------------------------------------------------------------------
# Generate User Wallets
# ----------------------------------------------------------------------------

print_section "Generating User Wallets"

num_wallets=3

for i in $(seq 1 $num_wallets); do
    print_step "Generating user wallet $i"

    wallet_dir="$KEYS_DIR/wallets/user-wallet-$i"
    mkdir -p "$wallet_dir"

    echo "Running: $NODE_CLI generate-key-pair --save --output-dir $wallet_dir"
    $NODE_CLI generate-key-pair --save --output-dir "$wallet_dir"

    private_key=$(cat "$wallet_dir/private_key.hex")
    public_key=$(cat "$wallet_dir/public_key.hex")

    echo ""
    echo "Generating REV address for user wallet $i..."
    # Extract only the REV address (after "🔑 REV address: ")
    rev_address=$($NODE_CLI generate-rev-address --public-key "$public_key" | grep "REV address:" | sed 's/.*REV address: //')

    if [[ -z "$rev_address" ]]; then
        echo "ERROR: Failed to extract REV address for user wallet $i"
        exit 1
    fi

    # Save just the REV address
    echo "$rev_address" > "$wallet_dir/rev_address.txt"

    echo ""
    echo "✓ Generated user wallet $i:"
    echo "  Private Key: $private_key"
    echo "  REV Address: $rev_address"
done

# ----------------------------------------------------------------------------
# Generate bonds.txt
# ----------------------------------------------------------------------------

print_section "Generating bonds.txt"

bonds_file="$SCRIPT_DIR/genesis/bonds.txt"

echo "Creating $bonds_file with validator bonds..."
cat > "$bonds_file" << EOF
$(cat "$KEYS_DIR/validator1/public_key.hex") $VALIDATOR_STAKE
$(cat "$KEYS_DIR/validator2/public_key.hex") $VALIDATOR_STAKE
$(cat "$KEYS_DIR/validator3/public_key.hex") $VALIDATOR_STAKE
EOF

echo "✓ Generated bonds.txt:"
cat "$bonds_file"

# ----------------------------------------------------------------------------
# Generate wallets.txt
# ----------------------------------------------------------------------------

print_section "Generating wallets.txt"

wallets_file="$SCRIPT_DIR/genesis/wallets.txt"

echo "Creating $wallets_file with funded wallets..."

# Start with bootstrap and validators
cat > "$wallets_file" << EOF
$(cat "$KEYS_DIR/bootstrap/rev_address.txt"),$INITIAL_WALLET_BALANCE
$(cat "$KEYS_DIR/validator1/rev_address.txt"),$INITIAL_WALLET_BALANCE
$(cat "$KEYS_DIR/validator2/rev_address.txt"),$INITIAL_WALLET_BALANCE
$(cat "$KEYS_DIR/validator3/rev_address.txt"),50000000000000
EOF

# Add user wallets
for i in $(seq 1 $num_wallets); do
    wallet_dir="$KEYS_DIR/wallets/user-wallet-$i"
    rev_address=$(cat "$wallet_dir/rev_address.txt")
    echo "$rev_address,$INITIAL_WALLET_BALANCE" >> "$wallets_file"
done

echo "✓ Generated wallets.txt:"
cat "$wallets_file"


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

print_section "Key Generation Complete!"

echo "Generated keys and certificates are stored in: $KEYS_DIR"
echo ""
echo "Directory structure:"
echo "  $KEYS_DIR/"
echo "    ├── bootstrap/"
echo "    │   ├── private_key.hex      (secp256k1 blockchain private key)"
echo "    │   ├── public_key.hex       (secp256k1 blockchain public key)"
echo "    │   ├── rev_address.txt      (REV address)"
echo "    │   ├── node_id.txt          (node ID from TLS cert)"
echo "    │   ├── node.key.pem         (secp256r1 TLS private key)"
echo "    │   └── node.certificate.pem (X.509 TLS certificate)"
echo "    ├── validator1/              (same structure as bootstrap)"
echo "    ├── validator2/              (same structure as bootstrap)"
echo "    ├── validator3/              (same structure as bootstrap)"
echo "    └── wallets/"
echo "        ├── user-wallet-1/       (private_key.hex, public_key.hex, rev_address.txt)"
echo "        ├── user-wallet-2/       (private_key.hex, public_key.hex, rev_address.txt)"
echo "        └── user-wallet-3/       (private_key.hex, public_key.hex, rev_address.txt)"
echo ""
echo "Genesis files updated:"
echo "  - $bonds_file"
echo "  - $wallets_file"
echo ""
echo "Bootstrap node ID: $bootstrap_node_id"
echo ""
echo "Next steps:"
echo "  1. Review the generated keys, certificates, and genesis files"
echo "  2. Start your F1R3FLY shard:"
echo "     cd $SCRIPT_DIR"
echo "     docker-compose -f shard-with-autopropose.yml up"
echo ""
echo "To regenerate keys and certificates, simply run this script again."
echo ""
