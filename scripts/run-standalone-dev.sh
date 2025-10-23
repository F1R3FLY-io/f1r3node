#!/bin/bash

set -e

# F1r3fly Standalone Development Node Runner
# This script runs the F1r3node directly with Java for fast development iteration
# Equivalent to the Docker standalone setup but much faster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
DATA_DIR="$PROJECT_ROOT/data/standalone-dev"
GENESIS_DIR="$DATA_DIR/genesis"
CONFIG_FILE="$SCRIPT_DIR/standalone-dev.conf"

# Private key for the standalone validator (Bootstrap node key)
VALIDATOR_PRIVATE_KEY="5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657"
VALIDATOR_PUBLIC_KEY="04ffc016579a68050d655d55df4e09f04605164543e257c8e6df10361e6068a5336588e9b355ea859c5ab4285a5ef0efdf62bc28b80320ce99e26bb1607b3ad93d"

# REV address for the validator
VALIDATOR_REV_ADDRESS="1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g"

echo "🚀 F1r3fly Standalone Development Node"
echo "======================================"

# Create data directory structure
echo "📁 Setting up data directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$GENESIS_DIR"
mkdir -p "$DATA_DIR/rgb_storage"

# Create bonds.txt file
echo "📝 Creating bonds.txt..."
cat > "$GENESIS_DIR/bonds.txt" << EOF
$VALIDATOR_PUBLIC_KEY 1000
EOF

# Create wallets.txt file
echo "📝 Creating wallets.txt..."
cat > "$GENESIS_DIR/wallets.txt" << EOF
$VALIDATOR_REV_ADDRESS,1000000000000000
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA,100000000000000
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH,100000000000000
1111LAd2PWaHsw84gxarNx99YVK2aZhCThhrPsWTV7cs1BPcvHftP,100000000000000
EOF

# Check configuration file
echo "📝 Using existing configuration file..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Please ensure the standalone-dev.conf file exists in the scripts directory"
    exit 1
fi
echo "✅ Configuration file found: $CONFIG_FILE"

echo "✅ Setup complete!"
echo ""
echo "🔧 Configuration:"
echo "   Data Directory: $DATA_DIR"
echo "   Config File: $CONFIG_FILE"
echo "   Validator Private Key: $VALIDATOR_PRIVATE_KEY"
echo "   Validator REV Address: $VALIDATOR_REV_ADDRESS"
echo ""
echo "🌐 API Endpoints (once running):"
echo "   HTTP API: http://localhost:40403"
echo "   External gRPC: localhost:40401"
echo "   Internal gRPC: localhost:40402"
echo "   Admin HTTP: http://localhost:40405"
echo ""
echo "🚀 Starting F1r3fly standalone node..."
echo ""

# Check if rnode executable exists
RNODE_EXECUTABLE="$PROJECT_ROOT/node/target/universal/stage/bin/rnode"
if [ ! -f "$RNODE_EXECUTABLE" ]; then
    echo "❌ rnode executable not found: $RNODE_EXECUTABLE"
    echo ""
    echo "Please build the project first by running:"
    echo "  sbt \";compile ;stage\""
    exit 1
fi

# Change to project root and run the node with your original command parameters
cd "$PROJECT_ROOT"
exec "$RNODE_EXECUTABLE" run \
    -s \
    --no-upnp \
    --allow-private-addresses \
    --synchrony-constraint-threshold=0.0 \
    --validator-private-key="$VALIDATOR_PRIVATE_KEY" \
    --data-dir="$DATA_DIR" \
    -c "$CONFIG_FILE"
