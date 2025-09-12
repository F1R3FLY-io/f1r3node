#!/bin/bash

set -e

# F1r3fly Standalone Development Node Runner
# This script runs the RChain node directly with Java for fast development iteration
# Equivalent to the Docker standalone setup but much faster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Configuration
DATA_DIR="$PROJECT_ROOT/data/standalone-dev"
GENESIS_DIR="$DATA_DIR/genesis"
JAR_FILE="$PROJECT_ROOT/node/target/scala-2.12/rnode-assembly-1.0.0-SNAPSHOT.jar"
CONFIG_FILE="$PROJECT_ROOT/standalone-dev.conf"

# Private key for the standalone validator (Bootstrap node key)
VALIDATOR_PRIVATE_KEY="5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657"
VALIDATOR_PUBLIC_KEY="04ffc016579a68050d655d55df4e09f04605164543e257c8e6df10361e6068a5336588e9b355ea859c5ab4285a5ef0efdf62bc28b80320ce99e26bb1607b3ad93d"

# REV address for the validator
VALIDATOR_REV_ADDRESS="1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g"

echo "🚀 F1r3fly Standalone Development Node"
echo "======================================"

# Check if JAR file exists
if [ ! -f "$JAR_FILE" ]; then
    echo "❌ JAR file not found: $JAR_FILE"
    echo "Please build the project first:"
    echo "  sbt ';compile ;project node ;assembly'"
    exit 1
fi

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

# Create standalone configuration file
echo "📝 Creating standalone configuration..."
cat > "$CONFIG_FILE" << 'EOF'
# F1r3fly Standalone Development Configuration
# Optimized for fast development with instant finalization

standalone = true
autopropose = true

protocol-server {
  network-id = "standalone-dev"
  allow-private-addresses = true
  no-upnp = true
  port = 40400
  grpc-max-recv-message-size = 256K
  grpc-max-recv-stream-message-size = 256M
  max-message-consumers = 400
  disable-state-exporter = false
}

protocol-client {
  network-id = ${protocol-server.network-id}
  disable-lfs = false
  batch-max-connections = 20
  network-timeout = 5 seconds
  grpc-max-recv-message-size = ${protocol-server.grpc-max-recv-message-size}
  grpc-stream-chunk-size = 256K
}

peers-discovery {
  port = 40404
  lookup-interval = 20 seconds
  cleanup-interval = 20 minutes
  heartbeat-batch-size = 100
  init-wait-loop-interval = 1 seconds
}

api-server {
  host = "0.0.0.0"
  port-grpc-external = 40401
  port-grpc-internal = 40402
  grpc-max-recv-message-size = 16M
  port-http = 40403
  port-admin-http = 40405
  max-blocks-limit = 50
  enable-reporting = false
  keep-alive-time = 2 hours
  keep-alive-timeout = 20 seconds
  permit-keep-alive-time = 5 minutes
  max-connection-idle = 1 hours
  max-connection-age = 1 hours
  max-connection-age-grace = 1 hours
}

storage {
  data-dir = ${default-data-dir}
}

tls {
  certificate-path = ${storage.data-dir}/node.certificate.pem
  key-path = ${storage.data-dir}/node.key.pem
  secure-random-non-blocking = false
  custom-certificate-location = false
  custom-key-location = false
}

casper {
  # Instant finalization for development
  fault-tolerance-threshold = 0.0
  synchrony-constraint-threshold = 0.0
  
  shard-name = root
  parent-shard-id = /
  
  # Fast intervals for development
  casper-loop-interval = 5 seconds
  requested-blocks-timeout = 60 seconds
  finalization-rate = 1
  max-number-of-parents = 1
  max-parent-depth = 2147483647
  fork-choice-stale-threshold = 30 seconds
  fork-choice-check-if-stale-interval = 1 minutes
  height-constraint-threshold = 1000

  round-robin-dispatcher {
    max-peer-queue-size = 100
    give-up-after-skipped = 0
    drop-peer-after-retries = 0
  }

  genesis-block-data {
    genesis-data-dir = ${storage.data-dir}/genesis
    bonds-file = ${casper.genesis-block-data.genesis-data-dir}/bonds.txt
    wallets-file = ${casper.genesis-block-data.genesis-data-dir}/wallets.txt
    bond-minimum = 1
    bond-maximum = 9223372036854775807
    epoch-length = 10
    quarantine-length = 10
    number-of-active-validators = 1
    genesis-block-number = 0
    pos-multi-sig-public-keys = [
      04ffc016579a68050d655d55df4e09f04605164543e257c8e6df10361e6068a5336588e9b355ea859c5ab4285a5ef0efdf62bc28b80320ce99e26bb1607b3ad93d
    ]
    pos-multi-sig-quorum = 1
  }

  genesis-ceremony {
    # Instant genesis approval for standalone development
    required-signatures = 0
    approve-interval = 1 seconds
    approve-duration = 1 seconds
    autogen-shard-size = 1
    genesis-validator-mode = false
    ceremony-master-mode = true
  }

  min-phlo-price = 1

  # Disable Bitcoin anchoring for development
  bitcoin-anchor {
    enabled = false
    network = "regtest"
    distributed = false
  }
}

# Disable metrics for development
metrics {
  prometheus = false
  influxdb = false
  influxdb-udp = false
  zipkin = false
  sigar = false
}

# Enable development mode
dev-mode = true

dev {
  # Pre-configured deployer private key for easy contract deployment
  deployer-private-key = "5f668a7ee96d944a4494cc947e4005e172d7ab3461ee5538f1f2a45a835e9657"
}
EOF

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

# Set environment variables for RGB integration
export RGB_ENABLED=true
export RGB_STORAGE_PATH="$DATA_DIR/rgb_storage"
export RGB_BITCOIN_NETWORK=testnet

# Run the node with your original command parameters
exec java \
    -Djna.library.path=./rust_libraries/release \
    -Xmx4g \
    -Xms1g \
    --add-opens java.base/sun.security.util=ALL-UNNAMED \
    --add-opens java.base/java.nio=ALL-UNNAMED \
    --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
    -jar "$JAR_FILE" \
    run \
    -s \
    --no-upnp \
    --allow-private-addresses \
    --synchrony-constraint-threshold=0.0 \
    --validator-private-key="$VALIDATOR_PRIVATE_KEY" \
    --data-dir="$DATA_DIR" \
    --config-file="$CONFIG_FILE"
EOF
