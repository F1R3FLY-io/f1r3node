# Standalone Node Setup Checklist

Complete walkthrough for setting up a single-validator F1R3FLY node with correct bonding.

---

## Prerequisites

- [ ] Built RNode assembly: `node/target/scala-2.12/rnode-assembly-*.jar`
- [ ] Nix environment active (for Ammonite REPL)
- [ ] Ports 40401 (gRPC external), 40402 (gRPC internal) available

---

## Step 1: Generate Validator Keys

- [ ] Start Ammonite REPL:
  ```bash
  amm
  ```

- [ ] Generate keypair:
  ```scala
  @ import $exec.scripts.playground
  @ val (privateKeyHex, publicKeyHex) = ECPrivateKey.freshPrivateKey
  @ println(s"Private Key: $privateKeyHex")
  @ println(s"Public Key: $publicKeyHex")
  @ exit
  ```

- [ ] **Save keys securely** (write them down now!):
  ```
  Private Key: _____________________________________________________
  Public Key: _______________________________________________________
  ```

⚠️ **CRITICAL**: These keys must match in all following steps!

---

## Step 2: Prepare Data Directory

- [ ] Set your data directory location:
  ```bash
  DATA_DIR=~/.rnode-standalone
  ```

- [ ] Clean previous data (if starting fresh):
  ```bash
  rm -rf $DATA_DIR
  ```

- [ ] Create directory structure:
  ```bash
  mkdir -p $DATA_DIR/genesis
  ```

- [ ] Set key variables (paste YOUR keys from Step 1):
  ```bash
  PRIVATE_KEY="your-private-key-hex-from-step-1"
  PUBLIC_KEY="your-public-key-hex-from-step-1"
  ```

---

## Step 3: Create Genesis Configuration

### bonds.txt

- [ ] Create bonds file with YOUR public key:
  ```bash
  echo "$PUBLIC_KEY 100" > $DATA_DIR/genesis/bonds.txt
  ```

- [ ] Verify it was created correctly:
  ```bash
  cat $DATA_DIR/genesis/bonds.txt
  ```

  **Expected**: Single line with your 130-char public key + space + `100`

### wallets.txt (Optional)

- [ ] Create wallets file for initial REV tokens:
  ```bash
  echo "secp256k1 $PUBLIC_KEY 1000000000" > $DATA_DIR/genesis/wallets.txt
  ```

- [ ] Verify:
  ```bash
  cat $DATA_DIR/genesis/wallets.txt
  ```

---

## Step 4: Save Private Key Securely

### Option A: For testing (command-line)

- [ ] Keep in shell variable:
  ```bash
  echo "Using PRIVATE_KEY=$PRIVATE_KEY"
  ```

### Option B: For production (file-based)

- [ ] Save to file with restricted permissions:
  ```bash
  echo "$PRIVATE_KEY" > $DATA_DIR/validator-key.hex
  chmod 600 $DATA_DIR/validator-key.hex
  ```

- [ ] Verify permissions:
  ```bash
  ls -la $DATA_DIR/validator-key.hex
  ```

  **Expected**: `-rw-------` (600 permissions)

---

## Step 5: Choose Startup Method

### Option A: Command-line (Quick Start)

- [ ] Run with command-line flags:
  ```bash
  java -jar node/target/scala-2.12/rnode-assembly-*.jar run \
    -s \
    --validator-private-key $PRIVATE_KEY \
    --synchrony-constraint-threshold 0.0 \
    --allow-private-addresses \
    --no-upnp \
    --data-dir $DATA_DIR
  ```

### Option B: Config File (Recommended)

- [ ] Create config file `$DATA_DIR/rnode.conf`:
  ```hocon
  standalone = true
  autopropose = false

  protocol-server {
    allow-private-addresses = true
    no-upnp = true
    network-id = "testnet"
  }

  casper {
    validator-private-key-path = ${storage.data-dir}/validator-key.hex

    synchrony-constraint-threshold = 0.0
    fault-tolerance-threshold = 0.0

    genesis-ceremony {
      required-signatures = 0
      ceremony-master-mode = true
    }

    genesis-block-data {
      genesis-data-dir = ${storage.data-dir}/genesis
      bonds-file = ${storage.data-dir}/genesis/bonds.txt
      wallets-file = ${storage.data-dir}/genesis/wallets.txt
    }
  }

  storage {
    data-dir = "YOUR_FULL_PATH_TO_DATA_DIR"
  }
  ```

- [ ] Update `storage.data-dir` with your actual path

- [ ] Run with config file:
  ```bash
  java -jar node/target/scala-2.12/rnode-assembly-*.jar run \
    --config-file $DATA_DIR/rnode.conf
  ```

---

## Step 6: Verify Genesis Ceremony

- [ ] Watch logs for successful genesis:
  ```bash
  tail -f $DATA_DIR/rnode.log | grep -E "ApprovedBlock|Running state"
  ```

- [ ] **Expected log output**:
  ```
  [INFO] Starting execution of ApprovedBlockProtocol. Waiting for 0 approvals...
  [INFO] Self-approving genesis block.
  [INFO] Broadcasting ApprovedBlock...
  [INFO] Finished execution of ApprovedBlockProtocol
  [INFO] Transition to Running state
  ```

- [ ] Check active validators (should show YOUR public key):
  ```bash
  grep "ACTIVE VALIDATORS" $DATA_DIR/rnode.log
  ```

---

## Step 7: Verify Node is Bonded

- [ ] Check bond status via gRPC:
  ```bash
  grpcurl -plaintext -d '{}' localhost:40401 casper.DeployService/lastFinalizedBlock
  ```

- [ ] **Expected**: Should return genesis block (not an error)

- [ ] Verify your validator is in active set:
  ```bash
  tail -100 $DATA_DIR/rnode.log | grep "ACTIVE VALIDATORS" -A 5
  ```

- [ ] **Confirm**: Your public key appears in the list

---

## Step 8: Create Your First Block

- [ ] Propose a block:
  ```bash
  grpcurl -plaintext -d '{}' localhost:40402 casper.ProposeService/propose
  ```

- [ ] **Expected success response**:
  ```json
  {
    "result": "Success! Block abc123... created and added."
  }
  ```

- [ ] If you see **"NotBonded" error** → Go to [Troubleshooting](#troubleshooting-checklist)

---

## Step 9: Query Your Blockchain

- [ ] Show main chain:
  ```bash
  grpcurl -plaintext -d '{"depth": 10}' \
    localhost:40401 casper.DeployService/showMainChain
  ```

- [ ] Get all blocks:
  ```bash
  grpcurl -plaintext -d '{"depth": 10}' \
    localhost:40401 casper.DeployService/getBlocks
  ```

- [ ] Get last finalized block:
  ```bash
  grpcurl -plaintext \
    localhost:40401 casper.DeployService/lastFinalizedBlock
  ```

- [ ] **Verify**: Block count increases after each proposal

---

## Step 10: Enable Auto-Propose (Optional)

- [ ] Stop the node (Ctrl+C)

- [ ] Edit config file, change:
  ```hocon
  autopropose = true
  ```

- [ ] Restart node

- [ ] **Result**: Blocks automatically created when deploys submitted

---

## Troubleshooting Checklist

### Problem: "NotBonded" error when proposing

- [ ] Check what's in bonds.txt:
  ```bash
  cat $DATA_DIR/genesis/bonds.txt
  ```

- [ ] Derive public key from your private key:
  ```bash
  amm -c "import \$exec.scripts.playground; println(ECPrivateKey.toPublic(\"$PRIVATE_KEY\"))"
  ```

- [ ] **Verify**: Both public keys match EXACTLY (130 chars, starts with `04`)

- [ ] If they don't match:
  - [ ] You're using the wrong private key
  - [ ] Nuclear option: Delete $DATA_DIR and start from Step 2 with correct keys

### Problem: Genesis ceremony hangs

- [ ] Check if `standalone = true` OR `required-signatures = 0`:
  ```bash
  grep -E "standalone|required-signatures" $DATA_DIR/rnode.conf
  ```

- [ ] Check logs for errors:
  ```bash
  tail -100 $DATA_DIR/rnode.log
  ```

### Problem: Can't connect to gRPC API

- [ ] Verify ports are listening:
  ```bash
  netstat -an | grep -E "40401|40402"
  ```

- [ ] Check if node is running:
  ```bash
  ps aux | grep rnode
  ```

- [ ] Try localhost vs 127.0.0.1:
  ```bash
  grpcurl -plaintext localhost:40401 list
  grpcurl -plaintext 127.0.0.1:40401 list
  ```

### Problem: Synchrony constraint errors in logs

- [ ] Verify synchrony-constraint-threshold is 0.0:
  ```bash
  grep "synchrony-constraint-threshold" $DATA_DIR/rnode.conf
  ```

- [ ] Expected: `synchrony-constraint-threshold = 0.0`

### Nuclear Option: Complete Reset

- [ ] Stop the node:
  ```bash
  pkill -f rnode
  ```

- [ ] Delete all data:
  ```bash
  rm -rf $DATA_DIR
  ```

- [ ] Start over from Step 2

---

## Quick Reference Card

Save these for future sessions:

```bash
# ==========================================
# YOUR NODE INFO (fill in once)
# ==========================================
PRIVATE_KEY="your-private-key-hex"
PUBLIC_KEY="your-public-key-hex"
DATA_DIR=~/.rnode-standalone

# ==========================================
# START NODE (command-line method)
# ==========================================
java -jar node/target/scala-2.12/rnode-assembly-*.jar run \
  -s \
  --validator-private-key $PRIVATE_KEY \
  --synchrony-constraint-threshold 0.0 \
  --allow-private-addresses \
  --no-upnp \
  --data-dir $DATA_DIR

# ==========================================
# COMMON COMMANDS
# ==========================================

# Propose a block
grpcurl -plaintext -d '{}' localhost:40402 casper.ProposeService/propose

# Show main chain (last 10 blocks)
grpcurl -plaintext -d '{"depth": 10}' localhost:40401 casper.DeployService/showMainChain

# Get all blocks
grpcurl -plaintext -d '{"depth": 10}' localhost:40401 casper.DeployService/getBlocks

# Last finalized block
grpcurl -plaintext localhost:40401 casper.DeployService/lastFinalizedBlock

# Watch logs
tail -f $DATA_DIR/rnode.log

# Stop node
pkill -f rnode
```

---

## Key Concepts

- **Standalone mode** (`-s`): Self-approving genesis, no network peers required
- **Synchrony threshold = 0.0**: Single validator doesn't need to see other validators
- **Main chain**: Linear path through highest-scored blocks (fork choice winner)
- **bonds.txt**: Immutable at genesis—defines initial validator set
- **Public key format**: 130 hex chars (65 bytes), starts with `04` (uncompressed secp256k1)

---

## Security Notes for Production

- [ ] **Never** use `--validator-private-key` CLI flag (visible in process table)
- [ ] **Always** use `validator-private-key-path` with 600 permissions
- [ ] Consider encrypted PEM format with `RNODE_VALIDATOR_PASSWORD` env var
- [ ] Restrict gRPC internal port (40402) to localhost only
- [ ] Use firewall to limit external API access
- [ ] Back up your private key securely (offline storage)
- [ ] Monitor logs for unauthorized proposal attempts

---

## Success Criteria

✅ Your node is working correctly when:

- [ ] Genesis ceremony completes without errors
- [ ] Your public key appears in ACTIVE VALIDATORS logs
- [ ] `grpcurl propose` returns "Success! Block ... created"
- [ ] Block count increases with each proposal
- [ ] `showMainChain` returns your blocks
- [ ] No "NotBonded" errors in logs

---

## Next Steps

Once your standalone node is running:

1. Deploy Rholang contracts
2. Query RSpace state
3. Set up monitoring/metrics
4. Connect a second node to form a multi-validator network
5. Explore sharding configuration

See `docs/` for advanced topics.
