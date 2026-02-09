# Heartbeat Block Proposer

## Overview

The **Heartbeat** is an automated block proposer that ensures blockchain liveness by creating blocks periodically when no user activity (deploys) occurs. Without heartbeat, the blockchain would stall during periods of zero traffic because finalization requires new blocks.

## Heartbeat vs Manual Proposal

| Aspect | Manual Proposal | Heartbeat |
|--------|-----------------|-----------|
| **Trigger** | Explicit `propose` API call or deploy with auto-propose | Timer (check-interval) or signal (deploy submission) |
| **User Action Required** | Yes - must call `propose` or submit deploy | No - automatic |
| **Block Content** | User deploys + system deploys | System deploys only (empty blocks) allowed |
| **Primary Use Case** | Normal transaction processing | Maintain liveness during idle periods |
| **Coordination** | User-driven | Coordinated via ProposerQueue (same as manual) |

### Flow Comparison

```
Manual Propose:
  User → Deploy API → ProposerQueue → ProposerInstance → Block Created
                   ↑
  User → Propose API ───┘

Heartbeat Propose:
  Timer (check-interval) ──┐
                           ├─→ HeartbeatProposer → ProposerQueue → ProposerInstance → Block Created
  Deploy Signal ───────────┘
```

Both flows go through the same `ProposerQueue` and `ProposerInstance` serialization, ensuring thread safety and preventing race conditions.

## When Heartbeat Proposes

The heartbeat proposes a block when **any** of these conditions are met:

1. **Pending User Deploys**: There are user deploys waiting to be included in a block
2. **Stale LFB + New Parents**: The Last Finalized Block (LFB) is older than `max-lfb-age` AND there are new blocks from other validators to reference

The heartbeat does **NOT** propose when:
- The validator is not bonded (not in active validators set)
- LFB is fresh (within `max-lfb-age`)
- No new parents exist (would violate validation - can't create block citing same parents)
- Casper is not yet initialized

## Configuration

Heartbeat is configured in the `casper.heartbeat` section of your node configuration (HOCON format):

```hocon
casper {
  heartbeat {
    # Enable/disable heartbeat block proposing
    enabled = true

    # How often to check if heartbeat is needed
    check-interval = 30 seconds

    # Maximum age of LFB before triggering heartbeat
    # If no block is finalized in this duration, propose an empty block
    max-lfb-age = 60 seconds
  }
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Master switch for heartbeat functionality |
| `check-interval` | `30 seconds` | How often the heartbeat wakes up to check conditions |
| `max-lfb-age` | `60 seconds` | If LFB is older than this, and new parents exist, propose a block |

### Default Configuration Locations

- **defaults.conf**: `/node/src/main/resources/defaults.conf` - default values
- **standalone-dev.conf**: `/scripts/standalone-dev.conf` - development preset
- **Docker configs**: `/docker/conf/*.conf` - various deployment presets

## CLI Usage

Heartbeat is configured via configuration files, not CLI flags. To enable heartbeat:

### Method 1: Override in Configuration File

Create or modify your node configuration file:

```hocon
# my-node.conf
include "defaults"

casper {
  heartbeat {
    enabled = true
    check-interval = 30 seconds
    max-lfb-age = 60 seconds
  }
}
```

Then start the node with:

```bash
./rnode run --config-file my-node.conf
```

### Method 2: Use Preset Configuration

For development, use the standalone-dev preset which has heartbeat enabled:

```bash
./rnode run --config-file scripts/standalone-dev.conf
```

### Verifying Heartbeat is Running

Check the node logs for heartbeat messages:

```bash
# Heartbeat starting
grep "Heartbeat: Starting" rnode.log

# Heartbeat proposing blocks
grep "Heartbeat: Proposing" rnode.log

# Heartbeat checks (debug level)
grep "Heartbeat:" rnode.log
```

## Singleton Test Cluster (1 Validator + 1 Observer)

> [!WARNING]
> Heartbeat has a **critical constraint**: it requires `max-number-of-parents > 1` for multi-validator networks. However, for a **singleton cluster with only 1 validator**, you can use `max-number-of-parents = 1`.

### Why This Works for Singletons

In a singleton cluster:
- Only one validator exists, so there are no competing blocks from other validators
- Empty heartbeat blocks won't fail `InvalidParents` validation because there are no "newer blocks" from others
- The validator's own blocks form a linear chain

### Singleton Validator Configuration

Create `validator.conf`:

```hocon
# Standalone mode - creates genesis automatically
standalone = true
autopropose = false

protocol-server {
  network-id = "singleton-test"
  allow-private-addresses = true
  port = 40400
}

casper {
  # Validator key (required for proposing)
  validator-private-key-path = /path/to/validator.key
  
  # Single parent is OK for singleton
  max-number-of-parents = 1
  
  # Enable heartbeat for liveness
  heartbeat {
    enabled = true
    check-interval = 10 seconds
    max-lfb-age = 20 seconds
  }
  
  genesis-ceremony {
    required-signatures = 0
    ceremony-master-mode = true
  }
  
  genesis-block-data {
    number-of-active-validators = 1
  }
}
```

Start the validator:

```bash
./rnode run --config-file validator.conf --data-dir /data/validator
```

### Observer Node Configuration

Create `observer.conf`:

```hocon
standalone = false

protocol-server {
  network-id = "singleton-test"
  allow-private-addresses = true
  port = 40410
}

protocol-client {
  # Bootstrap from validator
  bootstrap = "rnode://<validator-node-id>@localhost?protocol=40400&discovery=40404"
}

api-server {
  port-grpc-external = 40411
  port-grpc-internal = 40412
  port-http = 40413
  port-admin-http = 40415
}

peers-discovery {
  port = 40414
}

casper {
  # No validator key = read-only observer
  # validator-private-key-path = <not set>
  
  # Must match validator's settings
  max-number-of-parents = 1
  
  # Observers don't propose, so heartbeat is disabled
  heartbeat {
    enabled = false
  }
}
```

Start the observer:

```bash
./rnode run --config-file observer.conf --data-dir /data/observer
```

### Verification Steps

1. **Start Validator**: Should create genesis and start heartbeat loop
2. **Check Heartbeat Logs**:
   ```bash
   grep "Heartbeat:" /data/validator/rnode.log
   ```
   Expected output:
   ```
   Heartbeat: Starting with random initial delay of Xs (check interval: 10s, max LFB age: 20s)
   Heartbeat: Woke from timer
   Heartbeat: Proposing block - reason: LFB is stale...
   Heartbeat: Successfully created block
   ```

3. **Start Observer**: Should sync with validator
4. **Query Both Nodes**: Blocks should be visible on both:
   ```bash
   # On validator
   ./rnode show-blocks --grpc-port 40402
   
   # On observer
   ./rnode show-blocks --grpc-port 40412
   ```

## Multi-Validator Networks

> [!CAUTION]
> For multi-validator networks, you **MUST** set `max-number-of-parents` to at least 3× your shard size to avoid heartbeat block validation failures.

Example for a 3-validator network:

```hocon
casper {
  # 3 validators × 3 = 9 minimum
  max-number-of-parents = 9
  
  heartbeat {
    enabled = true
    check-interval = 30 seconds
    max-lfb-age = 60 seconds
  }
}
```

## Troubleshooting

### Heartbeat Not Proposing

**Check 1: Is heartbeat enabled?**
```bash
grep "Heartbeat: Starting" rnode.log
```
If no output, heartbeat is disabled in config.

**Check 2: Is validator bonded?**
```bash
grep "Heartbeat: Validator is not bonded" rnode.log
```
If found, the validator is not in the active validators set.

**Check 3: Is Casper available?**
```bash
grep "Heartbeat: Casper not available" rnode.log
```
If found, node is still initializing.

### Configuration Error: max-number-of-parents=1

If you see:
```
CONFIGURATION ERROR: Heartbeat incompatible with max-number-of-parents=1
The heartbeat thread is now DISABLED.
```

This means you have multiple validators but `max-number-of-parents = 1`. Either:
- Increase `max-number-of-parents` to 3× your shard size
- Or if truly a singleton, this error shouldn't appear

### Blocks Not Finalizing

If blocks are created but not finalized:
- Check `fault-tolerance-threshold` (should be 0.0 for testing)
- Ensure all validators have heartbeat enabled with similar timing
- Check network connectivity between validators

## Implementation Details

For developers, the heartbeat implementation consists of:

| File | Purpose |
|------|---------|
| [HeartbeatProposer.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly-scala/node/src/main/scala/coop/rchain/node/instances/HeartbeatProposer.scala) | Main heartbeat logic and stream |
| [HeartbeatSignal.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly-scala/casper/src/main/scala/coop/rchain/casper/HeartbeatSignal.scala) | Signal trait for external wake triggers |
| [CasperConf.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly-scala/casper/src/main/scala/coop/rchain/casper/CasperConf.scala) | HeartbeatConf case class |
| [Setup.scala](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly-scala/node/src/main/scala/coop/rchain/node/runtime/Setup.scala#L421-L441) | Heartbeat initialization and wiring |
| [defaults.conf](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly-scala/node/src/main/resources/defaults.conf#L350-L361) | Default configuration values |

### Key Algorithms

1. **Dual Trigger Mechanism**: Uses `fs2.Stream.merge` to race between timer wake and signal wake
2. **Random Initial Delay**: Prevents lock-step behavior between validators
3. **LFB Age Check**: Only proposes when finalization is stalled
4. **New Parents Check**: Uses DAG traversal to find blocks not in ancestor set
5. **ProposerQueue Integration**: Serializes with all other propose sources via same queue
