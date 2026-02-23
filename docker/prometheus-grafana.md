# Installing Grafana and Prometheus

## Prerequisites

- A running F1r3fly cluster

## Installation (Docker Compose extension for shard)

If you are running the shard with `docker/shard-with-autopropose.yml`, you can bring up Prometheus and Grafana on the same network and have Prometheus scrape all validators automatically.

1. Start/ensure your shard is running
```bash
docker compose -f docker/shard-with-autopropose.yml up -d
```

2. Start monitoring stack (uses the same Docker network as the shard)
```bash
docker compose -f docker/shard-monitoring.yml up -d
```

This will:
- Enable Prometheus scraping of `boot`, `validator1`, `validator2`, `validator3`, `readonly` at `http://<node>:40403/metrics`
- Start Prometheus on localhost:9090 with pre-configured recording rules
- Start Grafana on localhost:3000 with:
  - Pre-provisioned Prometheus datasource
  - Pre-loaded "Block Transfer Performance" dashboard

3. Access UIs
```bash
open http://localhost:9090   # Prometheus
open http://localhost:3000   # Grafana (default user: admin / password: admin)
```

Note: Grafana default credentials are `admin` / `admin`. You may be prompted to change the password on first login.

## Rust Metrics Status

The Rust node emits all block processing, transport, and system metrics via the `metrics` crate.
Prometheus metric names follow the pattern `<metric_name>{source="<source_label>"}` — the source
is a label, not part of the metric name.

| Metric Category | Rust metric name (example) | Source label |
|-----------------|---------------------------|--------------|
| Block retrieval | `block_download_end_to_end_time_bucket` | `f1r3fly.casper.block-retriever` |
| Block processing | `block_validation_time_bucket` | `f1r3fly.casper.block-processor` |
| Block replay | `block_processing_stage_replay_time_bucket` | `f1r3fly.casper` |
| Transport | `send_time_bucket`, `packets_received_total` | `f1r3fly.comm.rp.transport` |
| RSpace | `comm_consume_time_seconds_bucket` | `f1r3fly.rspace` |
| Discovery | `peers` | `f1r3fly.comm.discovery.kademlia` |
| System | `process_memory_rss_bytes`, `system_cpu_usage_percent` | `f1r3fly.system` |

> **Note**: All recording rules in `docker/monitoring/prometheus-rules.yml` use Rust-style
> label selectors (`metric_name{source="f1r3fly.category"}`). Kamon-style flat names are not used.


## Pre-Configured Dashboards

The monitoring stack includes a pre-provisioned "Block Transfer Performance" dashboard with:

### Metrics Tracked
- **Block Download Time (End-to-End)**: Total time from hash receipt to block stored
- **Block Validation Time**: Time spent validating blocks
- **Block Processing Stage Metrics** (fine-grained):
  - **Replay Stage**: Rholang execution time
  - **Validation Setup Stage**: CasperSnapshot creation time
  - **Storage Stage**: BlockStore.put() time
- **Block Size Distribution**: Average and p95 block sizes
- **Block Transfer Rate**: Calculated from size and download time
- **Block Request Rates**: Request and retry rates
- **Block Validation Success Rate**: Percentage of successful validations
- **Block Message Rates**: Hash broadcasts and block requests
- **Transport Layer Metrics**: Send times and packet handling
- **Summary Statistics**: Key metrics at a glance

### Prometheus Recording Rules

Pre-configured recording rules (30s interval, 5m window) compute:
- **Percentiles**: p50, p95, p99 for all timing metrics
- **Rates**: Blocks/sec, requests/sec, messages/sec
- **Success Rates**: Validation success percentage
- **Averages**: Block size averages

See `docker/monitoring/prometheus-rules.yml` for the complete rule definitions.

### Accessing the Dashboard

1. Open Grafana at http://localhost:3000
2. Navigate to "Dashboards" in the left sidebar
3. Select "Block Transfer Performance"

The dashboard auto-refreshes every 10 seconds and shows the last 1 hour by default.

## Manual Dashboard Import (Optional)

If you need to regenerate or customize dashboards:

1. Generate dashboard JSON from a node's metrics endpoint (pick any node):
```bash
# Example: bootstrap node exposes 40403 on localhost
../scripts/rnode-metric-counters-to-grafana-dash.sh http://127.0.0.1:40403/metrics > ../target/grafana.json
```

2. Import into Grafana:
   - Open http://localhost:3000
   - Left sidebar: "+" → "Import"
   - Click "Upload JSON file" and select `../target/grafana.json`
   - Ensure the Prometheus datasource is set to `Prometheus`
   - Click "Import"

## Performance Analysis

The block processing stage metrics are designed to isolate performance bottlenecks. Key findings:

- **Cold Cache Effect**: After node restart, expect 2-2.5x slower replay times for the first 10-20 blocks
  - Cold cache: ~800ms replay time
  - Warm cache: ~350-400ms replay time
- **Storage Performance**: BlockStore.put() is consistently fast (~6ms)
- **Validation Overhead**: CasperSnapshot creation takes 200-500ms

For detailed performance analysis, see `BLOCK_EXCHANGE_ANALYSIS.md`, Section 13.

## Phase 1 Memory Growth Analysis

Phase 1 observability (jemalloc + 16 in-memory structure gauges) was used to characterize memory growth
on a local 5-node Rust shard (bootstrap + 3 validators + observer). Three snapshots were taken at
~35, ~100, and ~147 blocks respectively with no user transactions submitted (genesis-level activity only).

### Observed growth rate

| Node type | ~35 blocks | ~100 blocks | ~147 blocks |
|-----------|-----------|------------|------------|
| Bootstrap | 360 MB | 1,116 MB | 1,570 MB |
| Validator | 509–717 MB | 1,674–1,706 MB | 2,337–2,415 MB |
| Observer | 417 MB | 1,151 MB | 1,571 MB |

**Growth rate: ~10–17 MB per block, linear and unbounded across all node types.**

### What is NOT the cause

- `rspace_history_cache_datums`, `rspace_history_cache_continuations`, `rspace_history_cache_joins` —
  all completely flat after genesis. RSpace caches are not contributing to growth.
- `rspace_hot_store_*` — stable at 56/3/3 across all snapshots (flushed to trie each block).
- `comm_stream_cache_size`, `transport_channels_count` — stable (1–4 per node).
- `casper_buffer_pending_blocks` — consistently 0 after sync.

### Confirmed growth drivers

**1. DAG retains all blocks in memory without pruning finalized blocks.**

`dag_height_map_entries` grows at ~0.66 entries/block (multiple blocks per height). Despite
`dag_finalized_blocks_total` tracking 92–95% of blocks as finalized, finalized blocks are not
removed from the in-memory DAG. At ~147 blocks: 140 finalized, 147 still in `dag_blocks_total`.
Per-block memory overhead is ~10–17 MB, far exceeding raw block size (~17 KB/block), indicating
full deserialized block data and associated state snapshots are retained per block.

**2. Block-retriever fetches nearly every block (~94% at 147 blocks).**

`casper_requested_blocks_count` reached 138 out of 147 total blocks. The block-retriever operates
as the primary block delivery mechanism rather than an exception path. Each fetched block is held
in memory through the full download + validation pipeline, increasing peak RSS.

### Known metric issues

**`process_memory_peak_bytes` is jemalloc virtual address space, not physical RAM.**
Values of ~15.1 TB are normal jemalloc behavior (arena reservation on 64-bit Linux). Use
`process_memory_rss_bytes` for actual physical memory consumption.

**`block_validation_step_*_time` histograms have a unit mismatch.**
All observations fall in the `+Inf` bucket (>10 seconds) despite total block validation averaging
~0.228 seconds. The step-level metrics appear to be recorded in nanoseconds or microseconds while
the histogram bucket boundaries are defined in seconds. Count and sum are still useful (divide sum
by count for average), but percentile analysis is broken. This is a Phase 2 fix candidate.

### Phase 2 priorities from this analysis

1. **DAG pruning**: Prune finalized blocks from the in-memory DAG structure. Only the tip set and
   a configurable finalization buffer need to remain in memory.
2. **Block gossip improvement**: `casper_requested_blocks_count` at 94% indicates validators are
   not receiving blocks proactively. Investigate why gossip propagation is not reaching peers before
   the block-retriever timeout triggers.
3. **Fix `block_validation_step_*_time` unit mismatch**: Align recorded values and histogram bucket
   boundaries to the same unit (seconds) so percentiles are meaningful.

## Monitoring Health

Use the dashboard to monitor:

1. **Memory trend**: Watch `process_memory_rss_bytes` rate of increase — should slow as DAG pruning
   is implemented. Currently grows ~10–17 MB/block without bound.
2. **DAG state**: `dag_blocks_total - dag_finalized_blocks_total` should stay small (ideally bounded
   by a pruning window). `dag_height_map_entries` should plateau after pruning is implemented.
3. **Block gossip health**: `casper_requested_blocks_count / dag_blocks_total` ratio should approach
   0 as gossip improves. Currently ~94%.
4. **Block Processing Performance**: Watch for replay time spikes indicating cold cache or other issues.
5. **Validation Success**: Monitor success rate for consensus issues.

Alert thresholds can be configured in Prometheus based on the recording rules.

## Uninstall

Docker Compose:
```sh
docker compose -f docker/shard-monitoring.yml down
```