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

## Monitoring Health

Use the dashboard to monitor:

1. **Block Processing Performance**: Watch for replay time spikes indicating cold cache or other issues
2. **Network Health**: Check block request/retry rates for sync problems
3. **Validation Success**: Monitor success rate for consensus issues
4. **System Load**: Track block processing rates and queue depths

Alert thresholds can be configured in Prometheus based on the recording rules.

## Uninstall

Docker Compose:
```sh
docker compose -f docker/shard-monitoring.yml down
```