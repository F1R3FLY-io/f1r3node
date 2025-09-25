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
docker compose -f docker/shard-with-autopropose.yml -f docker/shard-monitoring.yml up -d
```

This will:
- Enable Prometheus scraping of `boot`, `validator1`, `validator2`, `validator3` at `http://<node>:40403/metrics`
- Start Prometheus on localhost:9090
- Start Grafana on localhost:3000 with a pre-provisioned Prometheus datasource

3. Access UIs
```bash
open http://localhost:9090   # Prometheus
open http://localhost:3000   # Grafana (default user: admin / password: admin)
```

Note: Grafana default credentials are `admin` / `admin`. You may be prompted to change the password on first login.


## Generate and import a Grafana dashboard

1. Generate dashboard JSON from a node's metrics endpoint (pick any node):
```bash
# Example: bootstrap node exposes 40403 on localhost
../scripts/rnode-metric-counters-to-grafana-dash.sh http://127.0.0.1:40403/metrics > ../target/grafana.json
```

2. Import into Grafana:
   - Open http://localhost:3000
   - Left sidebar: “+” → “Import”
   - Click “Upload JSON file” and select `../target/grafana.json`
   - Ensure the Prometheus datasource is set to `Prometheus`
   - Click “Import”

## Uninstall

Docker Compose:
```sh
docker compose -f docker/shard-monitoring.yml down
```