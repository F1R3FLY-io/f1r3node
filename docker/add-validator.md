Repo - https://github.com/F1R3FLY-io/f1r3fly

Branch - `main`

1. Run initial shard: `docker compose -f docker/shard.yml up`
2. Wait 2 minutes for all current nodes to sync. Should see `Approved state for block Block #0 (875f655e81...) with empty parents (supposedly genesis) is successfully restored.` for each node.
3. In separate terminal, run read-only node: `docker compose -f docker/observer.yml up`.
4. Wait ~30 seconds till you see 4 peers in the terminal where the initial shard is running.

Repo - https://github.com/F1R3FLY-io/f1r3fly

Branch - `preston/rholang_rust` 

(The steps below to deploy/propose/check status can also be run via curl commands)

1. Run: `cd node-cli`
2. Check current bonds: `cargo run -- bonds -p 40453`. Must be run against observer node. Should see 4 bonds.
3. Deploy bond contract with validator4 private key to bootsrap node: `cargo run -- bond-validator --private-key 5ff3514bf79a7d18e8dd974c699678ba63b7762ce8d78c532346e52f0ad219cd`.
4. Deploy example contract to validator1 node: `cargo run -- deploy -f ../rholang/examples/stdout.rho -p 40412`.
5. Deploy example contract to validator2 node: `cargo run -- deploy -f ../rholang/examples/stdout.rho -p 40422`.
6. Call propose against bootstrap node: `cargo run -- propose`. This will return the block hash.
7. Wait ~20 seconds.
8. Verify block hash is finalized: `cargo run -- is-finalized -b <block_hash>`.
9. Check current bonds: `cargo run -- bonds -p 40453`. Must be run against observer node. Should now see 5 bonds.

Repo - https://github.com/F1R3FLY-io/f1r3fly

Branch - `main`

1. In separate terminal, run validator4 node: `docker compose -f docker/validator4.yml up`.
2. Wait ~10 seconds. In initial shard logs, should see 5 peers.