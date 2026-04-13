# Changelog

All notable changes to the Rust implementation of F1r3node will be documented in this file.
This changelog is automatically generated from conventional commits.


## [v0.4.11] - 2026-04-09

### Bug Fixes

- node API improvements, deploy cost estimation (#472) ([#472](https://github.com/F1R3FLY-io/f1r3node/pull/472))

### CI

- add rust/staging to CI triggers

### Documentation

- refactor docs structure, add consensus protocol walkthrough (#471) ([#471](https://github.com/F1R3FLY-io/f1r3node/pull/471))

### Performance

- remove adaptive deploy cap, bonds cache, deploy wake (#473) ([#473](https://github.com/F1R3FLY-io/f1r3node/pull/473))


## [v0.4.10] - 2026-04-06

### Bug Fixes

- v0.4.9 eval order, observer stability, and client-reported issues (#468) ([#468](https://github.com/F1R3FLY-io/f1r3node/pull/468))


## [v0.4.9] - 2026-04-01

### Bug Fixes

- deterministic datum selection in extract_data_candidates
- PR links in changelog, extract release notes, skip noisy commits
- remove unused rand imports
- remove non-deterministic shuffle from RSpace datum matching

### CI

- add concurrency block to cancel outdated workflow runs


## [v0.4.8] - 2026-03-31

### Refactoring

- separate produce error handling from NonDeterministicProcessFailure (Rust) (#456) ([#456](https://github.com/F1R3FLY-io/f1r3node/pull/456))


## [v0.4.7] - 2026-03-30

### Bug Fixes

- always compute real fault tolerance instead of hardcoded shortcut
- update rholang test expectations for receives-first eval order
- handle unused Result from remove_datum in test

### CI

- skip test_network_recovers_from_slow_deploy (f1r3node#463)
- add test_replay_determinism to integration test matrix


## [v0.4.6] - 2026-03-30

### Bug Fixes

- add retry logic to gRPC server port binding
- kill stale processes on test ports before integration tests
- align integration test matrix with python_files config

### CI

- migrate Rust CI from self-hosted Oracle runners to GitHub-hosted runners


## [v0.4.5] - 2026-03-29

### Bug Fixes

- re-enable integration tests, run unit tests in release mode


## [v0.4.4] - 2026-03-28

### Bug Fixes

- use bodyFile instead of bodyPath for release changelog


## [v0.4.3] - 2026-03-27

### Bug Fixes

- split credential push, fix Docker tag logic, add changelog to releases


## [v0.4.2] - 2026-03-27

### Bug Fixes

- use PAT checkout for tag triggers, disable integration tests, run tests in release mode
- replay determinism for duplicate channel sends and join consumes
- shard recovery from block rejection and DAG tip divergence


## [v0.4.1] - 2026-03-27

### Bug Fixes

- mark releases as stable and include changelog in release body
- use non-streaming docker compose logs for genesis check
- set synchrony-constraint-threshold to 0 across configs, env vars, and docs
- address PR review — shared version logic, CI skip, startup name, changelog
- add missing fields to RunOptions in test
- address PR #436 review comments
- Produce identity and non-deterministic process replay
- ReplayRSpace.is_replay() returns true during block replay
- prevent GInt subtraction and modulo overflow panics
- address PR #425 review feedback
- pin rholang-rs to master commit d25f953a (PR #90 merged)
- address PR #90 review feedback — BigInt cap removal, FixedPoint mul, error types
- pin rholang-rs dependency to master commit 600110e2
- remove unconfirmed race condition comments from config files
- address PR #429 review comments
- serialize bytes fields as base64 strings to match Scala node JSON output
- move clique oracle error log into else branch to prevent unconditional logging
- address PR #421 review feedback
- address PR #325 review feedback (round 2)
- address PR #325 review feedback
- address PR #325 review findings against Scala source
- prevent node crash from malformed secp256k1Verify/hash contract inputs
- detect and report integer overflow in Rholang arithmetic (#415) ([#415](https://github.com/F1R3FLY-io/f1r3node/pull/415))
- always restart Docker daemon on CI to clear TIME_WAIT sockets from joiner ports
- harden CI Docker networking for integration tests
- move effectiveEndBlockNumber before forward reference in getBlocksByHeights
- clamp BlockAPI depth parameters instead of rejecting requests
- use multiset union in ChannelChange.combine to prevent datum duplication
- revert pytest-timeout back to constant 600s
- scale pytest-timeout to 900s on arm64 CI
- add --timeout-scale=1.5 for arm64 CI integration tests
- pin bootstrap peer in KademliaStore to prevent discovery death spiral
- use deterministic seeds in MergeNumberChannelSpec
- make transport layer respect configured network-timeout and collapse CI integration jobs
- clean root-owned LMDB data before checkout in build_docker_image
- queue concurrent CI runs and add Docker cleanup step
- add handleErrorWith to ApprovedBlock fallback loop
- add ApprovedBlock fallback in GenesisValidator for late-connecting nodes
- LimitedParentDepthSpec tests were not executing
- Add stackPubKey to systemPublicKeys
- Make heartbeat integration test timing-agnostic
- Update test to expect deterministic finalization result
- Deterministic parent ordering for consistent finalization
- Fix WSL2 networking and HTTP deploy signature
- Add missing onBlockFinalized parameter to InitializingSpec
- Fixes for integration tests folder to start them locally (#291) (#300) ([#300](https://github.com/F1R3FLY-io/f1r3node/pull/300))
- resolve test compilation errors after Ollama system integration
- cleanup ai.rho Rholang example (#167) ([#167](https://github.com/F1R3FLY-io/f1r3node/pull/167))
- replace hardcoded bootstrap address with configurable default (#164) ([#164](https://github.com/F1R3FLY-io/f1r3node/pull/164))
- update produce reference handling in ReplayRSpace to ensure correct retrieval from produces
- update expected hash in RuntimeSpec test case
- enable mergeable tag in RhoRuntime creation
- backport integer overflow detection to Rust interpreter (#420) ([#420](https://github.com/F1R3FLY-io/f1r3node/pull/420))
- always restart Docker daemon on CI to clear TIME_WAIT sockets from joiner ports
- move sigar reporter off async runtime and fix flaky test
- set RUSTFLAGS per-arch to suppress +sse2 warning on aarch64
- DAG snapshot isolation, genesis early-exit, joins sorting, and LCA determinism
- clamp BlockAPI depth parameters instead of rejecting requests
- use multiset union in ChannelChange::combine to prevent datum duplication
- add mutex to CasperBuffer to prevent TOCTOU race in block dependency DAG
- use hardcoded genesis timestamp in validate_candidate to fix genesis consensus failure
- backport two-phase LCA, bootstrap pinning, and ApprovedBlock fallback from Scala PR #391
- deterministic seeds, test fixes, and Docker build caching
- always restart Docker daemon on CI to clear TIME_WAIT sockets from joiner ports
- harden CI Docker networking for integration tests
- move effectiveEndBlockNumber before forward reference in getBlocksByHeights
- clamp BlockAPI depth parameters instead of rejecting requests
- use multiset union in ChannelChange.combine to prevent datum duplication
- revert pytest-timeout back to constant 600s
- scale pytest-timeout to 900s on arm64 CI
- add --timeout-scale=1.5 for arm64 CI integration tests
- pin bootstrap peer in KademliaStore to prevent discovery death spiral
- use deterministic seeds in MergeNumberChannelSpec
- make transport layer respect configured network-timeout and collapse CI integration jobs
- clean root-owned LMDB data before checkout in build_docker_image
- queue concurrent CI runs and add Docker cleanup step
- add handleErrorWith to ApprovedBlock fallback loop
- add ApprovedBlock fallback in GenesisValidator for late-connecting nodes
- LimitedParentDepthSpec tests were not executing
- Add stackPubKey to systemPublicKeys
- Make heartbeat integration test timing-agnostic
- Update test to expect deterministic finalization result
- Deterministic parent ordering for consistent finalization
- Fix WSL2 networking and HTTP deploy signature
- Add missing onBlockFinalized parameter to InitializingSpec
- Fixes for integration tests folder to start them locally (#291) (#300)
- resolve test compilation errors after Ollama system integration
- cleanup ai.rho Rholang example (#167)
- replace hardcoded bootstrap address with configurable default (#164)
- update produce reference handling in ReplayRSpace to ensure correct retrieval from produces
- update expected hash in RuntimeSpec test case
- enable mergeable tag in RhoRuntime creation
- Update tests to use seq-num field name
- Rename seq_number to seq-num in WebSocket events to match Scala
- WebSocket event stream waker bug causing missed events
- Replace circe-generic-extras auto._ with explicit derivation
- Remove forked compilation for node project
- Increase SBT stack to 8m for circe-generic-extras macro expansion
- Increase SBT heap to 6g for forked node compilation
- Add RUSTFLAGS for gxhash AES/SSE2 requirement in CI
- Add duplicate finalization filter to match Scala PR #231
- Address PR review comments - optimize multiset_diff and remove duplication
- API format for explore-deploy and multi-parent mode configs
- Add missing TransportLayer trait methods to test impl
- Only apply RecentHashFilter to gossip messages (BlockHashMessage, HasBlock)
- Cache event_log in ReplayCache and call rig() on cache hit
- Disable replay/state caches by default to match Scala behavior
- Address review comments - bound BFS traversal and add clarifying comments
- Clean up multi-parent merging implementation and match Scala logging
- Match Scala PR #288 multi-parent merging behavior
- Update calls to match new with_ancestors signature
- remove static OpenSSL linking on all platforms
- disable static OpenSSL linking on macOS
- Address PR review comments
- serialize metrics tests to prevent flaky failures (#322) ([#322](https://github.com/F1R3FLY-io/f1r3node/pull/322))
- Make heartbeat integration test timing-agnostic
- Additional adjustments and uncommenting the docker push/tag lines
- fixed docker push triggering on pull request update
- Fixed incorrect naming
- Cleanup duplicated action
- Make the CI yaml ready for merge
- trying to optimize the build time for the release package
- CI fixes for multiplatform release build
- Fixes for review comments and remove unnecessary usage of qemu actions for integration test action
- Make the multi_arch build available only for tags and releases to speedup the regular ci pipelines
- Removed the docker prebuild caching and cleanup before build
- Combine cargo clean and build into single RUN command
- Add cargo clean to fix Docker build cache issue
- Fixed all integration tests for the Rust node which are also passing for the Scala one (#304) ([#304](https://github.com/F1R3FLY-io/f1r3node/pull/304))
- Provide fixes for the rust node in order to successfully pass the genesis ceremony integration test (#301) ([#301](https://github.com/F1R3FLY-io/f1r3node/pull/301))
- Fixes for integration tests folder to start them locally (#291) ([#291](https://github.com/F1R3FLY-io/f1r3node/pull/291))
- Fixed review comments
- Fixed review comments and applied some fixes after testing
- Fixed usage of the grpc port logic
- Fixed logic in listenInName implementation and tests
- Rename ProposeService and DeployService files, traits and structs to be consitent with the Scala version
- Added the limit for the grpc message size
- Fixed review comments

### CI

- revert system-integration clone to ref: main
- temp use system-integration PR branch for integration tests
- add dev branch to push triggers
- revert system-integration checkout to main branch
- rename runner label f1r3fly-ci to f1r3fly-scala-ci
- migrate build_docker_image and integration tests to OCI self-hosted runners
- trigger with static shard JVM tuning
- split asymmetric bond tests into individual jobs
- split integration tests into parallel jobs by test group
- trigger integration tests with staggered custom shard startup
- add --timeout-scale=3 and --timeout=1200 for integration tests
- increase timeouts and use system-integration cleanup branch
- increase command-timeout to 300s for amd64 integration tests
- Add all node unit tests to CI workflows
- classify NoNewDeploys as transient in latency benchmark
- add rust/main to push/PR triggers and promote to release branch
- revert system-integration checkout to main branch
- migrate Docker build and integration tests to self-hosted runners
- stabilize latency/soak tooling
- add dev branch to push triggers
- revert system-integration checkout to main branch
- rename runner label f1r3fly-ci to f1r3fly-scala-ci
- migrate build_docker_image and integration tests to OCI self-hosted runners
- trigger with static shard JVM tuning
- split asymmetric bond tests into individual jobs
- split integration tests into parallel jobs by test group
- trigger integration tests with staggered custom shard startup
- add --timeout-scale=3 and --timeout=1200 for integration tests
- increase timeouts and use system-integration cleanup branch
- increase command-timeout to 300s for amd64 integration tests
- Add all node unit tests to CI workflows
- stabilize latency/soak tooling
- report peer requery suppression metric in latency profile
- expand latency profiler and add correctness-first e2e reduction plan
- add finalizer health summary to validator soak output
- expand validator soak profiling and document 2026-02-21 findings
- Add test log artifacts on failure

### Documentation

- fix README flow — monitoring teardown belongs in Monitoring section
- add monitoring teardown to stop instructions
- split Rust codebase documentation into per-module files
- add .env.example, env var defaults in compose, AI services section
- rewrite README, remove Nix requirement, add rholang-cli, deprecate Scala
- add schnorr-frost-secp256k1 design and implementation status
- fix smoke test ports and add --tail=500 to logs grep
- update integration-tests references to point to system-integration repo
- update integration-tests references to point to system-integration repo
- update integration-tests references to point to system-integration repo
- add replicated 120s cooldown confirmation and decision
- add 120s peer requery cooldown sweep results
- add preliminary 1000ms vs 1500ms cooldown tuning results
- add replicated cooldown stability results
- record latest commit soak re-validation and variance

### Features

- version bump on rust/dev (patch) and split push to reduce CI triggers
- align monitoring stack with system-integration
- add auto-versioning, changelog generation, and release tagging
- add CLI flags for ceremony-master-mode and mergeable-channel-gc
- add extended numeric types (Float, BigInt, BigRat, FixedPoint)
- expose transfer details in DeployInfo (backport Scala PR #315)
- Ported ReportingCasper and TransactionApi tests
- publish Docker images on dev branch push
- Disable validator progress check in standalone mode
- Expose native-token transfer details in DeployInfo (#212)
- add OpenAI configuration and service integration
- publish Docker images on rust/dev branch push
- Disable validator progress check in standalone mode
- Expose native-token transfer details in DeployInfo (#212)
- add OpenAI configuration and service integration
- Add local development setup with just command runner
- Complete PR #320 backport - Kademlia removal and immediate disconnect
- Backport PR #320 - Configurable peer discovery intervals and aggressive cleanup
- Backport PR #261 - Add peer list to status endpoints
- Backport replay/state caching from Scala PR #244
- Backport RecentHashFilter from Scala PR #243
- Backport mergeable channels GC from Scala PR #218
- Backport WebSocket event system improvements
- Backport PRs 292, 344, 345 - Finalizer fix and new tests
- Backport PR #288 - Remove GHOST filtering of block parents in Casper
- Disable validator progress check in standalone mode
- Backport Casper protocol fixes from Scala PR #273 and #246
- Add allow_empty_blocks support to Proposer for heartbeat
- Added docker cross-compilation and removed qemu
- Updated the github workflows to support only Rust related build
- Running rust nodes cluster (#276) ([#276](https://github.com/F1R3FLY-io/f1r3node/pull/276))
- add max_dbs configuration support for LMDB environments
- Updated all workspace modules to use tracing crate and unify other third party crates versions (#263) ([#263](https://github.com/F1R3FLY-io/f1r3node/pull/263))
- Ported NodeEnvironment (#245) ([#245](https://github.com/F1R3FLY-io/f1r3node/pull/245))
- Ported instances package from Scala to Rust (#236) ([#236](https://github.com/F1R3FLY-io/f1r3node/pull/236))
- Ported the web package from Scala to Rust (#229) ([#229](https://github.com/F1R3FLY-io/f1r3node/pull/229))
- Ported admin_web_api, lsp_grpc_service and propose_grpc_service and deploy_grpc_service_v1 (#203) ([#203](https://github.com/F1R3FLY-io/f1r3node/pull/203))
- Ported repl grpc service (#201) ([#201](https://github.com/F1R3FLY-io/f1r3node/pull/201))
- Ported the WebApi from node from scala to rust and its related dependecies. Refactored BlockApi ApiResult type.
- Ported config checks for node startup
- Ported the runCLI method and all unimplemented dependencies required for this method to run
- GrpcProposeService porting to Rust
- Ported the gRPCReplClient to Rust. Updated the version of tonic to 0.14 for the whole repository
- Ported the node configuration from Scala to Rust

### Miscellaneous

- remove legacy run-standalone-dev.sh
- auto-publish GitHub Releases instead of drafts
- consolidate docker configs and remove stale Scala-era files
- remove broken optional-tests workflow, rename CI to Rust
- remove integration-tests (moved to system-integration repo)
- Reduce integration test timeouts from 30 minutes to 5 minutes
- trigger CI
- Add disable-late-block-filtering config to docker conf files
- trigger CI
- remove integration-tests (moved to system-integration repo)
- remove redundant Scala-reference comments from GenesisValidator
- checkpoint all pending changes for consistent branch state
- fully opt in to generic rho:vault/system URIs
- remove integration-tests (moved to system-integration repo)
- Reduce integration test timeouts from 30 minutes to 5 minutes
- trigger CI
- Add disable-late-block-filtering config to docker conf files
- trigger CI
- Trigger CI
- Fixed typo
- Refactor and simplified the pipeline execution logic
- Removed verification section for ci workaround check
- Updated for the multiplatform build check on CI
- check multiplatform build, revert after successfull ci check
- trigger CI

### Performance

- two-phase bounded LCA + scope walk to reduce merge cost from O(N*chain_length) to O(chain_length)
- bounded LCA algorithm and pre-computed lfbAncestors for DagMerger
- Add O(1) nonEmpty to DeployStorage
- cache DAG metadata lookups and trim hot-path logging
- reduce propose tail and RSS-probe overhead
- unblock deploy-driven frontier follow during synchrony catch-up
- bypass min-interval on first retry to reduce trigger jitter
- cut hot-store metric overhead and small-batch checkpoint parallelism
- make heartbeat event-driven and reuse snapshot in pending-deploy check
- throttle heartbeat recovery and reduce snapshot churn
- retune heartbeat/finalization loop for lower e2e latency
- reduce compute_state overhead in batched checkpoint path
- two-phase bounded LCA + scope walk to reduce merge cost from O(N*chain_length) to O(chain_length)
- bounded LCA algorithm and pre-computed lfbAncestors for DagMerger
- Add O(1) nonEmpty to DeployStorage

### Refactoring

- replace config overlays with CLI flags, align defaults.conf
- Remove non-functional StateSnapshotCache from PR #244
- Improve SystemContractInitializationSpec tests
- Migrate integration tests to the new F1R3FLY Python client library (#309) ([#309](https://github.com/F1R3FLY-io/f1r3node/pull/309))
- simplify produce reference handling in ReplayRSpace and update related tests
- add ProduceResult
- Remove non-functional StateSnapshotCache from PR #244
- Improve SystemContractInitializationSpec tests
- Migrate integration tests to the new F1R3FLY Python client library (#309)
- simplify produce reference handling in ReplayRSpace and update related tests
- add ProduceResult

### Style

- apply scalafmt to BlockAPI depth clamping changes
- normalize indentation in reduce overflow branch
- apply scalafmt to BlockAPI depth clamping changes

### Testing

- stabilize initializing sync and exploratory LFB assertions
- Fix concurrent_sends test for RecentHashFilter compatibility
- remove leftover folder creation from validate_test
- remove leftover folder creation from API tests
- use shared LMDB in lmdb_key_value_store_spec


## [v0.1.0] - 2025-08-15


