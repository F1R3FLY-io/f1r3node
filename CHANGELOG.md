# Changelog

All notable changes to the Scala implementation of F1r3node will be documented in this file.
This changelog is automatically generated from conventional commits.


## [v0.4.4] - 2026-03-28

### Bug Fixes

- use bodyFile instead of bodyPath for release changelog


## [v0.4.3] - 2026-03-27

### Bug Fixes

- split credential push, fix Docker tag logic, add changelog to releases


## [v0.4.2] - 2026-03-27

### Bug Fixes

- use PAT checkout for tag triggers, disable integration tests temporarily


## [v0.4.1] - 2026-03-27

### Bug Fixes

- mark releases as stable and include changelog in release body
- use non-streaming docker compose logs for genesis check
- set synchrony-constraint-threshold to 0 across configs and docs
- address PR review — loop guard, shared version logic, CI skip, startup name
- add --force to tag fetch to prevent clobber error
- GC gracefully skips the cycle instead of crashing (#450)
- preserve META-INF/services in assembly JAR to restore SLF4J logging
- use amount parameter instead of hardcoded 1000 in BondingUtil
- prevent node crash from malformed secp256k1Verify/hash contract inputs
- detect and report integer overflow in Rholang arithmetic (#415)
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
- run integration tests sequentially (remove -n 3 --dist=loadgroup)
- install docker-compose for system-integration tests
- run poetry lock before install to sync lock file with pyproject
- clone system-integration from integration-tests-and-parallel-execution
- migrate integration tests to system-integration suite
- Add all node unit tests to CI workflows

### Documentation

- fix README flow — monitoring teardown belongs in Monitoring section
- add monitoring teardown to stop instructions
- add .env.example, env var defaults in compose, AI services section
- rewrite README, add manual install, remove DEVELOPER.md
- update integration-tests references to point to system-integration repo

### Features

- version bump on dev (patch) and split push to reduce CI triggers
- align monitoring stack with system-integration
- add auto-versioning, changelog generation, and release tagging
- add --ceremony-master-mode and --enable/disable-mergeable-channel-gc CLI flags
- publish Docker images on dev branch push
- Disable validator progress check in standalone mode
- Expose native-token transfer details in DeployInfo (#212)

### Miscellaneous

- auto-publish GitHub Releases instead of drafts
- remove integration-tests (moved to system-integration repo)
- Reduce integration test timeouts from 30 minutes to 5 minutes
- trigger CI
- Add disable-late-block-filtering config to docker conf files
- trigger CI

### Performance

- two-phase bounded LCA + scope walk to reduce merge cost from O(N*chain_length) to O(chain_length)
- bounded LCA algorithm and pre-computed lfbAncestors for DagMerger
- Add O(1) nonEmpty to DeployStorage

### Refactoring

- separate produce error handling from NonDeterministicProces… (#453)
- remove --disable-mergeable-channel-gc from all compose files (#441 fixed)
- remove rspace-plus-plus from Scala code and tests
- replace config overlays with CLI flags, align defaults.conf
- Remove non-functional StateSnapshotCache from PR #244
- Improve SystemContractInitializationSpec tests
- Migrate integration tests to the new F1R3FLY Python client library (#309)

### Comm

- add RecentHashFilter to suppress redundant gossip hash broadcasts (#243)

### Style

- apply scalafmt to BlockAPI depth clamping changes


## [v0.1.1] - 2025-08-15

### Bug Fixes

- update produce reference handling in ReplayRSpace to ensure correct retrieval from produces
- update expected hash in RuntimeSpec test case
- enable mergeable tag in RhoRuntime creation

### Features

- add OpenAI configuration and service integration

### Refactoring

- simplify produce reference handling in ReplayRSpace and update related tests
- add ProduceResult

### Cleanup

- remove println and temporary comments


