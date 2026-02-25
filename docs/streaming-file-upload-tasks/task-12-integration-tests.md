# Task 12: Integration Tests & Manual Verification

## Architecture Reference

- [Verification Plan](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L926-L950)

---

## What to Implement

Add file upload/download integration test scenarios to the **existing** integration test infrastructure:
- **Scala unit tests**: Run via sbt in CI ([required-workflow.yml L156-L256](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/.github/workflows/required-workflow.yml#L156-L256))
- **Python integration tests**: Run via `poetry run pytest` against Dockerized nodes in the [system-integration](https://github.com/F1R3FLY-io/system-integration) repo ([required-workflow.yml L424-L563](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/.github/workflows/required-workflow.yml#L424-L563))

### Technical Details

#### Existing CI Pipeline

```
required-workflow.yml
├── build_base              → sbt compile
├── required_scala_unit_tests  → sbt testOnly (matrix per package)
├── build_docker_image      → f1r3flyindustries/f1r3fly-scala-node
└── required_integration_tests
    ├── Clone F1R3FLY-io/system-integration (main branch)
    ├── poetry install --with integration
    └── poetry run pytest integration-tests/test/ -v --tb=short
```

All file upload tests must fit into this pipeline — **no new CI jobs** needed.

#### 1. Scala Unit Tests (sbt matrix — already in CI)

New test files created by Tasks 1–11 will be picked up automatically by existing CI matrix entries:

| CI Matrix Entry | New Tests Picked Up |
|----------------|---------------------|
| `casper/test:testOnly coop.rchain.casper.engine.*` | `FileReplicationSpec`, `OrphanFileCleanupSpec`, `FileAvailabilityValidationSpec`, `DAGateSpec`, `ConsensusDASpec` |
| `casper/test:testOnly coop.rchain.casper.api.*` | `DownloadFileAPITest` |
| `node/test:testOnly coop.rchain.node.api.*` | `FileUploadAPISpec`, `SyntheticDeploySpec`, `FileUploadCostSpec` |
| `rholang/test:testOnly coop.rchain.rholang.*` | `FileSystemProcessSpec` |
| `models/test:testOnly coop.rchain.models.*` | `FileUploadProtoSpec` |

**Tests that need new CI matrix entries** (add to `required_scala_unit_tests.strategy.matrix.tests`):

```yaml
# In required-workflow.yml, under required_scala_unit_tests matrix:
- "'casper/test:testOnly coop.rchain.casper.FileRegistrySpec'"
- "'casper/test:testOnly coop.rchain.casper.FileDeploySelectionSpec'"
- "'casper/test:testOnly coop.rchain.casper.StorageCostEnforcementSpec'"
```

Or they can be included in existing `casper/test:testOnly coop.rchain.casper.batch1.*` if placed in the `batch1` package.

#### 2. Python Integration Tests (system-integration repo)

Add new Pytest test file in `system-integration/integration-tests/test/`:

**File**: `system-integration/integration-tests/test/test_file_upload.py`

This follows the same pattern as existing test files in the `system-integration` repo, using the `pyf1r3fly` client library and Docker Compose node fixtures.

```python
"""File upload/download integration tests.

Tests run against a Docker-based shard network (bootstrap + validators + observer)
managed by the existing conftest.py fixtures.
"""

class TestFileUpload:
    """Upload file via gRPC, verify blockchain registration and download."""

    def test_upload_file_returns_hash_and_deploy_id(self, bootstrap_node):
        """Upload 1MB file → verify FileUploadResponse contains fileHash + deployId."""
        ...

    def test_upload_file_deploy_appears_in_block(self, bootstrap_node):
        """Upload file → findDeploy(deployId) returns LightBlockInfo after propose."""
        ...

    def test_upload_file_is_finalized(self, bootstrap_node, validator_nodes):
        """Upload file → isFinalized(blockHash) returns true."""
        ...

    def test_upload_duplicate_file_skips_disk_write(self, bootstrap_node):
        """Upload same file twice → second returns same hash, no error."""
        ...

    def test_upload_insufficient_phlo_rejected(self, bootstrap_node):
        """Upload with phloLimit < required → immediate rejection."""
        ...


class TestFileDownload:
    """Download file via observer-only gRPC."""

    def test_download_from_observer(self, readonly_node):
        """Download file from observer → bytes match original."""
        ...

    def test_download_from_validator_rejected(self, bootstrap_node):
        """Download from validator → error: read-only only."""
        ...

    def test_download_nonexistent_file(self, readonly_node):
        """Download unknown hash → NOT_FOUND."""
        ...


class TestFileDeletion:
    """Owner-only file deletion with reference counting."""

    def test_delete_file_owner(self, bootstrap_node):
        """Upload file → delete as owner → verify removed after finalization."""
        ...

    def test_delete_file_nonowner_fails(self, bootstrap_node):
        """Upload file → attempt delete from different deployer → rejected."""
        ...


class TestFileReplication:
    """Cross-validator file sync during block propagation."""

    def test_file_replicated_to_validators(self, bootstrap_node, validator_nodes):
        """Upload on bootstrap → verify file exists on all validators."""
        ...
```

#### Run Command (local + CI)

```bash
# Local: run file upload tests only
cd system-integration
poetry run pytest integration-tests/test/test_file_upload.py -v --tb=short \
  --startup-timeout=600 --command-timeout=300 --timeout=600

# CI: all integration tests (existing command picks up new file automatically)
poetry run pytest integration-tests/test/ -v --tb=short --log-cli-level=WARNING \
  --startup-timeout=600 --command-timeout=300 --timeout=600
```

The CI step at [required-workflow.yml L535](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/.github/workflows/required-workflow.yml#L535) runs `pytest integration-tests/test/` which auto-discovers any `test_*.py` files — **no CI config change needed** for integration tests.

#### 3. CI Workflow Changes

**File**: [required-workflow.yml](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/.github/workflows/required-workflow.yml)

- Add new matrix entries for `FileRegistrySpec`, `FileDeploySelectionSpec`, `StorageCostEnforcementSpec` under `required_scala_unit_tests` (if not placed in existing packages)
- No changes needed for `required_integration_tests` — pytest auto-discovers `test_file_upload.py`

---

## Verification

### Run All Scala Unit Tests (file-upload-related)

```bash
# Run ALL file-upload-related unit tests across all modules
sbt 'models/testOnly coop.rchain.models.FileUploadProtoSpec' \
    'node/testOnly coop.rchain.node.api.FileUploadAPISpec' \
    'node/testOnly coop.rchain.node.api.SyntheticDeploySpec' \
    'node/testOnly coop.rchain.node.api.FileUploadCostSpec' \
    'node/testOnly coop.rchain.node.configuration.FileUploadConfigSpec' \
    'rholang/testOnly coop.rchain.rholang.interpreter.FileSystemProcessSpec' \
    'casper/testOnly coop.rchain.casper.FileRegistrySpec' \
    'casper/testOnly coop.rchain.casper.api.DownloadFileAPITest' \
    'casper/testOnly coop.rchain.casper.engine.FileReplicationSpec' \
    'casper/testOnly coop.rchain.casper.engine.OrphanFileCleanupSpec' \
    'casper/testOnly coop.rchain.casper.engine.FileAvailabilityValidationSpec' \
    'casper/testOnly coop.rchain.casper.engine.DAGateSpec' \
    'casper/testOnly coop.rchain.casper.engine.ConsensusDASpec' \
    'casper/testOnly coop.rchain.casper.FileDeploySelectionSpec' \
    'casper/testOnly coop.rchain.casper.StorageCostEnforcementSpec'
```

### Run Python Integration Tests (local)

```bash
cd system-integration
poetry run pytest integration-tests/test/test_file_upload.py -v --tb=short \
  --startup-timeout=600 --command-timeout=300 --timeout=600
```

### Run Full Regression Suite

```bash
sbt test
```

### CI Validation

Push to a branch → verify both CI jobs pass:
- `Required Unit Tests (Scala)` — all matrix entries green
- `Integration Tests ($arch)` — `test_file_upload.py` discovered and passes

---

## Subtasks

- [ ] Add `FileRegistrySpec` / `FileDeploySelectionSpec` / `StorageCostEnforcementSpec` to CI matrix in [required-workflow.yml](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/.github/workflows/required-workflow.yml) (or place tests in existing packages)
- [ ] Create `test_file_upload.py` in `system-integration/integration-tests/test/`
- [ ] Implement `TestFileUpload` class (upload + hash + deployId + finalization)
- [ ] Implement `TestFileDownload` class (observer-only + resume + errors)
- [ ] Implement `TestFileDeletion` class (owner delete + ref counting)
- [ ] Implement `TestFileReplication` class (cross-validator sync)
- [ ] Run integration tests locally against Docker shard
- [ ] Push branch → verify CI green for both unit + integration jobs
- [ ] Document any discovered issues
