# Task 11: Configuration & CLI Flags

## Architecture Reference

- [Layer 1 §1.4 — Module Changes](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L219-L230)
- [Layer 4 §4.2 — DA Configuration](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L734-L752)
- [Layer 5 §5.2 — Pricing Configuration](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L806-L822)
- [Module Summary](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md#L893-L922)

---

## What to Implement

All CLI flags, config file entries, and configuration data classes needed across layers. This is a cross-cutting task that provides the typed config objects other tasks consume.

### Technical Details

#### Modified File: `node/src/main/.../Options.scala`

Add CLI flags:
- `--file-upload-chunk-size` (default: 4MB) — chunk size for upload streaming
- `--file-replication-dir` (default: `<data-dir>/file-replication/`) — directory for stored files
- `--file-upload-phlo-per-storage-byte` (default: 1) — phlo per byte for storage pricing
- `--max-concurrent-downloads-per-ip` (default: 4) — download rate limit

#### Modified File: `node/src/main/resources/defaults.conf`

```hocon
f1r3fly {
  file-upload {
    chunk-size = 4194304          # 4MB
    replication-dir = "file-replication"
    phlo-per-storage-byte = 1
    base-register-phlo = 300
    max-concurrent-downloads-per-ip = 4
  }

  consensus {
    da {
      file-fetch-timeout = 10 minutes
      max-concurrent-downloads = 8
      max-concurrent-p2p-file-syncs = 4
    }
  }

  casper {
    max-file-data-size-per-block = 53687091200  # 50GB
    max-file-deploys-per-block = 10
  }
}
```

#### Modified File: `node/src/main/.../NodeRuntime.scala`

- Pass `FileUploadConfig` to `Runtime` and `FileUploadAPI`
- Create `file-replication/` directory on startup if it doesn't exist

#### Config data classes (if needed)

- `FileUploadConfig` — groups all file-upload settings
- Integrate DA config into existing `CasperConf`

---

## Verification

### New Test: `FileUploadConfigSpec`

**File**: `node/src/test/scala/coop/rchain/node/configuration/FileUploadConfigSpec.scala`

| Test Case | Assertion |
|-----------|-----------|
| Parse `defaults.conf` | All file-upload defaults parsed: chunk-size=4MB, phlo-per-byte=1, base-phlo=300 |
| Parse DA config | `file-fetch-timeout=10min`, `max-concurrent-downloads=8` |
| CLI override: `--file-replication-dir /tmp/x` | Overrides config file value |
| CLI override: `--file-upload-phlo-per-storage-byte 2` | Overrides default |
| Fresh start | `file-replication/` directory auto-created |
| Existing directory | No error on startup |

```bash
sbt 'node/testOnly coop.rchain.node.configuration.FileUploadConfigSpec'
```

### Existing Tests to Verify (regression)

```bash
# Options/config changes must not break existing node startup tests
sbt 'node/testOnly coop.rchain.node.*'
```

### Manual Verification

```bash
# Start node with custom file dir
./node --file-replication-dir /tmp/test-files --standalone
# Verify: /tmp/test-files/ directory created
ls -la /tmp/test-files/
```

---

## Subtasks

- [x] Add CLI flags to `Options.scala`
- [x] Add config section to `defaults.conf`
- [x] Create `FileUploadConfig` data class (if not already part of existing config)
- [x] Wire DA config into `CasperConf.scala`
- [x] Wire `fileReplicationDir` into `NodeRuntime.scala` (directory creation on startup)
- [x] Pass config to `FileUploadAPI`, `FileSystemProcess`, `BlockAPI`
- [x] Unit tests for config parsing and CLI overrides
