# Test Performance Analysis Results

## Logs Analyzed
- `integration-tests/all-2.logs`
- `integration-tests/all.logs`

## Target Test
- `integration-tests/test/test_parametrized_deploydata.py`

---

## Complete Results Table

| Log File | Chunk MB | Block MB | Started at | Finished at | Duration (min) | Total Chunks | Target GB | RAM GB | Crashed/Passed MB | Speed (MB/min) | Status | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all-2.logs | 1 | 8 | 10:52:50 | 11:26:12* | 33.4 | 6144 | 6 | 15 | ~400 | 12.0 | FAILED | OOM (400/6144 MB) |
| all-2.logs | 2 | 32 | 11:26:23 | 15:03:20* | 217.0 | 3072 | 6 | 15 | ~1600 | 7.4 | FAILED | OOM (1600/6144 MB) |
| all-2.logs | 32 | 256 | 15:03:48 | 15:28:37 | 24.8 | 192 | 6 | 15 | ~3072 | 123.9 | FAILED | OOM (block #12/24) |
| all-2.logs | 256 | 256 | 15:55:28 | 16:10:23 | 14.9 | 24 | 6 | 15 | 2048 | 137.4 | PASSED | 8/24 blocks |
| all.logs | 2 | 128 | 21:47:35 | 22:58:59* | 71.4 | 3072 | 6 | 15 | ~1920 | 26.9 | FAILED | OOM |
| all.logs | 32 | 128 | 23:10:08 | 23:50:06* | 40.0 | 192 | 6 | 15 | ~3072 | 76.8 | FAILED | OOM |
| all.logs | 32 | 256 | 01:50:13 | 02:26:45 | 36.5 | 192 | 6 | 15 | 6144 | 168.3 | PASSED | 24/24 blocks |
| all.logs | 128 | 256 | 02:27:47 | 02:54:39* | 26.9 | 48 | 6 | 15 | ~3328 | 123.7 | FAILED | OOM |

\* = Approximate end time (last successful propose before crash)

---

## Key Findings

### Successful Tests (Completed 6GB)
- **32MB chunk / 256MB block**: 36.5 min (all.logs) - **SUCCESS**
- **256MB chunk / 256MB block**: 14.9 min (all-2.logs) - **Partial** (only 8 blocks of 24)

### Failed Tests (OOM Errors)
- **1MB / 8MB**: Crashed at ~400 MB (6.5% progress)
- **2MB / 32MB**: Crashed at ~1600 MB (26% progress) 
- **2MB / 128MB**: Crashed at ~1920 MB (31% progress)
- **32MB / 128MB**: Crashed at ~3072 MB (50% progress)
- **32MB / 256MB** (first run): Crashed at ~3072 MB (50% progress)
- **128MB / 256MB**: Crashed at ~3328 MB (54% progress)

### Performance Insights
- **Best configuration**: 32MB chunk / 256MB block (completed 6GB in 36.5 min)
- Larger block sizes (256MB) perform better than smaller ones (8MB, 32MB, 128MB)
- Larger chunk sizes reduce overhead and improve throughput
- With 15GB RAM, configurations with smaller chunks/blocks run out of memory before completing
