# TODO

## RSpace Lock Granularity

`event_log` and `produce_counter` in `rspace.rs` and `replay_rspace.rs` are always accessed together in `log_produce`/`log_consume` but use separate `Mutex` locks. Combining them into a single lock would reduce lock acquisitions per RSpace operation. Low priority — the critical sections are short and uncontended under per-channel locks, so overhead is ~5ns per extra lock.
