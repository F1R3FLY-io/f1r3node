use bincode;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use super::blake2b256_hash::Blake2b256Hash;

const RSS_SAMPLE_MIN_INTERVAL_MS: u64 = 200;

struct RssCache {
    last_sample_ms: AtomicU64,
    value_kb: AtomicU64,
}

impl RssCache {
    fn new() -> Self {
        Self {
            last_sample_ms: AtomicU64::new(0),
            value_kb: AtomicU64::new(0),
        }
    }
}

fn mem_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("F1R3_BLOCK_CREATOR_PHASE_SUBSTEP_PROFILE")
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(false)
    })
}

fn monotonic_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cold]
fn read_rss_from_proc_statm_kb() -> Option<u64> {
    // Linux default page size is 4 KiB on our validator targets.
    const PAGE_SIZE_KB: u64 = 4;
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let rss_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(rss_pages.saturating_mul(PAGE_SIZE_KB))
}

fn read_rss_kb() -> Option<u64> {
    if !mem_profile_enabled() {
        return None;
    }

    static CACHE: OnceLock<RssCache> = OnceLock::new();
    let cache = CACHE.get_or_init(RssCache::new);
    let now_ms = monotonic_now_ms();
    let last_ms = cache.last_sample_ms.load(Ordering::Relaxed);
    if last_ms != 0 && now_ms.saturating_sub(last_ms) < RSS_SAMPLE_MIN_INTERVAL_MS {
        let cached = cache.value_kb.load(Ordering::Relaxed);
        if cached > 0 {
            return Some(cached);
        }
    }

    let measured = read_rss_from_proc_statm_kb()?;
    cache.value_kb.store(measured, Ordering::Relaxed);
    cache.last_sample_ms.store(now_ms, Ordering::Relaxed);
    Some(measured)
}

fn log_step_delta(func_name: &str, step: &str, before_kb: Option<u64>) {
    if !mem_profile_enabled() {
        return;
    }
    if let (Some(before), Some(after)) = (before_kb, read_rss_kb()) {
        let delta_kb = after as i64 - before as i64;
        if delta_kb != 0 {
            eprintln!(
                "stable_hash.mem fn={} step={} rss_kb={} delta_kb={}",
                func_name, step, after, delta_kb
            );
        }
    }
}

// See rspace/src/main/scala/coop/rchain/rspace/hashing/StableHashProvider.scala
pub fn hash<C: Serialize>(channel: &C) -> Blake2b256Hash {
    let before_serialize = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let bytes = bincode::serialize(channel).unwrap();
    log_step_delta("hash", "after_serialize_channel", before_serialize);
    let before_new = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let result = Blake2b256Hash::new(&bytes);
    log_step_delta("hash", "after_blake2b_new", before_new);
    result
}

// TODO: Double check the sorting here against scala side
pub fn hash_vec<C: Serialize>(channels: &Vec<C>) -> Vec<Blake2b256Hash> {
    let profile_enabled = mem_profile_enabled();
    let mut hashes = Vec::with_capacity(channels.len());
    let mut serialize_delta_total_kb: i64 = 0;
    let mut hash_delta_total_kb: i64 = 0;
    let mut serialize_delta_nonzero_count: usize = 0;
    let mut hash_delta_nonzero_count: usize = 0;

    for channel in channels.iter() {
        let before_serialize = if profile_enabled { read_rss_kb() } else { None };
        let bytes = bincode::serialize(&channel).unwrap();
        if let (Some(before), Some(after)) = (before_serialize, read_rss_kb()) {
            let delta_kb = after as i64 - before as i64;
            if delta_kb != 0 {
                serialize_delta_total_kb += delta_kb;
                serialize_delta_nonzero_count += 1;
            }
        }

        let before_hash = if profile_enabled { read_rss_kb() } else { None };
        hashes.push(Blake2b256Hash::new(&bytes));
        if let (Some(before), Some(after)) = (before_hash, read_rss_kb()) {
            let delta_kb = after as i64 - before as i64;
            if delta_kb != 0 {
                hash_delta_total_kb += delta_kb;
                hash_delta_nonzero_count += 1;
            }
        }
    }

    if profile_enabled && (serialize_delta_total_kb != 0 || hash_delta_total_kb != 0) {
        eprintln!(
            "stable_hash.hash_vec.summary channels={} serialize_delta_total_kb={} \
             serialize_nonzero_count={} hash_delta_total_kb={} hash_nonzero_count={}",
            channels.len(),
            serialize_delta_total_kb,
            serialize_delta_nonzero_count,
            hash_delta_total_kb,
            hash_delta_nonzero_count
        );
    }

    let before_sort = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    hashes.sort();
    log_step_delta("hash_vec", "after_sort_hashes", before_sort);
    hashes
}

pub fn hash_from_vec<C: Serialize>(channels: &Vec<C>) -> Blake2b256Hash {
    if channels.len() == 1 {
        let before_fast_path = if mem_profile_enabled() {
            read_rss_kb()
        } else {
            None
        };
        let result = hash(channels.first().unwrap());
        log_step_delta("hash_from_vec", "after_single_channel_fast_path", before_fast_path);
        return result;
    }

    let before_hash_vec = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let hashes = hash_vec(channels);
    log_step_delta("hash_from_vec", "after_hash_vec", before_hash_vec);
    let before_hash_from_hashes = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let result = hash_from_hashes(&hashes);
    log_step_delta("hash_from_vec", "after_hash_from_hashes", before_hash_from_hashes);
    result
}

// TODO: Double check the sorting here against scala side
pub fn hash_from_hashes(channels_hashes: &Vec<Blake2b256Hash>) -> Blake2b256Hash {
    let before_collect_refs = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let mut ord_refs: Vec<&Blake2b256Hash> = channels_hashes.iter().collect();
    log_step_delta("hash_from_hashes", "after_collect_refs", before_collect_refs);
    let before_sort_refs = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    ord_refs.sort();
    log_step_delta("hash_from_hashes", "after_sort_refs", before_sort_refs);
    let before_concat = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let mut concatenated: Vec<u8> = Vec::with_capacity(ord_refs.len() * 32);
    for h in ord_refs.iter() {
        concatenated.extend(h.0.iter().copied());
    }
    log_step_delta("hash_from_hashes", "after_concat", before_concat);
    let before_new = if mem_profile_enabled() {
        read_rss_kb()
    } else {
        None
    };
    let result = Blake2b256Hash::new(&concatenated);
    log_step_delta("hash_from_hashes", "after_blake2b_new", before_new);
    result
}

// See rspace/src/main/scala/coop/rchain/rspace/hashing/StableHashProvider.scala
pub fn hash_consume<P: Serialize, K: Serialize>(
    mut encoded_channels: Vec<Vec<u8>>,
    patterns: &[P],
    continuation: &K,
    persist: bool,
) -> Blake2b256Hash {
    let mut encoded_patterns = patterns
        .iter()
        .map(|pattern| bincode::serialize(pattern).unwrap())
        .collect::<Vec<_>>();
    encoded_patterns.sort();

    let encoded_continuation = bincode::serialize(continuation).unwrap();
    let encoded_persist = bincode::serialize(&persist).unwrap();

    encoded_channels.extend(encoded_patterns);
    encoded_channels.push(encoded_continuation);
    encoded_channels.push(encoded_persist);

    let encoded = bincode::serialize(&encoded_channels).unwrap();
    Blake2b256Hash::new(&encoded)
}

pub fn hash_produce<A: Serialize>(
    encoded_channel: Vec<u8>,
    datum: &A,
    persist: bool,
) -> Blake2b256Hash {
    let encoded_datum = bincode::serialize(datum).unwrap();
    let encoded_persist = bincode::serialize(&persist).unwrap();

    let encoded_vec = vec![encoded_channel, encoded_datum, encoded_persist];

    let encoded = bincode::serialize(&encoded_vec).unwrap();
    Blake2b256Hash::new(&encoded)
}
