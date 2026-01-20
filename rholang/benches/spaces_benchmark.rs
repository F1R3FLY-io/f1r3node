//! Benchmarks for Reified RSpaces
//!
//! These benchmarks measure the performance of core RSpace operations:
//! - Collection insertion/retrieval
//! - Phlogiston (gas) accounting
//! - VectorDB similarity operations (cosine, top-k, batch)
//! - Lock contention patterns
//!
//! Run with: cargo bench --bench spaces_benchmark
//! Report: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, BenchmarkId, Throughput,
};

use rholang::rust::interpreter::spaces::collections::{
    BagDataCollection, QueueDataCollection, StackDataCollection,
    SetDataCollection, DataCollection,
};
use rholang::rust::interpreter::spaces::{
    PhlogistonMeter, Operation,
};

// VectorDB and ndarray imports for similarity benchmarks
use ndarray::{Array1, Array2};
use rholang::rust::interpreter::tensor::{
    cosine_similarity, cosine_similarity_safe, top_k_similar,
    batch_cosine_similarity, l2_normalize,
};

// Concurrent access benchmarks
use std::sync::{Arc, RwLock};
use std::thread;
use std::collections::HashMap;

// =============================================================================
// Collection Type Comparison
// =============================================================================

fn bench_collection_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_types");
    let item_count = 1000;

    group.throughput(Throughput::Elements(item_count as u64));

    // Bag - unordered, O(1) insert
    group.bench_function("bag_insert_1000", |b| {
        b.iter(|| {
            let mut bag = BagDataCollection::<i32>::new();
            for i in 0..item_count {
                bag.put(black_box(i)).expect("put should succeed");
            }
        });
    });

    // Queue - FIFO, O(1) enqueue
    group.bench_function("queue_insert_1000", |b| {
        b.iter(|| {
            let mut queue = QueueDataCollection::<i32>::new();
            for i in 0..item_count {
                queue.put(black_box(i)).expect("put should succeed");
            }
        });
    });

    // Stack - LIFO, O(1) push
    group.bench_function("stack_insert_1000", |b| {
        b.iter(|| {
            let mut stack = StackDataCollection::<i32>::new();
            for i in 0..item_count {
                stack.put(black_box(i)).expect("put should succeed");
            }
        });
    });

    // Set - idempotent, hash-based
    group.bench_function("set_insert_1000", |b| {
        b.iter(|| {
            let mut set = SetDataCollection::<i32>::new();
            for i in 0..item_count {
                set.put(black_box(i)).expect("put should succeed");
            }
        });
    });

    // Set with duplicates (should be O(1) due to hash lookup)
    group.bench_function("set_insert_1000_duplicates", |b| {
        b.iter(|| {
            let mut set = SetDataCollection::<i32>::new();
            for i in 0..item_count {
                set.put(black_box(i % 100)).expect("put should succeed"); // Only 100 unique
            }
        });
    });

    group.finish();
}

fn bench_collection_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_retrieval");
    let item_count = 1000;

    // Pre-fill collections
    let filled_bag = {
        let mut bag = BagDataCollection::<i32>::new();
        for i in 0..item_count {
            bag.put(i).expect("put should succeed");
        }
        bag
    };

    let filled_queue = {
        let mut queue = QueueDataCollection::<i32>::new();
        for i in 0..item_count {
            queue.put(i).expect("put should succeed");
        }
        queue
    };

    let filled_stack = {
        let mut stack = StackDataCollection::<i32>::new();
        for i in 0..item_count {
            stack.put(i).expect("put should succeed");
        }
        stack
    };

    group.bench_function("bag_find_all", |b| {
        b.iter_batched(
            || filled_bag.clone(),
            |mut bag| {
                for i in 0..item_count {
                    let _ = bag.find_and_remove(|x| *x == i);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("queue_find_all", |b| {
        b.iter_batched(
            || filled_queue.clone(),
            |mut queue| {
                for i in 0..item_count {
                    let _ = queue.find_and_remove(|x| *x == i);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("stack_find_all", |b| {
        b.iter_batched(
            || filled_stack.clone(),
            |mut stack| {
                for i in (0..item_count).rev() {
                    let _ = stack.find_and_remove(|x| *x == i);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// Phlogiston (Gas) Accounting
// =============================================================================

fn bench_phlogiston(c: &mut Criterion) {
    let mut group = c.benchmark_group("phlogiston");

    for op_count in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*op_count as u64));

        group.bench_with_input(
            BenchmarkId::new("charge_operations", op_count),
            op_count,
            |b, &op_count| {
                b.iter(|| {
                    let meter = PhlogistonMeter::new(u64::MAX);

                    for i in 0..op_count {
                        let op = if i % 3 == 0 {
                            Operation::Send { data_size: 100 }
                        } else if i % 3 == 1 {
                            Operation::Receive
                        } else {
                            Operation::CreateChannel
                        };

                        let _ = meter.charge(&op);
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// VectorDB Similarity Benchmarks
// =============================================================================

/// Generate random f32 vector of given dimension
fn random_vector(dim: usize) -> Array1<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array1::from_vec((0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
}

/// Generate random embedding matrix (n_rows × dim)
fn random_embeddings(n_rows: usize, dim: usize) -> Array2<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array2::from_shape_vec(
        (n_rows, dim),
        (0..n_rows * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect(),
    ).expect("valid shape")
}

fn bench_vectordb_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectordb_cosine");

    // Benchmark different vector dimensions (common embedding sizes)
    for dim in [128, 384, 768, 1536].iter() {
        let a = random_vector(*dim);
        let b = random_vector(*dim);

        // Normalize for fair cosine comparison
        let a_norm = l2_normalize(&a);
        let b_norm = l2_normalize(&b);

        group.throughput(Throughput::Elements(*dim as u64));

        group.bench_with_input(
            BenchmarkId::new("single_pair", dim),
            dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(cosine_similarity(&a_norm, &b_norm))
                });
            },
        );

        // Safe version (handles edge cases)
        group.bench_with_input(
            BenchmarkId::new("single_pair_safe", dim),
            dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(cosine_similarity_safe(&a_norm, &b_norm))
                });
            },
        );
    }

    group.finish();
}

fn bench_vectordb_top_k_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectordb_top_k");
    let dim = 384; // Common embedding dimension

    for n_embeddings in [100, 1_000, 10_000].iter() {
        let query = l2_normalize(&random_vector(dim));
        let embeddings = random_embeddings(*n_embeddings, dim);

        group.throughput(Throughput::Elements(*n_embeddings as u64));

        // Top-5 selection (common for semantic search)
        group.bench_with_input(
            BenchmarkId::new("top_5", n_embeddings),
            n_embeddings,
            |bench, _| {
                bench.iter(|| {
                    black_box(top_k_similar(&query, &embeddings, 5))
                });
            },
        );

        // Top-10 selection
        group.bench_with_input(
            BenchmarkId::new("top_10", n_embeddings),
            n_embeddings,
            |bench, _| {
                bench.iter(|| {
                    black_box(top_k_similar(&query, &embeddings, 10))
                });
            },
        );

        // Top-50 selection (for retrieval-augmented generation)
        group.bench_with_input(
            BenchmarkId::new("top_50", n_embeddings),
            n_embeddings,
            |bench, _| {
                bench.iter(|| {
                    black_box(top_k_similar(&query, &embeddings, 50))
                });
            },
        );
    }

    group.finish();
}

fn bench_vectordb_batch_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectordb_batch_cosine");
    let dim = 384;

    // Batch queries against reference set
    for (n_queries, n_refs) in [(10, 100), (10, 1_000), (100, 1_000)].iter() {
        let queries = random_embeddings(*n_queries, dim);
        let references = random_embeddings(*n_refs, dim);

        let total_comparisons = (*n_queries * *n_refs) as u64;
        group.throughput(Throughput::Elements(total_comparisons));

        group.bench_with_input(
            BenchmarkId::new(&format!("{}x{}", n_queries, n_refs), total_comparisons),
            &total_comparisons,
            |bench, _| {
                bench.iter(|| {
                    black_box(batch_cosine_similarity(&queries, &references))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Lock Contention Benchmarks
// =============================================================================

/// Simulates space storage with RwLock for benchmarking lock patterns
struct SimulatedSpace {
    data: RwLock<HashMap<String, Vec<i32>>>,
}

impl SimulatedSpace {
    fn new() -> Self {
        SimulatedSpace {
            data: RwLock::new(HashMap::new()),
        }
    }

    fn produce(&self, channel: &str, value: i32) {
        let mut data = self.data.write().expect("write lock");
        data.entry(channel.to_string()).or_default().push(value);
    }

    fn consume(&self, channel: &str) -> Option<i32> {
        let mut data = self.data.write().expect("write lock");
        data.get_mut(channel).and_then(|v| v.pop())
    }

    fn peek(&self, channel: &str) -> Option<i32> {
        let data = self.data.read().expect("read lock");
        data.get(channel).and_then(|v| v.last().copied())
    }
}

fn bench_concurrent_same_channel(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_contention_same_channel");

    for n_threads in [2, 4, 8].iter() {
        let n_ops = 1000;
        group.throughput(Throughput::Elements((n_threads * n_ops) as u64));

        group.bench_with_input(
            BenchmarkId::new("produce", n_threads),
            n_threads,
            |bench, &n_threads| {
                bench.iter(|| {
                    let space = Arc::new(SimulatedSpace::new());
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let space = Arc::clone(&space);
                            thread::spawn(move || {
                                for i in 0..n_ops {
                                    space.produce("shared_channel", (t * n_ops + i) as i32);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );

        // Mixed produce/consume on same channel (high contention)
        group.bench_with_input(
            BenchmarkId::new("mixed_produce_consume", n_threads),
            n_threads,
            |bench, &n_threads| {
                bench.iter(|| {
                    let space = Arc::new(SimulatedSpace::new());
                    // Pre-populate to avoid empty consumes
                    for i in 0..1000 {
                        space.produce("shared_channel", i);
                    }

                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let space = Arc::clone(&space);
                            thread::spawn(move || {
                                for i in 0..n_ops {
                                    if (t + i) % 2 == 0 {
                                        space.produce("shared_channel", i as i32);
                                    } else {
                                        let _ = space.consume("shared_channel");
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_different_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_contention_different_channels");

    for n_threads in [2, 4, 8].iter() {
        let n_ops = 1000;
        group.throughput(Throughput::Elements((n_threads * n_ops) as u64));

        // Each thread uses its own channel (low contention, tests RwLock efficiency)
        group.bench_with_input(
            BenchmarkId::new("isolated_channels", n_threads),
            n_threads,
            |bench, &n_threads| {
                bench.iter(|| {
                    let space = Arc::new(SimulatedSpace::new());
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let space = Arc::clone(&space);
                            thread::spawn(move || {
                                let channel = format!("channel_{}", t);
                                for i in 0..n_ops {
                                    space.produce(&channel, i as i32);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );

        // Read-heavy workload (many peeks, few writes) - RwLock should excel here
        group.bench_with_input(
            BenchmarkId::new("read_heavy", n_threads),
            n_threads,
            |bench, &n_threads| {
                bench.iter(|| {
                    let space = Arc::new(SimulatedSpace::new());
                    // Pre-populate channels
                    for t in 0..n_threads {
                        let channel = format!("channel_{}", t);
                        for i in 0..100 {
                            space.produce(&channel, i as i32);
                        }
                    }

                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let space = Arc::clone(&space);
                            thread::spawn(move || {
                                let channel = format!("channel_{}", t);
                                for i in 0..n_ops {
                                    if i % 10 == 0 {
                                        // 10% writes
                                        space.produce(&channel, i as i32);
                                    } else {
                                        // 90% reads
                                        let _ = space.peek(&channel);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_registry_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_lookup");

    // Simulate registry with many spaces
    for n_spaces in [10, 100, 1000].iter() {
        let registry: Arc<RwLock<HashMap<String, i32>>> = Arc::new(RwLock::new(
            (0..*n_spaces).map(|i| (format!("space_{}", i), i as i32)).collect()
        ));

        group.throughput(Throughput::Elements(*n_spaces as u64));

        // Sequential lookups
        group.bench_with_input(
            BenchmarkId::new("sequential_lookup", n_spaces),
            n_spaces,
            |bench, &n_spaces| {
                bench.iter(|| {
                    let reg = registry.read().expect("read lock");
                    for i in 0..n_spaces {
                        let _ = black_box(reg.get(&format!("space_{}", i)));
                    }
                });
            },
        );

        // Concurrent lookups from multiple threads
        let n_threads = 4;
        group.bench_with_input(
            BenchmarkId::new("concurrent_lookup_4_threads", n_spaces),
            n_spaces,
            |bench, &n_spaces| {
                bench.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let reg = Arc::clone(&registry);
                            thread::spawn(move || {
                                let reader = reg.read().expect("read lock");
                                let start = (t * n_spaces) / n_threads;
                                let end = ((t + 1) * n_spaces) / n_threads;
                                for i in start..end {
                                    let _ = black_box(reader.get(&format!("space_{}", i % n_spaces)));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Pattern Matcher Benchmarks (Phase 0: Baseline)
// =============================================================================

use rholang::rust::interpreter::spaces::{
    Match, ExactMatch, VectorDBMatch, WildcardMatch,
};

fn bench_matcher_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matcher_comparison");

    // Test data: varying sizes of data to match
    for data_count in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*data_count as u64));

        // ExactMatch - simple equality
        group.bench_with_input(
            BenchmarkId::new("exact_match", data_count),
            data_count,
            |b, &count| {
                let matcher: ExactMatch<i32> = ExactMatch::new();
                let pattern = 500; // Middle value
                let data: Vec<i32> = (0..count).collect();

                b.iter(|| {
                    for d in &data {
                        black_box(matcher.matches(&pattern, d));
                    }
                });
            },
        );

        // WildcardMatch - always true (baseline for comparison)
        group.bench_with_input(
            BenchmarkId::new("wildcard_match", data_count),
            data_count,
            |b, &count| {
                let matcher: WildcardMatch<i32, i32> = WildcardMatch::new();
                let pattern = 500;
                let data: Vec<i32> = (0..count).collect();

                b.iter(|| {
                    for d in &data {
                        black_box(matcher.matches(&pattern, d));
                    }
                });
            },
        );
    }

    // VectorDB matcher with different dimensions
    for dim in [128, 384].iter() {
        let query = random_vector(*dim);
        let data: Vec<Vec<f32>> = (0..100)
            .map(|_| random_vector(*dim).to_vec())
            .collect();

        group.throughput(Throughput::Elements(100));

        group.bench_with_input(
            BenchmarkId::new("vectordb_match", dim),
            dim,
            |b, _| {
                let matcher = VectorDBMatch::new(0.8);
                let pattern = query.to_vec();

                b.iter(|| {
                    for d in &data {
                        black_box(matcher.matches(&pattern, d));
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Checkpoint/Replay Benchmarks (Phase 0: Baseline)
// =============================================================================

use rholang::rust::interpreter::spaces::{
    GenericRSpace, CheckpointableSpace, SpaceId, SpaceQualifier, SpaceAgent,
};
use rholang::rust::interpreter::spaces::channel_store::HashMapChannelStore;
use rholang::rust::interpreter::spaces::collections::BagContinuationCollection;
use serde::{Serialize, Deserialize};

/// Simple channel type that implements From<usize> for gensym.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
struct BenchChannel(usize);

impl From<usize> for BenchChannel {
    fn from(n: usize) -> Self {
        BenchChannel(n)
    }
}

impl AsRef<[u8]> for BenchChannel {
    fn as_ref(&self) -> &[u8] {
        // Not ideal but sufficient for benchmark purposes
        unsafe { std::slice::from_raw_parts(&self.0 as *const usize as *const u8, std::mem::size_of::<usize>()) }
    }
}

/// Type alias for the test space used in benchmarks.
type BenchSpace = GenericRSpace<
    HashMapChannelStore<
        BenchChannel,
        i32,  // Pattern
        i32,  // Data
        String,  // Continuation
        BagDataCollection<i32>,
        BagContinuationCollection<i32, String>,
    >,
    WildcardMatch<i32, i32>,
>;

/// Create a new benchmark space.
fn new_bench_space() -> BenchSpace {
    let store = HashMapChannelStore::new(
        BagDataCollection::new,
        BagContinuationCollection::new,
    );
    let matcher = WildcardMatch::new();
    GenericRSpace::new(store, matcher, SpaceId::default_space(), SpaceQualifier::Default)
}

fn bench_checkpoint_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_operations");

    // Benchmark checkpoint creation with varying space sizes
    for n_channels in [10, 100, 1_000].iter() {
        let items_per_channel = 10;

        group.throughput(Throughput::Elements((*n_channels * items_per_channel) as u64));

        // Soft checkpoint creation (in-memory, fast)
        group.bench_with_input(
            BenchmarkId::new("soft_checkpoint_create", n_channels),
            n_channels,
            |b, &n_channels| {
                b.iter_batched(
                    || {
                        // Setup: Create space with data
                        let mut space = new_bench_space();

                        // Populate with test data
                        for ch in 0..n_channels {
                            for i in 0..items_per_channel {
                                let channel = BenchChannel(ch);
                                let _ = space.produce(channel, (ch * items_per_channel + i) as i32, false, None);
                            }
                        }
                        space
                    },
                    |mut space| {
                        black_box(space.create_soft_checkpoint());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Soft checkpoint revert (restore from snapshot)
        group.bench_with_input(
            BenchmarkId::new("soft_checkpoint_revert", n_channels),
            n_channels,
            |b, &n_channels| {
                b.iter_batched(
                    || {
                        // Setup: Create space, checkpoint, then modify
                        let mut space = new_bench_space();

                        // Populate with test data
                        for ch in 0..n_channels {
                            for i in 0..items_per_channel {
                                let channel = BenchChannel(ch);
                                let _ = space.produce(channel, (ch * items_per_channel + i) as i32, false, None);
                            }
                        }

                        let checkpoint = space.create_soft_checkpoint();

                        // Modify after checkpoint
                        for ch in 0..n_channels {
                            let channel = BenchChannel(ch);
                            let _ = space.produce(channel, 99999, false, None);
                        }

                        (space, checkpoint)
                    },
                    |(mut space, checkpoint)| {
                        let _ = black_box(space.revert_to_soft_checkpoint(checkpoint));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// End-to-End Produce/Consume Cycles (Phase 0: Baseline)
// =============================================================================

fn bench_produce_consume_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("produce_consume_cycle");

    // Single channel produce/consume throughput
    for n_ops in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*n_ops as u64));

        // Produce only (no matching continuations)
        group.bench_with_input(
            BenchmarkId::new("produce_only", n_ops),
            n_ops,
            |b, &n_ops| {
                b.iter_batched(
                    new_bench_space,
                    |mut space| {
                        for i in 0..n_ops {
                            let channel = BenchChannel(0);
                            let _ = space.produce(channel, i as i32, false, None);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Consume from filled channel (immediate match)
        group.bench_with_input(
            BenchmarkId::new("consume_immediate_match", n_ops),
            n_ops,
            |b, &n_ops| {
                b.iter_batched(
                    || {
                        let mut space = new_bench_space();
                        // Pre-fill with data
                        for i in 0..n_ops {
                            let channel = BenchChannel(0);
                            let _ = space.produce(channel, i as i32, false, None);
                        }
                        space
                    },
                    |mut space| {
                        for _ in 0..n_ops {
                            let channel = BenchChannel(0);
                            let _ = space.consume(
                                vec![channel],           // channels
                                vec![0],                 // patterns (wildcard matches anything)
                                "continuation".to_string(),
                                false,
                                std::collections::BTreeSet::new(),  // no peeks
                            );
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Multiple channels (tests channel lookup overhead)
    for n_channels in [10, 100, 1_000].iter() {
        group.throughput(Throughput::Elements(*n_channels as u64));

        group.bench_with_input(
            BenchmarkId::new("produce_multi_channel", n_channels),
            n_channels,
            |b, &n_channels| {
                b.iter_batched(
                    new_bench_space,
                    |mut space| {
                        for ch in 0..n_channels {
                            let channel = BenchChannel(ch);
                            let _ = space.produce(channel, ch as i32, false, None);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// Registry Scaling Benchmarks (Phase 0: Baseline)
// =============================================================================

use dashmap::DashMap;

fn bench_registry_at_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_scale");

    // Test with larger registry sizes to measure scaling behavior
    for n_spaces in [100, 1_000, 10_000].iter() {
        // RwLock<HashMap> baseline (current implementation pattern)
        let rwlock_registry: Arc<RwLock<HashMap<String, i32>>> = Arc::new(RwLock::new(
            (0..*n_spaces).map(|i| (format!("space_{}", i), i as i32)).collect()
        ));

        // DashMap comparison (proposed optimization)
        let dashmap_registry: Arc<DashMap<String, i32>> = Arc::new(
            (0..*n_spaces).map(|i| (format!("space_{}", i), i as i32)).collect()
        );

        group.throughput(Throughput::Elements(*n_spaces as u64));

        // Sequential lookup - RwLock
        group.bench_with_input(
            BenchmarkId::new("rwlock_sequential", n_spaces),
            n_spaces,
            |b, &n_spaces| {
                b.iter(|| {
                    let reg = rwlock_registry.read().expect("read lock");
                    for i in 0..n_spaces {
                        let _ = black_box(reg.get(&format!("space_{}", i)));
                    }
                });
            },
        );

        // Sequential lookup - DashMap
        group.bench_with_input(
            BenchmarkId::new("dashmap_sequential", n_spaces),
            n_spaces,
            |b, &n_spaces| {
                b.iter(|| {
                    for i in 0..n_spaces {
                        let _ = black_box(dashmap_registry.get(&format!("space_{}", i)));
                    }
                });
            },
        );

        // Concurrent lookup - RwLock (8 threads)
        let n_threads = 8;
        group.bench_with_input(
            BenchmarkId::new("rwlock_concurrent_8t", n_spaces),
            n_spaces,
            |b, &n_spaces| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let reg = Arc::clone(&rwlock_registry);
                            thread::spawn(move || {
                                let reader = reg.read().expect("read lock");
                                let start = (t * n_spaces) / n_threads;
                                let end = ((t + 1) * n_spaces) / n_threads;
                                for i in start..end {
                                    let _ = black_box(reader.get(&format!("space_{}", i % n_spaces)));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );

        // Concurrent lookup - DashMap (8 threads)
        group.bench_with_input(
            BenchmarkId::new("dashmap_concurrent_8t", n_spaces),
            n_spaces,
            |b, &n_spaces| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let reg = Arc::clone(&dashmap_registry);
                            thread::spawn(move || {
                                let start = (t * n_spaces) / n_threads;
                                let end = ((t + 1) * n_spaces) / n_threads;
                                for i in start..end {
                                    let _ = black_box(reg.get(&format!("space_{}", i % n_spaces)));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );

        // Mixed read/write workload - DashMap
        group.bench_with_input(
            BenchmarkId::new("dashmap_mixed_rw_8t", n_spaces),
            n_spaces,
            |b, &n_spaces| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let reg = Arc::clone(&dashmap_registry);
                            thread::spawn(move || {
                                let start = (t * n_spaces) / n_threads;
                                let end = ((t + 1) * n_spaces) / n_threads;
                                for i in start..end {
                                    let key = format!("space_{}", i % n_spaces);
                                    if i % 10 == 0 {
                                        // 10% writes
                                        reg.insert(key, (i * 2) as i32);
                                    } else {
                                        // 90% reads
                                        let _ = black_box(reg.get(&key));
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("thread join");
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Atomic Ordering Benchmark (Phase 0: Baseline for Phase 1 optimization)
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

fn bench_atomic_orderings(c: &mut Criterion) {
    let mut group = c.benchmark_group("atomic_ordering");

    let n_ops = 10_000;
    group.throughput(Throughput::Elements(n_ops as u64));

    // SeqCst ordering (current implementation)
    group.bench_function("seqcst_fetch_add", |b| {
        let counter = AtomicU64::new(0);
        b.iter(|| {
            for _ in 0..n_ops {
                black_box(counter.fetch_add(1, Ordering::SeqCst));
            }
            counter.store(0, Ordering::Relaxed);
        });
    });

    // Release ordering (proposed optimization)
    group.bench_function("release_fetch_add", |b| {
        let counter = AtomicU64::new(0);
        b.iter(|| {
            for _ in 0..n_ops {
                black_box(counter.fetch_add(1, Ordering::Release));
            }
            counter.store(0, Ordering::Relaxed);
        });
    });

    // Relaxed ordering (maximum performance)
    group.bench_function("relaxed_fetch_add", |b| {
        let counter = AtomicU64::new(0);
        b.iter(|| {
            for _ in 0..n_ops {
                black_box(counter.fetch_add(1, Ordering::Relaxed));
            }
            counter.store(0, Ordering::Relaxed);
        });
    });

    // Compare-exchange patterns (phlogiston charging pattern)
    group.bench_function("seqcst_compare_exchange", |b| {
        let balance = AtomicU64::new(u64::MAX);
        b.iter(|| {
            for _ in 0..n_ops {
                let current = balance.load(Ordering::SeqCst);
                let _ = black_box(balance.compare_exchange_weak(
                    current,
                    current.saturating_sub(100),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ));
            }
        });
    });

    group.bench_function("release_compare_exchange", |b| {
        let balance = AtomicU64::new(u64::MAX);
        b.iter(|| {
            for _ in 0..n_ops {
                let current = balance.load(Ordering::Acquire);
                let _ = black_box(balance.compare_exchange_weak(
                    current,
                    current.saturating_sub(100),
                    Ordering::Release,
                    Ordering::Relaxed,
                ));
            }
        });
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    // Existing benchmarks
    bench_collection_types,
    bench_collection_retrieval,
    bench_phlogiston,
    bench_vectordb_cosine_similarity,
    bench_vectordb_top_k_selection,
    bench_vectordb_batch_cosine,
    bench_concurrent_same_channel,
    bench_concurrent_different_channels,
    bench_registry_lookup,
    // Phase 0 baseline benchmarks
    bench_matcher_comparison,
    bench_checkpoint_operations,
    bench_produce_consume_cycle,
    bench_registry_at_scale,
    bench_atomic_orderings,
);

criterion_main!(benches);
