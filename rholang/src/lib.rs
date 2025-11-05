use std::sync::atomic::{AtomicUsize, Ordering};

// Tracks total bytes currently allocated and leaked to JNA callers in rholang lib
static RHOLANG_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "C" fn rholang_get_allocated_bytes() -> usize {
    RHOLANG_ALLOCATED_BYTES.load(Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn rholang_reset_allocated_bytes() {
    RHOLANG_ALLOCATED_BYTES.store(0, Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn rholang_deallocate_memory(ptr: *mut u8, len: usize) {
    unsafe {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
    }
    RHOLANG_ALLOCATED_BYTES.fetch_sub(len, Ordering::SeqCst);
}
pub mod rust {
    pub mod build;
    pub mod interpreter;
}

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::rust::interpreter::compiler::compiler::Compiler;
use crypto::rust::hash::blake2b512_block::Blake2b512Block;
use crypto::rust::{hash::blake2b512_random::Blake2b512Random, public_key::PublicKey};
use models::rspace_plus_plus_types::*;
use models::{
    rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation},
    rholang_scala_rust_types::*,
};
use prost::Message;
use rspace_plus_plus::rspace::checkpoint::SoftCheckpoint;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::hot_store::{new_dashmap, HotStoreState};
use rspace_plus_plus::rspace::internal::{Datum, WaitingContinuation};
use rspace_plus_plus::rspace::replay_rspace::ReplayRSpace;
use rspace_plus_plus::rspace::trace::event::{Consume, Produce, COMM};
use rspace_plus_plus::rspace::{
    rspace::RSpace,
    trace::event::{Event, IOEvent},
};
use rust::interpreter::env::Env;
use rust::interpreter::system_processes::test_framework_contracts;
use rust::interpreter::{
    accounting::costs::Cost,
    rho_runtime::{
        bootstrap_registry as bootstrap_registry_internal, create_rho_runtime,
        RhoRuntime as RhoRuntimeTrait, RhoRuntimeImpl,
    },
    system_processes::BlockData,
};

#[repr(C)]
struct RhoRuntime {
    runtime: RhoRuntimeImpl,
}

#[repr(C)]
struct ReplayRhoRuntime {
    runtime: RhoRuntimeImpl,
}

#[repr(C)]
struct Space {
    rspace: Mutex<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>,
}

#[repr(C)]
struct ReplaySpace {
    replay_space: Mutex<ReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>,
}

/* RHO RUNTIME */

#[no_mangle]
extern "C" fn evaluate(
    runtime_ptr: *mut RhoRuntime,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> *const u8 {
    // println!("\nhit rust lib evaluate");

    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = EvaluateParams::decode(params_slice).unwrap();

    let term = params.term;
    let cost_proto = params.initial_phlo.unwrap();
    let initial_phlo = Cost::create(cost_proto.value.into(), cost_proto.operation);
    let normalizer_env = params.normalizer_env;
    let rand_proto = params.random_state.unwrap();
    let digest_proto = rand_proto.digest.unwrap();
    // println!(
    //     "\nrandPathPosition in rust evaluate: {}",
    //     rand_proto.path_position
    // );
    let rand = Blake2b512Random {
        digest: Blake2b512Block {
            chain_value: digest_proto
                .chain_value
                .into_iter()
                .map(|v| v.value)
                .collect(),
            t0: digest_proto.t0,
            t1: digest_proto.t1,
        },
        last_block: rand_proto.last_block.into_iter().map(|v| v as i8).collect(),
        path_view: rand_proto.path_view,
        count_view: rand_proto.count_view.into_iter().map(|v| v.value).collect(),
        hash_array: {
            let mut array = [0i8; 64];
            let vec = rand_proto.hash_array;
            let i8_slice: &[i8] = unsafe { std::mem::transmute(&vec[..64]) };
            array.copy_from_slice(i8_slice);
            array
        },
        position: rand_proto.position,
        path_position: rand_proto.path_position as usize,
    };
    // println!("\nrand in rust evaluate: ");
    // rand.debug_str();

    let rho_runtime = unsafe { &mut (*runtime_ptr).runtime };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let eval_result = rt.block_on(async {
        rho_runtime
            .evaluate(
                &term,
                initial_phlo,
                normalizer_env.into_iter().collect(),
                rand,
            )
            .await
            .unwrap()
    });

    // println!("\neval_result: {:?}", eval_result);

    let eval_result_proto = EvaluateResultProto {
        cost: Some(CostProto {
            value: eval_result.cost.value,
            operation: eval_result.cost.operation,
        }),
        errors: eval_result
            .errors
            .into_iter()
            .map(|err| err.to_string())
            .collect(),
        mergeable: eval_result.mergeable.into_iter().collect(),
    };

    let mut bytes = eval_result_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn inj(
    runtime_ptr: *mut RhoRuntime,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> () {
    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = InjParams::decode(params_slice).unwrap();

    let par = params.par.unwrap();
    let env_proto = params.env.unwrap();
    let env = Env {
        env_map: env_proto.env_map.into_iter().collect(),
        level: env_proto.level,
        shift: env_proto.shift,
    };

    let rand_proto = params.rand.unwrap();
    let digest_proto = rand_proto.digest.unwrap();
    let rand = Blake2b512Random {
        digest: Blake2b512Block {
            chain_value: digest_proto
                .chain_value
                .into_iter()
                .map(|v| v.value)
                .collect(),
            t0: digest_proto.t0,
            t1: digest_proto.t1,
        },
        last_block: rand_proto.last_block.into_iter().map(|v| v as i8).collect(),
        path_view: rand_proto.path_view,
        count_view: rand_proto.count_view.into_iter().map(|v| v.value).collect(),
        hash_array: {
            let mut array = [0i8; 64];
            let vec = rand_proto.hash_array;
            let i8_slice: &[i8] = unsafe { std::mem::transmute(&vec[..64]) };
            array.copy_from_slice(i8_slice);
            array
        },
        position: rand_proto.position,
        path_position: rand_proto.path_position as usize,
    };

    // println!("\nrand in rust inj: ");
    // rand.debug_str();

    let rho_runtime = unsafe { &(*runtime_ptr).runtime };
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async { rho_runtime.inj(par, env, rand).await.unwrap() })
}

#[no_mangle]
extern "C" fn create_soft_checkpoint(runtime_ptr: *mut RhoRuntime) -> *const u8 {
    // println!("\nhit rust lib create_soft_checkpoint");
    let runtime = unsafe { &mut (*runtime_ptr).runtime };
    let soft_checkpoint = runtime.create_soft_checkpoint();

    let mut conts_map_entries: Vec<StoreStateContMapEntry> = Vec::new();
    let mut installed_conts_map_entries: Vec<StoreStateInstalledContMapEntry> = Vec::new();
    let mut data_map_entries: Vec<StoreStateDataMapEntry> = Vec::new();
    let mut joins_map_entries: Vec<StoreStateJoinsMapEntry> = Vec::new();
    let mut installed_joins_map_entries: Vec<StoreStateInstalledJoinsMapEntry> = Vec::new();

    let hot_store_state = soft_checkpoint.cache_snapshot;

    for (key, value) in hot_store_state.continuations.clone().into_iter() {
        let wks: Vec<WaitingContinuationProto> = value
            .into_iter()
            .map(|wk| {
                let res = WaitingContinuationProto {
                    patterns: wk.patterns,
                    continuation: Some(wk.continuation.clone()),
                    persist: wk.persist,
                    peeks: wk
                        .peeks
                        .into_iter()
                        .map(|peek| SortedSetElement { value: peek as i32 })
                        .collect(),
                    source: Some(ConsumeProto {
                        channel_hashes: wk
                            .source
                            .channel_hashes
                            .iter()
                            .map(|hash| hash.bytes())
                            .collect(),
                        hash: wk.source.hash.bytes(),
                        persistent: wk.source.persistent,
                    }),
                };

                res
            })
            .collect();

        conts_map_entries.push(StoreStateContMapEntry { key, value: wks });
    }

    for (key, value) in hot_store_state.installed_continuations.clone().into_iter() {
        let wk = WaitingContinuationProto {
            patterns: value.patterns,
            continuation: Some(value.continuation.clone()),
            persist: value.persist,
            peeks: value
                .peeks
                .into_iter()
                .map(|peek| SortedSetElement { value: peek as i32 })
                .collect(),
            source: Some(ConsumeProto {
                channel_hashes: value
                    .source
                    .channel_hashes
                    .iter()
                    .map(|hash| hash.bytes())
                    .collect(),
                hash: value.source.hash.bytes(),
                persistent: value.source.persistent,
            }),
        };

        installed_conts_map_entries.push(StoreStateInstalledContMapEntry {
            key,
            value: Some(wk),
        });
    }

    for (key, value) in hot_store_state.data.clone().into_iter() {
        let datums = value
            .into_iter()
            .map(|datum| DatumProto {
                a: Some(datum.a),
                persist: datum.persist,
                source: Some(ProduceProto {
                    channel_hash: datum.source.channel_hash.bytes(),
                    hash: datum.source.hash.bytes(),
                    persistent: datum.source.persistent,
                    is_deterministic: datum.source.is_deterministic,
                    output_value: datum.source.output_value,
                }),
            })
            .collect();

        data_map_entries.push(StoreStateDataMapEntry {
            key: Some(key),
            value: datums,
        });
    }

    for (key, value) in hot_store_state.joins.clone().into_iter() {
        let joins = value.into_iter().map(|join| JoinProto { join }).collect();

        joins_map_entries.push(StoreStateJoinsMapEntry {
            key: Some(key),
            value: joins,
        });
    }

    for (key, value) in hot_store_state.installed_joins.clone().into_iter() {
        let joins = value.into_iter().map(|join| JoinProto { join }).collect();

        installed_joins_map_entries.push(StoreStateInstalledJoinsMapEntry {
            key: Some(key),
            value: joins,
        });
    }

    let hot_store_state_proto = HotStoreStateProto {
        continuations: conts_map_entries,
        installed_continuations: installed_conts_map_entries,
        data: data_map_entries,
        joins: joins_map_entries,
        installed_joins: installed_joins_map_entries,
    };

    let log = soft_checkpoint.log;
    let log_proto: Vec<EventProto> = log
        .into_iter()
        .map(|event| match event {
            Event::Comm(comm) => {
                let comm_proto = CommProto {
                    consume: {
                        Some(ConsumeProto {
                            channel_hashes: comm
                                .consume
                                .channel_hashes
                                .iter()
                                .map(|hash| hash.bytes())
                                .collect(),
                            hash: comm.consume.hash.bytes(),
                            persistent: comm.consume.persistent,
                        })
                    },
                    produces: {
                        comm.produces
                            .into_iter()
                            .map(|produce| ProduceProto {
                                channel_hash: produce.channel_hash.bytes(),
                                hash: produce.hash.bytes(),
                                persistent: produce.persistent,
                                is_deterministic: produce.is_deterministic,
                                output_value: produce.output_value,
                            })
                            .collect()
                    },
                    peeks: {
                        comm.peeks
                            .into_iter()
                            .map(|peek| SortedSetElement { value: peek as i32 })
                            .collect()
                    },
                    times_repeated: {
                        let mut produce_counter_map_entries: Vec<ProduceCounterMapEntry> =
                            Vec::new();
                        for (key, value) in comm.times_repeated {
                            let produce = ProduceProto {
                                channel_hash: key.channel_hash.bytes(),
                                hash: key.hash.bytes(),
                                persistent: key.persistent,
                                is_deterministic: key.is_deterministic,
                                output_value: key.output_value,
                            };

                            produce_counter_map_entries.push(ProduceCounterMapEntry {
                                key: Some(produce),
                                value,
                            });
                        }
                        produce_counter_map_entries
                    },
                };

                EventProto {
                    event_type: Some(event_proto::EventType::Comm(comm_proto)),
                }
            }
            Event::IoEvent(io_event) => match io_event {
                IOEvent::Produce(produce) => {
                    let produce_proto = ProduceProto {
                        channel_hash: produce.channel_hash.bytes(),
                        hash: produce.hash.bytes(),
                        persistent: produce.persistent,
                        is_deterministic: produce.is_deterministic,
                        output_value: produce.output_value,
                    };
                    EventProto {
                        event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
                            io_event_type: Some(io_event_proto::IoEventType::Produce(
                                produce_proto,
                            )),
                        })),
                    }
                }
                IOEvent::Consume(consume) => {
                    let consume_proto = ConsumeProto {
                        channel_hashes: consume
                            .channel_hashes
                            .iter()
                            .map(|hash| hash.bytes())
                            .collect(),
                        hash: consume.hash.bytes(),
                        persistent: consume.persistent,
                    };
                    EventProto {
                        event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
                            io_event_type: Some(io_event_proto::IoEventType::Consume(
                                consume_proto,
                            )),
                        })),
                    }
                }
            },
        })
        .collect();

    let mut produce_counter_map_entries: Vec<ProduceCounterMapEntry> = Vec::new();
    let produce_counter_map = soft_checkpoint.produce_counter;

    for (key, value) in produce_counter_map {
        let produce = ProduceProto {
            channel_hash: key.channel_hash.bytes(),
            hash: key.hash.bytes(),
            persistent: key.persistent,
            is_deterministic: key.is_deterministic,
            output_value: key.output_value,
        };

        produce_counter_map_entries.push(ProduceCounterMapEntry {
            key: Some(produce),
            value,
        });
    }

    let soft_checkpoint_proto = SoftCheckpointProto {
        cache_snapshot: Some(hot_store_state_proto),
        log: log_proto,
        produce_counter: produce_counter_map_entries,
    };

    let mut bytes = soft_checkpoint_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn revert_to_soft_checkpoint(
    runtime_ptr: *mut RhoRuntime,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> () {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let soft_checkpoint_proto = SoftCheckpointProto::decode(payload_slice).unwrap();
    let cache_snapshot_proto = soft_checkpoint_proto.cache_snapshot.unwrap();

    let conts_map = new_dashmap();
    for map_entry in cache_snapshot_proto.continuations {
        let key = map_entry.key;
        let value = map_entry
            .value
            .into_iter()
            .map(|cont_proto| WaitingContinuation {
                patterns: cont_proto.patterns,
                continuation: cont_proto.continuation.unwrap(),
                persist: cont_proto.persist,
                peeks: cont_proto
                    .peeks
                    .iter()
                    .map(|element| element.value)
                    .collect(),
                source: {
                    let consume_proto = cont_proto.source.unwrap();
                    Consume {
                        channel_hashes: consume_proto
                            .channel_hashes
                            .iter()
                            .map(|hash_bytes| Blake2b256Hash::from_bytes(hash_bytes.to_vec()))
                            .collect(),
                        hash: Blake2b256Hash::from_bytes(consume_proto.hash),
                        persistent: consume_proto.persistent,
                    }
                },
            })
            .collect();

        conts_map.insert(key, value);
    }

    let installed_conts_map = new_dashmap();
    for map_entry in cache_snapshot_proto.installed_continuations {
        let key = map_entry.key;
        let wk_proto = map_entry.value.unwrap();
        let value = WaitingContinuation {
            patterns: wk_proto.patterns,
            continuation: wk_proto.continuation.unwrap(),
            persist: wk_proto.persist,
            peeks: wk_proto.peeks.iter().map(|element| element.value).collect(),
            source: {
                let consume_proto = wk_proto.source.unwrap();
                Consume {
                    channel_hashes: consume_proto
                        .channel_hashes
                        .iter()
                        .map(|hash_bytes| Blake2b256Hash::from_bytes(hash_bytes.to_vec()))
                        .collect(),
                    hash: Blake2b256Hash::from_bytes(consume_proto.hash),
                    persistent: consume_proto.persistent,
                }
            },
        };

        installed_conts_map.insert(key, value);
    }

    let datums_map = new_dashmap();
    for map_entry in cache_snapshot_proto.data {
        let key = map_entry.key.unwrap();
        let value = map_entry
            .value
            .into_iter()
            .map(|datum_proto| Datum {
                a: datum_proto.a.unwrap(),
                persist: datum_proto.persist,
                source: {
                    let produce_proto = datum_proto.source.unwrap();
                    Produce {
                        channel_hash: Blake2b256Hash::from_bytes(produce_proto.channel_hash),
                        hash: Blake2b256Hash::from_bytes(produce_proto.hash),
                        persistent: produce_proto.persistent,
                        is_deterministic: produce_proto.is_deterministic,
                        output_value: produce_proto.output_value,
                    }
                },
            })
            .collect();

        datums_map.insert(key, value);
    }

    let joins_map = new_dashmap();
    for map_entry in cache_snapshot_proto.joins {
        let key = map_entry.key.unwrap();
        let value = map_entry
            .value
            .into_iter()
            .map(|join_proto| join_proto.join)
            .collect();

        joins_map.insert(key, value);
    }

    let installed_joins_map = new_dashmap();
    for map_entry in cache_snapshot_proto.installed_joins {
        let key = map_entry.key.unwrap();
        let value = map_entry
            .value
            .into_iter()
            .map(|join_proto| join_proto.join)
            .collect();

        installed_joins_map.insert(key, value);
    }

    let log: Vec<Event> = soft_checkpoint_proto
        .log
        .into_iter()
        .map(|log_entry| match log_entry.event_type.unwrap() {
            event_proto::EventType::Comm(comm_proto) => {
                let consume_proto = comm_proto.consume.unwrap();
                let comm = COMM {
                    consume: {
                        Consume {
                            channel_hashes: {
                                consume_proto
                                    .channel_hashes
                                    .iter()
                                    .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
                                    .collect()
                            },
                            hash: Blake2b256Hash::from_bytes(consume_proto.hash),
                            persistent: consume_proto.persistent,
                        }
                    },
                    produces: {
                        comm_proto
                            .produces
                            .into_iter()
                            .map(|produce_proto| Produce {
                                channel_hash: Blake2b256Hash::from_bytes(
                                    produce_proto.channel_hash,
                                ),
                                hash: Blake2b256Hash::from_bytes(produce_proto.hash),
                                persistent: produce_proto.persistent,
                                is_deterministic: produce_proto.is_deterministic,
                                output_value: produce_proto.output_value,
                            })
                            .collect()
                    },
                    peeks: {
                        comm_proto
                            .peeks
                            .iter()
                            .map(|element| element.value)
                            .collect()
                    },
                    times_repeated: {
                        comm_proto
                            .times_repeated
                            .into_iter()
                            .map(|map_entry| {
                                let key_proto = map_entry.key.unwrap();
                                let produce = Produce {
                                    channel_hash: Blake2b256Hash::from_bytes(
                                        key_proto.channel_hash,
                                    ),
                                    hash: Blake2b256Hash::from_bytes(key_proto.hash),
                                    persistent: key_proto.persistent,
                                    is_deterministic: key_proto.is_deterministic,
                                    output_value: key_proto.output_value,
                                };

                                let value = map_entry.value;

                                (produce, value)
                            })
                            .collect()
                    },
                };
                Event::Comm(comm)
            }
            event_proto::EventType::IoEvent(io_event) => match io_event.io_event_type.unwrap() {
                io_event_proto::IoEventType::Produce(produce_proto) => {
                    let produce = Produce {
                        channel_hash: Blake2b256Hash::from_bytes(produce_proto.channel_hash),
                        hash: Blake2b256Hash::from_bytes(produce_proto.hash),
                        persistent: produce_proto.persistent,
                        is_deterministic: produce_proto.is_deterministic,
                        output_value: produce_proto.output_value,
                    };
                    Event::IoEvent(IOEvent::Produce(produce))
                }
                io_event_proto::IoEventType::Consume(consume_proto) => {
                    let consume = Consume {
                        channel_hashes: {
                            consume_proto
                                .channel_hashes
                                .iter()
                                .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
                                .collect()
                        },
                        hash: Blake2b256Hash::from_bytes(consume_proto.hash),
                        persistent: consume_proto.persistent,
                    };
                    Event::IoEvent(IOEvent::Consume(consume))
                }
            },
        })
        .collect();

    let produce_counter_map: BTreeMap<Produce, i32> = soft_checkpoint_proto
        .produce_counter
        .into_iter()
        .map(|map_entry| {
            let key_proto = map_entry.key.unwrap();
            let produce = Produce {
                channel_hash: Blake2b256Hash::from_bytes(key_proto.channel_hash),
                hash: Blake2b256Hash::from_bytes(key_proto.hash),
                persistent: key_proto.persistent,
                is_deterministic: key_proto.is_deterministic,
                output_value: key_proto.output_value,
            };

            let value = map_entry.value;

            (produce, value)
        })
        .collect();

    let cache_snapshot = HotStoreState {
        continuations: conts_map,
        installed_continuations: installed_conts_map,
        data: datums_map,
        joins: joins_map,
        installed_joins: installed_joins_map,
    };

    let soft_checkpoint = SoftCheckpoint {
        cache_snapshot,
        log,
        produce_counter: produce_counter_map,
    };

    let runtime = unsafe { &mut (*runtime_ptr).runtime };

    runtime.revert_to_soft_checkpoint(soft_checkpoint);
}

#[no_mangle]
extern "C" fn create_checkpoint(runtime_ptr: *mut RhoRuntime) -> *const u8 {
    let runtime = unsafe { &mut (*runtime_ptr).runtime };
    let checkpoint = runtime.create_checkpoint();

    let log = checkpoint.log;
    let log_proto: Vec<EventProto> = log
        .into_iter()
        .map(|event| match event {
            Event::Comm(comm) => {
                let comm_proto = CommProto {
                    consume: {
                        Some(ConsumeProto {
                            channel_hashes: comm
                                .consume
                                .channel_hashes
                                .iter()
                                .map(|hash| hash.bytes())
                                .collect(),
                            hash: comm.consume.hash.bytes(),
                            persistent: comm.consume.persistent,
                        })
                    },
                    produces: {
                        comm.produces
                            .into_iter()
                            .map(|produce| ProduceProto {
                                channel_hash: produce.channel_hash.bytes(),
                                hash: produce.hash.bytes(),
                                persistent: produce.persistent,
                                is_deterministic: produce.is_deterministic,
                                output_value: produce.output_value,
                            })
                            .collect()
                    },
                    peeks: {
                        comm.peeks
                            .into_iter()
                            .map(|peek| SortedSetElement { value: peek as i32 })
                            .collect()
                    },
                    times_repeated: {
                        let mut produce_counter_map_entries: Vec<ProduceCounterMapEntry> =
                            Vec::new();
                        for (key, value) in comm.times_repeated {
                            let produce = ProduceProto {
                                channel_hash: key.channel_hash.bytes(),
                                hash: key.hash.bytes(),
                                persistent: key.persistent,
                                is_deterministic: key.is_deterministic,
                                output_value: key.output_value,
                            };

                            produce_counter_map_entries.push(ProduceCounterMapEntry {
                                key: Some(produce),
                                value,
                            });
                        }
                        produce_counter_map_entries
                    },
                };

                EventProto {
                    event_type: Some(event_proto::EventType::Comm(comm_proto)),
                }
            }
            Event::IoEvent(io_event) => match io_event {
                IOEvent::Produce(produce) => {
                    let produce_proto = ProduceProto {
                        channel_hash: produce.channel_hash.bytes(),
                        hash: produce.hash.bytes(),
                        persistent: produce.persistent,
                        is_deterministic: produce.is_deterministic,
                        output_value: produce.output_value,
                    };
                    EventProto {
                        event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
                            io_event_type: Some(io_event_proto::IoEventType::Produce(
                                produce_proto,
                            )),
                        })),
                    }
                }
                IOEvent::Consume(consume) => {
                    let consume_proto = ConsumeProto {
                        channel_hashes: consume
                            .channel_hashes
                            .iter()
                            .map(|hash| hash.bytes())
                            .collect(),
                        hash: consume.hash.bytes(),
                        persistent: consume.persistent,
                    };
                    EventProto {
                        event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
                            io_event_type: Some(io_event_proto::IoEventType::Consume(
                                consume_proto,
                            )),
                        })),
                    }
                }
            },
        })
        .collect();

    let checkpoint_proto = CheckpointProto {
        root: checkpoint.root.bytes(),
        log: log_proto,
    };

    let mut bytes = checkpoint_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn consume_result(
    runtime_ptr: *mut RhoRuntime,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> *const u8 {
    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let consume_result_params = ConsumeResultParams::decode(params_slice).unwrap();

    let channel = consume_result_params.channel;
    let pattern = consume_result_params.pattern;

    let consume_result_return = unsafe {
        (*runtime_ptr)
            .runtime
            .consume_result(channel, pattern)
            .unwrap()
    };

    match consume_result_return {
        None => std::ptr::null(),
        Some((tagged_cont, datums)) => {
            let consume_result_return_proto = ConsumeResultReturn {
                tagged_cont: Some(tagged_cont),
                datums,
            };

            let mut bytes = consume_result_return_proto.encode_to_vec();
            let len = bytes.len() as u32;
            let len_bytes = len.to_le_bytes().to_vec();
            let mut result = len_bytes;
            result.append(&mut bytes);
            let total_len = result.len();
            let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
            RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
            ptr
        }
    }
}

#[no_mangle]
extern "C" fn reset(
    runtime_ptr: *mut RhoRuntime,
    root_pointer: *const u8,
    root_bytes_len: usize,
) -> i32 {
    // println!("\nHit reset");

    let root_slice = unsafe { std::slice::from_raw_parts(root_pointer, root_bytes_len) };
    let root = Blake2b256Hash::from_bytes(root_slice.to_vec());

    // Access underlying space directly to capture Result and map to error code
    let runtime = unsafe { &mut (*runtime_ptr).runtime };

    let mut space_lock = match runtime.reducer.space.try_lock() {
        Ok(lock) => lock,
        Err(e) => {
            eprintln!("ERROR: failed to lock reducer.space in reset: {:?}", e);
            return 2; // lock error
        }
    };

    match space_lock.reset(&root) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("ERROR: reset failed: {:?}", e);
            1 // generic reset error (e.g. unknown root)
        }
    }
}

#[no_mangle]
extern "C" fn get_data(
    runtime_ptr: *mut RhoRuntime,
    channel_pointer: *const u8,
    channel_bytes_len: usize,
) -> *const u8 {
    let channel_slice = unsafe { std::slice::from_raw_parts(channel_pointer, channel_bytes_len) };
    let channel = Par::decode(channel_slice).unwrap();

    // let rt = tokio::runtime::Runtime::new().unwrap();
    // let datums =
    //     rt.block_on(async { unsafe { (*runtime_ptr).runtime.try_lock().unwrap().get_data(channel).await } });
    let datums = unsafe { (*runtime_ptr).runtime.get_data(&channel) };

    // println!("\ndatums in rust get_data: {:?}", datums);

    let datums_protos: Vec<DatumProto> = datums
        .into_iter()
        .map(|datum| DatumProto {
            a: Some(datum.a),
            persist: datum.persist,
            source: Some(ProduceProto {
                channel_hash: datum.source.channel_hash.bytes(),
                hash: datum.source.hash.bytes(),
                persistent: datum.source.persistent,
                is_deterministic: datum.source.is_deterministic,
                output_value: datum.source.output_value,
            }),
        })
        .collect();

    let datums_proto = DatumsProto {
        datums: datums_protos,
    };

    let mut bytes = datums_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn get_joins(
    runtime_ptr: *mut RhoRuntime,
    channel_pointer: *const u8,
    channel_bytes_len: usize,
) -> *const u8 {
    let channel_slice = unsafe { std::slice::from_raw_parts(channel_pointer, channel_bytes_len) };
    let channel = Par::decode(channel_slice).unwrap();

    let joins = unsafe { (*runtime_ptr).runtime.get_joins(channel) };

    let vec_join: Vec<JoinProto> = joins.into_iter().map(|join| JoinProto { join }).collect();
    let joins_proto = JoinsProto { joins: vec_join };

    let mut bytes = joins_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn get_waiting_continuations(
    runtime_ptr: *mut RhoRuntime,
    channels_pointer: *const u8,
    channels_bytes_len: usize,
) -> *const u8 {
    let channels_slice =
        unsafe { std::slice::from_raw_parts(channels_pointer, channels_bytes_len) };
    let channels_proto = ChannelsProto::decode(channels_slice).unwrap();

    let wks = unsafe {
        (*runtime_ptr)
            .runtime
            .get_continuations(channels_proto.channels)
    };

    let wks_protos: Vec<WaitingContinuationProto> = wks
        .into_iter()
        .map(|wk| {
            let res = WaitingContinuationProto {
                patterns: wk.patterns,
                continuation: Some(wk.continuation.clone()),
                persist: wk.persist,
                peeks: wk
                    .peeks
                    .into_iter()
                    .map(|peek| SortedSetElement { value: peek as i32 })
                    .collect(),
                source: Some(ConsumeProto {
                    channel_hashes: wk
                        .source
                        .channel_hashes
                        .iter()
                        .map(|hash| hash.bytes())
                        .collect(),
                    hash: wk.source.hash.bytes(),
                    persistent: wk.source.persistent,
                }),
            };

            res
        })
        .collect();

    let wks_proto = WaitingContinuationsProto { wks: wks_protos };

    let mut bytes = wks_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn set_block_data(
    runtime_ptr: *mut RhoRuntime,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> () {
    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = BlockDataProto::decode(params_slice).unwrap();
    let block_data = BlockData {
        time_stamp: params.time_stamp as i64,
        block_number: params.block_number as i64,
        sender: PublicKey::from_bytes(&params.public_key),
        seq_num: params.seq_num,
    };

    let runtime = unsafe { &mut (*runtime_ptr).runtime };
    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    tokio_runtime.block_on(async {
        runtime.set_block_data(block_data).await;
    });
}

#[no_mangle]
extern "C" fn set_invalid_blocks(
    runtime_ptr: *mut RhoRuntime,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> () {
    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = InvalidBlocksProto::decode(params_slice).unwrap();
    let invalid_blocks = params
        .invalid_blocks
        .into_iter()
        .map(|block| (block.block_hash.into(), block.validator.into()))
        .collect();

    let runtime = unsafe { &mut (*runtime_ptr).runtime };
    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    tokio_runtime.block_on(async {
        runtime.set_invalid_blocks(invalid_blocks).await;
    });
}

#[no_mangle]
extern "C" fn get_hot_changes(runtime_ptr: *mut RhoRuntime) -> *const u8 {
    let runtime = unsafe { &(*runtime_ptr).runtime };
    let hot_store_mapped = runtime.get_hot_changes();

    let mut map_entries: Vec<StoreToMapEntry> = Vec::new();

    for (key, value) in hot_store_mapped {
        let datums = value
            .data
            .into_iter()
            .map(|datum| DatumProto {
                a: Some(datum.a),
                persist: datum.persist,
                source: Some(ProduceProto {
                    channel_hash: datum.source.channel_hash.bytes(),
                    hash: datum.source.hash.bytes(),
                    persistent: datum.source.persistent,
                    is_deterministic: datum.source.is_deterministic,
                    output_value: datum.source.output_value,
                }),
            })
            .collect();

        let wks = value
            .wks
            .into_iter()
            .map(|wk| {
                let res = WaitingContinuationProto {
                    patterns: wk.patterns,
                    continuation: Some(wk.continuation.clone()),
                    persist: wk.persist,
                    peeks: wk
                        .peeks
                        .into_iter()
                        .map(|peek| SortedSetElement { value: peek as i32 })
                        .collect(),
                    source: Some(ConsumeProto {
                        channel_hashes: wk
                            .source
                            .channel_hashes
                            .iter()
                            .map(|hash| hash.bytes())
                            .collect(),
                        hash: wk.source.hash.bytes(),
                        persistent: wk.source.persistent,
                    }),
                };

                res
            })
            .collect();

        let value = StoreToMapValue { data: datums, wks };
        map_entries.push(StoreToMapEntry {
            key,
            value: Some(value),
        });
    }

    let to_map_result = StoreToMapResult { map_entries };

    let mut bytes = to_map_result.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
extern "C" fn set_cost_to_max(runtime_ptr: *mut RhoRuntime) -> () {
    unsafe {
        (*runtime_ptr).runtime.cost.set(Cost::unsafe_max());
    }
}

/* REPLAY RHO RUNTIME */

#[no_mangle]
extern "C" fn rig(
    runtime_ptr: *mut ReplayRhoRuntime,
    log_pointer: *const u8,
    log_bytes_len: usize,
) -> () {
    let log_slice = unsafe { std::slice::from_raw_parts(log_pointer, log_bytes_len) };
    let log_proto = LogProto::decode(log_slice).unwrap();

    let log: Vec<Event> = log_proto
        .log
        .into_iter()
        .map(|log_entry| match log_entry.event_type.unwrap() {
            event_proto::EventType::Comm(comm_proto) => {
                let consume_proto = comm_proto.consume.unwrap();
                let comm = COMM {
                    consume: {
                        Consume {
                            channel_hashes: {
                                consume_proto
                                    .channel_hashes
                                    .iter()
                                    .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
                                    .collect()
                            },
                            hash: Blake2b256Hash::from_bytes(consume_proto.hash),
                            persistent: consume_proto.persistent,
                        }
                    },
                    produces: {
                        comm_proto
                            .produces
                            .into_iter()
                            .map(|produce_proto| Produce {
                                channel_hash: Blake2b256Hash::from_bytes(
                                    produce_proto.channel_hash,
                                ),
                                hash: Blake2b256Hash::from_bytes(produce_proto.hash),
                                persistent: produce_proto.persistent,
                                is_deterministic: produce_proto.is_deterministic,
                                output_value: produce_proto.output_value,
                            })
                            .collect()
                    },
                    peeks: {
                        comm_proto
                            .peeks
                            .iter()
                            .map(|element| element.value)
                            .collect()
                    },
                    times_repeated: {
                        comm_proto
                            .times_repeated
                            .into_iter()
                            .map(|map_entry| {
                                let key_proto = map_entry.key.unwrap();
                                let produce = Produce {
                                    channel_hash: Blake2b256Hash::from_bytes(
                                        key_proto.channel_hash,
                                    ),
                                    hash: Blake2b256Hash::from_bytes(key_proto.hash),
                                    persistent: key_proto.persistent,
                                    is_deterministic: key_proto.is_deterministic,
                                    output_value: key_proto.output_value,
                                };

                                let value = map_entry.value;

                                (produce, value)
                            })
                            .collect()
                    },
                };
                Event::Comm(comm)
            }
            event_proto::EventType::IoEvent(io_event) => match io_event.io_event_type.unwrap() {
                io_event_proto::IoEventType::Produce(produce_proto) => {
                    let produce = Produce {
                        channel_hash: Blake2b256Hash::from_bytes(produce_proto.channel_hash),
                        hash: Blake2b256Hash::from_bytes(produce_proto.hash),
                        persistent: produce_proto.persistent,
                        is_deterministic: produce_proto.is_deterministic,
                        output_value: produce_proto.output_value,
                    };
                    Event::IoEvent(IOEvent::Produce(produce))
                }
                io_event_proto::IoEventType::Consume(consume_proto) => {
                    let consume = Consume {
                        channel_hashes: {
                            consume_proto
                                .channel_hashes
                                .iter()
                                .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
                                .collect()
                        },
                        hash: Blake2b256Hash::from_bytes(consume_proto.hash),
                        persistent: consume_proto.persistent,
                    };
                    Event::IoEvent(IOEvent::Consume(consume))
                }
            },
        })
        .collect();

    unsafe {
        (*runtime_ptr).runtime.rig(log).unwrap();
    }
}

#[no_mangle]
extern "C" fn check_replay_data(runtime_ptr: *mut ReplayRhoRuntime) -> () {
    unsafe {
        (*runtime_ptr).runtime.check_replay_data().unwrap();
    }
}

#[no_mangle]
extern "C" fn create_runtime(
    rspace_ptr: *mut Space,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> *mut RhoRuntime {
    let rspace = unsafe { (*rspace_ptr).rspace.try_lock().unwrap().clone() };

    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = CreateRuntimeParams::decode(params_slice).unwrap();

    let mergeable_tag_name = params.mergeable_tag_name.unwrap();
    let init_registry = params.init_registry;
    if params.rho_spec_system_processes {
        panic!("ERROR: There are additional system processes being passed to the rust rho_runtime that are not being handled.")
    };

    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    let rho_runtime = tokio_runtime.block_on(async {
        create_rho_runtime(rspace, mergeable_tag_name, init_registry, &mut Vec::new()).await
    });

    Box::into_raw(Box::new(RhoRuntime {
        runtime: rho_runtime,
    }))
}

#[no_mangle]
extern "C" fn create_runtime_with_test_framework(
    rspace_ptr: *mut Space,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> *mut RhoRuntime {
    let rspace = unsafe { (*rspace_ptr).rspace.try_lock().unwrap().clone() };

    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = CreateRuntimeParams::decode(params_slice).unwrap();

    let mergeable_tag_name = params.mergeable_tag_name.unwrap();
    let init_registry = params.init_registry;
    let mut extra_system_processes = if params.rho_spec_system_processes {
        test_framework_contracts()
    } else {
        Vec::new()
    };

    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    let rho_runtime = tokio_runtime.block_on(async {
        create_rho_runtime(
            rspace,
            mergeable_tag_name,
            init_registry,
            &mut extra_system_processes,
        )
        .await
    });

    Box::into_raw(Box::new(RhoRuntime {
        runtime: rho_runtime,
    }))
}

#[no_mangle]
extern "C" fn create_replay_runtime(
    replay_space_ptr: *mut ReplaySpace,
    params_ptr: *const u8,
    params_bytes_len: usize,
) -> *mut ReplayRhoRuntime {
    let rspace = unsafe { (*replay_space_ptr).replay_space.try_lock().unwrap().clone() };

    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = CreateRuntimeParams::decode(params_slice).unwrap();

    let mergeable_tag_name = params.mergeable_tag_name.unwrap();
    let init_registry = params.init_registry;
    if params.rho_spec_system_processes {
        panic!("ERROR: There are additional system processes being passed to the rust rho_runtime that are not being handled.")
    };

    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    let replay_rho_runtime = tokio_runtime.block_on(async {
        create_rho_runtime(rspace, mergeable_tag_name, init_registry, &mut Vec::new()).await
    });

    Box::into_raw(Box::new(ReplayRhoRuntime {
        runtime: replay_rho_runtime,
    }))
}

#[no_mangle]
extern "C" fn bootstrap_registry(runtime_ptr: *mut RhoRuntime) -> () {
    let runtime = unsafe { (*runtime_ptr).runtime.clone() };
    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    tokio_runtime.block_on(async {
        bootstrap_registry_internal(&runtime).await;
    });
}

#[no_mangle]
extern "C" fn source_to_adt(params_ptr: *const u8, params_bytes_len: usize) -> *const u8 {
    // Deserialization of parameters
    let params_slice = unsafe { std::slice::from_raw_parts(params_ptr, params_bytes_len) };
    let params = SourceToAdtParams::decode(params_slice).unwrap();

    // Execution of transformation logic
    // let result = match Compiler::source_to_adt_with_normalizer_env(
    //     &params.source,
    //     params.normalizer_env.into_iter().collect(),
    // ) {
    //     Ok(par) => {
    //         // println!("\npar in source_to_adt: {:?}", par);
    //         par
    //     }
    //     Err(error) => {
    //         println!("source_to_adt rust side error {:?}", error);
    //         return std::ptr::null();
    //     }
    // };

		let result = match Compiler::source_to_adt_with_normalizer_env(
			&params.source,
			params.normalizer_env.into_iter().collect(),
	) {
			Ok(par) => {
					// println!("\npar in source_to_adt: {:?}", par);
					par
			}
			Err(error) => {
					println!("source_to_adt rust side error {:?}", error);
					return std::ptr::null();
			}
	};

    // Serialization of the result in `Par`
    let mut result_bytes = result.encode_to_vec();
    let len = result_bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();

    // Add the length of the result at the beginning of the byte array
    let mut full_result = len_bytes;
    full_result.append(&mut result_bytes);

    // Return a pointer to the serialized result
    let total_len = full_result.len();
    let ptr = Box::leak(full_result.into_boxed_slice()).as_ptr();
    RHOLANG_ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}
