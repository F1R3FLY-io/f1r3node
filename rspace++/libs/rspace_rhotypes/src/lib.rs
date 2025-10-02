use std::sync::atomic::{AtomicUsize, Ordering};

// Tracks total bytes currently allocated and leaked to JNA callers
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "C" fn get_allocated_bytes() -> usize {
    ALLOCATED_BYTES.load(Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn reset_allocated_bytes() {
    ALLOCATED_BYTES.store(0, Ordering::SeqCst)
}
use models::rhoapi::*;
use models::rspace_plus_plus_types::*;
use prost::Message;
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::matcher::spatial_matcher::SpatialMatcherContext;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::hashing::stable_hash_provider::{hash, hash_from_vec};
use rspace_plus_plus::rspace::replay_rspace::ReplayRSpace;
use rspace_plus_plus::rspace::logging::BasicLogger;
use rspace_plus_plus::rspace::rspace::RSpace;
use rspace_plus_plus::rspace::rspace_interface::ISpace;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use rspace_plus_plus::rspace::shared::lmdb_dir_store_manager::GB;
use rspace_plus_plus::rspace::shared::rspace_store_manager::mk_rspace_store_manager;
use rspace_plus_plus::rspace::state::exporters::rspace_exporter_items::RSpaceExporterItems;
use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporterInstance;
use rspace_plus_plus::rspace::trace::event::{Event, IOEvent};
use shared::rust::ByteVector;
use std::ffi::{CStr, c_char};
use std::sync::{Arc, Mutex};

/*
 * This library contains predefined types for Channel, Pattern, Data, and Continuation - RhoTypes
 * These types (C, P, A, K) MUST MATCH the corresponding types on the Scala side in 'RSpacePlusPlus_RhoTypes.scala'
 */
#[repr(C)]
pub struct Space {
    rspace: Mutex<RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>,
}

#[repr(C)]
pub struct ReplaySpace {
    replay_space: Mutex<ReplayRSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>>,
}

#[no_mangle]
pub extern "C" fn space_new(path: *const c_char) -> *mut Space {
    let c_str = unsafe { CStr::from_ptr(path) };
    let data_dir = c_str.to_str().unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let rspace = rt
        .block_on(async {
            // println!("\nHit space_new");

            // let mut kvm = mk_rspace_store_manager(lmdb_path.into(), 1 * GB);
            // let store = kvm.r_space_stores().await.unwrap();

            let mut kvm =
                mk_rspace_store_manager((&format!("{}/rspace++/", data_dir)).into(), 1 * GB);
            let store = kvm.r_space_stores().await.unwrap();

            // println!("\nhistory store: {:?}", store.history.lock().unwrap().to_map().unwrap());
            // println!("\nroots store: {:?}", store.roots.lock().unwrap().to_map().unwrap());
            // roots_store.print_store();
            // println!("\ncold store: {:?}", store.cold.lock().unwrap().to_map().unwrap().len());
            // cold_store.print_store();

            // let store =
            //     get_or_create_rspace_store(&format!("{}/rspace++_{}/", data_dir, unique_id), 1 * GB)
            //         .expect("Error getting RSpaceStore: ");
            // let store = get_or_create_rspace_store(&format!("{}/rspace++/", data_dir), 1 * GB)
            //     .expect("Error getting RSpaceStore: ");

            RSpace::create(store, Arc::new(Box::new(Matcher)))
        })
        .unwrap();

    Box::into_raw(Box::new(Space {
        rspace: Mutex::new(rspace),
    }))
}

#[no_mangle]
pub extern "C" fn space_new_replay(rspace: *mut Space) -> *mut ReplaySpace {
    let rspace = unsafe { (*rspace).rspace.lock().unwrap() };
    let replay_space = ReplayRSpace::apply_with_logger(
        rspace.history_repository.clone(),
        rspace.store.clone(),
        Arc::new(Box::new(Matcher)),
        Box::new(BasicLogger::new()),
    );

    Box::into_raw(Box::new(ReplaySpace {
        replay_space: Mutex::new(replay_space),
    }))
}

#[no_mangle]
pub extern "C" fn space_print(rspace: *mut Space) -> () {
    unsafe { (*rspace).rspace.lock().unwrap().store.print() }
}

#[no_mangle]
pub extern "C" fn space_clear(rspace: *mut Space) -> () {
    unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .clear()
            .expect("Rust RSpacePlusPlus Library: Failed to clear");
    }
}

#[no_mangle]
pub extern "C" fn spatial_match_result(
    payload_pointer: *const u8,
    target_bytes_len: usize,
    pattern_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe {
        std::slice::from_raw_parts(payload_pointer, target_bytes_len + pattern_bytes_len)
    };
    let (target_slice, pattern_slice) = payload_slice.split_at(target_bytes_len);

    let target = Par::decode(target_slice).unwrap();
    let pattern = Par::decode(pattern_slice).unwrap();

    let mut spatial_matcher = SpatialMatcherContext::new();
    let result_option = spatial_matcher.spatial_match_result(target, pattern);

    match result_option {
        Some(free_map) => {
            let free_map_proto = FreeMapProto {
                entries: free_map.clone(),
            };

            let mut bytes = free_map_proto.encode_to_vec();
            let len = bytes.len() as u32;
            let len_bytes = len.to_le_bytes().to_vec();
            let mut result = len_bytes;
            result.append(&mut bytes);
            let total_len = result.len();
            let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
            ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
            ptr
        }
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn produce(
    rspace: *mut Space,
    payload_pointer: *const u8,
    channel_bytes_len: usize,
    data_bytes_len: usize,
    persist: bool,
) -> *const u8 {
    // println!("\nHit produce");

    let payload_slice =
        unsafe { std::slice::from_raw_parts(payload_pointer, channel_bytes_len + data_bytes_len) };
    let (channel_slice, data_slice) = payload_slice.split_at(channel_bytes_len);

    let channel = Par::decode(channel_slice).unwrap();
    let data = ListParWithRandom::decode(data_slice).unwrap();

    let result_option = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .produce(channel, data, persist)
    }
    .unwrap();

    match result_option {
        Some((cont_result, rspace_results, produce_event)) => {
            let protobuf_cont_result = ContResultProto {
                continuation: Some(cont_result.continuation.clone()),
                persistent: cont_result.persistent,
                channels: cont_result.channels,
                patterns: cont_result.patterns,
                peek: cont_result.peek,
            };

            let protobuf_results = rspace_results
                .into_iter()
                .map(|result| RSpaceResultProto {
                    channel: Some(result.channel),
                    matched_datum: Some(result.matched_datum),
                    removed_datum: Some(result.removed_datum),
                    persistent: result.persistent,
                })
                .collect();

            let protobuf_produce = ProduceProto {
                channel_hash: produce_event.channel_hash.bytes(),
                hash: produce_event.hash.bytes(),
                persistent: produce_event.persistent,
                is_deterministic: produce_event.is_deterministic,
                output_value: produce_event.output_value,
            };

            let maybe_action_result = ProduceResult {
                cont_result: Some(protobuf_cont_result),
                results: protobuf_results,
                produce: Some(protobuf_produce),
            };

            let mut bytes = maybe_action_result.encode_to_vec();
            let len = bytes.len() as u32;
            let len_bytes = len.to_le_bytes().to_vec();
            let mut result = len_bytes;
            result.append(&mut bytes);
            let total_len = result.len();
            let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
            ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
            ptr
        }
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn consume(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> *const u8 {
    // println!("\nHit rust consume");

    // let thread_id = thread::current().id();
    // let current_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // println!("Thread ID: {:?}, Current Time: {}", thread_id, current_time);

    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let consume_params = ConsumeParams::decode(payload_slice).unwrap();

    let channels = consume_params.channels;
    let patterns = consume_params.patterns;
    let continuation = consume_params.continuation.unwrap();
    let persist = consume_params.persist;
    let peeks = consume_params.peeks.into_iter().map(|e| e.value).collect();

    let result_option = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .consume(channels, patterns, continuation, persist, peeks)
    }
    .unwrap();

    match result_option {
        Some((cont_result, rspace_results)) => {
            let protobuf_cont_result = ContResultProto {
                continuation: Some(cont_result.continuation.clone()),
                persistent: cont_result.persistent,
                channels: cont_result.channels,
                patterns: cont_result.patterns,
                peek: cont_result.peek,
            };

            let protobuf_results = rspace_results
                .into_iter()
                .map(|result| RSpaceResultProto {
                    channel: Some(result.channel),
                    matched_datum: Some(result.matched_datum),
                    removed_datum: Some(result.removed_datum),
                    persistent: result.persistent,
                })
                .collect();

            let maybe_action_result = ConsumeResult {
                cont_result: Some(protobuf_cont_result),
                results: protobuf_results,
            };

            let mut bytes = maybe_action_result.encode_to_vec();
            let len = bytes.len() as u32;
            let len_bytes = len.to_le_bytes().to_vec();
            let mut result = len_bytes;
            result.append(&mut bytes);

            // println!("\nlen: {:?}", len);
            let total_len = result.len();
            let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
            ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
            ptr
        }
        None => {
            // println!("\nnone in rust consume");
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn install(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> *const u8 {
    // println!("\nHit install");

    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let consume_params = InstallParams::decode(payload_slice).unwrap();

    let channels = consume_params.channels;
    let patterns = consume_params.patterns;
    let continuation = consume_params.continuation.unwrap();

    let result_option = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .install(channels, patterns, continuation)
    }
    .unwrap();

    match result_option {
        None => std::ptr::null(),
        Some(_) => {
            panic!("RUST ERROR: Installing can be done only on startup")
        }
    }
}

// #[no_mangle]
// pub extern "C" fn create_checkpoint(rspace: *mut Space) -> *const u8 {
//     let checkpoint = unsafe {
//         (*rspace)
//             .rspace
//             .lock()
//             .unwrap()
//             .create_checkpoint()
//             .expect("Rust RSpacePlusPlus Library: Failed to create checkpoint")
//     };

//     let log = checkpoint.log;
//     let log_proto: Vec<EventProto> = log
//         .into_iter()
//         .map(|event| match event {
//             Event::Comm(comm) => {
//                 let comm_proto = CommProto {
//                     consume: {
//                         Some(ConsumeProto {
//                             channel_hashes: comm
//                                 .consume
//                                 .channel_hashes
//                                 .iter()
//                                 .map(|hash| hash.bytes())
//                                 .collect(),
//                             hash: comm.consume.hash.bytes(),
//                             persistent: comm.consume.persistent,
//                         })
//                     },
//                     produces: {
//                         comm.produces
//                             .into_iter()
//                             .map(|produce| ProduceProto {
//                                 channel_hash: produce.channel_hash.bytes(),
//                                 hash: produce.hash.bytes(),
//                                 persistent: produce.persistent,
//                             })
//                             .collect()
//                     },
//                     peeks: {
//                         comm.peeks
//                             .into_iter()
//                             .map(|peek| SortedSetElement { value: peek as i32 })
//                             .collect()
//                     },
//                     times_repeated: {
//                         let mut produce_counter_map_entries: Vec<ProduceCounterMapEntry> =
//                             Vec::new();
//                         for (key, value) in comm.times_repeated {
//                             let produce = ProduceProto {
//                                 channel_hash: key.channel_hash.bytes(),
//                                 hash: key.hash.bytes(),
//                                 persistent: key.persistent,
//                             };

//                             produce_counter_map_entries.push(ProduceCounterMapEntry {
//                                 key: Some(produce),
//                                 value,
//                             });
//                         }
//                         produce_counter_map_entries
//                     },
//                 };

//                 EventProto {
//                     event_type: Some(event_proto::EventType::Comm(comm_proto)),
//                 }
//             }
//             Event::IoEvent(io_event) => match io_event {
//                 IOEvent::Produce(produce) => {
//                     let produce_proto = ProduceProto {
//                         channel_hash: produce.channel_hash.bytes(),
//                         hash: produce.hash.bytes(),
//                         persistent: produce.persistent,
//                     };
//                     EventProto {
//                         event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
//                             io_event_type: Some(io_event_proto::IoEventType::Produce(
//                                 produce_proto,
//                             )),
//                         })),
//                     }
//                 }
//                 IOEvent::Consume(consume) => {
//                     let consume_proto = ConsumeProto {
//                         channel_hashes: consume
//                             .channel_hashes
//                             .iter()
//                             .map(|hash| hash.bytes())
//                             .collect(),
//                         hash: consume.hash.bytes(),
//                         persistent: consume.persistent,
//                     };
//                     EventProto {
//                         event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
//                             io_event_type: Some(io_event_proto::IoEventType::Consume(
//                                 consume_proto,
//                             )),
//                         })),
//                     }
//                 }
//             },
//         })
//         .collect();

//     let checkpoint_proto = CheckpointProto {
//         root: checkpoint.root.bytes(),
//         log: log_proto,
//     };

//     let mut bytes = checkpoint_proto.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

#[no_mangle]
pub extern "C" fn reset_rspace(
    rspace: *mut Space,
    root_pointer: *const u8,
    root_bytes_len: usize,
) -> i32 {
    let root_slice = unsafe { std::slice::from_raw_parts(root_pointer, root_bytes_len) };
    let root = Blake2b256Hash::from_bytes(root_slice.to_vec());

    let mut rs = match unsafe { (*rspace).rspace.lock() } {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("ERROR: failed to lock rspace in reset: poisoned");
            poisoned.into_inner()
        }
    };

    match rs.reset(&root) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("ERROR: rspace reset failed: {:?}", e);
            1
        }
    }
}

// #[no_mangle]
// pub extern "C" fn get_data(
//     rspace: *mut Space,
//     channel_pointer: *const u8,
//     channel_bytes_len: usize,
// ) -> *const u8 {
//     let channel_slice = unsafe { std::slice::from_raw_parts(channel_pointer, channel_bytes_len) };
//     let channel = Par::decode(channel_slice).unwrap();

//     // let rt = tokio::runtime::Runtime::new().unwrap();
//     // let datums =
//     //     rt.block_on(async { unsafe { (*rspace).rspace.lock().unwrap().get_data(channel).await } });
//     let datums = unsafe { (*rspace).rspace.lock().unwrap().get_data(channel) };

//     // println!("\ndatums in rust get_data: {:?}", datums);

//     let datums_protos: Vec<DatumProto> = datums
//         .into_iter()
//         .map(|datum| DatumProto {
//             a: Some(datum.a),
//             persist: datum.persist,
//             source: Some(ProduceProto {
//                 channel_hash: datum.source.channel_hash.bytes(),
//                 hash: datum.source.hash.bytes(),
//                 persistent: datum.source.persistent,
//             }),
//         })
//         .collect();

//     let datums_proto = DatumsProto {
//         datums: datums_protos,
//     };

//     let mut bytes = datums_proto.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

// #[no_mangle]
// pub extern "C" fn get_waiting_continuations(
//     rspace: *mut Space,
//     channels_pointer: *const u8,
//     channels_bytes_len: usize,
// ) -> *const u8 {
//     let channels_slice =
//         unsafe { std::slice::from_raw_parts(channels_pointer, channels_bytes_len) };
//     let channels_proto = ChannelsProto::decode(channels_slice).unwrap();

//     let wks = unsafe {
//         (*rspace)
//             .rspace
//             .lock()
//             .unwrap()
//             .get_waiting_continuations(channels_proto.channels)
//     };

//     let wks_protos: Vec<WaitingContinuationProto> = wks
//         .into_iter()
//         .map(|wk| {
//             let res = WaitingContinuationProto {
//                 patterns: wk.patterns,
//                 continuation: Some(wk.continuation.clone()),
//                 persist: wk.persist,
//                 peeks: wk
//                     .peeks
//                     .into_iter()
//                     .map(|peek| SortedSetElement { value: peek as i32 })
//                     .collect(),
//                 source: Some(ConsumeProto {
//                     channel_hashes: wk
//                         .source
//                         .channel_hashes
//                         .iter()
//                         .map(|hash| hash.bytes())
//                         .collect(),
//                     hash: wk.source.hash.bytes(),
//                     persistent: wk.source.persistent,
//                 }),
//             };

//             res
//         })
//         .collect();

//     let wks_proto = WaitingContinuationsProto { wks: wks_protos };

//     let mut bytes = wks_proto.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

// #[no_mangle]
// pub extern "C" fn get_joins(
//     rspace: *mut Space,
//     channel_pointer: *const u8,
//     channel_bytes_len: usize,
// ) -> *const u8 {
//     let channel_slice = unsafe { std::slice::from_raw_parts(channel_pointer, channel_bytes_len) };
//     let channel = Par::decode(channel_slice).unwrap();

//     let joins = unsafe { (*rspace).rspace.lock().unwrap().get_joins(channel) };

//     let vec_join: Vec<JoinProto> = joins.into_iter().map(|join| JoinProto { join }).collect();
//     let joins_proto = JoinsProto { joins: vec_join };

//     let mut bytes = joins_proto.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

#[no_mangle]
pub extern "C" fn to_map(rspace: *mut Space) -> *const u8 {
    let hot_store_mapped = unsafe { (*rspace).rspace.lock().unwrap().store.to_map() };

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
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn spawn(rspace: *mut Space) -> *mut Space {
    let rspace = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .spawn()
            .expect("Rust RSpacePlusPlus Library: Failed to spawn")
    };

    Box::into_raw(Box::new(Space {
        rspace: Mutex::new(rspace),
    }))
}

// #[no_mangle]
// pub extern "C" fn create_soft_checkpoint(rspace: *mut Space) -> *const u8 {
//     // println!("\nhit rust lib create_soft_checkpoint");
//     let soft_checkpoint = unsafe { (*rspace).rspace.lock().unwrap().create_soft_checkpoint() };

//     let mut conts_map_entries: Vec<StoreStateContMapEntry> = Vec::new();
//     let mut installed_conts_map_entries: Vec<StoreStateInstalledContMapEntry> = Vec::new();
//     let mut data_map_entries: Vec<StoreStateDataMapEntry> = Vec::new();
//     let mut joins_map_entries: Vec<StoreStateJoinsMapEntry> = Vec::new();
//     let mut installed_joins_map_entries: Vec<StoreStateInstalledJoinsMapEntry> = Vec::new();

//     let hot_store_state = soft_checkpoint.cache_snapshot;

//     for (key, value) in hot_store_state.continuations.clone().into_iter() {
//         let wks: Vec<WaitingContinuationProto> = value
//             .into_iter()
//             .map(|wk| {
//                 let res = WaitingContinuationProto {
//                     patterns: wk.patterns,
//                     continuation: Some(wk.continuation.clone()),
//                     persist: wk.persist,
//                     peeks: wk
//                         .peeks
//                         .into_iter()
//                         .map(|peek| SortedSetElement { value: peek as i32 })
//                         .collect(),
//                     source: Some(ConsumeProto {
//                         channel_hashes: wk
//                             .source
//                             .channel_hashes
//                             .iter()
//                             .map(|hash| hash.bytes())
//                             .collect(),
//                         hash: wk.source.hash.bytes(),
//                         persistent: wk.source.persistent,
//                     }),
//                 };

//                 res
//             })
//             .collect();

//         conts_map_entries.push(StoreStateContMapEntry { key, value: wks });
//     }

//     for (key, value) in hot_store_state.installed_continuations.clone().into_iter() {
//         let wk = WaitingContinuationProto {
//             patterns: value.patterns,
//             continuation: Some(value.continuation.clone()),
//             persist: value.persist,
//             peeks: value
//                 .peeks
//                 .into_iter()
//                 .map(|peek| SortedSetElement { value: peek as i32 })
//                 .collect(),
//             source: Some(ConsumeProto {
//                 channel_hashes: value
//                     .source
//                     .channel_hashes
//                     .iter()
//                     .map(|hash| hash.bytes())
//                     .collect(),
//                 hash: value.source.hash.bytes(),
//                 persistent: value.source.persistent,
//             }),
//         };

//         installed_conts_map_entries.push(StoreStateInstalledContMapEntry {
//             key,
//             value: Some(wk),
//         });
//     }

//     for (key, value) in hot_store_state.data.clone().into_iter() {
//         let datums = value
//             .into_iter()
//             .map(|datum| DatumProto {
//                 a: Some(datum.a),
//                 persist: datum.persist,
//                 source: Some(ProduceProto {
//                     channel_hash: datum.source.channel_hash.bytes(),
//                     hash: datum.source.hash.bytes(),
//                     persistent: datum.source.persistent,
//                 }),
//             })
//             .collect();

//         data_map_entries.push(StoreStateDataMapEntry {
//             key: Some(key),
//             value: datums,
//         });
//     }

//     for (key, value) in hot_store_state.joins.clone().into_iter() {
//         let joins = value.into_iter().map(|join| JoinProto { join }).collect();

//         joins_map_entries.push(StoreStateJoinsMapEntry {
//             key: Some(key),
//             value: joins,
//         });
//     }

//     for (key, value) in hot_store_state.installed_joins.clone().into_iter() {
//         let joins = value.into_iter().map(|join| JoinProto { join }).collect();

//         installed_joins_map_entries.push(StoreStateInstalledJoinsMapEntry {
//             key: Some(key),
//             value: joins,
//         });
//     }

//     let hot_store_state_proto = HotStoreStateProto {
//         continuations: conts_map_entries,
//         installed_continuations: installed_conts_map_entries,
//         data: data_map_entries,
//         joins: joins_map_entries,
//         installed_joins: installed_joins_map_entries,
//     };

//     let log = soft_checkpoint.log;
//     let log_proto: Vec<EventProto> = log
//         .into_iter()
//         .map(|event| match event {
//             Event::Comm(comm) => {
//                 let comm_proto = CommProto {
//                     consume: {
//                         Some(ConsumeProto {
//                             channel_hashes: comm
//                                 .consume
//                                 .channel_hashes
//                                 .iter()
//                                 .map(|hash| hash.bytes())
//                                 .collect(),
//                             hash: comm.consume.hash.bytes(),
//                             persistent: comm.consume.persistent,
//                         })
//                     },
//                     produces: {
//                         comm.produces
//                             .into_iter()
//                             .map(|produce| ProduceProto {
//                                 channel_hash: produce.channel_hash.bytes(),
//                                 hash: produce.hash.bytes(),
//                                 persistent: produce.persistent,
//                             })
//                             .collect()
//                     },
//                     peeks: {
//                         comm.peeks
//                             .into_iter()
//                             .map(|peek| SortedSetElement { value: peek as i32 })
//                             .collect()
//                     },
//                     times_repeated: {
//                         let mut produce_counter_map_entries: Vec<ProduceCounterMapEntry> =
//                             Vec::new();
//                         for (key, value) in comm.times_repeated {
//                             let produce = ProduceProto {
//                                 channel_hash: key.channel_hash.bytes(),
//                                 hash: key.hash.bytes(),
//                                 persistent: key.persistent,
//                             };

//                             produce_counter_map_entries.push(ProduceCounterMapEntry {
//                                 key: Some(produce),
//                                 value,
//                             });
//                         }
//                         produce_counter_map_entries
//                     },
//                 };

//                 EventProto {
//                     event_type: Some(event_proto::EventType::Comm(comm_proto)),
//                 }
//             }
//             Event::IoEvent(io_event) => match io_event {
//                 IOEvent::Produce(produce) => {
//                     let produce_proto = ProduceProto {
//                         channel_hash: produce.channel_hash.bytes(),
//                         hash: produce.hash.bytes(),
//                         persistent: produce.persistent,
//                     };
//                     EventProto {
//                         event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
//                             io_event_type: Some(io_event_proto::IoEventType::Produce(
//                                 produce_proto,
//                             )),
//                         })),
//                     }
//                 }
//                 IOEvent::Consume(consume) => {
//                     let consume_proto = ConsumeProto {
//                         channel_hashes: consume
//                             .channel_hashes
//                             .iter()
//                             .map(|hash| hash.bytes())
//                             .collect(),
//                         hash: consume.hash.bytes(),
//                         persistent: consume.persistent,
//                     };
//                     EventProto {
//                         event_type: Some(event_proto::EventType::IoEvent(IoEventProto {
//                             io_event_type: Some(io_event_proto::IoEventType::Consume(
//                                 consume_proto,
//                             )),
//                         })),
//                     }
//                 }
//             },
//         })
//         .collect();

//     let mut produce_counter_map_entries: Vec<ProduceCounterMapEntry> = Vec::new();
//     let produce_counter_map = soft_checkpoint.produce_counter;

//     for (key, value) in produce_counter_map {
//         let produce = ProduceProto {
//             channel_hash: key.channel_hash.bytes(),
//             hash: key.hash.bytes(),
//             persistent: key.persistent,
//         };

//         produce_counter_map_entries.push(ProduceCounterMapEntry {
//             key: Some(produce),
//             value,
//         });
//     }

//     let soft_checkpoint_proto = SoftCheckpointProto {
//         cache_snapshot: Some(hot_store_state_proto),
//         log: log_proto,
//         produce_counter: produce_counter_map_entries,
//     };

//     let mut bytes = soft_checkpoint_proto.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

// #[no_mangle]
// pub extern "C" fn revert_to_soft_checkpoint(
//     rspace: *mut Space,
//     payload_pointer: *const u8,
//     payload_bytes_len: usize,
// ) -> () {
//     let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
//     let soft_checkpoint_proto = SoftCheckpointProto::decode(payload_slice).unwrap();
//     let cache_snapshot_proto = soft_checkpoint_proto.cache_snapshot.unwrap();

//     let conts_map: DashMap<Vec<Par>, Vec<WaitingContinuation<BindPattern, TaggedContinuation>>> =
//         cache_snapshot_proto
//             .continuations
//             .into_iter()
//             .map(|map_entry| {
//                 let key = map_entry.key;
//                 let value = map_entry
//                     .value
//                     .into_iter()
//                     .map(|cont_proto| WaitingContinuation {
//                         patterns: cont_proto.patterns,
//                         continuation: cont_proto.continuation.unwrap(),
//                         persist: cont_proto.persist,
//                         peeks: cont_proto
//                             .peeks
//                             .iter()
//                             .map(|element| element.value)
//                             .collect(),
//                         source: {
//                             let consume_proto = cont_proto.source.unwrap();
//                             Consume {
//                                 channel_hashes: consume_proto
//                                     .channel_hashes
//                                     .iter()
//                                     .map(|hash_bytes| {
//                                         Blake2b256Hash::from_bytes(hash_bytes.to_vec())
//                                     })
//                                     .collect(),
//                                 hash: Blake2b256Hash::from_bytes(consume_proto.hash),
//                                 persistent: consume_proto.persistent,
//                             }
//                         },
//                     })
//                     .collect();

//                 (key, value)
//             })
//             .collect();

//     let installed_conts_map: DashMap<
//         Vec<Par>,
//         WaitingContinuation<BindPattern, TaggedContinuation>,
//     > = cache_snapshot_proto
//         .installed_continuations
//         .into_iter()
//         .map(|map_entry| {
//             let key = map_entry.key;
//             let wk_proto = map_entry.value.unwrap();
//             let value = WaitingContinuation {
//                 patterns: wk_proto.patterns,
//                 continuation: wk_proto.continuation.unwrap(),
//                 persist: wk_proto.persist,
//                 peeks: wk_proto.peeks.iter().map(|element| element.value).collect(),
//                 source: {
//                     let consume_proto = wk_proto.source.unwrap();
//                     Consume {
//                         channel_hashes: consume_proto
//                             .channel_hashes
//                             .iter()
//                             .map(|hash_bytes| Blake2b256Hash::from_bytes(hash_bytes.to_vec()))
//                             .collect(),
//                         hash: Blake2b256Hash::from_bytes(consume_proto.hash),
//                         persistent: consume_proto.persistent,
//                     }
//                 },
//             };

//             (key, value)
//         })
//         .collect();

//     let datums_map: DashMap<Par, Vec<Datum<ListParWithRandom>>> = cache_snapshot_proto
//         .data
//         .into_iter()
//         .map(|map_entry| {
//             let key = map_entry.key.unwrap();
//             let value = map_entry
//                 .value
//                 .into_iter()
//                 .map(|datum_proto| Datum {
//                     a: datum_proto.a.unwrap(),
//                     persist: datum_proto.persist,
//                     source: {
//                         let produce_proto = datum_proto.source.unwrap();
//                         Produce {
//                             channel_hash: Blake2b256Hash::from_bytes(produce_proto.channel_hash),
//                             hash: Blake2b256Hash::from_bytes(produce_proto.hash),
//                             persistent: produce_proto.persistent,
//                         }
//                     },
//                 })
//                 .collect();

//             (key, value)
//         })
//         .collect();

//     let joins_map: DashMap<Par, Vec<Vec<Par>>> = cache_snapshot_proto
//         .joins
//         .into_iter()
//         .map(|map_entry| {
//             let key = map_entry.key.unwrap();
//             let value = map_entry
//                 .value
//                 .into_iter()
//                 .map(|join_proto| join_proto.join)
//                 .collect();

//             (key, value)
//         })
//         .collect();

//     let installed_joins_map: DashMap<Par, Vec<Vec<Par>>> = cache_snapshot_proto
//         .installed_joins
//         .into_iter()
//         .map(|map_entry| {
//             let key = map_entry.key.unwrap();
//             let value = map_entry
//                 .value
//                 .into_iter()
//                 .map(|join_proto| join_proto.join)
//                 .collect();

//             (key, value)
//         })
//         .collect();

//     let log: Vec<Event> = soft_checkpoint_proto
//         .log
//         .into_iter()
//         .map(|log_entry| match log_entry.event_type.unwrap() {
//             event_proto::EventType::Comm(comm_proto) => {
//                 let consume_proto = comm_proto.consume.unwrap();
//                 let comm = COMM {
//                     consume: {
//                         Consume {
//                             channel_hashes: {
//                                 consume_proto
//                                     .channel_hashes
//                                     .iter()
//                                     .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
//                                     .collect()
//                             },
//                             hash: Blake2b256Hash::from_bytes(consume_proto.hash),
//                             persistent: consume_proto.persistent,
//                         }
//                     },
//                     produces: {
//                         comm_proto
//                             .produces
//                             .into_iter()
//                             .map(|produce_proto| Produce {
//                                 channel_hash: Blake2b256Hash::from_bytes(
//                                     produce_proto.channel_hash,
//                                 ),
//                                 hash: Blake2b256Hash::from_bytes(produce_proto.hash),
//                                 persistent: produce_proto.persistent,
//                             })
//                             .collect()
//                     },
//                     peeks: {
//                         comm_proto
//                             .peeks
//                             .iter()
//                             .map(|element| element.value)
//                             .collect()
//                     },
//                     times_repeated: {
//                         comm_proto
//                             .times_repeated
//                             .into_iter()
//                             .map(|map_entry| {
//                                 let key_proto = map_entry.key.unwrap();
//                                 let produce = Produce {
//                                     channel_hash: Blake2b256Hash::from_bytes(
//                                         key_proto.channel_hash,
//                                     ),
//                                     hash: Blake2b256Hash::from_bytes(key_proto.hash),
//                                     persistent: key_proto.persistent,
//                                 };

//                                 let value = map_entry.value;

//                                 (produce, value)
//                             })
//                             .collect()
//                     },
//                 };
//                 Event::Comm(comm)
//             }
//             event_proto::EventType::IoEvent(io_event) => match io_event.io_event_type.unwrap() {
//                 io_event_proto::IoEventType::Produce(produce_proto) => {
//                     let produce = Produce {
//                         channel_hash: Blake2b256Hash::from_bytes(produce_proto.channel_hash),
//                         hash: Blake2b256Hash::from_bytes(produce_proto.hash),
//                         persistent: produce_proto.persistent,
//                     };
//                     Event::IoEvent(IOEvent::Produce(produce))
//                 }
//                 io_event_proto::IoEventType::Consume(consume_proto) => {
//                     let consume = Consume {
//                         channel_hashes: {
//                             consume_proto
//                                 .channel_hashes
//                                 .iter()
//                                 .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
//                                 .collect()
//                         },
//                         hash: Blake2b256Hash::from_bytes(consume_proto.hash),
//                         persistent: consume_proto.persistent,
//                     };
//                     Event::IoEvent(IOEvent::Consume(consume))
//                 }
//             },
//         })
//         .collect();

//     let produce_counter_map: BTreeMap<Produce, i32> = soft_checkpoint_proto
//         .produce_counter
//         .into_iter()
//         .map(|map_entry| {
//             let key_proto = map_entry.key.unwrap();
//             let produce = Produce {
//                 channel_hash: Blake2b256Hash::from_bytes(key_proto.channel_hash),
//                 hash: Blake2b256Hash::from_bytes(key_proto.hash),
//                 persistent: key_proto.persistent,
//             };

//             let value = map_entry.value;

//             (produce, value)
//         })
//         .collect();

//     let cache_snapshot = HotStoreState {
//         continuations: conts_map,
//         installed_continuations: installed_conts_map,
//         data: datums_map,
//         joins: joins_map,
//         installed_joins: installed_joins_map,
//     };

//     let soft_checkpoint = SoftCheckpoint {
//         cache_snapshot,
//         log,
//         produce_counter: produce_counter_map,
//     };

//     unsafe {
//         (*rspace)
//             .rspace
//             .lock()
//             .unwrap()
//             .revert_to_soft_checkpoint(soft_checkpoint)
//             .expect("Rust RSpacePlusPlus Library: Failed to revert to soft checkpoint")
//     }
// }

/* HistoryRepo */

#[no_mangle]
pub extern "C" fn history_repo_root(rspace: *mut Space) -> *const u8 {
    let root = unsafe { (*rspace).rspace.lock().unwrap().history_repository.root() };

    let hash = hash(&root);
    let hash_proto = HashProto { hash: hash.bytes() };

    let mut bytes = hash_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

/* Exporter */
// #[no_mangle]
// pub extern "C" fn get_nodes(
//     rspace: *mut Space,
//     payload_pointer: *const u8,
//     payload_bytes_len: usize,
// ) -> *const u8 {
//     let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
//     let exporter_params = ExporterParams::decode(payload_slice).unwrap();

//     let start_path: Vec<(Blake2b256Hash, Option<Byte>)> = exporter_params
//         .start_path
//         .into_iter()
//         .map(|path_proto| {
//             let key_hash = Blake2b256Hash::from_bytes(path_proto.key_hash);
//             let value: Option<Byte> = if path_proto.optional_byte.len() > 0 {
//                 Some(path_proto.optional_byte[0])
//             } else {
//                 None
//             };

//             (key_hash, value)
//         })
//         .collect();

//     let nodes = unsafe {
//         let space = (*rspace).rspace.lock().unwrap();
//         space
//             .history_repository
//             .exporter()
//             .lock()
//             .unwrap()
//             .get_nodes(
//                 start_path,
//                 exporter_params.skip.try_into().unwrap(),
//                 exporter_params.take.try_into().unwrap(),
//             )
//     };

//     let trie_nodes_proto_vec: Vec<TrieNodeProto> = nodes
//         .into_iter()
//         .map(|node| TrieNodeProto {
//             hash: node.hash.bytes(),
//             is_leaf: node.is_leaf,
//             path: node
//                 .path
//                 .into_iter()
//                 .map(|path| PathProto {
//                     key_hash: path.0.bytes(),
//                     optional_byte: if path.1.is_some() {
//                         vec![path.1.unwrap()]
//                     } else {
//                         Vec::new()
//                     },
//                 })
//                 .collect(),
//         })
//         .collect();

//     let trie_nodes_proto_vec = TrieNodesProto {
//         nodes: trie_nodes_proto_vec,
//     };

//     let mut bytes = trie_nodes_proto_vec.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

#[no_mangle]
pub extern "C" fn get_history_items(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let keys_proto = KeysProto::decode(payload_slice).unwrap();

    let keys = keys_proto
        .keys
        .into_iter()
        .map(|key| Blake2b256Hash::from_bytes(key.hash))
        .collect();

    let history_items = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .exporter()
            .lock()
            .unwrap()
            .get_history_items(keys)
            .unwrap()
    };

    let history_items_proto = ItemsProto {
        items: history_items
            .into_iter()
            .map(|history_item| ItemProto {
                key_hash: history_item.0.bytes(),
                value: history_item.1,
            })
            .collect(),
    };

    let mut bytes = history_items_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn get_data_items(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let keys_proto = KeysProto::decode(payload_slice).unwrap();

    let keys = keys_proto
        .keys
        .into_iter()
        .map(|key| Blake2b256Hash::from_bytes(key.hash))
        .collect();

    let data_items = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .exporter()
            .lock()
            .unwrap()
            .get_data_items(keys)
            .unwrap()
    };

    let data_items_proto = ItemsProto {
        items: data_items
            .into_iter()
            .map(|data_item| ItemProto {
                key_hash: data_item.0.bytes(),
                value: data_item.1,
            })
            .collect(),
    };

    let mut bytes = data_items_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn get_history_and_data(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let exporter_params = ExporterParams::decode(payload_slice).unwrap();

    let start_path: Vec<(Blake2b256Hash, Option<u8>)> = exporter_params
        .path
        .into_iter()
        .map(|path_element| {
            (
                Blake2b256Hash::from_bytes(path_element.key_hash),
                if path_element.optional_byte.len() > 0 {
                    Some(path_element.optional_byte[0])
                } else {
                    None
                },
            )
        })
        .collect();

    let exporter = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .exporter()
    };

    let history_and_data_items = RSpaceExporterItems::get_history_and_data(
        exporter,
        start_path,
        exporter_params.skip,
        exporter_params.take,
    );

    let history_items_proto = StoreItemsProto {
        items: history_and_data_items
            .0
            .items
            .into_iter()
            .map(|history_item| ItemProto {
                key_hash: history_item.0.bytes(),
                value: history_item.1,
            })
            .collect(),
        last_path: history_and_data_items
            .0
            .last_path
            .into_iter()
            .map(|path| PathElement {
                key_hash: path.0.bytes(),
                optional_byte: {
                    if path.1.is_some() {
                        vec![path.1.unwrap()]
                    } else {
                        Vec::new()
                    }
                },
            })
            .collect(),
    };

    let data_items_proto = StoreItemsProto {
        items: history_and_data_items
            .1
            .items
            .into_iter()
            .map(|data_item| ItemProto {
                key_hash: data_item.0.bytes(),
                value: data_item.1,
            })
            .collect(),
        last_path: history_and_data_items
            .1
            .last_path
            .into_iter()
            .map(|path| PathElement {
                key_hash: path.0.bytes(),
                optional_byte: {
                    if path.1.is_some() {
                        vec![path.1.unwrap()]
                    } else {
                        Vec::new()
                    }
                },
            })
            .collect(),
    };

    let history_and_data_items_proto = HistoryAndDataItems {
        history_items: Some(history_items_proto),
        data_items: Some(data_items_proto),
    };

    let mut bytes = history_and_data_items_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn get_exporter_root(rspace: *mut Space) -> *const u8 {
    let root = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .exporter()
            .lock()
            .unwrap()
            .get_root()
            .unwrap()
    };

    let hash = hash(&root);
    let hash_proto = HashProto { hash: hash.bytes() };

    let mut bytes = hash_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

// #[no_mangle]
// pub extern "C" fn traverse_history(
//     rspace: *mut Space,
//     payload_pointer: *const u8,
//     payload_bytes_len: usize,
// ) -> *const u8 {
//     let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
//     let exporter_params = ExporterParams::decode(payload_slice).unwrap();

//     let start_path: Vec<(Blake2b256Hash, Option<Byte>)> = exporter_params
//         .start_path
//         .into_iter()
//         .map(|path_proto| {
//             let key_hash = Blake2b256Hash::from_bytes(path_proto.key_hash);
//             let value: Option<Byte> = if path_proto.optional_byte.len() > 0 {
//                 Some(path_proto.optional_byte[0])
//             } else {
//                 None
//             };

//             (key_hash, value)
//         })
//         .collect();

//     let nodes = RSpaceExporterInstance::traverse_history(
//         start_path,
//         exporter_params.skip.try_into().unwrap(),
//         exporter_params.take.try_into().unwrap(),
//         get_from_history,
//     );

//     let trie_nodes_proto_vec: Vec<TrieNodeProto> = nodes
//         .into_iter()
//         .map(|node| TrieNodeProto {
//             hash: node.hash.bytes(),
//             is_leaf: node.is_leaf,
//             path: node
//                 .path
//                 .into_iter()
//                 .map(|path| PathProto {
//                     key_hash: path.0.bytes(),
//                     optional_byte: if path.1.is_some() {
//                         vec![path.1.unwrap()]
//                     } else {
//                         Vec::new()
//                     },
//                 })
//                 .collect(),
//         })
//         .collect();

//     let trie_nodes_proto_vec = TrieNodesProto {
//         nodes: trie_nodes_proto_vec,
//     };

//     let mut bytes = trie_nodes_proto_vec.encode_to_vec();
//     let len = bytes.len() as u32;
//     let len_bytes = len.to_le_bytes().to_vec();
//     let mut result = len_bytes;
//     result.append(&mut bytes);
//     Box::leak(result.into_boxed_slice()).as_ptr()
// }

/* Importer */

#[no_mangle]
pub extern "C" fn validate_state_items(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> () {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let params = ValidateStateParams::decode(payload_slice).unwrap();

    let history_items: Vec<(Blake2b256Hash, ByteVector)> = params
        .history_items
        .into_iter()
        .map(|history_item_proto| {
            (Blake2b256Hash::from_bytes(history_item_proto.key_hash), history_item_proto.value)
        })
        .collect();

    let data_items: Vec<(Blake2b256Hash, ByteVector)> = params
        .data_items
        .into_iter()
        .map(|data_item_proto| {
            (Blake2b256Hash::from_bytes(data_item_proto.key_hash), data_item_proto.value)
        })
        .collect();

    let start_path: Vec<(Blake2b256Hash, Option<u8>)> = params
        .start_path
        .into_iter()
        .map(|path_element| {
            (
                Blake2b256Hash::from_bytes(path_element.key_hash),
                if path_element.optional_byte.len() > 0 {
                    Some(path_element.optional_byte[0])
                } else {
                    None
                },
            )
        })
        .collect();

    let importer = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .importer()
    };

    let _ = RSpaceImporterInstance::validate_state_items(
        history_items,
        data_items,
        start_path,
        params.chunk_size,
        params.skip,
        importer,
    );
}

#[no_mangle]
pub extern "C" fn set_history_items(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> () {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let items_proto = ItemsProto::decode(payload_slice).unwrap();

    let data: Vec<(Blake2b256Hash, ByteVector)> = items_proto
        .items
        .into_iter()
        .map(|item| {
            let key_hash = Blake2b256Hash::from_bytes(item.key_hash);
            let value: ByteVector = item.value;

            (key_hash, value)
        })
        .collect();

    let _ = unsafe {
        let space = (*rspace).rspace.lock().unwrap();
        space
            .history_repository
            .importer()
            .lock()
            .unwrap()
            .set_history_items(data)
    };
}

#[no_mangle]
pub extern "C" fn set_data_items(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> () {
    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let items_proto = ItemsProto::decode(payload_slice).unwrap();

    let data: Vec<(Blake2b256Hash, ByteVector)> = items_proto
        .items
        .into_iter()
        .map(|item| {
            let key_hash = Blake2b256Hash::from_bytes(item.key_hash);
            let value: ByteVector = item.value;

            (key_hash, value)
        })
        .collect();

    let _ = unsafe {
        let space = (*rspace).rspace.lock().unwrap();
        space
            .history_repository
            .importer()
            .lock()
            .unwrap()
            .set_data_items(data)
    };
}

#[no_mangle]
pub extern "C" fn set_root(
    rspace: *mut Space,
    root_pointer: *const u8,
    root_bytes_len: usize,
) -> () {
    let root_slice = unsafe { std::slice::from_raw_parts(root_pointer, root_bytes_len) };
    let root = Blake2b256Hash::from_bytes(root_slice.to_vec());

    unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .importer()
            .lock()
            .unwrap()
            .set_root(&root)
    }
}

#[no_mangle]
pub extern "C" fn get_history_item(
    rspace: *mut Space,
    hash_pointer: *const u8,
    hash_bytes_len: usize,
) -> *const u8 {
    let hash_slice = unsafe { std::slice::from_raw_parts(hash_pointer, hash_bytes_len) };
    let hash = Blake2b256Hash::from_bytes(hash_slice.to_vec());

    let history_item = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .importer()
            .lock()
            .unwrap()
            .get_history_item(hash)
    };

    if history_item.is_some() {
        let history_item_proto = ByteVectorProto {
            byte_vector: history_item.unwrap(),
        };

        let mut bytes = history_item_proto.encode_to_vec();
        let len = bytes.len() as u32;
        let len_bytes = len.to_le_bytes().to_vec();
        let mut result = len_bytes;
        result.append(&mut bytes);
        let total_len = result.len();
        let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
        ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
        ptr
    } else {
        std::ptr::null()
    }
}

/* HistoryReader */
#[no_mangle]
pub extern "C" fn history_reader_root(
    rspace: *mut Space,
    state_hash_pointer: *const u8,
    state_hash_bytes_len: usize,
) -> *const u8 {
    let state_hash_slice =
        unsafe { std::slice::from_raw_parts(state_hash_pointer, state_hash_bytes_len) };
    let state_hash = Blake2b256Hash::from_bytes(state_hash_slice.to_vec());

    let root = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .history_repository
            .get_history_reader(&state_hash)
            .unwrap()
            .root()
    };

    let hash = hash(&root);
    let hash_proto = HashProto { hash: hash.bytes() };

    let mut bytes = hash_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn get_history_data(
    rspace: *mut Space,
    payload_pointer: *const u8,
    state_hash_bytes_len: usize,
    key_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe {
        std::slice::from_raw_parts(payload_pointer, state_hash_bytes_len + key_bytes_len)
    };
    let (state_hash_slice, key_slice) = payload_slice.split_at(state_hash_bytes_len);

    let state_hash = Blake2b256Hash::from_bytes(state_hash_slice.to_vec());
    let key = Blake2b256Hash::from_bytes(key_slice.to_vec());

    let datums = unsafe {
        let space = (*rspace).rspace.lock().unwrap();
        space
            .history_repository
            .get_history_reader(&state_hash)
            .unwrap()
            .get_data(&key)
            .unwrap()
    };

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
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn get_history_waiting_continuations(
    rspace: *mut Space,
    payload_pointer: *const u8,
    state_hash_bytes_len: usize,
    key_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe {
        std::slice::from_raw_parts(payload_pointer, state_hash_bytes_len + key_bytes_len)
    };
    let (state_hash_slice, key_slice) = payload_slice.split_at(state_hash_bytes_len);

    let state_hash = Blake2b256Hash::from_bytes(state_hash_slice.to_vec());
    let key = Blake2b256Hash::from_bytes(key_slice.to_vec());

    let wks = unsafe {
        let space = (*rspace).rspace.lock().unwrap();
        space
            .history_repository
            .get_history_reader(&state_hash)
            .unwrap()
            .get_continuations(&key)
            .unwrap()
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
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn get_history_joins(
    rspace: *mut Space,
    payload_pointer: *const u8,
    state_hash_bytes_len: usize,
    key_bytes_len: usize,
) -> *const u8 {
    let payload_slice = unsafe {
        std::slice::from_raw_parts(payload_pointer, state_hash_bytes_len + key_bytes_len)
    };
    let (state_hash_slice, key_slice) = payload_slice.split_at(state_hash_bytes_len);

    let state_hash = Blake2b256Hash::from_bytes(state_hash_slice.to_vec());
    let key = Blake2b256Hash::from_bytes(key_slice.to_vec());

    let joins = unsafe {
        let space = (*rspace).rspace.lock().unwrap();
        space
            .history_repository
            .get_history_reader(&state_hash)
            .unwrap()
            .get_joins(&key)
            .unwrap()
    };

    let vec_join: Vec<JoinProto> = joins.into_iter().map(|join| JoinProto { join }).collect();
    let joins_proto = JoinsProto { joins: vec_join };

    let mut bytes = joins_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

/* ReplayRSpace */

#[no_mangle]
pub extern "C" fn replay_produce(
    rspace: *mut Space,
    payload_pointer: *const u8,
    channel_bytes_len: usize,
    data_bytes_len: usize,
    persist: bool,
) -> *const u8 {
    // println!("\nHit produce");

    let payload_slice =
        unsafe { std::slice::from_raw_parts(payload_pointer, channel_bytes_len + data_bytes_len) };
    let (channel_slice, data_slice) = payload_slice.split_at(channel_bytes_len);

    let channel = Par::decode(channel_slice).unwrap();
    let data = ListParWithRandom::decode(data_slice).unwrap();

    let result_option = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .produce(channel, data, persist)
    }
    .unwrap();

    match result_option {
        Some((cont_result, rspace_results, produce_event)) => {
            let protobuf_cont_result = ContResultProto {
                continuation: Some(cont_result.continuation.clone()),
                persistent: cont_result.persistent,
                channels: cont_result.channels,
                patterns: cont_result.patterns,
                peek: cont_result.peek,
            };

            let protobuf_results = rspace_results
                .into_iter()
                .map(|result| RSpaceResultProto {
                    channel: Some(result.channel),
                    matched_datum: Some(result.matched_datum),
                    removed_datum: Some(result.removed_datum),
                    persistent: result.persistent,
                })
                .collect();

            let protobuf_produce = ProduceProto {
                channel_hash: produce_event.channel_hash.bytes(),
                hash: produce_event.hash.bytes(),
                persistent: produce_event.persistent,
                is_deterministic: produce_event.is_deterministic,
                output_value: produce_event.output_value,
            };

            let maybe_action_result = ProduceResult {
                cont_result: Some(protobuf_cont_result),
                results: protobuf_results,
                produce: Some(protobuf_produce),
            };

            let mut bytes = maybe_action_result.encode_to_vec();
            let len = bytes.len() as u32;
            let len_bytes = len.to_le_bytes().to_vec();
            let mut result = len_bytes;
            result.append(&mut bytes);
            let total_len = result.len();
            let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
            ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
            ptr
        }
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn replay_consume(
    rspace: *mut Space,
    payload_pointer: *const u8,
    payload_bytes_len: usize,
) -> *const u8 {
    // println!("\nHit rust consume");

    // let thread_id = thread::current().id();
    // let current_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // println!("Thread ID: {:?}, Current Time: {}", thread_id, current_time);

    let payload_slice = unsafe { std::slice::from_raw_parts(payload_pointer, payload_bytes_len) };
    let consume_params = ConsumeParams::decode(payload_slice).unwrap();

    let channels = consume_params.channels;
    let patterns = consume_params.patterns;
    let continuation = consume_params.continuation.unwrap();
    let persist = consume_params.persist;
    let peeks = consume_params.peeks.into_iter().map(|e| e.value).collect();

    let result_option = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .consume(channels, patterns, continuation, persist, peeks)
    }
    .unwrap();

    match result_option {
        Some((cont_result, rspace_results)) => {
            let protobuf_cont_result = ContResultProto {
                continuation: Some(cont_result.continuation.clone()),
                persistent: cont_result.persistent,
                channels: cont_result.channels,
                patterns: cont_result.patterns,
                peek: cont_result.peek,
            };

            let protobuf_results = rspace_results
                .into_iter()
                .map(|result| RSpaceResultProto {
                    channel: Some(result.channel),
                    matched_datum: Some(result.matched_datum),
                    removed_datum: Some(result.removed_datum),
                    persistent: result.persistent,
                })
                .collect();

            let maybe_action_result = ConsumeResult {
                cont_result: Some(protobuf_cont_result),
                results: protobuf_results,
            };

            let mut bytes = maybe_action_result.encode_to_vec();
            let len = bytes.len() as u32;
            let len_bytes = len.to_le_bytes().to_vec();
            let mut result = len_bytes;
            result.append(&mut bytes);

            // println!("\nlen: {:?}", len);
            let total_len = result.len();
            let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
            ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
            ptr
        }
        None => {
            // println!("\nnone in rust consume");
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn replay_create_checkpoint(rspace: *mut Space) -> *const u8 {
    let checkpoint = unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .create_checkpoint()
            .expect("Rust RSpacePlusPlus Library: Failed to create checkpoint")
    };

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
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn replay_clear(rspace: *mut Space) -> () {
    unsafe {
        (*rspace)
            .rspace
            .lock()
            .unwrap()
            .clear()
            .expect("Rust RSpacePlusPlus Library: Failed to clear");
    }
}

#[no_mangle]
pub extern "C" fn replay_spawn(replay_rspace_ptr: *mut ReplaySpace) -> *mut ReplaySpace {
    let _replay_space = unsafe {
        (*replay_rspace_ptr)
            .replay_space
            .lock()
            .unwrap()
            .spawn()
            .expect("Rust RSpacePlusPlus Library: Failed to spawn")
    };

    Box::into_raw(Box::new(ReplaySpace {
        replay_space: Mutex::new(_replay_space),
    }))
}

/* IReplayRSpace */

// #[no_mangle]
// pub extern "C" fn rig(rspace: *mut Space, log_pointer: *const u8, log_bytes_len: usize) -> () {
//     let log_slice = unsafe { std::slice::from_raw_parts(log_pointer, log_bytes_len) };
//     let log_proto = LogProto::decode(log_slice).unwrap();

//     let log: Vec<Event> = log_proto
//         .log
//         .into_iter()
//         .map(|log_entry| match log_entry.event_type.unwrap() {
//             event_proto::EventType::Comm(comm_proto) => {
//                 let consume_proto = comm_proto.consume.unwrap();
//                 let comm = COMM {
//                     consume: {
//                         Consume {
//                             channel_hashes: {
//                                 consume_proto
//                                     .channel_hashes
//                                     .iter()
//                                     .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
//                                     .collect()
//                             },
//                             hash: Blake2b256Hash::from_bytes(consume_proto.hash),
//                             persistent: consume_proto.persistent,
//                         }
//                     },
//                     produces: {
//                         comm_proto
//                             .produces
//                             .into_iter()
//                             .map(|produce_proto| Produce {
//                                 channel_hash: Blake2b256Hash::from_bytes(
//                                     produce_proto.channel_hash,
//                                 ),
//                                 hash: Blake2b256Hash::from_bytes(produce_proto.hash),
//                                 persistent: produce_proto.persistent,
//                             })
//                             .collect()
//                     },
//                     peeks: {
//                         comm_proto
//                             .peeks
//                             .iter()
//                             .map(|element| element.value)
//                             .collect()
//                     },
//                     times_repeated: {
//                         comm_proto
//                             .times_repeated
//                             .into_iter()
//                             .map(|map_entry| {
//                                 let key_proto = map_entry.key.unwrap();
//                                 let produce = Produce {
//                                     channel_hash: Blake2b256Hash::from_bytes(
//                                         key_proto.channel_hash,
//                                     ),
//                                     hash: Blake2b256Hash::from_bytes(key_proto.hash),
//                                     persistent: key_proto.persistent,
//                                 };

//                                 let value = map_entry.value;

//                                 (produce, value)
//                             })
//                             .collect()
//                     },
//                 };
//                 Event::Comm(comm)
//             }
//             event_proto::EventType::IoEvent(io_event) => match io_event.io_event_type.unwrap() {
//                 io_event_proto::IoEventType::Produce(produce_proto) => {
//                     let produce = Produce {
//                         channel_hash: Blake2b256Hash::from_bytes(produce_proto.channel_hash),
//                         hash: Blake2b256Hash::from_bytes(produce_proto.hash),
//                         persistent: produce_proto.persistent,
//                     };
//                     Event::IoEvent(IOEvent::Produce(produce))
//                 }
//                 io_event_proto::IoEventType::Consume(consume_proto) => {
//                     let consume = Consume {
//                         channel_hashes: {
//                             consume_proto
//                                 .channel_hashes
//                                 .iter()
//                                 .map(|hash| Blake2b256Hash::from_bytes(hash.clone()))
//                                 .collect()
//                         },
//                         hash: Blake2b256Hash::from_bytes(consume_proto.hash),
//                         persistent: consume_proto.persistent,
//                     };
//                     Event::IoEvent(IOEvent::Consume(consume))
//                 }
//             },
//         })
//         .collect();

//     unsafe {
//         (*rspace).rspace.lock().unwrap().rig(log).unwrap();
//     }
// }

// #[no_mangle]
// pub extern "C" fn check_replay_data(rspace: *mut Space) -> () {
//     unsafe {
//         (*rspace)
//             .rspace
//             .lock()
//             .unwrap()
//             .check_replay_data()
//             .unwrap();
//     }
// }

/* Helper Functions */
#[no_mangle]
pub extern "C" fn hash_channel(channel_pointer: *const u8, channel_bytes_len: usize) -> *const u8 {
    let channel_slice = unsafe { std::slice::from_raw_parts(channel_pointer, channel_bytes_len) };
    let channel = Par::decode(channel_slice).unwrap();

    let hash = hash(&channel);
    let hash_proto = HashProto { hash: hash.bytes() };

    let mut bytes = hash_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn hash_channels(
    channels_pointer: *const u8,
    channels_bytes_len: usize,
) -> *const u8 {
    let channels_slice =
        unsafe { std::slice::from_raw_parts(channels_pointer, channels_bytes_len) };
    let channels_proto = ChannelsProto::decode(channels_slice).unwrap();

    let hash = hash_from_vec(&channels_proto.channels);
    let hash_proto = HashProto { hash: hash.bytes() };

    let mut bytes = hash_proto.encode_to_vec();
    let len = bytes.len() as u32;
    let len_bytes = len.to_le_bytes().to_vec();
    let mut result = len_bytes;
    result.append(&mut bytes);
    let total_len = result.len();
    let ptr = Box::leak(result.into_boxed_slice()).as_ptr();
    ALLOCATED_BYTES.fetch_add(total_len, Ordering::SeqCst);
    ptr
}

#[no_mangle]
pub extern "C" fn deallocate_memory(ptr: *mut u8, len: usize) {
    // SAFETY: The caller must guarantee that `ptr` is a valid pointer to a memory block
    // allocated by Rust, and that `len` is the correct size of the block.
    unsafe {
        // Convert the raw pointer back to a Box to allow Rust to deallocate the memory.
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        // The Box goes out of scope here, and Rust automatically deallocates the memory.
    }
    ALLOCATED_BYTES.fetch_sub(len, Ordering::SeqCst);
}
