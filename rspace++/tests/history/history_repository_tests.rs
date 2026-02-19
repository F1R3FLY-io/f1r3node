// See rspace/src/test/scala/coop/rchain/rspace/history/HistoryRepositorySpec.scala

use rand::prelude::SliceRandom;
use rspace_plus_plus::rspace::{
    errors::{HistoryError, RootError},
    hashing::blake2b256_hash::Blake2b256Hash,
    history::{
        history::HistoryInstances, history_repository::HistoryRepository,
        history_repository_impl::HistoryRepositoryImpl, instances::radix_history::RadixHistory,
        root_repository::RootRepository, roots_store::RootsStore,
    },
    hot_store_action::{
        DeleteAction, DeleteContinuations, DeleteData, DeleteJoins, HotStoreAction, InsertAction,
        InsertContinuations, InsertData, InsertJoins,
    },
    internal::{Datum, WaitingContinuation},
    shared::{
        in_mem_key_value_store::InMemoryKeyValueStore,
        trie_exporter::{KeyHash, NodePath, TrieExporter, TrieNode, Value},
        trie_importer::TrieImporter,
    },
    state::{rspace_exporter::RSpaceExporter, rspace_importer::RSpaceImporter},
    trace::event::{Consume, Produce},
};
use shared::rust::{
    ByteVector,
    store::key_value_store::{KeyValueStore, KvStoreError},
};
use std::{
    collections::{BTreeSet, HashSet},
    sync::{Arc, Mutex},
};

use crate::history::history_action_tests::{random_blake, zeros_blake};

#[tokio::test]
async fn history_repository_should_process_insert_one_datum() {
    let repo = create_empty_repository();
    let test_datum = datum(1);
    let insert_data = InsertData {
        channel: test_channel_data_prefix(),
        data: vec![test_datum.clone()],
    };

    let next_repo =
        repo.checkpoint(&vec![HotStoreAction::Insert(InsertAction::InsertData(insert_data))]);
    let history_reader = next_repo.get_history_reader(&next_repo.root());
    let data = history_reader
        .unwrap()
        .base()
        .get_data(&test_channel_data_prefix());
    let fetched = data.first().unwrap().clone();

    assert_eq!(fetched, test_datum);
}

#[tokio::test]
async fn history_repository_should_allow_insert_of_joins_datum_continuation_on_same_channel() {
    let repo = create_empty_repository();
    let channel = test_channel_continuations_prefix();

    let test_datum = datum(1);
    let data = InsertData {
        channel: channel.clone(),
        data: vec![test_datum.clone()],
    };

    let test_joins = join(1);
    let joins = InsertJoins {
        channel: channel.clone(),
        joins: test_joins,
    };

    let test_continuation = continuation(1);
    let continuations = InsertContinuations {
        channels: vec![channel.clone()],
        continuations: vec![test_continuation.clone()],
    };

    let next_repo = repo.checkpoint(&vec![
        HotStoreAction::Insert(InsertAction::InsertData(data)),
        HotStoreAction::Insert(InsertAction::InsertJoins(joins.clone())),
        HotStoreAction::Insert(InsertAction::InsertContinuations(continuations)),
    ]);
    let history_reader = next_repo.get_history_reader(&next_repo.root());
    let reader = history_reader.as_ref().unwrap().base();

    let fetched_data = reader.get_data(&channel);
    let fetched_continuation = reader.get_continuations(&vec![channel.clone()]);
    let fetched_joins = reader.get_joins(&channel);

    assert_eq!(fetched_data.len(), 1);
    assert_eq!(fetched_data.first().unwrap().clone(), test_datum);

    assert_eq!(fetched_continuation.len(), 1);
    assert_eq!(fetched_continuation.first().unwrap().clone(), test_continuation);

    assert_eq!(fetched_joins.len(), 2);
    assert_eq!(
        HashSet::<String>::from_iter(fetched_joins.into_iter().flatten()),
        HashSet::<String>::from_iter(joins.joins.into_iter().flatten())
    );
}

#[tokio::test]
async fn history_repository_should_process_insert_and_delete_of_thirty_mixed_elements() {
    let repo = create_empty_repository();

    let data: (Vec<_>, Vec<_>) = (0..=10).map(insert_datum).unzip();
    let joins: (Vec<_>, Vec<_>) = (0..=10).map(insert_join).unzip();
    let conts: (Vec<_>, Vec<_>) = (0..=10).map(insert_continuation).unzip();

    let mut elems: Vec<_> = [&data.0[..], &joins.0[..], &conts.0[..]].concat();
    let mut rng = rand::thread_rng();
    elems.shuffle(&mut rng);

    let data_delete: Vec<_> = data
        .clone()
        .1
        .into_iter()
        .map(|d| {
            HotStoreAction::<String, String, String, String>::Delete(DeleteAction::DeleteData(
                DeleteData { channel: d.channel },
            ))
        })
        .collect();

    let joins_delete: Vec<_> = joins
        .clone()
        .1
        .into_iter()
        .map(|j| {
            HotStoreAction::Delete(DeleteAction::DeleteJoins(DeleteJoins { channel: j.channel }))
        })
        .collect();

    let conts_delete: Vec<_> = conts
        .clone()
        .1
        .into_iter()
        .map(|c| {
            HotStoreAction::Delete(DeleteAction::DeleteContinuations(DeleteContinuations {
                channels: c.channels,
            }))
        })
        .collect();

    let delete_elems: Vec<_> = [&data_delete[..], &joins_delete[..], &conts_delete[..]].concat();

    let next_repo = repo.checkpoint(&elems);
    let history_reader = next_repo.get_history_reader(&next_repo.root()).unwrap();
    let next_reader = history_reader.base();

    let fetched_data: Vec<Vec<Datum<String>>> = data
        .1
        .iter()
        .map(|d| next_reader.get_data(&d.channel))
        .collect();
    assert_eq!(
        fetched_data,
        data.1
            .clone()
            .into_iter()
            .map(|d| d.data)
            .collect::<Vec<_>>()
    );

    let fetched_conts: Vec<Vec<WaitingContinuation<String, String>>> = conts
        .1
        .iter()
        .map(|c| next_reader.get_continuations(&c.channels))
        .collect();
    assert_eq!(
        fetched_conts,
        conts
            .1
            .clone()
            .into_iter()
            .map(|c| c.continuations)
            .collect::<Vec<_>>()
    );

    let fetched_joins: Vec<Vec<Vec<String>>> = joins
        .1
        .iter()
        .map(|j| next_reader.get_joins(&j.channel))
        .collect();
    let all_joins = HashSet::<String>::from_iter(fetched_joins.into_iter().flatten().flatten());
    let expected_joins: HashSet<String> = joins
        .clone()
        .1
        .into_iter()
        .flat_map(|j: InsertJoins<String>| j.joins.into_iter())
        .flatten()
        .collect();
    assert_eq!(all_joins, expected_joins);

    let deleted_repo = next_repo.checkpoint(&delete_elems);
    let history_reader = deleted_repo
        .get_history_reader(&deleted_repo.root())
        .unwrap();
    let deleted_reader = history_reader.base();

    let fetched_data: Vec<Vec<Datum<String>>> = data
        .1
        .iter()
        .map(|d| next_reader.get_data(&d.channel))
        .collect();
    assert_eq!(
        fetched_data,
        data.1
            .clone()
            .into_iter()
            .map(|d| d.data)
            .collect::<Vec<_>>()
    );

    let fetched_conts: Vec<Vec<WaitingContinuation<String, String>>> = conts
        .1
        .iter()
        .map(|c| next_reader.get_continuations(&c.channels))
        .collect();
    assert_eq!(
        fetched_conts,
        conts
            .1
            .clone()
            .into_iter()
            .map(|c| c.continuations)
            .collect::<Vec<_>>()
    );

    let fetched_joins: Vec<Vec<Vec<String>>> = joins
        .1
        .iter()
        .map(|j| next_reader.get_joins(&j.channel))
        .collect();
    let all_joins = HashSet::<String>::from_iter(fetched_joins.into_iter().flatten().flatten());
    assert_eq!(all_joins, expected_joins);

    let fetched_data: Vec<Vec<Datum<String>>> = data
        .1
        .iter()
        .map(|d| deleted_reader.get_data(&d.channel))
        .collect();
    assert!(fetched_data.iter().flatten().collect::<Vec<_>>().is_empty());

    let fetched_conts: Vec<Vec<WaitingContinuation<String, String>>> = conts
        .1
        .iter()
        .map(|c| deleted_reader.get_continuations(&c.channels))
        .collect();
    assert!(
        fetched_conts
            .iter()
            .flatten()
            .collect::<Vec<_>>()
            .is_empty()
    );

    let fetched_joins: Vec<Vec<Vec<String>>> = joins
        .1
        .iter()
        .map(|j| deleted_reader.get_joins(&j.channel))
        .collect();
    assert!(
        fetched_joins
            .iter()
            .flatten()
            .collect::<Vec<_>>()
            .is_empty()
    );
}

#[tokio::test]
async fn history_repository_should_not_allow_switching_to_a_not_existing_root() {
    let repo = create_empty_repository();

    match repo.reset(&zeros_blake()) {
        Err(HistoryError::RootError(RootError::UnknownRootError(err))) => {
            assert!(err.contains("unknown root"), "Expected error containing 'unknown root', got: {}", err)
        }
        Ok(_) => assert!(false, "Expected a failure"),
        _ => assert!(false, "Wrong error thrown"),
    }
}

#[tokio::test]
async fn history_repository_should_record_next_root_as_valid() {
    let repo = create_empty_repository();
    let test_datum = datum(1);
    let insert_data = InsertData {
        channel: test_channel_data_prefix(),
        data: vec![test_datum.clone()],
    };

    let next_repo =
        repo.checkpoint(&vec![HotStoreAction::Insert(InsertAction::InsertData(insert_data))]);
    let _ = repo.reset(&RadixHistory::empty_root_node_hash());
    let binding = next_repo.history();
    let next_repo_history = binding.lock().expect("Failed to acquire history lock");
    let _ = repo.reset(&next_repo_history.root());
}

fn test_channel_data_prefix() -> String {
    "channel-data".to_string()
}

fn test_channel_joins_prefix() -> String {
    "channel-joins".to_string()
}

fn test_channel_continuations_prefix() -> String {
    "channel-continuations".to_string()
}

fn insert_datum(
    s: i32,
) -> (HotStoreAction<String, String, String, String>, InsertData<String, String>) {
    let insert = InsertData {
        channel: format!("{}{}", test_channel_data_prefix(), s),
        data: vec![datum(s)],
    };

    (HotStoreAction::Insert(InsertAction::InsertData(insert.clone())), insert)
}

fn insert_join(s: i32) -> (HotStoreAction<String, String, String, String>, InsertJoins<String>) {
    let insert = InsertJoins {
        channel: format!("{}{}", test_channel_joins_prefix(), s),
        joins: join(s),
    };

    (HotStoreAction::Insert(InsertAction::InsertJoins(insert.clone())), insert)
}

fn insert_continuation(
    s: i32,
) -> (HotStoreAction<String, String, String, String>, InsertContinuations<String, String, String>) {
    let insert = InsertContinuations {
        channels: vec![format!("{}{}", test_channel_continuations_prefix(), s)],
        continuations: vec![continuation(s)],
    };

    (HotStoreAction::Insert(InsertAction::InsertContinuations(insert.clone())), insert)
}

fn join(s: i32) -> Vec<Vec<String>> {
    vec![
        vec![format!("abc{}", s), format!("def{}", s)],
        vec![format!("wer{}", s), format!("tre{}", s)],
    ]
}

fn continuation(s: i32) -> WaitingContinuation<String, String> {
    WaitingContinuation {
        patterns: vec![format!("pattern-{}", s)],
        continuation: format!("cont-{}", s),
        persist: true,
        peeks: BTreeSet::new(),
        source: Consume {
            channel_hashes: vec![random_blake()],
            hash: random_blake(),
            persistent: true,
        },
    }
}

fn datum(s: i32) -> Datum<String> {
    Datum {
        a: format!("data-{}", s),
        persist: false,
        source: Produce {
            channel_hash: random_blake(),
            hash: random_blake(),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        },
    }
}

fn create_empty_repository() -> HistoryRepositoryImpl<String, String, String, String> {
    let past_roots = root_repository();
    let empty_history = HistoryInstances::create(
        RadixHistory::empty_root_node_hash(),
        Arc::new(InMemoryKeyValueStore::new()),
    )
    .unwrap();

    let _ = past_roots.commit(&RadixHistory::empty_root_node_hash());

    HistoryRepositoryImpl {
        current_history: Arc::new(Mutex::new(Box::new(empty_history))),
        roots_repository: Arc::new(Mutex::new(past_roots)),
        leaf_store: create_inmem_cold_store(),
        rspace_exporter: Arc::new(EmptyExporter),
        rspace_importer: Arc::new(EmptyImporter),
        _marker: std::marker::PhantomData,
    }
}

struct InmemRootsStore {
    roots: Arc<Mutex<HashSet<Blake2b256Hash>>>,
    maybe_current_root: Arc<Mutex<Option<Blake2b256Hash>>>,
}

impl RootsStore for InmemRootsStore {
    fn current_root(&self) -> Result<Option<Blake2b256Hash>, RootError> {
        Ok(self.maybe_current_root.lock().unwrap().clone())
    }

    fn validate_and_set_current_root(
        &self,
        key: Blake2b256Hash,
    ) -> Result<Option<Blake2b256Hash>, RootError> {
        let roots_lock = self.roots.lock().unwrap();
        let mut maybe_current_root_lock = self.maybe_current_root.lock().unwrap();

        if roots_lock.contains(&key) {
            *maybe_current_root_lock = Some(key);
            Ok(maybe_current_root_lock.clone())
        } else {
            Ok(None)
        }
    }

    fn record_root(&self, key: &Blake2b256Hash) -> Result<(), RootError> {
        let mut roots_lock = self.roots.lock().unwrap();
        let mut maybe_current_root_lock = self.maybe_current_root.lock().unwrap();

        *maybe_current_root_lock = Some(key.clone());
        let _ = roots_lock.insert(key.clone());
        Ok(())
    }
}

impl InmemRootsStore {
    fn new() -> InmemRootsStore {
        InmemRootsStore {
            roots: Arc::new(Mutex::new(HashSet::new())),
            maybe_current_root: Arc::new(Mutex::new(None)),
        }
    }
}

fn root_repository() -> RootRepository {
    RootRepository {
        roots_store: Box::new(InmemRootsStore::new()),
    }
}

fn create_inmem_cold_store() -> Arc<dyn KeyValueStore> {
    Arc::new(InMemoryKeyValueStore::new())
}

struct EmptyExporter;

impl RSpaceExporter for EmptyExporter {
    fn get_root(&self) -> Result<KeyHash, RootError> {
        todo!()
    }
}

impl TrieExporter for EmptyExporter {
    fn get_nodes(&self, _start_path: NodePath, _skip: i32, _take: i32) -> Vec<TrieNode<KeyHash>> {
        todo!()
    }

    fn get_history_items(
        &self,
        _keys: Vec<KeyHash>,
    ) -> Result<Vec<(KeyHash, Value)>, KvStoreError> {
        todo!()
    }

    fn get_data_items(&self, _keys: Vec<KeyHash>) -> Result<Vec<(KeyHash, Value)>, KvStoreError> {
        todo!()
    }
}

struct EmptyImporter;

impl RSpaceImporter for EmptyImporter {
    fn get_history_item(&self, _hash: KeyHash) -> Option<ByteVector> {
        todo!()
    }
}

impl TrieImporter for EmptyImporter {
    fn set_history_items(&self, _data: Vec<(KeyHash, Value)>) -> () {
        todo!()
    }

    fn set_data_items(&self, _data: Vec<(KeyHash, Value)>) -> () {
        todo!()
    }

    fn set_root(&self, _key: &KeyHash) -> () {
        todo!()
    }
}
