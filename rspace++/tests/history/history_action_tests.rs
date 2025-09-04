use rand::distributions::{Alphanumeric, DistString};
use rspace_plus_plus::rspace::errors::{HistoryError, RadixTreeError};
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use shared::rust::store::key_value_store::KeyValueStore;
use shared::rust::{Byte, ByteVector};

// See rspace/src/test/scala/coop/rchain/rspace/history/HistoryActionTests.scala

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::sync::{Arc, Mutex};

use rand::Rng;
use rand::seq::SliceRandom;
use rspace_plus_plus::rspace::{
    history::{
        history::{History, HistoryInstances},
        history_action::{DeleteAction, HistoryAction, HistoryActionTrait, InsertAction, KeyPath},
        instances::radix_history::RadixHistory,
    },
    shared::in_mem_key_value_store::InMemoryKeyValueStore,
};

#[test]
fn creating_and_read_one_record_should_work() {
    let (history_action, insert_action) = history_insert(zeros());

    let empty_history = create_empty_history();
    let new_history = empty_history.process(vec![history_action]).unwrap();
    let read_value = new_history.read(zeros());

    assert!(read_value.is_ok());
    assert_eq!(read_value.unwrap(), Some(insert_action.hash.bytes()));
}

#[test]
fn reset_method_of_history_should_work() {
    let (history_action, insert_action) = history_insert(zeros());

    let empty_history = create_empty_history();
    let new_history = empty_history.process(vec![history_action]).unwrap();
    let history_one_reset = empty_history.reset(&new_history.root()).unwrap();
    let read_value = history_one_reset.read(zeros());

    assert!(read_value.is_ok());
    assert_eq!(read_value.unwrap(), Some(insert_action.hash.bytes()));
}

#[test]
fn creating_ten_records_should_work() {
    let (history_actions, insert_actions): (Vec<HistoryAction>, Vec<InsertAction>) =
        (0..10).map(zeros_and).map(history_insert).unzip();

    let empty_history = create_empty_history();
    let new_history = empty_history.process(history_actions.clone()).unwrap();

    let read_values: Vec<Option<ByteVector>> = history_actions
        .iter()
        .map(|action| new_history.read(action.key()))
        .map(|read_result| {
            assert!(read_result.is_ok());
            read_result.unwrap()
        })
        .collect();

    let expected_data: Vec<Option<ByteVector>> = insert_actions
        .iter()
        .map(|action| Some(action.hash.bytes()))
        .collect();
    assert_eq!(read_values, expected_data);
}

#[test]
fn history_should_allow_to_store_different_length_key_records_in_different_branches() {
    let (history_actions, insert_actions): (Vec<HistoryAction>, Vec<InsertAction>) = vec![
        history_insert(hex_key("01")),
        history_insert(hex_key("02")),
        history_insert(hex_key("0001")),
        history_insert(hex_key("0002")),
    ]
    .into_iter()
    .unzip();

    let empty_history = create_empty_history();
    let new_history = empty_history.process(history_actions.clone()).unwrap();

    let read_values: Vec<Option<ByteVector>> = history_actions
        .iter()
        .map(|action| new_history.read(action.key()))
        .map(|read_result| {
            assert!(read_result.is_ok());
            read_result.unwrap()
        })
        .collect();

    let expected_data: Vec<Option<ByteVector>> = insert_actions
        .iter()
        .map(|action| Some(action.hash.bytes()))
        .collect();
    assert_eq!(read_values, expected_data);
}

// // TODO: Don't works for MergingHistory
#[test]
fn deletion_of_a_non_existing_records_should_not_throw_error() {
    let (history_action1, _) = history_insert(hex_key("0011"));
    let (history_action2, _): (Vec<HistoryAction>, Vec<DeleteAction>) =
        vec![history_delete(hex_key("0012")), history_delete(hex_key("0011"))]
            .into_iter()
            .unzip();

    let empty_history = create_empty_history();
    let history_one = empty_history.process(vec![history_action1]).unwrap();
    let err = history_one.process(history_action2);

    assert!(err.is_ok());
}

#[test]
fn history_should_not_allow_to_store_different_length_key_records_in_same_branch() {
    let (history_actions, _): (Vec<HistoryAction>, Vec<InsertAction>) =
        vec![history_insert(hex_key("01")), history_insert(hex_key("0100"))]
            .into_iter()
            .unzip();
    let empty_history = create_empty_history();
    let err = empty_history.process(history_actions);

    assert!(err.is_err());
    match err {
        Err(HistoryError::RadixTreeError(RadixTreeError::PrefixError(message))) => {
            assert_eq!(message, "The length of all prefixes in the subtree must be the same.");
        }
        _ => panic!("Expected PrefixError variant"),
    }

    // TODO: For MergingHistory
    // ex shouldBe a[RuntimeException]
    // ex.getMessage shouldBe s"malformed trie"
}

#[test]
fn history_should_not_allow_to_process_history_actions_with_same_keys() {
    let (history_actions1, _): (Vec<HistoryAction>, Vec<InsertAction>) =
        vec![history_insert(zeros()), history_insert(zeros())]
            .into_iter()
            .unzip();

    let empty_history = create_empty_history();
    let err1 = empty_history.process(history_actions1);

    assert!(err1.is_err());
    match err1 {
        Err(HistoryError::ActionError(message)) => {
            assert_eq!(message, "Cannot process duplicate actions on one key.");
        }
        _ => panic!("Expected ActionError variant"),
    }

    let data2 = vec![history_insert(zeros()).0, history_delete(zeros()).0];
    let empty_history = create_empty_history();
    let err2 = empty_history.process(data2);

    assert!(err2.is_err());
    match err2 {
        Err(HistoryError::ActionError(message)) => {
            assert_eq!(message, "Cannot process duplicate actions on one key.");
        }
        _ => panic!("Expected ActionError variant"),
    }
}

#[test]
fn history_after_deleting_all_records_should_be_empty() {
    let insertions = vec![history_insert(zeros()).0];
    let deletions = vec![history_delete(zeros()).0];

    let empty_history = create_empty_history();
    let history_one = empty_history.process(insertions);
    assert!(history_one.is_ok());

    let history_two = history_one.unwrap().process(deletions);
    assert!(history_two.is_ok());
    assert_eq!(history_two.unwrap().root(), empty_history.root());
}

#[test]
fn reading_of_a_non_existing_records_should_return_none() {
    let key = hex_key("0011");
    let empty_history = create_empty_history();
    let read_value = empty_history.read(key);
    assert!(read_value.is_ok());
    assert!(read_value.unwrap().is_none());
}

#[test]
fn update_of_a_record_should_not_change_past_history() {
    let insert_one = vec![history_insert(zeros()).0];
    let insert_two = vec![history_insert(zeros()).0];

    let empty_history = create_empty_history();
    let history_one = empty_history.process(insert_one);
    assert!(history_one.is_ok());
    let read_value_one_pre = history_one.as_ref().unwrap().read(zeros());

    let history_two = history_one.as_ref().unwrap().process(insert_two);
    assert!(history_two.is_ok());
    let read_value_one_post = history_one.unwrap().read(zeros());
    let read_value_two = history_two.unwrap().read(zeros());

    assert!(read_value_one_pre.is_ok());
    assert!(read_value_one_post.is_ok());
    assert_eq!(read_value_one_pre.as_ref().unwrap(), &read_value_one_post.unwrap());
    assert_ne!(read_value_one_pre.unwrap(), read_value_two.unwrap());
}

#[test]
fn history_should_correctly_build_the_same_trees_in_different_ways() {
    let insert_one = vec![history_insert(hex_key("010000")).0, history_insert(hex_key("0200")).0];
    let insert_two = vec![history_insert(hex_key("010001")).0, history_insert(hex_key("0300")).0];
    let mut insert_one_two = insert_one.clone();
    insert_one_two.extend(insert_two.clone());

    let delete_one = vec![
        history_delete(insert_one.first().unwrap().key()).0,
        history_delete(insert_one.get(1).unwrap().key()).0,
    ];
    let delete_two = vec![
        history_delete(insert_two.first().unwrap().key()).0,
        history_delete(insert_two.get(1).unwrap().key()).0,
    ];
    let mut delete_one_two = delete_one.clone();
    delete_one_two.extend(delete_two.clone());

    let empty_history = create_empty_history();

    let history_one = empty_history.process(insert_one);
    let history_two = empty_history.process(insert_two.clone());

    let history_one_two = empty_history.process(insert_one_two);
    let history_one_two_another_way = history_one.as_ref().unwrap().process(insert_two);

    assert_eq!(
        history_one_two.as_ref().unwrap().root(),
        history_one_two_another_way.unwrap().root()
    );

    let history_one_another_way = history_one_two.as_ref().unwrap().process(delete_two);
    assert!(history_one_another_way.is_ok());
    assert_eq!(&history_one.unwrap().root(), &history_one_another_way.unwrap().root());

    let history_two_another_way = history_one_two.as_ref().unwrap().process(delete_one);
    assert!(history_two_another_way.is_ok());
    assert_eq!(&history_two.unwrap().root(), &history_two_another_way.unwrap().root());

    let empty_history_another_way = history_one_two.unwrap().process(delete_one_two);
    assert!(empty_history_another_way.is_ok());
    assert_eq!(&empty_history.root(), &empty_history_another_way.unwrap().root());
}

#[test]
fn adding_already_existing_records_should_not_change_history() {
    let inserts = vec![history_insert(zeros()).0];

    let (empty_history, in_mem_store) = create_empty_history_and_store();

    let in_mem_store_lock = in_mem_store
        .lock()
        .expect("History Action Tests: Failed to aquire lock on in_mem_store");
    let empty_history_size = in_mem_store_lock.size_bytes();
    drop(in_mem_store_lock);

    let history_one = empty_history.process(inserts.clone());
    assert!(history_one.is_ok());

    let in_mem_store_lock = in_mem_store
        .lock()
        .expect("History Action Tests: Failed to aquire lock on in_mem_store");
    let history_one_size = in_mem_store_lock.size_bytes();
    drop(in_mem_store_lock);

    let history_two = history_one.as_ref().unwrap().process(inserts);
    assert!(history_two.is_ok());

    let in_mem_store_lock = in_mem_store
        .lock()
        .expect("History Action Tests: Failed to aquire lock on in_mem_store");
    let history_two_size = in_mem_store_lock.size_bytes();
    drop(in_mem_store_lock);

    assert_eq!(history_one.unwrap().root(), history_two.unwrap().root());
    assert_eq!(empty_history_size, 0_usize);
    assert_eq!(history_one_size, history_two_size);
}

#[test]
fn collision_detecting_in_kvdb_should_work() {
    let insert_record = vec![history_insert(zeros()).0];
    let delete_record = vec![history_delete(zeros()).0];
    let collision_kv_pair = (RadixHistory::empty_root_node_hash().bytes(), random_blake().bytes());

    let (empty_history, in_mem_store) = create_empty_history_and_store();
    let new_history = empty_history.process(insert_record);
    assert!(new_history.is_ok());

    let mut in_mem_store_lock = in_mem_store
        .lock()
        .expect("History Action Tests: Failed to aquire lock on in_mem_store");
    assert!(in_mem_store_lock.put(vec![collision_kv_pair]).is_ok());
    drop(in_mem_store_lock);

    let err = new_history.unwrap().process(delete_record);
    let in_mem_store_lock = in_mem_store
        .lock()
        .expect("History Action Tests: Failed to aquire lock on in_mem_store");
    drop(in_mem_store_lock);
    assert!(err.is_err());

    match err {
        Err(HistoryError::RadixTreeError(RadixTreeError::CollisionError(message))) => {
            assert_eq!(
                message,
                format!(
                    "1 collisions in KVDB (first collision with key = {}.",
                    hex::encode(RadixHistory::empty_root_node_hash().bytes())
                )
            );
        }
        _ => panic!("Expected CollisionError variant"),
    }
}

#[test]
fn randomly_insert_or_delete_should_return_the_correct_result() {
    let size_inserts = 10000;
    let size_deletes = 3000;
    let size_updates = 1000;
    let state: BTreeMap<KeyPath, Blake2b256Hash> = BTreeMap::new();
    let inserts = generate_random_insert(0);

    let empty_history = create_empty_history();

    let res: Result<(Box<dyn History>, Vec<HistoryAction>, BTreeMap<Vec<u8>, Blake2b256Hash>), _> =
        (1..=10).try_fold((empty_history, inserts, state), |(history, inserts, state), _| {
            let new_inserts = generate_random_insert(size_inserts);
            let new_updates = generate_random_insert_from_insert(size_updates, &inserts);

            let update_keys: HashSet<KeyPath> =
                new_updates.iter().map(|action| action.key()).collect();

            let last_inserts: Vec<HistoryAction> = inserts
                .into_iter()
                .filter(|i| !update_keys.contains(&i.key()))
                .collect();

            let new_deletes = generate_random_delete_from_insert(size_deletes, last_inserts)
                .into_iter()
                .chain(generate_random_delete(size_deletes).into_iter())
                .collect::<Vec<_>>();

            let actions = new_inserts
                .clone()
                .into_iter()
                .chain(new_deletes.clone().into_iter())
                .chain(new_updates.into_iter())
                .collect::<Vec<_>>();

            println!("\nprocess {}", actions.len());
            let new_history = history.process(actions.clone());
            assert!(new_history.is_ok());
            let new_state = update_state(state, &actions);
            println!("\nThe new state size is {}", new_state.len());

            for (k, value) in new_state.iter() {
                match new_history.as_ref().unwrap().read(k.to_vec()) {
                    Ok(Some(v)) => assert!(v == value.bytes()),
                    Err(e) => {
                        println!("{:?}", e);
                        assert!(false, "Can not get value")
                    }
                    Ok(None) => {
                        assert!(false, "Can not get value")
                    }
                }
            }

            for d in new_deletes.iter() {
                match new_history.as_ref().unwrap().read(d.key()) {
                    Ok(None) => assert!(true),
                    _ => assert!(false, "got empty pointer after remove"),
                }
            }

            Ok::<
                (Box<dyn History>, Vec<HistoryAction>, BTreeMap<Vec<u8>, Blake2b256Hash>),
                Box<dyn Error>,
            >((new_history.unwrap(), new_inserts, new_state))
        });

    assert!(res.is_ok());
}

fn create_empty_history() -> Box<dyn History> {
    let empty_history = HistoryInstances::create(
        RadixHistory::empty_root_node_hash(),
        Arc::new(Mutex::new(Box::new(InMemoryKeyValueStore::new()))),
    )
    .expect("History Actions Tests: Failed to create empty history");

    Box::new(empty_history)
}

fn create_empty_history_and_store() -> (Box<dyn History>, Arc<Mutex<Box<dyn KeyValueStore>>>) {
    let store: Arc<Mutex<Box<dyn KeyValueStore>>> =
        Arc::new(Mutex::new(Box::new(InMemoryKeyValueStore::new())));

    let empty_history: Box<dyn History> = Box::new(
        HistoryInstances::create(RadixHistory::empty_root_node_hash(), store.clone())
            .expect("History Actions Tests: Failed to create empty history and store"),
    );

    (empty_history, store)
}

fn random_key(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            let num = rng.gen_range(-128..=127) as i8;
            num as u8
        })
        .collect()
}

fn generate_random_insert(size: usize) -> Vec<HistoryAction> {
    (0..size)
        .map(|_| history_insert(random_key(32)).0)
        .collect()
}

fn generate_random_delete(size: usize) -> Vec<HistoryAction> {
    (0..size)
        .map(|_| history_delete(random_key(32)).0)
        .collect()
}

fn generate_random_delete_from_insert(
    size: usize,
    inserts: Vec<HistoryAction>,
) -> Vec<HistoryAction> {
    let mut rng = rand::thread_rng();
    let mut shuffled_inserts = inserts;
    shuffled_inserts.shuffle(&mut rng);
    shuffled_inserts
        .into_iter()
        .take(size)
        .map(|i| history_delete(i.key()).0)
        .collect()
}

fn generate_random_insert_from_insert(
    size: usize,
    inserts: &Vec<HistoryAction>,
) -> Vec<HistoryAction> {
    let mut rng = rand::thread_rng();
    let mut shuffled_inserts = inserts.clone();
    shuffled_inserts.shuffle(&mut rng);
    shuffled_inserts
        .into_iter()
        .take(size)
        .map(|i| history_insert(i.key()).0)
        .collect()
}

fn update_state(
    mut state: BTreeMap<KeyPath, Blake2b256Hash>,
    actions: &Vec<HistoryAction>,
) -> BTreeMap<KeyPath, Blake2b256Hash> {
    for action in actions {
        match action {
            HistoryAction::Insert(InsertAction { key, hash }) => {
                state.insert(key.clone(), hash.clone());
            }
            HistoryAction::Delete(DeleteAction { key }) => {
                state.remove(&key.clone());
            }
        }
    }
    state
}

fn zeros() -> KeyPath {
    vec![0; 32]
}

fn thirty_one_zeros() -> KeyPath {
    vec![0; 31]
}

fn zeros_and(i: u8) -> KeyPath {
    let mut key_path = thirty_one_zeros();
    key_path.push(i);
    key_path
}

fn hex_key(s: &str) -> Vec<Byte> {
    hex::decode(s).unwrap()
}

fn history_insert(k: KeyPath) -> (HistoryAction, InsertAction) {
    let insert_action = InsertAction {
        key: k,
        hash: random_blake(),
    };

    (HistoryAction::Insert(insert_action.clone()), insert_action)
}

fn history_delete(k: KeyPath) -> (HistoryAction, DeleteAction) {
    let delete_action = DeleteAction { key: k };
    (HistoryAction::Delete(delete_action.clone()), delete_action)
}

pub fn random_blake() -> Blake2b256Hash {
    Blake2b256Hash::new(
        &Alphanumeric
            .sample_string(&mut rand::thread_rng(), 32)
            .into_bytes(),
    )
}

pub fn zeros_blake() -> Blake2b256Hash {
    let zero_bytes = vec![0u8; 32];
    Blake2b256Hash::new(&zero_bytes)
}
