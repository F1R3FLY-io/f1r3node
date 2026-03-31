// See rspace/src/test/scala/coop/rchain/rspace/StorageActionsTests.scala

use std::collections::{BTreeSet, HashSet, LinkedList};
use std::hash::Hash;
use std::sync::Arc;

use proptest::prelude::*;
use rspace_plus_plus::rspace::history::instances::radix_history::RadixHistory;
use rspace_plus_plus::rspace::hot_store_action::{
    HotStoreAction, InsertAction, InsertContinuations, InsertData,
};
use rspace_plus_plus::rspace::internal::{Datum, WaitingContinuation};
use rspace_plus_plus::rspace::r#match::Match;
use rspace_plus_plus::rspace::rspace::RSpace;
use rspace_plus_plus::rspace::rspace_interface::{ContResult, ISpace, RSpaceResult};
use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use rspace_plus_plus::rspace::trace::event::{Consume, Produce};
use rspace_plus_plus::rspace::util::{unpack_produce_tuple, unpack_tuple};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

// See rspace/src/main/scala/coop/rchain/rspace/examples/StringExamples.scala
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
enum Pattern {
    #[default]
    Wildcard,
    StringMatch(String),
}

#[derive(Clone)]
struct StringMatch;

impl Match<Pattern, String> for StringMatch {
    fn get(&self, p: Pattern, a: String) -> Option<String> {
        match p {
            Pattern::Wildcard => Some(a),
            Pattern::StringMatch(value) => {
                if value == a {
                    Some(a)
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
struct StringsCaptor {
    id: u64,
    res: LinkedList<Vec<String>>,
}

impl StringsCaptor {
    fn new() -> Self {
        StringsCaptor {
            id: 0,
            res: LinkedList::new(),
        }
    }

    fn with_id(id: u64) -> Self {
        StringsCaptor {
            id,
            res: LinkedList::new(),
        }
    }

    fn run_k(&mut self, data: Vec<String>) { self.res.push_back(data); }

    fn results(&self) -> Vec<Vec<String>> { self.res.iter().cloned().collect() }
}

// We only care that both vectors contain the same elements, not their ordering
fn check_same_elements<T: Hash + Eq>(vec1: Vec<T>, vec2: Vec<T>) -> bool {
    let set1: HashSet<_> = vec1.into_iter().collect();
    let set2: HashSet<_> = vec2.into_iter().collect();
    set1 == set2
}

// See rspace/src/main/scala/coop/rchain/rspace/util/package.scala
fn run_k<C, P>(
    cont: Option<(ContResult<C, P, StringsCaptor>, Vec<RSpaceResult<C, String>>)>,
) -> Vec<Vec<String>> {
    let mut cont_unwrapped = cont.unwrap();
    let unpacked_tuple = unpack_tuple(&cont_unwrapped);
    cont_unwrapped.0.continuation.run_k(unpacked_tuple.1);
    let cont_results = cont_unwrapped.0.continuation.results();
    let cloned_results: Vec<Vec<String>> = cont_results
        .iter()
        .map(|res| res.iter().map(|s| s.to_string()).collect())
        .collect();
    cloned_results
}

fn run_produce_k<C, P>(
    cont: Option<(ContResult<C, P, StringsCaptor>, Vec<RSpaceResult<C, String>>, Produce)>,
) -> Vec<Vec<String>> {
    let mut cont_unwrapped = cont.unwrap();
    let unpacked_tuple = unpack_produce_tuple(&cont_unwrapped);
    cont_unwrapped.0.continuation.run_k(unpacked_tuple.1);
    let cont_results = cont_unwrapped.0.continuation.results();
    let cloned_results: Vec<Vec<String>> = cont_results
        .iter()
        .map(|res| res.iter().map(|s| s.to_string()).collect())
        .collect();
    cloned_results
}

pub fn filter_enum_variants<C: Clone, P: Clone, A: Clone, K: Clone, V>(
    vec: Vec<HotStoreAction<C, P, A, K>>,
    variant: fn(HotStoreAction<C, P, A, K>) -> Option<V>,
) -> Vec<V> {
    vec.into_iter().filter_map(variant).collect()
}

async fn create_rspace() -> RSpace<String, Pattern, String, StringsCaptor> {
    let mut kvm = InMemoryStoreManager::new();
    // let mut kvm = mk_rspace_store_manager((&format!("{}/rspace++/",
    // "./tests/storage_actions_test_lmdb")).into(), 1 * GB);
    let store = kvm.r_space_stores().await.unwrap();

    RSpace::create(store, Arc::new(Box::new(StringMatch))).unwrap()
}

// NOTE: not implementing test checks for Log
#[tokio::test]
async fn produce_should_persist_data_in_store() {
    let mut rspace = create_rspace().await;

    let channel = "ch1".to_string();
    let key = vec![channel.clone()];

    let r = rspace.produce(key[0].clone(), "datum".to_string(), false);
    let data = rspace.store.get_data(&channel);
    assert_eq!(data, vec![Datum::create(&channel, "datum".to_string(), false)]);

    let cont = rspace.store.get_continuations(&key);
    assert_eq!(cont.len(), 0);
    assert!(r.unwrap().is_none());

    let insert_data: Vec<InsertData<_, _>> = filter_enum_variants(rspace.store.changes(), |e| {
        if let HotStoreAction::Insert(InsertAction::InsertData(d)) = e {
            Some(d)
        } else {
            None
        }
    });
    assert_eq!(insert_data.len(), 1);
    assert_eq!(
        insert_data
            .into_iter()
            .map(|d| d.channel)
            .collect::<String>(),
        channel
    );
}

#[tokio::test]
async fn producing_twice_on_same_channel_should_persist_two_pieces_of_data_in_store() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let key = vec![channel.clone()];

    let r1 = rspace.produce(key[0].clone(), "datum1".to_string(), false);
    let d1 = rspace.store.get_data(&channel);
    assert_eq!(d1, vec![Datum::create(&channel, "datum1".to_string(), false)]);

    let wc1 = rspace.store.get_continuations(&key.clone());
    assert_eq!(wc1.len(), 0);
    assert!(r1.unwrap().is_none());

    let r2 = rspace.produce(key[0].clone(), "datum2".to_string(), false);
    let d2 = rspace.store.get_data(&channel);
    assert!(check_same_elements(d2, vec![
        Datum::create(&channel, "datum1".to_string(), false),
        Datum::create(&channel, "datum2".to_string(), false)
    ]));

    let wc2 = rspace.store.get_continuations(&key.clone());
    assert_eq!(wc2.len(), 0);
    assert!(r2.unwrap().is_none());

    let insert_data: Vec<InsertData<_, _>> = filter_enum_variants(rspace.store.changes(), |e| {
        if let HotStoreAction::Insert(InsertAction::InsertData(d)) = e {
            Some(d)
        } else {
            None
        }
    });
    assert_eq!(insert_data.len(), 1);
    assert_eq!(
        insert_data
            .into_iter()
            .map(|d| d.channel)
            .collect::<String>(),
        channel
    );
}

#[tokio::test]
async fn consuming_on_one_channel_should_persist_continuation_in_store() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let key = vec![channel.clone()];
    let patterns = vec![Pattern::Wildcard];

    let r = rspace.consume(key.clone(), patterns, StringsCaptor::new(), false, BTreeSet::default());
    let d1 = rspace.store.get_data(&channel);
    assert_eq!(d1.len(), 0);

    let c1 = rspace.store.get_continuations(&key.clone());
    assert_ne!(c1.len(), 0);
    assert!(r.unwrap().is_none());

    let insert_continuations: Vec<InsertContinuations<_, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(InsertAction::InsertContinuations(c)) = e {
                Some(c)
            } else {
                None
            }
        });
    assert_eq!(insert_continuations.len(), 1);
    assert_eq!(
        insert_continuations
            .into_iter()
            .map(|c| c.channels)
            .flatten()
            .collect::<Vec<String>>(),
        key
    );
}

#[tokio::test]
async fn consuming_on_three_channels_should_persist_continuation_in_store() {
    let mut rspace = create_rspace().await;
    let key = vec!["ch1".to_string(), "ch2".to_string(), "ch3".to_string()];
    let patterns = vec![Pattern::Wildcard, Pattern::Wildcard, Pattern::Wildcard];

    let r = rspace.consume(key.clone(), patterns, StringsCaptor::new(), false, BTreeSet::default());
    let results: Vec<_> = key.iter().map(|k| rspace.store.get_data(k)).collect();
    for seq in &results {
        assert!(seq.is_empty(), "d should be empty");
    }

    let c1 = rspace.store.get_continuations(&key);
    assert_ne!(c1.len(), 0);
    assert!(r.unwrap().is_none());

    let insert_continuations: Vec<InsertContinuations<_, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(InsertAction::InsertContinuations(c)) = e {
                Some(c)
            } else {
                None
            }
        });
    assert_eq!(insert_continuations.len(), 1);
}

#[tokio::test]
async fn producing_then_consuming_on_same_channel_should_return_continuation_and_data() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let key = vec![channel.clone()];

    let r1 = rspace.produce(channel.clone(), "datum".to_string(), false);
    let d1 = rspace.store.get_data(&channel);
    assert_eq!(d1, vec![Datum::create(&channel, "datum".to_string(), false)]);

    let c1 = rspace.store.get_continuations(&key.clone());
    assert_eq!(c1.len(), 0);
    assert!(r1.unwrap().is_none());

    let r2 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let d2 = rspace.store.get_data(&channel);
    assert_eq!(d2.len(), 0);

    let c2 = rspace.store.get_continuations(&key);
    assert_eq!(c2.len(), 0);
    assert!(r2.clone().unwrap().is_some());

    let cont_results = run_k(r2.unwrap());
    assert!(check_same_elements(cont_results, vec![vec!["datum".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn producing_then_consuming_on_same_channel_with_peek_should_return_continuation_and_data_and_preserve_peeked_data()
 {
    // Peek semantics: peeked channels should NOT have their data removed.
    // Both the consume path (store_persistent_data) and the produce path
    // (remove_matched_datum_and_join) now honor the peeks set.
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let key = vec![channel.clone()];

    let r1 = rspace.produce(channel.clone(), "datum".to_string(), false);
    let d1 = rspace.store.get_data(&channel);
    assert_eq!(d1, vec![Datum::create(&channel, "datum".to_string(), false)]);

    let c1 = rspace.store.get_continuations(&key.clone());
    assert_eq!(c1.len(), 0);
    assert!(r1.unwrap().is_none());

    let r2 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        std::iter::once(0).collect(),
    );
    let d2 = rspace.store.get_data(&channel);
    assert_eq!(d2.len(), 1);

    let c2 = rspace.store.get_continuations(&key);
    assert_eq!(c2.len(), 0);
    assert!(r2.clone().unwrap().is_some());

    let cont_results = run_k(r2.unwrap());
    assert!(check_same_elements(cont_results, vec![vec!["datum".to_string()]]));

    // With peek semantics, data is preserved so there will be insert actions
    // for the remaining datum. We just verify the continuation fired correctly.
}

#[tokio::test]
async fn consuming_then_producing_on_same_channel_with_peek_should_return_continuation_and_data_and_preserve_peeked_data()
 {
    // Peek semantics: in the consume-then-produce path, the produce matches
    // the waiting peek continuation on-the-fly (datum_index = -1). Since the
    // channel is peeked, the data must persist for future consumers, so RSpace
    // stores it during remove_matched_datum_and_join.
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let key = vec![channel.clone()];

    let r1 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        std::iter::once(0).collect(),
    );
    assert!(r1.unwrap().is_none());
    let c1 = rspace.store.get_continuations(&key.clone());
    assert_eq!(c1.len(), 1);

    let r2 = rspace.produce(channel.clone(), "datum".to_string(), false);
    let d1 = rspace.store.get_data(&channel);
    assert_eq!(d1.len(), 1);

    let c2 = rspace.store.get_continuations(&key);
    assert_eq!(c2.len(), 0);
    assert!(r2.clone().unwrap().is_some());

    let cont_results = run_produce_k(r2.unwrap());
    assert!(check_same_elements(cont_results, vec![vec!["datum".to_string()]]));
}

#[tokio::test]
async fn consuming_then_producing_on_same_channel_with_persistent_flag_should_return_continuation_and_data_and_not_insert_persistent_data()
 {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let key = vec![channel.clone()];

    let r1 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r1.unwrap().is_none());
    let c1 = rspace.store.get_continuations(&key.clone());
    assert_eq!(c1.len(), 1);

    let r2 = rspace.produce(channel.clone(), "datum".to_string(), true);
    let d1 = rspace.store.get_data(&channel);
    assert!(d1.is_empty());

    let c2 = rspace.store.get_continuations(&key);
    assert_eq!(c2.len(), 0);
    assert!(r2.clone().unwrap().is_some());

    let cont_results = run_produce_k(r2.unwrap());
    assert!(check_same_elements(cont_results, vec![vec!["datum".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn producing_three_times_then_consuming_three_times_should_work() {
    let mut rspace = create_rspace().await;
    let possible_cont_results =
        vec![vec!["datum1".to_string()], vec!["datum2".to_string()], vec!["datum3".to_string()]];

    let r1 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r2 = rspace.produce("ch1".to_string(), "datum2".to_string(), false);
    let r3 = rspace.produce("ch1".to_string(), "datum3".to_string(), false);
    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());
    assert!(r3.unwrap().is_none());

    let r4 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let cont_results_r4 = run_k(r4.unwrap());
    assert!(
        possible_cont_results
            .iter()
            .any(|v| cont_results_r4.contains(v))
    );

    let r5 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let cont_results_r5 = run_k(r5.unwrap());
    assert!(
        possible_cont_results
            .iter()
            .any(|v| cont_results_r5.contains(v))
    );

    let r6 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let cont_results_r6 = run_k(r6.unwrap());
    assert!(
        possible_cont_results
            .iter()
            .any(|v| cont_results_r6.contains(v))
    );

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

// NOTE: Still not quite sure how this one works
//       The test is setup correctly though
#[tokio::test]
async fn producing_on_channel_then_consuming_on_that_channel_and_another_then_producing_on_other_channel_should_return_continuation_and_all_data()
 {
    let mut rspace = create_rspace().await;
    let produce_key_1 = vec!["ch1".to_string()];
    let produce_key_2 = vec!["ch2".to_string()];
    let consume_key = vec!["ch1".to_string(), "ch2".to_string()];
    let consume_pattern = vec![Pattern::Wildcard, Pattern::Wildcard];

    let r1 = rspace.produce(produce_key_1[0].clone(), "datum1".to_string(), false);
    let d1 = rspace.store.get_data(&produce_key_1[0]);
    assert_eq!(d1, vec![Datum::create(&produce_key_1[0], "datum1".to_string(), false)]);

    let c1 = rspace.store.get_continuations(&produce_key_1.clone());
    assert!(c1.is_empty());
    assert!(r1.unwrap().is_none());

    let r2 = rspace.consume(
        consume_key.clone(),
        consume_pattern,
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let d2 = rspace.store.get_data(&produce_key_1[0]);
    assert_eq!(d2, vec![Datum::create(&produce_key_1[0], "datum1".to_string(), false)]);

    let c2 = rspace.store.get_continuations(&produce_key_1.clone());
    let d3 = rspace.store.get_data(&produce_key_2[0]);
    let c3 = rspace.store.get_continuations(&consume_key.clone());
    assert!(c2.is_empty());
    assert!(d3.is_empty());
    assert_ne!(c3.len(), 0);
    assert!(r2.unwrap().is_none());

    let r3 = rspace.produce(produce_key_2[0].clone(), "datum2".to_string(), false);
    let c4 = rspace.store.get_continuations(&consume_key);
    let d4 = rspace.store.get_data(&produce_key_1[0]);
    let d5 = rspace.store.get_data(&produce_key_2[0]);
    assert!(c4.is_empty());
    assert!(d4.is_empty());
    assert!(d5.is_empty());
    assert!(r3.clone().unwrap().is_some());

    let cont_results = run_produce_k(r3.unwrap());
    assert!(check_same_elements(cont_results, vec![vec![
        "datum1".to_string(),
        "datum2".to_string()
    ]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn producing_on_three_channels_then_consuming_once_should_return_cont_and_all_data() {
    let mut rspace = create_rspace().await;
    let produce_key_1 = vec!["ch1".to_string()];
    let produce_key_2 = vec!["ch2".to_string()];
    let produce_key_3 = vec!["ch3".to_string()];
    let consume_key = vec!["ch1".to_string(), "ch2".to_string(), "ch3".to_string()];
    let patterns = vec![Pattern::Wildcard, Pattern::Wildcard, Pattern::Wildcard];

    let r1 = rspace.produce(produce_key_1[0].clone(), "datum1".to_string(), false);
    let d1 = rspace.store.get_data(&produce_key_1[0]);
    assert_eq!(d1, vec![Datum::create(&produce_key_1[0], "datum1".to_string(), false)]);

    let c1 = rspace.store.get_continuations(&produce_key_1);
    assert!(c1.is_empty());
    assert!(r1.unwrap().is_none());

    let r2 = rspace.produce(produce_key_2[0].clone(), "datum2".to_string(), false);
    let d2 = rspace.store.get_data(&produce_key_2[0]);
    assert_eq!(d2, vec![Datum::create(&produce_key_2[0], "datum2".to_string(), false)]);

    let c2 = rspace.store.get_continuations(&produce_key_2);
    assert!(c2.is_empty());
    assert!(r2.unwrap().is_none());

    let r3 = rspace.produce(produce_key_3[0].clone(), "datum3".to_string(), false);
    let d3 = rspace.store.get_data(&produce_key_3[0]);
    assert_eq!(d3, vec![Datum::create(&produce_key_3[0], "datum3".to_string(), false)]);

    let c3 = rspace.store.get_continuations(&produce_key_3);
    assert!(c3.is_empty());
    assert!(r3.unwrap().is_none());

    let r4 = rspace.consume(
        consume_key.clone(),
        patterns,
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let d4: Vec<_> = consume_key
        .iter()
        .map(|k| rspace.store.get_data(k))
        .collect();
    // let d4: Vec<Vec<Datum<String>>> = futures::future::join_all(futures);
    for seq in &d4 {
        assert!(seq.is_empty(), "d should be empty");
    }

    let c4 = rspace.store.get_continuations(&consume_key);
    assert!(c4.is_empty());
    assert!(r4.clone().unwrap().is_some());

    let cont_results = run_k(r4.unwrap());
    assert!(check_same_elements(cont_results, vec![vec![
        "datum1".to_string(),
        "datum2".to_string(),
        "datum3".to_string()
    ]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn producing_then_consuming_three_times_on_same_channel_should_return_three_pairs_of_conts_and_data()
 {
    let mut rspace = create_rspace().await;
    let captor = StringsCaptor::new();
    let key = vec!["ch1".to_string()];

    let r1 = rspace.produce(key[0].clone(), "datum1".to_string(), false);
    let r2 = rspace.produce(key[0].clone(), "datum2".to_string(), false);
    let r3 = rspace.produce(key[0].clone(), "datum3".to_string(), false);
    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());
    assert!(r3.unwrap().is_none());

    let r4 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        captor.clone(),
        false,
        BTreeSet::default(),
    );
    let r5 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        captor.clone(),
        false,
        BTreeSet::default(),
    );
    let r6 =
        rspace.consume(key.clone(), vec![Pattern::Wildcard], captor, false, BTreeSet::default());
    let c1 = rspace.store.get_continuations(&key);
    assert!(c1.is_empty());

    let continuations =
        vec![r4.clone().unwrap().clone(), r5.clone().unwrap().clone(), r6.clone().unwrap().clone()];
    assert!(continuations.iter().all(Option::is_some));
    let cont_results_r4 = run_k(r4.unwrap());
    let cont_results_r5 = run_k(r5.unwrap());
    let cont_results_r6 = run_k(r6.unwrap());
    let cont_results = [cont_results_r4, cont_results_r5, cont_results_r6].concat();
    assert!(check_same_elements(cont_results, vec![
        vec!["datum3".to_string()],
        vec!["datum2".to_string()],
        vec!["datum1".to_string()]
    ]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_then_producing_three_times_on_same_channel_should_return_conts_each_paired_with_distinct_data()
 {
    let mut rspace = create_rspace().await;
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::with_id(1),
        false,
        BTreeSet::default(),
    );
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::with_id(2),
        false,
        BTreeSet::default(),
    );
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::with_id(3),
        false,
        BTreeSet::default(),
    );

    let r1 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r2 = rspace.produce("ch1".to_string(), "datum2".to_string(), false);
    let r3 = rspace.produce("ch1".to_string(), "datum3".to_string(), false);
    assert!(r1.clone().unwrap().is_some());
    assert!(r2.clone().unwrap().is_some());
    assert!(r3.clone().unwrap().is_some());

    let possible_cont_results =
        vec![vec!["datum1".to_string()], vec!["datum2".to_string()], vec!["datum3".to_string()]];
    let cont_results_r1 = run_produce_k(r1.unwrap());
    let cont_results_r2 = run_produce_k(r2.unwrap());
    let cont_results_r3 = run_produce_k(r3.unwrap());
    assert!(
        possible_cont_results
            .iter()
            .any(|v| cont_results_r1.contains(v))
    );
    assert!(
        possible_cont_results
            .iter()
            .any(|v| cont_results_r2.contains(v))
    );
    assert!(
        possible_cont_results
            .iter()
            .any(|v| cont_results_r3.contains(v))
    );

    assert!(!check_same_elements(cont_results_r1.clone(), cont_results_r2.clone()));
    assert!(!check_same_elements(cont_results_r1, cont_results_r3.clone()));
    assert!(!check_same_elements(cont_results_r2, cont_results_r3));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_then_producing_three_times_on_same_channel_with_non_trivial_matches_should_return_three_conts_each_paired_with_matching_data()
 {
    let mut rspace = create_rspace().await;
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::StringMatch("datum1".to_string())],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::StringMatch("datum2".to_string())],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::StringMatch("datum3".to_string())],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );

    let r1 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r2 = rspace.produce("ch1".to_string(), "datum2".to_string(), false);
    let r3 = rspace.produce("ch1".to_string(), "datum3".to_string(), false);
    assert!(r1.clone().unwrap().is_some());
    assert!(r2.clone().unwrap().is_some());
    assert!(r3.clone().unwrap().is_some());

    assert_eq!(run_produce_k(r1.unwrap()), vec![vec!["datum1"]]);
    assert_eq!(run_produce_k(r2.unwrap()), vec![vec!["datum2"]]);
    assert_eq!(run_produce_k(r3.unwrap()), vec![vec!["datum3"]]);

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_on_two_channels_then_producing_on_each_should_return_cont_with_both_data() {
    let mut rspace = create_rspace().await;
    let r1 = rspace.consume(
        vec!["ch1".to_string(), "ch2".to_string()],
        vec![Pattern::Wildcard, Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );

    let r2 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r3 = rspace.produce("ch2".to_string(), "datum2".to_string(), false);

    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());
    assert!(r3.clone().unwrap().is_some());
    assert!(check_same_elements(run_produce_k(r3.unwrap()), vec![vec![
        "datum1".to_string(),
        "datum2".to_string()
    ]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn joined_consume_with_same_channel_given_twice_followed_by_produce_should_not_error() {
    let mut rspace = create_rspace().await;
    let channels = vec!["ch1".to_string(), "ch1".to_string()];

    let r1 = rspace.consume(
        channels,
        vec![
            Pattern::StringMatch("datum1".to_string()),
            Pattern::StringMatch("datum1".to_string()),
        ],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let r2 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r3 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);

    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());
    assert!(r3.clone().unwrap().is_some());
    assert!(check_same_elements(run_produce_k(r3.unwrap()), vec![vec![
        "datum1".to_string(),
        "datum1".to_string()
    ]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_then_producing_twice_on_same_channel_with_different_patterns_should_return_cont_with_expected_data()
 {
    let mut rspace = create_rspace().await;
    let channels = vec!["ch1".to_string(), "ch2".to_string()];

    let r1 = rspace.consume(
        channels.clone(),
        vec![
            Pattern::StringMatch("datum1".to_string()),
            Pattern::StringMatch("datum2".to_string()),
        ],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let r2 = rspace.consume(
        channels,
        vec![
            Pattern::StringMatch("datum3".to_string()),
            Pattern::StringMatch("datum4".to_string()),
        ],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );

    let r3 = rspace.produce("ch1".to_string(), "datum3".to_string(), false);
    let r4 = rspace.produce("ch2".to_string(), "datum4".to_string(), false);
    let r5 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r6 = rspace.produce("ch2".to_string(), "datum2".to_string(), false);

    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());
    assert!(r3.unwrap().is_none());
    assert!(r4.clone().unwrap().is_some());
    assert!(r5.unwrap().is_none());
    assert!(r6.clone().unwrap().is_some());

    assert!(check_same_elements(run_produce_k(r4.unwrap()), vec![vec![
        "datum3".to_string(),
        "datum4".to_string()
    ]]));
    assert!(check_same_elements(run_produce_k(r6.unwrap()), vec![vec![
        "datum1".to_string(),
        "datum2".to_string()
    ]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_and_producing_with_non_trivial_matches_should_work() {
    let mut rspace = create_rspace().await;

    let r1 = rspace.consume(
        vec!["ch1".to_string(), "ch2".to_string()],
        vec![Pattern::Wildcard, Pattern::StringMatch("datum1".to_string())],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let r2 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);

    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());

    let d1 = rspace.store.get_data(&"ch2".to_string());
    assert!(d1.is_empty());
    let d2 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d2, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), false)]);

    let c1 = rspace
        .store
        .get_continuations(&vec!["ch1".to_string(), "ch2".to_string()]);
    assert!(!c1.is_empty());
    let j1 = rspace.store.get_joins(&"ch1".to_string());
    assert_eq!(j1, vec![vec!["ch1".to_string(), "ch2".to_string()]]);
    let j2 = rspace.store.get_joins(&"ch2".to_string());
    assert_eq!(j2, vec![vec!["ch1".to_string(), "ch2".to_string()]]);

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(!insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_and_producing_twice_with_non_trivial_matches_should_work() {
    let mut rspace = create_rspace().await;

    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::StringMatch("datum1".to_string())],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let _ = rspace.consume(
        vec!["ch2".to_string()],
        vec![Pattern::StringMatch("datum2".to_string())],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );

    let r3 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r4 = rspace.produce("ch2".to_string(), "datum2".to_string(), false);

    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert!(d1.is_empty());
    let d2 = rspace.store.get_data(&"ch2".to_string());
    assert!(d2.is_empty());

    assert!(check_same_elements(run_produce_k(r3.unwrap()), vec![vec!["datum1".to_string()]]));
    assert!(check_same_elements(run_produce_k(r4.unwrap()), vec![vec!["datum2".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn consuming_on_two_channels_then_consuming_on_one_then_producing_on_both_separately_should_return_cont_paired_with_one_data()
 {
    let mut rspace = create_rspace().await;

    let _ = rspace.consume(
        vec!["ch1".to_string(), "ch2".to_string()],
        vec![Pattern::Wildcard, Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let _ = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );

    let r3 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r4 = rspace.produce("ch2".to_string(), "datum2".to_string(), false);

    let c1 = rspace
        .store
        .get_continuations(&vec!["ch1".to_string(), "ch2".to_string()]);
    assert!(!c1.is_empty());
    let c2 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!((c2.is_empty()));
    let c3 = rspace.store.get_continuations(&vec!["ch2".to_string()]);
    assert!(c3.is_empty());

    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert!(d1.is_empty());
    let d2 = rspace.store.get_data(&"ch2".to_string());
    assert_eq!(d2, vec![Datum::create(&"ch2".to_string(), "datum2".to_string(), false)]);

    assert!(r3.clone().unwrap().is_some());
    assert!(r4.unwrap().is_none());
    assert!(check_same_elements(run_produce_k(r3.unwrap()), vec![vec!["datum1".to_string()]]));

    let j1 = rspace.store.get_joins(&"ch1".to_string());
    assert_eq!(j1, vec![vec!["ch1".to_string(), "ch2".to_string()]]);
    let j2 = rspace.store.get_joins(&"ch2".to_string());
    assert_eq!(j2, vec![vec!["ch1".to_string(), "ch2".to_string()]]);

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(!insert_actions.is_empty());
}

/* Persist tests */

#[tokio::test]
async fn producing_then_persistent_consume_on_same_channel_should_return_cont_and_data() {
    let mut rspace = create_rspace().await;
    let key = vec!["ch1".to_string()];

    let r1 = rspace.produce(key[0].clone(), "datum".to_string(), false);
    let d1 = rspace.store.get_data(&key[0]);
    assert_eq!(d1, vec![Datum::create(&key[0], "datum".to_string(), false)]);
    let c1 = rspace.store.get_continuations(&key.clone());
    assert!(c1.is_empty());
    assert!(r1.unwrap().is_none());

    // Data exists so the write will not "stick"
    let r2 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_k(r2.unwrap()), vec![vec!["datum".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());

    let r3 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    let d2 = rspace.store.get_data(&key[0]);
    assert!(d2.is_empty());
    let c2 = rspace.store.get_continuations(&key);
    assert!(!c2.is_empty());
    assert!(r3.unwrap().is_none());
}

#[tokio::test]
async fn producing_then_persistent_consume_then_producing_again_on_same_channel_should_return_cont_for_first_and_second_produce()
 {
    let mut rspace = create_rspace().await;
    let key = vec!["ch1".to_string()];

    let r1 = rspace.produce(key[0].clone(), "datum1".to_string(), false);
    let d1 = rspace.store.get_data(&key[0]);
    assert_eq!(d1, vec![Datum::create(&key[0], "datum1".to_string(), false)]);
    let c1 = rspace.store.get_continuations(&key.clone());
    assert!(c1.is_empty());
    assert!(r1.unwrap().is_none());

    let r2 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_k(r2.unwrap()), vec![vec!["datum1".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());

    let r3 = rspace.consume(
        key.clone(),
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    assert!(r3.unwrap().is_none());

    let d2 = rspace.store.get_data(&key[0]);
    assert!(d2.is_empty());
    let c2 = rspace.store.get_continuations(&key.clone());
    assert!(!c2.is_empty());

    let r4 = rspace.produce(key[0].clone(), "datum2".to_string(), false);
    assert!(r4.clone().unwrap().is_some());
    let d3 = rspace.store.get_data(&key[0]);
    assert!(d3.is_empty());
    let c3 = rspace.store.get_continuations(&key);
    assert!(!c3.is_empty());
    assert!(check_same_elements(run_produce_k(r4.clone().unwrap()), vec![vec![
        "datum2".to_string()
    ]]))
}

#[tokio::test]
async fn doing_persistent_consume_and_producing_multiple_times_should_work() {
    let mut rspace = create_rspace().await;

    let r1 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert!(d1.is_empty());
    let c1 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(!c1.is_empty());
    assert!(r1.unwrap().is_none());

    let r2 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let d2 = rspace.store.get_data(&"ch1".to_string());
    assert!(d2.is_empty());
    let c2 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(!c2.is_empty());
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_produce_k(r2.unwrap().clone()), vec![vec![
        "datum1".to_string()
    ]]));

    let r3 = rspace.produce("ch1".to_string(), "datum2".to_string(), false);
    let d3 = rspace.store.get_data(&"ch1".to_string());
    assert!(d3.is_empty());
    let c3 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(!c3.is_empty());
    assert!(r3.clone().unwrap().is_some());

    let r3_results = run_produce_k(r3.clone().unwrap());

    // The below is commented out and replaced because the rust side does not allow
    // for modification of continuation in the history_store and have it reflect the
    // the hot_store. This would require the continuation to be wrapped in a
    // Arc<Mutex<>> which is not needed

    // assert!(check_same_elements(
    //     r3_results,
    //     vec![vec!["datum1".to_string()], vec!["datum2".to_string()]]
    // ));
    assert!(check_same_elements(r3_results, vec![vec!["datum2".to_string()], vec![
        "datum2".to_string()
    ]]));
}

#[tokio::test]
async fn consuming_and_doing_persistent_produce_should_work() {
    let mut rspace = create_rspace().await;

    let r1 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r1.unwrap().is_none());

    let r2 = rspace.produce("ch1".to_string(), "datum1".to_string(), true);
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_produce_k(r2.unwrap()), vec![vec!["datum1".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());

    let r3 = rspace.produce("ch1".to_string(), "datum1".to_string(), true);
    assert!(r3.unwrap().is_none());
    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d1, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), true)]);
    let c1 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c1.is_empty());
}

#[tokio::test]
async fn consuming_then_persistent_produce_then_consuming_should_work() {
    let mut rspace = create_rspace().await;

    let r1 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r1.unwrap().is_none());

    let r2 = rspace.produce("ch1".to_string(), "datum1".to_string(), true);
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_produce_k(r2.unwrap()), vec![vec!["datum1".to_string()]]));

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());

    let r3 = rspace.produce("ch1".to_string(), "datum1".to_string(), true);
    assert!(r3.unwrap().is_none());
    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d1, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), true)]);
    let c1 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c1.is_empty());

    let r4 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r4.clone().unwrap().is_some());
    let d2 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d2, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), true)]);
    let c2 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c2.is_empty());
    assert!(check_same_elements(run_k(r4.unwrap()), vec![vec!["datum1".to_string()]]))
}

#[tokio::test]
async fn doing_persistent_produce_and_consuming_twice_should_work() {
    let mut rspace = create_rspace().await;

    let r1 = rspace.produce("ch1".to_string(), "datum1".to_string(), true);
    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d1, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), true)]);
    let c1 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c1.is_empty());
    assert!(r1.unwrap().is_none());

    let r2 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let d2 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d2, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), true)]);
    let c2 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c2.is_empty());
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_k(r2.unwrap()), vec![vec!["datum1".to_string()]]));

    let r3 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    let d3 = rspace.store.get_data(&"ch1".to_string());
    assert_eq!(d3, vec![Datum::create(&"ch1".to_string(), "datum1".to_string(), true)]);
    let c3 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c3.is_empty());
    assert!(r3.clone().unwrap().is_some());
    assert!(check_same_elements(run_k(r3.unwrap()), vec![vec!["datum1".to_string()]]));
}

#[tokio::test]
async fn producing_three_times_then_doing_persistent_consume_should_work() {
    let mut rspace = create_rspace().await;
    let expected_data = vec![
        Datum::create(&"ch1".to_string(), "datum1".to_string(), false),
        Datum::create(&"ch1".to_string(), "datum2".to_string(), false),
        Datum::create(&"ch1".to_string(), "datum3".to_string(), false),
    ];
    let expected_conts =
        vec![vec!["datum1".to_string()], vec!["datum2".to_string()], vec!["datum3".to_string()]];

    let r1 = rspace.produce("ch1".to_string(), "datum1".to_string(), false);
    let r2 = rspace.produce("ch1".to_string(), "datum2".to_string(), false);
    let r3 = rspace.produce("ch1".to_string(), "datum3".to_string(), false);
    assert!(r1.unwrap().is_none());
    assert!(r2.unwrap().is_none());
    assert!(r3.unwrap().is_none());

    let r4 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    let d1 = rspace.store.get_data(&"ch1".to_string());
    assert!(expected_data.iter().any(|datum| d1.contains(datum)));
    let c1 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c1.is_empty());
    assert!(r4.clone().unwrap().is_some());
    let cont_results_r4 = run_k(r4.unwrap());
    assert!(
        expected_conts
            .iter()
            .any(|cont| cont_results_r4.contains(cont))
    );

    let r5 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    let d2 = rspace.store.get_data(&"ch1".to_string());
    assert!(expected_data.iter().any(|datum| d2.contains(datum)));
    let c2 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(c2.is_empty());
    assert!(r5.clone().unwrap().is_some());
    let cont_results_r5 = run_k(r5.unwrap());
    assert!(
        expected_conts
            .iter()
            .any(|cont| cont_results_r5.contains(cont))
    );

    let r6 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    assert!(r6.clone().unwrap().is_some());

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());

    let cont_results_r6 = run_k(r6.unwrap());
    assert!(
        expected_conts
            .iter()
            .any(|cont| cont_results_r6.contains(cont))
    );

    let r7 = rspace.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        true,
        BTreeSet::default(),
    );
    let d3 = rspace.store.get_data(&"ch1".to_string());
    assert!(d3.is_empty());
    let c3 = rspace.store.get_continuations(&vec!["ch1".to_string()]);
    assert!(!c3.is_empty());
    assert!(r7.unwrap().is_none());
}

#[tokio::test]
async fn persistent_produce_should_be_available_for_multiple_matches() {
    let mut rspace = create_rspace().await;
    let channel = "chan".to_string();

    let r1 = rspace.produce(channel.clone(), "datum".to_string(), true);
    assert!(r1.unwrap().is_none());

    let r2 = rspace.consume(
        vec![channel.clone(), channel.clone()],
        vec![Pattern::Wildcard, Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r2.clone().unwrap().is_some());
    assert!(check_same_elements(run_k(r2.unwrap()), vec![vec![
        "datum".to_string(),
        "datum".to_string()
    ]]));
}

#[tokio::test]
async fn clear_should_reset_to_the_same_hash_on_multiple_runs() {
    let mut rspace = create_rspace().await;
    let key = vec!["ch1".to_string()];
    let patterns = vec![Pattern::Wildcard];

    let empty_checkpoint = rspace.create_checkpoint().unwrap();

    // put some data so the checkpoint is != empty
    let _ = rspace.consume(key, patterns, StringsCaptor::new(), false, BTreeSet::default());

    let checkpoint0 = rspace.create_checkpoint().unwrap();
    assert!(!checkpoint0.log.is_empty());
    let _ = rspace.create_checkpoint().unwrap();

    // force clearing of trie store state
    let _ = rspace.clear().unwrap();

    // the checkpointing mechanism should not interfere with the empty root
    let checkpoint2 = rspace.create_checkpoint().unwrap();
    assert!(checkpoint2.log.is_empty());
    assert_eq!(checkpoint2.root, empty_checkpoint.root);
}

#[tokio::test]
async fn create_checkpoint_on_an_empty_store_should_return_the_expected_hash() {
    let mut rspace = create_rspace().await;
    let empty_checkpoint = rspace.create_checkpoint().unwrap();
    assert_eq!(empty_checkpoint.root, RadixHistory::empty_root_node_hash());
}

#[tokio::test]
async fn create_checkpoint_should_clear_the_store_contents() {
    let mut rspace = create_rspace().await;
    let key = vec!["ch1".to_string()];
    let patterns = vec![Pattern::Wildcard];

    let _ = rspace.consume(key, patterns, StringsCaptor::new(), false, BTreeSet::default());

    let _ = rspace.create_checkpoint().unwrap();
    let checkpoint0_changes = rspace.store.changes();
    assert_eq!(checkpoint0_changes.len(), 0);
}

#[tokio::test]
async fn reset_should_change_the_state_of_the_store_and_reset_the_trie_updates_log() {
    let mut rspace = create_rspace().await;
    let key = vec!["ch1".to_string()];
    let patterns = vec![Pattern::Wildcard];

    let checkpint0 = rspace.create_checkpoint().unwrap();
    let r = rspace.consume(key, patterns, StringsCaptor::new(), false, BTreeSet::default());
    assert!(r.unwrap().is_none());

    let checkpoint0_changes: Vec<InsertContinuations<String, Pattern, StringsCaptor>> = rspace
        .store
        .changes()
        .into_iter()
        .filter_map(|action| {
            if let HotStoreAction::Insert(InsertAction::InsertContinuations(val)) = action {
                Some(val)
            } else {
                None
            }
        })
        .collect();
    assert!(!checkpoint0_changes.is_empty());
    assert_eq!(checkpoint0_changes.len(), 1);

    let _ = rspace.reset(&checkpint0.root).unwrap();
    let reset_changes = rspace.store.changes();
    assert!(reset_changes.is_empty());
    assert_eq!(reset_changes.len(), 0);

    let checkpoint1 = rspace.create_checkpoint().unwrap();
    assert!(checkpoint1.log.is_empty());
}

#[tokio::test]
async fn consume_and_produce_a_match_and_then_checkpoint_should_result_in_an_empty_triestore() {
    let mut rspace = create_rspace().await;
    let channels = vec!["ch1".to_string()];

    let checkpoint_init = rspace.create_checkpoint().unwrap();
    assert_eq!(checkpoint_init.root, RadixHistory::empty_root_node_hash());

    let r1 = rspace.consume(
        channels,
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r1.unwrap().is_none());

    let r2 = rspace.produce("ch1".to_string(), "datum".to_string(), false);
    assert!(r2.unwrap().is_some());

    let checkpoint = rspace.create_checkpoint().unwrap();
    assert_eq!(checkpoint.root, RadixHistory::empty_root_node_hash());

    let _ = rspace.create_checkpoint();
    let checkpoint0_changes = rspace.store.changes();
    assert_eq!(checkpoint0_changes.len(), 0);
}

proptest! {
  #![proptest_config(ProptestConfig {
    cases: 50,
    .. ProptestConfig::default()
})]

  #[test]
  fn produce_a_bunch_and_then_create_checkpoint_then_consume_on_same_channels_should_result_in_checkpoint_pointing_at_empty_state(data in proptest::collection::vec(".*", 1..100)) {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
      let mut rspace = create_rspace().await;

      for channel in data.clone() {
        let _ = rspace.produce(channel, "data".to_string(),false);
      }

      let checkpoint1 = rspace.create_checkpoint().unwrap();

      for channel in data.iter() {
        let result = rspace.consume(vec![channel.to_string()], vec![Pattern::Wildcard], StringsCaptor::new(), false, BTreeSet::default());
        assert!(result.unwrap().is_some());
      }

      let checkpoint2 = rspace.create_checkpoint().unwrap();

      for channel in data.iter() {
        let result = rspace.consume(vec![channel.to_string()], vec![Pattern::Wildcard], StringsCaptor::new(), false, BTreeSet::default());
        assert!(result.unwrap().is_none());
      }

      assert_eq!(checkpoint2.root, RadixHistory::empty_root_node_hash());
      let _ = rspace.reset(&checkpoint1.root).unwrap();

      for channel in data.iter() {
        let result = rspace.consume(vec![channel.to_string()], vec![Pattern::Wildcard], StringsCaptor::new(), false, BTreeSet::default());
        assert!(result.unwrap().is_some());
      }

      let checkpoint3 = rspace.create_checkpoint().unwrap();
      assert_eq!(checkpoint3.root, RadixHistory::empty_root_node_hash());

    });
  }
}

#[tokio::test]
async fn an_install_should_not_allow_installing_after_a_produce_operation() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let datum = "datum1".to_string();
    let key = vec![channel.clone()];
    let patterns = vec![Pattern::Wildcard];

    let _ = rspace.produce(channel, datum, false);
    let install_attempt = rspace.install(key, patterns, StringsCaptor::new());
    assert!(install_attempt.is_err())
}

#[tokio::test]
#[should_panic(expected = "RUST ERROR: channels.length must equal patterns.length")]
async fn consuming_with_different_pattern_and_channel_lengths_should_error() {
    let mut rspace = create_rspace().await;
    let r1 = rspace.consume(
        vec!["ch1".to_string(), "ch2".to_string()],
        vec![Pattern::Wildcard],
        StringsCaptor::new(),
        false,
        BTreeSet::default(),
    );
    assert!(r1.unwrap().is_none());

    let insert_actions: Vec<InsertAction<_, _, _, _>> =
        filter_enum_variants(rspace.store.changes(), |e| {
            if let HotStoreAction::Insert(i) = e {
                Some(i)
            } else {
                None
            }
        });
    assert!(insert_actions.is_empty());
}

#[tokio::test]
async fn create_soft_checkpoint_should_capture_the_current_state_of_the_store() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let channels = vec![channel.clone()];
    let patterns = vec![Pattern::Wildcard];
    let continuation = StringsCaptor::new();

    let expected_continuation = vec![WaitingContinuation {
        patterns: patterns.clone(),
        continuation: continuation.clone(),
        persist: false,
        peeks: BTreeSet::default(),
        source: Consume::create(&channels, &patterns, &continuation, false),
    }];

    // do an operation
    let _ = rspace.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::default(),
    );

    // create a soft checkpoint
    let s = rspace.create_soft_checkpoint();

    // assert that the snapshot contains the continuation
    let snapshot_continuations_values: Vec<Vec<WaitingContinuation<Pattern, StringsCaptor>>> = s
        .cache_snapshot
        .continuations
        .iter()
        .map(|entry| entry.value().clone())
        .collect();
    assert_eq!(snapshot_continuations_values, vec![expected_continuation.clone()]);

    // consume again
    let _ = rspace.consume(channels, patterns, continuation, false, BTreeSet::default());

    // assert that the snapshot contains only the first continuation
    let snapshot_continuations_values: Vec<Vec<WaitingContinuation<Pattern, StringsCaptor>>> = s
        .cache_snapshot
        .continuations
        .iter()
        .map(|entry| entry.value().clone())
        .collect();
    assert_eq!(snapshot_continuations_values, vec![expected_continuation]);
}

#[tokio::test]
async fn create_soft_checkpoint_should_create_checkpoints_which_have_separate_state() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let channels = vec![channel.clone()];
    let datum = "datum1".to_string();
    let patterns = vec![Pattern::Wildcard];
    let continuation = StringsCaptor::new();

    let expected_continuation = vec![WaitingContinuation {
        patterns: patterns.clone(),
        continuation: continuation.clone(),
        persist: false,
        peeks: BTreeSet::default(),
        source: Consume::create(&channels, &patterns, &continuation, false),
    }];

    // do an operation
    let _ = rspace.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::default(),
    );

    // create a soft checkpoint
    let s1 = rspace.create_soft_checkpoint();

    // assert that the snapshot contains the continuation
    let snapshot_continuations_values: Vec<Vec<WaitingContinuation<Pattern, StringsCaptor>>> = s1
        .cache_snapshot
        .continuations
        .iter()
        .map(|entry| entry.value().clone())
        .collect();
    assert_eq!(snapshot_continuations_values, vec![expected_continuation.clone()]);

    // produce thus removing the continuation
    let _ = rspace.produce(channel, datum, false);
    let s2 = rspace.create_soft_checkpoint();

    // assert that the first snapshot still contains the first continuation
    let snapshot_continuations_values: Vec<Vec<WaitingContinuation<Pattern, StringsCaptor>>> = s1
        .cache_snapshot
        .continuations
        .iter()
        .map(|entry| entry.value().clone())
        .collect();
    assert_eq!(snapshot_continuations_values, vec![expected_continuation]);

    assert!(
        s2.cache_snapshot
            .continuations
            .get(&channels)
            .unwrap()
            .value()
            .is_empty()
    )
}

#[tokio::test]
async fn create_soft_checkpoint_should_clear_the_event_log() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let channels = vec![channel.clone()];
    let patterns = vec![Pattern::Wildcard];
    let continuation = StringsCaptor::new();

    // do an operation
    let _ = rspace.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::default(),
    );

    // create a soft checkpoint
    let s1 = rspace.create_soft_checkpoint();
    assert!(!s1.log.is_empty());

    let s2 = rspace.create_soft_checkpoint();
    assert!(s2.log.is_empty());
}

#[tokio::test]
async fn revert_to_soft_checkpoint_should_revert_the_state_of_the_store_to_the_given_checkpoint() {
    let mut rspace = create_rspace().await;
    let channel = "ch1".to_string();
    let channels = vec![channel.clone()];
    let patterns = vec![Pattern::Wildcard];
    let continuation = StringsCaptor::new();

    // create an initial soft checkpoint
    let s1 = rspace.create_soft_checkpoint();
    // do an operation
    let _ = rspace.consume(channels, patterns, continuation, false, BTreeSet::new());

    let changes: Vec<InsertContinuations<String, Pattern, StringsCaptor>> = rspace
        .store
        .changes()
        .into_iter()
        .filter_map(|action| {
            if let HotStoreAction::Insert(InsertAction::InsertContinuations(val)) = action {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    // the operation should be on the list of changes
    assert!(!changes.is_empty());
    let _ = rspace.revert_to_soft_checkpoint(s1).unwrap();

    let changes: Vec<InsertContinuations<String, Pattern, StringsCaptor>> = rspace
        .store
        .changes()
        .into_iter()
        .filter_map(|action| {
            if let HotStoreAction::Insert(InsertAction::InsertContinuations(val)) = action {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    // after reverting to the initial soft checkpoint the operation is no longer
    // present in the hot store
    assert!(changes.is_empty());
}

#[tokio::test]
async fn revert_to_soft_checkpoint_should_inject_the_event_log() {
    let mut rspace = create_rspace().await;

    let channel = "ch1".to_string();
    let channels = vec![channel.clone()];
    let patterns = vec![Pattern::Wildcard];
    let continuation = StringsCaptor::new();

    let _ = rspace.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::new(),
    );
    let s1 = rspace.create_soft_checkpoint();
    let _ = rspace.consume(channels, patterns, continuation, true, BTreeSet::new());
    let s2 = rspace.create_soft_checkpoint();

    assert_ne!(s2.log, s1.log);

    let _ = rspace.revert_to_soft_checkpoint(s1.clone());
    let s3 = rspace.create_soft_checkpoint();
    assert_eq!(s3.log, s1.log);
}

// =============================================================================
// Property-based tests for peek (<<-) semantics
// =============================================================================
//
// These tests vary *structural* parameters -- number of channels, which
// indices are peeked, persistence flags, number of sequential peek reads,
// number of waiting continuations, and operation ordering -- so that each
// proptest iteration exercises a genuinely different scenario rather than
// replaying the same fixed topology with different string values.

/// Generate a list of N *distinct* channel names.
fn distinct_channels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("ch{}", i)).collect()
}

/// Strategy: pick a channel count in 1..=max_channels, then for each channel
/// decide independently whether it is peeked.  Returns (channels, peek_set).
fn channels_and_peeks_strategy(
    max_channels: usize,
) -> impl Strategy<Value = (Vec<String>, BTreeSet<i32>)> {
    (1..=max_channels)
        .prop_flat_map(|n| {
            // For each of the n channels, independently decide peek (true) or
            // not (false).
            proptest::collection::vec(proptest::bool::ANY, n).prop_map(move |peek_flags| {
                let channels = distinct_channels(n);
                let peeks: BTreeSet<i32> = peek_flags
                    .iter()
                    .enumerate()
                    .filter(|(_, &is_peeked)| is_peeked)
                    .map(|(i, _)| i as i32)
                    .collect();
                (channels, peeks)
            })
        })
}

/// Strategy: generate (channels, peeks) where the peek set is guaranteed
/// non-empty (at least one channel is peeked).
fn channels_with_at_least_one_peek(
    max_channels: usize,
) -> impl Strategy<Value = (Vec<String>, BTreeSet<i32>)> {
    channels_and_peeks_strategy(max_channels)
        .prop_filter("need at least one peek", |(_, peeks)| !peeks.is_empty())
}

/// Strategy: generate (channels, peeks) where at least one channel is peeked
/// AND at least one channel is NOT peeked (mixed mode).
fn channels_with_mixed_peeks() -> impl Strategy<Value = (Vec<String>, BTreeSet<i32>)> {
    channels_and_peeks_strategy(5).prop_filter(
        "need at least one peeked and one non-peeked channel",
        |(channels, peeks)| {
            !peeks.is_empty() && peeks.len() < channels.len()
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 50,
        .. ProptestConfig::default()
    })]

    // =========================================================================
    // 1. Peek preserves data (produce-then-consume path)
    // =========================================================================
    //
    // For a randomly-chosen number of channels (1..=4) with a randomly-chosen
    // non-empty peek set, produce data on every channel, then consume with
    // peek.  All peeked channels must retain their data afterward.

    #[test]
    fn peek_preserves_data_produce_then_consume(
        (channels, peeks) in channels_with_at_least_one_peek(4),
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let n = channels.len();
            let data: Vec<String> = (0..n).map(|i| format!("datum{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; n];

            // Produce data on every channel.
            for i in 0..n {
                let r = rspace.produce(channels[i].clone(), data[i].clone(), false);
                prop_assert!(r.unwrap().is_none());
            }

            // Consume with peek.
            let r = rspace.consume(
                channels.clone(),
                patterns,
                StringsCaptor::new(),
                false,
                peeks.clone(),
            );

            // Should match and fire the continuation.
            prop_assert!(r.clone().unwrap().is_some());
            let cont_results = run_k(r.unwrap());
            prop_assert_eq!(cont_results, vec![data.clone()]);

            // Peeked channels must retain data; non-peeked channels must not.
            for i in 0..n {
                let d = rspace.store.get_data(&channels[i]);
                if peeks.contains(&(i as i32)) {
                    prop_assert_eq!(d.len(), 1,
                        "peeked channel {} should retain data", channels[i]);
                    prop_assert_eq!(&d[0].a, &data[i]);
                } else {
                    prop_assert_eq!(d.len(), 0,
                        "non-peeked channel {} should have data removed", channels[i]);
                }
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // 2. Peek preserves data (consume-then-produce path)
    // =========================================================================
    //
    // Register a peek consume on N channels (none have data yet), then produce
    // data on each channel one at a time.  The last produce completes the
    // match.  Peeked channel data must persist afterward.

    #[test]
    fn peek_preserves_data_consume_then_produce(
        (channels, peeks) in channels_with_at_least_one_peek(4),
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let n = channels.len();
            let data: Vec<String> = (0..n).map(|i| format!("datum{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; n];

            // Consume with peek (no data yet -- stores waiting continuation).
            let r_consume = rspace.consume(
                channels.clone(),
                patterns,
                StringsCaptor::new(),
                false,
                peeks.clone(),
            );
            prop_assert!(r_consume.unwrap().is_none());

            // Produce data on all channels except the last (no match yet).
            for i in 0..n.saturating_sub(1) {
                let r = rspace.produce(channels[i].clone(), data[i].clone(), false);
                prop_assert!(r.unwrap().is_none(),
                    "produce on channel {} should not complete match yet", channels[i]);
            }

            // Produce on the last channel -- this should complete the match.
            let last = n - 1;
            let r_last = rspace.produce(channels[last].clone(), data[last].clone(), false);
            prop_assert!(r_last.clone().unwrap().is_some(),
                "final produce should complete the multi-channel match");

            let cont_results = run_produce_k(r_last.unwrap());
            prop_assert_eq!(cont_results, vec![data.clone()]);

            // Verify peek/non-peek data retention.
            for i in 0..n {
                let d = rspace.store.get_data(&channels[i]);
                if peeks.contains(&(i as i32)) {
                    prop_assert_eq!(d.len(), 1,
                        "peeked channel {} should retain data", channels[i]);
                } else {
                    prop_assert_eq!(d.len(), 0,
                        "non-peeked channel {} should have data removed", channels[i]);
                }
            }

            // No waiting continuations should remain.
            let c = rspace.store.get_continuations(&channels);
            prop_assert_eq!(c.len(), 0);

            Ok(())
        })?;
    }

    // =========================================================================
    // 3. Non-peek always removes data (both paths, random structure)
    // =========================================================================
    //
    // With an empty peek set (standard consume), data should be removed
    // from ALL channels after a match, regardless of how many channels
    // are involved.  Tests both produce-then-consume and consume-then-produce
    // paths via a boolean toggle.

    #[test]
    fn non_peek_removes_all_data(
        num_channels in 1usize..=4,
        produce_first in proptest::bool::ANY,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channels = distinct_channels(num_channels);
            let data: Vec<String> = (0..num_channels).map(|i| format!("datum{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; num_channels];

            if produce_first {
                // Produce-then-consume path.
                for i in 0..num_channels {
                    let r = rspace.produce(channels[i].clone(), data[i].clone(), false);
                    prop_assert!(r.unwrap().is_none());
                }
                let r = rspace.consume(
                    channels.clone(), patterns, StringsCaptor::new(),
                    false, BTreeSet::default(),
                );
                prop_assert!(r.clone().unwrap().is_some());
                let cont_results = run_k(r.unwrap());
                prop_assert_eq!(cont_results, vec![data.clone()]);
            } else {
                // Consume-then-produce path.
                let r_consume = rspace.consume(
                    channels.clone(), patterns, StringsCaptor::new(),
                    false, BTreeSet::default(),
                );
                prop_assert!(r_consume.unwrap().is_none());

                for i in 0..num_channels.saturating_sub(1) {
                    let r = rspace.produce(channels[i].clone(), data[i].clone(), false);
                    prop_assert!(r.unwrap().is_none());
                }
                let last = num_channels - 1;
                let r_last = rspace.produce(channels[last].clone(), data[last].clone(), false);
                prop_assert!(r_last.clone().unwrap().is_some());
                let cont_results = run_produce_k(r_last.unwrap());
                prop_assert_eq!(cont_results, vec![data.clone()]);
            }

            // ALL data should be removed.
            for i in 0..num_channels {
                let d = rspace.store.get_data(&channels[i]);
                prop_assert_eq!(d.len(), 0,
                    "non-peek should remove data from channel {}", channels[i]);
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // 4. Mixed peek/non-peek on multiple channels (produce-then-consume)
    // =========================================================================
    //
    // With 2..=5 channels where at least one is peeked and at least one is
    // not, only non-peeked channels should have data removed.

    #[test]
    fn mixed_peek_selective_removal_produce_then_consume(
        (channels, peeks) in channels_with_mixed_peeks(),
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let n = channels.len();
            let data: Vec<String> = (0..n).map(|i| format!("d{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; n];

            for i in 0..n {
                let r = rspace.produce(channels[i].clone(), data[i].clone(), false);
                prop_assert!(r.unwrap().is_none());
            }

            let r = rspace.consume(
                channels.clone(), patterns, StringsCaptor::new(),
                false, peeks.clone(),
            );
            prop_assert!(r.clone().unwrap().is_some());
            let cont_results = run_k(r.unwrap());
            prop_assert_eq!(cont_results, vec![data.clone()]);

            for i in 0..n {
                let d = rspace.store.get_data(&channels[i]);
                if peeks.contains(&(i as i32)) {
                    prop_assert_eq!(d.len(), 1,
                        "peeked channel {} must retain data", channels[i]);
                } else {
                    prop_assert_eq!(d.len(), 0,
                        "non-peeked channel {} must lose data", channels[i]);
                }
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // 4b. Mixed peek/non-peek on multiple channels (consume-then-produce)
    // =========================================================================

    #[test]
    fn mixed_peek_selective_removal_consume_then_produce(
        (channels, peeks) in channels_with_mixed_peeks(),
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let n = channels.len();
            let data: Vec<String> = (0..n).map(|i| format!("d{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; n];

            let r_consume = rspace.consume(
                channels.clone(), patterns, StringsCaptor::new(),
                false, peeks.clone(),
            );
            prop_assert!(r_consume.unwrap().is_none());

            // Produce on all but the last (no match yet).
            for i in 0..n.saturating_sub(1) {
                let r = rspace.produce(channels[i].clone(), data[i].clone(), false);
                prop_assert!(r.unwrap().is_none());
            }

            // Final produce completes the match.
            let last = n - 1;
            let r_last = rspace.produce(channels[last].clone(), data[last].clone(), false);
            prop_assert!(r_last.clone().unwrap().is_some());
            let cont_results = run_produce_k(r_last.unwrap());
            prop_assert_eq!(cont_results, vec![data.clone()]);

            for i in 0..n {
                let d = rspace.store.get_data(&channels[i]);
                if peeks.contains(&(i as i32)) {
                    prop_assert_eq!(d.len(), 1,
                        "peeked channel {} must retain data", channels[i]);
                } else {
                    prop_assert_eq!(d.len(), 0,
                        "non-peeked channel {} must lose data", channels[i]);
                }
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // 5. Persistent + peek: data remains (varying persist and peek booleans)
    // =========================================================================
    //
    // With persist=true on produce and peek on consume, data must remain.
    // Also tests the four combinations: {persist, no-persist} x {peek, no-peek}
    // to verify the differential behavior.

    #[test]
    fn persist_and_peek_interaction(
        persist_data in proptest::bool::ANY,
        use_peek in proptest::bool::ANY,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channel = "ch0".to_string();
            let key = vec![channel.clone()];
            let datum = "value".to_string();

            let _ = rspace.produce(channel.clone(), datum.clone(), persist_data);

            let peeks: BTreeSet<i32> = if use_peek {
                std::iter::once(0).collect()
            } else {
                BTreeSet::default()
            };

            let r = rspace.consume(
                key.clone(), vec![Pattern::Wildcard], StringsCaptor::new(),
                false, peeks,
            );
            prop_assert!(r.clone().unwrap().is_some());
            let cont_results = run_k(r.unwrap());
            prop_assert_eq!(cont_results, vec![vec![datum.clone()]]);

            let d = rspace.store.get_data(&channel);

            // Data survives if persist OR peek (or both).
            let should_survive = persist_data || use_peek;
            if should_survive {
                prop_assert_eq!(d.len(), 1,
                    "data should survive (persist={}, peek={})", persist_data, use_peek);
            } else {
                prop_assert_eq!(d.len(), 0,
                    "data should be removed (persist={}, peek={})", persist_data, use_peek);
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // 5b. Persistent consume + peek: continuation and data both survive
    // =========================================================================
    //
    // A persistent consume with peek: after produce, the continuation
    // should remain (persistent) AND the data should remain (peeked).

    #[test]
    fn persistent_consume_with_peek_preserves_both(
        num_produces in 1usize..=3,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channel = "ch0".to_string();
            let key = vec![channel.clone()];

            // Persistent consume with peek.
            let peeks: BTreeSet<i32> = std::iter::once(0).collect();
            let r = rspace.consume(
                key.clone(), vec![Pattern::Wildcard], StringsCaptor::new(),
                true, peeks,
            );
            prop_assert!(r.unwrap().is_none());

            // Produce num_produces times; each should fire the persistent
            // continuation and leave data (peeked).
            for i in 0..num_produces {
                let datum = format!("datum{}", i);
                let r_prod = rspace.produce(channel.clone(), datum.clone(), false);
                prop_assert!(r_prod.clone().unwrap().is_some(),
                    "produce #{} should match persistent peek continuation", i);

                let cont_results = run_produce_k(r_prod.unwrap());
                prop_assert_eq!(cont_results, vec![vec![datum.clone()]]);

                // Continuation must remain (persistent).
                let c = rspace.store.get_continuations(&key);
                prop_assert!(!c.is_empty(),
                    "persistent continuation should remain after produce #{}", i);
            }

            // Data should be present (all peeked produces accumulated).
            let d = rspace.store.get_data(&channel);
            prop_assert!(d.len() >= 1,
                "at least the peeked data should remain");

            Ok(())
        })?;
    }

    // =========================================================================
    // 6. Multiple sequential peeks do not consume data
    // =========================================================================
    //
    // Produce once, then peek-consume N times in a row (2..=10).  Data must
    // survive every iteration.  The structural parameter is the repeat count.

    #[test]
    fn multiple_peeks_do_not_consume_data(
        num_peeks in 2u32..=10,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channel = "ch0".to_string();
            let key = vec![channel.clone()];
            let datum = "datum".to_string();

            let r = rspace.produce(channel.clone(), datum.clone(), false);
            prop_assert!(r.unwrap().is_none());

            for i in 0..num_peeks {
                let peeks: BTreeSet<i32> = std::iter::once(0).collect();
                let r = rspace.consume(
                    key.clone(), vec![Pattern::Wildcard], StringsCaptor::new(),
                    false, peeks,
                );
                prop_assert!(r.clone().unwrap().is_some(),
                    "peek #{} should find data", i);
                let cont_results = run_k(r.unwrap());
                prop_assert_eq!(cont_results, vec![vec![datum.clone()]],
                    "peek #{} should return the correct datum", i);

                let d = rspace.store.get_data(&channel);
                prop_assert_eq!(d.len(), 1, "data must survive peek #{}", i);
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // 6b. Multiple waiting peek consumes, then a single produce
    // =========================================================================
    //
    // Register N (2..=5) waiting peek consumes, then produce once.  One
    // continuation should fire, data should remain, and N-1 continuations
    // should still be waiting.

    #[test]
    fn multiple_waiting_peek_consumes_then_produce(
        num_waiters in 2usize..=5,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channel = "ch0".to_string();
            let key = vec![channel.clone()];
            let datum = "datum".to_string();

            // Register num_waiters peek consumes (all waiting).
            let peeks: BTreeSet<i32> = std::iter::once(0).collect();
            for i in 0..num_waiters {
                let r = rspace.consume(
                    key.clone(), vec![Pattern::Wildcard],
                    StringsCaptor::with_id(i as u64),
                    false, peeks.clone(),
                );
                prop_assert!(r.unwrap().is_none());
            }

            let c = rspace.store.get_continuations(&key);
            prop_assert_eq!(c.len(), num_waiters,
                "should have {} waiting continuations", num_waiters);

            // Produce fires one of them.
            let r_prod = rspace.produce(channel.clone(), datum.clone(), false);
            prop_assert!(r_prod.clone().unwrap().is_some());
            let cont_results = run_produce_k(r_prod.unwrap());
            prop_assert_eq!(cont_results, vec![vec![datum.clone()]]);

            // Data remains (peek).
            let d = rspace.store.get_data(&channel);
            prop_assert_eq!(d.len(), 1, "data should remain after peek produce-match");

            // N-1 continuations remain.
            let c2 = rspace.store.get_continuations(&key);
            prop_assert_eq!(c2.len(), num_waiters - 1,
                "should have {} waiting continuations remaining", num_waiters - 1);

            Ok(())
        })?;
    }

    // =========================================================================
    // Peek vs non-peek differential: same structure, peek flag toggles outcome
    // =========================================================================
    //
    // Run the same scenario (N channels, produce-then-consume) twice -- once
    // with peek on all channels, once without.  The returned data must be
    // identical, but peek must preserve data while non-peek must remove it.

    #[test]
    fn peek_vs_non_peek_differential(
        num_channels in 1usize..=4,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let channels = distinct_channels(num_channels);
            let data: Vec<String> = (0..num_channels).map(|i| format!("d{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; num_channels];
            let all_peeked: BTreeSet<i32> = (0..num_channels as i32).collect();

            // --- Peek scenario ---
            let mut rspace_peek = create_rspace().await;
            for i in 0..num_channels {
                let _ = rspace_peek.produce(channels[i].clone(), data[i].clone(), false);
            }
            let r_peek = rspace_peek.consume(
                channels.clone(), patterns.clone(), StringsCaptor::new(),
                false, all_peeked,
            );
            prop_assert!(r_peek.clone().unwrap().is_some());
            let peek_results = run_k(r_peek.unwrap());

            // --- Non-peek scenario ---
            let mut rspace_normal = create_rspace().await;
            for i in 0..num_channels {
                let _ = rspace_normal.produce(channels[i].clone(), data[i].clone(), false);
            }
            let r_normal = rspace_normal.consume(
                channels.clone(), patterns, StringsCaptor::new(),
                false, BTreeSet::default(),
            );
            prop_assert!(r_normal.clone().unwrap().is_some());
            let normal_results = run_k(r_normal.unwrap());

            // Both return the same data.
            prop_assert_eq!(&peek_results, &normal_results);

            // Peek preserves; non-peek removes.
            for i in 0..num_channels {
                let d_peek = rspace_peek.store.get_data(&channels[i]);
                let d_normal = rspace_normal.store.get_data(&channels[i]);
                prop_assert_eq!(d_peek.len(), 1,
                    "peek should preserve data on channel {}", channels[i]);
                prop_assert_eq!(d_normal.len(), 0,
                    "non-peek should remove data on channel {}", channels[i]);
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // Peek then non-peek removes data (sequential transitions)
    // =========================================================================
    //
    // Produce once, peek K times (data survives), then non-peek once (data
    // removed).  Verifies that peek does not corrupt internal state and that
    // a subsequent normal consume still works correctly.

    #[test]
    fn peek_then_non_peek_removes_data(
        num_peeks_before in 1u32..=5,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channel = "ch0".to_string();
            let key = vec![channel.clone()];
            let datum = "datum".to_string();

            let _ = rspace.produce(channel.clone(), datum.clone(), false);

            // Peek num_peeks_before times -- data survives.
            for i in 0..num_peeks_before {
                let peeks: BTreeSet<i32> = std::iter::once(0).collect();
                let r = rspace.consume(
                    key.clone(), vec![Pattern::Wildcard], StringsCaptor::new(),
                    false, peeks,
                );
                prop_assert!(r.unwrap().is_some(), "peek #{} should succeed", i);
                let d = rspace.store.get_data(&channel);
                prop_assert_eq!(d.len(), 1, "data should survive peek #{}", i);
            }

            // Non-peek consume -- data removed.
            let r = rspace.consume(
                key.clone(), vec![Pattern::Wildcard], StringsCaptor::new(),
                false, BTreeSet::default(),
            );
            prop_assert!(r.clone().unwrap().is_some());
            let cont_results = run_k(r.unwrap());
            prop_assert_eq!(cont_results, vec![vec![datum.clone()]]);

            let d = rspace.store.get_data(&channel);
            prop_assert_eq!(d.len(), 0, "non-peek should remove data after peeks");

            // Further consume should find nothing.
            let r2 = rspace.consume(
                key.clone(), vec![Pattern::Wildcard], StringsCaptor::new(),
                false, BTreeSet::default(),
            );
            prop_assert!(r2.unwrap().is_none());

            Ok(())
        })?;
    }

    // =========================================================================
    // Peek with StringMatch: matching vs non-matching pattern
    // =========================================================================
    //
    // Vary the number of channels, using StringMatch patterns that exactly
    // match the produced data.  Verify peek preserves data.  Then do a
    // consume with a deliberately wrong pattern on one channel to verify
    // no match.

    #[test]
    fn peek_with_string_match_patterns(
        num_channels in 1usize..=3,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channels = distinct_channels(num_channels);
            let data: Vec<String> = (0..num_channels).map(|i| format!("val{}", i)).collect();
            let patterns: Vec<Pattern> = data.iter()
                .map(|d| Pattern::StringMatch(d.clone()))
                .collect();
            let all_peeked: BTreeSet<i32> = (0..num_channels as i32).collect();

            for i in 0..num_channels {
                let _ = rspace.produce(channels[i].clone(), data[i].clone(), false);
            }

            // Matching StringMatch patterns + peek: should match, data preserved.
            let r = rspace.consume(
                channels.clone(), patterns, StringsCaptor::new(),
                false, all_peeked.clone(),
            );
            prop_assert!(r.clone().unwrap().is_some());
            let cont_results = run_k(r.unwrap());
            prop_assert_eq!(cont_results, vec![data.clone()]);

            for i in 0..num_channels {
                let d = rspace.store.get_data(&channels[i]);
                prop_assert_eq!(d.len(), 1, "peek should preserve data on channel {}", channels[i]);
            }

            // Non-matching pattern on the first channel: should NOT match.
            let mut bad_patterns: Vec<Pattern> = data.iter()
                .map(|d| Pattern::StringMatch(d.clone()))
                .collect();
            bad_patterns[0] = Pattern::StringMatch("WILL_NEVER_MATCH_XYZ".to_string());
            let r2 = rspace.consume(
                channels.clone(), bad_patterns, StringsCaptor::new(),
                false, all_peeked,
            );
            prop_assert!(r2.unwrap().is_none(),
                "non-matching StringMatch should cause consume to wait");

            // Data still present (peek from earlier + no consumption from failed match).
            for i in 0..num_channels {
                let d = rspace.store.get_data(&channels[i]);
                prop_assert_eq!(d.len(), 1,
                    "data should still be present on channel {}", channels[i]);
            }

            Ok(())
        })?;
    }

    // =========================================================================
    // Non-peek after peek finds nothing: ordering matters
    // =========================================================================
    //
    // Produce, then non-peek consume (removes data), then peek consume
    // (should find nothing and wait).  Verifies that peek does not
    // resurrect data that was already consumed by a non-peek.

    #[test]
    fn peek_after_non_peek_finds_nothing(
        num_channels in 1usize..=3,
    ) {
        let rt = Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let mut rspace = create_rspace().await;
            let channels = distinct_channels(num_channels);
            let data: Vec<String> = (0..num_channels).map(|i| format!("d{}", i)).collect();
            let patterns: Vec<Pattern> = vec![Pattern::Wildcard; num_channels];
            let all_peeked: BTreeSet<i32> = (0..num_channels as i32).collect();

            for i in 0..num_channels {
                let _ = rspace.produce(channels[i].clone(), data[i].clone(), false);
            }

            // Non-peek consume: removes all data.
            let r1 = rspace.consume(
                channels.clone(), patterns.clone(), StringsCaptor::new(),
                false, BTreeSet::default(),
            );
            prop_assert!(r1.unwrap().is_some());

            for i in 0..num_channels {
                let d = rspace.store.get_data(&channels[i]);
                prop_assert_eq!(d.len(), 0);
            }

            // Subsequent peek consume: should find nothing.
            let r2 = rspace.consume(
                channels.clone(), patterns, StringsCaptor::new(),
                false, all_peeked,
            );
            prop_assert!(r2.unwrap().is_none(),
                "peek after non-peek should find no data");

            // A waiting continuation should be stored.
            let c = rspace.store.get_continuations(&channels);
            prop_assert_eq!(c.len(), 1, "peek consume should store waiting continuation");

            Ok(())
        })?;
    }
}
