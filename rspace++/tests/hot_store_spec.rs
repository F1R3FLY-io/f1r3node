use std::{
    collections::{BTreeSet, HashMap, HashSet, LinkedList},
    sync::{Arc, Mutex},
};

use dashmap::DashMap;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest_derive::Arbitrary;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rspace_plus_plus::rspace::{
    history::history_reader::HistoryReaderBase,
    hot_store::{HotStore, HotStoreInstances, HotStoreState},
    hot_store_action::{
        DeleteAction, DeleteContinuations, DeleteData, DeleteJoins, HotStoreAction, InsertAction,
        InsertContinuations, InsertData, InsertJoins,
    },
    internal::{Datum, WaitingContinuation},
};
use rstest::*;
use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;

// See rspace/src/test/scala/coop/rchain/rspace/HotStoreSpec.scala

type Channel = String;
type Data = Datum<String>;
type Continuation = WaitingContinuation<Pattern, StringsCaptor>;
type Join = Vec<Channel>;
type Joins = Vec<Join>;

const SIZE_RANGE: usize = 10; // 10

proptest! {
  #![proptest_config(ProptestConfig {
    cases: 20, // 20
    failure_persistence: None,
    .. ProptestConfig::default()
})]

  // Double check these tests perform same logic as Scala tests for Joins
  // For example, 'arbitraryJoins' and double or nested vectors
  // What is difference between 'vec(any::<T>())' and 'any::<Vec<T>>()' in proptest?

  #[test]
  fn get_continuations_when_cache_is_empty_should_read_from_history_and_put_into_cache(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE), history_continuations
    in vec(any::<Continuation>(), 0..=SIZE_RANGE)) {
      let (state, history, hot_store) = fixture();

      history.put_continuations(channels.clone(), history_continuations.clone());

      let cache = state.lock().unwrap();
      assert!(cache.continuations.is_empty());
      drop(cache);

      let read_continuations = hot_store.get_continuations(&channels.clone());
      let cache = state.lock().unwrap();
      assert_eq!(cache.continuations.get(&channels).unwrap().clone(), history_continuations);
      assert_eq!(read_continuations, history_continuations);
  }

  #[test]
  fn get_continuations_when_cache_contains_data_should_read_from_cache_ignoring_history(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE), history_continuations
    in vec(any::<Continuation>(), 0..=SIZE_RANGE), cached_continuations in vec(any::<Continuation>(), 0..=SIZE_RANGE)) {
      let (state, history, hot_store) = fixture();

      history.put_continuations(channels.clone(), history_continuations.clone());
      let mut state_lock = state.lock().unwrap();
      *state_lock = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), cached_continuations.clone())]), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(state_lock);

      let read_continuations = hot_store.get_continuations(&channels.clone());
      let cache = state.lock().unwrap();
      assert_eq!(cache.continuations.get(&channels).unwrap().clone(), cached_continuations);
      assert_eq!(read_continuations, cached_continuations);
  }

  #[test]
  fn get_continuations_should_include_installed_continuations(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE), mut cached_continuations
    in vec(any::<Continuation>(), 0..=SIZE_RANGE), installed_continuation in any::<Continuation>()) {
      let (state, _, hot_store) = fixture();

      let mut state_lock = state.lock().unwrap();
      *state_lock = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), cached_continuations.clone())]), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(state_lock);

      hot_store.install_continuation(&channels.clone(), installed_continuation.clone());
      let res = hot_store.get_continuations(&channels);
      cached_continuations.insert(0, installed_continuation);
      assert_eq!(res, cached_continuations);
  }

  #[test]
  fn put_continuation_when_cache_is_empty_should_read_from_history_and_add_to_it(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE), mut history_continuations
    in vec(any::<Continuation>(), 0..=SIZE_RANGE), inserted_continuation in any::<Continuation>()) {
      let (state, history, hot_store) = fixture();

      history.put_continuations(channels.clone(), history_continuations.clone());
      hot_store.put_continuation(&channels.clone(), inserted_continuation.clone());

      let cache = state.lock().unwrap();
      history_continuations.insert(0, inserted_continuation);
      assert_eq!(cache.continuations.get(&channels).unwrap().clone(), history_continuations);
  }

  #[test]
  fn put_continuation_when_cache_contains_data_should_read_from_cache_and_add_to_it(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE), history_continuations
    in vec(any::<Continuation>(), 0..=SIZE_RANGE), mut cached_continuations in vec(any::<Continuation>(), 0..=SIZE_RANGE), inserted_continuation in any::<Continuation>()) {
      let (state, history, hot_store) = fixture();

      history.put_continuations(channels.clone(), history_continuations.clone());
      let mut state_lock = state.lock().unwrap();
      *state_lock = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), cached_continuations.clone())]), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(state_lock);

      hot_store.put_continuation(&channels.clone(),inserted_continuation.clone());

      let cache = state.lock().unwrap();
      cached_continuations.insert(0, inserted_continuation);
      assert_eq!(cache.continuations.get(&channels).unwrap().clone(), cached_continuations);
  }

  #[test]
  fn install_continuation_should_cache_installed_continuations_separately(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE), mut cached_continuations
    in vec(any::<Continuation>(), 0..=SIZE_RANGE), inserted_continuation in any::<Continuation>(), installed_continuation in any::<Continuation>()) {
      prop_assume!(inserted_continuation != installed_continuation);
      let (state, _, hot_store) = fixture();

      let mut state_lock = state.lock().unwrap();
      *state_lock = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), cached_continuations.clone())]), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(state_lock);

      hot_store.install_continuation(&channels.clone(), installed_continuation.clone());
      hot_store.put_continuation(&channels.clone(), inserted_continuation.clone());

      let cache = state.lock().unwrap();
      cached_continuations.insert(0, inserted_continuation);
      assert_eq!(cache.installed_continuations.get(&channels).unwrap().clone(), installed_continuation);
      assert_eq!(cache.continuations.get(&channels).unwrap().clone(), cached_continuations);
  }

  #[test]
  fn remove_continuation_when_cache_is_empty_should_read_from_history_and_remove_the_continuation_from_loaded_data(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE),
    history_continuations in vec(any::<Continuation>(), 0..=SIZE_RANGE), index in any::<i32>()) {
      let (state, history, hot_store) = fixture();

      history.put_continuations(channels.clone(), history_continuations.clone());
      let res = hot_store.remove_continuation(&channels.clone(), index);

      let state_lock = state.lock().unwrap();
      assert!(check_removal_works_or_fails_on_error(res, state_lock.continuations.get(&channels).map_or(Vec::new(), |x| x.clone()), history_continuations, index).is_ok());
  }

  #[test]
  fn remove_continuation_when_cache_contains_data_should_read_from_cache_and_remove_continuation_from_loaded_data(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE),
    history_continuations in vec(any::<Continuation>(), 0..=SIZE_RANGE), cached_continuations in vec(any::<Continuation>(), 0..=SIZE_RANGE), index in any::<i32>()) {
      let (state, history, hot_store) = fixture();

      history.put_continuations(channels.clone(), history_continuations.clone());
      let mut state_lock = state.lock().unwrap();
      *state_lock = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), cached_continuations.clone())]), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(state_lock);

      let res = hot_store.remove_continuation(&channels.clone(), index);
      let state_lock = state.lock().unwrap();
      assert!(check_removal_works_or_fails_on_error(res, state_lock.continuations.get(&channels).map_or(Vec::new(), |x| x.clone()), cached_continuations, index).is_ok());
  }

  #[test]
  fn remove_continuation_when_installed_continuation_is_present_should_not_allow_its_removal(channels in  vec(any::<Channel>(), 0..=SIZE_RANGE),
    mut cached_continuations in vec(any::<Continuation>(), 0..=SIZE_RANGE), installed_continuation in any::<Continuation>(), index in any::<i32>()) {
      let (state, _, hot_store) = fixture();

      let mut state_lock = state.lock().unwrap();
      *state_lock = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), cached_continuations.clone())]), installed_continuations:  DashMap::from_iter(vec![(channels.clone(), installed_continuation.clone())]),
        data: DashMap::new(), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(state_lock);

      let res = hot_store.remove_continuation(&channels.clone(), index);
      if index == 0 {
        assert!(res.is_none());
      } else {
        // index of the removed continuation includes the installed
        let conts = hot_store.get_continuations(&channels);
        cached_continuations.insert(0, installed_continuation);
        assert!(check_removal_works_or_fails_on_error(res, conts, cached_continuations, index).is_ok());
      }
  }

  #[test]
  fn get_data_when_cache_is_empty_should_read_from_history_and_put_into_the_cache(channel in  any::<Channel>(), history_data in vec(any::<Datum<String>>(), 0..=SIZE_RANGE)) {
      let (state, history, hot_store) = fixture();

      history.put_data(channel.clone(), history_data.clone());
      let cache = state.lock().unwrap();
      assert!(cache.data.is_empty());
      drop(cache);

      let read_data = hot_store.get_data(&channel);
      let cache = state.lock().unwrap();
      assert_eq!(cache.data.get(&channel).unwrap().clone(), history_data);
      assert_eq!(read_data, history_data);
  }

  #[test]
  fn get_data_when_cache_contains_data_should_read_from_cache_ignoring_history(channel in  any::<Channel>(), history_data in vec(any::<Data>(), 0..=SIZE_RANGE),
    cached_data in vec(any::<Data>(), 0..=SIZE_RANGE)) {
      let (state, history, hot_store) = fixture();

      history.put_data(channel.clone(), history_data.clone());
      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::from_iter(vec![(channel.clone(), cached_data.clone())]), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(cache);

      let read_data = hot_store.get_data(&channel);
      let cache = state.lock().unwrap();
      assert_eq!(cache.data.get(&channel).unwrap().clone(), cached_data);
      assert_eq!(read_data, cached_data);
  }

  #[test]
  fn put_datum_when_cache_is_empty_should_read_from_history_and_add_to_it(channel in  any::<Channel>(), mut history_data in vec(any::<Data>(), 0..=SIZE_RANGE),
    inserted_data in any::<Data>()) {
      let (state, history, hot_store) = fixture();

      history.put_data(channel.clone(), history_data.clone());
      hot_store.put_datum(&channel.clone(), inserted_data.clone());

      let cache = state.lock().unwrap();
      history_data.insert(0, inserted_data);
      assert_eq!(cache.data.get(&channel).unwrap().clone(), history_data);
  }

  #[test]
  fn put_datum_when_contains_data_should_read_from_cache_and_add_to_it(channel in  any::<Channel>(), history_data in vec(any::<Data>(), 0..=SIZE_RANGE),
    mut cached_data in vec(any::<Data>(), 0..=SIZE_RANGE), inserted_data in any::<Data>()) {
      let (state, history, hot_store) = fixture();

      history.put_data(channel.clone(), history_data.clone());
      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::from_iter(vec![(channel.clone(), cached_data.clone())]), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(cache);

      hot_store.put_datum(&channel.clone(), inserted_data.clone());
      let cache = state.lock().unwrap();
      cached_data.insert(0, inserted_data);
      assert_eq!(cache.data.get(&channel).unwrap().clone(), cached_data);
  }

  #[test]
  fn remove_datum_when_cache_is_empty_should_read_from_history_and_remove_datum_at_index(channel in  any::<Channel>(), history_data in vec(any::<Data>(), 0..=SIZE_RANGE),
    index in any::<i32>()) {
      let (state, history, hot_store) = fixture();

      history.put_data(channel.clone(), history_data.clone());
      let res = hot_store.remove_datum(&channel.clone(), index);

      let cache = state.lock().unwrap();
      assert!(check_removal_works_or_fails_on_error(res, cache.data.get(&channel).map_or(Vec::new(), |x| x.clone()), history_data, index).is_ok());
  }

  #[test]
  fn remove_datum_when_cache_contains_data_should_read_from_cache_and_remove_datum(channel in  any::<Channel>(), history_data in vec(any::<Data>(), 0..=SIZE_RANGE),
    cached_data in vec(any::<Data>(), 0..=SIZE_RANGE), index in any::<i32>()) {
      let (state, history, hot_store) = fixture();

      history.put_data(channel.clone(), history_data.clone());
      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::from_iter(vec![(channel.clone(), cached_data.clone())]), joins: DashMap::new(), installed_joins: DashMap::new() };
      drop(cache);

      let res = hot_store.remove_datum(&channel.clone(), index);
      let cache = state.lock().unwrap();
      assert!(check_removal_works_or_fails_on_error(res, cache.data.get(&channel).unwrap().clone(), cached_data, index).is_ok());
  }

  #[test]
  fn get_joins_when_cache_is_empty_should_read_from_history_and_put_into_the_cache(channel in  any::<Channel>(), history_joins in vec(any::<Vec<Channel>>(), 0..=SIZE_RANGE)) {
      let (state, history, hot_store) = fixture();

      history.put_joins(channel.clone(), history_joins.clone());
      let cache = state.lock().unwrap();
      assert!(cache.joins.is_empty());
      drop(cache);

      let read_joins = hot_store.get_joins(&channel.clone());
      let cache = state.lock().unwrap();
      assert_eq!(cache.joins.get(&channel).unwrap().clone(), history_joins);
      assert_eq!(read_joins, history_joins);
  }

  #[test]
  fn get_joins_when_cache_contains_data_should_read_from_cache_ignoring_history(channel in  any::<Channel>(), history_joins in vec(any::<Vec<Channel>>(), 0..=SIZE_RANGE),
    cached_joins in vec(any::<Vec<Channel>>(), 0..=SIZE_RANGE)) {
      let (state, history, hot_store) = fixture();

      history.put_joins(channel.clone(), history_joins.clone());
      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]), installed_joins: DashMap::new() };
      drop(cache);

      let read_joins = hot_store.get_joins(&channel.clone());
      let cache = state.lock().unwrap();
      assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
      assert_eq!(read_joins, cached_joins);
  }

  #[test]
  fn put_join_when_cache_is_empty_should_read_from_history_and_add_to_it(channel in  any::<Channel>(), mut history_joins in any::<Joins>(), inserted_join in any::<Join>()) {
      prop_assume!(!history_joins.contains(&inserted_join));
      let (state, history, hot_store) = fixture();

      history.put_joins(channel.clone(), history_joins.clone());
      hot_store.put_join(&channel.clone(), &inserted_join.clone());

      let cache = state.lock().unwrap();
      history_joins.insert(0, inserted_join);
      assert_eq!(cache.joins.get(&channel).unwrap().clone(), history_joins);
  }

  #[test]
  fn put_join_when_cache_contains_data_should_read_from_cache_and_add_to_it(channel in  any::<Channel>(), history_joins in any::<Joins>(), mut cached_joins in any::<Joins>(),
    inserted_join in any::<Join>()) {
      prop_assume!(!cached_joins.contains(&inserted_join));
      let (state, history, hot_store) = fixture();

      history.put_joins(channel.clone(), history_joins.clone());
      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]), installed_joins: DashMap::new() };
      drop(cache);

      hot_store.put_join(&channel.clone(), &inserted_join.clone());
      let cache = state.lock().unwrap();
      cached_joins.insert(0, inserted_join);
      assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
  }

  #[test]
  fn put_join_should_not_allow_inserting_duplicate_joins(channel in  any::<Channel>(), mut cached_joins in any::<Joins>(), inserted_join in any::<Join>()) {
      let (state, _, hot_store) = fixture();

      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]), installed_joins: DashMap::new() };
      drop(cache);

      hot_store.put_join(&channel.clone(), &inserted_join.clone());
      let cache = state.lock().unwrap();

      if !cached_joins.contains(&inserted_join) {
        cached_joins.insert(0, inserted_join);
        assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
      } else {
        assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
      }
  }

  #[test]
  fn install_join_should_cache_installed_joins_separately(channel in  any::<Channel>(), mut cached_joins in any::<Joins>(), inserted_join in any::<Join>(),
    installed_join in any::<Join>()) {
      prop_assume!(inserted_join != installed_join && !cached_joins.contains(&inserted_join));
      let (state, _, hot_store) = fixture();

      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]), installed_joins: DashMap::new() };
      drop(cache);

      hot_store.put_join(&channel.clone(), &inserted_join.clone());
      hot_store.install_join(&channel.clone(), &installed_join.clone());

      let cache = state.lock().unwrap();
      assert_eq!(cache.installed_joins.get(&channel).unwrap().clone(), vec![installed_join]);
      cached_joins.insert(0, inserted_join);
      assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
  }

  #[test]
  fn install_join_should_not_allow_installing_duplicate_joins_per_channel(channel in  any::<Channel>(), cached_joins in any::<Joins>(), installed_join in any::<Join>()) {
      let (state, _, hot_store) = fixture();

      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]), installed_joins: DashMap::new() };
      drop(cache);

      hot_store.install_join(&channel.clone(), &installed_join.clone());
      hot_store.install_join(&channel.clone(), &installed_join.clone());

      let cache = state.lock().unwrap();
      assert_eq!(cache.installed_joins.get(&channel).unwrap().clone(), vec![installed_join]);
  }

  #[test]
  fn remove_join_when_cache_is_empty_should_read_from_history_and_remove_join(channel in  any::<Channel>(), history_joins in any::<Joins>(), index in any::<i32>(), join in any::<Join>()) {
      prop_assume!(!history_joins.contains(&join));
      let (state, history, hot_store) = fixture();

      history.put_joins(channel.clone(), history_joins.clone());
      let to_remove = history_joins.get(index as usize).unwrap_or(&join).clone();
      let res = hot_store.remove_join(&channel.clone(), &to_remove);

      let cache = state.lock().unwrap();
      assert!(check_removal_works_or_ignores_errors(res, cache.joins.get(&channel).unwrap().clone(), history_joins, index).is_ok());
  }

  #[test]
  fn remove_join_when_cache_contains_data_should_read_from_the_cache_and_remove_join(channel in  any::<Channel>(), history_joins in any::<Joins>(), cached_joins in any::<Joins>(),
    index in any::<i32>(), join in any::<Join>()) {
      prop_assume!(!cached_joins.contains(&join));
      let (state, history, hot_store) = fixture();

      history.put_joins(channel.clone(), history_joins.clone());
      let to_remove = cached_joins.get(index as usize).unwrap_or(&join).clone();
      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]), installed_joins: DashMap::new() };
      drop(cache);

      let res = hot_store.remove_join(&channel.clone(), &to_remove);
      let cache = state.lock().unwrap();
      assert!(check_removal_works_or_ignores_errors(res, cache.joins.get(&channel).unwrap().clone(), cached_joins, index).is_ok());
  }

  #[test]
  fn remove_join_when_installed_joins_are_present_should_not_allow_removing_them(channel in  any::<Channel>(), cached_joins in any::<Joins>(), installed_joins in any::<Joins>()) {
      prop_assume!(cached_joins != installed_joins && !installed_joins.is_empty());
      let (state, _, hot_store) = fixture();

      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]),
        installed_joins: DashMap::from_iter(vec![(channel.clone(), installed_joins.clone())]) };
      drop(cache);

      let mut rng = thread_rng();
      let mut shuffled_joins = installed_joins.clone();
      shuffled_joins.shuffle(&mut rng);
      let to_remove = shuffled_joins.first().unwrap().clone();

      let res = hot_store.remove_join(&channel.clone(), &to_remove.clone());
      let cache = state.lock().unwrap();

      if !cached_joins.contains(&to_remove) {
        assert!(res.is_some());
        assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
      } else {
        let to_remove_count_in_cache = cache.joins.get(&channel).unwrap().clone().into_iter().filter(|x| x.clone() == to_remove).count();
        let to_remove_count_in_cached_joins = cached_joins.into_iter().filter(|x| x.clone() == to_remove).count();
        assert_eq!(to_remove_count_in_cache, to_remove_count_in_cached_joins - 1);
        assert_eq!(cache.installed_joins.get(&channel).unwrap().clone(), installed_joins);
      }
  }

  #[test]
  fn remove_join_should_not_remove_a_join_when_a_continuation_is_present(channel in  any::<Channel>(), continuation in any::<Continuation>(), cached_joins in any::<Joins>()) {
      prop_assume!(!cached_joins.is_empty());
      let (state, _, hot_store) = fixture();

      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::new(), installed_continuations: DashMap::new(), data: DashMap::new(), joins: DashMap::from_iter(vec![(channel.clone(), cached_joins.clone())]),
        installed_joins: DashMap::new() };
      drop(cache);

      let mut rng = thread_rng();
      let mut shuffled_joins = cached_joins.clone();
      shuffled_joins.shuffle(&mut rng);
      let to_remove = shuffled_joins.first().unwrap().clone();

      hot_store.put_continuation(&to_remove.clone(), continuation);
      let res = hot_store.remove_join(&channel.clone(), &to_remove.clone());
      let cache = state.lock().unwrap();

      assert!(res.is_some());
      assert_eq!(cache.joins.get(&channel).unwrap().clone(), cached_joins);
  }

    #[test]
  fn changes_should_return_information_to_be_persisted_in_history(channels in vec(any::<Channel>(), 0..=SIZE_RANGE), channel in  any::<Channel>(), continuations in  vec(any::<Continuation>(), 0..=SIZE_RANGE),
        installed_continuation in any::<Continuation>(), data in vec(any::<Data>(), 0..=SIZE_RANGE), joins in any::<Joins>()) {
      let (state, _, hot_store) = fixture();

      let mut cache = state.lock().unwrap();
      *cache = HotStoreState { continuations: DashMap::from_iter(vec![(channels.clone(), continuations.clone())]), installed_continuations: DashMap::from_iter(vec![(channels.clone(), installed_continuation.clone())]),
                data: DashMap::from_iter(vec![(channel.clone(), data.clone())]), joins: DashMap::from_iter(vec![(channel.clone(), joins.clone())]),
        installed_joins: DashMap::new() };
      drop(cache);

            let res = hot_store.changes();
            let cache = state.lock().unwrap();
            assert_eq!(res.len(), cache.continuations.len() + cache.data.len() + cache.joins.len());

            if continuations.is_empty() {
        assert!(res.contains(&HotStoreAction::Delete(DeleteAction::DeleteContinuations(DeleteContinuations { channels }))));
      } else {
        assert!(res.contains(&HotStoreAction::Insert(InsertAction::InsertContinuations(InsertContinuations { channels, continuations }))));
      }

      if data.is_empty() {
        assert!(res.contains(&HotStoreAction::Delete(DeleteAction::DeleteData(DeleteData { channel: channel.clone() }))));
      } else {
        assert!(res.contains(&HotStoreAction::Insert(InsertAction::InsertData(InsertData { channel: channel.clone(), data }))));
      }

      if joins.is_empty() {
        assert!(res.contains(&HotStoreAction::Delete(DeleteAction::DeleteJoins(DeleteJoins { channel }))));
      } else {
        assert!(res.contains(&HotStoreAction::Insert(InsertAction::InsertJoins(InsertJoins { channel, joins }))));
      }
  }

  #[test]
  fn concurrent_data_operations_on_disjoint_channels_should_not_mess_up_the_cache(channel1 in  any::<Channel>(), channel2 in  any::<Channel>(), mut history_data1 in vec(any::<Data>(), 0..=SIZE_RANGE), mut history_data2 in vec(any::<Data>(), 0..=SIZE_RANGE),
    inserted_data1 in any::<Data>(), inserted_data2 in any::<Data>()) {
      prop_assume!(channel1 != channel2);
      let (_, history, hot_store) = fixture();
      let hot_store = Arc::new(Mutex::new(hot_store));

      history.put_data(channel1.clone(), history_data1.clone());
      history.put_data(channel2.clone(), history_data2.clone());

      // Clone the Arc to share it between threads
     let hot_store1 = Arc::clone(&hot_store);
     let hot_store2 = Arc::clone(&hot_store);

     // Clone the channels and data to move them into the threads
     let channel1_clone = channel1.clone();
     let channel2_clone = channel2.clone();
     let inserted_data1_clone = inserted_data1.clone();
     let inserted_data2_clone = inserted_data2.clone();

      // Spawn two threads to run put_datum in parallel and waits for both threads to complete using join.
      let handle1 = std::thread::spawn(move || {
        let hot_store = hot_store1.lock().unwrap();
        hot_store.put_datum(&channel1_clone, inserted_data1_clone);
      });
      let handle2 = std::thread::spawn(move || {
        let hot_store = hot_store2.lock().unwrap();
        hot_store.put_datum(&channel2_clone, inserted_data2_clone);
      });
      handle1.join().unwrap();
      handle2.join().unwrap();

      let r1 = hot_store.lock().unwrap().get_data(&channel1);
      let r2 = hot_store.lock().unwrap().get_data(&channel2);
      history_data1.insert(0, inserted_data1);
      history_data2.insert(0, inserted_data2);

      assert_eq!(r1, history_data1);
      assert_eq!(r2, history_data2);
  }

  #[test]
  fn concurrent_coninuation_operations_on_disjoint_channels_should_not_mess_up_the_cache(channels1 in  vec(any::<Channel>(), 0..=SIZE_RANGE), channels2 in  vec(any::<Channel>(), 0..=SIZE_RANGE), mut history_continuations1 in vec(any::<Continuation>(), 0..=SIZE_RANGE),
    mut history_continuations2 in vec(any::<Continuation>(), 0..=SIZE_RANGE), inserted_continuation1 in any::<Continuation>(), inserted_continuation2 in any::<Continuation>()) {
      prop_assume!(channels1 != channels2);
      let (_, history, hot_store) = fixture();
      let hot_store = Arc::new(Mutex::new(hot_store));

      history.put_continuations(channels1.clone(), history_continuations1.clone());
      history.put_continuations(channels2.clone(), history_continuations2.clone());

      // Clone the Arc to share it between threads
     let hot_store1 = Arc::clone(&hot_store);
     let hot_store2 = Arc::clone(&hot_store);

     // Clone the channels and data to move them into the threads
     let channels1_clone = channels1.clone();
     let channels2_clone = channels2.clone();
     let inserted_continuation1_clone = inserted_continuation1.clone();
     let inserted_continuation2_clone = inserted_continuation2.clone();

      // Spawn two threads to run put_datum in parallel and waits for both threads to complete using join.
      let handle1 = std::thread::spawn(move || {
        let hot_store = hot_store1.lock().unwrap();
        hot_store.put_continuation(&channels1_clone, inserted_continuation1_clone);
      });
      let handle2 = std::thread::spawn(move || {
        let hot_store = hot_store2.lock().unwrap();
        hot_store.put_continuation(&channels2_clone, inserted_continuation2_clone);
      });
      handle1.join().unwrap();
      handle2.join().unwrap();

      let r1 = hot_store.lock().unwrap().get_continuations(&channels1);
      let r2 = hot_store.lock().unwrap().get_continuations(&channels2);
      history_continuations1.insert(0, inserted_continuation1);
      history_continuations2.insert(0, inserted_continuation2);

      assert_eq!(r1, history_continuations1);
      assert_eq!(r2, history_continuations2);
  }

  /* BREAK */

  #[test]
  fn put_datum_should_put_datum_in_a_new_channel(channel in  any::<String>(), datum_value in any::<String>()) {
      let (_, _, hot_store) = fixture();
      let key = channel.clone();
      let datum = Datum::create(channel, datum_value, false);

      hot_store.put_datum(&key.clone(), datum.clone());
      let res = hot_store.get_data(&key);
      assert!(check_same_elements(res, vec![datum]));
  }

  #[test]
  fn put_datum_should_append_datum_if_channel_already_exists(channel in  any::<String>(), datum_value in any::<String>()) {
      let (_, _, hot_store) = fixture();
      let key = channel.clone();
      let datum1 = Datum::create(channel.clone(), datum_value.clone(), false);
      let datum2 = Datum::create(channel, datum_value + "2", false);

      hot_store.put_datum(&key.clone(), datum1.clone());
      hot_store.put_datum(&key.clone(), datum2.clone());
      let res = hot_store.get_data(&key);
      assert!(check_same_elements(res, vec![datum1, datum2]));
  }

  // TODO: Set min_successful to 10
  // TODO: Double chck test case matches because of validIndices on Scala side
  #[test]
  fn remove_datum_should_remove_datum_at_index(channel in  any::<String>(), datum_value in any::<String>(), index in any::<i32>()) {
      let (_, _, hot_store) = fixture();
      let key = channel.clone();
      let data: Vec<Datum<String>> = (0..11)
        .map(|i| Datum::create(channel.clone(), datum_value.clone() + &i.to_string(), false))
        .collect();

      for d in data.clone() {
        hot_store.put_datum(&key.clone(), d);
      }

      hot_store.remove_datum(&key.clone(), index - 1);
      let res = hot_store.get_data(&key);
      let expected: Vec<Datum<String>> = data.into_iter()
         .filter(|d| d.a != datum_value.clone() + &(11 - index).to_string())
         .collect();
      assert!(check_same_elements(res, expected));
  }

  #[test]
  fn put_waiting_continuation_should_put_waiting_continuation_in_a_new_channel(channel in  any::<String>(), pattern in any::<String>()) {
      let (_, _, hot_store) = fixture();
      let key = vec![channel.clone()];
      let patterns = vec![Pattern::StringMatch(pattern)];
      let continuation = StringsCaptor::new();
      let wc = WaitingContinuation::create(key.clone(), patterns, continuation, false, BTreeSet::default());

      hot_store.put_continuation(&key.clone(), wc.clone());
      let res = hot_store.get_continuations(&key);
      assert_eq!(res, vec![wc]);
  }

  #[test]
  fn put_waiting_continuation_should_append_continuation_if_channel_already_exists(channel in  any::<String>(), pattern in any::<String>()) {
      let (_, _, hot_store) = fixture();
      let key = vec![channel.clone()];
      let patterns = vec![Pattern::StringMatch(pattern.clone())];
      let continuation = StringsCaptor::new();
      let wc1 = WaitingContinuation::create(key.clone(), patterns, continuation.clone(), false, BTreeSet::default());
      let wc2 = WaitingContinuation::create(key.clone(), vec![Pattern::StringMatch(pattern + "2")], continuation, false, BTreeSet::default());

      hot_store.put_continuation(&key.clone(), wc1.clone());
      hot_store.put_continuation(&key.clone(), wc2.clone());
      let res = hot_store.get_continuations(&key);
      assert!(check_same_elements(res, vec![wc1, wc2]));
  }

  #[test]
  fn remove_waiting_continuation_should_remove_waiting_continuation_from_index(channel in  any::<String>(), pattern in any::<String>()) {
      let (_, _, hot_store) = fixture();
      let key = vec![channel.clone()];
      let patterns = vec![Pattern::StringMatch(pattern.clone())];
      let continuation = StringsCaptor::new();
      let wc1 = WaitingContinuation::create(key.clone(), patterns, continuation.clone(), false, BTreeSet::default());
      let wc2 = WaitingContinuation::create(key.clone(), vec![Pattern::StringMatch(pattern + "2")], continuation, false, BTreeSet::default());

      hot_store.put_continuation(&key.clone(), wc1.clone());
      hot_store.put_continuation(&key.clone(), wc2.clone());
      hot_store.remove_continuation(&key.clone(), 0);
      let res = hot_store.get_continuations(&key);
      assert!(check_same_elements(res, vec![wc1]));
  }

  #[test]
  fn add_join_should_add_join_for_a_channel(channel in  any::<String>(), channels in vec(any::<String>(), 0..=SIZE_RANGE)) {
      let (_, _, hot_store) = fixture();

      hot_store.put_join(&channel.clone(), &channels.clone());
      let res = hot_store.get_joins(&channel);
      assert_eq!(res, vec![channels]);
  }

  #[test]
  fn remove_join_should_remove_join_for_a_channel(channel in  any::<String>(), channels in vec(any::<String>(), 0..=SIZE_RANGE)) {
      let (_, _, hot_store) = fixture();

      hot_store.put_join(&channel.clone(), &channels.clone());
      hot_store.remove_join(&channel.clone(), &channels.clone());
      let res = hot_store.get_joins(&channel);
      assert!(res.is_empty());
  }

  #[test]
  fn remove_join_should_remove_only_passed_in_joins_for_a_channel(channel in  any::<String>(), channels in vec(any::<String>(), 0..=SIZE_RANGE)) {
      let (_, _, hot_store) = fixture();

      hot_store.put_join(&channel.clone(), &channels.clone());
      hot_store.put_join(&channel.clone(), &vec!["other_channel".to_string()]);
      hot_store.remove_join(&channel.clone(), &channels.clone());
      let res = hot_store.get_joins(&channel);
      assert_eq!(res, vec![vec!["other_channel".to_string()]]);
  }

  #[test]
  fn snapshot_should_create_a_copy_of_the_cache(_cache in  any::<String>()) {
      let cache: HotStoreState<String, Pattern, String, StringsCaptor> = HotStoreState::random_state();
      let (_, _, hot_store) = fixture_with_cache(cache.clone());

      let snapshot = hot_store.snapshot();
      assert!(compare_dashmaps(&snapshot.continuations, &cache.continuations));
      assert!(compare_dashmaps(&snapshot.installed_continuations, &cache.installed_continuations));
      assert!(compare_dashmaps(&snapshot.data, &cache.data));
      assert!(compare_dashmaps(&snapshot.joins, &cache.joins));
      assert!(compare_dashmaps(&snapshot.installed_joins, &cache.installed_joins));
  }

  #[test]
  fn remove_join_should_create_a_deep_copy_of_the_continuations_in_the_cache(channels in  vec(any::<String>(), 0..=SIZE_RANGE), continuation1 in any::<Continuation>(), continuation2 in any::<Continuation>()) {
      prop_assume!(continuation1 != continuation2);
      let (_, _, hot_store) = fixture();

      hot_store.put_continuation(&channels.clone(), continuation1.clone());
      let snapshot = hot_store.snapshot();
      hot_store.put_continuation(&channels.clone(), continuation2.clone());
      assert!(snapshot.continuations.get(&channels).unwrap().clone().contains(&continuation1));
      assert!(!snapshot.continuations.get(&channels).unwrap().clone().contains(&continuation2));
  }

  #[test]
  fn remove_join_should_create_a_deep_copy_of_the_installed_continuations_in_the_cache(channels in  vec(any::<String>(), 0..=SIZE_RANGE), continuation1 in any::<Continuation>(), continuation2 in any::<Continuation>()) {
      prop_assume!(continuation1 != continuation2);
      let (_, _, hot_store) = fixture();

      hot_store.install_continuation(&channels.clone(), continuation1.clone());
      let snapshot = hot_store.snapshot();
      hot_store.install_continuation(&channels.clone(), continuation2.clone());
      assert_eq!(snapshot.installed_continuations.get(&channels).unwrap().clone(), continuation1);
  }

  #[test]
  fn remove_join_should_create_a_deep_copy_of_the_data_in_the_cache(channel in  any::<String>(), data1 in any::<Data>(), data2 in any::<Data>()) {
      prop_assume!(data1 != data2);
      let (_, _, hot_store) = fixture();

      hot_store.put_datum(&channel.clone(), data1.clone());
      let snapshot = hot_store.snapshot();
      hot_store.put_datum(&channel.clone(), data2.clone());
      assert!(!snapshot.data.get(&channel).unwrap().clone().contains(&data2));
  }

  #[test]
  fn remove_join_should_create_a_deep_copy_of_the_joins_in_the_cache(channel in  any::<String>(), join1 in any::<Join>(), join2 in any::<Join>()) {
      prop_assume!(join1 != join2);
      let (_, _, hot_store) = fixture();

      hot_store.put_join(&channel.clone(), &join1.clone());
      let snapshot = hot_store.snapshot();
      hot_store.put_join(&channel.clone(), &join2.clone());
      assert!(!snapshot.joins.get(&channel).unwrap().clone().contains(&join2));
  }

  #[test]
  fn remove_join_should_create_a_deep_copy_of_the_installed_joins_in_the_cache(channel in  any::<String>(), join1 in any::<Join>(), join2 in any::<Join>()) {
      prop_assume!(join1 != join2);
      let (_, _, hot_store) = fixture();

      hot_store.install_join(&channel.clone(), &join1.clone());
      let snapshot = hot_store.snapshot();
      hot_store.install_join(&channel.clone(), &join2.clone());
      assert!(snapshot.installed_joins.get(&channel).unwrap().clone().contains(&join1));
      assert!(!snapshot.installed_joins.get(&channel).unwrap().clone().contains(&join2));
  }
}

fn check_removal_works_or_fails_on_error<T>(
    res: Option<()>,
    actual: Vec<T>,
    initial: Vec<T>,
    index: i32,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: PartialEq + Debug + Clone,
{
    if index < 0 || index >= initial.len().try_into().unwrap() {
        assert!(res.is_none());
        assert_eq!(actual, initial);
    } else {
        assert!(res.is_some());
        let expected: Vec<T> = initial
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as i32 != index)
            .map(|(_, item)| item.clone())
            .collect();
        assert_eq!(actual, expected);
    }
    Ok(())
}

fn check_removal_works_or_ignores_errors<T>(
    res: Option<()>,
    actual: Vec<T>,
    initial: Vec<T>,
    index: i32,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: PartialEq + Debug + Clone,
{
    if index < 0 || index >= initial.len().try_into().unwrap() {
        assert!(res.is_some());
        assert_eq!(actual, initial);
    } else {
        assert!(res.is_some());
        let expected: Vec<T> = initial
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as i32 != index)
            .map(|(_, item)| item.clone())
            .collect();
        assert_eq!(actual, expected);
    }
    Ok(())
}

// We only care that both vectors contain the same elements, not their ordering
fn check_same_elements<T: Hash + Eq>(vec1: Vec<T>, vec2: Vec<T>) -> bool {
    let set1: HashSet<_> = vec1.into_iter().collect();
    let set2: HashSet<_> = vec2.into_iter().collect();
    set1 == set2
}

pub fn compare_dashmaps<K, V>(map1: &DashMap<K, V>, map2: &DashMap<K, V>) -> bool
where
    K: Eq + Hash + Clone,
    V: PartialEq + Clone,
{
    let hash_map1: HashMap<K, V> = map1
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();
    let hash_map2: HashMap<K, V> = map2
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();
    hash_map1 == hash_map2
}

// See rspace/src/main/scala/coop/rchain/rspace/examples/StringExamples.scala
#[derive(Clone, Debug, Default, PartialEq, Arbitrary, Serialize, Eq, Hash)]
pub enum Pattern {
    #[default]
    Wildcard,
    StringMatch(String),
}

#[derive(Clone, Debug, Default, PartialEq, Arbitrary, Serialize, Eq, Hash)]
pub struct StringsCaptor {
    res: LinkedList<Vec<String>>,
}

impl StringsCaptor {
    fn new() -> Self {
        StringsCaptor {
            res: LinkedList::new(),
        }
    }
}

// See rspace/src/test/scala/coop/rchain/rspace/HotStoreSpec.scala
#[derive(Clone)]
pub struct TestHistory<C: Eq + Hash, P: Clone, A: Clone, K: Clone> {
    state: Arc<Mutex<HotStoreState<C, P, A, K>>>,
}

impl<C: Clone + Eq + Hash + Send, P: Clone + Send, A: Clone + Send, K: Clone + Send>
    HistoryReaderBase<C, P, A, K> for TestHistory<C, P, A, K>
{
    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        let state_lock = self.state.lock().unwrap();
        let data = state_lock
            .data
            .get(channel)
            .map(|v| v.to_vec())
            .unwrap_or_else(|| Vec::new());
        data
    }

    fn get_continuations(&self, channels: &Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        let state_lock = self.state.lock().unwrap();
        let continuations = state_lock
            .continuations
            .get(channels)
            .map(|v| v.to_vec())
            .unwrap_or_else(|| Vec::new());
        continuations
    }

    fn get_joins(&self, channel: &C) -> Vec<Vec<C>> {
        let state_lock = self.state.lock().unwrap();
        let joins = state_lock
            .joins
            .get(channel)
            .map(|v| v.to_vec())
            .unwrap_or_else(|| Vec::new());
        joins
    }

    fn get_data_proj(&self, _key: &C) -> Vec<Datum<A>> {
        todo!()
    }

    fn get_continuations_proj(&self, _key: &Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        todo!()
    }

    fn get_joins_proj(&self, _key: &C) -> Vec<Vec<C>> {
        todo!()
    }
}

impl<C: Eq + Hash, P: Clone, A: Clone, K: Clone> TestHistory<C, P, A, K> {
    fn put_data(&self, channel: C, data: Vec<Datum<A>>) -> () {
        let state = self.state.lock().unwrap();
        state.data.insert(channel, data);
    }

    fn put_continuations(
        &self,
        channels: Vec<C>,
        continuations: Vec<WaitingContinuation<P, K>>,
    ) -> () {
        let state = self.state.lock().unwrap();
        state.continuations.insert(channels, continuations);
    }

    fn put_joins(&self, channel: C, joins: Vec<Vec<C>>) -> () {
        let state = self.state.lock().unwrap();
        state.joins.insert(channel, joins);
    }
}

type StateSetup = (
    Arc<Mutex<HotStoreState<String, Pattern, String, StringsCaptor>>>,
    TestHistory<String, Pattern, String, StringsCaptor>,
    Box<dyn HotStore<String, Pattern, String, StringsCaptor>>,
);

#[fixture]
pub fn fixture() -> StateSetup {
    let history_state =
        Arc::new(Mutex::new(HotStoreState::<String, Pattern, String, StringsCaptor>::default()));

    let history = TestHistory {
        state: history_state.clone(),
    };

    let cache =
        Arc::new(Mutex::new(HotStoreState::<String, Pattern, String, StringsCaptor>::default()));

    let hot_store =
        HotStoreInstances::create_from_mhs_and_hr(cache.clone(), Box::new(history.clone()));
    (cache, history, hot_store)
}

pub fn fixture_with_cache(
    cache: HotStoreState<String, Pattern, String, StringsCaptor>,
) -> StateSetup {
    let history_state =
        Arc::new(Mutex::new(HotStoreState::<String, Pattern, String, StringsCaptor>::default()));

    let history = TestHistory {
        state: history_state.clone(),
    };

    let cache = Arc::new(Mutex::new(cache));

    let hot_store =
        HotStoreInstances::create_from_mhs_and_hr(cache.clone(), Box::new(history.clone()));
    (cache, history, hot_store)
}
