use rspace_plus_plus::rspace::history::history_repository::HistoryRepositoryInstances;
use rspace_plus_plus::rspace::hot_store::{HotStoreInstances, HotStoreState};
use rspace_plus_plus::rspace::r#match::Match;
use rspace_plus_plus::rspace::reporting_rspace::{ReportingEvent, ReportingRspace};
use rspace_plus_plus::rspace::rspace::RSpace;
use rspace_plus_plus::rspace::rspace_interface::ISpace;
use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
enum Pattern {
    #[default]
    Wildcard,
}

#[derive(Clone)]
struct StringMatch;

impl Match<Pattern, String> for StringMatch {
    fn get(&self, _p: Pattern, a: String) -> Option<String> {
        Some(a)
    }
}

fn build_reporting_rspace()
-> (RSpace<String, Pattern, String, String>, ReportingRspace<String, Pattern, String, String>) {
    let mut kvm = InMemoryStoreManager::new();
    let store = futures::executor::block_on(kvm.r_space_stores()).unwrap();

    let history_repo = Arc::new(
        HistoryRepositoryInstances::<String, Pattern, String, String>::lmdb_repository(
            store.history.clone(),
            store.roots.clone(),
            store.cold.clone(),
        )
        .unwrap(),
    );

    let cache: HotStoreState<String, Pattern, String, String> = HotStoreState::default();
    let history_reader = history_repo
        .get_history_reader(&history_repo.root())
        .unwrap();
    let hot_store = {
        let hr = history_reader.base();
        HotStoreInstances::create_from_hs_and_hr(cache, hr)
    };
    let space = RSpace::apply(history_repo.clone(), hot_store, Arc::new(Box::new(StringMatch)));

    let reporting_store = {
        let hr = history_reader.base();
        HotStoreInstances::create_from_hr(hr)
    };
    let reporting = ReportingRspace::apply(
        history_repo,
        Arc::new(reporting_store),
        Arc::new(Box::new(StringMatch)),
    );

    (space, reporting)
}

#[tokio::test]
async fn reporting_rspace_should_capture_comm_event_in_soft_report() {
    // Verifies that during replay the logger intercepts a COMM event and persists
    // a ReportingComm entry into the reporting buffer.
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();

    let history_repo = Arc::new(
        HistoryRepositoryInstances::<String, Pattern, String, String>::lmdb_repository(
            store.history.clone(),
            store.roots.clone(),
            store.cold.clone(),
        )
        .unwrap(),
    );

    let cache: HotStoreState<String, Pattern, String, String> = HotStoreState::default();
    let history_reader = history_repo
        .get_history_reader(&history_repo.root())
        .unwrap();
    let hot_store = {
        let hr = history_reader.base();
        HotStoreInstances::create_from_hs_and_hr(cache, hr)
    };

    // base space to generate a valid trace log
    let mut space = RSpace::apply(history_repo.clone(), hot_store, Arc::new(Box::new(StringMatch)));

    let empty_point = space.create_checkpoint().unwrap();

    // Create a COMM in plain space to obtain a valid trace log
    let channels = vec!["ch1".to_string()];
    let patterns = vec![Pattern::Wildcard];
    let continuation = "k".to_string();
    let datum = "d".to_string();

    let _ = space.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::new(),
    );
    let _ = space.produce(channels[0].clone(), datum.clone(), false);
    let rig_point = space.create_checkpoint().unwrap();

    // Create ReportingRspace and run replay
    let reporting_store = {
        let hr = history_reader.base();
        HotStoreInstances::create_from_hr(hr)
    };
    let mut reporting = ReportingRspace::apply(
        history_repo,
        Arc::new(reporting_store),
        Arc::new(Box::new(StringMatch)),
    );

    let _ = reporting.rig_and_reset(empty_point.root, rig_point.log);

    let _ = reporting.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::new(),
    );
    let _ = reporting.produce(channels[0].clone(), datum.clone(), false);

    // Ensure soft_report contains ReportingComm
    let report = reporting.get_report().unwrap();
    assert!(!report.is_empty());
    let flat: Vec<_> = report.into_iter().flatten().collect();
    assert!(
        flat.iter()
            .any(|e| matches!(e, ReportingEvent::ReportingComm(_)))
    );
}

#[tokio::test]
async fn reporting_rspace_should_capture_consume_event_only() {
    // Verifies that calling consume alone produces a ReportingConsume entry and
    // does not require a matching produce to appear in the report.
    let (_space, mut reporting) = build_reporting_rspace();

    let _ = reporting.consume(
        vec!["ch1".to_string()],
        vec![Pattern::Wildcard],
        "k".to_string(),
        false,
        BTreeSet::new(),
    );

    let report = reporting.get_report().unwrap();
    let flat: Vec<_> = report.into_iter().flatten().collect();
    assert!(
        flat.iter()
            .any(|e| matches!(e, ReportingEvent::ReportingConsume(_)))
    );
}

#[tokio::test]
async fn reporting_rspace_should_capture_produce_event_only() {
    // Verifies that calling produce alone produces a ReportingProduce entry and
    // is captured by the reporting logger.
    let (_space, mut reporting) = build_reporting_rspace();

    let _ = reporting.produce("ch1".to_string(), "d".to_string(), false);

    let report = reporting.get_report().unwrap();
    let flat: Vec<_> = report.into_iter().flatten().collect();
    assert!(
        flat.iter()
            .any(|e| matches!(e, ReportingEvent::ReportingProduce(_)))
    );
}

#[tokio::test]
async fn reporting_rspace_should_capture_peeks_in_comm_event() {
    // Verifies that peek semantics are preserved: when a COMM is formed with peeks,
    // the resulting ReportingComm.consume.peeks contains the peeked indices.
    let (mut space, mut reporting) = build_reporting_rspace();
    let empty_point = space.create_checkpoint().unwrap();

    let channels = vec!["ch1".to_string()];
    let patterns = vec![Pattern::Wildcard];
    let continuation = "k".to_string();
    let datum = "d".to_string();

    let _ = space.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::from([0]),
    );
    let _ = space.produce(channels[0].clone(), datum.clone(), false);
    let rig_point = space.create_checkpoint().unwrap();

    let _ = reporting.rig_and_reset(empty_point.root, rig_point.log);

    let _ = reporting.consume(
        channels.clone(),
        patterns.clone(),
        continuation.clone(),
        false,
        BTreeSet::from([0]),
    );
    let _ = reporting.produce(channels[0].clone(), datum.clone(), false);

    let report = reporting.get_report().unwrap();
    let flat: Vec<_> = report.into_iter().flatten().collect();
    let has_peek = flat.iter().any(|e| match e {
        ReportingEvent::ReportingComm(comm) => comm.consume.peeks.contains(&0),
        _ => false,
    });
    assert!(has_peek);
}
