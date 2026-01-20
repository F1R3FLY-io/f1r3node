use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::history::instances::radix_history::RadixHistory;
use rspace_plus_plus::rspace::hot_store::HotStoreInstances;
use rspace_plus_plus::rspace::r#match::Match;
use rspace_plus_plus::rspace::rspace_interface::ISpace;
use rspace_plus_plus::rspace::state::exporters::rspace_exporter_items::RSpaceExporterItems;
use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporterInstance;
use rspace_plus_plus::rspace::{
    history::history_repository::HistoryRepositoryInstances,
    hot_store::HotStoreState,
    rspace::RSpace,
    shared::{
        in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
    },
    state::{rspace_exporter::RSpaceExporter, rspace_importer::RSpaceImporter},
};
use serde::{Deserialize, Serialize};
use shared::rust::{Byte, ByteVector};
use std::collections::BTreeSet;
use std::sync::Arc;

// See rspace/src/main/scala/coop/rchain/rspace/examples/StringExamples.scala
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
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

// See rspace/src/test/scala/coop/rchain/rspace/ExportImportTests.scala
// TODO: Don't works for MergingHistory
#[tokio::test]
async fn export_and_import_of_one_page_should_works_correctly() {
    let (mut space1, exporter1, importer1, mut space2, _, importer2) = test_setup().await;

    let page_size = 1000;
    let data_size = 10;
    let start_skip = 0;
    let pattern = vec![Pattern::Wildcard];
    let continuation = "continuation".to_string();

    // Generate init data in space1
    for i in 0..data_size {
        let res = space1.produce(format!("ch{}", i), format!("data{}", i), false, None);
        assert!(res.is_ok());
    }

    let init_point = space1.create_checkpoint().unwrap();

    // Export 1 page from space1
    let init_start_path: Vec<(Blake2b256Hash, Option<Byte>)> =
        vec![(init_point.root.clone(), None)];
    let export_data = RSpaceExporterItems::get_history_and_data(
        exporter1,
        init_start_path.clone(),
        start_skip,
        page_size,
    );
    let history_items = export_data.0.items;
    let data_items = export_data.1.items;

    // println!("\ndata_items_page: {:?}", data_items);

    // Validate exporting page
    let _ = RSpaceImporterInstance::validate_state_items(
        history_items.clone(),
        data_items.clone(),
        init_start_path,
        page_size,
        start_skip,
        importer1,
    );

    // Import page to space2
    let _ = importer2.set_history_items(history_items);
    let _ = importer2.set_data_items(data_items);
    let _ = importer2.set_root(&init_point.root);
    let _ = space2.reset(&init_point.root);

    // space2.store.print();

    // Testing data in space2 (match all installed channels)
    for i in 0..data_size {
        let res = space2.consume(
            vec![format!("ch{}", i)],
            pattern.clone(),
            continuation.clone(),
            false,
            BTreeSet::new(),
        );
        assert!(res.is_ok());
    }

    // println!("\nspace2: {:?}", space2.to_map());

    let end_point = space2.create_checkpoint().unwrap();
    assert_eq!(end_point.root, RadixHistory::empty_root_node_hash())
}

#[tokio::test]
async fn multipage_export_should_work_correctly() {
    let (mut space1, exporter1, importer1, mut space2, _, importer2) = test_setup().await;

    let page_size = 10;
    let data_size = 1000;
    let start_skip = 0;
    let pattern = vec![Pattern::Wildcard];
    let continuation = "continuation".to_string();

    type Params = (
        Vec<(Blake2b256Hash, ByteVector)>,   // HistoryItems
        Vec<(Blake2b256Hash, ByteVector)>,   // DataItems
        Vec<(Blake2b256Hash, Option<Byte>)>, // StartPath
    );

    let multipage_export = |params: Params,
                            exporter1: Arc<dyn RSpaceExporter>,
                            importer1: Arc<dyn RSpaceImporter>|
     -> Result<Params, Params> {
        match params {
            (history_items, data_items, start_path) => {
                // Export 1 page from space1
                let export_data = RSpaceExporterItems::get_history_and_data(
                    exporter1,
                    start_path.clone(),
                    start_skip,
                    page_size,
                );

                let history_items_page = export_data.0.items;
                let data_items_page = export_data.1.items;
                let last_path = export_data.0.last_path;

                // Validate exporting page
                let _ = RSpaceImporterInstance::validate_state_items(
                    history_items_page.clone(),
                    data_items_page.clone(),
                    start_path,
                    page_size,
                    start_skip,
                    importer1,
                );

                let r = (
                    [history_items, history_items_page.clone()].concat(),
                    [data_items, data_items_page].concat(),
                    last_path,
                );

                if (history_items_page.len() as i32) < page_size {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
        }
    };

    let init_history_items: Vec<(Blake2b256Hash, ByteVector)> = Vec::new();
    let init_data_items: Vec<(Blake2b256Hash, ByteVector)> = Vec::new();
    let init_child_num: Option<Byte> = None;

    // Generate init data in space1
    for i in 0..data_size {
        let res = space1.produce(format!("ch{}", i), format!("data{}", i), false, None);
        assert!(res.is_ok());
    }

    let init_point = space1.create_checkpoint().unwrap();

    // Multipage export from space1
    let init_start_path: Vec<(Blake2b256Hash, Option<Byte>)> =
        vec![(init_point.root.clone(), init_child_num)];
    let init_export_data = (init_history_items, init_data_items, init_start_path);
    let mut export_data = init_export_data;
    loop {
        match multipage_export(export_data, exporter1.clone(), importer1.clone()) {
            Ok(data) => {
                export_data = data;
            }
            Err(final_data) => {
                export_data = final_data;
                break;
            }
        }
    }

    let history_items = export_data.0;
    let data_items = export_data.1;

    // Import page to space2
    let _ = importer2.set_history_items(history_items);
    let _ = importer2.set_data_items(data_items);
    let _ = importer2.set_root(&init_point.root);
    let _ = space2.reset(&init_point.root);

    // Testing data in space2 (match all installed channels)
    for i in 0..data_size {
        let res = space2.consume(
            vec![format!("ch{}", i)],
            pattern.clone(),
            continuation.clone(),
            false,
            BTreeSet::new(),
        );
        assert!(res.is_ok());
    }
    let end_point = space2.create_checkpoint().unwrap();
    assert_eq!(end_point.root, RadixHistory::empty_root_node_hash())
}

// Attention! Skipped export is significantly slower than last path export.
// But on the other hand, this allows you to work simultaneously with several nodes.
#[tokio::test]
async fn multipage_export_with_skip_should_work_correctly() {
    let (mut space1, exporter1, importer1, mut space2, _, importer2) = test_setup().await;

    let page_size = 10;
    let data_size = 1000;
    let start_skip = 0;
    let pattern = vec![Pattern::Wildcard];
    let continuation = "continuation".to_string();

    type Params = (
        Vec<(Blake2b256Hash, ByteVector)>,   // HistoryItems
        Vec<(Blake2b256Hash, ByteVector)>,   // DataItems
        Vec<(Blake2b256Hash, Option<Byte>)>, // StartPath
        i32,                                 // Size of skip
    );

    let multipage_export_with_skip = |params: Params,
                                      exporter1: Arc<dyn RSpaceExporter>,
                                      importer1: Arc<dyn RSpaceImporter>|
     -> Result<Params, Params> {
        match params {
            (history_items, data_items, start_path, skip) => {
                // Export 1 page from space1
                let export_data = RSpaceExporterItems::get_history_and_data(
                    exporter1,
                    start_path.clone(),
                    skip,
                    page_size,
                );

                let history_items_page = export_data.0.items;
                let data_items_page = export_data.1.items;

                // Validate exporting page
                let _ = RSpaceImporterInstance::validate_state_items(
                    history_items_page.clone(),
                    data_items_page.clone(),
                    start_path.clone(),
                    page_size,
                    skip,
                    importer1,
                );

                let r = (
                    [history_items, history_items_page.clone()].concat(),
                    [data_items, data_items_page].concat(),
                    start_path,
                    skip + page_size,
                );

                if (history_items_page.len() as i32) < page_size {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
        }
    };

    let init_history_items: Vec<(Blake2b256Hash, ByteVector)> = Vec::new();
    let init_data_items: Vec<(Blake2b256Hash, ByteVector)> = Vec::new();
    let init_child_num: Option<Byte> = None;

    // Generate init data in space1
    for i in 0..data_size {
        let res = space1.produce(format!("ch{}", i), format!("data{}", i), false, None);
        assert!(res.is_ok());
    }

    let init_point = space1.create_checkpoint().unwrap();

    // Multipage export with skip from space1
    let init_start_path: Vec<(Blake2b256Hash, Option<Byte>)> =
        vec![(init_point.root.clone(), init_child_num)];
    let init_export_data = (init_history_items, init_data_items, init_start_path, start_skip);
    let mut export_data = init_export_data;
    loop {
        match multipage_export_with_skip(export_data, exporter1.clone(), importer1.clone()) {
            Ok(data) => {
                export_data = data;
            }
            Err(final_data) => {
                export_data = final_data;
                break;
            }
        }
    }

    let history_items = export_data.0;
    let data_items = export_data.1;

    // Import page to space2
    let _ = importer2.set_history_items(history_items);
    let _ = importer2.set_data_items(data_items);
    let _ = importer2.set_root(&init_point.root);
    let _ = space2.reset(&init_point.root);

    // Testing data in space2 (match all installed channels)
    for i in 0..data_size {
        let res = space2.consume(
            vec![format!("ch{}", i)],
            pattern.clone(),
            continuation.clone(),
            false,
            BTreeSet::new(),
        );
        assert!(res.is_ok());
    }
    let end_point = space2.create_checkpoint().unwrap();
    assert_eq!(end_point.root, RadixHistory::empty_root_node_hash())
}

async fn test_setup() -> (
    RSpace<String, Pattern, String, String>,
    Arc<dyn RSpaceExporter>,
    Arc<dyn RSpaceImporter>,
    RSpace<String, Pattern, String, String>,
    Arc<dyn RSpaceExporter>,
    Arc<dyn RSpaceImporter>,
) {
    let mut kvm = InMemoryStoreManager::new();

    let roots1 = kvm.store("roots1".to_string()).await.unwrap();
    let cold1 = kvm.store("cold1".to_string()).await.unwrap();
    let history1 = kvm.store("history1".to_string()).await.unwrap();

    let history_repository1 = Arc::new(
        HistoryRepositoryInstances::<String, Pattern, String, String>::lmdb_repository(
            roots1, cold1, history1,
        )
        .unwrap(),
    );

    let cache1: HotStoreState<String, Pattern, String, String> = HotStoreState::default();
    let history_reader = history_repository1
        .get_history_reader(&history_repository1.root())
        .unwrap();

    let store1 = {
        let hr = history_reader.base();
        HotStoreInstances::create_from_hs_and_hr(cache1, hr)
    };

    let exporter1 = history_repository1.exporter();
    let importer1 = history_repository1.importer();

    let space1 = RSpace::apply(history_repository1, store1, Arc::new(Box::new(StringMatch)));

    let roots2 = kvm.store("roots2".to_string()).await.unwrap();
    let cold2 = kvm.store("cold2".to_string()).await.unwrap();
    let history2 = kvm.store("history2".to_string()).await.unwrap();

    let history_repository2 = Arc::new(
        HistoryRepositoryInstances::<String, Pattern, String, String>::lmdb_repository(
            roots2, cold2, history2,
        )
        .unwrap(),
    );

    let cache2: HotStoreState<String, Pattern, String, String> = HotStoreState::default();
    let history_reader = history_repository2
        .get_history_reader(&history_repository2.root())
        .unwrap();

    let store2 = {
        let hr = history_reader.base();
        HotStoreInstances::create_from_hs_and_hr(cache2, hr)
    };

    let exporter2 = history_repository2.exporter();
    let importer2 = history_repository2.importer();
    
    let space2 = RSpace::apply(history_repository2, store2, Arc::new(Box::new(StringMatch)));

    (space1, exporter1, importer1, space2, exporter2, importer2)
}
