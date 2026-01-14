// See block-storage/src/test/scala/coop/rchain/blockstorage/dag/BlockDagStorageTest.scala
// See block-storage/src/test/scala/coop/rchain/blockstorage/dag/BlockDagKeyValueStorageTest.scala

use dashmap::{DashMap, DashSet};
use models::rust::equivocation_record::EquivocationRecord;
use once_cell::sync::Lazy;
use proptest::prelude::ProptestConfig;
use proptest::proptest;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Once};
use tokio::runtime::Runtime;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use models::rust::block_hash::BlockHash;
use models::rust::block_implicits::{
    block_element_gen, block_elements_with_parents_gen, block_hash_gen, block_with_new_hashes_gen,
    get_random_block, validator_gen,
};
use models::rust::block_metadata::BlockMetadata;
use models::rust::casper::protocol::casper_message::BlockMessage;
use models::rust::validator::Validator;
use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        let filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::DEBUG.into())
            .parse("")
            .unwrap();

        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_target(false)
                    .with_current_span(false) // logs only
                    .with_span_list(false) // logs only
                    .flatten_event(true), // put event fields at top level
            )
            .try_init()
            .unwrap();
    });
}

fn genesis_block() -> BlockMessage {
    get_random_block(
        Some(0),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vec![]),
        None,
        None,
        None,
        Some(vec![]),
        None,
        None,
    )
}

async fn create_dag_storage(genesis: &BlockMessage) -> BlockDagKeyValueStorage {
    let mut kvm = InMemoryStoreManager::new();
    let dag_storage = BlockDagKeyValueStorage::new(&mut kvm).await.unwrap();
    dag_storage.insert(genesis, false, true).unwrap();
    dag_storage
}

fn proptest_config() -> ProptestConfig {
    init_logger();

    ProptestConfig {
        cases: 5,
        max_shrink_iters: 5,
        ..ProptestConfig::default()
    }
}

static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().unwrap());

type LookupResult = (
    Vec<(
        Option<BlockMetadata>,
        Option<BlockHash>,
        Option<BlockMetadata>,
        Option<Arc<DashSet<BlockHash>>>,
        bool,
    )>,
    Arc<DashMap<Validator, BlockHash>>,
    HashMap<Validator, BlockMetadata>,
    Vec<Vec<BlockHash>>,
    i64,
);

fn lookup_elements(
    block_elements: &Vec<BlockMessage>,
    dag_storage: &BlockDagKeyValueStorage,
    topo_sort_start_block_number: Option<i64>,
) -> LookupResult {
    let topo_sort_start_block_number = topo_sort_start_block_number.unwrap_or(0);
    let dag = dag_storage.get_representation();
    let list: Vec<(
        Option<BlockMetadata>,
        Option<BlockHash>,
        Option<BlockMetadata>,
        Option<Arc<DashSet<BlockHash>>>,
        bool,
    )> = block_elements
        .iter()
        .map(|block_element| {
            let block_metadata = dag.lookup(&block_element.block_hash).unwrap();
            let latest_message_hash = dag.latest_message_hash(&block_element.sender);
            let latest_message = dag.latest_message(&block_element.sender).unwrap();
            let children = dag.children(&block_element.block_hash);
            let contains = dag.contains(&block_element.block_hash);
            (
                block_metadata,
                latest_message_hash,
                latest_message,
                children,
                contains,
            )
        })
        .collect();

    let latest_message_hashes = dag.latest_message_hashes();
    let latest_messages = dag.latest_messages().unwrap();
    let topo_sort = dag.topo_sort(topo_sort_start_block_number, None).unwrap();
    let latest_block_number = dag.latest_block_number();
    (
        list,
        latest_message_hashes,
        latest_messages,
        topo_sort,
        latest_block_number,
    )
}

fn test_lookup_elements_result(
    lookup_result: &LookupResult,
    block_elements: &Vec<BlockMessage>,
    genesis: &BlockMessage,
) {
    let (list, latest_message_hashes, latest_messages, topo_sort, latest_block_number) =
        lookup_result;

    let real_latest_messages =
        block_elements
            .iter()
            .fold(HashMap::new(), |mut acc, block_element| {
                if !block_element.sender.is_empty() {
                    acc.insert(
                        block_element.sender.clone(),
                        BlockMetadata::from_block(&block_element, false, None, None),
                    );
                }
                acc
            });

    list.iter().zip(block_elements.iter()).for_each(
        |(
            (block_metadata, latest_message_hash, latest_message, children, contains),
            block_element,
        )| {
            assert_eq!(
                *block_metadata,
                Some(BlockMetadata::from_block(&block_element, false, None, None))
            );

            assert_eq!(
                *latest_message_hash,
                real_latest_messages
                    .get(&block_element.sender)
                    .map(|metadata| metadata.block_hash.clone())
            );

            assert_eq!(
                *latest_message,
                real_latest_messages
                    .get(&block_element.sender)
                    .map(|metadata| metadata.clone())
            );

            let children_set = children.as_ref().map(|dash_set| {
                let mut set = HashSet::new();
                for item in dash_set.iter() {
                    set.insert(item.clone());
                }
                set
            });

            let expected_children: HashSet<BlockHash> = block_elements
                .iter()
                .filter(|b| {
                    b.header
                        .parents_hash_list
                        .contains(&block_element.block_hash)
                })
                .map(|b| b.block_hash.clone())
                .collect();

            assert_eq!(children_set, Some(expected_children));
            assert_eq!(*contains, true);
        },
    );

    let filtered_latest_message_hashes: HashMap<_, _> = latest_message_hashes
        .iter()
        .filter(|item| *item.value() != genesis.block_hash)
        .map(|item| (item.key().clone(), item.value().clone()))
        .collect();

    let expected_latest_message_hashes: HashMap<_, _> = real_latest_messages
        .iter()
        .map(|(validator, metadata)| (validator.clone(), metadata.block_hash.clone()))
        .collect();

    assert_eq!(
        filtered_latest_message_hashes,
        expected_latest_message_hashes
    );

    let filtered_latest_messages: HashMap<_, _> = latest_messages
        .iter()
        .filter(|(_, metadata)| metadata.block_hash != genesis.block_hash)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    assert_eq!(filtered_latest_messages, real_latest_messages);

    // Verify topo sort
    fn normalize(topo_sort: &[Vec<BlockHash>]) -> Vec<Vec<BlockHash>> {
        if topo_sort.len() == 1 && topo_sort[0].is_empty() {
            Vec::new()
        } else {
            topo_sort.to_vec()
        }
    }

    let real_topo_sort = normalize(&[block_elements
        .iter()
        .map(|b| b.block_hash.clone())
        .collect::<Vec<_>>()]);

    assert_eq!(topo_sort.len(), real_topo_sort.len());

    for (topo_sort_level, real_topo_sort_level) in topo_sort.iter().zip(real_topo_sort.iter()) {
        let topo_sort_set: HashSet<BlockHash> = topo_sort_level
            .iter()
            .filter(|&hash| *hash != genesis.block_hash)
            .cloned()
            .collect();

        let real_topo_sort_set: HashSet<BlockHash> = real_topo_sort_level.iter().cloned().collect();

        assert_eq!(topo_sort_set, real_topo_sort_set);
    }

    assert_eq!(*latest_block_number, topo_sort.len() as i64);
}

#[test]
fn dag_storage_should_be_able_to_lookup_a_stored_block() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
      let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

      for block_element in &block_elements {
        dag_storage.insert(block_element, false, false).unwrap();
      }

      let dag = dag_storage.get_representation();

      let block_element_lookups = block_elements.iter().map(|block_element| {
        let block_metadata = dag.lookup(&block_element.block_hash).unwrap();
        let latest_message_hash = dag.latest_message_hash(&block_element.sender);
        let latest_message = dag.latest_message(&block_element.sender).unwrap();
        (block_metadata, latest_message_hash, latest_message)
      });

      let latest_message_hashes = dag.latest_message_hashes();
      let latest_messages = dag.latest_messages().unwrap();

      block_element_lookups.zip(block_elements.clone()).for_each(|((block_metadata, latest_message_hash, latest_message), block_element)| {
        assert_eq!(block_metadata, Some(BlockMetadata::from_block(&block_element, false, None, None)));
        assert_eq!(latest_message_hash, Some(block_element.block_hash.clone()));
        assert_eq!(latest_message, Some(BlockMetadata::from_block(&block_element, false, None, None)));
      });

      assert_eq!(latest_message_hashes.len(), block_elements.len() + 1);
      assert_eq!(latest_messages.len(), block_elements.len() + 1);
    });
}

#[test]
fn dag_storage_should_be_able_to_handle_checking_if_contains_a_block_with_empty_hash() {
    let genesis = genesis_block();
    let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));
    let dag = dag_storage.get_representation();
    let contains = dag.contains(&prost::bytes::Bytes::new());
    assert_eq!(contains, false);
}

#[test]
fn dag_storage_should_be_able_to_restore_state_on_startup() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
      let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

      for block_element in &block_elements {
        dag_storage.insert(block_element, false, false).unwrap();
      }

      let result = lookup_elements(&block_elements, &dag_storage, None);
      test_lookup_elements_result(&result, &block_elements, &genesis);
    });
}

#[test]
fn dag_storage_should_be_able_to_restore_latest_messages_with_genesis_with_empty_sender_field() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
      let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

      let mut block_elements_with_genesis = block_elements.clone();
      if let Some(first) = block_elements_with_genesis.first_mut() {
          first.sender = prost::bytes::Bytes::new();
      }

      for block_element in &block_elements_with_genesis {
        dag_storage.insert(block_element, false, false).unwrap();
      }

      let result = lookup_elements(&block_elements_with_genesis, &dag_storage, None);
      test_lookup_elements_result(&result, &block_elements_with_genesis, &genesis);
    });
}

#[test]
fn dag_storage_should_be_able_to_restore_state_from_the_previous_two_instances() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(first_block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10),
      second_block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
        let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

        for block_element in &first_block_elements {
            dag_storage.insert(block_element, false, false).unwrap();
        }

        for block_element in &second_block_elements {
            dag_storage.insert(block_element, false, false).unwrap();
        }

        let mut all_block_elements = first_block_elements.clone();
        all_block_elements.extend(second_block_elements.clone());
        let result = lookup_elements(&all_block_elements, &dag_storage, None);
        test_lookup_elements_result(&result, &all_block_elements, &genesis);
    });
}

#[test]
fn dag_storage_should_be_able_to_restore_after_squashing_latest_messages() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
        proptest!(proptest_config(), |(
            second_block_elements in block_with_new_hashes_gen(block_elements.clone()),
            third_block_elements in block_with_new_hashes_gen(block_elements.clone())
        )| {
            let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

            for block_element in &block_elements {
                dag_storage.insert(block_element, false, false).unwrap();
            }

            for block_element in &second_block_elements {
                dag_storage.insert(block_element, false, false).unwrap();
            }

            for block_element in &third_block_elements {
                dag_storage.insert(block_element, false, false).unwrap();
            }

            let mut all_block_elements = block_elements.clone();
            all_block_elements.extend(second_block_elements.clone());
            all_block_elements.extend(third_block_elements.clone());

            let result = lookup_elements(&block_elements, &dag_storage, None);
            test_lookup_elements_result(&result, &all_block_elements, &genesis);
        });
    });
}

#[test]
fn dag_storage_should_be_able_to_restore_equivocations_tracker_on_startup() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10),
      equivocator in validator_gen(), block_hash in block_hash_gen())| {
        let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

        for block_element in &block_elements {
            dag_storage.insert(block_element, false, false).unwrap();
        }

        let equivocation_record = EquivocationRecord::new(equivocator, 0, BTreeSet::from([block_hash]));
        dag_storage.insert_equivocation_record(equivocation_record.clone()).unwrap();

        let records = dag_storage.equivocation_records().unwrap();
        assert_eq!(records, HashSet::from([equivocation_record]));

        let result = lookup_elements(&block_elements, &dag_storage, None);
        test_lookup_elements_result(&result, &block_elements, &genesis);

    });
}

#[test]
fn dag_storage_should_be_able_to_modify_equivocation_records() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(equivocator in validator_gen(), block_hash1 in block_hash_gen(),
      block_hash2 in block_hash_gen())| {
        let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

        let equivocation_record = EquivocationRecord::new(equivocator.clone(), 0, BTreeSet::from([block_hash1.clone()]));
        dag_storage.insert_equivocation_record(equivocation_record.clone()).unwrap();

        dag_storage.update_equivocation_record(equivocation_record, block_hash2.clone()).unwrap();

        let updated_equivocation_record = EquivocationRecord::new(equivocator, 0, BTreeSet::from([block_hash1, block_hash2]));
        let records = dag_storage.equivocation_records().unwrap();
        assert_eq!(records, HashSet::from([updated_equivocation_record]));
    });
}

#[test]
fn dag_storage_should_be_able_to_restore_invalid_blocks_on_startup() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
      let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

      for block_element in &block_elements {
        dag_storage.insert(block_element, true, false).unwrap();
      }

      let dag = dag_storage.get_representation();
      let invalid_blocks = dag.invalid_blocks();
      let invalid_blocks_set: HashSet<_> = invalid_blocks.iter().map(|item| item.clone()).collect();
      assert_eq!(invalid_blocks_set, block_elements.into_iter().map(|b| BlockMetadata::from_block(&b, true, None, None)).collect::<HashSet<_>>());
    });
}

#[test]
fn dag_storage_should_be_able_to_restore_deploy_index_on_startup() {
    let genesis = genesis_block();
    proptest!(proptest_config(), |(block_elements in block_elements_with_parents_gen(genesis.clone(), 0, 10))| {
      let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));

      for block_element in &block_elements {
        dag_storage.insert(block_element, true, false).unwrap();
      }

      let dag = dag_storage.get_representation();
      let mut deploy_sigs = Vec::new();
      let mut block_hashes = Vec::new();

      for block in &block_elements {
          for deploy in &block.body.deploys {
              deploy_sigs.push(deploy.deploy.sig.clone());
              block_hashes.push(block.block_hash.clone());
          }
      }

      let deploy_lookups: Vec<Option<BlockHash>> = deploy_sigs
          .iter()
          .map(|sig| dag.lookup_by_deploy_id(&sig.to_vec()).unwrap())
          .collect();

      assert_eq!(deploy_lookups, block_hashes.iter().map(|h| Some(h.clone())).collect::<Vec<_>>());
    });
}

#[test]
fn dag_storage_should_be_able_to_handle_blocks_with_invalid_numbers() {
    proptest!(proptest_config(), |(genesis in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None),
      block in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None))| {
        let dag_storage = RUNTIME.block_on(create_dag_storage(&genesis));
        let mut invalid_block = block.clone();
        invalid_block.body.state.block_number = 1000;
        dag_storage.insert(&genesis, false, false).unwrap();
        dag_storage.insert(&invalid_block, true, false).unwrap();
    });
}

#[tokio::test]
async fn recording_of_new_directly_finalized_block_should_record_finalized_all_non_finalized_ancestors_of_lfb(
) {
    let genesis = genesis_block();
    let dag_storage = create_dag_storage(&genesis).await;
    dag_storage.insert(&genesis, false, true).unwrap();

    let b1 = get_random_block(
        Some(1),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vec![genesis.block_hash.clone()]),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    dag_storage.insert(&b1, false, false).unwrap();

    let b2 = get_random_block(
        Some(2),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vec![b1.block_hash.clone()]),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    dag_storage.insert(&b2, false, false).unwrap();

    let b3 = get_random_block(
        Some(3),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vec![b2.block_hash.clone()]),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    dag_storage.insert(&b3, false, false).unwrap();

    let b4 = get_random_block(
        Some(4),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vec![b3.block_hash.clone()]),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let dag = dag_storage.insert(&b4, false, false).unwrap();

    assert_eq!(
        dag.lookup_unsafe(&genesis.block_hash).unwrap().finalized,
        true
    );
    assert_eq!(dag.is_finalized(&genesis.block_hash), true);

    assert_eq!(dag.lookup_unsafe(&b1.block_hash).unwrap().finalized, false);
    assert_eq!(dag.is_finalized(&b1.block_hash), false);

    assert_eq!(dag.lookup_unsafe(&b2.block_hash).unwrap().finalized, false);
    assert_eq!(dag.is_finalized(&b2.block_hash), false);

    assert_eq!(dag.lookup_unsafe(&b3.block_hash).unwrap().finalized, false);
    assert_eq!(dag.is_finalized(&b3.block_hash), false);

    assert_eq!(dag.lookup_unsafe(&b4.block_hash).unwrap().finalized, false);
    assert_eq!(dag.is_finalized(&b4.block_hash), false);

    let effects = std::sync::Arc::new(std::sync::Mutex::new(HashSet::new()));
    let effects_clone = effects.clone();
    dag_storage
        .record_directly_finalized(b3.block_hash.clone(), move |blocks: &HashSet<BlockHash>| {
            let blocks = blocks.clone();
            let effects_clone = effects_clone.clone();
            Box::pin(async move {
                let mut effects_guard = effects_clone.lock().unwrap();
                effects_guard.extend(blocks.iter().cloned());
                Ok(())
            })
        })
        .await
        .unwrap();

    let dag = dag_storage.get_representation();
    assert_eq!(dag.last_finalized_block(), b3.block_hash);
    assert_eq!(dag.is_finalized(&b1.block_hash), true);
    assert_eq!(dag.is_finalized(&b2.block_hash), true);
    assert_eq!(dag.is_finalized(&b3.block_hash), true);
    assert_eq!(dag.is_finalized(&b4.block_hash), false);

    let b1_meta = dag.lookup_unsafe(&b1.block_hash).unwrap();
    assert_eq!(b1_meta.finalized, true);
    assert_eq!(b1_meta.directly_finalized, false);

    let b2_meta = dag.lookup_unsafe(&b2.block_hash).unwrap();
    assert_eq!(b2_meta.finalized, true);
    assert_eq!(b2_meta.directly_finalized, false);

    let b3_meta = dag.lookup_unsafe(&b3.block_hash).unwrap();
    assert_eq!(b3_meta.finalized, true);
    assert_eq!(b3_meta.directly_finalized, true);

    let b4_meta = dag.lookup_unsafe(&b4.block_hash).unwrap();
    assert_eq!(b4_meta.finalized, false);
    assert_eq!(b4_meta.directly_finalized, false);

    // Check that all finalized blocks were captured in the effects
    let finalized_effects = effects.lock().unwrap();
    let expected_effects = HashSet::from([
        b1.block_hash.clone(),
        b2.block_hash.clone(),
        b3.block_hash.clone(),
    ]);
    assert_eq!(*finalized_effects, expected_effects);
}
