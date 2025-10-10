use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use casper::rust::api::graph_generator::*;
use graphz::rust::graphz::{GraphStyle, StringSerializer};
use hex;
use models::rust::casper::protocol::casper_message::{
    BlockMessage, Body, F1r3flyState, Header, Justification,
};
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::shared::in_mem_key_value_store::InMemoryKeyValueStore;
use std::collections::HashMap;
use std::sync::Arc;
use tokio;

// Helper function to create a mock BlockStore with test data
fn create_mock_block_store() -> KeyValueBlockStore {
    let store = InMemoryKeyValueStore::new();
    let mut block_store =
        KeyValueBlockStore::new(Arc::new(store), Arc::new(InMemoryKeyValueStore::new()));

    // Create mock blocks for testing
    let block1 = BlockMessage {
        block_hash: Bytes::from("block1"),
        header: Header {
            parents_hash_list: vec![],
            timestamp: 0,
            version: 1,
            extra_bytes: Bytes::new(),
        },
        body: Body {
            state: F1r3flyState {
                pre_state_hash: Bytes::new(),
                post_state_hash: Bytes::new(),
                bonds: vec![],
                block_number: 1,
            },
            deploys: vec![],
            rejected_deploys: vec![],
            system_deploys: vec![],
            extra_bytes: Bytes::new(),
        },
        justifications: vec![],
        sender: Bytes::from("alice"),
        seq_num: 1,
        sig: Bytes::new(),
        sig_algorithm: String::new(),
        shard_id: String::new(),
        extra_bytes: Bytes::new(),
    };

    let block2 = BlockMessage {
        block_hash: Bytes::from("block2"),
        header: Header {
            parents_hash_list: vec![],
            timestamp: 0,
            version: 1,
            extra_bytes: Bytes::new(),
        },
        body: Body {
            state: F1r3flyState {
                pre_state_hash: Bytes::new(),
                post_state_hash: Bytes::new(),
                bonds: vec![],
                block_number: 1,
            },
            deploys: vec![],
            rejected_deploys: vec![],
            system_deploys: vec![],
            extra_bytes: Bytes::new(),
        },
        justifications: vec![],
        sender: Bytes::from("bob"),
        seq_num: 1,
        sig: Bytes::new(),
        sig_algorithm: String::new(),
        shard_id: String::new(),
        extra_bytes: Bytes::new(),
    };

    let block3 = BlockMessage {
        block_hash: Bytes::from("block3"),
        header: Header {
            parents_hash_list: vec![Bytes::from("block1")],
            timestamp: 0,
            version: 1,
            extra_bytes: Bytes::new(),
        },
        body: Body {
            state: F1r3flyState {
                pre_state_hash: Bytes::new(),
                post_state_hash: Bytes::new(),
                bonds: vec![],
                block_number: 2,
            },
            deploys: vec![],
            rejected_deploys: vec![],
            system_deploys: vec![],
            extra_bytes: Bytes::new(),
        },
        justifications: vec![Justification {
            validator: Bytes::from("validator1"),
            latest_block_hash: Bytes::from("block1"),
        }],
        sender: Bytes::from("charlie"),
        seq_num: 1,
        sig: Bytes::new(),
        sig_algorithm: String::new(),
        shard_id: String::new(),
        extra_bytes: Bytes::new(),
    };

    // Add blocks to store
    block_store.put_block_message(&block1).unwrap();
    block_store.put_block_message(&block2).unwrap();
    block_store.put_block_message(&block3).unwrap();

    block_store
}

#[tokio::test]
async fn dag_as_cluster_basic() {
    let block_hash1 = Bytes::from("block1");
    let block_hash2 = Bytes::from("block2");
    let block_hash3 = Bytes::from("block3");

    let topo_sort = vec![vec![block_hash1, block_hash2], vec![block_hash3]];

    let last_finalized_block_hash = hex::encode(b"block1");
    let config = GraphConfig::default();
    let serializer = Arc::new(StringSerializer::new());
    let mut block_store = create_mock_block_store();

    let result = GraphzGenerator::dag_as_cluster(
        topo_sort,
        last_finalized_block_hash,
        config,
        serializer.clone(),
        &mut block_store,
    )
    .await;

    assert!(result.is_ok());

    let content = serializer.get_content().await;
    assert!(content.contains("digraph"));
    assert!(content.contains("dag"));
    assert!(content.contains("}"));
    assert!(content.contains("rankdir=BT"));
    assert!(content.contains("splines=false"));
}

#[tokio::test]
async fn dag_as_cluster_with_justifications() {
    let block_hash1 = Bytes::from("block1");
    let block_hash2 = Bytes::from("block2");

    let topo_sort = vec![vec![block_hash1, block_hash2]];

    let last_finalized_block_hash = hex::encode(b"block1");
    let config = GraphConfig {
        show_justification_lines: true,
    };
    let serializer = Arc::new(StringSerializer::new());
    let mut block_store = create_mock_block_store();

    let result = GraphzGenerator::dag_as_cluster(
        topo_sort,
        last_finalized_block_hash,
        config,
        serializer.clone(),
        &mut block_store,
    )
    .await;

    assert!(result.is_ok());

    let content = serializer.get_content().await;
    assert!(content.contains("digraph"));
    assert!(content.contains("dag"));
}

#[tokio::test]
async fn nodes_for_ts() {
    let mut blocks = HashMap::new();
    let validator_blocks = vec![ValidatorBlock {
        block_hash: "block1".to_string(),
        parents: vec!["parent1".to_string()],
        justifications: vec!["just1".to_string()],
    }];
    blocks.insert(100i64, validator_blocks);

    let nodes = GraphzGenerator::nodes_for_ts("val1", 100, &blocks, "block1");
    assert_eq!(nodes.len(), 1);
    assert!(nodes.contains_key("block1"));
    assert_eq!(nodes.get("block1"), Some(&Some(GraphStyle::Filled)));

    let nodes = GraphzGenerator::nodes_for_ts("val1", 200, &blocks, "block1");
    assert_eq!(nodes.len(), 1);
    assert!(nodes.contains_key("200_val1"));
    assert_eq!(nodes.get("200_val1"), Some(&Some(GraphStyle::Invis)));
}

#[tokio::test]
async fn accumulate_dag_info() {
    let acc = DagInfo::empty();
    let block_hash1 = Bytes::from("block1");
    let block_hash2 = Bytes::from("block2");
    let block_hashes = vec![block_hash1, block_hash2];
    let mut block_store = create_mock_block_store();

    let result = GraphzGenerator::accumulate_dag_info(acc, block_hashes, &mut block_store).await;
    assert!(result.is_ok());

    let dag_info = result.unwrap();
    assert_eq!(dag_info.timeseries.len(), 1);
    assert_eq!(dag_info.validators.len(), 2);

    assert!(dag_info.validators.contains_key("616c696365")); // alice
    assert!(dag_info.validators.contains_key("626f62")); // bob
}

#[tokio::test]
async fn init_graph() {
    let serializer = Arc::new(StringSerializer::new());
    let result = GraphzGenerator::init_graph("test_graph", serializer.clone()).await;

    assert!(result.is_ok());

    let content = serializer.get_content().await;
    assert!(content.contains("digraph"));
    assert!(content.contains("test_graph"));
    assert!(content.contains("rankdir=BT"));
    assert!(content.contains("splines=false"));
    assert!(content.contains("node ["));
    assert!(content.contains("width=0"));
    assert!(content.contains("height=0"));
    assert!(content.contains("margin=0.03"));
    assert!(content.contains("fontsize=8"));
}

#[tokio::test]
async fn draw_parent_dependencies() {
    let serializer = Arc::new(StringSerializer::new());
    let g = GraphzGenerator::init_graph("test", serializer.clone())
        .await
        .unwrap();

    let mut blocks1 = HashMap::new();
    let validator_blocks1 = vec![ValidatorBlock {
        block_hash: "block1".to_string(),
        parents: vec!["parent1".to_string()],
        justifications: vec![],
    }];
    blocks1.insert(100i64, validator_blocks1);

    let mut blocks2 = HashMap::new();
    let validator_blocks2 = vec![ValidatorBlock {
        block_hash: "block2".to_string(),
        parents: vec!["parent2".to_string()],
        justifications: vec![],
    }];
    blocks2.insert(100i64, validator_blocks2);

    let validators = vec![blocks1, blocks2];

    let result = GraphzGenerator::draw_parent_dependencies(&g, &validators).await;
    assert!(result.is_ok());

    let content = serializer.get_content().await;
    assert!(content.contains("block1"));
    assert!(content.contains("block2"));
}

#[tokio::test]
async fn draw_justification_dotted_lines() {
    let serializer = Arc::new(StringSerializer::new());
    let g = GraphzGenerator::init_graph("test", serializer.clone())
        .await
        .unwrap();

    let mut validators = HashMap::new();
    let mut blocks = HashMap::new();
    let validator_blocks = vec![ValidatorBlock {
        block_hash: "block1".to_string(),
        parents: vec![],
        justifications: vec!["just1".to_string()],
    }];
    blocks.insert(100i64, validator_blocks);
    validators.insert("validator1".to_string(), blocks);

    let result = GraphzGenerator::draw_justification_dotted_lines(&g, &validators).await;
    assert!(result.is_ok());

    let content = serializer.get_content().await;
    assert!(content.contains("block1"));
    assert!(content.contains("just1"));
}

#[tokio::test]
async fn style_for() {
    let style = GraphzGenerator::style_for("block1", "block1");
    assert_eq!(style, Some(GraphStyle::Filled));

    let style = GraphzGenerator::style_for("block2", "block1");
    assert_eq!(style, None);
}
