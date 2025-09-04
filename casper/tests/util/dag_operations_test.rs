// See casper/src/test/scala/coop/rchain/casper/util/DagOperationsTest.scala

use crate::helper::{
    block_dag_storage_fixture::with_storage,
    block_generator::{create_block, create_genesis_block},
};
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::util::dag_operations::DagOperations;
use models::rust::{
    block_hash::BlockHash, block_metadata::BlockMetadata,
    casper::protocol::casper_message::BlockMessage,
};
use shared::rust::dag::dag_ops;
use std::collections::{HashMap, HashSet};

#[test]
fn bf_traverse_f_should_lazily_breadth_first_traverse_a_dag_with_effectful_neighbours() {
    // Port of Scala test: val stream = DagOps.bfTraverseF[Id, Int](List(1))(i => List(i * 2, i * 3))
    // stream.take(10).toList shouldBe List(1, 2, 3, 4, 6, 9, 8, 12, 18, 27)
    //
    // Key difference: Scala's StreamT is lazy - it generates elements only when needed.
    // When .take(10) is called, the stream stops after producing exactly 10 elements.
    // Rust's bf_traverse is eager - it tries to traverse the entire graph before returning.
    // Since the graph i -> [i*2, i*3] is infinite, we need to limit neighbor generation
    // to simulate the lazy behavior and prevent infinite traversal/overflow.

    let neighbors = |i: &i32| vec![i * 2, i * 3];

    let mut count = 0;
    let result = dag_ops::bf_traverse(vec![1], |node| {
        count += 1;
        if count > 10 {
            vec![]
        } else {
            neighbors(node)
        }
    })
    .into_iter()
    .take(10)
    .collect::<Vec<_>>();

    let expected = vec![1, 2, 3, 4, 6, 9, 8, 12, 18, 27];
    assert_eq!(result, expected);
}

#[tokio::test]
async fn lowest_common_universal_ancestor_should_be_computed_properly() {
    fn create_block_with_meta(
        store: &mut KeyValueBlockStore,
        dag_store: &mut IndexedBlockDagStorage,
        genesis: &BlockMessage,
        bh: &[BlockHash],
    ) -> BlockMetadata {
        let block = create_block(
            store,
            dag_store,
            bh.to_vec(),
            genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        BlockMetadata::from_block(&block, false, None, None)
    }

    fn create_block_with_meta_and_seq(
        store: &mut KeyValueBlockStore,
        dag_store: &mut IndexedBlockDagStorage,
        genesis: &BlockMessage,
        seq_num: i32,
        bh: &[BlockHash],
    ) -> BlockMetadata {
        let block = create_block(
            store,
            dag_store,
            bh.to_vec(),
            genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(seq_num),
            None,
        );
        BlockMetadata::from_block(&block, false, None, None)
    }

    fn block_metadata_to_block_hash(metadata: &BlockMetadata) -> BlockHash {
        metadata.block_hash.clone()
    }

    with_storage(|mut block_store, mut block_dag_storage| async move {
        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        // DAG Looks like this:
        //
        //        b9   b10
        //          \ /
        //          b8
        //          / \
        //        b6   b7
        //      |  \ /  \
        //       |   b4  b5
        //       |    \ /
        //       b2    b3
        //         \  /
        //          b1
        //          |
        //         genesis

        let b1 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[genesis.block_hash.clone()],
        );

        let b2 = create_block_with_meta_and_seq(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            2,
            &[block_metadata_to_block_hash(&b1)],
        );

        let b3 = create_block_with_meta_and_seq(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            2,
            &[block_metadata_to_block_hash(&b1)],
        );

        let b4 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[block_metadata_to_block_hash(&b3)],
        );

        let b5 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[block_metadata_to_block_hash(&b3)],
        );

        let b6 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[
                block_metadata_to_block_hash(&b2),
                block_metadata_to_block_hash(&b4),
            ],
        );

        let b7 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[
                block_metadata_to_block_hash(&b4),
                block_metadata_to_block_hash(&b5),
            ],
        );

        let b8 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[
                block_metadata_to_block_hash(&b6),
                block_metadata_to_block_hash(&b7),
            ],
        );

        let b9 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[block_metadata_to_block_hash(&b8)],
        );

        let b10 = create_block_with_meta(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &[block_metadata_to_block_hash(&b8)],
        );

        let dag = block_dag_storage.get_representation();

        let result = DagOperations::lowest_universal_common_ancestor(&b1, &b5, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b2, &b3, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b3, &b2, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b6, &b7, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b2, &b2, &dag)
            .await
            .unwrap();
        assert_eq!(result, b2);

        let result = DagOperations::lowest_universal_common_ancestor(&b10, &b9, &dag)
            .await
            .unwrap();
        assert_eq!(result, b8);

        let result = DagOperations::lowest_universal_common_ancestor(&b3, &b7, &dag)
            .await
            .unwrap();
        assert_eq!(result, b3);

        let result = DagOperations::lowest_universal_common_ancestor(&b3, &b8, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b4, &b5, &dag)
            .await
            .unwrap();
        assert_eq!(result, b3);

        let result = DagOperations::lowest_universal_common_ancestor(&b4, &b6, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b7, &b7, &dag)
            .await
            .unwrap();
        assert_eq!(result, b7);

        let result = DagOperations::lowest_universal_common_ancestor(&b7, &b8, &dag)
            .await
            .unwrap();
        assert_eq!(result, b1);

        let result = DagOperations::lowest_universal_common_ancestor(&b8, &b9, &dag)
            .await
            .unwrap();
        assert_eq!(result, b8);

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .await
    .expect("Test should complete successfully");
}

#[tokio::test]
async fn uncommon_ancestors_should_be_computed_properly() {
    fn to_metadata(block: &BlockMessage) -> BlockMetadata {
        BlockMetadata::from_block(block, false, None, None)
    }

    with_storage(|mut block_store, mut block_dag_storage| async move {
        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        /*
         *  DAG Looks like this:
         *
         *         b6   b7
         *        |  \ / |
         *        b4  b5 |
         *          \ |  |
         *            b3 |
         *            |  |
         *           b1  b2
         *            |  /
         *          genesis
         */

        let b1 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b2 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b3 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b1.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b4 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b3.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b5 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b3.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b6 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b4.block_hash.clone(), b5.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b7 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone(), b5.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let dag = block_dag_storage.get_representation();

        let b1_meta = to_metadata(&b1);
        let b2_meta = to_metadata(&b2);
        let b3_meta = to_metadata(&b3);
        let b4_meta = to_metadata(&b4);
        let b5_meta = to_metadata(&b5);
        let b6_meta = to_metadata(&b6);
        let b7_meta = to_metadata(&b7);

        let result = DagOperations::uncommon_ancestors(&[b6_meta.clone(), b7_meta.clone()], &dag)
            .await
            .unwrap();

        let expected = HashMap::from([
            (b6_meta.clone(), HashSet::from_iter([0u8])),
            (b4_meta.clone(), HashSet::from_iter([0u8])),
            (b7_meta.clone(), HashSet::from_iter([1u8])),
            (b2_meta.clone(), HashSet::from_iter([1u8])),
        ]);

        assert_eq!(result, expected);

        let result = DagOperations::uncommon_ancestors(&[b6_meta.clone(), b3_meta.clone()], &dag)
            .await
            .unwrap();

        let expected = HashMap::from([
            (b6_meta.clone(), HashSet::from_iter([0u8])),
            (b4_meta.clone(), HashSet::from_iter([0u8])),
            (b5_meta.clone(), HashSet::from_iter([0u8])),
        ]);

        assert_eq!(result, expected);

        let result = DagOperations::uncommon_ancestors(
            &[b2_meta.clone(), b4_meta.clone(), b5_meta.clone()],
            &dag,
        )
        .await
        .unwrap();

        let expected = HashMap::from([
            (b2_meta.clone(), HashSet::from_iter([0u8])),
            (b4_meta.clone(), HashSet::from_iter([1u8])),
            (b5_meta.clone(), HashSet::from_iter([2u8])),
            (b3_meta.clone(), HashSet::from_iter([1u8, 2u8])),
            (b1_meta.clone(), HashSet::from_iter([1u8, 2u8])),
        ]);

        assert_eq!(result, expected);

        let result = DagOperations::uncommon_ancestors(&[b1_meta.clone()], &dag)
            .await
            .unwrap();

        let expected = HashMap::new();
        assert_eq!(result, expected);

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .await
    .expect("Test should complete successfully");
}
