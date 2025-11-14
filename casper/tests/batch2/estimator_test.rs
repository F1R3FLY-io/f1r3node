// See casper/src/test/scala/coop/rchain/casper/batch2/EstimatorTest.scala

use std::collections::HashMap;

use crate::helper::{
    block_dag_storage_fixture::with_storage,
    block_generator::{create_block, create_genesis_block},
    block_util::generate_validator,
};
use casper::rust::estimator::Estimator;
use models::rust::casper::protocol::casper_message::Bond;

// Macro to create justifications HashMap without excessive cloning
macro_rules! justifications {
    ($($validator:expr => $block_hash:expr),* $(,)?) => {
        {
            let mut map = std::collections::HashMap::new();
            $(
                map.insert($validator.clone(), $block_hash.clone());
            )*
            map
        }
    };
}

// Helper function to reduce cloning
fn create_test_block(
    block_store: &mut block_storage::rust::key_value_block_store::KeyValueBlockStore,
    block_dag_storage: &mut block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage,
    parents: &[models::rust::block_hash::BlockHash],
    genesis: &models::rust::casper::protocol::casper_message::BlockMessage,
    creator: &models::rust::validator::Validator,
    bonds: &[Bond],
    justifications: HashMap<
        models::rust::validator::Validator,
        models::rust::block_hash::BlockHash,
    >,
) -> models::rust::casper::protocol::casper_message::BlockMessage {
    create_block(
        block_store,
        block_dag_storage,
        parents.to_vec(),
        genesis,
        Some(creator.clone()),
        Some(bonds.to_vec()),
        Some(justifications),
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

#[tokio::test]
async fn estimator_on_empty_latest_messages_should_return_the_genesis_regardless_of_dag() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v1 = generate_validator(Some("Validator One"));
        let v2 = generate_validator(Some("Validator Two"));
        let v1_bond = Bond {
            validator: v1.clone(),
            stake: 2,
        };
        let v2_bond = Bond {
            validator: v2.clone(),
            stake: 3,
        };
        let bonds = vec![v1_bond, v2_bond];

        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b2 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[genesis.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => genesis.block_hash),
        );

        let b3 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[genesis.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => genesis.block_hash),
        );

        let b4 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b2.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => b2.block_hash),
        );

        let b5 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b2.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b3.block_hash, v2 => b2.block_hash),
        );

        let _b6 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b4.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => b5.block_hash, v2 => b4.block_hash),
        );

        let b7 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b4.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b5.block_hash, v2 => b4.block_hash),
        );

        let _b8 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b7.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b7.block_hash, v2 => b4.block_hash),
        );

        let mut dag = block_dag_storage.get_representation();
        let estimator = Estimator::apply(i32::MAX, None);
        let forkchoice = estimator
            .tips_with_latest_messages(&mut dag, &genesis, HashMap::new())
            .await
            .unwrap();

        assert_eq!(forkchoice.tips[0], genesis.block_hash);
    })
    .await
}

// See https://docs.google.com/presentation/d/1znz01SF1ljriPzbMoFV0J127ryPglUYLFyhvsb-ftQk/edit?usp=sharing slide 29 for diagram
#[tokio::test]
async fn estimator_on_simple_dag_should_return_the_appropriate_score_map_and_forkchoice() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v1 = generate_validator(Some("Validator One"));
        let v2 = generate_validator(Some("Validator Two"));
        let v1_bond = Bond {
            validator: v1.clone(),
            stake: 2,
        };
        let v2_bond = Bond {
            validator: v2.clone(),
            stake: 3,
        };
        let bonds = vec![v1_bond, v2_bond];

        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b2 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[genesis.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => genesis.block_hash),
        );

        let b3 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[genesis.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => genesis.block_hash),
        );

        let b4 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b2.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => b2.block_hash),
        );

        let b5 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b2.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b3.block_hash, v2 => b2.block_hash),
        );

        let b6 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b4.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => b5.block_hash, v2 => b4.block_hash),
        );

        let b7 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b4.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b5.block_hash, v2 => b4.block_hash),
        );

        let b8 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b7.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b7.block_hash, v2 => b4.block_hash),
        );

        let mut dag = block_dag_storage.get_representation();
        let latest_blocks = HashMap::from([
            (v1.clone(), b8.block_hash.clone()),
            (v2.clone(), b6.block_hash.clone()),
        ]);

        let estimator = Estimator::apply(i32::MAX, None);
        let forkchoice = estimator
            .tips_with_latest_messages(&mut dag, &genesis, latest_blocks)
            .await
            .unwrap();

        assert_eq!(forkchoice.tips[0], b6.block_hash);
        assert_eq!(forkchoice.tips[1], b8.block_hash);
    })
    .await
}

// See [[/docs/casper/images/no_finalizable_block_mistake_with_no_disagreement_check.png]]
#[tokio::test]
async fn estimator_on_flipping_forkchoice_dag_should_return_the_appropriate_score_map_and_forkchoice(
) {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v1 = generate_validator(Some("Validator One"));
        let v2 = generate_validator(Some("Validator Two"));
        let v3 = generate_validator(Some("Validator Three"));
        let v1_bond = Bond {
            validator: v1.clone(),
            stake: 25,
        };
        let v2_bond = Bond {
            validator: v2.clone(),
            stake: 20,
        };
        let v3_bond = Bond {
            validator: v3.clone(),
            stake: 15,
        };
        let bonds = vec![v1_bond, v2_bond, v3_bond];

        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b2 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[genesis.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => genesis.block_hash, v3 => genesis.block_hash),
        );

        let b3 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[genesis.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => genesis.block_hash, v3 => genesis.block_hash),
        );

        let b4 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b2.block_hash.clone()],
            &genesis,
            &v3,
            &bonds,
            justifications!(v1 => genesis.block_hash, v2 => b2.block_hash, v3 => b2.block_hash),
        );

        let b5 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b3.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => b3.block_hash, v2 => b2.block_hash, v3 => genesis.block_hash),
        );

        let b6 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b4.block_hash.clone()],
            &genesis,
            &v1,
            &bonds,
            justifications!(v1 => b3.block_hash, v2 => b2.block_hash, v3 => b4.block_hash),
        );

        let b7 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b5.block_hash.clone()],
            &genesis,
            &v3,
            &bonds,
            justifications!(v1 => b3.block_hash, v2 => b5.block_hash, v3 => b4.block_hash),
        );

        let b8 = create_test_block(
            &mut block_store,
            &mut block_dag_storage,
            &[b6.block_hash.clone()],
            &genesis,
            &v2,
            &bonds,
            justifications!(v1 => b6.block_hash, v2 => b5.block_hash, v3 => b4.block_hash),
        );

        let mut dag = block_dag_storage.get_representation();
        let latest_blocks = HashMap::from([
            (v1.clone(), b6.block_hash.clone()),
            (v2.clone(), b8.block_hash.clone()),
            (v3.clone(), b7.block_hash.clone()),
        ]);

        let estimator = Estimator::apply(i32::MAX, None);
        let forkchoice = estimator
            .tips_with_latest_messages(&mut dag, &genesis, latest_blocks)
            .await
            .unwrap();

        assert_eq!(forkchoice.tips[0], b8.block_hash);
        assert_eq!(forkchoice.tips[1], b7.block_hash);
    })
    .await
}
