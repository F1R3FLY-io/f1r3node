// See casper/src/test/scala/coop/rchain/casper/util/CasperUtilTest.scala

use casper::rust::test_utils::helper::{
    block_dag_storage_fixture::with_storage,
    block_generator::{create_block_fast, create_block_fast_with_creator, create_genesis_block},
    block_util::generate_validator,
};

#[tokio::test]
async fn is_in_main_chain_should_classify_appropriately() {
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

        let b2 = create_block_fast(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
        );

        let b3 = create_block_fast(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone()],
            &genesis,
        );

        let dag = block_dag_storage.get_representation();

        assert!(dag
            .is_in_main_chain(&genesis.block_hash, &b3.block_hash)
            .unwrap());
        assert!(dag
            .is_in_main_chain(&b2.block_hash, &b3.block_hash)
            .unwrap());
        assert!(!dag
            .is_in_main_chain(&b3.block_hash, &b2.block_hash)
            .unwrap());
        assert!(!dag
            .is_in_main_chain(&b3.block_hash, &genesis.block_hash)
            .unwrap());
    })
    .await
}

#[tokio::test]
async fn is_in_main_chain_should_classify_diamond_dags_appropriately() {
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

        let b2 = create_block_fast(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
        );

        let b3 = create_block_fast(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
        );
        let b4 = create_block_fast(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone(), b3.block_hash.clone()],
            &genesis,
        );

        let dag = block_dag_storage.get_representation();

        assert!(dag
            .is_in_main_chain(&genesis.block_hash, &b2.block_hash)
            .unwrap());
        assert!(dag
            .is_in_main_chain(&genesis.block_hash, &b3.block_hash)
            .unwrap());
        assert!(dag
            .is_in_main_chain(&genesis.block_hash, &b4.block_hash)
            .unwrap());
        assert!(dag
            .is_in_main_chain(&b2.block_hash, &b4.block_hash)
            .unwrap());
        assert_ne!(
            dag.is_in_main_chain(&b3.block_hash, &b4.block_hash)
                .unwrap(),
            true
        );
    })
    .await
}

// See https://docs.google.com/presentation/d/1znz01SF1ljriPzbMoFV0J127ryPglUYLFyhvsb-ftQk/edit?usp=sharing slide 29 for diagram
#[tokio::test]
async fn is_in_main_chain_should_classify_complicated_chains_appropriately() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v1 = generate_validator(Some("Validator One"));
        let v2 = generate_validator(Some("Validator Two"));

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

        let b2 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            v2.clone(),
        );

        let b3 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            v1.clone(),
        );

        let b4 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone()],
            &genesis,
            v2.clone(),
        );

        let b5 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone()],
            &genesis,
            v1.clone(),
        );

        let b6 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![b4.block_hash.clone()],
            &genesis,
            v2.clone(),
        );

        let b7 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![b4.block_hash.clone()],
            &genesis,
            v1.clone(),
        );

        let b8 = create_block_fast_with_creator(
            &mut block_store,
            &mut block_dag_storage,
            vec![b7.block_hash.clone()],
            &genesis,
            v1.clone(),
        );

        let dag = block_dag_storage.get_representation();

        assert!(dag
            .is_in_main_chain(&genesis.block_hash, &b2.block_hash)
            .unwrap());
        assert_ne!(
            dag.is_in_main_chain(&b2.block_hash, &b3.block_hash)
                .unwrap(),
            true
        );
        assert_ne!(
            dag.is_in_main_chain(&b3.block_hash, &b4.block_hash)
                .unwrap(),
            true
        );
        assert_ne!(
            dag.is_in_main_chain(&b4.block_hash, &b5.block_hash)
                .unwrap(),
            true
        );
        assert_ne!(
            dag.is_in_main_chain(&b5.block_hash, &b6.block_hash)
                .unwrap(),
            true
        );
        assert_ne!(
            dag.is_in_main_chain(&b6.block_hash, &b7.block_hash)
                .unwrap(),
            true
        );
        assert!(dag
            .is_in_main_chain(&b7.block_hash, &b8.block_hash)
            .unwrap());
        assert!(dag
            .is_in_main_chain(&b2.block_hash, &b6.block_hash)
            .unwrap());
        assert!(dag
            .is_in_main_chain(&b2.block_hash, &b8.block_hash)
            .unwrap());
        assert_ne!(
            dag.is_in_main_chain(&b4.block_hash, &b2.block_hash)
                .unwrap(),
            true
        );
    })
    .await
}
