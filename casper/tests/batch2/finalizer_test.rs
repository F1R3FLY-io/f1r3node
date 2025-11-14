// See casper/src/test/scala/coop/rchain/casper/batch2/FinalizerTest.scala

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::helper::{
    block_dag_storage_fixture::with_storage,
    block_generator::{create_block, create_genesis_block},
    block_util::generate_validator,
};
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::finality::finalizer::Finalizer;
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{BlockMessage, Bond},
    validator::Validator,
};

fn create_block_creator<'a>(
    bonds: &'a [Bond],
    genesis: &'a BlockMessage,
    creator: &'a Validator,
) -> impl Fn(
    &mut KeyValueBlockStore,
    &mut IndexedBlockDagStorage,
    Vec<&BlockMessage>,
    &HashMap<&Validator, &BlockMessage>,
) -> BlockMessage
       + 'a {
    move |block_store, block_dag_storage, parents, justifications| {
        let parent_hashes: Vec<BlockHash> = parents
            .iter()
            .map(|parent| parent.block_hash.clone())
            .collect();

        let justifications: HashMap<Validator, BlockHash> = justifications
            .iter()
            .map(|(validator, block_message)| {
                ((*validator).clone(), block_message.block_hash.clone())
            })
            .collect();

        create_block(
            block_store,
            block_dag_storage,
            parent_hashes,
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
}

//   *  *            b8 b9
//   *               b7         <- should not be LFB
//   *  *  *  *  *   b2 b3 b4 b5 b6
//   *               b1         <- should be LFB
//   v1 v2 v3 v4 v5
#[tokio::test]
async fn test_not_advance_finalization_if_no_new_lfb_found_advance_otherwise_invoke_all_effects() {
    with_storage(|mut store, mut dag_store| async move {
        let validators = vec![
            generate_validator(Some("Validator 1")),
            generate_validator(Some("Validator 2")),
            generate_validator(Some("Validator 3")),
            generate_validator(Some("Validator 4")),
            generate_validator(Some("Validator 5")),
        ];
        let bonds: Vec<Bond> = validators
            .iter()
            .map(|v| Bond {
                validator: v.clone(),
                stake: 3,
            })
            .collect();

        let lfb_store = Rc::new(RefCell::new(BlockHash::default()));
        let lfb_effect_invoked = Rc::new(RefCell::new(false));

        let genesis = create_genesis_block(
            &mut store,
            &mut dag_store,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let creator1 = create_block_creator(&bonds, &genesis, &validators[0]);
        let creator2 = create_block_creator(&bonds, &genesis, &validators[1]);
        let creator3 = create_block_creator(&bonds, &genesis, &validators[2]);
        let creator4 = create_block_creator(&bonds, &genesis, &validators[3]);
        let creator5 = create_block_creator(&bonds, &genesis, &validators[4]);

        let genesis_justification = HashMap::from([
            (&validators[0], &genesis),
            (&validators[1], &genesis),
            (&validators[2], &genesis),
            (&validators[3], &genesis),
            (&validators[4], &genesis),
        ]);
        let finalised_store = Rc::new(RefCell::new(HashSet::<BlockHash>::new()));

        let b1 = creator1(
            &mut store,
            &mut dag_store,
            vec![&genesis],
            &genesis_justification,
        );

        let b2 = creator1(
            &mut store,
            &mut dag_store,
            vec![&b1],
            &genesis_justification,
        );
        let b3 = creator2(
            &mut store,
            &mut dag_store,
            vec![&b1],
            &genesis_justification,
        );
        let b4 = creator3(
            &mut store,
            &mut dag_store,
            vec![&b1],
            &genesis_justification,
        );
        let b5 = creator4(
            &mut store,
            &mut dag_store,
            vec![&b1],
            &genesis_justification,
        );
        let b6 = creator5(
            &mut store,
            &mut dag_store,
            vec![&b1],
            &genesis_justification,
        );

        let dag = dag_store.get_representation();
        let _lms: Vec<(Validator, BlockHash)> = dag
            .latest_messages()
            .unwrap()
            .into_iter()
            .map(|(v, m)| (v, m.block_hash))
            .collect();
        let lfb = {
            let lfb_store = lfb_store.clone();
            Finalizer::run(&dag, -1.0, 0, move |m| {
                let lfb_store = lfb_store.clone();
                async move {
                    *lfb_store.borrow_mut() = m;
                    Ok(())
                }
            })
            .await
            .unwrap()
        };

        // check output
        assert_eq!(lfb, Some(b1.block_hash.clone()));
        // check if new LFB effect is invoked
        assert_eq!(*lfb_store.borrow(), b1.block_hash);

        let finalized_height = dag.lookup_unsafe(&lfb.unwrap()).unwrap().block_number;

        /* next layer */
        let b7 = creator1(
            &mut store,
            &mut dag_store,
            vec![&b2, &b3, &b4, &b5, &b6],
            &HashMap::from([
                (&validators[0], &b2),
                (&validators[1], &b3),
                (&validators[2], &b4),
                (&validators[3], &b4),
                (&validators[4], &b5),
            ]),
        );

        // add 2 children, this is not sufficient for finalization to advance
        creator1(
            &mut store,
            &mut dag_store,
            vec![&b7],
            &HashMap::from([
                (&validators[0], &b7),
                (&validators[1], &b3),
                (&validators[2], &b4),
                (&validators[3], &b5),
                (&validators[4], &b6),
            ]),
        );
        creator2(
            &mut store,
            &mut dag_store,
            vec![&b7],
            &HashMap::from([
                (&validators[0], &b7),
                (&validators[1], &b3),
                (&validators[2], &b4),
                (&validators[3], &b5),
                (&validators[4], &b6),
            ]),
        );

        let dag = dag_store.get_representation();
        let lfb = {
            let lfb_effect_invoked = lfb_effect_invoked.clone();
            Finalizer::run(&dag, -1.0, finalized_height, move |_m| {
                let lfb_effect_invoked = lfb_effect_invoked.clone();
                async move {
                    *lfb_effect_invoked.borrow_mut() = true;
                    Ok(())
                }
            })
            .await
            .unwrap()
        };

        // check output
        assert_eq!(lfb, None);
        // check if new LFB effect is invoked
        assert_eq!(*lfb_effect_invoked.borrow(), false);

        // add more 3 children - finalization should advance
        creator3(
            &mut store,
            &mut dag_store,
            vec![&b7],
            &HashMap::from([
                (&validators[0], &b7),
                (&validators[1], &b3),
                (&validators[2], &b4),
                (&validators[3], &b5),
                (&validators[4], &b6),
            ]),
        );
        creator4(
            &mut store,
            &mut dag_store,
            vec![&b7],
            &HashMap::from([
                (&validators[0], &b7),
                (&validators[1], &b3),
                (&validators[2], &b4),
                (&validators[3], &b5),
                (&validators[4], &b6),
            ]),
        );
        creator5(
            &mut store,
            &mut dag_store,
            vec![&b7],
            &HashMap::from([
                (&validators[0], &b7),
                (&validators[1], &b3),
                (&validators[2], &b4),
                (&validators[3], &b5),
                (&validators[4], &b6),
            ]),
        );

        let dag = dag_store.get_representation();
        let lfb = {
            let lfb_store = lfb_store.clone();
            let finalised_store = finalised_store.clone();
            Finalizer::run(&dag, -1.0, 0, move |m| {
                let lfb_store = lfb_store.clone();
                let finalised_store = finalised_store.clone();
                async move {
                    *lfb_store.borrow_mut() = m.clone();
                    finalised_store.borrow_mut().insert(m);
                    Ok(())
                }
            })
            .await
            .unwrap()
        };

        // check output
        assert_eq!(lfb, Some(b7.block_hash.clone()));
        // check if new LFB effect is invoked
        assert_eq!(*lfb_store.borrow(), b7.block_hash);

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .await
    .expect("Test should complete successfully");
}
