// See casper/src/test/scala/coop/rchain/casper/batch2/CliqueOracleTest.scala

use casper::rust::test_utils::helper::{
    block_dag_storage_fixture::with_storage, block_generator::create_genesis_block,
    block_util::generate_validator,
};
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::safety_oracle::{CliqueOracleImpl, SafetyOracle};
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{BlockMessage, Bond},
    validator::Validator,
};
use std::collections::HashMap;

fn create_block<'a>(
    bonds: &'a [Bond],
    genesis: &'a BlockMessage,
    creator: &'a Validator,
) -> impl Fn(
    &mut KeyValueBlockStore,
    &mut IndexedBlockDagStorage,
    &BlockMessage,
    &HashMap<&Validator, &BlockMessage>,
) -> BlockMessage
       + 'a {
    move |block_store, block_dag_storage, parent, justifications| {
        let justifications_map: HashMap<Validator, BlockHash> = justifications
            .iter()
            .map(|(validator, block_message)| {
                ((*validator).clone(), block_message.block_hash.clone())
            })
            .collect();

        casper::rust::test_utils::helper::block_generator::create_block(
            block_store,
            block_dag_storage,
            vec![parent.block_hash.clone()],
            genesis,
            Some(creator.clone()),
            Some(bonds.to_vec()),
            Some(justifications_map),
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }
}

// See [[/docs/casper/images/cbc-casper_ping_pong_diagram.png]]
/**
*       *     b8
*       |
*   *   *     b6 b7
*   | /
*   *   *     b4 b5
*   | /
*   *   *     b2 b3
*    \ /
*     *
*   c2 c1
*/
#[tokio::test]
async fn clique_oracle_should_detect_finality_as_appropriate() {
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

        let creator1 = create_block(&bonds, &genesis, &v1);
        let creator2 = create_block(&bonds, &genesis, &v2);

        let genesis_justification = HashMap::from([(&v1, &genesis), (&v2, &genesis)]);

        let b2 = creator2(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &genesis_justification,
        );

        let b3 = creator1(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &genesis_justification,
        );

        let b4 = creator2(
            &mut block_store,
            &mut block_dag_storage,
            &b2,
            &HashMap::from([(&v1, &genesis), (&v2, &b2)]),
        );

        let b5 = creator1(
            &mut block_store,
            &mut block_dag_storage,
            &b2,
            &HashMap::from([(&v1, &b3), (&v2, &b2)]),
        );

        let _b6 = creator2(
            &mut block_store,
            &mut block_dag_storage,
            &b4,
            &HashMap::from([(&v1, &b5), (&v2, &b4)]),
        );

        let b7 = creator1(
            &mut block_store,
            &mut block_dag_storage,
            &b4,
            &HashMap::from([(&v1, &b5), (&v2, &b4)]),
        );

        let _b8 = creator1(
            &mut block_store,
            &mut block_dag_storage,
            &b7,
            &HashMap::from([(&v1, &b7), (&v2, &b4)]),
        );

        let dag = block_dag_storage.get_representation();
        let safety_oracle = CliqueOracleImpl;

        let genesis_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &genesis.block_hash)
            .await
            .unwrap();
        assert!((genesis_fault_tolerance - 1.0).abs() < 0.01);

        let b2_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &b2.block_hash)
            .await
            .unwrap();
        assert!((b2_fault_tolerance - 1.0).abs() < 0.01);

        let b3_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &b3.block_hash)
            .await
            .unwrap();
        assert!((b3_fault_tolerance - (-1.0)).abs() < 0.01);

        let b4_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &b4.block_hash)
            .await
            .unwrap();
        assert!((b4_fault_tolerance - 0.2).abs() < 0.01);
    })
    .await
}

// See [[/docs/casper/images/no_finalizable_block_mistake_with_no_disagreement_check.png]]
#[tokio::test]
async fn clique_oracle_should_detect_possible_disagreements_appropriately() {
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

        let creator1 = create_block(&bonds, &genesis, &v1);
        let creator2 = create_block(&bonds, &genesis, &v2);
        let creator3 = create_block(&bonds, &genesis, &v3);

        let genesis_justification =
            HashMap::from([(&v1, &genesis), (&v2, &genesis), (&v3, &genesis)]);

        let b2 = creator2(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &genesis_justification,
        );

        let b3 = creator1(
            &mut block_store,
            &mut block_dag_storage,
            &genesis,
            &genesis_justification,
        );

        let b4 = creator3(
            &mut block_store,
            &mut block_dag_storage,
            &b2,
            &HashMap::from([(&v1, &genesis), (&v2, &b2), (&v3, &b2)]),
        );

        let b5 = creator2(
            &mut block_store,
            &mut block_dag_storage,
            &b3,
            &HashMap::from([(&v1, &b3), (&v2, &b2), (&v3, &genesis)]),
        );

        let b6 = creator1(
            &mut block_store,
            &mut block_dag_storage,
            &b4,
            &HashMap::from([(&v1, &b3), (&v2, &b2), (&v3, &b4)]),
        );

        let _b7 = creator3(
            &mut block_store,
            &mut block_dag_storage,
            &b5,
            &HashMap::from([(&v1, &b3), (&v2, &b5), (&v3, &b4)]),
        );

        let _b8 = creator2(
            &mut block_store,
            &mut block_dag_storage,
            &b6,
            &HashMap::from([(&v1, &b6), (&v2, &b5), (&v3, &b4)]),
        );

        let dag = block_dag_storage.get_representation();
        let safety_oracle = CliqueOracleImpl;

        let genesis_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &genesis.block_hash)
            .await
            .unwrap();
        assert!((genesis_fault_tolerance - 1.0).abs() < 0.01);

        let b2_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &b2.block_hash)
            .await
            .unwrap();
        assert!((b2_fault_tolerance - (-1.0 / 6.0)).abs() < 0.01);

        let b3_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &b3.block_hash)
            .await
            .unwrap();
        assert!((b3_fault_tolerance - (-1.0)).abs() < 0.01);

        let b4_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &b4.block_hash)
            .await
            .unwrap();
        assert!((b4_fault_tolerance - (-1.0 / 6.0)).abs() < 0.01);
    })
    .await
}

// See [[/docs/casper/images/no_majority_fork_safe_after_union.png]]
#[tokio::test]
async fn clique_oracle_should_identify_no_majority_fork_safe_after_union() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator Zero"));
        let v1 = generate_validator(Some("Validator One"));
        let v2 = generate_validator(Some("Validator Two"));
        let v3 = generate_validator(Some("Validator Three"));
        let v4 = generate_validator(Some("Validator Four"));
        let bonds = vec![
            Bond {
                validator: v0.clone(),
                stake: 500,
            },
            Bond {
                validator: v1.clone(),
                stake: 450,
            },
            Bond {
                validator: v2.clone(),
                stake: 600,
            },
            Bond {
                validator: v3.clone(),
                stake: 400,
            },
            Bond {
                validator: v4.clone(),
                stake: 525,
            },
        ];

        /*
        # create right hand side of fork and check for no safety
        'M-2-A SJ-1-A M-1-L0 SJ-0-L0 M-0-L1 SJ-1-L1 M-1-L2 SJ-0-L2 '
        'M-0-L3 SJ-1-L3 M-1-L4 SJ-0-L4 '
        # now, left hand side as well. should still have no safety
        'SJ-3-A M-3-R0 SJ-4-R0 M-4-R1 SJ-3-R1 M-3-R2 SJ-4-R2 M-4-R3 '
        'SJ-3-R3 M-3-R4 SJ-4-R4'
        */

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

        let creator0 = create_block(&bonds, &genesis, &v0);
        let creator1 = create_block(&bonds, &genesis, &v1);
        let creator2 = create_block(&bonds, &genesis, &v2);
        let creator3 = create_block(&bonds, &genesis, &v3);
        let creator4 = create_block(&bonds, &genesis, &v4);

        let mut gj_l = HashMap::from([
            (&v0, &genesis),
            (&v1, &genesis),
            (&v2, &genesis),
            (&v3, &genesis),
            (&v4, &genesis),
        ]);

        /*
         create left hand side of fork and check for no safety
        'M-2-A SJ-1-A M-1-L0 SJ-0-L0 M-0-L1 SJ-1-L1 M-1-L2 SJ-0-L2 '
        'M-0-L3 SJ-1-L3 M-1-L4 SJ-0-L4 '
         */
        let a = creator2(&mut block_store, &mut block_dag_storage, &genesis, &gj_l);

        gj_l.insert(&v2, &a);
        let l0 = creator1(&mut block_store, &mut block_dag_storage, &a, &gj_l);

        gj_l.insert(&v1, &l0);
        let l1 = creator0(&mut block_store, &mut block_dag_storage, &l0, &gj_l);

        gj_l.insert(&v0, &l1);
        let l2 = creator1(&mut block_store, &mut block_dag_storage, &l1, &gj_l);

        gj_l.insert(&v1, &l2);
        let l3 = creator0(&mut block_store, &mut block_dag_storage, &l2, &gj_l);

        gj_l.insert(&v0, &l3);
        let l4 = creator1(&mut block_store, &mut block_dag_storage, &l3, &gj_l);

        let mut gj_r = HashMap::from([
            (&v0, &genesis),
            (&v1, &genesis),
            (&v2, &genesis),
            (&v3, &genesis),
            (&v4, &genesis),
        ]);

        /*
         now, right hand side as well. should still have no safety
        'SJ-3-A M-3-R0 SJ-4-R0 M-4-R1 SJ-3-R1 M-3-R2 SJ-4-R2 M-4-R3 '
        'SJ-3-R3 M-3-R4 SJ-4-R4'
         */
        gj_r.insert(&v2, &a);
        let r0 = creator3(&mut block_store, &mut block_dag_storage, &a, &gj_r);

        gj_r.insert(&v3, &r0);
        let r1 = creator4(&mut block_store, &mut block_dag_storage, &r0, &gj_r);

        gj_r.insert(&v4, &r1);
        let r2 = creator3(&mut block_store, &mut block_dag_storage, &r1, &gj_r);

        gj_r.insert(&v3, &r2);
        let r3 = creator4(&mut block_store, &mut block_dag_storage, &r2, &gj_r);

        gj_r.insert(&v4, &r3);
        let r4 = creator3(&mut block_store, &mut block_dag_storage, &r3, &gj_r);

        let dag = block_dag_storage.get_representation();
        let safety_oracle = CliqueOracleImpl;

        let l0_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &l0.block_hash)
            .await
            .unwrap();
        assert!((l0_fault_tolerance - (-1.0)).abs() < 0.01);

        let r0_fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag, &r0.block_hash)
            .await
            .unwrap();
        assert!((r0_fault_tolerance - (-1.0)).abs() < 0.01);

        /*
         show all validators all messages
        'SJ-0-R4 SJ-1-R4 SJ-2-R4 SJ-2-L4 SJ-3-L4 SJ-4-L4 '
         */
        let mut aj = HashMap::from([(&v0, &l3), (&v1, &l4), (&v2, &a), (&v3, &r4), (&v4, &r3)]);

        /*
         two rounds of round robin, check have safety on the correct fork
        'M-0-J0 SJ-1-J0 M-1-J1 SJ-2-J1 M-2-J2 SJ-3-J2 M-3-J3 SJ-4-J3 M-4-J4 SJ-0-J4 '
        'M-0-J01 SJ-1-J01 M-1-J11 SJ-2-J11 M-2-J21 SJ-3-J21 M-3-J31 SJ-4-J31 M-4-J41 SJ-0-J41'
         */

        let j0 = creator0(&mut block_store, &mut block_dag_storage, &l4, &aj);

        aj.insert(&v0, &j0);
        let j1 = creator1(&mut block_store, &mut block_dag_storage, &j0, &aj);

        aj.insert(&v1, &j1);
        let j2 = creator2(&mut block_store, &mut block_dag_storage, &j1, &aj);

        aj.insert(&v2, &j2);
        let j3 = creator3(&mut block_store, &mut block_dag_storage, &j2, &aj);

        aj.insert(&v3, &j3);
        let j4 = creator4(&mut block_store, &mut block_dag_storage, &j3, &aj);

        aj.insert(&v4, &j4);
        let j01 = creator0(&mut block_store, &mut block_dag_storage, &j4, &aj);

        aj.insert(&v0, &j01);
        let j11 = creator1(&mut block_store, &mut block_dag_storage, &j01, &aj);

        aj.insert(&v1, &j11);
        let j21 = creator2(&mut block_store, &mut block_dag_storage, &j11, &aj);

        aj.insert(&v2, &j21);
        let j31 = creator3(&mut block_store, &mut block_dag_storage, &j21, &aj);

        aj.insert(&v3, &j31);
        let _j41 = creator4(&mut block_store, &mut block_dag_storage, &j31, &aj);

        let dag2 = block_dag_storage.get_representation();

        let fault_tolerance = safety_oracle
            .normalized_fault_tolerance(&dag2, &l0.block_hash)
            .await
            .unwrap();
        assert!((fault_tolerance - 1.0).abs() < 0.01);
    })
    .await
}
