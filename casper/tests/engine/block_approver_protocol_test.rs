// See casper/src/test/scala/coop/rchain/casper/engine/BlockApproverProtocolTest.scala

use crate::helper::test_node::TestNode;
use crate::util::comm::transport_layer_test_impl::TransportLayerTestImpl;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::engine::block_approver_protocol::BlockApproverProtocol;
use crypto::rust::public_key::PublicKey;
use models::rust::{
    block_implicits::get_random_block,
    casper::protocol::casper_message::{ApprovedBlockCandidate, BlockMessage, UnapprovedBlock},
};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

const SHARD_ID: &str = "root-shard";

struct TestContext {
    protocol: BlockApproverProtocol<TransportLayerTestImpl>,
    node: TestNode,
    required_sigs: i32,
}

impl TestContext {
    fn create_unapproved(required_sigs: i32, block: &BlockMessage) -> UnapprovedBlock {
        UnapprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: block.clone(),
                required_sigs,
            },
            timestamp: 0,
            duration: 0,
        }
    }

    async fn create_protocol() -> Result<Self, Box<dyn Error>> {
        let params = GenesisBuilder::build_genesis_parameters_with_defaults(None, None);
        let genesis_params = params.2.clone();

        let mut genesis_builder = GenesisBuilder::new();
        let genesis_context = genesis_builder
            .build_genesis_with_parameters(Some(params))
            .await?;

        let bonds: HashMap<PublicKey, i64> = genesis_params
            .proof_of_stake
            .validators
            .iter()
            .map(|v| (v.pk.clone(), v.stake))
            .collect();

        let required_sigs = (bonds.len() - 1) as i32;

        let mut nodes =
            TestNode::create_network(genesis_context, 1, None, None, None, None).await?;

        // Note: Using remove(0) instead of referencing nodes[0] because TestNode doesn't implement Clone
        // and we need an owned value. This is acceptable since networkSize=1 (only one element).
        let node = nodes.remove(0);

        let protocol = BlockApproverProtocol::new(
            node.validator_id_opt.clone().unwrap(),
            genesis_params.timestamp,
            genesis_params.vaults,
            bonds,
            genesis_params.proof_of_stake.minimum_bond,
            genesis_params.proof_of_stake.maximum_bond,
            genesis_params.proof_of_stake.epoch_length,
            genesis_params.proof_of_stake.quarantine_length,
            genesis_params.proof_of_stake.number_of_active_validators,
            required_sigs,
            genesis_params.proof_of_stake.pos_multi_sig_public_keys,
            genesis_params.proof_of_stake.pos_multi_sig_quorum,
            node.tle.clone(),
            Arc::new(node.rp_conf.clone()),
        )?;

        Ok(Self {
            protocol,
            node,
            required_sigs,
        })
    }
}

#[tokio::test]
async fn block_approver_protocol_should_respond_to_valid_approved_block_candidates() {
    // In Rust, we use TestContext struct to hold both protocol and node.
    let mut ctx = TestContext::create_protocol().await.unwrap();

    let genesis = ctx.node.genesis.clone();
    let unapproved = TestContext::create_unapproved(ctx.required_sigs, &genesis);

    ctx.protocol
        .unapproved_block_packet_handler(
            &mut ctx.node.runtime_manager,
            &ctx.node.local,
            unapproved,
            SHARD_ID,
        )
        .await
        .unwrap();

    // Note: Add log validation when LogStub mechanism from Scala is implemented in Rust
    // Scala: node.logEff.infos.exists(_.contains("Approval sent in response")) should be(true)
    // Scala: node.logEff.warns.isEmpty should be(true)

    let queue = ctx
        .node
        .tle
        .test_network()
        .peer_queue(&ctx.node.local)
        .unwrap();

    assert_eq!(queue.len(), 1);
}

#[tokio::test]
async fn block_approver_protocol_should_log_a_warning_for_invalid_approved_block_candidates() {
    let mut ctx = TestContext::create_protocol().await.unwrap();

    let different_unapproved1 = TestContext::create_unapproved(
        ctx.required_sigs / 2, // wrong number of signatures
        &ctx.node.genesis.clone(),
    );

    let different_unapproved2 = TestContext::create_unapproved(
        ctx.required_sigs,
        &get_random_block(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        ), // wrong block
    );

    ctx.protocol
        .unapproved_block_packet_handler(
            &mut ctx.node.runtime_manager,
            &ctx.node.local,
            different_unapproved1,
            SHARD_ID,
        )
        .await
        .unwrap();

    ctx.protocol
        .unapproved_block_packet_handler(
            &mut ctx.node.runtime_manager,
            &ctx.node.local,
            different_unapproved2,
            SHARD_ID,
        )
        .await
        .unwrap();

    // Note: Add log validation when LogStub mechanism from Scala is implemented in Rust
    // Scala: node.logEff.warns.count(_.contains("Received unexpected genesis block candidate")) should be(2)

    let queue = ctx
        .node
        .tle
        .test_network()
        .peer_queue(&ctx.node.local)
        .unwrap();

    assert!(queue.is_empty());
}

#[tokio::test]
async fn block_approver_protocol_should_successfully_validate_correct_candidate() {
    let mut ctx = TestContext::create_protocol().await.unwrap();

    let unapproved = TestContext::create_unapproved(ctx.required_sigs, &ctx.node.genesis.clone());

    // Scala: BlockApproverProtocol.validateCandidate[Effect](...) - static method call
    let result = BlockApproverProtocol::<TransportLayerTestImpl>::validate_candidate(
        &mut ctx.node.runtime_manager,
        &unapproved.candidate,
        ctx.protocol.required_sigs,
        ctx.protocol.deploy_timestamp,
        &ctx.protocol.vaults,
        &ctx.protocol.bonds_bytes,
        ctx.protocol.minimum_bond,
        ctx.protocol.maximum_bond,
        ctx.protocol.epoch_length,
        ctx.protocol.quarantine_length,
        ctx.protocol.number_of_active_validators,
        SHARD_ID,
        &ctx.protocol.pos_multi_sig_public_keys,
        ctx.protocol.pos_multi_sig_quorum,
    )
    .await;

    assert_eq!(result, Ok(()));
}

#[tokio::test]
async fn block_approver_protocol_should_reject_candidate_with_incorrect_bonds() {
    let mut ctx = TestContext::create_protocol().await.unwrap();

    let unapproved = TestContext::create_unapproved(ctx.required_sigs, &ctx.node.genesis.clone());

    // Scala: validateCandidate with bonds = Map.empty (incorrect bonds)
    let wrong_bonds = HashMap::new();

    let result = BlockApproverProtocol::<TransportLayerTestImpl>::validate_candidate(
        &mut ctx.node.runtime_manager,
        &unapproved.candidate,
        ctx.protocol.required_sigs,
        ctx.protocol.deploy_timestamp,
        &ctx.protocol.vaults,
        &wrong_bonds, // bonds are incorrect (empty)
        ctx.protocol.minimum_bond,
        ctx.protocol.maximum_bond,
        ctx.protocol.epoch_length,
        ctx.protocol.quarantine_length,
        ctx.protocol.number_of_active_validators,
        SHARD_ID,
        &ctx.protocol.pos_multi_sig_public_keys,
        ctx.protocol.pos_multi_sig_quorum,
    )
    .await;

    assert_eq!(result, Err("Block bonds don't match expected.".to_string()));
}

#[tokio::test]
async fn block_approver_protocol_should_reject_candidate_with_incorrect_vaults() {
    let mut ctx = TestContext::create_protocol().await.unwrap();

    let unapproved = TestContext::create_unapproved(ctx.required_sigs, &ctx.node.genesis.clone());

    // Scala: validateCandidate with vaults = Seq.empty[Vault] (incorrect vaults)
    let wrong_vaults = vec![];

    let result = BlockApproverProtocol::<TransportLayerTestImpl>::validate_candidate(
        &mut ctx.node.runtime_manager,
        &unapproved.candidate,
        ctx.protocol.required_sigs,
        ctx.protocol.deploy_timestamp,
        &wrong_vaults, // vaults are incorrect (empty)
        &ctx.protocol.bonds_bytes,
        ctx.protocol.minimum_bond,
        ctx.protocol.maximum_bond,
        ctx.protocol.epoch_length,
        ctx.protocol.quarantine_length,
        ctx.protocol.number_of_active_validators,
        SHARD_ID,
        &ctx.protocol.pos_multi_sig_public_keys,
        ctx.protocol.pos_multi_sig_quorum,
    )
    .await;

    assert_eq!(
        result,
        Err(
            "Mismatch between number of candidate deploys and expected number of deploys."
                .to_string()
        )
    );
}

#[tokio::test]
async fn block_approver_protocol_should_reject_candidate_with_incorrect_blessed_contracts() {
    let mut ctx = TestContext::create_protocol().await.unwrap();

    let unapproved = TestContext::create_unapproved(ctx.required_sigs, &ctx.node.genesis.clone());

    // Scala: validateCandidate with incorrect genesis params (minimumBond + 1, maximumBond - 1, etc.)
    let result = BlockApproverProtocol::<TransportLayerTestImpl>::validate_candidate(
        &mut ctx.node.runtime_manager,
        &unapproved.candidate,
        ctx.protocol.required_sigs,
        ctx.protocol.deploy_timestamp,
        &ctx.protocol.vaults,
        &ctx.protocol.bonds_bytes,
        ctx.protocol.minimum_bond + 1,                // incorrect
        ctx.protocol.maximum_bond - 1,                // incorrect
        ctx.protocol.epoch_length + 1,                // incorrect
        ctx.protocol.quarantine_length + 1,           // incorrect
        ctx.protocol.number_of_active_validators + 1, // incorrect
        SHARD_ID,
        &ctx.protocol.pos_multi_sig_public_keys,
        ctx.protocol.pos_multi_sig_quorum,
    )
    .await;

    assert!(result.is_err());
}
