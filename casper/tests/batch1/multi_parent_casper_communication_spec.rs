// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperCommunicationSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::casper::MultiParentCasper;
use casper::rust::util::construct_deploy;
use crypto::rust::signatures::signed::Signed;
use models::rust::casper::protocol::casper_message::DeployData;

#[tokio::test]
async fn multi_parent_casper_should_ask_peers_for_blocks_it_is_missing() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let shard_id = genesis.genesis_block.shard_id.clone();

    let deploy1 = construct_deploy::source_deploy_now(
        "for(_ <- @1){ Nil } | @1!(1)".to_string(),
        None,
        None,
        Some(shard_id.clone()),
    )
    .unwrap();

    let signed_block1 = nodes[0].add_block_from_deploys(&[deploy1]).await.unwrap();

    nodes[1].handle_receive().await.unwrap();

    nodes[2].shutoff().unwrap(); // nodes(2) misses this block

    let deploy2 = construct_deploy::source_deploy_now(
        "@2!(2)".to_string(),
        None,
        None,
        Some(shard_id.clone()),
    )
    .unwrap();

    let signed_block2 = nodes[0].add_block_from_deploys(&[deploy2]).await.unwrap();

    // signedBlock2 has signedBlock1 as a dependency
    // When node(2) tries to add signedBlock2, it should request signedBlock1
    let _ = nodes[2].add_block(signed_block2.clone()).await;

    // Scala: r <- nodes(2).requestedBlocks.get.map(v => v.get(signedBlock1.blockHash)).map { ... }
    // Check if signedBlock1 is in requestedBlocks of node(2)
    // TestNode.requested_blocks should be shared with BlockRetriever.requested_blocks
    let is_requested = {
        let state = nodes[2].requested_blocks.lock().unwrap();
        state.contains_key(&signed_block1.block_hash)
    };

    assert!(
        is_requested,
        "signedBlock1 should be in requestedBlocks of node(2)"
    );
}

/*
 * Scala comments:
 *
 *  DAG Looks like this:
 *
 *             h1
 *            /  \
 *           g1   g2
 *           |  X |
 *           f1   f2
 *            \  /
 *             e1
 *             |
 *             d1
 *            /  \
 *           c1   c2
 *           |  X |
 *           b1   b2
 *           |  X |
 *           a1   a2
 *            \  /
 *          genesis
 *
 * f2 has in its justifications list c2. This should be handled properly.
 * TODO: investigate why this test is so brittle - in presence of hashes it starts to pass
 * only when hashes are synchronized precisely as in the test - otherwise it will see 2 parents of h1
 *
 */
// TODO reenable when merging of token balances is done
#[tokio::test]
#[ignore = "Scala ignore"]
async fn multi_parent_casper_should_ask_peers_for_blocks_it_is_missing_and_add_them() {
    fn make_deploy(i: usize, shard_id: &str) -> Signed<DeployData> {
        let term = if i == 0 { "@2!(2)" } else { "@1!(1)" };
        let sec = if i == 0 {
            construct_deploy::DEFAULT_SEC.clone()
        } else {
            construct_deploy::DEFAULT_SEC2.clone()
        };

        construct_deploy::source_deploy_now(
            term.to_string(),
            Some(sec),
            None,
            Some(shard_id.to_string()),
        )
        .unwrap()
    }

    async fn step_split(nodes: &mut [TestNode], shard_id: &str) {
        let deploy0 = make_deploy(0, shard_id);
        let deploy1 = make_deploy(1, shard_id);

        // Split nodes for mutable access
        let (node0_slice, rest) = nodes.split_at_mut(1);
        let (node1_slice, node2_slice) = rest.split_at_mut(1);

        // nodes(0).addBlock
        let _block0 = node0_slice[0]
            .add_block_from_deploys(&[deploy0])
            .await
            .unwrap();

        // nodes(1).addBlock
        let _block1 = node1_slice[0]
            .add_block_from_deploys(&[deploy1])
            .await
            .unwrap();

        // nodes(0).syncWith(nodes(1))
        node0_slice[0]
            .sync_with_one(&mut node1_slice[0])
            .await
            .unwrap();

        // nodes(2).shutoff() - nodes(2) misses this block
        node2_slice[0].shutoff().unwrap();
    }

    async fn step_single(nodes: &mut [TestNode], shard_id: &str) {
        let deploy0 = make_deploy(0, shard_id);

        // Split nodes for mutable access
        let (node0_slice, rest) = nodes.split_at_mut(1);
        let (node1_slice, node2_slice) = rest.split_at_mut(1);

        // nodes(0).addBlock
        let _block = node0_slice[0]
            .add_block_from_deploys(&[deploy0])
            .await
            .unwrap();

        // nodes(1).syncWith(nodes(0))
        node1_slice[0]
            .sync_with_one(&mut node0_slice[0])
            .await
            .unwrap();

        // nodes(2).shutoff()
        node2_slice[0].shutoff().unwrap(); // nodes(2) misses this block
    }

    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let shard_id = genesis.genesis_block.shard_id.clone();

    // Build the DAG
    step_split(&mut nodes, &shard_id).await; // blocks a1 a2
    step_split(&mut nodes, &shard_id).await; // blocks b1 b2
    step_split(&mut nodes, &shard_id).await; // blocks c1 c2

    step_single(&mut nodes, &shard_id).await; // block d1
    step_single(&mut nodes, &shard_id).await; // block e1

    step_split(&mut nodes, &shard_id).await; // blocks f1 f2
    step_split(&mut nodes, &shard_id).await; // blocks g1 g2

    // This block will be propagated to all nodes and force nodes(2) to ask for missing blocks
    let deploy_h1 = make_deploy(0, &shard_id);
    let block_h1 = nodes[0].add_block_from_deploys(&[deploy_h1]).await.unwrap(); // block h1

    {
        let mut node_refs: Vec<&mut TestNode> = nodes.iter_mut().collect();
        TestNode::propagate(&mut node_refs).await.unwrap(); // force the network to communicate
    }

    // Casper in node2 should contain block and its parents, requested as dependencies
    let contains_h1 = nodes[2].contains(&block_h1.block_hash);
    assert!(
        contains_h1,
        "nodes(2) should contain block h1 after propagation"
    );

    for parent_hash in &block_h1.header.parents_hash_list {
        let contains_parent = nodes[2].contains(parent_hash);
        assert!(
            contains_parent,
            "nodes(2) should contain parent {:?}",
            hex::encode(parent_hash)
        );
    }
}

// Scala comments:
// TODO: investigate this test - it doesn't make much sense in the presence of hashes (see RCHAIN-3819)
// and why on earth does it test logs?
#[tokio::test]
#[ignore = "Scala ignore"]
async fn multi_parent_casper_should_handle_a_long_chain_of_block_requests_appropriately() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 2, None, None, None, None)
        .await
        .unwrap();

    // Create blocks 0-9 on node0, node1 misses all of them
    for i in 0..10 {
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();

        let _block = nodes[0].add_block_from_deploys(&[deploy]).await.unwrap();

        nodes[1].shutoff().unwrap(); //nodes(1) misses this block
    }

    let deploy_data10 =
        construct_deploy::basic_deploy_data(10, None, Some(genesis.genesis_block.shard_id.clone()))
            .unwrap();

    let _block11 = nodes[0]
        .add_block_from_deploys(&[deploy_data10])
        .await
        .unwrap();

    // Cycle of requesting and passing blocks until block #3 from nodes(0) to nodes(1)
    for _i in 0..=8 {
        nodes[1].handle_receive().await.unwrap();
        nodes[0].handle_receive().await.unwrap();
        nodes[1].handle_receive().await.unwrap();
        nodes[0].handle_receive().await.unwrap();
        nodes[1].handle_receive().await.unwrap();
        nodes[0].handle_receive().await.unwrap();
    }

    // We simulate a network failure here by not allowing block #2 to get passed to nodes(1)

    // And then we assume fetchDependencies eventually gets called
    {
        let casper = nodes[1].casper.clone();
        casper.fetch_dependencies().await.unwrap();
    }

    nodes[0].handle_receive().await.unwrap();
    nodes[0].handle_receive().await.unwrap();

    // Scala: nodes(1).logEff.infos.count(_ startsWith "Requested missing block") should be(10)
    // Scala: nodes(0).logEff.infos.count(s => (s startsWith "Received request for block") && (s endsWith "Response sent.")) should be(10)

    // TODO: In Rust we don't have LogStub, so we can't check log counts
    // This test would need to be rewritten to check actual behavior instead of logs
    // For now, we just verify the operations completed without errors

    println!("Note: This test needs log capturing implementation to fully match Scala behavior");
}
