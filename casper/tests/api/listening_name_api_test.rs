// See casper/src/test/scala/coop/rchain/casper/api/ListeningNameAPITest.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};
use casper::rust::api::block_api::BlockAPI;
use casper::rust::util::construct_deploy;
use casper::rust::util::construct_deploy::source_deploy_now;
use models::casper::WaitingContinuationInfo;
use models::rhoapi::{expr::ExprInstance, BindPattern, Expr, Par};

struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .expect("Failed to build genesis");

        Self { genesis }
    }
}

#[tokio::test]
async fn get_listening_name_data_response_should_work_with_unsorted_channels() {
    let ctx = TestContext::new().await;

    let mut standalone_node = TestNode::standalone(ctx.genesis.clone()).await.unwrap();

    let deploy = source_deploy_now(
        "@{ 3 | 2 | 1 }!(0)".to_string(),
        None,
        None,
        Some(ctx.genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let _block = standalone_node
        .add_block_from_deploys(&[deploy])
        .await
        .unwrap();

    let listening_name = Par {
        exprs: vec![
            Expr {
                expr_instance: Some(ExprInstance::GInt(2)),
            },
            Expr {
                expr_instance: Some(ExprInstance::GInt(1)),
            },
            Expr {
                expr_instance: Some(ExprInstance::GInt(3)),
            },
        ],
        ..Default::default()
    };

    let result_data = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GInt(0)),
        }],
        ..Default::default()
    };

    let listening_name_response = BlockAPI::get_listening_name_data_response(
        &standalone_node.engine_cell,
        i32::MAX,
        listening_name,
        i32::MAX,
    )
    .await;

    assert!(listening_name_response.is_ok());
    let (block_results, length) = listening_name_response.unwrap();

    let data1: Vec<Vec<Par>> = block_results
        .iter()
        .map(|br| br.post_block_data.clone())
        .collect();
    let blocks1: Vec<_> = block_results.iter().map(|br| &br.block).collect();

    assert_eq!(data1, vec![vec![result_data]]);
    assert_eq!(blocks1.len(), 1);
    assert_eq!(length, 1);
}

// TODO: Update test for multi-parent merging semantics - the main chain concept
// changes with multi-parent blocks where all validators' blocks are merged.
// Scala ignored this in PR #288.
#[tokio::test]
#[ignore = "Multi-parent merging changes main chain semantics"]
async fn get_listening_name_data_response_should_work_across_a_chain() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    // Note: We create deploys one by one with sleep to ensure unique timestamps.
    // In Scala, the Time effect provides unique timestamps automatically,
    // but in Rust we need to explicitly wait between deploys to avoid NoNewDeploys error.
    let mut deploy_datas = Vec::new();
    for _ in 0..8 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            0,
            None,
            Some(ctx.genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        deploy_datas.push(deploy);
    }

    let _block1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[0].clone()])
        .await
        .unwrap();

    let listening_name = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GInt(0)),
        }],
        ..Default::default()
    };

    let result_data = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GInt(0)),
        }],
        ..Default::default()
    };

    let listening_name_response1 = BlockAPI::get_listening_name_data_response(
        &nodes[0].engine_cell,
        i32::MAX,
        listening_name.clone(),
        i32::MAX,
    )
    .await;

    assert!(listening_name_response1.is_ok());

    let (block_results1, length1) = listening_name_response1.unwrap();

    let data1: Vec<Vec<Par>> = block_results1
        .iter()
        .map(|br| br.post_block_data.clone())
        .collect();
    let blocks1: Vec<_> = block_results1.iter().map(|br| &br.block).collect();

    assert_eq!(data1, vec![vec![result_data.clone()]]);
    assert_eq!(blocks1.len(), 1);
    assert_eq!(length1, 1);

    let _block2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[1].clone()])
        .await
        .unwrap();
    let _block3 = TestNode::propagate_block_at_index(&mut nodes, 2, &[deploy_datas[2].clone()])
        .await
        .unwrap();
    let _block4 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[3].clone()])
        .await
        .unwrap();

    let listening_name_response2 = BlockAPI::get_listening_name_data_response(
        &nodes[0].engine_cell,
        i32::MAX,
        listening_name.clone(),
        i32::MAX,
    )
    .await;

    assert!(listening_name_response2.is_ok());
    let (block_results2, length2) = listening_name_response2.unwrap();

    let data2: Vec<Vec<Par>> = block_results2
        .iter()
        .map(|br| br.post_block_data.clone())
        .collect();
    let blocks2: Vec<_> = block_results2.iter().map(|br| &br.block).collect();

    assert_eq!(
        data2,
        vec![
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![result_data.clone(), result_data.clone()],
            vec![result_data.clone()]
        ]
    );
    assert_eq!(blocks2.len(), 4);
    assert_eq!(length2, 4);

    let _block5 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[4].clone()])
        .await
        .unwrap();
    let _block6 = TestNode::propagate_block_at_index(&mut nodes, 2, &[deploy_datas[5].clone()])
        .await
        .unwrap();
    let _block7 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[6].clone()])
        .await
        .unwrap();

    let listening_name_response3 = BlockAPI::get_listening_name_data_response(
        &nodes[0].engine_cell,
        i32::MAX,
        listening_name.clone(),
        i32::MAX,
    )
    .await;

    assert!(listening_name_response3.is_ok());
    let (block_results3, length3) = listening_name_response3.unwrap();

    let data3: Vec<Vec<Par>> = block_results3
        .iter()
        .map(|br| br.post_block_data.clone())
        .collect();
    let blocks3: Vec<_> = block_results3.iter().map(|br| &br.block).collect();

    assert_eq!(
        data3,
        vec![
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![
                result_data.clone(),
                result_data.clone(),
                result_data.clone()
            ],
            vec![result_data.clone(), result_data.clone()],
            vec![result_data.clone()]
        ]
    );
    assert_eq!(blocks3.len(), 7);
    assert_eq!(length3, 7);

    let listening_name_response3_until_depth = BlockAPI::get_listening_name_data_response(
        &nodes[0].engine_cell,
        1,
        listening_name.clone(),
        i32::MAX,
    )
    .await;

    assert!(listening_name_response3_until_depth.is_ok());
    let (_, length_depth1) = listening_name_response3_until_depth.unwrap();
    assert_eq!(length_depth1, 1);

    let listening_name_response3_until_depth2 = BlockAPI::get_listening_name_data_response(
        &nodes[0].engine_cell,
        2,
        listening_name,
        i32::MAX,
    )
    .await;

    assert!(listening_name_response3_until_depth2.is_ok());
    let (_, length_depth2) = listening_name_response3_until_depth2.unwrap();
    assert_eq!(length_depth2, 2);
}

#[tokio::test]
async fn get_listening_name_continuation_response_should_work_with_unsorted_channels() {
    let ctx = TestContext::new().await;

    let mut standalone_node = TestNode::standalone(ctx.genesis.clone()).await.unwrap();

    let deploy = source_deploy_now(
        "for (@0 <- @{ 3 | 2 | 1 } & @1 <- @{ 2 | 1 }) { 0 }".to_string(),
        None,
        None,
        Some(ctx.genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let _block = standalone_node
        .add_block_from_deploys(&[deploy])
        .await
        .unwrap();

    let listening_names_shuffled1 = vec![
        Par {
            exprs: vec![
                Expr {
                    expr_instance: Some(ExprInstance::GInt(1)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(2)),
                },
            ],
            ..Default::default()
        },
        Par {
            exprs: vec![
                Expr {
                    expr_instance: Some(ExprInstance::GInt(2)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(1)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(3)),
                },
            ],
            ..Default::default()
        },
    ];

    let desired_result = WaitingContinuationInfo {
        post_block_patterns: vec![
            BindPattern {
                patterns: vec![Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GInt(1)),
                    }],
                    ..Default::default()
                }],
                remainder: None,
                free_count: 0,
            },
            BindPattern {
                patterns: vec![Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GInt(0)),
                    }],
                    ..Default::default()
                }],
                remainder: None,
                free_count: 0,
            },
        ],
        post_block_continuation: Some(Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GInt(0)),
            }],
            ..Default::default()
        }),
    };

    let listening_name_response1 = BlockAPI::get_listening_name_continuation_response(
        &standalone_node.engine_cell,
        i32::MAX,
        &listening_names_shuffled1,
        i32::MAX,
    )
    .await;

    assert!(listening_name_response1.is_ok());
    let (block_results1, length1) = listening_name_response1.unwrap();

    let continuations1: Vec<Vec<WaitingContinuationInfo>> = block_results1
        .iter()
        .map(|br| br.post_block_continuations.clone())
        .collect();
    let blocks1: Vec<_> = block_results1.iter().map(|br| &br.block).collect();

    assert_eq!(continuations1, vec![vec![desired_result.clone()]]);
    assert_eq!(blocks1.len(), 1);
    assert_eq!(length1, 1);

    let listening_names_shuffled2 = vec![
        Par {
            exprs: vec![
                Expr {
                    expr_instance: Some(ExprInstance::GInt(2)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(1)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(3)),
                },
            ],
            ..Default::default()
        },
        Par {
            exprs: vec![
                Expr {
                    expr_instance: Some(ExprInstance::GInt(1)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(2)),
                },
            ],
            ..Default::default()
        },
    ];

    let listening_name_response2 = BlockAPI::get_listening_name_continuation_response(
        &standalone_node.engine_cell,
        i32::MAX,
        &listening_names_shuffled2,
        i32::MAX,
    )
    .await;

    assert!(listening_name_response2.is_ok());
    let (block_results2, length2) = listening_name_response2.unwrap();

    let continuations2: Vec<Vec<WaitingContinuationInfo>> = block_results2
        .iter()
        .map(|br| br.post_block_continuations.clone())
        .collect();
    let blocks2: Vec<_> = block_results2.iter().map(|br| &br.block).collect();

    assert_eq!(continuations2, vec![vec![desired_result]]);
    assert_eq!(blocks2.len(), 1);
    assert_eq!(length2, 1);
}
