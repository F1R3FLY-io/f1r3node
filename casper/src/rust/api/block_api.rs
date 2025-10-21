// See casper/src/main/scala/coop/rchain/casper/api/BlockAPI.scala

use futures::future;
use prost::bytes::Bytes;
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;

use crypto::rust::{public_key::PublicKey, signatures::signed::Signed};
use models::casper::{
    BlockInfo, ContinuationsWithBlockInfo, DataWithBlockInfo, LightBlockInfo, RejectedDeployInfo,
    WaitingContinuationInfo,
};
use models::rhoapi::Par;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::{BlockMessage, DeployData};
use models::rust::rholang::sorter::{par_sort_matcher::ParSortMatcher, sortable::Sortable};
use models::rust::{block_hash::BlockHash, block_metadata::BlockMetadata};
use rspace_plus_plus::rspace::{
    hashing::stable_hash_provider,
    trace::event::{Event as RspaceEvent, IOEvent},
};

use crate::rust::casper::MultiParentCasper;

use crate::rust::{
    blocks::proposer::{propose_result::ProposeResult, proposer::ProposerResult},
    engine::engine_cell::EngineCell,
    errors::CasperError,
    genesis::contracts::standard_deploys,
    reporting_proto_transformer::ReportingProtoTransformer,
    state::instances::proposer_state::ProposerState,
    util::rholang::runtime_manager::RuntimeManager,
    util::{event_converter, proto_util, rholang::tools::Tools},
};

use crate::rust::ProposeFunction;

use crate::rust::safety_oracle::{CliqueOracleImpl, SafetyOracle};
use block_storage::rust::dag::block_dag_key_value_storage::DeployId;
use rspace_plus_plus::rspace::history::Either;
use shared::rust::ByteString;

pub struct BlockAPI;

pub type ApiErr<T> = eyre::Result<T>;

// Automatic error conversions for common error types used in this API
// We can only implement From for our own types, so we implement for CasperError -> String
impl From<CasperError> for String {
    fn from(err: CasperError) -> String {
        err.to_string()
    }
}

#[allow(dead_code)]
const BLOCK_API_METRICS_SOURCE: &str = "block-api";
#[allow(dead_code)]
const DEPLOY_SOURCE: &str = "block-api.deploy";
#[allow(dead_code)]
const GET_BLOCK_SOURCE: &str = "block-api.get-block";

lazy_static::lazy_static! {
    static ref REPORT_TRANSFORMER: ReportingProtoTransformer = ReportingProtoTransformer::new();
}

// TODO: Scala we should refactor BlockApi with applicative errors for better classification of errors and to overcome nesting when validating data.
#[derive(Debug)]
pub struct BlockRetrievalError {
    pub message: String,
}

impl std::fmt::Display for BlockRetrievalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockRetrievalError: {}", self.message)
    }
}

impl std::error::Error for BlockRetrievalError {}

#[derive(Debug)]
pub enum LatestBlockMessageError {
    ValidatorReadOnlyError,
    NoBlockMessageError,
}

impl std::fmt::Display for LatestBlockMessageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatestBlockMessageError::ValidatorReadOnlyError => write!(f, "ValidatorReadOnlyError"),
            LatestBlockMessageError::NoBlockMessageError => write!(f, "NoBlockMessageError"),
        }
    }
}

impl std::error::Error for LatestBlockMessageError {}

impl BlockAPI {
    pub async fn deploy(
        engine_cell: &EngineCell,
        d: Signed<DeployData>,
        trigger_propose: Option<Box<ProposeFunction>>,
        min_phlo_price: i64,
        is_node_read_only: bool,
        shard_id: &str,
    ) -> ApiErr<String> {
        async fn casper_deploy(
            casper: &dyn MultiParentCasper,
            deploy_data: Signed<DeployData>,
            trigger_propose: Option<Box<ProposeFunction>>,
        ) -> ApiErr<String> {
            let deploy_result = casper.deploy(deploy_data)?;
            let r: ApiErr<String> = match deploy_result {
                Either::Left(err) => Err(err.into()),
                Either::Right(deploy_id) => Ok(format!(
                    "Success!\nDeployId is: {}",
                    PrettyPrinter::build_string_no_limit(deploy_id.as_ref())
                )),
            };

            // call a propose if proposer defined
            if let Some(tp) = trigger_propose {
                let _proposer_result = tp(casper, true)?;
            }

            // yield r
            r
        }

        // Validation chain - mimics Scala's whenA pattern
        let validation_result: Result<(), String> = Ok(())
            .and_then(|_| {
                if is_node_read_only {
                    Err(
                        "Deploy was rejected because node is running in read-only mode."
                            .to_string(),
                    )
                } else {
                    Ok(())
                }
            })
            .and_then(|_| {
                if d.data.shard_id != shard_id {
                    Err(format!(
                        "Deploy shardId '{}' is not as expected network shard '{}'.",
                        d.data.shard_id, shard_id
                    ))
                } else {
                    Ok(())
                }
            })
            .and_then(|_| {
                let is_forbidden_key = standard_deploys::system_public_keys()
                    .iter()
                    .any(|pk| **pk == d.pk);
                if is_forbidden_key {
                    Err(
                        "Deploy refused because it's signed with forbidden private key."
                            .to_string(),
                    )
                } else {
                    Ok(())
                }
            })
            .and_then(|_| {
                if d.data.phlo_price < min_phlo_price {
                    Err(format!(
                        "Phlo price {} is less than minimum price {}.",
                        d.data.phlo_price, min_phlo_price
                    ))
                } else {
                    Ok(())
                }
            });

        // Return early if validation fails
        validation_result.map_err(|e| eyre::eyre!(e))?;

        let log_error_message =
            "Error: Could not deploy, casper instance was not available yet.".to_string();

        let eng = engine_cell.get().await;

        // Helper function for logging - mimic Scala logWarn
        let log_warn = |msg: &str| -> ApiErr<String> {
            log::warn!("{}", msg);
            Err(eyre::eyre!("{}", msg))
        };

        if let Some(casper) = eng.with_casper() {
            casper_deploy(casper, d, trigger_propose).await
        } else {
            log_warn(&log_error_message)
        }
    }

    pub async fn create_block(
        engine_cell: &EngineCell,
        trigger_propose_f: Box<ProposeFunction>,
        is_async: bool,
    ) -> ApiErr<String> {
        let log_debug = |err: &str| -> ApiErr<String> {
            log::debug!("{}", err);
            Err(eyre::eyre!("{}", err))
        };
        let log_success = |msg: &str| -> ApiErr<String> {
            log::info!("{}", msg);
            Ok(msg.to_string())
        };
        let log_warn = |msg: &str| -> ApiErr<String> {
            log::warn!("{}", msg);
            Err(eyre::eyre!("{}", msg))
        };

        let eng = engine_cell.get().await;

        if let Some(casper) = eng.with_casper() {
            // Trigger propose
            let proposer_result = trigger_propose_f(casper, is_async)?;

            let r: ApiErr<String> = match proposer_result {
                ProposerResult::Empty => log_debug("Failure: another propose is in progress"),
                ProposerResult::Failure(status, seq_number) => {
                    log_debug(&format!("Failure: {} (seqNum {})", status, seq_number))
                }
                ProposerResult::Started(seq_number) => {
                    log_success(&format!("Propose started (seqNum {})", seq_number))
                }
                ProposerResult::Success(_, block) => {
                    // TODO: Scala [WARNING] Format of this message is hardcoded in pyrchain when checking response result
                    //  Fix to use structured result with transport errors/codes.
                    // https://github.com/rchain/pyrchain/blob/a2959c75bf/rchain/client.py#L42
                    let block_hash_hex = PrettyPrinter::build_string_no_limit(&block.block_hash);
                    log_success(&format!(
                        "Success! Block {} created and added.",
                        block_hash_hex
                    ))
                }
            };

            // yield r
            r
        } else {
            log_warn("Failure: casper instance is not available.")
        }
    }

    pub async fn get_propose_result(proposer_state: &mut ProposerState) -> ApiErr<String> {
        let r = match proposer_state.curr_propose_result.take() {
            // return latest propose result
            None => {
                let default_result = (ProposeResult::not_enough_blocks(), None);
                let result = proposer_state
                    .latest_propose_result
                    .as_ref()
                    .unwrap_or(&default_result);
                let msg = match &result.1 {
                    Some(block) => {
                        let block_hash_hex =
                            PrettyPrinter::build_string_no_limit(&block.block_hash);
                        Ok(format!(
                            "Success! Block {} created and added.",
                            block_hash_hex
                        ))
                    }
                    None => Err(eyre::eyre!("{}", result.0.propose_status)),
                };
                msg
            }
            // wait for current propose to finish and return result
            Some(result_def) => {
                // this will hang API call until propose is complete, and then return result
                // TODO Scala: cancel this get when connection drops
                let result = result_def.await?;
                let msg = match &result.1 {
                    Some(block) => {
                        let block_hash_hex =
                            PrettyPrinter::build_string_no_limit(&block.block_hash);
                        Ok(format!(
                            "Success! Block {} created and added.",
                            block_hash_hex
                        ))
                    }
                    None => Err(eyre::eyre!("{}", result.0.propose_status)),
                };
                msg
            }
        };
        r
    }

    pub async fn get_listening_name_data_response(
        engine_cell: &EngineCell,
        depth: i32,
        listening_name: Par,
        max_blocks_limit: i32,
    ) -> ApiErr<(Vec<DataWithBlockInfo>, i32)> {
        let error_message =
            "Could not get listening name data, casper instance was not available yet.";

        async fn casper_response(
            casper: &dyn MultiParentCasper,
            depth: i32,
            listening_name: Par,
        ) -> ApiErr<(Vec<DataWithBlockInfo>, i32)> {
            let main_chain = BlockAPI::get_main_chain_from_tip(casper, depth).await?;
            let runtime_manager = casper.runtime_manager();
            let sorted_listening_name = ParSortMatcher::sort_match(&listening_name).term;

            let maybe_blocks_with_active_name: Vec<Option<DataWithBlockInfo>> =
                future::try_join_all(main_chain.iter().map(|block| {
                    BlockAPI::get_data_with_block_info(
                        casper,
                        runtime_manager.clone(),
                        &sorted_listening_name,
                        block,
                    )
                }))
                .await?;

            let blocks_with_active_name: Vec<DataWithBlockInfo> = maybe_blocks_with_active_name
                .into_iter()
                .flatten()
                .collect();

            Ok((
                blocks_with_active_name.clone(),
                blocks_with_active_name.len() as i32,
            ))
        }

        if depth > max_blocks_limit {
            Err(eyre::eyre!(
                "Your request on getListeningName depth {} exceed the max limit {}",
                depth,
                max_blocks_limit
            ))
        } else {
            let eng = engine_cell.get().await;
            if let Some(casper) = eng.with_casper() {
                casper_response(casper, depth, listening_name).await
            } else {
                log::warn!("{}", error_message);
                Err(eyre::eyre!("Error: {}", error_message))
            }
        }
    }

    pub async fn get_listening_name_continuation_response(
        engine_cell: &EngineCell,
        depth: i32,
        listening_names: &[Par],
        max_blocks_limit: i32,
    ) -> ApiErr<(Vec<ContinuationsWithBlockInfo>, i32)> {
        let error_message =
            "Could not get listening names continuation, casper instance was not available yet.";

        async fn casper_response(
            casper: &dyn MultiParentCasper,
            depth: i32,
            listening_names: &[Par],
        ) -> ApiErr<(Vec<ContinuationsWithBlockInfo>, i32)> {
            let main_chain = BlockAPI::get_main_chain_from_tip(casper, depth).await?;
            let runtime_manager = casper.runtime_manager();

            let sorted_listening_names: Vec<Par> = listening_names
                .iter()
                .map(|name| ParSortMatcher::sort_match(name).term)
                .collect();

            let maybe_blocks_with_active_name: Vec<Option<ContinuationsWithBlockInfo>> =
                future::try_join_all(main_chain.iter().map(|block| {
                    BlockAPI::get_continuations_with_block_info(
                        casper,
                        runtime_manager.clone(),
                        &sorted_listening_names,
                        block,
                    )
                }))
                .await?;

            let blocks_with_active_name: Vec<ContinuationsWithBlockInfo> =
                maybe_blocks_with_active_name
                    .into_iter()
                    .flatten()
                    .collect();

            Ok((
                blocks_with_active_name.clone(),
                blocks_with_active_name.len() as i32,
            ))
        }

        if depth > max_blocks_limit {
            Err(eyre::eyre!(
                "Your request on getListeningNameContinuation depth {} exceed the max limit {}",
                depth,
                max_blocks_limit
            ))
        } else {
            let eng = engine_cell.get().await;
            if let Some(casper) = eng.with_casper() {
                casper_response(casper, depth, listening_names).await
            } else {
                log::warn!("{}", error_message);
                Err(eyre::eyre!("Error: {}", error_message))
            }
        }
    }

    async fn get_main_chain_from_tip<M: MultiParentCasper + ?Sized>(
        casper: &M,
        depth: i32,
    ) -> ApiErr<Vec<BlockMessage>> {
        let mut dag = casper.block_dag().await?;
        let tip_hashes = casper.estimator(&mut dag).await?;
        let tip_hash = tip_hashes
            .first()
            .cloned()
            .ok_or_else(|| eyre::eyre!("No tip"))?;
        let tip = casper.block_store().get_unsafe(&tip_hash);
        let main_chain =
            proto_util::get_main_chain_until_depth(casper.block_store(), tip, Vec::new(), depth)?;
        Ok(main_chain)
    }

    async fn get_data_with_block_info(
        casper: &dyn MultiParentCasper,
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        sorted_listening_name: &Par,
        block: &BlockMessage,
    ) -> ApiErr<Option<DataWithBlockInfo>> {
        // TODO: Scala For Produce it doesn't make sense to have multiple names
        if BlockAPI::is_listening_name_reduced(block, &[sorted_listening_name.clone()]) {
            let state_hash = proto_util::post_state_hash(block);
            let data = runtime_manager
                .lock()
                .await
                .get_data(state_hash, sorted_listening_name)
                .await?;
            let block_info = BlockAPI::get_light_block_info(casper, block).await?;
            Ok(Some(DataWithBlockInfo {
                post_block_data: data,
                block: Some(block_info).into(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn get_continuations_with_block_info(
        casper: &dyn MultiParentCasper,
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        sorted_listening_names: &[Par],
        block: &BlockMessage,
    ) -> ApiErr<Option<ContinuationsWithBlockInfo>> {
        if Self::is_listening_name_reduced(block, sorted_listening_names) {
            let state_hash = proto_util::post_state_hash(block);

            let continuations = runtime_manager
                .lock()
                .await
                .get_continuation(state_hash, sorted_listening_names.to_vec())
                .await?;

            let continuation_infos: Vec<_> = continuations
                .into_iter()
                .map(
                    |(post_block_patterns, post_block_continuation)| WaitingContinuationInfo {
                        post_block_patterns,
                        post_block_continuation: Some(post_block_continuation),
                    },
                )
                .collect();

            let block_info = BlockAPI::get_light_block_info(casper, block).await?;
            Ok(Some(ContinuationsWithBlockInfo {
                post_block_continuations: continuation_infos,
                block: Some(block_info).into(),
            }))
        } else {
            Ok(None)
        }
    }

    fn is_listening_name_reduced(block: &BlockMessage, sorted_listening_name: &[Par]) -> bool {
        let serialized_log: Vec<_> = block
            .body
            .deploys
            .iter()
            .flat_map(|pd| pd.deploy_log.iter())
            .collect();

        let log: Vec<RspaceEvent> = serialized_log
            .iter()
            .map(|event| event_converter::to_rspace_event(event))
            .collect();

        log.iter().any(|event| match event {
            RspaceEvent::IoEvent(IOEvent::Produce(produce)) => {
                assert_eq!(
                    sorted_listening_name.len(),
                    1,
                    "Produce can have only one channel"
                );
                // channelHash == JNAInterfaceLoader.hashChannel(sortedListeningName.head)
                produce.channel_hash == stable_hash_provider::hash(&sorted_listening_name[0])
            }
            RspaceEvent::IoEvent(IOEvent::Consume(consume)) => {
                let mut expected_hashes: Vec<_> = sorted_listening_name
                    .iter()
                    .map(|name| stable_hash_provider::hash(name))
                    .collect();
                expected_hashes.sort();

                let mut actual_hashes = consume.channel_hashes.clone();
                actual_hashes.sort();

                actual_hashes == expected_hashes
            }

            RspaceEvent::Comm(comm) => {
                let mut expected_hashes: Vec<_> = sorted_listening_name
                    .iter()
                    .map(|name| stable_hash_provider::hash(name))
                    .collect();
                expected_hashes.sort();

                let mut consume_hashes = comm.consume.channel_hashes.clone();
                consume_hashes.sort();

                let consume_matches = consume_hashes == expected_hashes;

                let produce_matches = comm.produces.iter().any(|produce| {
                    produce.channel_hash
                        == stable_hash_provider::hash_from_vec(&sorted_listening_name.to_vec())
                });

                consume_matches || produce_matches
            }
        })
    }

    async fn toposort_dag<A: 'static + Send>(
        engine_cell: &EngineCell,
        depth: i32,
        max_depth_limit: i32,
        do_it: fn((&dyn MultiParentCasper, Vec<Vec<BlockHash>>)) -> ApiErr<A>,
    ) -> ApiErr<A> {
        let error_message =
            "Could not visualize graph, casper instance was not available yet.".to_string();

        async fn casper_response<A: 'static + Send>(
            casper: &dyn MultiParentCasper,
            depth: i32,
            do_it: fn((&dyn MultiParentCasper, Vec<Vec<BlockHash>>)) -> ApiErr<A>,
        ) -> ApiErr<A> {
            let dag = casper.block_dag().await?;

            let latest_block_number = dag.latest_block_number();

            let topo_sort = dag.topo_sort(latest_block_number - depth as i64, None)?;

            let result = do_it((casper, topo_sort));

            result
        }

        if depth > max_depth_limit {
            return Err(eyre::eyre!(
                "Your request depth {} exceed the max limit {}",
                depth,
                max_depth_limit
            ));
        }

        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            casper_response(casper, depth, do_it).await
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn get_blocks_by_heights(
        engine_cell: &EngineCell,
        start_block_number: i64,
        end_block_number: i64,
        max_blocks_limit: i32,
    ) -> ApiErr<Vec<LightBlockInfo>> {
        let error_message = format!(
            "Could not retrieve blocks from {} to {}",
            start_block_number, end_block_number
        );

        async fn casper_response(
            casper: &dyn MultiParentCasper,
            start_block_number: i64,
            end_block_number: i64,
        ) -> ApiErr<Vec<LightBlockInfo>> {
            let dag = casper.block_dag().await?;

            let topo_sort_dag = dag.topo_sort(start_block_number, Some(end_block_number))?;

            let result: ApiErr<Vec<LightBlockInfo>> = {
                let mut block_infos_at_height_acc = Vec::new();
                for block_hashes_at_height in topo_sort_dag {
                    let blocks_at_height: Vec<_> = block_hashes_at_height
                        .iter()
                        .map(|block_hash| casper.block_store().get_unsafe(block_hash))
                        .collect();

                    for block in blocks_at_height {
                        let block_info = BlockAPI::get_light_block_info(casper, &block).await?;
                        block_infos_at_height_acc.push(block_info);
                    }
                }
                Ok(block_infos_at_height_acc)
            };

            result
        }

        if end_block_number - start_block_number > max_blocks_limit as i64 {
            return Err(eyre::eyre!(
                "Your request startBlockNumber {} and endBlockNumber {} exceed the max limit {}",
                start_block_number,
                end_block_number,
                max_blocks_limit
            ));
        }

        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            casper_response(casper, start_block_number, end_block_number).await
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn visualize_dag<R: 'static>(
        engine_cell: &EngineCell,
        depth: i32,
        start_block_number: i32,
        visualizer: Box<dyn Fn(Vec<Vec<BlockHash>>, String) -> eyre::Result<()>>,
        serialize: Box<dyn Fn() -> eyre::Result<R>>,
    ) -> ApiErr<R> {
        let error_message = "visual dag failed".to_string();

        async fn casper_response<R: 'static>(
            casper: &dyn MultiParentCasper,
            depth: i32,
            start_block_number: i32,
            visualizer: Box<dyn Fn(Vec<Vec<BlockHash>>, String) -> eyre::Result<()>>,
            serialize: Box<dyn Fn() -> eyre::Result<R>>,
        ) -> ApiErr<R> {
            let dag = casper.block_dag().await?;

            let start_block_num = if start_block_number == 0 {
                dag.latest_block_number()
            } else {
                start_block_number as i64
            };

            let topo_sort_dag =
                dag.topo_sort(start_block_num - depth as i64, Some(start_block_num))?;

            let lfb_hash = dag.last_finalized_block();

            let _visualizer_result =
                visualizer(topo_sort_dag, PrettyPrinter::build_string_bytes(&lfb_hash))?;

            // result <- serialize
            let result = serialize().map_err(|e| eyre::eyre!(e.to_string()))?;

            Ok(result)
        }

        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            casper_response(casper, depth, start_block_number, visualizer, serialize).await
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn machine_verifiable_dag(
        engine_cell: &EngineCell,
        depth: i32,
        max_depth_limit: i32,
    ) -> ApiErr<String> {
        let do_it = |(_casper, topo_sort): (&dyn MultiParentCasper, Vec<Vec<BlockHash>>)| -> ApiErr<String> {
            // case (_, topoSort) => ...
            let fetch_parents = |block_hash: &BlockHash| -> Vec<BlockHash> {
                let block = _casper.block_store().get_unsafe(block_hash);
                block.header.parents_hash_list.clone()
            };

            //string will be converted to an ApiErr<String>
            let result = topo_sort
                .into_iter()
                .flat_map(|block_hashes| {
                    block_hashes.into_iter().flat_map(|block_hash| {
                        let block_hash_str = PrettyPrinter::build_string_bytes(&block_hash);
                        fetch_parents(&block_hash).into_iter().map(move |parent_hash| {
                            format!("{} {}", block_hash_str, PrettyPrinter::build_string_bytes(&parent_hash))
                        })
                    })
                })
                .collect::<Vec<String>>()
                .join("\n");
            Ok(result)
        };

        BlockAPI::toposort_dag(engine_cell, depth, max_depth_limit, do_it).await
    }

    pub async fn get_blocks(
        engine_cell: &EngineCell,
        depth: i32,
        max_depth_limit: i32,
    ) -> ApiErr<Vec<LightBlockInfo>> {
        let do_it = |(casper, topo_sort): (&dyn MultiParentCasper, Vec<Vec<BlockHash>>)| -> ApiErr<Vec<LightBlockInfo>> {
            let mut block_infos_acc = Vec::new();

            for block_hashes_at_height in topo_sort {
                let blocks_at_height: Vec<_> = block_hashes_at_height
                    .iter()
                    .map(|block_hash| casper.block_store().get_unsafe(block_hash))
                    .collect();

                for block in blocks_at_height {
                    let block_info = BlockAPI::construct_light_block_info(&block, 0.0);
                    block_infos_acc.push(block_info);
                }
            }

            block_infos_acc.reverse();
            Ok(block_infos_acc)
        };

        BlockAPI::toposort_dag(engine_cell, depth, max_depth_limit, do_it).await
    }

    pub async fn show_main_chain(
        engine_cell: &EngineCell,
        depth: i32,
        max_depth_limit: i32,
    ) -> Vec<LightBlockInfo> {
        let error_message =
            "Could not show main chain, casper instance was not available yet.".to_string();

        async fn casper_response(
            casper: &dyn MultiParentCasper,
            depth: i32,
        ) -> ApiErr<Vec<LightBlockInfo>> {
            let dag = casper.block_dag().await?;

            let mut dag_mut = dag;
            let tip_hashes = casper.estimator(&mut dag_mut).await?;

            let tip_hash = tip_hashes
                .first()
                .cloned()
                .ok_or_else(|| eyre::eyre!("No tip hashes found"))?;

            let tip = casper.block_store().get_unsafe(&tip_hash);

            let main_chain = proto_util::get_main_chain_until_depth(
                casper.block_store(),
                tip,
                Vec::new(),
                depth,
            )?;

            let mut block_infos = Vec::new();
            for block in main_chain {
                let block_info = BlockAPI::construct_light_block_info(&block, 0.0);
                block_infos.push(block_info);
            }

            Ok(block_infos)
        }

        if depth > max_depth_limit {
            return Vec::new();
        }

        let eng = engine_cell.get().await;

        if let Some(casper) = eng.with_casper() {
            casper_response(casper, depth)
                .await
                .unwrap_or_else(|_| Vec::new())
        } else {
            log::warn!("{}", error_message);
            Vec::new()
        }
    }

    pub async fn find_deploy(
        engine_cell: &EngineCell,
        deploy_id: &DeployId,
    ) -> ApiErr<LightBlockInfo> {
        let error_message =
            "Could not find block with deploy, casper instance was not available yet.".to_string();

        let eng = engine_cell.get().await;

        if let Some(casper) = eng.with_casper() {
            let dag = casper.block_dag().await?;
            let maybe_block_hash = dag.lookup_by_deploy_id(deploy_id)?;
            let maybe_block =
                maybe_block_hash.map(|block_hash| casper.block_store().get_unsafe(&block_hash));
            let response =
                maybe_block.map(|block| BlockAPI::construct_light_block_info(&block, 0.0));

            match response {
                Some(light_block_info) => Ok(light_block_info),
                None => Err(eyre::eyre!(
                    "Couldn't find block containing deploy with id: {}",
                    PrettyPrinter::build_string_no_limit(deploy_id)
                )),
            }
        } else {
            Err(eyre::eyre!("Error: errorMessage"))
        }
    }

    pub async fn get_block(engine_cell: &EngineCell, hash: &str) -> ApiErr<BlockInfo> {
        let error_message =
            "Could not get block, casper instance was not available yet.".to_string();

        async fn casper_response(casper: &dyn MultiParentCasper, hash: &str) -> ApiErr<BlockInfo> {
            if hash.len() < 6 {
                return Err(eyre::eyre!(
                    "Input hash value must be at least 6 characters: {}",
                    hash
                ));
            }

            let hash_byte_string = hex::decode(hash)
                .map_err(|_| eyre::eyre!("Input hash value is not valid hex string: {}", hash))?;

            let get_block = async {
                let block_hash = prost::bytes::Bytes::from(hash_byte_string);
                casper
                    .block_store()
                    .get(&block_hash)
                    .map_err(|e| eyre::eyre!(e.to_string()))
            };

            let find_block = async {
                let dag = casper
                    .block_dag()
                    .await
                    .map_err(|e| eyre::eyre!(e.to_string()))?;
                match dag.find(hash) {
                    Some(block_hash) => casper
                        .block_store()
                        .get(&block_hash)
                        .map_err(|e| eyre::eyre!(e.to_string())),
                    None => Ok(None),
                }
            };

            let block_f = if hash.len() == 64 {
                get_block.await
            } else {
                find_block.await
            };

            let block = block_f?
                .ok_or_else(|| eyre::eyre!("Error: Failure to find block with hash: {}", hash))?;

            let dag = casper.block_dag().await?;
            if dag.contains(&block.block_hash) {
                let block_info = BlockAPI::get_full_block_info(casper, &block).await?;
                Ok(block_info)
            } else {
                Err(eyre::eyre!(
                    "Error: Block with hash {} received but not added yet",
                    hash
                ))
            }
        }

        let eng = engine_cell.get().await;

        if let Some(casper) = eng.with_casper() {
            casper_response(casper, hash).await
        } else {
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    async fn get_block_info<M: MultiParentCasper + ?Sized, A: Sized + Send>(
        casper: &M,
        block: &BlockMessage,
        constructor: fn(&BlockMessage, f32) -> A,
    ) -> ApiErr<A> {
        let dag = casper.block_dag().await?;
        // TODO: Scala this is temporary solution to not calculate fault tolerance all the blocks
        let old_block =
            Some(dag.latest_block_number() - block.body.state.block_number).map(|diff| diff > 100);

        let normalized_fault_tolerance = if old_block.unwrap_or(false) {
            if dag.is_finalized(&block.block_hash) {
                1.0f32
            } else {
                -1.0f32
            }
        } else {
            let safety_oracle = CliqueOracleImpl;
            safety_oracle
                .normalized_fault_tolerance(&dag, &block.block_hash)
                .await?
        };

        let weights_map = proto_util::weight_map(block);
        let weights_u64: HashMap<Bytes, u64> = weights_map
            .into_iter()
            .map(|(k, v)| (k, v as u64))
            .collect();

        let initial_fault = casper.normalized_initial_fault(weights_u64)?;
        let fault_tolerance = normalized_fault_tolerance - initial_fault;

        let block_info = constructor(block, fault_tolerance);
        Ok(block_info)
    }

    async fn get_full_block_info<M: MultiParentCasper + ?Sized>(
        casper: &M,
        block: &BlockMessage,
    ) -> ApiErr<BlockInfo> {
        Self::get_block_info(casper, block, Self::construct_block_info).await
    }

    pub async fn get_light_block_info(
        casper: &dyn MultiParentCasper,
        block: &BlockMessage,
    ) -> ApiErr<LightBlockInfo> {
        Self::get_block_info(casper, block, Self::construct_light_block_info).await
    }

    fn construct_block_info(block: &BlockMessage, fault_tolerance: f32) -> BlockInfo {
        let light_block_info = Self::construct_light_block_info(block, fault_tolerance);
        let deploys = block
            .body
            .deploys
            .iter()
            .map(|processed_deploy| processed_deploy.clone().to_deploy_info())
            .collect();

        BlockInfo {
            block_info: Some(light_block_info).into(),
            deploys,
        }
    }

    fn construct_light_block_info(block: &BlockMessage, fault_tolerance: f32) -> LightBlockInfo {
        LightBlockInfo {
            block_hash: PrettyPrinter::build_string_no_limit(&block.block_hash),
            sender: PrettyPrinter::build_string_no_limit(&block.sender),
            seq_num: block.seq_num as i64,
            sig: PrettyPrinter::build_string_no_limit(&block.sig),
            sig_algorithm: block.sig_algorithm.clone(),
            shard_id: block.shard_id.clone(),
            extra_bytes: block.extra_bytes.clone(),
            version: block.header.version,
            timestamp: block.header.timestamp,
            header_extra_bytes: block.header.extra_bytes.clone(),
            parents_hash_list: block
                .header
                .parents_hash_list
                .iter()
                .map(|h| PrettyPrinter::build_string_no_limit(h))
                .collect(),
            block_number: block.body.state.block_number,
            pre_state_hash: PrettyPrinter::build_string_no_limit(&block.body.state.pre_state_hash),
            post_state_hash: PrettyPrinter::build_string_no_limit(
                &block.body.state.post_state_hash,
            ),
            body_extra_bytes: block.body.extra_bytes.clone(),
            bonds: block
                .body
                .state
                .bonds
                .iter()
                .map(proto_util::bond_to_bond_info)
                .collect(),
            block_size: block.to_proto().encode_to_vec().len().to_string(),
            deploy_count: block.body.deploys.len() as i32,
            fault_tolerance,
            justifications: block
                .justifications
                .iter()
                .map(proto_util::justification_to_justification_info)
                .collect(),
            rejected_deploys: block
                .body
                .rejected_deploys
                .iter()
                .map(|r| RejectedDeployInfo {
                    sig: PrettyPrinter::build_string_no_limit(&r.sig),
                })
                .collect(),
        }
    }

    // Be careful to use this method , because it would iterate the whole indexes to find the matched one which would cause performance problem
    // Trying to use BlockStore.get as much as possible would more be preferred
    #[allow(dead_code)]
    async fn find_block_from_store(
        engine_cell: &EngineCell,
        hash: &str,
    ) -> Result<Option<BlockMessage>, String> {
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            let dag = casper.block_dag().await?;
            let block_hash_opt = dag.find(hash);
            match block_hash_opt {
                Some(block_hash) => {
                    let message = casper
                        .block_store()
                        .get(&block_hash)
                        .map_err(|e| e.to_string())?;
                    Ok(message)
                }
                None => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    pub fn preview_private_names(
        deployer: &ByteString,
        timestamp: i64,
        name_qty: i32,
    ) -> ApiErr<Vec<ByteString>> {
        let mut rand = Tools::unforgeable_name_rng(&PublicKey::from_bytes(&deployer), timestamp);
        let safe_qty = std::cmp::min(name_qty, 1024) as usize;
        let ids: Vec<BlockHash> = (0..safe_qty)
            .map(|_| rand.next().into_iter().map(|b| b as u8).collect())
            .collect();
        Ok(ids.into_iter().map(|bytes| bytes.to_vec()).collect())
    }

    pub async fn last_finalized_block(engine_cell: &EngineCell) -> ApiErr<BlockInfo> {
        let error_message =
            "Could not get last finalized block, casper instance was not available yet.";
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            let last_finalized_block = casper.last_finalized_block().await?;
            let block_info = Self::get_full_block_info(casper, &last_finalized_block).await?;
            Ok(block_info)
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn is_finalized(engine_cell: &EngineCell, hash: &str) -> ApiErr<bool> {
        let error_message =
            "Could not check if block is finalized, casper instance was not available yet.";
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            let dag = casper.block_dag().await?;
            let given_block_hash =
                hex::decode(hash).map_err(|_| eyre::eyre!("Invalid hex string"))?;
            let result = dag.is_finalized(&given_block_hash.into());
            Ok(result)
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn bond_status(engine_cell: &EngineCell, public_key: &ByteString) -> ApiErr<bool> {
        let error_message =
            "Could not check if validator is bonded, casper instance was not available yet.";
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            let last_finalized_block = casper.last_finalized_block().await?;
            let runtime_manager = casper.runtime_manager();
            let post_state_hash = &last_finalized_block.body.state.post_state_hash;
            let bonds = runtime_manager
                .lock()
                .await
                .compute_bonds(post_state_hash)
                .await?;
            let validator_bond_opt = bonds.iter().find(|bond| bond.validator == *public_key);
            Ok(validator_bond_opt.is_some())
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    /// Explore the data or continuation in the tuple space for specific blockHash
    ///
    /// - `term`: the term you want to explore in the request. Be sure the first `new` should be `return`
    /// - `block_hash`: the block hash you want to explore
    /// - `use_pre_state_hash`: Each block has preStateHash and postStateHash. If `use_pre_state_hash` is true, the explore
    ///   would try to execute on preState.
    pub async fn exploratory_deploy(
        engine_cell: &EngineCell,
        term: String,
        block_hash: Option<String>,
        use_pre_state_hash: bool,
        dev_mode: bool,
    ) -> ApiErr<(Vec<Par>, LightBlockInfo)> {
        let error_message =
            "Could not execute exploratory deploy, casper instance was not available yet.";
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            let is_read_only = casper.get_validator().is_none();
            if is_read_only || dev_mode {
                let target_block = if block_hash.is_none() {
                    Some(casper.last_finalized_block().await?)
                } else {
                    let hash_byte_string =
                        hex::decode(block_hash.as_ref().unwrap()).map_err(|_| {
                            eyre::eyre!(
                                "Input hash value is not valid hex string: {:?}",
                                block_hash
                            )
                        })?;
                    casper.block_store().get(&hash_byte_string.into())?
                };

                let res = match target_block {
                    Some(b) => {
                        let post_state_hash = if use_pre_state_hash {
                            proto_util::pre_state_hash(&b)
                        } else {
                            proto_util::post_state_hash(&b)
                        };
                        let runtime_manager = casper.runtime_manager();
                        let res = runtime_manager
                            .lock()
                            .await
                            .play_exploratory_deploy(term, &post_state_hash)
                            .await?;
                        let light_block_info = Self::get_light_block_info(casper, &b).await?;
                        Some((res, light_block_info))
                    }
                    None => None,
                };
                res.ok_or_else(|| eyre::eyre!("Can not find block {:?}", block_hash))
            } else {
                Err(eyre::eyre!(
                    "Exploratory deploy can only be executed on read-only RNode."
                ))
            }
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn get_latest_message(engine_cell: &EngineCell) -> ApiErr<BlockMetadata> {
        let error_message = "Could not get latest message, casper instance was not available yet.";
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            let validator_opt = casper.get_validator();
            let validator = validator_opt.ok_or_else(|| {
                eyre::eyre!("{}", LatestBlockMessageError::ValidatorReadOnlyError)
            })?;
            let dag = casper.block_dag().await?;
            let latest_message_opt =
                dag.latest_message(&validator.public_key.bytes.clone().into())?;
            let latest_message = latest_message_opt
                .ok_or_else(|| eyre::eyre!("{}", LatestBlockMessageError::NoBlockMessageError))?;
            Ok(latest_message)
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }

    pub async fn get_data_at_par(
        engine_cell: &EngineCell,
        par: &Par,
        block_hash: String,
        _use_pre_state_hash: bool,
    ) -> ApiErr<(Vec<Par>, LightBlockInfo)> {
        async fn casper_response(
            casper: &dyn MultiParentCasper,
            par: &Par,
            block_hash: &str,
        ) -> ApiErr<(Vec<Par>, LightBlockInfo)> {
            let block_hash_bytes: BlockHash = hex::decode(block_hash)
                .map_err(|_| eyre::eyre!("Invalid block hash"))?
                .into();
            let block = casper.block_store().get_unsafe(&block_hash_bytes);
            let sorted_par = ParSortMatcher::sort_match(par).term;
            let runtime_manager = casper.runtime_manager();
            let data =
                BlockAPI::get_data_with_block_info(casper, runtime_manager, &sorted_par, &block)
                    .await?;
            if let Some(data_with_block_info) = data {
                Ok((
                    data_with_block_info.post_block_data,
                    data_with_block_info.block.unwrap_or_default(),
                ))
            } else {
                Err(eyre::eyre!("No data found"))
            }
        }

        let error_message = "Could not get data at par, casper instance was not available yet.";
        let eng = engine_cell.get().await;
        if let Some(casper) = eng.with_casper() {
            casper_response(casper, par, &block_hash).await
        } else {
            log::warn!("{}", error_message);
            Err(eyre::eyre!("Error: {}", error_message))
        }
    }
}
