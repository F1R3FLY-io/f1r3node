// See casper/src/main/scala/coop/rchain/casper/merging/BlockIndex.scala

use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{
        Event, ProcessedDeploy, ProcessedSystemDeploy, SystemDeployData,
    },
};

use rholang::rust::interpreter::rho_runtime::RhoHistoryRepository;
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    merger::{event_log_index::EventLogIndex, merging_logic::NumberChannelsDiff},
    trace::event::Produce,
};

use crate::rust::{
    errors::CasperError,
    merging::{deploy_chain_index::DeployChainIndex, deploy_index::DeployIndex},
    util::event_converter,
};

use rspace_plus_plus::rspace::merger::merging_logic;

#[derive(Clone)]
pub struct BlockIndex {
    pub block_hash: BlockHash,
    pub deploy_chains: Vec<DeployChainIndex>,
}

pub fn create_event_log_index(
    events: Vec<Event>,
    history_repository: RhoHistoryRepository,
    pre_state_hash: &Blake2b256Hash,
    mergeable_chs: NumberChannelsDiff,
) -> EventLogIndex {
    let pre_state_reader = history_repository
        .get_history_reader(&pre_state_hash)
        .unwrap();

    let produce_exists_in_pre_state = |p: &Produce| {
        pre_state_reader
            .get_data(&p.channel_hash)
            .map_or(false, |data| data.iter().any(|d| d.source == *p))
    };

    let produce_touches_pre_state_join = |p: &Produce| {
        pre_state_reader
            .get_joins(&p.channel_hash)
            .map_or(false, |joins| joins.iter().any(|j| j.len() > 1))
    };

    EventLogIndex::new(
        events
            .iter()
            .map(event_converter::to_rspace_event)
            .collect(),
        produce_exists_in_pre_state,
        produce_touches_pre_state_join,
        mergeable_chs,
    )
}

pub fn new(
    block_hash: &BlockHash,
    usr_processed_deploys: &Vec<ProcessedDeploy>,
    sys_processed_deploys: &Vec<ProcessedSystemDeploy>,
    pre_state_hash: &Blake2b256Hash,
    post_state_hash: &Blake2b256Hash,
    history_repository: &RhoHistoryRepository,
    mergeable_chs: &Vec<NumberChannelsDiff>,
) -> Result<BlockIndex, CasperError> {
    // Connect mergeable channels data with processed deploys by index
    let usr_count = usr_processed_deploys.len();
    let sys_count = sys_processed_deploys.len();
    let deploy_count = usr_count + sys_count;
    let mrg_count = mergeable_chs.len();

    // Number of deploys must match the size of mergeable channels maps
    assert_eq!(
        deploy_count, mrg_count,
        "Cache of mergeable channels ({}) doesn't match deploys count ({}).",
        mrg_count, deploy_count
    );

    // Connect deploy with corresponding mergeable channels map
    let (usr_mergeable_chs, sys_mergeable_chs) = mergeable_chs.split_at(usr_count);
    let usr_deploys_with_mergeable: Vec<_> = usr_processed_deploys
        .iter()
        .zip(usr_mergeable_chs.iter())
        .collect();
    let sys_deploys_with_mergeable: Vec<_> = sys_processed_deploys
        .iter()
        .zip(sys_mergeable_chs.iter())
        .collect();

    // Create user deploy indices - filter out failed deploys
    let mut usr_deploy_indices = Vec::new();
    for (deploy, merge_chs) in usr_deploys_with_mergeable {
        if !deploy.is_failed {
            let event_log_index = create_event_log_index(
                deploy.deploy_log.clone(),
                history_repository.clone(),
                pre_state_hash,
                merge_chs.clone(),
            );

            let deploy_index = DeployIndex::new(
                deploy.deploy.sig.clone(),
                deploy.cost.cost,
                deploy.deploy_log.clone(),
                |_| event_log_index.clone(),
            );

            usr_deploy_indices.push(deploy_index);
        }
    }

    // Create system deploy indices - collect successful system deploys
    let mut sys_deploy_indices = Vec::new();
    for (sys_deploy, merge_chs) in sys_deploys_with_mergeable {
        match sys_deploy {
            ProcessedSystemDeploy::Succeeded {
                system_deploy,
                event_list,
            } => {
                let (sig, cost) = match system_deploy {
                    SystemDeployData::Slash { .. } => {
                        let mut sig_bytes = block_hash.to_vec();
                        sig_bytes.extend_from_slice(DeployIndex::SYS_SLASH_DEPLOY_ID);
                        (sig_bytes.into(), DeployIndex::SYS_SLASH_DEPLOY_COST)
                    }
                    SystemDeployData::CloseBlockSystemDeployData => {
                        let mut sig_bytes = block_hash.to_vec();
                        sig_bytes.extend_from_slice(DeployIndex::SYS_CLOSE_BLOCK_DEPLOY_ID);
                        (sig_bytes.into(), DeployIndex::SYS_CLOSE_BLOCK_DEPLOY_COST)
                    }
                    SystemDeployData::Empty => {
                        let mut sig_bytes = block_hash.to_vec();
                        sig_bytes.extend_from_slice(DeployIndex::SYS_EMPTY_DEPLOY_ID);
                        (sig_bytes.into(), DeployIndex::SYS_EMPTY_DEPLOY_COST)
                    }
                };

                let event_log_index = create_event_log_index(
                    event_list.clone(),
                    history_repository.clone(),
                    pre_state_hash,
                    merge_chs.clone(),
                );

                let deploy_index =
                    DeployIndex::new(sig, cost, event_list.clone(), |_| event_log_index.clone());

                sys_deploy_indices.push(deploy_index);
            }
            ProcessedSystemDeploy::Failed { .. } => {
                // Skip failed system deploys
            }
        }
    }

    // Combine all deploy indices
    let mut all_deploy_indices = usr_deploy_indices;
    all_deploy_indices.extend(sys_deploy_indices);

    // Here deploys from a single block are examined. Atm deploys in block are executed sequentially,
    // so all conflicts are resolved according to order of sequential execution.
    // Therefore there won't be any conflicts between event logs. But there can be dependencies.
    let deploy_chains = merging_logic::compute_related_sets(
        &all_deploy_indices.into_iter().collect(),
        |l: &DeployIndex, r: &DeployIndex| {
            merging_logic::depends(&l.event_log_index, &r.event_log_index)
        },
    );

    // Convert deploy chains to DeployChainIndex
    let mut deploy_chain_indices = Vec::new();
    for deploy_chain in deploy_chains.0.iter() {
        let chain_index = DeployChainIndex::new(
            deploy_chain,
            pre_state_hash,
            post_state_hash,
            history_repository.clone(),
        )
        .map_err(|e| CasperError::HistoryError(e))?;
        deploy_chain_indices.push(chain_index);
    }

    Ok(BlockIndex {
        block_hash: block_hash.clone(),
        deploy_chains: deploy_chain_indices,
    })
}
