// See casper/src/main/scala/coop/rchain/casper/util/rholang/SystemDeployResult.scala

use models::rust::{
    block::state_hash::StateHash,
    casper::protocol::casper_message::{Event, ProcessedSystemDeploy, SystemDeployData},
};
use rspace_plus_plus::rspace::merger::merging_logic::NumberChannelsEndVal;

use super::system_deploy_user_error::SystemDeployUserError;

pub enum SystemDeployResult<A> {
    PlaySucceeded {
        state_hash: StateHash,
        processed_system_deploy: ProcessedSystemDeploy,
        mergeable_channels: NumberChannelsEndVal,
        result: A,
    },
    PlayFailed {
        processed_system_deploy: ProcessedSystemDeploy,
    },
}

impl<A> SystemDeployResult<A> {
    pub fn play_succeeded(
        state_hash: StateHash,
        log: Vec<Event>,
        system_deploy_data: SystemDeployData,
        mergeable_channels: NumberChannelsEndVal,
        result: A,
    ) -> Self {
        Self::PlaySucceeded {
            state_hash,
            processed_system_deploy: ProcessedSystemDeploy::Succeeded {
                event_list: log,
                system_deploy: system_deploy_data,
            },
            mergeable_channels,
            result,
        }
    }

    pub fn play_failed(log: Vec<Event>, system_deploy_error: SystemDeployUserError) -> Self {
        Self::PlayFailed {
            processed_system_deploy: ProcessedSystemDeploy::Failed {
                event_list: log,
                error_msg: system_deploy_error.error_message,
            },
        }
    }
}
