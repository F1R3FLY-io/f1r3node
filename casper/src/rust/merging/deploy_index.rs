// See casper/src/main/scala/coop/rchain/casper/merging/DeployIndex.scala

use models::rust::casper::protocol::casper_message::Event;
use rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex;

#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd)]
pub struct DeployIndex {
    pub deploy_id: prost::bytes::Bytes,
    pub cost: u64,
    pub event_log_index: EventLogIndex,
}

impl DeployIndex {
    // This cost is required because rejection option selection rule depends on how much branch costs.
    // For now system deploys do not have any weight, cost is 0.
    pub const SYS_SLASH_DEPLOY_COST: u64 = 0;
    pub const SYS_CLOSE_BLOCK_DEPLOY_COST: u64 = 0;
    pub const SYS_EMPTY_DEPLOY_COST: u64 = 0;

    // These are to be put in rejected set in blocks, so prefix format is defined for identification purposes.
    pub const SYS_SLASH_DEPLOY_ID: &'static [u8] = &[1];
    pub const SYS_CLOSE_BLOCK_DEPLOY_ID: &'static [u8] = &[2];
    pub const SYS_EMPTY_DEPLOY_ID: &'static [u8] = &[3];

    pub fn new(
        sig: prost::bytes::Bytes,
        cost: u64,
        events: Vec<Event>,
        create_event_log_index: impl Fn(Vec<Event>) -> EventLogIndex,
    ) -> Self {
        let event_log_index = create_event_log_index(events);

        Self {
            deploy_id: sig,
            cost,
            event_log_index,
        }
    }
}
