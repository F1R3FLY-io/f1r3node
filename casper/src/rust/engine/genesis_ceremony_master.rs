// See casper/src/main/scala/coop/rchain/casper/engine/GenesisCeremonyMaster.scala

use async_trait::async_trait;
use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use comm::rust::peer_node::PeerNode;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::transport::transport_layer::TransportLayer;
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::{ApprovedBlock, BlockMessage, CasperMessage};
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::rust::casper::{CasperShardConf, MultiParentCasper};
use crate::rust::engine::block_retriever::BlockRetriever;
use crate::rust::engine::engine::Engine;
use crate::rust::engine::engine_cell::EngineCell;
use crate::rust::errors::CasperError;
use crate::rust::estimator::Estimator;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::validator_identity::ValidatorIdentity;

pub struct GenesisCeremonyMaster<T: TransportLayer + Send + Sync + Clone + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TransportLayer + Send + Sync + Clone + 'static> GenesisCeremonyMaster<T> {
    pub fn new(
        _approve_protocol: crate::rust::engine::approve_block_protocol::ApproveBlockProtocolImpl<T>,
    ) -> Self {
        unimplemented!("GenesisCeremonyMaster::new - TODO: implement")
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn waiting_for_approved_block_loop(
        _block_processing_queue: Arc<
            Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
        >,
        _blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        _casper_shard_conf: CasperShardConf,
        _validator_id: Option<ValidatorIdentity>,
        _disable_state_exporter: bool,
        _transport_layer: Arc<T>,
        _rp_conf_ask: RPConf,
        _connections_cell: ConnectionsCell,
        _last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        _block_store: Arc<Mutex<Option<KeyValueBlockStore>>>,
        _block_dag_storage: Arc<Mutex<Option<BlockDagKeyValueStorage>>>,
        _deploy_storage: Arc<Mutex<Option<KeyValueDeployStorage>>>,
        _casper_buffer_storage: Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
        _rspace_state_manager: Arc<Mutex<Option<RSpaceStateManager>>>,
        _event_publisher: Arc<F1r3flyEvents>,
        _block_retriever: Arc<BlockRetriever<T>>,
        _engine_cell: Arc<EngineCell>,
        _runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        _estimator: Arc<Mutex<Option<Estimator>>>,
    ) -> Result<(), CasperError> {
        unimplemented!("GenesisCeremonyMaster::waiting_for_approved_block_loop - TODO: implement")
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync + Clone + 'static> Engine for GenesisCeremonyMaster<T> {
    async fn init(&self) -> Result<(), CasperError> {
        unimplemented!("GenesisCeremonyMaster::init - TODO: implement")
    }

    async fn handle(&self, _peer: PeerNode, _msg: CasperMessage) -> Result<(), CasperError> {
        unimplemented!("GenesisCeremonyMaster::handle - TODO: implement")
    }

    fn with_casper(&self) -> Option<&dyn MultiParentCasper> {
        None
    }
}
