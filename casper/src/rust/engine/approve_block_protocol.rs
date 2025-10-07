// See casper/src/main/scala/coop/rchain/casper/engine/ApproveBlockProtocol.scala

use comm::rust::{
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    test_instances::TransportLayerStub,
    transport::transport_layer::TransportLayer,
};
use crypto::rust::hash::blake2b256::Blake2b256;
use models::casper::Signature as ProtoSignature;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockCandidate, BlockApproval, BlockMessage, UnapprovedBlock,
};
use models::rust::casper::protocol::packet_type_tag::ToPacket;
use prost::{bytes, Message};
use shared::rust::shared::f1r3fly_events::{F1r3flyEvent, F1r3flyEvents};
use shared::rust::shared::metrics::Metrics;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;

use crate::rust::{
    errors::CasperError,
    genesis::{
        contracts::{proof_of_stake::ProofOfStake, validator::Validator},
        genesis::Genesis,
    },
    util::{
        bonds_parser::BondsParser, rholang::runtime_manager::RuntimeManager,
        vault_parser::VaultParser,
    },
    validate::Validate,
};

// Simple wrapper for ProtoSignature to work around orphan rules
#[derive(Debug, Clone)]
struct SignatureWrapper(pub ProtoSignature);

impl std::hash::Hash for SignatureWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.public_key.hash(state);
        self.0.algorithm.hash(state);
        self.0.sig.hash(state);
    }
}

impl PartialEq for SignatureWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0.public_key == other.0.public_key
            && self.0.algorithm == other.0.algorithm
            && self.0.sig == other.0.sig
    }
}

impl Eq for SignatureWrapper {}

pub struct ApproveBlockProtocolFactory;

impl ApproveBlockProtocolFactory {
    pub fn unsafe_new(
        genesis_block: BlockMessage,
        required_sigs: i32,
        duration: Duration,
        interval: Duration,
    ) -> ApproveBlockProtocolImpl<TransportLayerStub> {
        let sigs = Arc::new(Mutex::new(HashSet::new()));
        let last_approved_block = Arc::new(Mutex::new(None));
        let transport = Arc::new(TransportLayerStub::new());
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        ApproveBlockProtocolImpl::<TransportLayerStub>::new(
            genesis_block,
            required_sigs,
            start,
            duration,
            interval,
            sigs,
            last_approved_block,
            None,
            None,
            transport,
            None,
            None,
        )
    }

    pub fn unsafe_new_with_infrastructure<T: TransportLayer + Send + Sync + 'static>(
        genesis_block: BlockMessage,
        required_sigs: i32,
        duration: Duration,
        interval: Duration,
        metrics: Arc<dyn Metrics>,
        event_log: Arc<F1r3flyEvents>,
        transport: Arc<T>,
        connections_cell: Option<Arc<ConnectionsCell>>,
        conf: Option<Arc<RPConf>>,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
    ) -> ApproveBlockProtocolImpl<T> {
        let sigs = Arc::new(Mutex::new(HashSet::new()));
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        ApproveBlockProtocolImpl::new(
            genesis_block,
            required_sigs,
            start,
            duration,
            interval,
            sigs,
            last_approved_block,
            Some(metrics),
            Some((*event_log).clone()),
            transport,
            connections_cell,
            conf,
        )
    }

    pub async fn create<T: TransportLayer + Send + Sync + 'static>(
        bonds_path: String,
        autogen_shard_size: i32,
        _genesis_path: std::path::PathBuf,
        vaults_path: String,
        minimum_bond: i64,
        maximum_bond: i64,
        epoch_length: i32,
        quarantine_length: i32,
        number_of_active_validators: i32,
        shard_id: String,
        deploy_timestamp: Option<i64>,
        required_sigs: i32,
        duration: Duration,
        interval: Duration,
        block_number: i64,
        pos_multi_sig_public_keys: Vec<String>,
        pos_multi_sig_quorum: i32,
        runtime_manager: &mut RuntimeManager,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        metrics: Option<Arc<dyn Metrics>>,
        event_log: Option<F1r3flyEvents>,
        transport: Arc<T>,
        connections_cell: Arc<ConnectionsCell>,
        conf: Arc<RPConf>,
    ) -> Result<ApproveBlockProtocolImpl<T>, CasperError> {
        let now = Self::current_millis();
        let timestamp = deploy_timestamp.unwrap_or(now as i64);

        let vaults = VaultParser::parse_from_path_str(&vaults_path)
            .map_err(|e| CasperError::RuntimeError(format!("Failed to parse vaults: {}", e)))?;
        let bonds = BondsParser::parse_with_autogen(&bonds_path, autogen_shard_size as usize)
            .map_err(|e| CasperError::RuntimeError(format!("Failed to parse bonds: {}", e)))?;

        if bonds.len() <= required_sigs as usize {
            return Err(CasperError::RuntimeError(
                "Required sigs must be smaller than the number of bonded validators".to_string(),
            ));
        }

        let validators: Vec<Validator> = bonds
            .into_iter()
            .map(|(public_key, stake)| Validator {
                pk: public_key,
                stake,
            })
            .collect();

        let genesis = Genesis {
            shard_id,
            timestamp,
            block_number,
            proof_of_stake: ProofOfStake {
                minimum_bond,
                maximum_bond,
                validators,
                epoch_length,
                quarantine_length,
                number_of_active_validators,
                pos_multi_sig_public_keys,
                pos_multi_sig_quorum,
            },
            vaults,
            supply: i64::MAX,
            version: 1,
        };

        let genesis_block = Genesis::create_genesis_block(runtime_manager, &genesis).await?;
        let sigs = Arc::new(Mutex::new(HashSet::new()));

        Ok(ApproveBlockProtocolImpl::new(
            genesis_block,
            required_sigs,
            now,
            duration,
            interval,
            sigs,
            last_approved_block,
            metrics,
            event_log,
            transport,
            Some(connections_cell),
            Some(conf),
        ))
    }

    fn current_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

pub struct ApproveBlockProtocolImpl<T: TransportLayer + Send + Sync> {
    genesis_block: BlockMessage,
    required_sigs: i32,
    start: u64,
    duration: Duration,
    interval: Duration,
    sigs: Arc<Mutex<HashSet<SignatureWrapper>>>,
    last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,

    // Infrastructure - matching Scala implicits
    metrics: Option<Arc<dyn Metrics>>,
    event_log: Option<F1r3flyEvents>,
    transport: Arc<T>,
    connections_cell: Option<Arc<ConnectionsCell>>,
    conf: Option<Arc<RPConf>>,

    // Derived fields
    trusted_validators: HashSet<bytes::Bytes>,
    candidate: ApprovedBlockCandidate,
    candidate_hash: String,
    sig_data: Vec<u8>,
}

impl<T: TransportLayer + Send + Sync> ApproveBlockProtocolImpl<T> {
    fn new(
        genesis_block: BlockMessage,
        required_sigs: i32,
        start: u64,
        duration: Duration,
        interval: Duration,
        sigs: Arc<Mutex<HashSet<SignatureWrapper>>>,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        metrics: Option<Arc<dyn Metrics>>,
        event_log: Option<F1r3flyEvents>,
        transport: Arc<T>,
        connections_cell: Option<Arc<ConnectionsCell>>,
        conf: Option<Arc<RPConf>>,
    ) -> Self {
        let trusted_validators = genesis_block
            .body
            .state
            .bonds
            .iter()
            .map(|bond| bond.validator.clone())
            .collect();

        let candidate = ApprovedBlockCandidate {
            block: genesis_block.clone(), // Needed: genesis_block used again below and stored in struct
            required_sigs,
        };

        let candidate_hash = PrettyPrinter::build_string_bytes(&genesis_block.block_hash);

        // Clone needed: to_proto() takes ownership but candidate is stored in struct
        let sig_data = Blake2b256::hash(candidate.clone().to_proto().encode_to_vec());

        Self {
            genesis_block,
            required_sigs,
            start,
            duration,
            interval,
            sigs,
            last_approved_block,
            metrics,
            event_log,
            transport,
            connections_cell,
            conf,
            trusted_validators,
            candidate,
            candidate_hash,
            sig_data,
        }
    }

    fn signed_by_trusted_validator(&self, approval: &BlockApproval) -> bool {
        self.trusted_validators.contains(&approval.sig.public_key)
    }

    async fn send_unapproved_block(&self) -> Result<(), CasperError> {
        log::info!("Broadcasting UnapprovedBlock {}...", self.candidate_hash);

        if let (Some(connections_cell), Some(conf)) = (&self.connections_cell, &self.conf) {
            // Recreate UnapprovedBlock from its components
            let unapproved_block = UnapprovedBlock {
                candidate: self.candidate.clone(),
                timestamp: self.start as i64,
                duration: self.duration.as_millis() as i64,
            };
            let packet = unapproved_block.to_proto().mk_packet();

            self.transport
                .stream_packet_to_peers(connections_cell, conf, packet, None)
                .await
                .map_err(|e| {
                    CasperError::RuntimeError(format!("Failed to broadcast UnapprovedBlock: {}", e))
                })?;
        }

        // Publish event
        if let Some(event_log) = &self.event_log {
            event_log
                .publish(F1r3flyEvent::SentUnapprovedBlock(
                    self.candidate_hash.clone(),
                ))
                .map_err(|e| CasperError::RuntimeError(e))?;
        }

        Ok(())
    }

    async fn send_approved_block(&self, approved_block: &ApprovedBlock) -> Result<(), CasperError> {
        log::info!("Sending ApprovedBlock {} to peers...", self.candidate_hash);

        if let (Some(connections_cell), Some(conf)) = (&self.connections_cell, &self.conf) {
            let packet = approved_block.clone().to_proto().mk_packet();

            self.transport
                .stream_packet_to_peers(connections_cell, conf, packet, None)
                .await
                .map_err(|e| {
                    CasperError::RuntimeError(format!("Failed to send ApprovedBlock: {}", e))
                })?;
        }

        // Publish event
        if let Some(event_log) = &self.event_log {
            event_log
                .publish(F1r3flyEvent::SentApprovedBlock(self.candidate_hash.clone()))
                .map_err(|e| CasperError::RuntimeError(e))?;
        }

        Ok(())
    }

    async fn complete_if(
        &self,
        time: u64,
        signatures: &HashSet<SignatureWrapper>,
    ) -> Result<(), CasperError> {
        if (time >= self.start + self.duration.as_millis() as u64
            && signatures.len() >= self.required_sigs as usize)
            || self.required_sigs == 0
        {
            Box::pin(self.complete_genesis_ceremony(signatures.clone())).await
        } else {
            log::info!(
                "Failed to meet approval conditions. \
                Signatures: {} of {} required. \
                Duration {} ms of {} ms minimum. \
                Continue broadcasting UnapprovedBlock...",
                signatures.len(),
                self.required_sigs,
                time - self.start,
                self.duration.as_millis()
            );
            Box::pin(self.internal_run()).await
        }
    }

    async fn complete_genesis_ceremony(
        &self,
        signatures: HashSet<SignatureWrapper>,
    ) -> Result<(), CasperError> {
        let approved_block = ApprovedBlock {
            candidate: self.candidate.clone(),
            sigs: signatures.into_iter().map(|s| s.0).collect(),
        };

        // Set LastApprovedBlock
        {
            let mut last_approved = self.last_approved_block.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire LastApprovedBlock lock".to_string())
            })?;
            *last_approved = Some(approved_block.clone());
        }

        self.send_approved_block(&approved_block).await?;
        Ok(())
    }

    async fn internal_run(&self) -> Result<(), CasperError> {
        self.send_unapproved_block().await?;
        sleep(self.interval).await;

        let current_time = ApproveBlockProtocolFactory::current_millis();
        let signatures = {
            let sigs_guard = self.sigs.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire signatures lock".to_string())
            })?;
            sigs_guard.clone()
        };

        self.complete_if(current_time, &signatures).await
    }

    pub async fn add_approval(&self, approval: BlockApproval) -> Result<(), CasperError> {
        let valid_sig = approval.candidate == self.candidate
            && Validate::signature(&self.sig_data, &approval.sig);

        let sender = hex::encode(&approval.sig.public_key);

        if self.signed_by_trusted_validator(&approval) {
            if valid_sig {
                // Mimic Scala's modify operation that returns (before, after)
                let (before_size, after_size) = {
                    let mut sigs_guard = self.sigs.lock().map_err(|_| {
                        CasperError::RuntimeError("Failed to acquire signatures lock".to_string())
                    })?;

                    let before_size = sigs_guard.len();
                    sigs_guard.insert(SignatureWrapper(approval.sig.clone()));
                    let after_size = sigs_guard.len();

                    (before_size, after_size)
                };

                // Match Scala behavior exactly
                if after_size > before_size {
                    log::info!("New signature received");
                    // Increment metrics counter like Scala does
                    if let Some(metrics) = &self.metrics {
                        let _ = metrics.increment_counter("genesis").map_err(|e| {
                            log::warn!("Failed to increment metrics counter: {}", e);
                        });
                    }
                    // Publish BlockApprovalReceived event only for new signatures
                    if let Some(event_log) = &self.event_log {
                        event_log
                            .publish(F1r3flyEvent::BlockApprovalReceived(
                                shared::rust::shared::f1r3fly_event::BlockApprovalReceived {
                                    block_hash: self.candidate_hash.clone(),
                                    sender: sender.clone(),
                                },
                            ))
                            .map_err(|e| CasperError::RuntimeError(e))?;
                    }
                } else {
                    log::info!("No new sigs received");
                }

                log::info!("Received block approval from {}", sender);

                // Log signatures like Scala does
                let signatures_info = {
                    let sigs_guard = self.sigs.lock().map_err(|_| {
                        CasperError::RuntimeError("Failed to acquire signatures lock".to_string())
                    })?;

                    let sig_strings: Vec<String> = sigs_guard
                        .iter()
                        .map(|sig| PrettyPrinter::build_string_bytes(&sig.0.public_key))
                        .collect();

                    (sigs_guard.len(), sig_strings.join(", "))
                };

                log::info!(
                    "{} approvals received: {}",
                    signatures_info.0,
                    signatures_info.1
                );
            } else {
                log::warn!("Ignoring invalid block approval from {}", sender);
            }
        } else {
            log::warn!("Received BlockApproval from untrusted validator.");
        }

        Ok(())
    }

    pub async fn run(&self) -> Result<(), CasperError> {
        log::info!(
            "Starting execution of ApprovedBlockProtocol. \
            Waiting for {} approvals from genesis validators.",
            self.required_sigs
        );

        if self.required_sigs > 0 {
            self.internal_run().await
        } else {
            log::info!("Self-approving genesis block.");
            self.complete_genesis_ceremony(HashSet::new()).await?;
            log::info!("Finished execution of ApprovedBlockProtocol");
            Ok(())
        }
    }

    pub fn transport(&self) -> &Arc<T> {
        &self.transport
    }

    pub fn conf(&self) -> &Option<Arc<RPConf>> {
        &self.conf
    }
}
