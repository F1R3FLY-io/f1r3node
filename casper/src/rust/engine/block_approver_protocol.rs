// See casper/src/main/scala/coop/rchain/casper/engine/BlockApproverProtocol.scala

use std::collections::HashMap;
use std::sync::Arc;

use comm::rust::peer_node::PeerNode;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::transport::transport_layer::{Blob, TransportLayer};
use crypto::rust::hash::blake2b256::Blake2b256;
use log::{info, warn};
use models::rust::casper::protocol::casper_message::{
    ApprovedBlockCandidate, BlockApproval, ProcessedDeploy, ProcessedSystemDeploy, UnapprovedBlock,
};
use models::rust::casper::protocol::packet_type_tag::ToPacket;
use prost::bytes::Bytes;
use prost::Message;

use crate::rust::errors::CasperError;
use crate::rust::genesis::contracts::{
    proof_of_stake::ProofOfStake, validator::Validator, vault::Vault,
};
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use crate::rust::validator_identity::ValidatorIdentity;

/// Rust port of `coop.rchain.casper.engine.BlockApproverProtocol` from Scala.
/// The field layout and logic mirror the original as closely as possible.
#[derive(Clone)]
pub struct BlockApproverProtocol<T: TransportLayer + Send + Sync + 'static> {
    // Configuration / static data
    validator_id: ValidatorIdentity,
    deploy_timestamp: i64,
    vaults: Vec<Vault>,
    bonds: HashMap<crypto::rust::public_key::PublicKey, i64>,
    bonds_bytes: HashMap<Bytes, i64>, // helper map keyed by raw bytes
    minimum_bond: i64,
    maximum_bond: i64,
    epoch_length: i32,
    quarantine_length: i32,
    number_of_active_validators: i32,
    required_sigs: i32,
    pos_multi_sig_public_keys: Vec<String>,
    pos_multi_sig_quorum: i32,

    // Infrastructure
    transport: Arc<T>,
    conf: Arc<RPConf>,
}

impl<T: TransportLayer + Send + Sync + 'static> BlockApproverProtocol<T> {
    /// Corresponds to Scala `BlockApproverProtocol.of` – constructor with basic validation.
    pub fn new(
        validator_id: ValidatorIdentity,
        deploy_timestamp: i64,
        vaults: Vec<Vault>,
        bonds: HashMap<crypto::rust::public_key::PublicKey, i64>,
        minimum_bond: i64,
        maximum_bond: i64,
        epoch_length: i32,
        quarantine_length: i32,
        number_of_active_validators: i32,
        required_sigs: i32,
        pos_multi_sig_public_keys: Vec<String>,
        pos_multi_sig_quorum: i32,
        transport: Arc<T>,
        conf: Arc<RPConf>,
    ) -> Result<Self, CasperError> {
        if bonds.len() <= required_sigs as usize {
            return Err(CasperError::RuntimeError(format!(
                "Required sigs ({}) must be smaller than the number of bonded validators ({})",
                required_sigs,
                bonds.len()
            )));
        }

        let bonds_bytes: HashMap<Bytes, i64> = bonds
            .iter()
            .map(|(pk, stake)| (pk.bytes.clone(), *stake))
            .collect();

        Ok(Self {
            validator_id,
            deploy_timestamp,
            vaults,
            bonds,
            bonds_bytes,
            minimum_bond,
            maximum_bond,
            epoch_length,
            quarantine_length,
            number_of_active_validators,
            required_sigs,
            pos_multi_sig_public_keys,
            pos_multi_sig_quorum,
            transport,
            conf,
        })
    }

    /// Corresponds to Scala `BlockApproverProtocol.getBlockApproval` / `getApproval` –
    /// signs candidate ApprovedBlockCandidate and creates `BlockApproval`.
    pub fn get_block_approval(&self, candidate: &ApprovedBlockCandidate) -> BlockApproval {
        let sig_data = Blake2b256::hash(candidate.clone().to_proto().encode_to_vec());
        let sig = self.validator_id.signature(&sig_data);
        BlockApproval {
            candidate: candidate.clone(),
            sig,
        }
    }

    /// Corresponds to Scala `BlockApproverProtocol.validateCandidate` –
    /// performs full validation of the candidate genesis block.
    async fn validate_candidate(
        &self,
        runtime_manager: &mut RuntimeManager,
        candidate: &ApprovedBlockCandidate,
        shard_id: &str,
    ) -> Result<(), String> {
        // Basic checks – required sigs, absence of system deploys, bonds equality
        if candidate.required_sigs != self.required_sigs {
            return Err("Candidate didn't have required signatures number.".to_string());
        }

        let block = &candidate.block;
        if !block.body.system_deploys.is_empty() {
            return Err("Candidate must not contain system deploys.".to_string());
        }

        let block_bonds: HashMap<Bytes, i64> = block
            .body
            .state
            .bonds
            .iter()
            .map(|b| (b.validator.clone(), b.stake))
            .collect();

        if block_bonds != self.bonds_bytes {
            return Err("Block bonds don't match expected.".to_string());
        }

        // Prepare PoS params
        let validators: Vec<Validator> = block_bonds
            .iter()
            .map(|(pk_bytes, stake)| Validator {
                pk: crypto::rust::public_key::PublicKey::new(pk_bytes.clone()),
                stake: *stake,
            })
            .collect();

        let pos_params = ProofOfStake {
            minimum_bond: self.minimum_bond,
            maximum_bond: self.maximum_bond,
            validators,
            epoch_length: self.epoch_length,
            quarantine_length: self.quarantine_length,
            number_of_active_validators: self.number_of_active_validators,
            pos_multi_sig_public_keys: self.pos_multi_sig_public_keys.clone(),
            pos_multi_sig_quorum: self.pos_multi_sig_quorum,
        };

        // Expected blessed contracts
        let genesis_blessed_contracts =
            crate::rust::genesis::genesis::Genesis::default_blessed_terms_with_timestamp(
                self.deploy_timestamp,
                &pos_params,
                &self.vaults,
                i64::MAX,
                shard_id,
            );

        let block_deploys: &Vec<ProcessedDeploy> = &block.body.deploys;

        if block_deploys.len() != genesis_blessed_contracts.len() {
            return Err(
                "Mismatch between number of candidate deploys and expected number of deploys."
                    .to_string(),
            );
        }

        // Check deploys equality (order matters)
        let wrong_deploys: Vec<String> = block_deploys
            .iter()
            .zip(genesis_blessed_contracts.iter())
            .filter(|(candidate_deploy, expected_contract)| {
                candidate_deploy.deploy.data.term != expected_contract.data.term
            })
            .map(|(candidate_deploy, _)| {
                let term = &candidate_deploy.deploy.data.term;
                term.chars().take(100).collect::<String>()
            })
            .take(5)
            .collect();

        if !wrong_deploys.is_empty() {
            return Err(format!(
                "Genesis candidate deploys do not match expected blessed contracts.\nBad contracts (5 first):\n{}",
                wrong_deploys.join("\n")
            ));
        }

        // State hash checks
        let empty_state_hash = RuntimeManager::empty_state_hash_fixed();
        let state_hash = runtime_manager
            .replay_compute_state(
                &empty_state_hash,
                block_deploys.clone(),
                Vec::<ProcessedSystemDeploy>::new(),
                &rholang::rust::interpreter::system_processes::BlockData::from_block(block),
                None,
                true,
            )
            .await
            .map_err(|e| format!("Failed status during replay: {:?}.", e))?;

        if state_hash != block.body.state.post_state_hash {
            return Err("Tuplespace hash mismatch.".to_string());
        }

        // Bonds computed from tuplespace
        let tuplespace_bonds = runtime_manager
            .compute_bonds(&block.body.state.post_state_hash)
            .await
            .map_err(|e| format!("{:?}", e))?;

        let tuplespace_bonds_map: HashMap<Bytes, i64> = tuplespace_bonds
            .into_iter()
            .map(|b| (b.validator, b.stake))
            .collect();

        if tuplespace_bonds_map != self.bonds_bytes {
            return Err("Tuplespace bonds don't match expected ones.".to_string());
        }

        Ok(())
    }

    /// Corresponds to Scala `BlockApproverProtocol.unapprovedBlockPacketHandler` –
    /// verifies candidate message from peer and streams approval if valid.
    pub async fn unapproved_block_packet_handler(
        &self,
        runtime_manager: &mut RuntimeManager,
        peer: &PeerNode,
        unapproved_block: UnapprovedBlock,
        shard_id: &str,
    ) -> Result<(), CasperError> {
        let candidate = unapproved_block.candidate.clone();
        info!(
            "Received expected genesis block candidate from {}. Verifying...",
            peer.endpoint.host
        );

        match self
            .validate_candidate(runtime_manager, &candidate, shard_id)
            .await
        {
            Ok(_) => {
                let approval = self.get_block_approval(&candidate);
                let packet = approval.to_proto().mk_packet();
                let blob = Blob {
                    sender: self.conf.local.clone(),
                    packet,
                };

                self.transport.stream(peer, &blob).await.map_err(|e| {
                    CasperError::RuntimeError(format!(
                        "Failed to stream BlockApproval to peer: {}",
                        e
                    ))
                })?;

                info!(
                    "Approved genesis block candidate from {}. Approval sent in response.",
                    peer.endpoint.host
                );
            }
            Err(err_msg) => {
                warn!(
                    "Received unexpected genesis block candidate from {} because: {}",
                    peer.endpoint.host, err_msg
                );
            }
        }

        Ok(())
    }
}
