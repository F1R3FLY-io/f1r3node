// See casper/src/main/scala/coop/rchain/casper/genesis/Genesis.scala

use crypto::rust::signatures::signed::Signed;
use models::{
    rhoapi::{g_unforgeable::UnfInstance, GPrivate, GUnforgeable, Par},
    rust::{
        block::state_hash::StateHash,
        casper::{
            protocol::casper_message::{
                BlockMessage, Body, Bond, DeployData, F1r3flyState, ProcessedDeploy,
            },
        },
    },
};
use prost::bytes::Bytes;

use crate::rust::{
    errors::CasperError,
    util::{
        proto_util,
        rholang::{runtime_manager::RuntimeManager, tools::Tools},
    },
};

use super::contracts::{proof_of_stake::ProofOfStake, standard_deploys, vault::Vault};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Genesis {
    pub shard_id: String,
    pub timestamp: i64,
    pub block_number: i64,
    pub proof_of_stake: ProofOfStake,
    pub vaults: Vec<Vault>,
    pub supply: i64,
    pub version: i64,
}

impl Genesis {
    pub fn non_negative_mergeable_tag_name() -> Par {
        let mut rng = Tools::unforgeable_name_rng(
            &standard_deploys::NON_NEGATIVE_NUMBER_PUB_KEY,
            standard_deploys::NON_NEGATIVE_NUMBER_TIMESTAMP,
        );

        rng.next();
        let unforgeable_byte = rng.next();

        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: unforgeable_byte.into_iter().map(|b| b as u8).collect(),
            })),
        }])
    }

    pub fn default_blessed_terms_with_timestamp(
        timestamp: i64,
        pos_params: &ProofOfStake,
        vaults: &Vec<Vault>,
        supply: i64,
        shard_id: &str,
    ) -> Vec<Signed<DeployData>> {
        // Splits initial vaults creation in multiple deploys (batches)
        const BATCH_SIZE: usize = 100;

        // Create vault deploys only if vaults are not empty
        let mut vault_deploys = Vec::new();
        if !vaults.is_empty() {
            let batch_count = (vaults.len() + BATCH_SIZE - 1) / BATCH_SIZE;
            vault_deploys.reserve(batch_count);

            for (idx, chunk) in vaults.chunks(BATCH_SIZE).enumerate() {
                let is_last_batch = idx == batch_count - 1;
                let deploy_timestamp = timestamp + idx as i64;

                let batch_vaults = chunk.to_vec();

                let deploy = standard_deploys::vaults_generator(
                    batch_vaults,
                    supply,
                    deploy_timestamp,
                    is_last_batch,
                    shard_id,
                );

                vault_deploys.push(deploy);
            }
        }

        // Order of deploys is important for Registry to work correctly
        // - dependencies must be defined first in the list
        let registry = standard_deploys::registry(shard_id);
        let list_ops = standard_deploys::list_ops(shard_id);
        let either = standard_deploys::either(shard_id);
        let non_negative_number = standard_deploys::non_negative_number(shard_id);
        let make_mint = standard_deploys::make_mint(shard_id);
        let auth_key = standard_deploys::auth_key(shard_id);
        let system_vault = standard_deploys::system_vault(shard_id);
        let multi_sig_system_vault = standard_deploys::multi_sig_system_vault(shard_id);
        let stack = standard_deploys::stack(shard_id);
        let pos_generator = standard_deploys::pos_generator(&pos_params, shard_id);

        let mut all_deploys = Vec::with_capacity(10 + vault_deploys.len());
        all_deploys.push(registry);
        all_deploys.push(list_ops);
        all_deploys.push(either);
        all_deploys.push(non_negative_number);
        all_deploys.push(make_mint);
        all_deploys.push(auth_key);
        all_deploys.push(system_vault);
        all_deploys.push(multi_sig_system_vault);
        all_deploys.push(stack);
        all_deploys.extend(vault_deploys);
        all_deploys.push(pos_generator);

        all_deploys
    }

    pub fn default_blessed_terms(
        pos_params: &ProofOfStake,
        vaults: &Vec<Vault>,
        supply: i64,
        shard_id: &str,
    ) -> Vec<Signed<DeployData>> {
        // Use hardcoded timestamp for backwards compatibility
        const BASE_TIMESTAMP: i64 = 1565818101792;
        Self::default_blessed_terms_with_timestamp(
            BASE_TIMESTAMP,
            pos_params,
            vaults,
            supply,
            shard_id,
        )
    }

    pub async fn create_genesis_block(
        runtime_manager: &mut RuntimeManager,
        genesis: &Genesis,
    ) -> Result<BlockMessage, CasperError> {
        let blessed_terms = Self::default_blessed_terms(
            &genesis.proof_of_stake,
            &genesis.vaults,
            genesis.supply,
            &genesis.shard_id,
        );

        let (start_hash, state_hash, processed_deploys) = runtime_manager
            .compute_genesis(blessed_terms, genesis.timestamp, genesis.block_number)
            .await?;

        let block_message =
            Self::create_processed_deploy(genesis, start_hash, state_hash, processed_deploys);

        Ok(block_message)
    }

    fn create_processed_deploy(
        genesis: &Genesis,
        start_hash: StateHash,
        state_hash: StateHash,
        processed_deploys: Vec<ProcessedDeploy>,
    ) -> BlockMessage {
        let state = F1r3flyState {
            pre_state_hash: start_hash,
            post_state_hash: state_hash,
            block_number: genesis.block_number,
            bonds: Self::bonds_proto(&genesis.proof_of_stake),
        };

        let failed_deploys: Vec<_> = processed_deploys
            .iter()
            .filter(|deploy| deploy.is_failed)
            .collect();

        assert!(failed_deploys.is_empty(), "Failed deploys found");

        let sorted_deploys = processed_deploys
            .into_iter()
            .filter(|deploy| !deploy.is_failed)
            .map(|mut deploy| {
                use prost::Message;
                deploy.deploy_log.sort_by(|a, b| {
                    let a_bytes = a.to_proto().encode_to_vec();
                    let b_bytes = b.to_proto().encode_to_vec();
                    a_bytes.cmp(&b_bytes)
                });
                deploy
            })
            .collect();

        let body = Body {
            state,
            deploys: sorted_deploys,
            rejected_deploys: Vec::new(),
            system_deploys: Vec::new(),
            extra_bytes: Bytes::new(),
        };

        let header = proto_util::block_header(Vec::new(), genesis.version, genesis.timestamp);
        proto_util::unsigned_block_proto(body, header, Vec::new(), genesis.shard_id.clone(), None)
    }

    fn bonds_proto(proof_of_stake: &ProofOfStake) -> Vec<Bond> {
        let mut bonds: Vec<_> = proof_of_stake
            .validators
            .iter()
            .map(|validator| (validator.pk.clone(), validator.stake))
            .collect();

        bonds.sort_by_key(|(pk, _)| pk.bytes.clone());

        bonds
            .into_iter()
            .map(|(pk, stake)| Bond {
                validator: pk.bytes.into(),
                stake,
            })
            .collect()
    }
}
