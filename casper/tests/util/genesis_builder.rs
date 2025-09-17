// See casper/src/test/scala/coop/rchain/casper/util/GenesisBuilder.scala

use dashmap::DashMap;
use lazy_static::lazy_static;
use std::{collections::HashMap, path::PathBuf};
use tempfile::Builder;

use block_storage::rust::{
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    key_value_block_store::KeyValueBlockStore,
};

use casper::rust::{
    errors::CasperError,
    genesis::{
        contracts::{proof_of_stake::ProofOfStake, validator::Validator, vault::Vault},
        genesis::Genesis,
    },
    util::{
        construct_deploy::{DEFAULT_PUB, DEFAULT_PUB2, DEFAULT_SEC, DEFAULT_SEC2},
        rholang::runtime_manager::RuntimeManager,
    },
};
use crypto::rust::{
    hash::blake2b256::Blake2b256,
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use models::rust::casper::protocol::casper_message::{
    BlockMessage, Body, Bond, F1r3flyState, Header,
};
use prost::bytes;
use rholang::rust::interpreter::util::rev_address::RevAddress;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;

use crate::util::rholang::resources::mk_test_rnode_store_manager;

type GenesisParameters = (
    Vec<(PrivateKey, PublicKey)>,
    Vec<(PrivateKey, PublicKey)>,
    Genesis,
);

lazy_static! {

  static ref DEFAULT_VALIDATOR_KEY_PAIRS: [(PrivateKey, PublicKey); 4] = {
    std::array::from_fn(|_| {
      let secp = Secp256k1;
      let (secret_key, public_key) = secp.new_key_pair();
      (secret_key, public_key)
    })
  };

  // Equivalent to defaultValidatorSks in Scala
  pub static ref DEFAULT_VALIDATOR_SKS: [PrivateKey; 4] = {
    std::array::from_fn(|i| DEFAULT_VALIDATOR_KEY_PAIRS[i].0.clone())
  };

  // Equivalent to defaultValidatorPks in Scala
  pub static ref DEFAULT_VALIDATOR_PKS: [PublicKey; 4] = {
    std::array::from_fn(|i| DEFAULT_VALIDATOR_KEY_PAIRS[i].1.clone())
  };

  static ref DEFAULT_POS_MULTI_SIG_PUBLIC_KEYS: [String; 3] = [
      "04db91a53a2b72fcdcb201031772da86edad1e4979eb6742928d27731b1771e0bc40c9e9c9fa6554bdec041a87cee423d6f2e09e9dfb408b78e85a4aa611aad20c".to_string(),
      "042a736b30fffcc7d5a58bb9416f7e46180818c82b15542d0a7819d1a437aa7f4b6940c50db73a67bfc5f5ec5b5fa555d24ef8339b03edaa09c096de4ded6eae14".to_string(),
      "047f0f0f5bbe1d6d1a8dac4d88a3957851940f39a57cd89d55fe25b536ab67e6d76fd3f365c83e5bfe11fe7117e549b1ae3dd39bfc867d1c725a4177692c4e7754".to_string(),
  ];
}

pub struct GenesisBuilder {
    genesis_cache: DashMap<GenesisParameters, GenesisContext>,
    cache_accesses: u64,
    cache_misses: u64,
}

impl GenesisBuilder {
    pub fn new() -> Self {
        Self {
            genesis_cache: DashMap::new(),
            cache_accesses: 0,
            cache_misses: 0,
        }
    }

    pub fn create_bonds(validators: Vec<PublicKey>) -> HashMap<PublicKey, i64> {
        validators
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, (i as i64) * 2 + 1))
            .collect()
    }

    /// Lightweight test genesis creation following Scala approach pattern:
    /// buildGenesis(buildGenesisParameters(validatorKeyPairs, createBonds(validatorKeyPairs.map(_._2))))
    /// but using a simplified approach for testing to avoid heavy infrastructure
    pub fn build_test_genesis(validator_key_pairs: Vec<(PrivateKey, PublicKey)>) -> BlockMessage {
        // Extract validator public keys (equivalent to validatorKeyPairs.map(_._2))
        let validator_pks: Vec<PublicKey> = validator_key_pairs
            .iter()
            .map(|(_, pk)| pk.clone())
            .collect();

        // Create bonds using GenesisBuilder.createBonds logic (equivalent to createBonds(validatorKeyPairs.map(_._2)))
        let bonds_map = Self::create_bonds(validator_pks);

        // Convert to the Bond format used in genesis block
        let bonds: Vec<Bond> = bonds_map
            .into_iter()
            .map(|(public_key, stake)| Bond {
                validator: public_key.bytes.clone(),
                stake,
            })
            .collect();

        // Create genesis block structure following the buildGenesisParameters pattern
        let state = F1r3flyState {
            pre_state_hash: bytes::Bytes::new(),
            post_state_hash: bytes::Bytes::new(),
            block_number: 0,
            bonds,
        };

        let body = Body {
            state,
            deploys: vec![],
            rejected_deploys: vec![],
            system_deploys: vec![],
            extra_bytes: bytes::Bytes::new(),
        };

        let header = Header {
            parents_hash_list: vec![],
            timestamp: 0, // Using 0 like in GenesisBuilder
            version: 1,
            extra_bytes: bytes::Bytes::new(),
        };

        BlockMessage {
            block_hash: bytes::Bytes::from(Blake2b256::hash(b"test_genesis".to_vec())), // Create a deterministic hash
            header,
            body,
            justifications: vec![],
            sender: bytes::Bytes::new(),
            seq_num: 0,
            sig: bytes::Bytes::new(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "root".to_string(), // Using "root" like in GenesisBuilder
            extra_bytes: bytes::Bytes::new(),
        }
    }

    pub async fn create_genesis(&mut self) -> Result<BlockMessage, CasperError> {
        let context = self.build_genesis_with_parameters(None).await?;
        Ok(context.genesis_block)
    }

    pub fn build_genesis_parameters_with_defaults(
        bonds_function: Option<fn(Vec<PublicKey>) -> HashMap<PublicKey, i64>>,
        validators_num: Option<usize>,
    ) -> GenesisParameters {
        let bonds_function = bonds_function.unwrap_or(Self::create_bonds);
        let validators_num = validators_num.unwrap_or(4);

        Self::build_genesis_parameters(
            DEFAULT_VALIDATOR_KEY_PAIRS
                .iter()
                .take(validators_num)
                .cloned()
                .collect(),
            &bonds_function(
                DEFAULT_VALIDATOR_PKS
                    .iter()
                    .take(validators_num)
                    .cloned()
                    .collect(),
            ),
        )
    }

    pub fn build_genesis_parameters_with_random(
        bonds_function: Option<fn(Vec<PublicKey>) -> HashMap<PublicKey, i64>>,
        validators_num: Option<usize>,
    ) -> GenesisParameters {
        let bonds_function = bonds_function.unwrap_or(Self::create_bonds);
        let validators_num = validators_num.unwrap_or(4);

        // 4 default fixed validators, others are random generated
        let random_validator_key_pairs: Vec<(PrivateKey, PublicKey)> = (5..validators_num)
            .map(|_| Secp256k1.new_key_pair())
            .collect();
        let (_, random_validator_pks): (Vec<PrivateKey>, Vec<PublicKey>) =
            random_validator_key_pairs.iter().cloned().unzip();

        Self::build_genesis_parameters(
            DEFAULT_VALIDATOR_KEY_PAIRS
                .iter()
                .cloned()
                .chain(random_validator_key_pairs.into_iter())
                .collect(),
            &bonds_function(
                DEFAULT_VALIDATOR_PKS
                    .iter()
                    .cloned()
                    .chain(random_validator_pks.into_iter())
                    .collect(),
            ),
        )
    }

    pub fn build_genesis_parameters(
        validator_key_pairs: Vec<(PrivateKey, PublicKey)>,
        bonds: &HashMap<PublicKey, i64>,
    ) -> GenesisParameters {
        let mut genesis_vaults: Vec<(PrivateKey, PublicKey)> = vec![
            (DEFAULT_SEC.clone(), DEFAULT_PUB.clone()),
            (DEFAULT_SEC2.clone(), DEFAULT_PUB2.clone()),
        ];

        let secp = Secp256k1;
        for _ in 3..=validator_key_pairs.len() {
            let (secret_key, public_key) = secp.new_key_pair();
            genesis_vaults.push((secret_key, public_key));
        }

        let vaults: Vec<Vault> = genesis_vaults
            .iter()
            .map(|(_, pk)| Self::predefined_vault(pk))
            .collect::<Vec<Vault>>()
            .into_iter()
            .chain(bonds.iter().map(|(pk, _)| {
                // Initial validator vaults contain 0 Rev
                RevAddress::from_public_key(pk)
                    .map(|rev_address| Vault {
                        rev_address,
                        initial_balance: 0,
                    })
                    .expect("GenesisBuilder: Failed to create rev address")
            }))
            .collect();

        (
            validator_key_pairs,
            genesis_vaults,
            Genesis {
                shard_id: "root".to_string(),
                timestamp: 0,
                proof_of_stake: ProofOfStake {
                    minimum_bond: 1,
                    maximum_bond: i64::MAX,
                    // Epoch length is set to large number to prevent trigger of epoch change
                    // in PoS close block method, which causes block merge conflicts
                    // - epoch change can be set as a parameter in Rholang tests (e.g. PoSSpec)
                    epoch_length: 1000,
                    quarantine_length: 50000,
                    number_of_active_validators: 100,
                    validators: bonds
                        .into_iter()
                        .map(|(pk, stake)| Validator {
                            pk: pk.clone(),
                            stake: *stake,
                        })
                        .collect(),
                    pos_multi_sig_public_keys: DEFAULT_POS_MULTI_SIG_PUBLIC_KEYS.to_vec(),
                    pos_multi_sig_quorum: DEFAULT_POS_MULTI_SIG_PUBLIC_KEYS.len() as i32 - 1,
                },
                vaults,
                supply: i64::MAX,
                block_number: 0,
                version: 1,
            },
        )
    }

    fn predefined_vault(pubkey: &PublicKey) -> Vault {
        Vault {
            rev_address: RevAddress::from_public_key(pubkey)
                .expect("GenesisBuilder: Failed to create rev address"),
            initial_balance: 9000000,
        }
    }

    pub async fn build_genesis_with_parameters(
        &mut self,
        parameters: Option<GenesisParameters>,
    ) -> Result<GenesisContext, CasperError> {
        let parameters =
            parameters.unwrap_or(Self::build_genesis_parameters_with_defaults(None, None));
        self.cache_accesses += 1;

        if self.genesis_cache.contains_key(&parameters) {
            Ok(self.genesis_cache.get(&parameters).unwrap().value().clone())
        } else {
            let context = self.do_build_genesis(&parameters).await?;
            self.genesis_cache.insert(parameters, context.clone());
            Ok(context)
        }
    }

    pub async fn build_genesis_with_validators_num(
        &mut self,
        validators_num: usize,
    ) -> Result<GenesisContext, CasperError> {
        let parameters = Self::build_genesis_parameters_with_random(None, Some(validators_num));
        self.cache_accesses += 1;

        if self.genesis_cache.contains_key(&parameters) {
            Ok(self.genesis_cache.get(&parameters).unwrap().value().clone())
        } else {
            let context = self.do_build_genesis(&parameters).await?;
            self.genesis_cache.insert(parameters, context.clone());
            Ok(context)
        }
    }

    async fn do_build_genesis(
        &mut self,
        parameters: &GenesisParameters,
    ) -> Result<GenesisContext, CasperError> {
        self.cache_misses += 1;
        println!(
            "Genesis block cache miss, building a new genesis. Cache misses: {} / {} ({}%) cache accesses.",
            self.cache_misses,
            self.cache_accesses,
            (self.cache_misses / self.cache_accesses) as f64 * 100.0
        );

        println!("DEBUG: Starting do_build_genesis");
        let (validator_key_pairs, genesis_vaults, genesis_parameters) = parameters;
        println!("DEBUG: Extracted parameters");

        println!("DEBUG: Creating storage directory");
        let storage_directory = Builder::new()
            .prefix("hash-set-casper-test-genesis-")
            .tempdir()
            .expect("Failed to create temporary directory");

        // Convert to a path that won't be automatically deleted
        let storage_directory_path = storage_directory.into_path();
        println!("DEBUG: Storage directory created");

        println!("DEBUG: Creating kvs_manager");
        let mut kvs_manager = mk_test_rnode_store_manager(storage_directory_path.clone());
        println!("DEBUG: Creating r_store");
        let r_store = kvs_manager
            .r_space_stores()
            .await
            .expect("Failed to create RSpaceStore");
        println!("DEBUG: r_store created");

        println!("DEBUG: Creating m_store");
        let m_store = RuntimeManager::mergeable_store(&mut kvs_manager).await?;
        println!("DEBUG: Creating runtime_manager");
        let mut runtime_manager = RuntimeManager::create_with_store(
            r_store,
            m_store,
            Genesis::non_negative_mergeable_tag_name(),
        );
        println!("DEBUG: runtime_manager created");

        println!("DEBUG: About to call Genesis::create_genesis_block");
        let genesis =
            Genesis::create_genesis_block(&mut runtime_manager, genesis_parameters).await?;
        println!("DEBUG: Genesis::create_genesis_block completed");
        let mut block_store = KeyValueBlockStore::create_from_kvm(&mut kvs_manager).await?;
        block_store.put(genesis.block_hash.clone(), &genesis)?;

        let mut block_dag_storage = BlockDagKeyValueStorage::new(&mut kvs_manager).await?;
        block_dag_storage.insert(&genesis, false, true)?;

        // println!(
        //     "genesis_block pre_state_hash: {:?}",
        //     Blake2b256Hash::from_bytes_prost(&genesis.body.state.pre_state_hash)
        // );
        // println!(
        //     "genesis_block post_state_hash: {:?}",
        //     Blake2b256Hash::from_bytes_prost(&genesis.body.state.post_state_hash)
        // );

        Ok(GenesisContext {
            genesis_block: genesis,
            validator_key_pairs: validator_key_pairs.clone(),
            genesis_vaults: genesis_vaults.clone(),
            storage_directory: storage_directory_path,
        })
    }
}

#[derive(Clone)]
pub struct GenesisContext {
    pub genesis_block: BlockMessage,
    pub validator_key_pairs: Vec<(PrivateKey, PublicKey)>,
    pub genesis_vaults: Vec<(PrivateKey, PublicKey)>,
    pub storage_directory: PathBuf,
}

impl GenesisContext {
    pub fn validator_sks(&self) -> Vec<PrivateKey> {
        self.validator_key_pairs
            .iter()
            .map(|(sk, _)| sk.clone())
            .collect()
    }

    pub fn validator_pks(&self) -> Vec<PublicKey> {
        self.validator_key_pairs
            .iter()
            .map(|(_, pk)| pk.clone())
            .collect()
    }

    pub fn genesis_vaults_sks(&self) -> Vec<PrivateKey> {
        self.genesis_vaults
            .iter()
            .map(|(sk, _)| sk.clone())
            .collect()
    }

    pub fn genesis_vaults_pks(&self) -> Vec<PublicKey> {
        self.genesis_vaults
            .iter()
            .map(|(_, pk)| pk.clone())
            .collect()
    }
}
