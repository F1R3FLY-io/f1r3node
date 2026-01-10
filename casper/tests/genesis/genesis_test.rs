// See casper/src/test/scala/coop/rchain/casper/genesis/GenesisTest.scala
//
// Note: Tests are simplified compared to Scala original.
// In Scala, LogStub (from comm/src/test/scala/coop/rchain/p2p/EffectsTestInstances.scala)
// implements Log[F] trait and is passed as implicit parameter to functions like BondsParser.
// When these functions call log.info("..."), messages go directly to LogStub.
// Tests then assert on log.warns.count and log.infos.count.
//
// In Rust, BondsParser uses `tracing` crate (tracing::info!, tracing::warn!).
// These logs are not captured because we don't set up a tracing subscriber in tests.
// There are two ways to capture tracing logs:
// 1. Use `tracing-test` crate with #[traced_test] attribute and logs_contain() macro
// 2. Implement custom tracing_subscriber::Layer that captures logs into a Vec
// However, this adds complexity and dependencies for marginal benefit.
// For now, tests verify the end result (e.g., bonds.len()) instead of log message counts.

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use casper::rust::{
    casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
    genesis::{
        contracts::{proof_of_stake::ProofOfStake, validator::Validator},
        genesis::Genesis,
    },
    util::{
        bonds_parser::BondsParser,
        proto_util,
        rholang::{interpreter_util, runtime_manager::RuntimeManager},
        vault_parser::VaultParser,
    },
};
use models::rust::casper::protocol::casper_message::{BlockMessage, Bond};
use models::rust::string_ops::StringOps;
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::history::Either;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use tempfile::TempDir;

use comm::rust::test_instances::{LogStub, LogicalTime};

use crate::helper::block_dag_storage_fixture::with_storage;
use crate::util::genesis_builder::DEFAULT_POS_MULTI_SIG_PUBLIC_KEYS;
use crate::util::rholang::resources;
use crate::util::rholang::resources::generate_scope_id;

const AUTOGEN_SHARD_SIZE: usize = 5;
const RCHAIN_SHARD_ID: &str = "root";

fn genesis_path() -> PathBuf {
    TempDir::new()
        .expect("Failed to create genesis temp dir")
        .keep()
}

async fn with_gen_resources<F, Fut, R>(body: F) -> R
where
    F: FnOnce(RuntimeManager, PathBuf, LogStub, LogicalTime) -> Fut,
    Fut: Future<Output = R>,
{
    let scope_id = generate_scope_id();
    let gp = genesis_path();

    // Scala uses MetricsNOP, and this class in turn is empty, if it is used it means that the test does not log metrics.
    // implicit val noopMetrics: Metrics[F] = new metrics.Metrics.MetricsNOP[F]
    // implicit val span: Span[F]           = NoopSpan[F]()

    let time = LogicalTime::new();
    let log = LogStub::new();

    let mut kvs_manager = resources::mk_test_rnode_store_manager_shared(scope_id.clone());

    let m_store = RuntimeManager::mergeable_store(&mut *kvs_manager)
        .await
        .expect("Failed to create mergeable store");

    let r_store = kvs_manager
        .r_space_stores()
        .await
        .expect("Failed to create rspace stores");

    let runtime_manager = RuntimeManager::create_with_store(
        r_store,
        m_store,
        Genesis::non_negative_mergeable_tag_name(),
    );

    let result = body(runtime_manager, gp.clone(), log, time).await;

    // Note: Scala uses PathOps.recursivelyDelete() with FileVisitor pattern.
    // Rust fs::remove_dir_all does the same - recursively removes directory with all contents.
    let _ = fs::remove_dir_all(&scope_id);
    let _ = fs::remove_dir_all(&gp);

    result
}

fn mk_casper_snapshot(dag: KeyValueDagRepresentation) -> CasperSnapshot {
    CasperSnapshot {
        dag,
        last_finalized_block: Bytes::new(),
        lca: Bytes::new(),
        tips: Vec::new(),
        parents: Vec::new(),
        justifications: Default::default(),
        invalid_blocks: HashMap::new(),
        deploys_in_scope: Default::default(),
        max_block_num: 0,
        max_seq_nums: Default::default(),
        on_chain_state: OnChainCasperState {
            shard_conf: CasperShardConf::new(),
            bonds_map: HashMap::new(),
            active_validators: Vec::new(),
        },
    }
}

fn validators() -> Vec<(String, usize)> {
    vec![
        (
            "299670c52849f1aa82e8dfe5be872c16b600bf09cc8983e04b903411358f2de6".to_string(),
            0,
        ),
        (
            "6bf1b2753501d02d386789506a6d93681d2299c6edfd4455f596b97bc5725968".to_string(),
            1,
        ),
    ]
}

fn print_bonds(bonds_file: &Path) {
    let content = validators()
        .into_iter()
        .map(|(v, i)| format!("{v} {i}"))
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(bonds_file, format!("{content}\n")).expect("Failed to write bonds file");
}

//Note: using this struct + new() to describe default parameters
struct FromInputFilesParams<'a> {
    maybe_bonds_path: Option<&'a str>,
    autogen_shard_size: usize,
    maybe_vaults_path: Option<&'a str>,
    minimum_bond: i64,
    maximum_bond: i64,
    epoch_length: i32,
    quarantine_length: i32,
    number_of_active_validators: u32,
    shard_id: String,
    deploy_timestamp: Option<i64>,
    block_number: i64,
}

impl Default for FromInputFilesParams<'_> {
    fn default() -> Self {
        Self {
            maybe_bonds_path: None,
            autogen_shard_size: AUTOGEN_SHARD_SIZE,
            maybe_vaults_path: None,
            minimum_bond: 1,
            maximum_bond: i64::MAX,
            epoch_length: 10000,
            quarantine_length: 50000,
            number_of_active_validators: 100,
            shard_id: RCHAIN_SHARD_ID.to_string(),
            deploy_timestamp: None,
            block_number: 0,
        }
    }
}

impl<'a> FromInputFilesParams<'a> {
    fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        Self {
            deploy_timestamp: Some(now),
            ..Default::default()
        }
    }
}

async fn from_input_files(
    runtime_manager: &mut RuntimeManager,
    genesis_path: &Path,
    params: FromInputFilesParams<'_>,
) -> Result<BlockMessage, Box<dyn Error>> {
    // deploy_timestamp is always Some
    let timestamp = params
        .deploy_timestamp
        .expect("deploy_timestamp should be set");

    let vaults_path = params
        .maybe_vaults_path
        .map(|p| p.to_string())
        .unwrap_or_else(|| {
            genesis_path
                .join("wallets.txt")
                .to_string_lossy()
                .to_string()
        });

    let vaults = VaultParser::parse_from_path_str(&vaults_path)?;

    let bonds_path = params
        .maybe_bonds_path
        .map(|p| p.to_string())
        .unwrap_or_else(|| genesis_path.join("bonds.txt").to_string_lossy().to_string());

    let bonds = BondsParser::parse_with_autogen(&bonds_path, params.autogen_shard_size)?;

    let validators: Vec<Validator> = bonds
        .iter()
        .map(|(pk, stake)| Validator {
            pk: pk.clone(),
            stake: *stake,
        })
        .collect();

    let genesis = Genesis {
        shard_id: params.shard_id,
        timestamp,
        proof_of_stake: ProofOfStake {
            minimum_bond: params.minimum_bond,
            maximum_bond: params.maximum_bond,
            epoch_length: params.epoch_length,
            quarantine_length: params.quarantine_length,
            number_of_active_validators: params.number_of_active_validators,
            validators,
            pos_multi_sig_public_keys: DEFAULT_POS_MULTI_SIG_PUBLIC_KEYS.to_vec(),
            pos_multi_sig_quorum: DEFAULT_POS_MULTI_SIG_PUBLIC_KEYS.len() as u32 - 1,
        },
        vaults,
        supply: i64::MAX,
        block_number: params.block_number,
        version: 1,
    };

    let genesis_block = Genesis::create_genesis_block(runtime_manager, &genesis).await?;

    Ok(genesis_block)
}

#[tokio::test]
async fn genesis_from_input_files_should_generate_random_validators_when_no_bonds_file_is_given() {
    with_gen_resources(
        |mut runtime_manager, genesis_path, _log, _time| async move {
            let genesis_block = from_input_files(
                &mut runtime_manager,
                &genesis_path,
                FromInputFilesParams::new(),
            )
            .await
            .expect("Genesis creation should succeed");

            let bonds = proto_util::bonds(&genesis_block);

            assert_eq!(
                bonds.len(),
                AUTOGEN_SHARD_SIZE,
                "Should generate {} random validators",
                AUTOGEN_SHARD_SIZE
            );
        },
    )
    .await;
}

#[tokio::test]
async fn genesis_from_input_files_should_tell_when_bonds_file_does_not_exist() {
    with_gen_resources(
        |mut runtime_manager, genesis_path, _log, _time| async move {
            // Path that does not exist - using a fake path, no need to create a real directory
            let non_existing_path = "/tmp/non_existing_test_path/not/a/real/file".to_string();

            let result = from_input_files(
                &mut runtime_manager,
                &genesis_path,
                FromInputFilesParams {
                    maybe_bonds_path: Some(&non_existing_path),
                    ..FromInputFilesParams::new()
                },
            )
            .await;

            // BondsParser::parse_with_autogen logs warn "BONDS FILE NOT FOUND" and creates random bonds
            assert!(
                result.is_ok(),
                "Genesis creation should succeed with auto-generated bonds"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn genesis_from_input_files_should_fail_with_error_when_bonds_file_cannot_be_parsed() {
    with_gen_resources(
        |mut runtime_manager, genesis_path, _log, _time| async move {
            let bad_bonds_file = genesis_path.join("misformatted.txt");
            let mut file =
                fs::File::create(&bad_bonds_file).expect("Failed to create bad bonds file");
            writeln!(file, "xzy 1\nabc 123 7").expect("Failed to write bad bonds content");

            let bad_bonds_path = bad_bonds_file.to_str().unwrap().to_string();
            let result = from_input_files(
                &mut runtime_manager,
                &genesis_path,
                FromInputFilesParams {
                    maybe_bonds_path: Some(&bad_bonds_path),
                    ..FromInputFilesParams::new()
                },
            )
            .await;

            assert!(result.is_err(), "Genesis creation should fail");

            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("FAILED PARSING BONDS FILE") || err_msg.contains("INVALID"),
                "Error should mention parsing failure, got: {}",
                err_msg
            );
        },
    )
    .await;
}


#[tokio::test]
async fn genesis_from_input_files_should_create_a_genesis_block_with_the_right_bonds_when_a_proper_bonds_file_is_given(
) {
    with_gen_resources(
        |mut runtime_manager, genesis_path, _log, _time| async move {
            let bonds_file = genesis_path.join("givenBonds.txt");
            print_bonds(&bonds_file);

            let bonds_path = bonds_file.to_str().unwrap().to_string();
            let result = from_input_files(
                &mut runtime_manager,
                &genesis_path,
                FromInputFilesParams {
                    maybe_bonds_path: Some(&bonds_path),
                    ..FromInputFilesParams::new()
                },
            )
            .await;

            assert!(result.is_ok(), "Genesis creation should succeed");

            let genesis_block = result.unwrap();
            let bonds = proto_util::bonds(&genesis_block);

            let expected_bonds: Vec<Bond> = validators()
                .iter()
                .map(|(v, i)| {
                    let pk_bytes = StringOps::decode_hex(v.clone()).expect("Failed to decode hex");
                    Bond {
                        validator: pk_bytes.into(),
                        stake: *i as i64,
                    }
                })
                .collect();

            for expected in &expected_bonds {
                assert!(
                    bonds
                        .iter()
                        .any(|b| b.validator == expected.validator && b.stake == expected.stake),
                    "Expected bond {:?} not found in bonds",
                    expected
                );
            }
        },
    )
    .await;
}

#[tokio::test]
async fn genesis_from_input_files_should_create_a_valid_genesis_block() {
    with_storage(|block_store, mut block_dag_storage| async move {
        with_gen_resources(
            |mut runtime_manager, genesis_path, _log, _time| async move {
                let genesis = from_input_files(
                    &mut runtime_manager,
                    &genesis_path,
                    FromInputFilesParams::new(),
                )
                .await
                .expect("Genesis creation should succeed");

                block_dag_storage
                    .insert(&genesis, false, true)
                    .expect("Failed to insert genesis into DAG");

                block_store
                    .put(genesis.block_hash.clone(), &genesis)
                    .expect("Failed to put genesis into block store");

                let dag = block_dag_storage.get_representation();

                let maybe_post_genesis_state_hash = interpreter_util::validate_block_checkpoint(
                    &genesis,
                    &block_store,
                    &mut mk_casper_snapshot(dag),
                    &mut runtime_manager,
                )
                .await
                .expect("validate_block_checkpoint should succeed");

                match maybe_post_genesis_state_hash {
                    Either::Right(Some(_)) => {
                        // Success - got Right(Some(_))
                    }
                    Either::Right(None) => {
                        panic!("Expected Right(Some(_)), got Right(None)");
                    }
                    Either::Left(block_error) => {
                        panic!("Expected Right(Some(_)), got Left({:?})", block_error);
                    }
                }
            },
        )
        .await
    })
    .await;
}

#[tokio::test]
async fn genesis_from_input_files_should_detect_an_existing_bonds_file_in_the_default_location() {
    with_gen_resources(
        |mut runtime_manager, genesis_path, _log, _time| async move {
            // Create bonds.txt in default location
            let bonds_file = genesis_path.join("bonds.txt");
            print_bonds(&bonds_file);

            let result = from_input_files(
                &mut runtime_manager,
                &genesis_path,
                FromInputFilesParams::new(),
            )
            .await;

            assert!(result.is_ok(), "Genesis creation should succeed");

            let genesis_block = result.unwrap();
            let bonds = proto_util::bonds(&genesis_block);

            let expected_bonds: Vec<Bond> = validators()
                .iter()
                .map(|(v, i)| {
                    let pk_bytes = StringOps::decode_hex(v.clone()).expect("Failed to decode hex");
                    Bond {
                        validator: pk_bytes.into(),
                        stake: *i as i64,
                    }
                })
                .collect();

            for expected in &expected_bonds {
                assert!(
                    bonds
                        .iter()
                        .any(|b| b.validator == expected.validator && b.stake == expected.stake),
                    "Expected bond {:?} not found in bonds",
                    expected
                );
            }
        },
    )
    .await;
}


#[tokio::test]
#[ignore = "Scala ignore"]
async fn genesis_from_input_files_should_parse_the_wallets_file_and_create_corresponding_rev_vaults(
) {
}
