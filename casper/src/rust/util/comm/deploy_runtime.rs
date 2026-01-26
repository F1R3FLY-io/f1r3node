use std::{
    process::exit,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::rust::util::comm::{
    grpc_deploy_service::DeployService,
    grpc_propose_service::ProposeService,
    listen_at_name::{self, Name},
    ServiceResult,
};
use crypto::rust::{private_key::PrivateKey, public_key::PublicKey, signatures::signed::Signed};
use futures::FutureExt;
use models::casper::{
    BlockQuery, BlocksQuery, BondStatusQuery, ContinuationAtNameQuery, DataAtNameQuery,
    FindDeployQuery, IsFinalizedQuery, MachineVerifyQuery, VisualizeDagQuery,
};
use models::rust::casper::protocol::casper_message::DeployData;
use prost::bytes::Bytes;

pub struct DeployRuntime;

impl DeployRuntime {
    /// Like Scala: gracefulExit(program: F[Either[Seq[String], String]])
    async fn graceful_exit<F>(program: F)
    where
        F: std::future::Future<Output = ServiceResult<String>>,
    {
        let attempted = program.await;

        // Note: usage of the sync IO is not a good practice in the async context. The reason of such impl
        // is because ProposeService API is async and that DeployRuntime is used only in CLI context where it will not influence the performance.
        match attempted {
            Ok(msg) => {
                println!("{msg}");
            }
            Err(errors) => {
                for e in errors {
                    eprintln!("{e}");
                }
                exit(1);
            }
        }
    }

    pub async fn propose<S: ProposeService>(mut svc: S, print_unmatched_sends: bool) {
        Self::graceful_exit(
            svc.propose(print_unmatched_sends)
                .map(|res| res.map(|rs| format!("Response: {rs}"))),
        )
        .await
    }

    pub async fn get_block<S: DeployService>(svc: &mut S, hash: String) {
        Self::graceful_exit(svc.get_block(BlockQuery { hash })).await
    }

    pub async fn get_blocks<S: DeployService>(svc: &mut S, depth: i32) {
        Self::graceful_exit(svc.get_blocks(BlocksQuery { depth })).await
    }

    pub async fn visualize_dag<S: DeployService>(
        svc: &mut S,
        depth: i32,
        show_justification_lines: bool,
    ) {
        Self::graceful_exit(svc.visualize_dag(VisualizeDagQuery {
            depth,
            show_justification_lines,
            start_block_number: 0,
        }))
        .await
    }

    pub async fn machine_verifiable_dag<S: DeployService>(svc: &mut S) {
        Self::graceful_exit(svc.machine_verifiable_dag(MachineVerifyQuery {})).await
    }

    pub async fn listen_for_data_at_name<S: DeployService + Clone>(svc: &mut S, name: Name) {
        // Use graceful_exit like other methods, but with polling logic
        Self::graceful_exit(async {
            listen_at_name::listen_at_name_until_changes(name, |par| {
                let mut value = svc.clone();
                async move {
                    value
                        .listen_for_data_at_name(DataAtNameQuery {
                            depth: i32::MAX,
                            name: Some(par.clone()),
                        })
                        .await
                }
            })
            .await
            .map(|res| format!("{:?}", res))
        })
        .await
    }

    pub async fn listen_for_continuation_at_name<S: DeployService + Clone>(
        svc: &mut S,
        names: Vec<Name>,
    ) {
        // Use graceful_exit like other methods, but with polling logic
        Self::graceful_exit(async {
            listen_at_name::listen_at_names_until_changes(names, |pars| {
                let mut value = svc.clone();
                async move {
                    value
                        .listen_for_continuation_at_name(ContinuationAtNameQuery {
                            depth: i32::MAX,
                            names: pars,
                        })
                        .await
                }
            })
            .await
            .map(|res| format!("{:?}", res))
        })
        .await
    }

    pub async fn find_deploy<S: DeployService>(svc: &mut S, deploy_id: &[u8]) {
        Self::graceful_exit(svc.find_deploy(FindDeployQuery {
            deploy_id: Bytes::copy_from_slice(deploy_id),
        }))
        .await
    }

    /// Accepts a Rholang source file and deploys it
    #[allow(clippy::too_many_arguments)]
    pub async fn deploy_file_program<S: DeployService>(
        svc: &mut S,
        phlo_limit: i64,
        phlo_price: i64,
        valid_after_block: i64,
        private_key: &PrivateKey,
        file: &str,
        shard_id: &str,
    ) {
        Self::graceful_exit(async {
            // Try reading file (Scala: Try(Source.fromFile(...)).toEither)
            let code = match tokio::fs::read_to_string(file).await {
                Ok(code) => code,
                Err(e) => return Err(vec![format!("Error with given file:\n{e}")]),
            };

            let now_ms = match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(duration) => duration.as_millis() as i64,
                Err(e) => return Err(vec![format!("Clock error: {e}")]),
            };

            let d = DeployData {
                term: code,
                time_stamp: now_ms,
                phlo_price,
                phlo_limit,
                valid_after_block_number: valid_after_block,
                shard_id: shard_id.to_string(),
            };

            // Signed(d, Secp256k1, privateKey)
            let signed = match Signed::create(
                d,
                Box::new(crypto::rust::signatures::secp256k1::Secp256k1),
                private_key.clone(),
            ) {
                Ok(signed) => signed,
                Err(e) => return Err(vec![format!("Failed to sign deploy: {e}")]),
            };

            let resp = match svc.deploy(signed).await {
                Ok(resp) => resp,
                Err(e) => return Err(e),
            };
            Ok(format!("Response: {resp}"))
        })
        .await
    }

    pub async fn last_finalized_block<S: DeployService>(svc: &mut S) {
        Self::graceful_exit(svc.last_finalized_block()).await
    }

    pub async fn is_finalized<S: DeployService>(svc: &mut S, block_hash: String) {
        Self::graceful_exit(svc.is_finalized(IsFinalizedQuery { hash: block_hash })).await
    }

    pub async fn bond_status<S: DeployService>(svc: &mut S, public_key: &PublicKey) {
        Self::graceful_exit(svc.bond_status(BondStatusQuery {
            public_key: Bytes::from(public_key.bytes.clone()),
        }))
        .await
    }

    pub async fn status<S: DeployService>(svc: &mut S) {
        Self::graceful_exit(svc.status()).await
    }
}
