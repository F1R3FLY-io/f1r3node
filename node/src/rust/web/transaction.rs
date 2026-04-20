use casper::rust::api::block_report_api::BlockReportAPI;
use dashmap::DashMap;
use eyre::Result;
use futures::future::{BoxFuture, FutureExt, Shared};
use hex::ToHex;
use models::casper::{DeployInfoWithEventData, SingleReport, SystemDeployInfoWithEventData};
use models::rhoapi::Par;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_typed_store::KeyValueTypedStore;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use utoipa::ToSchema;

/// Transaction data structure
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Transaction {
    pub from_addr: String,
    pub to_addr: String,
    pub amount: i64,
    #[serde(skip, default)]
    pub ret_unforgeable: Par,
    pub fail_reason: Option<String>,
}

/// Transaction type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum TransactionType {
    #[serde(rename = "precharge")]
    PreCharge { deploy_id: String },
    #[serde(rename = "user_deploy")]
    UserDeploy { deploy_id: String },
    #[serde(rename = "refund")]
    Refund { deploy_id: String },
    #[serde(rename = "close_block")]
    CloseBlock { block_hash: String },
    #[serde(rename = "slashing_deploy")]
    SlashingDeploy { block_hash: String },
}

/// Transaction information combining transaction and type
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TransactionInfo {
    pub transaction: Transaction,
    pub transaction_type: TransactionType,
}

/// Transaction response containing list of transaction info
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TransactionResponse {
    pub data: Vec<TransactionInfo>,
}

/// Trait for transaction API operations
#[async_trait::async_trait]
pub trait TransactionAPI {
    /// Get transactions for a specific block hash
    async fn get_transaction(&self, block_hash: Blake2b256Hash) -> Result<Vec<TransactionInfo>>;
}

/// This API is totally based on how SystemVault.rho is written. If the `SystemVault.rho` is re-written or changed,
/// this API might end up with useless.
pub struct TransactionAPIImpl {
    block_report_api: BlockReportAPI,
    transfer_unforgeable: Par, // The transferUnforgeable can be retrieved based on the deployer and the timestamp of SystemVault.rho
                               // in the genesis ceremony.
}

impl TransactionAPIImpl {
    pub fn new(block_report_api: BlockReportAPI, transfer_unforgeable: Par) -> Self {
        Self {
            block_report_api,
            transfer_unforgeable,
        }
    }
}

#[async_trait::async_trait]
impl TransactionAPI for TransactionAPIImpl {
    async fn get_transaction(&self, block_hash: Blake2b256Hash) -> Result<Vec<TransactionInfo>> {
        let block_event_info = self
            .block_report_api
            .block_report(block_hash.to_bytes_prost(), false)
            .await;

        let block_event_info = match block_event_info {
            Ok(info) => info,
            Err(e) => return Err(eyre::eyre!("{}", e)),
        };

        let mut all_transactions = Vec::new();

        // Process user deploys
        for deploy in &block_event_info.deploys {
            let user_transactions = self.process_user_deploy(deploy).await?;
            all_transactions.extend(user_transactions);
        }

        // Process system deploys
        for system_deploy in &block_event_info.system_deploys {
            let system_transactions = self
                .process_system_deploy(system_deploy, &block_hash.encode_hex::<String>())
                .await?;
            all_transactions.extend(system_transactions);
        }

        Ok(all_transactions)
    }
}

impl TransactionAPIImpl {
    /// Process user deploy transactions (precharge, user deploy, refund)
    async fn process_user_deploy(
        &self,
        deploy: &DeployInfoWithEventData,
    ) -> Result<Vec<TransactionInfo>> {
        let mut transactions = Vec::new();

        // Get deploy info sig for transaction types
        let deploy_sig = deploy
            .deploy_info
            .as_ref()
            .map(|info| info.sig.clone())
            .unwrap_or_else(|| "unknown".to_string());

        if deploy.report.is_empty() {
            return Err(eyre::eyre!(
                "It is not possible that user report {} amount is 0",
                deploy_sig
            ));
        }

        // Precharge is always emitted in the first report batch.
        let first_report_transactions = self.find_transactions(&deploy.report[0]);
        let deployer_addr = first_report_transactions
            .first()
            .map(|t| t.from_addr.clone());
        for transaction in first_report_transactions {
            transactions.push(TransactionInfo {
                transaction,
                transaction_type: TransactionType::PreCharge {
                    deploy_id: deploy_sig.clone(),
                },
            });
        }

        // Subsequent batches may contain either user-transfer events, refund events, or both.
        // Classify by sender address: transactions sent by deployer are user deploy effects;
        // others are refund/system side-effects.
        for report in deploy.report.iter().skip(1) {
            let found_transactions = self.find_transactions(report);
            for transaction in found_transactions {
                let tx_type = match deployer_addr.as_ref() {
                    Some(addr) if transaction.from_addr == *addr => TransactionType::UserDeploy {
                        deploy_id: deploy_sig.clone(),
                    },
                    _ => TransactionType::Refund {
                        deploy_id: deploy_sig.clone(),
                    },
                };
                transactions.push(TransactionInfo {
                    transaction,
                    transaction_type: tx_type,
                });
            }
        }

        Ok(transactions)
    }

    /// Process system deploy transactions (close block, slashing deploy)
    async fn process_system_deploy(
        &self,
        system_deploy: &SystemDeployInfoWithEventData,
        block_hash: &str,
    ) -> Result<Vec<TransactionInfo>> {
        let mut transactions = Vec::new();

        // System deploys always have one report
        if let Some(report) = system_deploy.report.first() {
            let found_transactions = self.find_transactions(report);

            // Determine transaction type based on system deploy type
            let tx_type = match &system_deploy
                .system_deploy
                .as_ref()
                .and_then(|sd| sd.system_deploy.as_ref())
            {
                Some(
                    models::casper::system_deploy_data_proto::SystemDeploy::SlashSystemDeploy(_),
                ) => TransactionType::SlashingDeploy {
                    block_hash: block_hash.to_string(),
                },
                Some(
                    models::casper::system_deploy_data_proto::SystemDeploy::CloseBlockSystemDeploy(
                        _,
                    ),
                ) => TransactionType::CloseBlock {
                    block_hash: block_hash.to_string(),
                },
                None => {
                    return Err(eyre::eyre!("System deploy data is missing"));
                }
            };

            for transaction in found_transactions {
                transactions.push(TransactionInfo {
                    transaction,
                    transaction_type: tx_type.clone(),
                });
            }
        }

        Ok(transactions)
    }

    /// Find transactions in a single report (equivalent to Scala's findTransactions method)
    fn find_transactions(&self, report: &SingleReport) -> Vec<Transaction> {
        let mut transactions = Vec::new();

        // Find transactions from Comm events
        for event in &report.events {
            if let Some(models::casper::report_proto::Report::Comm(comm)) = &event.report {
                if let Some(channel) = comm.consume.as_ref().and_then(|c| c.channels.first()) {
                    // Check if this is the transfer channel we're looking for
                    if *channel == self.transfer_unforgeable {
                        if let Some(produce) = comm.produces.first() {
                            if let Some(transaction) =
                                helpers::parse_transaction_from_produce(produce)
                            {
                                transactions.push(transaction);
                            }
                        }
                    }
                }
            }
        }

        // Create a set of transaction return unforgeables for failure checking
        let transaction_ret_unforgeables: std::collections::HashSet<Par> = transactions
            .iter()
            .map(|t| t.ret_unforgeable.clone())
            .collect();

        // Find failure information from Produce events
        let mut failed_map: HashMap<Par, Option<String>> = HashMap::new();
        for event in &report.events {
            if let Some(models::casper::report_proto::Report::Produce(produce)) = &event.report {
                if let Some(channel) = &produce.channel {
                    if transaction_ret_unforgeables.contains(channel) {
                        if let Some(fail_reason) =
                            helpers::parse_failure_from_produce(&produce.data)
                        {
                            failed_map.insert(channel.clone(), fail_reason);
                        }
                    }
                }
            }
        }

        // Update transactions with failure information
        for transaction in &mut transactions {
            if let Some(fail_reason) = failed_map.get(&transaction.ret_unforgeable) {
                transaction.fail_reason = fail_reason.clone();
            }
        }

        transactions
    }
}

/// Cached transaction API that wraps another transaction API with caching
pub struct CacheTransactionAPI<TA, TS>
where
    TA: TransactionAPI,
    TS: KeyValueTypedStore<String, TransactionResponse> + Send + Sync + 'static,
{
    transaction_api: Arc<TA>,
    store: Arc<RwLock<TS>>,
    block_defer_map:
        Arc<DashMap<String, Shared<BoxFuture<'static, Result<TransactionResponse, String>>>>>,
}

impl<TA, TS> Clone for CacheTransactionAPI<TA, TS>
where
    TA: TransactionAPI,
    TS: KeyValueTypedStore<String, TransactionResponse> + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            transaction_api: self.transaction_api.clone(),
            store: self.store.clone(),
            block_defer_map: self.block_defer_map.clone(),
        }
    }
}

impl<TA, TS> CacheTransactionAPI<TA, TS>
where
    TA: TransactionAPI + Send + Sync + 'static,
    TS: KeyValueTypedStore<String, TransactionResponse> + Send + Sync + 'static,
{
    /// Create a new cached transaction API
    pub fn new(transaction_api: TA, store: TS) -> Self {
        Self {
            transaction_api: Arc::new(transaction_api),
            store: Arc::new(RwLock::new(store)),
            block_defer_map: Arc::new(DashMap::new()),
        }
    }

    /// Get transaction response for a block hash with caching
    pub async fn get_transaction(&self, block_hash: String) -> Result<TransactionResponse> {
        // Treat missing/empty cache entries as cache miss, not an error.
        let transaction_response = {
            let store = self.store.read().await;
            store
                .get(&vec![block_hash.clone()])?
                .into_iter()
                .next()
                .flatten()
        };

        if let Some(transaction_response) = transaction_response {
            return Ok(transaction_response);
        }

        let fetch_task = if let Some(entry) = self.block_defer_map.get(&block_hash) {
            entry.value().clone()
        } else {
            let transaction_api = self.transaction_api.clone();
            let block_hash_str = block_hash.clone();
            let store = self.store.clone();

            let task = async move {
                let data = transaction_api
                    .get_transaction(Blake2b256Hash::from_hex(&block_hash_str))
                    .await
                    .map_err(|e| e.to_string())?;

                let response = TransactionResponse { data };

                let store = store.write().await;
                store
                    .put(vec![(block_hash_str, response.clone())])
                    .map_err(|e| e.to_string())?;

                Ok(response)
            }
            .boxed()
            .shared();

            self.block_defer_map
                .insert(block_hash.clone(), task.clone());
            task
        };

        let res = fetch_task.await.map_err(|e| eyre::eyre!(e));
        // Cleanup in-flight dedup entry regardless of success/failure.
        self.block_defer_map.remove(&block_hash);
        let res = res?;

        Ok(res)
    }
}

// TODO: Port the next part to Rust
// Original Scala file: node/src/main/scala/coop/rchain/node/web/Transaction.scala.
// object Transaction {
//     type TransactionStore[F[_]] = KeyValueTypedStore[F, String, TransactionResponse]

//     object Encode {
//       import io.circe._, io.circe.generic.semiauto._
//       import coop.rchain.node.encode.JsonEncoder.{decodePar, encodePar}
//       implicit val encodeTransaction: Encoder[Transaction]         = deriveEncoder[Transaction]
//       implicit val encodeTransactionType: Encoder[TransactionType] = deriveEncoder[TransactionType]
//       implicit val encodeTransactionInfo: Encoder[TransactionInfo] = deriveEncoder[TransactionInfo]
//       implicit val encodeTransactionResponse: Encoder[TransactionResponse] =
//         deriveEncoder[TransactionResponse]

//       implicit val decodeTransaction: Decoder[Transaction]         = deriveDecoder[Transaction]
//       implicit val decodeTransactionType: Decoder[TransactionType] = deriveDecoder[TransactionType]
//       implicit val decodeTransactionInfo: Decoder[TransactionInfo] = deriveDecoder[TransactionInfo]
//       implicit val decodeTransactionResponse: Decoder[TransactionResponse] =
//         deriveDecoder[TransactionResponse]
//     }

//     object SCodec {
//       import scodec._
//       import scodec.bits._
//       import scodec.codecs._
//       import coop.rchain.rholang.interpreter.storage._

//       val transactionCodec: Codec[Transaction] =
//         (utf8_32 :: utf8_32 :: vlong :: serializePar.toSizeHeadCodec :: optional[String](
//           bool,
//           utf8_32
//         )).as[Transaction]
//       val precharge: Codec[PreCharge]   = utf8_32.as[PreCharge]
//       val refund: Codec[Refund]         = utf8_32.as[Refund]
//       val user: Codec[UserDeploy]       = utf8_32.as[UserDeploy]
//       val closeBlock: Codec[CloseBlock] = utf8_32.as[CloseBlock]
//       val slash: Codec[SlashingDeploy]  = utf8_32.as[SlashingDeploy]
//       val transactionType: Codec[TransactionType] = discriminated[TransactionType]
//         .by(uint8)
//         .subcaseP(0) {
//           case e: PreCharge => e
//         }(precharge)
//         .subcaseP(1) {
//           case s: UserDeploy => s
//         }(user)
//         .subcaseP(2) {
//           case pb: Refund => pb
//         }(refund)
//         .subcaseP(3) {
//           case c: CloseBlock => c
//         }(closeBlock)
//         .subcaseP(4) {
//           case s: SlashingDeploy => s
//         }(slash)

//       val transactionInfo: Codec[TransactionInfo] =
//         (transactionCodec :: transactionType).as[TransactionInfo]
//       val transactionResponseCodec: Codec[TransactionResponse] = listOfN(int32, transactionInfo)
//         .as[TransactionResponse]
//     }

/// (Historical ref: was RevVault.rho in upstream rchain/rchain)
/// https://github.com/rchain/rchain/blob/43257ddb7b2b53cffb59a5fe1d4c8296c18b8292/casper/src/main/resources/RevVault.rho#L25
/// This hard-coded value is only useful with current(above link version) `SystemVault.rho` implementation but it is
/// useful for all the networks(testnet, custom network and mainnet) starting with this `SystemVault.rho`.
///
/// This hard-coded value needs to be changed when:
/// 1. `SystemVault.rho` is changed
/// 2. `StandardDeploys.system_vault` is changed
/// 3. The random seed algorithm for unforgeable name of the deploy is changed
///
/// This is not needed when onChain transfer history is implemented and deployed to new network in the future.
pub fn transfer_unforgeable() -> Par {
    use casper::rust::genesis::contracts::standard_deploys::{
        to_public, SYSTEM_VAULT_PK, SYSTEM_VAULT_TIMESTAMP,
    };
    use casper::rust::util::rholang::tools::Tools;
    use models::rhoapi::{g_unforgeable::UnfInstance, GPrivate, GUnforgeable};

    let system_vault_pub_key = to_public(SYSTEM_VAULT_PK);
    let mut seed_for_system_vault =
        Tools::unforgeable_name_rng(&system_vault_pub_key, SYSTEM_VAULT_TIMESTAMP);

    // the 11th unforgeable name (drop 10, take the next one)
    for _ in 0..10 {
        seed_for_system_vault.next();
    }
    let unforgeable_bytes = seed_for_system_vault.next();

    Par {
        unforgeables: vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: unforgeable_bytes.into_iter().map(|b| b as u8).collect(),
            })),
        }],
        ..Default::default()
    }
}

/// JSON-based typed store for TransactionResponse caching.
/// Uses serde_json instead of bincode because TransactionResponse contains
/// internally-tagged enums (`#[serde(tag = "type")]` on TransactionType)
/// which bincode does not support.
pub struct JsonTypedStore {
    store: std::sync::Arc<dyn shared::rust::store::key_value_store::KeyValueStore>,
}

impl shared::rust::store::key_value_typed_store::KeyValueTypedStore<String, TransactionResponse>
    for JsonTypedStore
{
    fn get(
        &self,
        keys: &Vec<String>,
    ) -> Result<Vec<Option<TransactionResponse>>, shared::rust::store::key_value_store::KvStoreError>
    {
        let encoded_keys: Vec<Vec<u8>> = keys
            .iter()
            .map(|k| k.as_bytes().to_vec())
            .collect();
        let raw_values = self.store.get(&encoded_keys)?;
        Ok(raw_values
            .into_iter()
            .map(|opt| {
                opt.and_then(|bytes| serde_json::from_slice(&bytes).ok())
            })
            .collect())
    }

    fn put(
        &self,
        kv_pairs: Vec<(String, TransactionResponse)>,
    ) -> Result<(), shared::rust::store::key_value_store::KvStoreError> {
        let encoded: Vec<(Vec<u8>, Vec<u8>)> = kv_pairs
            .into_iter()
            .map(|(k, v)| {
                (
                    k.as_bytes().to_vec(),
                    serde_json::to_vec(&v).unwrap(),
                )
            })
            .collect();
        self.store.put(encoded)
    }

    fn delete(
        &self,
        keys: Vec<String>,
    ) -> Result<(), shared::rust::store::key_value_store::KvStoreError> {
        let encoded: Vec<Vec<u8>> = keys
            .into_iter()
            .map(|k| k.as_bytes().to_vec())
            .collect();
        self.store.delete(encoded)?;
        Ok(())
    }

    fn contains(
        &self,
        keys: Vec<String>,
    ) -> Result<Vec<bool>, shared::rust::store::key_value_store::KvStoreError> {
        let values = self.get(&keys)?;
        Ok(values.into_iter().map(|v| v.is_some()).collect())
    }

    fn collect<F, T>(&self, _f: F) -> Result<Vec<T>, shared::rust::store::key_value_store::KvStoreError>
    where
        F: FnMut((&String, &TransactionResponse)) -> Option<T>,
    {
        Ok(Vec::new())
    }

    fn to_map(
        &self,
    ) -> Result<
        std::collections::HashMap<String, TransactionResponse>,
        shared::rust::store::key_value_store::KvStoreError,
    > {
        Ok(std::collections::HashMap::new())
    }

    fn non_empty(&self) -> Result<bool, shared::rust::store::key_value_store::KvStoreError> {
        self.store.non_empty()
    }
}

/// Create a CacheTransactionAPI with a JSON-based cache store
pub async fn cache_transaction_api(
    transaction_api: TransactionAPIImpl,
    rnode_store_manager: &mut impl rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager,
) -> Result<CacheTransactionAPI<TransactionAPIImpl, JsonTypedStore>> {
    let store = rnode_store_manager
        .store("transaction".to_string())
        .await
        .map_err(|e| eyre::eyre!("Failed to create transaction store: {}", e))?;

    let json_store = JsonTypedStore { store };

    Ok(CacheTransactionAPI::new(transaction_api, json_store))
}

mod helpers {
    use crate::rust::web::transaction::Transaction;
    use models::rust::par_ext::ParExt;

    /// Parse a transaction from a produce event
    pub fn parse_transaction_from_produce(
        produce: &models::casper::ReportProduceProto,
    ) -> Option<Transaction> {
        let pars = &produce.data.as_ref()?.pars;

        // Extract transaction fields
        if pars.len() >= 6 {
            let from_addr = pars[0].get_g_string()?;
            let to_addr = pars[2].get_g_string()?;
            let amount = pars[3].get_g_int()?;
            let ret_unforgeable = pars[5].clone();

            Some(Transaction {
                from_addr,
                to_addr,
                amount,
                ret_unforgeable,
                fail_reason: None,
            })
        } else {
            None
        }
    }

    /// Parse failure information from a produce event
    pub fn parse_failure_from_produce(
        data: &Option<models::rhoapi::ListParWithRandom>,
    ) -> Option<Option<String>> {
        if let Some(data) = data {
            if let Some(first_par) = data.pars.first() {
                if let Some(tuple_body) = first_par.get_e_tuple_body() {
                    if let Some(ps) = tuple_body.ps.first() {
                        if let Some(success) = ps.get_g_bool() {
                            if success {
                                return Some(None); // Success, no failure reason
                            } else {
                                // Failure, get the failure reason from the second element
                                if tuple_body.ps.len() > 1 {
                                    if let Some(fail_reason) = tuple_body.ps[1].get_g_string() {
                                        return Some(Some(fail_reason));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}
