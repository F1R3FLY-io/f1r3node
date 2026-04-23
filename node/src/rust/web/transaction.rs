use models::rhoapi::Par;
use serde::{Deserialize, Serialize};
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

/// Compute the transfer_unforgeable Par from SystemVault.rho genesis constants.
///
/// This hard-coded value is only useful with current `SystemVault.rho` implementation
/// and needs to change when:
/// 1. `SystemVault.rho` is changed
/// 2. `StandardDeploys.system_vault` is changed
/// 3. The random seed algorithm for unforgeable name of the deploy is changed
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

pub mod helpers {
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
