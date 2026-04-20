use std::collections::HashMap;

use async_trait::async_trait;
use models::casper::{BlockInfo, TransferInfo};

use super::transaction::{TransactionResponse, TransactionType};

/// Maps transaction response data into per-deploy transfer info, keyed by deploy signature.
pub fn map_transactions_to_transfers(
    response: &TransactionResponse,
) -> HashMap<String, Vec<TransferInfo>> {
    let mut transfers_by_deploy: HashMap<String, Vec<TransferInfo>> = HashMap::new();

    for info in &response.data {
        let deploy_id = match &info.transaction_type {
            TransactionType::UserDeploy { deploy_id } => deploy_id,
            _ => continue,
        };

        let transfer = TransferInfo {
            from_addr: info.transaction.from_addr.clone(),
            to_addr: info.transaction.to_addr.clone(),
            amount: info.transaction.amount,
            success: info.transaction.fail_reason.is_none(),
            fail_reason: info.transaction.fail_reason.clone().unwrap_or_default(),
        };

        transfers_by_deploy
            .entry(deploy_id.clone())
            .or_default()
            .push(transfer);
    }

    transfers_by_deploy
}

/// Enriches a `BlockInfo` by populating the `transfers` field on each `DeployInfo`
/// using data from the transaction response.
pub fn enrich_block_info(
    mut block_info: BlockInfo,
    response: &TransactionResponse,
) -> BlockInfo {
    let transfers_by_deploy = map_transactions_to_transfers(response);

    for deploy_info in &mut block_info.deploys {
        if let Some(transfers) = transfers_by_deploy.get(&deploy_info.sig) {
            deploy_info.transfers = transfers.clone();
        }
    }

    block_info
}

/// Trait for enriching block info with transfer data.
/// Used to pass a type-erased enricher to gRPC services that don't know
/// the concrete `CacheTransactionAPI` generic parameters.
///
/// Returns `(BlockInfo, bool)` where the bool indicates whether enrichment
/// succeeded. When `true`, transfers on each deploy are meaningful (empty list
/// means no transfers). When `false`, transfers could not be extracted (e.g.
/// block replay unavailable on validator nodes) and should be treated as unknown.
#[async_trait]
pub trait BlockEnricher: Send + Sync {
    async fn enrich(&self, block_info: BlockInfo) -> (BlockInfo, bool);
}

/// Concrete `BlockEnricher` implementation backed by `CacheTransactionAPI`.
pub struct CacheTransactionEnricher<TA, TS>
where
    TA: super::transaction::TransactionAPI + Send + Sync + 'static,
    TS: shared::rust::store::key_value_typed_store::KeyValueTypedStore<String, TransactionResponse>
        + Send
        + Sync
        + 'static,
{
    cache_transaction_api: super::transaction::CacheTransactionAPI<TA, TS>,
}

impl<TA, TS> CacheTransactionEnricher<TA, TS>
where
    TA: super::transaction::TransactionAPI + Send + Sync + 'static,
    TS: shared::rust::store::key_value_typed_store::KeyValueTypedStore<String, TransactionResponse>
        + Send
        + Sync
        + 'static,
{
    pub fn new(cache_transaction_api: super::transaction::CacheTransactionAPI<TA, TS>) -> Self {
        Self {
            cache_transaction_api,
        }
    }
}

#[async_trait]
impl<TA, TS> BlockEnricher for CacheTransactionEnricher<TA, TS>
where
    TA: super::transaction::TransactionAPI + Send + Sync + 'static,
    TS: shared::rust::store::key_value_typed_store::KeyValueTypedStore<String, TransactionResponse>
        + Send
        + Sync
        + 'static,
{
    async fn enrich(&self, block_info: BlockInfo) -> (BlockInfo, bool) {
        let block_hash = block_info
            .block_info
            .as_ref()
            .map(|bi| bi.block_hash.clone())
            .unwrap_or_default();

        if block_hash.is_empty() {
            return (block_info, false);
        }

        match self
            .cache_transaction_api
            .get_transaction(block_hash.clone())
            .await
        {
            Ok(response) => (enrich_block_info(block_info, &response), true),
            Err(e) => {
                tracing::warn!(
                    target: "f1r3fly.api",
                    block_hash = %block_hash,
                    error = %e,
                    "Failed to extract transfers for block"
                );
                (block_info, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::transaction::{Transaction, TransactionInfo, TransactionResponse, TransactionType};
    use models::casper::{DeployInfo, LightBlockInfo};

    fn make_transaction(from: &str, to: &str, amount: i64, fail_reason: Option<String>) -> Transaction {
        Transaction {
            from_addr: from.to_string(),
            to_addr: to.to_string(),
            amount,
            ret_unforgeable: Default::default(),
            fail_reason,
        }
    }

    fn make_user_deploy(deploy_id: &str, tx: Transaction) -> TransactionInfo {
        TransactionInfo {
            transaction: tx,
            transaction_type: TransactionType::UserDeploy {
                deploy_id: deploy_id.to_string(),
            },
        }
    }

    fn make_precharge(deploy_id: &str, tx: Transaction) -> TransactionInfo {
        TransactionInfo {
            transaction: tx,
            transaction_type: TransactionType::PreCharge {
                deploy_id: deploy_id.to_string(),
            },
        }
    }

    fn make_refund(deploy_id: &str, tx: Transaction) -> TransactionInfo {
        TransactionInfo {
            transaction: tx,
            transaction_type: TransactionType::Refund {
                deploy_id: deploy_id.to_string(),
            },
        }
    }

    fn make_block_info(deploy_sigs: &[&str]) -> BlockInfo {
        BlockInfo {
            block_info: Some(LightBlockInfo {
                block_hash: "abc123".to_string(),
                ..Default::default()
            }),
            deploys: deploy_sigs
                .iter()
                .map(|sig| DeployInfo {
                    sig: sig.to_string(),
                    ..Default::default()
                })
                .collect(),
        }
    }

    #[test]
    fn map_transactions_to_transfers_filters_user_deploys() {
        let deploy_id = "deploy_sig_abc";
        let response = TransactionResponse {
            data: vec![
                make_precharge(deploy_id, make_transaction("alice", "system", 100, None)),
                make_user_deploy(deploy_id, make_transaction("alice", "bob", 1, None)),
                make_refund(deploy_id, make_transaction("system", "alice", 50, None)),
            ],
        };

        let result = map_transactions_to_transfers(&response);

        assert_eq!(result.len(), 1, "should have one deploy entry");
        let transfers = result.get(deploy_id).expect("missing deploy entry");
        assert_eq!(transfers.len(), 1, "should have one transfer");

        let t = &transfers[0];
        assert_eq!(t.from_addr, "alice");
        assert_eq!(t.to_addr, "bob");
        assert_eq!(t.amount, 1);
        assert!(t.success);
        assert_eq!(t.fail_reason, "");
    }

    #[test]
    fn enrich_block_info_populates_transfers() {
        let deploy_id = "deploy_sig_abc";
        let response = TransactionResponse {
            data: vec![
                make_precharge(deploy_id, make_transaction("alice", "system", 100, None)),
                make_user_deploy(deploy_id, make_transaction("alice", "bob", 5000000, None)),
                make_refund(deploy_id, make_transaction("system", "alice", 50, None)),
            ],
        };

        let block_info = make_block_info(&[deploy_id]);
        let enriched = enrich_block_info(block_info, &response);

        assert_eq!(enriched.deploys.len(), 1);
        let deploy = &enriched.deploys[0];
        assert_eq!(deploy.transfers.len(), 1);

        let t = &deploy.transfers[0];
        assert_eq!(t.from_addr, "alice");
        assert_eq!(t.to_addr, "bob");
        assert_eq!(t.amount, 5000000);
        assert!(t.success);
        assert_eq!(t.fail_reason, "");
    }

    #[test]
    fn enrich_block_info_returns_empty_transfers_when_no_user_deploys() {
        let deploy_id = "deploy_sig_noop";
        let response = TransactionResponse {
            data: vec![
                make_precharge(deploy_id, make_transaction("alice", "system", 100, None)),
                make_refund(deploy_id, make_transaction("system", "alice", 50, None)),
            ],
        };

        let block_info = make_block_info(&[deploy_id]);
        let enriched = enrich_block_info(block_info, &response);

        assert_eq!(enriched.deploys.len(), 1);
        assert!(
            enriched.deploys[0].transfers.is_empty(),
            "deploys with no UserDeploy transactions should have empty transfers"
        );
    }

}
