use std::collections::HashMap;

use models::casper::{BlockEventInfo, TransferInfo};
use models::rhoapi::Par;

use super::transaction::helpers;

/// Extract transfers from a `BlockEventInfo` for each deploy, keyed by deploy signature.
///
/// Scans each deploy's execution report for COMM events on the `transfer_unforgeable`
/// channel, then extracts from/to/amount/success from the produce data.
///
/// Only extracts user deploy transfers (not PreCharge/Refund/System deploys).
/// For user deploys: first report batch is precharge (skip), subsequent batches
/// where sender == deployer are user transfers.
pub fn extract_transfers_from_report(
    report: &BlockEventInfo,
    transfer_unforgeable: &Par,
) -> HashMap<String, Vec<TransferInfo>> {
    let mut transfers_by_deploy: HashMap<String, Vec<TransferInfo>> = HashMap::new();

    for deploy in &report.deploys {
        let deploy_sig = deploy
            .deploy_info
            .as_ref()
            .map(|info| info.sig.clone())
            .unwrap_or_default();

        if deploy.report.is_empty() {
            continue;
        }

        // First report batch is precharge — extract deployer address from it
        let first_batch_transactions = find_transfers_in_report(&deploy.report[0], transfer_unforgeable);
        let deployer_addr = first_batch_transactions
            .first()
            .map(|t| t.from_addr.clone());

        // Subsequent batches: transactions where sender == deployer are user transfers
        let mut user_transfers = Vec::new();
        for single_report in deploy.report.iter().skip(1) {
            let transfers = find_transfers_in_report(single_report, transfer_unforgeable);
            for transfer in transfers {
                match deployer_addr.as_ref() {
                    Some(addr) if transfer.from_addr == *addr => {
                        user_transfers.push(transfer);
                    }
                    _ => {
                        // Refund or system side-effect — not a user transfer
                    }
                }
            }
        }

        transfers_by_deploy.insert(deploy_sig, user_transfers);
    }

    transfers_by_deploy
}

/// Scan a single report for transfer events on the transfer_unforgeable channel.
fn find_transfers_in_report(
    report: &models::casper::SingleReport,
    transfer_unforgeable: &Par,
) -> Vec<TransferInfo> {
    let mut transfers = Vec::new();

    // Collect raw transactions from Comm events
    let mut raw_transactions: Vec<RawTransfer> = Vec::new();
    for event in &report.events {
        if let Some(models::casper::report_proto::Report::Comm(comm)) = &event.report {
            if let Some(channel) = comm.consume.as_ref().and_then(|c| c.channels.first()) {
                if *channel == *transfer_unforgeable {
                    if let Some(produce) = comm.produces.first() {
                        if let Some(tx) = helpers::parse_transaction_from_produce(produce) {
                            raw_transactions.push(RawTransfer {
                                from_addr: tx.from_addr,
                                to_addr: tx.to_addr,
                                amount: tx.amount,
                                ret_unforgeable: tx.ret_unforgeable,
                            });
                        }
                    }
                }
            }
        }
    }

    // Collect failure info from Produce events
    let ret_unforgeables: std::collections::HashSet<Par> = raw_transactions
        .iter()
        .map(|t| t.ret_unforgeable.clone())
        .collect();

    let mut failed_map: HashMap<Par, Option<String>> = HashMap::new();
    for event in &report.events {
        if let Some(models::casper::report_proto::Report::Produce(produce)) = &event.report {
            if let Some(channel) = &produce.channel {
                if ret_unforgeables.contains(channel) {
                    if let Some(fail_reason) = helpers::parse_failure_from_produce(&produce.data) {
                        failed_map.insert(channel.clone(), fail_reason);
                    }
                }
            }
        }
    }

    // Build TransferInfo with success/failure
    for tx in raw_transactions {
        let fail_reason = failed_map.get(&tx.ret_unforgeable).cloned().flatten();
        transfers.push(TransferInfo {
            from_addr: tx.from_addr,
            to_addr: tx.to_addr,
            amount: tx.amount,
            success: fail_reason.is_none(),
            fail_reason: fail_reason.unwrap_or_default(),
        });
    }

    transfers
}

struct RawTransfer {
    from_addr: String,
    to_addr: String,
    amount: i64,
    ret_unforgeable: Par,
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::casper::{
        report_proto, BlockEventInfo, DeployInfo, DeployInfoWithEventData,
        LightBlockInfo, ReportCommProto, ReportConsumeProto, ReportProduceProto,
        ReportProto, SingleReport,
    };
    use models::rhoapi::{
        expr::ExprInstance, g_unforgeable::UnfInstance, Expr, GPrivate, GUnforgeable,
        ListParWithRandom,
    };

    fn make_par_string(s: &str) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }],
            ..Default::default()
        }
    }

    fn make_par_int(n: i64) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GInt(n)),
            }],
            ..Default::default()
        }
    }

    fn make_transfer_unforgeable() -> Par {
        Par {
            unforgeables: vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                    id: vec![0x42; 32],
                })),
            }],
            ..Default::default()
        }
    }

    fn make_block_event_info_with_transfer(
        deploy_sig: &str,
        transfer_unforgeable: &Par,
        from: &str,
        to: &str,
        amount: i64,
    ) -> BlockEventInfo {
        // Transfer produce data: [from_addr, _, to_addr, amount, _, ret_unforgeable]
        let ret_unforg = make_transfer_unforgeable();
        let produce_data = ListParWithRandom {
            pars: vec![
                make_par_string(from),
                Par::default(),
                make_par_string(to),
                make_par_int(amount),
                Par::default(),
                ret_unforg,
            ],
            random_state: vec![],
        };

        let comm = ReportCommProto {
            consume: Some(ReportConsumeProto {
                channels: vec![transfer_unforgeable.clone()],
                patterns: vec![],
                peeks: vec![],
            }),
            produces: vec![ReportProduceProto {
                channel: Some(transfer_unforgeable.clone()),
                data: Some(produce_data),
            }],
        };

        // First report = precharge (transfer from deployer)
        let precharge_report = SingleReport {
            events: vec![ReportProto {
                report: Some(report_proto::Report::Comm(comm.clone())),
            }],
        };

        // Second report = user transfer (same from_addr as precharge = user deploy)
        let user_report = SingleReport {
            events: vec![ReportProto {
                report: Some(report_proto::Report::Comm(comm)),
            }],
        };

        BlockEventInfo {
            block_info: Some(LightBlockInfo::default()),
            deploys: vec![DeployInfoWithEventData {
                deploy_info: Some(DeployInfo {
                    sig: deploy_sig.to_string(),
                    ..Default::default()
                }),
                report: vec![precharge_report, user_report],
            }],
            system_deploys: vec![],
            post_state_hash: vec![].into(),
        }
    }

    #[test]
    fn extract_transfers_finds_user_transfers() {
        let transfer_unforgeable = make_transfer_unforgeable();
        let report = make_block_event_info_with_transfer(
            "deploy_abc",
            &transfer_unforgeable,
            "sender_addr",
            "receiver_addr",
            1000,
        );

        let result = extract_transfers_from_report(&report, &transfer_unforgeable);

        let transfers = result.get("deploy_abc").expect("should have deploy entry");
        assert_eq!(transfers.len(), 1, "should have one user transfer");

        let t = &transfers[0];
        assert_eq!(t.from_addr, "sender_addr");
        assert_eq!(t.to_addr, "receiver_addr");
        assert_eq!(t.amount, 1000);
        assert!(t.success);
        assert!(t.fail_reason.is_empty());
    }

    #[test]
    fn extract_transfers_returns_empty_for_no_transfer_deploy() {
        let transfer_unforgeable = make_transfer_unforgeable();

        // Deploy with empty reports (no COMM events on transfer channel)
        let report = BlockEventInfo {
            block_info: Some(LightBlockInfo::default()),
            deploys: vec![DeployInfoWithEventData {
                deploy_info: Some(DeployInfo {
                    sig: "deploy_no_transfer".to_string(),
                    ..Default::default()
                }),
                report: vec![SingleReport { events: vec![] }, SingleReport { events: vec![] }],
            }],
            system_deploys: vec![],
            post_state_hash: vec![].into(),
        };

        let result = extract_transfers_from_report(&report, &transfer_unforgeable);

        let transfers = result.get("deploy_no_transfer").expect("should have deploy entry");
        assert!(transfers.is_empty(), "should have no transfers");
    }
}
