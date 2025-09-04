// See models/src/main/scala/coop/rchain/casper/PrettyPrinter.scala

use crypto::rust::signatures::signed::Signed;
use shared::rust::ByteString;

use super::protocol::casper_message::{
    BlockHashMessage, BlockMessage, Bond, CasperMessage, DeployData, F1r3flyState, ProcessedDeploy,
};

pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn build_string_no_limit(b: &[u8]) -> String {
        hex::encode(b)
    }

    pub fn build_string(t: CasperMessage, short: bool) -> String {
        match t {
            CasperMessage::BlockMessage(block_message) => {
                Self::build_string_block_message(&block_message, short)
            }
            _ => "Unknown consensus protocol message".to_string(),
        }
    }

    pub fn build_string_block_message(b: &BlockMessage, short: bool) -> String {
        match b.header.parents_hash_list.first() {
            None => format!(
                "Block #{} ({}) with empty parents (supposedly genesis)",
                b.body.state.block_number,
                Self::build_string_bytes(&b.block_hash)
            ),
            Some(main_parent) => {
                if short {
                    format!(
                        "#{} ({})",
                        b.body.state.block_number,
                        Self::build_string_bytes(&b.block_hash)
                    )
                } else {
                    format!(
                      "Block #{} ({}) -- Sender ID {} -- M Parent Hash {} -- Contents {} -- Shard ID {}",
                      b.body.state.block_number,
                      Self::build_string_bytes(&b.block_hash),
                      Self::build_string_bytes(&b.sender),
                      Self::build_string_bytes(main_parent),
                      Self::build_string_f1r3fly_state(&b.body.state),
                      Self::limit(&b.shard_id, 10)
                  )
                }
            }
        }
    }

    pub fn build_string_block_hash_message(bh: &BlockHashMessage) -> String {
        format!("Block hash: {}", Self::build_string_bytes(&bh.block_hash))
    }

    fn limit(s: &str, max_length: usize) -> String {
        if s.len() > max_length {
            format!("{}...", &s[0..max_length])
        } else {
            s.to_string()
        }
    }

    pub fn build_string_processed_deploy(d: &ProcessedDeploy) -> String {
        format!(
            "User: {}, Cost: {:?} {}",
            Self::build_string_no_limit(&d.deploy.pk.bytes),
            d.cost,
            Self::build_string_signed_deploy_data(&d.deploy)
        )
    }

    pub fn build_string_bytes(bytes: &[u8]) -> String {
        Self::limit(&hex::encode(bytes), 10)
    }

    pub fn build_string_sig(bytes: &[u8]) -> String {
        let str1 = hex::encode(&bytes[0..10.min(bytes.len())]);
        let str2 = if bytes.len() > 10 {
            hex::encode(&bytes[bytes.len() - 10.min(bytes.len())..])
        } else {
            "".to_string()
        };
        format!("{}...{}", str1, str2)
    }

    pub fn build_string_signed_deploy_data(sd: &Signed<DeployData>) -> String {
        format!(
            "{} Sig: {} SigAlgorithm: {} ValidAfterBlockNumber: {}",
            Self::build_string_deploy_data(&sd.data),
            Self::build_string_sig(&sd.sig),
            sd.sig_algorithm.name(),
            sd.data.valid_after_block_number
        )
    }

    pub fn build_string_deploy_data(d: &DeployData) -> String {
        format!("DeployData #{} -- {}", d.time_stamp, d.term)
    }

    pub fn build_string_f1r3fly_state(r: &F1r3flyState) -> String {
        Self::build_string_bytes(&r.post_state_hash)
    }

    pub fn build_string_bond(b: &Bond) -> String {
        format!("{}: {}", Self::build_string_no_limit(&b.validator), b.stake)
    }

    pub fn build_string_hashes(hashes: &[ByteString]) -> String {
        let contents: Vec<String> = hashes
            .iter()
            .map(|hash| Self::build_string_bytes(hash))
            .collect();
        format!("[{}]", contents.join(" "))
    }
}
