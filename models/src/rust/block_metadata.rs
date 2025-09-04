// See models/src/main/scala/coop/rchain/models/BlockMetadata.scala

use prost::{bytes::Bytes, Message};
use std::{cmp::Ordering, collections::BTreeMap};

use crate::casper::{BlockMetadataInternal, BondProto};

use super::casper::protocol::casper_message::{BlockMessage, F1r3flyState, Justification};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Eq, PartialEq, Hash)]
pub struct BlockMetadata {
    #[serde(with = "shared::rust::serde_bytes")]
    pub block_hash: Bytes,
    #[serde(with = "shared::rust::serde_vec_bytes")]
    pub parents: Vec<Bytes>,
    #[serde(with = "shared::rust::serde_bytes")]
    pub sender: Bytes,
    pub justifications: Vec<Justification>,
    #[serde(with = "shared::rust::serde_btreemap_bytes_i64")]
    pub weight_map: BTreeMap<Bytes, i64>,
    pub block_number: i64,
    pub sequence_number: i32,
    pub invalid: bool,
    pub directly_finalized: bool,
    pub finalized: bool,
}

impl BlockMetadata {
    pub fn from_proto(proto: BlockMetadataInternal) -> Self {
        BlockMetadata {
            block_hash: proto.block_hash,
            parents: proto.parents,
            sender: proto.sender,
            justifications: proto
                .justifications
                .into_iter()
                .map(|j| Justification::from_proto(j))
                .collect(),
            weight_map: proto
                .bonds
                .into_iter()
                .map(|b| (b.validator.into(), b.stake))
                .collect(),
            block_number: proto.block_num,
            sequence_number: proto.seq_num,
            invalid: proto.invalid,
            directly_finalized: proto.directly_finalized,
            finalized: proto.finalized,
        }
    }

    pub fn to_proto(&self) -> BlockMetadataInternal {
        BlockMetadataInternal {
            block_hash: self.block_hash.clone(),
            parents: self.parents.clone(),
            sender: self.sender.clone(),
            justifications: self.justifications.iter().map(|j| j.to_proto()).collect(),
            bonds: self
                .weight_map
                .iter()
                .map(|(v, s)| BondProto {
                    validator: v.clone(),
                    stake: *s,
                })
                .collect(),
            block_num: self.block_number,
            seq_num: self.sequence_number,
            invalid: self.invalid,
            directly_finalized: self.directly_finalized,
            finalized: self.finalized,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_proto().encode_to_vec()
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let proto =
            BlockMetadataInternal::decode(bytes).expect("Failed to decode BlockMetadataInternal");
        Self::from_proto(proto)
    }

    fn bytes_ordering(left: &Bytes, right: &Bytes) -> Ordering {
        left.iter().cmp(right.iter())
    }

    pub fn ordering_by_num(left: &BlockMetadata, right: &BlockMetadata) -> Ordering {
        match left.block_number.cmp(&right.block_number) {
            Ordering::Equal => Self::bytes_ordering(&left.block_hash, &right.block_hash),
            other => other,
        }
    }

    fn weight_map(state: &F1r3flyState) -> BTreeMap<Bytes, i64> {
        state
            .bonds
            .iter()
            .map(|b| (b.validator.clone(), b.stake))
            .collect()
    }

    pub fn from_block(
        b: &BlockMessage,
        invalid: bool,
        directly_finalized: Option<bool>,
        finalized: Option<bool>,
    ) -> Self {
        let directly_finalized = directly_finalized.unwrap_or(false);
        let finalized = finalized.unwrap_or(false);
        Self {
            block_hash: b.block_hash.clone(),
            parents: b.header.parents_hash_list.clone(),
            sender: b.sender.clone(),
            justifications: b.justifications.clone(),
            weight_map: Self::weight_map(&b.body.state),
            block_number: b.body.state.block_number,
            sequence_number: b.seq_num,
            invalid,
            // this value is not used anywhere down the call pipeline, so its safe to set it to false
            directly_finalized,
            finalized,
        }
    }
}
