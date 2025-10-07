// See models/src/main/scala/coop/rchain/casper/protocol/PacketTypeTag.scala

use prost::Message;

use crate::{
    casper::{
        ApprovedBlockProto, ApprovedBlockRequestProto, BlockApprovalProto, BlockHashMessageProto,
        BlockMessageProto, BlockRequestProto, ForkChoiceTipRequestProto, HasBlockProto,
        HasBlockRequestProto, NoApprovedBlockAvailableProto, StoreItemsMessageProto,
        StoreItemsMessageRequestProto, UnapprovedBlockProto,
    },
    routing::Packet,
};

// Base trait for packet types
pub trait PacketType {
    fn tag() -> &'static str;
}

// Trait for converting to packets
pub trait ToPacket {
    type Type: PacketType;

    fn content(&self) -> prost::bytes::Bytes;

    fn mk_packet(&self) -> Packet {
        Packet {
            type_id: Self::Type::tag().to_string(),
            content: self.content(),
        }
    }
}

// Macro to implement both traits
#[macro_export]
macro_rules! impl_packet {
    ($type:ty, $tag:expr) => {
        impl PacketType for $type {
            fn tag() -> &'static str {
                $tag
            }
        }

        impl ToPacket for $type {
            type Type = Self;

            fn content(&self) -> prost::bytes::Bytes {
                self.encode_to_vec().into()
            }
        }
    };
}

// Implement for all message types
impl_packet!(BlockMessageProto, "BlockMessage");
impl_packet!(BlockHashMessageProto, "BlockHashMessage");
impl_packet!(ApprovedBlockProto, "ApprovedBlock");
impl_packet!(UnapprovedBlockProto, "UnapprovedBlock");
impl_packet!(BlockApprovalProto, "BlockApproval");
impl_packet!(NoApprovedBlockAvailableProto, "NoApprovedBlockAvailable");
impl_packet!(BlockRequestProto, "BlockRequest");
impl_packet!(ApprovedBlockRequestProto, "ApprovedBlockRequest");
impl_packet!(HasBlockRequestProto, "HasBlockRequest");
impl_packet!(HasBlockProto, "HasBlock");
impl_packet!(ForkChoiceTipRequestProto, "ForkChoiceTipRequest");
impl_packet!(StoreItemsMessageRequestProto, "StoreItemsMessageRequest");
impl_packet!(StoreItemsMessageProto, "StoreItemsMessage");
