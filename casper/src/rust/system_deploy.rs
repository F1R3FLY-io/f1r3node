/// System deploy marker constants.
/// System deploy IDs are 33 bytes: [32-byte blockHash][1-byte marker]

/// Marker for slash system deploys
pub const SLASH_MARKER: u8 = 0x01;
/// Marker for close block system deploys
pub const CLOSE_BLOCK_MARKER: u8 = 0x02;
/// Marker for empty/heartbeat system deploys
pub const HEARTBEAT_MARKER: u8 = 0x03;

/// System deploy ID length (32-byte hash + 1-byte marker)
pub const SYSTEM_DEPLOY_ID_LEN: usize = 33;

use prost::bytes::Bytes;

/// Detect if a deploy ID is a system deploy ID.
/// System deploy IDs are 33 bytes: [32-byte blockHash][1-byte marker]
/// Markers: 0x01 (slash), 0x02 (close block), 0x03 (empty/heartbeat)
pub fn is_system_deploy_id(id: &Bytes) -> bool {
    id.len() == SYSTEM_DEPLOY_ID_LEN && {
        let last_byte = id[32];
        last_byte == SLASH_MARKER || last_byte == CLOSE_BLOCK_MARKER || last_byte == HEARTBEAT_MARKER
    }
}
