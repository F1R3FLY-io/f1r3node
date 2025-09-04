// See comm/src/main/scala/coop/rchain/comm/transport/PacketOps.scala

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use models::routing::Packet;
use prost::Message;
use rand::RngCore;

use crate::rust::errors::{self, CommError};

/// Type alias for the concurrent cache used in streaming
pub type StreamCache = Arc<DashMap<String, Vec<u8>>>;

/// PacketOps provides functionality for storing and restoring packets in cache
pub struct PacketOps;

impl PacketOps {
    /// Restore a packet from cache using the given key
    pub fn restore(key: &str, cache: &StreamCache) -> Result<Packet, CommError> {
        let data = cache
            .get(key)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| {
                errors::unable_to_restore_packet(
                    key.to_string(),
                    "Key not found in cache".to_string(),
                )
            })?;

        Packet::decode(data.as_slice()).map_err(|e| {
            errors::unable_to_restore_packet(
                key.to_string(),
                format!("Failed to parse packet: {}", e),
            )
        })
    }

    /// Store a packet in cache and return the generated key
    pub fn store(packet: &Packet, cache: &StreamCache) -> Result<String, CommError> {
        let key = Self::create_cache_entry("packet_receive", cache)?;
        let packet_data = packet.encode_to_vec();
        cache.insert(key.clone(), packet_data);
        Ok(key)
    }

    /// Generate a unique key and put empty data in streaming cache
    pub fn create_cache_entry(prefix: &str, cache: &StreamCache) -> Result<String, CommError> {
        let key = format!("{}/{}", prefix, Self::timestamp());
        cache.insert(key.clone(), Vec::new());
        Ok(key)
    }

    /// Generate a timestamp-based unique identifier
    fn timestamp() -> String {
        // Get current time
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let datetime = DateTime::<Utc>::from_timestamp(now.as_secs() as i64, now.subsec_nanos())
            .expect("Invalid timestamp");

        // Format as yyyyMMddHHmmss
        let date_str = datetime.format("%Y%m%d%H%M%S").to_string();

        // Generate 4 random bytes and encode as hex
        let mut rng = rand::rng();
        let mut bytes = [0u8; 4];
        rng.fill_bytes(&mut bytes);
        let hex_str = hex::encode(bytes);

        format!("{}_{}", date_str, hex_str)
    }
}

/// Extension trait to add store method to Packet
pub trait PacketExt {
    fn store(&self, cache: &StreamCache) -> Result<String, CommError>;
}

impl PacketExt for Packet {
    fn store(&self, cache: &StreamCache) -> Result<String, CommError> {
        PacketOps::store(self, cache)
    }
}

// See comm/src/test/scala/coop/rchain/comm/transport/PacketStoreRestoreSpec.scala

#[cfg(test)]
mod tests {
    use super::*;
    use prost::bytes::Bytes;

    /// Create a test cache
    fn create_test_cache() -> StreamCache {
        Arc::new(DashMap::new())
    }

    #[test]
    fn test_packet_store_and_restore() {
        let cache = create_test_cache();
        let content = vec![1, 2, 3, 4, 5];
        let packet = Packet {
            type_id: "Test".to_string(),
            content: Bytes::from(content),
        };

        // Store the packet
        let key = packet.store(&cache).expect("Failed to store packet");

        // Restore the packet
        let restored = PacketOps::restore(&key, &cache).expect("Failed to restore packet");

        // Verify they are equal
        assert_eq!(packet.type_id, restored.type_id);
        assert_eq!(packet.content, restored.content);
    }

    #[test]
    fn test_restore_nonexistent_key() {
        let cache = create_test_cache();
        let result = PacketOps::restore("nonexistent", &cache);

        assert!(result.is_err());
        if let Err(CommError::UnableToRestorePacket(key, msg)) = result {
            assert_eq!(key, "nonexistent");
            assert!(msg.contains("Key not found"));
        } else {
            panic!("Expected UnableToRestorePacket error");
        }
    }

    #[test]
    fn test_create_cache_entry() {
        let cache = create_test_cache();
        let key = PacketOps::create_cache_entry("test_prefix", &cache)
            .expect("Failed to create cache entry");

        assert!(key.starts_with("test_prefix/"));

        // Verify entry was created in cache
        assert!(cache.contains_key(&key));
        assert_eq!(cache.get(&key).unwrap().value(), &Vec::<u8>::new());
    }

    #[test]
    fn test_timestamp_format() {
        let timestamp = PacketOps::timestamp();

        // Should be in format: yyyyMMddHHmmss_hex
        let parts: Vec<&str> = timestamp.split('_').collect();
        assert_eq!(parts.len(), 2);

        // Date part should be 14 characters (yyyyMMddHHmmss)
        assert_eq!(parts[0].len(), 14);

        // Hex part should be 8 characters (4 bytes as hex)
        assert_eq!(parts[1].len(), 8);

        // All characters should be valid
        assert!(parts[0].chars().all(|c| c.is_ascii_digit()));
        assert!(parts[1].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_timestamp_uniqueness() {
        // Generate multiple timestamps quickly - they should be unique due to random component
        let mut timestamps = std::collections::HashSet::new();
        for _ in 0..10 {
            let ts = PacketOps::timestamp();
            assert!(timestamps.insert(ts), "Timestamp should be unique");
        }
    }

    // Tests multiple content sizes
    #[test]
    fn test_packet_store_restore_property_equivalent() {
        let cache = create_test_cache();

        // Test multiple random content sizes (simulating property testing)
        let test_cases = vec![
            // Small sizes
            (10..100).map(|_| rand::random::<u8>()).collect::<Vec<u8>>(),
            (500..1000)
                .map(|_| rand::random::<u8>())
                .collect::<Vec<u8>>(),
            // Medium sizes
            (5000..10000)
                .map(|_| rand::random::<u8>())
                .collect::<Vec<u8>>(),
            // Large sizes
            (25000..50000)
                .map(|_| rand::random::<u8>())
                .collect::<Vec<u8>>(),
        ];

        for content in test_cases {
            // given
            let packet = Packet {
                type_id: "Test".to_string(),
                content: Bytes::from(content),
            };

            // when
            let stored_key = packet.store(&cache).expect("Failed to store packet");
            let restored =
                PacketOps::restore(&stored_key, &cache).expect("Failed to restore packet");

            // then - packet should equal restored
            assert_eq!(packet.type_id, restored.type_id);
            assert_eq!(packet.content, restored.content);
        }
    }
}
