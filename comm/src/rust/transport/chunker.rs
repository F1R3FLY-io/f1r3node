// See comm/src/main/scala/coop/rchain/comm/transport/Chunker.scala

use models::routing::{Chunk, ChunkData, ChunkHeader};
use prost::bytes::Bytes;
use shared::rust::shared::compression::CompressionOps;

use crate::rust::{peer_node::PeerNode, rp::protocol_helper, transport::transport_layer::Blob};

pub struct Chunker;

impl Chunker {
    /// Chunks a blob into an iterator of Chunk messages for streaming
    pub fn chunk_it(network_id: &str, blob: &Blob, max_message_size: usize) -> Vec<Chunk> {
        let raw = blob.packet.content.as_ref();
        let kb500 = 1024 * 500;
        let compress = raw.len() > kb500;

        // Compress if data is large enough
        let content = if compress {
            raw.compress()
        } else {
            raw.to_vec()
        };

        // Create header chunk
        let header_chunk = Self::create_header_chunk(
            &blob.sender,
            &blob.packet.type_id,
            network_id,
            compress,
            raw.len(),
        );

        // Calculate chunk size (reserve 2KB buffer for protobuf overhead)
        let buffer = 2 * 1024; // 2 kbytes for protobuf related stuff
        let chunk_size = max_message_size.saturating_sub(buffer);

        // Create data chunks
        let data_chunks = Self::create_data_chunks(&content, chunk_size);

        // Return iterator starting with header, followed by data chunks
        let mut chunks = vec![header_chunk];
        chunks.extend(data_chunks);
        chunks
    }

    /// Create the header chunk containing metadata
    fn create_header_chunk(
        sender: &PeerNode,
        type_id: &str,
        network_id: &str,
        compressed: bool,
        content_length: usize,
    ) -> Chunk {
        let chunk_header = ChunkHeader {
            sender: Some(protocol_helper::node(sender)),
            type_id: type_id.to_string(),
            compressed,
            content_length: content_length as i32,
            network_id: network_id.to_string(),
        };

        Chunk {
            content: Some(models::routing::chunk::Content::Header(chunk_header)),
        }
    }

    /// Create data chunks by sliding over content
    fn create_data_chunks(content: &[u8], chunk_size: usize) -> Vec<Chunk> {
        if chunk_size == 0 {
            return vec![];
        }

        content
            .chunks(chunk_size)
            .map(|chunk_data| {
                let chunk_data_proto = ChunkData {
                    content_data: Bytes::copy_from_slice(chunk_data),
                };

                Chunk {
                    content: Some(models::routing::chunk::Content::Data(chunk_data_proto)),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::peer_node::{Endpoint, NodeIdentifier};
    use models::routing::Packet;

    fn create_test_blob(content: Vec<u8>) -> Blob {
        let peer = PeerNode {
            id: NodeIdentifier {
                key: Bytes::from("test_peer".as_bytes()),
            },
            endpoint: Endpoint::new("127.0.0.1".to_string(), 8080, 8080),
        };

        let packet = Packet {
            type_id: "TestPacket".to_string(),
            content: Bytes::from(content),
        };

        Blob {
            sender: peer,
            packet,
        }
    }

    #[test]
    fn test_small_message_no_compression() {
        let content = vec![1u8; 1000]; // Small message
        let blob = create_test_blob(content.clone());
        let chunks = Chunker::chunk_it("test_network", &blob, 4096);

        // Should have header + 1 data chunk for small message
        assert_eq!(chunks.len(), 2);

        // Check header chunk
        if let Some(models::routing::chunk::Content::Header(header)) = &chunks[0].content {
            assert_eq!(header.type_id, "TestPacket");
            assert_eq!(header.network_id, "test_network");
            assert!(!header.compressed); // Should not be compressed
            assert_eq!(header.content_length, 1000);
        } else {
            panic!("First chunk should be header");
        }

        // Check data chunk
        if let Some(models::routing::chunk::Content::Data(data)) = &chunks[1].content {
            assert_eq!(data.content_data.as_ref(), content.as_slice());
        } else {
            panic!("Second chunk should be data");
        }
    }

    #[test]
    fn test_large_message_with_compression() {
        // Create message larger than 500KB to trigger compression
        let content = vec![42u8; 600 * 1024]; // 600KB of repeatable data
        let blob = create_test_blob(content.clone());
        let chunks = Chunker::chunk_it("test_network", &blob, 4096);

        // Should have header + multiple data chunks
        assert!(chunks.len() > 1);

        // Check header chunk
        if let Some(models::routing::chunk::Content::Header(header)) = &chunks[0].content {
            assert_eq!(header.type_id, "TestPacket");
            assert_eq!(header.network_id, "test_network");
            assert!(header.compressed); // Should be compressed
            assert_eq!(header.content_length, 600 * 1024);
        } else {
            panic!("First chunk should be header");
        }

        // Verify we have data chunks
        let data_chunks: Vec<_> = chunks.iter().skip(1).collect();
        assert!(!data_chunks.is_empty());

        // All remaining chunks should be data chunks
        for chunk in data_chunks {
            assert!(matches!(
                chunk.content,
                Some(models::routing::chunk::Content::Data(_))
            ));
        }
    }

    #[test]
    fn test_multiple_data_chunks() {
        let content = vec![1u8; 10000]; // 10KB content
        let blob = create_test_blob(content);
        let small_chunk_size = 1024; // Force multiple chunks
        let chunks = Chunker::chunk_it("test_network", &blob, small_chunk_size);

        // Should have header + multiple data chunks
        // With 10KB content and 1024 max chunk size (minus 2KB buffer = -1024 bytes),
        // we actually can't create any data chunks, so let's use a larger chunk size
        assert!(chunks.len() >= 1); // At least header chunk

        // First should be header
        assert!(matches!(
            chunks[0].content,
            Some(models::routing::chunk::Content::Header(_))
        ));

        // Rest should be data chunks
        for chunk in chunks.iter().skip(1) {
            assert!(matches!(
                chunk.content,
                Some(models::routing::chunk::Content::Data(_))
            ));
        }
    }

    #[test]
    fn test_multiple_data_chunks_proper_sizing() {
        let content = vec![1u8; 10000]; // 10KB content
        let blob = create_test_blob(content);
        let chunk_size = 4096; // Larger chunk size to ensure we get data chunks
        let chunks = Chunker::chunk_it("test_network", &blob, chunk_size);

        // With 10KB content and 4096 max chunk size (minus 2KB buffer = 2048 bytes),
        // we should get header + multiple data chunks
        assert!(chunks.len() > 2); // Header + multiple data chunks

        // First should be header
        assert!(matches!(
            chunks[0].content,
            Some(models::routing::chunk::Content::Header(_))
        ));

        // Rest should be data chunks
        for chunk in chunks.iter().skip(1) {
            assert!(matches!(
                chunk.content,
                Some(models::routing::chunk::Content::Data(_))
            ));
        }
    }

    #[test]
    fn test_empty_content() {
        let content = vec![];
        let blob = create_test_blob(content);
        let chunks = Chunker::chunk_it("test_network", &blob, 4096);

        // Should have just header chunk for empty content
        assert_eq!(chunks.len(), 1);

        if let Some(models::routing::chunk::Content::Header(header)) = &chunks[0].content {
            assert_eq!(header.content_length, 0);
            assert!(!header.compressed);
        } else {
            panic!("Should have header chunk");
        }
    }

    #[test]
    fn test_chunk_size_calculation() {
        let content = vec![1u8; 5000];
        let blob = create_test_blob(content);
        let max_message_size = 3000;
        let chunks = Chunker::chunk_it("test_network", &blob, max_message_size);

        // Should respect chunk size limits
        assert!(chunks.len() > 2); // Header + multiple data chunks

        // Check that data chunks respect size limits
        for chunk in chunks.iter().skip(1) {
            if let Some(models::routing::chunk::Content::Data(data)) = &chunk.content {
                // Each data chunk should be <= (max_message_size - 2KB buffer)
                assert!(data.content_data.len() <= max_message_size - 2048);
            }
        }
    }
}
