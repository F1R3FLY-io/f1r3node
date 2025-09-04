// See shared/src/main/scala/coop/rchain/shared/Compression.scala

use lz4_flex::{compress_prepend_size, decompress_size_prepended};

/// Compression utilities using LZ4 algorithm
pub struct Compression;

impl Compression {
    /// Compress data using LZ4 high compression
    pub fn compress(content: &[u8]) -> Vec<u8> {
        compress_prepend_size(content)
    }

    /// Decompress data with known decompressed length
    /// Returns None if decompression fails or length doesn't match
    pub fn decompress(compressed: &[u8], decompressed_length: usize) -> Option<Vec<u8>> {
        match decompress_size_prepended(compressed) {
            Ok(decompressed) => {
                if decompressed.len() == decompressed_length {
                    Some(decompressed)
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }
}

/// Extension trait
pub trait CompressionOps {
    /// Compress this data
    fn compress(&self) -> Vec<u8>;

    /// Decompress this data with expected length
    fn decompress(&self, decompressed_length: usize) -> Option<Vec<u8>>;
}

impl CompressionOps for [u8] {
    fn compress(&self) -> Vec<u8> {
        Compression::compress(self)
    }

    fn decompress(&self, decompressed_length: usize) -> Option<Vec<u8>> {
        Compression::decompress(self, decompressed_length)
    }
}

impl CompressionOps for Vec<u8> {
    fn compress(&self) -> Vec<u8> {
        Compression::compress(self)
    }

    fn decompress(&self, decompressed_length: usize) -> Option<Vec<u8>> {
        Compression::decompress(self, decompressed_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Generate random byte arrays for testing
    fn generate_byte_array(size: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..size).map(|_| rng.gen_range(0..=255) as u8).collect()
    }

    #[test]
    fn should_compress_without_exceptions() {
        // Test various sizes
        for size in [10, 100, 1000, 10000, 50000] {
            for seed in 0..5 {
                let ar = generate_byte_array(size, seed);
                // Should not panic
                let _compressed = Compression::compress(&ar);
            }
        }
    }

    #[test]
    fn decompress_should_return_none_when_decompressed_length_is_incorrect() {
        for size in [10, 100, 1000, 5000] {
            for seed in 0..3 {
                let ar = generate_byte_array(size, seed);
                let compressed = Compression::compress(&ar);
                let illegal_length = ar.len() + 1;

                // Should not panic and should return None
                let result = Compression::decompress(&compressed, illegal_length);
                assert_eq!(result, None);
            }
        }
    }

    #[test]
    fn should_decompress_to_uncompressed_data() {
        for size in [10, 100, 1000, 10000] {
            for seed in 0..5 {
                let ar = generate_byte_array(size, seed);
                let compressed = Compression::compress(&ar);
                let back_again = Compression::decompress(&compressed, ar.len())
                    .expect("Decompression should succeed");

                assert_eq!(back_again.len(), ar.len());
                assert_eq!(back_again, ar);
            }
        }
    }

    #[test]
    fn should_compress_effectively_when_data_is_compressible() {
        // Create repeatable pattern
        let mut rng = StdRng::seed_from_u64(42);
        let word: Vec<u8> = (0..1024)
            .map(|_| (rng.gen_range(0..24) + 33) as u8)
            .collect();

        // Repeat the pattern 1024 times
        let mut ar = Vec::new();
        for _ in 0..1024 {
            ar.extend_from_slice(&word);
        }

        let compressed = Compression::compress(&ar);
        let ratio = compressed.len() as f64 / ar.len() as f64;

        assert!(ar.len() > compressed.len());
        // Allow some tolerance in compression ratio
        assert!(
            ratio < 0.01,
            "Compression ratio {} should be less than 0.01",
            ratio
        );
    }

    #[test]
    fn test_extension_trait_methods() {
        let data = vec![1u8, 2, 3, 4, 5, 1, 2, 3, 4, 5]; // Some repeatable data

        // Test Vec<u8> extension
        let compressed = data.compress();
        let decompressed = compressed.decompress(data.len()).unwrap();
        assert_eq!(data, decompressed);

        // Test slice extension
        let slice_data: &[u8] = &data;
        let compressed2 = slice_data.compress();
        let decompressed2 = compressed2.decompress(data.len()).unwrap();
        assert_eq!(data, decompressed2);
    }

    #[test]
    fn test_empty_data() {
        let empty_data: Vec<u8> = vec![];
        let compressed = Compression::compress(&empty_data);
        let decompressed = Compression::decompress(&compressed, 0).unwrap();
        assert_eq!(empty_data, decompressed);
    }

    #[test]
    fn test_single_byte() {
        let single_byte = vec![42u8];
        let compressed = Compression::compress(&single_byte);
        let decompressed = Compression::decompress(&compressed, 1).unwrap();
        assert_eq!(single_byte, decompressed);
    }
}
