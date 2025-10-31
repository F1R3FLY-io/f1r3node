// casper/src/rust/report_store.rs
// See casper/src/main/scala/coop/rchain/casper/ReportStore.scala

use prost::Message;
use std::sync::Arc;

// BlockEventInfo is defined in models/src/main/protobuf/DeployServiceCommon.proto line 224
// It's in the "casper" package, compiled via build.rs and included in models::casper
use models::casper::BlockEventInfo;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::{
    store::{
        key_value_store::{KeyValueStore, KvStoreError},
        key_value_typed_store::KeyValueTypedStore,
    },
    BitVector, ByteString,
};

/// Type alias matching Scala: type ReportStore[F[_]] = KeyValueTypedStore[F, ByteString, BlockEventInfo]
pub type ReportStore = CompressedBlockEventInfoStore;

/// Factory function matching Scala: def store[F[_]](kvm: KeyValueStoreManager[F]): F[ReportStore[F]]
pub async fn report_store(
    kvm: &mut impl KeyValueStoreManager,
) -> Result<ReportStore, KvStoreError> {
    let store = kvm.store("reporting-cache".to_string()).await?;
    Ok(CompressedBlockEventInfoStore::new(store))
}

/// Custom KeyValueTypedStore implementation with LZ4 compression codec
/// Matches Scala's blockEventInfoCodecCompressed:
///   scodec.codecs.bytes.xmap[BlockEventInfo](
///     bv => BlockEventInfo.parseFrom(decompressor.decompress(bv.toArray)),
///     bei => ByteVector(compressor.compress(bei.toByteArray))
///   )
#[derive(Clone)]
pub struct CompressedBlockEventInfoStore {
    store: Arc<dyn KeyValueStore>,
}

impl CompressedBlockEventInfoStore {
    pub fn new(store: Arc<dyn KeyValueStore>) -> Self {
        Self { store }
    }

    /// Encode key - ByteString passes through as-is (matching codecByteString)
    fn encode_key(&self, key: &ByteString) -> Result<BitVector, KvStoreError> {
        Ok(key.clone())
    }

    /// Decode key - ByteString passes through as-is
    fn decode_key(&self, encoded_key: &BitVector) -> Result<ByteString, KvStoreError> {
        Ok(encoded_key.clone())
    }

    /// Encode value with compression
    /// Matches: bei => ByteVector(compressor.compress(bei.toByteArray))
    fn encode_value(&self, value: &BlockEventInfo) -> Result<BitVector, KvStoreError> {
        let proto_bytes = value.encode_to_vec();
        Ok(compress_bytes(&proto_bytes))
    }

    /// Decode value with decompression
    /// Matches: bv => BlockEventInfo.parseFrom(decompressor.decompress(bv.toArray))
    fn decode_value(&self, encoded_value: &BitVector) -> Result<BlockEventInfo, KvStoreError> {
        let decompressed = decompress_bytes(encoded_value)?;
        BlockEventInfo::decode(&*decompressed).map_err(|e| {
            KvStoreError::SerializationError(format!("BlockEventInfo decoding error: {}", e))
        })
    }
}

impl KeyValueTypedStore<ByteString, BlockEventInfo> for CompressedBlockEventInfoStore {
    fn get(&self, keys: &Vec<ByteString>) -> Result<Vec<Option<BlockEventInfo>>, KvStoreError> {
        let keys_encoded = keys
            .iter()
            .map(|key| self.encode_key(key))
            .collect::<Result<Vec<_>, _>>()?;

        let values_bytes = self.store.get(&keys_encoded)?;

        values_bytes
            .iter()
            .map(|value_opt| {
                value_opt
                    .as_ref()
                    .map(|bytes| self.decode_value(bytes))
                    .transpose()
            })
            .collect()
    }

    fn put(&self, kv_pairs: Vec<(ByteString, BlockEventInfo)>) -> Result<(), KvStoreError> {
        let pairs_encoded = kv_pairs
            .iter()
            .map(
                |(key, value)| -> Result<(BitVector, BitVector), KvStoreError> {
                    Ok((self.encode_key(key)?, self.encode_value(value)?))
                },
            )
            .collect::<Result<Vec<_>, KvStoreError>>()?;

        self.store.put(pairs_encoded)
    }

    fn delete(&self, keys: Vec<ByteString>) -> Result<(), KvStoreError> {
        let keys_encoded = keys
            .iter()
            .map(|key| self.encode_key(key))
            .collect::<Result<Vec<_>, _>>()?;

        self.store.delete(keys_encoded)?;
        Ok(())
    }

    fn contains(&self, keys: Vec<ByteString>) -> Result<Vec<bool>, KvStoreError> {
        let keys_encoded = keys
            .iter()
            .map(|key| self.encode_key(key))
            .collect::<Result<Vec<_>, _>>()?;

        let results = self.store.get(&keys_encoded)?;
        Ok(results.iter().map(|r| r.is_some()).collect())
    }

    fn collect<F, T>(&self, mut f: F) -> Result<Vec<T>, KvStoreError>
    where
        F: FnMut((&ByteString, &BlockEventInfo)) -> Option<T>,
    {
        let store_map = self.store.to_map()?;
        let mut result = Vec::new();

        for (key_bytes, value_bytes) in store_map {
            let key = self.decode_key(&key_bytes)?;
            let value = self.decode_value(&value_bytes)?;

            if let Some(item) = f((&key, &value)) {
                result.push(item);
            }
        }

        Ok(result)
    }

    fn to_map(
        &self,
    ) -> Result<std::collections::HashMap<ByteString, BlockEventInfo>, KvStoreError> {
        let mut result = std::collections::HashMap::new();
        let store_map = self.store.to_map()?;

        for (key_bytes, value_bytes) in store_map {
            let key = self.decode_key(&key_bytes)?;
            let value = self.decode_value(&value_bytes)?;
            result.insert(key, value);
        }

        Ok(result)
    }
}

// Compression functions matching ReportStore.scala compressor/decompressor
// See also: KeyValueBlockStore.compress_bytes/decompress_bytes

/// Compress bytes using LZ4
/// Matches: compressor.compress(bytes)
fn compress_bytes(bytes: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(bytes)
}

/// Decompress bytes using LZ4
/// Matches: decompressor.decompress(bytes)
fn decompress_bytes(bytes: &[u8]) -> Result<Vec<u8>, KvStoreError> {
    lz4_flex::decompress_size_prepended(bytes)
        .map_err(|e| KvStoreError::SerializationError(format!("LZ4 decompression failed: {}", e)))
}
