// See block-storage/src/main/scala/coop/rchain/blockstorage/KeyValueBlockStore.scala

use prost::Message;

use models::casper::{ApprovedBlockProto, BlockMessageProto};
use models::rust::casper::protocol::casper_message::{ApprovedBlock, BlockMessage};
use models::rust::{block_hash::BlockHash, casper::pretty_printer::PrettyPrinter};
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::store::key_value_store::{KeyValueStore, KvStoreError};

pub struct KeyValueBlockStore {
    store: Box<dyn KeyValueStore>,
    store_approved_block: Box<dyn KeyValueStore>,
    approved_block_key: [u8; 1],
}

impl Clone for KeyValueBlockStore {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone_box(),
            store_approved_block: self.store_approved_block.clone_box(),
            approved_block_key: self.approved_block_key,
        }
    }
}

impl KeyValueBlockStore {
    pub fn new(
        store: Box<dyn KeyValueStore>,
        store_approved_block: Box<dyn KeyValueStore>,
    ) -> Self {
        Self {
            store,
            store_approved_block,
            approved_block_key: [42],
        }
    }

    pub async fn create_from_kvm(kvm: &mut dyn KeyValueStoreManager) -> Result<Self, KvStoreError> {
        let store = kvm.store("blocks".to_string()).await?;
        let store_approved_block = kvm.store("blocks-approved".to_string()).await?;
        Ok(Self::new(store, store_approved_block))
    }

    fn error_block(hash: BlockHash, cause: String) -> String {
        format!(
            "Block decoding error, hash {}. Cause: {}",
            PrettyPrinter::build_string_bytes(&hash),
            cause
        )
    }

    pub fn get(&self, block_hash: &BlockHash) -> Result<Option<BlockMessage>, KvStoreError> {
        let bytes = self.store.get_one(&block_hash.to_vec())?;
        if bytes.is_none() {
            return Ok(None);
        }
        let bytes = bytes.unwrap();
        let block_proto = Self::bytes_to_block_proto(&bytes)?;
        let block = BlockMessage::from_proto(block_proto);
        match block {
            Ok(block) => Ok(Some(block)),
            Err(err) => Err(KvStoreError::SerializationError(Self::error_block(
                block_hash.clone(),
                err.to_string(),
            ))),
        }
    }

    /**
     * See block-storage/src/main/scala/coop/rchain/blockstorage/BlockStoreSyntax.scala
     *
     * Get block, "unsafe" because method expects block already in the block store.
     */
    pub fn get_unsafe(&self, block_hash: &BlockHash) -> BlockMessage {
        let err_msg = format!(
            "BlockStore is missing hash: {}",
            PrettyPrinter::build_string_bytes(&block_hash),
        );
        self.get(block_hash).expect(&err_msg).expect(&err_msg)
    }

    pub fn put(&mut self, block_hash: BlockHash, block: &BlockMessage) -> Result<(), KvStoreError> {
        let block_proto = block.to_proto();
        let bytes = Self::block_proto_to_bytes(&block_proto);
        self.store.put_one(block_hash.to_vec(), bytes)
    }

    pub fn put_block_message(&mut self, block: &BlockMessage) -> Result<(), KvStoreError> {
        self.put(block.block_hash.clone(), block)
    }

    pub fn contains(&self, block_hash: &BlockHash) -> Result<bool, KvStoreError> {
        match self.get(block_hash) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(err) => Err(err),
        }
    }

    fn error_approved_block(cause: String) -> String {
        format!("Approved block decoding error. Cause: {}", cause)
    }

    pub fn get_approved_block(&self) -> Result<Option<ApprovedBlock>, KvStoreError> {
        let bytes = self
            .store_approved_block
            .get_one(&self.approved_block_key.to_vec())?;

        if bytes.is_none() {
            return Ok(None);
        }

        let bytes = bytes.unwrap();
        let block_proto = ApprovedBlockProto::decode(&*bytes).map_err(|err| {
            KvStoreError::SerializationError(Self::error_approved_block(err.to_string()))
        })?;
        let block = ApprovedBlock::from_proto(block_proto).map_err(|err| {
            KvStoreError::SerializationError(Self::error_approved_block(err.to_string()))
        })?;
        Ok(Some(block))
    }

    pub fn put_approved_block(&mut self, block: &ApprovedBlock) -> Result<(), KvStoreError> {
        let block_proto = block.clone().to_proto();
        let bytes = block_proto.encode_to_vec();
        self.store_approved_block
            .put_one(self.approved_block_key.to_vec(), bytes)
    }

    fn bytes_to_block_proto(bytes: &[u8]) -> Result<BlockMessageProto, KvStoreError> {
        let bytes = Self::decompress_bytes(bytes);
        let decode_result = BlockMessageProto::decode(&*bytes);
        match decode_result {
            Ok(block_proto) => Ok(block_proto),
            Err(err) => Err(KvStoreError::SerializationError(err.to_string())),
        }
    }

    fn block_proto_to_bytes(block_proto: &BlockMessageProto) -> Vec<u8> {
        Self::compress_bytes(&block_proto.encode_to_vec())
    }

    fn compress_bytes(bytes: &[u8]) -> Vec<u8> {
        lz4_flex::compress_prepend_size(bytes)
    }

    fn decompress_bytes(bytes: &[u8]) -> Vec<u8> {
        lz4_flex::decompress_size_prepended(bytes).expect("Decompress of block failed")
    }
}

// See block-storage/src/test/scala/coop/rchain/blockstorage/KeyValueBlockStoreSpec.scala

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use std::sync::{Arc, Mutex};

    use models::rust::{
        block_implicits::block_element_gen,
        casper::protocol::casper_message::ApprovedBlockCandidate,
    };
    use shared::rust::{ByteBuffer, ByteString};

    use super::*;

    struct MockKeyValueStore {
        get_result: Option<ByteString>,
        input_keys: Arc<Mutex<Vec<ByteString>>>,
        input_puts: Arc<Mutex<Vec<ByteString>>>,
    }

    impl MockKeyValueStore {
        fn new(get_result: Option<Vec<u8>>) -> Self {
            Self {
                get_result,
                input_keys: Arc::new(Mutex::new(Vec::new())),
                input_puts: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn update_input_keys(&self, keys: Vec<ByteString>) {
            self.input_keys.lock().unwrap().extend(keys);
        }
    }

    impl KeyValueStore for MockKeyValueStore {
        fn get(&self, keys: &Vec<ByteBuffer>) -> Result<Vec<Option<ByteBuffer>>, KvStoreError> {
            self.update_input_keys(keys.iter().map(|k| k.clone()).collect());
            Ok(vec![self.get_result.clone()])
        }

        fn put(
            &mut self,
            kv_pairs: Vec<(shared::rust::ByteBuffer, shared::rust::ByteBuffer)>,
        ) -> Result<(), KvStoreError> {
            self.input_keys
                .lock()
                .unwrap()
                .extend(kv_pairs.iter().map(|(k, _)| k.clone()));
            self.input_puts
                .lock()
                .unwrap()
                .extend(kv_pairs.iter().map(|(_, v)| v.clone()));
            Ok(())
        }

        fn delete(&mut self, _keys: Vec<shared::rust::ByteBuffer>) -> Result<usize, KvStoreError> {
            todo!()
        }

        fn iterate(
            &self,
            _f: fn(shared::rust::ByteBuffer, shared::rust::ByteBuffer),
        ) -> Result<(), KvStoreError> {
            todo!()
        }

        fn clone_box(&self) -> Box<dyn KeyValueStore> {
            todo!()
        }

        fn to_map(
            &self,
        ) -> Result<
            std::collections::BTreeMap<shared::rust::ByteBuffer, shared::rust::ByteBuffer>,
            KvStoreError,
        > {
            todo!()
        }

        fn print_store(&self) -> () {
            todo!()
        }

        fn size_bytes(&self) -> usize {
            todo!()
        }
    }

    pub struct NotImplementedKV;

    impl KeyValueStore for NotImplementedKV {
        fn get(&self, _keys: &Vec<ByteBuffer>) -> Result<Vec<Option<ByteBuffer>>, KvStoreError> {
            todo!()
        }

        fn put(&mut self, _kv_pairs: Vec<(ByteBuffer, ByteBuffer)>) -> Result<(), KvStoreError> {
            todo!()
        }

        fn delete(&mut self, _keys: Vec<ByteBuffer>) -> Result<usize, KvStoreError> {
            todo!()
        }

        fn iterate(&self, _f: fn(ByteBuffer, ByteBuffer)) -> Result<(), KvStoreError> {
            todo!()
        }

        fn clone_box(&self) -> Box<dyn KeyValueStore> {
            todo!()
        }

        fn to_map(
            &self,
        ) -> Result<std::collections::BTreeMap<ByteBuffer, ByteBuffer>, KvStoreError> {
            todo!()
        }

        fn print_store(&self) -> () {
            todo!()
        }

        fn size_bytes(&self) -> usize {
            todo!()
        }
    }

    fn to_approved_block(block: BlockMessage) -> ApprovedBlock {
        let candidate = ApprovedBlockCandidate {
            block,
            required_sigs: 0,
        };
        ApprovedBlock {
            candidate,
            sigs: vec![],
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
          cases: 5,
          failure_persistence: None,
          .. ProptestConfig::default()
      })]

      /**
        * Block store tests.
        */
      #[test]
      fn block_store_should_get_data_from_underlying_key_value_store(block in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None),
        key_string in any::<String>()) {
          let block_bytes = KeyValueBlockStore::block_proto_to_bytes(&block.clone().to_proto());
          let kv = MockKeyValueStore::new(Some(block_bytes));
          let input_keys = Arc::clone(&kv.input_keys);
          let bs = KeyValueBlockStore::new(Box::new(kv), Box::new(NotImplementedKV));

          let key = key_string.into_bytes();
          let result = bs.get(&key.clone().into());
          assert!(result.is_ok());
          assert_eq!(*input_keys.lock().unwrap(), vec![key]);
          assert_eq!(result.unwrap(), Some(block));
      }

      #[test]
      fn block_store_should_not_get_data_if_not_exists_in_underlying_key_value_store(key_string in any::<String>()) {
          let kv = MockKeyValueStore::new(None);
          let bs = KeyValueBlockStore::new(Box::new(kv), Box::new(NotImplementedKV));
          let key = key_string.into_bytes();
          let result = bs.get(&key.into());
          assert!(result.is_ok());
          assert_eq!(result.unwrap(), None);
      }

      #[test]
      fn block_store_should_put_data_to_underlying_key_value_store(block in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None)) {
          let block_bytes = KeyValueBlockStore::block_proto_to_bytes(&block.clone().to_proto());
          let kv = MockKeyValueStore::new(Some(block_bytes.clone()));
          let input_keys = Arc::clone(&kv.input_keys);
          let input_puts = Arc::clone(&kv.input_puts);
          let mut bs = KeyValueBlockStore::new(Box::new(kv), Box::new(NotImplementedKV));

          let result = bs.put_block_message(&block);
          assert!(result.is_ok());
          assert_eq!(*input_keys.lock().unwrap(), vec![block.block_hash.to_vec()]);
          assert_eq!(*input_puts.lock().unwrap(), vec![block_bytes]);
      }

      /**
        * Approved block store
        */
      #[test]
      fn block_store_should_get_approved_block_from_underlying_key_value_store(block in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None)) {
          let approved_block = to_approved_block(block);
          let approved_block_bytes = approved_block.clone().to_proto().encode_to_vec();
          let kv = MockKeyValueStore::new(Some(approved_block_bytes));
          let input_keys = Arc::clone(&kv.input_keys);
          let bs = KeyValueBlockStore::new(Box::new(NotImplementedKV), Box::new(kv));

          let result = bs.get_approved_block();
          assert!(result.is_ok());
          assert_eq!(*input_keys.lock().unwrap(), vec![bs.approved_block_key]);
          assert_eq!(result.unwrap(), Some(approved_block));
      }

      #[test]
      fn block_store_should_not_get_approved_block_if_not_exists_in_underlying_key_value_store(_s in any::<String>()) {
          let kv = MockKeyValueStore::new(None);
          let bs = KeyValueBlockStore::new(Box::new(NotImplementedKV), Box::new(kv));
          let result = bs.get_approved_block();
          assert!(result.is_ok());
          assert_eq!(result.unwrap(), None);
      }

      #[test]
      fn block_store_should_put_approved_block_to_underlying_key_value_store(block in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None)) {
          let approved_block = to_approved_block(block);
          let approved_block_bytes = approved_block.clone().to_proto().encode_to_vec();
          let kv = MockKeyValueStore::new(Some(approved_block_bytes.clone()));
          let input_keys = Arc::clone(&kv.input_keys);
          let input_puts = Arc::clone(&kv.input_puts);
          let mut bs = KeyValueBlockStore::new(Box::new(NotImplementedKV), Box::new(kv));

          let result = bs.put_approved_block(&approved_block);
          assert!(result.is_ok());
          assert_eq!(*input_keys.lock().unwrap(), vec![bs.approved_block_key]);
          assert_eq!(*input_puts.lock().unwrap(), vec![approved_block_bytes]);
      }
    }
}
