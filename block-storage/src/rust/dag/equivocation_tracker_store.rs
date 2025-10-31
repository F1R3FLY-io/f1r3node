// See block-storage/src/main/scala/coop/rchain/blockstorage/dag/EquivocationTrackerStore.scala

use std::collections::{BTreeSet, HashSet};

use models::rust::{
    block_hash::BlockHashSerde,
    equivocation_record::{EquivocationRecord, SequenceNumber},
    validator::ValidatorSerde,
};
use shared::rust::store::{
    key_value_store::KvStoreError, key_value_typed_store::KeyValueTypedStore,
    key_value_typed_store_impl::KeyValueTypedStoreImpl,
};

#[derive(Clone)]
pub struct EquivocationTrackerStore {
    pub store: KeyValueTypedStoreImpl<(ValidatorSerde, SequenceNumber), BTreeSet<BlockHashSerde>>,
}

impl EquivocationTrackerStore {
    pub fn new(
        store: KeyValueTypedStoreImpl<(ValidatorSerde, SequenceNumber), BTreeSet<BlockHashSerde>>,
    ) -> Self {
        Self { store }
    }

    pub fn add(&self, record: EquivocationRecord) -> Result<(), KvStoreError> {
        self.store.put_one(
            (
                ValidatorSerde(record.equivocator),
                record.equivocation_base_block_seq_num,
            ),
            record
                .equivocation_detected_block_hashes
                .into_iter()
                .map(Into::into)
                .collect(),
        )
    }

    pub fn add_all(&mut self, records: Vec<EquivocationRecord>) -> Result<(), KvStoreError> {
        self.store.put(
            records
                .into_iter()
                .map(|record| {
                    (
                        (
                            ValidatorSerde(record.equivocator),
                            record.equivocation_base_block_seq_num,
                        ),
                        record
                            .equivocation_detected_block_hashes
                            .into_iter()
                            .map(Into::into)
                            .collect(),
                    )
                })
                .collect(),
        )
    }

    pub fn data(&self) -> Result<HashSet<EquivocationRecord>, KvStoreError> {
        self.store.to_map().map(|map| {
            map.into_iter()
                .map(
                    |(
                        (equivocator, equivocation_base_block_seq_num),
                        equivocation_detected_block_hashes,
                    )| {
                        EquivocationRecord::new(
                            equivocator.into(),
                            equivocation_base_block_seq_num,
                            equivocation_detected_block_hashes
                                .into_iter()
                                .map(Into::into)
                                .collect(),
                        )
                    },
                )
                .collect()
        })
    }
}
