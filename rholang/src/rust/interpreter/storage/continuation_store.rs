use std::collections::BTreeMap;

use crypto::rust::hash::blake2b256::Blake2b256;

const CONTINUATION_ENCODING_VERSION: u8 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ContinuationHandle {
    pub origin: Vec<u8>,
    pub nonce: u64,
}

impl ContinuationHandle {
    pub fn new(origin: Vec<u8>, nonce: u64) -> Self {
        Self { origin, nonce }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FundingPolicy {
    ProducerOnly,
    ExecutorPays,
}

impl FundingPolicy {
    fn to_u8(self) -> u8 {
        match self {
            FundingPolicy::ProducerOnly => 0,
            FundingPolicy::ExecutorPays => 1,
        }
    }

    fn from_u8(value: u8) -> Result<Self, ContinuationStoreError> {
        match value {
            0 => Ok(FundingPolicy::ProducerOnly),
            1 => Ok(FundingPolicy::ExecutorPays),
            _ => Err(ContinuationStoreError::InvalidFormat(format!(
                "unknown funding policy: {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationStatus {
    Active,
    Completed,
    Expired,
}

impl ContinuationStatus {
    fn to_u8(self) -> u8 {
        match self {
            ContinuationStatus::Active => 0,
            ContinuationStatus::Completed => 1,
            ContinuationStatus::Expired => 2,
        }
    }

    fn from_u8(value: u8) -> Result<Self, ContinuationStoreError> {
        match value {
            0 => Ok(ContinuationStatus::Active),
            1 => Ok(ContinuationStatus::Completed),
            2 => Ok(ContinuationStatus::Expired),
            _ => Err(ContinuationStoreError::InvalidFormat(format!(
                "unknown continuation status: {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedContinuation {
    pub handle: ContinuationHandle,
    pub origin_reference: Vec<u8>,
    pub serialized_state: Vec<u8>,
    pub gas_limit_per_step: i64,
    pub funding_policy: FundingPolicy,
    pub status: ContinuationStatus,
    pub version: u64,
    pub state_root: Vec<u8>,
    pub storage_bytes: u64,
}

impl PersistedContinuation {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u8(&mut out, CONTINUATION_ENCODING_VERSION);
        put_bytes(&mut out, &self.handle.origin);
        put_u64(&mut out, self.handle.nonce);
        put_bytes(&mut out, &self.origin_reference);
        put_bytes(&mut out, &self.serialized_state);
        put_i64(&mut out, self.gas_limit_per_step);
        put_u8(&mut out, self.funding_policy.to_u8());
        put_u8(&mut out, self.status.to_u8());
        put_u64(&mut out, self.version);
        put_bytes(&mut out, &self.state_root);
        put_u64(&mut out, self.storage_bytes);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContinuationStoreError> {
        let mut idx = 0usize;
        let version = read_u8(bytes, &mut idx)?;
        if version != CONTINUATION_ENCODING_VERSION {
            return Err(ContinuationStoreError::UnsupportedEncodingVersion(version));
        }

        let handle = ContinuationHandle {
            origin: read_vec(bytes, &mut idx)?,
            nonce: read_u64(bytes, &mut idx)?,
        };
        let origin_reference = read_vec(bytes, &mut idx)?;
        let serialized_state = read_vec(bytes, &mut idx)?;
        let gas_limit_per_step = read_i64(bytes, &mut idx)?;
        let funding_policy = FundingPolicy::from_u8(read_u8(bytes, &mut idx)?)?;
        let status = ContinuationStatus::from_u8(read_u8(bytes, &mut idx)?)?;
        let version = read_u64(bytes, &mut idx)?;
        let state_root = read_vec(bytes, &mut idx)?;
        let storage_bytes = read_u64(bytes, &mut idx)?;

        if idx != bytes.len() {
            return Err(ContinuationStoreError::InvalidFormat(
                "trailing bytes in continuation encoding".to_string(),
            ));
        }

        let continuation = Self {
            handle,
            origin_reference,
            serialized_state,
            gas_limit_per_step,
            funding_policy,
            status,
            version,
            state_root,
            storage_bytes,
        };

        if continuation.storage_bytes != bytes.len() as u64 {
            return Err(ContinuationStoreError::InvalidFormat(format!(
                "stored byte size {} does not match encoded size {}",
                continuation.storage_bytes,
                bytes.len()
            )));
        }

        let expected_root = continuation.compute_state_root();
        if expected_root != continuation.state_root {
            return Err(ContinuationStoreError::InvalidFormat(
                "state root integrity check failed".to_string(),
            ));
        }

        Ok(continuation)
    }

    fn compute_state_root(&self) -> Vec<u8> {
        let mut root_payload = Vec::new();
        put_u8(&mut root_payload, CONTINUATION_ENCODING_VERSION);
        put_bytes(&mut root_payload, &self.handle.origin);
        put_u64(&mut root_payload, self.handle.nonce);
        put_bytes(&mut root_payload, &self.origin_reference);
        put_bytes(&mut root_payload, &self.serialized_state);
        put_i64(&mut root_payload, self.gas_limit_per_step);
        put_u8(&mut root_payload, self.funding_policy.to_u8());
        put_u8(&mut root_payload, self.status.to_u8());
        put_u64(&mut root_payload, self.version);
        Blake2b256::hash(root_payload).to_vec()
    }

    fn refresh_derived_metadata(&mut self) {
        self.state_root = self.compute_state_root();
        self.storage_bytes = self.to_bytes().len() as u64;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CreateContinuation {
    pub handle: ContinuationHandle,
    pub origin_reference: Vec<u8>,
    pub serialized_state: Vec<u8>,
    pub gas_limit_per_step: i64,
    pub funding_policy: FundingPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContinuationStoreError {
    AlreadyExists(ContinuationHandle),
    NotFound(ContinuationHandle),
    InvalidTransition {
        handle: ContinuationHandle,
        from: ContinuationStatus,
        to: ContinuationStatus,
    },
    UnsupportedEncodingVersion(u8),
    InvalidFormat(String),
}

#[derive(Default, Clone)]
pub struct InMemoryContinuationStore {
    entries: BTreeMap<ContinuationHandle, PersistedContinuation>,
    total_storage_bytes: u64,
}

impl InMemoryContinuationStore {
    pub fn create(
        &mut self,
        create: CreateContinuation,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        if self.entries.contains_key(&create.handle) {
            return Err(ContinuationStoreError::AlreadyExists(create.handle));
        }

        let mut continuation = PersistedContinuation {
            handle: create.handle.clone(),
            origin_reference: create.origin_reference,
            serialized_state: create.serialized_state,
            gas_limit_per_step: create.gas_limit_per_step,
            funding_policy: create.funding_policy,
            status: ContinuationStatus::Active,
            version: 1,
            state_root: Vec::new(),
            storage_bytes: 0,
        };
        continuation.refresh_derived_metadata();

        self.total_storage_bytes = self
            .total_storage_bytes
            .saturating_add(continuation.storage_bytes);
        self.entries
            .insert(create.handle, continuation.clone());

        Ok(continuation)
    }

    pub fn load(
        &self,
        handle: &ContinuationHandle,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.entries
            .get(handle)
            .cloned()
            .ok_or_else(|| ContinuationStoreError::NotFound(handle.clone()))
    }

    pub fn update_state(
        &mut self,
        handle: &ContinuationHandle,
        serialized_state: Vec<u8>,
        gas_limit_per_step: i64,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update(handle, ContinuationStatus::Active, |c| {
            c.serialized_state = serialized_state.clone();
            c.gas_limit_per_step = gas_limit_per_step;
        })
    }

    pub fn complete(
        &mut self,
        handle: &ContinuationHandle,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update(handle, ContinuationStatus::Completed, |_| {})
    }

    pub fn expire(
        &mut self,
        handle: &ContinuationHandle,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update(handle, ContinuationStatus::Expired, |_| {})
    }

    pub fn total_storage_bytes(&self) -> u64 {
        self.total_storage_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    fn update<F>(
        &mut self,
        handle: &ContinuationHandle,
        next_status: ContinuationStatus,
        mut update_fn: F,
    ) -> Result<PersistedContinuation, ContinuationStoreError>
    where
        F: FnMut(&mut PersistedContinuation),
    {
        let existing = self
            .entries
            .get(handle)
            .cloned()
            .ok_or_else(|| ContinuationStoreError::NotFound(handle.clone()))?;

        let from_status = existing.status;
        let valid_transition = match (from_status, next_status) {
            (ContinuationStatus::Active, ContinuationStatus::Active)
            | (ContinuationStatus::Active, ContinuationStatus::Completed)
            | (ContinuationStatus::Active, ContinuationStatus::Expired) => true,
            _ => false,
        };
        if !valid_transition {
            return Err(ContinuationStoreError::InvalidTransition {
                handle: handle.clone(),
                from: from_status,
                to: next_status,
            });
        }

        let old_size = existing.storage_bytes;
        let mut updated = existing;
        updated.status = next_status;
        updated.version = updated.version.saturating_add(1);
        update_fn(&mut updated);
        updated.refresh_derived_metadata();

        self.total_storage_bytes = self
            .total_storage_bytes
            .saturating_sub(old_size)
            .saturating_add(updated.storage_bytes);
        self.entries.insert(handle.clone(), updated.clone());

        Ok(updated)
    }
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn read_u8(input: &[u8], idx: &mut usize) -> Result<u8, ContinuationStoreError> {
    if *idx + 1 > input.len() {
        return Err(ContinuationStoreError::InvalidFormat(
            "unexpected end of bytes while reading u8".to_string(),
        ));
    }
    let value = input[*idx];
    *idx += 1;
    Ok(value)
}

fn read_u32(input: &[u8], idx: &mut usize) -> Result<u32, ContinuationStoreError> {
    if *idx + 4 > input.len() {
        return Err(ContinuationStoreError::InvalidFormat(
            "unexpected end of bytes while reading u32".to_string(),
        ));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&input[*idx..*idx + 4]);
    *idx += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &[u8], idx: &mut usize) -> Result<u64, ContinuationStoreError> {
    if *idx + 8 > input.len() {
        return Err(ContinuationStoreError::InvalidFormat(
            "unexpected end of bytes while reading u64".to_string(),
        ));
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[*idx..*idx + 8]);
    *idx += 8;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(input: &[u8], idx: &mut usize) -> Result<i64, ContinuationStoreError> {
    if *idx + 8 > input.len() {
        return Err(ContinuationStoreError::InvalidFormat(
            "unexpected end of bytes while reading i64".to_string(),
        ));
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[*idx..*idx + 8]);
    *idx += 8;
    Ok(i64::from_le_bytes(bytes))
}

fn read_vec(input: &[u8], idx: &mut usize) -> Result<Vec<u8>, ContinuationStoreError> {
    let len = read_u32(input, idx)? as usize;
    if *idx + len > input.len() {
        return Err(ContinuationStoreError::InvalidFormat(
            "unexpected end of bytes while reading length-delimited bytes".to_string(),
        ));
    }
    let value = input[*idx..*idx + len].to_vec();
    *idx += len;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_create(nonce: u64, state: Vec<u8>) -> CreateContinuation {
        CreateContinuation {
            handle: ContinuationHandle::new(b"deploy-origin".to_vec(), nonce),
            origin_reference: b"origin-ref".to_vec(),
            serialized_state: state,
            gas_limit_per_step: 100,
            funding_policy: FundingPolicy::ProducerOnly,
        }
    }

    #[test]
    fn create_load_and_round_trip_serialization_should_work() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(1, vec![1, 2, 3]))
            .expect("create should succeed");
        let loaded = store.load(&created.handle).expect("load should succeed");
        assert_eq!(created, loaded);

        let bytes = loaded.to_bytes();
        let decoded =
            PersistedContinuation::from_bytes(&bytes).expect("continuation should decode");
        assert_eq!(loaded, decoded);
    }

    #[test]
    fn continuation_identity_should_remain_stable_across_versions() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(2, vec![1]))
            .expect("create should succeed");
        let updated = store
            .update_state(&created.handle, vec![1, 2], 120)
            .expect("update should succeed");

        assert_eq!(created.handle, updated.handle);
        assert_eq!(created.version + 1, updated.version);
    }

    #[test]
    fn state_root_should_change_across_state_transitions() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(3, vec![7]))
            .expect("create should succeed");
        let updated = store
            .update_state(&created.handle, vec![8], 100)
            .expect("update should succeed");
        let completed = store
            .complete(&created.handle)
            .expect("complete should succeed");

        assert_ne!(created.state_root, updated.state_root);
        assert_ne!(updated.state_root, completed.state_root);
    }

    #[test]
    fn storage_accounting_should_track_create_and_update_sizes() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(4, vec![1, 2, 3]))
            .expect("create should succeed");
        let after_create_total = store.total_storage_bytes();
        assert_eq!(after_create_total, created.storage_bytes);

        let updated = store
            .update_state(&created.handle, vec![1, 2, 3, 4, 5, 6], 100)
            .expect("update should succeed");
        let after_update_total = store.total_storage_bytes();
        assert_eq!(after_update_total, updated.storage_bytes);
        assert!(after_update_total > after_create_total);
    }

    #[test]
    fn serialization_should_be_stable_and_detect_tampering() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(5, vec![9, 9, 9]))
            .expect("create should succeed");
        let bytes1 = created.to_bytes();
        let bytes2 = created.to_bytes();
        assert_eq!(bytes1, bytes2);

        let mut tampered = bytes1;
        *tampered.last_mut().expect("encoded bytes should not be empty") ^= 0x01;
        let decode_result = PersistedContinuation::from_bytes(&tampered);
        assert!(decode_result.is_err());
    }
}
