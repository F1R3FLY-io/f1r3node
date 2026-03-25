use std::collections::BTreeMap;

use crypto::rust::hash::blake2b256::Blake2b256;

const CONTINUATION_ENCODING_VERSION: u8 = 3;

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
pub enum ContinuationVisibility {
    Private,
    Public,
}

impl ContinuationVisibility {
    fn to_u8(self) -> u8 {
        match self {
            ContinuationVisibility::Private => 0,
            ContinuationVisibility::Public => 1,
        }
    }

    fn from_u8(value: u8) -> Result<Self, ContinuationStoreError> {
        match value {
            0 => Ok(ContinuationVisibility::Private),
            1 => Ok(ContinuationVisibility::Public),
            _ => Err(ContinuationStoreError::InvalidFormat(format!(
                "unknown continuation visibility: {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BridgeFinalityPhase {
    Pending,
    Finalized,
    Retrying,
    RescueQueued,
}

impl BridgeFinalityPhase {
    fn to_u8(self) -> u8 {
        match self {
            BridgeFinalityPhase::Pending => 0,
            BridgeFinalityPhase::Finalized => 1,
            BridgeFinalityPhase::Retrying => 2,
            BridgeFinalityPhase::RescueQueued => 3,
        }
    }

    fn from_u8(value: u8) -> Result<Self, ContinuationStoreError> {
        match value {
            0 => Ok(BridgeFinalityPhase::Pending),
            1 => Ok(BridgeFinalityPhase::Finalized),
            2 => Ok(BridgeFinalityPhase::Retrying),
            3 => Ok(BridgeFinalityPhase::RescueQueued),
            _ => Err(ContinuationStoreError::InvalidFormat(format!(
                "unknown bridge finality phase: {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BridgeRewardPolicy {
    None,
    Fixed(u64),
}

impl BridgeRewardPolicy {
    fn reward_amount(&self) -> u64 {
        match self {
            BridgeRewardPolicy::None => 0,
            BridgeRewardPolicy::Fixed(amount) => *amount,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BridgeContinuationMetadata {
    pub lane_id: String,
    pub source_domain: String,
    pub target_domain: String,
    pub message_nonce: u64,
    pub finality_phase: BridgeFinalityPhase,
    pub deadline_epoch: Option<u64>,
    pub reward_policy: BridgeRewardPolicy,
    pub rescue_epoch: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContinuationSubtype {
    Standard,
    Bridge(BridgeContinuationMetadata),
}

impl ContinuationSubtype {
    pub fn as_bridge(&self) -> Option<&BridgeContinuationMetadata> {
        match self {
            ContinuationSubtype::Standard => None,
            ContinuationSubtype::Bridge(metadata) => Some(metadata),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationStatus {
    Active,
    Completed,
    Expired,
    Failed,
}

impl ContinuationStatus {
    fn to_u8(self) -> u8 {
        match self {
            ContinuationStatus::Active => 0,
            ContinuationStatus::Completed => 1,
            ContinuationStatus::Expired => 2,
            ContinuationStatus::Failed => 3,
        }
    }

    fn from_u8(value: u8) -> Result<Self, ContinuationStoreError> {
        match value {
            0 => Ok(ContinuationStatus::Active),
            1 => Ok(ContinuationStatus::Completed),
            2 => Ok(ContinuationStatus::Expired),
            3 => Ok(ContinuationStatus::Failed),
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
    pub visibility: ContinuationVisibility,
    pub bounty: Option<u64>,
    pub expires_at_epoch: Option<u64>,
    pub subtype: ContinuationSubtype,
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
        put_u8(&mut out, self.visibility.to_u8());
        put_opt_u64(&mut out, self.bounty);
        put_opt_u64(&mut out, self.expires_at_epoch);
        serialize_continuation_subtype(&self.subtype, &mut out);
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
        let visibility = ContinuationVisibility::from_u8(read_u8(bytes, &mut idx)?)?;
        let bounty = read_opt_u64(bytes, &mut idx)?;
        let expires_at_epoch = read_opt_u64(bytes, &mut idx)?;
        let subtype = deserialize_continuation_subtype(bytes, &mut idx)?;
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
            visibility,
            bounty,
            expires_at_epoch,
            subtype,
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
        put_u8(&mut root_payload, self.visibility.to_u8());
        put_opt_u64(&mut root_payload, self.bounty);
        put_opt_u64(&mut root_payload, self.expires_at_epoch);
        serialize_continuation_subtype(&self.subtype, &mut root_payload);
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
    pub visibility: ContinuationVisibility,
    pub bounty: Option<u64>,
    pub ttl_epochs: Option<u64>,
    pub subtype: ContinuationSubtype,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuationStoreSnapshot {
    pub epoch: u64,
    pub entries: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContinuationStoreError {
    AlreadyExists(ContinuationHandle),
    NotFound(ContinuationHandle),
    NotBridgeContinuation(ContinuationHandle),
    VersionMismatch {
        handle: ContinuationHandle,
        expected: u64,
        actual: u64,
    },
    EpochRegression {
        current: u64,
        attempted: u64,
    },
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
    epoch: u64,
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
            visibility: create.visibility,
            bounty: create.bounty,
            expires_at_epoch: create.ttl_epochs.map(|ttl| self.epoch.saturating_add(ttl)),
            subtype: create.subtype,
            status: ContinuationStatus::Active,
            version: 1,
            state_root: Vec::new(),
            storage_bytes: 0,
        };
        continuation.refresh_derived_metadata();

        self.total_storage_bytes = self
            .total_storage_bytes
            .saturating_add(continuation.storage_bytes);
        self.entries.insert(create.handle, continuation.clone());

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

    pub fn set_epoch(&mut self, epoch: u64) -> Result<(), ContinuationStoreError> {
        if epoch < self.epoch {
            return Err(ContinuationStoreError::EpochRegression {
                current: self.epoch,
                attempted: epoch,
            });
        }
        self.epoch = epoch;
        Ok(())
    }

    pub fn current_epoch(&self) -> u64 {
        self.epoch
    }

    pub fn expire_due(&mut self) -> Result<Vec<PersistedContinuation>, ContinuationStoreError> {
        let handles_to_expire: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(handle, continuation)| {
                if continuation.status == ContinuationStatus::Active
                    && continuation
                        .expires_at_epoch
                        .is_some_and(|epoch| epoch <= self.epoch)
                {
                    Some(handle.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut expired = Vec::with_capacity(handles_to_expire.len());
        for handle in handles_to_expire.into_iter() {
            let version = self.current_version(&handle)?;
            expired.push(self.expire_with_version(&handle, version)?);
        }

        Ok(expired)
    }

    pub fn public_active_continuations(&self) -> Vec<PersistedContinuation> {
        self.entries
            .values()
            .filter(|continuation| {
                continuation.status == ContinuationStatus::Active
                    && continuation.visibility == ContinuationVisibility::Public
            })
            .cloned()
            .collect()
    }

    pub fn bridge_active_continuations_sorted_by_deadline(&self) -> Vec<PersistedContinuation> {
        let mut items: Vec<_> = self
            .entries
            .values()
            .filter_map(|continuation| {
                if continuation.status != ContinuationStatus::Active {
                    return None;
                }
                continuation
                    .subtype
                    .as_bridge()
                    .map(|_| continuation.clone())
            })
            .collect();

        items.sort_by(|a, b| {
            let a_bridge = a.subtype.as_bridge().expect("bridge subtype must exist");
            let b_bridge = b.subtype.as_bridge().expect("bridge subtype must exist");
            a_bridge
                .deadline_epoch
                .unwrap_or(u64::MAX)
                .cmp(&b_bridge.deadline_epoch.unwrap_or(u64::MAX))
                .then(a_bridge.message_nonce.cmp(&b_bridge.message_nonce))
                .then(a.handle.cmp(&b.handle))
        });
        items
    }

    pub fn bridge_active_continuations_sorted_by_reward(&self) -> Vec<PersistedContinuation> {
        let mut items: Vec<_> = self
            .entries
            .values()
            .filter_map(|continuation| {
                if continuation.status != ContinuationStatus::Active {
                    return None;
                }
                continuation
                    .subtype
                    .as_bridge()
                    .map(|_| continuation.clone())
            })
            .collect();

        items.sort_by(|a, b| {
            let a_bridge = a.subtype.as_bridge().expect("bridge subtype must exist");
            let b_bridge = b.subtype.as_bridge().expect("bridge subtype must exist");
            b_bridge
                .reward_policy
                .reward_amount()
                .cmp(&a_bridge.reward_policy.reward_amount())
                .then(
                    a_bridge
                        .deadline_epoch
                        .unwrap_or(u64::MAX)
                        .cmp(&b_bridge.deadline_epoch.unwrap_or(u64::MAX)),
                )
                .then(a.handle.cmp(&b.handle))
        });
        items
    }

    pub fn bridge_rescue_candidates(&self, epoch: u64) -> Vec<PersistedContinuation> {
        self.entries
            .values()
            .filter_map(|continuation| {
                if continuation.status != ContinuationStatus::Active {
                    return None;
                }
                let bridge = continuation.subtype.as_bridge()?;
                if bridge
                    .rescue_epoch
                    .is_some_and(|rescue_epoch| rescue_epoch <= epoch)
                {
                    Some(continuation.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn update_bridge_finality_phase_with_version(
        &mut self,
        handle: &ContinuationHandle,
        expected_version: u64,
        finality_phase: BridgeFinalityPhase,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        let existing = self.load(handle)?;
        if existing.subtype.as_bridge().is_none() {
            return Err(ContinuationStoreError::NotBridgeContinuation(
                handle.clone(),
            ));
        }

        self.update_with_version(handle, expected_version, ContinuationStatus::Active, |c| {
            if let ContinuationSubtype::Bridge(bridge) = &mut c.subtype {
                bridge.finality_phase = finality_phase;
            }
        })
    }

    pub fn snapshot(&self) -> ContinuationStoreSnapshot {
        ContinuationStoreSnapshot {
            epoch: self.epoch,
            entries: self
                .entries
                .values()
                .map(|entry| entry.to_bytes())
                .collect(),
        }
    }

    pub fn restore(snapshot: ContinuationStoreSnapshot) -> Result<Self, ContinuationStoreError> {
        let mut entries = BTreeMap::new();
        let mut total_storage_bytes = 0u64;
        for encoded in snapshot.entries.into_iter() {
            let continuation = PersistedContinuation::from_bytes(&encoded)?;
            total_storage_bytes = total_storage_bytes.saturating_add(continuation.storage_bytes);
            if entries
                .insert(continuation.handle.clone(), continuation)
                .is_some()
            {
                return Err(ContinuationStoreError::InvalidFormat(
                    "duplicate continuation handle in snapshot".to_string(),
                ));
            }
        }

        Ok(Self {
            entries,
            total_storage_bytes,
            epoch: snapshot.epoch,
        })
    }

    pub fn update_state(
        &mut self,
        handle: &ContinuationHandle,
        serialized_state: Vec<u8>,
        gas_limit_per_step: i64,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        let expected_version = self.current_version(handle)?;
        self.update_with_version(handle, expected_version, ContinuationStatus::Active, |c| {
            c.serialized_state = serialized_state.clone();
            c.gas_limit_per_step = gas_limit_per_step;
        })
    }

    pub fn update_state_with_version(
        &mut self,
        handle: &ContinuationHandle,
        expected_version: u64,
        serialized_state: Vec<u8>,
        gas_limit_per_step: i64,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update_with_version(handle, expected_version, ContinuationStatus::Active, |c| {
            c.serialized_state = serialized_state.clone();
            c.gas_limit_per_step = gas_limit_per_step;
        })
    }

    pub fn complete(
        &mut self,
        handle: &ContinuationHandle,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        let expected_version = self.current_version(handle)?;
        self.update_with_version(
            handle,
            expected_version,
            ContinuationStatus::Completed,
            |_| {},
        )
    }

    pub fn complete_with_version(
        &mut self,
        handle: &ContinuationHandle,
        expected_version: u64,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update_with_version(
            handle,
            expected_version,
            ContinuationStatus::Completed,
            |_| {},
        )
    }

    pub fn expire(
        &mut self,
        handle: &ContinuationHandle,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        let expected_version = self.current_version(handle)?;
        self.update_with_version(
            handle,
            expected_version,
            ContinuationStatus::Expired,
            |_| {},
        )
    }

    pub fn expire_with_version(
        &mut self,
        handle: &ContinuationHandle,
        expected_version: u64,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update_with_version(
            handle,
            expected_version,
            ContinuationStatus::Expired,
            |_| {},
        )
    }

    pub fn fail_with_version(
        &mut self,
        handle: &ContinuationHandle,
        expected_version: u64,
    ) -> Result<PersistedContinuation, ContinuationStoreError> {
        self.update_with_version(handle, expected_version, ContinuationStatus::Failed, |_| {})
    }

    pub fn total_storage_bytes(&self) -> u64 {
        self.total_storage_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn handles(&self) -> Vec<ContinuationHandle> {
        self.entries.keys().cloned().collect()
    }

    fn current_version(&self, handle: &ContinuationHandle) -> Result<u64, ContinuationStoreError> {
        self.entries
            .get(handle)
            .map(|continuation| continuation.version)
            .ok_or_else(|| ContinuationStoreError::NotFound(handle.clone()))
    }

    fn update_with_version<F>(
        &mut self,
        handle: &ContinuationHandle,
        expected_version: u64,
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

        if existing.version != expected_version {
            return Err(ContinuationStoreError::VersionMismatch {
                handle: handle.clone(),
                expected: expected_version,
                actual: existing.version,
            });
        }

        let from_status = existing.status;
        let valid_transition = match (from_status, next_status) {
            (ContinuationStatus::Active, ContinuationStatus::Active)
            | (ContinuationStatus::Active, ContinuationStatus::Completed)
            | (ContinuationStatus::Active, ContinuationStatus::Expired)
            | (ContinuationStatus::Active, ContinuationStatus::Failed) => true,
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

fn serialize_continuation_subtype(subtype: &ContinuationSubtype, out: &mut Vec<u8>) {
    match subtype {
        ContinuationSubtype::Standard => {
            put_u8(out, 0);
        }
        ContinuationSubtype::Bridge(metadata) => {
            put_u8(out, 1);
            put_bytes(out, metadata.lane_id.as_bytes());
            put_bytes(out, metadata.source_domain.as_bytes());
            put_bytes(out, metadata.target_domain.as_bytes());
            put_u64(out, metadata.message_nonce);
            put_u8(out, metadata.finality_phase.to_u8());
            put_opt_u64(out, metadata.deadline_epoch);
            match metadata.reward_policy {
                BridgeRewardPolicy::None => {
                    put_u8(out, 0);
                }
                BridgeRewardPolicy::Fixed(amount) => {
                    put_u8(out, 1);
                    put_u64(out, amount);
                }
            }
            put_opt_u64(out, metadata.rescue_epoch);
        }
    }
}

fn deserialize_continuation_subtype(
    input: &[u8],
    idx: &mut usize,
) -> Result<ContinuationSubtype, ContinuationStoreError> {
    let subtype_tag = read_u8(input, idx)?;
    match subtype_tag {
        0 => Ok(ContinuationSubtype::Standard),
        1 => {
            let lane_id = String::from_utf8(read_vec(input, idx)?).map_err(|e| {
                ContinuationStoreError::InvalidFormat(format!(
                    "invalid UTF-8 in bridge lane_id: {}",
                    e
                ))
            })?;
            let source_domain = String::from_utf8(read_vec(input, idx)?).map_err(|e| {
                ContinuationStoreError::InvalidFormat(format!(
                    "invalid UTF-8 in bridge source_domain: {}",
                    e
                ))
            })?;
            let target_domain = String::from_utf8(read_vec(input, idx)?).map_err(|e| {
                ContinuationStoreError::InvalidFormat(format!(
                    "invalid UTF-8 in bridge target_domain: {}",
                    e
                ))
            })?;
            let message_nonce = read_u64(input, idx)?;
            let finality_phase = BridgeFinalityPhase::from_u8(read_u8(input, idx)?)?;
            let deadline_epoch = read_opt_u64(input, idx)?;
            let reward_policy = match read_u8(input, idx)? {
                0 => BridgeRewardPolicy::None,
                1 => BridgeRewardPolicy::Fixed(read_u64(input, idx)?),
                marker => {
                    return Err(ContinuationStoreError::InvalidFormat(format!(
                        "unknown bridge reward policy marker: {}",
                        marker
                    )))
                }
            };
            let rescue_epoch = read_opt_u64(input, idx)?;
            Ok(ContinuationSubtype::Bridge(BridgeContinuationMetadata {
                lane_id,
                source_domain,
                target_domain,
                message_nonce,
                finality_phase,
                deadline_epoch,
                reward_policy,
                rescue_epoch,
            }))
        }
        _ => Err(ContinuationStoreError::InvalidFormat(format!(
            "unknown continuation subtype tag: {}",
            subtype_tag
        ))),
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

fn put_opt_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(v) => {
            put_u8(out, 1);
            put_u64(out, v);
        }
        None => put_u8(out, 0),
    }
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

fn read_opt_u64(input: &[u8], idx: &mut usize) -> Result<Option<u64>, ContinuationStoreError> {
    match read_u8(input, idx)? {
        0 => Ok(None),
        1 => Ok(Some(read_u64(input, idx)?)),
        marker => Err(ContinuationStoreError::InvalidFormat(format!(
            "invalid optional u64 marker: {}",
            marker
        ))),
    }
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
            visibility: ContinuationVisibility::Private,
            bounty: None,
            ttl_epochs: None,
            subtype: ContinuationSubtype::Standard,
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
        *tampered
            .last_mut()
            .expect("encoded bytes should not be empty") ^= 0x01;
        let decode_result = PersistedContinuation::from_bytes(&tampered);
        assert!(decode_result.is_err());
    }

    #[test]
    fn update_with_version_should_reject_stale_version() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(6, vec![1, 2, 3]))
            .expect("create should succeed");

        let stale_update = store.update_state_with_version(&created.handle, 0, vec![7, 8, 9], 100);
        assert!(matches!(
            stale_update,
            Err(ContinuationStoreError::VersionMismatch {
                expected: 0,
                actual: 1,
                ..
            })
        ));
    }

    #[test]
    fn failed_status_should_be_terminal() {
        let mut store = InMemoryContinuationStore::default();
        let created = store
            .create(sample_create(7, vec![1]))
            .expect("create should succeed");
        let failed = store
            .fail_with_version(&created.handle, created.version)
            .expect("fail transition should succeed");
        assert_eq!(failed.status, ContinuationStatus::Failed);

        let retry = store.update_state(&created.handle, vec![2], 100);
        assert!(matches!(
            retry,
            Err(ContinuationStoreError::InvalidTransition {
                from: ContinuationStatus::Failed,
                to: ContinuationStatus::Active,
                ..
            })
        ));
    }

    #[test]
    fn public_active_continuations_should_filter_by_visibility_and_status() {
        let mut store = InMemoryContinuationStore::default();
        let private = store
            .create(sample_create(8, vec![1]))
            .expect("private continuation should be created");
        let public = store
            .create(CreateContinuation {
                handle: ContinuationHandle::new(b"deploy-origin".to_vec(), 9),
                origin_reference: b"origin-ref".to_vec(),
                serialized_state: vec![2],
                gas_limit_per_step: 100,
                funding_policy: FundingPolicy::ExecutorPays,
                visibility: ContinuationVisibility::Public,
                bounty: Some(10),
                ttl_epochs: None,
                subtype: ContinuationSubtype::Standard,
            })
            .expect("public continuation should be created");
        store
            .complete_with_version(&private.handle, private.version)
            .expect("completing private continuation should succeed");

        let public_queue = store.public_active_continuations();
        assert_eq!(public_queue.len(), 1);
        assert_eq!(public_queue[0].handle, public.handle);
        assert_eq!(public_queue[0].bounty, Some(10));
    }

    #[test]
    fn ttl_expiration_should_mark_due_continuations_expired() {
        let mut store = InMemoryContinuationStore::default();
        store.set_epoch(10).expect("epoch set should succeed");
        let created = store
            .create(CreateContinuation {
                handle: ContinuationHandle::new(b"deploy-origin".to_vec(), 10),
                origin_reference: b"origin-ref".to_vec(),
                serialized_state: vec![1, 2, 3],
                gas_limit_per_step: 100,
                funding_policy: FundingPolicy::ProducerOnly,
                visibility: ContinuationVisibility::Private,
                bounty: None,
                ttl_epochs: Some(5),
                subtype: ContinuationSubtype::Standard,
            })
            .expect("continuation creation should succeed");
        assert_eq!(created.expires_at_epoch, Some(15));

        store.set_epoch(14).expect("epoch set should succeed");
        let still_active = store.expire_due().expect("expiry sweep should succeed");
        assert!(still_active.is_empty());

        store.set_epoch(15).expect("epoch set should succeed");
        let expired = store.expire_due().expect("expiry sweep should succeed");
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].status, ContinuationStatus::Expired);
    }

    #[test]
    fn set_epoch_should_reject_regression() {
        let mut store = InMemoryContinuationStore::default();
        store.set_epoch(7).expect("epoch set should succeed");
        let regression = store.set_epoch(6);
        assert!(matches!(
            regression,
            Err(ContinuationStoreError::EpochRegression {
                current: 7,
                attempted: 6
            })
        ));
    }

    #[test]
    fn bridge_scheduler_ordering_should_support_deadline_and_reward_priorities() {
        let mut store = InMemoryContinuationStore::default();
        let make_bridge =
            |nonce: u64, deadline_epoch: Option<u64>, reward: u64| CreateContinuation {
                handle: ContinuationHandle::new(b"bridge-origin".to_vec(), nonce),
                origin_reference: b"origin-ref".to_vec(),
                serialized_state: vec![nonce as u8],
                gas_limit_per_step: 100,
                funding_policy: FundingPolicy::ExecutorPays,
                visibility: ContinuationVisibility::Public,
                bounty: Some(reward),
                ttl_epochs: None,
                subtype: ContinuationSubtype::Bridge(BridgeContinuationMetadata {
                    lane_id: "lane-a".to_string(),
                    source_domain: "eth".to_string(),
                    target_domain: "f1r3".to_string(),
                    message_nonce: nonce,
                    finality_phase: BridgeFinalityPhase::Pending,
                    deadline_epoch,
                    reward_policy: BridgeRewardPolicy::Fixed(reward),
                    rescue_epoch: None,
                }),
            };

        let c1 = store
            .create(make_bridge(1, Some(10), 5))
            .expect("bridge continuation should be created");
        let c2 = store
            .create(make_bridge(2, Some(5), 1))
            .expect("bridge continuation should be created");
        let c3 = store
            .create(make_bridge(3, Some(8), 20))
            .expect("bridge continuation should be created");

        let by_deadline = store.bridge_active_continuations_sorted_by_deadline();
        assert_eq!(by_deadline[0].handle, c2.handle);
        assert_eq!(by_deadline[1].handle, c3.handle);
        assert_eq!(by_deadline[2].handle, c1.handle);

        let by_reward = store.bridge_active_continuations_sorted_by_reward();
        assert_eq!(by_reward[0].handle, c3.handle);
        assert_eq!(by_reward[1].handle, c1.handle);
        assert_eq!(by_reward[2].handle, c2.handle);
    }

    #[test]
    fn bridge_rescue_candidates_should_return_due_entries() {
        let mut store = InMemoryContinuationStore::default();
        store
            .create(CreateContinuation {
                handle: ContinuationHandle::new(b"bridge-origin".to_vec(), 11),
                origin_reference: b"origin-ref".to_vec(),
                serialized_state: vec![1],
                gas_limit_per_step: 100,
                funding_policy: FundingPolicy::ExecutorPays,
                visibility: ContinuationVisibility::Public,
                bounty: Some(3),
                ttl_epochs: None,
                subtype: ContinuationSubtype::Bridge(BridgeContinuationMetadata {
                    lane_id: "lane-b".to_string(),
                    source_domain: "eth".to_string(),
                    target_domain: "f1r3".to_string(),
                    message_nonce: 11,
                    finality_phase: BridgeFinalityPhase::Retrying,
                    deadline_epoch: Some(20),
                    reward_policy: BridgeRewardPolicy::Fixed(3),
                    rescue_epoch: Some(7),
                }),
            })
            .expect("bridge continuation should be created");

        assert!(store.bridge_rescue_candidates(6).is_empty());
        let due = store.bridge_rescue_candidates(7);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].handle.nonce, 11);
    }

    #[test]
    fn snapshot_restore_should_preserve_runtime_state() {
        let mut store = InMemoryContinuationStore::default();
        store.set_epoch(5).expect("epoch set should succeed");

        let standard = store
            .create(CreateContinuation {
                handle: ContinuationHandle::new(b"restore-origin".to_vec(), 21),
                origin_reference: b"origin-ref".to_vec(),
                serialized_state: vec![1, 2, 3],
                gas_limit_per_step: 80,
                funding_policy: FundingPolicy::ProducerOnly,
                visibility: ContinuationVisibility::Private,
                bounty: None,
                ttl_epochs: Some(10),
                subtype: ContinuationSubtype::Standard,
            })
            .expect("standard continuation creation should succeed");

        let bridge = store
            .create(CreateContinuation {
                handle: ContinuationHandle::new(b"restore-origin".to_vec(), 22),
                origin_reference: b"origin-ref".to_vec(),
                serialized_state: vec![4, 5, 6],
                gas_limit_per_step: 90,
                funding_policy: FundingPolicy::ExecutorPays,
                visibility: ContinuationVisibility::Public,
                bounty: Some(7),
                ttl_epochs: Some(10),
                subtype: ContinuationSubtype::Bridge(BridgeContinuationMetadata {
                    lane_id: "lane-restore".to_string(),
                    source_domain: "eth".to_string(),
                    target_domain: "f1r3".to_string(),
                    message_nonce: 22,
                    finality_phase: BridgeFinalityPhase::Pending,
                    deadline_epoch: Some(8),
                    reward_policy: BridgeRewardPolicy::Fixed(7),
                    rescue_epoch: Some(9),
                }),
            })
            .expect("bridge continuation creation should succeed");

        store
            .update_state_with_version(&standard.handle, standard.version, vec![9, 9, 9], 81)
            .expect("state update should succeed");
        store
            .update_bridge_finality_phase_with_version(
                &bridge.handle,
                bridge.version,
                BridgeFinalityPhase::Retrying,
            )
            .expect("bridge phase update should succeed");

        let snapshot = store.snapshot();
        let restored =
            InMemoryContinuationStore::restore(snapshot).expect("restore should succeed");

        assert_eq!(restored.current_epoch(), 5);
        assert_eq!(store.total_storage_bytes(), restored.total_storage_bytes());
        assert_eq!(store.handles(), restored.handles());
        assert_eq!(
            store.bridge_active_continuations_sorted_by_deadline(),
            restored.bridge_active_continuations_sorted_by_deadline()
        );
        assert_eq!(
            store.load(&standard.handle),
            restored.load(&standard.handle)
        );
        assert_eq!(store.load(&bridge.handle), restored.load(&bridge.handle));
    }
}
