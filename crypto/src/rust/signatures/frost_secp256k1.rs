use std::collections::{HashMap, HashSet};

use super::{schnorr_secp256k1::SchnorrSecp256k1, signatures_alg::SignaturesAlg};
use crate::rust::{hash::blake2b256::Blake2b256, private_key::PrivateKey, public_key::PublicKey};

pub const FROST_SECP256K1_ALGORITHM_NAME: &str = "frost-secp256k1";
pub const FROST_SECP256K1_SIGNING_DOMAIN: &[u8] = b"f1r3node/frost-secp256k1/signing/v1";
const FROST_SESSION_DOMAIN: &[u8] = b"f1r3node/frost-secp256k1/session/v1";
const FROST_COMMITMENT_DOMAIN: &[u8] = b"f1r3node/frost-secp256k1/commitment/v1";
const FROST_PARTIAL_DOMAIN: &[u8] = b"f1r3node/frost-secp256k1/partial/v1";

#[derive(Clone, Debug, PartialEq)]
pub struct FrostSecp256k1;

impl FrostSecp256k1 {
    pub fn name() -> String {
        FROST_SECP256K1_ALGORITHM_NAME.to_string()
    }

    pub fn signing_preimage(serialized_payload: &[u8]) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(FROST_SECP256K1_SIGNING_DOMAIN.len() + 8 + serialized_payload.len());
        out.extend_from_slice(FROST_SECP256K1_SIGNING_DOMAIN);
        out.extend_from_slice(&(serialized_payload.len() as u64).to_be_bytes());
        out.extend_from_slice(serialized_payload);
        out
    }

    pub fn domain_separated_hash(serialized_payload: &[u8]) -> Vec<u8> {
        Blake2b256::hash(Self::signing_preimage(serialized_payload))
    }
}

impl SignaturesAlg for FrostSecp256k1 {
    fn verify(&self, data: &[u8], signature: &[u8], pub_key: &[u8]) -> bool {
        SchnorrSecp256k1.verify(data, signature, pub_key)
    }

    fn sign(&self, data: &[u8], sec: &[u8]) -> Vec<u8> {
        // FROST-compatible coordinators produce final BIP-340 Schnorr signatures.
        SchnorrSecp256k1.sign(data, sec)
    }

    fn to_public(&self, sec: &PrivateKey) -> PublicKey {
        SchnorrSecp256k1.to_public(sec)
    }

    fn new_key_pair(&self) -> (PrivateKey, PublicKey) {
        SchnorrSecp256k1.new_key_pair()
    }

    fn name(&self) -> String {
        Self::name()
    }

    fn sig_length(&self) -> usize {
        SchnorrSecp256k1.sig_length()
    }

    fn eq(&self, other: &dyn SignaturesAlg) -> bool {
        self.name() == other.name()
    }

    fn box_clone(&self) -> Box<dyn SignaturesAlg> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrostParticipantId(pub u16);

#[derive(Clone, Debug)]
pub struct FrostParticipantShare {
    pub participant_id: FrostParticipantId,
    pub private_share: PrivateKey,
    pub public_share: PublicKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrostNonceCommitment {
    pub participant_id: FrostParticipantId,
    pub commitment: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct FrostSigningSession {
    pub session_id: [u8; 32],
    pub threshold: usize,
    pub participants: Vec<FrostParticipantId>,
    pub message_hash: [u8; 32],
    pub commitments: HashMap<FrostParticipantId, FrostNonceCommitment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrostPartialSignature {
    pub participant_id: FrostParticipantId,
    pub session_id: [u8; 32],
    pub bytes: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrostAggregateSignature {
    pub session_id: [u8; 32],
    pub signature: Vec<u8>,
    pub sig_algorithm: String,
}

pub trait FrostThresholdSignerProvider {
    fn begin_session(
        &self,
        message_hash: [u8; 32],
        threshold: usize,
        participants: &[FrostParticipantId],
    ) -> Result<FrostSigningSession, String>;

    fn create_partial_signature(
        &self,
        session: &FrostSigningSession,
        participant: FrostParticipantId,
    ) -> Result<FrostPartialSignature, String>;

    fn aggregate(
        &self,
        session: &FrostSigningSession,
        partials: &[FrostPartialSignature],
    ) -> Result<FrostAggregateSignature, String>;

    fn aggregate_public_key(&self) -> PublicKey;
}

/// In-memory dev/test coordinator for FROST-compatible secp256k1 threshold Schnorr flows.
/// It models session/commitment/partial/aggregate boundaries while leaving MPC orchestration off-node.
#[derive(Clone, Debug)]
pub struct MockFrostSecp256k1Coordinator {
    aggregate_private_key: PrivateKey,
    aggregate_public_key: PublicKey,
    shares: HashMap<FrostParticipantId, FrostParticipantShare>,
}

impl MockFrostSecp256k1Coordinator {
    pub fn new(total_participants: usize) -> Result<Self, String> {
        if total_participants == 0 {
            return Err("total_participants must be > 0".to_string());
        }

        let mut shares = HashMap::new();
        for i in 1..=total_participants {
            let (sk, pk) = SchnorrSecp256k1.new_key_pair();
            let participant_id = FrostParticipantId(i as u16);
            shares.insert(
                participant_id,
                FrostParticipantShare {
                    participant_id,
                    private_share: sk,
                    public_share: pk,
                },
            );
        }

        let (aggregate_private_key, aggregate_public_key) = SchnorrSecp256k1.new_key_pair();

        Ok(Self {
            aggregate_private_key,
            aggregate_public_key,
            shares,
        })
    }

    pub fn participant_ids(&self) -> Vec<FrostParticipantId> {
        let mut ids: Vec<_> = self.shares.keys().copied().collect();
        ids.sort();
        ids
    }

    fn hash32(parts: &[&[u8]]) -> [u8; 32] {
        let mut preimage = Vec::new();
        for part in parts {
            preimage.extend_from_slice(&(part.len() as u64).to_be_bytes());
            preimage.extend_from_slice(part);
        }
        let digest = Blake2b256::hash(preimage);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    fn expected_commitment(
        &self,
        session_id: &[u8; 32],
        participant: FrostParticipantId,
    ) -> Result<[u8; 32], String> {
        let share = self
            .shares
            .get(&participant)
            .ok_or_else(|| format!("unknown participant {}", participant.0))?;
        Ok(Self::hash32(&[
            FROST_COMMITMENT_DOMAIN,
            session_id,
            &participant.0.to_be_bytes(),
            &share.public_share.bytes,
        ]))
    }

    fn expected_partial(
        &self,
        session: &FrostSigningSession,
        participant: FrostParticipantId,
    ) -> Result<[u8; 32], String> {
        let commitment = session
            .commitments
            .get(&participant)
            .ok_or_else(|| format!("missing commitment for participant {}", participant.0))?;
        Ok(Self::hash32(&[
            FROST_PARTIAL_DOMAIN,
            &session.session_id,
            &participant.0.to_be_bytes(),
            &session.message_hash,
            &commitment.commitment,
        ]))
    }
}

impl FrostThresholdSignerProvider for MockFrostSecp256k1Coordinator {
    fn begin_session(
        &self,
        message_hash: [u8; 32],
        threshold: usize,
        participants: &[FrostParticipantId],
    ) -> Result<FrostSigningSession, String> {
        if threshold == 0 {
            return Err("threshold must be > 0".to_string());
        }
        if participants.is_empty() {
            return Err("participants must be non-empty".to_string());
        }
        if threshold > participants.len() {
            return Err(format!(
                "threshold {} exceeds participant count {}",
                threshold,
                participants.len()
            ));
        }

        let mut participants_sorted = participants.to_vec();
        participants_sorted.sort();
        participants_sorted.dedup();
        if participants_sorted.len() != participants.len() {
            return Err("participants must be unique".to_string());
        }
        for participant in &participants_sorted {
            if !self.shares.contains_key(participant) {
                return Err(format!("unknown participant {}", participant.0));
            }
        }

        let mut session_material = Vec::new();
        session_material.extend_from_slice(FROST_SESSION_DOMAIN);
        session_material.extend_from_slice(&message_hash);
        session_material.extend_from_slice(&(threshold as u64).to_be_bytes());
        for participant in &participants_sorted {
            session_material.extend_from_slice(&participant.0.to_be_bytes());
        }
        let session_id = Self::hash32(&[&session_material]);

        let mut commitments = HashMap::new();
        for participant in &participants_sorted {
            let commitment = self.expected_commitment(&session_id, *participant)?;
            commitments.insert(
                *participant,
                FrostNonceCommitment {
                    participant_id: *participant,
                    commitment,
                },
            );
        }

        Ok(FrostSigningSession {
            session_id,
            threshold,
            participants: participants_sorted,
            message_hash,
            commitments,
        })
    }

    fn create_partial_signature(
        &self,
        session: &FrostSigningSession,
        participant: FrostParticipantId,
    ) -> Result<FrostPartialSignature, String> {
        if !session.participants.contains(&participant) {
            return Err(format!(
                "participant {} is not part of this session",
                participant.0
            ));
        }

        let expected = self.expected_partial(session, participant)?;
        Ok(FrostPartialSignature {
            participant_id: participant,
            session_id: session.session_id,
            bytes: expected,
        })
    }

    fn aggregate(
        &self,
        session: &FrostSigningSession,
        partials: &[FrostPartialSignature],
    ) -> Result<FrostAggregateSignature, String> {
        if partials.len() < session.threshold {
            return Err(format!(
                "need at least {} partials, got {}",
                session.threshold,
                partials.len()
            ));
        }

        let mut seen = HashSet::new();
        for partial in partials.iter().take(session.threshold) {
            if partial.session_id != session.session_id {
                return Err("partial signature session_id mismatch".to_string());
            }
            if !seen.insert(partial.participant_id) {
                return Err(format!(
                    "duplicate partial for participant {}",
                    partial.participant_id.0
                ));
            }
            let expected = self.expected_partial(session, partial.participant_id)?;
            if partial.bytes != expected {
                return Err(format!(
                    "invalid partial signature for participant {}",
                    partial.participant_id.0
                ));
            }
        }

        let final_sig =
            SchnorrSecp256k1.sign(&session.message_hash, &self.aggregate_private_key.bytes);
        if final_sig.is_empty() {
            return Err("failed to produce aggregate Schnorr signature".to_string());
        }

        Ok(FrostAggregateSignature {
            session_id: session.session_id,
            signature: final_sig,
            sig_algorithm: FrostSecp256k1::name(),
        })
    }

    fn aggregate_public_key(&self) -> PublicKey {
        self.aggregate_public_key.clone()
    }
}

pub fn verify_frost_aggregate(
    message_hash: &[u8],
    aggregate_signature: &[u8],
    aggregate_public_key: &[u8],
) -> bool {
    SchnorrSecp256k1.verify(message_hash, aggregate_signature, aggregate_public_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_of_one_behaves_like_single_signer_schnorr() {
        let coordinator = MockFrostSecp256k1Coordinator::new(1).expect("coordinator");
        let participants = coordinator.participant_ids();
        let message_hash = [7u8; 32];

        let session = coordinator
            .begin_session(message_hash, 1, &participants)
            .expect("session");
        let partial = coordinator
            .create_partial_signature(&session, participants[0])
            .expect("partial");
        let aggregate = coordinator
            .aggregate(&session, &[partial])
            .expect("aggregate");

        assert_eq!(aggregate.sig_algorithm, FrostSecp256k1::name());
        assert!(verify_frost_aggregate(
            &session.message_hash,
            &aggregate.signature,
            &coordinator.aggregate_public_key().bytes
        ));
    }

    #[test]
    fn threshold_two_of_three_aggregate_verifies() {
        let coordinator = MockFrostSecp256k1Coordinator::new(3).expect("coordinator");
        let participants = coordinator.participant_ids();
        let message_hash = [11u8; 32];

        let session = coordinator
            .begin_session(message_hash, 2, &participants)
            .expect("session");
        let p1 = coordinator
            .create_partial_signature(&session, participants[0])
            .expect("p1");
        let p2 = coordinator
            .create_partial_signature(&session, participants[1])
            .expect("p2");

        let aggregate = coordinator
            .aggregate(&session, &[p1, p2])
            .expect("aggregate");
        assert!(verify_frost_aggregate(
            &session.message_hash,
            &aggregate.signature,
            &coordinator.aggregate_public_key().bytes
        ));
    }

    #[test]
    fn malformed_partial_is_rejected() {
        let coordinator = MockFrostSecp256k1Coordinator::new(3).expect("coordinator");
        let participants = coordinator.participant_ids();
        let message_hash = [3u8; 32];

        let session = coordinator
            .begin_session(message_hash, 2, &participants)
            .expect("session");
        let mut p1 = coordinator
            .create_partial_signature(&session, participants[0])
            .expect("p1");
        let p2 = coordinator
            .create_partial_signature(&session, participants[1])
            .expect("p2");

        p1.bytes[0] ^= 0x01;
        let err = coordinator.aggregate(&session, &[p1, p2]).unwrap_err();
        assert!(err.contains("invalid partial signature"));
    }

    #[test]
    fn malformed_aggregate_signature_fails_verification() {
        let coordinator = MockFrostSecp256k1Coordinator::new(2).expect("coordinator");
        let participants = coordinator.participant_ids();
        let message_hash = [19u8; 32];

        let session = coordinator
            .begin_session(message_hash, 2, &participants)
            .expect("session");
        let p1 = coordinator
            .create_partial_signature(&session, participants[0])
            .expect("p1");
        let p2 = coordinator
            .create_partial_signature(&session, participants[1])
            .expect("p2");
        let mut aggregate = coordinator
            .aggregate(&session, &[p1, p2])
            .expect("aggregate");
        aggregate.signature[0] ^= 0x01;

        assert!(!verify_frost_aggregate(
            &session.message_hash,
            &aggregate.signature,
            &coordinator.aggregate_public_key().bytes
        ));
    }
}
