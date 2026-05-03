// Source of truth for the genesis-defined mergeable-channel tag identities.
//
// A mergeable tag is an unforgeable name (`Par`) derived deterministically
// from a (deployer-pubkey, timestamp) seed. The same seed values are also
// used to sign the corresponding genesis Rholang contract for those tags
// that are tied to a contract (e.g. NonNegativeNumber.rho), so casper's
// genesis-deploy code re-exports the constants from this module.
//
// Tags:
//   - NonNegativeNumber tag → IntegerAdd merge strategy. Used for vault
//     balance counters and gas accumulators.
//   - BitmaskOr tag → BitmaskOr merge strategy. Used for Registry.rho's
//     TreeHashMap interior-node bitmaps so concurrent registry inserts
//     into the same interior node don't conflict at multi-parent merge.

use std::collections::HashMap;

use crypto::rust::{
    hash::blake2b512_random::Blake2b512Random,
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use models::casper::DeployDataProto;
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::{GPrivate, GUnforgeable, Par};
use prost::Message;
use rspace_plus_plus::rspace::merger::merging_logic::MergeType;

pub const NON_NEGATIVE_NUMBER_PK: &str =
    "e33c9f1e925819d04733db4ec8539a84507c9e9abd32822059349449fe03997d";
pub const NON_NEGATIVE_NUMBER_TIMESTAMP: i64 = 1559156251792;

// Dedicated key for deriving the bitmask-OR mergeable tag's unforgeable
// name. Not used to sign any deploy; only seeds the RNG so the tag has
// an identity independent of any specific genesis contract.
pub const BITMASK_OR_TAG_PK: &str =
    "4d76b8e3f29a51c8d05e7b4f9a23c6e1d8b5f0a7c4e91b6d3a8f5c2e9b6d4a1c";
pub const BITMASK_OR_TAG_TIMESTAMP: i64 = 1762000000000;

// Dedicated key for deriving the MutexState mergeable tag's unforgeable
// name. Generated fresh via `openssl rand -hex 32` (2026-05-02). Same
// pattern as BITMASK_OR_TAG_PK — seeds the unforgeable-name RNG only;
// not used to sign any deploy.
pub const MUTEX_STATE_TAG_PK: &str =
    "d976a4490cbbf5733171a4549a28633d13fce6a65a29ce1f31a6d4f413b1bd67";
pub const MUTEX_STATE_TAG_TIMESTAMP: i64 = 1762100000000;

pub fn pub_key_from_hex(priv_key_hex: &str) -> PublicKey {
    let private_key =
        PrivateKey::from_bytes(&hex::decode(priv_key_hex).expect("invalid private key hex"));
    Secp256k1.to_public(&private_key)
}

fn unforgeable_name_rng(deployer: &PublicKey, timestamp: i64) -> Blake2b512Random {
    let seed = DeployDataProto {
        deployer: deployer.bytes.clone(),
        timestamp,
        ..Default::default()
    };
    Blake2b512Random::create_from_bytes(&seed.encode_to_vec())
}

fn tag_name(deployer_pk_hex: &str, timestamp: i64) -> Par {
    let pubkey = pub_key_from_hex(deployer_pk_hex);
    let mut rng = unforgeable_name_rng(&pubkey, timestamp);
    rng.next();
    let unforgeable_byte = rng.next();
    Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: unforgeable_byte.into_iter().map(|b| b as u8).collect(),
        })),
    }])
}

pub fn non_negative_mergeable_tag_name() -> Par {
    tag_name(NON_NEGATIVE_NUMBER_PK, NON_NEGATIVE_NUMBER_TIMESTAMP)
}

pub fn bitmask_or_mergeable_tag_name() -> Par {
    tag_name(BITMASK_OR_TAG_PK, BITMASK_OR_TAG_TIMESTAMP)
}

pub fn mutex_state_mergeable_tag_name() -> Par {
    tag_name(MUTEX_STATE_TAG_PK, MUTEX_STATE_TAG_TIMESTAMP)
}

/// Standard mergeable-tag registry installed at runtime startup. Maps each
/// genesis-defined tag `Par` to its merge strategy. Use this everywhere a
/// mergeable-tag table is needed unless a test specifically wants a custom
/// configuration.
pub fn default_mergeable_tags() -> HashMap<Par, MergeType> {
    let mut tags = HashMap::new();
    tags.insert(non_negative_mergeable_tag_name(), MergeType::IntegerAdd);
    tags.insert(bitmask_or_mergeable_tag_name(), MergeType::BitmaskOr);
    tags.insert(mutex_state_mergeable_tag_name(), MergeType::MutexState);
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collision-check regression: the MutexState tag's `Par` must not
    /// collide with the two existing tags. Catches a future regression
    /// where someone reuses (PK, timestamp) by mistake.
    #[test]
    fn mutex_state_tag_does_not_collide_with_existing_tags() {
        let nnn = non_negative_mergeable_tag_name();
        let bitmask = bitmask_or_mergeable_tag_name();
        let mutex_state = mutex_state_mergeable_tag_name();

        assert_ne!(mutex_state, nnn, "MutexState tag must differ from NonNegativeNumber tag");
        assert_ne!(mutex_state, bitmask, "MutexState tag must differ from BitmaskOr tag");
    }

    /// `default_mergeable_tags()` must contain exactly 3 distinct entries:
    /// IntegerAdd, BitmaskOr, MutexState.
    #[test]
    fn default_mergeable_tags_has_three_distinct_entries() {
        let tags = default_mergeable_tags();
        assert_eq!(tags.len(), 3, "expected 3 mergeable tags, got {}", tags.len());

        let mut merge_types: Vec<_> = tags.values().copied().collect();
        merge_types.sort();
        assert_eq!(
            merge_types,
            vec![MergeType::IntegerAdd, MergeType::BitmaskOr, MergeType::MutexState],
        );
    }
}
