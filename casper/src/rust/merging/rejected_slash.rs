use std::collections::HashSet;

use crypto::rust::public_key::PublicKey;
use models::rust::block_hash::BlockHash;

/// A slash system deploy whose containing chain was rejected by the merge.
/// The block creator uses this to re-issue the slash in the merge block
/// itself, ensuring the slash effect lands in the merged state regardless
/// of cost-optimal rejection of the source block's chain.
#[derive(Clone, Debug)]
pub struct RejectedSlash {
    pub invalid_block_hash: BlockHash,
    pub issuer_public_key: PublicKey,
    pub source_block_hash: BlockHash,
}

impl PartialEq for RejectedSlash {
    fn eq(&self, other: &Self) -> bool {
        self.invalid_block_hash == other.invalid_block_hash
            && self.issuer_public_key.bytes == other.issuer_public_key.bytes
    }
}

impl Eq for RejectedSlash {}

impl std::hash::Hash for RejectedSlash {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.invalid_block_hash.hash(state);
        self.issuer_public_key.bytes.hash(state);
    }
}

/// Filter rejected slashes for re-issuance by the merge proposer.
///
/// Two collapses happen here:
///
/// 1. **Drop already-covered equivocators.** If the proposer's own
///    slashing pass (`prepare_slashing_deploys`) already produces a
///    slash for the equivocator V, drop any merge-rejected slash for V.
///    The own-detected slash will land in the merge block — re-issuing
///    a redundant copy under the proposer's identity inflates body
///    size and wastes execution on a no-op slash (PoS slash is
///    keyed solely on `invalid_block_hash`, so a second slash for V
///    succeeds idempotently with no state change).
///
/// 2. **Dedup multiple rejected slashes for the same equivocator.**
///    When V1 and V2 both proposed slashes for V and both chains were
///    merge-rejected, the merge engine surfaces two `RejectedSlash`
///    entries — one per original issuer. The proposer is going to
///    re-sign both under their own identity, which would emit two
///    redundant SlashDeploys. Keep at most one survivor per
///    `invalid_block_hash`.
///
/// Output is sorted by `invalid_block_hash` so the surviving slash
/// for each equivocator is the same on every validator that runs the
/// same merge — required for body-hash determinism across replays.
pub fn filter_recoverable<I>(
    mut rejected: Vec<RejectedSlash>,
    own_invalid_block_hashes: I,
) -> Vec<RejectedSlash>
where
    I: IntoIterator<Item = BlockHash>,
{
    let covered: HashSet<BlockHash> = own_invalid_block_hashes.into_iter().collect();
    rejected.sort_by(|a, b| a.invalid_block_hash.cmp(&b.invalid_block_hash));
    let mut seen: HashSet<BlockHash> = HashSet::new();
    rejected
        .into_iter()
        .filter(|rs| !covered.contains(&rs.invalid_block_hash))
        .filter(|rs| seen.insert(rs.invalid_block_hash.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::bytes::Bytes;

    fn pk(byte: u8) -> PublicKey {
        PublicKey::from_bytes(&vec![byte; 32])
    }

    fn mk_slash(invalid_block_marker: u8, issuer_marker: u8) -> RejectedSlash {
        RejectedSlash {
            invalid_block_hash: Bytes::from(vec![invalid_block_marker; 32]),
            issuer_public_key: pk(issuer_marker),
            source_block_hash: Bytes::from(vec![0xFF; 32]),
        }
    }

    /// If the proposer's own slashing pass covers equivocator V, the
    /// merge-rejected slash for V is dropped — preventing two SlashDeploys
    /// for V (one own-detected, one re-issued) in the same block body.
    #[test]
    fn own_detected_slash_covers_merge_rejected_duplicate() {
        let rejected = vec![mk_slash(1, 2)];
        let own_invalid_block_hashes = std::iter::once(Bytes::from(vec![1u8; 32]));
        let out = filter_recoverable(rejected, own_invalid_block_hashes);
        assert!(
            out.is_empty(),
            "merge-rejected slash duplicating own slash must be dropped"
        );
    }

    /// A merge-rejected slash for an equivocator that the proposer's own
    /// `invalid_latest_messages` view does NOT cover must survive dedup
    /// and be re-issued in the merge block. Without this, an attacker who
    /// sustains cheap conflicts could starve slashing indefinitely.
    #[test]
    fn merge_rejected_slash_survives_when_not_covered_by_own() {
        let rejected = vec![mk_slash(1, 2)];
        let own_invalid_block_hashes: Vec<BlockHash> = vec![];
        let out = filter_recoverable(rejected, own_invalid_block_hashes);
        assert_eq!(out.len(), 1, "merge-rejected slash must survive dedup");
        assert_eq!(out[0].invalid_block_hash, Bytes::from(vec![1u8; 32]));
    }

    /// When multiple merge-rejected slashes refer to distinct equivocators,
    /// all uncovered ones must survive dedup, in deterministic order.
    /// Covered equivocators are dropped.
    #[test]
    fn mixed_coverage_keeps_uncovered_slashes() {
        let rejected = vec![
            mk_slash(1, 2), // covered by own
            mk_slash(3, 4), // not covered
            mk_slash(5, 6), // not covered
            mk_slash(7, 8), // covered by own
        ];
        let own_invalid_block_hashes = vec![Bytes::from(vec![1u8; 32]), Bytes::from(vec![7u8; 32])];
        let out = filter_recoverable(rejected, own_invalid_block_hashes);
        assert_eq!(out.len(), 2, "exactly the uncovered slashes must survive");
        // Sorted by invalid_block_hash for deterministic body composition.
        assert_eq!(out[0].invalid_block_hash, Bytes::from(vec![3u8; 32]));
        assert_eq!(out[1].invalid_block_hash, Bytes::from(vec![5u8; 32]));
    }

    /// Two merge-rejected slashes for the SAME equivocator from DIFFERENT
    /// original issuers (e.g., V1 and V2 both proposed slash chains for
    /// equivocator V and both got merge-rejected) must collapse to a
    /// single survivor. The proposer is going to re-sign whichever
    /// survives under their own identity, so emitting two SlashDeploys
    /// for V is wasted work — the second execution is a no-op against
    /// already-slashed PoS state.
    #[test]
    fn same_equivocator_across_issuers_dedups_to_one() {
        let rejected = vec![mk_slash(1, 2), mk_slash(1, 3)];
        let own_invalid_block_hashes: Vec<BlockHash> = vec![];
        let out = filter_recoverable(rejected, own_invalid_block_hashes);
        assert_eq!(
            out.len(),
            1,
            "two rejected slashes for the same equivocator must collapse to one"
        );
        assert_eq!(out[0].invalid_block_hash, Bytes::from(vec![1u8; 32]));
    }

    /// The proposer's own slash for E must drop a merge-rejected slash
    /// for E even when the rejected slash's original issuer is a
    /// different validator. The proposer re-signs every E1c slash under
    /// their own pk anyway, so the original issuer is provenance, not
    /// dedup-key material.
    #[test]
    fn own_detection_drops_rejected_from_other_issuer() {
        let rejected = vec![mk_slash(1, 99)]; // issuer = other validator
        let own_invalid_block_hashes = std::iter::once(Bytes::from(vec![1u8; 32])); // own slashes E
        let out = filter_recoverable(rejected, own_invalid_block_hashes);
        assert!(
            out.is_empty(),
            "own-detected slash for E must drop ANY merge-rejected slash for E, \
             regardless of original issuer. If this fails, dedup is keying on \
             issuer pk and will produce redundant SlashDeploys in the merge block body."
        );
    }

    /// Empty inputs produce empty output — the common non-slash merge path.
    /// Regression guard: block creators with no rejected slashes must
    /// return an empty list rather than panic or allocate spuriously.
    #[test]
    fn empty_inputs_produce_empty_output() {
        let out = filter_recoverable(Vec::<RejectedSlash>::new(), Vec::<BlockHash>::new());
        assert!(out.is_empty());
    }
}
