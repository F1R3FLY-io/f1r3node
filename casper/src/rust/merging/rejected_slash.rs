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

impl RejectedSlash {
    /// Canonical key for dedup: the equivocator being slashed plus the
    /// original issuer of the rejected slash. Two rejected slashes with
    /// the same key refer to the same slashing event.
    pub fn dedup_key(&self) -> (BlockHash, Vec<u8>) {
        (
            self.invalid_block_hash.clone(),
            self.issuer_public_key.bytes.to_vec(),
        )
    }
}

/// Filter rejected slashes to those not already covered by own-detected
/// slashing deploys. Own slashes take priority so that when both the
/// proposer's `invalid_latest_messages` view and the merge both surface
/// a slash for V, only one slash lands in the merge block.
///
/// `own_slash_keys` is an iterator of `(invalid_block_hash, issuer_pubkey)`
/// pairs drawn from the proposer's slashing_deploys.
pub fn filter_recoverable<I>(rejected: Vec<RejectedSlash>, own_slash_keys: I) -> Vec<RejectedSlash>
where
    I: IntoIterator<Item = (BlockHash, Vec<u8>)>,
{
    let covered: HashSet<(BlockHash, Vec<u8>)> = own_slash_keys.into_iter().collect();
    rejected
        .into_iter()
        .filter(|rs| !covered.contains(&rs.dedup_key()))
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

    /// Attack 1 defense: if the proposer's own slashing pass already covers
    /// an equivocator V, the merge-rejected slash for V should be deduped
    /// out. Otherwise V gets double-slashed in the same block.
    #[test]
    fn own_detected_slash_covers_merge_rejected_duplicate() {
        let rejected = vec![mk_slash(1, 2)];
        let own_keys = std::iter::once((Bytes::from(vec![1u8; 32]), vec![2u8; 32]));
        let out = filter_recoverable(rejected, own_keys);
        assert!(
            out.is_empty(),
            "merge-rejected slash duplicating own slash must be dropped"
        );
    }

    /// Attack 1 defense: a merge-rejected slash for an equivocator V that
    /// the proposer's own `invalid_latest_messages` does NOT cover must
    /// survive dedup and be re-issued in the merge block. Without this,
    /// an attacker who sustains cheap conflicts starves the slash
    /// indefinitely.
    #[test]
    fn merge_rejected_slash_survives_when_not_covered_by_own() {
        let rejected = vec![mk_slash(1, 2)];
        let own_keys: Vec<(BlockHash, Vec<u8>)> = vec![];
        let out = filter_recoverable(rejected, own_keys);
        assert_eq!(out.len(), 1, "merge-rejected slash must survive dedup");
        assert_eq!(out[0].invalid_block_hash, Bytes::from(vec![1u8; 32]));
    }

    /// Attack 4 / repeated rejection: when multiple merge-rejected slashes
    /// refer to distinct equivocators, all must survive dedup even if the
    /// proposer's own slashing pass covers only some of them.
    #[test]
    fn mixed_coverage_keeps_uncovered_slashes() {
        let rejected = vec![
            mk_slash(1, 2), // covered by own
            mk_slash(3, 4), // not covered
            mk_slash(5, 6), // not covered
            mk_slash(7, 8), // covered by own
        ];
        let own_keys = vec![
            (Bytes::from(vec![1u8; 32]), vec![2u8; 32]),
            (Bytes::from(vec![7u8; 32]), vec![8u8; 32]),
        ];
        let out = filter_recoverable(rejected, own_keys);
        assert_eq!(out.len(), 2, "exactly the uncovered slashes must survive");
        let invalid_hashes: Vec<_> = out.iter().map(|rs| rs.invalid_block_hash.clone()).collect();
        assert!(invalid_hashes.contains(&Bytes::from(vec![3u8; 32])));
        assert!(invalid_hashes.contains(&Bytes::from(vec![5u8; 32])));
    }

    /// Dedup key discriminates by both fields: same equivocator under
    /// different issuers is a different slash (e.g., two honest proposers
    /// independently slashing V). Both should be kept; picking one over
    /// the other is a separate policy decision outside this layer.
    #[test]
    fn dedup_key_discriminates_by_issuer() {
        let rejected = vec![mk_slash(1, 2), mk_slash(1, 3)];
        let own_keys = std::iter::once((Bytes::from(vec![1u8; 32]), vec![2u8; 32]));
        let out = filter_recoverable(rejected, own_keys);
        assert_eq!(
            out.len(),
            1,
            "different-issuer slash for same equivocator must survive"
        );
        assert_eq!(out[0].issuer_public_key.bytes.to_vec(), vec![3u8; 32]);
    }

    /// Empty inputs produce empty output — the common non-slash merge path.
    /// Regression guard: block creators with no rejected slashes must
    /// return an empty list rather than panic or allocate spuriously.
    #[test]
    fn empty_inputs_produce_empty_output() {
        let out = filter_recoverable::<Vec<_>>(vec![], vec![]);
        assert!(out.is_empty());
    }
}
