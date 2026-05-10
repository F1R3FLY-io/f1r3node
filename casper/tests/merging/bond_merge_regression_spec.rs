// Regression coverage for the bonding-drift class.
//
// When one branch's deploy contributes a bonds-map write (additive: only
// that branch has it) and sibling branches contain heartbeat-only
// `closeBlock` writes (idempotent: every branch produces equivalent
// state for its own writes), the merge primitive must keep the additive
// branch's contribution. A lowest-Blake2b256-hash LWW combine does NOT —
// it picks one Datum's full Map and discards the others, so roughly
// N-1 / N of the time the bond contribution is lost.
//
// The fix is `MergeType::AdditiveSet` whose combine function unions the
// two Datums' Maps with per-key Last-Writer-Wins by sequence number,
// applied to a dedicated bonds channel in PoS.rhox.

use std::collections::HashMap;

use models::rhoapi::{ListParWithRandom, Par};
use rholang::rust::interpreter::rho_type::{RhoByteArray, RhoMap, RhoNumber};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, internal::Datum, trace::event::Produce,
};

/// Build a Datum whose payload encodes the given bonds map.
///
/// Mirrors PoS.rhox's `getBonds` shape: `Map[GByteArray pubkey, GInt stake]`.
/// The Datum's `source` is constructed deterministically from `marker` so
/// different test scenarios produce different Blake2b256 hashes — this is
/// what makes the LWW tiebreak deterministic for the regression assertion.
fn bonds_datum(
    bonds: &[(Vec<u8>, i64)],
    marker: u8,
    random_state: Vec<u8>,
) -> Datum<ListParWithRandom> {
    let mut hash_map: HashMap<Par, Par> = HashMap::new();
    for (pk_bytes, stake) in bonds {
        let key = RhoByteArray::create_par(pk_bytes.clone());
        let value = RhoNumber::create_par(*stake);
        hash_map.insert(key, value);
    }
    let map_par = RhoMap::create_par(hash_map);

    Datum {
        a: ListParWithRandom {
            pars: vec![map_par],
            random_state,
        },
        persist: false,
        source: Produce {
            channel_hash: Blake2b256Hash::new(&[marker; 32]),
            hash: Blake2b256Hash::new(&[marker; 32]),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        },
    }
}

fn extract_bonds(datum: &Datum<ListParWithRandom>) -> HashMap<Vec<u8>, i64> {
    assert_eq!(
        datum.a.pars.len(),
        1,
        "bonds Datum must hold exactly one Par (the Map)"
    );
    let map = RhoMap::unapply(&datum.a.pars[0]).expect("Datum payload must be a Map Par");
    map.into_iter()
        .map(|(k_par, v_par)| {
            let k =
                RhoByteArray::unapply(&k_par).expect("bonds map key must be GByteArray (pubkey)");
            let v = RhoNumber::unapply(&v_par).expect("bonds map value must be GInt (stake)");
            (k, v)
        })
        .collect()
}

const V1: &[u8] = &[0x01; 32];
const V2: &[u8] = &[0x02; 32];
const V3: &[u8] = &[0x03; 32];
const V4: &[u8] = &[0x04; 32];

/// Bond write survives merge against a heartbeat-only sibling.
///
/// Two sibling branches at the same height:
///   - Branch A processed V4's bond deploy. Its post-state `bondsCh`
///     Datum carries `{V1, V2, V3, V4}`.
///   - Branch B is a heartbeat-only sibling. Its `bondsCh` Datum carries
///     `{V1, V2, V3}` (no V4 — V4's bond never touched this branch's
///     local PoS state).
///
/// Multi-parent merge of these two `bondsCh` Datums must contain V4 in
/// the merged Map. Under `MergeType::AdditiveSet`, the merge is
/// set-union-with-per-key-LWW and V4 from branch A survives. Under
/// `MergeType::MutexState`, the merge is byte-level lowest-Blake2b256
/// LWW and branch B can win the tiebreak — V4 silently disappears from
/// the merged bonds map.
#[test]
fn bond_write_survives_merge_against_heartbeat_only_sibling() {
    let branch_a = bonds_datum(
        &[
            (V1.to_vec(), 100),
            (V2.to_vec(), 100),
            (V3.to_vec(), 100),
            (V4.to_vec(), 100),
        ],
        0xAA,
        vec![0x11; 64],
    );
    let branch_b = bonds_datum(
        &[(V1.to_vec(), 100), (V2.to_vec(), 100), (V3.to_vec(), 100)],
        0xBB,
        vec![0x22; 64],
    );

    let merged = rholang::rust::interpreter::merging::rholang_merging_logic::combine_mergeable_state_additive(
        &branch_a,
        &branch_b,
    );

    let bonds = extract_bonds(&merged);
    assert!(
        bonds.contains_key(V1),
        "V1 must survive merge: {:?}",
        bonds.keys().collect::<Vec<_>>(),
    );
    assert!(
        bonds.contains_key(V2),
        "V2 must survive merge: {:?}",
        bonds.keys().collect::<Vec<_>>(),
    );
    assert!(
        bonds.contains_key(V3),
        "V3 must survive merge: {:?}",
        bonds.keys().collect::<Vec<_>>(),
    );
    assert!(
        bonds.contains_key(V4),
        "V4 must survive merge of branch-with-bond + heartbeat-only \
         sibling. Merged bonds: {:?}",
        bonds.keys().collect::<Vec<_>>(),
    );
    assert_eq!(bonds.get(V4), Some(&100), "V4 stake must round-trip merge");
}
