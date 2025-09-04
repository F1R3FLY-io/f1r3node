// See rholang/src/main/scala/coop/rchain/rholang/interpreter/merging/RholangMergingLogic.scala

use indexmap::IndexSet;
use rspace_plus_plus::rspace::errors::HistoryError;
use std::collections::{BTreeMap, HashSet};
use std::hash::Hash;

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rspace_plus_plus::rspace::hot_store_trie_action::TrieInsertAction;
use rspace_plus_plus::rspace::hot_store_trie_action::TrieInsertBinaryProduce;
use rspace_plus_plus::rspace::{
    hashing::{blake2b256_hash::Blake2b256Hash, stable_hash_provider},
    hot_store_trie_action::HotStoreTrieAction,
    internal::Datum,
    merger::channel_change::ChannelChange,
    serializers::serializers,
    trace::event::Produce,
};

use crate::rust::interpreter::rho_type::RhoNumber;

pub struct RholangMergingLogic;

impl RholangMergingLogic {
    /**
     * Transforms absolute values with the difference from initial values.
     *
     * Example for 3 state changes (A, B, C are channels, PSH is initial value/pre-state hash):
     *
     * Initial state (PSH):
     *   A = 10, B = 2, C = 20
     *
     * Final values:      Calculated diffs:
     * Change 0: A = 20   A = +10
     * Change 1: B = 5    B = +3
     * Change 2: A = 15   A = -5
     *           C = 10   C = -10
     *
     * @param channelValues Final values
     * @param getInitialValue Accessor to initial value
     */
    pub fn calculate_num_channel_diff<Key: Clone + Eq + Hash + Ord>(
        channel_values: Vec<BTreeMap<Key, i64>>,
        get_initial_value: impl Fn(&Key) -> Option<i64> + Send + Sync,
    ) -> Vec<BTreeMap<Key, i64>> {
        // First collect unique keys while preserving order
        let unique_keys: Vec<_> = channel_values
            .iter()
            .flat_map(|channel| channel.keys().cloned())
            .collect::<IndexSet<_>>()
            .into_iter()
            .collect();

        let mut state = unique_keys
            .iter()
            .map(|key| (key.clone(), get_initial_value(key).unwrap_or(0)))
            .collect::<BTreeMap<_, _>>();

        // Process each channel value map
        channel_values
            .into_iter()
            .map(|end_val_map| {
                let mut diffs = BTreeMap::new();

                for (ch, end_val) in end_val_map {
                    if let Some(prev_val) = state.get(&ch) {
                        let diff = end_val.wrapping_sub(*prev_val);
                        diffs.insert(ch.clone(), diff);
                        state.insert(ch, end_val);
                    }
                }
                diffs
            })
            .collect()
    }

    /**
     * Merge number channel value from multiple changes and base state.
     *
     * @param channelHash Channel hash
     * @param diff Difference from base state
     * @param changes Channel changes to calculate new random generator
     * @param getBaseData Base state value reader
     */
    pub fn calculate_number_channel_merge(
        channel_hash: &Blake2b256Hash,
        diff: i64,
        changes: &ChannelChange<Vec<u8>>,
        get_base_data: impl Fn(&Blake2b256Hash) -> Result<Vec<Datum<ListParWithRandom>>, HistoryError>,
    ) -> HotStoreTrieAction<Par, BindPattern, ListParWithRandom, TaggedContinuation> {
        // Read initial value of number channel from base state
        let init_val_opt = Self::convert_to_read_number(get_base_data)(&channel_hash);

        // Calculate number channel new value
        let init_num = init_val_opt.unwrap_or(0);
        let new_val = init_num + diff;

        // Calculate merged random generator (use only unique changes as input)
        let new_rnd = if changes.added.iter().collect::<HashSet<_>>().len() == 1 {
            // Single branch, just use available random generator
            Self::decode_rnd(changes.added.first().unwrap().to_vec())
        } else {
            // Multiple branches, merge random generators
            let rnd_added_sorted = changes
                .added
                .iter()
                .map(|bytes| Self::decode_rnd(bytes.to_vec()))
                .collect::<HashSet<_>>()
                .into_iter()
                .map(|rnd| (rnd.clone(), rnd.to_bytes()))
                .collect::<Vec<_>>();

            // Sort by bytes
            let mut sorted = rnd_added_sorted;
            sorted.sort_by(|a, b| a.1.cmp(&b.1));

            // Extract sorted random generators
            let sorted_rnds = sorted.into_iter().map(|(rnd, _)| rnd).collect::<Vec<_>>();

            // Merge the random generators
            Blake2b512Random::merge(sorted_rnds)
        };

        // Create final merged value
        let datum_encoded = Self::create_datum_encoded(&channel_hash, new_val, new_rnd);

        // Create update store action
        HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(
            TrieInsertBinaryProduce {
                hash: channel_hash.clone(),
                data: vec![datum_encoded],
            },
        ))
    }

    fn decode_rnd(par_with_rnd_encoded: Vec<u8>) -> Blake2b512Random {
        let datum: Datum<ListParWithRandom> = serializers::decode_datum(&par_with_rnd_encoded);
        let rnd = Blake2b512Random::from_bytes(&datum.a.random_state);
        rnd
    }

    pub fn get_number_with_rnd(par_with_rnd: &ListParWithRandom) -> (i64, Blake2b512Random) {
        assert!(
            par_with_rnd.pars.len() == 1,
            "Number channel should contain single Int term, found {:?}",
            par_with_rnd.pars
        );
        let num = RhoNumber::unapply(&par_with_rnd.pars[0]).unwrap();
        (
            num,
            Blake2b512Random::from_bytes(&par_with_rnd.random_state),
        )
    }

    fn create_datum_encoded(
        channel_hash: &Blake2b256Hash,
        num: i64,
        rnd: Blake2b512Random,
    ) -> Vec<u8> {
        // Create value with random generator
        let num_par = RhoNumber::create_par(num);
        let par_with_rnd = ListParWithRandom {
            pars: vec![num_par],
            random_state: rnd.to_bytes(),
        };

        // Create hash of the data
        let data_hash =
            stable_hash_provider::hash_produce(channel_hash.bytes(), &par_with_rnd, false);

        // Create produce
        let produce = Produce {
            channel_hash: channel_hash.clone(),
            hash: data_hash,
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
        };

        // Create datum
        let datum = Datum {
            a: par_with_rnd,
            persist: false,
            source: produce,
        };

        // Encode datum
        serializers::encode_datum(&datum)
    }

    /**
     * Converts function to get all data on a channel to function to get single number value.
     */
    pub fn convert_to_read_number<F>(get_data_func: F) -> impl Fn(&Blake2b256Hash) -> Option<i64>
    where
        F: Fn(&Blake2b256Hash) -> Result<Vec<Datum<ListParWithRandom>>, HistoryError>,
    {
        move |hash: &Blake2b256Hash| {
            let data = get_data_func(hash)
                .unwrap_or_else(|error| panic!("Error getting data: {:?}", error));
            assert!(
                data.len() <= 1,
                "To calculate difference on a number channel, single value is expected, found {:?}",
                data
            );
            data.first()
                .map(|datum| Self::get_number_with_rnd(&datum.a))
                .map(|(num, _)| num)
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DeployMergeableData {
    pub channels: Vec<NumberChannel>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NumberChannel {
    pub hash: Blake2b256Hash,
    pub diff: i64,
}

// See rholang/src/test/scala/coop/rchain/rholang/interpreter/merging/RholangMergingLogicSpec.scala
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_calculate_num_channel_diff() {
        /*
         *        A   B   C        A   B   C
         *  ---------------       ----------
         *  PSH  10      20
         *
         *   0.  20               10
         *   1.       3      ==>       3
         *   2.  15      10       -5     -10
         */

        // Create string hashes for readability
        let ch_a = "A".to_string();
        let ch_b = "B".to_string();
        let ch_c = "C".to_string();

        // Define initial values
        let mut init_values = HashMap::new();
        init_values.insert(ch_a.clone(), 10i64);
        init_values.insert(ch_c.clone(), 20i64);

        // Define the accessor function to get initial values
        let get_data_on_hash = |hash: String| -> Option<i64> { init_values.get(&hash).copied() };

        // Define input channel values (Vec of Maps)
        let mut input = Vec::new();

        // Map 0: {A -> 20}
        let mut map0 = BTreeMap::new();
        map0.insert(ch_a.clone(), 20i64);
        input.push(map0);

        // Map 1: {B -> 3}
        let mut map1 = BTreeMap::new();
        map1.insert(ch_b.clone(), 3i64);
        input.push(map1);

        // Map 2: {A -> 15, C -> 10}
        let mut map2 = BTreeMap::new();
        map2.insert(ch_a.clone(), 15i64);
        map2.insert(ch_c.clone(), 10i64);
        input.push(map2);

        // Calculate the differences
        let result =
            RholangMergingLogic::calculate_num_channel_diff(input, |arg0: &std::string::String| {
                get_data_on_hash(arg0.clone())
            });

        // Define expected results
        let mut expected = Vec::new();

        // Expected Map 0: {A -> 10}
        let mut expected_map0 = BTreeMap::new();
        expected_map0.insert(ch_a.clone(), 10i64);
        expected.push(expected_map0);

        // Expected Map 1: {B -> 3}
        let mut expected_map1 = BTreeMap::new();
        expected_map1.insert(ch_b.clone(), 3i64);
        expected.push(expected_map1);

        // Expected Map 2: {A -> -5, C -> -10}
        let mut expected_map2 = BTreeMap::new();
        expected_map2.insert(ch_a.clone(), -5i64);
        expected_map2.insert(ch_c.clone(), -10i64);
        expected.push(expected_map2);

        // Assert that the results match the expected values
        assert_eq!(result, expected);
    }
}
