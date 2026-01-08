use models::rhoapi::{
    connective, g_unforgeable, Bundle, Connective, EMap, GPrivate, GUnforgeable, KeyValuePair, Par,
};
use models::rust::rholang::sorter::ordering::Ordering;
use models::rust::rholang::sorter::par_sort_matcher::ParSortMatcher;
use models::rust::rholang::sorter::sortable::Sortable;
use models::rust::sorted_par_map::SortedParMap;
use models::rust::test_utils::test_utils::sort;
use models::rust::utils::{
    new_boundvar_par, new_eset_par, new_gint_par, new_gstring_par, new_par_from_par_set,
};
use prost::Message;
use std::collections::HashMap;

fn to_kv_pair(pair: (Par, Par)) -> KeyValuePair {
    KeyValuePair {
        key: Some(pair.0),
        value: Some(pair.1),
    }
}

fn serialize_emap(map: &SortedParMap) -> Vec<u8> {
    let kv_pairs: Vec<KeyValuePair> = map
        .sorted_list
        .iter()
        .map(|pair| to_kv_pair(pair.clone()))
        .collect();

    EMap {
        kvs: kv_pairs,
        locally_free: vec![],
        connective_used: false,
        remainder: None,
    }
    .encode_to_vec()
}

fn create_pars() -> Vec<(Par, Par)> {
    vec![
        (
            new_gint_par(7, vec![], false),
            new_gstring_par("Seven".to_string(), vec![], false),
        ),
        (
            new_gint_par(7, vec![], false),
            new_gstring_par("SeVen".to_string(), vec![], false),
        ),
        (
            new_boundvar_par(1, vec![], false),
            new_boundvar_par(0, vec![], false),
        ),
        (
            new_gint_par(2, vec![], false),
            new_eset_par(
                vec![
                    new_gint_par(2, vec![], false),
                    new_gint_par(1, vec![], false),
                ],
                vec![],
                false,
                None,
                vec![],
                false,
            ),
        ),
        (
            new_gint_par(2, vec![], false),
            new_eset_par(
                vec![new_gint_par(2, vec![], false)],
                vec![],
                false,
                None,
                vec![],
                false,
            ),
        ),
    ]
}

fn round_trip_test(par_map: &SortedParMap) {
    let serialized = serialize_emap(par_map);

    let expected_emap = EMap {
        kvs: par_map
            .sorted_list
            .iter()
            .map(|pair| to_kv_pair(pair.clone()))
            .collect(),
        locally_free: vec![],
        connective_used: false,
        remainder: None,
    };

    let deserialized = EMap::decode(&*serialized).expect("Failed to deserialize EMap");

    assert_eq!(
        deserialized, expected_emap,
        "Round trip serialization failed: deserialized EMap does not match expected"
    );
}

fn sample() -> SortedParMap {
    let pars = create_pars();
    SortedParMap::create_from_vec(pars)
}

#[test]
fn sorted_par_map_should_preserve_structure_during_round_trip_protobuf_serialization() {
    let sample_map = sample();
    round_trip_test(&sample_map);
}

#[test]
fn sorted_par_map_should_deduplicate_elements_where_last_seen_element_wins() {
    let mut expected_pairs = HashMap::new();
    expected_pairs.insert(
        new_gint_par(7, Vec::new(), false),
        new_gstring_par("SeVen".to_string(), Vec::new(), false),
    );
    expected_pairs.insert(
        new_gint_par(2, Vec::new(), false),
        new_par_from_par_set(
            vec![new_gint_par(2, Vec::new(), false)],
            Vec::new(),
            false,
            None,
        ),
    );

    let after_roundtrip_serialization = EMap::decode(&*serialize_emap(&sample()))
        .expect("Failed to deserialize EMap")
        .kvs;

    let result = expected_pairs.iter().all(|(key, value)| {
        after_roundtrip_serialization
            .iter()
            .any(|kv| kv.key.as_ref().unwrap() == key && kv.value.as_ref().unwrap() == value)
    });

    assert!(
        result,
        "Deduplication failed: expected pairs do not match serialized map"
    );
}

#[test]
fn sorted_par_map_should_preserve_ordering_during_serialization() {
    let reference_bytes = serialize_emap(&sample());

    for i in 1..=1000 {
        let serialized = serialize_emap(&sample());
        let res = serialized == reference_bytes; // Equivalent to util.Arrays.equals from Scala
        assert!(
            res,
            "Run #{} serialization failed: serialized bytes differ from reference bytes",
            i
        );
    }
}

#[test]
fn sorted_par_map_should_be_equal_when_it_is_equal() {
    let mut ps = HashMap::new();
    ps.insert(
        Par::default(),
        Par::default()
            .with_unforgeables(vec![
                GUnforgeable {
                    unf_instance: Some(g_unforgeable::UnfInstance::GPrivateBody(GPrivate {
                        id: vec![0],
                    })),
                },
                GUnforgeable {
                    unf_instance: Some(g_unforgeable::UnfInstance::GPrivateBody(
                        GPrivate::default(),
                    )),
                },
            ])
            .with_connective_used(true),
    );

    let map1 = SortedParMap::create_from_map(ps.clone());
    let map2 = SortedParMap::create_from_map(ps);

    assert_eq!(
        map1, map2,
        "SortedParMap instances created from the same elements should be equal"
    );
}

#[test]
fn sorted_par_map_should_keep_keys_sorted() {
    let sorted_par_map_keys = vec![
        Par::default().with_connective_used(true),
        Par::default().with_bundles(vec![Bundle {
            body: Some(Par::default()),
            write_flag: true,
            read_flag: false,
        }]),
        Par::default().with_connectives(vec![Connective {
            connective_instance: Some(connective::ConnectiveInstance::ConnBool(false)),
        }]),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(g_unforgeable::UnfInstance::GPrivateBody(GPrivate::default())),
        }]),
        Par::default(),
    ];

    let sorted_par_map = SortedParMap::create_from_vec(
        sorted_par_map_keys
            .iter()
            .cloned()
            .zip(vec![Par::default(); sorted_par_map_keys.len()])
            .collect(),
    );

    let keys = sorted_par_map.keys();

    assert_eq!(
        keys,
        Ordering::sort_pars(&keys),
        "Keys in SortedParMap are not sorted correctly"
    );
}

#[test]
fn sorted_par_map_should_sort_all_input() {
    let keys: Vec<Par> = create_pars().iter().map(|(key, _)| key.clone()).collect();
    let values: Vec<Par> = create_pars()
        .iter()
        .map(|(_, value)| value.clone())
        .collect();
    let kvs: Vec<(Par, Par)> = keys.into_iter().zip(values.into_iter()).collect();

    let map = SortedParMap::create_from_vec(kvs.clone());

    if let Some((unsorted_key, _)) = kvs.first() {
        let sorted_key = sort(unsorted_key);

        check_sorted_input(
            |key| map.clone().remove(key),
            unsorted_key.clone(),
            sorted_key.clone(),
        );
        check_sorted_input(
            |key| map.contains(key),
            unsorted_key.clone(),
            sorted_key.clone(),
        );
        check_sorted_input(
            |key| map.apply(key),
            unsorted_key.clone(),
            sorted_key.clone(),
        );
        check_sorted_input(
            |key| map.get_or_else(key, Par::default()),
            unsorted_key.clone(),
            sorted_key.clone(),
        );
    }

    let unsorted_keys: Vec<Par> = kvs.iter().map(|(key, _)| key.clone()).collect();
    let sorted_keys = Ordering::sort_pars(&unsorted_keys);
    check_sorted_input(
        |keys| map.clone().remove_multiple(keys),
        unsorted_keys,
        sorted_keys,
    );
}

#[test]
fn sorted_par_map_should_preserve_sortedness_for_all_operations() {
    let pars1 = create_pars();
    let pars2 = create_pars();

    let pairs1: Vec<(Par, Par)> = pars1
        .clone()
        .into_iter()
        .zip(pars2.clone().into_iter())
        .map(|((k1, v1), _)| (k1, v1)) // take pairs from the first list
        .collect();

    let pairs2: Vec<(Par, Par)> = pars2
        .clone()
        .into_iter()
        .zip(pars1.clone().into_iter())
        .map(|((k1, v1), _)| (k1, v1)) // take pairs from the second list
        .collect();

    let map1 = SortedParMap::create_from_vec(pairs1);
    let map2 = SortedParMap::create_from_vec(pairs2);

    check_sorted_par_map(&map1);
    check_sorted_par_map(&map2);

    //map1 operations
    if let Some((key, value)) = map1.sorted_list.first() {
        check_sorted_par_map(&map1.clone().insert((key.clone(), value.clone())));

        check_sorted_par_map(&map1.clone().remove(key.clone()));
    }

    //map2 operations
    if let Some((key, value)) = map2.sorted_list.first() {
        check_sorted_par_map(&map2.clone().insert((key.clone(), value.clone())));

        check_sorted_par_map(&map2.clone().remove(key.clone()));
    }
}

fn check_sorted_par_map(sorted_par_map: &SortedParMap) {
    check_sorted_vector_par(&sorted_par_map.keys());

    sorted_par_map.values().iter().for_each(|value| {
        assert!(
            is_sorted(value),
            "Value in SortedParMap is not sorted: {:?}",
            value
        );
    });

    sorted_par_map.sorted_list.iter().for_each(|(key, value)| {
        assert!(
            is_sorted(key),
            "Key in SortedParMap is not sorted: {:?}",
            key
        );
        assert!(
            is_sorted(value),
            "Value in SortedParMap is not sorted: {:?}",
            value
        );

        let fetched_value = sorted_par_map
            .get(key.clone())
            .expect("Key not found in SortedParMap");

        assert!(
            is_sorted(&fetched_value),
            "Fetched value in SortedParMap is not sorted: {:?}",
            fetched_value
        );
    });
}

fn check_sorted_vector_par(iterable: &Vec<Par>) {
    iterable.iter().for_each(|par| {
        assert!(
            is_sorted(par),
            "Element in iterable is not sorted: {:?}",
            par
        );
    });

    let sorted_iterable = Ordering::sort_pars(iterable);

    assert_eq!(
        iterable, &sorted_iterable,
        "Iterable is not sorted: {:?}",
        iterable
    );
}

fn check_sorted_input<A, B, F>(f: F, unsorted: A, sorted: A)
where
    A: Clone + PartialEq + std::fmt::Debug,
    B: PartialEq + std::fmt::Debug,
    F: Fn(A) -> B,
{
    assert_eq!(
        f(sorted.clone()),
        f(unsorted.clone()),
        "Function output for sorted and unsorted input differs"
    );
}

fn is_sorted(par: &Par) -> bool {
    let sorted_par = ParSortMatcher::sort_match(par);
    par == &sorted_par.term
}
