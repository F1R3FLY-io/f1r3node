// See casper/src/test/scala/coop/rchain/casper/util/CliqueTest.scala

use casper::rust::util::clique::Clique;
use std::cmp::Ordering;
use std::collections::BTreeMap;

struct CliqueTest;

impl CliqueTest {
    fn g1() -> Vec<(i32, i32)> {
        vec![
            (1, 6),
            (1, 2),
            (1, 3),
            (2, 6),
            (2, 4),
            (2, 3),
            (3, 6),
            (4, 6),
            (4, 7),
            (4, 5),
            (5, 7),
            (8, 9),
            (10, 11),
        ]
    }

    fn g2() -> Vec<(i32, i32)> {
        vec![
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 3),
            (3, 4),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ]
    }

    fn compare(l: &[i32], r: &[i32]) -> bool {
        fn compare_helper(l: &[i32], r: &[i32]) -> bool {
            match (l.first(), r.first()) {
                (Some(&lh), Some(&rh)) if lh == rh => compare_helper(&l[1..], &r[1..]),
                (Some(&lh), Some(&rh)) if lh < rh => true,
                _ => false,
            }
        }

        compare_helper(l, r)
    }

    fn assert_cliques_equal(src: &[Vec<i32>], expected: &[Vec<i32>]) {
        let src_sorted: Vec<Vec<i32>> = src
            .iter()
            .map(|clique| {
                let mut sorted = clique.clone();
                sorted.sort();
                sorted
            })
            .collect();

        let expected_sorted: Vec<Vec<i32>> = expected
            .iter()
            .map(|clique| {
                let mut sorted = clique.clone();
                sorted.sort();
                sorted
            })
            .collect();

        let mut src_with_compare = src_sorted.clone();
        src_with_compare.sort_by(|a, b| {
            if Self::compare(a, b) {
                Ordering::Less
            } else if Self::compare(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        let mut exp_with_compare = expected_sorted.clone();
        exp_with_compare.sort_by(|a, b| {
            if Self::compare(a, b) {
                Ordering::Less
            } else if Self::compare(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        assert_eq!(src_with_compare, exp_with_compare);
    }
}

#[test]
fn find_cliques_recursive_should_work_when_there_is_no_cliques() {
    let h: Vec<(i32, i32)> = vec![];
    let c = Clique::find_cliques_recursive(&h);
    let expected: Vec<Vec<i32>> = vec![];

    CliqueTest::assert_cliques_equal(&c, &expected);
}

#[test]
fn find_cliques_recursive_should_yield_all_cliques() {
    let c = Clique::find_cliques_recursive(&CliqueTest::g1());
    let expected = vec![
        vec![2, 6, 1, 3],
        vec![2, 6, 4],
        vec![5, 4, 7],
        vec![8, 9],
        vec![10, 11],
    ];

    CliqueTest::assert_cliques_equal(&c, &expected);
}

#[test]
fn findcliques_recursive_should_work_well_in_self_loops() {
    let mut g3 = vec![(1, 1)];
    g3.extend(CliqueTest::g1().iter().cloned());

    let c = Clique::find_cliques_recursive(&g3);
    let expected = vec![
        vec![2, 6, 1, 3],
        vec![2, 6, 4],
        vec![5, 4, 7],
        vec![8, 9],
        vec![10, 11],
    ];

    CliqueTest::assert_cliques_equal(&c, &expected);
}

#[test]
fn find_cliques_recursive_should_work_well_in_another_case() {
    let c = Clique::find_cliques_recursive(&CliqueTest::g2());
    let expected = vec![vec![1, 2], vec![1, 4, 5, 6], vec![2, 3], vec![3, 4, 6]];

    CliqueTest::assert_cliques_equal(&c, &expected);
}

#[test]
fn findmaximumcliquebyweight_should_work_with_same_weights() {
    let weights = BTreeMap::from([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]);

    assert_eq!(
        Clique::find_maximum_clique_by_weight(&CliqueTest::g2(), &weights),
        4
    );
}

#[test]
fn findmaximumcliquebyweight_should_pick_max_weight_over_weight_of_max_size() {
    let weights = BTreeMap::from([(1, 10), (2, 10), (3, 1), (4, 1), (5, 1), (6, 1)]);

    assert_eq!(
        Clique::find_maximum_clique_by_weight(&CliqueTest::g2(), &weights),
        20
    );
}
