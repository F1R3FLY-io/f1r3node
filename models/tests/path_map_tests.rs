use models::rhoapi::Par;
use models::rust::path_map::PathMapTrie;

fn make_par(_value: i64) -> Par {
    Par {
        sends: vec![],
        receives: vec![],
        news: vec![],
        exprs: vec![],
        matches: vec![],
        unforgeables: vec![],
        bundles: vec![],
        connectives: vec![],
        locally_free: vec![],
        connective_used: false,
    }
}

fn path(segments: Vec<&str>) -> Vec<Vec<u8>> {
    segments.into_iter().map(|s| s.as_bytes().to_vec()).collect()
}

#[test]
fn test_new_empty_pathmap() {
    let map = PathMapTrie::new();
    assert!(map.all_paths().is_empty());
}

#[test]
fn test_single_entry() {
    let p = path(vec!["books", "fiction", "don_quixote"]);
    let par = make_par(1);
    let map = PathMapTrie::single(p.clone(), par.clone());
    
    assert_eq!(map.all_paths().len(), 1);
    assert_eq!(map.get(&p), Some(&par));
}

#[test]
fn test_insert_and_get() {
    let mut map = PathMapTrie::new();
    let p1 = path(vec!["books", "fiction"]);
    let p2 = path(vec!["books", "non-fiction"]);
    let par1 = make_par(1);
    let par2 = make_par(2);

    map.insert(&p1, par1.clone());
    map.insert(&p2, par2.clone());

    assert_eq!(map.get(&p1), Some(&par1));
    assert_eq!(map.get(&p2), Some(&par2));
    assert_eq!(map.all_paths().len(), 2);
}

#[test]
fn test_get_nonexistent_path() {
    let map = PathMapTrie::new();
    let p = path(vec!["nonexistent"]);
    assert_eq!(map.get(&p), None);
}

#[test]
fn test_from_iter() {
    let paths_and_pars = vec![
        (path(vec!["a", "b"]), make_par(1)),
        (path(vec!["c", "d"]), make_par(2)),
        (path(vec!["e"]), make_par(3)),
    ];

    let map: PathMapTrie = paths_and_pars.clone().into_iter().collect();
    
    assert_eq!(map.all_paths().len(), 3);
    for (p, par) in paths_and_pars {
        assert_eq!(map.get(&p), Some(&par));
    }
}

#[test]
fn test_union_disjoint() {
    let mut map1 = PathMapTrie::new();
    map1.insert(&path(vec!["books", "don_quixote"]), make_par(1));
    map1.insert(&path(vec!["movies", "casablanca"]), make_par(2));

    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["books", "moby_dick"]), make_par(3));
    map2.insert(&path(vec!["music", "take_the_a_train"]), make_par(4));

    let result = map1.union(&map2);
    
    assert_eq!(result.all_paths().len(), 4);
    assert_eq!(result.get(&path(vec!["books", "don_quixote"])), Some(&make_par(1)));
    assert_eq!(result.get(&path(vec!["movies", "casablanca"])), Some(&make_par(2)));
    assert_eq!(result.get(&path(vec!["books", "moby_dick"])), Some(&make_par(3)));
    assert_eq!(result.get(&path(vec!["music", "take_the_a_train"])), Some(&make_par(4)));
}

#[test]
fn test_union_overlapping() {
    let mut map1 = PathMapTrie::new();
    map1.insert(&path(vec!["books", "gatsby"]), make_par(1));

    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["books", "gatsby"]), make_par(2));
    map2.insert(&path(vec!["books", "moby"]), make_par(3));

    let result = map1.union(&map2);
    
    // map1's value should be kept for overlapping path
    assert_eq!(result.get(&path(vec!["books", "gatsby"])), Some(&make_par(1)));
    assert_eq!(result.get(&path(vec!["books", "moby"])), Some(&make_par(3)));
}

#[test]
fn test_intersection() {
    let mut map1 = PathMapTrie::new();
    map1.insert(&path(vec!["books", "gatsby"]), make_par(1));
    map1.insert(&path(vec!["books", "moby_dick"]), make_par(2));
    map1.insert(&path(vec!["movies", "casablanca"]), make_par(3));

    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["books", "gatsby"]), make_par(4));
    map2.insert(&path(vec!["movies", "casablanca"]), make_par(5));
    map2.insert(&path(vec!["movies", "star_wars"]), make_par(6));

    let result = map1.intersection(&map2);
    
    assert_eq!(result.all_paths().len(), 2);
    assert_eq!(result.get(&path(vec!["books", "gatsby"])), Some(&make_par(1)));
    assert_eq!(result.get(&path(vec!["movies", "casablanca"])), Some(&make_par(3)));
    assert_eq!(result.get(&path(vec!["books", "moby_dick"])), None);
    assert_eq!(result.get(&path(vec!["movies", "star_wars"])), None);
}

#[test]
fn test_intersection_empty() {
    let mut map1 = PathMapTrie::new();
    map1.insert(&path(vec!["a"]), make_par(1));

    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["b"]), make_par(2));

    let result = map1.intersection(&map2);
    assert!(result.all_paths().is_empty());
}

#[test]
fn test_subtraction() {
    let mut map1 = PathMapTrie::new();
    map1.insert(&path(vec!["books", "don_quixote"]), make_par(1));
    map1.insert(&path(vec!["books", "gatsby"]), make_par(2));
    map1.insert(&path(vec!["books", "moby_dick"]), make_par(3));
    map1.insert(&path(vec!["movies", "casablanca"]), make_par(4));
    map1.insert(&path(vec!["music", "take_the_a_train"]), make_par(5));

    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["books", "don_quixote"]), make_par(10));
    map2.insert(&path(vec!["books", "moby_dick"]), make_par(11));
    map2.insert(&path(vec!["movies", "star_wars"]), make_par(12));

    let result = map1.subtraction(&map2);
    
    assert_eq!(result.all_paths().len(), 3);
    assert_eq!(result.get(&path(vec!["books", "gatsby"])), Some(&make_par(2)));
    assert_eq!(result.get(&path(vec!["movies", "casablanca"])), Some(&make_par(4)));
    assert_eq!(result.get(&path(vec!["music", "take_the_a_train"])), Some(&make_par(5)));
    assert_eq!(result.get(&path(vec!["books", "don_quixote"])), None);
    assert_eq!(result.get(&path(vec!["books", "moby_dick"])), None);
}

#[test]
fn test_restriction() {
    let mut map1 = PathMapTrie::new();
    map1.insert(&path(vec!["books", "fiction", "don_quixote"]), make_par(1));
    map1.insert(&path(vec!["books", "fiction", "gatsby"]), make_par(2));
    map1.insert(&path(vec!["books", "fiction", "moby_dick"]), make_par(3));
    map1.insert(&path(vec!["books", "non-fiction", "brief_history"]), make_par(4));
    map1.insert(&path(vec!["movies", "classic", "casablanca"]), make_par(5));
    map1.insert(&path(vec!["movies", "sci-fi", "star_wars"]), make_par(6));
    map1.insert(&path(vec!["music", "take_the_a_train"]), make_par(7));

    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["books", "fiction"]), make_par(100));
    map2.insert(&path(vec!["movies", "sci-fi"]), make_par(101));

    let result = map1.restriction(&map2);
    
    // Should keep only paths that have books:fiction or movies:sci-fi as prefix
    assert_eq!(result.all_paths().len(), 4);
    assert_eq!(result.get(&path(vec!["books", "fiction", "don_quixote"])), Some(&make_par(1)));
    assert_eq!(result.get(&path(vec!["books", "fiction", "gatsby"])), Some(&make_par(2)));
    assert_eq!(result.get(&path(vec!["books", "fiction", "moby_dick"])), Some(&make_par(3)));
    assert_eq!(result.get(&path(vec!["movies", "sci-fi", "star_wars"])), Some(&make_par(6)));
    assert_eq!(result.get(&path(vec!["books", "non-fiction", "brief_history"])), None);
    assert_eq!(result.get(&path(vec!["movies", "classic", "casablanca"])), None);
    assert_eq!(result.get(&path(vec!["music", "take_the_a_train"])), None);
}

#[test]
fn test_drop_head() {
    let mut map = PathMapTrie::new();
    map.insert(&path(vec!["books", "don_quixote"]), make_par(1));
    map.insert(&path(vec!["books", "gatsby"]), make_par(2));
    map.insert(&path(vec!["books", "moby_dick"]), make_par(3));

    let result = map.drop_head(1);
    
    // After dropping "books", should have 3 paths: don_quixote, gatsby, moby_dick
    assert_eq!(result.all_paths().len(), 3);
    assert_eq!(result.get(&path(vec!["don_quixote"])), Some(&make_par(1)));
    assert_eq!(result.get(&path(vec!["gatsby"])), Some(&make_par(2)));
    assert_eq!(result.get(&path(vec!["moby_dick"])), Some(&make_par(3)));
}

#[test]
fn test_drop_head_merging() {
    let mut map = PathMapTrie::new();
    map.insert(&path(vec!["a", "b", "c"]), make_par(1));
    map.insert(&path(vec!["x", "b", "d"]), make_par(2));

    let result = map.drop_head(1);
    
    // After dropping first segment, both paths start with "b"
    assert_eq!(result.all_paths().len(), 2);
    assert_eq!(result.get(&path(vec!["b", "c"])), Some(&make_par(1)));
    assert_eq!(result.get(&path(vec!["b", "d"])), Some(&make_par(2)));
}

#[test]
fn test_drop_head_zero() {
    let mut map = PathMapTrie::new();
    map.insert(&path(vec!["a", "b"]), make_par(1));

    let result = map.drop_head(0);
    
    // Dropping 0 segments should return identical map
    assert_eq!(result.all_paths().len(), 1);
    assert_eq!(result.get(&path(vec!["a", "b"])), Some(&make_par(1)));
}

#[test]
fn test_drop_head_all() {
    let mut map = PathMapTrie::new();
    map.insert(&path(vec!["a", "b"]), make_par(1));

    let result = map.drop_head(2);
    
    // Dropping all segments should result in single root value
    assert_eq!(result.all_paths().len(), 1);
    assert_eq!(result.get(&path(vec![])), Some(&make_par(1)));
}

#[test]
fn test_complex_scenario() {
    // Create a complex PathMapTrie with multiple levels
    let mut map = PathMapTrie::new();
    map.insert(&path(vec!["org", "dept1", "team1", "alice"]), make_par(1));
    map.insert(&path(vec!["org", "dept1", "team2", "bob"]), make_par(2));
    map.insert(&path(vec!["org", "dept2", "team1", "charlie"]), make_par(3));

    // Test various operations
    let mut map2 = PathMapTrie::new();
    map2.insert(&path(vec!["org", "dept1", "team1", "alice"]), make_par(10));
    map2.insert(&path(vec!["org", "dept3", "team1", "dave"]), make_par(4));

    // Union should have all paths
    let union_result = map.union(&map2);
    assert_eq!(union_result.all_paths().len(), 4);

    // Intersection should have only alice
    let intersection_result = map.intersection(&map2);
    assert_eq!(intersection_result.all_paths().len(), 1);

    // Subtraction should remove alice
    let subtraction_result = map.subtraction(&map2);
    assert_eq!(subtraction_result.all_paths().len(), 2);

    // Drop head should remove "org"
    let drop_result = map.drop_head(1);
    assert_eq!(drop_result.all_paths().len(), 3);
    assert_eq!(drop_result.get(&path(vec!["dept1", "team1", "alice"])), Some(&make_par(1)));
}

#[test]
fn test_empty_operations() {
    let empty = PathMapTrie::new();
    let mut non_empty = PathMapTrie::new();
    non_empty.insert(&path(vec!["a"]), make_par(1));

    // Union with empty
    let result = empty.union(&non_empty);
    assert_eq!(result.all_paths().len(), 1);

    // Intersection with empty
    let result = non_empty.intersection(&empty);
    assert!(result.all_paths().is_empty());

    // Subtraction with empty
    let result = non_empty.subtraction(&empty);
    assert_eq!(result.all_paths().len(), 1);
}

#[test]
fn test_run_method() {
    let map1 = PathMapTrie::new();
    let map2 = PathMapTrie::new();
    let result = map1.run(&map2);
    assert_eq!(result, map1);
}