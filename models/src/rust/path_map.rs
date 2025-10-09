use std::collections::HashMap;
use crate::rhoapi::Par;

#[derive(Debug, Clone)]
pub struct PathMapNode {
    pub value: Option<Par>,
    pub children: HashMap<Vec<u8>, PathMapNode>,
}

impl PathMapNode {
    pub fn new() -> Self {
        PathMapNode {
            value: None,
            children: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PathMap {
    pub root: PathMapNode,
}

impl PathMap {
    pub fn new() -> Self {
        PathMap { root: PathMapNode::new() }
    }

    pub fn single(path: Vec<Vec<u8>>, value: Par) -> Self {
        let mut map = PathMap::new();
        map.insert(&path, value);
        map
    }

    pub fn insert(&mut self, path: &[Vec<u8>], value: Par) {
        let mut node = &mut self.root;
        for segment in path {
            node = node.children.entry(segment.clone()).or_insert_with(PathMapNode::new);
        }
        node.value = Some(value);
    }

    pub fn get(&self, path: &[Vec<u8>]) -> Option<&Par> {
        let mut node = &self.root;
        for segment in path {
            match node.children.get(segment) {
                Some(child) => node = child,
                None => return None,
            }
        }
        node.value.as_ref()
    }

    /// Union: Creates the union of two tries, so a path present in any operand will be present in the result.
    /// If both tries have a value at the same path, the value from self is kept.
    pub fn union(&self, other: &PathMap) -> PathMap {
        let mut result = self.clone();
        Self::union_node(&mut result.root, &other.root);
        result
    }

    fn union_node(target: &mut PathMapNode, source: &PathMapNode) {
        // If source has a value and target doesn't, copy it
        if target.value.is_none() && source.value.is_some() {
            target.value = source.value.clone();
        }

        // Merge children
        for (key, source_child) in &source.children {
            let target_child = target.children.entry(key.clone()).or_insert_with(PathMapNode::new);
            Self::union_node(target_child, source_child);
        }
    }

    /// Intersection: Intersects two tries, so a path present in all operands will be present in the result.
    pub fn intersection(&self, other: &PathMap) -> PathMap {
        let mut result = PathMap::new();
        Self::intersection_node(&self.root, &other.root, &mut result.root);
        result
    }

    fn intersection_node(left: &PathMapNode, right: &PathMapNode, target: &mut PathMapNode) {
        // Only keep value if both nodes have it
        if left.value.is_some() && right.value.is_some() {
            target.value = left.value.clone();
        }

        // Only traverse children that exist in both tries
        for (key, left_child) in &left.children {
            if let Some(right_child) = right.children.get(key) {
                let target_child = target.children.entry(key.clone()).or_insert_with(PathMapNode::new);
                Self::intersection_node(left_child, right_child, target_child);
            }
        }
    }

    /// Subtraction: Removes all paths in the rvalue trie from the lvalue trie.
    /// A path present in lvalue that is not present in rvalue will be present in the result.
    pub fn subtraction(&self, other: &PathMap) -> PathMap {
        let mut result = PathMap::new();
        Self::subtraction_node(&self.root, &other.root, &mut result.root);
        result
    }

    fn subtraction_node(left: &PathMapNode, right: &PathMapNode, target: &mut PathMapNode) {
        // Keep value only if right doesn't have a value at this path
        if left.value.is_some() && right.value.is_none() {
            target.value = left.value.clone();
        }

        // Process children
        for (key, left_child) in &left.children {
            match right.children.get(key) {
                Some(right_child) => {
                    // Both have this child, recurse
                    let target_child = target.children.entry(key.clone()).or_insert_with(PathMapNode::new);
                    Self::subtraction_node(left_child, right_child, target_child);
                }
                None => {
                    // Right doesn't have this child, keep entire subtree
                    target.children.insert(key.clone(), left_child.clone());
                }
            }
        }
    }

    /// Restriction: Removes paths from lvalue that do not have a corresponding prefix in rvalue.
    /// This is like a prefix filter - only paths that start with a prefix from rvalue are kept.
    pub fn restriction(&self, other: &PathMap) -> PathMap {
        let mut result = PathMap::new();
        Self::restriction_node(&self.root, &other.root, &mut result.root, true);
        result
    }

    fn restriction_node(left: &PathMapNode, right: &PathMapNode, target: &mut PathMapNode, is_prefix_match: bool) {
        // If we're in a prefix match (right has a value or we're following a valid path in right)
        // and left has a value, keep it
        if is_prefix_match && left.value.is_some() {
            target.value = left.value.clone();
        }

        // Process children
        for (key, left_child) in &left.children {
            match right.children.get(key) {
                Some(right_child) => {
                    // Right has this path segment, continue with prefix match active
                    let target_child = target.children.entry(key.clone()).or_insert_with(PathMapNode::new);
                    let new_prefix_match = is_prefix_match || right_child.value.is_some();
                    Self::restriction_node(left_child, right_child, target_child, new_prefix_match);
                }
                None => {
                    // Right doesn't have this path segment
                    // If we're already in a prefix match (from a parent), keep the subtree
                    if is_prefix_match {
                        target.children.insert(key.clone(), left_child.clone());
                    }
                    // Otherwise, this path doesn't match any prefix in right, so skip it
                }
            }
        }
    }

    /// Drop Head: Collapses n bytes from all paths, joining together the sub-tries as it proceeds.
    /// This removes the first n bytes from every path in the trie.
    pub fn drop_head(&self, n: usize) -> PathMap {
        let mut result = PathMap::new();
        Self::drop_head_node(&self.root, &mut result.root, n, 0);
        result
    }

    fn drop_head_node(source: &PathMapNode, target: &mut PathMapNode, n: usize, current_depth: usize) {
        // If we've dropped enough bytes, merge this subtree into target
        if current_depth >= n {
            // Merge value
            if source.value.is_some() && target.value.is_none() {
                target.value = source.value.clone();
            }

            // Merge children
            for (key, child) in &source.children {
                match target.children.get_mut(key) {
                    Some(target_child) => {
                        // Key already exists, merge recursively
                        Self::merge_nodes(target_child, child);
                    }
                    None => {
                        // Key doesn't exist, just clone the subtree
                        target.children.insert(key.clone(), child.clone());
                    }
                }
            }
        } else {
            // Still dropping bytes, recurse into children
            for child in source.children.values() {
                Self::drop_head_node(child, target, n, current_depth + 1);
            }
        }
    }

    fn merge_nodes(target: &mut PathMapNode, source: &PathMapNode) {
        // Merge value (keep target's value if it exists)
        if target.value.is_none() && source.value.is_some() {
            target.value = source.value.clone();
        }

        // Merge children recursively
        for (key, source_child) in &source.children {
            match target.children.get_mut(key) {
                Some(target_child) => {
                    Self::merge_nodes(target_child, source_child);
                }
                None => {
                    target.children.insert(key.clone(), source_child.clone());
                }
            }
        }
    }

    /// Helper method to get all paths in the trie (for debugging/testing)
    pub fn all_paths(&self) -> Vec<Vec<Vec<u8>>> {
        let mut paths = Vec::new();
        let mut current_path = Vec::new();
        Self::collect_paths(&self.root, &mut current_path, &mut paths);
        paths
    }

    fn collect_paths(node: &PathMapNode, current_path: &mut Vec<Vec<u8>>, paths: &mut Vec<Vec<Vec<u8>>>) {
        if node.value.is_some() {
            paths.push(current_path.clone());
        }

        for (key, child) in &node.children {
            current_path.push(key.clone());
            Self::collect_paths(child, current_path, paths);
            current_path.pop();
        }
    }
}

impl<K, V> std::iter::FromIterator<(Vec<Vec<u8>>, Par)> for PathMap {
    fn from_iter<T: IntoIterator<Item = (Vec<Vec<u8>>, Par)>>(iter: T) -> Self {
        let mut map = PathMap::new();
        for (path, value) in iter {
            map.insert(&path, value);
        }
        map
    }
}

impl Default for PathMapNode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhoapi::Par;

    fn make_par(value: i64) -> Par {
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
        let map = PathMap::new();
        assert!(map.all_paths().is_empty());
    }

    #[test]
    fn test_single_entry() {
        let p = path(vec!["books", "fiction", "don_quixote"]);
        let par = make_par(1);
        let map = PathMap::single(p.clone(), par.clone());
        
        assert_eq!(map.all_paths().len(), 1);
        assert_eq!(map.get(&p), Some(&par));
    }

    #[test]
    fn test_insert_and_get() {
        let mut map = PathMap::new();
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
        let map = PathMap::new();
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

        let map: PathMap = paths_and_pars.clone().into_iter().collect();
        
        assert_eq!(map.all_paths().len(), 3);
        for (p, par) in paths_and_pars {
            assert_eq!(map.get(&p), Some(&par));
        }
    }

    #[test]
    fn test_union_disjoint() {
        let mut map1 = PathMap::new();
        map1.insert(&path(vec!["books", "don_quixote"]), make_par(1));
        map1.insert(&path(vec!["movies", "casablanca"]), make_par(2));

        let mut map2 = PathMap::new();
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
        let mut map1 = PathMap::new();
        map1.insert(&path(vec!["books", "gatsby"]), make_par(1));

        let mut map2 = PathMap::new();
        map2.insert(&path(vec!["books", "gatsby"]), make_par(2));
        map2.insert(&path(vec!["books", "moby"]), make_par(3));

        let result = map1.union(&map2);
        
        // map1's value should be kept for overlapping path
        assert_eq!(result.get(&path(vec!["books", "gatsby"])), Some(&make_par(1)));
        assert_eq!(result.get(&path(vec!["books", "moby"])), Some(&make_par(3)));
    }

    #[test]
    fn test_intersection() {
        let mut map1 = PathMap::new();
        map1.insert(&path(vec!["books", "gatsby"]), make_par(1));
        map1.insert(&path(vec!["books", "moby_dick"]), make_par(2));
        map1.insert(&path(vec!["movies", "casablanca"]), make_par(3));

        let mut map2 = PathMap::new();
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
        let mut map1 = PathMap::new();
        map1.insert(&path(vec!["a"]), make_par(1));

        let mut map2 = PathMap::new();
        map2.insert(&path(vec!["b"]), make_par(2));

        let result = map1.intersection(&map2);
        assert!(result.all_paths().is_empty());
    }

    #[test]
    fn test_subtraction() {
        let mut map1 = PathMap::new();
        map1.insert(&path(vec!["books", "don_quixote"]), make_par(1));
        map1.insert(&path(vec!["books", "gatsby"]), make_par(2));
        map1.insert(&path(vec!["books", "moby_dick"]), make_par(3));
        map1.insert(&path(vec!["movies", "casablanca"]), make_par(4));
        map1.insert(&path(vec!["music", "take_the_a_train"]), make_par(5));

        let mut map2 = PathMap::new();
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
        let mut map1 = PathMap::new();
        map1.insert(&path(vec!["books", "fiction", "don_quixote"]), make_par(1));
        map1.insert(&path(vec!["books", "fiction", "gatsby"]), make_par(2));
        map1.insert(&path(vec!["books", "fiction", "moby_dick"]), make_par(3));
        map1.insert(&path(vec!["books", "non-fiction", "brief_history"]), make_par(4));
        map1.insert(&path(vec!["movies", "classic", "casablanca"]), make_par(5));
        map1.insert(&path(vec!["movies", "sci-fi", "star_wars"]), make_par(6));
        map1.insert(&path(vec!["music", "take_the_a_train"]), make_par(7));

        let mut map2 = PathMap::new();
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
        let mut map = PathMap::new();
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
        let mut map = PathMap::new();
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
        let mut map = PathMap::new();
        map.insert(&path(vec!["a", "b"]), make_par(1));

        let result = map.drop_head(0);
        
        // Dropping 0 segments should return identical map
        assert_eq!(result.all_paths().len(), 1);
        assert_eq!(result.get(&path(vec!["a", "b"])), Some(&make_par(1)));
    }

    #[test]
    fn test_drop_head_all() {
        let mut map = PathMap::new();
        map.insert(&path(vec!["a", "b"]), make_par(1));

        let result = map.drop_head(2);
        
        // Dropping all segments should result in single root value
        assert_eq!(result.all_paths().len(), 1);
        assert_eq!(result.get(&path(vec![])), Some(&make_par(1)));
    }

    #[test]
    fn test_complex_scenario() {
        // Create a complex PathMap with multiple levels
        let mut map = PathMap::new();
        map.insert(&path(vec!["org", "dept1", "team1", "alice"]), make_par(1));
        map.insert(&path(vec!["org", "dept1", "team2", "bob"]), make_par(2));
        map.insert(&path(vec!["org", "dept2", "team1", "charlie"]), make_par(3));

        // Test various operations
        let mut map2 = PathMap::new();
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
        let empty = PathMap::new();
        let mut non_empty = PathMap::new();
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
}
