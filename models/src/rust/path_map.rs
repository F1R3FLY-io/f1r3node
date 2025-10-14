use std::collections::HashMap;
use crate::rhoapi::{Par, Var, Expr, EList};
use crate::rhoapi::expr::ExprInstance;
use crate::rust::par_to_sexpr::ParToSExpr;
use crate::rust::path_map_encoder::SExpr;

#[derive(Debug, Clone, PartialEq)]
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

/// Internal trie structure for PathMap
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathMapTrie {
    pub root: PathMapNode,
}

/// ParPathMap - wrapper for PathMap compatible with Rholang collection system
#[derive(Clone)]
pub struct ParPathMap {
    pub trie: PathMapTrie,
    pub connective_used: bool,
    pub locally_free: Vec<u8>,
    pub remainder: Option<Var>,
}

impl ParPathMap {
    pub fn new(
        trie: PathMapTrie,
        connective_used: bool,
        locally_free: Vec<u8>,
        remainder: Option<Var>,
    ) -> ParPathMap {
        ParPathMap {
            trie,
            connective_used,
            locally_free,
            remainder,
        }
    }

    pub fn create_from_elements(elements: Vec<Par>, remainder: Option<Var>) -> Self {
        let trie = PathMapTrie::from_elements(elements);
        ParPathMap {
            trie: trie.clone(),
            connective_used: Self::connective_used(&trie) || remainder.is_some(),
            locally_free: Self::update_locally_free(&trie),
            remainder,
        }
    }

    fn connective_used(trie: &PathMapTrie) -> bool {
        Self::check_connective_used_node(&trie.root)
    }

    fn check_connective_used_node(node: &PathMapNode) -> bool {
        if let Some(ref par) = node.value {
            if par.connective_used {
                return true;
            }
        }
        node.children.values().any(Self::check_connective_used_node)
    }

    fn update_locally_free(trie: &PathMapTrie) -> Vec<u8> {
        let mut result = Vec::new();
        Self::collect_locally_free(&trie.root, &mut result);
        result
    }

    fn collect_locally_free(node: &PathMapNode, result: &mut Vec<u8>) {
        if let Some(ref par) = node.value {
            *result = super::utils::union(result.clone(), par.locally_free.clone());
        }
        for child in node.children.values() {
            Self::collect_locally_free(child, result);
        }
    }

    pub fn equals(&self, other: &ParPathMap) -> bool {
        // For now, simple structural comparison
        // TODO: Implement proper trie equality
        self.remainder == other.remainder && self.connective_used == other.connective_used
    }
}

impl PathMapTrie {
    pub fn new() -> Self {
        PathMapTrie { root: PathMapNode::new() }
    }

    pub fn from_elements(elements: Vec<Par>) -> Self {
        let mut trie = PathMapTrie::new();
        for par in elements.into_iter() {
            // Check if this Par is a list - if so, create a multi-segment path
            if let Some(path_segments) = Self::extract_list_path(&par) {
                // List case: each element becomes a path segment
                trie.insert(&path_segments, par);
            } else {
                // Non-list case: convert to S-expression and use as single segment
                let sexpr_string = ParToSExpr::par_to_sexpr(&par);
                let sexpr = Self::parse_sexpr(&sexpr_string);
                let byte_path = sexpr.encode();
                trie.insert(&vec![byte_path], par);
            }
        }
        trie
    }
    
    /// Extract a list of Par objects from a Par that represents a list
    /// Returns None if the Par is not a list, or Some(path_segments) if it is
    fn extract_list_path(par: &Par) -> Option<Vec<Vec<u8>>> {
        // Check if this Par is a list expression
        if par.exprs.len() == 1 {
            if let Some(ExprInstance::EListBody(list)) = &par.exprs[0].expr_instance {
                // Convert each list element to a byte path segment
                let segments: Vec<Vec<u8>> = list.ps.iter().map(|p| {
                    let sexpr_string = ParToSExpr::par_to_sexpr(p);
                    let sexpr = Self::parse_sexpr(&sexpr_string);
                    sexpr.encode()
                }).collect();
                return Some(segments);
            }
        }
        None
    }

    /// Simple S-expression parser for converting string to SExpr
    fn parse_sexpr(s: &str) -> SExpr {
        let s = s.trim();

        // Handle symbols (non-parenthesized atoms)
        if !s.starts_with('(') {
            return SExpr::Symbol(s.to_string());
        }

        // Handle lists
        if s.starts_with('(') && s.ends_with(')') {
            let inner = &s[1..s.len()-1];
            let parts = Self::split_sexpr(inner);
            let children: Vec<SExpr> = parts.iter().map(|p| Self::parse_sexpr(p)).collect();
            return SExpr::List(children);
        }

        SExpr::Symbol(s.to_string())
    }

    /// Split S-expression string into top-level parts, respecting parentheses
    fn split_sexpr(s: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape = false;

        for ch in s.chars() {
            if escape {
                current.push(ch);
                escape = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                '(' if !in_string => {
                    depth += 1;
                    current.push(ch);
                }
                ')' if !in_string => {
                    depth -= 1;
                    current.push(ch);
                }
                ' ' | '\t' | '\n' if !in_string && depth == 0 => {
                    if !current.is_empty() {
                        parts.push(current.clone());
                        current.clear();
                    }
                }
                _ => current.push(ch),
            }
        }

        if !current.is_empty() {
            parts.push(current);
        }

        parts
    }

    pub fn single(path: Vec<Vec<u8>>, value: Par) -> Self {
        let mut trie = PathMapTrie::new();
        trie.insert(&path, value);
        trie
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
    pub fn union(&self, other: &PathMapTrie) -> PathMapTrie {
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
    pub fn intersection(&self, other: &PathMapTrie) -> PathMapTrie {
        let mut result = PathMapTrie::new();
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
    pub fn subtraction(&self, other: &PathMapTrie) -> PathMapTrie {
        let mut result = PathMapTrie::new();
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
    pub fn restriction(&self, other: &PathMapTrie) -> PathMapTrie {
        let mut result = PathMapTrie::new();
        Self::restriction_node(&self.root, &other.root, &mut result.root, false);
        result
    }

    fn restriction_node(left: &PathMapNode, right: &PathMapNode, target: &mut PathMapNode, is_prefix_match: bool) {
        // Check if we've reached a prefix marker in right (a node with a value)
        let new_prefix_match = is_prefix_match || right.value.is_some();
        
        // If we're in a prefix match and left has a value, keep it
        if new_prefix_match && left.value.is_some() {
            target.value = left.value.clone();
        }

        // Process children
        for (key, left_child) in &left.children {
            match right.children.get(key) {
                Some(right_child) => {
                    // Right has this path segment, continue recursion
                    let target_child = target.children.entry(key.clone()).or_insert_with(PathMapNode::new);
                    Self::restriction_node(left_child, right_child, target_child, new_prefix_match);
                }
                None => {
                    // Right doesn't have this path segment
                    // If we're already in a prefix match (from a parent), keep the entire subtree
                    if new_prefix_match {
                        target.children.insert(key.clone(), left_child.clone());
                    }
                    // Otherwise, this path doesn't match any prefix in right, so skip it
                }
            }
        }
    }

    /// Drop Head: Collapses n bytes from all paths, joining together the sub-tries as it proceeds.
    /// This removes the first n bytes from every path in the trie.
    pub fn drop_head(&self, n: usize) -> PathMapTrie {
        let mut result = PathMapTrie::new();
        Self::drop_head_node(&self.root, &mut result.root, n, 0);
        result
    }

    fn drop_head_node(source: &PathMapNode, target: &mut PathMapNode, n: usize, current_depth: usize) {
        // If we've dropped enough segments, merge this subtree into target
        if current_depth >= n {
            // Merge value - update it to reflect the dropped segments
            if let Some(value) = &source.value {
                if target.value.is_none() {
                    // Try to extract the remaining path from the list value
                    target.value = Self::drop_segments_from_par_value(value, n);
                }
            }

            // Merge children
            for (key, child) in &source.children {
                match target.children.get_mut(key) {
                    Some(target_child) => {
                        // Key already exists, merge recursively with same depth (we've already dropped n)
                        Self::merge_nodes(target_child, child);
                    }
                    None => {
                        // Key doesn't exist, clone subtree but update values
                        let updated_child = Self::clone_node_with_updated_values(child, n);
                        target.children.insert(key.clone(), updated_child);
                    }
                }
            }
        } else {
            // Still dropping segments, recurse into children
            for child in source.children.values() {
                Self::drop_head_node(child, target, n, current_depth + 1);
            }
        }
    }
    
    /// Clone a node and all its descendants, updating list values to drop n segments
    fn clone_node_with_updated_values(node: &PathMapNode, n: usize) -> PathMapNode {
        let updated_value = node.value.as_ref().and_then(|v| Self::drop_segments_from_par_value(v, n));
        let mut children = HashMap::new();
        for (key, child) in &node.children {
            children.insert(key.clone(), Self::clone_node_with_updated_values(child, n));
        }
        PathMapNode {
            value: updated_value,
            children,
        }
    }
    
    /// If the Par is a list, drop the first n elements and return the remaining list
    fn drop_segments_from_par_value(par: &Par, n: usize) -> Option<Par> {
        // Check if this is a list
        if par.exprs.len() == 1 {
            if let Some(ExprInstance::EListBody(list)) = &par.exprs[0].expr_instance {
                if n < list.ps.len() {
                    // Drop n elements from the list
                    let remaining_elements = list.ps[n..].to_vec();
                    let new_list = EList {
                        ps: remaining_elements,
                        locally_free: list.locally_free.clone(),
                        connective_used: list.connective_used,
                        remainder: list.remainder.clone(),
                    };
                    return Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EListBody(new_list)),
                        }],
                        ..par.clone()
                    });
                } else {
                    // Dropped all elements, return empty list
                    let new_list = EList {
                        ps: vec![],
                        locally_free: list.locally_free.clone(),
                        connective_used: list.connective_used,
                        remainder: list.remainder.clone(),
                    };
                    return Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::EListBody(new_list)),
                        }],
                        ..par.clone()
                    });
                }
            }
        }
        
        // Not a list, return unchanged
        Some(par.clone())
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

    /// Run method: currently ignores the provided map parameter and returns a clone of self.
    /// This is a placeholder for more complex semantics later.
    pub fn run(&self, _other: &Self) -> Self {
        self.clone()
    }
}

impl std::iter::FromIterator<(Vec<Vec<u8>>, Par)> for PathMapTrie {
    fn from_iter<T: IntoIterator<Item = (Vec<Vec<u8>>, Par)>>(iter: T) -> Self {
        let mut map = PathMapTrie::new();
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
}
