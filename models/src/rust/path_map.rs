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
