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
