// See models/src/main/scala/coop/rchain/models/SortedParMap.scala

use std::collections::HashMap;

use crate::rhoapi::Par;

use super::rholang::sorter::{
    ordering::Ordering, par_sort_matcher::ParSortMatcher, sortable::Sortable,
};

#[derive(Clone, Debug)]
pub struct SortedParMap {
    pub ps: HashMap<Par, Par>,
    // TODO: Merge `sortedList` and `sortedMap` into one VectorMap once available - OLD
    pub sorted_list: Vec<(Par, Par)>,
    sorted_map: HashMap<Par, Par>,
}

impl SortedParMap {
    pub fn create_from_map(map: HashMap<Par, Par>) -> Self {
        let sorted_list = Ordering::sort_map(&map);
        let sorted_map = sorted_list.clone().into_iter().collect();

        SortedParMap {
            ps: map,
            sorted_list,
            sorted_map,
        }
    }

    pub fn create_from_vec(vec: Vec<(Par, Par)>) -> Self {
        let map: HashMap<Par, Par> = vec.into_iter().collect();
        SortedParMap::create_from_map(map)
    }

    pub fn create_from_empty() -> Self {
        SortedParMap::create_from_map(HashMap::new())
    }

    pub fn map_iter<'a, F, T>(&'a self, f: F) -> impl Iterator<Item = T> + 'a
    where
        F: Fn((&Par, &Par)) -> T + 'a,
    {
        self.sorted_list.iter().map(move |(k, v)| f((k, v)))
    }

    // alias for '+'
    pub fn insert(&mut self, kv: (Par, Par)) -> SortedParMap {
        self.sorted_map.insert(kv.0, kv.1);
        Self::create_from_map(self.sorted_map.clone())
    }

    // alias for '++'
    pub fn extend(&mut self, kvs: Vec<(Par, Par)>) -> SortedParMap {
        for kv in kvs {
            self.insert(kv);
        }
        Self::create_from_map(self.sorted_map.clone())
    }

    // alias for '-'
    pub fn remove(&mut self, key: Par) -> SortedParMap {
        self.sorted_map.remove(&Self::sort(&key));
        Self::create_from_map(self.sorted_map.clone())
    }

    // alias for '--'
    pub fn remove_multiple(&mut self, keys: Vec<Par>) -> SortedParMap {
        for key in keys {
            self.sorted_map.remove(&Self::sort(&key));
        }
        Self::create_from_map(self.sorted_map.clone())
    }

    pub fn apply(&self, key: Par) -> Option<Par> {
        self.sorted_map.get(&Self::sort(&key)).cloned()
    }

    pub fn contains(&self, par: Par) -> bool {
        self.sorted_map.contains_key(&SortedParMap::sort(&par))
    }

    pub fn get(&self, key: Par) -> Option<Par> {
        self.sorted_map.get(&SortedParMap::sort(&key)).cloned()
    }

    pub fn get_or_else(&self, key: Par, default: Par) -> Par {
        // println!("\nsorted_map in get_or_else: {:?}", self.sorted_map);
        // println!("\nkey in get_or_else: {:?}", key);
        match self.sorted_map.get(&SortedParMap::sort(&key)) {
            Some(value) => value.clone(),
            None => default,
        }
    }

    pub fn keys(&self) -> Vec<Par> {
        self.sorted_list
            .clone()
            .into_iter()
            .map(|kv| kv.0)
            .collect()
    }

    pub fn values(&self) -> Vec<Par> {
        self.sorted_list
            .clone()
            .into_iter()
            .map(|kv| kv.1)
            .collect()
    }

    pub fn equals(&self, that: SortedParMap) -> bool {
        self.sorted_list == that.sorted_list
    }

    pub fn length(&self) -> usize {
        self.sorted_list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ps.is_empty() && self.sorted_list.is_empty() && self.sorted_map.is_empty()
    }

    fn sort(par: &Par) -> Par {
        ParSortMatcher::sort_match(&par).term.clone()
    }
}

impl IntoIterator for SortedParMap {
    type Item = (Par, Par);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.sorted_list.into_iter()
    }
}

impl PartialEq for SortedParMap {
    fn eq(&self, other: &Self) -> bool {
        self.sorted_list == other.sorted_list
    }
}
