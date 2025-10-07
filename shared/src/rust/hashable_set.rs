use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;

#[derive(Debug, Clone)]
pub struct HashableSet<T>(pub HashSet<T>);

impl<T: Eq + Hash> HashableSet<T> {
    pub fn new() -> Self {
        Self(HashSet::new())
    }
}

impl<T: Eq + Hash> PartialEq for HashableSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq + Hash> Eq for HashableSet<T> {}

// Implement PartialOrd for HashableSet with T that implements Ord
impl<T: Eq + Hash + Ord> PartialOrd for HashableSet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Implement Ord for HashableSet with T that implements Ord
impl<T: Eq + Hash + Ord> Ord for HashableSet<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort both sets for consistent comparison
        let mut self_vec: Vec<&T> = self.0.iter().collect();
        let mut other_vec: Vec<&T> = other.0.iter().collect();

        self_vec.sort();
        other_vec.sort();

        // First compare by length
        let len_cmp = self.0.len().cmp(&other.0.len());
        if len_cmp != Ordering::Equal {
            return len_cmp;
        }

        // Then lexicographically
        for (a, b) in self_vec.iter().zip(other_vec.iter()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                non_eq => return non_eq,
            }
        }

        Ordering::Equal
    }
}

impl<T: Eq + Hash> Hash for HashableSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Collect the element hashes and sort for order-independent hashing
        let mut element_hashes: Vec<u64> = self
            .0
            .iter()
            .map(|item| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                item.hash(&mut hasher);
                hasher.finish()
            })
            .collect();

        element_hashes.sort_unstable(); // Ensure same order regardless of insertion
        for h in element_hashes {
            h.hash(state);
        }
    }
}

// Implement IntoIterator for HashableSet
impl<T: Eq + Hash> IntoIterator for HashableSet<T> {
    type Item = T;
    type IntoIter = std::collections::hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// Implement IntoIterator for &HashableSet
impl<'a, T: Eq + Hash> IntoIterator for &'a HashableSet<T> {
    type Item = &'a T;
    type IntoIter = std::collections::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// Implement FromIterator for HashableSet
impl<T: Eq + Hash> FromIterator<T> for HashableSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        HashableSet(HashSet::from_iter(iter))
    }
}
