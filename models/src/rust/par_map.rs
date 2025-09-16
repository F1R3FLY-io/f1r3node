// See models/src/main/scala/coop/rchain/models/ParMap.scala

use crate::rhoapi::{Par, Var};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{sorted_par_map::SortedParMap, utils::union};

#[derive(Clone, Debug)]
pub struct ParMap {
    pub ps: SortedParMap,
    pub connective_used: bool,
    pub locally_free: Vec<u8>,
    pub remainder: Option<Var>,
}

impl ParMap {
    pub fn new(
        vec: Vec<(Par, Par)>,
        connective_used: bool,
        locally_free: Vec<u8>,
        remainder: Option<Var>,
    ) -> ParMap {
        ParMap {
            ps: SortedParMap::create_from_vec(vec),
            connective_used,
            locally_free,
            remainder,
        }
    }

    pub fn create_from_vec(vec: Vec<(Par, Par)>) -> Self {
        ParMap::new(
            vec.clone(),
            ParMap::connective_used(&vec),
            ParMap::update_locally_free(&vec),
            None,
        )
    }

    pub fn create_from_sorted_par_map(map: SortedParMap) -> Self {
        ParMap::create_from_vec(map.sorted_list)
    }

    pub fn equals(&self, other: ParMap) -> bool {
        self.ps.equals(other.ps)
            && self.remainder == other.remainder
            && self.connective_used == other.connective_used
    }

    fn connective_used(map: &Vec<(Par, Par)>) -> bool {
        map.iter()
            .any(|(k, v)| k.connective_used || v.connective_used)
    }

    fn update_locally_free(ps: &Vec<(Par, Par)>) -> Vec<u8> {
        ps.into_iter().fold(Vec::new(), |acc, (key, value)| {
            union(
                acc,
                union(key.locally_free.clone(), value.locally_free.clone()),
            )
        })
    }
}

// Serde implementation to match Scala JsonEncoder behavior
impl Serialize for ParMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as array of (Par, Par) tuples (like Scala's encodeParMap)
        use serde::ser::SerializeSeq;
        let seq = &self.ps.sorted_list;
        let mut s = serializer.serialize_seq(Some(seq.len()))?;
        for (key, value) in seq {
            s.serialize_element(&(key, value))?;
        }
        s.end()
    }
}

impl<'de> Deserialize<'de> for ParMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize from array of (Par, Par) tuples (like Scala's decodeParMap)
        let vec: Vec<(Par, Par)> = Vec::deserialize(deserializer)?;
        Ok(ParMap::create_from_vec(vec))
    }
}
