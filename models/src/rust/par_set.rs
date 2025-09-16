// See models/src/main/scala/coop/rchain/models/ParSet.scala

use crate::rhoapi::{Par, Var};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{sorted_par_hash_set::SortedParHashSet, utils::union};

#[derive(Clone)]
pub struct ParSet {
    pub ps: SortedParHashSet,
    pub connective_used: bool,
    pub locally_free: Vec<u8>,
    pub remainder: Option<Var>,
}

impl ParSet {
    pub fn new(
        vec: Vec<Par>,
        connective_used: bool,
        locally_free: Vec<u8>,
        remainder: Option<Var>,
    ) -> ParSet {
        ParSet {
            ps: SortedParHashSet::create_from_vec(vec),
            connective_used,
            locally_free,
            remainder,
        }
    }

    pub fn create_from_vec_and_remainder(vec: Vec<Par>, remainder: Option<Var>) -> Self {
        let shs = SortedParHashSet::create_from_vec(vec.clone());
        ParSet {
            ps: shs.clone(),
            connective_used: ParSet::connective_used(&vec) || remainder.is_some(),
            locally_free: ParSet::update_locally_free(&shs),
            remainder,
        }
    }

    pub fn create_from_vec(vec: Vec<Par>) -> Self {
        ParSet::create_from_vec_and_remainder(vec.clone(), None)
    }

    pub fn equals(&self, other: ParSet) -> bool {
        self.ps.equals(other.ps)
            && self.remainder == other.remainder
            && self.connective_used == other.connective_used
    }

    fn connective_used(vec: &Vec<Par>) -> bool {
        vec.iter().any(|p| p.connective_used)
    }

    fn update_locally_free(ps: &SortedParHashSet) -> Vec<u8> {
        ps.sorted_pars
            .clone()
            .into_iter()
            .fold(Vec::new(), |acc, p| union(acc, p.locally_free))
    }
}

// Serde implementation to match Scala JsonEncoder behavior
impl Serialize for ParSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as array of Par (like Scala's encodeParSet)
        use serde::ser::SerializeSeq;
        let seq = &self.ps.sorted_pars;
        let mut s = serializer.serialize_seq(Some(seq.len()))?;
        for par in seq {
            s.serialize_element(par)?;
        }
        s.end()
    }
}

impl<'de> Deserialize<'de> for ParSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize from array of Par (like Scala's decodeParSet)
        let vec: Vec<Par> = Vec::deserialize(deserializer)?;
        Ok(ParSet::create_from_vec(vec))
    }
}
